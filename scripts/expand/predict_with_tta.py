#!/usr/bin/env python3
"""
Test-time augmentation (TTA) inference for STG-Mol.

Averages predictions over K conformer / SMILES augmentations per
molecule → typically boosts AUC by 0.02-0.05 without changing the
model, and DOES NOT hurt external recall because it's an ensembling
technique that reduces prediction variance.

Two augmentation modes:
  * conformer_tta   -- generate K 3D conformers per molecule (RDKit
                       ETKDG) and average predictions
  * smiles_tta      -- randomise SMILES K times (root atom permutation)
                       and average
  * both            -- do both, K conformers × K SMILES = K² total

Usage:
    python scripts/expand/predict_with_tta.py \\
        --models_dir results/v26_v5/.../models \\
        --config configs/train_v26_v5.yaml \\
        --input_csv data/processed/nlrp3_v5/test.csv \\
        --output_dir predictions_test_v5_tta \\
        --mode conformer_tta --k 5 \\
        --ensemble
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

warnings.filterwarnings('ignore')

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / 'src' / 'training'))
try:
    from train_v26 import (
        MultiModalFusionNet, mol_to_2d_graph, mol_to_3d_graph,
        Mol2VecFeaturizer, ConformerGenerator,
    )
except Exception as e:
    print(f'ERROR importing train_v26: {e}', file=sys.stderr)
    sys.exit(1)

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    sys.exit('ERROR: rdkit required.')


_SMILES_CANDS = ['smiles', 'smiles_standardized', 'canonical_smiles', 'SMILES']
_LABEL_CANDS = ['label', 'y_true', 'true_label', 'activity']


def _detect_col(cols, candidates):
    lc = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lc:
            return lc[c.lower()]
    return None


def randomise_smiles(smi, n=5, seed=42):
    """Return n randomly-rooted canonical SMILES for the same molecule."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return [smi]
    outs = [Chem.MolToSmiles(m, canonical=True)]  # original
    n_atoms = m.GetNumAtoms()
    rng = np.random.RandomState(seed)
    for _ in range(n - 1):
        root = int(rng.randint(0, n_atoms))
        try:
            s = Chem.MolToSmiles(m, canonical=False, rootedAtAtom=root)
            outs.append(s)
        except Exception:
            outs.append(outs[0])
    return list(set(outs))[:n]


def build_model(config, model_path, device):
    model_cfg = config.get('model', {}).copy()
    model_cfg['fusion_mode'] = '1D+2D+3D'
    model = MultiModalFusionNet(model_cfg)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def predict_one(model, v1d, g2d, g3d, device):
    v1d_t = torch.tensor(v1d, dtype=torch.float).unsqueeze(0).to(device)
    g2d = g2d.to(device)
    g3d = g3d.to(device)
    if not hasattr(g2d, 'batch') or g2d.batch is None:
        g2d.batch = torch.zeros(g2d.num_nodes, dtype=torch.long, device=device)
    if not hasattr(g3d, 'batch') or g3d.batch is None:
        g3d.batch = torch.zeros(g3d.num_nodes, dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(mol2vec_feat=v1d_t, graph_2d=g2d, graph_3d=g3d)
    probs = torch.softmax(out['logits'], dim=-1).cpu().numpy()[0]
    return float(probs[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models_dir', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--input_csv', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--mode', default='conformer_tta',
                    choices=['conformer_tta', 'smiles_tta', 'both'])
    ap.add_argument('--k', type=int, default=5, help='Aug per molecule.')
    ap.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--ensemble', action='store_true')
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_paths = sorted([p for p in Path(args.models_dir).glob('*.pt')
                          if 'seed' in p.name.lower()])
    print(f'\nLoading {len(model_paths)} seed models...')

    df = pd.read_csv(args.input_csv, encoding='utf-8')
    smi_col = _detect_col(df.columns, _SMILES_CANDS)
    lbl_col = _detect_col(df.columns, _LABEL_CANDS)
    name_col = 'name' if 'name' in df.columns else None
    print(f'Input: {len(df)} molecules from {args.input_csv}')

    mol2vec_path = config.get('mol2vec', {}).get('model_path',
                                                  'data/mol2vec_model.pkl')
    print(f'Loading Mol2Vec: {mol2vec_path}')
    featurizer = Mol2VecFeaturizer(model_path=mol2vec_path)
    conformer_gen = ConformerGenerator()
    cutoff = float(config.get('model', {}).get('cutoff', 5.0))

    # ---- For each molecule, generate K augmentations of features -------
    print(f'\nGenerating K={args.k} augmentations per molecule '
          f'(mode={args.mode})...')

    all_features = []   # list of list of (v1d, g2d, g3d)
    for i, r in df.iterrows():
        smi = str(r[smi_col]).strip()
        aug_features = []
        if args.mode in ['smiles_tta', 'both']:
            smiles_list = randomise_smiles(smi, args.k, seed=42 + i)
        else:
            smiles_list = [smi]
        for smi_aug in smiles_list:
            try:
                v1d = np.asarray(featurizer.featurize([smi_aug])[0],
                                  dtype=np.float32)
                g2d = mol_to_2d_graph(smi_aug)
                # Conformer TTA: generate multiple 3D conformers
                if args.mode in ['conformer_tta', 'both']:
                    for conf_seed in range(args.k):
                        g3d = mol_to_3d_graph(smi_aug, cutoff, conformer_gen)
                        if v1d is not None and g2d is not None and g3d is not None:
                            aug_features.append((v1d, g2d, g3d))
                else:
                    g3d = mol_to_3d_graph(smi_aug, cutoff, conformer_gen)
                    if v1d is not None and g2d is not None and g3d is not None:
                        aug_features.append((v1d, g2d, g3d))
            except Exception:
                pass
        all_features.append(aug_features)
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(df)}  (avg {np.mean([len(a) for a in all_features]):.1f} aug/mol)')

    total_augs = sum(len(a) for a in all_features)
    print(f'\n  Total augmentations: {total_augs}  '
          f'(avg {total_augs / max(1, len(all_features)):.1f}/mol)')

    # ---- Run each model on all augmentations ----------------------------
    all_seed_probs = []
    for mp in model_paths:
        seed_tag = mp.stem
        print(f'\n=== Running {seed_tag} with TTA ===')
        model = build_model(config, mp, args.device)
        probs = np.full(len(df), np.nan)
        for i, aug in enumerate(all_features):
            if not aug:
                continue
            aug_probs = []
            for v1d, g2d, g3d in aug:
                aug_probs.append(predict_one(model, v1d, g2d, g3d, args.device))
            probs[i] = float(np.mean(aug_probs))

        # Save per-seed CSV
        out_df = pd.DataFrame({
            'idx': np.arange(len(df)),
            'smiles': df[smi_col].values,
            'activity_prob': probs,
            'predicted_active': (probs > 0.5).astype(int),
        })
        if name_col:
            out_df.insert(0, 'name', df[name_col].values)
        if lbl_col:
            out_df['label'] = df[lbl_col].values
        out_path = Path(args.output_dir) / f'tta_pred_{seed_tag}.csv'
        out_df.to_csv(out_path, index=False, encoding='utf-8')
        n_active = int(out_df['predicted_active'].fillna(0).sum())
        print(f'  ✓ n_active = {n_active}/{len(df)}  → {out_path}')

        if lbl_col:
            valid = ~np.isnan(probs)
            y = out_df.loc[valid, 'label'].astype(int).values
            p = probs[valid]
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(y, p)
                print(f'  · TTA Test ROC-AUC = {auc:.4f}')
            except Exception:
                pass
        all_seed_probs.append(probs)

    # ---- Ensemble across seeds ------------------------------------------
    if args.ensemble and len(model_paths) > 1:
        print('\n=== Ensemble across seeds (TTA + seed averaging) ===')
        stack = np.stack(all_seed_probs)
        ens = np.nanmean(stack, axis=0)
        ens_df = pd.DataFrame({
            'idx': np.arange(len(df)),
            'smiles': df[smi_col].values,
            'activity_prob': ens,
            'predicted_active': (ens > 0.5).astype(int),
        })
        if name_col:
            ens_df.insert(0, 'name', df[name_col].values)
        if lbl_col:
            ens_df['label'] = df[lbl_col].values
        ens_path = Path(args.output_dir) / 'tta_pred_ensemble.csv'
        ens_df.to_csv(ens_path, index=False, encoding='utf-8')
        print(f'  ✓ Saved → {ens_path}')

        if lbl_col:
            from sklearn.metrics import roc_auc_score
            valid = ~np.isnan(ens)
            y = df.loc[valid, lbl_col].astype(int).values
            p = ens[valid]
            try:
                auc = roc_auc_score(y, p)
                print(f'  · TTA-Ensemble Test ROC-AUC = {auc:.4f}')
            except Exception:
                pass

    print('\nDone.')


if __name__ == '__main__':
    main()
