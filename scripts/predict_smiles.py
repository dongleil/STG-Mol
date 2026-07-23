#!/usr/bin/env python3
"""
Batch-predict activity + ADMET for arbitrary SMILES using trained STG-Mol
(Multi-Task) models. Supports both single-model and 5-seed ensemble inference.

Two primary use cases:

  (A) **Positive-control test** — predict activity for published NLRP3
      inhibitors (MCC950, CY-09, OLT1177, Oridonin, Tranilast).

  (B) **Test-set inference for EF/BEDROC metrics** — re-generate per-seed
      predictions on the held-out NLRP3 test.csv, then feed to
      compute_ef_bedroc.py.

Usage:
    # Single model
    python scripts/predict_smiles.py \\
        --model_path results/.../models/model_1D_2D_3D_seed42.pt \\
        --config configs/train_v26_multitask.yaml \\
        --input_csv data/known_inhibitors.csv \\
        --output_csv predictions_known_inhibitors.csv

    # 5-seed ensemble on the test set (auto-detects label column, saves
    # one prediction CSV per seed plus an ensemble CSV)
    python scripts/predict_smiles.py \\
        --models_dir results/.../models \\
        --config configs/train_v26_multitask.yaml \\
        --input_csv data/processed/nlrp3/test.csv \\
        --output_dir predictions_test_5seed \\
        --ensemble

Input CSV columns:
    * `smiles` or `smiles_standardized`  (required)
    * `name`   (optional, for readability)
    * `label`  (optional, retained in output for downstream metrics)

Output CSV columns:
    * name, smiles, [label], activity_prob, predicted_active,
      admet_lipinski_prob, admet_qed_prob, admet_pains_prob,
      admet_sa_prob, admet_logp_prob
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

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / 'src' / 'training'))
try:
    from train_v26 import (
        MultiModalFusionNet, mol_to_2d_graph, mol_to_3d_graph,
        Mol2VecFeaturizer, ConformerGenerator,
    )
except Exception as e:
    print(f'ERROR importing train_v26: {e}', file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------
# CSV I/O
# --------------------------------------------------------------------------
_SMILES_CANDS = ['smiles', 'smiles_standardized', 'canonical_smiles', 'SMILES']
_LABEL_CANDS = ['label', 'y_true', 'true_label', 'activity', 'target', 'class', 'y']


def _detect_col(cols, candidates):
    lc = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lc:
            return lc[c.lower()]
    return None


def _load_input(csv_path):
    df = pd.read_csv(csv_path)
    smi_col = _detect_col(df.columns, _SMILES_CANDS)
    lbl_col = _detect_col(df.columns, _LABEL_CANDS)
    name_col = 'name' if 'name' in df.columns else None
    if not smi_col:
        sys.exit(f'ERROR: no smiles column found in {csv_path}. Columns: {df.columns.tolist()}')
    return df, smi_col, lbl_col, name_col


# --------------------------------------------------------------------------
# Model + featurisation
# --------------------------------------------------------------------------

def _build_model(config, model_path, device):
    """Instantiate a model, load state_dict, return eval-mode model."""
    model_cfg = config.get('model', {}).copy()
    model_cfg['fusion_mode'] = '1D+2D+3D'
    # If ChemBERTa embeddings are configured, set the input dim to match
    cb_hidden = config.get('mol2vec', {}).get('embedding_dim', 300)
    model_cfg['mol2vec_dim'] = cb_hidden
    model = MultiModalFusionNet(model_cfg)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


class _ChemBERTaLookup:
    """Drop-in replacement for Mol2VecFeaturizer that reads pre-computed
    ChemBERTa embeddings from a pickle. Same .featurize([smi]) interface."""

    def __init__(self, pkl_path):
        import pickle
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        self.embeddings = data['embeddings']
        self.hidden_size = data['hidden_size']
        print(f'📥 Loaded ChemBERTa embeddings ({self.hidden_size}-d) '
              f'for {len(self.embeddings)} molecules from {pkl_path}')

    def featurize(self, smiles_list):
        out = np.zeros((len(smiles_list), self.hidden_size), dtype=np.float32)
        miss = 0
        for i, s in enumerate(smiles_list):
            emb = self.embeddings.get(s)
            if emb is None:
                miss += 1
            else:
                out[i] = emb
        if miss:
            print(f'   ⚠ {miss}/{len(smiles_list)} SMILES not in ChemBERTa '
                  f'pkl (zero-filled — may hurt accuracy)')
        return out


def _featurise(smi, featurizer, conformer_gen, cutoff):
    """Return (v1d, g2d, g3d) or (None, None, None) on failure.

    Mol2Vec is missing some acpype-derived SMILES fragments (esp. novel
    scaffolds); on failure we zero-fill the 1D vector rather than skip
    the whole molecule — 2D/3D branches still carry most of the signal.
    Only if 2D or 3D graph construction also fails do we fall back to None.

    Wrap every step in a try/except: some downstream helpers raise instead
    of returning None on failure (e.g. ETKDG can throw on radicals /
    hypervalent atoms), and we don't want a single problematic SMILES to
    take down the whole ensemble scoring.
    """
    # 1D — Mol2Vec (zero-fill on failure)
    try:
        v1d = np.asarray(featurizer.featurize([smi])[0], dtype=np.float32)
    except Exception as e:
        emb_dim = int(getattr(featurizer, 'hidden_size',
                              getattr(featurizer, 'embedding_dim', 300)))
        v1d = np.zeros(emb_dim, dtype=np.float32)
        print(f'   ! Mol2Vec featurise failed on {smi[:50]!r}: '
              f'{type(e).__name__} — using zero 1D embedding')

    # 2D graph
    try:
        g2d = mol_to_2d_graph(smi)
    except Exception as e:
        print(f'   ! 2D graph failed on {smi[:50]!r}: {type(e).__name__}: {e}')
        return None, None, None
    if g2d is None:
        print(f'   ! 2D graph returned None for {smi[:50]!r}')
        return None, None, None

    # 3D graph (ETKDG conformer + SphereNet-ready features)
    try:
        g3d = mol_to_3d_graph(smi, cutoff, conformer_gen)
    except Exception as e:
        print(f'   ! 3D graph failed on {smi[:50]!r}: {type(e).__name__}: {e}')
        return None, None, None
    if g3d is None:
        print(f'   ! 3D graph returned None for {smi[:50]!r}')
        return None, None, None

    return v1d, g2d, g3d


def _predict_one(model, v1d, g2d, g3d, device):
    """Run one forward pass. Returns (activity_prob, admet_5x2_prob)."""
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
    admet = None
    if 'admet_logits' in out and out['admet_logits'] is not None:
        admet = torch.softmax(out['admet_logits'], dim=-1).cpu().numpy()[0]
    return float(probs[1]), admet


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument('--model_path', type=str,
                     help='Path to a single trained .pt checkpoint.')
    grp.add_argument('--models_dir', type=str,
                     help='Directory containing multiple checkpoints (e.g. 5-seed run).')
    ap.add_argument('--config', required=True)
    ap.add_argument('--input_csv', required=True)
    ap.add_argument('--output_csv', default=None,
                    help='Output CSV path (single-model mode).')
    ap.add_argument('--output_dir', default=None,
                    help='Output directory (multi-model mode).')
    ap.add_argument('--ensemble', action='store_true',
                    help='Additionally save an Ensemble CSV (probability-averaged).')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Collect model paths
    if args.model_path:
        model_paths = [Path(args.model_path)]
        if not args.output_csv:
            ap.error('--output_csv is required with --model_path')
    else:
        d = Path(args.models_dir)
        model_paths = sorted([p for p in d.glob('*.pt')
                              if 'seed' in p.name.lower()])
        if not model_paths:
            sys.exit(f'No seed models found in {d}')
        if not args.output_dir:
            ap.error('--output_dir is required with --models_dir')
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f'\nModels to run ({len(model_paths)}):')
    for m in model_paths:
        print(f'  · {m.name}')

    # Load input
    df, smi_col, lbl_col, name_col = _load_input(args.input_csv)
    print(f'\nInput CSV: {args.input_csv}   n={len(df)}')
    print(f'  SMILES column: {smi_col}')
    if lbl_col:
        print(f'  Label column: {lbl_col} (retained in output)')
    if name_col:
        print(f'  Name column: {name_col}')

    # Featurisers (shared across models)
    mol2vec_path = config.get('mol2vec', {}).get('model_path')
    chemberta_pkl = config.get('mol2vec', {}).get('chemberta_pkl')
    if chemberta_pkl and Path(chemberta_pkl).exists():
        print(f'\nLoading ChemBERTa featuriser from {chemberta_pkl}...')
        featurizer = _ChemBERTaLookup(chemberta_pkl)
        # Force config to reflect the pretrained embedding dim
        if 'mol2vec' not in config:
            config['mol2vec'] = {}
        config['mol2vec']['embedding_dim'] = featurizer.hidden_size
    else:
        if not mol2vec_path or not Path(mol2vec_path).exists():
            mol2vec_path = 'data/mol2vec_model.pkl'
        print(f'\nLoading Mol2Vec featuriser from {mol2vec_path}...')
        featurizer = Mol2VecFeaturizer(model_path=mol2vec_path)
    conformer_gen = ConformerGenerator()
    cutoff = float(config.get('model', {}).get('cutoff', 5.0))

    # Pre-compute features ONCE (shared across all models)
    print('\nPre-computing 1D/2D/3D features (once for all seeds)...')
    features = []       # list of (v1d, g2d, g3d) or None
    valid_idx = []
    for i, r in df.iterrows():
        smi = str(r[smi_col]).strip()
        v1d, g2d, g3d = _featurise(smi, featurizer, conformer_gen, cutoff)
        if v1d is None:
            features.append(None)
        else:
            features.append((v1d, g2d, g3d))
            valid_idx.append(i)
        if (i + 1) % 50 == 0 or (i + 1) == len(df):
            print(f'  {i+1}/{len(df)} molecules featurised', flush=True)

    n_valid = sum(1 for f in features if f is not None)
    print(f'  ✓ {n_valid}/{len(df)} valid; {len(df) - n_valid} invalid molecules')

    # Run each model
    all_activity_probs = []       # list of arrays [n_input] with NaN for invalid
    for mp in model_paths:
        seed_tag = mp.stem
        print(f'\n=== Running {seed_tag} ===')
        model = _build_model(config, mp, args.device)

        activity_probs = np.full(len(df), np.nan)
        admet_probs = np.full((len(df), 5), np.nan)
        for i, feat in enumerate(features):
            if feat is None:
                continue
            v1d, g2d, g3d = feat
            act_p, admet_p = _predict_one(model, v1d, g2d, g3d, args.device)
            activity_probs[i] = act_p
            if admet_p is not None:
                admet_probs[i] = admet_p[:, 1]

        # Build output CSV
        out = pd.DataFrame({
            'idx': np.arange(len(df)),
            'smiles': df[smi_col].values,
            'activity_prob': activity_probs,
            'predicted_active': (activity_probs > 0.5).astype(int),
        })
        if name_col:
            out.insert(0, 'name', df[name_col].values)
        if lbl_col:
            out['label'] = df[lbl_col].values
        for j, n in enumerate(['lipinski', 'qed', 'pains', 'sa', 'logp']):
            out[f'admet_{n}_prob'] = admet_probs[:, j]

        # Save
        if args.model_path:
            out_path = args.output_csv
        else:
            out_path = str(Path(args.output_dir) / f'test_pred_{seed_tag}.csv')
        out.to_csv(out_path, index=False)

        n_active = int(out['predicted_active'].fillna(0).sum())
        print(f'  ✓ n_active = {n_active}/{n_valid}  → {out_path}')

        if lbl_col:
            # Quick metrics
            valid_mask = ~out['activity_prob'].isna()
            y = out.loc[valid_mask, 'label'].astype(int).values
            p = out.loc[valid_mask, 'activity_prob'].values
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(y, p)
                print(f'  · Test ROC-AUC = {auc:.4f}')
            except Exception:
                pass

        all_activity_probs.append(activity_probs)

    # Ensemble
    if args.ensemble and len(model_paths) > 1:
        print('\n=== Ensemble (probability average) ===')
        stack = np.stack(all_activity_probs, axis=0)
        # NaN-aware mean
        ens_prob = np.nanmean(stack, axis=0)

        ens_out = pd.DataFrame({
            'idx': np.arange(len(df)),
            'smiles': df[smi_col].values,
            'activity_prob': ens_prob,
            'predicted_active': (ens_prob > 0.5).astype(int),
        })
        if name_col:
            ens_out.insert(0, 'name', df[name_col].values)
        if lbl_col:
            ens_out['label'] = df[lbl_col].values

        ens_path = str(Path(args.output_dir) / 'test_pred_ensemble.csv')
        ens_out.to_csv(ens_path, index=False)
        print(f'  ✓ Saved → {ens_path}')

        if lbl_col:
            from sklearn.metrics import roc_auc_score
            valid = ~np.isnan(ens_prob)
            y = df.loc[valid, lbl_col].astype(int).values
            p = ens_prob[valid]
            try:
                auc = roc_auc_score(y, p)
                print(f'  · Ensemble Test ROC-AUC = {auc:.4f}')
            except Exception:
                pass

    print('\nDone.')


if __name__ == '__main__':
    main()
