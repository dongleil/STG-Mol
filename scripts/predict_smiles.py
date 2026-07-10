#!/usr/bin/env python3
"""
Batch-predict activity + ADMET for arbitrary SMILES using a trained STG-Mol
(Multi-Task) model.

Primary use case: **Positive-control test** — verify that the trained model
correctly recognises published NLRP3 inhibitors (MCC950, CY-09, OLT1177,
Oridonin, Tranilast, etc.).

Usage:
    # Predict a small CSV of known inhibitors
    python scripts/predict_smiles.py \\
        --model_path results/v26_multitask/.../models/MultiModal_1D+2D+3D_seed42.pt \\
        --config configs/train_v26_multitask.yaml \\
        --input_csv data/known_inhibitors.csv \\
        --output_csv predictions_known_inhibitors.csv

Input CSV expects columns:
    * `smiles`   (required)
    * `name`     (optional, for readability)

Output CSV columns:
    * name, smiles, activity_prob, predicted_active,
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

# Import training code (must be run from repo root or with STG-Mol in PYTHONPATH)
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


def build_features_for_one(smi: str, mol2vec_featurizer,
                            conformer_gen, cutoff: float):
    """Build 1D/2D/3D features for one SMILES. Returns (v1d, g2d, g3d)."""
    try:
        v1d = mol2vec_featurizer.featurize([smi])[0]
    except Exception:
        v1d = np.zeros(300, dtype=np.float32)
    g2d = mol_to_2d_graph(smi)
    g3d = mol_to_3d_graph(smi, cutoff, conformer_gen)
    return v1d, g2d, g3d


def predict_one(model, v1d, g2d, g3d, device):
    """Return activity_prob (P(active)) and admet_probs (5 dims)."""
    v1d_t = torch.tensor(np.asarray(v1d), dtype=torch.float).unsqueeze(0).to(device)
    g2d = g2d.to(device)
    g3d = g3d.to(device)
    # Provide dummy batch tensor for PyG batching
    if not hasattr(g2d, 'batch') or g2d.batch is None:
        g2d.batch = torch.zeros(g2d.num_nodes, dtype=torch.long, device=device)
    if not hasattr(g3d, 'batch') or g3d.batch is None:
        g3d.batch = torch.zeros(g3d.num_nodes, dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(mol2vec_feat=v1d_t, graph_2d=g2d, graph_3d=g3d)

    logits = out['logits']
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    admet_probs = None
    if 'admet_logits' in out and out['admet_logits'] is not None:
        admet_logits = out['admet_logits']    # [1, 5, 2]
        admet_probs = torch.softmax(admet_logits, dim=-1).cpu().numpy()[0]   # [5, 2]

    return float(probs[1]), admet_probs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model_path', required=True,
                    help='Path to a trained STG-Mol .pt checkpoint.')
    ap.add_argument('--config', required=True,
                    help='Config YAML used during training (for hyper-parameters).')
    ap.add_argument('--input_csv', required=True,
                    help='CSV with a `smiles` column (and optional `name`).')
    ap.add_argument('--output_csv', default='predictions.csv')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Load model
    print(f'\nBuilding model from config...')
    model_cfg = config.get('model', {}).copy()
    model_cfg['fusion_mode'] = '1D+2D+3D'
    model = MultiModalFusionNet(model_cfg)
    state = torch.load(args.model_path, map_location=args.device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state, strict=False)
    model = model.to(args.device).eval()
    print(f'  ✓ Loaded {args.model_path}')

    # Featurisers
    mol2vec_path = config.get('mol2vec', {}).get('model_path')
    if not mol2vec_path or not Path(mol2vec_path).exists():
        print(f'  ⚠ mol2vec model_path not found: {mol2vec_path}')
        print(f'  Trying default: data/mol2vec_model.pkl')
        mol2vec_path = 'data/mol2vec_model.pkl'
    print(f'  ✓ Loading Mol2Vec from {mol2vec_path}')
    featurizer = Mol2VecFeaturizer(model_path=mol2vec_path)
    conformer_gen = ConformerGenerator()
    cutoff = float(config.get('model', {}).get('cutoff', 5.0))

    # Read input
    df = pd.read_csv(args.input_csv)
    if 'smiles' not in df.columns:
        sys.exit(f'ERROR: input CSV must have a `smiles` column. Got: {df.columns.tolist()}')
    has_name = 'name' in df.columns
    print(f'\nPredicting {len(df)} SMILES...')

    rows = []
    for i, r in df.iterrows():
        smi = str(r['smiles']).strip()
        name = str(r['name']) if has_name else smi[:30]
        try:
            v1d, g2d, g3d = build_features_for_one(
                smi, featurizer, conformer_gen, cutoff)
            if g2d is None or g3d is None:
                rows.append({'name': name, 'smiles': smi, 'error': 'invalid molecule'})
                print(f'  [{i+1}/{len(df)}] {name}: ⚠ invalid')
                continue
            activity_p, admet_p = predict_one(model, v1d, g2d, g3d, args.device)
            row = {
                'name': name, 'smiles': smi,
                'activity_prob': round(activity_p, 4),
                'predicted_active': int(activity_p > 0.5),
            }
            if admet_p is not None:
                names = ['lipinski', 'qed', 'pains', 'sa', 'logp']
                for j, n in enumerate(names):
                    row[f'admet_{n}_prob'] = round(float(admet_p[j, 1]), 4)
            rows.append(row)
            marker = '✓' if row['predicted_active'] else '✗'
            print(f'  [{i+1}/{len(df)}] {name:30s} {marker}  activity={activity_p:.4f}')
        except Exception as e:
            rows.append({'name': name, 'smiles': smi, 'error': str(e)})
            print(f'  [{i+1}/{len(df)}] {name}: ERROR - {e}')

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output_csv, index=False)

    # Summary
    if 'predicted_active' in out_df.columns:
        n_pred_active = int(out_df['predicted_active'].fillna(0).sum())
        n_total = len(out_df)
        print(f'\n=== Summary ===')
        print(f'  Predicted active: {n_pred_active} / {n_total}')
        print(f'  Recall on known inhibitors: {n_pred_active/n_total:.1%}')
    print(f'\n✓ Predictions saved → {args.output_csv}')


if __name__ == '__main__':
    main()
