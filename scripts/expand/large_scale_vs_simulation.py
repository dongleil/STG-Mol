#!/usr/bin/env python3
"""
Large-scale virtual-screening simulation.

Simulates a realistic VS deployment scenario where actives are rare
(<1% of the screening library), unlike the balanced test set (44%
pos rate).

Protocol (per Truchon & Bayly 2007 recommendation):
  1. Take K published NLRP3 inhibitors as "spike-in actives"
     (default: the 5 known inhibitors held out from training).
  2. Sample N random "decoy" molecules from a large pool
     (default: ZINC or PubChem inactive subset). These are assumed
     inactive by lack of NLRP3 test.
  3. Run STG-Mol on all N + K molecules, rank by activity_prob.
  4. Compute EF@k% and BEDROC at the low pos-rate regime:
       pos_rate = K / (K + N), typically < 1%.

For example, K=5 actives + N=10000 decoys → pos_rate = 0.05% and
EF theoretical max ≈ 2001.

Usage:
    python scripts/expand/large_scale_vs_simulation.py \\
        --models_dir results/v26_v5/.../models \\
        --config configs/train_v26_v5.yaml \\
        --actives_csv data/known_inhibitors.csv \\
        --decoys_csv data/raw/pubchem_nlrp3_strict.csv \\
        --n_decoys 10000 \\
        --output_dir large_scale_vs
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def enrichment_factor(y_true, y_score, top_frac):
    n = len(y_true)
    top_n = max(1, int(np.ceil(n * top_frac)))
    order = np.argsort(-np.asarray(y_score, dtype=float))
    top_hits = int(np.sum(np.asarray(y_true)[order[:top_n]]))
    total_hits = int(np.sum(y_true))
    if total_hits == 0:
        return 0.0
    return (top_hits / top_n) / (total_hits / n)


def bedroc(y_true, y_score, alpha=20.0):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    n_actives = int(y_true.sum())
    if n_actives == 0 or n_actives == n:
        return 0.0
    R_a = n_actives / n
    order = np.argsort(-y_score)
    active_ranks = np.where(y_true[order] == 1)[0] + 1
    S = float(np.sum(np.exp(-alpha * active_ranks / n)))
    factor = R_a * np.sinh(alpha / 2.0) / (
        np.cosh(alpha / 2.0) - np.cosh(alpha / 2.0 - alpha * R_a))
    numerator = S * factor
    denominator = R_a * (1.0 - np.exp(-alpha)) / (np.exp(alpha / n) - 1.0)
    correction_term = 1.0 / (1.0 - np.exp(alpha * (1.0 - R_a)))
    val = numerator / denominator + correction_term
    return float(max(0.0, min(1.0, val)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--actives_csv', required=True,
                    help='CSV of known actives to spike into the pool.')
    ap.add_argument('--decoys_csv', required=True,
                    help='CSV of decoy / assumed-inactive molecules.')
    ap.add_argument('--n_decoys', type=int, default=10000,
                    help='Number of decoys to sample.')
    ap.add_argument('--models_dir', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Build pool -----------------------------------------------------
    print('\n[1/3] Building screening pool...')
    a_df = pd.read_csv(args.actives_csv, encoding='utf-8')
    a_smi_col = next((c for c in ['smiles', 'smiles_standardized',
                                    'canonical_smiles']
                       if c in a_df.columns), None)
    print(f'  Actives:      {len(a_df)} molecules  (spike-in)')

    d_df = pd.read_csv(args.decoys_csv, encoding='utf-8')
    d_smi_col = next((c for c in ['smiles', 'smiles_standardized',
                                    'canonical_smiles']
                       if c in d_df.columns), None)
    # Prefer decoys with label=0
    if 'label' in d_df.columns:
        d_df = d_df[d_df['label'] == 0].reset_index(drop=True)
    if len(d_df) > args.n_decoys:
        rng = np.random.RandomState(args.seed)
        d_df = d_df.iloc[rng.choice(len(d_df), args.n_decoys,
                                       replace=False)].reset_index(drop=True)
    print(f'  Decoys pool:  {len(d_df)} molecules')

    # ---- Assemble the pool CSV ------------------------------------------
    pool = pd.DataFrame({
        'smiles': list(a_df[a_smi_col].astype(str)) +
                   list(d_df[d_smi_col].astype(str)),
        'label':  [1] * len(a_df) + [0] * len(d_df),
        'name':   (list(a_df['name']) if 'name' in a_df.columns
                    else [f'active_{i}' for i in range(len(a_df))]) +
                  [f'decoy_{i}' for i in range(len(d_df))],
    })
    pool_path = out / 'screening_pool.csv'
    pool.to_csv(pool_path, index=False, encoding='utf-8')
    pos_rate = pool['label'].mean()
    print(f'\n  Pool assembled: {len(pool)} total, {int(pool["label"].sum())} '
          f'actives ({pos_rate*100:.3f}% pos rate)')
    print(f'  Pool → {pool_path}')

    # ---- Instructions ---------------------------------------------------
    print(f'\n[2/3] Now run predict_smiles.py on this pool:')
    print(f'\n  python scripts/predict_smiles.py \\')
    print(f'      --models_dir {args.models_dir} \\')
    print(f'      --config {args.config} \\')
    print(f'      --input_csv {pool_path} \\')
    print(f'      --output_dir {out}/predictions \\')
    print(f'      --ensemble')
    print(f'\n  Then re-run this script with --predictions_csv option to '
          f'compute EF/BEDROC.')

    # ---- If predictions already exist, compute metrics ------------------
    pred_csv = out / 'predictions' / 'test_pred_ensemble.csv'
    if pred_csv.exists():
        print(f'\n[3/3] Predictions exist at {pred_csv} — computing metrics...')
        pred = pd.read_csv(pred_csv)
        # Merge pred back to pool via smiles
        pool['activity_prob'] = pool['smiles'].map(
            dict(zip(pred['smiles'], pred['activity_prob'])))
        pool = pool.dropna(subset=['activity_prob']).reset_index(drop=True)

        y = pool['label'].astype(int).values
        p = pool['activity_prob'].astype(float).values

        print(f'\n=== Large-scale VS simulation ({len(pool)} molecules) ===')
        print(f'  Pos rate:      {y.mean()*100:.4f}%  ({y.sum()}/{len(y)})')
        print(f'  Theoretical max EF: {len(y)/y.sum():.1f}')

        from sklearn.metrics import roc_auc_score
        print(f'  ROC-AUC:       {roc_auc_score(y, p):.4f}')

        for tf in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
            ef = enrichment_factor(y, p, tf)
            top_n = int(np.ceil(len(y) * tf))
            print(f'  EF@{tf*100:.1f}% (top {top_n:5d}): {ef:8.2f}')

        for a in [20.0, 80.0, 160.0]:
            print(f'  BEDROC@α={a:.0f}:   {bedroc(y, p, a):.4f}')

        # Save summary
        rows = []
        for tf in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
            rows.append({
                'metric': f'EF@{tf*100}%',
                'top_n': int(np.ceil(len(y) * tf)),
                'value': enrichment_factor(y, p, tf),
                'theoretical_max': min(len(y) / y.sum(), 1 / tf),
            })
        for a in [20.0, 80.0, 160.0]:
            rows.append({'metric': f'BEDROC@α={a}', 'top_n': None,
                         'value': bedroc(y, p, a), 'theoretical_max': 1.0})
        rows.append({'metric': 'ROC-AUC', 'top_n': None,
                     'value': float(roc_auc_score(y, p)),
                     'theoretical_max': 1.0})
        summary = pd.DataFrame(rows)
        summary.to_csv(out / 'metrics_summary.csv', index=False,
                       encoding='utf-8')
        print(f'\n  ✓ Metrics summary → {out}/metrics_summary.csv')
    else:
        print(f'\n[3/3] Predictions not yet generated at {pred_csv}.')
        print('       Run the predict_smiles.py command above, then re-run '
              'this script.')


if __name__ == '__main__':
    main()
