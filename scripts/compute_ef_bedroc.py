#!/usr/bin/env python3
"""
Compute virtual-screening-specific metrics: EF@k% and BEDROC.

These metrics are standard in the virtual-screening literature but were
absent from the original STG-Mol reporting. Adding them significantly
strengthens the paper's methodological rigour.

Metrics:
    * EF@1% / EF@5% / EF@10%   -- Enrichment Factor at the top k% ranked by
                                    model score (higher = better early ranking)
    * BEDROC(α=20)              -- Boltzmann-Enhanced Discrimination of ROC
                                    (Truchon & Bayly, 2007); α=20 emphasises
                                    early recognition, values in [0, 1]
    * BEDROC(α=80)              -- α=80 for even earlier recognition emphasis

Usage:
    # Single prediction file (from training's per-seed test predictions)
    python scripts/compute_ef_bedroc.py \\
        --pred_csv results/v26_multitask/.../predictions/test_pred_seed42.csv

    # Aggregate across 5 seeds in a directory (typical for a full run)
    python scripts/compute_ef_bedroc.py \\
        --pred_dir results/v26_multitask/.../predictions

The script auto-detects label & probability columns and supports the
naming used by v26 training pipeline out-of-the-box.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Metric implementations
# --------------------------------------------------------------------------

def enrichment_factor(y_true: np.ndarray, y_score: np.ndarray,
                      top_frac: float = 0.01) -> float:
    """
    EF@k% = fraction of actives in top k% / overall fraction of actives.
    A random ranker gives EF = 1.0. Higher is better.
    """
    n = len(y_true)
    top_n = max(1, int(np.ceil(n * top_frac)))
    order = np.argsort(-np.asarray(y_score, dtype=float))
    top_hits = int(np.sum(np.asarray(y_true)[order[:top_n]]))
    total_hits = int(np.sum(y_true))
    if total_hits == 0 or top_n == 0:
        return 0.0
    ef = (top_hits / top_n) / (total_hits / n)
    return float(ef)


def bedroc(y_true: np.ndarray, y_score: np.ndarray,
           alpha: float = 20.0) -> float:
    """
    BEDROC (Truchon & Bayly 2007). Result in [0, 1], higher = better.
    Higher alpha weights the very top of the ranking more heavily.
    Reference implementation (matches RDKit's Metrics.CalcBEDROC).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    n_actives = int(y_true.sum())
    if n_actives == 0 or n_actives == n:
        return 0.0

    R_a = n_actives / n
    order = np.argsort(-y_score)
    active_ranks = np.where(y_true[order] == 1)[0] + 1     # 1-based

    # numerator: Σ_i exp(-α * r_i / n) over active ranks r_i
    S = float(np.sum(np.exp(-alpha * active_ranks / n)))

    # BEDROC formula (Truchon & Bayly 2007, Eq. 36)
    factor = R_a * np.sinh(alpha / 2.0) / (
        np.cosh(alpha / 2.0) - np.cosh(alpha / 2.0 - alpha * R_a))
    numerator = S * factor
    denominator = R_a * (1.0 - np.exp(-alpha)) / (np.exp(alpha / n) - 1.0)
    correction_term = 1.0 / (1.0 - np.exp(alpha * (1.0 - R_a)))
    bedroc_val = numerator / denominator + correction_term
    return float(max(0.0, min(1.0, bedroc_val)))


def compute_all_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        'EF@1%':        enrichment_factor(y_true, y_score, 0.01),
        'EF@2%':        enrichment_factor(y_true, y_score, 0.02),
        'EF@5%':        enrichment_factor(y_true, y_score, 0.05),
        'EF@10%':       enrichment_factor(y_true, y_score, 0.10),
        'EF@20%':       enrichment_factor(y_true, y_score, 0.20),
        'BEDROC@α=20':  bedroc(y_true, y_score, alpha=20.0),
        'BEDROC@α=80':  bedroc(y_true, y_score, alpha=80.0),
        'BEDROC@α=160': bedroc(y_true, y_score, alpha=160.0),
    }


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------

_LABEL_CANDIDATES = ['label', 'y_true', 'true_label', 'y_true_test', 'target', 'y']
_PROB_CANDIDATES = [
    'prob', 'pred_prob', 'y_pred_prob', 'prediction', 'y_score', 'score',
    'test_pred_prob', 'test_prob',
]


def _detect(colnames, candidates):
    lc = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand.lower() in lc:
            return lc[cand.lower()]
    # substring match fallback
    for c in colnames:
        for cand in candidates:
            if cand.lower() in c.lower():
                return c
    return None


def _load_prediction_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    lbl = _detect(df.columns, _LABEL_CANDIDATES)
    prob = _detect(df.columns, _PROB_CANDIDATES)
    if lbl is None or prob is None:
        raise ValueError(
            f'Cannot find label & probability columns in {csv_path}\n'
            f'  Columns: {df.columns.tolist()}\n'
            f'  Label candidates: {_LABEL_CANDIDATES}\n'
            f'  Prob candidates:  {_PROB_CANDIDATES}')
    y_true = df[lbl].astype(int).values
    y_score = df[prob].astype(float).values
    return y_true, y_score, len(df)


def _print_metrics(metrics: dict, prefix: str = ''):
    for k, v in metrics.items():
        print(f'  {prefix}{k:16s} = {v:.4f}')


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred_csv', type=str,
                    help='Single prediction CSV to evaluate.')
    ap.add_argument('--pred_dir', type=str,
                    help='Directory containing multiple prediction CSVs '
                         '(one per seed). Aggregates mean±std and Ensemble.')
    ap.add_argument('--output', type=str, default='ef_bedroc_report.json')
    args = ap.parse_args()

    if not args.pred_csv and not args.pred_dir:
        ap.error('Must specify --pred_csv or --pred_dir')

    if args.pred_csv:
        y_true, y_score, n = _load_prediction_csv(Path(args.pred_csv))
        metrics = compute_all_metrics(y_true, y_score)
        print(f'\n=== EF & BEDROC on {args.pred_csv} (n={n}) ===')
        _print_metrics(metrics)
        with open(args.output, 'w') as f:
            json.dump({'file': str(args.pred_csv), 'n': n,
                       'metrics': metrics}, f, indent=2)
        print(f'\n✓ Report → {args.output}')
        return

    # ---- aggregate over multiple CSVs ----
    pred_dir = Path(args.pred_dir)
    csvs = sorted([p for p in pred_dir.glob('*.csv')
                   if 'pred' in p.name.lower() or 'test' in p.name.lower()])
    if not csvs:
        # Fall back to any csv
        csvs = sorted(pred_dir.glob('*.csv'))
    if not csvs:
        sys.exit(f'No prediction CSVs found in {pred_dir}')

    print(f'\nFound {len(csvs)} prediction CSVs in {pred_dir}:')
    for c in csvs:
        print(f'  · {c.name}')

    all_scores = []
    y_true_common = None
    n_common = None
    per_seed_metrics = []

    for csv in csvs:
        yt, ys, n = _load_prediction_csv(csv)
        if y_true_common is None:
            y_true_common, n_common = yt, n
        else:
            if len(yt) != n_common or not np.array_equal(yt, y_true_common):
                print(f'  ⚠ {csv.name}: labels differ from first CSV — skipped.')
                continue
        all_scores.append(ys)
        per_seed_metrics.append(compute_all_metrics(yt, ys))

    if len(all_scores) == 0:
        sys.exit('No valid prediction CSVs.')

    # Ensemble = mean probability across seeds
    ensemble_score = np.mean(np.stack(all_scores, axis=0), axis=0)
    ensemble_metrics = compute_all_metrics(y_true_common, ensemble_score)

    # Per-seed aggregation
    keys = list(per_seed_metrics[0].keys())
    agg = {}
    for k in keys:
        vals = np.array([m[k] for m in per_seed_metrics], dtype=float)
        agg[k] = {'mean': float(vals.mean()), 'std': float(vals.std(ddof=0))}

    print(f'\n=== Aggregate over {len(all_scores)} seeds (mean ± std) ===')
    for k, v in agg.items():
        print(f'  {k:16s} = {v["mean"]:.4f} ± {v["std"]:.4f}')

    print(f'\n=== Ensemble (probability-average of {len(all_scores)} models) ===')
    _print_metrics(ensemble_metrics)

    report = {
        'source': str(pred_dir),
        'num_prediction_files': len(all_scores),
        'test_n': int(n_common),
        'per_seed_mean_std': agg,
        'ensemble': ensemble_metrics,
        'per_seed_detail': per_seed_metrics,
    }
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\n✓ Report → {args.output}')


if __name__ == '__main__':
    main()
