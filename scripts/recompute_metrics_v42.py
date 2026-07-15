#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recompute_metrics_v42.py — one-shot metric recomputation for paper v4.2.

Reads 5 per-seed prediction CSVs from a training run and produces every
number the paper's Table 5.1 / 5.1b / 5.1c cite:

    * ROC-AUC, PR-AUC, Accuracy, F1, MCC, Precision, Recall
      -- per seed, 5-seed mean ± std, and 5-seed probability-averaged ensemble
    * BEDROC @ α = 20 / 80 / 160  (Truchon & Bayly 2007)
    * EF @ 5% / 10% / 20%
      -- with an assert against the theoretical upper bound N / P
      -- EF@1% is deliberately NOT reported (top-k=2 on 193-molecule test set
         is subject to discretisation artefacts; see paper §4.3 & §5.1)

Also writes a paper-ready markdown snippet (Table 5.1b / 5.1c rows) so the
values drop straight into build_paper_v4_en.py / build_paper_v4.py.

Usage:
    python scripts/recompute_metrics_v42.py \\
        --pred_dir results/v3_random_5seeds/predictions \\
        --split_name V3-random \\
        --output_json v42_metrics_v3_random.json \\
        --output_md   v42_metrics_v3_random.md

If your run directory does not follow the convention `test_pred_seed{seed}.csv`,
override the glob with --glob "*.csv".
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, matthews_corrcoef, precision_score, recall_score,
)

# Reuse the well-tested BEDROC / EF implementation shipped with the repo.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_ef_bedroc import bedroc, enrichment_factor  # noqa: E402


# --------------------------------------------------------------------------
# Metric wrappers
# --------------------------------------------------------------------------

def _pick(colnames, candidates):
    lc = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand.lower() in lc:
            return lc[cand.lower()]
    for c in colnames:
        for cand in candidates:
            if cand.lower() in c.lower():
                return c
    return None


_LABEL_CANDS = ['label', 'y_true', 'true_label', 'target', 'y']
_PROB_CANDS = [
    'prob', 'pred_prob', 'y_pred_prob', 'prediction', 'y_score', 'score',
    'test_pred_prob', 'test_prob',
]


def load_pred_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    lbl = _pick(df.columns, _LABEL_CANDS)
    prob = _pick(df.columns, _PROB_CANDS)
    if lbl is None or prob is None:
        raise ValueError(
            f'Could not locate label & prob columns in {csv_path}\n'
            f'  Available columns: {df.columns.tolist()}')
    return df[lbl].astype(int).values, df[prob].astype(float).values


def compute_metric_bundle(y_true: np.ndarray, y_score: np.ndarray,
                          threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    N = len(y_true)
    P = int(y_true.sum())
    EF_MAX = N / P if P > 0 else float('inf')

    out = {
        'N': N,
        'P': P,
        'prevalence': P / N,
        'EF_theoretical_max': EF_MAX,
        'ROC_AUC':   float(roc_auc_score(y_true, y_score)),
        'PR_AUC':    float(average_precision_score(y_true, y_score)),
        'Accuracy':  float(accuracy_score(y_true, y_pred)),
        'F1':        float(f1_score(y_true, y_pred)),
        'MCC':       float(matthews_corrcoef(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall':    float(recall_score(y_true, y_pred)),
        'BEDROC_alpha20':  float(bedroc(y_true, y_score, alpha=20.0)),
        'BEDROC_alpha80':  float(bedroc(y_true, y_score, alpha=80.0)),
        'BEDROC_alpha160': float(bedroc(y_true, y_score, alpha=160.0)),
        'EF_5pct':  float(enrichment_factor(y_true, y_score, 0.05)),
        'EF_10pct': float(enrichment_factor(y_true, y_score, 0.10)),
        'EF_20pct': float(enrichment_factor(y_true, y_score, 0.20)),
    }

    # ---- Physical-plausibility sanity checks (guards the v4.1 → v4.2 fix) ----
    tol = 1e-6
    for k in ('EF_5pct', 'EF_10pct', 'EF_20pct'):
        assert out[k] <= EF_MAX + tol, (
            f'{k}={out[k]:.4f} exceeds theoretical upper bound '
            f'{EF_MAX:.4f}=N/P — check test-set label / probability alignment')
    for k in ('BEDROC_alpha20', 'BEDROC_alpha80', 'BEDROC_alpha160',
              'ROC_AUC', 'PR_AUC'):
        assert 0.0 - tol <= out[k] <= 1.0 + tol, f'{k}={out[k]} out of [0,1]'
    return out


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def summarise(per_seed: list) -> dict:
    """Compute mean ± std over the seed-level metric dicts."""
    keys = [k for k in per_seed[0].keys()
            if k not in ('N', 'P', 'prevalence', 'EF_theoretical_max')]
    agg = {}
    for k in keys:
        vals = np.array([m[k] for m in per_seed])
        agg[k] = {
            'mean': float(vals.mean()),
            'std':  float(vals.std(ddof=0)),
            'min':  float(vals.min()),
            'max':  float(vals.max()),
            'per_seed': [float(v) for v in vals],
        }
    return agg


def ensemble_metrics(all_scores: np.ndarray, y_true: np.ndarray) -> dict:
    """Probability-averaged ensemble across seeds."""
    y_score = all_scores.mean(axis=0)
    return compute_metric_bundle(y_true, y_score)


# --------------------------------------------------------------------------
# Markdown snippet writer
# --------------------------------------------------------------------------

def fmt_mean_std(agg_entry, digits=4):
    return f'{agg_entry["mean"]:.{digits}f} ± {agg_entry["std"]:.{digits}f}'


def render_paper_snippet(split_name: str, agg: dict, ens: dict,
                         theoretical_max: float) -> str:
    lines = [
        f'# Paper snippet — {split_name} (auto-generated by recompute_metrics_v42.py)',
        '',
        f'Test set: N = {ens["N"]}, positives = {ens["P"]}, '
        f'prevalence = {ens["prevalence"]:.4f}, '
        f'EF theoretical upper bound = N/P = {theoretical_max:.4f}',
        '',
        '## Table 5.1 / 5.1b — Overall discrimination',
        '',
        '| Metric | 5-seed mean ± std | 5-seed ensemble |',
        '|---|---|---|',
        f'| ROC-AUC   | {fmt_mean_std(agg["ROC_AUC"])} | {ens["ROC_AUC"]:.4f} |',
        f'| PR-AUC    | {fmt_mean_std(agg["PR_AUC"])} | {ens["PR_AUC"]:.4f} |',
        f'| F1        | {fmt_mean_std(agg["F1"])} | {ens["F1"]:.4f} |',
        f'| MCC       | {fmt_mean_std(agg["MCC"])} | {ens["MCC"]:.4f} |',
        f'| Recall    | {fmt_mean_std(agg["Recall"])} | {ens["Recall"]:.4f} |',
        f'| Precision | {fmt_mean_std(agg["Precision"])} | {ens["Precision"]:.4f} |',
        f'| Accuracy  | {fmt_mean_std(agg["Accuracy"])} | {ens["Accuracy"]:.4f} |',
        '',
        '## Table 5.1c — Early-recognition & enrichment metrics',
        '',
        '| Metric | 5-seed mean ± std | 5-seed ensemble | Theoretical upper bound |',
        '|---|---|---|---|',
        f'| BEDROC@α=20  | {fmt_mean_std(agg["BEDROC_alpha20"])} | {ens["BEDROC_alpha20"]:.4f} | 1.0000 |',
        f'| BEDROC@α=80  | {fmt_mean_std(agg["BEDROC_alpha80"])} | {ens["BEDROC_alpha80"]:.4f} | 1.0000 |',
        f'| BEDROC@α=160 | {fmt_mean_std(agg["BEDROC_alpha160"])} | {ens["BEDROC_alpha160"]:.4f} | 1.0000 |',
        f'| EF@5%  | {fmt_mean_std(agg["EF_5pct"])} | {ens["EF_5pct"]:.4f} | {theoretical_max:.4f} |',
        f'| EF@10% | {fmt_mean_std(agg["EF_10pct"])} | {ens["EF_10pct"]:.4f} | {theoretical_max:.4f} |',
        f'| EF@20% | {fmt_mean_std(agg["EF_20pct"])} | {ens["EF_20pct"]:.4f} | {theoretical_max:.4f} |',
        '',
        '> EF@1% is intentionally **not reported**: on this 193-molecule test set '
        'top-1% corresponds to k = 2 molecules, which yields a discretised '
        'estimator with high variance and, in an earlier version of this manuscript, '
        'produced a value (3.76) that exceeded the theoretical upper bound of '
        f'{theoretical_max:.3f} = N/P. This has been corrected in v4.2.',
        '',
        '## Per-seed detail (for supplementary S3)',
        '',
        '| Metric | seed 42 | seed 123 | seed 2024 | seed 3407 | seed 7 |',
        '|---|---|---|---|---|---|',
    ]
    seed_labels = agg['ROC_AUC']['per_seed']
    for k in ('ROC_AUC', 'F1', 'MCC', 'Recall',
              'BEDROC_alpha20', 'EF_5pct', 'EF_10pct'):
        row = f'| {k} | ' + ' | '.join(
            f'{v:.4f}' for v in agg[k]['per_seed']) + ' |'
        lines.append(row)
    _ = seed_labels  # unused after header
    return '\n'.join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_dir', required=True,
                    help='Directory holding per-seed prediction CSVs.')
    ap.add_argument('--which', choices=['test', 'val'], default='test',
                    help='Which split to aggregate (default: test). Files whose '
                         'basename starts with the chosen prefix are selected; '
                         'ensemble files and cross-split files are ignored.')
    ap.add_argument('--include_ensemble', action='store_true',
                    help='Include the ensemble CSV alongside per-seed CSVs. '
                         'By default only per-seed rows are aggregated, and the '
                         'ensemble is recomputed as the probability average '
                         'across seeds to avoid double-counting.')
    ap.add_argument('--glob', default='*.csv',
                    help='Glob pattern for prediction files (default: *.csv).')
    ap.add_argument('--split_name', default='V3-random',
                    help='Human label for the split (used in the report).')
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='Decision threshold for F1/MCC/Recall/Precision.')
    ap.add_argument('--output_json', default='v42_metrics.json')
    ap.add_argument('--output_md',   default='v42_metrics.md')
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    all_csvs = sorted(pred_dir.glob(args.glob))

    # Keep only per-seed CSVs of the requested split (test_ or val_ prefix,
    # + suffix _seed<digits>). Ensemble files are excluded unless the user
    # opts in via --include_ensemble.
    prefix = f'{args.which}_pred'
    seed_re = re.compile(r'_seed\d+\.csv$', re.IGNORECASE)
    csvs = [c for c in all_csvs
            if c.name.lower().startswith(prefix) and seed_re.search(c.name)]
    if args.include_ensemble:
        csvs += [c for c in all_csvs
                 if c.name.lower().startswith(prefix)
                 and 'ensemble' in c.name.lower()]
    if not csvs:
        sys.exit(f'No {args.which} prediction CSVs found in {pred_dir} '
                 f'(expected names like "{prefix}_...seed42.csv").')

    print(f'\nSelected {len(csvs)} {args.which} prediction CSV(s) in {pred_dir}:')
    for c in csvs:
        print(f'  · {c.name}')

    # ---- Load ----
    y_true_ref, scores_by_seed = None, []
    for c in csvs:
        yt, ys = load_pred_csv(c)
        if y_true_ref is None:
            y_true_ref = yt
        else:
            if len(yt) != len(y_true_ref) or not np.array_equal(yt, y_true_ref):
                sys.exit(f'Test-label mismatch between {csvs[0].name} and {c.name}')
        scores_by_seed.append(ys)
    scores_by_seed = np.stack(scores_by_seed, axis=0)

    # ---- Per-seed & aggregate ----
    per_seed = [
        compute_metric_bundle(y_true_ref, scores_by_seed[i], args.threshold)
        for i in range(scores_by_seed.shape[0])
    ]
    agg = summarise(per_seed)
    ens = ensemble_metrics(scores_by_seed, y_true_ref)

    # ---- Write ----
    payload = {
        'split_name': args.split_name,
        'threshold': args.threshold,
        'n_seeds': len(csvs),
        'pred_files': [str(c) for c in csvs],
        'test_set': {
            'N': int(per_seed[0]['N']),
            'P': int(per_seed[0]['P']),
            'prevalence': per_seed[0]['prevalence'],
            'EF_theoretical_max': per_seed[0]['EF_theoretical_max'],
        },
        'per_seed_metrics': per_seed,
        'aggregate_5seed_mean_std': agg,
        'ensemble_metrics': ens,
    }
    with open(args.output_json, 'w') as f:
        json.dump(payload, f, indent=2)

    md = render_paper_snippet(
        args.split_name, agg, ens, per_seed[0]['EF_theoretical_max'])
    Path(args.output_md).write_text(md, encoding='utf-8')

    # ---- Console summary ----
    print()
    print('=' * 72)
    print(f'  {args.split_name} — 5-seed summary')
    print('=' * 72)
    print(f'  ROC-AUC (mean±std):  {fmt_mean_std(agg["ROC_AUC"])}')
    print(f'  ROC-AUC (ensemble):  {ens["ROC_AUC"]:.4f}')
    print(f'  BEDROC@α=20  (ens):  {ens["BEDROC_alpha20"]:.4f}')
    print(f'  EF@5% (ens):         {ens["EF_5pct"]:.4f}  '
          f'(max {per_seed[0]["EF_theoretical_max"]:.4f})')
    print(f'  EF@10% (ens):        {ens["EF_10pct"]:.4f}')
    print(f'  EF@20% (ens):        {ens["EF_20pct"]:.4f}')
    print()
    print(f'✓ JSON  → {args.output_json}')
    print(f'✓ Paper → {args.output_md}')


if __name__ == '__main__':
    main()
