#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarise_fusion_ablation.py — aggregate fusion-component ablation results
into a single paper-ready Table 5.2.3 (§5.2.3 Hierarchical Fusion Ablation).

Expected directory layout (produced by run_fusion_ablation.sh):

    results/ablation_fusion/
        fusion_full/            <-- predictions/ or test_pred_*.csv here
        fusion_no_cross_attn/
        fusion_no_gated/
        fusion_no_bilinear/
        fusion_no_importance_net/

For each configuration we call recompute_metrics_v42.compute_metric_bundle on
every seed CSV, then report 5-seed mean ± std of the paper's headline metrics.
The final markdown table is emitted alongside a JSON blob for record-keeping.

Usage:
    python scripts/summarise_fusion_ablation.py \\
        --results_root results/ablation_fusion \\
        --output_md   fusion_ablation_table.md \\
        --output_json fusion_ablation.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recompute_metrics_v42 import (  # noqa: E402
    load_pred_csv, compute_metric_bundle, summarise, ensemble_metrics,
)


ABLATION_ORDER = [
    ('fusion_no_cross_attn',     '− Cross-Modal Attention'),
    ('fusion_no_gated',          '− Gated Fusion Unit'),
    ('fusion_no_bilinear',       '− Low-Rank Bilinear'),
    ('fusion_no_importance_net', '− Importance Net'),
    ('fusion_full',              '**Full Hierarchical (ours)**'),
]

METRICS_TO_REPORT = [
    ('ROC_AUC',        'ROC-AUC'),
    ('F1',             'F1'),
    ('MCC',            'MCC'),
    ('Recall',         'Recall'),
    ('BEDROC_alpha20', 'BEDROC@α=20'),
]


def _find_pred_csvs(config_dir: Path):
    """Locate per-seed prediction CSVs under a run directory.

    Tolerates a few conventions:
        <run>/predictions/*.csv
        <run>/*/predictions/*.csv     (5-seed nested runs)
        <run>/test_pred_seed*.csv
    """
    candidates = []
    candidates.extend(config_dir.rglob('test_pred_seed*.csv'))
    candidates.extend(config_dir.rglob('predictions/*.csv'))
    # De-dup preserving order
    seen, uniq = set(), []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    return sorted(uniq)


def evaluate_config(config_dir: Path):
    csvs = _find_pred_csvs(config_dir)
    if not csvs:
        return None
    per_seed, y_true_ref, scores = [], None, []
    for c in csvs:
        yt, ys = load_pred_csv(c)
        if y_true_ref is None:
            y_true_ref = yt
        elif len(yt) != len(y_true_ref) or not np.array_equal(yt, y_true_ref):
            print(f'  ⚠ skipping {c.name} — test labels differ from reference')
            continue
        per_seed.append(compute_metric_bundle(y_true_ref, ys))
        scores.append(ys)
    if not per_seed:
        return None
    agg = summarise(per_seed)
    ens = ensemble_metrics(np.stack(scores, axis=0), y_true_ref)
    return {'per_seed': per_seed, 'aggregate': agg, 'ensemble': ens,
            'n_seeds': len(per_seed),
            'test_N': int(per_seed[0]['N']), 'test_P': int(per_seed[0]['P'])}


def _fmt(agg_entry, digits=4):
    return f'{agg_entry["mean"]:.{digits}f} ± {agg_entry["std"]:.{digits}f}'


def render(results):
    lines = [
        '# Table 5.2.3 — Hierarchical Fusion Component Ablation',
        '',
        'V3-random 5-seed mean ± std. Full = all four components active; each '
        'row above removes one component to isolate its contribution.',
        '',
        '| Configuration | ' + ' | '.join(m[1] for m in METRICS_TO_REPORT) + ' |',
        '|---' * (1 + len(METRICS_TO_REPORT)) + '|',
    ]
    for key, label in ABLATION_ORDER:
        r = results.get(key)
        if r is None:
            row = f'| {label} | ' + ' | '.join('—' for _ in METRICS_TO_REPORT) + ' |'
        else:
            row = f'| {label} | ' + ' | '.join(
                _fmt(r['aggregate'][m[0]]) for m in METRICS_TO_REPORT) + ' |'
        lines.append(row)

    lines += ['', '## Ensemble (deployment-time) view', '',
              '| Configuration | ' + ' | '.join(m[1] for m in METRICS_TO_REPORT) + ' |',
              '|---' * (1 + len(METRICS_TO_REPORT)) + '|']
    for key, label in ABLATION_ORDER:
        r = results.get(key)
        if r is None:
            row = f'| {label} | ' + ' | '.join('—' for _ in METRICS_TO_REPORT) + ' |'
        else:
            row = f'| {label} | ' + ' | '.join(
                f'{r["ensemble"][m[0]]:.4f}' for m in METRICS_TO_REPORT) + ' |'
        lines.append(row)
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_root', required=True,
                    help='Directory containing one sub-dir per ablation config.')
    ap.add_argument('--output_md',    default='fusion_ablation_table.md')
    ap.add_argument('--output_json',  default='fusion_ablation.json')
    args = ap.parse_args()

    root = Path(args.results_root)
    results = {}
    for key, label in ABLATION_ORDER:
        # Match either exact dir name or any prefix (v26_v3_random_<key>).
        candidates = [d for d in root.iterdir()
                       if d.is_dir() and (d.name == key or key in d.name)]
        if not candidates:
            print(f'  · {key}: no sub-directory found under {root}')
            results[key] = None
            continue
        run_dir = candidates[0]
        r = evaluate_config(run_dir)
        results[key] = r
        if r is None:
            print(f'  · {key}: no prediction CSVs found in {run_dir}')
        else:
            print(f'  ✓ {key} ({r["n_seeds"]} seeds): '
                  f'ROC-AUC = {_fmt(r["aggregate"]["ROC_AUC"])}, '
                  f'ensemble ROC-AUC = {r["ensemble"]["ROC_AUC"]:.4f}')

    md = render(results)
    Path(args.output_md).write_text(md, encoding='utf-8')
    Path(args.output_json).write_text(json.dumps({
        k: (None if r is None else {
            'n_seeds': r['n_seeds'],
            'test_N': r['test_N'], 'test_P': r['test_P'],
            'aggregate': r['aggregate'], 'ensemble': r['ensemble'],
        }) for k, r in results.items()
    }, indent=2), encoding='utf-8')

    print()
    print(f'✓ Table   → {args.output_md}')
    print(f'✓ Details → {args.output_json}')


if __name__ == '__main__':
    main()
