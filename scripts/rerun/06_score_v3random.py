#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_score_v3random.py — STG-Mol V3-random 5-seed ensemble re-scoring
                       of the 8 selected candidates.

Delegates the heavy lifting to `scripts/predict_smiles.py` (already tested
on positive controls), then aggregates the per-seed CSVs into:
  * v3random_scores.json — {compound_idx, smiles, mean_prob, per_seed[5], range}
  * table_5_7_rerun.md   — paper-ready markdown table 5.7

Rationale for reuse: predict_smiles.py already handles Mol2Vec featurisation +
2D/3D graph build via the same featuriser used by train_v26.py, and supports
--ensemble to average the 5 seed checkpoints.

Usage:
  conda activate nlrp3
  python3 scripts/rerun/06_score_v3random.py

Prereqs:
  # models_dir contains model_1D_2D_3D_seed{0..4}.pt (or seed42.. seed46 etc.)
  # config points to configs/train_v26_v3_random.yaml
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config():
    return yaml.safe_load(
        (Path(__file__).resolve().parent / 'config.yaml').read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top8', help='override path to top8.csv')
    ap.add_argument('--models_dir', help='override stg_mol_ckpt_dir')
    ap.add_argument('--stg_config',  help='override stg_mol_config')
    ap.add_argument('--out_json',    help='override output JSON path')
    ap.add_argument('--out_md',      help='override output markdown path')
    args = ap.parse_args()

    cfg = load_config()
    out = Path(cfg['output_dir'])
    cand_dir = out / 'candidates_v4.3'
    scores_dir = out / 'v3random_scores'
    scores_dir.mkdir(parents=True, exist_ok=True)

    top8_csv = Path(args.top8) if args.top8 else cand_dir / 'top8.csv'
    models_dir = Path(args.models_dir or cfg['stg_mol_ckpt_dir'])
    stg_config = Path(args.stg_config or cfg['stg_mol_config'])
    out_json = Path(args.out_json) if args.out_json else scores_dir / 'v3random_scores.json'
    out_md   = Path(args.out_md)   if args.out_md   else scores_dir / 'table_5_7_rerun.md'

    if not top8_csv.exists():
        raise FileNotFoundError(top8_csv)

    # Prepare input CSV in the schema predict_smiles.py expects
    input_csv = scores_dir / 'top8_for_predict.csv'
    rows = []
    with open(top8_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                'name': f'compound_{row["compound_idx"]}',
                'smiles': row['smiles'],
            })
    pd.DataFrame(rows).to_csv(input_csv, index=False)
    print(f'Wrote {input_csv} ({len(rows)} rows)')

    # Call predict_smiles.py in 5-seed ensemble mode
    # Usage (reproduced from scripts/predict_smiles.py --help header):
    #   python scripts/predict_smiles.py \
    #     --models_dir <ckpt_dir_with_5_seeds> \
    #     --config configs/train_v26_v3_random.yaml \
    #     --input_csv <top8.csv> \
    #     --output_dir <out_dir> \
    #     --ensemble
    repo_root = Path(__file__).resolve().parents[2]  # STG-Mol/
    predict_py = repo_root / 'scripts' / 'predict_smiles.py'
    if not predict_py.exists():
        raise FileNotFoundError(f'{predict_py} not found')

    cmd = [
        sys.executable, str(predict_py),
        '--models_dir', str(models_dir),
        '--config', str(stg_config),
        '--input_csv', str(input_csv),
        '--output_dir', str(scores_dir),
        '--ensemble',
    ]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)

    # Discover per-seed CSVs + the ensemble CSV
    seed_csvs = sorted(scores_dir.glob('predictions_seed*.csv'))
    ens_csvs  = sorted(scores_dir.glob('predictions_ensemble*.csv'))
    if not seed_csvs or not ens_csvs:
        # Fall back: use any CSV predict_smiles.py wrote
        seed_csvs = sorted(scores_dir.glob('*seed*.csv'))
        ens_csvs  = sorted(scores_dir.glob('*ensemble*.csv'))

    print(f'Found {len(seed_csvs)} seed CSVs, {len(ens_csvs)} ensemble CSVs')
    dfs = [pd.read_csv(p) for p in seed_csvs]
    df_ens = pd.read_csv(ens_csvs[0]) if ens_csvs else None

    # Aggregate → JSON + Markdown
    results = []
    with open(top8_csv) as f:
        top8_rows = list(csv.DictReader(f))
    for row in top8_rows:
        smi = row['smiles']
        idx = int(row['compound_idx'])
        per_seed = []
        for df in dfs:
            r_ = df[df['smiles'] == smi]
            if not r_.empty and 'activity_prob' in r_.columns:
                per_seed.append(float(r_['activity_prob'].iloc[0]))
        mean_prob = float(np.mean(per_seed)) if per_seed else None
        rng = (float(np.min(per_seed)), float(np.max(per_seed))) if per_seed else None
        # Optional: use the ensemble file if it has averaged probs
        ens_prob = None
        if df_ens is not None and 'smiles' in df_ens.columns:
            r_ = df_ens[df_ens['smiles'] == smi]
            if not r_.empty and 'activity_prob' in r_.columns:
                ens_prob = float(r_['activity_prob'].iloc[0])
        results.append({
            'compound_idx': idx,
            'smiles': smi,
            'vina_dG': float(row['vina_dG_kcal']),
            'per_seed_probs': per_seed,
            'mean_prob': mean_prob,
            'ensemble_prob': ens_prob,
            'range': rng,
        })

    out_json.write_text(json.dumps(results, indent=2))
    print(f'Wrote {out_json}')

    # Markdown table
    lines = [
        '# Table 5.7 — STG-Mol V3-random re-score of the 8 candidates',
        '',
        '| Compound | SMILES (short) | Vina ΔG (kcal/mol) | STG-Mol prob (mean ± range) |',
        '|---:|:---|---:|:---|',
    ]
    for r in results:
        smi_short = (r['smiles'][:40] + '…') if len(r['smiles']) > 40 else r['smiles']
        if r['mean_prob'] is None:
            prob_cell = 'N/A'
        else:
            lo, hi = r['range']
            prob_cell = f'{r["mean_prob"]:.3f}  ({lo:.3f}–{hi:.3f})'
        lines.append(
            f'| {r["compound_idx"]} | `{smi_short}` | {r["vina_dG"]:+.2f} | {prob_cell} |'
        )
    out_md.write_text('\n'.join(lines) + '\n')
    print(f'Wrote {out_md}')
    print('\nNext: python3 scripts/rerun/07_aggregate_tables.py')


if __name__ == '__main__':
    main()
