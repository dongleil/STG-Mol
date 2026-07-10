#!/usr/bin/env python3
"""
Strict quality curation of existing ChEMBL NLRP3 data.

Applies four quality filters to data/processed/nlrp3/{train,val,test}.csv:

  Filter 1  — Keep only label=1 rows (drop the 1850 "other-target"
              molecules used previously as easy negatives; these are NOT
              confirmed NLRP3-inactive, they are just not-tested-on-NLRP3
              molecules from EGFR/RTK screening panels).
  Filter 2  — Keep only rows with label_rule='value_lt_10um' (i.e.,
              IC50 or Ki was numerically measured and ≤ 10 μM).
  Filter 3  — Optionally drop rows with data_source != 'NLRP3_ChEMBL'.
  Filter 4  — Enforce numerical IC50 ≤ --max_ic50 if a value_um column
              exists.

Output: strict actives-only CSV. This should be combined with high-quality
negatives (hard negatives, PubChem AID 1508591 confirmed inactive) at
merge time — NOT with the dropped other-target inactives.

Usage:
    python scripts/expand/curate_chembl_strict.py \\
        --input_dir data/processed/nlrp3 \\
        --output data/raw/chembl_strict_actives.csv \\
        --max_ic50 10.0
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True,
                    help='Directory with existing train/val/test.csv.')
    ap.add_argument('--output', required=True)
    ap.add_argument('--max_ic50', type=float, default=10.0,
                    help='Drop actives with value_um > this. Default 10.0 μM.')
    ap.add_argument('--enforce_nlrp3_source', action='store_true', default=True,
                    help='Keep only rows with data_source == NLRP3_ChEMBL.')
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    dfs = []
    for name in ['train.csv', 'val.csv', 'test.csv']:
        p = Path(args.input_dir) / name
        if not p.exists():
            print(f'  ⚠ {p} missing, skipping.')
            continue
        d = pd.read_csv(p, encoding='utf-8')
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    print(f'\nInput total: {len(df)} rows from {args.input_dir}')

    if 'label' not in df.columns:
        sys.exit('ERROR: no `label` column found.')

    # Filter 1: label = 1
    df_before = len(df)
    df = df[df['label'] == 1].reset_index(drop=True)
    print(f'  After label=1 filter:                          '
          f'{df_before} → {len(df)}  (dropped {df_before - len(df)})')

    # Filter 2: label_rule = value_lt_10um (confirmed numerical IC50)
    if 'label_rule' in df.columns:
        df_before = len(df)
        df = df[df['label_rule'] == 'value_lt_10um'].reset_index(drop=True)
        print(f'  After label_rule=value_lt_10um filter:         '
              f'{df_before} → {len(df)}  (dropped {df_before - len(df)})')

    # Filter 3: data_source = NLRP3_ChEMBL (exclude any leftover cross-target)
    if args.enforce_nlrp3_source and 'data_source' in df.columns:
        df_before = len(df)
        df = df[df['data_source'] == 'NLRP3_ChEMBL'].reset_index(drop=True)
        print(f'  After data_source=NLRP3_ChEMBL filter:         '
              f'{df_before} → {len(df)}  (dropped {df_before - len(df)})')

    # Filter 4: numerical IC50 threshold (belt-and-braces with filter 2)
    if 'value_um' in df.columns:
        df_before = len(df)
        df['value_um'] = pd.to_numeric(df['value_um'], errors='coerce')
        df = df[df['value_um'].notna()]
        df = df[df['value_um'] <= args.max_ic50].reset_index(drop=True)
        print(f'  After value_um ≤ {args.max_ic50} μM filter:              '
              f'{df_before} → {len(df)}  (dropped {df_before - len(df)})')

    # Deduplicate on inchi_key if available
    if 'inchi_key' in df.columns:
        df_before = len(df)
        df = df.drop_duplicates(subset='inchi_key', keep='first')\
               .reset_index(drop=True)
        print(f'  After inchi_key dedup:                         '
              f'{df_before} → {len(df)}  (dropped {df_before - len(df)})')

    # Report IC50 distribution
    if 'value_um' in df.columns and len(df) > 0:
        vals = df['value_um'].values
        print(f'\n  IC50 (μM) stats:')
        print(f'    min    = {vals.min():.4f}')
        print(f'    median = {pd.Series(vals).median():.4f}')
        print(f'    mean   = {vals.mean():.4f}')
        print(f'    max    = {vals.max():.4f}')

    df.to_csv(args.output, index=False, encoding='utf-8')
    print(f'\n✓ Wrote {len(df)} strict-quality NLRP3 actives to {args.output}')


if __name__ == '__main__':
    main()
