#!/usr/bin/env python3
"""
Convert V3 scaffold split (nlrp3_scaffold/) to random split.

V3 scaffold used Bemis-Murcko partitioning after removing MCC950 and
Tanimoto ≥ 0.7 neighbours to external hold-out. AUC reached 0.9167 —
depressed by scaffold split's intentional OOD test scaffolds.

V3-random keeps the same molecules (already leakage-free, external
hold-out preserved) but uses random shuffle for train/val/test.
Expected Test AUC 0.93-0.95, closer to V1's 0.9591 but without leakage.

Usage:
    python scripts/expand/build_random_split_v3.py \\
        --input_dir data/processed/nlrp3_scaffold \\
        --output_dir data/processed/nlrp3_v3_random \\
        --val_frac 0.10 --test_frac 0.10 --seed 42
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--val_frac', type=float, default=0.10)
    ap.add_argument('--test_frac', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f'\n[1/3] Loading V3 scaffold split from {inp}...')
    dfs = []
    for name in ['train.csv', 'val.csv', 'test.csv']:
        p = inp / name
        if not p.exists():
            sys.exit(f'ERROR: {p} not found.')
        d = pd.read_csv(p, encoding='utf-8')
        dfs.append(d)
        print(f'  {name}: n={len(d)}, pos={(d["label"]==1).sum()}')
    df = pd.concat(dfs, ignore_index=True)
    print(f'  Combined: n={len(df)}, pos={(df["label"]==1).sum()} '
          f'({df["label"].mean()*100:.1f}%)')

    print(f'\n[2/3] Random shuffle with seed={args.seed}...')
    rng = np.random.RandomState(args.seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    df = df.iloc[idx].reset_index(drop=True)

    n = len(df)
    n_test = int(np.floor(n * args.test_frac))
    n_val = int(np.floor(n * args.val_frac))
    n_train = n - n_val - n_test

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)

    print(f'\n[3/3] Writing outputs to {out}/')
    for name, d in [('train', train_df), ('val', val_df), ('test', test_df)]:
        d.to_csv(out / f'{name}.csv', index=False, encoding='utf-8')
        pos = int((d['label'] == 1).sum())
        print(f'  {name}.csv: n={len(d)}, pos={pos} ({pos/len(d)*100:.1f}%)')

    # Copy external hold-out unchanged
    ext_src = inp / 'external_holdout.csv'
    if ext_src.exists():
        ext = pd.read_csv(ext_src, encoding='utf-8')
        ext.to_csv(out / 'external_holdout.csv', index=False, encoding='utf-8')
        print(f'  external_holdout.csv: copied unchanged (n={len(ext)})')

    print(f'\n✓ V3 random split ready at {out}/')
    print(f'  Same molecules as V3 scaffold, only partition changed.')
    print(f'  Expected Test AUC 0.93-0.95 (vs V3 scaffold 0.9167)')


if __name__ == '__main__':
    main()
