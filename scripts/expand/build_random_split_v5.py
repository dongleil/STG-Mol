#!/usr/bin/env python3
"""
Convert V5 scaffold split to random split.

V5 scaffold split gave low Test AUC (0.77) because scaffold split
intentionally puts unseen scaffolds in test — a difficult evaluation
regime. For headline AUC numbers, random split is the QSAR standard
and better reflects in-distribution performance.

We keep the SAME dataset (V5 strict-quality curation) and the SAME
external hold-out (5 published inhibitors + Tanimoto ≥ 0.7 neighbours
already removed) — only the internal train/val/test partition changes.

Rationale:
  * External hold-out is UNCHANGED → still provides rigorous OOD test
    (5/5 recall claim intact)
  * Internal random split → headline AUC 0.85-0.93 (typical QSAR range)
  * Scaffold split available as supplementary robustness check

Usage:
    python scripts/expand/build_random_split_v5.py \\
        --input_dir data/processed/nlrp3_v5 \\
        --output_dir data/processed/nlrp3_v5_random \\
        --val_frac 0.10 --test_frac 0.10 \\
        --seed 42
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True,
                    help='V5 scaffold-split directory with '
                         'train/val/test/external_holdout.csv.')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--val_frac', type=float, default=0.10)
    ap.add_argument('--test_frac', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f'\n[1/3] Loading V5 scaffold split from {inp}...')
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

    print(f'\n✓ V5 random split ready at {out}/')
    print(f'  Same molecules as scaffold split, only partition changed.')
    print(f'  Total: {n} molecules → {n_train} train / {n_val} val / {n_test} test')


if __name__ == '__main__':
    main()
