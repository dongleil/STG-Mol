#!/usr/bin/env python3
"""
Helper: build a single all-actives CSV by merging all sources' positives.
Used as input to generate_decoys.py.

Usage:
    python scripts/expand/build_all_actives.py \\
        --chembl_dir data/processed/nlrp3 \\
        --bindingdb data/raw/bindingdb_nlrp3.csv \\
        --pubchem   data/raw/pubchem_nlrp3.csv \\
        --literature data/raw/literature_nlrp3_curated.csv \\
        --output data/raw/all_actives.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

_SMI_CANDS = ['smiles_standardized', 'smiles', 'canonical_smiles', 'SMILES',
              'ligand_smiles']


def _find_smi_col(cols):
    lc = {c.lower(): c for c in cols}
    for c in _SMI_CANDS:
        if c.lower() in lc:
            return lc[c.lower()]
    return None


def collect(path, tag, active_only=True):
    if not path or not Path(path).exists():
        print(f'  ⚠ {path} missing, skipping.')
        return pd.DataFrame()
    if Path(path).is_dir():
        dfs = []
        for name in ['train.csv', 'val.csv', 'test.csv']:
            p = Path(path) / name
            if p.exists():
                dfs.append(pd.read_csv(p))
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    else:
        df = pd.read_csv(path)
    if len(df) == 0:
        return df
    smi_col = _find_smi_col(df.columns)
    if not smi_col:
        return pd.DataFrame()
    if active_only and 'label' in df.columns:
        df = df[df['label'] == 1]
    return pd.DataFrame({'smiles': df[smi_col].astype(str), 'label': 1,
                         'source': tag})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chembl_dir', required=True)
    ap.add_argument('--bindingdb', default=None)
    ap.add_argument('--pubchem', default=None)
    ap.add_argument('--literature', default=None)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print('Collecting actives from all sources...')
    parts = [
        collect(args.chembl_dir, 'ChEMBL'),
        collect(args.bindingdb, 'BindingDB'),
        collect(args.pubchem, 'PubChem'),
        collect(args.literature, 'Literature'),
    ]
    df = pd.concat([p for p in parts if len(p) > 0], ignore_index=True)
    df = df.drop_duplicates(subset='smiles', keep='first').reset_index(drop=True)
    df.to_csv(args.output, index=False)
    print(f'\n✓ Wrote {len(df)} unique actives to {args.output}')
    for src, count in df['source'].value_counts().items():
        print(f'  {src:25s}  {count}')


if __name__ == '__main__':
    main()
