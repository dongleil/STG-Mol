#!/usr/bin/env python3
"""
Rebuild the NLRP3 dataset with a leakage-free, scaffold-based split.

Design goals (per Sheridan 2013, Yang et al. 2019 QSAR reviews):
  1. **External hold-out** — the 5 published NLRP3 inhibitors (MCC950,
     CY-09, OLT1177, Oridonin, Tranilast) plus their close neighbours
     (Morgan Tanimoto ≥ 0.7) are moved into an external_holdout.csv so
     they never influence training or validation.
  2. **Scaffold split** — the remaining molecules are partitioned by
     Bemis-Murcko scaffold with the largest scaffolds going to train
     and the rarest scaffolds going to val/test. This measures scaffold
     generalisation, not memorisation.

Inputs:
  data/processed/nlrp3/{train,val,test}.csv        -- current random split
  data/known_inhibitors.csv                        -- 5 published inhibitors

Outputs (in --output_dir):
  train.csv, val.csv, test.csv, external_holdout.csv, split_report.txt

Usage:
    python scripts/rebuild_dataset_scaffold_split.py \\
        --input_dir data/processed/nlrp3 \\
        --known_csv data/known_inhibitors.csv \\
        --output_dir data/processed/nlrp3_scaffold \\
        --near_tanimoto 0.7 \\
        --val_frac 0.10 --test_frac 0.10 \\
        --seed 42
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem.inchi import MolToInchiKey
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    sys.exit('ERROR: rdkit is required.')


# --------------------------------------------------------------------------
# Molecule helpers
# --------------------------------------------------------------------------
_SMI_CANDS = ['smiles_standardized', 'smiles', 'canonical_smiles', 'SMILES']


def _detect_smi_col(cols):
    lc = {c.lower(): c for c in cols}
    for c in _SMI_CANDS:
        if c.lower() in lc:
            return lc[c.lower()]
    return None


def _to_inchikey(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return MolToInchiKey(m) if m else None
    except Exception:
        return None


def _to_fp(smi, radius=2, n_bits=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)


def _to_scaffold(smi, include_chirality=False):
    """Return Bemis-Murcko scaffold SMILES. Empty for acyclic mols."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ''
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m,
                                                   includeChirality=include_chirality)
    except Exception:
        return ''


# --------------------------------------------------------------------------
# Scaffold split (from Chemprop / MoleculeNet)
# --------------------------------------------------------------------------
def scaffold_split(smiles_list, val_frac=0.10, test_frac=0.10, seed=42):
    """
    Deterministic scaffold split.
      * Scaffolds are grouped; groups are sorted by size (largest first).
      * Largest scaffold groups go to train; smaller ones to val/test.
      * Empty-scaffold molecules (acyclic) are treated as a single group.
    Returns (train_idx, val_idx, test_idx).
    """
    scaffold_to_indices = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scf = _to_scaffold(smi)
        scaffold_to_indices[scf].append(i)

    n = len(smiles_list)
    n_val = int(np.floor(n * val_frac))
    n_test = int(np.floor(n * test_frac))
    n_train = n - n_val - n_test

    # Sort scaffolds: big-first (as in Chemprop). Break ties deterministically.
    rng = np.random.RandomState(seed)
    groups = sorted(scaffold_to_indices.values(),
                    key=lambda g: (len(g), rng.random()),
                    reverse=True)

    train_idx, val_idx, test_idx = [], [], []
    for g in groups:
        if len(train_idx) + len(g) <= n_train:
            train_idx.extend(g)
        elif len(val_idx) + len(g) <= n_val:
            val_idx.extend(g)
        else:
            test_idx.extend(g)

    return train_idx, val_idx, test_idx, scaffold_to_indices


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True,
                    help='Directory with train/val/test.csv (random split).')
    ap.add_argument('--known_csv', required=True,
                    help='CSV of published NLRP3 inhibitors to hold out.')
    ap.add_argument('--output_dir', required=True,
                    help='Where to write the new scaffold-split CSVs.')
    ap.add_argument('--near_tanimoto', type=float, default=0.7,
                    help='Molecules with Tanimoto ≥ this to any known '
                         'inhibitor are also removed. Default 0.7.')
    ap.add_argument('--val_frac', type=float, default=0.10)
    ap.add_argument('--test_frac', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load & merge existing random split ------------------------------
    print(f'\n[1/5] Loading current random split from {inp}...')
    dfs = []
    for name in ['train.csv', 'val.csv', 'test.csv']:
        p = inp / name
        if not p.exists():
            sys.exit(f'ERROR: {p} not found.')
        d = pd.read_csv(p)
        d['_source_split'] = name.replace('.csv', '')
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    smi_col = _detect_smi_col(df.columns)
    if not smi_col:
        sys.exit(f'ERROR: no SMILES column in {df.columns.tolist()}')
    print(f'  Merged: n={len(df)}  (train={len(dfs[0])}, val={len(dfs[1])}, test={len(dfs[2])})')
    print(f'  SMILES column: {smi_col}')
    print(f'  Overall positive rate: {df["label"].astype(int).mean():.3f}')

    # ---- Dedup by canonical InChIKey ------------------------------------
    print(f'\n[2/5] Deduplicating by canonical InChIKey...')
    df['_inchikey'] = df[smi_col].apply(_to_inchikey)
    n_before = len(df)
    df = df[df['_inchikey'].notna()].copy()
    df = df.drop_duplicates(subset='_inchikey', keep='first').reset_index(drop=True)
    print(f'  Dropped {n_before - len(df)} rows (dup / invalid SMILES). '
          f'n_unique = {len(df)}')

    # ---- Load known inhibitors ------------------------------------------
    print(f'\n[3/5] Loading known inhibitors from {args.known_csv}...')
    known = pd.read_csv(args.known_csv)
    ksmi_col = _detect_smi_col(known.columns)
    if not ksmi_col:
        sys.exit(f'ERROR: no SMILES in {args.known_csv}')
    known['_inchikey'] = known[ksmi_col].apply(_to_inchikey)
    known['_fp'] = known[ksmi_col].apply(_to_fp)
    known_key_set = set(known['_inchikey'].dropna().tolist())
    known_fps = [(r['name'], r['_fp']) for _, r in known.iterrows()
                 if r['_fp'] is not None]
    print(f'  {len(known)} known inhibitors loaded.')

    # ---- Compute fingerprints for dataset (once) ------------------------
    print(f'\n[4/5] Computing Morgan fingerprints for dataset '
          f'(n={len(df)}, this takes ~30s)...')
    df['_fp'] = df[smi_col].apply(_to_fp)
    df = df[df['_fp'].notna()].reset_index(drop=True)

    # ---- Identify external hold-out --------------------------------------
    # Exact matches → hold-out (relabelled from known_inhibitors)
    # Near-neighbours (Tanimoto ≥ threshold) → REMOVED entirely
    exact_mask = df['_inchikey'].isin(known_key_set)

    near_mask = np.zeros(len(df), dtype=bool)
    near_info = []
    for i, fp in enumerate(df['_fp']):
        if exact_mask.iloc[i]:
            continue
        max_sim, hit_name = 0.0, ''
        for name, kfp in known_fps:
            s = DataStructs.TanimotoSimilarity(fp, kfp)
            if s > max_sim:
                max_sim, hit_name = s, name
        if max_sim >= args.near_tanimoto:
            near_mask[i] = True
            near_info.append((i, hit_name, max_sim))

    print(f'  Exact matches to known inhibitors: {int(exact_mask.sum())}')
    print(f'  Near-neighbours (Tanimoto ≥ {args.near_tanimoto}): {int(near_mask.sum())}')
    for i, name, s in near_info[:20]:
        print(f'    idx={i}  match={name}  Tanimoto={s:.3f}')
    if len(near_info) > 20:
        print(f'    ... and {len(near_info) - 20} more')

    # Build external hold-out set explicitly from known_inhibitors.csv
    ext = known.copy()
    # add label=1 (known active), and any extra columns needed for consistency
    ext['label'] = 1
    # Match column layout of main dataset if possible
    # Just ensure smiles column name matches
    if smi_col != ksmi_col:
        ext[smi_col] = ext[ksmi_col]
    keep_cols = [c for c in df.columns
                 if c not in ('_inchikey', '_fp', '_source_split')]
    for c in keep_cols:
        if c not in ext.columns:
            ext[c] = ''
    ext['label'] = 1
    ext['_source'] = 'external_holdout'
    ext = ext[keep_cols + ['_source']]

    # Remove exact + near from working df
    remove_mask = exact_mask | pd.Series(near_mask)
    df_clean = df[~remove_mask].reset_index(drop=True)
    print(f'  Working set after removal: n={len(df_clean)}')
    print(f'  Working set positive rate: {df_clean["label"].astype(int).mean():.3f}')

    # ---- Scaffold split --------------------------------------------------
    print(f'\n[5/5] Running Bemis-Murcko scaffold split '
          f'(val={args.val_frac}, test={args.test_frac}, seed={args.seed})...')
    train_idx, val_idx, test_idx, scaffold_map = scaffold_split(
        df_clean[smi_col].tolist(),
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    print(f'  Split sizes:  train={len(train_idx)}  '
          f'val={len(val_idx)}  test={len(test_idx)}')
    print(f'  n_unique_scaffolds = {len(scaffold_map)}')

    # ---- Write outputs ---------------------------------------------------
    write_cols = [c for c in df_clean.columns
                  if c not in ('_inchikey', '_fp', '_source_split')]
    train_df = df_clean.iloc[train_idx][write_cols].reset_index(drop=True)
    val_df = df_clean.iloc[val_idx][write_cols].reset_index(drop=True)
    test_df = df_clean.iloc[test_idx][write_cols].reset_index(drop=True)

    train_df.to_csv(out / 'train.csv', index=False)
    val_df.to_csv(out / 'val.csv', index=False)
    test_df.to_csv(out / 'test.csv', index=False)
    ext.to_csv(out / 'external_holdout.csv', index=False)

    # ---- Report ----------------------------------------------------------
    report = {
        'strategy': 'Bemis-Murcko scaffold split + explicit external hold-out',
        'source': str(inp),
        'output': str(out),
        'near_tanimoto_threshold': args.near_tanimoto,
        'seed': args.seed,
        'n_input_total': int(len(df)),
        'n_removed_exact_match': int(exact_mask.sum()),
        'n_removed_near_neighbour': int(near_mask.sum()),
        'n_working': int(len(df_clean)),
        'n_train': int(len(train_idx)),
        'n_val': int(len(val_idx)),
        'n_test': int(len(test_idx)),
        'n_external_holdout': int(len(ext)),
        'n_unique_scaffolds_in_working': int(len(scaffold_map)),
        'pos_rate_train': float(train_df['label'].astype(int).mean()),
        'pos_rate_val': float(val_df['label'].astype(int).mean()),
        'pos_rate_test': float(test_df['label'].astype(int).mean()),
        'near_neighbours_removed': [
            {'idx': int(i), 'hit_known_inhibitor': name,
             'tanimoto': float(s)} for i, name, s in near_info
        ],
    }
    (out / 'split_report.json').write_text(json.dumps(report, indent=2))
    txt = [
        f'Scaffold-split rebuild report',
        f'=============================',
        f'Source:     {inp}',
        f'Output:     {out}',
        f'Seed:       {args.seed}',
        f'',
        f'Input total: {len(df)} unique molecules',
        f'',
        f'External hold-out (moved out):',
        f'  Exact matches to known inhibitors: {int(exact_mask.sum())}',
        f'  Near-neighbours (Tanimoto ≥ {args.near_tanimoto}): {int(near_mask.sum())}',
        f'  External hold-out CSV rows: {len(ext)}',
        f'',
        f'Working set: {len(df_clean)} molecules, {len(scaffold_map)} scaffolds',
        f'  train: {len(train_idx)} (pos rate {train_df["label"].astype(int).mean():.3f})',
        f'  val:   {len(val_idx)} (pos rate {val_df["label"].astype(int).mean():.3f})',
        f'  test:  {len(test_idx)} (pos rate {test_df["label"].astype(int).mean():.3f})',
    ]
    (out / 'split_report.txt').write_text('\n'.join(txt))
    print('\n' + '\n'.join(txt))
    print(f'\n✓ Wrote 4 CSVs + split_report.{{json,txt}} to {out}/')


if __name__ == '__main__':
    main()
