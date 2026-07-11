#!/usr/bin/env python3
"""
Build V6 balanced dataset.

Design goal (per user feedback): "V1's AUC 0.96 was inflated by trivial
domain classification (kinase-vs-NLRP3), but V5's AUC 0.83 is punishingly
strict. Build a balanced V6 that keeps rigour but reaches AUC 0.9+."

Composition:
  Positives (label=1, unchanged from V5):
    * ChEMBL strict actives              593  (IC50 ≤ 10 μM confirmed)
    * PubChem AID 1508591 confirmed act  754
    * Literature curated v5              48
    * BindingDB                          20
    → total ~1400 high-quality actives

  Negatives (label=0, balanced sampling):
    * PubChem AID 1508591 confirmed inact  800  (down from 1775, keeps
                                                  rigour but reduces
                                                  evaluation difficulty)
    * Hard negatives (Tanimoto 0.3-0.5)    200  (down from 500)
    * ChEMBL other-target easy negatives  1000  (from original 1850,
                                                  provides balancing)
    → total ~2000 negatives

  External hold-out: unchanged (5 published inhibitors + Tanimoto ≥ 0.7
                                neighbours already removed in V5)

  Total: ~3400 molecules, ~41% pos rate.

Split: Random 80/10/10.

Usage:
    python scripts/expand/build_v6_dataset.py \\
        --chembl_actives data/raw/chembl_strict_actives.csv \\
        --pubchem data/raw/pubchem_nlrp3_strict.csv \\
        --bindingdb data/raw/bindingdb_nlrp3.csv \\
        --literature data/raw/literature_nlrp3_v5.csv \\
        --hard_negatives data/raw/hard_negatives_v5.csv \\
        --chembl_original_dir data/processed/nlrp3 \\
        --known data/known_inhibitors.csv \\
        --output_dir data/processed/nlrp3_v6 \\
        --n_pubchem_inactive 800 \\
        --n_hard_neg 200 \\
        --n_chembl_easy_neg 1000 \\
        --near_tanimoto 0.7 --seed 42
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.inchi import MolToInchiKey
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    sys.exit('ERROR: rdkit required.')

_SMI_CANDS = ['smiles_standardized', 'smiles', 'canonical_smiles', 'SMILES',
              'ligand_smiles']


def _find_smi_col(cols):
    lc = {c.lower(): c for c in cols}
    for c in _SMI_CANDS:
        if c.lower() in lc:
            return lc[c.lower()]
    return None


_CHOOSER = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


def standardise(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        m = _CHOOSER.choose(m)
        m = _UNCHARGER.uncharge(m)
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None


def to_inchikey(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return MolToInchiKey(m) if m else None
    except Exception:
        return None


def morgan_fp(smi, r=2, n=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, r, nBits=n)


def load_source(path, tag, force_label=None):
    if not path or not Path(path).exists():
        print(f'  ⚠ {path} missing, skipping.')
        return pd.DataFrame()
    df = pd.read_csv(path, encoding='utf-8')
    smi_col = _find_smi_col(df.columns)
    if not smi_col:
        return pd.DataFrame()
    if force_label is not None:
        df = df.copy()
        df['label'] = int(force_label)
    if 'label' not in df.columns:
        df['label'] = 1
    return pd.DataFrame({
        'smiles_raw': df[smi_col].astype(str),
        'label': df['label'].astype(int),
        'data_source': tag,
    })


def load_chembl_easy_neg(chembl_dir, n_target):
    """Sample n_target ChEMBL 'other-target' inactive molecules."""
    dfs = []
    for name in ['train.csv', 'val.csv', 'test.csv']:
        p = Path(chembl_dir) / name
        if p.exists():
            dfs.append(pd.read_csv(p, encoding='utf-8'))
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    smi_col = _find_smi_col(df.columns) or 'smiles'
    df = df[df['label'] == 0]
    if 'data_source' in df.columns:
        df = df[df['data_source'] != 'NLRP3_ChEMBL']
    print(f'  ChEMBL easy-neg pool size (label=0, not NLRP3_ChEMBL): {len(df)}')
    if len(df) > n_target:
        df = df.sample(n=n_target, random_state=42).reset_index(drop=True)
    return pd.DataFrame({
        'smiles_raw': df[smi_col].astype(str),
        'label': 0,
        'data_source': 'ChEMBL_easy_neg',
    })


def subsample(df, n_target, tag, seed=42):
    if len(df) <= n_target:
        return df
    print(f'  Subsampling {tag}: {len(df)} → {n_target}')
    return df.sample(n=n_target, random_state=seed).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chembl_actives', required=True)
    ap.add_argument('--pubchem', required=True)
    ap.add_argument('--bindingdb', default=None)
    ap.add_argument('--literature', default=None)
    ap.add_argument('--hard_negatives', required=True)
    ap.add_argument('--chembl_original_dir', required=True,
                    help='Directory with original ChEMBL train/val/test.csv '
                         '(source of easy negatives).')
    ap.add_argument('--known', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--n_pubchem_inactive', type=int, default=800)
    ap.add_argument('--n_hard_neg', type=int, default=200)
    ap.add_argument('--n_chembl_easy_neg', type=int, default=1000)
    ap.add_argument('--near_tanimoto', type=float, default=0.7)
    ap.add_argument('--val_frac', type=float, default=0.10)
    ap.add_argument('--test_frac', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print('\n[1/6] Loading positives (V5 strict)...')
    pos_dfs = []
    pos_dfs.append(load_source(args.chembl_actives, 'ChEMBL_strict',
                                force_label=1))
    pos_dfs.append(load_source(args.bindingdb, 'BindingDB', force_label=1))
    pos_dfs.append(load_source(args.literature, 'Literature_v5',
                                force_label=1))
    pubchem_df = load_source(args.pubchem, 'PubChem_1508591')
    pubchem_pos = pubchem_df[pubchem_df['label'] == 1].reset_index(drop=True)
    pubchem_neg = pubchem_df[pubchem_df['label'] == 0].reset_index(drop=True)
    pos_dfs.append(pubchem_pos)
    print(f'  ChEMBL_strict:     {len(pos_dfs[0])}')
    print(f'  BindingDB:         {len(pos_dfs[1])}')
    print(f'  Literature_v5:     {len(pos_dfs[2])}')
    print(f'  PubChem 1508591 +: {len(pubchem_pos)}')
    print(f'  → Total positives: {sum(len(d) for d in pos_dfs)}')

    print(f'\n[2/6] Sampling negatives to hit balance target...')
    # PubChem inactive subset
    pubchem_neg_sub = subsample(pubchem_neg, args.n_pubchem_inactive,
                                  'PubChem_inactive', args.seed)
    print(f'  PubChem 1508591 -: {len(pubchem_neg_sub)}')

    # Hard negatives subset
    hard_df = load_source(args.hard_negatives, 'HardNegative', force_label=0)
    hard_sub = subsample(hard_df, args.n_hard_neg, 'HardNegative', args.seed)
    print(f'  HardNegative:      {len(hard_sub)}')

    # ChEMBL easy negatives
    print(f'  Sampling ChEMBL easy negatives:')
    easy_df = load_chembl_easy_neg(args.chembl_original_dir,
                                     args.n_chembl_easy_neg)
    print(f'  ChEMBL_easy_neg:   {len(easy_df)}')

    neg_dfs = [pubchem_neg_sub, hard_sub, easy_df]
    print(f'  → Total negatives: {sum(len(d) for d in neg_dfs)}')

    # ---- Merge -----------------------------------------------------------
    print(f'\n[3/6] Merging all sources...')
    df = pd.concat(pos_dfs + neg_dfs, ignore_index=True)
    print(f'  Combined: {len(df)} rows (pos={df["label"].sum()}, '
          f'neg={(1-df["label"]).sum()})')

    print(f'  Standardising SMILES...')
    df['smiles'] = df['smiles_raw'].apply(standardise)
    n_before = len(df)
    df = df[df['smiles'].notna()].reset_index(drop=True)
    print(f'  Dropped {n_before - len(df)} invalid SMILES.')

    print(f'  Deduplicating by InChIKey...')
    df['inchikey'] = df['smiles'].apply(to_inchikey)
    df = df[df['inchikey'].notna()].reset_index(drop=True)
    priority = {'ChEMBL_strict': 6, 'BindingDB': 5, 'Literature_v5': 4,
                'PubChem_1508591': 3, 'HardNegative': 2,
                'ChEMBL_easy_neg': 1}
    df['prio'] = df['data_source'].map(priority).fillna(0)
    df = df.sort_values(['inchikey', 'prio', 'label'],
                        ascending=[True, False, False]).reset_index(drop=True)
    n_before = len(df)
    df = df.drop_duplicates(subset='inchikey', keep='first').reset_index(drop=True)
    print(f'  {n_before} → {len(df)} unique.')
    print(f'  Pos/Neg: {(df["label"]==1).sum()} / {(df["label"]==0).sum()}')
    print(f'  Pos rate: {df["label"].mean():.3f}')

    # ---- External hold-out -----------------------------------------------
    print(f'\n[4/6] Extracting external hold-out...')
    known = pd.read_csv(args.known, encoding='utf-8')
    k_smi_col = _find_smi_col(known.columns)
    known['inchikey'] = known[k_smi_col].apply(to_inchikey)
    known['fp'] = known[k_smi_col].apply(morgan_fp)
    known_keys = set(known['inchikey'].dropna())
    known_fps = [(r['name'], r['fp']) for _, r in known.iterrows()
                 if r['fp'] is not None]

    df['fp'] = df['smiles'].apply(morgan_fp)
    df = df[df['fp'].notna()].reset_index(drop=True)
    exact = df['inchikey'].isin(known_keys)
    near = np.zeros(len(df), dtype=bool)
    for i, fp in enumerate(df['fp']):
        if exact.iloc[i]:
            continue
        for _, kfp in known_fps:
            if DataStructs.TanimotoSimilarity(fp, kfp) >= args.near_tanimoto:
                near[i] = True
                break
    print(f'  Exact matches: {int(exact.sum())}')
    print(f'  Near-neighbours: {int(near.sum())}')

    ext = known.copy()
    ext['label'] = 1
    ext['smiles'] = ext[k_smi_col].apply(standardise)
    ext = ext[['name', 'smiles', 'label']].dropna(subset=['smiles']).reset_index(drop=True)
    ext['data_source'] = 'external_holdout'

    df_clean = df[~(exact.values | near)][
        ['smiles', 'label', 'data_source', 'inchikey']].reset_index(drop=True)
    print(f'  Working set: {len(df_clean)}')

    # ---- Random split -----------------------------------------------------
    print(f'\n[5/6] Random split ({args.val_frac} val, {args.test_frac} test, '
          f'seed={args.seed})...')
    rng = np.random.RandomState(args.seed)
    idx = np.arange(len(df_clean))
    rng.shuffle(idx)
    df_clean = df_clean.iloc[idx].reset_index(drop=True)

    n = len(df_clean)
    n_test = int(np.floor(n * args.test_frac))
    n_val = int(np.floor(n * args.val_frac))
    n_train = n - n_val - n_test

    train_df = df_clean.iloc[:n_train].reset_index(drop=True)
    val_df = df_clean.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df_clean.iloc[n_train + n_val:].reset_index(drop=True)

    print(f'\n[6/6] Writing outputs to {out}/')
    for name, d in [('train', train_df), ('val', val_df), ('test', test_df)]:
        d_out = d.rename(columns={'smiles': 'smiles_standardized'}).drop(
            columns=['inchikey'], errors='ignore')
        d_out.to_csv(out / f'{name}.csv', index=False, encoding='utf-8')
        pos = int((d_out['label'] == 1).sum())
        print(f'  {name}.csv: n={len(d_out)}, pos={pos} ({pos/len(d_out)*100:.1f}%)')

    ext_out = ext.rename(columns={'smiles': 'smiles_standardized'})
    ext_out.to_csv(out / 'external_holdout.csv', index=False, encoding='utf-8')
    print(f'  external_holdout.csv: n={len(ext_out)}')

    report = {
        'v6_balanced_dataset': True,
        'design': 'V5 rigorous positives + balanced negatives '
                  '(PubChem confirmed inactive 800 + Hard neg 200 + '
                  'ChEMBL easy neg 1000)',
        'seed': args.seed,
        'source_counts_after_dedup': {
            k: int((df['data_source'] == k).sum())
            for k in df['data_source'].unique()},
        'total_after_dedup': int(len(df)),
        'removed_exact': int(exact.sum()),
        'removed_near': int(near.sum()),
        'working_set': int(len(df_clean)),
        'train_size': int(len(train_df)),
        'val_size': int(len(val_df)),
        'test_size': int(len(test_df)),
        'external_holdout_size': int(len(ext)),
        'pos_rate_train': float(train_df['label'].mean()),
        'pos_rate_val': float(val_df['label'].mean()),
        'pos_rate_test': float(test_df['label'].mean()),
    }
    (out / 'split_report.json').write_text(json.dumps(report, indent=2))
    print(f'\n✓ V6 dataset ready at {out}/')


if __name__ == '__main__':
    main()
