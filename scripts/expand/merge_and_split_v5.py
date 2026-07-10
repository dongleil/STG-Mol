#!/usr/bin/env python3
"""
V5 strict merge — build a leakage-free, high-quality NLRP3 dataset.

Difference from merge_and_split_v4.py: this script does NOT read
data/processed/nlrp3/{train,val,test}.csv directly (which contains
1850 "other-target" molecules used as easy negatives). Instead it
consumes only quality-filtered CSVs:

  Positives (label=1):
    * chembl_strict_actives.csv    (IC50 ≤ 10 μM confirmed)
    * bindingdb_nlrp3.csv          (IC50 ≤ 10 μM confirmed)
    * pubchem_nlrp3_strict.csv     (AID 1508591 confirmed active only)
    * literature_nlrp3_v5.csv      (IC50 filter applied upstream)

  Negatives (label=0):
    * pubchem_nlrp3_strict.csv     (AID 1508591 confirmed inactive)
    * hard_negatives_v5.csv        (Tanimoto 0.3-0.5 mined negatives)
    * (Optional) decoys_v5.csv     (DUD-E-style, only if needed to balance)

Standardises SMILES → dedups by InChIKey → moves known inhibitors +
Tanimoto ≥ 0.7 neighbours to external_holdout → Bemis-Murcko scaffold
split of the remainder.

Usage:
    python scripts/expand/merge_and_split_v5.py \\
        --chembl_actives data/raw/chembl_strict_actives.csv \\
        --pubchem        data/raw/pubchem_nlrp3_strict.csv \\
        --bindingdb      data/raw/bindingdb_nlrp3.csv \\
        --literature     data/raw/literature_nlrp3_v5.csv \\
        --hard_negatives data/raw/hard_negatives_v5.csv \\
        --known          data/known_inhibitors.csv \\
        --output_dir     data/processed/nlrp3_v5 \\
        --near_tanimoto  0.7 --val_frac 0.10 --test_frac 0.10 --seed 42
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
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    sys.exit('ERROR: rdkit is required.')

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


def standardise_smiles(smi):
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


def murcko_scaffold(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m,
                                                    includeChirality=False) if m else ''
    except Exception:
        return ''


def load_source(path, tag, force_label=None):
    """
    Load a single-source CSV. If force_label is not None, overrides the
    label column to that value (used for actives-only or negatives-only
    inputs). Otherwise, respects the existing 'label' column.
    """
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


def scaffold_split(smiles_list, val_frac=0.10, test_frac=0.10, seed=42):
    scaffold_to_indices = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaffold_to_indices[murcko_scaffold(smi)].append(i)
    n = len(smiles_list)
    n_val = int(np.floor(n * val_frac))
    n_test = int(np.floor(n * test_frac))
    n_train = n - n_val - n_test
    rng = np.random.RandomState(seed)
    groups = sorted(scaffold_to_indices.values(),
                    key=lambda g: (len(g), rng.random()),
                    reverse=True)
    tr, va, te = [], [], []
    for g in groups:
        if len(tr) + len(g) <= n_train:
            tr.extend(g)
        elif len(va) + len(g) <= n_val:
            va.extend(g)
        else:
            te.extend(g)
    return tr, va, te, scaffold_to_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chembl_actives', required=True,
                    help='Actives-only ChEMBL CSV from curate_chembl_strict.py.')
    ap.add_argument('--pubchem', default=None,
                    help='PubChem AID 1508591 strict CSV (both label=0/1).')
    ap.add_argument('--bindingdb', default=None)
    ap.add_argument('--literature', default=None)
    ap.add_argument('--hard_negatives', default=None)
    ap.add_argument('--decoys', default=None,
                    help='(Optional) DUD-E decoys CSV. Used only to balance '
                         'class ratio if too few negatives.')
    ap.add_argument('--known', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--near_tanimoto', type=float, default=0.7)
    ap.add_argument('--val_frac', type=float, default=0.10)
    ap.add_argument('--test_frac', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print('\n[1/6] Loading strict-quality sources...')
    dfs = []
    print('  ChEMBL strict actives:')
    dfs.append(load_source(args.chembl_actives, 'ChEMBL_strict', force_label=1))
    print(f'    → {len(dfs[-1])} rows')
    print('  BindingDB actives:')
    dfs.append(load_source(args.bindingdb, 'BindingDB', force_label=1))
    print(f'    → {len(dfs[-1])} rows')
    print('  Literature curated:')
    dfs.append(load_source(args.literature, 'Literature_v5', force_label=1))
    print(f'    → {len(dfs[-1])} rows')
    print('  PubChem AID 1508591 (both active + inactive):')
    dfs.append(load_source(args.pubchem, 'PubChem_1508591'))
    print(f'    → {len(dfs[-1])} rows '
          f'({(dfs[-1]["label"]==1).sum() if len(dfs[-1]) else 0} active, '
          f'{(dfs[-1]["label"]==0).sum() if len(dfs[-1]) else 0} inactive)')
    print('  Hard negatives:')
    dfs.append(load_source(args.hard_negatives, 'HardNegative', force_label=0))
    print(f'    → {len(dfs[-1])} rows')
    print('  DUD-E decoys (optional):')
    dfs.append(load_source(args.decoys, 'DUD-E_decoy', force_label=0))
    print(f'    → {len(dfs[-1])} rows')

    df = pd.concat([d for d in dfs if len(d) > 0], ignore_index=True)
    print(f'\n  Total combined: {len(df)} rows')

    print(f'\n[2/6] Standardising SMILES with RDKit...')
    df['smiles'] = df['smiles_raw'].apply(standardise_smiles)
    n_before = len(df)
    df = df[df['smiles'].notna()].reset_index(drop=True)
    print(f'  Dropped {n_before - len(df)} rows with invalid SMILES.')

    print(f'\n[3/6] Deduplicating by canonical InChIKey...')
    df['inchikey'] = df['smiles'].apply(to_inchikey)
    df = df[df['inchikey'].notna()].reset_index(drop=True)
    priority = {'ChEMBL_strict': 5, 'BindingDB': 4, 'Literature_v5': 3,
                'PubChem_1508591': 2, 'HardNegative': 1, 'DUD-E_decoy': 0}
    df['prio'] = df['data_source'].map(priority).fillna(0)
    # If conflict: keep highest priority AND prefer label=1 tie-break
    df = df.sort_values(['inchikey', 'prio', 'label'],
                        ascending=[True, False, False]).reset_index(drop=True)
    n_before = len(df)
    df = df.drop_duplicates(subset='inchikey', keep='first').reset_index(drop=True)
    print(f'  {n_before} → {len(df)} unique molecules.')
    print(f'  Active/Inactive: {(df["label"]==1).sum()} / {(df["label"]==0).sum()}')
    print(f'  Pos rate: {df["label"].mean():.3f}')

    print(f'\n[4/6] Extracting external hold-out (known + Tanimoto ≥ '
          f'{args.near_tanimoto} neighbours)...')
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
    near_info = []
    for i, fp in enumerate(df['fp']):
        if exact.iloc[i]:
            continue
        max_s, hit = 0.0, ''
        for name, kfp in known_fps:
            s = DataStructs.TanimotoSimilarity(fp, kfp)
            if s > max_s:
                max_s, hit = s, name
        if max_s >= args.near_tanimoto:
            near[i] = True
            near_info.append((i, hit, max_s))
    print(f'  Exact matches:       {int(exact.sum())}')
    print(f'  Near-neighbours (Tanimoto ≥ {args.near_tanimoto}): {int(near.sum())}')

    ext = known.copy()
    ext['label'] = 1
    ext['smiles'] = ext[k_smi_col].apply(standardise_smiles)
    ext = ext[['name', 'smiles', 'label']].dropna(subset=['smiles']).reset_index(drop=True)
    ext['data_source'] = 'external_holdout'

    df_clean = df[~(exact.values | near)][
        ['smiles', 'label', 'data_source', 'inchikey']].reset_index(drop=True)
    print(f'\n  Working set: {len(df_clean)}  '
          f'(active={(df_clean["label"]==1).sum()}, '
          f'inactive={(df_clean["label"]==0).sum()})')

    print(f'\n[5/6] Bemis-Murcko scaffold split '
          f'(val={args.val_frac}, test={args.test_frac}, seed={args.seed})...')
    tr, va, te, scf = scaffold_split(df_clean['smiles'].tolist(),
                                     args.val_frac, args.test_frac, args.seed)
    print(f'  Sizes:  train={len(tr)}  val={len(va)}  test={len(te)}')
    print(f'  n_unique_scaffolds = {len(scf)}')

    train_df = df_clean.iloc[tr].reset_index(drop=True)
    val_df = df_clean.iloc[va].reset_index(drop=True)
    test_df = df_clean.iloc[te].reset_index(drop=True)

    print(f'\n[6/6] Writing outputs to {out}/')
    for name, d in [('train', train_df), ('val', val_df), ('test', test_df)]:
        d_out = d.rename(columns={'smiles': 'smiles_standardized'}).drop(
            columns=['inchikey'], errors='ignore')
        d_out.to_csv(out / f'{name}.csv', index=False, encoding='utf-8')
        pos = (d_out['label']==1).sum()
        print(f'  {name}.csv: n={len(d_out)}, pos={pos} ({pos/len(d_out)*100:.1f}%)')

    ext_out = ext.rename(columns={'smiles': 'smiles_standardized'})
    ext_out.to_csv(out / 'external_holdout.csv', index=False, encoding='utf-8')
    print(f'  external_holdout.csv: n={len(ext_out)}')

    report = {
        'v5_strict_dataset': True,
        'seed': args.seed,
        'source_counts_after_dedup': {
            k: int((df['data_source'] == k).sum())
            for k in df['data_source'].unique()
        },
        'total_after_dedup': int(len(df)),
        'removed_exact': int(exact.sum()),
        'removed_near': int(near.sum()),
        'working_set': int(len(df_clean)),
        'train_size': int(len(tr)),
        'val_size': int(len(va)),
        'test_size': int(len(te)),
        'external_holdout_size': int(len(ext)),
        'num_scaffolds': int(len(scf)),
        'pos_rate_train': float(train_df['label'].mean()),
        'pos_rate_val': float(val_df['label'].mean()),
        'pos_rate_test': float(test_df['label'].mean()),
        'near_neighbours_removed': [
            {'idx': int(i), 'known': n, 'tanimoto': float(s)}
            for i, n, s in near_info],
    }
    (out / 'split_report.json').write_text(json.dumps(report, indent=2))
    print(f'\n✓ Full report → {out}/split_report.json')


if __name__ == '__main__':
    main()
