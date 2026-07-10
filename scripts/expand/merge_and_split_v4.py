#!/usr/bin/env python3
"""
Merge all data sources into a single V4 NLRP3 dataset, then produce a
leakage-free scaffold split.

Data sources (merged in order — later duplicates dropped):
  1. ChEMBL (data/processed/nlrp3/{train,val,test}.csv)
  2. BindingDB (data/raw/bindingdb_nlrp3.csv)
  3. PubChem BioAssay (data/raw/pubchem_nlrp3.csv)
  4. Literature curated (data/raw/literature_nlrp3_curated.csv)
  5. DUD-E decoys (data/raw/decoys.csv)

Processing:
  * SMILES standardisation via RDKit (canonical, remove salts, largest
    fragment, neutralise charges).
  * Deduplicate by canonical InChIKey (first-seen wins; conflict warns).
  * Move 5 published inhibitors (+ Tanimoto ≥ 0.7 neighbours) to
    external_holdout.csv.
  * Bemis-Murcko scaffold split of the remaining molecules (80/10/10).

Usage:
    python scripts/expand/merge_and_split_v4.py \\
        --chembl_dir data/processed/nlrp3 \\
        --bindingdb  data/raw/bindingdb_nlrp3.csv \\
        --pubchem    data/raw/pubchem_nlrp3.csv \\
        --literature data/raw/literature_nlrp3_curated.csv \\
        --decoys     data/raw/decoys.csv \\
        --known      data/known_inhibitors.csv \\
        --output_dir data/processed/nlrp3_v4 \\
        --near_tanimoto 0.7 --val_frac 0.10 --test_frac 0.10 --seed 42
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


# ---- Standardisation ------------------------------------------------------
_STD_CHOOSER = rdMolStandardize.LargestFragmentChooser()
_STD_UNCHARGER = rdMolStandardize.Uncharger()


def standardise_smiles(smi):
    """Return canonical, largest-fragment, neutral SMILES. None if invalid."""
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        m = _STD_CHOOSER.choose(m)
        m = _STD_UNCHARGER.uncharge(m)
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None


def to_inchikey(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return MolToInchiKey(m) if m else None
    except Exception:
        return None


def morgan_fp(smi, radius=2, n_bits=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)


def murcko_scaffold(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ''
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        return ''


# ---- Loading each source -------------------------------------------------
def load_chembl(chembl_dir):
    rows = []
    for name in ['train.csv', 'val.csv', 'test.csv']:
        p = Path(chembl_dir) / name
        if not p.exists():
            print(f'  ⚠ {p} missing, skipping.')
            continue
        d = pd.read_csv(p)
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    smi_col = _find_smi_col(df.columns) or 'smiles'
    return pd.DataFrame({
        'smiles_raw': df[smi_col].astype(str),
        'label': df['label'].astype(int),
        'data_source': 'ChEMBL',
        'source_meta': df.get('molecule_chembl_id', pd.Series([''] * len(df))).astype(str),
    })


def load_source(path, tag):
    if not path or not Path(path).exists():
        print(f'  ⚠ {path} missing, skipping.')
        return pd.DataFrame()
    df = pd.read_csv(path)
    smi_col = _find_smi_col(df.columns)
    if not smi_col:
        print(f'  ⚠ no SMILES in {path}, skipping.')
        return pd.DataFrame()
    if 'label' not in df.columns:
        # Assume active for curated inhibitors
        df['label'] = 1
    return pd.DataFrame({
        'smiles_raw': df[smi_col].astype(str),
        'label': df['label'].astype(int),
        'data_source': tag,
        'source_meta': df.get('source', pd.Series([''] * len(df))).astype(str),
    })


# ---- Scaffold split (from rebuild script) --------------------------------
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
    train_idx, val_idx, test_idx = [], [], []
    for g in groups:
        if len(train_idx) + len(g) <= n_train:
            train_idx.extend(g)
        elif len(val_idx) + len(g) <= n_val:
            val_idx.extend(g)
        else:
            test_idx.extend(g)
    return train_idx, val_idx, test_idx, scaffold_to_indices


# ---- Main ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chembl_dir', required=True)
    ap.add_argument('--bindingdb', default=None)
    ap.add_argument('--pubchem', default=None)
    ap.add_argument('--literature', default=None)
    ap.add_argument('--decoys', default=None)
    ap.add_argument('--known', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--near_tanimoto', type=float, default=0.7)
    ap.add_argument('--val_frac', type=float, default=0.10)
    ap.add_argument('--test_frac', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load all sources --------------------------------------------
    print('\n[1/6] Loading data sources...')
    dfs = []
    print('  ChEMBL:')
    df_chembl = load_chembl(args.chembl_dir)
    print(f'    → {len(df_chembl)} rows')
    dfs.append(df_chembl)

    print('  BindingDB:')
    df_bdb = load_source(args.bindingdb, 'BindingDB')
    print(f'    → {len(df_bdb)} rows')
    dfs.append(df_bdb)

    print('  PubChem BioAssay:')
    df_pc = load_source(args.pubchem, 'PubChem')
    print(f'    → {len(df_pc)} rows')
    dfs.append(df_pc)

    print('  Literature curated:')
    df_lit = load_source(args.literature, 'Literature_curated')
    print(f'    → {len(df_lit)} rows')
    dfs.append(df_lit)

    print('  DUD-E decoys:')
    df_dec = load_source(args.decoys, 'DUD-E_decoy')
    print(f'    → {len(df_dec)} rows')
    dfs.append(df_dec)

    df = pd.concat([d for d in dfs if len(d) > 0], ignore_index=True)
    print(f'\n  Total combined: {len(df)} rows')

    # ---- 2. Standardise SMILES ------------------------------------------
    print(f'\n[2/6] Standardising SMILES with RDKit...')
    df['smiles'] = df['smiles_raw'].apply(standardise_smiles)
    n_before = len(df)
    df = df[df['smiles'].notna()].reset_index(drop=True)
    print(f'  Dropped {n_before - len(df)} rows with invalid SMILES.')

    # ---- 3. Deduplicate by InChIKey (conflict handling) -----------------
    print(f'\n[3/6] Deduplicating by canonical InChIKey...')
    df['inchikey'] = df['smiles'].apply(to_inchikey)
    df = df[df['inchikey'].notna()].reset_index(drop=True)

    # Conflict handling: if same InChIKey has different labels,
    #  * If ANY source says active → keep as active
    #  * Prioritise sources: ChEMBL > BindingDB > PubChem > Literature > DUD-E
    priority = {'ChEMBL': 5, 'BindingDB': 4, 'PubChem': 3,
                'Literature_curated': 2, 'DUD-E_decoy': 1}
    df['prio'] = df['data_source'].map(priority).fillna(0)
    df = df.sort_values(['inchikey', 'prio', 'label'],
                        ascending=[True, False, False]).reset_index(drop=True)
    n_before = len(df)
    df_dedup = df.drop_duplicates(subset='inchikey', keep='first').reset_index(drop=True)
    print(f'  {n_before} → {len(df_dedup)} rows after InChIKey dedup.')
    print(f'  Label distribution: active={(df_dedup["label"]==1).sum()}  '
          f'inactive={(df_dedup["label"]==0).sum()}  '
          f'pos_rate={df_dedup["label"].mean():.3f}')

    # ---- 4. External hold-out --------------------------------------------
    print(f'\n[4/6] Extracting external hold-out (known + Tanimoto ≥ '
          f'{args.near_tanimoto} neighbours)...')
    known = pd.read_csv(args.known)
    k_smi_col = _find_smi_col(known.columns)
    known['inchikey'] = known[k_smi_col].apply(to_inchikey)
    known['fp'] = known[k_smi_col].apply(morgan_fp)
    known_keys = set(known['inchikey'].dropna().tolist())
    known_fps = [(r['name'], r['fp']) for _, r in known.iterrows()
                 if r['fp'] is not None]

    df_dedup['fp'] = df_dedup['smiles'].apply(morgan_fp)
    df_dedup = df_dedup[df_dedup['fp'].notna()].reset_index(drop=True)

    exact_mask = df_dedup['inchikey'].isin(known_keys)
    near_mask = np.zeros(len(df_dedup), dtype=bool)
    near_info = []
    for i, fp in enumerate(df_dedup['fp']):
        if exact_mask.iloc[i]:
            continue
        max_sim = 0.0
        hit = ''
        for name, kfp in known_fps:
            s = DataStructs.TanimotoSimilarity(fp, kfp)
            if s > max_sim:
                max_sim, hit = s, name
        if max_sim >= args.near_tanimoto:
            near_mask[i] = True
            near_info.append((i, hit, max_sim))
    print(f'  Exact matches:       {int(exact_mask.sum())}')
    print(f'  Near-neighbours (Tanimoto ≥ {args.near_tanimoto}): {int(near_mask.sum())}')
    for i, name, s in near_info[:20]:
        print(f'    idx={i}  match={name}  Tanimoto={s:.3f}')

    # Build external hold-out: 5 known inhibitors (all positive)
    ext = known.copy()
    ext['label'] = 1
    ext['smiles'] = ext[k_smi_col].apply(standardise_smiles)
    ext = ext[['name', 'smiles', 'label']].copy()
    ext['data_source'] = 'external_holdout'
    ext = ext.dropna(subset=['smiles']).reset_index(drop=True)

    remove_mask = exact_mask.values | near_mask
    df_clean = df_dedup[~remove_mask][['smiles', 'label', 'data_source',
                                       'inchikey']].reset_index(drop=True)
    print(f'\n  Working set: {len(df_clean)} molecules')
    print(f'    active:    {(df_clean["label"] == 1).sum()}')
    print(f'    inactive:  {(df_clean["label"] == 0).sum()}')

    # ---- 5. Scaffold split ----------------------------------------------
    print(f'\n[5/6] Running Bemis-Murcko scaffold split '
          f'(val={args.val_frac}, test={args.test_frac}, seed={args.seed})...')
    train_idx, val_idx, test_idx, scf_map = scaffold_split(
        df_clean['smiles'].tolist(),
        val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    print(f'  Sizes:  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}')
    print(f'  n_unique_scaffolds = {len(scf_map)}')

    train_df = df_clean.iloc[train_idx].reset_index(drop=True)
    val_df = df_clean.iloc[val_idx].reset_index(drop=True)
    test_df = df_clean.iloc[test_idx].reset_index(drop=True)

    # ---- 6. Write outputs -----------------------------------------------
    print(f'\n[6/6] Writing outputs to {out}/')
    # Rename smiles → smiles_standardized to match existing pipeline expectations
    for name, d in [('train', train_df), ('val', val_df), ('test', test_df)]:
        d_out = d.rename(columns={'smiles': 'smiles_standardized'}).drop(
            columns=['inchikey'], errors='ignore')
        d_out.to_csv(out / f'{name}.csv', index=False)
        print(f'  {name}.csv: n={len(d_out)}, pos={(d_out["label"]==1).sum()}')

    ext_out = ext.rename(columns={'smiles': 'smiles_standardized'})
    ext_out.to_csv(out / 'external_holdout.csv', index=False)
    print(f'  external_holdout.csv: n={len(ext_out)}')

    report = {
        'v4_dataset_report': True,
        'seed': args.seed,
        'sources': {
            'ChEMBL': int((df_dedup['data_source'] == 'ChEMBL').sum()),
            'BindingDB': int((df_dedup['data_source'] == 'BindingDB').sum()),
            'PubChem': int((df_dedup['data_source'] == 'PubChem').sum()),
            'Literature_curated': int((df_dedup['data_source'] == 'Literature_curated').sum()),
            'DUD-E_decoy': int((df_dedup['data_source'] == 'DUD-E_decoy').sum()),
        },
        'combined_after_dedup': int(len(df_dedup)),
        'removed_exact_matches': int(exact_mask.sum()),
        'removed_near_neighbours': int(near_mask.sum()),
        'working_set': int(len(df_clean)),
        'train_size': int(len(train_idx)),
        'val_size': int(len(val_idx)),
        'test_size': int(len(test_idx)),
        'external_holdout_size': int(len(ext)),
        'num_scaffolds': int(len(scf_map)),
        'pos_rate_train': float(train_df['label'].mean()),
        'pos_rate_val': float(val_df['label'].mean()),
        'pos_rate_test': float(test_df['label'].mean()),
        'near_neighbours_removed': [
            {'idx': int(i), 'known': n, 'tanimoto': float(s)}
            for i, n, s in near_info
        ],
    }
    (out / 'split_report.json').write_text(json.dumps(report, indent=2))
    print(f'\n✓ Full report → {out}/split_report.json')


if __name__ == '__main__':
    main()
