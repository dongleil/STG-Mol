#!/usr/bin/env python3
"""
Find hard negatives from a pool of PubChem inactives / ChEMBL molecules.

Definition (per QSAR literature):
  Hard negatives = molecules that are structurally similar to NLRP3
  actives (Tanimoto 0.3 - 0.5) but have been experimentally confirmed
  inactive against NLRP3 (or measured on a different, unrelated target).

Why hard negatives matter:
  * Trivial negatives (very different structure) are easy for the model
    to reject → model relies on gross features, not fine SAR.
  * Hard negatives force the model to learn subtle activity-determining
    features → better generalisation to novel scaffolds.

Selection criteria:
  * SMILES parseable by RDKit
  * Not in the actives set (canonical InChIKey mismatch)
  * Tanimoto to nearest active in [--tanimoto_lower, --tanimoto_upper]
  * Optionally: not similar (Tanimoto < 0.7) to any already-selected
    hard negative → maximise diversity within the hard-negative set

Usage:
    python scripts/expand/find_hard_negatives.py \\
        --actives_csv data/raw/all_actives.csv \\
        --pool_csv    data/raw/pubchem_nlrp3.csv \\
        --n_target    500 \\
        --tanimoto_lower 0.3 --tanimoto_upper 0.5 \\
        --output data/raw/hard_negatives.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.inchi import MolToInchiKey
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


def to_fp(smi, r=2, n=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, r, nBits=n)


def to_inchikey(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return MolToInchiKey(m) if m else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--actives_csv', required=True,
                    help='CSV of NLRP3 actives (used to compute Tanimoto).')
    ap.add_argument('--pool_csv', required=True,
                    help='Pool of candidate hard-negative SMILES.')
    ap.add_argument('--n_target', type=int, default=500,
                    help='Max number of hard negatives to select.')
    ap.add_argument('--tanimoto_lower', type=float, default=0.3)
    ap.add_argument('--tanimoto_upper', type=float, default=0.5)
    ap.add_argument('--dedup_tanimoto', type=float, default=0.7,
                    help='Reject candidates that are already similar '
                         '(≥ this Tanimoto) to a previously-selected hard '
                         'negative. Ensures within-set diversity.')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ---- Load actives -------------------------------------------------
    print(f'\nLoading actives from {args.actives_csv}...')
    a_df = pd.read_csv(args.actives_csv, encoding='utf-8')
    a_smi_col = _find_smi_col(a_df.columns)
    if not a_smi_col:
        sys.exit('No SMILES column in actives CSV.')
    if 'label' in a_df.columns:
        a_df = a_df[a_df['label'] == 1].reset_index(drop=True)
    print(f'  {len(a_df)} actives.')

    active_keys, active_fps = set(), []
    for smi in a_df[a_smi_col].astype(str):
        k = to_inchikey(smi)
        fp = to_fp(smi)
        if k and fp:
            active_keys.add(k)
            active_fps.append(fp)
    print(f'  {len(active_fps)} valid Morgan FPs for actives.')

    # ---- Load pool ---------------------------------------------------
    print(f'\nLoading candidate pool from {args.pool_csv}...')
    p_df = pd.read_csv(args.pool_csv, encoding='utf-8')
    p_smi_col = _find_smi_col(p_df.columns)
    if not p_smi_col:
        sys.exit('No SMILES column in pool CSV.')
    # Prefer confirmed inactives (label=0) if labels present
    if 'label' in p_df.columns:
        n_before = len(p_df)
        p_df = p_df[p_df['label'] == 0].reset_index(drop=True)
        print(f'  Filtered to label=0: {n_before} → {len(p_df)} molecules.')

    print('  Computing FPs + InChIKeys for pool (this may take a minute)...')
    pool = []
    for i, smi in enumerate(p_df[p_smi_col].astype(str)):
        fp = to_fp(smi)
        k = to_inchikey(smi)
        if fp and k and k not in active_keys:
            pool.append({'smiles': smi, 'fp': fp, 'inchikey': k})
        if (i + 1) % 500 == 0:
            print(f'    {i+1}/{len(p_df)}')
    print(f'  {len(pool)} valid candidates in pool.')

    # ---- Find hard negatives ------------------------------------------
    print(f'\nFinding hard negatives (Tanimoto ∈ '
          f'[{args.tanimoto_lower}, {args.tanimoto_upper}])...')
    selected = []
    selected_fps = []

    # Randomise pool order for diversity (deterministic seed)
    rng = np.random.RandomState(42)
    rng.shuffle(pool)

    for cand in pool:
        # Nearest-neighbour Tanimoto to actives
        max_a = 0.0
        for afp in active_fps:
            s = DataStructs.TanimotoSimilarity(cand['fp'], afp)
            if s > max_a:
                max_a = s
                if max_a > args.tanimoto_upper:
                    break
        if not (args.tanimoto_lower <= max_a <= args.tanimoto_upper):
            continue
        # Within-set diversity check
        if selected_fps:
            max_sel = max(
                DataStructs.TanimotoSimilarity(cand['fp'], sfp)
                for sfp in selected_fps
            )
            if max_sel >= args.dedup_tanimoto:
                continue
        selected.append({
            'smiles': cand['smiles'],
            'label': 0,
            'nearest_active_tanimoto': round(max_a, 3),
            'source': 'PubChem_hard_negative',
            'data_source': 'HardNegative',
            'source_target': 'NLRP3',
        })
        selected_fps.append(cand['fp'])
        if len(selected) >= args.n_target:
            break
        if len(selected) % 50 == 0:
            print(f'  Selected {len(selected)}/{args.n_target}')

    # ---- Write -------------------------------------------------------
    out = pd.DataFrame(selected)
    out.to_csv(args.output, index=False, encoding='utf-8')
    print(f'\n✓ Wrote {len(selected)} hard negatives to {args.output}')
    if selected:
        tanimoto_vals = out['nearest_active_tanimoto'].values
        print(f'  Nearest-active Tanimoto: '
              f'min={tanimoto_vals.min():.3f}  '
              f'mean={tanimoto_vals.mean():.3f}  '
              f'max={tanimoto_vals.max():.3f}')


if __name__ == '__main__':
    main()
