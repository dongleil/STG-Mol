#!/usr/bin/env python3
"""
Generate DUD-E-style property-matched decoys for active molecules.

DUD-E (Directory of Useful Decoys, Enhanced; Mysinger et al. 2012) is
the gold-standard virtual-screening benchmark protocol. For each
active, we generate N decoys that:

  * Match active in physicochemical properties (property matching):
      - Molecular weight        (± 20 Da)
      - LogP                    (± 1.0 log units)
      - Number of rotatable bonds (± 2)
      - Number of H-bond donors   (± 1)
      - Number of H-bond acceptors (± 1)
      - Net formal charge        (exact match)
  * Are structurally dissimilar from ALL actives (Tanimoto < 0.4)

This yields negatives that are hard for a QSAR model to distinguish
by property alone (forces the model to learn structural features).

We draw decoys from a large chemical library (default: ChEMBL
drug-like subset provided as SMILES). If no library is provided,
we sample from the existing training set's inactive molecules —
which is sub-optimal but still better than random.

Usage:
    python scripts/expand/generate_decoys.py \\
        --actives_csv data/raw/all_actives.csv \\
        --library_csv data/processed/nlrp3/train.csv \\
        --n_decoys_per_active 30 \\
        --output data/raw/decoys.csv
"""
import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Lipinski, Crippen, DataStructs
    from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    sys.exit('ERROR: rdkit is required.')


_SMI_CANDS = ['smiles', 'smiles_standardized', 'canonical_smiles', 'SMILES',
              'ligand_smiles']


def _find_smi_col(cols):
    lc = {c.lower(): c for c in cols}
    for c in _SMI_CANDS:
        if c.lower() in lc:
            return lc[c.lower()]
    return None


def compute_properties(smi):
    """Return (MW, logP, HBD, HBA, RotB, net_charge) or None."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        mw = Descriptors.MolWt(m)
        logp = Crippen.MolLogP(m)
        hbd = Lipinski.NumHDonors(m)
        hba = Lipinski.NumHAcceptors(m)
        rotb = CalcNumRotatableBonds(m)
        charge = Chem.GetFormalCharge(m)
        return (mw, logp, hbd, hba, rotb, charge)
    except Exception:
        return None


def morgan_fp(smi, radius=2, n_bits=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)


def property_match(props_active, props_cand,
                   mw_tol=20, logp_tol=1.0, hbd_tol=1, hba_tol=1,
                   rotb_tol=2, charge_exact=True):
    mw_a, lp_a, hbd_a, hba_a, rb_a, ch_a = props_active
    mw_c, lp_c, hbd_c, hba_c, rb_c, ch_c = props_cand
    if charge_exact and ch_a != ch_c:
        return False
    if abs(mw_a - mw_c) > mw_tol:
        return False
    if abs(lp_a - lp_c) > logp_tol:
        return False
    if abs(hbd_a - hbd_c) > hbd_tol:
        return False
    if abs(hba_a - hba_c) > hba_tol:
        return False
    if abs(rb_a - rb_c) > rotb_tol:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--actives_csv', required=True,
                    help='CSV with active molecules (label=1).')
    ap.add_argument('--library_csv', required=True,
                    help='CSV of candidate decoy molecules to sample from.')
    ap.add_argument('--n_decoys_per_active', type=int, default=30)
    ap.add_argument('--max_tanimoto_to_actives', type=float, default=0.4,
                    help='Decoy candidates with Tanimoto ≥ this to any '
                         'active are rejected. Default 0.4.')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ---- Load actives -----------------------------------------------------
    print(f'\nLoading actives from {args.actives_csv}...')
    a_df = pd.read_csv(args.actives_csv)
    a_smi_col = _find_smi_col(a_df.columns)
    if not a_smi_col:
        sys.exit(f'ERROR: no SMILES column in actives CSV.')

    # Filter to label=1 if present
    if 'label' in a_df.columns:
        a_df = a_df[a_df['label'] == 1].reset_index(drop=True)
    print(f'  {len(a_df)} actives loaded.')

    print('  Computing properties + fingerprints for actives...')
    actives = []
    for smi in a_df[a_smi_col].astype(str):
        props = compute_properties(smi)
        fp = morgan_fp(smi)
        if props and fp:
            actives.append({'smiles': smi, 'props': props, 'fp': fp})
    print(f'  {len(actives)} valid actives after RDKit filtering.')

    # ---- Load library ----------------------------------------------------
    print(f'\nLoading candidate library from {args.library_csv}...')
    lib_df = pd.read_csv(args.library_csv)
    lib_smi_col = _find_smi_col(lib_df.columns)
    if not lib_smi_col:
        sys.exit(f'ERROR: no SMILES column in library.')

    # Prefer inactive candidates (label=0) — they are least likely to be
    # true actives. If label absent, use all.
    if 'label' in lib_df.columns:
        lib_df = lib_df[lib_df['label'] == 0].reset_index(drop=True)
    print(f'  {len(lib_df)} candidate library molecules loaded.')

    print('  Computing properties + fingerprints for library '
          '(this may take 1-2 min)...')
    lib = []
    for i, smi in enumerate(lib_df[lib_smi_col].astype(str)):
        props = compute_properties(smi)
        fp = morgan_fp(smi)
        if props and fp:
            lib.append({'smiles': smi, 'props': props, 'fp': fp, 'used': False})
        if (i + 1) % 500 == 0:
            print(f'    {i+1}/{len(lib_df)}')
    print(f'  {len(lib)} valid library molecules.')

    # ---- For each active, find N property-matched, structurally-distant decoys
    print(f'\nGenerating up to {args.n_decoys_per_active} decoys per active...')
    all_decoys = []
    rng = np.random.RandomState(42)

    for ai, active in enumerate(actives):
        if (ai + 1) % 50 == 0:
            print(f'  Active {ai+1}/{len(actives)}, '
                  f'total decoys so far: {len(all_decoys)}')

        # First-pass: property match candidates
        candidates = []
        for lc in lib:
            if lc['used']:
                continue
            if not property_match(active['props'], lc['props']):
                continue
            # Structural dissimilarity to ALL actives
            max_sim = 0.0
            for other in actives:
                s = DataStructs.TanimotoSimilarity(lc['fp'], other['fp'])
                if s > max_sim:
                    max_sim = s
                    if max_sim >= args.max_tanimoto_to_actives:
                        break
            if max_sim >= args.max_tanimoto_to_actives:
                continue
            candidates.append(lc)
            if len(candidates) >= args.n_decoys_per_active * 3:
                break

        if not candidates:
            continue
        # Sample n_decoys from candidates
        rng.shuffle(candidates)
        chosen = candidates[:args.n_decoys_per_active]
        for c in chosen:
            c['used'] = True
            all_decoys.append({
                'smiles': c['smiles'],
                'label': 0,
                'source': f'DUD-E_decoy_of_active_{ai}',
                'data_source': 'DUD-E_decoy',
                'source_target': 'NLRP3',
            })

    # ---- Write ----------------------------------------------------------
    with open(args.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['smiles', 'label', 'source',
                                          'data_source', 'source_target'])
        w.writeheader()
        w.writerows(all_decoys)

    print(f'\n✓ Wrote {len(all_decoys)} decoys to {args.output}')
    print(f'  Mean decoys per active: {len(all_decoys) / max(1, len(actives)):.1f}')


if __name__ == '__main__':
    main()
