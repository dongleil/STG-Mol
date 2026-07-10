#!/usr/bin/env python3
"""
Applicability-domain analysis for the 5 known NLRP3 inhibitors.

Answers two critical questions for the paper:

  Q1: Are any of the 5 published inhibitors already present in train/val?
      Uses RDKit canonical InChIKey (canonical tautomer + stereochemistry)
      as the identity match — the standard way to dedup molecules.

  Q2: For each published inhibitor, what is its nearest-neighbour Tanimoto
      similarity to the training set (Morgan fingerprint, radius=2,
      2048 bits)? This is the classical Applicability-Domain (AD) test.

      Interpretation:
        Tanimoto ≥ 0.7  → in-domain, model should generalise well
        Tanimoto 0.4-0.7 → borderline
        Tanimoto < 0.4  → out-of-domain, model prediction may be unreliable

Output: per-compound table with in-train flag, activity_prob (from
predictions_known_inhibitors.csv if given), nearest-neighbour Tanimoto,
and the single most-similar train molecule.

Usage:
    python scripts/check_known_inhibitor_novelty.py \
        --train_csv data/processed/nlrp3/train.csv \
        --val_csv   data/processed/nlrp3/val.csv \
        --test_csv  data/processed/nlrp3/test.csv \
        --known_csv data/known_inhibitors.csv \
        --pred_csv  predictions_known_inhibitors.csv
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
except ImportError:
    sys.exit('ERROR: rdkit not available.')


_SMI_CANDS = ['smiles', 'smiles_standardized', 'canonical_smiles', 'SMILES']


def _find_smi_col(cols):
    lc = {c.lower(): c for c in cols}
    for c in _SMI_CANDS:
        if c.lower() in lc:
            return lc[c.lower()]
    sys.exit(f'No SMILES column in {cols}')


def _to_inchikey(smi):
    """Return canonical InChIKey (14-char connectivity block)."""
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return MolToInchiKey(m)
    except Exception:
        return None


def _to_fp(smi, radius=2, n_bits=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)


def _load(csv, tag):
    df = pd.read_csv(csv)
    smi_col = _find_smi_col(df.columns)
    smis = df[smi_col].astype(str).tolist()
    print(f'  {tag:8s} {csv}  n={len(smis)}')
    keys = [_to_inchikey(s) for s in smis]
    fps = [_to_fp(s) for s in smis]
    return df, smi_col, smis, keys, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--val_csv', required=True)
    ap.add_argument('--test_csv', default=None)
    ap.add_argument('--known_csv', required=True)
    ap.add_argument('--pred_csv', default=None,
                    help='Optional predictions_known_inhibitors.csv to merge in.')
    args = ap.parse_args()

    print('\n=== Loading datasets ===')
    _, _, train_smi, train_key, train_fp = _load(args.train_csv, 'train')
    _, _, val_smi, val_key, val_fp = _load(args.val_csv, 'val')
    if args.test_csv:
        _, _, test_smi, test_key, test_fp = _load(args.test_csv, 'test')
    else:
        test_smi, test_key, test_fp = [], [], []

    known_df = pd.read_csv(args.known_csv)
    ksmi_col = _find_smi_col(known_df.columns)

    # Optional predictions merge
    pred_df = None
    if args.pred_csv and Path(args.pred_csv).exists():
        pred_df = pd.read_csv(args.pred_csv)

    train_key_set = set(k for k in train_key if k)
    val_key_set = set(k for k in val_key if k)
    test_key_set = set(k for k in test_key if k)

    rows = []
    print('\n=== Per-compound AD analysis ===')
    for _, r in known_df.iterrows():
        name = r.get('name', '?')
        smi = str(r[ksmi_col])
        key = _to_inchikey(smi)
        fp = _to_fp(smi)

        in_train = key in train_key_set if key else False
        in_val = key in val_key_set if key else False
        in_test = key in test_key_set if key else False

        # Nearest-neighbour in train (Morgan Tanimoto)
        best_sim, best_nn = 0.0, ''
        if fp is not None:
            for tsmi, tfp in zip(train_smi, train_fp):
                if tfp is None:
                    continue
                s = DataStructs.TanimotoSimilarity(fp, tfp)
                if s > best_sim:
                    best_sim, best_nn = s, tsmi

        # Merge in activity_prob if provided
        prob = np.nan
        if pred_df is not None:
            match = pred_df[pred_df['name'] == name]
            if len(match) > 0:
                prob = float(match.iloc[0]['activity_prob'])

        # AD verdict
        if in_train:
            verdict = '❗ IN TRAIN (leakage — cannot validate)'
        elif in_val:
            verdict = '⚠️  IN VAL (partial leakage)'
        elif in_test:
            verdict = 'ℹ️  IN TEST (already counted in test AUC)'
        elif best_sim >= 0.7:
            verdict = '✓ IN DOMAIN (Tanimoto ≥ 0.7)'
        elif best_sim >= 0.4:
            verdict = '~ BORDERLINE (0.4 ≤ Tanimoto < 0.7)'
        else:
            verdict = '✗ OUT OF DOMAIN (Tanimoto < 0.4)'

        rows.append({
            'name': name,
            'in_train': in_train,
            'in_val': in_val,
            'in_test': in_test,
            'nearest_tanimoto': round(best_sim, 3),
            'activity_prob': round(prob, 3) if not np.isnan(prob) else np.nan,
            'verdict': verdict,
        })
        print(f'\n  {name}')
        print(f'    InChIKey        {key}')
        print(f'    in train / val / test:  {in_train} / {in_val} / {in_test}')
        print(f'    nearest-NN Tanimoto:    {best_sim:.3f}')
        print(f'    nearest-NN SMILES:      {best_nn}')
        print(f'    predicted activity_prob:{prob:.3f}' if not np.isnan(prob) else '')
        print(f'    → {verdict}')

    summary = pd.DataFrame(rows)
    out_path = 'ad_analysis_known_inhibitors.csv'
    summary.to_csv(out_path, index=False)
    print(f'\n=== SUMMARY  → {out_path} ===')
    print(summary.to_string(index=False))

    n_in_train = int(summary['in_train'].sum())
    n_in_val = int(summary['in_val'].sum())
    n_in_test = int(summary['in_test'].sum())
    n_novel = len(summary) - n_in_train - n_in_val - n_in_test
    print(f'\n  {n_in_train}/{len(summary)} known inhibitors are in train set')
    print(f'  {n_in_val}/{len(summary)} known inhibitors are in val set')
    print(f'  {n_in_test}/{len(summary)} known inhibitors are in test set')
    print(f'  {n_novel}/{len(summary)} known inhibitors are strictly novel')

    if n_novel:
        novel = summary[~summary['in_train'] & ~summary['in_val'] & ~summary['in_test']]
        avg = novel['nearest_tanimoto'].mean()
        print(f'\n  Among strictly-novel inhibitors:')
        print(f'    Mean nearest-NN Tanimoto to train: {avg:.3f}')
        if not novel['activity_prob'].isna().all():
            correct = int((novel['activity_prob'] >= 0.5).sum())
            print(f'    Recall @ threshold=0.5: {correct}/{len(novel)}')


if __name__ == '__main__':
    main()
