#!/usr/bin/env python3
"""
Random Forest baseline on V5-random dataset.

Diagnostic purpose: establish where the AUC ceiling actually sits.
  * If RF + Morgan reaches 0.85+ → STG-Mol architecture is
    under-performing → need pretrained encoder / bigger model.
  * If RF only reaches ~0.80 → V5 dataset AUC ceiling is around
    that value → need to rebuild dataset (V6) with less-adversarial
    negatives.

Uses default RF hyperparameters (500 trees, sqrt features) —
this is a "cheap sanity check" not a fair competitor to STG-Mol.

Usage:
    python scripts/expand/rf_baseline_v5.py \\
        --train_csv data/processed/nlrp3_v5_random/train.csv \\
        --val_csv   data/processed/nlrp3_v5_random/val.csv \\
        --test_csv  data/processed/nlrp3_v5_random/test.csv \\
        --external_csv data/processed/nlrp3_v5_random/external_holdout.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    sys.exit('ERROR: rdkit required.')


def morgan_fp(smi, radius=2, n_bits=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits))


def load_features(csv_path, smi_col_cands=('smiles', 'smiles_standardized')):
    df = pd.read_csv(csv_path, encoding='utf-8')
    smi_col = next((c for c in smi_col_cands if c in df.columns), None)
    if smi_col is None:
        sys.exit(f'No SMILES in {csv_path}')

    fps, labels, smis, sources = [], [], [], []
    src_col = 'data_source' if 'data_source' in df.columns else None
    for _, r in df.iterrows():
        fp = morgan_fp(str(r[smi_col]))
        if fp is None:
            continue
        fps.append(fp)
        labels.append(int(r['label']))
        smis.append(str(r[smi_col]))
        sources.append(str(r[src_col]) if src_col else 'unknown')
    return np.array(fps), np.array(labels), smis, sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--val_csv', required=True)
    ap.add_argument('--test_csv', required=True)
    ap.add_argument('--external_csv', default=None)
    ap.add_argument('--n_estimators', type=int, default=500)
    ap.add_argument('--n_seeds', type=int, default=5)
    args = ap.parse_args()

    print('\n[1/3] Featurising all sets (Morgan r=2, 2048 bits)...')
    X_tr, y_tr, _, src_tr = load_features(args.train_csv)
    X_va, y_va, _, src_va = load_features(args.val_csv)
    X_te, y_te, smi_te, src_te = load_features(args.test_csv)
    print(f'  train: {X_tr.shape}, pos={y_tr.sum()}')
    print(f'  val:   {X_va.shape}, pos={y_va.sum()}')
    print(f'  test:  {X_te.shape}, pos={y_te.sum()}')

    all_probs = []
    print(f'\n[2/3] Training {args.n_seeds}-seed RF ensemble...')
    for i, seed in enumerate([42, 123, 2024, 3407, 7][:args.n_seeds]):
        rf = RandomForestClassifier(n_estimators=args.n_estimators,
                                     random_state=seed,
                                     n_jobs=-1)
        rf.fit(X_tr, y_tr)
        p_te = rf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, p_te)
        print(f'  seed {seed}: test AUC = {auc:.4f}')
        all_probs.append(p_te)

    # Ensemble
    p_ens = np.mean(all_probs, axis=0)
    auc_ens = roc_auc_score(y_te, p_ens)
    f1_ens = f1_score(y_te, (p_ens >= 0.5).astype(int))
    mcc_ens = matthews_corrcoef(y_te, (p_ens >= 0.5).astype(int))

    print(f'\n=== RF Ensemble Test Results ===')
    print(f'  Test AUC:  {auc_ens:.4f}')
    print(f'  Test F1:   {f1_ens:.4f}')
    print(f'  Test MCC:  {mcc_ens:.4f}')

    # Per-source AUC on test set
    print(f'\n=== Per-source Test AUC (Ensemble) ===')
    for source in sorted(set(src_te)):
        mask = np.array([s == source for s in src_te])
        y_sub = y_te[mask]
        p_sub = p_ens[mask]
        n_pos, n_neg = int(y_sub.sum()), int((1 - y_sub).sum())
        if n_pos > 0 and n_neg > 0:
            auc_sub = roc_auc_score(y_sub, p_sub)
            print(f'  {source:25s}  n={len(y_sub):4d}  '
                  f'({n_pos} pos, {n_neg} neg)  AUC={auc_sub:.4f}')
        else:
            avg_p = p_sub.mean() if len(p_sub) else 0
            print(f'  {source:25s}  n={len(y_sub):4d}  '
                  f'({n_pos} pos, {n_neg} neg)  [homogeneous, avg_p={avg_p:.3f}]')

    # External hold-out
    if args.external_csv:
        print(f'\n[3/3] External hold-out recall (RF ensemble)...')
        X_ext, y_ext, smi_ext, _ = load_features(args.external_csv)
        p_ext = np.mean([RandomForestClassifier(
            n_estimators=args.n_estimators, random_state=s, n_jobs=-1
        ).fit(X_tr, y_tr).predict_proba(X_ext)[:, 1]
                         for s in [42, 123, 2024, 3407, 7][:args.n_seeds]],
                        axis=0)
        pred_active = (p_ext >= 0.5).astype(int)
        print(f'  External recall: {pred_active.sum()}/{len(p_ext)}')
        for i, smi in enumerate(smi_ext):
            print(f'    smi={smi[:50]:52s}  prob={p_ext[i]:.3f}  '
                  f'active={pred_active[i]}')


if __name__ == '__main__':
    main()
