#!/usr/bin/env python3
"""
Diagnose test-set errors by data source.

Given STG-Mol's predictions on V5-random test set, break down:
  * Per-source AUC (ChEMBL vs PubChem vs Literature vs HardNeg)
  * Per-source recall / precision at threshold 0.5
  * Which molecules are misclassified and their sources

Identifies the source that is dragging AUC down. If PubChem confirmed
inactives are the culprit, we can rebuild V6 with a subset.

Usage:
    python scripts/expand/diagnose_test_errors_by_source.py \\
        --test_csv data/processed/nlrp3_v5_random/test.csv \\
        --pred_csv predictions_test_v5_random/test_pred_ensemble.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_csv', required=True)
    ap.add_argument('--pred_csv', required=True)
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()

    test_df = pd.read_csv(args.test_csv, encoding='utf-8')
    pred_df = pd.read_csv(args.pred_csv, encoding='utf-8')

    # Find SMILES columns
    test_smi = next((c for c in ['smiles', 'smiles_standardized']
                     if c in test_df.columns), None)
    pred_smi = next((c for c in ['smiles', 'smiles_standardized']
                     if c in pred_df.columns), None)
    if not test_smi or not pred_smi:
        sys.exit('No SMILES column found.')

    # Merge on SMILES
    merged = test_df.merge(pred_df[[pred_smi, 'activity_prob']],
                            left_on=test_smi, right_on=pred_smi,
                            how='inner')
    print(f'\nMerged {len(merged)} molecules from test set with predictions.')

    y = merged['label'].astype(int).values
    p = merged['activity_prob'].astype(float).values
    pred = (p >= args.threshold).astype(int)

    print(f'\n=== Overall Test Results ===')
    print(f'  n = {len(y)}, pos = {y.sum()}, neg = {(1-y).sum()}')
    print(f'  Test AUC (recomputed) = {roc_auc_score(y, p):.4f}')
    cm = confusion_matrix(y, pred)
    print(f'  Confusion matrix (thr={args.threshold}):')
    print(f'    TN={cm[0,0]:3d}  FP={cm[0,1]:3d}')
    print(f'    FN={cm[1,0]:3d}  TP={cm[1,1]:3d}')

    # Per-source breakdown
    if 'data_source' in merged.columns:
        print(f'\n=== Per-source Breakdown ===')
        for source in sorted(merged['data_source'].unique()):
            mask = merged['data_source'] == source
            y_s = y[mask]
            p_s = p[mask]
            pred_s = pred[mask]
            n = mask.sum()
            n_pos = int(y_s.sum())
            n_neg = int((1 - y_s).sum())
            if n_pos > 0 and n_neg > 0:
                auc = roc_auc_score(y_s, p_s)
            else:
                auc = None

            correct = (pred_s == y_s).sum()
            wrong = n - correct
            print(f'\n  {source}:')
            print(f'    n_total = {n}   n_pos = {n_pos}   n_neg = {n_neg}')
            print(f'    AUC     = {auc:.4f}' if auc is not None else '    AUC     = (homogeneous)')
            print(f'    Accuracy= {correct}/{n} = {correct/n:.3f}')
            print(f'    Mean prob: pos={p_s[y_s==1].mean():.3f} '
                  f'neg={p_s[y_s==0].mean():.3f}' if n_pos and n_neg else '')

    # Top misclassified (false negatives)
    fn_mask = (y == 1) & (pred == 0)
    if fn_mask.any():
        print(f'\n=== Top False Negatives (active but predicted inactive) ===')
        fn_df = merged[fn_mask].copy()
        fn_df['prob'] = p[fn_mask]
        fn_df = fn_df.sort_values('prob').head(20)
        print(f'  Total false negatives: {fn_mask.sum()}')
        for _, r in fn_df.iterrows():
            src = r.get('data_source', 'unknown')
            print(f'    prob={r["prob"]:.3f}  source={src:25s}  '
                  f'smi={str(r[test_smi])[:70]}')

    # Top false positives
    fp_mask = (y == 0) & (pred == 1)
    if fp_mask.any():
        print(f'\n=== Top False Positives (inactive but predicted active) ===')
        fp_df = merged[fp_mask].copy()
        fp_df['prob'] = p[fp_mask]
        fp_df = fp_df.sort_values('prob', ascending=False).head(20)
        print(f'  Total false positives: {fp_mask.sum()}')
        for _, r in fp_df.iterrows():
            src = r.get('data_source', 'unknown')
            print(f'    prob={r["prob"]:.3f}  source={src:25s}  '
                  f'smi={str(r[test_smi])[:70]}')


if __name__ == '__main__':
    main()
