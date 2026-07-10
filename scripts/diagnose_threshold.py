#!/usr/bin/env python3
"""
Diagnose threshold sensitivity + known-inhibitor recall.

Prints:
  * activity_prob distribution on test set (all / positives / negatives)
  * Youden-optimal threshold and F1-optimal threshold
  * Known-inhibitor predictions at multiple thresholds

Usage:
    python scripts/diagnose_threshold.py \
        --test_csv predictions_test_5seed/test_pred_ensemble.csv \
        --known_csv predictions_known_inhibitors.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             confusion_matrix, roc_auc_score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_csv', required=True)
    ap.add_argument('--known_csv', required=True)
    args = ap.parse_args()

    # ---- Test-set distribution -------------------------------------------
    df = pd.read_csv(args.test_csv)
    y = df['label'].astype(int).values
    p = df['activity_prob'].astype(float).values

    print(f'\n=== TEST SET   n={len(y)}   AUC={roc_auc_score(y, p):.4f} ===')
    print(f'  Positive rate: {y.mean():.3f}  ({int(y.sum())}/{len(y)})')

    def q(x, tag):
        print(f'  {tag:12s}  min={x.min():.3f}  '
              f'q25={np.quantile(x, .25):.3f}  '
              f'q50={np.median(x):.3f}  '
              f'q75={np.quantile(x, .75):.3f}  '
              f'max={x.max():.3f}')

    q(p, 'all')
    q(p[y == 1], 'positives')
    q(p[y == 0], 'negatives')

    # ---- Threshold search -------------------------------------------------
    fpr, tpr, thr_roc = roc_curve(y, p)
    j = tpr - fpr
    thr_youden = float(thr_roc[j.argmax()])
    print(f'\n  Youden-optimal threshold: {thr_youden:.4f}  '
          f'(TPR={tpr[j.argmax()]:.3f}, FPR={fpr[j.argmax()]:.3f})')

    prec, rec, thr_pr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    thr_f1 = float(thr_pr[f1[:-1].argmax()])
    print(f'  F1-optimal threshold:     {thr_f1:.4f}  '
          f'(P={prec[f1[:-1].argmax()]:.3f}, R={rec[f1[:-1].argmax()]:.3f}, '
          f'F1={f1[:-1].max():.3f})')

    # Confusion matrices at three thresholds
    for name, t in [('0.5 (default)', 0.5),
                    (f'Youden ({thr_youden:.3f})', thr_youden),
                    (f'F1-opt ({thr_f1:.3f})', thr_f1)]:
        pred = (p >= t).astype(int)
        cm = confusion_matrix(y, pred)
        print(f'\n  Threshold = {name}')
        print(f'    Confusion matrix (rows=true, cols=pred):\n'
              f'      TN={cm[0,0]:3d}  FP={cm[0,1]:3d}\n'
              f'      FN={cm[1,0]:3d}  TP={cm[1,1]:3d}')

    # ---- Known-inhibitor recall ------------------------------------------
    kdf = pd.read_csv(args.known_csv)
    print(f'\n=== KNOWN INHIBITORS   n={len(kdf)} ===')
    print(kdf[['name', 'activity_prob', 'predicted_active']].to_string(index=False))

    for name, t in [('0.5 (default)', 0.5),
                    (f'Youden ({thr_youden:.3f})', thr_youden),
                    (f'F1-opt ({thr_f1:.3f})', thr_f1)]:
        n_hit = int((kdf['activity_prob'] >= t).sum())
        print(f'  Threshold {name}: recall = {n_hit}/{len(kdf)} '
              f'= {n_hit/len(kdf)*100:.0f}%')


if __name__ == '__main__':
    main()
