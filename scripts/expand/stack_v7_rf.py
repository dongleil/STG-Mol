#!/usr/bin/env python3
"""
Stacking ensemble: STG-Mol V7 (ChemBERTa) + Random Forest.

Simple probability averaging (with weight sweep) between the two
complementary models:
  * STG-Mol V7 — deep multi-modal, better BEDROC + external recall
  * RF+Morgan  — tabular ML, higher raw AUC on V6

Expected: stacking pulls both AUC and BEDROC up beyond either alone.

Usage:
    python scripts/expand/stack_v7_rf.py \\
        --v7_test_csv predictions_test_v7/test_pred_ensemble.csv \\
        --v7_ext_csv predictions_external_v7/test_pred_ensemble.csv \\
        --train_csv data/processed/nlrp3_v6/train.csv \\
        --test_csv data/processed/nlrp3_v6/test.csv \\
        --ext_csv data/processed/nlrp3_v6/external_holdout.csv \\
        --output_dir stacking_v7_rf
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


def morgan(smi, r=2, n=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(m, r, nBits=n))


def load_fp(csv, need_labels=True):
    df = pd.read_csv(csv, encoding='utf-8')
    smi_col = next((c for c in ['smiles_standardized', 'smiles']
                     if c in df.columns), None)
    fps, y, smis = [], [], []
    for _, r in df.iterrows():
        fp = morgan(str(r[smi_col]))
        if fp is None:
            continue
        fps.append(fp)
        smis.append(str(r[smi_col]))
        if need_labels and 'label' in df.columns:
            y.append(int(r['label']))
    return np.array(fps), np.array(y) if y else None, smis


def enrichment_factor(y, p, k):
    n = len(y)
    top = max(1, int(np.ceil(n * k)))
    order = np.argsort(-p)
    return ((y[order[:top]].sum() / top) / (y.sum() / n)) if y.sum() > 0 else 0


def bedroc(y, p, alpha):
    n = len(y)
    n_a = int(y.sum())
    if n_a == 0 or n_a == n:
        return 0.0
    R_a = n_a / n
    order = np.argsort(-p)
    ranks = np.where(y[order] == 1)[0] + 1
    S = float(np.sum(np.exp(-alpha * ranks / n)))
    fac = R_a * np.sinh(alpha / 2) / (
        np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * R_a))
    num = S * fac
    den = R_a * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1)
    corr = 1.0 / (1.0 - np.exp(alpha * (1 - R_a)))
    return float(max(0, min(1, num / den + corr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v7_test_csv', required=True)
    ap.add_argument('--v7_ext_csv', required=True)
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--test_csv', required=True)
    ap.add_argument('--ext_csv', required=True)
    ap.add_argument('--output_dir', required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Train RF on V6 train set --------------------------------------
    print('\n[1/4] Training 5-seed RF on V6 train set...')
    X_tr, y_tr, _ = load_fp(args.train_csv)
    print(f'  train: {X_tr.shape}, pos={y_tr.sum()}')

    X_te, y_te, smi_te = load_fp(args.test_csv)
    X_ext, y_ext, smi_ext = load_fp(args.ext_csv)
    print(f'  test:  {X_te.shape}, pos={y_te.sum()}')
    print(f'  ext:   {X_ext.shape}, pos={y_ext.sum()}')

    rf_probs_test, rf_probs_ext = [], []
    for seed in [42, 123, 2024, 3407, 7]:
        rf = RandomForestClassifier(n_estimators=500, random_state=seed,
                                     n_jobs=-1)
        rf.fit(X_tr, y_tr)
        rf_probs_test.append(rf.predict_proba(X_te)[:, 1])
        rf_probs_ext.append(rf.predict_proba(X_ext)[:, 1])
    rf_test = np.mean(rf_probs_test, axis=0)
    rf_ext = np.mean(rf_probs_ext, axis=0)
    print(f'  RF Test AUC: {roc_auc_score(y_te, rf_test):.4f}')
    print(f'  RF External recall (thr 0.5): '
          f'{int((rf_ext >= 0.5).sum())}/{len(rf_ext)}')

    # ---- Load V7 predictions -------------------------------------------
    print('\n[2/4] Loading V7 predictions...')
    v7_test_df = pd.read_csv(args.v7_test_csv, encoding='utf-8')
    v7_ext_df = pd.read_csv(args.v7_ext_csv, encoding='utf-8')

    # Match V7 predictions to test set by SMILES order
    def match_probs(pred_df, target_smis):
        d = dict(zip(pred_df['smiles'].astype(str),
                     pred_df['activity_prob'].astype(float)))
        return np.array([d.get(s, np.nan) for s in target_smis])

    v7_test = match_probs(v7_test_df, smi_te)
    v7_ext = match_probs(v7_ext_df, smi_ext)

    # Handle NaN
    valid_te = ~np.isnan(v7_test)
    valid_ext = ~np.isnan(v7_ext)
    if valid_te.sum() < len(v7_test):
        print(f'  ⚠ {(~valid_te).sum()} test SMILES not matched between V7 and RF')

    print(f'  V7 Test AUC: {roc_auc_score(y_te[valid_te], v7_test[valid_te]):.4f}')
    print(f'  V7 External recall (thr 0.5): '
          f'{int((v7_ext[valid_ext] >= 0.5).sum())}/{valid_ext.sum()}')

    # ---- Stacking weight sweep -----------------------------------------
    print('\n[3/4] Weight sweep: p_stacked = w * V7 + (1-w) * RF')
    print(f'{"w":>6} {"Test AUC":>10} {"BEDROC@20":>11} '
          f'{"BEDROC@80":>11} {"EF@1%":>7} {"Ext":>4}')
    print('-' * 60)
    best_w, best_auc = 0.5, 0
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        p_te = w * v7_test + (1 - w) * rf_test
        p_ext = w * v7_ext + (1 - w) * rf_ext
        auc = roc_auc_score(y_te[valid_te], p_te[valid_te])
        b20 = bedroc(y_te[valid_te], p_te[valid_te], 20)
        b80 = bedroc(y_te[valid_te], p_te[valid_te], 80)
        ef1 = enrichment_factor(y_te[valid_te], p_te[valid_te], 0.01)
        rec = int((p_ext[valid_ext] >= 0.5).sum())
        marker = '  ← best' if auc > best_auc else ''
        if auc > best_auc:
            best_w, best_auc = w, auc
        print(f'  {w:>4.1f} {auc:>10.4f} {b20:>11.4f} {b80:>11.4f} '
              f'{ef1:>7.3f} {rec}/{valid_ext.sum()}{marker}')

    # ---- Save best stacking output --------------------------------------
    print(f'\n[4/4] Best weight = {best_w:.1f} (V7 weight)')
    p_te_best = best_w * v7_test + (1 - best_w) * rf_test
    p_ext_best = best_w * v7_ext + (1 - best_w) * rf_ext

    out_test = pd.DataFrame({
        'smiles': smi_te,
        'v7_prob': v7_test,
        'rf_prob': rf_test,
        'stack_prob': p_te_best,
        'label': y_te,
        'predicted_active': (p_te_best >= 0.5).astype(int),
    })
    out_test.to_csv(out / 'stack_test_predictions.csv', index=False,
                    encoding='utf-8')

    out_ext = pd.DataFrame({
        'smiles': smi_ext,
        'v7_prob': v7_ext,
        'rf_prob': rf_ext,
        'stack_prob': p_ext_best,
        'label': y_ext,
        'predicted_active': (p_ext_best >= 0.5).astype(int),
    })
    if 'name' in pd.read_csv(args.ext_csv, encoding='utf-8').columns:
        names = pd.read_csv(args.ext_csv, encoding='utf-8')['name'].tolist()
        out_ext.insert(0, 'name', names[:len(smi_ext)])
    out_ext.to_csv(out / 'stack_external_predictions.csv', index=False,
                   encoding='utf-8')

    print(f'\n=== Final Stacking Results (w={best_w:.1f}) ===')
    print(f'  Test AUC:      {best_auc:.4f}')
    print(f'  BEDROC@α=20:   {bedroc(y_te[valid_te], p_te_best[valid_te], 20):.4f}')
    print(f'  BEDROC@α=80:   {bedroc(y_te[valid_te], p_te_best[valid_te], 80):.4f}')
    print(f'  EF@1%:         {enrichment_factor(y_te[valid_te], p_te_best[valid_te], 0.01):.4f}')
    print(f'  External:      {int((p_ext_best[valid_ext] >= 0.5).sum())}/{valid_ext.sum()}')
    print(f'\n  ✓ Saved test predictions → {out}/stack_test_predictions.csv')
    print(f'  ✓ Saved external predictions → {out}/stack_external_predictions.csv')
    print(f'\n  External hold-out details (name, V7, RF, stacked):')
    print(out_ext[['name', 'v7_prob', 'rf_prob',
                    'stack_prob', 'predicted_active']].to_string(index=False)
          if 'name' in out_ext.columns
          else out_ext.to_string(index=False))


if __name__ == '__main__':
    main()
