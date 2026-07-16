#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_classical_baselines.py — ECFP4 + {SVM, RF, XGBoost} on the leakage-free
NLRP3 dataset, run under both V3-scaffold (primary) and V3-random (reference)
protocols. Produces:

    (a) per-seed prediction CSVs under
        results/baselines_classical/<split>/predictions/
        that scripts/recompute_metrics_v42.py can consume directly, and
    (b) a paper-ready summary markdown table for the baseline rows of
        Table 5.1a (V3-scaffold) and Table 5.1b (V3-random).

Seeds default to {42, 123, 2024, 3407, 7} to match STG-Mol's 5-seed protocol.

Usage:
    # Uses configs/train_v26_scaffold.yaml and configs/train_v26_v3_random.yaml
    # to locate the split CSVs (data/processed/nlrp3/ for scaffold,
    # data/processed/nlrp3_v3_random/ for random). Data paths can also be
    # overridden explicitly.
    python scripts/train_classical_baselines.py

    # Or single split:
    python scripts/train_classical_baselines.py --split scaffold
    python scripts/train_classical_baselines.py --split random \\
        --data_root data/processed/nlrp3_v3_random

Dependencies: rdkit, scikit-learn, xgboost, pandas, numpy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, precision_score, recall_score,
)

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except (ImportError, OSError) as _xgb_err:
    _XGB_AVAILABLE = False
    _XGB_ERR = str(_xgb_err)


RDLogger.DisableLog('rdApp.*')  # suppress RDKit atom-valence warnings


SEEDS = [42, 123, 2024, 3407, 7]

SPLIT_DATA_ROOTS = {
    'scaffold': 'data/processed/nlrp3',            # V3-scaffold (primary)
    'random':   'data/processed/nlrp3_v3_random',  # V3-random  (reference)
}

# --------------------------------------------------------------------------
# Featurisation
# --------------------------------------------------------------------------

def _fp_bits(mol, n_bits=2048, radius=2):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr


def featurise(smiles_list, n_bits=2048, radius=2):
    X = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    n_bad = 0
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            n_bad += 1
            continue
        X[i] = _fp_bits(mol, n_bits=n_bits, radius=radius)
    if n_bad:
        print(f'   ! {n_bad}/{len(smiles_list)} SMILES failed RDKit parsing '
              f'(fingerprint left all-zero)')
    return X


def _pick(colnames, candidates):
    lc = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand.lower() in lc:
            return lc[cand.lower()]
    return None


def load_split(root: Path):
    """Return (X_train, y_train, X_val, y_val, X_test, y_test)."""
    train = pd.read_csv(root / 'train.csv')
    val   = pd.read_csv(root / 'val.csv')
    test  = pd.read_csv(root / 'test.csv')
    smi = _pick(train.columns, ['smiles_standardized', 'smiles', 'SMILES'])
    if smi is None:
        raise ValueError(f'No SMILES column in {root / "train.csv"}: '
                         f'{train.columns.tolist()}')
    lbl = _pick(train.columns, ['label', 'activity', 'y'])
    if lbl is None:
        raise ValueError(f'No label column in {root / "train.csv"}')
    print(f'   Loaded {len(train)} train / {len(val)} val / {len(test)} test '
          f'from {root}')
    print(f'   Featurising Morgan ECFP4 (radius=2, n_bits=2048)...')
    return (
        featurise(train[smi].tolist()), train[lbl].astype(int).values,
        featurise(val[smi].tolist()),   val[lbl].astype(int).values,
        featurise(test[smi].tolist()),  test[lbl].astype(int).values,
    )


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def build_svm(seed):
    return SVC(kernel='rbf', C=1.0, gamma='scale',
                probability=True, class_weight='balanced', random_state=seed)


def build_rf(seed):
    return RandomForestClassifier(
        n_estimators=500, max_features='sqrt',
        min_samples_split=2, n_jobs=-1, class_weight='balanced',
        random_state=seed)


def build_xgb(seed):
    return XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        objective='binary:logistic', eval_metric='logloss',
        tree_method='hist', n_jobs=-1, random_state=seed)


MODELS = {'SVM': build_svm, 'RF': build_rf}
if _XGB_AVAILABLE:
    MODELS['XGBoost'] = build_xgb
else:
    print(f'! XGBoost unavailable, running SVM + RF only.\n'
          f'  Reason: {_XGB_ERR[:120]}...\n'
          f'  To enable: brew install libomp   (Mac) '
          f'or apt-get install libgomp1 (Linux)\n')


# --------------------------------------------------------------------------
# Metrics (mirrors recompute_metrics_v42.py exactly)
# --------------------------------------------------------------------------

def _proba(clf, X):
    if hasattr(clf, 'predict_proba'):
        return clf.predict_proba(X)[:, 1]
    if hasattr(clf, 'decision_function'):
        z = clf.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(f'{type(clf).__name__} has no probability output')


def compute_bundle(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    return {
        'ROC_AUC':   float(roc_auc_score(y_true, y_score)),
        'PR_AUC':    float(average_precision_score(y_true, y_score)),
        'F1':        float(f1_score(y_true, y_pred)),
        'MCC':       float(matthews_corrcoef(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall':    float(recall_score(y_true, y_pred)),
    }


def agg(per_seed_metrics):
    out = {}
    for k in per_seed_metrics[0].keys():
        vals = np.array([m[k] for m in per_seed_metrics])
        out[k] = {'mean': float(vals.mean()),
                  'std':  float(vals.std(ddof=0)),
                  'per_seed': [float(v) for v in vals]}
    return out


# --------------------------------------------------------------------------
# Train / eval / dump
# --------------------------------------------------------------------------

def run_one_split(split_name, data_root: Path, out_root: Path):
    print(f'\n=== {split_name} split from {data_root} ===')
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_split(data_root)
    print(f'   test set: N = {len(y_te)}, P = {int(y_te.sum())}, '
          f'prevalence = {y_te.mean():.4f}, '
          f'EF_max = {len(y_te) / y_te.sum():.4f}')

    pred_dir = out_root / split_name / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for model_name, builder in MODELS.items():
        print(f'\n   --- {model_name} ---')
        per_seed, seed_scores = [], []
        for seed in SEEDS:
            clf = builder(seed)
            clf.fit(X_tr, y_tr)
            y_score = _proba(clf, X_te)
            m = compute_bundle(y_te, y_score)
            per_seed.append(m)
            seed_scores.append(y_score)
            pd.DataFrame({'label': y_te.astype(int),
                          'pred_prob': y_score.astype(float)}).to_csv(
                pred_dir / f'test_pred_{model_name}_seed{seed}.csv',
                index=False, float_format='%.6f')
            print(f'     seed {seed}: ROC-AUC = {m["ROC_AUC"]:.4f}, '
                  f'F1 = {m["F1"]:.4f}, MCC = {m["MCC"]:.4f}')

        ens_score = np.mean(seed_scores, axis=0)
        ens_bundle = compute_bundle(y_te, ens_score)
        pd.DataFrame({'label': y_te.astype(int),
                      'pred_prob': ens_score.astype(float)}).to_csv(
            pred_dir / f'test_pred_{model_name}_ensemble.csv',
            index=False, float_format='%.6f')

        results[model_name] = {'per_seed': per_seed,
                                'aggregate': agg(per_seed),
                                'ensemble': ens_bundle}

    # Save JSON
    (out_root / split_name / 'summary.json').write_text(json.dumps(results,
                                                                    indent=2),
                                                        encoding='utf-8')
    return results


# --------------------------------------------------------------------------
# Markdown report — paper-ready rows for Table 5.1a / 5.1b
# --------------------------------------------------------------------------

def _fmt(agg_entry, digits=4):
    return f'{agg_entry["mean"]:.{digits}f} ± {agg_entry["std"]:.{digits}f}'


def render_markdown(results_by_split):
    lines = ['# Classical QSAR baselines — 5-seed mean ± std / ensemble',
             '']
    for split_name, results in results_by_split.items():
        lines += [
            f'## {split_name.upper()} split (protocol: '
            f'{"scaffold, primary" if split_name == "scaffold" else "random, reference"})',
            '',
            '| Model | ROC-AUC (5-seed mean ± std) | F1 | MCC | Recall | Precision |',
            '|---|---|---|---|---|---|',
        ]
        for m_name, r in results.items():
            a = r['aggregate']
            lines.append(
                f'| ECFP4 + {m_name} | '
                f'{_fmt(a["ROC_AUC"])} | {_fmt(a["F1"])} | {_fmt(a["MCC"])} | '
                f'{_fmt(a["Recall"])} | {_fmt(a["Precision"])} |')
        lines.append('')
        lines.append(
            '| Model | ROC-AUC (5-seed ensemble) | F1 | MCC | Recall | Precision |')
        lines.append('|---|---|---|---|---|---|')
        for m_name, r in results.items():
            e = r['ensemble']
            lines.append(
                f'| ECFP4 + {m_name} | {e["ROC_AUC"]:.4f} | {e["F1"]:.4f} | '
                f'{e["MCC"]:.4f} | {e["Recall"]:.4f} | {e["Precision"]:.4f} |')
        lines.append('')
    return '\n'.join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['scaffold', 'random', 'both'],
                    default='both')
    ap.add_argument('--data_root_scaffold', default=SPLIT_DATA_ROOTS['scaffold'])
    ap.add_argument('--data_root_random',   default=SPLIT_DATA_ROOTS['random'])
    ap.add_argument('--out_root', default='results/baselines_classical')
    ap.add_argument('--output_md', default='baseline_classical_summary.md')
    args = ap.parse_args()

    out_root = Path(args.out_root)
    results_by_split = {}
    if args.split in ('scaffold', 'both'):
        results_by_split['scaffold'] = run_one_split(
            'scaffold', Path(args.data_root_scaffold), out_root)
    if args.split in ('random', 'both'):
        results_by_split['random'] = run_one_split(
            'random', Path(args.data_root_random), out_root)

    Path(args.output_md).write_text(render_markdown(results_by_split),
                                     encoding='utf-8')
    print(f'\n✓ Summary → {args.output_md}')
    print(f'✓ Per-seed prediction CSVs → {out_root}/<split>/predictions/')
    print(f'\nNext:')
    print(f'  # Cross-check with recompute_metrics_v42.py (should agree with '
          f'this script\'s numbers):')
    print(f'  python scripts/recompute_metrics_v42.py \\\n'
          f'      --pred_dir {out_root}/scaffold/predictions \\\n'
          f'      --split_name V3-scaffold-RF \\\n'
          f'      --glob \'test_pred_RF_seed*.csv\'')


if __name__ == '__main__':
    main()
