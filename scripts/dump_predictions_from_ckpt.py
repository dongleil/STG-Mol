#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dump_predictions_from_ckpt.py — rescue script.

Reloads saved .pt checkpoints from a prior v26 training run and re-runs
inference on the val / test sets, writing per-seed prediction CSVs in the
naming convention expected by scripts/recompute_metrics_v42.py.

Use this when the original training was performed before the v4.2 CSV-dump
hook was added to train_v26.py (i.e. runs with .pt but no predictions/*.csv).

Usage:
    python scripts/dump_predictions_from_ckpt.py \\
        --config configs/train_v26_v3_random.yaml \\
        --run_dir results/v26_v3_random/v26_v3_random_5seeds_20260713_193347

The script auto-detects saved .pt files under <run_dir>/models/ and writes
CSVs under <run_dir>/predictions/. Fusion mode is inferred from the .pt
filename (model_<fusion_tag>_seed<seed>.pt).
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.training.train_v26 import (  # noqa: E402
    MultiModalFusionNet, MultiModalMoleculeDataset, collate_multimodal,
    Mol2VecFeaturizer, smart_read_csv,
)


CKPT_PATTERN = re.compile(r'model_(?P<fusion>[^\.]+?)_seed(?P<seed>\d+)\.pt$')


def load_config(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_featurizer(config, train_df, val_df, test_df, fusion_mode):
    """Return (mol2vec_train, mol2vec_val, mol2vec_test) or (None, None, None)."""
    if '1D' not in fusion_mode:
        return None, None, None

    # ---- ChemBERTa branch (as in train_v26.py) ----
    chemberta_pkl = config.get('mol2vec', {}).get('chemberta_pkl')
    if chemberta_pkl and Path(chemberta_pkl).exists():
        import pickle
        with open(chemberta_pkl, 'rb') as f:
            data = pickle.load(f)
        emb, hidden = data['embeddings'], data['hidden_size']

        def featurize(smiles):
            out = np.zeros((len(smiles), hidden), dtype=np.float32)
            for i, s in enumerate(smiles):
                if emb.get(s) is not None:
                    out[i] = emb[s]
            return out

        # Match train_v26.py side-effect: override mol2vec dim
        config.setdefault('model', {})['mol2vec_dim'] = hidden
        config.setdefault('mol2vec', {})['embedding_dim'] = hidden
        return (featurize(train_df['smiles'].tolist()),
                featurize(val_df['smiles'].tolist()),
                featurize(test_df['smiles'].tolist()))

    # ---- Mol2Vec branch ----
    mol2vec_cfg = config.get('mol2vec', {})
    featurizer = Mol2VecFeaturizer(
        model_path=mol2vec_cfg.get('model_path', 'data/mol2vec_model.pkl'),
        embedding_dim=mol2vec_cfg.get('embedding_dim', 300),
        radius=mol2vec_cfg.get('radius', 1))
    return (featurizer.featurize(train_df['smiles'].tolist()),
            featurizer.featurize(val_df['smiles'].tolist()),
            featurizer.featurize(test_df['smiles'].tolist()))


def build_loaders(config, fusion_mode):
    train_df = smart_read_csv(Path(config['data']['train_path']))
    val_df   = smart_read_csv(Path(config['data']['val_path']))
    test_df  = smart_read_csv(Path(config['data']['test_path']))

    m1d_train, m1d_val, m1d_test = build_featurizer(
        config, train_df, val_df, test_df, fusion_mode)

    admet_cols = ['admet_lipinski', 'admet_qed', 'admet_no_pains',
                  'admet_sa', 'admet_logp']
    has_admet = all(c in train_df.columns for c in admet_cols)
    a_val  = val_df[admet_cols].values  if has_admet else None
    a_test = test_df[admet_cols].values if has_admet else None

    cutoff = config.get('model', {}).get('cutoff', 8.0)
    val_ds = MultiModalMoleculeDataset(
        val_df['smiles'].tolist(),  val_df['label'].values,
        m1d_val,  fusion_mode, cutoff, admet_labels=a_val)
    test_ds = MultiModalMoleculeDataset(
        test_df['smiles'].tolist(), test_df['label'].values,
        m1d_test, fusion_mode, cutoff, admet_labels=a_test)

    bs = config.get('training', {}).get('batch_size', 32)
    val_loader  = DataLoader(val_ds,  batch_size=bs,
                              collate_fn=collate_multimodal, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=bs,
                              collate_fn=collate_multimodal, num_workers=0)
    return val_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    import torch.nn.functional as F
    model.eval()
    probs, labels = [], []
    for batch in loader:
        kwargs = {'mol2vec_feat': None, 'graph_2d': None, 'graph_3d': None}
        for key in ('mol2vec', 'graph_2d', 'graph_3d'):
            if key in batch:
                target_key = 'mol2vec_feat' if key == 'mol2vec' else key
                kwargs[target_key] = batch[key].to(device)
        result = model(**kwargs)
        p = F.softmax(result['logits'], dim=1)[:, 1].cpu().numpy()
        probs.extend(p.tolist())
        labels.extend(batch['labels'].numpy().tolist())
    return np.array(probs), np.array(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',  required=True, type=Path)
    ap.add_argument('--run_dir', required=True, type=Path,
                    help='Directory holding a `models/` subdir with .pt files.')
    ap.add_argument('--out_dir', type=Path, default=None,
                    help='Where to write CSVs. Defaults to <run_dir>/predictions.')
    ap.add_argument('--device',  default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    print(f'Device: {device}')

    models_dir = args.run_dir / 'models'
    if not models_dir.exists():
        sys.exit(f'✗ {models_dir} does not exist')
    ckpts = sorted(models_dir.glob('model_*_seed*.pt'))
    if not ckpts:
        sys.exit(f'✗ No .pt files under {models_dir}')

    out_dir = args.out_dir or (args.run_dir / 'predictions')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output → {out_dir}')

    # Group by fusion tag; each group gets its own loader (built once).
    groups = {}
    for c in ckpts:
        m = CKPT_PATTERN.search(c.name)
        if not m:
            print(f'  ⚠ skip {c.name}: unrecognised naming')
            continue
        groups.setdefault(m.group('fusion'), []).append((int(m.group('seed')), c))

    for fusion_tag, items in groups.items():
        fusion_mode = fusion_tag.replace('_', '+')
        print(f'\n=== fusion_mode = {fusion_mode} ({len(items)} seeds) ===')
        val_loader, test_loader = build_loaders(config, fusion_mode)

        model_config = config.get('model', {}).copy()
        model_config['fusion_mode'] = fusion_mode
        model_config['mol2vec_dim'] = config.get('mol2vec', {}).get('embedding_dim', 300)

        seed_test_scores, seed_val_scores = [], []
        y_test_ref, y_val_ref = None, None

        for seed, ckpt in sorted(items):
            model = MultiModalFusionNet(model_config).to(device)
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state)
            print(f'  · seed {seed}: loaded {ckpt.name}')

            test_p, test_l = evaluate(model, test_loader,  device)
            val_p,  val_l  = evaluate(model, val_loader,   device)

            pd.DataFrame({'label': test_l.astype(int),
                          'pred_prob': test_p.astype(float)}
                        ).to_csv(out_dir / f'test_pred_{fusion_tag}_seed{seed}.csv',
                                  index=False, float_format='%.6f')
            pd.DataFrame({'label': val_l.astype(int),
                          'pred_prob': val_p.astype(float)}
                        ).to_csv(out_dir / f'val_pred_{fusion_tag}_seed{seed}.csv',
                                  index=False, float_format='%.6f')

            seed_test_scores.append(test_p)
            seed_val_scores.append(val_p)
            if y_test_ref is None:
                y_test_ref, y_val_ref = test_l, val_l

        # Ensemble
        ens_test = np.mean(seed_test_scores, axis=0)
        ens_val  = np.mean(seed_val_scores,  axis=0)
        pd.DataFrame({'label': y_test_ref.astype(int),
                      'pred_prob': ens_test}
                    ).to_csv(out_dir / f'test_pred_{fusion_tag}_ensemble.csv',
                              index=False, float_format='%.6f')
        pd.DataFrame({'label': y_val_ref.astype(int),
                      'pred_prob': ens_val}
                    ).to_csv(out_dir / f'val_pred_{fusion_tag}_ensemble.csv',
                              index=False, float_format='%.6f')
        print(f'  ✓ wrote {2 * len(items) + 2} CSVs for fusion={fusion_tag}')

    print(f'\n✓ Done. Next:\n'
          f'  python scripts/recompute_metrics_v42.py --pred_dir {out_dir} '
          f'--split_name <e.g. V3-random>')


if __name__ == '__main__':
    main()
