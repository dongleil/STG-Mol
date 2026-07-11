#!/usr/bin/env python3
"""
Pre-compute ChemBERTa embeddings for all molecules in V6/V5 datasets.

ChemBERTa (Chithrananda et al. 2020, arxiv 2010.09885): BERT-style
transformer trained on 77M SMILES from PubChem. Provides much richer
representations than Mol2Vec (300-d word2vec) — expected AUC boost
of 0.05-0.10 on limited-data QSAR tasks.

Uses HuggingFace: DeepChem/ChemBERTa-77M-MTR (multi-task-regression
head, which produces strong pretrained features).

Output: pickle file mapping canonical SMILES → 384-d embedding vector.
Loaded by train_v26.py at runtime (no online HF inference needed).

Usage:
    pip install transformers torch
    python scripts/expand/precompute_chemberta.py \\
        --csvs data/processed/nlrp3_v6/train.csv \\
               data/processed/nlrp3_v6/val.csv \\
               data/processed/nlrp3_v6/test.csv \\
               data/processed/nlrp3_v6/external_holdout.csv \\
        --output data/chemberta_embeddings_v6.pkl \\
        --model_name DeepChem/ChemBERTa-77M-MTR
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_SMI_CANDS = ['smiles_standardized', 'smiles', 'canonical_smiles', 'SMILES']


def _find_smi_col(cols):
    lc = {c.lower(): c for c in cols}
    for c in _SMI_CANDS:
        if c.lower() in lc:
            return lc[c.lower()]
    return None


def _batch(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csvs', nargs='+', required=True,
                    help='One or more CSVs to extract SMILES from.')
    ap.add_argument('--output', required=True)
    ap.add_argument('--model_name',
                    default='DeepChem/ChemBERTa-77M-MTR',
                    help='HuggingFace model ID. Alternatives: '
                         'seyonec/ChemBERTa-zinc-base-v1, '
                         'DeepChem/ChemBERTa-77M-MLM')
    ap.add_argument('--hf_endpoint',
                    default='https://hf-mirror.com',
                    help='HuggingFace mirror endpoint. Default '
                         'https://hf-mirror.com works from mainland China. '
                         'Set to https://huggingface.co for direct access.')
    ap.add_argument('--local_model_dir', default=None,
                    help='If set, load model from local directory instead '
                         'of downloading (useful for air-gapped environments).')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    # Configure HuggingFace endpoint for mirror access
    if args.hf_endpoint and 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = args.hf_endpoint
        print(f'Using HuggingFace endpoint: {args.hf_endpoint}')

    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        sys.exit('ERROR: pip install transformers')

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ---- Collect all SMILES --------------------------------------------
    print(f'\n[1/3] Collecting SMILES from {len(args.csvs)} CSVs...')
    all_smiles = set()
    for csv in args.csvs:
        p = Path(csv)
        if not p.exists():
            print(f'  ⚠ {csv} missing, skipping.')
            continue
        df = pd.read_csv(p, encoding='utf-8')
        smi_col = _find_smi_col(df.columns)
        if not smi_col:
            print(f'  ⚠ no SMILES column in {csv}')
            continue
        smis = df[smi_col].astype(str).tolist()
        all_smiles.update(smis)
        print(f'  {csv}: {len(smis)} SMILES')
    print(f'  → {len(all_smiles)} unique SMILES total')

    # ---- Load model ----------------------------------------------------
    src = args.local_model_dir if args.local_model_dir else args.model_name
    print(f'\n[2/3] Loading ChemBERTa model: {src}...')
    if args.local_model_dir and Path(args.local_model_dir).exists():
        print(f'  Loading from local directory (no HuggingFace access needed)')
    try:
        tokenizer = AutoTokenizer.from_pretrained(src)
        model = AutoModel.from_pretrained(src)
    except Exception as e:
        print(f'\n❌ Failed to load model: {e}')
        print(f'\nOptions:')
        print(f'  1. Try mirror: --hf_endpoint https://hf-mirror.com  (default)')
        print(f'  2. Try model:  --model_name seyonec/ChemBERTa-zinc-base-v1')
        print(f'  3. Manual download: git clone https://hf-mirror.com/'
              f'DeepChem/ChemBERTa-77M-MTR ./chemberta_local/')
        print(f'                     then --local_model_dir ./chemberta_local')
        print(f'  4. Or: set HF_HUB_OFFLINE=1 env var after first download.')
        sys.exit(1)
    model.to(args.device).eval()
    hidden_size = model.config.hidden_size
    print(f'  Hidden size (embedding dim): {hidden_size}')
    print(f'  Device: {args.device}')

    # ---- Embed --------------------------------------------------------
    print(f'\n[3/3] Encoding {len(all_smiles)} SMILES '
          f'in batches of {args.batch_size}...')
    smiles_list = sorted(all_smiles)
    embeddings = {}

    with torch.no_grad():
        for i, batch in enumerate(_batch(smiles_list, args.batch_size)):
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=512, return_tensors='pt')
            enc = {k: v.to(args.device) for k, v in enc.items()}
            out = model(**enc)
            # Mean-pool over token dimension (respecting attention mask)
            attn = enc['attention_mask'].unsqueeze(-1).float()
            hidden = out.last_hidden_state
            pooled = (hidden * attn).sum(1) / attn.sum(1).clamp(min=1e-6)
            pooled = pooled.cpu().numpy()
            for smi, emb in zip(batch, pooled):
                embeddings[smi] = emb.astype(np.float32)
            if (i + 1) % 10 == 0 or (i + 1) * args.batch_size >= len(smiles_list):
                done = min((i + 1) * args.batch_size, len(smiles_list))
                print(f'  {done}/{len(smiles_list)} embedded')

    with open(args.output, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'model_name': args.model_name,
            'hidden_size': hidden_size,
            'n_molecules': len(embeddings),
        }, f)
    print(f'\n✓ Saved {len(embeddings)} embeddings ({hidden_size}-d) to {args.output}')


if __name__ == '__main__':
    main()
