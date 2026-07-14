#!/usr/bin/env python3
"""
Predict full 41-property ADMET profile for candidate compounds using
admet-ai (Broad Institute, Wu et al. 2023).

admet-ai wraps Chemprop-based models trained on TDC ADMET benchmarks
(Therapeutics Data Commons) — the current academic standard for
open-source ADMET prediction. Coverage includes:

  Absorption:  Caco-2, HIA, F20% bioavailability, Pgp inhibitor/substrate,
               Lipophilicity, PAMPA, Hydration Free Energy, Solubility
  Distribution: BBB, PPB, VDss
  Metabolism:  CYP450 (1A2/2C9/2C19/2D6/3A4) inhibitor + substrate
  Excretion:   Total clearance (Hepatocyte, Microsome), T1/2
  Toxicity:    hERG, AMES, DILI, Carcinogenicity, Skin Sensitisation,
               LD50 (rat), NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase,
               NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5,
               SR-HSE, SR-MMP, SR-p53

Install:
    pip install admet-ai

Usage:
    python scripts/expand/predict_admet_ai.py \\
        --input_csv data/candidates_8.csv \\
        --output_csv predictions_8_candidates/admet_full.csv
"""
import argparse
import os
import sys
from pathlib import Path

# Configure HF mirror before importing admet-ai (which uses transformers)
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print(f'Using HuggingFace endpoint: {os.environ["HF_ENDPOINT"]}')

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_csv', required=True,
                    help='CSV with a smiles column and optional name column.')
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--smiles_column', default=None,
                    help='SMILES column name (auto-detect if unset).')
    args = ap.parse_args()

    # Import admet-ai here so the mirror env var is already set
    try:
        from admet_ai import ADMETModel
    except ImportError:
        print('\n❌ admet-ai not installed. Install:')
        print('   pip install admet-ai')
        print('\nIf HuggingFace mirror is needed (mainland China):')
        print('   $env:HF_ENDPOINT="https://hf-mirror.com"; pip install admet-ai')
        sys.exit(1)

    # Load input
    df = pd.read_csv(args.input_csv, encoding='utf-8')
    smi_col = args.smiles_column
    if not smi_col:
        for c in ['smiles', 'smiles_standardized', 'canonical_smiles', 'SMILES']:
            if c in df.columns:
                smi_col = c
                break
    if not smi_col:
        sys.exit(f'No SMILES column in {args.input_csv}. '
                 f'Columns: {df.columns.tolist()}')
    smiles_list = df[smi_col].astype(str).tolist()
    print(f'\nInput: {len(smiles_list)} molecules from {args.input_csv}')

    print('\nLoading admet-ai pretrained models '
          '(first run downloads ~500 MB of Chemprop weights)...')
    model = ADMETModel()

    print('\nPredicting 41 ADMET properties...')
    preds = model.predict(smiles=smiles_list)
    # preds is a DataFrame indexed by SMILES

    # Prepend name/idx if present
    out = preds.reset_index()
    if 'name' in df.columns:
        # Match by order (admet-ai preserves smiles order)
        out.insert(0, 'name', df['name'].tolist()[:len(out)])
    if 'idx' in df.columns:
        out.insert(0, 'idx', df['idx'].tolist()[:len(out)])

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False, encoding='utf-8')
    print(f'\n✓ Wrote {len(out)} rows × {len(out.columns)} columns '
          f'to {args.output_csv}')

    # Print a compact summary of key toxicity endpoints
    key_cols = [c for c in out.columns
                if any(k in c.lower() for k in
                       ['dili', 'herg', 'ames', 'clint', 'bioavail',
                        'caco', 'hia', 'bbb', 'ppb', 'ld50',
                        'lipophilicity', 'solubility'])]
    if key_cols:
        print(f'\n=== Key ADMET endpoints (first {min(len(out), 8)} molecules) ===')
        show_cols = ['name'] if 'name' in out.columns else ['idx']
        show_cols += key_cols[:12]
        print(out[show_cols].head(8).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
