#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_run_admet_full.py — full ADMET profile for the 8 selected candidates.

Combines:
  1. RDKit rules  (Lipinski, QED, PAINS, SA, LogP, MW, HBD/HBA, TPSA, rot. bonds)
  2. admet-ai     (41 ADMET properties from the TDC-Admet leaderboard model)

Outputs:
  {output_dir}/admet/admet_full.csv  — one row per compound, wide format
  {output_dir}/admet/table_5_10.md   — paper-ready markdown table

Usage:
  conda activate nlrp3
  python3 scripts/rerun/05_run_admet_full.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import (
    AllChem, Descriptors, Lipinski, Crippen, QED, rdMolDescriptors,
)
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# SA scorer (RDKit contrib)
try:
    from rdkit.Chem import RDConfig
    sys.path.append(str(Path(RDConfig.RDContribDir) / 'SA_Score'))
    import sascorer  # noqa: E402
    HAS_SA = True
except Exception:
    HAS_SA = False

# PAINS filter (build once)
_p = FilterCatalogParams()
_p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS = FilterCatalog(_p)


def load_config():
    return yaml.safe_load(
        (Path(__file__).resolve().parent / 'config.yaml').read_text())


def rdkit_properties(smiles: str) -> dict:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return {'error': 'bad_smiles'}
    mw   = Descriptors.MolWt(m)
    logp = Crippen.MolLogP(m)
    hbd  = Lipinski.NumHDonors(m)
    hba  = Lipinski.NumHAcceptors(m)
    tpsa = rdMolDescriptors.CalcTPSA(m)
    rotb = Lipinski.NumRotatableBonds(m)
    qed  = QED.qed(m)
    sa   = sascorer.calculateScore(m) if HAS_SA else float('nan')
    lipi_ok = (mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10)
    pains_hit = _PAINS.HasMatch(m)
    return dict(
        MW=round(mw, 2),
        LogP=round(logp, 2),
        HBD=int(hbd),
        HBA=int(hba),
        TPSA=round(tpsa, 2),
        RotBonds=int(rotb),
        QED=round(qed, 3),
        SA=round(sa, 3) if HAS_SA else None,
        Lipinski_OK=bool(lipi_ok),
        PAINS_hit=bool(pains_hit),
    )


def run_admet_ai(smiles_list):
    """Run admet-ai's 41-column predictor. Returns DataFrame indexed by SMILES."""
    try:
        from admet_ai import ADMETModel
    except ImportError:
        print('! admet-ai not installed → pip install admet-ai', file=sys.stderr)
        return None
    model = ADMETModel()
    return model.predict(smiles_list)  # DataFrame, one row per input SMILES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', help='Override top8.smi path')
    ap.add_argument('--output_csv', help='Override output CSV path')
    ap.add_argument('--output_md',  help='Override output MD path')
    args = ap.parse_args()

    cfg = load_config()
    out = Path(cfg['output_dir'])
    admet_dir = out / 'admet'
    admet_dir.mkdir(parents=True, exist_ok=True)

    top8_smi = Path(args.input) if args.input else out / 'candidates_v4.3' / 'top8.smi'
    if not top8_smi.exists():
        raise FileNotFoundError(f'top8.smi not found at {top8_smi}')

    smiles, names = [], []
    for ln in top8_smi.read_text().splitlines():
        parts = ln.strip().split()
        if not parts:
            continue
        smiles.append(parts[0])
        names.append(parts[1] if len(parts) > 1 else f'compound_{len(smiles)}')

    print(f'Read {len(smiles)} SMILES from {top8_smi}')

    # 1. RDKit properties
    rows = []
    for i, (name, smi) in enumerate(zip(names, smiles), 1):
        r = {'compound_idx': i, 'compound_name': name, 'smiles': smi}
        r.update(rdkit_properties(smi))
        rows.append(r)
    df_rd = pd.DataFrame(rows)

    # 2. admet-ai
    print('Running admet-ai (may take 2-5 min for the first call)...')
    df_ai = run_admet_ai(smiles)
    if df_ai is not None:
        df_ai = df_ai.reset_index().rename(columns={'index': 'smiles'})
        df = df_rd.merge(df_ai, on='smiles', how='left')
    else:
        print('! Skipping admet-ai — only RDKit rules will be in the output')
        df = df_rd

    csv_path = Path(args.output_csv) if args.output_csv else admet_dir / 'admet_full.csv'
    md_path  = Path(args.output_md)  if args.output_md  else admet_dir / 'table_5_10.md'
    df.to_csv(csv_path, index=False)
    print(f'Wrote {csv_path}  ({df.shape[0]} rows × {df.shape[1]} cols)')

    # 3. Paper-ready markdown: RDKit summary + 5 admet-ai highlights
    highlight = ['compound_idx', 'MW', 'LogP', 'HBD', 'HBA', 'TPSA',
                 'QED', 'SA', 'Lipinski_OK', 'PAINS_hit']
    for col in ('hERG', 'HIA_Hou', 'BBB_Martins', 'Bioavailability_Ma',
                'DILI', 'Ames', 'Carcinogens_Lagunin', 'Skin_Reaction'):
        if col in df.columns:
            highlight.append(col)

    md_cols = [c for c in highlight if c in df.columns]
    md = df[md_cols].copy()
    with open(md_path, 'w') as f:
        f.write('# Table 5.10 — ADMET profile of the 8 candidates\n\n')
        f.write(md.to_markdown(index=False, floatfmt='.3f'))
        f.write('\n\nFull 41-property CSV: `admet_full.csv`\n')
    print(f'Wrote {md_path}')
    print('\nNext: python3 scripts/rerun/06_score_v3random.py')


if __name__ == '__main__':
    main()
