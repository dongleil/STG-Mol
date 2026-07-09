#!/usr/bin/env python3
"""
Add multi-task ADMET labels to train/val/test.csv.

Generates 5 binary labels per molecule using RDKit rules:
  * is_lipinski_ok   -- Lipinski's rule of five compliant
  * is_druglike_qed  -- QED > 0.5
  * is_low_pains     -- No PAINS structural alert
  * is_easy_synth    -- SA score < 5 (RDKit synthetic accessibility)
  * is_moderate_logp -- 0 <= LogP <= 5

Usage:
    python scripts/add_admet_labels.py \\
        --input_dir data/processed/nlrp3 \\
        --output_dir data/processed/nlrp3

Outputs (in-place or new dir):
    train.csv, val.csv, test.csv
    (each with 5 new columns: admet_lipinski, admet_qed, admet_pains,
                              admet_sa, admet_logp)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import (
    Descriptors, Lipinski, Crippen, QED,
    rdMolDescriptors,
)
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# --------------------------------------------------------------------------
# SA (Synthetic Accessibility) score — RDKit Contrib sascorer
# --------------------------------------------------------------------------
try:
    import sys as _sys
    from rdkit.Chem import RDConfig
    _sys.path.append(str(Path(RDConfig.RDContribDir) / 'SA_Score'))
    import sascorer  # noqa: E402
    HAS_SA = True
except Exception:
    HAS_SA = False
    print("Warning: RDKit sascorer not available, using QED score as SA proxy.",
          file=sys.stderr)


# --------------------------------------------------------------------------
# Build PAINS filter catalog (once)
# --------------------------------------------------------------------------
_pains_params = FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS_CATALOG = FilterCatalog(_pains_params)


def compute_admet_labels(smiles: str) -> dict:
    """
    Compute 5 binary ADMET labels for a SMILES string.
    Returns dict with keys: admet_lipinski / admet_qed / admet_pains /
                            admet_sa / admet_logp
    All labels are int (0 or 1). Failure returns all zeros.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'admet_lipinski': 0, 'admet_qed': 0, 'admet_pains': 0,
                    'admet_sa': 0, 'admet_logp': 0}

        # 1) Lipinski's Rule of Five
        mw   = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd  = Lipinski.NumHDonors(mol)
        hba  = Lipinski.NumHAcceptors(mol)
        is_lipinski = int(mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)

        # 2) QED (drug-likeness)
        qed_val = QED.qed(mol)
        is_druglike = int(qed_val > 0.5)

        # 3) PAINS (0 = no alert, 1 = alert). We want NO alert => 1.
        is_low_pains = int(not _PAINS_CATALOG.HasMatch(mol))

        # 4) SA score (< 5 = easy). Fall back to QED if sascorer unavailable.
        if HAS_SA:
            sa = sascorer.calculateScore(mol)
            is_easy_synth = int(sa < 5.0)
        else:
            is_easy_synth = is_druglike  # proxy fallback

        # 5) LogP moderate (0 to 5 = balanced)
        is_moderate_logp = int(0.0 <= logp <= 5.0)

        return {
            'admet_lipinski': is_lipinski,
            'admet_qed':      is_druglike,
            'admet_pains':    is_low_pains,
            'admet_sa':       is_easy_synth,
            'admet_logp':     is_moderate_logp,
        }
    except Exception:
        return {'admet_lipinski': 0, 'admet_qed': 0, 'admet_pains': 0,
                'admet_sa': 0, 'admet_logp': 0}


def _detect_smiles_col(df: pd.DataFrame) -> str:
    """Auto-detect SMILES column name."""
    candidates = ['smiles', 'smiles_standardized', 'canonical_smiles', 'SMILES']
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f'No SMILES column found. Available: {df.columns.tolist()}')


def process_csv(in_path: Path, out_path: Path) -> None:
    df = pd.read_csv(in_path)
    smiles_col = _detect_smiles_col(df)
    print(f'  Processing {in_path.name}: {len(df)} molecules  '
          f'(smiles column = "{smiles_col}")')

    labels = [compute_admet_labels(s) for s in df[smiles_col]]
    labels_df = pd.DataFrame(labels)

    # Concatenate (or overwrite existing admet_* columns)
    for c in ['admet_lipinski', 'admet_qed', 'admet_pains',
              'admet_sa', 'admet_logp']:
        if c in df.columns:
            df = df.drop(columns=[c])
    df = pd.concat([df, labels_df], axis=1)

    df.to_csv(out_path, index=False)

    # Print label distribution
    print(f'    -> {out_path.name}: label distribution:')
    for c in ['admet_lipinski', 'admet_qed', 'admet_pains',
              'admet_sa', 'admet_logp']:
        pos_ratio = df[c].mean()
        print(f'       {c:20s} pos_ratio = {pos_ratio:.3f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input_dir',  required=True,
                    help='Directory containing train.csv / val.csv / test.csv')
    ap.add_argument('--output_dir', default=None,
                    help='Output directory (defaults to --input_dir, i.e. in-place)')
    args = ap.parse_args()

    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ['train.csv', 'val.csv', 'test.csv']:
        in_path  = in_dir / name
        out_path = out_dir / name
        if not in_path.exists():
            print(f'  Skipping {name}: not found in {in_dir}')
            continue
        process_csv(in_path, out_path)

    print('\nDone.')


if __name__ == '__main__':
    main()
