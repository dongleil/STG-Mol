# Data directory

This directory hosts the datasets used by STG-Mol. Raw data files are **not tracked by git** (see `.gitignore`); this README explains how to obtain and prepare them.

---

## Expected layout

```
data/
├── processed/
│   ├── nlrp3/                # Curated NLRP3 activity dataset (2,591 compounds)
│   │   ├── train.csv         # scaffold split, 80%
│   │   ├── val.csv           # scaffold split, 10%
│   │   ├── test.csv          # scaffold split, 10%
│   │   └── metadata.json     # split statistics & class distribution
│   └── zinc/                 # ZINC screening library subset
│       └── zinc_druglike.csv
├── raw/                      # Raw downloads (optional; kept for provenance)
│   ├── chembl_nlrp3.csv
│   ├── pubchem_nlrp3.csv
│   └── bindingdb_nlrp3.csv
├── external/                 # External benchmarks (optional)
│   ├── bace/
│   ├── hiv/
│   └── tox21/
└── candidates/               # Eight prioritised candidates
    ├── smiles.csv
    ├── docking_results.csv
    ├── md_trajectories/
    └── admet_predictions.csv
```

---

## NLRP3 activity dataset

Source databases:

| Source | Query | URL |
|---|---|---|
| ChEMBL v33 | `NLRP3` target search, `IC50` records | https://www.ebi.ac.uk/chembl/ |
| PubChem | BioAssay records tagged `NLRP3` | https://pubchem.ncbi.nlm.nih.gov/ |
| BindingDB | Target-search `NLRP3 (human)` | https://www.bindingdb.org/ |

Curation pipeline (see paper §4.1.1):

1. Deduplicate by InChI Key; keep geometric-mean IC50 for multi-entries.
2. SMILES validation and standardisation (RDKit).
3. Label as active if IC50 ≤ 1 μM, else inactive.
4. Augment negatives via **DUD-E** decoy generation (target-matched physicochemistry, distinct topology).
5. Split by **Murcko scaffold** (8 : 1 : 1).

Final dataset: 2,591 compounds (648 actives / 1,943 inactives, ratio ~1:3).

---

## ZINC screening library

- **Source**: [ZINC20](https://zinc20.docking.org/), *drug-like subset*.
- **Size**: 8,882,615 compounds.
- **Download** (example, using ZINC's tranch download tool):

  ```bash
  # Use ZINC20 tranch selection interface, then:
  wget -i zinc_urls.txt   # a list of subset URLs
  ```

- After Stage-0 rule-based filtering (Lipinski + Veber + PAINS + DILI), ~7.2 M compounds remain for Stage-1 model screening.

---

## Eight prioritised candidates

The complete data for the eight final NLRP3 candidates identified in this study — SMILES, docking scores, MD trajectory statistics, MMPBSA free energies, and ADMET predictions — are archived on **Zenodo** for permanent access:

> **DOI:** `10.xxxx/zenodo.xxxxxxx` *(to be inserted upon paper acceptance)*

---

## External benchmarks (optional)

If you wish to reproduce the cross-benchmark experiments in the paper Supplementary Information, download the following from [MoleculeNet](https://moleculenet.org/):

- BACE (classification; 1,513 molecules)
- HIV (classification; 41,127 molecules)
- Tox21 (multi-task classification; 7,831 molecules × 12 tasks)

Use the scaffold split provided by MoleculeNet for consistency.
