# STG-Mol

**An AI-Driven Multi-Modal Virtual Screening Framework Integrating Sequence, Topology, and Geometry for NLRP3 Inhibitor Discovery**

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.12%2B-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
  <img alt="Status" src="https://img.shields.io/badge/Status-Under_Review-orange">
</p>

STG-Mol is an adaptive multi-modal deep-learning framework for NLRP3 inhibitor discovery. It jointly encodes **1D sequence semantics** (Mol2Vec), **2D topological graphs** (D-MPNN), and **3D geometric conformers** (SchNet), and performs **sample-level dynamic modality fusion** via a cross-modal Transformer with a learnable query vector. Two dedicated regularisation mechanisms—**Soft-Bound Balance Loss** and **Modality Dropout**—effectively mitigate modality collapse. A **dual-precision cascaded virtual-screening architecture** further extends the framework to library sizes of tens of millions of compounds.

Applied to 8.8 million ZINC compounds, STG-Mol identifies **eight novel NLRP3 candidate compounds**, each supported by a complete multi-level computational evidence chain: molecular docking, 100 ns molecular dynamics simulation, MMPBSA free-energy analysis, and ADMET evaluation.

---

## 📖 Paper

**Title.** STG-Mol: An AI-Driven Multi-Modal Virtual Screening Framework Integrating Sequence, Topology, and Geometry for NLRP3 Inhibitor Discovery

**Status.** Submitted to *Computers in Biology and Medicine* (2026).

---

## ✨ Features

- **Tri-modal molecular representation** — Mol2Vec (1D) + D-MPNN (2D) + SchNet (3D)
- **Sample-adaptive fusion** — Cross-modal Transformer with a learnable query vector, allowing per-molecule dynamic modality weighting
- **Anti-collapse regularisation** — Soft-bound balance loss + modality dropout, effective on small-sample targets
- **Cascaded virtual screening** — Two-stage precision–throughput trade-off, ~12× speed-up over single-stage full-modality screening
- **Multi-level validation pipeline** — Molecular docking (AutoDock Vina) + Molecular dynamics (GROMACS) + MMPBSA + ADMET
- **Fully reproducible** — Deterministic seeds, YAML-driven configuration, one-command experiments

---

## 🛠 Installation

### Prerequisites

- Python ≥ 3.9
- CUDA ≥ 11.6 (recommended for training; CPU works but is slow)
- One NVIDIA GPU with ≥ 16 GB memory (A100 / V100 / RTX 3090 / RTX 4090)

### Setup

```bash
# 1. Clone
git clone https://github.com/<your-username>/STG-Mol.git
cd STG-Mol

# 2. Create environment (conda recommended)
conda create -n stgmol python=3.10 -y
conda activate stgmol

# 3. Install PyTorch (choose your CUDA version)
pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install PyTorch Geometric
pip install torch-geometric==2.4.0
pip install pyg-lib torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# 5. Install the remaining dependencies
pip install -r requirements.txt
```

Detailed environment troubleshooting is provided in [docs/SETUP.md](docs/SETUP.md).

---

## 🚀 Quick Start

### 1. Prepare data

Place the NLRP3 dataset under `data/processed/nlrp3/`:

```
data/processed/nlrp3/
├── train.csv    # columns: smiles, label
├── val.csv
└── test.csv
```

See [data/README.md](data/README.md) for download and preprocessing instructions.

### 2. Train the main model

```bash
python src/training/train.py --config configs/train_main.yaml
```

Approximate wall-clock: ~6 h on a single A100 (550 epochs).

### 3. Run ablation experiments

```bash
bash scripts/run_ablation.sh   # 6 modality combinations + 4 regularisation ablations
```

### 4. Reproduce with 5 random seeds

```bash
bash scripts/run_5seeds.sh     # seeds {42, 123, 2024, 3407, 7}
```

### 5. Virtual screening on ZINC

```bash
python src/utils/virtual_screening.py --config configs/screening/zinc.yaml
```

### 6. Multi-level validation of candidates

```bash
python src/evaluation/validation_pipeline.py --config configs/validation.yaml
```

---

## 📁 Repository Layout

```
STG-Mol/
├── configs/
│   ├── train_main.yaml            # Main training configuration (v4.4)
│   └── ablation/                  # Ablation study configurations
│       ├── modality_*.yaml        # Single- / dual-modality ablations
│       ├── fusion_concat.yaml     # Fusion strategy ablation
│       └── no_*.yaml              # Regularisation ablations
├── src/
│   ├── training/
│   │   └── train.py               # STG-Mol training pipeline
│   ├── utils/
│   │   └── virtual_screening.py   # Two-stage cascaded screening
│   └── evaluation/
│       ├── validation_pipeline.py # Docking + MD + MMPBSA + ADMET
│       ├── validation_report.py   # Result aggregator & report generator
│       ├── analyze_best_model.py  # Best-model diagnostics
│       └── compare_experiments.py # Cross-experiment comparison
├── scripts/
│   ├── run_5seeds.sh              # 5-seed main experiment
│   ├── run_ablation.sh            # One-shot ablation suite
│   ├── screen_zinc.sh             # ZINC screening pipeline
│   ├── lipinski_analysis.py       # Drug-likeness analysis
│   ├── scaffold_analyzer_improved.py
│   └── Visualization.py
├── data/
│   └── README.md                  # Data download instructions
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

---

## 📊 Reproducing Paper Results

| Table / Figure | Command | Reference config |
|---|---|---|
| Table 5.1 (main comparison) | `bash scripts/run_5seeds.sh` | `configs/train_main.yaml` |
| Table 5.2 (modality ablation) | `bash scripts/run_ablation.sh --modality` | `configs/ablation/modality_*.yaml` |
| Table 5.3 (fusion ablation) | `bash scripts/run_ablation.sh --fusion` | `configs/ablation/fusion_*.yaml` |
| Table 5.4 (regularisation ablation) | `bash scripts/run_ablation.sh --reg` | `configs/ablation/no_*.yaml` |
| Section 5.4 (large-scale VS) | `bash scripts/screen_zinc.sh` | `configs/screening/zinc.yaml` |
| Section 5.5 (multi-level validation) | `python src/evaluation/validation_pipeline.py` | `configs/validation.yaml` |

---

## 🧬 Data Availability

- **Training / validation / test set**: derived from ChEMBL v33, PubChem, and BindingDB. 2 591 compounds after preprocessing; scaffold split at 8:1:1.
- **ZINC library**: the drug-like subset of ZINC20, 8 882 615 compounds.
- **Eight prioritised candidates**: complete SMILES, docking scores, MD trajectories, and ADMET predictions are provided in the paper Supporting Information and archived on Zenodo (DOI: `10.xxxx/zenodo.xxxxxxx`).

Note: raw data files are not tracked in this repository (see `.gitignore`). Refer to [data/README.md](data/README.md) for download.

---

## 📄 Citation

If you use STG-Mol in your research, please cite:

```bibtex
@article{stg_mol_2026,
  title   = {STG-Mol: An AI-Driven Multi-Modal Virtual Screening Framework
             Integrating Sequence, Topology, and Geometry for NLRP3 Inhibitor Discovery},
  author  = {Dong, Leilei and Ma, Zhenhe and Yang, Yanqiu and others},
  journal = {Computers in Biology and Medicine},
  year    = {2026},
  note    = {Under review}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is also provided.

---

## 📝 License

This project is released under the [MIT License](LICENSE).

---

## 🤝 Acknowledgements

- ZINC database — https://zinc20.docking.org/
- ChEMBL, PubChem, BindingDB — public bioactivity data sources
- RDKit, PyTorch Geometric, GROMACS, AutoDock Vina — open-source scientific libraries that this work depends on

---

## 📮 Contact

Questions and issues are welcome via GitHub Issues, or contact the corresponding author (see paper).
