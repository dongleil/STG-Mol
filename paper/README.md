# Paper — STG-Mol manuscript sources

The versioned manuscript sources for the paper accompanying this codebase.
Everything here is **plain Python that emits a `.docx`** — no hidden state
beyond `references_v4.py`, no cloud tooling, so anyone who clones the repo
can reproduce every figure and number by running one script.

## Files

| File | Purpose |
|---|---|
| `build_paper_v4.py`        | Builds the **Chinese** manuscript v4.2 → `STG-Mol_论文_v4.2_中文.docx` |
| `build_paper_v4_en.py`     | Builds the **English** manuscript v4.2 → `STG-Mol_Paper_v4.2_English.docx` |
| `build_paper_v3.py`        | Shared `python-docx` helpers (`add_h1`, `add_table`, `add_formula`, …) inherited by v4.x builders |
| `references_v4.py`         | 78 canonical references — a Python list consumed by both builders |
| `STG-Mol_Paper_v4.2_English.docx` | Latest built English docx (regenerable) |
| `STG-Mol_论文_v4.2_中文.docx`     | Latest built Chinese docx (regenerable) |

## Build

```bash
pip install python-docx        # only runtime dependency
python paper/build_paper_v4_en.py   # → STG-Mol_Paper_v4.2_English.docx
python paper/build_paper_v4.py      # → STG-Mol_论文_v4.2_中文.docx
```

All output paths resolve relative to this directory (`__file__`), so the
scripts work from any clone location without editing.

## Numbers pipeline

Numbers in Tables 5.1a / 5.1b / 5.1c come from `scripts/recompute_metrics_v42.py`
(one directory up). That script reads per-seed prediction CSVs written by the
`v4.2` CSV-dump hook in `src/training/train_v26.py` and emits a paper-ready
Markdown snippet you can paste directly into these builders. Never hand-edit
the numbers into the docx — always update the builder Python.

### Reproducing the two evaluation protocols

**V3-scaffold (Table 5.1a, primary):**
```bash
# Data already lives at data/processed/nlrp3/{train,val,test}.csv (scaffold split)
python src/training/train_v26.py --config configs/train_v26_scaffold.yaml
# ~90 min per seed on RTX 4090, ~7.5 h total for 5 seeds
python scripts/recompute_metrics_v42.py \
    --pred_dir results/v26_v3_scaffold/<run-timestamp>/predictions \
    --split_name V3-scaffold \
    --output_md v42_metrics_v3_scaffold.md
# Paste values into paper/build_paper_v4_en.py Table 5.1a / 5.1c V3-scaffold column
```

**V3-random (Table 5.1b, reference):**
```bash
python src/training/train_v26.py --config configs/train_v26_v3_random.yaml
python scripts/recompute_metrics_v42.py \
    --pred_dir results/v26_v3_random/<run-timestamp>/predictions \
    --split_name V3-random \
    --output_md v42_metrics_v3_random.md
```

## Version history

| Version | Fixes | Status |
|---|---|---|
| v4.0 | Initial CBM-targeted draft | archived |
| v4.1 | 8 candidates, V3-random main experiment, AD-aware external validation | archived (had EF@1%=3.76 headline; withdrawn in v4.2) |
| **v4.2** | P0 fixes: dual scaffold+random reporting, primary metric = 5-seed mean±std, "well-calibrated OOD" reworded as AD-aware confidence, EF@1% removed (top-k=3 too noisy) → EF@5/10/20 with split-dependent upper bound | **current** |
