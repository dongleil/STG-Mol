# v4.3 Rerun Pipeline — Vina + MD + MMPBSA + ADMET + STG-Mol

End-to-end validation pipeline for the NLRP3 paper's Section 5.6, producing
paper Tables **5.6 – 5.10**. Runs entirely under WSL2 Ubuntu on a single
RTX 4090.

The pipeline addresses two specific failure modes noted in the v4.0/v4.1 draft:

1. Vina docking used a slightly mis-aligned box → **v4.3 re-docks against
   the CRID3-aligned centroid** in `7PZC` chain A (0.06 Å drift).
2. MD tables lacked evidence of convergence → **hardened equilibration
   protocol + adaptive extension** (see `mdp/*.mdp`).

## Prerequisites

- WSL2 Ubuntu 22.04, NVIDIA driver ≥ 535, CUDA 12.x visible in WSL.
- **conda envs** (see `scripts/md/environment.yml` for the base spec):
  - `nlrp3`     — python 3.10, rdkit, meeko, vina, acpype, gmx_MMPBSA, admet_ai, mdtraj
  - `gmx_cuda`  — GROMACS 2023+ compiled with `-DGMX_GPU=CUDA`
- **`gmx_MMPBSA`** installed inside the `nlrp3` env (pip). Verify with
  `gmx_MMPBSA --version` and `which gmx_MMPBSA`.
- **AmberTools** on PATH (needed by gmx_MMPBSA and acpype).
- Receptor PDB/PDBQT and top-2000 CSV placed at the paths listed in
  `config.yaml` under `receptor_pdb`, `receptor_pdbqt`, `input_csv`.

## Files in this directory

| File | Purpose |
|---|---|
| `config.yaml`             | Single source of truth for paths, box, MD length, etc. |
| `01_vina_top2000.sh`      | Stage A — Vina dock all 2000 candidates |
| `02_select_top8.py`       | Stage B — top-10 Vina + Tanimoto → 8 diverse |
| `03_run_md_8.sh`          | Stage C — GROMACS MD, 8 × 100 ns |
| `04_run_mmpbsa_8.sh`      | Stage D — gmx_MMPBSA, 200 frames per compound |
| `05_run_admet_full.py`    | Stage E — RDKit rules + admet-ai (41 props) |
| `06_score_v3random.py`    | Stage F — STG-Mol V3-random 5-seed re-score |
| `07_aggregate_tables.py`  | Stage G — assemble paper tables 5.6 – 5.10 |
| `smoke_test.sh`           | 1-2 h sanity check on compound 1 |
| `mdp/em1_steep.mdp`       | Steepest-descent minimisation |
| `mdp/em2_cg.mdp`          | Conjugate-gradient minimisation |
| `mdp/nvt.mdp`             | 100 ps NVT + POSRES |
| `mdp/npt_restr.mdp`       | 200 ps NPT + POSRES + Parrinello-Rahman |
| `mdp/npt_prod.mdp`        | 500 ps NPT unrestrained |
| `mdp/md_prod.mdp`         | 100 ns production |

## Full timeline (RTX 4090)

| Stage | Description | Per unit | ×N | Wall time |
|---|---|---|---|---|
| A | Vina 2000 ligands @ exhaust=32, 8 workers × 4 cpu | ~30 s | 2000 | **~15 h** (CPU-bound) |
| B | Select 8 via Tanimoto dedup | seconds | 1 | 5 min |
| C | MD 100 ns (setup + eq + prod) | ~8 h | 8 | **~64 h** (GPU) |
| D | MMPBSA on 200 frames | ~30 min | 8 | **~4 h** (CPU-bound) |
| E | admet-ai on 8 SMILES | ~5 min | 1 | 5 min |
| F | STG-Mol V3-random ensemble | ~2 min | 1 | 2 min |
| G | Aggregate tables | seconds | 1 | < 1 min |
| **Total** | | | | **~3.5 days** |

## Usage

```bash
# One-time env setup (see prerequisites)
conda activate nlrp3

# Stage A — Vina 2000
bash 01_vina_top2000.sh

# Stage B — pick 8
python3 02_select_top8.py

# Stage C — MD (put behind nohup on a long-lived tmux)
tmux new -s md
conda activate gmx_cuda   # then `nlrp3` for acpype — the script switches envs internally if needed
bash 03_run_md_8.sh 2>&1 | tee md_full.log
# detach with Ctrl-b d, reattach with `tmux attach -t md`

# Stage D — MMPBSA
bash 04_run_mmpbsa_8.sh

# Stage E — ADMET
python3 05_run_admet_full.py

# Stage F — STG-Mol V3-random re-score
python3 06_score_v3random.py

# Stage G — paper tables
python3 07_aggregate_tables.py
ls $(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['output_dir'])")/tables/
```

## Resume after interruption

All long-running stages are **idempotent** and **resumable**:

- `01_vina_top2000.sh` — writes `scored.csv` incrementally; a re-run
  reads the CSV, computes the set of already-docked row indices, and
  only submits the remaining ones.
- `03_run_md_8.sh` — each stage inside a compound writes `.STAGENAME.done`
  before proceeding. On restart, completed stages are skipped. A per-
  compound `all.done` marker skips finished compounds entirely.
  Production restarts use `gmx mdrun -cpi md.cpt` (append mode) so
  crash recovery is native.
- `04_run_mmpbsa_8.sh` — skips compounds that already have
  `mmpbsa_summary.json`.
- `05_run_admet_full.py`, `06_score_v3random.py`, `07_aggregate_tables.py`
  — safe to re-run; overwrite their outputs.

To force re-execution of a stage in one compound: remove the corresponding
`.done` marker (e.g. `rm md/comp_3/.md_run.done`).

## Output files → paper tables mapping

| Output path (relative to `output_dir`) | Feeds paper table |
|---|---|
| `vina2000/scored.csv`                 | Stage A audit trail |
| `candidates_v4.3/top8.csv`            | **Table 5.6** input |
| `md/comp_i/summary.json`              | **Table 5.8** input |
| `md/comp_i/rmsd_bb.xvg`, `rmsd_lig.xvg`, `rmsf.xvg`, `hbonds.xvg` | Figures 5.x |
| `md/comp_i/mmpbsa_summary.json`       | **Table 5.9** input |
| `admet/admet_full.csv`                | **Table 5.10** (full CSV) |
| `admet/table_5_10.md`                 | **Table 5.10** (markdown) |
| `v3random_scores/v3random_scores.json`| **Table 5.7** input |
| `tables/table_5_{6..10}.md`           | Direct paste into paper |

## Troubleshooting

**`acpype fails on unusual atoms`**
`acpype` chokes on halogens or unusual protonation. Try:

```bash
acpype -i pose_comp_i.pdb -b lig -o gmx -a gaff2 -c bcc -n 0
# or add explicit charge:
acpype -i pose_comp_i.pdb -b lig -o gmx -a gaff2 -c bcc -n -1
```

If GAFF still fails, switch to OpenFF (`--force_field openff`) or Sobtop.
Delete `md/comp_i/.acpype.done` to force re-parametrisation on retry.

**`gmx_MMPBSA fails on PB`**
The Poisson-Boltzmann step is more fragile than GB (fillratio, grid
size). The pipeline **also** computes GB (igb=5) in the same input
file — GB-only ΔG estimates are still valid and are reported in the
same `mmpbsa_summary.json`. Cross-check with your Kd order-of-magnitude
expectations before discarding a compound.

**`RMSD not converged after 200 ns`**
The adaptive extension in `03_run_md_8.sh` already tries 200 ns. If
the compound is still drifting > 0.5 Å in the last 30 ns:

1. Inspect `rmsd_bb.xvg` and `rmsf.xvg` for a large-scale conformational
   change (often an unrestrained loop or terminus). Trim loops before
   pdb2gmx and re-run.
2. Consider that the pose is unstable: this **is** a valid result to
   report — the compound docks reasonably but does not maintain the
   binding mode. Flag it explicitly in Table 5.8 (`converged = no`).
3. As a last resort, run 3 replicas at 100 ns each with different
   `gen_seed` values and pool statistics.

**`gmx pdb2gmx complains about missing atoms`**
Add `-ignh` (already in the script) and clean up the receptor with
`pdbfixer` or `reduce` before running.

**`Index groups Protein_LIG / Water_and_ions not created`**
The driver falls back to a Python-generated `index.ndx` that parses
residue names directly from `em2.gro`. This handles unusual atom
numbering. If your ligand residue name is not `LIG` (rename in
`lig.itp`), update the fallback block.

**`Not enough diverse compounds after Tanimoto filter`**
Relax the cutoff (`config.yaml : tanimoto_dedup_cutoff`) from 0.7 → 0.8
or increase `select_top_n_by_vina` above 10.
