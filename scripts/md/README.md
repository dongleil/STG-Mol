# Multi-Level Validation Pipeline — MD + MMPBSA + ADMET

Rerun the paper's Section 5.6 physical/computational validation for the
eight prioritised NLRP3 candidates, targeting **paper Tables 5.8 / 5.9 / 5.10**.

## Why this rerun exists (v4.2)

The original v4.0/v4.1 draft reported RMSD < 3 Å as a one-line note, without
per-compound tables and — more importantly — **without evidence that MD
converged**. This rerun addresses the specific failure mode "RMSD did not
converge" by hardening the equilibration protocol (see Design choices below).

## Deliverables

| Table | Content | Script producing it |
|---|---|---|
| 5.8  | Per-compound MD summary: ligand RMSD (mean ± SEM, 20–100 ns), RMSF hot residues, H-bond count, contact fraction | `analyse_md.py` |
| 5.9  | Per-compound MMPBSA ΔG decomposition (E_vdW, E_ele, E_pol, E_apolar, ΔG_binding) | `analyse_mmpbsa.py` |
| 5.10 | Per-compound ADMET (41 admet-ai properties + Lipinski + QED + PAINS + SA + DILI + hERG) | `run_admet_full.py` |

## Design choices (defensive against RMSD non-convergence)

1. **Two-stage energy minimisation**: steepest descent (5000 steps, tol
   1000 kJ/mol/nm) → conjugate gradient (5000 steps, tol 100 kJ/mol/nm).
2. **Three-step equilibration**: NVT 100 ps (V-rescale, τ_T = 0.1 ps,
   position-restrained heavy atoms) → NPT-restrained 200 ps (Parrinello-
   Rahman, τ_P = 2.0 ps) → NPT-unrestrained 500 ps.
3. **First 20 ns of production discarded as further equilibration**;
   all reported RMSD / MMPBSA statistics use frames 20–100 ns.
4. **Adaptive length**: if backbone RMSD in the last 30 ns still drifts
   > 0.5 Å (95% CI over 5-ns blocks), the pipeline extends production to
   200 ns for that compound. Log flags this event explicitly.
5. **LINCS h-bond constraints** with `dt = 2 fs`, integrator `md`,
   AMBER99SB-ILDN + GAFF2 (Sobtop), TIP3P water, 0.15 M NaCl,
   300 K / 1 atm, cubic box 1.0 nm padding.
6. **MMPBSA on 200 frames sampled evenly from 70–100 ns** (post-convergence);
   report ΔG_binding ± SEM.

## Where things live

    scripts/md/
        run_all_md.sh              — WSL driver: 8 compounds × 100 ns MD
        run_admet_full.py          — CPU, ~5 min for all 8 SMILES
        analyse_md.py              — post-processing → Table 5.8 markdown
        analyse_mmpbsa.py          — post-processing → Table 5.9 markdown
        mdp/
            em1_steep.mdp
            em2_cg.mdp
            nvt_restr.mdp
            npt_restr.mdp
            npt_prod.mdp
        ligand_setup.sh            — ACPYPE / Sobtop / antechamber
        complex_setup.sh           — receptor + ligand → solvate → ionise

## Usage (WSL2 Ubuntu, RTX 4090)

```bash
# One-time setup
cd ~/STG-Mol
mamba env create -f scripts/md/environment.yml   # gromacs + gmx_MMPBSA + admet_ai
conda activate stgmol-md

# Sanity check on receptor + 1 compound
bash scripts/md/smoke_test.sh   # ~30 min, one compound to 20 ns

# Full 8-compound sweep
bash scripts/md/run_all_md.sh data/candidates/8_smiles.csv results/md/

# Aggregate into paper-ready tables
python scripts/md/analyse_md.py       --results_dir results/md   --output_md table_5_8.md
python scripts/md/analyse_mmpbsa.py   --results_dir results/md   --output_md table_5_9.md
python scripts/md/run_admet_full.py   --input data/candidates/8_smiles.csv \
                                       --output_csv admet_full.csv \
                                       --output_md  table_5_10.md
```

## Estimated wall clock (RTX 4090)

| Stage | Per compound | Total (×8) |
|---|---|---|
| Setup (ligand param + solvate + ionise + EM + eq) | ~10 min | 1.5 h |
| Production 100 ns MD (2 fs steps) | ~8–10 h | ~72 h |
| MMPBSA on 200 frames | ~30 min | 4 h |
| Analysis / plotting | ~5 min | 40 min |
| ADMET-AI + SwissADME (CPU) | ~30 s | ~5 min |
| **Total** | ~11 h | **~4 days GPU + 5 min CPU** |

The `--adaptive` flag can extend up to 200 ns per compound if RMSD does not
plateau — worst case ~8 days.

## Skeleton status (2026-07-16)

- [ ] `mdp/` templates — inheriting from user's `~/NLRP3_2/` files pending inventory
- [ ] `ligand_setup.sh` — pending confirmation of ACPYPE vs Sobtop vs antechamber preference
- [ ] `complex_setup.sh` — pending confirmation of receptor prep tool (pdb2gmx flavour)
- [ ] `run_all_md.sh` — awaits confirmed receptor PDB path + 8-compound SMILES CSV path
- [ ] `analyse_md.py` — will use `gmx rms`, `gmx rmsf`, `gmx hbond`, MDAnalysis for contacts
- [ ] `analyse_mmpbsa.py` — will parse `gmx_MMPBSA` `FINAL_RESULTS_MMPBSA.dat`
- [ ] `run_admet_full.py` — will call `admet_ai.AdmetModel().predict(list_of_smiles)`
- [ ] `environment.yml` — will pin gromacs=2024, gmx_MMPBSA=1.6+, admet_ai latest,
      openbabel, acpype
