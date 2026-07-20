#!/usr/bin/env bash
# ============================================================================
# smoke_test.sh — end-to-end pipeline sanity check on compound_1 only.
#
# Runs:
#   Stage A (Vina)    — top-5 ligands from input CSV, exh=8 (fast)
#   Stage B (Select)  — take top-1 by Vina ΔG
#   Stage C (MD)      — compound 1, production shortened to 5 ns
#   Stage D (MMPBSA)  — frames 3-5 ns, ~10 sampled frames
#   Stage E (ADMET)   — RDKit + admet-ai on compound 1 SMILES
#   Stage F (STG-Mol) — V3-random ensemble on compound 1
#   Stage G (Tables)  — aggregate, verify formatting
#
# Expected wall clock on RTX 4090: ~1-2 hours total.
# Purpose: catch environment / topology / gmx_MMPBSA issues before the
# 4-day full run kicks off.
# ============================================================================
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${SCRIPT_DIR}/config.yaml"

output_dir=$(python3 -c "import yaml; print(yaml.safe_load(open('${CFG}'))['output_dir'])")
input_csv=$(python3 -c "import yaml; print(yaml.safe_load(open('${CFG}'))['input_csv'])")

echo "═══════════════════════════════════════════════════════════════"
echo "  SMOKE TEST — 5-ligand Vina → 1-compound 5 ns MD → MMPBSA 3-5 ns"
echo "  Output tree: ${output_dir}"
echo "═══════════════════════════════════════════════════════════════"

# ── Prepare a tiny 5-ligand input for stage A ─────────────────────────
smoke_dir="${output_dir}/smoke_prep"
mkdir -p "${smoke_dir}"
smoke_input="${smoke_dir}/input_top5.csv"
head -6 "${input_csv}" > "${smoke_input}"        # header + 5 rows
echo "[smoke] Prepared 5-ligand input: ${smoke_input}"

# ── Config override for the smoke run ──────────────────────────────────
# Rewrite paths & tiny params, restore on EXIT.
TMP_CFG=$(mktemp)
python3 - "$CFG" "$TMP_CFG" "$smoke_input" <<'PYEOF'
import sys, yaml
src, dst, tiny_input = sys.argv[1:]
c = yaml.safe_load(open(src))
c['input_csv']              = tiny_input
c['vina_exhaustiveness']    = 8       # fast, still meaningful
c['vina_n_workers']         = 4
c['select_top_n_by_vina']   = 3
c['tanimoto_dedup_cutoff']  = 0.99    # accept everything (no dedup for smoke)
c['final_n_candidates']     = 1
c['mmpbsa_start_ns']        = 3
c['mmpbsa_end_ns']          = 5
c['mmpbsa_n_frames']        = 10
c['mmpbsa_interval_frames'] = 20
open(dst,'w').write(yaml.safe_dump(c))
PYEOF
cp "${CFG}" "${CFG}.bak.smoke"
cp "${TMP_CFG}" "${CFG}"
trap 'cp "${CFG}.bak.smoke" "${CFG}" 2>/dev/null; rm -f "${CFG}.bak.smoke" "${TMP_CFG}"; echo "[smoke] Restored original config.yaml"' EXIT

# ── Stage A (Vina 5-ligand) ────────────────────────────────────────────
echo ""; echo "[smoke] Stage A — Vina on 5 ligands (exh=8)"
bash "${SCRIPT_DIR}/01_vina_top2000.sh"

# ── Stage B (Select top-1) ─────────────────────────────────────────────
echo ""; echo "[smoke] Stage B — select top 1 candidate"
python3 "${SCRIPT_DIR}/02_select_top8.py" --final_n 1 --top_n_vina 3 --cutoff 0.99

# ── Stage C (MD) ───────────────────────────────────────────────────────
echo ""; echo "[smoke] Stage C — MD (compound 1, --total_ns 5 --no_extend)"
bash "${SCRIPT_DIR}/03_run_md_8.sh" --only 1 --total_ns 5 --no_extend

# ── Stage D (MMPBSA) — tiny window ─────────────────────────────────────
echo ""; echo "[smoke] Stage D — MMPBSA (frames 3-5 ns, ~10 frames)"
bash "${SCRIPT_DIR}/04_run_mmpbsa_8.sh"

# ── Stage E (ADMET) — single-SMILES run ─────────────────────────────────
echo ""; echo "[smoke] Stage E — ADMET on compound 1 only"
mkdir -p "${output_dir}/admet_smoke"
head -1 "${output_dir}/candidates_v4.3/top8.smi" > "${output_dir}/admet_smoke/one.smi"
python3 "${SCRIPT_DIR}/05_run_admet_full.py" \
    --input "${output_dir}/admet_smoke/one.smi" \
    --output_csv "${output_dir}/admet_smoke/admet_full.csv" \
    --output_md  "${output_dir}/admet_smoke/table_5_10.md"

# ── Stage F (STG-Mol re-score) ─────────────────────────────────────────
echo ""; echo "[smoke] Stage F — STG-Mol V3-random re-score"
python3 "${SCRIPT_DIR}/06_score_v3random.py" || {
    echo "! 06_score_v3random failed — check ckpt paths in config.yaml"
}

# ── Stage G — table aggregation smoke ──────────────────────────────────
echo ""; echo "[smoke] Stage G — aggregate tables"
python3 "${SCRIPT_DIR}/07_aggregate_tables.py"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  SMOKE TEST COMPLETE"
echo "  Verify these files look sane before launching the full run:"
echo "    ${output_dir}/vina2000/scored.csv                  (5 rows)"
echo "    ${output_dir}/candidates_v4.3/top8.csv             (1 row)"
echo "    ${output_dir}/md/comp_1/summary.json"
echo "    ${output_dir}/md/comp_1/mmpbsa_summary.json"
echo "    ${output_dir}/admet_smoke/table_5_10.md"
echo "    ${output_dir}/v3random_scores/table_5_7_rerun.md"
echo "    ${output_dir}/tables/*.md"
echo "═══════════════════════════════════════════════════════════════"
echo "Next: rm -rf ${output_dir} && bash ${SCRIPT_DIR}/01_vina_top2000.sh"
echo "      → full 2000-ligand run, then 02→03→04→05→06→07 for real."
