#!/usr/bin/env bash
# ============================================================================
# 03_run_md_8.sh — GROMACS MD driver for the 8 selected candidates.
#
# For each compound (i = 1..8):
#   ligand param (acpype/GAFF2) → complex build → solvate → ionise
#   → EM (steep + cg) → NVT 100 ps → NPT 200 ps (restr) → NPT 500 ps
#   → production 100 ns (with adaptive extension to 200 ns if RMSD
#   still drifts) → RMSD/RMSF/HBond analysis → summary.json.
#
# Idempotent: writes ".done" after each stage, skips completed compounds.
# Failures are logged and the next compound continues.
#
# Usage:
#   bash 03_run_md_8.sh                     # full run, all 8 compounds
#   bash 03_run_md_8.sh --total_ns 5        # short run (smoke test)
#   bash 03_run_md_8.sh --only 1            # single compound
# ============================================================================
set -eo pipefail   # NOTE: no -u; conda activate hooks reference unset vars
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${SCRIPT_DIR}/config.yaml"
MDP_DIR="${SCRIPT_DIR}/mdp"

# ── CLI overrides ──────────────────────────────────────────────────────
OVERRIDE_TOTAL_NS=""
ONLY_COMPOUND=""
SKIP_CONV_EXTEND="false"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --total_ns)      OVERRIDE_TOTAL_NS="$2"; shift 2;;
        --only)          ONLY_COMPOUND="$2"; shift 2;;
        --no_extend)     SKIP_CONV_EXTEND="true"; shift;;
        -h|--help)
            grep '^#' "$0" | head -30
            exit 0;;
        *) echo "unknown arg: $1"; exit 2;;
    esac
done

# ── Parse config.yaml (python is more reliable than awk on lists) ───────
eval "$(python3 - "$CFG" <<'PYEOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
kv = {
    "output_dir":       c["output_dir"],
    "receptor_pdb":     c["receptor_pdb"],
    "force_field":      c["force_field"],
    "water_model":      c["water_model"],
    "box_type":         c["box_type"],
    "box_padding_nm":   c["box_padding_nm"],
    "ion_conc_M":       c["ion_conc_M"],
    "temperature_K":    c["temperature_K"],
    "md_total_ns":      c["md_total_ns"],
    "md_dt_fs":         c["md_dt_fs"],
    "md_output_dt_ps":  c["md_output_dt_ps"],
    "md_gpu_id":        c["md_gpu_id"],
    "md_threads":       c["md_threads"],
    "md_convergence_check": str(c["md_convergence_check"]).lower(),
    "md_convergence_last_ns": c["md_convergence_last_ns"],
    "md_convergence_tol_A":   c["md_convergence_tol_A"],
    "md_extend_to_ns":  c["md_extend_to_ns"],
    "md_analysis_start_ns": c["md_analysis_start_ns"],
    "final_n_candidates": c["final_n_candidates"],
    "gmx_executable":   c["gmx_executable"],
}
for k, v in kv.items():
    print(f'{k}="{v}"')
PYEOF
)"

# CLI override for total ns
if [[ -n "${OVERRIDE_TOTAL_NS}" ]]; then
    md_total_ns="${OVERRIDE_TOTAL_NS}"
    echo "[override] md_total_ns = ${md_total_ns}"
fi

cand_dir="${output_dir}/candidates_v4.3"
md_root="${output_dir}/md"
log_dir="${output_dir}/logs"
mkdir -p "${md_root}" "${log_dir}"

GMX="${gmx_executable}"
DT_S=$(python3 -c "print(${md_dt_fs} * 1e-15)")
NSTEPS_PROD=$(python3 -c "print(int(${md_total_ns} * 1e-9 / ${DT_S}))")

# ---- acpype cross-env invocation ------------------------------------
# acpype lives in the `nlrp3` conda env, but this script normally runs
# under `gmx_cuda` (where gmx is on PATH). If acpype is on PATH we call
# it directly; otherwise fall back to `conda run -n nlrp3 acpype`.
if command -v acpype >/dev/null 2>&1; then
    ACPYPE_CMD="acpype"
else
    if ! command -v conda >/dev/null 2>&1; then
        for c in "$HOME/miniconda3/etc/profile.d/conda.sh" \
                 "$HOME/anaconda3/etc/profile.d/conda.sh"; do
            [[ -f "$c" ]] && source "$c" && break
        done
    fi
    ACPYPE_CMD="conda run --no-capture-output -n nlrp3 acpype"
    echo "[env] acpype not on PATH; using: ${ACPYPE_CMD}"
fi

echo "============================================================"
echo "  03_run_md_8 — GROMACS MD driver"
echo "  candidates dir: ${cand_dir}"
echo "  md root:        ${md_root}"
echo "  force field:    ${force_field}  water: ${water_model}"
echo "  production:     ${md_total_ns} ns  (${NSTEPS_PROD} steps @ ${md_dt_fs} fs)"
echo "  GPU id: ${md_gpu_id}  threads: ${md_threads}"
echo "============================================================"

# ── Helper: run a stage with idempotent .done marker ────────────────────
run_stage () {
    # $1 = stage name  $2..$N = command
    local name="$1"; shift
    local marker=".${name}.done"
    if [[ -f "${marker}" ]]; then
        echo "  [skip]  ${name} (already done)"
        return 0
    fi
    echo "  [run ]  ${name}"
    # Explicit exit-on-fail. Don't rely on `set -e` propagation because
    # the enclosing subshell does `exec > >(tee ...) 2>&1`, which breaks
    # errexit inheritance. Without this, a failing stage silently continues
    # and touches marker files for every downstream stage that also fails.
    if ! "$@"; then
        local rc=$?
        echo "  [FAIL]  ${name} exited with code ${rc}" >&2
        exit "${rc}"
    fi
    # GROMACS gmx sometimes exits 0 even on Fatal error. If EXPECT_FILE is
    # set (per-stage envvar), also verify the expected artefact exists.
    if [[ -n "${EXPECT_FILE:-}" ]] && [[ ! -s "${EXPECT_FILE}" ]]; then
        echo "  [FAIL]  ${name} exited 0 but expected output '${EXPECT_FILE}' is missing/empty" >&2
        exit 1
    fi
    touch "${marker}"
    unset EXPECT_FILE
}

# ── Per-compound pipeline ───────────────────────────────────────────────
run_compound () {
    local i="$1"
    local pose_pdb="${cand_dir}/pose_comp_${i}.pdb"
    local wd="${md_root}/comp_${i}"
    local log="${log_dir}/md_comp_${i}.log"

    if [[ -f "${wd}/all.done" ]]; then
        echo "comp_${i}: already fully complete → skip"
        return 0
    fi
    if [[ ! -f "${pose_pdb}" ]]; then
        echo "comp_${i}: MISSING ${pose_pdb} → skip" | tee -a "${log}"
        return 1
    fi

    mkdir -p "${wd}"
    (
        cd "${wd}"
        # Redirect all stdout+stderr for this compound to its log
        exec > >(tee -a "${log}") 2>&1
        echo ""
        echo "──────── comp_${i} @ $(date '+%F %T') ────────"

        # 1. Ligand parametrisation with acpype (GAFF2)
        run_stage acpype bash -c "
            ${ACPYPE_CMD} -i '${pose_pdb}' -b lig -o gmx -c bcc -a gaff2 -n 0
        "
        # Locate acpype output (older/newer versions place it in .acpype)
        local acp_dir="lig.acpype"
        if [[ ! -d "${acp_dir}" ]]; then
            acp_dir=$(find . -maxdepth 2 -type d -name '*.acpype' | head -1)
        fi
        cp -f "${acp_dir}/lig_GMX.gro"    lig.gro
        cp -f "${acp_dir}/lig_GMX.itp"    lig.itp
        cp -f "${acp_dir}/lig_GMX.top"    lig.top
        cp -f "${acp_dir}/posre_lig.itp"  posre_lig.itp 2>/dev/null || true

        # 2. Receptor prep via pdb2gmx (AMBER99SB-ILDN + TIP3P)
        run_stage pdb2gmx bash -c "
            printf '1\n1\n' | ${GMX} pdb2gmx -f '${receptor_pdb}' \
                -o receptor.gro -p receptor.top -i posre_prot.itp \
                -water ${water_model} -ff ${force_field} -ignh
        "

        # 3. Build complex — merge coordinates and topology
        run_stage build_complex python3 - <<PYEOF
# Merge receptor.gro + lig.gro → complex.gro; edit topology to include ligand.
# CRITICAL: acpype's lig.itp contains BOTH [atomtypes] and [moleculetype].
# GROMACS requires [atomtypes] to appear BEFORE any [moleculetype], which
# means it can't sit inside an #include that comes after Protein's own
# [moleculetype] in receptor.top. So we split it into:
#     lig_atomtypes.itp  — only the [ atomtypes ] section (included EARLY)
#     lig.itp            — everything else, no [ atomtypes ] (included LATE)
from pathlib import Path
import re

def read_gro(path):
    lines = Path(path).read_text().splitlines()
    title = lines[0]
    n = int(lines[1].strip())
    atoms = lines[2:2 + n]
    box   = lines[2 + n]
    return title, atoms, box

_, rec_atoms, _   = read_gro('receptor.gro')
_, lig_atoms, box = read_gro('lig.gro')
total = len(rec_atoms) + len(lig_atoms)
out  = ['Protein-ligand complex', f'{total:5d}']
out += rec_atoms + lig_atoms + [box]
Path('complex.gro').write_text('\n'.join(out) + '\n')

# ---- Split lig.itp into atomtypes.itp + moleculetype-and-later.itp ----
raw = Path('lig.itp').read_text()
# Split on section headers
sections = re.split(r'(?m)^(\[ *[A-Za-z_]+ *\])', raw)
# sections = ['<preamble>', '[ header1 ]', '<body1>', '[ header2 ]', '<body2>', ...]
atomtypes_block = ''
other_blocks = ''
if sections and sections[0].strip():
    other_blocks += sections[0]  # preamble comments
for i in range(1, len(sections), 2):
    header = sections[i]
    body = sections[i+1] if i+1 < len(sections) else ''
    if re.search(r'atomtypes', header, re.IGNORECASE):
        atomtypes_block += header + body
    else:
        other_blocks += header + body

if atomtypes_block:
    Path('lig_atomtypes.itp').write_text(atomtypes_block)
    Path('lig.itp').write_text(other_blocks)
    print('Split lig.itp → lig_atomtypes.itp + lig.itp (no atomtypes)')
else:
    print('lig.itp has no [ atomtypes ] — leaving unchanged')

# ---- Compose topol.top ----
#   receptor.top has structure:
#     #include ".../forcefield.itp"
#     [ moleculetype ] Protein_chain_A
#     ...
#     [ system ]
#     [ molecules ]
#     Protein_chain_A 1
#
# We must insert:
#   (a) #include "lig_atomtypes.itp" AFTER the forcefield include but
#       BEFORE the first [ moleculetype ]
#   (b) #include "lig.itp" ANYWHERE after that (but before [ system ] is neat)
#   (c) "LIG              1" line under [ molecules ]

top_in = Path('receptor.top').read_text()

# (a) Insert lig_atomtypes include right after the forcefield include line
if Path('lig_atomtypes.itp').exists():
    ff_include_re = re.compile(r'(#include\s+"[^"]*forcefield\.itp"\s*\n)')
    if ff_include_re.search(top_in):
        top_in = ff_include_re.sub(
            r'\1\n; Ligand atom types (must precede any [ moleculetype ])\n'
            r'#include "lig_atomtypes.itp"\n', top_in, count=1)
    else:
        # fallback: prepend
        top_in = ('; Ligand atom types (must precede any [ moleculetype ])\n'
                  '#include "lig_atomtypes.itp"\n' + top_in)

# (b) Insert lig.itp include just before [ system ]
lig_itp_include = '\n; Include ligand moleculetype\n#include "lig.itp"\n'
if '[ system ]' in top_in:
    top_in = top_in.replace('[ system ]', lig_itp_include + '\n[ system ]', 1)
else:
    top_in = top_in + lig_itp_include

# (c) Append LIG line to [ molecules ]
if not re.search(r'^\s*LIG\s+\d', top_in, re.MULTILINE):
    top_in = top_in.rstrip() + '\nLIG              1\n'

Path('topol.top').write_text(top_in)
print('complex.gro atoms =', total)
print('topol.top rewritten with split ligand includes')
PYEOF

        # 4. Solvate
        run_stage editconf ${GMX} editconf -f complex.gro -o boxed.gro \
            -bt ${box_type} -d ${box_padding_nm} -c
        run_stage solvate ${GMX} solvate -cp boxed.gro -cs spc216.gro \
            -o solvated.gro -p topol.top

        # 5. Add ions to reach 0.15 M NaCl
        EXPECT_FILE=ions.tpr run_stage ion_grompp ${GMX} grompp -f "${MDP_DIR}/em1_steep.mdp" \
            -c solvated.gro -p topol.top -o ions.tpr -maxwarn 3
        EXPECT_FILE=ionised.gro run_stage ionise bash -c "
            printf 'SOL\n' | ${GMX} genion -s ions.tpr -o ionised.gro -p topol.top \
                -pname NA -nname CL -neutral -conc ${ion_conc_M}
        "

        # 6. EM stage 1 — steepest descent
        EXPECT_FILE=em1.tpr run_stage em1_grompp ${GMX} grompp -f "${MDP_DIR}/em1_steep.mdp" \
            -c ionised.gro -p topol.top -o em1.tpr -maxwarn 3
        EXPECT_FILE=em1.gro run_stage em1_run ${GMX} mdrun -deffnm em1 -gpu_id ${md_gpu_id} -v

        # 7. EM stage 2 — conjugate gradient
        EXPECT_FILE=em2.tpr run_stage em2_grompp ${GMX} grompp -f "${MDP_DIR}/em2_cg.mdp" \
            -c em1.gro -p topol.top -o em2.tpr -maxwarn 3
        EXPECT_FILE=em2.gro run_stage em2_run ${GMX} mdrun -deffnm em2 -gpu_id ${md_gpu_id} -v

        # 8. Build a merged index for coupling groups Protein_LIG / Water_and_ions
        run_stage make_ndx bash -c "
            printf '1 | 13\nname 22 Protein_LIG\n15 | 14\nname 23 Water_and_ions\nq\n' \
                | ${GMX} make_ndx -f em2.gro -o index.ndx
        "
        # NOTE: default numeric groups vary — driver may need tweaking per system.
        # Defensive fallback: if the names weren't created, regenerate manually.
        if ! grep -q 'Protein_LIG' index.ndx 2>/dev/null; then
            python3 - <<PYEOF
# Fallback: build index.ndx by parsing atom groups from em2.gro
import re, pathlib
gro = pathlib.Path('em2.gro').read_text().splitlines()
n = int(gro[1])
protein, ligand, water_ions = [], [], []
for i, line in enumerate(gro[2:2+n], start=1):
    resname = line[5:10].strip()
    if resname == 'LIG':
        ligand.append(i)
    elif resname in ('SOL','HOH','NA','CL','K','MG','CA','ZN'):
        water_ions.append(i)
    else:
        protein.append(i)
protein_lig = protein + ligand
def block(name, ids):
    out = [f'[ {name} ]']
    for i in range(0, len(ids), 15):
        out.append(' '.join(f'{x:4d}' for x in ids[i:i+15]))
    return '\n'.join(out) + '\n'
open('index.ndx','w').write(
    block('System', protein + ligand + water_ions) +
    block('Protein_LIG', protein_lig) +
    block('Water_and_ions', water_ions) +
    block('LIG', ligand) +
    block('Protein', protein))
print('index.ndx rebuilt via fallback')
PYEOF
        fi

        # 9. NVT — 100 ps position-restrained
        EXPECT_FILE=nvt.tpr run_stage nvt_grompp ${GMX} grompp -f "${MDP_DIR}/nvt.mdp" \
            -c em2.gro -r em2.gro -p topol.top -n index.ndx \
            -o nvt.tpr -maxwarn 3
        EXPECT_FILE=nvt.gro run_stage nvt_run ${GMX} mdrun -deffnm nvt \
            -gpu_id ${md_gpu_id} -nt ${md_threads} -pin on -v

        # 10. NPT restrained — 200 ps
        EXPECT_FILE=npt_restr.tpr run_stage npt_restr_grompp ${GMX} grompp -f "${MDP_DIR}/npt_restr.mdp" \
            -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -n index.ndx \
            -o npt_restr.tpr -maxwarn 3
        EXPECT_FILE=npt_restr.gro run_stage npt_restr_run ${GMX} mdrun -deffnm npt_restr \
            -gpu_id ${md_gpu_id} -nt ${md_threads} -pin on -v

        # 11. NPT unrestrained — 500 ps
        EXPECT_FILE=npt_prod.tpr run_stage npt_prod_grompp ${GMX} grompp -f "${MDP_DIR}/npt_prod.mdp" \
            -c npt_restr.gro -t npt_restr.cpt -p topol.top -n index.ndx \
            -o npt_prod.tpr -maxwarn 3
        EXPECT_FILE=npt_prod.gro run_stage npt_prod_run ${GMX} mdrun -deffnm npt_prod \
            -gpu_id ${md_gpu_id} -nt ${md_threads} -pin on -v

        # 12. Production — 100 ns (or override)
        # Rewrite nsteps in a temp mdp so --total_ns can shorten runs.
        awk -v nst="${NSTEPS_PROD}" '/^nsteps/ {print "nsteps = "nst; next} {print}' \
            "${MDP_DIR}/md_prod.mdp" > md_prod_local.mdp

        EXPECT_FILE=md.tpr run_stage md_grompp ${GMX} grompp -f md_prod_local.mdp \
            -c npt_prod.gro -t npt_prod.cpt -p topol.top -n index.ndx \
            -o md.tpr -maxwarn 3
        EXPECT_FILE=md.gro run_stage md_run ${GMX} mdrun -deffnm md \
            -gpu_id ${md_gpu_id} -nt ${md_threads} -pin on -v -cpi md.cpt

        # 13. Analysis: RMSD backbone, RMSD ligand, RMSF, H-bond
        run_stage rms_bb bash -c "
            printf '4\n4\n' | ${GMX} rms -s md.tpr -f md.xtc \
                -o rmsd_bb.xvg -tu ns -n index.ndx
        "
        run_stage rms_lig bash -c "
            printf '4\nLIG\n' | ${GMX} rms -s md.tpr -f md.xtc \
                -o rmsd_lig.xvg -tu ns -n index.ndx
        "
        run_stage rmsf bash -c "
            printf '1\n' | ${GMX} rmsf -s md.tpr -f md.xtc \
                -o rmsf.xvg -res -n index.ndx
        "
        run_stage hbond bash -c "
            printf 'Protein\nLIG\n' | ${GMX} hbond -s md.tpr -f md.xtc \
                -num hbonds.xvg -n index.ndx
        "

        # 14. Convergence check + adaptive extension
        python3 - "${i}" "${md_total_ns}" "${md_convergence_last_ns}" \
            "${md_convergence_tol_A}" "${md_extend_to_ns}" \
            "${md_analysis_start_ns}" "${md_convergence_check}" \
            "${SKIP_CONV_EXTEND}" <<'PYEOF'
import json, subprocess, sys, math, os
from pathlib import Path

(idx, total_ns, last_ns, tol_A, extend_ns, analysis_start_ns,
 conv_check, skip_extend) = sys.argv[1:]
total_ns  = float(total_ns);   last_ns   = float(last_ns)
tol_A     = float(tol_A);      extend_ns = float(extend_ns)
analysis_start_ns = float(analysis_start_ns)
do_check = conv_check.lower() == 'true' and skip_extend.lower() != 'true'

def read_xvg(path):
    xs, ys = [], []
    for ln in Path(path).read_text().splitlines():
        if not ln or ln.startswith(('#', '@')): continue
        t, v = ln.split()[:2]
        xs.append(float(t)); ys.append(float(v))
    return xs, ys

# GROMACS reports RMSD in nm; convert to Å.
xs, ys = read_xvg('rmsd_bb.xvg')
ys_A = [y * 10.0 for y in ys]

# Slice to last N ns
t_max = xs[-1] if xs else 0.0
t_lo  = max(t_max - last_ns, 0.0)
block_ys = []
i0 = 0
while i0 < len(xs):
    if xs[i0] >= t_lo: break
    i0 += 1
# split into 5-ns blocks
block_len_ns = 5.0
blocks = []
cur = []
cur_t0 = None
for t, y in zip(xs[i0:], ys_A[i0:]):
    if cur_t0 is None:
        cur_t0 = t
    if t - cur_t0 >= block_len_ns:
        if cur:
            blocks.append(sum(cur)/len(cur))
        cur = []
        cur_t0 = t
    cur.append(y)
if cur:
    blocks.append(sum(cur)/len(cur))

def mean_ci(vals):
    n = len(vals)
    if n < 2: return (vals[0] if vals else 0.0, 0.0)
    m = sum(vals)/n
    s = (sum((v-m)**2 for v in vals) / (n-1))**0.5
    return m, 1.96 * s / (n**0.5)

mean_bb, ci_bb = mean_ci(blocks)
converged = ci_bb < tol_A
extended_to = None

if not converged and do_check and total_ns < extend_ns:
    print(f'[comp_{idx}] backbone CI {ci_bb:.3f} Å ≥ tol {tol_A} — extending to {extend_ns} ns')
    # convert-tpr + append
    target_ps = int(extend_ns * 1000)
    subprocess.run(['gmx', 'convert-tpr', '-s', 'md.tpr',
                    '-until', str(target_ps), '-o', 'md_ext.tpr'], check=True)
    os.replace('md_ext.tpr', 'md.tpr')
    subprocess.run(['gmx', 'mdrun', '-deffnm', 'md', '-cpi', 'md.cpt',
                    '-append', '-v', '-pin', 'on'], check=True)
    extended_to = extend_ns
    # Re-run RMSD/H-bond
    for cmd in [
        "printf '4\\n4\\n' | gmx rms  -s md.tpr -f md.xtc -o rmsd_bb.xvg  -tu ns -n index.ndx",
        "printf '4\\nLIG\\n' | gmx rms -s md.tpr -f md.xtc -o rmsd_lig.xvg -tu ns -n index.ndx",
        "printf '1\\n' | gmx rmsf -s md.tpr -f md.xtc -o rmsf.xvg -res -n index.ndx",
        "printf 'Protein\\nLIG\\n' | gmx hbond -s md.tpr -f md.xtc -num hbonds.xvg -n index.ndx",
    ]:
        subprocess.run(cmd, shell=True, check=True)
    # recompute convergence
    xs, ys = read_xvg('rmsd_bb.xvg')
    ys_A = [y * 10.0 for y in ys]
    t_max = xs[-1] if xs else 0.0
    t_lo  = max(t_max - last_ns, 0.0)
    blocks = [y for t, y in zip(xs, ys_A) if t >= t_lo]
    mean_bb, ci_bb = mean_ci(blocks) if len(blocks) > 1 else (0.0, 0.0)
    converged = ci_bb < tol_A

# ── Analysis metrics on post-analysis-start-ns portion ──
def sliced(xs, ys, t_start):
    return [(x, y) for x, y in zip(xs, ys) if x >= t_start]

def sem(vals):
    if len(vals) < 2: return 0.0
    m = sum(vals)/len(vals)
    return ((sum((v-m)**2 for v in vals)/(len(vals)-1))/len(vals))**0.5

xs, ys = read_xvg('rmsd_bb.xvg')
seg = sliced(xs, [y*10 for y in ys], analysis_start_ns)
rmsd_bb_vals = [y for _, y in seg]
rmsd_bb_mean = sum(rmsd_bb_vals)/len(rmsd_bb_vals) if rmsd_bb_vals else 0.0
rmsd_bb_sem  = sem(rmsd_bb_vals)

xs, ys = read_xvg('rmsd_lig.xvg')
seg = sliced(xs, [y*10 for y in ys], analysis_start_ns)
rmsd_lig_vals = [y for _, y in seg]
rmsd_lig_mean = sum(rmsd_lig_vals)/len(rmsd_lig_vals) if rmsd_lig_vals else 0.0
rmsd_lig_sem  = sem(rmsd_lig_vals)

xs, ys = read_xvg('hbonds.xvg')
# hbond time is in ps → convert to ns for the slice condition
seg = [(x/1000.0, y) for x, y in zip(xs, ys) if (x/1000.0) >= analysis_start_ns]
hb_vals = [y for _, y in seg]
hb_mean = sum(hb_vals)/len(hb_vals) if hb_vals else 0.0
hb_sem  = sem(hb_vals)

# Top-5 RMSF residues
resids, rmsf_vals = [], []
for ln in Path('rmsf.xvg').read_text().splitlines():
    if not ln or ln.startswith(('#','@')): continue
    r, v = ln.split()[:2]
    resids.append(int(float(r))); rmsf_vals.append(float(v)*10.0)  # nm → Å
top5 = sorted(zip(resids, rmsf_vals), key=lambda p: -p[1])[:5]
rmsf_top5 = {str(rid): round(v, 3) for rid, v in top5}

# SMILES / vina lookup
smi = ""
vina = None
try:
    import csv
    with open(Path('..').parent / 'candidates_v4.3' / 'top8.csv') as f:
        for row in csv.DictReader(f):
            if int(row['compound_idx']) == int(idx):
                smi = row['smiles']; vina = float(row['vina_dG_kcal']); break
except Exception:
    pass

summary = dict(
    compound_idx=int(idx),
    smiles=smi,
    vina_dG=vina,
    rmsd_bb_mean_A=round(rmsd_bb_mean, 3),
    rmsd_bb_sem_A=round(rmsd_bb_sem, 3),
    rmsd_lig_mean_A=round(rmsd_lig_mean, 3),
    rmsd_lig_sem_A=round(rmsd_lig_sem, 3),
    rmsf_top5_residues=rmsf_top5,
    hbond_mean=round(hb_mean, 3),
    hbond_sem=round(hb_sem, 3),
    n_frames_analysed=len(rmsd_bb_vals),
    converged=bool(converged),
    extended_to_ns=extended_to,
)
Path('summary.json').write_text(json.dumps(summary, indent=2))
print(f'comp_{idx}: prod OK, RMSD converged {rmsd_bb_mean:.2f} ± {rmsd_bb_sem:.2f} Å, '
      f'hbond mean {hb_mean:.1f}')
PYEOF

        touch all.done
    ) || {
        echo "comp_${i}: FAILED (see ${log})" | tee -a "${log}"
        return 1
    }
}

# ── Main loop over compounds ─────────────────────────────────────────────
n=${final_n_candidates}
if [[ -n "${ONLY_COMPOUND}" ]]; then
    run_compound "${ONLY_COMPOUND}" || true
else
    for i in $(seq 1 "${n}"); do
        echo ""
        echo "══════════ Compound ${i}/${n} ══════════"
        run_compound "${i}" || echo "→ comp_${i} failed, continuing"
    done
fi

echo ""
echo "[$(date '+%F %T')] 03_run_md_8 finished."
echo "Next: bash ${SCRIPT_DIR}/04_run_mmpbsa_8.sh"
