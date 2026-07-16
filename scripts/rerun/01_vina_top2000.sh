#!/usr/bin/env bash
# ============================================================================
# 01_vina_top2000.sh — Vina dock all 2000 candidates with CRID3-aligned box
# ============================================================================
# Runs under conda env `nlrp3` (has RDKit + Vina + Meeko for pdbqt gen).
# Each ligand: SMILES → RDKit 3D embed → Meeko pdbqt → Vina dock → best pose.
# Output: results_v4.3/vina2000/scored.csv with columns
#         smiles, stg_score, vina_dG_kcal, pose_pdbqt
# ============================================================================
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${SCRIPT_DIR}/config.yaml"

# ---- parse yaml (bash-native, robust to minor formatting) ----
receptor=$(awk '/^receptor_pdbqt:/ {print $2}' "$CFG")
input_csv=$(awk '/^input_csv:/ {print $2}' "$CFG")
output_dir=$(awk '/^output_dir:/ {print $2}' "$CFG")
center=$(awk '/^box_center:/ {gsub(/[\[\],]/, ""); print $2, $3, $4}' "$CFG")
box_size=$(awk '/^box_size:/ {gsub(/[\[\],]/, ""); print $2, $3, $4}' "$CFG")
read -r cx cy cz <<<"$center"
read -r sx sy sz <<<"$box_size"
exh=$(awk '/^vina_exhaustiveness:/ {print $2}' "$CFG")
seed=$(awk '/^vina_seed:/ {print $2}' "$CFG")
nposes=$(awk '/^vina_n_poses:/ {print $2}' "$CFG")
nworkers=$(awk '/^vina_n_workers:/ {print $2}' "$CFG")

work="${output_dir}/vina2000"
mkdir -p "${work}/pdbqt" "${work}/poses" "${work}/logs"
scored="${work}/scored.csv"

echo "[$(date '+%F %T')] Vina rerun — CRID3-aligned box"
echo "  receptor:  ${receptor}"
echo "  input:     ${input_csv} ($(wc -l < "${input_csv}") lines)"
echo "  box:       centre (${cx}, ${cy}, ${cz})  size (${sx}, ${sy}, ${sz}) Å"
echo "  exhaust:   ${exh}   n_poses: ${nposes}   seed: ${seed}"
echo "  workers:   ${nworkers}"
echo "  output:    ${scored}"

# Emit CSV header if new
if [[ ! -s "${scored}" ]]; then
    echo "row_idx,smiles,stg_score,vina_dG_kcal,pose_pdbqt,status" > "${scored}"
fi

# Python worker: takes one row → does everything → prints CSV line to stdout
python3 - "${scored}" "${input_csv}" "${work}" "${receptor}" \
        "${cx}" "${cy}" "${cz}" "${sx}" "${sy}" "${sz}" \
        "${exh}" "${nposes}" "${seed}" "${nworkers}" <<'PYEOF'
import csv, os, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

(scored_csv, input_csv, work_dir, receptor_pdbqt,
 cx, cy, cz, sx, sy, sz, exh, nposes, seed, nworkers) = sys.argv[1:]
cx, cy, cz = float(cx), float(cy), float(cz)
sx, sy, sz = float(sx), float(sy), float(sz)
exh, nposes, seed = int(exh), int(nposes), int(seed)
work = Path(work_dir); pdbqt_dir = work / 'pdbqt'; pose_dir = work / 'poses'
log_dir = work / 'logs'

# Skip rows we already processed (resumable)
done = set()
if os.path.exists(scored_csv):
    with open(scored_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            done.add(int(row['row_idx']))

rows = []
with open(input_csv) as f:
    r = csv.DictReader(f)
    for i, row in enumerate(r):
        if i in done: continue
        rows.append((i, row['smiles'], float(row.get('score', 0))))

print(f'[worker] {len(rows)} ligands to dock (skipping {len(done)} already done)',
      file=sys.stderr, flush=True)


def dock_one(item):
    i, smi, stg = item
    try:
        # 1. RDKit 3D embed
        m = Chem.MolFromSmiles(smi)
        if m is None: return (i, smi, stg, None, None, 'bad_smiles')
        m = Chem.AddHs(m)
        p = AllChem.ETKDGv3(); p.randomSeed = seed
        if AllChem.EmbedMolecule(m, p) != 0: return (i, smi, stg, None, None, 'embed_fail')
        AllChem.MMFFOptimizeMolecule(m, maxIters=200)

        # 2. Write SDF then convert to pdbqt with Meeko
        lig_pdbqt = pdbqt_dir / f'lig_{i:05d}.pdbqt'
        if not lig_pdbqt.exists():
            sdf = tempfile.NamedTemporaryFile(suffix='.sdf', delete=False).name
            w = Chem.SDWriter(sdf); w.write(m); w.close()
            # Meeko command-line tool
            r = subprocess.run(['mk_prepare_ligand.py', '-i', sdf,
                                 '-o', str(lig_pdbqt), '--rigid_macrocycles'],
                                capture_output=True, text=True, timeout=120)
            os.unlink(sdf)
            if r.returncode != 0 or not lig_pdbqt.exists():
                return (i, smi, stg, None, None, 'meeko_fail')

        # 3. Vina
        out_pdbqt = pose_dir / f'pose_{i:05d}.pdbqt'
        log = log_dir / f'log_{i:05d}.txt'
        r = subprocess.run(
            ['vina', '--receptor', receptor_pdbqt, '--ligand', str(lig_pdbqt),
             '--center_x', str(cx), '--center_y', str(cy), '--center_z', str(cz),
             '--size_x', str(sx), '--size_y', str(sy), '--size_z', str(sz),
             '--exhaustiveness', str(exh), '--num_modes', str(nposes),
             '--seed', str(seed), '--cpu', '4',
             '--out', str(out_pdbqt)],
            capture_output=True, text=True, timeout=600)
        with open(log, 'w') as f: f.write(r.stdout + '\n' + r.stderr)
        if r.returncode != 0: return (i, smi, stg, None, None, 'vina_fail')

        # Parse best pose energy: first mode is best
        with open(out_pdbqt) as f:
            for ln in f:
                if ln.startswith('REMARK VINA RESULT:'):
                    dG = float(ln.split()[3])
                    return (i, smi, stg, dG, str(out_pdbqt), 'ok')
        return (i, smi, stg, None, None, 'no_result')
    except subprocess.TimeoutExpired:
        return (i, smi, stg, None, None, 'timeout')
    except Exception as e:
        return (i, smi, stg, None, None, f'err:{type(e).__name__}')


with ProcessPoolExecutor(max_workers=int(nworkers)) as ex, \
     open(scored_csv, 'a') as out:
    w = csv.writer(out)
    n_done = 0
    for fut in as_completed(ex.submit(dock_one, r) for r in rows):
        i, smi, stg, dG, pose, st = fut.result()
        w.writerow([i, smi, stg, dG if dG is not None else '',
                     pose or '', st])
        out.flush()
        n_done += 1
        if n_done % 20 == 0:
            print(f'[{n_done}/{len(rows)}]', file=sys.stderr, flush=True)
PYEOF

echo "[$(date '+%F %T')] Vina rerun DONE. Rows in scored.csv:"
wc -l "${scored}"
echo ""
echo "Next: python3 scripts/rerun/02_select_top8.py"
