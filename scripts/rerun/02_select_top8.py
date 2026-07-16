#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_select_top8.py — pick 8 candidates from Vina-scored top2000.

Selection rule (per user's decision):
  (1) rank all successfully-docked ligands by Vina ΔG ascending;
  (2) take the top-10;
  (3) Tanimoto (Morgan r=2, 2048 bits) redundancy filter with cutoff 0.7
      → keep first 8 diverse compounds (greedy).

Outputs:
  {output_dir}/candidates_v4.3/top8.csv          — SMILES + Vina ΔG + reason
  {output_dir}/candidates_v4.3/top8.smi          — SMILES only, one per line
  {output_dir}/candidates_v4.3/pose_comp_{i}.pdbqt  — best Vina pose (i=1..8)
  {output_dir}/candidates_v4.3/pose_comp_{i}.pdb    — same, PDB format for MD

Usage:
  conda activate nlrp3
  python3 scripts/rerun/02_select_top8.py
"""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger, DataStructs
RDLogger.DisableLog('rdApp.*')


def load_config():
    cfg_path = Path(__file__).resolve().parent / 'config.yaml'
    return yaml.safe_load(cfg_path.read_text())


def morgan_fp(smiles, n_bits=2048, radius=2):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=n_bits)


def diversity_greedy(rows, cutoff=0.7, k=8):
    """rows already sorted by Vina ΔG asc. Keep 1st; each next only if
    Tanimoto to every previously-kept < cutoff. Stop at k."""
    kept = []
    kept_fps = []
    for r in rows:
        fp = morgan_fp(r['smiles'])
        if fp is None:
            r['reject_reason'] = 'bad_smiles'
            continue
        if any(DataStructs.TanimotoSimilarity(fp, k) >= cutoff for k in kept_fps):
            r['reject_reason'] = f'Tanimoto_ge_{cutoff:.1f}_to_prior'
            continue
        r['reject_reason'] = ''
        kept.append(r)
        kept_fps.append(fp)
        if len(kept) >= k:
            break
    return kept


def pdbqt_to_pdb(pdbqt, pdb):
    """Extract MODEL 1 from pdbqt and convert to PDB via OpenBabel."""
    tmp = pdbqt.with_suffix('.model1.pdbqt')
    with open(pdbqt) as fin, open(tmp, 'w') as fout:
        in_first = False
        for ln in fin:
            if ln.startswith('MODEL 1'):
                in_first = True
                continue
            if ln.startswith('ENDMDL'):
                if in_first:
                    break
            if in_first:
                fout.write(ln)
    r = subprocess.run(['obabel', str(tmp), '-O', str(pdb)],
                        capture_output=True, text=True, timeout=60)
    tmp.unlink(missing_ok=True)
    return r.returncode == 0 and pdb.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoff', type=float, default=None,
                    help='Override Tanimoto cutoff from config.')
    ap.add_argument('--top_n_vina', type=int, default=None)
    ap.add_argument('--final_n', type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    out = Path(cfg['output_dir'])
    scored_csv = out / 'vina2000' / 'scored.csv'
    cand_dir = out / 'candidates_v4.3'
    cand_dir.mkdir(parents=True, exist_ok=True)

    top_n = args.top_n_vina or cfg['select_top_n_by_vina']
    cutoff = args.cutoff or cfg['tanimoto_dedup_cutoff']
    final_n = args.final_n or cfg['final_n_candidates']

    # Load Vina results
    with open(scored_csv) as f:
        rows = list(csv.DictReader(f))
    ok = [r for r in rows if r['status'] == 'ok' and r['vina_dG_kcal']]
    for r in ok:
        r['vina_dG_kcal'] = float(r['vina_dG_kcal'])
    ok.sort(key=lambda x: x['vina_dG_kcal'])
    top = ok[:top_n]
    print(f'Loaded {len(rows)} docked, {len(ok)} successful, taking top-{top_n}:')
    for r in top:
        print(f'  {r["vina_dG_kcal"]:+.3f}  {r["smiles"][:60]}...')

    # Greedy diversification
    kept = diversity_greedy(top, cutoff=cutoff, k=final_n)
    print(f'\nAfter Tanimoto-≥{cutoff} redundancy filter → {len(kept)} kept:')
    for i, r in enumerate(kept, 1):
        print(f'  Compound {i}: ΔG {r["vina_dG_kcal"]:+.3f}  '
              f'{r["smiles"][:60]}...')
    if len(kept) < final_n:
        print(f'! only {len(kept)} candidates diverse enough. Consider '
              f'raising --top_n_vina above {top_n} or relaxing --cutoff '
              f'above {cutoff}.')

    # Save
    with open(cand_dir / 'top8.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['compound_idx', 'smiles', 'vina_dG_kcal', 'stg_score',
                     'orig_row_idx', 'reject_reason'])
        for i, r in enumerate(kept, 1):
            w.writerow([i, r['smiles'], r['vina_dG_kcal'], r['stg_score'],
                         r['row_idx'], r['reject_reason']])

    with open(cand_dir / 'top8.smi', 'w') as f:
        for i, r in enumerate(kept, 1):
            f.write(f'{r["smiles"]}\tcompound_{i}\n')

    # Copy pdbqt and convert to pdb for downstream MD
    n_pdb_ok = 0
    for i, r in enumerate(kept, 1):
        src = Path(r['pose_pdbqt'])
        dst_pdbqt = cand_dir / f'pose_comp_{i}.pdbqt'
        dst_pdb = cand_dir / f'pose_comp_{i}.pdb'
        shutil.copyfile(src, dst_pdbqt)
        if pdbqt_to_pdb(dst_pdbqt, dst_pdb):
            n_pdb_ok += 1
        else:
            print(f'! Compound {i}: obabel pdbqt→pdb failed')
    print(f'\n✓ Wrote top8.csv + top8.smi + {n_pdb_ok}/{len(kept)} PDB poses to {cand_dir}')
    print(f'\nNext: bash scripts/rerun/03_run_md_8.sh')


if __name__ == '__main__':
    main()
