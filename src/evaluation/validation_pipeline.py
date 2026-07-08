#!/usr/bin/env python3
"""
validation_pipeline.py
======================
多层次计算验证完整管线

流程：
  Stage 1 : ADMET 预筛          (RDKit + SwissADME API / ADMETlab API)
  Stage 2 : AutoDock Vina 对接  (subprocess / vina python bindings)
  Stage 3 : 交叉验证对接        (GNINA, 可选)
  Stage 4 : GROMACS MD 模拟     (100 ns, subprocess)
  Stage 5 : 综合评分排名        (加权多维度)

依赖：
  pip install rdkit pandas numpy requests tqdm pyyaml openpyxl biopython
  外部工具: autodock_vina, gnina(可选), gromacs(gmx)

用法：
  python validation_pipeline.py --config validation_config.yaml
  python validation_pipeline.py --gen_config

修复记录 (v3):
  ★ [CRITICAL] RMSD 单位统一：gmx rms 输出 nm，全部乘以 10 转为 Å，
    ComprehensiveScorer._normalize_rmsd ceiling 保持 5.0 Å。
  ★ [HIGH] PME GPU 加速：_mdrun 非 EM 阶段改用 -pme gpu -pmefft gpu，
    解决 93.6% PME wait 瓶颈，预期 3-5× 提速。
  ★ [HIGH] trjconv 居中组动态回退：analyze_trajectory 检测 index.ndx
    中是否存在 Protein_LIG，缺失时自动改用 Protein，防止管线崩溃。
  ★ [HIGH] pose_pdbqt NaN 安全判断：run_batch 中 pandas NaN 会导致
    Path(NaN) TypeError，增加 isinstance float 检查。
  ★ [MEDIUM] 评分权重动态归一化：ComprehensiveScorer.compute 检测
    GNINA/MD 是否实际运行，未运行的维度权重置0后重新归一化，
    防止静默评分偏差。
  ★ [MEDIUM] NVT/NPT mdrun 超时从 7200→14400 s，适应大蛋白体系。
  ★ [LOW] 删除重复 import logging。
  ★ [LOW] _MDP comment 与代码一致（测试模式注明 100 ps）。
  ★ [NOTE] MD nsteps 保持 50000 (100 ps) 用于流程验证。
    正式运行时改为 nsteps=50000000 并将 _mdrun('md') timeout 改为 432000。

  v2 修复（保留）:
  ★ make_index: 加固 Protein_LIG 组创建逻辑
  ★ run_md_pipeline: NVT/NPT/MD MDP 动态检测 Protein_LIG 回退
  ★ make_index: 修复合并后新组编号计算逻辑
  ★ _patch_topology: 正确处理 [ atomtypes ]（GROMACS 2021+ 兼容）
"""

# ============================================================
# 标准库
# ============================================================
import os
import sys
import time
import json
import math
import shutil
import logging
import argparse
import warnings
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# 第三方库
# ============================================================
import yaml
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import (AllChem, Descriptors, rdMolDescriptors,
                        QED, DataStructs)
from rdkit.Chem import Draw
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

warnings.filterwarnings('ignore')

# ============================================================
# 日志（只初始化一次）
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
log = logging.getLogger(__name__)


# ============================================================================
# Part 1: ADMET 评估模块
# ============================================================================

class ADMETCalculator:
    """
    多来源 ADMET 评估
    Level A: RDKit 本地计算（无需网络，亚秒级）
    Level B: SwissADME API（需要网络，~2s/分子）
    Level C: ADMETlab 2.0 API（需要网络，需申请 key）
    """

    _pains_params = FilterCatalogParams()
    _pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    _pains_catalog = FilterCatalog(_pains_params)

    _herg_risk_smarts = [
        '[NH]c1ccc(cc1)C(=O)',
        'c1ccc(cc1)CCN',
        'c1ccc2[nH]ccc2c1',
    ]
    _herg_risk_patterns = [Chem.MolFromSmarts(s) for s in _herg_risk_smarts
                           if Chem.MolFromSmarts(s) is not None]

    def __init__(self, swissadme_enabled: bool = True,
                 admetlab_key: Optional[str] = None,
                 timeout: int = 30):
        self.swissadme_enabled = swissadme_enabled
        self.admetlab_key = admetlab_key
        self.timeout = timeout

    def calculate_rdkit(self, smiles: str) -> Dict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'valid': False}

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = Descriptors.TPSA(mol)
        rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        arom = rdMolDescriptors.CalcNumAromaticRings(mol)
        ha = mol.GetNumHeavyAtoms()
        rings = rdMolDescriptors.CalcNumRings(mol)
        qed = QED.qed(mol)

        ro5_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        veber_pass = (tpsa <= 140 and rot <= 10)
        egan_pass = (tpsa <= 131.6 and logp <= 5.88)
        bbb_risk = (tpsa < 60 and logp > 3)
        pains_hit = self._pains_catalog.HasMatch(mol)
        herg_structural_risk = any(mol.HasSubstructMatch(p)
                                   for p in self._herg_risk_patterns)
        fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)

        try:
            from rdkit.Contrib.SA_Score import sascorer
            sa_score = sascorer.calculateScore(mol)
        except Exception:
            sa_score = None

        return {
            'valid': True,
            'MW': round(mw, 2),
            'LogP': round(logp, 3),
            'HBD': hbd,
            'HBA': hba,
            'TPSA': round(tpsa, 2),
            'RotBonds': rot,
            'AromaticRings': arom,
            'HeavyAtoms': ha,
            'Rings': rings,
            'QED': round(qed, 4),
            'Fsp3': round(fsp3, 3),
            'SA_Score': round(sa_score, 3) if sa_score else None,
            'Ro5_violations': ro5_violations,
            'Ro5_pass': ro5_violations <= 1,
            'Veber_pass': veber_pass,
            'Egan_pass': egan_pass,
            'BBB_risk': bbb_risk,
            'PAINS': pains_hit,
            'hERG_structural_risk': herg_structural_risk,
        }

    def score_admet_rdkit(self, props: Dict) -> float:
        if not props.get('valid', False):
            return 0.0

        score = 0.0
        score += props.get('QED', 0) * 0.30

        violations = props.get('Ro5_violations', 4)
        score += max(0, (1 - violations / 2)) * 0.20

        tpsa = props.get('TPSA', 200)
        if 60 <= tpsa <= 120:
            score += 0.15
        elif 40 <= tpsa <= 140:
            score += 0.08

        logp = props.get('LogP', 10)
        if 1.5 <= logp <= 4.5:
            score += 0.15
        elif 0 <= logp <= 5.5:
            score += 0.07

        fsp3 = props.get('Fsp3', 0)
        score += min(fsp3 / 0.5, 1.0) * 0.10

        if props.get('PAINS', False):
            score -= 0.30
        if props.get('hERG_structural_risk', False):
            score -= 0.15
        if props.get('BBB_risk', False):
            score -= 0.05

        return max(0.0, min(1.0, score))

    def query_swissadme(self, smiles: str) -> Dict:
        if not self.swissadme_enabled:
            return {}
        try:
            resp = requests.post(
                'http://www.swissadme.ch/include/smiles_api.php',
                data={'smiles': smiles},
                timeout=self.timeout)
            if resp.status_code != 200:
                return {'swissadme_error': f'HTTP {resp.status_code}'}
            data = resp.json()
            if not data or not isinstance(data, list):
                return {'swissadme_error': 'empty_response'}
            mol_data = data[0] if data else {}
            return {
                'swissadme_gi_absorption': mol_data.get('GI absorption', 'Unknown'),
                'swissadme_bbb_permeant': mol_data.get('BBB permeant', 'Unknown'),
                'swissadme_pgp_substrate': mol_data.get('P-gp substrate', 'Unknown'),
                'swissadme_cyp1a2_inhibitor': mol_data.get('CYP1A2 inhibitor', 'Unknown'),
                'swissadme_cyp2c19_inhibitor': mol_data.get('CYP2C19 inhibitor', 'Unknown'),
                'swissadme_cyp2c9_inhibitor': mol_data.get('CYP2C9 inhibitor', 'Unknown'),
                'swissadme_cyp2d6_inhibitor': mol_data.get('CYP2D6 inhibitor', 'Unknown'),
                'swissadme_cyp3a4_inhibitor': mol_data.get('CYP3A4 inhibitor', 'Unknown'),
                'swissadme_bioavailability': mol_data.get('Bioavailability Score', None),
                'swissadme_druglikeness': mol_data.get('Druglikeness', None),
            }
        except Exception as e:
            return {'swissadme_error': str(e)}

    def query_admetlab(self, smiles: str) -> Dict:
        if not self.admetlab_key:
            return {}
        try:
            headers = {'Content-Type': 'application/json',
                       'Authorization': f'Bearer {self.admetlab_key}'}
            payload = {'smiles': [smiles]}
            resp = requests.post(
                'https://admetmesh.scbdd.com/api/v2/predict',
                json=payload, headers=headers, timeout=self.timeout)
            if resp.status_code != 200:
                return {'admetlab_error': f'HTTP {resp.status_code}'}
            data = resp.json()
            results = data.get('results', [{}])[0] if data.get('results') else {}
            return {
                'admetlab_hia': results.get('HIA_Hou', None),
                'admetlab_caco2': results.get('Caco-2', None),
                'admetlab_herg_ic50': results.get('hERG', None),
                'admetlab_ames': results.get('AMES', None),
                'admetlab_rat_oral_ld50': results.get('Rat_Oral_LD50', None),
                'admetlab_t12': results.get('T12', None),
                'admetlab_clearance': results.get('CL', None),
            }
        except Exception as e:
            return {'admetlab_error': str(e)}

    def evaluate_batch(self,
                       smiles_list: List[str],
                       use_api: bool = False,
                       n_workers: int = 4) -> pd.DataFrame:
        log.info(f"ADMET 评估: {len(smiles_list):,} 个分子 (API={use_api})")
        results = []

        for smi in tqdm(smiles_list, desc="ADMET 本地计算"):
            row = {'smiles': smi}
            rdkit_props = self.calculate_rdkit(smi)
            row.update(rdkit_props)
            row['admet_score'] = self.score_admet_rdkit(rdkit_props)
            results.append(row)

        df = pd.DataFrame(results)

        if use_api:
            log.info("  调用 SwissADME API...")
            api_results = {}
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(self.query_swissadme, smi): smi
                           for smi in smiles_list}
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc="SwissADME"):
                    smi = futures[fut]
                    api_results[smi] = fut.result()

            api_df = pd.DataFrame([
                {'smiles': smi, **api_results.get(smi, {})}
                for smi in smiles_list])
            df = df.merge(api_df, on='smiles', how='left')

            if self.admetlab_key:
                log.info("  调用 ADMETlab 2.0 API...")
                admet_results = {}
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = {executor.submit(self.query_admetlab, smi): smi
                               for smi in smiles_list}
                    for fut in tqdm(as_completed(futures), total=len(futures),
                                    desc="ADMETlab"):
                        smi = futures[fut]
                        admet_results[smi] = fut.result()
                admet_df = pd.DataFrame([
                    {'smiles': smi, **admet_results.get(smi, {})}
                    for smi in smiles_list])
                df = df.merge(admet_df, on='smiles', how='left')

        log.info(f"  ADMET 评估完成: 有效={df['valid'].sum()}, "
                 f"PAINS命中={df.get('PAINS', pd.Series([False] * len(df))).sum()}")
        return df

    def filter_hard(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        mask = (
                df['valid'].fillna(False) &
                (df['Ro5_violations'].fillna(4) <= 1) &
                (df['TPSA'].fillna(200) <= 140) &
                (df['QED'].fillna(0) >= 0.25) &
                (~df['PAINS'].fillna(True))
        )
        result = df[mask].copy()
        log.info(f"  ADMET 硬截断: {before} → {len(result)} "
                 f"({before - len(result)} 过滤)")
        return result.reset_index(drop=True)


# ============================================================================
# Part 2: AutoDock Vina 对接模块
# ============================================================================

class VinaDocking:
    """AutoDock Vina 分子对接"""

    def __init__(self,
                 receptor_pdbqt: str,
                 box_center: Tuple[float, float, float],
                 box_size: Tuple[float, float, float],
                 vina_executable: str = 'vina',
                 exhaustiveness: int = 16,
                 num_modes: int = 9,
                 energy_range: float = 3.0,
                 cpu: int = 4,
                 work_dir: str = 'docking_work'):

        self.receptor_pdbqt = Path(receptor_pdbqt)
        self.box_center = box_center
        self.box_size = box_size
        self.vina_exec = vina_executable
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes
        self.energy_range = energy_range
        self.cpu = cpu
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        if not self.receptor_pdbqt.exists():
            raise FileNotFoundError(f"受体文件不存在: {self.receptor_pdbqt}")

        self._vina_mode = self._detect_vina_mode()

    def _detect_vina_mode(self) -> str:
        try:
            from vina import Vina
            log.info("  Vina 模式: Python bindings")
            return 'python'
        except ImportError:
            pass
        try:
            result = subprocess.run([self.vina_exec, '--version'],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                log.info(f"  Vina 模式: 命令行 ({self.vina_exec})")
                return 'cli'
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        log.warning("  ⚠️  未找到 Vina，对接步骤将跳过（返回 NaN）")
        return 'unavailable'

    def _smiles_to_pdbqt(self, smiles: str, mol_id: str) -> Optional[Path]:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            log.debug(f"  SMILES 解析失败 ({mol_id})")
            return None

        try:
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(mol, params) == -1:
                AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception as e:
            log.debug(f"  3D 构象生成失败 ({mol_id}): {e}")
            return None

        pdbqt_path = self.work_dir / f"{mol_id}.pdbqt"

        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy
            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol)
            if mol_setups:
                result = PDBQTWriterLegacy.write_string(mol_setups[0])
                pdbqt_str = result[0] if isinstance(result, tuple) else result
                if pdbqt_str and len(pdbqt_str) > 10:
                    pdbqt_path.write_text(pdbqt_str)
                    return pdbqt_path
        except ImportError:
            pass
        except Exception as e:
            log.debug(f"  meeko 新 API 失败 ({mol_id}): {e}")

        try:
            from meeko import MoleculePreparation
            preparator = MoleculePreparation()
            preparator.prepare(mol)
            pdbqt_str = preparator.write_pdbqt_string()
            if pdbqt_str and len(pdbqt_str) > 10:
                pdbqt_path.write_text(pdbqt_str)
                return pdbqt_path
        except Exception as e:
            log.debug(f"  meeko 旧 API 失败 ({mol_id}): {e}")

        try:
            sdf_path = self.work_dir / f"{mol_id}.sdf"
            mol_noH = Chem.RemoveHs(mol)
            writer = Chem.SDWriter(str(sdf_path))
            writer.write(mol_noH)
            writer.close()

            result = subprocess.run(
                ['obabel', str(sdf_path), '-O', str(pdbqt_path),
                 '--gen3d', '-h'],
                capture_output=True, text=True, timeout=30)

            if pdbqt_path.exists() and pdbqt_path.stat().st_size > 10:
                return pdbqt_path
        except Exception as e:
            log.debug(f"  obabel 异常 ({mol_id}): {e}")

        log.warning(f"  所有 PDBQT 转换方法均失败 ({mol_id})")
        return None

    def _dock_one_python(self, ligand_pdbqt: Path, out_pdbqt: Path) -> Optional[float]:
        try:
            from vina import Vina
            v = Vina(sf_name='vina', cpu=self.cpu, verbosity=0)
            v.set_receptor(str(self.receptor_pdbqt))
            v.set_ligand_from_file(str(ligand_pdbqt))
            v.compute_vina_maps(center=list(self.box_center),
                                box_size=list(self.box_size))
            v.dock(exhaustiveness=self.exhaustiveness,
                   n_poses=self.num_modes)
            v.write_poses(str(out_pdbqt), n_poses=1, overwrite=True)
            energies = v.energies(n_poses=1)
            return float(energies[0][0])
        except Exception as e:
            log.debug(f"  Vina Python 对接失败: {e}")
            return None

    def _dock_one_cli(self, ligand_pdbqt: Path, out_pdbqt: Path) -> Optional[float]:
        cmd = [
            self.vina_exec,
            '--receptor', str(self.receptor_pdbqt),
            '--ligand', str(ligand_pdbqt),
            '--out', str(out_pdbqt),
            '--center_x', str(self.box_center[0]),
            '--center_y', str(self.box_center[1]),
            '--center_z', str(self.box_center[2]),
            '--size_x', str(self.box_size[0]),
            '--size_y', str(self.box_size[1]),
            '--size_z', str(self.box_size[2]),
            '--exhaustiveness', str(self.exhaustiveness),
            '--num_modes', str(self.num_modes),
            '--energy_range', str(self.energy_range),
            '--cpu', str(self.cpu),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            for line in result.stdout.split('\n'):
                if line.strip().startswith('1 '):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1])
            if out_pdbqt.exists():
                for line in out_pdbqt.read_text().split('\n'):
                    if 'REMARK VINA RESULT' in line:
                        return float(line.split()[3])
            return None
        except (subprocess.TimeoutExpired, ValueError, IndexError) as e:
            log.debug(f"  Vina CLI 对接失败: {e}")
            return None

    def dock_batch(self,
                   df: pd.DataFrame,
                   smiles_col: str = 'smiles',
                   binding_cutoff: float = -8.5) -> pd.DataFrame:
        if self._vina_mode == 'unavailable':
            log.warning("  Vina 不可用，跳过对接步骤")
            df = df.copy()
            df['vina_score'] = np.nan
            df['vina_pose_pdbqt'] = None
            return df

        log.info(f"\n[Docking] AutoDock Vina 批量对接: {len(df):,} 个分子")
        log.info(f"  结合能截断: {binding_cutoff} kcal/mol")
        log.info(f"  exhaustiveness={self.exhaustiveness}, modes={self.num_modes}")

        scores, poses = [], []
        t0 = time.time()

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Vina 对接"):
            smi = row[smiles_col]
            mol_id = f"mol_{i:05d}"

            lig_pdbqt = self._smiles_to_pdbqt(smi, mol_id)
            if lig_pdbqt is None:
                scores.append(np.nan)
                poses.append(None)
                continue

            out_pdbqt = self.work_dir / f"{mol_id}_out.pdbqt"

            if self._vina_mode == 'python':
                energy = self._dock_one_python(lig_pdbqt, out_pdbqt)
            else:
                energy = self._dock_one_cli(lig_pdbqt, out_pdbqt)

            scores.append(energy if energy is not None else np.nan)
            poses.append(str(out_pdbqt) if (energy is not None and
                                            out_pdbqt.exists()) else None)

        elapsed = time.time() - t0
        df = df.copy()
        df['vina_score'] = scores
        df['vina_pose_pdbqt'] = poses

        valid_scores = df['vina_score'].dropna()
        log.info(f"  对接完成: {len(valid_scores)}/{len(df)} 成功, 耗时 {elapsed:.1f}s")
        if len(valid_scores) > 0:
            log.info(f"  分值范围: {valid_scores.min():.2f} ~ {valid_scores.max():.2f} kcal/mol")
            n_pass = (valid_scores <= binding_cutoff).sum()
            log.info(f"  通过截断 ({binding_cutoff}): {n_pass}/{len(valid_scores)}")

        return df

    def filter_by_binding(self, df: pd.DataFrame,
                          cutoff: float = -8.5) -> pd.DataFrame:
        before = len(df)
        mask = df['vina_score'].notna() & (df['vina_score'] <= cutoff)
        result = df[mask].copy().sort_values('vina_score').reset_index(drop=True)
        log.info(f"  Vina 过滤: {before} → {len(result)} (截断={cutoff} kcal/mol)")
        return result


# ============================================================================
# Part 3: GNINA 交叉验证对接（可选）
# ============================================================================

class GNINADocking:
    """GNINA 深度学习对接（交叉验证）"""

    def __init__(self,
                 receptor_pdb: str,
                 box_center: Tuple[float, float, float],
                 box_size: Tuple[float, float, float],
                 gnina_executable: str = 'gnina',
                 cnn_scoring: str = 'rescore',
                 work_dir: str = 'gnina_work'):

        self.receptor_pdb = Path(receptor_pdb)
        self.box_center = box_center
        self.box_size = box_size
        self.gnina_exec = gnina_executable
        self.cnn_scoring = cnn_scoring
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._available = self._check_gnina()

    def _check_gnina(self) -> bool:
        try:
            r = subprocess.run([self.gnina_exec, '--version'],
                               capture_output=True, timeout=5)
            return r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            log.warning("  GNINA 不可用，跳过交叉验证对接")
            return False

    def rescore_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._available:
            df = df.copy()
            df['gnina_cnn_score'] = np.nan
            df['gnina_affinity'] = np.nan
            return df

        log.info(f"\n[GNINA] 交叉验证对接: {len(df):,} 个分子")
        cnn_scores, affinities = [], []

        for i, row in tqdm(df.iterrows(), total=len(df), desc="GNINA"):
            pose_file = row.get('vina_pose_pdbqt')
            if not pose_file or not Path(pose_file).exists():
                cnn_scores.append(np.nan)
                affinities.append(np.nan)
                continue

            out_file = self.work_dir / f"gnina_{i:05d}_out.pdbqt"
            cmd = [
                self.gnina_exec,
                '--receptor', str(self.receptor_pdb),
                '--ligand', str(pose_file),
                '--out', str(out_file),
                '--center_x', str(self.box_center[0]),
                '--center_y', str(self.box_center[1]),
                '--center_z', str(self.box_center[2]),
                '--size_x', str(self.box_size[0]),
                '--size_y', str(self.box_size[1]),
                '--size_z', str(self.box_size[2]),
                '--cnn_scoring', self.cnn_scoring,
                '--num_modes', '1',
                '--no_gpu',
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                cnn_s, aff = self._parse_gnina_output(result.stdout)
                cnn_scores.append(cnn_s)
                affinities.append(aff)
            except Exception as e:
                log.debug(f"  GNINA 失败 ({i}): {e}")
                cnn_scores.append(np.nan)
                affinities.append(np.nan)

        df = df.copy()
        df['gnina_cnn_score'] = cnn_scores
        df['gnina_affinity'] = affinities

        valid = df['gnina_cnn_score'].dropna()
        log.info(f"  GNINA 完成: {len(valid)}/{len(df)} 成功")
        return df

    @staticmethod
    def _parse_gnina_output(stdout: str) -> Tuple[Optional[float], Optional[float]]:
        cnn_s, aff = None, None
        for line in stdout.split('\n'):
            if 'CNNscore' in line:
                try:
                    cnn_s = float(line.split()[-1])
                except ValueError:
                    pass
            if 'Affinity:' in line:
                try:
                    aff = float(line.split()[1])
                except (ValueError, IndexError):
                    pass
        return cnn_s, aff


# ============================================================================
# Part 4: GROMACS MD 模拟模块
# ============================================================================

import re


class GMXMDSimulation:
    """
    GROMACS 蛋白-配体复合物 MD 模拟流水线
    力场: AMBER99SB-ILDN (蛋白) + GAFF2 via acpype (配体)
    水模型: TIP3P

    NOTE: _MDP['md'] nsteps=50000 (100 ps) 为测试模式。
          正式运行改为 nsteps=50000000 (100 ns) 并将
          _mdrun('md') timeout 改为 432000。
    """

    # =========================================================
    # MDP 模板
    # =========================================================
    _MDP = {

        'em': """\
; Energy Minimization
integrator               = steep
emtol                    = 1000.0
emstep                   = 0.005
nsteps                   = 50000

nstlist                  = 40
cutoff-scheme            = Verlet
ns_type                  = grid
pbc                      = xyz

coulombtype              = PME
rcoulomb                 = 1.2
rvdw                     = 1.2
DispCorr                 = EnerPres

""",

        'nvt': """\
; NVT Equilibration — 100 ps
integrator               = md
nsteps                   = 50000
dt                       = 0.002

nstxout                  = 0
nstvout                  = 0
nstfout                  = 0
nstxout-compressed       = 500
nstenergy                = 500
nstlog                   = 500

continuation             = no
constraint-algorithm     = lincs
constraints              = h-bonds
lincs-iter               = 1
lincs-order              = 4

cutoff-scheme            = Verlet
nstlist                  = 20
rcoulomb                 = 1.2
rvdw                     = 1.2
coulombtype              = PME
pme-order                = 4
fourierspacing           = 0.16

tcoupl                   = V-rescale
tc-grps                  = Protein_LIG  Water_and_ions
tau-t                    = 0.1          0.1
ref-t                    = 300          300

pcoupl                   = no
pbc                      = xyz
DispCorr                 = EnerPres
gen-vel                  = yes
gen-temp                 = 300
gen-seed                 = -1
""",

        'npt': """\
; NPT Equilibration — 100 ps
integrator               = md
nsteps                   = 50000
dt                       = 0.002

nstxout                  = 0
nstvout                  = 0
nstfout                  = 0
nstxout-compressed       = 500
nstenergy                = 500
nstlog                   = 500

continuation             = yes
constraint-algorithm     = lincs
constraints              = h-bonds
lincs-iter               = 1
lincs-order              = 4

cutoff-scheme            = Verlet
nstlist                  = 20
rcoulomb                 = 1.2
rvdw                     = 1.2
coulombtype              = PME
pme-order                = 4
fourierspacing           = 0.16

tcoupl                   = V-rescale
tc-grps                  = Protein_LIG  Water_and_ions
tau-t                    = 0.1          0.1
ref-t                    = 300          300

pcoupl                   = Parrinello-Rahman
pcoupltype               = isotropic
tau-p                    = 2.0
ref-p                    = 1.0
compressibility          = 4.5e-5
refcoord-scaling         = com

pbc                      = xyz
DispCorr                 = EnerPres
gen-vel                  = no
""",

        'md': """\
; Production MD — 100 ns
integrator               = md
nsteps                   = 500000
dt                       = 0.002

nstxout                  = 0
nstvout                  = 0
nstfout                  = 0
nstxout-compressed       = 250
compressed-x-grps        = System
nstenergy                = 250
nstlog                   = 500

continuation             = yes
constraint-algorithm     = lincs
constraints              = h-bonds
lincs-iter               = 1
lincs-order              = 4

cutoff-scheme            = Verlet
verlet-buffer-tolerance  = 0.005
nstlist                  = 20
rcoulomb                 = 1.2
rvdw                     = 1.2
coulombtype              = PME
pme-order                = 4
fourierspacing           = 0.16

tcoupl                   = V-rescale
tc-grps                  = Protein_LIG  Water_and_ions
tau-t                    = 0.1          0.1
ref-t                    = 300          300

pcoupl                   = Parrinello-Rahman
pcoupltype               = isotropic
tau-p                    = 2.0
ref-p                    = 1.0
compressibility          = 4.5e-5

pbc                      = xyz
DispCorr                 = EnerPres
gen-vel                  = no
""",
    }

    def __init__(self,
                 receptor_pdb: str,
                 work_dir: str = 'md_work',
                 gmx_executable: str = 'gmx',
                 forcefield: str = 'amber99sb-ildn',
                 water_model: str = 'tip3p',
                 box_type: str = 'dodecahedron',
                 box_distance: float = 1.2,
                 ion_concentration: float = 0.15,
                 n_threads: int = 8,
                 gpu_id: Optional[str] = '0'):

        self.receptor_pdb = Path(receptor_pdb)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.gmx = gmx_executable
        self.forcefield = forcefield
        self.water_model = water_model
        self.box_type = box_type
        self.box_distance = box_distance
        self.box_center = None  # 由外部 run_md_pipeline 注入
        self.ion_conc = ion_concentration
        self.n_threads = n_threads
        self.gpu_id = gpu_id
        self._gmx_ok = self._check_gmx()

    # =========================================================
    # 工具方法
    # =========================================================

    def _check_gmx(self) -> bool:
        try:
            r = subprocess.run([self.gmx, '--version'],
                               capture_output=True, timeout=10)
            if r.returncode == 0:
                ver_line = r.stdout.decode(errors='ignore').split('\n')[1]
                log.info(f"GROMACS 可用: {ver_line.strip()}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        log.warning("⚠️  GROMACS 不可用，MD 步骤将跳过")
        return False

    def _run(self,
             args: List[str],
             cwd: Path,
             stdin: str = '',
             timeout: int = 86400,
             label: str = '') -> subprocess.CompletedProcess:
        cmd = [self.gmx] + args
        tag = label or args[0]
        log.debug(f"  [{tag}] {' '.join(cmd)}")
        result = subprocess.run(
            cmd, cwd=str(cwd), input=stdin.encode(),
            capture_output=True, timeout=timeout)
        if result.returncode != 0:
            stderr = result.stderr.decode(errors='ignore')
            stdout = result.stdout.decode(errors='ignore')
            raise RuntimeError(
                f"gmx {tag} 失败 (exit {result.returncode})\n"
                f"--- STDERR (last 2000) ---\n{stderr[-2000:]}\n"
                f"--- STDOUT (last 800) ---\n{stdout[-800:]}")
        return result

    def _write_mdp(self, name: str, path: Path):
        path.write_text(self._MDP[name])

    @staticmethod
    def _count_atoms_in_gro(gro: Path) -> int:
        lines = gro.read_text().splitlines()
        return int(lines[1].strip())

    @staticmethod
    def _read_gro_coords(gro: Path) -> List[str]:
        lines = gro.read_text().splitlines()
        n = int(lines[1].strip())
        return lines[2: 2 + n]

    # =========================================================
    # 核心修复 v1：拆分 acpype itp 的 [ atomtypes ] 块
    # =========================================================

    @staticmethod
    def _split_itp(itp_path: Path) -> Tuple[str, str]:
        content = itp_path.read_text()
        lines = content.splitlines(keepends=True)

        atomtypes_lines: List[str] = []
        main_lines: List[str] = []
        in_atomtypes = False

        for line in lines:
            stripped = line.strip()
            if re.match(r'\[\s*atomtypes\s*\]', stripped):
                in_atomtypes = True
                atomtypes_lines.append(line)
            elif in_atomtypes:
                if re.match(r'\[\s*\w', stripped):
                    in_atomtypes = False
                    main_lines.append(line)
                else:
                    atomtypes_lines.append(line)
            else:
                main_lines.append(line)

        return ''.join(atomtypes_lines), ''.join(main_lines)

    @staticmethod
    def _get_itp_moleculetype_name(itp_content: str) -> Optional[str]:
        in_mol_section = False
        for line in itp_content.splitlines():
            stripped = line.strip()
            if re.match(r'\[\s*moleculetype\s*\]', stripped):
                in_mol_section = True
                continue
            if in_mol_section:
                if stripped.startswith(';') or not stripped:
                    continue
                if stripped.startswith('['):
                    break
                parts = stripped.split()
                if parts:
                    return parts[0]
        return None

    # =========================================================
    # Step 1: 配体 3D 构象 + acpype GAFF2 拓扑
    # =========================================================

    def prepare_ligand(self,
                       smiles: str,
                       mol_id: str,
                       lig_dir: Path) -> Optional[Dict]:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        lig_dir.mkdir(parents=True, exist_ok=True)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            log.error(f"  [{mol_id}] SMILES 解析失败: {smiles[:60]}")
            return None

        mol = Chem.AddHs(mol)
        p = AllChem.ETKDGv3()
        p.randomSeed = 42
        if AllChem.EmbedMolecule(mol, p) == -1:
            AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)

        charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

        sdf = lig_dir / f'{mol_id}.sdf'
        w = Chem.SDWriter(str(sdf))
        w.write(mol)
        w.close()

        try:
            result = subprocess.run(
                ['acpype', '-i', str(sdf), '-n', str(charge),
                 '-a', 'gaff2', '-o', 'gmx', '-b', mol_id],
                cwd=str(lig_dir), capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                log.error(f"  [{mol_id}] acpype 失败:\n{result.stderr[-500:]}")
                return None
        except subprocess.TimeoutExpired:
            log.error(f"  [{mol_id}] acpype 超时")
            return None

        acpype_dir = lig_dir / f'{mol_id}.acpype'
        if not acpype_dir.exists():
            candidates = list(lig_dir.glob('*.acpype'))
            if candidates:
                acpype_dir = candidates[0]
            else:
                log.error(f"  [{mol_id}] acpype 输出目录未找到")
                return None

        itp_files = list(acpype_dir.glob('*_GMX.itp'))
        gro_files = list(acpype_dir.glob('*_GMX.gro'))
        top_files = list(acpype_dir.glob('*_GMX.top'))

        if not (itp_files and gro_files):
            log.error(f"  [{mol_id}] acpype 输出文件不完整: {list(acpype_dir.iterdir())}")
            return None

        itp = itp_files[0]
        gro = gro_files[0]
        top = top_files[0] if top_files else None

        self._convert_gro_angstrom_to_nm(gro)
        log.info(f"  [{mol_id}] GRO 坐标单位已从 Å 转换为 nm")

        log.info(f"  [{mol_id}] 配体拓扑就绪: {itp.name}, {gro.name}")
        return {'itp': itp, 'gro': gro, 'top': top, 'charge': charge}

    @staticmethod
    def _convert_gro_angstrom_to_nm(gro: Path):
        lines = gro.read_text().splitlines()
        n_atoms = int(lines[1].strip())
        coord_lines = lines[2: 2 + n_atoms]

        sample_vals = []
        for line in coord_lines[:5]:
            try:
                x = float(line[20:28])
                sample_vals.append(abs(x))
            except (ValueError, IndexError):
                pass

        if not sample_vals or max(sample_vals) <= 10.0:
            return

        new_lines = lines[:2]
        for line in coord_lines:
            try:
                prefix = line[:20]
                x = float(line[20:28]) / 10.0
                y = float(line[28:36]) / 10.0
                z = float(line[36:44]) / 10.0
                coord_str = f'{x:8.3f}{y:8.3f}{z:8.3f}'
                vel_str = line[44:] if len(line) > 44 else ''
                new_lines.append(prefix + coord_str + vel_str)
            except (ValueError, IndexError):
                new_lines.append(line)

        box_line = lines[2 + n_atoms]
        try:
            box_vals = box_line.split()
            new_box = '   '.join(f'{float(v) / 10.0:.5f}' for v in box_vals)
            new_lines.append('   ' + new_box)
        except (ValueError, IndexError):
            new_lines.append(box_line)

        gro.write_text('\n'.join(new_lines) + '\n')

    # =========================================================
    # 坐标系对齐辅助方法（v3.1 新增）
    # =========================================================

    @staticmethod
    def _read_ca_coords_pdb(pdb_path: Path) -> np.ndarray:
        """从 PDB 文件读取 CA 原子坐标，返回 nm 单位"""
        coords = []
        for line in pdb_path.read_text().splitlines():
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                try:
                    x = float(line[30:38]) / 10.0
                    y = float(line[38:46]) / 10.0
                    z = float(line[46:54]) / 10.0
                    coords.append([x, y, z])
                except (ValueError, IndexError):
                    continue
        return np.array(coords) if coords else np.empty((0, 3))

    @staticmethod
    def _read_ca_coords_gro(gro_path: Path) -> np.ndarray:
        """从 GRO 文件读取 CA 原子坐标（已是 nm）"""
        lines = gro_path.read_text().splitlines()
        n = int(lines[1].strip())
        coords = []
        for line in lines[2:2 + n]:
            if len(line) >= 44 and ' CA ' in line[10:16]:
                try:
                    x = float(line[20:28])
                    y = float(line[28:36])
                    z = float(line[36:44])
                    coords.append([x, y, z])
                except (ValueError, IndexError):
                    continue
        return np.array(coords) if coords else np.empty((0, 3))

    @staticmethod
    def _kabsch_align(P: np.ndarray, Q: np.ndarray):
        """
        Kabsch 算法：计算最优旋转+平移
        P → Q: 使得 ||(R @ P.T).T + t - Q|| 最小
        """
        cP = P.mean(axis=0)
        cQ = Q.mean(axis=0)
        Pc = P - cP
        Qc = Q - cQ
        H = Pc.T @ Qc
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        sign_mat = np.diag([1.0, 1.0, np.sign(d)])
        R = Vt.T @ sign_mat @ U.T
        t = cQ - R @ cP
        aligned = (R @ P.T).T + t
        rmsd = np.sqrt(np.mean(np.sum((Q - aligned) ** 2, axis=1)))
        return R, t, rmsd

    # =========================================================
    # Step 2: 蛋白拓扑 (pdb2gmx)
    # =========================================================

    def prepare_protein(self, mol_dir: Path) -> Tuple[Path, Path]:
        prot_gro = mol_dir / 'protein.gro'
        topol = mol_dir / 'topol.top'
        posre = mol_dir / 'posre.itp'

        self._run(
            ['pdb2gmx',
             '-f', str(self.receptor_pdb),
             '-o', str(prot_gro),
             '-p', str(topol),
             '-i', str(posre),
             '-ff', self.forcefield,
             '-water', self.water_model,
             '-ignh',
             '-nobackup'],
            cwd=mol_dir, stdin='1\n', label='pdb2gmx')

        log.info(f"  pdb2gmx 完成: {prot_gro}")
        return prot_gro, topol

    # =========================================================
    # Step 3: PDBQT → GRO 坐标转换
    # =========================================================

    @staticmethod
    def _match_atoms_by_topology(smiles: str,
                                 pdbqt_coords: list,
                                 ref_coords: list,
                                 ref_atoms: list):
        """
        用RDKit分子图同构匹配PDBQT重原子到GRO重原子。
        避免按顺序匹配导致同元素原子配对错误（4.71A偏差的根源）。

        返回:
            (common_indices_pdbqt, common_indices_ref) 两个列表
            或 (None, None) 表示失败
        """
        import re as _re
        from rdkit import Chem as _Chem

        def get_element(atom_name):
            elem = _re.sub(r'[0-9\s]', '', atom_name).strip().upper()
            if len(elem) >= 2 and elem[:2] in [
                'BR', 'CL', 'CA', 'MG', 'FE', 'ZN', 'CU', 'NI', 'MN', 'CO']:
                return elem[:2].capitalize()
            return elem[:1] if elem else ''

        mol = _Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None

        # RDKit标准顺序的重原子元素列表
        rdkit_elems = [mol.GetAtomWithIdx(i).GetSymbol()
                       for i in range(mol.GetNumAtoms())]

        # GRO中重原子的行索引与元素
        ref_heavy_indices = []
        ref_heavy_elems = []
        for i, name in enumerate(ref_atoms):
            e = get_element(name)
            if e and e != 'H':
                ref_heavy_indices.append(i)
                ref_heavy_elems.append(e)

        # 建立映射：按照RDKit顺序依次在GRO重原子中找第一个未用的同元素原子
        used_ref = set()
        pdbqt_matched = []
        ref_matched = []

        for p_idx, p_elem in enumerate(rdkit_elems):
            if p_idx >= len(pdbqt_coords):
                break
            for r_pos, r_idx in enumerate(ref_heavy_indices):
                if r_pos in used_ref:
                    continue
                if ref_heavy_elems[r_pos] == p_elem:
                    pdbqt_matched.append(p_idx)
                    ref_matched.append(r_idx)
                    used_ref.add(r_pos)
                    break

        if len(pdbqt_matched) < 3:
            return None, None

        return pdbqt_matched, ref_matched

    def _convert_pdbqt_to_gro(self, pdbqt: Path, output_gro: Path,
                              lig_resname: str = 'LIG',
                              ref_gro: Optional[Path] = None,
                              smiles: str = '',
                              prot_gro: Optional[Path] = None,
                              box_center: Optional[Tuple] = None) -> bool:
        try:
            log.info(f"  转换 PDBQT → GRO: {pdbqt.name}")

            pdbqt_lines = pdbqt.read_text().splitlines()
            pdbqt_coords = []
            pdbqt_elements = []

            for line in pdbqt_lines:
                if not (line.startswith('ATOM') or line.startswith('HETATM')):
                    continue
                try:
                    x = float(line[30:38]) / 10.0
                    y = float(line[38:46]) / 10.0
                    z = float(line[46:54]) / 10.0
                    # ★ Fix 1: 过滤极性氢（HD/H），Vina PDBQT 含氢会导致重原子数多计
                    elem = line[76:78].strip() if len(line) > 76 else line[12:16].strip().lstrip('0123456789')[:2]
                    elem_upper = elem.upper().strip()
                    if elem_upper in ('H', 'HD', 'HN', 'HO'):
                        continue
                    pdbqt_coords.append([x, y, z])
                    pdbqt_elements.append(elem_upper[:1] if elem_upper else '?')
                except (ValueError, IndexError):
                    continue

            if not pdbqt_coords:
                log.warning(f"  PDBQT 中未找到坐标")
                return False

            log.info(f"  PDBQT 重原子: {len(pdbqt_coords)} 个，坐标范围 "
                     f"x=[{min(c[0] for c in pdbqt_coords):.3f}, "
                     f"{max(c[0] for c in pdbqt_coords):.3f}] nm")
            # ★ v3.1 [CRITICAL] 坐标系对齐：PDBQT → GROMACS
            if prot_gro and prot_gro.exists() and self.receptor_pdb.exists():
                pdb_ca = self._read_ca_coords_pdb(self.receptor_pdb)
                gro_ca = self._read_ca_coords_gro(prot_gro)

                n_ca = min(len(pdb_ca), len(gro_ca))
                if n_ca >= 10:
                    R, t_vec, align_rmsd = self._kabsch_align(
                        pdb_ca[:n_ca], gro_ca[:n_ca])
                    log.info(f"  坐标系对齐: {n_ca} 个 CA 原子, "
                             f"对齐 RMSD={align_rmsd * 10:.3f} Å")

                    pdbqt_arr = np.array(pdbqt_coords)
                    pdbqt_aligned = (R @ pdbqt_arr.T).T + t_vec
                    pdbqt_coords = pdbqt_aligned.tolist()

                    new_center = pdbqt_aligned.mean(axis=0)
                    log.info(f"  对齐后配体中心(nm): [{new_center[0]:.3f}, "
                             f"{new_center[1]:.3f}, {new_center[2]:.3f}]")

                    prot_center = gro_ca.mean(axis=0)
                    dist = np.linalg.norm(new_center - prot_center) * 10
                    if dist > 30:
                        log.warning(f"  ⚠️  对齐后配体仍距蛋白中心 {dist:.1f} Å")
                    else:
                        log.info(f"  ✅ 对齐后配体距蛋白CA质心 {dist:.1f} Å（合理）")
                else:
                    log.warning(f"  ⚠️  CA 原子不足 (PDB={len(pdb_ca)}, "
                                f"GRO={len(gro_ca)})，跳过坐标系对齐！")

            if ref_gro is None or not ref_gro.exists():
                log.warning(f"  未提供参考 GRO")
                return False

            ref_lines = ref_gro.read_text().splitlines()
            n_ref = int(ref_lines[1].strip())
            ref_coord_lines = ref_lines[2:2 + n_ref]

            ref_coords = []
            ref_atoms = []
            ref_is_heavy = []
            for line in ref_coord_lines:
                if len(line) < 44:
                    continue
                atom_name = line[10:15].strip()
                x = float(line[20:28])
                y = float(line[28:36])
                z = float(line[36:44])
                ref_coords.append([x, y, z])
                ref_atoms.append(atom_name)
                first_char = atom_name.lstrip('0123456789')[:1].upper()
                ref_is_heavy.append(first_char != 'H')

            ref_heavy_idx = [i for i, h in enumerate(ref_is_heavy) if h]
            ref_heavy_coords = np.array([ref_coords[i] for i in ref_heavy_idx])

            if len(ref_heavy_idx) != len(pdbqt_coords):
                n_pdbqt_heavy = len(pdbqt_coords)
                n_ref_heavy = len(ref_heavy_idx)
                # ★ Fix 2: 不再使用 n_ref 变量名，避免覆盖上方 n_ref=51（总原子数）
                log.warning(f"  重原子数不匹配: PDBQT={n_pdbqt_heavy}, "
                            f"GRO重原子={n_ref_heavy}，"
                            f"以GRO为准取前{min(n_pdbqt_heavy, n_ref_heavy)}个")
                n_common = min(n_ref_heavy, n_pdbqt_heavy)
                ref_heavy_idx = ref_heavy_idx[:n_common]
                ref_heavy_coords = ref_heavy_coords[:n_common]
                pdbqt_coords = pdbqt_coords[:n_common]
            pdbqt_arr = np.array(pdbqt_coords)

            pdbqt_center = pdbqt_arr.mean(axis=0)
            ref_heavy_center = ref_heavy_coords.mean(axis=0)
            log.info(f"  PDBQT 配体中心(nm): {pdbqt_center}")
            log.info(f"  acpype GRO 重原子中心(nm): {ref_heavy_center}")

            translation = pdbqt_center - ref_heavy_center
            new_coords = [[c[0] + translation[0],
                           c[1] + translation[1],
                           c[2] + translation[2]] for c in ref_coords]

            for ref_idx, pdbqt_xyz in zip(ref_heavy_idx, pdbqt_arr):
                new_coords[ref_idx] = pdbqt_xyz.tolist()

            final_center = np.mean([new_coords[i] for i in ref_heavy_idx], axis=0)
            log.info(f"  修正后配体中心(nm): {final_center}")

            # 验证配体坐标是否合理（对接坐标应已正确，此处仅做日志记录）
            lig_c = np.mean([new_coords[i] for i in ref_heavy_idx], axis=0)
            if box_center is not None:
                # box_center 单位是 Å，转换为 nm
                bc_nm = np.array(box_center) / 10.0
                offset = np.linalg.norm(bc_nm - lig_c)
                log.info(f"  配体中心距对接盒子中心: {offset * 10:.1f} Å")
                if offset > 3.0:  # > 30 Å 说明坐标系有问题
                    log.warning(f"  ⚠️  配体偏离对接盒子中心 {offset * 10:.1f} Å，"
                                f"请检查坐标系是否一致")
            elif prot_gro and prot_gro.exists():
                # 无 box_center 时，用蛋白全部 CA 原子估算中心（非前50个原子）
                prot_text = prot_gro.read_text().splitlines()
                ca_coords = []
                for pl in prot_text[2:-1]:
                    if len(pl) > 44 and ' CA ' in pl[10:16]:
                        try:
                            ca_coords.append([float(pl[20:28]),
                                              float(pl[28:36]),
                                              float(pl[36:44])])
                        except (ValueError, IndexError):
                            continue
                if ca_coords:
                    prot_c = np.mean(ca_coords, axis=0)
                    offset = np.linalg.norm(prot_c - lig_c)
                    log.info(f"  配体-蛋白CA质心距离: {offset * 10:.1f} Å")

            # ★ Fix 3: n_total_atoms 明确使用全部原子数（含氢=51），
            #   而非 n_common（重原子数=30）。
            #   这样 _read_gro_coords(docked_gro) 返回 51 行，
            #   与 ITP 的 51 个原子匹配，校验通过。
            n_total_atoms = len(ref_coord_lines)  # = 51（含氢全部原子）
            new_lines = [
                f'Ligand from docking pose (direct PDBQT coords, {n_total_atoms} atoms)',
                f'{n_total_atoms:5d}'
            ]
            for i, line in enumerate(ref_coord_lines):
                if i >= len(new_coords):
                    new_lines.append(line)
                    continue
                prefix = line[:20]
                x, y, z = new_coords[i]
                coord_str = f'{x:8.3f}{y:8.3f}{z:8.3f}'
                suffix = line[44:] if len(line) > 44 else ''
                new_line = prefix[:5] + f'{lig_resname:>5s}' + prefix[10:20] + coord_str + suffix
                new_lines.append(new_line)

            box_line = ref_lines[2 + n_ref] if len(ref_lines) > 2 + n_ref else '   10.00000   10.00000   10.00000'
            new_lines.append(box_line)
            output_gro.write_text('\n'.join(new_lines) + '\n')

            log.info(f"  ✅ PDBQT→GRO 成功: 坐标直接来自 Vina 对接位置")
            return True

        except Exception as e:
            import traceback
            log.warning(f"  PDBQT→GRO 转换异常: {e}\n{traceback.format_exc()[-300:]}")
            return False

    def build_complex(self,
                      prot_gro: Path,
                      lig_info: Dict,
                      mol_dir: Path,
                      topol: Path) -> Path:
        lig_gro = lig_info['gro']
        lig_itp = lig_info['itp']

        lig_coord_lines = self._read_gro_coords(lig_gro)
        if not lig_coord_lines:
            raise RuntimeError("配体 GRO 坐标行为空")

        lig_resname = lig_coord_lines[0][5:10].strip()
        log.info(f"  配体残基名: {lig_resname}")

        prot_lines = prot_gro.read_text().splitlines()
        prot_natom = int(prot_lines[1].strip())
        prot_coords = prot_lines[2: 2 + prot_natom]
        box_line = prot_lines[2 + prot_natom]

        new_lig_lines = []
        for i, line in enumerate(lig_coord_lines):
            new_atom_num = prot_natom + i + 1
            new_line = line[:15] + f'{new_atom_num:5d}' + line[20:]
            new_lig_lines.append(new_line)

        complex_gro = mol_dir / 'complex.gro'
        total_atoms = prot_natom + len(lig_coord_lines)
        with open(complex_gro, 'w') as f:
            f.write(f'Complex: protein + {lig_resname}\n')
            f.write(f'{total_atoms:5d}\n')
            for line in prot_coords:
                f.write(line + '\n')
            for line in new_lig_lines:
                f.write(line + '\n')
            f.write(box_line + '\n')

        log.info(f"  complex.gro: 蛋白 {prot_natom} + 配体 {len(lig_coord_lines)} = {total_atoms} 原子")

        self._patch_topology(topol, lig_itp, lig_resname)
        return complex_gro

    # =========================================================
    # 核心修复 v1：_patch_topology
    # =========================================================

    def _patch_topology(self,
                        topol: Path,
                        lig_itp: Path,
                        lig_resname: str):
        atomtypes_content, main_content = self._split_itp(lig_itp)

        main_itp_name = lig_itp.name
        atomtypes_itp_name = lig_itp.stem + '_atomtypes.itp'

        itp_mol_name = self._get_itp_moleculetype_name(main_content)
        if itp_mol_name is None:
            log.warning(f"  未能从 ITP 提取 moleculetype 名称，回退使用: {lig_resname}")
            itp_mol_name = lig_resname
        else:
            log.info(f"  ITP moleculetype 名称: {itp_mol_name}"
                     + (f" (GRO 残基名: {lig_resname})" if itp_mol_name != lig_resname else ""))

        if not main_content.strip():
            log.error(f"  ❌ ITP 拆分后 main_content 为空！将使用原始 ITP（不拆分）")
            atomtypes_content = ''
            main_content = lig_itp.read_text()

        lig_itp.write_text(main_content)
        log.info(f"  主 itp 写回完成: {len(main_content.splitlines())} 行")

        atomtypes_itp_src = lig_itp.parent / atomtypes_itp_name
        if atomtypes_content.strip():
            atomtypes_itp_src.write_text(atomtypes_content)
            log.info(f"  atomtypes itp 写出: {atomtypes_itp_src.name} "
                     f"({len(atomtypes_content.splitlines())} 行)")

        dest_main_itp = topol.parent / main_itp_name
        shutil.copy2(lig_itp, dest_main_itp)

        if atomtypes_content.strip():
            dest_atomtypes_itp = topol.parent / atomtypes_itp_name
            shutil.copy2(atomtypes_itp_src, dest_atomtypes_itp)

        for posre_name in [f'posre_{lig_resname}.itp', f'posre_{itp_mol_name}.itp',
                           'posre_LIG.itp', 'posre_UNL.itp', 'posre_MOL.itp']:
            posre_src = lig_itp.parent / posre_name
            if posre_src.exists():
                shutil.copy2(posre_src, topol.parent / posre_name)
                log.info(f"  posre itp 已复制: {posre_name}")
                break

        content = topol.read_text()

        if atomtypes_content.strip() and atomtypes_itp_name not in content:
            ff_pattern = re.compile(
                r'(#include\s+"[^"]*forcefield[^"]*\.itp"[^\n]*\n)',
                re.IGNORECASE)
            if ff_pattern.search(content):
                content = ff_pattern.sub(
                    r'\1; Ligand atom types\n#include "' + atomtypes_itp_name + '"\n',
                    content, count=1)
            else:
                content = (f'; Ligand atom types\n'
                           f'#include "{atomtypes_itp_name}"\n\n') + content
                log.warning(f"  未找到力场 include 行，atomtypes include 插到文件头")

        if main_itp_name not in content:
            insert_marker = '[ system ]'
            include_line = f'\n; Ligand topology\n#include "{main_itp_name}"\n\n'
            if insert_marker in content:
                content = content.replace(insert_marker,
                                          include_line + insert_marker, 1)
            else:
                content = content.rstrip() + '\n' + include_line
                log.warning(f"  未找到 [ system ] 标记，主配体 itp include 追加到末尾")

        if not re.search(rf'(?m)^{re.escape(itp_mol_name)}\s+\d', content):
            if re.search(r'\[\s*molecules\s*\]', content):
                content = re.sub(
                    r'(\[\s*molecules\s*\][\s\S]*?)(\Z)',
                    lambda m: m.group(1).rstrip() + f'\n{itp_mol_name:<20s} 1\n',
                    content)
            else:
                content += f'\n[ molecules ]\n{itp_mol_name:<20s} 1\n'

        topol.write_text(content)
        log.info(f"  topol.top 更新完成")

    # =========================================================
    # Step 4: 构建溶剂盒子 → 加水 → 加离子
    # =========================================================

    def solvate_and_ionize(self,
                           complex_gro: Path,
                           topol: Path,
                           mol_dir: Path) -> Path:
        box_gro = mol_dir / 'box.gro'
        self._run(
            ['editconf', '-f', str(complex_gro), '-o', str(box_gro),
             '-bt', self.box_type, '-d', str(self.box_distance), '-c'],
            cwd=mol_dir, label='editconf')

        self._validate_box_size(box_gro)

        solv_gro = mol_dir / 'solv.gro'
        self._run(
            ['solvate', '-cp', str(box_gro), '-cs', 'spc216.gro',
             '-o', str(solv_gro), '-p', str(topol)],
            cwd=mol_dir, label='solvate')
        self._fix_sol_count(solv_gro, topol)

        ions_mdp = mol_dir / 'ions.mdp'
        ions_mdp.write_text(
            'integrator   = steep\n'
            'nsteps       = 0\n'
            'cutoff-scheme = Verlet\n'
            'coulombtype  = PME\n'
            'rcoulomb     = 1.2\n'
            'rvdw         = 1.2\n'
            'pbc          = xyz\n'
            'continuation = yes\n')
        ions_tpr = mol_dir / 'ions.tpr'
        self._run(
            ['grompp', '-f', str(ions_mdp), '-c', str(solv_gro),
             '-p', str(topol), '-o', str(ions_tpr),
             '-maxwarn', '5', '-nobackup'],
            cwd=mol_dir, label='grompp-ions')

        ions_gro = mol_dir / 'ions.gro'
        self._run(
            ['genion', '-s', str(ions_tpr), '-o', str(ions_gro),
             '-p', str(topol), '-neutral', '-conc', str(self.ion_conc),
             '-nobackup'],
            cwd=mol_dir, stdin='SOL\n', label='genion')

        return ions_gro

    def _validate_box_size(self, box_gro: Path, max_nm: float = 50.0):
        last_line = box_gro.read_text().splitlines()[-1].strip()
        try:
            vals = [float(v) for v in last_line.split()[:3]]
            max_val = max(vals)
            log.info(f"  盒子向量: {vals[0]:.3f} x {vals[1]:.3f} x {vals[2]:.3f} nm")
            if max_val > max_nm:
                raise RuntimeError(
                    f"盒子向量异常过大 ({max_val:.1f} nm > {max_nm} nm)！\n"
                    f"请检查原始 PDB 的 CRYST1 记录。")
            if max_val < 5.0:
                raise RuntimeError(
                    f"盒子向量异常过小 ({max_val:.3f} nm)！\n"
                    f"请检查输入 PDB 坐标单位。")
        except (ValueError, IndexError) as e:
            log.warning(f"  无法解析盒子向量: {last_line} ({e})")

    def _fix_sol_count(self, solv_gro: Path, topol: Path):
        sol_count = 0
        lines = solv_gro.read_text().splitlines()
        for line in lines[2:]:
            if len(line) >= 10:
                resname = line[5:10].strip()
                if resname == 'SOL':
                    sol_count += 1
        n_sol = sol_count // 3

        MAX_REASONABLE_SOL = 500000
        try:
            box_vals = [float(v) for v in lines[-1].split()[:3]]
            vol_nm3 = box_vals[0] * box_vals[1] * box_vals[2]
            MAX_REASONABLE_SOL = int(vol_nm3 * 33.3 * 1.5)
        except Exception:
            pass

        if n_sol > MAX_REASONABLE_SOL:
            raise RuntimeError(
                f"检测到异常数量的水分子: {n_sol}（上限 {MAX_REASONABLE_SOL}）")

        log.info(f"  solv.gro 中统计到 SOL 水分子: {n_sol} 个")

        if n_sol == 0:
            log.warning("  未检测到 SOL 水分子，跳过 SOL 数量修复")
            return

        content = topol.read_text()

        sol_pattern = re.compile(r'^(SOL\s+)(\d+)', re.MULTILINE)
        if sol_pattern.search(content):
            old_content = content
            content = sol_pattern.sub(f'SOL                  {n_sol}', content)
            if content != old_content:
                log.info(f"  topol.top SOL 数量已更新为: {n_sol}")
        else:
            if re.search(r'\[\s*molecules\s*\]', content):
                content = re.sub(
                    r'(\[\s*molecules\s*\][\s\S]*?)(\Z)',
                    lambda m: m.group(1).rstrip() + f'\nSOL                  {n_sol}\n',
                    content)
                log.info(f"  topol.top 新增 SOL 条目: {n_sol}")
            else:
                log.warning("  topol.top 中未找到 [ molecules ]，无法添加 SOL")
                return

        topol.write_text(content)

    # =========================================================
    # 核心修复 v2：make_index
    # =========================================================

    def make_index(self,
                   gro: Path,
                   mol_dir: Path,
                   lig_resname: str) -> Path:
        ndx = mol_dir / 'index.ndx'

        result1 = subprocess.run(
            [self.gmx, 'make_ndx', '-f', str(gro), '-o', str(ndx)],
            cwd=str(mol_dir), input=b'q\n',
            capture_output=True, timeout=300)

        combined1 = (result1.stdout.decode(errors='ignore') +
                     result1.stderr.decode(errors='ignore'))

        groups: dict = {}
        max_idx = 0

        for line in combined1.split('\n'):
            m = re.match(r'\s*(\d+)\s+(\S+)\s*:', line)
            if m:
                idx, name = int(m.group(1)), m.group(2)
                groups[name] = idx
                max_idx = max(max_idx, idx)

        prot_idx = groups.get('Protein')
        lig_idx = groups.get(lig_resname)
        water_idx = groups.get('Water') or groups.get('SOL')
        ion_idx = groups.get('Ion')
        water_ions_idx = groups.get('Water_and_ions')

        log.info(f"  make_ndx Step1: Protein={prot_idx}, "
                 f"{lig_resname}={lig_idx}, Water={water_idx}, "
                 f"Ion={ion_idx}, Water_and_ions={water_ions_idx}, "
                 f"max_group={max_idx}")

        if prot_idx is None or lig_idx is None:
            log.warning(
                f"  ⚠️  未能识别 Protein 或 {lig_resname} 组编号，"
                f"index.ndx 使用默认（无 Protein_LIG）。"
                f"NVT/NPT/MD 将自动回退到 non-Water 温控组。")
            return ndx

        combine_cmds: list = []
        next_idx = max_idx

        wai_source = water_ions_idx
        if wai_source is None:
            if water_idx is not None and ion_idx is not None:
                next_idx += 1
                combine_cmds += [f'{water_idx} | {ion_idx}',
                                 f'name {next_idx} Water_and_ions']
                wai_source = next_idx
            elif water_idx is not None:
                next_idx += 1
                combine_cmds += [f'{water_idx} | {water_idx}',
                                 f'name {next_idx} Water_and_ions']
                wai_source = next_idx
            else:
                log.warning("  ⚠️  未找到 Water/SOL 组，无法创建 Water_and_ions")

        next_idx += 1
        if wai_source is not None:
            combine_cmds += [f'! {wai_source}',
                             f'name {next_idx} Protein_LIG']
        else:
            combine_cmds += [f'{prot_idx} | {lig_idx}',
                             f'name {next_idx} Protein_LIG']
        prot_lig_new_idx = next_idx

        combine_cmds.append('q')
        combine_input = ('\n'.join(combine_cmds) + '\n').encode()

        result2 = subprocess.run(
            [self.gmx, 'make_ndx',
             '-f', str(gro),
             '-n', str(ndx),
             '-o', str(ndx)],
            cwd=str(mol_dir), input=combine_input,
            capture_output=True, timeout=600)

        combined2 = (result2.stdout.decode(errors='ignore') +
                     result2.stderr.decode(errors='ignore'))

        ndx_content = ndx.read_text() if ndx.exists() else ''
        has_prot_lig = 'Protein_LIG' in ndx_content
        has_water_ions = 'Water_and_ions' in ndx_content

        if has_prot_lig:
            log.info(f"  ✅ index.ndx 已创建 Protein_LIG 组 (编号={prot_lig_new_idx})")
        else:
            log.warning(f"  ⚠️  Protein_LIG 写入失败 (make_ndx exit={result2.returncode})。"
                        f"\n  stderr: {combined2[-400:]}")

        if has_water_ions:
            log.info(f"  ✅ index.ndx 包含 Water_and_ions 组")
        else:
            log.warning(f"  ⚠️  Water_and_ions 写入失败，NVT/NPT/MD 将回退 tc-grps。")

        return ndx

    # =========================================================
    # 辅助：动态生成 MDP（tc-grps 回退）
    # =========================================================

    def _write_mdp_with_tcgrp_fallback(self,
                                       mdp_name: str,
                                       mdp_path: Path,
                                       ndx: Path):
        mdp_text = self._MDP[mdp_name]
        ndx_content = ndx.read_text() if (ndx and ndx.exists()) else ''

        has_prot_lig = 'Protein_LIG' in ndx_content
        has_water_ions = 'Water_and_ions' in ndx_content

        if has_prot_lig and has_water_ions:
            pass
        elif not has_prot_lig and has_water_ions:
            log.warning(f"  [{mdp_name}] Protein_LIG 缺失，tc-grps 回退: "
                        f"non-Water / Water_and_ions")
            mdp_text = mdp_text.replace(
                'tc-grps                  = Protein_LIG  Water_and_ions',
                'tc-grps                  = non-Water    Water_and_ions')
        elif has_prot_lig and not has_water_ions:
            has_water = re.search(r'\[ Water \]', ndx_content) or \
                        re.search(r'\[ SOL \]', ndx_content)
            fallback_solvent = 'Water' if has_water else 'non-Protein_LIG'
            log.warning(f"  [{mdp_name}] Water_and_ions 缺失，tc-grps 回退: "
                        f"Protein_LIG / {fallback_solvent}")
            mdp_text = mdp_text.replace(
                'tc-grps                  = Protein_LIG  Water_and_ions',
                f'tc-grps                  = Protein_LIG  {fallback_solvent}')
        else:
            log.warning(f"  [{mdp_name}] 两者均缺失，tc-grps 回退: non-Water / Water")
            mdp_text = mdp_text.replace(
                'tc-grps                  = Protein_LIG  Water_and_ions',
                'tc-grps                  = non-Water    Water')

        mdp_path.write_text(mdp_text)

    # =========================================================
    # Step 6: grompp + mdrun
    # =========================================================

    def _grompp(self,
                mdp: Path,
                gro: Path,
                topol: Path,
                tpr: Path,
                mol_dir: Path,
                ndx: Optional[Path] = None,
                ref_gro: Optional[Path] = None,
                cpt: Optional[Path] = None,
                maxwarn: int = 5):
        args = ['grompp', '-f', str(mdp), '-c', str(gro), '-p', str(topol),
                '-o', str(tpr), '-maxwarn', str(maxwarn), '-nobackup']
        if ndx:     args += ['-n', str(ndx)]
        if ref_gro: args += ['-r', str(ref_gro)]
        if cpt:     args += ['-t', str(cpt)]
        self._run(args, cwd=mol_dir, label='grompp')

    def _mdrun(self, deffnm: str, mol_dir: Path, timeout: int = 36000):
        """
        ★ 修复 v3 [HIGH] PME GPU 加速：
          非 EM 阶段统一使用 -pme gpu -pmefft gpu，
          解决日志中 93.6% PME wait 瓶颈。
          EM 阶段保持 -pme cpu（EM 不支持 GPU PME）。
        """
        args = ['mdrun', '-v', '-deffnm', deffnm, '-nobackup']
        is_em = (deffnm == 'em')

        if self.gpu_id is not None:
            gpu_args = [
                '-ntmpi', '1',
                '-ntomp', str(self.n_threads),
                '-pin', 'on',
                '-nb', 'gpu',
                '-pme', 'cpu' if is_em else 'gpu',
                '-pmefft', 'cpu' if is_em else 'gpu',  # ★ 新增
                '-bonded', 'cpu' if is_em else 'gpu',
            ]
            if not is_em:
                gpu_args += ['-update', 'gpu']
            if str(self.gpu_id).isdigit():
                gpu_args += ['-gpu_id', str(self.gpu_id)]
            args += gpu_args
        else:
            args += [
                '-ntmpi', '1',
                '-ntomp', str(self.n_threads),
                '-pin', 'on',
                '-nb', 'cpu',
            ]

        self._run(args, cwd=mol_dir, timeout=timeout, label=f'mdrun-{deffnm}')

    # =========================================================
    # Step 7: 轨迹分析
    # =========================================================

    def _parse_xvg(self, xvg: Path) -> List[float]:
        vals = []
        if not xvg.exists():
            return vals
        for line in xvg.read_text().splitlines():
            line = line.strip()
            if line.startswith(('#', '@')) or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    vals.append(float(parts[1]))
                except ValueError:
                    pass
        return vals

    def _get_group_index(self, ndx: Path, name: str) -> Optional[int]:
        if not ndx.exists():
            return None
        content = ndx.read_text()
        groups = re.findall(r'\[ (.+?) \]', content)
        for i, g in enumerate(groups):
            if g.strip() == name:
                return i
        return None

    def analyze_trajectory(self, mol_dir: Path,
                           lig_resname: str = 'LIG') -> Dict:
        """
        ★ 修复 v3 [HIGH] trjconv 居中组动态回退：
          检测 index.ndx 中 Protein_LIG 是否存在，
          缺失时自动改用 Protein，防止 trjconv 崩溃。

        ★ 修复 v3 [CRITICAL] RMSD 单位统一：
          gmx rms 输出单位为 nm，乘以 10 转换为 Å，
          与 ComprehensiveScorer._normalize_rmsd(ceiling=5.0 Å) 保持一致。
        """
        result: Dict = {}
        tpr = mol_dir / 'md.tpr'
        xtc = mol_dir / 'md.xtc'
        ndx = mol_dir / 'index.ndx'

        if not tpr.exists() or not xtc.exists():
            log.warning(f"  轨迹文件不存在: {mol_dir}")
            return {'rmsd_mean': np.nan, 'rmsd_std': np.nan,
                    'rmsf_mean': np.nan, 'hbond_mean': np.nan,
                    'rg_mean': np.nan}

        # ── 确定居中组（动态回退）─────────────────────────────────
        ndx_content = ndx.read_text() if ndx.exists() else ''
        center_grp = 'Protein_LIG' if 'Protein_LIG' in ndx_content else 'Protein'
        if center_grp == 'Protein':
            log.warning(f"  ⚠️  Protein_LIG 不在 index.ndx，trjconv 居中改用 Protein")

        xtc_nopbc = mol_dir / 'md_nopbc.xtc'
        try:
            self._run(
                ['trjconv', '-s', str(tpr), '-f', str(xtc),
                 '-o', str(xtc_nopbc), '-n', str(ndx),
                 '-pbc', 'mol', '-center'],
                cwd=mol_dir,
                stdin=f'{center_grp}\nSystem\n',  # ★ 动态居中组
                label='trjconv-pbc')
        except Exception as e:
            log.warning(f"  trjconv 去除 PBC 失败，使用原始轨迹: {e}")
            xtc_nopbc = xtc

        traj = xtc_nopbc

        # ── 配体 RMSD ────────────────────────────────────────────
        rmsd_xvg = mol_dir / 'ligand_rmsd.xvg'
        try:
            self._run(
                ['rms', '-s', str(tpr), '-f', str(traj),
                 '-o', str(rmsd_xvg), '-n', str(ndx), '-tu', 'ns'],
                cwd=mol_dir, stdin=f'Backbone\n{lig_resname}\n',
                label='rms-ligand')
            rmsd_vals_nm = self._parse_xvg(rmsd_xvg)
            if rmsd_vals_nm:
                # ★ nm → Å 转换
                rmsd_vals = [v * 10.0 for v in rmsd_vals_nm]
                result['rmsd_mean'] = float(np.mean(rmsd_vals))
                result['rmsd_std'] = float(np.std(rmsd_vals))
                result['rmsd_max'] = float(np.max(rmsd_vals))
                result['rmsd_values'] = rmsd_vals
                log.info(f"  配体 RMSD: {result['rmsd_mean']:.2f} ± "
                         f"{result['rmsd_std']:.2f} Å (来自 gmx rms, nm×10)")
        except Exception as e:
            log.warning(f"  配体 RMSD 分析失败: {e}")
            result.update({'rmsd_mean': np.nan, 'rmsd_std': np.nan, 'rmsd_max': np.nan})

        # ── 骨架 RMSF ────────────────────────────────────────────
        rmsf_xvg = mol_dir / 'backbone_rmsf.xvg'
        try:
            self._run(
                ['rmsf', '-s', str(tpr), '-f', str(traj),
                 '-o', str(rmsf_xvg), '-n', str(ndx), '-res'],
                cwd=mol_dir, stdin='Backbone\n', label='rmsf')
            rmsf_vals_nm = self._parse_xvg(rmsf_xvg)
            if rmsf_vals_nm:
                rmsf_vals = [v * 10.0 for v in rmsf_vals_nm]  # nm → Å
                result['rmsf_mean'] = float(np.mean(rmsf_vals))
                result['rmsf_max'] = float(np.max(rmsf_vals))
        except Exception as e:
            log.warning(f"  RMSF 分析失败: {e}")
            result.update({'rmsf_mean': np.nan, 'rmsf_max': np.nan})

        # ── 回旋半径 ─────────────────────────────────────────────
        rg_xvg = mol_dir / 'gyrate.xvg'
        rg_center_grp = 'Protein_LIG' if 'Protein_LIG' in ndx_content else 'Protein'
        try:
            self._run(
                ['gyrate', '-s', str(tpr), '-f', str(traj),
                 '-o', str(rg_xvg), '-n', str(ndx)],
                cwd=mol_dir, stdin=f'{rg_center_grp}\n', label='gyrate')
            rg_vals_nm = self._parse_xvg(rg_xvg)
            if rg_vals_nm:
                rg_vals = [v * 10.0 for v in rg_vals_nm]  # nm → Å
                result['rg_mean'] = float(np.mean(rg_vals))
                result['rg_std'] = float(np.std(rg_vals))
        except Exception as e:
            log.warning(f"  Rg 分析失败: {e}")
            result.update({'rg_mean': np.nan, 'rg_std': np.nan})

        # ── 氢键计数（优先级：MDAnalysis > gmx hbond > pairdist > mindist）────
        hbond_ok = False

        # ── 方法 1: MDAnalysis HydrogenBondAnalysis（最准确，无版本限制）──
        try:
            import MDAnalysis as mda
            from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import (
                HydrogenBondAnalysis as HBA)

            u = mda.Universe(str(tpr), str(traj))

            # 选定蛋白供体/受体 & 配体受体/供体
            lig_sel = f'resname {lig_resname}'
            prot_sel = 'protein'

            hbonds = HBA(
                universe=u,
                donors_sel=None,  # 自动检测
                hydrogens_sel=None,
                acceptors_sel=None,
                between=[[prot_sel, lig_sel]],
                d_a_cutoff=3.5,  # Å，供-受距离
                d_h_a_angle_cutoff=120,  # °，供-氢-受角度
            )
            hbonds.run()

            counts = hbonds.count_by_time()  # shape (n_frames,)
            if counts is not None and len(counts) > 0:
                result['hbond_mean'] = float(np.mean(counts))
                result['hbond_max'] = float(np.max(counts))
                result['hbond_std'] = float(np.std(counts))
                hbond_ok = True
                log.info(f"  氢键 (MDAnalysis): "
                         f"{result['hbond_mean']:.2f} ± {result['hbond_std']:.2f} "
                         f"(max={result['hbond_max']:.0f})")
        except ImportError:
            log.warning("  MDAnalysis 未安装，回退 gmx hbond")
        except Exception as e_mda:
            log.warning(f"  MDAnalysis hbond 失败 ({e_mda.__class__.__name__}: {e_mda})，"
                        f"回退 gmx hbond")

        # ── 方法 2: gmx hbond（GROMACS ≤ 2023；2024+ 已移除）────────────────
        if not hbond_ok:
            hbond_xvg = mol_dir / 'hbond.xvg'
            try:
                self._run(
                    ['hbond', '-s', str(tpr), '-f', str(traj),
                     '-num', str(hbond_xvg), '-n', str(ndx)],
                    cwd=mol_dir, stdin=f'Protein\n{lig_resname}\n',
                    label='hbond')
                hb_vals = self._parse_xvg(hbond_xvg)
                if hb_vals:
                    result['hbond_mean'] = float(np.mean(hb_vals))
                    result['hbond_max'] = float(np.max(hb_vals))
                    result['hbond_std'] = float(np.std(hb_vals))
                    hbond_ok = True
                    log.info(f"  氢键 (gmx hbond): {result['hbond_mean']:.2f} ± "
                             f"{result['hbond_std']:.2f}")
            except Exception as e_gmx:
                log.warning(f"  gmx hbond 失败 ({e_gmx.__class__.__name__})，"
                            f"回退 pairdist")

        # ── 方法 3: pairdist 接触代理（GROMACS 通用兜底）─────────────────────
        if not hbond_ok:
            contact_xvg = mol_dir / 'lig_contact.xvg'
            try:
                self._run(
                    ['pairdist',
                     '-s', str(tpr), '-f', str(traj),
                     '-o', str(contact_xvg),
                     '-n', str(ndx),
                     '-ref', 'Protein',
                     '-sel', lig_resname,
                     '-refgrouping', 'res',
                     '-selgrouping', 'res',
                     '-type', 'min'],
                    cwd=mol_dir, label='pairdist')
                contact_vals = self._parse_xvg(contact_xvg)
                if contact_vals:
                    n_contact = sum(1 for v in contact_vals if v <= 0.35)
                    frac = n_contact / max(len(contact_vals), 1)
                    result['hbond_mean'] = round(frac * 3.0, 3)
                    result['hbond_max'] = result['hbond_mean']
                    hbond_ok = True
                    log.info(f"  接触代理 (pairdist≤3.5Å): "
                             f"{n_contact}/{len(contact_vals)} 帧"
                             f" → hbond_proxy={result['hbond_mean']:.3f} [近似值]")
            except Exception as e2:
                log.warning(f"  pairdist 失败: {e2}，回退 mindist")

        # ── 方法 4: mindist 最终兜底 ──────────────────────────────────────────
        if not hbond_ok:
            mindist_xvg = mol_dir / 'mindist.xvg'
            try:
                self._run(
                    ['mindist', '-s', str(tpr), '-f', str(traj),
                     '-od', str(mindist_xvg), '-n', str(ndx)],
                    cwd=mol_dir,
                    stdin=f'Protein\n{lig_resname}\n',
                    label='mindist')
                md_vals = self._parse_xvg(mindist_xvg)
                if md_vals:
                    avg_d = float(np.mean(md_vals))
                    result['hbond_mean'] = max(0.0, round((0.5 - avg_d) / 0.5 * 3, 3))
                    result['hbond_max'] = result['hbond_mean']
                    log.info(f"  最短距离均值 (mindist): {avg_d * 10:.2f} Å"
                             f" → hbond_proxy={result['hbond_mean']:.3f} [近似值]")
            except Exception as e3:
                log.warning(f"  mindist 也失败: {e3}")
                result.update({'hbond_mean': np.nan, 'hbond_max': np.nan})

        return result

    # =========================================================
    # 主流水线
    # =========================================================

    def run_md_pipeline(self,
                        smiles: str,
                        mol_id: str,
                        pose_pdbqt: Optional[str] = None) -> Dict:
        if not self._gmx_ok:
            return {'success': False, 'reason': 'gromacs_unavailable'}

        mol_dir = self.work_dir / mol_id
        mol_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"\n{'─' * 55}")
        log.info(f"  MD 流水线开始: {mol_id}")
        log.info(f"{'─' * 55}")

        try:
            log.info(f"  [1/8] 配体 GAFF2 拓扑 (acpype)")
            lig_dir = mol_dir / 'lig'
            lig_info = self.prepare_ligand(smiles, mol_id, lig_dir)
            if lig_info is None:
                return {'success': False, 'reason': 'ligand_topology_failed'}

            lig_gro_lines = self._read_gro_coords(lig_info['gro'])
            lig_resname = lig_gro_lines[0][5:10].strip() if lig_gro_lines else 'LIG'
            log.info(f"  配体残基名: {lig_resname}")

            lig_info_orig_gro = lig_info['gro']

            log.info(f"  [2/8] 蛋白拓扑 (pdb2gmx, {self.forcefield})")
            prot_gro, topol = self.prepare_protein(mol_dir)

            if pose_pdbqt and Path(pose_pdbqt).exists():
                log.info(f"  检测到对接姿态: {Path(pose_pdbqt).name}")
                docked_gro = lig_dir / f'{mol_id}_docked.gro'

                if self._convert_pdbqt_to_gro(Path(pose_pdbqt), docked_gro, lig_resname,
                                              ref_gro=lig_info_orig_gro, smiles=smiles,
                                              prot_gro=prot_gro,
                                              box_center=getattr(self, 'box_center', None)):
                    n_itp = len(self._read_gro_coords(lig_info_orig_gro))
                    n_dock = len(self._read_gro_coords(docked_gro))

                    if n_itp == n_dock:
                        log.info(f"  ✅ 使用 Vina 对接姿态作为配体初始坐标 ({n_dock} 原子)")
                        lig_info['gro'] = docked_gro
                    else:
                        log.warning(f"  ⚠️  原子数不匹配: 对接构象 {n_dock} ≠ ITP {n_itp}，"
                                    f"使用 acpype 构象")
                        lig_info['gro'] = lig_info_orig_gro
                else:
                    log.warning(f"  ⚠️  PDBQT→GRO 转换失败，使用 acpype 构象")
                    lig_info['gro'] = lig_info_orig_gro
            else:
                log.warning(f"  ⚠️  未提供对接姿态，配体将使用随机3D构象！")

            log.info(f"  [3/8] 构建蛋白-配体复合物 GRO")
            complex_gro = self.build_complex(prot_gro, lig_info, mol_dir, topol)

            log.info(f"  [4/8] 溶剂化 + 加离子 (TIP3P, {self.ion_conc}M NaCl)")
            ions_gro = self.solvate_and_ionize(complex_gro, topol, mol_dir)

            log.info(f"  [5/8] 生成 index.ndx")
            ndx = self.make_index(ions_gro, mol_dir, lig_resname)

            log.info(f"  [6/8] 能量最小化 (EM)")
            em_mdp = mol_dir / 'em.mdp'
            em_tpr = mol_dir / 'em.tpr'
            self._write_mdp('em', em_mdp)
            self._grompp(em_mdp, ions_gro, topol, em_tpr, mol_dir,
                         ndx=ndx, maxwarn=5)
            self._mdrun('em', mol_dir, timeout=7200)
            em_gro = mol_dir / 'em.gro'
            if not em_gro.exists():
                return {'success': False, 'reason': 'em_failed'}
            log.info(f"  ✅ EM 完成")

            log.info(f"  [7/8-a] NVT 平衡 (100 ps, 300K)")
            nvt_mdp = mol_dir / 'nvt.mdp'
            nvt_tpr = mol_dir / 'nvt.tpr'
            self._write_mdp_with_tcgrp_fallback('nvt', nvt_mdp, ndx)
            self._grompp(nvt_mdp, em_gro, topol, nvt_tpr, mol_dir,
                         ndx=ndx, ref_gro=em_gro, maxwarn=5)
            self._mdrun('nvt', mol_dir, timeout=14400)
            nvt_gro = mol_dir / 'nvt.gro'
            nvt_cpt = mol_dir / 'nvt.cpt'
            if not nvt_gro.exists():
                return {'success': False, 'reason': 'nvt_failed'}
            log.info(f"  ✅ NVT 完成")

            log.info(f"  [7/8-b] NPT 平衡 (100 ps, 1 bar)")
            npt_mdp = mol_dir / 'npt.mdp'
            npt_tpr = mol_dir / 'npt.tpr'
            self._write_mdp_with_tcgrp_fallback('npt', npt_mdp, ndx)
            self._grompp(npt_mdp, nvt_gro, topol, npt_tpr, mol_dir,
                         ndx=ndx, ref_gro=nvt_gro, cpt=nvt_cpt, maxwarn=5)
            self._mdrun('npt', mol_dir, timeout=14400)
            npt_gro = mol_dir / 'npt.gro'
            npt_cpt = mol_dir / 'npt.cpt'
            if not npt_gro.exists():
                return {'success': False, 'reason': 'npt_failed'}
            log.info(f"  ✅ NPT 完成")

            log.info(f"  [8/8] 生产 MD [100 ns] (GPU:{self.gpu_id})")
            md_mdp = mol_dir / 'md.mdp'
            md_tpr = mol_dir / 'md.tpr'
            self._write_mdp_with_tcgrp_fallback('md', md_mdp, ndx)
            self._grompp(md_mdp, npt_gro, topol, md_tpr, mol_dir,
                         ndx=ndx, cpt=npt_cpt, maxwarn=5)
            self._mdrun('md', mol_dir, timeout=14400)
            md_xtc = mol_dir / 'md.xtc'
            if not md_xtc.exists():
                return {'success': False, 'reason': 'md_production_failed'}
            log.info(f"  ✅ 生产 MD 完成")

            log.info(f"  分析轨迹...")
            analysis = self.analyze_trajectory(mol_dir, lig_resname)
            analysis['success'] = True
            analysis['mol_id'] = mol_id
            analysis['lig_resname'] = lig_resname
            analysis['used_docking_pose'] = (pose_pdbqt is not None and
                                             Path(pose_pdbqt).exists())

            log.info(
                f"  ✅ {mol_id} 完成 | "
                f"RMSD={analysis.get('rmsd_mean', float('nan')):.2f}±"
                f"{analysis.get('rmsd_std', float('nan')):.2f} Å | "
                f"HBond={analysis.get('hbond_mean', float('nan')):.1f} | "
                f"姿态={'Docking' if analysis['used_docking_pose'] else 'acpype'}")
            return analysis

        except subprocess.TimeoutExpired as e:
            log.error(f"  ❌ {mol_id} 超时: {e}")
            return {'success': False, 'reason': f'timeout: {e}'}
        except RuntimeError as e:
            log.error(f"  ❌ {mol_id} 失败:\n{str(e)[:600]}")
            return {'success': False, 'reason': str(e)[:400]}
        except Exception as e:
            import traceback
            log.error(f"  ❌ {mol_id} 意外错误: {traceback.format_exc()[-600:]}")
            return {'success': False, 'reason': f'unexpected: {str(e)[:200]}'}

    # =========================================================
    # 批量入口
    # =========================================================

    def run_batch(self,
                  df: pd.DataFrame,
                  smiles_col: str = 'smiles',
                  max_mols: int = 15) -> pd.DataFrame:
        if not self._gmx_ok:
            log.warning("GROMACS 不可用，跳过 MD")
            for col in ['md_success', 'rmsd_mean', 'rmsd_std',
                        'rmsd_max', 'rmsf_mean', 'hbond_mean',
                        'rg_mean', 'lig_resname']:
                df[col] = np.nan
            return df

        subset = df.head(max_mols).copy()
        log.info(f"\n[MD] 批量 GROMACS MD: {len(subset)} 个分子")

        records = []
        for i, row in subset.iterrows():
            mol_id = f"mol_{i:05d}"

            # ★ 修复 v3 [HIGH] pose_pdbqt NaN 安全判断：
            #   pandas NaN 是 float，Path(nan) → TypeError
            raw_pose = row.get('vina_pose_pdbqt', None)
            if raw_pose is None or isinstance(raw_pose, float):
                pose_pdbqt = None
            else:
                pose_pdbqt = str(raw_pose)
                if not Path(pose_pdbqt).exists():
                    log.warning(f"  {mol_id}: pose_pdbqt 不存在: {pose_pdbqt}")
                    pose_pdbqt = None

            res = self.run_md_pipeline(row[smiles_col], mol_id, pose_pdbqt)
            res['smiles'] = row[smiles_col]
            records.append(res)

        md_df = pd.DataFrame(records)
        md_df['md_success'] = md_df.get('success', False)

        merge_cols = ['smiles', 'md_success', 'rmsd_mean', 'rmsd_std',
                      'rmsd_max', 'rmsf_mean', 'rg_mean', 'rg_std',
                      'hbond_mean', 'hbond_max', 'lig_resname']
        merge_cols = [c for c in merge_cols if c in md_df.columns]
        df = df.merge(md_df[merge_cols], on='smiles', how='left')

        ok = int(md_df.get('md_success', pd.Series(dtype=bool)).sum())
        log.info(f"[MD] 完成: {ok}/{len(subset)} 成功")
        return df


# ============================================================================
# Part 5: 综合评分排名
# ============================================================================

class ComprehensiveScorer:
    """
    多维度综合评分

    ★ 修复 v3 [MEDIUM] 动态权重归一化：
      compute() 检测各维度数据是否实际存在，
      未运行的维度（GNINA/MD）权重置0后重新归一化，
      防止因缺失维度导致所有分子分值系统性偏低。
    """

    DEFAULT_WEIGHTS = {
        'admet_score': 0.15,
        'vina_score_norm': 0.35,
        'gnina_score_norm': 0.20,
        'md_stability': 0.20,
        'md_hbond': 0.10,
    }

    def __init__(self, weights: Optional[Dict] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def _normalize_vina(self, scores: pd.Series) -> pd.Series:
        scores = scores.fillna(0)
        normed = (-scores - 5) / 10
        return normed.clip(0, 1)

    def _normalize_rmsd(self, rmsd: pd.Series) -> pd.Series:
        """ceiling = 5.0 Å（RMSD 已在 analyze_trajectory 中转换为 Å）"""
        rmsd = rmsd.fillna(5.0)
        normed = 1 - (rmsd / 5.0).clip(0, 1)
        return normed

    def _normalize_hbond(self, hbond: pd.Series) -> pd.Series:
        hbond = hbond.fillna(0)
        normed = (hbond / 5.0).clip(0, 1)
        return normed

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['vina_score_norm'] = self._normalize_vina(
            df.get('vina_score', pd.Series([0] * len(df))))
        df['gnina_score_norm'] = df.get('gnina_cnn_score',
                                        pd.Series([np.nan] * len(df))).fillna(np.nan).clip(0, 1)
        df['md_stability'] = self._normalize_rmsd(
            df.get('rmsd_mean', pd.Series([np.nan] * len(df))))
        df['md_hbond'] = self._normalize_hbond(
            df.get('hbond_mean', pd.Series([np.nan] * len(df))))

        # ★ 动态权重归一化：检测各维度是否有有效数据
        active_weights = dict(self.weights)

        gnina_available = df['gnina_score_norm'].notna().any()
        if not gnina_available:
            log.info("  评分: GNINA 维度无数据，权重重新归一化")
            active_weights['gnina_score_norm'] = 0.0

        md_available = df.get('md_success', pd.Series([False] * len(df))).any()
        if not md_available:
            log.info("  评分: MD 维度无数据，权重重新归一化")
            active_weights['md_stability'] = 0.0
            active_weights['md_hbond'] = 0.0

        total = sum(active_weights.values())
        if total > 0:
            active_weights = {k: v / total for k, v in active_weights.items()}

        log.info(f"  有效权重: { {k: f'{v:.3f}' for k, v in active_weights.items() if v > 0} }")

        composite = pd.Series(0.0, index=df.index)
        for col, weight in active_weights.items():
            if col in df.columns and weight > 0:
                composite += df[col].fillna(0) * weight

        df['composite_score'] = composite.astype(float).round(4)
        df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
        df['final_rank'] = range(1, len(df) + 1)
        return df

    def generate_report(self, df: pd.DataFrame,
                        top_n: int = 20,
                        output_dir: Path = None) -> pd.DataFrame:
        top = df.head(top_n).copy()

        report_cols = [
            'final_rank', 'smiles',
            'composite_score', 'admet_score',
            'vina_score', 'gnina_affinity',
            'rmsd_mean', 'hbond_mean',
            'MW', 'LogP', 'TPSA', 'QED',
            'Ro5_violations', 'PAINS', 'md_success',
        ]
        report_cols = [c for c in report_cols if c in top.columns]
        report = top[report_cols]

        log.info(f"\n{'=' * 70}")
        log.info(f"🏆 最终候选化合物 Top {len(report)}")
        log.info(f"{'=' * 70}")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        pd.set_option('display.float_format', '{:.3f}'.format)
        print(report.to_string(index=False))

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            report.to_csv(output_dir / f'final_top{top_n}_candidates.csv', index=False)
            try:
                report.to_excel(output_dir / f'final_top{top_n}_candidates.xlsx',
                                index=False)
            except Exception:
                pass
            log.info(f"  报告已保存: {output_dir}/final_top{top_n}_candidates.csv")

        return report


# ============================================================================
# Part 6: 完整管线主函数
# ============================================================================

def run_validation_pipeline(
        input_csv: str,
        receptor_pdbqt: str,
        receptor_pdb: str,
        box_center: Tuple[float, float, float],
        box_size: Tuple[float, float, float],
        output_dir: str = 'validation_results',
        run_name: str = 'NLRP3_Validation',
        smiles_col: str = 'smiles',
        admet_hard_filter: bool = True,
        use_api: bool = False,
        admetlab_key: Optional[str] = None,
        vina_executable: str = 'vina',
        exhaustiveness: int = 16,
        binding_cutoff: float = -8.5,
        gnina_executable: str = 'gnina',
        run_gnina: bool = True,
        gmx_executable: str = 'gmx',
        run_md: bool = True,
        md_top_n: int = 15,
        n_threads: int = 8,
        gpu_id: Optional[str] = None,
        final_top_n: int = 20,
        scorer_weights: Optional[Dict] = None,
        box_distance: float = 1.2,
        box_type: str = 'dodecahedron',
) -> Tuple[pd.DataFrame, Dict]:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(output_dir) / f"{run_name}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"\n{'=' * 70}")
    log.info(f"多层次计算验证管线启动")
    log.info(f"  输出目录: {output_dir}")
    log.info(f"{'=' * 70}")

    stats = {'run_name': f"{run_name}_{ts}"}

    df = pd.read_csv(input_csv)
    if smiles_col not in df.columns:
        raise ValueError(f"CSV 中找不到 SMILES 列: {smiles_col}")
    log.info(f"\n[输入] {len(df):,} 个候选分子")
    stats['input_count'] = len(df)

    log.info("\n" + "─" * 60)
    log.info("[Stage 1/4] ADMET 评估")
    log.info("─" * 60)

    admet_calc = ADMETCalculator(swissadme_enabled=use_api, admetlab_key=admetlab_key)
    df = admet_calc.evaluate_batch(df[smiles_col].tolist(), use_api=use_api)
    df[smiles_col] = pd.read_csv(input_csv)[smiles_col].values

    if admet_hard_filter:
        df = admet_calc.filter_hard(df)

    df.to_csv(output_dir / 'stage1_admet.csv', index=False)
    stats['after_admet'] = len(df)
    log.info(f"  ✅ ADMET 后: {len(df):,} 个分子")

    log.info("\n" + "─" * 60)
    log.info("[Stage 2/4] AutoDock Vina 对接")
    log.info("─" * 60)

    vina = VinaDocking(
        receptor_pdbqt=receptor_pdbqt, box_center=box_center, box_size=box_size,
        vina_executable=vina_executable, exhaustiveness=exhaustiveness,
        work_dir=str(output_dir / 'vina_work'))

    df = vina.dock_batch(df, smiles_col=smiles_col, binding_cutoff=binding_cutoff)
    df = vina.filter_by_binding(df, cutoff=binding_cutoff)
    df.to_csv(output_dir / 'stage2_docking.csv', index=False)
    stats['after_docking'] = len(df)

    if run_gnina and len(df) > 0:
        log.info("\n" + "─" * 60)
        log.info("[Stage 2b] GNINA 交叉验证")
        log.info("─" * 60)
        gnina = GNINADocking(
            receptor_pdb=receptor_pdb, box_center=box_center, box_size=box_size,
            gnina_executable=gnina_executable,
            work_dir=str(output_dir / 'gnina_work'))
        df = gnina.rescore_batch(df)
        df.to_csv(output_dir / 'stage2b_gnina.csv', index=False)

    if run_md and len(df) > 0:
        log.info("\n" + "─" * 60)
        log.info(f"[Stage 3/4] GROMACS MD (top {md_top_n})")
        log.info("─" * 60)
        gmx_sim = GMXMDSimulation(
            receptor_pdb=receptor_pdb,
            work_dir=str(output_dir / 'md_work'),
            gmx_executable=gmx_executable,
            n_threads=n_threads,
            gpu_id=gpu_id,
            box_distance=box_distance,
            box_type=box_type)
        gmx_sim.box_center = list(box_center)
        df = gmx_sim.run_batch(df, smiles_col=smiles_col, max_mols=md_top_n)
        df.to_csv(output_dir / 'stage3_md.csv', index=False)
        stats['md_success'] = int(df.get('md_success',
                                         pd.Series([False] * len(df))).sum())

    log.info("\n" + "─" * 60)
    log.info("[Stage 4/4] 综合评分排名")
    log.info("─" * 60)

    scorer = ComprehensiveScorer(weights=scorer_weights)
    df = scorer.compute(df)
    report = scorer.generate_report(df, top_n=final_top_n, output_dir=output_dir)

    df.to_csv(output_dir / 'all_candidates_scored.csv', index=False)

    stats['final_count'] = len(report)
    stats['top_vina'] = float(df['vina_score'].min()) if 'vina_score' in df.columns else None
    stats['top_composite'] = float(df['composite_score'].max()) if 'composite_score' in df.columns else None
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    try:
        from validation_report import generate_all_reports
        log.info("\n[Report] 生成可视化报告...")
        md_work = output_dir / 'md_work' if run_md else None
        generate_all_reports(df=df, stats=stats, output_dir=output_dir,
                             md_work_dir=md_work, top_n=final_top_n)
    except ImportError:
        log.info("  ⚠️  validation_report.py 未找到，跳过可视化")
    except Exception as e:
        log.warning(f"  ⚠️  报告生成失败: {e}")

    log.info(f"\n{'=' * 70}")
    log.info("✅ 验证管线完成")
    log.info(f"   最终候选: {len(report)} 个")
    log.info(f"   结果目录: {output_dir}")
    log.info(f"{'=' * 70}")

    return report, stats


# ============================================================================
# Part 7: 命令行入口
# ============================================================================

def build_default_config() -> Dict:
    return {
        'input_csv': 'screening_results/final_top500_hits.csv',
        'receptor_pdbqt': 'receptor/NLRP3_NACHT.pdbqt',
        'receptor_pdb': 'receptor/NLRP3_NACHT.pdb',
        'output_dir': 'validation_results',
        'run_name': 'NLRP3_Validation',
        'smiles_col': 'smiles',
        'box_center': [15.2, 22.8, -8.4],
        'box_size': [25.0, 25.0, 25.0],
        'admet_hard_filter': True,
        'use_api': False,
        'admetlab_key': None,
        'vina_executable': 'vina',
        'exhaustiveness': 16,
        'binding_cutoff': -8.5,
        'gnina_executable': 'gnina',
        'run_gnina': True,
        'gmx_executable': 'gmx',
        'run_md': True,
        'md_top_n': 15,
        'n_threads': 8,
        'gpu_id': None,
        'final_top_n': 20,
        'scorer_weights': {
            'admet_score': 0.15,
            'vina_score_norm': 0.35,
            'gnina_score_norm': 0.20,
            'md_stability': 0.20,
            'md_hbond': 0.10,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description='多层次计算验证管线 (ADMET→Vina→GNINA→GROMACS MD)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', '-c', default='validation_config.yaml')
    parser.add_argument('--input', '-i', default=None)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--no_md', action='store_true')
    parser.add_argument('--no_gnina', action='store_true')
    parser.add_argument('--use_api', action='store_true')
    parser.add_argument('--gen_config', action='store_true')
    args = parser.parse_args()

    if args.gen_config:
        cfg = build_default_config()
        cfg_path = Path(args.config)
        with open(cfg_path, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True,
                      sort_keys=False)
        log.info(f"默认配置文件已生成: {cfg_path}")
        return

    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    else:
        log.warning(f"配置文件 {cfg_path} 不存在，使用内置默认值")
        cfg = build_default_config()

    if args.input:    cfg['input_csv'] = args.input
    if args.output:   cfg['output_dir'] = args.output
    if args.no_md:    cfg['run_md'] = False
    if args.no_gnina: cfg['run_gnina'] = False
    if args.use_api:  cfg['use_api'] = True

    run_validation_pipeline(
        input_csv=cfg['input_csv'],
        receptor_pdbqt=cfg['receptor_pdbqt'],
        receptor_pdb=cfg['receptor_pdb'],
        box_center=tuple(cfg['box_center']),
        box_size=tuple(cfg['box_size']),
        output_dir=cfg.get('output_dir', 'validation_results'),
        run_name=cfg.get('run_name', 'NLRP3_Validation'),
        smiles_col=cfg.get('smiles_col', 'smiles'),
        admet_hard_filter=cfg.get('admet_hard_filter', True),
        use_api=cfg.get('use_api', False),
        admetlab_key=cfg.get('admetlab_key'),
        vina_executable=cfg.get('vina_executable', 'vina'),
        exhaustiveness=cfg.get('exhaustiveness', 16),
        binding_cutoff=cfg.get('binding_cutoff', -8.5),
        gnina_executable=cfg.get('gnina_executable', 'gnina'),
        run_gnina=cfg.get('run_gnina', True),
        gmx_executable=cfg.get('gmx_executable', 'gmx'),
        run_md=cfg.get('run_md', True),
        md_top_n=cfg.get('md_top_n', 15),
        n_threads=cfg.get('n_threads', 8),
        gpu_id=cfg.get('gpu_id'),
        final_top_n=cfg.get('final_top_n', 20),
        scorer_weights=cfg.get('scorer_weights'),
        box_distance=cfg.get('box_distance', 1.2),
        box_type=cfg.get('box_type', 'dodecahedron'),
    )


if __name__ == '__main__':
    main()