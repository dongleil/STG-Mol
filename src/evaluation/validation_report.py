#!/usr/bin/env python3
"""
validation_report.py
====================
多层次计算验证结果可视化与报告生成模块

功能:
  - ADMET 雷达图 / 分布图
  - Vina 对接结合能分布
  - GNINA vs Vina 相关性散点图
  - MD RMSD / RMSF 轨迹图
  - 综合评分排名条形图
  - 最终候选化合物 2D 结构网格图
  - 完整 HTML 报告

依赖:
  pip install matplotlib seaborn plotly rdkit pandas numpy openpyxl
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

warnings.filterwarnings('ignore')

# ── 全局绘图风格 ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       ['Noto Sans CJK JP', 'DejaVu Sans'],
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'figure.dpi':        150,
})
PALETTE = {
    'primary':   '#2E86AB',
    'secondary': '#A23B72',
    'success':   '#2DC653',
    'warning':   '#F18F01',
    'danger':    '#C73E1D',
    'neutral':   '#6B7280',
}


# ============================================================================
# 1. ADMET 可视化
# ============================================================================

def plot_admet_overview(df: pd.DataFrame, output_path: Path, top_n: int = 50):
    """
    ADMET 概览：
      左上 - 属性分布（MW / LogP / TPSA / QED）
      右上 - PAINS / hERG 风险饼图
      下   - ADMET 综合分前 top_n 条形图
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('ADMET 评估概览', fontsize=16, fontweight='bold', y=0.98)
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.4)

    # ── 属性分布 ─────────────────────────────────────────────────────
    prop_axes = {
        'MW':   (gs[0, 0], [150, 600], '#e0f2fe'),
        'LogP': (gs[0, 1], [-2, 7],    '#fce7f3'),
        'TPSA': (gs[0, 2], [0, 200],   '#fef9c3'),
        'QED':  (gs[0, 3], [0, 1],     '#d1fae5'),
    }
    for prop, (spec, xlim, color) in prop_axes.items():
        ax = fig.add_subplot(spec)
        vals = df[prop].dropna() if prop in df.columns else pd.Series()
        if len(vals) > 0:
            ax.hist(vals, bins=30, color=color, edgecolor='white', linewidth=0.5)
            ax.axvline(vals.mean(), color=PALETTE['danger'], lw=1.5,
                       linestyle='--', label=f'Mean={vals.mean():.1f}')
        ax.set_xlabel(prop, fontsize=10)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title(f'{prop} Distribution', fontsize=10)
        if xlim:
            ax.set_xlim(xlim)
        ax.legend(fontsize=8)

    # ── 风险饼图 ──────────────────────────────────────────────────────
    ax_pie = fig.add_subplot(gs[1, :2])
    categories = []
    if 'PAINS' in df.columns:
        n_pains  = df['PAINS'].sum()
        n_herg   = df.get('hERG_structural_risk', pd.Series([False]*len(df))).sum()
        n_clean  = len(df) - n_pains - n_herg + min(n_pains, n_herg)
        sizes    = [max(n_clean, 0), n_pains, n_herg]
        labels   = ['Clean', 'PAINS hit', 'hERG risk']
        colors   = [PALETTE['success'], PALETTE['danger'], PALETTE['warning']]
        sizes_nz = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
        if sizes_nz:
            sizes, labels, colors = zip(*sizes_nz)
            ax_pie.pie(sizes, labels=labels, colors=colors,
                       autopct='%1.1f%%', startangle=90,
                       textprops={'fontsize': 9})
    ax_pie.set_title('结构警示分布', fontsize=11)

    # ── 综合分排名 ────────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, 2:])
    if 'admet_score' in df.columns and len(df) > 0:
        top = df.nlargest(min(top_n, len(df)), 'admet_score')
        colors_bar = [PALETTE['success'] if s >= 0.6
                      else PALETTE['warning'] if s >= 0.4
                      else PALETTE['danger']
                      for s in top['admet_score']]
        ax_bar.barh(range(len(top)), top['admet_score'],
                    color=colors_bar, edgecolor='white', linewidth=0.3)
        ax_bar.axvline(0.5, color='gray', lw=1, linestyle='--', alpha=0.7)
        ax_bar.set_xlabel('ADMET Score', fontsize=10)
        ax_bar.set_ylabel('Molecule Index', fontsize=9)
        ax_bar.set_title(f'Top {len(top)} ADMET 综合分', fontsize=11)
        ax_bar.invert_yaxis()

    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_radar_admet(df: pd.DataFrame, output_path: Path, top_n: int = 8):
    """
    Top N 候选分子的 ADMET 属性雷达图（蜘蛛图）
    """
    props = ['QED', 'Ro5_pass_score', 'TPSA_score',
             'LogP_score', 'Fsp3', 'admet_score']

    # 构建归一化分值
    plot_df = df.head(top_n).copy()
    plot_df['Ro5_pass_score'] = 1 - plot_df.get('Ro5_violations',
                                                  pd.Series([2]*len(plot_df))) / 4
    tpsa = plot_df.get('TPSA', pd.Series([140]*len(plot_df)))
    plot_df['TPSA_score'] = 1 - (tpsa / 140).clip(0, 1)
    logp = plot_df.get('LogP', pd.Series([5]*len(plot_df)))
    plot_df['LogP_score'] = 1 - ((logp - 3).abs() / 4).clip(0, 1)
    plot_df['Fsp3'] = plot_df.get('Fsp3', pd.Series([0.3]*len(plot_df)))

    available = [p for p in props if p in plot_df.columns]
    if len(available) < 3:
        return

    N      = len(available)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10),
                             subplot_kw=dict(polar=True))
    fig.suptitle('候选分子 ADMET 雷达图', fontsize=14, fontweight='bold')

    cmap = plt.cm.tab10
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        if idx >= 8:
            break
        ax    = axes[idx // 4][idx % 4]
        vals  = [float(row.get(p, 0)) for p in available]
        vals += vals[:1]
        color = cmap(idx / 8)
        ax.plot(angles, vals, 'o-', linewidth=2, color=color)
        ax.fill(angles, vals, alpha=0.25, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available, size=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=6)
        rank = row.get('final_rank', idx + 1)
        ax.set_title(f'Rank #{int(rank)}', size=10, pad=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


# ============================================================================
# 2. 对接结果可视化
# ============================================================================

def plot_docking_results(df: pd.DataFrame, output_path: Path):
    """
    对接结果可视化：
      左   - Vina 分值分布直方图
      中上 - Vina vs GNINA 相关性散点图
      中下 - Vina vs ADMET 气泡图
      右   - Top 20 结合能排名
    """
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('分子对接结果分析', fontsize=15, fontweight='bold', y=0.99)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    has_vina  = 'vina_score' in df.columns and df['vina_score'].notna().any()
    has_gnina = 'gnina_cnn_score' in df.columns and df['gnina_cnn_score'].notna().any()

    # ── Vina 分值分布 ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[:, 0])
    if has_vina:
        scores = df['vina_score'].dropna()
        ax1.hist(scores, bins=30, color=PALETTE['primary'],
                 edgecolor='white', linewidth=0.5, orientation='horizontal')
        ax1.axhline(-8.5, color=PALETTE['danger'], lw=2, linestyle='--',
                    label='截断 -8.5 kcal/mol')
        ax1.axhline(scores.mean(), color=PALETTE['warning'], lw=1.5,
                    linestyle=':', label=f'均值 {scores.mean():.2f}')
        ax1.set_ylabel('Vina Score (kcal/mol)', fontsize=11)
        ax1.set_xlabel('Count', fontsize=11)
        ax1.set_title('Vina 结合能分布', fontsize=12)
        ax1.legend(fontsize=9)
        ax1.invert_yaxis()

    # ── Vina vs GNINA 相关性 ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if has_vina and has_gnina:
        sub = df[df['vina_score'].notna() & df['gnina_cnn_score'].notna()]
        sc  = ax2.scatter(sub['vina_score'], sub['gnina_cnn_score'],
                          c=sub.get('admet_score',
                                    pd.Series([0.5]*len(sub))),
                          cmap='RdYlGn', alpha=0.7, s=50, edgecolors='none')
        plt.colorbar(sc, ax=ax2, label='ADMET Score')
        ax2.axvline(-8.5, color=PALETTE['danger'], lw=1.5, linestyle='--',
                    alpha=0.7)
        ax2.set_xlabel('Vina Score (kcal/mol)', fontsize=10)
        ax2.set_ylabel('GNINA CNN Score', fontsize=10)
        ax2.set_title('Vina vs GNINA 交叉验证', fontsize=11)
        # 计算相关系数
        if len(sub) > 3:
            r = np.corrcoef(sub['vina_score'], sub['gnina_cnn_score'])[0, 1]
            ax2.text(0.05, 0.92, f'r = {r:.3f}', transform=ax2.transAxes,
                     fontsize=9, color=PALETTE['secondary'])
    else:
        ax2.text(0.5, 0.5, 'GNINA 数据不可用', ha='center', va='center',
                 transform=ax2.transAxes, color='gray', fontsize=11)
        ax2.set_title('Vina vs GNINA 交叉验证', fontsize=11)

    # ── Vina vs QED 气泡图 ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    if has_vina and 'QED' in df.columns:
        sub = df[df['vina_score'].notna() & df['QED'].notna()]
        sizes = sub.get('MW', pd.Series([300]*len(sub))).fillna(300)
        sizes = ((sizes - sizes.min()) / (sizes.max() - sizes.min() + 1) * 150 + 30)
        sc2 = ax3.scatter(sub['vina_score'], sub['QED'],
                          s=sizes, c=sub.get('admet_score',
                                              pd.Series([0.5]*len(sub))),
                          cmap='RdYlGn', alpha=0.6, edgecolors='none')
        plt.colorbar(sc2, ax=ax3, label='ADMET Score')
        ax3.axvline(-8.5, color=PALETTE['danger'], lw=1.5, linestyle='--',
                    alpha=0.7)
        ax3.set_xlabel('Vina Score (kcal/mol)', fontsize=10)
        ax3.set_ylabel('QED', fontsize=10)
        ax3.set_title('结合能 vs 类药性 (气泡大小=MW)', fontsize=11)

    # ── Top 20 结合能排名 ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[:, 2])
    if has_vina:
        top20 = df.nsmallest(min(20, len(df)), 'vina_score')
        colors = [PALETTE['success'] if s <= -9.5
                  else PALETTE['warning'] if s <= -8.5
                  else PALETTE['neutral']
                  for s in top20['vina_score']]
        bars = ax4.barh(range(len(top20)), top20['vina_score'],
                        color=colors, edgecolor='white', linewidth=0.3)
        ax4.axvline(-8.5, color=PALETTE['danger'], lw=1.5, linestyle='--',
                    alpha=0.8, label='截断 -8.5')
        ax4.set_xlabel('Vina Score (kcal/mol)', fontsize=10)
        ax4.set_title('Top 20 结合能', fontsize=12)
        ax4.set_yticks(range(len(top20)))
        ax4.set_yticklabels([f'#{i+1}' for i in range(len(top20))], fontsize=8)
        ax4.invert_yaxis()
        ax4.legend(fontsize=9)
        # 标注数值
        for bar, val in zip(bars, top20['vina_score']):
            ax4.text(val - 0.1, bar.get_y() + bar.get_height() / 2,
                     f'{val:.2f}', ha='right', va='center', fontsize=7.5,
                     color='white', fontweight='bold')

    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


# ============================================================================
# 3. MD 结果可视化
# ============================================================================

def plot_md_results(df: pd.DataFrame, output_path: Path,
                    md_work_dir: Optional[Path] = None):
    """
    MD 模拟结果：
      左上  - RMSD 均值 + 误差棒（按排名）
      右上  - RMSF 均值对比
      左下  - 氢键均值对比
      右下  - 各分子 RMSD 轨迹（若 xvg 文件可用）
    """
    md_cols_available = ('rmsd_mean' in df.columns or
                         'rmsf_mean' in df.columns)
    if not md_cols_available:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'MD 数据不可用（未运行或全部失败）',
                ha='center', va='center', transform=ax.transAxes,
                color='gray', fontsize=12)
        ax.set_title('MD 模拟结果', fontsize=13)
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        return

    md_df = df[df.get('md_success', pd.Series([False]*len(df))) == True].copy()
    if len(md_df) == 0:
        md_df = df[df['rmsd_mean'].notna()].copy()

    fig  = plt.figure(figsize=(18, 12))
    fig.suptitle('GROMACS 100ns MD 模拟分析', fontsize=15,
                 fontweight='bold', y=0.99)
    gs   = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    mol_labels = [f"Rank#{int(r)}" for r in
                  md_df.get('final_rank', range(1, len(md_df)+1))]

    # ── RMSD 均值 ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if 'rmsd_mean' in md_df.columns and md_df['rmsd_mean'].notna().any():
        rmsd_mean = md_df['rmsd_mean'].fillna(0)
        rmsd_std  = md_df.get('rmsd_std', pd.Series([0]*len(md_df))).fillna(0)
        colors = [PALETTE['success'] if v < 2.0
                  else PALETTE['warning'] if v < 3.5
                  else PALETTE['danger']
                  for v in rmsd_mean]
        ax1.bar(range(len(md_df)), rmsd_mean, yerr=rmsd_std,
                color=colors, capsize=4, edgecolor='white', linewidth=0.5)
        ax1.axhline(2.0, color=PALETTE['warning'], lw=1.5, linestyle='--',
                    label='稳定阈值 2.0 Å')
        ax1.axhline(3.5, color=PALETTE['danger'], lw=1.5, linestyle='--',
                    label='不稳定阈值 3.5 Å')
        ax1.set_xticks(range(len(md_df)))
        ax1.set_xticklabels(mol_labels, rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('RMSD (Å)', fontsize=11)
        ax1.set_title('配体 RMSD（均值 ± 标准差）', fontsize=12)
        ax1.legend(fontsize=9)

    # ── RMSF 对比 ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if 'rmsf_mean' in md_df.columns and md_df['rmsf_mean'].notna().any():
        rmsf_vals = md_df['rmsf_mean'].fillna(0)
        ax2.bar(range(len(md_df)), rmsf_vals,
                color=PALETTE['primary'], edgecolor='white', linewidth=0.5)
        ax2.set_xticks(range(len(md_df)))
        ax2.set_xticklabels(mol_labels, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('RMSF (Å)', fontsize=11)
        ax2.set_title('蛋白骨架 RMSF（越低越稳定）', fontsize=12)

    # ── 氢键均值 ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if 'hbond_mean' in md_df.columns and md_df['hbond_mean'].notna().any():
        hb_vals = md_df['hbond_mean'].fillna(0)
        colors  = [PALETTE['success'] if v >= 2
                   else PALETTE['warning'] if v >= 1
                   else PALETTE['neutral']
                   for v in hb_vals]
        ax3.bar(range(len(md_df)), hb_vals, color=colors,
                edgecolor='white', linewidth=0.5)
        ax3.set_xticks(range(len(md_df)))
        ax3.set_xticklabels(mol_labels, rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('平均氢键数', fontsize=11)
        ax3.set_title('蛋白-配体氢键（均值）', fontsize=12)

    # ── RMSD 轨迹（从 xvg 文件读取） ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if md_work_dir and md_work_dir.exists():
        plotted = 0
        cmap    = plt.cm.tab10
        xvg_files = sorted(md_work_dir.rglob('ligand_rmsd.xvg'))
        for j, xvg in enumerate(xvg_files[:8]):
            times, rmsds = [], []
            for line in xvg.read_text().splitlines():
                if line.startswith(('#', '@')) or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        times.append(float(parts[0]))
                        rmsds.append(float(parts[1]))
                    except ValueError:
                        pass
            if times:
                mol_id = xvg.parent.name
                ax4.plot(times, rmsds, linewidth=1, color=cmap(j / 8),
                         alpha=0.8, label=mol_id)
                plotted += 1
        if plotted > 0:
            ax4.axhline(2.0, color='gray', lw=1, linestyle='--', alpha=0.6)
            ax4.set_xlabel('Time (ns)', fontsize=11)
            ax4.set_ylabel('RMSD (Å)', fontsize=11)
            ax4.set_title('配体 RMSD 轨迹', fontsize=12)
            ax4.legend(fontsize=8, ncol=2)
        else:
            ax4.text(0.5, 0.5, 'RMSD 轨迹文件不可用',
                     ha='center', va='center', transform=ax4.transAxes,
                     color='gray')
            ax4.set_title('配体 RMSD 轨迹', fontsize=12)
    else:
        ax4.text(0.5, 0.5, 'MD 工作目录不可用',
                 ha='center', va='center', transform=ax4.transAxes, color='gray')
        ax4.set_title('配体 RMSD 轨迹', fontsize=12)

    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


# ============================================================================
# 4. 综合评分可视化
# ============================================================================

def plot_final_ranking(df: pd.DataFrame, output_path: Path, top_n: int = 20):
    """
    最终综合评分排名图：
      主图   - 堆叠条形图（各维度贡献）
      插图   - 综合分 vs Vina 分散点图
    """
    top = df.head(min(top_n, len(df))).copy()
    if len(top) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('综合评分最终排名', fontsize=15, fontweight='bold', y=0.99)

    # ── 堆叠条形图 ───────────────────────────────────────────────────
    ax = axes[0]
    score_cols = {
        'admet_score':      ('ADMET',    PALETTE['success']),
        'vina_score_norm':  ('Vina',     PALETTE['primary']),
        'gnina_score_norm': ('GNINA',    PALETTE['secondary']),
        'md_stability':     ('MD稳定性', PALETTE['warning']),
        'md_hbond':         ('氢键',     '#8B5CF6'),
    }
    bottoms = np.zeros(len(top))
    for col, (label, color) in score_cols.items():
        vals = top.get(col, pd.Series([0]*len(top))).fillna(0).values
        ax.barh(range(len(top)), vals, left=bottoms,
                label=label, color=color, edgecolor='white', linewidth=0.3)
        bottoms += vals

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"Rank #{int(r)}" for r in
                        top.get('final_rank', range(1, len(top)+1))],
                       fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('综合分（各维度叠加）', fontsize=11)
    ax.set_title(f'Top {len(top)} 候选化合物综合评分', fontsize=13)
    ax.legend(loc='lower right', fontsize=9)

    # 标注综合分
    for i, (_, row) in enumerate(top.iterrows()):
        cs = row.get('composite_score', 0)
        ax.text(bottoms[i] + 0.01, i, f'{cs:.3f}',
                va='center', fontsize=8, color=PALETTE['neutral'])

    # ── 综合分 vs Vina 散点图 ─────────────────────────────────────────
    ax2 = axes[1]
    if 'vina_score' in df.columns and 'composite_score' in df.columns:
        sub = df[df['vina_score'].notna() & df['composite_score'].notna()]
        is_top = sub.index.isin(top.index)

        ax2.scatter(sub[~is_top]['vina_score'],
                    sub[~is_top]['composite_score'],
                    c='lightgray', s=40, alpha=0.5, label='其余分子')
        sc = ax2.scatter(sub[is_top]['vina_score'],
                         sub[is_top]['composite_score'],
                         c=sub[is_top].get('admet_score',
                                            pd.Series([0.5]*is_top.sum())),
                         cmap='RdYlGn', s=100, alpha=0.9,
                         edgecolors='black', linewidth=0.5,
                         label=f'Top {top_n}')
        plt.colorbar(sc, ax=ax2, label='ADMET Score')

        # 标注 top 5
        for _, row in sub[is_top].head(5).iterrows():
            ax2.annotate(f"#{int(row.get('final_rank', '?'))}",
                         (row['vina_score'], row['composite_score']),
                         textcoords='offset points', xytext=(5, 5),
                         fontsize=8, color=PALETTE['secondary'])

        ax2.axvline(-8.5, color=PALETTE['danger'], lw=1.5, linestyle='--',
                    alpha=0.7, label='Vina 截断')
        ax2.set_xlabel('Vina Score (kcal/mol)', fontsize=11)
        ax2.set_ylabel('综合评分', fontsize=11)
        ax2.set_title('Vina 结合能 vs 综合评分', fontsize=13)
        ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


# ============================================================================
# 5. 2D 结构网格图
# ============================================================================

def plot_structure_grid(df: pd.DataFrame, output_path: Path, top_n: int = 20):
    """
    Top N 候选化合物 2D 结构网格图（带评分标注）
    """
    top = df.head(min(top_n, len(df))).copy()
    if len(top) == 0:
        return

    mols, legends = [], []
    for _, row in top.iterrows():
        smi = row.get('smiles', '')
        mol = Chem.MolFromSmiles(smi) if smi else None
        mols.append(mol)
        rank   = int(row.get('final_rank', 0))
        cs     = row.get('composite_score', 0)
        vina   = row.get('vina_score', float('nan'))
        qed    = row.get('QED', float('nan'))
        rmsd   = row.get('rmsd_mean', float('nan'))
        legend = (f"Rank#{rank}\n"
                  f"Comp={cs:.3f}\n"
                  f"Vina={vina:.2f}" if not np.isnan(vina) else f"Rank#{rank}\n"
                  f"Comp={cs:.3f}")
        if not np.isnan(rmsd):
            legend += f"\nRMSD={rmsd:.2f}Å"
        legends.append(legend)

    # 过滤无效分子
    valid = [(m, l) for m, l in zip(mols, legends) if m is not None]
    if not valid:
        return
    mols, legends = zip(*valid)

    cols = 4
    rows = math.ceil(len(mols) / cols)
    img  = Draw.MolsToGridImage(
        list(mols), molsPerRow=cols, subImgSize=(350, 280),
        legends=list(legends), useSVG=False)

    img.save(str(output_path))


# ============================================================================
# 6. HTML 报告生成
# ============================================================================

def generate_html_report(df: pd.DataFrame,
                          stats: Dict,
                          output_dir: Path,
                          figures: Dict[str, Path],
                          top_n: int = 20) -> Path:
    """
    生成完整 HTML 报告
    """
    import base64

    def img_to_b64(path: Path) -> str:
        if path and path.exists():
            return base64.b64encode(path.read_bytes()).decode()
        return ''

    top     = df.head(top_n)
    ts      = stats.get('run_name', 'Unknown')

    # 表格 HTML
    table_cols = ['final_rank', 'smiles', 'composite_score',
                  'vina_score', 'admet_score', 'rmsd_mean',
                  'hbond_mean', 'QED', 'MW', 'LogP']
    table_cols = [c for c in table_cols if c in top.columns]
    table_rows = ''
    for _, row in top.iterrows():
        cells = ''
        for col in table_cols:
            val = row.get(col, '')
            if isinstance(val, float):
                val = f'{val:.3f}' if not np.isnan(val) else '-'
            cells += f'<td>{val}</td>'
        table_rows += f'<tr>{cells}</tr>\n'
    table_header = ''.join(f'<th>{c}</th>' for c in table_cols)

    # 图片嵌入
    img_sections = ''
    fig_titles = {
        'admet':     'ADMET 评估概览',
        'radar':     'ADMET 雷达图',
        'docking':   '分子对接分析',
        'md':        'MD 模拟结果',
        'ranking':   '综合评分排名',
        'structures':'候选化合物 2D 结构',
    }
    for key, title in fig_titles.items():
        path = figures.get(key)
        b64  = img_to_b64(path) if path else ''
        if b64:
            img_sections += f'''
            <section class="figure-section">
              <h2>{title}</h2>
              <img src="data:image/png;base64,{b64}"
                   style="max-width:100%; border-radius:8px;
                          box-shadow:0 2px 8px rgba(0,0,0,0.15);">
            </section>
'''

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>多层次计算验证报告 - {ts}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body  {{ font-family: -apple-system, "Segoe UI", sans-serif;
              background: #f0f4f8; color: #1a202c; line-height: 1.6; }}
    header {{ background: linear-gradient(135deg, #1e3a5f 0%, #2E86AB 100%);
               color: white; padding: 2.5rem 2rem; }}
    header h1 {{ font-size: 2rem; margin-bottom: 0.3rem; }}
    header p  {{ opacity: 0.85; font-size: 0.95rem; }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem; }}
    .stats-grid {{ display: grid;
                   grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                   gap: 1rem; margin: 1.5rem 0 2rem; }}
    .stat-card {{ background: white; border-radius: 10px;
                  padding: 1.2rem; text-align: center;
                  box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
    .stat-card .value {{ font-size: 2rem; font-weight: 700;
                          color: #2E86AB; line-height: 1.2; }}
    .stat-card .label {{ font-size: 0.8rem; color: #6B7280;
                          margin-top: 0.3rem; }}
    .figure-section {{ background: white; border-radius: 10px;
                        padding: 1.5rem; margin-bottom: 1.5rem;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
    .figure-section h2 {{ font-size: 1.2rem; margin-bottom: 1rem;
                           color: #1e3a5f; border-left: 4px solid #2E86AB;
                           padding-left: 0.75rem; }}
    .table-section {{ background: white; border-radius: 10px;
                       padding: 1.5rem; margin-bottom: 1.5rem;
                       box-shadow: 0 1px 4px rgba(0,0,0,0.1);
                       overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
    thead tr {{ background: #1e3a5f; color: white; }}
    th, td {{ padding: 0.6rem 0.8rem; text-align: left;
               border-bottom: 1px solid #e5e7eb; }}
    tbody tr:nth-child(even) {{ background: #f9fafb; }}
    tbody tr:hover {{ background: #eff6ff; }}
    tbody tr:first-child td {{ font-weight: 600; background: #fff8e1; }}
    footer {{ text-align: center; color: #9CA3AF; font-size: 0.8rem;
               padding: 2rem; }}
  </style>
</head>
<body>
<header>
  <h1>🔬 多层次计算验证报告</h1>
  <p>运行标识: {ts} &nbsp;|&nbsp; 生成时间: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</header>
<div class="container">

  <!-- 统计摘要 -->
  <div class="stats-grid">
    <div class="stat-card">
      <div class="value">{stats.get("input_count", "N/A")}</div>
      <div class="label">输入候选分子</div>
    </div>
    <div class="stat-card">
      <div class="value">{stats.get("after_admet", "N/A")}</div>
      <div class="label">ADMET 过滤后</div>
    </div>
    <div class="stat-card">
      <div class="value">{stats.get("after_docking", "N/A")}</div>
      <div class="label">对接通过 (-8.5)</div>
    </div>
    <div class="stat-card">
      <div class="value">{stats.get("md_success", "N/A")}</div>
      <div class="label">MD 模拟成功</div>
    </div>
    <div class="stat-card">
      <div class="value">{stats.get("final_count", "N/A")}</div>
      <div class="label">最终候选</div>
    </div>
    <div class="stat-card">
      <div class="value">{f"{stats.get('top_vina', 0):.2f}" if stats.get("top_vina") else "N/A"}</div>
      <div class="label">最佳 Vina 分</div>
    </div>
    <div class="stat-card">
      <div class="value">{f"{stats.get('top_composite', 0):.3f}" if stats.get("top_composite") else "N/A"}</div>
      <div class="label">最高综合分</div>
    </div>
  </div>

  <!-- 图表 -->
  {img_sections}

  <!-- 最终候选表格 -->
  <div class="table-section">
    <h2 style="font-size:1.2rem; margin-bottom:1rem; color:#1e3a5f;
               border-left:4px solid #2E86AB; padding-left:0.75rem;">
      🏆 Top {len(top)} 最终候选化合物
    </h2>
    <table>
      <thead><tr>{table_header}</tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>

</div>
<footer>
  <p>本报告由多层次计算验证管线自动生成 &nbsp;|&nbsp; NLRP3 抑制剂发现项目</p>
</footer>
</body>
</html>'''

    html_path = output_dir / 'validation_report.html'
    html_path.write_text(html, encoding='utf-8')
    return html_path


# ============================================================================
# 7. 主报告生成接口
# ============================================================================

def generate_all_reports(df: pd.DataFrame,
                          stats: Dict,
                          output_dir: Path,
                          md_work_dir: Optional[Path] = None,
                          top_n: int = 20) -> Dict[str, Path]:
    """
    生成所有图表 + HTML 报告，返回各图表路径字典
    """
    import math as _math
    global math
    math = _math

    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    figures = {}

    print("  生成 ADMET 概览图...")
    try:
        p = fig_dir / 'admet_overview.png'
        plot_admet_overview(df, p)
        figures['admet'] = p
    except Exception as e:
        print(f"    ⚠️  {e}")

    print("  生成 ADMET 雷达图...")
    try:
        p = fig_dir / 'admet_radar.png'
        plot_radar_admet(df, p, top_n=min(8, top_n))
        figures['radar'] = p
    except Exception as e:
        print(f"    ⚠️  {e}")

    print("  生成对接结果图...")
    try:
        p = fig_dir / 'docking_results.png'
        plot_docking_results(df, p)
        figures['docking'] = p
    except Exception as e:
        print(f"    ⚠️  {e}")

    print("  生成 MD 结果图...")
    try:
        p = fig_dir / 'md_results.png'
        plot_md_results(df, p, md_work_dir=md_work_dir)
        figures['md'] = p
    except Exception as e:
        print(f"    ⚠️  {e}")

    print("  生成综合排名图...")
    try:
        p = fig_dir / 'final_ranking.png'
        plot_final_ranking(df, p, top_n=top_n)
        figures['ranking'] = p
    except Exception as e:
        print(f"    ⚠️  {e}")

    print("  生成 2D 结构网格图...")
    try:
        p = fig_dir / 'structures_grid.png'
        plot_structure_grid(df, p, top_n=top_n)
        figures['structures'] = p
    except Exception as e:
        print(f"    ⚠️  {e}")

    print("  生成 HTML 报告...")
    try:
        html_path = generate_html_report(df, stats, output_dir,
                                          figures, top_n=top_n)
        figures['html'] = html_path
        print(f"  ✅ HTML 报告: {html_path}")
    except Exception as e:
        print(f"    ⚠️  HTML 生成失败: {e}")

    return figures