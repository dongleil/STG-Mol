#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graphical_abstract.py — render the paper's Graphical Abstract as
SVG (vector) + PNG (300 dpi) for Elsevier submission.

Layout (left → right):

  Modalities              Fusion & MTL heads              Deployment
  ┌──────────┐            ┌────────────────┐             ┌──────────┐
  │ 1D SMILES│──╮         │  Hierarchical  │──▶ Activity │  8 novel │
  │ Mol2Vec  │  │         │  Tri-modal     │            │NLRP3     │
  └──────────┘  │         │  Fusion:       │──▶ 5-ADMET │candidates│
  ┌──────────┐  │         │  CrossAttn +   │           (Docking,   │
  │ 2D Graph │──┼──────▶ │  Gated +       │            │MD, MMPBSA,│
  │ D-MPNN   │  │         │  Bilinear +    │            │joint     │
  └──────────┘  │         │  ImportanceNet │            │ADMET)    │
  ┌──────────┐  │         │                │            │           │
  │ 3D Geom  │──╯         │                │──▶ (screen │           │
  │ SphereNet│            │                │   8.8 M   │           │
  └──────────┘            └────────────────┘   ZINC)    └──────────┘

Elsevier GA specs (2026):
  * Landscape 1328 × 531 px minimum; 200 dpi preferred
  * Vector or high-res raster (we output both)
  * No caption, no title (visible content only)
  * All text ≥ 8 pt when printed at final size

Usage:
    python paper/graphical_abstract.py
    # outputs: paper/graphical_abstract.svg + paper/graphical_abstract.png

Requires only matplotlib.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# Elsevier canonical GA aspect ratio 1328 x 531 → 2.5 : 1 landscape
FIG_W_IN = 11.0
FIG_H_IN = 4.4
DPI      = 300


# Palette — colour-blind-safe, print-friendly (Okabe-Ito)
C_1D  = '#0072B2'  # blue     (SMILES / Mol2Vec)
C_2D  = '#009E73'  # green    (D-MPNN)
C_3D  = '#D55E00'  # vermilion (SphereNet)
C_FUS = '#332288'  # dark purple (fusion box)
C_ACT = '#CC79A7'  # magenta   (activity head)
C_ADM = '#E69F00'  # orange    (ADMET head)
C_OUT = '#117733'  # dark green (candidates)
C_TXT = '#111111'
C_FRAME = '#333333'


def _rounded_box(ax, xy, w, h, fc, ec=C_FRAME, lw=1.2, alpha=1.0,
                  pad=0.02, rounding=0.02):
    """Rounded rectangular box centred at (xy) with (w, h)."""
    x, y = xy
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f'round,pad={pad},rounding_size={rounding}',
        linewidth=lw, edgecolor=ec, facecolor=fc, alpha=alpha)
    ax.add_patch(box)
    return box


def _arrow(ax, xy0, xy1, color=C_FRAME, lw=1.6, style='->'):
    a = FancyArrowPatch(xy0, xy1,
                         arrowstyle=style, mutation_scale=14,
                         color=color, linewidth=lw,
                         shrinkA=4, shrinkB=4)
    ax.add_patch(a)
    return a


def _label(ax, xy, text, size=9, weight='normal', color=C_TXT, ha='center',
            va='center', style='normal'):
    ax.text(xy[0], xy[1], text, ha=ha, va=va, fontsize=size,
             fontweight=weight, color=color, fontstyle=style)


def draw():
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    # ───────────────────────────────── Column 1: 3 modality inputs ─────
    mod_x = 1.15
    mod_w, mod_h = 1.9, 0.72
    _rounded_box(ax, (mod_x, 3.15), mod_w, mod_h, fc=C_1D, alpha=0.85)
    _label(ax, (mod_x, 3.32), '1D  Sequence', size=10, weight='bold',
            color='white')
    _label(ax, (mod_x, 3.00), 'SMILES • Mol2Vec', size=8, color='white')

    _rounded_box(ax, (mod_x, 2.05), mod_w, mod_h, fc=C_2D, alpha=0.85)
    _label(ax, (mod_x, 2.22), '2D  Topology', size=10, weight='bold',
            color='white')
    _label(ax, (mod_x, 1.90), 'Molecular graph • D-MPNN', size=8,
            color='white')

    _rounded_box(ax, (mod_x, 0.95), mod_w, mod_h, fc=C_3D, alpha=0.85)
    _label(ax, (mod_x, 1.12), '3D  Geometry', size=10, weight='bold',
            color='white')
    _label(ax, (mod_x, 0.80), 'ETKDG conformer • SphereNet', size=8,
            color='white')

    # Column-header label
    _label(ax, (mod_x, 3.75), 'Molecular representations',
            size=10, weight='bold', color=C_TXT)

    # ─────────────────────────────── Column 2: Hierarchical fusion box ─
    fus_x = 4.7
    fus_w, fus_h = 2.6, 2.4
    _rounded_box(ax, (fus_x, 2.05), fus_w, fus_h, fc=C_FUS, alpha=0.15,
                  ec=C_FUS, lw=1.6)
    _label(ax, (fus_x, 3.35), 'Hierarchical\nTri-modal Fusion',
            size=11, weight='bold', color=C_FUS)

    # Fusion components stack
    comp_w, comp_h = 2.1, 0.34
    comps = [
        ('Cross-Modal Attention', 2.75),
        ('Gated Fusion Unit',     2.30),
        ('Low-Rank Bilinear',     1.85),
        ('Importance Net (sample-level)', 1.40),
    ]
    for name, y in comps:
        _rounded_box(ax, (fus_x, y), comp_w, comp_h, fc='white',
                      ec=C_FUS, lw=1.0, alpha=0.95)
        _label(ax, (fus_x, y), name, size=8, color=C_FUS)

    # Arrows: modalities → fusion box
    for y in (3.15, 2.05, 0.95):
        _arrow(ax, (mod_x + mod_w / 2, y), (fus_x - fus_w / 2, 2.05),
                lw=1.4)

    # ─────────────────────────────── Column 3: Two output heads ───────
    head_x = 7.6
    head_w, head_h = 1.5, 0.55

    _rounded_box(ax, (head_x, 2.75), head_w, head_h, fc=C_ACT, alpha=0.85)
    _label(ax, (head_x, 2.85), 'Activity', size=10, weight='bold',
            color='white')
    _label(ax, (head_x, 2.62), 'NLRP3 inhibitor', size=7.5, color='white')

    _rounded_box(ax, (head_x, 1.90), head_w, head_h, fc=C_ADM, alpha=0.90)
    _label(ax, (head_x, 2.00), '5 × ADMET', size=10, weight='bold',
            color='white')
    _label(ax, (head_x, 1.78), 'Lipinski / QED / PAINS /', size=7,
            color='white')
    _label(ax, (head_x, 1.66), 'SA / LogP', size=7, color='white')

    # Cascaded VS box below the heads
    vs_w, vs_h = 1.5, 0.55
    _rounded_box(ax, (head_x, 1.00), vs_w, vs_h, fc='#88CCEE', alpha=0.60,
                  ec=C_FRAME, lw=1.0)
    _label(ax, (head_x, 1.11), 'Cascaded VS', size=9.5, weight='bold',
            color=C_TXT)
    _label(ax, (head_x, 0.88), '8.8 M ZINC → 142 → 8', size=7.5,
            color=C_TXT)

    _arrow(ax, (fus_x + fus_w / 2, 2.55), (head_x - head_w / 2, 2.75), lw=1.4)
    _arrow(ax, (fus_x + fus_w / 2, 2.05), (head_x - head_w / 2, 1.90), lw=1.4)
    _arrow(ax, (fus_x + fus_w / 2, 1.55), (head_x - head_w / 2, 1.00), lw=1.4)

    # ─────────────────────────────── Column 4: Final candidates panel ──
    out_x = 9.35
    out_w, out_h = 1.35, 2.75
    _rounded_box(ax, (out_x, 2.05), out_w, out_h, fc=C_OUT, alpha=0.90)
    _label(ax, (out_x, 3.20), '8 novel', size=11, weight='bold',
            color='white')
    _label(ax, (out_x, 2.95), 'NLRP3', size=11, weight='bold', color='white')
    _label(ax, (out_x, 2.70), 'candidates', size=11, weight='bold',
            color='white')
    _label(ax, (out_x, 2.30), '─────', size=9, color='white')
    _label(ax, (out_x, 2.05), 'Docking', size=8, color='white')
    _label(ax, (out_x, 1.85), '100 ns MD', size=8, color='white')
    _label(ax, (out_x, 1.65), 'MMPBSA', size=8, color='white')
    _label(ax, (out_x, 1.45), 'DILI-aware', size=8, color='white')
    _label(ax, (out_x, 1.25), 'ADMET', size=8, color='white')
    _label(ax, (out_x, 0.90), 'in silico', size=7.5, style='italic',
            color='white')

    _arrow(ax, (head_x + head_w / 2, 2.05), (out_x - out_w / 2, 2.05),
            lw=1.8, color=C_OUT)

    # ─────────────────────────────── Bottom banner ────────────────────
    banner_y = 0.28
    ax.add_patch(mpatches.Rectangle((0.15, banner_y - 0.16), 9.7, 0.30,
                                     facecolor='#F2F2F2', edgecolor='none'))
    _label(ax, (5.0, banner_y),
            'STG-Mol  •  leakage-free NLRP3 dataset (2,521 molecules)  •  '
            'primary Test AUC 0.9167 (scaffold split)  •  '
            'BEDROC@α=20 = 0.9028  •  AD-aware external evaluation',
            size=8.5, weight='bold', color='#222222')

    # Save
    out_dir = os.path.dirname(os.path.abspath(__file__))
    svg_path = os.path.join(out_dir, 'graphical_abstract.svg')
    png_path = os.path.join(out_dir, 'graphical_abstract.png')
    fig.savefig(svg_path, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(png_path, dpi=DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'✓ SVG → {svg_path}')
    print(f'✓ PNG → {png_path}  ({DPI} dpi)')


if __name__ == '__main__':
    draw()
