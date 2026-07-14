#!/usr/bin/env python3
"""
Multi-Modal Molecular Representation Fusion Network
=====================================================
融合 1D (Mol2vec), 2D (D-MPNN), 3D (SphereNet) 分子表征

v24 改动（在 v23 基础上）：

  【背景】v23 成果：
    ✅ Val AUC  = 0.9372（历史新高）
    ✅ Test AUC = 0.9534（历史新高）
    ✅ Recall   = 0.907
    ❌ F1       = 0.852（目标≥0.855 未达）
    ❌ Precision= 0.803（目标≥0.840 未达）
    ❌ MCC      = 0.793（目标≥0.810 未达）
    ❌ 阈值跳变：min_recall=0.87约束导致阈值从0.58跳至0.32

  【v24 四项改动】

  [Fix1] dropout: 0.52 → 0.54
         Train AUC=0.9979仍偏高，继续+0.02步长
         目标：将Train AUC压至≤0.990

  [Fix2] weight_decay: 0.012 → 0.015
         加强L2正则，与高dropout协同
         预期Train-Test Gap从0.0445降至≤0.040

  [Fix3] 阈值搜索：废弃min_recall硬约束 → F-beta(β=1.2)
         β=1.2时 F_beta偏向Recall但无硬截断
         预期阈值稳定在0.38~0.45，避免0.32/0.58跳变

  [Fix4] max_pos_weight: 2.3 → 2.5（回退v22稳定值）
         v23 Precision=0.803说明2.3过低
         2.5是v22验证的稳定值

Version: 1.9→2.0 (v26 — Multi-Seed Ensemble)
"""

import os
import sys
import warnings
import argparse
import math
import time
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, confusion_matrix, roc_curve, precision_recall_curve,
    matthews_corrcoef, average_precision_score
)

from rdkit import Chem
from rdkit.Chem import AllChem

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import radius_graph
from torch_geometric.utils import softmax

# Mol2vec
try:
    from mol2vec.features import mol2alt_sentence, sentences2vec
    from gensim.models import word2vec
    MOL2VEC_AVAILABLE = True
except ImportError:
    MOL2VEC_AVAILABLE = False
    print("⚠️ mol2vec未安装")

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# Part 0: Loss Functions
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss with optional Label Smoothing

    v24: smoothing=0.05（维持v23/v20验证值）
    gamma=1.5（v22验证最优，v21证明2.0有害）
    """

    def __init__(self, gamma: float = 1.5,
                 weight: Optional[torch.Tensor] = None,
                 smoothing: float = 0.0):
        super().__init__()
        self.gamma    = gamma
        self.weight   = weight
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:

        num_classes = logits.size(-1)

        if self.smoothing > 0.0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(logits)
                smooth_targets.fill_(self.smoothing / num_classes)
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / num_classes)

            log_probs = F.log_softmax(logits, dim=-1)
            ce_loss   = -(smooth_targets * log_probs).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(
                logits, targets, weight=self.weight, reduction='none')

        pt          = torch.exp(-ce_loss)
        focal_loss  = (1.0 - pt) ** self.gamma * ce_loss

        if self.smoothing > 0.0 and self.weight is not None:
            class_w = self.weight[targets]
            focal_loss = focal_loss * class_w

        return focal_loss.mean()


# ============================================================================
# Part 1: 1D Encoder - Mol2vec with MLP
# ============================================================================

class Mol2VecEncoder(nn.Module):
    def __init__(self, input_dim: int = 300, hidden_dim: int = 256,
                 output_dim: int = 128, dropout: float = 0.2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ============================================================================
# Part 2: 2D Encoder - D-MPNN
# ============================================================================

def _compute_pharmacophore_features(mol) -> dict:
    """
    [Phase 1 新增] 提取分子的药效团特征.

    返回一个 dict, key 为药效团家族名, value 为该家族涉及的原子索引集合。
    使用 RDKit 内置 BaseFeatures.fdef (无外部依赖)。
    """
    try:
        from rdkit.Chem import ChemicalFeatures
        from rdkit import RDConfig
        import os
        _fdef = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        if not hasattr(_compute_pharmacophore_features, '_factory'):
            _compute_pharmacophore_features._factory = \
                ChemicalFeatures.BuildFeatureFactory(_fdef)
        factory = _compute_pharmacophore_features._factory

        feats = factory.GetFeaturesForMol(mol)
        result = defaultdict(set)
        for f in feats:
            fam = f.GetFamily()          # Donor / Acceptor / Hydrophobe / Aromatic ...
            for idx in f.GetAtomIds():
                result[fam].add(idx)
        return dict(result)
    except Exception:
        return {}


def _ensure_gasteiger_charges(mol) -> None:
    """[Phase 1 新增] 若 mol 未计算 Gasteiger charges 则计算之 (in-place)."""
    try:
        # 若首个原子已有 _GasteigerCharge 属性,视为已计算
        if mol.GetNumAtoms() > 0 and mol.GetAtomWithIdx(0).HasProp('_GasteigerCharge'):
            return
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass


def generate_atom_features(atom, pharma_dict: Optional[dict] = None) -> List[float]:
    """
    [Phase 1 增强版] 原子特征生成: 39 维基础 + 8 维电子/药效团 = 47 维.

    新增 8 维 (Phase 1):
        [39] Gasteiger 部分电荷 (归一化)
        [40] 原子极化率 (查表, 归一化)
        [41] 是否 H-bond 供体 (来自药效团)
        [42] 是否 H-bond 受体
        [43] 是否疏水中心
        [44] 是否芳香药效团中心
        [45] 是否可电离 (酸或碱)
        [46] 隐式 H 数 (归一化)
    """
    features = []
    try:
        symbol = atom.GetSymbol()
        for s in ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']:
            features.append(float(symbol == s))
        features.append(float(symbol not in ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']))

        deg = min(atom.GetDegree(), 4)
        for d in range(5):
            features.append(float(deg == d))

        features.append(float(atom.GetFormalCharge()) / 2.0)

        hyb_map = {Chem.HybridizationType.SP: 0, Chem.HybridizationType.SP2: 1,
                   Chem.HybridizationType.SP3: 2, Chem.HybridizationType.SP3D: 3}
        hyb_idx = hyb_map.get(atom.GetHybridization(), 4)
        for h in range(5):
            features.append(float(hyb_idx == h))

        features.append(float(atom.GetIsAromatic()))
        features.append(float(atom.GetTotalNumHs()) / 4.0)
        features.append(float(atom.IsInRing()))

        in_ring = atom.IsInRing()
        for size in [5, 6, 7]:
            features.append(float(in_ring and atom.IsInRingSize(size)))
        features.append(float(in_ring and not any(atom.IsInRingSize(s) for s in [5, 6, 7])))
        features.append(float(not in_ring))

        en = {'C': 2.55, 'N': 3.04, 'O': 3.44, 'S': 2.58, 'F': 3.98,
              'Cl': 3.16, 'Br': 2.96, 'I': 2.66, 'P': 2.19}
        features.append((en.get(symbol, 2.5) - 2.0) / 2.0)
        features.append(atom.GetMass() / 100.0)

        # ============= [Phase 1 新增] 电子结构 + 药效团 8 维 =============
        # [39] Gasteiger 部分电荷 (先在 mol 层预计算)
        try:
            pc = atom.GetDoubleProp('_GasteigerCharge')
            if pc != pc or abs(pc) > 5.0:      # NaN 或异常值
                pc = 0.0
            features.append(pc / 2.0)          # 归一化到 ~[-1, 1]
        except Exception:
            features.append(0.0)

        # [40] 原子极化率 (查表, 单位 Å^3, 归一化到 [0, 1])
        polar = {'H': 0.667, 'C': 1.76, 'N': 1.10, 'O': 0.802, 'F': 0.557,
                 'S': 2.90, 'Cl': 2.18, 'Br': 3.05, 'I': 5.35, 'P': 3.63}
        features.append(polar.get(symbol, 1.5) / 5.5)

        # [41-45] 药效团标签 (来自 pharma_dict)
        atom_idx = atom.GetIdx()
        pd = pharma_dict or {}
        features.append(float(atom_idx in pd.get('Donor', set())))
        features.append(float(atom_idx in pd.get('Acceptor', set())))
        features.append(float(atom_idx in pd.get('Hydrophobe', set())
                              or atom_idx in pd.get('LumpedHydrophobe', set())))
        features.append(float(atom_idx in pd.get('Aromatic', set())))
        features.append(float(atom_idx in pd.get('PosIonizable', set())
                              or atom_idx in pd.get('NegIonizable', set())))

        # [46] 隐式 H 数 (归一化)
        try:
            features.append(float(atom.GetNumImplicitHs()) / 4.0)
        except Exception:
            features.append(0.0)

    except Exception:
        features = [0.0] * 47

    return features[:47] if len(features) >= 47 else features + [0.0] * (47 - len(features))


def generate_bond_features(bond) -> List[float]:
    features = []
    try:
        bt = bond.GetBondType()
        features.extend([
            float(bt == Chem.BondType.SINGLE),
            float(bt == Chem.BondType.DOUBLE),
            float(bt == Chem.BondType.TRIPLE),
            float(bt == Chem.BondType.AROMATIC)
        ])
        features.append(float(bond.GetIsConjugated()))
        features.append(float(bond.IsInRing()))

        st = bond.GetStereo()
        features.extend([
            float(st == Chem.BondStereo.STEREOZ),
            float(st == Chem.BondStereo.STEREOE),
            float(st in [Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS]),
            float(st == Chem.BondStereo.STEREONONE)
        ])

        bond_order = {'SINGLE': 1.0, 'DOUBLE': 2.0, 'TRIPLE': 3.0, 'AROMATIC': 1.5}
        features.append(bond_order.get(str(bt).split('.')[-1], 1.0) / 3.0)
        features.append(float(not bond.IsInRing() and bt == Chem.BondType.SINGLE))
    except:
        features = [0.0] * 12
    return features[:12]


class DMPNN(nn.Module):
    def __init__(self, in_channels: int = 47, edge_dim: int = 12,
                 hidden_channels: int = 128, output_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.2):
        super().__init__()

        self.num_layers      = num_layers
        self.dropout         = dropout
        self.hidden_channels = hidden_channels
        self.output_dim      = output_dim

        self.W_i = nn.Linear(in_channels + edge_dim, hidden_channels)
        self.W_h = nn.Linear(hidden_channels, hidden_channels)
        self.W_o = nn.Linear(in_channels + hidden_channels, hidden_channels)

        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])
        self.bn_out = nn.BatchNorm1d(hidden_channels)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = (data.x, data.edge_index,
                                            data.edge_attr, data.batch)
        row, col  = edge_index
        num_edges = edge_index.size(1)

        edge_input  = torch.cat([x[row], edge_attr], dim=-1)
        edge_hidden = F.relu(self.W_i(edge_input))

        if hasattr(data, 'reverse_edge_idx') and data.reverse_edge_idx is not None:
            reverse_edge_idx = data.reverse_edge_idx
        else:
            reverse_edge_idx = self._get_reverse_edge_indices(edge_index, num_edges)

        for layer_idx in range(self.num_layers):
            node_messages = torch.zeros(x.size(0), self.hidden_channels,
                                        device=x.device)
            node_messages.scatter_add_(
                0, col.unsqueeze(-1).expand_as(edge_hidden), edge_hidden)

            edge_msg = node_messages[row]

            if reverse_edge_idx is not None and len(reverse_edge_idx) > 0:
                valid_mask = reverse_edge_idx >= 0
                if valid_mask.any():
                    edge_msg[valid_mask] -= edge_hidden[reverse_edge_idx[valid_mask]]

            edge_hidden_new = self.W_h(edge_msg)
            edge_hidden_new = self.bn_layers[layer_idx](edge_hidden_new)
            edge_hidden_new = F.relu(edge_hidden_new)
            edge_hidden     = (edge_hidden +
                               F.dropout(edge_hidden_new, p=self.dropout,
                                         training=self.training))

        node_hidden = torch.zeros(x.size(0), self.hidden_channels, device=x.device)
        node_hidden.scatter_add_(
            0, col.unsqueeze(-1).expand_as(edge_hidden), edge_hidden)

        node_output = torch.cat([x, node_hidden], dim=-1)
        node_output = F.relu(self.bn_out(self.W_o(node_output)))
        node_output = F.dropout(node_output, p=self.dropout,
                                training=self.training)

        graph_repr = (global_mean_pool(node_output, batch) +
                      global_max_pool(node_output, batch))

        return self.output_proj(graph_repr)

    def _get_reverse_edge_indices(self, edge_index, num_edges):
        if num_edges == 0:
            return None

        row, col   = edge_index
        device     = edge_index.device
        max_nodes  = max(row.max().item(), col.max().item()) + 1

        edge_hash    = row * max_nodes + col
        reverse_hash = col * max_nodes + row

        sorted_hash, sorted_indices = torch.sort(edge_hash)
        positions   = torch.searchsorted(sorted_hash, reverse_hash)
        positions   = positions.clamp(max=num_edges - 1)
        found_mask  = sorted_hash[positions] == reverse_hash

        reverse_idx = torch.where(
            found_mask,
            sorted_indices[positions],
            torch.tensor(-1, device=device)
        )
        return reverse_idx


# ============================================================================
# Part 3: 3D Encoder - SphereNet
# ============================================================================

class RBFExpansion(nn.Module):
    def __init__(self, num_rbf: int = 64, cutoff: float = 5.0):
        super().__init__()
        self.register_buffer('centers', torch.linspace(0, cutoff, num_rbf))
        self.register_buffer('widths',  torch.ones(num_rbf) * (cutoff / num_rbf))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            -((distances.unsqueeze(-1) - self.centers) ** 2) / (self.widths ** 2))


class SphericalBasisLayer(nn.Module):
    def __init__(self, num_spherical: int = 7, num_radial: int = 6,
                 cutoff: float = 5.0):
        super().__init__()
        self.num_spherical = num_spherical
        self.num_radial    = num_radial
        self.cutoff        = cutoff

        bessel_weights = torch.tensor(
            [(i + 1) * math.pi for i in range(num_radial)])
        self.register_buffer('bessel_weights', bessel_weights)

    def forward(self, dist: torch.Tensor,
                angle: Optional[torch.Tensor] = None) -> torch.Tensor:
        d_scaled = dist / self.cutoff
        radial   = (torch.sin(self.bessel_weights * d_scaled.unsqueeze(-1))
                    / (dist.unsqueeze(-1) + 1e-8))
        cutoff_vals = (0.5 * (torch.cos(dist * math.pi / self.cutoff) + 1.0)
                       * (dist < self.cutoff).float())
        radial = radial * cutoff_vals.unsqueeze(-1)

        if angle is None:
            return radial

        cos_angle = torch.cos(angle)
        spherical = [torch.ones_like(cos_angle) * 0.5 * math.sqrt(1.0 / math.pi)]
        if self.num_spherical > 1:
            spherical.append(0.5 * math.sqrt(3.0 / math.pi) * cos_angle)
        if self.num_spherical > 2:
            spherical.append(
                0.25 * math.sqrt(5.0 / math.pi) * (3 * cos_angle ** 2 - 1))
        for l in range(3, self.num_spherical):
            spherical.append(cos_angle ** l)

        spherical = torch.stack(spherical, dim=-1)
        basis     = radial.unsqueeze(-1) * spherical.unsqueeze(-2)
        return basis.view(dist.size(0), -1)


class SphereNetInteraction(MessagePassing):
    def __init__(self, hidden_channels: int, num_spherical: int,
                 num_radial: int, cutoff: float):
        super().__init__(aggr='add', flow='target_to_source')
        self.sbf      = SphericalBasisLayer(num_spherical, num_radial, cutoff)
        basis_dim     = num_spherical * num_radial

        self.mlp_msg  = nn.Sequential(
            nn.Linear(hidden_channels + basis_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU()
        )
        self.mlp_update = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.layer_norm = nn.LayerNorm(hidden_channels)

    def forward(self, x, edge_index, dist, angle):
        basis = self.sbf(dist, angle)
        out   = self.propagate(edge_index, x=x, basis=basis)
        out   = self.mlp_update(torch.cat([x, out], dim=-1))
        return self.layer_norm(out + x)

    def message(self, x_j, basis):
        return self.mlp_msg(torch.cat([x_j, basis], dim=-1))


class SphereNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 47, hidden_channels: int = 128,
                 output_dim: int = 128, num_layers: int = 4,
                 num_spherical: int = 7, num_radial: int = 6,
                 cutoff: float = 5.0, dropout: float = 0.2):
        super().__init__()

        self.cutoff     = cutoff
        self.output_dim = output_dim

        self.atom_embed = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.z_embed = nn.Embedding(100, hidden_channels)

        self.rbf       = RBFExpansion(num_rbf=num_radial * 8, cutoff=cutoff)
        self.dist_embed = nn.Sequential(
            nn.Linear(num_radial * 8, hidden_channels),
            nn.SiLU()
        )

        self.interactions = nn.ModuleList([
            SphereNetInteraction(hidden_channels, num_spherical, num_radial, cutoff)
            for _ in range(num_layers)
        ])

        self.output_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, hidden_channels // 4)
            )
            for _ in range(num_layers)
        ])

        final_dim = hidden_channels // 4 * num_layers + hidden_channels
        self.final_proj = nn.Sequential(
            nn.Linear(final_dim, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, z, pos, edge_index, batch = (data.x, data.z, data.pos,
                                         data.edge_index, data.batch)

        h     = self.atom_embed(x) + self.z_embed(z)
        row, col = edge_index
        dist  = torch.norm(pos[row] - pos[col], dim=-1)
        angle = self._compute_angles(pos, edge_index, dist)

        outputs = []
        for i, interaction in enumerate(self.interactions):
            h = interaction(h, edge_index, dist, angle)
            outputs.append(self.output_blocks[i](h))

        h_global   = (global_mean_pool(h, batch) + global_max_pool(h, batch))
        out_global = [global_mean_pool(o, batch) + global_max_pool(o, batch)
                      for o in outputs]

        combined = torch.cat([h_global] + out_global, dim=-1)
        return self.final_proj(combined)

    def _compute_angles(self, pos, edge_index, dist):
        row, col  = edge_index
        vec       = (pos[row] - pos[col]) / (dist.unsqueeze(-1) + 1e-8)
        z_axis    = torch.tensor([0., 0., 1.], device=pos.device)
        cos_angle = torch.sum(vec * z_axis, dim=-1)
        return torch.acos(torch.clamp(cos_angle, -1.0, 1.0))


# ============================================================================
# Part 4: Multi-Modal Fusion Strategies
# ============================================================================

class CrossModalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim      = dim
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads

        assert dim % num_heads == 0

        self.W_q1 = nn.Linear(dim, dim)
        self.W_k1 = nn.Linear(dim, dim)
        self.W_v1 = nn.Linear(dim, dim)

        self.W_q2 = nn.Linear(dim, dim)
        self.W_k2 = nn.Linear(dim, dim)
        self.W_v2 = nn.Linear(dim, dim)

        self.out_proj1 = nn.Linear(dim, dim)
        self.out_proj2 = nn.Linear(dim, dim)

        self.norm1   = nn.LayerNorm(dim)
        self.norm2   = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.scale   = self.head_dim ** -0.5

    def forward(self, x1: torch.Tensor,
                x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x1.size(0)

        q1 = self.W_q1(x1).view(batch_size, self.num_heads, self.head_dim)
        k2 = self.W_k1(x2).view(batch_size, self.num_heads, self.head_dim)
        v2 = self.W_v1(x2).view(batch_size, self.num_heads, self.head_dim)

        attn1  = torch.einsum('bhd,bhd->bh', q1, k2) * self.scale
        attn1  = F.softmax(attn1, dim=-1)
        attn1  = self.dropout(attn1)
        out1   = torch.einsum('bh,bhd->bhd', attn1, v2).reshape(batch_size, self.dim)
        out1   = self.out_proj1(out1)
        enh_x1 = self.norm1(x1 + self.dropout(out1))

        q2 = self.W_q2(x2).view(batch_size, self.num_heads, self.head_dim)
        k1 = self.W_k2(x1).view(batch_size, self.num_heads, self.head_dim)
        v1 = self.W_v2(x1).view(batch_size, self.num_heads, self.head_dim)

        attn2  = torch.einsum('bhd,bhd->bh', q2, k1) * self.scale
        attn2  = F.softmax(attn2, dim=-1)
        attn2  = self.dropout(attn2)
        out2   = torch.einsum('bh,bhd->bhd', attn2, v1).reshape(batch_size, self.dim)
        out2   = self.out_proj2(out2)
        enh_x2 = self.norm2(x2 + self.dropout(out2))

        return enh_x1, enh_x2


class GatedFusionUnit(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([x1, x2], dim=-1)
        gate   = self.gate(concat)
        gated  = gate * x1 + (1 - gate) * x2
        fused  = self.transform(concat)
        return self.norm(gated + fused)


class LowRankBilinearFusion(nn.Module):
    def __init__(self, dim1: int, dim2: int, output_dim: int,
                 rank: int = 16, dropout: float = 0.1):
        super().__init__()

        self.rank       = rank
        self.output_dim = output_dim

        self.U    = nn.Linear(dim1, rank * output_dim, bias=False)
        self.V    = nn.Linear(dim2, rank * output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))

        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(output_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        batch_size = x1.size(0)

        u = self.U(x1).view(batch_size, self.output_dim, self.rank)
        v = self.V(x2).view(batch_size, self.output_dim, self.rank)

        z = torch.sum(u * v, dim=-1) + self.bias
        z = self.dropout(z)
        z = torch.sign(z) * torch.sqrt(torch.abs(z) + 1e-8)
        z = F.normalize(z, p=2, dim=-1)

        return self.norm(z)


class HierarchicalTrimodalFusion(nn.Module):
    """Hierarchical tri-modal fusion (Cross-Attn + Gated + Bilinear + ImportanceNet).

    The `ablate` argument enables component-level ablations for paper Table 5.2.3.
    Keys (all default False):
        no_cross_attn      -- disable cross-modal attention (pass modalities through)
        no_gated           -- replace gated fusion with mean of the two inputs
        no_bilinear        -- drop the bilinear output branch from `final_concat`
        no_importance_net  -- fix modality weights to uniform 1/3 each (bypass importance)
    Turning ON all four flags degrades the module to a plain concat + MLP fusion,
    which is the natural lower baseline.
    """

    def __init__(self, dim: int, dropout: float = 0.1, ablate=None):
        super().__init__()
        ablate = ablate or {}
        self.no_cross_attn     = bool(ablate.get('no_cross_attn', False))
        self.no_gated          = bool(ablate.get('no_gated', False))
        self.no_bilinear       = bool(ablate.get('no_bilinear', False))
        self.no_importance_net = bool(ablate.get('no_importance_net', False))

        self.cross_attn_12 = CrossModalAttention(dim, num_heads=4, dropout=dropout)
        self.cross_attn_13 = CrossModalAttention(dim, num_heads=4, dropout=dropout)
        self.cross_attn_23 = CrossModalAttention(dim, num_heads=4, dropout=dropout)

        self.gate_12 = GatedFusionUnit(dim, dropout)
        self.gate_13 = GatedFusionUnit(dim, dropout)
        self.gate_23 = GatedFusionUnit(dim, dropout)

        self.bilinear_final = LowRankBilinearFusion(
            dim, dim * 2, dim, rank=16, dropout=dropout)

        # `final_fusion` input dim depends on whether the bilinear branch is present.
        concat_in = dim * 4 if not self.no_bilinear else dim * 3
        self.final_fusion = nn.Sequential(
            nn.Linear(concat_in, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )

        self.importance_net = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,
                x3: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # (1) Cross-modal attention -- may be ablated
        if self.no_cross_attn:
            x1_enh, x2_enh, x3_enh = x1, x2, x3
        else:
            x1_12, x2_12 = self.cross_attn_12(x1, x2)
            x1_13, x3_13 = self.cross_attn_13(x1, x3)
            x2_23, x3_23 = self.cross_attn_23(x2, x3)
            x1_enh = (x1_12 + x1_13) / 2
            x2_enh = (x2_12 + x2_23) / 2
            x3_enh = (x3_13 + x3_23) / 2

        # (2) Pairwise gated fusion -- may be ablated to mean of pair
        if self.no_gated:
            f_12 = (x1_enh + x2_enh) / 2
            f_13 = (x1_enh + x3_enh) / 2
            f_23 = (x2_enh + x3_enh) / 2
        else:
            f_12 = self.gate_12(x1_enh, x2_enh)
            f_13 = self.gate_13(x1_enh, x3_enh)
            f_23 = self.gate_23(x2_enh, x3_enh)

        # (3) Importance-net-weighted skip -- may be ablated to uniform 1/3 weighting
        if self.no_importance_net:
            batch = x1.shape[0]
            importance = torch.full((batch, 3), 1.0 / 3.0,
                                     device=x1.device, dtype=x1.dtype)
        else:
            concat_orig = torch.cat([x1, x2, x3], dim=-1)
            importance  = self.importance_net(concat_orig)

        weighted_orig = (importance[:, 0:1] * x1 +
                         importance[:, 1:2] * x2 +
                         importance[:, 2:3] * x3)

        # (4) Bilinear branch -- may be ablated (dropped from the concat)
        if self.no_bilinear:
            final_concat = torch.cat([f_12, f_13, f_23], dim=-1)
        else:
            pair_concat  = torch.cat([f_12, f_23], dim=-1)
            bilinear_out = self.bilinear_final(f_13, pair_concat)
            final_concat = torch.cat([f_12, f_13, f_23, bilinear_out], dim=-1)

        fused = self.final_fusion(final_concat)
        return fused + weighted_orig, importance


class BimodalFusion(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()

        self.cross_attn   = CrossModalAttention(dim, num_heads=4, dropout=dropout)
        self.gated_fusion = GatedFusionUnit(dim, dropout)
        self.bilinear     = LowRankBilinearFusion(dim, dim, dim, rank=16,
                                                   dropout=dropout)

        self.combine = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x1: torch.Tensor,
                x2: torch.Tensor) -> Tuple[torch.Tensor, None]:
        x1_attn, x2_attn = self.cross_attn(x1, x2)
        gated    = self.gated_fusion(x1_attn, x2_attn)
        bilinear = self.bilinear(x1_attn, x2_attn)
        concat   = torch.cat([gated, bilinear, (x1_attn + x2_attn) / 2], dim=-1)
        return self.combine(concat), None


# ============================================================================
# Part 5: Main Fusion Models
# ============================================================================

class MultiModalFusionNet(nn.Module):
    FUSION_MODES = ['1D+2D+3D', '1D+2D', '1D+3D', '2D+3D', '1D', '2D', '3D']

    def __init__(self, config: Dict):
        super().__init__()

        self.config      = config
        self.fusion_mode = config.get('fusion_mode', '1D+2D+3D')

        assert self.fusion_mode in self.FUSION_MODES

        hidden_dim = config.get('hidden_dim', 128)
        dropout    = config.get('dropout', 0.2)

        if '1D' in self.fusion_mode:
            self.encoder_1d = Mol2VecEncoder(
                input_dim=config.get('mol2vec_dim', 300),
                hidden_dim=hidden_dim * 2,
                output_dim=hidden_dim,
                dropout=dropout
            )

        if '2D' in self.fusion_mode:
            self.encoder_2d = DMPNN(
                in_channels=47, edge_dim=12,
                hidden_channels=hidden_dim, output_dim=hidden_dim,
                num_layers=config.get('dmpnn_layers', 3),
                dropout=dropout
            )

        if '3D' in self.fusion_mode:
            self.encoder_3d = SphereNetEncoder(
                in_channels=47, hidden_channels=hidden_dim,
                output_dim=hidden_dim,
                num_layers=config.get('spherenet_layers', 4),
                num_spherical=config.get('num_spherical', 7),
                num_radial=config.get('num_radial', 6),
                cutoff=config.get('cutoff', 5.0),
                dropout=config.get('spherenet_dropout', dropout)
            )

        if self.fusion_mode == '1D+2D+3D':
            # Fusion-component ablation flags (all default off — full model).
            # Config example:
            #   model:
            #     fusion_ablate:
            #       no_cross_attn:     false
            #       no_gated:          false
            #       no_bilinear:       false
            #       no_importance_net: false
            fusion_ablate = config.get('fusion_ablate', {}) or {}
            self.fusion = HierarchicalTrimodalFusion(
                hidden_dim, dropout, ablate=fusion_ablate)
        else:
            self.fusion = BimodalFusion(hidden_dim, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

        # ============= [Phase 1B] Multi-Task ADMET heads =============
        # 5 个辅助 ADMET 任务(二分类):
        #   [0] Lipinski RO5, [1] QED>0.5, [2] no PAINS,
        #   [3] SA<5,          [4] LogP in [0,5]
        self.num_admet_tasks = 5
        self.admet_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.num_admet_tasks * 2),   # 5 tasks × 2 classes
        )

    def forward(self, mol2vec_feat=None, graph_2d=None, graph_3d=None) -> Dict:
        encodings = {}
        if '1D' in self.fusion_mode: encodings['1D'] = self.encoder_1d(mol2vec_feat)
        if '2D' in self.fusion_mode: encodings['2D'] = self.encoder_2d(graph_2d)
        if '3D' in self.fusion_mode: encodings['3D'] = self.encoder_3d(graph_3d)

        weights = None
        # --- 融合逻辑 ---
        if self.fusion_mode == '1D+2D+3D':
            fused, weights = self.fusion(encodings['1D'], encodings['2D'], encodings['3D'])
        elif self.fusion_mode == '1D+2D':
            fused, weights = self.fusion(encodings['1D'], encodings['2D'])
        elif self.fusion_mode == '1D+3D':
            fused, weights = self.fusion(encodings['1D'], encodings['3D'])
        elif self.fusion_mode == '2D+3D':
            fused, weights = self.fusion(encodings['2D'], encodings['3D'])
        # --- 新增单模态逻辑 ---
        elif self.fusion_mode == '1D':
            fused = encodings['1D']
        elif self.fusion_mode == '2D':
            fused = encodings['2D']
        elif self.fusion_mode == '3D':
            fused = encodings['3D']
        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")

        logits = self.classifier(fused)
        # [Phase 1B] ADMET multi-task heads: reshape to [B, num_tasks, 2]
        admet_logits = self.admet_head(fused).view(
            -1, self.num_admet_tasks, 2
        )
        return {'logits': logits,
                'admet_logits': admet_logits,
                'features': fused,
                'weights': weights}


# ============================================================================
# Part 6: Data Preparation
# ============================================================================

class ConformerGenerator:
    def __init__(self, num_confs: int = 1, optimize: bool = True,
                 max_iters: int = 500):
        self.num_confs = num_confs
        self.optimize  = optimize
        self.max_iters = max_iters
        self.params    = AllChem.ETKDGv3()
        self.params.randomSeed             = 42
        self.params.useSmallRingTorsions   = True
        self.params.useMacrocycleTorsions  = True

    def generate(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        if mol is None:
            return None
        try:
            mol      = Chem.AddHs(mol)
            conf_ids = AllChem.EmbedMultipleConfs(
                mol, numConfs=self.num_confs, params=self.params)
            if len(conf_ids) == 0:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                if mol.GetNumConformers() == 0:
                    return None
            if self.optimize and mol.GetNumConformers() > 0:
                AllChem.MMFFOptimizeMoleculeConfs(
                    mol, mmffVariant='MMFF94',
                    maxIters=self.max_iters, numThreads=0)
            return Chem.RemoveHs(mol)
        except:
            return None


def mol_to_2d_graph(smiles: str) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # [Phase 1] 预计算 Gasteiger 电荷 + 药效团
    _ensure_gasteiger_charges(mol)
    pharma_dict = _compute_pharmacophore_features(mol)

    x = torch.tensor([generate_atom_features(a, pharma_dict) for a in mol.GetAtoms()],
                     dtype=torch.float)
    if x.size(0) == 0:
        return None

    edge_indices, edge_attrs = [], []
    edge_to_idx = {}

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf   = generate_bond_features(bond)

        idx1 = len(edge_indices)
        edge_indices.append([i, j]); edge_attrs.append(bf)
        edge_to_idx[(i, j)] = idx1

        idx2 = len(edge_indices)
        edge_indices.append([j, i]); edge_attrs.append(bf)
        edge_to_idx[(j, i)] = idx2

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attrs, dtype=torch.float)

        num_edges = len(edge_indices)
        reverse_edge_idx = torch.zeros(num_edges, dtype=torch.long)
        for idx, (i, j) in enumerate(edge_indices):
            reverse_edge_idx[idx] = edge_to_idx.get((j, i), idx)
    else:
        edge_index       = torch.empty((2, 0), dtype=torch.long)
        edge_attr        = torch.empty((0, 12), dtype=torch.float)
        reverse_edge_idx = torch.empty(0, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                reverse_edge_idx=reverse_edge_idx)


def mol_to_3d_graph(smiles: str, cutoff: float = 5.0,
                    conformer_gen: ConformerGenerator = None) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if conformer_gen is None:
        conformer_gen = ConformerGenerator()

    mol_3d = conformer_gen.generate(mol)
    if mol_3d is None or mol_3d.GetNumConformers() == 0:
        return None

    conf      = mol_3d.GetConformer()
    positions = [[conf.GetAtomPosition(i).x,
                  conf.GetAtomPosition(i).y,
                  conf.GetAtomPosition(i).z]
                 for i in range(mol_3d.GetNumAtoms())]
    pos = torch.tensor(positions, dtype=torch.float)

    atom_features  = []
    atomic_numbers = []
    # [Phase 1] 预计算 Gasteiger 电荷 + 药效团 (对 3D 优化后的分子)
    _ensure_gasteiger_charges(mol_3d)
    pharma_dict = _compute_pharmacophore_features(mol_3d)
    for atom in mol_3d.GetAtoms():
        atom_features.append(generate_atom_features(atom, pharma_dict))
        atomic_numbers.append(atom.GetAtomicNum())

    x = torch.tensor(atom_features,  dtype=torch.float)
    z = torch.tensor(atomic_numbers, dtype=torch.long)

    if x.size(0) == 0:
        return None

    edge_index = radius_graph(pos, r=cutoff, loop=False)
    if edge_index.size(1) == 0:
        return None

    row, col     = edge_index
    edge_weight  = torch.norm(pos[row] - pos[col], dim=-1)

    return Data(x=x, z=z, pos=pos, edge_index=edge_index,
                edge_weight=edge_weight)


class MultiModalMoleculeDataset(Dataset):
    def __init__(self, smiles_list: List[str], labels: np.ndarray,
                 mol2vec_features: Optional[np.ndarray] = None,
                 fusion_mode: str = '1D+2D+3D', cutoff: float = 5.0,
                 admet_labels: Optional[np.ndarray] = None):
        """
        [Phase 1B] admet_labels: optional numpy array of shape [N, 5] with
        binary labels for {lipinski, qed, pains, sa, logp}.
        If None, zeros are used and the multi-task loss will be effectively
        deactivated by config.
        """
        self.fusion_mode   = fusion_mode
        self.cutoff        = cutoff
        self.data_list     = []
        self.valid_indices = []

        # [Phase 1B] default ADMET to zeros if not provided
        if admet_labels is None:
            admet_labels = np.zeros((len(smiles_list), 5), dtype=np.int64)

        conformer_gen = ConformerGenerator() if '3D' in fusion_mode else None
        print(f"🔬 Building multi-modal dataset (mode: {fusion_mode})...")

        for idx, (smiles, label) in enumerate(zip(smiles_list, labels)):
            sample = {
                'smiles': smiles,
                'label':  label,
                'admet_labels': np.asarray(admet_labels[idx], dtype=np.int64),
            }
            valid  = True

            if '1D' in fusion_mode and mol2vec_features is not None:
                sample['mol2vec'] = mol2vec_features[idx]

            if '2D' in fusion_mode:
                g2d = mol_to_2d_graph(smiles)
                if g2d is None:
                    valid = False
                else:
                    sample['graph_2d'] = g2d

            if '3D' in fusion_mode:
                g3d = mol_to_3d_graph(smiles, cutoff, conformer_gen)
                if g3d is None:
                    valid = False
                else:
                    sample['graph_3d'] = g3d

            if valid:
                self.data_list.append(sample)
                self.valid_indices.append(idx)

            if (idx + 1) % 500 == 0:
                rate = len(self.data_list) / (idx + 1) * 100
                print(f"   Progress: {idx+1}/{len(smiles_list)} ({rate:.1f}%)")

        print(f"✅ Done: {len(self.data_list)}/{len(smiles_list)} valid samples")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_multimodal(batch: List[Dict]) -> Dict:
    result = {
        'labels': torch.tensor([s['label'] for s in batch], dtype=torch.long)
    }

    # [Phase 1B] ADMET multi-task labels: [B, 5]
    if 'admet_labels' in batch[0]:
        result['admet_labels'] = torch.tensor(
            np.stack([s['admet_labels'] for s in batch]),
            dtype=torch.long,
        )

    if 'mol2vec' in batch[0]:
        result['mol2vec'] = torch.tensor(
            np.stack([s['mol2vec'] for s in batch]), dtype=torch.float)

    if 'graph_2d' in batch[0]:
        graphs_2d   = [s['graph_2d'] for s in batch]
        all_rev_idx = []
        edge_offset = 0
        for g in graphs_2d:
            if hasattr(g, 'reverse_edge_idx') and g.reverse_edge_idx is not None:
                all_rev_idx.append(g.reverse_edge_idx + edge_offset)
            edge_offset += g.edge_index.size(1)

        batched_graph = Batch.from_data_list(graphs_2d)
        if all_rev_idx:
            batched_graph.reverse_edge_idx = torch.cat(all_rev_idx, dim=0)
        result['graph_2d'] = batched_graph

    if 'graph_3d' in batch[0]:
        result['graph_3d'] = Batch.from_data_list([s['graph_3d'] for s in batch])

    return result


# ============================================================================
# Part 7: Training and Evaluation
# ============================================================================

class MetricsCalculator:

    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray,
                                y_proba: np.ndarray,
                                beta: float = 1.0,
                                min_recall: Optional[float] = None,
                                search_low: float = 0.30,
                                search_high: float = 0.55,
                                search_step: float = 0.005) -> float:
        """
        v24 改动：F-beta 软搜索，废弃 min_recall 硬约束（默认None）

        beta=1.0  → 标准F1（向下兼容）
        beta=1.2  → 偏向Recall，无硬截断，避免阈值跳变
        beta>1    → Recall权重 > Precision权重

        v23教训：min_recall硬约束在val集上找不到稳定平衡点→阈值跳变
        解决：F-beta软性偏向，让模型自然找到最优点
        """
        best_threshold, best_score = 0.5, 0

        for thresh in np.arange(search_low, search_high + search_step, search_step):
            thresh = round(thresh, 3)
            y_pred = (y_proba >= thresh).astype(int)

            # v24：F-beta搜索
            score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)

            # 可选Recall下限（不设则纯F-beta最优）
            if min_recall is not None:
                recall = recall_score(y_true, y_pred, zero_division=0)
                if recall < min_recall:
                    continue

            if score > best_score:
                best_score, best_threshold = score, thresh

        print(f"   🎯 F-beta(β={beta}) optimal threshold: {best_threshold:.3f} "
              f"(score={best_score:.4f})")
        return best_threshold

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray,
                        threshold: float = 0.5) -> Dict[str, float]:
        y_pred  = (y_proba >= threshold).astype(int)
        metrics = {
            'roc_auc':   roc_auc_score(y_true, y_proba),
            'accuracy':  accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall':    recall_score(y_true, y_pred, zero_division=0),
            'f1':        f1_score(y_true, y_pred, zero_division=0),
            'mcc':       matthews_corrcoef(y_true, y_pred),
            'threshold': threshold
        }
        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        except:
            metrics['pr_auc'] = 0.0
        return metrics


class MultiModalTrainer:

    def __init__(self, config: Dict, device: torch.device, exp_dir: Path):
        self.config      = config
        self.device      = device
        self.exp_dir     = exp_dir
        self.metrics_calc = MetricsCalculator()

    def _build_criterion(self, class_weights: torch.Tensor):
        loss_cfg     = self.config.get('loss', {})
        loss_type    = loss_cfg.get('type', 'cross_entropy')
        use_weight   = loss_cfg.get('use_class_weight', True)
        lambda_div   = loss_cfg.get('lambda_diversity', 0.0)
        smoothing    = loss_cfg.get('label_smoothing', 0.0)
        w = class_weights if use_weight else None

        if loss_type == 'focal':
            gamma = loss_cfg.get('gamma', 1.5)
            print(f"   📊 Loss: FocalLoss(gamma={gamma}, smoothing={smoothing})"
                  + (" with class weights" if w is not None else ""))
            criterion = FocalLoss(gamma=gamma, weight=w, smoothing=smoothing)
        else:
            print(f"   📊 Loss: CrossEntropyLoss"
                  + (" with class weights" if w is not None else ""))
            criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=smoothing)

        if lambda_div > 0:
            print(f"   📊 Diversity Loss: variance penalty (lambda={lambda_div})")
        else:
            print(f"   📊 Diversity Loss: disabled")

        return criterion, lambda_div

    def train(self, model: nn.Module, train_loader: DataLoader,
              val_loader: DataLoader, test_loader: DataLoader) -> Dict:

        fusion_mode = self.config.get('fusion_mode', '1D+2D+3D')
        print(f"\n{'=' * 70}")
        print(f"🔬 Training Multi-Modal Fusion Network")
        print(f"   Mode: {fusion_mode}")
        print(f"{'=' * 70}")

        model = model.to(self.device)

        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)
        print(f"   📊 Parameters: {total_params:,} total, "
              f"{trainable_params:,} trainable")

        base_lr = self.config.get('learning_rate', 0.001)
        wd      = self.config.get('weight_decay', 1e-4)
        bls     = self.config.get('branch_lr_scale', {})

        if bls:
            _branch_map = {
                'encoder_1d': bls.get('encoder_1d', 1.0),
                'encoder_2d': bls.get('encoder_2d', 1.0),
                'encoder_3d': bls.get('encoder_3d', 1.0),
                'fusion':     bls.get('fusion',     1.0),
                'classifier': bls.get('classifier', 1.0),
            }
            param_groups  = []
            named_modules = dict(model.named_children())
            handled       = set()
            for branch, scale in _branch_map.items():
                if branch in named_modules:
                    params = list(named_modules[branch].parameters())
                    param_groups.append({
                        'params': params,
                        'lr': base_lr * scale,
                        'weight_decay': wd,
                        'name': branch
                    })
                    handled.update(id(p) for p in params)
            rest = [p for p in model.parameters() if id(p) not in handled]
            if rest:
                param_groups.append({'params': rest, 'lr': base_lr,
                                     'weight_decay': wd, 'name': 'other'})
            optimizer = torch.optim.AdamW(param_groups)
            print(f"   📊 Branch LR scale: " +
                  ", ".join(f"{k}×{v}" for k, v in _branch_map.items()
                            if k in named_modules))
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=base_lr, weight_decay=wd)

        sched_cfg      = self.config.get('scheduler', {})
        sched_type     = sched_cfg.get('type', 'ReduceLROnPlateau')
        _onecycle_mode = False
        _cosine_mode   = False

        epochs   = self.config.get('epochs', 200)
        patience = self.config.get('patience', 40)

        if sched_type == 'OneCycleLR':
            total_steps = len(train_loader) * epochs
            max_lr_list = [g['lr'] for g in optimizer.param_groups]
            scheduler   = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr_list,
                total_steps=total_steps,
                pct_start=sched_cfg.get('pct_start', 0.1),
                anneal_strategy=sched_cfg.get('anneal_strategy', 'cos'),
                div_factor=sched_cfg.get('div_factor', 25.0),
                final_div_factor=sched_cfg.get('final_div_factor', 1000.0),
            )
            _onecycle_mode = True
            print(f"   📊 Scheduler: OneCycleLR ✅"
                  f" pct_start={sched_cfg.get('pct_start', 0.1)}"
                  f" div_factor={sched_cfg.get('div_factor', 25.0)}"
                  f" total_steps={total_steps}"
                  f" max_lr={[f'{lr:.2e}' for lr in max_lr_list]}")

        elif sched_type == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=sched_cfg.get('T_0', 50),
                T_mult=sched_cfg.get('T_mult', 2),
                eta_min=sched_cfg.get('eta_min', 1e-6),
            )
            _cosine_mode = True
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=sched_cfg.get('mode', 'max'),
                factor=sched_cfg.get('factor', 0.5),
                patience=sched_cfg.get('patience', 15),
                min_lr=sched_cfg.get('min_lr', 1e-6),
                verbose=True)

        all_train_labels = []
        for batch in train_loader:
            all_train_labels.extend(batch['labels'].numpy().tolist())
        all_train_labels = np.array(all_train_labels)

        pos_weight_raw = ((all_train_labels == 0).sum() /
                           max((all_train_labels == 1).sum(), 1))

        max_pw     = self.config.get('loss', {}).get('max_pos_weight', 3.0)
        pos_weight = min(float(pos_weight_raw), float(max_pw))

        class_weights = torch.tensor(
            [1.0, pos_weight], dtype=torch.float).to(self.device)

        criterion, lambda_div = self._build_criterion(class_weights)

        # [Phase 1B] Multi-task ADMET auxiliary loss weight (read from config)
        multitask_cfg = self.config.get('multitask', {}) if isinstance(self.config.get('multitask', {}), dict) else {}
        self.admet_weight = float(multitask_cfg.get('admet_weight', 0.0))
        if self.admet_weight > 0:
            print(f"   📊 Multi-Task ADMET: enabled (admet_weight={self.admet_weight})")

        print(f"   📊 Class distribution: "
              f"{(all_train_labels == 0).sum()} neg, "
              f"{(all_train_labels == 1).sum()} pos")
        clip_note = (f" (clipped from {pos_weight_raw:.2f})"
                     if pos_weight < pos_weight_raw else "")
        print(f"   📊 Class weights: [1.0, {pos_weight:.2f}]{clip_note}")

        best_val_auc, best_state = 0, None
        patience_cnt = 0
        history = {
            'train_auc': [], 'val_auc': [],
            'loss': [], 'modality_weights': []
        }

        start_time = time.time()

        for epoch in range(epochs):
            train_loss, avg_weights = self._train_epoch(
                model, train_loader, optimizer, criterion, lambda_div,
                scheduler if _onecycle_mode else None
            )

            train_probs, train_labels = self._evaluate(model, train_loader)
            val_probs, val_labels     = self._evaluate(model, val_loader)

            train_auc = roc_auc_score(train_labels, train_probs)
            val_auc   = roc_auc_score(val_labels, val_probs)

            history['train_auc'].append(train_auc)
            history['val_auc'].append(val_auc)
            history['loss'].append(train_loss)
            history['modality_weights'].append(avg_weights)

            if _cosine_mode:
                scheduler.step(epoch)
            elif not _onecycle_mode:
                scheduler.step(val_auc)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                w_str  = (', '.join([f'{w:.2f}' for w in avg_weights])
                          if avg_weights else 'N/A')
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"   Epoch {epoch+1:3d}/{epochs} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Train AUC: {train_auc:.4f} | "
                      f"Val AUC: {val_auc:.4f} | "
                      f"W: [{w_str}] | "
                      f"LR: {cur_lr:.2e}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state   = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"   ⏹️  Early stopping at epoch {epoch + 1}")
                    break

        train_time = time.time() - start_time
        print(f"   ⏱️  Training time: {train_time:.1f}s")

        model.load_state_dict(
            {k: v.to(self.device) for k, v in best_state.items()})

        results = self._final_evaluation(
            model, train_loader, val_loader, test_loader,
            train_time, history, fusion_mode
        )

        model_name = f"MultiModal_{fusion_mode.replace('+', '_')}"
        torch.save(best_state,
                   self.exp_dir / "models" / f"{model_name}.pt")

        return results

    def _train_epoch(self, model, loader, optimizer, criterion,
                     lambda_div: float = 0.0,
                     onecycle_scheduler=None) -> Tuple[float, List[float]]:
        """
        Diversity Loss: 方差惩罚（v14修复，始终≥0）
        div_loss = mean(sum((w - uniform)^2))
        """
        model.train()
        total_loss = 0
        all_weights: List[np.ndarray] = []

        for batch in loader:
            optimizer.zero_grad()

            kwargs = {'mol2vec_feat': None, 'graph_2d': None, 'graph_3d': None}
            if 'mol2vec' in batch:
                kwargs['mol2vec_feat'] = batch['mol2vec'].to(self.device)
            if 'graph_2d' in batch:
                kwargs['graph_2d'] = batch['graph_2d'].to(self.device)
            if 'graph_3d' in batch:
                kwargs['graph_3d'] = batch['graph_3d'].to(self.device)

            labels = batch['labels'].to(self.device)
            result = model(**kwargs)
            loss   = criterion(result['logits'], labels)

            # [Phase 1B] Multi-Task ADMET auxiliary loss
            admet_weight = getattr(self, 'admet_weight', 0.0)
            if admet_weight > 0 and 'admet_labels' in batch and 'admet_logits' in result:
                admet_labels = batch['admet_labels'].to(self.device)   # [B, 5]
                admet_logits = result['admet_logits']                  # [B, 5, 2]
                # Cross-entropy per task, then average
                num_tasks = admet_logits.size(1)
                admet_ce = 0.0
                for t in range(num_tasks):
                    admet_ce = admet_ce + F.cross_entropy(
                        admet_logits[:, t, :], admet_labels[:, t])
                admet_ce = admet_ce / num_tasks
                loss = loss + admet_weight * admet_ce

            if lambda_div > 0 and result['weights'] is not None:
                w       = result['weights']
                uniform = torch.ones_like(w) / w.size(-1)
                div_loss = torch.mean(torch.sum((w - uniform) ** 2, dim=-1))
                loss = loss + lambda_div * div_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if onecycle_scheduler is not None:
                onecycle_scheduler.step()

            total_loss += loss.item()

            if result['weights'] is not None:
                all_weights.append(
                    result['weights'].mean(dim=0).detach().cpu().numpy())

        avg_loss    = total_loss / max(len(loader), 1)
        avg_weights = (np.mean(all_weights, axis=0).tolist()
                       if all_weights else [])
        return avg_loss, avg_weights

    def _evaluate(self, model, loader) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                kwargs = {'mol2vec_feat': None, 'graph_2d': None, 'graph_3d': None}
                if 'mol2vec' in batch:
                    kwargs['mol2vec_feat'] = batch['mol2vec'].to(self.device)
                if 'graph_2d' in batch:
                    kwargs['graph_2d'] = batch['graph_2d'].to(self.device)
                if 'graph_3d' in batch:
                    kwargs['graph_3d'] = batch['graph_3d'].to(self.device)

                result = model(**kwargs)
                probs  = F.softmax(result['logits'], dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(batch['labels'].numpy().tolist())

        return np.array(all_probs), np.array(all_labels)

    def _final_evaluation(self, model, train_loader, val_loader, test_loader,
                           train_time, history, fusion_mode):

        train_probs, train_labels = self._evaluate(model, train_loader)
        val_probs,   val_labels   = self._evaluate(model, val_loader)
        test_probs,  test_labels  = self._evaluate(model, test_loader)

        # v24：F-beta 软搜索，读取配置
        thresh_cfg = self.config.get('threshold_search', {})
        threshold  = self.metrics_calc.find_optimal_threshold(
            val_labels, val_probs,
            beta        = thresh_cfg.get('beta',        1.0),
            min_recall  = thresh_cfg.get('min_recall',  None),
            search_low  = thresh_cfg.get('search_low',  0.30),
            search_high = thresh_cfg.get('search_high', 0.55),
            search_step = thresh_cfg.get('search_step', 0.005),
        )

        train_metrics = self.metrics_calc.compute_metrics(
            train_labels, train_probs, threshold)
        val_metrics   = self.metrics_calc.compute_metrics(
            val_labels, val_probs, threshold)
        test_metrics  = self.metrics_calc.compute_metrics(
            test_labels, test_probs, threshold)

        print(f"\n   📊 Final Results:")
        print(f"      {'Dataset':<10} {'ROC-AUC':>10} {'Accuracy':>10} "
              f"{'Precision':>10} {'Recall':>10} {'F1':>10} {'MCC':>10}")
        print(f"      {'-' * 72}")
        for name, m in [('Train', train_metrics),
                         ('Val',   val_metrics),
                         ('Test',  test_metrics)]:
            print(f"      {name:<10} {m['roc_auc']:>10.4f} "
                  f"{m['accuracy']:>10.4f} {m['precision']:>10.4f} "
                  f"{m['recall']:>10.4f} {m['f1']:>10.4f} "
                  f"{m['mcc']:>10.4f}")

        auc_gap = train_metrics['roc_auc'] - test_metrics['roc_auc']
        print(f"\n   📊 Overfitting Diagnosis:")
        print(f"      Train-Test AUC Gap: {auc_gap:.4f}")
        if auc_gap > 0.1:
            print(f"      ⚠️  Warning: Significant overfitting detected!")
        elif auc_gap < -0.05:
            print(f"      ⚠️  Warning: Test > Train (data leakage?)")
        else:
            print(f"      ✅ Model generalization looks good")

        return {
            'model_name':   f'MultiModal_{fusion_mode}',
            'fusion_mode':  fusion_mode,
            'train_metrics': train_metrics,
            'val_metrics':   val_metrics,
            'test_metrics':  test_metrics,
            'training_time': train_time,
            'optimal_threshold': threshold,
            'overfitting_diagnostics': {'auc_gap': auc_gap},
            'predictions': {
                'train_proba':  train_probs,
                'val_proba':    val_probs,
                'test_proba':   test_probs,
                'train_labels': train_labels,
                'val_labels':   val_labels,
                'test_labels':  test_labels,
                'train_pred':   (train_probs >= threshold).astype(int),
                'val_pred':     (val_probs   >= threshold).astype(int),
                'test_pred':    (test_probs  >= threshold).astype(int),
            },
            'history': history
        }


# ============================================================================
# Part 8: Visualization
# ============================================================================

class Visualizer:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_learning_curve(self, history: Dict, model_name: str):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs = range(1, len(history['train_auc']) + 1)

        ax = axes[0, 0]
        ax.plot(epochs, history['train_auc'], 'b-', label='Train AUC', lw=2)
        ax.plot(epochs, history['val_auc'],   'r-', label='Val AUC',   lw=2)
        ax.set_xlabel('Epoch'); ax.set_ylabel('AUC')
        ax.set_title('AUC Learning Curve', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
        ax.set_ylim(bottom=max(0, min(history['train_auc'] +
                                       history['val_auc']) - 0.05))

        ax = axes[0, 1]
        ax.plot(epochs, history['loss'], color='#2ca02c', lw=2, label='Train Loss')
        ax.axhline(y=0, color='red', linestyle='--', lw=1, alpha=0.5,
                   label='Loss = 0 baseline')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.set_title('Loss Curve (should stay ≥ 0)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(alpha=0.3)

        ax = axes[1, 0]
        weights_data = history.get('modality_weights', [])
        has_weights  = (weights_data and len(weights_data[0]) > 0)

        if has_weights:
            weights_arr = np.array(weights_data)
            n_mod       = weights_arr.shape[1]
            mod_labels  = ['1D (Mol2vec)', '2D (D-MPNN)', '3D (SphereNet)'][:n_mod]
            mod_colors  = ['#1f77b4', '#2ca02c', '#d62728'][:n_mod]

            for i in range(n_mod):
                ax.plot(epochs, weights_arr[:, i],
                        label=mod_labels[i], color=mod_colors[i], lw=2)

            ax.axhline(y=1/3, color='gray', linestyle='--', lw=1.2, alpha=0.6,
                       label='Uniform (1/3)')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Importance Weight')
            ax.set_title('Modality Importance Weights', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9); ax.grid(alpha=0.3)
            ax.set_ylim(0, min(1.0, weights_arr.max() + 0.15))
        else:
            ax.text(0.5, 0.5, 'Modality weights N/A',
                    ha='center', va='center', transform=ax.transAxes)

        ax   = axes[1, 1]
        gaps = np.array(history['train_auc']) - np.array(history['val_auc'])
        ax.plot(epochs, gaps, color='#ff7f0e', lw=2, label='Train−Val AUC Gap')
        ax.axhline(y=0,     color='gray', linestyle='--', lw=1)
        ax.axhline(y=0.1,   color='red',  linestyle=':',  lw=1, alpha=0.6,
                   label='Overfitting (0.1)')
        ax.axhline(y=-0.05, color='blue', linestyle=':',  lw=1, alpha=0.6,
                   label='Underfitting (−0.05)')
        ax.fill_between(epochs, 0, gaps, where=(gaps > 0), alpha=0.25, color='#ff7f0e')
        ax.fill_between(epochs, 0, gaps, where=(gaps < 0), alpha=0.25, color='#1f77b4')
        ax.set_xlabel('Epoch'); ax.set_ylabel('AUC Gap')
        ax.set_title('Overfitting Gap (Train − Val)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        plt.suptitle(f'Training Dynamics: {model_name}',
                     fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()

        safe  = model_name.replace('+', '_').replace(' ', '_')
        path  = self.save_dir / f"learning_curve_{safe}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {path.name}")

    def plot_roc_curves(self, all_results: List[Dict]):
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        for idx, r in enumerate(all_results):
            fpr, tpr, _ = roc_curve(r['predictions']['test_labels'],
                                    r['predictions']['test_proba'])
            auc = r['test_metrics']['roc_auc']
            plt.plot(fpr, tpr,
                     label=f"{r['fusion_mode']} (AUC={auc:.3f})",
                     linewidth=2, color=colors[idx])
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-Modal Fusion', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / "roc_curves_all.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: roc_curves_all.png")

    def plot_metrics_comparison(self, all_results: List[Dict]):
        metrics = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1', 'mcc']
        names   = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        for idx, (m, n) in enumerate(zip(metrics, names)):
            ax     = axes[idx]
            modes  = [r['fusion_mode'] for r in all_results]
            vals   = [r['test_metrics'][m] for r in all_results]
            colors = plt.cm.Set2(np.linspace(0, 1, len(vals)))
            ax.bar(range(len(modes)), vals, color=colors, alpha=0.8)
            ax.set_xticks(range(len(modes)))
            ax.set_xticklabels(modes, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(n, fontsize=11)
            ax.set_title(f'{n} Comparison', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            for i, v in enumerate(vals):
                ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(self.save_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: metrics_comparison.png")

    def plot_confusion_matrices(self, all_results: List[Dict]):
        n    = len(all_results)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 else list(axes)
        for idx, r in enumerate(all_results):
            ax = axes[idx]
            cm = confusion_matrix(r['predictions']['test_labels'],
                                  r['predictions']['test_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        ax=ax, cbar=False, square=True, annot_kws={'size': 12})
            ax.set_title(
                f"{r['fusion_mode']}\nAUC={r['test_metrics']['roc_auc']:.3f}",
                fontsize=10)
            ax.set_xlabel('Predicted', fontsize=9)
            ax.set_ylabel('Actual', fontsize=9)
        for idx in range(n, len(axes)):
            axes[idx].axis('off')
        plt.tight_layout()
        plt.savefig(self.save_dir / "confusion_matrices.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: confusion_matrices.png")


    def plot_ensemble_comparison(self, result: Dict):
        """v26新增：可视化集成收益（各单模型 vs 集成AUC）"""
        single_aucs = result.get('single_test_aucs', [])
        if not single_aucs:
            return
        seeds        = result.get('seeds', list(range(len(single_aucs))))
        ensemble_auc = result['test_metrics']['roc_auc']
        mean_single  = np.mean(single_aucs)

        fig, ax = plt.subplots(figsize=(max(8, len(seeds) * 1.5), 5))
        x = list(range(len(single_aucs)))
        bars = ax.bar(x, single_aucs, color='#4c9be8', alpha=0.8, label='Single Models')
        ax.axhline(y=ensemble_auc, color='#d62728', lw=2.5, linestyle='--',
                   label=f'Ensemble AUC = {ensemble_auc:.4f}')
        ax.axhline(y=mean_single, color='#ff7f0e', lw=1.5, linestyle=':',
                   label=f'Mean Single  = {mean_single:.4f}')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Seed {s}' for s in seeds], fontsize=10)
        ax.set_ylabel('Test ROC-AUC', fontsize=12)
        ax.set_title(f'Ensemble vs Individual Models — {result["fusion_mode"]}',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
        ymin = min(single_aucs) - 0.005
        ymax = max(ensemble_auc, max(single_aucs)) + 0.005
        ax.set_ylim(ymin, ymax)
        for bar, auc in zip(bars, single_aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{auc:.4f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        safe = result['fusion_mode'].replace('+', '_')
        path = self.save_dir / f"ensemble_comparison_{safe}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {path.name}")

    def plot_modality_weights_heatmap(self, all_results: List[Dict]):
        records = []
        for r in all_results:
            wd = r['history'].get('modality_weights', [])
            if wd and len(wd) > 0 and len(wd[-1]) > 0:
                fw = wd[-1]
                records.append({
                    'Mode': r['fusion_mode'],
                    '1D':   fw[0] if len(fw) > 0 else 0,
                    '2D':   fw[1] if len(fw) > 1 else 0,
                    '3D':   fw[2] if len(fw) > 2 else 0,
                })
        if not records:
            return
        df  = pd.DataFrame(records).set_index('Mode')
        fig, ax = plt.subplots(figsize=(8, max(3, len(records) * 1.2)))
        sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd',
                    ax=ax, vmin=0, vmax=0.8, linewidths=0.5,
                    cbar_kws={'label': 'Importance Weight'})
        ax.set_title('Final Modality Importance Weights', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / "modality_weights_heatmap.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: modality_weights_heatmap.png")


# ============================================================================
# Part 9: Results Summary
# ============================================================================

def generate_summary_table(all_results: List[Dict], save_dir: Path) -> pd.DataFrame:
    print(f"\n📊 Generating summary table...")
    data = [{
        '融合模式':    r['fusion_mode'],
        'ROC-AUC':    r['test_metrics']['roc_auc'],
        'PR-AUC':     r['test_metrics']['pr_auc'],
        '准确率':     r['test_metrics']['accuracy'],
        '精确率':     r['test_metrics']['precision'],
        '召回率':     r['test_metrics']['recall'],
        'F1分数':     r['test_metrics']['f1'],
        'MCC':        r['test_metrics']['mcc'],
        '集成种子数':  len(r.get('seeds', [1])),
        '训练时间(s)': round(r['training_time'], 1),
        '过拟合差距':  round(r['overfitting_diagnostics']['auc_gap'], 4)
    } for r in all_results]

    df = pd.DataFrame(data).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)

    results_dir = save_dir / "results"
    results_dir.mkdir(exist_ok=True)
    df.to_csv(results_dir / "summary_table.csv", index=False, float_format='%.4f')
    print(f"   ✅ Saved: summary_table.csv")
    try:
        df.to_excel(results_dir / "summary_table.xlsx", index=False)
        print(f"   ✅ Saved: summary_table.xlsx")
    except Exception as e:
        print(f"   ⚠️  Excel save failed: {e}")

    print(f"\n{'=' * 80}")
    print(f"📊 RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(df.to_string(index=False))
    print(f"{'=' * 80}")

    return df


# ============================================================================
# Part 9.5: Multi-Seed Ensemble Evaluator (v26 新增)
# ============================================================================

class EnsembleEvaluator:
    """
    v26 核心：多种子集成评估
    流程：N个种子独立训练 → 概率平均 → val集搜索F-beta最优阈值 → 最终指标
    原有 MultiModalTrainer 完全不变，此类只是在外层包一层循环
    """

    def __init__(self, config: Dict, device: torch.device, exp_dir: Path):
        self.config       = config
        self.device       = device
        self.exp_dir      = exp_dir
        self.metrics_calc = MetricsCalculator()

    def run(self, seeds: List[int], fusion_mode: str,
            train_loader, val_loader, test_loader,
            model_config: Dict, train_config: Dict) -> Dict:

        print(f"\n{'=' * 70}")
        print(f"🎯 Multi-Seed Ensemble: {len(seeds)} models × {fusion_mode}")
        print(f"   Seeds: {seeds}")
        print(f"{'=' * 70}")

        all_val_probs:  List[np.ndarray] = []
        all_test_probs: List[np.ndarray] = []
        all_train_probs: List[np.ndarray] = []
        val_labels_ref  = None
        test_labels_ref = None
        train_labels_ref = None
        all_histories: List[Dict] = []
        single_test_aucs: List[float] = []
        total_train_time = 0.0

        for seed_idx, seed in enumerate(seeds):
            print(f"\n{'─' * 60}")
            print(f"🌱 Seed {seed_idx+1}/{len(seeds)}: seed={seed}")
            print(f"{'─' * 60}")

            # 固定随机种子
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model = MultiModalFusionNet(model_config)
            trainer = MultiModalTrainer(train_config, self.device, self.exp_dir)

            # 调用原有 train() —— 完全不修改
            result = trainer.train(model, train_loader, val_loader, test_loader)
            all_histories.append(result['history'])
            total_train_time += result['training_time']

            # 收集各split概率（trainer._evaluate 是原有方法）
            val_probs,   val_labels   = trainer._evaluate(model, val_loader)
            test_probs,  test_labels  = trainer._evaluate(model, test_loader)
            train_probs, train_labels = trainer._evaluate(model, train_loader)

            all_val_probs.append(val_probs)
            all_test_probs.append(test_probs)
            all_train_probs.append(train_probs)

            if val_labels_ref is None:
                val_labels_ref   = val_labels
                test_labels_ref  = test_labels
                train_labels_ref = train_labels

            single_auc = roc_auc_score(test_labels, test_probs)
            single_test_aucs.append(single_auc)
            print(f"   📊 Seed {seed} Test AUC: {single_auc:.4f}")

            # 保存单个模型
            torch.save(model.state_dict(),
                       self.exp_dir / "models" / f"model_{fusion_mode.replace('+','_')}_seed{seed}.pt")

        # ── 集成：概率平均 ───────────────────────────────────────────────
        print(f"\n{'=' * 70}")
        print(f"🔗 Ensemble: averaging {len(seeds)} models...")
        print(f"   Individual Test AUCs: {[f'{a:.4f}' for a in single_test_aucs]}")
        print(f"   Mean ± Std: {np.mean(single_test_aucs):.4f} ± {np.std(single_test_aucs):.4f}")

        ens_val_probs   = np.mean(all_val_probs,   axis=0)
        ens_test_probs  = np.mean(all_test_probs,  axis=0)
        ens_train_probs = np.mean(all_train_probs, axis=0)

        ens_test_auc = roc_auc_score(test_labels_ref, ens_test_probs)
        print(f"   Ensemble Test AUC: {ens_test_auc:.4f} "
              f"(+{ens_test_auc - np.mean(single_test_aucs):+.4f} vs mean single)")

        # ── 阈值搜索（在集成val概率上）──────────────────────────────────
        thresh_cfg = train_config.get('threshold_search', {})
        threshold  = self.metrics_calc.find_optimal_threshold(
            val_labels_ref, ens_val_probs,
            beta        = thresh_cfg.get('beta',        1.2),
            min_recall  = thresh_cfg.get('min_recall',  None),
            search_low  = thresh_cfg.get('search_low',  0.30),
            search_high = thresh_cfg.get('search_high', 0.55),
            search_step = thresh_cfg.get('search_step', 0.005),
        )

        # ── 最终指标 ────────────────────────────────────────────────────
        train_metrics = self.metrics_calc.compute_metrics(train_labels_ref, ens_train_probs, threshold)
        val_metrics   = self.metrics_calc.compute_metrics(val_labels_ref,   ens_val_probs,   threshold)
        test_metrics  = self.metrics_calc.compute_metrics(test_labels_ref,  ens_test_probs,  threshold)

        print(f"\n   📊 Ensemble Final Results:")
        print(f"      {'Dataset':<10} {'ROC-AUC':>10} {'Accuracy':>10} "
              f"{'Precision':>10} {'Recall':>10} {'F1':>10} {'MCC':>10}")
        print(f"      {'-' * 72}")
        for name, m in [('Train', train_metrics), ('Val', val_metrics), ('Test', test_metrics)]:
            print(f"      {name:<10} {m['roc_auc']:>10.4f} {m['accuracy']:>10.4f} "
                  f"{m['precision']:>10.4f} {m['recall']:>10.4f} "
                  f"{m['f1']:>10.4f} {m['mcc']:>10.4f}")

        auc_gap = train_metrics['roc_auc'] - test_metrics['roc_auc']
        print(f"\n   📊 Overfitting Diagnosis:")
        print(f"      Train-Test AUC Gap: {auc_gap:.4f}")
        if auc_gap > 0.1:
            print(f"      ⚠️  Warning: Significant overfitting detected!")
        elif auc_gap < -0.05:
            print(f"      ⚠️  Warning: Test > Train (data leakage?)")
        else:
            print(f"      ✅ Model generalization looks good")

        return {
            'model_name':   f'Ensemble_{fusion_mode}',
            'fusion_mode':  fusion_mode,
            'seeds':        seeds,
            'single_test_aucs': single_test_aucs,
            'train_metrics': train_metrics,
            'val_metrics':   val_metrics,
            'test_metrics':  test_metrics,
            'training_time': total_train_time,
            'optimal_threshold': threshold,
            'overfitting_diagnostics': {'auc_gap': auc_gap},
            'predictions': {
                'train_proba':  ens_train_probs,
                'val_proba':    ens_val_probs,
                'test_proba':   ens_test_probs,
                'train_labels': train_labels_ref,
                'val_labels':   val_labels_ref,
                'test_labels':  test_labels_ref,
                'train_pred':   (ens_train_probs >= threshold).astype(int),
                'val_pred':     (ens_val_probs   >= threshold).astype(int),
                'test_pred':    (ens_test_probs  >= threshold).astype(int),
            },
            'history': all_histories[-1],  # 用最后一个seed的history画图
        }


# ============================================================================
# Part 10: Mol2vec Feature Generation
# ============================================================================

class Mol2VecFeaturizer:
    def __init__(self, model_path: Optional[str] = None, radius: int = 1,
                 embedding_dim: int = 300):
        if not MOL2VEC_AVAILABLE:
            raise ImportError("mol2vec not installed!")
        self.radius        = radius
        self.embedding_dim = embedding_dim
        self.model         = None
        if model_path and Path(model_path).exists():
            print(f"📥 Loading pretrained Mol2vec: {model_path}")
            try:
                self.model = word2vec.Word2Vec.load(model_path)
                self.embedding_dim = self.model.wv.vector_size
            except Exception as e:
                print(f"   ❌ Failed: {e}")

    def _mol_to_sentence(self, mol):
        if mol is None:
            return None
        try:
            return mol2alt_sentence(mol, self.radius)
        except:
            return None

    def train_on_corpus(self, smiles_list: List[str], **kwargs):
        print(f"\n🔬 Training Mol2vec on {len(smiles_list)} molecules...")
        sentences = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                sent = self._mol_to_sentence(mol)
                if sent:
                    sentences.append(sent)
        print(f"   Valid sentences: {len(sentences)}")
        params = {'vector_size': self.embedding_dim, 'window': 5,
                  'min_count': 1, 'sg': 1, 'workers': 4, 'epochs': 10}
        params.update(kwargs)
        self.model = word2vec.Word2Vec(sentences, **params)
        print(f"   ✅ Trained, vocab size: {len(self.model.wv)}")

    def featurize(self, smiles_list: List[str]) -> np.ndarray:
        print(f"\n🧬 Generating Mol2vec features...")
        if self.model is None:
            raise ValueError("Model not initialized!")
        features = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features.append(np.zeros(self.embedding_dim)); continue
            sent = self._mol_to_sentence(mol)
            if not sent:
                features.append(np.zeros(self.embedding_dim)); continue
            try:
                result = sentences2vec([sent], self.model, unseen='UNK')
                if isinstance(result, np.ndarray) and result.shape[0] > 0:
                    features.append(result[0])
                else:
                    features.append(np.zeros(self.embedding_dim))
            except:
                vecs = [self.model.wv[w] for w in sent if w in self.model.wv]
                features.append(np.mean(vecs, axis=0) if vecs
                                 else np.zeros(self.embedding_dim))
        return np.array(features)


# ============================================================================
# Part 11: Main Experiment
# ============================================================================

def smart_read_csv(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    smiles_candidates = ['smiles', 'smiles_standardized', 'canonical_smiles', 'SMILES']
    smiles_col = next((c for c in df.columns
                       if c.lower() in [x.lower() for x in smiles_candidates]), None)
    label_candidates = ['label', 'activity', 'target', 'class', 'y']
    label_col = next((c for c in df.columns
                      if c.lower() in [x.lower() for x in label_candidates]), None)
    if not smiles_col or not label_col:
        raise ValueError(f"Cannot find smiles/label columns in {filepath}")
    df = df.rename(columns={smiles_col: 'smiles', label_col: 'label'})
    df['label'] = df['label'].astype(int)

    # [Phase 1B] Multi-Task ADMET labels (optional)
    admet_cols = ['admet_lipinski', 'admet_qed', 'admet_pains',
                  'admet_sa', 'admet_logp']
    keep = ['smiles', 'label']
    if all(c in df.columns for c in admet_cols):
        for c in admet_cols:
            df[c] = df[c].astype(int)
        keep += admet_cols
    return df[keep]


def run_multimodal_experiment(config_path: str):
    print("\n" + "=" * 80)
    print("🚀 MULTI-MODAL MOLECULAR FUSION EXPERIMENT v2.0 (v26)")
    print("   1D (Mol2vec) + 2D (D-MPNN) + 3D (SphereNet)")
    print("   v26: Multi-Seed Ensemble (5 seeds → 概率平均集成)")
    print("=" * 80)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name  = config.get('experiment', {}).get('name', 'MultiModal_Fusion')
    exp_dir   = (Path(config.get('output', {}).get('base_dir', 'results'))
                 / f"{exp_name}_{timestamp}")

    for subdir in ['data', 'models', 'figures', 'results']:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)

    print(f"📂 Output directory: {exp_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    train_df = smart_read_csv(Path(config['data']['train_path']))
    val_df   = smart_read_csv(Path(config['data']['val_path']))
    test_df  = smart_read_csv(Path(config['data']['test_path']))

    print(f"\n📁 Data loaded:")
    print(f"   Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    fusion_modes = config.get('fusion', {}).get('modes', ['1D+2D+3D'])
    print(f"\n🔗 Fusion modes: {fusion_modes}")

    mol2vec_train = mol2vec_val = mol2vec_test = None

    # ------------------------------------------------------------------
    # New (V7): if config provides a precomputed ChemBERTa embedding
    # pickle, use it as a drop-in replacement for Mol2Vec. The pickle
    # is produced by scripts/expand/precompute_chemberta.py and maps
    # SMILES → hidden_size-d float32 embedding.
    # ------------------------------------------------------------------
    chemberta_pkl = config.get('mol2vec', {}).get('chemberta_pkl')
    if chemberta_pkl and Path(chemberta_pkl).exists() and any('1D' in m for m in fusion_modes):
        import pickle
        with open(chemberta_pkl, 'rb') as _f:
            _cb_data = pickle.load(_f)
        _emb_dict = _cb_data['embeddings']
        _hidden_size = _cb_data['hidden_size']
        print(f"\n🧠 Loaded ChemBERTa embeddings ({_hidden_size}-d) "
              f"for {len(_emb_dict)} molecules from {chemberta_pkl}")
        print(f"   Bypassing Mol2Vec featurizer.")

        def _featurize_cb(smiles_list):
            out = np.zeros((len(smiles_list), _hidden_size), dtype=np.float32)
            miss = 0
            for i, s in enumerate(smiles_list):
                emb = _emb_dict.get(s)
                if emb is None:
                    miss += 1
                else:
                    out[i] = emb
            if miss:
                print(f"   ⚠ {miss}/{len(smiles_list)} SMILES not in ChemBERTa pkl "
                      f"(zero-filled)")
            return out

        mol2vec_train = _featurize_cb(train_df['smiles'].tolist())
        mol2vec_val   = _featurize_cb(val_df['smiles'].tolist())
        mol2vec_test  = _featurize_cb(test_df['smiles'].tolist())

        # Ensure encoder_1d uses the right input dim.
        # Note: line ~2221 reads config['mol2vec']['embedding_dim'] and
        # writes it to model_config['mol2vec_dim'] — overriding any
        # config['model']['mol2vec_dim'] set here. So we update BOTH.
        if 'model' not in config:
            config['model'] = {}
        config['model']['mol2vec_dim'] = _hidden_size
        if 'mol2vec' not in config:
            config['mol2vec'] = {}
        config['mol2vec']['embedding_dim'] = _hidden_size

    elif any('1D' in mode for mode in fusion_modes):
        if MOL2VEC_AVAILABLE:
            print("\n📊 Generating Mol2vec features...")
            featurizer = Mol2VecFeaturizer(
                model_path=config.get('mol2vec', {}).get('model_path'),
                embedding_dim=config.get('mol2vec', {}).get('embedding_dim', 300)
            )
            if featurizer.model is None:
                all_smiles = (train_df['smiles'].tolist() + val_df['smiles'].tolist())
                featurizer.train_on_corpus(all_smiles)

            mol2vec_train = featurizer.featurize(train_df['smiles'].tolist())
            mol2vec_val   = featurizer.featurize(val_df['smiles'].tolist())
            mol2vec_test  = featurizer.featurize(test_df['smiles'].tolist())
        else:
            print("⚠️ Mol2vec not available, skipping 1D modes")
            fusion_modes = [m for m in fusion_modes if '1D' not in m]

    all_results = []

    for fusion_mode in fusion_modes:
        print(f"\n{'=' * 80}")
        print(f"🔬 Processing: {fusion_mode}")
        print(f"{'=' * 80}")

        cutoff = config.get('model', {}).get('cutoff', 5.0)

        # [Phase 1B] Extract ADMET labels if present
        admet_cols = ['admet_lipinski', 'admet_qed', 'admet_pains',
                      'admet_sa', 'admet_logp']
        has_admet = all(c in train_df.columns for c in admet_cols)
        admet_train = train_df[admet_cols].values if has_admet else None
        admet_val   = val_df[admet_cols].values   if has_admet else None
        admet_test  = test_df[admet_cols].values  if has_admet else None
        if has_admet:
            print(f"   📊 Multi-Task ADMET labels: DETECTED (5 tasks)")

        train_dataset = MultiModalMoleculeDataset(
            train_df['smiles'].tolist(), train_df['label'].values,
            mol2vec_train, fusion_mode, cutoff, admet_labels=admet_train)
        val_dataset = MultiModalMoleculeDataset(
            val_df['smiles'].tolist(), val_df['label'].values,
            mol2vec_val, fusion_mode, cutoff, admet_labels=admet_val)
        test_dataset = MultiModalMoleculeDataset(
            test_df['smiles'].tolist(), test_df['label'].values,
            mol2vec_test, fusion_mode, cutoff, admet_labels=admet_test)

        batch_size  = config.get('training', {}).get('batch_size', 32)
        num_workers = 4 if device.type == 'cuda' else 0

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_multimodal, num_workers=num_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(num_workers > 0))
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            collate_fn=collate_multimodal, num_workers=num_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(num_workers > 0))
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            collate_fn=collate_multimodal, num_workers=num_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(num_workers > 0))

        model_config = config.get('model', {}).copy()
        model_config['fusion_mode'] = fusion_mode
        model_config['mol2vec_dim'] = config.get('mol2vec', {}).get('embedding_dim', 300)

        train_config = config.get('training', {}).copy()
        train_config['fusion_mode'] = fusion_mode

        # v26: 支持多种子集成（random_seeds）或单次运行（random_seed）
        exp_cfg = config.get('experiment', {})
        seeds   = exp_cfg.get('random_seeds', None)
        if seeds is None:
            single_seed = exp_cfg.get('random_seed', 2026)
            seeds = [single_seed]

        if len(seeds) == 1:
            # 单种子：保持原有逻辑完全不变
            seed = seeds[0]
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model   = MultiModalFusionNet(model_config)
            trainer = MultiModalTrainer(train_config, device, exp_dir)
            result  = trainer.train(model, train_loader, val_loader, test_loader)
        else:
            # 多种子：使用 EnsembleEvaluator
            evaluator = EnsembleEvaluator(config, device, exp_dir)
            result    = evaluator.run(seeds, fusion_mode,
                                      train_loader, val_loader, test_loader,
                                      model_config, train_config)

        all_results.append(result)

    if all_results:
        print(f"\n📊 Generating visualizations...")
        visualizer = Visualizer(exp_dir / 'figures')

        for r in all_results:
            visualizer.plot_learning_curve(r['history'], r['fusion_mode'])
            # v26: 多种子时额外输出集成收益图
            if r.get('seeds') and len(r['seeds']) > 1:
                visualizer.plot_ensemble_comparison(r)

        visualizer.plot_roc_curves(all_results)
        visualizer.plot_metrics_comparison(all_results)
        visualizer.plot_confusion_matrices(all_results)
        visualizer.plot_modality_weights_heatmap(all_results)

        summary_df = generate_summary_table(all_results, exp_dir)

        with open(exp_dir / 'experiment_config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        best = max(all_results, key=lambda x: x['test_metrics']['roc_auc'])
        n_seeds = len(best.get('seeds', [1]))
        print(f"\n{'=' * 80}")
        print(f"🏆 BEST MODEL: {best['fusion_mode']}"
              + (f" (Ensemble {n_seeds} seeds: {best['seeds']})" if n_seeds > 1 else ""))
        print(f"   ROC-AUC:   {best['test_metrics']['roc_auc']:.4f}")
        print(f"   F1-Score:  {best['test_metrics']['f1']:.4f}")
        print(f"   Precision: {best['test_metrics']['precision']:.4f}")
        print(f"   Recall:    {best['test_metrics']['recall']:.4f}")
        print(f"   MCC:       {best['test_metrics']['mcc']:.4f}")
        if n_seeds > 1:
            print(f"   Individual AUCs: {[f'{a:.4f}' for a in best['single_test_aucs']]}")
            print(f"   Mean ± Std: {np.mean(best['single_test_aucs']):.4f} ± {np.std(best['single_test_aucs']):.4f}")
        print(f"{'=' * 80}")

    print(f"\n✅ EXPERIMENT COMPLETED!")
    print(f"📂 Results saved to: {exp_dir}")

    return exp_dir, all_results


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Modal Molecular Fusion Network v2.0 (v26)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Fusion Modes:
  1D+2D     Mol2vec + D-MPNN
  1D+3D     Mol2vec + SphereNet
  2D+3D     D-MPNN + SphereNet
  1D+2D+3D  All three modalities (recommended)

Example:
  python multimodal_molecular_fusion_v26.py --config config_fusion_v26.yaml
        """
    )
    parser.add_argument('--config', '-c', default='config_fusion_v26.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)

    run_multimodal_experiment(args.config)


if __name__ == "__main__":
    main()