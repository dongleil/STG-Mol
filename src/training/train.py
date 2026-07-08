#!/usr/bin/env python3
"""
Multi-Modal Molecular Fusion Network - v4.4
=======================================================
v4.3 → v4.4 核心改动（基于 v4.3 实际训练曲线诊断）：

  [问题1] 过拟合持续恶化（最核心问题）
    - epoch470 best_val_auc=0.9547 后 val 持续下降，train→1.0
    - gap 在 epoch650 达到 0.0821，正则强度整体偏低一档
    → dropout 0.45→0.48，spherenet_dropout 0.32→0.35
      weight_decay 0.022→0.028，modality_drop_prob 0.45→0.48
      label_smoothing 0.15→0.18

  [问题2] patience 过大，有效训练 epoch 仅~470
    - patience=220 让训练白跑至 epoch650，浪费算力
    → epochs 650→550，patience 220→160

  [问题3] warmup 过短导致前期过快收敛
    - warmup_pct=0.20 使 gap 在 epoch100 时已为 -0.014，
      然后迅速反弹过拟合
    → warmup_pct 0.20→0.25（放缓前期 LR 上升斜率）

  [问题4] Balance loss 归零过早，约束已失效
    - epoch200 后 balance_loss ≈ 0，weight=0.15 形同虚设
    → balance_loss_weight 0.15→0.08，进一步释放 attention 动态性

  [新增1] SWA（随机权重平均）
    - epoch 300+ 启用 SWA，与 EMA 并行
    - 平滑 loss landscape，改善泛化

  [新增2] 1D 特征噪声增强
    - 训练时对 mol2vec 特征加入 std=0.02 的高斯噪声
    - 防止 1D encoder 过拟合固定嵌入

  预期效果：
    Gap:          0.046 → <0.038
    Best Val AUC: 0.9547 → >0.958
    Test AUC:     0.9353 → >0.940
    Train 最终:   1.000  → <0.980

Version: 4.4
"""

import os
import sys
import shutil
import warnings
import argparse
import math
import time
import json
import copy
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
    f1_score, confusion_matrix, roc_curve, matthews_corrcoef,
    average_precision_score
)

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, SWALR               # [v4.4] SWA

from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import radius_graph
from torch_geometric.utils import softmax as pyg_softmax

try:
    from mol2vec.features import mol2alt_sentence, sentences2vec
    from gensim.models import word2vec
    MOL2VEC_AVAILABLE = True
except ImportError:
    MOL2VEC_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# Part 1: 工具函数
# ============================================================================

def rotation_matrix(axis: str, angle: torch.Tensor) -> torch.Tensor:
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    zero, one = torch.zeros_like(angle), torch.ones_like(angle)
    if axis == 'x':
        return torch.stack([
            torch.stack([one, zero, zero]),
            torch.stack([zero, cos_a, -sin_a]),
            torch.stack([zero, sin_a, cos_a])
        ]).squeeze()
    elif axis == 'y':
        return torch.stack([
            torch.stack([cos_a, zero, sin_a]),
            torch.stack([zero, one, zero]),
            torch.stack([-sin_a, zero, cos_a])
        ]).squeeze()
    else:
        return torch.stack([
            torch.stack([cos_a, -sin_a, zero]),
            torch.stack([sin_a, cos_a, zero]),
            torch.stack([zero, zero, one])
        ]).squeeze()


def random_rotation_matrix(device='cpu') -> torch.Tensor:
    angles = torch.rand(3, device=device) * 2 * math.pi
    Rx = rotation_matrix('x', angles[0])
    Ry = rotation_matrix('y', angles[1])
    Rz = rotation_matrix('z', angles[2])
    return Rz @ Ry @ Rx


# ============================================================================
# Part 2: 3D 构象生成器
# ============================================================================

class EnhancedConformerGenerator:
    def __init__(self, num_confs: int = 10, optimize: bool = True,
                 max_iters: int = 500, select_by_energy: bool = True):
        self.num_confs = num_confs
        self.optimize = optimize
        self.max_iters = max_iters
        self.select_by_energy = select_by_energy
        self.params = AllChem.ETKDGv3()
        self.params.randomSeed = 42
        self.params.useSmallRingTorsions = True
        self.params.useMacrocycleTorsions = True
        self.params.numThreads = 0

    def generate(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        if mol is None:
            return None
        try:
            mol = Chem.AddHs(mol)
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                pass

            conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=self.num_confs,
                                                   params=self.params)
            if len(conf_ids) == 0:
                result = AllChem.EmbedMolecule(mol, self.params)
                if result == -1:
                    AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
                if mol.GetNumConformers() == 0:
                    return None
                conf_ids = [0]

            if self.optimize and mol.GetNumConformers() > 0:
                results = AllChem.MMFFOptimizeMoleculeConfs(
                    mol, mmffVariant='MMFF94s',
                    maxIters=self.max_iters, numThreads=0)
                if self.select_by_energy and len(results) > 0:
                    energies = [(i, e if c == 0 else float('inf'))
                                for i, (c, e) in enumerate(results)]
                    best_idx = min(energies, key=lambda x: x[1])[0]
                    new_mol = Chem.RWMol(mol)
                    for i in sorted(range(mol.GetNumConformers()), reverse=True):
                        if i != best_idx:
                            new_mol.RemoveConformer(i)
                    mol = new_mol.GetMol()

            mol = Chem.RemoveHs(mol)
            return mol
        except Exception:
            return None


class ConformerAugmentation:
    @staticmethod
    def random_rotation(pos: torch.Tensor) -> torch.Tensor:
        R = random_rotation_matrix(pos.device)
        center = pos.mean(dim=0, keepdim=True)
        return (pos - center) @ R.T + center

    @staticmethod
    def random_translation(pos: torch.Tensor, std: float = 0.5) -> torch.Tensor:
        return pos + torch.randn(1, 3, device=pos.device) * std

    @staticmethod
    def random_noise(pos: torch.Tensor, std: float = 0.05) -> torch.Tensor:
        return pos + torch.randn_like(pos) * std

    @staticmethod
    def augment(pos: torch.Tensor, rotation=True, translation=True,
                noise=True, noise_std=0.03) -> torch.Tensor:
        if rotation:
            pos = ConformerAugmentation.random_rotation(pos)
        if translation:
            pos = ConformerAugmentation.random_translation(pos, std=0.3)
        if noise:
            pos = ConformerAugmentation.random_noise(pos, std=noise_std)
        return pos


# ============================================================================
# Part 3: 特征生成
# ============================================================================

def generate_atom_features_extended(atom, mol=None) -> List[float]:
    """扩展原子特征（47 维）"""
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

        hyb_map = {
            Chem.HybridizationType.SP: 0,
            Chem.HybridizationType.SP2: 1,
            Chem.HybridizationType.SP3: 2,
            Chem.HybridizationType.SP3D: 3
        }
        hyb_idx = hyb_map.get(atom.GetHybridization(), 4)
        for h in range(5):
            features.append(float(hyb_idx == h))

        features.append(float(atom.GetIsAromatic()))
        features.append(float(atom.GetTotalNumHs()) / 4.0)

        in_ring = atom.IsInRing()
        features.append(float(in_ring))
        for size in [3, 4, 5, 6, 7]:
            features.append(float(in_ring and atom.IsInRingSize(size)))

        en = {'C': 2.55, 'N': 3.04, 'O': 3.44, 'S': 2.58, 'F': 3.98,
              'Cl': 3.16, 'Br': 2.96, 'I': 2.66, 'P': 2.19, 'H': 2.20}
        features.append((en.get(symbol, 2.5) - 2.0) / 2.0)

        features.append(atom.GetMass() / 100.0)

        try:
            gc = float(atom.GetProp('_GasteigerCharge'))
            gc = 0.0 if math.isnan(gc) or math.isinf(gc) else gc
        except:
            gc = 0.0
        features.append(gc)

        features.append(float(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED))

        is_donor = atom.GetTotalNumHs() > 0 and symbol in ['N', 'O']
        features.append(float(is_donor))

        is_acceptor = symbol in ['N', 'O', 'F']
        features.append(float(is_acceptor))

        features.append(float(atom.GetTotalValence()) / 6.0)
        features.append(float(atom.GetImplicitValence()) / 4.0)
        features.append(float(atom.GetNumRadicalElectrons()))
        features.append(atom.GetAtomicNum() / 53.0)
        features.append(float(in_ring and atom.GetIsAromatic()))

        heavy_neighbors = len([n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1])
        features.append(float(heavy_neighbors) / 4.0)

        features.append(float(atom.GetFormalCharge() < 0))
        features.append(float(atom.GetFormalCharge() > 0))

        try:
            ring_info = atom.GetOwningMol().GetRingInfo()
            num_rings = len([r for r in ring_info.AtomRings() if atom.GetIdx() in r])
        except:
            num_rings = 0
        features.append(float(num_rings) / 3.0)

        features.append(float(symbol not in ['C', 'H']))
        features.append(float(atom.GetHybridization() == Chem.HybridizationType.SP3D2))
        features.append(float(atom.GetExplicitValence()) / 6.0)

    except Exception:
        features = [0.0] * 47

    if len(features) < 47:
        features.extend([0.0] * (47 - len(features)))
    return features[:47]


def generate_bond_features_extended(bond) -> List[float]:
    """扩展边特征（14 维）"""
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

        bond_lengths = {'SINGLE': 1.54, 'DOUBLE': 1.34, 'TRIPLE': 1.20, 'AROMATIC': 1.40}
        features.append(bond_lengths.get(str(bt).split('.')[-1], 1.5) / 2.0)
        features.append(float(bt == Chem.BondType.SINGLE and not bond.GetIsAromatic()))

    except Exception:
        features = [0.0] * 14

    if len(features) < 14:
        features.extend([0.0] * (14 - len(features)))
    return features[:14]


# ============================================================================
# Part 4: 1D 编码器 —— 残差网络
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.block(x))


class Mol2VecEncoderImproved(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=256, output_dim=128,
                 num_blocks=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.output_dim = output_dim

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_proj(x)


# ============================================================================
# Part 5: 2D 编码器 —— D-MPNN
# ============================================================================

class DMPNNImproved(nn.Module):
    def __init__(self, in_channels=47, edge_dim=14, hidden_channels=128,
                 output_dim=128, num_layers=5, dropout=0.2,
                 use_virtual_node=True):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.output_dim = output_dim
        self.use_virtual_node = use_virtual_node

        self.W_i = nn.Linear(in_channels + edge_dim, hidden_channels)
        self.W_h = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers)
        ])
        self.edge_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels * 2 + edge_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, edge_dim)
            ) for _ in range(num_layers)
        ])
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])

        if use_virtual_node:
            self.virtual_node_embed = nn.Embedding(1, hidden_channels)
            self.virtual_mlp = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels),
                    nn.LayerNorm(hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ) for _ in range(num_layers)
            ])

        self.W_o = nn.Linear(in_channels + hidden_channels, hidden_channels)
        self.bn_out = nn.BatchNorm1d(hidden_channels)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (data.x, data.edge_index,
                                            data.edge_attr, data.batch)
        row, col = edge_index
        num_edges = edge_index.size(1)
        num_nodes = x.size(0)
        batch_size = batch.max().item() + 1

        edge_input = torch.cat([x[row], edge_attr], dim=-1)
        edge_hidden = F.relu(self.W_i(edge_input))
        reverse_edge_idx = self._get_reverse_edge_indices(edge_index, num_edges)

        if self.use_virtual_node:
            virtual_node = self.virtual_node_embed.weight.expand(batch_size, -1)

        for layer_idx in range(self.num_layers):
            node_messages = torch.zeros(num_nodes, self.hidden_channels,
                                        device=x.device)
            node_messages.scatter_add_(
                0, col.unsqueeze(-1).expand_as(edge_hidden), edge_hidden)

            if self.use_virtual_node:
                virtual_agg = global_mean_pool(node_messages, batch)
                virtual_node = self.virtual_mlp[layer_idx](
                    torch.cat([virtual_node, virtual_agg], dim=-1)
                ) + virtual_node
                node_messages = node_messages + virtual_node[batch]

            edge_msg = node_messages[row]
            if reverse_edge_idx is not None:
                valid_mask = reverse_edge_idx >= 0
                if valid_mask.any():
                    edge_msg[valid_mask] = (edge_msg[valid_mask]
                                            - edge_hidden[reverse_edge_idx[valid_mask]])

            edge_update_input = torch.cat([edge_hidden, edge_msg, edge_attr], dim=-1)
            edge_attr = edge_attr + self.edge_update[layer_idx](edge_update_input)

            edge_hidden_new = self.W_h[layer_idx](edge_msg)
            edge_hidden_new = self.bn_layers[layer_idx](edge_hidden_new)
            edge_hidden_new = F.relu(edge_hidden_new)
            edge_hidden = (edge_hidden
                           + F.dropout(edge_hidden_new, p=self.dropout,
                                       training=self.training))

        node_hidden = torch.zeros(num_nodes, self.hidden_channels, device=x.device)
        node_hidden.scatter_add_(0, col.unsqueeze(-1).expand_as(edge_hidden), edge_hidden)

        node_output = torch.cat([x, node_hidden], dim=-1)
        node_output = F.relu(self.bn_out(self.W_o(node_output)))
        node_output = F.dropout(node_output, p=self.dropout, training=self.training)

        h_mean = global_mean_pool(node_output, batch)
        h_max = global_max_pool(node_output, batch)
        graph_repr = torch.cat([h_mean, h_max], dim=-1)
        return self.output_proj(graph_repr)

    def _get_reverse_edge_indices(self, edge_index, num_edges):
        if num_edges == 0:
            return None
        row, col = edge_index
        device = edge_index.device
        max_nodes = max(row.max().item(), col.max().item()) + 1
        edge_hash = row * max_nodes + col
        reverse_hash = col * max_nodes + row
        sorted_hash, sorted_indices = torch.sort(edge_hash)
        positions = torch.searchsorted(
            sorted_hash, reverse_hash).clamp(max=num_edges - 1)
        found_mask = sorted_hash[positions] == reverse_hash
        return torch.where(found_mask, sorted_indices[positions],
                           torch.tensor(-1, device=device))


# ============================================================================
# Part 6: 3D 编码器 —— SchNet
# ============================================================================

class RBFExpansion(nn.Module):
    def __init__(self, num_rbf=50, cutoff=8.0, trainable=False):
        super().__init__()
        centers = torch.linspace(0, cutoff, num_rbf)
        widths = torch.ones(num_rbf) * (cutoff / num_rbf)
        if trainable:
            self.centers = nn.Parameter(centers)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('widths', widths)

    def forward(self, distances):
        return torch.exp(
            -((distances.unsqueeze(-1) - self.centers) ** 2)
            / (self.widths ** 2 + 1e-8))


class SchNetInteraction(nn.Module):
    def __init__(self, hidden_channels, num_rbf=48, cutoff=8.0, dropout=0.3):
        super().__init__()
        self.cutoff = cutoff
        self.rbf = RBFExpansion(num_rbf=num_rbf, cutoff=cutoff, trainable=True)

        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.layer_norm = nn.LayerNorm(hidden_channels)

    def forward(self, x, edge_index, dist):
        row, col = edge_index
        rbf_feat = self.rbf(dist)
        filter_weight = self.filter_net(rbf_feat)
        messages = self.msg_mlp(x[col]) * filter_weight

        aggr = torch.zeros_like(x)
        aggr.scatter_add_(0, row.unsqueeze(-1).expand_as(messages), messages)

        out = self.update_mlp(torch.cat([x, aggr], dim=-1))
        return self.layer_norm(x + out)


class SchNetEncoder(nn.Module):
    def __init__(self, in_channels=47, hidden_channels=128, output_dim=128,
                 num_layers=8, num_rbf=48, cutoff=10.0, dropout=0.20, **kwargs):
        super().__init__()
        self.cutoff = cutoff
        self.output_dim = output_dim

        self.atom_embed = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.z_embed = nn.Embedding(100, hidden_channels)

        self.interactions = nn.ModuleList([
            SchNetInteraction(hidden_channels, num_rbf, cutoff, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, data):
        x, z, pos, edge_index, batch = (data.x, data.z, data.pos,
                                         data.edge_index, data.batch)
        h = self.atom_embed(x) + self.z_embed(z)

        row, col = edge_index
        dist = torch.norm(pos[row] - pos[col], dim=-1)

        for interaction in self.interactions:
            h = interaction(h, edge_index, dist)

        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_sum = global_add_pool(h, batch)
        graph_repr = torch.cat([h_mean, h_max, h_sum], dim=-1)
        return self.output_proj(graph_repr)


# ============================================================================
# Part 7: 融合模块
# ============================================================================

class SelfAttentionFusion(nn.Module):
    """
    Self-Attention 多模态融合
    forward(h_1d, h_2d, h_3d) -> (fused [B,D], weights [B,3])
    权重由 pool_attn 的 softmax 输出决定，可在 [0,1] 自由变化
    由 balance_loss 软边界约束在 [lower, upper] 内
    """

    def __init__(self, dim_1d=128, dim_2d=128, dim_3d=128,
                 fusion_dim=128, num_heads=8, num_layers=2, dropout=0.2):
        super().__init__()
        self.fusion_dim = fusion_dim

        self.proj_1d = nn.Sequential(nn.Linear(dim_1d, fusion_dim),
                                     nn.LayerNorm(fusion_dim))
        self.proj_2d = nn.Sequential(nn.Linear(dim_2d, fusion_dim),
                                     nn.LayerNorm(fusion_dim))
        self.proj_3d = nn.Sequential(nn.Linear(dim_3d, fusion_dim),
                                     nn.LayerNorm(fusion_dim))

        self.modality_embedding = nn.Embedding(3, fusion_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=fusion_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.pool_query = nn.Parameter(torch.randn(1, 1, fusion_dim))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True)
        self.pool_norm = nn.LayerNorm(fusion_dim)

    def forward(self, h_1d: torch.Tensor, h_2d: torch.Tensor,
                h_3d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = h_1d.size(0)
        device = h_1d.device

        z1 = self.proj_1d(h_1d)
        z2 = self.proj_2d(h_2d)
        z3 = self.proj_3d(h_3d)

        mod_emb = self.modality_embedding(torch.arange(3, device=device))
        z1 = z1 + mod_emb[0]
        z2 = z2 + mod_emb[1]
        z3 = z3 + mod_emb[2]

        seq = torch.stack([z1, z2, z3], dim=1)        # [B, 3, d]
        seq_out = self.transformer_encoder(seq)        # [B, 3, d]

        query = self.pool_query.expand(B, -1, -1)      # [B, 1, d]
        pooled, attn_weights = self.pool_attn(
            query=query, key=seq_out, value=seq_out)
        # pooled: [B, 1, d],  attn_weights: [B, 1, 3]

        fused = self.pool_norm(pooled.squeeze(1))      # [B, d]
        weights = attn_weights.squeeze(1)              # [B, 3]
        return fused, weights


class ConcatFusion(nn.Module):
    """拼接融合（对照组）"""

    def __init__(self, dim_1d=128, dim_2d=128, dim_3d=128,
                 fusion_dim=128, dropout=0.2):
        super().__init__()
        total_dim = dim_1d + dim_2d + dim_3d
        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_dim, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

    def forward(self, h_1d: torch.Tensor, h_2d: torch.Tensor,
                h_3d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        concat = torch.cat([h_1d, h_2d, h_3d], dim=-1)
        fused = self.fusion_mlp(concat)
        B = h_1d.size(0)
        weights = torch.ones(B, 3, device=h_1d.device) / 3.0
        return fused, weights


# ============================================================================
# Part 8: 主模型
# ============================================================================

class MultiModalFusionNet(nn.Module):
    """多模态融合网络 v4.4"""

    FUSION_MODES = ['1D', '2D', '3D','1D+2D','1D+3D','2D+3D' '1D+2D+3D']

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.fusion_mode = config.get('fusion_mode', '1D+2D+3D')
        self.fusion_type = config.get('fusion_type', 'self_attention')

        hidden_dim = config.get('hidden_dim', 128)
        dropout    = config.get('dropout', 0.48)          # [v4.4] 默认 0.48
        atom_dim   = config.get('atom_dim', 47)
        edge_dim   = config.get('edge_dim', 14)

        # ---------- 三个 Encoder ----------
        if '1D' in self.fusion_mode:
            self.encoder_1d = Mol2VecEncoderImproved(
                input_dim=config.get('mol2vec_dim', 300),
                hidden_dim=hidden_dim * 2,
                output_dim=hidden_dim,
                num_blocks=config.get('num_res_blocks', 1),
                dropout=dropout
            )

        if '2D' in self.fusion_mode:
            self.encoder_2d = DMPNNImproved(
                in_channels=atom_dim, edge_dim=edge_dim,
                hidden_channels=hidden_dim, output_dim=hidden_dim,
                num_layers=config.get('dmpnn_layers', 5),
                dropout=dropout,
                use_virtual_node=config.get('use_virtual_node', True)
            )

        if '3D' in self.fusion_mode:
            self.encoder_3d = SchNetEncoder(
                in_channels=atom_dim, hidden_channels=hidden_dim,
                output_dim=hidden_dim,
                num_layers=config.get('spherenet_layers', 8),
                cutoff=config.get('cutoff', 10.0),
                dropout=config.get('spherenet_dropout', 0.35)  # [v4.4] 默认 0.35
            )

        # ---------- 融合模块 ----------
        if self.fusion_mode == '1D+2D+3D':
            # 三模态融合（保持不变）
            if self.fusion_type == 'self_attention':
                self.fusion = SelfAttentionFusion(
                    dim_1d=hidden_dim, dim_2d=hidden_dim, dim_3d=hidden_dim,
                    fusion_dim=hidden_dim,
                    num_heads=config.get('fusion_num_heads', 8),
                    num_layers=config.get('fusion_num_layers', 2),
                    dropout=dropout
                )
            elif self.fusion_type == 'concat':
                self.fusion = ConcatFusion(
                    dim_1d=hidden_dim, dim_2d=hidden_dim, dim_3d=hidden_dim,
                    fusion_dim=hidden_dim, dropout=dropout
                )
            classifier_input_dim = hidden_dim

        elif self.fusion_mode == '1D+2D':
            # 新增：双模态融合（拼接 + 线性投影）
            self.fusion = None  # 不使用注意力，后续直接在 forward 中拼接
            # 定义投影层：将拼接后的 [hidden_dim*2] 映射回 hidden_dim
            self.fusion_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            classifier_input_dim = hidden_dim

        elif self.fusion_mode == '1D+3D':
            # 新增：双模态融合（拼接 + 线性投影）
            self.fusion = None  # 不使用注意力，后续直接在 forward 中拼接
            # 定义投影层：将拼接后的 [hidden_dim*2] 映射回 hidden_dim
            self.fusion_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            classifier_input_dim = hidden_dim

        elif self.fusion_mode == '2D+3D':
            # 新增：双模态融合（拼接 + 线性投影）
            self.fusion = None  # 不使用注意力，后续直接在 forward 中拼接
            # 定义投影层：将拼接后的 [hidden_dim*2] 映射回 hidden_dim
            self.fusion_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            classifier_input_dim = hidden_dim

        else:
            # 单模态：1D、2D、3D
            self.fusion = None
            classifier_input_dim = hidden_dim

        # ---------- 分类头 ----------
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, mol2vec_feat=None, graph_2d=None, graph_3d=None,
                ablation_mask=None):
        # ── Step 1: 各模态编码 ──────────────────────────────────────────
        f1d = self.encoder_1d(mol2vec_feat) if (
            mol2vec_feat is not None and hasattr(self, 'encoder_1d')) else None
        f2d = self.encoder_2d(graph_2d) if (
            graph_2d is not None and hasattr(self, 'encoder_2d')) else None
        f3d = self.encoder_3d(graph_3d) if (
            graph_3d is not None and hasattr(self, 'encoder_3d')) else None

        # ── Step 2: 随机模态丢弃 ────────────────────────────────────────
        if self.training and self.fusion_mode == '1D+2D+3D' and ablation_mask is None:
            drop_prob = self.config.get('modality_drop_prob', 0.48)
            dev = next(p for p in [f1d, f2d, f3d] if p is not None).device
            m_masks = torch.ones(3, device=dev)

            if f1d is not None and torch.rand(1).item() < drop_prob:
                m_masks[0] = 0
            if f2d is not None and torch.rand(1).item() < drop_prob:
                m_masks[1] = 0
            if f3d is not None and torch.rand(1).item() < drop_prob:
                m_masks[2] = 0

            if m_masks.sum() == 0:
                m_masks[torch.randint(0, 3, (1,)).item()] = 1

            if f1d is not None: f1d = f1d * m_masks[0]
            if f2d is not None: f2d = f2d * m_masks[1]
            if f3d is not None: f3d = f3d * m_masks[2]

        # ── Step 3: 消融实验掩码 ────────────────────────────────────────
        if ablation_mask is not None:
            if not ablation_mask.get('1d', True) and f1d is not None:
                f1d = torch.zeros_like(f1d)
            if not ablation_mask.get('2d', True) and f2d is not None:
                f2d = torch.zeros_like(f2d)
            if not ablation_mask.get('3d', True) and f3d is not None:
                f3d = torch.zeros_like(f3d)

        # ── Step 4: 特征融合 ────────────────────────────────────────────
        weights = None

        if self.fusion_mode == '1D':
            x = f1d
        elif self.fusion_mode == '2D':
            x = f2d
        elif self.fusion_mode == '3D':
            x = f3d
        elif self.fusion_mode == '1D+2D':
            # 双模态拼接 + 投影
            x = torch.cat([f1d, f2d], dim=-1)  # [B, hidden_dim*2]
            if hasattr(self, 'fusion_proj'):
                x = self.fusion_proj(x)  # [B, hidden_dim]
            weights = None  # 双模态无权重输出

        elif self.fusion_mode == '1D+3D':
            # 双模态拼接 + 投影
            x = torch.cat([f1d, f3d], dim=-1)  # [B, hidden_dim*2]
            if hasattr(self, 'fusion_proj'):
                x = self.fusion_proj(x)  # [B, hidden_dim]
            weights = None  # 双模态无权重输出

        elif self.fusion_mode == '2D+3D':
            # 双模态拼接 + 投影
            x = torch.cat([f2d, f3d], dim=-1)  # [B, hidden_dim*2]
            if hasattr(self, 'fusion_proj'):
                x = self.fusion_proj(x)  # [B, hidden_dim]
            weights = None  # 双模态无权重输出
        else:
            # 三模态融合（保持不变）
            x, weights = self.fusion(f1d, f2d, f3d)

        # ── Step 5: 分类 ────────────────────────────────────────────────
        logits = self.classifier(x)

        return {
            'logits':   logits,
            'features': x,
            'weights':  weights
        }

    def get_parameter_groups(self, base_lr, branch_scales=None):
        if branch_scales is None:
            branch_scales = {}
        param_groups = []

        if hasattr(self, 'encoder_1d'):
            param_groups.append({
                'params': self.encoder_1d.parameters(),
                'lr': base_lr * branch_scales.get('encoder_1d', 0.10),  # [v4.4] 默认 0.10
                'name': 'encoder_1d'
            })
        if hasattr(self, 'encoder_2d'):
            param_groups.append({
                'params': self.encoder_2d.parameters(),
                'lr': base_lr * branch_scales.get('encoder_2d', 2.5),
                'name': 'encoder_2d'
            })
        if hasattr(self, 'encoder_3d'):
            param_groups.append({
                'params': self.encoder_3d.parameters(),
                'lr': base_lr * branch_scales.get('encoder_3d', 0.80),  # [v4.4] 默认 0.80
                'name': 'encoder_3d'
            })
        if self.fusion is not None:
            param_groups.append({
                'params': self.fusion.parameters(),
                'lr': base_lr * branch_scales.get('fusion', 0.8),
                'name': 'fusion'
            })
        param_groups.append({
            'params': self.classifier.parameters(),
            'lr': base_lr * branch_scales.get('classifier', 1.0),
            'name': 'classifier'
        })
        return param_groups


# ============================================================================
# Part 9: 损失函数
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight,
                                  reduction='none',
                                  label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ============================================================================
# Part 10: EMA
# ============================================================================

class EMA:
    def __init__(self, model, decay=0.9998):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (self.decay * self.shadow[name]
                                     + (1 - self.decay) * param.data)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ============================================================================
# Part 11: 数据处理
# ============================================================================

def mol_to_2d_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except:
        pass

    x = torch.tensor([generate_atom_features_extended(atom, mol)
                       for atom in mol.GetAtoms()], dtype=torch.float)
    if x.size(0) == 0:
        return None

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = generate_bond_features_extended(bond)
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([bf, bf])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 14), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def mol_to_3d_graph(smiles, cutoff=10.0, conformer_gen=None):
    """返回 (Data, status_str) 或 (None, reason_str)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid_smiles"

    if conformer_gen is None:
        conformer_gen = EnhancedConformerGenerator()

    mol_3d = conformer_gen.generate(mol)
    if mol_3d is None or mol_3d.GetNumConformers() == 0:
        return None, "conformer_failed"

    conf = mol_3d.GetConformer()
    positions = []
    for i in range(mol_3d.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        positions.append([pos.x, pos.y, pos.z])
    pos_tensor = torch.tensor(positions, dtype=torch.float)

    atom_features, atomic_numbers = [], []
    for atom in mol_3d.GetAtoms():
        atom_features.append(generate_atom_features_extended(atom, mol_3d))
        atomic_numbers.append(atom.GetAtomicNum())

    x = torch.tensor(atom_features, dtype=torch.float)
    z = torch.tensor(atomic_numbers, dtype=torch.long)
    if x.size(0) == 0:
        return None, "empty_atoms"

    edge_index = radius_graph(pos_tensor, r=cutoff, loop=False)
    if edge_index.size(1) == 0:
        return None, "no_edges"

    row, col = edge_index
    edge_weight = torch.norm(pos_tensor[row] - pos_tensor[col], dim=-1)
    return (Data(x=x, z=z, pos=pos_tensor, edge_index=edge_index,
                 edge_weight=edge_weight),
            "ok")


class MultiModalDataset(Dataset):
    def __init__(self, smiles_list, labels, mol2vec_features=None,
                 fusion_mode='1D+2D+3D', cutoff=10.0, augment_3d=False,
                 mol2vec_noise_std=0.02):
        self.fusion_mode = fusion_mode
        self.cutoff = cutoff
        self.augment_3d = augment_3d
        self.mol2vec_noise_std = mol2vec_noise_std  # [v4.4] 1D 噪声增强强度
        self._training = False
        self.data_list = []

        conformer_gen = (EnhancedConformerGenerator(num_confs=10)
                         if '3D' in fusion_mode else None)

        print(f"Building dataset (mode: {fusion_mode})...")
        for idx, (smiles, label) in enumerate(zip(smiles_list, labels)):
            sample = {'smiles': smiles, 'label': int(label)}
            valid = True

            if '1D' in fusion_mode and mol2vec_features is not None:
                sample['mol2vec'] = torch.as_tensor(
                    mol2vec_features[idx], dtype=torch.float)

            if '2D' in fusion_mode:
                graph_2d = mol_to_2d_graph(smiles)
                if graph_2d is None:
                    valid = False
                else:
                    sample['graph_2d'] = graph_2d

            if '3D' in fusion_mode:
                graph_3d, status = mol_to_3d_graph(smiles, cutoff, conformer_gen)
                if graph_3d is None:
                    valid = False
                else:
                    sample['graph_3d'] = graph_3d

            if valid:
                self.data_list.append(sample)

            if (idx + 1) % 500 == 0:
                print(f"   Progress: {idx + 1}/{len(smiles_list)} "
                      f"({len(self.data_list) / (idx + 1) * 100:.1f}% success)")

        print(f"Done: {len(self.data_list)}/{len(smiles_list)} valid samples")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx].copy()

        # [v4.4] 1D 特征高斯噪声增强
        if (self._training and 'mol2vec' in sample
                and self.mol2vec_noise_std > 0):
            feat = sample['mol2vec'].clone()
            feat = feat + torch.randn_like(feat) * self.mol2vec_noise_std
            sample['mol2vec'] = feat

        # 3D 构象增强
        if self.augment_3d and 'graph_3d' in sample and self._training:
            g3d = sample['graph_3d']
            augmented_pos = ConformerAugmentation.augment(g3d.pos.clone())
            sample['graph_3d'] = Data(
                x=g3d.x, z=g3d.z, pos=augmented_pos,
                edge_index=g3d.edge_index, edge_weight=g3d.edge_weight
            )
        return sample

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, mode: bool):
        self._training = mode


def collate_multimodal(batch):
    batch = [s for s in batch if s is not None]
    if len(batch) == 0:
        return None

    result = {}

    if 'mol2vec' in batch[0] and batch[0]['mol2vec'] is not None:
        result['mol2vec'] = torch.stack(
            [torch.as_tensor(s['mol2vec'], dtype=torch.float) for s in batch])

    if 'graph_2d' in batch[0] and batch[0]['graph_2d'] is not None:
        g2d_list = []
        for s in batch:
            g = s['graph_2d']
            if isinstance(g, tuple):
                g = g[0]
            if isinstance(g, Data):
                g2d_list.append(g)
        if g2d_list:
            result['graph_2d'] = Batch.from_data_list(g2d_list)

    if 'graph_3d' in batch[0] and batch[0]['graph_3d'] is not None:
        g3d_list = []
        for s in batch:
            g = s['graph_3d']
            if isinstance(g, tuple):
                g = g[0]
            if isinstance(g, Data) and hasattr(g, 'pos'):
                g3d_list.append(g)
        if g3d_list:
            result['graph_3d'] = Batch.from_data_list(g3d_list)

    y_list = []
    for s in batch:
        for key in ['label', 'y', 'labels']:
            if key in s:
                y_list.append(s[key])
                break
    if y_list:
        result['labels'] = torch.tensor(y_list, dtype=torch.long)
    else:
        result['labels'] = torch.zeros(len(batch), dtype=torch.long)

    return result


# ============================================================================
# Part 12: 训练器
# ============================================================================
# balance_loss 软边界惩罚策略：
#   dominance_penalty = ReLU(w - upper).sum()  → 惩罚任意模态超过上界
#   neglect_penalty   = ReLU(lower - w).sum()  → 惩罚任意模态低于下界
#   → 在 [lower, upper] 区间内完全自由，attention 保留动态性
#
# [v4.4 新增] SWA（随机权重平均）：
#   - epoch >= swa_start_epoch 时启用 SWA
#   - SWA 与 EMA 并行，互不干扰
#   - 训练结束前对 SWA 模型做一次 BN 统计更新
#   - 使用 SWA 模型做最终 test set 评估
# ============================================================================

class Trainer:
    def __init__(self, config, device, exp_dir):
        self.config  = config
        self.device  = device
        self.exp_dir = exp_dir

    def train(self, model, train_loader, val_loader, test_loader):
        fusion_mode = self.config.get('fusion_mode', '1D+2D+3D')
        fusion_type = self.config.get('fusion_type', 'self_attention')

        print(f"\n{'=' * 70}")
        print(f"Training: mode={fusion_mode}, fusion={fusion_type}")
        print(f"{'=' * 70}")

        model = model.to(self.device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {total_params:,}")

        ema = EMA(model, decay=self.config.get('ema_decay', 0.9998))

        # ── [v4.4] SWA 初始化 ────────────────────────────────────────────
        swa_start_epoch = self.config.get('swa_start_epoch', 300)
        use_swa         = self.config.get('use_swa', True)
        swa_model       = AveragedModel(model) if use_swa else None
        swa_activated   = False
        print(f"   SWA: {'enabled' if use_swa else 'disabled'}"
              + (f" (start epoch={swa_start_epoch})" if use_swa else ""))

        base_lr       = self.config.get('learning_rate', 0.00020)
        branch_scales = self.config.get('branch_lr_scale', {})
        param_groups  = model.get_parameter_groups(base_lr, branch_scales)

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.get('weight_decay', 0.028))

        epochs    = self.config.get('epochs', 550)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[pg['lr'] for pg in param_groups],
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=self.config.get('warmup_pct', 0.25),
            anneal_strategy='cos'
        )

        # SWA LR scheduler（线性衰减，从 swa_start 开始）
        swa_scheduler = SWALR(
            optimizer,
            swa_lr=base_lr * 0.5,
            anneal_epochs=50,
            anneal_strategy='cos'
        ) if use_swa else None

        all_labels = np.array([s['label'] for s in train_loader.dataset.data_list])
        pos_weight = (all_labels == 0).sum() / max((all_labels == 1).sum(), 1)
        class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float).to(self.device)

        if self.config.get('use_focal_loss', True):
            criterion = FocalLoss(
                alpha=0.25, gamma=2.0, weight=class_weights,
                label_smoothing=self.config.get('label_smoothing', 0.18))
        else:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=self.config.get('label_smoothing', 0.18))

        balance_loss_weight = self.config.get('balance_loss_weight', 0.08)
        w_upper = self.config.get('balance_upper', 0.55)
        w_lower = self.config.get('balance_lower', 0.18)

        print(f"   Class weights: [1.0, {pos_weight:.2f}]")
        print(f"   Balance loss: soft-boundary penalty "
              f"(weight={balance_loss_weight}, "
              f"allowed=[{w_lower:.2f}, {w_upper:.2f}])")

        best_val_auc  = 0
        best_state    = None
        patience_cnt  = 0
        history       = defaultdict(list)
        patience      = self.config.get('patience', 160)
        start_time    = time.time()

        for epoch in range(epochs):
            train_loader.dataset.training = True

            # ── 训练 ──────────────────────────────────────────────────
            model.train()
            total_loss         = 0
            total_task_loss    = 0
            total_balance_loss = 0
            all_weights        = []

            for batch in train_loader:
                if batch is None:
                    continue
                optimizer.zero_grad()

                kwargs = {}
                if 'mol2vec' in batch:
                    kwargs['mol2vec_feat'] = batch['mol2vec'].to(self.device)
                if 'graph_2d' in batch:
                    kwargs['graph_2d'] = batch['graph_2d'].to(self.device)
                if 'graph_3d' in batch:
                    kwargs['graph_3d'] = batch['graph_3d'].to(self.device)

                labels = batch['labels'].to(self.device)
                result = model(**kwargs)

                # ── 任务损失 ────────────────────────────────────────
                task_loss = criterion(result['logits'], labels)

                # ── 软边界均衡损失 ──────────────────────────────────
                balance_loss = torch.tensor(0.0, device=self.device)
                if (balance_loss_weight > 0
                        and result['weights'] is not None
                        and result['weights'].shape[-1] == 3):
                    w = result['weights']      # [B, 3]
                    w_mean = w.mean(dim=0)     # [3]

                    dominance_penalty = F.relu(w_mean - w_upper).sum()
                    neglect_penalty   = F.relu(w_lower - w_mean).sum()
                    balance_loss = dominance_penalty + neglect_penalty

                loss = task_loss + balance_loss_weight * balance_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # [v4.4] SWA 激活后切换 scheduler
                if use_swa and (epoch + 1) >= swa_start_epoch:
                    if not swa_activated:
                        swa_activated = True
                        print(f"   [SWA] Activated at epoch {epoch + 1}")
                    swa_model.update_parameters(model)
                else:
                    scheduler.step()

                ema.update()

                total_loss         += loss.item()
                total_task_loss    += task_loss.item()
                total_balance_loss += balance_loss.item()

                if result['weights'] is not None:
                    all_weights.append(
                        result['weights'].mean(dim=0).detach().cpu().numpy())

            # SWA scheduler step（每 epoch）
            if use_swa and swa_activated and swa_scheduler is not None:
                swa_scheduler.step()

            avg_loss         = total_loss / max(len(train_loader), 1)
            avg_task_loss    = total_task_loss / max(len(train_loader), 1)
            avg_balance_loss = total_balance_loss / max(len(train_loader), 1)
            avg_weights      = (np.mean(all_weights, axis=0).tolist()
                                if all_weights else [])

            # ── 评估（EMA 参数）────────────────────────────────────────
            ema.apply_shadow()
            try:
                train_probs, train_labels_arr = self._evaluate(model, train_loader)
                val_probs, val_labels_arr     = self._evaluate(model, val_loader)
            finally:
                ema.restore()

            train_auc = roc_auc_score(train_labels_arr, train_probs)
            val_auc   = roc_auc_score(val_labels_arr,   val_probs)

            history['train_auc'].append(train_auc)
            history['val_auc'].append(val_auc)
            history['loss'].append(avg_loss)
            history['task_loss'].append(avg_task_loss)
            history['balance_loss'].append(avg_balance_loss)
            history['modality_weights'].append(avg_weights)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                w_str = (', '.join([f'{w:.3f}' for w in avg_weights])
                         if avg_weights else 'N/A')
                gap = train_auc - val_auc
                swa_tag = ' [SWA]' if swa_activated else ''
                print(f"   Epoch {epoch + 1:3d}/{epochs} | "
                      f"Loss: {avg_task_loss:.4f}+{avg_balance_loss:.4f} | "
                      f"Train: {train_auc:.4f} | Val: {val_auc:.4f} | "
                      f"Gap: {gap:.4f} | W: [{w_str}]{swa_tag}")

                # ── Epoch 50 诊断 ────────────────────────────────────
                if (epoch + 1) == 50 and avg_weights:
                    w1d = avg_weights[0]
                    w2d = avg_weights[1] if len(avg_weights) > 1 else 0
                    w3d = avg_weights[2] if len(avg_weights) > 2 else 0
                    weight_std = np.std(avg_weights)

                    print(f"   [INFO] Epoch50: "
                          f"1D={w1d:.3f}, 2D={w2d:.3f}, 3D={w3d:.3f}, "
                          f"std={weight_std:.4f}")

                    if weight_std < 0.02:
                        print(f"   [WARN] weight_std={weight_std:.4f}<0.02，"
                              f"attention 仍被过度约束，"
                              f"建议将 balance_loss_weight 降至 0.05")
                    if w1d > 0.50:
                        print(f"   [WARN] 1D 权重{w1d:.3f}>0.50，"
                              f"建议将 encoder_1d scale 降至 0.08")
                    if w2d < 0.18:
                        print(f"   [WARN] 2D 权重{w2d:.3f}<0.18，"
                              f"建议将 encoder_2d scale 升至 3.0")
                    if (w_lower <= w1d <= w_upper
                            and w_lower <= w2d <= w_upper
                            and w_lower <= w3d <= w_upper
                            and weight_std > 0.02):
                        print(f"   [OK] 所有模态权重均在"
                              f"[{w_lower:.2f},{w_upper:.2f}]内，"
                              f"attention 动态性正常 ✓")

                # ── Epoch 100 诊断 ───────────────────────────────────
                if (epoch + 1) == 100:
                    gap = train_auc - val_auc
                    if gap > 0.05:
                        print(f"   [WARN] Epoch100: gap={gap:.4f}>0.05，"
                              f"建议 dropout→0.50，weight_decay→0.032")
                    elif gap < 0:
                        print(f"   [WARN] Epoch100: gap={gap:.4f}<0（仍欠拟合），"
                              f"建议 warmup_pct→0.20，dropout→0.40")
                    if avg_weights and len(avg_weights) > 2 and avg_weights[2] < 0.10:
                        print(f"   [WARN] Epoch100: 3D 权重{avg_weights[2]:.3f}<0.10，"
                              f"建议将 spherenet_dropout 降至 0.25")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                ema.apply_shadow()
                try:
                    best_state = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}
                finally:
                    ema.restore()
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"   Early stopping at epoch {epoch + 1}")
                    break

        train_time = time.time() - start_time
        print(f"   Training time: {train_time:.1f}s")
        print(f"   Best Val AUC: {best_val_auc:.4f}")

        # ── [v4.4] SWA BN 统计更新（自定义，兼容 dict DataLoader）─────
        if use_swa and swa_activated and swa_model is not None:
            print("   [SWA] Updating BN statistics...")
            self._update_bn_for_swa(swa_model, train_loader)
            print("   [SWA] BN update done.")

        # 恢复 EMA best 权重
        model.load_state_dict({k: v.to(self.device)
                               for k, v in best_state.items()})

        results = self._final_evaluation(
            model, train_loader, val_loader, test_loader,
            train_time, dict(history), fusion_mode, fusion_type,
            swa_model=swa_model if use_swa and swa_activated else None)

        model_path = self.exp_dir / "models" / f"{fusion_mode}_{fusion_type}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, model_path)

        # [v4.4] 也保存 SWA 权重
        if use_swa and swa_activated and swa_model is not None:
            swa_path = self.exp_dir / "models" / f"{fusion_mode}_{fusion_type}_swa.pt"
            torch.save(swa_model.state_dict(), swa_path)
            print(f"   [SWA] Model saved → {swa_path.name}")

        return results

    def _update_bn_for_swa(self, swa_model, loader):
        """
        [v4.4] 自定义 SWA BN 统计更新。
        PyTorch 内置 update_bn 要求 DataLoader yield tensor，
        但本项目 yield dict，因此手动实现等价逻辑。
        """
        # 将所有 BN 层的 running_mean/var 归零，num_batches_tracked 置 0
        for module in swa_model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.reset_running_stats()
                module.momentum = None        # cumulative moving average 模式

        swa_model.train()
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                kwargs = {}
                if 'mol2vec' in batch:
                    kwargs['mol2vec_feat'] = batch['mol2vec'].to(self.device)
                if 'graph_2d' in batch:
                    kwargs['graph_2d'] = batch['graph_2d'].to(self.device)
                if 'graph_3d' in batch:
                    kwargs['graph_3d'] = batch['graph_3d'].to(self.device)
                try:
                    swa_model(**kwargs)
                except Exception:
                    pass

        # 恢复默认 momentum
        for module in swa_model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.momentum = 0.1

    def _evaluate(self, model, loader):
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                kwargs = {}
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

    def _evaluate_swa(self, swa_model, loader):
        """[v4.4] 对 SWA 模型做推理（不依赖 forward 的 weights 字段）"""
        swa_model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                kwargs = {}
                if 'mol2vec' in batch:
                    kwargs['mol2vec_feat'] = batch['mol2vec'].to(self.device)
                if 'graph_2d' in batch:
                    kwargs['graph_2d'] = batch['graph_2d'].to(self.device)
                if 'graph_3d' in batch:
                    kwargs['graph_3d'] = batch['graph_3d'].to(self.device)

                result = swa_model(**kwargs)
                probs  = F.softmax(result['logits'], dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(batch['labels'].numpy().tolist())
        return np.array(all_probs), np.array(all_labels)

    def _final_evaluation(self, model, train_loader, val_loader, test_loader,
                          train_time, history, fusion_mode, fusion_type,
                          swa_model=None):
        train_probs, train_labels = self._evaluate(model, train_loader)
        val_probs,   val_labels   = self._evaluate(model, val_loader)
        test_probs,  test_labels  = self._evaluate(model, test_loader)

        best_f1, best_thresh = 0, 0.5
        for thresh in np.arange(0.3, 0.7, 0.01):
            f1 = f1_score(val_labels, (val_probs >= thresh).astype(int),
                          zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh

        def compute_metrics(y_true, y_proba, threshold):
            y_pred = (y_proba >= threshold).astype(int)
            return {
                'roc_auc':   roc_auc_score(y_true, y_proba),
                'pr_auc':    average_precision_score(y_true, y_proba),
                'accuracy':  accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall':    recall_score(y_true, y_pred, zero_division=0),
                'f1':        f1_score(y_true, y_pred, zero_division=0),
                'mcc':       matthews_corrcoef(y_true, y_pred)
            }

        train_metrics = compute_metrics(train_labels, train_probs, best_thresh)
        val_metrics   = compute_metrics(val_labels,   val_probs,   best_thresh)
        test_metrics  = compute_metrics(test_labels,  test_probs,  best_thresh)

        print(f"\n   Final Results (threshold={best_thresh:.2f}):")
        header = (f"       {'Set':<12} {'AUC':>8} {'ACC':>8} "
                  f"{'Prec':>8} {'Recall':>8} {'F1':>8} {'MCC':>8}")
        print(header)
        for name, m in [('Train', train_metrics),
                        ('Val',   val_metrics),
                        ('Test',  test_metrics)]:
            print(f"       {name:<12} {m['roc_auc']:>8.4f} {m['accuracy']:>8.4f} "
                  f"{m['precision']:>8.4f} {m['recall']:>8.4f} "
                  f"{m['f1']:>8.4f} {m['mcc']:>8.4f}")

        # [v4.4] SWA 模型额外评估
        swa_test_metrics = None
        if swa_model is not None:
            try:
                swa_test_probs, swa_test_labels = self._evaluate_swa(
                    swa_model, test_loader)
                swa_test_metrics = compute_metrics(
                    swa_test_labels, swa_test_probs, best_thresh)
                m = swa_test_metrics
                print(f"       {'Test(SWA)':<12} {m['roc_auc']:>8.4f} "
                      f"{m['accuracy']:>8.4f} {m['precision']:>8.4f} "
                      f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['mcc']:>8.4f}")
            except Exception as e:
                print(f"   [WARN] SWA evaluation failed: {e}")

        return {
            'model_name':        f'{fusion_mode}_{fusion_type}',
            'fusion_mode':       fusion_mode,
            'fusion_type':       fusion_type,
            'train_metrics':     train_metrics,
            'val_metrics':       val_metrics,
            'test_metrics':      test_metrics,
            'swa_test_metrics':  swa_test_metrics,
            'training_time':     train_time,
            'optimal_threshold': best_thresh,
            'predictions': {
                'train_proba':  train_probs,
                'val_proba':    val_probs,
                'test_proba':   test_probs,
                'train_labels': train_labels,
                'val_labels':   val_labels,
                'test_labels':  test_labels,
            },
            'history': history
        }


# ============================================================================
# Part 13: Mol2Vec
# ============================================================================

class Mol2VecFeaturizer:
    def __init__(self, model_path=None, embedding_dim=300, radius=1):
        if not MOL2VEC_AVAILABLE:
            raise ImportError("mol2vec not installed! pip install mol2vec")
        self.radius = radius
        self.embedding_dim = embedding_dim
        self.model = None
        if model_path and Path(model_path).exists():
            self.model = word2vec.Word2Vec.load(model_path)
            self.embedding_dim = self.model.wv.vector_size

    def _mol_to_sentence(self, mol):
        if mol is None:
            return None
        try:
            return mol2alt_sentence(mol, self.radius)
        except:
            return None

    def train_on_corpus(self, smiles_list, **kwargs):
        print("Training Mol2vec...")
        sentences = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                sent = self._mol_to_sentence(mol)
                if sent:
                    sentences.append(sent)
        params = {'vector_size': self.embedding_dim, 'window': 5,
                  'min_count': 1, 'sg': 1, 'workers': 4, 'epochs': 10}
        params.update(kwargs)
        self.model = word2vec.Word2Vec(sentences, **params)
        print(f"   Trained, vocab: {len(self.model.wv)}")

    def featurize(self, smiles_list):
        features = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features.append(np.zeros(self.embedding_dim))
                continue
            sent = self._mol_to_sentence(mol)
            if not sent:
                features.append(np.zeros(self.embedding_dim))
                continue
            try:
                result = sentences2vec([sent], self.model, unseen='UNK')
                if isinstance(result, np.ndarray) and result.shape[0] > 0:
                    features.append(result[0])
                else:
                    features.append(np.zeros(self.embedding_dim))
            except:
                vecs = [self.model.wv[w] for w in sent if w in self.model.wv]
                features.append(np.mean(vecs, axis=0)
                                 if vecs else np.zeros(self.embedding_dim))
        return np.array(features)


# ============================================================================
# Part 14: 可视化
# ============================================================================

class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_learning_curve(self, history, model_name):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        epochs = range(1, len(history['train_auc']) + 1)

        axes[0, 0].plot(epochs, history['train_auc'], 'b-', label='Train', lw=2)
        axes[0, 0].plot(epochs, history['val_auc'],   'r-', label='Val',   lw=2)
        axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('AUC')
        axes[0, 0].set_title('AUC Curve'); axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(epochs, history['loss'], 'g-', lw=2, label='Total')
        if 'task_loss' in history:
            axes[0, 1].plot(epochs, history['task_loss'], 'b--',
                            lw=1.5, label='Task', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Loss Curve'); axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        if 'balance_loss' in history and history['balance_loss']:
            axes[0, 2].plot(epochs, history['balance_loss'], color='purple', lw=2)
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Balance Loss (soft-boundary)')
            axes[0, 2].set_title('Modality Balance Loss\n(0 = all weights in bounds)')
            axes[0, 2].grid(alpha=0.3)

        weights_data = history.get('modality_weights', [])
        has_weights  = (weights_data and len(weights_data) > 0
                        and len(weights_data[0]) > 0)
        if has_weights:
            weights = np.array(weights_data)
            n       = weights.shape[1]
            labels  = ['1D', '2D', '3D'][:n]
            colors  = ['blue', 'green', 'red'][:n]
            for i in range(n):
                axes[1, 0].plot(epochs, weights[:, i], label=labels[i],
                                color=colors[i], lw=2)
            axes[1, 0].axhline(y=0.55, color='orange', linestyle='--',
                               alpha=0.6, label='上界(0.55)')
            axes[1, 0].axhline(y=0.18, color='orange', linestyle=':',
                               alpha=0.6, label='下界(0.18)')
            axes[1, 0].axhline(y=1/3,  color='gray',   linestyle='--',
                               alpha=0.3, label='均匀(0.333)')
            axes[1, 0].fill_between(range(1, len(epochs)+1),
                                    0.18, 0.55, alpha=0.05, color='green',
                                    label='允许动态区间')
            axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Weight')
            axes[1, 0].set_title('[v4.4] Modality Weights\n(动态分配，非强制均匀)')
            axes[1, 0].legend(fontsize=7); axes[1, 0].grid(alpha=0.3)
            axes[1, 0].set_ylim(0, 0.85)

        gaps = np.array(history['train_auc']) - np.array(history['val_auc'])
        axes[1, 1].plot(epochs, gaps, 'orange', lw=2)
        axes[1, 1].axhline(y=0,    color='gray',  linestyle='--')
        axes[1, 1].axhline(y=0.04, color='green', linestyle='--',
                           alpha=0.7, label='目标 gap=0.04')
        axes[1, 1].fill_between(epochs, 0, gaps, alpha=0.3, color='orange')
        axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Gap')
        axes[1, 1].set_title('Overfitting Gap (目标:<0.04)')
        axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

        val_auc = np.array(history['val_auc'])
        window = min(20, len(val_auc) // 5)
        if window > 1:
            val_smooth = np.convolve(val_auc, np.ones(window)/window, mode='valid')
            axes[1, 2].plot(epochs, val_auc, 'r-', alpha=0.4, lw=1, label='Val AUC')
            axes[1, 2].plot(range(window, len(val_auc)+1), val_smooth,
                            'r-', lw=2.5, label=f'MA({window})')
        else:
            axes[1, 2].plot(epochs, val_auc, 'r-', lw=2, label='Val AUC')
        axes[1, 2].set_xlabel('Epoch'); axes[1, 2].set_ylabel('Val AUC')
        axes[1, 2].set_title('Val AUC (Smoothed)')
        axes[1, 2].legend(); axes[1, 2].grid(alpha=0.3)

        plt.suptitle(f'Training Curves: {model_name}', fontsize=14, y=1.01)
        plt.tight_layout()
        safe_name = model_name.replace('+', '_').replace(' ', '_')
        plt.savefig(self.save_dir / f"learning_{safe_name}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_curves(self, all_results):
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        for idx, r in enumerate(all_results):
            y_test  = r['predictions']['test_labels']
            y_proba = r['predictions']['test_proba']
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc   = r['test_metrics']['roc_auc']
            label = r['model_name']
            plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})",
                     lw=2, color=colors[idx])

            # [v4.4] SWA 曲线
            if r.get('swa_test_metrics') is not None:
                swa_auc = r['swa_test_metrics']['roc_auc']
                plt.plot(fpr, tpr, linestyle=':', lw=1.5, color=colors[idx],
                         label=f"{label}_SWA (AUC={swa_auc:.3f})", alpha=0.7)

        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel('FPR', fontsize=12); plt.ylabel('TPR', fontsize=12)
        plt.title('ROC Curves', fontsize=14)
        plt.legend(loc='lower right', fontsize=9); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / "roc_curves_all.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_weight_distribution(self, all_results):
        fusion_results = [r for r in all_results
                          if r['fusion_mode'] == '1D+2D+3D']
        if not fusion_results:
            return

        fig, axes = plt.subplots(1, len(fusion_results),
                                 figsize=(6 * len(fusion_results), 5))
        if len(fusion_results) == 1:
            axes = [axes]

        for ax, r in zip(axes, fusion_results):
            weights_hist = r['history'].get('modality_weights', [])
            if not weights_hist or not weights_hist[-1]:
                continue
            last_n = min(50, len(weights_hist))
            recent = np.array(weights_hist[-last_n:])
            means  = recent.mean(axis=0)
            stds   = recent.std(axis=0)
            bars   = ax.bar(['1D', '2D', '3D'], means,
                            color=['blue', 'green', 'red'], alpha=0.7,
                            yerr=stds, capsize=5)
            ax.axhline(y=0.55, color='orange', linestyle='--',
                       alpha=0.8, label='上界(0.55)')
            ax.axhline(y=0.18, color='orange', linestyle=':',
                       alpha=0.8, label='下界(0.18)')
            ax.axhline(y=1/3,  color='gray',   linestyle='--',
                       alpha=0.4, label='均匀(0.333)')
            ax.fill_between([-0.5, 2.5], 0.18, 0.55,
                            alpha=0.08, color='green')
            ax.set_ylim(0, 0.8)
            ax.set_ylabel('Attention Weight')
            weight_std = stds.mean()
            ax.set_title(f'{r["model_name"]}\n'
                         f'最后{last_n}epoch均值 | std={weight_std:.3f}\n'
                         f'(std>0.02 表明 attention 有动态性 ✓)')
            ax.legend(fontsize=8)
            for bar, mean, std in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.015,
                        f'{mean:.3f}±{std:.3f}', ha='center', va='bottom',
                        fontsize=9)

        plt.tight_layout()
        plt.savefig(self.save_dir / "weight_distribution.png",
                    dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================================
# Part 15: 配置保存工具
# ============================================================================

def save_config_to_exp_dir(config: dict, config_path: str,
                           exp_dir: Path, timestamp: str,
                           device: torch.device):
    config_copy_path = exp_dir / f"config_{timestamp}.yaml"
    shutil.copy2(config_path, config_copy_path)

    config_with_meta = copy.deepcopy(config)
    config_with_meta['_meta'] = {
        'config_source':  str(Path(config_path).resolve()),
        'experiment_dir': str(exp_dir.resolve()),
        'timestamp':      timestamp,
        'device':         str(device),
        'python_version': sys.version.split()[0],
        'torch_version':  torch.__version__,
        'version':        'v4.4',
    }
    meta_path = exp_dir / "config_with_meta.yaml"
    with open(meta_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_with_meta, f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)

    print(f"   Config saved → {config_copy_path.name}")
    print(f"   Meta config  → {meta_path.name}")
    return config_copy_path, meta_path


def print_config_snapshot(config: dict, exp_dir: Path, timestamp: str):
    print(f"\n{'=' * 80}")
    print("CONFIG SNAPSHOT")
    print(f"{'=' * 80}")
    print(f"   Config file  : {exp_dir / f'config_{timestamp}.yaml'}")
    print(f"   Full meta    : {exp_dir / 'config_with_meta.yaml'}")
    print(f"   {'─' * 50}")

    key_params = {
        'hidden_dim':           config.get('model', {}).get('hidden_dim'),
        'dropout':              config.get('model', {}).get('dropout'),
        'spherenet_dropout':    config.get('model', {}).get('spherenet_dropout'),
        'num_res_blocks':       config.get('model', {}).get('num_res_blocks'),
        'dmpnn_layers':         config.get('model', {}).get('dmpnn_layers'),
        'spherenet_layers':     config.get('model', {}).get('spherenet_layers'),
        'balance_loss_weight':  config.get('model', {}).get('balance_loss_weight'),
        'balance_upper':        config.get('model', {}).get('balance_upper'),
        'balance_lower':        config.get('model', {}).get('balance_lower'),
        'learning_rate':        config.get('training', {}).get('learning_rate'),
        'weight_decay':         config.get('training', {}).get('weight_decay'),
        'warmup_pct':           config.get('training', {}).get('warmup_pct'),
        'modality_drop_prob':   config.get('training', {}).get('modality_drop_prob'),
        'label_smoothing':      config.get('training', {}).get('label_smoothing'),
        'epochs':               config.get('training', {}).get('epochs'),
        'patience':             config.get('training', {}).get('patience'),
        'swa_start_epoch':      config.get('training', {}).get('swa_start_epoch'),
        'use_swa':              config.get('training', {}).get('use_swa'),
        'mol2vec_noise_std':    config.get('training', {}).get('mol2vec_noise_std'),
        'branch_lr_scale':      config.get('training', {}).get('branch_lr_scale'),
    }

    for k, v in key_params.items():
        if k == 'branch_lr_scale' and isinstance(v, dict):
            print(f"   {'branch_lr_scale':<25}:")
            for sk, sv in v.items():
                print(f"     {sk:<23}: {sv}")
        else:
            print(f"   {k:<25}: {v}")

    print(f"{'=' * 80}")


# ============================================================================
# Part 16: Main
# ============================================================================

def smart_read_csv(filepath):
    df = pd.read_csv(filepath)
    smiles_col = next((c for c in df.columns
                       if c.lower() in ['smiles', 'smiles_standardized',
                                        'canonical_smiles']), None)
    label_col = next((c for c in df.columns
                      if c.lower() in ['label', 'activity', 'target',
                                       'class', 'y']), None)
    if not smiles_col or not label_col:
        raise ValueError(f"Cannot find smiles/label columns in {filepath}. "
                         f"Columns: {list(df.columns)}")
    df = df.rename(columns={smiles_col: 'smiles', label_col: 'label'})
    df['label'] = df['label'].astype(int)
    return df[['smiles', 'label']]


def run_experiment(config_path):
    print("\n" + "=" * 80)
    print("MULTI-MODAL FUSION v4.4 - Optimized Edition")
    print("=" * 80)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name  = config.get('experiment', {}).get('name', 'MultiModal_v4_4')
    exp_dir   = (Path(config.get('output', {}).get('base_dir', 'results'))
                 / f"{exp_name}_{timestamp}")
    for subdir in ['data', 'models', 'figures', 'results']:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Output: {exp_dir}")
    print(f"Device: {device}")

    save_config_to_exp_dir(config, config_path, exp_dir, timestamp, device)

    train_df = smart_read_csv(Path(config['data']['train_path']))
    val_df   = smart_read_csv(Path(config['data']['val_path']))
    test_df  = smart_read_csv(Path(config['data']['test_path']))
    print(f"\nData: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    experiments = config.get('experiments', [
        {'fusion_mode': '2D',       'fusion_type': 'none'},
        {'fusion_mode': '1D+2D+3D', 'fusion_type': 'self_attention'},
    ])

    mol2vec_train = mol2vec_val = mol2vec_test = None
    need_1d = any('1D' in exp['fusion_mode'] for exp in experiments)
    if need_1d and MOL2VEC_AVAILABLE:
        featurizer = Mol2VecFeaturizer(embedding_dim=300)
        featurizer.train_on_corpus(
            train_df['smiles'].tolist() + val_df['smiles'].tolist())
        mol2vec_model_path = exp_dir / "mol2vec.model"
        featurizer.model.save(str(mol2vec_model_path))
        print(f"Mol2Vec model saved to {mol2vec_model_path}")

        mol2vec_train = featurizer.featurize(train_df['smiles'].tolist())
        mol2vec_val   = featurizer.featurize(val_df['smiles'].tolist())
        mol2vec_test  = featurizer.featurize(test_df['smiles'].tolist())

    all_results   = []
    dataset_cache = {}

    mol2vec_noise_std = config.get('training', {}).get('mol2vec_noise_std', 0.02)

    for exp in experiments:
        fusion_mode = exp['fusion_mode']
        fusion_type = exp['fusion_type']

        print(f"\n{'=' * 80}")
        print(f"Experiment: {fusion_mode} + {fusion_type}")
        print(f"{'=' * 80}")

        if fusion_mode not in dataset_cache:
            cutoff = config.get('model', {}).get('cutoff', 10.0)
            train_dataset = MultiModalDataset(
                train_df['smiles'].tolist(), train_df['label'].values,
                mol2vec_train, fusion_mode, cutoff,
                augment_3d=True,
                mol2vec_noise_std=mol2vec_noise_std)     # [v4.4]
            val_dataset = MultiModalDataset(
                val_df['smiles'].tolist(), val_df['label'].values,
                mol2vec_val, fusion_mode, cutoff,
                mol2vec_noise_std=0.0)                   # 验证/测试不加噪声
            test_dataset = MultiModalDataset(
                test_df['smiles'].tolist(), test_df['label'].values,
                mol2vec_test, fusion_mode, cutoff,
                mol2vec_noise_std=0.0)
            dataset_cache[fusion_mode] = (train_dataset, val_dataset, test_dataset)
        else:
            train_dataset, val_dataset, test_dataset = dataset_cache[fusion_mode]

        batch_size   = config.get('training', {}).get('batch_size', 64)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_multimodal,
                                  num_workers=0, drop_last=True)
        val_loader   = DataLoader(val_dataset,  batch_size=batch_size,
                                  collate_fn=collate_multimodal)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                                  collate_fn=collate_multimodal)

        model_config = config.get('model', {}).copy()
        model_config.update({
            'fusion_mode': fusion_mode,
            'fusion_type': fusion_type,
            'mol2vec_dim': 300,
            'atom_dim':    47,
            'edge_dim':    14,
        })

        model = MultiModalFusionNet(model_config)

        train_config = config.get('training', {}).copy()
        train_config.update({
            'fusion_mode':         fusion_mode,
            'fusion_type':         fusion_type,
            'balance_loss_weight': model_config.get('balance_loss_weight', 0.08),
            'balance_upper':       model_config.get('balance_upper', 0.55),
            'balance_lower':       model_config.get('balance_lower', 0.18),
        })

        trainer = Trainer(train_config, device, exp_dir)
        result  = trainer.train(model, train_loader, val_loader, test_loader)
        all_results.append(result)

    # ── 消融实验 ────────────────────────────────────────────────────────
    sa_results = [r for r in all_results if r['fusion_type'] == 'self_attention']
    if sa_results and '1D+2D+3D' in dataset_cache:
        print(f"\n{'=' * 80}")
        print("Ablation Experiments")
        print(f"{'=' * 80}")

        model_config = config.get('model', {}).copy()
        model_config.update({
            'fusion_mode': '1D+2D+3D',
            'fusion_type': 'self_attention',
            'mol2vec_dim': 300,
            'atom_dim':    47,
            'edge_dim':    14,
        })
        ablation_model = MultiModalFusionNet(model_config).to(device)
        model_path = exp_dir / "models" / "1D+2D+3D_self_attention.pt"

        if model_path.exists():
            state = torch.load(model_path, map_location=device)
            ablation_model.load_state_dict(state)
            ablation_model.eval()

            _, _, test_dataset = dataset_cache['1D+2D+3D']
            test_loader = DataLoader(test_dataset, batch_size=64,
                                     collate_fn=collate_multimodal)

            ablation_configs = [
                ('w/o 1D', {'1d': False, '2d': True,  '3d': True}),
                ('w/o 2D', {'1d': True,  '2d': False, '3d': True}),
                ('w/o 3D', {'1d': True,  '2d': True,  '3d': False}),
            ]

            print(f"\n       {'Config':<12} {'AUC':>8} {'F1':>8} {'MCC':>8}")
            for name, mask in ablation_configs:
                all_probs, all_labels = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        if batch is None:
                            continue
                        kwargs = {}
                        if 'mol2vec' in batch:
                            kwargs['mol2vec_feat'] = batch['mol2vec'].to(device)
                        if 'graph_2d' in batch:
                            kwargs['graph_2d'] = batch['graph_2d'].to(device)
                        if 'graph_3d' in batch:
                            kwargs['graph_3d'] = batch['graph_3d'].to(device)
                        kwargs['ablation_mask'] = mask

                        result = ablation_model(**kwargs)
                        probs  = F.softmax(result['logits'], dim=1)[:, 1].cpu().numpy()
                        all_probs.extend(probs.tolist())
                        all_labels.extend(batch['labels'].numpy().tolist())

                y_true  = np.array(all_labels)
                y_proba = np.array(all_probs)
                y_pred  = (y_proba >= 0.5).astype(int)
                auc = roc_auc_score(y_true, y_proba)
                f1  = f1_score(y_true, y_pred, zero_division=0)
                mcc = matthews_corrcoef(y_true, y_pred)
                print(f"       {name:<12} {auc:>8.4f} {f1:>8.4f} {mcc:>8.4f}")

    # ── 汇总 & 可视化 ────────────────────────────────────────────────────
    if all_results:
        visualizer = Visualizer(exp_dir / 'figures')
        for r in all_results:
            visualizer.plot_learning_curve(r['history'], r['model_name'])
        visualizer.plot_roc_curves(all_results)
        visualizer.plot_weight_distribution(all_results)

        data = []
        for r in all_results:
            row = {
                'Mode':      r['fusion_mode'],
                'Fusion':    r['fusion_type'],
                'ROC-AUC':   r['test_metrics']['roc_auc'],
                'Accuracy':  r['test_metrics']['accuracy'],
                'Precision': r['test_metrics']['precision'],
                'Recall':    r['test_metrics']['recall'],
                'F1':        r['test_metrics']['f1'],
                'MCC':       r['test_metrics']['mcc']
            }
            if r.get('swa_test_metrics'):
                row['SWA-AUC'] = r['swa_test_metrics']['roc_auc']
                row['SWA-F1']  = r['swa_test_metrics']['f1']
            data.append(row)

        df = pd.DataFrame(data).sort_values('ROC-AUC', ascending=False)
        df.to_csv(exp_dir / 'results' / 'summary.csv', index=False)

        print(f"\n{'=' * 80}")
        print(df.to_string(index=False))
        print(f"{'=' * 80}")

        best = max(all_results, key=lambda x: x['test_metrics']['roc_auc'])
        print(f"\nBEST: {best['model_name']}")
        print(f"   ROC-AUC: {best['test_metrics']['roc_auc']:.4f}")
        print(f"   F1:      {best['test_metrics']['f1']:.4f}")
        if best.get('swa_test_metrics'):
            print(f"   SWA ROC-AUC: {best['swa_test_metrics']['roc_auc']:.4f}")
            print(f"   SWA F1:      {best['swa_test_metrics']['f1']:.4f}")

    print_config_snapshot(config, exp_dir, timestamp)

    print(f"\nCOMPLETED! Results: {exp_dir}")
    return exp_dir, all_results


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Modal Molecular Fusion Network v4.4')
    parser.add_argument('--config', '-c', default='config_v4_4.yaml')
    args = parser.parse_args()

    if not Path(args.config).exists():
        default_config = {
            'experiment': {'name': 'NLRP3_v4_4_Optimized'},
            'data': {
                'train_path': 'data/processed/new/balanced_split/train.csv',
                'val_path':   'data/processed/new/balanced_split/val.csv',
                'test_path':  'data/processed/new/balanced_split/test.csv'
            },
            'output': {'base_dir': 'results/tuned_v4_4'},
            'experiments': [
                {'fusion_mode': '2D',       'fusion_type': 'none'},
                {'fusion_mode': '1D+2D+3D', 'fusion_type': 'self_attention'},
            ],
            'model': {
                'hidden_dim':           128,
                'atom_dim':             47,
                'edge_dim':             14,
                # [v4.4] 正则加强
                'dropout':              0.48,
                'spherenet_dropout':    0.35,
                'mol2vec_dim':          300,
                'num_res_blocks':       1,
                'dmpnn_layers':         5,
                'spherenet_layers':     8,
                'cutoff':               10.0,
                'use_virtual_node':     True,
                'fusion_num_heads':     8,
                'fusion_num_layers':    2,
                # [v4.4] balance loss 进一步降低
                'balance_loss_weight':  0.08,
                'balance_upper':        0.55,
                'balance_lower':        0.18,
            },
            'training': {
                'batch_size':           64,
                # [v4.4] 缩短训练，防止无效过拟合
                'epochs':               550,
                'patience':             160,
                # [v4.4] 正则加强
                'learning_rate':        0.00020,
                'weight_decay':         0.028,
                'warmup_pct':           0.25,
                'modality_drop_prob':   0.48,
                'label_smoothing':      0.18,
                # [v4.4] SWA
                'use_swa':              True,
                'swa_start_epoch':      300,
                # [v4.4] 1D 噪声增强
                'mol2vec_noise_std':    0.02,
                'use_focal_loss':       True,
                'ema_decay':            0.9998,
                'branch_lr_scale': {
                    'encoder_1d':  0.10,
                    'encoder_2d':  2.5,
                    'encoder_3d':  0.80,
                    'fusion':      0.8,
                    'classifier':  1.0
                }
            }
        }
        with open(args.config, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False,
                      allow_unicode=True, sort_keys=False)
        print(f"Created default config: {args.config}")
        print("Please update data paths and run again.")
        return

    run_experiment(args.config)


if __name__ == "__main__":
    main()