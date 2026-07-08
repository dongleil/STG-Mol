#!/usr/bin/env python3
"""
two_stage_screening.py
======================
两阶段虚拟筛选完整实现

架构：
  Stage 0 : Lipinski / ADMET 预过滤           (RDKit, 极快)
  Stage 1 : 1D+2D 快速粗筛                    (无需3D构象, 快)
  Stage 2 : 1D+2D+3D 融合模型精筛             (完整融合模型, 准)
  Post    : 多样性去冗余 + 结果导出

依赖：
  pip install rdkit torch torch_geometric pandas numpy joblib tqdm pyyaml
  (mol2vec / gensim 可选，若 fusion_mode 包含 1D)

用法：
  python two_stage_screening.py --config screening_config.yaml
  python two_stage_screening.py --database zinc.csv \
         --stage1_model results/1D+2D_none.pt \
         --stage2_model results/1D+2D+3D_self_attention.pt \
         --top_k_ratio 0.05 --output screening_out
"""

# ============================================================
# 标准库
# ============================================================
import os
import sys
import time
import math
import json
import pickle
import hashlib
import argparse
import warnings
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict

# ============================================================
# 第三方库
# ============================================================
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch
from torch_geometric.nn import (global_mean_pool, global_max_pool,
                                 global_add_pool, radius_graph)

from rdkit import Chem
from rdkit.Chem import (AllChem, Descriptors, rdMolDescriptors,
                         QED, DataStructs,Crippen)
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

from sklearn.metrics import roc_auc_score, f1_score

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("[WARN] joblib not available, parallel conformer gen disabled")

try:
    from mol2vec.features import mol2alt_sentence, sentences2vec
    from gensim.models import word2vec as gensim_w2v
    MOL2VEC_AVAILABLE = True
except ImportError:
    MOL2VEC_AVAILABLE = False

warnings.filterwarnings('ignore')

# ============================================================
# 日志
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


# ============================================================================
# Part 1: 原子/键特征（与训练代码保持完全一致）
# ============================================================================

def generate_atom_features_extended(atom, mol=None) -> List[float]:
    """47 维原子特征，必须与训练时完全一致"""
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
        features.append(float(atom.GetTotalNumHs() > 0 and symbol in ['N', 'O']))
        features.append(float(symbol in ['N', 'O', 'F']))
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
    """14 维键特征"""
    features = []
    try:
        bt = bond.GetBondType()
        features.extend([
            float(bt == Chem.BondType.SINGLE), float(bt == Chem.BondType.DOUBLE),
            float(bt == Chem.BondType.TRIPLE), float(bt == Chem.BondType.AROMATIC)
        ])
        features.append(float(bond.GetIsConjugated()))
        features.append(float(bond.IsInRing()))
        st = bond.GetStereo()
        features.extend([
            float(st == Chem.BondStereo.STEREOZ), float(st == Chem.BondStereo.STEREOE),
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
# Part 2: 3D 构象生成器（与训练代码一致）
# ============================================================================

class EnhancedConformerGenerator:
    def __init__(self, num_confs: int = 5, optimize: bool = True,
                 max_iters: int = 300, select_by_energy: bool = True):
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
                    mol, mmffVariant='MMFF94s', maxIters=self.max_iters, numThreads=0)
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


# ============================================================================
# Part 3: 图构建函数
# ============================================================================

def mol_to_2d_graph(smiles: str) -> Optional[Data]:
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


def mol_to_3d_graph(smiles: str, cutoff: float = 8.0,
                    conformer_gen=None) -> Tuple[Optional[Data], str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid_smiles"
    if conformer_gen is None:
        conformer_gen = EnhancedConformerGenerator()
    mol_3d = conformer_gen.generate(mol)
    if mol_3d is None or mol_3d.GetNumConformers() == 0:
        return None, "conformer_failed"
    conf = mol_3d.GetConformer()
    positions = [[conf.GetAtomPosition(i).x,
                  conf.GetAtomPosition(i).y,
                  conf.GetAtomPosition(i).z]
                 for i in range(mol_3d.GetNumAtoms())]
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
    return Data(x=x, z=z, pos=pos_tensor, edge_index=edge_index,
                edge_weight=edge_weight), "ok"


# ============================================================================
# Part 4: 神经网络模块（与训练代码完全一致）
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.block(x))


class Mol2VecEncoderImproved(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=256, output_dim=128,
                 num_blocks=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim))
        self.output_dim = output_dim

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_proj(x)


class DMPNNImproved(nn.Module):
    def __init__(self, in_channels=47, edge_dim=14, hidden_channels=128,
                 output_dim=128, num_layers=4, dropout=0.2, use_virtual_node=True):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.output_dim = output_dim
        self.use_virtual_node = use_virtual_node

        self.W_i = nn.Linear(in_channels + edge_dim, hidden_channels)
        self.W_h = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.edge_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels * 2 + edge_dim, hidden_channels),
                nn.ReLU(), nn.Linear(hidden_channels, edge_dim))
            for _ in range(num_layers)])
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])

        if use_virtual_node:
            self.virtual_node_embed = nn.Embedding(1, hidden_channels)
            self.virtual_mlp = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels),
                    nn.LayerNorm(hidden_channels), nn.ReLU(), nn.Dropout(dropout))
                for _ in range(num_layers)])

        self.W_o = nn.Linear(in_channels + hidden_channels, hidden_channels)
        self.bn_out = nn.BatchNorm1d(hidden_channels)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels, output_dim), nn.LayerNorm(output_dim))

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
            node_messages = torch.zeros(num_nodes, self.hidden_channels, device=x.device)
            node_messages.scatter_add_(0, col.unsqueeze(-1).expand_as(edge_hidden), edge_hidden)

            if self.use_virtual_node:
                virtual_agg = global_mean_pool(node_messages, batch)
                virtual_node = self.virtual_mlp[layer_idx](
                    torch.cat([virtual_node, virtual_agg], dim=-1)) + virtual_node
                node_messages = node_messages + virtual_node[batch]

            edge_msg = node_messages[row]
            if reverse_edge_idx is not None:
                valid_mask = reverse_edge_idx >= 0
                if valid_mask.any():
                    edge_msg[valid_mask] = (edge_msg[valid_mask]
                                            - edge_hidden[reverse_edge_idx[valid_mask]])

            edge_update_input = torch.cat([edge_hidden, edge_msg, edge_attr], dim=-1)
            edge_attr = edge_attr + self.edge_update[layer_idx](edge_update_input)

            edge_hidden_new = self.bn_layers[layer_idx](self.W_h[layer_idx](edge_msg))
            edge_hidden_new = F.relu(edge_hidden_new)
            edge_hidden = (edge_hidden +
                           F.dropout(edge_hidden_new, p=self.dropout, training=self.training))

        node_hidden = torch.zeros(num_nodes, self.hidden_channels, device=x.device)
        node_hidden.scatter_add_(0, col.unsqueeze(-1).expand_as(edge_hidden), edge_hidden)
        node_output = F.relu(self.bn_out(self.W_o(torch.cat([x, node_hidden], dim=-1))))
        node_output = F.dropout(node_output, p=self.dropout, training=self.training)

        h_mean = global_mean_pool(node_output, batch)
        h_max = global_max_pool(node_output, batch)
        return self.output_proj(torch.cat([h_mean, h_max], dim=-1))

    def _get_reverse_edge_indices(self, edge_index, num_edges):
        if num_edges == 0:
            return None
        row, col = edge_index
        device = edge_index.device
        max_nodes = max(row.max().item(), col.max().item()) + 1
        edge_hash = row * max_nodes + col
        reverse_hash = col * max_nodes + row
        sorted_hash, sorted_indices = torch.sort(edge_hash)
        positions = torch.searchsorted(sorted_hash, reverse_hash).clamp(max=num_edges - 1)
        found_mask = sorted_hash[positions] == reverse_hash
        return torch.where(found_mask, sorted_indices[positions],
                           torch.tensor(-1, device=device))


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
            -((distances.unsqueeze(-1) - self.centers) ** 2) / (self.widths ** 2 + 1e-8))


class SchNetInteraction(nn.Module):
    def __init__(self, hidden_channels, num_rbf=48, cutoff=8.0, dropout=0.3):
        super().__init__()
        self.cutoff = cutoff
        self.rbf = RBFExpansion(num_rbf=num_rbf, cutoff=cutoff, trainable=True)
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_channels), nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels))
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden_channels, hidden_channels))
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden_channels, hidden_channels))
        self.layer_norm = nn.LayerNorm(hidden_channels)

    def forward(self, x, edge_index, dist):
        row, col = edge_index
        filter_weight = self.filter_net(self.rbf(dist))
        messages = self.msg_mlp(x[col]) * filter_weight
        aggr = torch.zeros_like(x)
        aggr.scatter_add_(0, row.unsqueeze(-1).expand_as(messages), messages)
        out = self.update_mlp(torch.cat([x, aggr], dim=-1))
        return self.layer_norm(x + out)


class SchNetEncoder(nn.Module):
    def __init__(self, in_channels=47, hidden_channels=128, output_dim=128,
                 num_layers=4, num_rbf=48, cutoff=8.0, dropout=0.3, **kwargs):
        super().__init__()
        self.cutoff = cutoff
        self.output_dim = output_dim
        self.atom_embed = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels))
        self.z_embed = nn.Embedding(100, hidden_channels)
        self.interactions = nn.ModuleList([
            SchNetInteraction(hidden_channels, num_rbf, cutoff, dropout)
            for _ in range(num_layers)])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden_channels, output_dim),
            nn.LayerNorm(output_dim))

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
        return self.output_proj(torch.cat([h_mean, h_max, h_sum], dim=-1))


class SelfAttentionFusion(nn.Module):
    def __init__(self, dim_1d=128, dim_2d=128, dim_3d=128,
                 fusion_dim=128, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.proj_1d = nn.Sequential(nn.Linear(dim_1d, fusion_dim), nn.LayerNorm(fusion_dim))
        self.proj_2d = nn.Sequential(nn.Linear(dim_2d, fusion_dim), nn.LayerNorm(fusion_dim))
        self.proj_3d = nn.Sequential(nn.Linear(dim_3d, fusion_dim), nn.LayerNorm(fusion_dim))
        self.modality_embedding = nn.Embedding(3, fusion_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, nhead=num_heads, dim_feedforward=fusion_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool_query = nn.Parameter(torch.randn(1, 1, fusion_dim))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.pool_norm = nn.LayerNorm(fusion_dim)

    def forward(self, h_1d, h_2d, h_3d):
        B = h_1d.size(0)
        device = h_1d.device
        mod_emb = self.modality_embedding(torch.arange(3, device=device))
        z1 = self.proj_1d(h_1d) + mod_emb[0]
        z2 = self.proj_2d(h_2d) + mod_emb[1]
        z3 = self.proj_3d(h_3d) + mod_emb[2]
        seq = torch.stack([z1, z2, z3], dim=1)
        seq_out = self.transformer_encoder(seq)
        query = self.pool_query.expand(B, -1, -1)
        pooled, attn_weights = self.pool_attn(query=query, key=seq_out, value=seq_out)
        fused = self.pool_norm(pooled.squeeze(1))
        weights = attn_weights.squeeze(1)
        return fused, weights


class ConcatFusion(nn.Module):
    def __init__(self, dim_1d=128, dim_2d=128, dim_3d=128,
                 fusion_dim=128, dropout=0.2):
        super().__init__()
        total_dim = dim_1d + dim_2d + dim_3d
        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_dim, fusion_dim * 2), nn.LayerNorm(fusion_dim * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim), nn.LayerNorm(fusion_dim))

    def forward(self, h_1d, h_2d, h_3d):
        fused = self.fusion_mlp(torch.cat([h_1d, h_2d, h_3d], dim=-1))
        weights = torch.ones(h_1d.size(0), 3, device=h_1d.device) / 3.0
        return fused, weights


class MultiModalFusionNet(nn.Module):
    """完整融合模型（与训练代码保持完全一致）"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.fusion_mode = config.get('fusion_mode', '1D+2D+3D')
        self.fusion_type = config.get('fusion_type', 'self_attention')

        hidden_dim = config.get('hidden_dim', 128)
        dropout    = config.get('dropout', 0.3)
        atom_dim   = config.get('atom_dim', 47)
        edge_dim   = config.get('edge_dim', 14)

        if '1D' in self.fusion_mode:
            self.encoder_1d = Mol2VecEncoderImproved(
                input_dim=config.get('mol2vec_dim', 300),
                hidden_dim=hidden_dim * 2, output_dim=hidden_dim,
                num_blocks=config.get('num_res_blocks', 3), dropout=dropout)

        if '2D' in self.fusion_mode:
            self.encoder_2d = DMPNNImproved(
                in_channels=atom_dim, edge_dim=edge_dim,
                hidden_channels=hidden_dim, output_dim=hidden_dim,
                num_layers=config.get('dmpnn_layers', 4), dropout=dropout,
                use_virtual_node=config.get('use_virtual_node', True))

        if '3D' in self.fusion_mode:
            self.encoder_3d = SchNetEncoder(
                in_channels=atom_dim, hidden_channels=hidden_dim, output_dim=hidden_dim,
                num_layers=config.get('spherenet_layers', 4),
                cutoff=config.get('cutoff', 8.0),
                dropout=config.get('spherenet_dropout', 0.35))

        if self.fusion_mode == '1D+2D+3D':
            if self.fusion_type == 'self_attention':
                self.fusion = SelfAttentionFusion(
                    dim_1d=hidden_dim, dim_2d=hidden_dim, dim_3d=hidden_dim,
                    fusion_dim=hidden_dim,
                    num_heads=config.get('fusion_num_heads', 4),
                    num_layers=config.get('fusion_num_layers', 2), dropout=dropout)
            elif self.fusion_type == 'concat':
                self.fusion = ConcatFusion(
                    dim_1d=hidden_dim, dim_2d=hidden_dim, dim_3d=hidden_dim,
                    fusion_dim=hidden_dim, dropout=dropout)
            else:
                raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
            classifier_input_dim = hidden_dim
        else:
            self.fusion = None
            classifier_input_dim = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2))

    def forward(self, mol2vec_feat=None, graph_2d=None, graph_3d=None,
                ablation_mask=None):
        f1d = self.encoder_1d(mol2vec_feat) if (
            mol2vec_feat is not None and hasattr(self, 'encoder_1d')) else None
        f2d = self.encoder_2d(graph_2d) if (
            graph_2d is not None and hasattr(self, 'encoder_2d')) else None
        f3d = self.encoder_3d(graph_3d) if (
            graph_3d is not None and hasattr(self, 'encoder_3d')) else None

        if ablation_mask is not None:
            if not ablation_mask.get('1d', True) and f1d is not None:
                f1d = torch.zeros_like(f1d)
            if not ablation_mask.get('2d', True) and f2d is not None:
                f2d = torch.zeros_like(f2d)
            if not ablation_mask.get('3d', True) and f3d is not None:
                f3d = torch.zeros_like(f3d)

        weights = None
        if self.fusion_mode == '1D':
            x = f1d
        elif self.fusion_mode == '2D':
            x = f2d
        elif self.fusion_mode == '3D':
            x = f3d
        else:
            x, weights = self.fusion(f1d, f2d, f3d)

        logits = self.classifier(x)
        return {'logits': logits, 'features': x, 'weights': weights}

    def get_parameter_groups(self, base_lr, branch_scales=None):
        if branch_scales is None:
            branch_scales = {}
        groups = []
        for name in ['encoder_1d', 'encoder_2d', 'encoder_3d']:
            if hasattr(self, name):
                groups.append({'params': getattr(self, name).parameters(),
                                'lr': base_lr * branch_scales.get(name, 1.0),
                                'name': name})
        if self.fusion is not None:
            groups.append({'params': self.fusion.parameters(),
                           'lr': base_lr * branch_scales.get('fusion', 0.8),
                           'name': 'fusion'})
        groups.append({'params': self.classifier.parameters(),
                       'lr': base_lr * branch_scales.get('classifier', 1.0),
                       'name': 'classifier'})
        return groups


# ============================================================================
# Part 5: Mol2Vec 特征化（Stage1 需要 1D 时使用）
# ============================================================================

class Mol2VecFeaturizer:
    def __init__(self, model_path: Optional[str] = None,
                 embedding_dim: int = 300, radius: int = 1):
        if not MOL2VEC_AVAILABLE:
            raise ImportError("请先安装: pip install mol2vec gensim")
        self.radius = radius
        self.embedding_dim = embedding_dim
        self.model = None
        if model_path and Path(model_path).exists():
            self.model = gensim_w2v.Word2Vec.load(model_path)
            self.embedding_dim = self.model.wv.vector_size
            log.info(f"Mol2vec 模型加载完成，向量维度: {self.embedding_dim}")

    def _mol_to_sentence(self, mol):
        try:
            return mol2alt_sentence(mol, self.radius)
        except Exception:
            return None

    def train(self, smiles_list: List[str], save_path: Optional[str] = None):
        log.info(f"训练 Mol2vec，语料: {len(smiles_list)} 个分子...")
        sentences = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                sent = self._mol_to_sentence(mol)
                if sent:
                    sentences.append(sent)
        self.model = gensim_w2v.Word2Vec(
            sentences, vector_size=self.embedding_dim,
            window=5, min_count=1, sg=1, workers=4, epochs=10)
        if save_path:
            self.model.save(save_path)
            log.info(f"Mol2vec 模型已保存: {save_path}")

    def featurize(self, smiles_list: List[str]) -> np.ndarray:
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
                features.append(result[0] if result.shape[0] > 0
                                 else np.zeros(self.embedding_dim))
            except Exception:
                vecs = [self.model.wv[w] for w in sent if w in self.model.wv]
                features.append(np.mean(vecs, axis=0) if vecs
                                 else np.zeros(self.embedding_dim))
        return np.array(features)


# ============================================================================
# Part 6: Stage 0 —— 药性预过滤
# ============================================================================

def drug_like_filter(smiles: str,
                     strict: bool = False,
                     filter_dili: bool = True) -> Tuple[bool, Dict]:
    """
    完整版：Lipinski Ro5 + 扩展 ADMET + 肝毒性(DILI)预过滤

    Args:
        smiles: SMILES 表达式
        strict: 是否使用严格模式（更窄的阈值范围）
        filter_dili: 是否启用肝毒性结构警报过滤

    Returns:
        (是否通过, 属性字典)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, {}

    # ============================================================
    # 1. 肝毒性结构警报检查（基于警示结构）
    # ============================================================
    def _has_hepatotoxicity_alert(m: Chem.Mol) -> bool:
        """检测肝毒性相关警示结构"""
        if m is None:
            return False

        tox_alerts = [
            '[N;!H0]-[N;!H0]-[C;!H0]=[O,S]',  # 肼/酰肼类
            'C(=O)-C=C',  # 迈克尔受体（α,β-不饱和羰基）
            '[nX3](=O)[O-]',  # 芳香族硝基化合物
            'c1ccccc1N',  # 苯胺类
            'C1(=O)C=CC(=O)C=C1',  # 醌类
            '[#6]-S(=O)(=O)-N',  # 磺胺类
            'C1OC1',  # 环氧乙烷类
            'C(=O)-O-C(=O)-C',  # 酸酐类
            'S(=O)(=O)-[#6]-[#6]-S(=O)(=O)',  # 磺酰胺类双取代
            'N=C=S',  # 异硫氰酸酯
            'N=C=O',  # 异氰酸酯
            'CCl',  # 氯代烃（部分）
            'CBr',  # 溴代烃（部分）
        ]
        for smarts in tox_alerts:
            patt = Chem.MolFromSmarts(smarts)
            if patt and m.HasSubstructMatch(patt):
                return True
        return False

    # 执行肝毒性过滤
    if filter_dili:
        if _has_hepatotoxicity_alert(mol):
            # 仍需计算基本属性以便记录
            props = {
                'MW': Descriptors.MolWt(mol),
                'LogP': Crippen.MolLogP(mol),
                'HBD': rdMolDescriptors.CalcNumHBD(mol),
                'HBA': rdMolDescriptors.CalcNumHBA(mol),
                'TPSA': Descriptors.TPSA(mol),
                'RotBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'Rings': rdMolDescriptors.CalcNumRings(mol),
                'QED': QED.qed(mol),
                'HeavyAtoms': mol.GetNumHeavyAtoms(),
                'AromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'Fail_Reason': 'Hepatotoxicity_Alert'
            }
            return False, props

    # ============================================================
    # 2. 计算物理化学性质
    # ============================================================
    try:
        # 尝试计算 Gasteiger 电荷（可选，不影响过滤）
        AllChem.ComputeGasteigerCharges(mol)
    except:
        pass

    props = {
        'MW': Descriptors.MolWt(mol),
        'LogP': Crippen.MolLogP(mol),  # 使用 Crippen 方法计算 LogP
        'HBD': rdMolDescriptors.CalcNumHBD(mol),
        'HBA': rdMolDescriptors.CalcNumHBA(mol),
        'TPSA': Descriptors.TPSA(mol),
        'RotBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'Rings': rdMolDescriptors.CalcNumRings(mol),
        'QED': QED.qed(mol),
        'HeavyAtoms': mol.GetNumHeavyAtoms(),
        'AromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
    }

    # ============================================================
    # 3. Lipinski 类药五规则（Ro5）
    # ============================================================
    ro5_violations = sum([
        props['MW'] > 500,
        props['LogP'] > 5,
        props['HBD'] > 5,
        props['HBA'] > 10,
    ])

    # ============================================================
    # 4. 基础过滤条件
    # ============================================================
    basic_pass = (
            ro5_violations <= 1 and  # Lipinski 违反 ≤ 1
            props['TPSA'] <= 140 and  # 极性表面积 ≤ 140 Å²
            props['RotBonds'] <= 10 and  # 可旋转键 ≤ 10
            props['HeavyAtoms'] >= 10 and  # 重原子数 ≥ 10
            props['QED'] >= 0.25  # 类药性评分 ≥ 0.25
    )

    if not basic_pass:
        props['Fail_Reason'] = 'Basic_ADMET'

    # ============================================================
    # 5. 严格模式（更严格的阈值，用于高效筛选）
    # ============================================================
    if strict:
        strict_pass = (
                basic_pass and
                200 <= props['MW'] <= 450 and  # 分子量 200-450
                props['LogP'] <= 4 and  # LogP ≤ 4
                props['TPSA'] <= 90  # TPSA ≤ 90
        )
        if not strict_pass and 'Fail_Reason' not in props:
            props['Fail_Reason'] = 'Strict_ADMET'
        return strict_pass, props

    return basic_pass, props

# ============================================================================
# Part 7: 并行构象生成（方案B的精华，用于 Stage2 预处理加速）
# ============================================================================

def _generate_3d_one(smiles: str, cutoff: float,
                     num_confs: int) -> Tuple[str, Optional[Dict], str]:
    """
    单进程任务：生成一个分子的3D图，返回可序列化的字典
    Returns: (smiles, feature_dict_or_None, status)
    """
    try:
        gen = EnhancedConformerGenerator(num_confs=num_confs, select_by_energy=True)
        graph, status = mol_to_3d_graph(smiles, cutoff, gen)
        if graph is None:
            return smiles, None, status
        # 转成可 pickle 的 dict
        feat = {
            'x':           graph.x.numpy(),
            'z':           graph.z.numpy(),
            'pos':         graph.pos.numpy(),
            'edge_index':  graph.edge_index.numpy(),
            'edge_weight': graph.edge_weight.numpy(),
        }
        return smiles, feat, "ok"
    except Exception as e:
        return smiles, None, str(e)


def parallel_conformer_cache(smiles_list: List[str],
                              cache_path: str,
                              cutoff: float = 8.0,
                              num_confs: int = 5,
                              n_jobs: int = -1,
                              batch_size: int = 500) -> Dict[str, Any]:
    """
    并行生成 3D 构象并缓存到磁盘
    返回: smiles -> feature_dict 的字典
    """
    cache_path = Path(cache_path)

    # 如果缓存已存在，直接加载
    if cache_path.exists():
        log.info(f"加载已有构象缓存: {cache_path}")
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        log.info(f"  缓存命中: {len(cache)} 个分子")
        return cache

    log.info(f"并行生成 3D 构象 (n_jobs={n_jobs}, 总计 {len(smiles_list):,} 个)...")
    t0 = time.time()
    cache = {}

    if JOBLIB_AVAILABLE and n_jobs != 1:
        # joblib 并行
        results = Parallel(n_jobs=n_jobs, verbose=0, batch_size=batch_size)(
            delayed(_generate_3d_one)(smi, cutoff, num_confs)
            for smi in tqdm(smiles_list, desc="构象生成", unit="mol")
        )
    else:
        # 串行 fallback
        results = []
        for smi in tqdm(smiles_list, desc="构象生成（串行）", unit="mol"):
            results.append(_generate_3d_one(smi, cutoff, num_confs))

    ok_count = 0
    for smi, feat, status in results:
        if feat is not None:
            cache[smi] = feat
            ok_count += 1

    elapsed = time.time() - t0
    log.info(f"构象生成完成: {ok_count}/{len(smiles_list)} 成功, "
             f"耗时 {elapsed:.1f}s ({len(smiles_list)/elapsed:.0f} mol/s)")

    # 保存缓存
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f, protocol=4)
    log.info(f"构象缓存已保存: {cache_path}")
    return cache


# ============================================================================
# Part 8: Dataset（支持缓存预加载 3D 特征）
# ============================================================================

def collate_multimodal(batch: List[Dict]) -> Optional[Dict]:
    batch = [s for s in batch if s is not None]
    if not batch:
        return None
    result = {}

    if 'mol2vec' in batch[0] and batch[0]['mol2vec'] is not None:
        result['mol2vec'] = torch.stack(
            [torch.as_tensor(s['mol2vec'], dtype=torch.float) for s in batch])

    if 'graph_2d' in batch[0]:
        g2d_list = [s['graph_2d'] for s in batch
                    if isinstance(s.get('graph_2d'), Data)]
        if g2d_list:
            result['graph_2d'] = Batch.from_data_list(g2d_list)

    if 'graph_3d' in batch[0]:
        g3d_list = [s['graph_3d'] for s in batch
                    if isinstance(s.get('graph_3d'), Data) and hasattr(s['graph_3d'], 'pos')]
        if g3d_list:
            result['graph_3d'] = Batch.from_data_list(g3d_list)

    y_list = [s[k] for s in batch for k in ['label', 'y', 'labels'] if k in s]
    result['labels'] = (torch.tensor(y_list[:len(batch)], dtype=torch.long)
                        if y_list else torch.zeros(len(batch), dtype=torch.long))
    result['smiles_list'] = [s['smiles'] for s in batch]
    result['props_list']  = [s.get('props', {}) for s in batch]
    result['orig_idx']    = [s.get('orig_idx', -1) for s in batch]
    return result


class ScreeningDataset(Dataset):
    """
    筛选专用 Dataset
    - Stage1: 只生成 2D 图（可选 1D），跳过耗时的 3D 构象
    - Stage2: 从磁盘缓存加载预计算的 3D 特征，或实时生成
    """

    def __init__(self,
                 smiles_list:       List[str],
                 fusion_mode:       str = '1D+2D+3D',
                 cutoff:            float = 8.0,
                 mol2vec_features:  Optional[np.ndarray] = None,
                 conformer_cache:   Optional[Dict] = None,   # 预计算缓存
                 apply_filter:      bool = True,
                 strict_filter:     bool = False,
                 show_progress:     bool = True):

        self.fusion_mode = fusion_mode
        self.valid_data  = []
        self.failed      = []

        conformer_gen = (EnhancedConformerGenerator(num_confs=5)
                         if ('3D' in fusion_mode and conformer_cache is None)
                         else None)

        iter_obj = (tqdm(enumerate(smiles_list), total=len(smiles_list),
                         desc="构建数据集", unit="mol")
                    if show_progress else enumerate(smiles_list))

        for i, smi in iter_obj:
            # Stage 0: 药性过滤
            if apply_filter:
                ok, props = drug_like_filter(smi, strict=strict_filter)
                if not ok:
                    self.failed.append({'smiles': smi, 'reason': 'drug_like_filter'})
                    continue
            else:
                props = {}

            sample = {'smiles': smi, 'props': props, 'orig_idx': i, 'label': 0}

            # 1D 特征
            if '1D' in fusion_mode and mol2vec_features is not None:
                sample['mol2vec'] = mol2vec_features[i]

            # 2D 图
            if '2D' in fusion_mode:
                g2d = mol_to_2d_graph(smi)
                if g2d is None:
                    self.failed.append({'smiles': smi, 'reason': '2d_failed'})
                    continue
                sample['graph_2d'] = g2d

            # 3D 图（优先从缓存加载）
            if '3D' in fusion_mode:
                if conformer_cache is not None and smi in conformer_cache:
                    feat = conformer_cache[smi]
                    sample['graph_3d'] = Data(
                        x=torch.tensor(feat['x'], dtype=torch.float),
                        z=torch.tensor(feat['z'], dtype=torch.long),
                        pos=torch.tensor(feat['pos'], dtype=torch.float),
                        edge_index=torch.tensor(feat['edge_index'], dtype=torch.long),
                        edge_weight=torch.tensor(feat['edge_weight'], dtype=torch.float))
                else:
                    g3d, status = mol_to_3d_graph(smi, cutoff, conformer_gen)
                    if g3d is None:
                        self.failed.append({'smiles': smi, 'reason': f'3d_{status}'})
                        continue
                    sample['graph_3d'] = g3d

            self.valid_data.append(sample)

        log.info(f"数据集构建完成: {len(self.valid_data):,} 有效 / "
                 f"{len(self.failed):,} 过滤")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        return self.valid_data[idx]


# ============================================================================
# Part 9: 推理引擎
# ============================================================================

class VirtualScreener:
    def __init__(self, model_path: str, model_config: Dict,
                 device: str = 'auto', threshold: float = 0.5):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        log.info(f"加载模型: {model_path}  设备: {self.device}")
        self.model = MultiModalFusionNet(model_config)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.threshold   = threshold
        self.fusion_mode = model_config.get('fusion_mode', '1D+2D+3D')

        n_params = sum(p.numel() for p in self.model.parameters())
        log.info(f"模型参数: {n_params:,}")

    @torch.no_grad()
    def screen(self, dataset: ScreeningDataset,
               batch_size: int = 64,
               num_workers: int = 0) -> pd.DataFrame:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_multimodal, num_workers=num_workers,
                            pin_memory=(self.device.type == 'cuda'))

        all_probs   = []
        all_weights = []

        log.info(f"开始推理: {len(dataset):,} 个分子...")
        t0 = time.time()

        for batch in tqdm(loader, desc="推理中", unit="batch"):
            if batch is None:
                continue
            kwargs = {}
            if 'mol2vec' in batch:
                kwargs['mol2vec_feat'] = batch['mol2vec'].to(self.device)
            if 'graph_2d' in batch:
                kwargs['graph_2d'] = batch['graph_2d'].to(self.device)
            if 'graph_3d' in batch:
                kwargs['graph_3d'] = batch['graph_3d'].to(self.device)

            result = self.model(**kwargs)
            probs = F.softmax(result['logits'], dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())

            if result.get('weights') is not None:
                all_weights.extend(result['weights'].cpu().numpy().tolist())

        elapsed = time.time() - t0
        log.info(f"推理完成: {len(all_probs):,} 个, 耗时 {elapsed:.1f}s "
                 f"({len(all_probs)/elapsed:.0f} mol/s)")

        n = len(all_probs)
        smiles_list = [s['smiles']    for s in dataset.valid_data[:n]]
        props_list  = [s['props']     for s in dataset.valid_data[:n]]
        orig_idx    = [s['orig_idx']  for s in dataset.valid_data[:n]]

        df = pd.DataFrame({
            'smiles':   smiles_list,
            'score':    all_probs,
            'active':   [int(p >= self.threshold) for p in all_probs],
            'orig_idx': orig_idx,
        })

        # 合并物化性质
        props_df = pd.DataFrame(props_list)
        if not props_df.empty:
            df = pd.concat([df, props_df], axis=1)

        # 合并模态权重
        if all_weights and len(all_weights) == n:
            w_arr = np.array(all_weights)
            n_cols = w_arr.shape[1]
            col_names = ['w_1d', 'w_2d', 'w_3d'][:n_cols]
            for j, col in enumerate(col_names):
                df[col] = w_arr[:, j]

        return df.sort_values('score', ascending=False).reset_index(drop=True)


# ============================================================================
# Part 10: 多样性过滤（Tanimoto 去冗余）
# ============================================================================

def diversity_filter(df: pd.DataFrame,
                     top_n: int = 500,
                     similarity_threshold: float = 0.65,
                     radius: int = 2,
                     n_bits: int = 2048) -> pd.DataFrame:
    """
    基于 Morgan 指纹 Tanimoto 相似度的贪心去冗余
    从高分到低分选择，保证候选集结构多样性
    """
    log.info(f"多样性过滤: {len(df)} → 目标 {top_n} (阈值={similarity_threshold})")

    # 只在高分候选中操作（加速）
    candidates = df.head(min(top_n * 10, len(df))).copy().reset_index(drop=True)

    fps, valid_mask = [], []
    for smi in candidates['smiles']:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits))
            valid_mask.append(True)
        else:
            fps.append(None)
            valid_mask.append(False)

    selected_idx = []
    selected_fps = []

    for i, (fp, valid) in enumerate(zip(fps, valid_mask)):
        if not valid or fp is None:
            continue
        if selected_fps:
            sims = DataStructs.BulkTanimotoSimilarity(fp, selected_fps)
            if max(sims) >= similarity_threshold:
                continue
        selected_idx.append(i)
        selected_fps.append(fp)
        if len(selected_idx) >= top_n:
            break

    result = candidates.iloc[selected_idx].copy()
    result['diversity_rank'] = range(1, len(result) + 1)
    log.info(f"  多样性过滤后: {len(result)} 个化合物")
    return result.reset_index(drop=True)


# ============================================================================
# Part 11: 两阶段主筛选流程
# ============================================================================

def run_two_stage_screening(
    smiles_list:         List[str],
    stage1_model_path:   str,               # 1D+2D 训练好的模型权重
    stage2_model_path:   str,               # 1D+2D+3D 融合模型权重
    stage1_config:       Dict,              # Stage1 模型配置
    stage2_config:       Dict,              # Stage2 模型配置
    output_dir:          str  = 'screening_results',
    stage1_threshold:    float = 0.30,      # 宽松阈值，避免漏掉真阳性
    stage2_threshold:    float = 0.50,      # 最终判决阈值
    top_k_ratio:         float = 0.05,      # Stage1 保留比例
    top_k_min:           int   = 200,       # Stage1 最少保留数量
    final_top_n:         int   = 500,       # 最终候选数量
    diversity_threshold: float = 0.65,      # 多样性过滤阈值
    batch_size:          int   = 64,
    mol2vec_path_1d:     Optional[str] = None,   # Stage1 Mol2Vec 模型路径
    mol2vec_path_2d:     Optional[str] = None,   # Stage2 Mol2Vec 模型路径（可同一个）
    conformer_cache_path: Optional[str] = None,  # 3D 构象缓存路径（可选）
    n_jobs_conformer:    int   = -1,        # 并行构象生成进程数（-1=全部核心）
    apply_filter:        bool  = True,
    strict_filter:       bool  = False,
    device:              str   = 'auto',
) -> Tuple[pd.DataFrame, Dict]:
    """
    两阶段虚拟筛选主函数

    Stage0: Lipinski/ADMET 药性过滤（极快）
    Stage1: 1D+2D 粗筛，宽松阈值保留 top_k_ratio
    Stage2: 1D+2D+3D 融合精筛，使用预计算/并行构象缓存
    Post:   多样性去冗余 + 结果导出

    Returns:
        final_hits (DataFrame): 最终候选化合物
        stats (Dict):           各阶段统计信息
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_input = len(smiles_list)
    stats = {'total_input': total_input}

    log.info("\n" + "=" * 70)
    log.info("两阶段虚拟筛选开始")
    log.info(f"  输入总量: {total_input:,}")
    log.info(f"  输出目录: {output_dir}")
    log.info("=" * 70)

    # ── Stage 1: 1D+2D 快速粗筛 ────────────────────────────────────────
    log.info("\n[Stage 1/2] 1D+2D 快速粗筛...")
    t1 = time.time()

    # 准备 Stage1 的 Mol2Vec 特征
    s1_fusion_mode = stage1_config.get('fusion_mode', '1D+2D')
    mol2vec_s1 = None
    if '1D' in s1_fusion_mode and mol2vec_path_1d and MOL2VEC_AVAILABLE:
        log.info("  生成 Stage1 Mol2Vec 特征...")
        feat1d = Mol2VecFeaturizer(model_path=mol2vec_path_1d)
        mol2vec_s1 = feat1d.featurize(smiles_list)

    stage1_dataset = ScreeningDataset(
        smiles_list,
        fusion_mode=s1_fusion_mode,
        mol2vec_features=mol2vec_s1,
        apply_filter=apply_filter,
        strict_filter=strict_filter,
        show_progress=True,
    )
    stats['stage0_filtered'] = total_input - len(stage1_dataset)
    stats['stage1_input']    = len(stage1_dataset)

    screener1 = VirtualScreener(
        stage1_model_path, stage1_config,
        device=device, threshold=stage1_threshold)
    stage1_results = screener1.screen(stage1_dataset, batch_size=batch_size)

    # 保留 Top K%（至少 top_k_min 个）
    top_k = max(int(len(stage1_results) * top_k_ratio), top_k_min)
    top_k = min(top_k, len(stage1_results))
    stage1_candidates = stage1_results.head(top_k).copy()
    candidate_smiles  = stage1_candidates['smiles'].tolist()

    stats['stage1_output'] = len(candidate_smiles)
    stage1_candidates.to_csv(output_dir / 'stage1_candidates.csv', index=False)
    log.info(f"Stage1 完成: {total_input:,} → {len(candidate_smiles):,} 候选 "
             f"(耗时 {time.time()-t1:.1f}s)")

    # ── 可选：并行预计算 Stage2 的 3D 构象 ────────────────────────────
    conformer_cache = None
    cutoff = stage2_config.get('cutoff', 8.0)

    if conformer_cache_path is not None:
        # 若缓存文件不存在则并行生成，存在则直接加载
        conformer_cache = parallel_conformer_cache(
            candidate_smiles,
            cache_path=conformer_cache_path,
            cutoff=cutoff,
            num_confs=5,
            n_jobs=n_jobs_conformer,
        )

    # ── Stage 2: 1D+2D+3D 融合精筛 ──────────────────────────────────
    log.info("\n[Stage 2/2] 1D+2D+3D 融合精筛...")
    t2 = time.time()

    # 准备 Stage2 的 Mol2Vec 特征
    s2_fusion_mode = stage2_config.get('fusion_mode', '1D+2D+3D')
    mol2vec_s2 = None
    if '1D' in s2_fusion_mode and mol2vec_path_2d and MOL2VEC_AVAILABLE:
        log.info("  生成 Stage2 Mol2Vec 特征...")
        feat2d = Mol2VecFeaturizer(model_path=mol2vec_path_2d)
        mol2vec_s2 = feat2d.featurize(candidate_smiles)

    stage2_dataset = ScreeningDataset(
        candidate_smiles,
        fusion_mode=s2_fusion_mode,
        mol2vec_features=mol2vec_s2,
        conformer_cache=conformer_cache,
        apply_filter=False,            # Stage1 已过滤，避免重复
        show_progress=True,
    )
    stats['stage2_input'] = len(stage2_dataset)

    screener2 = VirtualScreener(
        stage2_model_path, stage2_config,
        device=device, threshold=stage2_threshold)
    stage2_results = screener2.screen(stage2_dataset, batch_size=batch_size)

    # 保存精筛全量结果
    stage2_results.to_csv(output_dir / 'stage2_results.csv', index=False)
    log.info(f"Stage2 完成: {len(candidate_smiles):,} → 打分完毕 "
             f"(耗时 {time.time()-t2:.1f}s)")

    # ── Post: 多样性过滤 + 最终输出 ──────────────────────────────────
    actives_df = stage2_results[stage2_results['active'] == 1].copy()
    stats['stage2_actives'] = len(actives_df)

    if len(actives_df) == 0:
        log.warning("Stage2 未找到活性分子！请检查模型或降低 stage2_threshold")
        # 降级：取分最高的 final_top_n 个
        actives_df = stage2_results.head(final_top_n)

    final_hits = diversity_filter(
        actives_df,
        top_n=final_top_n,
        similarity_threshold=diversity_threshold)

    stats['final_hits'] = len(final_hits)

    # 保存最终结果
    final_hits.to_csv(output_dir / f'final_top{final_top_n}_hits.csv', index=False)

    # ── 汇总报告 ────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("两阶段筛选汇总")
    log.info("=" * 70)
    log.info(f"  原始输入:       {stats['total_input']:>10,}")
    log.info(f"  药性过滤后:     {stats['stage0_filtered']:>10,}  (已过滤)")
    log.info(f"  Stage1 输入:    {stats['stage1_input']:>10,}")
    log.info(f"  Stage1 粗筛出:  {stats['stage1_output']:>10,}")
    log.info(f"  Stage2 有效:    {stats['stage2_input']:>10,}")
    log.info(f"  Stage2 预测活性:{stats['stage2_actives']:>10,}")
    log.info(f"  最终候选:       {stats['final_hits']:>10,}")
    log.info(f"  结果目录:       {output_dir}")

    if 'score' in final_hits.columns:
        log.info(f"  分值范围:       {final_hits['score'].min():.3f} ~ "
                 f"{final_hits['score'].max():.3f}")
    if 'QED' in final_hits.columns:
        log.info(f"  平均 QED:       {final_hits['QED'].mean():.3f}")

    # 保存统计到 JSON
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    return final_hits, stats


# ============================================================================
# Part 12: 读取数据库工具函数
# ============================================================================

def load_database(db_path: str) -> List[str]:
    """支持 CSV / SDF 格式"""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")

    if db_path.suffix.lower() == '.sdf':
        suppl = Chem.SDMolSupplier(str(db_path))
        smiles_list = []
        for mol in tqdm(suppl, desc="读取SDF", unit="mol"):
            if mol is not None:
                smi = Chem.MolToSmiles(mol)
                if smi:
                    smiles_list.append(smi)
        log.info(f"SDF 加载: {len(smiles_list):,} 个有效分子")
        return smiles_list
    else:
        df = pd.read_csv(db_path)
        smi_col = next((c for c in df.columns if 'smiles' in c.lower()), None)
        if smi_col is None:
            raise ValueError(f"CSV 中找不到 SMILES 列: {list(df.columns)}")
        smiles = df[smi_col].dropna().astype(str).tolist()
        log.info(f"CSV 加载: {len(smiles):,} 个分子")
        return smiles


# ============================================================================
# Part 13: 命令行入口
# ============================================================================

def build_default_config() -> Dict:
    return {
        'database_path':       r'D:\dll\NLRP3_2\src\data\library.csv',
        'stage1_model_path':   r'D:\dll\NLRP3_2\results\self_attention_fusion\NLRP3_SelfAttention_Fusion_20260304_215802\models\1D_none.pt',
        'stage2_model_path':   r'D:\dll\NLRP3_2\results\self_attention_fusion\NLRP3_SelfAttention_Fusion_20260304_215802\models\1D_none.pt',
        'mol2vec_path':        None,           # 若有则填写路径
        'conformer_cache_path': None,          # 设置路径则启用缓存
        'output_dir':          'screening_results',
        'stage1_threshold':    0.30,
        'stage2_threshold':    0.50,
        'top_k_ratio':         0.05,
        'top_k_min':           200,
        'final_top_n':         500,
        'diversity_threshold': 0.65,
        'batch_size':          64,
        'n_jobs_conformer':    -1,
        'apply_filter':        True,
        'strict_filter':       False,
        'device':              'auto',
        'stage1_config': {
            'fusion_mode':      '2D',
            'fusion_type':      'none',
            'hidden_dim':       128,
            'dropout':          0.3,
            'dmpnn_layers':     4,
            'use_virtual_node': True,
            'num_res_blocks':   3,
            'mol2vec_dim':      300,
            'atom_dim':         47,
            'edge_dim':         14,
        },
        'stage2_config': {
            'fusion_mode':       '2D',
            'fusion_type':       'self_attention',
            'hidden_dim':        128,
            'dropout':           0.3,
            'spherenet_dropout': 0.35,
            'dmpnn_layers':      4,
            'spherenet_layers':  4,
            'cutoff':            8.0,
            'use_virtual_node':  True,
            'num_res_blocks':    3,
            'fusion_num_heads':  4,
            'fusion_num_layers': 2,
            'mol2vec_dim':       300,
            'atom_dim':          47,
            'edge_dim':          14,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description='两阶段虚拟筛选 (Stage1: 1D+2D, Stage2: 1D+2D+3D)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config',   '-c', default='screening_config.yaml',
                        help='YAML 配置文件路径')
    parser.add_argument('--database', '-d', default=None,
                        help='分子数据库（覆盖配置）')
    parser.add_argument('--stage1_model', default=None, help='Stage1 模型路径')
    parser.add_argument('--stage2_model', default=None, help='Stage2 模型路径')
    parser.add_argument('--output',   '-o', default=None, help='输出目录')
    parser.add_argument('--top_k_ratio', type=float, default=None,
                        help='Stage1 保留比例 (0~1)')
    parser.add_argument('--final_top_n', type=int, default=None,
                        help='最终保留候选数')
    parser.add_argument('--batch_size',  type=int, default=None)
    parser.add_argument('--n_jobs', type=int, default=None,
                        help='并行构象生成进程数 (-1=全部CPU)')
    parser.add_argument('--cache', default=None,
                        help='3D 构象缓存文件路径')
    parser.add_argument('--gen_config', action='store_true',
                        help='生成默认配置文件后退出')
    args = parser.parse_args()

    # 生成默认配置文件
    if args.gen_config:
        cfg = build_default_config()
        cfg_path = Path(args.config)
        with open(cfg_path, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        log.info(f"默认配置文件已生成: {cfg_path}")
        log.info("请修改 database_path / stage1_model_path / stage2_model_path 后运行")
        return

    # 加载配置
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        log.info(f"配置文件已加载: {cfg_path}")
    else:
        log.warning(f"配置文件 {cfg_path} 不存在，使用内置默认值")
        log.warning("可用 --gen_config 生成默认配置文件")
        cfg = build_default_config()

    # 命令行参数覆盖配置文件
    if args.database:      cfg['database_path']        = args.database
    if args.stage1_model:  cfg['stage1_model_path']    = args.stage1_model
    if args.stage2_model:  cfg['stage2_model_path']    = args.stage2_model
    if args.output:        cfg['output_dir']            = args.output
    if args.top_k_ratio:   cfg['top_k_ratio']           = args.top_k_ratio
    if args.final_top_n:   cfg['final_top_n']           = args.final_top_n
    if args.batch_size:    cfg['batch_size']             = args.batch_size
    if args.n_jobs:        cfg['n_jobs_conformer']       = args.n_jobs
    if args.cache:         cfg['conformer_cache_path']  = args.cache

    # 加载数据库
    smiles_list = load_database(cfg['database_path'])

    # 运行两阶段筛选
    final_hits, stats = run_two_stage_screening(
        smiles_list          = smiles_list,
        stage1_model_path    = cfg['stage1_model_path'],
        stage2_model_path    = cfg['stage2_model_path'],
        stage1_config        = cfg['stage1_config'],
        stage2_config        = cfg['stage2_config'],
        output_dir           = cfg.get('output_dir', 'screening_results'),
        stage1_threshold     = cfg.get('stage1_threshold', 0.30),
        stage2_threshold     = cfg.get('stage2_threshold', 0.50),
        top_k_ratio          = cfg.get('top_k_ratio', 0.05),
        top_k_min            = cfg.get('top_k_min', 200),
        final_top_n          = cfg.get('final_top_n', 500),
        diversity_threshold  = cfg.get('diversity_threshold', 0.65),
        batch_size           = cfg.get('batch_size', 64),
        mol2vec_path_1d      = cfg.get('mol2vec_path'),
        mol2vec_path_2d      = cfg.get('mol2vec_path'),
        conformer_cache_path = cfg.get('conformer_cache_path'),
        n_jobs_conformer     = cfg.get('n_jobs_conformer', -1),
        apply_filter         = cfg.get('apply_filter', True),
        strict_filter        = cfg.get('strict_filter', False),
        device               = cfg.get('device', 'auto'),
    )

    log.info(f"\n全部完成，最终候选: {len(final_hits)} 个")
    log.info(f"结果文件: {cfg.get('output_dir', 'screening_results')}/final_top*_hits.csv")


if __name__ == '__main__':
    main()