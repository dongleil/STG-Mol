"""
scaffold_analyzer_improved.py
=============================
改进的BMS骨架分析模块 - 支持大骨架拆分

Author: Research Team
Date: 2025-01-XX
Version: 2.0 - 支持大骨架拆分
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import json
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold


def smart_read_csv(filepath: Path) -> pd.DataFrame:
    """
    智能读取CSV文件，自动识别SMILES和label列
    ✅ 适配你的数据集格式
    """
    df = pd.read_csv(filepath)

    print(f"📋 原始列名: {list(df.columns)[:5]}...")

    # 智能识别SMILES列
    smiles_candidates = [
        'smiles_standardized',  # 你的数据集使用这个
        'smiles', 'SMILES', 'Smiles',
        'canonical_smiles', 'Canonical_Smiles',
        'smi', 'SMI'
    ]
    smiles_col = None
    for col in smiles_candidates:
        if col in df.columns:
            smiles_col = col
            break

    if smiles_col is None:
        raise ValueError(f"❌ 找不到SMILES列! 可用列: {list(df.columns)}")

    # 智能识别标签列
    label_candidates = [
        'label', 'Label', 'LABEL',
        'activity', 'Activity', 'ACTIVITY',
        'class', 'Class', 'CLASS',
        'target', 'Target'
    ]
    label_col = None
    for col in label_candidates:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        raise ValueError(f"❌ 找不到label列! 可用列: {list(df.columns)}")

    print(f"✅ 列映射成功:")
    print(f"   SMILES: '{smiles_col}' → 'smiles'")
    print(f"   Label:  '{label_col}' → 'label'")

    # 重命名列并只保留需要的列
    df_clean = df[[smiles_col, label_col]].copy()
    df_clean.columns = ['smiles', 'label']

    # 移除空值
    before_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['smiles', 'label'])
    after_count = len(df_clean)

    if before_count > after_count:
        print(f"⚠️  移除了 {before_count - after_count} 行空值")

    print(f"✅ 加载完成: {len(df_clean)} 条有效数据")

    return df_clean


class ScaffoldAnalyzer:
    """BMS骨架分析器"""

    def __init__(self, scaffold_type: str = 'generic'):
        """
        Parameters:
        -----------
        scaffold_type: str
            'murcko' - 保留原子类型和键类型
            'generic' - 通用骨架（推荐）
            'framework' - 只保留环系统
        """
        if scaffold_type not in ['murcko', 'generic', 'framework']:
            raise ValueError(f"scaffold_type must be one of: murcko, generic, framework")

        self.scaffold_type = scaffold_type

    def get_scaffold(self, smiles: str) -> Optional[str]:
        """提取单个分子的BMS骨架"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # 获取Murcko骨架
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)

            if scaffold is None:
                return None

            if self.scaffold_type == 'generic':
                scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
            elif self.scaffold_type == 'framework':
                scaffold = Chem.DeleteSubstructs(
                    scaffold,
                    Chem.MolFromSmarts('[!R]')
                )

            return Chem.MolToSmiles(scaffold) if scaffold else None

        except Exception as e:
            return None

    def extract_scaffolds(self, df: pd.DataFrame) -> Dict[str, str]:
        """提取DataFrame中所有分子的骨架"""
        print(f"   提取 {self.scaffold_type} 骨架...")

        scaffolds = {}
        valid_count = 0

        for smiles in df['smiles']:
            scaffold = self.get_scaffold(smiles)
            if scaffold:
                scaffolds[smiles] = scaffold
                valid_count += 1

        print(f"   有效骨架: {valid_count}/{len(df)}")

        return scaffolds

    def analyze_dataset_leakage(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        save_dir: Optional[Path] = None
    ) -> Dict:
        """全面诊断数据泄露问题"""

        print("\n" + "="*80)
        print(f"🔬 BMS骨架泄露分析报告 (scaffold_type={self.scaffold_type})")
        print("="*80)

        # 1. 提取所有骨架
        print("\n[步骤 1/6] 提取骨架...")
        train_scaffolds = self.extract_scaffolds(train_df)
        val_scaffolds = self.extract_scaffolds(val_df)
        test_scaffolds = self.extract_scaffolds(test_df)

        train_scaffold_set = set(train_scaffolds.values())
        val_scaffold_set = set(val_scaffolds.values())
        test_scaffold_set = set(test_scaffolds.values())

        # 2. 基本统计
        print("\n[步骤 2/6] 📊 基本统计")
        print("-" * 80)
        print(f"{'数据集':<15} {'分子数':<12} {'唯一骨架数':<15} {'分子/骨架比':<15}")
        print("-" * 80)
        print(f"{'训练集':<15} {len(train_df):<12} {len(train_scaffold_set):<15} {len(train_df)/len(train_scaffold_set) if train_scaffold_set else 0:.2f}")
        print(f"{'验证集':<15} {len(val_df):<12} {len(val_scaffold_set):<15} {len(val_df)/len(val_scaffold_set) if val_scaffold_set else 0:.2f}")
        print(f"{'测试集':<15} {len(test_df):<12} {len(test_scaffold_set):<15} {len(test_df)/len(test_scaffold_set) if test_scaffold_set else 0:.2f}")
        print("-" * 80)

        # 3. 骨架重叠分析
        print("\n[步骤 3/6] 🚨 骨架重叠分析")
        print("-" * 80)

        train_val_overlap = train_scaffold_set & val_scaffold_set
        train_val_ratio = len(train_val_overlap) / len(val_scaffold_set) if val_scaffold_set else 0

        train_test_overlap = train_scaffold_set & test_scaffold_set
        train_test_ratio = len(train_test_overlap) / len(test_scaffold_set) if test_scaffold_set else 0

        val_test_overlap = val_scaffold_set & test_scaffold_set
        val_test_ratio = len(val_test_overlap) / len(test_scaffold_set) if test_scaffold_set else 0

        print(f"\n{'对比':<20} {'重叠骨架数':<15} {'重叠率':<15} {'状态'}")
        print("-" * 80)

        # 训练 vs 验证
        status = self._get_risk_status(train_val_ratio)
        print(f"{'训练 vs 验证':<20} {len(train_val_overlap):<15} {train_val_ratio:<15.2%} {status}")

        # 训练 vs 测试 (最重要)
        status = self._get_risk_status(train_test_ratio)
        print(f"{'训练 vs 测试':<20} {len(train_test_overlap):<15} {train_test_ratio:<15.2%} {status}")

        # 验证 vs 测试
        status = self._get_risk_status(val_test_ratio)
        print(f"{'验证 vs 测试':<20} {len(val_test_overlap):<15} {val_test_ratio:<15.2%} {status}")

        print("-" * 80)

        # 风险评估
        risk_level = self._evaluate_risk(train_test_ratio)
        print(f"\n⚠️  总体风险等级: {risk_level}")

        if risk_level == "CRITICAL":
            print("   🚨🚨🚨 严重警告: 测试集骨架重叠超过70%！")
            print("          模型性能被严重高估，必须重新划分数据集！")
        elif risk_level == "HIGH":
            print("   ⚠️⚠️  高风险: 测试集骨架重叠超过50%")
            print("         强烈建议使用Scaffold Split重新划分")
        elif risk_level == "MEDIUM":
            print("   ⚠️  中等风险: 测试集骨架重叠超过30%")
            print("      模型泛化能力可能被高估")
        else:
            print("   ✅ 低风险: 骨架重叠率在合理范围内")

        # 4. 分子级别的泄露统计
        print("\n[步骤 4/6] 🔍 分子级别泄露统计")
        print("-" * 80)

        test_seen_scaffold_count = sum(
            1 for scaffold in test_scaffolds.values()
            if scaffold in train_scaffold_set
        )
        test_seen_ratio = test_seen_scaffold_count / len(test_df) if len(test_df) > 0 else 0

        print(f"测试集总分子数: {len(test_df)}")
        print(f"测试集中见过骨架的分子数: {test_seen_scaffold_count}")
        print(f"⚠️  见过骨架的分子占比: {test_seen_ratio:.2%}")

        if test_seen_ratio > 0.7:
            print("\n   🚨 超过70%的测试分子具有训练集中见过的骨架！")
            print("      模型可能只是在'记忆'骨架，而非真正学习")

        # 5. 标签-骨架关系
        print("\n[步骤 5/6] 📋 标签-骨架关系分析")
        print("-" * 80)
        label_analysis = self._analyze_label_scaffold_relationship(
            train_df, test_df, train_scaffolds, test_scaffolds
        )

        # 6. 分子相似度分析
        print("\n[步骤 6/6] 🔬 分子相似度分析（Tanimoto，采样100个测试分子）")
        print("-" * 80)
        similarity_results = self._calculate_similarity(
            train_df['smiles'].tolist(),
            test_df['smiles'].tolist(),
            sample_size=min(100, len(test_df))
        )

        print(f"平均最大相似度: {similarity_results['avg_max_sim']:.3f}")
        print(f"高相似度(>0.8)分子数: {similarity_results['high_sim_count']}/{similarity_results['sample_size']}")
        print(f"⚠️  高相似度占比: {similarity_results['high_sim_ratio']:.2%}")

        if similarity_results['high_sim_ratio'] > 0.5:
            print("\n   ⚠️  超过50%的测试分子与训练集高度相似（Tanimoto>0.8）")

        # 汇总结果
        results = {
            'scaffold_type': self.scaffold_type,
            'risk_level': risk_level,
            'train_test_overlap_ratio': train_test_ratio,
            'train_val_overlap_ratio': train_val_ratio,
            'test_seen_scaffold_ratio': test_seen_ratio,
            'train_unique_scaffolds': len(train_scaffold_set),
            'val_unique_scaffolds': len(val_scaffold_set),
            'test_unique_scaffolds': len(test_scaffold_set),
            'overlap_scaffolds': len(train_test_overlap),
            'similarity_results': similarity_results,
            'label_analysis': label_analysis,
            'train_scaffolds': train_scaffolds,
            'val_scaffolds': val_scaffolds,
            'test_scaffolds': test_scaffolds
        }

        # 最终结论
        print("\n" + "="*80)
        print("📋 诊断结论")
        print("="*80)
        print(f"风险等级: {risk_level}")
        print(f"训练-测试骨架重叠率: {train_test_ratio:.2%}")
        print(f"测试集见过骨架分子占比: {test_seen_ratio:.2%}")
        print(f"高相似度测试分子占比: {similarity_results['high_sim_ratio']:.2%}")

        if risk_level in ["CRITICAL", "HIGH"]:
            print("\n💡 建议行动:")
            print("   1. 使用 perform_scaffold_split() 重新划分数据集")
            print("   2. 重新训练所有模型")
            print("   3. 对比 Random Split 和 Scaffold Split 的性能差异")
            print("   4. 在论文中报告 Scaffold Split 的结果作为真实泛化能力")

        # 保存报告
        if save_dir:
            self._save_report(results, save_dir)
            self._plot_scaffold_distribution(results, save_dir)

        print("="*80 + "\n")

        return results

    def _get_risk_status(self, ratio: float) -> str:
        """获取风险状态标签"""
        if ratio > 0.7:
            return "🚨 严重"
        elif ratio > 0.5:
            return "⚠️  高"
        elif ratio > 0.3:
            return "⚠️  中"
        else:
            return "✅ 低"

    def _evaluate_risk(self, train_test_ratio: float) -> str:
        """评估整体风险等级"""
        if train_test_ratio > 0.7:
            return "CRITICAL"
        elif train_test_ratio > 0.5:
            return "HIGH"
        elif train_test_ratio > 0.3:
            return "MEDIUM"
        else:
            return "LOW"

    def _analyze_label_scaffold_relationship(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_scaffolds: Dict,
        test_scaffolds: Dict
    ) -> Dict:
        """分析标签-骨架关系"""

        # 构建训练集骨架-标签映射
        train_scaffold_labels = defaultdict(list)
        for idx, row in train_df.iterrows():
            smiles = row['smiles']
            label = row['label']
            if smiles in train_scaffolds:
                scaffold = train_scaffolds[smiles]
                train_scaffold_labels[scaffold].append(label)

        # 统计测试集中见过骨架的标签一致性
        consistent = 0
        inconsistent = 0
        not_seen = 0

        for idx, row in test_df.iterrows():
            smiles = row['smiles']
            label = row['label']
            if smiles in test_scaffolds:
                scaffold = test_scaffolds[smiles]
                if scaffold in train_scaffold_labels:
                    train_labels = train_scaffold_labels[scaffold]
                    if label in train_labels:
                        consistent += 1
                    else:
                        inconsistent += 1
                else:
                    not_seen += 1

        total_seen = consistent + inconsistent

        print(f"测试集分子分类:")
        print(f"  骨架未见过: {not_seen} ({not_seen/len(test_df):.2%})")
        print(f"  骨架见过，标签一致: {consistent} ({consistent/len(test_df):.2%})")
        print(f"  骨架见过，标签不一致: {inconsistent} ({inconsistent/len(test_df):.2%})")

        if total_seen > 0:
            consistency_ratio = consistent / total_seen
            print(f"\n在见过骨架的分子中，标签一致率: {consistency_ratio:.2%}")

            if consistency_ratio > 0.8:
                print("   🚨 警告: 超过80%的'见过骨架'分子标签也一致！")
                print("      模型可能只是在记忆 '骨架→标签' 的映射")

        return {
            'not_seen': not_seen,
            'consistent': consistent,
            'inconsistent': inconsistent,
            'consistency_ratio': consistent / total_seen if total_seen > 0 else 0
        }

    def _calculate_similarity(
        self,
        train_smiles: List[str],
        test_smiles: List[str],
        sample_size: int = 100
    ) -> Dict:
        """计算Tanimoto相似度"""

        # 采样
        if len(test_smiles) > sample_size:
            np.random.seed(42)
            test_sample = list(np.random.choice(test_smiles, sample_size, replace=False))
        else:
            test_sample = test_smiles

        # 生成训练集指纹
        print("   生成训练集指纹...")
        train_fps = []
        for smi in train_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                train_fps.append(fp)

        print(f"   有效训练集指纹: {len(train_fps)}/{len(train_smiles)}")

        # 计算测试集相似度
        print("   计算测试集相似度...")
        max_sims = []
        high_sim_count = 0

        for test_smi in test_sample:
            test_mol = Chem.MolFromSmiles(test_smi)
            if test_mol is None:
                continue
            test_fp = AllChem.GetMorganFingerprintAsBitVect(test_mol, 2, 2048)

            # 计算与所有训练集的相似度
            sims = [DataStructs.TanimotoSimilarity(test_fp, train_fp)
                   for train_fp in train_fps]

            if sims:
                max_sim = max(sims)
                max_sims.append(max_sim)

                if max_sim > 0.8:
                    high_sim_count += 1

        return {
            'avg_max_sim': float(np.mean(max_sims)) if max_sims else 0.0,
            'high_sim_count': high_sim_count,
            'high_sim_ratio': high_sim_count / len(max_sims) if max_sims else 0.0,
            'sample_size': len(max_sims)
        }

    def _save_report(self, results: Dict, save_dir: Path):
        """保存JSON报告"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 准备可序列化的数据
        report = {
            'scaffold_type': results['scaffold_type'],
            'risk_level': results['risk_level'],
            'train_test_overlap_ratio': results['train_test_overlap_ratio'],
            'train_val_overlap_ratio': results['train_val_overlap_ratio'],
            'test_seen_scaffold_ratio': results['test_seen_scaffold_ratio'],
            'train_unique_scaffolds': results['train_unique_scaffolds'],
            'val_unique_scaffolds': results['val_unique_scaffolds'],
            'test_unique_scaffolds': results['test_unique_scaffolds'],
            'overlap_scaffolds': results['overlap_scaffolds'],
            'similarity_results': results['similarity_results'],
            'label_analysis': results['label_analysis']
        }

        report_path = save_dir / "scaffold_leakage_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✅ 报告已保存到: {report_path}")

    def _plot_scaffold_distribution(self, results: Dict, save_dir: Path):
        """绘制骨架分布可视化"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左图：骨架数量对比
        ax1 = axes[0]
        datasets = ['训练集', '验证集', '测试集']
        scaffold_counts = [
            results['train_unique_scaffolds'],
            results['val_unique_scaffolds'],
            results['test_unique_scaffolds']
        ]

        bars = ax1.bar(datasets, scaffold_counts, color=['steelblue', 'coral', 'lightgreen'])
        ax1.set_ylabel('唯一骨架数', fontsize=12)
        ax1.set_title('骨架分布对比', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        for bar, count in zip(bars, scaffold_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=11)

        # 右图：重叠率
        ax2 = axes[1]
        comparisons = ['训练 vs 验证', '训练 vs 测试', '验证 vs 测试']
        overlap_ratios = [
            results['train_val_overlap_ratio'] * 100,
            results['train_test_overlap_ratio'] * 100,
            0  # Val vs Test 暂不展示
        ]

        colors = ['lightblue', 'orangered', 'lightgray']
        bars = ax2.barh(comparisons, overlap_ratios, color=colors)
        ax2.set_xlabel('重叠率 (%)', fontsize=12)
        ax2.set_title('骨架重叠分析', fontsize=13, fontweight='bold')
        ax2.set_xlim([0, 100])
        ax2.grid(axis='x', alpha=0.3)

        # 添加风险线
        ax2.axvline(70, color='red', linestyle='--', linewidth=2, alpha=0.7, label='严重 (70%)')
        ax2.axvline(50, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='高风险 (50%)')
        ax2.axvline(30, color='yellow', linestyle='--', linewidth=2, alpha=0.7, label='中等 (30%)')
        ax2.legend(loc='lower right', fontsize=9)

        for bar, ratio in zip(bars, overlap_ratios):
            width = bar.get_width()
            if width > 0:
                ax2.text(width + 2, bar.get_y() + bar.get_height()/2.,
                        f'{ratio:.1f}%',
                        ha='left', va='center', fontsize=11)

        plt.tight_layout()

        save_path = save_dir / "scaffold_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 可视化已保存到: {save_path}")


def perform_scaffold_split_improved(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    scaffold_type: str = 'generic',
    max_scaffold_size: int = 20  # 新增：限制最大骨架大小
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    改进的Scaffold Split - 支持大骨架拆分，精确控制比例

    Parameters:
    -----------
    max_scaffold_size: int
        最大骨架大小，超过此大小的骨架会被拆分
    """

    print("\n" + "="*80)
    print("🔬 执行改进版Scaffold Split（支持大骨架拆分）")
    print("="*80)

    analyzer = ScaffoldAnalyzer(scaffold_type=scaffold_type)

    # 1. 提取所有骨架
    print("\n[1/5] 提取骨架...")
    scaffolds_dict = defaultdict(list)

    for idx, row in df.iterrows():
        scaffold = analyzer.get_scaffold(row['smiles'])
        if scaffold:
            scaffolds_dict[scaffold].append(idx)

    print(f"总分子数: {len(df)}")
    print(f"唯一骨架数: {len(scaffolds_dict)}")
    print(f"平均每个骨架的分子数: {len(df)/len(scaffolds_dict):.2f}")

    # 2. 识别和处理大骨架
    print(f"\n[2/5] 处理大骨架(>{max_scaffold_size})...")
    large_scaffolds = {}
    normal_scaffolds = {}

    for scaffold, indices in scaffolds_dict.items():
        if len(indices) > max_scaffold_size:
            large_scaffolds[scaffold] = indices
        else:
            normal_scaffolds[scaffold] = indices

    print(f"大骨架数: {len(large_scaffolds)}")
    print(f"正常骨架数: {len(normal_scaffolds)}")

    # 显示大骨架信息
    if large_scaffolds:
        print("\n📊 大骨架详情:")
        for scaffold, indices in sorted(large_scaffolds.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
            print(f"  - 骨架 {scaffold[:30]}...: {len(indices)} 个分子")

    # 3. 重新组织数据：拆分大骨架
    print("\n[3/5] 重新组织数据（拆分大骨架）...")
    all_groups = []

    # 添加正常骨架
    for scaffold, indices in normal_scaffolds.items():
        all_groups.append(indices)

    # 拆分大骨架为多个小组
    large_group_count = 0
    for scaffold, indices in large_scaffolds.items():
        # 随机打乱大骨架内的分子
        np.random.seed(random_state)
        np.random.shuffle(indices)

        # 计算合适的组数（每组约10-20个分子）
        group_size = max(10, min(20, len(indices) // 3))
        num_groups = max(2, len(indices) // group_size)

        # 分成多个组
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = start_idx + group_size if i < num_groups - 1 else len(indices)
            chunk = indices[start_idx:end_idx]
            if len(chunk) > 0:
                all_groups.append(chunk)
                large_group_count += 1

    print(f"拆分后总组数: {len(all_groups)}")
    print(f"其中大骨架拆分出的组数: {large_group_count}")

    # 4. 按组大小排序并分配
    print("\n[4/5] 按比例分配...")
    all_groups.sort(key=len, reverse=True)

    train_indices = []
    val_indices = []
    test_indices = []

    train_target = int(len(df) * (1 - test_size - val_size))
    val_target = int(len(df) * val_size)
    test_target = int(len(df) * test_size)

    print(f"目标分配:")
    print(f"  训练集: {train_target} ({train_target/len(df)*100:.1f}%)")
    print(f"  验证集: {val_target} ({val_target/len(df)*100:.1f}%)")
    print(f"  测试集: {test_target} ({test_target/len(df)*100:.1f}%)")

    train_count = 0
    val_count = 0
    test_count = 0

    # 优先分配大组到训练集
    for group in all_groups:
        if train_count < train_target:
            train_indices.extend(group)
            train_count += len(group)
        elif val_count < val_target:
            val_indices.extend(group)
            val_count += len(group)
        else:
            test_indices.extend(group)
            test_count += len(group)

    # 5. 精确调整比例
    print("\n[5/5] 精确调整比例...")

    # 如果训练集过多，移动一些到验证集或测试集
    while train_count > train_target + 50 and val_count < val_target:
        # 从训练集移动一个小组到验证集
        if train_indices:
            move_size = min(20, train_count - train_target)
            moved = train_indices[-move_size:]
            train_indices = train_indices[:-move_size]
            val_indices.extend(moved)
            train_count -= len(moved)
            val_count += len(moved)

    while train_count > train_target + 50 and test_count < test_target:
        # 从训练集移动一个小组到测试集
        if train_indices:
            move_size = min(20, train_count - train_target)
            moved = train_indices[-move_size:]
            train_indices = train_indices[:-move_size]
            test_indices.extend(moved)
            train_count -= len(moved)
            test_count += len(moved)

    # 打乱
    np.random.seed(random_state)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    print(f"\n✅ 改进版Scaffold Split完成:")
    print(f"   训练集: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%) - 目标: {train_target}")
    print(f"   验证集: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%) - 目标: {val_target}")
    print(f"   测试集: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%) - 目标: {test_target}")

    # 验证骨架不重叠
    print("\n🔍 验证骨架不重叠...")
    train_scaffolds = set(analyzer.get_scaffold(s) for s in train_df['smiles'] if analyzer.get_scaffold(s))
    test_scaffolds = set(analyzer.get_scaffold(s) for s in test_df['smiles'] if analyzer.get_scaffold(s))
    val_scaffolds = set(analyzer.get_scaffold(s) for s in val_df['smiles'] if analyzer.get_scaffold(s))

    train_test_overlap = train_scaffolds & test_scaffolds
    train_val_overlap = train_scaffolds & val_scaffolds
    val_test_overlap = val_scaffolds & test_scaffolds

    print(f"   训练集骨架: {len(train_scaffolds)}")
    print(f"   验证集骨架: {len(val_scaffolds)}")
    print(f"   测试集骨架: {len(test_scaffolds)}")
    print(f"   训练-测试重叠: {len(train_test_overlap)}")
    print(f"   训练-验证重叠: {len(train_val_overlap)}")
    print(f"   验证-测试重叠: {len(val_test_overlap)}")

    if len(train_test_overlap) == 0:
        print("\n   ✅ 完美！训练集和测试集骨架完全不重叠")
    else:
        print(f"\n   ⚠️  警告：训练集和测试集有 {len(train_test_overlap)} 个重叠骨架")

    print("="*80 + "\n")

    return train_df, val_df, test_df


def quick_scaffold_analysis(
    train_path: str,
    val_path: str,
    test_path: str,
    save_dir: Optional[str] = None
) -> Dict:
    """快速分析现有数据集"""
    print("📂 加载数据集...")
    train_df = smart_read_csv(Path(train_path))
    val_df = smart_read_csv(Path(val_path))
    test_df = smart_read_csv(Path(test_path))

    analyzer = ScaffoldAnalyzer(scaffold_type='generic')

    results = analyzer.analyze_dataset_leakage(
        train_df, val_df, test_df,
        save_dir=Path(save_dir) if save_dir else None
    )

    return results


if __name__ == "__main__":
    print("✅ 改进版Scaffold Analyzer模块已加载!")
    print("\n📚 主要改进:")
    print("  - 支持大骨架拆分 (max_scaffold_size参数)")
    print("  - 精确控制8:1:1比例")
    print("  - 更好的比例调整算法")
    print("\n🚀 使用示例:")
    print("  train_df, val_df, test_df = perform_scaffold_split_improved(df, test_size=0.1, val_size=0.1)")