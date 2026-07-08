"""
================================================================================
Step 2: Lipinski五规则与药物相似性分析
================================================================================

分析内容：
1. Lipinski's Rule of Five (RO5) - 口服药物相似性
2. Veber's Rules - 口服生物利用度
3. 理化性质分布分析
4. 数据集间的性质对比

文献支撑：
[1] Lipinski CA, et al. Experimental and computational approaches to estimate
    solubility and permeability in drug discovery and development settings.
    Adv Drug Deliv Rev. 2001;46(1-3):3-26.
    - 定义了著名的"Rule of Five"
    - MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10

[2] Veber DF, et al. Molecular properties that influence the oral bioavailability
    of drug candidates.
    J Med Chem. 2002;45(12):2615-2623.
    - 定义了口服生物利用度规则
    - RotBonds ≤ 10, TPSA ≤ 140 Å²

[3] Ghose AK, et al. A knowledge-based approach in designing combinatorial or
    medicinal chemistry libraries for drug discovery.
    J Comb Chem. 1999;1(1):55-68.
    - 定义了Ghose规则
    - 160 ≤ MW ≤ 480, -0.4 ≤ LogP ≤ 5.6, 40 ≤ MR ≤ 130, 20 ≤ atoms ≤ 70

依赖: pip install rdkit pandas numpy matplotlib seaborn scipy
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, MolSurf
import os
import warnings

warnings.filterwarnings('ignore')

# 论文级别图表设置
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'Arial',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ==================== 配置区域 ====================
TRAIN_FILE = 'data/processed/scaffold_split/train.csv'
VALID_FILE = 'data/processed/scaffold_split/val.csv'
TEST_FILE = 'data/processed/scaffold_split/test.csv'
SMILES_COL = 'smiles_standardized'

OUTPUT_DIR = 'results/lipinski_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =================================================


class DrugLikenessAnalyzer:
    """
    药物相似性分析器

    基于以下文献：
    - Lipinski's Rule of Five [Adv Drug Deliv Rev, 2001]
    - Veber's Rules [J Med Chem, 2002]
    - Ghose Filter [J Comb Chem, 1999]
    """

    # 规则定义
    LIPINSKI_RULES = {
        'MW': ('Molecular Weight', '≤', 500, 'Da'),
        'LogP': ('LogP', '≤', 5, ''),
        'HBD': ('H-Bond Donors', '≤', 5, ''),
        'HBA': ('H-Bond Acceptors', '≤', 10, ''),
    }

    VEBER_RULES = {
        'RotBonds': ('Rotatable Bonds', '≤', 10, ''),
        'TPSA': ('TPSA', '≤', 140, 'Å²'),
    }

    def __init__(self, smiles_list, name="Dataset"):
        self.name = name
        self.smiles_list = smiles_list
        self.properties_df = None
        self.n_molecules = 0

        self._calculate_properties()

    def _calculate_properties(self):
        """计算所有理化性质"""
        properties = []

        for smi in self.smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            props = {
                'SMILES': smi,
                # Lipinski描述符
                'MW': Descriptors.MolWt(mol),
                'LogP': Crippen.MolLogP(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                # Veber描述符
                'RotBonds': Descriptors.NumRotatableBonds(mol),
                'TPSA': MolSurf.TPSA(mol),
                # 其他有用描述符
                'nRings': Descriptors.RingCount(mol),
                'nAromRings': Descriptors.NumAromaticRings(mol),
                'nHeavyAtoms': Descriptors.HeavyAtomCount(mol),
                'FractionCSP3': Descriptors.FractionCSP3(mol),
                'MR': Crippen.MolMR(mol),  # Molar Refractivity (Ghose)
            }
            properties.append(props)

        self.properties_df = pd.DataFrame(properties)
        self.n_molecules = len(self.properties_df)
        print(f"[{self.name}] 计算完成: {self.n_molecules} 分子")

    def check_lipinski(self):
        """
        检查Lipinski's Rule of Five

        参考: Lipinski CA, et al. Adv Drug Deliv Rev. 2001

        规则:
        - MW ≤ 500
        - LogP ≤ 5
        - HBD ≤ 5
        - HBA ≤ 10
        允许最多1个违规仍被认为是类药的
        """
        df = self.properties_df.copy()

        # 检查每条规则
        df['MW_pass'] = df['MW'] <= 500
        df['LogP_pass'] = df['LogP'] <= 5
        df['HBD_pass'] = df['HBD'] <= 5
        df['HBA_pass'] = df['HBA'] <= 10

        # 计算违规数
        df['Lipinski_violations'] = 4 - (df['MW_pass'].astype(int) +
                                         df['LogP_pass'].astype(int) +
                                         df['HBD_pass'].astype(int) +
                                         df['HBA_pass'].astype(int))

        # RO5通过: 违规数 ≤ 1
        df['RO5_pass'] = df['Lipinski_violations'] <= 1

        # 统计
        results = {
            'name': self.name,
            'n_molecules': self.n_molecules,
            'rule_pass_count': {
                'MW ≤ 500': df['MW_pass'].sum(),
                'LogP ≤ 5': df['LogP_pass'].sum(),
                'HBD ≤ 5': df['HBD_pass'].sum(),
                'HBA ≤ 10': df['HBA_pass'].sum(),
            },
            'rule_pass_rate': {
                'MW ≤ 500': df['MW_pass'].mean() * 100,
                'LogP ≤ 5': df['LogP_pass'].mean() * 100,
                'HBD ≤ 5': df['HBD_pass'].mean() * 100,
                'HBA ≤ 10': df['HBA_pass'].mean() * 100,
            },
            'violations_distribution': df['Lipinski_violations'].value_counts().sort_index().to_dict(),
            'ro5_pass_count': df['RO5_pass'].sum(),
            'ro5_pass_rate': df['RO5_pass'].mean() * 100,
            'detailed_df': df
        }

        return results

    def check_veber(self):
        """
        检查Veber's Rules (口服生物利用度)

        参考: Veber DF, et al. J Med Chem. 2002

        规则:
        - Rotatable Bonds ≤ 10
        - TPSA ≤ 140 Å²
        """
        df = self.properties_df.copy()

        df['RotBonds_pass'] = df['RotBonds'] <= 10
        df['TPSA_pass'] = df['TPSA'] <= 140
        df['Veber_pass'] = df['RotBonds_pass'] & df['TPSA_pass']

        results = {
            'name': self.name,
            'rule_pass_rate': {
                'RotBonds ≤ 10': df['RotBonds_pass'].mean() * 100,
                'TPSA ≤ 140': df['TPSA_pass'].mean() * 100,
            },
            'veber_pass_rate': df['Veber_pass'].mean() * 100,
        }

        return results

    def check_ghose(self):
        """
        检查Ghose Filter

        参考: Ghose AK, et al. J Comb Chem. 1999

        规则:
        - 160 ≤ MW ≤ 480
        - -0.4 ≤ LogP ≤ 5.6
        - 40 ≤ MR ≤ 130
        - 20 ≤ nAtoms ≤ 70
        """
        df = self.properties_df.copy()

        df['Ghose_MW'] = (df['MW'] >= 160) & (df['MW'] <= 480)
        df['Ghose_LogP'] = (df['LogP'] >= -0.4) & (df['LogP'] <= 5.6)
        df['Ghose_MR'] = (df['MR'] >= 40) & (df['MR'] <= 130)
        df['Ghose_atoms'] = (df['nHeavyAtoms'] >= 20) & (df['nHeavyAtoms'] <= 70)
        df['Ghose_pass'] = df['Ghose_MW'] & df['Ghose_LogP'] & df['Ghose_MR'] & df['Ghose_atoms']

        results = {
            'name': self.name,
            'rule_pass_rate': {
                '160 ≤ MW ≤ 480': df['Ghose_MW'].mean() * 100,
                '-0.4 ≤ LogP ≤ 5.6': df['Ghose_LogP'].mean() * 100,
                '40 ≤ MR ≤ 130': df['Ghose_MR'].mean() * 100,
                '20 ≤ atoms ≤ 70': df['Ghose_atoms'].mean() * 100,
            },
            'ghose_pass_rate': df['Ghose_pass'].mean() * 100,
        }

        return results

    def get_property_statistics(self):
        """获取理化性质统计"""
        props = ['MW', 'LogP', 'HBD', 'HBA', 'RotBonds', 'TPSA', 'nRings', 'FractionCSP3']

        stats_dict = {}
        for prop in props:
            stats_dict[prop] = {
                'mean': self.properties_df[prop].mean(),
                'std': self.properties_df[prop].std(),
                'median': self.properties_df[prop].median(),
                'min': self.properties_df[prop].min(),
                'max': self.properties_df[prop].max(),
                'Q1': self.properties_df[prop].quantile(0.25),
                'Q3': self.properties_df[prop].quantile(0.75),
            }

        return stats_dict


def plot_lipinski_comprehensive(analyzers_dict, save_path):
    """绘制Lipinski规则综合分析图"""
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    names = list(analyzers_dict.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    # 获取所有分析结果
    lipinski_results = {name: analyzer.check_lipinski() for name, analyzer in analyzers_dict.items()}
    veber_results = {name: analyzer.check_veber() for name, analyzer in analyzers_dict.items()}

    # ===== (a) 各规则通过率 =====
    ax1 = fig.add_subplot(gs[0, 0])
    rules = ['MW ≤ 500', 'LogP ≤ 5', 'HBD ≤ 5', 'HBA ≤ 10']
    x = np.arange(len(rules))
    width = 0.2

    for i, name in enumerate(names):
        rates = [lipinski_results[name]['rule_pass_rate'][r] for r in rules]
        ax1.bar(x + i * width, rates, width, label=name, color=colors[i], edgecolor='black', linewidth=0.5)

    ax1.set_ylabel('Pass Rate (%)')
    ax1.set_xticks(x + width * (len(names) - 1) / 2)
    ax1.set_xticklabels(rules, rotation=30, ha='right')
    ax1.set_ylim([0, 105])
    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    ax1.legend(frameon=False, loc='lower left')
    ax1.set_title("(a) Lipinski's Rule-wise Pass Rate")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ===== (b) 违规数分布 =====
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(5)  # 0-4违规

    for i, name in enumerate(names):
        violations = lipinski_results[name]['violations_distribution']
        counts = [violations.get(v, 0) / lipinski_results[name]['n_molecules'] * 100 for v in range(5)]
        ax2.bar(x + i * width, counts, width, label=name, color=colors[i], edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Number of Violations')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_xticks(x + width * (len(names) - 1) / 2)
    ax2.set_xticklabels(['0', '1', '2', '3', '4'])
    ax2.legend(frameon=False)
    ax2.set_title('(b) Lipinski Violations Distribution')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ===== (c) RO5总通过率 =====
    ax3 = fig.add_subplot(gs[0, 2])
    ro5_rates = [lipinski_results[name]['ro5_pass_rate'] for name in names]

    bars = ax3.bar(names, ro5_rates, color=colors[:len(names)], edgecolor='black', linewidth=1)
    ax3.set_ylabel('RO5 Pass Rate (%)')
    ax3.set_ylim([0, 105])
    ax3.axhline(y=90, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax3.text(len(names) - 0.5, 91, '90% threshold', fontsize=9, color='red')
    ax3.set_title("(c) Rule of Five Compliance")
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    for bar, rate in zip(bars, ro5_rates):
        ax3.annotate(f'{rate:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    # ===== (d-i) 理化性质分布 =====
    properties = [
        ('MW', 'Molecular Weight (Da)', 500, None),
        ('LogP', 'LogP', 5, None),
        ('HBD', 'H-Bond Donors', 5, None),
        ('HBA', 'H-Bond Acceptors', 10, None),
        ('TPSA', 'TPSA (Å²)', 140, None),
        ('RotBonds', 'Rotatable Bonds', 10, None),
    ]

    for idx, (prop, label, upper_thresh, lower_thresh) in enumerate(properties):
        row = 1 + idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])

        # 收集所有数据用于确定bins
        all_data = []
        for name, analyzer in analyzers_dict.items():
            all_data.extend(analyzer.properties_df[prop].tolist())

        bins = np.linspace(min(all_data), max(all_data), 40)

        for i, (name, analyzer) in enumerate(analyzers_dict.items()):
            data = analyzer.properties_df[prop]
            ax.hist(data, bins=bins, alpha=0.5, label=name, color=colors[i], edgecolor='white', linewidth=0.3)

        if upper_thresh:
            ax.axvline(x=upper_thresh, color='red', linestyle='--', linewidth=2, label=f'Threshold ({upper_thresh})')

        ax.set_xlabel(label)
        ax.set_ylabel('Frequency')
        ax.set_title(f'({chr(100 + idx)}) {label} Distribution')
        ax.legend(frameon=False, loc='upper right', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Lipinski综合分析图已保存: {save_path}")


def plot_violin_comparison(analyzers_dict, save_path):
    """绘制小提琴图对比"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))

    properties = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'nRings', 'FractionCSP3']
    labels = ['Molecular Weight', 'LogP', 'H-Bond Donors', 'H-Bond Acceptors',
              'TPSA (Å²)', 'Rotatable Bonds', 'Ring Count', 'Fraction CSP³']
    thresholds = [500, 5, 5, 10, 140, 10, None, None]

    names = list(analyzers_dict.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    for idx, (prop, label, thresh) in enumerate(zip(properties, labels, thresholds)):
        ax = axes[idx // 4, idx % 4]

        data_list = [analyzers_dict[name].properties_df[prop].values for name in names]

        parts = ax.violinplot(data_list, positions=range(len(names)), showmeans=True, showmedians=True)

        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        if thresh:
            ax.axhline(y=thresh, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"小提琴图已保存: {save_path}")


def plot_druglikeness_radar(analyzers_dict, save_path):
    """绘制药物相似性雷达图"""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # 指标（归一化到0-100）
    categories = ['RO5 Pass', 'Veber Pass', 'Ghose Pass', 'MW in range', 'LogP in range', 'TPSA in range']
    n_cats = len(categories)

    angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
    angles += angles[:1]  # 闭合

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    for i, (name, analyzer) in enumerate(analyzers_dict.items()):
        lipinski = analyzer.check_lipinski()
        veber = analyzer.check_veber()
        ghose = analyzer.check_ghose()

        values = [
            lipinski['ro5_pass_rate'],
            veber['veber_pass_rate'],
            ghose['ghose_pass_rate'],
            lipinski['rule_pass_rate']['MW ≤ 500'],
            lipinski['rule_pass_rate']['LogP ≤ 5'],
            veber['rule_pass_rate']['TPSA ≤ 140'],
        ]
        values += values[:1]  # 闭合

        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim([0, 105])
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), frameon=False)
    ax.set_title('Drug-likeness Profile Comparison', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"雷达图已保存: {save_path}")


def generate_summary_tables(analyzers_dict, save_path):
    """生成汇总表"""

    # 表1: Lipinski & Veber & Ghose 通过率
    table1_data = []
    for name, analyzer in analyzers_dict.items():
        lipinski = analyzer.check_lipinski()
        veber = analyzer.check_veber()
        ghose = analyzer.check_ghose()

        row = {
            'Dataset': name,
            'N_Molecules': analyzer.n_molecules,
            'MW≤500 (%)': f"{lipinski['rule_pass_rate']['MW ≤ 500']:.1f}",
            'LogP≤5 (%)': f"{lipinski['rule_pass_rate']['LogP ≤ 5']:.1f}",
            'HBD≤5 (%)': f"{lipinski['rule_pass_rate']['HBD ≤ 5']:.1f}",
            'HBA≤10 (%)': f"{lipinski['rule_pass_rate']['HBA ≤ 10']:.1f}",
            'RO5_Pass (%)': f"{lipinski['ro5_pass_rate']:.1f}",
            'Veber_Pass (%)': f"{veber['veber_pass_rate']:.1f}",
            'Ghose_Pass (%)': f"{ghose['ghose_pass_rate']:.1f}",
        }
        table1_data.append(row)

    df_rules = pd.DataFrame(table1_data)

    # 表2: 理化性质统计
    table2_data = []
    for name, analyzer in analyzers_dict.items():
        stats = analyzer.get_property_statistics()
        row = {'Dataset': name}
        for prop in ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds']:
            row[f'{prop}_mean'] = f"{stats[prop]['mean']:.2f}"
            row[f'{prop}_std'] = f"{stats[prop]['std']:.2f}"
        table2_data.append(row)

    df_stats = pd.DataFrame(table2_data)

    # 保存
    with pd.ExcelWriter(save_path.replace('.csv', '.xlsx')) as writer:
        df_rules.to_excel(writer, sheet_name='Drug-likeness_Rules', index=False)
        df_stats.to_excel(writer, sheet_name='Property_Statistics', index=False)

    df_rules.to_csv(save_path, index=False)

    print(f"汇总表已保存: {save_path}")
    print(f"Excel版本: {save_path.replace('.csv', '.xlsx')}")

    return df_rules, df_stats


def print_analysis_report(analyzers_dict):
    """打印分析报告"""
    print("\n" + "=" * 80)
    print("药物相似性分析报告 (Drug-likeness Analysis Report)")
    print("=" * 80)

    for name, analyzer in analyzers_dict.items():
        lipinski = analyzer.check_lipinski()
        veber = analyzer.check_veber()
        ghose = analyzer.check_ghose()
        stats = analyzer.get_property_statistics()

        print(f"\n【{name}】 (n = {analyzer.n_molecules})")
        print("-" * 60)

        print("\n  [Lipinski's Rule of Five - Adv Drug Deliv Rev, 2001]")
        for rule, rate in lipinski['rule_pass_rate'].items():
            print(f"    {rule}: {rate:.1f}%")
        print(f"    → RO5通过率 (≤1 violation): {lipinski['ro5_pass_rate']:.1f}%")

        print(f"\n  [Veber's Rules - J Med Chem, 2002]")
        for rule, rate in veber['rule_pass_rate'].items():
            print(f"    {rule}: {rate:.1f}%")
        print(f"    → Veber通过率: {veber['veber_pass_rate']:.1f}%")

        print(f"\n  [Ghose Filter - J Comb Chem, 1999]")
        print(f"    → Ghose通过率: {ghose['ghose_pass_rate']:.1f}%")

        print(f"\n  [理化性质统计]")
        print(
            f"    MW:       {stats['MW']['mean']:.1f} ± {stats['MW']['std']:.1f} (range: {stats['MW']['min']:.1f}-{stats['MW']['max']:.1f})")
        print(f"    LogP:     {stats['LogP']['mean']:.2f} ± {stats['LogP']['std']:.2f}")
        print(f"    HBD:      {stats['HBD']['mean']:.1f} ± {stats['HBD']['std']:.1f}")
        print(f"    HBA:      {stats['HBA']['mean']:.1f} ± {stats['HBA']['std']:.1f}")
        print(f"    TPSA:     {stats['TPSA']['mean']:.1f} ± {stats['TPSA']['std']:.1f}")
        print(f"    RotBonds: {stats['RotBonds']['mean']:.1f} ± {stats['RotBonds']['std']:.1f}")

    print("\n" + "=" * 80)


# ==================== 主程序 ====================
def main():
    print("=" * 80)
    print("Step 2: Lipinski五规则与药物相似性分析")
    print("=" * 80)

    # 1. 加载数据
    print("\n[Step 1] 加载数据...")

    train_df = pd.read_csv(TRAIN_FILE)
    valid_df = pd.read_csv(VALID_FILE)
    test_df = pd.read_csv(TEST_FILE)

    train_smiles = train_df[SMILES_COL].dropna().tolist()
    valid_smiles = valid_df[SMILES_COL].dropna().tolist()
    test_smiles = test_df[SMILES_COL].dropna().tolist()
    full_smiles = train_smiles + valid_smiles + test_smiles

    print(
        f"  Full: {len(full_smiles)} | Train: {len(train_smiles)} | Valid: {len(valid_smiles)} | Test: {len(test_smiles)}")

    # 2. 创建分析器
    print("\n[Step 2] 计算理化性质...")

    analyzers = {
        'Full': DrugLikenessAnalyzer(full_smiles, "Full"),
        'Train': DrugLikenessAnalyzer(train_smiles, "Train"),
        'Valid': DrugLikenessAnalyzer(valid_smiles, "Valid"),
        'Test': DrugLikenessAnalyzer(test_smiles, "Test"),
    }

    # 3. 打印报告
    print_analysis_report(analyzers)

    # 4. 生成可视化
    print("\n[Step 3] 生成可视化...")

    plot_lipinski_comprehensive(analyzers, f'{OUTPUT_DIR}/lipinski_comprehensive.png')
    plot_violin_comparison(analyzers, f'{OUTPUT_DIR}/property_violin.png')
    plot_druglikeness_radar(analyzers, f'{OUTPUT_DIR}/druglikeness_radar.png')

    # 5. 生成表格
    print("\n[Step 4] 生成汇总表...")
    df_rules, df_stats = generate_summary_tables(analyzers, f'{OUTPUT_DIR}/druglikeness_summary.csv')

    print("\n【汇总表1: 药物相似性规则通过率】")
    print(df_rules.to_string(index=False))

    print("\n【汇总表2: 理化性质统计】")
    print(df_stats.to_string(index=False))

    # 6. 完成
    print("\n" + "=" * 80)
    print("Step 2 分析完成! 输出文件:")
    print(f"  目录: {OUTPUT_DIR}/")
    print("  - lipinski_comprehensive.png   (Lipinski规则综合分析)")
    print("  - property_violin.png          (理化性质小提琴图)")
    print("  - druglikeness_radar.png       (药物相似性雷达图)")
    print("  - druglikeness_summary.csv     (汇总表CSV)")
    print("  - druglikeness_summary.xlsx    (汇总表Excel)")
    print("=" * 80)

    return analyzers


if __name__ == "__main__":
    analyzers = main()