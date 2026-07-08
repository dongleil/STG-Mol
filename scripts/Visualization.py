"""
================================================================================
Step 3: 化学空间可视化分析 (Chemical Space Visualization) - 修复版
================================================================================
"""

import os
import warnings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit import DataStructs

warnings.filterwarnings('ignore')

# 尝试导入UMAP
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("提示: 未安装umap-learn，将跳过UMAP分析。安装命令: pip install umap-learn")

# 论文级别图表设置
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'Arial',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ==================== 配置区域 ====================
TRAIN_FILE = 'data/processed/nlrp3/train.csv'
VALID_FILE = 'data/processed/nlrp3/val.csv'
TEST_FILE = 'data/processed/nlrp3/test.csv'
SMILES_COL = 'smiles_standardized'

OUTPUT_DIR = 'results/chemical_space'
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42


class ChemicalSpaceAnalyzer:
    """化学空间分析器"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.datasets = {}
        self.fingerprints = {}
        self.descriptors = {}

    def add_dataset(self, name, smiles_list):
        """添加数据集"""
        mols = []
        valid_smiles = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)
                valid_smiles.append(smi)

        self.datasets[name] = {
            'smiles': valid_smiles,
            'mols': mols,
            'n_molecules': len(mols)
        }
        print("[{}] 添加 {} 分子".format(name, len(mols)))

    def compute_fingerprints(self, fp_type='morgan', radius=2, n_bits=2048):
        """计算分子指纹"""
        print("\n计算 {} 指纹 (radius={}, bits={})...".format(fp_type.upper(), radius, n_bits))

        for name, data in self.datasets.items():
            fps = []
            for mol in data['mols']:
                if fp_type == 'morgan':
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                elif fp_type == 'maccs':
                    fp = MACCSkeys.GenMACCSKeys(mol)
                elif fp_type == 'rdkit':
                    fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
                else:
                    raise ValueError("Unknown fingerprint type: {}".format(fp_type))

                arr = np.zeros((n_bits if fp_type != 'maccs' else 167,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)

            self.fingerprints[name] = np.array(fps)
            print("  [{}] 指纹维度: {}".format(name, self.fingerprints[name].shape))

    def compute_descriptors(self):
        """计算分子描述符"""
        print("\n计算分子描述符...")

        descriptor_names = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds',
                            'nRings', 'nAromRings', 'FractionCSP3', 'nHeavyAtoms']

        for name, data in self.datasets.items():
            desc_list = []
            for mol in data['mols']:
                desc = [
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Descriptors.NumRotatableBonds(mol),
                    Descriptors.RingCount(mol),
                    Descriptors.NumAromaticRings(mol),
                    Descriptors.FractionCSP3(mol),
                    Descriptors.HeavyAtomCount(mol),
                ]
                desc_list.append(desc)

            self.descriptors[name] = pd.DataFrame(desc_list, columns=descriptor_names)
            print("  [{}] 描述符维度: {}".format(name, self.descriptors[name].shape))

    def run_pca(self, data_type='fingerprint', n_components=2):
        """PCA降维"""
        print("\n运行PCA ({})...".format(data_type))

        if data_type == 'fingerprint':
            all_data = np.vstack([self.fingerprints[name] for name in self.datasets.keys()])
        else:
            all_data = pd.concat([self.descriptors[name] for name in self.datasets.keys()]).values
            scaler = StandardScaler()
            all_data = scaler.fit_transform(all_data)

        pca = PCA(n_components=n_components, random_state=self.random_state)
        transformed = pca.fit_transform(all_data)

        results = {}
        idx = 0
        for name, data in self.datasets.items():
            n = data['n_molecules']
            results[name] = transformed[idx:idx + n]
            idx += n

        explained_var = pca.explained_variance_ratio_
        print("  解释方差: PC1={:.1f}%, PC2={:.1f}%".format(explained_var[0] * 100, explained_var[1] * 100))

        return results, explained_var, pca

    def run_tsne(self, data_type='fingerprint', perplexity=30, n_iter=1000):
        """t-SNE降维 - 修复版，使用n_jobs=1避免多线程问题"""
        print("\n运行t-SNE ({}, perplexity={})...".format(data_type, perplexity))

        if data_type == 'fingerprint':
            all_data = np.vstack([self.fingerprints[name] for name in self.datasets.keys()])
        else:
            all_data = pd.concat([self.descriptors[name] for name in self.datasets.keys()]).values
            scaler = StandardScaler()
            all_data = scaler.fit_transform(all_data)

        # 先用PCA降维到50维
        if all_data.shape[1] > 50:
            pca = PCA(n_components=50, random_state=self.random_state)
            all_data = pca.fit_transform(all_data)

        # 关键修复: 添加 n_jobs=1 避免多线程问题
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=self.random_state,
            init='pca',
            learning_rate='auto',
            n_jobs=1  # 强制单线程
        )
        transformed = tsne.fit_transform(all_data)

        results = {}
        idx = 0
        for name, data in self.datasets.items():
            n = data['n_molecules']
            results[name] = transformed[idx:idx + n]
            idx += n

        return results

    def run_umap(self, data_type='fingerprint', n_neighbors=15, min_dist=0.1):
        """UMAP降维"""
        if not HAS_UMAP:
            print("UMAP未安装，跳过...")
            return None

        print("\n运行UMAP ({}, n_neighbors={})...".format(data_type, n_neighbors))

        if data_type == 'fingerprint':
            all_data = np.vstack([self.fingerprints[name] for name in self.datasets.keys()])
        else:
            all_data = pd.concat([self.descriptors[name] for name in self.datasets.keys()]).values
            scaler = StandardScaler()
            all_data = scaler.fit_transform(all_data)

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=self.random_state,
            metric='jaccard' if data_type == 'fingerprint' else 'euclidean',
            n_jobs=1  # 强制单线程
        )
        transformed = reducer.fit_transform(all_data)

        results = {}
        idx = 0
        for name, data in self.datasets.items():
            n = data['n_molecules']
            results[name] = transformed[idx:idx + n]
            idx += n

        return results

    def calculate_coverage_overlap(self, coords_dict, grid_size=50):
        """计算化学空间覆盖和重叠"""
        all_coords = np.vstack(list(coords_dict.values()))
        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()

        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= 0.05 * x_range
        x_max += 0.05 * x_range
        y_min -= 0.05 * y_range
        y_max += 0.05 * y_range

        coverage_grids = {}
        for name, coords in coords_dict.items():
            grid = np.zeros((grid_size, grid_size), dtype=bool)
            for x, y in coords:
                xi = min(int((x - x_min) / (x_max - x_min) * grid_size), grid_size - 1)
                yi = min(int((y - y_min) / (y_max - y_min) * grid_size), grid_size - 1)
                grid[xi, yi] = True
            coverage_grids[name] = grid

        total_cells = grid_size * grid_size
        results = {'coverage': {}, 'overlap': {}}

        for name, grid in coverage_grids.items():
            results['coverage'][name] = grid.sum() / total_cells * 100

        names = list(coverage_grids.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i + 1:]:
                overlap = (coverage_grids[name1] & coverage_grids[name2]).sum()
                union = (coverage_grids[name1] | coverage_grids[name2]).sum()
                jaccard = overlap / union * 100 if union > 0 else 0
                results['overlap']['{}_vs_{}'.format(name1, name2)] = {
                    'overlap_cells': overlap,
                    'jaccard_similarity': jaccard
                }

        return results


def plot_chemical_space_comprehensive(analyzer, pca_results, tsne_results, umap_results,
                                      explained_var, save_path):
    """绘制综合化学空间可视化"""
    n_plots = 3 if umap_results else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5.5))

    colors = {'Train': '#3498db', 'Valid': '#e74c3c', 'Test': '#2ecc71'}
    markers = {'Train': 'o', 'Valid': 's', 'Test': '^'}

    # PCA
    ax1 = axes[0]
    for name in ['Train', 'Valid', 'Test']:
        if name in pca_results:
            coords = pca_results[name]
            ax1.scatter(coords[:, 0], coords[:, 1], c=colors[name], marker=markers[name],
                        s=20, alpha=0.6, label='{} (n={})'.format(name, len(coords)),
                        edgecolors='white', linewidth=0.3)

    ax1.set_xlabel('PC1 ({:.1f}%)'.format(explained_var[0] * 100))
    ax1.set_ylabel('PC2 ({:.1f}%)'.format(explained_var[1] * 100))
    ax1.set_title('(a) PCA of Morgan Fingerprints', fontweight='bold')
    ax1.legend(frameon=False, loc='best')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # t-SNE
    ax2 = axes[1]
    for name in ['Train', 'Valid', 'Test']:
        if name in tsne_results:
            coords = tsne_results[name]
            ax2.scatter(coords[:, 0], coords[:, 1], c=colors[name], marker=markers[name],
                        s=20, alpha=0.6, label='{} (n={})'.format(name, len(coords)),
                        edgecolors='white', linewidth=0.3)

    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.set_title('(b) t-SNE of Morgan Fingerprints', fontweight='bold')
    ax2.legend(frameon=False, loc='best')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # UMAP
    if umap_results:
        ax3 = axes[2]
        for name in ['Train', 'Valid', 'Test']:
            if name in umap_results:
                coords = umap_results[name]
                ax3.scatter(coords[:, 0], coords[:, 1], c=colors[name], marker=markers[name],
                            s=20, alpha=0.6, label='{} (n={})'.format(name, len(coords)),
                            edgecolors='white', linewidth=0.3)

        ax3.set_xlabel('UMAP 1')
        ax3.set_ylabel('UMAP 2')
        ax3.set_title('(c) UMAP of Morgan Fingerprints', fontweight='bold')
        ax3.legend(frameon=False, loc='best')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("化学空间综合图已保存: {}".format(save_path))


def plot_density_comparison(pca_results, tsne_results, save_path):
    """绘制密度对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = {'Train': '#3498db', 'Valid': '#e74c3c', 'Test': '#2ecc71'}

    for idx, name in enumerate(['Train', 'Valid', 'Test']):
        ax = axes[0, idx]
        if name in pca_results:
            coords = pca_results[name]
            try:
                from scipy.stats import gaussian_kde
                xy = np.vstack([coords[:, 0], coords[:, 1]])
                z = gaussian_kde(xy)(xy)
                scatter = ax.scatter(coords[:, 0], coords[:, 1], c=z, s=15, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, ax=ax, label='Density')
            except:
                ax.scatter(coords[:, 0], coords[:, 1], c=colors[name], s=15, alpha=0.5)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('PCA - {}'.format(name))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for idx, name in enumerate(['Train', 'Valid', 'Test']):
        ax = axes[1, idx]
        if name in tsne_results:
            coords = tsne_results[name]
            try:
                from scipy.stats import gaussian_kde
                xy = np.vstack([coords[:, 0], coords[:, 1]])
                z = gaussian_kde(xy)(xy)
                scatter = ax.scatter(coords[:, 0], coords[:, 1], c=z, s=15, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, ax=ax, label='Density')
            except:
                ax.scatter(coords[:, 0], coords[:, 1], c=colors[name], s=15, alpha=0.5)

        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('t-SNE - {}'.format(name))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("密度对比图已保存: {}".format(save_path))


def plot_descriptor_space(analyzer, save_path):
    """绘制基于描述符的化学空间"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = {'Train': '#3498db', 'Valid': '#e74c3c', 'Test': '#2ecc71'}

    pairs = [
        ('MW', 'LogP'),
        ('TPSA', 'LogP'),
        ('HBD', 'HBA'),
        ('MW', 'TPSA'),
        ('RotBonds', 'nRings'),
        ('FractionCSP3', 'nAromRings')
    ]

    for idx, (x_prop, y_prop) in enumerate(pairs):
        ax = axes[idx // 3, idx % 3]

        for name in ['Train', 'Valid', 'Test']:
            if name in analyzer.descriptors:
                df = analyzer.descriptors[name]
                ax.scatter(df[x_prop], df[y_prop], c=colors[name], s=15, alpha=0.5, label=name)

        ax.set_xlabel(x_prop)
        ax.set_ylabel(y_prop)
        ax.set_title('{} vs {}'.format(x_prop, y_prop))
        ax.legend(frameon=False, loc='best', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("描述符空间图已保存: {}".format(save_path))


def plot_coverage_analysis(coverage_results, save_path):
    """绘制覆盖率分析图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    names = list(coverage_results['coverage'].keys())
    coverages = [coverage_results['coverage'][name] for name in names]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    bars = ax1.bar(names, coverages, color=colors[:len(names)], edgecolor='black', linewidth=1)
    ax1.set_ylabel('Chemical Space Coverage (%)')
    ax1.set_title('(a) Chemical Space Coverage by Dataset')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    for bar, cov in zip(bars, coverages):
        ax1.annotate('{:.1f}%'.format(cov), xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2 = axes[1]
    n = len(names)
    overlap_matrix = np.zeros((n, n))

    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i == j:
                overlap_matrix[i, j] = 100
            elif i < j:
                key = '{}_vs_{}'.format(name1, name2)
                if key in coverage_results['overlap']:
                    overlap_matrix[i, j] = coverage_results['overlap'][key]['jaccard_similarity']
                    overlap_matrix[j, i] = overlap_matrix[i, j]

    im = ax2.imshow(overlap_matrix, cmap='YlOrRd', vmin=0, vmax=100)
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(names)
    ax2.set_yticklabels(names)
    ax2.set_title('(b) Chemical Space Overlap (Jaccard %)')

    for i in range(n):
        for j in range(n):
            text = '{:.1f}'.format(overlap_matrix[i, j]) if i != j else '-'
            color = 'white' if overlap_matrix[i, j] > 50 else 'black'
            ax2.text(j, i, text, ha='center', va='center', fontsize=11, color=color)

    plt.colorbar(im, ax=ax2, label='Jaccard Similarity (%)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("覆盖率分析图已保存: {}".format(save_path))


def generate_summary_report(analyzer, pca_results, tsne_results, coverage_results, explained_var, save_path):
    """生成汇总报告"""
    report = []
    for name, data in analyzer.datasets.items():
        row = {
            'Dataset': name,
            'N_Molecules': data['n_molecules'],
            'Coverage(%)': "{:.2f}".format(coverage_results['coverage'].get(name, 0)),
        }
        report.append(row)

    df_report = pd.DataFrame(report)

    overlap_rows = []
    for key, value in coverage_results['overlap'].items():
        overlap_rows.append({
            'Comparison': key,
            'Jaccard_Similarity(%)': "{:.2f}".format(value['jaccard_similarity']),
            'Overlap_Cells': value['overlap_cells']
        })

    df_overlap = pd.DataFrame(overlap_rows)

    with pd.ExcelWriter(save_path.replace('.csv', '.xlsx')) as writer:
        df_report.to_excel(writer, sheet_name='Coverage', index=False)
        df_overlap.to_excel(writer, sheet_name='Overlap', index=False)

    df_report.to_csv(save_path, index=False)

    print("汇总报告已保存: {}".format(save_path))
    return df_report, df_overlap


def print_analysis_summary(analyzer, coverage_results, explained_var):
    """打印分析汇总"""
    print("\n" + "=" * 80)
    print("化学空间分析报告 (Chemical Space Analysis Report)")
    print("=" * 80)

    print("\n【方法说明】")
    print("  - 指纹: Morgan/ECFP4 (radius=2, 2048 bits) [Rogers & Hahn, 2010]")
    print("  - PCA: 主成分分析 [线性降维]")
    print("  - t-SNE: t分布随机邻域嵌入 [van der Maaten & Hinton, 2008]")
    if HAS_UMAP:
        print("  - UMAP: 均匀流形近似与投影 [McInnes et al., 2018]")

    print("\n【PCA解释方差】")
    print("  PC1: {:.2f}%".format(explained_var[0] * 100))
    print("  PC2: {:.2f}%".format(explained_var[1] * 100))
    print("  累计: {:.2f}%".format((explained_var[0] + explained_var[1]) * 100))

    print("\n【化学空间覆盖率】")
    for name, cov in coverage_results['coverage'].items():
        print("  {}: {:.2f}%".format(name, cov))

    print("\n【数据集间重叠 (Jaccard Similarity)】")
    for key, value in coverage_results['overlap'].items():
        print("  {}: {:.2f}%".format(key, value['jaccard_similarity']))

    print("\n【解读】")
    train_test_overlap = coverage_results['overlap'].get('Train_vs_Test', {}).get('jaccard_similarity', 0)
    if train_test_overlap < 30:
        print("  Train与Test化学空间重叠较低，Scaffold Split有效分离了化学空间")
    else:
        print("  Train与Test化学空间有一定重叠，但仍属正常范围")

    print("=" * 80)


def main():
    print("=" * 80)
    print("Step 3: 化学空间可视化分析")
    print("=" * 80)

    print("\n[Step 1] 加载数据...")
    train_df = pd.read_csv(TRAIN_FILE)
    valid_df = pd.read_csv(VALID_FILE)
    test_df = pd.read_csv(TEST_FILE)

    train_smiles = train_df[SMILES_COL].dropna().tolist()
    valid_smiles = valid_df[SMILES_COL].dropna().tolist()
    test_smiles = test_df[SMILES_COL].dropna().tolist()

    print("\n[Step 2] 初始化分析器...")
    analyzer = ChemicalSpaceAnalyzer(random_state=RANDOM_STATE)
    analyzer.add_dataset('Train', train_smiles)
    analyzer.add_dataset('Valid', valid_smiles)
    analyzer.add_dataset('Test', test_smiles)

    print("\n[Step 3] 计算分子表示...")
    analyzer.compute_fingerprints(fp_type='morgan', radius=2, n_bits=2048)
    analyzer.compute_descriptors()

    print("\n[Step 4] 降维分析...")
    pca_results, explained_var, pca_model = analyzer.run_pca(data_type='fingerprint')
    tsne_results = analyzer.run_tsne(data_type='fingerprint', perplexity=30)
    umap_results = analyzer.run_umap(data_type='fingerprint') if HAS_UMAP else None

    print("\n[Step 5] 计算化学空间覆盖...")
    coverage_results = analyzer.calculate_coverage_overlap(tsne_results, grid_size=50)

    print_analysis_summary(analyzer, coverage_results, explained_var)

    print("\n[Step 6] 生成可视化...")
    plot_chemical_space_comprehensive(
        analyzer, pca_results, tsne_results, umap_results, explained_var,
        '{}/chemical_space_comprehensive.png'.format(OUTPUT_DIR)
    )
    plot_density_comparison(pca_results, tsne_results, '{}/density_comparison.png'.format(OUTPUT_DIR))
    plot_descriptor_space(analyzer, '{}/descriptor_space.png'.format(OUTPUT_DIR))
    plot_coverage_analysis(coverage_results, '{}/coverage_analysis.png'.format(OUTPUT_DIR))

    print("\n[Step 7] 生成报告...")
    df_report, df_overlap = generate_summary_report(
        analyzer, pca_results, tsne_results, coverage_results, explained_var,
        '{}/chemical_space_summary.csv'.format(OUTPUT_DIR)
    )

    print("\n【覆盖率汇总】")
    print(df_report.to_string(index=False))

    print("\n【重叠分析】")
    print(df_overlap.to_string(index=False))

    print("\n" + "=" * 80)
    print("Step 3 分析完成! 输出文件:")
    print("  目录: {}/".format(OUTPUT_DIR))
    print("  - chemical_space_comprehensive.png  (PCA/t-SNE/UMAP综合图)")
    print("  - density_comparison.png            (密度对比图)")
    print("  - descriptor_space.png              (描述符空间图)")
    print("  - coverage_analysis.png             (覆盖率分析图)")
    print("  - chemical_space_summary.csv        (汇总报告)")
    print("  - chemical_space_summary.xlsx       (Excel报告)")
    print("=" * 80)

    return analyzer, pca_results, tsne_results, coverage_results


if __name__ == "__main__":
    results = main()