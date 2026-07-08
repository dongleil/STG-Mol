"""
NLRP3 模型分析脚本 - 阈值调整 & 特征重要性
==================================================

功能：
1. 阈值优化分析（Precision-Recall权衡）
2. 特征重要性可视化
3. 结构-活性关系（SAR）分析

使用方法：
python analyze_best_model.py --exp-dir results/training_experiments/Exp006_*

Author: Research Team
Date: 2025-11-08
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# 机器学习
from sklearn.metrics import (
    precision_recall_curve, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
)

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from rdkit.Chem.Draw import IPythonConsole

# 设置
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


# ============================================================================
# 1. 阈值优化分析
# ============================================================================

def analyze_threshold_optimization(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_dir: Path
) -> Dict:
    """
    分析不同阈值下的Precision-Recall权衡

    Args:
        y_true: 真实标签
        y_proba: 预测概率
        save_dir: 保存目录

    Returns:
        包含最优阈值信息的字典
    """
    print("\n" + "=" * 80)
    print("📊 THRESHOLD OPTIMIZATION ANALYSIS")
    print("=" * 80)

    # 计算Precision-Recall曲线
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # 计算F1分数
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)

    # 找到F1最大的阈值
    best_f1_idx = np.argmax(f1_scores)
    best_threshold_f1 = thresholds[best_f1_idx]
    best_precision_f1 = precision[best_f1_idx]
    best_recall_f1 = recall[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]

    # 默认阈值（0.5）的性能
    y_pred_default = (y_proba >= 0.5).astype(int)
    default_precision = precision_score(y_true, y_pred_default)
    default_recall = recall_score(y_true, y_pred_default)
    default_f1 = f1_score(y_true, y_pred_default)

    # 推荐的其他阈值
    thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
    threshold_results = []

    for thresh in thresholds_to_test:
        y_pred = (y_proba >= thresh).astype(int)
        result = {
            'threshold': thresh,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred)
        }
        threshold_results.append(result)

    # 打印结果
    print(f"\n📈 Default Threshold (0.5):")
    print(f"   Precision: {default_precision:.4f}")
    print(f"   Recall:    {default_recall:.4f}")
    print(f"   F1-Score:  {default_f1:.4f}")

    print(f"\n⭐ Optimal Threshold for F1 ({best_threshold_f1:.3f}):")
    print(f"   Precision: {best_precision_f1:.4f}")
    print(f"   Recall:    {best_recall_f1:.4f}")
    print(f"   F1-Score:  {best_f1:.4f}")
    print(f"   Improvement: +{(best_f1 - default_f1) * 100:.2f}%")

    print(f"\n📊 Threshold Comparison Table:")
    print(f"   {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'MCC':<12}")
    print(f"   {'-' * 72}")
    for result in threshold_results:
        print(f"   {result['threshold']:<12.2f} "
              f"{result['precision']:<12.4f} "
              f"{result['recall']:<12.4f} "
              f"{result['f1']:<12.4f} "
              f"{result['accuracy']:<12.4f} "
              f"{result['mcc']:<12.4f}")

    # 可视化1: Precision-Recall曲线
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图: PR曲线
    ax = axes[0]
    ax.plot(recall, precision, linewidth=2, color='steelblue', label='PR Curve')
    ax.scatter(best_recall_f1, best_precision_f1, s=300, c='red', marker='*',
               zorder=5, label=f'Best F1={best_f1:.3f} @ threshold={best_threshold_f1:.3f}')
    ax.scatter(default_recall, default_precision, s=150, c='orange', marker='o',
               zorder=5, label=f'Default @ threshold=0.5')

    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])

    # 右图: 不同阈值下的指标对比
    ax = axes[1]
    thresholds_plot = [r['threshold'] for r in threshold_results]
    precision_plot = [r['precision'] for r in threshold_results]
    recall_plot = [r['recall'] for r in threshold_results]
    f1_plot = [r['f1'] for r in threshold_results]

    ax.plot(thresholds_plot, precision_plot, marker='o', linewidth=2,
            label='Precision', color='blue')
    ax.plot(thresholds_plot, recall_plot, marker='s', linewidth=2,
            label='Recall', color='green')
    ax.plot(thresholds_plot, f1_plot, marker='^', linewidth=2,
            label='F1-Score', color='red')

    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default (0.5)')
    ax.axvline(x=best_threshold_f1, color='orange', linestyle='--', alpha=0.5,
               label=f'Optimal ({best_threshold_f1:.2f})')

    ax.set_xlabel('Classification Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax.set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.05])

    plt.tight_layout()
    save_path = save_dir / "threshold_optimization_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {save_path}")
    plt.close()

    # 可视化2: 混淆矩阵对比（默认 vs 最优）
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 默认阈值的混淆矩阵
    y_pred_default = (y_proba >= 0.5).astype(int)
    cm_default = confusion_matrix(y_true, y_pred_default)

    ax = axes[0]
    sns.heatmap(cm_default, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
    ax.set_title(f'Confusion Matrix @ Threshold=0.5\n'
                 f'Precision={default_precision:.3f}, Recall={default_recall:.3f}',
                 fontsize=12, fontweight='bold')
    ax.set_xticklabels(['Inactive', 'Active'])
    ax.set_yticklabels(['Inactive', 'Active'])

    # 最优阈值的混淆矩阵
    y_pred_optimal = (y_proba >= best_threshold_f1).astype(int)
    cm_optimal = confusion_matrix(y_true, y_pred_optimal)

    ax = axes[1]
    sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Greens', ax=ax,
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
    ax.set_title(f'Confusion Matrix @ Threshold={best_threshold_f1:.2f}\n'
                 f'Precision={best_precision_f1:.3f}, Recall={best_recall_f1:.3f}',
                 fontsize=12, fontweight='bold')
    ax.set_xticklabels(['Inactive', 'Active'])
    ax.set_yticklabels(['Inactive', 'Active'])

    plt.tight_layout()
    save_path = save_dir / "confusion_matrix_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()

    # 保存详细结果
    results = {
        'default_threshold': {
            'threshold': 0.5,
            'precision': float(default_precision),
            'recall': float(default_recall),
            'f1': float(default_f1)
        },
        'optimal_threshold': {
            'threshold': float(best_threshold_f1),
            'precision': float(best_precision_f1),
            'recall': float(best_recall_f1),
            'f1': float(best_f1)
        },
        'all_thresholds': threshold_results,
        'recommendations': {
            'high_precision': {
                'threshold': 0.7,
                'use_case': 'When experimental cost is high, minimize false positives',
                'metrics': threshold_results[4]
            },
            'balanced': {
                'threshold': float(best_threshold_f1),
                'use_case': 'Optimal F1-score, balanced precision and recall',
                'metrics': {
                    'precision': float(best_precision_f1),
                    'recall': float(best_recall_f1),
                    'f1': float(best_f1)
                }
            },
            'high_recall': {
                'threshold': 0.3,
                'use_case': 'Early screening, ensure not missing active compounds',
                'metrics': threshold_results[0]
            }
        }
    }

    json_path = save_dir / "threshold_analysis_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved: {json_path}")

    return results


# ============================================================================
# 2. 特征重要性分析
# ============================================================================

def analyze_feature_importance(
        model,
        feature_names: List[str],
        save_dir: Path,
        top_n: int = 50
) -> Dict:
    """
    分析并可视化特征重要性

    Args:
        model: 训练好的模型（支持feature_importances_）
        feature_names: 特征名称列表
        save_dir: 保存目录
        top_n: 展示最重要的N个特征

    Returns:
        特征重要性字典
    """
    print("\n" + "=" * 80)
    print("🔬 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    # 获取特征重要性
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        print("⚠️ Model does not support feature importance analysis")
        return {}

    # 排序
    indices = np.argsort(importance)[::-1]

    # Top N特征
    top_indices = indices[:top_n]
    top_importance = importance[top_indices]
    top_features = [feature_names[i] for i in top_indices]

    print(f"\n📊 Top {top_n} Most Important Features:")
    print(f"   {'Rank':<6} {'Feature':<20} {'Importance':<12} {'Cumulative %':<15}")
    print(f"   {'-' * 60}")

    cumulative = 0
    total_importance = importance.sum()
    for i, (feat, imp) in enumerate(zip(top_features, top_importance), 1):
        cumulative += imp
        print(f"   {i:<6} {feat:<20} {imp:<12.6f} {cumulative / total_importance * 100:<15.2f}%")

    # 可视化1: 柱状图（Top N）
    fig, ax = plt.subplots(figsize=(12, 10))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    bars = ax.barh(range(top_n), top_importance, color=colors)

    # 高亮前3名
    bars[0].set_color('orangered')
    bars[1].set_color('orange')
    bars[2].set_color('gold')

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fingerprint Bit / Feature', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Important Features\n'
                 f'(Model: {type(model).__name__})',
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 添加数值标签
    for i, (bar, imp) in enumerate(zip(bars, top_importance)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2,
                f' {imp:.4f}', ha='left', va='center', fontsize=8)

    plt.tight_layout()
    save_path = save_dir / f"feature_importance_top{top_n}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {save_path}")
    plt.close()

    # 可视化2: 累积重要性曲线
    fig, ax = plt.subplots(figsize=(12, 6))

    sorted_importance = importance[indices]
    cumulative_importance = np.cumsum(sorted_importance) / total_importance * 100

    ax.plot(range(1, len(cumulative_importance) + 1), cumulative_importance,
            linewidth=2, color='steelblue')
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.5,
               label='80% Threshold')
    ax.axhline(y=90, color='orange', linestyle='--', alpha=0.5,
               label='90% Threshold')

    # 标注80%和90%所需特征数
    idx_80 = np.argmax(cumulative_importance >= 80)
    idx_90 = np.argmax(cumulative_importance >= 90)

    ax.scatter(idx_80, 80, s=100, c='red', zorder=5)
    ax.text(idx_80, 82, f'{idx_80} features\n(80%)', ha='center', fontsize=9)

    ax.scatter(idx_90, 90, s=100, c='orange', zorder=5)
    ax.text(idx_90, 92, f'{idx_90} features\n(90%)', ha='center', fontsize=9)

    ax.set_xlabel('Number of Features (Ranked by Importance)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Importance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, len(cumulative_importance)])
    ax.set_ylim([0, 105])

    plt.tight_layout()
    save_path = save_dir / "feature_importance_cumulative.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()

    # 可视化3: 特征重要性分布
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(importance, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=importance.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean = {importance.mean():.6f}')
    ax.axvline(x=np.median(importance), color='orange', linestyle='--', linewidth=2,
               label=f'Median = {np.median(importance):.6f}')

    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Feature Importance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / "feature_importance_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()

    # 保存详细结果
    results = {
        'top_features': [
            {
                'rank': i + 1,
                'feature': feat,
                'importance': float(imp),
                'cumulative_percentage': float(cumulative_importance[indices[i]])
            }
            for i, (feat, imp) in enumerate(zip(top_features, top_importance))
        ],
        'statistics': {
            'total_features': int(len(importance)),
            'mean_importance': float(importance.mean()),
            'median_importance': float(np.median(importance)),
            'std_importance': float(importance.std()),
            'features_for_80_percent': int(idx_80),
            'features_for_90_percent': int(idx_90)
        }
    }

    json_path = save_dir / "feature_importance_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved: {json_path}")

    return results


# ============================================================================
# 3. 结构-活性关系（SAR）分析（针对RDKit指纹）
# ============================================================================

def analyze_structure_activity_relationship(
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        smiles_test: List[str],
        save_dir: Path,
        top_n: int = 10
):
    """
    分析结构-活性关系

    Args:
        model: 训练好的模型
        X_test: 测试集特征
        y_test: 测试集标签
        smiles_test: 测试集SMILES
        save_dir: 保存目录
        top_n: 展示top N的预测
    """
    print("\n" + "=" * 80)
    print("🧬 STRUCTURE-ACTIVITY RELATIONSHIP (SAR) ANALYSIS")
    print("=" * 80)

    # 预测
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # 找到高置信度的预测
    # 1. True Positives with high confidence
    tp_mask = (y_test == 1) & (y_pred == 1)
    tp_indices = np.where(tp_mask)[0]
    tp_proba = y_proba[tp_indices]
    tp_smiles = [smiles_test[i] for i in tp_indices]

    # 排序：按预测概率降序
    tp_sorted = sorted(zip(tp_indices, tp_proba, tp_smiles),
                       key=lambda x: x[1], reverse=True)

    print(f"\n✅ Top {min(top_n, len(tp_sorted))} True Positives (High Confidence):")
    print(f"   {'Rank':<6} {'Index':<8} {'Probability':<12} {'SMILES':<50}")
    print(f"   {'-' * 80}")
    for i, (idx, prob, smi) in enumerate(tp_sorted[:top_n], 1):
        print(f"   {i:<6} {idx:<8} {prob:<12.4f} {smi:<50}")

    # 2. False Positives (需要分析为什么预测错误)
    fp_mask = (y_test == 0) & (y_pred == 1)
    fp_indices = np.where(fp_mask)[0]
    fp_proba = y_proba[fp_indices]
    fp_smiles = [smiles_test[i] for i in fp_indices]

    fp_sorted = sorted(zip(fp_indices, fp_proba, fp_smiles),
                       key=lambda x: x[1], reverse=True)

    print(f"\n⚠️ Top {min(top_n, len(fp_sorted))} False Positives (Model Mistakes):")
    print(f"   {'Rank':<6} {'Index':<8} {'Probability':<12} {'SMILES':<50}")
    print(f"   {'-' * 80}")
    for i, (idx, prob, smi) in enumerate(fp_sorted[:top_n], 1):
        print(f"   {i:<6} {idx:<8} {prob:<12.4f} {smi:<50}")

    # 3. False Negatives (活性化合物被遗漏)
    fn_mask = (y_test == 1) & (y_pred == 0)
    fn_indices = np.where(fn_mask)[0]
    fn_proba = y_proba[fn_indices]
    fn_smiles = [smiles_test[i] for i in fn_indices]

    fn_sorted = sorted(zip(fn_indices, fn_proba, fn_smiles),
                       key=lambda x: x[1], ascending=True)

    print(f"\n❌ Top {min(top_n, len(fn_sorted))} False Negatives (Missed Actives):")
    print(f"   {'Rank':<6} {'Index':<8} {'Probability':<12} {'SMILES':<50}")
    print(f"   {'-' * 80}")
    for i, (idx, prob, smi) in enumerate(fn_sorted[:top_n], 1):
        print(f"   {i:<6} {idx:<8} {prob:<12.4f} {smi:<50}")

    # 绘制分子结构（如果RDKit可用）
    try:
        # True Positives
        tp_mols = [Chem.MolFromSmiles(smi) for _, _, smi in tp_sorted[:min(5, len(tp_sorted))]]
        tp_mols = [mol for mol in tp_mols if mol is not None]

        if tp_mols:
            img = Draw.MolsToGridImage(
                tp_mols,
                molsPerRow=5,
                subImgSize=(300, 300),
                legends=[f"TP #{i + 1}\nProb={prob:.3f}"
                         for i, (_, prob, _) in enumerate(tp_sorted[:len(tp_mols)])]
            )
            img_path = save_dir / "top_true_positives.png"
            img.save(img_path)
            print(f"\n✅ Saved molecule images: {img_path}")
    except Exception as e:
        print(f"\n⚠️ Could not generate molecule images: {e}")

    # 保存SAR分析结果
    sar_results = {
        'true_positives': [
            {'rank': i + 1, 'index': int(idx), 'probability': float(prob), 'smiles': smi}
            for i, (idx, prob, smi) in enumerate(tp_sorted[:top_n])
        ],
        'false_positives': [
            {'rank': i + 1, 'index': int(idx), 'probability': float(prob), 'smiles': smi}
            for i, (idx, prob, smi) in enumerate(fp_sorted[:top_n])
        ],
        'false_negatives': [
            {'rank': i + 1, 'index': int(idx), 'probability': float(prob), 'smiles': smi}
            for i, (idx, prob, smi) in enumerate(fn_sorted[:top_n])
        ]
    }

    json_path = save_dir / "sar_analysis_results.json"
    with open(json_path, 'w') as f:
        json.dump(sar_results, f, indent=2)
    print(f"✅ Saved: {json_path}")


# ============================================================================
# 4. 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze NLRP3 model: threshold optimization and feature importance"
    )
    parser.add_argument(
        '--exp-dir',
        type=str,
        required=True,
        help='Experiment directory (e.g., results/training_experiments/Exp006_*)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='LightGBM',
        help='Model name (default: LightGBM)'
    )
    parser.add_argument(
        '--fingerprint',
        type=str,
        default='RDKit',
        help='Fingerprint type (default: RDKit)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=50,
        help='Number of top features to analyze (default: 50)'
    )

    args = parser.parse_args()

    # 查找实验目录
    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        # 尝试通配符匹配
        import glob
        matches = glob.glob(args.exp_dir)
        if matches:
            exp_dir = Path(matches[0])
        else:
            raise FileNotFoundError(f"Experiment directory not found: {args.exp_dir}")

    print("=" * 80)
    print("🔬 NLRP3 MODEL ANALYSIS")
    print("=" * 80)
    print(f"Experiment Directory: {exp_dir}")
    print(f"Model: {args.model_name}")
    print(f"Fingerprint: {args.fingerprint}")

    # 创建分析输出目录
    analysis_dir = exp_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    # 1. 加载模型
    model_path = exp_dir / "models" / f"{args.model_name}_{args.fingerprint}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"\n📥 Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded: {type(model).__name__}")

    # 2. 加载测试数据
    test_data_path = exp_dir / "data" / "test.csv"
    print(f"\n📥 Loading test data from: {test_data_path}")
    test_df = pd.read_csv(test_data_path)

    # 智能识别列名
    smiles_col = None
    for col in ['smiles', 'SMILES', 'Smiles', 'smiles_standardized']:
        if col in test_df.columns:
            smiles_col = col
            break

    label_col = None
    for col in ['label', 'Label', 'activity', 'Activity']:
        if col in test_df.columns:
            label_col = col
            break

    if smiles_col is None or label_col is None:
        raise ValueError(f"Cannot find SMILES or label column in test data")

    smiles_test = test_df[smiles_col].tolist()
    y_test = test_df[label_col].values

    print(f"✅ Test data loaded: {len(test_df)} samples")

    # 3. 生成指纹
    print(f"\n🔬 Generating {args.fingerprint} fingerprints...")
    from rdkit.Chem import AllChem, MACCSkeys

    mols = [Chem.MolFromSmiles(smi) for smi in smiles_test]

    if args.fingerprint == 'ECFP4':
        fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
               if mol else None for mol in mols]
        X_test = np.array([[fp[i] for i in range(2048)] if fp else [0] * 2048
                           for fp in fps])
        feature_names = [f'ECFP4_Bit{i}' for i in range(2048)]
    elif args.fingerprint == 'MACCS':
        fps = [MACCSkeys.GenMACCSKeys(mol) if mol else None for mol in mols]
        X_test = np.array([[fp[i] for i in range(167)] if fp else [0] * 167
                           for fp in fps])
        feature_names = [f'MACCS_Bit{i}' for i in range(167)]
    elif args.fingerprint == 'RDKit':
        from rdkit.Chem import RDKFingerprint
        fps = [Chem.RDKFingerprint(mol, fpSize=2048) if mol else None for mol in mols]
        X_test = np.array([[fp[i] for i in range(2048)] if fp else [0] * 2048
                           for fp in fps])
        feature_names = [f'RDKit_Bit{i}' for i in range(2048)]
    else:
        raise ValueError(f"Unknown fingerprint: {args.fingerprint}")

    print(f"✅ Fingerprints generated: {X_test.shape}")

    # 4. 获取预测概率
    print(f"\n🔮 Making predictions...")
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"✅ Predictions complete")

    # 5. 阈值优化分析
    threshold_results = analyze_threshold_optimization(
        y_test, y_proba, analysis_dir
    )

    # 6. 特征重要性分析
    if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
        importance_results = analyze_feature_importance(
            model, feature_names, analysis_dir, top_n=args.top_n
        )
    else:
        print(f"\n⚠️ Model {type(model).__name__} does not support feature importance")

    # 7. SAR分析
    analyze_structure_activity_relationship(
        model, X_test, y_test, smiles_test, analysis_dir, top_n=10
    )

    # 8. 生成综合报告
    print("\n" + "=" * 80)
    print("📝 GENERATING SUMMARY REPORT")
    print("=" * 80)

    report = f"""
# NLRP3 Model Analysis Report

## Experiment Information
- **Experiment Directory:** {exp_dir}
- **Model:** {args.model_name}
- **Fingerprint:** {args.fingerprint}
- **Test Set Size:** {len(y_test)} samples

## 1. Threshold Optimization

### Default Threshold (0.5)
- Precision: {threshold_results['default_threshold']['precision']:.4f}
- Recall: {threshold_results['default_threshold']['recall']:.4f}
- F1-Score: {threshold_results['default_threshold']['f1']:.4f}

### Optimal Threshold ({threshold_results['optimal_threshold']['threshold']:.3f})
- Precision: {threshold_results['optimal_threshold']['precision']:.4f}
- Recall: {threshold_results['optimal_threshold']['recall']:.4f}
- F1-Score: {threshold_results['optimal_threshold']['f1']:.4f}
- **F1 Improvement:** +{(threshold_results['optimal_threshold']['f1'] - threshold_results['default_threshold']['f1']) * 100:.2f}%

### Recommendations

1. **High Precision Mode (Threshold = 0.7)**
   - Use when: Experimental validation cost is high
   - Minimize false positives

2. **Balanced Mode (Threshold = {threshold_results['optimal_threshold']['threshold']:.2f})**
   - Use when: Need optimal F1-score
   - Balanced precision and recall

3. **High Recall Mode (Threshold = 0.3)**
   - Use when: Early screening, ensure not missing actives
   - Cast a wider net

## 2. Feature Importance

Top 10 most important features:
"""

    if hasattr(model, 'feature_importances_'):
        for feat_info in importance_results['top_features'][:10]:
            report += f"\n{feat_info['rank']}. {feat_info['feature']}: {feat_info['importance']:.6f}"

        report += f"""

### Key Insights
- **Total Features:** {importance_results['statistics']['total_features']}
- **Features for 80% Importance:** {importance_results['statistics']['features_for_80_percent']}
- **Features for 90% Importance:** {importance_results['statistics']['features_for_90_percent']}
"""

    report += """

## 3. Generated Outputs

1. `threshold_optimization_analysis.png` - PR curve and metrics vs threshold
2. `confusion_matrix_comparison.png` - Default vs optimal threshold
3. `feature_importance_top{}.png` - Top N important features
4. `feature_importance_cumulative.png` - Cumulative importance curve
5. `feature_importance_distribution.png` - Importance distribution
6. `threshold_analysis_results.json` - Detailed threshold analysis
7. `feature_importance_analysis.json` - Detailed feature importance
8. `sar_analysis_results.json` - Structure-activity relationship analysis

## 4. Next Steps

1. **For Virtual Screening:**
   - Use threshold = 0.3-0.4 for initial screening (high recall)
   - Use threshold = 0.6-0.7 for hit validation (high precision)

2. **For Model Interpretation:**
   - Analyze top important fingerprint bits
   - Map to molecular substructures
   - Identify key pharmacophores

3. **For Model Improvement:**
   - Focus on feature engineering
   - Consider ensemble with other fingerprints
   - Incorporate 3D information (Stage 3)
""".format(args.top_n)

    report_path = analysis_dir / "ANALYSIS_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✅ Saved: {report_path}")

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"📁 All results saved to: {analysis_dir}")
    print("\n📊 Generated files:")
    for file in sorted(analysis_dir.glob("*")):
        print(f"   - {file.name}")
    print("=" * 80)


if __name__ == "__main__":
    main()