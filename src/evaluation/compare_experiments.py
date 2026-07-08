import json
from pathlib import Path
import pandas as pd


def compare_experiments(exp_dirs):
    """对比多个实验的结果"""
    results = []

    for exp_dir in exp_dirs:
        summary_path = Path(exp_dir) / "experiment_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
                results.append({
                    'Experiment': Path(exp_dir).name,
                    'Config': data.get('config_path', 'Unknown'),
                    'Best Model': data['best_model']['model_name'],
                    'Test AUC': data['best_model']['test_metrics']['roc_auc'],
                    'Test Acc': data['best_model']['test_metrics']['accuracy'],
                    'Test F1': data['best_model']['test_metrics']['f1'],
                    'Training Time (min)': data['best_model']['training_time'] / 60
                })

    df = pd.DataFrame(results)
    df = df.sort_values('Test AUC', ascending=False)

    print("\n" + "=" * 100)
    print("📊 EXPERIMENT COMPARISON")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100 + "\n")

    return df


# 使用
exp_dirs = [
    "results/training_experiments/Exp009_Stage2_2D_Enhanced_Stage2_Quick_Test_20251108_230103",
    "results/training_experiments/Exp010_...",  # 添加其他实验
]

compare_experiments(exp_dirs)