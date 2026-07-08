"""
实验管理器 - 兼容原NLRP3项目
==================================================
管理训练实验的编号、目录创建和元数据保存

Author: Research Team
Date: 2025-11-05
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class ExperimentManager:
    """实验管理器 - 兼容原项目"""
    
    def __init__(self, base_dir: str = "results/training_experiments"):
        """
        初始化实验管理器
        
        Args:
            base_dir: 实验结果基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 实验计数器文件（JSON格式，兼容原项目）
        self.counter_file = self.base_dir.parent / ".experiment_counter.json"
    
    def get_next_experiment_number(self) -> int:
        """获取下一个实验编号"""
        if self.counter_file.exists():
            with open(self.counter_file, 'r') as f:
                data = json.load(f)
                current = data.get('last_experiment_number', 0)
        else:
            current = 0
        
        next_num = current + 1
        
        # 保存新的编号
        with open(self.counter_file, 'w') as f:
            json.dump({
                'last_experiment_number': next_num,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
        
        return next_num
    
    def create_experiment_directory(
        self,
        experiment_type: str = "Training",
        description: str = ""
    ) -> Path:
        """
        创建实验目录
        
        Args:
            experiment_type: 实验类型
            description: 实验描述
        
        Returns:
            实验目录路径
        """
        exp_num = self.get_next_experiment_number()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 目录命名：Type_编号_时间戳
        dir_name = f"{experiment_type}_{exp_num:02d}_{timestamp}"
        exp_dir = self.base_dir / dir_name
        
        # 创建子目录
        subdirs = ["data", "models", "figures", "logs", "individual_model_results"]
        for subdir in subdirs:
            (exp_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # 保存实验信息
        info = {
            "experiment_number": exp_num,
            "experiment_type": experiment_type,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "directory": str(exp_dir)
        }
        
        with open(exp_dir / "experiment_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"✅ 创建实验目录: {dir_name}")
        return exp_dir
    
    def save_summary(self, exp_dir: Path, summary: Dict[str, Any]):
        """保存实验摘要"""
        summary_file = exp_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
