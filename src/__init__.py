"""
NLRP3 Screening Project - Source Code Package
==================================================

Modules:
- data: Data loading and preprocessing
- features: Feature generation (fingerprints, graphs, 3D)
- models: Machine learning and deep learning models
- training: Training utilities
- evaluation: Model evaluation
- utils: Utility functions

Author: Research Team
"""

__version__ = "2.0.0"
__author__ = "Research Team"

from .experiment_manager import ExperimentManager

__all__ = [
    "ExperimentManager",
]
