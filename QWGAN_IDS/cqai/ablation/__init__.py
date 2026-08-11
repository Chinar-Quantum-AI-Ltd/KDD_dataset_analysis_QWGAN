"""FR-6 Four-Arm Three-Seed Ablation Package for CQAI QWGAN-IDS.

Arms:
- Arm A: Real data only
- Arm B: Real + SMOTE / ADASYN
- Arm C: Real + Classical WGAN-GP
- Arm D: Real + QWGAN-GP
"""
from .arms import build_ablation_arm
from .runner import AblationRunner, compute_paired_ttest

__all__ = [
    "AblationRunner",
    "build_ablation_arm",
    "compute_paired_ttest",
]
