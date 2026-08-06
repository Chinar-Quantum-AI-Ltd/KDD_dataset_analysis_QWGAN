"""Public FR-3 hybrid QWGAN-GP interface."""

from .config import QWGANConfig
from .critic import Critic
from .data_contract import TrainingAngles
from .generator import DecodedQuantumGenerator, QuantumGenerator
from .losses import gradient_penalty
from .report import (
    StabilityThresholds,
    summarise_run,
    write_stability_report,
)
from .runner import (
    RunResult,
    TrainingPlan,
    detect_barren_plateau,
    diversity_ratio,
    run_training,
)
from .trainer import QWGANTrainer

__all__ = [
    "Critic",
    "DecodedQuantumGenerator",
    "QWGANConfig",
    "QuantumGenerator",
    "QWGANTrainer",
    "RunResult",
    "StabilityThresholds",
    "TrainingAngles",
    "TrainingPlan",
    "detect_barren_plateau",
    "diversity_ratio",
    "gradient_penalty",
    "run_training",
    "summarise_run",
    "write_stability_report",
]
