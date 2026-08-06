"""FR-4 fidelity metrics, domain constraints, and the class-conditional gate.

This package judges samples. It deliberately does not generate them -- that is
``cqai.synthesis`` -- so the gate can never be quietly tuned to whatever the
current generator happens to produce.
"""

from .domain import (
    EXCLUSIVE_GROUPS,
    INDICATOR_TOLERANCE,
    RULES,
    DomainReport,
    Rule,
    check_domain,
)
from .gate import GATE_VERSION, FidelityThresholds, evaluate_gate, released
from .metrics import CHANCE_AUC, c2st_auc, coverage, ks_statistic, wasserstein_1

__all__ = [
    "CHANCE_AUC",
    "EXCLUSIVE_GROUPS",
    "GATE_VERSION",
    "INDICATOR_TOLERANCE",
    "RULES",
    "DomainReport",
    "FidelityThresholds",
    "Rule",
    "c2st_auc",
    "check_domain",
    "coverage",
    "evaluate_gate",
    "ks_statistic",
    "released",
    "wasserstein_1",
]
