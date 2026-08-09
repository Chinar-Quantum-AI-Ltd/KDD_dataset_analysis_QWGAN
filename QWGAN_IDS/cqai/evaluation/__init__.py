"""Evaluation package for FR-5 / FR-8.

Exports metrics calculation and EvaluationHarness manifest generator.
"""
from __future__ import annotations

from .evaluator import EvaluationHarness
from .metrics import evaluate_classifier_metrics

__all__ = [
    "evaluate_classifier_metrics",
    "EvaluationHarness",
]
