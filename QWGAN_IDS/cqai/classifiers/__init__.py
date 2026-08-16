"""FR-5 classifier panel with lazy optional-dependency imports.

Importing a classical serving component must not import Torch, PennyLane, or
XGBoost.  Individual implementations load only when explicitly requested.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BaseClassifier": ("cqai.classifiers.base", "BaseClassifier"),
    "RFClassifier": ("cqai.classifiers.random_forest", "RFClassifier"),
    "XGBClassifierWrapper": ("cqai.classifiers.xgboost_model", "XGBClassifierWrapper"),
    "PyTorchDNNClassifier": ("cqai.classifiers.dnn", "PyTorchDNNClassifier"),
    "QuantumKernelSVM": ("cqai.classifiers.quantum_svm", "QuantumKernelSVM"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
