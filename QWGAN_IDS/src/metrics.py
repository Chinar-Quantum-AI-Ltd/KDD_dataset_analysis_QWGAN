"""Evaluation metrics utilities for FR-6 experiments.

Provides compute_classification_metrics which returns a dictionary with:
- precision, recall, f1
- roc_auc, pr_auc
- benign_fpr (false positive rate for benign class 0)
- confusion_matrix as dict {tn, fp, fn, tp}

Handles edge cases where only one class present.
"""
from typing import Dict, Optional
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def compute_classification_metrics(y_true, y_pred_prob: Optional[np.ndarray], y_pred, pos_label=1) -> Dict[str, float]:
    """Compute standard classification metrics used in FR-6.

    Args:
        y_true: array-like of true labels (binary expected: 0=benign, 1=malicious)
        y_pred_prob: array-like of predicted probabilities for the positive class or None
        y_pred: array-like of predicted labels
        pos_label: value representing the positive (malicious) class. Default 1.

    Returns:
        dict with keys: precision, recall, f1, roc_auc, pr_auc, benign_fpr, confusion_matrix (nested dict)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {}

    # Precision / Recall / F1 (use pos_label)
    try:
        metrics["precision"] = float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    except Exception:
        metrics["precision"] = float("nan")

    try:
        metrics["recall"] = float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    except Exception:
        metrics["recall"] = float("nan")

    try:
        metrics["f1"] = float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    except Exception:
        metrics["f1"] = float("nan")

    # ROC AUC and PR AUC require probability scores for the positive class
    if y_pred_prob is None:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    else:
        y_pred_prob = np.array(y_pred_prob)
        try:
            # roc_auc_score needs at least two classes in y_true
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_prob))
        except Exception:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_pred_prob))
        except Exception:
            metrics["pr_auc"] = float("nan")

    # Confusion matrix and benign FPR
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, pos_label]).ravel()
        metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        # benign FPR = fp / (fp + tn)
        denom = float(fp + tn)
        if denom == 0:
            metrics["benign_fpr"] = float("nan")
        else:
            metrics["benign_fpr"] = float(fp / denom)
    except Exception:
        metrics["confusion_matrix"] = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
        metrics["benign_fpr"] = float("nan")

    return metrics


if __name__ == "__main__":
    # quick smoke test
    y_true = [0, 0, 1, 1, 0, 1]
    y_pred = [0, 1, 1, 0, 0, 1]
    y_prob = [0.1, 0.6, 0.8, 0.3, 0.2, 0.9]
    print(compute_classification_metrics(y_true, y_prob, y_pred))
