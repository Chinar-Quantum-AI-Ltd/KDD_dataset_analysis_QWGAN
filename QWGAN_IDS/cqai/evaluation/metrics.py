"""Unified Evaluation Metrics Suite for FR-5.

Computes per-class and macro-averaged metrics (Precision, Recall, F1, ROC-AUC, PR-AUC),
benign false-positive rate (FPR), and confusion matrices.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prob: np.ndarray | None = None,
    classes: list[str] | np.ndarray | None = None,
    benign_class: str = "normal",
) -> dict[str, Any]:
    """Compute the full FR-5 metric suite on held-out test predictions.

    Parameters
    ----------
    y_true :
        Ground truth labels (N,).
    y_pred :
        Predicted labels (N,).
    y_prob :
        Predicted class probabilities (N, C), optional.
    classes :
        Ordered class labels.
    benign_class :
        Name of the benign class (default "normal") for benign FPR.

    Returns
    -------
    dict with per_class, macro, benign_fpr, and confusion_matrix records.
    """
    y_true_arr = y_true.to_numpy() if isinstance(y_true, pd.Series) else np.asarray(y_true)
    y_pred_arr = y_pred.to_numpy() if isinstance(y_pred, pd.Series) else np.asarray(y_pred)

    if classes is None:
        classes_arr = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
    else:
        classes_arr = np.asarray(classes)

    class_list = classes_arr.tolist()
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=classes_arr)

    # Per-class & macro metrics
    prec_per = precision_score(y_true_arr, y_pred_arr, labels=classes_arr, average=None, zero_division=0)
    rec_per = recall_score(y_true_arr, y_pred_arr, labels=classes_arr, average=None, zero_division=0)
    f1_per = f1_score(y_true_arr, y_pred_arr, labels=classes_arr, average=None, zero_division=0)

    per_class: dict[str, dict[str, float]] = {}
    for idx, cls_name in enumerate(class_list):
        per_class[str(cls_name)] = {
            "precision": float(prec_per[idx]),
            "recall": float(rec_per[idx]),
            "f1": float(f1_per[idx]),
        }

    # Macro averages
    macro_prec = float(np.mean(prec_per))
    macro_rec = float(np.mean(rec_per))
    macro_f1 = float(np.mean(f1_per))

    # Benign FPR (for benign class, FP / (FP + TN))
    benign_fpr: float | None = None
    if benign_class in class_list:
        b_idx = class_list.index(benign_class)
        # FP is column b_idx sum minus diagonal element (cm[b_idx, b_idx])
        fp = float(np.sum(cm[:, b_idx]) - cm[b_idx, b_idx])
        # TN is sum of all non-benign rows excluding non-benign predictions for benign
        tn = float(np.sum(cm) - np.sum(cm[b_idx, :]) - np.sum(cm[:, b_idx]) + cm[b_idx, b_idx])
        benign_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # ROC-AUC & PR-AUC if probabilities are supplied
    roc_auc_macro: float | None = None
    pr_auc_macro: float | None = None

    if y_prob is not None and len(class_list) >= 2:
        try:
            # One-hot encode y_true for multi-class ROC-AUC
            y_true_onehot = pd.get_dummies(y_true_arr).reindex(columns=class_list, fill_value=0).values
            roc_auc_macro = float(roc_auc_score(y_true_onehot, y_prob, multi_class="ovr", average="macro"))

            # Calculate PR-AUC macro
            pr_aucs = []
            for c_idx in range(len(class_list)):
                prec_c, rec_c, _ = precision_recall_curve(y_true_onehot[:, c_idx], y_prob[:, c_idx])
                # Trapezoidal area under Precision-Recall curve
                pr_auc_c = float(np.trapz(prec_c, rec_c))
                pr_aucs.append(pr_auc_c)
            pr_auc_macro = float(np.mean(pr_aucs))
        except Exception:
            pass

    return {
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "per_class": per_class,
        "macro": {
            "precision": macro_prec,
            "recall": macro_rec,
            "f1": macro_f1,
            "roc_auc": roc_auc_macro,
            "pr_auc": pr_auc_macro,
        },
        "benign_fpr": benign_fpr,
        "confusion_matrix": {
            "labels": class_list,
            "matrix": cm.tolist(),
        },
    }
