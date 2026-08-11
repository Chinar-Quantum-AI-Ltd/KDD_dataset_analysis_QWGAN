import numpy as np
from QWGAN_IDS.src.metrics import compute_classification_metrics


def test_compute_classification_metrics_basic():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.6, 0.8, 0.3, 0.2, 0.9])
    metrics = compute_classification_metrics(y_true, y_prob, y_pred, pos_label=1)
    expected_keys = {"precision", "recall", "f1", "roc_auc", "pr_auc", "benign_fpr", "confusion_matrix"}
    assert expected_keys.issubset(set(metrics.keys()))
    # values are floats or dict for confusion_matrix
    assert isinstance(metrics["precision"], float)
    assert isinstance(metrics["recall"], float)
    assert isinstance(metrics["f1"], float)
    assert isinstance(metrics["confusion_matrix"], dict)
