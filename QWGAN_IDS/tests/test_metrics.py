import unittest
import numpy as np
from src.metrics import compute_classification_metrics


class TestMetrics(unittest.TestCase):
    def test_compute_classification_metrics_basic(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.6, 0.8, 0.3, 0.2, 0.9])
        metrics = compute_classification_metrics(y_true, y_prob, y_pred, pos_label=1)
        expected_keys = {"precision", "recall", "f1", "roc_auc", "pr_auc", "benign_fpr", "confusion_matrix"}
        self.assertTrue(expected_keys.issubset(set(metrics.keys())))
        self.assertIsInstance(metrics["precision"], float)
        self.assertIsInstance(metrics["recall"], float)
        self.assertIsInstance(metrics["f1"], float)
        self.assertIsInstance(metrics["confusion_matrix"], dict)
