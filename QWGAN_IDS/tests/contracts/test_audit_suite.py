"""Comprehensive Audit Test Suite covering all 6 system dimensions.

Dimensions Audited:
1. Data Leakage & Split Isolation
2. Contract Invariants & Domain Range
3. Model Architecture & Gradient Finiteness
4. Fidelity Gate & Quarantine Security
5. Classifier Panel & Metric Sanity
6. Seed Replay & FR-8 Lineage Hashes
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cqai.classifiers import (
    PyTorchDNNClassifier,
    QuantumKernelSVM,
    RFClassifier,
    XGBClassifierWrapper,
)
from cqai.data.nslkdd import load_train_contract
from cqai.evaluation import evaluate_classifier_metrics
from cqai.fidelity import evaluate_gate, released
from cqai.qwgan import Critic, QWGANConfig, QuantumGenerator
from cqai.qwgan.losses import gradient_penalty, gradient_penalty_and_norm
from cqai.qwgan.data_contract import TrainingAngles


class ExhaustiveAuditSuite(unittest.TestCase):
    """Exhaustive Audit Suite for CQAI QWGAN-IDS."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.contract_path = Path("/home/tarik/Projects/chinarqai/KDD_dataset_analysis_QWGAN/QWGAN_IDS/artifacts/contracts/nslkdd-train-only-v1")
        if cls.contract_path.exists():
            cls.contract = load_train_contract(cls.contract_path)
        else:
            cls.contract = None

    # -- 1. Data Leakage & Split Isolation Audit ---------------------------- #
    def test_audit_1_1_split_row_indices_are_strictly_disjoint(self) -> None:
        """Train and Val partitions must have 0 overlapping rows."""
        if self.contract is None:
            self.skipTest("Contract not found on disk.")
        train_rows = set(self.contract.row_index("train"))
        val_rows = set(self.contract.row_index("val"))
        overlap = train_rows.intersection(val_rows)
        self.assertEqual(len(overlap), 0, f"Data Leakage Detected! {len(overlap)} rows overlap between train and val.")

    def test_audit_1_2_training_angles_refuses_non_train_partition(self) -> None:
        """TrainingAngles wrapper must reject validation/test data."""
        if self.contract is None:
            self.skipTest("Contract not found on disk.")
        config = QWGANConfig()
        val_data = self.contract.validation_angles("r2l")
        latent_cols = tuple(f"z{i}" for i in range(10))
        with self.assertRaises(ValueError):
            # TrainingAngles.from_array directly with val partition must fail
            TrainingAngles.from_array(val_data, config=config, partition="val", attack_class="r2l", latent_columns=latent_cols)

    # -- 2. Contract Invariants & Domain Range Audit ------------------------ #
    def test_audit_2_1_all_contract_angles_in_declared_range(self) -> None:
        """All angle features must lie strictly in [0, pi]."""
        if self.contract is None:
            self.skipTest("Contract not found on disk.")
        for part in ("train", "val"):
            angles = self.contract.angles(part)
            self.assertTrue(np.isfinite(angles).all(), f"Non-finite values found in {part} angles.")
            self.assertTrue((angles >= 0.0).all(), f"Negative angles found in {part}.")
            self.assertTrue((angles <= np.pi + 1e-7).all(), f"Angles exceeding pi found in {part}.")

    def test_audit_2_2_attack_families_canonical_utf8(self) -> None:
        """Labels must be clean canonical strings without replacement characters."""
        if self.contract is None:
            self.skipTest("Contract not found on disk.")
        for part in ("train", "val"):
            families = self.contract.families(part)
            for fam in set(families):
                self.assertNotIn("\ufffd", str(fam), "Mojibake/Encoding corruption detected in attack family string.")

    # -- 3. Model Architecture & Gradient Finiteness Audit ------------------- #
    def test_audit_3_1_critic_has_no_batchnorm_or_sigmoid(self) -> None:
        """Critic must use LayerNorm/LeakyReLU without BatchNorm or Sigmoid."""
        critic = Critic(input_dim=10)
        for module in critic.modules():
            self.assertNotIsInstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Sigmoid))

    def test_audit_3_2_gradient_penalty_is_finite_and_non_negative(self) -> None:
        """Gradient Penalty must be finite and >= 0."""
        critic = Critic(input_dim=10)
        real = torch.randn(16, 10)
        fake = torch.randn(16, 10)
        gp, norm = gradient_penalty_and_norm(critic, real, fake)
        self.assertTrue(torch.isfinite(gp).item())
        self.assertGreaterEqual(gp.item(), 0.0)
        self.assertTrue(torch.isfinite(norm).item())

    # -- 4. Fidelity Gate & Quarantine Security Audit ----------------------- #
    def test_audit_4_1_gate_quarantines_failing_distribution(self) -> None:
        """Gate must reject/quarantine shifted distributions failing C2ST."""
        rng = np.random.default_rng(42)
        real = pd.DataFrame(rng.normal(loc=0.0, scale=1.0, size=(100, 10)), columns=[f"z{i}" for i in range(10)])
        shifted_synth = pd.DataFrame(rng.normal(loc=3.0, scale=1.0, size=(100, 10)), columns=[f"z{i}" for i in range(10)])

        res = evaluate_gate(real, shifted_synth, seed=42)
        self.assertFalse(released(res), "Gate failed to quarantine a clearly distinguishable distribution.")
        self.assertIn(res["verdict"], ("fail", "quarantine"))

    # -- 5. Classifier Panel & Metric Sanity Audit -------------------------- #
    def test_audit_5_1_classifier_probability_distribution_sums_to_one(self) -> None:
        """All 4 classifiers must output valid probability distributions summing to 1.0."""
        rng = np.random.default_rng(42)
        X_tr = rng.normal(size=(40, 10))
        y_tr = np.array(["normal"] * 25 + ["dos"] * 15)
        X_te = rng.normal(size=(10, 10))

        classifiers = [
            RFClassifier(n_estimators=5, random_state=42),
            XGBClassifierWrapper(n_estimators=5, random_state=42),
            PyTorchDNNClassifier(hidden_dim=16, epochs=2, random_state=42),
            QuantumKernelSVM(n_qubits=10, max_samples=20, random_state=42),
        ]

        for clf in classifiers:
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_te)
            self.assertEqual(probs.shape, (10, 2))
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-4, err_msg=f"{clf.name} probabilities do not sum to 1.0")

    def test_audit_5_2_benign_fpr_calculation_correctness(self) -> None:
        """Benign FPR must strictly evaluate FP / (FP + TN) for benign class."""
        y_true = np.array(["normal", "normal", "dos", "dos"])
        y_pred = np.array(["normal", "dos", "normal", "dos"])
        res = evaluate_classifier_metrics(y_true, y_pred, classes=["dos", "normal"], benign_class="normal")
        # 2 actual normal: 1 correctly normal (TN), 1 predicted as dos (FP) -> FPR = 1/(1+1) = 0.5
        self.assertEqual(res["benign_fpr"], 0.5)

    # -- 6. Seed Replay & Lineage Audit ------------------------------------- #
    def test_audit_6_1_seed_replay_determinism(self) -> None:
        """Identical seeds must yield identical classifier predictions."""
        rng1 = np.random.default_rng(42)
        X1 = rng1.normal(size=(30, 8))
        y1 = np.array(["normal"] * 20 + ["r2l"] * 10)

        clf1 = RFClassifier(n_estimators=10, random_state=123)
        clf1.fit(X1, y1)
        p1 = clf1.predict(X1)

        clf2 = RFClassifier(n_estimators=10, random_state=123)
        clf2.fit(X1, y1)
        p2 = clf2.predict(X1)

        np.testing.assert_array_equal(p1, p2)
