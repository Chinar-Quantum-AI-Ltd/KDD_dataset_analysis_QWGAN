"""FR-6 Four-Arm Three-Seed Ablation Protocol Runner and Statistical Engine.

Calculates mean +- std metrics, paired t-tests, success criteria checks,
and outputs FR-8 machine-readable manifests.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from typing import Any, Callable, Sequence
import numpy as np
from scipy import stats

from cqai.ablation.arms import build_ablation_arm
from cqai.classifiers import (
    BaseClassifier,
    PyTorchDNNClassifier,
    QuantumKernelSVM,
    RFClassifier,
    XGBClassifierWrapper,
)
from cqai.evaluation import evaluate_classifier_metrics


def compute_paired_ttest(
    scores_d: Sequence[float],
    scores_other: Sequence[float],
) -> tuple[float, float]:
    """Compute paired Student's t-test statistic and two-tailed p-value.

    Parameters
    ----------
    scores_d : Sequence[float]
        Metric scores across seeds for Arm D.
    scores_other : Sequence[float]
        Metric scores across seeds for Arm A, B, or C.

    Returns
    -------
    tuple[float, float]
        (t_statistic, p_value)
    """
    a = np.asarray(scores_d, dtype=float)
    b = np.asarray(scores_other, dtype=float)

    if len(a) != len(b):
        raise ValueError("Sample sizes across seeds must match for paired t-test.")
    if len(a) < 2:
        return 0.0, 1.0

    diff = a - b
    if np.allclose(diff, 0.0):
        return 0.0, 1.0

    res = stats.ttest_rel(a, b)
    t_stat = float(res.statistic) if np.isfinite(res.statistic) else 0.0
    p_val = float(res.pvalue) if np.isfinite(res.pvalue) else 1.0
    return t_stat, p_val


class AblationRunner:
    """FR-6 Four-Arm Three-Seed Ablation Campaign Runner."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        target_class: str | int = "r2l",
        benign_class: str | int = "normal",
        *,
        seeds: Sequence[int] = (42, 123, 1337),
        target_ratio: float = 0.30,
        qwgan_generator_func: Callable[[int, int], np.ndarray] | None = None,
        classifiers: list[BaseClassifier] | None = None,
    ) -> None:
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.X_test = np.asarray(X_test)
        self.y_test = np.asarray(y_test)

        self.target_class = target_class
        self.benign_class = benign_class
        self.seeds = list(seeds)
        self.target_ratio = target_ratio
        self.qwgan_generator_func = qwgan_generator_func

        self.classes = sorted(list(set(y_train).union(set(y_test))))
        self.classifiers = classifiers or [
            RFClassifier(n_estimators=10),
            XGBClassifierWrapper(n_estimators=10),
            PyTorchDNNClassifier(hidden_dim=16, epochs=3),
        ]

    def run_ablation(self) -> dict[str, Any]:
        """Execute the 4-arm 3-seed ablation campaign across all classifiers.

        Returns
        -------
        dict[str, Any]
            Complete structured ablation report with per-arm per-seed results,
            mean +- std summaries, paired t-tests, and primary success verdict.
        """
        run_id = f"ablation_{uuid.uuid4().hex[:12]}"
        timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        arms = ["A", "B", "C", "D"]

        raw_results: dict[str, dict[str, list[dict[str, Any]]]] = {
            clf.name: {arm: [] for arm in arms} for clf in self.classifiers
        }

        # 1. Run all Arms x Seeds x Classifiers
        for arm in arms:
            for seed in self.seeds:
                # Build augmented dataset for this arm & seed
                X_aug, y_aug = build_ablation_arm(
                    arm=arm,
                    X_train=self.X_train,
                    y_train=self.y_train,
                    target_class=self.target_class,
                    target_ratio=self.target_ratio,
                    seed=seed,
                    qwgan_generator_func=self.qwgan_generator_func,
                )

                for clf_base in self.classifiers:
                    # Instantiate fresh classifier per seed
                    clf = type(clf_base)()
                    if hasattr(clf, "random_state"):
                        setattr(clf, "random_state", seed)

                    clf.fit(X_aug, y_aug)
                    y_pred = clf.predict(self.X_test)
                    y_prob = clf.predict_proba(self.X_test)

                    metrics = evaluate_classifier_metrics(
                        self.y_test,
                        y_pred,
                        y_prob=y_prob,
                        classes=self.classes,
                        benign_class=self.benign_class,
                    )
                    metrics["seed"] = seed
                    raw_results[clf.name][arm].append(metrics)

        # 2. Compute Statistics, Means, Stds, and Paired t-tests
        summary_by_clf: dict[str, Any] = {}

        for clf_name, arm_results in raw_results.items():
            clf_summary: dict[str, Any] = {"arms": {}}

            f1_by_arm: dict[str, list[float]] = {}
            fpr_by_arm: dict[str, list[float]] = {}

            for arm in arms:
                seed_metrics = arm_results[arm]
                f1_scores = [m["macro_f1"] for m in seed_metrics]
                fpr_scores = [m["benign_fpr"] for m in seed_metrics]

                f1_by_arm[arm] = f1_scores
                fpr_by_arm[arm] = fpr_scores

                clf_summary["arms"][arm] = {
                    "macro_f1_mean": float(np.mean(f1_scores)),
                    "macro_f1_std": float(np.std(f1_scores)),
                    "benign_fpr_mean": float(np.mean(fpr_scores)),
                    "benign_fpr_std": float(np.std(fpr_scores)),
                    "per_seed": seed_metrics,
                }

            # Paired t-tests comparing Arm D vs A, B, C
            f1_d = f1_by_arm["D"]
            fpr_d = fpr_by_arm["D"]

            ttests: dict[str, Any] = {}
            for other_arm in ["A", "B", "C"]:
                f1_stat, f1_p = compute_paired_ttest(f1_d, f1_by_arm[other_arm])
                fpr_stat, fpr_p = compute_paired_ttest(fpr_d, fpr_by_arm[other_arm])
                ttests[f"D_vs_{other_arm}"] = {
                    "macro_f1_t_stat": f1_stat,
                    "macro_f1_p_val": f1_p,
                    "benign_fpr_t_stat": fpr_stat,
                    "benign_fpr_p_val": fpr_p,
                }

            clf_summary["paired_ttests"] = ttests

            # Primary Success Criteria Check (FR-6):
            # Arm D macro-F1 >= Arm A macro-F1 + 0.05 AND benign_fpr increase <= 0.005
            f1_lift = clf_summary["arms"]["D"]["macro_f1_mean"] - clf_summary["arms"]["A"]["macro_f1_mean"]
            fpr_increase = clf_summary["arms"]["D"]["benign_fpr_mean"] - clf_summary["arms"]["A"]["benign_fpr_mean"]

            f1_pass = f1_lift >= 0.05
            fpr_pass = fpr_increase <= 0.005
            beats_b_c = (
                clf_summary["arms"]["D"]["macro_f1_mean"] >= clf_summary["arms"]["B"]["macro_f1_mean"]
                and clf_summary["arms"]["D"]["macro_f1_mean"] >= clf_summary["arms"]["C"]["macro_f1_mean"]
            )

            quantum_benefit_claimed = bool(f1_pass and fpr_pass and beats_b_c)

            clf_summary["success_criteria"] = {
                "quantum_benefit_claimed": quantum_benefit_claimed,
                "macro_f1_lift_vs_real": float(f1_lift),
                "benign_fpr_increase": float(fpr_increase),
                "f1_lift_passed": bool(f1_pass),
                "fpr_threshold_passed": bool(fpr_pass),
                "beats_smote_and_wgan": bool(beats_b_c),
                "retains_classical_fallback": bool(not quantum_benefit_claimed),
            }

            summary_by_clf[clf_name] = clf_summary

        # Hash dataset input
        data_hash = hashlib.sha256(self.X_train.tobytes() + self.y_train.tobytes()).hexdigest()

        ablation_report = {
            "run_id": run_id,
            "timestamp_utc": timestamp_utc,
            "target_class": str(self.target_class),
            "benign_class": str(self.benign_class),
            "target_ratio": float(self.target_ratio),
            "seeds": self.seeds,
            "dataset_sha256": data_hash,
            "summary_by_classifier": summary_by_clf,
        }

        return ablation_report
