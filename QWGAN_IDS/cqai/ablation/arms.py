"""Ablation Arm Dataset Builder for FR-6.

Provides dataset construction for Arms A, B, C, and D:
- Arm A: Real data only
- Arm B: Real + SMOTE synthetic data
- Arm C: Real + Classical WGAN-GP synthetic data
- Arm D: Real + QWGAN-GP synthetic data
"""
from __future__ import annotations

from typing import Any, Callable
import numpy as np

from src.augmentation.dataset_builder import build_augmented_dataset, _compute_target_minority_count


def build_ablation_arm(
    arm: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_class: str | int,
    *,
    target_ratio: float = 0.30,
    seed: int = 42,
    qwgan_generator_func: Callable[[int, int], np.ndarray] | None = None,
    wgan_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct augmented training dataset (X_aug, y_aug) for a specific ablation arm.

    Parameters
    ----------
    arm : str
        One of 'A', 'B', 'C', 'D' (case-insensitive).
    X_train : np.ndarray
        Real training feature matrix.
    y_train : np.ndarray
        Real training label array.
    target_class : str | int
        Target minority attack class to augment.
    target_ratio : float, default=0.30
        Target ratio of minority count relative to majority class count.
    seed : int, default=42
        Random seed for sampling reproducibility.
    qwgan_generator_func : Callable, optional
        Function `func(n_samples, seed) -> np.ndarray` producing QWGAN samples for Arm D.
    wgan_config : dict, optional
        Hyperparameters for classical WGAN training in Arm C.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (X_augmented, y_augmented)
    """
    arm_upper = arm.upper().strip()
    X = np.asarray(X_train)
    y = np.asarray(y_train)

    if arm_upper == "A":
        # Arm A: Real data only
        return X.copy(), y.copy()

    elif arm_upper == "B":
        # Arm B: Real + SMOTE
        return build_augmented_dataset(
            X,
            y,
            method="smote",
            target_class=target_class,
            target_ratio=target_ratio,
            random_state=seed,
        )

    elif arm_upper == "C":
        # Arm C: Real + Classical WGAN-GP
        return build_augmented_dataset(
            X,
            y,
            method="classical_wgan",
            target_class=target_class,
            target_ratio=target_ratio,
            random_state=seed,
            wgan_config=wgan_config or {"epochs": 30, "batch_size": 32},
        )

    elif arm_upper == "D":
        # Arm D: Real + QWGAN-GP
        existing_minority = int((y == target_class).sum())
        target_minority_count = _compute_target_minority_count(y, target_class, target_ratio)
        n_to_generate = target_minority_count - existing_minority

        if n_to_generate <= 0 or qwgan_generator_func is None:
            return X.copy(), y.copy()

        X_synth = qwgan_generator_func(n_to_generate, seed)
        y_synth = np.array([target_class] * len(X_synth))

        X_aug = np.vstack([X, np.asarray(X_synth)])
        y_aug = np.hstack([y, np.asarray(y_synth)])
        return X_aug, y_aug

    else:
        raise ValueError(f"Unknown ablation arm '{arm}'. Must be one of 'A', 'B', 'C', or 'D'.")
