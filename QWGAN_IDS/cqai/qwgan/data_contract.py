from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import QWGANConfig


@dataclass(frozen=True, slots=True)
class TrainingAngles:
    """Validated train-only FR-2 handoff consumed by FR-3."""

    values: torch.Tensor
    attack_class: str
    latent_columns: tuple[str, ...]

    @classmethod
    def from_array(
        cls,
        values: np.ndarray | torch.Tensor,
        *,
        config: QWGANConfig,
        partition: str,
        attack_class: str,
        latent_columns: tuple[str, ...],
        tolerance: float = 1e-12,
    ) -> "TrainingAngles":
        if partition != "train":
            raise ValueError("FR-3 training accepts only the train partition")
        if not attack_class or "�" in attack_class:
            raise ValueError("attack_class must be a canonical UTF-8 label")
        expected_columns = tuple(f"z{i}" for i in range(config.n_qubits))
        if latent_columns != expected_columns:
            raise ValueError(f"latent_columns must be ordered as {expected_columns}")

        tensor = torch.as_tensor(values, dtype=torch.float64).detach().clone()
        if tensor.ndim != 2 or tensor.shape[1] != config.n_qubits:
            raise ValueError(
                f"angles must have shape (samples, {config.n_qubits})"
            )
        if tensor.shape[0] < 1:
            raise ValueError("angles must contain at least one sample")
        if not torch.isfinite(tensor).all():
            raise ValueError("angles must contain only finite values")
        if (tensor < -tolerance).any() or (tensor > np.pi + tolerance).any():
            raise ValueError("angles must be within [0, pi]")

        return cls(
            values=tensor.clamp(0.0, float(np.pi)),
            attack_class=attack_class,
            latent_columns=latent_columns,
        )
