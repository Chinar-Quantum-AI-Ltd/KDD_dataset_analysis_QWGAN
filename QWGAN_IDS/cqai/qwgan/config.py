from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QWGANConfig:
    """Validated FR-3 model and optimization defaults."""

    n_qubits: int = 10
    n_layers: int = 4
    backend: str = "default.qubit"
    diff_method: str = "backprop"
    seed: int = 42
    lambda_gp: float = 10.0
    n_critic: int = 5
    learning_rate: float = 1e-4
    beta1: float = 0.0
    beta2: float = 0.9

    def __post_init__(self) -> None:
        if not 8 <= self.n_qubits <= 12:
            raise ValueError("n_qubits must be between 8 and 12")
        if not 3 <= self.n_layers <= 6:
            raise ValueError("n_layers must be between 3 and 6")
        if self.lambda_gp < 0:
            raise ValueError("lambda_gp must be non-negative")
        if self.n_critic < 1:
            raise ValueError("n_critic must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
