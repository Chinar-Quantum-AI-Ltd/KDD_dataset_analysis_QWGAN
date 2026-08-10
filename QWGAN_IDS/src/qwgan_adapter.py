"""QWGAN Adapter module for FR-6 generation and smoke testing."""
from __future__ import annotations

import numpy as np


class QWGANAdapter:
    """Adapter for loading trained PennyLane QWGAN checkpoints and generating synthetic samples."""

    def __init__(self, checkpoint_path: str | None = None) -> None:
        if checkpoint_path is None:
            raise RuntimeError("PennyLane checkpoint path missing or unreadable")
        self.checkpoint_path = checkpoint_path


class FakeQWGANAdapter:
    """Mock QWGAN Adapter for smoke tests and fast CPU unit testing."""

    def __init__(self, n_features: int = 8) -> None:
        self.n_features = n_features

    def generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(low=0.0, high=np.pi, size=(n_samples, self.n_features))
