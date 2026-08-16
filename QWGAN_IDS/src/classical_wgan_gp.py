"""Simple Classical WGAN-GP interface and fake generator for smoke tests.

This module provides:
- class ClassicalWGAN(model_path: Optional[str] = None)
    - load(model_path)  # placeholder
    - generate(n_samples, seed=42, n_features=41, batch_size=256) -> np.ndarray
- FakeClassicalWGAN for deterministic smoke runs.

Real training code should be placed here by the repository owner; this file provides a minimal
stable interface so the FR-6 runner can execute Arm C in smoke tests.
"""
from typing import Optional
import numpy as np
import os
from cqai.lineage import verified_joblib_load


class ClassicalWGAN:
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_features: int = 41,
        *,
        expected_sha256: str | None = None,
        fitting_versions: dict[str, str] | None = None,
    ):
        self.model_path = model_path
        self.n_features = n_features
        self._generator = None
        if model_path is not None:
            if expected_sha256 is None:
                raise ValueError("expected_sha256 is required for a persisted WGAN")
            if fitting_versions is None:
                raise ValueError("fitting_versions are required for a persisted WGAN")
            self.load(
                model_path,
                expected_sha256=expected_sha256,
                fitting_versions=fitting_versions,
            )

    def load(
        self,
        model_path: str,
        *,
        expected_sha256: str,
        fitting_versions: dict[str, str],
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Classical WGAN model not found at {model_path}")
        obj = verified_joblib_load(
            model_path,
            expected_sha256=expected_sha256,
            expected_kind="classical_wgan",
            fitting_versions=fitting_versions,
        )
        # expected to find a dict with 'generator' key or a callable
        if isinstance(obj, dict) and 'generator' in obj:
            self._generator = obj['generator']
        elif hasattr(obj, 'generate'):
            self._generator = obj
        else:
            # store state; user must reconstruct generator logic
            self._generator = obj

    def generate(self, n_samples: int, seed: int = 42, batch_size: int = 256) -> np.ndarray:
        """Generate synthetic samples using the classical WGAN generator.

        If a real generator is loaded and exposes .generate(n, seed) it will be used.
        Otherwise this placeholder returns gaussian noise with mean 0 and small variance.
        """
        if hasattr(self._generator, 'generate') and callable(getattr(self._generator, 'generate')):
            samples = []
            generated = 0
            while generated < n_samples:
                take = min(batch_size, n_samples - generated)
                batch = self._generator.generate(take, seed + generated)
                samples.append(np.asarray(batch, dtype=np.float32))
                generated += take
            return np.vstack(samples)
        # fallback: deterministic gaussian noise (small variance)
        rng = np.random.default_rng(seed)
        return rng.normal(loc=0.0, scale=0.1, size=(n_samples, self.n_features)).astype('float32')


class FakeClassicalWGAN(ClassicalWGAN):
    """Deterministic fake generator for unit tests and smoke runs."""
    def __init__(self, n_features: int = 41):
        super().__init__(model_path=None, n_features=n_features)
        self.n_features = n_features

    def generate(self, n_samples: int, seed: int = 42, batch_size: int = 256) -> np.ndarray:
        rng = np.random.default_rng(seed)
        # return zeros with tiny noise for determinism
        return (rng.normal(scale=1e-6, size=(n_samples, self.n_features))).astype('float32')
