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
    #: Latent noise is drawn from ``[0, latent_scale * pi]``.
    #:
    #: Data re-uploading applies ``RY(noise)`` at every layer, so at the full
    #: ``[0, pi]`` range a four-layer circuit accumulates up to ``4*pi`` of
    #: rotation and the Pauli-Z expectation averages to zero over the latent.
    #: Every qubit's output mean is then pinned near ``pi/2`` regardless of the
    #: weights -- measured, and the reason the first FR-3 campaign's generators
    #: could not match the per-feature means. See
    #: ``docs/fr3-generator-diagnosis.md``.
    #:
    #: The default stays 1.0 so no previously reported run changes meaning; a
    #: config must opt in, and the manifest records which value was used.
    latent_scale: float = 1.0
    #: Which layer indices get a CNOT ring. ``None`` means every layer, which
    #: is the TDD's hardware-efficient structure and stays the default.
    #:
    #: A ring after every layer drives the single-qubit reduced states toward
    #: maximally mixed, so the local Pauli-Z expectations concentrate at zero:
    #: measured on a 10-qubit, 4-layer circuit, the per-qubit output spread is
    #: 0.028 with the full ring against 0.441 with none, while the real ``r2l``
    #: data needs 0.181. Entangling only the last layer gives 0.111.
    #:
    #: Reducing it is a real trade: the correlations a ring buys are worth
    #: little after PCA has already decorrelated the components, but an empty
    #: tuple leaves a separable circuit that is classically trivial to simulate
    #: and forfeits the quantum claim. See ``docs/fr3-generator-diagnosis.md``.
    entangling_layers: tuple[int, ...] | None = None

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
        # Above 1.0 the latent would exceed the angle domain the decoder was
        # fitted on and wrap the Bloch sphere even harder.
        if not 0 < self.latent_scale <= 1:
            raise ValueError("latent_scale must be in (0, 1]")
        if self.entangling_layers is not None:
            layers = tuple(self.entangling_layers)
            if len(set(layers)) != len(layers):
                raise ValueError("entangling_layers must not repeat a layer")
            if any(not 0 <= layer < self.n_layers for layer in layers):
                raise ValueError(
                    f"entangling_layers must index layers 0..{self.n_layers - 1}"
                )
