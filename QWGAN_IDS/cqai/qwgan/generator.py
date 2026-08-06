from __future__ import annotations

import pennylane as qml
import torch
from torch import nn

from .config import QWGANConfig


class QuantumGenerator(nn.Module):
    """PennyLane data-reuploading generator with local Pauli-Z outputs."""

    def __init__(self, config: QWGANConfig) -> None:
        super().__init__()
        self.config = config
        self.device = qml.device(config.backend, wires=config.n_qubits)

        random = torch.Generator().manual_seed(config.seed)
        initial_weights = 0.01 * torch.randn(
            config.n_layers,
            config.n_qubits,
            3,
            dtype=torch.float64,
            generator=random,
        )
        self.weights = nn.Parameter(initial_weights)
        self._circuit = self._build_circuit()

    def _build_circuit(self):
        n_qubits = self.config.n_qubits

        @qml.qnode(
            self.device,
            interface="torch",
            diff_method=self.config.diff_method,
        )
        def circuit(noise: torch.Tensor, weights: torch.Tensor):
            for layer in range(self.config.n_layers):
                for wire in range(n_qubits):
                    # Data re-uploading: the same noise angle enters every layer.
                    qml.RY(noise[..., wire], wires=wire)
                    qml.Rot(*weights[layer, wire], wires=wire)
                for wire in range(n_qubits):
                    qml.CNOT(wires=[wire, (wire + 1) % n_qubits])
            return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits))

        return circuit

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.config.n_qubits:
            raise ValueError(
                f"noise must have shape (batch, {self.config.n_qubits})"
            )
        if not torch.isfinite(noise).all():
            raise ValueError("noise must contain only finite values")
        noise = noise.to(dtype=torch.float64)
        # The QNode is broadcast over the batch dimension: one simulator call
        # per batch instead of one per sample. Samples stay independent.
        expectations = self._circuit(noise, self.weights)
        return torch.stack(expectations, dim=-1)

    def circuit_resources(self) -> dict[str, int]:
        """Return compiled gate count and depth for lineage diagnostics."""

        sample = torch.zeros(1, self.config.n_qubits, dtype=torch.float64)
        resources = qml.specs(self._circuit)(sample, self.weights)["resources"]
        return {
            "circuit_depth": int(resources.depth),
            "circuit_gate_count": int(resources.num_gates),
        }


class DecodedQuantumGenerator(nn.Module):
    """Compose quantum expectations with a differentiable classical decoder.

    The affine adapter is explicit: Pauli-Z expectations in ``[-1, 1]`` are
    mapped to the FR-2 angle domain ``[0, pi]`` before decoding.
    """

    def __init__(self, quantum: QuantumGenerator, decoder: nn.Module) -> None:
        super().__init__()
        self.quantum = quantum
        self.decoder = decoder

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        expectations = self.quantum(noise)
        decoder_input = (expectations + 1.0) * (torch.pi / 2.0)
        return self.decoder(decoder_input)
