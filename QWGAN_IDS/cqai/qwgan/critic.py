from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class Critic(nn.Module):
    """Classical WGAN-GP critic returning one unbounded score per sample."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        *,
        use_layer_norm: bool = True,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if not hidden_dims or any(width < 1 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")

        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(previous, width))
            if use_layer_norm:
                layers.append(nn.LayerNorm(width))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers).to(dtype=torch.float64)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        if samples.ndim != 2:
            raise ValueError("samples must be a two-dimensional batch")
        return self.network(samples.to(dtype=torch.float64))
