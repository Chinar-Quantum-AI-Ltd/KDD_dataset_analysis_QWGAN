from __future__ import annotations

import torch
from torch import nn


def gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    *,
    random_generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return the canonical per-sample WGAN-GP interpolation penalty."""

    penalty, _ = gradient_penalty_and_norm(
        critic, real, fake, random_generator=random_generator
    )
    return penalty


def gradient_penalty_and_norm(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    *,
    random_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the penalty and detached mean input-gradient norm."""

    if real.shape != fake.shape or real.ndim != 2:
        raise ValueError("real and fake must be two-dimensional batches of equal shape")
    if not torch.isfinite(real).all() or not torch.isfinite(fake).all():
        raise ValueError("real and fake must contain only finite values")

    detached_fake = fake.detach().to(dtype=real.dtype, device=real.device)
    epsilon = torch.rand(
        real.shape[0],
        1,
        dtype=real.dtype,
        device=real.device,
        generator=random_generator,
    )
    interpolated = epsilon * real + (1.0 - epsilon) * detached_fake
    interpolated.requires_grad_(True)

    scores = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient_norm = gradients.flatten(start_dim=1).norm(2, dim=1)
    penalty = ((gradient_norm - 1.0) ** 2).mean()
    return penalty, gradient_norm.detach().mean()
