from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from .config import QWGANConfig
from .critic import Critic
from .data_contract import TrainingAngles
from .generator import QuantumGenerator
from .losses import gradient_penalty_and_norm


class QWGANTrainer:
    """Alternating optimizer for the FR-3 hybrid WGAN-GP core."""

    def __init__(
        self,
        config: QWGANConfig,
        *,
        generator: QuantumGenerator | None = None,
        critic: Critic | None = None,
        critic_hidden_dims: Sequence[int] = (128, 64),
    ) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self._random = torch.Generator().manual_seed(config.seed)
        self.critic_hidden_dims = tuple(critic_hidden_dims)
        self.generator = generator or QuantumGenerator(config)
        self.critic = critic or Critic(
            input_dim=config.n_qubits,
            hidden_dims=self.critic_hidden_dims,
        )
        optimizer_args = {
            "lr": config.learning_rate,
            "betas": (config.beta1, config.beta2),
        }
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), **optimizer_args
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), **optimizer_args
        )
        self.global_step = 0

    def _sample_noise(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return torch.rand(
            batch_size,
            self.config.n_qubits,
            dtype=torch.float64,
            generator=generator if generator is not None else self._random,
        ) * (self.config.latent_scale * float(np.pi))

    @torch.no_grad()
    def sample_noise(
        self, batch_size: int, *, seed: int | None = None
    ) -> torch.Tensor:
        """Latent noise, optionally drawn from an independently seeded stream.

        FR-4 synthesis passes an explicit seed so its samples depend on the
        synthesis config rather than on the epoch a checkpoint happened to be
        taken at: a restored trainer's RNG carries the training history with it,
        which would make a run unreplayable from a different checkpoint.
        """

        stream = (
            torch.Generator().manual_seed(seed) if seed is not None else None
        )
        return self._sample_noise(batch_size, stream)

    @torch.no_grad()
    def generate_from_noise(self, noise: torch.Tensor) -> torch.Tensor:
        """Map explicit latent noise to the FR-2 angle domain."""

        return self._to_angle_domain(
            self.generator(noise.to(dtype=torch.float64))
        )

    @staticmethod
    def _to_angle_domain(expectations: torch.Tensor) -> torch.Tensor:
        return (expectations + 1.0) * (float(np.pi) / 2.0)

    @torch.no_grad()
    def generate(self, batch_size: int) -> torch.Tensor:
        """Draw ``batch_size`` synthetic samples in the FR-2 angle domain.

        This is the read-only view of the generator used by diagnostics and,
        later, by FR-4 synthesis. It never touches optimizer state.
        """

        return self.generate_from_noise(self._sample_noise(batch_size))

    def critic_step(self, real: torch.Tensor) -> dict[str, float]:
        real = real.to(dtype=torch.float64)
        self.generator_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)

        noise = self._sample_noise(real.shape[0])
        fake = self._to_angle_domain(self.generator(noise)).detach()
        real_scores = self.critic(real)
        fake_scores = self.critic(fake)
        penalty, gradient_norm = gradient_penalty_and_norm(
            self.critic,
            real,
            fake,
            random_generator=self._random,
        )
        wasserstein_estimate = real_scores.mean() - fake_scores.mean()
        loss = (
            fake_scores.mean()
            - real_scores.mean()
            + self.config.lambda_gp * penalty
        )
        loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": float(loss.detach()),
            "gradient_penalty": float(penalty.detach()),
            "critic_gradient_norm": float(gradient_norm),
            "wasserstein_estimate": float(wasserstein_estimate.detach()),
        }

    def generator_step(self, batch_size: int) -> dict[str, float]:
        self.generator_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)

        try:
            noise = self._sample_noise(batch_size)
            fake = self._to_angle_domain(self.generator(noise))
            loss = -self.critic(fake).mean()
            loss.backward()
            gradients = torch.cat(
                [
                    parameter.grad.detach().flatten()
                    for parameter in self.generator.parameters()
                    if parameter.grad is not None
                ]
            )
            gradient_variance = gradients.var(unbiased=False)
            self.generator_optimizer.step()
        finally:
            for parameter in self.critic.parameters():
                parameter.requires_grad_(True)

        return {
            "generator_loss": float(loss.detach()),
            "generator_gradient_variance": float(gradient_variance),
        }

    def train_step(self, batch: TrainingAngles) -> dict[str, float | int | str]:
        """Run ``n_critic`` critic updates and one generator update."""

        started = perf_counter()
        critic_metrics: dict[str, float] = {}
        for _ in range(self.config.n_critic):
            critic_metrics = self.critic_step(batch.values)
        generator_metrics = self.generator_step(batch.values.shape[0])
        self.global_step += 1
        resources = self.generator.circuit_resources()

        return {
            "critic_loss": critic_metrics["critic_loss"],
            "generator_loss": generator_metrics["generator_loss"],
            "wasserstein_estimate": critic_metrics["wasserstein_estimate"],
            "critic_gradient_norm": critic_metrics["critic_gradient_norm"],
            "generator_gradient_variance": generator_metrics[
                "generator_gradient_variance"
            ],
            "gradient_penalty": critic_metrics["gradient_penalty"],
            **resources,
            "wall_time_seconds": perf_counter() - started,
            "device": self.config.backend,
            "estimated_cost_usd": 0.0,
            "global_step": self.global_step,
            "attack_class": batch.attack_class,
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Persist model, optimizer, configuration, and lineage metadata."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "critic_hidden_dims": self.critic_hidden_dims,
                "generator_state": self.generator.state_dict(),
                "critic_state": self.critic.state_dict(),
                "generator_optimizer_state": self.generator_optimizer.state_dict(),
                "critic_optimizer_state": self.critic_optimizer.state_dict(),
                "random_generator_state": self._random.get_state(),
                "global_step": self.global_step,
                "metadata": dict(metadata or {}),
            },
            destination,
        )
        return destination

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
    ) -> tuple["QWGANTrainer", dict[str, Any]]:
        """Restore a trainer and return its caller-supplied metadata."""

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        trainer = cls(
            QWGANConfig(**checkpoint["config"]),
            critic_hidden_dims=tuple(checkpoint["critic_hidden_dims"]),
        )
        trainer.generator.load_state_dict(checkpoint["generator_state"])
        trainer.critic.load_state_dict(checkpoint["critic_state"])
        trainer.generator_optimizer.load_state_dict(
            checkpoint["generator_optimizer_state"]
        )
        trainer.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer_state"]
        )
        trainer._random.set_state(checkpoint["random_generator_state"])
        trainer.global_step = int(checkpoint["global_step"])
        return trainer, dict(checkpoint["metadata"])
