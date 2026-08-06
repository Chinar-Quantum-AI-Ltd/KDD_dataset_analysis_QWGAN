from __future__ import annotations

import unittest

import torch

from cqai.qwgan import Critic, gradient_penalty


class GradientPenaltyTests(unittest.TestCase):
    def test_penalty_is_finite_non_negative_and_backwardable(self) -> None:
        torch.manual_seed(13)
        critic = Critic(input_dim=8, hidden_dims=(16, 8))
        real = torch.randn(4, 8, dtype=torch.float64)
        fake = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)

        penalty = gradient_penalty(critic, real, fake)
        penalty.backward()

        self.assertEqual(penalty.ndim, 0)
        self.assertTrue(torch.isfinite(penalty))
        self.assertGreaterEqual(float(penalty.detach()), 0.0)
        self.assertIsNone(fake.grad)
        critic_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in critic.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(critic_gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
