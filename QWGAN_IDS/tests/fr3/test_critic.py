from __future__ import annotations

import unittest

import torch
from torch import nn

from cqai.qwgan import Critic


class CriticTests(unittest.TestCase):
    def test_returns_one_unbounded_score_without_prohibited_layers(self) -> None:
        critic = Critic(input_dim=8, hidden_dims=(16, 8), use_layer_norm=True)
        scores = critic(torch.randn(4, 8))

        self.assertEqual(scores.shape, (4, 1))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(any(isinstance(m, nn.LeakyReLU) for m in critic.modules()))
        self.assertTrue(any(isinstance(m, nn.LayerNorm) for m in critic.modules()))
        self.assertFalse(any(isinstance(m, nn.BatchNorm1d) for m in critic.modules()))
        self.assertFalse(any(isinstance(m, nn.Sigmoid) for m in critic.modules()))


if __name__ == "__main__":
    unittest.main()
