from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cqai.fidelity import FidelityThresholds, evaluate_gate, novelty_ratio

COLUMNS = ["a", "b", "c"]


def _rows(n: int, *, seed: int, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(shift, 1.0, size=(n, len(COLUMNS))), columns=COLUMNS
    )


class NoveltyRatioTests(unittest.TestCase):
    """Distance to the training set, relative to what real held-out data shows.

    An absolute distance says nothing on its own -- a naturally dense class sits
    close to itself. The reference is how far genuine held-out rows of the same
    class fall from the training set.
    """

    def test_independent_samples_score_around_one(self) -> None:
        train = _rows(400, seed=0)
        held_out = _rows(200, seed=1)
        synthetic = _rows(400, seed=2)

        self.assertAlmostEqual(
            novelty_ratio(synthetic, train, held_out), 1.0, delta=0.35
        )

    def test_copying_the_training_set_scores_near_zero(self) -> None:
        """The failure this exists to catch.

        A model that emits its training rows passes a two-sample test against
        held-out data trivially, and contributes nothing to an augmented set.
        """

        train = _rows(400, seed=0)
        held_out = _rows(200, seed=1)

        self.assertLess(novelty_ratio(train.copy(), train, held_out), 0.05)

    def test_a_far_away_generator_scores_above_one(self) -> None:
        train = _rows(400, seed=0)
        held_out = _rows(200, seed=1)
        distant = _rows(400, seed=2, shift=5.0)

        self.assertGreater(novelty_ratio(distant, train, held_out), 2.0)


class GateNoveltyTests(unittest.TestCase):
    LOOSE = FidelityThresholds(
        max_c2st_auc=1.0,
        max_wasserstein_1=1e9,
        max_ks_statistic=1.0,
        min_coverage=0.0,
        min_domain_validity=0.0,
        min_real_samples=1,
    )

    def test_the_criterion_is_skipped_without_a_training_reference(self) -> None:
        """Backwards compatible: an existing caller keeps its verdict.

        Every reported gate result was produced without this signal, and
        silently adding it would change what those runs mean.
        """

        result = evaluate_gate(
            _rows(200, seed=1), _rows(400, seed=2), thresholds=self.LOOSE, seed=5
        )

        self.assertIsNone(result["criteria"]["novelty"]["value"])
        self.assertFalse(result["criteria"]["novelty"]["fired"])
        self.assertEqual(result["verdict"], "pass")

    def test_a_memorising_batch_is_rejected(self) -> None:
        train = _rows(400, seed=0)
        held_out = _rows(200, seed=1)

        result = evaluate_gate(
            held_out,
            train.copy(),
            train_reference=train,
            thresholds=self.LOOSE,
            seed=5,
        )

        self.assertEqual(result["verdict"], "fail")
        self.assertIn("memorised_training_data", result["reasons"])

    def test_a_genuinely_novel_batch_is_not_rejected(self) -> None:
        train = _rows(400, seed=0)
        held_out = _rows(200, seed=1)

        result = evaluate_gate(
            held_out,
            _rows(400, seed=2),
            train_reference=train,
            thresholds=self.LOOSE,
            seed=5,
        )

        self.assertNotIn("memorised_training_data", result["reasons"])
        self.assertEqual(result["verdict"], "pass")

    def test_novelty_is_recorded_next_to_its_threshold(self) -> None:
        train = _rows(400, seed=0)
        result = evaluate_gate(
            _rows(200, seed=1),
            _rows(400, seed=2),
            train_reference=train,
            thresholds=self.LOOSE,
            seed=5,
        )

        criterion = result["criteria"]["novelty"]
        self.assertIsNotNone(criterion["value"])
        self.assertEqual(
            criterion["threshold"], self.LOOSE.min_novelty_ratio
        )
        # Real held-out data is the reference, so its own ratio is 1 by
        # construction; recording it keeps the number readable.
        self.assertEqual(criterion["null"], 1.0)


if __name__ == "__main__":
    unittest.main()
