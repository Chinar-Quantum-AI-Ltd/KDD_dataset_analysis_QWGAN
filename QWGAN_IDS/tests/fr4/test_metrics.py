from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cqai.fidelity import c2st_auc, coverage, ks_statistic, wasserstein_1

COLUMNS = ["a", "b", "c"]


def _frame(rows: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


def _sample(seed: int, *, shift: float = 0.0, scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return _frame(rng.normal(loc=shift, scale=scale, size=(400, len(COLUMNS))))


class DistanceMetricTests(unittest.TestCase):
    """W-1 and KS must be near zero on samples of the same distribution."""

    def test_same_distribution_is_close_to_zero(self) -> None:
        real, synthetic = _sample(0), _sample(1)

        w = wasserstein_1(real, synthetic)
        ks = ks_statistic(real, synthetic)

        self.assertEqual(set(w["per_feature"]), set(COLUMNS))
        self.assertLess(w["max"], 0.25)
        self.assertLess(ks["max"], 0.15)
        # The summaries must be consistent with the per-feature values, so a
        # threshold can be applied to either without them disagreeing.
        self.assertAlmostEqual(w["max"], max(w["per_feature"].values()))
        self.assertAlmostEqual(ks["mean"], float(np.mean(list(ks["per_feature"].values()))))

    def test_a_shifted_distribution_moves_both_metrics_up(self) -> None:
        real = _sample(0)
        shifted = _sample(1, shift=3.0)

        self.assertGreater(wasserstein_1(real, shifted)["max"], 2.0)
        self.assertGreater(ks_statistic(real, shifted)["max"], 0.8)

    def test_wasserstein_is_normalised_so_one_band_fits_every_feature(self) -> None:
        """Raw W-1 is scale-dependent; a shared threshold needs a shared scale.

        ``src_bytes`` runs to thousands and ``same_srv_rate`` is bounded by 1.
        A raw band loose enough for the first cannot constrain the second at
        all, so the gate uses the spread-normalised values.
        """

        rng = np.random.default_rng(0)
        real = pd.DataFrame(
            {"big": rng.normal(0, 1000, 400), "small": rng.normal(0, 1, 400)}
        )
        # Both features are off by half of their own spread.
        synthetic = pd.DataFrame(
            {"big": real["big"] + 500.0, "small": real["small"] + 0.5}
        )

        w = wasserstein_1(real, synthetic)

        self.assertGreater(w["per_feature"]["big"], 100 * w["per_feature"]["small"])
        # After normalisation the two are comparable, which is the point.
        self.assertAlmostEqual(
            w["per_feature_normalised"]["big"],
            w["per_feature_normalised"]["small"],
            delta=0.05,
        )
        self.assertEqual(w["max_normalised"], max(w["per_feature_normalised"].values()))

    def test_a_constant_real_column_does_not_divide_by_zero(self) -> None:
        real = _frame(np.zeros((100, len(COLUMNS))))
        synthetic = _frame(np.ones((100, len(COLUMNS))))

        w = wasserstein_1(real, synthetic)

        self.assertTrue(all(np.isfinite(list(w["per_feature_normalised"].values()))))
        self.assertAlmostEqual(w["max_normalised"], 1.0)

    def test_metrics_are_computed_per_feature_not_pooled(self) -> None:
        """One bad feature must be visible, not averaged away.

        A generator that nails two features and destroys the third is a failure;
        pooling every column into one number would hide exactly that.
        """

        real = _sample(0)
        synthetic = _sample(1)
        synthetic["c"] = synthetic["c"] + 5.0

        w = wasserstein_1(real, synthetic)
        self.assertGreater(w["per_feature"]["c"], 4.0)
        self.assertLess(w["per_feature"]["a"], 0.25)
        self.assertEqual(w["max"], w["per_feature"]["c"])


class C2STTests(unittest.TestCase):
    def test_indistinguishable_samples_score_near_one_half(self) -> None:
        auc = c2st_auc(_sample(0), _sample(1), seed=7)

        self.assertAlmostEqual(auc["mean"], 0.5, delta=0.12)
        self.assertEqual(auc["n_per_side"], 400)
        self.assertEqual(auc["folds"], 5)
        self.assertGreaterEqual(auc["std"], 0.0)

    def test_a_shifted_distribution_is_separable(self) -> None:
        auc = c2st_auc(_sample(0), _sample(1, shift=3.0), seed=7)
        self.assertGreater(auc["mean"], 0.95)

    def test_the_synthetic_side_is_balanced_against_the_real_side(self) -> None:
        """Class imbalance inflates AUC; the comparison must be balanced.

        The synthetic pool is far larger than the held-out real one by design
        (a ratio sweep produces tens of thousands of samples against 198 real
        rows), so balancing is not optional.
        """

        real = _sample(0).iloc[:50]
        synthetic = _sample(1)

        auc = c2st_auc(real, synthetic, seed=7)
        self.assertEqual(auc["n_per_side"], 50)

    def test_the_same_seed_replays(self) -> None:
        first = c2st_auc(_sample(0), _sample(1), seed=3)
        second = c2st_auc(_sample(0), _sample(1), seed=3)
        self.assertEqual(first["mean"], second["mean"])


class CoverageTests(unittest.TestCase):
    def test_matching_samples_cover_the_real_modes(self) -> None:
        self.assertGreater(coverage(_sample(0), _sample(1)), 0.7)

    def test_a_collapsed_generator_covers_almost_nothing(self) -> None:
        """The signal FR-3's diversity ratio cannot give on its own.

        A generator emitting one point repeatedly can still be scored as
        "not collapsed" by a spread ratio if the real class is tight. Coverage
        asks the sharper question: are the real modes actually reached?
        """

        real = _sample(0)
        collapsed = _frame(np.zeros((400, len(COLUMNS))))

        self.assertLess(coverage(real, collapsed), 0.1)

    def test_covering_one_mode_of_two_scores_near_one_half(self) -> None:
        rng = np.random.default_rng(0)
        mode_a = rng.normal(loc=0.0, scale=0.3, size=(200, len(COLUMNS)))
        mode_b = rng.normal(loc=10.0, scale=0.3, size=(200, len(COLUMNS)))
        real = _frame(np.vstack([mode_a, mode_b]))
        one_mode = _frame(
            rng.normal(loc=0.0, scale=0.3, size=(400, len(COLUMNS)))
        )

        self.assertAlmostEqual(coverage(real, one_mode), 0.5, delta=0.15)


if __name__ == "__main__":
    unittest.main()
