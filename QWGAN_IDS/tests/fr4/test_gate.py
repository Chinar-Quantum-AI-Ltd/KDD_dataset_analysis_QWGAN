from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cqai.fidelity import FidelityThresholds, evaluate_gate

COLUMNS = ["src_bytes", "same_srv_rate", "logged_in"]

# Permissive bands, so a test that expects a pass is not fighting the defaults.
LOOSE = FidelityThresholds(
    max_c2st_auc=0.65,
    max_wasserstein_1=5.0,
    max_ks_statistic=0.5,
    min_coverage=0.3,
    min_domain_validity=0.9,
    min_real_samples=50,
)


def _real(n: int = 400, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "src_bytes": rng.integers(0, 1000, size=n).astype(float),
            "same_srv_rate": rng.uniform(0.0, 1.0, size=n),
            "logged_in": rng.integers(0, 2, size=n).astype(float),
        }
    )


class GateVerdictTests(unittest.TestCase):
    def test_a_second_real_sample_passes(self) -> None:
        result = evaluate_gate(_real(seed=0), _real(seed=1), thresholds=LOOSE, seed=5)

        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["n_real"], 400)
        self.assertEqual(result["n_synthetic"], 400)
        for name in ("c2st", "wasserstein_1", "ks", "coverage", "domain"):
            self.assertIn(name, result["metrics"])

    def test_a_distinguishable_batch_fails_and_names_the_criterion(self) -> None:
        real = _real(seed=0)
        shifted = _real(seed=1)
        shifted["src_bytes"] = shifted["src_bytes"] + 5000.0

        result = evaluate_gate(real, shifted, thresholds=LOOSE, seed=5)

        self.assertEqual(result["verdict"], "fail")
        self.assertIn("c2st_distinguishable", result["reasons"])
        self.assertGreater(result["metrics"]["c2st"]["mean"], 0.65)

    def test_each_mandatory_criterion_can_fail_on_its_own(self) -> None:
        """Every gate signal must be able to reject a batch by itself.

        A gate where only the C2ST ever decides is a C2ST wearing four extra
        metrics as decoration.
        """

        real = _real(seed=0)

        # Domain validity alone: statistically identical, but invalid rows.
        invalid = _real(seed=1)
        invalid.loc[invalid.index[:200], "same_srv_rate"] = 4.2
        domain_result = evaluate_gate(
            real, invalid, thresholds=LOOSE, seed=5
        )
        self.assertEqual(domain_result["verdict"], "fail")
        self.assertIn("domain_invalid", domain_result["reasons"])

        # Coverage alone: a collapsed generator.
        collapsed = pd.DataFrame(
            {name: np.full(400, real[name].median()) for name in COLUMNS}
        )
        collapsed_result = evaluate_gate(
            real, collapsed, thresholds=LOOSE, seed=5
        )
        self.assertIn("low_coverage", collapsed_result["reasons"])

        # KS alone: same support and mean, different shape.
        reshaped = _real(seed=1)
        reshaped["same_srv_rate"] = np.clip(
            np.random.default_rng(2).normal(0.5, 0.02, size=400), 0.0, 1.0
        )
        ks_result = evaluate_gate(
            real,
            reshaped,
            thresholds=FidelityThresholds(
                max_c2st_auc=1.0,
                max_wasserstein_1=5.0,
                max_ks_statistic=0.2,
                min_coverage=0.0,
                min_domain_validity=0.9,
                min_real_samples=50,
            ),
            seed=5,
        )
        self.assertIn("ks_out_of_band", ks_result["reasons"])


class InsufficientEvidenceTests(unittest.TestCase):
    def test_a_thin_real_reference_yields_neither_pass_nor_fail(self) -> None:
        """NSL-KDD ``u2r`` has ten held-out rows.

        A C2ST AUC computed against ten real rows cannot support a claim in
        either direction. Returning ``pass`` there would be worse than useless,
        so the gate fails closed and says why.
        """

        result = evaluate_gate(
            _real(10, seed=0), _real(400, seed=1), thresholds=LOOSE, seed=5
        )

        self.assertEqual(result["verdict"], "insufficient_evidence")
        self.assertIn("insufficient_real_samples", result["reasons"])
        self.assertEqual(result["n_real"], 10)

    def test_domain_validity_is_still_reported_without_a_real_reference(self) -> None:
        """Domain validity needs no real data, so it is always computed.

        Even an uncertifiable class should tell us whether its samples are
        schema-valid at all.
        """

        invalid = _real(400, seed=1)
        invalid["same_srv_rate"] = 4.2

        result = evaluate_gate(
            _real(10, seed=0), invalid, thresholds=LOOSE, seed=5
        )

        self.assertEqual(result["verdict"], "insufficient_evidence")
        self.assertEqual(result["metrics"]["domain"]["valid_fraction"], 0.0)
        self.assertIn("domain_invalid", result["reasons"])

    def test_a_thin_reference_never_crashes_the_cross_validated_c2st(self) -> None:
        result = evaluate_gate(
            _real(3, seed=0), _real(400, seed=1), thresholds=LOOSE, seed=5
        )
        self.assertEqual(result["verdict"], "insufficient_evidence")
        self.assertIsNone(result["metrics"]["c2st"])


class NullReferenceTests(unittest.TestCase):
    def test_the_gate_records_what_real_data_scores_against_itself(self) -> None:
        """A threshold is only meaningful next to its null baseline.

        If splitting the real sample in half already yields C2ST 0.7 because
        the sample is small, then a synthetic 0.7 says nothing about the
        generator. Recording the null makes that visible instead of leaving a
        reader to assume 0.5.
        """

        result = evaluate_gate(_real(seed=0), _real(seed=1), thresholds=LOOSE, seed=5)

        null = result["null_reference"]
        self.assertIsNotNone(null)
        self.assertIn("c2st", null)
        self.assertAlmostEqual(null["c2st"]["mean"], 0.5, delta=0.2)
        self.assertEqual(null["n_per_side"], 200)

    def test_the_null_reference_is_skipped_when_the_sample_cannot_be_halved(self) -> None:
        """Eight rows halve to four, below the five folds the C2ST needs."""

        result = evaluate_gate(
            _real(8, seed=0),
            _real(400, seed=1),
            thresholds=FidelityThresholds(min_real_samples=5),
            seed=5,
        )
        self.assertIsNone(result["null_reference"])

    def test_a_criterion_real_data_also_fails_is_not_charged_to_the_generator(
        self,
    ) -> None:
        """The decode is lossy, so some bands reject real data too.

        Decoding genuine held-out NSL-KDD rows through the fitted PCA and
        scalers yields a domain validity of 0.0: the decode cannot reproduce a
        schema-valid record even from real data. Firing ``domain_invalid`` there
        would blame the generator for the decode's limits. A criterion counts
        only when the synthetic batch is worse than the configured band *and*
        worse than what real data of the same size scores.
        """

        # Real rows that themselves violate a rule, so the null baseline is bad.
        real = _real(seed=0)
        real["same_srv_rate"] = 4.2
        synthetic = _real(seed=1)
        synthetic["same_srv_rate"] = 4.2

        result = evaluate_gate(
            real,
            synthetic,
            thresholds=FidelityThresholds(
                max_c2st_auc=1.0,
                max_wasserstein_1=1e9,
                max_ks_statistic=1.0,
                min_coverage=0.0,
                min_domain_validity=0.95,
                min_real_samples=50,
            ),
            seed=5,
        )

        self.assertEqual(result["metrics"]["domain"]["valid_fraction"], 0.0)
        self.assertNotIn("domain_invalid", result["reasons"])
        self.assertEqual(result["verdict"], "pass")
        # The measurement is still recorded; only the blame is withheld.
        self.assertFalse(result["criteria"]["domain_validity"]["fired"])
        self.assertEqual(result["criteria"]["domain_validity"]["null"], 0.0)

    def test_a_criterion_still_fires_when_real_data_clears_it(self) -> None:
        """The escape hatch must not become a way through the gate."""

        real = _real(seed=0)
        invalid = _real(seed=1)
        invalid["same_srv_rate"] = 4.2

        result = evaluate_gate(real, invalid, thresholds=LOOSE, seed=5)

        self.assertIn("domain_invalid", result["reasons"])
        self.assertTrue(result["criteria"]["domain_validity"]["fired"])
        self.assertEqual(result["criteria"]["domain_validity"]["null"], 1.0)

    def test_every_criterion_records_its_value_threshold_and_null(self) -> None:
        result = evaluate_gate(_real(seed=0), _real(seed=1), thresholds=LOOSE, seed=5)

        for name in (
            "c2st_auc",
            "wasserstein_1",
            "ks_statistic",
            "coverage",
            "domain_validity",
        ):
            criterion = result["criteria"][name]
            self.assertIn("value", criterion, name)
            self.assertIn("threshold", criterion, name)
            self.assertIn("null", criterion, name)
            self.assertIn("fired", criterion, name)

    def test_without_a_null_the_absolute_band_decides_alone(self) -> None:
        """A reference too small to halve cannot calibrate anything.

        Falling back to the absolute band keeps the gate strict rather than
        letting a missing baseline wave a batch through.
        """

        real = _real(60, seed=0)
        invalid = _real(400, seed=1)
        invalid["same_srv_rate"] = 4.2

        result = evaluate_gate(
            real,
            invalid,
            thresholds=FidelityThresholds(
                max_c2st_auc=1.0,
                max_wasserstein_1=1e9,
                max_ks_statistic=1.0,
                min_coverage=0.0,
                min_domain_validity=0.95,
                min_real_samples=5,
                c2st_folds=40,  # 30 per side < 40 folds, so no null is computed
            ),
            seed=5,
        )

        self.assertIsNone(result["null_reference"])
        self.assertIn("domain_invalid", result["reasons"])


class ThresholdRecordTests(unittest.TestCase):
    def test_the_resolved_thresholds_travel_with_the_result(self) -> None:
        """A pass/fail claim is unreadable without the bands that produced it."""

        result = evaluate_gate(_real(seed=0), _real(seed=1), thresholds=LOOSE, seed=5)

        self.assertEqual(result["thresholds"]["max_c2st_auc"], 0.65)
        self.assertEqual(result["thresholds"]["min_real_samples"], 50)
        self.assertEqual(result["seed"], 5)

    def test_the_tdd_hard_threshold_is_the_default(self) -> None:
        self.assertEqual(FidelityThresholds().max_c2st_auc, 0.65)


if __name__ == "__main__":
    unittest.main()
