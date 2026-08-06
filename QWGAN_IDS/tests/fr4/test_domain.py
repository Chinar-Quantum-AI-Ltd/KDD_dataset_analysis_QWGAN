from __future__ import annotations

import unittest

import pandas as pd

from cqai.fidelity import RULES, check_domain

VALID_ROW = {
    "src_bytes": 491.0,
    "dst_bytes": 0.0,
    "count": 2.0,
    "srv_count": 2.0,
    "dst_host_count": 150.0,
    "dst_host_srv_count": 25.0,
    "same_srv_rate": 1.0,
    "diff_srv_rate": 0.0,
    "serror_rate": 0.0,
    "srv_serror_rate": 0.0,
    "dst_host_same_srv_rate": 0.17,
    "dst_host_diff_srv_rate": 0.03,
    "dst_host_serror_rate": 0.0,
    "dst_host_srv_serror_rate": 0.0,
    "dst_host_same_src_port_rate": 0.17,
    "dst_host_srv_diff_host_rate": 0.0,
    "logged_in": 1.0,
    "flag_SF": 1.0,
    "flag_S0": 0.0,
    "service_http": 1.0,
}


def _frame(*overrides: dict[str, float]) -> pd.DataFrame:
    rows = [dict(VALID_ROW) | dict(override) for override in (overrides or ({},))]
    return pd.DataFrame(rows)


class DomainRuleTests(unittest.TestCase):
    def test_a_valid_frame_is_fully_valid(self) -> None:
        report = check_domain(_frame())

        self.assertEqual(report.valid_fraction, 1.0)
        self.assertEqual(report.violations, {})
        self.assertEqual(report.rows, 1)

    def test_negative_bytes_are_rejected_and_nothing_else_fires(self) -> None:
        """A failure must name its rule, not just lower a score.

        A single validity fraction tells nobody what to fix; the per-rule
        breakdown is what makes a rejected batch actionable.
        """

        report = check_domain(_frame({"src_bytes": -1.0}))

        self.assertEqual(report.valid_fraction, 0.0)
        self.assertEqual(set(report.violations), {"src_bytes"})
        self.assertEqual(report.violations["src_bytes"], 1.0)

    def test_a_rate_outside_the_unit_interval_is_rejected(self) -> None:
        report = check_domain(_frame({"same_srv_rate": 1.4}))
        self.assertEqual(set(report.violations), {"same_srv_rate"})

    def test_counts_above_the_nsl_kdd_cap_are_rejected(self) -> None:
        report = check_domain(
            _frame({"count": 900.0}, {"dst_host_count": 300.0}, {})
        )
        self.assertEqual(set(report.violations), {"count", "dst_host_count"})
        # Two of three rows are bad, one per rule.
        self.assertAlmostEqual(report.valid_fraction, 1 / 3)
        self.assertAlmostEqual(report.violations["count"], 1 / 3)

    def test_an_indicator_between_zero_and_one_is_rejected(self) -> None:
        """One-hot columns survive PCA as continuous values.

        A decoded 0.5 is not a category; accepting it would push a value into
        the augmented set that the encoder could never have produced.
        """

        report = check_domain(_frame({"logged_in": 0.5}))
        self.assertEqual(set(report.violations), {"logged_in"})

        near_one = check_domain(_frame({"logged_in": 0.999}))
        self.assertEqual(near_one.violations, {})

    def test_mutually_exclusive_flags_cannot_both_be_set(self) -> None:
        report = check_domain(_frame({"flag_SF": 1.0, "flag_S0": 1.0}))
        self.assertIn("flag_exclusive", report.violations)

    def test_a_non_finite_value_is_a_violation_not_a_crash(self) -> None:
        report = check_domain(_frame({"src_bytes": float("nan")}))
        self.assertEqual(report.valid_fraction, 0.0)
        self.assertIn("src_bytes", report.violations)

    def test_columns_the_contract_did_not_select_are_simply_not_checked(self) -> None:
        """The contract selects 20 of the expanded feature space by MI.

        A different contract spec selects a different subset, so the rule table
        must apply to whatever is present rather than demanding a fixed schema.
        """

        subset = _frame()[["src_bytes", "same_srv_rate", "logged_in"]]
        report = check_domain(subset)

        self.assertEqual(report.valid_fraction, 1.0)
        self.assertEqual(report.checked_columns, ["src_bytes", "same_srv_rate", "logged_in"])

    def test_the_rule_table_does_not_treat_count_columns_as_binary(self) -> None:
        """Guards against reusing ``src.preprocessing.BINARY_COLS``.

        That list is named "binary" but contains count columns such as ``hot``,
        ``num_root`` and ``num_compromised``. Driving the rule table from it
        would license any value in {0, 1} for a counter and silently accept
        invalid records.
        """

        for name in ("hot", "num_root", "num_compromised", "num_file_creations"):
            self.assertNotIn(
                name,
                [column for column, rule in RULES.items() if rule.kind == "indicator"],
            )


if __name__ == "__main__":
    unittest.main()
