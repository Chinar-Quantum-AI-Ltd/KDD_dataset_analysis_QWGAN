from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cqai.data import ContractSpec, build_train_contract
from cqai.qwgan import (
    QWGANConfig,
    StabilityThresholds,
    TrainingPlan,
    run_training,
    summarise_run,
    write_stability_report,
)

from ..fixtures import write_nslkdd_fixture

SPEC = ContractSpec(n_qubits=8, top_k=12, val_fraction=0.25, split_seed=3)
CONFIG = QWGANConfig(n_qubits=8, n_layers=3, n_critic=1, seed=5)
PLAN = TrainingPlan(
    attack_classes=("u2r",),
    seeds=(1, 2, 3),
    epochs=2,
    batch_size=4,
    critic_hidden_dims=(8,),
)


def _finished_run(root: Path):
    source = write_nslkdd_fixture(root / "KDDTrain+.txt", rows_per_label=24)
    contract = build_train_contract(source, root / "contract", spec=SPEC)
    return run_training(
        contract, config=CONFIG, plan=PLAN, output_dir=root / "runs"
    )


class SummariseRunTests(unittest.TestCase):
    def test_aggregates_every_seed_of_every_class(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = _finished_run(root)

            summary = summarise_run(result.root)

            self.assertEqual(summary["parent_run_id"], result.run_id)
            self.assertEqual(summary["requirement"], "FR-3")
            self.assertEqual(len(summary["parent_manifest_sha256"]), 64)
            self.assertEqual(set(summary["classes"]), {"u2r"})

            entry = summary["classes"]["u2r"]
            self.assertEqual(entry["seeds"], [1, 2, 3])
            self.assertEqual(entry["seed_count"], 3)
            for field in (
                "final_wasserstein_estimate",
                "median_generator_gradient_variance",
                "median_diversity_ratio",
            ):
                self.assertEqual(
                    set(entry[field]), {"mean", "std", "min", "max"}, field
                )
            # One averaged point per epoch, so a reader can see the trend
            # rather than a single end-of-run number.
            trajectory = entry["epoch_wasserstein_mean"]
            self.assertEqual(len(trajectory), PLAN.epochs)
            self.assertAlmostEqual(
                entry["wasserstein_trend"], trajectory[-1] - trajectory[0]
            )
            self.assertIn("stable", entry)
            self.assertIn("reasons", entry)

    def test_the_trend_is_recorded_but_never_gates_the_verdict(self) -> None:
        """Agreement across seeds and convergence are different questions.

        A still-rising Wasserstein estimate means the critic is outpacing the
        generator — real information, but not evidence that the seeds disagree.
        Recording it without gating on it keeps the verdict honest in both
        directions.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = _finished_run(root)

            summary = summarise_run(result.root)
            entry = summary["classes"]["u2r"]
            self.assertIn("wasserstein_trend", entry)
            self.assertNotIn("wasserstein_trend", entry["reasons"])
            self.assertNotIn("not_converged", entry["reasons"])

    def test_a_single_seed_is_never_reported_as_stable(self) -> None:
        """Three seeds is an FR-6 requirement, not a nicety.

        One seed cannot distinguish a stable configuration from a lucky one,
        so the summary must refuse the claim rather than report ``stable``.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_nslkdd_fixture(root / "KDDTrain+.txt", rows_per_label=24)
            contract = build_train_contract(source, root / "contract", spec=SPEC)
            result = run_training(
                contract,
                config=CONFIG,
                plan=TrainingPlan(
                    attack_classes=("u2r",),
                    seeds=(1,),
                    epochs=1,
                    batch_size=4,
                    critic_hidden_dims=(8,),
                ),
                output_dir=root / "runs",
            )

            summary = summarise_run(result.root)
            entry = summary["classes"]["u2r"]
            self.assertFalse(entry["stable"])
            self.assertIn("insufficient_seeds", entry["reasons"])
            self.assertFalse(summary["stable"])


class StabilityVerdictTests(unittest.TestCase):
    """The verdict is computed from the manifest, so it is testable directly."""

    def _summary(self, results: list[dict], **overrides) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "diagnostics.jsonl").write_text("", encoding="utf-8")
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": "synthetic",
                        "requirement": "FR-3",
                        "seeds": [13, 42, 1337],
                        "results": results,
                    }
                ),
                encoding="utf-8",
            )
            return summarise_run(run_dir, thresholds=StabilityThresholds(**overrides))

    @staticmethod
    def _result(seed: int, **overrides) -> dict:
        entry = {
            "attack_class": "r2l",
            "seed": seed,
            "training_rows": 797,
            "epochs_completed": 30,
            "final_wasserstein_estimate": 1.0,
            "median_generator_gradient_variance": 1e-5,
            "median_diversity_ratio": 0.4,
            "barren_plateau_suspected": False,
            "mode_collapse_suspected": False,
        }
        entry.update(overrides)
        return entry

    def test_agreeing_seeds_are_stable(self) -> None:
        summary = self._summary(
            [
                self._result(13, final_wasserstein_estimate=1.00),
                self._result(42, final_wasserstein_estimate=1.05),
                self._result(1337, final_wasserstein_estimate=0.98),
            ]
        )
        self.assertTrue(summary["classes"]["r2l"]["stable"])
        self.assertEqual(summary["classes"]["r2l"]["reasons"], [])
        self.assertTrue(summary["stable"])

    def test_wildly_disagreeing_seeds_are_not_stable(self) -> None:
        summary = self._summary(
            [
                self._result(13, final_wasserstein_estimate=0.5),
                self._result(42, final_wasserstein_estimate=48.0),
                self._result(1337, final_wasserstein_estimate=-12.0),
            ]
        )
        entry = summary["classes"]["r2l"]
        self.assertFalse(entry["stable"])
        self.assertIn("wasserstein_spread", entry["reasons"])

    def test_a_near_zero_mean_with_a_tiny_spread_is_not_flagged(self) -> None:
        """Guards the coefficient-of-variation trap.

        Three seeds converging on ~0 have a huge *relative* spread and a
        negligible absolute one. Flagging that would punish the best case, so
        the spread rule requires both to be exceeded.
        """

        summary = self._summary(
            [
                self._result(13, final_wasserstein_estimate=0.001),
                self._result(42, final_wasserstein_estimate=-0.002),
                self._result(1337, final_wasserstein_estimate=0.0005),
            ]
        )
        self.assertTrue(summary["classes"]["r2l"]["stable"])

    def test_a_flag_on_any_single_seed_sinks_the_class(self) -> None:
        barren = self._summary(
            [
                self._result(13),
                self._result(42, barren_plateau_suspected=True),
                self._result(1337),
            ]
        )
        self.assertFalse(barren["classes"]["r2l"]["stable"])
        self.assertIn("barren_plateau", barren["classes"]["r2l"]["reasons"])
        self.assertEqual(barren["classes"]["r2l"]["barren_plateau_seeds"], [42])

        collapsed = self._summary(
            [
                self._result(13),
                self._result(42),
                self._result(1337, mode_collapse_suspected=True),
            ]
        )
        self.assertFalse(collapsed["classes"]["r2l"]["stable"])
        self.assertIn("mode_collapse", collapsed["classes"]["r2l"]["reasons"])
        self.assertEqual(
            collapsed["classes"]["r2l"]["mode_collapse_seeds"], [1337]
        )

    def test_vanishing_gradients_are_reported_even_without_the_run_flag(self) -> None:
        """The monitor threshold and the report threshold are independent.

        A run configured with a permissive ``barren_plateau_variance`` must not
        be able to launder a dead gradient signal past the report.
        """

        summary = self._summary(
            [self._result(seed, median_generator_gradient_variance=1e-16)
             for seed in (13, 42, 1337)],
            min_median_gradient_variance=1e-12,
        )
        entry = summary["classes"]["r2l"]
        self.assertFalse(entry["stable"])
        self.assertIn("vanishing_gradients", entry["reasons"])


class WriteStabilityReportTests(unittest.TestCase):
    def test_writes_json_and_markdown_without_touching_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = _finished_run(root)
            manifest_before = (result.root / "manifest.json").read_bytes()

            path = write_stability_report(result.root)

            self.assertEqual(path, result.root / "stability.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["parent_run_id"], result.run_id)
            self.assertIn("thresholds", payload)

            markdown = (result.root / "stability.md").read_text(encoding="utf-8")
            self.assertIn(result.run_id, markdown)
            self.assertIn("u2r", markdown)

            # The run's own evidence is immutable: a derived report must never
            # rewrite the manifest it was computed from.
            self.assertEqual(
                (result.root / "manifest.json").read_bytes(), manifest_before
            )


if __name__ == "__main__":
    unittest.main()
