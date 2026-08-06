from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from cqai.data import ContractSpec, build_train_contract
from cqai.qwgan import QWGANConfig, TrainingPlan, run_training
from cqai.synthesis.cli import build_parser, load_experiment, main

from ..fixtures import write_nslkdd_fixture

SPEC = {"n_qubits": 8, "top_k": 12, "val_fraction": 0.25, "split_seed": 3}
CONFIG = QWGANConfig(n_qubits=8, n_layers=3, n_critic=1, seed=5)


def _experiment(template: str) -> dict:
    return {
        "version": "1.0.0",
        "contract": dict(SPEC),
        "thresholds": {
            "max_c2st_auc": 1.0,
            "max_wasserstein_1": 1e9,
            "max_ks_statistic": 1.0,
            "min_coverage": 0.0,
            "min_domain_validity": 0.0,
            "min_real_samples": 1,
        },
        "synthesis": {
            "attack_classes": ["u2r"],
            "seeds": [13],
            "ratios": [0.8],
            "majority_class": "dos",
            "checkpoints": {"u2r": template},
            "chunk_size": 64,
        },
    }


def _prepared(root: Path) -> tuple[Path, str]:
    source = write_nslkdd_fixture(root / "KDDTrain+.txt", rows_per_label=40)
    contract = build_train_contract(
        source, root / "contract", spec=ContractSpec(**SPEC)
    )
    training = run_training(
        contract,
        config=CONFIG,
        plan=TrainingPlan(
            attack_classes=("u2r",),
            seeds=(13,),
            epochs=1,
            batch_size=4,
            critic_hidden_dims=(8,),
        ),
        output_dir=root / "runs",
    )
    template = str(
        training.root / "checkpoints" / "u2r" / "seed{seed}" / "epoch0001.pt"
    )
    return source, template


class ExperimentConfigTests(unittest.TestCase):
    def test_resolves_to_spec_plan_and_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.yaml"
            path.write_text(yaml.safe_dump(_experiment("cp/seed{seed}.pt")))

            spec, plan, thresholds = load_experiment(path)

            self.assertEqual(spec.n_qubits, 8)
            self.assertEqual(plan.attack_classes, ("u2r",))
            self.assertEqual(plan.ratios, (0.8,))
            self.assertEqual(plan.majority_class, "dos")
            self.assertEqual(thresholds.min_real_samples, 1)

    def test_rejects_a_class_without_a_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document = _experiment("cp/seed{seed}.pt")
            document["synthesis"]["attack_classes"] = ["u2r", "r2l"]
            path = Path(directory) / "experiment.yaml"
            path.write_text(yaml.safe_dump(document))

            with self.assertRaises(ValueError):
                load_experiment(path)

    def test_rejects_a_checkpoint_template_without_a_seed_placeholder(self) -> None:
        """One generator per seed is an FR-6 requirement.

        A fixed path would gate the same model three times and report it as
        three seeds agreeing, which is the exact opposite of what three seeds
        are for.
        """

        with tempfile.TemporaryDirectory() as directory:
            document = _experiment("cp/fixed.pt")
            path = Path(directory) / "experiment.yaml"
            path.write_text(yaml.safe_dump(document))

            with self.assertRaises(ValueError) as caught:
                load_experiment(path)
            self.assertIn("seed", str(caught.exception))


class ShippedConfigTests(unittest.TestCase):
    def test_every_shipped_fr4_config_is_valid(self) -> None:
        """A directory sweep, so a new config cannot ship untested."""

        configs = sorted(
            (Path(__file__).resolve().parents[2] / "configs").glob("fr4_*.yaml")
        )
        self.assertTrue(configs, "no FR-4 experiment configs found")

        for shipped in configs:
            with self.subTest(config=shipped.name):
                spec, plan, thresholds = load_experiment(shipped)
                self.assertEqual(
                    len(plan.seeds), 3, "FR-6 requires exactly three seeds"
                )
                self.assertEqual(
                    thresholds.max_c2st_auc,
                    0.65,
                    "the TDD hard threshold must not be relaxed in a config",
                )
                # The TDD sweeps 0.2-0.4 and warns against naive 1:1 balancing.
                self.assertTrue(plan.ratios)
                for ratio in plan.ratios:
                    self.assertGreaterEqual(ratio, 0.2)
                    self.assertLessEqual(ratio, 0.4)
                self.assertGreaterEqual(spec.n_qubits, 8)
                self.assertLessEqual(spec.n_qubits, 12)


class CliRunTests(unittest.TestCase):
    def test_end_to_end_run_writes_a_gated_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, template = _prepared(root)
            experiment = root / "experiment.yaml"
            experiment.write_text(yaml.safe_dump(_experiment(template)))

            exit_code = main(
                [
                    "--experiment", str(experiment),
                    "--train-file", str(source),
                    "--contract-dir", str(root / "contract"),
                    "--output-dir", str(root / "synth"),
                    "--run-id", "cli-run",
                ]
            )

            run_dir = root / "synth" / "cli-run"
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["requirement"], "FR-4")
            self.assertTrue((run_dir / "accepted_manifest.json").is_file())
            self.assertTrue((run_dir / "quarantine_manifest.json").is_file())
            # Exit code is the release decision, not "the process did not throw".
            self.assertEqual(exit_code, 0 if manifest["accepted_batches"] else 2)

    def test_a_run_that_releases_nothing_does_not_exit_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, template = _prepared(root)
            document = _experiment(template)
            document["thresholds"]["min_domain_validity"] = 1.0
            document["thresholds"]["min_coverage"] = 1.0
            experiment = root / "experiment.yaml"
            experiment.write_text(yaml.safe_dump(document))

            exit_code = main(
                [
                    "--experiment", str(experiment),
                    "--train-file", str(source),
                    "--contract-dir", str(root / "contract"),
                    "--output-dir", str(root / "synth"),
                    "--run-id", "nothing-released",
                ]
            )

            self.assertEqual(exit_code, 2)
            manifest = json.loads(
                (root / "synth" / "nothing-released" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["accepted_batches"], 0)

    def test_dry_run_generates_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, template = _prepared(root)
            experiment = root / "experiment.yaml"
            experiment.write_text(yaml.safe_dump(_experiment(template)))

            exit_code = main(
                [
                    "--experiment", str(experiment),
                    "--train-file", str(source),
                    "--contract-dir", str(root / "contract"),
                    "--output-dir", str(root / "synth"),
                    "--dry-run",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertFalse((root / "synth").exists())

    def test_parser_requires_an_experiment_file(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args([])


if __name__ == "__main__":
    unittest.main()
