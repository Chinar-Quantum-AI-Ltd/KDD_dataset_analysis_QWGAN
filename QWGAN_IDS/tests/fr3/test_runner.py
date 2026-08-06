from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cqai.data import ContractSpec, build_train_contract
from cqai.qwgan import (
    QWGANConfig,
    QWGANTrainer,
    TrainingPlan,
    detect_barren_plateau,
    diversity_ratio,
    run_training,
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


def _contract(root: Path):
    source = write_nslkdd_fixture(root / "KDDTrain+.txt", rows_per_label=24)
    return build_train_contract(source, root / "contract", spec=SPEC)


class MonitorTests(unittest.TestCase):
    """The monitors are pure so a plateau can be tested without inducing one."""

    def test_barren_plateau_fires_only_on_vanishing_gradients(self) -> None:
        self.assertTrue(detect_barren_plateau([1e-18, 1e-20, 1e-19], 1e-12))
        self.assertFalse(detect_barren_plateau([1e-4, 1e-5, 1e-3], 1e-12))
        # A single healthy step among dead ones must not clear the flag: the
        # decision is on the median, not the maximum.
        self.assertTrue(detect_barren_plateau([1e-18, 1e-20, 1.0], 1e-12))
        self.assertFalse(detect_barren_plateau([], 1e-12))

    def test_diversity_ratio_detects_collapse_against_the_real_batch(self) -> None:
        real = torch.rand(32, 8, dtype=torch.float64) * np.pi
        collapsed = torch.full((32, 8), 1.5, dtype=torch.float64)
        healthy = torch.rand(32, 8, dtype=torch.float64) * np.pi

        self.assertLess(diversity_ratio(collapsed, real), 0.05)
        self.assertGreater(diversity_ratio(healthy, real), 0.5)


class RunnerTests(unittest.TestCase):
    def test_writes_diagnostics_checkpoints_and_a_complete_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = _contract(root)

            result = run_training(
                contract, config=CONFIG, plan=PLAN, output_dir=root / "runs"
            )

            run_dir = result.root
            self.assertTrue(run_dir.is_dir())

            # -- diagnostics: one JSONL record per training step ------------- #
            lines = [
                json.loads(line)
                for line in (run_dir / "diagnostics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertTrue(lines)
            required = {
                "run_id",
                "attack_class",
                "seed",
                "epoch",
                "step_in_epoch",
                "global_step",
                "critic_loss",
                "generator_loss",
                "wasserstein_estimate",
                "critic_gradient_norm",
                "generator_gradient_variance",
                "gradient_penalty",
                "diversity_ratio",
                "circuit_depth",
                "circuit_gate_count",
                "wall_time_seconds",
                "device",
                "estimated_cost_usd",
            }
            for record in lines:
                self.assertTrue(required <= set(record), required - set(record))
            self.assertEqual({record["seed"] for record in lines}, {1, 2, 3})
            self.assertEqual({record["epoch"] for record in lines}, {1, 2})
            self.assertEqual({record["attack_class"] for record in lines}, {"u2r"})

            # -- checkpoints: one per epoch per seed -------------------------- #
            checkpoints = sorted(run_dir.glob("checkpoints/u2r/seed*/epoch*.pt"))
            self.assertEqual(len(checkpoints), 3 * PLAN.epochs)
            restored, metadata = QWGANTrainer.load_checkpoint(checkpoints[-1])
            self.assertEqual(metadata["attack_class"], "u2r")
            self.assertEqual(metadata["requirement"], "FR-3")
            # The checkpointed config carries the seed of *this* run, not the
            # template's, so a restored generator is unambiguously attributable.
            self.assertEqual(restored.config.seed, metadata["seed"])
            self.assertEqual(restored.config.n_qubits, CONFIG.n_qubits)
            self.assertEqual(restored.config.n_layers, CONFIG.n_layers)

            # -- manifest: FR-8 lineage --------------------------------------- #
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            for key in (
                "run_id",
                "requirement",
                "started_utc",
                "completed_utc",
                "dataset_id",
                "dataset_version",
                "schema_version",
                "source_sha256",
                "parent_contract",
                "code_commit",
                "environment",
                "config",
                "plan",
                "seeds",
                "partition",
                "device",
                "n_qubits",
                "n_layers",
                "shots",
                "results",
                "output_sha256",
            ):
                self.assertIn(key, manifest)

            self.assertEqual(manifest["requirement"], "FR-3")
            self.assertEqual(manifest["partition"], "train")
            self.assertEqual(manifest["seeds"], [1, 2, 3])
            self.assertEqual(manifest["source_sha256"], contract.source_sha256)
            self.assertEqual(
                manifest["parent_contract"]["contract_id"], "nslkdd-train-only"
            )
            self.assertIsNone(manifest["shots"])  # analytic simulator

            # Every claimed output hash matches the file actually on disk.
            self.assertIn("diagnostics.jsonl", manifest["output_sha256"])
            for name, digest in manifest["output_sha256"].items():
                self.assertTrue((run_dir / name).exists(), name)
                self.assertEqual(len(digest), 64, name)

            # -- results: one entry per class/seed, trained on train rows only - #
            self.assertEqual(len(manifest["results"]), 3)
            expected_rows = contract.class_counts("train")["u2r"]
            for entry in manifest["results"]:
                self.assertEqual(entry["attack_class"], "u2r")
                self.assertEqual(entry["training_rows"], expected_rows)
                self.assertEqual(entry["epochs_completed"], PLAN.epochs)
                self.assertIn("barren_plateau_suspected", entry)
                self.assertIn("mode_collapse_suspected", entry)
                self.assertIn("final_wasserstein_estimate", entry)
                self.assertTrue(np.isfinite(entry["final_wasserstein_estimate"]))

    def test_same_seed_replays_and_different_seeds_diverge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = _contract(root)
            plan = TrainingPlan(
                attack_classes=("u2r",),
                seeds=(7,),
                epochs=1,
                batch_size=4,
                critic_hidden_dims=(8,),
            )

            first = run_training(
                contract, config=CONFIG, plan=plan, output_dir=root / "a"
            )
            second = run_training(
                contract, config=CONFIG, plan=plan, output_dir=root / "b"
            )
            other = run_training(
                contract,
                config=CONFIG,
                plan=TrainingPlan(
                    attack_classes=("u2r",),
                    seeds=(8,),
                    epochs=1,
                    batch_size=4,
                    critic_hidden_dims=(8,),
                ),
                output_dir=root / "c",
            )

            def losses(result) -> list[float]:
                return [
                    json.loads(line)["critic_loss"]
                    for line in (result.root / "diagnostics.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip()
                ]

            self.assertEqual(losses(first), losses(second))
            self.assertNotEqual(losses(first), losses(other))

    def test_batch_size_is_clamped_to_a_rare_class_and_never_pads(self) -> None:
        """A class with fewer rows than ``batch_size`` must still train.

        NSL-KDD u2r has 42 training rows against a design batch size of 64.
        Silently padding or repeating rows would fabricate density.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = _contract(root)
            available = contract.class_counts("train")["u2r"]

            result = run_training(
                contract,
                config=CONFIG,
                plan=TrainingPlan(
                    attack_classes=("u2r",),
                    seeds=(1,),
                    epochs=1,
                    batch_size=available * 10,
                    critic_hidden_dims=(8,),
                ),
                output_dir=root / "runs",
            )

            entry = result.manifest["results"][0]
            self.assertEqual(entry["effective_batch_size"], available)
            self.assertEqual(entry["steps_per_epoch"], 1)
            self.assertEqual(entry["training_rows"], available)

    def test_refuses_a_class_the_contract_does_not_carry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = _contract(root)
            with self.assertRaises(KeyError):
                run_training(
                    contract,
                    config=CONFIG,
                    plan=TrainingPlan(
                        attack_classes=("heartbleed",),
                        seeds=(1,),
                        epochs=1,
                        batch_size=4,
                        critic_hidden_dims=(8,),
                    ),
                    output_dir=root / "runs",
                )


if __name__ == "__main__":
    unittest.main()
