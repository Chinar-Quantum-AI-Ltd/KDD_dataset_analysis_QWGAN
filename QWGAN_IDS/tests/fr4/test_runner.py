from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cqai.data import ContractSpec, build_train_contract
from cqai.fidelity import FidelityThresholds
from cqai.qwgan import QWGANConfig, TrainingPlan, run_training
from cqai.synthesis import SynthesisPlan, run_synthesis

from ..fixtures import write_nslkdd_fixture

SPEC = ContractSpec(n_qubits=8, top_k=12, val_fraction=0.25, split_seed=3)
CONFIG = QWGANConfig(n_qubits=8, n_layers=3, n_critic=1, seed=5)
SEEDS = (13, 42)

#: Forces every batch through the gate, so accept/quarantine routing is tested
#: rather than the generator's (untrained, on a fixture) fidelity.
ALWAYS_PASS = FidelityThresholds(
    max_c2st_auc=1.0,
    max_wasserstein_1=1e9,
    max_ks_statistic=1.0,
    min_coverage=0.0,
    min_domain_validity=0.0,
    min_real_samples=1,
)
ALWAYS_FAIL = FidelityThresholds(
    max_c2st_auc=0.0,
    max_wasserstein_1=0.0,
    max_ks_statistic=0.0,
    min_coverage=1.0,
    min_domain_validity=1.0,
    min_real_samples=1,
)


def _prepared(root: Path):
    """A tiny real FR-3 run whose checkpoints FR-4 then consumes."""

    source = write_nslkdd_fixture(root / "KDDTrain+.txt", rows_per_label=40)
    contract = build_train_contract(source, root / "contract", spec=SPEC)
    training = run_training(
        contract,
        config=CONFIG,
        plan=TrainingPlan(
            attack_classes=("u2r",),
            seeds=SEEDS,
            epochs=1,
            batch_size=4,
            critic_hidden_dims=(8,),
        ),
        output_dir=root / "runs",
    )
    template = str(
        training.root / "checkpoints" / "u2r" / "seed{seed}" / "epoch0001.pt"
    )
    return contract, training, template


def _plan(template: str, **overrides) -> SynthesisPlan:
    arguments = {
        "attack_classes": ("u2r",),
        "seeds": SEEDS,
        "ratios": (0.8,),
        "checkpoints": {"u2r": template},
        "chunk_size": 64,
        # The fixture's majority class is ``dos`` (90 train rows), not
        # ``normal``: unlike real NSL-KDD, the fixture gives ``normal`` a single
        # raw label. Naming it explicitly keeps the volume arithmetic honest.
        "majority_class": "dos",
    }
    arguments.update(overrides)
    return SynthesisPlan(**arguments)


class VolumeTests(unittest.TestCase):
    def test_volume_comes_from_the_ratio_and_the_majority_class(self) -> None:
        """The TDD sweeps a target minority ratio, not a naive 1:1 balance.

        Doubling a 42-row class to 84 is not what the design asks for; the
        target is a fraction of the majority class, minus what is already real.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, _, template = _prepared(root)

            result = run_synthesis(
                contract,
                plan=_plan(template, seeds=(13,), ratios=(0.9,)),
                thresholds=ALWAYS_PASS,
                output_dir=root / "synth",
            )

            counts = contract.class_counts("train")
            expected = round(0.9 * counts["dos"]) - counts["u2r"]
            entry = result.manifest["results"][0]
            self.assertEqual(entry["requested"], expected)
            self.assertEqual(entry["generated"], expected)
            self.assertEqual(entry["real_train_rows"], counts["u2r"])
            self.assertEqual(entry["majority_class"], "dos")

    def test_a_class_already_above_the_target_requests_nothing(self) -> None:
        """A ratio below the class's existing share must not over-synthesise."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, _, template = _prepared(root)

            result = run_synthesis(
                contract,
                plan=_plan(template, seeds=(13,), ratios=(0.01,)),
                thresholds=ALWAYS_PASS,
                output_dir=root / "synth",
            )

            entry = result.manifest["results"][0]
            self.assertEqual(entry["requested"], 0)
            self.assertEqual(entry["verdict"], "skipped")
            self.assertEqual(entry["generated"], 0)


class QuarantineTests(unittest.TestCase):
    def test_a_failing_batch_never_reaches_the_accepted_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, _, template = _prepared(root)

            result = run_synthesis(
                contract,
                plan=_plan(template),
                thresholds=ALWAYS_FAIL,
                output_dir=root / "synth",
            )

            self.assertFalse(result.accepted)
            self.assertEqual(len(result.quarantined), len(SEEDS))
            self.assertFalse(list((result.root / "accepted").rglob("*.npy")))
            self.assertTrue(list((result.root / "quarantine").rglob("*.npy")))
            for entry in result.manifest["results"]:
                self.assertEqual(entry["verdict"], "fail")
                self.assertTrue(entry["reasons"])

    def test_a_passing_batch_lands_only_in_the_accepted_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, _, template = _prepared(root)

            result = run_synthesis(
                contract,
                plan=_plan(template),
                thresholds=ALWAYS_PASS,
                output_dir=root / "synth",
            )

            self.assertEqual(len(result.accepted), len(SEEDS))
            self.assertFalse(result.quarantined)
            self.assertFalse(list((result.root / "quarantine").rglob("*.npy")))

    def test_insufficient_evidence_is_quarantined_like_a_failure(self) -> None:
        """Uncertifiable is not the same as bad, but it is equally unusable.

        The verdict is preserved for reporting; the routing is identical.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, _, template = _prepared(root)

            result = run_synthesis(
                contract,
                plan=_plan(template, seeds=(13,)),
                thresholds=FidelityThresholds(min_real_samples=10_000),
                output_dir=root / "synth",
            )

            entry = result.manifest["results"][0]
            self.assertEqual(entry["verdict"], "insufficient_evidence")
            self.assertIn("insufficient_real_samples", entry["reasons"])
            self.assertFalse(result.accepted)
            self.assertEqual(len(result.quarantined), 1)


class ManifestTests(unittest.TestCase):
    def test_accepted_and_quarantine_manifests_are_written_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, training, template = _prepared(root)

            result = run_synthesis(
                contract,
                plan=_plan(template),
                thresholds=ALWAYS_PASS,
                output_dir=root / "synth",
            )

            accepted = json.loads(
                (result.root / "accepted_manifest.json").read_text(encoding="utf-8")
            )
            quarantine = json.loads(
                (result.root / "quarantine_manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(len(accepted["entries"]), len(SEEDS))
            self.assertEqual(quarantine["entries"], [])
            for entry in accepted["entries"]:
                for key in (
                    "attack_class",
                    "seed",
                    "synthesis_seed",
                    "ratio",
                    "count",
                    "checkpoint",
                    "checkpoint_sha256",
                    "thresholds",
                    "gate",
                    "decode_artifacts",
                    "angles_path",
                    "decoded_path",
                ):
                    self.assertIn(key, entry, key)
                self.assertEqual(len(entry["checkpoint_sha256"]), 64)
                # The decode artefact versions travel with the samples: the same
                # angles decoded by different transforms are different records.
                self.assertIn("pca.joblib", entry["decode_artifacts"])

    def test_the_run_manifest_carries_fr8_lineage_and_verified_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, training, template = _prepared(root)

            result = run_synthesis(
                contract,
                plan=_plan(template),
                thresholds=ALWAYS_PASS,
                output_dir=root / "synth",
            )

            manifest = result.manifest
            for key in (
                "manifest_version",
                "run_id",
                "requirement",
                "started_utc",
                "completed_utc",
                "dataset_id",
                "schema_version",
                "source_sha256",
                "parent_contract",
                "parent_run_ids",
                "code_commit",
                "environment",
                "plan",
                "thresholds",
                "partition",
                "results",
                "output_sha256",
            ):
                self.assertIn(key, manifest)

            self.assertEqual(manifest["requirement"], "FR-4")
            self.assertEqual(manifest["partition"], "train")
            self.assertEqual(manifest["source_sha256"], contract.source_sha256)
            self.assertIn(training.run_id, manifest["parent_run_ids"])

            for name, digest in manifest["output_sha256"].items():
                path = result.root / name
                self.assertTrue(path.exists(), name)
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), digest, name
                )

    def test_synthetic_rows_are_disjoint_from_the_held_out_real_rows(self) -> None:
        """The gate reads val; the output must never contain it.

        Writing real held-out rows into an "accepted synthetic" pool would leak
        evaluation data straight into FR-5's training set.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, _, template = _prepared(root)

            result = run_synthesis(
                contract,
                plan=_plan(template, seeds=(13,)),
                thresholds=ALWAYS_PASS,
                output_dir=root / "synth",
            )

            produced = np.load(result.accepted[0]["angles_path"])
            real = contract.validation_angles("u2r")
            for row in real:
                self.assertFalse(
                    np.any(np.all(np.isclose(produced, row), axis=1)),
                    "a held-out real row appeared in the accepted pool",
                )


class RefusalTests(unittest.TestCase):
    def test_refuses_a_class_the_contract_does_not_carry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, _, template = _prepared(root)
            with self.assertRaises(KeyError):
                run_synthesis(
                    contract,
                    plan=_plan(
                        template,
                        attack_classes=("heartbleed",),
                        checkpoints={"heartbleed": template},
                    ),
                    thresholds=ALWAYS_PASS,
                    output_dir=root / "synth",
                )

    def test_refuses_a_class_without_a_configured_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, _, template = _prepared(root)
            with self.assertRaises(KeyError):
                run_synthesis(
                    contract,
                    plan=_plan(template, attack_classes=("u2r", "r2l")),
                    thresholds=ALWAYS_PASS,
                    output_dir=root / "synth",
                )

    def test_refuses_to_overwrite_an_existing_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, _, template = _prepared(root)
            arguments = {
                "plan": _plan(template, seeds=(13,)),
                "thresholds": ALWAYS_PASS,
                "output_dir": root / "synth",
                "run_id": "collide",
            }
            run_synthesis(contract, **arguments)
            with self.assertRaises(FileExistsError):
                run_synthesis(contract, **arguments)


if __name__ == "__main__":
    unittest.main()
