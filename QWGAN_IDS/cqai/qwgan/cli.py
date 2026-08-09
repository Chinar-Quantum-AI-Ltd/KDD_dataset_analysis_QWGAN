"""FR-3 command-line entry point.

Loads a resolved, versioned experiment config, builds or reuses the train-only
contract, validates the cross-team invariants, invokes the library, and writes
a run manifest. It deliberately holds no model logic of its own.

    python -m cqai.qwgan.cli --experiment configs/fr3_nslkdd.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from ..data.nslkdd import ContractSpec, build_train_contract, load_train_contract
from .config import QWGANConfig
from .report import write_stability_report
from .runner import TrainingPlan, run_training

DEFAULT_TRAIN_FILE = Path("datasets/KDDTrain+.txt")
DEFAULT_CONTRACT_DIR = Path("artifacts/contracts/nslkdd-train-only-v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/runs")

#: Model values the TDD fixes. A config may still change any of them, but only
#: by naming the change under ``deviations:`` with a reason. ``AGENTS.md``
#: requires deviations to be recorded and reported rather than applied quietly,
#: and enforcing it in the loader means an unreviewed run cannot drift.
TDD_DEFAULTS: dict[str, Any] = {
    "n_layers": 4,
    "lambda_gp": 10.0,
    "n_critic": 5,
    "learning_rate": 1e-4,
    "beta1": 0.0,
    "beta2": 0.9,
    "latent_scale": 1.0,
}


def _check_deviations(config: QWGANConfig, declared: dict[str, Any]) -> None:
    """Refuse a config that changes a TDD value without declaring it."""

    actual = {name: getattr(config, name) for name in TDD_DEFAULTS}
    deviating = {
        name for name, value in actual.items() if value != TDD_DEFAULTS[name]
    }
    justified = {
        name
        for name, reason in declared.items()
        if isinstance(reason, str) and reason.strip()
    }

    undeclared = sorted(deviating - justified)
    if undeclared:
        raise ValueError(
            "these values deviate from the TDD without a declaration: "
            + ", ".join(
                f"{name} ({actual[name]!r} != {TDD_DEFAULTS[name]!r})"
                for name in undeclared
            )
            + ". Add a `deviations:` entry naming each one with the reason; "
            "a reported number has to carry the reason it was produced that way."
        )

    unknown = sorted(set(declared) - set(TDD_DEFAULTS))
    if unknown:
        raise ValueError(
            f"`deviations` names values that are not TDD-fixed: {unknown}"
        )

    # A declaration for a value that matches the default would sit there
    # unread and quietly license a future change to it.
    stale = sorted(set(declared) - deviating - set(unknown))
    if stale:
        raise ValueError(
            f"`deviations` declares {stale}, which does not deviate from the "
            "TDD; remove the stale entry rather than leaving a standing "
            "permission slip"
        )


def load_experiment(
    path: str | Path,
) -> tuple[ContractSpec, QWGANConfig, TrainingPlan]:
    """Resolve a YAML experiment file into the three library inputs."""

    document: dict[str, Any] = yaml.safe_load(
        Path(path).read_text(encoding="utf-8")
    )
    if not isinstance(document, dict):
        raise ValueError(f"{path} is not a mapping")

    spec = ContractSpec(**document.get("contract", {}))
    config = QWGANConfig(**document.get("model", {}))

    training = dict(document.get("training", {}))
    for key in ("attack_classes", "seeds", "critic_hidden_dims"):
        if key in training:
            training[key] = tuple(training[key])
    plan = TrainingPlan(**training)

    # The generator's qubit count and the contract's latent width are one
    # number in the design. Catching a mismatch here beats discovering it after
    # a contract build.
    if spec.n_qubits != config.n_qubits:
        raise ValueError(
            f"model.n_qubits ({config.n_qubits}) must equal contract.n_qubits "
            f"({spec.n_qubits}); the latent width is the qubit count"
        )

    _check_deviations(config, dict(document.get("deviations", {}) or {}))
    return spec, config, plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cqai.qwgan.cli",
        description="FR-3 per-attack-class hybrid QWGAN-GP training",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        type=Path,
        help="Versioned YAML experiment config (see configs/fr3_nslkdd.yaml).",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=DEFAULT_TRAIN_FILE,
        help="NSL-KDD training file. The test file is never read.",
    )
    parser.add_argument(
        "--contract-dir",
        type=Path,
        default=DEFAULT_CONTRACT_DIR,
        help="Where the train-only contract lives; reused when already built.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Parent directory for run artefacts.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Explicit run ID. Defaults to a timestamped unique ID.",
    )
    parser.add_argument(
        "--rebuild-contract",
        action="store_true",
        help="Rebuild the contract even if one is already present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config and contract, report the plan, and train nothing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec, config, plan = load_experiment(args.experiment)

    contract_manifest = Path(args.contract_dir) / "contract.json"
    if contract_manifest.is_file() and not args.rebuild_contract:
        contract = load_train_contract(args.contract_dir)
        if contract.spec != spec:
            raise ValueError(
                f"the contract at {args.contract_dir} was built with a "
                f"different spec ({contract.spec}); pass --rebuild-contract or "
                "point --contract-dir at a new directory rather than mixing "
                "two contracts in one lineage"
            )
        print(f"[contract] reusing {args.contract_dir}")
    else:
        print(f"[contract] building from {args.train_file}")
        contract = build_train_contract(args.train_file, args.contract_dir, spec=spec)

    counts = contract.class_counts("train")
    print(f"[contract] train rows per class: {counts}")
    for attack_class in plan.attack_classes:
        if attack_class not in counts:
            raise KeyError(
                f"contract does not carry {attack_class!r}; available: "
                f"{sorted(counts)}"
            )
        print(
            f"[plan] {attack_class}: {counts[attack_class]} train rows, "
            f"batch {min(plan.batch_size, counts[attack_class])}, "
            f"{plan.epochs} epochs x {len(plan.seeds)} seeds"
        )

    if args.dry_run:
        print("[dry-run] nothing trained")
        return 0

    result = run_training(
        contract,
        config=config,
        plan=plan,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print(f"[run] {result.run_id} -> {result.root}")
    for entry in result.manifest["results"]:
        flags = []
        if entry["barren_plateau_suspected"]:
            flags.append("BARREN-PLATEAU")
        if entry["mode_collapse_suspected"]:
            flags.append("MODE-COLLAPSE")
        print(
            f"  {entry['attack_class']} seed={entry['seed']} "
            f"W={entry['final_wasserstein_estimate']:+.4f} "
            f"div={entry['median_diversity_ratio']:.3f} "
            f"{' '.join(flags)}".rstrip()
        )
    # The cross-seed verdict is derived from the finished manifest, so it is
    # written after the run and hashes the manifest rather than being hashed
    # into it. Regenerating the report cannot alter the run's own evidence.
    stability_path = write_stability_report(result.root)
    stability = json.loads(stability_path.read_text(encoding="utf-8"))
    print(f"[stability] {stability_path}")
    for name, entry in sorted(stability["classes"].items()):
        verdict = "stable" if entry["stable"] else ", ".join(entry["reasons"])
        print(f"[stability] {name}: {verdict}")

    print(json.dumps({"run_id": result.run_id, "root": str(result.root)}))
    # A non-zero exit on an unstable run keeps an unusable result from being
    # mistaken for a successful campaign by whatever calls this.
    return 0 if stability["stable"] else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
