"""FR-4 command-line entry point.

Resolves a versioned experiment file into a contract spec, a synthesis plan and
the gate's thresholds, then runs synthesis. It holds no fidelity logic of its
own -- the thresholds a claim rests on must come from a file under version
control, never from a flag typed at a prompt.

    python -m cqai.synthesis.cli --experiment configs/fr4_nslkdd.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from ..data.nslkdd import ContractSpec, build_train_contract, load_train_contract
from ..fidelity import FidelityThresholds
from .runner import SynthesisPlan, requested_volume, run_synthesis

DEFAULT_TRAIN_FILE = Path("datasets/KDDTrain+.txt")
DEFAULT_CONTRACT_DIR = Path("artifacts/contracts/nslkdd-train-only-v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/synthesis")


def load_experiment(
    path: str | Path,
) -> tuple[ContractSpec, SynthesisPlan, FidelityThresholds]:
    """Resolve a YAML experiment file into the three library inputs."""

    document: dict[str, Any] = yaml.safe_load(
        Path(path).read_text(encoding="utf-8")
    )
    if not isinstance(document, dict):
        raise ValueError(f"{path} is not a mapping")

    spec = ContractSpec(**document.get("contract", {}))
    thresholds = FidelityThresholds(**document.get("thresholds", {}))

    synthesis = dict(document.get("synthesis", {}))
    for key in ("attack_classes", "seeds", "ratios"):
        if key in synthesis:
            synthesis[key] = tuple(synthesis[key])
    plan = SynthesisPlan(**synthesis)

    missing = [
        name for name in plan.attack_classes if name not in plan.checkpoints
    ]
    if missing:
        raise ValueError(
            f"no checkpoint configured for {missing}; every class in "
            "attack_classes needs a checkpoint template containing '{seed}'"
        )
    for name, template in plan.checkpoints.items():
        if "{seed}" not in template:
            raise ValueError(
                f"checkpoint template for {name!r} has no '{{seed}}' placeholder; "
                "one generator per seed is an FR-6 requirement, and a fixed path "
                "would silently gate the same model three times"
            )
    return spec, plan, thresholds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cqai.synthesis.cli",
        description="FR-4 synthesis with a class-conditional fidelity gate",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        type=Path,
        help="Versioned YAML experiment config (see configs/fr4_nslkdd.yaml).",
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
        help="Train-only contract; must be the one the generators were trained on.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Parent directory for synthesis run artefacts.",
    )
    parser.add_argument("--run-id", default=None, help="Explicit run ID.")
    parser.add_argument(
        "--rebuild-contract",
        action="store_true",
        help="Rebuild the contract even if one is already present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the plan and report requested volumes without generating.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec, plan, thresholds = load_experiment(args.experiment)

    contract_manifest = Path(args.contract_dir) / "contract.json"
    if contract_manifest.is_file() and not args.rebuild_contract:
        contract = load_train_contract(args.contract_dir)
        if contract.spec != spec:
            raise ValueError(
                f"the contract at {args.contract_dir} was built with a "
                f"different spec ({contract.spec}); the generators were trained "
                "against one contract and must be gated against the same one"
            )
        print(f"[contract] reusing {args.contract_dir}")
    else:
        print(f"[contract] building from {args.train_file}")
        contract = build_train_contract(
            args.train_file, args.contract_dir, spec=spec
        )

    for attack_class in plan.attack_classes:
        for ratio in plan.ratios:
            count, real, majority = requested_volume(
                contract,
                attack_class,
                ratio=ratio,
                majority_class=plan.majority_class,
            )
            print(
                f"[plan] {attack_class} ratio {ratio}: {count} synthetic "
                f"(real train {real}, majority {plan.majority_class} {majority}) "
                f"x {len(plan.seeds)} seeds"
            )
        print(
            f"[plan] {attack_class}: gate reference "
            f"{len(contract.validation_angles(attack_class))} held-out real rows"
        )

    if args.dry_run:
        print("[dry-run] nothing generated")
        return 0

    result = run_synthesis(
        contract,
        plan=plan,
        thresholds=thresholds,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print(f"[run] {result.run_id} -> {result.root}")
    for entry in result.manifest["results"]:
        if entry["verdict"] == "skipped":
            print(
                f"  {entry['attack_class']} seed={entry['seed']} "
                f"ratio={entry['ratio']} skipped (already at target)"
            )
            continue
        auc = entry.get("c2st_auc")
        fields = [
            f"{entry['attack_class']} seed={entry['seed']}",
            f"ratio={entry['ratio']}",
            f"n={entry['generated']}",
            f"C2ST={auc:.4f}" if auc is not None else "C2ST=n/a",
            f"domain={entry['domain_validity']:.3f}",
            f"cov={entry['coverage']:.3f}",
            f"-> {entry['verdict'].upper()}",
        ]
        if entry["reasons"]:
            fields.append(f"[{', '.join(entry['reasons'])}]")
        print("  " + " ".join(fields))

    print(
        f"[gate] accepted {result.manifest['accepted_batches']} batches, "
        f"quarantined {result.manifest['quarantined_batches']}"
    )
    print(json.dumps({"run_id": result.run_id, "root": str(result.root)}))
    # Nothing released is a legitimate outcome, but it must not look like a
    # successful campaign to whatever calls this.
    return 0 if result.accepted else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
