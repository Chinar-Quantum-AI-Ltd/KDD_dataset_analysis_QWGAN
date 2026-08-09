"""FR-4 synthesis runner: generate, decode, gate, quarantine, record.

One pass per (attack class, generator seed, target ratio). Every batch is
written to disk exactly once, under ``accepted/`` only when the gate returns
``pass``. ``fail`` and ``insufficient_evidence`` both route to ``quarantine/``
-- the verdict is preserved so a report can distinguish "measured and bad" from
"could not be measured", but neither releases samples.

Nothing here trains anything. The held-out validation partition is read for one
purpose only: as the gate's real reference.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..fidelity import FidelityThresholds, evaluate_gate, released
from .generate import DEFAULT_CHUNK_SIZE, generate_samples, sha256_file

if TYPE_CHECKING:  # pragma: no cover - import cycle guard only
    from ..data.nslkdd import TrainContract

REQUIREMENT = "FR-4"
MANIFEST_VERSION = "1.0.0"

#: Transform artefacts whose versions decide what a decoded row means.
DECODE_ARTIFACTS = (
    "minmax_scaler.joblib",
    "pca.joblib",
    "robust_scaler.joblib",
    "encoder.joblib",
)


@dataclass(frozen=True, slots=True)
class SynthesisPlan:
    """Which generators to sample, how much, and at which target ratios."""

    attack_classes: tuple[str, ...]
    #: class -> checkpoint path template containing ``{seed}``.
    checkpoints: dict[str, str]
    seeds: tuple[int, ...] = (13, 42, 1337)
    #: Fractions of the majority class. The TDD sweeps 0.2-0.4 and warns
    #: against naive 1:1 balancing, which over-synthesises and costs precision.
    ratios: tuple[float, ...] = (0.2, 0.3, 0.4)
    majority_class: str = "normal"
    chunk_size: int = DEFAULT_CHUNK_SIZE
    #: Offset added to the generator seed to derive the synthesis seed, so the
    #: samples are not drawn from the same stream that trained the model.
    synthesis_seed_offset: int = 1_000_000

    def __post_init__(self) -> None:
        if not self.attack_classes:
            raise ValueError("attack_classes must not be empty")
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        if not self.ratios:
            raise ValueError("ratios must not be empty")
        if any(ratio <= 0 for ratio in self.ratios):
            raise ValueError("ratios must be positive")


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    run_id: str
    root: Path
    manifest: dict[str, Any] = field(repr=False)
    accepted: list[dict[str, Any]] = field(default_factory=list, repr=False)
    quarantined: list[dict[str, Any]] = field(default_factory=list, repr=False)


# --------------------------------------------------------------------------- #
# Lineage helpers (mirrors cqai.qwgan.runner so both manifests read alike)
# --------------------------------------------------------------------------- #
def _code_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    return result.stdout.strip() or None


def _environment() -> dict[str, Any]:
    def version(name: str) -> str | None:
        try:
            from importlib.metadata import version as _version

            return _version(name)
        except Exception:  # noqa: BLE001 - absent package is not an error here
            return None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ("pennylane", "torch", "numpy", "scikit-learn", "scipy")
        },
    }


def _parent_run_id(checkpoint: str) -> str | None:
    """The FR-3 run a checkpoint came from, taken from its path.

    Run directories are immutable and named by run ID, so the path is the
    lineage. A checkpoint stored elsewhere simply records ``None`` rather than
    inventing a parent.
    """

    parts = Path(checkpoint).parts
    if "checkpoints" in parts:
        index = parts.index("checkpoints")
        if index > 0:
            return parts[index - 1]
    return None


def requested_volume(
    contract: "TrainContract", attack_class: str, *, ratio: float, majority_class: str
) -> tuple[int, int, int]:
    """How many synthetic rows a target ratio implies.

    Returns ``(requested, real_train_rows, majority_rows)``. The target is a
    fraction of the majority class minus what is already real, floored at zero:
    a class already above its target is not synthesised for at all.
    """

    counts = contract.class_counts("train")
    if attack_class not in counts:
        raise KeyError(
            f"contract does not carry {attack_class!r}; available: {sorted(counts)}"
        )
    if majority_class not in counts:
        raise KeyError(
            f"contract does not carry majority class {majority_class!r}"
        )
    real = int(counts[attack_class])
    majority = int(counts[majority_class])
    return max(0, round(ratio * majority) - real), real, majority


def _write_batch(
    directory: Path, batch, gate: dict[str, Any]
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    angles_path = directory / "angles.npy"
    decoded_path = directory / "decoded.npy"
    np.save(angles_path, batch.angles)
    np.save(decoded_path, np.asarray(batch.decoded, dtype=np.float64))
    (directory / "gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "angles_path": str(angles_path),
        "decoded_path": str(decoded_path),
        "gate_path": str(directory / "gate.json"),
    }


def run_synthesis(
    contract: "TrainContract",
    *,
    plan: SynthesisPlan,
    thresholds: FidelityThresholds | None = None,
    output_dir: str | Path,
    run_id: str | None = None,
) -> SynthesisResult:
    """Generate, gate, and route one batch per (class, seed, ratio)."""

    thresholds = thresholds or FidelityThresholds()
    counts = contract.class_counts("train")

    unknown = [name for name in plan.attack_classes if name not in counts]
    if unknown:
        raise KeyError(
            f"contract does not carry {unknown}; available: {sorted(counts)}"
        )
    missing = [name for name in plan.attack_classes if name not in plan.checkpoints]
    if missing:
        raise KeyError(f"no checkpoint configured for {missing}")

    run_id = run_id or (
        f"fr4-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    )
    run_dir = Path(output_dir) / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    started = datetime.now(timezone.utc)
    decode_artifacts = {
        name: contract.manifest["artifact_sha256"][name]
        for name in DECODE_ARTIFACTS
        if name in contract.manifest["artifact_sha256"]
    }

    results: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    parent_run_ids: set[str] = set()

    for attack_class in plan.attack_classes:
        real_reference = pd.DataFrame(
            contract.decode_angles(contract.validation_angles(attack_class))
        )
        # The rows the generators were fitted on, for the novelty check. A model
        # that echoes its training set passes a two-sample test trivially and
        # contributes nothing to an augmented one.
        train_rows = contract.angles("train")[
            contract.families("train") == attack_class
        ]
        train_reference = pd.DataFrame(contract.decode_angles(train_rows))
        for seed in plan.seeds:
            checkpoint = plan.checkpoints[attack_class].format(seed=seed)
            parent = _parent_run_id(checkpoint)
            if parent:
                parent_run_ids.add(parent)

            for ratio in plan.ratios:
                count, real_rows, majority_rows = requested_volume(
                    contract,
                    attack_class,
                    ratio=ratio,
                    majority_class=plan.majority_class,
                )
                entry: dict[str, Any] = {
                    "attack_class": attack_class,
                    "seed": seed,
                    "ratio": ratio,
                    "requested": count,
                    "generated": 0,
                    "real_train_rows": real_rows,
                    "real_reference_rows": int(len(real_reference)),
                    "majority_class": plan.majority_class,
                    "majority_train_rows": majority_rows,
                    "checkpoint": checkpoint,
                    "parent_run_id": parent,
                }

                if count == 0:
                    # Already at or above the target share: synthesising here
                    # would push the class past the ratio the sweep is testing.
                    entry.update(verdict="skipped", reasons=[])
                    results.append(entry)
                    continue

                synthesis_seed = seed + plan.synthesis_seed_offset
                batch = generate_samples(
                    checkpoint,
                    contract=contract,
                    count=count,
                    seed=synthesis_seed,
                    chunk_size=plan.chunk_size,
                )
                gate = evaluate_gate(
                    real_reference,
                    batch.decoded,
                    train_reference=train_reference,
                    thresholds=thresholds,
                    seed=synthesis_seed,
                )

                pool = "accepted" if released(gate) else "quarantine"
                paths = _write_batch(
                    run_dir
                    / pool
                    / attack_class
                    / f"seed{seed}"
                    / f"ratio{ratio}",
                    batch,
                    gate,
                )

                record = {
                    **entry,
                    "generated": len(batch),
                    "synthesis_seed": synthesis_seed,
                    "count": len(batch),
                    "checkpoint_sha256": batch.checkpoint_sha256,
                    "checkpoint_epoch": batch.epoch,
                    "columns": list(batch.decoded.columns),
                    "thresholds": asdict(thresholds),
                    "gate": gate,
                    "decode_artifacts": decode_artifacts,
                    "pool": pool,
                    **paths,
                }
                (accepted if pool == "accepted" else quarantined).append(record)

                entry.update(
                    generated=len(batch),
                    verdict=gate["verdict"],
                    reasons=list(gate["reasons"]),
                    pool=pool,
                    c2st_auc=(
                        gate["metrics"]["c2st"]["mean"]
                        if gate["metrics"]["c2st"]
                        else None
                    ),
                    domain_validity=gate["metrics"]["domain"]["valid_fraction"],
                    coverage=gate["metrics"]["coverage"],
                )
                results.append(entry)

    for name, entries in (
        ("accepted_manifest.json", accepted),
        ("quarantine_manifest.json", quarantined),
    ):
        (run_dir / name).write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "requirement": REQUIREMENT,
                    "pool": name.split("_")[0],
                    "entries": entries,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "run_id": run_id,
        "requirement": REQUIREMENT,
        "experiment_type": "per-class synthesis with fidelity gating",
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": contract.manifest["dataset_id"],
        "dataset_version": contract.manifest["contract_version"],
        "schema_version": contract.manifest["schema_version"],
        "source_file": contract.manifest["source_file"],
        "source_sha256": contract.source_sha256,
        "parent_contract": {
            "contract_id": contract.manifest["contract_id"],
            "contract_version": contract.manifest["contract_version"],
            "root": str(contract.root),
            "spec": contract.manifest["spec"],
            "selected_features": contract.manifest["selected_features"],
        },
        "parent_run_ids": sorted(parent_run_ids),
        "code_commit": _code_commit(),
        "environment": _environment(),
        "plan": asdict(plan),
        "thresholds": asdict(thresholds),
        #: Samples are generated from train-fitted generators; the val partition
        #: is read only as the gate's real reference and never written out.
        "partition": "train",
        "gate_reference_partition": "val",
        "decode_artifacts": decode_artifacts,
        "accepted_batches": len(accepted),
        "quarantined_batches": len(quarantined),
        "runtime_seconds": (
            datetime.now(timezone.utc) - started
        ).total_seconds(),
        "results": results,
    }

    outputs = sorted(path for path in run_dir.rglob("*") if path.is_file())
    manifest["output_sha256"] = {
        str(path.relative_to(run_dir)): sha256_file(path) for path in outputs
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    return SynthesisResult(
        run_id=run_id,
        root=run_dir,
        manifest=manifest,
        accepted=accepted,
        quarantined=quarantined,
    )
