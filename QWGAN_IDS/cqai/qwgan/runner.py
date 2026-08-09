"""FR-3 per-attack-class training runner with FR-8 lineage.

One generator per rare attack class is the FR-3 baseline; class conditioning is
a later optimization. This module turns the tested ``QWGANTrainer`` core into a
reproducible experiment: mini-batch epochs over a leakage-safe train-only
contract, per-epoch checkpoints, JSONL diagnostics, barren-plateau and
mode-collapse monitors, and a hashed run manifest.

Nothing here reads the validation or test partitions. Synthesis and fidelity
gating are FR-4 and deliberately live elsewhere.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .config import QWGANConfig
from .data_contract import TrainingAngles
from .trainer import QWGANTrainer

if TYPE_CHECKING:  # pragma: no cover - import cycle guard only
    from ..data.nslkdd import TrainContract

REQUIREMENT = "FR-3"
MANIFEST_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    """The experiment shape: which classes, which seeds, how long."""

    attack_classes: tuple[str, ...]
    #: FR-6 requires exactly three centrally configured seeds. A shorter tuple
    #: is allowed here so tests and smoke runs stay cheap, but a real ablation
    #: must pass three.
    seeds: tuple[int, ...] = (13, 42, 1337)
    epochs: int = 20
    batch_size: int = 64
    max_steps_per_epoch: int | None = None
    checkpoint_every: int = 1
    critic_hidden_dims: tuple[int, ...] = (128, 64)
    barren_plateau_variance: float = 1e-12
    mode_collapse_ratio: float = 0.1

    #: Stop once the training-side Wasserstein estimate has not reached a new
    #: best for this many consecutive checks. ``None`` runs the full schedule.
    #:
    #: The signal is deliberately the *training* estimate. Stopping on held-out
    #: performance would be ordinary model selection, but those are the same
    #: rows the FR-4 gate scores against, and selecting on them would turn the
    #: gate's verdict into a measure of its own selection.
    early_stop_patience: int | None = None
    early_stop_check_every: int = 10
    #: WGAN dynamics are noisy early on; a dip in epoch two means nothing.
    early_stop_min_epochs: int = 50
    #: Relative improvement required to count as a new best.
    early_stop_tolerance: float = 1e-3

    def __post_init__(self) -> None:
        if not self.attack_classes:
            raise ValueError("attack_classes must not be empty")
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")
        if self.early_stop_patience is not None and self.early_stop_patience < 1:
            raise ValueError("early_stop_patience must be at least 1 when set")
        if self.early_stop_check_every < 1:
            raise ValueError("early_stop_check_every must be at least 1")
        if self.early_stop_min_epochs < 1:
            raise ValueError("early_stop_min_epochs must be at least 1")


@dataclass(frozen=True, slots=True)
class RunResult:
    """Where a finished run lives and what it recorded."""

    run_id: str
    root: Path
    manifest: dict[str, Any] = field(repr=False)


# --------------------------------------------------------------------------- #
# Monitors — pure, so they can be tested without inducing a real plateau
# --------------------------------------------------------------------------- #
def detect_barren_plateau(
    gradient_variances: Sequence[float], threshold: float
) -> bool:
    """True when the generator's gradient signal has effectively vanished.

    The decision uses the median, not the minimum or maximum: one lucky step
    should not clear the flag, and one dead step should not raise it.
    """

    finite = [value for value in gradient_variances if np.isfinite(value)]
    if not finite:
        return False
    return bool(median(finite) < threshold)


def diversity_ratio(generated: torch.Tensor, real: torch.Tensor) -> float:
    """Generated spread as a fraction of the real batch's spread.

    A ratio near zero means every sample looks the same — the classic WGAN
    mode-collapse signature. It is measured against the real batch so a
    naturally tight attack class is not mistaken for collapse.
    """

    epsilon = 1e-12
    real_spread = float(real.detach().std(dim=0, correction=0).mean())
    generated_spread = float(generated.detach().std(dim=0, correction=0).mean())
    return generated_spread / max(real_spread, epsilon)


# --------------------------------------------------------------------------- #
# Lineage helpers
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_commit() -> str | None:
    """Best-effort source commit; ``None`` when Git metadata is unavailable."""

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
            for name in ("pennylane", "torch", "numpy", "scikit-learn", "pandas")
        },
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _minibatches(
    batch: TrainingAngles, *, batch_size: int, generator: torch.Generator
) -> list[TrainingAngles]:
    """Shuffle once per epoch and cut into non-overlapping mini-batches.

    A trailing partial batch is kept rather than dropped: rare classes cannot
    afford to discard rows, and the critic tolerates a smaller batch.
    """

    order = torch.randperm(batch.values.shape[0], generator=generator)
    chunks: list[TrainingAngles] = []
    for start in range(0, len(order), batch_size):
        rows = batch.values[order[start : start + batch_size]]
        chunks.append(
            TrainingAngles(
                values=rows,
                attack_class=batch.attack_class,
                latent_columns=batch.latent_columns,
            )
        )
    return chunks


def _train_one(
    *,
    contract: "TrainContract",
    config: QWGANConfig,
    plan: TrainingPlan,
    attack_class: str,
    seed: int,
    run_id: str,
    run_dir: Path,
    diagnostics: Any,
) -> dict[str, Any]:
    """Train a single (class, seed) generator and return its result entry."""

    seeded_config = QWGANConfig(**{**asdict(config), "seed": seed})
    batch = contract.training_angles(attack_class, config=seeded_config)
    trainer = QWGANTrainer(
        seeded_config, critic_hidden_dims=plan.critic_hidden_dims
    )

    training_rows = int(batch.values.shape[0])
    effective_batch_size = min(plan.batch_size, training_rows)
    shuffle_rng = torch.Generator().manual_seed(seed)

    checkpoint_dir = run_dir / "checkpoints" / attack_class / f"seed{seed}"
    gradient_variances: list[float] = []
    diversity_ratios: list[float] = []
    last: dict[str, Any] = {}
    steps_per_epoch = 0
    epochs_completed = 0
    stop_reason = "schedule_complete"
    best_wasserstein: float | None = None
    checks_without_improvement = 0

    started = datetime.now(timezone.utc)
    for epoch in range(1, plan.epochs + 1):
        chunks = _minibatches(
            batch, batch_size=effective_batch_size, generator=shuffle_rng
        )
        if plan.max_steps_per_epoch is not None:
            chunks = chunks[: plan.max_steps_per_epoch]
        steps_per_epoch = len(chunks)
        epoch_wasserstein: list[float] = []

        for step_in_epoch, chunk in enumerate(chunks, start=1):
            metrics = trainer.train_step(chunk)

            # Measure diversity on a fresh batch the critic did not just see.
            fake = trainer.generate(chunk.values.shape[0])
            ratio = diversity_ratio(fake, chunk.values)

            gradient_variances.append(
                float(metrics["generator_gradient_variance"])
            )
            diversity_ratios.append(ratio)
            epoch_wasserstein.append(float(metrics["wasserstein_estimate"]))
            last = dict(metrics)

            record = {
                "run_id": run_id,
                "requirement": REQUIREMENT,
                "seed": seed,
                "epoch": epoch,
                "step_in_epoch": step_in_epoch,
                "batch_rows": int(chunk.values.shape[0]),
                "diversity_ratio": ratio,
                **metrics,
            }
            diagnostics.write(json.dumps(record) + "\n")
            diagnostics.flush()

        epochs_completed = epoch
        stop_now = False

        if (
            plan.early_stop_patience is not None
            and epoch >= plan.early_stop_min_epochs
            and epoch % plan.early_stop_check_every == 0
        ):
            # Lower is better: the estimate should shrink as the generator
            # closes on the real distribution.
            current = float(np.mean(epoch_wasserstein))
            improved = best_wasserstein is None or current < best_wasserstein * (
                1.0 - plan.early_stop_tolerance
            )
            if improved:
                best_wasserstein = current
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
                if checks_without_improvement >= plan.early_stop_patience:
                    stop_reason = "wasserstein_plateau"
                    stop_now = True

        # A stopped run must not leave a checkpoint claiming an epoch it never
        # finished, so the final-epoch checkpoint follows the actual last epoch.
        if (
            epoch % plan.checkpoint_every == 0
            or epoch == plan.epochs
            or stop_now
        ):
            trainer.save_checkpoint(
                checkpoint_dir / f"epoch{epoch:04d}.pt",
                metadata={
                    "run_id": run_id,
                    "requirement": REQUIREMENT,
                    "attack_class": attack_class,
                    "seed": seed,
                    "epoch": epoch,
                    "contract_source_sha256": contract.source_sha256,
                },
            )

        if stop_now:
            break

    return {
        "attack_class": attack_class,
        "seed": seed,
        "training_rows": training_rows,
        "effective_batch_size": effective_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "epochs_completed": epochs_completed,
        "epochs_scheduled": plan.epochs,
        "early_stopped": epochs_completed < plan.epochs,
        "stop_reason": stop_reason,
        #: Named so a reader can confirm the decision did not consume the
        #: held-out rows the FR-4 gate scores against.
        "early_stop_signal": "train_wasserstein_estimate",
        "best_epoch_wasserstein_estimate": best_wasserstein,
        "global_steps": int(trainer.global_step),
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "final_critic_loss": float(last["critic_loss"]),
        "final_generator_loss": float(last["generator_loss"]),
        "final_wasserstein_estimate": float(last["wasserstein_estimate"]),
        "median_generator_gradient_variance": float(median(gradient_variances)),
        "median_diversity_ratio": float(median(diversity_ratios)),
        "barren_plateau_suspected": detect_barren_plateau(
            gradient_variances, plan.barren_plateau_variance
        ),
        "mode_collapse_suspected": bool(
            median(diversity_ratios) < plan.mode_collapse_ratio
        ),
        "circuit_depth": int(last["circuit_depth"]),
        "circuit_gate_count": int(last["circuit_gate_count"]),
        "checkpoint_dir": str(checkpoint_dir.relative_to(run_dir)),
    }


def run_training(
    contract: "TrainContract",
    *,
    config: QWGANConfig,
    plan: TrainingPlan,
    output_dir: str | Path,
    run_id: str | None = None,
) -> RunResult:
    """Train one generator per (attack class, seed) and record full lineage.

    Every artefact lands under ``output_dir/<run_id>/`` and is never
    overwritten, so a prior run's evidence stays intact.
    """

    unknown = [
        name for name in plan.attack_classes if name not in contract.attack_classes
    ]
    if unknown:
        raise KeyError(
            f"contract does not carry {unknown}; available classes: "
            f"{contract.attack_classes}"
        )

    run_id = run_id or f"fr3-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    run_dir = Path(output_dir) / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    started = datetime.now(timezone.utc)
    results: list[dict[str, Any]] = []
    with open(run_dir / "diagnostics.jsonl", "w", encoding="utf-8") as diagnostics:
        for attack_class in plan.attack_classes:
            for seed in plan.seeds:
                results.append(
                    _train_one(
                        contract=contract,
                        config=config,
                        plan=plan,
                        attack_class=attack_class,
                        seed=seed,
                        run_id=run_id,
                        run_dir=run_dir,
                        diagnostics=diagnostics,
                    )
                )

    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "run_id": run_id,
        "requirement": REQUIREMENT,
        "experiment_type": "per-class hybrid QWGAN-GP training",
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
            "train_angles_sha256": contract.manifest["partitions"]["train"][
                "angles_sha256"
            ],
            "selected_features": contract.manifest["selected_features"],
        },
        "partition": "train",
        "latent_columns": list(contract.latent_columns),
        "angle_range": list(contract.angle_range),
        "code_commit": _code_commit(),
        "environment": _environment(),
        "config": asdict(config),
        "plan": asdict(plan),
        "seeds": list(plan.seeds),
        "device": config.backend,
        "diff_method": config.diff_method,
        "n_qubits": config.n_qubits,
        "n_layers": config.n_layers,
        #: ``default.qubit``/``lightning.qubit`` run analytically, so there is
        #: no shot budget to record. A QPU or finite-shot run must set this.
        "shots": None,
        "estimated_cost_usd": 0.0,
        "runtime_seconds": (
            datetime.now(timezone.utc) - started
        ).total_seconds(),
        "results": results,
    }

    outputs = sorted(
        path for path in run_dir.rglob("*") if path.is_file()
    )
    manifest["output_sha256"] = {
        str(path.relative_to(run_dir)): _sha256_file(path) for path in outputs
    }

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return RunResult(run_id=run_id, root=run_dir, manifest=manifest)
