"""FR-3 cross-seed stability report.

``AGENTS.md`` closes FR-3 only when the trained core is "stable enough to
evaluate across three seeds". A per-(class, seed) manifest states outcomes; it
does not state whether the seeds *agree*. This module turns that judgement into
a derived, hashed artefact so the claim can be checked instead of asserted.

The report is deliberately downstream of the run: it reads an immutable run
directory, never rewrites it, and records the parent run ID plus the SHA-256 of
the manifest it was computed from. Regenerating it cannot change the evidence.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

STABILITY_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class StabilityThresholds:
    """Versioned pass/fail bands for the cross-seed verdict.

    These are reporting thresholds, independent of the monitor thresholds a run
    was configured with. A permissive run configuration therefore cannot launder
    a dead gradient signal or a collapsed generator past the report.
    """

    #: FR-6 mandates three seeds; fewer cannot separate stable from lucky.
    required_seeds: int = 3
    #: A class is flagged only when the seed spread is large *both* relative to
    #: the mean and in absolute terms. Requiring both avoids the
    #: coefficient-of-variation trap, where three seeds converging on ~0 have a
    #: huge relative spread and a negligible real one.
    max_wasserstein_relative_spread: float = 0.5
    min_wasserstein_absolute_spread: float = 0.05
    min_median_gradient_variance: float = 1e-12
    min_median_diversity_ratio: float = 0.1


def _spread(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(fmean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _epoch_means(run_dir: Path) -> dict[str, list[float]]:
    """Mean Wasserstein estimate per epoch per class, averaged over seeds.

    A single end-of-run number cannot distinguish a model that improved from
    one that oscillated, so the trajectory is summarised alongside it.
    """

    path = run_dir / "diagnostics.jsonl"
    if not path.exists():
        return {}

    buckets: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            buckets[record["attack_class"]][int(record["epoch"])].append(
                float(record["wasserstein_estimate"])
            )

    return {
        attack_class: [
            float(fmean(epochs[epoch])) for epoch in sorted(epochs)
        ]
        for attack_class, epochs in buckets.items()
    }


def _verdict(
    entry: dict[str, Any], thresholds: StabilityThresholds
) -> list[str]:
    """Machine-readable reasons a class is not stable; empty means stable."""

    reasons: list[str] = []
    if entry["seed_count"] < thresholds.required_seeds:
        reasons.append("insufficient_seeds")

    wasserstein = entry["final_wasserstein_estimate"]
    relative = wasserstein["std"] / max(abs(wasserstein["mean"]), 1e-12)
    if (
        relative > thresholds.max_wasserstein_relative_spread
        and wasserstein["std"] > thresholds.min_wasserstein_absolute_spread
    ):
        reasons.append("wasserstein_spread")

    if entry["barren_plateau_seeds"]:
        reasons.append("barren_plateau")
    if (
        entry["median_generator_gradient_variance"]["min"]
        < thresholds.min_median_gradient_variance
    ):
        reasons.append("vanishing_gradients")

    if entry["mode_collapse_seeds"]:
        reasons.append("mode_collapse")
    if (
        entry["median_diversity_ratio"]["min"]
        < thresholds.min_median_diversity_ratio
    ):
        reasons.append("low_diversity")

    return reasons


def summarise_run(
    run_dir: str | Path,
    *,
    thresholds: StabilityThresholds | None = None,
) -> dict[str, Any]:
    """Aggregate a finished run's per-seed results into a stability verdict."""

    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    thresholds = thresholds or StabilityThresholds()

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in manifest["results"]:
        by_class[result["attack_class"]].append(result)

    trajectories = _epoch_means(root)
    classes: dict[str, Any] = {}
    for attack_class, results in by_class.items():
        results = sorted(results, key=lambda item: item["seed"])
        trajectory = trajectories.get(attack_class, [])
        entry: dict[str, Any] = {
            "seeds": [int(item["seed"]) for item in results],
            "seed_count": len(results),
            "training_rows": int(results[0]["training_rows"]),
            "epochs_completed": int(results[0]["epochs_completed"]),
            "final_wasserstein_estimate": _spread(
                [float(item["final_wasserstein_estimate"]) for item in results]
            ),
            "median_generator_gradient_variance": _spread(
                [
                    float(item["median_generator_gradient_variance"])
                    for item in results
                ]
            ),
            "median_diversity_ratio": _spread(
                [float(item["median_diversity_ratio"]) for item in results]
            ),
            "epoch_wasserstein_mean": trajectory,
            #: Last epoch minus first, averaged over seeds. Recorded, never
            #: gated on: a still-rising estimate means the critic is outpacing
            #: the generator — a convergence signal, not a disagreement between
            #: seeds, and the two must not be conflated in one verdict.
            "wasserstein_trend": (
                float(trajectory[-1] - trajectory[0]) if trajectory else 0.0
            ),
            "barren_plateau_seeds": [
                int(item["seed"])
                for item in results
                if item["barren_plateau_suspected"]
            ],
            "mode_collapse_seeds": [
                int(item["seed"])
                for item in results
                if item["mode_collapse_suspected"]
            ],
        }
        entry["reasons"] = _verdict(entry, thresholds)
        entry["stable"] = not entry["reasons"]
        classes[attack_class] = entry

    return {
        "stability_version": STABILITY_VERSION,
        "requirement": manifest["requirement"],
        "parent_run_id": manifest["run_id"],
        "parent_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": asdict(thresholds),
        "classes": classes,
        "stable": all(entry["stable"] for entry in classes.values()),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# FR-3 cross-seed stability — `{summary['parent_run_id']}`",
        "",
        f"- Generated: {summary['generated_utc']}",
        f"- Parent manifest SHA-256: `{summary['parent_manifest_sha256']}`",
        f"- Overall verdict: **{'stable' if summary['stable'] else 'not stable'}**",
        "",
        "| Class | Rows | Seeds | W mean ± std | W trend | Grad var (min) | Diversity (min) | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, entry in sorted(summary["classes"].items()):
        wasserstein = entry["final_wasserstein_estimate"]
        verdict = "stable" if entry["stable"] else ", ".join(entry["reasons"])
        lines.append(
            f"| `{name}` | {entry['training_rows']} | {entry['seed_count']} | "
            f"{wasserstein['mean']:.4f} ± {wasserstein['std']:.4f} | "
            f"{entry['wasserstein_trend']:+.4f} | "
            f"{entry['median_generator_gradient_variance']['min']:.3e} | "
            f"{entry['median_diversity_ratio']['min']:.3f} | {verdict} |"
        )

    lines += [
        "",
        "Thresholds are versioned in `stability.json` under `thresholds`. "
        "They are reporting bands, independent of the monitor thresholds the "
        "run itself was configured with.",
        "",
        "`W trend` is the last epoch's mean Wasserstein estimate minus the "
        "first's, averaged over seeds. It is recorded, never gated on: a "
        "still-rising estimate means the critic is outpacing the generator, "
        "which is a convergence signal rather than a disagreement between "
        "seeds. A stable verdict is therefore not a claim of convergence.",
        "",
    ]
    return "\n".join(lines)


def write_stability_report(
    run_dir: str | Path,
    *,
    thresholds: StabilityThresholds | None = None,
) -> Path:
    """Write ``stability.json`` and ``stability.md`` beside a run's manifest."""

    root = Path(run_dir)
    summary = summarise_run(root, thresholds=thresholds)

    destination = root / "stability.json"
    destination.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / "stability.md").write_text(_markdown(summary), encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m cqai.qwgan.report",
        description="Summarise a finished FR-3 run's cross-seed stability.",
    )
    parser.add_argument("run_dir", help="Directory holding manifest.json")
    arguments = parser.parse_args(argv)

    path = write_stability_report(arguments.run_dir)
    summary = json.loads(path.read_text(encoding="utf-8"))
    print(f"[stability] {path}")
    for name, entry in sorted(summary["classes"].items()):
        verdict = "stable" if entry["stable"] else ", ".join(entry["reasons"])
        print(f"[stability] {name}: {verdict}")
    return 0 if summary["stable"] else 1


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
