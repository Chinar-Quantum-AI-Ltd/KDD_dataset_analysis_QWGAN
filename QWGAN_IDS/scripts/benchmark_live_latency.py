#!/usr/bin/env python3
"""Run the audit-reproduction benchmark against registered classical artifacts.

Example:
    python scripts/benchmark_live_latency.py \
      --transform-bundle artifacts/serving/transform_bundle.joblib \
      --transform-sha <trusted_sha256> \
      --classifier artifacts/serving/classical_classifier.joblib \
      --classifier-sha <trusted_sha256> \
      --fixture artifacts/serving/benchmark_flows.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cqai.serving import ClassicalServingPipeline, benchmark_batch_latency
from src.transform_bundle import TransformBundle


CLASSIFIER_TYPES = ("random_forest", "xgboost")


def load_classifier(
    kind: str,
    path: Path,
    expected_sha256: str,
    fitting_versions: dict[str, str],
):
    if kind == "random_forest":
        from cqai.classifiers import RFClassifier

        return RFClassifier.load(
            path,
            expected_sha256=expected_sha256,
            fitting_versions=fitting_versions,
        )
    if kind == "xgboost":
        from cqai.classifiers import XGBClassifierWrapper

        return XGBClassifierWrapper.load(
            path,
            expected_sha256=expected_sha256,
            fitting_versions=fitting_versions,
        )
    raise ValueError(f"Unsupported classical classifier type: {kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark registered transform + classical classifier request latency"
    )
    parser.add_argument("--transform-bundle", type=Path, required=True)
    parser.add_argument("--transform-sha", required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--classifier-sha", required=True)
    parser.add_argument(
        "--fitting-versions-json",
        type=Path,
        required=True,
        help="Trusted JSON object containing scikit_learn/numpy/joblib fitting versions",
    )
    parser.add_argument("--classifier-type", choices=CLASSIFIER_TYPES, default="random_forest")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16, 64, 256, 1024])
    parser.add_argument("--warmup-runs", type=int, default=20)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--sla-ms", type=float, default=50.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/fr7")
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    args = parse_args()

    # Setup only: loading and fixture I/O are intentionally outside timing.
    fitting_versions = json.loads(
        args.fitting_versions_json.read_text(encoding="utf-8")
    )
    transform = TransformBundle.load(
        args.transform_bundle,
        expected_sha256=args.transform_sha,
        fitting_versions=fitting_versions,
    )
    classifier = load_classifier(
        args.classifier_type,
        args.classifier,
        args.classifier_sha,
        fitting_versions,
    )
    fixture = pd.read_csv(args.fixture)
    pipeline = ClassicalServingPipeline(
        classifier,
        transform,
        max_batch_size=max(args.batch_sizes),
    )

    report = benchmark_batch_latency(
        pipeline,
        fixture,
        batch_sizes=args.batch_sizes,
        warmup_runs=args.warmup_runs,
        measured_runs=args.runs,
        max_p99_ms=args.sla_ms,
        raise_on_sla=False,
    )
    environment = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "versions": {
            name: package_version(name)
            for name in ("numpy", "pandas", "scikit-learn", "joblib", "xgboost")
        },
        "classifier_type": args.classifier_type,
        "classifier_path": str(args.classifier),
        "classifier_sha256": args.classifier_sha,
        "transform_bundle_path": str(args.transform_bundle),
        "transform_bundle_sha256": args.transform_sha,
        "fixture": str(args.fixture),
        "fixture_rows": len(fixture),
        "fixture_columns": list(fixture.columns),
        "live_path_exclusions": [
            "artifact loading",
            "dataset loading",
            "training",
            "synthetic generation",
            "WGAN",
            "QWGAN",
            "quantum circuit",
            "quantum kernel",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_latency.csv"
    summary_path = args.output_dir / "latency_benchmark.csv"
    json_path = args.output_dir / "latency_summary.json"

    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ("batch_size", "run_id", "request_latency_ms", "amortized_ms_per_flow")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["raw_measurements"])

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        fields = tuple(report["summary"][0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["summary"])

    document = {key: value for key, value in report.items() if key != "raw_measurements"}
    document["environment"] = environment
    json_path.write_text(
        json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["overall"], indent=2))

    # Results are written first so a CI failure remains independently inspectable.
    if not report["overall"]["sla_pass"]:
        raise AssertionError(
            "Latency SLA violated: worst request p99="
            f"{report['overall']['worst_request_p99_ms']:.3f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
