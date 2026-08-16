"""PERF-01 request-latency benchmark for the classical serving path."""
from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

from cqai.serving.pipeline import ClassicalServingPipeline


AUDIT_BATCH_SIZES = (1, 16, 64, 256, 1024)


def latency_percentiles(values_ms: Sequence[float]) -> dict[str, float]:
    """Return raw, unfiltered request-latency statistics."""

    values = np.asarray(values_ms, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("latency sample must be non-empty and one-dimensional")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("latencies must be finite and non-negative")
    return {
        "min_ms": float(values.min()),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
    }


def _batch(flows: pd.DataFrame, batch_size: int, index: int) -> pd.DataFrame:
    if batch_size > len(flows):
        raise ValueError(
            f"Fixture has {len(flows)} rows but batch_size={batch_size} was requested"
        )
    possible_starts = len(flows) - batch_size + 1
    start = (index * batch_size) % possible_starts
    return flows.iloc[start : start + batch_size]


def benchmark_batch_latency(
    pipeline: ClassicalServingPipeline,
    raw_flows: pd.DataFrame,
    *,
    batch_sizes: Sequence[int] = AUDIT_BATCH_SIZES,
    warmup_runs: int = 20,
    measured_runs: int = 500,
    max_p99_ms: float = 50.0,
    raise_on_sla: bool = True,
) -> dict[str, Any]:
    """Measure request latency for validation+transform+one classifier traversal.

    Each observation is the latency experienced by the whole request and thus
    by every flow waiting for that response.  Amortized milliseconds per flow
    and throughput are reported only as diagnostics and never determine SLA.
    """

    if not isinstance(raw_flows, pd.DataFrame):
        raise TypeError("raw_flows must be a pandas DataFrame")
    sizes = [int(size) for size in batch_sizes]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("batch sizes must be positive")
    if measured_runs < 100:
        raise ValueError("At least 100 measurements are required for p99")
    if warmup_runs < 0:
        raise ValueError("warmup_runs cannot be negative")

    summaries: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for batch_size in sizes:
        for index in range(warmup_runs):
            pipeline.predict_flow(_batch(raw_flows, batch_size, index))

        request_values: list[float] = []
        for run_id in range(measured_runs):
            sample = _batch(raw_flows, batch_size, run_id + warmup_runs)
            started = time.perf_counter_ns()
            result = pipeline.predict_flow(sample)
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            if result["flow_count"] != batch_size:
                raise RuntimeError("Serving path dropped flows during the benchmark")
            request_values.append(elapsed_ms)
            raw.append(
                {
                    "batch_size": batch_size,
                    "run_id": run_id,
                    "request_latency_ms": elapsed_ms,
                    "amortized_ms_per_flow": elapsed_ms / batch_size,
                }
            )

        stats = latency_percentiles(request_values)
        sla_pass = stats["p99_ms"] <= max_p99_ms
        summaries.append(
            {
                "batch_size": batch_size,
                "warmup_runs": warmup_runs,
                "measured_runs": measured_runs,
                "flows_measured": measured_runs * batch_size,
                **stats,
                "mean_amortized_ms_per_flow": stats["mean_ms"] / batch_size,
                "throughput_flows_per_second": (
                    batch_size * 1000.0 / stats["mean_ms"]
                    if stats["mean_ms"] > 0
                    else None
                ),
                "sla_limit_ms": float(max_p99_ms),
                "sla_pass": bool(sla_pass),
            }
        )

    worst = max(summaries, key=lambda row: row["p99_ms"])
    report = {
        "latency_definition": (
            "request time for serving validation + registered transform + one "
            "registered classical classifier predict_proba traversal"
        ),
        "timer": "time.perf_counter_ns",
        "percentiles": "numpy.percentile over all raw observations; no outlier removal",
        "summary": summaries,
        "raw_measurements": raw,
        "overall": {
            "criterion": "all supported batch sizes have request p99 <= SLA",
            "sla_pass": all(row["sla_pass"] for row in summaries),
            "worst_batch_size": worst["batch_size"],
            "worst_request_p99_ms": worst["p99_ms"],
        },
    }
    if raise_on_sla and not report["overall"]["sla_pass"]:
        raise AssertionError(
            f"Latency SLA violated: batch={worst['batch_size']} request "
            f"p99={worst['p99_ms']:.3f} ms > {max_p99_ms:.3f} ms"
        )
    return report


def benchmark_serving_latency(
    pipeline: ClassicalServingPipeline,
    X_sample: pd.DataFrame,
    *,
    n_iterations: int = 100,
    max_p99_ms: float = 50.0,
) -> dict[str, Any]:
    """Backward-compatible single-batch adapter."""

    report = benchmark_batch_latency(
        pipeline,
        X_sample,
        batch_sizes=(len(X_sample),),
        warmup_runs=1,
        measured_runs=n_iterations,
        max_p99_ms=max_p99_ms,
        raise_on_sla=False,
    )
    result = dict(report["summary"][0])
    result["sla_threshold_ms"] = result["sla_limit_ms"]
    result["sla_passed"] = result["sla_pass"]
    result["total_iterations"] = result["measured_runs"]
    return result
