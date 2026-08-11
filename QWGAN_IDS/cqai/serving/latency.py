"""FR-7 Latency Benchmarking Engine for Live Scoring.

Measures p50, p95, and p99 end-to-end scoring latency under representative load.
"""
from __future__ import annotations

import time
from typing import Any
import numpy as np

from cqai.serving.pipeline import ClassicalServingPipeline


def benchmark_serving_latency(
    pipeline: ClassicalServingPipeline,
    X_sample: np.ndarray,
    *,
    n_iterations: int = 100,
    max_p99_ms: float = 50.0,
) -> dict[str, Any]:
    """Benchmark end-to-end latency of the classical serving pipeline.

    Parameters
    ----------
    pipeline : ClassicalServingPipeline
        Fitted serving pipeline.
    X_sample : np.ndarray
        Representative sample flow batch.
    n_iterations : int, default=100
        Number of scoring iterations to run for latency stats.
    max_p99_ms : float, default=50.0
        Maximum allowed p99 latency SLA in milliseconds.

    Returns
    -------
    dict[str, Any]
        Latency summary dict containing p50_ms, p95_ms, p99_ms, max_ms, and SLA verdict.
    """
    latencies_ms: list[float] = []

    # Warmup run
    pipeline.predict_flow(X_sample[:1])

    for _ in range(n_iterations):
        t0 = time.perf_counter()
        pipeline.predict_flow(X_sample)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    p50_ms = float(np.percentile(latencies_ms, 50))
    p95_ms = float(np.percentile(latencies_ms, 95))
    p99_ms = float(np.percentile(latencies_ms, 99))
    max_ms = float(np.max(latencies_ms))

    sla_passed = p99_ms <= max_p99_ms

    return {
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "max_ms": max_ms,
        "sla_threshold_ms": max_p99_ms,
        "sla_passed": bool(sla_passed),
        "total_iterations": n_iterations,
    }
