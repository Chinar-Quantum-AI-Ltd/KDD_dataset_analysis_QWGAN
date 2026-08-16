"""FR-7 Low-Latency Classical Serving and Latency Benchmarking Package.

Enforces pure classical offline-prepared inference path with zero live quantum calls
and benchmarks latency SLAs (p99 <= 50 ms/flow).
"""
from .latency import benchmark_batch_latency, benchmark_serving_latency, latency_percentiles
from .pipeline import ClassicalServingPipeline

__all__ = [
    "ClassicalServingPipeline",
    "benchmark_serving_latency",
    "benchmark_batch_latency",
    "latency_percentiles",
]
