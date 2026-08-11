"""FR-7 Low-Latency Classical Serving and Latency Benchmarking Package.

Enforces pure classical offline-prepared inference path with zero live quantum calls
and benchmarks latency SLAs (p99 <= 50 ms/flow).
"""
from .latency import benchmark_serving_latency
from .pipeline import ClassicalServingPipeline

__all__ = [
    "ClassicalServingPipeline",
    "benchmark_serving_latency",
]
