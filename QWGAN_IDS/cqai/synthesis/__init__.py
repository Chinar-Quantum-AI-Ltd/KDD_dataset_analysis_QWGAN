"""FR-4 synthesis: sample trained generators, decode, and route through the gate.

This package makes samples. It does not decide whether they are good -- that is
``cqai.fidelity`` -- so the gate can never be tuned to whatever the current
generator happens to produce.
"""

from .generate import (
    DEFAULT_CHUNK_SIZE,
    SyntheticBatch,
    describe,
    generate_samples,
    sha256_file,
)
from .runner import (
    DECODE_ARTIFACTS,
    SynthesisPlan,
    SynthesisResult,
    requested_volume,
    run_synthesis,
)

__all__ = [
    "DECODE_ARTIFACTS",
    "DEFAULT_CHUNK_SIZE",
    "SynthesisPlan",
    "SynthesisResult",
    "SyntheticBatch",
    "describe",
    "generate_samples",
    "requested_volume",
    "run_synthesis",
    "sha256_file",
]
