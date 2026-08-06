"""FR-3-owned adapters over the FR-1/FR-2 NSL-KDD pipeline.

These modules read the teammate-owned ``src/`` helpers but never modify them,
and never write into ``data/`` or ``artifacts/``. They exist so FR-3 can train
on a leakage-safe, versioned, train-only handoff instead of the merged
prototype outputs checked into ``data/angles.npy``.
"""

from .nslkdd import (
    ATTACK_FAMILIES,
    CONTRACT_ID,
    CONTRACT_VERSION,
    ContractSpec,
    TrainContract,
    build_train_contract,
    load_train_contract,
)

__all__ = [
    "ATTACK_FAMILIES",
    "CONTRACT_ID",
    "CONTRACT_VERSION",
    "ContractSpec",
    "TrainContract",
    "build_train_contract",
    "load_train_contract",
]
