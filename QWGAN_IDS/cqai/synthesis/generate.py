"""Sample a trained FR-3 generator and decode into the fidelity feature space.

Generation is forward-pass only: no optimizer, no gradients, no training data.
The one non-obvious responsibility here is refusing to decode with the wrong
transforms. An inverse PCA fitted on a different contract runs happily and
returns plausible-looking numbers, so a mismatch cannot be noticed downstream
in a metric -- it has to be caught by hash before anything is produced.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from ..qwgan.trainer import QWGANTrainer

if TYPE_CHECKING:  # pragma: no cover - import cycle guard only
    from ..data.nslkdd import TrainContract

#: Samples per simulator call. Broadcasting 21 550 rows through
#: ``default.qubit`` in one call allocates roughly 350 MB of statevector; the
#: noise is drawn in full up front so the chunk size cannot change the result.
DEFAULT_CHUNK_SIZE = 2048


@dataclass(frozen=True, slots=True)
class SyntheticBatch:
    """One generated batch, in both the angle and decoded feature domains."""

    attack_class: str
    seed: int
    angles: np.ndarray = field(repr=False)
    noise: np.ndarray = field(repr=False)
    decoded: pd.DataFrame = field(repr=False)
    checkpoint: str
    checkpoint_sha256: str
    epoch: int | None = None

    def __len__(self) -> int:
        return int(self.angles.shape[0])


def resolve_checkpoint(pattern: str | Path) -> Path:
    """Resolve a checkpoint path that may end in a wildcard.

    Early stopping means each seed finishes on a different epoch, so a template
    with a fixed epoch number points at nothing for some seeds. A pattern such
    as ``.../seed13/epoch*.pt`` resolves to the highest epoch present.

    Ordering is numeric rather than lexicographic: the runner zero-pads, but
    relying on that would break quietly the first time something else writes a
    checkpoint without padding.
    """

    path = Path(pattern)
    if "*" not in str(pattern):
        if not path.is_file():
            raise FileNotFoundError(f"no checkpoint at {path}")
        return path

    matches = sorted(
        path.parent.glob(path.name),
        key=lambda candidate: (
            int("".join(ch for ch in candidate.stem if ch.isdigit()) or -1),
            candidate.name,
        ),
    )
    if not matches:
        raise FileNotFoundError(f"no checkpoint matching {pattern}")
    return matches[-1]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_samples(
    checkpoint: str | Path,
    *,
    contract: "TrainContract",
    count: int,
    seed: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> SyntheticBatch:
    """Draw ``count`` samples from an FR-3 checkpoint and decode them."""

    if count < 1:
        raise ValueError("count must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    path = resolve_checkpoint(checkpoint)
    trainer, metadata = QWGANTrainer.load_checkpoint(path)

    recorded = metadata.get("contract_source_sha256")
    if recorded != contract.source_sha256:
        raise ValueError(
            "checkpoint was trained against a different contract "
            f"({recorded!r} != {contract.source_sha256!r}); decoding with the "
            "wrong transforms fails silently, so it is refused here"
        )

    # Draw the whole noise block first: chunking is a memory concern and must
    # not be observable in the output.
    noise = trainer.sample_noise(count, seed=seed)
    pieces = [
        trainer.generate_from_noise(noise[start : start + chunk_size])
        for start in range(0, count, chunk_size)
    ]
    angles = torch.cat(pieces, dim=0).detach().numpy()

    low, high = contract.angle_range
    angles = np.clip(angles, low, high)

    return SyntheticBatch(
        attack_class=str(metadata["attack_class"]),
        seed=seed,
        angles=angles,
        noise=noise.detach().numpy(),
        decoded=contract.decode_angles(angles),
        checkpoint=str(path),
        checkpoint_sha256=sha256_file(path),
        epoch=metadata.get("epoch"),
    )


def describe(batch: SyntheticBatch) -> dict[str, Any]:
    """Lineage fields a manifest needs from a batch."""

    return {
        "attack_class": batch.attack_class,
        "synthesis_seed": batch.seed,
        "count": len(batch),
        "checkpoint": batch.checkpoint,
        "checkpoint_sha256": batch.checkpoint_sha256,
        "checkpoint_epoch": batch.epoch,
    }
