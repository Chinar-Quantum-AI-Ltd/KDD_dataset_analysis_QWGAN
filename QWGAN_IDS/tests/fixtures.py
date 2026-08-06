"""Deterministic tiny NSL-KDD fixtures for fast FR-3 and FR-4 tests.

The real ``KDDTrain+.txt`` has 125 973 rows. Nothing in the test suite may
depend on it: tests must stay fast, offline, and independent of Git LFS.
"""
from __future__ import annotations

import random
from pathlib import Path

# NSL-KDD raw labels grouped the way the contract builder groups them.
FIXTURE_LABELS = {
    "normal": ["normal"],
    "dos": ["neptune", "smurf", "back"],
    "probe": ["satan", "ipsweep"],
    "r2l": ["guess_passwd", "warezclient"],
    "u2r": ["buffer_overflow", "rootkit"],
}

PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = ["http", "private", "domain_u", "smtp", "ftp_data"]
FLAGS = ["SF", "S0", "REJ", "RSTO"]


def _row(rng: random.Random, label: str, family: str) -> list[str]:
    """Build one 43-column NSL-KDD record (41 features + label + difficulty).

    Values are drawn per family so the families are actually separable; a
    contract built on noise would make the fidelity-relevant tests vacuous.
    """

    offset = {"normal": 0.0, "dos": 2.0, "probe": 4.0, "r2l": 6.0, "u2r": 8.0}[family]

    def rate() -> str:
        return str(round(min(1.0, abs(rng.gauss(offset / 10.0, 0.1))), 4))

    def count() -> str:
        return str(rng.randint(0, 255))

    fields = [
        str(round(abs(rng.gauss(offset, 0.5)), 4)),          # 1  duration
        rng.choice(PROTOCOLS),                               # 2  protocol_type
        rng.choice(SERVICES),                                # 3  service
        rng.choice(FLAGS),                                   # 4  flag
        str(round(abs(rng.gauss(offset * 100, 50.0)), 4)),   # 5  src_bytes
        str(round(abs(rng.gauss(offset * 10, 5.0)), 4)),     # 6  dst_bytes
    ]
    fields += [str(rng.randint(0, 1)) for _ in range(16)]    # 7-22 binary flags
    fields += [count(), count()]                             # 23-24 count, srv_count
    fields += [rate() for _ in range(7)]                     # 25-31 serror..srv_diff_host
    fields += [count(), count()]                             # 32-33 dst_host counts
    fields += [rate() for _ in range(8)]                     # 34-41 dst_host rates
    fields += [label, str(rng.randint(0, 21))]               # 42-43 label, difficulty

    assert len(fields) == 43, len(fields)
    return fields


def write_nslkdd_fixture(
    path: str | Path,
    *,
    rows_per_label: int = 24,
    seed: int = 0,
) -> Path:
    """Write a tiny KDDTrain+-shaped file and return its path."""

    rng = random.Random(seed)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    records: list[list[str]] = []
    for family, labels in FIXTURE_LABELS.items():
        for label in labels:
            for _ in range(rows_per_label):
                records.append(_row(rng, label, family))
    rng.shuffle(records)

    destination.write_text(
        "\n".join(",".join(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return destination
