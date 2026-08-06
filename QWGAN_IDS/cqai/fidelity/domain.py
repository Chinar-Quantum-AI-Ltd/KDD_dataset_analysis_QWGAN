"""NSL-KDD domain constraints over the decoded feature space.

The rule table below is written out explicitly rather than derived from
``src.preprocessing.BINARY_COLS``. That list is named "binary" but contains
count columns -- ``hot``, ``num_root``, ``num_compromised``,
``num_file_creations`` and others -- so driving validity from it would license
any value in ``{0, 1}`` for a counter and silently accept invalid records.

Bounds come from the NSL-KDD schema: connection counters are capped at 511 and
per-host counters at 255, rates are proportions in ``[0, 1]``, and one-hot
indicators must survive the lossy decode close enough to ``0`` or ``1`` to still
be a category. A decoded ``0.5`` for ``logged_in`` is not a value the encoder
could ever have produced, so it is a violation rather than a rounding artefact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

RuleKind = Literal["count", "rate", "indicator"]

#: How far a decoded indicator may sit from 0 or 1 and still count as that
#: category. PCA plus top-k selection is lossy, so an exact comparison would
#: reject essentially every decoded row, real ones included.
INDICATOR_TOLERANCE = 0.1


@dataclass(frozen=True, slots=True)
class Rule:
    kind: RuleKind
    minimum: float = 0.0
    maximum: float = float("inf")


#: NSL-KDD connection counters saturate at 511, per-host counters at 255.
RULES: dict[str, Rule] = {
    # Byte counters: non-negative, no schema ceiling.
    "src_bytes": Rule("count"),
    "dst_bytes": Rule("count"),
    # Connection window counters.
    "count": Rule("count", maximum=511),
    "srv_count": Rule("count", maximum=511),
    # Per-destination-host counters.
    "dst_host_count": Rule("count", maximum=255),
    "dst_host_srv_count": Rule("count", maximum=255),
    # Proportions.
    "same_srv_rate": Rule("rate", maximum=1.0),
    "diff_srv_rate": Rule("rate", maximum=1.0),
    "serror_rate": Rule("rate", maximum=1.0),
    "srv_serror_rate": Rule("rate", maximum=1.0),
    "rerror_rate": Rule("rate", maximum=1.0),
    "srv_rerror_rate": Rule("rate", maximum=1.0),
    "srv_diff_host_rate": Rule("rate", maximum=1.0),
    "dst_host_same_srv_rate": Rule("rate", maximum=1.0),
    "dst_host_diff_srv_rate": Rule("rate", maximum=1.0),
    "dst_host_serror_rate": Rule("rate", maximum=1.0),
    "dst_host_srv_serror_rate": Rule("rate", maximum=1.0),
    "dst_host_rerror_rate": Rule("rate", maximum=1.0),
    "dst_host_srv_rerror_rate": Rule("rate", maximum=1.0),
    "dst_host_same_src_port_rate": Rule("rate", maximum=1.0),
    "dst_host_srv_diff_host_rate": Rule("rate", maximum=1.0),
    # Genuinely binary flags (not the misnamed BINARY_COLS list).
    "logged_in": Rule("indicator", maximum=1.0),
    "land": Rule("indicator", maximum=1.0),
    "root_shell": Rule("indicator", maximum=1.0),
    "is_host_login": Rule("indicator", maximum=1.0),
    "is_guest_login": Rule("indicator", maximum=1.0),
}

#: One-hot groups that cannot both be set on one flow. Only the members the
#: contract actually selected are checked, so the constraint is "at most one"
#: rather than "exactly one".
EXCLUSIVE_GROUPS: dict[str, tuple[str, ...]] = {
    "flag_exclusive": ("flag_SF", "flag_S0", "flag_REJ", "flag_RSTO", "flag_RSTR"),
    "protocol_exclusive": ("protocol_type_tcp", "protocol_type_udp", "protocol_type_icmp"),
}


@dataclass(frozen=True, slots=True)
class DomainReport:
    """Overall validity plus the per-rule breakdown that makes it actionable."""

    valid_fraction: float
    rows: int
    checked_columns: list[str] = field(default_factory=list)
    #: rule name -> fraction of rows violating it. Absent means never violated.
    violations: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "valid_fraction": self.valid_fraction,
            "rows": self.rows,
            "checked_columns": list(self.checked_columns),
            "violations": dict(self.violations),
        }


def check_domain(
    frame: pd.DataFrame, *, indicator_tolerance: float = INDICATOR_TOLERANCE
) -> DomainReport:
    """Score decoded rows against the NSL-KDD schema.

    Only columns present in ``frame`` are checked: the contract selects its
    features by mutual information, so a different spec yields a different
    subset and the rule table must adapt rather than demand a fixed schema.
    """

    rows = int(len(frame))
    if rows == 0:
        return DomainReport(valid_fraction=0.0, rows=0)

    checked = [name for name in frame.columns if name in RULES]
    violations: dict[str, float] = {}
    invalid_rows = np.zeros(rows, dtype=bool)

    for name in checked:
        rule = RULES[name]
        values = np.asarray(frame[name], dtype=np.float64)
        bad = ~np.isfinite(values)
        finite = np.isfinite(values)

        if rule.kind == "indicator":
            distance = np.minimum(
                np.abs(values - rule.minimum), np.abs(values - rule.maximum)
            )
            bad |= finite & (distance > indicator_tolerance)
        else:
            bad |= finite & (values < rule.minimum)
            bad |= finite & (values > rule.maximum)

        if bad.any():
            violations[name] = float(bad.mean())
        invalid_rows |= bad

    for group, members in EXCLUSIVE_GROUPS.items():
        present = [name for name in members if name in frame.columns]
        if len(present) < 2:
            continue
        # Count how many members read as "set"; more than one is impossible for
        # a one-hot encoding of a single categorical field.
        as_set = sum(
            (np.asarray(frame[name], dtype=np.float64) > 0.5).astype(int)
            for name in present
        )
        bad = as_set > 1
        if bad.any():
            violations[group] = float(bad.mean())
        invalid_rows |= bad

    return DomainReport(
        valid_fraction=float(1.0 - invalid_rows.mean()),
        rows=rows,
        checked_columns=checked,
        violations=violations,
    )
