"""The FR-4 class-conditional fidelity gate.

Synthetic samples are not trusted by default. This module turns the five
mandatory signals -- C2ST AUC, per-feature Wasserstein-1, per-feature KS, mode
coverage, and domain validity -- into one of three verdicts:

``pass``
    Every mandatory criterion cleared its versioned band.
``fail``
    At least one criterion did not. ``reasons`` names which.
``insufficient_evidence``
    The held-out real reference is too small to support a claim in either
    direction. NSL-KDD ``u2r`` has ten held-out rows; a C2ST AUC computed
    against ten rows cannot certify anything, and returning ``pass`` there would
    be worse than useless. The gate fails closed and the samples stay
    quarantined.

Only ``pass`` releases samples. The other two verdicts are equivalent as far as
downstream use is concerned -- they are distinguished so a report can tell
"we measured this and it was bad" apart from "we could not measure this".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .domain import check_domain
from .metrics import (
    c2st_auc,
    coverage,
    ks_statistic,
    novelty_ratio,
    wasserstein_1,
)

GATE_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class FidelityThresholds:
    """Versioned pass/fail bands.

    ``max_c2st_auc`` is the TDD's hard design threshold. The Wasserstein, KS,
    coverage and domain bands are tolerances that must be set in a versioned
    config before any pass/fail claim -- the defaults here are starting points,
    not calibrated values, and the run manifest records whichever were used.
    """

    #: TDD section 15.1 hard threshold. 0.5 means indistinguishable.
    max_c2st_auc: float = 0.65
    #: Applied to the *normalised* per-feature W-1 (distance divided by the
    #: real column's robust spread), so one band covers features whose units
    #: differ by orders of magnitude.
    max_wasserstein_1: float = 0.5
    #: Scale-free, bounded in [0, 1].
    max_ks_statistic: float = 0.2
    min_coverage: float = 0.5
    min_domain_validity: float = 0.95
    #: Below this many held-out real rows the gate refuses to certify.
    min_real_samples: int = 50
    #: Synthetic samples must sit at least this fraction as far from the
    #: training set as genuine held-out rows do. Below it, the model is echoing
    #: what it was fitted on: such samples pass a two-sample test trivially and
    #: add nothing to an augmented set.
    min_novelty_ratio: float = 0.8
    c2st_folds: int = 5


def _metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    thresholds: FidelityThresholds,
    seed: int,
    with_c2st: bool,
) -> dict[str, Any]:
    return {
        "c2st": (
            c2st_auc(real, synthetic, seed=seed, folds=thresholds.c2st_folds)
            if with_c2st
            else None
        ),
        "wasserstein_1": wasserstein_1(real, synthetic),
        "ks": ks_statistic(real, synthetic),
        "coverage": coverage(real, synthetic),
        "domain": check_domain(synthetic).as_dict(),
    }


def _null_reference(
    real: pd.DataFrame, *, thresholds: FidelityThresholds, seed: int
) -> dict[str, Any] | None:
    """What genuinely real data of this size scores against itself.

    A threshold is only readable next to its null baseline. If halving the real
    sample already yields a C2ST of 0.7 because the sample is small, then a
    synthetic 0.7 says nothing about the generator; and because the decode is
    lossy, decoded real rows can fail a domain or distance band outright. Each
    criterion is therefore calibrated against this baseline: it fires only when
    the synthetic batch is both outside its band and worse than real data of
    the same size.

    The baseline cannot loosen a criterion into a pass on its own -- it can only
    withhold blame the generator did not earn.
    """

    half = len(real) // 2
    if half < thresholds.c2st_folds:
        return None

    shuffled = real.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    left = shuffled.iloc[:half]
    right = shuffled.iloc[half : 2 * half]

    reference = _metrics(
        left, right, thresholds=thresholds, seed=seed, with_c2st=True
    )
    reference["n_per_side"] = half
    return reference


def _criterion(
    name: str,
    *,
    reason: str,
    value: float | None,
    threshold: float,
    null: float | None,
    higher_is_worse: bool,
) -> dict[str, Any]:
    """One gate criterion, calibrated against its null baseline.

    A criterion fires only when the synthetic batch is outside the configured
    band **and** worse than what genuinely real data of the same size scores.
    Both conditions are needed because the decode is lossy: decoding real
    held-out NSL-KDD rows through the fitted PCA and scalers yields a domain
    validity of 0.0, so an absolute check alone would blame the generator for
    the decode's limits. Where no null exists -- too few real rows to halve --
    the absolute band decides alone, which keeps the gate strict rather than
    letting a missing baseline wave a batch through.
    """

    if value is None:
        return {
            "value": None,
            "threshold": threshold,
            "null": null,
            "fired": False,
            "reason": reason,
            "note": "not measurable",
        }

    def worse(left: float, right: float) -> bool:
        return left > right if higher_is_worse else left < right

    out_of_band = worse(value, threshold)
    worse_than_null = null is None or worse(value, null)
    return {
        "value": value,
        "threshold": threshold,
        "null": null,
        "fired": bool(out_of_band and worse_than_null),
        "reason": reason,
    }


def evaluate_gate(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    train_reference: pd.DataFrame | None = None,
    thresholds: FidelityThresholds | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Score a decoded synthetic batch against decoded held-out real rows.

    Both sides must already be decoded: the TDD requires domain and
    feature-space checks on the decoded representation, not on angles.

    ``train_reference`` enables the novelty check. It is optional because every
    previously reported gate result was produced without it, and adding the
    signal silently would change what those runs mean.
    """

    thresholds = thresholds or FidelityThresholds()
    n_real = int(len(real))
    n_synthetic = int(len(synthetic))

    thin = n_real < thresholds.min_real_samples
    can_run_c2st = min(n_real, n_synthetic) >= thresholds.c2st_folds
    metrics = _metrics(
        real,
        synthetic,
        thresholds=thresholds,
        seed=seed,
        with_c2st=can_run_c2st,
    )
    # Novelty needs the rows the generator was fitted on, which the gate does
    # not otherwise see. Absent them the criterion is simply not measured.
    metrics["novelty"] = (
        novelty_ratio(synthetic, train_reference, real)
        if train_reference is not None
        else None
    )
    null = _null_reference(real, thresholds=thresholds, seed=seed)

    def from_null(*path: str) -> float | None:
        if null is None:
            return None
        node: Any = null
        for key in path:
            if node is None:
                return None
            node = node[key]
        return node

    criteria = {
        # The C2ST threshold is a TDD design requirement rather than a
        # tolerance, but its null is still recorded: chance is 0.5 in theory and
        # something else entirely on a few hundred rows.
        "c2st_auc": _criterion(
            "c2st_auc",
            reason="c2st_distinguishable",
            value=metrics["c2st"]["mean"] if metrics["c2st"] else None,
            threshold=thresholds.max_c2st_auc,
            null=from_null("c2st", "mean"),
            higher_is_worse=True,
        ),
        # Normalised, not raw: one band cannot serve features whose units
        # differ by three orders of magnitude.
        "wasserstein_1": _criterion(
            "wasserstein_1",
            reason="wasserstein_out_of_band",
            value=metrics["wasserstein_1"]["max_normalised"],
            threshold=thresholds.max_wasserstein_1,
            null=from_null("wasserstein_1", "max_normalised"),
            higher_is_worse=True,
        ),
        "ks_statistic": _criterion(
            "ks_statistic",
            reason="ks_out_of_band",
            value=metrics["ks"]["max"],
            threshold=thresholds.max_ks_statistic,
            null=from_null("ks", "max"),
            higher_is_worse=True,
        ),
        "coverage": _criterion(
            "coverage",
            reason="low_coverage",
            value=metrics["coverage"],
            threshold=thresholds.min_coverage,
            null=from_null("coverage"),
            higher_is_worse=False,
        ),
        "domain_validity": _criterion(
            "domain_validity",
            reason="domain_invalid",
            value=metrics["domain"]["valid_fraction"],
            threshold=thresholds.min_domain_validity,
            null=from_null("domain", "valid_fraction"),
            higher_is_worse=False,
        ),
        # Real held-out rows are the reference, so their own ratio is 1 by
        # construction. There is no sample-size null to compute here.
        "novelty": _criterion(
            "novelty",
            reason="memorised_training_data",
            value=metrics["novelty"],
            threshold=thresholds.min_novelty_ratio,
            null=1.0 if metrics["novelty"] is not None else None,
            higher_is_worse=False,
        ),
    }

    reasons = [
        criterion["reason"] for criterion in criteria.values() if criterion["fired"]
    ]
    if thin:
        reasons.insert(0, "insufficient_real_samples")
        verdict = "insufficient_evidence"
    else:
        verdict = "fail" if reasons else "pass"

    return {
        "gate_version": GATE_VERSION,
        "verdict": verdict,
        "reasons": reasons,
        "criteria": criteria,
        "thresholds": asdict(thresholds),
        "seed": seed,
        "n_real": n_real,
        "n_synthetic": n_synthetic,
        "metrics": metrics,
        "null_reference": null,
    }


def released(result: dict[str, Any]) -> bool:
    """True only for ``pass``.

    ``fail`` and ``insufficient_evidence`` are equally disqualifying for
    downstream use; the distinction exists for reporting, not for release.
    """

    return result["verdict"] == "pass"
