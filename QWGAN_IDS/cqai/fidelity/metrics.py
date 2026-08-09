"""Class-conditional fidelity metrics for the FR-4 gate.

Every metric here is **per feature or per real point**, never pooled into a
single global number. A generator that reproduces two features and destroys the
third is a failure, and pooling would average exactly that away. The gate
applies thresholds to the summaries; the per-feature values are what make a
failure diagnosable.

All four signals the TDD mandates live here: per-feature Wasserstein-1,
per-feature Kolmogorov-Smirnov, classifier two-sample test AUC, and mode
coverage. Domain validity is the fifth and lives in ``domain.py``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

#: TDD section 15.1 targets C2ST AUC <= 0.65; 0.5 means indistinguishable.
CHANCE_AUC = 0.5


def _aligned(real: pd.DataFrame, synthetic: pd.DataFrame) -> list[str]:
    """Columns common to both frames, in the real frame's order.

    The real frame is authoritative: a synthetic frame carrying extra columns
    is a bug upstream, and silently comparing on the union would hide it.
    """

    columns = list(real.columns)
    missing = [name for name in columns if name not in synthetic.columns]
    if missing:
        raise KeyError(f"synthetic frame is missing columns: {missing}")
    return columns


def _summarise(per_feature: dict[str, float]) -> dict[str, Any]:
    values = list(per_feature.values())
    return {
        "per_feature": per_feature,
        "max": float(max(values)) if values else 0.0,
        "mean": float(np.mean(values)) if values else 0.0,
    }


def _scale(values: np.ndarray) -> float:
    """Robust spread of a real feature, used to make W-1 comparable.

    Interquartile range first: NSL-KDD byte counters are extremely heavy-tailed,
    and a standard deviation dominated by a handful of outliers would make the
    normalised distance toothless. Sparse columns such as ``dst_bytes`` are zero
    for most rows and have IQR 0, so the standard deviation is the fallback; a
    genuinely constant column falls back to 1.0, which leaves the raw distance
    in place rather than dividing by an arbitrary epsilon.
    """

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    iqr = float(np.percentile(finite, 75) - np.percentile(finite, 25))
    if iqr > 0:
        return iqr
    deviation = float(np.std(finite))
    return deviation if deviation > 0 else 1.0


def wasserstein_1(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any]:
    """Per-feature Wasserstein-1 distance, raw and scale-normalised.

    The raw distance is in each feature's own units, so one tolerance band
    cannot serve both ``src_bytes`` (thousands) and ``same_srv_rate`` (bounded
    by 1) -- a threshold loose enough for the first is meaningless for the
    second. Each distance is therefore also divided by the *real* column's
    robust spread, and the gate applies its band to the normalised values.
    Raw values are kept because they are what a reader can sanity-check against
    the feature's real units.
    """

    columns = _aligned(real, synthetic)
    raw: dict[str, float] = {}
    normalised: dict[str, float] = {}
    for name in columns:
        real_values = np.asarray(real[name], dtype=np.float64)
        distance = float(
            wasserstein_distance(
                real_values, np.asarray(synthetic[name], dtype=np.float64)
            )
        )
        raw[name] = distance
        normalised[name] = distance / _scale(real_values)

    summary = _summarise(raw)
    summary["per_feature_normalised"] = normalised
    values = list(normalised.values())
    summary["max_normalised"] = float(max(values)) if values else 0.0
    summary["mean_normalised"] = float(np.mean(values)) if values else 0.0
    return summary


def ks_statistic(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any]:
    """Per-feature two-sample Kolmogorov-Smirnov statistic.

    Unlike Wasserstein-1 this is scale-free and bounded in ``[0, 1]``, which is
    why the gate carries both: KS catches a distribution-shape mismatch that a
    small absolute W-1 on a narrow feature would understate.
    """

    columns = _aligned(real, synthetic)
    return _summarise(
        {
            name: float(
                ks_2samp(
                    np.asarray(real[name], dtype=np.float64),
                    np.asarray(synthetic[name], dtype=np.float64),
                ).statistic
            )
            for name in columns
        }
    )


def c2st_auc(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    seed: int,
    folds: int = 5,
) -> dict[str, Any]:
    """Cross-validated classifier two-sample test AUC.

    A discriminator is trained to separate real from synthetic rows; an AUC near
    ``0.5`` means it cannot, which is the pass direction. Two choices matter:

    * **Balanced.** The synthetic pool is far larger than the held-out real one
      by design -- a ratio sweep produces tens of thousands of rows against 198
      real ones -- and class imbalance would inflate the AUC on its own. The
      synthetic side is subsampled to the real count with a seeded RNG.
    * **Cross-validated.** A single split of a few hundred rows has too much
      variance to support a pass/fail claim, and the standard deviation across
      folds is itself reported so a borderline result is visible as borderline.
    """

    columns = _aligned(real, synthetic)
    real_values = np.asarray(real[columns], dtype=np.float64)
    synthetic_values = np.asarray(synthetic[columns], dtype=np.float64)

    rng = np.random.default_rng(seed)
    n_per_side = int(min(len(real_values), len(synthetic_values)))
    if n_per_side < folds:
        raise ValueError(
            f"need at least {folds} rows per side for {folds}-fold C2ST, "
            f"got {n_per_side}"
        )
    if len(synthetic_values) > n_per_side:
        chosen = rng.choice(len(synthetic_values), size=n_per_side, replace=False)
        synthetic_values = synthetic_values[chosen]
    if len(real_values) > n_per_side:
        chosen = rng.choice(len(real_values), size=n_per_side, replace=False)
        real_values = real_values[chosen]

    features = np.vstack([real_values, synthetic_values])
    labels = np.concatenate(
        [np.zeros(n_per_side, dtype=int), np.ones(n_per_side, dtype=int)]
    )

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores: list[float] = []
    for train_index, test_index in splitter.split(features, labels):
        classifier = RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=1
        )
        classifier.fit(features[train_index], labels[train_index])
        probabilities = classifier.predict_proba(features[test_index])[:, 1]
        scores.append(float(roc_auc_score(labels[test_index], probabilities)))

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "per_fold": scores,
        "folds": folds,
        "n_per_side": n_per_side,
    }


def novelty_ratio(
    synthetic: pd.DataFrame,
    train_reference: pd.DataFrame,
    held_out: pd.DataFrame,
) -> float:
    """How far synthetic samples sit from the training set, relative to real data.

    A model can pass a two-sample test by drifting toward the rows it was fitted
    on: samples that are near-copies of the training set look exactly like real
    data of that class, and contribute nothing to an augmented one. Measured on
    NSL-KDD ``r2l``, a 120-component Gaussian mixture clears every other gate
    criterion while sitting 29 % closer to the training rows than genuine
    held-out rows do.

    An absolute distance cannot express that -- a naturally dense class sits
    close to itself. The reference is therefore how far real held-out rows of
    the same class fall from the training set: a ratio near 1 means the
    synthetic samples are as novel as real data, and a ratio near 0 means the
    model is echoing what it was fitted on.
    """

    columns = _aligned(train_reference, synthetic)
    _aligned(train_reference, held_out)
    train_values = np.asarray(train_reference[columns], dtype=np.float64)
    if len(train_values) == 0:
        return 1.0

    neighbours = NearestNeighbors(n_neighbors=1).fit(train_values)

    def median_distance(frame: pd.DataFrame) -> float:
        values = np.asarray(frame[columns], dtype=np.float64)
        if len(values) == 0:
            return 0.0
        return float(np.median(neighbours.kneighbors(values)[0][:, 0]))

    reference = median_distance(held_out)
    if reference <= 0:
        return 1.0
    return median_distance(synthetic) / reference


def coverage(
    real: pd.DataFrame, synthetic: pd.DataFrame, *, k: int = 5
) -> float:
    """Fraction of real points whose neighbourhood contains a synthetic point.

    Each real point gets a radius from its own ``k``-th nearest *real*
    neighbour, and counts as covered when at least one synthetic point falls
    inside it. Deriving the radius from the real data rather than a fixed
    constant means a naturally tight attack class is not scored as uncovered.

    This is the mode-coverage signal the TDD asks for, and it answers a
    different question from FR-3's ``diversity_ratio``: a generator emitting one
    point repeatedly can still show a plausible spread ratio against a tight
    real class, but it cannot reach the real modes.
    """

    columns = _aligned(real, synthetic)
    real_values = np.asarray(real[columns], dtype=np.float64)
    synthetic_values = np.asarray(synthetic[columns], dtype=np.float64)
    if len(real_values) == 0 or len(synthetic_values) == 0:
        return 0.0

    # k+1 because a point is its own nearest neighbour.
    neighbours = min(k + 1, len(real_values))
    real_nn = NearestNeighbors(n_neighbors=neighbours).fit(real_values)
    radii = real_nn.kneighbors(real_values)[0][:, -1]

    synthetic_nn = NearestNeighbors(n_neighbors=1).fit(synthetic_values)
    nearest_synthetic = synthetic_nn.kneighbors(real_values)[0][:, 0]

    return float(np.mean(nearest_synthetic <= radii))
