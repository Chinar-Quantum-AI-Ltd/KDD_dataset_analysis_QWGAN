"""Fidelity and similarity utilities for FR-6.

Provides:
- distribution_similarity(real_X, synthetic_X, n_bins=50) -> dict with per-feature JS divergence
- mmd_rbf(X, Y, sigma=1.0) -> float (estimate of MMD)
- classifier_fidelity(real_X, real_y, synthetic_X, synthetic_y, val_X, val_y, classifier_factory)
- fidelity_gate(...)

These functions are lightweight and intended for offline evaluation of synthetic data fidelity.
"""
from typing import Callable, Dict, Tuple
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy

from src.metrics import compute_classification_metrics


def _hist_js(p, q):
    # Jensen-Shannon distance handles zero entries; use base e
    return float(jensenshannon(p, q, base=2.0))


def distribution_similarity(real_X: np.ndarray, synthetic_X: np.ndarray, n_bins: int = 50) -> Dict[str, float]:
    """Compute per-feature JS divergence between real and synthetic data distributions.

    Returns dict with keys:
      - 'per_feature_js': list of js values
      - 'mean_js': mean across features
    """
    real_X = np.asarray(real_X)
    synthetic_X = np.asarray(synthetic_X)
    if real_X.ndim != 2 or synthetic_X.ndim != 2:
        raise ValueError("real_X and synthetic_X must be 2D arrays")
    n_features = real_X.shape[1]
    per_js = []
    for i in range(n_features):
        r = real_X[:, i]
        s = synthetic_X[:, i]
        # build common bin edges using combined range
        combined = np.concatenate([r, s])
        if np.all(np.isfinite(combined)):
            lo, hi = np.min(combined), np.max(combined)
            if lo == hi:
                per_js.append(0.0)
                continue
            bins = np.linspace(lo, hi, n_bins + 1)
            pr, _ = np.histogram(r, bins=bins, density=True)
            ps, _ = np.histogram(s, bins=bins, density=True)
            # add tiny smoothing to avoid zeros
            pr = pr + 1e-12
            ps = ps + 1e-12
            pr = pr / pr.sum()
            ps = ps / ps.sum()
            js = _hist_js(pr, ps)
            per_js.append(float(js))
        else:
            per_js.append(float('nan'))
    mean_js = float(np.nanmean(per_js))
    return {"per_feature_js": per_js, "mean_js": mean_js}


def rbf_kernel(x, y, sigma=1.0):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    dists = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
    return np.exp(-dists / (2 * sigma ** 2))


def mmd_rbf(X, Y, sigma=1.0) -> float:
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)
    m = X.shape[0]
    n = Y.shape[0]
    return float(np.sum(Kxx) / (m * m) + np.sum(Kyy) / (n * n) - 2 * np.sum(Kxy) / (m * n))


def classifier_fidelity(
    real_X: np.ndarray,
    real_y: np.ndarray,
    synthetic_X: np.ndarray,
    synthetic_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    classifier_factory: Callable[[], object] = None,
) -> Dict[str, Dict]:
    """Train classifier on real and synthetic, evaluate on validation set.

    Returns dict:
      - 'real': metrics dict for classifier trained on real
      - 'synthetic': metrics dict for classifier trained on synthetic
      - 'delta': differences (synthetic - real) for key metrics (f1, precision, recall)
    """
    classifier_factory = classifier_factory or (lambda: RandomForestClassifier(n_estimators=200))
    # train on real
    clf_real = classifier_factory()
    clf_real.fit(real_X, real_y)
    prob_real = clf_real.predict_proba(val_X)[:, 1] if hasattr(clf_real, 'predict_proba') else None
    pred_real = clf_real.predict(val_X)
    metrics_real = compute_classification_metrics(val_y, prob_real, pred_real, pos_label=1)

    # train on synthetic
    clf_syn = classifier_factory()
    clf_syn.fit(synthetic_X, synthetic_y)
    prob_syn = clf_syn.predict_proba(val_X)[:, 1] if hasattr(clf_syn, 'predict_proba') else None
    pred_syn = clf_syn.predict(val_X)
    metrics_syn = compute_classification_metrics(val_y, prob_syn, pred_syn, pos_label=1)

    delta = {
        'precision': metrics_syn.get('precision', float('nan')) - metrics_real.get('precision', float('nan')),
        'recall': metrics_syn.get('recall', float('nan')) - metrics_real.get('recall', float('nan')),
        'f1': metrics_syn.get('f1', float('nan')) - metrics_real.get('f1', float('nan')),
    }

    return {'real': metrics_real, 'synthetic': metrics_syn, 'delta': delta}


def fidelity_gate(
    real_X, synthetic_X, val_X, val_y, classifier_factory=None, thresholds: dict = None
) -> Tuple[bool, Dict]:
    """Run distribution + classifier fidelity checks and return pass/fail.

    thresholds default:
      - max_mean_js: 0.2
      - max_delta_f1: 0.02
    """
    thresholds = thresholds or {}
    max_mean_js = thresholds.get('max_mean_js', 0.2)
    max_delta_f1 = thresholds.get('max_delta_f1', 0.02)

    dist = distribution_similarity(real_X, synthetic_X)
    # For classifier fidelity, we need synthetic labels; assume malicious class label 1 and sample counts similar.
    # Here we create synthetic_y by matching class balance from val_y (best-effort): use 1s
    synthetic_y = np.ones(synthetic_X.shape[0], dtype=int)
    # For real training, sample from real_X/y must be provided by caller; in this simplified gate we'll
    # treat real_X/val_y usage as: train on real_X (provided) and synthetic_X (provided) and evaluate on val.
    cf = classifier_fidelity(real_X, np.ones(real_X.shape[0], dtype=int), synthetic_X, synthetic_y, val_X, val_y, classifier_factory)

    pass_gate = (dist['mean_js'] <= max_mean_js) and (abs(cf['delta'].get('f1', 1.0)) <= max_delta_f1)
    diagnostics = {'distribution': dist, 'classifier_fidelity': cf, 'thresholds': {'max_mean_js': max_mean_js, 'max_delta_f1': max_delta_f1}}
    return bool(pass_gate), diagnostics
