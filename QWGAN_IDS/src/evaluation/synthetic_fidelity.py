import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import wasserstein_distance


def c2st_auc(real, synthetic, classifier='logreg', random_state=None):
    """Classifier Two-Sample Test (C2ST) returning AUC distinguishing real vs synthetic.

    Trains a classifier to distinguish real (label 0) vs synthetic (label 1) and returns AUC on holdout.
    """
    X_real = np.asarray(real)
    X_syn = np.asarray(synthetic)
    X = np.vstack([X_real, X_syn])
    y = np.hstack([np.zeros(len(X_real)), np.ones(len(X_syn))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)

    if classifier == 'logreg':
        clf = LogisticRegression(max_iter=1000)
    else:
        clf = LogisticRegression(max_iter=1000)

    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    return float(auc)


def wasserstein_distance_per_feature(real, synthetic):
    """Compute 1D Wasserstein (earth mover's) distance per feature.

    Returns a dict: {feature_idx: distance}
    """
    real = np.asarray(real)
    synthetic = np.asarray(synthetic)
    assert real.shape[1] == synthetic.shape[1], "Feature dimension mismatch"
    results = {}
    for i in range(real.shape[1]):
        results[i] = float(wasserstein_distance(real[:, i], synthetic[:, i]))
    return results


def validate_feature_constraints(samples):
    """Basic domain checks: finite values. Extend with dataset-specific rules as needed."""
    import numpy as np
    s = np.asarray(samples)
    ok = np.isfinite(s).all()
    return bool(ok)
