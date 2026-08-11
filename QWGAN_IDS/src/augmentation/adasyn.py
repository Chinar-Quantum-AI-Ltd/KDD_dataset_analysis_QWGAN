import numpy as np
from imblearn.over_sampling import ADASYN

def generate_adasyn_samples(X_train, y_train, target_class, target_ratio, random_state=None):
    """Generate synthetic minority samples using ADASYN.

    Returns X_synthetic, y_synthetic (only the generated samples for target_class).
    """
    X = np.asarray(X_train)
    y = np.asarray(y_train)

    classes, counts = np.unique(y, return_counts=True)
    majority_count = counts.max()
    target_minority_count = int(np.ceil(target_ratio * majority_count))
    existing_minority = int((y == target_class).sum())
    n_to_generate = target_minority_count - existing_minority

    if n_to_generate <= 0:
        return np.empty((0, X.shape[1])), np.empty((0,), dtype=y.dtype)

    key = int(target_class) if str(target_class).isdigit() else target_class
    sampling_strategy = {key: target_minority_count}
    ad = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = ad.fit_resample(X, y)

    mask_target = (y_res == target_class)
    target_indices = np.where(mask_target)[0]
    synthetic_indices = target_indices[-n_to_generate:]

    X_synth = X_res[synthetic_indices]
    y_synth = y_res[synthetic_indices]
    return np.asarray(X_synth), np.asarray(y_synth)
