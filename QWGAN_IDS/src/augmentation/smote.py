import numpy as np
from imblearn.over_sampling import SMOTE

def generate_smote_samples(X_train, y_train, target_class, target_ratio, random_state=None):
    """Generate synthetic minority samples using SMOTE.

    Returns X_synthetic, y_synthetic (only the generated samples for target_class).
    """
    X = np.asarray(X_train)
    y = np.asarray(y_train)

    # compute majority count
    classes, counts = np.unique(y, return_counts=True)
    majority_count = counts.max()
    target_minority_count = int(np.ceil(target_ratio * majority_count))
    existing_minority = int((y == target_class).sum())
    n_to_generate = target_minority_count - existing_minority

    if n_to_generate <= 0:
        return np.empty((0, X.shape[1])), np.empty((0,), dtype=y.dtype)

    # SMOTE sampling_strategy accepts dict mapping class to desired number of samples
    sampling_strategy = {int(target_class): target_minority_count}
    sm = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)

    # imblearn appends synthetic samples after original ones. Extract last n_to_generate samples
    # Filter those with class == target_class from the tail
    mask_target = (y_res == target_class)
    target_indices = np.where(mask_target)[0]
    # The first existing_minority indices correspond to original minority; the rest are synthetic.
    if len(target_indices) >= target_minority_count:
        synthetic_indices = target_indices[-n_to_generate:]
    else:
        synthetic_indices = target_indices[-n_to_generate:]

    X_synth = X_res[synthetic_indices]
    y_synth = y_res[synthetic_indices]
    return np.asarray(X_synth), np.asarray(y_synth)
