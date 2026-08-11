import numpy as np
from .smote import generate_smote_samples
from .adasyn import generate_adasyn_samples
from .classical_wgan_gp import generate_wgan_samples


def _compute_target_minority_count(y_train, target_class, target_ratio):
    classes, counts = np.unique(y_train, return_counts=True)
    majority_count = counts.max()
    target_minority_count = int(np.ceil(target_ratio * majority_count))
    return target_minority_count


def build_augmented_dataset(
    X_train,
    y_train,
    method='smote',
    target_class=1,
    target_ratio=0.30,
    random_state=None,
    wgan_config=None,
):
    """Return X_augmented, y_augmented using the requested method.

    method: 'smote', 'adasyn', or 'classical_wgan'
    wgan_config: dict passed to generate_wgan_samples when method == 'classical_wgan'
    """
    X = np.asarray(X_train)
    y = np.asarray(y_train)

    existing_minority = int((y == target_class).sum())
    target_minority_count = _compute_target_minority_count(y, target_class, target_ratio)
    n_to_generate = target_minority_count - existing_minority

    if n_to_generate <= 0:
        return X.copy(), y.copy()

    if method == 'smote':
        X_synth, y_synth = generate_smote_samples(X, y, target_class, target_ratio, random_state)
    elif method == 'adasyn':
        X_synth, y_synth = generate_adasyn_samples(X, y, target_class, target_ratio, random_state)
    elif method == 'classical_wgan':
        # Train WGAN on minority samples only
        X_min = X[y == target_class]
        if wgan_config is None:
            wgan_config = {}
        X_synth = generate_wgan_samples(
            X_minority=X_min,
            n_samples=n_to_generate,
            latent_dim=wgan_config.get('latent_dim', 10),
            hidden_dim=wgan_config.get('hidden_dim', 128),
            batch_size=wgan_config.get('batch_size', 64),
            epochs=wgan_config.get('epochs', 100),
            lr=wgan_config.get('learning_rate', 1e-4),
            betas=(wgan_config.get('beta1', 0.0), wgan_config.get('beta2', 0.9)),
            lambda_gp=wgan_config.get('lambda_gp', 10.0),
            n_critic=wgan_config.get('n_critic', 5),
            device=wgan_config.get('device', None),
            random_state=random_state,
        )
        y_synth = np.array([target_class] * len(X_synth))
    else:
        raise ValueError(f"Unknown augmentation method: {method}")

    # Combine
    X_aug = np.vstack([X, np.asarray(X_synth)])
    y_aug = np.hstack([y, np.asarray(y_synth)])
    return X_aug, y_aug
