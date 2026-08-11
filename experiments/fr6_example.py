"""FR-6 ablation runner (dry-run/example).

This script demonstrates how to use the augmentation builders and fidelity checks on a toy dataset (sklearn.make_classification).
It is intended as a minimal usage example. For real experiments, adapt to your dataset loader and training pipeline.
"""
import json
import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.augmentation.dataset_builder import build_augmented_dataset
from src.evaluation.synthetic_fidelity import c2st_auc, wasserstein_distance_per_feature, validate_feature_constraints


DEF_OUT = 'results/fr6'
os.makedirs(DEF_OUT, exist_ok=True)


def run_example(seed=42):
    X, y = make_classification(n_samples=2000, n_features=10, n_informative=6, n_redundant=2,
                               n_clusters_per_class=1, weights=[0.95, 0.05], flip_y=0, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)

    target_class = 1
    target_ratio = 0.30

    # Arm A: real only
    X_A, y_A = X_train.copy(), y_train.copy()

    # Arm B: SMOTE
    X_B, y_B = build_augmented_dataset(X_train, y_train, method='smote', target_class=target_class, target_ratio=target_ratio, random_state=seed)

    # Arm B2: ADASYN
    X_B2, y_B2 = build_augmented_dataset(X_train, y_train, method='adasyn', target_class=target_class, target_ratio=target_ratio, random_state=seed)

    # Arm C: classical WGAN (short epochs for example)
    wgan_cfg = dict(latent_dim=10, hidden_dim=128, batch_size=64, epochs=10, learning_rate=1e-4, beta1=0.0, beta2=0.9, lambda_gp=10, n_critic=5)
    X_C, y_C = build_augmented_dataset(X_train, y_train, method='classical_wgan', target_class=target_class, target_ratio=target_ratio, random_state=seed, wgan_config=wgan_cfg)

    results = {}

    for name, (X_arm, y_arm) in [('A', (X_A, y_A)), ('B_SMOTE', (X_B, y_B)), ('B_ADASYN', (X_B2, y_B2)), ('C_WGAN', (X_C, y_C))]:
        # collect synthetic samples only for fidelity checks
        if name == 'A':
            results[name] = {'n_samples': len(X_arm)}
            continue

        # synthetic = new samples beyond original train
        n_original = len(X_train)
        X_synthetic = X_arm[n_original:]

        # basic validation
        ok = validate_feature_constraints(X_synthetic)
        auc = None
        wdist = None
        if X_synthetic.shape[0] > 0 and ok:
            auc = c2st_auc(X_train[y_train == target_class], X_synthetic, random_state=seed)
            wdist = wasserstein_distance_per_feature(X_train[y_train == target_class], X_synthetic)

        manifest = {
            'arm': name,
            'seed': int(seed),
            'n_original': int(n_original),
            'n_synthetic': int(X_synthetic.shape[0]),
            'fidelity': {
                'c2st_auc': auc,
                'mean_wasserstein_1': float(np.mean(list(wdist.values()))) if wdist else None,
            }
        }
        results[name] = manifest

        # save manifest
        with open(os.path.join(DEF_OUT, f'manifest_{name}_seed_{seed}.json'), 'w') as f:
            json.dump(manifest, f, indent=2)

    # save summary
    with open(os.path.join(DEF_OUT, f'summary_seed_{seed}.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print('Example run complete. Results saved to', DEF_OUT)


if __name__ == '__main__':
    run_example(42)
