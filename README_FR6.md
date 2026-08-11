FR-6 Augmentation README

This folder contains code to run the FR-6 augmentation ablation (SMOTE, ADASYN, classical WGAN-GP) and basic fidelity checks.

Files added
- src/augmentation/: SMOTE, ADASYN, classical WGAN-GP, and dataset builder that returns consistent X_augmented, y_augmented.
- src/evaluation/synthetic_fidelity.py: C2ST AUC, per-feature Wasserstein-1, basic domain checks.
- configs/augmentation.yaml: default experiment configuration (seeds, target ratios, WGAN hyperparameters).
- experiments/fr6_example.py: minimal example using sklearn.make_classification to demo usage (dry-run).
- experiments/fr6_ablation.py: simple loop over seeds calling the example.

How to run (example)
1) Install dependencies (recommended in a virtualenv):
   pip install torch torchvision torchaudio  # or the appropriate CUDA build
   pip install scikit-learn imbalanced-learn scipy pyyaml

2) Run the example (dry-run):
   python experiments/fr6_example.py

3) Run the ablation over seeds:
   python experiments/fr6_ablation.py

Outputs
- results/fr6/: manifests and summaries for each run (do not commit generated datasets to the repo). For production experiments, upload generated CSVs and manifests to your artifact storage (S3, etc.) and include hashes and lineage metadata.
