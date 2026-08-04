# CQAI QWGAN-IDS

Hybrid quantum-classical data augmentation for minority-class network intrusion detection.

CQAI QWGAN-IDS uses a PennyLane parameterized quantum circuit as the generator and a classical WGAN-GP critic to synthesize rare attack flows. The synthetic data is intended to improve minority-class detection by downstream classical IDS classifiers while keeping the benign false-positive rate stable.

The governing design is **`CQAI_QWGAN_IDS_TDD_v1.pdf`** (`CQAI-DDD-QWGAN-IDS-001`, version 1.0). This repository is an implementation workspace; the design document remains authoritative when implementation details differ.

> **Current status:** this repository contains the initial NSL-KDD FR-1/FR-2 prototype and the first tested FR-3 QWGAN-GP core (quantum generator, critic, gradient penalty, alternating optimizer, diagnostics, and checkpoints). It does **not** yet constitute a completed FR-3 experiment, an FR-4–FR-8 implementation, or a validated quantum-advantage result.

## Why this project exists

Network-flow datasets are severely imbalanced. A classifier can achieve high headline accuracy while missing the rare attacks that matter most operationally. This project tests whether a compact quantum generator can produce useful minority-class samples without sacrificing statistical fidelity or increasing false alarms.

The intended production design follows five principles:

1. **Fidelity before novelty:** synthetic traffic must be statistically and semantically valid.
2. **Classical controls and fallbacks:** quantum results must be compared with strong classical baselines.
3. **NISQ realism:** circuits, depth, qubit counts, shots, runtime, and cost remain bounded.
4. **Reproducibility:** every reported metric must be traceable to immutable data, configuration, seed, and artifact hashes.
5. **Security by construction:** sensitive telemetry and model artifacts require encryption, least privilege, isolation, and auditability.

## Reference architecture

```text
NSL-KDD / UNSW-NB15 / CIC-IDS2017
                  │
                  ▼
       schema validation + cleaning
                  │
                  ▼
 train-only transforms → latent features → angles in [0, π]
                  │
                  ▼
      per-attack-class hybrid QWGAN-GP
  PennyLane quantum generator + classical critic
                  │
                  ▼
       decode + class-conditional fidelity gate
                  │
          accepted samples only
                  ▼
 RF / XGBoost / DNN augmentation evaluation
                  │
                  ▼
 registered classical model → streaming IDS inference
```

Quantum generation is strictly offline. The live scoring path contains only registered transforms and a classical classifier; no quantum circuit is permitted in the latency-critical inference path.

## Functional requirements

| ID | Requirement | Repository status |
|---|---|---|
| FR-1 | Ingest NSL-KDD, UNSW-NB15, and CIC-IDS2017 into a unified, schema-validated representation | Partial; NSL-KDD prototype present |
| FR-2 | Reduce features to a qubit-matched latent space, encode to `[0, π]`, and persist a documented decode path | Partial; NSL-KDD PCA/angle prototype present |
| FR-3 | Train a per-attack-class PennyLane quantum generator with a classical WGAN-GP critic | In progress; tested CPU core present, production data runner pending |
| FR-4 | Generate configurable minority samples and quarantine them until all fidelity gates pass | Not implemented |
| FR-5 | Evaluate RF, XGBoost, DNN, and a separately reported quantum-kernel SVM track | Not implemented |
| FR-6 | Run four controlled augmentation arms across exactly three seeds | Not implemented |
| FR-7 | Deploy the registered classical classifier to a low-latency streaming path | Not implemented |
| FR-8 | Persist complete artifact lineage, hashes, configuration, device, runtime, and cost metadata | Partial prototype artifacts only |

## Mandatory design constraints

### Data and encoding

- Fit selectors, encoders, scalers, and reducers on the training partition only.
- Keep validation and test partitions real, immutable, and free of synthetic samples.
- Use 8–12 latent dimensions/qubits; the default design uses 10.
- Map normalized continuous values with `theta = πx`, producing angles in `[0, π]`.
- Validate finite values, ordered latent columns, canonical labels, split membership, decoder order, units, and artifact hashes at startup.
- Treat PCA/feature-selection decoding as lossy and autoencoder reconstruction as approximate; do not claim exact recovery of dropped features.

### Hybrid WGAN-GP baseline

- One generator per rare attack class is the baseline; class conditioning is a later optimization.
- Use data re-uploading at every variational layer, hardware-efficient rotations, a CNOT ring, and local Pauli-Z expectations.
- Use 3–6 variational layers; the default is 4.
- The critic returns one unbounded scalar, uses LeakyReLU, and must not use BatchNorm or a final sigmoid.
- Canonical defaults: `lambda_gp=10`, `n_critic=5`, Adam `lr=1e-4`, `betas=(0.0, 0.9)`, batch size 64–256.
- Production simulation targets `lightning.gpu` with adjoint differentiation. CPU tests use `lightning.qubit` or `default.qubit`.

Canonical losses:

```text
critic    = mean(D(fake)) - mean(D(real)) + lambda_gp * GP
generator = -mean(D(fake))
GP        = mean((||∇x̂ D(x̂)||₂ - 1)²)
```

### Fidelity gate

Synthetic data must remain quarantined until every mandatory class-conditional gate passes:

- held-out classifier two-sample test (C2ST) AUC;
- per-feature Wasserstein-1 distance;
- per-feature Kolmogorov-Smirnov statistic;
- diversity and mode coverage;
- domain validity after decoding.

The hard C2ST threshold is **AUC ≤ 0.65**. W1, KS, diversity, and validity tolerances must be defined in versioned configuration before any pass/fail claim. Failed batches must never enter classifier training.

## Evaluation protocol

All arms use identical real splits, classifier search policy, evaluation code, and exactly three centrally configured seeds.

| Arm | Training data |
|---|---|
| A | Real data only |
| B | Real + SMOTE/ADASYN |
| C | Real + classical WGAN-GP synthetic data |
| D | Real + QWGAN synthetic data |

Report per-class and macro Precision, Recall, F1, ROC-AUC, PR-AUC, benign false-positive rate, and confusion matrices. Results must include mean ± standard deviation, confidence intervals, and a paired significance test from seed-level results.

The primary success criterion is:

- at least **5 percentage points absolute minority macro-F1 lift** for arm D over arm A;
- arm D must also exceed arms B and C;
- benign FPR increase must be **≤ 0.5 percentage points**;
- results must be stable across all three seeds.

A null result is valid. If QWGAN does not outperform the classical WGAN-GP control, the result must be reported plainly and the classical fallback retained. Quantum-kernel SVM experiments are a separate research track and are not evidence for QWGAN augmentation benefit.

## Repository layout

```text
.
├── README.md
└── QWGAN_IDS/
    ├── artifacts/       # fitted encoder/scaler/PCA artifacts
    ├── data/            # processed NSL-KDD outputs (Git LFS)
    ├── datasets/        # KDDTrain+.txt and KDDTest+.txt (Git LFS)
    ├── notebooks/       # exploratory FR-1/FR-2 walkthroughs
    ├── src/             # loader, preprocessing, encoding, selection, quantum checks
    ├── requirements.txt
    └── run_pipeline.py
```

## Data files

`QWGAN_IDS/data/` currently contains:

```text
angles.npy
decoded_features.npy
feature_matrix.pkl
kdd_clean.csv
kdd_raw.csv
latent_features.npy
mi_ranking.csv
selected_feature_names.npy
```

Large data files are stored with Git LFS. Install LFS before cloning or pulling:

```bash
git lfs install
git clone https://github.com/Chinar-Quantum-AI-Ltd/KDD_dataset_analysis_QWGAN.git
cd KDD_dataset_analysis_QWGAN
git lfs pull
```

Do not treat the checked-in processed files as final benchmark evidence. The current prototype merges the official NSL-KDD train and test files before fitting preprocessing transforms. That behavior violates the final design requirement to fit learned transforms on training data only and to preserve an immutable held-out real test partition.

## Running the current NSL-KDD prototype

```bash
cd QWGAN_IDS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py --skip-download
```

Run selected stages with:

```bash
python run_pipeline.py --skip-download --stages load clean encode select angles verify
```

The current script is an exploratory FR-1/FR-2 pipeline. It is not the final leakage-safe, multi-dataset contract described by the design document.

## FR-3 QWGAN-GP core

The first FR-3 implementation slice lives under `QWGAN_IDS/cqai/qwgan/`. It deliberately does not consume the checked-in merged `data/angles.npy`; training requires an explicit train-only handoff:

```python
import numpy as np

from cqai.qwgan import QWGANConfig, QWGANTrainer, TrainingAngles

config = QWGANConfig(
    n_qubits=8,
    n_layers=3,
    backend="default.qubit",
    diff_method="backprop",
    seed=42,
)

training_data = TrainingAngles.from_array(
    np.load("path/to/train/angles.npy"),
    config=config,
    partition="train",
    attack_class="u2r",
    latent_columns=tuple(f"z{i}" for i in range(config.n_qubits)),
)

trainer = QWGANTrainer(config)
diagnostics = trainer.train_step(training_data)
trainer.save_checkpoint(
    "artifacts/run-id/checkpoint.pt",
    metadata={"run_id": "run-id", "attack_class": "u2r"},
)
```

Run the fast CPU tests from `QWGAN_IDS/`:

```bash
python -m unittest discover -s tests -v
```

This core is not yet the full FR-3 deliverable. A versioned FR-1/FR-2 split contract, per-class dataloader/epoch runner, immutable run manifest, checkpoint retention policy, and three-seed training evidence remain pending.

## Verification required for future implementation

Before experimental results are reported, verify that:

1. the exact real split is shared across all arms;
2. preprocessing is fitted on training data only;
3. no synthetic row appears outside training;
4. fidelity gates use held-out real minority samples;
5. all three seeds complete or failures are reported;
6. metrics trace back to run and artifact hashes;
7. statistical comparisons use seed-level results rather than pooled predictions.

Fast automated tests must cover quantum output shape/bounds/gradients, critic constraints, gradient penalty, alternating updates, deterministic replay, fidelity metrics and rejection behavior, split integrity, metrics, manifests, hashes, and serialization round-trips.

## Production targets

- Synthetic generation: offline only.
- Classical streaming inference: p99 **≤ 50 ms per flow** at representative load.
- Promotion: signed, versioned, fidelity-gated and evaluation-gated classifier artifacts only.
- Deployment: canary rollout with explicit rollback criteria.
- Security: payload minimization, TLS/KMS encryption, least-privilege IAM, private networking, audit logs, and tenant isolation.

## Documentation authority

Implementation and experiment decisions must follow this order:

1. the latest explicit project instruction;
2. repository collaboration and safety rules;
3. `CQAI_QWGAN_IDS_TDD_v1.pdf`;
4. versioned contracts and configuration;
5. existing reports, notebooks, and prototype code.

Do not silently reinterpret acceptance criteria, and do not describe prototype artifacts as completed production work.
