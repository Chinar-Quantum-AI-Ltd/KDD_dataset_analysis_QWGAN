# CQAI QWGAN-IDS

Hybrid quantum-classical data augmentation for minority-class network intrusion detection.

CQAI QWGAN-IDS uses a PennyLane parameterized quantum circuit as the generator and a classical WGAN-GP critic to synthesize rare attack flows. The synthetic data is intended to improve minority-class detection by downstream classical IDS classifiers while keeping the benign false-positive rate stable.

The governing design is **`CQAI_QWGAN_IDS_TDD_v1.pdf`** (`CQAI-DDD-QWGAN-IDS-001`, version 1.0). This repository is an implementation workspace; the design document remains authoritative when implementation details differ.

> **Current status:** this repository contains the initial NSL-KDD FR-1/FR-2 prototype, a completed FR-3 implementation (leakage-safe train-only contract, hybrid QWGAN-GP core, per-class runner with diagnostics, checkpoints, hashed manifests, and a reported three-seed campaign that is stable across seeds), and a completed FR-4 synthesis and fidelity gate. **The reported FR-4 run released nothing:** all 18 batches were quarantined, `r2l` failing the C2ST, Wasserstein, KS and coverage criteria by margins the null baseline rules out as threshold artefacts. The current generators do not produce usable synthetic data, so no sample is cleared for FR-5. FR-5–FR-8 are not implemented and there is no validated quantum-advantage result.

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
| FR-3 | Train a per-attack-class PennyLane quantum generator with a classical WGAN-GP critic | Complete on NSL-KDD: train-only contract, per-class runner, diagnostics, checkpoints, manifests, and a reported three-seed campaign on `u2r` and `r2l` reported stable across seeds |
| FR-4 | Generate configurable minority samples and quarantine them until all fidelity gates pass | Complete on NSL-KDD: ratio-swept synthesis, five-signal null-calibrated gate, separate accepted/quarantine manifests. Reported run released **nothing** — the FR-3 generators fail the gate |
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
    │   ├── contracts/   # built train-only contracts (generated, Git-ignored)
    │   ├── runs/        # immutable FR-3 run outputs (generated, Git-ignored)
    │   └── synthesis/   # immutable FR-4 gated pools (generated, Git-ignored)
    ├── cqai/            # FR-3 data/qwgan, FR-4 fidelity/synthesis
    ├── configs/         # versioned experiment configuration
    ├── data/            # processed NSL-KDD outputs (Git LFS)
    ├── datasets/        # KDDTrain+.txt and KDDTest+.txt (Git LFS)
    ├── notebooks/       # exploratory FR-1/FR-2 walkthroughs
    ├── src/             # loader, preprocessing, encoding, selection, quantum checks
    ├── tests/           # fast CPU tests (tests/fr3, tests/fr4, shared fixtures)
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

## FR-3 hybrid QWGAN-GP

> **Known defect — two campaigns, nothing released.** The v1 generators had
> every qubit's output mean pinned at π/2; the cause was the latent range
> interacting with data re-uploading, and `QWGANConfig.latent_scale` fixes it
> (default stays 1.0, so no reported run changes meaning). The v2 campaign with
> that fix trained properly — the Wasserstein estimate fell instead of rising —
> and improved four of five gate criteria, clearing coverage outright. **The
> C2ST AUC stayed at ~1.0 and all 18 batches were quarantined again.** The
> remaining failure concentrates in the `serror_rate` family, where the decoded
> real data is nearly constant and the generator is not; the evidence points
> upstream at MinMax over heavy-tailed PCA components. All of it is measured in
> [`docs/fr3-generator-diagnosis.md`](QWGAN_IDS/docs/fr3-generator-diagnosis.md).
> No synthetic sample is available to FR-5.

FR-3 lives under `QWGAN_IDS/cqai/`:

```text
cqai/data/nslkdd.py     leakage-safe train-only angle contract (FR-1/FR-2 adapter)
cqai/qwgan/generator.py PennyLane data-reuploading generator, batched QNode
cqai/qwgan/critic.py    classical WGAN-GP critic
cqai/qwgan/losses.py    canonical interpolation gradient penalty
cqai/qwgan/trainer.py   alternating optimizer, diagnostics, checkpoints
cqai/qwgan/runner.py    per-class epoch runner, monitors, FR-8 run manifest
cqai/qwgan/report.py    cross-seed stability verdict over a finished run
cqai/qwgan/cli.py       config-driven entry point
configs/                versioned experiment configurations
```

### The train-only contract

FR-3 deliberately refuses the checked-in `data/angles.npy`: that file was
produced by merging `KDDTrain+.txt` with `KDDTest+.txt` and fitting every
transform on the merge, which is evaluation leakage.

`cqai/data/nslkdd.py` rebuilds the same FR-2 chain — one-hot + `log1p` →
RobustScaler → mutual-information top-k → PCA → MinMax → `× π` — with every
transform fitted on the training partition alone. It reads `KDDTrain+.txt` and
nothing else; there is no parameter for a test file, so it cannot fit on one.
`KDDTest+.txt` therefore stays an immutable held-out real test set for FR-5/FR-6.

The contract emits `angles_{train,val}.npy`, per-row family labels, the fitted
transform artifacts, and a `contract.json` carrying the source hash, the
resolved spec, per-class counts, and a SHA-256 for every artifact.

On the real `KDDTrain+.txt` (125 973 rows, no duplicates) it produces:

| Partition | normal | dos | probe | r2l | u2r | total |
|---|---|---|---|---|---|---|
| train | 53 875 | 36 744 | 9 327 | 797 | 42 | 100 785 |
| val (held out for FR-4) | 13 468 | 9 183 | 2 329 | 198 | 10 | 25 188 |

`u2r` has 42 training rows against a design batch size of 64. The runner clamps
the batch to the class size rather than padding or resampling, because
fabricating density in the rarest class is exactly the failure this project
exists to avoid.

### Running FR-3

```bash
cd QWGAN_IDS
python -m cqai.qwgan.cli --experiment configs/fr3_nslkdd.yaml --dry-run
python -m cqai.qwgan.cli --experiment configs/fr3_nslkdd.yaml
python -m cqai.qwgan.cli --experiment configs/fr3_nslkdd_u2r_long.yaml
```

Two versioned configs ship. `fr3_nslkdd.yaml` is the shared 30-epoch schedule
over `u2r` and `r2l`. `fr3_nslkdd_u2r_long.yaml` is identical except for the
schedule, and exists because an epoch is not a comparable unit across these two
classes: `r2l` has 797 training rows and gets 13 generator updates per epoch,
`u2r` has 42 and gets exactly one.

The contract is built on first use and reused afterwards; a config whose spec
disagrees with an existing contract is rejected rather than silently mixed into
one lineage. Each run writes an immutable `artifacts/runs/<run_id>/` containing:

- `diagnostics.jsonl` — one record per training step: critic/generator loss,
  Wasserstein estimate, gradient penalty, critic gradient norm, generator
  gradient variance, diversity ratio, circuit depth and gate count, wall time,
  device, and cost estimate;
- `checkpoints/<class>/seed<n>/epoch*.pt` — generator, critic, both optimizers,
  RNG state, config, and lineage metadata;
- `manifest.json` — run ID, UTC timestamps, dataset ID/version/hash, schema
  version, parent contract and its hashes, code commit, environment and package
  versions, resolved config, all seeds, device/qubits/layers/shots/runtime/cost,
  per-run results, and a SHA-256 for every output file;
- `stability.json` / `stability.md` — the cross-seed verdict (see below).

Two monitors run continuously and are recorded per class and seed:
**barren plateau** (median generator gradient variance below threshold) and
**mode collapse** (generated spread as a fraction of the real batch's spread).

### The cross-seed stability verdict

A manifest states what each seed did; it does not state whether the seeds
*agree*. `cqai/qwgan/report.py` derives that judgement from a finished run and
writes it beside the manifest, recording the parent run ID and the SHA-256 of
the manifest it was computed from. It never rewrites the run, so regenerating a
report cannot alter the evidence:

```bash
python -m cqai.qwgan.report artifacts/runs/<run_id>
```

A class is reported stable only when all of the following hold: three seeds were
run; no seed raised the barren-plateau or mode-collapse flag; the minimum median
gradient variance and diversity ratio clear the reporting thresholds; and the
final Wasserstein estimates do not disagree across seeds. Disagreement requires
the spread to be large *both* relative to the mean and in absolute terms — three
seeds converging on ~0 have a huge relative spread and a negligible real one, and
flagging that would punish the best case.

The reporting thresholds are versioned inside `stability.json` and are
independent of the monitor thresholds a run was configured with, so a permissive
run config cannot launder a dead gradient signal past the report. `cli.py` exits
`2` on an unstable run, so an unusable campaign cannot be mistaken for a
successful one.

**A stable verdict is not a claim of convergence.** The report also records the
per-epoch Wasserstein trajectory and its first-to-last trend, but never gates on
them: a still-rising estimate means the critic is outpacing the generator, which
is a convergence signal, not a disagreement between seeds. Conflating the two in
one pass/fail number would hide whichever problem the other explains away.

### Reported runs

Both runs used the contract at `artifacts/contracts/nslkdd-train-only-v1`
(source `KDDTrain+.txt`, 125 973 rows), 10 qubits, 4 layers, `default.qubit`
with backprop, `lambda_gp=10`, `n_critic=5`, Adam `1e-4` / `(0.0, 0.9)`, and
seeds `13, 42, 1337`. Run artefacts are Git-ignored; the run IDs, manifest
hashes and configs below are what makes them reproducible.

| Run | Config | Class | Schedule | Final W (mean ± std) | Diversity (min) | Verdict |
|---|---|---|---|---|---|---|
| `fr3-nslkdd-full-001` | `fr3_nslkdd.yaml` | `r2l` (797 rows) | 30 epochs × 13 steps | 1.9227 ± 0.0331 | 0.320 | stable |
| `fr3-nslkdd-full-001` | `fr3_nslkdd.yaml` | `u2r` (42 rows) | 30 epochs × 1 step | 1.7938 ± 0.0980 | 0.201 | stable |
| `fr3-nslkdd-u2r-long-001` | `fr3_nslkdd_u2r_long.yaml` | `u2r` (42 rows) | 300 epochs × 1 step | 2.2794 ± 0.0143 | 0.201 | stable |

Total 25.9 min and 14.4 min on CPU. No seed in either run raised the
barren-plateau or mode-collapse flag; the minimum median generator gradient
variance was `1.9e-05` and `3.4e-05` respectively, five orders of magnitude
above the plateau threshold.

The Wasserstein estimate flattens in both classes — `r2l` by roughly epoch 10
(1.9414 → 1.9689 from epoch 10 to 30) and `u2r` only under the long schedule
(2.2444 at epoch 150 → 2.2794 at epoch 300, a 0.7 % change over the last 100
epochs). Under the shared 30-epoch schedule `u2r` was still climbing at the end
of training, which is what the second config exists to correct.

**Read these numbers carefully.** A flattened Wasserstein estimate means
training became reproducible and stopped moving, not that the generator matched
the real distribution: the estimate plateaus at a large positive value, so the
critic can still separate real from generated. Likewise, a diversity ratio of
0.20–0.32 clears the mode-collapse threshold while saying plainly that the
generated spread is a fraction of the real batch's. Whether these samples are
good enough for anything is precisely the question the FR-4 fidelity gate exists
to answer, and it does not exist yet.

### Library use

```python
from cqai.data import ContractSpec, build_train_contract
from cqai.qwgan import QWGANConfig, TrainingPlan, run_training

contract = build_train_contract(
    "datasets/KDDTrain+.txt",
    "artifacts/contracts/nslkdd-train-only-v1",
    spec=ContractSpec(n_qubits=10),
)
result = run_training(
    contract,
    config=QWGANConfig(n_qubits=10, n_layers=4),
    plan=TrainingPlan(attack_classes=("u2r", "r2l"), seeds=(13, 42, 1337)),
    output_dir="artifacts/runs",
)
```

### Tests

```bash
cd QWGAN_IDS
python -m unittest discover -s tests -t . -v
```

38 fast CPU tests, roughly 70 seconds, offline, and independent of Git LFS: they
run against a tiny generated NSL-KDD fixture rather than the real dataset.

### What FR-3 still does not have

- **No evidence that the generated samples are any good.** The campaign shows
  training is reproducible and stable across seeds; it does not show statistical
  fidelity, and the plateaued Wasserstein estimate suggests the critic still
  separates real from generated. That judgement belongs to the FR-4 gate.
- Only `u2r` and `r2l` were trained. `normal`, `dos` and `probe` are majority or
  near-majority classes and are not augmentation targets.
- No hyperparameter search. Layer count, learning rate and critic width are the
  TDD defaults; nothing here establishes them as good choices for this data.
- `lightning.gpu` with adjoint differentiation is the production simulation
  target in the TDD. Every reported number comes from `default.qubit` on CPU.
- The contract's decode path is **approximate**. PCA and top-k selection both
  discard information, so `decode_angles` recovers the selected features only,
  not the full 41-column NSL-KDD record.
- Only NSL-KDD is covered. UNSW-NB15 and CIC-IDS2017 are FR-1 work.

## FR-4 synthesis and fidelity gate

FR-3 shows the generators train reproducibly. It says nothing about whether
their samples are any good. FR-4 is the requirement that decides, and until a
batch passes its gate no synthetic sample is cleared for any downstream use.

```text
cqai/fidelity/metrics.py   per-feature W-1 and KS, cross-validated C2ST, coverage
cqai/fidelity/domain.py    NSL-KDD schema rules over the decoded feature space
cqai/fidelity/gate.py      thresholds -> pass / fail / insufficient_evidence
cqai/synthesis/generate.py checkpoint -> chunked sampling -> decode
cqai/synthesis/runner.py   per (class, seed, ratio): generate, gate, route, record
cqai/synthesis/cli.py      config-driven entry point
configs/fr4_nslkdd.yaml    versioned thresholds and volumes
```

`fidelity/` judges samples and `synthesis/` makes them. They are separate
packages so the gate can never be quietly tuned to whatever the current
generator happens to produce.

### The three verdicts

| Verdict | Meaning | Released? |
|---|---|---|
| `pass` | every mandatory criterion cleared its versioned band | yes |
| `fail` | at least one did not; `reasons` names which | no |
| `insufficient_evidence` | the held-out real reference is too small to support a claim either way | no |

The third verdict exists because NSL-KDD `u2r` has **10 held-out real rows**. A
C2ST AUC computed against ten rows cannot certify anything, and returning `pass`
there would be worse than returning nothing, so the gate fails closed. `fail`
and `insufficient_evidence` route identically to quarantine; they are
distinguished so a report can tell "measured and bad" from "could not measure".

### What the gate measures

All five signals the TDD mandates, class-conditionally, on the **decoded**
representation rather than on angles:

- **C2ST AUC** — a RandomForest trained to separate real from synthetic, over
  stratified 5-fold CV. The synthetic side is subsampled to the real count,
  because the pools differ by two orders of magnitude and imbalance inflates AUC
  on its own. TDD hard threshold `<= 0.65`; `0.5` means indistinguishable.
- **Per-feature Wasserstein-1**, reported raw and normalised by the real
  column's robust spread. One band cannot serve `src_bytes` (thousands) and
  `same_srv_rate` (bounded by 1), so the gate applies its threshold to the
  normalised values and keeps the raw ones for sanity-checking.
- **Per-feature Kolmogorov-Smirnov** — scale-free, so it catches a shape
  mismatch that a small absolute W-1 on a narrow feature would understate.
- **Coverage** — the fraction of real points whose nearest synthetic neighbour
  falls inside that point's own k-th nearest *real* neighbour radius. This is
  the mode-coverage question FR-3's diversity ratio cannot answer: a generator
  emitting one point repeatedly can still show a plausible spread ratio.
- **Domain validity** — an explicit NSL-KDD rule table (non-negative byte
  counters, connection counters capped at 511 and per-host at 255, rates in
  `[0, 1]`, indicators within tolerance of `{0, 1}`, mutually exclusive one-hot
  flags), reported with a per-rule violation breakdown so a rejection is
  actionable. The table is written out rather than derived from
  `src.preprocessing.BINARY_COLS`, which despite its name contains count columns
  and would license any value in `{0, 1}` for a counter.

### Every criterion is calibrated against a null baseline

Each result carries a **null reference**: the same metrics computed by splitting
the held-out real sample in half and comparing it against itself. This is not a
decoration. Measured on the real contract, decoding genuine held-out NSL-KDD
rows through the fitted PCA and scalers yields a **domain validity of 0.0** —
the lossy decode cannot reproduce a schema-valid record even from real data —
and real-vs-real `r2l` scores a normalised W-1 of 14.6 and a KS of 0.25, both
outside the configured bands.

An absolute-only gate would therefore have charged the generator for three
failures the decode and the sample size produced on their own. So a criterion
fires only when the synthetic batch is **outside its configured band *and* worse
than real data of the same size**. Where no null exists — too few real rows to
halve — the absolute band decides alone, which keeps the gate strict rather than
letting a missing baseline wave a batch through. The baseline can never turn a
failure into a pass by itself; it can only withhold blame the generator did not
earn.

Every criterion records its `value`, `threshold`, `null` and `fired` flag in
`gate.json`, so each reason in a verdict can be checked rather than trusted.

### Running FR-4

```bash
cd QWGAN_IDS
python -m cqai.synthesis.cli --experiment configs/fr4_nslkdd.yaml --dry-run
python -m cqai.synthesis.cli --experiment configs/fr4_nslkdd.yaml
```

Volume is a target minority ratio, not a naive 1:1 balance: `round(ratio ×
majority_train_count) − real_train_count`, swept over 0.2 / 0.3 / 0.4 as the TDD
requires. Against the real contract that is 10 733–21 508 synthetic `u2r` rows
and 9 978–20 753 `r2l` rows per seed.

Each run writes an immutable `artifacts/synthesis/<run_id>/`:

```text
accepted/<class>/seed<n>/ratio<r>/    angles.npy, decoded.npy, gate.json
quarantine/<class>/seed<n>/ratio<r>/  angles.npy, decoded.npy, gate.json
accepted_manifest.json                per-batch lineage for released samples
quarantine_manifest.json              the same for everything held back
manifest.json                         FR-8 lineage + a SHA-256 for every output
```

Accepted and quarantined manifests are separate files. Each entry carries the
generator checkpoint and its hash, class, seed, ratio, count, the resolved
thresholds, the full gate result, and the decode artefact hashes — the same
angles decoded by different transforms are different records, so the transform
versions travel with the samples. The CLI exits `2` when nothing is released.

### Reported gate result — nothing passed

Run `fr4-nslkdd-002`, 11.5 min CPU, gating the FR-3 campaign's generators
(`fr3-nslkdd-u2r-long-001` for `u2r`, `fr3-nslkdd-full-001` for `r2l`) across
three seeds and three ratios — **18 batches, 0 accepted, 18 quarantined.**

`r2l` is the meaningful test: 198 held-out real rows is enough to certify
against. It fails decisively, and every fired criterion is worse than what real
data of the same size scores:

| Criterion | Synthetic | Threshold | Null (real vs real) | Fired |
|---|---:|---:|---:|:--:|
| C2ST AUC | 1.0000 | ≤ 0.65 | 0.4789 | yes |
| Normalised W-1 (max) | 490.06 | ≤ 0.50 | 14.60 | yes |
| KS (max) | 0.9943 | ≤ 0.20 | 0.2525 | yes |
| Coverage | 0.1566 | ≥ 0.50 | 0.9697 | yes |
| Domain validity | 0.0043 | ≥ 0.95 | 0.0000 | no — real data scores worse |

A C2ST AUC of 1.0 against a null of 0.48 means the discriminator separates real
from synthetic **perfectly**, on every fold. Coverage of 0.16 against a null of
0.97 means the samples reach a sixth of the real modes. These are not marginal
misses.

`u2r` returns `insufficient_evidence` on all nine batches: 10 held-out rows
cannot certify anything either way. Its recorded metrics point the same
direction, but the gate refuses to make a claim from them.

**The conclusion is that the FR-3 generators do not produce usable synthetic
data, and no sample is released to FR-5.** This is a real result, not a
configuration problem: it holds across three seeds, three volumes and both
classes, and the null baseline rules out the thresholds being the cause. FR-4
is complete in the sense that matters — the gate works, and what it found is
that the generator does not.

The domain row is the reason the null calibration exists. Decoding genuine
held-out rows through the fitted PCA and scalers yields 0.0 validity, so an
absolute-only gate would have reported `domain_invalid` as a fifth generator
failure. It is a decode limitation, and the gate now says so.

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
