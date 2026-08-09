# Why the FR-3 generators failed the FR-4 gate

The FR-4 fidelity gate released nothing: 18 batches, all quarantined, with `r2l`
scoring a C2ST AUC of 1.0 against a null of 0.48. This document records why, and
what the evidence says the fix is.

The short version: **the generator's per-qubit output mean is structurally
pinned at π/2, and no amount of training moves it.** The cause is the latent
range interacting with data re-uploading, not the optimizer, not the critic, and
not a barren plateau.

## What the generator actually produced

Comparing the trained `r2l` generator (seed 13, epoch 30) against the real train
angles, in the **angle domain, before any decoding**:

| Per qubit | Real train | Generated |
|---|---|---|
| mean | 0.739, 1.994, 0.411, 1.570, 1.163, 1.300, 1.169, 1.578, 1.339, 1.549 | 1.576, 1.575, 1.581, 1.583, 1.578, 1.578, 1.579, 1.579, 1.579, 1.576 |
| std (mean over qubits) | 0.285 | 0.097 |
| range | 0.020 – 3.140 | 0.748 – 2.357 |

Every generated mean sits at ≈1.578 ≈ π/2 — the exact centre of the output
range, corresponding to a Pauli-Z expectation of zero. The real means span
0.411–1.994. The generator learned none of them.

This also means the FR-4 result was never in doubt: a distribution collapsed
onto one point in every dimension cannot pass a two-sample test.

## Step 1 — the weights never moved

| | weight rms | steps |
|---|---:|---:|
| initialization | 0.0102 | — |
| `r2l` after 30 epochs | 0.0099 | 390 |
| `u2r` after 300 epochs | 0.0087 | 300 |

With Adam at `lr=1e-4`, the largest distance any single weight can travel in 390
steps is ≈0.039 rad. The weights started at ≈0.01 rad and a hardware-efficient
circuit needs rotations of order 1 rad to express anything. The generators are
still, to a good approximation, at their small-angle initialization.

This re-reads the FR-3 stability report. "Stable across three seeds" was true,
and the report's own caveat — *a stable verdict is not a claim of convergence* —
was the right one. The runs were stable because nothing was moving. The rising,
then flattening Wasserstein estimate was the critic (which gets five updates per
generator update) learning to separate a static distribution and then saturating.

## Step 2 — but training harder does not fix it

Raising the learning rate lets the weights move, and barely helps. 120 adversarial
steps on `r2l`:

| lr | weight rms | mean L2 error | std ratio |
|---:|---:|---:|---:|
| 1e-4 | 0.009 | 1.652 | 0.341 |
| 1e-3 | 0.021 | 1.648 | 0.338 |
| 1e-2 | 0.243 | 1.561 | 0.351 |
| 5e-2 | 0.735 | 1.402 | 0.391 |

Removing the critic entirely and fitting the generator directly to the real
angles with an MMD loss — no adversarial dynamics, 400 steps, aggressive learning
rates — reaches the same wall:

| lr | weight rms | mean L2 error | std ratio |
|---:|---:|---:|---:|
| 1e-2 | 0.408 | 1.513 | 0.419 |
| 5e-2 | 0.474 | 1.533 | 0.447 |

The weights genuinely moved to order 0.5 rad and the per-qubit means still would
not leave π/2. Direct supervised-style fitting cannot do it, so this is not an
optimization problem.

## Step 3 — the weights have no leverage over the mean

On an **untrained** circuit, scaling the weights by hand at the full latent
range:

| weight scale | weight rms | output mean span | std |
|---:|---:|---|---:|
| ×1 | 0.010 | 1.5737 – 1.5865 | 0.0945 |
| ×10 | 0.102 | 1.5730 – 1.5836 | 0.0851 |
| ×50 | 0.510 | 1.5701 – 1.5748 | 0.0533 |

A fifty-fold increase in weight magnitude does not move the mean at all, and
*reduces* the spread. The trainable parameters simply do not control this
quantity.

## Step 4 — the latent range does

Same untrained circuit, varying the latent range instead:

| latent range | output mean span | std |
|---|---|---:|
| [0, π] | 1.5737 – 1.5865 | 0.0945 |
| [0, 0.5π] | 1.5649 – 1.5922 | 0.0956 |
| [0, 0.25π] | 1.5387 – 1.6709 | 0.1349 |
| [0, 0.1π] | 2.4940 – 2.5949 | 0.1473 |
| [0, 0.05π] | 2.9511 – 2.9875 | 0.0480 |

### The mechanism

`RY(noise)` is applied at **every** layer — that is what data re-uploading
means. With `noise ~ U[0, π]` and four layers, the accumulated rotation on each
wire spans up to `4π`, wrapping the Bloch sphere twice. Averaged over the
latent, `⟨Z⟩ → 0`, which maps to exactly π/2 in the angle domain. The trainable
`Rot` gates are fixed per sample while the latent varies, so they cannot survive
that average.

Narrowing the latent stops the wrap-around, and then the weights matter again.
250 direct-MMD steps at `lr=1e-2`:

| latent range | mean L2 error | std ratio | generated mean span |
|---|---:|---:|---|
| [0, π] | 1.5148 | 0.420 | 1.430 – 1.617 |
| [0, 0.5π] | 0.5182 | **0.940** | 0.710 – 2.003 |
| [0, 0.25π] | **0.2951** | 0.681 | **0.387 – 2.046** |
| [0, 0.1π] | 0.5640 | 0.263 | 0.307 – 2.159 |

Real: means span 0.411 – 1.994, std 0.285.

At a quarter range the mean error drops five-fold and the per-qubit means cover
the real span almost exactly. At half range the *spread* is nearly perfect
(ratio 0.94). Neither is a tuned result — both come from 250 steps of a crude
probe — but they bracket a working region that the full range does not contain.

## What was changed

`QWGANConfig.latent_scale`, drawing the latent from `[0, latent_scale * π]`.

**The default stays 1.0.** Every number reported so far was produced at the full
range, and shipping a new default would silently change what the existing FR-3
manifests mean. A config has to opt in, and the value travels into every run
manifest.

Nothing else was altered: the circuit structure, the WGAN-GP objective, and the
TDD's hyperparameters are untouched. Changing the latent range is a deviation
worth stating plainly — the TDD fixes the latent dimension at the qubit count
but does not fix its range — and it is recorded here rather than applied
quietly.

## What the fix actually bought — and the wall behind it

The v2 campaign (`fr3-nslkdd-v2-001`, `latent_scale 0.25`, `lr 1e-3`, 200-epoch
ceiling with early stopping, 2.6 h) was run and gated (`fr4-nslkdd-v2-001`).
The latent fix works, and it is not enough.

Training behaved the way it should have the first time. The Wasserstein estimate
**fell** from 1.44 to 0.70 for `r2l` instead of rising to 1.95 and flattening,
and the weight rms reached 0.51 instead of staying at its 0.01 initialization.
`u2r` early-stopped at 160 / 130 / 90 epochs; `r2l` used its full 200 and was
still improving.

Gate metrics for `r2l`, same thresholds, same held-out rows:

| Criterion | v1 | v2 | Threshold | Null |
|---|---:|---:|---:|---:|
| C2ST AUC | 1.0000 | 0.9995 | ≤ 0.65 | 0.4789 |
| Normalised W-1 (max) | 490.06 | **78.86** | ≤ 0.50 | 14.60 |
| KS (max) | 0.9943 | **0.6304** | ≤ 0.20 | 0.2525 |
| Coverage | 0.1566 | **0.5556** | ≥ 0.50 | 0.9697 |

Coverage cleared its threshold outright — the generator now reaches the real
modes, which it previously did not. W-1 improved six-fold and KS by a third.
**The C2ST did not move.** Still 18 batches quarantined, still nothing released.

### Why the C2ST stays at 1.0

The failure concentrates in one family of features: `serror_rate`,
`srv_serror_rate`, `dst_host_srv_serror_rate`, `diff_srv_rate`, `flag_S0` — the
worst normalised W-1 (78.9) and the worst KS (0.63) are all here.

Decoded, the real `r2l` values in these columns are almost constant while the
synthetic ones are not:

| Column | real IQR | real std | synthetic std |
|---|---:|---:|---:|
| `serror_rate` | 0.0068 | 0.051 | 0.095 |
| `srv_serror_rate` | 0.0082 | 0.051 | 0.096 |
| `dst_host_srv_serror_rate` | 0.0068 | 0.048 | 0.095 |

A classifier only has to threshold `|serror_rate| > 0.05` to separate the two
sets perfectly. That is the whole C2ST.

It is **not** an excess-spread problem in the latent. In the angle domain the v2
generator is narrower than real on every qubit (std ratio 0.37–0.81). It is a
*shape* mismatch. The real angle components are sharply peaked with heavy tails
— on some qubits the interquartile range is 5 % of π while the standard
deviation is over four times that IQR — so the bulk sits in a spike and a few
outliers claim the rest of the range. A smooth circuit output spread evenly
across a comparable range reproduces the standard deviation and none of the
spike, and after the inverse PCA that difference lands squarely in the
low-variance columns.

The upstream cause is the FR-2 transform chain: MinMax over heavy-tailed PCA
components maps the bulk of the data into a narrow sub-interval while rare
outliers define the endpoints. The generator is being asked to hit a near-delta
in several dimensions at once. A quantile or rank-based transform in place of
MinMax would give the generator a target it can actually represent, and that is
the next lever — a change to the contract, not to the circuit.

## What this does not establish

- **Nothing has passed the gate.** v2 improved four of five criteria and cleared
  one; it released zero batches. No synthetic sample is available to FR-5.
- The transform-chain hypothesis above is a **hypothesis**. It explains the
  measurements, but no quantile-transformed contract has been built or trained
  against, so it is not established.
- `r2l` was still improving at its 200-epoch ceiling. How much of the remaining
  gap is schedule and how much is representational is unmeasured.
- The probes used an MMD objective the production trainer does not use. It was
  chosen to isolate expressivity from adversarial dynamics, not to propose a new
  loss.
- Only `r2l` was examined in detail. `u2r` shows the same failures, but its ten
  held-out rows cannot support any conclusion either way — it returns
  `insufficient_evidence` in both campaigns and will keep doing so at this
  split.
- No hyperparameter search was run. `latent_scale 0.25` and `lr 1e-3` come from
  a four-point probe and a single-seed pilot, not an optimisation.
