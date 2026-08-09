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

## The real bottleneck: the target is multimodal, and moments cannot reach it

Two further hypotheses were tested and **both were wrong**. They are recorded
because the pattern they form is the actual finding.

### Refuted: the transform squeezes the target

If MinMax over heavy-tailed PCA components were the problem, a quantile map
should help. It measurably improves the target's shape — bulk IQR 0.14π → 0.35π,
excess kurtosis +3.32 → +1.36, inverse round-trip error 3e-13 — and makes the
fit **worse**:

| Transform | mean L2 | std ratio | C2ST | coverage |
|---|---:|---:|---:|---:|
| MinMax | 0.4069 | 0.660 | 0.9990 | 0.1212 |
| quantile | 1.3786 | 0.361 | 1.0000 | 0.0303 |

The quantile target is 2.3× wider, and the circuit was already short on spread.
A prettier target it cannot reach is not an improvement.

### Refuted: entanglement crushes the spread

A CNOT ring after every layer does concentrate the local expectations, badly.
Measured on an untrained 10-qubit, 4-layer circuit at weight magnitudes matching
a trained model:

| CNOT ring on | std ⟨Z⟩ | mean span |
|---|---:|---:|
| every layer (TDD) | 0.0285 | 0.0567 |
| last layer only | 0.1110 | 0.3304 |
| no layer | 0.4408 | 1.1497 |
| **real `r2l` needs** | **0.181** | **1.008** |

Removing it transforms the marginals — std ratio 0.66 → **0.953**, mean error
0.407 → **0.125**, coverage 0.121 → 0.157. And the C2ST does not move:
0.9990 → 0.9978.

### What that pattern means

The C2ST is inert to every intervention tried:

| Intervention | C2ST |
|---|---|
| latent range (v1 → v2) | 1.0000 → 0.9995 |
| transform (MinMax → quantile) | 0.9990 → 1.0000 |
| entanglement (every layer → none) | 0.9990 → 0.9978 |

Because none of them addresses the actual gap. Three reference points locate it:

| | C2ST |
|---|---:|
| real train vs real val — **the achievable floor** | **0.5227** |
| a Gaussian with the real mean and full covariance | 0.9881 |
| the v2 generator | 0.9978 |

The target is reachable: real data scores 0.52 against itself. But a distribution
matching the real mean **and full covariance exactly** still scores 0.988. No
model that gets the first two moments right can pass this gate, and our generator
is already essentially at that ceiling.

The reason is that the real `r2l` angle distribution is strongly multimodal.
Fitting Gaussian mixtures to the 797 training rows, BIC falls monotonically and
is still falling at twelve components:

| components | 1 | 2 | 3 | 5 | 8 | 12 |
|---|---:|---:|---:|---:|---:|---:|
| BIC | −4 693 | −14 608 | −23 517 | −32 065 | −36 863 | −42 782 |

with mean \|skew\| 0.70 and mean excess kurtosis +3.32 across qubits. Discreteness
is not the explanation — all 797 rows are distinct in every dimension, with no
repeated values.

A smooth circuit driven by uniform latent noise produces one broad blob. The
target is a dozen-plus separated clusters. Every intervention so far moved the
blob's centre and width; none of them could give it modes.

### What that implies for the next attempt

- The latent has to carry mode information. A continuous uniform latent cannot
  induce multimodality through a smooth map; a discrete or mixed latent (a
  computational-basis mode selector, or per-class mixture components) can.
- Or the difficulty is upstream: 797 rows forming twelve-plus modes in ten
  dimensions is roughly sixty samples per mode, learned adversarially. That may
  simply be under-determined, in which case no generator architecture rescues it
  and the honest answer is that NSL-KDD `r2l` is too small at this split.
- Either way, further tuning of latent range, learning rate, entanglement or the
  transform is not worth spending compute on. Those levers are exhausted, and the
  measurements above say why.

## Classical controls: nobody passes, and the decode is why

The multimodality finding raised an obvious question — is this target learnable
by *anything*? Three classical controls were scored against the same held-out
`r2l` rows.

In the **angle domain**, before decoding:

| Model | C2ST | coverage | time |
|---|---:|---:|---:|
| real vs real — the floor | 0.5178 | — | — |
| GMM(120) | **0.6140** | 1.0000 | 17 s |
| GMM(30) | 0.6908 | 0.9949 | 28 s |
| classical WGAN-GP, 8 000 steps | 0.9623 | 0.7121 | 17 min |
| classical WGAN-GP, 2 000 steps | 0.9894 | 0.3636 | 6 min |
| quantum WGAN-GP (v2) | 0.9978 | 0.1566 | ~2.5 h |

Two things follow. The target **is** learnable — a Gaussian mixture reaches
0.614 in seventeen seconds. And the WGAN-GP setup itself, not the circuit, is
the larger handicap: a classical MLP generator with the identical critic, loss,
gradient penalty, `n_critic` and optimizer only reaches 0.962 after 8 000 steps.
The circuit then adds its own penalty on top, 0.998 against the MLP's 0.962.

### The correction that matters

Those numbers are measured in the angle domain. **The gate scores decoded
samples**, and the same models look very different there:

| | angle domain | decoded |
|---|---:|---:|
| floor (real vs real) | 0.5178 | 0.5909 |
| GMM(120) | 0.6140 | **0.9197** |

The floor rises modestly. The mixture's score nearly saturates. The decode is
not inflating everything equally — it specifically amplifies a model's error,
because the chain ends in `expm1`: a small angle-space deviation, pushed through
the inverse PCA and inverse RobustScaler and then exponentiated, becomes a large
one in the heavy-tailed byte and counter columns. Real data survives it because
it is real.

So **no model passes the gate**, classical mixtures included, and a large part
of the difficulty is the FR-2 transform chain rather than any generator. That is
also why the angle-domain improvements from the latent fix never showed up in
the gate: they were real, and the decode ate them.

This does not make the gate wrong. Anything downstream consumes decoded rows, so
decoded fidelity is what FR-5 would actually depend on. It does mean the
`log1p`/`expm1` step is a fidelity bottleneck in its own right, and it belongs
on the list of things to change before another generator campaign.

### A hole in our own gate

Chasing the mixture result exposed a missing criterion. GMM(120) fits about
7 800 parameters to 797 rows, and in the angle domain its samples sit **29 %
closer to the training rows than genuine held-out rows do** (median nearest-
neighbour distance 0.029 against the real 0.041). A model that echoes its
training set passes a two-sample test trivially and contributes nothing to an
augmented one — and none of the five mandated signals notices.

`novelty_ratio` and a `memorised_training_data` criterion now close that. The
measure is the median distance from synthetic samples to their nearest training
row, divided by the same statistic for real held-out rows: near 1 means as novel
as real data, near 0 means memorisation. It is optional and off unless a
training reference is supplied, because every previously reported gate result
was produced without it.

Its threshold, 0.8, is **not calibrated** — and the signal is space-dependent in
the same way everything else here is: GMM(120) scores 0.71 in the angle domain
but 0.95 decoded, so in the gate's own space it would not have fired. It is
recorded and enforced, but no reported number currently rests on it.

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
