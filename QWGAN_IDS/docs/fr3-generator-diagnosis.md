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

## What this does not establish

- **No retrained generator, no new gate result.** The probes above are
  diagnostics on `r2l` seed 13, not a campaign. Whether a narrowed latent
  actually clears the FR-4 gate is unmeasured, and matching two marginal moments
  is a much weaker claim than passing a C2ST.
- The learning rate is still almost certainly too low and the update count too
  small; those are real problems that this change does not address.
- The probe used an MMD objective the production trainer does not use. It was
  chosen to isolate expressivity from adversarial dynamics, not to propose a new
  loss.
- Only `r2l` was examined in detail. `u2r` shows the same pinned means, but its
  ten held-out rows cannot support any conclusion either way.
