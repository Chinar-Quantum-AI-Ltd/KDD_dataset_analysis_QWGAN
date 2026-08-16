# TombakNet Audit Remediation Plan

## Scope

This plan is based only on `CQAI_QuantumWGAN_TombakNet_AuditReport.pdf`, assessed against commit `9f04bd0` on 12 August 2026.

In scope:

- SEC-01, SEC-02, SEC-03, SEC-04, SEC-05, SEC-07, and SEC-08.
- PERF-01 and the reported p99 latency regression.
- QML-01 and QML-02.
- The report's tracked-bytecode housekeeping item.

Out of scope:

- New FR-6 ablation functionality.
- A general FR-8 manifest redesign beyond what is required to enforce safe artifact loading.
- Training a replacement production classifier merely to make a benchmark pass.
- Changes to QWGAN architecture or physics; the audit validated the circuit implementation.
- QPU, cloud, full-data, or paid runs.

## Delivery order

### AUD-01 — Close unsafe deserialization boundaries

Findings: SEC-01, SEC-02, SEC-03.

Targets:

- `src/transform_bundle.py`
- `src/inference_transform.py`
- `cqai/classifiers/random_forest.py`
- `cqai/classifiers/xgboost_model.py`
- `cqai/classifiers/dnn.py`
- `cqai/classifiers/quantum_svm.py`
- Other `joblib.load` call sites found by repository search.
- `cqai/lineage/manifest.py` only where necessary for pre-load verification.

Actions:

1. Replace `allow_pickle=True` for the string-only selected-feature array with `allow_pickle=False`; migrate to JSON if the existing array dtype is incompatible.
2. Introduce one reusable verified-load helper.
3. Require an expected SHA-256 from a trusted registration record before invoking `joblib.load`.
4. Reject missing or mismatched digests before deserialization.
5. Keep post-load type checks as defense in depth, not as the security boundary.
6. Document that an unsigned, attacker-writable manifest is not a trust anchor.
7. Plan safe-format migration (`skops`, ONNX, or `safetensors`) for artifacts crossing a trust boundary.

Acceptance:

- Tampered artifact never reaches `joblib.load`.
- Missing hash fails closed.
- Feature-name loading does not enable pickle.
- Every production load path uses the same verified loader.
- Tests use a deserializer spy to prove zero calls on integrity failure.

### AUD-02 — Make serving fail closed

Findings: SEC-05, SEC-07, SEC-08.

Targets:

- `cqai/serving/pipeline.py`
- Existing transform-bundle validation entry points.
- Focused serving tests.

Actions:

1. Make the registered transformer mandatory for production construction.
2. Validate input type, schema, order, finiteness, and batch shape at the serving boundary.
3. Validate the transformed classifier matrix before prediction.
4. Reject unknown categorical values instead of silently accepting the encoder's all-zero block.
5. Add a configurable maximum request size before transformation or list materialization.
6. Preserve a pure-classical path: registered transform bundle followed by registered classical classifier.

Acceptance:

- NaN, Inf, missing columns, extra columns, extreme sentinels, unknown categories, and oversized batches are rejected.
- Malformed flows cannot receive a benign prediction.
- No `fit`, QNode, PennyLane, quantum kernel, QWGAN, WGAN, or generator call is reachable from live scoring.
- Valid registered inputs retain the expected predictions.

Ownership note: changes to the FR-1/FR-2 raw schema or fitted transforms require an adapter or explicit cross-team contract decision. The remediation must not silently refit teammate-owned preprocessing.

### AUD-03 — Remove duplicate forest traversal and benchmark honestly

Finding: PERF-01.

Targets:

- `cqai/serving/pipeline.py`
- `cqai/serving/latency.py`
- A focused local benchmark script and result directory, if missing.

Actions:

1. Call `predict_proba` once.
2. Derive labels using `classes_[probabilities.argmax(axis=1)]`.
3. Prove label equivalence against the former `predict` result in tests.
4. Load artifacts and fixture before timing.
5. Warm up before collecting measurements.
6. Use `time.perf_counter_ns` and retain all raw measurements.
7. Reproduce the report's representative request sizes: 1, 16, 64, 256, and 1024 flows, unless serving enforces a lower production batch cap.
8. Report request p50, p95, and p99 plus throughput/amortized cost as a separate diagnostic.

SLA definition:

- Primary: end-to-end request latency for transform plus one classical classifier traversal.
- Target: request p99 <= 50 ms for every supported production batch size.
- Amortized milliseconds per flow must not be used to hide a request exceeding 50 ms.

Acceptance:

- `predict_proba` is called once and `predict` is not called in the live path.
- Loading, warm-up, training, generation, and result serialization are outside timed regions.
- Raw and summary measurements are persisted without outlier deletion.
- An SLA failure remains a failure and produces an inspectable result.
- No PASS is reported without a real registered classical classifier and actual execution.

### AUD-04 — Stabilize QML and classifier tests

Findings: QML-01, QML-02.

Targets:

- `tests/fr4/test_generate.py`
- `cqai/classifiers/xgboost_model.py`
- Existing CI/test configuration.

Actions:

1. Replace bitwise equality in the chunking test with `assert_allclose` using `atol=1e-12` and an explicitly documented relative tolerance.
2. Keep the test sensitive to meaningful behavioral drift.
3. Replace XGBoost `n_jobs=-1` with an explicit bounded default.
4. Run the combined Torch, PennyLane, and XGBoost suite with `OMP_NUM_THREADS=1` in CI.
5. Do not use `KMP_DUPLICATE_LIB_OK`.

Acceptance:

- Chunked output passes within the physics-appropriate tolerance.
- A deliberately larger difference still fails the test.
- The full combined suite completes without SIGSEGV.
- Test totals demonstrate that no modules were silently skipped by interpreter termination.

### AUD-05 — Enforce dependency and artifact compatibility

Finding: SEC-04 and the report's unpinned-stack observations.

Targets:

- `requirements.txt`
- A repository-appropriate lock or constraints file.
- Transform/model registration metadata and load-time checks.
- Tracked `__pycache__` and `.pyc` files.

Actions:

1. Select and document one tested Python/NumPy/scikit-learn/joblib/XGBoost/Torch/PennyLane stack.
2. Generate a reproducible lock or constraints file rather than relying on unbounded lower constraints.
3. Record estimator fitting versions in artifact metadata.
4. Reject incompatible runtime/artifact versions before inference.
5. Rebuild or migrate incompatible persisted estimators under the supported environment.
6. Remove tracked bytecode and retain ignore rules.

Acceptance:

- Clean installation resolves to documented versions.
- Compatible artifacts load without `InconsistentVersionWarning`.
- Incompatible artifacts fail before scoring.
- Artifact metadata identifies fitting-library versions.
- `git ls-files` reports no tracked Python bytecode.

## Verification sequence

1. Run focused unit tests for the changed boundary.
2. Run transform-bundle and classifier serialization round trips.
3. Run serving malformed-input and zero-quantum tests.
4. Run the combined QML/XGBoost suite with the controlled OpenMP setting.
5. Run the complete local test suite.
6. Execute the latency benchmark only after a real registered classical classifier is available.
7. Compare measured request p99 against 50 ms without normalizing away request delay.

## Reporting rules

- Report each audit finding as `PASS`, `FAIL`, or `NOT VERIFIED`.
- Do not claim SEC-01 fully closed if the manifest itself is not protected by a trust mechanism.
- Do not claim production latency compliance using a newly trained benchmark-only classifier.
- Preserve raw latency observations and report the environment and artifact hashes.
- Record any cross-team FR-1/FR-2 contract impact before changing it.

## Implementation status

- AUD-01: implemented and focused tests passed.
- AUD-02: implemented and focused tests passed.
- AUD-03: implementation and benchmark harness complete; registered-artifact run
  remains not executed.
- AUD-04: code and CI safeguards implemented; the complete 182-test combined
  Torch/PennyLane/XGBoost suite passes with `OMP_NUM_THREADS=1` and no segfault.
- AUD-05: lock and fail-closed compatibility controls implemented; legacy serving
  artifacts require rebuild under the locked environment.

See `docs/audit-remediation-report.md` for the evidence and PASS/NOT VERIFIED
matrix. No production SLA result is claimed without an actual benchmark.
