# Independent implementation review: quaternion radial-gauge experiment

Verdict: PASS

Reviewed outcome-blind on 2026-07-16. I did not run the harness `seal`, `audit`, or `run`
actions; did not construct an official seed/configuration; did not claim either attempt marker; and
did not inspect an official Phase-A or Phase-B result. At the chronology checkpoint
`2026-07-16T00:37:20Z`, the canonical seal, Phase-A marker, and Phase-B marker were absent. The
preregistration hash matched its frozen constant.

## Blocking and major findings

### 1. Blocking: the Phase-A marker cannot satisfy the Phase-B source-aggregate check

`attempt_marker()` stores `loaded_source_aggregate` returned by `loaded_source_hashes()`. That
aggregate covers the repository Python modules loaded in the current process plus four explicitly
added files. The seal's `source_aggregate`, however, is produced by `source_hashes()` over
`_sealed_paths()`, which contains every `src/rtgs/**/*.py` and `tests/**/*.py` file. These are
different hash domains.

`validate_phase_a_attempt_payload()` nevertheless requires
`marker["loaded_source_aggregate"] == seal["source_aggregate"]`. An outcome-free import probe found
38 paths in the marker-style loaded set and 75 paths in the sealed set; the focused test modules
alone demonstrate that the latter is a strict superset. Consequently, a Phase-A marker written by
this harness will be rejected during Phase-B authorization even when every bound file is unchanged.
This blocks the preregistered Phase-A-to-Phase-B route.

Relevant implementation locations are `loaded_source_hashes()` and
`verify_loaded_sources_against_seal()` at harness lines 500-520 and 698-711, marker construction at
2841-2867, and the incompatible equality at 3070-3083. The current toy test supplies equal placeholder
aggregates and therefore does not exercise the producer/consumer mismatch.

Required repair: define and bind the same explicit path set at marker creation and authorization,
or retain separate full-sealed and loaded-subset aggregates and validate each against its matching
domain. Add a test that creates a marker payload through the real aggregate producer and validates
that exact payload through the authorization-side checker.

### 2. Major: authorization and decision recomputation do not consistently derive summaries from
the serialized raw evidence

The preregistration requires an independent Phase-A validity/materiality recomputation from raw
evidence. Several decisive or validity-bearing values remain trusted summaries:

- `derive_phase_a_auc()` checks stored pooled SSE/count against stored pooled PSNR/AUC, but never
  verifies that pooled SSE/count equal the sums of the nine serialized per-view SSE/count records.
  A toy arm with deliberately contradictory per-view and pooled evidence was accepted.
- `_phase_a_invariants()` trusts serialized active-gradient/active-displacement counts, optimizer
  state hash equality, step identity/order fields, construction-equivalence summaries, and several
  stored validity summaries rather than deriving every applicable value from the serialized
  quaternion/gradient/checkpoint evidence. It also does not fully revalidate the frozen diagnostic
  selection and step schedule against their row-level records.
- `_phase_b_arm_metrics()` derives metrics from stored pooled counts and trusts stored validity
  booleans; it does not first require pooled counts to equal per-view sums. A toy checkpoint with
  contradictory pooled and per-view SSE/count evidence was accepted. Likewise,
  `recompute_phase_b_decision()` trusts the stored step-zero pass flag instead of reconstructing the
  invariant from the serialized step-zero checkpoints.

These gaps mean a self-consistently edited summary can survive the function described as the
independent recomputation, even though lower-level evidence disagrees. The strict JSON parser and
artifact hashes protect bytes after review; they do not establish semantic consistency before the
scientist-review hash is issued.

Required repair: recompute pooled Phase-A and Phase-B counts from per-view evidence; recompute every
available step count, norm, displacement, ratio, schedule position, and policy equation from raw
arrays; reconstruct step-zero equality from checkpoints; and compare every stored summary/flag to
the derived result. Add adversarial tests that alter one field at each evidence layer and require a
fail-closed rejection.

## Checks that passed

- Frozen seeds, physical train/held-out split, stage-1 and Depth-lifter configurations, diagnostic
  selection rule, perturbation, policies, optimizers, schedules, checkpoints, materiality thresholds,
  Phase-B utility/safety thresholds, and candidate preference formulas match the preregistration in
  the producing code.
- Covariance, row-Frobenius, SSE, norm, quantile, and checkpoint diagnostic reductions promote to or
  accumulate in float64 where frozen. Native forward/loss arithmetic remains float32 as specified.
- Phase A physically subsets to nine training views, removes `gt_gaussians`, drops the full-scene
  handle, and retains no evaluator capability. Phase B validates authorization before fresh scene
  construction and confines held-out cameras/GT to evaluation callbacks; Trainer receives only the
  nine-view training scene.
- `TrainConfig.quaternion_update_policy` defaults to `current`; non-current+density rejection occurs
  before parameter/optimizer construction; entry normalization occurs once after device/SH handling
  and before parameter construction; `q_old`, `q_star`, and policy application have the frozen
  ordering; policy application does not mutate Adam state; observer snapshots are detached clones.
- Default versus explicit `current` is bit-exact for fields and non-time history in the focused
  tests. A separate outcome-free toy run confirmed that
  `tangent_displacement_retraction` coexists with mixed-resolution `TrainStepControl` values while
  preserving the unit-norm invariant and step-control metadata.
- Seal digest construction, duplicate/non-finite JSON rejection, exclusive writes, fixed seal path,
  output-name guards, source-drift check around verification, and pre-marker output preflight are
  otherwise fail-closed in the inspected paths.

## Commands executed

All commands were outcome-free and CPU-only where rendering/training occurred.

```text
.venv/bin/python -m ruff check benchmarks/quaternion_gauge_ablation.py tests/test_quaternion_gauge.py tests/test_quaternion_gauge_ablation.py src/rtgs/optim/trainer.py tests/test_optim.py
.venv/bin/python -m ruff format --check benchmarks/quaternion_gauge_ablation.py tests/test_quaternion_gauge.py tests/test_quaternion_gauge_ablation.py src/rtgs/optim/trainer.py tests/test_optim.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_quaternion_gauge.py tests/test_quaternion_gauge_ablation.py tests/test_optim.py tests/test_multiscale_refinement.py::test_none_and_all_unit_controls_preserve_established_training_bit_exactly tests/test_multiscale_refinement.py::test_all_four_toy_arms_keep_the_same_sampled_view_schedule tests/test_multiscale_refinement.py::test_resolution_transition_keeps_optimizer_state_and_full_resolution_observers
git diff --check
```

Ruff check and format passed. Pytest passed with 4 CUDA skips. `git diff --check` passed. Additional
inline toy-only Python probes confirmed Trainer/`TrainStepControl` coexistence, the loaded-versus-sealed
aggregate mismatch, rejection of the resulting marker aggregate during authorization, and acceptance
of contradictory pooled/per-view evidence by the current Phase-A and Phase-B reducers.

## Reviewed source binding

The reviewed inputs had these SHA-256 values at the final pre-write snapshot:

```text
f1ba26d2520e6f78731b404babe0e091f2341d16ab5e30607b25ba32692c764e  benchmarks/results/20260716_quaternion_gauge_PREREG.md
5701e9fd593b02e024732afeb8bd99de09f143c09b673d95b43c6f803d246a90  benchmarks/quaternion_gauge_ablation.py
26105043c13453b7904c6ec8626cce2622f580485ffb335f99ec59a52a3a1d36  tests/test_quaternion_gauge.py
b9e1392b07207c8ec19b0d3cf646d0249c22ed8fcfbfa855f0d3a3da1202eae9  tests/test_quaternion_gauge_ablation.py
3bb73a2071ff3525c07c0d1a57387ecccb5b5f16a3cc18398091b2606752053f  src/rtgs/optim/trainer.py
1e8c3d7d532fa47f11e7766f88872ca714fe06b428948ae8098655802fcc4995  tests/test_optim.py
```

Do not create the implementation seal or consume an official attempt under this implementation.

---

## Independent post-repair re-review — 2026-07-16T03:45:44+02:00

Re-review verdict: FAIL

The preceding initial review is retained unchanged as the historical first-pass record; its exact
pre-append SHA-256 was
`77d2f9842fd6ee03ef791b652c18f16517f93d07505e5e070574f1c8fdccc318`. This appendix is an
outcome-blind review of the frozen repair snapshot. I again did not invoke `seal`, `audit`, or
`run`; did not construct an official seed/configuration; did not claim either attempt marker; and
did not inspect any official scientific result. The canonical seal and both canonical attempt
markers were absent at `2026-07-16T03:45:44+02:00`.

### Remaining blocking finding: full verification cannot reach a seal

The original unequal-aggregate producer/consumer bug was repaired in the harness, but the new real
producer/consumer regression test is process-global and fails under the exact mandatory full-suite
collection used by `create_seal()`.

`loaded_source_hashes()` scans every repository Python module currently present in `sys.modules`.
The repaired test at `tests/test_quaternion_gauge_ablation.py:166` assumes that this set is a strict
subset of `_sealed_paths()`. That assumption holds when the quaternion test module is collected by
itself, but not when pytest has already imported another benchmark harness through another test
module: `_sealed_paths()` includes this harness, all `src/rtgs/**/*.py`, and all `tests/**/*.py`, but
not unrelated `benchmarks/*.py` files.

A minimal ordinary co-collection reproduced the problem:

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_multiscale_refinement.py tests/test_quaternion_gauge_ablation.py::test_phase_a_attempt_payload_binds_real_full_and_loaded_hash_domains
```

The test failed because the loaded set additionally contained
`benchmarks/multiscale_refinement_ablation.py`. More importantly, the exact mandatory non-slow test
command also exited 1 with this as its sole failure:

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q -m 'not slow'
```

In the full collection, the unexpected loaded paths were:

```text
benchmarks/tum_rgbd_signed_attribution.py
benchmarks/kernel_support_taper_ablation.py
benchmarks/multiscale_refinement_ablation.py
benchmarks/stage1_weight_gauge_audit.py
benchmarks/tum_rgbd_oriented_validity.py
```

`run_verification()` executes that non-slow command and raises on any nonzero return code before
`create_seal()` can write a seal. The scientific harness may have a valid loaded subset in a fresh
direct process, but the repository's frozen admission path does not pass. This is therefore
blocking, not a cosmetic test defect.

Required repair: make the regression test obtain the actual marker-producer loaded domain in an
isolated fresh subprocess, or make the loaded domain explicit and independent of ambient pytest
collection state. Feed that real produced mapping into the authorization validator, retain the
adversarial aggregate/path/output mutations, and require the exact full non-slow command to pass.

### Closure audit of the two original findings

1. **Source-domain finding: repaired in the runtime protocol, but not yet sealable.**
   `attempt_marker()` now binds the seal's full `sealed_source_aggregate` separately from an exact
   `loaded_source_hashes` mapping and its canonical `loaded_source_aggregate`.
   `verify_loaded_sources_against_seal()` rejects unexpected or mismatched loaded files, while
   `validate_phase_a_attempt_payload()` checks the loaded mapping as a subset of the sealed mapping,
   recomputes its aggregate, requires the preregistration/review/harness/`pyproject.toml` entries,
   and validates exact input/output bindings. This removes the original incompatible aggregate
   equality. The remaining failure is the collection-dependent real-domain test described above.

2. **Raw-evidence recomputation finding: resolved.** Phase A now reconstructs serialized
   initialization and diagnostic fields, covariance, per-field hashes, selection eigenvalues and
   rank, target identity, perturbation, and schedule. Checkpoint reduction verifies the nine
   per-view SSE/count records before pooled PSNR/AUC. Per-step derivation reconstructs raw
   `q_old`, gradient, `q_star`, and `q_new` arrays; their hashes and norms; active-row counts;
   radial/tangent components; displacement, effective-LR and angular diagnostics; policy equations;
   and optimizer-state records. `_phase_a_invariants()` additionally replays fresh Adam from the
   serialized gradients and checks step order, optimizer state, arm trajectory, checkpoint
   continuity, schedules, construction equivalence, current normalized-copy equivalence, and
   canonical collapse. Seed and pooled decisions are then reduced from those derived values.

   Phase B validates the serialized held-out truth arrays and their hashes, then reconstructs each
   checkpoint's full-image and foreground SSE/count/PSNR, SSIM, depth error, IoU, support coverage,
   fields, covariance, raw quaternion norms, and hashes from serialized predictions and fields.
   Arm validation checks the exact configuration, 120-step schedule, arm identity, all
   `q_old`/`q_star`/`q_new` hashes, policy equations, unit-norm rules, step/checkpoint trajectory
   continuity, training histories, and validity summaries. Decision recomputation enforces exact
   training-hash schemas and aggregates, common raw/effective input hashes, and derives the
   step-zero invariant rather than trusting its stored flag. Adversarial tests reject changed
   pooled or per-view evidence, raw predictions, raw fields, scene extent, validity, quaternion
   policy output, stored step-zero status, common input, truth support, and arm identity.

No additional blocking or major mathematical, metric, configuration, common-input, training-hash,
quaternion-policy, or trajectory-continuity issue was found in this frozen snapshot. The Trainer
ordering remains faithful to the preregistration: candidate entry normalization occurs once before
parameter/optimizer construction; `q_old` is captured before quaternion Adam, `q_star` immediately
after Adam, and `q_new` after the selected policy; tangent displacement removes
`q_old * dot(q_old, q_star-q_old)` before normalization; and policy write-back does not alter Adam
state. Default/explicit `current` behavior remains exact in the focused contracts.

### Re-review commands and results

All commands were outcome-blind and CPU-only where training/rendering occurred.

```text
.venv/bin/python -m ruff check benchmarks/quaternion_gauge_ablation.py tests/test_quaternion_gauge.py tests/test_quaternion_gauge_ablation.py src/rtgs/optim/trainer.py tests/test_optim.py
.venv/bin/python -m ruff format --check benchmarks/quaternion_gauge_ablation.py tests/test_quaternion_gauge.py tests/test_quaternion_gauge_ablation.py src/rtgs/optim/trainer.py tests/test_optim.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_quaternion_gauge.py tests/test_quaternion_gauge_ablation.py tests/test_optim.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_multiscale_refinement.py tests/test_quaternion_gauge_ablation.py::test_phase_a_attempt_payload_binds_real_full_and_loaded_hash_domains
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q -m 'not slow'
git diff --check
```

Ruff check and format passed. The isolated quaternion/harness/trainer test set passed with four
expected CUDA skips. The co-collection and full non-slow commands failed only at the loaded-domain
assertion described above. `git diff --check` passed after this append.

### Frozen repaired source binding

```text
7426f166742203b907c992abc24c0d7503a0da7783eb59ccb2515c51e5735b2c  pyproject.toml
f1ba26d2520e6f78731b404babe0e091f2341d16ab5e30607b25ba32692c764e  benchmarks/results/20260716_quaternion_gauge_PREREG.md
fd58d01ade1dcd8582acd915b1eb4478df2fc52e105d2ede1b51079d68cdc747  benchmarks/quaternion_gauge_ablation.py
1585244237292bdb358abd06442daeb3360ba6152da2c088521d00a28633ff7b  tests/test_quaternion_gauge_ablation.py
26105043c13453b7904c6ec8626cce2622f580485ffb335f99ec59a52a3a1d36  tests/test_quaternion_gauge.py
3bb73a2071ff3525c07c0d1a57387ecccb5b5f16a3cc18398091b2606752053f  src/rtgs/optim/trainer.py
1e8c3d7d532fa47f11e7766f88872ca714fe06b428948ae8098655802fcc4995  tests/test_optim.py
```

Do not create the implementation seal or consume an official attempt under this snapshot.

---

## Final independent import-order re-review — 2026-07-16T03:52:38+02:00

Final re-review verdict: PASS

The two preceding FAIL reviews remain intact as append-only chronology for their respective source
snapshots. The top-level status is now updated to PASS for the frozen snapshot below because the
sole remaining blocker was repaired without changing the harness, protocol, Trainer, or scientific
logic. The exact pre-append review SHA-256 was
`7b851b3d67c1bcbf9d9c838f7e1ac879ce40fca203df1c952c50322d224cfbff`.

This final pass remained outcome-blind. I did not invoke `seal`, `audit`, or `run`; did not construct
an official seed/configuration; did not claim an attempt marker; and did not inspect an official
scientific result. The canonical seal and both canonical attempt markers were absent immediately
before this review update.

### Disposition of the last blocker

The repaired test now obtains the marker-style loaded-source domain in an actually fresh Python
subprocess. It launches `sys.executable -c` from the repository root, imports only
`benchmarks.quaternion_gauge_ablation`, invokes the real `loaded_source_hashes()`, and serializes
both the exact path-to-digest mapping and its aggregate to the parent test process. The parent then:

- recomputes and checks the canonical loaded aggregate;
- checks that the child-produced loaded mapping is a proper subset of the real full sealed mapping
  and that the two domain aggregates differ;
- injects that real child-produced mapping into the real `attempt_marker()` producer;
- passes the produced payload through the real `validate_phase_a_attempt_payload()` consumer; and
- retains the fail-closed aggregate-tamper, mandatory-path-removal, and output-binding-tamper cases.

Only the seal lookup is replaced by a toy callback so that the test can exercise the once-only
marker constructor without creating an official seal. Marker construction, serialization, loaded
mapping fields, input/output bindings, and authorization-side validation remain the production
functions. Because collection-state benchmark imports remain in the parent pytest process and do
not enter the fresh child, the test now measures the intended direct harness runtime rather than
ambient test collection order.

An independent fresh-process probe observed 38 loaded paths and 75 full sealed paths, verified every
loaded digest against its sealed counterpart, found all four mandatory paths, recomputed the loaded
aggregate, and confirmed a proper subset with distinct domain aggregates. The prior adversarial
co-collection command now passes. Most importantly, the exact mandatory full non-slow suite exits
zero. The source-domain protocol repair and the raw-evidence recomputation repair documented above
therefore both stand; no blocking or major finding remains.

### Final commands and results

All commands were outcome-blind and CPU-only where training/rendering occurred.

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -c 'from benchmarks import quaternion_gauge_ablation as g; loaded, la = g.loaded_source_hashes(); full, fa = g.source_hashes(); mandatory = {str(g.PREREGISTRATION), str(g.IMPLEMENTATION_REVIEW), str(g.HARNESS_PATH), "pyproject.toml"}; assert la == g.canonical_json_hash(loaded); assert set(loaded) < set(full); assert mandatory <= set(loaded); assert all(full[p] == d for p, d in loaded.items()); print({"loaded": len(loaded), "sealed": len(full), "proper_subset": True, "aggregates_distinct": la != fa, "mandatory_present": True})'
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_multiscale_refinement.py tests/test_quaternion_gauge_ablation.py::test_phase_a_attempt_payload_binds_real_full_and_loaded_hash_domains
.venv/bin/python -m ruff check benchmarks/quaternion_gauge_ablation.py tests/test_quaternion_gauge.py tests/test_quaternion_gauge_ablation.py src/rtgs/optim/trainer.py tests/test_optim.py
.venv/bin/python -m ruff format --check benchmarks/quaternion_gauge_ablation.py tests/test_quaternion_gauge.py tests/test_quaternion_gauge_ablation.py src/rtgs/optim/trainer.py tests/test_optim.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q -m 'not slow'
git diff --check
.venv/bin/python -c 'from benchmarks import quaternion_gauge_ablation as g; print(g.verify_implementation_review())'
```

The fresh probe passed and reported `loaded=38`, `sealed=75`, proper subset true, distinct aggregates
true, and every mandatory path present. The 20-test adversarial co-collection passed. Ruff check and
format passed. The exact full non-slow suite reached 100% and exited 0, with six expected CUDA skips.
`git diff --check` passed, and `verify_implementation_review()` recognized this document's exact
top-level PASS line.

### Final reviewed source binding

```text
7426f166742203b907c992abc24c0d7503a0da7783eb59ccb2515c51e5735b2c  pyproject.toml
f1ba26d2520e6f78731b404babe0e091f2341d16ab5e30607b25ba32692c764e  benchmarks/results/20260716_quaternion_gauge_PREREG.md
fd58d01ade1dcd8582acd915b1eb4478df2fc52e105d2ede1b51079d68cdc747  benchmarks/quaternion_gauge_ablation.py
e8c33135be51c56a5335d0e410b63f8bc5c3ea13020e799ce66279a9d905456a  tests/test_quaternion_gauge_ablation.py
26105043c13453b7904c6ec8626cce2622f580485ffb335f99ec59a52a3a1d36  tests/test_quaternion_gauge.py
3bb73a2071ff3525c07c0d1a57387ecccb5b5f16a3cc18398091b2606752053f  src/rtgs/optim/trainer.py
1e8c3d7d532fa47f11e7766f88872ca714fe06b428948ae8098655802fcc4995  tests/test_optim.py
```

This implementation is cleared for the repository's seal step. This review does not itself create
a seal, authorize Phase A, or predict any scientific outcome.
