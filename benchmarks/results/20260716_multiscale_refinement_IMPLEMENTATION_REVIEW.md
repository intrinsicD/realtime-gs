# Multiscale fixed-topology refinement implementation review

Verdict: PASS

## Independent re-review disposition — 2026-07-16

The implementation was repaired without running the seal action, the official `run` action,
seeds `3,4,5` under the official configuration, an official schedule probe, or any scientific
arm. I independently reviewed the repaired snapshot line by line against the complete frozen
preregistration and reran the focused outcome-free checks. No blocking or major finding remains,
so sealing is now authorized subject to the repository-wide verification gate. This verdict does
not authorize a scientific claim, default change, or interpretation before the official artifact
receives the separately required results audit.

The initial `FAIL` review and its four findings are retained verbatim below as chronology. Their
dispositions are:

1. **Paired reductions — resolved.** `_candidate_decision` now constructs and serializes literal
   per-seed `candidate-full` vectors for alpha-IoU and foreground coverage and applies
   `statistics.fmean` to those vectors. The boundary test uses finite values for which the frozen
   paired reduction passes `-0.02` while subtraction of separately rounded means fails, proving
   that the required order—not merely an algebraically equivalent real-valued formula—is used.
2. **Verification/seal snapshot — resolved.** `capture_seal_snapshot` binds the complete sealed
   path list, per-file hashes, source aggregate, preregistration, implementation review, and
   default seam both before and after the five verification commands. Seal creation refuses any
   difference. A mocked verification that edits a bound source exercises this refusal.
3. **Seal and marker tamper resistance — resolved.** The CLI and loader require the exact frozen
   seal path. Seal loading validates the current complete snapshot, the pre/post snapshot digest,
   verification commands and captured output hashes, environment, and source manifest. The
   once-only marker is rebound from both raw bytes and canonical payload immediately after
   exclusive creation and is re-read after all arms, after seal/source verification, during
   result validation, immediately before returning, and immediately before serialization. Tests
   independently reject a copied seal, altered source/verification evidence, duplicate marker,
   and modified marker payload.
4. **Runtime pyramid/exposure evidence — resolved.** Every official training image now compares
   float32 area pooling, cast to float64, against a direct float64 2x2 sum with the frozen
   tolerances and hashes both values. Runtime accounting separately validates and serializes
   optimization render/loss pixels, the common `82944` native-evaluation pixels per arm, `27648`
   held-out callback pixels per arm, and the once-per-seed `6912` manual step-zero pixels; only
   optimization rendering enters the `0.625` ratio.

The repaired implementation continues to preserve the exact `step_controls=None` baseline and
the persistent optimizer/RNG path. The newly coexisting quaternion seam is dormant under the
frozen default `quaternion_update_policy="current"`; the normal optimizer loop remains the exact
default expression. Training receives only the physical nine-view subset with GT fields removed;
held-out tensors are constructed only after fitting, Carve initialization, schedules, and source
bindings are fixed, and are consumed solely by detached full-resolution checkpoint evaluation.
All decisions are recomputed from hashed per-view evidence; timing remains descriptive and absent
from every gate.

Re-reviewed snapshot:

- preregistration: `b4c17da489a6e66950ec15ecda78dd7ddb063d65055e25bb4f76df6e8cca0a59`;
- harness: `7a70d315e7e8e5b1c0934e18f75b6c85452313ac39c549642afdbe0f190aa580`;
- focused tests: `4a9beb35de2133709813a0263370cfb5c9af300f026aa7e671b573491016b1ea`;
- Trainer: `3bb73a2071ff3525c07c0d1a57387ecccb5b5f16a3cc18398091b2606752053f`;
- optimizer export: `1196f76c9386d808b88a0940f562b29a85b3598e182ba7e997ebf2f769e4d53a`.

Re-review commands:

```text
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m pytest -q \
  tests/test_multiscale_refinement.py tests/test_optim.py
...............................ssss                                      [100%]
31 passed, 4 skipped

.venv/bin/python -m ruff check src/rtgs/optim/trainer.py \
  src/rtgs/optim/__init__.py benchmarks/multiscale_refinement_ablation.py \
  tests/test_multiscale_refinement.py
All checks passed!

.venv/bin/python -m ruff format --check src/rtgs/optim/trainer.py \
  src/rtgs/optim/__init__.py benchmarks/multiscale_refinement_ablation.py \
  tests/test_multiscale_refinement.py
4 files already formatted

git diff --check -- src/rtgs/optim/trainer.py src/rtgs/optim/__init__.py \
  benchmarks/multiscale_refinement_ablation.py tests/test_multiscale_refinement.py
```

## Scope

This was an independent, outcome-free review of the complete preregistration and the frozen
Trainer seam, harness, exports, and focused tests. I did not run the seal action, the official
`run` action, a fit or lift for seeds `3,4,5`, an official schedule probe, or an official arm. No
attempt marker, seal, result, or scientific metric was created.

Reviewed snapshot:

- `benchmarks/results/20260716_multiscale_refinement_PREREG.md`:
  `b4c17da489a6e66950ec15ecda78dd7ddb063d65055e25bb4f76df6e8cca0a59`
- `src/rtgs/optim/trainer.py`:
  `3bb73a2071ff3525c07c0d1a57387ecccb5b5f16a3cc18398091b2606752053f`
- `src/rtgs/optim/__init__.py`:
  `1196f76c9386d808b88a0940f562b29a85b3598e182ba7e997ebf2f769e4d53a`
- `benchmarks/multiscale_refinement_ablation.py`:
  `408367d5c83abbe0fb02a42c452cbf28bf7b0fdc5c0344a162b2366d525eec1a`
- `tests/test_multiscale_refinement.py`:
  `3c5a89080675aeea86423137481ca0202f93486e02bb430d2b74fdc9cebcf67b`

## Unresolved findings

### 1. Blocking — alpha-IoU and coverage gates do not use the frozen paired-seed reduction

The preregistration requires every across-seed mean delta to be the arithmetic mean of the three
paired seed deltas. `_candidate_decision` instead computes each of these two quantities as a
difference of separately rounded means:

```python
statistics.fmean(values(arm, "alpha_iou"))
- statistics.fmean(values("full", "alpha_iou"))

statistics.fmean(values(arm, "foreground_coverage"))
- statistics.fmean(values("full", "foreground_coverage"))
```

Those expressions are algebraically equivalent over exact reals but are not the frozen floating
reduction. For ordinary finite inputs, `fmean(arm)-fmean(full)` and
`fmean([arm_i-full_i])` can differ (an outcome-free probe found a difference of
`3.729655473350135e-17`). Either value can therefore land on a different side of the inclusive
`-0.02` gate. Build the per-seed signed alpha and coverage delta lists first, serialize them, and
use `statistics.fmean` on those lists for the reported means and decisions. Add exact boundary
tests that distinguish the two reduction orders.

### 2. Blocking — seal creation does not prove that verification tested the sealed source snapshot

`create_seal` runs the five repository verification commands and only afterwards computes
`source_hashes()`. It never hashes the complete sealed path set before verification or compares a
before/after manifest. A source or test can change while verification is running, after its
relevant check has completed, and the seal will then bind the changed file even though that
snapshot did not pass the recorded gate. This is especially material in a shared worktree.

Hash the complete sealed manifest before verification, rerun the hash afterwards, and refuse seal
creation unless paths, per-file digests, aggregate, preregistration/default-seam evidence, and
implementation-review binding are unchanged. Add an outcome-free test that mutates a temporary
bound source during a mocked verification and proves seal refusal.

### 3. Blocking — the preregistered seal/marker tamper checks and marker end-binding are absent

The focused-test contract explicitly requires tampering with the seal and marker to be rejected.
The current final test checks source-manifest drift and duplicate marker creation, but it does not
tamper with and revalidate a seal, and it does not alter an already-created marker. More
importantly, the official run never re-reads or hashes the marker after `claim_attempt`; it carries
only the originally returned `{path, sha256}` record into the result. The marker can be modified
during the long run without invalidating the in-process result.

Bind the exact marker payload/digest immediately after exclusive creation, re-read and compare it
before decision acceptance and artifact serialization, and make marker drift fail closed. Add
focused tests that independently alter a sealed source/verification field and a claimed marker and
show both paths refuse acceptance. The official `run` action should also require the fixed
preregistered seal path rather than accepting an equivalent seal copied to an arbitrary path.

### 4. Major — the frozen runtime evidence omits required pyramid and exposure cross-checks

`verify_training_pyramid_invariants` computes each official 24x24 image with
`area_downsample_2x`, but it does not perform or serialize the separately required float64 direct
2x2-sum comparison for the nine training images. The analytic unit test is useful but is not the
runtime assertion required for the official tensors.

The harness recomputes optimization exposure and checks nonbaseline Trainer metadata, but it does
not explicitly compute, serialize, and verify the common native-evaluation exposure of `82944`
pixels per arm. It also does not separately report the held-out observer exposure and the manual
step-zero exposure, despite the preregistration requiring those counts to remain outside the
optimization ratio. Add these exact runtime records and assertions; they must remain descriptive
and outside every quality gate.

## Verified behavior that does not clear the findings

- `step_controls=None` retains the established branch; the all-unit seam is bit-exact in focused
  tests, and control metadata is additive.
- `TrainStepControl` is frozen, schedules and pixel counts are correct, controls consume no RNG,
  and the four toy arms reproduce the same sampled-view schedule.
- Camera scaling, pixel-center rays, area-pool dtype/device, persistent parameter/Adam identity,
  full-resolution native evaluation, callback isolation/timing exclusion, fixed topology, and
  degree-zero behavior are covered by passing toy tests.
- The held-out evaluator uses raw unclamped float32 fields, the frozen crop/SSIM convention,
  float64-before-subtraction sums, exact support masks, per-view arithmetic means, and the frozen
  trapezoidal AUC. The remaining decision issue is the paired reduction in Finding 1.
- Candidate schedules use one persistent `Trainer.train` call, while the baseline uses the exact
  `None` path. Held-out truth stays outside fitting, lifting, training scenes, schedules, and
  optimization losses.

## Commands

```text
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m pytest -q tests/test_multiscale_refinement.py
...............                                                          [100%]
15 passed

.venv/bin/python -m ruff check src/rtgs/optim/trainer.py \
  src/rtgs/optim/__init__.py benchmarks/multiscale_refinement_ablation.py \
  tests/test_multiscale_refinement.py
All checks passed!

.venv/bin/python -m ruff format --check src/rtgs/optim/trainer.py \
  src/rtgs/optim/__init__.py benchmarks/multiscale_refinement_ablation.py \
  tests/test_multiscale_refinement.py
4 files already formatted
```

The focused tests being green does not satisfy the frozen protocol while the four findings above
remain. Sealing and the official attempt are not authorized from this review.
