# Prospective Protocol Review

- Task ID: `20260909_field_target_gradient_stage_frame00008`
- Protocol SHA-256: `2246a380965a67c7d487ebae11ba607c42829e115e82a788b5bfe51e50c57426`
- Reviewer: `Codex-protocol-reviewer`
- Verdict: `approved`
- Outcome Access: `none`
- Review date: `2026-09-10`

## Scope

Approve one development diagnostic under this exact protocol and source envelope. It measures
cached field-minus-photograph residuals on 22 training views and compares the two targets'
L1/SSIM gradients at nine unchanged, predeclared saved 3D states. No optimization, topology
change, heldout evaluation, checkpoint selection or new target acquisition is authorized.
The evidence may describe target residuals and within-state local gradient differences; it
does not identify the cause of prior reconstruction defects or establish a repair, physical
density recovery, strict field-only reconstruction, high-quality capability or performance gain.

This is final implementation approval, following the separately preserved design-only review.
The Driver must register this approval, initialize the development run, preserve exact source
and provenance, and complete required verification before launching the measurement worker.
The canonical run directory does not yet exist. Prior source-experiment exposure is known;
no outcomes of this new residual/gradient diagnostic were generated or inspected by this review.

## Exact reviewed source

The 118-file live source envelope independently matches aggregate SHA-256
`247d75336fe7f1d03563a245ea30895646d4c1c557b5b01ed3e3e08654e7d317`.
The task binds Python, CUDA, C++/header and JavaScript reconstruction/rendering sources,
the contract/bundle tools, task driver/helper, focused tests and package configuration.

| File | SHA-256 |
|---|---|
| `scripts/experiments/20260909_field_target_gradient_stage_frame00008.py` | `b70e5ae11ff17bcdc6117e156ca09d21cb1797f5c980eab10b674a0b07672eea` |
| `scripts/experiments/20260909_field_target_gradient_stage_frame00008_report.py` | `85fbfb5678f02a726d16c41e790bbfd9258aee5541713572c0cc9eb604b7077e` |
| `tests/test_field_target_gradient.py` | `2e339d903e8f6a5691a7569ded6a8429067d7e5a7107d88ff463d8f086731924` |
| `tests/test_field_target_gradient_report.py` | `74f247dfd03489012e9fcf7a0c0fd69c7f30e9618c468a6f5dfb2556b227cae2` |

Reviewed draft task bytes have SHA-256
`4fa3e9fc8e7735c59c1cd86d0feb68cd4268e4a9ec349e7af0dc80c742ce5d0e`.
Registering ready/review metadata changes task-file bytes under the contract's normal lifecycle;
the approved protocol digest and source envelope must remain unchanged, and the run lock binds
the final registered task bytes.

## Checks

- Read the complete driver, report helper and both focused test files, plus the relevant
  saved-model, camera, renderer, Trainer, SSIM, preview and report-contract APIs. The driver
  uses existing APIs without changing production reconstruction code or defaults.
- Independently rehashed all 76 unique cached input/state files and their recorded sizes:
  244,219,355 bytes, all matching. Their camera metadata covers the exact 22 training views.
  Hashing does not decode their arrays. The result worker verifies the exact allowlist before
  loading and at exit, rejects raw-dataset opens and unlisted reads from the source run, and
  rejects writes to that prior evidence. Masks are used only in residual strata.
- Administrative inherited raw-seal validation remains outside the restricted numerical
  worker. The Driver must retain its before/after validation receipts and repeat it at the
  bundle gate. These operations may hash heldout bytes but must not decode or score heldout
  sources. Worker source/cache receipts are a separate evidence class.
- Residuals are computed in float64 from the unchanged float32 caches. The square-radius-three
  boundary, interior and exterior partition is exhaustive and disjoint. Channel means use
  N pixels; RGB scalar means/MAE/MSE use 3N. Full-canvas contribution and absolute-error share
  use their own denominators. Empty and zero-error cases retain explicit nulls. Full-image
  reflect-padded box means precede region scoring; lowpass/highpass terms are not presented
  as an orthogonal error-energy decomposition.
- Every target and identity path receives fresh parameter leaves and an independent render
  graph. Autograd component extraction does not reuse accumulated leaf gradients. Masks,
  regularizers, optimizer steps and topology updates are absent. The loss uses unclamped
  renderer RGB, the fixed full-canvas 0.8 L1 + 0.2 DSSIM, and the existing tiled SSIM operator.
- Initial opacity reproduces the CUDA clamp/logit/sigmoid entry convention. Final opacity
  is used verbatim without a clamp or logit roundtrip. The effective-opacity gradient is
  multiplied by alpha*(1-alpha) to obtain the stated local logit-coordinate derivative;
  endpoint counts and zero mapped derivatives are explicit. Raw saved quaternions and
  row identity are preserved, with normalization only inside the renderer. Log scales and
  SH0/SHN are the stated coordinates; initial inactive SHN gradients are explicit zeros.
- L1 and DSSIM gradients are unweighted components; total gradients are their weighted sum.
  Synthetic direct-full-loss backward and independent logit-graph tests exercise this
  convention. Each state checks both targets through the actual routes with independent
  equal-RGB render/backward graphs at C0004. All component/group losses, norm/difference
  statistics and elementwise tolerance violations are retained. The frozen control tolerances
  are not adjusted after capture outcome access.
- Every state covers every training view in the declared order. Per-view statistics reduce
  float32 CUDA gradients in float64. Float64 gradient-of-view-mean arrays are saved and hashed
  separately from means of per-view scalar statistics. Null ratios/cosines follow the frozen
  norm threshold; different states/topologies and parameter-group units are not conflated.
  Saved model and effective-parameter identities, dimensions and state-unchanged checks are
  recorded. This is a local coordinate comparison, not recovery of raw optimizer state.
- State and total wall deadlines, CUDA-only execution, finite checks, control failure and
  no-overwrite/no-automatic-retry handling stop the diagnostic and preserve partial evidence.
  Python signal deadlines are verified synthetically; they are not a claim of hard preemption
  of an indefinitely blocked native GPU call. Such an infrastructure failure must remain
  recorded and must not trigger outcome-based retries or tolerance changes.
- The report validates completed/no-update execution, exact state/view coverage, all three
  components and six groups, finite absolute statistics, allowed null ratios, and detailed
  identity-control records. It recomputes state and nested headline reductions. Undefined
  nonprimary values display as undefined and have omitted numeric bars with explicit notes
  and retained counts. A wholly unavailable primary produces the prospectively declared
  failed-presentation receipt with raw evidence preserved, not an invented metric.
- Both canonical RESULT files are checked for conflict before publication writes. The completed
  receipt is written after result/report sources. Synthetic tests exercise JSON/Markdown
  collisions and full publication through the existing metric/history schema validators.
  History steps represent measured views, not training iterations. Repeated global stage
  intervals are labelled context and cannot be summed across series.
- Preview generation uses the selected unchanged saved models and cached training cameras/RGB;
  no raw scene loader or heldout source is involved. NPZ copies are exact, PLYs are display
  exports, and previews are display-clamped. This is not new fitting or numerical quality
  evidence. Report port 8765 and viewer port 8880 match the supported delivery route.

## Independently executed checks

```text
.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260909_field_target_gradient_stage_frame00008.json
.venv/bin/python -m pytest -q tests/test_field_target_gradient.py tests/test_field_target_gradient_report.py
.venv/bin/python -m pytest -q tests/test_field_target_gradient.py::test_cuda_rendered_components_opacity_and_identity
git diff --check
```

The focused suite passed 21 cases with the CUDA case skipped in the sandbox. The same CUDA
case then passed independently on the host GPU using generated synthetic tensors only. Checks
cover direct/component gradient agreement, opacity chain and endpoints, raw quaternion
preservation, inactive SH, unchanged states, NumPy-window residual parity, norm aggregation,
real installed file-open guard denials, signal timeout, environment schema, report completeness,
nulls and publication conflicts. The Driver's separate CPU/CUDA selftest JSON receipts were
also inspected; both report passing controls. No calibrated diagnostic or fitting run was
executed as a prelaunch check.

Read-only Python checks independently invoked `verify_source_binding` and rebuilt the complete
118-file aggregate, checked all 76 cache/state byte hashes, and confirmed that the new canonical
run directory remains absent. Existing full-repository verification, administrative data-seal
validation, live environment receipts and source preservation are Driver execution gates;
this review does not claim they have already been completed for the new run.

## Findings

The earlier design corrections fixed region denominators/nulls and the canonical report port.
Implementation review then required complete aggregate/control records, null-safe reporting,
conflict preflight and an explicit unavailable-primary failure policy. Synthetic parity
tolerances were frozen before capture outcomes. The last prelaunch amendment narrowed resource
wording to the implementation: CUDA process-wide peaks after warmup through diagnostics/previews,
host process-lifetime high-water mark, per-state/view wall intervals only, and worker wall ending
before publication. External administrative seal checks and shared report generation are excluded.
This supersedes the unlaunched `41dbd95a9146ee913345d27d5eb0e53458edf0b54272e708195466db0656c67a`
digest without changing the reviewed source envelope or scientific computation.

Equal-target numerical controls cover one predeclared view per state and are not a bound on
every view's CUDA variation. Three saved seeds remain one exposed capture; initial geometry is
shared and final topologies differ. Aggregate scalar summaries do not identify regional gradient
causes or an intervention. Shared-GPU timings and memory have no performance interpretation.

After execution, a distinct results-audit phase must check raw arrays, reductions, controls,
state/input identities, failure chronology and visual/report presentation before claims enter
docs/ARA. The failed RTGS-021 reconstruction prerequisites remain closed. No private data or
source context was uploaded to Claude, and no protected outcome was consumed in this review.

## Protected Actions Not Taken

No calibrated residual/gradient measurement, fitting, optimizer or topology update, heldout
decoding/evaluation, checkpoint selection, new target acquisition, prior-evidence modification,
or private upload was performed. Only synthetic tests, source inspection, metadata and byte-hash
checks were used. This review does not initialize or launch the protected run.

Before initialization, the contract rejected this review's original section headings. The
headings were corrected to the required canonical structure without changing the approval,
scientific content, protocol digest or source envelope. No run directory or protected outcome
existed during this formatting repair; the Driver preserves the rejected initialization log.
