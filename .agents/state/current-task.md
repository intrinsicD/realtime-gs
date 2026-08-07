# Current Task

## Title

Test GPS-Gaussian depth from fitted 2D Gaussian fields

## Task ID

RTGS-014

## Role Assignment

- Driver: Codex-gps-field-driver
- Reviewer: Codex-gps-protocol-reviewer
- Turn: human

## Mode

Implement

## Risk

Protected

## Maturity

- Target: Calibrated
- Reached: Pipeline-integrated

## Goal

Implement and execute the preregistered development killing test that renders fitted 2D Gaussian
fields into transient 1024-square proxies, applies the frozen GPS-Gaussian stereo-depth network,
samples metric depth at field-component centers, lifts/fuses those components into an exact-budget
3D initialization, and compares it with the strongest existing compact initializers.

## Motivation

GPS-Gaussian supplies a learned symmetric stereo prior that may resolve geometry missed by the
current deterministic Splat-SfM, compact-carve, and Beam paths. Its released implementation is
dense and pixel-wise, so it must be isolated as a geometry experiment and measured inside the
post-fit reconstruction boundary rather than treated as evidence for compact VRAM superiority.

## Success Criteria

- Register, data-seal, and obtain a distinct prospective review for one development-only
  experiment before implementing or executing outcome-bearing code.
- Add a CPU-first stereo-depth protocol, exact compact-field proxy renderer, pure-torch calibrated
  rectification, safe lazy GPS-Gaussian adapter, component-depth sampler, covariance lift, and
  confidence-weighted global fusion without adding source RGB or masks to reconstruction inputs.
- Load only a weights-only sanitized state dict whose source checkpoint, sanitized payload, and
  official GPS code revision are checksum-bound and reported.
- Compare the GPS field-proxy initializer, an exact field/camera-shuffled negative control, a
  same-pair/full-field deterministic compact-carve control, global compact carve, Splat-SfM, and
  Beam Fusion at the frozen 3,000-Gaussian count and identical 250-step compact refinement
  schedule over three paired seeds.
- Measure the complete field-load through proxy/depth/fusion/refinement/save boundary with NVML,
  CUDA allocated/reserved, host RSS, stage runtimes, input bytes, and output bytes; keep the dense
  adapter separate from the primary direct-compact VRAM arm.
- Preserve held-out cameras for reporting only, save initialization/final PLYs and previews,
  complete the v2 report/viewer lifecycle, run an independent results audit, and label any missing
  hardware or viewer evidence explicitly.
- Revise the draft VRAM experiment program so eager RGB, CPU-streamed RGB, decode-on-demand RGB,
  and matched sparse point-sampled RGB are named causal controls before any broad VRAM claim.
- Pass focused CPU tests, diff review, docs sync, ARA checks, and `./scripts/verify.sh`.

## Constraints

- Reconstruction inputs remain calibration plus fitted `.rtgsv` Gaussian fields. Source RGB,
  masks, PIL, `SceneData`, and the dense RGB trainer are forbidden inside compact workers.
- Dense proxy tensors and GPS inference are inside the measured reconstruction boundary; their
  cost cannot be moved upstream or described as sparse-throughout.
- The official external GPS repository remains outside the package and is imported only lazily;
  CPU-only import and tests must work without it, CUDA, SciPy, or its checkpoint.
- GPS scale/rotation/opacity maps are not used. The fitted 2D covariance and color are retained,
  and only depth plus depth uncertainty come from stereo.
- Frame 00008 is outcome-exposed development data. It cannot establish generalization,
  paper-strength RGB quality, physical geometry accuracy, or a compact VRAM advantage.
- Do not change production defaults, consume held-out fields during construction/refinement, or
  promote a claim without the distinct results audit and ARA proof boundary.
- The user has now authorized committing the completed scoped work, merging it into `main`, and
  pushing `main` to the configured remote after review and verification. The experiment itself is
  still initialized as a dirty-tree non-official development run because outcome production must
  precede that final commit.

## Non-Goals

- Training or fine-tuning GPS-Gaussian, using its per-pixel Gaussian parameter heads, or copying
  its renderer into this repository.
- Implementing the field-native learned stereo network before the dense adapter passes its frozen
  geometry gate; that is a contingent successor task.
- Claiming that transient dense proxies are images on disk, sparse tensors, or evidence of lower
  peak VRAM than a streamed/sampled RGB implementation.
- Replacing the direct compact, Splat-SfM, compact-carve, or Beam defaults from this single scene.

## Selected Skills

- `rtgs-core`
- `rtgs-task-workflow`
- `rtgs-experiment`
- `realtime-gs-results-audit`
- `rtgs-review`
- `rtgs-docs-sync`
- `rtgs-verify`

## Experiment Contract

experiments/tasks/20260804_gps_field_proxy_depth_stage_frame00008_gaussianimage.json

## Current Evidence

- GPS-Gaussian official code revision `0024776deee4824f270d4bb534a17ffd85f63cf2`
  predicts symmetric dense stereo disparity from two rectified 1024-square RGB tensors and has no
  published peak-VRAM comparison.
- The selected full-resolution GaussianImage bundle contains 26 calibrated 11,000-component
  fields. The frozen train/held-out split uses 23/3 views; the calibration-only pair closest to
  the paper's 45-degree regime is `C0001`/`C0022` at about 45.026 degrees.
- The source checkpoint is 54,118,869 bytes with SHA-256
  `d7ec20fb959a6949424b70f984d6e78def753f6dc0473215720c54c2ca1c76eb`. It was converted once
  inside a networkless, read-only bubblewrap sandbox; the weights-only state dict is 20,680,271
  bytes with SHA-256 `6699a109af8f4cee0664fdbf9d581a9bbec74650b68e2acb90e4f040f6c5ba90`.
- Current compact field sweep, Splat-SfM, Beam Fusion, covariance lifting, merging, fixed-topology
  compact training, and compact held-out evaluation provide reusable controls and downstream
  seams. No GPS adapter result exists yet.
- The new compact data seal binds the calibration, production manifest, compact manifest, and all
  26 full-resolution field archives; `experiment_contract validate-data` and repository-wide
  contract validation pass. Prospective review tightened preprocessing, geometry, evaluation,
  failure, warmup, fairness-control, external-import, and real archive loader bindings before
  approving the current exact digest; no outcome was accessed during either review cycle.
- The frozen `.venv` now contains SciPy 1.17.1 and `nvidia-ml-py` 13.610.43; NVML initializes
  against driver 580.159.04 and detects the one local GPU. These exact versions are bound in the
  revised protocol and will be repeated in every environment/resource receipt.
- The distinct reviewer re-approved corrected protocol digest
  `28cac666e0e210e26c2ec066883062799f228b4d3e8734c74bfa0930856515b5` with outcome access
  `none`. All 26 sealed archives declare the required 8,388,608-byte loader cap while their sealed
  actual maximum remains 369,477 bytes; the task is ready for outcome-bearing implementation and
  execution.

## Minimal Plan

1. Freeze and independently review the exact development protocol and external-model bindings.
2. Implement the CPU contract, optional GPS backend, component-native lift/fusion, and guarded
   task driver with deterministic tests.
3. Run focused checks and a scratch mechanism smoke, then execute the exact task as a source-bound
   non-official development run if the worktree remains intentionally uncommitted.
4. Perform the independent results audit, render/smoke the result bundle where possible, update
   experiment/ARA/task records, and run the complete repository gate.

## Status

Blocked on human decision

## Human Decisions

Record escalated questions and dated answers here. An answer that exists only in chat is not
durable task state. Use one block per decision:

```markdown
### Question
### Options
### Recommendation
### Decision
### Date
```

### Question

Should GPS-Gaussian-style depth be tested while fitted 2D Gaussian fields remain the only
per-scene reconstruction observations?

### Options

- Keep only the current deterministic compact geometry paths.
- Replace the compact representation with source RGB and the literal GPS pixel cloud.
- Keep compact fields, test a transient dense GPS depth adapter first, and build a field-native
  stereo successor only if the geometry mechanism passes.

### Recommendation

Keep the compact observations and isolate the frozen dense adapter as the cheapest killing test;
do not let it carry the primary VRAM claim.

### Decision

On 2026-08-04 the user instructed: “make this the current tasks/experiments we want to do. then
do exactly this.”

### Date

2026-08-04

### Question

May the completed reviewed work be committed to `main` and pushed to the configured remote?

### Options

- Leave the completed changes local and uncommitted.
- Commit directly on the already active `main` branch and push `main` after all required gates.

### Recommendation

Commit the coherent task only after its result audit and full verification pass, then push the
verified `main` tip without force.

### Decision

On 2026-08-04 the user instructed: “when you are done please commit and merge everything into
main. then push to remote”.

### Date

2026-08-04

## Handoff Log

Append Driver handoffs, Reviewer verdicts, and session-completion entries in chronological order.
Use `###` for entries and `####` for their fields so entries remain nested below this section.
Never delete earlier entries. On terminal closeout, archive the complete record as
`docs/tasks/<task-id>-<slug>.md`, change the archived `Turn` to `none`, and reset this file to the
unchanged template.

### Prospective protocol review of initial digest

#### Driver

Codex-gps-field-driver

#### Reviewer

Codex-gps-protocol-reviewer

#### Verdict

Revision required for digest
`ad11faa5523f860da3e0da2de7148a322f32fca1f5d18522f0fa469c1e238b74`.

#### Outcome access

None. The reviewer did not execute the scaffold, access source RGB/masks, or inspect experiment
outcomes.

#### Required changes

Freeze proxy and rectification conventions; inverse-depth signs; consistency, confidence, and
uncertainty equations; component sampling/fusion/ties; exact-count failures; common held-out area
sampling; the precise shuffled intervention; warmup isolation/order; a same-pair/full-field
deterministic control; and strict external key/shape/import-origin binding.

#### Driver response

Revised the preregistration before implementation or outcome access and requested a fresh review
of the new digest.

### Prospective protocol review of second digest

#### Driver

Codex-gps-field-driver

#### Reviewer

Codex-gps-protocol-reviewer

#### Verdict

Revision required for digest
`4aa7be98363493e6119b76ba1d2852704ffb6e1025ffbf755734313bff080f21`.

#### Outcome access

None. The reviewer again performed no initialization/run and accessed no RGB, masks, or protected
outcomes.

#### Required changes

Freeze GPS eval/parameter/inference/autocast/dataflow semantics; CPU proxy backend and GPU tensor
lifetime; place external import/model/checkpoint activity inside the resource boundary; turn the
1-pixel consistency value into a live gate; bind a sufficient compact archive cap and
`load_alpha=False`; and prepare exact SciPy/NVML dependencies. The driver also accepted the
reviewer's optional request to enumerate complete effective baseline/trainer configurations.

#### Driver response

Applied every required and optional tightening, installed and verified the exact runtime
dependencies, and prepared the new digest for another fresh review before implementation.

### Prospective approval of third digest

#### Driver

Codex-gps-field-driver

#### Reviewer

Codex-gps-protocol-reviewer

#### Verdict

Approved for digest
`aaf07179108ca3224f1360406844e32269094ad809fc238473ae0710a381cc04`.

#### Outcome access

None.

#### Artifact

`experiments/reviews/20260804_gps_field_proxy_depth_stage_frame00008_gaussianimage_PROTOCOL_REVIEW.md`

### Environment-only first execution attempt

#### Attempt

Initialized the reviewed development lock at 2026-08-04T13:42:15Z and launched the exact frozen
command. The first `gps_field_proxy` warmup remained entirely in `resource_scope_start` and timed
out after the frozen 300-second GPU-quiescence wait because the unrelated user process
`ExtrinsicSandbox` (PID 229006) remained registered as a CUDA compute process with about 1.4 GiB
allocated.

#### Outcome access

None. The failure receipt records an empty stage trace, `heldout_loaded=false`,
`quality_metrics_written=false`, no input-boundary/model receipt, and no CUDA model allocation.
No compact archive, GPS checkpoint, model prediction, metric, RGB, or mask was opened or produced.

#### Blocker and safe continuation

The user must close or authorize stopping `ExtrinsicSandbox` before a quiescent, comparable VRAM
run is possible. Preserve the failed attempt, archive it under a clearly named environment-failure
directory, reinitialize the canonical development run against the then-current source diff, and
disclose both attempts in the final experiment record. Do not weaken or bypass the zero-foreign-
process guard.

#### Driver response

Set the administrative task status/review envelope to ready without changing the approved
protocol digest, then began the reviewed implementation.

### Pre-outcome loader-contract correction

Before initializing or executing a run, the driver exercised the selected-view loader against the
sealed real compact archives and found that their internal metadata declares an exact 8,388,608-byte
loader-contract ceiling. The earlier 400,000-byte caller cap covered every sealed archive's actual
size but was correctly rejected because `CompactView.load` requires the caller ceiling to equal the
archive-declared ceiling. No model prediction, metric, held-out evaluation, or other experimental
outcome was produced or inspected.

The driver reopened the task as draft, changed only the compact caller ceiling and its explanatory
policy to 8,388,608 bytes, retained the independently sealed maximum actual archive size of 369,477
bytes, and returned the resulting exact digest for a fresh outcome-unseen prospective review before
any run initialization or execution.

### Prospective re-approval after loader-contract correction

#### Driver

Codex-gps-field-driver

#### Reviewer

Codex-gps-protocol-reviewer

#### Verdict

Approved for corrected digest
`28cac666e0e210e26c2ec066883062799f228b4d3e8734c74bfa0930856515b5`.

#### Outcome access

None. The reviewer did not initialize or run the experiment and inspected no RGB, masks, run
directories, metrics, predictions, or held-out outcomes.

#### Decision

The 8,388,608-byte caller cap is required by the exact `CompactView.load` equality contract and is
declared by every one of the 26 sealed archives. The amendment remains fail-closed: the data seal
and compact manifest bind each exact archive hash and actual size before load, the largest actual
archive is 369,477 bytes, internal member and semantic/payload checks remain enforced, and
`load_alpha=False` does not read or decode the packed alpha payload. The corrected protocol is
approved and ready for implementation and execution within its unchanged development-only claim
boundary.

#### Artifact

`experiments/reviews/20260804_gps_field_proxy_depth_stage_frame00008_gaussianimage_PROTOCOL_REVIEW.md`

### Divergent-lineage reconciliation and task renumbering

#### Event

The local worktree carried this uncommitted record and its staged implementation on top of
`36630c7`, while `origin/main` advanced by one commit (`5aff5f5`) that independently completed
RTGS-011 (2D-to-3D literature review), RTGS-012 (probabilistic compact field lifting pipeline),
and RTGS-013 (six image-backed Janelle Gaussian2D experiments). `main` was fast-forwarded to
`5aff5f5`; only `.agents/state/current-task.md` was contested.

#### Resolution

On 2026-08-07 the owner directed that the completed RTGS-013 record be archived and that this
in-flight task take the active slot. RTGS-013 was copied verbatim to
`docs/tasks/RTGS-013-gaussian2d-image-refinement-janelle-frame00008.md` with only its archived
`Turn` changed to `none`, and this record was restored as the single active task.

#### Renumbering

The identifier changed from `RTGS-011` to `RTGS-014` because the incoming lineage had already
archived `RTGS-011` for the literature review. The old identifier appeared only in this file; the
approved protocol JSON, its data seal, the prospective review, and every recorded protocol digest
are unchanged, so digest
`28cac666e0e210e26c2ec066883062799f228b4d3e8734c74bfa0930856515b5` remains the approved,
outcome-unseen protocol.

#### Outcome access

None. No run was initialized or executed, no compact archive, checkpoint, prediction, metric, RGB,
or mask was opened, and no staged implementation file was modified during the reconciliation.

#### Unchanged blocker

The task remains `Blocked on human decision`: a quiescent GPU is still required before the frozen
command can run, because the foreign `ExtrinsicSandbox` CUDA process trips the zero-foreign-process
guard.
