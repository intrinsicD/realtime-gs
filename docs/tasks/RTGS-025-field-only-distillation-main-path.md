# Current Task

## Title

Gaussian-only photometric distillation as the current main 2D→3D path

## Task ID

RTGS-025

## Role Assignment

- Driver: Claude-Code-Opus-5.5-driver
- Reviewer: Claude-Code-Fable-5.1-reviewer
- Turn: none

## Mode

Implement

## Risk

Protected

## Maturity

- Target: CPU-contracted
- Reached: CPU-contracted

## Goal

Record the user's direction decision: the current main path from compact 2D Gaussians plus
cameras to 3DGS is forward rendering with alpha compositing against the decoded 2D fields
(photometric distillation), not tomographic inversion. Tomography, Beam Fusion and field-lift
remain optional initializers that must beat a simple control before use.

## Motivation

The 2026-09-26 review of RTGS-016 found that its tomography forward model treats opaque RGB
observations as additive line integrals, uses a non-identified 2D fit weight as density, and binds
3D primitives to independently fitted 2D footprints. It also lacked a photometric positive
control. RTGS-021/C47 already showed that decoded 100k/view fields can train a recognizable
Janelle reconstruction with the existing RGB trainer, but with a photograph/mask-derived
initialization and boundary halos. Unregistered 2026-09-26 diagnostics
(`ara/evidence/tables/20260926_gaussian_only_diagnostic/`) point the same way.

## Success Criteria

- Roadmap and README name the main path and its open questions without new quantitative claims.
- The diagnostic scripts and raw outputs are preserved with an explicit non-registered boundary.
- The next result-bearing step is proposed as a separate registered experiment, not executed here.

## Constraints

No change to production defaults, historical evidence, C43/C47 wording or frozen protocols.
Diagnostics are single-seed and unaudited; they may appear only as staging observations.

## Non-Goals

Implementing a new trainer, registering or running the follow-up experiment, promoting a claim,
or retiring the tomography/field-lift code.

## Selected Skills

- `rtgs-core`
- `rtgs-task-workflow`
- `rtgs-docs-sync`
- `rtgs-verify`

## Experiment Contract

experiments/tasks/20260926_field_only_distillation_stage_frame00008.json

## Current Evidence

RTGS-016 (C43), RTGS-021 (C47), RTGS-024 (C48) and the unregistered 2026-09-26 diagnostic tables.

## Minimal Plan

1. Preserve the diagnostics and record a staging observation.
2. Update roadmap/README main-path wording.
3. Verify, self-review, hand off; propose the registered follow-up.

## Status

Accepted with follow-up

## Human Decisions

### Question
Which 2D→3D path is the current main path for compact-Gaussian-only reconstruction?
### Options
Tomographic/field-lift inversion; photometric distillation against decoded 2D fields.
### Recommendation
Photometric distillation, with tomography only as an initializer candidate.
### Decision
The user asked to set photometric distillation as the current main path and to demonstrate it on
the Janelle (stage with fabric) 2D Gaussians.
### Date
2026-09-26

### Question
Merge branch `rtgs-025-main-path` (main-path decision, `rtgs.data.field_targets`, audited
field-only screen) into `main`, and register the N248 successor that lifts the photograph
reference above its unchanged floor?
### Options
Merge and register the successor; merge only; hold the branch.
### Recommendation
Merge and register the successor; the decoder and experiment records are verified and the
experiment carries an independent accepted_with_limits audit.
### Decision
2026-09-27: the user directed a successor instead of an immediate merge decision: identify
floaters with all masks and actively move them inside the masks; hold out two RGB-field views as
novel views; if the method works, retrain on all data. The merge of `rtgs-025-main-path` is
deferred and carried into RTGS-026.
### Date
2026-09-27

## Handoff Log

Append Driver handoffs, Reviewer verdicts, and session-completion entries in chronological order.
Use `###` for entries and `####` for their fields so entries remain nested below this section.
Never delete earlier entries. On terminal closeout, archive the complete record as
`docs/tasks/<task-id>-<slug>.md`, change the archived `Turn` to `none`, and reset this file to the
unchanged template.

### Handoff (2026-09-26)

#### Objective
Record photometric distillation from decoded compact fields as the current main path and show it on Janelle.

#### Reviewed state
Uncommitted working tree on `25dc83d4db3e6e620d2d24bb29d74364cbb654c2`: README.md, docs/ROADMAP.md,
docs/EXPERIMENTS.md, ara/staging/observations.yaml (O178), this record, and
`ara/evidence/tables/20260926_gaussian_only_diagnostic/`.

#### Changes
Main-path section in the roadmap and README (no numbers); dated experiment-log entry; staged O178
with preserved diagnostic scripts and raw outputs. No source, default, protocol or claim change.

#### Evidence
`./scripts/verify.sh` exit 0 on this tree. Diagnostics are single-seed and unregistered.

#### Assumptions
Packed alpha in compact views counts as part of the 2D representation for the demo.

#### Uncertainties
Halos outside the silhouette are unmeasured; field-only initialization without alpha is untested.

#### Review Focus
Wording must not read as a quality claim; O178 boundary.

#### Protected actions not taken
No registered experiment, claim promotion, default change, commit or push.

#### Recommended Next Action
Register the field-only initializer/halo experiment (roadmap main-path list) with a distinct prospective reviewer.

### Review (2026-09-26, self)

#### Verdict
Accepted

#### Self-reviewed
Yes

#### Correctness
Docs state a direction decision only; numbers live in O178 and its evidence table.

#### Evidence Quality
Diagnostic only; explicitly bounded.

#### Simplicity
Documentation and evidence preservation only.

#### Missing Cases
Independent review of the wording; halo measurement.

#### Required Changes
None.

#### Optional Improvements
Archive to docs/tasks after a human or distinct reviewer promotes it.

### Human decision and handoff (2026-09-26, target decoder)

#### Objective
The user asked to commit the decision record and to implement faster compact-field target
decoding. This extends RTGS-025 from Decide to a bounded Implement slice; a trainer remains out
of scope.

#### Reviewed state
Branch `rtgs-025-main-path` after `3c096c4`, plus `src/rtgs/data/field_targets.py`,
`tests/test_field_targets.py`, and ARCHITECTURE/CLAUDE.md map rows.

#### Changes
`rtgs.data.field_targets`: exact edge-origin pinhole downscale, box quadrature sites, decode of
a compact view through the CSR tile index or its lazy CUDA twin (in-window sites only, zero
outside), packed-alpha fraction on the same grid, and an optional SHA-256-keyed local cache.

#### Evidence
CPU tests: index decode equals the all-component reference and a direct clamped box average;
coverage/zero-outside; alpha fraction; cache round trip; projection consistency of the scaled
camera. CUDA parity test self-skips on CPU and passed on the local RTX 3050. Local timing
(diagnostic, contended desktop GPU, not a benchmark claim): Janelle `gaussians2d_native_fullres`
26 views at downscale 8 decoded in about 2 s after the one-time CUDA JIT build, against the
reference scan used in the O178 demo; the CPU index took about 0.6 s per view. Max
difference to the reference scan on 3000 in-window sites was 1.8e-7. `./scripts/verify.sh` exit 0.

#### Assumptions
Clamping each site to [0,1] before averaging matches the RTGS-021 target convention.

#### Uncertainties
No tracked benchmark entry; timings are not claims. Cache files derived from private captures
must stay in untracked local directories.

#### Review Focus
Coordinate convention of sites and scaled cameras; unbounded index caps for offline decoding.

#### Protected actions not taken
No default, trainer, protocol or claim change; no push or merge.

#### Recommended Next Action
Use `decode_compact_view` in the registered field-only initializer/halo experiment.

### Review (2026-09-26, self, target decoder)

#### Verdict
Accepted

#### Self-reviewed
Yes

#### Correctness
Parity against the reference scan and a direct box average; CPU-first import (CUDA lazy).

#### Evidence Quality
CPU-contracted; GPU parity locally only.

#### Simplicity
Reuses the existing CSR index and CUDA backend; no new kernel.

#### Missing Cases
Normalized-blend fields with partial windows are covered only through the shared query paths.

#### Required Changes
None.

#### Optional Improvements
Add a tracked `benchmarks/run.py` decode entry if decode time becomes a reported quantity.

### Human decision and handoff (2026-09-26, experiment draft)

#### Objective
The user asked to push the branch and draft the field-only experiment, using 2D Gaussians fitted
with a masked loss but without mask containment, supervising colour only inside the mask, and
avoiding floaters.

#### Reviewed state
Branch `rtgs-025-main-path` after `8814d92`; draft task
`experiments/tasks/20260926_field_only_distillation_stage_frame00008.json` (review digest
`3c63b72f076e2afa9edb63ac623c0f01b1d94c1f6020503ef69bc72fcc605d5b`) and its data seal.

#### Changes
Draft protocol: 7 conditions x 3 paired seeds. The existing Trainer's `use_masks` objective
supplies mask-restricted colour plus a silhouette/outside-alpha term, so floaters are
controlled without trusting field colour outside the mask. Primary teacher
`gaussians2d_structsplat_no_boundary_fullres`; contrasts: mask_contained, gaussianimage,
photographs, alpha-hull initialization, premultiplied-black objective. Held-out colour metrics
are inside the held-out mask only; floaters are measured by outside-mask alpha.

#### Evidence
`experiment_contract validate` and `validate-data` pass. Packed training alpha equals the
thresholded undistorted source mask pixel-for-pixel for C0004 and C0022 in all three families.
no_boundary retains 5000-8592 Gaussians/view versus 11000 for the other families (recorded
confound for H2).

#### Assumptions
The existing Trainer `use_masks` objective is the intended mask-restricted colour plus
silhouette supervision; photographs are an oracle reference, not a field-only arm.

#### Uncertainties
Teacher-family differences confound H2 (count, topology schedule, containment).

#### Review Focus
Leakage boundaries, mask-restricted scoring, gate definitions.

#### Protected actions not taken
No driver, source binding, prospective review, init-run or outcome access.

#### Recommended Next Action
Implement the driver and its CPU contract tests, freeze `source_binding`, then obtain a distinct
prospective reviewer named/authorized by the user.

### Human decision and handoff (2026-09-26, driver for prospective review)

#### Objective
The user asked to implement the driver and named Fable 5.1 as the prospective reviewer
(label `Claude-Code-Fable-5.1-reviewer`).

#### Reviewed state
Task `experiments/tasks/20260926_field_only_distillation_stage_frame00008.json`, status draft,
protocol digest `ef904cd749a80de5f25b9c6706cc705617aebfe56bd0b16912695626a99e44af`, source binding
118 files, aggregate `b924ea17f43cd8d8a9041ccc5b76a4a1d8ff8d4e2d535ea25ba4af937e24b77d`.

#### Changes
Driver `scripts/experiments/20260926_field_only_distillation_stage_frame00008.py`, report module
`..._report.py`, CPU contract tests `tests/test_field_only_distillation.py`; protocol gained
machine-readable initialization parameters, the mask family, decoder parity seeds, the
source binding, and two disclosed boundary notes.

#### Evidence
Focused tests pass on CPU and GPU; `./scripts/verify.sh` exit 0. A non-protocol GPU smoke in
`.scratch/` (60 iterations, one seed, two training views standing in for held-out) exercised
prepare, initialize, all seven fit conditions, evaluate and the gate code end to end.

#### Assumptions
Each phase runs in a fresh worker process; the audit-hook guards are per process (the smoke
confirmed the prepare guard denies later held-out opens within the same process).

#### Uncertainties
The smoke exposed training-view teacher fidelity to the Driver before review (disclosed in the
claim boundary; thresholds unchanged). The alpha hull has fewer shell voxels than n_points.

#### Review Focus
Leakage boundaries per phase; mask-restricted scoring; gate inequalities; init count confound;
whether the smoke disclosure affects approval.

#### Protected actions not taken
No init-run, protected execution, held-out access or outcome inspection.

#### Recommended Next Action
Fable 5.1 prospective protocol review of the exact digest.

### Review (2026-09-26, Fable 5.1 prospective protocol review V1)

#### Verdict
Revision required

#### Self-reviewed
No

#### Correctness
Claude-Code-Fable-5.1-reviewer (claude-fable-5-1, effort max, no nested agents, no outcome
access) rejected digest `ef904cd7...`: B1 environment lookup of a non-existent `realtime-gs`
distribution outside the coordinator's error handling; B2/B3 protocol wording precision.
Verbatim: `experiments/reviews/20260926_field_only_distillation_stage_frame00008_PROTOCOL_REVIEW_V1_REJECTED.md`.

#### Evidence Quality
Reviewer recomputed the digest, ran validate/validate-data and the focused tests; it could not
recompute the source-binding aggregate in its sandbox.

#### Simplicity
Design accepted as is; fixes are bounded to driver start-up and protocol text.

#### Missing Cases
Coordinator start-up path was not exercised by the smoke.

#### Required Changes
B1-B3.

#### Optional Improvements
R1-R8.

### Handoff (2026-09-27, revision 1 for review round 2)

#### Objective
Obtain a second prospective review of the revised digest.

#### Reviewed state
Digest `05048698cc34656fd562c8f60d534037745d99acf91b9cac6712a5f314b23f61`, source binding 118
files `587419c468566af2a1a3181f992012a00eff253a5942950e3bfd90be74dab206`.

#### Changes
See `experiments/reviews/20260926_field_only_distillation_stage_frame00008_DRIVER_RESPONSE_R1.md`:
B1-B3 and R1-R6, R8 adopted; R7 declined with reason.

#### Evidence
Focused tests pass; `./scripts/verify.sh` exit 0; repeated non-protocol GPU smoke passed including
preflight and the new prepare checks.

#### Assumptions
Each phase runs in a fresh process; the V1 design findings stand unchanged.

#### Uncertainties
The live source-binding aggregate was not independently recomputed by the V1 reviewer.

#### Review Focus
B1-B3 fixes and the adopted R1-R6/R8 changes; confirm no scientific design drift.

#### Protected actions not taken
No init-run, protected execution, held-out access or outcome inspection.

#### Recommended Next Action
Fable 5.1 review round 2. A second rejection escalates to a human decision per workflow.

### Review (2026-09-27, Fable 5.1 prospective protocol review round 2)

#### Verdict
Accepted

#### Self-reviewed
No

#### Correctness
Claude-Code-Fable-5.1-reviewer approved digest
`05048698cc34656fd562c8f60d534037745d99acf91b9cac6712a5f314b23f61`: B1-B3 resolved, adopted
optional changes correct, no design drift. This accepts readiness to execute, not results.
Verbatim: `experiments/reviews/20260926_field_only_distillation_stage_frame00008_PROTOCOL_REVIEW.md`.

#### Evidence Quality
Digest, validate, validate-data and focused tests reproduced by the reviewer; the source-binding
aggregate could not be recomputed in its sandbox, so source is pinned by git object ids
(`src/rtgs` tree `641dbc4ee15b2c9ff697c2b3e2f83fb25135c195`) and enforced by `init-run`.

#### Simplicity
No further changes required.

#### Missing Cases
Residual execution conditions 1-6 in the review (single `run` invocation policy, audit must name
the boundary-band difference and H2 family confound, optional hardening deferred).

#### Required Changes
None.

#### Optional Improvements
Coordinator re-entry hardening deferred to a later task (condition 6).

Driver note: condition 4 satisfied by a non-protocol prepare-only smoke over all 22 frozen
training views in `.scratch/` (0 clipped mask pixels, all source digests match the seal; no
held-out access). The task is `ready`; `init-run` and execution await the user's go-ahead.

### Handoff (2026-09-27, protected run completed; results audit pending)

#### Objective
Execute the approved protocol once and hand the producer records to an independent results audit.

#### Reviewed state
Canonical run `runs/20260926_field_only_distillation_stage_frame00008/` locked at commit
`bc28af326babde2bd49eae7dc71bdedea9ba3db0` (clean tree, non-development), protocol digest
`05048698...`; producer RESULT under `benchmarks/results/20260926_field_only_distillation_stage_frame00008_RESULT.*`.

#### Changes
None to source or protocol. One `run` invocation, exit 0, 06:01:43-06:59:20 UTC; 21/21 cells.

#### Evidence
Producer decision: G0 photograph reference failed its frozen 24.0 dB floor in all seeds
(ph_rand_ms mean 23.831 dB), so H1 and H2 are formally inconclusive. Numerically, every H1
per-seed inequality holds, and the H2 foreground margin (0.07-0.20 dB) is below the frozen 0.2 dB
in all seeds. These are pre-audit producer values, not claims.

#### Assumptions
The GPU was shared at launch with an unrelated process (experiments.latent_agent, ~3.7 GB);
peak cell memory stayed at about 0.2 GB. Timings are contended and descriptive only.

#### Uncertainties
Visual adequacy, boundary-band teacher differences and the H2 family confound await the audit.

#### Review Focus
Results audit per the approved review's conditions 5; no post-hoc reinterpretation of the G0 floor.

#### Protected actions not taken
No report render, viewer receipt, results audit dispatch, claim promotion, default change or merge.
The results audit would read dome-derived previews; user authorization for that payload is required.

#### Recommended Next Action
Ask the user to authorize the Fable 5.1 results audit including dome-derived previews; then render,
viewer smoke, check-run, bundle check, EXPERIMENTS.md and ARA updates.

### Review (2026-09-27, independent results audit)

#### Verdict
Accepted with follow-up

#### Self-reviewed
No

#### Correctness
Claude-Code-Fable-5.1-reviewer (claude-fable-5-1, effort max) returned `accepted_with_limits`
for the protected run after the user explicitly authorized reading 22 dome-derived previews.
All producer numbers reproduce to 1e-9; G0 fails in every seed; H1/H2 inconclusive; the
descriptive H4 expectation is refuted; H3 is not an initialization-quality test (random start
pruned to about 1100 Gaussians at step 600). Verbatim:
`benchmarks/results/20260926_field_only_distillation_stage_frame00008_AUDIT.md/json`.

#### Evidence Quality
Receipts, bindings, input boundaries and all 21 cells checked; downscaled contact sheets only.

#### Simplicity
No rerun or source change.

#### Missing Cases
Full-resolution human inspection of halos/floaters; the photograph reference floor.

#### Required Changes
Record C49/C50 as untested/refuted with the audit's wording limits (done).

#### Optional Improvements
Successor task (N248): lift the photograph reference above the unchanged floor, then retest H1/H2.

### Handoff (2026-09-27, bundle complete)

#### Objective
Close the RTGS-025 experiment bundle after the independent audit.

#### Reviewed state
Run `runs/20260926_field_only_distillation_stage_frame00008/` with AUDIT records, second render,
viewer receipt; tracked receipts in `ara/evidence/tables/20260927_field_only_final_handoff/`.

#### Changes
AUDIT.md/json persisted verbatim; HTTP link mirrors and neutral favicon in the run root; headless
Chrome viewer receipt; report re-rendered; EXPERIMENTS.md entry; N247/N248, O179/O180, C49/C50.

#### Evidence
`check-run` OK, `check_results_bundle.py` OK (with previews), 974 report targets HTTP 200,
`./scripts/verify.sh` exit 0.

#### Assumptions
Headless SwiftShader WebGL is an acceptable viewer receipt when the app pane is unavailable (as in RTGS-016).

#### Uncertainties
Report served on port 8766 instead of the frozen 8765 (occupied by an unrelated server).

#### Review Focus
Promotion of the whole RTGS-025 record (decision, decoder, experiment) by a human or distinct reviewer.

#### Protected actions not taken
No default change, no merge into main, no new experiment.

#### Recommended Next Action
Human decision on merging `rtgs-025-main-path`; register the N248 successor before any rerun.

### Closeout (2026-09-27)

RTGS-025 reached its CPU-contracted target (field-target decoder) and completed its independently
audited development screen (accepted_with_limits). The follow-up owning the untested parity
question and the user's active floater relocation proposal is RTGS-026.
