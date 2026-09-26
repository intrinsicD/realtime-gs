# Current Task

## Title

Gaussian-only photometric distillation as the current main 2D→3D path

## Task ID

RTGS-025

## Role Assignment

- Driver: Claude-Code-Opus-5.5-driver
- Reviewer: Claude-Code-Opus-5.5-driver
- Turn: driver

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

In progress

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

#### Protected actions not taken
No driver, source binding, prospective review, init-run or outcome access.

#### Recommended Next Action
Implement the driver and its CPU contract tests, freeze `source_binding`, then obtain a distinct
prospective reviewer named/authorized by the user.
