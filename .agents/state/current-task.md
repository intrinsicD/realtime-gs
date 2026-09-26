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

Decide

## Risk

Protected

## Maturity

- Target: Not applicable
- Reached: Not applicable

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

None

## Current Evidence

RTGS-016 (C43), RTGS-021 (C47), RTGS-024 (C48) and the unregistered 2026-09-26 diagnostic tables.

## Minimal Plan

1. Preserve the diagnostics and record a staging observation.
2. Update roadmap/README main-path wording.
3. Verify, self-review, hand off; propose the registered follow-up.

## Status

Provisionally accepted (self-reviewed)

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
