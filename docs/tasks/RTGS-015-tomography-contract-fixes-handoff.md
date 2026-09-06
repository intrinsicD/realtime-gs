# Current Task

## Title

Correct the tomography field objective and expose source-footprint relaxation

## Task ID

RTGS-015

## Role Assignment

- Driver: Codex-tomography-driver
- Reviewer: Codex-tomography-driver
- Turn: none

## Mode

Implement

## Risk

Protected

## Maturity

- Target: Pipeline-integrated
- Reached: Pipeline-integrated

## Goal

Fix decomposition-dependent RGB normalization, add an opt-in fixed-topology source-footprint relaxation, and clarify the analytic projection/CI contracts.

## Motivation

The user accepted the 2026-09-05 tomography assessment and requested these corrections. Reconstruction remains compact-field-only.

## Success Criteria

- Exact co-located half-weight splits preserve the combined objective and its gradients.
- Reference RGB energy is cached per fit; historical coefficient normalization is explicit.
- Relaxed geometry starts identically, stays SPD, is differentiated and reports actual source drift; hard mode retains its equality.
- The public field lifter exercises the opt-in mode and rejects unsupported topology combinations.
- Existing Beam covariance repair and native compact teacher validation remain the supported paths.
- Focused regressions, full CPU verification, documentation, and self-review complete.

## Constraints

CPU-first, deterministic, image-free reconstruction. Preserve historical protocols/results and the paused GPS work. No performance or reconstruction-quality conclusion from unit tests.

## Non-Goals

A new CT forward operator, changing the production initializer, a calibrated quality comparison, dynamic topology for relaxed footprints, or releasing the SH color anchor.

## Selected Skills

- `rtgs-core`
- `rtgs-task-workflow`
- `realtime-gs-results-audit`
- `rtgs-review`
- `rtgs-docs-sync`
- `rtgs-verify`

## Experiment Contract

None

## Current Evidence

The prior assessment reproduced the RGB split-normalization defect against clean commit 2ebe52cc28a1b0e812a6c436d49225fac4ee943f. Existing covariance repair already implements renderer-aware footprint fitting. Implementation checks below will not establish calibrated quality.

## Minimal Plan

1. Correct/cache normalization and add regression controls.
2. Expose source-footprint relaxation and pipeline diagnostics.
3. Clarify operator/CI semantics, verify, and record a provisional self-review.

## Status

Superseded

## Human Decisions

2026-09-05: User requested “ok please fix these points as recommended”. This selects tomography fixes as the current work; RTGS-014 retains its protocol and pending blocker in the archived handoff.

## Handoff Log

### Session start — 2026-09-05

Implement the requested corrections. Target is pipeline integration with deterministic CPU evidence, not calibrated reconstruction improvement.

### Handoff

#### Objective
Correct the analytic RGB weighting defect and expose a controlled geometry-only source-footprint
relaxation while preserving existing compact reconstruction and covariance-repair paths.

#### Reviewed state
Base commit `2ebe52cc28a1b0e812a6c436d49225fac4ee943f` with source/test diff SHA-256
`2ec04c6c9e44f3efa5408342a0992ad1e231357ee3d6dd4e98c31ecf3d3d4a75`.
Per-file hashes and exact verification scope are in
`ara/evidence/tables/20260905_tomography_contract_checks.json`.

#### Changes
Cached target RGB field-energy normalization plus explicit legacy replay; optional relative source
mean/Cholesky coordinates with a dimensionless tether; original three-parameter hard-fiber API;
fixed-topology guard; explicit operator and drift diagnostics; persisted geometry controls and
round-trip tests; corrected CI/projection documentation. Existing renderer-aware carrier repair
already enforces the recommended path. RTGS-014 history and pending GPS protocol are preserved.

#### Evidence
All 155 tests in the affected contract modules pass. Complete CPU suite including slow tests:
1987 passed, 24 skipped, two frozen-protocol failures. Canonical verify repeats those two failures;
lint and format pass. Docs sync, ARA, script layout, agent workflow and whitespace checks pass.
A scratch calibrated compact-field pipeline execution passes without source RGB, alpha or held-out
fields; no quality metric/model is retained. See the audit note for the exact boundary.

#### Assumptions
Source observations stay frozen during refitting. Soft mode releases only geometry and requires
fixed topology; SH remains anchored at its initial direction. Legacy mode preserves the older
objective, not eligibility to rerun an immutable experiment on changed source.

#### Uncertainties
No calibrated hard-versus-soft quality or complete-cost comparison, GPU execution or independent
review ran. Target-energy preprocessing is quadratic in reference count, cached once and charged
inside refitting. A fixed 1e-12 RGB energy floor bounds black-target normalization in the tests.
The full repository gate is not green: its source-binding failure predates the patch, and the
other frozen-configuration guard rejects the new controls. Historical records remain untouched.

#### Review Focus
Combined-objective/gradient split invariance; hard optimizer compatibility; soft zero-offset/SPD/
finite-difference/persistence contracts; initial versus actual SH anchor directions; avoiding false
physical-density, global-CI or reconstruction-quality claims.

#### Protected actions not taken
No commit, push, official experiment, held-out data consumption, historical protocol/hash refresh,
production initializer promotion, or foreign process termination.

#### Recommended Next Action
Independent review of this source-bound correction. A result-bearing hard-versus-soft comparison
requires a new prospective protocol. Resolve historical protocol lifecycle separately before a
clean commit gate; do not edit old result or approval hashes to accept the current source.

### Review

#### Verdict
Accepted

#### Self-reviewed
Yes

#### Correctness
The scoped regression contracts pass. Full-suite counterexamples led to preserving the hard
fiber's parameter list and the initial history time convention. The zero-iteration legacy replay
edge is covered. The soft parameterization preserves SPD in its finite parameter regime and its
projected gradients match finite differences. Unsupported topology combinations fail explicitly.

#### Evidence Quality
Deterministic CPU contracts with exact source/test hashes, a public pipeline/CLI round trip and a
bounded calibrated execution smoke. This is provisional self-review. Full-gate failure is retained;
no independent scientific audit, calibrated efficacy or performance claim is inferred.

#### Simplicity
Reuse existing fiber geometry, carrier repair and compact teacher semantics. Source offsets are
buffers in hard mode and become parameters only in soft mode; no new renderer or CT pipeline.

#### Missing Cases
Matched calibrated quality/cost evidence, GPU behavior, soft-topology state transfer, and a preferred
source-tether strength remain outside this implementation scope.

#### Required Changes
No further change to the scoped correction is identified. Commit readiness is blocked by the
explicit frozen-protocol verification conflicts; they must not be hidden or fixed by rewriting
historical evidence. Provisional status remains until a distinct reviewer or the user accepts it.

#### Optional Improvements
A separately preregistered native-color versus analytic-proxy and hard-versus-soft comparison can
measure whether the newly exposed geometry freedom is useful downstream.

### Active-slot handoff — 2026-09-05

The user selected RTGS-016: acquire a person Gaussian reference, resolve historical protocol checks, and freeze/run a matched tomography comparison with maskless and second-capture follow-ups. This supersedes only the active slot; the implementation remains provisionally self-reviewed and its evidence is unchanged. RTGS-016 owns the remaining historical checks and calibrated comparison.
