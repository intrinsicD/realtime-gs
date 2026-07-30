# Current Task

## Title

Additive-field analytic 2D-to-3D refinement objective: registration, additive teachers, and
matched comparison against point-sampled supervision

## Task ID

RTGS-006

## Role Assignment

- Driver: Claude-additive-objective-driver
- Reviewer: Claude-additive-objective-driver
- Turn: driver

## Mode

Implement

## Risk

Protected

## Maturity

- Target: Calibrated
- Reached: Scaffolded

## Goal

Test whether an exact analytic objective on native additive 2D Gaussian fields — closed-form
density/RGB-numerator matching of projected 3D Gaussians under surfel-constrained covariances
with detached transmittance-times-incidence per-view weighting — improves image-free 3D
refinement over the incumbent point-sampled compositing objective, and whether an
analytic-warmup-then-sampled-finish hybrid beats both, at matched initialization, topology,
and budget. The protocol authority is
`experiments/tasks/20260730_additive_analytic_objective_stage_frames00008_00009.json`; this
record coordinates the work and does not duplicate its arms, seeds, gates, or command.

## Motivation

Owner decisions (2026-07-30, in chat, recorded below): build on additive native fields only,
because for `blend_mode="additive"` with zero support fade the analytic density/RGB loss is
exact rather than the documented normalized-blend proxy (`src/rtgs/lift/field_loss.py`
docstring; the `analytic_semantics` branch in `src/rtgs/lift/field_lifter.py`), and the
priority is a working 2D-to-3D pipeline. The occlusion mismatch of additive matching is
addressed with detached center-transmittance weights (`src/rtgs/lift/field_visibility.py`)
combined with surfel incidence weighting in the spirit of ADR-XXXX (`surfel_lift.py`), and
diagnosed with a per-primitive conflict-score table.

## Success Criteria

- The experiment task JSON is registered as `draft`, passes
  `scripts/experiment_contract.py validate`, and names every open blocker honestly
  (additive bundles and seal, implementation, config freeze, prospective review).
- Additive native 2D bundles for both protocol frames exist under `dataset/`, produced by
  deterministic mask-aware stage-1 native fitting with recorded config and seed; the additive
  data seal is built with `build-seal` and swapped into the task before review.
- The surfel-constrained visibility-weighted analytic objective, the hybrid schedule, and the
  matched point-sampled budget rule are implemented behind existing seams with CPU tests
  covering analytic-equals-rendered-image exactness on additive fields, occlusion-weighting
  behavior, and determinism.
- A distinct prospective reviewer approves the protocol digest without outcome access; the
  run executes under the one canonical run root; the bundle passes `check-run` and
  `scripts/check_results_bundle.py`; the outcome is logged in `docs/EXPERIMENTS.md` and
  audited before any claim or default motion.
- `./scripts/verify.sh` passes at every commit.

## Constraints

- Hard Rules 7-9: task-first registration before result-bearing driver code; no run before an
  approved prospective review; reconstruction and evaluation stay image-free per the
  `direct_compact` arm; decision metrics come only from the exact selected rasterizer
  semantics on the frozen split.
- The incumbent point-sampled compact trainer remains the untouched baseline; no behavior
  changes to StructSplat or normalized-blend paths.
- No edits to other registered experiment tasks, their reviews, or their run roots.

## Non-Goals

- StructSplat or normalized-blend upgrades, and any exact-quotient (cross-multiplied)
  normalized loss — deferred by owner decision until the additive path is shown to work.
- Beam-fusion or carrier-path changes, GPU work, performance claims, cross-dataset
  generalization, topology/densification changes, or production-default changes.
- Promoting any quantitative statement without the results audit and ARA gates.

## Selected Skills

- rtgs-core
- rtgs-task-workflow
- rtgs-experiment
- rtgs-review
- realtime-gs-results-audit

## Experiment Contract

experiments/tasks/20260730_additive_analytic_objective_stage_frames00008_00009.json

## Current Evidence

- Registration-only; no runs and no result numbers. Design basis verified in source:
  exact-vs-proxy semantics (`field_loss.py`, `field_lifter.py` `analytic_semantics`),
  detached transmittance visibility (`field_visibility.py`), incidence gating and thin-axis
  surfels (`surfel_lift.py`, rho 0.1, 70-degree gate), the additive legacy conversion seam
  (`data/field_inputs.py`), and the point-sampled baseline loss
  (`optim/compact_trainer.py:1672`).
- Empirical cautions from `docs/EXPERIMENTS.md` motivating the protocol shape: 2026-07-17
  (latent hard-min correspondence settles in stable wrong basins) and 2026-07-20
  (a correspondence-free consensus objective improved itself while distance-to-truth
  worsened). Decisions therefore bind to held-out compact metrics, never to the training
  objective, and the occlusion story gets an explicit unweighted-analytic control arm.

## Minimal Plan

1. Register the draft experiment task JSON with frozen splits, seeds, stages, comparators,
   metrics, resource protocol, and honest blockers. (this change)
2. Produce additive native bundles for both frames with deterministic stage-1 tooling; build
   and swap the additive data seal.
3. Implement the surfel/visibility-weighted analytic objective and hybrid schedule with CPU
   tests behind existing seams; freeze the provisional refinement configuration.
4. Write and freeze the task driver and `run_command`; obtain the distinct prospective
   review; set the task `ready`.
5. `init-run`, execute, render and gate the bundle, log the outcome, and run the results
   audit before any claim motion.

## Status

In progress

## Human Decisions

### Question

Which 2D-field semantics does the analytic-objective experiment build on, given the
documented normalized-blend proxy gap?

### Options

Additive native fields only; keep StructSplat normalized fields under the density/numerator
proxy; implement the exact cross-multiplied quotient loss for normalized fields now.

### Recommendation

Additive-only: the analytic loss becomes exact by construction, the experiment isolates the
2D-to-3D objective question, and the normalized handling stays available later because
`blend_mode` is stored per field artifact.

### Decision

(Owner, in chat.) Use additive native fields only, not StructSplat. Everything potentially
given up by that choice (StructSplat stage-1 quality, exact handling of the normalized mode)
is addressed later, after the approach is shown to work. The priority is that the 2D-to-3D
pipeline works.

### Date

2026-07-30

## Handoff Log

Append Driver handoffs, Reviewer verdicts, and session-completion entries in chronological order.
Use `###` for entries and `####` for their fields so entries remain nested below this section.
Never delete earlier entries. On terminal closeout, archive the complete record as
`docs/tasks/<task-id>-<slug>.md`, change the archived `Turn` to `none`, and reset this file to the
unchanged template.
