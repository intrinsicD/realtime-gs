# Current Task

## Title

Deterministic BENCH-019 source adapters and Stage-1 predictor collector

## Task ID

RTGS-010

## Role Assignment

- Driver: Codex-bench019-adapters-driver
- Reviewer: Codex-independent-bench019-reviewer
- Turn: reviewer

## Mode

Implement

## Risk

Protected

## Maturity

- Target: Pipeline-integrated
- Reached: Pipeline-integrated

## Goal

Implement the deterministic, CPU-first source-adapter and Stage-1 predictor boundary needed after
RTGS-009: bind the selected Stage capture and development TUM RGB-D sequences to exact 26-view
camera/mask/split recipes, materialize only authorized development inputs reproducibly, and collect
semantics-preserving Stage-1 predictors into sealed JSON without fabricating alpha, conditioning,
correspondence, or perceptual metrics that the available artifacts do not establish.

## Motivation

Accepted RTGS-009 can export and assemble exact downstream receipts, but five portfolio groups lack
matched fields and there is no frozen adapter or common predictor producer. Stage already has
calibrated RGB, masks, cameras, and two complete field families. The official TUM archives have
registered RGB/depth and ground-truth poses but need one deterministic association, keyframe,
camera, mask, and train/held-out contract. Karate has RGB and cameras but no masks, so silently using
full-frame alpha or Gaussian weight sums would invalidate the comparison. BENCH-019 cannot freeze a
defensible prospective protocol until these seams exist and their unavailable quantities fail
closed.

## Success Criteria

- A versioned exact-key adapter manifest binds capture/source identity, selection policy, mask
  policy, ordered views, cameras, source references, train/held-out roles, and a semantic digest.
- The Stage adapter selects the portfolio's exact 26 views and held-out ordinals 7, 15, and 23,
  binds every RGB/mask/calibration artifact, and reproduces the calibrated camera convention.
- The TUM adapter safely reads pinned archives without extraction, uses strict less-than-20-ms
  RGB/depth association, at-most-20-ms pose interpolation, 0.08-m-or-8-degree pose keyframes,
  endpoint-preserving half-up selection to 26 views, ROS-default registered RGB-D intrinsics, and a
  0.3-to-5.0-m registered-depth validity mask. The same held-out ordinals are used.
- Development materialization is exclusive-new and receipt-bound; it emits exact RGB, derived mask,
  calibration, and source metadata. Confirmation materialization is rejected by default.
- Karate remains explicitly unavailable until a source-backed mask policy is supplied; neither
  full-frame alpha nor field density/weight is accepted as a substitute.
- A versioned predictor collector strictly reloads compact fields, preserves additive versus
  normalized equations, validates source/camera bindings, uses deterministic pixel samples, and
  emits only named supported predictors with per-view sample digests and aggregate sufficient
  statistics.
- Supported predictors cover sampled foreground and boundary RGB error, exact support confusion,
  rows, and complete field bytes. Requests for alpha agreement, MS-SSIM, LPIPS, track yield, or
  field conditioning fail closed until separately defined/implemented.
- Adversarial CPU tests cover archive/path safety, association boundaries, interpolation,
  half-up selection, split identity, camera convention, mask derivation, confirmation blocking,
  semantic relabelling, source/camera drift, sample determinism, aggregation, and unavailable
  predictor rejection.
- A calibrated development-only diagnostic validates the Stage adapter and both TUM development
  archives without fitting fields, opening confirmation payloads, or producing BENCH-019 outcomes.
- Public architecture/task docs, focused tests, self-review, and `./scripts/verify.sh` pass before
  handoff.

## Constraints

- Work only in `/home/alex/Documents/realtime-gs-bench019`; leave the dirty primary
  `/home/alex/Documents/realtime-gs` worktree and RTGS-008 artifacts unchanged.
- Preserve the accepted RTGS-009 exporter and StructSplat portable-v1 contracts.
- Treat the portfolio split as sealed: development adapters may inspect payloads; confirmation
  records may be validated as source inventory but confirmation RGB/depth/field outcomes are not
  decoded or materialized in this task.
- Do not fit Stage-1 fields, run realtime-gs reconstruction, freeze a formal BENCH-019 protocol,
  execute correlation analysis, select a loss, or promote a production default.
- Do not call normalized-renderer weight sum alpha. Support overlap is a distinct structural
  diagnostic and must retain that name.
- Writes are exclusive-new or empty-directory only; source archives, owner captures, existing
  compact fields, receipts, and result bundles are immutable.

## Non-Goals

- Resolving Karate segmentation/matting, producing its confirmation fields, or replacing it with a
  different capture without an owner/protocol decision.
- Implementing MS-SSIM/LPIPS, feature tracks, field conditioning, or a learned Stage-1 objective.
- Comparing downstream quality, speed, convergence, or compression; this task creates the
  pre-outcome measurement boundary only.
- Rewriting the sealed historical TUM audit harness or changing realtime-gs fitting/training
  defaults.

## Selected Skills

- rtgs-core
- rtgs-task-workflow
- rtgs-review
- rtgs-docs-sync
- rtgs-verify

## Experiment Contract

None

## Current Evidence

- RTGS-009 is independently accepted at review commit
  `97f60c5a9f262acfcc4464c011514087eb20c553`; its accepted exporter implementation is
  `2b9be15096bad9016432eb5d667d34deb3c42b48` and its complete record is archived.
- The committed portfolio binds three development and three confirmation groups and 88 exact
  source files. Confirmation field evidence and formal-protocol gates remain closed.
- Stage frame 00008 has 26 exact calibrated RGB/mask views. Existing repository convention selects
  zero-based held-out ordinals 7, 15, and 23.
- Development TUM metadata inspection found 789 associated triples / 77 pose keyframes for
  `fr1/xyz` and 687 associated triples / 148 pose keyframes for `fr1/rpy`; both support a strictly
  increasing endpoint-preserving 26-view selection.
- The repository already contains strict compact-view loading, exact additive/normalized point
  queries, packed source alpha, calibrated camera conversion, and deterministic sampling helpers.
- Official TUM guidance recommends the ROS-default calibration for its pre-registered RGB/depth
  images; in this repository's half-integer camera convention that is fx/fy 525 and cx/cy 320/240.
- No RTGS-010 field production, predictor artifact, downstream row, or confirmation outcome exists.
- `rtgs.bench019_adapters` now implements exact-key Stage/TUM manifests, deterministic source
  replay, safe extraction-free archive reads, canonical camera/mask derivation, exclusive-new
  development materialization, and receipt validation. Karate and confirmation fail before
  payload access.
- `rtgs.bench019_predictors` now strictly reloads complete compact datasets, rederives source
  alpha, preserves additive/normalized equations, uses deterministic exact CSR queries, emits
  per-view/split sufficient statistics plus supported predictor values, and can replay the full
  artifact. All five named unavailable predictor classes fail before input access.
- Twenty-one focused CPU cases pass across Stage and synthetic TUM end-to-end paths, including archive
  safety, policy boundaries, materialization, semantic relabelling, source/camera/alpha drift,
  deterministic aggregation, unavailable requests, and full predictor replay.
- Development-only calibrated adapters were generated and source-replayed without fitting fields:
  Stage `a47fbce3551c29a7c294aa7db3186405705784b901a5983de8989b218f0d196e`, TUM
  `fr1/xyz` `539dec4d279a124e1a6b8c58afbb80d6104cf69e250c1f3e931fbe9d0c334280`, and
  TUM `fr1/rpy` `9e264a536721eeb6714a5393726884a03c6641bfed5fb088bbdd310f5cbdcc71`.
  The TUM replays reproduce 789/77 and 687/148 associated-triple/keyframe counts respectively.
- The authoritative bound-environment gate passes with
  `PYTHONPATH=/home/alex/Documents/realtime-gs-bench019/src
  PY=/home/alex/Documents/realtime-gs/.venv/bin/python ./scripts/verify.sh`, covering Ruff,
  formatting, the complete non-slow CPU suite, docs sync, the ARA ledger, script layout, agent
  workflow, and experiment contracts.

## Minimal Plan

1. Completed: define and test the strict adapter/source-reference contracts and common split.
2. Completed: implement Stage plus TUM planning/materialization and explicit Karate unavailability.
3. Completed: implement the exact-semantics sampled predictor collector and unavailable-metric
   gate.
4. Completed: add bounded CLIs, public architecture documentation, and calibrated development-only
   adapter diagnostics.
5. Completed: self-review, focused and full verification, implementation commit, and evidence-bound
   handoff to the same independent reviewer.

## Status

In review

## Human Decisions

### Question

Should the missing Karate mask be replaced implicitly so all six groups look production-ready?

### Options

Invent full-frame alpha or reinterpret Gaussian density, versus keep Karate visibly blocked until a
source-backed policy or replacement capture is approved.

### Recommendation

Keep it blocked. An implicit mask would change the Stage-1 objective and make alpha/support metrics
non-comparable.

### Decision

(Owner, standing direction in chat.) Continue the recommended evidence-first implementation and
identify missing experiments rather than manufacturing readiness. RTGS-010 therefore fails closed
on Karate and leaves its resolution to a distinct follow-up decision.

### Date

2026-08-03

## Handoff Log

Append Driver handoffs, Reviewer verdicts, and session-completion entries in chronological order.
Use `###` for entries and `####` for their fields so entries remain nested below this section.
Never delete earlier entries. On terminal closeout, archive the complete record as
`docs/tasks/<task-id>-<slug>.md`, change the archived `Turn` to `none`, and reset this file to the
unchanged template.

### Handoff (deterministic adapter and predictor boundary)

#### Objective

Independently audit RTGS-010's development-only source adapters and Stage-1 predictor collector for
correctness, fail-closed protected-data behavior, deterministic replay, exact family semantics, and
documentation/evidence consistency.

#### Reviewed state

Implementation revision `a592e30bd7bf44a8ca384d5ce141e8985f4e2de2` on branch
`rtgs/010-bench019-adapters` in `/home/alex/Documents/realtime-gs-bench019`.

#### Changes

- Added exact-key Stage and TUM adapter manifests with deterministic source replay, safe
  extraction-free TUM archive reads, frozen association/interpolation/keyframe/selection policies,
  canonical camera and mask derivation, and receipt-bound exclusive-new development
  materialization.
- Added a strict compact-field predictor collector that rederives source alpha, preserves additive
  versus normalized equations, samples foreground/boundary/query pixels deterministically, emits
  sufficient statistics plus supported RGB/support/size predictors, and fully replays its sealed
  artifact.
- Added bounded adapter and predictor CLIs, three source-replayed development manifests, public
  architecture/README contracts, ARA observation/architecture/claim entries, and adversarial CPU
  coverage. Karate, confirmation payloads, and unavailable predictor classes fail closed.

#### Evidence

- Focused adapter/predictor suite: 21 passed.
- All BENCH-019 tests: 52 passed.
- Development adapter files were source-replayed at SHA-256
  `a47fbce3551c29a7c294aa7db3186405705784b901a5983de8989b218f0d196e` (Stage),
  `539dec4d279a124e1a6b8c58afbb80d6104cf69e250c1f3e931fbe9d0c334280` (TUM xyz), and
  `9e264a536721eeb6714a5393726884a03c6641bfed5fb088bbdd310f5cbdcc71` (TUM rpy).
- `PYTHONPATH=/home/alex/Documents/realtime-gs-bench019/src
  PY=/home/alex/Documents/realtime-gs/.venv/bin/python ./scripts/verify.sh`: passed, including the
  complete non-slow CPU suite and every structural checker; only the two established PyTorch
  warnings were emitted.
- `git diff --check`, focused Ruff, docs sync, ARA, script-layout, agent-workflow, and experiment
  contract checks passed.

#### Assumptions

The official pre-registered TUM RGB-D convention is represented by the repository's half-integer
camera coordinates at fx/fy 525 and cx/cy 320/240. Normalized-renderer support is exactly
`weight_sum > 0` and remains a structural support diagnostic, never alpha.

#### Uncertainties

This task establishes a reproducible measurement boundary, not predictor validity against eventual
downstream outcomes. It does not resolve Karate segmentation, perceptual metrics, feature-track
yield, field conditioning, convergence, quality, speed, compression, or loss/default selection.

#### Review Focus

Audit archive traversal/link/special-member defenses and materialization cleanup/publication order;
replay Stage/TUM sources and verify camera/mask/split identities; compare sparse source sampling to
the calibrated dense convention; challenge compact dataset completeness and byte accounting;
verify family relabelling and alpha/support confusion fail closed; and check that no confirmation or
scientific outcome was accessed or implied.

#### Protected actions not taken

No confirmation archive payload was opened, no Stage-1 field was fitted, no realtime-gs
reconstruction was run, no formal BENCH-019 protocol/correlation/result was produced, no loss or
default was selected, and the dirty primary realtime-gs worktree was not modified.

#### Recommended Next Action

If accepted, archive RTGS-010 and open the next evidence task for the missing development-field
production/metric experiments. Keep confirmation sealed until the prospective protocol and
development predictor evidence are independently accepted.
