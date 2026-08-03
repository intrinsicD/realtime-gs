# Current Task

## Title

Deterministic BENCH-019 source adapters and Stage-1 predictor collector

## Task ID

RTGS-010

## Role Assignment

- Driver: Codex-bench019-adapters-driver
- Reviewer: Codex-independent-bench019-reviewer
- Turn: human

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
  rows, and complete loader-consumed manifest-plus-view field bytes. Requests for alpha agreement,
  MS-SSIM, LPIPS, track yield, or field conditioning fail closed until separately
  defined/implemented.
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
- Twenty-six focused CPU cases pass across Stage and synthetic TUM end-to-end paths, including archive
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
- Independent review revision `73e8072e0178830f3c9ef2577ed3196b960631e4` reproduced four
  evidence-boundary defects: non-ordinary tar file aliases, materialization inventory aliases and
  special nodes, stale predictor publication, and caller-selected identity between the two
  normalized families. The revision now uses an exact tar-type allowlist, a lexical `lstat` tree,
  full replay at publication, and an evidence-complete portfolio plus production-receipt binding.
- Both completed real Stage receipts pass the repaired family binding without collecting predictor
  outcomes: GaussianImage
  `6f30d7dfbe64762071d314ef033dcd5fd5eebec95440d242b016f95f5f99112b` and StructSplat
  no-boundary `54b61f1ef2608adf932dd573d86ccdbd17d2e3af7682e0b50174011351bd894d`.
- All three maintained adapters still replay their exact sources after the tar allowlist repair;
  the repaired focused suite is 26 cases and the complete BENCH-019 suite is 57 cases.

## Minimal Plan

1. Completed: define and test the strict adapter/source-reference contracts and common split.
2. Completed: implement Stage plus TUM planning/materialization and explicit Karate unavailability.
3. Completed: implement the exact-semantics sampled predictor collector and unavailable-metric
   gate.
4. Completed: add bounded CLIs, public architecture documentation, and calibrated development-only
   adapter diagnostics.
5. Blocked on human decision after the second consecutive revision verdict: three prior
   evidence-boundary defects are closed, but a GNU/PAX sparse regular-type alias still bypasses the
   tar allowlist. The recommended final cycle is limited to rejecting sparse metadata and resuming
   the same reviewer pass.

## Status

Blocked on human decision

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

### Escalation (second consecutive revision verdict)

#### Question

Should RTGS-010 receive one final bounded driver/reviewer cycle for the GNU/PAX sparse regular-type
alias, or stop with the ordinary-archive evidence boundary knowingly incomplete?

#### Options

Authorize only the exact sparse-metadata rejection plus its regression and the same acceptance
pass, versus stop/retire the task or explicitly accept the remaining archive gap.

#### Recommendation

Authorize the bounded repair. Rejecting `TarInfo.sparse is not None` closes the reproduced alias
without changing the adapter policy, predictor contract, scientific scope, or defaults.

#### Decision

Pending human direction. Repository workflow requires this escalation before a third driver cycle.

#### Date

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

### Review (independent evidence-boundary review)

#### Verdict

Revision required

#### Self-reviewed

No

#### Correctness

The Stage and TUM source policies, camera conventions, sparse RGB/mask sampling, compact query
equations, deterministic sampling and aggregation, support-not-alpha naming, unavailable-metric
gate, confirmation/Karate ordering, and development-manifest replay all matched their intended
contracts at implementation revision `a592e30bd7bf44a8ca384d5ce141e8985f4e2de2`. The independent
review nevertheless reproduced four fail-closed defects:

- `SafeTumArchive` accepts a POSIX contiguous-file member (`tarfile.CONTTYPE`, type `b"7"`) because
  `TarInfo.isfile()` treats it as a regular file. This contradicts the class's special-member
  rejection contract and leaves other `isfile()` aliases, including implementation-dependent
  sparse types, inside an archive boundary that claims to admit only ordinary files and
  directories.
- File-verified materialization validation accepts undeclared filesystem nodes. Adding both an
  undeclared symlink to a declared RGB file and an undeclared FIFO left
  `validate_materialization(..., verify_files=True)` successful: `Path.is_file()` ignores the FIFO
  and follows the symlink, while `resolve()` collapses the alias into the declared target.
- `write_stage1_predictors` can publish a stale complete-looking artifact because it validates with
  `verify_files=False`. After collecting an honest artifact, appending bytes to a bound `.rtgsv`
  file, and calling the public writer, publication succeeded with `state=complete_development`;
  the same artifact then failed `validate_stage1_predictors(..., verify_files=True)` because its
  compact field differed from the binding.
- Field-family identity is caller-selected rather than artifact-bound. The exact same normalized
  compact directory was accepted as both `structsplat_normalized_no_boundary` and
  `structsplat_normalized_mask_contained`; its compact binding, per-view statistics, and aggregates
  were identical. Provider/blend checks preserve the normalized equation but cannot distinguish
  these two scientific arms or prevent the currently incomplete mask-contained family from being
  relabelled complete.

#### Evidence Quality

The reviewer reran the 21 focused adapter/predictor tests and all 52 BENCH-019 tests successfully.
All three committed development adapters were independently source-replayed: Stage retained
26/23/3 views with held-out ordinals 7/15/23; TUM `fr1/rpy` reproduced 687 associated triples and
148 keyframes; TUM `fr1/xyz` reproduced 789 associated triples and 77 keyframes. The selected TUM
indices, strict association, bounded interpolation, inclusive keyframe thresholds, half-up
endpoints, registered camera, and inclusive depth-mask policy match the sealed historical harness.
Manual transaction probes confirmed traversal rejection, receipt-last publication, and cleanup
after a mid-publication failure. The four negative controls above were accepted despite those
green checks. The authoritative repository gate and unfiltered CPU suite were rerun after this
review record; no confirmation payload, fit, reconstruction, correlation, loss selection, or
default change was accessed or performed.

#### Simplicity

The passive CPU-only module split, exact-key schemas, extraction-free reads, deterministic CSR
queries, sufficient-statistic aggregation, and bounded CLIs are appropriately scoped. Each repair
can remain local to the existing adapter/predictor validators and tests. The field-family repair
needs one explicit production-receipt binding, not another metric or execution framework.

#### Missing Cases

The suite lacks an exact tar-type allowlist case, lexical `lstat` materialization-tree cases for
symlinks/FIFOs/undeclared directories and files, a post-collection field/source-drift publication
case, and a cross-relabel case for the two normalized StructSplat arms. Existing confirmation and
Karate tests assert the expected rejection but do not instrument payload readers to prove the
fail-before-access ordering. No downstream predictor validity, convergence, quality, performance,
compression, or correlation evidence exists; those remain deliberate non-goals of RTGS-010.

#### Required Changes

1. Admit only exact ordinary tar file types (`REGTYPE`/`AREGTYPE`) and directories; reject every
   other member type before indexing it. Add contiguous-file and sparse/special-type regressions
   alongside the existing link/FIFO cases.
2. Make file-verified materialization validation enumerate every descendant lexically with
   `lstat`, require exactly the canonical ordinary directory/file tree, and reject symlinks,
   special nodes, undeclared files, and undeclared directories without resolving aliases into
   declared targets. Add each negative case.
3. Do not publish a `complete_development` predictor from structural validation alone. Require a
   file-verified deterministic replay at the public publication boundary, or an equivalently
   strict collection transaction that proves the exact adapter/source/compact snapshot and
   statistics before receipt-last publication. Add field deletion/drift and source-drift tests.
4. Bind the requested field-family ID to an exact portfolio/production receipt that names and
   hashes the compact manifest and distinguishes `no_boundary` from `mask_contained`; provider and
   blend mode alone are insufficient. Reject incomplete/unbound arm evidence and add both
   cross-family relabelling directions. Reconcile C34/O149 and public docs so the supported claim
   is no stronger than the repaired binding.

#### Optional Improvements

Cap archive member count and aggregate metadata before building the member table, replace the
quadratic TUM association candidate construction with an equivalent bounded search, instrument
confirmation/Karate tests with payload-access sentinels, and document or enforce crash-durability
expectations for directory publication. Keep manifest-plus-view byte accounting named explicitly;
the current byte arithmetic itself matched that documented contract.

### Handoff (complete evidence-boundary repair)

#### Objective

Resume the same independent acceptance pass after systematically closing all four reproduced
archive, filesystem, publication, and family-identity counterexamples without widening RTGS-010's
pre-outcome scope.

#### Reviewed state

Implementation revision `687f8051c9c021ef9c97ee80df73fff83881ee33` on branch
`rtgs/010-bench019-adapters`, following the `Revision required` verdict at
`73e8072e0178830f3c9ef2577ed3196b960631e4`.

#### Changes

- Replaced `TarInfo.isfile()` classification with an exact `REGTYPE`/`AREGTYPE`/`DIRTYPE`
  allowlist, and replaced resolved regular-file inventory comparison with lexical `lstat`
  enumeration of the exact materialized file/directory tree.
- Made predictor publication require file-verified deterministic replay, so deleted/drifted compact
  files and drifted source RGB fail before an output is created.
- Bound every family ID to an evidence-complete record in the adapter's exact portfolio and its
  adjacent hashed production receipt. The receipt must name the expected arm and bind the compact
  manifest plus every ordered view, so the two normalized families cannot be cross-labelled.
- Reconciled public architecture/README language and ARA C34/O149 with the exact family receipt,
  archive/filesystem allowlists, full-replay publication, and manifest-plus-view byte definition.

#### Evidence

- Focused adapter/predictor suite: 26 passed, including contiguous and GNU sparse tar members,
  symlink/FIFO/undeclared file/directory entries, field append/deletion and source drift before
  publication, and both normalized-family relabel directions.
- Complete BENCH-019 suite: 57 passed.
- All three maintained development adapters replayed their exact sources after the tar allowlist
  repair. The real completed Stage receipts matched their exact compact manifests/views at
  `6f30d7dfbe64762071d314ef033dcd5fd5eebec95440d242b016f95f5f99112b` (GaussianImage) and
  `54b61f1ef2608adf932dd573d86ccdbd17d2e3af7682e0b50174011351bd894d` (StructSplat no-boundary),
  without collecting predictor outcomes.
- `PYTHONPATH=/home/alex/Documents/realtime-gs-bench019/src
  PY=/home/alex/Documents/realtime-gs/.venv/bin/python ./scripts/verify.sh`: passed, including the
  complete non-slow CPU suite and every structural checker; only the two established PyTorch
  warnings were emitted.
- Ruff, formatting, docs sync, ARA, task workflow, script layout, experiment contracts, and
  `git diff --check` passed.

#### Assumptions

A field family is a provenance identity that cannot be inferred from array values when two arms
share normalized equations. Its portfolio-pinned production receipt is therefore authoritative;
the compact manifest and view files remain the loader-consumed bytes used by the size predictor,
while the receipt is provenance metadata rather than coded field payload.

#### Uncertainties

The repair establishes family/provenance integrity, not that any supported predictor forecasts a
downstream response. Archive aggregate/member caps, faster association, extra payload-access
sentinels, and cross-filesystem directory crash durability remain optional hardening, not observed
correctness failures in the maintained inputs.

#### Review Focus

Reproduce the four original counterexamples exactly; challenge both normalized relabel directions
and production-receipt manifest/view drift; verify the official TUM archives and all three adapter
replays still pass; rerun 26 focused tests, 57 BENCH-019 tests, and the authoritative full gate on
the exact revision.

#### Protected actions not taken

No confirmation payload was opened, no field was fitted, no predictor outcome artifact or
reconstruction/correlation result was produced, no loss/default was selected, and the dirty primary
realtime-gs worktree plus the owner's active Stage conversion remained untouched.

#### Recommended Next Action

If accepted, archive RTGS-010 and open the next development-only task for matched field production
and missing predictor-metric experiments. Keep confirmation sealed until those artifacts and a
prospective protocol receive their own distinct reviews.

### Review (independent repair acceptance pass)

#### Verdict

Revision required

#### Self-reviewed

No

#### Correctness

Implementation revision `687f8051c9c021ef9c97ee80df73fff83881ee33` closes the prior
materialization-tree, stale-publication, and normalized-family-identity defects. It also rejects
the literal `tarfile.CONTTYPE` and `tarfile.GNUTYPE_SPARSE` member types. One archive alias remains:
a GNU/PAX sparse file is exposed by Python as `TarInfo.type == tarfile.REGTYPE` with a non-empty
`TarInfo.sparse` map. The type-only allowlist therefore classified it as ordinary, and
`SafeTumArchive` accepted an archive produced by `tar --format=pax --sparse` whose extra member had
`type=b"0"` and `sparse=[(1048576, 1), (1048577, 0)]`. The ordinary-file archive claim is not yet
fail closed.

Every other requested repair survived independent counterexamples. File-verified materialization
rejected a symlink, FIFO, undeclared file, and undeclared directory while accepting the clean exact
tree. Predictor publication rejected post-collection field append, field deletion, and source RGB
drift without creating an output. Both normalized cross-label directions failed. Production
receipt arm, compact-manifest, view-order, output-binding, and non-adjacent-path mutations failed,
as did incomplete state, observed/required-count drift, and missing evidence.

#### Evidence Quality

The 26 focused adapter/predictor tests and all 57 BENCH-019 tests passed. The literal contiguous
and old GNU sparse member types were independently rejected, but the generated GNU/PAX sparse
negative control above was accepted and is absent from the committed suite. Both real completed
Stage receipts passed exact binding without collecting predictor outcomes: GaussianImage at
`6f30d7dfbe64762071d314ef033dcd5fd5eebec95440d242b016f95f5f99112b` and StructSplat
no-boundary at `54b61f1ef2608adf932dd573d86ccdbd17d2e3af7682e0b50174011351bd894d`;
the incomplete mask-contained family remained rejected. All three maintained development adapters
replayed their exact sources. The authoritative repository gate and unfiltered CPU suite passed
after this review record with only the two established PyTorch warnings. No confirmation payload,
real predictor outcome, fit, reconstruction, correlation, loss selection, or default change was
accessed or performed.

#### Simplicity

The repaired lexical-tree validator, publication replay, and portfolio-pinned production receipt
remain small and appropriate. Closing the remaining defect needs no new abstraction: ordinary file
types must additionally have no sparse map before they enter the member table.

#### Missing Cases

The tar regression corpus covers `CONTTYPE`, raw `GNUTYPE_SPARSE`, FIFO, and links but not GNU sparse
metadata carried by a regular PAX member. A regression must exercise the parsed representation
(`type == REGTYPE` and `sparse is not None`), not only another member-type byte. Downstream
predictor validity, convergence, quality, performance, compression, and correlation remain outside
RTGS-010 and were not assessed.

#### Required Changes

1. Reject every otherwise allowed regular member whose parsed `TarInfo.sparse` value is not
   `None`, before root discovery or member indexing. Add a GNU/PAX sparse regression that proves a
   regular-type member with `GNU.sparse.map` cannot pass; retain the literal `CONTTYPE` and
   `GNUTYPE_SPARSE` cases.
2. Rerun the 26 focused cases, complete BENCH-019 suite, all three development adapter replays, and
   authoritative gate, then resume this same bounded acceptance pass. C34/O149 and public ordinary-
   archive wording may remain supported only after the PAX alias is closed.

#### Optional Improvements

The earlier optional hardening remains unchanged: cap archive member/metadata totals, replace the
quadratic association candidate construction with an equivalent bounded search, add explicit
payload-access sentinels for confirmation/Karate, and document crash-durability expectations for
directory publication. No additional receipt, byte-accounting, or family-binding change is
requested by this review.
