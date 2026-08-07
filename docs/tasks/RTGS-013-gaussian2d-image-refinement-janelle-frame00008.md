# Current Task

## Title

Run six image-backed Janelle Gaussian2D lifting experiments

## Task ID

RTGS-013

## Role Assignment

- Driver: Codex-root
- Reviewer: Volta-protocol_review
- Turn: none

## Mode

Implement

## Risk

Protected

## Maturity

- Target: Calibrated
- Reached: Calibrated

## Goal

Produce one independent, reproducible image-backed experiment and canonical child `index.html`
for each of the six `gaussians2d*` folders in Janelle `frame_00008`, comparing masked and
unmasked execution of the same 2D-field lift plus standard RGB 3DGS refinement.

## Motivation

The preceding RTGS-012 run treated compact fields as image-free proxy datasets and therefore did
not answer the owner's request to use the Janelle images. The corrected experiment must join each
named Gaussian2D folder to the matching calibrated Janelle RGB views and lossless masks.

## Success Criteria

- The exact six owner-named folders are six independently reported dataset units; no other frame
  or Gaussian2D family enters the comparison.
- Every unit uses the same frozen optimizer/validation/test camera partition and three seeds.
- Masked and unmasked arms differ only in the declared support and RGB-mask policy, while both use
  the folder's own Gaussian2D fields for initialization and the matching Janelle JPGs for 3DGS.
- Saved evidence includes initial/final PLYs, validation convergence curves, final held-out image
  metrics, resource receipts, calibrated previews, and one orbit-viewer manifest per folder.
- The canonical v2 report renders six `runs/.../datasets/<folder>/index.html` child pages, and the
  page plus orbit viewer for each completed folder is opened and browser-smoked.
- Focused CPU tests, experiment-contract checks, the independent results audit, and repository
  verification pass or any failure is explicitly preserved without a positive claim.

## Constraints

- Use only optimizer-training cameras for field lifting and RGB gradients; validation/test images
  are reporting-only and may not drive stopping, selection, or tuning.
- Keep all six source manifests and their full component counts sealed. Any deterministic carrier
  reduction must be explicit, identical across datasets, and receipted; no silent 512-field proxy.
- Use the repository's standard `SceneData`, `Trainer`, rasterizer, report-v2, and viewer paths.
- Preserve existing dirty-tree work and unrelated `.idea` changes; do not commit or push.

## Non-Goals

- Reproducing GPS-Gaussian or claiming state-of-the-art quality.
- Establishing cross-scene generality, production defaults, or real-time performance.
- Using the earlier image-free field proxy metrics as RGB-image evidence.

## Selected Skills

- rtgs-core
- rtgs-task-workflow
- rtgs-experiment
- realtime-gs-results-audit
- rtgs-review
- rtgs-verify

## Experiment Contract

experiments/tasks/20260806_gaussian2d_image_refinement_janelle_frame00008.json

## Current Evidence

- The six folders are `gaussians2d`, `gaussians2d_additive`,
  `gaussians2d_gaussianimage_fullres`, `gaussians2d_native_fullres`,
  `gaussians2d_structsplat_mask_contained_fullres`, and
  `gaussians2d_structsplat_no_boundary_fullres`; each contains all 26 camera ids.
- Matching Janelle RGB and mask files exist for all 26 views and are now hard-linked into the
  ignored canonical dataset `rgb/` and `mask/` paths without duplicating bytes.
- Source cardinality ranges from 640 to 100,000 2D Gaussians per view, so reductions and resource
  consequences must be surfaced per folder.

## Minimal Plan

1. Register and seal the six-dataset image-backed protocol; discard a bounded scratch preflight.
2. Implement/test the common driver, metrics, per-folder report summaries, and viewer manifests.
3. Obtain prospective independent protocol approval, initialize the immutable development run,
   and execute all masked/unmasked seed cells.
4. Audit, render, browser-smoke, open every child report/viewer, verify, and close the task.

## Status

Accepted

## Human Decisions

### Correct dataset unit and image requirement

#### Question

Which local Gaussian2D inputs constitute the requested experiments, and must Janelle RGB be used?

#### Options

Treat every discovered compact folder as a proxy dataset, or run one image-backed experiment for
each of the six `gaussians2d*` folders inside Janelle `frame_00008`.

#### Recommendation

Use exactly the six folders in `frame_00008`, pair each with matching Janelle RGB/masks, and keep
each as an independent report unit.

#### Decision

The owner selected exactly those six subfolders and stated: “one experiment for each folder.”

#### Date

2026-08-06

### Whether to authorize one bounded V4 repair/review loop

#### Question

Should RTGS-013 receive one additional bounded repair and outcome-blind review after three
prospective rejections, despite the workflow's two-revision escalation limit?

#### Options

Authorize only the terminal completion-order repair, new source binding/digest, and one fresh V4
review; or stop the current experiment and close or rescope RTGS-013.

#### Recommendation

Authorize the bounded V4 loop because all five V2 surfaces passed and the remaining defect is one
reproduced terminal-state ordering bug with a small, falsifiable repair and test boundary.

#### Decision

Authorized. The owner replied “yes please” to the recommended bounded V4 loop: repair only the
terminal completion ordering, add the outcome-free write-failure/retry tests, freeze a new source
binding and digest, and obtain one fresh independent review before execution.

#### Date

2026-08-06

### Whether to authorize one administrative V5 freeze/review

#### Question

Should RTGS-013 receive one outcome-blind administrative V5 loop after V4 verified the terminal
repair but found that the prospectively hashed task still contains its approval-only blocker and
therefore cannot validly transition to `ready` under the reviewed digest?

#### Options

Authorize only clearing the stale approval blocker while the task remains draft, freezing the new
digest, and obtaining one fresh prospective review with no code, data, scientific-matrix, command,
or outcome change; or stop RTGS-013 without initializing the protected run.

#### Recommendation

Authorize the administrative V5 loop. The V4 terminal repair and every prior guard passed, and the
remaining issue is the contract-bound approval blocker itself. Keep the loop strictly outcome
blind and do not initialize until the blocker-free digest is independently approved.

#### Decision

Authorized. After asking for and receiving the recommendation to proceed, the owner replied “yes
ok” to the strictly administrative V5: preserve V4, clear only the stale approval blocker while
draft, freeze the blocker-free digest, and obtain one fresh outcome-blind review before execution.

#### Date

2026-08-06

## Handoff Log

### Handoff

#### Objective

Independently determine whether the repaired, outcome-blind RTGS-013 protocol is safe to approve
for one image-backed experiment per owner-selected Gaussian2D folder.

#### Reviewed state

Prospective digest
`19a47a047479e3817d78c55d3d4875716caa0b88404446201aa34d767d1ed898`; source binding
`2fa9a8932d3e4c85985833c8e9e867839036c1bc0e2091bb418e4be17caeca87`; V1 rejection preserved
unchanged as `experiments/reviews/..._PROTOCOL_REVIEW_V1_REJECTED.md`.

#### Changes

Replaced the direct worker surface with authenticated canonical tickets, isolated validation from
Trainer timing, froze resources before held-out access, added transitive cell receipts, enforced
live source/data hashes, and deferred six real viewer launches until all measurement endpoints.

#### Evidence

Focused optimizer, experiment-contract, and Janelle protocol tests pass; Python compilation,
Ruff, task validation, and the full 215-file/490,153,435-byte hybrid data-seal validation pass.

#### Assumptions

The six folders are independent experiment/report units over one shared calibrated Janelle frame;
seed 80601 is prospectively fixed for visual presentation only.

#### Uncertainties

No official CUDA matrix has run. Runtime, GPU capacity, result quality, and viewer startup remain
unknown and may not be inferred from discarded scratch preflights.

#### Review Focus

Re-test every V1 execution-boundary finding and search for new leakage, stale-cell substitution,
resource-scope, report-order, or ticket-integrity failures.

#### Protected actions not taken

No run was initialized; no official worker, held-out outcome, result artifact, report, or viewer was
created or inspected.

#### Recommended Next Action

Write a fresh canonical outcome-blind protocol review. Approve only if all blocking findings are
closed under the current digest; otherwise return the task to the driver with exact counterexamples.

### Review (V2 prospective protocol)

#### Verdict

Revision required

#### Self-reviewed

No

#### Correctness

The V1 worker, timing, resource-boundary, cell-resume, source/data, and viewer-launch repairs pass,
but the official lock still fails to authenticate the current review artifact and the generic final
bundle replay is not semantically equivalent to the task-specific receipt validator.

#### Evidence Quality

Outcome access remained `none`; 88 focused tests, contract/data validation, compilation, workflow,
docs-sync, and ARA checks passed under prospective digest `19a47a...d1ed898`.

#### Simplicity

The coordinator needs one terminal post-matrix boundary and idempotent canonical publication rather
than separate partial-success paths.

#### Missing Cases

Review-artifact deletion/drift/malformed locks, coherent receipt-inventory tampering, late
aggregation/viewer failure with same-root retry, complete resource curves with distinct clocks, and
six-entry page/viewer browser smoke receipts are not yet covered.

#### Required Changes

Repair the five blocking findings in the canonical V2 protocol review without weakening prior
source/data/cell guards; preserve that rejection unchanged and obtain another fresh digest/review.

#### Optional Improvements

Keep late presentation recovery observable in `progress.json` and reuse already-live canonical
viewer processes on a retry when safe.

- 2026-08-06 — Codex-root → Volta-protocol_review: prospective review requested after the
  exact six-folder hybrid seal, task-specific producer, CPU protocol tests, and discarded
  largest-folder scratch preflight passed. Outcome access remains `none`; no official run exists.
- 2026-08-06 — Volta-protocol_review → Codex-root: prospective digest
  `7a470a851444f69ce236ed30636741835708886812888b16361f2a5d13129744` rejected with
  outcome access `none`. Repairs required for worker authorization, validation timing, CUDA/total
  resource intervals, stale-cell substitution, live source/data enforcement, and viewer launch.
- 2026-08-06 — Codex-root → Volta-protocol_review: all six blocking surfaces were repaired,
  counterexample tests plus live hybrid-seal/source checks pass, the rejected review is preserved
  unchanged as V1, and a fresh outcome-blind review was requested under the new digest.
- 2026-08-06 — Volta-protocol_review → Codex-root: prospective digest
  `19a47a047479e3817d78c55d3d4875716caa0b88404446201aa34d767d1ed898` rejected with outcome
  access `none`; five repairs are required for review-byte locking, semantic bundle replay,
  retry-safe publication, complete metric/clock curves, and all-six browser smoke evidence.

### Handoff (V3 prospective protocol)

#### Objective

Independently determine whether the second repaired, still outcome-blind RTGS-013 protocol is safe
to approve for the exact six owner-selected, image-backed Janelle Gaussian2D experiments.

#### Reviewed state

Prospective digest
`c3a334db13b3cc51cbe2634cd44f5d04339cdd51115c6420eea5902497ac7b05`; source binding
`6b1f0469735e28e404c6d45d28026cf19c67551fc37035af9055e8d63f246ba4`; V1 and V2
rejections are preserved unchanged with SHA-256 values `ac15bbb7...` and `512f2f200...`.

#### Changes

Authenticated the canonical review artifact bytes in every official lock and worker ticket; made
generic bundle replay semantically and type strict; made aggregation, evidence publication, and
six-viewer launch fail-closed, idempotent, and same-root retry safe; added complete resource curves
with distinct optimizer and worker clocks; and required six independent page/WebGL/orbit smoke
entries.

#### Evidence

The full non-slow CPU suite passes. Targeted counterexamples cover review deletion/drift/wrong-path
locks, receipt omission and boolean aliases, coherent artifact removal, late publication failures,
same-root recovery, partial/conflicting canonical evidence, one-viewer relaunch with five safe
reuses, and missing/failed sixth smoke entries. Contract, 215-file data seal, source binding,
workflow, docs-sync, ARA, Ruff, formatting, and compilation checks pass.

#### Assumptions

The six folders are separate experiment/report units over one shared calibrated Janelle frame;
masked and unmasked arms use the same folder inputs, camera split, optimizer, iteration budget, and
seed schedule. Seed 80601 remains prospectively selected for presentation only.

#### Uncertainties

No official CUDA matrix has run. Runtime, GPU capacity, numerical outcomes, visual quality, and
viewer startup remain unknown and cannot be inferred from discarded scratch work.

#### Review Focus

Reproduce the V2 counterexamples against the current digest, then inspect for any remaining path
from review drift, malformed-but-self-consistent cell artifacts, partial publication, clock mixing,
or incomplete all-six smoke evidence to an accepted result bundle.

#### Protected actions not taken

No official run was initialized and no official worker, held-out outcome, canonical result,
rendered report, or viewer was created or inspected. Outcome access remains `none`.

#### Recommended Next Action

Write a fresh canonical outcome-blind review. Approve only if all prior blocking findings are closed
and no new blocker remains under this exact digest; otherwise return exact counterexamples.

### Review (V3 prospective protocol)

#### Verdict

Revision required

#### Self-reviewed

No

#### Correctness

All five V2 repair surfaces are materially present, but the final completed run receipt and
completed progress transition remain outside the caught post-matrix boundary. An injected final
progress-write failure leaves a nonzero coordinator exit with `run_receipt.json` claiming
`completed` and `progress.json` still `running`.

#### Evidence Quality

Outcome access remained `none`. The exact digest/source binding, prior-review hashes, task/data
contracts, 97-test focused CPU collection, compilation, Ruff, workflow, docs-sync, and diff hygiene
were independently reproduced; the blocker was demonstrated in a temporary outcome-free fixture.

#### Simplicity

One ordered terminal commit is sufficient: write completed progress inside the caught boundary,
publish the successful run receipt last, and route either write failure to canonical failed state.

#### Missing Cases

The tests do not inject failure of either final terminal write and therefore do not cover the
false-completed-receipt ordering counterexample.

#### Required Changes

Move both terminal completion writes into the shared post-matrix failure boundary; ensure no
fallible action follows the successful run receipt; add failure injections for both writes and a
same-root recovery assertion; then freeze a new source binding/digest and obtain a fresh
outcome-unseen review.

#### Optional Improvements

Have the final bundle validator reject disagreement between `progress.json` and a completed run
receipt as defense in depth; the producer ordering remains the primary fix.

- 2026-08-06 — Volta-protocol_review → Codex-root: V3 prospective digest
  `c3a334db13b3cc51cbe2634cd44f5d04339cdd51115c6420eea5902497ac7b05` rejected with outcome
  access `none`. All five V2 surfaces passed, but an injected final progress-write exception left
  a completed run receipt after a nonzero coordinator exit; the terminal commit must be reordered
  inside the fail-closed boundary and re-reviewed under a fresh digest.
- 2026-08-06 — Owner → Codex-root: explicitly authorized one bounded V4 loop containing only the
  terminal completion-order repair, its failure/retry tests, a new source binding/digest, and one
  fresh outcome-blind independent review before any official execution.

### Handoff (V4 bounded prospective protocol)

#### Objective

Independently decide whether the owner-authorized terminal-order repair closes the sole V3 blocker
without changing the six-folder scientific matrix or weakening any V1/V2 guard.

#### Reviewed state

Prospective digest
`56178a3e48eb12829a66476c7bac7b2f22fdd7273ebcaae6102e43d032fb48b5`; source binding
`b10eb15c38bd44da97ad42464870fee64eb5a158f722e3b2cf3a6a1d77f4445a`. V1, V2, and V3
rejections are preserved unchanged with SHA-256 prefixes `ac15bbb7`, `512f2f20`, and `acd84e5a`.

#### Changes

Moved both terminal completion writes inside the shared caught post-matrix boundary. Completed
progress is written first, the completion log is emitted next, and the successful run receipt is
the final fallible action. Added one-shot failures for each completed terminal write and required
both to leave failed receipt/progress state before the same canonical root retries to completion.

#### Evidence

The two new parameterized counterexamples and the existing post-matrix recovery test pass. The
99-test focused Janelle/contract/optimizer collection passes. The exact source/data bindings,
contract validation, canonical full verification, docs-sync, ARA, workflow, script layout, Ruff,
formatting, compilation, and full non-slow CPU suite pass.

#### Assumptions

The owner authorization is bounded to this terminal transition and review. No algorithm, input,
camera split, metric, arm, seed, iteration count, viewer requirement, or claim boundary changed.

#### Uncertainties

No official CUDA cell has run and no held-out outcome, model, metric, preview, report, RESULT,
AUDIT, or viewer has been opened. GPU capacity, runtime, convergence, and visual quality remain
unknown.

#### Review Focus

Inject failure of completed progress and completed run-receipt writes; prove no completed receipt
survives, prove same-root recovery, and verify that no fallible success action follows the final
receipt. Reconfirm all prior digests, source/data seals, and scientific protocol fields.

#### Protected actions not taken

No `init-run`, official coordinator, authenticated worker, held-out evaluation, result rendering,
or viewer launch was invoked. Outcome access remains `none`.

#### Recommended Next Action

Approve only if this exact candidate closes V3 and introduces no bypass. Otherwise reject and stop
the bounded loop with precise evidence; do not broaden the repair.

### Review (V4 bounded prospective protocol)

#### Verdict

Revision required

#### Self-reviewed

No

#### Correctness

The terminal-order repair closes the sole V3 code blocker: completed progress precedes completion
logging, the successful run receipt is the final fallible action, and injected failure of either
terminal write leaves failed receipt/progress state before a same-root retry completes. The exact
reviewed task nevertheless cannot form a valid approved `ready` state because its non-empty
approval-only blocker is protocol-bound.

#### Evidence Quality

Outcome access remained `none`. The exact digest/source/data bindings, three rejected-review
hashes, 99 focused tests, terminal failure injections, contract/data validation, compilation,
Ruff, workflow, docs-sync, ARA, script layout, and diff hygiene were independently reproduced.
The administrative counterexample was also reproduced: clearing `blockers` changes the digest
from `56178a3e...48b5` to `61ba8855...119b`, while retaining it invalidates `ready`.

#### Simplicity

No source or scientific repair is required. The smallest valid path is to clear the stale blocker
prospectively in draft state, freeze the resulting digest, and conduct one fresh outcome-blind
administrative review.

#### Missing Cases

The handoff did not verify that the exact reviewed protocol could perform its administrative
approval transition without changing a digest-bound field.

#### Required Changes

Do not initialize. Obtain explicit owner authority for V5; then clear only the approval blocker,
freeze the new digest, and obtain a fresh review before setting the task to `ready`.

#### Optional Improvements

Add a contract/workflow lint that rejects approval-only blockers before a draft is handed to a
prospective reviewer, preventing this circular transition in future tasks.

- 2026-08-06 — Volta-protocol_review → Codex-root: V4 prospective digest
  `56178a3e48eb12829a66476c7bac7b2f22fdd7273ebcaae6102e43d032fb48b5` rejected with outcome
  access `none`. The terminal repair and all prior guards pass, but the reviewed task hashes its
  non-empty approval blocker; removing it yields `61ba8855...119b`, while retaining it forbids a
  valid ready state. Returned to human decision for an explicitly bounded administrative V5; no
  run was initialized.
- 2026-08-07 — Owner → Codex-root: authorized the recommended blocker-only administrative V5
  freeze/review. No scientific code, data, matrix, command, metric, or outcome change is
  authorized; initialization remains contingent on exact blocker-free approval.

### Handoff (V5 administrative prospective protocol)

#### Objective

Independently decide whether the blocker-free draft can validly transition to approved `ready`
state under one exact digest, without any scientific or source change from the green V4 candidate.

#### Reviewed state

Prospective digest
`61ba885523c4941c842744b142bf85700ac78eb6a2c61e28a20a22637371179b`; unchanged source binding
`b10eb15c38bd44da97ad42464870fee64eb5a158f722e3b2cf3a6a1d77f4445a`. V1–V4 rejections are
preserved unchanged; V4 SHA-256 is `ada7ed58905582d6807641e4fa2bb7620d733c172d219cf62310c5af15034793`.

#### Changes

Cleared only the stale approval-only `blockers` entry while the experiment remained uninitialized
and draft. Reset prior rejected review metadata to pending for the fresh review. No source, data,
scientific configuration, command, dataset, split, arm, seed, metric, report, viewer, or claim
field changed from V4.

#### Evidence

The blocker-free digest recomputes exactly. Source binding remains exactly unchanged. Global task
validation, the complete sealed-data validation, and workflow validation pass; the official run
root remains absent. V4 independently established that all code repairs and prior guards pass.

#### Assumptions

Experiment status and protocol-review metadata are administrative fields excluded from the review
digest; `blockers` is protocol-bound and is now prospectively empty, permitting a valid ready
transition only after approval.

#### Uncertainties

No protected outcome exists or has been accessed. This review establishes authorization integrity,
not GPU operability, runtime, convergence, reconstruction quality, or viewer behavior.

#### Review Focus

Diff the V4 and V5 task payloads semantically; require only the authorized blocker removal plus
administrative status/review reset. Recompute the exact digest/source/data bindings, simulate the
approved ready transition, and reject any digest drift or non-administrative change.

#### Protected actions not taken

No run initialization, coordinator, worker, held-out evaluation, result rendering, or viewer
launch occurred. Outcome access remains `none`.

#### Recommended Next Action

If and only if this exact blocker-free digest validates after inserting its approval metadata and
setting status `ready`, write the canonical approval, perform that administrative transition, and
return the task to the Driver for exact-command execution.

### Review (V5 administrative prospective protocol)

#### Verdict

Accepted

#### Self-reviewed

No

#### Correctness

The blocker-free V5 task is validly approved and `ready` under exact digest
`61ba885523c4941c842744b142bf85700ac78eb6a2c61e28a20a22637371179b`. Reconstructing V4 by
restoring only its stale blocker reproduces exact V4 digest `56178a3e...48b5`; the semantic diff
contains only `blockers` plus the digest-excluded `status` and `protocol_review` envelope. The
live ready state retains the same V5 digest and passes the full task validator.

#### Evidence Quality

Outcome access remained `none`. The live source binding is unchanged at `b10eb15c...44a`; the
complete 215-file data seal revalidates at `1199a410...5a63`; V1–V4 rejection hashes remain exact;
and an in-memory insertion of these exact canonical approval bytes, reviewer metadata, and ready
status returned zero task-validation errors and passed the protected driver's task contract before
the live administrative transition was applied. The final task/data validators pass and the
official run root remains absent.

#### Simplicity

V5 changes no code, data, scientific matrix, command, metric, report, viewer, or claim surface.
Removing the stale blocker before review is the smallest contract-correct repair.

#### Missing Cases

No CUDA cell, held-out evaluation, resource measurement, convergence curve, reconstruction
quality result, report, browser smoke, or viewer has run. Those are downstream experiment and
results-audit obligations, not evidence supplied by prospective approval.

#### Required Changes

None before immutable initialization and exact-command execution under the approved task. Any
protocol or behavior-bearing source change requires a new task digest and prospective review.

#### Optional Improvements

Add a future workflow lint preventing approval-only blockers from reaching prospective review;
this is not required for the now-valid V5 task.

- 2026-08-07 — Volta-protocol_review → Codex-root: administrative-only V5 prospective digest
  `61ba885523c4941c842744b142bf85700ac78eb6a2c61e28a20a22637371179b` approved with outcome
  access `none`. V4 reconstruction proves the sole protocol change is removal of its stale
  blocker; exact ready-state simulation and final task/data validation pass, V1–V4 remain
  byte-identical, and the official run remains absent. Returned to the Driver for exact-command
  initialization and execution.

### Handoff (independent results audit)

#### Objective

Adversarially audit the completed six-folder Janelle result from raw immutable cell artifacts before
any quantitative interpretation enters durable documentation or the report bundle.

#### Reviewed state

Run `runs/20260806_gaussian2d_image_refinement_janelle_frame00008`; approved protocol digest
`61ba885523c4941c842744b142bf85700ac78eb6a2c61e28a20a22637371179b`; source binding
`b10eb15c38bd44da97ad42464870fee64eb5a158f722e3b2cf3a6a1d77f4445a`; review artifact SHA-256
`b243f8eef02c3c660b44a5f952dd37f824a7fa00d45d47422f6b211321ed37f1`.

#### Changes

Initialized the approved dirty-tree development run and executed the exact frozen command. All 36
measured cells plus the warmup completed. The first post-matrix pass failed closed because six live
viewer processes exceeded the one-second HTTP probe; after all six reached HTTP 200, the exact
same-root retry validated every receipt, reused all viewers, and wrote the completed terminal state.

#### Evidence

`run_receipt.json` and `progress.json` are completed with 36 unique cells and no failed cells.
Entry/exit source and 215-file data seals match. The completed viewer launch receipt has six exact
dataset entries, live PIDs, HTTP readiness, and `reused: true`. Machine-readable RESULT JSON/MD,
per-cell raw histories/metrics/resources/input receipts, six dataset summaries, previews, and
viewer manifests exist. Pre-audit `check-run` reports only the expected missing AUDIT and rendered
bundle files; the bundle checker additionally expects post-render browser smoke.

#### Assumptions

Each folder remains an independent development result unit; root medians are navigation-only. GPU
timings are host-local descriptive measurements and must be rejected for performance comparison if
contention/idle-state evidence is insufficient.

#### Uncertainties

No raw metric has yet received an independent recomputation or claim disposition. The six browser
clients opened by `--open`, but visible-framebuffer/orbit smoke has not yet been attested. Reports
have not been rendered and docs have not been updated.

#### Review Focus

Use the results-audit skill. Rebind task/review/lock/source/data/cell identities; independently
recompute all 36 metrics, per-folder paired medians/wins, convergence clocks, capacities, and
resource accounting; verify 20/3/3 isolation and no held-out optimization; classify the viewer
startup retry; and confirm, narrow, or retire every prospective claim without pooling folders.

#### Protected actions not taken

The Driver has not edited raw outcomes, RESULT evidence, task/source/data protocol bytes, report
HTML, public docs, or ARA claims after outcome access. No default or SOTA/real-time claim was made.

#### Recommended Next Action

Write append-only canonical `*_AUDIT.md` and `*_AUDIT.json` with exact dispositions. Accept the
result for rendering only if all required invariants recompute; otherwise preserve the failure and
return exact corrections without mutating raw artifacts.

### Review (independent results audit)

#### Verdict

Accepted with follow-up

#### Scope

Accepted for bounded per-folder development interpretation and report rendering.

#### Self-reviewed

No

#### Correctness

All 36 measured cells and the held-out-free warmup bind to the approved task, review, source, data,
command, and lock. Strict task-specific semantic replay validates all 37 cell receipts and every
transitively hashed artifact. Independent recomputation exactly reproduces the per-view held-out
means, per-folder medians, resource mappings, clocks, and endpoint counts; validation-AUC replay
differs by at most `7.105427357601002e-15`. The 20/3/3 partitions are disjoint, training samples
use optimizer cameras only, and held-out access follows the frozen endpoint in every cell.

#### Evidence Quality

The canonical append-only audit is
`benchmarks/results/20260806_gaussian2d_image_refinement_janelle_frame00008_AUDIT.{md,json}`
(SHA-256 `f964f04f2a7ee5390e87f1da9492c219401fa33f6b2de4f07a8346a14420cd59` and
`cbf88c1c21d78c7166b34f648fd842c983816c0cda8d0bb5984667bda9585d9a`). Masked foreground,
crop, silhouette, leakage, and validation-AUC directions hold in `3/3` paired seeds separately for
every folder; full-canvas PSNR has the expected adverse direction in `3/3` seeds of every folder.
No folders were pooled. Per-view metric records, rather than held-out render pixels, are retained,
so aggregation is independently replayed but pixel-level metric kernels were not rerendered.

#### Simplicity

The disposition preserves the frozen result units and uses the preregistered paired seeds and
metrics directly. It makes no default change, adds no post-outcome threshold, and does not infer a
cross-folder treatment effect.

#### Missing Cases

The fixed-threshold/higher-capacity/boundary-aware convergence statement is untested because no
threshold crossing was frozen; it is retired for this protocol. Timings and memory remain
descriptive because host idleness/load is absent, arm order is fixed, and endpoint capacities
differ. The successful viewer retry overwrote its initial incomplete receipt, so only final
same-root reuse plus bound code/test behavior is auditable. Report rendering, manifest validation,
visible WebGL/orbit browser smoke, durable docs updates, and full repository verification remain
downstream.

#### Required Changes

Before closeout, render the frozen root and six child reports, generate/check their manifest, run
structured root/child and WebGL orbit smoke, update append-only documentation using only the audit's
bounded wording, and run repository verification. Do not rewrite producer RESULT or AUDIT evidence.

#### Optional Improvements

Future coordinators should retain append-only attempt-numbered integrity and viewer-launch receipts
so a successful same-root retry cannot overwrite the first failure chronology.

- 2026-08-07 — Volta-protocol_review → Codex-root: independent scientist pass accepted the exact
  six-folder run for bounded, per-folder development interpretation and report rendering. All 36
  raw cells replay without metric/accounting mismatch; mask-associated foreground/silhouette and
  validation-AUC directions are confirmed per folder, the full-canvas tradeoff is confirmed, the
  unoperationalized fixed-threshold/cross-folder claim is retired, and all timing/memory evidence is
  descriptive. Turn returned to the Driver for render, manifest, browser smoke, docs, and verify.

### Review (terminal closeout)

#### Verdict

Accepted

#### Scope

Terminal downstream closeout of the frozen root and six child reports, manifest, structured report
and viewer browser receipts, live HTTP/WebGL evidence, bounded documentation updates, official run
and bundle checks, and repository verification. Producer RESULT/AUDIT evidence and run outcomes
were not changed.

#### Self-reviewed

No

#### Correctness

The report set contains exactly the root plus one child for each of the six frozen folders. Every
child renders all 15 seed-level final/resource metrics for both arms and all five validation
convergence families, with the expected three seeds and 16-point validation histories. The
1,699-entry manifest has 1,695 run-scoped and four repository-scoped entries with no duplicate
identity; `experiment_contract.py check-run` reports `experiment_run: OK`, and
`check_results_bundle.py --json` reports `complete: true` with no problems.

#### Evidence Quality

`report_browser_smoke.json` binds the exact seven-page target set, 1,783 checked run-local targets,
HTTP 200 page loads, and empty client-error arrays. `viewer_smoke.json` schema v2 binds the exact six
dataset commands and child targets; Chrome 149/WebGL2 rendered 95,892--132,871 non-background
sampled pixels per viewer and changed every orbit camera. Its values agree with the six
`report_browser_smoke.json` WebGL diagnostics, including `gl_error: 0`; all seven report endpoints
on port 8766 and all six viewer endpoints on ports 8400--8405 also returned HTTP 200 during this
review. The canonical repository gate passed, the CPU full suite reached 100% with exit code zero,
and `git diff --check` passed.

#### Simplicity

The downstream presentation reads the frozen metric summaries directly and preserves each folder
as an independent result unit. The documentation repeats the six independently audited paired
medians exactly, labels root medians navigation-only, and adds no pooled estimator, post-outcome
threshold, default change, or performance/generalization claim.

#### Missing Cases

The two viewer warning strings are retained: a THREE.Clock deprecation and the explicitly named
pre-repair Gaussian-quad bounding-sphere diagnostic. They are non-blocking because WebGL2 readiness,
visible framebuffer content, orbit motion, and zero GL/client errors all pass. The immutable
producer summary still says its decision was pending independent audit; the same report and README
link the completed canonical audit, so this remains conservative provenance rather than a positive
or stale scientific claim. Timing/memory evidence remains host-local and descriptive, and the
retired fixed-threshold, cross-folder, SOTA/GPS-Gaussian, real-time, and generalization cases remain
outside this closeout.

#### Required Changes

None.

#### Optional Improvements

A future report schema could display immutable producer status and downstream audit disposition in
separate fields, and could label the final metric table explicitly as per-metric arm medians. A
future viewer stack can remove the two retained warning sources without changing this evidence.

- 2026-08-07 — Volta-protocol_review → Codex-root: terminal closeout accepted. Exact report,
  manifest, browser-receipt, live-endpoint, documentation-boundary, run/bundle, canonical-verify,
  and full-CPU checks pass. No required change remains; Turn returned to the Driver for user
  handoff.
