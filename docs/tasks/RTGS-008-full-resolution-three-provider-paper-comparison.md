# Current Task

## Title

Full-resolution three-provider paper comparison on Stage frame 00008

## Task ID

RTGS-008

## Role Assignment

- Driver: Codex-three-provider-driver
- Reviewer: Hegel
- Turn: none

## Mode

Implement

## Risk

Protected

## Maturity

- Target: Calibrated
- Reached: CPU-contracted

## Goal

Run the latest paper three-path comparison on the exact full-resolution frame-00008 fields supplied
by the owner: GaussianImage, StructSplat mask-contained, and StructSplat no-boundary. Preserve each
provider's native field semantics, hold the train/held-out split, initializer construction, compact
teacher objective, topology schedule, update budget, and evaluation fixed, and make the Stage-1
provider the only intended factor between matched provider blocks. Produce a source-bound Bundle
Contract v2 result, native-resolution previews, an interactive viewer, and an independent audit.
The protocol authority is
`experiments/tasks/20260801_paper_three_provider_fullres_stage_frame00008.json`.

## Motivation

RTGS-007 integrated the current bounded-random, Splat-SfM, and Beam Fusion paper paths, but its
development realization deliberately excluded StructSplat and stopped below calibrated evidence.
The owner has now supplied one complete GaussianImage full-resolution bundle and requested the two
corresponding latest StructSplat variants. A fresh task is required because provider identity and
blend semantics are a new experimental factor, the source bundles are external and only partly
complete, and no comparison outcome may be exposed before the matched protocol is frozen and
reviewed.

## Success Criteria

- The exact owner-supplied frame is inventoried and all three 26-view native full-resolution
  bundles complete under the shared 11,000-row maximum capacity, with per-view receipts,
  histories, QA renders, manifests, and source/config/environment provenance.
- GaussianImage additive semantics, StructSplat mask-contained normalized semantics, and
  StructSplat no-boundary normalized semantics reload and replay through the common compact-field
  interface without source RGB or masks entering reconstruction.
- The three provider bundles are copied into repository-relative immutable inputs, sealed by exact
  bytes, and checked against the common 23-train/3-heldout split.
- A provider-neutral task driver runs bounded-random, Splat-SfM, and Beam Fusion from every provider
  with a globally matched starting count, identical downstream compact training and density
  controls, three frozen seeds, fail-closed receipts, and no cross-cell overwrite.
- Every canonical cell produces initial/checkpoint/final PLYs, deterministic held-out compact
  sample metrics, density histories, native 5328-by-4608 calibrated previews, source hashes, and
  resource/stage timing.
- The canonical v2 report and synchronized viewer pass bundle validation and browser visibility
  checks; every provider/initializer/seed cell remains directly inspectable.
- An outcome-unseen prospective protocol review precedes `init-run`; an independent result audit,
  `rtgs-review`, and full repository verification precede closeout.
- Conclusions remain limited to this outcome-exposed frame and distinguish Stage-1 acquisition QA
  from downstream comparison outcomes.

## Constraints

- Use exactly
  `/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008` as the owner-supplied
  acquisition source; never alter or delete source RGB, masks, partial fields, receipts, or QA.
- Preserve provider-native rendering semantics. GaussianImage remains additive; StructSplat
  remains normalized, with mask containment or no boundary specialization exactly as named.
- Reconstruction consumes only calibration and `.rtgsv` fields. RGB/masks are permitted for the
  already-specified Stage-1 production and isolated final visual QA, never for 3D optimization.
- All provider blocks use the same views, seeds, component-selection policy, globally feasible
  initial count, optimizer, RNG domains, camera schedule, sample counts, sampler algorithms,
  density schedule, hard capacity, checkpoints, and evaluation rules. Provider-conditioned fit
  windows and proposal distributions yield provider-conditioned realized coordinates; initializer
  is the frozen within-block paper comparator.
- Do not inspect or tune against canonical comparison outcomes before prospective approval. Input
  production QA and mechanism-only preflight may be inspected and must be labelled diagnostic.
- No paper-quality, generalization, causal provider superiority, compact-VRAM, production-default,
  or cross-scene claim is authorized from this single outcome-exposed frame.
- Preserve unrelated user-owned changes in the external StructSplat checkout and repository.

## Non-Goals

- Changing GaussianImage or StructSplat fitting algorithms, defaults, safe schedules, renderer
  semantics, or the already-produced GaussianImage bundle.
- Repairing Beam Fusion, tuning an initializer per provider, or introducing COLMAP/Original 3DGS.
- Treating Stage-1 RGB/mask access as part of the compact reconstruction resource boundary.
- Selecting a preferred Stage-1 provider for production from this one frame.
- Reopening or promoting RTGS-007's unsealed development endpoint observations.

## Selected Skills

- rtgs-core
- rtgs-task-workflow
- rtgs-experiment
- rtgs-bench
- rtgs-docs-sync
- rtgs-review
- realtime-gs-results-audit
- rtgs-verify

## Experiment Contract

experiments/tasks/20260801_paper_three_provider_fullres_stage_frame00008.json

## Current Evidence

- The realtime-gs checkout is at current `origin/main` commit `4c1a7a5`; the external StructSplat
  checkout was fast-forwarded to current `origin/main` commit `fb357de`. The relevant StructSplat
  fit, renderer, initializer, and safe-schedule source files are unchanged across that update.
- All three owner-supplied arms are complete and pass the final three-arm verifier: 26 views per
  provider and 78 compact fields total. GaussianImage contains exactly 11,000 Gaussians per view,
  9,451,334 compact bytes, and mean foreground PSNR 33.9510 dB. StructSplat mask-contained contains
  exactly 11,000 Gaussians per view, 8,499,734 compact bytes, and mean foreground PSNR 29.1213 dB.
  StructSplat no-boundary contains 5,000 to 8,592 Gaussians per view, 5,419,088 compact bytes, and
  mean foreground PSNR 30.3919 dB. These are Stage-1 acquisition QA values, not downstream
  comparison outcomes.
- The three exact source directories and their producer protocol, verification record, and source
  script were copied without overwrite into the repository dataset tree and compare byte-for-byte
  with the owner-supplied source. The compact data seal binds the shared calibration, all three
  manifests and production manifests, and all 78 compact fields as 85 selected files totaling
  24,831,997 bytes; `experiment_contract.py validate-data` passes.
- All three arms use the native 5328-by-4608 canvas and a common 11,000-row capacity, but their
  renderer contracts intentionally differ: GaussianImage is additive and StructSplat is
  normalized blend. GaussianImage stores exactly 11,000 rows per view; StructSplat's dynamic safe
  schedule may stop below that shared maximum (the existing no-boundary views contain 5,016 to
  7,928 live rows).
- The producer environment recorded by the existing StructSplat receipts is Python 3.11.15,
  PyTorch 2.13.0+cu130, CUDA on an NVIDIA GeForce RTX 3050. A task-local matching environment is
  being recreated before resuming production.
- RTGS-007's reusable loader, compact observation query, bounded-random/Splat-SfM/Beam
  initializers, classic density controller, Bundle Contract v2 renderer, and viewer pipeline pass
  independent closure review. They have not yet been exercised as one provider-neutral canonical
  matrix.
- The RTGS-008 driver now binds every provider semantic, full optimizer/query/topology control,
  seed, split, data seal, task lock, output path, and v2 source artifact. Focused CPU tests validate
  the frozen controls, provider semantics, command surface, and complete ordered stage history.
- Train-only mechanism preflight passed the compact reconstruction boundary for every provider,
  excluded all three held-out views, and yielded feasible matched initializer counts of 3,293 for
  GaussianImage, 3,857 for StructSplat mask-contained, and 3,727 for StructSplat no-boundary. The
  globally shared starting count is therefore frozen at 3,293 before canonical training.
- No RTGS-008 downstream result has been run or inspected. The external Stage-1 receipts are
  outcome-exposed input-preparation evidence only.
- Goodall's first outcome-blind review rejected prospective digest `6115bb16...` with seven
  protocol blockers. The revised checkpoint resolves them by narrowing the hypothesis to
  provider-native fidelity; constructing only the selected arm in each fresh worker; freezing and
  validating the full resource schema; publishing the promised root contact sheet and
  reconstruction/orbit/elevation GIFs; requiring a synchronized comparison manifest for the
  viewer; publishing structured failure evidence; enforcing a clean, source-hashed production
  lock; and describing provider-conditioned realized sample coordinates precisely.
- The revised driver additionally publishes held-out checkpoint convergence and final compact
  model compression receipts. Exact single-arm A/A checks reproduce the global-count preflight
  initialization for every provider, initializer, and seed at 3,293 rows. Nine focused tests pass,
  and the complete repository verification gate passes at implementation checkpoint
  `40eb4c0b809c85c4e8d3669b49a34bab4860266d`.
- Hegel's outcome-blind V2 review recomputed digest `eb78742c...`, the exact 85-file seal, source
  hashes and ancestry, all compact input semantics, 53 broader focused tests, and the repository
  gate. It accepted the intended hypothesis/resource/presentation/viewer/sampling and
  convergence/compression definitions but found three remaining fail-closed blockers: incomplete
  transitive source binding, pre-`try` root failures without structured publication, and an
  unguarded canonical compact semantic load whose later receipt asserts no image access.
- The owner-authorized bounded third repair closes the first-load boundary and improves the other
  two without changing the scientific design: committed descendants are restricted to an exact
  metadata allowlist, in-`try` root phases reach the failed-bundle publisher, and the first compact
  load is guarded and receipt-bound. The independent V3 review nevertheless finds that live dirty
  execution bytes remain unbound and task/lock preconditions still precede root publication.
- Eighteen focused paper-driver/initializer tests, including real temporary-Git source mutation,
  root preflight failure publication, and fresh-process first-load image denial, pass. The full
  integrated repository gate passes after combining accepted RTGS-009/010, the optimizer
  mechanisms lineage, and RTGS-008 at implementation checkpoint
  `73dcc5f6ea4c9adb2ee99abd1184333e74b333e2`. No protected run was initialized and no downstream
  result path was opened.
- The rejected V2 artifact is preserved byte-for-byte as
  `experiments/reviews/20260801_paper_three_provider_fullres_stage_frame00008_PROTOCOL_REVIEW_V2_REJECTED.md`.
  The third prospective digest is
  `77b4e3a0cf137b21162ed9a52294aebce72a52d7fb8be0b44816640b54d02a2b`.
- Hegel's outcome-blind V3 review rejects that digest. A temporary-Git counterexample proves an
  uncommitted tracked behavior edit passes `_source_binding_passes`, and a malformed task at an
  initialized-looking exact root proves task loading can fail without failed-v2 publication. The
  first-load repair passes its fresh-process denial control. A concurrent Popper draft independently
  recorded the same findings at SHA-256 `73bcfebe...`; no extra artifact path was added because the
  frozen metadata allowlist does not authorize one.

## Minimal Plan

1. Close and archive RTGS-007 with its independent follow-up verdict; register RTGS-008 and its
   draft three-provider contract.
2. Recreate the source-matched producer environment, resume both incomplete StructSplat arms, and
   independently verify every view and manifest.
3. Import the three exact bundles into repository-relative immutable paths, seal their bytes, and
   freeze the globally matched initializer capacity from mechanism-only preflight.
4. Completed through two outcome-unseen reviews: make the paper driver provider-neutral, add
   CPU-first and guarded CUDA tests, freeze the canonical command, and audit both the original and
   revised protocol.
5. Authorized by the owner: complete one bounded third repair cycle for the three exact
   fail-closed/source-boundary gaps in Hegel's V2 review and obtain a fresh outcome-unseen review.
6. If subsequently approved, initialize the protected run, execute every
   provider/initializer/seed cell, render the v2 report/viewer, and validate all receipts.
7. Run an independent scientific audit, log only supported findings, complete repository review
   and verification, and close or hand off any maturity shortfall explicitly.

## Status

Rejected

## Human Decisions

### Question

Which full-resolution inputs should enter the latest paper comparison?

### Options

Reuse only the earlier native-additive fields; compare arbitrary available providers; or use the
exact GaussianImage, StructSplat mask-contained, and StructSplat no-boundary directories supplied
for frame 00008.

### Recommendation

Use the three exact supplied providers and preserve their native field semantics under one matched
downstream protocol.

### Decision

(Owner, in chat.) Make the latest paper comparison experiments for the frame-00008 dataset using
GaussianImage full-resolution and the StructSplat mask-contained and no-boundary variants.

### Date

2026-08-01

### Question (second prospective-review escalation)

After two consecutive outcome-blind revision verdicts, should RTGS-008 receive one bounded third
technical repair cycle for the three remaining fail-closed/source-boundary gaps, or stop without
initializing the protected run?

### Options

Authorize only: (1) complete reviewed-source-tree binding, (2) structured root failure publication
for every trusted post-initialization phase, and (3) a live guard/receipt for the canonical semantic
compact load; then require a fresh digest and outcome-unseen review. Alternatively, stop RTGS-008
at its preserved CPU-contracted protocol checkpoint and run nothing.

### Recommendation

Authorize the bounded repair only if the owner still wants this single-frame development matrix.
Do not weaken the source/input/failure claims and do not execute the current rejected protocol.

### Decision

(Owner, in chat.) Authorize the bounded third technical revision. Afterward, commit the completed
work, merge all genuinely unmerged project branches into `main`, and push `main` to the remote.

### Date

2026-08-04

### Question (post-V3 rejection)

After the owner-authorized third outcome-blind review rejected live source binding and the outer
root failure boundary, should RTGS-008 stop at its preserved CPU-contracted checkpoint, or receive
new explicit authority for another bounded repair and fresh prospective review?

### Options

Stop without initializing the protected run; or explicitly authorize only live tracked/untracked
execution-tree binding, task/lock-load failure publication, their outcome-free negative controls,
and one new outcome-unseen review.

### Recommendation

Stop the current rejected protocol. Do not begin an unbounded revision loop or initialize the run;
authorize another cycle only through a new explicit owner decision.

### Decision

Pending owner direction.

### Date

2026-08-04

## Handoff Log

Append Driver handoffs, Reviewer verdicts, and session-completion entries in chronological order.
Use `###` for entries and `####` for their fields so entries remain nested below this section.
Never delete earlier entries. On terminal closeout, archive the complete record as
`docs/tasks/<task-id>-<slug>.md`, change the archived `Turn` to `none`, and reset this file to the
unchanged template.

### Handoff (2026-08-04 prospective protocol review)

#### Objective

Independently review the frozen RTGS-008 protocol before any protected run is initialized or any
downstream comparison outcome is exposed.

#### Reviewed state

Base commit `4c1a7a53b6463473155f8c7a48be91a12de13109`; prospective protocol digest
`6115bb16b1ef170e9c93095de3a40b60f79cf6ef898b82ace86b0b56c2d76a0d`. The task, data seal,
driver, and focused-test byte digests are respectively `46f9e410...`, `645e2991...`,
`e0c8e96a...`, and `d0e57598...`.

#### Changes

Completed and independently verified all 78 Stage-1 compact fields; copied the three exact bundles
and producer sidecars byte-for-byte into repository-relative inputs; sealed 85 selected files;
froze a globally feasible 3,293-row starting count from train-only mechanism preflight; and updated
the provider-neutral driver contract and focused test.

#### Evidence

The final three-provider acquisition verifier passes. Source/destination recursive comparison
passes. `experiment_contract.py validate-data` and task validation pass. Preflight excludes the
three held-out views and passes every no-image/import negative control for all providers. Feasible
counts are GaussianImage 3,293, StructSplat mask-contained 3,857, and StructSplat no-boundary
3,727. The four focused driver tests and Ruff check pass.

#### Assumptions

The owner-selected frame and provider directories are exact. Stage-1 foreground PSNR remains
acquisition QA only. The one-frame development design cannot establish a provider ranking,
generalization, or a production default.

#### Uncertainties

The protected CUDA matrix and native-resolution presentation have not run. Resource feasibility,
cell runtime, report completeness, and native-resolution renderer behavior remain unobserved.

#### Review Focus

Check exact data and task binding, provider-native compositor semantics, train/held-out isolation,
global initializer matching, equal downstream schedules, fresh-process resource scope, failure
behavior, v2 artifact completeness, claim limits, and the absence of hidden RGB/mask access or
provider-specific fallback.

#### Protected actions not taken

No run was initialized; no warmup or measured training cell ran; no downstream metric, preview,
model, report, or result artifact exists or was inspected.

#### Recommended Next Action

Write `experiments/reviews/20260801_paper_three_provider_fullres_stage_frame00008_PROTOCOL_REVIEW.md`
bound to the exact prospective digest, recording `Outcome Access: none`. Approve only if the
protocol safely isolates the intended factors and is executable without protected edits.

### Review (2026-08-04 prospective protocol)

#### Verdict

Revision required

#### Self-reviewed

No.

#### Correctness

Goodall independently recomputed protocol digest `6115bb16...`, validated the task and exact data
seal, loaded all 78 compact fields with alpha disabled, reproduced the disjoint 23/3 split and
3,293/3,857/3,727 train-only feasibility counts, and passed 48 outcome-free focused tests plus
Ruff. Provider-native additive/normalized semantics and no-image guards are sound.

#### Evidence Quality

The input and mechanism evidence is sufficient, but the executable evidence contract is not yet
approval-ready. The provider-native held-out risk does not test the directional boundary-leakage
hypothesis; several frozen resource fields are not emitted; promised v2 root previews and complete
comparison-viewer smoke are absent; failure publication and clean reviewed-source enforcement are
incomplete; and “identical sample stream” overstates provider-conditioned coordinates.

#### Simplicity

Prefer narrowing the hypothesis to provider-native fidelity and clarifying shared RNG/algorithm
controls over adding a new image-backed common-reference boundary metric to this compact-only task.

#### Missing Cases

Resource-schema enforcement, root contact sheet and orbit/elevation GIF publication, synchronized
nine-method viewer smoke, auditable failed-run publication, production-lock rejection, exact source
hash binding, and explicit provider-conditioned sampling semantics.

#### Required Changes

Resolve all seven findings in
`experiments/reviews/20260801_paper_three_provider_fullres_stage_frame00008_PROTOCOL_REVIEW.md`,
recompute the digest, and obtain a fresh outcome-unseen prospective review. Do not initialize or
execute the protected matrix first.

#### Optional Improvements

None; the seven findings are protocol blockers.

### Handoff (2026-08-04 revised prospective protocol review)

#### Objective

Independently review the corrected, source-bound RTGS-008 protocol before any protected run is
initialized or any downstream comparison outcome is exposed.

#### Reviewed state

Implementation checkpoint `40eb4c0b809c85c4e8d3669b49a34bab4860266d`; revised prospective
protocol digest `eb78742c579930b6c04d4bd538ae33c6287ad0fb7f59190ab33adaef777a4fc1`.
The task binds the six executable/configuration files that define the canonical run by exact
SHA-256 and requires the reviewed checkpoint to be an ancestor of the execution commit.

#### Changes

Resolved all seven blockers from Goodall's first review. Added an exact single-arm canonical
initializer API so resource workers do not construct unused comparators; complete frozen
resource receipts and idle-GPU/NVML guards; native-resolution root presentation artifacts and a
synchronized nine-method viewer manifest; structured worker/root failure publication; clean
production-lock and exact source-hash enforcement; explicit provider-conditioned sampling
semantics; held-out checkpoint convergence; and compact-input-to-final-model compression
accounting.

#### Evidence

All nine provider/initializer/seed exact A/A initialization checks pass at the frozen global count
of 3,293. Nine focused contract/initializer/driver tests pass. The task and data contract validate,
the 85-file data seal remains exact, and `./scripts/verify.sh` passes in full at the implementation
checkpoint. The canonical CUDA matrix and downstream outcome remain unexecuted and unseen.

#### Assumptions

The owner-selected frame and three sealed provider directories are the intended inputs. The one
global warmup is representative enough to initialize CUDA/runtime state without becoming a
measured cell. Stage-1 acquisition QA is not a downstream comparison outcome.

#### Uncertainties

Protected CUDA feasibility, 10,000-step cell runtime, native-resolution final rendering, and
cross-cell report completeness remain unobserved. This one frame cannot establish provider
superiority, generalization, or a production default.

#### Review Focus

Recompute the revised digest and source hashes; verify each former blocker is closed; inspect
single-arm worker isolation, convergence/compression definitions, failure publication, clean-lock
behavior, resource scopes, root presentation gates, and comparison-viewer synchronization; and
confirm that no result path needs to be opened to reach the verdict.

#### Protected actions not taken

No run was initialized; no warmup or measured training cell ran; no downstream metric, preview,
model, report, or result artifact exists or was inspected.

#### Recommended Next Action

Write a new V2 prospective-review artifact bound to digest `eb78742c...` with `Outcome Access:
none`. Approve only if the revised protocol is outcome-independent, source-exact, executable, and
fail-closed; otherwise return another revision-required verdict without initializing the run.

### Review (2026-08-04 outcome-blind prospective protocol V2)

#### Verdict

Revision required

#### Self-reviewed

No

#### Correctness

The revised protocol closes most of Goodall's seven findings, but three blocking counterexamples
remain. First, `_source_binding_passes` hashes only six registered files while accepting any clean
descendant of `40eb4c0...`; behavior-bearing compact loader, initializer dependency, trainer,
density, renderer, CLI, and viewer files are unbound, so a clean unreviewed dependency change can
pass. Second, `_orchestrate` performs binding, `validate-data`, collision, and environment steps
before its structured-failure `try`; a synthetic `validate-data` failure produced no root failure
publication. Third, `_canonical_worker` performs its initial `CompactDataset.load` before the
guarded initializer/trainer calls, yet its later input receipt asserts that no RGB or mask was
opened. The current loader is safe, but the canonical guard does not establish that whole-worker
assertion.

The narrowed provider-native hypothesis, selected-arm construction, frozen resource schema and
idle guard, root contact sheet and reconstruction/orbit/elevation GIF gates, synchronized
nine-method viewer, provider-conditioned sampling wording, held-out checkpoint convergence rule,
and descriptive compact-to-model byte ratio are otherwise coherent and appropriately bounded.

#### Evidence Quality

Hegel independently recomputed protocol digest
`eb78742c579930b6c04d4bd538ae33c6287ad0fb7f59190ab33adaef777a4fc1`, all six declared source
hashes and ancestry, and all 85 sealed files totaling 24,831,997 bytes. All 78 fields reloaded with
alpha disabled under the live guard and reproduced exact provider semantics, 26 views, and the
disjoint 23/3 split. Twelve narrow and 53 broader focused tests passed. A no-CUDA-allocation NVML
smoke passed the three-sample idle and foreign-process controls. The canonical bound-environment
repository gate passed after this review record. Outcome Access remained none: no run was
initialized or invoked and no downstream path or artifact was opened, listed, or inspected.

#### Simplicity

No new method or abstraction is required. Exact tree/source-digest enforcement, moving all trusted
post-init phases inside the existing one-shot root failure publisher, and guarding or eliminating
the one pre-guard semantic load are the smallest repairs. Hegel changed no implementation source.

#### Missing Cases

The focused suite has no negative control for an unbound behavior dependency in a clean descendant,
no assertion that data/environment preflight failures publish the failed v2 sources, and no test
that the first canonical compact load is covered by the image-denial receipt. Protected CUDA
training, native presentation, and downstream bundle completeness remain deliberately unobserved.

#### Required Changes

1. Bind the complete behavior-bearing execution tree, or an equivalent exact reviewed tree digest
   that excludes only enumerated review metadata; reject any other descendant source change.
2. Publish the structured failed-run receipt and v2 diagnostic sources for every trusted phase
   after `init-run`, including sealed-data and environment/preflight failures.
3. Place the initial canonical compact semantic load under a live no-image guard (or remove the
   duplicate load) and bind its guard result into the no-image receipt.
4. Add outcome-free negative controls for all three cases, recompute the protocol digest, and seek
   explicit human authorization before any third technical review cycle.

#### Optional Improvements

None. Do not expand scientific scope, alter convergence/compression definitions, or initialize the
protected run while these three boundaries remain unresolved.

### Handoff (2026-08-04 third prospective protocol review)

#### Objective

Independently determine whether the owner-authorized bounded repair closes Hegel's three exact V2
blockers before any protected RTGS-008 run is initialized or any downstream outcome is accessed.

#### Reviewed state

Integrated implementation checkpoint `73dcc5f6ea4c9adb2ee99abd1184333e74b333e2`; prospective
protocol digest `77b4e3a0cf137b21162ed9a52294aebce72a52d7fb8be0b44816640b54d02a2b`.
The only permitted descendant changes are the eight task-declared review/ARA metadata paths.

#### Changes

Replaced the six-file source hash boundary with exact reviewed-tree ancestry plus a closed metadata
allowlist; moved binding, sealed-data, collision, environment, worker-matrix, and aggregation
phases under structured root failure publication; removed the redundant unguarded compact load;
and made the guarded first load's semantic and denial records authoritative for the worker receipt.
Preserved both prior rejected review artifacts and integrated every accepted/unmerged project
lineage without altering the RTGS-008 hypothesis, inputs, schedules, metrics, or result contract.

#### Evidence

The focused suite passes 18 tests. Negative controls reject a clean behavior-bearing descendant
and a widened allowlist, publish schema-valid failed v2 sources for binding/data/environment
failures, and make a fresh worker's first compact load fail when it attempts to open `.png`.
The 85-file data seal validates, and the complete integrated `./scripts/verify.sh` gate passes.

#### Assumptions

The exact owner-selected inputs and provider-native semantic differences remain the intended
single-frame development comparison. The enumerated post-checkpoint paths contain review/task/ARA
metadata only and must not acquire execution behavior.

#### Uncertainties

Protected CUDA feasibility, cell runtime, native-resolution result rendering, convergence, and
bundle completeness remain unobserved. This review cannot establish any provider ranking or
production default.

#### Review Focus

Recompute the digest and data seal; adversarially test the complete-tree boundary, every root
preflight failure path, and the first-load image guard/receipt; verify the integrated optimizer and
BENCH-019 changes are inside the reviewed base; and confirm that approval requires no result path.

#### Protected actions not taken

No `init-run`, canonical worker, training cell, downstream renderer, result directory listing, or
RTGS-008 outcome inspection occurred. All checks were source/input/mechanism-only.

#### Recommended Next Action

Write a fresh V3 artifact at the canonical protocol-review path with `Outcome Access: none`.
Approve only if all three V2 blockers and their negative controls are closed; otherwise preserve
the rejection and stop without initializing the protected matrix.

### Review (2026-08-04 outcome-blind prospective protocol V3)

#### Verdict

Rejected

#### Self-reviewed

No

#### Correctness

The first canonical compact load is now correctly covered by the live image-denial guard and bound
into the worker receipt. Exact reviewed-tree binding and complete root failure publication remain
incomplete. `_source_binding_passes` accepts an uncommitted behavior-bearing source edit because it
checks only `base..HEAD` and historical lock cleanliness, while `_orchestrate` still checks the
root/lock and loads task JSON before its publishing `try`; malformed task JSON produced zero root
publisher calls. Protected execution remains prohibited.

#### Evidence Quality

Hegel independently recomputed prospective digest `77b4e3a0...`, the exact 85-file seal totaling
24,831,997 bytes, and all relevant ancestry. The accepted BENCH-019 and optimizer-mechanism
lineages are inside implementation base `73dcc5f...`; Hegel's V2 artifact remains byte-preserved.
All 18 focused tests and the 59-test broader outcome-free suite pass, including
committed-descendant, in-`try` failure-publication, and first-load denial controls. Two additional
isolated negative controls reproduce the dirty-tree and pre-`try` task-load gaps. A concurrent
Popper draft independently recorded the same findings at SHA-256 `73bcfebe...`. Outcome Access
remained none.

#### Simplicity

No scientific or pipeline redesign is needed. The smallest repairs are a live worktree/index
cleanliness check at root and worker binding, plus a publisher that can record task/lock loading
failure from one outer post-root exception boundary.

#### Missing Cases

The suite lacks a negative control for uncommitted or untracked behavior bytes after `init-run`,
and lacks a root-publisher assertion for missing/corrupt task or lock metadata. Protected CUDA
feasibility, runtime, downstream presentation, and bundle completeness remain deliberately unseen.

#### Required Changes

1. Reject any live tracked or untracked execution-tree change not represented by the reviewed
   commit and permitted review metadata; do not rely only on cleanliness captured at `init-run`.
2. Publish one schema-valid failed root receipt when exact-root task or lock loading/checking fails,
   before any worker can start.
3. Add outcome-free negative controls for both cases and obtain explicit authority before any
   further prospective cycle. Do not initialize or execute the current rejected task.

#### Optional Improvements

None. Preserve the now-sound first-load repair and do not alter the scientific protocol.

### Closeout (2026-08-04 final bounded protocol disposition)

#### Outcome

Rejected. Hegel's outcome-blind V3 review at
`5c020443442664ed5c9d9e3316305810f7410398` found that dirty live behavior bytes can evade the
commit-only execution binding and that malformed task/root metadata can fail before structured
failure publication. The first-load guard/receipt repair is sound. The protected run was never
initialized; zero warmup or measured cells ran, and no downstream outcome path was accessed.

#### Integration disposition

At the owner's explicit request, the rejected RTGS-008 lineage, its sealed single-frame payloads,
accepted RTGS-009/010 BENCH-019 work, and default-off optimizer mechanisms are retained together in
`main` for reproducibility and inspection. Integration does not approve the rejected protocol,
initialize its protected run, promote a provider claim, or make any RTGS-008 method a default.

#### Recommended Next Action

Do not initialize or execute RTGS-008. Any further repair and prospective review requires a new,
explicit owner decision; otherwise treat this archived V3 rejection as the terminal disposition.
