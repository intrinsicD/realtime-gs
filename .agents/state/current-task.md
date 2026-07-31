# Current Task

## Title

Full-resolution three-path paper pipeline from native 2D Gaussian fields through densified 3DGS

## Task ID

RTGS-007

## Role Assignment

- Driver: Codex-three-path-driver
- Reviewer: Codex-three-path-driver
- Turn: driver

## Mode

Implement

## Risk

Protected

## Maturity

- Target: Calibrated
- Reached: Pipeline-integrated

## Goal

Produce an inspectable, full-resolution development realization of the paper's three reconstruction
paths from one shared, native-additive, GaussianImage-style 2D Gaussian capture: bounded-random,
Splat-SfM, and Beam Fusion initialization. For the immediate owner-facing visual demonstration,
natively replay the sealed fields into the established full-crop 3DGS trainer and give all three
arms the same complete clone/split/prune/reset schedule. Keep the protected RGB-free compact
optimizer as a separate evidence path, record its sparse-supervision visual failure honestly, and
do not substitute the dense development path into a compact/VRAM claim. Preserve the Beam endpoint
as the starting point for a later fresh diagnosis rather than repairing it inside this comparison.
The protocol authority is
`experiments/tasks/20260730_paper_three_path_fullres_stage_frames00008_00009.json`.

## Motivation

The owner rejected the visually poor 640-component, 100-update Stage-1 smoke and clarified the
paper demonstration they expect to see. Native canvas dimensions alone are not full-resolution
quality. The current code has the component pieces, but not the promised end-to-end system:
`CompactTrainer` is fixed-topology unless an external research controller is supplied, Beam and
Splat-SfM are separate initializer functions rather than registered three-path arms, and the
capture has no real COLMAP sparse model. This task closes those integration gaps without conflating
the standard RGB trainer with the paper's compact-field-supervised reconstruction.

Remote `main` now supplies the owner-adopted A17 Experiment Bundle Contract v2 and its hardened
viewer smoke. RTGS-007 must consume that shared contract before any future official run; merging it
does not retroactively turn the outcome-exposed development artifacts into protected evidence.

## Success Criteria

- A one-view mechanism pilot establishes a visibly useful non-StructSplat native-additive Stage-1
  capacity/update setting on the native 5328-by-4608 canvas; the chosen setting is then frozen
  prospectively before all-view production.
- Both protocol frames have strict, source-bound native-additive `.rtgsv` bundles at the frozen
  high-capacity setting, complete production receipts, full-resolution render QA, and one exact
  data seal. The old 640-by-100 bundles remain labelled as superseded smoke inputs.
- One owner-facing development cell replays the sealed native fields without StructSplat or source
  RGB, completes the same established 30k full-crop 3DGS fit for Random, Splat-SfM, and Beam, and
  presents native-resolution initialization/final renders plus an interactive comparison. This
  demonstration is explicitly outside compact/VRAM evidence.
- A reusable compact density controller drives the established classic clone/split/prune/opacity
  reset surgery from point-rasterizer screen gradients, preserves Adam survivor rows and exact-zero
  newborn moments, maintains complete persistent lineage, enforces a hard count cap, and has
  CPU-first tests.
- One task driver constructs bounded-random, explicitly named Splat-SfM, and Beam Fusion
  initializations from the same train-only compact inputs, then gives all three the identical
  compact objective, sampling schedule, learning rates, SH schedule, topology policy, update
  budget, and capacity. No source RGB, mask, `SceneData`, dense trainer, or held-out field enters
  reconstruction.
- The protocol passes contract/data validation and an outcome-unseen prospective review before
  `init-run`. The canonical run produces initial/checkpoint/final PLYs, native-resolution
  calibrated renders, density histories, metrics, a three-model viewer manifest, and a complete
  Bundle Contract v2 report that passes the independent result-bundle gates.
- The Beam arm is shown without post-outcome repair or special schedule. Its visible failures and
  topology trajectory are reported descriptively, then left to a distinct successor task.
- Focused tests, `rtgs-review`, `realtime-gs-results-audit`, and `./scripts/verify.sh` pass, with
  any pre-existing host failure reproduced and scoped rather than hidden.

## Constraints

- Stage 1 is native additive GaussianImage-style fitting. StructSplat and normalized-blend fields
  are forbidden.
- Reconstruction after Stage 1 consumes only calibrated compact fields. RGB/masks may be used only
  for Stage-1 fitting and isolated final visual evaluation outside the reconstruction process.
- All arms share exact teacher bytes, training views, sample streams, optimizer controls,
  densification policy, hard final capacity, and checkpoint schedule; only initialization differs.
- Splat-SfM is labelled exactly as such. It is not a real COLMAP sparse reconstruction and must not
  be called Original 3DGS or conventional SfM.
- No official result run before prospective protocol approval; no Beam-specific fix is selected
  from this comparison's outcomes.
- Preserve the unrelated user-owned `.idea/rtgs.iml` modification and all prior experiment
  artifacts.

## Non-Goals

- Proving a paper, general VRAM, quality, speed, or cross-dataset claim from the two
  outcome-exposed stage frames.
- Implementing or installing COLMAP, inventing a sparse point model, or relabelling Splat-SfM.
- Changing the standard RGB trainer, gsplat density defaults, or production defaults.
- Continuing RTGS-006's analytic-objective comparison.
- Diagnosing, tuning, or repairing Beam Fusion after the matched endpoint is visible; that begins
  under a fresh task and protocol.

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

experiments/tasks/20260730_paper_three_path_fullres_stage_frames00008_00009.json

## Current Evidence

- The RTX 4090 host exposes 24,564 MiB and modern `gsplat 1.5.3`; an existing all-view
  native-resolution RGB run reached 100,000 3D Gaussians, so full-resolution rendering and dynamic
  cardinality are supported on this machine.
- `CompactTrainer` supports all six 3DGS parameter families, native-coordinate point supervision,
  an opt-in topology protocol, persistent IDs, Adam-boundary verification, and
  checkpoint/evaluation receipts. The new `ClassicCompactDensityController` implements production
  research cardinality changes behind that opt-in seam; fixed topology remains the default.
- `DensityController` already implements classic screen-gradient clone/split/prune, hard-cap
  enforcement, optimizer surgery, and opacity reset. `PointRenderOutput` carries the visible-row
  and retained screen-gradient data needed to drive it.
- `structure_from_splats` and `fuse_gaussian_beams` consume the same strict
  `ReconstructionInputs`. The registered legacy `SfMLifter` instead needs `SceneData.points` and
  ignores 2D fields; it is not the arm selected here.
- No `colmap` executable or cameras/images/points3D sparse model exists for the two stage frames.
- The existing native-additive bundles use only 640 Gaussians and 100 updates per view. Their
  native image dimensions are correct, but their approximately 22 dB mean foreground fit and
  visible blur make them mechanism smoke, not the requested paper-quality input.
- The replacement acquisition is complete for both frames: every view contains 100,000 native
  additive Gaussians fitted for 2,000 updates and reloads strictly at the native 5328-by-4608
  canvas. Equal-view mean foreground PSNR is 34.5186 dB on frame 00008 and 34.8426 dB on frame
  00009; the sealed compact payloads total 80,777,830 and 80,973,603 bytes respectively.
- The strict point-query development path completed 10,000 updates on frame 00008 Random and
  exercised 199 classic topology transactions (3,330 to 100,000 Gaussians, including clone,
  split, prune, and two resets), but its native render remained dark and visibly soft. It is a
  rejected integration diagnostic, not the requested endpoint.
- The established ordinary full-crop `Trainer` path is now exposed separately as
  `train-standard`: it reconstructs crop tensors natively from the sealed `provider=native`
  fields without StructSplat or source-image access, then uses the repository's proven 30k
  gsplat DefaultStrategy recipe. This path supplies the visual functionality demonstration but
  is explicitly ineligible for the compact/VRAM claim because it materializes dense tensors.
- The three train-only initializers are frozen at an exact common 3,330 rows from one identical
  field/camera boundary. Splat-SfM produced 3,330 tracks and Beam at least 5,000 components from
  the shared 2,000-row-per-view structural work subset; all downstream teacher replays retain the
  complete 100,000-row fields.
- The frame-00008/seed-300701 development cell completed the same 30,000-update standard 3DGS
  schedule for all arms. Random grew 3,330 to 28,352 rows and reached held-out crop
  PSNR/SSIM 30.4976/0.96277; Splat-SfM grew to 28,187 and reached 30.2340/0.96119; Beam grew to
  27,938 and reached 30.1512/0.96128. Random therefore slightly led this single development cell;
  Splat-SfM did not show an initializer advantage.
- Native 5328-by-4608 inspection on held-out C0014 shows a detailed Stage-1 field and recognizable,
  detail-bearing endpoints for all arms. Splat-SfM retains conspicuous floating splat fragments.
  Beam starts almost black with a diffuse low-opacity carrier and recovers through the shared fit,
  but its endpoint remains somewhat smoother and more smeared than Random. No Beam-specific repair
  was applied.
- Every standard run passed the no-image-open boundary. Splat-SfM and Beam explicitly recorded no
  loaded StructSplat modules; Random used the same native replay path and records
  `structsplat_used=false`, but completed before the explicit module-list receipt field was added.
  The standard path peaked at approximately 4.36 GB host RSS and 2.34/6.57 GB CUDA
  allocated/reserved, so none of these figures are compact/VRAM claims.
- Remote A17 adds the shared v2 report renderer, dimensioned elapsed-time histories with explicit
  stage boundaries, manifest checksums, and browser/WebGL/orbit visibility receipts. It remains
  infrastructure evidence, not an RTGS-007 scientific result.
- Focused CPU tests (74), focused CUDA renderer/query tests, Ruff, docs-sync, workflow validation,
  experiment contract/data validation, and the quick benchmark pass. Full `verify.sh` reaches the
  end of the CPU suite with exactly six frozen-harness failures caused by the host lacking the
  pinned `/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33` (the installed system version is
  `.6.0.35`); the frozen benchmark sources are unchanged and the ABI pin was not weakened.

## Minimal Plan

1. Register this task and its draft protocol; archive RTGS-006 as owner-superseded.
2. Implement and test the reusable compact classic-density controller plus shared three-initializer
   orchestration.
3. Run a bounded one-view native-additive capacity/update pilot, inspect the full-resolution
   render, and freeze the all-view Stage-1 setting.
4. Produce and seal both high-capacity full-resolution compact datasets.
5. Freeze the matched compact-training schedule and driver through mechanism tests, adopt Bundle
   Contract v2, then obtain prospective protocol approval.
6. Run the canonical three-path comparison, build and open the viewer/results page, audit the
   outcome, and hand Beam's observed problems to a new task.

## Status

In progress

## Human Decisions

### Question

Which pipeline should replace the additive analytic-objective detour?

### Options

Continue the fixed-topology analytic objective; show only the 2D bundle; or build the three
paper paths with full compact densification.

### Recommendation

Build the three paper paths and treat the old 640-by-100 inputs as smoke only.

### Decision

(Owner, in chat.) Show image to full-resolution non-StructSplat 2D Gaussian fields, then complete
3DGS fits from random and SfM initialization with full densification, and a third complete fit from
Beam Fusion initialization so the Beam problems are visible. Diagnose those problems later from a
fresh start.

### Date

2026-07-30

## Handoff Log

Append Driver handoffs, Reviewer verdicts, and session-completion entries in chronological order.
Use `###` for entries and `####` for their fields so entries remain nested below this section.
Never delete earlier entries. On terminal closeout, archive the complete record as
`docs/tasks/<task-id>-<slug>.md`, change the archived `Turn` to `none`, and reset this file to the
unchanged template.

### 2026-07-30 — Driver progress: full-resolution three-path development viewer

#### Work

Produced and sealed both 100,000-by-2,000 native additive Stage-1 captures, integrated the matched
three initializers, completed Random/Splat-SfM/Beam through the same 30k standard full-crop 3DGS
schedule on frame 00008, and generated a held-out C0014 presentation with native-resolution
initial/final renders and a synchronized interactive-viewer manifest.

#### Evidence

Random/Splat-SfM/Beam end at 28,352/28,187/27,938 Gaussians. Held-out crop PSNR is
30.4976/30.2340/30.1512 dB respectively. All generated PNGs are exactly 5328 by 4608 pixels.
Visual review confirms the nearly black Beam initialization, Splat-SfM floaters, and usable but
still smoother-than-2D-teacher endpoints. The protected compact diagnostic independently exercised
clone/split/prune/reset to 100,000 rows but remained dark/soft because 128 point queries per update
did not supply native-image detail.

#### Review state

Development visualization is ready for owner inspection. The official protocol remains draft:
there is no prospective independent approval or canonical compact run, and no compact/VRAM or
cross-scene claim is authorized. Beam diagnosis and repair remain deliberately deferred to a fresh
successor task.

### 2026-07-31 — Driver review: publishable development implementation, protected run still gated

#### Work

Reviewed the complete local implementation and development artifacts before publication, repaired
the compact-density lineage ledger so births remain traceable after later pruning, and added trainer
coverage proving historical lineage does not contaminate summaries of currently surviving rows.
The owner authorized committing the complete worktree and reconciling it with remote `main`.

#### Evidence

The focused native-observation, initializer, density, point-renderer, data-production, compact-view,
contract, and CUDA-observation test set passes (CUDA-only nodes skip when unavailable). Both exact
data seals pass `experiment_contract.py validate-data`, and every produced field still matches its
sealed bytes and production receipt.

#### Uncertainties

The live producer-source verification commands intentionally fail closed because the untracked
producer and provider sources continued to evolve after the bundles were generated. The stored
full-resolution source aggregate `fbdd7926...` differs from the current `920e3f33...`; the stored
additive aggregate `2a37b336...` differs from the current `26e499...`. Exact produced bytes,
manifests, receipts, and their seals remain preserved, but the executed dirty source bytes are
available only as stored hashes, so these development inputs are not represented as
source-replay-complete official evidence.

#### Protected actions not taken

No prospective approval, `init-run`, canonical protected execution, official result bundle,
claim/default promotion, or Beam-specific repair was fabricated. The protocol remains `draft` with
its three explicit blockers.

#### Recommended Next Action

Publish this pipeline-integrated development state as requested. If official compact evidence is
still desired, begin from an outcome-unseen review of a freshly source-bound acquisition and the
existing draft protocol rather than relabelling the development viewer as the canonical run.

### 2026-07-31 — Driver handoff: remote-main reconciliation reviewed

#### Objective

Reconcile the complete RTGS-007 development commit with the current remote `main`, resolve its
concurrent RTGS-006/A17 task history, and prepare the reviewed combined tree for direct-main
publication at the owner’s request.

#### Reviewed state

Local checkpoint `d3f73a7` diverged from remote `ed51da5` by one local and four remote commits.
Git reported two textual conflicts: the active task record and the experiment-contract tests.
Four other overlapping files merged automatically and were reviewed semantically.

#### Changes

RTGS-007 remains the one active task and now explicitly freezes Bundle Contract v2. Remote A17
reporting, stage-timeline, manifest, and visible-browser-smoke infrastructure is retained; the
local optional production-manifest data sealing is retained within it. Both branches’ RTGS-006
decisions and handoffs are preserved in the superseded archive, with their concurrent chronology
and shared-infrastructure boundary stated explicitly. The contract-test resolution retains both
the v2 fixture coverage and production-sidecar sealing coverage.

#### Evidence

The combined experiment-contract/viewer suite and the complete RTGS-007 focused suite pass. The
full non-slow CPU suite passes when exactly the six frozen historical ABI nodes are deselected.
Ruff, docs-sync, ARA, script layout, agent workflow, experiment-contract validation, both exact
data-seal validations, and `git diff --check` pass. The canonical `./scripts/verify.sh` reproduces
only those same six failures because this host lacks pinned
`/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33` and provides `.6.0.35`; no frozen harness was
weakened.

#### Assumptions

The owner’s explicit request to commit all changes, merge remote `main`, resolve conflicts, and
push authorizes direct-main publication and inclusion of the CLion metadata plus all generated
field/QA artifacts.

#### Uncertainties

The publish contains roughly 366 MB of new artifacts and may take longer than a normal source-only
push. The producer-source mismatch and draft official-protocol boundary recorded in the preceding
handoff remain unchanged.

#### Review Focus

Review the combined `scripts/experiment_contract.py`, the v2 declaration in the RTGS-007 task, the
active/archive task split, persistent density lineage, and the absence of any claim that the
development viewer is an official compact result.

#### Protected actions not taken

No prospective approval, `init-run`, canonical protected execution, official result bundle,
scientific claim promotion, default change, or Beam-specific repair was introduced during merge
resolution.

#### Recommended Next Action

Commit the reviewed merge, confirm remote `main` has not moved, fast-forward local `main`, and push
the combined history.
