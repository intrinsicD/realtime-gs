# Prospective Protocol Review

- Task ID: `20260927_silhouette_relocation_stage_frame00008`
- Protocol SHA-256: `57370d132b561b48b5bc95b7f7c15a8bfd8b3b01c129e55d200f209ce084f76f`
- Reviewer: `Claude-Code-Fable-5.1-reviewer`
- Verdict: `rejected`
- Outcome Access: `none`

## Scope

Round 1 of the prospective review of the RTGS-026 draft at commit `6016dab` (task JSON, driver,
report module, protocol tests) on top of the mechanism commit `d41eb5e`
(`rtgs.optim.silhouette_relocation`, the opt-in `Trainer.train(parameter_step_callback=...)` seam,
`tests/test_silhouette_relocation.py`). The Driver's handoff in `.agents/state/current-task.md` was
treated as a claim to be checked, not as evidence. Much of the driver is copied from the RTGS-025
driver approved in round 2 of that task's review; the copied parts were re-read here rather than
assumed correct, and the RTGS-025 RESULT and AUDIT records were not opened.

Question (frozen): when the masks of all 26 calibrated views of frame 00008 define a visual hull
and Gaussians that any view's 1-pixel-dilated silhouette rejects are moved into that hull at
completed steps 500, 600, ..., 7000 of an 8000-step gsplat fit at downscale 8, does colour inside
the two held-out masks (C0001, C0029) improve over the RTGS-025 masked objective alone (H1: every
paired seed at least +0.1 dB foreground PSNR with crop LPIPS no worse than +0.005)? The photograph
relocation pair and the field-versus-photograph pairs are descriptive with no verdict.

Evidence maturity this protocol may establish: a development screen on one previously
outcome-exposed frame, one split (24 colour-training views, two held-out colour views), three
paired seeds and one fixed budget. The user's design decisions are accepted as given and were
checked for implementation and disclosure only: single-view rejection, all 26 masks including both
held-out views' masks define the hull, floaters are moved not pruned, and a passing gate permits an
all-view production model declared not evidence. Consequences of that design that the protocol
must state, and does: held-out alpha, floater and hull-rejection metrics are in-sample; only
held-out colour is a novel-view measurement; the first relocation event re-places most of the
random initialization onto the hull surface, so H1 measures hull-guided placement plus ongoing
relocation as one method. No default, SOTA, generalization, physical-geometry, speed or VRAM claim
can follow. Approval, when given, says only that the frozen design is fit to execute.

## Checks

Read in full: the task JSON; `src/rtgs/optim/silhouette_relocation.py`; the trainer diff of
`d41eb5e` and the trainer loop around it (`trainer.py` 455-489, 656-658, 837-909, 960-984);
`rtgs.optim.strategies` (`GsplatStrategyController.post_backward`, `enforce_budget`); the installed
`gsplat/strategy/default.py` `step_post_backward` (152-201); `rtgs.core.camera` (`project`,
`in_image`, `__post_init__`); `rtgs.data.calibrated._object_bounds`; `rtgs.data.field_targets`
(`downscale_pinhole`, `decode_alpha_grid` shapes and floor indexing);
`rtgs.data.compact_views.CompactView.load` (`load_alpha` semantics); the driver, the report module
and both new test files; the full `diff` of the RTGS-025 and RTGS-026 drivers and report modules;
`scripts/experiment_contract.py` (digest scope, `TASK_STATUSES`, `init_run`, `_manifest_errors`,
`_inventory_descriptors`, `_media_type`, `_render_run_v2`); `scripts/check_results_bundle.py`
(`_check_v2_manifest`, `_check_receipts`); `experiments/README.md`, `experiments/reviews/README.md`
and the review template; `.agents/state/current-task.md`; the data seal's JSON header.

Commands executed (all read-only or CPU unit tests):

```text
git status --short; git rev-parse HEAD; git log --oneline -8
git show d41eb5e --stat; git show d41eb5e -- src/rtgs/optim/trainer.py CLAUDE.md docs/ARCHITECTURE.md
git show 6016dab --stat
git diff --stat f5ddaa2 HEAD -- src/rtgs scripts/experiment_contract.py scripts/check_results_bundle.py pyproject.toml tests/test_field_targets.py
git rev-parse HEAD:src/rtgs HEAD:<each named bound file> b79fde6:src/rtgs f5ddaa2:src/rtgs HEAD:src/rtgs/optim/trainer.py f5ddaa2:src/rtgs/optim/trainer.py
git status --ignored --short -- <bound patterns>
diff scripts/experiments/20260926_field_only_distillation_stage_frame00008.py scripts/experiments/20260927_silhouette_relocation_stage_frame00008.py
diff scripts/experiments/20260926_..._report.py scripts/experiments/20260927_..._report.py
sha256sum <task JSON, both data seals, task-owned sources, trainer, mechanism, field_targets>
ls runs/ experiments/reviews/ experiments/data/ benchmarks/results/   (names only)
.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260927_silhouette_relocation_stage_frame00008.json
.venv/bin/python scripts/experiment_contract.py validate
.venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/20260927_silhouette_relocation_stage_frame00008.json
.venv/bin/python -m pytest -q tests/test_silhouette_relocation.py tests/test_silhouette_relocation_protocol.py tests/test_conditional_density.py
```

Denied by the review sandbox and re-done in an allowed form, none touching protected material: a
`diff` over process substitutions with `sed` (re-run as a plain two-file `diff`), and a `sha256sum`
batch that included an inline Python read of the seal's key counts (re-run as `sha256sum` alone;
the seal header was read as text). The `3f0bd1b` revision named in the assignment does not exist
in this repository; `git show d41eb5e` was used instead.

Results: `review-digest` prints `57370d132b561b48b5bc95b7f7c15a8bfd8b3b01c129e55d200f209ce084f76f`,
the digest the Driver reported. `validate` and `validate-data` print OK. Pytest: 19 passed
(9 mechanism, 5 protocol, 5 conditional-density), 0 skipped, 0 failed. Worktree clean at
`6016dab`; no untracked or ignored files under the bound patterns other than `__pycache__`. No
`runs/20260927_silhouette_relocation_stage_frame00008/` root and no `benchmarks/results/20260927_*`
file exist. `.scratch/` exists and was not entered.

- Digest and drift. The digest drops only `status` and `protocol_review` (contract 247-250), so
  recording a verdict will not move it. Since the RTGS-025 approval state `f5ddaa2` (tree
  `641dbc4e`), `src/rtgs` changed by exactly two files: the new mechanism (+170 lines) and the
  trainer seam (+15 lines). `scripts/experiment_contract.py`, `scripts/check_results_bundle.py`,
  `pyproject.toml`, `tests/test_field_targets.py` and `src/rtgs/data/field_targets.py` carry the
  object ids and SHA-256 values recorded in the RTGS-025 review. The data seal is byte-identical to
  the RTGS-025 seal (`d01254e7...`), so the sealed inputs are unchanged; `validate-data` re-hashed
  them.
- Source binding. 120 files = 108 `.py` + 4 `.cu/.cpp/.h` under `src/rtgs` + the eight named
  files (Glob). The aggregate `263b011a...` was not independently reproduced (no allowed tooling);
  `init_run` validates it fail-closed through `validate_task` with its default
  `check_live_source=True` (contract 411, 786-787), requires `status: ready` (1169), refuses a dirty
  tree (1182) and an existing root, and hashes the review artifact into the lock. The registry
  `validate` does not check live source, so its OK does not confirm the aggregate.
- Trainer seam (`trainer.py` 904-909). The callback runs under `no_grad` after all optimizer
  steps (837-838), the means-LR decay (878) and the density branch (880-902), before evaluation;
  it receives the loop's live `params` dict (rebound by the classic and init controllers, mutated
  in place by gsplat) and the `optimizers` dict, and the row count is checked before and after.
  With `None` the only added instruction is the `if`.
  `test_trainer_calls_parameter_step_callback_every_iteration` asserts steps 1..5 and byte-equal
  means for `None` versus a no-op callback; `test_trainer_rejects_count_changing_callback` covers
  the guard; `test_conditional_density.py` still passes. Default-path invariance holds. Boundary
  not stated in the docstring: under `gaussian_storage_policy="geometric"` the dict holds arena
  capacity rows, so a callback would see parked rows and the guard would compare capacities; this
  protocol uses `dynamic`.
- Step alignment (checked; not disclosed in the task). `completed_step = global_it + 1`
  (656-658). The callback receives `completed_step`; the gsplat controller receives `global_it`
  (899). gsplat refines when `global_it > 500`, `global_it % 100 == 0` and
  `global_it % 3000 >= 100` (default.py 167-171 with `pause_refine_after_reset=every`), i.e. at
  global iterations 600..2900 and 3100..5900, which are completed steps 601..2901 and 3101..5901.
  Relocation fires at completed steps 500, 600, ..., 7000. So for every event from 600 to 2900 and
  from 3100 to 5900 the gsplat refinement of the same nominal step runs on the following iteration
  and selects clones and splits from `grad2d/count` accumulated over the previous 100 iterations at
  the pre-relocation positions of relocated rows; the callback cannot zero those accumulators
  because the seam does not expose strategy state. This is a property of the frozen method,
  identical for both relocation arms, and not a defect; it should be disclosed (R1). Also verified:
  in the installed gsplat the opacity-reset line reads `if step % self.reset_every == 0 & step > 0`
  (default.py 195), which Python parses as a chained comparison that is always false; the trainer
  and the dynamic controller path have no other reset (strategies 297 is the arena path). No arm
  ever resets opacity and `opacity_reset_every: 3000` is inert; this equally affected the RTGS-025
  baseline and is descriptive here (R1).
- Hull construction. `SilhouetteHull` requires one mask per camera with the camera's shape
  (61-69), binarizes at 0.5 and dilates by `2d+1` max-pooling (70-74). `_views` marks a point seen
  when `depth > near` and `in_image(uv)` (79); `in_image` uses `[0.5, W-0.5]`, so the outer
  half-pixel ring counts as out of frame (conservative: fewer rejections); `floor()` indexing
  (80-81) matches the edge-origin pixel convention of `decode_alpha_grid` (field_targets 176-177)
  and `_object_bounds` (+0.5 centres). `rejected` is true when any view sees the point outside
  (89); `supported` requires no rejection and at least one view seeing it inside (100).
  `occupied_voxels` builds the 128^3 lattice of cell centres over the cube of side `extent` about
  `center` (105-111) and raises when empty (113-114). Driver `initialize` (579-586) recovers the
  integer lattice indices exactly and aborts if any occupied voxel lies on a face or fewer than
  1000 are occupied; both aborts happen after `prepare`, so they consume the root (accepted by the
  frozen stopping rule; the Driver's smoke reports the hull built on this data). Hull masks and
  cameras come from all 26 compact views at downscale 8 (`downscale_pinhole` floor division equals
  the `decode_alpha_grid` shape). Bounds come from the 24 training cameras and thresholded soft
  masks only (494).
- Held-out contribution before evaluation (verified by reading every open). `prepare` loads each
  of the 26 `.rtgsv` files with `load_alpha=True` and uses only `.camera` and `.alpha` for the
  hull (497-505), then `del view`; no `decode_compact_view`, no `.observation` access, and no
  photograph or mask PNG for C0001 or C0029. The audit hook denies image opens for held-out stems
  in `prepare` and `initialize` (433-434) and denies every frame open in `fit` and `initialize`
  (435-436). `CompactView.load` has no camera/alpha-only mode, so the held-out colour field is
  parsed into memory but never read; that guarantee rests on code discipline and this review, not
  on the hook (R10). `initialize` opens nothing under the frame. Fitting workers may open only
  `targets/<own family>`, `targets/training_masks`, `targets/hull` and `metadata.json` (424-428);
  the `selftest` denies four probes including a `.rtgsv` and a foreign-family target, and the
  protocol test asserts the count. `evaluate` is the first phase that opens held-out photographs
  and mask PNGs (749-758) and decodes the held-out fields for the diagnostic only (760). Held-out
  cameras for the hull are taken from the compact views; training views are cross-checked against
  calibration (466-467) but the held-out ones are not (R9).
- Relocation correctness. `relocate` (139-151): rejected rows from the frozen hull in chunks of
  65536; nearest occupied voxel by exact `cdist` argmin in chunks of 1024 rows against all M
  targets (132-137); uniform jitter in `[-0.25, 0.25) * voxel` (jitter_fraction 0.5); jittered
  points that leave the hull fall back to the voxel centre, which is supported by construction
  (149-150); `means` is not modified by `relocate`. `__call__` (153-170): inactive steps return
  without an event record; active steps write the moved rows into `params["means"].data`, then zero
  the `exp_avg` and `exp_avg_sq` rows for every parameter of every param group of every optimizer
  (162-168). Adam `step` is untouched, so the first post-reset update carries the usual zero-moment
  bias (about three times the steady-state magnitude), the same behaviour gsplat applies to new
  rows. Colour, scale, rotation, opacity and count are untouched. The
  `state[key].shape[:1] == parameter.shape[:1]` test is tautological for Adam state and would index
  out of range on a non-per-Gaussian parameter; harmless for the six per-Gaussian fields. Events
  record step, count and relocated rows but not the jitter-fallback count (R7). Schedule:
  `active()` is inclusive at both ends and validated (38-45); 500..7000 gives 66 events, past the
  6000 densification stop and 1000 steps before the end. The generator is seeded with the paired
  seed on the hull device (130). Tests cover single-view rejection, behind-camera and out-of-frame
  handling, dilation, voxel support, moved rows and moments, determinism and schedule; none covers
  the mechanism together with any density controller (R6). The `dilation_px` and `near` fields of
  `SilhouetteRelocationConfig` are dead, and `task["hull"]["threshold"]` is not passed to
  `load_hull` (the class default 0.5 coincides) (R2).
- Interaction with gsplat-default densification. gsplat's parameter surgery and the trainer's
  `enforce_budget` (strategies 328-338) replace `params[name]`, `param_groups[0]["params"]` and the
  optimizer state for the new Parameter in place in the same dict and optimizer objects, so the
  callback that runs afterwards sees consistent objects; per-row strategy state is sliced on
  pruning (339-342) and untouched by relocation, which changes no rows. The relocator holds no
  stale Parameter reference (it reads `params["means"]` per call). Memory: `cdist` allocates
  `1024 x M` float32 per chunk, where M is the occupied-voxel count on a 128^3 lattice and can
  plausibly reach several hundred thousand, a transient of the order of 1-3 GB at the first event
  when most of the 20000 initial points are rejected; the Driver's smoke exercised this on the real
  hull, and a CUDA OOM would fail the cell and consume the root (R4).
- Fitting cells and paired design. `train_config` round-trips the frozen config (610-619); the
  three resolved configs differ only in `seed` (unit-tested); `nb_ms` is the RTGS-025 main-path
  trainer call with `parameter_step_callback=None`. The four cells of a seed share one random
  initialization (digest-checked, 704-708) drawn in float64 from the paired seed inside the
  training-only bound ball; `initialization.json` records the init's hull-rejected fraction per
  seed, which quantifies the disclosed re-initialization effect. Warmup, seeding and peak-reset
  order match `execution_controls` and `resource_protocol.scope`.
- Evaluation and gates. Held-out references and masks use the four-site operator and a 0.5
  binarization (751-758); colour is scored inside the mask, alpha outside the 3-px-dilated mask and
  inside the 3-px-eroded mask, LPIPS on the masked 8-px-padded crop (`mask_scores`, unchanged from
  RTGS-025); `hull_rejected_fraction` uses the same frozen hull on the final means, one value per
  cell repeated per row (778, 788). `gates()` (report 64-95) implements `h1_relocation` exactly as
  written: per seed pass iff `on.psnr >= off.psnr + 0.1 and on.lpips <= off.lpips + 0.005`,
  reverse iff `on.psnr <= off.psnr - 0.1`; all-pass gives pass, else all-reverse gives reject, else
  inconclusive; no absolute floor; the descriptive pairs carry no verdict. The frozen-gate unit
  test exercises the LPIPS-fail, small-gain and reverse cases. `decision_text` matches
  `consequence`. `SELECTED` equals `preview_policy`.
- Production gating. `production()` refuses unless `comparison.json` records
  `h1_relocation == pass` (821-823, unit-tested), reads compact views only (image opens denied,
  437-438), trains `nb_ms_reloc` on all 26 decoded fields with packed alpha from the shared
  seed-9361 initialization, and writes its receipt with a not-evidence boundary under
  `runs/<task_id>/production/`. `source_guard` requires the task to remain `ready` with bytes equal
  to the lock and the whole source snapshot unchanged, so production must run before any bound
  source changes on the branch; `TASK_STATUSES` has no completed state, so the task stays `ready`.
  Two lifecycle defects, see B1: the protocol does not say where production sits relative to
  publish, audit, `render`, the viewer smoke and the two bundle gates; `render`/`check-run` and
  `check_results_bundle.py` require `manifest.json` to inventory every file under the run root
  (contract 2293-2305; bundle checker 493-504), so a production directory written after the final
  render invalidates the bundle until `render` is rerun, and the published note "Held-out colour
  (fields and photographs) never enters fitting" (report 413) becomes false for the bundle as a
  whole without a qualifying note. The renderer accepts the new `targets/hull/voxels.pt` file
  (unknown suffixes map to octet-stream, contract 3251). The coordinator never invokes
  `production`, so its `timeout=3600` branch (948) is dead code and production has no wall limit
  (R8).
- Resources, stopping, failure. Fresh worker per cell; `fit` cells time out at 3600 s with a
  `timed_out` receipt written by the coordinator before the failure path (964-976); non-finite
  renders (`validate_render_finite`), checkpoints (648-651) and losses (673-674) abort the cell;
  every exception writes the cell receipt as failed and stops the matrix; `evaluate` refuses unless
  all 12 receipts read `completed`. Re-entry (pre-existing, deferred from the RTGS-025 review to
  this task, not adopted): a second `run` on a root where `prepare` began fails at `targets/`
  creation but, inside the `try`, `publish_failure` overwrites `run_receipt.json` (report 133-150;
  the completed-status guard is bypassed when `failed` is set) and `metrics.json` (497) and
  truncates `logs/prepare.log` (936), also for a completed and audited bundle. The production phase
  now requires a second manual driver invocation on the same root, which makes this hazard
  operational (B2).
- Disclosure. `claim_boundary` states the outcome exposure of frame 00008, the in-sample status of
  every alpha, floater and hull metric, the field-plus-silhouette nature of the field arms, the
  non-evidence status of the production model, the re-initialization effect of the first event,
  and the pre-review smoke (60 iterations, training stand-ins, no held-out colour, thresholds
  unchanged). The handoff adds that the smoke "reduced the final hull-rejected fraction", a
  mechanism descriptor from a non-protocol run, not a protocol outcome. The step-alignment
  property above is not disclosed (R1).
- Counterexamples considered. A Gaussian whose centre is inside the hull but whose footprint
  spills outside is not a floater under the centre rule and can still render alpha outside a mask
  (by design; the in-sample alpha metrics will show it). A point that no camera sees in front and
  in frame is neither rejected nor supported and is never relocated; it also renders in no view,
  so it only inflates the non-rejected denominator of `hull_rejected_fraction`. One erroneous mask
  relocates a true surface Gaussian; the tolerance is one output pixel at downscale 8, i.e. eight
  full-resolution pixels (user's design). At the first event most initial points map to the
  hull-surface voxels nearest to them; clumping is bounded by the uniform ball but not measured.
  Simpler designs that would not change the question: a precomputed nearest-occupied-voxel index
  over the lattice instead of `cdist` (R4); an alpha-and-camera-only compact reader so the
  packed-alpha-only guarantee becomes hook-enforceable (R10). A hull-initialized no-relocation arm
  would separate placement from ongoing relocation; the user's design accepts the combined
  measurement, so this is noted for a later task only (R10).
- Task record. Driver and Reviewer labels are distinct; Turn is `reviewer`; the handoff's
  `verify.sh` and GPU-smoke claims were not verified here (outside the allowed command set). The
  record's Motivation paragraph summarizes the RTGS-025 outcome in one qualitative sentence; no
  RTGS-025 RESULT or AUDIT file was opened.

Reviewed task-owned identities at `6016dab`:

| File | SHA-256 |
|---|---|
| `experiments/tasks/20260927_silhouette_relocation_stage_frame00008.json` | `30a86757cf65ba68b0ace159c389df3d19854fd4991e6a460f69bfb0ece73b21` |
| `experiments/data/20260927_silhouette_relocation_stage_frame00008.json` | `d01254e739961dcb1acc4a9abca6694bb63e319ff179581ebe2ec2ae08706087` |
| `scripts/experiments/20260927_silhouette_relocation_stage_frame00008.py` | `78cfa2e3fb009062097ce27eb53fc141c9162011e9738da8aa06fb728715c665` |
| `scripts/experiments/20260927_silhouette_relocation_stage_frame00008_report.py` | `563642589ce385b252f4a7b3ee74731865a21138c1992fffc6c964674dff150c` |
| `tests/test_silhouette_relocation.py` | `6449802495f34798a71a7a30e8afc2792a5415af0b2cbefef34e33307a4ea032` |
| `tests/test_silhouette_relocation_protocol.py` | `c5f2c7e807bde17139b45cc9111fc83c62a54bb178dcdd9312732d5a80f11468` |
| `tests/test_field_targets.py` | `0a817166c45df13d64cd8e8f83aafbf7656e1bc37ef8bf26176bf6def75a72cc` |
| `src/rtgs/optim/silhouette_relocation.py` | `3ede89965faa59f034e6611ac4e23f5f51f2c7af31818e8f2f395260ed12888b` |
| `src/rtgs/optim/trainer.py` | `e226d01528897fd00dea8c03f345190c9abd3b4cf5cd5e2013967ca5802d39ba` |
| `src/rtgs/data/field_targets.py` | `59901d820ef291381c27cb45ba68161cdad1195eff7ea891ae281b60bca9829f` |

Git object ids of the bound source at `6016dab` (`git rev-parse HEAD:<path>`):

| Path | Object id |
|---|---|
| `src/rtgs` (tree) | `fc4c645e30f295a8e744553c58141d0e907e3018` |
| `src/rtgs/optim/trainer.py` | `81f88540a171907c49bd37cbff8f5b463c231e0d` |
| `scripts/experiment_contract.py` | `3d4f251972226e5372dfb5346283c65f2e73e62e` |
| `scripts/check_results_bundle.py` | `4be3439031998a35caf1b6d023085dccee659ceb` |
| `pyproject.toml` | `f3911412950125f290bc65a49818ed6949c820a7` |
| `scripts/experiments/20260927_silhouette_relocation_stage_frame00008.py` | `9b1293f249a602a57a671d7058f8a8891d056397` |
| `scripts/experiments/20260927_silhouette_relocation_stage_frame00008_report.py` | `b4f431e2cab49df882b24b92f8e06c28637943a4` |
| `tests/test_silhouette_relocation.py` | `1ea72439ba4042d47d8069e23ae71c93899e4e86` |
| `tests/test_silhouette_relocation_protocol.py` | `a60c52d0265fcf9d4925a7e27a514d8df37161bc` |
| `tests/test_field_targets.py` | `1f522a73de24a3ce91c293ee52459f9592f439cd` |

## Findings

Rejected, for two bounded defects in the execution lifecycle of the new production phase. The
scientific core is fit to execute as implemented: the leakage boundary (held-out views contribute
packed alpha and cameras only before `evaluate`), the hull and its aborts, the relocation mechanism
and its optimizer-state handling, the seam's default-path invariance, the paired design with a
shared initialization, the mask-restricted scoring, the per-seed inclusive relative gate in written
form, the numeric production gate, and the disclosures of the in-sample metrics, the
re-initialization effect and the pre-review smoke are correct. No hypothesis, threshold, seed,
split, comparator, metric or budget needs to change.

Required changes (bounded; each alters the digest or the source binding, so one further review
round is needed):

- B1. Fix the production phase's place in the bundle lifecycle. In the task, extend `production`
  (or `decision_policy.consequence`) with the frozen sequence: either (a) production runs after
  `run` completes and before the first `render`, so the shared renderer inventories `production/`
  once, or (b) production runs after the audited bundle has passed `check-run` and
  `check_results_bundle.py` and is followed by one `render` rerun and both gates again. State that
  `production/` is never listed as evidence and that the production model may not enter RESULT or
  AUDIT comparisons. In the report module, make the notes true for the whole bundle: qualify
  "Held-out colour (fields and photographs) never enters fitting" to the twelve evidence cells, and
  add a note that a `production/` directory, if present, was trained on all 26 views including the
  held-out fields and is not evidence. Keep `publish` writing these notes unconditionally (it runs
  before production can exist).
- B2. Make the coordinator refuse re-entry before it writes anything. In `coordinate`, before the
  `try`, raise if `run/targets` or `run/preparation.json` exists, so a second `run` on a consumed
  root cannot overwrite `execution_failure.json`, `run_receipt.json` or `metrics.json` or truncate
  `logs/prepare.log` (the RTGS-025 review deferred this hardening to this task; the production
  phase now requires a second manual invocation on the same root). Add a protocol test that
  `coordinate` on a temporary root containing `targets/` raises and leaves the root unchanged.
  Opening worker logs in append mode is then optional.

Optional changes (not conditions of approval):

- R1. Disclose in `relocation.definition` or `claim_boundary` that gsplat refinement runs on the
  iteration after each relocation event and uses screen-gradient statistics accumulated at
  pre-relocation positions for relocated rows, and that the installed gsplat never executes its
  opacity reset (`default.py` 195), so `opacity_reset_every: 3000` is inert for every arm including
  the RTGS-025 baseline. The second point also belongs in the audit's reading of the density
  schedule and, later, in the ARCHITECTURE wording for gsplat Default.
- R2. Pass `threshold=task["hull"]["threshold"]` in `load_hull` and remove or use the dead
  `dilation_px` and `near` fields of `SilhouetteRelocationConfig`, so each frozen hull value has
  one wiring.
- R3. In `prepare`, compare the two held-out compact views' recorded `source["mask"]` and
  `source["rgb"]` SHA-256 with the seal (no PNG or JPEG open needed), so a conversion mismatch
  aborts before twelve fitting cells rather than at `evaluate`; keep the pixel-equality check in
  `evaluate`.
- R4. Bound the nearest-target memory: reduce the `cdist` chunk or precompute a
  nearest-occupied-voxel index over the 128^3 lattice (distance transform with indices) and look
  floaters up by lattice cell; either removes a GB-scale transient at the first event.
- R5. Name held-out packed alpha explicitly in `input_policy.reconstruction_allowed` (for example
  `heldout_mask_alpha`); the contract requires only a superset of the RGB modalities.
- R6. Tests: the seam with `densify=True` on the CPU classic controller (callback sees the
  post-surgery dict, count guard holds); `initialize`'s face and minimum-voxel aborts on a
  synthetic lattice; relocation leaves colour, scale, rotation and opacity rows byte-identical.
- R7. Record the jitter-fallback count per relocation event.
- R8. Wording: `claim_boundary` "arms are field-plus-silhouette" to "the field arms are
  field-plus-silhouette and the photograph arms photograph-plus-silhouette"; note in `stopping`
  that production is outside the 3600 s cell limit, and remove the dead `production` timeout branch
  or route production through `worker`.
- R9. In `evaluate`, assert `camera_record(view.camera) == camera_record(source.cameras[0])` for
  each held-out view, mirroring the training-view check, since the hull used the compact-view
  cameras of the held-out views.
- R10. For a later task, not this one: a hull-initialized no-relocation arm to attribute H1
  between placement and ongoing relocation; an alpha-and-camera-only `.rtgsv` reader so the
  packed-alpha-only guarantee becomes hook-enforceable.

Recording this round: change only `protocol_review` (this reviewer, verdict `rejected`, this
digest, this artifact path) and `status: blocked`; preserve this record verbatim as
`..._PROTOCOL_REVIEW_V1_REJECTED.md` when the revision lands. The next round will review the diff
from `6016dab`, recompute the digest and the source-binding count, and re-run the same CPU tests.

## Protected Actions Not Taken

The reviewer did not run `init-run`, did not execute the driver's `run`, `prepare`, `initialize`,
`fit`, `evaluate`, `production` or `selftest` commands directly (the allowed protocol test suite
runs `selftest` as a CPU subprocess with CUDA hidden; it attempts four opens under a nonexistent
scratch path, all denied), performed no GPU work, and did not enter `.scratch/` or any smoke
directory. No image, mask, `.rtgsv`, `.npz`, `.ply`, run or benchmark outcome content was opened
or printed; `validate-data` hashed sealed bytes opaquely; the only data-seal content read was its
JSON header (schema, slug, view ids, modalities). The RTGS-025 RESULT and AUDIT records were not
read; the RTGS-026 task record was read and contains one qualitative sentence about the RTGS-025
outcome and the Driver's smoke descriptors, which were not used as evidence. No RTGS-026 run root,
result or audit exists. No file was modified, no run root was created, no checkpoint was selected,
and the owner's task record was not written. The two sandbox-denied commands touched no protected
material and were re-run only in the allowed forms listed above. This review has no outcome access.
