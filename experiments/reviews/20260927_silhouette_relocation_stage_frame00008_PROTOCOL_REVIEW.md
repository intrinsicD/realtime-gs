# Prospective Protocol Review

- Task ID: `20260927_silhouette_relocation_stage_frame00008`
- Protocol SHA-256: `f53d1074c29c8ddc8614f9872a05556014ab4b99ec131d4d9ef83fb5c814f2ad`
- Reviewer: `Claude-Code-Fable-5.1-reviewer`
- Verdict: `approved`
- Outcome Access: `none`

## Scope

Round 2 of the prospective review of the RTGS-026 protocol, at commit `9f57008` on branch
`rtgs-025-main-path` (worktree clean). Round 1 rejected digest
`57370d132b561b48b5bc95b7f7c15a8bfd8b3b01c129e55d200f209ce084f76f` at commit `6016dab` for two
bounded lifecycle defects (B1 production sequencing, B2 coordinator re-entry); that record is
preserved verbatim as `..._PROTOCOL_REVIEW_V1_REJECTED.md` and the Driver's response as
`..._DRIVER_RESPONSE_R1.md`. This round reviews every change since `6016dab` (one commit, nine
files: the task JSON, driver, report module, mechanism, both test files, the task record and the two
review artifacts), recomputes the digest, re-checks the source binding and re-runs the same CPU
tests. The Driver's response was treated as a set of claims to be checked against the diff, not as
evidence.

Question (unchanged from round 1): when the masks of all 26 calibrated views of frame 00008 define a
visual hull and Gaussians that any view's 1-pixel-dilated silhouette rejects are moved into that hull
at completed steps 500, 600, ..., 7000 of an 8000-step gsplat fit at downscale 8, does colour inside
the two held-out masks (C0001, C0029) improve over the RTGS-025 masked objective alone (H1: every
paired seed at least +0.1 dB foreground PSNR with crop LPIPS no worse than +0.005)? The photograph
relocation pair and the field-versus-photograph pairs are descriptive with no verdict.

Evidence maturity this protocol may establish (unchanged): a development screen on one previously
outcome-exposed frame, one split (24 colour-training views, two held-out colour views), three paired
seeds, one fixed budget. Held-out alpha, floater and hull-rejection metrics are in-sample because all
26 masks define the hull; only held-out colour is a novel-view measurement; the first relocation event
re-places most of the random initialization onto the hull surface, so H1 measures hull-guided
placement plus ongoing relocation as one method. No default, SOTA, generalization, physical-geometry,
speed or VRAM claim can follow. The conditional all-view production model is not evidence. Approval
says only that the frozen design is fit to execute.

## Checks

Read in full: the revised task JSON; `git diff 6016dab HEAD` for every changed file; the current
`src/rtgs/optim/silhouette_relocation.py`; the driver functions touched by the revision in their full
context (`access_guard`, `prepare`, `load_hull`, `initialize`, `train_config`, `relocator_for`,
`_train`, `fit`, `evaluate`, `production`, `selftest`, `coordinate`, `main`, plus `camera_record`,
`source_guard`, `data_guard`, `snapshot_source`, `write_environment`, `preflight`,
`check_protocol_tables`); the report module's `publish` notes, RESULT writing, `_once` and
`publish_failure`; both test files; `scripts/experiment_contract.py` (`build_source_binding`,
`_source_binding_errors`, `verify_source_binding`, `init_run`, `_locked_task`, `_inventory_role`,
`_media_type`, `_inventory_descriptors`, `_render_run_v2` head, `render_run`, CLI subcommands);
`scripts/check_results_bundle.py` (`_check_receipts`, receipt names, manifest field checks);
`experiments/reviews/README.md`; the review template; `.agents/state/current-task.md`; the data
seal's path entries for the four held-out photograph and mask files (paths only); the two round-1
artifacts; the installed `gsplat/strategy/default.py` line 195 and the `gsplat-1.5.3.dist-info`
directory name.

Commands executed (all read-only or CPU unit tests):

```text
git status --short; git rev-parse HEAD; git log --oneline 6016dab..HEAD; git diff --stat 6016dab HEAD
git diff 6016dab HEAD -- <task JSON> <driver> <report> <mechanism> <both tests> .agents/state/current-task.md
git rev-parse HEAD:<each bound path> 6016dab:src/rtgs 6016dab:src/rtgs/optim/trainer.py
git status --ignored --short -- <bound patterns>
sha256sum <task JSON, data seal, driver, report, three test files, mechanism, trainer, field_targets, contract, bundle checker, pyproject, both round-1 artifacts>
.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260927_silhouette_relocation_stage_frame00008.json
.venv/bin/python scripts/experiment_contract.py validate
.venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/20260927_silhouette_relocation_stage_frame00008.json
.venv/bin/python -m pytest -q tests/test_silhouette_relocation.py tests/test_silhouette_relocation_protocol.py tests/test_conditional_density.py
```

Denied by the review sandbox and re-done in an allowed form, none touching protected material: the
pytest invocation with an environment prefix and a pipe (re-run plainly), two `git ls-tree` batches
(re-done as one `git rev-parse` with multiple `HEAD:<path>` arguments), a compound `for` loop
combining `git rev-parse`, `ls` and `grep` (split into `git rev-parse`, Glob and `git status`), and
a shell pipeline that would have rebuilt the contract's source-binding JSON from `sha256sum` and
`stat` over the 120 files. No allowed tool can run that encoding, so the aggregate was not
reproduced; the reviewed source is pinned by git object ids below and `init-run` verifies the
aggregate fail-closed (`validate_task` with `check_live_source=True`, contract 411 and 786-787;
`init_run` 1161-1216 also requires `status: ready`, a clean tracked worktree and no existing run
root, and hashes the review artifact into the lock).

Results: `review-digest` prints `f53d1074c29c8ddc8614f9872a05556014ab4b99ec131d4d9ef83fb5c814f2ad`,
the digest the Driver reported. `validate` and `validate-data` print OK; the data seal is
byte-identical to the round-1 and RTGS-025 seal (`d01254e7...`). Pytest: 22 dots, no failure or
skip, consistent with 11 mechanism, 6 protocol and 5 conditional-density tests. Worktree clean at
`9f57008`; nothing untracked or ignored under the bound patterns except `__pycache__`. No
`runs/20260927_silhouette_relocation_stage_frame00008/` root, no `benchmarks/results/20260927_*`
file and no RTGS-026 RESULT or AUDIT exists. `.scratch/` was not entered.

- Drift. The task JSON diff since `6016dab` touches exactly six places: `claim_boundary` wording for
  the photograph arms (R8); `heldout_mask_alpha` appended to `input_policy.reconstruction_allowed`
  (R5); two disclosure sentences appended to `relocation.definition` (R1); one sentence appended to
  `decision_policy.stopping` (B2); a new `production.sequence` string (B1); and the source-binding
  aggregate. Question, hypothesis, `claim_boundary` substance, datasets, splits (24 + C0001/C0029),
  seeds 9361-9363, comparators, primary metrics, charts, resource protocol, run command,
  preprocessing, hull parameters (dilation 1, threshold 0.5, near 0.001, grid 128, minimum 1000),
  relocation schedule (100/500/7000, jitter 0.5, paired seed), initialization (20000 points),
  training budget (8000 steps, gsplat-default, stop 6000, max 100000), all three resolved configs,
  scoring, the H1 gate text (+0.1 dB, +0.005 LPIPS, reverse -0.1 dB, inclusive, no floor),
  consequence, production seed/views/gate/command, execution order, warmup, RNG policy and preview
  policy are byte-identical to the rejected draft. `src/rtgs` changed by one file (the mechanism,
  +14/-9); `trainer.py` object id `81f88540` is unchanged, so the seam reviewed in round 1 is the seam
  that will run. Contract, bundle checker, `pyproject.toml`, `field_targets.py` and
  `tests/test_field_targets.py` are unchanged (object ids below).
- B1 (production lifecycle) resolved. `production.sequence` freezes option (b): run completes, the
  independent audit is persisted, the report is rendered with its viewer receipt, `check-run` and
  `check_results_bundle.py` pass on the evidence bundle, and only if `h1_relocation == pass` the
  production command runs once, followed by one rerender and both gates again; `production/` is
  never listed as evidence and never enters RESULT or AUDIT; production is a single manual
  invocation outside the 3600 s cell limit. The report notes (report 413-416) now restrict "held-out
  colour never enters" to the twelve evidence cells and describe a possible `production/` directory
  as trained on all 26 views including held-out fields and not evidence; `publish` writes them
  unconditionally (static list). Mechanics verified for option (b): `_render_run_v2` re-validates
  the run and the lock through `_locked_task`, which re-verifies the live source binding and the
  data seal (contract 2202-2208), then re-inventories every file under the root
  (`_inventory_descriptors` 3254-3289); the shared role map labels `production/*.ply` as `model`,
  `production/receipt.json` as `receipt`, `.npz`/`.json` as `artifact`, and reserves `evidence` for
  repository-scope `metrics["evidence"]` items (3213-3232), so the rendered page cannot list
  `production/` as evidence. `_check_receipts` (bundle checker 603-625) only requires the strings
  `rtgs view` and `index.html` in a receipt, so the existing viewer receipt survives a rerender.
  RESULT.md/RESULT.json are written once under `benchmarks/results/<task_id>_RESULT.*`
  (`_once`, report 38-42 and 450-470) and pin the task lock, not the manifest, so the rerender does
  not invalidate them. `production()` (driver 824-880) still refuses without
  `comparison.json` gate `pass`, reads compact views only under the `production` audit hook, and its
  `output.mkdir()` without `exist_ok` makes a second invocation fail before any write. The dead
  `timeout` branch is removed (958).
- B2 (re-entry) resolved. `coordinate` (931-934) raises `RuntimeError("... refusing re-entry")`
  when `run/targets` or `run/preparation.json` exists, before `split_digest`, before `logs.mkdir`
  and before the `try` whose `except` writes `execution_failure.json` and calls `publish_failure`.
  `main()` writes nothing before `coordinate` (`check_protocol_tables` is pure).
  `test_coordinator_refuses_a_consumed_run_root` builds a temporary root containing `targets/`,
  asserts the match and that the root's recursive listing is unchanged. `prepare` creates `targets/`
  as its first write (454) and `preparation.json` as its last (524), so any failure after `prepare`
  began leaves the guard file in place. Residual, disclosed below: a failure inside the `try` before
  `worker("prepare")` (snapshot, `source_guard`, data seal, `preflight`) leaves a root with
  `execution_failure.json`, `run_receipt.json`, `metrics.json` and `logs/` that the guard does not
  protect; the frozen stopping rule only consumes the root once prepare began.
- R1 disclosure. `relocation.definition` now states that gsplat Default refines on the iteration
  after an event using screen-gradient statistics accumulated partly at pre-relocation positions,
  and that the installed gsplat 1.5.3 never executes its opacity reset because of operator
  precedence, so `opacity_reset_every` is inert in every arm and was in RTGS-025. Verified: the
  installed package directory is `gsplat-1.5.3.dist-info`; `strategy/default.py` line 195 reads
  `if step % self.reset_every == 0 & step > 0:`, a chained comparison whose right member is `0 > 0`.
  `preflight()` records `gsplat.__version__` into `preflight.json`, so the run will carry the
  version. The repository-wide correction is correctly left outside this task.
- R2 wiring. `load_hull` passes `threshold=task["hull"]["threshold"]` (549); `SilhouetteHull`
  accepts `threshold` (mechanism 55, used at 68). `SilhouetteRelocationConfig` drops `dilation_px`
  and `near`; `relocator_for` (633-639) and the tests no longer pass them; no other caller exists
  (repository grep) and no document names the fields. Each frozen hull value now has one wiring.
  Nit: the `dilation_px >= 0` check left with the removed config field; `SilhouetteHull` does not
  validate it, but the frozen value is 1.
- R3 seal cross-check in `prepare`. For each of the 26 hull views the compact view's recorded
  `source["rgb"]["sha256"]` and `source["mask"]["sha256"]` are compared with the seal entries
  `<frame>/rgb/<view>.jpg` and `<frame>/mask/mask_<view>.png` (500-505); the seal contains those
  keys for C0001 and C0029 (seal lines 196, 276, 326, 406; paths only were read). The check reads
  `.rtgsv` metadata and the seal JSON only; no photograph or mask PNG is opened, so the `prepare`
  audit hook (433-434) is not triggered. For the 24 training views the check duplicates 468-473;
  harmless.
- R4 memory. `nearest_targets` uses 256-row `cdist` chunks (133-135); slicing is exact. The
  transient is `256 x M` float32 per chunk; M is the occupied-voxel count of the real hull (the
  Driver's smoke figure of about 21 MB implies M near 20000). A CUDA OOM at the first event would
  still fail the cell and consume the root (accepted, as in round 1).
- R5. `heldout_mask_alpha` is listed as an allowed reconstruction input beside the forbidden
  `heldout_rgb` and `heldout_fields`; consistent with the `heldout_views_contribute_packed_alpha_only`
  guard; the rendered page prints the allowed list (contract 3763). The driver does not read
  `input_policy`, so behaviour is unchanged.
- R6 tests. `test_relocation_leaves_non_position_rows_unchanged` asserts `scales` rows are
  byte-identical after an active step (only one non-position parameter is checked; the helper
  carries `means` and `scales` only). `test_seam_sees_post_density_parameters_with_classic_densification`
  runs the CPU classic controller (`every=3`, threshold `1e-9`, cap 32) for six iterations and
  asserts the callback observed a count above the initial count and that its last observation equals
  the returned model's count and the recorded history; this confirms the callback sees the rebound
  post-surgery dict. The `initialize` face and minimum-voxel aborts remain covered by reading
  (589-593) only, as the Driver states.
- R7. `SilhouetteRelocator.last_fallbacks` is reset at the start of `relocate` (143) and set to the
  number of jittered points that left the hull (151); events record `jitter_fallbacks` (159-164,
  176), zero when nothing moved. The modified event test bounds it in `[0, 2]`.
- R8. `claim_boundary` wording adopted; the stopping rule now names the re-entry refusal; the
  sequence names production as a manual invocation outside the cell limit; the dead branch is gone.
- R9. `evaluate` asserts `camera_record(view.camera) == camera_record(source.cameras[0])` for each
  held-out view (759-760), mirroring the training check (466-467); the held-out calibrated camera
  can only be loaded there because `load_source` opens the held-out photograph, so this check
  necessarily runs after the twelve cells; a mismatch aborts and consumes the root (accepted by the
  frozen stopping rule, as the packed-alpha equality check already did in round 1).
- Leakage boundary re-verified after the changes. Before `evaluate`, held-out views still
  contribute only `.camera`, `.alpha` and recorded source digests (497-511); fitting workers may open
  only their own target family, training masks, hull cache and metadata (424-428); `production`
  denies image opens (437-438); `evaluate` is the first phase to open held-out photographs and mask
  PNGs. The `selftest` still denies four probes and the protocol test asserts the count.
- Source binding. Glob confirms 108 `.py` plus 4 `.cu/.cpp/.h` under `src/rtgs` and eight named
  files, 120 in total, matching `file_count`. The aggregate `f7877b92...` was not independently
  reproduced (see the denied pipeline above) and is pinned by object ids instead.
- Task record. Driver `Claude-Code-Opus-5.5-driver`, Reviewer `Claude-Code-Fable-5.1-reviewer`,
  Turn `reviewer`; the round-1 Review entry and the revision Handoff are appended, nothing deleted.
  The task JSON still reads `status: draft` with `protocol_review` pending; the round-1 rejection
  was not recorded there as `blocked` before revision. Since the digest excludes both fields and the
  V1 record is preserved as a file, this is a procedural note, not a defect. The handoff's
  `verify.sh` and repeated GPU-smoke claims were not verified (outside the allowed command set) and
  were not used as evidence.

Reviewed identities at `9f57008` (`sha256sum`):

| File | SHA-256 |
|---|---|
| `experiments/tasks/20260927_silhouette_relocation_stage_frame00008.json` | `9e5db6918ed1ce96301a51e523bd90a79f404212e7b50177053e279464b0be93` |
| `experiments/data/20260927_silhouette_relocation_stage_frame00008.json` | `d01254e739961dcb1acc4a9abca6694bb63e319ff179581ebe2ec2ae08706087` |
| `scripts/experiments/20260927_silhouette_relocation_stage_frame00008.py` | `1957cbbbc64320630e02cf61abf2e0c8578780cd701219452361235c209b67c8` |
| `scripts/experiments/20260927_silhouette_relocation_stage_frame00008_report.py` | `8d6584206d730769f2b57c83c7be5ad5881e4a4ffc52eb014cdec06aa5a1a1d7` |
| `tests/test_silhouette_relocation.py` | `17c3a763f4b5e28fe9ff4c17566819acd95583dd5701dbefb6a7c74e1def6226` |
| `tests/test_silhouette_relocation_protocol.py` | `b5fd9d8fb646cddb487d0bb48e7cd2588fb633811942f4acf3032ec64e37426e` |
| `tests/test_field_targets.py` | `0a817166c45df13d64cd8e8f83aafbf7656e1bc37ef8bf26176bf6def75a72cc` |
| `src/rtgs/optim/silhouette_relocation.py` | `e61981e0ad00c16cc610fd6c8e367bd6cde11143869c2347ba0a6098380512b0` |
| `src/rtgs/optim/trainer.py` | `e226d01528897fd00dea8c03f345190c9abd3b4cf5cd5e2013967ca5802d39ba` |
| `src/rtgs/data/field_targets.py` | `59901d820ef291381c27cb45ba68161cdad1195eff7ea891ae281b60bca9829f` |
| `scripts/experiment_contract.py` | `15f129398aabb46a50d435c76fec27b4df6c75e6d08d2e7d78a5a96657b25959` |
| `scripts/check_results_bundle.py` | `0f5e1293a4c6651d1fed80237bd1319f44ed2ca268d81cb27ce87d2e54f2fedf` |
| `pyproject.toml` | `e5b15e72400e8e14b83a26dc64ef8a85ea985642ef9ef079e425326b5e0bc8d9` |
| `experiments/reviews/..._PROTOCOL_REVIEW_V1_REJECTED.md` | `d894f3ddc1728ee56a97ff0a0d98f50d17505db1385a3e3eed58085c6e39ac25` |
| `experiments/reviews/..._DRIVER_RESPONSE_R1.md` | `cfedf22f462ccd1fbd882537aa3278382fc5cf61957600c0c1afc73681e24935` |

Git object ids of the bound source at `9f57008` (`git rev-parse HEAD:<path>`); `unchanged` means
identical to the round-1 state `6016dab`:

| Path | Object id |
|---|---|
| `src/rtgs` (tree) | `9293a19aefd5a848227bce58630705bcb5e107dd` (was `fc4c645e30f295a8e744553c58141d0e907e3018`) |
| `src/rtgs/optim/silhouette_relocation.py` | `ddb9b5c1601c874c85aa1e6eb925a6a58902261f` |
| `src/rtgs/optim/trainer.py` | `81f88540a171907c49bd37cbff8f5b463c231e0d` (unchanged) |
| `src/rtgs/data/field_targets.py` | `80caeefc9c4f3e1ebcc4f03e39688004feeb46ee` (unchanged) |
| `scripts/experiment_contract.py` | `3d4f251972226e5372dfb5346283c65f2e73e62e` (unchanged) |
| `scripts/check_results_bundle.py` | `4be3439031998a35caf1b6d023085dccee659ceb` (unchanged) |
| `pyproject.toml` | `f3911412950125f290bc65a49818ed6949c820a7` (unchanged) |
| `scripts/experiments/20260927_silhouette_relocation_stage_frame00008.py` | `4405cc272a4050d01bee847e85ea0342eaf60e92` |
| `scripts/experiments/20260927_silhouette_relocation_stage_frame00008_report.py` | `2cd45be2966ec2d8c66a9df3688b2928867fd74d` |
| `tests/test_silhouette_relocation.py` | `ad07d0fe494a67b712caef10d1ef148fd565a9a6` |
| `tests/test_silhouette_relocation_protocol.py` | `d5b6c821ac7c8480c194ee73d3e02af0a893e61c` |
| `tests/test_field_targets.py` | `1f522a73de24a3ce91c293ee52459f9592f439cd` (unchanged) |
| `experiments/tasks/20260927_silhouette_relocation_stage_frame00008.json` | `41b05f43129cb9bc8b5eddaeffec3df6501dc3c6` |
| `experiments/data/20260927_silhouette_relocation_stage_frame00008.json` | `dc22987ddbfefac83d10deae666616451b8cff8c` (unchanged) |

## Findings

Approved. B1 and B2 are resolved exactly as specified, every adopted optional change is implemented
without a new defect, and no hypothesis, threshold, seed, split, comparator, metric, budget or
execution order moved. The scientific core accepted in round 1 is unchanged: the leakage boundary
(held-out views contribute packed alpha, cameras and recorded digests only before `evaluate`), the
hull and its aborts, the relocation mechanism and optimizer-state handling, the seam's default-path
invariance, the paired design with a shared digest-checked initialization, the mask-restricted
scoring, the inclusive per-seed relative gate, the numeric production gate, and the disclosures of
the in-sample metrics, the re-initialization effect, the pre-review smoke, the post-event refinement
statistics and the inert opacity reset. Approval says the frozen design is fit to execute; it says
nothing about the result.

Residual execution conditions:

1. Recording. Set `protocol_review` to this reviewer, verdict `approved`, digest
   `f53d1074c29c8ddc8614f9872a05556014ab4b99ec131d4d9ef83fb5c814f2ad` and artifact
   `experiments/reviews/20260927_silhouette_relocation_stage_frame00008_PROTOCOL_REVIEW.md`, set
   `status: ready`, change nothing else, and confirm `review-digest` still prints the same digest
   and `validate` prints OK. Persist this file verbatim; `init-run` hashes it into the lock and
   `source_guard` requires it byte-identical thereafter. Append the round-2 verdict to the task
   record and pass the Turn to the Driver.
2. Source state at `init-run`. Run `init-run` from a clean tree whose bound source matches the
   object ids above (`src/rtgs` tree `9293a19a...`, the eight named files as listed). `init-run`
   verifies the 120-file aggregate `f7877b92...` fail-closed; this review did not reproduce it. Any
   change to a bound file before `init-run` invalidates this approval and needs a new digest and
   review only if the task JSON changes; a bound-source change with an unchanged task JSON makes
   `init-run` refuse, and the aggregate may not be refreshed without a new review round.
3. One `run` invocation. The coordinator refuses a root containing `targets/` or `preparation.json`.
   If the coordinator fails before `prepare` begins (snapshot, `source_guard`, data seal or
   `preflight`), the root holds a failure receipt that the guard does not protect: preserve
   `execution_failure.json` and the logs (quote or copy them into the task record) before any
   re-entry, and disclose the re-entry in the task record and the RESULT. A failure after `prepare`
   began consumes the root; per Hard Rule 9 no sibling root may be created, and a repeat requires a
   new task id and review.
4. Production sequence exactly as `production.sequence`: run completed, independent audit
   persisted, report rendered with its viewer receipt, `check-run` and
   `check_results_bundle.py` passing, then and only if `h1_relocation == pass`, one production
   invocation, one `render`, both gates again. Between `init-run` and the post-production rerender
   the bound source, the task JSON (which stays `ready`; no completed status exists), the data seal
   and this review artifact must remain byte-identical, because `source_guard` and `_locked_task`
   re-verify them at production, render and check time. Bound-source fixes found during the audit
   must wait until after the rerender or forgo production. The rendered inventory will show
   `production/*.ply` with role `model` and `production/receipt.json` with role `receipt`; neither
   may be cited in RESULT, AUDIT, `ara/` or docs, and the RESULT files are write-once, so record
   the post-production rerender and its two gate outcomes in the task record's handoff log.
5. Disclosures the RESULT and AUDIT must carry: frame 00008 is outcome-exposed; held-out alpha,
   floater and hull-rejection metrics are in-sample; the first relocation event re-initializes most
   random start points onto the hull; gsplat refinement after each event uses statistics partly from
   pre-relocation positions; the installed gsplat never resets opacity, also for the RTGS-025
   baseline; `heldout_mask_alpha` is a declared reconstruction input; the pre-review smoke exposed
   relocation counts and hull size to the Driver.
6. Accepted late aborts that consume the root: the `initialize` hull-face and minimum-voxel checks,
   a CUDA OOM at the first relocation event, and the held-out camera and packed-alpha checks that can
   only run in `evaluate` after twelve cells.

Optional, not conditions: validate `dilation_px >= 0` in `SilhouetteHull` now that the config check
is gone; check at least one more non-position parameter in
`test_relocation_leaves_non_position_rows_unchanged`; the round-10 items of the V1 record (a
hull-initialized no-relocation arm; an alpha-and-camera-only `.rtgsv` reader) remain for a later
task.

## Protected Actions Not Taken

The reviewer did not run `init-run`, did not execute the driver's `run`, `prepare`, `initialize`,
`fit`, `evaluate`, `production` or `selftest` commands directly (the allowed protocol test suite runs
`selftest` as a CPU subprocess with CUDA hidden and `coordinate` on a temporary root whose guard
raises before any write), performed no GPU work, and did not enter `.scratch/` or any smoke
directory. No image, mask, `.rtgsv`, `.npz`, `.ply`, run or benchmark outcome content was opened or
printed; `validate-data` hashed sealed bytes opaquely; the only data-seal content read was four path
strings. The RTGS-025 RESULT and AUDIT records were not read; the RTGS-026 task record was read and
contains one qualitative sentence about the RTGS-025 outcome and the Driver's smoke descriptors,
which were not used as evidence. No RTGS-026 run root, result or audit exists. No file was modified,
no run root was created, no checkpoint was selected, and the owner's task record was not written.
The sandbox-denied commands listed above touched no protected material and were re-done only in the
allowed forms. This review has no outcome access.
