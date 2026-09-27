# Prospective Protocol Review

- Task ID: `20260927_silhouette_relocation_index_decode_stage_frame00008`
- Protocol SHA-256: `27f7b23ac0ad4be45e22fc2f8afbd17bb223d2d1680e70dc2a336b288000f815`
- Reviewer: `Claude-Code-Fable-5.1-reviewer`
- Verdict: `approved`
- Outcome Access: `none`

## Scope

Prospective review of the RTGS-026 retry task at commit `492996f` on branch `rtgs-025-main-path`
(worktree clean). The predecessor `20260927_silhouette_relocation_stage_frame00008` was approved by
this reviewer at digest `f53d1074c29c8ddc8614f9872a05556014ab4b99ec131d4d9ef83fb5c814f2ad` (round 2,
`experiments/reviews/20260927_silhouette_relocation_stage_frame00008_PROTOCOL_REVIEW.md`, recorded
at `f88f3d0`). Its single `run` invocation failed closed inside `prepare` at the frozen
CUDA-versus-CPU decoder-parity gate, before `initialize`, before any fitting cell and before any
held-out access; the receipts are preserved under
`ara/evidence/tables/20260927_silhouette_relocation_failed_run/`. The Driver registered this retry
with the claim that the only substantive change is exact CPU-index decoding (parity against the
reference scan only, no CUDA decoder anywhere) plus disclosure text. This review treated that claim
as a hypothesis: it diffed the task JSON, driver, report module, protocol tests and data seal
byte-for-byte against the approved predecessor, recomputed the digest, checked the disclosure
against the failure receipts, checked whether seeds may be reused, checked the fitness of the CPU
index decoder, pinned the bound source, and re-examined every residual condition of the predecessor
approval. The Driver's task-record handoff was treated as a set of claims to check, not as evidence.

Question (unchanged from the predecessor): when the masks of all 26 calibrated views of frame 00008
define a visual hull and Gaussians that any view's 1-pixel-dilated silhouette rejects are moved into
that hull at completed steps 500, 600, ..., 7000 of an 8000-step gsplat fit at downscale 8, does
colour inside the two held-out masks (C0001, C0029) improve over the RTGS-025 masked objective alone
(H1: every paired seed at least +0.1 dB foreground PSNR with crop LPIPS no worse than +0.005)? The
photograph relocation pair and the field-versus-photograph pairs are descriptive with no verdict.
The one design change is that the compact fields are decoded with the exact CPU tile index instead
of the CUDA query kernel, for the training targets, the held-out field-consistency diagnostic and
production alike.

Evidence maturity this protocol may establish (unchanged): a development screen on one previously
outcome-exposed frame, one split (24 colour-training views, two held-out colour views), three paired
seeds, one fixed budget. Held-out alpha, floater and hull-rejection metrics are in-sample because
all 26 masks define the hull; only held-out colour is a novel-view measurement; the first relocation
event re-places most of the random initialization onto the hull surface, so H1 measures hull-guided
placement plus ongoing relocation as one method. No default, SOTA, generalization,
physical-geometry, speed or VRAM claim can follow. The conditional all-view production model is not
evidence. Approval says only that the frozen design is fit to execute.

## Checks

Read in full: the new task JSON (as a diff against the predecessor plus the unchanged sections
needed in context: seeds, input policy, execution guards, stages, comparators, primary metrics,
resource protocol, production sequence, execution order, execution controls, preview policy, source
lifecycle and frozen configuration); the four `git diff --no-index` outputs (task JSON, driver,
report module, protocol tests) and the data-seal diff; the new driver's `parity_record`,
`access_guard`, `prepare`, `load_hull`, `initialize`, `selftest`, `coordinate`, `worker` and
`main`, plus `load_source`, `load_view`, `random_initialization`, `snapshot_source` and the
`evaluate`/`production` decode lines; the new protocol test file; `src/rtgs/data/field_targets.py`
in full; `src/rtgs/core/observation2d.py` `GaussianObservationField.query`,
`GaussianObservationIndex` (constructor, `_build_csr`, `_component_tile_ranges`, `query`) and the
support-rectangle lines of `_cross_values`/`_paired_values`; `scripts/experiment_contract.py`
`_review_artifact_errors`, `REVIEW_FIELD_RE`, `REVIEW_SECTIONS`, `TASK_STATUSES`,
`_validate_protocol_review`, `validate_task` tail, the `validate` loop, `build_source_binding`,
`_source_binding_errors`, `verify_source_binding` and `init_run`; `tests/test_field_targets.py`
index-versus-reference, cache and CUDA tests; `experiments/README.md` lifecycle lines 38-77;
`experiments/reviews/README.md` (grep for retry guidance: none); the review template; the
predecessor review; `.agents/state/current-task.md` header and the new handoff; and the seven
files of the failed-run receipts directory. The RTGS-025 driver was grepped only for its decode
backend (`backend="cuda"` at its lines 434 and 715).

Commands executed (all read-only or CPU unit tests):

```text
git status --short; git rev-parse HEAD; git log --oneline -8; git show --stat HEAD
git diff --stat 9f57008 HEAD; git diff 9f57008 f88f3d0 -- <predecessor task JSON>
git diff HEAD~1 HEAD -- .agents/state/current-task.md
git diff --no-index <predecessor task JSON> <new task JSON>
git diff --no-index <predecessor driver> <new driver>
git diff --no-index <predecessor report> <new report>
git diff --no-index tests/test_silhouette_relocation_protocol.py tests/test_silhouette_relocation_index_decode_protocol.py
git diff --no-index <predecessor data seal> <new data seal>
git rev-parse HEAD:<each bound path and each predecessor/new artifact listed below>
git status --ignored --short -- <bound patterns, experiments/tasks, experiments/data, experiments/reviews>
sha256sum <task JSONs, data seals, drivers, reports, tests, mechanism, trainer, field_targets, contract, bundle checker, pyproject, predecessor review, seven receipt files>
head -c 600 experiments/data/20260927_silhouette_relocation_index_decode_stage_frame00008.json
ls -la ara/evidence/tables/20260927_silhouette_relocation_failed_run/; ls runs/; ls benchmarks/results/ (filtered for 20260927)
.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260927_silhouette_relocation_index_decode_stage_frame00008.json
.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260927_silhouette_relocation_stage_frame00008.json
.venv/bin/python scripts/experiment_contract.py validate
.venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/20260927_silhouette_relocation_index_decode_stage_frame00008.json
.venv/bin/python -m pytest -q tests/test_silhouette_relocation.py tests/test_silhouette_relocation_index_decode_protocol.py
```

Denied by the review sandbox and re-done in an allowed form, none touching protected material: five
compound invocations that appended an exit-code echo to the four `git diff --no-index` calls and to
a paired `review-digest` call; each was re-run as a plain single command. As in the predecessor
round, no allowed tool can produce the contract's source-binding encoding (a compact sorted JSON
array of `{path, bytes, sha256}` records, `build_source_binding` 1022-1052), and the `validate`
subcommand runs `validate_task` with `check_live_source=False` (866), so the aggregate was not
reproduced; the bound source is pinned by git object ids below, the file count was confirmed, and
`init-run` verifies the aggregate fail-closed (`validate_task` default `check_live_source=True`,
411 and 786-787, through `verify_source_binding` 1085-1101).

Results: `review-digest` prints `27f7b23ac0ad4be45e22fc2f8afbd17bb223d2d1680e70dc2a336b288000f815`
for the new task, the digest the Driver reported, and still prints `f53d1074...` for the
predecessor (its recorded approval did not move its digest, confirming that `status` and
`protocol_review` are outside the digest). `validate` prints OK; `validate-data` prints OK. Pytest:
17 dots, no failure or skip (11 mechanism, 6 protocol), matching the Driver's count. Worktree
clean at `492996f`; nothing untracked or ignored under the bound patterns, `experiments/tasks`,
`experiments/data` or `experiments/reviews` except `__pycache__`. `runs/` contains only the
consumed predecessor root; no `runs/20260927_silhouette_relocation_index_decode_stage_frame00008/`
exists, no `benchmarks/results/20260927_*` file exists, and no RTGS-026 RESULT or AUDIT exists.
`.scratch/` and the predecessor run root were not entered.

- Drift, task JSON. The diff against the approved predecessor touches exactly these places:
  `task_id` and `task_slug`; `status` back to `draft` and `protocol_review` back to
  pending/null (required for a new task); `title` suffixed "(exact CPU-index decoding retry)";
  two sentences appended to `claim_boundary` disclosing the predecessor failure; the `prepare`
  stage `purpose` (exact CPU tile index with reference-scan parity; photograph targets "for the
  control arms only"); the `data_seal` path; the three path entries of `run_command`;
  `preprocessing.target_operator`, which now names `backend="index"` and states that the CUDA
  query kernel is not used; `preprocessing.decoder_parity`, which drops the CUDA-versus-index rule,
  keeps "512 in-window sites from torch.Generator seed 926200 + training-view index; CPU index vs
  reference scan max abs <= 1e-5, else abort", and adds "No CUDA decoder is used for targets, the
  held-out field-consistency diagnostic or production"; the three path entries of
  `production.command`; and the three renamed entries plus the new aggregate of
  `source_binding` (`file_count` 120 unchanged). Every other byte is identical: question,
  hypothesis, evidence phase, datasets, splits (24 + C0001/C0029), seeds 9361-9363, input policy
  (including `heldout_mask_alpha`), execution guards, comparators, primary metrics, charts, resource
  protocol, blockers, field inputs, hull (dilation 1, threshold 0.5, near 0.001, grid 128, minimum
  1000), relocation definition and schedule, initialization (20000 points), training budget and all
  three resolved configs, scoring, the H1 gate and its consequence, production seed/views/gate and
  the option-(b) `sequence`, execution order, execution controls, RNG policy, preview policy,
  source lifecycle, live integrity policy and `report_template_version: 2`.
- Drift, driver. Five hunks and nothing else: the module docstring; `parity_record` (261-275),
  which no longer constructs a `"cuda"` query backend and now computes only
  `index_vs_reference` with the same 512 sites, the same generator seeding and the same 1e-5
  abort; `prepare` line 482, `evaluate` line 767 and `production` line 841, each
  `backend="cuda"` replaced by `backend="index"`. `PARITY_SITES = 512` (63) is unchanged. A
  case-insensitive grep for `cuda` in the new driver finds only device placements (LPIPS, the
  warmup render, preflight, the environment receipt, the relocator, the hull, the gsplat renderer
  and seeding); `make_query_backend` is called once, with `"index"`. `_resolve_backend`
  (field_targets 89-96) returns an explicit backend unchanged, so nothing auto-resolves to CUDA.
  The access guard, the seal cross-checks, the packed-alpha equality and fit-window checks, the
  hull loop, `initialize`, `train_config`, `relocator_for`, `fit`, `evaluate`, `production`,
  `selftest`, the re-entry refusal (932-934), the fit-only 3600 s timeout (958), the canonical-root
  check in `main` (1025-1026) and `check_protocol_tables` are byte-identical to the approved
  driver.
- Drift, report and tests. The report module differs only in its docstring; it carries no static
  note about the decoder (grep for parity/decode finds nothing; the only "CUDA" string is the
  descriptive peak-memory chart title), so no stale disclosure survives. The protocol test differs
  only in `TASK_ID`, so the same six contracts (tables/split/hull views, resolved configs differ
  only in seed, inclusive per-seed gate, production requires a passing gate, selftest denies four
  probes, coordinator refuses a consumed root) bind the new driver. The mechanism test file is
  unchanged.
- Drift, data seal and bound source. The new data seal is byte-identical to the predecessor and
  RTGS-025 seal (`d01254e7...`, object `dc22987d`); its header carries `data_slug`, not a task id,
  so sharing is legitimate. The `src/rtgs` tree object `9293a19a...` is unchanged since `9f57008`,
  the commit this reviewer approved, so the mechanism, the trainer seam and `field_targets.py` that
  will run are the ones already reviewed. Contract, bundle checker, `pyproject.toml`,
  `tests/test_field_targets.py` and `tests/test_silhouette_relocation.py` are unchanged. Glob
  confirms 108 `.py` plus 4 `.cu/.cpp/.h` under `src/rtgs` and eight named files, 120 in total,
  matching `file_count`. The predecessor task JSON changed only by the recording of its approval
  (`git diff 9f57008 f88f3d0`), and neither it nor its driver, report, tests or review artifact
  changed in `492996f`.
- Failure disclosure versus receipts. `run_receipt.json`: status `failed`, `failure_phase`
  `execution`, exit code 1, started 2026-09-27T12:20:08Z, finished 12:20:50Z.
  `execution_failure.json`: `CalledProcessError` from the `prepare` worker raised in `coordinate`
  (predecessor driver line 968 via `worker` 947), split digest `0ab039b7...`.
  `prepare_log_tail.txt`: `prepare train 14/24 C0026`, abort in `parity_record` (predecessor line
  274) with `index 1.7881393432617188e-07, cuda 0.00018894672393798828`. `preflight.json` shows
  the preflight passed (RTX 3050, gsplat 1.5.3, LPIPS probe). `environment.json`: Python 3.12.9,
  torch 2.9.0, numpy 2.1.3, gsplat 1.5.3, lpips 0.1.4, CUDA 12.8. `launch_context.json` records a
  shared GPU, consistent with the descriptive-only resource scope. The `claim_boundary` sentence
  ("1.889e-4 at 1 of 512 sites of training view C0026, a near-empty normalized-blend region, before
  any initialization, fitting cell or held-out access") agrees with these receipts: C0026 is the
  1-based 14th training view, so index 13 and seed 926213 as the README states; in the predecessor
  `prepare` the hull loop that opens held-out compact views for packed alpha follows the training
  loop, `preparation.json` was never written, and `initialize` never ran, so the code order plus
  the log support "no held-out access" and "no reconstruction outcome". The "1 of 512" count and
  the weight sums (4.19e-7 CPU versus 4.24e-7 CUDA) come from the Driver's post-failure diagnosis
  on a training view, and the JSON correctly attributes them as such rather than as receipts. The
  README's statement that the lock was at `f88f3d0` is consistent with the approval commit. The
  failed run wrote decoded targets for 13 training views under the consumed predecessor root;
  those are training-view inputs, not outcomes, and the new task's root is different.
- Seeds. Seeds 9361-9363 are first consumed in `initialize` (`random_initialization`) and `fit`;
  neither ran. The parity seeds `926200 + index` are deterministic gate constants: given the frozen
  fields they produce the same sites every time, and neither the seed base nor the 1e-5 threshold
  moved. No fitting outcome, held-out score or model exists that could have informed a seed choice,
  and the paired design needs identical seeds across cells only within this task. Reuse is sound.
- CPU index decoder fitness. `GaussianObservationIndex` is the CPU reference backend by
  construction (it refuses non-CPU fields, observation2d 1008-1009; `make_query_backend` moves the
  field to CPU, field_targets 115). Its `query` (1275-1321) evaluates the same `_paired_values`
  as the all-component scan, whose weights use the same clipped support rectangle
  (`support_centers ± radii`, `radii = sigma_cutoff * sqrt(var)`, lines 299 and 908-916), the same
  amplitude and valid-domain factors, and the same normalized blend
  `numerator / (denominator + epsilon)` (1312-1313 versus 524-525). `_component_tile_ranges`
  (1148-1182) enumerates every component with positive amplitude whose clipped support rectangle
  overlaps a tile, computed from the same centres and radii in float64, so every component with a
  non-zero weight at a site is visited; components are visited in ascending id order within each
  tile row (stable sort, 1079-1081) and accumulated in float32 by `index_add_`. The only
  difference from the reference scan is float32 summation order, which the bound CPU test bounds at
  `atol=1e-6` (test_field_targets 69-80) and the failed run measured at 1.79e-7 on the real C0026
  view. `decode_field_grid` (126-161) is backend-agnostic in everything else: it queries only
  in-window sites on the CPU, clamps, box-averages `supersample²` sites and returns the same
  `color`/`coverage`, so target semantics, the fit-window clip check and the teacher PSNR are
  defined exactly as before. The CUDA extension is imported only when the resolved name is
  `"cuda"` (121-123), so no CUDA decoder code runs.
- Gate semantics. The retry does not weaken a gate after seeing it fail: the removed
  CUDA-versus-index rule existed to license the CUDA decoder, which is no longer used anywhere;
  the retained index-versus-reference rule is the stricter of the two original tolerances and
  still aborts `prepare` fail-closed. With `epsilon` 1e-8 and normalized blending, a near-empty
  site could in principle also expose summation-order differences between the index and the scan;
  if that ever exceeded 1e-5 the run would fail closed exactly as the predecessor did, and the
  Driver's disclosed smoke (all 24 views at or below 1.8e-7) makes that unlikely. No threshold was
  tuned to the smoke.
- Cache inertness. `decode_compact_view` keys its optional on-disk cache without the backend
  (182-193, 207-209), which would matter if a CUDA-decoded cache from the predecessor could be
  served to this run; it cannot, because the driver never passes `cache_dir` (grep: no
  occurrence), so `path` is `None` and every decode is recomputed (215-239). The docstring claim
  that CUDA and CPU index results agree to float32 rounding is contradicted at near-empty sites by
  the predecessor receipts; `field_targets.py` is bound, so that correction belongs to a later
  task (see conditions).
- Timeouts and resources. The coordinator applies the 3600 s limit to `fit` workers only (958);
  `prepare`, `initialize`, `evaluate` and `production` have no timeout, so slower CPU decoding of
  24, 2 and 26 views cannot produce a spurious time-out, and the resource protocol times decoding
  separately and is descriptive only. `torch.set_num_threads(2)` and the thread environment are
  unchanged.
- Predecessor state, dependencies and retry policy. `TASK_STATUSES` is `draft`/`ready`/`blocked`
  and an approved review requires `ready` (contract 61, 401), so the predecessor task correctly
  remains `ready` and immutable; its failure lives in the run receipts and the evidence README, as
  `experiments/README.md` lines 64-65 prescribe. `depends_on: []` is correct: `init_run` requires
  every dependency to be a complete canonical run (1174-1180), which the failed predecessor is not.
  The README's `attempts/` mechanism (66-68) is for unchanged tasks; this task changed the decoder
  and parity rule, so "task changes after a run starts require a new task id" applies, which is
  what the Driver did and what condition 3 of the predecessor approval required. The predecessor's
  coordinator refuses re-entry on its consumed root in any case.
- Second pre-review smoke. The task record discloses a non-protocol `prepare` plus `initialize`
  smoke over the frozen split that exposed index parity values for all 24 views, the hull voxel
  count (21391) and, by the code path, the per-view teacher PSNR and each seed's initial
  hull-rejected fraction. None of these is a fitting outcome; the hull size was already disclosed
  in `claim_boundary`; nothing in the frozen design depends on their values. The smoke's mechanism
  is not verifiable from tracked state (the driver's `main` allows only the canonical root, so it
  must have used a scratch task id or a programmatic call), but the canonical root of this task
  does not exist, which is what `init-run` requires. The task JSON does not name this second smoke;
  since the digest is otherwise final, it is carried as a disclosure condition below rather than a
  reason to reject.
- Task record. Driver `Claude-Code-Opus-5.5-driver`, Reviewer `Claude-Code-Fable-5.1-reviewer`,
  Turn `reviewer`, Status `In review`, Experiment Contract pointing at the new task, the new
  handoff appended with the digest, the aggregate, the failure summary, the seed-reuse assumption
  and the open uncertainty about the CUDA kernel; nothing deleted. The handoff's `verify.sh`
  claim was not verified (outside the allowed command set) and was not used as evidence.
- Leakage boundary. Unchanged from the approved driver: before `evaluate`, held-out views
  contribute only `.camera`, `.alpha` and recorded source digests; fitting workers may open only
  their own target family, the training masks, the hull cache and metadata; `production` denies
  image opens; `evaluate` is the first phase to open held-out photographs and mask PNGs; the
  `selftest` still denies four probes and the protocol test asserts the count.

Reviewed identities at `492996f` (`sha256sum`):

| File | SHA-256 |
|---|---|
| `experiments/tasks/20260927_silhouette_relocation_index_decode_stage_frame00008.json` | `3b54aaeec8ec60917aa7a553883c7f9f8f69611b8dc6b10c0ba1d84e97d43a96` |
| `experiments/data/20260927_silhouette_relocation_index_decode_stage_frame00008.json` | `d01254e739961dcb1acc4a9abca6694bb63e319ff179581ebe2ec2ae08706087` (identical to the predecessor seal) |
| `scripts/experiments/20260927_silhouette_relocation_index_decode_stage_frame00008.py` | `d2f68c21d3ac1df9c1a9d2e6fc49e6071c173c34295664d692e682c790e9866c` |
| `scripts/experiments/20260927_silhouette_relocation_index_decode_stage_frame00008_report.py` | `7c234f01120e92afaa2f8030cefb701ef4f48373984c4405575c486fc3fc7d73` |
| `tests/test_silhouette_relocation_index_decode_protocol.py` | `57202cdb9c760ff3283f8b7aa7a6afe637a0d15b25461858f7835a2a58292f3e` |
| `tests/test_silhouette_relocation.py` | `17c3a763f4b5e28fe9ff4c17566819acd95583dd5701dbefb6a7c74e1def6226` (unchanged) |
| `tests/test_field_targets.py` | `0a817166c45df13d64cd8e8f83aafbf7656e1bc37ef8bf26176bf6def75a72cc` (unchanged) |
| `src/rtgs/optim/silhouette_relocation.py` | `e61981e0ad00c16cc610fd6c8e367bd6cde11143869c2347ba0a6098380512b0` (unchanged) |
| `src/rtgs/optim/trainer.py` | `e226d01528897fd00dea8c03f345190c9abd3b4cf5cd5e2013967ca5802d39ba` (unchanged) |
| `src/rtgs/data/field_targets.py` | `59901d820ef291381c27cb45ba68161cdad1195eff7ea891ae281b60bca9829f` (unchanged) |
| `scripts/experiment_contract.py` | `15f129398aabb46a50d435c76fec27b4df6c75e6d08d2e7d78a5a96657b25959` (unchanged) |
| `scripts/check_results_bundle.py` | `0f5e1293a4c6651d1fed80237bd1319f44ed2ca268d81cb27ce87d2e54f2fedf` (unchanged) |
| `pyproject.toml` | `e5b15e72400e8e14b83a26dc64ef8a85ea985642ef9ef079e425326b5e0bc8d9` (unchanged) |
| `experiments/tasks/20260927_silhouette_relocation_stage_frame00008.json` | `92d569e6046d5252c43074bc162090086e3f633a413c385e3fbe161eb59d282a` (approval recorded at `f88f3d0`; unchanged since) |
| `scripts/experiments/20260927_silhouette_relocation_stage_frame00008.py` | `1957cbbbc64320630e02cf61abf2e0c8578780cd701219452361235c209b67c8` (unchanged) |
| `scripts/experiments/20260927_silhouette_relocation_stage_frame00008_report.py` | `8d6584206d730769f2b57c83c7be5ad5881e4a4ffc52eb014cdec06aa5a1a1d7` (unchanged) |
| `tests/test_silhouette_relocation_protocol.py` | `b5fd9d8fb646cddb487d0bb48e7cd2588fb633811942f4acf3032ec64e37426e` (unchanged) |
| `experiments/reviews/20260927_silhouette_relocation_stage_frame00008_PROTOCOL_REVIEW.md` | `fcab35ad92dc69466d83ffe8a0841e029023adad98723e4ca8e13c6f4170f016` |
| `ara/evidence/tables/20260927_silhouette_relocation_failed_run/README.md` | `f654430b2ee8787374af6e9d45554de42b0d8da2bd026d0ce87ccae7c63f5607` |
| `.../failed_run/execution_failure.json` | `5140e51be2e062ad0fa3ca50d89d99cfae79460167224e4c759790a85f175067` |
| `.../failed_run/prepare_log_tail.txt` | `26da903935221cbaccde9d2a6fa47712235bbe61d4085509e584ba1f70d0cc54` |
| `.../failed_run/run_receipt.json` | `929b5e4dbee326a517acb9ce6c5fb195429af6d6331e344ac8bc084beb3f705d` |
| `.../failed_run/environment.json` | `48df639af890ab9bd4645e34aee095f398fedeb311b698aa11607f97bf89941f` |
| `.../failed_run/preflight.json` | `7be5e14ef5e95f04670d3a23ee03ac58ee51bb96be687770996b83007ea93c4e` |
| `.../failed_run/launch_context.json` | `1e6084858a89609339291df9c57fc4664bc684e9071f78d952528ddc05c925b1` |

Git object ids of the bound source at `492996f` (`git rev-parse HEAD:<path>`); `unchanged` means
identical to the approved predecessor state `9f57008`:

| Path | Object id |
|---|---|
| `src/rtgs` (tree) | `9293a19aefd5a848227bce58630705bcb5e107dd` (unchanged) |
| `src/rtgs/optim/silhouette_relocation.py` | `ddb9b5c1601c874c85aa1e6eb925a6a58902261f` (unchanged) |
| `src/rtgs/optim/trainer.py` | `81f88540a171907c49bd37cbff8f5b463c231e0d` (unchanged) |
| `src/rtgs/data/field_targets.py` | `80caeefc9c4f3e1ebcc4f03e39688004feeb46ee` (unchanged) |
| `scripts/experiment_contract.py` | `3d4f251972226e5372dfb5346283c65f2e73e62e` (unchanged) |
| `scripts/check_results_bundle.py` | `4be3439031998a35caf1b6d023085dccee659ceb` (unchanged) |
| `pyproject.toml` | `f3911412950125f290bc65a49818ed6949c820a7` (unchanged) |
| `scripts/experiments/20260927_silhouette_relocation_index_decode_stage_frame00008.py` | `83d8a474d93ba77729b0f46e37431da8df0ef277` |
| `scripts/experiments/20260927_silhouette_relocation_index_decode_stage_frame00008_report.py` | `1dc3ef43d594780584472f317888b8bd8b33b8b3` |
| `tests/test_silhouette_relocation.py` | `ad07d0fe494a67b712caef10d1ef148fd565a9a6` (unchanged) |
| `tests/test_silhouette_relocation_index_decode_protocol.py` | `edaecfde9fca82f7468bd6b1c98a10a0f30cf067` |
| `tests/test_field_targets.py` | `1f522a73de24a3ce91c293ee52459f9592f439cd` (unchanged) |
| `experiments/tasks/20260927_silhouette_relocation_index_decode_stage_frame00008.json` | `a9b79b5312a5ac335dac0f9d910222ae3119b00b` |
| `experiments/data/20260927_silhouette_relocation_index_decode_stage_frame00008.json` | `dc22987ddbfefac83d10deae666616451b8cff8c` (same object as the predecessor seal) |
| `experiments/tasks/20260927_silhouette_relocation_stage_frame00008.json` | `623acaf43b32a7a339e01d6a458085560c3183d8` (was `41b05f43...` before its approval was recorded) |
| `scripts/experiments/20260927_silhouette_relocation_stage_frame00008.py` | `4405cc272a4050d01bee847e85ea0342eaf60e92` (unchanged) |
| `scripts/experiments/20260927_silhouette_relocation_stage_frame00008_report.py` | `2cd45be2966ec2d8c66a9df3688b2928867fd74d` (unchanged) |
| `tests/test_silhouette_relocation_protocol.py` | `d5b6c821ac7c8480c194ee73d3e02af0a893e61c` (unchanged) |
| `experiments/reviews/20260927_silhouette_relocation_stage_frame00008_PROTOCOL_REVIEW.md` | `8f6255e41f810e94df28510fe1d14fee5debd0b7` |

## Findings

Approved. The Driver's claim is verified at the byte level: apart from identifiers, paths, the
review-state reset and disclosure text, the retry differs from the approved predecessor only in
decoding the compact fields with the exact CPU tile index instead of the CUDA query kernel and in
dropping the now-moot CUDA parity rule while keeping the stricter index-versus-reference abort. The
mechanism, trainer seam, decoder module, contract, bundle checker, data seal, question,
hypothesis, thresholds, seeds, split, comparators, metrics, budget, execution order, leakage
boundary and production lifecycle are the ones approved at `9f57008`. The failure is disclosed in
the task JSON consistently with the preserved receipts, the predecessor root was consumed and not
re-entered, and the retry uses a new task id as the repository lifecycle and condition 3 of the
predecessor approval require. Seeds may be reused because no initialization, fit, held-out score
or model exists from the predecessor. The CPU index decoder is fit for purpose: it is the
repository's exact CPU reference backend, evaluates the same weights and normalized blend as the
reference scan over the same clipped supports, differs only by float32 summation order, and is
gated at 1e-5 on 512 fixed sites per view. Approval says the frozen design is fit to execute; it
says nothing about the result.

Residual execution conditions:

1. Recording. Set `protocol_review` to this reviewer, verdict `approved`, digest
   `27f7b23ac0ad4be45e22fc2f8afbd17bb223d2d1680e70dc2a336b288000f815` and artifact
   `experiments/reviews/20260927_silhouette_relocation_index_decode_stage_frame00008_PROTOCOL_REVIEW.md`,
   set `status: ready`, change nothing else, and confirm `review-digest` still prints the same
   digest and `validate` prints OK. Persist this file verbatim; `init-run` hashes it into the lock
   and `source_guard` requires it byte-identical thereafter. Append the verdict to the task record
   and pass the Turn to the Driver. Leave the predecessor task JSON, its review artifacts and the
   failed-run receipts untouched.
2. Source state at `init-run`. Run `init-run` from a clean tree whose bound source matches the
   object ids above (`src/rtgs` tree `9293a19a...`, the eight named files as listed). `init-run`
   verifies the 120-file aggregate `8dacd07f...` fail-closed; this review did not reproduce it.
   Any change to a bound file before `init-run` invalidates this approval: a bound-source change
   with an unchanged task JSON makes `init-run` refuse, and the aggregate may not be refreshed
   without a new digest and review round. In particular, do not correct the `decode_compact_view`
   cache docstring or key, the CUDA query kernel's near-zero-weight normalization, or the bound
   CUDA-versus-index test tolerance before the run; those belong to a later task.
3. One `run` invocation. The coordinator refuses a root containing `targets/` or
   `preparation.json`. If the coordinator fails before `prepare` begins (snapshot, `source_guard`,
   data seal or `preflight`), preserve `execution_failure.json` and the logs in the task record
   before any re-entry and disclose the re-entry in the task record and the RESULT. A failure after
   `prepare` began, including an index-parity abort, consumes the root; no sibling root may be
   created, and a repeat requires a new task id and review, as it did this time.
4. Production sequence exactly as `production.sequence`: run completed, independent audit
   persisted, report rendered with its viewer receipt, `check-run` and
   `check_results_bundle.py` passing, then and only if `h1_relocation == pass`, one production
   invocation, one `render`, both gates again. Between `init-run` and the post-production rerender
   the bound source, the task JSON (which stays `ready`), the data seal and this review artifact
   must remain byte-identical. `production/*.ply` and `production/receipt.json` may not be cited in
   RESULT, AUDIT, `ara/` or docs; the RESULT files are write-once, so record the post-production
   rerender and its two gate outcomes in the task record's handoff log.
5. Disclosures the RESULT and AUDIT must carry: frame 00008 is outcome-exposed; held-out alpha,
   floater and hull-rejection metrics are in-sample; the first relocation event re-initializes
   most random start points onto the hull; gsplat refinement after each event uses statistics
   partly from pre-relocation positions; the installed gsplat never resets opacity, also for the
   RTGS-025 baseline; `heldout_mask_alpha` is a declared reconstruction input; the first
   pre-review smoke exposed relocation counts, hull size and a meaningless 60-iteration gate value
   to the Driver; the second pre-review smoke (`prepare` plus `initialize` over the frozen split)
   exposed the per-view index parity values, the hull voxel count, the per-view teacher PSNR and
   the initial hull-rejected fractions, and no threshold changed afterwards; the predecessor task
   failed closed at its CUDA parity gate with the receipts at
   `ara/evidence/tables/20260927_silhouette_relocation_failed_run/`; and this task's training
   targets, held-out field diagnostic and production inputs are decoded with the exact CPU index,
   so the `nb_ms` arm is an in-task re-fit whose targets differ from the CUDA-decoded targets of
   the published RTGS-025 run at float32 rounding level and by up to about 1.9e-4 at near-empty
   normalized-blend sites, and the RTGS-025 numbers are not comparators in the gate.
6. Accepted late aborts that consume the root: the index-parity gate in `prepare`, the
   `initialize` hull-face and minimum-voxel checks, a CUDA OOM at the first relocation event, and
   the held-out camera and packed-alpha checks that can only run in `evaluate` after twelve cells.

Optional, not conditions, for a later task: record the CUDA query kernel's near-zero-weight
normalization behaviour as an observation in `ara/staging/observations.yaml` and fix or bound it;
make the field-target cache key include the backend; validate `dilation_px >= 0` in
`SilhouetteHull`; broaden the non-position-row test; the hull-initialized no-relocation arm and the
alpha-and-camera-only `.rtgsv` reader from the V1 record.

## Protected Actions Not Taken

The reviewer did not run `init-run`, did not execute the driver's `run`, `prepare`, `initialize`,
`fit`, `evaluate`, `production` or `selftest` commands directly (the allowed protocol test suite
runs `selftest` as a CPU subprocess with CUDA hidden and `coordinate` on a temporary root whose
guard raises before any write), performed no GPU work, and did not enter `.scratch/`, the consumed
predecessor run root or any smoke directory. No image, mask, `.rtgsv`, `.npz`, `.ply`, run or
benchmark outcome content was opened or printed; `validate-data` hashed sealed bytes opaquely; the
only data-seal content read was its 600-byte header (schema, data slug, input profile, dataset
id/role/path and view ids). The failed-run receipts directory was read in full; it contains no
reconstruction outcome. The RTGS-025 RESULT and AUDIT records were not read; the RTGS-026 task
record was read and its smoke descriptors were treated as claims, not evidence. No run root, RESULT
or AUDIT exists for this task. No file was modified, no run root was created, no checkpoint was
selected, and the owner's task record was not written. The sandbox-denied commands listed above
touched no protected material and were re-done only in the allowed forms. This review has no
outcome access.
