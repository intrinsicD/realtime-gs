# Prospective Protocol Review

- Task ID: `20260926_field_only_distillation_stage_frame00008`
- Protocol SHA-256: `05048698cc34656fd562c8f60d534037745d99acf91b9cac6712a5f314b23f61`
- Reviewer: `Claude-Code-Fable-5.1-reviewer`
- Verdict: `approved`
- Outcome Access: `none`

## Scope

Round 2 of the prospective review. Round 1 rejected digest
`ef904cd749a80de5f25b9c6706cc705617aebfe56bd0b16912695626a99e44af` for three bounded start-up
and wording defects (B1-B3) and listed eight optional improvements (R1-R8); that record is
preserved verbatim as `..._PROTOCOL_REVIEW_V1_REJECTED.md`. This round reviews the Driver's
revision at commit `f5ddaa2` against the round-1 reviewed state `298a9b5`, with the Driver's
response (`..._DRIVER_RESPONSE_R1.md`) treated as a claim to be checked, not as evidence.

The question is unchanged: whether a random-initialized 3DGS trained only from decoded compact 2D
Gaussian fields of 22 training views, their calibrated cameras and the packed training alpha,
with colour supervised inside the mask and silhouette/outside-alpha terms against floaters,
matches photograph supervision inside four held-out masks without more floaters (H1), and whether
the teacher fitted without mask containment beats the mask-contained teacher (H2). H3 (random
versus packed-alpha visual-hull initializer) and H4 (silhouette objective versus premultiplied
black) are descriptive with paired deltas and no pass/fail.

Evidence maturity this protocol may establish is unchanged: a development screen on one
previously outcome-exposed frame (00008), one frozen train/held-out split, three paired seeds and
one fixed 8000-step gsplat budget at downscale 8. A pass may keep the RTGS-025 main path and
select the uncontained teacher family for the next registered task. It cannot establish a
default, SOTA, generalization, physical-geometry, speed or VRAM claim. The packed alpha is
mask-derived and drives supervision and the hull, so the field arms are field-plus-silhouette,
not strictly Gaussian-only. H2 compares teacher families (count, topology schedule, containment
and, per B2, boundary-band colour differ together), not containment alone. Approval says only
that the frozen design is fit to execute; it is not evidence that the method works and it does
not verify the Driver's smoke or `verify.sh` claims.

## Checks

Read: the round-1 review and the Driver response; `git diff 298a9b5 HEAD` in full for the task
JSON, the driver, the report module, `tests/test_field_only_distillation.py` and
`.agents/state/current-task.md`; the complete revised driver, report module and test file;
`rtgs.data.field_targets` (unchanged; `coverage` semantics), `rtgs.data.compact_views` (loader
`source` record, writer signature), `scripts/convert_datasets_to_gaussians2d.py` (what the
recorded source digests hash), `rtgs.optim.trainer` (`reset_cuda_peak_stats` uses),
`scripts/experiment_contract.py` (review-header parser, `init_run`, `validate_task` live-source
default, `build_source_binding`), `scripts/check_results_bundle.py` (receipt handling), the
reviews README and template, the data-seal path format, and the `has_alpha` flags of the three
family manifests.

Commands executed (all read-only or CPU unit tests; none touched images, masks, `.rtgsv`,
`.npz`, `.ply`, `.scratch/` or any run outcome):

```text
git status --short; git rev-parse HEAD; git log --oneline 298a9b5..HEAD
git diff --stat 298a9b5 HEAD
git diff 298a9b5 HEAD -- experiments/tasks/20260926_field_only_distillation_stage_frame00008.json
git diff 298a9b5 HEAD -- scripts/experiments/20260926_field_only_distillation_stage_frame00008.py
git diff 298a9b5 HEAD -- scripts/experiments/..._report.py tests/test_field_only_distillation.py
git diff 298a9b5 HEAD -- .agents/state/current-task.md
git status --ignored --short -- src/rtgs scripts/experiment_contract.py scripts/check_results_bundle.py scripts/experiments tests/test_field_only_distillation.py tests/test_field_targets.py pyproject.toml
git rev-parse HEAD:src/rtgs HEAD:<each named bound file>
.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260926_field_only_distillation_stage_frame00008.json
.venv/bin/python scripts/experiment_contract.py validate
.venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/20260926_field_only_distillation_stage_frame00008.json
.venv/bin/python -m pytest -q tests/test_field_only_distillation.py tests/test_field_targets.py
sha256sum <task-owned sources, data seal, task JSON>   (table below)
```

Denied by the review sandbox, none touching protected material: the pytest command with a
`CUDA_VISIBLE_DEVICES=""` prefix (re-run in the listed form), a read-only Python call to
`build_source_binding` to recompute the aggregate, and a shell loop around `git rev-parse`
(re-run as one plain `git rev-parse`).

Results: `review-digest` reproduces `05048698cc34656fd562c8f60d534037745d99acf91b9cac6712a5f314b23f61`
exactly, the digest the Driver reported; `validate` and `validate-data` print OK; pytest reports
16 passed and 1 skipped (the CUDA decode-parity test self-skips on CPU). The worktree is clean at
`f5ddaa2`. No `runs/20260926_field_only_distillation_stage_frame00008/` root and no
`benchmarks/results/20260926_*` evidence exist. No untracked or ignored files other than
`__pycache__` exist under the bound source patterns.

- Change inventory and drift. The diff since `298a9b5` touches seven files: the task JSON, the
  driver, the report module, `tests/test_field_only_distillation.py`, the task record and the two
  round-1 review records. `src/rtgs`, `scripts/experiment_contract.py`,
  `scripts/check_results_bundle.py`, `pyproject.toml`, `tests/test_field_targets.py` and
  `src/rtgs/data/field_targets.py` are unchanged; the last two and the data seal hash to their
  round-1 values. In the task JSON exactly these fields changed: `resource_protocol.scope` (R4
  sentence), `preprocessing.target_operator` (B2), `preprocessing.training_masks` (R2/R3/R6),
  `initialization.sharing` (the scale-difference disclosure round 1 called undisclosed),
  `reset_cuda_peak_stats` true to false in all six resolved configs (R4),
  `decision_policy.h1_main_path` and `h2_teacher_containment` (B3/R5 wording),
  `decision_policy.stopping` (preflight and timed-out receipt sentence) and
  `frozen_configuration.source_binding.aggregate_sha256`. `hypothesis`, `claim_boundary`,
  `seeds`, `splits`, `comparators`, `primary_metrics`, `training`, every other resolved-config
  field, `scoring`, `execution_order`, `execution_controls`, `preview_policy`, `run_command`,
  `blockers`, `owner`, `field_inputs` and the data seal are byte-identical. No hypothesis,
  threshold, seed, split, comparator, metric, budget or execution order moved.
- Digest scope. `protocol_sha256` still drops only `status` and `protocol_review` (contract
  script unchanged), so recording this verdict does not change the digest.
- B1 resolved. (a) `write_environment` (driver 880-905) queries `rtgs` and wraps every lookup in
  `except PackageNotFoundError`; `test_environment_record_uses_installed_distributions` asserts
  the `rtgs` version resolves and that a second call is a byte-identical no-op. (b) It is
  independent of `snapshot_source` and runs on every start that lacks `environment.json`.
  (c) `snapshot_source`, `write_environment`, `source_guard`, the entry `data_guard` and
  `preflight` run inside the coordinator `try` (953-958); any start-up exception writes
  `execution_failure.json` and the failure report via `publish_failure`, which needs only
  `task.lock.json`. (d) `preflight` (908-922) requires CUDA, imports gsplat and runs one
  LPIPS(alex, pretrained) forward on `cuda:0` before the first protected worker, recording
  `preflight.json`. Note, descriptive only: the coordinator now also holds cuDNN/cuBLAS state
  for the whole run in the CUDA context it already opened in round 1 through
  `get_device_name`; per-cell peaks are measured inside the worker processes.
- B2 resolved. `target_operator` states the composite `target*mask + random_background*(1-mask)`
  with L1 weight `0.1 + 0.9*mask`, the zero-fraction regime (target colour ignored, alpha
  penalized) and the 0.25/0.5/0.75 regime (outside-mask sites enter in proportion to the
  fraction), names the family difference in that band (contained teacher black, uncontained
  teachers extrapolated foreground, photographs room background) and marks it in scope for H2
  and required in the audit. This matches the Trainer objective as read in round 1; the Trainer
  source is unchanged.
- B3 resolved. `h2_teacher_containment` spells out the pass pair and the mirrored reject pair.
  `gates()` (report 91-110) evaluates per seed `nb >= mc + 0.2 and nb_out <= mc_out + 0.005`
  for pass and `nb <= mc - 0.2 and nb_out >= mc_out - 0.005` for reverse; all pass gives pass,
  else all reverse gives reject, else inconclusive; G0 failure overrides both to inconclusive.
  `h1_main_path` gained "otherwise fail", which the code already implemented.
- R1 adopted. `access_guard` resolves the frame directory, the target-cache root and every
  opened path (159-160, 172) before the prefix tests; `publish` refuses any run root other than
  `runs/<task_id>` (report 255-257), with `ROOT` resolved identically in both modules;
  `test_publish_refuses_non_canonical_run_root` covers it and the selftest still reports five
  denied probes.
- R2 adopted. `coverage` is the float fraction of the four sites inside the fit window
  (field_targets 44, 160), returned on the CPU alongside `color`; `(soft > 0) & (coverage < 1)`
  counts exactly the training pixels with any mask support and any site outside that family's
  window; a nonzero count aborts before the target is cached (434-437) and the count is
  recorded per family and view.
- R3 adopted. `CompactView.load` parses the `source` record for every view regardless of
  `load_alpha` (compact_views 422-425, 466-475). The conversion pipeline records `file_sha256`
  of the JPEG and PNG (convert_datasets_to_gaussians2d 162-164, 1029-1034), the same operator
  as the seal, and the seal keys used by `prepare` (396-399) match the seal's path format. All
  26 views of all three families declare `has_alpha: true` in their manifests, and the schema
  ties a declared alpha to a declared source mask, so the `source["mask"]` record cannot be null
  on this data. The check precedes parity and decoding and opens no new dataset file.
- R4 adopted. `reset_cuda_peak_stats` affects only the Trainer's own peak/accumulated-stat reset
  (trainer 641-643); flipping it changes no optimisation behaviour. The driver resets peaks after
  warmup and seeding (617-618), so each receipt peak includes cache loading and the initial
  upload, as the new scope sentence states. The six configs still round-trip and differ between
  objectives only in `use_masks` and `random_background` (unit-tested).
- R5 and R6 adopted. H1/H2 inequalities are evaluated in written form and the policy text says
  so; the frozen-gate unit test still passes at its boundary values. `training_masks` describes
  the per-view prepare checks performed once on behalf of all arms.
- R8 adopted. `worker` runs fitting cells with `timeout=3600` (949); `subprocess.run` kills the
  child and raises `TimeoutExpired`, which the coordinator catches to write a `timed_out` receipt
  before re-raising into the failure path (963-977). `evaluate` still refuses unless every cell
  receipt reads `completed`, so a timed-out cell cannot be evaluated.
- R7 not adopted. `gi_rand_ms` and `ph_hull_ms` remain non-gating; acceptable.
- Execution order. Coordinator: snapshot, environment, source guard, entry data guard,
  preflight, prepare, initialize, the 21 frozen cells in the frozen order, evaluate, exit source
  guard, exit data guard with split hash, publish; all inside one `try`. Unchanged apart from
  the start-up additions.
- Source binding. The 118-file count equals 107 `.py` plus 4 `.cu/.cpp/.h` under `src/rtgs`
  plus the seven named files (Glob), with no untracked or ignored files under the patterns. The
  aggregate `587419c468566af2a1a3181f992012a00eff253a5942950e3bfd90be74dab206` was not
  independently reproduced: the sandbox denied the read-only call, and the allowed command set
  has no byte-size or join tooling. The three changed bound files (driver, report module, test)
  fully account for the change from the round-1 aggregate. Enforcement is fail-closed:
  `init_run` calls `validate_task` with its default `check_live_source=True` (contract 411,
  786-787, 1166), refuses an existing root (1188-1189) and locks the review-artifact hash; the
  registry `validate` still does not check live source. The reviewed identities are pinned
  below by SHA-256 and git object id so the audit can compare them with the snapshot's
  `source_commit`.
- Re-entry semantics (pre-existing, not a new defect). Re-invoking `run` on a root where
  `prepare` already began fails closed at `targets/` creation, but the failure path overwrites
  `execution_failure.json`, `run_receipt.json`, `metrics.json` and `logs/prepare.log`, also for
  a completed bundle; cell receipts and cell logs survive. Neither `check-run` nor
  `check_results_bundle.py` inspects `execution_failure.json`. A re-entry cannot produce a
  results-bearing bundle, so this is an operating condition below and an optional hardening.
- Task record. Driver and Reviewer labels are distinct; the round-1 verdict is recorded as
  revision required with the verbatim artifact; the handoff's `./scripts/verify.sh` exit 0 and
  the repeated GPU smoke claims were not verified here (outside the allowed command set). The
  smoke description ("two training views standing in for held-out") leaves open whether all 22
  frozen training views passed the new prepare checks; see condition 4.

Reviewed task-owned identities at `f5ddaa2`:

| File | SHA-256 |
|---|---|
| `scripts/experiments/20260926_field_only_distillation_stage_frame00008.py` | `31c6fd870b4fd963afc577e99972768686644f4854f8c094004b73ea6ea8e360` |
| `scripts/experiments/20260926_field_only_distillation_stage_frame00008_report.py` | `bba7e8f7f1c4293103be97edacac8234ae8759a61deb66e7f9752b067e1acc81` |
| `tests/test_field_only_distillation.py` | `06127a75aac78ec18846ad7c08118d69a3b7496e44a20a61822f85cbf954bd5f` |
| `tests/test_field_targets.py` | `0a817166c45df13d64cd8e8f83aafbf7656e1bc37ef8bf26176bf6def75a72cc` |
| `src/rtgs/data/field_targets.py` | `59901d820ef291381c27cb45ba68161cdad1195eff7ea891ae281b60bca9829f` |
| `experiments/data/20260926_field_only_distillation_stage_frame00008.json` | `d01254e739961dcb1acc4a9abca6694bb63e319ff179581ebe2ec2ae08706087` |
| `experiments/tasks/20260926_field_only_distillation_stage_frame00008.json` | `9fd9252467d8dd3d337885427b3bcff55add86b57c35f6675f70a5fd88289b94` |

Git object ids of the bound source at `f5ddaa2` (`git rev-parse HEAD:<path>`):

| Path | Object id |
|---|---|
| `src/rtgs` (tree) | `641dbc4ee15b2c9ff697c2b3e2f83fb25135c195` |
| `scripts/experiment_contract.py` | `3d4f251972226e5372dfb5346283c65f2e73e62e` |
| `scripts/check_results_bundle.py` | `4be3439031998a35caf1b6d023085dccee659ceb` |
| `pyproject.toml` | `f3911412950125f290bc65a49818ed6949c820a7` |
| `scripts/experiments/20260926_field_only_distillation_stage_frame00008.py` | `4d680c7f199541f7306a37450ae3b365e74f945a` |
| `scripts/experiments/20260926_field_only_distillation_stage_frame00008_report.py` | `2d6415f9e1334a63e06d10b3d956823eb9ac18dd` |
| `tests/test_field_only_distillation.py` | `2684ddfe0d2556b5d57298417b667501b0a131ff` |
| `tests/test_field_targets.py` | `1f522a73de24a3ce91c293ee52459f9592f439cd` |

## Findings

Approved. B1-B3 are resolved exactly as required, each adopted optional change is correct as
implemented and unit-covered where it can be, and the revision introduces no new defect in the
fitting, evaluation, gating or leakage paths. The scientific design is unchanged from the round-1
assessment: comparison design, leakage boundaries, mask-restricted scoring, initializer sharing,
premultiplied control, per-seed inclusive gates and fixed-budget stopping are correct as
implemented, and no bound quantity drifted between the two digests except the wording, the
measurement-only peak-stats flag and the source-binding aggregate.

Residual execution conditions:

1. Record the approval by changing only `protocol_review` (this reviewer label, verdict
   `approved`, this digest, this artifact path) and `status: ready`; `review-digest` must still
   print `05048698cc34656fd562c8f60d534037745d99acf91b9cac6712a5f314b23f61` and `validate` must
   pass. Any other edit to the task, driver, report module, tests or bound source invalidates
   this approval.
2. Run `init-run` from a tree whose bound paths equal the object ids above (in particular
   `git rev-parse <commit>:src/rtgs` equal to `641dbc4ee15b2c9ff697c2b3e2f83fb25135c195`);
   later commits that touch only `experiments/`, `docs/` or `.agents/` do not alter the binding.
   If `init-run` reports a source-binding mismatch, stop and request a new review; do not edit
   `aggregate_sha256` to match.
3. Invoke `run` once. If it fails before `targets/` exists (start-up or preflight), one
   re-invocation is acceptable but must be disclosed in the task record, and the retained
   `execution_failure.json` must be named for the audit. If it fails after `prepare` began,
   treat the root as consumed and do not re-invoke: the coordinator would replace the top-level
   failure record and, for a completed bundle, its `run_receipt.json` and `metrics.json`.
4. If the pre-review smokes did not run `prepare` over all 22 frozen training views, a
   non-protocol smoke of the prepare stage alone over the frozen training split, on training
   views only and under the same disclosure class as the earlier smokes, is acceptable before
   `init-run`, so that the new fail-closed source-digest and fit-window checks do not consume the
   canonical root. This is optional; the checks are correct as written either way.
5. The independent audit must name the boundary-band colour difference (B2) and the H2 family
   confound when reading H2, must judge visual adequacy for G0, and must be told of any
   re-invocation under condition 3.
6. Optional hardening, deferred to the next task because adopting it would change the driver
   and require a new digest and review: refuse to start the coordinator when `targets/` or
   `preparation.json` already exists, and open worker logs in append mode.

## Protected Actions Not Taken

The reviewer did not run `init-run`, did not execute the driver's `run`, `prepare`,
`initialize`, `fit`, `evaluate` or `selftest` commands, performed no GPU work, and did not enter
`.scratch/` or any smoke directory. No image, mask, `.rtgsv`, `.npz`, `.ply`, run or benchmark
outcome content was opened or printed; `validate-data` hashed sealed bytes opaquely, and the only
dataset metadata touched was a text search for `has_alpha` flags in the three family manifests.
The three denied commands (a prefixed pytest invocation, a read-only aggregate recompute and a
shell loop around `git rev-parse`) touched no protected material and were not retried in any
other form beyond the listed commands. The prior task's RESULT and AUDIT records were not read.
No file was modified, no run root was created, no checkpoint was selected, and the owner's task
record was not written. This review has no outcome access.
