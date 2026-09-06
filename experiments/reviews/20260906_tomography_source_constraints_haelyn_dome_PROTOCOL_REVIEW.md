# Prospective Protocol Review

- Task ID: `20260906_tomography_source_constraints_haelyn_dome`
- Protocol SHA-256: `3b0b25fbb0721b138cfcb42ddfeddbc5dc7f0e883ab53dbea8d2c045931600e7`
- Reviewer: `Claude-Code-Fable-5.1-reviewer`
- Verdict: `approved`
- Outcome Access: `none`

## Scope

This is follow-up review 2 of the revised RTGS-016 protocol, after the first rejection preserved verbatim in `experiments/reviews/20260906_tomography_source_constraints_haelyn_dome_PROTOCOL_REVIEW_V1_REJECTED.md`, whose SHA-256 is `5229175043fd0fce7053c45f64d11950621885cff758d58ee6f0853bec0d8ed2`. Reviewer constraint: model `claude-fable-5-1` at effort max, no nested agents, no fallback model, no model change. The reviewer label differs from the owner label `Codex-tomography-driver`. The Driver response was treated as untrusted orientation, and every digest below was recomputed in this session.

Reviewed state: commit `2ebe52cc28a1b0e812a6c436d49225fac4ee943f` with a dirty worktree of 24 modified tracked files, 969 insertions and 385 deletions, whitespace check clean, plus the untracked RTGS-016 task, seals, scripts, tests and evidence directories.

Independently recomputed values:

- Protocol digest from `review-digest`: `3b0b25fbb0721b138cfcb42ddfeddbc5dc7f0e883ab53dbea8d2c045931600e7`, equal to the header above and to the Driver response.
- Live source binding from `build_source_binding`: 122 files, aggregate `3a3a7f0c1e1dc92c300c43e7a0ca53aa3b1e780fb2dcd8bfb39e93174d0c1430`, unchanged patterns, equal to the frozen `source_binding`.
- Raw task file SHA-256: `2abaa5fb0866f4f8aa0393e13494fcab810402a3c281de364a7b8d4d79ce344e`.
- Compact data seal SHA-256: `9085780d67dca0a8df1a8b7485153fc66fe22a527745935fd0338196b45dbf19`, unchanged from review 1.
- External reference seal SHA-256: `a0b9173030b242badb52d22fa0240ee8e8afafbda2fb33b25eed428a838a573c`, unchanged from review 1 and equal to the frozen `reference_files_seal`.
- The measured effective-configuration table equals the frozen `effective_configuration` for all 36 dataset, arm and seed cells.

Question this protocol may establish: a bounded development comparison at 256 carriers, 100 proxy iterations, 120 native iterations and three paired seeds, of hard versus soft versus free source footprints on held-out native teacher MSE before and after identical compact-only refinement, per capture and per mask condition, with Beam as a descriptive reference only. Limits that remain in force: Haelyn is a downloaded captured representation rendered offline and is not physical density truth; the maskless Haelyn condition has a uniform black rendered background; the second dome capture was previously outcome-exposed; three seeds support sign consistency only; no production default, full-capacity quality, uncontended GPU timing or confirmatory significance claim can follow from this screen.

## Checks

Commands executed, each individually and exactly as permitted:

```
.venv/bin/python -m pytest -q tests/test_tomography_person_protocol.py tests/test_experiment_contract.py
.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260906_tomography_source_constraints_haelyn_dome.json
.venv/bin/python scripts/experiment_contract.py validate
.venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/20260906_tomography_source_constraints_haelyn_dome.json
.venv/bin/python .scratch/rtgs016/claude_review_02/verify_inputs.py
git status --short
git rev-parse HEAD
git diff --stat
git diff --check
git diff scripts/experiment_contract.py
sha256sum experiments/tasks/20260906_tomography_source_constraints_haelyn_dome.json experiments/data/haelyn_dome_source_constraints.json experiments/data/haelyn_dome_source_constraints_external.json experiments/reviews/20260906_tomography_source_constraints_haelyn_dome_PROTOCOL_REVIEW_V1_REJECTED.md
```

Results: 44 tests passed with only dots and no failure output; `experiment_contract: OK`; `experiment_data: OK`; the verify helper printed task validation passed, compact and external seals passed, effective configuration matched, and the digests listed above. I read `verify_inputs.py` before running it. It imports source, validates the task with live source equality, recomputes the source binding, hashes sealed input bytes opaquely, compares effective configurations, verifies the external seal, and prints. It writes nothing and exposes no outcome.

Verified by reading the current code and task:

1. B1, relative argv. `main` resolves `--run` before `validate_binding` at `scripts/experiments/20260906_tomography_source_constraints_haelyn_dome.py:611`. `worker` at line 216, `coordinator` at line 496 and `aggregate` at `scripts/experiments/20260906_tomography_source_constraints_haelyn_dome_report.py:193` resolve again. `aggregate` derives `serve_report` and `viewer` from `run.relative_to(ROOT)` at report lines 456 and 462, which the contract requires to equal `runs/<task_id>` at `scripts/experiment_contract.py:1744` to `1752`. `test_relative_run_aggregation_and_success_receipt_are_complete` calls `aggregate` with the frozen relative argument from a repository-root working directory and asserts the produced sources; `test_warmup_uses_frozen_counts_and_main_resolves_relative_argv` asserts `main` hands the coordinator a resolved path.
2. Receipt-last success. `aggregate` validates `training_history.json` at report line 226 and `metrics.json` at line 491 against the contract before writing them, writes RESULT.json and RESULT.md, and writes `run_receipt.json` with status completed last at line 509. A failure between these points leaves no completed receipt.
3. B2, gate integrity. `blockers` is empty at task line 250; status is `draft`; the review envelope is pending with null reviewer, digest and artifact. A pending review with status ready fails at `scripts/experiment_contract.py:343`, and ready with a non-approved verdict fails at line 405. The tracked contract diff touches only the historical registry opt-out, where `validate_repository` passes `check_live_source=False` at line 866, and the envelope-syntax extraction `_source_binding_errors`. `init_run` at line 1166 and `_locked_task` at line 2137 still call `validate_task` with live source equality. The Driver's in-memory negative control in `gate_separation_check.json` agrees with this reading.
4. R1, placement fallback. `check_placement` at driver line 196 raises on any non-null `placement_fallback_reason`. It runs on the raw `_place` output inside `timed_place` at line 340 before the initial state is saved, and again on `lifted.diagnostics` at line 355. `FieldLifter.fit` calls the module-level `_place` at `src/rtgs/lift/field_lifter.py:1866`, so the interception is effective, and merges `placement.diagnostics` at line 1956 so the key reaches the summary at driver line 439. The warmup cell uses the same path. The fallback itself remains at `field_lifter.py:849` to `857`, so it is fail-closed rather than removed, which is the bounded fix that was requested.
5. R2, failure publication. `coordinator` refuses existing outputs at driver line 498 outside the try block, then wraps `_coordinate`, calls `failure()` with the recorded phase and re-raises at lines 503 to 508. `failure()` at report lines 526 to 642 builds the six sources in `REQUIRED_V2_FAILURE_ARTIFACTS`, validates `metrics.json` with `_metric_errors_v2(completed=False)` and the receipt with `_run_receipt_errors`, moves any pre-existing partial source into `failure_preserved/` with exact bytes, and never overwrites. `test_coordinator_failure_is_renderable_and_retry_preserves_bytes` verifies that `_v2_source_errors` returns not completed with no errors, that the partial bytes are preserved, and that a refused retry leaves every byte identical. The frozen `failure_policy` now states abort on first failure, no later cells, no retry and no replacement cell, matching the code.
6. R3, non-finite proxy state. `check_proxy` at driver line 202 rejects any non-finite entry of `objective_history` or `elapsed_seconds`. In `fit_field_fibers`, a rejected step appends the finite `before` value and halves the rate at `src/rtgs/lift/field_refit.py:656` to `665`, so a non-finite value can enter the history only when the current objective itself is non-finite, and the cell then fails. `accepted_steps` from line 708 is recorded as `proxy_accepted_steps` at driver line 442. `_float32` at driver line 178 rejects empty or non-finite states for every saved model, including the initial state and each checkpoint.
7. R4, metric semantics and full-canvas windows. `_evaluate_measure` averages squared error over the teacher fit window at the half-integer offset and equal-weights views at `src/rtgs/optim/compact_trainer.py:2091` to `2152`, and the task metric text now states exactly that. My review-1 inference that masked windows are mask crops was wrong. `fit_image` crops internally and restores full-image coordinates at `src/rtgs/image2gs/fit.py:553`; the preparation driver exports without a `fit_window` argument at `scripts/experiments/20260906_tomography_source_constraints_haelyn_dome_prepare.py:229`; `native_gaussians_to_observation` therefore defaults to the full canvas at `src/rtgs/image2gs/native_observation.py:43`. The masked archives are consistent with that because `save_compact_view` requires the alpha crop to match the fit window at `src/rtgs/data/compact_views.py:603`. The Driver's metadata receipt `fit_window_input_check.json` lists 47 full-canvas windows: 16 masked and 16 maskless Haelyn views at 192 by 256, and 15 dome views at 253 by 219, so the masked and maskless Haelyn denominators are identical. I opened no archive. `check_windows` at driver line 208 enforces the invariant at run time on train, validation and heldout views before any optimizer or evaluator use, so a wrong receipt would abort the run rather than change the metric. This receipt is input metadata, not a quality outcome.
8. R5, alpha declaration. `alpha_policy` in the task states that masked field arms consume compact-embedded alpha for source-support gating, that the unmasked condition and Beam consume none, and that raw masks stay report-only. The worker sets `load_alpha` only for masked field arms at line 262, sets `mask_mode` none only for `haelyn_unmasked` at line 88, and records `load_alpha`, `mask_mode` and `embedded_alpha_use` in the input-boundary record at lines 414 to 416.
9. R6, background. `claim_boundary` now names the uniform black rendered background of maskless Haelyn and the fixed initial render opacity of proxy-stage models.
10. O1 to O5. `configs` reads the frozen warmup counts and derives the warmup checkpoints and repair steps from them at lines 79 to 83 and 92 to 93; the test covers refit, repair and native counts. Observer-excluded endpoint time is reported as a secondary group value and paired ratio at report lines 315 and 341, while the primary clock stays endpoint-inclusive as `resource_clock` states. The soft and free comparator purposes declare the five extra coordinates and the shared global gradient clip; the Beam purpose declares alpha-free initialization; `failure_policy` declares no retry. The stage-runtime chart reports each frozen stage from the recorded markers at report lines 394 to 418.
11. Original requirements retained. Causal arms share a byte-identical initial PLY per seed, enforced in the worker at line 296 and cross-checked by the coordinator at lines 566 to 573; association and topology settings are identical across the causal arms by the configuration test, so the state entering refit differs only by the treatment. Heldout archives become readable only after `gaussians.ply` exists at lines 398 to 404 and never enter history records. The compact-only boundary reuses the tested import and open guard with the reference directory and unselected archives denied. Data seal and external seal are verified at coordinator entry and exit, selected inputs are hashed at worker entry and exit, and the source binding is verified in every process at entry and again at worker and coordinator exit.
12. Three-dataset acceptance. The tests exercise one-dataset variants, so I read the validators against the real 36-cell payload. `dataset_summaries` is optional at `scripts/experiment_contract.py:1916`. `_metric_errors_v1` requires the three charts in frozen order, all four primary metrics with frozen directions, the canonical evidence paths and a viewer argv, all of which `aggregate` produces. `_history_errors` requires six ordered markers per series and records inside their stage, which the worker's marker and record steps satisfy for the frozen checkpoints at 0, 30, 60, 90 and 120. `_render_run_v2` adds no structural requirement beyond `validate_run`.

Not performed, and why:

- I could not confirm the final state of `/tmp/rtgs016_revision_verify_final.log`. The Read tool refused the path as outside the restricted working directory, and `ls`, `tail` and `cat` on that path were each denied by the permission mode. The revision preflight `receipt.json` records the 44 tests, the seals and 1,703 unchanged historical files, but no `verify.sh` exit status. The full CPU suite and `./scripts/verify.sh` on the current source are therefore unverified by me.
- `git -C <path> status --short` and `git -C <path> rev-parse HEAD` were denied; the plain forms succeeded and are what is reported here.
- I did not run `init-run`, any worker, the coordinator, warmup, render-reference, the converter, the import replay, the phase evaluator, `aggregate`, `render`, `check-run`, or the results-bundle gate. I opened no file under `runs/` or `benchmarks/results/`, and no PLY, splat, rtgsv, image, mask or reference model.
- I did not recompute the 1,703-file historical preservation manifest; `git status` shows no tracked historical task, review, benchmark result or run file modified.
- `validate_binding` with a real lock, and `render` plus `check-run` on a real 36-cell bundle, are exercised only at run time. The Driver's fixture-only producer receipt claims exit 0 but does not state which gates ran.
- I could not diff the worktree against the review-1 snapshot. Every line anchor cited in review 1 for `field_lifter.py`, `field_refit.py`, `compact_trainer.py` and `experiment_contract.py` still resolves to the same statement, which supports the Driver's statement that no algorithm changed, but this is anchor evidence rather than a byte-level proof.

## Findings

### Blocking defects

None remaining. B1 and B2 are fixed as verified above. R1 to R6 are implemented as bounded fail-closed checks and protocol wording, with source-bound tests for the relative aggregation, the failure publication and retry refusal, the fatal guards, and the frozen warmup counts. The distinct-review gate is unchanged from HEAD. Sealed inputs, effective configurations and comparative budgets are unchanged, so nothing in this revision was tuned toward an outcome.

### Conditions before init-run that do not change the protocol digest

1. Preserve this review verbatim as `experiments/reviews/20260906_tomography_source_constraints_haelyn_dome_PROTOCOL_REVIEW.md`, copy the reviewer label, the verdict `approved`, the digest above and the artifact path into `protocol_review`, set status `ready`, and rerun `validate`. Editing any other field changes the digest and voids this approval by construction.
2. Record the final `./scripts/verify.sh` exit status and log digest in the revision-1 preflight directory before `init-run`. If that gate fails for a reason that changes any file under the source binding, the aggregate changes and a new digest and review are required. A task-record syntax fix alone does not.
3. Use `init-run --development` because the tree is dirty, then execute the frozen `run_command` unchanged from the repository root. Nothing else may run under this task id.

### Non-blocking observations for the results stage

- The bundle gate's default preview set is the `rtgs run --preview` set at `scripts/check_results_bundle.py:44` to `49`, which this compact-only producer never writes. The run instead emits 36 per-cell comparison previews that are inventoried and link-checked. The final gate will therefore need `--no-previews`, and the RESULT note and audit should say so explicitly so the flag is not misread as a missing artifact.
- The placement stage interval starts at driver entry through the marker at driver line 316, so its reported duration includes worker imports, protocol and source validation and input loading. `resource_clock` documents the clock start, but the stage chart title mentions only validation observers. The raw markers are retained; the audit should read the stage chart with this in mind.
- If `failure()` itself raises while publishing, the run root keeps raw cell artifacts but no renderable v2 failure sources. The original error is chained, so nothing is hidden.
- `configs` substitutes the hard treatment for `beam_reference` in the effective-configuration hash even though Beam never runs the field refit. This is descriptive only and alters no Beam computation.

### Assessment of the design

Unchanged from review 1. The paired soft-minus-hard and free-minus-hard differences on held-out teacher MSE, before and after identical native refinement, per capture and per mask condition over three seeds, can be answered descriptively. Fixed endpoints, prospectively fixed tether weights, deterministic placement with byte-level initial equality, immutable full-canvas teachers, frozen effective-configuration hashes and a separate report-only RGB evaluator are sound. Beam remains descriptive and is excluded from causal claims. Nothing in this protocol may be read as evidence of physical density, a default improvement, full-capacity quality, uncontended GPU timing or confirmatory significance.

## Protected Actions Not Taken

I did not run `init-run`, the coordinator, any worker cell, the warmup, render-reference, the converter, the import replay, the phase evaluator, `aggregate`, `render`, `check-run`, or the results-bundle gate. I did not open any file under `runs/` or `benchmarks/results/`, and no `.ply`, `.splat`, `.rtgsv`, image, mask or reference model. I ran only the five permitted verification commands, the permitted read-only git and hashing commands, and file reads of source, tests, task, seals, receipts and logs inside the repository. Denied commands were reported, not bypassed. I modified no source, task state, review file, historical evidence, git state or settings. Outcome Access remained none throughout.
