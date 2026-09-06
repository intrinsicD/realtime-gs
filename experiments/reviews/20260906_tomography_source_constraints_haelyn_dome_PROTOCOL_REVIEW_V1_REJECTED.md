# Prospective Protocol Review

- Task ID: `20260906_tomography_source_constraints_haelyn_dome`
- Protocol SHA-256: `e8c2e3d7a45f2710675521d470f4aed8b6d43bb98f245f8eb34f6eea7dc86ed3`
- Reviewer: `Claude-Code-Fable-5.1-reviewer`
- Verdict: `rejected`
- Outcome Access: `none`

## Scope

Reviewed state: the uncommitted worktree on commit `2ebe52cc28a1b0e812a6c436d49225fac4ee943f`, with 24 modified tracked files and the untracked RTGS-016 files named in the dispatch. Reviewer constraint: model `claude-fable-5-1` at effort max, no nested agents, no model change. The reviewer label differs from the owner label `Codex-tomography-driver`.

Digests under review. The protocol digest above and the source binding aggregate `95884b5650ee5e8fb5eae524a33321c065e7bf823085f5e907c9760d3497b308` across 122 files are quoted from the task and packet; the Checks section states what was and was not recomputed. Independently recomputed values:

- Raw task file SHA-256: `f12adeedaac4e18b1617dda729092a55b3804ccb3bb1a957916e817acafbb1c6`
- Compact data seal SHA-256: `9085780d67dca0a8df1a8b7485153fc66fe22a527745935fd0338196b45dbf19`
- External seal SHA-256: `a0b9173030b242badb52d22fa0240ee8e8afafbda2fb33b25eed428a838a573c`, equal to the value frozen in the task's `reference_files_seal`

Question this protocol may establish: a bounded development comparison, at 256 carriers, 100 proxy iterations, 120 native iterations and three paired seeds, of hard versus soft versus free source footprints on held-out native teacher MSE before and after identical compact-only refinement, per capture and per mask condition, with Beam as a descriptive reference. It cannot establish physical density truth, general improvement, a default change, full-capacity quality, GPU timing, or confirmatory significance.

## Checks

Performed:

1. Read CLAUDE.md, the active task record, AGENT_WORKFLOW, experiments/README, the review, task-workflow, experiment and results-audit skills, the packet, the task JSON, both seals, all four RTGS-016 scripts, the new test, the guard base script, the contract validators, the changed lift, refit, fiber and CLI source with their diffs, the compact trainer, compact views, field inputs, calibrated bounds, the native fitter crop, the three compact manifests, the render, acquisition and alignment receipts, the preflight receipt and logs, and the RTGS-015 handoff.
2. Ran `.venv/bin/python -m pytest -q tests/test_tomography_person_protocol.py tests/test_experiment_contract.py`. Every test passed; the runner printed only dots and no failure.
3. Ran `git diff --stat` and `git diff --check`. The diff is 24 files with 917 insertions and 422 deletions, and the whitespace check is clean.
4. Ran `sha256sum` on the task and both seals, giving the values above.
5. Counted the source-binding pattern matches with the file search tool: 103 Python and 4 CUDA/C++ files under `src/rtgs`, 7 templates, 4 RTGS-016 scripts, the 20260729 driver, `scripts/experiment_contract.py`, `scripts/check_results_bundle.py` and `pyproject.toml`. That is 122, matching `file_count`.
6. Confirmed from `git status` that no tracked file under `experiments/reviews`, `benchmarks/results`, `runs`, or any historical task is modified.
7. Verified by code reading: hard, soft and free configs differ only in the two treatment fields, as `tests/test_tomography_person_protocol.py:36` asserts. Identical initial state is enforced by PLY hash in the worker at `scripts/experiments/20260906_tomography_source_constraints_haelyn_dome.py:253` and `:317`, and cross-checked by the coordinator at `:481`. Soft mode starts from zero offsets on the hard geometry at `src/rtgs/lift/inverse_projection_fiber.py:266`, differing only by Cholesky round-off. The five relaxed coordinates have finite-difference-checked gradients. The projection check compares against the current soft targets plus the fixed dilation at `src/rtgs/lift/field_refit.py:669`. SH stays anchored at the initial source direction and the actual final-ray colour error is reported at `:712`. Visibility is frozen centre transmittance with forced source visibility, and gains are a density-only ridge solve. RGB normalisation is cached target field energy with a 1e-12 floor.
8. Verified the maskless boundary. Unmasked views were fitted with no mask and no crop, at `scripts/experiments/20260906_tomography_source_constraints_haelyn_dome_prepare.py:221` and `src/rtgs/image2gs/fit.py:263`. Bounds are camera-only and byte-identical between the masked and unmasked manifests, and `src/rtgs/data/calibrated.py:150` gives the frozen 0.72 factor. The unmasked worker loads no alpha and uses `mask_mode` none. Heldout masks are opened only by the separate report process.
9. Verified partition access. Validation views are removed from the fitting set at driver line 196, loaded without alpha, and used only by a deterministic observer that cannot touch the per-step trainer generators. Heldout archives become readable and are opened only after `gaussians.ply` exists at driver line 352. Heldout values never enter training-history records.
10. Verified binding. The frozen command path validates the lock, the live source binding, comparator order and the effective-configuration hashes before any work at driver line 94. Workers rehash selected inputs at entry and exit and re-verify the source binding at exit. The coordinator verifies the data seal at entry and exit. The guard denies the reference directory and unselected archives.
11. Verified the historical lifecycle change. Only `validate_repository` opts out of live source equality at `scripts/experiment_contract.py:862`. `init_run` at `:1166`, `_locked_task` at `:2137` and therefore `validate_run` keep live checks. Malformed envelopes still fail at `:1055`. The janelle tests now assert the old driver rejects current semantics. The new fixture is configuration only and self-verifies against the frozen task digest.
12. Verified finite-value behaviour in the trainer and evaluator, degree handling of degree-0 initial models at `src/rtgs/render/torch_points.py:137`, and that `--no-open` is a valid generated flag at `src/rtgs/cli.py:956`.

Not performed, and why:

- I could not run `review-digest`, `validate`, `validate-data`, or a source-binding recomputation. The harness denied direct Python script invocation in this session and permitted only pytest and read-only shell commands. The protocol digest and the 122-file aggregate are therefore not independently recomputed. The raw file hashes and the file count are independent.
- I did not run the full CPU suite or `./scripts/verify.sh`; I only read the Driver's logs, which show all tests passing and every gate OK.
- I executed no worker, coordinator, `init-run`, render-reference, converter, import, phase evaluator, or aggregation, and opened no run directory, PLY, splat, rtgsv, image or mask.
- I did not verify the 1,703-file preservation manifest beyond the tracked-file evidence from `git status`.
- No test exercises `validate_binding` with a real lock, the `aggregate` function, or the coordinator failure path. The test at `tests/test_tomography_person_protocol.py:134` calls `worker` directly. My findings on those paths come from reading only.

## Findings

### Blocking defects

B1. The final aggregation crashes under the frozen command. `main` passes the unresolved relative `--run` value to `coordinator` at `scripts/experiments/20260906_tomography_source_constraints_haelyn_dome.py:530`. `aggregate` then calls `run.relative_to(ROOT)` at `scripts/experiments/20260906_tomography_source_constraints_haelyn_dome_report.py:438`, `:445` and `:447`. A relative path is not inside an absolute root, so `Path.relative_to` raises `ValueError`. This fires after every cell and phase report has completed, after `run_receipt.json` and `cell_results.json` are written, and before `metrics.json` and the RESULT records. A rerun is refused by the existing-output check at driver line 432. The phase script resolves its own run path at report line 501; the coordinator path does not. Bounded fix: resolve `args.run` in `main` before calling `worker` or `coordinator`, and add a test that calls `aggregate` from the repository root with the frozen relative argument.

B2. The current digest cannot produce a valid ready transition. The `blockers` list is inside the protocol digest per `scripts/experiment_contract.py:246`. A ready task may not retain blockers at `:782`, and approval requires status ready with a matching digest at `:364` and `:401`. The retained blocker at task lines 250 to 252 is self-referential, so no approval of the digest above can be copied into the task without invalidating itself. The RTGS-013 V5 review resolved the same circularity by an owner-authorized removal of the stale blocker before the review. Bounded fix: the Driver sets `blockers` to `[]` in the same protocol edit as the fixes below, keeps status `draft` with the review envelope pending, recomputes the digest, and requests a follow-up exact-digest review. Do not weaken the validator and do not edit any historical protocol or review.

### Required bounded fixes before the follow-up review

R1. Silent placement fallback contradicts the frozen protocol. `_place` catches carve failure and substitutes a deterministic random-in-bounds initialization at `src/rtgs/lift/field_lifter.py:849`, recording only `placement_fallback_reason` in diagnostics at `:998`. The worker never inspects that key, so a cell whose placement was not `compact_carve` would be marked completed. This violates the frozen failure policy's ban on silent fallback and the placement stage definition. Fix: raise in the worker when the fallback reason is not None, for every field arm including warmup, and copy the key into `summary.json`.

R2. Failure semantics differ between protocol text and code. The coordinator raises at the first failed cell at driver line 471 and writes no failure `run_receipt.json`, `metrics.json`, or the other v2 failure sources, so `render` cannot produce the explicit failure report that experiments/README lifecycle step 9 requires. The frozen `failure_policy` at task line 376 says the affected cell stops and retains failed status, which reads as continuing other cells. Fix in either direction: wrap the coordinator so a failed run writes the v2 failure sources it reached with status failed, a failure phase and a non-zero exit code, and state in `failure_policy` that the run aborts at the first failed cell and no later cell executes. The no-overwrite refusal may remain.

R3. Non-finite proxy objectives are absorbed instead of failing. In `fit_field_fibers`, a non-finite objective value only reverts the step and halves the rate at `src/rtgs/lift/field_refit.py:656`. The history then carries NaN, which `safe` converts to null before `summary.json` is written with status completed. Fix: after the lift, raise in the worker if any entry of `objective_history` or `elapsed_seconds` is non-finite, and record `accepted_steps` in the summary.

R4. The primary metric text is inexact. `native_teacher_mse` at task line 199 says a mean over full-canvas pixels. `_evaluate_measure` averages over each teacher's fit window at `src/rtgs/optim/compact_trainer.py:2091` and equal-weights views at `:2152`. For masked teachers that window is the mask crop with margin, per `src/rtgs/image2gs/fit.py:792`. Fix: state that the mean is over the frozen teacher fit window at half-integer centres, equal-weighted per view, and that masked and maskless denominators differ.

R5. Embedded alpha use is not declared. Masked arms load the lossless compact alpha at driver line 224 and use it for source-support gating through `mask_mode` hard at driver line 85 and `src/rtgs/lift/field_lifter.py:890`. The packet's statement of report-only mask use is true only for the maskless condition. Fix: add an explicit statement to the task, under `input_policy` or `frozen_configuration.field_lift`, that masked conditions consume the compact-embedded alpha for placement support gating and the unmasked condition consumes none, and record `load_alpha` and `mask_mode` in each worker's input-boundary record.

R6. The claim boundary omits the maskless background. The Haelyn views are offline renders on a black background at prepare lines 104 to 118, so the maskless condition has a uniform background and does not represent a cluttered capture. Add this to `claim_boundary` or the converter note before any outcome exists.

### Optional improvements

- O1. `configs` hard-codes the warmup iteration counts at driver lines 78 and 89 instead of reading `frozen_configuration.warmup`. Read the frozen values so the record is authoritative.
- O2. `wall_seconds` includes validation observer time. It is reported separately, but an observer-excluded endpoint time in the groups and paired ratios would be cleaner.
- O3. Soft and free arms add five parameters per track to one Adam optimizer and one global gradient-norm clip at `src/rtgs/lift/field_refit.py:648`. This is inherent to the treatment; say so in the comparator purpose.
- O4. Beam also differs from the causal arms by never consuming embedded alpha, because `to_reconstruction_inputs` drops it. Mention this in the Beam comparator purpose.
- O5. The README retry convention under `runs/<task_id>/attempts/` is not implemented. Either implement it or state that any failure closes this task id.
- O6. A committed baseline before `init-run` would avoid a development lock with a dirty diff hash. That needs user authorization.

### Assessment of the design

The sealed budgets can answer the bounded development question descriptively: paired soft-minus-hard and free-minus-hard differences on held-out teacher MSE, before and after refinement, per capture and per mask condition, over three seeds. Fixed endpoints, prospectively fixed tether weights, deterministic placement with byte-level initial equality, immutable teachers, frozen effective-configuration hashes, and a separate report-only RGB evaluator are sound. Three seeds support sign consistency only, which the task already states. The proxy-stage comparison happens at the fixed initial render opacity of 0.1, so before-refinement differences reflect geometry and visibility under dim compositing; the boundary may say so. Nothing in this protocol should be read as evidence of physical density, a default improvement, or full-capacity quality.

### What the follow-up exact-digest review needs

1. The edited task with `blockers` empty, the R4, R5 and R6 wording, and any `failure_policy` change, still `draft` with the review envelope pending.
2. The fixed driver and report scripts, a refreshed `source_binding` with unchanged patterns and a new file count and aggregate, and unchanged data seals unless data changed. The effective-configuration table should not change because no dataclass changes are required.
3. A test covering `aggregate` from the repository root with the frozen relative run argument, and a test for the coordinator failure receipt path.
4. The new `review-digest` value, plus permission for the reviewer to execute `review-digest`, `validate`, `validate-data`, and a source-binding recomputation, so the follow-up can bind an independently recomputed digest.
5. This review preserved verbatim as `experiments/reviews/20260906_tomography_source_constraints_haelyn_dome_PROTOCOL_REVIEW_V1_REJECTED.md`. If the Driver instead copies this rejection into `protocol_review`, status must become `blocked` and the artifact must use the canonical name; because the protocol changes immediately, preserving the rejected file and keeping draft with a pending envelope is the simpler valid path.
6. After approval: copy the metadata, set `ready`, run `validate`, then `init-run`, with `--development` if the tree is still dirty, and execute the frozen command unchanged.

## Protected Actions Not Taken

I did not run `init-run`, the coordinator, any worker cell, the warmup, render-reference, the converter, the import replay, the phase evaluator, `aggregate`, `render`, `check-run`, or the results-bundle gate. I did not open any file under `runs/` or `benchmarks/results/`, and no `.ply`, `.splat`, `.rtgsv`, image or mask. I read only source, tests, task, seal, manifest, receipt and log metadata, and I ran only the two permitted CPU contract test files plus read-only git and hashing commands. I modified no source, task state, review file, historical evidence, git state or configuration. Outcome Access remained none throughout.
