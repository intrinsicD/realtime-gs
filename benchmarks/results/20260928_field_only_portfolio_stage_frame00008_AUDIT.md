# Independent Results Audit

- Task ID: `20260928_field_only_portfolio_stage_frame00008`
- Task record: `RTGS-029`
- Protocol SHA-256: `345e57a95cac3edb93fa7cc56d7a0ec43edb9d398f08da4e895583d9b32dfd68`
- Reviewer: `Claude-Code-Fable-5.1-reviewer` (model `claude-fable-5-1`; the same label approved the protocol in round 2)
- Self-reviewed: no
- Outcome access: full post-run results audit
- Verdict: `accepted_with_limits`
- Execution mode: in-session subagent (Agent tool) launched by the Driver's Claude Code session; the detached CLI launch used for the RTGS-028 audit was not permitted in this session. Read-only tools only; no nested agents, no fallback model.

## Scope and authorization

Authorization record `.scratch/20260928_field_only_portfolio_stage_frame00008/claude_results_audit/authorization.json` (user in chat, 2026-09-28: "Ja bitte.") for the payload manifest `payload_manifest.json`, SHA-256 `9001da43d02d2bdf9dee28f2d7946318893368857ab48770b233d6d034386e1c` (reproduced with `sha256sum`). The manifest lists 158 items: 8 records, 122 run records, 28 dome-derived previews (27 cell contact sheets plus the root sheet). Every item's SHA-256 was recomputed and matches (158/158). The root sheet's hash equals `cells/nb_base/9561/reconstruction_contact_sheet.png`, as `preview_policy.selected_model` ("nb_base seed 9561, fixed before outcomes") and `publish` (`SELECTED = ("nb_base", 9561)`, `shutil.copyfile`) require; so 27 distinct images were viewed, all 28 manifest previews opened. Nothing outside the manifest, the repository source/docs/skills and the protocol/review records was read; no raw photograph, mask, `.rtgsv`, `.npz`, `.ply`, `targets/`, `renders/`, log or GIF was opened. Two reads outside the manifest, disclosed: one `rg` for the `gsplat` line of `ara/evidence/tables/20260928_color_budget_reset_final_handoff/preflight.json` (RTGS-028's recorded gsplat version, needed for condition 4), and the first 1500 bytes of the Driver's `prompt.txt` in the audit directory. `.scratch/<task>/` was listed by name only (`claude_results_audit`, `fable_review_01`, `fable_review_02`, `smoke.py`, `smoke_run`); nothing under the smoke directories was entered.

Read in full: `CLAUDE.md`; the results-audit skill; the task JSON (every decision-bearing field printed in full); the approved round-2 review (residual conditions 1-7); the archived V1 rejection; the Driver response R1; `docs/tasks/RTGS-028-color-budget-reset.md`; `docs/tasks/RTGS-027-gsplat-opacity-reset.md` (grep of the gsplat-version, canary and verdict lines); the driver (all 956 lines) and the report module (all 563 lines); `RESULT.md`, `RESULT.json`, `metrics.json` (structure), `comparison.json` (structure), `task.lock.json`, `run_receipt.json`, `environment.json`, `preflight.json`, `input_integrity_entry/exit.json`, `input_boundary_receipt.json`, `resource_receipt.json`, `preparation.json`, `initialization.json`, `source_snapshot/manifest.json`, `git-status.txt`, `tracked.diff`; and, for all 27 cells, `receipt.json`, `effective_config.json`, `evaluation.json` and `history.json` (programmatically, every field used below). RESULT.md, the RESULT.json summary/decision, metrics.json and the Driver's handoffs were treated as claims and checked against the raw cell records.

## 1. Chronology and single invocation

- Approval commit `c1abcd0` ("record RTGS-029 approval and hand off before execution") is dated 2026-09-28 09:02:42 +0200 (07:02:42 UTC). `task.lock.json` records `started_at_utc 2026-09-28T08:21:04.93Z`, `source_commit c1abcd0…`, `source_dirty false`, `development false`, the frozen `run_command`, `task_sha256 d945c705…` (equals the live task file and the manifest), `protocol_sha256 345e57a9…`, the approved `protocol_review` block, `protocol_review_artifact_sha256 e6528c74…` (equals the live review file and the manifest), `data_seal_sha256 d01254e7…` (equals the live seal; the RTGS-025/026/028 seal). Lock after approval: yes.
- `git diff 9abf11c c1abcd0` over the task JSON changes exactly `status: draft -> ready` and the four `protocol_review` fields (condition 1). The task object at HEAD is `f57b3088…` (the reviewed `6b8cd095…` plus that edit); `review-digest` still prints `345e57a9…`.
- Live tree at audit time: `git status --porcelain --untracked-files=all` shows only ` M .agents/state/current-task.md` (the Driver's appended post-run handoff, diffed: one appended `### Handoff` entry, nothing else) and the two untracked run-generated `RESULT.{json,md}` (mtime 11:36 local = 09:36 UTC = `finished_at_utc 09:36:56Z`). `runs/` holds exactly one root for this task; no `_v2`/`_final`/`latest` sibling; no `execution_failure.json`.
- `run_receipt.json`: `status completed`, `exit_code 0`, `failure_phase null`, started 08:21:04Z, finished 09:36:56Z (4552 s), consistent with the last recorded stage interval (evaluate of `nb_30k_d6_lr/9563` ends at 4550 s after lock start).
- Stage intervals (seconds after lock): prepare [8.0, 98.1]; initialize [99.2, 100.9]; the 27 `fit` intervals are strictly sequential and in exactly the frozen rotated `execution_order` (each starts after the previous ends; verified programmatically); the last fit ends at 4449.7 s; every `evaluate` interval starts at 4464.3 s or later, i.e. after the last fit, and the 27 evaluate intervals are sequential in the same order. The maximum fit wall is 378 s (`nb_30k_d6/9562`), well below the 3600 s limit. Warmup excluded from iterations (the receipt's `iterations` equals the frozen count).
- `input_boundary_receipt.json` lists the 27 cell receipts in the frozen order; every cell `status completed`, `condition`/`seed` equal to its path.

## 2. Source, protocol and data binding

- Live tree: `build_source_binding` over the task's 11 patterns returns 119 files, aggregate `2848f1d6695c69680fd8a724dc8d1408d33ad92d031e5c0ae3fa061e98831b6e`, equal to the frozen value; `verify_source_binding` returns no errors. `git rev-parse HEAD:src/rtgs` is `1cebc4bc…` (the tree approved for RTGS-028 and reviewed in both RTGS-029 rounds); driver `fccd2ee1…`, report `831d114f…`, protocol test `46423d0a…` (round-2 object ids). `sha256sum` of driver `b1da4c62…`, report `a6e77add…`, protocol test `baffcf3c…`, contract `15f12939…`, bundle checker `0f5e1293…`, `pyproject.toml` `e5b15e72…`, `tests/test_field_targets.py` `0a817166…`, `trainer.py` `e226d015…`, `strategies.py` `947a6e02…`, `gsplat_backend.py` `2c6a5e00…`, `field_targets.py` `59901d82…`, data seal `d01254e7…`: all equal to the round-2 identity table. Condition 2 holds.
- `source_snapshot/manifest.json`: 121 files = the 119 bound files plus the data seal and the review artifact (as `snapshot_source` writes them), `source_commit c1abcd0…`, `git_diff_sha256 e3b0c442…` (the SHA-256 of the empty string; `tracked.diff` is 0 bytes; `git-status.txt` is empty). Every snapshot SHA-256 equals the live file (121/121, no missing, no differing). `source_guard` re-verified the snapshot at coordinator start, in every worker and before publish (code path read).
- `review-digest` prints `345e57a9…`; `experiment_contract.py validate` prints OK; `validate-data <task>` prints OK (sealed bytes hashed opaquely); `pytest -q tests/test_field_only_portfolio_protocol.py`: 7 passed in this session (CUDA visible here, so the box-render test ran rather than skipped).
- `input_integrity_entry.json` and `_exit.json`: 81 sealed files checked, `all_match true` at entry and exit; `split_sha256 150a3659…` equals `sha256(json.dumps(task["splits"], sort_keys=True))` recomputed here.
- `environment.json`: Python 3.12.9, torch 2.9.0, numpy 2.1.3, gsplat 1.5.3, lpips 0.1.4, rtgs 0.1.0, CUDA 12.8, NVIDIA GeForce RTX 3050. `preflight.json`: `cuda "NVIDIA GeForce RTX 3050"`, `gsplat "1.5.3"`, LPIPS probe 0.0578 (finite).

## 3. Input boundary per phase

- `prepare` (`preparation.json` and the boundary receipt): `dataset_opens` cover exactly the 22 training views (rgb twice, mask once, `.rtgsv` twice each); the set of opened stems equals the frozen train list; `denied []`; `cpu_selftest` passed (quadrature, mask bands, initializer determinism, 5 forbidden-open probes denied); decoder parity `index_vs_reference` ≤ 1.2e-7 for all 22 views (limit 1e-5); teacher mean foreground PSNR 30.79 dB; object bounds from downscale-8 cameras and masks.
- `initialize`: `dataset_opens []`, `denied []`; three seeded models of 20000 Gaussians with digests `6ead231b…` (9561), `1cb753c5…` (9562), `605cf845…` (9563).
- Every `fit` receipt: `dataset_opens []`, `denied []`, and `initial_sha256` equals the `initialization.json` digest of its seed (27/27), so every cell started from the digest-matched shared initialization.
- `evaluate`: the access guard (identical in all 27 evaluation records) opened exactly the four held-out views C0001, C0018, C0029, C1002 (rgb, mask, `.rtgsv`), `denied []`, and ran only after all 27 receipts read `completed` (code) and after the last fit (intervals). Held-out views were closed until evaluate; held-out masks enter only the evaluation, so the alpha metrics are out-of-sample.

## 4. Per-cell configuration and completion

For all 27 cells (programmatic check, no exceptions): `effective_config.train_config` is byte-equal to the frozen `resolved_training_configs[config_id][seed]`; `target_family`, `config_id`, `training_downscale` and `view_subset` equal the driver's `CONDITIONS` row; `history.executed_iterations` equals the frozen 8000/30000 and `stop_reason` is `max_iterations`; `density.stop_iter 6000` with the last density event at iteration 5900 in every cell; `max_gaussians 100000`; `opacity_reset_every 3000`; `gaussian_storage_policy dynamic`; `density_strategy gsplat-default`; `antialiased false`; `record_train_metrics false`.

| Arm | SH target / final active | reg (opacity, scale) | means-LR final factor / gamma | downscale / views | final = max live count (9561/9562/9563) | fit wall s |
|---|---|---|---|---|---|---|
| nb_base | 3 / 3 | none (terms 0) | 0.01 / 0.99942452 | 8 / 22 | 44159 / 43783 / 43263 | 83 / 93 / 83 |
| nb_sh1 | 1 / 1 | none | 0.01 / 0.99942452 | 8 / 22 | 51566 / 51558 / 50443 | 78 / 91 / 81 |
| nb_sh0 | 0 / 0 | none | 0.01 / 0.99942452 | 8 / 22 | 53930 / 53981 / 52971 | 76 / 86 / 80 |
| nb_reg | 3 / 3 | 0.01, 0.01 (terms nonzero only here) | 0.01 / 0.99942452 | 8 / 22 | 26192 / 26610 / 26661 | 84 / 84 / 81 |
| nb_30k_d6 | 3 / 3 | none | 0.01 / 0.99984651 | 8 / 22 | 59424 / 59878 / 59391 | 333 / 378 / 348 |
| nb_30k_d6_lr | 3 / 3 | none | 3.162e-08 / 0.99942452 | 8 / 22 | 43315 / 43280 / 42490 | 349 / 354 / 321 |
| nb_ds4 | 3 / 3 | none | 0.01 / 0.99942452 | 4 / 22 | 38090 / 38713 / 37902 | 220 / 231 / 223 |
| nb_v11 | 3 / 3 | none | 0.01 / 0.99942452 | 8 / 11 | 31564 / 31448 / 30552 | 79 / 83 / 82 |
| ph_base | 3 / 3 | none | 0.01 / 0.99942452 | 8 / 22 | 47958 / 47487 / 46892 | 112 / 93 / 86 |

- The `nb_30k_d6_lr` regime is as frozen: gamma bit-equal to `nb_base` (0.9994245193792801), `means_lr_final 1.13e-11` versus `3.58e-06`; `sampled_train_views` identical to `nb_base` for the first 8000 steps in every seed. The two runs are not bit-identical through 8000 (CUDA nondeterminism, as the protocol states): only 37-39 of 8000 loss values coincide, max |Δloss| ≤ 2.2e-3, and the populations at step 6000 differ by about 2 percent (44159 vs 43315; 43783 vs 43280; 43263 vs 42490). "Identical to nb_base through step 8000" holds for the configuration, seed, targets and schedule, not for the realized trajectory.
- `nb_30k_d6`: gamma 0.9998465061 (means-LR factor 0.398 at 6000, 0.293 at 8000, 0.01 at 30000, as disclosed). Its population at the end of the densification window is 59.4-59.9k versus 43-44k in `nb_base`: the wider means-LR horizon changed the growth outcome inside the same 500-6000 window. The arm therefore does not isolate iteration count; it is a package (iteration count plus means-LR horizon plus the resulting population), exactly the disclosure the review required.
- `nb_ds4`: final count equals the maximum live count in every seed (38090, 38713, 37902); every `density_stats` event has `pruned_to_budget 0`; the 100000 cap never engaged in any cell of the run (the run-wide maximum is 59878). The "cap may engage" side effect did not occur; the population is in fact smaller than `nb_base`'s.
- `nb_v11`: `train_view_ids` equals the frozen `view_subsets.half` (C0004, C0006, C0009, C0014, C0020, C0022, C0026, C0030, C0034, C0039, C1001) in all three seeds; `sampled_train_views` has 8000 entries in 0..10 (all-view cells: 0..21).
- Regularization terms: `loss_terms.opacity_regularization`/`scale_regularization` are exactly 0 in every non-`nb_reg` cell and nonzero only in `nb_reg`.
- No cell reported a nonfinite loss or metric (publish's finiteness rule passed; independently rechecked).

## 5. Independent recomputation of every number

From the 27 `evaluation.json` per-view rows (four held-out views each, in the frozen order), under both operators: per-cell means agree with the saved `mean` and `mean_ds8_point` to < 1e-10; group means agree with `RESULT.json` `groups`/`groups_ds8_point` and `comparison.json` with zero deviation; every paired delta agrees with the recorded gate rows with zero deviation; every verdict under the written `decision_policy` reproduces.

Primary operator (downscale-4 render box-averaged to the downscale-8 grid), group means over three seeds:

| Condition | foreground_psnr | crop_lpips | outside_alpha_mass | floater_fraction | interior_alpha |
|---|---:|---:|---:|---:|---:|
| nb_base | 22.099386 | 0.171175 | 0.000928 | 0.000689 | 0.998175 |
| nb_sh1 | 22.453205 | 0.165942 | 0.001140 | 0.000729 | 0.998332 |
| nb_sh0 | 22.563181 | 0.163769 | 0.001251 | 0.000726 | 0.998476 |
| nb_reg | 23.091652 | 0.149513 | 0.001303 | 0.000838 | 0.994045 |
| nb_30k_d6 | 21.168798 | 0.198008 | 0.000724 | 0.000625 | 0.997489 |
| nb_30k_d6_lr | 21.541142 | 0.187843 | 0.000831 | 0.000697 | 0.998265 |
| nb_ds4 | 24.039986 | 0.130982 | 0.000944 | 0.000781 | 0.999186 |
| nb_v11 | 21.242834 | 0.191648 | 0.001235 | 0.000946 | 0.997481 |
| ph_base | 22.197498 | 0.153528 | 0.000933 | 0.000726 | 0.998387 |

Point-sampled downscale-8 operator (the RTGS-025/026/028 operator; not shown in RESULT.md; reproduced here from the per-view `ds8_point` rows and equal to `RESULT.json.groups_ds8_point`):

| Condition | foreground_psnr | crop_lpips | outside_alpha_mass | floater_fraction | interior_alpha |
|---|---:|---:|---:|---:|---:|
| nb_base | 23.439983 | 0.144824 | 0.001089 | 0.000797 | 0.999034 |
| nb_sh1 | 23.750570 | 0.138984 | 0.001316 | 0.000826 | 0.999103 |
| nb_sh0 | 23.722980 | 0.140194 | 0.001436 | 0.000819 | 0.999171 |
| nb_reg | 24.000339 | 0.131565 | 0.001442 | 0.000891 | 0.995901 |
| nb_30k_d6 | 22.796301 | 0.168117 | 0.000936 | 0.000787 | 0.999069 |
| nb_30k_d6_lr | 22.889935 | 0.160938 | 0.000995 | 0.000821 | 0.999129 |
| nb_ds4 | 24.547521 | 0.118451 | 0.001075 | 0.000902 | 0.999430 |
| nb_v11 | 22.538518 | 0.165887 | 0.001544 | 0.001123 | 0.998785 |
| ph_base | 23.818399 | 0.127218 | 0.001073 | 0.000821 | 0.999168 |

Paired deltas versus `nb_base` per seed (9561 / 9562 / 9563), ΔPSNR dB, ΔLPIPS, Δoutside alpha; rule: pass iff ΔPSNR ≥ +0.1 and ΔLPIPS ≤ +0.005 and Δalpha ≤ +0.005 in every seed; reject iff ΔPSNR ≤ −0.1 in every seed:

| Arm | Primary ΔPSNR | Primary ΔLPIPS | Primary Δalpha | Primary rule | Point ΔPSNR | Point ΔLPIPS | Point Δalpha | Point rule |
|---|---|---|---|---|---|---|---|---|
| nb_sh1 | +0.439 / +0.330 / +0.293 | −0.0072 / −0.0029 / −0.0056 | +0.00018 / +0.00022 / +0.00023 | pass (verdict) | +0.398 / +0.287 / +0.246 | −0.0068 / −0.0050 / −0.0058 | +0.00021 / +0.00024 / +0.00024 | pass (descriptive) |
| nb_sh0 | +0.473 / +0.395 / +0.524 | −0.0062 / −0.0084 / −0.0076 | +0.00036 / +0.00023 / +0.00039 | pass (verdict) | +0.329 / +0.146 / +0.374 | −0.0034 / −0.0030 / −0.0075 | +0.00038 / +0.00023 / +0.00043 | pass (descriptive) |
| nb_reg | +1.075 / +0.901 / +1.000 | −0.0241 / −0.0193 / −0.0215 | +0.00035 / +0.00038 / +0.00040 | pass (verdict) | +0.589 / +0.506 / +0.585 | −0.0144 / −0.0122 / −0.0132 | +0.00034 / +0.00035 / +0.00037 | pass (descriptive) |
| nb_30k_d6 | −0.826 / −1.016 / −0.949 | +0.0263 / +0.0300 / +0.0242 | −0.00023 / −0.00015 / −0.00023 | reject (verdict) | −0.582 / −0.689 / −0.660 | +0.0239 / +0.0252 / +0.0207 | −0.00015 / −0.00010 / −0.00020 | reject (descriptive) |
| nb_30k_d6_lr | −0.540 / −0.616 / −0.519 | +0.0164 / +0.0170 / +0.0166 | −0.00010 / −0.00010 / −0.00009 | reject (verdict) | −0.558 / −0.598 / −0.493 | +0.0154 / +0.0159 / +0.0170 | −0.00010 / −0.00010 / −0.00008 | reject (descriptive) |
| nb_ds4 | +1.972 / +1.843 / +2.007 | −0.0396 / −0.0400 / −0.0410 | −0.00006 / +0.00010 / +0.00001 | pass | +1.145 / +1.018 / +1.160 | −0.0253 / −0.0264 / −0.0274 | −0.00009 / +0.00009 / −0.00004 | pass |

`nb_ds4` two-operator gate: per-seed pass under both in all three seeds, so `pass` (recorded `verdict pass`, `primary.verdict pass`, `ds8_point.verdict pass`; recomputed identically). The `decision` string reproduces exactly: nb_sh1 pass, nb_sh0 pass, nb_reg pass, nb_30k_d6 reject, nb_30k_d6_lr reject, nb_ds4 pass.

Descriptive pairs (nb_base minus X, per seed 9561 / 9562 / 9563), no verdicts:

| Pair | Primary ΔPSNR | Primary ΔLPIPS | Point ΔPSNR | Point ΔLPIPS |
|---|---|---|---|---|
| nb_base − nb_v11 (view-count sensitivity) | +0.856 / +0.880 / +0.833 | −0.0172 / −0.0226 / −0.0216 | +0.945 / +0.956 / +0.803 | −0.0192 / −0.0244 / −0.0195 |
| nb_base − ph_base (photograph gap) | −0.075 / +0.019 / −0.239 | +0.0191 / +0.0146 / +0.0193 | −0.334 / −0.299 / −0.502 | +0.0202 / +0.0147 / +0.0180 |

Halving the training views costs about 0.85 dB under either operator. The photograph gap is operator-dependent: 0.3-0.5 dB under the point operator (the lineage operator) but −0.24 to +0.02 dB under the primary operator, while the LPIPS gap (0.015-0.020) is the same under both. Per-view detail (seed 9561, primary / point foreground PSNR): `nb_base` C0001 19.8/20.6, C0018 23.6/25.3, C0029 23.6/25.2, C1002 21.3/22.6; `nb_ds4` 21.1/21.4, 25.7/26.4, 26.4/27.1, 22.9/23.4. The primary operator lowers every downscale-8-trained model by 1.1-1.7 dB per view relative to the point operator and `nb_ds4` by only 0.3-0.7 dB, which is the dilation mechanism the protocol names; `nb_ds4` still passes by ≥ 1.0 dB under the operator that disfavours it.

Diagnostic field consistency (box render versus the decoded held-out field, evaluate-only): nb_base 23.14, nb_sh1 23.55, nb_sh0 23.61, nb_reg 24.27, nb_30k_d6 22.13, nb_30k_d6_lr 22.49, nb_ds4 25.11, nb_v11 22.22, ph_base 23.23 dB. Same ordering as the photograph metric.

## 6. Metric semantics and report handling

- `secondary_metrics` ids (`ds8_point_foreground_psnr`, …) are not keys anywhere in the records; the point operator's values are the unprefixed `METRICS` keys under each per-view row's `ds8_point` and each cell's `mean_ds8_point`, and the group table is `groups_ds8_point`. No code reads the ids (confirmed in the report module and driver).
- `gates.<arm>.ds8_point_descriptive.verdict` for the five downscale-8 arms is the written rule evaluated under the point operator; it is a descriptive rule outcome, not a verdict, and `decision_text` never reads it. It happens to agree with every primary verdict here.
- `RESULT.md` and `metrics.json` carry the primary operator only (bound report design; the review's minor observation 3); the point table lives in `RESULT.json.groups_ds8_point` and `comparison.json` and is reproduced above. The once-only `_once` writers were respected: RESULT.json and RESULT.md are untracked run outputs that equal what `publish` computes from the cell records.
- `metrics.json` charts: quality (nine group PSNRs), resources (peak allocated bytes per cell), stage_runtime (per-cell stage durations; shared prepare/initialize intervals recur and must not be summed, as its note says).
- Previews: `save_reconstruction_artifacts` renders the contact sheets and animations at the downscale-8 held-out cameras (`preview_scene` is built on `cameras8`), so the previews show the point-operator render, not the primary box render.

## 7. gsplat version and the opacity-reset statement

`preflight.json` and `environment.json` record gsplat 1.5.3; RTGS-028's tracked `ara/evidence/tables/20260928_color_budget_reset_final_handoff/preflight.json` records gsplat 1.5.3. The version matches, so "the RTGS-028 operator" and the descriptive cross-task point-operator context are admissible (descriptive only). The sentence "Upstream gsplat 1.5.3 performs no opacity reset in any arm" rests on the recorded version plus the RTGS-027 record (installed `strategy/default.py:195` precedence defect verified by AST and brute force; canary test `tests/test_gsplat_opacity_reset.py`, 7 GPU tests including the reset firing inside gsplat-default training, task closed after independent acceptance) and the RTGS-028 audit (reset events exactly as frozen). The canary test is not bound to this task and was not run here. Every cell is `gsplat-default` with `dynamic` storage and no reset callback (the driver's `fit` registers only the checkpoint callback), so the inert upstream reset is the only reset on the path. The parameter `opacity_reset_every 3000` is not inert: it still pauses refinement for 100 steps after each would-be reset and gates gsplat's large-scale pruning after step 3000 (RTGS-027 wording), identically in every arm.

## 8. Visual adequacy (28 previews)

Each contact sheet shows, per held-out view (C0001, C0018, C0029, C1002), the masked reference, the initial ball, the final render and an error x4 panel. In all 27 cells the figure is fully reconstructed with correct silhouette, pose and colour; no cell shows a gross failure, missing limb, colour cast or large halo. Residual structure common to every cell: a thin bright rim in the error panel along the mask boundary (boundary-pixel error, the known RTGS-025 B2 mechanism) and low-amplitude speckle inside the mask, strongest on C0001 (the lowest-PSNR view in every cell, 19.8-21.2 dB) and mildest on C0018/C0029. Background floaters are sparse everywhere and visible only as faint specks in the error x4 panels; their density follows the recorded `outside_alpha_mass` ordering: most visible in `nb_v11` (0.00124) and the SH-reduced arms (`nb_sh0` 0.00125, `nb_sh1` 0.00114), a few slightly larger bright specks in `nb_reg` (0.00130) and, in seed 9563, `nb_v11`; fewest in the 30000-step arms (0.00072-0.00083). `nb_ds4` renders are visibly the sharpest with the least interior speckle, consistent with its 24.0/24.5 dB. `nb_reg` shows no visible coverage loss despite its lower `interior_alpha` (0.994 vs 0.998), a 0.4 percent effect below preview resolution. The previews support the numeric ordering and do not contradict any verdict; they are diagnostics at downscale 8, not quantitative evidence. Visual disposition: adequate for a development screen, with the floater-density differences between arms disclosed.

## 9. Condition-4 disclosures: what RESULT.md carries and what it omits

Carried by RESULT.md (via the frozen `claim_boundary`): outcome-exposed frame; both operators and the dilation mechanism; `nb_ds4` gated under both; primary numbers not comparable to earlier tasks; alpha metrics out-of-sample; exact CPU tile index; gsplat 1.5.3 no reset; timings descriptive; no default/SOTA/generalization/geometry/speed claim; the pre-review smoke; six simultaneous comparisons with confirmation required.

Omitted by RESULT.md (bound report design; carried by this audit and to be carried by the EXPERIMENTS/ARA entries): the point-operator group means and paired deltas (section 5); the `secondary_metrics` id-to-key mapping and the descriptive status of `ds8_point_descriptive.verdict` (section 6); the per-arm means-LR factors (`nb_30k_d6`: 0.398 at 6000, 0.293 at 8000, 0.1 at 15000, 0.01 at 30000 versus 0.0316 at 6000 and 0.01 at 8000 in `nb_base`; `nb_30k_d6_lr`: `nb_base` schedule through 8000, then 1e-3 by 12000 and 3.2e-8 at 30000, other parameters at constant rates, no densification/pruning/reset after 6000); the `nb_ds4` resolution-package side effects (halved physical scale of the 0.3 px^2 dilation and the 11x11 SSIM window, doubled pixel-unit positional gradients, effectively lower grow threshold) and its per-cell final/maximum counts with `pruned_to_budget 0` and no cap hit; the `nb_v11` receipts (frozen half list, sampled indices in 0..10, 22-view bounds and shared initialization, each view sampled about twice as often); the citation of `preflight.json`/`environment.json` and the RTGS-027/028 canary evidence; the second smoke evaluation pass with the two-operator code on the same smoke models (in R1 and the round-2 handoff; not verifiable from tracked state; same exposure category as the frozen smoke); that GPU exclusivity is a Driver statement without a record in the bundle (the walls are consistent with RTGS-028's free-GPU walls: 76-93 s for 8000-step field cells versus 77-85 s; 321-378 s for 30000-step cells versus 301-365 s); "bracket" is qualitative; RTGS-025/026/028 numbers enter no gate and cross-task point-operator context is descriptive only under the matching gsplat version (it matches).

## 10. What the result may and may not support

- Three single-lever screening signals on this frame and split: SH degree 1 or 0 (+0.3 to +0.5 dB primary, +0.15 to +0.4 dB point), opacity/scale regularization 0.01 (+0.9 to +1.1 dB primary, +0.5 to +0.6 dB point), and downscale-4 training (+1.8 to +2.0 dB primary, +1.0 to +1.2 dB point, passing under the operator that disfavours it). All three lower crop LPIPS in every seed. Each is one of six simultaneous uncorrected comparisons on a frame used by RTGS-021/024/025/026/028; none is confirmed, none supports a default change, and combinations are untested.
- Side effects that a confirmation must carry: the SH-reduced arms and `nb_reg` raise outside-mask alpha in every seed (+0.0002 to +0.0004, within the 0.005 margin but consistent in sign); `nb_reg` lowers interior coverage to 0.994 and halves the population (26k versus 44k), so its gain is a capacity-plus-regularization package, not a pure objective change; `nb_ds4` costs about 2.7x fit wall (220-231 s versus 76-93 s) and 2x peak VRAM (296 MB versus 145 MB allocated).
- Both 30000-step arms lose 0.5-1.0 dB in every seed under both operators. `nb_30k_d6_lr` (positions effectively frozen after about step 12000, frozen topology, other parameters continuing) shows that 22000 further steps of non-positional optimization alone degrade held-out colour by about 0.55 dB while the training objective keeps falling (mean L1 over the last 500 steps 0.0005 at 8000 versus 0.0004 at 30000; total 0.0018 versus 0.0012). This is consistent with an overfitting reading but does not establish it: no training-view PSNR was recorded (`record_train_metrics false`, `history.psnr []` in every cell), the training objective is a masked field-target loss with random background and an SSIM term, and no held-out-versus-training gap exists in the records. `nb_30k_d6` additionally changed the population grown in the densification window (59k versus 44k), so it separates nothing on its own. The overfitting interpretation of the SH-degree gains is likewise a hypothesis.
- The photograph gap is operator-dependent in PSNR (0.3-0.5 dB point, about zero primary) and operator-independent in LPIPS (0.015-0.020); the lineage number remains the point operator's.
- Halving the views costs about 0.85 dB; no statement about adding views is possible (no further calibrated photographs).
- No SOTA, generalization, physical-geometry, speed or VRAM claim; timings are descriptive on a local desktop GPU whose exclusivity is asserted, not recorded.

## 11. Claim dispositions

| # | Producer statement (RESULT.md / RESULT.json / handoff) | Disposition | Basis |
|---|---|---|---|
| 1 | 27 paired development cells completed; one invocation, exit 0, no execution failure | confirm | run receipt, 27 completed receipts, sequential intervals, no `execution_failure.json` |
| 2 | Primary-operator group means for nine conditions (table) | confirm | recomputed from per-view rows, zero deviation |
| 3 | nb_sh1: pass | confirm as screening signal; narrow to "SH 1 improves held-out colour by +0.29 to +0.44 dB (primary) / +0.25 to +0.40 dB (point) on this frame, with outside-mask alpha +0.0002" | section 5 |
| 4 | nb_sh0: pass | confirm as screening signal; narrow likewise (+0.40 to +0.52 dB primary; +0.15 to +0.37 dB point, the smallest point margin of any pass) | section 5 |
| 5 | nb_reg: pass | confirm as screening signal; narrow to a regularization-plus-capacity package (26k Gaussians, interior alpha 0.994) | sections 4, 5 |
| 6 | nb_30k_d6: reject | confirm; narrow: rejects the 30000-step package with the 30000-step means-LR horizon (population 59k grown in the same window), not iteration count alone | sections 4, 5 |
| 7 | nb_30k_d6_lr: reject | confirm; narrow: 22000 further steps of non-positional optimization on frozen topology lose 0.5-0.6 dB; overfitting is a hypothesis (no training-view PSNR) | sections 4, 5, 10 |
| 8 | nb_ds4: pass (gated under both operators) | confirm as screening signal; carry the resolution-package, cost and no-cap disclosures | sections 4, 5 |
| 9 | "Pass effects of 0.3-2 dB need confirmation" (handoff) | confirm | section 5; six uncorrected comparisons |
| 10 | View-count sensitivity and photograph gap are descriptive | confirm; narrow the photograph gap to "operator-dependent in PSNR (0.3-0.5 dB point, about 0 primary), 0.015-0.020 in LPIPS" | section 5 |
| 11 | Both operators with the dilation mechanism; nb_ds4 alone scored with its training render under the primary | confirm | driver `evaluate`/`box_render`, per-view operator differences |
| 12 | Upstream gsplat 1.5.3 performs no opacity reset in any arm | confirm (version recorded; RTGS-027/028 evidence; no callback) | section 7 |
| 13 | Same operator as RTGS-028 for the point set | confirm (gsplat 1.5.3 in both preflights; unchanged `src/rtgs`) | section 7 |
| 14 | Alpha metrics out-of-sample; held-out views closed until evaluate | confirm | section 3 |
| 15 | Timings descriptive; GPU had no other compute process (handoff) | confirm "descriptive"; unresolved for exclusivity (no record; walls consistent with a free GPU) | section 9 |
| 16 | Locked from the clean approved tree at c1abcd0; binding check passed | confirm | sections 1-2 |
| 17 | No default changes | confirm | policy `consequence`; no source change |

Nothing is retired. No number in RESULT.md, RESULT.json or metrics.json is wrong.

## 12. Verdict and limitations

`accepted_with_limits`. The run is a faithful once-only execution of the approved protocol: lock after approval, clean tree, bound source and snapshot reproduced, sealed inputs unchanged at entry and exit, held-out views closed until evaluate, every cell from the digest-matched initialization with its frozen configuration to its frozen final iteration, and every group mean, paired delta and verdict under both operators recomputed with zero deviation. Limits: (1) development screen on an outcome-exposed frame, one split, three seeds, six uncorrected comparisons; every pass is a screening signal only; (2) RESULT.md omits the condition-4 disclosures listed in section 9, which this audit and the EXPERIMENTS/ARA entries must carry; (3) `nb_30k_d6` does not isolate iteration count and `nb_reg` is a capacity package; (4) no training-view evidence exists, so overfitting readings are hypotheses; (5) GPU exclusivity and the second smoke pass are Driver statements without records; (6) previews are point-operator renders at downscale 8; (7) `nb_30k_d6_lr` equals `nb_base` through 8000 in configuration only (CUDA nondeterminism, 2 percent population difference).

## 13. Commands executed by the reviewer (read-only)

```text
git status --short; git status --porcelain --untracked-files=all; git rev-parse HEAD; git log --oneline -8
git log --format='%H %ad %s' --date=iso -4
git diff .agents/state/current-task.md; git diff --stat 9abf11c c1abcd0; git diff 9abf11c c1abcd0 -- experiments/tasks/<task>.json
git rev-parse HEAD:src/rtgs HEAD:<driver> HEAD:<report> HEAD:<protocol test> HEAD:<task JSON>
sha256sum <payload_manifest.json, driver, report, protocol test, contract, bundle checker, pyproject.toml, tests/test_field_targets.py, data seal, trainer.py, strategies.py, gsplat_backend.py, field_targets.py, source_snapshot/tracked.diff>
ls / ls -la over runs/, the run root, one cell directory, source_snapshot/, .scratch/<task>/ (names only), the audit directory; wc -c tracked.diff
cat / sed -n over CLAUDE.md, the skill, the task JSON, both reviews, R1, RTGS-027 (rg) and RTGS-028 records, RESULT.md, the driver and report module, task.lock.json, run_receipt.json, environment.json, preflight.json, input_integrity_*.json, input_boundary_receipt.json, resource_receipt.json (head), preparation.json (head), initialization.json, source_snapshot/git-status.txt
rg -n "gsplat" ara/evidence/tables/20260928_color_budget_reset_final_handoff/preflight.json
.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/<task>.json   -> 345e57a9...
.venv/bin/python scripts/experiment_contract.py validate                                      -> OK
.venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/<task>.json   -> OK
.venv/bin/python -m pytest -q tests/test_field_only_portfolio_protocol.py                     -> 7 passed
.venv/bin/python -c / heredoc scripts (read-only): manifest hash verification (158/158); build_source_binding + verify_source_binding on the live tree (119 files, 2848f1d6...); snapshot-manifest-vs-live comparison (121/121); per-cell config/iteration/count/view/interval/guard checks (27 cells); recomputation of per-cell means, group means, paired deltas, verdicts and descriptive pairs under both operators; split digest; nb_30k_d6_lr-vs-nb_base trajectory comparison; last-500-step training-objective means; field-consistency diagnostic means; PIL image size of one preview
Read (image view) of the 28 manifest previews (27 distinct files)
```

## 14. Not executed

`init-run`; the driver's `run`, `prepare`, `initialize`, `fit`, `evaluate` or `selftest`; `check-run`; `scripts/check_results_bundle.py`; any render, viewer smoke, GIF or GPU work; `./scripts/verify.sh`, `ruff`, the full pytest suite; no file was created or modified in the repository (the only artefact of this audit is this report); no raw photograph, mask, `.rtgsv`, `.npz`, `.ply`, `targets/`, `renders/`, log or GIF was opened; no RTGS-025/026/028 RESULT or AUDIT record was read (only the one gsplat-version line of the RTGS-028 preflight receipt); the smoke directories were not entered; the RTGS-027 canary test was not run.

## 15. Remaining handoff checks for the Driver

1. Persist this audit verbatim as `benchmarks/results/<task>_AUDIT.md` and the JSON block as `<task>_AUDIT.json`; do not edit RESULT.md/RESULT.json.
2. Run `check-run` and `python scripts/check_results_bundle.py runs/<task>` after rendering the report page and the headless viewer receipt; the bundle gate is a structural check and does not replace this audit.
3. Write the `docs/EXPERIMENTS.md` entry and the ARA rows carrying: all three passes as screening signals under both operators with the deltas above; the two rejections with the package/hypothesis qualifiers; the section-9 disclosures RESULT.md omits (point table, means-LR factors, nb_ds4 package/counts/no-cap, nb_v11 receipts, gsplat 1.5.3 citation of preflight/environment and RTGS-027/028, second smoke pass, GPU-exclusivity statement, operator-dependent photograph gap, previews at downscale 8). Any `supported` claim must cite `RESULT.json`, `comparison.json` or the cell records on disk; no README/docs number may exceed "development screening signal".
4. Record in the task record that the audit ran as an in-session subagent, `Turn` and `Status` accordingly, and that combining passing arms (SH degree, regularization, downscale 4) is a separately registered confirmation on new seeds, with training-view PSNR (`record_train_metrics`) recorded so the overfitting hypothesis becomes testable.
5. No default change.

```json
{
  "schema_version": 1,
  "task_id": "20260928_field_only_portfolio_stage_frame00008",
  "task_record": "RTGS-029",
  "reviewer": "Claude-Code-Fable-5.1-reviewer",
  "model": "claude-fable-5-1",
  "self_reviewed": false,
  "verdict": "accepted_with_limits",
  "protocol_sha256": "345e57a95cac3edb93fa7cc56d7a0ec43edb9d398f08da4e895583d9b32dfd68",
  "outcome_access": "full post-run results audit",
  "execution_mode": "in-session subagent (Agent tool); detached CLI launch not permitted in this session; read-only tools; no nested agents; no fallback model",
  "authorization": {
    "record": ".scratch/20260928_field_only_portfolio_stage_frame00008/claude_results_audit/authorization.json",
    "authorized_by": "user in chat 2026-09-28 ('Ja bitte.')",
    "payload_manifest_sha256": "9001da43d02d2bdf9dee28f2d7946318893368857ab48770b233d6d034386e1c",
    "manifest_items": 158,
    "previews_viewed": 28,
    "distinct_preview_files": 27,
    "root_preview_equals": "cells/nb_base/9561/reconstruction_contact_sheet.png (frozen preview_policy)",
    "all_item_hashes_match": true,
    "reads_outside_manifest": [
      "rg gsplat line of ara/evidence/tables/20260928_color_budget_reset_final_handoff/preflight.json (RTGS-028 gsplat version 1.5.3)",
      "first 1500 bytes of .scratch/<task>/claude_results_audit/prompt.txt",
      ".scratch/<task>/ directory names only"
    ]
  },
  "chronology": {
    "approval_commit": "c1abcd00876d9f16bc9b1f4eb31d847bb5bbe6e8",
    "approval_commit_time": "2026-09-28T07:02:42Z",
    "lock_started_at_utc": "2026-09-28T08:21:04.932543Z",
    "finished_at_utc": "2026-09-28T09:36:56.943838Z",
    "source_dirty": false,
    "tracked_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "single_invocation": true,
    "execution_failure_json": false,
    "run_roots_for_task": 1,
    "fit_intervals_sequential_in_frozen_order": true,
    "last_fit_end_s": 4449.65,
    "first_evaluate_start_s": 4464.26,
    "max_fit_wall_s": 378,
    "cell_timeout_s": 3600,
    "post_run_tree_changes": [".agents/state/current-task.md (Driver handoff appended)", "benchmarks/results/<task>_RESULT.json (untracked, run-written)", "benchmarks/results/<task>_RESULT.md (untracked, run-written)"]
  },
  "binding": {
    "task_sha256": "d945c7051b988f0516c6bbd7c6b3ff11f9dba82c481e351499d54492b57c6c58",
    "review_artifact_sha256": "e6528c748c4f4f50fef639e86486c90824d5f75deeb62cf27329c385ad4d40c0",
    "data_seal_sha256": "d01254e739961dcb1acc4a9abca6694bb63e319ff179581ebe2ec2ae08706087",
    "source_binding_files": 119,
    "source_binding_aggregate_sha256": "2848f1d6695c69680fd8a724dc8d1408d33ad92d031e5c0ae3fa061e98831b6e",
    "source_binding_reproduced_on_live_tree": true,
    "snapshot_files": 121,
    "snapshot_matches_live_tree": true,
    "src_rtgs_tree": "1cebc4bc92e5e83a904b4fc4394fc0429423bbd2",
    "review_digest_reproduced": true,
    "validate": "OK",
    "validate_data": "OK",
    "protocol_tests": "7 passed",
    "input_integrity_entry": {"files_checked": 81, "all_match": true},
    "input_integrity_exit": {"files_checked": 81, "all_match": true, "split_sha256_reproduced": true},
    "environment": {"python": "3.12.9", "torch": "2.9.0", "gsplat": "1.5.3", "lpips": "0.1.4", "device": "NVIDIA GeForce RTX 3050", "cuda": "12.8"},
    "preflight_gsplat": "1.5.3",
    "rtgs028_preflight_gsplat": "1.5.3"
  },
  "input_boundary": {
    "prepare_opened_views": "exactly the 22 frozen training views; denied []",
    "initialize_opens": [],
    "fit_opens_all_cells": [],
    "evaluate_opened_views": ["C0001", "C0018", "C0029", "C1002"],
    "evaluate_after_all_receipts_completed": true,
    "heldout_closed_before_evaluate": true,
    "alpha_metrics_out_of_sample": true
  },
  "cells": {
    "count": 27,
    "all_completed": true,
    "all_effective_configs_equal_frozen": true,
    "all_initial_sha256_match_seed_initialization": true,
    "all_executed_iterations_equal_frozen": true,
    "density_stop_iter_all": 6000,
    "last_density_event_iteration_all": 5900,
    "cap_100000_reached_any_cell": false,
    "run_wide_max_live_count": 59878,
    "nb_ds4": {"final_counts": {"9561": 38090, "9562": 38713, "9563": 37902}, "max_live_counts_equal_final": true, "pruned_to_budget_total": 0, "training_downscale": 4},
    "nb_v11": {"train_view_ids_equal_frozen_half": true, "sampled_train_views_range": [0, 10], "sampled_length": 8000},
    "nb_30k_d6": {"means_lr_gamma": 0.9998465061085267, "means_lr_final_factor": 0.01, "final_counts": {"9561": 59424, "9562": 59878, "9563": 59391}},
    "nb_30k_d6_lr": {"means_lr_gamma": 0.9994245193792801, "means_lr_final_factor": 3.16227766016838e-08, "final_counts": {"9561": 43315, "9562": 43280, "9563": 42490}, "sampled_views_identical_to_nb_base_first_8000": true, "trajectory_bit_identical_to_nb_base": false},
    "nb_reg": {"opacity_reg": 0.01, "scale_reg": 0.01, "regularization_terms_nonzero_only_here": true, "final_counts": {"9561": 26192, "9562": 26610, "9563": 26661}},
    "sh_degrees": {"nb_sh1": 1, "nb_sh0": 0, "others": 3},
    "record_train_metrics": false,
    "training_view_psnr_available": false
  },
  "official_gates": {
    "recomputed_from_per_view_rows": true,
    "operators": ["primary_box_ds4_to_ds8", "ds8_point"],
    "max_abs_deviation_group_means": 0.0,
    "max_abs_deviation_paired_deltas": 0.0,
    "rule": "pass iff every seed: dPSNR >= +0.1 dB and dLPIPS <= +0.005 and d_outside_alpha <= +0.005; reject iff every seed dPSNR <= -0.1 dB; else inconclusive; nb_ds4 pass/reject requires both operators",
    "arms": {
      "nb_sh1": {"verdict": "pass", "primary_dpsnr": [0.439, 0.330, 0.293], "primary_rule": "pass", "point_dpsnr": [0.398, 0.287, 0.246], "point_rule_descriptive": "pass"},
      "nb_sh0": {"verdict": "pass", "primary_dpsnr": [0.473, 0.395, 0.524], "primary_rule": "pass", "point_dpsnr": [0.329, 0.146, 0.374], "point_rule_descriptive": "pass"},
      "nb_reg": {"verdict": "pass", "primary_dpsnr": [1.075, 0.901, 1.000], "primary_rule": "pass", "point_dpsnr": [0.589, 0.506, 0.585], "point_rule_descriptive": "pass"},
      "nb_30k_d6": {"verdict": "reject", "primary_dpsnr": [-0.826, -1.016, -0.949], "primary_rule": "reject", "point_dpsnr": [-0.582, -0.689, -0.660], "point_rule_descriptive": "reject"},
      "nb_30k_d6_lr": {"verdict": "reject", "primary_dpsnr": [-0.540, -0.616, -0.519], "primary_rule": "reject", "point_dpsnr": [-0.558, -0.598, -0.493], "point_rule_descriptive": "reject"},
      "nb_ds4": {"verdict": "pass", "gated_under_both": true, "primary_dpsnr": [1.972, 1.843, 2.007], "primary_rule": "pass", "point_dpsnr": [1.145, 1.018, 1.160], "point_rule": "pass"}
    },
    "descriptive_pairs": {
      "nb_base_minus_nb_v11": {"primary_dpsnr": [0.856, 0.880, 0.833], "point_dpsnr": [0.945, 0.956, 0.803]},
      "nb_base_minus_ph_base": {"primary_dpsnr": [-0.075, 0.019, -0.239], "point_dpsnr": [-0.334, -0.299, -0.502], "primary_dlpips": [0.0191, 0.0146, 0.0193], "point_dlpips": [0.0202, 0.0147, 0.0180]}
    },
    "decision_string_reproduced": true
  },
  "point_operator_table": {
    "operator": "point-sampled downscale-8 render (RTGS-025/026/028 operator), gsplat 1.5.3 classic mode",
    "source": "RESULT.json.groups_ds8_point == comparison.json.groups_ds8_point == recomputed from per_view[].ds8_point",
    "groups": {
      "nb_base": {"foreground_psnr": 23.439983, "crop_lpips": 0.144824, "outside_alpha_mass": 0.001089, "floater_fraction": 0.000797, "interior_alpha": 0.999034},
      "nb_sh1": {"foreground_psnr": 23.750570, "crop_lpips": 0.138984, "outside_alpha_mass": 0.001316, "floater_fraction": 0.000826, "interior_alpha": 0.999103},
      "nb_sh0": {"foreground_psnr": 23.722980, "crop_lpips": 0.140194, "outside_alpha_mass": 0.001436, "floater_fraction": 0.000819, "interior_alpha": 0.999171},
      "nb_reg": {"foreground_psnr": 24.000339, "crop_lpips": 0.131565, "outside_alpha_mass": 0.001442, "floater_fraction": 0.000891, "interior_alpha": 0.995901},
      "nb_30k_d6": {"foreground_psnr": 22.796301, "crop_lpips": 0.168117, "outside_alpha_mass": 0.000936, "floater_fraction": 0.000787, "interior_alpha": 0.999069},
      "nb_30k_d6_lr": {"foreground_psnr": 22.889935, "crop_lpips": 0.160938, "outside_alpha_mass": 0.000995, "floater_fraction": 0.000821, "interior_alpha": 0.999129},
      "nb_ds4": {"foreground_psnr": 24.547521, "crop_lpips": 0.118451, "outside_alpha_mass": 0.001075, "floater_fraction": 0.000902, "interior_alpha": 0.999430},
      "nb_v11": {"foreground_psnr": 22.538518, "crop_lpips": 0.165887, "outside_alpha_mass": 0.001544, "floater_fraction": 0.001123, "interior_alpha": 0.998785},
      "ph_base": {"foreground_psnr": 23.818399, "crop_lpips": 0.127218, "outside_alpha_mass": 0.001073, "floater_fraction": 0.000821, "interior_alpha": 0.999168}
    },
    "paired_deltas_vs_nb_base_dpsnr": {
      "nb_sh1": [0.398, 0.287, 0.246], "nb_sh0": [0.329, 0.146, 0.374], "nb_reg": [0.589, 0.506, 0.585],
      "nb_30k_d6": [-0.582, -0.689, -0.660], "nb_30k_d6_lr": [-0.558, -0.598, -0.493], "nb_ds4": [1.145, 1.018, 1.160],
      "nb_base_minus_nb_v11": [0.945, 0.956, 0.803], "nb_base_minus_ph_base": [-0.334, -0.299, -0.502]
    },
    "paired_deltas_vs_nb_base_dlpips": {
      "nb_sh1": [-0.0068, -0.0050, -0.0058], "nb_sh0": [-0.0034, -0.0030, -0.0075], "nb_reg": [-0.0144, -0.0122, -0.0132],
      "nb_30k_d6": [0.0239, 0.0252, 0.0207], "nb_30k_d6_lr": [0.0154, 0.0159, 0.0170], "nb_ds4": [-0.0253, -0.0264, -0.0274]
    },
    "metric_semantics": "secondary_metrics ids ds8_point_* map to the unprefixed METRICS keys under per_view[].ds8_point and mean_ds8_point; gates.<arm>.ds8_point_descriptive.verdict is a descriptive rule outcome, not a verdict"
  },
  "claim_dispositions": [
    {"claim": "27 paired development cells completed; one invocation; exit 0; no execution failure", "disposition": "confirm"},
    {"claim": "Primary-operator group means (RESULT.md table)", "disposition": "confirm"},
    {"claim": "nb_sh1 pass", "disposition": "confirm as screening signal; narrow: +0.29..+0.44 dB primary, +0.25..+0.40 dB point; outside alpha +0.0002 every seed"},
    {"claim": "nb_sh0 pass", "disposition": "confirm as screening signal; narrow: +0.40..+0.52 dB primary, +0.15..+0.37 dB point"},
    {"claim": "nb_reg pass", "disposition": "confirm as screening signal; narrow: regularization-plus-capacity package (26k Gaussians, interior alpha 0.994)"},
    {"claim": "nb_30k_d6 reject", "disposition": "confirm; narrow: rejects the 30000-step package with the 30000-step means-LR horizon and a 59k population; does not isolate iteration count"},
    {"claim": "nb_30k_d6_lr reject", "disposition": "confirm; narrow: 22000 further non-positional steps on frozen topology lose 0.5-0.6 dB; overfitting reading is a hypothesis (no training-view PSNR)"},
    {"claim": "nb_ds4 pass gated under both operators", "disposition": "confirm as screening signal; carry resolution-package, ~2.7x wall / ~2x VRAM, no cap hit, final = max count"},
    {"claim": "Pass effects of 0.3-2 dB need confirmation (handoff)", "disposition": "confirm"},
    {"claim": "View-count sensitivity and photograph gap are descriptive", "disposition": "confirm; narrow: photograph gap operator-dependent in PSNR (0.3-0.5 dB point, ~0 primary), 0.015-0.020 in LPIPS under both"},
    {"claim": "Both operators with the dilation mechanism; only nb_ds4 scored with its training render under the primary", "disposition": "confirm"},
    {"claim": "Upstream gsplat 1.5.3 performs no opacity reset in any arm", "disposition": "confirm (preflight/environment 1.5.3; RTGS-027/028 evidence; no reset callback; canary test not run here)"},
    {"claim": "Point set is the RTGS-028 operator", "disposition": "confirm (gsplat 1.5.3 in both preflights; src/rtgs unchanged)"},
    {"claim": "Alpha metrics out-of-sample; held-out closed until evaluate", "disposition": "confirm"},
    {"claim": "Timings descriptive; GPU had no other compute process", "disposition": "confirm descriptive; unresolved for exclusivity (no record in bundle; walls consistent with RTGS-028 free-GPU walls)"},
    {"claim": "Locked from the clean approved tree at c1abcd0; binding check passed", "disposition": "confirm"},
    {"claim": "No default changes", "disposition": "confirm"}
  ],
  "visual_disposition": {
    "previews_viewed": 28,
    "render_operator_of_previews": "downscale-8 held-out cameras (point-operator geometry)",
    "gross_failures": 0,
    "summary": "All 27 cells reconstruct the figure with correct silhouette and colour; residual thin rim error along the mask boundary and low-amplitude interior speckle in every cell (strongest on C0001); sparse background floaters visible only as faint specks in the error panels, densest in nb_v11, nb_sh0, nb_sh1 and nb_reg, sparsest in the 30000-step arms, matching the recorded outside_alpha_mass ordering; nb_ds4 visibly sharpest; no halo beyond the rim band; adequate for a development screen"
  },
  "limitations": [
    "development screen on an outcome-exposed frame (RTGS-021/024/025/026/028), one split, three seeds; six simultaneous uncorrected comparisons; every pass is a screening signal requiring a separately registered confirmation",
    "RESULT.md omits the condition-4 disclosures listed in section 9 (point table, means-LR factors, nb_ds4 package/counts, nb_v11 receipts, preflight/RTGS-027/028 citation, second smoke pass, GPU-exclusivity statement, operator-dependent photograph gap); this audit and the EXPERIMENTS/ARA entries must carry them",
    "nb_30k_d6 does not isolate iteration count (means-LR horizon changed the population grown in the same window); nb_reg is a regularization-plus-capacity package",
    "no training-view PSNR recorded; overfitting interpretations of the 30000-step and SH-degree effects are hypotheses; the falling training objective is consistent with but does not establish them",
    "GPU exclusivity and the second two-operator smoke evaluation are Driver statements without records in the bundle; timings descriptive",
    "previews are point-operator renders at downscale 8, not the primary box render",
    "nb_30k_d6_lr equals nb_base through step 8000 in configuration, seed, targets and schedule only; realized trajectories differ by CUDA nondeterminism (about 2 percent population difference at 6000)",
    "the smallest pass margin under the descriptive point operator (nb_sh0 seed 9562, +0.146 dB) is close to the 0.1 dB floor; primary margins are all >= +0.29 dB",
    "no default, SOTA, generalization, physical-geometry, speed or VRAM claim"
  ],
  "commands_executed_by_reviewer": [
    "git status --short; git status --porcelain --untracked-files=all; git rev-parse HEAD; git log --oneline -8; git log --format='%H %ad %s' --date=iso -4",
    "git diff .agents/state/current-task.md; git diff --stat 9abf11c c1abcd0; git diff 9abf11c c1abcd0 -- experiments/tasks/<task>.json",
    "git rev-parse HEAD:src/rtgs HEAD:<driver> HEAD:<report> HEAD:<protocol test> HEAD:<task JSON>",
    "sha256sum over the payload manifest, bound driver/report/test, contract, bundle checker, pyproject.toml, tests/test_field_targets.py, data seal, trainer.py, strategies.py, gsplat_backend.py, field_targets.py, source_snapshot/tracked.diff",
    "ls / ls -la / wc -c over runs/, the run root, one cell, source_snapshot/, .scratch/<task>/ names, the audit directory",
    "cat / sed -n / rg over CLAUDE.md, the audit skill, the task JSON, both reviews, R1, RTGS-027/028 records, RESULT.md, driver, report module, lock, receipts, environment, preflight, integrity, boundary, resource, preparation, initialization, snapshot manifest, git-status.txt",
    "rg -n gsplat ara/evidence/tables/20260928_color_budget_reset_final_handoff/preflight.json",
    ".venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/<task>.json",
    ".venv/bin/python scripts/experiment_contract.py validate",
    ".venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/<task>.json",
    ".venv/bin/python -m pytest -q tests/test_field_only_portfolio_protocol.py",
    ".venv/bin/python -c / heredoc read-only recomputation scripts over the manifest JSON files (manifest hashes, source binding, snapshot comparison, per-cell checks, both-operator means/deltas/verdicts, split digest, trajectory comparison, training-objective means, diagnostic means, one PIL image size)",
    "Read (image view) of the 28 manifest previews"
  ],
  "not_executed": [
    "init-run; driver run/prepare/initialize/fit/evaluate/selftest; check-run; check_results_bundle.py",
    "any render, viewer smoke, GIF or GPU work; verify.sh; ruff; full pytest suite; RTGS-027 canary test",
    "no file created or modified in the repository",
    "no raw photograph, mask, .rtgsv, .npz, .ply, targets/, renders/, log or GIF opened; smoke directories not entered",
    "no RTGS-025/026/028 RESULT or AUDIT record read (only the gsplat line of the RTGS-028 preflight receipt)"
  ],
  "remaining_handoff_checks": [
    "persist AUDIT.md and AUDIT.json verbatim; do not edit RESULT.md/RESULT.json",
    "render the report page, headless viewer receipt, second render; run check-run and check_results_bundle.py",
    "EXPERIMENTS.md entry and ARA rows carrying all section-9 disclosures and the both-operator deltas; supported claims cite RESULT.json / comparison.json / cell records; nothing above development screening signal",
    "task record: audit ran as in-session subagent; Turn/Status; confirmation of combined passing arms on new seeds with record_train_metrics so overfitting becomes testable",
    "no default change"
  ]
}
```