# Prospective Protocol Review

- Task ID: `20260927_color_budget_reset_stage_frame00008`
- Protocol SHA-256: `7a55197b8d83df6c241b41fd9037acc4e6cd4f93bc101e26e463231e68b096b9`
- Reviewer: `Claude-Code-Fable-5.1-reviewer`
- Verdict: `approved`
- Outcome Access: `none`

## Scope

Prospective review of the RTGS-028 draft task at commit `a496016` on branch
`rtgs-028-color-budget-reset` (worktree clean, including untracked files). The Driver derived the
driver, report module and protocol tests from the RTGS-026 retry driver approved by this reviewer
at `492996f` (`experiments/reviews/20260927_silhouette_relocation_index_decode_stage_frame00008_PROTOCOL_REVIEW.md`,
digest `27f7b23a...`), removing the hull, relocation and production paths and adding two training
budgets plus the opt-in `IntendedOpacityReset` accepted in RTGS-027
(`docs/tasks/RTGS-027-gsplat-opacity-reset.md`). This review treated the Driver's handoff in
`.agents/state/current-task.md` as claims to check: it diffed the task JSON, driver and report
module against the approved RTGS-026 retry files, recomputed the protocol digest and the
source-binding aggregate, ran the bound CPU tests, traced the reset wiring through the bound
trainer seam and the accepted RTGS-027 mechanism, compared the frozen configurations with the
RTGS-025 resolved configuration, and probed the frozen schedules read-only. No RESULT or AUDIT
record of any earlier task was read.

Question: for field-only distillation from decoded `no_boundary` compact fields with the RTGS-025
masked objective on frame 00008 (22 training views; held-out C0001, C0018, C0029, C1002;
downscale 8), does a 30000-step budget with densification to step 15000 improve held-out colour
inside the four held-out masks over the 8000-step, densification-to-6000 configuration of
RTGS-025/026 (H1: every paired seed at least +0.2 dB foreground PSNR with crop LPIPS no worse than
+0.005), and at 30000 steps does the RTGS-027 intended opacity reset (gsplat iterations 3000, 6000,
9000, 12000) improve on the inert upstream reset (H2: every paired seed at least +0.1 dB with crop
LPIPS no worse than +0.005 and outside-mask alpha mass no worse than +0.005)? The reset at 8000
steps, the field-versus-photograph gaps at 8000 steps and at 30000 steps with the reset, and the
photograph budget pair are descriptive paired deltas without verdicts.

Evidence maturity this protocol may establish: a development screen on one previously
outcome-exposed frame, one split, three fresh paired seeds (9461-9463), two budgets, 18 cells.
Held-out photographs, masks and fields never enter preparation, initialization or fitting, so the
colour metrics are novel-view measurements and the alpha, floater and coverage descriptors are
out-of-sample. H1 measures the 30000-step convention as a package (iterations, densification
window and, derived from `iterations`, the means learning-rate decay horizon), not iteration count
alone. H2 isolates the reset callback exactly. No default, SOTA, generalization, physical-geometry,
speed or VRAM claim can follow; a pass only informs the configuration of the next registered
field-only task. Approval says only that the frozen design is fit to execute.

## Checks

Read in full: the task JSON; the driver; the report module; the protocol test file;
`tests/test_gsplat_opacity_reset.py`; `src/rtgs/optim/strategies.py`; the review template; the
RTGS-026 retry review; `.agents/state/current-task.md`; `docs/tasks/RTGS-027-gsplat-opacity-reset.md`;
the `diff -u` outputs of driver, report and task JSON against the RTGS-026 retry files. Read in
part: `src/rtgs/optim/trainer.py` (`train` signature and docstring 293-332, learning rates and
`means_gamma` 480-497, loop head 656-713, objective 722-752, optimizer step through parameter
callback and checkpoint 832-1033, `_resolve_schedule_iterations` 1575-1602);
`scripts/experiment_contract.py` (`protocol_sha256` 246-259, `_review_artifact_errors` 262-316,
`REVIEW_FIELD_RE` and `REVIEW_SECTIONS` 103-113, `build_source_binding` and
`verify_source_binding` 1022-1101, `init_run` 1161-1216); `src/rtgs/data/calibrated.py`
`load_calibrated_scene` 154-262; `src/rtgs/data/compact_views.py` `CompactView.load` (grep);
`src/rtgs/optim/density.py` `DensityConfig` defaults (grep); `experiments/README.md` 36-85;
`experiments/reviews/README.md`; the RTGS-025 task JSON split, seeds and resolved configurations;
the first 700 bytes of the data seal (schema, data slug, input profile, dataset id/role/path, view
ids).

Commands executed (read-only, or the allowed CPU tests):

```text
git status --short; git rev-parse HEAD; git log --oneline -8; git show --stat --oneline HEAD
git log --oneline 492996f..HEAD; git diff --name-status 492996f HEAD
git diff --stat 492996f HEAD -- src/rtgs scripts/experiment_contract.py scripts/check_results_bundle.py pyproject.toml tests/test_field_targets.py
git status --porcelain --untracked-files=all
git status --ignored --short -- <bound patterns, experiments/tasks, experiments/data, experiments/reviews, tests>
git rev-parse HEAD:<each path in the object-id table>; git rev-parse 492996f:src/rtgs 492996f:src/rtgs/optim/strategies.py
diff -u <RTGS-026 retry driver> <new driver>; diff -u <RTGS-026 retry report> <new report>
diff -u <RTGS-026 retry task JSON> <new task JSON>
sha256sum <task JSON, data seals, driver, report, tests, bound src files, contract, bundle checker, pyproject, predecessor files, reviews, task records>
head -c 700 experiments/data/20260927_color_budget_reset_stage_frame00008.json
ls experiments/reviews/ runs/ benchmarks/results/ (filtered for 20260927); ls -d .scratch/* (names only)
.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260927_color_budget_reset_stage_frame00008.json
.venv/bin/python scripts/experiment_contract.py validate
.venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/20260927_color_budget_reset_stage_frame00008.json
.venv/bin/python -m pytest -q tests/test_color_budget_reset_protocol.py tests/test_gsplat_opacity_reset.py
.venv/bin/python -c <build_source_binding, verify_source_binding, validate_task(check_live_source=True) on the task>
.venv/bin/python -c <TrainConfig round-trip of both budgets; IntendedOpacityReset.from_density fired set; means_gamma per budget>
.venv/bin/python -c <RTGS-025 masked_silhouette/9261 versus 8k/9461; 8k versus 30k; per-seed differences; cell and file counts>
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```

Denied by the review sandbox and re-done in an allowed form, none touching protected material: a
pytest invocation piped through `tail`, a heredoc Python script, and a pytest invocation prefixed
with `CUDA_VISIBLE_DEVICES=""` and `-rs`; each was re-run as the plain single command above. CUDA
is not visible in this session, so the one CUDA-marked test skipped.

Results: `review-digest` prints `7a55197b8d83df6c241b41fd9037acc4e6cd4f93bc101e26e463231e68b096b9`,
the digest the Driver reported (the digest excludes only `status` and `protocol_review`, contract
249-251). `validate` prints OK; `validate-data` prints OK. `build_source_binding` over the task's
patterns returns 120 files and aggregate
`28b7fb80b74d1b520e9ec176e0e9161d6c81ec699868340c43f6736158486af5`, equal to the frozen values;
`verify_source_binding` and `validate_task(check_live_source=True)` return no errors, so, unlike
the RTGS-026 rounds, the aggregate was reproduced here. Glob confirms 108 `.py` plus 4
`.cu/.cpp/.h` files under `src/rtgs` and eight named files. Pytest: 15 collected, 14 passed, 1
skipped (the CUDA test); the protocol file's six tests match the Driver's count. Worktree clean at
`a496016` including untracked files; nothing ignored under the bound patterns except
`__pycache__`. `runs/` holds only the two RTGS-026 roots; no
`runs/20260927_color_budget_reset_stage_frame00008/`, no
`benchmarks/results/20260927_color_budget_reset_*` file and no RTGS-028 review artifact exist.
`.scratch/20260927_color_budget_reset_stage_frame00008/` exists (the disclosed smoke's scratch
space per `experiments/README.md` line 81) and was not entered.

- Drift, driver versus the approved RTGS-026 retry driver. The diff touches exactly: the module
  docstring; `HULL` removed and `CONDITIONS` now a six-entry table of
  `(family, budget, intended_reset)`; `hull_rejected_fraction` removed from `METRICS`;
  `check_protocol_tables` drops the hull-view rule and adds the rule that
  `resolved_training_configs` keys equal the frozen budgets; `access_guard` drops the hull cache
  from the fit allow-list, denies any held-out source open (photograph, mask PNG or compact view)
  in every phase but `evaluate` where the predecessor denied only held-out images in
  `prepare`/`initialize`, and drops the production branch; the parity seed base `926200` becomes
  `926300`; `prepare` drops the hull loop, the hull mask cache and the hull metadata; `load_hull`
  and `relocator_for` are removed; `cached_scene` loses its `view_ids` override; `initialize`
  drops the hull voxels and rejected fractions; `train_config` gains the budget key; `_train` is
  inlined into `fit`, which builds `IntendedOpacityReset.from_density(config.density)` when the
  condition requests it, raises if that returns `None`, records `opacity_reset_events` instead of
  `relocation_events`, and adds `iterations` to the receipt; `evaluate` loads held-out compact
  views with `load_alpha=False` and drops the held-out packed-alpha equality check and the hull;
  `production` is removed together with its CLI choice; `selftest` and `report_module` change
  only their fake ids and module name. Byte-identical: `mask_scores`, `dilate`/`erode`, the
  quadrature and image-query helpers, `random_initialization`, `model_from`, `load_source`,
  `load_view`, `parity_record`, `warmup`, `snapshot_source`, `write_environment`, `preflight`,
  `data_guard`, `source_guard`, the coordinator's re-entry refusal (786-788) and fit-only 3600 s
  timeout (812), and the canonical-root check in `main` (879-880). A case-insensitive grep for
  `cuda` finds only device placements; `make_query_backend` is called once with `"index"` and
  `decode_compact_view` twice with `backend="index"`.
- Drift, report module. Docstring; `CONDITIONS` and `SELECTED = ("nb_30k_rs", 9461)`;
  `ITERATIONS` replaced by `BUDGET_ITERATIONS` and `condition_iterations` (29-30), used for the
  `executed_iterations` check in `publish` (256) and the per-cell stage markers in `history`
  (181-195); the single relocation rule generalized into
  `_rule(treatment, control, gain, lpips_margin, alpha_margin)` (68-93); `gates` now returns
  `h1_budget`, `h2_reset`, four descriptive pairs and the audit placeholder (96-109);
  `decision_text`, the summary string, the mask-role sentence and two notes reworded. Everything
  else (once-only RESULT records, run receipt, config/comparison/history/resource/input-boundary
  writers, artifact and chart lists, failure publisher) is byte-identical.
- Drift, task JSON. Beyond identifiers, paths, the review-state reset, title, question,
  hypothesis and claim boundary: the split returns C0018 and C1002 to held-out (22 + 4, identical
  to the RTGS-025 split); seeds 9461-9463; `heldout_mask_alpha` leaves the allowed inputs and
  `heldout_masks` joins the forbidden ones; the hull guard leaves `execution_guards`; stages,
  comparators and metric texts drop every hull, relocation, production and in-sample sentence;
  the `hull`, `relocation` and `production` blocks are removed; an `opacity_reset` block and
  `training.budgets` are added; `resolved_training_configs` is nested by budget; the H1 rule
  becomes `h1_budget` at +0.2 dB with the `h2_reset` rule added; the execution order lists 18
  cells, each condition/seed exactly once (checked by `check_protocol_tables`); the preview model
  is `nb_30k_rs` seed 9461; and the source binding swaps the relocation files for the two new
  tests. Datasets, the 28 sealed field files, the preprocessing operator and mask rules,
  initialization (except the six-cell wording), resource protocol, execution controls, source
  lifecycle, live-integrity policy and `report_template_version: 2` are unchanged.
- Configurations. Every 8k config differs from the RTGS-025 `masked_silhouette` resolved config
  only in `seed` (density block identical), so `nb_8k_up` is the RTGS-025/026 main-path
  configuration as its purpose states. Each 30k config differs from the same-seed 8k config only
  in `iterations` (30000) and `density.stop_iter` (15000); the three seeds of a budget differ only
  in `seed`. `train_config` reconstructs `TrainConfig` and re-serializes byte-equal for both
  budgets and all seeds, and the protocol test pins these facts. `sh_degree_interval` is explicit
  (1000), so SH reaches degree 3 at the same step in both budgets.
- Reset wiring. `fit` passes the reset as `parameter_step_callback` (623-625); the trainer calls
  it under `no_grad` after the optimizer step and after gsplat density control with
  `completed_step = global_it + 1` (trainer 657-658, 894-909) and rejects any count change.
  `IntendedOpacityReset.due` evaluates `g = step - 1` against `0 < g < stop_iter` and
  `g % 3000 == 0` (strategies 401-404), clamps to `logit(2 * prune_opacity) = logit(0.01)` and
  zeroes the opacity Adam moments (406-427); this is the mechanism accepted in RTGS-027 round 4,
  and `strategies.py` is bound. Probing the frozen configs: the 8k reset fires only at
  `g = 3000` (`g = 6000` is excluded by `stop_iter`); the 30k reset fires at
  `g = 3000, 6000, 9000, 12000` (`15000` excluded); `due(iterations)` is false for both budgets,
  and the final iterations `g = 7999` and `g = 29999` are not period multiples, so no cell ends on
  a reset. `from_density` cannot return `None` at period 3000, and the driver raises if it did.
  Storage is `dynamic` and the strategy `gsplat-default`, so neither the arena reset nor the
  classic controller can double-reset. `opacity_reset_every = 3000` is identical in `_up` and
  `_rs` cells, so gsplat's refine pause after each would-be reset and its large-scale pruning
  onset gate are shared; the callback is the only difference between the two reset arms. The
  `opacity_reset_value: 0.011` field is a classic-controller value unused on this path; the
  intended reset uses 0.01, as `opacity_reset.intended` states.
- Gates versus written policy. `_rule` implements pass as
  `a.psnr >= b.psnr + gain and a.lpips <= b.lpips + 0.005` (plus
  `a.outside_alpha_mass <= b.outside_alpha_mass + 0.005` for H2), reverse as
  `a.psnr <= b.psnr - gain`, verdict pass iff every seed passes, reject iff every seed reverses,
  else inconclusive; H1 binds `nb_30k_up` versus `nb_8k_up` at 0.2 and H2 `nb_30k_rs` versus
  `nb_30k_up` at 0.1. This is the `decision_policy` text in written inclusive form, and the test
  exercises a pass at exact margins, an alpha-margin inconclusive and a reject. The question's
  phrase "without more floaters" is operationalized as outside-mask alpha mass;
  `floater_fraction` is reported but not gated.
- Leakage boundaries. `prepare` opens only the 22 training photographs, mask PNGs and compact
  views: `load_calibrated_scene(view_ids=[view])` lists the RGB directory without opening it and
  opens only the requested image and its mask (calibrated 191-252); the calibration JSON lies
  outside the frame path; `CompactView.load` opens only the named archive. Object bounds and soft
  masks come from training views alone; no held-out camera or alpha is used anywhere before
  `evaluate`. `initialize` reads only `targets/metadata.json`. A fitting worker may open only
  `targets/<its family>/`, `targets/training_masks/` and `metadata.json`; any frame open is
  denied (driver 420-430). `evaluate` is the first phase to open held-out photographs, masks and
  compact views and runs only after all 18 receipts read `completed` (664-667). The coordinator
  hashes sealed bytes, including held-out archives, opaquely at entry and exit, as approved
  before. The `selftest` still denies four probes and the protocol test asserts the count. A
  stricter guard than the predecessor's cannot abort `prepare` spuriously because nothing in
  `prepare` touches a held-out file.
- Confounds. (1) The budget couples iterations, the densification window and the means
  learning-rate horizon: `means_gamma = 0.01 ** (1 / iterations)` (trainer 488, applied at 878),
  so the 8k arm decays to 1% of the initial means LR by step 8000, whereas the 30k arm is at
  29.3% at step 8000, 10% when densification stops at 15000 and 1% at 30000. The claim boundary
  names the densification window but not the LR horizon, and the report note "scale only
  iterations and the densification stop" is true of the configuration and silent about the
  derived schedule. This is the 3DGS convention the task declares, and acceptable for a
  development screen if disclosed (condition 5). (2) `max_gaussians = 100000` is enforced on
  every post-backward step by `enforce_budget`; with densification to 15000 the 30k arms may
  reach the cap where the 8k arms did not, in which case least-significant-row pruning becomes an
  active mechanism that differs between arms. `history["density_stats"]` (`n_before`, `n_after`,
  `pruned_to_budget` per event) is saved per cell, so the audit can see it (condition 5). (3) The
  photograph pair `ph_30k_rs` versus `ph_8k_up` couples budget and reset; it is descriptive only.
  (4) The upstream premise of H2 (gsplat 1.5.3 never resets) is verified by the bound canary test
  but not asserted at run time; `preflight.json` records the version (condition 4).
- Stopping and timeout. The 3600 s limit applies to `fit` workers only; `prepare`, `initialize`
  and `evaluate` are untimed. A timed-out cell receives a `timed_out` receipt from the
  coordinator, stops the matrix and consumes the root. The Driver estimates about 10 minutes per
  30000-step cell on a possibly shared GPU; this review could not verify that, and the disclosed
  60/120-iteration smoke cannot bound it (condition 3). Non-finite renders, models and losses
  abort fail-closed as before.
- Seeds. 9461-9463 are fresh (RTGS-025 used 9261-9263, RTGS-026 9361-9363); the parity seeds
  `926300 + index` are gate constants. No outcome exists that could have informed them.
- Smoke disclosure. `claim_boundary` discloses one pre-review non-protocol smoke (60/120
  iterations, reset period 20, two training views standing in for held-out, no real held-out
  access) that exposed training-view teacher fidelity, reset events and a meaningless tiny-budget
  gate value, with no threshold change afterwards; the handoff agrees. Its mechanism is not
  verifiable from tracked state (`main` allows only the canonical root, which does not exist), and
  the scratch directory was not entered. Teacher fidelity is a preparation quantity, not a
  fitting outcome.
- Report handling of two budgets. `condition_iterations` maps `nb_8k_*` and `ph_8k_*` to 8000
  and `*_30k_*` to 30000 (test-pinned); `publish` rejects a cell whose `executed_iterations`
  differs, and `fit` already rejects a run that did not reach its frozen final iteration; stage
  markers use the per-cell total. `history` joins `elapsed`, `n_gaussians` and the checkpoint
  observer on the same `eval_every` steps (trainer 911-981), which divide 8000 and 30000 alike.
- Source binding completeness. Bound: all of `src/rtgs` (including the RTGS-027 `strategies.py`,
  whose object changed from `5b797aa7` to `8c4e99a2`, the only `src/rtgs` change since
  `492996f`), the contract, the bundle checker, the driver, the report, the two new tests,
  `tests/test_field_targets.py` and `pyproject.toml`. Not bound and recorded only by
  `environment.json` and `preflight.json`: the installed torch, gsplat and lpips packages and the
  LPIPS AlexNet weights, as in RTGS-025/026.
- Task record. Driver `Claude-Code-Opus-5.5-driver`, Reviewer label
  `Claude-Code-Fable-5.1-reviewer`, Turn `reviewer`, Status `In review`, contract path correct,
  handoff with a digest and aggregate that both reproduce. The handoff's `verify.sh` and
  GPU-smoke statements were not verified and were not used as evidence.

Reviewed identities at `a496016` (`sha256sum`; `unchanged` means identical to the state approved
at `492996f`):

| File | SHA-256 |
|---|---|
| `experiments/tasks/20260927_color_budget_reset_stage_frame00008.json` | `53768705a423502876cfaf857d46458d7375ce0fde354bbc8c08098489513a96` |
| `experiments/data/20260927_color_budget_reset_stage_frame00008.json` | `d01254e739961dcb1acc4a9abca6694bb63e319ff179581ebe2ec2ae08706087` (identical to the RTGS-025/026 seal) |
| `scripts/experiments/20260927_color_budget_reset_stage_frame00008.py` | `9de0cc5ca8815a00282053dce7d1fbebf8b3c56d863060bcd846ddbfd3f18ba0` |
| `scripts/experiments/20260927_color_budget_reset_stage_frame00008_report.py` | `d85dcef08ee15592866fe7c11e995bbeafafc387d3871e299fd0ee1fdfcaa36e` |
| `tests/test_color_budget_reset_protocol.py` | `c8210a245eb7a27517727cccc7efba9e04c3fc8c244a864f1f0f7b5c5733c8fb` |
| `tests/test_gsplat_opacity_reset.py` | `ebc463ef295d18c108837cdeb83b039e513e6fb744678961bbb09c96212464bc` |
| `tests/test_field_targets.py` | `0a817166c45df13d64cd8e8f83aafbf7656e1bc37ef8bf26176bf6def75a72cc` (unchanged) |
| `src/rtgs/optim/strategies.py` | `947a6e02c229c17ee639be98aede565292ccc3aa0303c84f5130b897c2bea91d` (RTGS-027 additions) |
| `src/rtgs/optim/trainer.py` | `e226d01528897fd00dea8c03f345190c9abd3b4cf5cd5e2013967ca5802d39ba` (unchanged) |
| `src/rtgs/optim/density.py` | `ad889cd67f7d6e54c7223836dfc27d10555a18c98110f5f465f0df71d352254a` (unchanged) |
| `src/rtgs/data/field_targets.py` | `59901d820ef291381c27cb45ba68161cdad1195eff7ea891ae281b60bca9829f` (unchanged) |
| `scripts/experiment_contract.py` | `15f129398aabb46a50d435c76fec27b4df6c75e6d08d2e7d78a5a96657b25959` (unchanged) |
| `scripts/check_results_bundle.py` | `0f5e1293a4c6651d1fed80237bd1319f44ed2ca268d81cb27ce87d2e54f2fedf` (unchanged) |
| `pyproject.toml` | `e5b15e72400e8e14b83a26dc64ef8a85ea985642ef9ef079e425326b5e0bc8d9` (unchanged) |
| `scripts/experiments/20260927_silhouette_relocation_index_decode_stage_frame00008.py` | `d2f68c21d3ac1df9c1a9d2e6fc49e6071c173c34295664d692e682c790e9866c` (diff base, unchanged) |
| `scripts/experiments/20260927_silhouette_relocation_index_decode_stage_frame00008_report.py` | `7c234f01120e92afaa2f8030cefb701ef4f48373984c4405575c486fc3fc7d73` (diff base, unchanged) |
| `tests/test_silhouette_relocation_index_decode_protocol.py` | `57202cdb9c760ff3283f8b7aa7a6afe637a0d15b25461858f7835a2a58292f3e` (unchanged) |
| `experiments/tasks/20260927_silhouette_relocation_index_decode_stage_frame00008.json` | `fdcfca2bad4ebce826a2d822b8ef68c94dd44f1581e1856cb765dd28ed9a867d` (diff base; changed since `492996f` only by its recorded approval) |
| `experiments/reviews/20260927_silhouette_relocation_index_decode_stage_frame00008_PROTOCOL_REVIEW.md` | `a58ce3e924fe2c8d1305dcd2437f108c9848ac32a76c194a97a96c36bb62f400` |
| `experiments/tasks/20260926_field_only_distillation_stage_frame00008.json` | `27afde3837e3da01ef9e01cd07348afb7694842b12edea267bb2316a2bef1335` (split and configuration reference) |
| `docs/tasks/RTGS-027-gsplat-opacity-reset.md` | `943b5d410a43324461654f7812702efca9079441aa160aa719ba2310f77e3154` |
| `.agents/state/current-task.md` | `0617d3d2b9825a78893507683ebd4d1c75172bd1e269efecea9aadddacd697aa` |

Git object ids at `a496016` (`git rev-parse HEAD:<path>`):

| Path | Object id |
|---|---|
| `src/rtgs` (tree) | `1cebc4bc92e5e83a904b4fc4394fc0429423bbd2` (was `9293a19a...` at `492996f`; only `optim/strategies.py` differs, +85 lines) |
| `src/rtgs/optim/strategies.py` | `8c4e99a2b462811bc9fca5a529123baaee1516fa` (was `5b797aa7...`) |
| `src/rtgs/optim/trainer.py` | `81f88540a171907c49bd37cbff8f5b463c231e0d` (unchanged) |
| `src/rtgs/optim/density.py` | `54c4d0df8d08153585727973e8d40f3aacb5c5b7` |
| `src/rtgs/data/field_targets.py` | `80caeefc9c4f3e1ebcc4f03e39688004feeb46ee` (unchanged) |
| `scripts/experiment_contract.py` | `3d4f251972226e5372dfb5346283c65f2e73e62e` (unchanged) |
| `scripts/check_results_bundle.py` | `4be3439031998a35caf1b6d023085dccee659ceb` (unchanged) |
| `pyproject.toml` | `f3911412950125f290bc65a49818ed6949c820a7` (unchanged) |
| `tests/test_field_targets.py` | `1f522a73de24a3ce91c293ee52459f9592f439cd` (unchanged) |
| `tests/test_gsplat_opacity_reset.py` | `f21b50343558a8df890e8d5bd5c1fb0663ad8a65` |
| `tests/test_color_budget_reset_protocol.py` | `154add33385373d85d06e2631c4cfdd98014ab68` |
| `scripts/experiments/20260927_color_budget_reset_stage_frame00008.py` | `fc265e6c3f473c4f42b5b569eb99d2b65519c9e7` |
| `scripts/experiments/20260927_color_budget_reset_stage_frame00008_report.py` | `24177606dadf0cc7a5872f79a9a5317ab20e6f68` |
| `experiments/tasks/20260927_color_budget_reset_stage_frame00008.json` | `72bdeb4a70ccef139e3a72b29171f4604850c31f` |
| `experiments/data/20260927_color_budget_reset_stage_frame00008.json` | `dc22987ddbfefac83d10deae666616451b8cff8c` (same object as the RTGS-025/026 seal) |
| `experiments/reviews/20260927_silhouette_relocation_index_decode_stage_frame00008_PROTOCOL_REVIEW.md` | `b03869c5bc8458310799389a2025e621d241f5db` |
| `docs/tasks/RTGS-027-gsplat-opacity-reset.md` | `fbeea9c95adeeae0871e0ff068f9dae655e02a56` |
| `.agents/state/current-task.md` | `d030222c9025e92339d3d2be76c4d02a81107f2b` |

## Findings

Approved. The Driver's derivation claim is verified at the byte level: apart from identifiers,
paths, the review-state reset and wording, the driver and report differ from the approved
RTGS-026 retry only by removing the hull, relocation and production paths, adding the budget
dimension and wiring the accepted RTGS-027 reset through the existing `parameter_step_callback`
seam. The decoder, parity gate, target operator, masked objective, initialization, seal
cross-checks, coordinator lifecycle, resource protocol and once-only evidence writers are the ones
approved at `492996f`, and `trainer.py`, `field_targets.py`, the contract and the bundle checker
are unchanged since then. The frozen 8k configuration equals the RTGS-025 main-path configuration
up to the seed, the 30k configuration differs only in `iterations` and `density.stop_iter`, and
both round-trip. The reset fires on exactly gsplat's intended set for these budgets (3000 at 8k;
3000, 6000, 9000, 12000 at 30k) and never on a final iteration, and the only difference between
the `_up` and `_rs` arms is the callback. Held-out photographs, masks and fields are closed to
every phase before `evaluate`, so the alpha descriptors are genuinely out-of-sample. The gates
implement the written policy inclusively per seed, and no threshold was tuned to the disclosed
smoke. The design is the minimal factorial for the three questions the record says the user asked
for; dropping `nb_8k_rs` or the photograph arms would be simpler but would drop the reset-at-8k or
gap questions. Two disclosure gaps, the derived means-LR horizon and the possible 100000 cap
interaction at 30000 steps, are inherent to the declared convention and are carried as conditions
rather than as reasons to reject, because a one-sentence protocol edit would require a new digest
and round without changing what runs. Approval says the frozen design is fit to execute; it says
nothing about the result.

Residual execution conditions:

1. Recording. Set `protocol_review` to this reviewer label, verdict `approved`, digest
   `7a55197b8d83df6c241b41fd9037acc4e6cd4f93bc101e26e463231e68b096b9` and artifact
   `experiments/reviews/20260927_color_budget_reset_stage_frame00008_PROTOCOL_REVIEW.md`, set
   `status: ready`, change nothing else, and confirm `review-digest` still prints the same digest
   and `validate` prints OK. Persist this file verbatim; `init-run` hashes it into the lock and
   `source_guard` and `snapshot_source` bind it thereafter. Append the verdict to the task record
   and pass the Turn to the Driver.
2. Source state at `init-run`. Run `init-run` from a clean tree whose bound source matches the
   object ids above (`src/rtgs` tree `1cebc4bc...` and the eight named files); `init-run`
   re-verifies the 120-file aggregate `28b7fb80...` fail-closed. Any change to a bound file before
   `init-run` invalidates this approval and requires a new digest and review round; in particular
   do not touch `strategies.py`, `trainer.py`, `field_targets.py`, the driver, the report or the
   two bound tests before the run.
3. Runtime versus the frozen 3600 s cell limit. Before `init-run`, check from the RTGS-025/026
   cell receipts available to the Driver and from the current GPU load that a 30000-step cell,
   growing toward the 100000 cap, fits inside 3600 s with margin. If the observed 8000-step wall
   time exceeded roughly 700 s, or the GPU is shared with another heavy process, do not start:
   re-register with a larger limit (new digest and review). A time-out after `prepare` began
   consumes the root and requires a new task id; do not kill other GPU processes to make room.
4. Upstream premise of H2. Before `init-run`, run the bound `tests/test_gsplat_opacity_reset.py`
   in the GPU venv so the canary test asserts the installed gsplat still carries the inert
   expression; the RESULT and AUDIT must cite the gsplat version from `preflight.json` (1.5.3
   expected) as the basis of the "upstream (inert) reset" control.
5. Disclosures the RESULT and AUDIT must carry: frame 00008 is outcome-exposed; H1 measures the
   30000-step convention as a package, including the means learning-rate horizon derived from
   `iterations` (state the factors 29.3% at step 8000, 10% at 15000, 1% at 30000 for the 30k arm
   versus 1% at 8000 for the 8k arm, or the two `means_gamma` values); whether any cell reached
   the 100000 cap (`density_stats.pruned_to_budget` in each `history.json`) and the recorded
   reset events per `_rs` cell; the reset value is 0.01 (`2 * prune_opacity`), not the inert
   `opacity_reset_value` 0.011; H2's floater clause is gated by outside-mask alpha mass while
   `floater_fraction` is descriptive; the photograph budget pair couples budget and reset; the
   pre-review smoke as stated in `claim_boundary`; timings on a possibly shared GPU are
   descriptive; and the RTGS-025/026 numbers are not comparators in any gate.
6. One `run` invocation. The coordinator refuses a root containing `targets/` or
   `preparation.json`. If the coordinator fails before `prepare` begins (snapshot, `source_guard`,
   data seal or `preflight`), preserve `execution_failure.json` and the logs in the task record
   before any re-entry and disclose the re-entry in the task record and the RESULT. A failure
   after `prepare` began consumes the root; no sibling root may be created, and a repeat requires
   a new task id and review.
7. Accepted late aborts that consume the root: the index-parity, packed-alpha equality, sealed
   digest and fit-window checks in `prepare`; a CUDA OOM in a 30000-step cell; and the held-out
   camera-record checks, which can only run in `evaluate` after all 18 cells.
8. No production or all-view phase exists in this task; nothing beyond the 18 registered cells
   may be trained under this root, and the `nb_30k_rs` seed 9461 preview is the only root-level
   model.

Optional, not conditions, for a later task: a 30000-step arm with densification stop 6000 to
separate iteration count from the densification window; a `ph_30k_up` arm to make the photograph
budget pair clean; reading each cell's budget from `effective_config.json` instead of parsing the
condition id in `condition_iterations`; a run-time assertion of gsplat's inert reset expression in
`preflight`.

## Protected Actions Not Taken

The reviewer did not run `init-run`, did not execute the driver's `run`, `prepare`, `initialize`,
`fit`, `evaluate` or `selftest` commands directly (the allowed protocol test suite runs `selftest`
as a CPU subprocess with CUDA hidden and `coordinate` on a temporary root whose guard raises
before any write), performed no GPU work (CUDA is not visible in this session), and did not enter
`.scratch/`, either RTGS-026 run root or any smoke directory. No image, mask, `.rtgsv`, `.npz`,
`.ply`, run or benchmark outcome content was opened or printed; `validate-data` hashed sealed bytes
opaquely; the only data-seal content read was its 700-byte header. The RTGS-025 and RTGS-026
RESULT and AUDIT records were not read; the RTGS-025 task JSON was read only for its split, seeds
and resolved configuration, and the RTGS-027 closed task record was read as the acceptance record
of the mechanism. No run root, RESULT or AUDIT exists for this task. No file was modified, no run
root was created, no checkpoint was selected, and the owner's task record was not written. The
sandbox-denied commands listed above touched no protected material and were re-done only in the
allowed forms. This review has no outcome access.
