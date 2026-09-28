# Prospective Protocol Review

- Task ID: `20260928_field_only_portfolio_stage_frame00008`
- Protocol SHA-256: `4dd390c27ff30d7db4b5fdb4be0d59f86c849f8be1dd910930adbd756868236c`
- Reviewer: `Claude-Code-Fable-5.1-reviewer`
- Verdict: `rejected`
- Outcome Access: `none`

## Scope

Prospective review of the RTGS-029 draft task at commit `e858611` on branch
`rtgs-029-field-only-portfolio` (worktree clean, including untracked files). The Driver derived
the driver, report module and protocol tests from the RTGS-028 driver approved by this reviewer at
`a496016` (`experiments/reviews/20260927_color_budget_reset_stage_frame00008_PROTOCOL_REVIEW.md`,
digest `7a55197b...`), removing the opacity-reset dimension and adding a nine-condition
one-factor portfolio with a downscale-4 training arm, a half-view arm and a new evaluation
operator. This review treated the Driver's handoff in `.agents/state/current-task.md` as claims to
check: it diffed the task JSON, driver, report module and tests against the approved RTGS-028
files, recomputed the protocol digest and the source-binding aggregate, ran the bound CPU tests,
traced every arm's configuration difference through the bound trainer and strategy code, traced
the evaluation operator through the bound gsplat backend and the installed gsplat package, and
probed the frozen schedules read-only.

Outcome access, stated precisely. No outcome of this task exists: no run root, no RESULT or AUDIT
record, no review artifact. At the Driver's explicit authorization, because this protocol is
motivated by the published RTGS-028 outcome, this review read in full the published records
`benchmarks/results/20260927_color_budget_reset_stage_frame00008_RESULT.md`, `_RESULT.json` and
`_AUDIT.md`. They were used for three purposes only: the motivation check (H1 rejected at −0.68 to
−0.75 dB per seed, H2 inconclusive), the runtime and capacity estimates for the 3600 s cell limit
(8000-step walls 77-85 s, 30000-step walls 301-365 s with 72-74k Gaussians, cap never reached),
and the seed-noise floor for the 0.1 dB rule (null-like paired deltas of −0.03 to −0.05 dB and
+0.01 to +0.17 dB). No content under `runs/` or `.scratch/`, and no RTGS-025 or RTGS-026 RESULT
or AUDIT record, was read. The header field keeps the contract's literal `none`, which refers to
sealed outcomes of the protocol under review.

Question: for field-only distillation from decoded `no_boundary` compact fields with the RTGS-025
masked objective on frame 00008 (22 training views; held-out C0001, C0018, C0029, C1002), which
single change to the 8000-step configuration improves held-out colour inside the four held-out
masks: SH degree 1 or 0, opacity and scale regularization 0.01, 30000 steps with densification
still stopping at 6000 (with the 30000-step means learning-rate horizon, or with the 8000-step
decay continued), or training at downscale 4; and, descriptively, how much does halving the
training views cost and how large is the photograph gap. Every treatment is gated against
`nb_base` per paired seed at +0.1 dB with crop LPIPS and outside-mask alpha margins of 0.005.

Evidence maturity this protocol may establish: a development screen of six simultaneous
comparisons on one previously outcome-exposed frame, one split, three fresh paired seeds
(9561-9563), 27 cells, with a new evaluation operator that makes the numbers non-comparable to
RTGS-025/026/028. A pass is a screening signal only; no default, SOTA, generalization,
physical-geometry, speed or VRAM claim can follow. Approval would say only that the frozen design
is fit to execute. This review finds that the design is fit for five of the six gated arms and
not for the sixth, for a reason that disclosure cannot cure and that a confirmation would inherit.

## Checks

Read in full: the task JSON; the driver; the report module; the protocol test file; the review
template; `experiments/reviews/README.md`; the RTGS-028 protocol review; the three RTGS-028
published records named above; `.agents/state/current-task.md`; the `diff -u` outputs of driver,
report and tests against the RTGS-028 files. Read in part: `src/rtgs/optim/trainer.py` (train
head and schedule resolution 330-400, learning rates and `means_gamma` 480-490, SH interval and
regularizer resolution 570-625, loop head, objective and regularizers 686-760, loss terms and the
means-LR step 860-885, train-metric switch 626-655, `_resolve_sh_interval`,
`_resolve_schedule_iterations` and `_resolve_means_lr_final_factor` 1555-1615);
`src/rtgs/optim/strategies.py` (DefaultStrategy construction 50-110, arena step 180-310);
`src/rtgs/render/gsplat_backend.py` 25-165; `src/rtgs/data/field_targets.py` 40-180;
`src/rtgs/core/camera.py` docstring; `src/rtgs/data/scene.py` (`center_and_extent` and
`bounds_hint`, grep); `scripts/experiment_contract.py` (`protocol_sha256` 246-259,
`_review_artifact_errors` 262-316, `_validate_protocol_review` 319-405, statuses and verdicts
61-62); the installed gsplat 1.5.3 package (`rendering.py` line 46 `eps2d: float = 0.3`,
`cuda/_torch_impl.py` 293-344 dilation and compensation, `strategy/default.py` 225-226 gradient
scaling, `cuda/csrc/RasterizeToPixels3DGSFwd.cu` 62-63 pixel centres); the first 700 bytes of the
data seal; the calibration JSON's camera ids; the file names under the frame's `rgb/` and `mask/`.

Commands executed (read-only, or the allowed CPU tests):

```text
git status --short; git rev-parse HEAD; git log --oneline -3; git log --oneline 1428175..HEAD
git diff --name-status 1428175 HEAD
git diff --stat 1428175 HEAD -- src/rtgs scripts/experiment_contract.py scripts/check_results_bundle.py pyproject.toml tests/test_field_targets.py
git status --porcelain --untracked-files=all; git status --ignored --short -- src/rtgs scripts/experiments tests experiments
git rev-parse HEAD:<each path in the object-id table> 1428175:src/rtgs
diff -u <RTGS-028 driver> <new driver>; diff -u <RTGS-028 report> <new report>; diff -u <RTGS-028 tests> <new tests>
sha256sum <task JSON, both data seals, driver, report, tests, bound src files, contract, bundle checker, pyproject, RTGS-028 files, task record>
head -c 700 experiments/data/20260928_field_only_portfolio_stage_frame00008.json
ls runs/ benchmarks/results/ experiments/reviews/ (filtered); ls -d .scratch/* (names only)
ls dataset/2025_03_07_stage_with_fabric/frame_00008/{rgb,mask,} (names only)
.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260928_field_only_portfolio_stage_frame00008.json
.venv/bin/python scripts/experiment_contract.py validate
.venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/20260928_field_only_portfolio_stage_frame00008.json
.venv/bin/python -m pytest -q tests/test_field_only_portfolio_protocol.py
.venv/bin/python -c <build_source_binding and verify_source_binding on the task's binding block>
.venv/bin/python -c <top-level task JSON diff against the RTGS-028 task JSON; seal and split equality>
.venv/bin/python -c <base vs RTGS-028 8k configs; per-config and per-seed differences; density-block equality; means-LR gamma and factors at 500..30000; 0.01**3.75; cell count; view-subset rule>
.venv/bin/python -c <calibration camera ids and count>
```

Denied by the review sandbox and re-done in an allowed form, none touching protected material:
one compound shell line (many-file `sha256sum` followed by a `for` loop of `git rev-parse`, a
`head -c` and a `git status --ignored`) and one `python -c` JSON comparison; each was re-run as
separate single commands. One `python -c` failed on this reviewer's own misuse of
`build_source_binding` (it takes the binding block, not the pattern list) and was re-run
correctly; `validate_task(check_live_source=True)` was not re-run because it requires a path
argument and the CLI `validate` covers it. CUDA is not visible in this session, so the one
CUDA-marked test skipped.

Results: `review-digest` prints `4dd390c27ff30d7db4b5fdb4be0d59f86c849f8be1dd910930adbd756868236c`,
the digest the Driver reported (the digest excludes only `status` and `protocol_review`,
contract 249-251; the raw file hashes to `d96fb6eb...`). `validate` prints OK; `validate-data`
prints OK. `build_source_binding` over the task's patterns returns 119 files and aggregate
`7f720ea0c04c83d642f776d364b6590f3e2cd7ecd98297b5d47beed6ce2b176f`, equal to the frozen values
(RTGS-028 bound 120 files; the difference is `tests/test_gsplat_opacity_reset.py`, no longer
bound); `verify_source_binding` returns no errors. Pytest: 6 collected, 5 passed, 1 skipped
(`test_box_render_matches_quadrature_average`, CUDA). Worktree clean at `e858611` including
untracked files; nothing ignored under the bound patterns except `__pycache__`. `git diff --stat
1428175 HEAD` over `src/rtgs`, the contract, the bundle checker, `pyproject.toml` and
`tests/test_field_targets.py` is empty, and the `src/rtgs` tree object is `1cebc4bc...` at both
commits. `runs/` holds no RTGS-029 root; no `benchmarks/results/20260928_*` file and no RTGS-029
review artifact exist. `.scratch/20260928_field_only_portfolio_stage_frame00008/` exists (the
disclosed smoke's scratch space) and was not entered. The data seal is byte-identical to the
RTGS-025/026/028 seal (`d01254e7...`, object `dc22987d...`): 26 view ids, held-out included,
same frame path and calibration.

- Drift, driver versus the approved RTGS-028 driver. The diff touches exactly: the module
  docstring; `CONDITIONS` now a nine-entry table of `(family, config id, training downscale,
  view subset)` (45-55) and a `TREATMENTS` tuple (56); new helpers `target_dir`, `mask_dir` and
  `view_subset` (390-404); `check_protocol_tables` checks config ids instead of budgets and adds
  the rule that the frozen `view_subsets` equal `{"all": train, "half": train[::2]}` (417-421);
  `access_guard` allows a fit worker only its own per-downscale target and mask directories
  (434-437, 448-451); `prepare` decodes each training view at downscales 8 and 4 with the
  fit-window check, mask cache and field cache per downscale, keeps photograph targets, teacher
  fidelity and object bounds at downscale 8 only, and stores cameras per downscale (476-526);
  `cached_scene` takes the downscale and the view list (541-561); `train_config` is keyed by
  config id (588-598); `fit` drops the reset callback, loads the frozen view subset, and records
  `config_id`, `training_downscale` and `view_subset` in the effective config (604-654); new
  `box_render` (687-694); `evaluate` builds downscale-4 held-out cameras and scores the
  box-averaged downscale-4 render against the unchanged downscale-8 reference, mask and field
  diagnostic (709-749); `selftest` adds the site-coincidence assertion (777-780), probes
  `nb_ds4` and adds a fifth probe on the downscale-8 field cache (792-808); the parity seed base
  becomes `926400`; `report_module` renames its module. Byte-identical: `mask_scores`,
  `dilate`/`erode`, the quadrature and image-query helpers, `random_initialization`,
  `model_from`, `load_source`, `load_view`, `parity_record`, `warmup`, `snapshot_source`,
  `write_environment`, `preflight`, `data_guard`, `source_guard`, `initialize`, the
  coordinator (re-entry refusal 827-829, fit-only 3600 s timeout 853, timed-out receipt
  869-881) and the canonical-root and cell-registration checks in `main` (920-921, 932-933).
  `decode_compact_view` is called with `backend="index"` in both phases; `make_query_backend`
  once with `"index"`.
- Drift, report module. `CONDITIONS`, `TREATMENTS`, `SELECTED = ("nb_base", 9561)`; `LONG` and
  `condition_iterations` (37-41); `gates` applies `_rule(arm, "nb_base", 0.1, 0.005, 0.005)` to
  the six treatments and two descriptive pairs (107-114); `decision_text`, the summary string
  and two notes reworded. `_rule`, `_paired`, `publish`, `history`, the once-only RESULT
  writers and `publish_failure` are byte-identical.
- Drift, tests. Renamed fixtures; the tables test adds the `TREATMENTS` equality, the 22/11 view
  counts and the broken-subset check; the config test asserts each config's changed keys and the
  `it30k_d6_lr8k` gamma; a new CUDA-skipped `box_render` test; the gate test covers pass at exact
  margins, an alpha-margin inconclusive, a reject and a zero-delta inconclusive; the selftest
  count becomes five probes.
- Drift, task JSON. Beyond identifiers, dates, the review-state reset and wording: `view_subsets`
  added, `opacity_reset` removed; seeds 9561-9563; `preprocessing.training_downscales [8, 4]`
  with the two-downscale operator and mask rules; `training.configs` replaces `budgets`;
  `resolved_training_configs` keyed by six config ids; `scoring.evaluation_render` replaces
  `evaluation_grid`; `decision_policy.per_arm`, `descriptive` and `multiplicity` replace the H1
  and H2 rules; 27 cells; preview `nb_base` 9561; source binding swaps the RTGS-028 files for
  the new three and drops the reset test. Datasets, split, the 28 sealed field files, the
  initialization text, resource protocol, execution controls, source lifecycle and
  `report_template_version: 2` are unchanged.
- Configurations. Every `base` config differs from the RTGS-028 `8k` config of the same seed
  position only in `seed`, so `nb_base` is the RTGS-025/026/028 main-path configuration as its
  purpose states. Against the same-seed `base`: `sh1` and `sh0` differ only in
  `target_sh_degree`; `reg` only in `opacity_reg` and `scale_reg` (0.01 each); `it30k_d6` only
  in `iterations`; `it30k_d6_lr8k` in `iterations` and `means_lr_final_factor`
  (`3.16227766016838e-08 = 0.01**3.75`). The density block is identical in all 18 configs
  (stop 6000), the three seeds of a config differ only in `seed`, and `train_config`
  round-trips all 18 (test-pinned). `check_protocol_tables` confirms 27 cells, each condition
  and seed exactly once; the order rotates the nine conditions by three positions per seed, so no
  condition always runs first or last.
- What each arm changes in the trainer. Only the means learning rate is scheduled
  (`optimizers["means"].param_groups[0]["lr"] *= means_gamma`, trainer 878, with
  `means_gamma = means_lr_final_factor ** (1 / iterations)`, 488); quaternion, scale, opacity,
  `sh0` and `shN` learning rates are constant (480-486). The SH band schedule is explicit
  (`sh_degree_interval` 1000; `active_degree = min(target, global_it // 1000)`, 693): degree 3
  from step 3000 in every SH-3 arm, degree 1 from step 1000 in `sh1`, degree 0 throughout in
  `sh0` (the model is built with `max(init.sh_degree, target_sh_degree)` bands, 386). The
  regularizers add `0.01 * sigmoid(opacities).mean()` and `0.01 * exp(scales).mean()` (744-751)
  on the `gsplat-default` strategy. In the strategy, the refine window 500-6000 every 100, the
  100-step refine pause after each would-be reset (`pause_refine_after_reset = every`, strategies
  76), the large-scale prune gate after step 3000 and the 100000 `enforce_budget` cap are
  identical in every arm; storage is `dynamic`, so the arena reset (296-297) is not on the path
  and gsplat 1.5.3's inert reset never fires.
- `nb_30k_d6_lr` versus the 8000-step decay. Its gamma is `(0.01**3.75)**(1/30000) =
  0.01**(1/8000) = 0.9994245193792801`, bit-equal to `base`'s gamma, so through step 8000 the
  cell is the `nb_base` cell continued (identical configuration, seed, targets and schedule, up
  to CUDA nondeterminism). Factors of the initial means LR: 0.75 at 500, 0.178 at 3000, 0.0316
  at 6000, 0.01 at 8000, 1e-3 at 12000, 1.8e-4 at 15000, 1e-5 at 20000, 3.2e-8 at 30000. What
  still differs from `nb_base` after step 8000: 22000 further steps in which the positions are
  effectively frozen from about step 12000, while quaternions, scales, opacities and all SH bands
  continue at their constant learning rates with frozen topology (no densification, no opacity
  or scale pruning and no reset after 6000). It isolates "more steps for the non-positional
  parameters", which is what the comparator purpose says once read with the factors.
  `nb_30k_d6` (gamma 0.9998465061085267) keeps 0.926 at 500, 0.631 at 3000, 0.398 at 6000,
  0.293 at 8000, 0.158 at 12000, 0.1 at 15000, 0.046 at 20000 and 0.01 at 30000: the RTGS-028
  package without the wider densification window. Neither arm separates the densification
  window inside this run; that separation is only relative to RTGS-028's `nb_30k_up`, which
  used different seeds and a different evaluation operator.
- Evaluation operator, geometry. `Camera` puts the top-left pixel centre at (0.5, 0.5)
  (camera.py line 4); `downscale_pinhole` divides `fx, fy, cx, cy` by the factor (field_targets
  52-72); gsplat evaluates at `(j + 0.5, i + 0.5)` (RasterizeToPixels3DGSFwd.cu 62-63);
  `quadrature_sites` offsets are `(0.25, 0.75) * factor` (74-87). Downscale-4 pixel centres
  `(2i + 0.5) * 4 = 8i + 2` and `(2i + 1.5) * 4 = 8i + 6` therefore coincide with the
  downscale-8 sites, as the selftest asserts (777-780); `avg_pool2d(2)` of a `W // 4` grid
  gives `W // 8` because `downscale_pinhole(..., 8)` runs first in `evaluate` and rejects a
  canvas not divisible by 8. The reference is the four-site bilinear mean of the photograph, the
  mask is the four-site mean of the lossless mask binarized at 0.5, and the alpha descriptors use
  the box-averaged alpha; the LPIPS crop is unchanged. So the site positions match the reference
  quadrature exactly. That is a statement about positions only.
- Evaluation operator, neutrality. The bound backend calls `rasterization` without `eps2d` and
  with `rasterize_mode="classic"` (gsplat_backend 127-143; `antialiased` False in every frozen
  config), so gsplat adds `0.3 * I` to every projected 2D covariance in the render's own pixel
  units without opacity compensation (`_torch_impl.py` 333, compensation only when requested
  341-344; `rendering.py` 46 default). A model trained at downscale 8 learned to reproduce its
  targets with footprints of variance `sigma^2 + 0.3` (downscale-8 px^2); rendered at downscale
  4 the same Gaussian has variance `sigma^2 + 0.075` in downscale-8 units, that is 0.225 px^2
  thinner than in training, with unchanged peak opacity. Counterexample by numbers: a Gaussian
  of sigma 0.5 px renders with sigma 0.742 px during training and 0.570 px at evaluation (23
  percent thinner, about 41 percent less coverage mass); sigma 1 px: 1.140 to 1.037 (9 percent,
  about 17 percent less mass); sigma 2 px: 2.074 to 2.019 (2.6 percent). RTGS-028 ended its
  8000-step cells with 43-48k Gaussians on foregrounds of 19-31k downscale-8 pixels, so a large
  share of Gaussians sit at or below one pixel, and on a black background every lost coverage
  inside the mask darkens the pixel. The `nb_ds4` model, trained with the downscale-4 dilation,
  is rendered at evaluation with exactly its training operator; a second, smaller mismatch runs
  the same way (the downscale-8 arms matched a four-site box prefilter with a point render and
  are now box-averaged a second time, while the downscale-4 arm's render sites are the reference
  sites). Both effects are systematic, not seed noise, and both favour `nb_ds4` in its gate
  against `nb_base` by an amount that cannot be bounded prospectively and is plausibly of the
  order of the 0.1 dB rule; a three-seed unanimity rule offers no protection against a
  systematic bias. The other five treatments share `nb_base`'s mismatch and are compared fairly
  among themselves; the descriptive pairs (`nb_v11`, `ph_base`) are downscale-8 arms and are
  fair. The Driver's assumption in the handoff ("box-averaged downscale-4 rendering is the fair
  common evaluation for mixed training resolutions") is therefore not established, and the claim
  boundary's "(matching the reference quadrature)" must not be read as operator neutrality.
- Downscale-4 targets and masks. `prepare` runs the same pipeline at both factors: soft
  packed-alpha fraction from `decode_alpha_grid` (four sites per pixel at either factor), the
  index decode with `supersample=2`, and the fit-window abort `(soft > 0) & (coverage < 1)`
  (496-502). The downscale-4 sites sit 1 and 3 full-resolution pixels from the pixel edge, so a
  view that passed the downscale-8 check in RTGS-028 is at least as likely to pass here; a
  failure is an accepted late abort that consumes the root. Object bounds and the shared
  initialization use downscale-8 cameras and masks only (520), so the initialization and the
  extent-scaled means learning rate are identical in every arm. What "training at downscale 4
  with the same configuration" also changes, beyond pixel count: gsplat scales `means2d`
  gradients by `width / 2` and `height / 2` (`strategy/default.py` 225-226), so for the same
  NDC-space error the pixel-unit positional gradient doubles and the 0.0002 grow threshold is
  effectively halved (more cloning and splitting, and the 100000 cap may engage where RTGS-028
  never reached it); the 11 x 11 SSIM window and the 0.3 px^2 dilation cover half their physical
  extent; the soft masks carry 16 sites per downscale-8 pixel. This is a "resolution package",
  inherent to the declared treatment, and must be disclosed as such.
- View subset. `half = train[::2]` gives C0004, C0006, C0009, C0014, C0020, C0022, C0026,
  C0030, C0034, C0039, C1001 (11 views), frozen in `view_subsets` and enforced by
  `check_protocol_tables` and the test. The `nb_v11` worker keeps the 22-view bounds and shared
  initialization, may open any of the 22 cached downscale-8 target files (the guard allows the
  directory) but loads the frozen 11 (620-621, recorded as `train_view_ids`), and at the fixed
  8000-step budget samples each view about twice as often. Descriptive only, as frozen.
- Leakage boundaries. `prepare` opens only the 22 training photographs, mask PNGs and compact
  views (the guard denies any held-out source open before `evaluate`, 455-456, and any frame
  open in `fit` or `initialize`, 457-458); `initialize` reads only `targets/metadata.json`;
  a fitting worker may open only its own family-and-downscale target directory, its own mask
  directory and the metadata; `evaluate` is the first phase to open held-out photographs, masks
  and compact views and runs only after all 27 receipts read `completed` (702-705). Sealed
  bytes, including held-out archives, are hashed opaquely at entry and exit; the split digest is
  written at exit. The selftest denies five probes (test-pinned). The frame directory holds
  exactly the 26 sealed photographs and their masks; the dome calibration lists 34 cameras, of
  which eight (C0000, C0007, C0010, C0013, C0016, C0024, C0038, C1005) have no photograph in
  this frame, so "no further calibrated photographs" is correct for this frame.
- Gates versus written policy. `_rule` implements pass as `psnr >= base + 0.1 and lpips <= base
  + 0.005 and outside_alpha_mass <= base + 0.005`, reverse as `psnr <= base - 0.1`, verdict pass
  iff every seed passes, reject iff every seed reverses, else inconclusive, for each of the six
  treatments against `nb_base` (report 79-114). This is `decision_policy.per_arm` in written
  inclusive form; the hypothesis text agrees; the test exercises all four outcomes. The
  descriptive pairs carry no verdict. Multiplicity: six uncorrected comparisons, disclosed in
  the policy, the claim boundary, the decision text and a report note; with the RTGS-028
  null-like paired spreads (0.03 to 0.17 dB across seeds) the unanimous +0.1 dB floor is a
  reasonable screen against seed noise, but not against the systematic operator bias above.
- Stopping and timeout. The 3600 s limit applies to `fit` workers only; `prepare` (now two
  decodes per view) and `evaluate` are untimed. From the RTGS-028 walls on the same GPU (RTX
  3050: 77-85 s per 8000-step cell, 301-365 s per 30000-step cell carrying 72-74k Gaussians),
  a `nb_30k_d6` cell carrying about 44k Gaussians should take about 300 s and a `nb_ds4` cell,
  with four times the pixels and a larger population, a few hundred to roughly 800 s; both are
  well inside the limit if no other process shares the GPU. Non-finite renders, models and losses
  abort fail-closed as before.
- Seeds. 9561-9563 are fresh (RTGS-025 9261-9263, RTGS-026 9361-9363, RTGS-028 9461-9463);
  the parity seeds `926400 + index` are gate constants.
- Smoke disclosure. `claim_boundary` discloses one pre-review non-protocol smoke (60/120
  iterations, two training views standing in for held-out, no real held-out access) that
  exposed training-view teacher fidelity and meaningless tiny-budget values, with no threshold
  change afterwards; the handoff agrees ("all phases and arms"). Not verifiable from tracked
  state (`main` allows only the canonical root, which does not exist); the scratch directory was
  not entered.
- Report handling. `condition_iterations` maps the two `LONG` conditions to 30000 and the rest
  to 8000, test-pinned against the resolved configs; `publish` rejects a cell whose
  `executed_iterations` differs and `fit` rejects a run that did not reach its frozen final
  iteration; stage markers use the per-cell total; the summary lists all nine group PSNRs.
- Source binding completeness. Bound: all of `src/rtgs` (unchanged since the RTGS-028 approval
  commit), the contract, the bundle checker, the driver, the report, the new test,
  `tests/test_field_targets.py` and `pyproject.toml`. Not bound and recorded only by
  `environment.json` and `preflight.json`: the installed torch, gsplat and lpips packages and
  the LPIPS AlexNet weights, as in RTGS-025/026/028. The RTGS-027 canary test is no longer
  bound, so the boundary sentence "upstream gsplat 1.5.3 performs no opacity reset in any arm"
  rests on the preflight version plus the RTGS-027/028 evidence.
- Task record. Driver `Claude-Code-Opus-5.5-driver`, Reviewer label
  `Claude-Code-Fable-5.1-reviewer`, Turn `reviewer`, Status `In review`, contract path correct,
  handoff digest reproduces. The handoff's `verify.sh` and GPU-smoke statements were not
  verified and were not used as evidence.

Reviewed identities at `e858611` (`sha256sum`; `unchanged` means identical to the state approved
for RTGS-028 at `a496016`):

| File | SHA-256 |
|---|---|
| `experiments/tasks/20260928_field_only_portfolio_stage_frame00008.json` | `d96fb6eba88e8ff9eab68e0c610bb3ff05efc43625bcef4113f15b6e176b7cbf` |
| `experiments/data/20260928_field_only_portfolio_stage_frame00008.json` | `d01254e739961dcb1acc4a9abca6694bb63e319ff179581ebe2ec2ae08706087` (identical to the RTGS-025/026/028 seal) |
| `scripts/experiments/20260928_field_only_portfolio_stage_frame00008.py` | `6d6db12476ca1d43f0b2a1cf8d7d07a8e1bf3bc63c372dd8e8b9c0e7d8093340` |
| `scripts/experiments/20260928_field_only_portfolio_stage_frame00008_report.py` | `3ef08928987efd760e1fe3053a47fa213f27dd8ede2f6fee5a3fc257ea1f283b` |
| `tests/test_field_only_portfolio_protocol.py` | `f2833e2fe9deec0390c3bf3bbd6aee125eec932077882ddca2c41f5958cbef2c` |
| `tests/test_field_targets.py` | `0a817166c45df13d64cd8e8f83aafbf7656e1bc37ef8bf26176bf6def75a72cc` (unchanged) |
| `src/rtgs/optim/trainer.py` | `e226d01528897fd00dea8c03f345190c9abd3b4cf5cd5e2013967ca5802d39ba` (unchanged) |
| `src/rtgs/optim/strategies.py` | `947a6e02c229c17ee639be98aede565292ccc3aa0303c84f5130b897c2bea91d` (unchanged) |
| `src/rtgs/optim/density.py` | `ad889cd67f7d6e54c7223836dfc27d10555a18c98110f5f465f0df71d352254a` (unchanged) |
| `src/rtgs/data/field_targets.py` | `59901d820ef291381c27cb45ba68161cdad1195eff7ea891ae281b60bca9829f` (unchanged) |
| `src/rtgs/render/gsplat_backend.py` | `2c6a5e0029af86c85e45975fcfceb16d68d342458893eba7fd227db4747bf409` (unchanged) |
| `scripts/experiment_contract.py` | `15f129398aabb46a50d435c76fec27b4df6c75e6d08d2e7d78a5a96657b25959` (unchanged) |
| `scripts/check_results_bundle.py` | `0f5e1293a4c6651d1fed80237bd1319f44ed2ca268d81cb27ce87d2e54f2fedf` (unchanged) |
| `pyproject.toml` | `e5b15e72400e8e14b83a26dc64ef8a85ea985642ef9ef079e425326b5e0bc8d9` (unchanged) |
| `scripts/experiments/20260927_color_budget_reset_stage_frame00008.py` | `9de0cc5ca8815a00282053dce7d1fbebf8b3c56d863060bcd846ddbfd3f18ba0` (diff base, unchanged) |
| `scripts/experiments/20260927_color_budget_reset_stage_frame00008_report.py` | `d85dcef08ee15592866fe7c11e995bbeafafc387d3871e299fd0ee1fdfcaa36e` (diff base, unchanged) |
| `tests/test_color_budget_reset_protocol.py` | `c8210a245eb7a27517727cccc7efba9e04c3fc8c244a864f1f0f7b5c5733c8fb` (diff base, unchanged) |
| `experiments/tasks/20260927_color_budget_reset_stage_frame00008.json` | `8f7c6fc33ac1b098e473bedc0cc53ee441900476a23723e39f50cb0a6553b7c8` (diff base; approved state) |
| `experiments/reviews/20260927_color_budget_reset_stage_frame00008_PROTOCOL_REVIEW.md` | `331b1db6b829ca36b41513de6d8c8de0a0a598798850902d8251e657f4736141` |
| `benchmarks/results/20260927_color_budget_reset_stage_frame00008_RESULT.md` | `1ddef7f2722ba7de413fe54a9fb9e5ad4891a1dc2f8d5d848949b737d6c68a19` (read, authorized) |
| `benchmarks/results/20260927_color_budget_reset_stage_frame00008_RESULT.json` | `c2a8dca1602c79223799fe132fd6dbacff80266a204f7a23e5a7a4e70fbccd9f` (read, authorized) |
| `benchmarks/results/20260927_color_budget_reset_stage_frame00008_AUDIT.md` | `b364bb80d9c6e8d13ec3ecca144d5e8d60c7b57341b540ed97a24240ba54a3be` (read, authorized) |
| `.agents/state/current-task.md` | `5bec79e102435909d9e75ef9ff54a81cbc621186570aeb7305c6d9548f3104ed` |

Git object ids at `e858611` (`git rev-parse HEAD:<path>`):

| Path | Object id |
|---|---|
| `src/rtgs` (tree) | `1cebc4bc92e5e83a904b4fc4394fc0429423bbd2` (identical at `1428175`) |
| `src/rtgs/optim/trainer.py` | `81f88540a171907c49bd37cbff8f5b463c231e0d` |
| `src/rtgs/optim/strategies.py` | `8c4e99a2b462811bc9fca5a529123baaee1516fa` |
| `src/rtgs/optim/density.py` | `54c4d0df8d08153585727973e8d40f3aacb5c5b7` |
| `src/rtgs/data/field_targets.py` | `80caeefc9c4f3e1ebcc4f03e39688004feeb46ee` |
| `src/rtgs/render/gsplat_backend.py` | `8415ca230376f0ca27aa81b0b413af5a19ef766d` |
| `scripts/experiment_contract.py` | `3d4f251972226e5372dfb5346283c65f2e73e62e` |
| `scripts/check_results_bundle.py` | `4be3439031998a35caf1b6d023085dccee659ceb` |
| `pyproject.toml` | `f3911412950125f290bc65a49818ed6949c820a7` |
| `tests/test_field_targets.py` | `1f522a73de24a3ce91c293ee52459f9592f439cd` |
| `tests/test_field_only_portfolio_protocol.py` | `1eee0d563c79a3baf25c7e654ef97f77953ed289` |
| `scripts/experiments/20260928_field_only_portfolio_stage_frame00008.py` | `923960b2499902c5819bb9c223a43e71a0de37ff` |
| `scripts/experiments/20260928_field_only_portfolio_stage_frame00008_report.py` | `178cf099953c4baa97fe9d8cc956ada8b6fbe0b0` |
| `experiments/tasks/20260928_field_only_portfolio_stage_frame00008.json` | `316d42e50c3240acd218c26b903f36ff186f7892` |
| `experiments/data/20260928_field_only_portfolio_stage_frame00008.json` | `dc22987ddbfefac83d10deae666616451b8cff8c` (same object as the RTGS-025/026/028 seal) |
| `.agents/state/current-task.md` | `b13333d2f27e6b648ede1ac6610694c73d557de9` |

## Findings

Rejected. The derivation claim is verified at the byte level, the leakage boundaries, seal
cross-checks, coordinator lifecycle, resource protocol and once-only evidence writers are the ones
approved for RTGS-028, every arm changes exactly its declared configuration factor, the frozen
schedules do what the comparator purposes say, the gates implement the written policy inclusively
per seed, and the multiplicity is disclosed. Five of the six gated comparisons (`nb_sh1`,
`nb_sh0`, `nb_reg`, `nb_30k_d6`, `nb_30k_d6_lr`) and both descriptive pairs are sound under the
frozen operator, because every arm involved shares `nb_base`'s train-versus-evaluation render
mismatch. The sixth, `nb_ds4`, is not: the evaluation render equals the training render for that
arm alone, and gsplat's classic-mode dilation (0.3 px^2 in the render's own pixel units, no
opacity compensation, bound backend and installed package verified) makes every downscale-8-trained
Gaussian render measurably thinner at downscale 4 than it did in training, while the
downscale-4-trained model is unaffected. The bias is systematic, one-sided, of unknown size and
plausibly of the order of the 0.1 dB rule; a three-seed unanimity rule cannot filter it, and a
confirmation that inherits this evaluation convention would inherit the bias, so a spurious
"resolution helps" result could reach the ledger as confirmed. Disclosure cannot cure a gate
whose verdict the operator can produce on its own; the fix is small and does not add a cell. The
nine-condition set is otherwise the minimal one-factor design for the levers the record says the
user asked for; dropping `nb_30k_d6` or `ph_base` would be simpler but would drop questions the
user posed. Alternatives considered and not required: evaluating each arm at its own training
resolution (replaces the dilation bias with a point-versus-box reference bias of the opposite
sign), antialiased evaluation renders (a train-versus-evaluation mode mismatch for every arm), or
a downscale-4 reference (the same asymmetry). The two-operator design below brackets the true
effect from both sides.

Required changes, bounded to one round:

1. Two-operator evaluation. Keep the downscale-4 box operator as the primary metric set and add,
   for every cell and held-out view, the RTGS-028 operator: a point-sampled render at the
   downscale-8 held-out camera (`renderer.render(final, cameras8[index])`, colour and alpha
   exactly as in the approved RTGS-028 driver) scored by the unchanged `mask_scores` on the same
   reference, mask, bands and LPIPS crop, stored per view as a second metric set (for example
   `ds8_point`) with a second per-cell mean, and checked in `publish` with the same finiteness
   and row-order rules. Report group means and paired deltas under both operators. Gate
   `nb_ds4` under both: pass iff the written rule passes under both operators in every seed,
   reject iff it reverses under both in every seed, otherwise inconclusive (operator-dependent).
   The five downscale-8 treatments keep the primary gate, with the second operator reported
   descriptively; this also restores comparability of the downscale-8 arms with the
   RTGS-025/026/028 lineage. Task JSON: `scoring` describes both operators and names the
   mechanism (classic-mode 2D dilation of 0.3 px^2 in the render's pixel units, so `nb_ds4` is
   the only arm whose evaluation render equals its training render; site coincidence is a
   property of positions only); `decision_policy.per_arm` carries the two-operator clause for
   `nb_ds4`; `primary_metrics` stay as they are and a second metric block defines the
   point-sampled set; `claim_boundary` states both operators and drops or qualifies "(matching
   the reference quadrature)". Tests: extend the gate test with a two-operator case (pass under
   one operator only gives inconclusive) and keep the CUDA-skipped box test.
2. Disclosures written into the frozen protocol (the digest changes with change 1 anyway), so
   that the once-only RESULT boundary carries them instead of the audit, as happened in RTGS-028
   with the means-LR horizon. Exact wording is the Driver's; the facts must be present:
   (a) `nb_30k_d6` purpose: means-LR factors 0.398 at 6000, 0.293 at 8000, 0.1 at 15000, 0.01 at
   30000, against 0.0316 at 6000 and 0.01 at 8000 in `nb_base`; (b) `nb_30k_d6_lr` purpose:
   identical to `nb_base` through step 8000, then 22000 further steps with the means LR at 1e-3
   of its initial value by 12000 and 3.2e-8 at 30000 (positions effectively frozen), quaternions,
   scales, opacities and SH at constant rates, no densification, pruning or reset after 6000;
   (c) `nb_ds4` purpose: a resolution package that also halves the physical scale of the 0.3 px^2
   dilation and the 11 x 11 SSIM window and doubles pixel-unit positional gradients for the same
   NDC error, so the 0.0002 grow threshold is effectively lower and the 100000 cap may engage;
   (d) hypothesis or question: the densification-window separation is relative to RTGS-028's
   `nb_30k_up` (different seeds and evaluation operator), not a within-run comparison; (e)
   `nb_v11` purpose: keeps the 22-view object bounds and shared initialization, loads the frozen
   11 views, and samples each view about twice as often at the fixed budget.

Conditions carried to the resubmitted protocol, so that the next round can be short:

1. Resubmission. Make only the changes above, re-run `review-digest`, `validate`,
   `validate-data`, the protocol tests and `build_source_binding` (new file count and aggregate
   in the task), and hand off with the new digest; this reviewer will diff against the
   identities recorded above. Set `status: blocked` with this artifact recorded until then
   (contract 403-404).
2. Source state at `init-run`. Clean tree; `src/rtgs` tree `1cebc4bc...`, the contract, bundle
   checker, `pyproject.toml` and `tests/test_field_targets.py` at the object ids above; the new
   driver, report and test at their resubmitted objects; `init-run` re-verifies the aggregate
   fail-closed. Any bound change after the new approval invalidates it.
3. Runtime versus the frozen 3600 s cell limit. The RTGS-028 walls on the same GPU give about
   300 s for a `nb_30k_d6` cell and a few hundred to roughly 800 s for a `nb_ds4` cell; the
   limit is adequate only if no other compute process shares the GPU (`resource_protocol.scope`
   and the `preserve_existing_gpu_processes` guard). A time-out after `prepare` began consumes
   the root; do not kill other GPU processes to make room.
4. Disclosures the RESULT and AUDIT must carry: frame 00008 is outcome-exposed; both operators
   with the dilation mechanism and, for every arm, the group means under both; the per-arm
   means-LR factors; for `nb_ds4` the final and maximum live counts and `density_stats.
   pruned_to_budget` per cell; for `nb_v11` that `train_view_ids` equals the frozen half list
   and `sampled_train_views` lie in 0..10; the gsplat version from `preflight.json` (1.5.3
   expected) as the basis of "no opacity reset in any arm", citing the RTGS-027/028 canary
   evidence since the canary test is not bound here; the pre-review smoke as frozen; timings on
   a possibly shared GPU are descriptive; six uncorrected comparisons, any pass a screening
   signal; RTGS-025/026/028 numbers are not comparators in any gate.
5. One `run` invocation; the coordinator refuses a root containing `targets/` or
   `preparation.json`. A failure before `prepare` begins is preserved in `execution_failure.json`
   and disclosed before any re-entry; a failure after `prepare` began consumes the root, no
   sibling root may be created, and a repeat requires a new task id and review.
6. Accepted late aborts that consume the root: the index-parity, packed-alpha equality, sealed
   digest and fit-window checks in `prepare` at either downscale; a CUDA OOM in a downscale-4 or
   30000-step cell; and the held-out camera-record checks in `evaluate`.
7. Nothing beyond the 27 registered cells may be trained under this root, and the `nb_base` seed
   9561 preview is the only root-level model.

Optional, not conditions: `record_train_metrics` (which requires
`internal_checkpoint_evaluation=True`, trainer 636-638) so that the overfitting hypothesis the
question invokes has training-view PSNR on record, as the RTGS-028 audit suggested; reading each
cell's iterations from `effective_config.json` instead of the condition id in
`condition_iterations`.

## Protected Actions Not Taken

The reviewer did not run `init-run`, did not execute the driver's `run`, `prepare`,
`initialize`, `fit`, `evaluate` or `selftest` commands directly (the allowed protocol test suite
runs `selftest` as a CPU subprocess with CUDA hidden and `coordinate` on a temporary root whose
guard raises before any write), performed no GPU work (CUDA is not visible in this session), and
did not enter `.scratch/`, any run root or any smoke directory. No image, mask, `.rtgsv`,
`.npz`, `.ply`, run content or `runs/` file was opened or printed; `validate-data` hashed sealed
bytes opaquely; the only data-seal content read was its 700-byte header; the only dataset content
read was file names under `rgb/` and `mask/` and the camera ids in the calibration JSON. The
RTGS-028 RESULT.md, RESULT.json and AUDIT.md were read in full under the Driver's explicit
authorization as published records of a closed task, for the purposes stated in Scope; no
RTGS-025 or RTGS-026 RESULT or AUDIT record was read. No run root, RESULT or AUDIT exists for this
task. No file was modified, no run root was created, no checkpoint was selected, and the owner's
task record was not written. The sandbox-denied commands listed above touched no protected
material and were re-done only in the allowed forms. This review has no outcome access to the
protocol under review.
