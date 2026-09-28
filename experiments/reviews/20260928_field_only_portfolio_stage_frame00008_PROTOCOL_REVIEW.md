# Prospective Protocol Review

- Task ID: `20260928_field_only_portfolio_stage_frame00008`
- Protocol SHA-256: `345e57a95cac3edb93fa7cc56d7a0ec43edb9d398f08da4e895583d9b32dfd68`
- Reviewer: `Claude-Code-Fable-5.1-reviewer`
- Verdict: `approved`
- Outcome Access: `none`

## Scope

Round 2 of the prospective review of the RTGS-029 draft task, at commit `9abf11c` on branch
`rtgs-029-field-only-portfolio` (worktree clean, including untracked files). Round 1 rejected
digest `4dd390c27ff30d7db4b5fdb4be0d59f86c849f8be1dd910930adbd756868236c` at commit `e858611`;
that record is archived at
`experiments/reviews/20260928_field_only_portfolio_stage_frame00008_PROTOCOL_REVIEW_V1_REJECTED.md`
and the Driver's response at
`experiments/reviews/20260928_field_only_portfolio_stage_frame00008_DRIVER_RESPONSE_R1.md`. This
review treated the response and the handoff in `.agents/state/current-task.md` as claims to check:
it diffed every tracked change since `e858611`, compared the two task JSONs key by key, recomputed
the protocol digest and the source-binding aggregate, ran the validators and the bound CPU tests,
traced the new evaluation operator against the approved RTGS-028 driver, and traced the two-operator
gate through the report module and its tests. Nothing in the round-1 review is re-derived here
except where the revision touched it; the round-1 identities and reasoning stand for everything
else.

Outcome access, stated precisely. No outcome of this task exists: no run root under `runs/`, no
`benchmarks/results/20260928_*` file, no canonical review artifact. In this round no RTGS-028,
RTGS-025 or RTGS-026 RESULT or AUDIT record was read; the RTGS-028 figures quoted below (walls,
seed-noise floor) are taken from this reviewer's archived round-1 review, which read the RTGS-028
records under the Driver's explicit authorization. Nothing under `runs/` or `.scratch/` was read
or listed. The header field keeps the contract's literal `none`, which refers to sealed outcomes
of the protocol under review.

Question (unchanged from round 1): for field-only distillation from decoded `no_boundary` compact
fields with the RTGS-025 masked objective on frame 00008 (22 training views; held-out C0001,
C0018, C0029, C1002), which single change to the 8000-step configuration improves held-out colour
inside the four held-out masks: SH degree 1 or 0, opacity and scale regularization 0.01, 30000
steps with densification still stopping at 6000 (with the 30000-step means learning-rate horizon,
or with the 8000-step decay continued), or training at downscale 4; and, descriptively, how much
halving the training views costs and how large the photograph gap is. Every treatment is gated
against `nb_base` per paired seed at +0.1 dB with crop LPIPS and outside-mask alpha margins of
0.005. New in this revision: every cell is scored under two evaluation operators (downscale-4 render
box-averaged to the downscale-8 grid, primary; point-sampled downscale-8 render, the
RTGS-025/026/028 operator), and `nb_ds4` alone is gated under both.

Evidence maturity this protocol may establish: a development screen of six simultaneous,
uncorrected comparisons on one previously outcome-exposed frame, one split, three fresh paired
seeds (9561-9563), 27 cells. A pass is a screening signal only; no default, SOTA, generalization,
physical-geometry, speed or VRAM claim can follow, and no cross-task comparison enters any gate.
Approval says only that the frozen design is fit to execute.

## Checks

Read in full: the `git diff e858611 HEAD` of the task JSON, driver, report module, protocol tests
and task record; the two review files named above; the report module and the protocol test file
at HEAD; the review template; `experiments/reviews/README.md`; `.agents/state/current-task.md`.
Read in part: the driver at HEAD (`CONDITIONS`, `TREATMENTS`, `METRICS` 41-68; `mask_scores`
169-206; `box_render` and `evaluate` 689-786); the approved RTGS-028 driver `evaluate` 659-734;
`scripts/experiment_contract.py` (`protocol_sha256` 246-259, review-artifact validation 262-407,
required task keys 425-470, source binding 1022-1101, lock-time source verification 2190-2209,
review header regex 103-113); the task JSON fields printed by the commands below.

Commands executed (read-only, or the allowed CPU tests):

```text
git status --short; git rev-parse HEAD; git log --oneline -5; git log --format='%H %an %ad %s' --date=iso -2
git diff e858611 HEAD --stat; git diff --name-status e858611 HEAD
git diff e858611 HEAD -- <task JSON> <driver> <report> <tests> .agents/state/current-task.md
git diff --stat 1428175 HEAD -- src/rtgs scripts/experiment_contract.py scripts/check_results_bundle.py pyproject.toml tests/test_field_targets.py
git diff e858611 HEAD --stat -- experiments/data/ src/ pyproject.toml scripts/experiment_contract.py scripts/check_results_bundle.py
git status --porcelain --untracked-files=all
git rev-parse HEAD:<path>   (one call per path in the object-id table)
git show e858611:experiments/tasks/<task>.json > <scratchpad>/task_e858611.json; sha256sum <that file>
sha256sum <task JSON, data seal, driver, report, tests, bound src files, contract, bundle checker, pyproject, task record, both review files, RTGS-028 driver>
ls runs/ benchmarks/results/ experiments/reviews/   (names only, filtered by 20260928)
.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/<task>.json
.venv/bin/python scripts/experiment_contract.py validate
.venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/<task>.json
.venv/bin/python -m pytest -q tests/test_field_only_portfolio_protocol.py
.venv/bin/python -c <build_source_binding and verify_source_binding on the task's binding block>
.venv/bin/python -c <key-level comparison of the e858611 and HEAD task JSONs, including comparators, scoring, decision_policy, frozen_configuration, source_binding>
.venv/bin/python -c <print seeds, cells, split, view_subsets, config ids, downscales, metric ids, preview policy, status, protocol_review, owner, run_command>
.venv/bin/python -c <print primary-metric aggregation text, scoring.evaluation_render, decision_policy, unchanged comparator purposes>
rg over scripts/ for primary_metrics / secondary_metrics and the contract's review regexes
```

Denied by the review sandbox and re-done in an allowed form, none touching protected material: the
pytest invocation with a `CUDA_VISIBLE_DEVICES=""` prefix and a pipe (re-run plain; CUDA is not
visible in this session anyway, so the one CUDA-marked test skipped); one multi-line `python -c`
(re-run as one line); a `for` loop of `git rev-parse` and two `git ls-tree` calls (not among the
allowed git verbs), re-done as single `git rev-parse` calls.

Results:

- `review-digest` prints `345e57a95cac3edb93fa7cc56d7a0ec43edb9d398f08da4e895583d9b32dfd68`, the
  digest the Driver reported (the raw file hashes to `93f9cdc0...`; the digest excludes only
  `status` and `protocol_review`, contract 249-251). `validate` prints OK; `validate-data` prints
  OK. Pytest: 7 collected, 6 passed, 1 skipped (`test_box_render_matches_quadrature_average`,
  CUDA), consistent with the Driver's "7 pass on the GPU". `build_source_binding` over the task's
  eleven patterns returns 119 files and aggregate
  `2848f1d6695c69680fd8a724dc8d1408d33ad92d031e5c0ae3fa061e98831b6e`, equal to the frozen
  values; `verify_source_binding` returns no errors.
- Tracked changes since `e858611` are exactly seven files: the task record, the two review files
  (added), the task JSON, the driver, the report module and the protocol tests. `src/rtgs` (tree
  `1cebc4bc...`, identical at `1428175`, `e858611` and HEAD), the contract, the bundle checker,
  `pyproject.toml`, `tests/test_field_targets.py` and the data seal are unchanged; the worktree is
  clean. The round-1 task JSON extracted from `e858611` hashes to `d96fb6eb...`, the value in the
  round-1 identity table, so the comparison base is the reviewed state.
- No drift. Key-level comparison of the two task JSONs: one key added (`secondary_metrics`); six
  keys changed (`hypothesis`, `claim_boundary`, `comparators` in the `purpose` of `nb_30k_d6`,
  `nb_30k_d6_lr`, `nb_ds4` and `nb_v11` only, `scoring.evaluation_render`,
  `decision_policy.per_arm`, `frozen_configuration.source_binding.aggregate_sha256`); everything
  else byte-equal: seeds `[9561, 9562, 9563]`; split 22 training and held-out C0001, C0018,
  C0029, C1002; `view_subsets.half` the frozen 11 views; `execution_order` 27 cells in the
  rotated order; `resolved_training_configs` six ids; `training`; `preprocessing`
  (`training_downscales [8, 4]`); `stages`; `datasets`; `data_seal`; `run_command`;
  `resource_protocol`; `execution_guards`; `execution_controls`; `input_policy`;
  `primary_metrics`; `preview_policy` (`nb_base` seed 9561); `blockers`; `owner`;
  `source_lifecycle`; `report_template_version: 2`; `status: draft` and a pending
  `protocol_review`. The thresholds in `per_arm` (0.1 dB, 0.005, 0.005) and the `descriptive`,
  `multiplicity`, `consequence` and `stopping` clauses are unchanged. Driver diff: docstring and
  `evaluate` only; `check_protocol_tables`, `access_guard`, `prepare`, `initialize`, `fit`,
  `selftest`, `coordinate` and `main` are untouched, so the leakage boundaries, re-entry refusal,
  3600 s fit-only timeout and once-only writers reviewed in round 1 stand.
- Required change 1, evaluation operator. `evaluate` builds `cameras8` with
  `downscale_pinhole(source.cameras[0], 8)`, the references and masks with
  `quadrature_sites(..., 8)` and `quadrature_image`, and `cameras4` with factor 4 (713-723). For
  every cell and held-out view it renders the box operator first, then
  `renderer.render(final, cameras8[index].to("cuda:0"))`, takes `.color.cpu()` and
  `.alpha.cpu().reshape(masks[index].shape)`, and scores with the unchanged `mask_scores` on the
  same reference, mask, bands and LPIPS model (749-758). That is, token for token, the RTGS-028
  driver's operator (lines 703-708 there, with its `factor` equal to 8), on the same
  `get_rasterizer("gsplat", device="cuda:0")` and the same unchanged `src/rtgs`. The point row
  keeps exactly the five `METRICS` keys; `mean_ds8_point` is the per-cell mean over those keys
  (778-780). Both operators score the same `final` model under `no_grad` in the same loop, after
  all 27 receipts read `completed`; no new file is opened, so the access guard is unchanged.
- Required change 1, publish and gates. `publish` rebuilds the point rows from the per-view rows
  in held-out order, runs `row_means` (order and finiteness), requires agreement with the saved
  `mean_ds8_point` within 1e-10 and replaces it (300-307), and writes `groups_ds8_point` over all
  nine conditions into `comparison.json` and `RESULT.json` (336-342, 349, 506). `gates` computes
  the written rule under both operators for every treatment; for `nb_ds4` the per-seed pass is
  primary pass and point pass, the per-seed reverse is primary reverse and point reverse, and the
  verdict is pass iff every seed passes, reject iff every seed reverses, else inconclusive
  (122-138), which is `decision_policy.per_arm` exactly; both sub-results with their deltas are
  stored. The five downscale-8 treatments keep the primary rule with the point rule stored as
  `ds8_point_descriptive` (140). The descriptive pairs are reported under both operators
  (141-144). `decision_text` reads only each arm's `verdict`, so the once-only decision string is
  fixed by the code, not by a later choice of operator.
- Required change 1, task text and tests. `scoring.evaluation_render` names both operators,
  classic mode without antialiasing, the 0.3 px^2 dilation in the render's pixel units without
  opacity compensation, and states that `nb_ds4` is the only arm unaffected. `claim_boundary`
  states both operators, qualifies the site coincidence as "in position only", says `nb_ds4` is
  gated under both and that primary-operator numbers are not comparable to earlier tasks.
  `primary_metrics` is unchanged; `secondary_metrics` has five entries mirroring the primary
  schema. The new test covers pass under both (pass), pass under the primary only (inconclusive,
  with the sub-verdicts pass and inconclusive: base 24.0, point 24.0, delta zero, so neither pass
  nor reverse, which is exactly the operator-dependent branch), and a downscale-8 arm staying on
  the primary gate with the point rule descriptive; the existing test keeps `nb_ds4` reject under
  both; the CUDA-skipped box test is kept.
- Required change 2, disclosures. (a) `nb_30k_d6` purpose carries the means-LR factors 0.398 at
  6000, 0.293 at 8000, 0.1 at 15000, 0.01 at 30000 versus 0.0316 and 0.01 in `nb_base`; (b)
  `nb_30k_d6_lr` purpose states identity with `nb_base` through 8000, then 22000 steps with the
  means LR at 1e-3 by 12000 and 3.2e-8 at 30000, constant rates for the other parameters, no
  densification, pruning or reset after 6000; (c) `nb_ds4` purpose states the resolution
  package: halved physical scale of the 0.3 px^2 dilation and the 11 x 11 SSIM window, doubled
  pixel-unit positional gradients, effectively lower 0.0002 grow threshold, possible 100000 cap;
  (d) the hypothesis ends with the sentence that the densification-window separation is relative
  to RTGS-028's `nb_30k_up` (different seeds and operator), not a within-run comparison; (e)
  `nb_v11` purpose states the 22-view bounds and shared initialization, the frozen 11 views, and
  the roughly doubled per-view sampling. The facts match the round-1 derivations.
- Operator neutrality, re-checked with the second operator in place. Under the primary operator
  the downscale-8 arms render 0.225 px^2 thinner than in training and `nb_ds4` renders with its
  training operator, so the primary favours `nb_ds4`. Under the point operator `nb_ds4` renders
  with a dilation of 1.2 px^2 in its training units and is point-sampled at downscale-8 centres
  against a four-site box reference, so the point operator disfavours `nb_ds4`, and the five
  downscale-8 arms are matched as in RTGS-028. A unanimous pass under both therefore exceeds both
  biases, a unanimous reversal under both is a real loss, and anything in between is frozen as
  inconclusive. The word "bracket" in the scoring text is qualitative (opposite-sign biases of
  unknown size), not a numeric interval.
- New-defect search, minor observations that do not change the verdict: (1) the
  `secondary_metrics` ids are prefixed `ds8_point_*` while the stored keys are the unprefixed
  `METRICS` names under `ds8_point` and `mean_ds8_point`; no code reads the ids, the mapping is
  stated in each `aggregation`, and the contract neither validates nor rejects the block (no
  unknown-key rule; `validate` passes; the `init-run` lock hashes the whole task). (2) The
  `ds8_point_descriptive` entries carry a `verdict` string for the five downscale-8 arms; it is a
  rule outcome, not a verdict, and can only mislead in prose. (3) `RESULT.md`'s table and
  `metrics.json` carry the primary operator only; the point-operator group means and deltas live
  in the once-only `RESULT.json` and in `comparison.json`, which meets the round-1 requirement.
  (4) The archived round-1 file begins with a stray line (a fragment of this reviewer's transmittal
  sentence) before the heading; no lock hashes that file. (5) `status` stays `draft` with a
  pending `protocol_review` rather than `blocked`/`rejected` as round-1 condition 1 asked; that is
  the correct choice under the contract, because a recorded `rejected` review must carry a digest
  equal to the current protocol (364-365) and the protocol changed; the rejection is archived at
  a non-canonical path instead. (6) The Driver re-ran the smoke evaluation with the two-operator
  code on the existing smoke models (training views as stand-ins, no held-out access): the same
  exposure category as the frozen smoke disclosure, not verifiable from tracked state. (7)
  Runtime: `evaluate` is untimed and gains 108 point renders with LPIPS, negligible; the 3600 s
  fit-only limit is unaffected.
- Task record. Driver `Claude-Code-Opus-5.5-driver`, Reviewer label
  `Claude-Code-Fable-5.1-reviewer`, Turn `reviewer`, Status `In review`; the round-1 verdict is
  recorded as "Revision required" with a pointer to the archived artifact; the round-2 handoff
  states the digest and binding this review reproduces. The handoff's GPU test run and smoke
  statements were not verified and were not used as evidence.

Reviewed identities at `9abf11c` (`sha256sum`; `unchanged` means identical to the round-1 table):

| File | SHA-256 |
|---|---|
| `experiments/tasks/20260928_field_only_portfolio_stage_frame00008.json` | `93f9cdc0f98a98a9db5c5eee0388628879b50ba99e5adf07cfd6c7987ff91c88` |
| `experiments/data/20260928_field_only_portfolio_stage_frame00008.json` | `d01254e739961dcb1acc4a9abca6694bb63e319ff179581ebe2ec2ae08706087` (unchanged; the RTGS-025/026/028 seal) |
| `scripts/experiments/20260928_field_only_portfolio_stage_frame00008.py` | `b1da4c62a8be0358bf3bed337257e616ce7f1e2e11f697fd1446ec764b70e2f3` |
| `scripts/experiments/20260928_field_only_portfolio_stage_frame00008_report.py` | `a6e77add18e7834c345fd73a043c90cb7e3fb212fedb9a41b65fe95bccf17db0` |
| `tests/test_field_only_portfolio_protocol.py` | `baffcf3c78fd21dcd67e76d056edfef0c0bf9a063ceb308752789f02971cad91` |
| `tests/test_field_targets.py` | `0a817166c45df13d64cd8e8f83aafbf7656e1bc37ef8bf26176bf6def75a72cc` (unchanged) |
| `src/rtgs/optim/trainer.py` | `e226d01528897fd00dea8c03f345190c9abd3b4cf5cd5e2013967ca5802d39ba` (unchanged) |
| `src/rtgs/optim/strategies.py` | `947a6e02c229c17ee639be98aede565292ccc3aa0303c84f5130b897c2bea91d` (unchanged) |
| `src/rtgs/optim/density.py` | `ad889cd67f7d6e54c7223836dfc27d10555a18c98110f5f465f0df71d352254a` (unchanged) |
| `src/rtgs/data/field_targets.py` | `59901d820ef291381c27cb45ba68161cdad1195eff7ea891ae281b60bca9829f` (unchanged) |
| `src/rtgs/render/gsplat_backend.py` | `2c6a5e0029af86c85e45975fcfceb16d68d342458893eba7fd227db4747bf409` (unchanged) |
| `scripts/experiment_contract.py` | `15f129398aabb46a50d435c76fec27b4df6c75e6d08d2e7d78a5a96657b25959` (unchanged) |
| `scripts/check_results_bundle.py` | `0f5e1293a4c6651d1fed80237bd1319f44ed2ca268d81cb27ce87d2e54f2fedf` (unchanged) |
| `pyproject.toml` | `e5b15e72400e8e14b83a26dc64ef8a85ea985642ef9ef079e425326b5e0bc8d9` (unchanged) |
| `scripts/experiments/20260927_color_budget_reset_stage_frame00008.py` | `9de0cc5ca8815a00282053dce7d1fbebf8b3c56d863060bcd846ddbfd3f18ba0` (operator diff base, unchanged) |
| `experiments/reviews/20260928_field_only_portfolio_stage_frame00008_PROTOCOL_REVIEW_V1_REJECTED.md` | `ccf4fc4b3fc1572725b667ac9f90c674c77faaa616346f4fd7ab4356a6b89914` |
| `experiments/reviews/20260928_field_only_portfolio_stage_frame00008_DRIVER_RESPONSE_R1.md` | `c6ce0a46e76ce63ea13466a583618363289463959b2657e79e0662314262772c` |
| `.agents/state/current-task.md` | `3209d758276c374bd69330d247409acc9c5f5d0ab0da6e3ba6a8011000700883` |

Git object ids at `9abf11c` (`git rev-parse HEAD:<path>`):

| Path | Object id |
|---|---|
| `src/rtgs` (tree) | `1cebc4bc92e5e83a904b4fc4394fc0429423bbd2` (identical at `1428175` and `e858611`; every per-file id in the round-1 table therefore still holds) |
| `scripts/experiment_contract.py` | `3d4f251972226e5372dfb5346283c65f2e73e62e` (unchanged) |
| `scripts/check_results_bundle.py` | `4be3439031998a35caf1b6d023085dccee659ceb` (unchanged) |
| `pyproject.toml` | `f3911412950125f290bc65a49818ed6949c820a7` (unchanged) |
| `tests/test_field_targets.py` | `1f522a73de24a3ce91c293ee52459f9592f439cd` (unchanged) |
| `tests/test_field_only_portfolio_protocol.py` | `46423d0aaaf3651aa4add47536c32e4a001bc1c2` |
| `scripts/experiments/20260928_field_only_portfolio_stage_frame00008.py` | `fccd2ee1d2bb3e499f75a39918762d9f23b219e2` |
| `scripts/experiments/20260928_field_only_portfolio_stage_frame00008_report.py` | `831d114f8721a758e89108603a9aae9f8e767971` |
| `experiments/tasks/20260928_field_only_portfolio_stage_frame00008.json` | `6b8cd095e23a6b5c2421ae42c1f742e509f3f849` |
| `experiments/data/20260928_field_only_portfolio_stage_frame00008.json` | `dc22987ddbfefac83d10deae666616451b8cff8c` (unchanged) |
| `experiments/reviews/20260928_field_only_portfolio_stage_frame00008_PROTOCOL_REVIEW_V1_REJECTED.md` | `a2a68303ac9175635d53a7c235895cc5d648cf3b` |
| `experiments/reviews/20260928_field_only_portfolio_stage_frame00008_DRIVER_RESPONSE_R1.md` | `12a24c8075e7fce3634815043596bff1e7c2f5eb` |
| `.agents/state/current-task.md` | `ed4c474892869e5583fd988a6e0cca53ea12a245` |

## Findings

Approved. Both required changes are resolved as specified and nothing else moved. The second
operator is the RTGS-028 operator token for token on the unchanged bound source; it is stored per
view and per cell, recomputed in `publish` under the same order and finiteness rules, and
reported as group means and paired deltas for every arm. The `nb_ds4` gate now requires the
written rule to pass, or to reverse, under both operators in every seed, so the one-sided
dilation bias identified in round 1 can no longer produce a verdict on its own, and a
confirmation inheriting this convention inherits the bracket rather than the bias. The five
downscale-8 treatments keep the primary gate they share fairly with `nb_base`, and their
point-operator numbers restore descriptive comparability with the RTGS-025/026/028 lineage. The
five disclosures now sit in the frozen protocol, so the once-only RESULT boundary will carry them.
Conditions, seeds, split, view subsets, configurations, thresholds and execution order are
byte-identical to the round-1 state; the source binding covers the revised driver, report and
tests and reproduces; the leakage boundaries, coordinator lifecycle and resource protocol are the
ones approved for RTGS-028. The seven minor observations above are presentation, record-keeping
or disclosure matters and are carried as residual conditions.

Residual execution conditions:

1. Recording the approval. Persist this artifact at
   `experiments/reviews/20260928_field_only_portfolio_stage_frame00008_PROTOCOL_REVIEW.md`,
   beginning at the heading; set `protocol_review` to reviewer `Claude-Code-Fable-5.1-reviewer`,
   verdict `approved`, digest `345e57a9...` and that artifact path; set `status: ready`; re-run
   `validate`. Those two fields are outside the digest, so the digest stays `345e57a9...`. Any
   other change to the task, or any change to a bound file, invalidates this approval and requires
   a new digest and review. Recommended, not blocking: trim the stray first line of the archived
   round-1 file so it begins at its heading, leaving the rest byte-identical, and note the
   correction in the handoff.
2. Source state at `init-run`. Clean tree; `src/rtgs` tree `1cebc4bc...`; the contract, bundle
   checker, `pyproject.toml` and `tests/test_field_targets.py` at the object ids above; the
   driver, report and protocol test at `fccd2ee1...`, `831d114f...` and `46423d0a...`; the task
   JSON at `6b8cd095...` before the status and review edit of condition 1 (that edit changes the
   raw file, not the digest or the binding, since the task file is not a binding pattern).
   `init-run` re-verifies the 119-file aggregate `2848f1d6...` fail-closed. The installed torch,
   gsplat (1.5.3 expected) and lpips packages and the LPIPS AlexNet weights are not bound; the
   RESULT must cite `preflight.json` and `environment.json`, and the phrase "the RTGS-028
   operator" holds only if the recorded gsplat version equals RTGS-028's.
3. Runtime versus the frozen 3600 s cell limit. From the RTGS-028 walls on the same GPU (about
   300 s for a `nb_30k_d6` cell; a few hundred to roughly 800 s for a `nb_ds4` cell) the limit
   is adequate only if no other compute process shares the GPU (`resource_protocol.scope` and the
   `preserve_existing_gpu_processes` guard). A time-out after `prepare` began consumes the root;
   do not kill other GPU processes to make room.
4. Disclosures the RESULT and AUDIT must carry: frame 00008 is outcome-exposed; both operators
   with the dilation mechanism, and for every arm the group means and paired deltas under both
   (the point-operator table is in `RESULT.json` under `groups_ds8_point` and in
   `comparison.json`; `RESULT.md` and the results page show the primary operator only, so the
   AUDIT must reproduce the point table); the `secondary_metrics` ids map to the unprefixed keys
   under `ds8_point` and `mean_ds8_point`; the `ds8_point_descriptive.verdict` strings for the
   five downscale-8 arms are descriptive rule outcomes and must not be presented as verdicts or
   used to promote or demote a primary verdict; an operator-dependent `nb_ds4` outcome is
   inconclusive and must be reported as such with both sub-results; "bracket" is qualitative;
   the per-arm means-LR factors; for `nb_ds4` the final and maximum live counts and
   `density_stats.pruned_to_budget` per cell; for `nb_v11` that `train_view_ids` equals the
   frozen half list and `sampled_train_views` lie in 0..10; the gsplat version from
   `preflight.json` as the basis of "no opacity reset in any arm", citing the RTGS-027/028 canary
   evidence since the canary test is not bound here; the pre-review smoke as frozen plus the
   second smoke evaluation pass with the two-operator code on the same smoke models; timings on a
   possibly shared GPU are descriptive; six uncorrected comparisons, any pass a screening signal;
   RTGS-025/026/028 numbers are not comparators in any gate, and cross-task point-operator
   context is descriptive only and only under a matching gsplat version.
5. One `run` invocation; the coordinator refuses a root containing `targets/` or
   `preparation.json`. A failure before `prepare` begins is preserved in `execution_failure.json`
   and disclosed before any re-entry; a failure after `prepare` began consumes the root, no
   sibling root may be created, and a repeat requires a new task id and review.
6. Accepted late aborts that consume the root: the index-parity, packed-alpha equality, sealed
   digest and fit-window checks in `prepare` at either downscale; a CUDA OOM in a downscale-4 or
   30000-step cell; the held-out camera-record checks in `evaluate`; a nonfinite metric under
   either operator.
7. Nothing beyond the 27 registered cells may be trained under this root, and the `nb_base` seed
   9561 preview is the only root-level model.

Optional, not conditions: a future task template could carry the point-operator group means in
`RESULT.md` and `metrics.json` (changing the bound report now would require a new binding and
review); `record_train_metrics` for training-view PSNR as suggested in round 1.

## Protected Actions Not Taken

The reviewer did not run `init-run`, did not execute the driver's `run`, `prepare`,
`initialize`, `fit`, `evaluate` or `selftest` commands directly (the allowed protocol test suite
runs `selftest` as a CPU subprocess with CUDA hidden and `coordinate` on a temporary root whose
guard raises before any write), performed no GPU work (CUDA is not visible in this session), and
did not enter, list or read `.scratch/`, any run root or any smoke directory. No image, mask,
`.rtgsv`, `.npz`, `.ply`, run content or `runs/` file was opened or printed; `validate-data`
hashed sealed bytes opaquely; no data-seal content and no dataset content was read in this round.
No RTGS-025, RTGS-026 or RTGS-028 RESULT or AUDIT record was read in this round; the RTGS-028
figures quoted above come from the archived round-1 review. No run root, RESULT or AUDIT exists
for this task. No file was modified, no run root was created, no checkpoint was selected, and the
owner's task record was not written; the only file written was a copy of the round-1 task JSON
in the session scratchpad outside the repository, used for the key-level comparison. The
sandbox-denied commands listed above touched no protected material and were re-done only in the
allowed forms. This review has no outcome access to the protocol under review.
