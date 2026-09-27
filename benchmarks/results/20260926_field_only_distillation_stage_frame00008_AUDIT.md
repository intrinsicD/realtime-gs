# Independent Results Audit

- Task ID: `20260926_field_only_distillation_stage_frame00008`
- Protocol SHA-256: `05048698cc34656fd562c8f60d534037745d99acf91b9cac6712a5f314b23f61`
- Reviewer: `Claude-Code-Fable-5.1-reviewer` (claude-fable-5-1, effort max, no nested agents, no fallback model)
- Outcome Access: `full post-run results audit`
- Verdict: `accepted_with_limits`
- Self-reviewed: no. The same label wrote the prospective review; this is the separate post-run audit over the raw bundle, and every producer narrative was treated as a claim.

## Scope

The audit covers the single protected run under `runs/20260926_field_only_distillation_stage_frame00008/`, its producer records `benchmarks/results/20260926_field_only_distillation_stage_frame00008_RESULT.{md,json}`, and the Driver's handoff text in `.agents/state/current-task.md`. It read only the 126 files listed in the authorized payload manifest (SHA-256 `ee2ab3d7…5d95ab`, matching the authorization record), repository source, docs and skills, the protocol and review records, and `.scratch/…/launch_context.json`. Raw photographs, masks, `.rtgsv`, `.npz`, `.ply`, `targets/`, `renders/`, GIFs and logs were not opened.

The verdict concerns the soundness of the evidence and what it permits. It does not upgrade the scientific outcome, which is a clean G0 failure with H1 and H2 inconclusive by the frozen policy.

## Chronology, single invocation, and binding

**Chronology.** The approval commit `bc28af3` is dated 2026-09-27 05:49:29 UTC. `init-run` locked the task at 06:01:43.75 UTC (`task.lock.json`, `development: false`, `source_dirty: false`). The Driver's launch note records 06:02:21 UTC. Stage intervals in run seconds after the lock form one continuous timeline.

| Stage | Start (s) | End (s) |
|---|---:|---:|
| prepare | 42.0 | 109.6 |
| initialize | 110.9 | 114.0 |
| 21 fitting cells, frozen order | 115.8 | 3389.1 |
| 21 evaluations | 3401.9 | about 3455.5 |
| `run_receipt.json` finished | 3456.6 (06:59:20.31 UTC) | |

Fit intervals are sequential with no overlap and gaps of at most 2 s; every evaluation started after the last fit ended.

**Single invocation.** `execution_failure.json` and `attempts/` do not exist and `run_receipt.json` reads `completed`. A failed first invocation would have left `execution_failure.json` (nothing deletes it); a re-invocation after `prepare` began fails closed at `targets/` creation and would have overwritten the receipt with `failed`. The continuous timeline excludes a restart. Condition 3 of the approved review is satisfied; no re-invocation was disclosed and none is evidenced. Logs were not inspected (outside the manifest); the receipt's `exit_code: 0` is written by the report module, not measured from the process.

**Source binding.** `source_snapshot/manifest.json` lists 120 files (the 118 bound files plus the data seal and the review artifact). Every listed SHA-256 and byte size equals the live tree, and all 118 archived copies equal their manifest entries. `git_diff_sha256` equals the SHA-256 of empty bytes, `tracked.diff` and `git-status.txt` are 0 bytes, and `source_commit` is `bc28af3`. Git object ids at `bc28af3` for `src/rtgs` (`641dbc4e…`), `scripts/experiment_contract.py`, `scripts/check_results_bundle.py`, `pyproject.toml`, the driver, the report module and both test files equal the ids pinned at `f5ddaa2` in the approved review; the diff `f5ddaa2..bc28af3` touches only the task record, the review artifact, and the task JSON's `status` and `protocol_review` fields (condition 1). The bound paths in the working tree at audit time are unchanged against `HEAD`. The lock's task, review-artifact and data-seal digests equal the live files. `review-digest` reproduces the protocol digest; `validate` and `validate-data` print OK; the focused CPU tests report 16 passed and 1 skipped (CUDA parity self-skip). The `aggregate_sha256` in the task was again not recomputed (outside the allowed command set); per-file equality of all 118 bound files and the object ids bind the executed source more strongly than the aggregate would.

## Input boundary

- `input_integrity_entry.json` and `_exit.json`: 137 sealed files hashed, all match at entry and exit; exit `split_sha256` recorded. The 137 equal the seal (calibration, 26 photographs, 26 masks, 26 no_boundary views, 2 manifests) plus the two other family directories.
- `prepare`: 242 dataset opens over exactly the 22 training stems, no held-out stem, no denial. Packed alpha equals the thresholded source mask in all 22 views (0 mismatched pixels). 0 mask pixels outside any family's fit window. Every field view's recorded source photograph and mask digest equals the seal. Decoder parity maxima: index vs reference 2.4e-7 (limit 1e-5), CUDA vs index 2.2e-6 (limit 2e-5). CPU self-test passed with 5 denied probes.
- `initialize`: zero dataset opens, zero denials.
- `fit`: all 21 receipts show zero dataset opens and zero denials; field workers read cached targets only.
- `evaluate`: 36 opens, all of the four held-out views (photograph, mask, and the three families' compact views for the field-consistency diagnostic), after every cell receipt read `completed`. Held-out data reached nothing but reporting.

## Cells, initialization, and configuration

All 21 cells: `status: completed`, `executed_iterations: 8000`, 8000 finite losses, `stop_reason: max_iterations`, no in-training PSNR evaluation (`psnr: []`). Each `effective_config.json` equals the frozen resolved configuration for its objective and seed byte for byte, names the frozen 22 training views, the preparation bounds, and `masks_supplied_to_trainer` true except for `nb_rand_pm`. Each `initial_sha256` equals the digest in `initialization.json` for the correct key; the five random cells of a seed share one byte-identical initialization. `final_count` equals the last `n_gaussians` entry.

Initialization: 20000 random points per seed (isotropic scale 0.0375, 0.0372, 0.0372), visual hull 7881 shell voxels of 18600 occupied, voxel 0.01749, bounds extent 2.239. The protocol text quotes 7740 shell voxels from the pre-review smoke; the official count is 7881 and supersedes it.

Two mechanism facts from `history.json`, descriptive only:

- The first density event at iteration 600 prunes every random initialization from 20000 to 1084 to 1154 live Gaussians (950 to 977 in the premultiplied cells) and the model regrows from those survivors. Hull cells grow from 7881 to about 8340 at the same step. H3 therefore compares a mostly discarded random start against a retained hull start.
- The six masked cells of a seed share an identical 8000-step training-view schedule; `nb_rand_pm` does not (about 4 percent of positions coincide), because random-background draws consume the same generator. H4 therefore also differs in view order and in final capacity.

Final Gaussian counts and per-process resources (contended GPU, descriptive only):

| Condition | Final count (3 seeds) | Wall s | Peak allocated MB | Peak reserved MB |
|---|---|---|---|---|
| nb_rand_ms | 42863 to 43664 | 147.6 to 151.0 | 137.9 to 139.4 | 202 to 204 |
| nb_hull_ms | 46535 to 46906 | 151.0 to 152.7 | 137.5 to 138.8 | 202 to 204 |
| nb_rand_pm | 18932 to 19609 | 185.8 to 186.8 | 101.1 to 102.0 | 144 to 146 |
| mc_rand_ms | 32389 to 33846 | 142.5 to 144.7 | 124.0 to 125.7 | 160 to 178 |
| gi_rand_ms | 40105 to 41296 | 145.7 to 146.6 | 133.3 to 135.1 | 200 to 202 |
| ph_rand_ms | 46197 to 48167 | 148.3 to 149.6 | 140.7 to 143.3 | 204 to 206 |
| ph_hull_ms | 50514 to 51683 | 152.8 to 153.5 | 143.2 to 145.1 | 206 to 224 |

Host `ru_maxrss` 1.66 to 1.71 GB per worker; no OOM and no allocation retries. Environment: Python 3.12.9, torch 2.9.0, gsplat 1.5.3, lpips 0.1.4, rtgs 0.1.0, CUDA 12.8, RTX 3050.

## Independent recomputation

Per-view rows of all 21 `evaluation.json` files cover the four frozen held-out views in order with finite values; the four masks are identical across cells (foreground 27927, 20427, 30518, 18830 pixels). Per-cell means, group means in `comparison.json`, `metrics.json`, `RESULT.json` and `RESULT.md`, and every producer paired delta reproduce to 1e-9. The RESULT table is correct.

Per-seed cell means:

| Cell | Foreground PSNR dB | Crop LPIPS | Outside alpha | Floater fraction | Interior alpha |
|---|---:|---:|---:|---:|---:|
| nb_rand_ms/9261 | 23.4559 | 0.1467 | 0.001222 | 0.000898 | 0.99916 |
| nb_rand_ms/9262 | 23.5920 | 0.1461 | 0.001100 | 0.000786 | 0.99903 |
| nb_rand_ms/9263 | 23.4281 | 0.1445 | 0.001093 | 0.000802 | 0.99912 |
| nb_hull_ms/9261 | 23.4480 | 0.1471 | 0.000978 | 0.000809 | 0.99928 |
| nb_hull_ms/9262 | 23.5055 | 0.1445 | 0.001045 | 0.000848 | 0.99929 |
| nb_hull_ms/9263 | 23.5108 | 0.1457 | 0.001038 | 0.000875 | 0.99933 |
| nb_rand_pm/9261 | 22.0991 | 0.1694 | 0.000462 | 0.000427 | 0.98247 |
| nb_rand_pm/9262 | 22.0149 | 0.1705 | 0.000413 | 0.000393 | 0.97963 |
| nb_rand_pm/9263 | 22.0881 | 0.1707 | 0.000434 | 0.000406 | 0.98126 |
| mc_rand_ms/9261 | 23.2591 | 0.1398 | 0.000969 | 0.000755 | 0.99922 |
| mc_rand_ms/9262 | 23.4386 | 0.1406 | 0.001043 | 0.000797 | 0.99914 |
| mc_rand_ms/9263 | 23.3534 | 0.1407 | 0.000997 | 0.000766 | 0.99916 |
| gi_rand_ms/9261 | 23.5659 | 0.1320 | 0.001001 | 0.000793 | 0.99928 |
| gi_rand_ms/9262 | 23.6467 | 0.1304 | 0.001029 | 0.000754 | 0.99904 |
| gi_rand_ms/9263 | 23.6424 | 0.1334 | 0.001020 | 0.000781 | 0.99917 |
| ph_rand_ms/9261 | 23.8868 | 0.1279 | 0.001100 | 0.000864 | 0.99914 |
| ph_rand_ms/9262 | 23.7698 | 0.1294 | 0.001092 | 0.000812 | 0.99912 |
| ph_rand_ms/9263 | 23.8367 | 0.1294 | 0.001042 | 0.000773 | 0.99908 |
| ph_hull_ms/9261 | 23.9008 | 0.1248 | 0.000931 | 0.000802 | 0.99933 |
| ph_hull_ms/9262 | 23.9689 | 0.1239 | 0.000974 | 0.000782 | 0.99937 |
| ph_hull_ms/9263 | 23.8958 | 0.1250 | 0.000908 | 0.000740 | 0.99933 |

Official gates, recomputed in the frozen written form:

| Seed | ph_rand_ms PSNR | G0 (floor 24.0) | nb−ph PSNR | nb−ph LPIPS | nb−ph outside | H1 inequalities | nb−mc PSNR | nb−mc outside | H2 pass (≥ +0.2) | H2 reverse |
|---|---:|---|---:|---:|---:|---|---:|---:|---|---|
| 9261 | 23.8868 | fail, short 0.113 | −0.4309 | +0.0188 | +0.000122 | hold | +0.1968 | +0.000254 | no | no |
| 9262 | 23.7698 | fail, short 0.230 | −0.1779 | +0.0167 | +0.000008 | hold | +0.1534 | +0.000057 | no | no |
| 9263 | 23.8367 | fail, short 0.163 | −0.4087 | +0.0151 | +0.000051 | hold | +0.0746 | +0.000096 | no | no |

- **G0**: numeric floor fails in every seed. Verdict fail. The producer applied the frozen consequence verbatim: H1 and H2 inconclusive. The decision sentence in the RESULT, including "the trainer/initializer must be resolved first", is the text frozen in the bound report module before execution; it is not a post-hoc reinterpretation, but it is an interpretation and is narrowed below.
- **H1**: all three per-seed inequalities hold, and the frozen verdict is nonetheless inconclusive because the reference failed its own adequacy floor. Matching a sub-floor reference supports no absolute quality statement.
- **H2**: inconclusive independently of G0. The uncontained family leads in every seed but below the frozen 0.2 dB margin in every seed, by 0.003 dB in seed 9261 and by 0.125 dB in seed 9263; the reverse condition fails everywhere. This is not a near miss overall.

Descriptive deltas (no pass or fail):

| Comparison | PSNR dB per seed | LPIPS per seed | Outside alpha per seed |
|---|---|---|---|
| H3 field: nb_rand_ms − nb_hull_ms | +0.008, +0.087, −0.083 | −0.0004, +0.0017, −0.0012 | +0.000245, +0.000056, +0.000055 |
| H3 photo: ph_rand_ms − ph_hull_ms | −0.014, −0.199, −0.059 | +0.0031, +0.0055, +0.0044 | +0.000169, +0.000119, +0.000134 |
| H4: nb_rand_ms − nb_rand_pm | +1.357, +1.577, +1.340 | −0.0227, −0.0244, −0.0262 | +0.000760, +0.000687, +0.000659 |
| gi_rand_ms − nb_rand_ms | +0.110, +0.055, +0.214 | −0.0147, −0.0157, −0.0110 | −0.000221, −0.000072, −0.000073 |

H4 also shows floater fraction +0.00047, +0.00039, +0.00040 and interior alpha +0.0167, +0.0194, +0.0179 for the masked objective. The frozen H4 statement expected the silhouette objective to lower outside-mask alpha relative to premultiplied black; the observed direction is the opposite in all three seeds. The premultiplied objective has fewer floaters and lower outside alpha but 1.3 to 1.6 dB lower foreground PSNR, worse LPIPS, and about 2 percent lower interior coverage, with half the final Gaussian count and a different view schedule. The RESULT records the deltas but is silent on this direction; it must be stated wherever H4 is reported.

Per-view structure, mean over seeds: C0001 is the hardest view for every arm (20.7 to 21.1 dB, outside alpha about 0.003, LPIPS 0.18 to 0.22) and C1002 the second (22.4 to 22.9 dB, mask box spanning the full image width); C0018 and C0029 sit at 24.9 to 26.2 dB. The photograph reference exceeds the floor on the two easy views and misses it on the two hard ones; the gate uses the mean, and no view-level reinterpretation is permitted.

Teacher qualification (report only, training views): no_boundary 30.79 dB (24.90 to 37.68), gaussianimage 29.17 dB, mask_contained 26.22 dB (24.22 to 31.65). Held-out field-consistency diagnostic of each student against its own family: nb 24.5 to 24.7 dB, gi 25.6 to 25.8, mc 25.9 to 26.1, pm 23.1 to 23.2.

## B2 boundary band and the H2 family confound

At training pixels whose packed-alpha fraction is 0.25, 0.5 or 0.75, the four-site box average contains outside-mask sites, and the Trainer supervises `target*mask + random_background*(1-mask)` with L1 weight `0.1 + 0.9*mask`, so outside-mask target colour enters in proportion to the fraction (trainer lines 690 to 735 at the bound source, verified). In that band the families differ: the mask-contained teacher is black at outside sites, the uncontained teachers extrapolate foreground colour, and photographs contain the room background. The held-out score binarizes the mask and scores colour inside only, so the band acts through learned edge colour and alpha. The band population is not recorded in any readable artifact and the effect is unquantified.

H2 therefore compares teacher families that differ jointly in count (no_boundary 5000 to 8592 per view, mean 6833, versus 11000), topology-schedule outcome, containment, band colour, and inside-mask teacher fidelity (4.6 dB gap on training views). The mask-contained students reproduce their own teacher better on held-out views (26.0 versus 24.6 dB) while scoring lower against photographs. The uncontained family also has slightly more outside-mask alpha than the contained family in all three seeds, within tolerance and consistent with, but not evidence for, band extrapolation. No component can be attributed. This confound would have applied even to a pass.

## Visual audit

All 22 dome-derived contact sheets were viewed (21 cells plus the root sheet, which is byte-identical to `nb_rand_ms/9261`). Each sheet shows the four held-out views as rows of reference (masked photograph), initial, final, and error times four, at a display of 2000 by 450 from 5328 by 1200; no full-resolution crops were generated because the payload allowlist does not cover derived images.

- **G0 visual component.** In all three `ph_rand_ms` seeds, and in every other cell, the final render shows a recognizable and complete lying figure in every held-out view: the head, arms and dress are present, without missing limbs or holes. Interior alpha of 0.999 corroborates completeness. The visual component of G0 is satisfied; G0 still fails on the numeric floor.
- **Floaters and halos.** Small isolated speckles are visible in the black background in most cells, most clearly around C0001 and C1002, consistent with floater fractions of 0.0008 to 0.003 per view. The error panels show bright rims along the silhouette in every arm and a consistent head-region colour error in C0001. The premultiplied cells look marginally cleaner, matching their lower outside alpha; the difference is at the limit of this resolution.
- **Between-arm differences** are not resolvable visually at this scale; quantitative decisions rest on the exact metrics, as the protocol requires.
- Hull cells show a coloured silhouette as the initial panel; random cells show the grey ball, as specified.

## Metric semantics, resources, and environment

`mask_scores` computes squared RGB error only over the binarized held-out mask, LPIPS on mask-multiplied crops padded by 8 px, outside alpha and floater fraction outside the 3-pixel dilated mask, and interior alpha inside the 3-pixel eroded mask, from a black-background gsplat render of the saved final model; the report requires exact view coverage and rejects non-finite values. All match the frozen definitions. No colour is evaluated outside the mask.

The GPU was shared at launch with an unrelated process (about 3.7 GB) and a pytest process. All wall-clock values, including the stage-runtime chart and the 25 percent longer premultiplied cells (full-frame SSIM instead of the mask crop), are contended and non-decisional; the protocol and `resource_receipt.json` already declare `performance_inference: false`. Per-process peak memory is a valid descriptive receipt, includes cache loading and the initial upload, and excludes the coordinator's own CUDA context. The contention is recorded only in `.scratch/…/launch_context.json` and the task record; this audit carries it into the bundle.

## Claim dispositions

| # | Statement (source) | Disposition | Basis and permitted wording |
|---|---|---|---|
| 1 | 21 paired development cells completed (RESULT) | confirm | Receipts, histories and configs verified for every cell. |
| 2 | Group-mean table, 7 conditions by 5 metrics (RESULT) | confirm, bounded | Reproduced from per-view rows to 1e-9; development screen on an exposed frame, one split, three seeds, downscale 8, 8000 steps. |
| 3 | Photograph reference failed its frozen gate; H1 and H2 inconclusive (RESULT) | confirm | G0 fails in all seeds by 0.11 to 0.23 dB; policy applied verbatim; visual component satisfied but irrelevant to the numeric failure. |
| 4 | "the trainer/initializer must be resolved first" (RESULT decision text) | narrow | Frozen wording, but it is an interpretation. Supported: the photograph-supervised reference under this trainer, budget and resolution is below the floor, so the reference protocol, not the field-only path, is the binding limitation. Which component (schedule, budget, downscale, initializer, or the floor itself) is unresolved. |
| 5 | "Numerically, every H1 per-seed inequality holds" (handoff) | narrow | May be recorded only as a descriptive observation with the G0 caveat: nb_rand_ms was within 0.18 to 0.43 dB, 0.015 to 0.019 LPIPS and 0.00013 outside alpha of a reference that itself fell short of its adequacy floor. It may not be worded as "field-only matches photographs" in README, docs or claims. |
| 6 | "H2 foreground margin 0.07 to 0.20 dB, below 0.2 dB in all seeds" (handoff) | confirm, descriptive | Inconclusive independently of G0; direction favours the uncontained family in all seeds; fully confounded (count, topology, band colour, teacher fidelity). No family selection follows. |
| 7 | H3: random not worse than the hull by more than 0.5 dB (task, descriptive) | confirm, descriptive | Absolute deltas at most 0.2 dB in all six pairs; random has slightly more outside alpha in all six. Mechanism: the random start is pruned to about 1100 points at step 600, so this is not a test of initialization quality. |
| 8 | H4: the silhouette objective lowers outside-mask alpha relative to premultiplied black (task, descriptive) | retire the stated direction | Opposite in all three seeds; premultiplied black has lower outside alpha and floater fraction at the cost of 1.3 to 1.6 dB PSNR, worse LPIPS and lower coverage; confounded by view schedule and capacity. Report as a refuted descriptive expectation. |
| 9 | gaussianimage versus no_boundary (report, descriptive) | confirm, descriptive | gi leads by 0.05 to 0.21 dB and 0.011 to 0.016 LPIPS with lower outside alpha in all seeds; non-gating. |
| 10 | Count confound 5000 to 8592 versus 11000; smoke disclosure (claim boundary) | confirm | Verified from preparation records; the hull count is 7881 officially, not the 7740 of the smoke. |
| 11 | No default, SOTA, generalization, geometry or speed claim (claim boundary) | confirm | Additionally: no timing statement of any kind, the GPU was contended. |
| 12 | One `run` invocation, exit 0, 06:01:43 to 06:59:20 UTC, 21 of 21 cells (handoff) | confirm | Receipts, fail-closed re-entry semantics and a continuous timeline; logs not inspected; exit code synthesized by the report module. |
| 13 | Peak cell memory about 0.2 GB (handoff) | confirm, descriptive | Per-process peak reserved 144 to 224 MB, allocated 101 to 145 MB. |
| 14 | Colour scored only inside held-out masks; floaters from outside alpha (RESULT notes) | confirm | Code verified. |
| 15 | Field arms are field plus silhouette, not Gaussian-only (RESULT notes) | confirm | Mask-derived packed alpha supervises silhouette and hull. |
| 16 | Teacher training-view fidelity 30.79, 26.22, 29.17 dB (comparison) | confirm, report only | Reproduced from per-view rows. |
| 17 | "Report: runs/…/index.html" (RESULT) | unresolved | `index.html`, `README.md`, `manifest.json` and `viewer_smoke.json` do not exist; render, viewer smoke, `check-run` and the bundle gate are pending. |
| 18 | Consequence: pass keeps the main path and selects the uncontained family (task) | not triggered | No selection or promotion follows from this run. |

## Limitations

- Development screen on frame 00008, previously outcome-exposed, one split, three seeds, one budget, downscale 8; nothing generalizes.
- G0 failed, so H1 and H2 carry no claim; H3 and H4 are descriptive and single-frame.
- The visual audit used downscaled contact sheets only; between-arm visual differences and fine halos are unjudged.
- Timing is contended and non-decisional; memory is per-process and descriptive.
- The source-binding aggregate was not recomputed; binding rests on per-file digests and git object ids.
- The boundary-band population is unquantified; the teacher-family confound is unresolvable within this design.
- Logs, targets, renders and models were not inspected; the audit relies on receipts and checksums.

## Commands executed by the reviewer

```text
git status --short; git rev-parse HEAD; git log --oneline -12; wc -c <run JSON records>
git rev-parse bc28af3:<8 bound paths>; git rev-parse f5ddaa2:<8 bound paths>; git rev-parse HEAD:src/rtgs
git diff --stat f5ddaa2 bc28af3; git log -6 --format='%h %ci %s'
git diff --stat HEAD -- <bound paths>; git status --ignored --short -- <bound paths> | grep -v __pycache__
git diff f5ddaa2 bc28af3 -- experiments/tasks/20260926_field_only_distillation_stage_frame00008.json
.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260926_field_only_distillation_stage_frame00008.json
.venv/bin/python scripts/experiment_contract.py validate
.venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/20260926_field_only_distillation_stage_frame00008.json
.venv/bin/python -m pytest -q tests/test_field_only_distillation.py tests/test_field_targets.py   (16 passed, 1 skipped)
.venv/bin/python -c <six read-only recomputation scripts over the manifest JSON files: payload/snapshot/lock hashes and run-root existence; group means, paired deltas and gates; per-cell receipts, histories and configs; preparation, training-history, resource, metrics and seal summaries; density and schedule extraction; view-schedule sharing>
Read tool: the manifest records, driver and report sources, trainer objective (lines 676 to 745), visualize.py (lines 17 to 80), experiment_contract.py and gsplat_backend.py greps, launch_context.json, and the 22 preview PNGs
```

Denied by the sandbox and re-issued in `-c` form: four heredoc `python -` invocations. No other command form was attempted.

## Protected actions not taken

No driver command (`run`, `prepare`, `initialize`, `fit`, `evaluate`, `selftest`), no `init-run`, `render`, `check-run` or `check_results_bundle.py`, no GPU work, no `verify.sh` (outside the allowed set). No raw photograph, mask, `.rtgsv`, `.npz`, `.ply`, GIF, log, target, render or scratch smoke directory was opened; no derived image was written. No file was modified, no evidence was overwritten, no default or claim was changed.

## Remaining handoff checks for the Driver

1. Persist this audit verbatim as `..._AUDIT.md` and the JSON block as `..._AUDIT.json`; leave both RESULT files untouched.
2. Copy `launch_context.json` into the run root, or cite it in the experiment-log entry, so the contention travels with the bundle; then run `render`, the browser viewer smoke, `render` again, `check-run`, and `scripts/check_results_bundle.py`.
3. Append the `docs/EXPERIMENTS.md` entry with: G0 fail in all seeds, H1 and H2 inconclusive, H4 opposite direction, the H3 prune mechanism, teacher fidelity, hull count 7881, and the contention note.
4. ARA: stage the observations; H1 and H2 rows as `untested` with the G0 falsification record; the H4 direction as `refuted (descriptive, single frame)`; bind all to `RESULT.json` and this audit.
5. Register the next task before any rerun: lift the photograph reference above the frozen floor (budget, schedule, resolution or initializer) without touching the floor, then retest H1 and H2 with paired seeds; treat frame 00008 as exposed.
6. Have a human inspect the full-resolution contact sheets and the viewer for halos and floaters; this audit could not.
7. Record the Reviewer verdict in the task record with distinct labels; maturity stays development, no default change.

```json
{
  "schema_version": 1,
  "task_id": "20260926_field_only_distillation_stage_frame00008",
  "reviewer": "Claude-Code-Fable-5.1-reviewer",
  "model": "claude-fable-5-1",
  "effort": "max",
  "self_reviewed": false,
  "verdict": "accepted_with_limits",
  "protocol_sha256": "05048698cc34656fd562c8f60d534037745d99acf91b9cac6712a5f314b23f61",
  "outcome_access": "full post-run results audit",
  "audit_date_utc": "2026-09-27",
  "payload_manifest_sha256": "ee2ab3d7547ae86093d2e3385b45fd99c6a45754dcad0b9e05908e25065d95ab",
  "payload_items_verified": 126,
  "source_binding": {
    "source_commit": "bc28af326babde2bd49eae7dc71bdedea9ba3db0",
    "snapshot_files": 120,
    "snapshot_live_drift": 0,
    "snapshot_copy_drift": 0,
    "git_diff_sha256_is_empty_diff": true,
    "src_rtgs_tree_object_id": "641dbc4ee15b2c9ff697c2b3e2f83fb25135c195",
    "bound_object_ids_equal_reviewed_f5ddaa2": true,
    "approval_commit_changed_only_status_and_protocol_review": true,
    "task_sha256": "27afde3837e3da01ef9e01cd07348afb7694842b12edea267bb2316a2bef1335",
    "review_artifact_sha256": "ef2b5e075de41de59bb68bae621137c86c2d197e2121b2d35b11307be0a8e6be",
    "data_seal_sha256": "d01254e739961dcb1acc4a9abca6694bb63e319ff179581ebe2ec2ae08706087",
    "review_digest_reproduced": true,
    "validate_ok": true,
    "validate_data_ok": true,
    "focused_tests": "16 passed, 1 skipped",
    "aggregate_sha256_recomputed": false
  },
  "run_chronology": {
    "approval_commit_utc": "2026-09-27T05:49:29Z",
    "lock_started_at_utc": "2026-09-27T06:01:43.754696+00:00",
    "launch_note_utc": "2026-09-27T06:02:21Z",
    "finished_at_utc": "2026-09-27T06:59:20.311491+00:00",
    "development_lock": false,
    "source_dirty": false,
    "single_invocation": true,
    "execution_failure_json_exists": false,
    "attempts_dir_exists": false,
    "stage_intervals_run_seconds": {
      "prepare": [42.0, 109.6],
      "initialize": [110.9, 114.0],
      "fit_cells": [115.8, 3389.1],
      "evaluate": [3401.9, 3455.5]
    },
    "fit_intervals_sequential_non_overlapping": true
  },
  "input_boundary": {
    "sealed_files_checked_entry_exit": 137,
    "all_match": true,
    "prepare_opens": 242,
    "prepare_stems": 22,
    "prepare_heldout_opens": 0,
    "prepare_denied": 0,
    "packed_alpha_mismatch_pixels": 0,
    "mask_pixels_outside_fit_window": 0,
    "source_digests_match_seal": true,
    "parity_max_index_vs_reference": 2.38e-07,
    "parity_max_cuda_vs_index": 2.21e-06,
    "initialize_opens": 0,
    "fit_opens_all_cells": 0,
    "fit_denied_all_cells": 0,
    "evaluate_opens": 36,
    "evaluate_denied": 0,
    "heldout_first_opened_in_evaluate_after_all_receipts": true
  },
  "cells": {
    "count": 21,
    "all_completed": true,
    "all_executed_iterations_8000": true,
    "all_losses_finite": true,
    "all_effective_configs_equal_frozen": true,
    "all_initial_digests_match_initialization_json": true,
    "random_init_points": 20000,
    "hull_shell_voxels": 7881,
    "hull_shell_voxels_in_protocol_smoke_text": 7740,
    "first_density_event_prunes_random_init_to": "1084 to 1154 (950 to 977 premultiplied)",
    "masked_cells_share_view_schedule_per_seed": true,
    "premultiplied_cells_share_view_schedule": false,
    "final_count_ranges": {
      "nb_rand_ms": [42863, 43664],
      "nb_hull_ms": [46535, 46906],
      "nb_rand_pm": [18932, 19609],
      "mc_rand_ms": [32389, 33846],
      "gi_rand_ms": [40105, 41296],
      "ph_rand_ms": [46197, 48167],
      "ph_hull_ms": [50514, 51683]
    }
  },
  "official_gates": {
    "recomputed_from_per_view_rows": true,
    "producer_values_reproduced_to": 1e-09,
    "g0_photo_reference": {
      "floor_db": 24.0,
      "per_seed_ph_rand_ms_foreground_psnr": {"9261": 23.886778, "9262": 23.769848, "9263": 23.836728},
      "shortfall_db": {"9261": 0.113, "9262": 0.230, "9263": 0.163},
      "numeric_pass": false,
      "visual_recognizable_complete_subject": true,
      "verdict": "fail"
    },
    "h1_main_path": {
      "verdict": "inconclusive",
      "reason": "G0 failed; frozen policy",
      "all_seed_inequalities_hold": true,
      "per_seed_delta_nb_minus_ph": {
        "9261": {"foreground_psnr": -0.4309, "crop_lpips": 0.0188, "outside_alpha_mass": 0.000122},
        "9262": {"foreground_psnr": -0.1779, "crop_lpips": 0.0167, "outside_alpha_mass": 0.000008},
        "9263": {"foreground_psnr": -0.4087, "crop_lpips": 0.0151, "outside_alpha_mass": 0.000051}
      }
    },
    "h2_teacher_containment": {
      "verdict": "inconclusive",
      "inconclusive_independent_of_g0": true,
      "per_seed_delta_nb_minus_mc": {
        "9261": {"foreground_psnr": 0.1968, "crop_lpips": 0.0069, "outside_alpha_mass": 0.000254, "pass": false, "reverse": false},
        "9262": {"foreground_psnr": 0.1534, "crop_lpips": 0.0055, "outside_alpha_mass": 0.000057, "pass": false, "reverse": false},
        "9263": {"foreground_psnr": 0.0746, "crop_lpips": 0.0037, "outside_alpha_mass": 0.000096, "pass": false, "reverse": false}
      },
      "confounds_named": ["per-view count 5000-8592 vs 11000", "topology schedule outcome", "boundary-band colour (B2)", "teacher inside-mask fidelity gap 4.6 dB"]
    },
    "h3_initializer_descriptive": {
      "field_nb_rand_minus_nb_hull_psnr": {"9261": 0.0078, "9262": 0.0865, "9263": -0.0828},
      "photo_ph_rand_minus_ph_hull_psnr": {"9261": -0.0140, "9262": -0.1991, "9263": -0.0591},
      "within_0_5_db": true,
      "note": "random start pruned to about 1100 points at step 600; not a test of initialization quality"
    },
    "h4_objective_descriptive": {
      "nb_rand_ms_minus_nb_rand_pm": {
        "9261": {"foreground_psnr": 1.3567, "crop_lpips": -0.0227, "outside_alpha_mass": 0.000760, "floater_fraction": 0.000471, "interior_alpha": 0.0167},
        "9262": {"foreground_psnr": 1.5771, "crop_lpips": -0.0244, "outside_alpha_mass": 0.000687, "floater_fraction": 0.000393, "interior_alpha": 0.0194},
        "9263": {"foreground_psnr": 1.3400, "crop_lpips": -0.0262, "outside_alpha_mass": 0.000659, "floater_fraction": 0.000397, "interior_alpha": 0.0179}
      },
      "direction_vs_frozen_statement": "opposite for outside_alpha_mass in all three seeds"
    },
    "gaussianimage_vs_no_boundary_descriptive": {
      "gi_minus_nb_psnr": {"9261": 0.1100, "9262": 0.0547, "9263": 0.2144},
      "gi_minus_nb_lpips": {"9261": -0.0147, "9262": -0.0157, "9263": -0.0110}
    },
    "group_means": {
      "nb_rand_ms": {"foreground_psnr": 23.491974, "crop_lpips": 0.145762, "outside_alpha_mass": 0.001138, "floater_fraction": 0.000829, "interior_alpha": 0.999105},
      "nb_hull_ms": {"foreground_psnr": 23.488124, "crop_lpips": 0.145750, "outside_alpha_mass": 0.001020, "floater_fraction": 0.000844, "interior_alpha": 0.999300},
      "nb_rand_pm": {"foreground_psnr": 22.067391, "crop_lpips": 0.170215, "outside_alpha_mass": 0.000436, "floater_fraction": 0.000408, "interior_alpha": 0.981121},
      "mc_rand_ms": {"foreground_psnr": 23.350365, "crop_lpips": 0.140380, "outside_alpha_mass": 0.001003, "floater_fraction": 0.000773, "interior_alpha": 0.999173},
      "gi_rand_ms": {"foreground_psnr": 23.618332, "crop_lpips": 0.131946, "outside_alpha_mass": 0.001016, "floater_fraction": 0.000776, "interior_alpha": 0.999165},
      "ph_rand_ms": {"foreground_psnr": 23.831118, "crop_lpips": 0.128898, "outside_alpha_mass": 0.001078, "floater_fraction": 0.000816, "interior_alpha": 0.999110},
      "ph_hull_ms": {"foreground_psnr": 23.921834, "crop_lpips": 0.124561, "outside_alpha_mass": 0.000938, "floater_fraction": 0.000774, "interior_alpha": 0.999343}
    },
    "teacher_training_view_foreground_psnr": {"no_boundary": 30.788, "mask_contained": 26.217, "gaussianimage": 29.166}
  },
  "claim_dispositions": [
    {"id": 1, "statement": "21 paired development cells completed", "source": "RESULT", "disposition": "confirm"},
    {"id": 2, "statement": "group-mean table, 7 conditions by 5 metrics", "source": "RESULT", "disposition": "confirm", "boundary": "development screen, exposed frame 00008, one split, three seeds, downscale 8, 8000 steps"},
    {"id": 3, "statement": "photograph reference failed its frozen adequacy gate; H1 and H2 inconclusive", "source": "RESULT", "disposition": "confirm"},
    {"id": 4, "statement": "the trainer/initializer must be resolved first", "source": "RESULT decision text (frozen)", "disposition": "narrow", "permitted": "the photograph-supervised reference under this trainer, budget and resolution is below the floor; which component is responsible is unresolved"},
    {"id": 5, "statement": "numerically every H1 per-seed inequality holds", "source": "Driver handoff", "disposition": "narrow", "permitted": "descriptive observation with the G0 caveat only; never 'field-only matches photographs'"},
    {"id": 6, "statement": "H2 foreground margin 0.07-0.20 dB below the frozen 0.2 dB in all seeds", "source": "Driver handoff", "disposition": "confirm", "boundary": "descriptive; inconclusive independent of G0; fully confounded; no family selection"},
    {"id": 7, "statement": "H3 random not worse than the hull by more than 0.5 dB", "source": "task hypothesis (descriptive)", "disposition": "confirm", "boundary": "absolute deltas at most 0.2 dB; random start mostly pruned at step 600"},
    {"id": 8, "statement": "H4 silhouette objective lowers outside-mask alpha relative to premultiplied black", "source": "task hypothesis (descriptive)", "disposition": "retire", "basis": "opposite direction in all three seeds; premultiplied black has lower outside alpha and floaters at 1.3-1.6 dB lower PSNR, worse LPIPS, lower coverage; confounded by schedule and capacity"},
    {"id": 9, "statement": "gaussianimage leads no_boundary in PSNR and LPIPS", "source": "report descriptive", "disposition": "confirm", "boundary": "descriptive, non-gating"},
    {"id": 10, "statement": "count confound 5000-8592 vs 11000; pre-review smoke disclosure", "source": "claim_boundary", "disposition": "confirm", "correction": "official hull shell count 7881, not 7740"},
    {"id": 11, "statement": "no default, SOTA, generalization, physical-geometry or speed claim", "source": "claim_boundary", "disposition": "confirm", "addition": "no timing statement; GPU contended"},
    {"id": 12, "statement": "one run invocation, exit 0, 06:01:43-06:59:20 UTC, 21/21 cells", "source": "Driver handoff", "disposition": "confirm", "boundary": "logs not inspected; exit code synthesized by report module"},
    {"id": 13, "statement": "peak cell memory about 0.2 GB", "source": "Driver handoff", "disposition": "confirm", "boundary": "per-process peak reserved 144-224 MB, allocated 101-145 MB, descriptive"},
    {"id": 14, "statement": "colour scored only inside held-out masks; floaters from alpha outside the 3-px dilated mask", "source": "RESULT notes", "disposition": "confirm"},
    {"id": 15, "statement": "field arms are field-plus-silhouette, not Gaussian-only", "source": "RESULT notes", "disposition": "confirm"},
    {"id": 16, "statement": "teacher training-view fidelity 30.79 / 26.22 / 29.17 dB", "source": "comparison.json", "disposition": "confirm", "boundary": "report only"},
    {"id": 17, "statement": "Report: runs/<task_id>/index.html", "source": "RESULT", "disposition": "unresolved", "basis": "render, viewer smoke, check-run and bundle gate pending"},
    {"id": 18, "statement": "a pass keeps the main path and selects the uncontained teacher family", "source": "task consequence", "disposition": "not triggered"}
  ],
  "visual_disposition": {
    "sheets_viewed": 22,
    "resolution": "contact sheets displayed at 2000x450 from 5328x1200; no crops generated",
    "g0_recognizable_complete_subject": true,
    "floaters_or_halos": "small background speckles in most cells, most visible around C0001 and C1002; bright silhouette rims and a consistent head-region colour error in error panels",
    "between_arm_differences_resolvable": false,
    "conclusion": "visual component of G0 satisfied for ph_rand_ms in all seeds; G0 fails on the numeric floor"
  },
  "resource_disposition": {
    "gpu": "NVIDIA GeForce RTX 3050, shared at launch with pid 3823481 (experiments.latent_agent, about 3.7 GB) and pid 3823530 (pytest)",
    "timing_usable_for_claims": false,
    "memory_usable_as_descriptive_per_process_peak": true,
    "contention_recorded_in_bundle_before_this_audit": false
  },
  "limitations": [
    "development screen on the previously outcome-exposed frame 00008; one split, three seeds, one budget, downscale 8",
    "G0 failed; H1 and H2 carry no claim; H3 and H4 descriptive only",
    "visual audit at contact-sheet resolution only",
    "timing contended and non-decisional; memory per-process and descriptive",
    "source-binding aggregate not recomputed; binding rests on per-file digests and git object ids",
    "boundary-band population unquantified; teacher-family confound unresolvable in this design",
    "logs, targets, renders and models not inspected"
  ],
  "commands_executed_by_reviewer": [
    "git status --short; git rev-parse HEAD; git log --oneline -12; wc -c <run JSON records>",
    "git rev-parse bc28af3:<8 bound paths>; git rev-parse f5ddaa2:<8 bound paths>; git rev-parse HEAD:src/rtgs",
    "git diff --stat f5ddaa2 bc28af3; git log -6 --format='%h %ci %s'",
    "git diff --stat HEAD -- <bound paths>; git status --ignored --short -- <bound paths> | grep -v __pycache__",
    "git diff f5ddaa2 bc28af3 -- experiments/tasks/20260926_field_only_distillation_stage_frame00008.json",
    ".venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260926_field_only_distillation_stage_frame00008.json",
    ".venv/bin/python scripts/experiment_contract.py validate",
    ".venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/20260926_field_only_distillation_stage_frame00008.json",
    ".venv/bin/python -m pytest -q tests/test_field_only_distillation.py tests/test_field_targets.py",
    ".venv/bin/python -c <read-only recomputation: payload/snapshot/lock hashes and run-root existence>",
    ".venv/bin/python -c <read-only recomputation: per-view means, group means, paired deltas, gates>",
    ".venv/bin/python -c <read-only recomputation: 21 cell receipts, histories, effective configs, chronology, resources>",
    ".venv/bin/python -c <read-only summary: preparation, training_history, resource_receipt, metrics, data seal>",
    ".venv/bin/python -c <read-only extraction: n_gaussians trajectories, density events, history fields>",
    ".venv/bin/python -c <read-only check: training-view schedule sharing across paired cells>",
    "Read: manifest records, driver and report sources, trainer.py 676-745, visualize.py 17-80, experiment_contract.py and gsplat_backend.py greps, launch_context.json, 22 preview PNGs"
  ],
  "denied_by_sandbox": ["four heredoc 'python -' invocations, re-issued as python -c with identical content"],
  "not_executed": [
    "driver run/prepare/initialize/fit/evaluate/selftest",
    "init-run, render, check-run, check_results_bundle.py",
    "scripts/verify.sh (outside allowed set)",
    "any GPU work",
    "source-binding aggregate recomputation",
    "opening raw photographs, masks, .rtgsv, .npz, .ply, GIFs, logs, targets/, renders/, or scratch smoke directories",
    "generation of derived images"
  ],
  "remaining_handoff_checks": [
    "persist AUDIT.md and AUDIT.json verbatim; leave RESULT files untouched",
    "carry launch_context.json into the run root or the experiment-log entry, then render, browser viewer smoke, render again, check-run, check_results_bundle.py",
    "append docs/EXPERIMENTS.md: G0 fail all seeds, H1/H2 inconclusive, H4 opposite direction, H3 prune mechanism, teacher fidelity, hull count 7881, contention note",
    "ARA: staging observations; H1/H2 rows untested with G0 falsification record; H4 direction refuted (descriptive, single frame); bind to RESULT.json and this audit",
    "register the next task to lift the photograph reference above the unchanged floor before retesting H1/H2; frame 00008 remains exposed",
    "human inspection of full-resolution contact sheets and the viewer for halos and floaters",
    "record the Reviewer verdict in the task record with distinct labels; maturity development; no default change"
  ]
}
```
