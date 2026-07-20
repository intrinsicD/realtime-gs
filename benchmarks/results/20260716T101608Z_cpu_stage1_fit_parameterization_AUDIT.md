# Independent scientist audit: Stage-1 fit-time appearance parameterization

Reviewer: Codex independent N78 result scientist (`/root/n78_result_scientist`)  
Reviewed at: `2026-07-16T10:38:20Z`  
Verdict: **PASS**

The artifact is valid and claim-admitting only in its preregistered scope: deterministic
CPU-synthetic Stage-1 source-image fitting at 48x48, 150 fixed components, and 120 Adam updates.
The bounded unit-weight 8-parameter arm did not improve the appearance-only curve and was not
non-inferior during joint fitting. The preregistered disposition is therefore to retain the
current 9-parameter fit and close this exact bounded candidate on this setup without tuning.
The audit authorizes no production-default, downstream, real-image, CUDA, runtime, memory, or
compression claim.

## Evidence binding and chronology

| Artifact or source domain | SHA-256 / bound value | Audit |
|---|---|---|
| Preregistration | `d1440fde596667fd59e996113dd4ffa4414e23e8c783a401343d7476f00afb22` | exact |
| Implementation review | `fd3721787296768beb866b73fe603c66351cb934b27ecbfb8dc30498cb96145d` | exact, PASS before seal |
| Implementation seal | `6aeee81f97409f0dfdfdd6af84f7e26970a09f1b79b35eaaa0acfbfcf25a33a0` | exact |
| Once-only attempt | `edaf0f8a05ea8677ebf077004e1e231de126e27e7fd90a699dbaa92dce17d867` | exact, resume false |
| Harness | `687dac366147968616e2888266282869a0a25ef9e9b656b3264743cfae8c67c7` | exact |
| Focused tests | `8ea7f74ad13270a0acd11358292ca0a114e43efd578405dde3ca6714f8a4e3bd` | exact |
| Production fit seam | `eb13bdda3a207253fba7a36c55788d75b703b61daf4e9831891d381d8c32b99e` | exact |
| Fit-seam tests | `4db63c62288b29c8b020365023a3712d53f2bf523c799e7fd8f0305a0a077e32` | exact |
| Scientific JSON | `041873c09c949a47d12c6ae05553d9722a9756050c12478855d55dd45f9c4315` | exact |
| Raw NPZ file | `028c93f350b30b61debebd5bf0706ff128f2c54faaee04614d1ee12191a3aeb7` | exact |
| Raw semantic collection | `383e8372d81c78f263111126071ad6b55fd617a70f139fc456401536fae4e352` | exact |
| Result note | `6d48340a17e8dd655d4983904a5bce0faf04fcc5efea015cd058865ebc6f1d4b` | exact |
| Full sealed-source collection | `92b18992e149ab4d836d9f5a718fdf766dd0fdb010f2fd4f4f44a4ccdedc22fa` | exact, 78 paths |
| Runtime-loaded source collection | `d7eae633b5ff3a0f8566752fa61ee584f756d877a4faa19738b5ef8af68f34c6` | exact before/after, 23 paths |
| Recomputed-result canonical digest | `4d41441b65d143648c6230bb500821f60578193591bec13c6616cd76d1a36d90` | exact |

Chronology is coherent: preregistration was frozen before implementation; its two explicitly
logged feasibility amendments precede the outcome-free implementation review at
`2026-07-16T10:13:25Z`; seal creation was `10:15:28Z`; the prospective output name encodes
`10:16:08Z`; the exclusive attempt was claimed at `10:16:15Z`; and the valid result was completed
at `10:20:25Z`. The exact seal and run commands, repository root, absolute interpreter/harness,
seal path, and six prospective output paths all match. Only the valid JSON/raw/note triple exists;
the invalid triple is absent.

The dirty repository was not treated as clean: revision
`2dddca4aff59702341af9faceefa76ad2505dd83` plus the tracked binary diff and 214 untracked-file
hashes were sealed. All 78 sealed paths still match. The complete runtime-loaded map is identical
before and after execution. The environment is exact Torch `2.9.0+cu128`, CPU-only,
`CUDA_VISIBLE_DEVICES=""`, four Torch/OMP/MKL threads, deterministic algorithms enabled, and no
gsplat or StructSplat module loaded.

Seal-time Ruff, format, the complete non-slow pytest suite, docs-sync, and `git diff --check` all
returned zero. The reviewer reran the 46 protocol tests plus 29 production-seam tests, relevant
Ruff checks, and `git diff --check`; all passed. The production default remains
`weight_color_9p`, and the current/default path is bound by bit-exact masked and unmasked legacy
parity tests.

## Isolation and raw-evidence audit

The NPZ has exactly 360 unique, uncompressed `ZIP_STORED` members, is readable with
`allow_pickle=False`, has no object array or unsafe logical name, and contains 576,767,760 raw
payload bytes. Every manifest dtype, shape, byte count, semantic content hash, file hash, and the
collection digest was independently reconstructed. All 222 floating arrays are finite.

The exact ordered populations are six scenes, 54 common initializers, 108 fits, 12,960 optimizer
updates, 864 checkpoints, 26,784 callbacks, 129,600 checkpoint-component rows, 972,000 mechanism
component-update rows, and 486,000 current-null rows. Blocks, seeds, selected views, arm order,
component indices, checkpoints, table transcripts, scheduler events, and completion counters all
match the preregistration.

A reviewer-only source replay called the frozen synthetic scene factory once for each of the six
official seeds and read only original views `[0,1,2,4,5,6,8,9,10]`. All 54 stored targets matched
bit-exactly, global Torch RNG was unchanged, and the receipts were:

- target collection: `9786d6bdba9bfc6b3e2887d71a4423731e508e0c736779a3c33403bb60b82b6b`;
- target-generator source collection:
  `3a106700ce54d48d116af635c3374417c37d1a036b1c2f07c77256c0d428a616`;
- environment fingerprint:
  `ceb569464f6e746a45e851ee4d10f5e9382114fccf349486047417bc0017ea68`.

All target/g0 hashes, generator-state hashes, shapes, ranges, exact initial weight 0.5, center
bounds, and positive Cholesky diagonals pass. The two arms share bit-exact raw and built geometry.
The candidate has exact unit weight. Across all 54 common forwards, the worst candidate-amplitude
error was `2.9802322e-8`, render maximum absolute difference `4.7683716e-7`, render relative L1
`2.7022903e-8`, loss absolute difference `1.4901161e-8`, and loss relative difference
`2.3561248e-7`; the minimum step-zero equivalence PSNR was 120 dB. Every frozen limit passes.

The reviewer freshly rebuilt and rendered all 864 checkpoints. Raw-to-built fields, target links,
fresh renders, objective loss, float64 SSE/count/MSE, clamp images/masks/counts, float32 PSNR, and
SSIM were exact. Candidate weight remained exact one; mechanism geometry stayed bit-identical in
all 6,534 recorded states; every result equals its terminal checkpoint; all fits completed 120
updates with no count change or early stop.

For all 6,480 appearance-only updates, a fresh direct-amplitude render, loss, and backward pass
reproduced the stored amplitude gradient bit-exactly. Probe render maximum difference was
`4.7683716e-7`; every probe denominator and equivalence gate passed. Independent chain-rule maxima
were:

| Identity | Maximum absolute error | Maximum relative error | Limit | Result |
|---|---:|---:|---:|---|
| Current color raw gradient | `1.3969839e-9` | `5.1886041e-6` | `2e-6`, `1e-4` | pass |
| Current weight raw gradient | `4.1909516e-9` | `7.1239770e-6` | `2e-6`, `1e-4` | pass |
| Candidate amplitude raw gradient | `2.3283064e-10` | `1.8821490e-7` | `2e-6`, `1e-4` | pass |

Adam first/second moments, step/state presence, learning rate, raw displacement, and state
continuity were independently reconstructed for every mechanism update and all 378 retained joint
transitions; maximum displacement residual was exact zero. Fresh backward passes at all 378 joint
transition states reproduced every stored geometry and appearance gradient bit-exactly. The
sealed valid-result path also performed its full deterministic 54-fit joint-trajectory replay
before reduction and again after raw reload, closing otherwise unanchored gaps between retained
checkpoints; the audited source makes that replay mandatory for a valid artifact. A separate
post-result deep validation reproduced canonical digest `4d41441b...` without reissuing the
once-only command.

Independent NumPy SVD reproduced all 129,600 analytic Jacobians/ranks/singular values. Every rank
was three and neither arm had a weakly responsive mechanism checkpoint row. The largest current
`||Jn||` was `9.7762096e-17`, minimum analytic/SVD null alignment was
`0.9999999999999997`, gradient-null maximum absolute dot was `1.6331114e-9`, and maximum defined
cosine was `1.1926495e-5`; all gates pass. Every saturation count, defined mask, fraction, and
fixed-edge histogram was independently reconstructed.

## Independent metric and decision recomputation

All deltas below are candidate minus current and use the seed as the replicate.

| Block / seed | PSNR AUC delta (dB) | Final PSNR delta (dB) | Final SSIM delta |
|---|---:|---:|---:|
| Appearance-only / 7727 | -1.287365 | -1.686669 | -0.035519 |
| Appearance-only / 8837 | -1.405451 | -1.937620 | -0.039218 |
| Appearance-only / 9941 | -1.299169 | -1.764070 | -0.037254 |
| Joint / 10007 | -1.198153 | -1.349526 | -0.047798 |
| Joint / 11003 | -1.425684 | -1.728286 | -0.051041 |
| Joint / 12007 | -1.255075 | -1.426764 | -0.046418 |

The appearance-only means are -1.330662 dB AUC, -1.796120 dB final PSNR, and -0.037330 final
SSIM. Thus the required mean AUC `>=+0.10`, two-seed AUC win count, worst AUC `>=-0.10`, mean and
worst final-PSNR guards, and mean and worst SSIM guards all fail.

The global current-null pool has exact numerator `5.703013170606898`, denominator
`46.39580504236935`, ratio `0.12292087970881893`, and 451,741/486,000 rows at null fraction
`>=0.10` (`0.9295082304526749`). Per-seed ratios are `0.12089275680081243`,
`0.12414213336434801`, and `0.12367576765163267`. The null-motion gate passes. Current and
candidate weak counts are both 0/32,400 globally and 0/10,800 in every seed, so the saturation
guard also passes. These diagnostics do not rescue the failed curve and do not prove a finite
nonlinear update was globally wasted.

The joint means are -1.292971 dB AUC, -1.501525 dB final PSNR, and -0.048419 final SSIM. No seed
meets the `-0.10 dB` final-PSNR per-seed guard; the worst final PSNR is -1.728286 dB; the worst AUC
is -1.425684 dB. The mean, seed-count, worst-seed, SSIM, and AUC non-inferiority requirements all
fail. Material improvement is consequently false.

The independently recomputed frozen decisions are exactly:

| Decision | Value | Disposition |
|---|---:|---|
| `appearance_curve_improved` | false | confirm |
| `null_update_material` | true | confirm, mechanism diagnostic only |
| `candidate_saturation_guard_passed` | true | confirm |
| `fit_time_redundant_coordinate_interference_consistent` | false | confirm |
| `joint_stage1_noninferior` | false | confirm |
| `joint_stage1_material_improvement` | false | confirm |

## Claim table and scope disposition

| # | Claim | Kind and scope | Evidence | Disposition |
|---:|---|---|---|---|
| 1 | The artifact is valid and replay-complete under the frozen CPU protocol. | measured/provenance, six synthetic seeds | seal, attempt, JSON, raw NPZ, source/target replay | confirm |
| 2 | The bounded 8p arm improves the frozen-geometry appearance curve. | measured, Stage-1 source fit only | all raw checkpoints and AUCs | retire; false in 3/3 seeds |
| 3 | Current Adam updates contain material local null-direction motion. | measured local mechanism diagnostic | 486,000 raw rows | confirm, but do not call it globally wasted optimization |
| 4 | The bounded 8p arm avoids a weak-response penalty at frozen checkpoints. | measured guard, not a quality benefit | exact weak counts/histograms | confirm |
| 5 | The result is consistent with redundant-coordinate interference improving the curve. | compound preregistered interpretation | curve/null/saturation gates | retire; compound gate false |
| 6 | The bounded 8p arm is non-inferior or materially better in joint Stage-1 fitting. | measured, CPU synthetic source views | all joint raw checkpoints | retire; false in 3/3 seeds |
| 7 | The candidate learns one fewer scalar per component. | structural fact | exact optimizer fields and arm definitions | confirm narrowly; no memory/runtime/bitrate claim |
| 8 | This result supports a default or downstream change. | asserted capability | preregistered scope and failed utility gate | retire; explicitly unauthorized |

The exact bounded candidate is closed on this deterministic CPU-synthetic, fixed-count,
fixed-budget setup. It was not tested on real images, novel/held-out views, Stage 2 lifting,
merging, retention, coverage semantics, Stage 3 refinement, CUDA/gsplat, throughput, VRAM, or
compression. No timing was decision-bearing. A different activation, unbounded GaussianImage
feature, variable-projection solve, learning rate, initialization, primitive count, or schedule
requires a fresh outcome-independent protocol; none may be selected from this result.

At review time, public docs still contain only outcome-free N78 infrastructure prose, including
now-stale statements that no result exists. Those statements must be replaced after this review
with the bounded negative result above; no pre-review quantitative N78 claim was found. The ARA
likewise has no crystallized N78 result claim yet.

## Commands and checks actually executed

- Independent strict-JSON, file/hash, source-map, Git snapshot, chronology, command/path, ZIP, and
  semantic-array-manifest reconstruction under the pinned CPU environment.
- Independent schema/order/completion, target/g0/common-forward, 864 fresh-render/metric,
  NumPy-Jacobian/SVD/null, saturation/histogram, all-mechanism Adam/chain, 6,480 probe-backward,
  378 joint-backward/Adam, seed aggregation/AUC, and frozen-decision recomputations directly from
  the NPZ.
- Reviewer-only source-target replay through
  `reviewer_replay_source_targets(...)`; this regenerated targets but did not fit an arm.
- Sealed harness secondary validation with `deep=False`; exact recomputed object and digest.
- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_stage1_fit_parameterization.py tests/test_stage1_fit_seam.py`.
- Relevant `.venv/bin/python -m ruff check ...` and `git diff --check`.

The reviewer did not replay the once-only scientific command and did not run a new optimizer fit.
CUDA/GPU behavior, timing, and memory were deliberately not tested.
