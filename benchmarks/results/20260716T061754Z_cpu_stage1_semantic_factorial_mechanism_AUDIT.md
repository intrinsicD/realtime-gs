# Independent results audit: Stage-1 semantic-factorial mechanism

Verdict: PASS

Reviewed at `2026-07-16T06:31:50Z` by the independent N77 Phase-A results-audit
session (`/root/n77_phase_a_audit`). I did not implement or execute the harness and did not
participate in the preregistration, implementation review, seal, attempt-marker claim, or official
Phase-A run. I followed the repository `realtime-gs-results-audit` procedure and recomputed the
global Phase-A decision from the raw NPZ with `allow_pickle=False`; no decisive value below is
taken on trust from the result JSON.

There are no blocking, major, or qualifying findings. The narrow Phase-A mechanism claim is
confirmed. This review authorizes only the preregistered fresh-seed Phase-B utility factorial. It
does not establish utility, physical opacity or albedo, real-data behavior, CUDA/gsplat behavior,
performance, upstream-training equivalence, or grounds for a production/default change.

## Claim disposition

| # | Claim | Kind and scope | Evidence | Source/protocol bound | Disposition |
|---|---|---|---|---|---|
| 1 | The complete Phase-A mechanism gate passed. | Measured; CPU synthetic; three seeds; post-fit Stage-1/Stage-2 boundary only. | Bound JSON and raw NPZ named below. | Yes: preregistration, seal, harness, tests, marker, JSON, raw, and note all rehashed. | **Confirm.** Every decisive gate recomputed true from raw. |
| 2 | `m_amp__rgb_obs`, `m_amp__h_norm`, and `unit_weight__a_amp` are operationally invariant under the two frozen product-preserving transforms for both ordinary unmerged lifters. | Measured mechanism result; 54 lifts and 36 transformed-versus-identity comparisons; no utility claim. | Candidate, coverage, retention, sidecar, lift, source-key, field, and render arrays in the NPZ. | Yes. | **Confirm at the frozen synthetic CPU scope.** |
| 3 | Phase B may be opened after an independent PASS review. | Protocol authorization, not a scientific outcome. | This audit and the matching machine review. | Yes. | **Confirm.** `phase_b_authorized=true` is limited to the sole frozen Phase-B command. |
| 4 | The candidate improves held-out reconstruction or should replace the current boundary. | Not measured in Phase A. | None yet. | Not applicable. | **Remain unverified.** Phase B and its own independent result audit are required. |

## Chronology, isolation, and artifact routing

- The preregistration SHA-256 is
  `f53146f12894d5e804baf699b0ba0df51d5768ef708884f5a0343c523d96e1ce`.
  The independent preregistration review and implementation review both predate the seal and
  official attempt.
- Seal creation, marker claim, and result completion are ordered at
  `2026-07-16T06:17:31+00:00`, `2026-07-16T06:18:01+00:00`, and
  `2026-07-16T06:18:24+00:00`, respectively.
- The seal SHA-256 is
  `d07131a90357627e0589016bf9b10c88a804af689482e29ba7576a200b7d8adb`.
  Its 79 sealed files all still match, its complete sealed-source collection is
  `bea80b126dc1caad55f61e05bd52a07c0d8ed32aea85fe8cf6d37c9bfb34b8e2`,
  and the seal's own validator accepted the current source/environment snapshot.
- The harness and focused-test hashes are
  `6baf5455da4f3901ff97e305ba498ea91c956157baf74392c7c1c1622d27e4a7` and
  `28841fb5e4bd482647dfd68b6f0328613211b30f5f76c86e650349cb9d2953e6`.
  The reviewed default native fit remains `weight_color_9p` with geometry unfrozen.
- The once-only marker SHA-256 is
  `0d973c038c14eb4c0cb0eb60cff810d8ce4b93cdaef34aebcf76fd583f695d9f`.
  It binds the sole command, seal, environment, and all six prospective paths. Only the valid
  JSON/NPZ/note triple exists; no invalid sibling exists and resume is forbidden.
- The official environment is CPU-only with empty `CUDA_VISIBLE_DEVICES`, four OMP/MKL/Torch
  threads, Torch `2.9.0+cu128`, and deterministic algorithms enabled. Phase-A raw inputs contain
  only the nine physically subset training views and mapping `[0,1,2,4,5,6,8,9,10]`; there is no
  held-out render, target, metric, or refinement namespace in the archive.

## Raw archive recomputation

The result JSON SHA-256 is
`cfb8b522e7ebf42bbf560227f46419d2bc09d19447e799778ab7ca152ce48d14`.
The raw NPZ SHA-256 is
`fb2b26f2c2da87555e61e90a89a2dd20a4a92ed9499d2b2c7c5731f9a3393310`.

I loaded the NPZ with `allow_pickle=False` and independently inspected every ZIP member and every
array. The archive has exactly 12,849 unique, uncompressed members, all numeric or boolean, with
no object dtype. All 8,417 floating/complex arrays are finite. The arrays contain 398,993,861
uncompressed data bytes. For every array I recomputed the frozen little-endian
dtype/shape/data digest and matched the complete sorted JSON manifest exactly. Recomputing the
name/content collection produced
`017e9665c912a879c8fd6bd467f60fb870302d739f4f1914b059f5cf3b1fd584`,
equal to the JSON and result note.

The JSON's preparation, 54 source rows, 81 candidate rows, 162 coverage rows, 54 lift rows, and 36
lift-comparison summaries were then regenerated from raw arrays and matched their serialized
values. The human result note has SHA-256
`55ddd5548b06c05f94fbf6e150aa28abd7328181fb29962f8a17a9e862463b53`
and correctly binds the completed JSON, raw file, collection digest, and array count.

## Decisive-gate recomputation

### Preparation and fit contract

- Seeds are exactly `[1103,2203,3301]`; all 27 fitted views contain exactly 150 components, for
  4,050 total fitted components.
- The raw global and per-seed fit configurations match the frozen native 120-iteration,
  150-component configuration, including `adaptive_density=true`, `relocate_fraction=0`,
  `appearance_parameterization=weight_color_9p`, and `freeze_geometry=false`.
- All fitted public fields have the frozen shapes and ranges, every Cholesky diagonal is positive,
  every source key is exactly `(seed, local_view, component)`, and all 27 histories reach the
  frozen final iteration 119 without an early-stop branch.
- The nine images, depths, cameras, local/original mapping, scene center/extent, retained world
  priors, and undefined-mask encodings are present and finite. No held-out field entered this
  Phase-A archive.

### Source equivalence and candidate/factorial integrity

- All 54 transformed source comparisons pass. The worst raw transformed-minus-identity source
  render has maximum absolute RGB error `1.7881393432617188e-07`, relative L1 error
  `2.9695985512593094e-08`, and the minimum floored diagnostic PSNR is `120.0 dB`, against frozen
  limits `5e-6`, `1e-6`, and `100 dB`.
- An independent dense evaluation of the frozen `q<12` additive formula from raw gauge fields
  reproduces every archived source render within `2.384185791015625e-07` maximum absolute error.
  The gauge amplitudes themselves differ from identity by at most `1.4901161193847656e-08`
  absolute and `1.127048463445135e-07` relative.
- All raw `a`, `m`, `h`, and bilinear `o` fields were recomputed from fitted/gauge arrays and source
  images. `o` replays bit-exactly. Transformed `m` and `h` are bit-exact to identity; exact zero
  handling, finite `[0,1]` bounds, bit-identical geometry, and the three routed representations
  all pass. The worst `m*h` product error is `1.4901161193847656e-08` absolute and
  `1.127048463445135e-07` relative.
- The four utility-arm tensors satisfy the frozen `00/10/01/11` field identities exactly. Scalar
  and color treatment separation are each 100% in every seed and in the 4,050-component pool,
  exceeding the 10% identifiability floor without using a quality result.

### Coverage and retention invariance

- Independent dense coverage evaluation from each raw routed representation reproduces all 243
  archived maps within `1.7881393432617188e-07` maximum absolute error.
- All 162 transformed-versus-identity coverage comparisons are bit-exact here: worst absolute and
  relative-L1 differences are zero, with zero strict `0.40` threshold crossings.
- The strict `weight>0.05` retained-component arrays were regenerated from routed weights. Every
  identity/transformed array and source-key list is exact, with zero retention crossings.
- All 243 JSON Carve coverage references point to the one canonical
  `coverage/seed=.../gauge=.../arm=.../view=...` raw array. Every independently recomputed content
  digest equals both the manifest digest and the Carve reference; there is no relabeled coverage
  hash domain or duplicate coverage copy.

### Production/independent lift parity and mechanism invariance

- Exactly 54 ordinary unmerged production lifts exist: 27 Depth and 27 Carve. Every output is
  nonempty; counts range from 356 to 1,293.
- All 54 ordered source-key lists were reconstructed from raw Depth `component_indices` or Carve
  `placed_indices`. They match exactly. For every call, production versus independent means,
  covariance, opacity, and SH have maximum absolute error zero under the frozen
  `atol=2e-6, rtol=2e-5` parity gate.
- Raw quaternions/log-scales independently reproduce every archived covariance to within
  `3.725290298461914e-08`. Depth bilinear samples replay bit-exactly. The archived Carve volume
  equations replay within `1.1920928955078125e-07`, and all 243 per-view Carve score equations
  replay within `2.384185791015625e-07`; their retained, argmax, placement, selected-depth,
  variance, sigma, and source-key relations are exact or within the frozen field tolerance.
- Across all 36 transformed-versus-identity lift comparisons, ordered source keys and counts are
  exact. Worst field errors are: means `0`, covariance `5.122274160385132e-09`, opacity `0`, and
  SH `1.1920928955078125e-07`, all below the frozen tolerance. Gauge-invariant Carve volume
  sidecars are also bit-exact across gauges.
- All 324 transformed-versus-identity train-render view comparisons are finite. The worst color
  maximum absolute error is `2.9802322387695312e-08` and the worst aggregate relative L1 error is
  `4.666844380538406e-09`, against limits `5e-6` and `1e-6`. Archived color, alpha, and depth
  deltas equal direct subtraction of their raw render arrays. Alpha and accumulated depth remain
  diagnostics only.

### Completion and global decision

The raw completion tuple is exactly `(phase_code=11, completed_seeds=3,
completed_lifts=54, completed_models=0)`. Required semantic cell counts are complete:
54 source checks, 81 candidate rows, 162 coverage/retention checks, 54 production/independent
lift cells, 36 lift-invariance comparisons, and 324 lift-render view comparisons. Every conjunct
passes, so recomputing the preregistered all-cells rule yields `phase_a_pass=true`. No averaging,
backend rescue, tolerance repair, or utility outcome is involved.

## Commands and skipped evidence

Commands actually executed from the repository root included:

```text
sha256sum <preregistration, reviews, seal, marker, harness, tests, JSON, RAW, note>
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python <seal-validation script>
.venv/bin/python <independent allow_pickle=False manifest/finiteness audit over all 12,849 arrays>
.venv/bin/python <independent preparation/source/candidate/coverage/retention raw recomputation>
.venv/bin/python <independent Depth/Carve sidecar, lift, key, field, render, and completion recomputation>
.venv/bin/python <independent raw-to-JSON summary reconciliation>
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_stage1_semantic_factorial.py tests/test_stage1_fit_seam.py
```

The focused run passed 59/59 tests. I did not replay the once-only official scientific command,
construct an official seed, execute Phase B, run CUDA/gsplat, or collect timing or memory data.
Those omissions are required by the one-shot protocol and the claim boundary, not evidence gaps
for the narrow Phase-A decision.

## Final disposition

The official Phase-A artifact is a valid, fully source-bound PASS for operational gauge invariance
of the three frozen routed representations on the three frozen CPU synthetic seeds under both
ordinary unmerged lifters. The matching machine review may set `phase_b_authorized=true` because
all nine required raw-recomputation gates pass without qualification. Any held-out utility,
real-data, CUDA, performance, physical-semantics, or default claim remains open and requires its
separately frozen evidence and independent audit.
