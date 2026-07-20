# Independent results audit: dense confidence-gated initialization E1

## Disposition

**ACCEPT THE CALIBRATED MEASUREMENT; REJECT THE PREREGISTERED “BETTER INIT”
DECISION; OPEN I1 NARROWLY.**

Dense+merge improves foreground PSNR on all seven fitted compact views, but it
uses 13.48× the top-K primitive count and lowers mean SSIM. The evidence
supports using the multiplicity distribution to implement the preregistered
confidence gate. It does not support a default change, a held-out-quality
claim, a general scene claim, or a portable speed claim.

## Claim audit

| # | Claim | Kind and scope | Evidence | Disposition |
|---:|---|---|---|---|
| 1 | Dense+merge improves mean fitted-view foreground PSNR over top-K on the calibrated seven-view bundle. | Measured, real calibrated compact bundle, init-only | Official `init_eval.json`; independently recomputed `+1.971401 dB` | **Confirm narrowly.** |
| 2 | No calibrated fitted view regresses by more than 0.25 dB. | Measured, paired seven views | Recomputed paired deltas; worst is `+1.637341 dB` | **Confirm.** |
| 3 | Dense+merge is a count-controlled better initialization. | Preregistered decision | 2,319 versus 172 Gaussians (`13.4826×`) | **Reject.** Count gate fails. |
| 4 | Dense+merge improves initialization quality generally. | Asserted/general | Mean SSIM changes `0.760636 → 0.631656`; no held-out cameras scored | **Retire/general claim.** PSNR improves on fitted compact teachers only. |
| 5 | The merge groups have a useful easy/hard multiplicity distribution. | Measured diagnostic | Histogram sums to all 2,319 groups; 1,819 monocular, 500 multiplicity ≥2, 94 ≥3 | **Confirm as I1 parameterization.** It is not yet a validated confidence classifier. |
| 6 | The new evaluation path is faster. | Single-machine diagnostic | Official and fast receipts; C0001 profile/parity | **Narrow.** Observed locally only; no warmup, repeats, idle guarantee, or controlled CPU/GPU comparison. |
| 7 | E1 establishes held-out or downstream utility. | Unsupported | Held-out C1004 excluded; no optimizer/density-control run | **Reject.** Reserved for E2. |

## Chronology and source binding

- The E1 hypothesis and gate predate outcome access in commit `3856c5c`.
- The official command ran from clean commit
  `68d0a6dc4557c7f9dd5023bf7ecd5cd8808659d6`.
- Strict input manifest SHA-256:
  `6ed60cf3df1f9ca476dfabace18b8d868d63eebd041522313c5b4fd644ee2614`.
- Official JSON SHA-256:
  `7bf4ac973fe373c5b4cf7170877001041f46f9e63ccbf0e16a9b0d3d744f6ea6`.
- Official/replay top-K PLY SHA-256:
  `d83ee1e764ee6bc0d1cf7696e848df91b0a92d33ad5c9932c9e1138e8564e9fb`.
- Official/replay dense PLY SHA-256:
  `56ce5f1ac3a321f6912506dc4e2c8484c1c3b9d5930eb140b84253faf106cff7`.

The complete histogram comes from a post-outcome reporting replay, not the
official JSON. This is acceptable only because both saved placement PLYs are
byte-identical and the reporting change does not alter placement/merge. The
group tensor itself is not serialized, so the histogram is reproducible rather
than cryptographically bound to the official process.

## Protocol and metric findings

1. The seven views are calibrated training/development views from one Janelle
   frame. C1004 was excluded from acquisition. There is no validation subset
   and no held-out E1 metric; no E1 tuning consumed held-out data.
2. The primary gate fails solely on primitive count. Positive PSNR and
   per-view directions cannot rescue that failure.
3. Mean SSIM falls by `−0.128979`. Any prose saying “quality improves” must name
   foreground PSNR and disclose the SSIM regression.
4. Multiplicity `≥2` would retain 500 clusters, still 2.91× the 172-Gaussian
   top-K. Multiplicity `≥3` would retain 94 before other confidence signals.
   These observations may parameterize the preregistered I1 classifier but are
   not post-hoc permission to change E1's count rule.
5. The artifact viewer passed HTTP smoke without a scene. Calibrated snapshots
   are unavailable because raw frame RGB is absent locally; this is a handoff
   limitation, not a failure of strict compact-bundle evaluation.

## Performance and environment findings

- `perf` could not run under `perf_event_paranoid=4`; no privileged sysctl was
  changed.
- `cProfile` assigns 634.0 of 638.7 C0001 seconds to
  `TorchRasterizer.render`; teacher rendering and metrics are small.
- Full-frame rendering followed by cropping was unnecessary. Rendering an
  adjusted fit-window camera matches the official C0001 metrics to
  `≤3.8e-6 dB` / `≤6.0e-7 SSIM`.
- gsplat requires preloading
  `/usr/lib/x86_64-linux-gnu/libstdc++.so.6` in this Conda-launched
  environment; without it the extension fails on missing `CXXABI_1.3.15`.
- CUDA parity tests pass with that preload. Timings remain diagnostic because
  the GPU was not isolated, warmed up, or repeated.

## Checks executed

```bash
.venv/bin/python -m ruff check \
  src/rtgs/lift/compact_init_eval.py src/rtgs/render/torch_ref.py \
  benchmarks/compact_init_eval.py tests/test_compact_init_eval.py tests/test_render.py

CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  tests/test_compact_init_eval.py tests/test_compact_carve.py tests/test_render.py

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python -m pytest -q tests/test_render.py -m cuda

git diff --check
```

Focused disposition at audit time: Ruff passed; CPU tests passed with two CUDA
skips; both CUDA parity tests passed with the required preload. The complete
repository verification result must be appended to the experiment log before
handoff.

## Evidence required for promotion

- I1: freeze exact thresholds for multiplicity, spatial spread, depth width,
  score margin/coverage, color variance, and reprojection residual; add
  constructed deterministic fixtures and reproduce this histogram.
- E2: three matched downstream arms on a frozen train/validation/held-out
  protocol, same optimizer/density schedule and primitive budget, with
  held-out PSNR/SSIM/alpha-IoU, count trajectory, and time-to-quality.
- Performance: idle named machine/GPU, warmup, repeats, aggregation rule, and
  raw receipts before making a portable speedup claim.
