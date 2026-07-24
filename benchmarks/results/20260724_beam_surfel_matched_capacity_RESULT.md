# Matched-capacity falsification of the surfel-initialization count result — result

Date: 2026-07-24

Verdict: **`M1_CAPACITY_ADVANTAGE_HOLDS`** — unanimous across all three seeds. The capacity claim
made by the 20260724 screen **survives** a test designed to withdraw it. Guardrail passed. **No
default change is authorized**; this is a same-root mechanism confirmation, not generalization.

Protocol:
[`20260724_beam_surfel_matched_capacity_PREREG.md`](20260724_beam_surfel_matched_capacity_PREREG.md)

Machine-readable result: `runs/beam_surfel_matched_capacity_20260724/summary.json`

Results page: `runs/beam_surfel_matched_capacity_20260724/index.html`

## What was being falsified

The screen reported 22.3279 dB at 2,230 primitives versus the control's 21.8745 dB at 5,030, but
those budgets were **unmatched** — both arms ran under one 8,000 cap and stopped growing at
different points. The alternative explanation was specific and plausible: density control is
greedy, the control's under-sized primitives qualify for it far more often (0.6125 vs 0.2550), and
its extra ~2,800 primitives bought a train-view advantage (28.6049 vs 27.2435 dB) that reversed
held out. On that reading the control was merely allowed to overshoot, and at a common budget it
would catch up.

It does not catch up. It gets worse.

## Result

Matched hard budget 2,400 (= 3 × the frozen 800-Gaussian initialization, derived from `N_init`
and not from any observed count); seeds 0/1/2; held-out cameras `C0004, C0025, C1004`.

| arm | seed | final N | hit cap | held-out FG PSNR | α-IoU | α-out | PSNR AUC | `C1004` | train PSNR |
|---|---:|---:|:--:|---:|---:|---:|---:|---:|---:|
| `ci` | 0 | 2400 | yes | 21.6781 | 0.9205 | 0.0470 | 17.8735 | 19.9360 | 26.3284 |
| `ci` | 1 | 2400 | yes | 21.6567 | 0.9240 | 0.0459 | 17.8463 | 19.7835 | 26.3042 |
| `ci` | 2 | 2400 | yes | 21.6039 | 0.9137 | 0.0483 | 17.9501 | 19.7361 | 26.4617 |
| `surfel` | 0 | 2287 | no | **22.1855** | 0.9241 | 0.0441 | 19.9453 | 20.4537 | 27.6596 |
| `surfel` | 1 | 2370 | no | **22.3017** | 0.9263 | 0.0432 | 19.9264 | 20.4808 | 27.8644 |
| `surfel` | 2 | 2311 | no | **22.4088** | 0.9278 | 0.0458 | 19.9699 | 20.6744 | 27.7401 |
| `cover-iso-op` | 0 | 2230 | no | 22.3279 | 0.9298 | 0.0424 | 20.0137 | 20.5739 | 27.6666 |
| `cover-iso-op` | 1 | 2115 | no | 22.2254 | 0.9252 | 0.0431 | 20.0386 | 20.4831 | 27.5761 |
| `cover-iso-op` | 2 | 2154 | no | 22.0910 | 0.9307 | 0.0435 | 20.0416 | 20.5441 | 27.7378 |

Mean held-out foreground PSNR: `ci` **21.6462** (sd 0.0382) at 2,400.0 primitives; `surfel`
**22.2987** (sd 0.1117) at 2,322.7; `cover-iso-op` **22.2148** (sd 0.1188) at 2,166.3.

Preregistered per-seed decision (margin 0.15 dB, treatment may not spend more primitives):

| seed | `surfel` − `ci` | N (surfel / ci) | cell |
|---:|---:|---:|:--:|
| 0 | **+0.5074 dB** | 2287 / 2400 | M1 |
| 1 | **+0.6450 dB** | 2370 / 2400 | M1 |
| 2 | **+0.8049 dB** | 2311 / 2400 | M1 |

**3 of 3 → `M1_CAPACITY_ADVANTAGE_HOLDS`.** Guardrail: worst final held-out alpha-outside is
0.04829 (`ci`/seed2), inside the 0.05 bound for every one of the nine cells.

## Why this is stronger than the screen it confirms

1. **The control is budget-limited, not overshooting.** Capping it at 2,400 *costs* it 0.196 dB
   versus its own uncapped 5,030-primitive endpoint (21.6462 mean vs 21.8745). It genuinely wants
   those primitives. The treatment, given the same 2,400, **never reaches the cap** in any seed
   (2,287 / 2,370 / 2,311) and still wins by 0.51–0.80 dB.
2. **The overfit ambiguity is gone.** In the screen the control led on train views and trailed
   held out, which is consistent with either "better initialization" or "more capacity". At
   matched capacity the treatment leads on *both*: train 27.66–27.86 vs 26.30–26.46, held out
   22.19–22.41 vs 21.60–21.68. There is no longer a metric on which the control is ahead.
3. **The effect dwarfs seed noise.** The 0.65 dB mean separation is 6–17× the within-arm seed
   standard deviation (0.038–0.119 dB), and every one of the three seeds falls the same way.
4. **It also holds on the extrapolative camera.** `C1004` alone: 20.45–20.67 for `surfel` versus
   19.74–19.94 for `ci`, in all three seeds.

## What this does *not* establish

- **Same root.** This runs on `frame_00009`, the root of the screen under test. It confirms a
  mechanism and cannot generalize. A genuine multi-scene test needs mask-bearing captures: the
  two remaining checked-in roots (`karate/frame_00005`, `karate/frame_00060`) carry **no packed
  alpha**, so the mask-based coverage/leakage protocol cannot run on them without changing the
  supervision regime, which would confound scene with protocol.
- **`cover-iso-op` remains post-hoc.** It was selected from the screen's own outcome and is
  reported here as a labeled secondary arm. It performs comparably to `surfel` (mean 22.2148 vs
  22.2987) at ~7% fewer primitives, but a run cannot confirm the arm it selected.
- **Nothing about production topology or speed.** CPU reference rasterizer, classic CPU
  controller, downscale 32, 1,000 steps, 8 of 26 views. No CUDA gsplat, no MCMC/relocation, no
  timing claim. `max_gaussians=2400` is a hard cap, not a tuned budget.
- **The initialization's silhouette halo is untouched.** The screen's G1 failure (initial
  alpha-outside 0.13125) is a property of the initialization and is unaffected by this run;
  it is trained away by step 1,000 in every cell but remains an unfixed defect at step 0.

## Provenance

- Protocol `59d72e125797a704a6b1bf1da775aee4a4f20cd9471e582e8896c6463fd06b20`, unchanged since
  the run and confirmed against `summary.json`.
- Executed sources: harness
  `98a391700125251744f9c81f94fa7e2a7e7ef4ffdd592c6806e775eedbb01eee`, imported screen harness
  `6f2baad9dcd1937b0594c9495ad6706df81f5b73bef936e7500238ee7a56eeef`, module
  `984a8aad33c4afbe7668ce7a3afb6752aec4cd670c802e010fcc2214ff2f0702`.
- **The committed harness hash differs from the executed one.** After the run, `write_index` and
  `write_viewer_manifest` were added so the mandated results page and manifest are reproducible
  going forward. Both are called after `summary.json` is written and neither can reach a
  measurement; the page and manifest for this run were generated post-hoc by calling those two
  functions on the saved summary. No numerical path changed.
- Beam Fusion is deterministic and its placement was computed once and shared by all nine cells,
  so every arm is bit-identical in means, SH/color, and count; only quaternions, log-scales, and
  opacity differ, and only the trainer seed varies across cells.

## Viewer

```bash
.venv/bin/rtgs view \
  --comparison-manifest benchmarks/results/20260724_beam_surfel_matched_capacity_VIEWER.json \
  --device cpu --port 8795 --no-open
```

The smoke loaded all 18 models (9 shared initializations plus 9 matched-budget finals), PID 31973
owned `127.0.0.1:8795`, HTTP returned 200, no CUDA device was present, every relative link on the
results page resolved, and the server was stopped afterwards. Receipt:
[`20260724_beam_surfel_matched_capacity_VIEWER_RECEIPT.json`](20260724_beam_surfel_matched_capacity_VIEWER_RECEIPT.json).
The WebGL view is qualitative; all numbers above come from the exact Torch CPU rasterizer.
