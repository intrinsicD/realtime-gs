# Compact point training: sampling result and bounded calibrated interaction

## Official sealed CPU synthetic result

- Protocol: 3 seeds x 4 explicit arms, 120 updates, 128 fixed attempts/update, fixed
  `N_init^3D=N_opt^3D=4`, no RGB access.
- Bindings: preregistration `865f86d...65ea`, seal `b04875a...9f4b`, RAW
  `077024a...8a2e`, RESULT `2339dd3...ed93`.
- Independent verdict: PASS for the literal protocol and decision.

| Matched risk | Gaussian `G_init` | Gaussian `G_final` | Gaussian `G_AUC` | Decision |
| --- | ---: | ---: | ---: | --- |
| Discrete pixels | 0.5537714436 | 1.0681355694 | 1.0245665262 | NEUTRAL_OR_NEGATIVE |
| Continuous area | 0.5079419542 | 0.9873547158 | 0.9910818462 | NONINFERIOR, not material |

Global decision: `NO_GLOBAL_SAMPLING_WIN`. Pixel and area rows have different target risks and are
not cross-domain arms. No proposal or production default is authorized.

## Full-resolution calibrated phase-local evidence

- Seven 5328x4608 views, 640 StructSplat components/view, 100 Stage-1 updates; 4,480 total teacher
  components in seven archives totaling 140,945 bytes plus a 4,146-byte manifest.
- CompactCarve: 3,340 candidates, 1,433 eligible, 835 selected.
- Fixed topology: 40 RGB-denied optimizer steps, five effective degree-zero parameter families
  moved; the empty higher-order SH group clock advanced without motion.
- Equal-view compact-teacher MSE: 0.2846208576 -> 0.2267813044 (-20.32%).
- Held-out C1004, 4,096 sampled RGB points: 0.3904081687 -> 0.3402060777 (-12.86%, +0.598 dB).
- Held-out foreground, 256 samples: -8.40% MSE (+0.381 dB).

The calibrated result is terminal FAIL. The first exact-render operation failed during gsplat
import because `LD_PRELOAD` was assigned after Miniconda's older `libstdc++` was already loaded;
no authorized snapshot was saved and HTTP smoke was never reached.

## Post-failure diagnostics and boundary

- Exact native snapshots: `runs/compact_point_training_20260716/postfailure_exact_snapshots/`.
- ABI diagnostic SHA-256: `05c57e38c9024034a5a0fb6daf123fb99d49a71a51738e621a223c7ee8d63e34`.
- Live-viewer diagnostic SHA-256: `23d756f6b190189eb8dc3f3a919ec37c96cbb7938fc554b5404c6991fd405ea9`.
- Failure audit: `runs/compact_point_training_20260716/CALIBRATED_FAILURE_AUDIT.md`.

These diagnostics are separately labelled and do not repair the once-only result. A future success
requires a fresh preregistered namespace and an actual bound spawned gsplat/CUDA lifecycle.
