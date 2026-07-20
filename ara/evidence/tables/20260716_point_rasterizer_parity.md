# Sparse point-rasterizer and discrete-pixel parity evidence

## Bound artifacts

- Preregistration: `benchmarks/results/20260716_point_rasterizer_parity_PREREG.md`, SHA-256
  `afc9d036ad1c037a5cb3eab7fd5b19f97d37d920f520cb5c51bf37f41f989916`.
- Seal: `benchmarks/results/20260716_point_rasterizer_parity_SEAL.json`, SHA-256
  `51d9a5c75397568f311325064943671a3c2fa5a00c2743d9e1ae6d3e00b1801d`.
- One-shot Phase-A result: `benchmarks/results/20260716_point_rasterizer_parity_RESULT.json`,
  SHA-256 `1abbdec0fd0fb71a3aa746430ca7f84b08999476951eb5386852d804cbfd4d85`.
- Calibrated interaction: `runs/point_rasterizer_parity_20260716/calibrated_parity.json`,
  SHA-256 `d8779c15224881f3f61a2bdb11cffd1ab28d009e594bc6ac3e036e7d73c7bdf4`.
- Independent Phase-A, calibrated, and viewer review:
  `benchmarks/results/20260716_point_rasterizer_parity_AUDIT.md`.

## Synthetic mechanism result

| Gate | Scope | Worst observed | Frozen limit | Result |
|---|---|---:|---:|---|
| Forward color | 108 arms | `5.9604645e-08` abs | `2e-6` abs | PASS |
| Forward alpha | 108 arms | `1.1920929e-07` abs | `2e-6` abs | PASS |
| Forward depth | 108 arms | `2.3841858e-07` abs | `2e-6` abs | PASS |
| Parameter/screen gradients | 27 arms, all families | `1.8626451e-09` abs | `4e-6` abs | PASS |
| Global-compositor intervention | non-proposer near Gaussian | `0.3537486792` color change | `1e-4` floor | PASS |

All four supplemental activation/kernel cases, empty contracts, visible-order invariants, and
finite arbitrary-coordinate checks passed. The arbitrary-coordinate xy gradients were exactly
zero in all frozen cases, so active off-grid differentiation is not established.

The exact discrete target and enumerated importance expectation both equal `55/96`. Across 64
Monte Carlo seeds, pooled absolute error was `0.0075645968 < 0.0141208072`; worst per-seed error
was `0.0864502052 < 0.2259329157`; worst fixed-attempt microchunk discrepancy was
`2.2204460e-16 < 2e-12`. Branch counts reconcile exactly across 32,768 attempts.

## Calibrated and viewer interactions

The no-RGB/no-mask calibrated route read C0001 calibration and the existing 835-Gaussian PLY
directly. On 4,096 uniform replacement draws (3,998 unique) from the 333x288 downscale-16 domain,
worst absolute color/alpha/depth errors were `8.9406967e-08`, `1.7881393e-07`, and
`4.7683716e-07`, all below `2e-6`. Its `1.3903779` seconds includes both renderers and provenance
checks and is not a speed measurement.

The separate viewer smoke intentionally loaded RGB references. HTTP/UI integration saved scene
camera 0 (`C0000`) as `viewer_snapshots/final_camera_0000.png`: RGB, 333x288, SHA-256
`5392c6d4c03a6965dd043291de7fa2e89e53823a618e2a81d2dd0b32aa8df209`. This proves a live viewer
handoff and exact Torch/CPU snapshot action, not visual quality or another C0001 parity result.

## Authorized conclusion

CPU selected-point compositor parity and the discrete-risk estimator identity are supported in
their frozen scopes. No optimization, convergence, reconstruction quality, full-resolution
training, end-to-end memory, speed, density control, CUDA/gsplat parity, or production-default
claim is authorized.
