# Implementation review: sparse point-rasterizer parity

Reviewed at `2026-07-16T20:05:40+02:00`, before an implementation seal, attempt marker,
official fixture construction, official RNG initialization, calibrated PLY load, or outcome.

## Verdict: `PASS`

The implementation is executable against the final preregistration SHA-256
`afc9d036ad1c037a5cb3eab7fd5b19f97d37d920f520cb5c51bf37f41f989916`. Two independent
outcome-free reviews initially returned `FAIL` on harness provenance details; all findings were
repaired and both final re-reviews returned `PASS`. No official seed, frozen teacher, calibrated
input, seal, attempt, or result was invoked during development or review.

## Reviewed implementation

| path | SHA-256 before this review file was added |
|---|---|
| `src/rtgs/render/point_base.py` | `252e66eda091a7b9a769155889e11a2ed3f905a5bdf984164e842820c11203f7` |
| `src/rtgs/render/torch_points.py` | `f0648a20e357f28414337f55fe387d8f9a6b785a8eb53ac9600848790067645b` |
| `src/rtgs/core/observation2d.py` | `c380f6ab921ca18b7947d7764bb49bc13bf80ec9091804475ca3c7c3d3dc2441` |
| `benchmarks/point_rasterizer_parity.py` | `89b0cda4de01ef3c2b5898c22096e86c725b97af2c7271f7d56c4e4ad0cd645a` |
| `tests/test_point_render.py` | `1b54d9340eb47ce9542fceccaff57245a8567190332bd8473d954ad97bcf2bee` |
| `tests/test_observation2d.py` | `9f4a250bf149c65e3e3d2bcb00b36fe088918bbbaf55597efef20a29fc24f5fc` |
| `tests/test_point_rasterizer_parity.py` | `e93aec3ffcf034b2b203fead0646bc9b17d01c8a27cbaeeaf42ade57e98a4c13` |

The future seal rehashes these files, this review, every repository Python source/test, the
preregistration, and `pyproject.toml`; these pre-review hashes are descriptive and cannot override
that source manifest.

## Requirement audit

- `TorchPointRasterizer` duplicates the dense projection/EWA/default-visibility equations and
  retains one camera-wide visible set, one global center-depth order, and one `means2d` tensor. Its
  API has no proposal, component, or lineage argument.
- `point_chunk` and `gaussian_chunk` independently bound every pair-shaped temporary. Gaussian
  chunks preserve order and carry transmittance without in-place graph mutation. The terminal
  background is literally the transmittance before the final Gaussian times `1-alpha_last`, not
  the post-epsilon carry or `1-accumulated_alpha`.
- Default visibility margin `3.0` is the only supported margin. Existing nondefault SH activation
  and kernel modes are mirrored; unsupported visibility fails explicitly.
- Development tests compare dense versus sparse color, alpha, unnormalized depth, all five 3D
  parameter gradients, and retained `means2d.grad`; they also cover continuous `xy.grad`, empty
  inputs, global occlusion, distinct-depth input reversal, terminal background, and pair caps.
- `GaussianPixelProposal` stores only clipped per-component rectangles/counts/masses. It selects
  components by `amplitude*area`, uses exact grouped `torch.randint` rectangle draws, applies exact
  component-weight rejection, retains null attempts, and reports the correct active marginal,
  target probability, and importance. Fixed-attempt reduction divides by all attempts.
- The harness freezes literal official constructors and RNG call order, all forward/gradient/chunk
  arms, degree-0/2 and q=12/16 boundary checks, nonvacuity gates, analytic estimator variance, and
  all calibrated sampling branches.
- Seal creation reruns both focused and repository-wide verification while proving the sealed
  source manifest is unchanged. Run/calibrated commands require the same full environment
  fingerprint, source aggregate, git revision/tracked diff, and actual seal-file hash. The attempt
  and result use exclusive creation, and the marker is rehashed before a PASS can be written.
- A preexisting audit blocks seal/run. Calibrated execution requires an independent verdict that
  contains the exact preregistration, seal-file, and Phase-A-result hashes. Its C0001 camera loader
  reads calibration JSON directly and never imports or calls the calibrated RGB loader.

## Verification evidence

The following outcome-free commands passed on CPU after the final repairs:

```text
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_point_render.py tests/test_observation2d.py \
  tests/test_point_rasterizer_parity.py
# 61 passed

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ./scripts/verify.sh
# ruff PASS; format PASS; non-slow pytest PASS; docs_sync PASS; verify OK

git diff --check
# PASS
```

The seal command repeats and records the focused and full verification output and refuses to seal
if either command fails or changes a sealed source.

## Claim limits

This review authorizes only the once-only CPU mechanism run. A per-tensor pair cap is not evidence
of lower peak memory because autograd may retain multiple chunk graphs. Continuous coordinates are
tested only at the four frozen in-canvas points away from hard boundaries. There is no CUDA/gsplat,
speed, RSS, full-resolution scalability, optimization, convergence, quality, density-control,
production-default, or state-of-the-art claim.
