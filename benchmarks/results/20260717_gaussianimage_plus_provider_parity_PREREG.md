# GaussianImage++ provider parity preregistration — 2026-07-17

## Status and scope

This document was frozen before the sealed run. The experiment is a renderer/checkpoint-adapter
mechanism gate only. It does not fit an image, read source RGB, evaluate source-image quality,
initialize or refine 3D Gaussians, change a default, close a pipeline-quality question, or make a
speed/scalability claim. It therefore does not require or produce a 3D viewer handoff. A later
calibrated Stage-1 provider experiment remains mandatory before GaussianImage++ can be integrated
or preferred.

The run has two aims:

1. Confirm that a pure-CPU checkpoint adapter and dense integer-pixel tile reference renderer
   reproduce the bundled GaussianImage++ direct-covariance additive renderer.
2. Confirm a deterministic provider-eligibility policy: preserve only finite components with
   positive covariance diagonals and positive determinant, never silently repair them, and compare
   the resulting set to a native re-render of that exact filtered set.

## Frozen external inputs

- GaussianImage++ repository:
  `/home/alex/Documents/GaussianImage_plus`
- Required clean commit:
  `549cfaab2b400248f685c12782a180f3cfc038b0`
- Bundled ignored extension:
  `/home/alex/Documents/GaussianImage_plus/gsplat/gsplat/csrc.so`
- Extension SHA-256:
  `9b57b7e0531a50d87c529d3541fbf370f9d85455836ac0cf5414c01ce48ac222`
- Isolated Python:
  `/home/alex/Documents/structsplat/results/native_envs/image_gs/bin/python`
- Required Python prefix:
  `/home/alex/Documents/structsplat/results/native_envs/image_gs`
- Required Torch ABI: `2.9.0+cu128`, CUDA runtime label `12.8`
- Sole required `LD_PRELOAD`:
  `/usr/lib/x86_64-linux-gnu/libstdc++.so.6`
- Preload SHA-256:
  `1fd75fe70354a416d75aef22bcae68c47bd25d20e2d0568c30b1a9838cf62f11`
- Frozen real checkpoint:
  `/home/alex/Documents/structsplat/results/native_gaussianimage_plus_matched_proxy/cells/COCO_train2014_000000000009/s160_n640_seed0/native_logs/COCO_train2014_000000000009_s160/gaussian_model.pth.tar`
- Checkpoint SHA-256:
  `ad611facd72e813dece1b95c3268dbfd82f8af01cdb5ad67e1c7675cc670794b`
- Checkpoint render dimensions: 160×120, `color_norm=false`, `clip_coe=3`,
  `radius_clip=1`, tile size 16×16.

The seal additionally binds the external Python/CUDA source files that define checkpoint
parameterization, projection, binning, and rasterization. The foreign checkout must be clean. The
extension is Git-ignored, so its content hash is a separate mandatory binding. An explicit
`import gsplat.csrc` must succeed before any lazy CUDA API is called; JIT fallback is forbidden.
During the official run, every checkpoint consumer hashes the exact in-memory bytes before decode
and rejects any mismatch with the seal; the worker reports that bound input hash and the parent
requires an exact match.

## Frozen native semantics

For checkpoint state `s`:

- means are `s["_xyz"]` in source pixel coordinates;
- direct packed covariance is
  `s["_cov2d"] + checkpoint["slv_bound"]` in
  `[sigma_xx, sigma_xy, sigma_yy]` order;
- colors are `s["_features_dc"]` because the frozen checkpoint used `color_norm=false`;
- opacities are `s["_opacity"]`;
- pixel samples are integer lattice points `(j, i)`, not half-pixel centers;
- the conic is the direct 2×2 covariance inverse;
- `sigma = 0.5*(a*dx^2 + c*dy^2) + b*dx*dy`;
- `alpha = min(1, opacity*exp(-sigma))`;
- a contribution is discarded when `sigma < 0` or `alpha < 1/255`; equality at the cutoff remains;
- component colors are added without normalization or transmittance;
- clamping to `[0,1]` happens once after the complete sum;
- if at least one tile intersection exists, the background is ignored and otherwise untouched
  pixels are black;
- if all components are culled before binning, the Python wrapper returns the supplied background;
- circular long-axis support determines tile candidates;
- the binary executes only its first 256 candidates per tile. Because all 2D depths are zero,
  which candidates survive above 256 can depend on equal-key sorting. The provider experiment
  therefore rejects any field with a tile population greater than 256 rather than emulating this
  truncation.

## CPU provider eligibility

For each finite covariance `[xx, xy, yy]`, eligibility is exactly:

```text
xx > 0 and yy > 0 and xx*yy - xy*xy > 0
```

The mask is evaluated in stored float32 order and preserves component order. Invalid components
are removed. They are not eigenvalue-clamped, jittered, recolored, or replaced. The original
checkpoint replay remains diagnostic only because its renderer accepts some indefinite matrices.
The provider-bearing comparison uses a compact NPZ containing the exact filtered tensors and asks
the isolated native worker to re-render that set.

Known preflight information obtained before implementation: the frozen checkpoint contains 639
components, of which 13 fail the predicate, and its maximum native tile population was 134. These
values are diagnostic expectations, not pass criteria for renderer parity. An implementation smoke
also showed that reciprocal arithmetic can have small absolute differences for large conic values;
the conic endpoint therefore uses combined absolute and relative tolerance. This choice is frozen
before the official attempt. The invalid-component effect on the image is reported but cannot
establish source-image quality because source RGB is out of scope.

## Synthetic cases

All synthetic cases are literal tensors constructed only after the one-shot attempt marker.

1. `overlap_upper_clamp`: three SPD components, two coincident, requiring raw sum above one and
   final upper clamp.
2. `fractional_rotated_lower_clamp`: fractional centers, nonzero covariance cross terms, negative
   and above-one component colors, and final lower clamp.
3. `cutoff`: identity covariance with a retained three-pixel contribution and discarded four-pixel
   contribution, safely away from the exact `1/255` boundary.
4. `all_culled_background`: one far-out component and non-white background, exercising the global
   no-intersection background branch.
5. `one_hit_background_ignored`: one in-frame component with the same non-white background,
   requiring a far pixel to remain black.
6. `radius_clip`: an otherwise valid component removed by frozen short-radius clipping.
7. `out_of_frame_intersection`: center outside the image whose support intersects the image tile.
8. `tile_cap_sentinel`: 257 coincident components. Success means deterministic rejection before
   render dispatch, not agreement with the native truncated image.

## Parity endpoints and thresholds

For every rendered synthetic field and for both real-checkpoint arms:

- foreign worker input tensors equal CPU tensors exactly;
- projected means maximum absolute error ≤ `2e-6`;
- inverse conics satisfy `torch.allclose(atol=2e-6, rtol=5e-6)`; maximum absolute error is also
  reported;
- integer radii and per-component tile hit counts are exactly equal;
- candidate ID sets are exactly equal per tile; order is ignored only because the ≤256 guard makes
  addition commutative over the complete set;
- raw and clamped images satisfy `torch.allclose(atol=1e-5, rtol=1e-5)`;
- raw maximum absolute error is reported and raw mean absolute error must be ≤ `1e-6`;
- maximum tile population must be ≤256, except for the rejection sentinel where it must equal 257;
- all serialized metrics must be finite standard JSON values.

Overall `PASS` requires every synthetic semantic assertion, every rendered parity arm, the tile-cap
rejection sentinel, raw-checkpoint diagnostic parity, and filtered-provider parity to pass. A raw
checkpoint parity failure fails this implementation assay even though that arm is not provider
eligible. A filtered-provider parity failure prevents provider integration. No threshold may be
changed after sealing.

## Lifecycle and artifacts

Only these new implementation files are source-sealed:

- `benchmarks/gaussianimage_plus_provider_parity.py`
- `benchmarks/gaussianimage_plus_native_worker.py`
- `tests/test_gaussianimage_plus_provider_parity.py`
- this preregistration;
- `benchmarks/results/20260717_gaussianimage_plus_provider_parity_IMPLEMENTATION_REVIEW.md`;
- `benchmarks/results/20260717_gaussianimage_plus_provider_parity_IMPLEMENTATION_REVIEW_ADDENDUM_1.md`.

The initial independent review and exact `Verdict: FAIL` remain immutable historical evidence. The
seal command additionally requires a fresh independent addendum with exact `Verdict: PASS`,
re-runs the CPU-only focused tests and Ruff checks, binds all six files and frozen external inputs,
and refuses preexisting seal, attempt, or result artifacts. The official run performs only
outcome-free seal, sealed-source, and static external checks before it creates its attempt marker
with exclusive creation. It constructs fixtures and reads, hashes, or decodes checkpoint bytes only
after that marker. A complete result is then written with exclusive creation; a non-finite or
otherwise non-serializable candidate result is reduced to a finite terminal failure receipt rather
than leaving an orphan attempt. Worker outputs are stored under
`runs/gaussianimage_plus_provider_parity_20260717/`.

Seal command:

```bash
.venv/bin/python benchmarks/gaussianimage_plus_provider_parity.py seal
```

Official run command, prohibited until explicit root-agent authorization:

```bash
.venv/bin/python benchmarks/gaussianimage_plus_provider_parity.py run
```

## Claim limits and next step

A passing result supports only that the frozen CPU adapter/dense reference renderer matches the
frozen external native renderer and that deterministic SPD filtering is faithfully re-rendered. It
does not show that
GaussianImage++ fits the calibrated dataset well, that filtering preserves source-image quality,
that full-resolution fitting fits in memory, that it beats StructSplat, or that it improves 3D
initialization/refinement. The next separately preregistered experiment must fit selected calibrated
views sequentially at full resolution inside Stage 1, preserve variable per-view `m_i`, close RGB
access after compact export, lift the resulting fields, and show the 3D result in the gsplat viewer.
