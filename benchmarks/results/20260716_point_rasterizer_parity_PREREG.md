# Preregistration: sparse point-rasterizer parity and discrete sampled risk

## Chronology, question, and claim boundary

Frozen at `2026-07-16T19:09:24+02:00`, after source/design inspection and before adding a
point rasterizer, a discrete pixel sampler, a benchmark harness, an official fixture, an attempt
marker, a parity result, a calibrated render, or a timing sample for this experiment.

Outcome-blind preimplementation amendment at `2026-07-16T19:13:01+02:00`, still before any of
those actions or outcomes. The initial document had SHA-256
`44f985ed3d17d36482585a6b00770add132cf8c44b5489f900295c762079237d`. Independent protocol and
renderer-math reviews identified two omissions: a point-only chunk cap still leaves the pair
temporary proportional to every visible Gaussian, and a uniform-only sampled-risk gate would not
exercise the Gaussian proposal/rejection method motivating compact refinement. This amendment
adds an independently configurable visible-Gaussian stream and freezes an exact discrete
Gaussian-mixture/rejection proposal. It changes no official scene seed, parity tolerance,
calibrated input, stopping rule, or claim boundary.

Second outcome-blind preimplementation amendment at `2026-07-16T19:18:12+02:00`, still before any
implementation, fixture construction, official RNG draw, seal, attempt, or outcome. The first
amended document had SHA-256
`78782a4914d92982ea7518558729785dd4f76860acde266c781cc0d2e501b136` and its independent
executability review returned `FAIL`. This amendment closes only that review's blockers by
freezing literal fixture-construction code and arbitrary coordinates, adding nonvacuity and
analytic-variance requirements, fixing one once-only artifact namespace and atomic marker
lifecycle, restricting the point renderer to the tested default visibility margin, and specifying
a JSON-calibration-only real-data route. No implementation or outcome informed these choices.

Third outcome-blind preimplementation amendment at `2026-07-16T19:23:04+02:00`, still before any
implementation, fixture construction, official RNG draw, seal, attempt, or outcome. The second
amended document had SHA-256
`d70654c0435a73de22f73c2972bf02cbeece960f07326448bc1d0be6ade91878`; its re-review found the
previous blockers closed except for two remaining postimplementation-selectable fixture details.
This amendment freezes the exact gradient-coefficient RNG calls, requires nonvacuous dense
gradients in every parameter family, and freezes the global-compositor mutation numerically. It
changes no tolerance, seed, renderer equation, sampling equation, lifecycle, or claim.

The question is a prerequisite question, not a quality contest: can an independent sparse query
surface evaluate the current pure-Torch 3D Gaussian renderer at selected image coordinates while
preserving its forward values, global depth compositor, parameter gradients, and retained
screen-space center gradients? A second gate asks whether fixed-attempt sampling of discrete pixel
centers estimates the exact finite-image loss without normalization drift across microchunks.

Passing authorizes a later, separately preregistered compact 3D-to-2D-teacher refinement
experiment. It does not itself establish successful refinement, convergence, improved quality,
reduced memory, greater speed, CUDA/gsplat parity, arbitrary-continuous-coordinate equivalence to
a dense image, a density policy, or a production default. Failure blocks use of this point path as
a training oracle. Tolerances, fixtures, and seeds cannot be changed after an official outcome in
this namespace.

The source snapshot inspected before implementation is git revision
`2dddca4aff59702341af9faceefa76ad2505dd83`, with a pre-existing dirty worktree. The experiment
seal must bind the complete executed diff. Frozen source hashes are:

| path | SHA-256 |
|---|---|
| `src/rtgs/render/torch_ref.py` | `61716787329e85a186982f81c2a89cb270255473ca26688c409191a1b53bd86e` |
| `src/rtgs/render/base.py` | `1175cf359e2800ff3a518849b43c4d9a6fd6dccc3dfb7c24459f13e9f81ca0b9` |
| `src/rtgs/core/gaussians3d.py` | `d417a4a103ae7ea1e3f4a7799c2b709597014b8966acb0e72b2bd447a0ad0ba5` |
| `src/rtgs/core/camera.py` | `1e6a42c7cd9fa14b2ffff19808e6e88c106df4562d30fc18b0ca107c00072ac2` |
| `src/rtgs/core/sh.py` | `554f3a25e25c7312248a98c15685e9bf805c85a81a96f56e13e1481619eb4687` |
| `src/rtgs/core/observation2d.py` | `ff2e54899b1a21b773075c2a053c14a8d66ed4b397870b2df63228a12fd49fd6` |
| `src/rtgs/lift/compact_carve.py` | `87efa40b4e5ac40684367e723a57a46a2f08b8613d225124daf3144cff1afa83` |
| `src/rtgs/data/reconstruction_inputs.py` | `2f93b571760c61d8fce6ecc5bfcfe103ecbce2049d4c15c3c43c33132577376b` |

## Frozen implementation surface

Implementation may add a CPU reference `PointRasterizer` protocol/output and a
`TorchPointRasterizer`, plus discrete pixel-center sampling/loss helpers, exports, focused tests,
and one benchmark harness. The dense `TorchRasterizer` arithmetic and all defaults remain
unchanged. Sharing a helper is allowed only if a test-local frozen legacy implementation proves
bit-exact dense forward values and gradients; duplicating the established equations is preferred
for this first semantic anchor.

The point call is conceptually:

```text
render_points(gaussians, camera, xy, background=None, sh_degree=None)
```

where `xy` has shape `(S,2)` in full-canvas pixel coordinates and the result contains color
`(S,3)`, alpha `(S,)`, depth `(S,)`, retained `means2d`, and sorted global `visible` indices.
Inputs must be floating, finite, on the Gaussian device, and have shape `(S,2)`; an empty query is
valid. The implementation has positive `point_chunk` and `gaussian_chunk` controls and limits
every pair-shaped temporary to `point_chunk*gaussian_chunk`. Streaming Gaussian chunks must carry
the exact front-to-back transmittance and accumulated color/alpha/depth across chunks without
reordering or dropping zero-kernel columns. It may expose only the existing CPU renderer controls
for SH activation/SMU parameter and hard or declared existing kernel mode. This first
implementation supports only visibility margin `3.0`; if a constructor argument is exposed, every
other value must fail explicitly. Unsupported diagnostics must likewise fail rather than silently
change semantics.

For each camera the point path must independently reproduce the dense renderer's complete default-
margin coarse visibility set and global camera-depth order. It may not accept lineage/component
IDs and may not filter candidate 3D Gaussians by the 2D component that proposed a query. A
selected point is composited against every coarsely visible Gaussian in that view. The dense
renderer's nondefault expanded-margin ordering branch is outside this experiment and remains
blocked until separately tested.

The frozen equations are the established `torch_ref.py` equations: near plane `z>0.05`, EWA
projection with screen dilation `0.3 pixel^2`, default spectral-radius image culling, global depth
sort, analytic 2x2 inverse with determinant
floor `1e-12`, the selected kernel-support mode, alpha clamp `[0,0.999]`, exclusive
`cumprod(1-alpha+1e-10)`, SH view direction and activation, alpha-weighted unnormalized depth, and
the exact terminal background weight. No nearest-neighbor, bilinear dense lookup, local primitive
subset, normalized depth, or normalized alpha blend is permitted.

Pixel centers are `xy=(column+0.5,row+0.5)`. At those coordinates the point implementation must
match a row-major gather from a dense render. At the four frozen in-canvas continuous coordinates,
which the harness must assert are away from declared hard boundaries, outputs and coordinate
gradients must be finite. No universal differentiability or dense-equivalence claim is made because
hard support is discontinuous at its boundary and the dense renderer has no arbitrary-coordinate
API.

## Development boundary and implementation seal

Development may use only seeds `424201`, `424202`, and `424203`, images at most `11x13`, and at
most eight Gaussians. These fixtures may diagnose code but may not change the official protocol or
support a scientific claim. Official seeds and the calibrated PLY must not be constructed, loaded,
rendered, or inspected until:

1. focused development tests and the full CPU verification command pass;
2. an implementation review confirms the frozen equations and no lineage restriction;
3. source, harness, preregistration, environment, command, and dirty diff are bound in an
   append-only seal; and
4. an attempt marker is written before the first official fixture is constructed.

After the attempt marker, any exception, nonfinite value, invariant failure, or threshold failure
is an official failure. There is no repair-and-rerun under this namespace.

## Frozen Phase A fixtures

Run with `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`, and `MKL_NUM_THREADS=4`; use the repository
`.venv`, deterministic Torch algorithms, four Torch intra-op threads, CPU, and no optional gsplat
import. Record Python/Torch/NumPy/platform/CPU/thread/environment metadata, git revision, dirty diff
hash, loaded source hashes, and commands. Runtime is descriptive only.

Official seeds are exactly `91301`, `91302`, and `91303`, with no replacement. For each seed the
harness must execute this constructor literally and in this call order; importing the harness
must not call it:

```python
def official_fixture(seed: int) -> tuple[Gaussians3D, Camera]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    camera = Camera.look_at(
        torch.tensor([0.0, 0.0, -3.0]), torch.zeros(3),
        fov_x_deg=55.0, width=19, height=15,
    )
    means = torch.empty(11, 3, dtype=torch.float32)
    means[0] = torch.tensor([0.0, 0.0, -0.65])
    means[1] = torch.tensor([0.0, 0.0, 0.35])
    means[2] = torch.tensor([-0.25, 0.12, 0.0])
    means[3] = torch.tensor([0.31, -0.18, 0.0])
    means[4:9, 0] = 1.2 * torch.rand(5, generator=generator) - 0.6
    means[4:9, 1] = 0.8 * torch.rand(5, generator=generator) - 0.4
    means[4:9, 2] = 1.2 * torch.rand(5, generator=generator) - 0.6
    means[9] = torch.tensor([2.5, 0.0, 0.0])
    means[10] = torch.tensor([0.0, 0.0, -3.2])
    quats = torch.randn(11, 4, generator=generator)
    log_scales = torch.log(0.04 + 0.14 * torch.rand(11, 3, generator=generator))
    opacity = 0.12 + 0.76 * torch.rand(11, generator=generator)
    sh = 0.04 * torch.randn(11, 9, 3, generator=generator)
    sh[:, 0] = rgb_to_sh(0.15 + 0.70 * torch.rand(11, 3, generator=generator))
    return Gaussians3D(means, quats, log_scales, opacity, sh), camera
```

Rows 0 and 1 share projected center `(9.5,7.5)` with distinct depths; rows 2 and 3 have equal
world `z`; row 9 is intended outside the default coarse envelope and row 10 behind the near plane.
The harness must assert those facts, nondegenerate quaternion norms, and that no tested `q`, alpha,
SH preactivation, near depth, or cull envelope lies within `1e-5` of a hard boundary. A failed
assertion is an official failure, not permission to move a row.

The empty-visible fixture uses the seed-91301 fields with every mean replaced by
`[0.0,0.0,-3.2]`. The empty query is `torch.empty(0,2,dtype=torch.float32)`. The arbitrary
continuous coordinates are literally
`[[0.75,0.75],[4.125,3.875],[18.25,14.25],[2.2,5.4]]` in float32. No other official fixture or
coordinate may be added.

For every official scene, evaluate all `19*15` pixel centers in row-major order with black and
nonblack background (`[0.13,0.29,0.47]`) and `sh_degree` values `0` and `2`. Compare the point
result to direct gathers from `TorchRasterizer(row_chunk=4)` and compare point chunks `1`, `7`, and
`4096`, crossed with Gaussian chunks `1`, `3`, and `4096`. The unchunked semantic anchor is the
`4096x4096` arm; bounded arms may differ only within the frozen numeric tolerance. Focused tests
must intercept the kernel input and assert no pair tensor exceeds the configured product. The
hard SH activation, hard kernel, and default visibility margin are primary. One
supplemental forward-only case exercises each already-supported nondefault activation/kernel
without extending the primary claim.

## Frozen forward, global-compositor, and gradient gates

For all primary forward comparisons require identical `visible` indices and order, and
`torch.allclose` for color, alpha, and depth with `atol=2e-6, rtol=2e-5`. Require the empty-visible
background/zeros and empty-query shapes exactly. Every output must be finite.

The global-compositor intervention clones the seed-91301 fixture, sets row 0 opacity to exactly
`0.77`, zeroes `sh[0]`, then assigns
`sh[0,0]=rgb_to_sh(torch.tensor([0.95,0.05,0.15],dtype=torch.float32))`. It queries exactly
`[[9.5,7.5]]` before and after while labeling row 1 as the synthetic “proposer” in harness metadata
only. The color output must change by at least `1e-4`; the renderer API and implementation must
contain no proposer/lineage argument. This is an invariant, not an effect-size claim.

For gradients, use all 285 centers, nonblack background, `sh_degree=2`, and independent leaf
clones for dense and point paths. Construct identical coefficient tensors with these literal calls
and order:

```python
coefficient_generator = torch.Generator(device="cpu").manual_seed(seed + 100000)
wc = torch.randn(285, 3, generator=coefficient_generator, dtype=torch.float32)
wa = torch.randn(285, generator=coefficient_generator, dtype=torch.float32)
wd = torch.randn(285, generator=coefficient_generator, dtype=torch.float32)
```

Require each coefficient tensor to contain a nonzero finite value. Contract gathered
color/alpha/depth using the literal scalar:

```text
sum(color * wc) / (3*S) + 0.17 * sum(alpha * wa) / S
                           + 0.03 * sum(depth * wd) / S
```

Compare gradients for means, quaternions, log-scales, opacity, and every SH coefficient, plus
retained `means2d.grad`. A missing gradient is failure. Require finite gradients, identical
gradient shapes, and at least one dense-anchor gradient magnitude greater than `1e-10` in each of
the five parameter tensors and retained `means2d` for every official seed. Require
`allclose(atol=4e-6,rtol=5e-5)`. The same gate applies to the crossed point and
Gaussian chunks above; each chunked result is computed from a fresh leaf clone. Coordinates for
this parity gate are constants. A separate four-point non-pixel-center query with `xy` requiring
grad must yield finite `xy.grad`; it has no numeric dense target. A changed near Gaussian must
alter the queried point even when a farther Gaussian is named as proposal metadata only, and
reversing input rows at distinct depths must preserve the result within tolerance.

## Frozen discrete-risk gate

This gate deliberately targets the finite set of fitted-window pixel centers. It adds a separate
discrete proposal rather than rounding the existing continuous `GaussianPointProposal`, and it
does not claim equivalence to that proposal's `continuous_area` risk measure.

Construct exactly this float64 teacher; colors are metadata for future teacher queries and do not
enter the frozen estimator loss in this mechanism gate:

```python
teacher = GaussianObservationField(
    width=5, height=4, fit_window=(1, 1, 3, 2),
    means=torch.tensor([[2.5, 1.5], [3.2, 2.1]], dtype=torch.float64),
    log_scales=torch.log(torch.tensor([[0.8, 1.1], [1.2, 0.7]], dtype=torch.float64)),
    rotations=torch.tensor([0.0, 0.3], dtype=torch.float64),
    colors=torch.tensor([[0.2, 0.7, 1.1], [0.9, -0.1, 0.4]], dtype=torch.float64),
    amplitudes=torch.tensor([0.7, 0.4], dtype=torch.float64),
    epsilon=0.2, sigma_cutoff=3.0, support_fade_alpha=0.4,
)
losses = torch.tensor([0.0, 1/16, 1/4, 9/16, 1.0, 25/16], dtype=torch.float64)
```

The six losses correspond to `teacher.pixel_centers()` in row-major order and have exact risk
`55/96`. This nonconstant literal table isolates proposal/importance arithmetic from point-renderer
parity; a later trainer will supply its differentiable 3D-to-teacher loss. Let `P=6`. For teacher
component `j`,
clip its established rounded integer support rectangle to the fit window, let `A_j` be the number
of pixel centers in that rectangle, and define envelope mass

```text
M = sum_j amplitude_j * A_j.
```

The Gaussian branch selects `j` with probability `amplitude_j*A_j/M`, selects one discrete pixel
uniformly from its rectangle, and accepts with probability
`component_weight_j(pixel)/amplitude_j`. Rejections remain explicit zero-loss null attempts and
must not be resampled. With frozen uniform fraction `eta=0.20`, the active marginal probability of
pixel `p` is

```text
q_p = eta/P + (1-eta) * teacher_weight_sum(p)/M.
```

The target probability is `u_p=1/P`; each active draw has importance `u_p/q_p` and each null has
zero importance. The sampler stores only O(number-of-components) rectangle/count/mass state; it
must not allocate an image, a per-component pixel list, or a support-overlap table. It divides
accumulated weighted loss by the total number of attempted samples and never resamples nulls,
divides by active count, or averages per-microchunk means equally.

First enumerate all six `q_p` values and require the analytic identity

```text
sum_p q_p * (u_p/q_p) * ell(p) == exact risk
```

at `atol=1e-12,rtol=1e-12`, require finite positive nonuniform `q_p`, exact risk `>0`, positive loss
standard deviation, positive analytic variance, positive analytic null probability, and importance
no greater than `1/eta + 1e-12`. For each of 64 sampling seeds `92000..92063`, draw exactly 512
attempts. Across the complete official sample require observed uniform and Gaussian attempts,
accepted Gaussian attempts, and rejected Gaussian attempts. Compute one estimate in a single
chunk and again in microchunks `(1,7,31,473)` whose sizes sum to 512. The two estimates must be
exactly equal if their sampled record is shared and accumulation order is shared; if the
implementation deliberately accumulates chunk sums in a different floating order, the gate is
`atol=2e-12,rtol=2e-12`. Let `d_j=estimate_j-exact`.

The harness computes the analytic one-attempt variance including null mass,
`variance=sum_p q_p*(u_p*ell(p)/q_p)^2-exact^2`. Require every seed estimate to satisfy
`abs(d_j)<=6*sqrt(variance/512)+1e-12`, and require the pooled mean to satisfy
`abs(mean_j d_j)<=3*sqrt(variance/(64*512))+1e-12`. Sample variance never widens a gate. The exact
identity is the primary unbiasedness proof; Monte Carlo catches branch, indexing, RNG, rejection,
and normalization errors. No seed is removed as an outlier.

## Frozen append-only lifecycle and paths

The only files in this experiment namespace are:

```text
benchmarks/results/20260716_point_rasterizer_parity_PREREG.md
benchmarks/results/20260716_point_rasterizer_parity_IMPLEMENTATION_REVIEW.md
benchmarks/results/20260716_point_rasterizer_parity_SEAL.json
benchmarks/results/20260716_point_rasterizer_parity_ATTEMPT.json
benchmarks/results/20260716_point_rasterizer_parity_RESULT.json
benchmarks/results/20260716_point_rasterizer_parity_AUDIT.md
runs/point_rasterizer_parity_20260716/calibrated_parity.json
runs/point_rasterizer_parity_20260716/viewer.log
runs/point_rasterizer_parity_20260716/viewer_snapshots/
```

The seal command is exactly
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python benchmarks/point_rasterizer_parity.py seal`
and fails if the seal, attempt, result, or calibrated parity path already exists. The run command
is the same prefix followed by `benchmarks/point_rasterizer_parity.py run`. It verifies the sealed
source manifest, then atomically creates the attempt with exclusive-create semantics before calling
an official fixture constructor or initializing an official generator. The marker binds the exact
preregistration and seal SHA-256, command, environment fingerprint, source aggregate, and result
path. If the attempt or result already exists, or any sibling path matching
`20260716_point_rasterizer_parity_{ATTEMPT,RESULT}*` exists, run fails closed. A caught exception is
written atomically with exclusive-create semantics to the one result path as status `FAIL` and is
then re-raised. An interrupted process leaves its marker and cannot be rerun. A `PASS` result uses
the same path. No overwrite flag or alternate output argument is permitted.

After a PASS result, the independent scientist review writes only the audit path above. The
calibrated command is exactly
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python benchmarks/point_rasterizer_parity.py calibrated`;
it requires the sealed source manifest, PASS result, and audit verdict `PASS` or `QUALIFIED`, then
exclusive-creates `calibrated_parity.json`. The exact viewer command is
`.venv/bin/rtgs view --gaussians runs/dataset_viewer_fullres_20260716/gaussians_init.ply --scene dataset/2025_03_07_stage_with_fabric/frame_00008 --downscale 16 --device cpu --rasterizer torch --snapshot-dir runs/point_rasterizer_parity_20260716/viewer_snapshots --host 127.0.0.1 --port 8767 --no-open`.

## Decision and stopping rules

Phase A is `PASS` only if every schema, visibility/order, forward, empty, compositor, parameter-
gradient, `means2d`-gradient, continuous-coordinate finiteness, exact categorical expectation,
Monte Carlo, and microchunk gate passes for every declared case. Otherwise it is `FAIL` and compact
training remains blocked.

After a Phase-A pass and independent audit only, run one bounded calibrated interaction check using
`runs/dataset_viewer_fullres_20260716/gaussians_init.ply` and
`dataset/2025_03_07_stage_with_fabric/frame_00008`, CPU, calibrated downscale 16, and camera
`C0001`. The parity process reads only `dataset/2025_03_07_stage_with_fabric/calibration_dome.json`
and the PLY. It constructs the ideal pinhole camera directly from C0001's calibration `resolution`,
`camera_matrix`, and `view_matrix`, using width/height integer-divided by 16 and the established
`(+0.5)*scale` principal-point convention. It must not import PIL, enumerate/open an RGB or mask
file, call `load_calibrated_scene`, or construct `SceneData`; a focused test patches image open and
the calibrated RGB loader to fail. Query a sealed deterministic set consisting of all pixel
centers when the
downscaled image has at most 4096 pixels, otherwise 4096 uniformly drawn pixel centers using seed
`93001`. Require the same forward tolerances and visible order against the dense CPU renderer.
This check loads no source RGB and reports no reconstruction metric.

Save the exact append-only JSON paths above and then smoke the normal viewer on the unchanged PLY,
same calibrated scene, downscale 16, CPU Torch renderer, and a fresh snapshot directory. HTTP
startup and one snapshot are required. The viewer demonstrates integration visibility only; it is
not a new reconstruction or evidence that point-sampled refinement works. The viewer is explicitly
allowed to decode RGB for display and is outside the parity process's no-RGB claim.

An independent results audit must review chronology, seal, official-attempt lifecycle, source
drift, all per-case values (not only aggregates), calibrated routing, and claim scope before any
documentation records a pass. No default changes in this experiment.
