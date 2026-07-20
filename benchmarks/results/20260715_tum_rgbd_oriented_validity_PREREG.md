# Preregistration: real RGB-D oriented-point validity on TUM

Frozen: 2026-07-15T15:42:19+02:00, before downloading either archive body, decoding an
RGB/depth frame, implementing the backend or harness, constructing a target, or observing any
development or confirmatory metric. Repository revision:
`2dddca4aff59702341af9faceefa76ad2505dd83` plus the dirty worktree bound by the eventual
artifact's source hashes and Git metadata.

Pre-decode protocol-audit amendment: 2026-07-15T15:56:28+02:00, after the initial document and
public API implementation began but before either archive member list, manifest, pose, RGB/depth
payload, target, development metric, or confirmatory metric was opened or observed. This amendment
separates depth-valid from oriented-valid audit populations and freezes previously implicit
numerical conventions. It changes no source, target grid/stencil, threshold formula, or gate.

- A **depth-valid pair** is construction-visible and has a finite validation center depth in
  `[0.3,5.0]` m. It does not require a valid neighbor stencil, validation normal, or incidence.
  `D90` uses targets with at least two distinct depth-valid validation views. `F` uses every
  depth-valid pair directly, without a target-level median. This prevents edge/grazing normal
  failures from hiding independently observed depth or free-space contradictions.
- An **oriented-valid pair** is depth-valid and additionally passes the complete five-depth
  discontinuity, cross-norm, and incidence rules for the validation normal. The existing `S`,
  `S_10`, `R90`, `C50`, and `C10` definitions use targets with at least two distinct
  oriented-valid validation views. The artifact must separately report depth/oriented pair and
  supported-target counts per view and source stratum. Neither population filters the other.
- All metric arithmetic and reductions use detached CPU float64 tensors. Every median is
  `torch.quantile(x,0.5,interpolation="linear")`; p10 and p90 use the same float64 linear-quantile
  convention. Empty or nonfinite required populations invalidate the harness. Normal dot products
  are absolute and clamped to `[0,1]` before reduction.
- Uniform 64-frame subsampling uses integer half-up indices
  `floor(j*(M-1)/63+0.5)`, not language-level banker rounding. Translation distance is Euclidean.
  Quaternion geodesic is `2*acos(clamp(abs(dot(q1,q2)),0,1))` after normalization. Shortest-arc
  SLERP negates the second quaternion when the raw dot is negative, uses normalized linear
  interpolation when the resulting dot exceeds `0.9995`, and otherwise uses the standard sine
  weights. Bracketing timestamps and interpolation fractions are computed from integer
  nanoseconds; quaternion/pose calculations use float64.
- A projected target is in frame only when depth and coordinates are finite, depth is positive,
  and repository coordinates satisfy `0.5<=u<=639.5`, `0.5<=v<=479.5`; its array index is
  `(floor(v),floor(u))`. Z-buffer ties retain the same minimum scalar depth and do not reorder or
  filter targets. Camera-facing normal hashing negates a camera normal exactly when
  `dot(n_camera,-P_camera)<0`; a zero dot is left unchanged.
- Reading and hashing the archive byte stream, listing tar headers, and opening the three text
  manifests are metadata operations. Phase A payload access must use an explicit allowlist equal
  to the selected `T` and `V` depth member paths; the harness records every payload member it
  opens and hard-fails unless the sets are identical. No RGB member and no `H` depth member may be
  passed to `extractfile`, PIL, or another payload decoder. The whole-archive SHA-256 binds sealed
  `H`; individual `H` payload hashes are deferred because computing them would read forbidden
  bytes.
- The scalar-ray identity is unconditional, but `t_star` equals the registered sensor depth only
  when Phase B reuses the exact depth-timestamp camera frozen by this backend. RGB-timestamp or
  otherwise different source cameras make that equality approximate because TUM RGB/depth streams
  are asynchronous. Any Phase-B protocol must reuse this camera or state and measure the mismatch.

Numerical clarification: 2026-07-15T15:58:53+02:00, still before any archive manifest/member or
payload was opened. This supersedes only the preceding in-frame interval: a projection is inside
the 640x480 pixel footprints when `0<=u<640` and `0<=v<480`, and maps to
`(row,column)=(floor(v),floor(u))`; the independently sampled validation point is unprojected at
that array pixel's center `(column+0.5,row+0.5)`. Construction visibility is
`z_pred-z_min<=0.020` m (the minimum makes the difference nonnegative up to roundoff). Exact pose
timestamp matches use that normalized pose; otherwise require strict bracketing
`t0<t_depth<t1` and `t1-t0<=20 ms`. Both 0.08 m and 8 degree keyframe thresholds are inclusive.
Per-construction-view support is zero when a view has eligible targets but none are supported.
Diagnostic `T` bounds are the componentwise min/max of all target points and their Euclidean AABB
diagonal, are non-gating, and never enter target construction. After any `xyz` payload decode, only
a correction proven on a synthetic, TUM-value-free fixture to make code match this frozen text is
allowed; retain any superseded development artifact. No protocol, constructor, visibility,
population, metric, or role change is a defect correction, and desk pixels remain sealed.
Where the original text says an archive or `H` bytes remain "unopened" or "inaccessible", read it
as no semantic PNG decoding, tensorization, or pixel inspection; archive hashing/header traversal
remains permitted as defined above. Target order uses local `T` ordinal 0 through 47 induced by
ascending selected global keyframe ordinal, followed by row and column; both ordinals are
serialized. The TUM backend supplies no confidence tensor. Finite/nonzero geometry and normals are
required only on true validity; invalid canonical entries are zero and exempt.

Pre-decode implementation-audit amendment: 2026-07-15T16:32:17+02:00, after both archive byte
streams and text metadata had been frozen and synthetic fixtures had run, but still before any
RGB or depth PNG was semantically decoded or any TUM target/metric was observed. This amendment
closes implementation auditability gaps without changing a source, selected frame, target rule,
metric, transfer formula, or gate:

- Pose interpolation, camera extrinsics, depth-normal construction, canonical world maps, target
  tensors, and audit arithmetic all remain CPU float64. The repository `Camera` training default
  is explicitly replaced with the already validated float64 extrinsics inside this harness. A
  construction view with zero eligible targets invalidates the harness because its `A_min` and
  `S_10` strata would otherwise contain an undefined `0/0`; zero support is defined only for a
  nonempty eligible stratum.
- Every RGB/depth manifest payload path must be unique and name a unique regular tar member. The
  complete semantic RGB and depth member sets must be disjoint. Before the first payload decode,
  the exact 56-member `T`/`V` depth allowlist and exact eight-member `H` depth set must be unique,
  contained in the semantic depth set, and disjoint from each other and all RGB members. Target
  construction receives a backend containing only the 48 `T` depths; independent audit receives
  a separate backend containing only the eight `V` depths.
- The threshold manifest and its development artifact are each read once as bytes, hashed from
  those same bytes, parsed with duplicate keys and nonfinite constants forbidden, and checked for
  schema, experiment, exact finite metric fields, source, preregistration, implementation, and
  mechanical threshold consistency. A fixed repository-path confirmatory-attempt seal is created
  atomically with `O_EXCL` after threshold and desk-archive preflight but before `_run_audit` can
  reach PIL. It binds the desk SHA, threshold-manifest SHA, configuration, implementation,
  preregistration, and requested output, and remains consumed even if the run crashes.
- Each validation view conserves all targets across out-of-frame, construction-invisible,
  invalid-center, and depth-valid counts; it also reports oriented exclusions and globally
  supported targets represented in that view. Each construction view reports all 1,200 inspected
  grid positions, eligibility exclusions, pair counts, and depth/oriented support.
- No contemporaneous downloader request-timestamp log survives. The bound acquisition record
  `benchmarks/results/20260715_tum_rgbd_ACQUISITION.json` therefore records the archive filesystem
  birth and modification times only as local acquisition/download-completion proxies alongside
  the frozen URL, observed HTTP metadata, length, and SHA-256. They are not represented as a
  server-observed retrieval time. This is a transparent provenance limitation, not a target or
  metric input.

## Question and literature boundary

Does a CPU-first pluggable registered-RGB-D backend produce metric points and depth-Jacobian
normals that are independently consistent across calibrated views of a real static scene?

Incremental Gaussian Triangulation (IGT,
[arXiv 2607.10690](https://arxiv.org/abs/2607.10690)) consumes known-pose RGB-D oriented points and
uses point-to-plane and shortest-axis normal objectives. This experiment validates only the input
contract suggested by that mechanism. It does not fit 2D Gaussians, lift or optimize a scene,
render an image, reproduce IGT, or test utility. A pass authorizes a separate Phase-B
preregistration; it does not authorize an unregistered optimization rerun.

For this repository's bounded source ray `mu(t)=o+t d` and a fixed plane `(p,n)`,

`|n^T(mu-p)| = |n^T d| |t-t_star|`, where `t_star=n^T(p-o)/n^T d`.

With a same-pixel RGB-D point, `t_star` is the sensor depth. A future plane term must therefore be
called sensor-plane-derived, incidence-weighted ray-depth regularization and compared with an
ordinary extra-depth anchor. It is not free 3D plane pulling. Normal alignment remains a separate
hypothesis.

## Frozen sources and isolation

Both archives are official TUM RGB-D Benchmark data, licensed CC BY 4.0. TUM publishes no
cryptographic archive checksum, so the first authorized download must verify the HTTP length and
then freeze a locally computed SHA-256 before any member is decoded.

| role | sequence | URL | HTTP length | ETag | Last-Modified |
| --- | --- | --- | ---: | --- | --- |
| development | `fr1/xyz` | `https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz` | 448204271 | `"1ab70def-4ae2a1dc2ae80"` | `Fri, 30 Sep 2011 15:16:58 GMT` |
| confirmatory | `fr1/desk` | `https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz` | 344011403 | `"1481328b-4ae2a151e2840"` | `Fri, 30 Sep 2011 15:14:33 GMT` |

`fr1/xyz` is development data: implementation defects may be repaired there, but the target
constructor, keyframe rule, metrics, and transfer formulas below do not change. `fr1/desk` is one
confirmatory case study. Its filenames/timestamps/poses may be parsed to apply the pose-only split,
but no desk RGB or depth member may be decoded before the development artifact has frozen all
numeric desk thresholds. The confirmatory audit is run exactly once. Any later desk adaptation is
exploratory and cannot replace the official artifact.

The source manifest records retrieval UTC, URL, HTTP metadata, archive SHA-256, hashes of
`rgb.txt`, `depth.txt`, `groundtruth.txt`, canonical association and split tables, decoder/library
versions, repository revision, dirty status, and the harness/source aggregate hash. Archive paths
must be relative, traversal-free, unique, and consistent with the text manifests.

## Frozen association, calibration, and pose convention

- Parse original timestamp tokens exactly with decimal arithmetic and convert them to integer
  nanoseconds. Reject duplicate timestamps and nonfinite pose values.
- Reproduce the official TUM greedy association semantics for RGB and depth: offset zero; candidate
  pairs satisfy strict absolute timestamp difference below 20,000,000 ns; sort candidates by
  `(absolute_delta, rgb_timestamp, depth_timestamp)`; greedily accept unused endpoints; finally
  sort by RGB timestamp.
- At each associated depth timestamp, interpolate the public ground-truth pose between its two
  bracketing timestamps using linear translation and shortest-arc quaternion SLERP. Require a
  bracketing interval no larger than 20 ms and never extrapolate.
- TUM depth is a 640x480 16-bit map registered one-to-one to RGB. Convert raw value `r` to camera
  optical-axis depth `z=r/5000` metres; raw zero is invalid. TUM already applied the Freiburg-1
  depth-scale correction, so do not apply 1.035 again.
- Use TUM's recommended ROS-default intrinsics for the registered, non-undistorted maps. In TUM's
  integer-center convention these are `fx=fy=525`, `cx=319.5`, `cy=239.5`. In this repository's
  half-pixel convention they are `fx=fy=525`, `cx=320`, `cy=240`. No distortion, undistortion,
  resizing, or depth interpolation is allowed in Phase A.
- A TUM pose contains camera center `C=(tx,ty,tz)` in world metres and camera-to-world optical-frame
  quaternion `(qx,qy,qz,qw)`. Normalize after requiring norm error at most `1e-3`, form `R_c2w`,
  then construct repository extrinsics `R=R_c2w^T`, `t=-R C`. Apply no axis flip. Require
  `R C+t` near zero, `det(R)` near one, and camera project/unproject roundtrip agreement.

## Frozen pose-only keyframes and roles

After RGB/depth association and pose interpolation, sort records by RGB timestamp. Select pose
keyframes without opening any image or depth member:

1. retain the first record;
2. retain a later record when translation from the last retained pose is at least 0.08 m or
   rotation geodesic is at least 8 degrees;
3. require at least 64 retained records;
4. if more exist, select exactly 64 source indices `round(j*(M-1)/63)` for `j=0,...,63`, rejecting
   duplicate rounded indices rather than repairing them.

Split the selected records by zero-based ordinal `j`:

- `H` (sealed future utility): `j mod 8 == 7`, exactly 8 views;
- `V` (independent target audit): `j mod 8 == 3`, exactly 8 views;
- `T` (target construction): all remaining ordinals, exactly 48 views.

Only `T` and `V` depth members may be decoded in Phase A. `H` RGB/depth bytes, all RGB appearance,
and any all-view point cloud are inaccessible to target construction, bounds, filtering,
thresholds, metrics, and stopping decisions. Bounds and extent diagnostics use `T` points only.

## Pluggable backend contract

The implementation adds an `OrientedPointBackend` protocol, an immutable prediction/provenance
record, a canonical world-map validator, and a deterministic registered-depth normal utility. The
backend is keyed by the stable view ID and receives the corresponding image shape and calibrated
camera; no stateful cursor or unordered global cloud is allowed.

Predictions explicitly distinguish camera-z depth, camera-space points, and world-space points,
and distinguish camera/world normal frames. Canonicalization must validate shapes, detached
ownership, finite positive geometry on valid pixels, nonzero normals, optional `[0,1]` confidence,
view/config provenance, and post-cast finiteness. It returns detached cloned world points,
normalized unoriented world normals, validity, confidence, and immutable provenance. Invalid pixels
may contain nonfinite source sentinels but become safe zeros and false validity in the canonical
map. Camera-z depth uses array pixel `(row,column)` at repository coordinate
`(column+0.5,row+0.5)`; it is never Euclidean ray range.

Pure-Torch CPU tests must cover nonidentity camera/world conversion, z-depth versus range,
half-pixel centers, a planar depth fixture, invalid neighborhoods, detached clone ownership,
confidence and enum validation, zero/nonfinite normals, dtype overflow, and provenance mismatch.
The default lifting path is not integrated or changed in Phase A.

## Frozen target constructor

For every `T` view, inspect exactly 1,200 array pixels:

- column `c=8+16i`, `i=0,...,39`;
- row `r=8+16j`, `j=0,...,29`.

The repository pixel coordinate is `(c+0.5,r+0.5)`. At every center use the center depth and the
four depths at `(r,c-2)`, `(r,c+2)`, `(r-2,c)`, `(r+2,c)`:

1. require all five depths in `[0.3,5.0]` m;
2. require every neighbor/center absolute depth difference to be at most
   `max(0.04 m, 0.02*z_center)`;
3. unproject the five camera points with the frozen intrinsics;
4. set `du=P(r,c+2)-P(r,c-2)`, `dv=P(r+2,c)-P(r-2,c)`, and
   `n=normalize(cross(du,dv))`; require cross norm greater than `1e-8`;
5. require `|n dot unit_ray| >= 0.20`, orient the camera normal toward the camera only to make
   hashes deterministic, then transform the center point and normal to world coordinates.

Every eligible grid point becomes a target in `(T role ordinal,row,column)` order. No validation
depth, RGB, fitted primitive, scene result, confidence heuristic, target residual, cross-view PCA,
or post-audit repair may select, remove, reweight, or rematch it. Serialize and hash target
points/normals, validity, source IDs/pixels, calibration, and backend configuration.

## Frozen construction-only visibility and independent audit

For every `V` view, project all frozen `T` targets. Convert a repository pixel coordinate to the
nearest array index with `floor(coordinate)`. At each in-frame array pixel, build a z-buffer using
the minimum positive projected target camera-z. A projected target is construction-visible in that
view when its predicted depth lies within 0.02 m of that construction-only minimum. This selection
uses no `V` depth or residual.

At the selected `V` pixel, independently construct the validation point and normal using that
view's five-depth stencil and all constructor validity rules above. A target is supported only when
at least two distinct `V` views supply a valid audit pair. Per supported target, reduce each metric
by the median over validation supports before reducing across targets.

Report construction-visible pairs with invalid/missing validation depth separately. Also report
association offsets and all counts per `T` and `V` view; no excluded count disappears.

## Frozen metrics

Let:

- `A`: eligible targets divided by `48*1200`;
- `A_min`: minimum eligible fraction over the 48 construction views;
- `S`: targets supported by at least two validation views divided by eligible targets;
- `S_10`: p10 of per-construction-view support fractions;
- `R90`: p90 target-level symmetric point-to-plane residual in metres, where each pair is
  `0.5*(|n_i^T(q_iv-p_i)|+|m_iv^T(p_i-q_iv)|)`;
- `D90`: p90 target-level relative validation depth residual, where each pair is
  `|z_pred-z_V|/max(z_V,0.3)`;
- `C50`: median target-level unoriented normal cosine `|n_i^T m_iv|`;
- `C10`: p10 of that target-level cosine;
- `F`: fraction of valid audit pairs for which the construction target lies in independently
  observed free space: `z_pred < z_V-max(0.05 m,0.03*z_V)`.

Every target-level value is the median over its valid validation pairs. The artifact also reports
raw distributions, pair/support counts, per-view strata, train-only point bounds/extent, archive
and tensor hashes, runtime, environment, code provenance, and all structural invariants.

## Frozen development-to-confirmatory thresholds

Run the target-only audit once on `fr1/xyz`. No manual threshold choice follows. If any development
metric is nonfinite, any input/shape/pose/hash invariant fails, fewer than 64 pose keyframes exist,
or any canonical target tensor mutates, the harness is invalid and `fr1/desk` remains unopened.

Otherwise compute and freeze the desk thresholds mechanically:

```text
A*     = max(0.30, 0.70 * A_xyz)
Amin*  = max(0.10, 0.50 * Amin_xyz)
S*     = max(0.20, 0.60 * S_xyz)
S10*   = max(0.05, 0.50 * S10_xyz)
R90*   = min(0.050, max(0.020, 1.50 * R90_xyz + 0.005)) metres
D90*   = min(0.050, max(0.020, 1.50 * D90_xyz + 0.005))
C50*   = max(0.65, C50_xyz - 0.15)
C10*   = max(0.10, C10_xyz - 0.15)
F*     = min(0.10, max(0.02, 1.50 * F_xyz + 0.01))
```

Write these exact values and the development artifact/hash into an append-only frozen-threshold
manifest before decoding any desk RGB/depth member. The constructor, visibility rule, and code do
not change in response to `xyz` outcomes except for genuine implementation defects demonstrated by
CPU fixtures; any such repair invalidates and replaces the development artifact before desk access.

## Sole confirmatory gate and stopping

The one official `fr1/desk` target audit passes iff all nine comparisons pass independently:

```text
A >= A*       Amin >= Amin*       S >= S*       S10 >= S10*
R90 <= R90*   D90 <= D90*         C50 >= C50*   C10 >= C10*   F <= F*
```

No metric compensates for another. There is no inference, target repair, normal-stencil variant,
threshold relaxation, confirmatory rerun, or optimization in this experiment.

- Failure emits the sole tracked stopped artifact, rejects this direct TUM RGB-D/depth-Jacobian
  backend for `fr1/desk`, and authorizes no plane/normal utility run.
- Pass validates cross-view consistency for this backend on one Kinect sequence and authorizes a
  separate preregistered Phase-B utility experiment with ordinary-depth and shuffled-plane/normal
  controls. It changes no production default.

Either outcome is sensor-consistency evidence, not absolute ground truth, broad scene
generalization, RGB-only inference, or an IGT reproduction.
