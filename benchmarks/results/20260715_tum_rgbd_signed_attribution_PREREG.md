# Preregistration: signed TUM RGB-D occlusion/rigidity attribution

Frozen conceptually on 2026-07-15 at 17:46 Europe/Berlin, while the two archive byte streams were
still downloading and before either archive member list, text manifest, RGB PNG, depth PNG,
selected frame, target, residual, or metric was opened or observed. The final source SHA-256
values and implementation aggregate will be frozen in the acquisition record and development
artifact before the first PNG decode. Repository revision
`2dddca4aff59702341af9faceefa76ad2505dd83`; the dirty worktree is bound by per-file hashes.

Pre-decode numerical clarification (17:55 Europe/Berlin, with both archive downloads still
incomplete and no archive or PNG opened): pose bins use upper-inclusive boundaries, matching
`torch.bucketize(...,right=False)`. At least **four**, not three, usable cells are required because
a 25% maximum cell weight cannot sum to one with only three cells. This repairs an internal
arithmetic inconsistency and changes no observed-data choice.

Pre-decode acquisition freeze (17:56 Europe/Berlin, after both byte streams completed but before
opening either archive): sitting SHA-256 is
`05c071672cda22a668860a935124737a4eb4fa772cbad372e73d5a99ce4be205`; walking SHA-256 is
`1459e9488ac0e61a2ec80dfbc35cfb77942f6d8eabded1c8d26a70be650d0e1d`. Both exact byte lengths
match the HEAD responses above. Filesystem birth/mtime proxies and their evidence limitation are
frozen in `20260715_tum_rgbd_signed_attribution_ACQUISITION.json`.

## Question and boundary

The preceding `fr1/xyz` to `fr1/desk` oriented-point audit retained broad support but failed its
point-to-plane and relative-depth p90 transfer gates through a heavy residual tail. This new,
disjoint experiment asks two narrower questions before any utility test:

1. Does a denser **construction-only** visibility model selectively remove signed residuals in
   which a frozen target lies behind an independently observed surface?
2. After that control, is the remaining bilateral contradiction rate materially higher in TUM's
   fast-motion capture than its slow-motion capture?

The experiment diagnoses the target/audit relation only. It fits no Gaussian, changes no training
loss or production default, decodes no RGB, uses no semantic/dynamic label, and makes no rendering
or utility claim. Any ordinary-depth control or optimization is a later, separately preregistered
experiment.

The design is motivated by recent work that explicitly separates moving scene elements,
visibility, and geometric support rather than treating every residual as one loss: Grassmannian
Splatting I models moving spacetime surfels ([arXiv 2607.10489](https://arxiv.org/abs/2607.10489));
Hallo4D uses motion-aware keyframes and visibility pruning
([arXiv 2607.12752](https://arxiv.org/abs/2607.12752)); and IGT consumes known-pose RGB-D oriented
points for plane-based surface pulling ([arXiv 2607.10690](https://arxiv.org/abs/2607.10690)). This
audit tests a prerequisite suggested by those mechanisms; it is not a reproduction.

## Frozen sources and one-shot isolation

Both sources are official TUM RGB-D Benchmark CC BY 4.0 archives. TUM describes `fr3/sitting_xyz`
as two seated people talking and gesturing slightly with xyz camera motion (slowly moving dynamic
objects), and `fr3/walking_xyz` as two people walking through the office with xyz camera motion
(quickly moving objects in large visible regions).

| role | sequence | URL | HTTP bytes | ETag | Last-Modified |
| --- | --- | --- | ---: | --- | --- |
| development | `rgbd_dataset_freiburg3_sitting_xyz` | `https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_xyz.tgz` | 775406859 | `"2e37c50b-4c6b1a7e29cc0"` | `Tue, 07 Aug 2012 19:03:55 GMT` |
| confirmatory | `rgbd_dataset_freiburg3_walking_xyz` | `https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz` | 527550055 | `"1f71c667-4c6b17936fb00"` | `Tue, 07 Aug 2012 18:50:52 GMT` |

Archive download, byte hashing, safe tar-header traversal, and reading `rgb.txt`, `depth.txt`, and
`groundtruth.txt` are metadata operations. Semantic PNG decoding is outcome access. Sitting may be
decoded only after this protocol, acquisition record, implementation, dependency hashes, and
synthetic tests are frozen. Walking filenames/timestamps/poses may then be parsed, but its first
depth PNG decode consumes the sole confirmatory attempt through an atomic repository seal. No RGB
or H-role depth payload is decoded. A failed development gate leaves walking pixels unopened.

Walking is a deliberately untouched confirmation source, not a tuning set. Any code or protocol
change after its attempt seal makes later results exploratory and requires another new sequence.

## Reused calibration, association, targets, and roles

The new standalone harness imports the sealed `benchmarks/tum_rgbd_oriented_validity.py` only for
its already tested, source-independent mechanics; that file remains unchanged and is included in
the new implementation aggregate.

- Parse timestamps exactly to integer nanoseconds. Greedily associate RGB/depth at strict
  absolute difference below 20 ms and interpolate the public pose at the depth timestamp only
  across a bracketing interval no longer than 20 ms.
- Use registered 640x480 depth `z=raw/5000` metres, with zero invalid and valid audit depths in
  `[0.3,5.0]` m. Freiburg-3 scale correction is already applied. Use TUM's recommended registered
  map intrinsics `fx=fy=525`, integer-index `cx=319.5,cy=239.5`, represented by
  `cx=320,cy=240` in this repository's half-pixel convention. Apply no re-undistortion.
- Reuse the pose-only keyframe rule: retain the first association, then another at at least 0.08 m
  translation or 8 degrees rotation from the last retained pose; half-up uniformly select 64.
  Ordinals `j mod 8 == 7` are eight sealed `H` views, `j mod 8 == 3` are eight independent `V`
  views, and the remaining 48 are construction `T` views.
- Reuse the stride-16 audit target grid and the exact five-depth oriented eligibility rule from
  the prior harness. This deliberately holds target identity fixed while changing only the
  construction visibility model. Target order is `(T role ordinal,row,column)`.
- Decode exactly the selected 48 T and eight V depth maps once, in physical tar order. Separate
  construction and validation capability maps must be exact and disjoint. The whole archive hash
  binds all forbidden RGB/H payloads.

## Frozen construction-only visibility arms

For every T view, additionally unproject all finite depths in `[0.3,5.0]` at array rows and columns
`0,8,...` into a dense world point set. Define nested construction sets:

- `sparse`: the eligible audit targets;
- `dense_T`: the explicit concatenation of `sparse` and every valid stride-8 T point.

For each V camera and arm `a`, project its construction set and form a minimum camera-z buffer per
array pixel. A target is construction-visible iff its own projection is finite, positive and
in-frame and `z_pred <= z_min + 0.020 m`. Projection coordinates map by `floor`; exact bounds are
`0<=u<640, 0<=v<480`. No V depth, V normal, RGB, residual, semantic mask, or post-hoc neighborhood
enters either z-buffer. `dense_T-visible` must be a subset of `sparse-visible` bit-for-bit.

The dense set, targets, raw visibility masks, capability lists, selected frames, and source
manifests are serialized or cryptographically hashed. Pair counts must conserve every target in
every V view across out-of-frame, invisible, invalid observed depth, and depth-valid states.

## Signed audit and target-balanced summaries

At the target's projected V pixel, let `z_obs` be the independently decoded center depth. For
finite `z_obs` in `[0.3,5.0]`, define

```text
d = z_pred - z_obs
r = d / z_obs
tau = max(0.050 m, 0.03*z_obs)
```

- `d > tau`: positive / behind-observed / **occlusion-like**;
- `d < -tau`: negative / in-front-of-observed / observed-free-space contradiction;
- otherwise neutral.

The signs are descriptive: a nearer observation is not ground-truth occlusion, and motion or
sensor failure can produce either sign. V depth labels residuals only after both construction
visibility masks are frozen; it never filters or repairs them.

For arm `a`, a target is supported with at least two depth-valid V observations. For each
supported target, average positive and negative indicators over its observations, then average
those rates equally over targets. Let `P_a+`, `P_a-`, and `C_a=P_a++P_a-` be those
target-balanced rates. `D90_a` is the p90 across supported targets of their median
`abs(r)`. CPU float64 linear quantiles are used. Pair-weighted rates are diagnostics only.

## Development occlusion-like gate

Let `S` be sparse-visible depth-valid pairs, `D` dense-visible depth-valid pairs, and
`R=S\\D`. For targets having at least one removed and one retained observation, define target-level
positive/negative rates in each subset and

```text
E+  = mean_i(P_i,R+ - P_i,D+)
E-  = mean_i(P_i,R- - P_i,D-)
RR+ = (mean_i P_i,R+ + 0.001) / (mean_i P_i,D+ + 0.001)
```

Structural support requires all of:

- at least 20,000 sparse depth-valid pairs;
- dense pair retention `|D|/|S| >= 0.70`;
- at least max(1% of `S`, 1,000) removed depth-valid pairs;
- at least 100 targets with both removed and retained observations;
- at least 500 dense-supported targets.

The occlusion-like gate passes only if support passes and all effect floors hold:

```text
E+ >= 0.10
RR+ >= 2.0
E+ - max(E-,0) >= 0.05
P_sparse+ - P_dense+ >= max(0.01, 0.15*P_sparse+)
P_dense- <= P_sparse- + 0.01
D90_dense <= 1.05*D90_sparse
```

This requires positive-sign selectivity, not merely deletion of difficult pairs. Sitting failure
stops the experiment and leaves walking PNGs unopened. No threshold, stride, tolerance, target,
or metric changes after sitting outcome access.

## Time and pose sensitivity

For target source timestamp `t_i`, V timestamp `t_v`, and selected-source span
`T=t_max-t_min`, use normalized gap `g=abs(t_i-t_v)/T`. `near` is `g<=0.20`, `far` is
`g>=0.60`, and the middle is reported but does not enter the contrast. On dense-valid pairs,
compute each target's `C_far-C_near` where both strata exist and average targets equally to obtain
`temporal_delta`.

Time and camera pose are correlated. A mandatory sensitivity jointly bins source/V camera-center
distance at upper-inclusive bins ending at `0.10,0.25,0.50,1.0,inf` metres and rotation at
upper-inclusive bins ending at `5,15,30,inf` degrees. A cell is usable only with at least 250 pairs and 50 targets in each
temporal stratum. Cell target-balanced far-minus-near effects are combined by the smaller stratum
target count, with no cell above 25% weight. At least four cells and 1,000 pairs per temporal
stratum are required; otherwise this sensitivity is `not_estimable`, never silently zero.

The artifact also reports fixed pose-bin contradiction rates and a cross-sequence standardized
sensitivity over bins supported in both captures. These are diagnostics because separate captures
cannot perfectly isolate object motion.

## Confirmatory motion-regime and transfer gates

If sitting passes, freeze its append-only artifact and decision manifest, validate the unchanged
implementation aggregate, then atomically consume walking before its first PNG decode. Walking
first must independently pass the same occlusion-like gate. Strong motion-regime-associated
rigidity evidence additionally requires, on unchanged `dense_T` visibility:

```text
C_walking - C_sitting >= 0.05
C_walking / max(C_sitting,0.01) >= 1.25
D90_walking / D90_sitting >= 1.25
```

and at least one signed/time discriminator:

```text
P_walking- - P_sitting- >= 0.02
```

or

```text
temporal_delta_walking >= 0.03
temporal_delta_walking - temporal_delta_sitting >= 0.02
and the estimable pose-conditioned walking temporal effect has the same sign.
```

Classification is frozen:

- walking occlusion transfer and rigidity gate both pass: `TWO_MECHANISMS_SUPPORTED`;
- exactly one passes: `PARTIAL_ATTRIBUTION`;
- neither passes: `ATTRIBUTION_REJECTED`;
- any invariant/support failure: `INDETERMINATE`.

## Uncertainty, stopping, and permitted claims

Use 1,000 deterministic target-cluster bootstrap replicates with seed `20260715`, resampling
targets and retaining all their V observations. Cross-sequence resampling is independent. Report
two-sided 90% intervals (equivalently the relevant one-sided 95% bound). In addition to every
point-estimate floor, the lower bound for each asserted positive effect must be above zero; a ratio
uses its log. The nonincrease and D90 safeguards remain point-estimate constraints. Empty or
nonfinite required populations invalidate the harness.

Synthetic, TUM-value-free tests may repair a defect before sitting decode. After sitting decode,
only a correction proven on such a fixture to make implementation match this frozen text is
allowed, requires retaining the superseded sitting artifact, and still cannot inspect walking.
After the walking seal, no correction can produce another confirmatory result on walking.

Positive language is limited to “consistent with construction-predicted occlusion” and “associated
with the fast-motion capture regime.” The captures differ in people, duration, geometry, camera
speed/path, visibility, and missing-depth behavior. Even with pose/time strata, the experiment
cannot identify a causal object-motion effect or ground-truth occlusion.
