# Preregistration: Stage-1 scalar/color semantics factorial

## Chronology, question, and scope

Frozen at `2026-07-16T03:17:22+02:00`, before implementation, a pilot, use of any
official seed, construction of an official candidate arm, lift, refinement, or outcome for this
experiment.

The qualified Stage-1 gauge audit established a narrow representation-contract problem: on its
three CPU synthetic seeds, product-preserving changes to the fitted `(weight,color)` factorization
left native Stage-1 RGB reconstructions equivalent but materially changed both unmerged Depth and
Carve lifts. That audit deliberately did not identify a correct representation or measure final
utility. This new, append-only experiment asks two separate questions in order:

1. **Mechanism:** can a gauge-invariant scalar and a source-observation color be constructed at
   the Stage-1/Stage-2 boundary such that their fields, retention, coverage, and ordinary
   production lifts are invariant under the same exact gauges?
2. **Utility:** at an exact matched primitive count and fixed refinement budget, which part of the
   boundary matters for held-out reconstruction: the scalar interpretation, the color
   interpretation, their interaction, or neither?

This is a two-phase deterministic CPU synthetic experiment. Phase A is a mandatory algebraic and
production-path validity gate. Phase B is forbidden unless every Phase-A gate passes and an
independent scientist review confirms that pass from the raw archive. Phase B is a fresh-seed
`2 x 2` causal factorial with train/held-out isolation, two unmerged lifter backends, exact
per-source-view capacity matching, and fixed-topology refinement.

This protocol does not change production source, defaults, CLI behavior, public documentation, or
the ARA. It makes no real-data, CUDA/gsplat, speed, memory, general physical-occupancy, or surface-
albedo claim. A positive result cannot change a default without a post-result independent results
audit and separate integration decision. An invalid or negative result may not be repaired by
tuning this namespace.

## Prior evidence and literature boundary

The prior result and its independent audit are background evidence only:

- `benchmarks/results/20260716T003140Z_cpu_stage1_weight_gauge_audit.json`;
- `benchmarks/results/20260716T003140Z_cpu_stage1_weight_gauge_audit_AUDIT.md`, verdict
  `QUALIFIED`.

The audit's confirmed narrow result licenses testing a repair, not selecting one. Its qualification
also fixes two requirements here: decisive raw tensors must be independently decodable, and an
identical coverage tensor must have a common content-hash definition wherever it appears. No
numerical outcome, threshold distance, or official seed from that audit determines a candidate,
utility seed, or utility gate below.

The 2026-07-12 through 2026-07-16 Scholar Inbox digest supplies analogies, not validation.
[AsySplat](https://arxiv.org/abs/2607.10995) motivates separating geometry-bearing and
appearance-bearing attributes; [SalientGS](https://arxiv.org/abs/2607.11285) motivates treating
allocation/salience separately from color; [MAC-Splat](https://arxiv.org/abs/2607.10792) motivates
an explicit multi-attribute consistency contract; and
[Incremental Gaussian Triangulation](https://arxiv.org/abs/2607.10690) motivates checking the
geometry-facing boundary rather than only final pixels. None studies the exact transform, scalar,
source-RGB sampling rule, lifters, or gates used here. Their methods, losses, matching, and
training machinery are not implemented or claimed.

The closer primary-source control is
[GaussianImage](https://github.com/Xinjie-Q/GaussianImage) (Zhang et al., ECCV 2024,
[arXiv:2403.08551](https://arxiv.org/abs/2403.08551)). Its official Cholesky implementation has
eight learned parameters per component, fixes its raster opacity tensor to ones, and directly
optimizes one three-vector color/feature. The repository's separately trainable scalar `w` is
therefore a local ninth-parameter extension, not an upstream GaussianImage contract. This fact
motivates the upstream-style `unit_weight__a_amp=(1,a)` Phase-A boundary control below; it does not
reproduce upstream optimization dynamics or establish that unit downstream coverage or
amplitude-as-SH is best for this pipeline.

## Frozen notation and candidate semantics

For fitted component `i` in source view `v`, native Stage 1 renders the additive term

`a_vi * exp(-q_vi/2) * 1[q_vi<12]`, where `a_vi = w_vi c_vi`.

All following arithmetic is float32 unless a reduction is explicitly float64. Preserve fitted
`xy`, `chol`, component order, and source-view order. Define:

- fitted scalar `w_vi` and fitted RGB factor `c_vi`;
- additive amplitude `a_vi = w_vi * c_vi`;
- gauge-invariant peak amplitude `m_vi = max_k a_vi,k`;
- source observation `o_vi = bilinear_sample(I_v, xy_vi)` using the repository's exact
  pixel-center convention: subtract `0.5`, clamp to the valid image interior, and perform standard
  four-neighbor bilinear interpolation;
- normalized amplitude/chroma `h_vi = a_vi/m_vi` when `m_vi>0`, and the exact zero vector when
  `m_vi=0`.

Because fitted `w,c` lie in `[0,1]`, `a,m,o,h` must be finite and in `[0,1]`. The scalar `m` is a
gauge-invariant component-amplitude or salience candidate; it is **not** asserted to be true
physical opacity, occupancy, visibility, or confidence. Likewise, `o` is the composite source RGB
observed at the component center. It can contain background, occlusion, and contributions from
overlapping components and is **not** asserted to be surface albedo.

This is an attribute-routing experiment at the already-fitted boundary. The immutable original
`a` remains the Stage-1 reconstruction field. The routed pair `(m,o)` is consumed only by Stage-2
coverage/retention/tunnel/SH code and is not required to satisfy `m*o=a` or to re-render Stage 1.
The exact-product controls below make that distinction measurable. All `max`, division, and image
sampling happen on detached post-fit tensors with no gradient. Consequently, a smooth maximum
unit (SMU) cannot affect this protocol; a differentiable fit-time gauge regularizer or smooth-max
parameterization would be a different experiment.

Phase B has exactly these four arms, constructed independently in every fitted source view:

| Arm | scalar passed to Stage 2 | RGB passed to Stage 2 | role |
|---|---|---|---|
| `w_fit__c_fit` | `w` | `c` | current/current (`00`) |
| `m_amp__c_fit` | `m` | `c` | scalar-only (`10`) |
| `w_fit__rgb_obs` | `w` | `o` | color-only (`01`) |
| `m_amp__rgb_obs` | `m` | `o` | full candidate (`11`) |

No arm changes `xy`, `chol`, order, image, camera, depth, mask, or source view. The auxiliary
`m_amp__h_norm=(m,h)` representation appears only in Phase A as an algebraic positive control:
`m*h=a`, so it is both gauge-invariant and product-equivalent to the fitted Stage-1 source. It is
not a fifth utility arm because normalized amplitude is not the proposed physical RGB semantics
and adding it would break the frozen factorial. Phase A also contains
`unit_weight__a_amp=(1,a)`, the upstream-style eight-parameter boundary analogue. The amplitude
still comes from this repository's already-completed nine-parameter fit, so this is not an upstream
fit comparison. It changes scalar and color jointly relative to the current boundary, cannot
estimate either factorial main effect, and is restricted to mechanism/reporting rather than
becoming a fifth utility arm. This keeps Phase B as the sharper orthogonal test of scalar routing
versus source-observation-color routing.

## Common execution environment and synthetic split

- CPU only: `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, Torch intra-op
  threads `4`, deterministic algorithms enabled, no optional StructSplat or gsplat import, and no
  network access during either scientific command.
- Phase-A mechanism seeds are exactly `[1103,2203,3301]`.
- Phase-B utility seeds are exactly `[4409,5519,6637]`. The two sets are disjoint from each other
  and from the prior audit's `[0,1,2]`.
- For each seed, call
  `make_synthetic_scene(n_gaussians=40,n_cameras=12,image_size=48,seed=seed)` exactly once.
  Freeze original training indices `[0,1,2,4,5,6,8,9,10]`, held-out indices `[3,7,11]`, and the
  local-to-original mapping. The physically subset train scene must have local
  `training_views=[0,1,2,3,4,5,6,7,8]` in that order and no testing views.
- Fit only the physically subset nine-view training scene. Before either lifter, create a
  field-minimal training scene with `gt_gaussians=None`, the nine aligned images/cameras/depths,
  no held-out fields, and only the points, visibility, bounds hint, center, and extent consumed by
  the frozen lifters.
- Fit exactly once per seed and share immutable fitted tensors across all representations and
  lifters:
  `FitConfig(n_gaussians=150,max_gaussians=5000,iterations=120,backend="native",
  adaptive_density=True,growth_waves=5,relocate_fraction=0.0,
  structsplat_renderer="auto",lr=0.01,grad_init_mix=0.7,row_chunk=64,log_every=50,
  convergence_patience=0,convergence_tol=0.05,convergence_check_every=25)` and
  `fit_views(train_scene.images,config,seed=seed,masks=train_scene.masks)`.
- Require nine fitted sets of exactly 150 finite components. Validate all public field shapes and
  ranges, positive Cholesky diagonals, source ordering, and train-scene alignment before a
  representation is formed. Hash all fitted fields, fit histories, train-only inputs, retained
  world priors, local/original indices, and center/extent.
- No refit, extra Stage-1 iteration, component reorder, representation-dependent clamp, mask
  change, source-view substitution, or adaptive hyperparameter is permitted.

The full synthetic helper necessarily constructs all twelve views. In Phase A the held-out views
are discarded immediately after physical subsetting and are never subsequently read, hashed,
rendered, or scored. In Phase B, held-out data are moved immediately into a guarded payload whose
tensor/camera/depth accessors raise until an explicit global unlock. The payload is not passed to
fit, lift, selection, training, validation, checkpointing, or a pre-unlock hash. Unit tests must
prove the guard raises. All `3 seeds x 2 backends x 4 arms = 24` final models, natural unpruned
lifts, matched initial models, hashes, schedules, counts, and pre-held-out validity gates must be
frozen before one global unlock permits final reporting. No held-out statistic is printed or
serialized before that unlock.

## Raw tensor and hash contract

Every valid scientific JSON has one uncompressed `numpy.savez` sidecar. It must be readable with
`allow_pickle=False`; object arrays and Python-pickled data are forbidden. Arrays use stable,
slash-separated names and include, at minimum, all fitted fields, gauge fields, representation
fields, images/depths needed to recompute a reported metric, production and independent-lifter
output fields, source-key arrays, coverage/volume tensors, initial/final 3D fields, checkpoint
renders, held-out target renders, and every decisive source/coverage/lift/final render.
Nullable logical fields use a numeric `<name>/value` array plus a boolean `<name>/defined` mask;
undefined values are exact zero in the value array. NaN is not a null encoding, and all raw
floating arrays must be finite.

The JSON contains a sorted manifest entry for every array: name, dtype, shape, byte length, and
`raw_content_sha256`. Sidecar arrays must be numeric or boolean. For array `x`, define
`dtype_token=np.dtype(x.dtype).newbyteorder("<").str.encode("ascii")`,
`shape_bytes=np.asarray(x.shape,dtype="<i8").tobytes(order="C")`, and `data_bytes` as the
C-contiguous bytes after converting multibyte data to the corresponding little-endian dtype. The
content digest is exactly

`SHA256(dtype_token || b"\0" || shape_bytes || b"\0" || data_bytes)`.

It deliberately excludes the archive path and logical array name, so two asserted-identical
tensors at different sites must have the same content digest. A separate sorted collection digest
is SHA-256 of UTF-8
`json.dumps(sorted([[name,content_digest],...]),separators=(",",":"),ensure_ascii=True)`.
The JSON and companion result note bind the completed `.npz` file
SHA-256; the earlier once-only marker can bind only its prospective path. Raw metric reductions
use float64 and are serialized alongside their derived values. The independent scientist pass
must recompute decisive values from arrays, not trust stored summaries.

All source coverage maps use the one logical content domain
`coverage/seed=<s>/gauge=<g>/arm=<a>/view=<v>`. A standalone coverage record and a Carve sidecar
record referring to the same tensor must point to the same manifest array and content digest;
rehashing it under a different label is forbidden.

## Phase A: gauge-repair mechanism prerequisite

### Gauge construction and source equivalence

For every fitted component, compute identity amplitude `a=w*c` once. Construct exactly the prior
audit's three valid gauges, independently on each Phase-A seed:

1. `identity`: `(w,c)` unchanged;
2. `unit_weight`: `(1,a)`;
3. `peak_color`: let `p=max_k(a_k)` and use `(p,a/p)` when `p>0`, exact zeros otherwise.

Copy `xy/chol` bit-for-bit and preserve order. Require finite bounded fields and amplitude agreement
with identity at maximum absolute error `<=1e-7` and maximum relative error `<=1e-6`, where the
relative comparison is made only for identity amplitude magnitude `>1e-8`.

Before coverage, retention, representation mapping, or a lifter is permitted, render the three
gauges with
`render_gaussians_2d(height=48,width=48,row_chunk=64)`, black additive background, no clamp, and
no gradient. Across all `3 seeds x 9 views x 2 transforms = 54` transformed source views require:

- finite raw renders;
- maximum absolute transformed-minus-identity RGB error `<=5e-6`;
- float64 `sum(abs(delta))/sum(abs(identity)) <=1e-6` with positive finite denominator;
- PSNR `>=100 dB`, using an MSE floor of `1e-12` only for this equivalence diagnostic.

This prerequisite is global. One failure writes a fail-closed invalid Phase-A artifact containing
only preparation, transform, equivalence, and already-formed raw arrays. It must not compute or
expose candidate coverage, retention, lifts, or a Phase-B decision.

### Candidate construction and factorial-integrity checks

After global source equivalence, independently compute `a^g,m^g,h^g,o^g` from each named gauge
using that gauge's fields and the immutable source image. Require, relative to `identity`:

- bit-exact `xy/chol` and bit-exact sampled `o`;
- `m` maximum absolute error `<=1e-7` and maximum relative error `<=1e-6` where identity
  `m>1e-8`;
- `h` maximum absolute error `<=2e-6` and maximum relative error `<=2e-5` where identity
  `|h|>1e-6`;
- exact zero handling for `m=0`, and finite `[0,1]` fields everywhere;
- `m*h` versus identity `a` maximum absolute error `<=2e-7` and relative error `<=2e-6` where
  `|a|>1e-8`.

Construct `m_amp__rgb_obs`, `m_amp__h_norm`, and `unit_weight__a_amp=(1,a^g)` from every gauge.
Their `xy/chol` must be bit-exact. For the unit control, scalar must be bit-exact one and color
must agree with identity `a` at maximum absolute error `<=1e-7` and maximum relative error
`<=1e-6` where `|a|>1e-8`. All preceding field tolerances must pass for every
seed/view/component before a lifter.

On the identity gauge only, construct the four utility arms and assert the factorial mechanically:

- `00` is bit-exact to fitted fields;
- `10` differs from `00` only in scalar;
- `01` differs from `00` only in color;
- `11` has exactly the scalar field of `10` and color field of `01`;
- every arm preserves `xy/chol/order` bit-for-bit;
- at least `10%` of all components have `|m-w|>1e-7`, and at least `10%` have any
  `|o-c|>1e-7`, both per seed and in the seed-tagged pool.

The last condition is an identifiability gate only. It does not use quality and cannot be relaxed.

### Coverage, retention, and production-lifter invariance

For all three Phase-A representations (`m_amp__rgb_obs`, `m_amp__h_norm`,
`unit_weight__a_amp`) and all three gauges:

- render `C=1-exp(-sum_i r_i G_i)`, where routed scalar `r=m` for the first two
  representations and exact `r=1` for `unit_weight__a_amp`, with the unchanged
  `render_gaussian_coverage_2d(...,48,48,row_chunk=64)`;
- require identity/transformed maximum absolute coverage error `<=2e-6`, float64
  `sum(abs(delta))/sum(abs(identity)) <=1e-6`, positive finite denominator, and zero pixels for
  which `(C_g>0.40) xor (C_identity>0.40)`;
- require exact equality of strict retained source keys under `r>0.05`.

Run ordinary, fresh, **unmerged** production lifts for all three representations from every gauge
under both exact configurations:

`DepthLifter(backend=GroundTruthDepth(train_scene.gt_depths),sh_degree=0,min_weight=0.05,
init_opacity=0.1,normal_thickness=0.15,covariance_mode="surface",isotropic_sigma=None,
robust_depth_gradients=True,merge=False,merge_voxel_frac=0.01)`.

`CarveLifter(grid_res=48,bounds_scale=0.5,min_views=2,hull_fraction=0.85,
color_std_sigma=0.20,color_match_sigma=0.35,coverage_thresh=0.40,samples_per_ray=64,
min_score=0.05,min_weight=0.05,merge=False,merge_voxel_scale=1.0,init_opacity=0.1,
sh_degree=0)`.

Each `(seed,gauge,representation,backend)` calls the ordinary production lifter exactly once. An
independent Depth mask reconstruction and Carve diagnostic sidecar recover immutable source keys
and all current arithmetic without a second production lift. Require production/independent
ordered-key and output parity for every call: output counts and keys exact, and means, covariance,
opacity, and SH within `atol=2e-6,rtol=2e-5`. Compare covariance instead of quaternion signs.
Require every global output nonempty.

For each backend, representation, seed, and transformed gauge relative to identity require:

- exact emitted source-key equality and exact count equality;
- means, covariance, opacity, and SH within `atol=2e-6,rtol=2e-5`;
- all nine raw Torch renders finite under
  `TorchRasterizer(sh_color_activation="hard",kernel_support_mode="hard",
  visibility_margin_sigma=3.0)`, degree 0, black background;
- raw color render maximum absolute error `<=5e-6` and float64
  `sum(abs(delta))/sum(abs(identity)) <=1e-6` with positive finite denominator;
- zero `0.40` coverage-threshold crossings and zero `0.05` retention-threshold crossings.

Serialize raw color, alpha, and accumulated-depth tensors and deltas. Only the color thresholds are
decision gates; alpha/depth must be finite and are diagnostics. The Carve sidecar must also
serialize its coverage/volume tensors and prove that its source coverage references exactly the
common manifest arrays described above.

### Phase-A decision

`phase_a_pass=true` if and only if **every** source-equivalence, bounded-field,
factorial-identifiability, candidate-field, coverage, retention, production/independent parity,
nonempty-output, output-key, output-field, render, raw-manifest, and routing condition passes for
all three seeds, both transformed gauges, all three representations, and both backends. There is no
averaging, majority rule, tolerance repair, or backend rescue.

Interpret failures without tuning:

- source-equivalence failure: invalid gauge test; no candidate mechanism was reached;
- candidate-field or coverage/key failure: the proposed map is not operationally gauge-invariant;
- production/sidecar parity failure: invalid measurement implementation;
- `m_amp__h_norm` or `unit_weight__a_amp` fails while `m_amp__rgb_obs` passes: an
  exact-product/upstream-style control failure still blocks utility;
- `m_amp__rgb_obs` fails while both exact-product controls pass: the proposed source-observation
  color routing is not production-invariant and utility remains forbidden;
- all pass: the repair mechanism is qualified to enter the separately gated utility phase, with
  no utility or default claim yet.

## Phase-A artifacts, seal, commands, and independent gate

Fresh append-only namespace:

- future harness: `benchmarks/stage1_semantic_factorial.py`;
- focused tests: `tests/test_stage1_semantic_factorial.py`;
- outcome-free implementation review:
  `benchmarks/results/20260716_stage1_semantic_factorial_IMPLEMENTATION_REVIEW.md`;
- seal: `benchmarks/results/20260716_stage1_semantic_factorial_SEAL.json`;
- Phase-A marker:
  `benchmarks/results/20260716_stage1_semantic_factorial_PHASE_A_ATTEMPT.json`;
- valid outputs: `<UTC>_cpu_stage1_semantic_factorial_mechanism.json`, matching
  `_mechanism_RAW.npz`, and matching `_mechanism_RESULT.md`;
- invalid siblings: replace terminal `_mechanism.json`, `_mechanism_RAW.npz`, and
  `_mechanism_RESULT.md` with `_mechanism_invalid.json`, `_mechanism_invalid_RAW.npz`, and
  `_mechanism_invalid_RESULT.md` respectively;
- independent review: matching `_mechanism_AUDIT.md` and
  `_mechanism_SCIENTIST_REVIEW.json`.

The implementation review must be complete and say `PASS` before sealing. The sole outcome-free
seal command is:

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python benchmarks/stage1_semantic_factorial.py seal --output benchmarks/results/20260716_stage1_semantic_factorial_SEAL.json`.

Seal creation runs, in order, exactly `.venv/bin/python -m ruff check .`,
`.venv/bin/python -m ruff format --check .`, `.venv/bin/python -m pytest -q -m "not slow"`,
`.venv/bin/python scripts/docs_sync.py`, and `git diff --check`. It records literal commands,
complete outputs, return codes, and output hashes and refuses to seal on a failure. The seal binds
this preregistration, implementation review, harness, focused tests, every repository-owned loaded
source, complete revision/dirty diff, environment, configs, artifact rules, and both command
templates. Focused tests may use only tiny nonofficial fixtures; neither seal creation nor tests
may construct, fit, transform, lift, or render an official seed.

The sole Phase-A scientific command is:

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python benchmarks/stage1_semantic_factorial.py mechanism --seal benchmarks/results/20260716_stage1_semantic_factorial_SEAL.json --output benchmarks/results/<fresh-UTC>_cpu_stage1_semantic_factorial_mechanism.json`.

Before creating the marker with exclusive creation, the harness derives every valid/invalid JSON,
raw, and note path; requires all to be absent; rehashes the seal and all bound sources; and binds
all possible paths into the marker. A valid run writes only the valid triple. A failed run writes
only the invalid triple. Interruption consumes the marker. It may not redirect after observing an
outcome, overwrite, resume, or retry in this namespace.

After a valid Phase-A result, a reviewer independent of implementation and execution must use the
repository results-audit protocol and recompute the global decision from the `.npz`. The machine
review must bind the preregistration, seal, JSON, raw archive, harness, tests, and audit Markdown
SHA-256; record reviewer identity and UTC time; recompute every decisive gate; have verdict
`PASS`; and set `phase_b_authorized=true`. A qualified, inconclusive, failed, or missing review
does not authorize Phase B. The review cannot amend thresholds or excuse a missing raw tensor.

## Phase B: fresh-seed exact-capacity utility factorial

### Preparation and arms

Phase B uses only `[4409,5519,6637]`, repeats the common train-only construction and one native fit
per seed, and forms the four arms from the **identity** fitted representation. It does not recreate
or select among the Phase-A gauges. All field, factorial-integrity, range, fit-count, source-order,
and raw-manifest checks remain mandatory. Phase-A results supply authorization only; no Phase-A
metric, tensor, model, count, or threshold is loaded into a Phase-B arm.

For each seed, arm, and backend, run exactly one ordinary unmerged Depth or Carve lift using the
exact Phase-A configurations. Independently reconstruct source keys and require the same
production parity tolerances before capacity matching. Record natural unpruned counts, exact
source keys, per-view quotas, set intersections/Jaccards, and step-zero train renders as diagnostics
only. No merge or density operation is permitted.

### Exact per-view primitive-capacity control

For seed `s`, backend `b`, arm `a`, and local training view `v`, let `E_s,b,a,v` be the exact source
keys emitted by the ordinary lift. Freeze

`K_s,b,v = min_a |E_s,b,a,v|`.

Require `K_s,b,v>=8` for every view and `sum_v K_s,b,v>=270` for every seed/backend. A violation
invalidates Phase B before refinement; it does not permit removing a view, changing a threshold,
or lowering the floor.

Rank every available source key using one common, gauge-invariant integrated scalar-density mass.
For its original fitted 2D component, let `Sigma=L L^T` and compute in float64

`rho_i = float64(m_i) * float64(L_i,11) * float64(L_i,22)`.

Because the Cholesky diagonal is positive, `sqrt(det(Sigma))=L_11 L_22`. Thus `rho` is proportional
to the exact continuous-plane integral of `m_i G_i`: the omitted constant is `2*pi` for the
untruncated Gaussian and `2*pi*(1-exp(-6))` under the repository's common `q<12` support. It is not
claimed to equal the discretized finite-canvas pixel sum. Compute `m` once from the immutable
identity fitted amplitude and use the same `rho` table for all four arms and both backends. Within
each `(s,b,a,v)`, sort available keys by descending `rho`; break an exact float64 tie by ascending
`SHA256("stage1-semantic-factorial-v1|<seed>|<backend>|<local-view>|<component>")`, then ascending
component index. Select exactly the first `K_s,b,v`, then concatenate views in ascending local-view
order and keys in ascending original-component order.

This contribution/area score is fixed before any lift or quality result and uses no source RGB,
arm-specific scalar, Carve score, 3D position, depth, held-out field, or final metric. `K` is the
minimum natural production output count rather than the identity count because the matched budget
must be achievable in every arm without synthesizing a missing production placement or bypassing
the frozen lifter. The natural identity count is still reported. The quota is symmetric across
arms and fixed before refinement. Serialize the complete `rho`, digest, rank, availability, and
selected-key table.

Within each seed/backend, all four selected initial models must therefore have exactly
`K_s,b=sum_v K_s,b,v` primitives and identical per-view count vectors. Source-key identity may
differ and is a reported mechanism diagnostic. Initial and final topology/count trajectories must
remain exact and equal across arms.

### Fixed-topology refinement and RNG contract

Refine every matched initialization for exactly 120 steps with:

`TrainConfig(iterations=120,lr_means=1.6e-4,lr_quats=1e-3,lr_scales=5e-3,
lr_opacity=5e-2,lr_sh=2.5e-3,lr_sh_rest=1.25e-4,ssim_lambda=0.2,
rasterizer="torch",device="cpu",densify=False,density_strategy="classic",eval_every=30,
target_sh_degree=0,sh_degree_interval=None,use_masks=False,outside_alpha_lambda=0.01,
mask_alpha_lambda=0.05,random_background=False,opacity_reg=0.0,scale_reg=0.0,
packed=False,antialiased=False,sh_color_activation="hard",collect_sh_color_diagnostics=False,
kernel_support_mode="hard",collect_kernel_support_diagnostics=False,
visibility_margin_sigma=3.0,validate_render_finite=True,
quaternion_update_policy="current",seed=<train-seed>)`.

Only the nine-view training scene is passed to `Trainer`. There is no density controller action,
primitive addition/removal/merge, SH growth, random background, mask, early stopping, adaptive
schedule, arm-specific learning rate, or checkpoint selection. Checkpoints at steps
`0,30,60,90,120` are reporting snapshots; step 120 is always final.

Use backend codes `Depth=0`, `Carve=1` and exact trainer seed

`train_seed(s,b) = 2_000_000 + 10*s + backend_code(b)`.

All four arms of a seed/backend use the same seed. The harness independently constructs one
`schedule_generator=torch.Generator(device="cpu").manual_seed(train_seed)`, then builds the
expected 120-view schedule by repeatedly calling
`torch.randint(0,9,(1,),generator=schedule_generator)` and converting its sole value to `int`.
Require
every recorded schedule to be bit-identical to this expectation and to the other arms, require all
nine training views to occur at least once, and serialize the int64 schedule and content hash.
Any schedule/count mismatch invalidates Phase B before held-out unlock.

### Held-out reporting and metrics

After all 24 final models, natural unpruned lifts, and matched initial models are immutable,
finite, count-validated, hashed, and written to the in-memory raw collection, globally unlock
held-out original views `[3,7,11]`. Render every natural unpruned lift, matched initial model, and
final model in every held-out camera with the same frozen degree-0 Torch renderer, hard
color/support, 3-sigma visibility margin, and black background. No model update, selection, rerun,
exception-based repair, or branch is permitted after unlock. Natural-unpruned held-out metrics are
count-confounded diagnostics only and cannot enter a gate.

For each held-out camera, compute `image_metrics(pred,target,mask=None)` after clamping both to
`[0,1]`. Thus PSNR uses full-canvas float32 MSE with floor `1e-12`; SSIM is the repository's
single-scale 11x11 separable Gaussian-window implementation with sigma `1.5`, constants
`0.01^2,0.03^2`, and mean over pixels/channels. Serialize raw per-camera MSE, PSNR, SSIM, color,
alpha, accumulated depth, and target tensors. Aggregate cameras by an unweighted arithmetic mean
within a seed. Seeds, not cameras or pixels, are the replicate unit.

As nondecisional diagnostics, report training-view step-0/checkpoint/final PSNR, natural-unpruned
held-out PSNR/SSIM, natural versus matched counts, exact source-set overlap, final parameter
displacement, per-camera held-out alpha sum, and normalized depth error on the fixed target-valid
mask `gt_depth>0.05`. For depth, use
`predicted_depth=accumulated_depth/clamp_min(alpha,1e-6)`, report the target-mask RMSE divided by
the frozen scene extent and the fraction of target-valid pixels with `alpha>0.05`. Neither depth
nor alpha can rescue or reverse a PSNR/SSIM decision. Timing is recorded as nondeterministic
metadata and supports no performance comparison.

### Frozen estimands and decisions

For backend `b`, seed `s`, let `Y00,Y10,Y01,Y11` be the mean final held-out metric for the four
arms. For both PSNR and SSIM report:

- scalar main effect
  `Delta_scalar = 0.5*((Y10-Y00)+(Y11-Y01))`;
- color main effect
  `Delta_color = 0.5*((Y01-Y00)+(Y11-Y10))`;
- interaction `Delta_interaction = Y11-Y10-Y01+Y00`;
- primary full-candidate difference `d_s,b = Y11-Y00`.

Report all per-camera values, per-seed estimands, and unweighted paired means/minima/maxima over the
three seeds. Do not pool cameras as independent samples, perform a significance test, form a
post-hoc confidence interval, drop a seed, or select a backend.

The primary full candidate is **non-inferior** for one backend only if all are true:

1. mean PSNR `d >= -0.25 dB`;
2. at least two of three seed PSNR differences are `>=-0.25 dB`;
3. worst seed PSNR difference is `>=-0.75 dB`;
4. mean SSIM difference is `>=-0.005`;
5. worst seed SSIM difference is `>=-0.020`;
6. all initial/final renders and metrics are finite, every final held-out PSNR is `>=10 dB`, and
   all exact-count/schedule/isolation/raw-evidence gates pass.

It shows **material improvement** for one backend only if it is non-inferior and additionally:

1. mean PSNR difference is `>=+0.25 dB`;
2. at least two of three seed PSNR differences are `>=+0.10 dB`;
3. worst seed PSNR difference is `>=-0.25 dB`;
4. mean SSIM difference is nonnegative.

`repair_utility_survives=true` only if the full candidate is non-inferior for both Depth and
Carve. `cross_backend_material_improvement=true` only if it materially improves both. Backend-
specific non-inferiority/improvement is still reported separately; one backend cannot rescue the
other in either cross-backend decision.

For attribution only, call a named PSNR main effect or interaction a `material_driver` for one
backend when its absolute three-seed mean is at least `0.25 dB` and at least two of three seed
effects have the same nonzero sign as that mean. These labels explain the frozen four-arm result;
they do not select a new arm, change the primary comparison, or authorize a follow-up sweep.

Interpretation is frozen:

- full candidate fails non-inferiority in either backend: it does not survive as a general repair
  under this protocol; retain the current boundary and do not tune;
- non-inferior in both but materially improves neither: the tested invariant semantics preserve
  matched-capacity quality but show no improvement evidence;
- materially improves one backend only: a backend-specific research lead, not a cross-backend
  repair or default;
- materially improves both: cross-backend synthetic utility evidence, still requiring an
  independent result audit and separate real-data/default protocol;
- factorial main effects/interaction describe which semantic factor drove this exact experiment
  but do not establish physical truth.

## Phase-B artifacts and sole command

Fresh append-only Phase-B namespace:

- marker: `benchmarks/results/20260716_stage1_semantic_factorial_PHASE_B_ATTEMPT.json`;
- valid outputs: `<UTC>_cpu_stage1_semantic_factorial_utility.json`, matching
  `_utility_RAW.npz`, and matching `_utility_RESULT.md`;
- invalid siblings: `_utility_invalid.json`, `_utility_invalid_RAW.npz`, and
  `_utility_invalid_RESULT.md`;
- post-result independent review: matching `_utility_AUDIT.md` and optional machine-readable
  `_utility_SCIENTIST_REVIEW.json`.

The sole Phase-B scientific command is:

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python benchmarks/stage1_semantic_factorial.py utility --seal benchmarks/results/20260716_stage1_semantic_factorial_SEAL.json --phase-a benchmarks/results/<phase-a-UTC>_cpu_stage1_semantic_factorial_mechanism.json --phase-a-raw benchmarks/results/<phase-a-UTC>_cpu_stage1_semantic_factorial_mechanism_RAW.npz --phase-a-review benchmarks/results/<phase-a-UTC>_cpu_stage1_semantic_factorial_mechanism_SCIENTIST_REVIEW.json --output benchmarks/results/<fresh-UTC>_cpu_stage1_semantic_factorial_utility.json`.

Before claiming the marker, the harness must rehash the seal, all bound sources, Phase-A JSON/raw,
and machine review; require their mutually bound hashes; require review verdict `PASS` and
`phase_b_authorized=true`; validate all prospective valid/invalid JSON/raw/note paths are absent;
and bind every path and input digest into the marker. It then exclusively creates the marker
before constructing a utility seed. A valid run writes only the valid triple; a fail-closed run
writes only the invalid triple. Interruption consumes Phase B. No overwrite, resume, second
attempt, seed replacement, gate relaxation, or outcome-guided retry is allowed in this namespace.

The Phase-B artifact must receive an independent repository results audit before any quantitative
claim enters `README.md`, `docs/`, or `ara/`, before any new default is proposed, and before a
real-data or CUDA follow-up is opened. This preregistration itself is outcome-free and authorizes
only future implementation, review, sealing, and the two scientific commands in their gated order.
