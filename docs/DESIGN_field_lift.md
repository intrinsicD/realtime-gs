# Design: field-level 2D→3D lifting with discrete topology control

Status: **implemented research path.** The CPU-first implementation covers the typed compact
input boundary, measurement and observability controls, analytic field proxy, visibility/gain,
fiber refit, deterministic topology moves, maskless placement, registry/pipeline/CLI integration,
and semantic validation. One audited calibrated all-fitted-view development run now establishes
bounded execution and fitted-target quality only; no held-out/generalization, performance,
production-default, or topology-utility claim has been established.

Related docs: `docs/ARCHITECTURE.md` (current pipeline), `docs/RESEARCH.md` (SOTA survey and
reuse decisions), `docs/EXPERIMENTS.md` (the empirical record this design leans on),
`docs/ROADMAP.md` (milestones — this is the M3/stage-2 rework).

Empirical update (2026-07-21): on the complete 26-view `frame_00008` compact bundle, the public
path placed 128 tracks and returned 127 after one accepted topology move, with no fallback. All
outputs had covariance-observability rank 6 (maximum condition 2.3109), and source projection and
color errors were at most `2.18e-11` and `2.22e-16`. Ordinary adaptive density then grew the model
to 39,059 and reached the common 70k fitted-target plateau at 37.2408 dB foreground PSNR. Field
ranked sixth of seven by that metric and did not support a default change. The execution saved
aggregate topology counts (7 proposals/1 acceptance) but omitted the protocol-required individual
move receipts; final quality is audited, while move-level topology utility is not. See
`benchmarks/results/20260721_all_initializers_frame00008_{RESULT,AUDIT}.md`.

---

## 1. Motivation

The repository's thesis is: fit every image with 2D gaussians, then **lift** those 2D
gaussians into a 3D gaussian set that serves as the initialization for 3DGS refinement. The
central longstanding sub-problem is **cross-view correspondence**: every surface patch is observed
by 2D gaussians in several images, but the established lifters do not represent that
correspondence explicitly. They handle it implicitly and brittly in three different disguises:

- `gradient` lifts every per-view 2D gaussian to its own 3D gaussian on its ray, and lets
  observations interact only through the rendered photometric loss — no direct cross-view
  coupling. `merge_by_voxel` then resolves correspondence **post hoc** by a hard voxel hash.
- `carve` mediates correspondence through a shared voxel volume (cross-view color mean/variance)
  — a fixed, non-learned aggregation.
- `depth`/`hybrid` sidestep correspondence with a monocular prior.

The experiment log already shows the cost of this: the 2026-07-08 entry found that voxel merging
"barely fires because the geometry is scattered — redundancy and under-constrained depth are
coupled." The goal of this redesign is to make cross-view correspondence a **first-class,
soft, many-to-many** object and to fold placement, correspondence, and topology cleanup into a
single image-free optimization stage.

Two facts make this the right moment:

1. **Preprocessing is moving stage 1 offline.** After preprocessing, the pipeline consumes only
   per-view 2D gaussians + cameras (+ optional masks/depth/SfM). Images never re-enter the loop.
2. **The fitted field is the Stage-2 observation boundary:** a few hundred primitives instead of
   `H×W` pixels. For additive peak-Gaussian mixtures, the whole-plane density `D` and RGB
   numerator `N` have closed-form product-kernel losses. Normalized StructSplat rendering with
   finite support/fade and optional affine color is not that additive objective; the implementation
   evaluates those frozen-teacher semantics separately with bounded deterministic point sampling.

---

## 2. Theory established in discussion (the constraints the design must respect)

These results are the reason the design looks the way it does. Several correct a naive first
cut. Where a claim was checked numerically in this session it is marked ✓verified.

### 2.1 The projection fiber (per-view lift ambiguity) — 4 DOF

The EWA projection Jacobian `J` annihilates the viewing ray: `J·d = 0`. A 3D gaussian has 9
geometric DOF; one view constrains 5 (center 2, covariance-tangent-block 3). The unconstrained
**fiber** is 4-dimensional: depth `t`, two tangent–ray shears, and the along-ray variance. In a
ray-adapted basis this is exactly one extra Cholesky row appended to the (measured) 2D Cholesky
factor — parametrizing the fiber by that row is unconstrained and degeneracy-free (no
quaternion normalization, no `_MIN_THICKNESS` hack). The current `lift_covariance` sits at the
untilted point (shears = 0, thickness = `sigma_ray`) of this fiber. The implemented
`inverse_projection_fiber` module provides this parametrization.

### 2.2 Two views do NOT isolate the covariance — observability table (✓verified)

Writing the per-view covariance projection `C_v = A_v Σ A_vᵀ` with `A_v = J_v R_v` (2×3, rank 2,
`A_v d_v = 0`), the symmetric indefinite matrix `Q = d₁d₂ᵀ + d₂d₁ᵀ` satisfies
`A₁ Q A₁ᵀ = A₂ Q A₂ᵀ = 0`. Therefore `Σ` and `Σ + λQ` are **indistinguishable from two views**,
and both stay PD for a range of `λ`. Numerically (random cameras, float32): two-view projection
differences ~1e-7, stacked two-view linear system **rank 5 of 6**, three generic views rank 6,
and as the baseline shrinks `Q → 2 d₁d₁ᵀ` (the along-ray thickness mode).

| Views | Mean | Covariance | Complete gaussian |
|---|---|---|---|
| 1 | 1 ray DOF free | 3 null DOF | 4-D fiber |
| 2 | triangulated | 1 null DOF (`λQ` line) | 1-D intersection (segment, PSD-bounded) |
| 3 generic | triangulated | rank 6, determined | isolated solution |

Consequences that the design **must** carry: (a) no two-view minimal covariance solver without
an explicit prior fixing `λ`; (b) a covariance-observability **gate** — a gaussian gets full
covariance freedom only where ≥3 well-conditioned views see it, else its `λQ` mode is pinned by
prior; (c) view-triple **conditioning** matters, so report rank + condition number and prefer
well-spread triples; narrow-baseline captures are near-degenerate even with many views.

### 2.3 Components have no stable identity — losses must be field-level

Fitted 2D components are decomposition artifacts, not relabeled projections of latent 3D
gaussians. The unequal-decomposition control measured **83.75% of moment-split child centers
differ from the latent parent's projected center**. One latent gaussian may be split into several fitted
components, and one fitted component may absorb several latent gaussians; occlusion/compositing
break conditional independence further. Therefore:

- **Every continuous quantity in the loss must be decomposition-invariant** — defined on the
  *field* (the mixture as a function), not on component pairs. The field discrepancy
  `∫‖F̂_v − F_v‖²` expands into closed-form product-kernel terms; correspondence variables never
  appear in it.
- Component-level statistics are still useful, but **only as diagnostics that propose discrete
  moves** (redundancy → merge; unexplained reference mass → birth), where identity need not be
  stable, only informative.
- A rank-K near-one-hot correspondence **tensor is not a valid data model** here (it presumes
  component identity). What survives of the tensor view: observability as operator algebra
  (§2.2), field-level moment observables, and soft aggregation as post-hoc bookkeeping.

### 2.4 Soft weights cannot remove duplicates — discrete moves are required

Sinkhorn plans, BCPD responsibilities, and Dirichlet `α`s cannot, on their own, delete a
duplicate, for two structural reasons:

- **Render-equivalence flatness.** Two co-located half-mass copies produce the identical field.
  Every field/likelihood objective is exactly flat along the mass-exchange direction, so
  continuous dynamics have zero gradient toward pruning; only a parsimony term breaks the tie,
  and only asymptotically.
- **Mode capture.** Once all candidates converge onto one mode, recovering a lost mode requires
  passing through high-cost states — a topology change no continuous deformation performs.

So the architecture is **"posteriors propose, discrete moves dispose"**: soft association
supplies evidence; explicit birth/death/merge/split enact topology, each accepted only if the
implemented analytic additive density proxy plus parsimony improves. Precedent already in-repo:
gsplat MCMC relocation/teleport (stage 3) and StructSplat residual growth (stage 1).

### 2.5 Attention ≡ the E-step (conceptual background; no transformer is added)

Row-softmax over candidates is one-sided entropic OT; Sinkhorn-with-dustbin is
partial/entropic OT with both marginals (SuperGlue layer); the propose/refit alternation is EM
(Coherent Point Drift generalized from points to gaussians). Temperature `τ→0` recovers the hard
graph, so hard-vs-soft is a temperature/normalizer ablation, not two systems. Four quantities
stay **distinct** (a discipline the current `merge_by_voxel` violates by conflating mass and
confidence): geometric-compatibility **logit**; observation **mass** (footprint area/energy, an
OT marginal / capacity); association **posterior** (softmax/Sinkhorn output = confidence);
rendering **opacity** (existence, never copied from the others). Association mass to a **dustbin**
absorbs occlusion and decomposition mismatch and is *not* evidence of non-existence.

BCPD contributes, if wanted later, three optional ingredients with no counterpart in plain
attention: a **coherence prior** (GP-smooth depth field per source view; geodesic kernel to
avoid smearing across occlusion edges), **data-driven annealing** (closed-form `σ²` schedule),
and **posterior uncertainty** (per-track depth variance → principled along-ray thickness and a
"tracks stabilized" criterion for the geometry→color phase switch). Sinkhorn-EM (Mena et al.)
is the convergence-guaranteed bridge between the OT E-step and the EM outer loop.

---

## 3. Target pipeline

```
preprocessing (offline, images only here) ─► per-view artifact: 2D gaussians + camera
                                                (+ optional masks / depth prior / SfM tracks)
        │
        ▼
[stage 2a] volume / placement init
   masked   : carve  → consistency-peak placement along ray tunnels (current carve, best lifter)
   maskless : ladder (§5) — frustum-consensus bounds + field-sweep placement + optional anchors
        │  Gaussians3D on/near the surface, fiber-parametrized (exact source projection at init)
        ▼
[stage 2b] FIELD-FIT loop  (image-free; implemented research core)
   repeat until field loss converges or budget reached:
     • backproject each 3D gaussian into each reference view (closed form)
     • analytic proxy loss on additive density D and RGB numerator N, with per-gaussian
       per-view VISIBILITY weights and a per-view GAIN
     • continuous step: fiber-constrained refit (depth + covariance free column; color/SH staged)
     • discrete moves: prune / merge / split / birth, accepted on analytic-proxy objective delta
   outputs: 3D gaussian set + dense soft cross-view correspondences (byproduct)
        │
        ▼
[stage 3] short image-based 3DGS polish  (gsplat; existing) — lifts the stage-1 approximation
                                                                ceiling off the final result
```

The native path is `SceneFits` → `run_field_pipeline` / `FieldLifter.fit`, or
`rtgs lift-field --dataset ...`; it never loads source images. `SceneFits` preserves optional
lossless `PackedAlpha` and requires a complete, disjoint, explicit train/held-out partition.
The CLI saves the standard 3DGS PLY/NPZ requested by `--out`, plus
`Path(--out).with_suffix(".field.npz")` containing field masses, render opacity, source/fiber
state, visibility, gains, split indices, and per-view soft correspondences. Strict
`Path(--out).with_suffix(".diagnostics.json")` stores semantic validation and diagnostics.
Stage 3 remains separate and unchanged.

---

## 4. Stage 2b — the field-fit loop (detail)

### 4.1 Analytic additive proxy and frozen-teacher semantic validation

For an **additive peak-Gaussian mixture**, compare the projected and reference density `D` and RGB
numerator `N` by whole-plane L2 discrepancies whose cross terms are product-kernel evaluations
`⟨N₁,N₂⟩ = N(μ₁−μ₂; 0, Σ₁+Σ₂)` — closed form with analytic gradients:

```
L_v = L2(D̂_v,D_v) + L2(N̂_v,N_v)
D̂_v(x) = Σ_i g_iv · a_iv(x),   N̂_v(x) = Σ_i g_iv · c_iv · a_iv(x)
g_iv = visibility · gain · field_mass
```

This analytic proxy is exact for `D` and `N` under those additive whole-plane semantics. It is
**not** the normalized finite-support/fade StructSplat RGB renderer, and optional affine color is
not preserved by the proxy. `field_validation` therefore evaluates each immutable teacher through
its actual query equation at a deterministic bounded sample, reporting isolated train and held-out
density/RGB aggregates. Those sampled metrics validate semantics; they are not silently substituted
into the analytic optimizer. **No Gaussian needs a component correspondence in either path.**

### 4.2 Visibility and amplitude semantics (must be explicit)

The reference fields were fit to **occluded** images, and accumulated 2D amplitude is **not**
3D alpha opacity (`docs/RESEARCH.md` §8). So a naive additive backprojection punishes back-surface
gaussians for missing views they are invisible in. Fix (EM-style, cheap): per-gaussian per-view
**visibility weight** `v_iv` (transmittance to the gaussian along the ray, from current geometry),
held fixed for a block of iterations and refreshed periodically; plus a per-view **gain** to
absorb amplitude non-conservation. Small effect at init with masks/convex captures; large near
convergence and for maskless scenes.

### 4.3 Continuous step — fiber-constrained refit

Optimize each gaussian on its fiber: depth `t` and the free covariance column (tangent–ray shears
+ along-ray thickness), with the tangent block slaved to the source fit (exact source projection
preserved). Color/SH **staged**: geometry-only first; enable appearance after per-track
stabilization; enforce the source-view emitted color as a directional constraint (soft or gated
on local component dominance — it is exact only for the fitted component's representation, not
the isolated physical color). A topology merge retains its representative anchor's actually
observed source color rather than inventing an averaged anchor. The observability gate (§2.2)
decides which gaussians get full
covariance freedom vs. a pinned `λQ` mode.

### 4.4 Discrete moves — the scheduler

"Propose from diagnostics, dispose on analytic additive density-proxy delta + parsimony,"
SMEM-style, deterministic and CPU-testable. The current integrated defaults are:

- **merge**: bounded multi-view projected Runnalls-KL field-bound ranking proposes a pair;
- **prune (death)**: the lowest field-mass component is proposed;
- **split**: the integrated default is an exactly co-located mass split; a directed residual split
  exists behind `TopologyOps` and in tests but is not the default;
- **birth**: unexplained-reference-field residual scoring selects an unused source component.

Accept a move iff `Δ(additive density proxy) + Δ(parsimony) < 0`, with visibility and gains fixed
for the transaction. Advanced multi-move scheduling and evidence that these moves improve a real
reconstruction remain future empirical work.
Merge construction retains one representative source fiber in full; it never averages depth or
shear coordinates expressed in different camera-ray bases. Mass, lineage, and alpha coverage are
combined, and the exact transaction objective rejects a representative-fiber merge that changes
the field too much.

### 4.5 Correspondences as output

At convergence the normalized product-kernel overlaps between backprojected and reference
gaussians are the dense soft cross-view correspondences. Each projected row is gated by the
detached center-transmittance visibility of that 3D gaussian in that view before normalization.
Visibility, field mass, rendering opacity, and the resulting association posterior remain
separate quantities. The correspondence is attention as a **byproduct**, never an input and never
turned into opacity/existence.

---

## 5. Maskless ladder (RGB-only, non-dome datasets)

Maskless is the **main** path for casual/outside-in captures (e.g. the roadmap's MipNeRF-360
`garden`/`bicycle`), not a degraded fallback. Key point: with SfM poses you almost never have
*only* RGB — COLMAP yields a sparse point cloud + per-view tracks for free (already parsed by
`rtgs/data/colmap.py`, already consumed by `rtgs/depth/align.py`). Degrade gracefully:

- **Tier 0 — working volume (always):** SfM-percentile box when points exist; else
  **frustum-consensus** region (positions inside ≥K frusta) as ROI; explicit far shell at radius
  `R` for unbounded scenes (background class, not forced into the volume).
- **Tier 1 — depth anchors (best available, all optional):** SfM-track depth (EDGS-style pinning
  + alignment target); monocular depth (Depth Anything V2 Small) run **at preprocessing time**,
  aligned, stored as per-gaussian prior + confidence in the artifact; or nothing (Tier 2 runs
  unanchored with a wider range). The implementation uses explicit per-component priors directly;
  for trusted train-only sparse geometry, it also takes a confidence-weighted seed from the nearest
  visible projected SfM point inside the source component's footprint.
- **Tier 2 — placement by discrete field-sweep along rays, not descent.** Port the unmerged
  `cost`/plane-sweep lifter and swap image sampling for **field evaluation**: K depth hypotheses
  in Tier-0 bounds (warm-started by Tier-1), scored by robust cross-view field agreement (best
  ~60% of neighbor views for occlusion), peak → `t`, peak width → along-ray `σ`, fiber gives the
  rest. Justification: the 2026-07-09 experiment showed discrete plane-sweep roughly **halved**
  per-ray descent's geometric error, and photometric depth-polish was **actively harmful**.
  Optional carve-style occupancy consensus with the hull term replaced by thresholded
  photo-consistency.
- **Tier 3 — identical to the masked field-fit loop (§4).** A mediocre maskless init is
  recoverable because birth/teleport targets unexplained field mass. Random-in-bounds is the
  honest floor; **per-ray gradient descent stays off the ladder** (the log is the argument).

Maskless-specific cautions: the observability gate matters *more* (outside-in arcs are
two-view-conditioned over large regions); expect a **foreground/background split** to replace the
mask semantically (far-shell gaussians as a separate population with a coarser budget).

---

## 6. Preprocessing artifact schema

`SceneFits` is the implemented image-free boundary. It carries ordered
`GaussianObservationField` teachers and cameras, unique view names, optional lossless
`PackedAlpha`, depth priors/confidences, neighbor lists, sparse points/visibility, and bounds.
Its train and held-out indices must form a complete, unique, disjoint partition with at least one
training view. `SceneFits.from_compact_dataset` deliberately preserves alpha instead of routing
through the legacy `CompactDataset.to_reconstruction_inputs()` adapter that drops it.
Optional points/bounds carry an explicit `geometry_is_train_only` provenance bit. With held-out
views, unverified geometry is retained for reporting but excluded from fitting; training-camera
frustum consensus supplies the working volume. If every view trains, or the caller explicitly
attests train-only provenance, the points/bounds may be used.

---

## 7. Testing and verification strategy

Everything below is CPU-only, deterministic (seeded), and tiny — consistent with the repo's hard
rules (test scenes ≤64×64, ≤300 gaussians, ≤200 iters; suite < ~3 min).

- **GT correspondence harness.** Project known GT 3D gaussians into each synthetic view and use
  the projections *as* the per-view "fits" → association known by construction. Moment-split each
  projection into `k ∈ {1,2,3}` sub-gaussians with **per-view** `k` → known many-to-many,
  deliberately unequal per-view counts. Metrics: track **purity/completeness**, independent of
  photometrics. This is a matcher unit test, not an end-to-end vibe check.
- **Oracle observability baseline (the corrected 8-step control).** (1) oracle parent membership;
  (2) moment-merge children per parent per view; (3) triangulate aggregate centers; (4) subtract
  known EWA dilation; (5) solve covariance from ≥3 generic views, **reporting rank + condition
  number**; (6) a two-view control that explicitly exposes its 1-D `λ` null coordinate;
  (7) compare direct linear geometry vs. nonlinear fiber refinement; (8) only if oracle aggregates
  recover geometry, reintroduce inferred topology / soft association. This cleanly separates the
  three failure modes: **topology vs. observability vs. optimization.**
- **Field-loss unit tests.** Closed-form value/gradient vs. finite differences; invariance under
  exact co-located amplitude splitting and permutation; chunked/dense parity.
- **Move-scheduler tests.** Each move is accepted only when the configured proxy objective
  improves; a duplicate pair is removed by merge; a planted second mode is recovered by birth;
  determinism under seed.
- **Ablations to log in `docs/EXPERIMENTS.md`.** hard graph (`τ→0`) vs. soft, on the prior
  session's failing root case and on unequal per-view counts; field-sweep vs. per-ray descent;
  visibility-weights on/off; observability-gate on/off; anchored vs. anchored+consensus. Field L2
  gives an image-free metric; held-out image PSNR remains the reported number (stage 3).

---

## 8. Implemented phases

Ordering favors the shared core first and the highest-risk/highest-value pieces early, each with
its falsifier. Tasks are tracked in the session task list; IDs referenced here after creation.

- [x] **Phase 0 — scaffolding & boundaries.** Loader boundary; a `SceneFits`
  container; and a `TopologyOps` interface (prune/merge/split/birth). *(Task: scaffolding.)*
- [x] **Phase 1 — measurement first.** GT correspondence harness + oracle observability baseline
  (§7). This is deliberately before the optimizer: it tells us whether topology, observability,
  or optimization is the actual bottleneck, and it is the control every later ablation compares
  against. *(Tasks: GT harness; oracle baseline.)*
- [x] **Phase 2 — field loss.** Closed-form additive `D`/`N` product-kernel discrepancy and
  gradients, plus separate sampled frozen-teacher semantic validation. *(Task: field loss.)*
- [x] **Phase 3 — visibility & gain.** Per-gaussian per-view transmittance weights + per-view gain,
  block-fixed and refreshed; tests that back-surface gaussians are not penalized in views that
  don't see them. *(Task: visibility/gain.)*
- [x] **Phase 4 — continuous fiber refit.** Depth + free covariance column on the fiber, staged
  color/SH with the source directional constraint and observability gate, reusing
  `inverse_projection_fiber`. *(Task: fiber refit.)*
- [x] **Phase 5 — discrete move scheduler.** Deterministic prune/merge/split/birth proposals with
  transactional additive-density-proxy-plus-parsimony acceptance. *(Task: scheduler.)*
- [x] **Phase 6 — maskless ladder.** Frustum-consensus bounds; field-sweep placement,
  evaluation; wire optional SfM/depth anchors; far-shell/background population. *(Task: maskless
  ladder.)*
- [x] **Phase 7 — integration.** Register as a lifter / stage-2 entry (`get_lifter`), pipeline test,
  image-free CLI and persistent field-state sidecars, mechanism benchmark entry, focused CPU
  tests, and synchronized architecture/roadmap/research documentation. *(Task: integration & docs.)*

Dependency sketch: 0 → {1, 2}; 2 → 3 → 4 → 5; {4,5} → 6; everything → 7.

---

## 9. Open empirical questions

The single all-fitted-view suite now measures one bounded initialization and terminal quality
point, where field ranked sixth by foreground PSNR; it does not answer whether field lifting
improves held-out quality, time to quality, topology recovery, or memory/runtime on calibrated
data. The required next evidence is a frozen train/held-out protocol comparing: topology moves
on/off and by proposal type; analytic proxy change versus frozen-teacher semantic metrics;
visibility/gain ablations; observability-gate behavior under narrow baselines; and masked versus
maskless placement. None of those questions is closed, and the research path is not a default.

---

## 10. Risks and falsifiers

- **Field-loss ceiling.** Stage 2b can only be as good as stage-1 fit fidelity; held-out PSNR is
  optimized only by proxy. *Mitigation:* keep the short image-based stage-3 polish; do not let the
  fits' approximation error become the system's ceiling.
- **Occlusion breaks field additivity.** Cross-view additivity holds only where the surface is
  visible. *Mitigation:* visibility gating (§4.2); robust neighbor-subset scoring in the sweep.
- **Over-smoothing (if the optional BCPD coherence prior is added).** Stationary kernels smear
  across depth discontinuities. *Falsifier:* if edge-heavy scenes dominate and a geodesic kernel
  doesn't rescue it, drop the GP — the skeleton survives without it.
- **Move scheduler miscalibration.** Sparse Dirichlet / aggressive death can kill true thin
  structure. *Falsifier:* track purity/completeness on the GT harness; fall back to
  threshold-free synchronization for track formation.
- **Soft association blurs incompatible decompositions.** The stated core risk. *Falsifier:* the
  hard-vs-soft ablation on the failing root case and on unequal per-view counts; entmax/sparsemax
  (exact zeros) as the anti-blur normalizer, with the hard graph as the `τ→0` limit.
- **Two-view degeneracy fit as noise.** Without the observability gate the optimizer fills the
  `λQ` null direction with garbage. *Mitigation:* the gate (§2.2), on by construction.

---

## References (discussion trail)

SuperGlue (1911.11763), LightGlue (2306.13643), Coherent Point Drift (Myronenko & Song 2010),
Bayesian CPD (Hirose 2020) + Geodesic BCPD, Sinkhorn-EM (Mena et al. 2006.16548), Delon–Desolneux
MW₂ (1907.05254), RegGS (2507.08136), Gaussian Herding across Pens (2506.09534), Gaussian Graph
Network (2503.16338), GaussianFormer (2405.17429), unbalanced OT (Chizat 1607.05816),
sparsemax/entmax (1602.02068 / 1905.05702), permutation synchronization (Pachauri 2013),
EDGS (2504.13204), GaussianObject (2402.10259), pixelSplat footprint prior (2312.12337). Full
context and reuse decisions in `docs/RESEARCH.md`.
