# ADR-002 — Carrier Refinement Schedule

**Status:** compact-only carrier implementation policy accepted; empirical support remains
single-scene development evidence
**Date:** 2026-07-27; decision revised 2026-07-28
**Author:** Alex

---

## Decision update — 2026-07-28

The carrier path now has a hard data-boundary and a smaller accepted sequence:

1. fit-only Beam Fusion from `ReconstructionInputs`, with at least three contributing views per
   carrier and no free primitive birth;
2. renderer-aware covariance repair only;
3. 380 compact, fixed-topology SH0 updates with means, rotations, scales, opacity, and SH0
   trainable;
4. strict projected-center pruning against every fitting-view 2D Gaussian field; and
5. 380 further compact SH0 updates with means frozen.

`run_carrier_pipeline(inputs, config)` accepts no `SceneData` or image argument. Its implementation
does not import the dense image trainer, does not load optional packed alpha, and contains no
clone, split, insertion, densification, opacity reset, higher-SH phase, or dense handover. The
second compact phase freezes means and rechecks containment, so the pruning invariant cannot be
undone.

The earlier Phase 1–7 proposal below is retained as the historical design that the experiments
tested. It is superseded for the carrier implementation: legacy covariance, opacity, and
appearance repair; soft support; clone/particle maturation; SH expansion; and standard
RGB-backed 3DGS handover are not part of the accepted carrier path.

### Corrected covariance objective

The point renderer projects a 3D covariance as

`P = J Σ Jᵀ + 0.3 I`.

The previous repair omitted the renderer's EWA dilation and minimized a one-sided whitened
Frobenius residual. That objective penalized expansion more strongly than collapse and was
empirically harmful. The accepted repair minimizes the reciprocal, symmetric generalized-log
residual

`sqrt(mean(log(lambda(C^-1/2 P C^-1/2))^2))`,

where `C` is the observed 2D covariance. Some fitted targets are sharper than the renderer's
0.3-pixel floor, so zero residual is not always attainable.

The old opacity repair is invalid as a physical inference: normalized 2D mixture amplitude does
not identify 3D alpha. The old cross-view amplitude-weighted appearance repair has the same gauge
problem and was empirically immaterial. Both are retired as separate stages. Compact sampled RGB
risk remains a valid objective for the rendered student versus the frozen 2D fields, but its
alpha/color gauge freedom explains why phase 1 must allow several parameter families to co-adapt;
means-only optimization was strongly rejected.

### Stage math disposition

| Stage | Math review | Policy |
| --- | --- | --- |
| Beam closest-ray center | The closest-ray signs and camera/world conventions are consistent. | Keep; require at least three contributing views. |
| Beam covariance intersection | Equal-weight CI is internally consistent for correlated observations and avoids the overconfidence of multiplying correlated beams. | Keep as the bounded carrier initializer. |
| Tangent covariance lift | The lift is correct, but the 3D student renderer adds the 0.3 EWA term absent from the raw lift target. | Keep as Beam initialization; follow with corrected covariance repair. |
| Legacy covariance repair | Its one-sided whitened Frobenius residual omits dilation and is shrink-biased. | Retire. |
| Opacity repair | Fitted normalized-mixture amplitude is a representation gauge, not identifiable 3D transmittance. | Retire. |
| Appearance repair | Robust averaging is a valid initializer operation, but amplitude weighting across views is not physical and uniform weighting was immaterial. | Retire as a stage. |
| Compact point-color loss | It is the correct sampled student-render versus frozen-field objective. Alpha/color gauge freedom means it cannot identify opacity or support independently. | Keep, with all parameter families trainable in phase 1. |
| Means-only optimization | Mathematically valid but blocks the scale/orientation/opacity/color co-adaptation required by the compositor. | Reject. |
| Soft support loss | The nearest-ellipse hinge has an outside gradient and is opacity-independent, but it did not reduce violations materially. | Retire at the tested formulation. |
| Clone-all | Copying opacity changes coincident alpha from `a` to `1-(1-a)^2`; `a'=1-sqrt(1-a)` preserves optical density, but even the preserving variant failed its stage gate. | No topology growth. |
| Higher SH | Compact supervision is possible, but incremental SH3 improved `J_Q` by less than the frozen 5% materiality floor. | Keep SH0. |

### Projected-center containment

For every retained center `x_i` and every fitting view `v`, the implementation enforces

`depth_v(x_i) > near`

and

`min_j (pi_v(x_i)-mu_vj)ᵀ C_vj^-1 (pi_v(x_i)-mu_vj) <= 9`.

The threshold is fixed at three standard deviations and cannot be weakened through the carrier
configuration. This removes carriers whose projected centers fall outside the union of
positive-amplitude fitted 2D Gaussians in any fitting view. It deliberately uses the frozen 2D
Gaussian fields rather than RGB masks.

This is a visual-hull center invariant, not a proof that every retained Gaussian occupies a
physical surface. It cannot detect a floater inside the multi-view hull. A mathematical Gaussian
has infinite support, so whole-Gaussian mask containment is impossible; a finite-footprint rule
would erode legitimate silhouette-boundary primitives and has not been validated.

### Evidence and scope

Three preregistered, independently audited 2026-07-28 experiments selected this sequence:

- [`20260728_compact_only_carrier_stage_ablation_RESULT.md`](../benchmarks/results/20260728_compact_only_carrier_stage_ablation_RESULT.md)
  corrected the covariance math, rejected legacy repair and means-only optimization, and rejected
  the tested soft-support loss;
- [`20260728_compact_only_carrier_policy_closure_RESULT.md`](../benchmarks/results/20260728_compact_only_carrier_policy_closure_RESULT.md)
  selected the two fixed-topology phases, rejected cloning, and found that phase-2 mean freezing
  passes the frozen 2% non-inferiority gate; and
- [`20260728_compact_only_carrier_sequence_interaction_RESULT.md`](../benchmarks/results/20260728_compact_only_carrier_sequence_interaction_RESULT.md)
  selected prune-before-phase-2 ordering and rejected an additional SH3 stage at the frozen 5%
  materiality threshold.

All three are single-scene fitted-view development results. They justify the structural carrier
policy, not a general quality, VRAM, runtime, no-floater, production-readiness, or publication
claim. A controlled dense-versus-compact, multi-scene resource experiment is still required for
the VRAM selling point.

---

## Motivation

Beam Fusion reconstructs an initial sparse set of 3D Gaussians from compact 2D Gaussian captures.

Experiments indicate that

- the estimated means are already surprisingly accurate,
- however the remaining parameters (covariance, opacity, appearance) are not yet sufficiently
  initialized,
- therefore the standard 3DGS optimizer gradually replaces the initial Gaussians by newly
  generated ones (primarily through split).

As a consequence the current optimization does **not** measure the quality of Beam Fusion itself,
but instead measures how well the standard densification process can recover from an imperfect
initialization.

The objective of this schedule is therefore:

> Preserve and mature Beam-Fusion carriers before standard 3DGS optimization takes over.

---

## Design Principle

Beam Fusion should not be viewed as a drop-in initializer.

Instead it produces **carrier Gaussians**.

Those carriers are refined until they become indistinguishable from ordinary optimized 3DGS
Gaussians.

Only afterwards should the standard optimization pipeline take over.

---

## Phase 0 – Beam Fusion

**Input**

- 2D Gaussian captures
- camera poses

**Output**

Sparse carrier Gaussians

**Parameters**

- Mean
- Covariance
- Opacity
- SH
- Color

---

## Phase 1 – Covariance Repair

**Freeze**

- Means

**Optimize**

- Covariance

**Objective**

Project every carrier Gaussian back into all associated views and minimize the discrepancy between

projected covariance

and

observed 2D covariance.

This is a pure covariance optimization.

Means remain fixed.

**Regularization**

- aspect ratio limits
- eigenvalue clamping
- positive definiteness

---

## Phase 2 – Opacity Repair

**Freeze**

- Means
- Covariance

**Optimize**

- Opacity

Instead of directly fitting alpha, optimize optical density (transmittance space).

**Objective**

The projected carrier should explain the observed 2D Gaussian opacity.

---

## Phase 3 – Appearance Initialization

**Initialize**

- SH0

using robust weighted averaging over all associated 2D Gaussians.

Higher-order SH coefficients remain disabled.

---

## Phase 4 – Fixed Topology Warm-Up

**No**

- Split
- Clone
- Insert
- Prune

**Only optimize**

- Mean
- Covariance
- Opacity
- SH0

until convergence.

**Purpose**

Determine whether the Beam Fusion carriers are already sufficient.

---

## Phase 5 – Clone-only Densification

**Enable**

Clone

**Disable**

Split

**Disable**

Insert

**Disable**

Pruning of original carriers

Parent Gaussians remain alive.

Children inherit

- lineage
- covariance
- appearance
- opacity

with small perturbations.

**Purpose**

Refine the carrier representation instead of replacing it.

---

## Phase 6 – Appearance Expansion

**Enable**

Higher-order SH

after

- geometry
- opacity

have stabilized.

---

## Phase 7 – Standard 3DGS

Once

- geometry stabilized
- carrier survival stabilized
- convergence plateau reached

hand over to the standard 3DGS optimization schedule.

Possible:

- split
- pruning
- default learning rates
- full SH

At this point the representation is a normal 3DGS model.

---

## Drop-in Boundary

Beam Fusion itself is **not** a drop-in replacement for existing initialization methods.

The complete package

Beam Fusion

\+

Carrier Refinement

\+

Clone Maturation

acts as the drop-in module.

**Output**

A standard-compatible 3DGS state.

---

## Diagnostics

Log

- Initial carrier survival rate
- Mean displacement
- Covariance evolution
- Opacity histogram
- PSNR
- SSIM
- LPIPS
- Number of cloned descendants
- Number of surviving original carriers

The central hypothesis is

"Good carrier means are currently hidden by poor initialization of the remaining Gaussian
parameters."

---

## Future Extension

### Error-driven Clone Selection

The first convergence experiment intentionally uses a **clone-every-carrier** schedule. This is a
mechanism/control experiment: every current Gaussian is cloned at each scheduled wave, parents
survive, and children receive small perturbations in the parent's tangent plane.

A later, separately preregistered experiment should make clone selection **error-driven**. Candidate
scores may combine

- multi-view image residual attributed to the carrier,
- missing projected coverage,
- cross-view disagreement,
- insufficient opacity support.

The error-driven variant must be compared with clone-every-carrier under matched clone budgets,
wave cadence, optimization budgets, and tangent-space perturbations. Selection thresholds and
stopping rules must be frozen before the run, and reporting/held-out views must not contribute to
the score. This extension is explicitly **not** part of the all-26-view clone-schedule experiment
specified below.

### Carrier-guided Particle Generation

After Clone-only has demonstrated that mature carriers are stable, they may become **particle
sources**.

Instead of classical split

each mature carrier samples new children from a local proposal distribution.

**Characteristics**

- parent remains alive
- children start with low opacity
- children inherit carrier parameters
- small stochastic perturbations
- aggressive pruning of unsuccessful children

This differs fundamentally from split.

It is a guided local exploration process originating from verified carrier Gaussians.

**Potential triggers**

- high residual
- missing coverage
- multi-view disagreement
- insufficient opacity support

This should remain an optional late-stage refinement strategy.

---

## Historical implementation and 2026-07-27 evidence

The superseded Phase 1–7 design was implemented in:

- `rtgs.lift.carrier_refinement`: fixed-lineage covariance, optical-density opacity, and robust
  SH0 appearance repair;
- `rtgs.optim.carrier_schedule`: warm-up, protected clone/particle maturation, higher-SH
  expansion, standard handover, and lineage diagnostics;
- `rtgs.pipeline.run_carrier_pipeline`: typed end-to-end Beam Fusion -> repair -> maturation
  orchestration.

At the time, every control was opt-in and the incumbent optimizer defaults were unchanged. The native-resolution
masked-Janelle development experiment and independent scientist pass are recorded in
`benchmarks/results/20260727_carrier_refinement_fullres_RESULT.md` and
`benchmarks/results/20260727_carrier_refinement_fullres_AUDIT.md`.

That experiment rejects only its frozen short 30/40/30/60 instantiation on the exposed development
scene. It did not run any phase to convergence and did not save/render every repair and clone-wave
boundary, so it does not test the full process specified above. Carrier survival moved as designed,
but downstream quality did not improve within the short budget; covariance repair also showed a
local-objective/downstream-quality dissociation.

The subsequent all-26-view V2 attempt adds the missing explicit plateau criteria, tangent-only
clone-every-carrier waves with recovery, and complete boundary/trajectory artifacts. It completed
128,000 optimizer updates and a 100,000-Gaussian fitted-view reconstruction, but clone recoveries
1–3 and higher-SH each exhausted their 15,000-update safety cap with
`plateau.converged=false`; only fixed topology and final standard settle reached the plateau. The
result and audit are
`benchmarks/results/20260727_carrier_maturation_all26_RESULT.md` and
`benchmarks/results/20260727_carrier_maturation_all26_AUDIT.md`.

This falsifies convergence-between-stages for that frozen execution, not the possibility that a
different cap policy or schedule could work. It also does not isolate carrier value: all cameras
were fitted, there is no matched no-clone or immediate-standard continuation, and every optimized
phase consumed native RGB. That result did not authorize its RGB-backed schedule as an
optimization policy. The compact-only successor accepted above supplies paired controls and
predeclared stopping budgets, but remains single-scene and fitted-view. Held-out cameras, fresh
scenes, and a controlled VRAM comparison remain required.
