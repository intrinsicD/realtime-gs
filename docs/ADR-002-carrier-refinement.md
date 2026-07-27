# ADR-002 — Carrier Refinement Schedule

**Status:** implemented as an opt-in research path; not accepted as a default
**Date:** 2026-07-27
**Author:** Alex

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

## Implementation and Evidence Status

The design is implemented in:

- `rtgs.lift.carrier_refinement`: fixed-lineage covariance, optical-density opacity, and robust
  SH0 appearance repair;
- `rtgs.optim.carrier_schedule`: warm-up, protected clone/particle maturation, higher-SH
  expansion, standard handover, and lineage diagnostics;
- `rtgs.pipeline.run_carrier_pipeline`: typed end-to-end Beam Fusion -> repair -> maturation
  orchestration.

Every control is opt-in and the incumbent optimizer defaults are unchanged. The native-resolution
masked-Janelle development experiment and independent scientist pass are recorded in
`benchmarks/results/20260727_carrier_refinement_fullres_RESULT.md` and
`benchmarks/results/20260727_carrier_refinement_fullres_AUDIT.md`.

That experiment rejects only its frozen short 30/40/30/60 instantiation on the exposed development
scene. It did not run any phase to convergence and did not save/render every repair and clone-wave
boundary, so it does not test the full process specified above. Carrier survival moved as designed,
but downstream quality did not improve within the short budget; covariance repair also showed a
local-objective/downstream-quality dissociation. This ADR therefore records an available research
seam, not an accepted optimization policy. The next implementation must add explicit train-only
plateau criteria, tangent-only clone waves with recovery between waves, and complete boundary/
trajectory artifacts before the design itself can be judged.
