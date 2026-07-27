# ADR-002 — Carrier Refinement Schedule

**Status:** draft
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
