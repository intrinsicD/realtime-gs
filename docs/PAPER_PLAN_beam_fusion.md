# Paper Plan

**Evidence status (2026-07-27): hypothesis only; not established.**

The carrier-refinement implementation and the complete runnable development matrix now exist, but
the audited native-resolution Janelle run does not authorize the main claim. Optimized carrier
arms still consume RGB, Beam-only is not competitive, and this capture lacks the sparse
reconstruction needed for the two Original-3DGS baselines. See
`benchmarks/results/20260727_carrier_refinement_fullres_RESULT.md` and its independent
`20260727_carrier_refinement_fullres_AUDIT.md`. The paper remains a plan, not a result.

**Working title**

Beam Fusion:
Compact 2D Gaussian Captures for Memory-Efficient 3D Gaussian Splatting

---

## Main Claim

Compact 2D Gaussian captures contain sufficient information to reconstruct a standard 3D Gaussian
Splat representation.

The resulting model can afterwards be optimized using an ordinary 3DGS pipeline.

This enables reconstruction from datasets that are too large to fit into GPU memory when using raw
RGB images.

---

## Motivation

Camera domes generate enormous image datasets.

Current 3DGS pipelines require all images to be available during optimization.

This causes

- GPU memory bottlenecks
- storage bottlenecks
- bandwidth bottlenecks

Beam Fusion replaces RGB images by compact Gaussian captures.

---

## Contributions

1. Compact 2D Gaussian capture representation.

2. Beam Fusion reconstruction.

3. Carrier refinement schedule.

4. Standard-compatible handover.

---

## Required Experiments

### Baselines

Original 3DGS

Original 3DGS with compressed RGB

Beam Fusion only

Beam Fusion + Carrier Schedule

Beam Fusion + Carrier Schedule + Clone

---

## Ablations

Without covariance repair

Without opacity repair

Without warm-up

Split immediately

Clone-only

Carrier particle generation

Random initialization

Means only

---

## Metrics

PSNR

SSIM

LPIPS

Training time

Peak VRAM

Storage

IO bandwidth

Carrier survival

The experiment harness also records alpha IoU/leakage, primitive count, repair-objective
residuals, lineage drift/descendants, source hashes, split isolation, and final density events.
Future paper runs must include recovery updates after the final topology event.

---

## Related Work

3D Gaussian Splatting

2D Gaussian methods

Compact Gaussian representations

Compressed reconstruction pipelines

Gaussian compression

The comparison should focus on

input representation

instead of

final model compression.

---

## Non-goals

Not

- a better densification algorithm

Not

- a new split operator

Not

- a new renderer

Not

- a universal drop-in initializer

---

## Long-term Vision

The carrier concept naturally extends toward dynamic scenes.

Future work

2D Gaussian Video

↓

Space-Time Beam Fusion

↓

4D Carrier Representation

↓

Standard 4D Gaussian Splatting

The static paper intentionally avoids introducing temporal modeling and instead establishes the
carrier concept as the foundation for future dynamic reconstruction work.
