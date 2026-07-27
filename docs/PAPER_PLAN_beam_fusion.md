# Paper Plan

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
