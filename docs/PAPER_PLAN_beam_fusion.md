# Paper Plan

**Evidence status (2026-07-28): hypothesis only; not established.**

The first audited native-resolution Janelle carrier run does not authorize the main claim:
its optimized arms consume RGB, Beam-only is not competitive, and the capture lacks the sparse
reconstruction needed for the two Original-3DGS baselines. See
`benchmarks/results/20260727_carrier_refinement_fullres_RESULT.md` and its independent
`20260727_carrier_refinement_fullres_AUDIT.md`.

A subsequent instrumented all-26-view maturation attempt completed 128,000 updates and produced a
coherent all-fitted endpoint at 28.7641 dB native / 31.7410 dB compact-teacher FG PSNR, but three
clone recoveries and higher-SH exhausted their safety caps without the frozen plateau. It has no
held-out cameras or matched no-clone/immediate-standard control, and all optimization still
consumes native RGB. See
`benchmarks/results/20260727_carrier_maturation_all26_RESULT.md` and
`20260727_carrier_maturation_all26_AUDIT.md`.

The 2026-07-28 successor is structurally compact-only from fitted 2D fields through its final
3D Gaussians. Three paired-root, independently audited development experiments select corrected
covariance repair, two fixed-topology SH0 phases, strict fitting-view projected-center pruning,
and a phase-2 mean freeze; they reject legacy opacity/appearance repair, clone-all, and an
additional SH3 stage. This closes the implementation-policy question on one exposed scene, not
the paper claim. There is still no controlled dense-versus-compact VRAM comparison, fresh-scene
held-out replication, physical no-floater proof, or Original-3DGS baseline. See
`benchmarks/results/20260728_compact_only_carrier_stage_ablation_RESULT.md`,
`20260728_compact_only_carrier_policy_closure_RESULT.md`, and
`20260728_compact_only_carrier_sequence_interaction_RESULT.md`, together with their audits. The
paper remains a plan, not a result.

**Working title**

Compact 2D Gaussian Captures for Memory-Efficient 3D Gaussian Splatting

Beam Fusion is an experimental reconstruction arm, not a prerequisite for the compact-input
claim or the title.

---

## Claims

### Systems / VRAM claim

Compact 2D Gaussian captures contain sufficient information to supervise reconstruction and
refinement of a standard 3D Gaussian Splat representation without returning to source RGB.

The intended benefit is reconstruction from datasets whose raw RGB working set does not fit in
GPU memory. That benefit remains unestablished until the controlled resource experiment passes.

This claim must be tested by a no-Beam arm. It is established only by held-out quality parity and
a controlled reduction in measured peak VRAM versus the RGB-backed baseline; the structural
absence of RGB alone is insufficient.

### Beam Fusion research hypothesis

Given the same compact 2D Gaussian captures, Beam Fusion may provide a better 3D initialization
than the no-Beam arm, improving time-to-quality, final held-out quality, geometry, or required
3D capacity at matched compute.

This is a separate exploratory claim. A null or negative Beam result must not weaken or be used to
reject the systems / VRAM claim.

---

## Two Pipeline Arms

### Arm 1 — direct compact reconstruction (no Beam)

```text
source RGB --offline fit--> frozen per-view 2D Gaussians --discard RGB-->
Beam-independent 3D initialization --> compact-field-supervised 3DGS --> final 3D Gaussians
```

Purpose: test whether the compact 2D representation can replace source RGB during 3DGS
optimization and reduce peak VRAM without losing held-out quality.

Hard boundary:

- no Beam Fusion, Beam contributor lineage, Beam covariance repair, or Beam-selected schedule;
- use the same 3D initialization, optimizer, topology budget, and stopping rule as the matched
  RGB-backed control;
- query or stream compact fields on demand rather than materializing and retaining a dense
  all-view RGB tensor;
- use train compact fields for optimization, validation views for selection, and untouched test
  cameras for final reporting.

### Arm 2 — compact reconstruction with Beam Fusion

```text
source RGB --offline fit--> frozen per-view 2D Gaussians --discard RGB-->
Beam Fusion --> optional Beam-lineage repair --> the same compact-field-supervised 3DGS
--> final 3D Gaussians
```

Purpose: measure the incremental value of Beam Fusion after the compact-input / VRAM pipeline
already works.

Hard comparison:

- consume the exact same frozen 2D captures and train/validation/test split as Arm 1;
- match optimizer, update or wall-clock budget, final capacity, stopping rule, metrics, and seeds;
- account separately for Beam construction time and peak memory;
- compare against Arm 1 at initialization, time-to-quality, and the selected final endpoint;
- keep Arm 2 labelled exploratory until it replicates on fresh scenes and paired seeds.

Arm 1 carries the systems / VRAM claim. Arm 2 carries only the Beam Fusion hypothesis.

---

## Motivation

Camera domes generate enormous image datasets.

Current 3DGS pipelines require all images to be available during optimization.

This causes

- GPU memory bottlenecks
- storage bottlenecks
- bandwidth bottlenecks

The compact-capture pipeline replaces RGB images by compact Gaussian captures. Beam Fusion is one
optional way to initialize the resulting 3D reconstruction.

---

## Contributions

1. Compact 2D Gaussian capture representation and compact-field supervision.

2. Measured memory/quality trade-off of direct compact reconstruction without Beam.

3. Beam Fusion reconstruction as a separately evaluated optional initializer.

4. Compact fixed-topology carrier refinement and projected-center containment as Beam-arm
   research components, not prerequisites for contribution 2.

---

## Required Experiments

### Baselines

Original 3DGS

Original 3DGS with compressed RGB

Direct compact reconstruction with the same initialization as Original 3DGS

Beam Fusion + the same compact optimizer used by the direct arm

Beam Fusion + accepted compact carrier schedule, reported separately when it differs from the
matched optimizer

---

## Ablations

Without corrected covariance repair

One compact phase versus two

Phase-1 means frozen

Means only

Without projected-center containment

Prune before versus after phase 2

Phase-2 means trainable versus frozen

SH0 versus incremental SH3

The rejected legacy opacity/appearance, soft-support, and clone-all stages remain useful
single-scene negative controls, but they are not candidate production stages.

---

## Metrics

PSNR

SSIM

LPIPS

Training time

Peak VRAM

Peak host RAM

Storage

IO bandwidth

Carrier survival

The compact experiment harness also records queried-field risks, primitive count, covariance
residuals, per-family optimizer motion, projected-center/near-plane violations, source hashes,
split isolation, and resource receipts. Future paper runs must compare peak VRAM under a matched,
idle-device protocol and record the compact working-set size separately from model/optimizer
memory. Beam-specific runs must retain the fixed-topology and frozen-mean containment recheck.

### Decision rule

- If Arm 1 reaches the preregistered held-out parity floor and materially lowers peak VRAM, the
  systems claim can proceed regardless of Beam.
- If Arm 2 also beats Arm 1 under the matched Beam gate, Beam becomes an additional method
  contribution.
- If Arm 2 ties or loses, retain the direct compact pipeline and report Beam as a bounded negative
  or exploratory result; do not fold Beam into the systems claim.

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
