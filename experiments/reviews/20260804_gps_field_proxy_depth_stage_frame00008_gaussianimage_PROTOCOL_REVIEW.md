# Prospective Protocol Review

- Task ID: `20260804_gps_field_proxy_depth_stage_frame00008_gaussianimage`
- Protocol SHA-256: `28cac666e0e210e26c2ec066883062799f228b4d3e8734c74bfa0930856515b5`
- Reviewer: `Codex-gps-protocol-reviewer`
- Verdict: `approved`
- Outcome Access: `none`

## Scope

This review covers only the frozen development protocol, sealed compact-field inputs, external
GPS-Gaussian source and checkpoint bindings, field-proxy stereo geometry, matched initialization
and refinement controls, held-out isolation, failure policy, and complete-process resource
measurement. It does not assess an experimental outcome. A successful run may provide
single-frame development evidence for the learned stereo mechanism and descriptive memory cost of
this dense field-proxy implementation. It cannot establish source-RGB quality, physical depth
accuracy, generalization, sparse-throughout execution, a production default, or compact-VRAM
superiority over streamed, decode-on-demand, or point-sampled RGB controls.

## Checks

- Recomputed the exact prospective digest above and passed both task-wide contract validation and
  sealed-data validation without initializing or executing a run.
- Confirmed the 23-train/3-held-out split, compact-only reconstruction and evaluation policy,
  alpha-disabled CPU loading with the exact 8,388,608-byte loader-contract cap, and held-out access
  restricted to the two terminal reporting evaluations.
- Independently inspected `CompactView.load` and all 26 sealed archive metadata records. The loader
  requires the caller cap to equal the signed semantic-metadata declaration, and every archive
  declares 8,388,608 bytes. The seal and dataset manifest still bind each exact outer size and
  SHA-256 before loading; actual archives span 346,052--369,477 bytes, and the largest uncompressed
  teacher member is 359,088 bytes. Exact ZIP members, semantic and teacher digests, member-size
  limits, teacher schema, camera binding, and Gaussian counts remain fail-closed. With
  `load_alpha=False`, the packed alpha payload is neither read nor decoded while its declaration,
  member presence, and uncompressed length are checked.
- Verified the clean external repository at commit
  `0024776deee4824f270d4bb534a17ffd85f63cf2` and tree
  `ad9815910afe3cd441458a1e79dd1f56bef3ab7e`, plus the exact source and sanitized checkpoint sizes
  and hashes. Independently reproduced the frozen 132-key used-state signature and its two strict
  submodule signatures from the tensor-only, weights-only payload.
- Audited the fully frozen CPU proxy query, native-to-proxy map, GPU rectification, GPS
  normalization, eval/inference/autocast execution, symmetric-flow split, inverse-depth sign,
  left-right cycle test, confidence, axial-uncertainty conversion, component-native covariance
  lift, score-weighted voxel fusion, stable ranking, exact-count selection, and fail-closed gates.
- Confirmed all structural-baseline and compact-refinement parameters are explicit, including the
  same-pair full-field compact-carve control, common 3,000-Gaussian topology, common 250-step
  refinement, fixed held-out midpoint lattice, no held-out checkpoint selection, three paired
  seeds, rotated execution orders, and the quality-free train-only warmups.
- Confirmed the shuffled-field control changes only field/camera correspondence; its exact-count
  completion rule and numeric degradation gate are separate from qualitative construction
  failure, preventing failed controls from being counted as numeric wins.
- Verified Python 3.11.15, SciPy 1.17.1, `nvidia-ml-py` 13.610.43, and working NVML on the named
  RTX 4090. The frozen fresh-process boundary begins before data, external imports, model and
  checkpoint loading, proxy/index construction, or preprocessing; retains unreset CUDA peaks
  through lift, evaluation, refinement, and save; includes the sanitized checkpoint in GPS input
  bytes; and records allocated, reserved, NVML, RSS, byte, and stage-runtime evidence.
- Confirmed that the experiment driver implementation now exists but remains unexecuted, and that
  no experiment run or result exists. Approval binds the frozen design as fit to execute, not the
  implementation's correctness, an outcome, or a claim.

## Findings

Approved prospectively for the exact digest above. This approval supersedes the prior approval at
digest `aaf07179108ca3224f1360406844e32269094ad809fc238473ae0710a381cc04`: before any run was
initialized, the driver found that its 400,000-byte caller cap was incompatible with the archives'
exact declared loader contract and amended only that cap and its explanation. The corrected
8,388,608-byte value remains fail-closed because archive identity and actual byte counts are sealed
independently, while the loader enforces the declared ceiling and internal integrity checks. The
protocol is sufficiently explicit to test the field-proxy GPS mechanism against its frozen
controls while keeping the VRAM evidence descriptive and the scientific claim boundary narrow.
Any change outside the administrative status and review envelope requires a new digest and a fresh
outcome-unseen review.

## Protected Actions Not Taken

I did not initialize or execute the experiment, invoke its driver, inspect any run directory,
metric, preview, RGB image, or mask, load held-out fields for reconstruction, or evaluate model
outputs. Only task/seal records, loader source, archive ZIP structure and metadata, immutable model
receipts, and environment readiness were inspected. No result-bearing artifact was created or
consumed during this review.
