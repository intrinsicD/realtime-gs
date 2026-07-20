# Evidence Index

- `tables/depth_covariance_probes.md`: seed-0 pilots plus the completed three-iteration,
  three-seed CPU depth-covariance ablation.
- `tables/cpu_baseline_audit.md`: CPU verification, smoke baseline, bottleneck summary, and
  replay-integrity checks.
- `tables/depth_anchor_baseline.md`: exact confidence-invariance check and three-seed uniform
  bounded-ray anchor baseline.
- `tables/depth_anchor_ablation.md`: preregistered three-seed confidence-anchor result and
  post-run attribution audit.
- `tables/depth_anchor_attribution.md`: exact sampled-weight attribution repair, invariants,
  preregistered null result, and stopping decision.
- `tables/cross_view_supervision.md`: preregistered Gradient/Hybrid all-versus-LOSO-versus-matched
  supervision result, source/exposure invariants, independent audit, and stopping pivot.
- `tables/world_position_consistency.md`: preregistered fixed correct-versus-degree-shuffled
  position-consistency result, local/global gates, graph coverage, control limitation, and pivot.
- `tables/dense_train_position.md`: frozen train-only raw-patch/epipolar graph structure, strict
  semantic precision rejection, withheld-arm decision, provenance, and plane/normal pivot.
- `tables/surface_plane_normal.md`: frozen corrupted-depth cross-view plane-target structure,
  post-freeze clean-plane rejection, withheld Hybrid arms, provenance, and calibrated-input pivot.
- `tables/tum_rgbd_oriented_validity.md`: sealed real TUM development/confirmatory target audit,
  mechanically transferred gates, heavy-tail rejection, exact-once provenance, and withheld Phase B.
- `tables/tum_rgbd_signed_attribution.md`: sealed signed TUM visibility audit, target-balanced
  partial occlusion evidence, failed development magnitude gates, temporal sensitivity, and
  unopened walking confirmation.
- `tables/smooth_support_audits.md`: independently audited SH-floor, kernel-tail, and visibility-
  margin experiments, including mechanism gates, held-out effects, and stopping decisions.
- `tables/20260716_stage1_carve_multiscale_quaternion.md`: audited Carve materiality failure,
  qualified Stage-1 gauge dependence, negative 24-to-48 schedule, and two invalid Quaternion
  attempts with explicit no-outcome boundaries.
- `tables/20260716_stage1_fit_parameterization.md`: sealed and independently audited N78
  comparison, exact raw bindings, material local null motion, negative 8p appearance/joint effects,
  and the scoped retain-current disposition.
- `benchmarks/results/20260716T101608Z_cpu_stage1_fit_parameterization_GSPLAT_REPLAY.md`:
  audited post-hoc 2D CUDA gsplat replay of all 108 terminal N78 fits, with compact and exhaustive
  figure links, native-agreement measurements, and an explicit no-3D/no-portability boundary.
- `benchmarks/results/20260714_depth_covariance_REPLAY.md`: source-bound replay manifest and
  effective commands for every official run.
- `runs/dataset_viewer_smoke_20260716/` and the matching `docs/EXPERIMENTS.md` entry: audited
  frame_00008 Torch/CPU integration smoke with strict 7/1 camera isolation, finite initial/final
  PLYs, calibrated previews, independently recomputed metrics, and a live viewer handoff. This is
  explicitly non-replay-complete and supports no quality, performance, default, or 3D-gsplat claim.
- `runs/dataset_viewer_fullres_20260716/fit_manifest.json` and `AUDIT.md`: audited frame_00008
  native-resolution integration with seven finite 640-row train-fit archives, held-out C1004,
  byte-identical finite 835-splat initialization PLYs, and a live `--downscale 1` viewer. Capacity
  was selected post-outcome, the streaming driver is unarchived, and no refined or held-out 3D
  evidence exists; the artifact supports integration only.
- `tables/20260716_point_rasterizer_parity.md`: sealed and independently audited CPU sparse
  point-renderer forward/gradient parity, exact discrete-pixel risk identity, bounded no-RGB
  C0001 sampled interaction over the existing 835-Gaussian PLY, and separate RGB-loading C0000
  viewer smoke, with explicit no-optimization/no-quality/no-performance boundaries.
- `tables/20260716_compact_point_training.md`: sealed four-arm fixed-topology sampling result,
  full-resolution RGB-denied phase-local refinement diagnostics, terminal calibrated viewer failure,
  separately bound post-failure diagnostics, and the no-default/no-overall-PASS boundaries.
- `tables/20260717_stage1_teacher_fidelity.md`: post-failure isolated evaluation showing exact
  archive/CUDA-render parity but visibly low-fidelity 640-Gaussian, 100-step, unmasked full-frame
  teachers; this is a configuration diagnosis, not a StructSplat quality ceiling.
- `tables/20260717_masked_bundle_acquisition.md`: qualified recovery of the immutable seven-view
  masked 640/100 compact-teacher bundle, including strict content integrity, the terminal
  postcondition false positive, and the known incorrect descriptive per-view InitConfig seed.
- `tables/20260720_dense_confidence_gated_init.md`: audited calibrated E1/I1/E2 chain,
  count/quality decisions, late-release held-out result, profiler decomposition, exact target-cache
  parity, artifact hashes, and explicit local-performance/default boundaries.
