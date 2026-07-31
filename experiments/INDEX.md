# Experiment catalog

Use this page to find an experiment by method, date, or data scope. New task-first experiments use
the canonical identifier:

`YYYYMMDD_<task_slug>_<data_slug>`

Historical evidence predates that contract. Its files and run directories are immutable because
claims, checksums, source seals, and audit notes cite their original paths. The names below are
therefore **catalog aliases**, not retroactive task IDs: they apply the current naming scheme
without changing provenance or implying that an old study had a prospective task contract.

Data slugs are deliberately small and reusable:

- `stage_frame00008`, `stage_frame00009`, or `stage_frames00008_00009` for the checked-in Stage
  capture;
- `tum_rgbd` for registered TUM RGB-D inputs;
- `synthetic` for deterministic mechanism fixtures;
- `mixed` when one study deliberately crosses those boundaries;
- `repository` for implementation, integration, or correction records with no result-bearing
  dataset.

The [dated experiment log](../docs/EXPERIMENTS.md) remains the interpretation authority. The
[append-only evidence directory](../benchmarks/results/README.md) remains the artifact authority.

## Task-first registry

These are real task IDs. Their task status remains immutable after a protected run starts, so a
`ready` task may already have a completed or failed result bundle.

| Canonical task ID | Task status | Experiment | Primary record |
| --- | --- | --- | --- |
| `20260731_residual_mixture_sampling_stage_frames00008_00009` | draft | Residual-aware mixture sampling for compact point supervision | [task](tasks/20260731_residual_mixture_sampling_stage_frames00008_00009.json) |
| `20260731_coarse_to_fine_density_stage_frames00008_00009` | draft | Coarse-to-fine standard 3DGS with late densification in one optimizer lifecycle | [task](tasks/20260731_coarse_to_fine_density_stage_frames00008_00009.json) |
| `20260731_selective_teacher_querying_stage_frames00008_00009` | draft | Disagreement-routed selective querying of matched-capacity compact teachers | [task](tasks/20260731_selective_teacher_querying_stage_frames00008_00009.json) |
| `20260731_topology_moment_inheritance_stage_frames00008_00009` | draft | Parent-to-child Adam moment inheritance at arena clone/split topology events | [task](tasks/20260731_topology_moment_inheritance_stage_frames00008_00009.json) |
| `20260731_active_set_updates_stage_frames00008_00009` | draft | Priority-ranked active-set parameter updates from existing strategy statistics | [task](tasks/20260731_active_set_updates_stage_frames00008_00009.json) |
| `20260730_paper_three_path_fullres_stage_frames00008_00009` | draft | Full-resolution compact-field 3DGS from random, Splat-SfM, and Beam initialization | [task](tasks/20260730_paper_three_path_fullres_stage_frames00008_00009.json) |
| `20260730_field_sweep_placement_f64_stage_frames00008_00009` | ready | Robust compact-field plane-sweep placement, float64 successor | [result](../benchmarks/results/20260730_field_sweep_placement_f64_stage_frames00008_00009_RESULT.md) · [audit](../benchmarks/results/20260730_field_sweep_placement_f64_stage_frames00008_00009_AUDIT.md) |
| `20260730_additive_analytic_objective_stage_frames00008_00009` | draft | Additive-field analytic objective versus point-sampled compositing | [task](tasks/20260730_additive_analytic_objective_stage_frames00008_00009.json) |
| `20260729_field_sweep_placement_stage_frames00008_00009` | ready | Robust compact-field plane-sweep placement, consumed failed predecessor | [result](../benchmarks/results/20260729_field_sweep_placement_stage_frames00008_00009_RESULT.md) · [audit](../benchmarks/results/20260729_field_sweep_placement_stage_frames00008_00009_AUDIT.md) |
| `20260728_vram_claim_stage_frames00008_00009` | draft | Direct compact 2D-to-3D reconstruction-process VRAM question | [task](tasks/20260728_vram_claim_stage_frames00008_00009.json) |
| `20260728_rgb_3dgs_comparison_stage_frames00008_00009` | draft | RGB-trained 3DGS from a matched compact initialization | [task](tasks/20260728_rgb_3dgs_comparison_stage_frames00008_00009.json) |
| `20260728_beam_fusion_claim_stage_frames00008_00009` | draft | Matched Beam Fusion initializer and stage ablation | [task](tasks/20260728_beam_fusion_claim_stage_frames00008_00009.json) |

## Historical catalog aliases

Each row names one historical study or independently interpretable arm. Search the alias to find
the relevant method and data scope; follow the primary record for the original filenames, run
roots, protocol revisions, and related receipts. The catalog covers every named historical
`PREREG` and `RESULT` family, including blocked, failed, withdrawn, and never-run protocols, plus
log-only mechanism and integration studies. Timestamped raw receipts retain their original names
and are reached through those primary records.

### 2026-07-28

| Catalog alias | Experiment | Primary record |
| --- | --- | --- |
| `20260728_compact_only_carrier_stage_ablation_stage_frame00008` | Compact-only carrier stage ablation | [result](../benchmarks/results/20260728_compact_only_carrier_stage_ablation_RESULT.md) · [audit](../benchmarks/results/20260728_compact_only_carrier_stage_ablation_AUDIT.md) |
| `20260728_compact_only_carrier_policy_closure_stage_frame00008` | Compact-only carrier policy closure | [result](../benchmarks/results/20260728_compact_only_carrier_policy_closure_RESULT.md) · [audit](../benchmarks/results/20260728_compact_only_carrier_policy_closure_AUDIT.md) |
| `20260728_compact_only_carrier_sequence_interaction_stage_frame00008` | Compact-only carrier sequence interaction | [result](../benchmarks/results/20260728_compact_only_carrier_sequence_interaction_RESULT.md) · [audit](../benchmarks/results/20260728_compact_only_carrier_sequence_interaction_AUDIT.md) |

### 2026-07-27

| Catalog alias | Experiment | Primary record |
| --- | --- | --- |
| `20260727_carrier_maturation_all26_stage_frame00008` | All-26 carrier maturation | [result](../benchmarks/results/20260727_carrier_maturation_all26_RESULT.md) · [audit](../benchmarks/results/20260727_carrier_maturation_all26_AUDIT.md) |
| `20260727_adr002_scope_correction_repository` | Scope correction: the ADR-002 convergence pipeline was not tested | [log](../docs/EXPERIMENTS.md) |
| `20260727_carrier_refinement_fullres_stage_frame00008` | Full-resolution ADR-002 carrier refinement | [result](../benchmarks/results/20260727_carrier_refinement_fullres_RESULT.md) · [audit](../benchmarks/results/20260727_carrier_refinement_fullres_AUDIT.md) |
| `20260727_carrier_refinement_recovery_stage_frame00008` | Post-outcome fixed-topology recovery protocol | [preregistration](../benchmarks/results/20260727_carrier_refinement_recovery_PREREG.md) |
| `20260727_surfel_init_schedule_screen_stage_frames00008_00009` | Surfel initialization parameters and init-preserving schedule | [result](../benchmarks/results/20260727_surfel_init_schedule_screen_RESULT.md) · [audit](../benchmarks/results/20260727_surfel_init_schedule_screen_AUDIT.md) |

### 2026-07-26

| Catalog alias | Experiment | Primary record |
| --- | --- | --- |
| `20260726_init_value_dev_screen_karate_frames00005_00060` | Withdrawn-protocol initialization-value development screen | [result](../benchmarks/results/20260726_init_value_dev_screen_RESULT.md) · [withdrawal](../benchmarks/results/20260726_init_value_program_REWRITE_WITHDRAWN.md) |

### 2026-07-25

| Catalog alias | Experiment | Primary record |
| --- | --- | --- |
| `20260725_residual_decomposition_stage_frame00009` | Stage-0 residual decomposition: interior holes versus silhouette error | [driver](../benchmarks/residual_decomposition.py) · [runbook](../benchmarks/results/20260725_gpu_RUNBOOK.md) |
| `20260725_gpu_stage1_initialization_stage_frame00009` | Cover-consistent Stage-1 initialization under the production GPU stack | [preregistration](../benchmarks/results/20260725_gpu_stage1_initialization_PREREG.md) |
| `20260725_init_cost_to_target_stage_frame00009` | Initialization value as optimization cost-to-target | [preregistration](../benchmarks/results/20260725_init_cost_to_target_PREREG.md) |
| `20260725_init_refit_ceiling_stage_frame00009` | Beam-mean refit-ceiling factorial | [preregistration](../benchmarks/results/20260725_init_refit_ceiling_PREREG.md) |
| `20260725_init_value_program_mixed` | Initialization Value Program, blocked preregistration | [preregistration](../benchmarks/results/20260725_init_value_program_PREREG.md) · [failed review](../benchmarks/results/20260725_init_value_program_PREREG_REVIEW_02_FAIL.md) |

### 2026-07-24

| Catalog alias | Experiment | Primary record |
| --- | --- | --- |
| `20260724_beam_surfel_scale_gradient_stage_frame00009` | Why an under-sized primitive does not grow before density events | [result](../benchmarks/results/20260724_beam_surfel_scale_gradient_RESULT.md) |
| `20260724_beam_surfel_birth_attribution_stage_frame00009` | Original-versus-newborn Beam/Surfel attribution | [result](../benchmarks/results/20260724_beam_surfel_birth_attribution_RESULT.md) |
| `20260724_beam_surfel_init_stage_frames00008_00009` | Cover-consistent Surfel initialization for Beam Fusion | [result](../benchmarks/results/20260724_beam_surfel_init_RESULT.md) · [preregistration](../benchmarks/results/20260724_beam_surfel_init_PREREG.md) |
| `20260724_beam_surfel_schedule_confound_stage_frame00009` | Initialization scale versus density-schedule confound | [preregistration](../benchmarks/results/20260724_beam_surfel_schedule_confound_PREREG.md) |
| `20260724_beam_surfel_matched_capacity_stage_frames00008_00009` | Cover-consistent covariance/opacity at matched capacity | [result](../benchmarks/results/20260724_beam_surfel_matched_capacity_RESULT.md) |
| `20260724_geometric_arena_stage_frame00008` | Geometric Stage-3 Gaussian arena | [result](../benchmarks/results/20260724_geometric_arena_frame00008_RESULT.md) · [audit](../benchmarks/results/20260724_geometric_arena_frame00008_AUDIT.md) |
| `20260724_pool_structure_wse_10k_stage_frame00008` | Pooled structure/WSE trajectories over a fresh 10k schedule | [result](../benchmarks/results/20260724_pool_structure_wse_10k_frame00008_RESULT.md) · [audit](../benchmarks/results/20260724_pool_structure_wse_10k_frame00008_AUDIT.md) |
| `20260724_pool_structure_wse_stage_frame00008` | Pool and structure tensor with and without WSE | [result](../benchmarks/results/20260724_pool_structure_wse_frame00008_RESULT.md) · [audit](../benchmarks/results/20260724_pool_structure_wse_frame00008_AUDIT.md) |
| `20260724_new_variants_stage_frame00008` | All new opt-in variants on Janelle | [result](../benchmarks/results/20260724_new_variants_frame00008_RESULT.md) · [audit](../benchmarks/results/20260724_new_variants_frame00008_AUDIT.md) |

### 2026-07-23

| Catalog alias | Experiment | Primary record |
| --- | --- | --- |
| `20260723_stage1_gaussian_pool_synthetic` | Fixed-capacity 2D-Gaussian pool and free list | [implementation](../src/rtgs/image2gs/pool.py) · [tests](../tests/test_image2gs_pool.py) |
| `20260723_beam_partition_opacity_probe_stage_frame00008` | Post-hoc Beam partition optical-thickness probe | [result](../benchmarks/results/20260723_beam_partition_opacity_probe_RESULT.md) · [audit](../benchmarks/results/20260723_beam_partition_opacity_probe_AUDIT.json) |
| `20260723_beam_partition_covariance_stage_frame00008` | Masked native-anchor Beam density partitions | [result](../benchmarks/results/20260723_beam_partition_covariance_RESULT.md) · [audit](../benchmarks/results/20260723_beam_partition_covariance_AUDIT.md) |
| `20260723_beam_covariance_refit_stage_frame00008` | Beam-track LSQ/robust covariance refit | [result](../benchmarks/results/20260723_beam_covariance_refit_RESULT.md) · [audit](../benchmarks/results/20260723_beam_covariance_refit_AUDIT.md) |
| `20260723_beam_convergence_replication_stage_frame00008` | Independent replication of Beam convergence dynamics | [result](../benchmarks/results/20260723_beam_convergence_dynamics_REPLICATION_RESULT.md) · [audit](../benchmarks/results/20260723_beam_convergence_dynamics_REPLICATION_AUDIT.md) |
| `20260723_beam_convergence_dynamics_stage_frame00008` | Why on-surface Beam initialization does not help densified convergence | [log](../docs/EXPERIMENTS.md) · [driver](../benchmarks/beam_convergence_dynamics.py) |

### 2026-07-22

| Catalog alias | Experiment | Primary record |
| --- | --- | --- |
| `20260722_compact_query_budget_checkpointing_repository` | Aggregate CSR index budgets and checkpointed pair-chunk queries | [log](../docs/EXPERIMENTS.md) · [tests](../tests/test_compact_budgets_and_checkpoint.py) |
| `20260722_all_initializers_viewer_stage_frame00008` | Seven-initializer endpoint comparison viewer | [viewer manifest](../benchmarks/results/20260721_all_initializers_frame00008_VIEWER.json) |

### 2026-07-21

| Catalog alias | Experiment | Primary record |
| --- | --- | --- |
| `20260721_compact_query_cuda_repository` | Indexed CUDA compact-teacher query backend | [log](../docs/EXPERIMENTS.md) · [tests](../tests/test_observation2d_cuda.py) |
| `20260721_stage1_batch_views_synthetic` | Fused multi-view Stage-1 fitting and CUDA extension skeleton | [benchmark](../benchmarks/results/20260721T191424Z_cpu.json) |
| `20260721_all_initializers_stage_frame00008` | Full compact-compatible initializer convergence suite | [result](../benchmarks/results/20260721_all_initializers_frame00008_RESULT.md) · [audit](../benchmarks/results/20260721_all_initializers_frame00008_AUDIT.md) |
| `20260721_beam_fusion_full_stage_frame00008` | Full bounded Beam Fusion and convergence | [result](../benchmarks/results/20260721_beam_fusion_full_frame00008_RESULT.md) · [audit](../benchmarks/results/20260721_beam_fusion_full_frame00008_AUDIT.md) |
| `20260721_structsplat_teacher_gallery_stage_frames00008_00009` | Full compact StructSplat 2D reconstruction gallery | [log](../docs/EXPERIMENTS.md) |
| `20260721_beam_fusion_initializer_synthetic` | Tomographic Gaussian Beam Fusion mechanism | [log](../docs/EXPERIMENTS.md) · [screen](../benchmarks/run_compact_initializer_suite.py) |
| `20260721_splat_sfm_initializer_synthetic` | Structure-from-splats calibrated SfM analog | [log](../docs/EXPERIMENTS.md) · [screen](../benchmarks/splat_sfm_screen.py) |

### 2026-07-20

| Catalog alias | Experiment | Primary record |
| --- | --- | --- |
| `20260720_dense_confidence_gated_init_e1_stage_frame00008` | Dense confidence-gated initialization, E1 | [result](../benchmarks/results/20260720_dense_confidence_gated_init_e1_RESULT.md) |
| `20260720_dense_confidence_gated_init_i1_stage_frame00008` | Dense confidence-gated initialization, I1 | [result](../benchmarks/results/20260720_dense_confidence_gated_init_i1_RESULT.md) |
| `20260720_dense_confidence_gated_init_e2_stage_frame00008` | Dense confidence-gated initialization, E2 | [result](../benchmarks/results/20260720_dense_confidence_gated_init_e2_RESULT.md) · [audit](../benchmarks/results/20260720_dense_confidence_gated_init_e2_AUDIT.md) |
| `20260720_dense_voxel_refine_synthetic` | Dense all-Gaussian initialization, voxel merge, and 4-DoF refine | [log](../docs/EXPERIMENTS.md) |
| `20260720_observation_csr_index_repository` | Flattened exact CPU CSR observation index | [log](../docs/EXPERIMENTS.md) · [benchmark](../benchmarks/run.py) |
| `20260720_full_compact_reconstruction_stage_frame00008` | Full compact all-view reconstruction and placement diagnosis | [driver](../benchmarks/full_compact_reconstruction.py) · [log](../docs/EXPERIMENTS.md) |

### 2026-07-18

| Catalog alias | Experiment | Primary record |
| --- | --- | --- |
| `20260718_structsplat_masked_168kb_stage_frame00008` | Mask-gated StructSplat view under a 168,000-byte cap | [result](../benchmarks/results/20260718_structsplat_masked_168kb_example_RESULT.md) · [audit](../benchmarks/results/20260718_structsplat_masked_168kb_example_AUDIT.md) |
| `20260718_inverse_projection_fiber_iter3_synthetic` | Capacity-aware inverse-projection-fiber correspondence, iteration 3 | [failure audit](../benchmarks/results/20260717_inverse_projection_fiber_iter3_FAILURE_AUDIT.md) |

### 2026-07-17

| Catalog alias | Experiment | Primary record |
| --- | --- | --- |
| `20260717_inverse_projection_fiber_iter1_synthetic` | Exact inverse-projection fibers, initial consumed attempts | [preregistration](../benchmarks/results/20260717_inverse_projection_fiber_iter1_PREREG.md) |
| `20260717_inverse_projection_fiber_iter1b_synthetic` | Inverse-projection fibers, implementation-reviewed iteration 1b | [preregistration](../benchmarks/results/20260717_inverse_projection_fiber_iter1b_PREREG.md) · [review](../benchmarks/results/20260717_inverse_projection_fiber_iter1b_IMPLEMENTATION_REVIEW.md) |
| `20260717_inverse_projection_fiber_iter1c_synthetic` | Inverse-projection fibers, polluted/rejected iteration 1c | [preregistration](../benchmarks/results/20260717_inverse_projection_fiber_iter1c_PREREG.md) · [pollution inventory](../benchmarks/results/20260717_inverse_projection_fiber_iter1c_POLLUTION_INVENTORY.md) |
| `20260717_inverse_projection_fiber_iter1d_synthetic` | Inverse-projection fibers, review-failed iteration 1d | [preregistration](../benchmarks/results/20260717_inverse_projection_fiber_iter1d_PREREG.md) · [failed review](../benchmarks/results/20260717_inverse_projection_fiber_iter1d_PREREG_REVIEW_FAIL.md) |
| `20260717_inverse_projection_fiber_iter1e_synthetic` | Exact inverse-projection fibers, valid iteration 1e | [result](../benchmarks/results/20260717_inverse_projection_fiber_iter1e_RESULT.md) · [audit](../benchmarks/results/20260717_inverse_projection_fiber_iter1e_AUDIT.md) |
| `20260717_inverse_projection_fiber_iter2_synthetic` | Residual topology repair for inverse-projection fibers | [result](../benchmarks/results/20260717_inverse_projection_fiber_iter2_RESULT.md) · [audit](../benchmarks/results/20260717_inverse_projection_fiber_iter2_AUDIT.md) |
| `20260717_compact_responsibility_birth_allocation_iter1_stage_frame00008` | Compact residual-responsibility birth allocation, lifecycle-failed iteration 1 | [preregistration](../benchmarks/results/20260717_compact_responsibility_birth_allocation_PREREG.md) · [failure audit](../benchmarks/results/20260717_compact_responsibility_birth_allocation_FAILURE_AUDIT.md) |
| `20260717_compact_responsibility_birth_allocation_iter2_stage_frame00008` | Compact residual-responsibility birth allocation, lifecycle-failed iteration 2 | [preregistration](../benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_PREREG.md) · [failure audit](../benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_FAILURE_AUDIT.md) |
| `20260717_compact_responsibility_birth_allocation_iter3_stage_frame00008` | Compact residual-responsibility birth allocation, review-gated iteration 3 | [preregistration](../benchmarks/results/20260717_compact_responsibility_birth_allocation_iter3_PREREG.md) · [review](../benchmarks/results/20260717_compact_responsibility_birth_allocation_iter3_PREREG_REVIEW.md) |
| `20260717_compact_occupancy_refinement_factorial_iter1_synthetic` | Compact proposal-target refinement factorial, first attempt | [result](../benchmarks/results/20260717_compact_occupancy_refinement_factorial_RESULT.json) · [failure audit](../benchmarks/results/20260717_compact_occupancy_refinement_factorial_FAILURE_AUDIT.md) |
| `20260717_compact_occupancy_refinement_factorial_iter2_synthetic` | Compact proposal-target refinement factorial, iteration 2 | [result](../benchmarks/results/20260717_compact_occupancy_refinement_factorial_iter2_RESULT.json) · [failure audit](../benchmarks/results/20260717_compact_occupancy_refinement_factorial_iter2_FAILURE_AUDIT.md) |
| `20260717_compact_occupancy_refinement_factorial_iter3_synthetic` | Compact proposal-target refinement factorial, iteration 3 | [result](../benchmarks/results/20260717_compact_occupancy_refinement_factorial_iter3_RESULT.json) · [audit](../benchmarks/results/20260717_compact_occupancy_refinement_factorial_iter3_AUDIT.md) |
| `20260717_gaussianimage_plus_provider_parity_synthetic` | GaussianImage-plus direct-covariance provider parity | [result](../benchmarks/results/20260717_gaussianimage_plus_provider_parity_RESULT.json) · [audit](../benchmarks/results/20260717_gaussianimage_plus_provider_parity_AUDIT.md) |
| `20260717_stage1_mask_residual_screen_stage_frame00008` | One-view full-resolution Stage-1 mask and residual-growth screen | [driver](../benchmarks/compact_stage1_mask_screen.py) · [log](../docs/EXPERIMENTS.md) |
| `20260717_compact_teacher_acquisition_stage_frame00008` | Seven-view masked 640/100 compact-teacher acquisition | [driver](../benchmarks/compact_masked_bundle_acquisition.py) · [log](../docs/EXPERIMENTS.md) |
| `20260717_compact_lift_occupancy_screen_stage_frame00008` | Masked compact-lift occupancy screen and replay qualification | [driver](../benchmarks/compact_masked_lift_screen.py) · [log](../docs/EXPERIMENTS.md) |
| `20260717_compact_occupancy_scalar_stage_frame00008` | Footprint occupancy-scalar ablation | [driver](../benchmarks/compact_occupancy_scalar_ablation.py) · [log](../docs/EXPERIMENTS.md) |

### 2026-07-16

| Catalog alias | Experiment | Primary record |
| --- | --- | --- |
| `20260716_compact_point_training_mixed` | RGB-free compact point refinement and full-resolution interaction | [result](../benchmarks/results/20260716_compact_point_training_RESULT.json) · [audit](../benchmarks/results/20260716_compact_point_training_AUDIT.md) |
| `20260716_point_rasterizer_parity_synthetic` | Sparse point compositor and discrete-pixel proposal parity | [result](../benchmarks/results/20260716_point_rasterizer_parity_RESULT.json) · [audit](../benchmarks/results/20260716_point_rasterizer_parity_AUDIT.md) |
| `20260716_compact_carve_synthetic` | RGB-free compact-Carve initialization | [log](../docs/EXPERIMENTS.md) |
| `20260716_structsplat_teacher_contract_synthetic` | Exact RGB-free StructSplat teacher contract | [log](../docs/EXPERIMENTS.md) |
| `20260716_dataset_viewer_fullres_stage_frame00008` | Native-resolution calibrated-data viewer handoff | [log](../docs/EXPERIMENTS.md) |
| `20260716_dataset_viewer_smoke_mixed` | Local calibrated-data and viewer workflow smoke | [log](../docs/EXPERIMENTS.md) |
| `20260716_quaternion_gauge_iter1_synthetic` | Quaternion radial-gauge optimizer audit, invalid first attempt | [result](../benchmarks/results/20260716T015517Z_cpu_quaternion_gauge_invalid_RESULT.md) · [audit](../benchmarks/results/20260716T015517Z_cpu_quaternion_gauge_invalid_AUDIT.md) |
| `20260716_quaternion_gauge_iter2_synthetic` | Quaternion radial-gauge optimizer audit, invalid second attempt | [result](../benchmarks/results/20260716T030759Z_cpu_quaternion_gauge_iter2_invalid_RESULT.md) · [audit](../benchmarks/results/20260716T030759Z_cpu_quaternion_gauge_iter2_invalid_AUDIT.md) |
| `20260716_stage1_semantic_factorial_synthetic` | Gauge-invariant Stage-1-to-lifter semantic factorial | [mechanism](../benchmarks/results/20260716T061754Z_cpu_stage1_semantic_factorial_mechanism_RESULT.md) · [utility](../benchmarks/results/20260716T063637Z_cpu_stage1_semantic_factorial_utility_RESULT.md) |
| `20260716_stage1_fit_parameterization_infrastructure_repository` | Stage-1 fit-time parameterization infrastructure | [implementation review](../benchmarks/results/20260716_stage1_fit_parameterization_IMPLEMENTATION_REVIEW.md) |
| `20260716_stage1_fit_parameterization_synthetic` | Stage-1 fit-time parameterization outcome | [result](../benchmarks/results/20260716T101608Z_cpu_stage1_fit_parameterization_RESULT.md) |
| `20260716_stage1_weight_gauge_synthetic` | Stage-1 weight/color gauge contract | [result](../benchmarks/results/20260716_stage1_weight_gauge_SEAL_RESULT.md) |
| `20260716_multiscale_refinement_synthetic` | Fixed-topology 24-to-48 multiscale refinement | [result](../benchmarks/results/20260716T003735Z_cpu_multiscale_refinement_RESULT.md) |
| `20260716_carve_merge_controls_synthetic` | Carve equal-count merge controls, final consumed iteration | [result](../benchmarks/results/20260716_carve_merge_controls_iter2_SEAL_RESULT.md) |
| `20260716_residual_responsibility_density_synthetic` | Residual-responsibility density allocation, preregistered but not run | [preregistration](../benchmarks/results/20260716_residual_responsibility_density_PREREG.md) · [review](../benchmarks/results/20260716_residual_responsibility_density_PREREG_REVIEW.md) |

### 2026-07-15

| Catalog alias | Experiment | Primary record |
| --- | --- | --- |
| `20260715_visibility_margin_synthetic` | Coarse visibility-margin support audit | [result](../benchmarks/results/20260715_visibility_margin_iter2_SEAL_RESULT.md) |
| `20260715_kernel_support_taper_synthetic` | Hard kernel-support C1 taper | [result](../benchmarks/results/20260715_kernel_support_taper_iter2_SEAL_RESULT.md) |
| `20260715_sh_activation_synthetic` | SH color-floor incidence and SMU-1 | [result](../benchmarks/results/20260715_sh_activation_iter2_SEAL_RESULT.md) |
| `20260715_signed_occlusion_attribution_tum_rgbd` | Signed RGB-D occlusion attribution | [result](../benchmarks/results/20260715_tum_rgbd_signed_attribution_RESULT.md) |
| `20260715_oriented_point_validity_tum_rgbd` | Registered-RGB-D oriented-point transfer | [result](../benchmarks/results/20260715_tum_rgbd_oriented_validity_RESULT.md) |
| `20260715_surface_plane_normal_mixed` | Local-plane and shortest-axis targets | [result](../benchmarks/results/20260715_surface_plane_normal_RESULT.md) |
| `20260715_dense_train_position_synthetic` | Dense train-only patch/epipolar matcher | [result](../benchmarks/results/20260715_dense_train_position_RESULT.md) |
| `20260715_world_position_consistency_synthetic` | Fixed-match world-frame position consistency | [result](../benchmarks/results/20260715_world_position_consistency_RESULT.md) |
| `20260715_cross_view_supervision_synthetic` | Leave-one-source-view-out photometric supervision | [result](../benchmarks/results/20260715_cross_view_supervision_RESULT.md) |
| `20260715_depth_anchor_attribution_synthetic` | Exact sampled-confidence attribution repair | [result](../benchmarks/results/20260715_depth_anchor_attribution_RESULT.md) |
| `20260715_depth_anchor_synthetic` | Confidence-weighted bounded-ray anchor | [audit](../benchmarks/results/20260715_depth_anchor_AUDIT.md) |

### 2026-07-14 and earlier

| Catalog alias | Experiment | Primary record |
| --- | --- | --- |
| `20260714_depth_covariance_synthetic` | Three-iteration depth-covariance ablation | [replay](../benchmarks/results/20260714_depth_covariance_REPLAY.md) |
| `20260714_gsplat_density_stage_frame00008` | gsplat density strategies, full-SH convergence, and novel-view repair | [log](../docs/EXPERIMENTS.md) |
| `20260714_compact_2d_cuda_ablation_mixed` | Compact 2D starts, strict held-out metrics, and CUDA Janelle ablation | [log](../docs/EXPERIMENTS.md) |
| `20260713_geometry_device_smoke_mixed` | Geometry/device correctness and calibrated Janelle smoke | [log](../docs/EXPERIMENTS.md) |
| `20260708_gradient_depth_rotation_scale_synthetic` | Gradient lift with depth, rotation, scale, and merge | [benchmark](../benchmarks/results/20260708T225210Z_cpu.json) |
| `20260707_pipeline_v1_synthetic` | Pipeline-v1 sanity | [benchmark](../benchmarks/results/20260707T130703Z_cpu.json) |

## Original paths remain authoritative

Do not rename an old result, driver, or run root to match a catalog alias. When citing or replaying
historical work:

1. use the catalog alias for discovery and discussion;
2. use the linked legacy path for hashes, commands, and evidence;
3. use a new task ID for any materially changed protocol or rerun.
