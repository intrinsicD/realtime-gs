"""CPU contracts for ADR-002 carrier repair and maturation."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rtgs.carrier_pipeline import CarrierPipelineConfig, run_carrier_pipeline
from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.core.sh import sh_to_rgb
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.beam_fusion import BeamFusionConfig, BeamFusionResult
from rtgs.lift.carrier_refinement import (
    CarrierObservationTable,
    CarrierRepairConfig,
    appearance_link_residuals,
    build_carrier_observation_table,
    covariance_reprojection_residuals,
    opacity_optical_density_residuals,
    refine_carrier_appearance,
    refine_carrier_covariances,
    refine_carrier_opacities,
)
from rtgs.optim.carrier_schedule import (
    CarrierOptimizationConfig,
    optimize_carriers,
    prune_carrier_centers,
)
from rtgs.optim.compact_trainer import CompactTrainConfig
from rtgs.optim.density import DensityConfig, DensityController
from rtgs.render.projection import project_covariances_ewa


def _camera(x: float, y: float) -> Camera:
    return Camera.look_at(
        eye=torch.tensor([x, y, -2.8]),
        target=torch.tensor([0.0, 0.0, 0.1]),
        width=32,
        height=28,
        fov_x_deg=52.0,
    )


def _covariance_to_rs(covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.tensor([1, 0])
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = torch.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
    return 0.5 * eigenvalues.log(), angle


def _fixture(*, dilation: float = 0.0) -> tuple[BeamFusionResult, ReconstructionInputs]:
    means = torch.tensor([[-0.15, -0.06, 0.02], [0.14, 0.10, 0.16]])
    true_covariances = torch.tensor(
        [
            [[0.010, 0.0015, -0.0008], [0.0015, 0.006, 0.0007], [-0.0008, 0.0007, 0.004]],
            [[0.005, -0.0006, 0.0004], [-0.0006, 0.009, 0.0012], [0.0004, 0.0012, 0.006]],
        ]
    )
    cameras = [_camera(-0.8, 0.0), _camera(0.75, 0.15), _camera(0.0, 0.85)]
    colors = torch.tensor([[0.78, 0.20, 0.14], [0.12, 0.68, 0.82]])
    observations = []
    for view, camera in enumerate(cameras):
        projected = project_covariances_ewa(
            means,
            true_covariances,
            camera,
            dilation=dilation,
        )
        log_scales, rotations = [], []
        for covariance in projected.covariances2d:
            scale, rotation = _covariance_to_rs(covariance)
            log_scales.append(scale)
            rotations.append(rotation)
        observations.append(
            GaussianObservationField(
                width=camera.width,
                height=camera.height,
                means=projected.means2d,
                log_scales=torch.stack(log_scales),
                rotations=torch.stack(rotations),
                colors=(colors + torch.tensor([0.02 * view, -0.01 * view, 0.01])).clamp(0, 1),
                amplitudes=torch.tensor([0.55 + 0.08 * view, 0.72 - 0.06 * view]),
                view_id=f"C{view:04d}",
                provider="synthetic_fixture",
            )
        )
    inputs = ReconstructionInputs(
        observations=observations,
        cameras=cameras,
        view_names=[f"C{view:04d}" for view in range(3)],
        bounds_hint=(torch.zeros(3), 1.0),
        name="carrier-test",
    )
    initial_covariances = true_covariances.clone()
    initial_covariances[:, 0, 0] *= 1.8
    initial_covariances[:, 1, 1] *= 0.65
    initial_covariances[:, 2, 2] *= 1.3
    initial_covariances = 0.5 * (initial_covariances + initial_covariances.transpose(-1, -2))
    gaussians = Gaussians3D.from_means_covs(
        means,
        initial_covariances,
        colors=torch.tensor([[0.3, 0.3, 0.3], [0.4, 0.4, 0.4]]),
        opacity=torch.full((2,), 0.1),
    )
    view_indices = torch.tensor([0, 1, 2, 0, 1, 2])
    component_indices = torch.tensor([0, 0, 0, 1, 1, 1])
    depths = torch.cat(
        [
            torch.stack([camera.project(means[index : index + 1])[1][0] for camera in cameras])
            for index in range(2)
        ]
    ).to(torch.float64)
    result = BeamFusionResult(
        gaussians=gaussians,
        component_offsets=torch.tensor([0, 3, 6]),
        contributor_view_indices=view_indices,
        contributor_component_indices=component_indices,
        contributor_depths=depths,
        component_weights=torch.ones(2),
        unmatched_per_view=(0, 0, 0),
        diagnostics={},
    )
    return result, inputs


def test_carrier_repairs_reduce_fixed_track_residuals_and_freeze_other_fields():
    result, inputs = _fixture()
    table = build_carrier_observation_table(result, inputs)
    config = CarrierRepairConfig(
        covariance_steps=180,
        opacity_steps=100,
        covariance_prior_weight=0.0,
        opacity_prior_weight=0.0,
        covariance_residual_mode="legacy_whitened",
        projection_dilation=0.0,
        appearance_weight_mode="amplitude",
    )

    covariance_before = covariance_reprojection_residuals(
        table, result.gaussians.covariance().double()
    ).mean()
    covariance, covariance_diagnostics = refine_carrier_covariances(
        result.gaussians,
        table,
        extent=1.0,
        config=config,
    )
    covariance_after = covariance_reprojection_residuals(
        table, covariance.covariance().double()
    ).mean()
    assert covariance_after < covariance_before * 0.10
    assert torch.equal(covariance.means, result.gaussians.means)
    assert torch.equal(covariance.opacity, result.gaussians.opacity)
    assert torch.equal(covariance.sh, result.gaussians.sh)
    assert covariance_diagnostics["means_frozen_bit_exact"]
    eigenvalues = torch.linalg.eigvalsh(covariance.covariance())
    assert bool((eigenvalues > 0).all())

    opacity_before = opacity_optical_density_residuals(covariance, table).mean()
    opacity, _ = refine_carrier_opacities(covariance, table, config=config)
    opacity_after = opacity_optical_density_residuals(opacity, table).mean()
    assert opacity_after < opacity_before * 0.25
    assert torch.equal(opacity.means, covariance.means)
    assert torch.equal(opacity.quats, covariance.quats)
    assert torch.equal(opacity.log_scales, covariance.log_scales)
    assert torch.equal(opacity.sh, covariance.sh)

    appearance_before = appearance_link_residuals(opacity, table).mean()
    appearance, diagnostics = refine_carrier_appearance(opacity, table, config=config)
    appearance_after = appearance_link_residuals(appearance, table).mean()
    assert appearance_after < appearance_before * 0.25
    assert appearance.sh.shape[1] == 1
    assert diagnostics["higher_sh_bands"] == "disabled"
    assert torch.equal(appearance.means, opacity.means)
    assert torch.equal(appearance.opacity, opacity.opacity)


def test_renderer_log_covariance_residual_is_reciprocal_and_includes_ewa_dilation():
    table = CarrierObservationTable(
        carrier_indices=torch.tensor([0]),
        view_indices=torch.tensor([0]),
        component_indices=torch.tensor([0]),
        jacobians=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float64),
        covariances2d=torch.eye(2, dtype=torch.float64)[None],
        amplitudes=torch.ones(1, dtype=torch.float64),
        colors=torch.ones(1, 3, dtype=torch.float64),
    )
    expanded = torch.diag(torch.tensor([4.0, 4.0, 1.0], dtype=torch.float64))[None]
    contracted = torch.diag(torch.tensor([0.25, 0.25, 1.0], dtype=torch.float64))[None]
    legacy_expanded = covariance_reprojection_residuals(table, expanded)
    legacy_contracted = covariance_reprojection_residuals(table, contracted)
    assert not torch.allclose(legacy_expanded, legacy_contracted)

    log_expanded = covariance_reprojection_residuals(
        table,
        expanded,
        mode="renderer_log",
    )
    log_contracted = covariance_reprojection_residuals(
        table,
        contracted,
        mode="renderer_log",
    )
    torch.testing.assert_close(log_expanded, log_contracted)
    rendered_match = torch.diag(torch.tensor([0.7, 0.7, 1.0], dtype=torch.float64))[None]
    residual = covariance_reprojection_residuals(
        table,
        rendered_match,
        projection_dilation=0.3,
        mode="renderer_log",
    )
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-12, rtol=0)


def test_default_carrier_repair_is_renderer_aware_covariance_only():
    config = CarrierRepairConfig()

    assert config.repair_covariance
    assert not config.repair_opacity
    assert not config.repair_appearance
    assert config.covariance_residual_mode == "renderer_log"
    assert config.projection_dilation == 0.3
    assert config.appearance_weight_mode == "uniform_view"


def test_renderer_log_covariance_repair_reduces_rendered_footprint_residual():
    result, inputs = _fixture(dilation=0.3)
    table = build_carrier_observation_table(result, inputs)
    config = CarrierRepairConfig(
        covariance_steps=180,
        covariance_prior_weight=0.0,
        covariance_residual_mode="renderer_log",
        projection_dilation=0.3,
    )
    before = covariance_reprojection_residuals(
        table,
        result.gaussians.covariance().double(),
        projection_dilation=config.projection_dilation,
        mode=config.covariance_residual_mode,
    ).mean()
    repaired, diagnostics = refine_carrier_covariances(
        result.gaussians,
        table,
        extent=1.0,
        config=config,
    )
    after = covariance_reprojection_residuals(
        table,
        repaired.covariance().double(),
        projection_dilation=config.projection_dilation,
        mode=config.covariance_residual_mode,
    ).mean()
    assert after < before * 0.15
    assert diagnostics["residual_mode"] == "renderer_log"
    assert diagnostics["projection_dilation"] == 0.3


def test_view_uniform_appearance_does_not_treat_normalized_amplitude_as_confidence():
    base = Gaussians3D.from_means_covs(
        means=torch.zeros(1, 3),
        covs=torch.eye(3)[None],
        colors=torch.full((1, 3), 0.5),
        opacity=torch.tensor([0.1]),
    )
    table = CarrierObservationTable(
        carrier_indices=torch.tensor([0, 0, 0]),
        view_indices=torch.tensor([0, 1, 2]),
        component_indices=torch.tensor([0, 0, 0]),
        jacobians=torch.zeros(3, 2, 3, dtype=torch.float64),
        covariances2d=torch.eye(2, dtype=torch.float64).repeat(3, 1, 1),
        amplitudes=torch.tensor([1.0, 0.01, 0.01], dtype=torch.float64),
        colors=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        ),
    )
    amplitude, amplitude_diagnostics = refine_carrier_appearance(
        base,
        table,
        config=CarrierRepairConfig(appearance_weight_mode="amplitude"),
    )
    uniform, uniform_diagnostics = refine_carrier_appearance(
        base,
        table,
        config=CarrierRepairConfig(appearance_weight_mode="uniform_view"),
    )
    amplitude_rgb = sh_to_rgb(amplitude.sh[:, 0])
    uniform_rgb = sh_to_rgb(uniform.sh[:, 0])
    assert amplitude_rgb[0, 0] > amplitude_rgb[0, 1]
    assert uniform_rgb[0, 1] > uniform_rgb[0, 0]
    assert amplitude_diagnostics["weight_mode"] == "amplitude"
    assert uniform_diagnostics["weight_mode"] == "uniform_view"


def test_clone_only_density_protects_parents_and_creates_local_low_opacity_children():
    count = 3
    params = {
        "means": torch.nn.Parameter(torch.zeros(count, 3)),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count)),
        "scales": torch.nn.Parameter(torch.full((count, 3), -1.5)),
        "opacities": torch.nn.Parameter(torch.zeros(count)),
        "sh0": torch.nn.Parameter(torch.zeros(count, 1, 3)),
        "shN": torch.nn.Parameter(torch.zeros(count, 0, 3)),
    }
    optimizers = {
        name: torch.optim.Adam([{"params": [parameter], "lr": 1e-3, "name": name}])
        for name, parameter in params.items()
    }
    controller = DensityController(
        DensityConfig(
            start_iter=1,
            every=1,
            grad_threshold=0.0,
            max_gaussians=6,
            prune_opacity=0.9,
            prune_scale_frac=0.01,
            clone_only=True,
            clone_jitter_fraction=0.2,
            clone_child_opacity_scale=0.25,
            protect_first_n=count,
            opacity_reset_every=0,
        ),
        count,
        scene_extent=1.0,
    )
    controller.grad_accum.fill_(1.0)
    controller.count.fill_(1.0)
    changed = controller.step(
        1,
        params,
        optimizers,
        generator=torch.Generator().manual_seed(17),
    )
    assert changed["means"].shape[0] == 6
    assert torch.equal(changed["means"][:count], params["means"])
    assert bool((changed["means"][count:] != params["means"]).any())
    assert torch.equal(changed["scales"][count:], params["scales"])
    assert torch.allclose(
        torch.sigmoid(changed["opacities"][count:]),
        torch.sigmoid(params["opacities"]) * 0.25,
    )
    assert controller.stats[-1] == {
        "iteration": 1,
        "n_before": 3,
        "n_after": 6,
        "cloned": 3,
        "split": 0,
        "pruned": 0,
    }


def test_clone_all_tangent_wave_clones_unobserved_rows_without_normal_displacement():
    count = 4
    params = {
        "means": torch.nn.Parameter(torch.zeros(count, 3)),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count)),
        # With identity quaternions the shortest x axis is the local surface normal.
        "scales": torch.nn.Parameter(
            torch.log(torch.tensor([[0.01, 0.20, 0.30]]).expand(count, -1).clone())
        ),
        "opacities": torch.nn.Parameter(torch.zeros(count)),
        "sh0": torch.nn.Parameter(torch.zeros(count, 1, 3)),
        "shN": torch.nn.Parameter(torch.zeros(count, 0, 3)),
    }
    optimizers = {
        name: torch.optim.Adam([{"params": [parameter], "lr": 1e-3, "name": name}])
        for name, parameter in params.items()
    }
    controller = DensityController(
        DensityConfig(
            start_iter=1,
            stop_iter=1,
            every=1,
            max_gaussians=2 * count,
            prune_opacity=0.0,
            prune_scale_frac=100.0,
            clone_only=True,
            clone_all=True,
            clone_jitter_fraction=0.1,
            clone_tangent_only=True,
            protect_first_n=count,
            opacity_reset_every=0,
        ),
        count,
        scene_extent=1.0,
    )

    changed = controller.step(
        1,
        params,
        optimizers,
        generator=torch.Generator().manual_seed(27027),
    )

    children = changed["means"][count:]
    assert changed["means"].shape[0] == 2 * count
    assert torch.equal(children[:, 0], torch.zeros(count))
    assert bool((children[:, 1:].abs() > 0).any())
    assert controller.stats[-1]["cloned"] == count


def test_compact_carrier_schedule_prunes_outside_center_and_freezes_means_afterward():
    beam, inputs = _fixture()
    outside = Gaussians3D.from_means_covs(
        means=torch.tensor([[10.0, 0.0, 0.0]]),
        covs=torch.eye(3)[None] * 0.01,
        colors=torch.full((1, 3), 0.5),
        opacity=torch.tensor([0.1]),
    )
    initialization = Gaussians3D.cat([beam.gaussians, outside])
    template = CompactTrainConfig(
        iterations=1,
        attempts_per_step=8,
        proposal_mode="area_gaussian",
        schedule_mode="balanced_cycle",
        target_mode="proposal_attempt",
        uniform_fraction=0.25,
        device="cpu",
        point_chunk=8,
        gaussian_chunk=8,
        outer_microbatch=4,
        query_component_chunk=8,
        evaluation_chunk=16,
        checkpoints=(0, 1),
        evaluate_checkpoint_risks=False,
    )
    config = CarrierOptimizationConfig(
        phase1_iterations=1,
        phase2_iterations=1,
        containment_gaussian_chunk=2,
        containment_component_chunk=2,
        template=template,
    )
    result = optimize_carriers(inputs, initialization, config)

    assert set(result.phase_histories) == {"phase1", "phase2"}
    assert set(result.phase_snapshots) == {
        "initial",
        "phase1",
        "contained",
        "phase2",
    }
    assert result.diagnostics["containment"]["removed"] == 1
    assert result.diagnostics["containment_recheck"]["removed"] == 0
    assert result.diagnostics["phase2_means_frozen_bit_exact"]
    assert result.diagnostics["topology_growth"] is False
    assert result.gaussians.n == 2
    assert result.gaussians.sh.shape[1] == 1
    assert result.phase_histories["phase1"]["seed"] == template.seed
    assert result.phase_histories["phase2"]["seed"] == template.seed + 1000
    assert result.phase_histories["phase2"]["optimizer_group_motion"]["means"]["max_abs"] == 0.0


def test_compact_carrier_policy_cannot_weaken_containment_thresholds():
    with pytest.raises(ValueError, match="containment_sigma=3.0"):
        CarrierOptimizationConfig(containment_sigma=4.0)
    with pytest.raises(ValueError, match="containment_near"):
        CarrierOptimizationConfig(containment_near=0.001)


def test_compact_carrier_schedule_requires_three_sigma_teacher_fields():
    beam, inputs = _fixture()
    bad_inputs = ReconstructionInputs(
        observations=[
            replace(inputs.observations[0], sigma_cutoff=4.0),
            *inputs.observations[1:],
        ],
        cameras=inputs.cameras,
        view_names=inputs.view_names,
        bounds_hint=inputs.bounds_hint,
        name=inputs.name,
    )
    with pytest.raises(ValueError, match="three-sigma"):
        optimize_carriers(
            bad_inputs,
            beam.gaussians,
            CarrierOptimizationConfig(phase1_iterations=1, phase2_iterations=1),
        )


def test_strict_carrier_center_prune_is_compact_and_reports_scope_limit():
    beam, inputs = _fixture()
    outside = beam.gaussians.detach()
    outside.means[1] = torch.tensor([20.0, 0.0, 0.0])

    pruned, receipt = prune_carrier_centers(
        inputs,
        outside,
        gaussian_chunk=1,
        component_chunk=1,
        device="cpu",
    )

    assert pruned.n == 1
    assert receipt["removed_rows"] == [1]
    assert receipt["projected_center_containment_after"]
    assert receipt["surface_occupancy_proven"] is False
    assert receipt["interior_visual_hull_floaters_detected"] is False


def test_compact_carrier_pipeline_runs_without_scene_data():
    _beam_fixture, inputs = _fixture(dilation=0.3)
    template = CompactTrainConfig(
        iterations=1,
        attempts_per_step=8,
        proposal_mode="area_gaussian",
        schedule_mode="balanced_cycle",
        target_mode="proposal_attempt",
        uniform_fraction=0.25,
        device="cpu",
        point_chunk=8,
        gaussian_chunk=8,
        outer_microbatch=4,
        query_component_chunk=8,
        evaluation_chunk=16,
        checkpoints=(0, 1),
        evaluate_checkpoint_risks=False,
    )
    config = CarrierPipelineConfig(
        beam=BeamFusionConfig(
            min_views=3,
            max_components=8,
            source_chunk=8,
            fold_in_chunk=8,
        ),
        repair=CarrierRepairConfig(
            covariance_steps=2,
            repair_opacity=False,
            repair_appearance=False,
        ),
        optimization=CarrierOptimizationConfig(
            phase1_iterations=1,
            phase2_iterations=1,
            containment_gaussian_chunk=8,
            containment_component_chunk=8,
            template=template,
        ),
    )

    result = run_carrier_pipeline(inputs, config)

    assert result.metrics["input_views"] == 3
    assert result.metrics["minimum_contributor_views"] >= 3
    assert result.metrics["final_n_gaussians"] <= 8
    assert result.optimization.diagnostics["source_rgb_used"] is False
    assert result.optimization.diagnostics["source_masks_used"] is False
    assert result.optimization.diagnostics["projected_center_containment"]
    assert result.optimization.diagnostics["topology_growth"] is False
