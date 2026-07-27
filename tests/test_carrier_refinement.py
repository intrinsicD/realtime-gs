"""CPU contracts for ADR-002 carrier repair and maturation."""

from __future__ import annotations

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.data.synthetic import make_synthetic_scene
from rtgs.lift.beam_fusion import BeamFusionResult
from rtgs.lift.carrier_refinement import (
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
)
from rtgs.optim.density import DensityConfig, DensityController
from rtgs.optim.trainer import TrainConfig
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


def _fixture() -> tuple[BeamFusionResult, ReconstructionInputs]:
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
        projected = project_covariances_ewa(means, true_covariances, camera, dilation=0.0)
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


def test_tiny_carrier_schedule_records_phase_boundaries_and_lineage():
    scene = make_synthetic_scene(n_gaussians=4, n_cameras=3, image_size=10, seed=9)
    template = TrainConfig(
        rasterizer="torch",
        device="cpu",
        eval_every=1,
        ssim_lambda=0.0,
        use_masks=False,
        random_background=False,
        density=DensityConfig(
            grad_threshold=0.0,
            max_gaussians=32,
            opacity_reset_every=0,
        ),
    )
    config = CarrierOptimizationConfig(
        warmup_iterations=1,
        clone_iterations=1,
        higher_sh_iterations=1,
        standard_iterations=1,
        clone_every=1,
        standard_every=1,
        clone_grad_threshold=0.0,
        clone_growth_factor=2.0,
        standard_growth_factor=4.0,
        template=template,
    )
    result = optimize_carriers(scene, scene.gt_gaussians, config)
    assert set(result.phase_histories) == {"warmup", "clone", "higher-sh", "standard"}
    assert len(result.diagnostics["phase_boundaries"]) == 5
    assert result.diagnostics["phase_boundaries"][0]["carrier_survival_rate"] == 1.0
    assert len(result.diagnostics["final_root_ids"]) == result.gaussians.n
    assert result.gaussians.sh.shape[1] == 16
