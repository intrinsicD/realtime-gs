"""Tiny image-free integration tests for the field Stage-2 orchestrator."""

from __future__ import annotations

from dataclasses import replace

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.field_inputs import SceneFits
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.field_lifter import FieldLiftConfig, FieldLifter, _place
from rtgs.lift.field_refit import FieldRefitConfig
from rtgs.render.projection import EWA_DILATION, project_gaussians_ewa


def _observation(
    gaussians: Gaussians3D,
    camera: Camera,
    view_id: str,
    colors: torch.Tensor,
) -> GaussianObservationField:
    projection = project_gaussians_ewa(gaussians, camera)
    intrinsic = projection.covariances2d - EWA_DILATION * torch.eye(2)
    eigenvalues, eigenvectors = torch.linalg.eigh(intrinsic)
    axis = eigenvectors[:, :, 0]
    return GaussianObservationField(
        width=camera.width,
        height=camera.height,
        means=projection.means2d.float(),
        log_scales=0.5 * eigenvalues.float().log(),
        rotations=torch.atan2(axis[:, 1], axis[:, 0]).float(),
        colors=colors.float(),
        amplitudes=gaussians.opacity.float(),
        blend_mode="normalized",
        aa_dilation=EWA_DILATION,
        sigma_cutoff=3.0,
        support_fade_alpha=1.0,
        view_id=view_id,
        n_init=gaussians.n,
        provider="synthetic_fixture",
    )


def _fits() -> SceneFits:
    dtype = torch.float64
    means = torch.tensor(
        [[-0.30, 0.0, 0.0], [0.35, 0.05, 0.05]],
        dtype=dtype,
    )
    covariances = torch.stack(
        [
            torch.diag(torch.tensor([0.012, 0.010, 0.009], dtype=dtype)),
            torch.diag(torch.tensor([0.010, 0.014, 0.011], dtype=dtype)),
        ]
    )
    colors = torch.tensor([[0.8, 0.2, 0.1], [0.1, 0.5, 0.9]], dtype=dtype)
    opacity = torch.tensor([0.8, 0.7], dtype=dtype)
    gaussians = Gaussians3D.from_means_covs(means, covariances, colors, opacity)
    cameras = [
        Camera.look_at(torch.tensor([1.4, 0.2, 1.5]), torch.zeros(3), width=40, height=40),
        Camera.look_at(torch.tensor([-1.4, 0.3, 1.6]), torch.zeros(3), width=40, height=40),
        Camera.look_at(torch.tensor([0.1, -1.5, 1.5]), torch.zeros(3), width=40, height=40),
    ]
    names = ["v0", "v1", "v2"]
    inputs = ReconstructionInputs(
        observations=[
            _observation(gaussians, camera, name, colors)
            for camera, name in zip(cameras, names, strict=True)
        ],
        cameras=cameras,
        view_names=names,
        bounds_hint=(torch.zeros(3), 2.0),
        name="field-lifter-tiny",
    )
    return SceneFits.from_reconstruction_inputs(
        inputs,
        train_view_indices=(0, 1),
        heldout_view_indices=(2,),
    )


def _config() -> FieldLiftConfig:
    return FieldLiftConfig(
        max_tracks=2,
        max_train_views=2,
        depth_samples=12,
        candidate_multiplier=2,
        min_views=2,
        min_placement_score=0.0,
        background_fraction=0.0,
        refit=FieldRefitConfig(
            iterations=2,
            appearance_start=2,
            learning_rate=0.01,
            visibility_refresh=1,
            chunk_size=8,
        ),
    )


def test_field_lifter_runs_without_images_and_keeps_heldout_out_of_fitting() -> None:
    result = FieldLifter(_config()).fit(_fits())
    assert result.gaussians.n >= 2
    assert result.optimized_view_indices == (0, 1)
    assert result.heldout_view_indices == (2,)
    assert len(result.correspondences) == 3
    assert result.correspondences[2].shape == (result.gaussians.n, 2)
    assert result.correspondence_visibility.shape == (3, result.gaussians.n)
    for track, source_view in enumerate(result.placement.source_global_view_indices):
        assert result.correspondence_visibility[int(source_view), track] == 1
    assert result.topology_receipts
    assert any(
        receipt.proposal.tag == "projected-runnalls-bound" for receipt in result.topology_receipts
    )
    assert all(
        receipt.penalized_candidate < receipt.penalized_before
        for receipt in result.topology_receipts
        if receipt.accepted
    )
    assert result.diagnostics["heldout_views"] == [2]
    assert result.diagnostics["bounds_source"] == "frustum_consensus"
    assert result.diagnostics["unverified_geometry_ignored"] is True
    assert result.diagnostics["source_projection_max_error"] < 2e-4
    assert result.diagnostics["analytic_semantics"].startswith("untruncated")
    assert result.semantic_validation.train.n_views == 2
    assert result.semantic_validation.heldout is not None
    assert result.semantic_validation.heldout.n_views == 1
    assert torch.isfinite(torch.tensor(result.semantic_validation.heldout.density_mse))


def test_field_lifter_is_deterministic_for_the_same_compact_fields() -> None:
    first = FieldLifter(_config()).fit(_fits())
    second = FieldLifter(_config()).fit(_fits())
    assert torch.equal(first.gaussians.means, second.gaussians.means)
    assert torch.equal(first.gaussians.log_scales, second.gaussians.log_scales)
    assert torch.equal(first.gaussians.opacity, second.gaussians.opacity)
    assert first.refit.objective_history == second.refit.objective_history
    assert torch.equal(first.correspondence_visibility, second.correspondence_visibility)
    for left, right in zip(first.correspondences, second.correspondences, strict=True):
        assert torch.equal(left, right)


def test_maskless_placement_uses_frustum_consensus_and_far_shell() -> None:
    fits = replace(_fits(), bounds_hint=None)
    config = replace(
        _config(),
        max_tracks=4,
        background_fraction=0.5,
        topology_rounds=0,
    )
    placement = _place(fits, (0, 1), config)
    assert placement.diagnostics["bounds_source"] == "frustum_consensus"
    assert placement.fiber.n == 4
    assert int(placement.background_mask.sum()) == 2
    assert torch.all(
        placement.fiber.depths()[placement.background_mask]
        > placement.fiber.depths()[~placement.background_mask].amin()
    )


def test_lossless_alpha_filters_sources_and_depth_priors_seed_selected_rays() -> None:
    base = _fits()
    alphas = []
    depth_priors = []
    confidences = []
    for observation, camera in zip(base.observations, base.cameras, strict=True):
        alpha = torch.ones((camera.height, camera.width), dtype=torch.bool)
        first_xy = observation.native_means(0)
        alpha[int(first_xy[1].floor()), int(first_xy[0].floor())] = False
        alphas.append(alpha)
        center_depth = camera.project(torch.zeros((1, 3)))[1][0]
        depth_priors.append(center_depth.expand(observation.n).clone())
        confidences.append(torch.ones(observation.n, dtype=observation.dtype))
    fits = replace(
        base,
        alphas=tuple(alphas),
        depth_priors=tuple(depth_priors),
        depth_confidences=tuple(confidences),
    )
    config = replace(
        _config(),
        max_tracks=4,
        background_fraction=0.0,
        topology_rounds=0,
    )
    placement = _place(fits, (0, 1), config)
    assert placement.diagnostics["alpha_rejected_sources"] == 2
    assert placement.fiber.n == 2
    for row in range(placement.fiber.n):
        global_view = int(placement.source_global_view_indices[row])
        component = int(placement.fiber.source_component_indices[row])
        assert component == 1
        depth = placement.fiber.depths()[row]
        expected = depth_priors[global_view][component].to(depth)
        assert torch.allclose(depth, expected, atol=2e-6, rtol=0.0)


def test_trusted_sparse_points_anchor_source_ray_depths() -> None:
    base = _fits()
    points = torch.tensor(
        [[-0.30, 0.0, 0.0], [0.35, 0.05, 0.05]],
        dtype=base.observations[0].dtype,
    )
    visibility = tuple(torch.tensor([0, 1]) for _ in range(base.n_views))
    fits = replace(
        base,
        points=points,
        point_visibility=visibility,
        geometry_is_train_only=True,
    )
    config = replace(
        _config(),
        max_tracks=4,
        background_fraction=0.0,
        topology_rounds=0,
    )
    placement = _place(fits, (0, 1), config)
    assert placement.diagnostics["bounds_source"] == "explicit_train_only_hint"
    assert placement.diagnostics["sparse_depth_anchor_tracks"] == placement.fiber.n
    for row in range(placement.fiber.n):
        local_view = int(placement.fiber.source_view_indices[row])
        component = int(placement.fiber.source_component_indices[row])
        expected = base.cameras[local_view].project(points[component : component + 1])[1][0]
        torch.testing.assert_close(
            placement.fiber.depths()[row],
            expected.to(placement.fiber.depths()),
            rtol=0.0,
            atol=2e-5,
        )
