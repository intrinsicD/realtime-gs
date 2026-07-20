"""CPU tests for the correspondence-free local 4-dof depth refine prototype.

These pin the *mechanism* (the smooth consensus objective is optimized, deterministically, into a
clean grad-free, mergeable result) — not a geometry claim. Correspondence-free consensus does not
guarantee a geometric improvement; that limitation is the point of the prototype and is recorded in
docs/EXPERIMENTS.md.
"""

from __future__ import annotations

import math

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.compact_carve import CompactCarveConfig, CompactCarveInitializer
from rtgs.lift.compact_refine import (
    LocalDepthRefineConfig,
    build_fiber_from_initialization,
    refine_initialization_depths,
)
from rtgs.lift.merge import merge_by_voxel

_TARGETS = torch.tensor(
    [
        [-0.20, -0.16, 0.12],
        [0.20, -0.16, -0.12],
        [-0.18, 0.18, 0.06],
        [0.20, 0.16, -0.10],
        [0.00, 0.00, 0.16],
    ]
)
_COLORS = torch.tensor(
    [
        [0.85, 0.15, 0.15],
        [0.15, 0.80, 0.20],
        [0.15, 0.25, 0.90],
        [0.80, 0.75, 0.15],
        [0.60, 0.20, 0.70],
    ]
)


def _camera(x: float, y: float) -> Camera:
    return Camera.look_at(
        eye=torch.tensor([x, y, -3.0]),
        target=torch.zeros(3),
        width=64,
        height=64,
        fov_x_deg=55.0,
    )


def _field(camera: Camera, view_id: str) -> GaussianObservationField:
    means, depth = camera.project(_TARGETS)
    assert bool((depth > 0).all())
    return GaussianObservationField(
        width=64,
        height=64,
        means=means,
        log_scales=torch.log(torch.full((len(_TARGETS), 2), 1.8)),
        rotations=torch.tensor([0.0, 0.2, -0.15, 0.35, 0.1]),
        colors=_COLORS,
        amplitudes=torch.tensor([0.9, 0.8, 0.7, 1.0, 0.85]),
        view_id=view_id,
        n_init=len(_TARGETS),
    )


def _inputs() -> ReconstructionInputs:
    poses = [(-0.9, 0.1), (-0.4, -0.1), (0.0, 0.15), (0.5, -0.05), (0.9, 0.1)]
    cameras = [_camera(x, y) for x, y in poses]
    names = [f"v{i}" for i in range(len(cameras))]
    return ReconstructionInputs(
        observations=[_field(c, n) for c, n in zip(cameras, names, strict=True)],
        cameras=cameras,
        view_names=names,
        bounds_hint=(torch.zeros(3), 1.2),
        name="refine-fixture",
    )


def _dense_result(inputs, samples_per_ray=16):
    config = CompactCarveConfig(
        n_init_3d=1,
        select_all_eligible=True,
        candidate_multiplier=4,
        anchor_mode="component_centers",
        samples_per_ray=samples_per_ray,
        query_batch_size=256,
        seed=17,
        min_views=2,
        hull_fraction=0.6,
        coverage_scale=2.0,
        coverage_threshold=0.2,
        color_std_sigma=0.3,
        min_score=0.005,
        init_opacity=0.5,
    )
    return CompactCarveInitializer(config).initialize(inputs)


def test_build_fiber_matches_initialization_depths_and_lineage():
    inputs = _inputs()
    result = _dense_result(inputs)
    fiber = build_fiber_from_initialization(inputs, result, depth_window=0.5)
    assert fiber.n == result.gaussians.n
    assert torch.equal(fiber.source_view_indices, result.lineage.source_view_indices)
    # The initial fiber reproduces the carve depths within the sigmoid parameterization.
    torch.testing.assert_close(
        fiber.depths(), result.depths.to(fiber.depths().dtype), atol=1e-5, rtol=1e-5
    )


def test_refine_increases_consensus_objective_and_is_deterministic():
    inputs = _inputs()
    result = _dense_result(inputs)
    config = LocalDepthRefineConfig(iterations=50, learning_rate=0.05, depth_window=0.5)

    refined = refine_initialization_depths(inputs, result, config)

    # The optimizer maximizes the consensus objective, so it must not decrease it.
    assert refined.refined_objective >= refined.initial_objective - 1e-6
    assert refined.diagnostics["mean_absolute_depth_change"] > 0.0  # depths actually moved

    # Output is a clean, finite, grad-free Gaussians3D that downstream merge accepts.
    assert not refined.gaussians.means.requires_grad
    assert bool(torch.isfinite(refined.gaussians.means).all())
    merged, group = merge_by_voxel(refined.gaussians, 0.06, return_group=True)
    assert merged.n <= refined.gaussians.n
    assert group.shape == (refined.gaussians.n,)

    # Determinism for fixed inputs/config.
    repeat = refine_initialization_depths(inputs, result, config)
    assert torch.equal(refined.refined_depths, repeat.refined_depths)
    assert torch.equal(refined.gaussians.means, repeat.gaussians.means)


def test_refine_config_validates_controls():
    with pytest.raises(TypeError, match="iterations must be an integer"):
        LocalDepthRefineConfig(iterations=2.0)
    with pytest.raises(ValueError, match="depth_window must be in"):
        LocalDepthRefineConfig(depth_window=1.0)
    with pytest.raises(ValueError, match="learning_rate must be finite and positive"):
        LocalDepthRefineConfig(learning_rate=0.0)


def test_refine_rejects_wrong_backend_count():
    inputs = _inputs()
    result = _dense_result(inputs)
    with pytest.raises(ValueError, match="one query backend per view"):
        refine_initialization_depths(inputs, result, backends=[])


def test_refine_ray_scale_option_optimizes_covariance_null_scale():
    inputs = _inputs()
    result = _dense_result(inputs)
    config = LocalDepthRefineConfig(iterations=30, learning_rate=0.05, refine_ray_scale=True)
    refined = refine_initialization_depths(inputs, result, config)
    assert refined.diagnostics["refine_ray_scale"] is True
    assert math.isfinite(refined.refined_objective)
    assert not refined.gaussians.log_scales.requires_grad
