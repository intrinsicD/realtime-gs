"""Tests for exact-count paper-path initialization without image-backed dependencies."""

from __future__ import annotations

from dataclasses import replace

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.beam_fusion import BeamFusionConfig, BeamFusionResult
from rtgs.lift.paper_initializers import (
    PaperInitializerConfig,
    bounded_random_initialization,
    build_matched_paper_initializations,
    structural_initialization_inputs,
)
from rtgs.lift.splat_sfm import SplatSfMConfig, SplatSfMResult


def _inputs() -> ReconstructionInputs:
    cameras = [
        Camera.look_at(
            eye=torch.tensor([x, 0.0, -3.0]),
            target=torch.zeros(3),
            width=32,
            height=24,
            fov_x_deg=50.0,
        )
        for x in (-0.5, 0.5)
    ]
    fields = [
        GaussianObservationField(
            width=32,
            height=24,
            means=torch.tensor([[16.0, 12.0]]),
            log_scales=torch.zeros(1, 2),
            rotations=torch.zeros(1),
            colors=torch.full((1, 3), 0.5),
            amplitudes=torch.ones(1),
            view_id=f"v{index}",
            n_init=1,
        )
        for index in range(2)
    ]
    return ReconstructionInputs(
        observations=fields,
        cameras=cameras,
        view_names=["v0", "v1"],
        bounds_hint=(torch.tensor([1.0, 2.0, 3.0]), 4.0),
        name="paper-init-fixture",
    )


def _gaussians(count: int) -> Gaussians3D:
    return Gaussians3D(
        means=torch.arange(count * 3, dtype=torch.float32).reshape(count, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        log_scales=torch.zeros(count, 3),
        opacity=torch.full((count,), 0.1),
        sh=torch.zeros(count, 1, 3),
    )


def test_bounded_random_is_deterministic_inside_requested_sphere():
    inputs = _inputs()
    first = bounded_random_initialization(inputs, 100, seed=7, bounds_scale=0.25)
    second = bounded_random_initialization(inputs, 100, seed=7, bounds_scale=0.25)

    assert torch.equal(first.means, second.means)
    assert first.n == 100
    distance = torch.linalg.vector_norm(first.means - torch.tensor([1.0, 2.0, 3.0]), dim=1)
    assert float(distance.max()) <= 1.0
    assert torch.equal(first.opacity, torch.full((100,), 0.1))


def test_three_initializers_are_exact_count_matched_by_native_evidence(monkeypatch):
    sfm = SplatSfMResult(
        gaussians=_gaussians(4),
        track_offsets=torch.tensor([0, 2, 5, 7, 11]),
        member_view_indices=torch.zeros(11, dtype=torch.long),
        member_component_indices=torch.arange(11),
        track_reprojection_error=torch.tensor([0.1, 0.4, 0.2, 0.3]),
        track_triangulation_angle_deg=torch.tensor([10.0, 20.0, 30.0, 40.0]),
        track_covariance_residual=torch.tensor([0.2, 0.1, 0.3, 0.4]),
        unmatched_per_view=(0, 0),
        diagnostics={"kind": "sfm"},
    )
    beam = BeamFusionResult(
        gaussians=_gaussians(3),
        component_offsets=torch.tensor([0, 2, 5, 7]),
        contributor_view_indices=torch.zeros(7, dtype=torch.long),
        contributor_component_indices=torch.arange(7),
        contributor_depths=torch.ones(7),
        component_weights=torch.tensor([0.2, 0.9, 0.5]),
        unmatched_per_view=(0, 0),
        diagnostics={"kind": "beam"},
    )
    monkeypatch.setattr(
        "rtgs.lift.paper_initializers.structure_from_splats",
        lambda inputs, config: sfm,
    )
    monkeypatch.setattr(
        "rtgs.lift.paper_initializers.fuse_gaussian_beams",
        lambda inputs, config: beam,
    )

    result = build_matched_paper_initializations(
        _inputs(),
        PaperInitializerConfig(
            random_seed=9,
            max_starting_gaussians=2,
            sfm=SplatSfMConfig(init_opacity=0.1),
            beam=BeamFusionConfig(init_opacity=0.1),
        ),
    )

    assert result.count == 2
    assert result.bounded_random.n == result.splat_sfm.n == result.beam_fusion.n == 2
    assert result.splat_sfm_selected_rows.tolist() == [3, 1]
    assert result.beam_fusion_selected_rows.tolist() == [1, 2]
    assert result.receipt["source_counts"] == {"splat_sfm": 4, "beam_fusion": 3}
    assert result.receipt["selected_rows"]["splat_sfm"] == [3, 1]


def test_structural_input_selection_is_shared_bounded_and_deterministic():
    inputs = _inputs()
    field = inputs.observations[0]
    expanded = GaussianObservationField(
        width=field.width,
        height=field.height,
        means=torch.tensor(
            [
                [2.0, 2.0],
                [3.0, 3.0],
                [27.0, 2.0],
                [28.0, 3.0],
                [2.0, 20.0],
                [27.0, 20.0],
            ]
        ),
        log_scales=torch.zeros(6, 2),
        rotations=torch.zeros(6),
        colors=torch.full((6, 3), 0.5),
        amplitudes=torch.tensor([0.1, 0.9, 0.8, 0.2, 0.7, 0.6]),
        view_id="v0",
        n_init=6,
    )
    expanded_inputs = ReconstructionInputs(
        observations=[expanded, replace(expanded, view_id="v1")],
        cameras=inputs.cameras,
        view_names=inputs.view_names,
        bounds_hint=inputs.bounds_hint,
        name="expanded",
    )

    first, first_rows = structural_initialization_inputs(expanded_inputs, 4)
    second, second_rows = structural_initialization_inputs(expanded_inputs, 4)

    assert first.observations[0].n == first.observations[1].n == 4
    assert torch.equal(first_rows[0], first_rows[1])
    assert torch.equal(first_rows[0], second_rows[0])
    assert torch.equal(first.observations[0].means, second.observations[0].means)
