from __future__ import annotations

import math

import numpy as np
import torch

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.compact_views import PackedAlpha
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.beam_fusion import BeamFusionConfig, fuse_gaussian_beams
from rtgs.lift.beam_partition import (
    MaskedPartitionConfig,
    partition_masked_gaussian_density,
    refit_beam_covariances_from_masked_partitions,
)
from rtgs.render.projection import project_covariances_ewa


def _packed_alpha(mask: torch.Tensor, origin: tuple[int, int] = (0, 0)) -> PackedAlpha:
    packed = np.packbits(mask.cpu().numpy().reshape(-1), bitorder="little")
    return PackedAlpha(
        payload=packed.tobytes(),
        shape=(int(mask.shape[0]), int(mask.shape[1])),
        origin=origin,
        foreground_count=int(mask.sum()),
    )


def _covariances_to_rs(covariances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(covariances)
    rotations = torch.atan2(eigenvectors[:, 1, 0], eigenvectors[:, 0, 0])
    return eigenvalues.sqrt().log(), rotations


def _field(
    means: torch.Tensor,
    covariances: torch.Tensor,
    *,
    width: int,
    height: int,
    amplitudes: torch.Tensor | None = None,
    view_id: str = "test",
) -> GaussianObservationField:
    log_scales, rotations = _covariances_to_rs(covariances)
    count = means.shape[0]
    if amplitudes is None:
        amplitudes = torch.ones(count, dtype=torch.float64)
    return GaussianObservationField(
        width=width,
        height=height,
        means=means.to(torch.float64),
        log_scales=log_scales.to(torch.float64),
        rotations=rotations.to(torch.float64),
        colors=torch.full((count, 3), 0.5, dtype=torch.float64),
        amplitudes=amplitudes.to(torch.float64),
        view_id=view_id,
        provider="synthetic_fixture",
    )


def test_single_anchor_full_mask_recovers_native_gaussian_moment_and_mass():
    covariance = torch.tensor([[[4.0, 0.6], [0.6, 1.5]]], dtype=torch.float64)
    field = _field(
        torch.tensor([[32.5, 32.5]], dtype=torch.float64),
        covariance,
        width=64,
        height=64,
        amplitudes=torch.tensor([0.7], dtype=torch.float64),
    )
    alpha = _packed_alpha(torch.ones(64, 64, dtype=torch.bool))

    partition = partition_masked_gaussian_density(
        field,
        alpha,
        torch.tensor([0]),
        MaskedPartitionConfig(quadrature_order=5),
    )

    expected_mass = 0.7 * 2.0 * math.pi * math.sqrt(float(torch.linalg.det(covariance[0])))
    assert torch.allclose(partition.covariances2d, covariance, atol=1e-12, rtol=1e-12)
    assert math.isclose(float(partition.masses[0]), expected_mass, rel_tol=1e-12)
    assert partition.diagnostics["partition_of_unity_relative_error"] < 1e-15


def test_masked_partition_excludes_source_density_wholly_outside_foreground():
    covariance = torch.eye(2, dtype=torch.float64).expand(3, 2, 2).clone() * 0.01
    field = _field(
        torch.tensor([[4.5, 8.5], [12.5, 8.5], [27.5, 8.5]], dtype=torch.float64),
        covariance,
        width=32,
        height=16,
    )
    mask = torch.ones(16, 32, dtype=torch.bool)
    mask[:, 11:15] = False
    partition = partition_masked_gaussian_density(
        field,
        _packed_alpha(mask),
        torch.tensor([0, 2]),
    )

    one_component_mass = 2.0 * math.pi * 0.01
    assert math.isclose(float(partition.masses.sum()), 2.0 * one_component_mass, rel_tol=1e-12)
    assert partition.diagnostics["positive_masked_samples"] == 50
    assert torch.allclose(
        partition.covariances2d,
        covariance[[0, 2]],
        atol=1e-12,
        rtol=1e-12,
    )


def test_fixed_anchor_partition_includes_between_component_second_moment():
    covariance = torch.eye(2, dtype=torch.float64).expand(3, 2, 2).clone()
    field = _field(
        torch.tensor([[10.5, 16.5], [12.5, 16.5], [50.5, 16.5]], dtype=torch.float64),
        covariance,
        width=64,
        height=32,
    )
    partition = partition_masked_gaussian_density(
        field,
        _packed_alpha(torch.ones(32, 64, dtype=torch.bool)),
        torch.tensor([0, 2]),
    )

    expected_left = torch.tensor([[3.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    assert torch.allclose(partition.covariances2d[0], expected_left, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        partition.covariances2d[1],
        torch.eye(2, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_end_to_end_partition_refit_uses_exact_lineage_and_freezes_other_fields():
    cameras = [
        Camera.look_at(torch.tensor([-0.8, 0.0, -3.0]), torch.zeros(3), width=96, height=96),
        Camera.look_at(torch.tensor([0.0, 0.6, -3.0]), torch.zeros(3), width=96, height=96),
        Camera.look_at(torch.tensor([0.8, 0.0, -3.0]), torch.zeros(3), width=96, height=96),
    ]
    mean = torch.zeros(1, 3, dtype=torch.float64)
    covariance3d = torch.diag(torch.tensor([4e-4, 6e-4, 5e-4], dtype=torch.float64))[None]
    fields = []
    for view_index, camera in enumerate(cameras):
        projection = project_covariances_ewa(mean, covariance3d, camera, dilation=0.0)
        field = _field(
            projection.means2d,
            projection.covariances2d,
            width=96,
            height=96,
            amplitudes=torch.tensor([0.8], dtype=torch.float64),
            view_id=f"v{view_index}",
        )
        fields.append(field)
    inputs = ReconstructionInputs(
        observations=fields,
        cameras=cameras,
        view_names=[f"v{index}" for index in range(3)],
        bounds_hint=(torch.zeros(3), 1.2),
        name="partition-refit",
    )
    beam_config = BeamFusionConfig(min_views=3)
    beam = fuse_gaussian_beams(inputs, beam_config)
    alphas = [_packed_alpha(torch.ones(96, 96, dtype=torch.bool)) for _ in cameras]

    fitted = refit_beam_covariances_from_masked_partitions(
        inputs,
        alphas,
        beam,
        beam_config,
    )

    assert beam.contributor_depths.shape == beam.contributor_view_indices.shape == (3,)
    assert fitted.diagnostics["uses_3d_projection_for_anchor_discovery"] is False
    assert fitted.diagnostics["n_unique_view_component_anchors"] == 3
    assert fitted.diagnostics["native_ci_roundtrip_relative_error"]["max"] < 1e-5
    for treatment in (fitted.area_matched, fitted.full_moment):
        assert torch.equal(treatment.means, beam.gaussians.means)
        assert torch.equal(treatment.opacity, beam.gaussians.opacity)
        assert torch.equal(treatment.sh, beam.gaussians.sh)
        assert torch.allclose(
            treatment.covariance(),
            beam.gaussians.covariance(),
            atol=2e-7,
            rtol=2e-5,
        )
