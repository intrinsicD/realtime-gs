"""CPU checks for covariance observability and prior-aware linear recovery."""

from __future__ import annotations

import math

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.lift.field_observability import (
    analyze_covariance_observability,
    covariance_to_vector,
    solve_projected_covariance,
    triangulate_projected_mean,
    vector_to_covariance,
)
from rtgs.render.projection import project_covariances_ewa


def _camera(translation: tuple[float, float, float]) -> Camera:
    return Camera(
        fx=71.0,
        fy=67.0,
        cx=40.0,
        cy=36.0,
        width=80,
        height=72,
        R=torch.eye(3),
        t=torch.tensor(translation),
    )


def _cameras() -> tuple[Camera, Camera, Camera]:
    return (
        _camera((0.0, 0.0, 0.0)),
        _camera((-1.0, 0.0, 0.0)),
        _camera((0.0, -1.0, 0.0)),
    )


def _project(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    cameras: tuple[Camera, ...],
    dilation: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        [
            project_covariances_ewa(
                mean[None],
                covariance[None],
                camera,
                dilation=float(view_dilation),
            ).covariances2d[0]
            for camera, view_dilation in zip(cameras, dilation, strict=True)
        ]
    )


def test_covariance_vector_round_trip() -> None:
    covariance = torch.tensor(
        [[1.7, 0.2, -0.1], [0.2, 0.9, 0.3], [-0.1, 0.3, 2.1]],
        dtype=torch.float64,
    )
    assert torch.equal(vector_to_covariance(covariance_to_vector(covariance)), covariance)


def test_weighted_calibrated_mean_triangulation_recovers_world_point() -> None:
    mean = torch.tensor([0.2, -0.15, 2.3], dtype=torch.float64)
    cameras = _cameras()
    projected = torch.stack([camera.project(mean[None])[0][0] for camera in cameras])
    result = triangulate_projected_mean(
        projected,
        cameras,
        view_weights=torch.tensor([1.0, 0.25, 2.0], dtype=torch.float64),
        rcond=1e-12,
    )

    assert result.report.full_rank
    assert result.report.rank == 3
    assert result.report.null_basis.shape == (3, 0)
    assert math.isfinite(result.report.condition_number)
    assert torch.allclose(result.mean, mean, atol=1e-12, rtol=1e-12)
    assert torch.allclose(result.projected_means, projected, atol=1e-12, rtol=1e-12)
    assert float(result.reprojection_errors.max()) <= 1e-12
    assert float(result.weighted_residual_norm) <= 1e-12
    assert bool((result.depths > 0).all())


def test_mean_triangulation_rejects_unobservable_repeated_camera() -> None:
    mean = torch.tensor([0.2, -0.15, 2.3], dtype=torch.float64)
    camera = _cameras()[0]
    projected = camera.project(mean[None])[0].expand(2, -1).clone()

    with pytest.raises(ValueError, match="rank deficient"):
        triangulate_projected_mean(projected, (camera, camera), rcond=1e-12)


def test_two_views_report_rank_five_and_generic_triple_reports_rank_six() -> None:
    mean = torch.tensor([0.2, -0.15, 2.3], dtype=torch.float64)
    cameras = _cameras()
    pair = analyze_covariance_observability(mean, cameras[:2], rcond=1e-10)
    triple = analyze_covariance_observability(mean, cameras, rcond=1e-10)

    assert pair.rank == 5
    assert pair.null_basis.shape == (6, 1)
    assert math.isinf(pair.condition_number)
    assert math.isfinite(pair.observable_condition_number)
    assert pair.singular_values.shape == (6,)

    assert triple.rank == 6
    assert triple.null_basis.shape == (6, 0)
    assert math.isfinite(triple.condition_number)
    assert triple.condition_number == triple.observable_condition_number
    assert triple.singular_values.shape == (6,)


def test_weighted_triple_solve_removes_dilation_and_recovers_spd_covariance() -> None:
    mean = torch.tensor([0.2, -0.15, 2.3], dtype=torch.float64)
    covariance = torch.tensor(
        [[0.18, 0.025, -0.012], [0.025, 0.11, 0.018], [-0.012, 0.018, 0.24]],
        dtype=torch.float64,
    )
    cameras = _cameras()
    dilation = torch.tensor([0.3, 0.15, 0.45], dtype=torch.float64)
    projected = _project(mean, covariance, cameras, dilation)
    result = solve_projected_covariance(
        mean,
        cameras,
        projected,
        view_weights=torch.tensor([1.0, 0.4, 2.0], dtype=torch.float64),
        dilation=dilation,
        minimum_eigenvalue=1e-10,
        rcond=1e-10,
    )

    assert result.report.rank == 6
    assert result.used_null_prior is False
    assert torch.allclose(result.raw_covariance, covariance, atol=2e-12, rtol=2e-12)
    assert torch.allclose(result.covariance, covariance, atol=2e-12, rtol=2e-12)
    assert torch.allclose(
        result.projected_covariances,
        projected,
        atol=2e-10,
        rtol=2e-12,
    )
    assert float(result.weighted_residual_norm) <= 2e-10
    assert bool((torch.linalg.eigvalsh(result.covariance) > 0).all())


def test_two_view_solve_requires_prior_and_uses_only_its_null_coordinate() -> None:
    mean = torch.tensor([0.2, -0.15, 2.3], dtype=torch.float64)
    covariance = torch.tensor(
        [[0.18, 0.025, -0.012], [0.025, 0.11, 0.018], [-0.012, 0.018, 0.24]],
        dtype=torch.float64,
    )
    cameras = _cameras()[:2]
    dilation = torch.tensor([0.3, 0.15], dtype=torch.float64)
    projected = _project(mean, covariance, cameras, dilation)
    with pytest.raises(ValueError, match="rank deficient"):
        solve_projected_covariance(
            mean,
            cameras,
            projected,
            dilation=dilation,
            rcond=1e-10,
        )

    result = solve_projected_covariance(
        mean,
        cameras,
        projected,
        dilation=dilation,
        prior_covariance=covariance,
        rcond=1e-10,
    )
    assert result.report.rank == 5
    assert result.used_null_prior is True
    assert torch.allclose(result.covariance, covariance, atol=2e-12, rtol=2e-12)
    assert torch.allclose(result.projected_covariances, projected, atol=2e-10, rtol=2e-12)


def test_inconsistent_linear_solution_is_projected_to_psd() -> None:
    mean = torch.tensor([0.2, -0.15, 2.3], dtype=torch.float64)
    cameras = _cameras()
    indefinite = torch.diag(torch.tensor([-0.2, 0.6, 0.9], dtype=torch.float64))
    dilation = torch.full((3,), 1000.0, dtype=torch.float64)
    projected = _project(mean, indefinite, cameras, dilation)
    assert bool((torch.linalg.eigvalsh(projected) > 0).all())

    result = solve_projected_covariance(
        mean,
        cameras,
        projected,
        dilation=dilation,
        minimum_eigenvalue=0.0,
        rcond=1e-10,
    )
    assert float(torch.linalg.eigvalsh(result.raw_covariance).min()) < -0.1
    assert float(torch.linalg.eigvalsh(result.covariance).min()) >= -1e-14
    assert torch.equal(result.covariance, result.covariance.T)
