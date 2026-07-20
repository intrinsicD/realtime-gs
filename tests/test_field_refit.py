"""CPU tests for decomposition-free continuous fitting on exact source fibers."""

from __future__ import annotations

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.lift.field_loss import AnalyticGaussianField2D
from rtgs.lift.field_refit import FieldRefitConfig, fit_field_fibers
from rtgs.lift.inverse_projection_fiber import InverseProjectionFiber
from rtgs.render.projection import EWA_DILATION, project_gaussians_ewa


def _fixture(
    n_views: int = 3,
) -> tuple[
    Gaussians3D,
    tuple[Camera, ...],
    tuple[AnalyticGaussianField2D, ...],
    torch.Tensor,
]:
    dtype = torch.float64
    means = torch.tensor(
        [[-0.18, 0.05, 0.0], [0.24, -0.10, 0.08]],
        dtype=dtype,
    )
    covariances = torch.stack(
        [
            torch.diag(torch.tensor([0.010, 0.014, 0.008], dtype=dtype)),
            torch.diag(torch.tensor([0.012, 0.009, 0.016], dtype=dtype)),
        ]
    )
    colors = torch.tensor([[0.8, 0.2, 0.15], [0.1, 0.55, 0.85]], dtype=dtype)
    opacity = torch.tensor([0.65, 0.50], dtype=dtype)
    gaussians = Gaussians3D.from_means_covs(means, covariances, colors, opacity)
    cameras = (
        Camera.look_at(
            torch.tensor([1.5, 0.2, 1.6]),
            torch.zeros(3),
            width=48,
            height=48,
        ),
        Camera.look_at(
            torch.tensor([-1.4, 0.3, 1.7]),
            torch.zeros(3),
            width=48,
            height=48,
        ),
        Camera.look_at(
            torch.tensor([0.1, -1.6, 1.5]),
            torch.zeros(3),
            width=48,
            height=48,
        ),
    )[:n_views]
    fields = []
    for camera in cameras:
        projection = project_gaussians_ewa(gaussians, camera)
        fields.append(
            AnalyticGaussianField2D(
                means=projection.means2d,
                covariances=projection.covariances2d,
                density_amplitudes=opacity,
                rgb_amplitudes=opacity[:, None] * colors,
            )
        )
    return gaussians, cameras, tuple(fields), colors


def _fiber(
    gaussians: Gaussians3D,
    cameras: tuple[Camera, ...],
    fields: tuple[AnalyticGaussianField2D, ...],
) -> InverseProjectionFiber:
    count = gaussians.n
    source_views = torch.arange(count, dtype=torch.long) % len(cameras)
    source_components = torch.arange(count, dtype=torch.long)
    source_means = torch.stack([fields[int(source_views[row])].means[row] for row in range(count)])
    source_covariances = torch.stack(
        [fields[int(source_views[row])].covariances[row] for row in range(count)]
    )
    source_depths = torch.stack(
        [
            cameras[int(source_views[row])].project(gaussians.means[row : row + 1])[1][0]
            for row in range(count)
        ]
    )
    return InverseProjectionFiber(
        cameras=cameras,
        source_view_indices=source_views,
        source_component_indices=source_components,
        source_means2d=source_means,
        source_covariances2d=source_covariances,
        initial_depths=source_depths * 1.08,
        depth_lower=source_depths * 0.55,
        depth_upper=source_depths * 1.55,
        dilation=EWA_DILATION,
    )


def test_field_refit_preserves_source_projection_and_reduces_geometry_objective() -> None:
    gaussians, cameras, fields, colors = _fixture()
    fiber = _fiber(gaussians, cameras, fields)
    result = fit_field_fibers(
        fiber=fiber,
        reference_fields=fields,
        cameras=cameras,
        source_colors=colors,
        field_masses=gaussians.opacity,
        render_opacity=torch.full_like(gaussians.opacity, 0.15),
        config=FieldRefitConfig(
            iterations=8,
            appearance_start=8,
            learning_rate=0.015,
            visibility_refresh=4,
            chunk_size=8,
        ),
    )
    assert result.gaussians.n == gaussians.n
    assert result.objective_history[-1] <= result.objective_history[0] + 1e-10
    assert result.accepted_steps > 0
    assert result.source_projection_max_error < 1e-8
    assert result.source_color_max_error < 1e-10
    assert torch.equal(result.gaussians.opacity, torch.full_like(gaussians.opacity, 0.15))
    assert torch.equal(result.field_masses, gaussians.opacity)
    assert all(report.rank == 6 for report in result.observability)
    assert result.covariance_free_mask.tolist() == [True, True]


def test_two_view_refit_reports_the_pinned_covariance_null_mode() -> None:
    gaussians, cameras, fields, colors = _fixture(n_views=2)
    fiber = _fiber(gaussians, cameras, fields)
    result = fit_field_fibers(
        fiber=fiber,
        reference_fields=fields,
        cameras=cameras,
        source_colors=colors,
        field_masses=gaussians.opacity,
        render_opacity=torch.full_like(gaussians.opacity, 0.1),
        config=FieldRefitConfig(
            iterations=3,
            appearance_start=3,
            visibility_refresh=1,
            chunk_size=8,
        ),
    )
    assert [report.rank for report in result.observability] == [5, 5]
    assert [report.null_basis.shape for report in result.observability] == [
        (6, 1),
        (6, 1),
    ]
    assert result.covariance_free_mask.tolist() == [False, False]
    assert torch.equal(result.fiber.cross, torch.zeros_like(result.fiber.cross))
