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


def _split_target(target: AnalyticGaussianField2D) -> AnalyticGaussianField2D:
    return AnalyticGaussianField2D(
        means=target.means.repeat_interleave(2, dim=0).flip(0),
        covariances=target.covariances.repeat_interleave(2, dim=0).flip(0),
        density_amplitudes=target.density_amplitudes.repeat_interleave(2, dim=0).flip(0) / 2,
        rgb_amplitudes=target.rgb_amplitudes.repeat_interleave(2, dim=0).flip(0) / 2,
    )


def test_combined_objective_and_all_prediction_gradients_ignore_exact_target_splits() -> None:
    from rtgs.lift.field_refit import _target_rgb_scale, _variable_field_objective

    _, _, fields, _ = _fixture()
    target = fields[0]
    split = _split_target(target)
    tensors = [
        (target.means + 0.3).detach().requires_grad_(),
        (target.covariances * 1.2).detach().requires_grad_(),
        (target.density_amplitudes * 0.7).detach().requires_grad_(),
        (target.rgb_amplitudes * 0.8).detach().requires_grad_(),
    ]
    predicted = AnalyticGaussianField2D(*tensors)
    outcomes = []
    for reference in (target, split):
        objective = _variable_field_objective(
            predicted,
            reference,
            density_weight=1.0,
            rgb_weight=0.25,
            include_rgb=True,
            chunk_size=2,
            target_rgb_scale=_target_rgb_scale(reference, chunk_size=2),
        )
        outcomes.append((objective, torch.autograd.grad(objective, tensors, retain_graph=True)))
    torch.testing.assert_close(outcomes[0][0], outcomes[1][0], atol=1e-12, rtol=1e-12)
    for original, divided in zip(outcomes[0][1], outcomes[1][1], strict=True):
        torch.testing.assert_close(original, divided, atol=1e-12, rtol=1e-12)
    # Preserve the old decomposition-dependent scale only for explicit replay.
    legacy = _target_rgb_scale(target, normalization="legacy_coefficients", chunk_size=2)
    legacy_split = _target_rgb_scale(split, normalization="legacy_coefficients", chunk_size=2)
    torch.testing.assert_close(legacy, legacy_split * 2)


def test_black_target_rgb_scale_keeps_objective_and_gradients_finite() -> None:
    from rtgs.lift.field_refit import _variable_field_objective

    _, _, fields, _ = _fixture()
    for dtype in (torch.float32, torch.float64):
        target = AnalyticGaussianField2D(
            means=fields[0].means.to(dtype),
            covariances=fields[0].covariances.to(dtype),
            density_amplitudes=fields[0].density_amplitudes.to(dtype),
            rgb_amplitudes=torch.zeros_like(fields[0].rgb_amplitudes).to(dtype),
        )
        rgb = fields[0].rgb_amplitudes.to(dtype).detach().requires_grad_()
        predicted = AnalyticGaussianField2D(
            target.means,
            target.covariances,
            target.density_amplitudes,
            rgb,
        )
        objective = _variable_field_objective(
            predicted,
            target,
            density_weight=1.0,
            rgb_weight=0.25,
            include_rgb=True,
            chunk_size=2,
        )
        assert torch.isfinite(objective)
        assert torch.isfinite(torch.autograd.grad(objective, rgb)[0]).all()


def test_rgb_refit_caches_target_energy_and_is_invariant_to_target_splits(monkeypatch) -> None:
    import rtgs.lift.field_refit as module

    gaussians, cameras, fields, colors = _fixture()
    original = module._target_rgb_scale
    calls = []

    def tracked(target, **kwargs):
        calls.append(target)
        return original(target, **kwargs)

    monkeypatch.setattr(module, "_target_rgb_scale", tracked)
    results = []
    for targets in (fields, tuple(_split_target(field) for field in fields)):
        results.append(
            fit_field_fibers(
                fiber=_fiber(gaussians, cameras, fields),
                reference_fields=targets,
                cameras=cameras,
                source_colors=colors,
                field_masses=gaussians.opacity,
                render_opacity=torch.full_like(gaussians.opacity, 0.1),
                config=FieldRefitConfig(iterations=3, appearance_start=0, chunk_size=2),
            )
        )
    assert len(calls) == 2 * len(cameras)
    torch.testing.assert_close(
        results[0].gaussians.means, results[1].gaussians.means, atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(
        results[0].gaussians.covariance(), results[1].gaussians.covariance(), atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(
        torch.tensor(results[0].objective_history), torch.tensor(results[1].objective_history)
    )


def test_soft_source_fit_starts_identically_and_reports_actual_drift() -> None:
    gaussians, cameras, fields, colors = _fixture()
    fiber = _fiber(gaussians, cameras, fields)
    initial = tuple(value.detach().clone() for value in fiber.means_covariances())
    fiber.set_source_relaxation(True)
    for before, after in zip(initial, fiber.means_covariances(), strict=True):
        torch.testing.assert_close(before, after, atol=1e-14, rtol=1e-14)
    result = fit_field_fibers(
        fiber=fiber,
        reference_fields=fields,
        cameras=cameras,
        source_colors=colors,
        field_masses=gaussians.opacity,
        render_opacity=torch.full_like(gaussians.opacity, 0.1),
        config=FieldRefitConfig(
            iterations=4,
            appearance_start=0,
            chunk_size=2,
            source_constraint="soft",
            source_anchor_weight=0.1,
        ),
    )
    assert result.accepted_steps > 0
    assert result.source_constraint == "soft"
    assert result.source_mean_max_error > 1e-6
    assert result.source_covariance_max_error > 1e-6
    assert torch.linalg.eigvalsh(fiber.means_covariances()[1]).min() > 0
    assert fiber.source_mean_offset.grad is not None
    assert fiber.source_shape_offset.grad is not None
    assert torch.isfinite(fiber.source_mean_offset.grad).all()
    assert torch.isfinite(fiber.source_shape_offset.grad).all()
    assert result.source_projection_max_error == max(
        result.source_mean_max_error, result.source_covariance_max_error
    )
    expected_means, expected_covariances = fiber.source_targets()
    actual_means, actual_covariances, _ = fiber.source_projection()
    torch.testing.assert_close(actual_means, expected_means, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        actual_covariances,
        expected_covariances + fiber.dilation * torch.eye(2, dtype=expected_covariances.dtype),
        atol=1e-10,
        rtol=1e-10,
    )


def test_soft_source_geometry_subset_and_anchor_penalty_preserve_state() -> None:
    import pytest

    gaussians, cameras, fields, _ = _fixture()
    fiber = _fiber(gaussians, cameras, fields)
    fiber.set_source_relaxation(True)
    with torch.no_grad():
        fiber.source_mean_offset.copy_(torch.tensor([[0.2, -0.1], [0.1, 0.3]]))
        fiber.source_shape_offset.copy_(torch.tensor([[0.1, -0.2, 0.3], [0.2, 0.3, -0.1]]))
    child = fiber.subset(torch.tensor([1]))
    assert child.relax_source
    for full, selected in zip(fiber.means_covariances(), child.means_covariances(), strict=True):
        torch.testing.assert_close(full[1:], selected)
    penalty = fiber.source_anchor_penalty()
    gradients = torch.autograd.grad(penalty, (fiber.source_mean_offset, fiber.source_shape_offset))
    for value, gradient in zip(
        (fiber.source_mean_offset, fiber.source_shape_offset), gradients, strict=True
    ):
        assert torch.all(value * gradient >= 0)
    with pytest.raises(ValueError, match="cannot restore hard"):
        fiber.set_source_relaxation(False)


def test_soft_source_projection_gradients_match_finite_differences() -> None:
    gaussians, cameras, fields, _ = _fixture()
    fiber = _fiber(gaussians, cameras, fields)
    fiber.set_source_relaxation(True)

    def loss():
        projected = fiber.project(cameras[2])
        return projected.means2d.square().mean() + projected.covariances2d.square().mean()

    parameters = (fiber.source_mean_offset, fiber.source_shape_offset)
    analytic = torch.autograd.grad(loss(), parameters)
    h = 1e-5
    for parameter, gradient in zip(parameters, analytic, strict=True):
        for coordinate in range(parameter.shape[1]):
            with torch.no_grad():
                parameter[0, coordinate] = h
                plus = loss().item()
                parameter[0, coordinate] = -h
                minus = loss().item()
                parameter[0, coordinate] = 0
            torch.testing.assert_close(
                gradient[0, coordinate],
                gradient.new_tensor((plus - minus) / (2 * h)),
                atol=1e-7,
                rtol=1e-6,
            )


def test_zero_iteration_refit_honors_explicit_legacy_normalization(monkeypatch) -> None:
    import rtgs.lift.field_refit as module

    gaussians, cameras, fields, colors = _fixture()
    modes = []
    original = module._target_rgb_scale

    def tracked(target, **kwargs):
        modes.append(kwargs.get("normalization", "field_energy"))
        return original(target, **kwargs)

    monkeypatch.setattr(module, "_target_rgb_scale", tracked)
    result = fit_field_fibers(
        fiber=_fiber(gaussians, cameras, fields),
        reference_fields=fields,
        cameras=cameras,
        source_colors=colors,
        field_masses=gaussians.opacity,
        render_opacity=gaussians.opacity,
        config=FieldRefitConfig(
            iterations=0, appearance_start=0, rgb_normalization="legacy_coefficients"
        ),
    )
    assert modes == ["legacy_coefficients"] * len(cameras)
    assert result.rgb_normalization == "legacy_coefficients"
