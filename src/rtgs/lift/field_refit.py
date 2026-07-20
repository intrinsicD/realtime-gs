"""Continuous field-level refitting on exact source-projection fibers.

The optimizer never assigns a reference component to a track.  It compares the projected
mixture against each reference field through exact additive density/RGB-numerator inner
products.  For normalized StructSplat observations this is the explicitly documented analytic
proxy; callers can separately evaluate the frozen teacher's sampled renderer semantics.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.sh import rgb_to_sh
from rtgs.lift.field_loss import AnalyticGaussianField2D, mixture_inner_product
from rtgs.lift.field_observability import (
    CovarianceObservabilityReport,
    analyze_covariance_observability,
    covariance_to_vector,
)
from rtgs.lift.field_visibility import (
    CenterVisibility,
    center_transmittance_visibility,
    solve_view_gains,
)
from rtgs.lift.inverse_projection_fiber import InverseProjectionFiber
from rtgs.lift.source_anchored_sh import SourceAnchoredSH


@dataclass(frozen=True)
class FieldRefitConfig:
    """Bounded deterministic controls for one continuous fiber-refit stage."""

    iterations: int = 40
    learning_rate: float = 0.025
    appearance_start: int = 20
    sh_degree: int = 1
    density_weight: float = 1.0
    rgb_weight: float = 0.25
    null_prior_weight: float = 0.05
    observability_condition_limit: float = 1e5
    visibility_refresh: int = 5
    gain_ridge: float = 1e-6
    chunk_size: int = 256
    gradient_clip: float = 10.0
    force_source_visible: bool = True

    def __post_init__(self) -> None:
        integer_fields = (
            "iterations",
            "appearance_start",
            "sh_degree",
            "visibility_refresh",
            "chunk_size",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        if not 0 <= self.appearance_start <= self.iterations:
            raise ValueError("appearance_start must be in [0,iterations]")
        if not 1 <= self.sh_degree <= 3:
            raise ValueError("field refit uses source-anchored SH degree in [1,3]")
        if self.visibility_refresh <= 0 or self.chunk_size <= 0:
            raise ValueError("visibility_refresh and chunk_size must be positive")
        positive = (
            "learning_rate",
            "density_weight",
            "observability_condition_limit",
            "gain_ridge",
            "gradient_clip",
        )
        for name in positive:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("rgb_weight", "null_prior_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class FieldRefitResult:
    """Optimized fiber state and diagnostics with mass/opacity kept separate."""

    gaussians: Gaussians3D
    fiber: InverseProjectionFiber
    appearance: SourceAnchoredSH
    field_masses: torch.Tensor
    render_opacity: torch.Tensor
    visibility: CenterVisibility
    gains: torch.Tensor
    objective_history: tuple[float, ...]
    accepted_steps: int
    observability: tuple[CovarianceObservabilityReport, ...]
    covariance_free_mask: torch.Tensor
    source_projection_max_error: float
    source_color_max_error: float


def _constant_initialized_appearance(
    *,
    degree: int,
    source_directions: torch.Tensor,
    source_colors: torch.Tensor,
) -> SourceAnchoredSH:
    model = SourceAnchoredSH(
        degree=degree,
        source_directions=source_directions,
        source_colors=source_colors,
    )
    desired = torch.zeros_like(model.free)
    desired[:, 0] = rgb_to_sh(source_colors)
    with torch.no_grad():
        # A constant SH function satisfies the source equality.  Feeding its difference from
        # the particular solution through SourceAnchoredSH's null projection reconstructs it.
        model.free.copy_(desired - model.particular)
    tolerance = 1e-10 if source_colors.dtype == torch.float64 else 3e-6
    if float(model.source_max_abs_error().detach()) > tolerance:
        raise RuntimeError("constant source-anchored SH initialization lost its equality")
    return model


def _view_directions(means: torch.Tensor, cameras: Sequence[Camera]) -> torch.Tensor:
    return torch.stack(
        [
            torch.nn.functional.normalize(
                means - camera.position.to(means),
                dim=-1,
            )
            for camera in cameras
        ]
    )


def _source_directions(
    means: torch.Tensor,
    cameras: Sequence[Camera],
    source_view_indices: torch.Tensor,
) -> torch.Tensor:
    result = torch.empty_like(means)
    for view_index, camera in enumerate(cameras):
        rows = (source_view_indices == view_index).nonzero(as_tuple=True)[0]
        if rows.numel():
            result[rows] = torch.nn.functional.normalize(
                means[rows] - camera.position.to(means),
                dim=-1,
            )
    return result


def _materialize(
    fiber: InverseProjectionFiber,
    appearance: SourceAnchoredSH,
    render_opacity: torch.Tensor,
) -> Gaussians3D:
    means, covariances = fiber.means_covariances()
    base = Gaussians3D.from_means_covs(
        means,
        covariances,
        appearance.source_colors.clamp(0.0, 1.0),
        render_opacity,
        sh_degree=appearance.degree,
    )
    return Gaussians3D(
        means=base.means,
        quats=base.quats,
        log_scales=base.log_scales,
        opacity=render_opacity,
        sh=appearance.coefficients(),
    )


def _predicted_field(
    *,
    fiber: InverseProjectionFiber,
    appearance: SourceAnchoredSH,
    camera: Camera,
    visibility: torch.Tensor,
    gain: torch.Tensor,
    field_masses: torch.Tensor,
) -> AnalyticGaussianField2D:
    projection = fiber.project(camera)
    directions = torch.nn.functional.normalize(
        fiber.means_covariances()[0] - camera.position.to(field_masses),
        dim=-1,
    )
    color = appearance.preactivation(directions)
    density = field_masses * visibility * gain
    return AnalyticGaussianField2D(
        means=projection.means2d,
        covariances=projection.covariances2d,
        density_amplitudes=density,
        rgb_amplitudes=density[:, None] * color,
    )


def _variable_field_objective(
    predicted: AnalyticGaussianField2D,
    target: AnalyticGaussianField2D,
    *,
    density_weight: float,
    rgb_weight: float,
    include_rgb: bool,
    chunk_size: int,
) -> torch.Tensor:
    """Return the prediction-dependent part of exact L2.

    The omitted ``<target,target>`` term is constant for continuous steps and every topology
    proposal, so objective deltas remain exact while avoiding an O(M²) recomputation for frozen
    reference fields.
    """

    pred_density = mixture_inner_product(
        predicted.means,
        predicted.covariances,
        predicted.density_amplitudes,
        predicted.means,
        predicted.covariances,
        predicted.density_amplitudes,
        chunk_size=chunk_size,
    )
    cross_density = mixture_inner_product(
        predicted.means,
        predicted.covariances,
        predicted.density_amplitudes,
        target.means,
        target.covariances,
        target.density_amplitudes,
        chunk_size=chunk_size,
    )
    density_scale = (
        target.density_amplitudes.sum().square().clamp_min(torch.finfo(predicted.means.dtype).tiny)
    )
    result = density_weight * (pred_density - 2.0 * cross_density) / density_scale
    if include_rgb and rgb_weight:
        pred_rgb = mixture_inner_product(
            predicted.means,
            predicted.covariances,
            predicted.rgb_amplitudes,
            predicted.means,
            predicted.covariances,
            predicted.rgb_amplitudes,
            chunk_size=chunk_size,
        )
        cross_rgb = mixture_inner_product(
            predicted.means,
            predicted.covariances,
            predicted.rgb_amplitudes,
            target.means,
            target.covariances,
            target.rgb_amplitudes,
            chunk_size=chunk_size,
        )
        rgb_scale = (
            target.rgb_amplitudes.square().sum().clamp_min(torch.finfo(predicted.means.dtype).tiny)
        )
        result = result + rgb_weight * (pred_rgb - 2.0 * cross_rgb) / rgb_scale
    return result


def _observability_reports(
    means: torch.Tensor,
    cameras: Sequence[Camera],
    visibility: torch.Tensor,
    condition_limit: float,
) -> tuple[CovarianceObservabilityReport, ...]:
    reports: list[CovarianceObservabilityReport] = []
    for track in range(means.shape[0]):
        weights = visibility[:, track].detach().clone()
        if not bool((weights > 0).any()):
            weights[0] = 1.0
        reports.append(
            analyze_covariance_observability(
                means[track].detach(),
                cameras,
                view_weights=weights,
                rcond=1.0 / condition_limit,
            )
        )
    return tuple(reports)


def _null_prior(
    covariances: torch.Tensor,
    initial_covariances: torch.Tensor,
    reports: Sequence[CovarianceObservabilityReport],
) -> torch.Tensor:
    current = covariance_to_vector(covariances)
    initial = covariance_to_vector(initial_covariances)
    penalty = current.new_zeros(())
    terms = 0
    for track, report in enumerate(reports):
        if report.null_basis.shape[1]:
            coordinate = report.null_basis.transpose(0, 1) @ (current[track] - initial[track])
            penalty = penalty + coordinate.square().sum()
            terms += coordinate.numel()
    return penalty / max(terms, 1)


def _gain_for_view(
    *,
    fiber: InverseProjectionFiber,
    appearance: SourceAnchoredSH,
    camera: Camera,
    visibility: torch.Tensor,
    field_masses: torch.Tensor,
    target: AnalyticGaussianField2D,
    ridge: float,
    chunk_size: int,
) -> torch.Tensor:
    base = _predicted_field(
        fiber=fiber,
        appearance=appearance,
        camera=camera,
        visibility=visibility,
        gain=field_masses.new_tensor(1.0),
        field_masses=field_masses,
    )
    energy = mixture_inner_product(
        base.means,
        base.covariances,
        base.density_amplitudes,
        base.means,
        base.covariances,
        base.density_amplitudes,
        chunk_size=chunk_size,
    )
    cross = mixture_inner_product(
        base.means,
        base.covariances,
        base.density_amplitudes,
        target.means,
        target.covariances,
        target.density_amplitudes,
        chunk_size=chunk_size,
    )
    return solve_view_gains(
        energy.reshape(1, 1),
        cross.reshape(1, 1),
        ridge=ridge,
    ).gains[0]


def fit_field_fibers(
    *,
    fiber: InverseProjectionFiber,
    reference_fields: Sequence[AnalyticGaussianField2D],
    cameras: Sequence[Camera],
    source_colors: torch.Tensor,
    field_masses: torch.Tensor,
    render_opacity: torch.Tensor,
    config: FieldRefitConfig | None = None,
) -> FieldRefitResult:
    """Fit geometry and staged source-exact SH appearance without component matching."""

    config = FieldRefitConfig() if config is None else config
    camera_tuple = tuple(cameras)
    target_tuple = tuple(reference_fields)
    if len(camera_tuple) != len(target_tuple) or not camera_tuple:
        raise ValueError("reference_fields and cameras must have equal nonzero length")
    if len(camera_tuple) != len(fiber.cameras) or any(
        actual is not expected for actual, expected in zip(camera_tuple, fiber.cameras, strict=True)
    ):
        raise ValueError("fiber cameras must be the ordered fitting cameras")
    count = fiber.n
    for name, tensor, shape in (
        ("source_colors", source_colors, (count, 3)),
        ("field_masses", field_masses, (count,)),
        ("render_opacity", render_opacity, (count,)),
    ):
        if tensor.shape != shape or tensor.device != fiber.source_means2d.device:
            raise ValueError(f"{name} must have shape {shape} and share the fiber device")
        if tensor.dtype != fiber.source_means2d.dtype or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite and share the fiber dtype")
    if bool((field_masses < 0).any()):
        raise ValueError("field_masses must be non-negative")
    if bool(((render_opacity < 0) | (render_opacity > 1)).any()):
        raise ValueError("render_opacity must lie in [0,1]")

    initial_means, initial_covariances = fiber.means_covariances()
    initial_cross = fiber.cross.detach().clone()
    initial_log_ray_scale = fiber.log_ray_scale.detach().clone()
    source_directions = _source_directions(
        initial_means,
        camera_tuple,
        fiber.source_view_indices,
    )
    appearance = _constant_initialized_appearance(
        degree=config.sh_degree,
        source_directions=source_directions,
        source_colors=source_colors,
    )
    optimizer = torch.optim.Adam(
        [fiber.depth_logits, fiber.cross, fiber.log_ray_scale, appearance.free],
        lr=config.learning_rate,
    )

    gains = field_masses.new_ones(len(camera_tuple))
    current_gaussians = _materialize(fiber, appearance, render_opacity)
    visibility = center_transmittance_visibility(
        current_gaussians,
        camera_tuple,
        source_view_indices=fiber.source_view_indices,
        force_source_visible=config.force_source_visible,
        dilation=fiber.dilation,
    )
    reports = _observability_reports(
        initial_means,
        camera_tuple,
        visibility.weights,
        config.observability_condition_limit,
    )
    covariance_free_mask = torch.tensor(
        [report.full_rank for report in reports],
        dtype=torch.bool,
        device=field_masses.device,
    )

    def objective(step: int) -> torch.Tensor:
        total = field_masses.new_zeros(())
        include_rgb = step >= config.appearance_start
        for view, (camera, target) in enumerate(zip(camera_tuple, target_tuple, strict=True)):
            predicted = _predicted_field(
                fiber=fiber,
                appearance=appearance,
                camera=camera,
                visibility=visibility.weights[view],
                gain=gains[view],
                field_masses=field_masses,
            )
            total = total + _variable_field_objective(
                predicted,
                target,
                density_weight=config.density_weight,
                rgb_weight=config.rgb_weight,
                include_rgb=include_rgb,
                chunk_size=config.chunk_size,
            )
        total = total / len(camera_tuple)
        if config.null_prior_weight:
            _means, covariances = fiber.means_covariances()
            total = total + config.null_prior_weight * _null_prior(
                covariances,
                initial_covariances.detach(),
                reports,
            )
        return total

    history: list[float] = [float(objective(0).detach())]
    accepted = 0
    parameters = [fiber.depth_logits, fiber.cross, fiber.log_ray_scale, appearance.free]
    for step in range(config.iterations):
        if step % config.visibility_refresh == 0:
            with torch.no_grad():
                current_gaussians = _materialize(fiber, appearance, render_opacity)
                visibility = center_transmittance_visibility(
                    current_gaussians,
                    camera_tuple,
                    source_view_indices=fiber.source_view_indices,
                    force_source_visible=config.force_source_visible,
                    dilation=fiber.dilation,
                )
                gains = torch.stack(
                    [
                        _gain_for_view(
                            fiber=fiber,
                            appearance=appearance,
                            camera=camera,
                            visibility=visibility.weights[view],
                            field_masses=field_masses,
                            target=target,
                            ridge=config.gain_ridge,
                            chunk_size=config.chunk_size,
                        )
                        for view, (camera, target) in enumerate(
                            zip(camera_tuple, target_tuple, strict=True)
                        )
                    ]
                )
                means, _covariances = fiber.means_covariances()
                reports = _observability_reports(
                    means,
                    camera_tuple,
                    visibility.weights,
                    config.observability_condition_limit,
                )
                covariance_free_mask = torch.tensor(
                    [report.full_rank for report in reports],
                    dtype=torch.bool,
                    device=field_masses.device,
                )

        optimizer.zero_grad(set_to_none=True)
        before = objective(step)
        before.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
        snapshot = [parameter.detach().clone() for parameter in parameters]
        optimizer.step()
        with torch.no_grad():
            pinned = ~covariance_free_mask
            fiber.cross[pinned] = initial_cross[pinned]
            fiber.log_ray_scale[pinned] = initial_log_ray_scale[pinned]
        after = objective(step)
        if bool(torch.isfinite(after)) and float(after.detach()) <= float(before.detach()) + 1e-10:
            accepted += 1
            history.append(float(after.detach()))
        else:
            with torch.no_grad():
                for parameter, saved in zip(parameters, snapshot, strict=True):
                    parameter.copy_(saved)
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
            history.append(float(before.detach()))

        source_means, source_covariances, _depth = fiber.source_projection()
        source_error = torch.maximum(
            (source_means - fiber.source_means2d).abs().amax(),
            (source_covariances - fiber.source_covariances2d).abs().amax(),
        )
        tolerance = 2e-9 if source_means.dtype == torch.float64 else 2e-4
        detached_source_error = source_error.detach()
        if (
            not bool(torch.isfinite(detached_source_error))
            or float(detached_source_error) > tolerance
        ):
            raise RuntimeError("fiber optimizer violated the exact source projection")

    gaussians = _materialize(fiber, appearance, render_opacity).detach()
    source_means, source_covariances, _depth = fiber.source_projection()
    source_projection_error = torch.maximum(
        (source_means - fiber.source_means2d).abs().amax(),
        (source_covariances - fiber.source_covariances2d).abs().amax(),
    )
    return FieldRefitResult(
        gaussians=gaussians,
        fiber=fiber,
        appearance=appearance,
        field_masses=field_masses.detach().clone(),
        render_opacity=render_opacity.detach().clone(),
        visibility=visibility,
        gains=gains.detach().clone(),
        objective_history=tuple(history),
        accepted_steps=accepted,
        observability=reports,
        covariance_free_mask=covariance_free_mask.detach().clone(),
        source_projection_max_error=float(source_projection_error.detach()),
        source_color_max_error=float(appearance.source_max_abs_error().detach()),
    )


__all__ = [
    "FieldRefitConfig",
    "FieldRefitResult",
    "fit_field_fibers",
]
