"""Semantic validation for projected Gaussian observation fields.

The validator deliberately queries both the frozen observation field and the
projected 3D prediction through :meth:`GaussianObservationField.query`.  This
keeps validation aligned with normalized/additive blending, compact support,
support fading, epsilon handling, and affine teacher colors.

Rendering opacity is not an observation-field mass.  Callers must provide
``field_masses`` explicitly; :class:`Gaussians3D.opacity` is never consulted.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.core.sh import eval_sh
from rtgs.data.field_inputs import SceneFits
from rtgs.render.projection import EWA_NEAR, project_gaussians_ewa


@dataclass(frozen=True)
class FieldValidationConfig:
    """Deterministic, bounded sampling controls."""

    sample_cap_per_view: int = 4096
    seed: int = 0
    component_chunk: int = 256

    def __post_init__(self) -> None:
        if self.sample_cap_per_view <= 0:
            raise ValueError("sample_cap_per_view must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.component_chunk <= 0:
            raise ValueError("component_chunk must be positive")


@dataclass(frozen=True)
class ViewFieldMetrics:
    """Density and RGB errors for one view."""

    view_index: int
    view_name: str
    split: str
    n_samples: int
    density_mse: float
    density_mae: float
    rgb_mse: float
    rgb_mae: float
    sample_sha256: str


@dataclass(frozen=True)
class FieldMetricSummary:
    """Sample-weighted metric summary for a view split."""

    n_views: int
    n_samples: int
    density_mse: float
    density_mae: float
    rgb_mse: float
    rgb_mae: float


@dataclass(frozen=True)
class FieldValidationResult:
    """Per-view metrics plus isolated train and held-out aggregates."""

    per_view: tuple[ViewFieldMetrics, ...]
    train: FieldMetricSummary
    heldout: FieldMetricSummary | None


def _view_seed(seed: int, view_index: int, view_name: str) -> int:
    payload = f"rtgs-field-validation-v1\0{seed}\0{view_index}\0{view_name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def _sample_points(
    field: GaussianObservationField,
    *,
    view_index: int,
    view_name: str,
    config: FieldValidationConfig,
) -> Tensor:
    x0, y0, window_width, window_height = field.fit_window
    population = window_width * window_height
    count = min(config.sample_cap_per_view, population)

    if count == population:
        linear = torch.arange(population, dtype=torch.int64)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_view_seed(config.seed, view_index, view_name))
        offset = int(torch.randint(population, (), generator=generator).item())
        stride = int(torch.randint(1, population, (), generator=generator).item())
        while math.gcd(stride, population) != 1:
            stride += 1
            if stride == population:
                stride = 1
        linear = (offset + stride * torch.arange(count, dtype=torch.int64)) % population

    xs = linear.remainder(window_width) + x0
    ys = torch.div(linear, window_width, rounding_mode="floor") + y0
    xy = torch.stack((xs, ys), dim=-1).to(device=field.means.device, dtype=field.means.dtype)
    return xy + 0.5


def _sample_digest(xy: Tensor) -> str:
    cpu_xy = xy.detach().to(device="cpu").contiguous()
    header = f"{tuple(cpu_xy.shape)}:{cpu_xy.dtype}".encode()
    return hashlib.sha256(header + cpu_xy.numpy().tobytes()).hexdigest()


def _select_field_masses(
    field_masses: Tensor,
    *,
    view_index: int,
    n_views: int,
    n_gaussians: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if field_masses.ndim == 1:
        if field_masses.shape != (n_gaussians,):
            raise ValueError(
                "one-dimensional field_masses must have shape "
                f"({n_gaussians},), got {tuple(field_masses.shape)}"
            )
        masses = field_masses
    elif field_masses.ndim == 2:
        if field_masses.shape != (n_views, n_gaussians):
            raise ValueError(
                "two-dimensional field_masses must have shape "
                f"({n_views}, {n_gaussians}), got {tuple(field_masses.shape)}"
            )
        masses = field_masses[view_index]
    else:
        raise ValueError("field_masses must have shape (N,) or (V, N)")

    if masses.device != device or masses.dtype != dtype:
        raise ValueError("field_masses must share the Gaussians3D device and dtype")
    if not torch.isfinite(masses).all() or (masses < 0).any():
        raise ValueError("field_masses must be finite and non-negative")
    return masses


def _projected_field(
    gaussians: Gaussians3D,
    camera: Camera,
    masses: Tensor,
    teacher: GaussianObservationField,
) -> GaussianObservationField | None:
    """Project 3D Gaussians into a field with the teacher's query semantics."""

    projection = project_gaussians_ewa(gaussians, camera, dilation=0.0)
    finite_covs = torch.isfinite(projection.covariances2d).all(dim=(-2, -1))
    finite_means = torch.isfinite(projection.means2d).all(dim=-1)
    valid = (projection.depth > EWA_NEAR) & finite_covs & finite_means

    candidate_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        return None

    covs = projection.covariances2d[candidate_indices]
    eigenvalues, eigenvectors = torch.linalg.eigh(covs)
    spd = torch.isfinite(eigenvalues).all(dim=-1) & (eigenvalues > 0).all(dim=-1)
    candidate_indices = candidate_indices[spd]
    eigenvalues = eigenvalues[spd]
    eigenvectors = eigenvectors[spd]
    if candidate_indices.numel() == 0:
        return None

    means_2d = projection.means2d[candidate_indices]
    first_axis = eigenvectors[:, :, 0]
    rotations = torch.atan2(first_axis[:, 1], first_axis[:, 0])
    log_scales = 0.5 * torch.log(eigenvalues)

    camera_position = camera.position.to(device=gaussians.means.device, dtype=gaussians.means.dtype)
    directions = gaussians.means[candidate_indices] - camera_position
    direction_norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    directions = directions / direction_norms.clamp_min(torch.finfo(directions.dtype).eps)
    colors = eval_sh(
        gaussians.sh_degree,
        gaussians.sh[candidate_indices],
        directions,
    )

    return GaussianObservationField(
        width=teacher.width,
        height=teacher.height,
        means=means_2d,
        log_scales=log_scales,
        rotations=rotations,
        colors=colors,
        amplitudes=masses[candidate_indices],
        provider="synthetic_fixture",
        view_id=teacher.view_id,
        sigma_cutoff=teacher.sigma_cutoff,
        support_fade_alpha=teacher.support_fade_alpha,
        aa_dilation=teacher.aa_dilation,
        blend_mode=teacher.blend_mode,
        epsilon=teacher.epsilon,
        fit_window=teacher.fit_window,
    )


def _zero_prediction(xy: Tensor) -> tuple[Tensor, Tensor]:
    density = torch.zeros(xy.shape[0], device=xy.device, dtype=xy.dtype)
    color = torch.zeros((xy.shape[0], 3), device=xy.device, dtype=xy.dtype)
    return density, color


def _view_metrics(
    *,
    view_index: int,
    view_name: str,
    split: str,
    teacher: GaussianObservationField,
    predicted: GaussianObservationField | None,
    config: FieldValidationConfig,
) -> ViewFieldMetrics:
    xy = _sample_points(
        teacher,
        view_index=view_index,
        view_name=view_name,
        config=config,
    )
    target = teacher.query(xy, component_chunk=config.component_chunk)
    if predicted is None:
        predicted_density, predicted_color = _zero_prediction(xy)
    else:
        prediction = predicted.query(
            xy.to(dtype=predicted.dtype),
            component_chunk=config.component_chunk,
        )
        predicted_density = prediction.weight_sum.to(target.weight_sum)
        predicted_color = prediction.color.to(target.color)

    density_error = (predicted_density - target.weight_sum).double()
    rgb_error = (predicted_color - target.color).double()
    return ViewFieldMetrics(
        view_index=view_index,
        view_name=view_name,
        split=split,
        n_samples=xy.shape[0],
        density_mse=float(density_error.square().mean().item()),
        density_mae=float(density_error.abs().mean().item()),
        rgb_mse=float(rgb_error.square().mean().item()),
        rgb_mae=float(rgb_error.abs().mean().item()),
        sample_sha256=_sample_digest(xy),
    )


def _summarize(
    per_view: tuple[ViewFieldMetrics, ...],
    indices: tuple[int, ...],
) -> FieldMetricSummary | None:
    selected = tuple(per_view[index] for index in indices)
    if not selected:
        return None
    total_samples = sum(metric.n_samples for metric in selected)

    def weighted(name: str) -> float:
        return (
            sum(float(getattr(metric, name)) * metric.n_samples for metric in selected)
            / total_samples
        )

    return FieldMetricSummary(
        n_views=len(selected),
        n_samples=total_samples,
        density_mse=weighted("density_mse"),
        density_mae=weighted("density_mae"),
        rgb_mse=weighted("rgb_mse"),
        rgb_mae=weighted("rgb_mae"),
    )


def validate_field_semantics(
    fits: SceneFits,
    cameras: Sequence[Camera],
    gaussians: Gaussians3D,
    field_masses: Tensor,
    *,
    config: FieldValidationConfig | None = None,
) -> FieldValidationResult:
    """Compare frozen teacher fields with projected fitted 3D Gaussians.

    Every view is evaluated, but train and held-out aggregates are formed from
    their respective :class:`SceneFits` index sets only.  Held-out observations
    are therefore reporting-only and cannot affect train metrics or samples.
    """

    config = config or FieldValidationConfig()
    if len(cameras) != fits.n_views:
        raise ValueError(f"expected {fits.n_views} cameras, got {len(cameras)}")

    geometry_tensors = (
        gaussians.means,
        gaussians.quats,
        gaussians.log_scales,
        gaussians.sh,
    )
    if not all(torch.isfinite(value).all() for value in geometry_tensors):
        raise ValueError("Gaussians3D geometry and SH values must be finite")

    train_indices = set(fits.train_view_indices)
    per_view_metrics: list[ViewFieldMetrics] = []
    for view_index, (teacher, camera, view_name) in enumerate(
        zip(fits.observations, cameras, fits.view_names, strict=True)
    ):
        if camera.width != teacher.width or camera.height != teacher.height:
            raise ValueError(f"camera/field canvas mismatch for view {view_index}")
        if teacher.means.device != gaussians.means.device:
            raise ValueError("teacher fields and Gaussians3D must share a device")
        masses = _select_field_masses(
            field_masses,
            view_index=view_index,
            n_views=fits.n_views,
            n_gaussians=gaussians.n,
            device=gaussians.means.device,
            dtype=gaussians.means.dtype,
        )
        predicted = _projected_field(gaussians, camera, masses, teacher)
        per_view_metrics.append(
            _view_metrics(
                view_index=view_index,
                view_name=view_name,
                split="train" if view_index in train_indices else "heldout",
                teacher=teacher,
                predicted=predicted,
                config=config,
            )
        )

    per_view = tuple(per_view_metrics)
    train = _summarize(per_view, fits.train_view_indices)
    if train is None:
        raise RuntimeError("SceneFits must contain at least one train view")
    heldout = _summarize(per_view, fits.heldout_view_indices)
    return FieldValidationResult(
        per_view=per_view,
        train=train,
        heldout=heldout,
    )


__all__ = [
    "FieldMetricSummary",
    "FieldValidationConfig",
    "FieldValidationResult",
    "ViewFieldMetrics",
    "validate_field_semantics",
]
