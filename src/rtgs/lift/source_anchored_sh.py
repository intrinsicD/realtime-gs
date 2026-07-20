"""Source-exact spherical-harmonics appearance for fitted Gaussian tracks.

The geometry fiber has an exact source projection.  Appearance can use the same separation:
one linear SH evaluation row is fixed to the source observation, while all coefficients in that
row's null space remain trainable from other views.  This module intentionally knows nothing
about opacity or association; callers provide detached supervision weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from rtgs.core.sh import C0, C1, C2, C3, eval_sh_preactivation, num_sh_bases


def real_sh_basis(degree: int, directions: torch.Tensor) -> torch.Tensor:
    """Return the unshifted 3DGS SH evaluation row for each direction.

    ``eval_sh_preactivation(degree, coefficients, directions)`` equals the row-wise inner
    product of this basis and ``coefficients``, plus the standard ``0.5`` color shift.
    Nonzero finite directions are normalized so the constraint and subsequent evaluation use
    exactly the same unit-vector convention.
    """

    if degree < 0 or degree > 3:
        raise ValueError("degree must be in [0,3]")
    if directions.ndim < 2 or directions.shape[-1] != 3:
        raise ValueError("directions must end in shape (3,)")
    if not directions.is_floating_point() or not bool(torch.isfinite(directions).all()):
        raise ValueError("directions must be finite and floating point")
    norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    if bool((norms <= 0).any()):
        raise ValueError("directions must be nonzero")
    unit = directions / norms
    x, y, z = unit.unbind(dim=-1)
    rows = [torch.full_like(x, C0)]
    if degree >= 1:
        rows.extend([-C1 * y, C1 * z, -C1 * x])
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        rows.extend(
            [
                C2[0] * x * y,
                C2[1] * y * z,
                C2[2] * (2.0 * zz - xx - yy),
                C2[3] * x * z,
                C2[4] * (xx - yy),
            ]
        )
    if degree >= 3:
        xx, yy, zz = x * x, y * y, z * z
        rows.extend(
            [
                C3[0] * y * (3.0 * xx - yy),
                C3[1] * x * y * z,
                C3[2] * y * (4.0 * zz - xx - yy),
                C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
                C3[4] * x * (4.0 * zz - xx - yy),
                C3[5] * z * (xx - yy),
                C3[6] * x * (xx - 3.0 * yy),
            ]
        )
    result = torch.stack(rows, dim=-1)
    if result.shape[-1] != num_sh_bases(degree):
        raise RuntimeError("internal SH basis count mismatch")
    return result


class SourceAnchoredSH(nn.Module):
    """SH coefficients constrained to one exact source color per Gaussian."""

    def __init__(
        self,
        *,
        degree: int,
        source_directions: torch.Tensor,
        source_colors: torch.Tensor,
    ) -> None:
        super().__init__()
        if source_directions.ndim != 2 or source_directions.shape[1] != 3:
            raise ValueError("source_directions must have shape (N,3)")
        count = source_directions.shape[0]
        if source_colors.shape != (count, 3):
            raise ValueError("source_colors must have shape (N,3)")
        if (
            not source_colors.is_floating_point()
            or source_colors.device != source_directions.device
            or source_colors.dtype != source_directions.dtype
            or not bool(torch.isfinite(source_colors).all())
        ):
            raise ValueError("source colors and directions must be finite with common dtype/device")
        if count == 0:
            raise ValueError("at least one source appearance is required")
        basis = real_sh_basis(degree, source_directions)
        norm_squared = basis.square().sum(dim=-1, keepdim=True)
        particular = (
            basis[:, :, None] * (source_colors - 0.5)[:, None, :] / norm_squared[:, :, None]
        )
        self.degree = degree
        self.register_buffer("source_basis", basis.detach().clone())
        self.register_buffer("source_colors", source_colors.detach().clone())
        self.register_buffer("particular", particular.detach().clone())
        self.free = nn.Parameter(torch.zeros_like(particular))

    @property
    def n(self) -> int:
        return int(self.source_colors.shape[0])

    def coefficients(self) -> torch.Tensor:
        """Return coefficients after exact orthogonal projection into the source null space."""

        basis = self.source_basis
        norm_squared = basis.square().sum(dim=-1, keepdim=True)
        parallel_scale = (self.free * basis[:, :, None]).sum(dim=1, keepdim=True)
        null = self.free - basis[:, :, None] * parallel_scale / norm_squared[:, :, None]
        return self.particular + null

    def preactivation(self, directions: torch.Tensor) -> torch.Tensor:
        """Evaluate shifted SH color before the renderer's nonnegative activation."""

        if directions.shape != (self.n, 3):
            raise ValueError("directions must have shape (N,3)")
        unit = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True).clamp_min(
            torch.finfo(directions.dtype).tiny
        )
        return eval_sh_preactivation(self.degree, self.coefficients(), unit)

    def source_preactivation(self) -> torch.Tensor:
        """Evaluate the exact source row without reconstructing source directions."""

        return (self.source_basis[:, :, None] * self.coefficients()).sum(dim=1) + 0.5

    def source_max_abs_error(self) -> torch.Tensor:
        return (self.source_preactivation() - self.source_colors).abs().amax()


@dataclass(frozen=True)
class SourceAnchoredSHFitConfig:
    """Deterministic optimizer settings for the post-geometry appearance phase."""

    degree: int = 1
    iterations: int = 100
    learning_rate: float = 0.05
    huber_delta: float = 0.10

    def __post_init__(self) -> None:
        if self.degree < 0 or self.degree > 3:
            raise ValueError("degree must be in [0,3]")
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        for name in ("learning_rate", "huber_delta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class SourceAnchoredSHFitResult:
    model: SourceAnchoredSH
    losses: tuple[float, ...]
    source_max_abs_error: float


def fit_source_anchored_sh(
    *,
    source_directions: torch.Tensor,
    source_colors: torch.Tensor,
    view_directions: torch.Tensor,
    target_colors: torch.Tensor,
    weights: torch.Tensor,
    source_view_indices: torch.Tensor,
    config: SourceAnchoredSHFitConfig | None = None,
) -> SourceAnchoredSHFitResult:
    """Fit only source-null SH coordinates to detached multi-view color targets.

    Inputs use shape ``(V,N,...)``. The source-view entry for each Gaussian is zeroed internally,
    so the exact anchor can never count as optimization evidence. ``weights`` are association or
    visibility support supplied by the caller; they are not opacity.
    """

    if config is None:
        config = SourceAnchoredSHFitConfig()
    if view_directions.ndim != 3 or view_directions.shape[-1] != 3:
        raise ValueError("view_directions must have shape (V,N,3)")
    views, count = view_directions.shape[:2]
    if target_colors.shape != (views, count, 3) or weights.shape != (views, count):
        raise ValueError("target colors/weights must match view directions")
    if source_directions.shape != (count, 3) or source_colors.shape != (count, 3):
        raise ValueError("source tensors must match the track count")
    if source_view_indices.shape != (count,) or source_view_indices.dtype != torch.long:
        raise ValueError("source_view_indices must have shape (N,) and dtype long")
    tensors = (source_directions, source_colors, view_directions, target_colors, weights)
    if any(tensor.device != source_directions.device for tensor in tensors):
        raise ValueError("all appearance tensors must share a device")
    if any(tensor.dtype != source_directions.dtype for tensor in tensors):
        raise ValueError("all floating appearance tensors must share a dtype")
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise TypeError("appearance tensors must be floating point")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise ValueError("appearance tensors must be finite")
    if bool((weights < 0).any()):
        raise ValueError("weights must be non-negative")
    if source_view_indices.device != source_directions.device:
        raise ValueError("source indices must share the appearance device")
    if count == 0 or views == 0:
        raise ValueError("appearance fitting requires non-empty views and tracks")
    if int(source_view_indices.min()) < 0 or int(source_view_indices.max()) >= views:
        raise ValueError("source_view_indices contains an unavailable view")

    active_weights = weights.detach().clone()
    active_weights[source_view_indices, torch.arange(count, device=weights.device)] = 0.0
    weight_sum = active_weights.sum()
    if not bool(weight_sum > 0):
        raise ValueError("non-source appearance weight must be positive")
    target = target_colors.detach()
    directions = view_directions.detach()
    model = SourceAnchoredSH(
        degree=config.degree,
        source_directions=source_directions,
        source_colors=source_colors,
    )
    optimizer = torch.optim.Adam([model.free], lr=config.learning_rate)

    def objective() -> torch.Tensor:
        flat_directions = directions.reshape(views * count, 3)
        repeated = model.coefficients()[None].expand(views, -1, -1, -1)
        flat_coefficients = repeated.reshape(views * count, repeated.shape[2], 3)
        unit = flat_directions / torch.linalg.vector_norm(
            flat_directions, dim=-1, keepdim=True
        ).clamp_min(torch.finfo(flat_directions.dtype).tiny)
        predicted = eval_sh_preactivation(
            config.degree,
            flat_coefficients,
            unit,
        ).reshape(views, count, 3)
        residual = predicted - target
        absolute = residual.abs()
        delta = config.huber_delta
        huber = torch.where(
            absolute <= delta,
            0.5 * residual.square(),
            delta * (absolute - 0.5 * delta),
        )
        return (huber * active_weights[:, :, None]).sum() / (3.0 * weight_sum)

    losses = [float(objective().detach())]
    tolerance = 1e-10 if source_colors.dtype == torch.float64 else 2e-6
    for _step in range(config.iterations):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        optimizer.step()
        error = float(model.source_max_abs_error().detach())
        if not math.isfinite(error) or error > tolerance:
            raise RuntimeError("source-anchored SH equality drifted during optimization")
        losses.append(float(objective().detach()))
    return SourceAnchoredSHFitResult(
        model=model,
        losses=tuple(losses),
        source_max_abs_error=float(model.source_max_abs_error().detach()),
    )


__all__ = [
    "SourceAnchoredSH",
    "SourceAnchoredSHFitConfig",
    "SourceAnchoredSHFitResult",
    "fit_source_anchored_sh",
    "real_sh_basis",
]
