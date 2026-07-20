"""Block-fixed visibility and gain estimates for image-free field fitting.

Visibility is the incoming transmittance at each projected Gaussian center.  It is deliberately
returned detached: the field-fit loop may refresh it between optimizer blocks, but must not use
visibility as a differentiable escape route.  Per-view gains are likewise closed-form nuisance
parameters, separate from both 2D mixture amplitude and 3D rendering opacity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.render.projection import EWA_DILATION, EWA_NEAR, project_gaussians_ewa

GainMode = Literal["scalar", "rgb"]


@dataclass(frozen=True)
class CenterVisibility:
    """Detached per-view center-transmittance evidence."""

    weights: torch.Tensor  # (V,N)
    valid: torch.Tensor  # (V,N), in front of near and center inside the canvas
    means2d: torch.Tensor  # (V,N,2)
    depths: torch.Tensor  # (V,N)
    source_forced: torch.Tensor  # (V,N)


@dataclass(frozen=True)
class ViewGainEstimate:
    """Closed-form positive gain and its unclamped sufficient-statistic solution."""

    gains: torch.Tensor  # (V,) for scalar or (V,C) for RGB
    unconstrained: torch.Tensor
    numerator: torch.Tensor
    denominator: torch.Tensor
    clipped: torch.Tensor
    mode: GainMode


def center_transmittance_visibility(
    gaussians: Gaussians3D,
    cameras: Sequence[Camera],
    *,
    source_view_indices: torch.Tensor | None = None,
    force_source_visible: bool = False,
    near: float = EWA_NEAR,
    dilation: float = EWA_DILATION,
    target_chunk_size: int = 512,
) -> CenterVisibility:
    """Evaluate deterministic incoming transmittance at projected Gaussian centers.

    For target Gaussian ``i`` in one view, every Gaussian ``j`` with strictly smaller
    camera-space center depth contributes

    ``alpha_ij = opacity_j exp(-0.5 (u_i-u_j)^T C_j^-1 (u_i-u_j))``.

    The visibility is ``prod_j (1-alpha_ij)``. Exact center-depth ties do not occlude one
    another, which makes the result permutation-equivariant instead of depending on a hidden
    sort tie. A target behind ``near`` or whose center is off-canvas always receives zero.
    Source forcing only replaces otherwise-valid source entries by one; it never resurrects an
    invalid projection.
    """

    camera_tuple = tuple(cameras)
    if not camera_tuple:
        raise ValueError("center visibility requires at least one camera")
    if not isinstance(target_chunk_size, int) or isinstance(target_chunk_size, bool):
        raise TypeError("target_chunk_size must be an integer")
    if target_chunk_size <= 0:
        raise ValueError("target_chunk_size must be positive")
    if not math.isfinite(near) or near <= 0:
        raise ValueError("near must be finite and positive")
    if not math.isfinite(dilation) or dilation < 0:
        raise ValueError("dilation must be finite and non-negative")
    if (
        not gaussians.opacity.is_floating_point()
        or not bool(torch.isfinite(gaussians.opacity).all())
        or bool(((gaussians.opacity < 0) | (gaussians.opacity > 1)).any())
    ):
        raise ValueError("Gaussian opacity must be finite and lie in [0,1]")
    if source_view_indices is not None:
        if (
            source_view_indices.shape != (gaussians.n,)
            or source_view_indices.dtype != torch.long
            or source_view_indices.device != gaussians.means.device
        ):
            raise ValueError(
                "source_view_indices must have shape (N,), dtype long, and share device"
            )
        if source_view_indices.numel() and (
            int(source_view_indices.min()) < 0
            or int(source_view_indices.max()) >= len(camera_tuple)
        ):
            raise ValueError("source_view_indices contains an unavailable camera")
    if force_source_visible and source_view_indices is None:
        raise ValueError("force_source_visible requires source_view_indices")

    weights: list[torch.Tensor] = []
    valids: list[torch.Tensor] = []
    projected_means: list[torch.Tensor] = []
    depths: list[torch.Tensor] = []
    forced_masks: list[torch.Tensor] = []
    with torch.no_grad():
        opacity = gaussians.opacity.detach()
        for view_index, camera in enumerate(camera_tuple):
            projection = project_gaussians_ewa(
                gaussians,
                camera,
                dilation=dilation,
                near=near,
            )
            mean = projection.means2d.detach()
            depth = projection.depth.detach()
            covariance = projection.covariances2d.detach()
            if (
                not bool(torch.isfinite(mean).all())
                or not bool(torch.isfinite(depth).all())
                or not bool(torch.isfinite(covariance).all())
            ):
                raise ValueError("projected Gaussian geometry must be finite")
            if covariance.numel() and bool((torch.linalg.eigvalsh(covariance) <= 0).any()):
                raise ValueError("projected Gaussian covariance must be positive definite")
            inverse = torch.linalg.inv(covariance)
            valid = (depth > near) & camera.in_image(mean)
            occluder_valid = depth > near
            local = mean.new_zeros((gaussians.n,))
            for start in range(0, gaussians.n, target_chunk_size):
                end = min(start + target_chunk_size, gaussians.n)
                delta = mean[start:end, None, :] - mean[None, :, :]
                mahalanobis = torch.einsum(
                    "tji,jik,tjk->tj",
                    delta,
                    inverse,
                    delta,
                ).clamp_min(0)
                alpha = opacity[None, :] * torch.exp(-0.5 * mahalanobis)
                front = depth[None, :] < depth[start:end, None]
                active = front & occluder_valid[None, :]
                alpha = torch.where(active, alpha.clamp(0.0, 1.0), torch.zeros_like(alpha))
                local[start:end] = torch.prod(1.0 - alpha, dim=1)
            local = torch.where(valid, local, torch.zeros_like(local))

            forced = torch.zeros_like(valid)
            if force_source_visible:
                assert source_view_indices is not None
                forced = valid & (source_view_indices == view_index)
                local = torch.where(forced, torch.ones_like(local), local)
            weights.append(local)
            valids.append(valid)
            projected_means.append(mean)
            depths.append(depth)
            forced_masks.append(forced)

    return CenterVisibility(
        weights=torch.stack(weights),
        valid=torch.stack(valids),
        means2d=torch.stack(projected_means),
        depths=torch.stack(depths),
        source_forced=torch.stack(forced_masks),
    )


def solve_view_gains(
    predicted_energy: torch.Tensor,
    predicted_target_inner: torch.Tensor,
    *,
    mode: GainMode = "scalar",
    ridge: float = 1e-6,
    prior_gain: float = 1.0,
    min_gain: float = 0.05,
    max_gain: float = 20.0,
) -> ViewGainEstimate:
    """Solve positive scalar/RGB field gains from per-view sufficient statistics.

    Inputs have shape ``(V,C)``. They correspond to ``<Fhat,Fhat>`` and
    ``<Fhat,Fref>`` for each view/channel. The ridge is centered on ``prior_gain``:

    ``g = (cross + ridge*prior_gain) / (energy + ridge)``.
    """

    if predicted_energy.ndim != 2 or predicted_target_inner.shape != predicted_energy.shape:
        raise ValueError("gain statistics must have equal shape (V,C)")
    if 0 in predicted_energy.shape:
        raise ValueError("gain statistics must be non-empty")
    if (
        not predicted_energy.is_floating_point()
        or predicted_target_inner.device != predicted_energy.device
        or predicted_target_inner.dtype != predicted_energy.dtype
    ):
        raise ValueError("gain statistics must share floating dtype and device")
    if (
        not bool(torch.isfinite(predicted_energy).all())
        or not bool(torch.isfinite(predicted_target_inner).all())
        or bool((predicted_energy < 0).any())
    ):
        raise ValueError("gain energies must be non-negative and all statistics finite")
    if mode not in {"scalar", "rgb"}:
        raise ValueError("gain mode must be 'scalar' or 'rgb'")
    for name, value in (
        ("ridge", ridge),
        ("prior_gain", prior_gain),
        ("min_gain", min_gain),
        ("max_gain", max_gain),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    if prior_gain <= 0 or min_gain <= 0 or max_gain < min_gain:
        raise ValueError("gain prior and bounds must be positive with min_gain <= max_gain")

    with torch.no_grad():
        energy = predicted_energy.detach()
        cross = predicted_target_inner.detach()
        if mode == "scalar":
            energy = energy.sum(dim=-1)
            cross = cross.sum(dim=-1)
        numerator = cross + ridge * prior_gain
        denominator = energy + ridge
        unconstrained = numerator / denominator
        gains = unconstrained.clamp(min=min_gain, max=max_gain)
    return ViewGainEstimate(
        gains=gains,
        unconstrained=unconstrained,
        numerator=numerator,
        denominator=denominator,
        clipped=gains != unconstrained,
        mode=mode,
    )


def estimate_view_gains(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    mode: GainMode = "scalar",
    ridge: float = 1e-6,
    prior_gain: float = 1.0,
    min_gain: float = 0.05,
    max_gain: float = 20.0,
) -> ViewGainEstimate:
    """Convenience least-squares gain estimate for sampled ``(V,S,C)`` field values."""

    if predicted.ndim != 3 or target.shape != predicted.shape or 0 in predicted.shape:
        raise ValueError("predicted and target must have equal non-empty shape (V,S,C)")
    if (
        not predicted.is_floating_point()
        or target.device != predicted.device
        or target.dtype != predicted.dtype
        or not bool(torch.isfinite(predicted).all())
        or not bool(torch.isfinite(target).all())
    ):
        raise ValueError("predicted and target must be finite with common floating dtype/device")
    if weights is None:
        expanded_weights = torch.ones_like(predicted)
    else:
        if weights.shape == predicted.shape[:2]:
            expanded_weights = weights[:, :, None].expand_as(predicted)
        elif weights.shape == predicted.shape:
            expanded_weights = weights
        else:
            raise ValueError("weights must have shape (V,S) or (V,S,C)")
        if (
            not expanded_weights.is_floating_point()
            or expanded_weights.device != predicted.device
            or expanded_weights.dtype != predicted.dtype
            or not bool(torch.isfinite(expanded_weights).all())
            or bool((expanded_weights < 0).any())
        ):
            raise ValueError("weights must be finite, non-negative, and match dtype/device")
    energy = (expanded_weights * predicted.square()).sum(dim=1)
    cross = (expanded_weights * predicted * target).sum(dim=1)
    return solve_view_gains(
        energy,
        cross,
        mode=mode,
        ridge=ridge,
        prior_gain=prior_gain,
        min_gain=min_gain,
        max_gain=max_gain,
    )


__all__ = [
    "CenterVisibility",
    "GainMode",
    "ViewGainEstimate",
    "center_transmittance_visibility",
    "estimate_view_gains",
    "solve_view_gains",
]
