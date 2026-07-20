"""Shared perspective-EWA projection equations.

This module is intentionally renderer-independent.  Geometry-fitting code needs the same
Jacobian and dilation convention as the dense and sparse reference rasterizers; keeping the
equations here prevents a source-footprint constraint from silently targeting a different
camera model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D

EWA_DILATION = 0.3
EWA_NEAR = 0.05


@dataclass(frozen=True)
class EWAProjection:
    """Projected Gaussian geometry before visibility filtering."""

    means2d: torch.Tensor
    depth: torch.Tensor
    covariances2d: torch.Tensor
    means_cam: torch.Tensor
    jacobians: torch.Tensor


def project_covariances_ewa(
    means: torch.Tensor,
    covariances: torch.Tensor,
    camera: Camera,
    *,
    dilation: float = EWA_DILATION,
    near: float = EWA_NEAR,
) -> EWAProjection:
    """Project explicit world means and covariances with the reference EWA equations.

    ``dilation`` is variance in pixel squared.  Passing zero returns the intrinsic projected
    covariance.  Points behind ``near`` retain their signed returned depth, while the Jacobian
    uses the same clamped depth as the reference rasterizers.
    """
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape (N,3)")
    if covariances.shape != (means.shape[0], 3, 3):
        raise ValueError("covariances must have shape (N,3,3)")
    if not means.is_floating_point() or not covariances.is_floating_point():
        raise TypeError("means and covariances must be floating point")
    if means.device != covariances.device or means.dtype != covariances.dtype:
        raise ValueError("means and covariances must share device and dtype")
    if not bool(torch.isfinite(means).all()) or not bool(torch.isfinite(covariances).all()):
        raise ValueError("means and covariances must be finite")
    dilation_tensor = means.new_tensor(dilation)
    near_tensor = means.new_tensor(near)
    if not bool(torch.isfinite(dilation_tensor)) or dilation < 0:
        raise ValueError("dilation must be finite and non-negative")
    if not bool(torch.isfinite(near_tensor)) or near <= 0:
        raise ValueError("near must be finite and positive")

    means_cam = camera.world_to_cam(means)
    depth = means_cam[:, 2]

    # Keep the separate Camera.project call and arithmetic order used by both reference
    # rasterizers.  This matters for parity at float32.
    means2d, _ = camera.project(means)

    r_wc = camera.R.to(means)
    covariances_cam = r_wc @ covariances @ r_wc.T
    safe_depth = depth.clamp_min(near)
    jacobians = torch.zeros(
        means.shape[0],
        2,
        3,
        device=means.device,
        dtype=means.dtype,
    )
    jacobians[:, 0, 0] = camera.fx / safe_depth
    jacobians[:, 0, 2] = -camera.fx * means_cam[:, 0] / safe_depth.square()
    jacobians[:, 1, 1] = camera.fy / safe_depth
    jacobians[:, 1, 2] = -camera.fy * means_cam[:, 1] / safe_depth.square()
    covariances2d = jacobians @ covariances_cam @ jacobians.transpose(-1, -2)
    if dilation != 0:
        covariances2d = covariances2d + dilation_tensor * torch.eye(
            2,
            device=means.device,
            dtype=means.dtype,
        )
    return EWAProjection(
        means2d=means2d,
        depth=depth,
        covariances2d=covariances2d,
        means_cam=means_cam,
        jacobians=jacobians,
    )


def project_gaussians_ewa(
    gaussians: Gaussians3D,
    camera: Camera,
    *,
    dilation: float = EWA_DILATION,
    near: float = EWA_NEAR,
) -> EWAProjection:
    """Project a :class:`Gaussians3D` without visibility filtering or compositing."""
    return project_covariances_ewa(
        gaussians.means,
        gaussians.covariance(),
        camera,
        dilation=dilation,
        near=near,
    )


__all__ = [
    "EWA_DILATION",
    "EWA_NEAR",
    "EWAProjection",
    "project_covariances_ewa",
    "project_gaussians_ewa",
]
