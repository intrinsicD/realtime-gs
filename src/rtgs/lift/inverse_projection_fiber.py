"""Exact source-projection fibers for multi-view Gaussian geometry fitting.

A single 2D Gaussian does not invert to a unique 3D Gaussian.  Its center defines a camera
ray and its covariance fixes three linear combinations of the six entries of a 3D covariance.
This module parameterizes exactly the remaining inverse-projection fiber: one depth coordinate
for the mean and three null coordinates for the covariance.

The implementation is CPU-first and independent of RGB, opacity, correspondence weights, and
topology control.  Those mechanisms can consume the geometry, but must not be conflated with
the source-projection invariant.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.render.projection import (
    EWA_DILATION,
    EWAProjection,
    project_covariances_ewa,
)


def _validate_spd(name: str, matrices: torch.Tensor, *, tolerance: float = 0.0) -> None:
    if matrices.ndim != 3 or matrices.shape[-2:] not in {(2, 2), (3, 3)}:
        raise ValueError(f"{name} must have shape (N,2,2) or (N,3,3)")
    if not matrices.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(matrices).all()):
        raise ValueError(f"{name} must be finite")
    symmetry_error = (matrices - matrices.transpose(-1, -2)).abs().amax()
    if float(symmetry_error.detach()) > 1e-8:
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = torch.linalg.eigvalsh(matrices)
    if bool((eigenvalues <= tolerance).any()):
        minimum = float(eigenvalues.min().detach())
        raise ValueError(f"{name} must be positive definite (minimum eigenvalue {minimum:.6g})")


def _camera_ray_geometry(
    camera: Camera,
    means2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return camera-depth rays and stable orthonormal camera-space bases."""
    ray = torch.stack(
        [
            (means2d[:, 0] - camera.cx) / camera.fx,
            (means2d[:, 1] - camera.cy) / camera.fy,
            torch.ones_like(means2d[:, 0]),
        ],
        dim=-1,
    )
    normal = torch.nn.functional.normalize(ray, dim=-1)

    # Choose the coordinate axis least aligned with each ray.  This remains well-conditioned
    # for arbitrary fields of view, unlike always projecting the x axis.
    axis_index = normal.abs().argmin(dim=-1)
    seed = torch.nn.functional.one_hot(axis_index, num_classes=3).to(means2d)
    tangent0 = seed - normal * (seed * normal).sum(dim=-1, keepdim=True)
    tangent0 = torch.nn.functional.normalize(tangent0, dim=-1)
    tangent1 = torch.linalg.cross(normal, tangent0, dim=-1)
    basis = torch.stack([tangent0, tangent1, normal], dim=-1)
    return ray, basis


def _camera_to_world_exact(
    camera: Camera,
    points_cam: torch.Tensor,
) -> torch.Tensor:
    """Invert the stored world-to-camera matrix, including float round-off.

    ``Camera.cam_to_world`` uses ``R.T`` because calibrated rotations are intended to be
    orthonormal.  Here exact source reprojection is an invariant, so solving with the stored
    matrix avoids amplifying its float32 orthogonality round-off.
    """
    rotation = camera.R.to(points_cam)
    inverse = torch.linalg.inv(rotation)
    return (points_cam - camera.t.to(points_cam)) @ inverse.T


def _covariance_camera_to_world_exact(
    camera: Camera,
    covariance_cam: torch.Tensor,
) -> torch.Tensor:
    rotation = camera.R.to(covariance_cam)
    inverse = torch.linalg.inv(rotation)
    return inverse @ covariance_cam @ inverse.T


class InverseProjectionFiber(nn.Module):
    """Trainable 3D geometry that exactly preserves one source 2D Gaussian per row."""

    def __init__(
        self,
        *,
        cameras: Sequence[Camera],
        source_view_indices: torch.Tensor,
        source_component_indices: torch.Tensor,
        source_means2d: torch.Tensor,
        source_covariances2d: torch.Tensor,
        initial_depths: torch.Tensor,
        depth_lower: torch.Tensor | float,
        depth_upper: torch.Tensor | float,
        dilation: float = EWA_DILATION,
    ) -> None:
        super().__init__()
        if not cameras:
            raise ValueError("at least one camera is required")
        if source_means2d.ndim != 2 or source_means2d.shape[1] != 2:
            raise ValueError("source_means2d must have shape (N,2)")
        count = source_means2d.shape[0]
        if source_covariances2d.shape != (count, 2, 2):
            raise ValueError("source_covariances2d must have shape (N,2,2)")
        for name, value in (
            ("source_view_indices", source_view_indices),
            ("source_component_indices", source_component_indices),
        ):
            if value.shape != (count,) or value.dtype != torch.long:
                raise ValueError(f"{name} must have shape (N,) and dtype long")
        if source_view_indices.device != source_means2d.device:
            raise ValueError("source indices and geometry must share a device")
        if source_component_indices.device != source_means2d.device:
            raise ValueError("source indices and geometry must share a device")
        if not source_means2d.is_floating_point():
            raise TypeError("source geometry must be floating point")
        if (
            source_covariances2d.device != source_means2d.device
            or source_covariances2d.dtype != source_means2d.dtype
        ):
            raise ValueError("source means and covariances must share device and dtype")
        if not bool(torch.isfinite(source_means2d).all()):
            raise ValueError("source_means2d must be finite")
        _validate_spd("source_covariances2d", source_covariances2d)
        if count == 0:
            raise ValueError("at least one source Gaussian is required")
        if int(source_view_indices.min()) < 0 or int(source_view_indices.max()) >= len(cameras):
            raise ValueError("source_view_indices contains an unavailable camera")
        if bool((source_component_indices < 0).any()):
            raise ValueError("source_component_indices must be non-negative")
        if initial_depths.shape != (count,):
            raise ValueError("initial_depths must have shape (N,)")
        if initial_depths.device != source_means2d.device:
            raise ValueError("initial_depths and source geometry must share a device")

        lower = torch.as_tensor(
            depth_lower,
            dtype=source_means2d.dtype,
            device=source_means2d.device,
        ).expand(count)
        upper = torch.as_tensor(
            depth_upper,
            dtype=source_means2d.dtype,
            device=source_means2d.device,
        ).expand(count)
        if not bool(torch.isfinite(lower).all()) or not bool(torch.isfinite(upper).all()):
            raise ValueError("depth bounds must be finite")
        if bool((lower <= 0).any()) or bool((upper <= lower).any()):
            raise ValueError("depth bounds must satisfy 0 < lower < upper")
        if not bool(torch.isfinite(initial_depths).all()):
            raise ValueError("initial_depths must be finite")
        if bool((initial_depths <= lower).any()) or bool((initial_depths >= upper).any()):
            raise ValueError("initial_depths must lie strictly inside the depth bounds")
        dilation_value = source_means2d.new_tensor(dilation)
        if not bool(torch.isfinite(dilation_value)) or dilation < 0:
            raise ValueError("dilation must be finite and non-negative")
        intrinsic = source_covariances2d - dilation_value * torch.eye(
            2,
            dtype=source_means2d.dtype,
            device=source_means2d.device,
        )
        _validate_spd("source_covariances2d - dilation*I", intrinsic)

        fraction = (initial_depths - lower) / (upper - lower)
        depth_logits = torch.logit(fraction)

        self.cameras = tuple(cameras)
        self.dilation = float(dilation)
        self.register_buffer("source_view_indices", source_view_indices.detach().clone())
        self.register_buffer("source_component_indices", source_component_indices.detach().clone())
        self.register_buffer("source_means2d", source_means2d.detach().clone())
        self.register_buffer("source_covariances2d", source_covariances2d.detach().clone())
        self.register_buffer("intrinsic_covariances2d", intrinsic.detach().clone())
        self.register_buffer("depth_lower", lower.detach().clone())
        self.register_buffer("depth_upper", upper.detach().clone())
        self.depth_logits = nn.Parameter(depth_logits.detach().clone())
        self.cross = nn.Parameter(source_means2d.new_zeros((count, 2)))

        # Initialize the ray standard deviation to the geometric-mean tangent scale at the
        # initial depth.  This is neutral with respect to either source ellipse axis.
        with torch.no_grad():
            tangent_blocks = self._tangent_covariances(initial_depths)
            log_ray_scale = 0.25 * torch.linalg.slogdet(tangent_blocks).logabsdet
        self.log_ray_scale = nn.Parameter(log_ray_scale.detach().clone())

    @property
    def n(self) -> int:
        return int(self.source_means2d.shape[0])

    def subset(self, indices: torch.Tensor) -> InverseProjectionFiber:
        """Copy selected fibers while preserving their exact source anchors and state."""

        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("indices must be a one-dimensional long tensor")
        if indices.device != self.source_means2d.device:
            raise ValueError("indices and fiber state must share a device")
        if indices.numel() == 0:
            raise ValueError("indices must not be empty")
        if int(indices.min()) < 0 or int(indices.max()) >= self.n:
            raise IndexError("indices contain an unavailable fiber")
        if int(torch.unique(indices).numel()) != int(indices.numel()):
            raise ValueError("indices must be unique")

        child = InverseProjectionFiber(
            cameras=self.cameras,
            source_view_indices=self.source_view_indices[indices],
            source_component_indices=self.source_component_indices[indices],
            source_means2d=self.source_means2d[indices],
            source_covariances2d=self.source_covariances2d[indices],
            initial_depths=self.depths().detach()[indices],
            depth_lower=self.depth_lower[indices],
            depth_upper=self.depth_upper[indices],
            dilation=self.dilation,
        )
        with torch.no_grad():
            child.depth_logits.copy_(self.depth_logits[indices])
            child.cross.copy_(self.cross[indices])
            child.log_ray_scale.copy_(self.log_ray_scale[indices])
        return child

    def depths(self) -> torch.Tensor:
        fraction = torch.sigmoid(self.depth_logits)
        return self.depth_lower + fraction * (self.depth_upper - self.depth_lower)

    def _tangent_covariances(self, depths: torch.Tensor) -> torch.Tensor:
        result = self.source_covariances2d.new_empty((self.n, 2, 2))
        for view_index, camera in enumerate(self.cameras):
            row_indices = (self.source_view_indices == view_index).nonzero(as_tuple=True)[0]
            if row_indices.numel() == 0:
                continue
            means2d = self.source_means2d[row_indices]
            depth = depths[row_indices]
            ray, basis = _camera_ray_geometry(camera, means2d)
            points_cam = depth[:, None] * ray
            jacobian = means2d.new_zeros((row_indices.numel(), 2, 3))
            jacobian[:, 0, 0] = camera.fx / depth
            jacobian[:, 0, 2] = -camera.fx * points_cam[:, 0] / depth.square()
            jacobian[:, 1, 1] = camera.fy / depth
            jacobian[:, 1, 2] = -camera.fy * points_cam[:, 1] / depth.square()
            tangent_projection = jacobian @ basis[:, :, :2]
            inverse_projection = torch.linalg.inv(tangent_projection)
            target = self.intrinsic_covariances2d[row_indices]
            tangent = inverse_projection @ target @ inverse_projection.transpose(-1, -2)
            result[row_indices] = 0.5 * (tangent + tangent.transpose(-1, -2))
        return result

    def means_covariances(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build differentiable world means and SPD covariances."""
        depths = self.depths()
        tangent_covariances = self._tangent_covariances(depths)
        means = self.source_means2d.new_empty((self.n, 3))
        covariances = self.source_means2d.new_empty((self.n, 3, 3))

        for view_index, camera in enumerate(self.cameras):
            row_indices = (self.source_view_indices == view_index).nonzero(as_tuple=True)[0]
            if row_indices.numel() == 0:
                continue
            means2d = self.source_means2d[row_indices]
            depth = depths[row_indices]
            ray, basis = _camera_ray_geometry(camera, means2d)
            points_cam = depth[:, None] * ray
            means[row_indices] = _camera_to_world_exact(camera, points_cam)

            tangent = tangent_covariances[row_indices]
            cross_coordinate = self.cross[row_indices]
            tangent_cross = tangent @ cross_coordinate.unsqueeze(-1)
            ray_variance = (
                cross_coordinate.unsqueeze(1) @ tangent @ cross_coordinate.unsqueeze(-1)
            ).squeeze(-1).squeeze(-1) + (2.0 * self.log_ray_scale[row_indices]).exp()
            covariance_basis = means2d.new_zeros((row_indices.numel(), 3, 3))
            covariance_basis[:, :2, :2] = tangent
            covariance_basis[:, :2, 2] = tangent_cross.squeeze(-1)
            covariance_basis[:, 2, :2] = tangent_cross.squeeze(-1)
            covariance_basis[:, 2, 2] = ray_variance
            covariance_cam = basis @ covariance_basis @ basis.transpose(-1, -2)
            covariances[row_indices] = _covariance_camera_to_world_exact(camera, covariance_cam)

        return means, 0.5 * (covariances + covariances.transpose(-1, -2))

    def project(self, camera: Camera) -> EWAProjection:
        means, covariances = self.means_covariances()
        return project_covariances_ewa(
            means,
            covariances,
            camera,
            dilation=self.dilation,
        )

    def source_projection(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project each row through its own source camera in original row order."""
        means, covariances = self.means_covariances()
        means2d = self.source_means2d.new_empty((self.n, 2))
        covariance2d = self.source_covariances2d.new_empty((self.n, 2, 2))
        depth = self.source_means2d.new_empty((self.n,))
        for view_index, camera in enumerate(self.cameras):
            row_indices = (self.source_view_indices == view_index).nonzero(as_tuple=True)[0]
            if row_indices.numel() == 0:
                continue
            projected = project_covariances_ewa(
                means[row_indices],
                covariances[row_indices],
                camera,
                dilation=self.dilation,
            )
            means2d[row_indices] = projected.means2d
            covariance2d[row_indices] = projected.covariances2d
            depth[row_indices] = projected.depth
        return means2d, covariance2d, depth

    def as_gaussians(
        self,
        *,
        colors: torch.Tensor | None = None,
        opacity: torch.Tensor | float = 0.5,
    ) -> Gaussians3D:
        """Materialize the fiber geometry as degree-zero 3DGS parameters."""
        means, covariances = self.means_covariances()
        if colors is None:
            colors = means.new_full((self.n, 3), 0.5)
        if colors.shape != (self.n, 3):
            raise ValueError("colors must have shape (N,3)")
        colors = colors.to(means)
        opacity_tensor = torch.as_tensor(opacity, dtype=means.dtype, device=means.device).expand(
            self.n
        )
        if not bool(torch.isfinite(colors).all()) or not bool(torch.isfinite(opacity_tensor).all()):
            raise ValueError("colors and opacity must be finite")
        if bool((opacity_tensor < 0).any()) or bool((opacity_tensor > 1).any()):
            raise ValueError("opacity must be in [0,1]")
        return Gaussians3D.from_means_covs(
            means,
            covariances,
            colors,
            opacity_tensor,
        )


class FreeGaussianGeometry(nn.Module):
    """Free mean and Cholesky-SPD geometry control initialized from a fiber state."""

    def __init__(self, means: torch.Tensor, covariances: torch.Tensor) -> None:
        super().__init__()
        if means.ndim != 2 or means.shape[1] != 3:
            raise ValueError("means must have shape (N,3)")
        if covariances.shape != (means.shape[0], 3, 3):
            raise ValueError("covariances must have shape (N,3,3)")
        if not means.is_floating_point() or not covariances.is_floating_point():
            raise TypeError("means and covariances must be floating point")
        if means.device != covariances.device or means.dtype != covariances.dtype:
            raise ValueError("means and covariances must share device and dtype")
        if not bool(torch.isfinite(means).all()):
            raise ValueError("means must be finite")
        _validate_spd("covariances", covariances)
        cholesky = torch.linalg.cholesky(covariances)
        self.means = nn.Parameter(means.detach().clone())
        self.log_diagonal = nn.Parameter(
            torch.diagonal(cholesky, dim1=-2, dim2=-1).log().detach().clone()
        )
        self.lower = nn.Parameter(
            torch.stack(
                [
                    cholesky[:, 1, 0],
                    cholesky[:, 2, 0],
                    cholesky[:, 2, 1],
                ],
                dim=-1,
            )
            .detach()
            .clone()
        )

    @property
    def n(self) -> int:
        return int(self.means.shape[0])

    def covariances(self) -> torch.Tensor:
        cholesky = self.means.new_zeros((self.n, 3, 3))
        diagonal = self.log_diagonal.exp()
        cholesky[:, 0, 0] = diagonal[:, 0]
        cholesky[:, 1, 1] = diagonal[:, 1]
        cholesky[:, 2, 2] = diagonal[:, 2]
        cholesky[:, 1, 0] = self.lower[:, 0]
        cholesky[:, 2, 0] = self.lower[:, 1]
        cholesky[:, 2, 1] = self.lower[:, 2]
        return cholesky @ cholesky.transpose(-1, -2)

    def project(self, camera: Camera, *, dilation: float = EWA_DILATION) -> EWAProjection:
        return project_covariances_ewa(
            self.means,
            self.covariances(),
            camera,
            dilation=dilation,
        )

    def as_gaussians(
        self,
        *,
        colors: torch.Tensor | None = None,
        opacity: torch.Tensor | float = 0.5,
    ) -> Gaussians3D:
        if colors is None:
            colors = self.means.new_full((self.n, 3), 0.5)
        if colors.shape != (self.n, 3):
            raise ValueError("colors must have shape (N,3)")
        colors = colors.to(self.means)
        opacity_tensor = torch.as_tensor(
            opacity,
            dtype=self.means.dtype,
            device=self.means.device,
        ).expand(self.n)
        if not bool(torch.isfinite(colors).all()) or not bool(torch.isfinite(opacity_tensor).all()):
            raise ValueError("colors and opacity must be finite")
        if bool((opacity_tensor < 0).any()) or bool((opacity_tensor > 1).any()):
            raise ValueError("opacity must be in [0,1]")
        return Gaussians3D.from_means_covs(
            self.means,
            self.covariances(),
            colors,
            opacity_tensor,
        )


def pairwise_center_cost(
    predicted_means2d: torch.Tensor,
    predicted_covariances2d: torch.Tensor,
    target_means2d: torch.Tensor,
    target_covariances2d: torch.Tensor,
) -> torch.Tensor:
    """Symmetric Mahalanobis center cost with covariance gradients stopped."""
    if predicted_means2d.ndim != 2 or predicted_means2d.shape[1] != 2:
        raise ValueError("predicted_means2d must have shape (N,2)")
    if target_means2d.ndim != 2 or target_means2d.shape[1] != 2:
        raise ValueError("target_means2d must have shape (M,2)")
    if predicted_covariances2d.shape != (predicted_means2d.shape[0], 2, 2):
        raise ValueError("predicted_covariances2d must have shape (N,2,2)")
    if target_covariances2d.shape != (target_means2d.shape[0], 2, 2):
        raise ValueError("target_covariances2d must have shape (M,2,2)")
    _validate_spd("predicted_covariances2d", predicted_covariances2d)
    _validate_spd("target_covariances2d", target_covariances2d)
    delta = predicted_means2d[:, None, :] - target_means2d[None, :, :]
    metric = 0.5 * (predicted_covariances2d[:, None, :, :] + target_covariances2d[None, :, :, :])
    solved = torch.linalg.solve(metric.detach(), delta.unsqueeze(-1))
    return (delta.unsqueeze(-2) @ solved).squeeze(-1).squeeze(-1)


def spd_affine_invariant_squared(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Squared affine-invariant distance between broadcast-compatible SPD matrices."""
    if first.shape[-2:] != second.shape[-2:] or first.shape[-1] not in {2, 3}:
        raise ValueError("SPD inputs must end in matching 2x2 or 3x3 matrices")
    dimension = first.shape[-1]
    broadcast_shape = torch.broadcast_shapes(first.shape[:-2], second.shape[:-2])
    first_b = first.expand(*broadcast_shape, dimension, dimension)
    second_b = second.expand(*broadcast_shape, dimension, dimension)
    first_eigenvalues, first_eigenvectors = torch.linalg.eigh(first_b)
    if bool((first_eigenvalues <= 0).any()):
        raise ValueError("first input must be SPD")
    inverse_sqrt = (
        first_eigenvectors * first_eigenvalues.rsqrt().unsqueeze(-2)
    ) @ first_eigenvectors.transpose(-1, -2)
    relative = inverse_sqrt @ second_b @ inverse_sqrt
    relative = 0.5 * (relative + relative.transpose(-1, -2))
    relative_eigenvalues = torch.linalg.eigvalsh(relative)
    if bool((relative_eigenvalues <= 0).any()):
        raise ValueError("second input must be SPD")
    return relative_eigenvalues.log().square().sum(dim=-1)


def pairwise_conic_cost(
    predicted_covariances2d: torch.Tensor,
    target_covariances2d: torch.Tensor,
) -> torch.Tensor:
    """All-pairs squared affine-invariant covariance distances."""
    if predicted_covariances2d.ndim != 3 or predicted_covariances2d.shape[-2:] != (2, 2):
        raise ValueError("predicted_covariances2d must have shape (N,2,2)")
    if target_covariances2d.ndim != 3 or target_covariances2d.shape[-2:] != (2, 2):
        raise ValueError("target_covariances2d must have shape (M,2,2)")
    return spd_affine_invariant_squared(
        predicted_covariances2d[:, None, :, :],
        target_covariances2d[None, :, :, :],
    )


def pairwise_gaussian_geometry_cost(
    predicted_means2d: torch.Tensor,
    predicted_covariances2d: torch.Tensor,
    target_means2d: torch.Tensor,
    target_covariances2d: torch.Tensor,
    *,
    include_conic: bool,
    conic_weight: float = 0.25,
) -> torch.Tensor:
    """Pairwise center or center-plus-conic correspondence cost."""
    center = pairwise_center_cost(
        predicted_means2d,
        predicted_covariances2d,
        target_means2d,
        target_covariances2d,
    )
    if not include_conic:
        return center
    if not torch.isfinite(center.new_tensor(conic_weight)) or conic_weight < 0:
        raise ValueError("conic_weight must be finite and non-negative")
    conic = pairwise_conic_cost(predicted_covariances2d, target_covariances2d)
    return center + conic_weight * conic


def hard_correspondence_loss(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One-way hard set matching, invariant to co-located duplicate targets."""
    if cost.ndim != 2 or cost.shape[0] == 0 or cost.shape[1] == 0:
        raise ValueError("cost must be a non-empty matrix")
    if not bool(torch.isfinite(cost).all()):
        raise ValueError("cost must be finite")
    minimum, assignment = cost.min(dim=1)
    return minimum.mean(), assignment


def covariance_projection_design(
    means: torch.Tensor,
    cameras: Sequence[Camera],
) -> torch.Tensor:
    """Linear design from world symmetric-covariance entries to projected entries.

    Rows are grouped by camera and ordered ``(00, 01, 11)``. Columns represent
    ``(xx, yy, zz, xy, xz, yz)`` where off-diagonal bases populate both symmetric entries.
    A generic two-view design has rank five; three generic views have rank six.
    """
    if means.shape != (len(cameras), 3):
        raise ValueError("means must have shape (len(cameras),3)")
    if not means.is_floating_point() or not bool(torch.isfinite(means).all()):
        raise ValueError("means must be finite floating point")
    bases = means.new_zeros((6, 3, 3))
    bases[0, 0, 0] = 1
    bases[1, 1, 1] = 1
    bases[2, 2, 2] = 1
    bases[3, 0, 1] = bases[3, 1, 0] = 1
    bases[4, 0, 2] = bases[4, 2, 0] = 1
    bases[5, 1, 2] = bases[5, 2, 1] = 1

    rows: list[torch.Tensor] = []
    for mean, camera in zip(means, cameras, strict=True):
        mean_batch = mean[None, :]
        means_cam = camera.world_to_cam(mean_batch)
        depth = means_cam[:, 2]
        if float(depth) <= 0:
            raise ValueError("design mean must be in front of every camera")
        jacobian = means.new_zeros((2, 3))
        jacobian[0, 0] = camera.fx / depth
        jacobian[0, 2] = -camera.fx * means_cam[0, 0] / depth.square()
        jacobian[1, 1] = camera.fy / depth
        jacobian[1, 2] = -camera.fy * means_cam[0, 1] / depth.square()
        world_jacobian = jacobian @ camera.R.to(means)
        projected = world_jacobian[None, :, :] @ bases @ world_jacobian.T[None, :, :]
        rows.append(
            torch.stack(
                [
                    projected[:, 0, 0],
                    projected[:, 0, 1],
                    projected[:, 1, 1],
                ],
                dim=0,
            )
        )
    return torch.cat(rows, dim=0)


__all__ = [
    "FreeGaussianGeometry",
    "InverseProjectionFiber",
    "covariance_projection_design",
    "hard_correspondence_loss",
    "pairwise_center_cost",
    "pairwise_conic_cost",
    "pairwise_gaussian_geometry_cost",
    "spd_affine_invariant_squared",
]
