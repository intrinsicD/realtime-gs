"""Lifter protocol and shared 2D-to-3D lifting geometry.

The central primitive: a 2D image gaussian (center u, covariance S2 in px^2, color,
weight) plus a depth z becomes a 3D gaussian with
  - mean: unproject(u, z)
  - lateral covariance: A S2 A^T with A = diag(z/fx, z/fy)  (back-projected footprint)
  - along-ray variance sigma_ray^2: supplied by the variant (depth-gradient formula,
    sample spread, or isotropic fallback)
assembled in an orthonormal basis (e1, e2, ray) and rotated to world space. See
docs/RESEARCH.md §"Missing-dimension covariance".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.scene import SceneData

if TYPE_CHECKING:
    from rtgs.data.reconstruction_inputs import ReconstructionInputs
    from rtgs.lift.compact_carve import CompactInitializationResult


class Lifter(Protocol):
    """Turns per-view 2D gaussians into one 3D gaussian set."""

    def lift(self, gaussians2d: list[Gaussians2D], scene: SceneData) -> Gaussians3D:
        """Lift all views into a single world-space gaussian set."""
        ...


class CompactInitializer(Protocol):
    """Pluggable initializer over serialized RGB-free reconstruction inputs."""

    def initialize(self, inputs: ReconstructionInputs) -> CompactInitializationResult:
        """Create an independently budgeted 3D initialization."""
        ...


def bilinear_sample(grid: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample an (H, W) or (H, W, C) map at (N, 2) pixel coordinates.

    Pixel-center convention: value of pixel (i, j) lives at uv = (j + 0.5, i + 0.5).
    Coordinates are clamped to the valid interior.
    """
    h, w = grid.shape[0], grid.shape[1]
    x = (uv[:, 0] - 0.5).clamp(0, w - 1)
    y = (uv[:, 1] - 0.5).clamp(0, h - 1)
    x0 = x.floor().long()
    y0 = y.floor().long()
    x1 = (x0 + 1).clamp_max(w - 1)
    y1 = (y0 + 1).clamp_max(h - 1)
    fx = (x - x0.float()).unsqueeze(-1) if grid.ndim == 3 else x - x0.float()
    fy = (y - y0.float()).unsqueeze(-1) if grid.ndim == 3 else y - y0.float()
    v00, v01 = grid[y0, x0], grid[y0, x1]
    v10, v11 = grid[y1, x0], grid[y1, x1]
    return v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) + v10 * (1 - fx) * fy + v11 * fx * fy


def ray_basis(camera: Camera, uv: torch.Tensor) -> torch.Tensor:
    """(N,3,3) orthonormal camera-space bases whose columns are (e1, e2, ray).

    e1/e2 span the plane perpendicular to the pixel ray, with e1 aligned to the image
    x-axis as closely as possible (so 2D covariance axes map onto e1/e2).
    """
    d = torch.stack(
        [
            (uv[:, 0] - camera.cx) / camera.fx,
            (uv[:, 1] - camera.cy) / camera.fy,
            torch.ones_like(uv[:, 0]),
        ],
        dim=-1,
    )
    ray = torch.nn.functional.normalize(d, dim=-1)
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=uv.device).expand_as(ray)
    e1 = torch.nn.functional.normalize(x_axis - ray * (x_axis * ray).sum(-1, keepdim=True), dim=-1)
    e2 = torch.linalg.cross(ray, e1)
    return torch.stack([e1, e2, ray], dim=-1)  # columns


def lift_covariance(
    camera: Camera,
    uv: torch.Tensor,
    cov2d: torch.Tensor,
    depth: torch.Tensor,
    sigma_ray: torch.Tensor,
) -> torch.Tensor:
    """Lift (N,2,2) pixel covariances at (N,) depths into (N,3,3) world covariances."""
    # Work in the plane perpendicular to the ray, but solve for the tangent covariance through
    # the exact local projection Jacobian. Simply applying diag(z/f) and then rotating that plane
    # is only correct on the principal axis and inflates footprints near image borders.
    xn = (uv[:, 0] - camera.cx) / camera.fx
    yn = (uv[:, 1] - camera.cy) / camera.fy
    points_cam = torch.stack([xn * depth, yn * depth, depth], dim=-1)
    z = depth.clamp_min(1e-8)
    n = uv.shape[0]
    jac = torch.zeros(n, 2, 3, device=uv.device, dtype=uv.dtype)
    jac[:, 0, 0] = camera.fx / z
    jac[:, 0, 2] = -camera.fx * points_cam[:, 0] / z.square()
    jac[:, 1, 1] = camera.fy / z
    jac[:, 1, 2] = -camera.fy * points_cam[:, 1] / z.square()
    basis = ray_basis(camera, uv)
    tangent = basis[:, :, :2]
    projection_on_tangent = jac @ tangent
    inv_projection = torch.linalg.inv(projection_on_tangent)
    cov_tangent = inv_projection @ cov2d @ inv_projection.transpose(-1, -2)

    m = torch.zeros(n, 3, 3, device=uv.device, dtype=uv.dtype)
    m[:, :2, :2] = cov_tangent
    m[:, 2, 2] = sigma_ray**2
    cov_cam = basis @ m @ basis.transpose(-1, -2)
    r_c2w = camera.R.T.to(uv)
    return r_c2w @ cov_cam @ r_c2w.T


def surface_covariance_from_depth(
    camera: Camera,
    g2d: Gaussians2D,
    depth_map: torch.Tensor,
    depth_at_center: torch.Tensor,
    normal_thickness: float = 0.15,
    robust_depth_gradients: bool = True,
) -> torch.Tensor:
    """Lift image covariance through the local depth surface Jacobian.

    For ``X(u,v) = D(u,v) K^-1 [u,v,1]``, ``J Sigma_2D J^T`` is the covariance tangent to
    the reconstructed surface. A small normal variance makes it positive definite without
    turning surface slope into artificial ray thickness.
    """
    gx, gy = depth_map_gradients(depth_map, validity_aware=robust_depth_gradients)
    du = bilinear_sample(gx, g2d.xy)
    dv = bilinear_sample(gy, g2d.xy)
    q = torch.stack(
        [
            (g2d.xy[:, 0] - camera.cx) / camera.fx,
            (g2d.xy[:, 1] - camera.cy) / camera.fy,
            torch.ones_like(depth_at_center),
        ],
        dim=-1,
    )
    ex = torch.zeros_like(q)
    ey = torch.zeros_like(q)
    ex[:, 0] = depth_at_center / camera.fx
    ey[:, 1] = depth_at_center / camera.fy
    j_u = ex + q * du[:, None]
    j_v = ey + q * dv[:, None]
    jac_surface = torch.stack([j_u, j_v], dim=-1)  # (N,3,2)
    cov_cam = jac_surface @ g2d.covariance() @ jac_surface.transpose(-1, -2)

    normal = torch.nn.functional.normalize(torch.linalg.cross(j_u, j_v), dim=-1)
    evals2d = eigvals_2x2(g2d.covariance())
    pixel_minor = evals2d[:, 0].clamp_min(1e-8).sqrt()
    f_mean = 0.5 * (camera.fx + camera.fy)
    sigma_normal = normal_thickness * depth_at_center / f_mean * pixel_minor
    cov_cam = cov_cam + sigma_normal.square()[:, None, None] * (
        normal[:, :, None] * normal[:, None, :]
    )
    r_c2w = camera.R.T.to(g2d.xy)
    return r_c2w @ cov_cam @ r_c2w.T


def footprint_sigma_ray(
    camera: Camera,
    g2d: Gaussians2D,
    depth_map: torch.Tensor,
    depth_at_center: torch.Tensor,
    robust_depth_gradients: bool = True,
) -> torch.Tensor:
    """Along-ray std from the depth spread across each gaussian's footprint.

    sigma_ray^2 = grad(D)^T S2 grad(D) + (z/f)^2 * s_min^2, where grad(D) is the depth
    gradient at the center (slanted surfaces get elongated gaussians) and s_min is the
    2D minor-axis std (a one-footprint thickness floor). Clamped against the lateral
    extent for stability.
    """
    gx, gy = depth_map_gradients(depth_map, validity_aware=robust_depth_gradients)
    grad = torch.stack([bilinear_sample(gx, g2d.xy), bilinear_sample(gy, g2d.xy)], dim=-1)  # (N,2)
    cov = g2d.covariance()
    slant_var = (grad.unsqueeze(1) @ cov @ grad.unsqueeze(-1)).reshape(-1)

    evals = eigvals_2x2(cov)
    s_min = evals[:, 0].clamp_min(1e-8).sqrt()
    s_max = evals[:, 1].clamp_min(1e-8).sqrt()
    f_mean = 0.5 * (camera.fx + camera.fy)
    floor_var = (depth_at_center / f_mean * s_min) ** 2
    sigma = (slant_var + floor_var).clamp_min(1e-12).sqrt()

    lat_min = depth_at_center / f_mean * s_min
    lat_max = depth_at_center / f_mean * s_max
    return sigma.clamp(0.1 * lat_min, 3.0 * lat_max)


def depth_map_gradients(
    depth_map: torch.Tensor,
    validity_aware: bool = True,
    min_depth: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite-difference depth gradients, optionally avoiding invalid-depth boundaries.

    The legacy central difference treats a transition from a valid surface to zero/NaN
    background as a steep surface. The validity-aware form uses central differences only
    when both neighbors and the center are valid, falls back to a valid one-sided
    difference, and otherwise returns zero.
    """
    gx = torch.zeros_like(depth_map)
    gy = torch.zeros_like(depth_map)
    if not validity_aware:
        gx[:, 1:-1] = 0.5 * (depth_map[:, 2:] - depth_map[:, :-2])
        gy[1:-1, :] = 0.5 * (depth_map[2:, :] - depth_map[:-2, :])
        return gx, gy

    valid = torch.isfinite(depth_map) & (depth_map > min_depth)
    if depth_map.shape[1] > 1:
        pair = valid[:, 0] & valid[:, 1]
        gx[:, 0] = torch.where(pair, depth_map[:, 1] - depth_map[:, 0], 0.0)
        pair = valid[:, -1] & valid[:, -2]
        gx[:, -1] = torch.where(pair, depth_map[:, -1] - depth_map[:, -2], 0.0)
    if depth_map.shape[1] > 2:
        center = depth_map[:, 1:-1]
        center_valid = valid[:, 1:-1]
        left_valid = center_valid & valid[:, :-2]
        right_valid = center_valid & valid[:, 2:]
        central = 0.5 * (depth_map[:, 2:] - depth_map[:, :-2])
        forward = depth_map[:, 2:] - center
        backward = center - depth_map[:, :-2]
        gx[:, 1:-1] = torch.where(
            left_valid & right_valid,
            central,
            torch.where(right_valid, forward, torch.where(left_valid, backward, 0.0)),
        )
    if depth_map.shape[0] > 1:
        pair = valid[0, :] & valid[1, :]
        gy[0, :] = torch.where(pair, depth_map[1, :] - depth_map[0, :], 0.0)
        pair = valid[-1, :] & valid[-2, :]
        gy[-1, :] = torch.where(pair, depth_map[-1, :] - depth_map[-2, :], 0.0)
    if depth_map.shape[0] > 2:
        center = depth_map[1:-1, :]
        center_valid = valid[1:-1, :]
        upper_valid = center_valid & valid[:-2, :]
        lower_valid = center_valid & valid[2:, :]
        central = 0.5 * (depth_map[2:, :] - depth_map[:-2, :])
        forward = depth_map[2:, :] - center
        backward = center - depth_map[:-2, :]
        gy[1:-1, :] = torch.where(
            upper_valid & lower_valid,
            central,
            torch.where(lower_valid, forward, torch.where(upper_valid, backward, 0.0)),
        )
    return gx, gy


def minor_axis_sigma_ray(
    camera: Camera,
    g2d: Gaussians2D,
    depth_at_center: torch.Tensor,
) -> torch.Tensor:
    """Match ray thickness to each gaussian's minor lateral footprint.

    This is the no-gradient floor used by :func:`footprint_sigma_ray`. Unlike a
    globally isotropic control, it remains depth- and gaussian-size-dependent.
    """
    evals = eigvals_2x2(g2d.covariance())
    s_min = evals[:, 0].clamp_min(1e-8).sqrt()
    f_mean = 0.5 * (camera.fx + camera.fy)
    return depth_at_center / f_mean * s_min


def eigvals_2x2(cov: torch.Tensor) -> torch.Tensor:
    """(N,2) eigenvalues of symmetric (N,2,2) matrices, ascending, clamped >= 0."""
    a, b, c = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    mean = 0.5 * (a + c)
    disc = (0.25 * (a - c) ** 2 + b**2).clamp_min(0.0).sqrt()
    return torch.stack([(mean - disc).clamp_min(0.0), mean + disc], dim=-1)


def lift_view_at_depth(
    camera: Camera,
    g2d: Gaussians2D,
    depth: torch.Tensor,
    sigma_ray: torch.Tensor,
    sh_degree: int = 0,
    opacity: torch.Tensor | None = None,
) -> Gaussians3D:
    """Lift one view's gaussians to given per-gaussian depths (all inputs pre-filtered)."""
    means = camera.unproject(g2d.xy, depth)
    covs = lift_covariance(camera, g2d.xy, g2d.covariance(), depth, sigma_ray)
    return Gaussians3D.from_means_covs(
        means=means,
        covs=covs,
        colors=g2d.color.clamp(0.0, 1.0),
        opacity=(g2d.weight if opacity is None else opacity).clamp(0.02, 0.99),
        sh_degree=sh_degree,
    )


def lift_view_from_depth_map(
    camera: Camera,
    g2d: Gaussians2D,
    depth_map: torch.Tensor,
    depth: torch.Tensor,
    sh_degree: int = 0,
    opacity: torch.Tensor | None = None,
    normal_thickness: float = 0.15,
    robust_depth_gradients: bool = True,
) -> Gaussians3D:
    """Lift one view using a local depth surface for both means and covariances."""
    means = camera.unproject(g2d.xy, depth)
    covs = surface_covariance_from_depth(
        camera,
        g2d,
        depth_map,
        depth,
        normal_thickness=normal_thickness,
        robust_depth_gradients=robust_depth_gradients,
    )
    return Gaussians3D.from_means_covs(
        means=means,
        covs=covs,
        colors=g2d.color.clamp(0.0, 1.0),
        opacity=(g2d.weight if opacity is None else opacity).clamp(0.02, 0.99),
        sh_degree=sh_degree,
    )
