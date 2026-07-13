"""Baseline lifters for comparison: classic SfM-point init and random init.

These ignore the 2D gaussian fits (they take them for interface compatibility) and
reproduce the standard 3DGS starting points, so benchmarks can quantify what the
2D-gaussian lifting actually buys.
"""

from __future__ import annotations

import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.scene import SceneData
from rtgs.lift.base import bilinear_sample


def _knn_scale(points: torch.Tensor, k: int = 3) -> torch.Tensor:
    """3DGS-style isotropic scale: mean distance to the k nearest neighbors."""
    d = torch.cdist(points, points)
    d.fill_diagonal_(float("inf"))
    knn = d.topk(min(k, points.shape[0] - 1), largest=False).values
    return knn.mean(dim=-1).clamp_min(1e-4)


def _colors_from_views(points: torch.Tensor, scene: SceneData) -> torch.Tensor:
    """Sample each point's color from the first view it projects into (gray fallback)."""
    colors = torch.full((points.shape[0], 3), 0.5, device=points.device)
    remaining = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
    for image, cam in zip(scene.images, scene.cameras):
        if not bool(remaining.any()):
            break
        uv, z = cam.project(points)
        hit = remaining & (z > 0.05) & cam.in_image(uv, margin=-0.5)
        if bool(hit.any()):
            colors[hit] = bilinear_sample(image, uv[hit])
            remaining &= ~hit
    return colors


def _isotropic(
    points: torch.Tensor, scales: torch.Tensor, colors: torch.Tensor, opacity: float
) -> Gaussians3D:
    n = points.shape[0]
    quats = torch.zeros(n, 4, device=points.device, dtype=points.dtype)
    quats[:, 0] = 1.0
    covs_scales = scales[:, None].expand(n, 3)
    from rtgs.core.sh import rgb_to_sh

    sh = torch.zeros(n, 1, 3, device=points.device, dtype=points.dtype)
    sh[:, 0] = rgb_to_sh(colors)
    return Gaussians3D(
        means=points,
        quats=quats,
        log_scales=covs_scales.log(),
        opacity=torch.full((n,), opacity, device=points.device, dtype=points.dtype),
        sh=sh,
    )


class SfMLifter:
    """Classic 3DGS initialization from the sparse point cloud (the baseline to beat)."""

    def __init__(self, opacity: float = 0.1):
        self.opacity = opacity

    def lift(self, gaussians2d: list[Gaussians2D], scene: SceneData) -> Gaussians3D:
        """Isotropic gaussians at the sparse points, k-NN scales, view-sampled colors."""
        if scene.points is None or scene.points.shape[0] < 4:
            raise ValueError("sfm baseline requires scene.points")
        pts = scene.points
        return _isotropic(pts, _knn_scale(pts), _colors_from_views(pts, scene), self.opacity)


class RandomLifter:
    """Uniform random init inside the scene bounds (the lower-bound baseline)."""

    def __init__(self, n: int = 2000, opacity: float = 0.1, seed: int = 0):
        self.n = n
        self.opacity = opacity
        self.seed = seed

    def lift(self, gaussians2d: list[Gaussians2D], scene: SceneData) -> Gaussians3D:
        """Random points in a sphere of half the scene extent around the center."""
        center, extent = scene.center_and_extent()
        device = center.device
        gen = torch.Generator(device=device).manual_seed(self.seed)
        v = torch.randn(self.n, 3, generator=gen, device=device)
        v = v / v.norm(dim=-1, keepdim=True)
        r = 0.5 * extent * torch.rand(self.n, 1, generator=gen, device=device) ** (1 / 3)
        pts = center + v * r
        scales = torch.full(
            (self.n,),
            0.5 * extent / max(self.n, 1) ** (1 / 3),
            device=device,
        )
        colors = torch.full((self.n, 3), 0.5, device=device)
        return _isotropic(pts, scales, colors, self.opacity)
