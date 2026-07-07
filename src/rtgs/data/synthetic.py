"""Procedural ground-truthed scenes for tests and CPU benchmarks.

A random set of opaque-ish 3D gaussians inside the unit sphere, viewed by a ring of
inward-looking cameras, rendered (images + depth) with the reference rasterizer. Because
ground truth is a gaussian set, the whole pipeline is measurable: lifting should recover
positions near GT, refinement should push rendered PSNR up.
"""

from __future__ import annotations

import math

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.sh import num_sh_bases, rgb_to_sh
from rtgs.data.scene import SceneData
from rtgs.render.torch_ref import TorchRasterizer


def make_gt_gaussians(
    n: int = 40,
    seed: int = 0,
    radius: float = 0.6,
    scale_range: tuple[float, float] = (0.04, 0.12),
    sh_degree: int = 0,
) -> Gaussians3D:
    """Random anisotropic gaussians inside a sphere of the given radius."""
    gen = torch.Generator().manual_seed(seed)
    means = torch.randn(n, 3, generator=gen)
    means = (
        means
        / means.norm(dim=-1, keepdim=True)
        * (torch.rand(n, 1, generator=gen) ** (1 / 3) * radius)
    )
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=gen), dim=-1)
    lo, hi = scale_range
    scales = lo + (hi - lo) * torch.rand(n, 3, generator=gen)
    opacity = 0.75 + 0.2 * torch.rand(n, generator=gen)
    colors = 0.15 + 0.7 * torch.rand(n, 3, generator=gen)
    sh = torch.zeros(n, num_sh_bases(sh_degree), 3)
    sh[:, 0] = rgb_to_sh(colors)
    return Gaussians3D(means=means, quats=quats, log_scales=scales.log(), opacity=opacity, sh=sh)


def make_ring_cameras(
    n_cameras: int = 12,
    distance: float = 2.4,
    height: float = 0.6,
    image_size: int = 48,
    fov_x_deg: float = 45.0,
) -> list[Camera]:
    """Cameras on a ring around the origin, looking at the center."""
    cams = []
    for i in range(n_cameras):
        angle = 2 * math.pi * i / n_cameras
        eye = torch.tensor(
            [distance * math.cos(angle), height * math.sin(2 * angle), distance * math.sin(angle)]
        )
        cams.append(
            Camera.look_at(
                eye, torch.zeros(3), fov_x_deg=fov_x_deg, width=image_size, height=image_size
            )
        )
    return cams


def make_synthetic_scene(
    n_gaussians: int = 40,
    n_cameras: int = 12,
    image_size: int = 48,
    seed: int = 0,
    n_sparse_points: int = 200,
) -> SceneData:
    """Build a fully ground-truthed scene: images, depth, sparse points, GT gaussians.

    Sparse points are gaussian-mean samples (a stand-in for SfM points) used by depth
    alignment and the `sfm` baseline lifter.
    """
    gt = make_gt_gaussians(n=n_gaussians, seed=seed)
    cams = make_ring_cameras(n_cameras=n_cameras, image_size=image_size)
    rasterizer = TorchRasterizer()

    images, depths = [], []
    with torch.no_grad():
        for cam in cams:
            out = rasterizer.render(gt, cam)
            images.append(out.color.clamp(0, 1))
            # Expected depth where something was hit; 0 elsewhere (treated as invalid).
            depth = torch.where(out.alpha > 0.05, out.depth / out.alpha.clamp_min(1e-6), 0.0)
            depths.append(depth)

    gen = torch.Generator().manual_seed(seed + 1)
    idx = torch.randint(0, gt.n, (n_sparse_points,), generator=gen)
    jitter = 0.5 * gt.scales[idx] * torch.randn(n_sparse_points, 3, generator=gen)
    points = gt.means[idx] + jitter

    return SceneData(
        images=images,
        cameras=cams,
        points=points,
        gt_depths=depths,
        gt_gaussians=gt,
        name=f"synthetic_g{n_gaussians}_c{n_cameras}_s{image_size}",
    )
