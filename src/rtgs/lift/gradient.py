"""Variant A: lift per-image 2D gaussians, then optimize them along their rays.

Implements the staged idea: each fitted 2D gaussian is lifted to a 3D gaussian whose
mean is pinned to the pixel's viewing ray (lateral position frozen — the "on the ray"
constraint) and whose along-ray extent starts at a *footprint-matched* thickness (the
"epsilon along the missing dimension", scaled by ``ray_thickness``). Then a multi-view
photometric optimization moves each gaussian to find:

  * position along the ray  (depth ``t``, always),
  * rotation                (``optimize_rotation``),
  * scale                   (``optimize_scale``),

with **color / SH / opacity frozen** (trusted from the 2D fit). Finally redundant
gaussians from overlapping views are fused by moment-matched merging (``merge``).

Depth from scratch is a hard, multi-modal search (photo-consistency along a ray has many
local minima), so descent from the scene-centre init gives scattered geometry — see
docs/EXPERIMENTS.md. The ``cost`` lifter (plane sweep) produces a much better depth
estimate and feeds this optimizer as a *polish* via :func:`optimize_rays`.

Why not a literal epsilon: the covariance is stored as log-scales and the rotation is
recovered by eigendecomposition, so a razor-thin disk (a) underflows / goes singular and
(b) has an ill-defined rotation and near-zero gradient to rotate into the true surface.
``ray_thickness`` is therefore clamped to a small positive fraction of the footprint.
"""

from __future__ import annotations

import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.metrics import ssim
from rtgs.data.scene import SceneData
from rtgs.lift.base import eigvals_2x2, lift_view_at_depth
from rtgs.lift.merge import merge_by_voxel
from rtgs.render.base import get_rasterizer

# Floor on ray_thickness: a disk thinner than this (relative to its lateral footprint)
# makes the eigendecomposition rotation ill-conditioned and the covariance near-singular.
_MIN_THICKNESS = 0.05


def optimize_rays(
    scene: SceneData,
    origins: torch.Tensor,
    dirs: torch.Tensor,
    base_unit: Gaussians3D,
    log_t0: torch.Tensor,
    *,
    iterations: int,
    lr: float = 0.1,
    lr_rotation: float = 5e-3,
    lr_scale: float = 5e-3,
    optimize_depth: bool = True,
    optimize_rotation: bool = True,
    optimize_scale: bool = True,
    ssim_lambda: float = 0.0,
    rasterizer: str = "auto",
    generator: torch.Generator | None = None,
) -> tuple[Gaussians3D, list[float]]:
    """Optimize per-gaussian ray depth (+rotation +scale) by multi-view photometric descent.

    ``base_unit`` holds the gaussians lifted at **unit depth** (rotation + unit-depth
    scales, color/opacity frozen); ``origins``/``dirs`` are the per-gaussian rays (``dirs``
    has unit camera-space depth so ``t`` == depth); ``log_t0`` is the initial log-depth.
    The footprint scales with ``t`` so it stays projection-consistent. ``iterations=0``
    just realizes ``base_unit`` at ``log_t0`` (no optimization) — used to lift at a fixed
    depth estimate. Lateral position and color/SH/opacity stay frozen throughout.
    """
    log_t = log_t0.detach().clone().requires_grad_(optimize_depth)
    quat_p = base_unit.quats.detach().clone().requires_grad_(optimize_rotation)
    dscale = torch.zeros_like(base_unit.log_scales).requires_grad_(optimize_scale)

    def build() -> Gaussians3D:
        t = log_t.exp()
        return Gaussians3D(
            means=origins + t[:, None] * dirs,
            quats=torch.nn.functional.normalize(quat_p, dim=-1),
            log_scales=base_unit.log_scales + log_t[:, None] + dscale,
            opacity=base_unit.opacity,
            sh=base_unit.sh,
        )

    history: list[float] = []
    groups = []
    if optimize_depth:
        groups.append({"params": [log_t], "lr": lr})
    if optimize_rotation:
        groups.append({"params": [quat_p], "lr": lr_rotation})
    if optimize_scale:
        groups.append({"params": [dscale], "lr": lr_scale})
    if iterations > 0 and groups:
        opt = torch.optim.Adam(groups)
        renderer = get_rasterizer(rasterizer)
        n_views = scene.n_views
        for _ in range(iterations):
            v = int(torch.randint(0, n_views, (1,), generator=generator))
            out = renderer.render(build(), scene.cameras[v])
            loss = (out.color - scene.images[v]).abs().mean()
            if ssim_lambda > 0:
                loss = loss + ssim_lambda * (1.0 - ssim(out.color, scene.images[v]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            history.append(float(loss.detach()))
    return build().detach(), history


def lift_unit_depth(
    camera,
    g2d: Gaussians2D,
    ray_thickness: float,
    sh_degree: int,
) -> tuple[torch.Tensor, torch.Tensor, Gaussians3D]:
    """Lift one view's gaussians at unit depth; return (origins, dirs, base gaussians).

    The along-ray sigma is the footprint minor axis scaled by ``ray_thickness`` (the
    "epsilon" knob). ``dirs`` has unit camera-space depth so ``t`` equals camera depth.
    """
    n = g2d.n
    o, d = camera.pixel_rays(g2d.xy)
    s_min = eigvals_2x2(g2d.covariance())[:, 0].clamp_min(1e-8).sqrt()
    sigma_unit = ray_thickness * s_min / (0.5 * (camera.fx + camera.fy))
    base = lift_view_at_depth(camera, g2d, torch.ones(n), sigma_unit, sh_degree=sh_degree)
    return o.expand(n, 3), d, base


class GradientLifter:
    """Lift 2D gaussians and optimize depth (+rotation +scale) by multi-view descent."""

    def __init__(
        self,
        iterations: int = 150,
        lr: float = 0.1,
        lr_rotation: float = 5e-3,
        lr_scale: float = 5e-3,
        rasterizer: str = "auto",
        min_weight: float = 0.05,
        depth_jitter: float = 0.15,
        ray_thickness: float = 1.0,
        optimize_rotation: bool = True,
        optimize_scale: bool = True,
        ssim_lambda: float = 0.0,
        merge: bool = True,
        merge_voxel_frac: float = 0.01,
        sh_degree: int = 0,
        seed: int = 0,
    ):
        self.iterations = iterations
        self.lr = lr
        self.lr_rotation = lr_rotation
        self.lr_scale = lr_scale
        self.rasterizer = rasterizer
        self.min_weight = min_weight
        self.depth_jitter = depth_jitter
        self.ray_thickness = max(ray_thickness, _MIN_THICKNESS)
        self.optimize_rotation = optimize_rotation
        self.optimize_scale = optimize_scale
        self.ssim_lambda = ssim_lambda
        self.merge = merge
        self.merge_voxel_frac = merge_voxel_frac
        self.sh_degree = sh_degree
        self.seed = seed
        self.history: list[float] = []
        self.n_before_merge = 0

    def lift(self, gaussians2d: list[Gaussians2D], scene: SceneData) -> Gaussians3D:
        """Initialize all rays at the scene-center depth and descend on photometric loss."""
        center, extent = scene.center_and_extent()
        gen = torch.Generator().manual_seed(self.seed)

        origins, dirs, log_t0, base_parts = [], [], [], []
        for g2d, camera in zip(gaussians2d, scene.cameras):
            keep = g2d.weight > self.min_weight
            g2d_v = Gaussians2D(
                xy=g2d.xy[keep], chol=g2d.chol[keep], color=g2d.color[keep], weight=g2d.weight[keep]
            )
            if g2d_v.n == 0:
                continue
            o, d, base = lift_unit_depth(camera, g2d_v, self.ray_thickness, self.sh_degree)
            origins.append(o)
            dirs.append(d)
            # Initial depth: distance from the camera to the scene center (+ jitter to
            # break the all-at-one-shell symmetry).
            t_init = (center - camera.position).norm()
            jitter = 1.0 + self.depth_jitter * (2 * torch.rand(g2d_v.n, generator=gen) - 1)
            log_t0.append((t_init * jitter).log())
            base_parts.append(base)
        if not base_parts:
            raise ValueError("no gaussians above the weight threshold to lift")

        result, self.history = optimize_rays(
            scene,
            torch.cat(origins),
            torch.cat(dirs),
            Gaussians3D.cat(base_parts),
            torch.cat(log_t0),
            iterations=self.iterations,
            lr=self.lr,
            lr_rotation=self.lr_rotation,
            lr_scale=self.lr_scale,
            optimize_rotation=self.optimize_rotation,
            optimize_scale=self.optimize_scale,
            ssim_lambda=self.ssim_lambda,
            rasterizer=self.rasterizer,
            generator=gen,
        )
        self.n_before_merge = result.n
        if self.merge:
            result = merge_by_voxel(result, self.merge_voxel_frac * extent)
        return result
