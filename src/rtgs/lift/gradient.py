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
gaussians from overlapping views are fused by moment-matched merging (``merge``) so the
downstream 3DGS stage does not start with N-views of duplicated geometry.

Why not a literal epsilon: the covariance is stored as log-scales and the rotation is
recovered by eigendecomposition, so a razor-thin disk (a) underflows / goes singular and
(b) has an ill-defined rotation and near-zero gradient to rotate into the true surface.
``ray_thickness`` is therefore clamped to a small positive fraction of the footprint. See
docs/RESEARCH.md §"Missing-dimension covariance" and docs/EXPERIMENTS.md.
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
        """Lift every view, optimize along the rays, and merge the result."""
        center, extent = scene.center_and_extent()
        gen = torch.Generator().manual_seed(self.seed)

        origins, dirs, log_t0 = [], [], []
        base_parts: list[Gaussians3D] = []
        for g2d, camera in zip(gaussians2d, scene.cameras):
            keep = g2d.weight > self.min_weight
            g2d_v = Gaussians2D(
                xy=g2d.xy[keep], chol=g2d.chol[keep], color=g2d.color[keep], weight=g2d.weight[keep]
            )
            n = g2d_v.n
            if n == 0:
                continue
            o, d = camera.pixel_rays(g2d_v.xy)  # d has unit camera-space depth: t == depth
            origins.append(o.expand(n, 3))
            dirs.append(d)
            # Initial depth: distance from the camera to the scene center (+ jitter to
            # break the all-at-one-shell symmetry).
            t_init = (center - camera.position).norm()
            jitter = 1.0 + self.depth_jitter * (2 * torch.rand(n, generator=gen) - 1)
            log_t0.append((t_init * jitter).log())
            # Unit-depth lift fixes each gaussian's rotation and unit-depth scales; the
            # along-ray sigma is the footprint minor axis scaled by ray_thickness (the
            # "epsilon" knob, clamped away from degeneracy).
            s_min = eigvals_2x2(g2d_v.covariance())[:, 0].clamp_min(1e-8).sqrt()
            sigma_unit = self.ray_thickness * s_min / (0.5 * (camera.fx + camera.fy))
            base_parts.append(
                lift_view_at_depth(
                    camera, g2d_v, torch.ones(n), sigma_unit, sh_degree=self.sh_degree
                )
            )
        if not base_parts:
            raise ValueError("no gaussians above the weight threshold to lift")

        base = Gaussians3D.cat(base_parts)
        origins_t = torch.cat(origins)
        dirs_t = torch.cat(dirs)

        # Optimized parameters. Lateral position and color/SH/opacity stay frozen.
        log_t = torch.cat(log_t0).clone().requires_grad_(True)
        quat_p = base.quats.detach().clone().requires_grad_(self.optimize_rotation)
        dscale = torch.zeros_like(base.log_scales).requires_grad_(self.optimize_scale)

        groups = [{"params": [log_t], "lr": self.lr}]
        if self.optimize_rotation:
            groups.append({"params": [quat_p], "lr": self.lr_rotation})
        if self.optimize_scale:
            groups.append({"params": [dscale], "lr": self.lr_scale})
        opt = torch.optim.Adam(groups)

        def build() -> Gaussians3D:
            t = log_t.exp()
            return Gaussians3D(
                means=origins_t + t[:, None] * dirs_t,
                quats=torch.nn.functional.normalize(quat_p, dim=-1),
                # t-scaling keeps the projected footprint ~constant with depth; dscale is
                # the free per-axis correction the optimizer learns.
                log_scales=base.log_scales + log_t[:, None] + dscale,
                opacity=base.opacity,
                sh=base.sh,
            )

        renderer = get_rasterizer(self.rasterizer)
        self.history = []
        n_views = scene.n_views
        for _ in range(self.iterations):
            v = int(torch.randint(0, n_views, (1,), generator=gen))
            out = renderer.render(build(), scene.cameras[v])
            loss = (out.color - scene.images[v]).abs().mean()
            if self.ssim_lambda > 0:
                loss = loss + self.ssim_lambda * (1.0 - ssim(out.color, scene.images[v]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            self.history.append(float(loss.detach()))

        result = build().detach()
        self.n_before_merge = result.n
        if self.merge:
            result = merge_by_voxel(result, self.merge_voxel_frac * extent)
        return result
