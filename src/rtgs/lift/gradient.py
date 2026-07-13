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
        init_opacity: float = 0.1,
        optimize_rotation: bool = True,
        optimize_scale: bool = True,
        max_log_scale_delta: float = 1.0,
        depth_prior_lambda: float = 1e-3,
        scale_prior_lambda: float = 1e-4,
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
        self.init_opacity = init_opacity
        self.optimize_rotation = optimize_rotation
        self.optimize_scale = optimize_scale
        self.max_log_scale_delta = max_log_scale_delta
        self.depth_prior_lambda = depth_prior_lambda
        self.scale_prior_lambda = scale_prior_lambda
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
        device = gaussians2d[0].xy.device
        center = center.to(device)
        gen = torch.Generator(device=device).manual_seed(self.seed)

        origins, dirs, t_nears, t_fars, raw_t0 = [], [], [], [], []
        base_parts: list[Gaussians3D] = []
        half = 0.5 * extent
        for view_index, (g2d, camera) in enumerate(zip(gaussians2d, scene.cameras)):
            keep = g2d.weight > self.min_weight
            if scene.masks is not None:
                from rtgs.lift.base import bilinear_sample

                keep &= bilinear_sample(scene.masks[view_index].to(g2d.xy), g2d.xy) > 0.5
            g2d_v = Gaussians2D(
                xy=g2d.xy[keep], chol=g2d.chol[keep], color=g2d.color[keep], weight=g2d.weight[keep]
            )
            n = g2d_v.n
            if n == 0:
                continue
            o, d = camera.pixel_rays(g2d_v.xy)  # d has unit camera-space depth: t == depth
            near, far = _ray_box(o, d, center - half, center + half)
            intersects = far > near.clamp_min(0.05)
            if not bool(intersects.any()):
                continue
            g2d_v = Gaussians2D(
                xy=g2d_v.xy[intersects],
                chol=g2d_v.chol[intersects],
                color=g2d_v.color[intersects],
                weight=g2d_v.weight[intersects],
            )
            d = d[intersects]
            near = near[intersects].clamp_min(0.05)
            far = far[intersects]
            n = g2d_v.n
            origins.append(o.expand(n, 3))
            dirs.append(d)
            t_nears.append(near)
            t_fars.append(far)
            # Initialize at the optical-axis center depth, then jitter within the valid ray/AABB
            # interval. Unlike Euclidean camera distance this is the correct parameter for d.z=1.
            center_depth = camera.project(center[None])[1][0]
            fraction = ((center_depth - near) / (far - near)).clamp(0.05, 0.95)
            fraction = (
                fraction + self.depth_jitter * (torch.rand(n, generator=gen, device=device) - 0.5)
            ).clamp(0.01, 0.99)
            raw_t0.append(torch.logit(fraction))
            # Unit-depth lift fixes each gaussian's rotation and unit-depth scales; the
            # along-ray sigma is the footprint minor axis scaled by ray_thickness (the
            # "epsilon" knob, clamped away from degeneracy).
            s_min = eigvals_2x2(g2d_v.covariance())[:, 0].clamp_min(1e-8).sqrt()
            sigma_unit = self.ray_thickness * s_min / (0.5 * (camera.fx + camera.fy))
            base_parts.append(
                lift_view_at_depth(
                    camera,
                    g2d_v,
                    torch.ones(n, device=device),
                    sigma_unit,
                    sh_degree=self.sh_degree,
                    opacity=torch.full((n,), self.init_opacity, device=device),
                )
            )
        if not base_parts:
            raise ValueError("no gaussians above the weight threshold to lift")

        base = Gaussians3D.cat(base_parts)
        origins_t = torch.cat(origins)
        dirs_t = torch.cat(dirs)
        near_t = torch.cat(t_nears)
        far_t = torch.cat(t_fars)

        # Optimized parameters. Lateral position and color/SH/opacity stay frozen.
        raw_t_init = torch.cat(raw_t0)
        raw_t = raw_t_init.clone().requires_grad_(True)
        quat_p = base.quats.detach().clone().requires_grad_(self.optimize_rotation)
        raw_scale = torch.zeros_like(base.log_scales).requires_grad_(self.optimize_scale)

        groups = [{"params": [raw_t], "lr": self.lr}]
        if self.optimize_rotation:
            groups.append({"params": [quat_p], "lr": self.lr_rotation})
        if self.optimize_scale:
            groups.append({"params": [raw_scale], "lr": self.lr_scale})
        opt = torch.optim.Adam(groups)

        def build() -> Gaussians3D:
            t = near_t + torch.sigmoid(raw_t) * (far_t - near_t)
            dscale = self.max_log_scale_delta * torch.tanh(raw_scale)
            return Gaussians3D(
                means=origins_t + t[:, None] * dirs_t,
                quats=torch.nn.functional.normalize(quat_p, dim=-1),
                # t-scaling keeps the projected footprint ~constant with depth; dscale is
                # the free per-axis correction the optimizer learns.
                log_scales=base.log_scales + t.log()[:, None] + dscale,
                opacity=base.opacity,
                sh=base.sh,
            )

        renderer = get_rasterizer(self.rasterizer)
        self.history = []
        train_views = scene.training_views
        for _ in range(self.iterations):
            view_pos = int(torch.randint(0, len(train_views), (1,), generator=gen, device=device))
            v = train_views[view_pos]
            out = renderer.render(build(), scene.cameras[v])
            target = scene.images[v]
            if scene.masks is not None:
                mask = scene.masks[v].to(target)[..., None]
                target = target * mask
                loss = ((out.color - target).abs() * (0.1 + 0.9 * mask)).mean()
            else:
                loss = (out.color - target).abs().mean()
            if self.ssim_lambda > 0:
                loss = loss + self.ssim_lambda * (1.0 - ssim(out.color, target))
            loss = loss + self.depth_prior_lambda * (raw_t - raw_t_init).square().mean()
            if self.optimize_scale:
                loss = loss + self.scale_prior_lambda * raw_scale.square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            self.history.append(float(loss.detach()))

        result = build().detach()
        self.n_before_merge = result.n
        if self.merge:
            result = merge_by_voxel(result, self.merge_voxel_frac * extent, opacity_mode="mean")
        return result


def _ray_box(
    origin: torch.Tensor, directions: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Intersect one-origin camera rays with an axis-aligned scene volume."""
    safe = torch.where(directions.abs() < 1e-9, torch.full_like(directions, 1e-9), directions)
    ta = (lo[None] - origin[None]) / safe
    tb = (hi[None] - origin[None]) / safe
    return torch.minimum(ta, tb).amax(dim=-1), torch.maximum(ta, tb).amin(dim=-1)
