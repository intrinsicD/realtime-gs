"""Variant A: move per-view gaussians into the scene by multi-view photometric descent.

Each 2D gaussian stays constrained to its own camera ray; the only free parameter is its
depth t (optimized in log space). We exploit that the lifted covariance scales exactly
quadratically with depth: lift once at unit depth (fixing the rotation), then during
optimization means(t) = origin + t * ray and log_scales(t) = log_scales(1) + log(t) —
fully differentiable with no eigendecomposition in the loop. The loss renders the joint
gaussian set into randomly sampled views and compares against the input images (L1).
"""

from __future__ import annotations

import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.scene import SceneData
from rtgs.lift.base import eigvals_2x2, lift_view_at_depth
from rtgs.render.base import get_rasterizer


class GradientLifter:
    """Optimize per-gaussian ray depths by rendering into the other views."""

    def __init__(
        self,
        iterations: int = 120,
        lr: float = 0.1,
        rasterizer: str = "auto",
        min_weight: float = 0.05,
        depth_jitter: float = 0.15,
        sh_degree: int = 0,
        seed: int = 0,
    ):
        self.iterations = iterations
        self.lr = lr
        self.rasterizer = rasterizer
        self.min_weight = min_weight
        self.depth_jitter = depth_jitter
        self.sh_degree = sh_degree
        self.seed = seed
        self.history: list[float] = []

    def lift(self, gaussians2d: list[Gaussians2D], scene: SceneData) -> Gaussians3D:
        """Initialize all rays at the scene-center depth and descend on photometric loss."""
        center, _ = scene.center_and_extent()
        gen = torch.Generator().manual_seed(self.seed)

        origins, dirs, log_t0 = [], [], []
        unit_parts: list[Gaussians3D] = []
        for g2d, camera in zip(gaussians2d, scene.cameras):
            keep = g2d.weight > self.min_weight
            g2d_v = Gaussians2D(
                xy=g2d.xy[keep], chol=g2d.chol[keep], color=g2d.color[keep], weight=g2d.weight[keep]
            )
            n = g2d_v.n
            if n == 0:
                continue
            o, d = camera.pixel_rays(g2d_v.xy)  # d has unit camera-space depth
            origins.append(o.expand(n, 3))
            dirs.append(d)
            # Initial depth: distance from the camera to the scene center (+ jitter to
            # break the all-at-one-shell symmetry).
            t_init = (center - camera.position).norm()
            jitter = 1.0 + self.depth_jitter * (2 * torch.rand(n, generator=gen) - 1)
            log_t0.append((t_init * jitter).log())
            # Unit-depth lift fixes per-gaussian rotation and unit-depth scales.
            s_min = eigvals_2x2(g2d_v.covariance())[:, 0].clamp_min(1e-8).sqrt()
            sigma_unit = s_min / (0.5 * (camera.fx + camera.fy))
            unit_parts.append(
                lift_view_at_depth(
                    camera, g2d_v, torch.ones(n), sigma_unit, sh_degree=self.sh_degree
                )
            )
        if not unit_parts:
            raise ValueError("no gaussians above the weight threshold to lift")

        unit = Gaussians3D.cat(unit_parts)
        origins_t = torch.cat(origins)
        dirs_t = torch.cat(dirs)
        log_t = torch.cat(log_t0).clone().requires_grad_(True)

        renderer = get_rasterizer(self.rasterizer)
        opt = torch.optim.Adam([log_t], lr=self.lr)
        self.history = []

        def build(current_log_t: torch.Tensor) -> Gaussians3D:
            t = current_log_t.exp()
            return Gaussians3D(
                means=origins_t + t[:, None] * dirs_t,
                quats=unit.quats,
                log_scales=unit.log_scales + current_log_t[:, None],
                opacity=unit.opacity,
                sh=unit.sh,
            )

        n_views = scene.n_views
        for _ in range(self.iterations):
            v = int(torch.randint(0, n_views, (1,), generator=gen))
            out = renderer.render(build(log_t), scene.cameras[v])
            loss = (out.color - scene.images[v]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            self.history.append(float(loss.detach()))

        return build(log_t.detach()).detach()
