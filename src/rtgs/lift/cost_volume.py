"""Variant D: model-free depth from a multi-view plane sweep (cost volume), then lift.

Motivation: with no external depth model and no SfM points, we still must place each 2D
gaussian at a depth to bring all views into one coordinate frame. Optimizing depth by
per-ray gradient descent (the `gradient` lifter) is fragile — photo-consistency along a
ray is highly multi-modal, so descent from a single init lands in a local minimum. This
lifter instead does what classical multi-view stereo does: evaluate *many discrete depth
candidates* along each pixel ray, score each by cross-view color consistency, and pick the
most consistent one (a soft-argmin). That escapes local minima while staying fully
self-contained — images + camera poses only (poses define the rays; the sweep supplies
depth). Occlusion is handled by robustly averaging only the best-agreeing neighbor views;
a coarse-to-fine schedule narrows the depth range each round. The width of the consistency
minimum sets the per-gaussian along-ray sigma. Optionally the `gradient` ray optimizer
polishes the result (``polish_iters``). See docs/RESEARCH.md §5 and docs/EXPERIMENTS.md.
"""

from __future__ import annotations

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.scene import SceneData
from rtgs.lift.base import bilinear_sample, lift_view_at_depth
from rtgs.lift.gradient import optimize_rays
from rtgs.lift.merge import merge_by_voxel

_NEAR = 0.05


class CostVolumeLifter:
    """Estimate per-gaussian depth by a plane sweep against neighboring views, then lift."""

    def __init__(
        self,
        n_depths: int = 64,
        coarse_to_fine_rounds: int = 3,
        refine_ratio: float = 0.3,
        robust_keep_frac: float = 0.6,
        min_valid_views: int = 2,
        max_cost: float = 0.03,
        cost_tau: float = 0.01,
        max_neighbors: int | None = None,
        depth_range_frac: float = 1.1,
        ray_thickness: float = 1.0,
        min_weight: float = 0.05,
        merge: bool = True,
        merge_voxel_frac: float = 0.01,
        polish_iters: int = 0,
        polish_kwargs: dict | None = None,
        rasterizer: str = "auto",
        sh_degree: int = 0,
        seed: int = 0,
    ):
        self.n_depths = n_depths
        self.coarse_to_fine_rounds = coarse_to_fine_rounds
        self.refine_ratio = refine_ratio
        self.robust_keep_frac = robust_keep_frac
        self.min_valid_views = min_valid_views
        self.max_cost = max_cost
        self.cost_tau = cost_tau
        self.max_neighbors = max_neighbors
        self.depth_range_frac = depth_range_frac
        self.ray_thickness = ray_thickness
        self.min_weight = min_weight
        self.merge = merge
        self.merge_voxel_frac = merge_voxel_frac
        self.polish_iters = polish_iters
        self.polish_kwargs = polish_kwargs or {}
        self.rasterizer = rasterizer
        self.sh_degree = sh_degree
        self.seed = seed
        self.n_before_merge = 0

    def _neighbors(self, scene: SceneData, i: int) -> list[int]:
        others = [j for j in range(scene.n_views) if j != i]
        if self.max_neighbors is None or len(others) <= self.max_neighbors:
            return others
        pos_i = scene.cameras[i].position
        others.sort(key=lambda j: float((scene.cameras[j].position - pos_i).norm()))
        return others[: self.max_neighbors]

    def _estimate_depth(
        self,
        cam: Camera,
        uv: torch.Tensor,
        colors: torch.Tensor,
        neighbor_cams: list[Camera],
        neighbor_imgs: list[torch.Tensor],
        near: float,
        far: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Plane-sweep depth for one view.

        Returns (t_best, sigma_world, best_cost, seen) per gaussian, where ``best_cost`` is
        the color-consistency cost at the chosen depth (a confidence: low = reliable).
        """
        n = uv.shape[0]
        o, d = cam.pixel_rays(uv)  # d: unit camera-space depth
        steps = torch.linspace(0.0, 1.0, self.n_depths)
        lo = torch.full((n,), near)
        hi = torch.full((n,), far)

        t_best = torch.full((n,), 0.5 * (near + far))
        t_std = torch.full((n,), 0.1 * (far - near))
        best_cost = torch.full((n,), float("inf"))
        seen = torch.zeros(n, dtype=torch.bool)

        for _ in range(max(self.coarse_to_fine_rounds, 1)):
            ts = lo[:, None] + (hi - lo)[:, None] * steps[None, :]  # (N,K)
            pts = o.reshape(1, 1, 3) + ts[:, :, None] * d[:, None, :]  # (N,K,3)
            flat = pts.reshape(-1, 3)

            per_view = []
            for ncam, nimg in zip(neighbor_cams, neighbor_imgs):
                uv_j, z_j = ncam.project(flat)
                inside = (z_j > near) & ncam.in_image(uv_j, margin=-0.5)
                col = bilinear_sample(nimg, uv_j)  # (N*K,3)
                cost = ((col - colors.repeat_interleave(self.n_depths, dim=0)) ** 2).sum(-1)
                cost = torch.where(inside, cost, torch.full_like(cost, float("inf")))
                per_view.append(cost.reshape(n, self.n_depths))

            stacked = torch.stack(per_view, dim=0)  # (M,N,K), invalid = inf
            n_valid = torch.isfinite(stacked).sum(0)  # (N,K)
            m = stacked.shape[0]
            keep = min(max(self.min_valid_views, round(self.robust_keep_frac * m)), m)
            topk, _ = torch.sort(stacked, dim=0)
            topk = topk[:keep]  # (keep,N,K) — best-agreeing views (robust to occlusion)
            finite = torch.isfinite(topk)
            agg = torch.where(finite, topk, torch.zeros_like(topk)).sum(0) / finite.sum(
                0
            ).clamp_min(1)
            agg = torch.where(
                n_valid >= self.min_valid_views, agg, torch.full_like(agg, float("inf"))
            )

            finite_k = torch.isfinite(agg)  # (N,K)
            round_seen = finite_k.any(dim=1)
            agg_safe = torch.where(finite_k, agg, torch.full_like(agg, 1e9))
            round_cost, best_idx = agg_safe.min(dim=1)
            t_round = ts.gather(1, best_idx[:, None])[:, 0]
            # Soft spread around the minimum → along-ray depth uncertainty.
            w = torch.softmax(-(agg_safe - round_cost[:, None]) / self.cost_tau, 1)
            w = (w * finite_k).clamp_min(0)
            w = w / w.sum(1, keepdim=True).clamp_min(1e-8)
            t_mean = (w * ts).sum(1)
            t_round_std = (w * (ts - t_mean[:, None]) ** 2).sum(1).clamp_min(0).sqrt()

            t_best = torch.where(round_seen, t_round, t_best)
            t_std = torch.where(round_seen, t_round_std, t_std)
            best_cost = torch.where(round_seen, round_cost, best_cost)
            seen = seen | round_seen

            span = (hi - lo).clamp_min(1e-6)
            lo = (t_best - self.refine_ratio * span).clamp_min(near)
            hi = (t_best + self.refine_ratio * span).clamp_min(lo + 1e-4)

        dnorm = d.norm(dim=-1)
        return t_best, (t_std * dnorm), best_cost, seen

    def lift(self, gaussians2d: list[Gaussians2D], scene: SceneData) -> Gaussians3D:
        """Plane-sweep depth per view, lift, optionally polish, and merge."""
        center, extent = scene.center_and_extent()
        origins, dirs, log_t0, base_parts = [], [], [], []

        for i, (g2d, cam) in enumerate(zip(gaussians2d, scene.cameras)):
            keep = g2d.weight > self.min_weight
            g2d_v = Gaussians2D(
                xy=g2d.xy[keep], chol=g2d.chol[keep], color=g2d.color[keep], weight=g2d.weight[keep]
            )
            if g2d_v.n == 0:
                continue
            nbr = self._neighbors(scene, i)
            cam_dist = float((cam.position - center).norm())
            near = max(_NEAR, cam_dist - self.depth_range_frac * extent)
            far = cam_dist + self.depth_range_frac * extent

            t_best, sigma_world, best_cost, seen = self._estimate_depth(
                cam,
                g2d_v.xy,
                g2d_v.color.clamp(0, 1),
                [scene.cameras[j] for j in nbr],
                [scene.images[j] for j in nbr],
                near,
                far,
            )
            # Reject gaussians whose depth could not be reliably estimated (unseen or a
            # poor best-match cost) — these are the floaters that wreck refinement.
            conf = seen & (best_cost < self.max_cost)
            if int(conf.sum()) == 0:
                continue
            g2d_v = Gaussians2D(
                xy=g2d_v.xy[conf],
                chol=g2d_v.chol[conf],
                color=g2d_v.color[conf],
                weight=g2d_v.weight[conf],
            )
            t_best, sigma_world = t_best[conf], sigma_world[conf]

            o, d = cam.pixel_rays(g2d_v.xy)
            # Lift at unit depth: convert the world-space along-ray sigma to unit-depth
            # (it scales with t) so that at t_best the footprint matches sigma_world.
            sigma_unit = (sigma_world / t_best.clamp_min(1e-4)).clamp_min(1e-4)
            base = lift_view_at_depth(
                cam, g2d_v, torch.ones(g2d_v.n), sigma_unit, sh_degree=self.sh_degree
            )
            origins.append(o.expand(g2d_v.n, 3))
            dirs.append(d)
            log_t0.append(t_best.clamp_min(_NEAR).log())
            base_parts.append(base)

        if not base_parts:
            raise ValueError("cost-volume lifting produced no gaussians")

        gen = torch.Generator().manual_seed(self.seed)
        result, _ = optimize_rays(
            scene,
            torch.cat(origins),
            torch.cat(dirs),
            Gaussians3D.cat(base_parts),
            torch.cat(log_t0),
            iterations=self.polish_iters,
            rasterizer=self.rasterizer,
            generator=gen,
            **self.polish_kwargs,
        )
        self.n_before_merge = result.n
        if self.merge:
            result = merge_by_voxel(result, self.merge_voxel_frac * extent)
        return result
