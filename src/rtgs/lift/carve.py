"""Variant C: voxel color-consistency carving + placement along projection tunnels.

Three classic ideas fused for 2D-gaussian lifting (Laurentini visual hull; Seitz-Dyer
voxel coloring; Kutulakos-Seitz space carving — docs/RESEARCH.md §5):

1. A voxel grid over the scene accumulates, per voxel, the colors it projects onto in
   every view plus a *coverage* test (does any 2D gaussian actually cover that pixel?).
   Voxels not covered in views that see them are carved (silhouette/hull constraint from
   the 2D gaussian fits themselves); surviving voxels are scored by cross-view color
   consistency (photo-consistency).
2. Each 2D gaussian marches its own projection tunnel (camera ray through the grid) and
   is placed at the depth where voxel score x color-match peaks; the local spread of
   good depths sets the along-ray sigma.
3. Gaussians from different views landing in the same cell are fused by moment-matched
   merging (rtgs.lift.merge).
"""

from __future__ import annotations

import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.scene import SceneData
from rtgs.image2gs.renderer2d import render_gaussians_2d
from rtgs.lift.base import bilinear_sample, lift_view_at_depth
from rtgs.lift.merge import merge_by_voxel

_NEAR = 0.05


class CarveLifter:
    """Carve a consistency volume from the 2D-gaussian fits and place gaussians in it."""

    def __init__(
        self,
        grid_res: int = 48,
        bounds_scale: float = 0.5,
        min_views: int = 2,
        hull_fraction: float = 0.85,
        color_std_sigma: float = 0.20,
        color_match_sigma: float = 0.35,
        coverage_thresh: float = 0.08,
        samples_per_ray: int = 64,
        min_score: float = 0.05,
        min_weight: float = 0.05,
        merge: bool = True,
        merge_voxel_scale: float = 1.0,
        sh_degree: int = 0,
    ):
        self.grid_res = grid_res
        self.bounds_scale = bounds_scale
        self.min_views = min_views
        self.hull_fraction = hull_fraction
        self.color_std_sigma = color_std_sigma
        self.color_match_sigma = color_match_sigma
        self.coverage_thresh = coverage_thresh
        self.samples_per_ray = samples_per_ray
        self.min_score = min_score
        self.min_weight = min_weight
        self.merge = merge
        self.merge_voxel_scale = merge_voxel_scale
        self.sh_degree = sh_degree

    def lift(self, gaussians2d: list[Gaussians2D], scene: SceneData) -> Gaussians3D:
        """Build the carved volume, then place and merge all views' gaussians."""
        center, extent = scene.center_and_extent()
        half = extent * self.bounds_scale
        lo = center - half
        voxel = 2.0 * half / self.grid_res
        g = self.grid_res

        # Voxel centers, flattened (G^3, 3) with index = (z*G + y)*G + x.
        r = torch.arange(g, dtype=torch.float32)
        zz, yy, xx = torch.meshgrid(r, r, r, indexing="ij")
        idx3 = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        centers = lo[None, :] + (idx3 + 0.5) * voxel
        nv = centers.shape[0]

        # Per-view coverage maps: luminance of the stage-1 reconstruction. This is a soft
        # silhouette — near-zero splat energy reads as background/unknown. Scenes without
        # dark backgrounds are simply "covered everywhere" and rely on the
        # photo-consistency test alone.
        coverage_maps = []
        with torch.no_grad():
            for g2d, cam in zip(gaussians2d, scene.cameras):
                recon = render_gaussians_2d(g2d, cam.height, cam.width)
                coverage_maps.append(recon.clamp_min(0.0).mean(dim=-1))

        # Color statistics accumulate over ALL views that see a voxel (not only covered
        # ones): a free-space voxel projecting onto the object in some views and onto
        # background in others must show its cross-view variance to get carved.
        n_seen = torch.zeros(nv)
        n_covered = torch.zeros(nv)
        c_sum = torch.zeros(nv, 3)
        c_sqsum = torch.zeros(nv, 3)
        for image, cov_map, cam in zip(scene.images, coverage_maps, scene.cameras):
            uv, z = cam.project(centers)
            inside = (z > _NEAR) & cam.in_image(uv, margin=-0.5)
            idx = inside.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            covered = bilinear_sample(cov_map, uv[idx]) > self.coverage_thresh
            colors = bilinear_sample(image, uv[idx])
            n_seen[idx] += 1
            n_covered[idx] += covered.float()
            c_sum[idx] += colors
            c_sqsum[idx] += colors**2

        seen_ok = n_seen >= self.min_views
        hull = seen_ok & (n_covered >= self.hull_fraction * n_seen) & (n_covered >= self.min_views)
        cnt = n_seen.clamp_min(1.0)
        mean_color = c_sum / cnt[:, None]
        var = (c_sqsum / cnt[:, None] - mean_color**2).clamp_min(0.0)
        std = var.mean(dim=-1).sqrt()
        consistency = hull.float() * torch.exp(-(std**2) / (2 * self.color_std_sigma**2))

        # Place each view's gaussians along their tunnels.
        parts: list[Gaussians3D] = []
        s = self.samples_per_ray
        for g2d, cam in zip(gaussians2d, scene.cameras):
            keep = g2d.weight > self.min_weight
            if int(keep.sum()) == 0:
                continue
            g2d_v = Gaussians2D(
                xy=g2d.xy[keep], chol=g2d.chol[keep], color=g2d.color[keep], weight=g2d.weight[keep]
            )
            o, d = cam.pixel_rays(g2d_v.xy)  # d has unit camera-space depth: t == depth
            t0, t1 = _ray_box(o, d, lo, lo + 2 * half)
            valid_ray = t1 > t0.clamp_min(_NEAR)
            t0 = t0.clamp_min(_NEAR)
            steps = torch.linspace(0.0, 1.0, s)
            ts = t0[:, None] + (t1 - t0).clamp_min(0.0)[:, None] * steps[None, :]  # (N,S)
            pts = o.reshape(1, 1, 3) + ts[:, :, None] * d[:, None, :]  # (N,S,3)

            vox = torch.floor((pts - lo) / voxel).long()  # (N,S,3)
            in_grid = ((vox >= 0) & (vox < g)).all(dim=-1) & valid_ray[:, None]
            vox = vox.clamp(0, g - 1)
            flat = (vox[..., 2] * g + vox[..., 1]) * g + vox[..., 0]  # (N,S)

            score = consistency[flat] * in_grid.float()
            vcolor = mean_color[flat]  # (N,S,3)
            cdiff = ((vcolor - g2d_v.color[:, None, :]) ** 2).sum(-1)
            score = score * torch.exp(-cdiff / (2 * self.color_match_sigma**2))

            best_score, best_idx = score.max(dim=1)
            placed = best_score > self.min_score
            if int(placed.sum()) == 0:
                continue

            # sigma_ray from the score-weighted depth spread near the peak.
            t_best = torch.gather(ts, 1, best_idx[:, None])[:, 0]
            near_peak = (ts - t_best[:, None]).abs() <= 3.0 * voxel
            w_loc = score * near_peak.float()
            w_sum = w_loc.sum(dim=1).clamp_min(1e-8)
            t_mean = (w_loc * ts).sum(dim=1) / w_sum
            t_var = (w_loc * (ts - t_mean[:, None]) ** 2).sum(dim=1) / w_sum
            sigma_ray = t_var.sqrt().clamp(0.25 * voxel, 2.0 * voxel)

            sub = Gaussians2D(
                xy=g2d_v.xy[placed],
                chol=g2d_v.chol[placed],
                color=g2d_v.color[placed],
                weight=g2d_v.weight[placed],
            )
            parts.append(
                lift_view_at_depth(cam, sub, t_best[placed], sigma_ray[placed], self.sh_degree)
            )

        if not parts:
            raise ValueError(
                "carving produced no gaussians — loosen color_std_sigma/min_score or check "
                "that the scene bounds cover the content (bounds_scale)"
            )
        result = Gaussians3D.cat(parts)
        if self.merge:
            result = merge_by_voxel(result, voxel * self.merge_voxel_scale)
        return result


def _ray_box(
    o: torch.Tensor, d: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slab-method ray/AABB intersection: entry and exit t for rays o + t*d (t >= 0)."""
    d_safe = torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
    ta = (lo[None, :] - o[None, :]) / d_safe
    tb = (hi[None, :] - o[None, :]) / d_safe
    tmin = torch.minimum(ta, tb).max(dim=-1).values
    tmax = torch.maximum(ta, tb).min(dim=-1).values
    return tmin.clamp_min(0.0), tmax
