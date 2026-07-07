"""Variant B: lift 2D gaussians to feed-forward monocular depth.

Each gaussian center samples the view's depth map; non-metric predictions are first
aligned to the scene scale against sparse points (rtgs.depth.align). The along-ray
variance comes from the depth spread across the gaussian's footprint
(:func:`rtgs.lift.base.footprint_sigma_ray`).
"""

from __future__ import annotations

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.scene import SceneData
from rtgs.depth.align import align_depth_to_points
from rtgs.depth.base import DepthBackend
from rtgs.lift.base import bilinear_sample, footprint_sigma_ray, lift_view_at_depth

_MIN_DEPTH = 0.05


class DepthLifter:
    """Lift per-view gaussians using a monocular depth backend."""

    def __init__(
        self,
        backend: DepthBackend | None = None,
        sh_degree: int = 0,
        min_weight: float = 0.05,
    ):
        self.backend = backend
        self.sh_degree = sh_degree
        self.min_weight = min_weight

    def _resolve_backend(self, scene: SceneData) -> DepthBackend:
        if self.backend is not None:
            return self.backend
        if scene.gt_depths is not None:
            from rtgs.depth.mock import GroundTruthDepth

            return GroundTruthDepth(scene.gt_depths)
        from rtgs.depth.depth_anything import DepthAnythingV2

        return DepthAnythingV2()

    def lift(self, gaussians2d: list[Gaussians2D], scene: SceneData) -> Gaussians3D:
        """Lift every view at its predicted depth and concatenate."""
        backend = self._resolve_backend(scene)
        parts: list[Gaussians3D] = []
        for g2d, camera, image in zip(gaussians2d, scene.cameras, scene.images):
            pred = backend.predict(image)
            depth_map = pred.depth
            if pred.kind != "metric":
                if scene.points is None or scene.points.shape[0] < 2:
                    raise ValueError(
                        f"depth backend returned '{pred.kind}' depth but the scene has no "
                        "sparse points to align against"
                    )
                depth_map = align_depth_to_points(depth_map, camera, scene.points)

            z = bilinear_sample(depth_map, g2d.xy)
            valid = (z > _MIN_DEPTH) & (g2d.weight > self.min_weight)
            if int(valid.sum()) == 0:
                continue
            g2d_v = Gaussians2D(
                xy=g2d.xy[valid],
                chol=g2d.chol[valid],
                color=g2d.color[valid],
                weight=g2d.weight[valid],
            )
            z_v = z[valid]
            sigma_ray = footprint_sigma_ray(camera, g2d_v, depth_map, z_v)
            parts.append(lift_view_at_depth(camera, g2d_v, z_v, sigma_ray, self.sh_degree))
        if not parts:
            raise ValueError("no gaussians survived depth lifting (all invalid depth/weight)")
        return Gaussians3D.cat(parts)
