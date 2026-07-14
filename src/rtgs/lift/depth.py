"""Variant B: lift 2D gaussians to feed-forward monocular depth.

Each gaussian center samples the view's depth map; non-metric predictions are first
aligned to the scene scale against sparse points (rtgs.depth.align). The along-ray
variance comes from the depth spread across the gaussian's footprint
(:func:`rtgs.lift.base.footprint_sigma_ray`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.scene import SceneData
from rtgs.depth.align import (
    align_depth_to_bounds,
    align_depth_to_points,
    align_inverse_depth_to_bounds,
    align_inverse_depth_to_points,
)
from rtgs.depth.base import DepthBackend
from rtgs.lift.base import bilinear_sample, lift_view_from_depth_map
from rtgs.lift.merge import merge_by_voxel

_MIN_DEPTH = 0.05


@dataclass
class AlignedDepthPrior:
    """Metric scene-aligned depth and optional per-pixel confidence for one view."""

    depth: torch.Tensor
    confidence: torch.Tensor | None


def predict_aligned_depth_priors(
    scene: SceneData, backend: DepthBackend | None = None
) -> list[AlignedDepthPrior]:
    """Predict and align every view's depth without committing geometry to that prediction."""
    if backend is None:
        if scene.gt_depths is not None:
            from rtgs.depth.mock import GroundTruthDepth

            backend = GroundTruthDepth(scene.gt_depths)
        else:
            from rtgs.depth.depth_anything import DepthAnythingV2

            backend = DepthAnythingV2()
    center, extent = scene.center_and_extent()
    priors = []
    for view_idx, (camera, image) in enumerate(zip(scene.cameras, scene.images)):
        pred = backend.predict(image)
        depth_map = pred.depth.to(image)
        if pred.kind != "metric":
            points = None
            if scene.points is not None:
                points = scene.points
                if scene.point_visibility is not None:
                    points = points[scene.point_visibility[view_idx]]
            mask = None if scene.masks is None else scene.masks[view_idx]
            aligned_to_points = False
            if points is not None and points.shape[0] >= 2:
                try:
                    if pred.kind == "inverse":
                        depth_map = align_inverse_depth_to_points(depth_map, camera, points)
                    else:
                        depth_map = align_depth_to_points(depth_map, camera, points)
                    aligned_to_points = True
                except ValueError:
                    pass
            if not aligned_to_points and pred.kind == "inverse":
                depth_map = align_inverse_depth_to_bounds(depth_map, camera, center, extent, mask)
            elif not aligned_to_points:
                depth_map = align_depth_to_bounds(depth_map, camera, center, extent, mask)
        confidence = None if pred.confidence is None else pred.confidence.to(image).clamp(0, 1)
        priors.append(AlignedDepthPrior(depth=depth_map, confidence=confidence))
    return priors


class DepthLifter:
    """Lift per-view gaussians using a monocular depth backend."""

    def __init__(
        self,
        backend: DepthBackend | None = None,
        sh_degree: int = 0,
        min_weight: float = 0.05,
        init_opacity: float = 0.1,
        normal_thickness: float = 0.15,
        merge: bool = True,
        merge_voxel_frac: float = 0.01,
    ):
        self.backend = backend
        self.sh_degree = sh_degree
        self.min_weight = min_weight
        self.init_opacity = init_opacity
        self.normal_thickness = normal_thickness
        self.merge = merge
        self.merge_voxel_frac = merge_voxel_frac

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
        priors = predict_aligned_depth_priors(scene, self._resolve_backend(scene))
        parts: list[Gaussians3D] = []
        component_weights = []
        for view_idx, (g2d, camera, prior) in enumerate(zip(gaussians2d, scene.cameras, priors)):
            depth_map = prior.depth.to(g2d.xy)
            z = bilinear_sample(depth_map, g2d.xy)
            valid = torch.isfinite(z) & (z > _MIN_DEPTH) & (g2d.weight > self.min_weight)
            if scene.masks is not None:
                valid &= bilinear_sample(scene.masks[view_idx].to(g2d.xy), g2d.xy) > 0.5
            confidence = torch.ones_like(z)
            if prior.confidence is not None:
                confidence = bilinear_sample(prior.confidence.to(g2d.xy), g2d.xy)
                valid &= confidence > 0.1
            if int(valid.sum()) == 0:
                continue
            g2d_v = Gaussians2D(
                xy=g2d.xy[valid],
                chol=g2d.chol[valid],
                color=g2d.color[valid],
                weight=g2d.weight[valid],
            )
            z_v = z[valid]
            component_weights.append((g2d.weight[valid] * confidence[valid]).clamp_min(1e-3))
            opacity = torch.full_like(z_v, self.init_opacity)
            parts.append(
                lift_view_from_depth_map(
                    camera,
                    g2d_v,
                    depth_map,
                    z_v,
                    self.sh_degree,
                    opacity=opacity,
                    normal_thickness=self.normal_thickness,
                )
            )
        if not parts:
            raise ValueError("no gaussians survived depth lifting (all invalid depth/weight)")
        result = Gaussians3D.cat(parts)
        if self.merge:
            _, extent = scene.center_and_extent()
            result = merge_by_voxel(
                result,
                self.merge_voxel_frac * extent,
                opacity_mode="mean",
                component_weights=torch.cat(component_weights),
            )
        return result
