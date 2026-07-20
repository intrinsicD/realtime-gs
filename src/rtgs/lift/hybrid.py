"""Depth-seeded bounded-ray photometric lifting.

Feed-forward depth supplies a fast prior, while calibrated multi-view optimization can correct it
along each observation ray before confidence/color-aware voxel fusion.
"""

from __future__ import annotations

import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.scene import SceneData
from rtgs.depth.base import DepthBackend
from rtgs.lift.depth import predict_aligned_depth_priors
from rtgs.lift.gradient import GradientLifter
from rtgs.lift.surface import OrientedPointTargets


class HybridLifter:
    """Seed GradientLifter with aligned monocular depth and confidence."""

    def __init__(
        self,
        backend: DepthBackend | None = None,
        iterations: int = 100,
        rasterizer: str = "auto",
        depth_prior_lambda: float = 0.01,
        merge_color_bin_size: float | None = 0.15,
        depth_anchor_mode: str = "legacy",
        photometric_supervision_mode: str = "all",
        position_consistency_lambda: float = 0.0,
        position_consistency_beta: float = 0.05,
        plane_consistency_lambda: float = 0.0,
        normal_consistency_lambda: float = 0.0,
        **gradient_kwargs,
    ):
        self.backend = backend
        self.merge_color_bin_size = merge_color_bin_size
        # A depth-seeded run needs only enough jitter to break ties; the gradient-only default
        # intentionally explores much more of each bounded ray.
        gradient_kwargs.setdefault("depth_jitter", 0.02)
        self.gradient = GradientLifter(
            iterations=iterations,
            rasterizer=rasterizer,
            depth_prior_lambda=depth_prior_lambda,
            depth_anchor_mode=depth_anchor_mode,
            photometric_supervision_mode=photometric_supervision_mode,
            position_consistency_lambda=position_consistency_lambda,
            position_consistency_beta=position_consistency_beta,
            plane_consistency_lambda=plane_consistency_lambda,
            normal_consistency_lambda=normal_consistency_lambda,
            **gradient_kwargs,
        )

    @property
    def history(self) -> list[float]:
        return self.gradient.history

    @property
    def anchor_history(self) -> list[float]:
        return self.gradient.anchor_history

    @property
    def position_history(self) -> list[float]:
        return self.gradient.position_history

    @property
    def plane_history(self) -> list[float]:
        return self.gradient.plane_history

    @property
    def normal_history(self) -> list[float]:
        return self.gradient.normal_history

    def lift(self, gaussians2d: list[Gaussians2D], scene: SceneData) -> Gaussians3D:
        priors = predict_aligned_depth_priors(scene, self.backend)
        return self.gradient.lift_with_priors(
            gaussians2d,
            scene,
            priors,
            merge_color_bin_size=self.merge_color_bin_size,
        )

    def lift_with_position_pairs(
        self,
        gaussians2d: list[Gaussians2D],
        scene: SceneData,
        position_pairs: torch.Tensor,
    ) -> Gaussians3D:
        """Depth-seeded lift with one fixed position-only cross-view graph."""
        priors = predict_aligned_depth_priors(scene, self.backend)
        return self.gradient.lift_with_priors(
            gaussians2d,
            scene,
            priors,
            merge_color_bin_size=self.merge_color_bin_size,
            position_pairs=position_pairs,
        )

    def lift_with_oriented_points(
        self,
        gaussians2d: list[Gaussians2D],
        scene: SceneData,
        oriented_targets: OrientedPointTargets,
    ) -> Gaussians3D:
        """Depth-seeded lift with fixed local-plane and normal-alignment targets."""
        priors = predict_aligned_depth_priors(scene, self.backend)
        return self.gradient.lift_with_priors(
            gaussians2d,
            scene,
            priors,
            merge_color_bin_size=self.merge_color_bin_size,
            oriented_targets=oriented_targets,
        )
