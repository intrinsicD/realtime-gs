"""SceneData: the multi-view input bundle every pipeline stage consumes."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D


@dataclass
class SceneData:
    """Posed multi-view images plus whatever ground truth / priors are available.

    ``points`` are sparse world points (SfM output or synthetic GT samples) used for
    depth alignment and the `sfm` baseline lifter. ``gt_depths``/``gt_gaussians`` exist
    only for synthetic scenes and are used by tests and the GT depth backend.
    """

    images: list[torch.Tensor]  # each (H, W, 3) float32 in [0,1]
    cameras: list[Camera]
    points: torch.Tensor | None = None  # (M, 3) sparse world points
    gt_depths: list[torch.Tensor] | None = None  # each (H, W)
    gt_gaussians: Gaussians3D | None = None
    name: str = "scene"
    _extent_cache: tuple[torch.Tensor, float] | None = field(default=None, repr=False)

    @property
    def n_views(self) -> int:
        """Number of posed images."""
        return len(self.images)

    def center_and_extent(self) -> tuple[torch.Tensor, float]:
        """Scene center (3,) and extent (scalar radius) estimated from camera geometry.

        Center: mean of per-camera 'look-at' proxies (camera center + viewing direction
        scaled by the median camera-pair distance); extent: max camera distance to that
        center. Matches 3DGS's use of camera bounds for LR scaling and pruning.
        """
        if self._extent_cache is not None:
            return self._extent_cache
        positions = torch.stack([c.position for c in self.cameras])  # (C,3)
        if len(self.cameras) > 1:
            dists = torch.cdist(positions, positions)
            baseline = dists[dists > 0].median()
        else:
            baseline = torch.tensor(1.0)
        forwards = torch.stack([c.R[2] for c in self.cameras])  # camera +z in world
        center = (positions + forwards * baseline).mean(dim=0)
        extent = float((positions - center).norm(dim=-1).max().clamp_min(1e-3))
        self._extent_cache = (center, extent)
        return self._extent_cache

    def validate(self) -> None:
        """Raise on inconsistent view counts or image/camera size mismatches."""
        if len(self.images) != len(self.cameras):
            raise ValueError("images and cameras count mismatch")
        for img, cam in zip(self.images, self.cameras):
            if img.shape[0] != cam.height or img.shape[1] != cam.width:
                raise ValueError("image size does not match camera")
        if self.gt_depths is not None and len(self.gt_depths) != len(self.images):
            raise ValueError("gt_depths count mismatch")
