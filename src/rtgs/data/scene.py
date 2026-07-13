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
    view_names: list[str] | None = None  # stable ids used to align external per-view fits
    points: torch.Tensor | None = None  # (M, 3) sparse world points
    # Per-view indices into ``points``. COLMAP loaders populate these from observation tracks so
    # depth alignment never uses points that were not actually observed by that camera.
    point_visibility: list[torch.Tensor] | None = None
    masks: list[torch.Tensor] | None = None  # optional (H,W) foreground probability/bool maps
    gt_depths: list[torch.Tensor] | None = None  # each (H, W)
    gt_gaussians: Gaussians3D | None = None
    train_indices: list[int] | None = None
    test_indices: list[int] | None = None
    # Optional object bounds supplied by a calibrated dataset loader: center and full diameter.
    bounds_hint: tuple[torch.Tensor, float] | None = None
    name: str = "scene"
    _extent_cache: tuple[torch.Tensor, float] | None = field(default=None, repr=False)

    @property
    def n_views(self) -> int:
        """Number of posed images."""
        return len(self.images)

    def center_and_extent(self) -> tuple[torch.Tensor, float]:
        """Scene center and padded full-diameter extent estimated from camera geometry.

        Prefer robust sparse-point bounds. When points are unavailable, intersect the cameras'
        optical axes in least squares instead of guessing object depth from camera baselines.
        ``extent`` is the padded full diameter of the object volume.
        """
        if self._extent_cache is not None:
            return self._extent_cache
        if self.bounds_hint is not None:
            center, extent = self.bounds_hint
            self._extent_cache = (center, float(extent))
            return self._extent_cache

        positions = torch.stack([c.position for c in self.cameras])  # (C,3)
        if self.points is not None:
            finite = self.points[torch.isfinite(self.points).all(dim=-1)]
        else:
            finite = positions.new_empty((0, 3))
        if finite.shape[0] >= 4:
            if finite.shape[0] >= 20:
                lo = torch.quantile(finite, 0.01, dim=0)
                hi = torch.quantile(finite, 0.99, dim=0)
            else:
                lo, hi = finite.amin(dim=0), finite.amax(dim=0)
            center = 0.5 * (lo + hi)
            radius = (finite - center).norm(dim=-1)
            radius = torch.quantile(radius, 0.99) if radius.numel() >= 20 else radius.max()
        else:
            forwards = torch.stack([c.R[2] for c in self.cameras])
            forwards = torch.nn.functional.normalize(forwards, dim=-1)
            eye = torch.eye(3, dtype=positions.dtype, device=positions.device)
            projectors = eye[None] - forwards[:, :, None] * forwards[:, None, :]
            a = projectors.sum(dim=0)
            b = (projectors @ positions[:, :, None]).sum(dim=0)[:, 0]
            center = torch.linalg.lstsq(a, b).solution
            camera_distance = (positions - center).norm(dim=-1)
            radius = 0.25 * camera_distance.median().clamp_min(1e-3)
        extent = float((2.2 * radius).clamp_min(1e-3))
        self._extent_cache = (center, extent)
        return self._extent_cache

    @property
    def training_views(self) -> list[int]:
        """Indices used for optimization (all views unless an explicit split exists)."""
        return self.train_indices if self.train_indices is not None else list(range(self.n_views))

    @property
    def testing_views(self) -> list[int]:
        """Held-out indices used for reporting."""
        return self.test_indices or []

    def to(self, device: torch.device | str) -> SceneData:
        """Return a scene copy on ``device`` without carrying a stale bounds cache."""
        hint = None
        if self.bounds_hint is not None:
            hint = (self.bounds_hint[0].to(device), self.bounds_hint[1])
        return SceneData(
            images=[image.to(device) for image in self.images],
            cameras=[camera.to(device) for camera in self.cameras],
            view_names=None if self.view_names is None else list(self.view_names),
            points=None if self.points is None else self.points.to(device),
            point_visibility=(
                None
                if self.point_visibility is None
                else [indices.to(device) for indices in self.point_visibility]
            ),
            masks=None if self.masks is None else [mask.to(device) for mask in self.masks],
            gt_depths=(
                None if self.gt_depths is None else [depth.to(device) for depth in self.gt_depths]
            ),
            gt_gaussians=None if self.gt_gaussians is None else self.gt_gaussians.to(device),
            train_indices=None if self.train_indices is None else list(self.train_indices),
            test_indices=None if self.test_indices is None else list(self.test_indices),
            bounds_hint=hint,
            name=self.name,
        )

    def validate(self) -> None:
        """Raise on inconsistent view counts or image/camera size mismatches."""
        if len(self.images) != len(self.cameras):
            raise ValueError("images and cameras count mismatch")
        if self.view_names is not None and len(self.view_names) != len(self.images):
            raise ValueError("view_names count mismatch")
        for img, cam in zip(self.images, self.cameras):
            if img.shape[0] != cam.height or img.shape[1] != cam.width:
                raise ValueError("image size does not match camera")
        if self.gt_depths is not None and len(self.gt_depths) != len(self.images):
            raise ValueError("gt_depths count mismatch")
        if self.masks is not None:
            if len(self.masks) != len(self.images):
                raise ValueError("masks count mismatch")
            for mask, image in zip(self.masks, self.images):
                if mask.shape != image.shape[:2]:
                    raise ValueError("mask size does not match image")
        if self.point_visibility is not None:
            if self.points is None or len(self.point_visibility) != len(self.images):
                raise ValueError("point_visibility requires points and one entry per view")
            for indices in self.point_visibility:
                if indices.numel() and (
                    int(indices.min()) < 0 or int(indices.max()) >= len(self.points)
                ):
                    raise ValueError("point_visibility index out of range")
        all_split = (self.train_indices or []) + (self.test_indices or [])
        split_overlaps = len(set(all_split)) != len(all_split)
        split_out_of_range = any(i < 0 or i >= self.n_views for i in all_split)
        if split_overlaps or split_out_of_range:
            raise ValueError("invalid or overlapping train/test indices")
