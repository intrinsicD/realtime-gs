"""Active silhouette-hull floater relocation (RTGS-026, opt-in research seam).

A 3D Gaussian centre is *rejected* by a calibrated view when it lies in front of that camera,
projects inside its image, and lands outside that view's (dilated) silhouette mask. A single
rejecting view marks the Gaussian as a floater: a point outside any silhouette cannot lie on the
object. Floaters are moved, not pruned: each is placed at the nearest occupied visual-hull voxel
(plus a small in-voxel jitter that stays inside the hull), and the Adam moments of its rows are
reset in every parameter group so stale momentum does not carry it back out. Colour, scale,
rotation and opacity are kept, so capacity is preserved.

The relocator plugs into ``Trainer.train(parameter_step_callback=...)``. Nothing here is a
default; masks may come from any calibrated views, including views whose colours are withheld.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from rtgs.core.camera import Camera


@dataclass(frozen=True)
class SilhouetteRelocationConfig:
    """Schedule and tolerances; steps are completed Trainer iterations."""

    every: int = 100
    start: int = 500
    stop: int = 7000
    jitter_fraction: float = 0.5
    seed: int = 0
    chunk: int = 65536

    def __post_init__(self) -> None:
        if self.every <= 0 or self.start < 0 or self.stop < self.start:
            raise ValueError("relocation schedule requires every > 0 and 0 <= start <= stop")
        if not 0.0 <= self.jitter_fraction <= 1.0:
            raise ValueError("jitter_fraction must lie in [0, 1]")

    def active(self, step: int) -> bool:
        return self.start <= step <= self.stop and (step - self.start) % self.every == 0


class SilhouetteHull:
    """Silhouette tests for 3D points against a set of calibrated binary masks."""

    def __init__(
        self,
        cameras: list[Camera],
        masks: list[torch.Tensor],
        *,
        dilation_px: int = 1,
        threshold: float = 0.5,
        near: float = 1e-3,
        device: torch.device | str = "cpu",
    ) -> None:
        if len(cameras) != len(masks) or not cameras:
            raise ValueError("need one mask per camera and at least one view")
        self.device = torch.device(device)
        self.near = near
        self.cameras = [camera.to(self.device) for camera in cameras]
        self.masks = []
        for camera, mask in zip(self.cameras, masks):
            if tuple(mask.shape) != (camera.height, camera.width):
                raise ValueError("mask shape must equal the camera image shape")
            binary = (mask.to(self.device, torch.float32) >= threshold).float()
            if dilation_px:
                kernel = 2 * dilation_px + 1
                binary = F.max_pool2d(binary[None, None], kernel, 1, dilation_px)[0, 0]
            self.masks.append(binary > 0.5)

    def _views(self, points: torch.Tensor):
        for camera, mask in zip(self.cameras, self.masks):
            uv, depth = camera.project(points)
            seen = (depth > self.near) & camera.in_image(uv)
            column = uv[:, 0].floor().long().clamp(0, camera.width - 1)
            row = uv[:, 1].floor().long().clamp(0, camera.height - 1)
            yield seen, mask[row, column]

    def rejected(self, points: torch.Tensor) -> torch.Tensor:
        """True where at least one view sees the point outside its dilated silhouette."""
        points = points.to(self.device, torch.float32)
        out = torch.zeros(points.shape[0], dtype=torch.bool, device=self.device)
        for seen, inside in self._views(points):
            out |= seen & ~inside
        return out

    def supported(self, points: torch.Tensor) -> torch.Tensor:
        """True where no view rejects the point and at least one view sees it inside."""
        points = points.to(self.device, torch.float32)
        rejected = torch.zeros(points.shape[0], dtype=torch.bool, device=self.device)
        support = torch.zeros_like(rejected)
        for seen, inside in self._views(points):
            rejected |= seen & ~inside
            support |= seen & inside
        return support & ~rejected

    def occupied_voxels(
        self, center: torch.Tensor, extent: float, grid: int
    ) -> tuple[torch.Tensor, float]:
        """Centres of supported voxels on a ``grid``^3 lattice over the cube of side ``extent``."""
        step = extent / grid
        axis = (torch.arange(grid, device=self.device, dtype=torch.float32) + 0.5) * step
        axis = axis - extent / 2
        lattice = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
        points = lattice + center.to(self.device, torch.float32)
        keep = torch.cat([self.supported(chunk) for chunk in points.split(262144)])
        occupied = points[keep]
        if occupied.shape[0] == 0:
            raise RuntimeError("the silhouette hull is empty on this lattice")
        return occupied, step


@dataclass
class SilhouetteRelocator:
    """``parameter_step_callback`` that moves hull-rejected Gaussian centres into the hull."""

    hull: SilhouetteHull
    targets: torch.Tensor  # (M,3) occupied voxel centres
    voxel: float
    config: SilhouetteRelocationConfig = field(default_factory=SilhouetteRelocationConfig)
    events: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.targets = self.targets.to(self.hull.device, torch.float32)
        self._generator = torch.Generator(device=self.hull.device).manual_seed(self.config.seed)
        self.last_fallbacks = 0

    def nearest_targets(self, points: torch.Tensor) -> torch.Tensor:
        chosen = torch.empty(points.shape[0], dtype=torch.long, device=points.device)
        for start in range(0, points.shape[0], 256):
            distances = torch.cdist(points[start : start + 256], self.targets)
            chosen[start : start + 256] = distances.argmin(1)
        return self.targets[chosen]

    def relocate(self, means: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(rows, new_means)`` for rejected centres; ``means`` is not modified."""
        points = means.detach().to(self.hull.device, torch.float32)
        rejected = torch.cat([self.hull.rejected(c) for c in points.split(self.config.chunk)])
        rows = torch.nonzero(rejected).squeeze(1)
        self.last_fallbacks = 0
        if rows.numel() == 0:
            return rows, points[:0]
        base = self.nearest_targets(points[rows])
        jitter = torch.rand(base.shape, generator=self._generator, device=base.device) - 0.5
        moved = base + jitter * self.config.jitter_fraction * self.voxel
        outside = ~self.hull.supported(moved)
        moved[outside] = base[outside]
        self.last_fallbacks = int(outside.sum())
        return rows, moved

    def __call__(self, params: dict, optimizers: dict, step: int) -> None:
        if not self.config.active(step):
            return
        means = params["means"]
        rows, moved = self.relocate(means)
        record = {
            "step": int(step),
            "n_gaussians": int(means.shape[0]),
            "relocated": 0,
            "jitter_fallbacks": 0,
        }
        if rows.numel():
            rows_param = rows.to(means.device)
            means.data[rows_param] = moved.to(means.device, means.dtype)
            for optimizer in optimizers.values():
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        state = optimizer.state.get(parameter, {})
                        for key in ("exp_avg", "exp_avg_sq"):
                            if key in state and state[key].shape[:1] == parameter.shape[:1]:
                                state[key][rows_param] = 0
            record["relocated"] = int(rows.numel())
            record["jitter_fallbacks"] = self.last_fallbacks
        self.events.append(record)
