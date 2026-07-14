"""Pinhole camera with COLMAP/OpenCV conventions.

Extrinsics are world-to-camera: ``x_cam = R @ x_world + t``; +z looks forward, +y down.
Pixel coordinates: (u, v) = (column, row); the center of the top-left pixel is (0.5, 0.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Camera:
    """A single pinhole camera (intrinsics + world-to-camera extrinsics)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    R: torch.Tensor  # (3,3) world-to-camera rotation
    t: torch.Tensor  # (3,) world-to-camera translation

    def __post_init__(self) -> None:
        self.R = torch.as_tensor(self.R, dtype=torch.float32)
        self.t = torch.as_tensor(self.t, dtype=torch.float32)
        if self.R.shape != (3, 3) or self.t.shape != (3,):
            raise ValueError("R must be (3,3) and t must be (3,)")

    @property
    def K(self) -> torch.Tensor:
        """3x3 intrinsic matrix."""
        return torch.tensor(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=self.R.dtype,
            device=self.R.device,
        )

    @property
    def viewmat(self) -> torch.Tensor:
        """4x4 world-to-camera matrix."""
        m = torch.eye(4, dtype=self.R.dtype, device=self.R.device)
        m[:3, :3] = self.R
        m[:3, 3] = self.t
        return m

    @property
    def position(self) -> torch.Tensor:
        """Camera center in world coordinates: -R^T t."""
        return -self.R.T @ self.t

    def world_to_cam(self, points: torch.Tensor) -> torch.Tensor:
        """Transform (M,3) world points to camera coordinates."""
        return points @ self.R.to(points).T + self.t.to(points)

    def cam_to_world(self, points: torch.Tensor) -> torch.Tensor:
        """Transform (M,3) camera points to world coordinates."""
        return (points - self.t.to(points)) @ self.R.to(points)

    def project(self, points_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project (M,3) world points; returns pixel coords (M,2) and camera-space depth (M,).

        Points behind the camera get negative depth; callers cull on it.
        """
        pc = self.world_to_cam(points_world)
        z = pc[:, 2]
        z_safe = torch.where(z.abs() < 1e-8, torch.full_like(z, 1e-8), z)
        u = self.fx * pc[:, 0] / z_safe + self.cx
        v = self.fy * pc[:, 1] / z_safe + self.cy
        return torch.stack([u, v], dim=-1), z

    def unproject(self, uv: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Back-project (M,2) pixels at (M,) camera-space depths to (M,3) world points."""
        x = (uv[:, 0] - self.cx) / self.fx * depth
        y = (uv[:, 1] - self.cy) / self.fy * depth
        return self.cam_to_world(torch.stack([x, y, depth], dim=-1))

    def pixel_rays(self, uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """World-space rays through (M,2) pixels: (origin (3,), directions (M,3)).

        Directions have unit *camera-space depth* (z = 1 before rotation), so
        ``origin + depth * dirs`` lands exactly on ``unproject(uv, depth)``. Normalize the
        directions yourself if you need unit euclidean length.
        """
        d_cam = torch.stack(
            [
                (uv[:, 0] - self.cx) / self.fx,
                (uv[:, 1] - self.cy) / self.fy,
                torch.ones_like(uv[:, 0]),
            ],
            dim=-1,
        )
        d_world = d_cam @ self.R.to(uv)  # rotate to world (R^T @ d, batched)
        return self.position.to(uv), d_world

    def in_image(self, uv: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
        """Boolean mask of pixels inside the image bounds (with optional margin)."""
        return (
            (uv[:, 0] >= 0.5 - margin)
            & (uv[:, 0] <= self.width - 0.5 + margin)
            & (uv[:, 1] >= 0.5 - margin)
            & (uv[:, 1] <= self.height - 0.5 + margin)
        )

    def to(self, device: torch.device | str) -> Camera:
        """Return a copy whose extrinsics live on ``device``."""
        return Camera(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            width=self.width,
            height=self.height,
            R=self.R.to(device),
            t=self.t.to(device),
        )

    @staticmethod
    def look_at(
        eye: torch.Tensor,
        target: torch.Tensor,
        up: torch.Tensor | None = None,
        fov_x_deg: float = 60.0,
        width: int = 64,
        height: int = 64,
    ) -> Camera:
        """Build a camera at ``eye`` looking at ``target`` (OpenCV axes: +y down, +z forward)."""
        eye = torch.as_tensor(eye, dtype=torch.float32)
        target = torch.as_tensor(target, dtype=torch.float32)
        up = (
            eye.new_tensor([0.0, 1.0, 0.0])
            if up is None
            else torch.as_tensor(up, dtype=torch.float32)
        )
        z = torch.nn.functional.normalize(target - eye, dim=0)
        x = torch.nn.functional.normalize(torch.linalg.cross(z, up), dim=0)
        if not torch.isfinite(x).all() or x.norm() < 1e-6:
            x = torch.nn.functional.normalize(
                torch.linalg.cross(z, eye.new_tensor([1.0, 0.0, 0.0])), dim=0
            )
        y = torch.linalg.cross(z, x)
        r_c2w = torch.stack([x, y, z], dim=1)  # columns are camera axes in world
        r = r_c2w.T
        t = -r @ eye
        fx = 0.5 * width / torch.tan(torch.deg2rad(torch.tensor(fov_x_deg / 2))).item()
        return Camera(
            fx=fx, fy=fx, cx=width / 2, cy=height / 2, width=width, height=height, R=r, t=t
        )
