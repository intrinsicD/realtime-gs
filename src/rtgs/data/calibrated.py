"""Loader for object-centric RGB captures with per-camera calibration JSON.

The supported JSON layout stores a flattened OpenCV ``camera_matrix``, five Brown-Conrady
distortion coefficients, and a flattened world-to-camera ``view_matrix`` per camera.  Images
may be named either ``C0004.jpg`` or ``rgb_4.jpeg``.  The loader resizes and undistorts both
RGB and masks without adding an OpenCV dependency.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from rtgs.core.camera import Camera
from rtgs.data.scene import SceneData

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _camera_id(path: Path) -> str | None:
    """Map capture filenames to canonical calibration ids."""
    stem = path.stem
    if stem.lower().startswith("mask_"):
        stem = stem[5:]
    match = re.fullmatch(r"[Cc](\d+)", stem) or re.fullmatch(r"rgb_(\d+)", stem.lower())
    return None if match is None else f"C{int(match.group(1)):04d}"


def _find_calibration(frame_dir: Path) -> Path:
    for parent in (frame_dir, *frame_dir.parents[:3]):
        candidate = parent / "calibration_dome.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"could not find calibration_dome.json above {frame_dir}")


def _sample_evenly(paths: list[Path], maximum: int | None) -> list[Path]:
    if maximum is None or maximum >= len(paths):
        return paths
    if maximum <= 0:
        raise ValueError("max_images must be positive")
    indices = torch.linspace(0, len(paths) - 1, maximum).round().long().unique().tolist()
    return [paths[i] for i in indices]


def _resize_image(path: Path, width: int, height: int, *, mask: bool = False) -> torch.Tensor:
    mode = "L" if mask else "RGB"
    interpolation = PILImage.Resampling.NEAREST if mask else PILImage.Resampling.LANCZOS
    image = PILImage.open(path).convert(mode)
    if image.size != (width, height):
        image = image.resize((width, height), interpolation)
    array = np.array(image, dtype=np.float32, copy=True)
    tensor = torch.from_numpy(array) / 255.0
    return tensor if mask else tensor[..., :3]


def _undistort(
    image: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion: list[float],
    *,
    mask: bool = False,
) -> torch.Tensor:
    """Sample a distorted image onto an ideal pinhole grid (OpenCV distortion model)."""
    if not distortion or max(abs(float(v)) for v in distortion) < 1e-12:
        return image
    height, width = image.shape[:2]
    v, u = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) + 0.5,
        torch.arange(width, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    x = (u - cx) / fx
    y = (v - cy) / fy
    k1, k2, p1, p2, k3 = (list(distortion) + [0.0] * 5)[:5]
    r2 = x.square() + y.square()
    radial = 1.0 + k1 * r2 + k2 * r2.square() + k3 * r2.pow(3)
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x.square())
    yd = y * radial + p1 * (r2 + 2.0 * y.square()) + 2.0 * p2 * x * y
    grid = torch.stack(
        [2.0 * (fx * xd + cx) / width - 1.0, 2.0 * (fy * yd + cy) / height - 1.0],
        dim=-1,
    )
    source = image[None, None] if mask else image.permute(2, 0, 1)[None]
    sampled = F.grid_sample(
        source,
        grid[None],
        mode="nearest" if mask else "bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0]
    return sampled[0] if mask else sampled.permute(1, 2, 0)


def _axis_intersection(cameras: list[Camera]) -> torch.Tensor:
    positions = torch.stack([camera.position for camera in cameras])
    forwards = F.normalize(torch.stack([camera.R[2] for camera in cameras]), dim=-1)
    return _ray_intersection(positions, forwards)


def _ray_intersection(origins: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    directions = F.normalize(directions, dim=-1)
    eye = torch.eye(3, dtype=origins.dtype, device=origins.device)
    projectors = eye[None] - directions[:, :, None] * directions[:, None, :]
    a = projectors.sum(dim=0)
    b = (projectors @ origins[:, :, None]).sum(dim=0)[:, 0]
    return torch.linalg.lstsq(a, b).solution


def _object_bounds(
    cameras: list[Camera], masks: list[torch.Tensor] | None
) -> tuple[torch.Tensor, float]:
    center = _axis_intersection(cameras)
    if masks is not None:
        origins, directions = [], []
        for camera, mask in zip(cameras, masks):
            foreground = torch.nonzero(mask > 0.5)
            if foreground.shape[0] < 16:
                continue
            centroid_vu = foreground.float().mean(dim=0) + 0.5
            uv = centroid_vu[[1, 0]][None]
            origin, direction = camera.pixel_rays(uv)
            origins.append(origin)
            directions.append(direction[0])
        if len(origins) >= 2:
            center = _ray_intersection(torch.stack(origins), torch.stack(directions))
    distances = torch.stack([(camera.position - center).norm() for camera in cameras])
    radii: list[torch.Tensor] = []
    if masks is not None:
        for camera, mask, distance in zip(cameras, masks, distances):
            foreground = torch.nonzero(mask > 0.5)
            if foreground.shape[0] < 16:
                continue
            v0, u0 = foreground.amin(dim=0).float() + 0.5
            v1, u1 = foreground.amax(dim=0).float() + 0.5
            half_width = 0.5 * (u1 - u0) / camera.fx
            half_height = 0.5 * (v1 - v0) / camera.fy
            radii.append(distance * torch.sqrt(half_width.square() + half_height.square()))
    radius = torch.stack(radii).median() if radii else 0.3 * distances.median()
    return center, float((2.4 * radius).clamp_min(1e-3))


def load_calibrated_scene(
    frame_dir: str | Path,
    *,
    calibration_path: str | Path | None = None,
    downscale: int = 1,
    max_images: int | None = None,
    test_every: int = 8,
    load_masks: bool = True,
    undistort: bool = True,
) -> SceneData:
    """Load one calibrated object-centric frame.

    ``frame_dir`` can point either to the frame or its ``rgb`` directory.  ``max_images``
    retains evenly spaced cameras so quick runs do not accidentally use one side of the rig.
    """
    frame = Path(frame_dir).expanduser().resolve()
    if frame.name == "rgb":
        frame = frame.parent
    rgb_dir = frame / "rgb"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"missing RGB directory: {rgb_dir}")
    if downscale < 1:
        raise ValueError("downscale must be at least one")

    calibration_file = (
        Path(calibration_path).expanduser().resolve()
        if calibration_path is not None
        else _find_calibration(frame)
    )
    calibration = json.loads(calibration_file.read_text())
    records = {record["camera_id"].upper(): record for record in calibration["cameras"]}
    paths = [
        path
        for path in rgb_dir.iterdir()
        if path.suffix.lower() in _IMAGE_SUFFIXES and _camera_id(path) in records
    ]
    paths = _sample_evenly(sorted(paths, key=lambda path: _camera_id(path) or ""), max_images)
    if not paths:
        raise ValueError(f"no calibrated RGB images found in {rgb_dir}")

    images: list[torch.Tensor] = []
    cameras: list[Camera] = []
    masks: list[torch.Tensor] = []
    missing_masks: list[str] = []
    mask_dir = frame / "mask"
    for path in paths:
        camera_id = _camera_id(path)
        assert camera_id is not None
        record = records[camera_id]
        intrinsics = record["intrinsics"]
        calibration_width, calibration_height = map(int, intrinsics["resolution"])
        with PILImage.open(path) as source:
            source_width, source_height = source.size
        width = max(1, source_width // downscale)
        height = max(1, source_height // downscale)
        sx = width / calibration_width
        sy = height / calibration_height
        matrix = intrinsics["camera_matrix"]
        fx, fy = float(matrix[0]) * sx, float(matrix[4]) * sy
        # OpenCV stores the first pixel center at integer coordinates.  Camera uses +0.5.
        cx, cy = (float(matrix[2]) + 0.5) * sx, (float(matrix[5]) + 0.5) * sy
        view = torch.tensor(record["extrinsics"]["view_matrix"], dtype=torch.float32).view(4, 4)
        camera = Camera(fx, fy, cx, cy, width, height, view[:3, :3], view[:3, 3])
        image = _resize_image(path, width, height)
        distortion = intrinsics.get("distortion_coefficients", [])
        if undistort:
            image = _undistort(image, fx, fy, cx, cy, distortion)
        images.append(image)
        cameras.append(camera)

        if load_masks:
            candidates = [mask_dir / f"mask_{camera_id}.png", mask_dir / f"mask_{camera_id}.jpg"]
            mask_path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if mask_path is None:
                missing_masks.append(camera_id)
            else:
                mask = _resize_image(mask_path, width, height, mask=True)
                masks.append(
                    _undistort(mask, fx, fy, cx, cy, distortion, mask=True) if undistort else mask
                )

    scene_masks = masks if load_masks and not missing_masks and len(masks) == len(images) else None
    if missing_masks and mask_dir.is_dir():
        warnings.warn(
            f"ignoring incomplete masks: missing {len(missing_masks)} of {len(images)} views",
            stacklevel=2,
        )
    test_indices = list(range(test_every - 1, len(images), test_every)) if test_every > 0 else []
    train_indices = [index for index in range(len(images)) if index not in set(test_indices)]
    scene = SceneData(
        images=images,
        cameras=cameras,
        view_names=[path.stem for path in paths],
        masks=scene_masks,
        train_indices=train_indices,
        test_indices=test_indices,
        bounds_hint=_object_bounds(cameras, scene_masks),
        name=frame.name,
    )
    scene.validate()
    return scene
