"""COLMAP sparse-reconstruction loading (text and binary formats).

Reads ``cameras``, ``images``, ``points3D`` from a COLMAP model directory (typically
``<dataset>/sparse/0``) and the corresponding image files, producing a
:class:`~rtgs.data.scene.SceneData`. PINHOLE, SIMPLE_PINHOLE, (SIMPLE_)RADIAL and OPENCV
models are supported; distorted models are resampled onto their ideal pinhole grid.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rtgs.core.camera import Camera
from rtgs.data.scene import SceneData

_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
}
_MODEL_NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}


@dataclass
class ColmapModel:
    """Raw parsed COLMAP model: intrinsics, per-image extrinsics, sparse points."""

    cameras: dict[int, dict]  # camera_id -> {model, width, height, params}
    images: dict[int, dict]  # image_id -> {name, qvec, tvec, camera_id}
    points: torch.Tensor  # (M, 3)
    point_ids: torch.Tensor  # (M,) original COLMAP point3D ids


def _qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def read_cameras_text(path: Path) -> dict[int, dict]:
    """Parse cameras.txt."""
    cams: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cams[int(parts[0])] = {
            "model": parts[1],
            "width": int(parts[2]),
            "height": int(parts[3]),
            "params": [float(p) for p in parts[4:]],
        }
    return cams


def read_images_text(path: Path) -> dict[int, dict]:
    """Parse images.txt (skipping the per-image 2D point lines)."""
    images: dict[int, dict] = {}
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("#")):
            i += 1
        if i >= len(lines):
            break
        parts = lines[i].strip().split(maxsplit=9)
        i += 1
        while i < len(lines) and lines[i].lstrip().startswith("#"):
            i += 1
        observations = lines[i].strip().split() if i < len(lines) else []
        i += 1
        if len(observations) % 3:
            raise ValueError("malformed COLMAP images.txt observation line")
        images[int(parts[0])] = {
            "qvec": np.array([float(p) for p in parts[1:5]]),
            "tvec": np.array([float(p) for p in parts[5:8]]),
            "camera_id": int(parts[8]),
            "name": parts[9],
            "point3d_ids": [int(v) for v in observations[2::3] if int(v) >= 0],
        }
    return images


def read_points3d_text(path: Path) -> torch.Tensor:
    """Parse points3D.txt into an (M, 3) tensor."""
    points, _ = _read_points3d_text_full(path)
    return points


def _read_points3d_text_full(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse coordinates and stable COLMAP point ids from points3D.txt."""
    pts = []
    ids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        ids.append(int(parts[0]))
        pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    points = torch.tensor(pts, dtype=torch.float32) if pts else torch.zeros(0, 3)
    point_ids = torch.tensor(ids, dtype=torch.long) if ids else torch.zeros(0, dtype=torch.long)
    return points, point_ids


def read_cameras_binary(path: Path) -> dict[int, dict]:
    """Parse cameras.bin."""
    cams: dict[int, dict] = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cam_id, model_id, width, height = struct.unpack("<iiQQ", f.read(24))
            num = _MODEL_NUM_PARAMS.get(model_id)
            if num is None:
                raise ValueError(f"unsupported COLMAP camera model id {model_id}")
            params = struct.unpack(f"<{num}d", f.read(8 * num))
            cams[cam_id] = {
                "model": _MODEL_NAMES[model_id],
                "width": int(width),
                "height": int(height),
                "params": list(params),
            }
    return cams


def read_images_binary(path: Path) -> dict[int, dict]:
    """Parse images.bin."""
    images: dict[int, dict] = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<4d", f.read(32)))
            tvec = np.array(struct.unpack("<3d", f.read(24)))
            (camera_id,) = struct.unpack("<I", f.read(4))
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (n_pts,) = struct.unpack("<Q", f.read(8))
            f.read(24 * n_pts)  # skip 2D points (x, y, point3D_id)
            # Seek back and retain the track ids needed for per-view depth alignment.
            f.seek(-24 * n_pts, 1)
            point3d_ids = []
            for _ in range(n_pts):
                _, _, point_id = struct.unpack("<ddq", f.read(24))
                if point_id >= 0:
                    point3d_ids.append(point_id)
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name.decode("utf-8"),
                "point3d_ids": point3d_ids,
            }
    return images


def read_points3d_binary(path: Path) -> torch.Tensor:
    """Parse points3D.bin into an (M, 3) tensor."""
    points, _ = _read_points3d_binary_full(path)
    return points


def _read_points3d_binary_full(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse coordinates and stable COLMAP point ids from points3D.bin."""
    pts = []
    ids = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            (point_id,) = struct.unpack("<Q", f.read(8))
            ids.append(point_id)
            xyz = struct.unpack("<3d", f.read(24))
            f.read(3)  # rgb
            f.read(8)  # error
            (track_len,) = struct.unpack("<Q", f.read(8))
            f.read(8 * track_len)
            pts.append(xyz)
    points = torch.tensor(pts, dtype=torch.float32) if pts else torch.zeros(0, 3)
    point_ids = torch.tensor(ids, dtype=torch.long) if ids else torch.zeros(0, dtype=torch.long)
    return points, point_ids


def read_model(model_dir: str | Path) -> ColmapModel:
    """Read a COLMAP model directory, preferring binary files when both exist."""
    d = Path(model_dir)
    if (d / "cameras.bin").exists():
        cams = read_cameras_binary(d / "cameras.bin")
        images = read_images_binary(d / "images.bin")
        points, point_ids = _read_points3d_binary_full(d / "points3D.bin")
    elif (d / "cameras.txt").exists():
        cams = read_cameras_text(d / "cameras.txt")
        images = read_images_text(d / "images.txt")
        points, point_ids = _read_points3d_text_full(d / "points3D.txt")
    else:
        raise FileNotFoundError(f"no COLMAP model (cameras.bin/.txt) in {d}")
    return ColmapModel(cameras=cams, images=images, points=points, point_ids=point_ids)


def _intrinsics(cam: dict) -> tuple[float, float, float, float]:
    model, p = cam["model"], cam["params"]
    if model == "SIMPLE_PINHOLE":
        return p[0], p[0], p[1], p[2]
    if model == "PINHOLE":
        return p[0], p[1], p[2], p[3]
    if model in ("SIMPLE_RADIAL", "RADIAL", "OPENCV"):
        if model == "SIMPLE_RADIAL" or model == "RADIAL":
            return p[0], p[0], p[1], p[2]
        return p[0], p[1], p[2], p[3]
    raise ValueError(f"unsupported COLMAP camera model {model}")


def _distortion(cam: dict) -> list[float]:
    model, p = cam["model"], cam["params"]
    if model in ("PINHOLE", "SIMPLE_PINHOLE"):
        return []
    if model == "SIMPLE_RADIAL":
        return [p[3], 0.0, 0.0, 0.0, 0.0]
    if model == "RADIAL":
        return [p[3], p[4], 0.0, 0.0, 0.0]
    if model == "OPENCV":
        return [p[4], p[5], p[6], p[7], 0.0]
    raise ValueError(f"unsupported COLMAP camera model {model}")


def load_colmap_scene(
    dataset_dir: str | Path,
    model_subdir: str = "sparse/0",
    images_subdir: str = "images",
    downscale: int = 1,
    max_images: int | None = None,
    test_every: int = 8,
    undistort: bool = True,
) -> SceneData:
    """Load a COLMAP dataset directory into a SceneData (images + cameras + points)."""
    from PIL import Image as PILImage

    root = Path(dataset_dir)
    model = read_model(root / model_subdir)

    images: list[torch.Tensor] = []
    cameras: list[Camera] = []
    point_visibility: list[torch.Tensor] = []
    view_names: list[str] = []
    point_index = {int(point_id): i for i, point_id in enumerate(model.point_ids.tolist())}
    ordered = sorted(model.images.items(), key=lambda kv: kv[1]["name"])
    if max_images is not None and max_images <= 0:
        raise ValueError("max_images must be positive")
    if max_images is not None and max_images < len(ordered):
        sample = torch.linspace(0, len(ordered) - 1, max_images).round().long().unique()
        ordered = [ordered[int(index)] for index in sample]
    for _, meta in ordered:
        cam = model.cameras[meta["camera_id"]]
        fx, fy, cx, cy = _intrinsics(cam)
        img_path = root / images_subdir / meta["name"]
        pil = PILImage.open(img_path).convert("RGB")
        if downscale > 1:
            source_width, source_height = pil.size
            width, height = source_width // downscale, source_height // downscale
            pil = pil.resize((width, height), PILImage.Resampling.LANCZOS)
            sx, sy = width / source_width, height / source_height
            fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy
        img = torch.from_numpy(np.array(pil, dtype=np.float32, copy=True) / 255.0)
        distortion = _distortion(cam)
        if undistort and distortion:
            from rtgs.data.calibrated import _undistort

            img = _undistort(img, fx, fy, cx, cy, distortion)
        r = _qvec_to_rotmat(meta["qvec"])
        cameras.append(
            Camera(
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=pil.width,
                height=pil.height,
                R=torch.from_numpy(r).float(),
                t=torch.from_numpy(meta["tvec"]).float(),
            )
        )
        images.append(img)
        view_names.append(Path(meta["name"]).stem)
        visible = sorted(
            {point_index[pid] for pid in meta.get("point3d_ids", []) if pid in point_index}
        )
        point_visibility.append(torch.tensor(visible, dtype=torch.long))
    test_indices = list(range(test_every - 1, len(images), test_every)) if test_every > 0 else []
    test_set = set(test_indices)
    scene = SceneData(
        images=images,
        cameras=cameras,
        view_names=view_names,
        points=model.points,
        point_visibility=point_visibility,
        train_indices=[index for index in range(len(images)) if index not in test_set],
        test_indices=test_indices,
        name=root.name,
    )
    scene.validate()
    return scene
