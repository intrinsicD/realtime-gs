"""COLMAP sparse-reconstruction loading (text and binary formats).

Reads ``cameras``, ``images``, ``points3D`` from a COLMAP model directory (typically
``<dataset>/sparse/0``) and the corresponding image files, producing a
:class:`~rtgs.data.scene.SceneData`. Only PINHOLE / SIMPLE_PINHOLE / (SIMPLE_)RADIAL
camera models are supported (radial distortion is ignored with a warning — undistort
with ``colmap image_undistorter`` for correct results).
"""

from __future__ import annotations

import struct
import warnings
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
    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        images[int(parts[0])] = {
            "qvec": np.array([float(p) for p in parts[1:5]]),
            "tvec": np.array([float(p) for p in parts[5:8]]),
            "camera_id": int(parts[8]),
            "name": parts[9],
        }
    return images


def read_points3d_text(path: Path) -> torch.Tensor:
    """Parse points3D.txt into an (M, 3) tensor."""
    pts = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return torch.tensor(pts, dtype=torch.float32) if pts else torch.zeros(0, 3)


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
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name.decode("utf-8"),
            }
    return images


def read_points3d_binary(path: Path) -> torch.Tensor:
    """Parse points3D.bin into an (M, 3) tensor."""
    pts = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(8)  # point id
            xyz = struct.unpack("<3d", f.read(24))
            f.read(3)  # rgb
            f.read(8)  # error
            (track_len,) = struct.unpack("<Q", f.read(8))
            f.read(8 * track_len)
            pts.append(xyz)
    return torch.tensor(pts, dtype=torch.float32) if pts else torch.zeros(0, 3)


def read_model(model_dir: str | Path) -> ColmapModel:
    """Read a COLMAP model directory, preferring binary files when both exist."""
    d = Path(model_dir)
    if (d / "cameras.bin").exists():
        cams = read_cameras_binary(d / "cameras.bin")
        images = read_images_binary(d / "images.bin")
        points = read_points3d_binary(d / "points3D.bin")
    elif (d / "cameras.txt").exists():
        cams = read_cameras_text(d / "cameras.txt")
        images = read_images_text(d / "images.txt")
        points = read_points3d_text(d / "points3D.txt")
    else:
        raise FileNotFoundError(f"no COLMAP model (cameras.bin/.txt) in {d}")
    return ColmapModel(cameras=cams, images=images, points=points)


def _intrinsics(cam: dict) -> tuple[float, float, float, float]:
    model, p = cam["model"], cam["params"]
    if model == "SIMPLE_PINHOLE":
        return p[0], p[0], p[1], p[2]
    if model == "PINHOLE":
        return p[0], p[1], p[2], p[3]
    if model in ("SIMPLE_RADIAL", "RADIAL", "OPENCV"):
        warnings.warn(
            f"COLMAP model {model}: ignoring distortion parameters; "
            "run `colmap image_undistorter` for correct geometry",
            stacklevel=2,
        )
        if model == "SIMPLE_RADIAL" or model == "RADIAL":
            return p[0], p[0], p[1], p[2]
        return p[0], p[1], p[2], p[3]
    raise ValueError(f"unsupported COLMAP camera model {model}")


def load_colmap_scene(
    dataset_dir: str | Path,
    model_subdir: str = "sparse/0",
    images_subdir: str = "images",
    downscale: int = 1,
    max_images: int | None = None,
) -> SceneData:
    """Load a COLMAP dataset directory into a SceneData (images + cameras + points)."""
    from PIL import Image as PILImage

    root = Path(dataset_dir)
    model = read_model(root / model_subdir)

    images: list[torch.Tensor] = []
    cameras: list[Camera] = []
    for _, meta in sorted(model.images.items(), key=lambda kv: kv[1]["name"]):
        if max_images is not None and len(images) >= max_images:
            break
        cam = model.cameras[meta["camera_id"]]
        fx, fy, cx, cy = _intrinsics(cam)
        img_path = root / images_subdir / meta["name"]
        pil = PILImage.open(img_path).convert("RGB")
        if downscale > 1:
            pil = pil.resize((pil.width // downscale, pil.height // downscale), PILImage.LANCZOS)
            fx, fy, cx, cy = fx / downscale, fy / downscale, cx / downscale, cy / downscale
        img = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0)
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
    scene = SceneData(images=images, cameras=cameras, points=model.points, name=root.name)
    scene.validate()
    return scene
