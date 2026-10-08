"""Render TOSCA cat0 into the calibrated-scene layout (task 20261007_jet_consistency_prior).

Task id: 20261007_jet_consistency_prior_tosca_cat0.

    PYOPENGL_PLATFORM=egl .venv/bin/python scripts/experiments/prepare_tosca_cat0.py

Mesh: ``cat0.obj`` (TOSCA hires) with per-vertex colours averaged from the face-corner colours
of ``cat0_colored_v1.ply`` (same faces, stored in the frame (x, -z, y)); both are copied under
``dataset/external/tosca_cat0/source/`` with a provenance file. 160 cameras on a Fibonacci sphere
(polar axis +z) of radius 2.2 x bbox diagonal around the mesh centroid, looking at it, 512x512,
yfov 0.5 rad, three fixed world directional lights plus a weak ambient term, white background,
masks from the depth buffer. Scene units are mesh units. Writes ``rgb/C####.jpg`` (quality 100,
4:4:4; the contract's canonical RGB pattern is JPEG), ``mask/mask_C####.png``,
``calibration_dome.json`` (OpenCV world-to-camera view matrix, integer-centred principal point),
``views/manifest.json`` (view list bound by the data seal) and ``render_receipt.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path.home() / "Dropbox/Work/Datasets/obj/tosca_hires"
OUT = ROOT / "dataset/external/tosca_cat0"
N_VIEWS, SIZE, YFOV, RADIUS_FACTOR = 160, 512, 0.5, 2.2
# Fixed world light directions (direction the light travels) and intensities.
LIGHTS = [((-0.4, -0.3, -1.0), 2.0), ((1.0, 0.6, -0.2), 1.2), ((-0.6, 1.0, 0.5), 0.8)]
AMBIENT = 0.25


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fibonacci(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    z = 1 - (2 * i + 1) / n
    r = np.sqrt(1 - z * z)
    phi = i * math.pi * (3 - math.sqrt(5))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """OpenCV world-to-camera rotation (rows: right, down, forward)."""
    forward = (target - eye) / np.linalg.norm(target - eye)
    up = np.array([0.0, 0.0, 1.0]) if abs(forward[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward])


def coloured_mesh():
    import trimesh

    mesh = trimesh.load(SOURCE / "cat0.obj", process=False)
    coloured = trimesh.load(SOURCE / "cat0_colored_v1.ply", process=False)
    v = mesh.vertices
    in_ply_frame = np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)[mesh.faces]
    error = np.abs(in_ply_frame - coloured.vertices[coloured.faces]).max()
    if mesh.faces.shape != coloured.faces.shape or error > 1e-3:
        raise RuntimeError(f"coloured PLY does not match cat0.obj face-for-face ({error})")
    corners = coloured.visual.vertex_colors[:, :3].astype(np.float64)[coloured.faces.reshape(-1)]
    total = np.zeros((len(v), 3))
    count = np.zeros(len(v))
    np.add.at(total, mesh.faces.reshape(-1), corners)
    np.add.at(count, mesh.faces.reshape(-1), 1)
    colours = np.round(total / count[:, None]).astype(np.uint8)
    out = trimesh.Trimesh(v, mesh.faces, vertex_colors=colours, process=False)
    return out, {"corner_frame_error": float(error), "isolated_vertices": int((count == 0).sum())}


def main() -> None:
    import pyrender
    from PIL import Image

    mesh, check = coloured_mesh()
    centroid = mesh.centroid.astype(np.float64)
    diagonal = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    radius = RADIUS_FACTOR * diagonal
    for name in ("rgb", "mask", "views", "source"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    for name in ("cat0.obj", "cat0_colored_v1.ply"):
        shutil.copyfile(SOURCE / name, OUT / "source" / name)

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0], metallicFactor=0.0, roughnessFactor=1.0
    )
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[AMBIENT] * 3)
    # Passing material= to from_trimesh drops vertex colours; set it on the primitive instead.
    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    for primitive in render_mesh.primitives:
        if primitive.color_0 is None:
            raise RuntimeError("vertex colours were not attached")
        primitive.material = material
    scene.add(render_mesh)
    for direction, intensity in LIGHTS:
        d = np.asarray(direction, dtype=np.float64)
        d /= np.linalg.norm(d)
        # pyrender lights shine along their local -z axis.
        z = -d
        x = np.cross([0.0, 0.0, 1.0] if abs(z[2]) < 0.99 else [0.0, 1.0, 0.0], z)
        x /= np.linalg.norm(x)
        pose = np.eye(4)
        pose[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=intensity), pose=pose)
    camera = pyrender.PerspectiveCamera(yfov=YFOV, aspectRatio=1.0, znear=1.0, zfar=4 * radius)
    camera_node = scene.add(camera)
    renderer = pyrender.OffscreenRenderer(SIZE, SIZE)

    focal = (SIZE / 2) / math.tan(YFOV / 2)
    cameras, views, coverage, inside = [], [], [], []
    vertices = mesh.vertices
    for index, direction in enumerate(fibonacci(N_VIEWS)):
        view_id = f"C{index + 1:04d}"
        eye = centroid + radius * direction
        rotation = look_at(eye, centroid)
        pose = np.eye(4)
        pose[:3, :3] = rotation.T @ np.diag([1.0, -1.0, -1.0])  # OpenCV -> OpenGL axes
        pose[:3, 3] = eye
        scene.set_pose(camera_node, pose)
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        mask = depth > 0
        Image.fromarray(color[..., :3]).save(
            OUT / "rgb" / f"{view_id}.jpg", quality=100, subsampling=0
        )
        Image.fromarray((mask * 255).astype(np.uint8)).save(OUT / "mask" / f"mask_{view_id}.png")
        view = np.eye(4)
        view[:3, :3] = rotation
        view[:3, 3] = -rotation @ eye
        # Silhouette check: projected vertices must land on the depth mask (1 px tolerance).
        cam = (vertices - eye) @ rotation.T
        u = focal * cam[:, 0] / cam[:, 2] + SIZE / 2
        v = focal * cam[:, 1] / cam[:, 2] + SIZE / 2
        ui = np.clip(u.astype(int), 0, SIZE - 1)
        vi = np.clip(v.astype(int), 0, SIZE - 1)
        padded = np.pad(mask, 1)
        near = np.zeros_like(ui, dtype=bool)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                near |= padded[vi + dy, ui + dx]
        inside.append(float(near.mean()))
        coverage.append(float(mask.mean()))
        cameras.append(
            {
                "camera_id": view_id,
                "intrinsics": {
                    "resolution": [SIZE, SIZE],
                    "camera_matrix": [focal, 0.0, SIZE / 2 - 0.5, 0.0, focal, SIZE / 2 - 0.5]
                    + [0.0, 0.0, 1.0],
                    "distortion_coefficients": [0.0] * 5,
                },
                "extrinsics": {"view_matrix": view.reshape(-1).tolist()},
            }
        )
        views.append({"view_id": view_id, "path": f"../rgb/{view_id}.jpg"})
    renderer.delete()
    if min(inside) < 0.999:
        raise RuntimeError(f"calibration/silhouette mismatch: min inside fraction {min(inside)}")
    (OUT / "calibration_dome.json").write_text(json.dumps({"cameras": cameras}, indent=1) + "\n")
    (OUT / "views/manifest.json").write_text(
        json.dumps(
            {
                "schema": "rtgs-view-list",
                "note": "RGB/mask-only scene: no compact 2D Gaussian fields exist; this view list "
                "is the manifest the experiment contract binds in the data seal.",
                "views": views,
            },
            indent=1,
        )
        + "\n"
    )
    provenance = {
        "dataset": "TOSCA high-resolution, cat0 (Bronstein, Bronstein, Kimmel)",
        "copied_from": str(SOURCE),
        "files": {
            name: {
                "sha256": sha256(OUT / "source" / name),
                "bytes": (OUT / "source" / name).stat().st_size,
            }
            for name in ("cat0.obj", "cat0_colored_v1.ply")
        },
        "colour_rule": "per-vertex mean of the PLY face-corner colours (PLY frame = (x, -z, y) "
        "of the OBJ, identical face order)",
        "colour_check": check,
        "boundary": "Only this preparation and the post-training geometry evaluation read the "
        "mesh; reconstruction sees RGB, masks and calibration of the training views only.",
    }
    (OUT / "source/PROVENANCE.json").write_text(json.dumps(provenance, indent=1) + "\n")
    receipt = {
        "schema_version": 1,
        "task_id": "20261007_jet_consistency_prior_tosca_cat0",
        "renderer": f"pyrender {pyrender.__version__} (EGL offscreen)",
        "mesh_units": True,
        "centroid": centroid.tolist(),
        "bbox_diagonal": diagonal,
        "camera_radius": radius,
        "views": N_VIEWS,
        "size": [SIZE, SIZE],
        "yfov_rad": YFOV,
        "focal_px": focal,
        "lights": [{"direction": list(d), "intensity": i} for d, i in LIGHTS],
        "ambient": AMBIENT,
        "background": [1.0, 1.0, 1.0],
        "mask_rule": "depth > 0",
        "rgb_encoding": "JPEG quality 100, 4:4:4",
        "mask_coverage": {"min": min(coverage), "max": max(coverage)},
        "vertex_on_silhouette_fraction_min": min(inside),
        "source_code_sha256": sha256(Path(__file__)),
    }
    (OUT / "render_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
