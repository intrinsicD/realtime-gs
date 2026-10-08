"""Render an analytic sphere / triaxial ellipsoid into the calibrated-scene layout.

Plan: docs/TASK_jet2_synthetic_sphere_ellipsoid.md (P2).

    PYOPENGL_PLATFORM=egl .venv/bin/python scripts/experiments/prepare_analytic_surfaces.py \
        sphere   # or: ellipsoid
    .venv/bin/python scripts/experiments/prepare_analytic_surfaces.py reference

``reference`` writes ``source/reference.json`` + ``source/PROVENANCE.json`` (evaluation-only ground
truth; reconstruction may not open ``source/``): sphere LBO eigenvalues ``l(l+1)`` with
multiplicities, ellipsoid lambda_0..lambda_64 of the resolved P1 surface-FEM reference of
ContinuousLevelSetLBOGS E12 (level 7, relative uncertainty 3.1e-4), copied with its file hash.

Surface ``F(x) = sum_k x_k^2 / a_k^2 = 1`` with ``a = (1, 1, 1)`` (sphere) or ``(1, 0.75, 0.5)``
(ellipsoid), centred at the origin, scene units = surface units. Rendered as an icosphere of
subdivision 6 (40962 vertices) scaled by ``a``, with smooth (analytic) vertex normals and a
deterministic procedural vertex colour so photometry constrains geometry beyond shading. Cameras,
lights, image size and file layout are those of ``prepare_tosca_cat0.py`` (imported, not copied):
160 Fibonacci cameras at 2.2 x bbox diagonal, 512^2, yfov 0.5 rad, white background, masks from
the depth buffer. The render mesh is a rendering device only; evaluation uses the analytic F.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent))
cat = importlib.import_module("prepare_tosca_cat0")

ROOT = cat.ROOT
AXES = {"sphere": (1.0, 1.0, 1.0), "ellipsoid": (1.0, 0.75, 0.5)}
SUBDIVISIONS = 6


def procedural_colour(unit: np.ndarray) -> np.ndarray:
    """Smooth multi-frequency RGB pattern on the unit-sphere parameter (uint8)."""
    x, y, z = unit.T
    r = 0.5 + 0.22 * np.sin(5 * x + 1.3) + 0.18 * np.sin(11 * y - 0.4) + 0.08 * np.sin(23 * z)
    g = 0.5 + 0.22 * np.sin(6 * y + 0.7) + 0.18 * np.sin(13 * z + 2.1) + 0.08 * np.sin(19 * x)
    b = 0.5 + 0.22 * np.sin(7 * z - 1.1) + 0.18 * np.sin(9 * x + 0.5) + 0.08 * np.sin(29 * y)
    return np.round(255 * np.clip(np.stack([r, g, b], 1), 0, 1)).astype(np.uint8)


def analytic_mesh(axes: tuple[float, float, float]):
    import trimesh

    unit = trimesh.creation.icosphere(subdivisions=SUBDIVISIONS, radius=1.0)
    a = np.asarray(axes)
    vertices = unit.vertices * a
    normals = unit.vertices / a  # grad F / 2 at x = a u is u / a
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    mesh = trimesh.Trimesh(
        vertices,
        unit.faces,
        vertex_colors=procedural_colour(unit.vertices),
        vertex_normals=normals,
        process=False,
    )
    return mesh


def main(surface: str) -> None:
    import pyrender
    from PIL import Image

    axes = AXES[surface]
    out = ROOT / f"dataset/external/analytic_{surface}"
    if out.exists():
        raise FileExistsError(f"{out} exists; renders are never overwritten")
    mesh = analytic_mesh(axes)
    centroid = np.zeros(3)
    diagonal = float(2 * np.linalg.norm(axes))
    radius = cat.RADIUS_FACTOR * diagonal
    for name in ("rgb", "mask", "views"):
        (out / name).mkdir(parents=True)

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0], metallicFactor=0.0, roughnessFactor=1.0
    )
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[cat.AMBIENT] * 3)
    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    for primitive in render_mesh.primitives:
        if primitive.color_0 is None:
            raise RuntimeError("vertex colours were not attached")
        primitive.material = material
    scene.add(render_mesh)
    for direction, intensity in cat.LIGHTS:
        d = np.asarray(direction, dtype=np.float64)
        d /= np.linalg.norm(d)
        z = -d
        x = np.cross([0.0, 0.0, 1.0] if abs(z[2]) < 0.99 else [0.0, 1.0, 0.0], z)
        x /= np.linalg.norm(x)
        pose = np.eye(4)
        pose[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=intensity), pose=pose)
    camera = pyrender.PerspectiveCamera(
        yfov=cat.YFOV, aspectRatio=1.0, znear=0.1 * radius, zfar=4 * radius
    )
    camera_node = scene.add(camera)
    size = cat.SIZE
    renderer = pyrender.OffscreenRenderer(size, size)
    focal = (size / 2) / math.tan(cat.YFOV / 2)
    cameras, views, coverage, inside = [], [], [], []
    for index, direction in enumerate(cat.fibonacci(cat.N_VIEWS)):
        view_id = f"C{index + 1:04d}"
        eye = centroid + radius * direction
        rotation = cat.look_at(eye, centroid)
        pose = np.eye(4)
        pose[:3, :3] = rotation.T @ np.diag([1.0, -1.0, -1.0])
        pose[:3, 3] = eye
        scene.set_pose(camera_node, pose)
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        mask = depth > 0
        Image.fromarray(color[..., :3]).save(
            out / "rgb" / f"{view_id}.jpg", quality=100, subsampling=0
        )
        Image.fromarray((mask * 255).astype(np.uint8)).save(out / "mask" / f"mask_{view_id}.png")
        view = np.eye(4)
        view[:3, :3] = rotation
        view[:3, 3] = -rotation @ eye
        cam = (mesh.vertices - eye) @ rotation.T
        front = cam[:, 2] > 0
        u = focal * cam[front, 0] / cam[front, 2] + size / 2
        v = focal * cam[front, 1] / cam[front, 2] + size / 2
        ui = np.clip(u.astype(int), 0, size - 1)
        vi = np.clip(v.astype(int), 0, size - 1)
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
                    "resolution": [size, size],
                    "camera_matrix": [focal, 0.0, size / 2 - 0.5, 0.0, focal, size / 2 - 0.5]
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
    (out / "calibration_dome.json").write_text(json.dumps({"cameras": cameras}, indent=1) + "\n")
    (out / "views/manifest.json").write_text(
        json.dumps({"schema": "rtgs-view-list", "views": views}, indent=1) + "\n"
    )
    receipt = {
        "schema_version": 1,
        "surface": surface,
        "implicit": "F(x) = sum_k x_k^2 / a_k^2 = 1, centred at the origin",
        "axes": list(axes),
        "render_mesh": f"trimesh icosphere subdivision {SUBDIVISIONS} scaled by axes, "
        "analytic vertex normals",
        "colour": "procedural_colour(unit-sphere parameter), see source",
        "renderer": f"pyrender {pyrender.__version__} (EGL offscreen)",
        "bbox_diagonal": diagonal,
        "camera_radius": radius,
        "views": cat.N_VIEWS,
        "size": [size, size],
        "yfov_rad": cat.YFOV,
        "focal_px": focal,
        "lights": [{"direction": list(d), "intensity": i} for d, i in cat.LIGHTS],
        "ambient": cat.AMBIENT,
        "background": [1.0, 1.0, 1.0],
        "mask_rule": "depth > 0",
        "rgb_encoding": "JPEG quality 100, 4:4:4",
        "mask_coverage": {"min": min(coverage), "max": max(coverage)},
        "vertex_on_silhouette_fraction_min": min(inside),
        "source_code_sha256": cat.sha256(Path(__file__)),
        "renderer_module_sha256": cat.sha256(Path(cat.__file__)),
    }
    (out / "render_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1))


E12_EVIDENCE = Path.home() / "Documents/ContinuousLevelSetLBOGS/results/e12/evidence.json"


def reference() -> None:
    """Evaluation-only ground truth next to each render (never overwritten)."""
    evidence = json.loads(E12_EVIDENCE.read_text())["reference"]
    finest = evidence["refinements"][-1]
    sphere = [float(d * (d + 1)) for d in range(8) for _ in range(2 * d + 1)]
    tables = {
        "sphere": {"eigenvalues": sphere, "area": 4 * math.pi, "source": "analytic l(l+1), R = 1"},
        "ellipsoid": {
            "eigenvalues": finest["eigenvalues"],
            "area": finest["area"],
            "relative_uncertainty": evidence["eigen_uncertainty"],
            "source": f"{E12_EVIDENCE} reference.refinements[-1] (P1 FEM level "
            f"{finest['level']}, {finest['vertices']} vertices), sha256 {cat.sha256(E12_EVIDENCE)}",
        },
    }
    for surface, table in tables.items():
        source = ROOT / f"dataset/external/analytic_{surface}/source"
        source.mkdir()
        payload = {"surface": surface, "axes": list(AXES[surface]), **table}
        (source / "reference.json").write_text(json.dumps(payload, indent=1) + "\n")
        (source / "PROVENANCE.json").write_text(
            json.dumps(
                {
                    "files": {"reference.json": {"sha256": cat.sha256(source / "reference.json")}},
                    "boundary": "Evaluation only; reconstruction never opens source/.",
                    "source_code_sha256": cat.sha256(Path(__file__)),
                },
                indent=1,
            )
            + "\n"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in (*AXES, "reference"):
        raise SystemExit(f"usage: {sys.argv[0]} {{{','.join(AXES)},reference}}")
    reference() if sys.argv[1] == "reference" else main(sys.argv[1])
