#!/usr/bin/env python3
"""Prepare sealed 2D observations for RTGS-016 outside reconstruction workers.

This program may read images, masks and the downloaded reference. Reconstruction
receives only its separate compact datasets. No 3D comparison is executed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "20260906_tomography_source_constraints_haelyn_dome"
TASK_PATH = ROOT / "experiments/tasks" / f"{TASK_ID}.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _task() -> dict:
    task = json.loads(TASK_PATH.read_text())
    if task["task_id"] != TASK_ID or task["status"] != "draft":
        raise ValueError("input production is allowed only before the protocol is ready")
    return task


def reference_views(task: dict) -> None:
    """Render declared pinhole cameras, preserving all reference SH coefficients."""
    import numpy as np
    import torch
    from PIL import Image

    from rtgs.core.camera import Camera
    from rtgs.core.gaussians3d import Gaussians3D
    from rtgs.render.gsplat_backend import GsplatRasterizer

    cfg = task["frozen_configuration"]["source_reference"]
    path = ROOT / cfg["path"]
    if sha(path) != cfg["sha256"]:
        raise ValueError("reference Gaussian bytes changed")
    parent = path.parent
    alignment = json.loads((parent / "import_alignment.json").read_text())
    rotation = np.array(alignment["rotation"])
    translation = np.array(alignment["translation"])
    scale = alignment["scale"]
    camera_cfg = cfg["cameras"]
    # These are acquisition geometry, not a scene bound passed to the lifter.
    target = np.array(camera_cfg["target_published"])
    directory = parent / "prepared"
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite prepared source views: {directory}")
    (directory / "rgb").mkdir(parents=True)
    (directory / "mask").mkdir()
    model = Gaussians3D.load_ply(path).to("cuda")
    renderer = GsplatRasterizer(packed=False, antialiased=False)
    records = []
    inputs = []
    started = time.perf_counter()
    for index in range(camera_cfg["count"]):
        angle = math.radians(camera_cfg["azimuth_start_deg"] + 360 * index / camera_cfg["count"])
        elevation = math.radians(camera_cfg["elevation_deg"])
        radius = camera_cfg["radius_published"]
        eye = target + radius * np.array(
            [
                math.cos(elevation) * math.sin(angle),
                math.sin(elevation),
                math.cos(elevation) * math.cos(angle),
            ]
        )
        pub = Camera.look_at(
            torch.tensor(eye),
            torch.tensor(target),
            width=camera_cfg["width"],
            height=camera_cfg["height"],
            fov_x_deg=camera_cfg["fov_x_deg"],
        )
        cam_rotation = pub.R.numpy() @ rotation
        raw_eye = rotation.T @ (eye - translation) / scale
        camera = Camera(
            pub.fx,
            pub.fy,
            pub.cx,
            pub.cy,
            pub.width,
            pub.height,
            cam_rotation,
            -cam_rotation @ raw_eye,
        )
        view_id = f"C{index + 1:04d}"
        with torch.no_grad():
            result = renderer.render(
                model, camera.to("cuda"), background=torch.zeros(3, device="cuda")
            )
        if not torch.isfinite(result.color).all() or not torch.isfinite(result.alpha).all():
            raise ValueError(f"nonfinite reference view {view_id}")
        rgb = directory / "rgb" / f"{view_id}.jpg"
        mask = directory / "mask" / f"mask_{view_id}.png"
        Image.fromarray(
            (result.color.clamp(0, 1).cpu().numpy() * 255).round().astype("uint8")
        ).save(rgb, quality=100, subsampling=0)
        binary = (result.alpha.cpu().numpy() >= cfg["alpha_threshold"]).astype("uint8") * 255
        if not binary.any():
            raise ValueError(f"empty reference support in {view_id}")
        Image.fromarray(binary).save(mask)
        matrix = camera.K.numpy().copy()
        # The calibrated loader converts integer-center OpenCV intrinsics to +0.5.
        matrix[0, 2] -= 0.5
        matrix[1, 2] -= 0.5
        records.append(
            {
                "camera_id": view_id,
                "intrinsics": {
                    "resolution": [camera.width, camera.height],
                    "camera_matrix": matrix.flatten().tolist(),
                    "distortion_coefficients": [0.0] * 5,
                },
                "extrinsics": {"view_matrix": camera.viewmat.flatten().tolist()},
            }
        )
        inputs.extend(
            {"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha(p)}
            for p in (rgb, mask)
        )
        print(f"reference view {view_id} ready", flush=True)
    write_new(directory / "calibration_dome.json", {"cameras": records})
    write_new(
        directory / "render_receipt.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "reference": cfg,
            "alignment_sha256": sha(parent / "import_alignment.json"),
            "source_code_sha256": sha(Path(__file__)),
            "renderer": "gsplat 1.5.3 classic unpacked",
            "background": [0, 0, 0],
            "wall_seconds": time.perf_counter() - started,
            "claim_boundary": (
                "Offline known-representation observations; "
                "not acquired dome photos or reconstruction evidence."
            ),
            "files": inputs,
        },
    )


def convert_dataset(task: dict, dataset_id: str) -> None:
    """Fit once per image and freeze complete native fields, with optional alpha."""
    import torch
    from PIL import Image

    from rtgs.data.calibrated import _object_bounds, load_calibrated_scene
    from rtgs.data.compact_views import save_compact_view, write_compact_dataset_manifest
    from rtgs.image2gs.fit import FitConfig, fit_image
    from rtgs.image2gs.native_observation import native_gaussians_to_observation

    dataset = next(d for d in task["datasets"] if d["id"] == dataset_id)
    frame = ROOT / dataset["frame_path"]
    directory = (ROOT / dataset["compact_manifest"]).parent
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite compact inputs: {directory}")
    cfg = task["frozen_configuration"]["converter"]
    masked = dataset_id != "haelyn_unmasked"
    partition = task["splits"][dataset_id]
    ids = sorted(partition["train"] + partition["heldout"])
    first = frame / "rgb" / f"{ids[0]}.jpg"
    with Image.open(first) as image:
        downscale = math.ceil(max(image.size) / cfg["target_long_side"])
    kwargs = {
        k: cfg[k]
        for k in (
            "backend",
            "native_renderer",
            "n_gaussians",
            "max_gaussians",
            "adaptive_density",
            "iterations",
            "lr",
            "convergence_patience",
        )
    }
    config = FitConfig(**kwargs)
    config_digest = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
    calibration = ROOT / dataset["calibration"]
    calibration_digest = sha(calibration)
    # Both support conditions use the same camera-only bound. No mask-derived
    # center, reference point position, depth, or covariance crosses this seam.
    bound_scene = load_calibrated_scene(
        frame,
        calibration_path=calibration,
        downscale=downscale,
        view_ids=partition["train"],
        load_masks=False,
    )
    bounds = _object_bounds(bound_scene.cameras, None)
    del bound_scene
    directory.mkdir(parents=True)
    paths = []
    receipts = []
    source_files = [{"path": str(calibration.relative_to(ROOT)), "sha256": calibration_digest}]
    started = time.perf_counter()
    for index, view_id in enumerate(ids):
        scene = load_calibrated_scene(
            frame,
            calibration_path=calibration,
            downscale=downscale,
            view_ids=[view_id],
            load_masks=masked,
        )
        image = scene.images[0].to("cuda")
        mask = scene.masks[0].to("cuda") if masked else None
        tick = time.perf_counter()
        fitted, history = fit_image(image, config, seed=cfg["seed"] + index, mask=mask)
        torch.cuda.synchronize()
        fit_seconds = time.perf_counter() - tick
        field = native_gaussians_to_observation(
            fitted,
            canvas_size=image.shape[:2],
            view_id=view_id,
            producer_version="RTGS-016 frozen native converter",
            producer_source_digest=sha(Path(__file__)),
            fit_config_digest=config_digest,
        )
        rgb = frame / "rgb" / f"{view_id}.jpg"
        mask_path = frame / "mask" / f"mask_{view_id}.png"
        source_files.append({"path": str(rgb.relative_to(ROOT)), "sha256": sha(rgb)})
        extra = {}
        if masked:
            if mask is None:
                raise ValueError("masked preparation requires an actual source mask")
            source_files.append(
                {"path": str(mask_path.relative_to(ROOT)), "sha256": sha(mask_path)}
            )
            extra = {
                "alpha_crop": mask.cpu(),
                "source_mask_name": mask_path.name,
                "source_mask_sha256": sha(mask_path),
            }
        path = directory / f"{view_id}.rtgsv"
        byte_count = save_compact_view(
            path,
            field,
            scene.cameras[0],
            calibration_sha256=calibration_digest,
            source_rgb_name=rgb.name,
            source_rgb_sha256=sha(rgb),
            byte_cap=cfg["view_byte_cap"],
            **extra,
        )
        paths.append(path)
        receipts.append(
            {
                "view_id": view_id,
                "seed": cfg["seed"] + index,
                "gaussians": field.n,
                "bytes": byte_count,
                "mask_read": masked,
                "fit_seconds": fit_seconds,
                "finite": bool(torch.isfinite(fitted.xy).all()),
                "fit_history": history,
            }
        )
        print(f"{dataset_id} {view_id}: {field.n} Gaussians, {byte_count} bytes", flush=True)
        del scene, image, mask, fitted, field
    write_compact_dataset_manifest(
        directory,
        name=dataset_id,
        calibration_sha256=calibration_digest,
        view_paths=paths,
        bounds_hint=bounds,
        byte_cap=cfg["view_byte_cap"],
    )
    write_new(
        ROOT / dataset["production_manifest"],
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "dataset_id": dataset_id,
            "config": asdict(config),
            "config_sha256": config_digest,
            "source_code_sha256": sha(Path(__file__)),
            "source_files": source_files,
            "downscale": downscale,
            "mask_read": masked,
            "bounds_policy": "camera-only",
            "bounds_center": bounds[0].tolist(),
            "bounds_extent": bounds[1],
            "wall_seconds": time.perf_counter() - started,
            "views": receipts,
            "semantics": (
                "Native additive fit, rounded-AABB compact teacher; finite support differs "
                "from elliptical CUDA cutoff at boundary pixels. "
                "The teacher is frozen before reconstruction."
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["render-reference", "convert"])
    parser.add_argument("--dataset", choices=["haelyn_masked", "haelyn_unmasked", "dome_masked"])
    args = parser.parse_args()
    task = _task()
    if args.action == "render-reference":
        reference_views(task)
    elif args.dataset is None:
        parser.error("convert requires --dataset")
    else:
        convert_dataset(task, args.dataset)


if __name__ == "__main__":
    main()
