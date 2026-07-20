#!/usr/bin/env python3
"""Acquire seven native-resolution masked StructSplat teachers as a compact bundle.

This is an exploratory Stage-1 acquisition harness. RGB and masks are available only while each
fresh worker fits and evaluates its teacher. The parent process then assembles the final
``ReconstructionInputs`` solely from lossless compact archives and calibrated camera records.
No 3D lift, refinement, or viewer run is performed.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from compact_stage1_mask_screen import (
    effective_external_configs,
    observation_summary,
    raw_metrics,
    render_observation,
    requested_config,
    resize_hwc,
    save_rgb,
    sha256_file,
    structsplat_source_binding,
    support_union,
    tensor_hash,
)
from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.calibrated import _resize_image, _undistort
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.image2gs.fit import _crop_to_mask, fit_image

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008"
CALIBRATION = ROOT / "dataset/2025_03_07_stage_with_fabric/calibration_dome.json"
DEFAULT_OUT = ROOT / "runs/compact_masked_bundle_640_20260717"
VIEWS = ("C0001", "C0008", "C0014", "C0021", "C0026", "C0031", "C0039")
HELDOUT_VIEW = "C1004"
SEEDS = tuple(range(len(VIEWS)))
REQUIRED_TEACHER_MEMBERS = frozenset(
    {
        "metadata_utf8.npy",
        "means.npy",
        "log_scales.npy",
        "rotations.npy",
        "colors.npy",
        "amplitudes.npy",
    }
)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def repository_source_binding() -> dict[str, Any]:
    paths = [
        Path("benchmarks/compact_masked_bundle_acquisition.py"),
        Path("benchmarks/compact_stage1_mask_screen.py"),
        *sorted(Path("src/rtgs").rglob("*.py")),
    ]
    hashes = {path.as_posix(): sha256_file(ROOT / path) for path in paths}
    return {
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "files": hashes,
        "aggregate": canonical_hash(hashes),
    }


def calibration_records() -> dict[str, dict[str, Any]]:
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    records: dict[str, dict[str, Any]] = {}
    for record in payload["cameras"]:
        camera_id = str(record["camera_id"]).upper()
        if camera_id in records:
            raise RuntimeError(f"duplicate calibration camera {camera_id}")
        records[camera_id] = record
    return records


def camera_for_view(view_id: str) -> tuple[Camera, list[float]]:
    record = calibration_records()[view_id]
    intrinsics = record["intrinsics"]
    matrix = intrinsics["camera_matrix"]
    width, height = (int(value) for value in intrinsics["resolution"])
    view = torch.tensor(record["extrinsics"]["view_matrix"], dtype=torch.float32).reshape(4, 4)
    return (
        Camera(
            fx=float(matrix[0]),
            fy=float(matrix[4]),
            cx=float(matrix[2]) + 0.5,
            cy=float(matrix[5]) + 0.5,
            width=width,
            height=height,
            R=view[:3, :3],
            t=view[:3, 3],
        ),
        [float(value) for value in intrinsics.get("distortion_coefficients", [])],
    )


def camera_record(camera: Camera) -> dict[str, Any]:
    return {
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "width": camera.width,
        "height": camera.height,
        "R": camera.R.reshape(-1).tolist(),
        "t": camera.t.tolist(),
    }


def camera_from_record(record: dict[str, Any]) -> Camera:
    return Camera(
        fx=float(record["fx"]),
        fy=float(record["fy"]),
        cx=float(record["cx"]),
        cy=float(record["cy"]),
        width=int(record["width"]),
        height=int(record["height"]),
        R=torch.tensor(record["R"], dtype=torch.float32).reshape(3, 3),
        t=torch.tensor(record["t"], dtype=torch.float32),
    )


def view_paths(view_id: str) -> tuple[Path, Path]:
    return SCENE / f"rgb/{view_id}.jpg", SCENE / f"mask/mask_{view_id}.png"


def prepare_view(
    view_id: str,
) -> tuple[Camera, list[float], torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
    camera, distortion = camera_for_view(view_id)
    rgb_path, mask_path = view_paths(view_id)
    with Image.open(rgb_path) as source:
        native_size = source.size
    if native_size != (camera.width, camera.height):
        raise RuntimeError(f"{view_id} native size {native_size} differs from calibration")
    rgb = _undistort(
        _resize_image(rgb_path, camera.width, camera.height),
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        distortion,
    )
    foreground = (
        _undistort(
            _resize_image(mask_path, camera.width, camera.height, mask=True),
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            distortion,
            mask=True,
        )
        > 0.5
    )
    crop, _, offset = _crop_to_mask(rgb, foreground)
    fit_window = (int(offset[0]), int(offset[1]), int(crop.shape[1]), int(crop.shape[0]))
    return camera, distortion, rgb, foreground, fit_window


def plan_view(view_id: str, seed: int) -> dict[str, Any]:
    rgb_path, mask_path = view_paths(view_id)
    camera, distortion, rgb, foreground, fit_window = prepare_view(view_id)
    config = requested_config(n_start=640, n_max=640, iterations=100)
    return {
        "view_id": view_id,
        "seed": seed,
        "rgb": {"path": rgb_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(rgb_path)},
        "mask": {
            "path": mask_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(mask_path),
        },
        "camera": camera_record(camera),
        "distortion": distortion,
        "undistorted_rgb_tensor_sha256": tensor_hash(rgb),
        "undistorted_mask_tensor_sha256": tensor_hash(foreground),
        "foreground_pixels": int(foreground.sum()),
        "fit_window": list(fit_window),
        "effective_structsplat": effective_external_configs(config, fit_window[3], fit_window[2]),
    }


def validate_plan_view(record: dict[str, Any]) -> tuple[Camera, torch.Tensor, torch.Tensor]:
    view_id = record["view_id"]
    rgb_path, mask_path = view_paths(view_id)
    if sha256_file(rgb_path) != record["rgb"]["sha256"]:
        raise RuntimeError(f"{view_id} RGB changed after plan")
    if sha256_file(mask_path) != record["mask"]["sha256"]:
        raise RuntimeError(f"{view_id} mask changed after plan")
    camera, distortion, rgb, foreground, fit_window = prepare_view(view_id)
    checks = {
        "camera": camera_record(camera),
        "distortion": distortion,
        "undistorted_rgb_tensor_sha256": tensor_hash(rgb),
        "undistorted_mask_tensor_sha256": tensor_hash(foreground),
        "foreground_pixels": int(foreground.sum()),
        "fit_window": list(fit_window),
    }
    if any(record[key] != value for key, value in checks.items()):
        raise RuntimeError(f"{view_id} calibrated preprocessing differs from plan")
    return camera, rgb, foreground


def extension_record() -> dict[str, str]:
    import structsplat.cuda_render as cuda_render

    extension = getattr(cuda_render, "_EXT", None)
    if extension is None:
        raise RuntimeError("StructSplat CUDA extension did not load")
    path = Path(extension.__file__).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def teacher_members(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        names = sorted(item.filename for item in archive.infolist())
    if frozenset(names) != REQUIRED_TEACHER_MEMBERS:
        raise RuntimeError(f"unexpected teacher NPZ members: {names}")
    return names


def worker(plan_path: Path, plan_sha256: str, index: int, out: Path) -> int:
    record_path = out / "acquisition" / f"{index:04d}.json"
    teacher_path = out / "acquisition" / f"{index:04d}.teacher.npz"
    preview_path = out / "previews" / f"{index:04d}_{VIEWS[index]}.png"
    try:
        if sha256_file(plan_path) != plan_sha256:
            raise RuntimeError("plan hash changed before worker")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        source_before = repository_source_binding()
        external_before = structsplat_source_binding()
        if source_before != plan["repository"]:
            raise RuntimeError("realtime-gs source changed before worker")
        if external_before != plan["external_structsplat"]:
            raise RuntimeError("StructSplat source changed before worker")
        if sha256_file(CALIBRATION) != plan["inputs"]["calibration_sha256"]:
            raise RuntimeError("calibration changed before worker")

        planned = plan["views"][index]
        if planned["view_id"] != VIEWS[index] or planned["seed"] != SEEDS[index]:
            raise RuntimeError("worker index differs from frozen view/seed order")
        camera, rgb, foreground = validate_plan_view(planned)
        config = requested_config(n_start=640, n_max=640, iterations=100)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        observations: list[GaussianObservationField] = []
        started = time.perf_counter()
        _, history = fit_image(
            rgb.to("cuda"),
            config,
            seed=planned["seed"],
            mask=foreground.to("cuda"),
            observation_callback=observations.append,
            observation_view_id=planned["view_id"],
        )
        wall_seconds = time.perf_counter() - started
        if len(observations) != 1:
            raise RuntimeError("fit did not export exactly one observation")
        teacher = observations[0].to("cpu")
        expected_digest = planned["effective_structsplat"]["fit_config_digest"]
        if not (
            teacher.provider == "structsplat"
            and teacher.blend_mode == "normalized"
            and teacher.view_id == planned["view_id"]
            and teacher.width == 5328
            and teacher.height == 4608
            and list(teacher.fit_window) == planned["fit_window"]
            and teacher.n_init == 640
            and teacher.n == 640
            and teacher.fit_config_digest == expected_digest
            and teacher.producer_source_digest == external_before["provider_source_digest"]
        ):
            raise RuntimeError("exported teacher differs from frozen contract")

        rendered = render_observation(teacher, device=torch.device("cuda"))
        fit_x, fit_y, fit_width, fit_height = teacher.fit_window
        target = rgb[fit_y : fit_y + fit_height, fit_x : fit_x + fit_width]
        mask_crop = foreground[fit_y : fit_y + fit_height, fit_x : fit_x + fit_width]
        masked_target = target * mask_crop[..., None]
        coverage = torch.from_numpy(support_union(teacher))
        generator = torch.Generator().manual_seed(17072026 + index)
        sample_count = 4096
        sample_x = torch.randint(fit_width, (sample_count,), generator=generator)
        sample_y = torch.randint(fit_height, (sample_count,), generator=generator)
        sample_xy = torch.stack([sample_x + fit_x, sample_y + fit_y], dim=-1).float() + 0.5
        queried = teacher.query(sample_xy, component_chunk=128).color
        parity = (queried - rendered[sample_y, sample_x]).abs()
        metrics = {
            "foreground_rgb": raw_metrics(rendered[mask_crop], target[mask_crop]),
            "masked_fit_crop": raw_metrics(rendered, masked_target),
            "foreground_pixels": int(mask_crop.sum()),
            "foreground_hole_fraction": float((~coverage[mask_crop]).float().mean()),
            "archive_query_cuda_raster_parity": {
                "sample_count": sample_count,
                "max_abs_error": float(parity.max()),
                "mean_abs_error": float(parity.mean()),
            },
        }
        target_preview = resize_hwc(masked_target, height=240, width=640)
        render_preview = resize_hwc(rendered, height=240, width=640)
        combined = torch.cat([target_preview, render_preview], dim=1)
        save_rgb(preview_path, combined)

        teacher.save_npz(teacher_path)
        teacher_sha256 = sha256_file(teacher_path)
        members = teacher_members(teacher_path)
        reloaded = GaussianObservationField.load_npz(teacher_path, strict=True)
        if observation_summary(reloaded) != observation_summary(teacher):
            raise RuntimeError("strict teacher reload changed semantics")
        external_after = structsplat_source_binding()
        source_after = repository_source_binding()
        validate_plan_view(planned)
        if source_after != source_before or external_after != external_before:
            raise RuntimeError("source changed during worker")
        if sha256_file(CALIBRATION) != plan["inputs"]["calibration_sha256"]:
            raise RuntimeError("calibration changed during worker")
        write_json_exclusive(
            record_path,
            {
                "artifact_type": "compact_masked_teacher_acquisition_view_v1",
                "status": "PASS",
                "plan_sha256": plan_sha256,
                "index": index,
                "view_id": planned["view_id"],
                "seed": planned["seed"],
                "camera": camera_record(camera),
                "fit_window": planned["fit_window"],
                "teacher_path": teacher_path.relative_to(out).as_posix(),
                "teacher_sha256": teacher_sha256,
                "teacher_npz_members": members,
                "teacher": observation_summary(reloaded),
                "history": history,
                "metrics": metrics,
                "runtime": {
                    "wall_seconds": wall_seconds,
                    "fit_seconds": history["fit_seconds"],
                    "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                },
                "preview": preview_path.relative_to(out).as_posix(),
                "preview_sha256": sha256_file(preview_path),
                "external_structsplat": external_after,
                "loaded_cuda_extension": extension_record(),
            },
        )
        return 0
    except BaseException as error:
        if not record_path.exists():
            write_json_exclusive(
                record_path,
                {
                    "status": "FAIL",
                    "index": index,
                    "view_id": VIEWS[index],
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise


def assemble(out: Path, plan: dict[str, Any], plan_sha256: str) -> dict[str, Any]:
    fields: list[GaussianObservationField] = []
    cameras: list[Camera] = []
    records: list[dict[str, Any]] = []
    for index, view_id in enumerate(VIEWS):
        record_path = out / "acquisition" / f"{index:04d}.json"
        teacher_path = out / "acquisition" / f"{index:04d}.teacher.npz"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "PASS" or record["view_id"] != view_id:
            raise RuntimeError(f"invalid acquisition record for {view_id}")
        if record["plan_sha256"] != plan_sha256:
            raise RuntimeError(f"{view_id} record has wrong plan hash")
        if sha256_file(teacher_path) != record["teacher_sha256"]:
            raise RuntimeError(f"{view_id} teacher hash changed before assembly")
        field = GaussianObservationField.load_npz(teacher_path, strict=True)
        if observation_summary(field) != record["teacher"]:
            raise RuntimeError(f"{view_id} teacher semantics changed before assembly")
        fields.append(field)
        cameras.append(camera_from_record(record["camera"]))
        records.append(record)

    if len({canonical_hash(record["loaded_cuda_extension"]) for record in records}) != 1:
        raise RuntimeError("workers loaded different CUDA extensions")
    bundle = ReconstructionInputs(
        observations=fields,
        cameras=cameras,
        view_names=list(VIEWS),
        points=None,
        point_visibility=None,
        bounds_hint=None,
        name="compact_masked_bundle_640_20260717",
    )
    bundle_path = out / "reconstruction_inputs"
    bundle.save(bundle_path)
    reloaded = ReconstructionInputs.load(bundle_path, device="cpu", strict=True)
    if reloaded.view_names != list(VIEWS) or reloaded.n_init_2d != [640] * 7:
        raise RuntimeError("strict bundle reload changed view order or initialization counts")
    if reloaded.n_opt_2d != [640] * 7 or any(
        value is not None
        for value in (reloaded.points, reloaded.point_visibility, reloaded.bounds_hint)
    ):
        raise RuntimeError("strict bundle reload changed counts or introduced geometry")
    if [observation_summary(field) for field in reloaded.observations] != [
        observation_summary(field) for field in fields
    ]:
        raise RuntimeError("strict bundle reload changed teacher semantics")

    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["geometry"] is not None or HELDOUT_VIEW in manifest_path.read_text(
        encoding="utf-8"
    ):
        raise RuntimeError("bundle contains geometry or held-out view")
    forbidden = ("rgb", "mask", "image_path", "source_path")
    if any(token in manifest_path.read_text(encoding="utf-8").lower() for token in forbidden):
        raise RuntimeError("bundle manifest contains a forbidden raster/path field")
    bundle_files = sorted(path for path in bundle_path.rglob("*") if path.is_file())
    bundle_hashes = {
        path.relative_to(bundle_path).as_posix(): sha256_file(path) for path in bundle_files
    }
    return {
        "artifact_type": "compact_masked_bundle_acquisition_result_v1",
        "status": "PASS",
        "decision_bearing": False,
        "scope": "seven training-view masked 640/100 Stage-1 acquisition only; no 3D lift",
        "plan_sha256": plan_sha256,
        "training_views": list(VIEWS),
        "seeds": list(SEEDS),
        "heldout_view_excluded": HELDOUT_VIEW,
        "n_views": reloaded.n_views,
        "n_init_2d": reloaded.n_init_2d,
        "n_opt_2d": reloaded.n_opt_2d,
        "total_components": sum(reloaded.n_opt_2d),
        "bundle": {
            "path": bundle_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "semantic_digest": manifest["semantic_digest"],
            "calibration_digest": manifest["calibration_digest"],
            "files": bundle_hashes,
            "aggregate_sha256": canonical_hash(bundle_hashes),
            "archive_stats": dataclasses.asdict(reloaded.archive_stats),
            "contains_geometry": False,
            "contains_dense_rgb_mask_or_source_path": False,
        },
        "views": records,
        "foreground_psnr_db": {
            record["view_id"]: record["metrics"]["foreground_rgb"]["psnr_db"] for record in records
        },
        "mean_foreground_psnr_db": sum(
            record["metrics"]["foreground_rgb"]["psnr_db"] for record in records
        )
        / len(records),
        "external_structsplat": plan["external_structsplat"],
        "repository": plan["repository"],
        "viewer": {
            "status": "NOT_APPLICABLE_STAGE1_ONLY",
            "reason": (
                "rtgs view requires a 3D PLY; this requested acquisition stops at 2D teachers"
            ),
        },
    }


def run(out: Path) -> int:
    out = out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite acquisition directory: {out}")
    if not torch.cuda.is_available():
        raise RuntimeError("native-resolution StructSplat acquisition requires CUDA")
    required = [CALIBRATION]
    for view_id in (*VIEWS, HELDOUT_VIEW):
        required.extend(view_paths(view_id))
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("one or more calibrated inputs are missing")
    out.mkdir(parents=True)
    (out / "acquisition").mkdir()
    (out / "previews").mkdir()
    try:
        config = requested_config(n_start=640, n_max=640, iterations=100)
        plan = {
            "artifact_type": "compact_masked_bundle_acquisition_plan_v1",
            "decision_bearing": False,
            "purpose": "input-only promotion of the C0001 masked 640/100 screen winner",
            "training_views": list(VIEWS),
            "seeds": list(SEEDS),
            "heldout_view_excluded": HELDOUT_VIEW,
            "source_rgb_boundary": (
                "RGB/masks are permitted only in Stage-1 acquisition/QA workers; the final parent "
                "assembles ReconstructionInputs from compact archives and camera records only."
            ),
            "requested_config": dataclasses.asdict(config),
            "requested_config_note": (
                "wrapper fields not present in effective_structsplat are not forwarded externally"
            ),
            "inputs": {
                "scene": SCENE.relative_to(ROOT).as_posix(),
                "calibration": CALIBRATION.relative_to(ROOT).as_posix(),
                "calibration_sha256": sha256_file(CALIBRATION),
            },
            "repository": repository_source_binding(),
            "external_structsplat": structsplat_source_binding(),
            "environment": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "ld_preload": os.environ.get("LD_PRELOAD"),
            },
            "views": [plan_view(view_id, seed) for view_id, seed in zip(VIEWS, SEEDS, strict=True)],
            "outputs": {
                "bundle": "reconstruction_inputs",
                "acquisition": "teacher_acquisition.json",
                "result": "result.json",
            },
        }
        plan_path = out / "plan.json"
        write_json_exclusive(plan_path, plan)
        plan_sha256 = sha256_file(plan_path)
        for index, view_id in enumerate(VIEWS):
            print(f"[{index + 1}/{len(VIEWS)}] acquiring {view_id}", flush=True)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--out",
                    str(out),
                    "--plan-sha256",
                    plan_sha256,
                    "--index",
                    str(index),
                ],
                cwd=ROOT,
                check=False,
                timeout=7200,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"acquisition worker failed for {view_id}")

        records = [
            json.loads((out / "acquisition" / f"{index:04d}.json").read_text(encoding="utf-8"))
            for index in range(len(VIEWS))
        ]
        acquisition = {
            "artifact_type": "compact_masked_bundle_teacher_acquisition_v1",
            "status": "PASS",
            "plan_sha256": plan_sha256,
            "views": records,
        }
        write_json_exclusive(out / "teacher_acquisition.json", acquisition)
        result = assemble(out, plan, plan_sha256)
        result["teacher_acquisition_sha256"] = sha256_file(out / "teacher_acquisition.json")
        if repository_source_binding() != plan["repository"]:
            raise RuntimeError("realtime-gs source changed across acquisition")
        if structsplat_source_binding() != plan["external_structsplat"]:
            raise RuntimeError("StructSplat source changed across acquisition")
        write_json_exclusive(out / "result.json", result)
        print(json.dumps(result, indent=2))
        return 0
    except BaseException as error:
        failure_path = out / "failure.json"
        if not failure_path.exists():
            write_json_exclusive(
                failure_path,
                {
                    "artifact_type": "compact_masked_bundle_acquisition_failure_v1",
                    "status": "FAIL",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plan-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--index", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out.expanduser().resolve()
    if args.worker:
        if args.plan_sha256 is None or args.index is None or not 0 <= args.index < len(VIEWS):
            raise ValueError("worker requires a valid plan hash and view index")
        return worker(out / "plan.json", args.plan_sha256, args.index, out)
    return run(out)


if __name__ == "__main__":
    raise SystemExit(main())
