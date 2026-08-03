#!/usr/bin/env python3
"""Produce three full-resolution, fixed-capacity RTGSV view bundles for frame_00008.

This is a resumable production-conversion driver, not claim-ready benchmark machinery.  It keeps
the source RGB/mask/calibration immutable, isolates every production view in its own process, and
records enough source/config/checksum information to audit or resume the conversion.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import asdict, replace
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

STRUCTSPLAT_ROOT = Path("/home/alex/Documents/structsplat")
REALTIME_ROOT = Path("/home/alex/Documents/realtime-gs")
for source_root in (STRUCTSPLAT_ROOT / "src", REALTIME_ROOT / "src"):
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

import numpy as np
from PIL import Image
import torch

from rtgs.core.metrics import masked_psnr, psnr
from rtgs.data.calibrated import load_calibrated_scene
from rtgs.data.compact_views import (
    CompactDataset,
    CompactView,
    file_sha256,
    save_compact_view,
    write_compact_dataset_manifest,
)
from rtgs.image2gs.fit import FitConfig as NativeFitConfig
from rtgs.image2gs.fit import _crop_to_mask, fit_image
from rtgs.image2gs.native_observation import native_gaussians_to_observation
from rtgs.image2gs.renderer2d import render_gaussians_2d
from rtgs.image2gs.structsplat_backend import field_to_observation
from structsplat.pipeline import run_current_pipeline


FRAME = Path(__file__).resolve().parent
CALIBRATION = FRAME.parent / "calibration_dome.json"
PROTOCOL_PATH = FRAME / "gaussians2d_11k_protocol.json"
BYTE_CAP = 8 * 1024 * 1024
CAPACITY = 11_000
SEED_BASE = 30_074_000
DEVICE = "cuda:0"
CUDA_INDEX = 0
PILOT_VIEW = "C0001"
PILOT_HORIZONS = (5_000, 10_000, 20_000)
PILOT_WINDOW = 500
PILOT_MAX_GAIN_DB = 0.05
ARMS = (
    "structsplat_no_boundary",
    "structsplat_mask_contained",
    "gaussianimage",
)
OUTPUT_NAMES = {
    "structsplat_no_boundary": "gaussians2d_structsplat_no_boundary_fullres",
    "structsplat_mask_contained": "gaussians2d_structsplat_mask_contained_fullres",
    "gaussianimage": "gaussians2d_gaussianimage_fullres",
}
ARM_LABELS = {
    "structsplat_no_boundary": "StructSplat — no boundary specialization",
    "structsplat_mask_contained": "StructSplat — mask-contained boundary specialization",
    "gaussianimage": "GaussianImage-style native fixed-count",
}
COMMON_SOURCE_FILES = (
    Path(__file__).resolve(),
    PROTOCOL_PATH,
    REALTIME_ROOT / "src/rtgs/core/observation2d.py",
    REALTIME_ROOT / "src/rtgs/data/calibrated.py",
    REALTIME_ROOT / "src/rtgs/data/compact_views.py",
)
STRUCTSPLAT_SOURCE_FILES = (
    STRUCTSPLAT_ROOT / "src/structsplat/pipeline.py",
    STRUCTSPLAT_ROOT / "src/structsplat/safe_schedule.py",
    STRUCTSPLAT_ROOT / "src/structsplat/fit.py",
    STRUCTSPLAT_ROOT / "src/structsplat/init.py",
    STRUCTSPLAT_ROOT / "src/structsplat/render.py",
    REALTIME_ROOT / "src/rtgs/image2gs/structsplat_backend.py",
)
NATIVE_SOURCE_FILES = (
    REALTIME_ROOT / "src/rtgs/image2gs/fit.py",
    REALTIME_ROOT / "src/rtgs/image2gs/renderer2d.py",
    REALTIME_ROOT / "src/rtgs/image2gs/native_observation.py",
    REALTIME_ROOT / "src/rtgs/image2gs/cuda_backend.py",
    REALTIME_ROOT / "src/rtgs/image2gs/cuda/renderer2d_ext.cpp",
    REALTIME_ROOT / "src/rtgs/image2gs/cuda/renderer2d_ext.cu",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise TypeError(f"refusing to serialize non-scalar tensor with shape {value.shape}")
        return value.detach().cpu().item()
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("non-finite value in JSON artifact")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _git_record(root: Path) -> dict[str, object]:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).splitlines()
    return {"root": str(root), "revision": revision, "status_short": status}


def _source_binding(arm: str) -> dict[str, object]:
    source_files = list(COMMON_SOURCE_FILES)
    source_files.extend(NATIVE_SOURCE_FILES if arm == "gaussianimage" else STRUCTSPLAT_SOURCE_FILES)
    hashes = {str(path): file_sha256(path) for path in source_files}
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "structsplat_git": _git_record(STRUCTSPLAT_ROOT),
        "realtime_git": _git_record(REALTIME_ROOT),
        "files": hashes,
        "aggregate_sha256": _digest(hashes),
    }


def _output(arm: str) -> Path:
    return FRAME / OUTPUT_NAMES[arm]


def _view_ids() -> list[str]:
    values = sorted(
        path.stem.upper()
        for path in (FRAME / "rgb").iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and not path.stem.lower().startswith("mask_")
    )
    if len(values) != 26 or len(values) != len(set(values)):
        raise RuntimeError(f"expected 26 unique RGB views, found {len(values)}")
    for view_id in values:
        if not (FRAME / "mask" / f"mask_{view_id}.png").is_file():
            raise FileNotFoundError(f"missing canonical PNG mask for {view_id}")
    return values


def _seed(view_id: str) -> int:
    return SEED_BASE + _view_ids().index(view_id)


def _paths(arm: str, view_id: str) -> dict[str, Path]:
    output = _output(arm)
    return {
        "field": output / f"{view_id}.rtgsv",
        "receipt": output / "receipts" / f"{view_id}.json",
        "history": output / "history" / f"{view_id}.json",
        "crop": output / "qa" / f"{view_id}_reconstruction_crop.jpg",
        "canvas": output / "qa" / f"{view_id}_reconstruction_canvas.jpg",
    }


def _input_paths(view_id: str) -> tuple[Path, Path]:
    return (
        FRAME / "rgb" / f"{view_id}.jpg",
        FRAME / "mask" / f"mask_{view_id}.png",
    )


def _load_view(view_id: str) -> dict[str, Any]:
    rgb_path, mask_path = _input_paths(view_id)
    scene = load_calibrated_scene(
        FRAME,
        calibration_path=CALIBRATION,
        downscale=1,
        test_every=0,
        load_masks=True,
        undistort=True,
        view_ids=[view_id],
    )
    if scene.view_names != [view_id] or scene.masks is None:
        raise RuntimeError(f"failed to isolate RGB/mask pair for {view_id}")
    image = scene.images[0].contiguous()
    mask = (scene.masks[0] > 0.5).contiguous()
    camera = scene.cameras[0]
    crop, mask_crop, offset = _crop_to_mask(image, mask)
    fit_window = (
        int(offset[0]),
        int(offset[1]),
        int(crop.shape[1]),
        int(crop.shape[0]),
    )
    return {
        "rgb_path": rgb_path,
        "mask_path": mask_path,
        "image": image,
        "mask": mask,
        "camera": camera,
        "crop": crop.contiguous(),
        "mask_crop": mask_crop.bool().contiguous(),
        "fit_window": fit_window,
    }


def _pil_rgb(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().float().clamp(0.0, 1.0).mul(255.0).round().byte().numpy()
    return Image.fromarray(array, mode="RGB")


def _save_jpeg_atomic(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, format="JPEG", quality=92, subsampling=0, optimize=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_qa(
    rendered: torch.Tensor,
    *,
    crop_path: Path,
    canvas_path: Path,
    canvas_size: tuple[int, int],
    offset: tuple[int, int],
) -> None:
    crop = _pil_rgb(rendered)
    _save_jpeg_atomic(crop_path, crop)
    height, width = canvas_size
    canvas = Image.new("RGB", (width, height))
    canvas.paste(crop, offset)
    _save_jpeg_atomic(canvas_path, canvas)


def _metrics(
    rendered: torch.Tensor,
    crop: torch.Tensor,
    mask_crop: torch.Tensor,
) -> dict[str, float]:
    device = rendered.device
    target = crop.to(device=device, dtype=rendered.dtype)
    mask = mask_crop.to(device=device, dtype=torch.bool)
    scored = rendered.clamp(0.0, 1.0)
    matted_target = target * mask[..., None].to(target)
    outside = ~mask
    outside_values = scored[outside]
    return {
        "foreground_psnr_db": masked_psnr(scored, target, mask),
        "matted_crop_psnr_db": psnr(scored, matted_target),
        "outside_max_abs": (float(outside_values.abs().max()) if outside_values.numel() else 0.0),
        "outside_mean_abs": (float(outside_values.abs().mean()) if outside_values.numel() else 0.0),
    }


def _input_record(prepared: dict[str, Any]) -> dict[str, object]:
    return {
        "rgb": {
            "name": prepared["rgb_path"].name,
            "bytes": prepared["rgb_path"].stat().st_size,
            "sha256": file_sha256(prepared["rgb_path"]),
        },
        "mask": {
            "name": prepared["mask_path"].name,
            "bytes": prepared["mask_path"].stat().st_size,
            "sha256": file_sha256(prepared["mask_path"]),
        },
        "calibration": {
            "path": str(CALIBRATION),
            "bytes": CALIBRATION.stat().st_size,
            "sha256": file_sha256(CALIBRATION),
        },
    }


def _no_boundary_transform(schedule):
    return replace(
        schedule,
        boundary_enabled=False,
        boundary=replace(schedule.boundary, name="general_closure"),
    )


def _native_config(iterations: int) -> NativeFitConfig:
    return NativeFitConfig(
        n_gaussians=CAPACITY,
        max_gaussians=CAPACITY,
        iterations=int(iterations),
        backend="native",
        adaptive_density=False,
        growth_waves=5,
        relocate_fraction=0.0,
        structsplat_renderer="auto",
        native_renderer="cuda",
        batch_views=False,
        lr=1e-2,
        grad_init_mix=0.7,
        init_strategy="gradient",
        structure_sampling="wse",
        row_chunk=64,
        log_every=100,
        convergence_patience=0,
        convergence_tol=0.05,
        convergence_check_every=25,
        appearance_parameterization="weight_color_9p",
        freeze_geometry=False,
        pool=False,
        pool_capacity=None,
        pool_triage_every=50,
        pool_prune_count=32,
        pool_spawn_count=32,
        pool_min_live=1,
        mask_coverage_weight=0.0,
    )


def _reset_cuda_stats(device: torch.device) -> None:
    # This development Torch build rejects reset_peak_memory_stats before the first CUDA
    # allocation has initialized the context, even though ordinary CUDA device discovery works.
    torch.empty(0, device=device)
    torch.cuda.reset_peak_memory_stats(CUDA_INDEX)


def _common_receipt(
    arm: str,
    view_id: str,
    prepared: dict[str, Any],
    binding: dict[str, object],
    config: dict[str, Any],
    metrics: dict[str, float],
    *,
    n_gaussians: int,
    runtime: dict[str, float | int],
    history_path: Path,
    field_path: Path,
    crop_path: Path,
    canvas_path: Path,
) -> dict[str, object]:
    loaded = CompactView.load(field_path, byte_cap=BYTE_CAP)
    receipt = {
        "schema": "janelle.frame00008.three_arm_11k_view.v1",
        "status": "PASS",
        "classification": "production conversion; diagnostic comparison only",
        "arm": arm,
        "view_id": view_id,
        "view_index": _view_ids().index(view_id),
        "seed": _seed(view_id),
        "input": _input_record(prepared),
        "source_binding": binding,
        "config": config,
        "config_digest": _digest(config),
        "canvas_size": [prepared["camera"].height, prepared["camera"].width],
        "fit_window": list(prepared["fit_window"]),
        "foreground_pixels": int(prepared["mask_crop"].sum()),
        "metrics": metrics,
        "n_gaussians": int(n_gaussians),
        "runtime": runtime,
        "output": {
            "path": str(field_path),
            "bytes": loaded.bytes,
            "sha256": loaded.sha256,
            "provider": loaded.observation.provider,
            "blend_mode": loaded.observation.blend_mode,
            "sigma_cutoff": loaded.observation.sigma_cutoff,
            "support_fade_alpha": loaded.observation.support_fade_alpha,
            "aa_dilation": loaded.observation.aa_dilation,
        },
        "history": {
            "path": str(history_path),
            "bytes": history_path.stat().st_size,
            "sha256": file_sha256(history_path),
        },
        "qa": {
            "crop": {
                "path": str(crop_path),
                "bytes": crop_path.stat().st_size,
                "sha256": file_sha256(crop_path),
            },
            "canvas": {
                "path": str(canvas_path),
                "bytes": canvas_path.stat().st_size,
                "sha256": file_sha256(canvas_path),
            },
        },
    }
    receipt["semantic_digest"] = _digest(receipt)
    return receipt


def _persist_native(
    view_id: str,
    prepared: dict[str, Any],
    fitted,
    rendered: torch.Tensor,
    history: dict[str, Any],
    config: NativeFitConfig,
    runtime: dict[str, float | int],
) -> dict[str, object]:
    paths = _paths("gaussianimage", view_id)
    binding = _source_binding("gaussianimage")
    config_dict = asdict(config)
    config_digest = _digest(config_dict)
    observation = native_gaussians_to_observation(
        fitted.to("cpu"),
        canvas_size=(prepared["camera"].height, prepared["camera"].width),
        fit_window=prepared["fit_window"],
        view_id=view_id,
        n_init=CAPACITY,
        producer_version=binding["realtime_git"]["revision"],
        producer_source_digest=binding["aggregate_sha256"],
        fit_config_digest=config_digest,
    )
    save_compact_view(
        paths["field"],
        observation,
        prepared["camera"],
        calibration_sha256=file_sha256(CALIBRATION),
        source_rgb_name=prepared["rgb_path"].name,
        source_rgb_sha256=file_sha256(prepared["rgb_path"]),
        alpha_crop=prepared["mask_crop"],
        source_mask_name=prepared["mask_path"].name,
        source_mask_sha256=file_sha256(prepared["mask_path"]),
        byte_cap=BYTE_CAP,
    )
    metrics = _metrics(rendered, prepared["crop"], prepared["mask_crop"])
    _write_json_atomic(paths["history"], history)
    _save_qa(
        rendered.clamp(0.0, 1.0),
        crop_path=paths["crop"],
        canvas_path=paths["canvas"],
        canvas_size=(prepared["camera"].height, prepared["camera"].width),
        offset=prepared["fit_window"][:2],
    )
    receipt = _common_receipt(
        "gaussianimage",
        view_id,
        prepared,
        binding,
        config_dict,
        metrics,
        n_gaussians=fitted.n,
        runtime=runtime,
        history_path=paths["history"],
        field_path=paths["field"],
        crop_path=paths["crop"],
        canvas_path=paths["canvas"],
    )
    _write_json_atomic(paths["receipt"], receipt)
    return receipt


def _fit_native(view_id: str, iterations: int) -> dict[str, Any]:
    prepared = _load_view(view_id)
    config = _native_config(iterations)
    seed = _seed(view_id)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(DEVICE)
    torch.cuda.empty_cache()
    _reset_cuda_stats(device)
    started = time.perf_counter()
    fitted, history = fit_image(
        prepared["crop"].to(device),
        config,
        seed=seed,
        mask=prepared["mask_crop"].to(device),
    )
    if fitted.n != CAPACITY:
        raise RuntimeError(f"native fixed-capacity fit returned {fitted.n}, expected {CAPACITY}")
    rendered = render_gaussians_2d(
        fitted,
        prepared["crop"].shape[0],
        prepared["crop"].shape[1],
        renderer="cuda",
    ).clamp(0.0, 1.0)
    torch.cuda.synchronize(device)
    runtime = {
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(CUDA_INDEX)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(CUDA_INDEX)),
    }
    return {
        "prepared": prepared,
        "config": config,
        "fitted": fitted,
        "rendered": rendered,
        "history": history,
        "runtime": runtime,
    }


def _pilot_gain(history: dict[str, Any], horizon: int) -> tuple[float, int, int]:
    curve = [(int(step), float(value)) for step, value in history["psnr"]]
    if not curve:
        raise ValueError("native pilot produced no PSNR curve")
    final_step, final_value = curve[-1]
    threshold = max(0, final_step - PILOT_WINDOW)
    start_step, start_value = min(curve, key=lambda row: abs(row[0] - threshold))
    if final_step < horizon - 2:
        raise ValueError("fixed-horizon pilot unexpectedly stopped early")
    return final_value - start_value, start_step, final_step


def _pilot_gaussianimage() -> None:
    output = _output("gaussianimage")
    selection_path = output / "horizon_selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        print(json.dumps(selection, indent=2), flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, object]] = []
    selected = None
    for horizon in PILOT_HORIZONS:
        payload = _fit_native(PILOT_VIEW, horizon)
        gain, start_step, final_step = _pilot_gain(payload["history"], horizon)
        attempt = {
            "iterations": horizon,
            "final_window_start_step": start_step,
            "final_step": final_step,
            "final_window_gain_db": gain,
            "threshold_db": PILOT_MAX_GAIN_DB,
            "passes": gain <= PILOT_MAX_GAIN_DB or horizon == PILOT_HORIZONS[-1],
            "runtime": payload["runtime"],
        }
        attempts.append(attempt)
        print(json.dumps(attempt, sort_keys=True), flush=True)
        if attempt["passes"]:
            selected = horizon
            selection = {
                "schema": "janelle.frame00008.gaussianimage_11k_horizon_selection.v1",
                "development_view": PILOT_VIEW,
                "candidate_iterations": list(PILOT_HORIZONS),
                "window_updates": PILOT_WINDOW,
                "maximum_window_gain_db": PILOT_MAX_GAIN_DB,
                "attempts": attempts,
                "selected_iterations": selected,
                "selection_is_outcome_exposed": True,
            }
            selection["semantic_digest"] = _digest(selection)
            _write_json_atomic(selection_path, selection)
            protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
            protocol["arms"]["gaussianimage"]["horizon_selection"]["selected_iterations"] = selected
            _write_json_atomic(PROTOCOL_PATH, protocol)
            _persist_native(
                PILOT_VIEW,
                payload["prepared"],
                payload["fitted"],
                payload["rendered"],
                payload["history"],
                payload["config"],
                payload["runtime"],
            )
            del payload
            torch.cuda.empty_cache()
            break
        del payload
        torch.cuda.empty_cache()
    if selected is None:
        raise RuntimeError("GaussianImage horizon selection failed")
    print(f"selected GaussianImage horizon: {selected}", flush=True)


def _fit_structsplat(arm: str, view_id: str) -> dict[str, object]:
    if arm not in {"structsplat_no_boundary", "structsplat_mask_contained"}:
        raise ValueError(f"not a StructSplat arm: {arm}")
    prepared = _load_view(view_id)
    seed = _seed(view_id)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(DEVICE)
    torch.cuda.empty_cache()
    _reset_cuda_stats(device)
    transform = _no_boundary_transform if arm == "structsplat_no_boundary" else None
    started = time.perf_counter()
    output = run_current_pipeline(
        prepared["crop"].numpy(),
        mask=prepared["mask_crop"].numpy(),
        device=DEVICE,
        seed=seed,
        schedule_transform=transform,
        verbose=False,
    )
    torch.cuda.synchronize(device)
    runtime = {
        "wall_seconds": time.perf_counter() - started,
        "pipeline_seconds": float(output["timing"]["total_seconds"]),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(CUDA_INDEX)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(CUDA_INDEX)),
    }
    field = output["field"]
    if field.n > CAPACITY:
        raise RuntimeError(f"StructSplat returned {field.n} rows above capacity {CAPACITY}")
    result = output["schedule_result"]
    expected_boundary = arm == "structsplat_mask_contained"
    if bool(result["schedule"]["boundary_enabled"]) != expected_boundary:
        raise RuntimeError("resolved boundary specialization differs from requested arm")
    if bool(result["fit_config"]["mask_contain"]) != expected_boundary:
        raise RuntimeError("resolved hard-containment flag differs from requested arm")
    rendered = output["render"].clamp(0.0, 1.0)
    metrics = _metrics(rendered, prepared["crop"], prepared["mask_crop"])
    if expected_boundary and metrics["outside_max_abs"] != 0.0:
        raise RuntimeError(
            f"contained arm rendered nonzero outside mask: {metrics['outside_max_abs']}"
        )
    binding = _source_binding(arm)
    config = {
        "profile": output["profile"],
        "initialization": output["initialization"],
        "fit_config": output["fit_config"],
        "schedule": output["schedule"],
        "resolved_boundary_specialization": expected_boundary,
    }
    config_digest = _digest(config)
    paths = _paths(arm, view_id)
    observation = field_to_observation(
        field,
        canvas_size=(prepared["camera"].height, prepared["camera"].width),
        fit_window=prepared["fit_window"],
        blend_mode="normalized",
        epsilon=1e-8,
        sigma_cutoff=float(result["fit_config"]["sigma_cutoff"]),
        support_fade_alpha=1.0,
        aa_dilation=float(result["fit_config"]["aa_dilation"]),
        view_id=view_id,
        n_init=5_000,
        producer_version=binding["structsplat_git"]["revision"],
        producer_source_digest=binding["aggregate_sha256"],
        fit_config_digest=config_digest,
    ).to("cpu")
    save_compact_view(
        paths["field"],
        observation,
        prepared["camera"],
        calibration_sha256=file_sha256(CALIBRATION),
        source_rgb_name=prepared["rgb_path"].name,
        source_rgb_sha256=file_sha256(prepared["rgb_path"]),
        alpha_crop=prepared["mask_crop"],
        source_mask_name=prepared["mask_path"].name,
        source_mask_sha256=file_sha256(prepared["mask_path"]),
        byte_cap=BYTE_CAP,
    )
    history = {
        "metrics": result["metrics"],
        "events": result["history"],
        "attempted_steps": result["attempted_steps"],
        "accepted_steps": result["accepted_steps"],
        "converged": result["converged"],
        "storage": result["storage"],
        "error_tail": result["error_tail"],
        "pursuit_tail": result["pursuit_tail"],
    }
    _write_json_atomic(paths["history"], history)
    _save_qa(
        rendered,
        crop_path=paths["crop"],
        canvas_path=paths["canvas"],
        canvas_size=(prepared["camera"].height, prepared["camera"].width),
        offset=prepared["fit_window"][:2],
    )
    receipt = _common_receipt(
        arm,
        view_id,
        prepared,
        binding,
        config,
        metrics,
        n_gaussians=field.n,
        runtime=runtime,
        history_path=paths["history"],
        field_path=paths["field"],
        crop_path=paths["crop"],
        canvas_path=paths["canvas"],
    )
    receipt["schedule_metrics"] = result["metrics"]
    receipt["attempted_steps"] = result["attempted_steps"]
    receipt["accepted_steps"] = result["accepted_steps"]
    receipt["semantic_digest"] = _digest(
        {key: value for key, value in receipt.items() if key != "semantic_digest"}
    )
    _write_json_atomic(paths["receipt"], receipt)
    del output, field, rendered, observation
    torch.cuda.empty_cache()
    return receipt


def _selected_native_horizon() -> int:
    path = _output("gaussianimage") / "horizon_selection.json"
    if not path.is_file():
        raise FileNotFoundError("run pilot-gaussianimage before GaussianImage production")
    selection = json.loads(path.read_text(encoding="utf-8"))
    value = int(selection["selected_iterations"])
    if value not in PILOT_HORIZONS:
        raise ValueError("invalid selected GaussianImage horizon")
    return value


def _worker(arm: str, view_id: str, iterations: int | None) -> None:
    paths = _paths(arm, view_id)
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite partial output: {existing[0]}")
    try:
        if arm == "gaussianimage":
            if iterations is None:
                raise ValueError("GaussianImage worker requires --iterations")
            payload = _fit_native(view_id, iterations)
            receipt = _persist_native(
                view_id,
                payload["prepared"],
                payload["fitted"],
                payload["rendered"],
                payload["history"],
                payload["config"],
                payload["runtime"],
            )
        else:
            receipt = _fit_structsplat(arm, view_id)
        print(
            f"{arm}/{view_id}: PASS · {receipt['n_gaussians']} splats · "
            f"{receipt['metrics']['foreground_psnr_db']:.3f} dB · "
            f"{receipt['runtime']['wall_seconds']:.1f}s",
            flush=True,
        )
    except BaseException as error:
        failure = {
            "schema": "janelle.frame00008.three_arm_11k_view_failure.v1",
            "status": "FAIL",
            "arm": arm,
            "view_id": view_id,
            "seed": _seed(view_id),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json_atomic(paths["receipt"], failure)
        for key in ("field", "history", "crop", "canvas"):
            paths[key].unlink(missing_ok=True)
        raise


def _reusable(arm: str, view_id: str) -> bool:
    paths = _paths(arm, view_id)
    try:
        receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
        loaded = CompactView.load(paths["field"], byte_cap=BYTE_CAP)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return (
        receipt.get("status") == "PASS"
        and receipt.get("arm") == arm
        and receipt.get("view_id") == view_id
        and receipt.get("seed") == _seed(view_id)
        and receipt.get("output", {}).get("sha256") == loaded.sha256
        and receipt.get("output", {}).get("bytes") == loaded.bytes
        and receipt.get("n_gaussians") == loaded.observation.n
    )


def _publish(arm: str) -> None:
    output = _output(arm)
    view_ids = _view_ids()
    receipts = [
        json.loads(_paths(arm, view_id)["receipt"].read_text(encoding="utf-8"))
        for view_id in view_ids
    ]
    if any(receipt.get("status") != "PASS" for receipt in receipts):
        raise RuntimeError(f"cannot publish incomplete arm {arm}")
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        write_compact_dataset_manifest(
            output,
            name=f"frame_00008_{arm}_11k_fullres",
            calibration_sha256=file_sha256(CALIBRATION),
            view_paths=[output / f"{view_id}.rtgsv" for view_id in view_ids],
            bounds_hint=None,
            byte_cap=BYTE_CAP,
        )
    cards = []
    for receipt in receipts:
        view_id = receipt["view_id"]
        cards.append(
            f"<article><h2>{html.escape(view_id)} · "
            f"{receipt['n_gaussians']:,} splats · "
            f"{receipt['metrics']['foreground_psnr_db']:.2f} dB</h2>"
            f'<a href="qa/{view_id}_reconstruction_crop.jpg">'
            f'<img src="qa/{view_id}_reconstruction_crop.jpg" loading="lazy" '
            f'alt="{html.escape(view_id)} reconstruction"></a></article>'
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(ARM_LABELS[arm])}</title><style>
body{{font-family:system-ui;background:#111;color:#eee;margin:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:18px}}
article{{background:#1c1c1c;padding:12px;border-radius:8px}}img{{width:100%;height:auto}}
h2{{font-size:15px}}a{{color:#8cc8ff}}</style></head><body>
<h1>{html.escape(ARM_LABELS[arm])}</h1>
<p>26 calibrated 5328×4608 views; fitting uses native-resolution foreground windows.
This is a production conversion and diagnostic comparison, not claim-ready evidence.</p>
<p><a href="manifest.json">compact manifest</a> ·
<a href="production_manifest.json">production manifest</a></p>
<main class="grid">{"".join(cards)}</main></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    production = {
        "schema": "janelle.frame00008.three_arm_11k_production.v1",
        "arm": arm,
        "label": ARM_LABELS[arm],
        "classification": "production conversion; diagnostic comparison only",
        "protocol": {
            "path": str(PROTOCOL_PATH),
            "sha256": file_sha256(PROTOCOL_PATH),
        },
        "manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": file_sha256(manifest_path),
        },
        "views": receipts,
        "summary": {
            "view_count": len(receipts),
            "total_gaussians": sum(int(row["n_gaussians"]) for row in receipts),
            "minimum_gaussians": min(int(row["n_gaussians"]) for row in receipts),
            "maximum_gaussians": max(int(row["n_gaussians"]) for row in receipts),
            "total_compact_bytes": sum(int(row["output"]["bytes"]) for row in receipts),
            "mean_foreground_psnr_db": sum(
                float(row["metrics"]["foreground_psnr_db"]) for row in receipts
            )
            / len(receipts),
            "mean_wall_seconds": sum(float(row["runtime"]["wall_seconds"]) for row in receipts)
            / len(receipts),
        },
    }
    production["semantic_digest"] = _digest(production)
    _write_json_atomic(output / "production_manifest.json", production)


def _produce(arm: str, workers: int, only_views: list[str] | None) -> None:
    if arm == "gaussianimage":
        horizon = _selected_native_horizon()
    else:
        horizon = None
    requested = _view_ids() if not only_views else [value.upper() for value in only_views]
    unknown = sorted(set(requested) - set(_view_ids()))
    if unknown:
        raise ValueError(f"unknown views: {unknown}")
    commands = []
    for view_id in requested:
        if _reusable(arm, view_id):
            print(f"{arm}/{view_id}: verified existing", flush=True)
            continue
        paths = _paths(arm, view_id)
        if paths["receipt"].exists():
            failure = json.loads(paths["receipt"].read_text(encoding="utf-8"))
            if failure.get("status") != "FAIL":
                raise FileExistsError(f"unverified receipt blocks production: {paths['receipt']}")
            paths["receipt"].unlink()
        leftovers = [
            paths[key] for key in ("field", "history", "crop", "canvas") if paths[key].exists()
        ]
        if leftovers:
            raise FileExistsError(f"unverified output blocks production: {leftovers[0]}")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_worker",
            "--arm",
            arm,
            "--view",
            view_id,
        ]
        if horizon is not None:
            command.extend(["--iterations", str(horizon)])
        commands.append(command)
    print(
        f"{arm}: {len(commands)} fresh views · workers={workers} · capacity={CAPACITY}",
        flush=True,
    )

    def execute(command: list[str]) -> None:
        subprocess.run(command, cwd=FRAME, check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, command) for command in commands]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    if all(_reusable(arm, view_id) for view_id in _view_ids()):
        _publish(arm)


def _verify_arm(arm: str) -> dict[str, object]:
    output = _output(arm)
    view_ids = _view_ids()
    compact = CompactDataset.load(output, byte_cap=BYTE_CAP, load_alpha=True)
    if [view.view_id for view in compact.views] != view_ids:
        raise ValueError(f"{arm}: manifest view order differs")
    rows = []
    for view_id, view in zip(view_ids, compact.views):
        if not _reusable(arm, view_id):
            raise ValueError(f"{arm}/{view_id}: receipt or compact field is not reusable")
        receipt = json.loads(_paths(arm, view_id)["receipt"].read_text(encoding="utf-8"))
        rgb_path, mask_path = _input_paths(view_id)
        if (
            receipt["input"]["rgb"]["sha256"] != file_sha256(rgb_path)
            or receipt["input"]["mask"]["sha256"] != file_sha256(mask_path)
            or receipt["input"]["calibration"]["sha256"] != file_sha256(CALIBRATION)
        ):
            raise ValueError(f"{arm}/{view_id}: input hash drift")
        if view.observation.n > CAPACITY:
            raise ValueError(f"{arm}/{view_id}: capacity exceeded")
        if view.alpha is None:
            raise ValueError(f"{arm}/{view_id}: packed alpha missing")
        if arm == "gaussianimage":
            if not (
                view.observation.n == CAPACITY
                and view.observation.provider == "native"
                and view.observation.blend_mode == "additive"
            ):
                raise ValueError(f"{arm}/{view_id}: native fixed-count contract differs")
        else:
            if not (
                view.observation.provider == "structsplat"
                and view.observation.blend_mode == "normalized"
                and view.observation.support_fade_alpha == 1.0
            ):
                raise ValueError(f"{arm}/{view_id}: StructSplat semantic contract differs")
            expected = arm == "structsplat_mask_contained"
            if bool(receipt["config"]["resolved_boundary_specialization"]) != expected:
                raise ValueError(f"{arm}/{view_id}: boundary flag differs")
            if expected and receipt["metrics"]["outside_max_abs"] != 0.0:
                raise ValueError(f"{arm}/{view_id}: contained render leaked outside mask")
        rows.append(receipt)
    production_path = output / "production_manifest.json"
    production = json.loads(production_path.read_text(encoding="utf-8"))
    expected_digest = _digest(
        {key: value for key, value in production.items() if key != "semantic_digest"}
    )
    if production.get("semantic_digest") != expected_digest:
        raise ValueError(f"{arm}: production manifest semantic digest differs")
    return {
        "arm": arm,
        "status": "PASS",
        "views": len(rows),
        "gaussian_range": [
            min(int(row["n_gaussians"]) for row in rows),
            max(int(row["n_gaussians"]) for row in rows),
        ],
        "total_compact_bytes": sum(int(row["output"]["bytes"]) for row in rows),
        "mean_foreground_psnr_db": sum(float(row["metrics"]["foreground_psnr_db"]) for row in rows)
        / len(rows),
        "output": str(output),
    }


def _verify(arm: str | None) -> None:
    arms = ARMS if arm is None else (arm,)
    summaries = [_verify_arm(value) for value in arms]
    result = {
        "schema": "janelle.frame00008.three_arm_11k_verification.v1",
        "status": "PASS",
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "arms": summaries,
    }
    result["semantic_digest"] = _digest(result)
    _write_json_atomic(FRAME / "gaussians2d_11k_verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


def _inventory() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["shared_budget"]["maximum_gaussians_per_view"] != CAPACITY:
        raise ValueError("driver capacity differs from protocol")
    if protocol["shared_budget"]["byte_cap_per_view"] != BYTE_CAP:
        raise ValueError("driver byte cap differs from protocol")
    records = []
    for view_id in _view_ids():
        rgb_path, mask_path = _input_paths(view_id)
        records.append(
            {
                "view_id": view_id,
                "rgb_bytes": rgb_path.stat().st_size,
                "rgb_sha256": file_sha256(rgb_path),
                "mask_bytes": mask_path.stat().st_size,
                "mask_sha256": file_sha256(mask_path),
                "seed": _seed(view_id),
            }
        )
    print(
        json.dumps(
            {
                "frame": str(FRAME),
                "calibration_sha256": file_sha256(CALIBRATION),
                "views": records,
                "cuda": torch.cuda.get_device_name(torch.device(DEVICE)),
                "protocol_sha256": file_sha256(PROTOCOL_PATH),
            },
            indent=2,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inventory")
    subparsers.add_parser("pilot-gaussianimage")
    produce = subparsers.add_parser("produce")
    produce.add_argument("--arm", choices=ARMS, required=True)
    produce.add_argument("--workers", type=int, default=1)
    produce.add_argument("--view", action="append", dest="views")
    worker = subparsers.add_parser("_worker")
    worker.add_argument("--arm", choices=ARMS, required=True)
    worker.add_argument("--view", required=True)
    worker.add_argument("--iterations", type=int)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--arm", choices=ARMS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "inventory":
        _inventory()
    elif args.command == "pilot-gaussianimage":
        _pilot_gaussianimage()
    elif args.command == "produce":
        if args.workers <= 0 or args.workers > 2:
            raise ValueError("workers must be in [1, 2]")
        if args.arm != "gaussianimage" and args.workers != 1:
            raise ValueError("StructSplat production uses one GPU worker")
        _produce(args.arm, args.workers, args.views)
    elif args.command == "_worker":
        _worker(args.arm, args.view.upper(), args.iterations)
    else:
        _verify(args.arm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
