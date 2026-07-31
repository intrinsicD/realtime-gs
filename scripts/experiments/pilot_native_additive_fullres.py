#!/usr/bin/env python3
"""Run one explicitly non-claim Stage-1 native-additive full-resolution pilot.

The pilot is allowed while the downstream experiment remains draft.  It decodes one calibrated
RGB/mask pair, fits a fixed-capacity GaussianImage-style native field on the tight foreground
window at native pixel density, and writes direct-resolution QA artifacts under ``.scratch``.
It does not publish or seal a reconstruction dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from PIL import Image

from rtgs.data.calibrated import load_calibrated_scene
from rtgs.data.compact_views import file_sha256, save_compact_view
from rtgs.image2gs.fit import FitConfig, _crop_to_mask, fit_image
from rtgs.image2gs.native_observation import native_gaussians_to_observation
from rtgs.image2gs.renderer2d import render_gaussians_2d

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAME = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008"
DEFAULT_CALIBRATION = ROOT / "dataset/2025_03_07_stage_with_fabric/calibration_dome.json"
DEFAULT_OUTPUT_ROOT = ROOT / ".scratch/rtgs007_stage1_pilot"


def _save_rgb(path: Path, value: torch.Tensor) -> None:
    pixels = value.detach().float().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    Image.fromarray(pixels, mode="RGB").save(path)


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run(args: argparse.Namespace) -> Path:
    frame = args.frame.resolve()
    calibration = args.calibration.resolve()
    output = (
        args.output_root.resolve()
        / f"{frame.name}_{args.view}_n{args.gaussians:06d}_i{args.iterations:06d}_s{args.seed}"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing pilot: {output}")
    output.mkdir(parents=True)

    scene = load_calibrated_scene(
        frame,
        calibration_path=calibration,
        downscale=1,
        test_every=0,
        load_masks=True,
        undistort=True,
        view_ids=[args.view],
    )
    if scene.view_names != [args.view] or scene.masks is None:
        raise RuntimeError("pilot failed to isolate the requested calibrated RGB/mask view")
    image = scene.images[0]
    mask = scene.masks[0] > 0.5
    camera = scene.cameras[0]
    crop, mask_crop, offset = _crop_to_mask(image, mask)
    fit_window = (
        int(offset[0]),
        int(offset[1]),
        int(crop.shape[1]),
        int(crop.shape[0]),
    )
    config = FitConfig(
        n_gaussians=args.gaussians,
        max_gaussians=args.gaussians,
        iterations=args.iterations,
        backend="native",
        adaptive_density=False,
        native_renderer="cuda",
        batch_views=False,
        lr=args.lr,
        init_strategy="gradient",
        appearance_parameterization="weight_color_9p",
        freeze_geometry=False,
        pool=False,
        mask_coverage_weight=0.0,
        convergence_patience=0,
        log_every=max(1, min(100, args.iterations)),
    )
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    fitted, history = fit_image(
        crop.to(device),
        config,
        seed=args.seed,
        mask=mask_crop.to(device),
    )
    rendered = render_gaussians_2d(
        fitted,
        crop.shape[0],
        crop.shape[1],
        renderer="cuda",
    ).clamp(0.0, 1.0)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    crop_target = crop.to(device) * mask_crop.to(device)[..., None]
    residual = (rendered - crop_target).abs().mul(4.0).clamp(0.0, 1.0)
    full_canvas = torch.zeros_like(image, device=device)
    x, y, width, height = fit_window
    full_canvas[y : y + height, x : x + width] = rendered
    _save_rgb(output / "target_native_crop.png", crop_target)
    _save_rgb(output / "fit_native_crop.png", rendered)
    _save_rgb(output / "residual_x4_native_crop.png", residual)
    _save_rgb(output / "fit_native_full_canvas.png", full_canvas)

    fit_config = asdict(config)
    fit_digest = _digest(fit_config)
    observation = native_gaussians_to_observation(
        fitted.to("cpu"),
        canvas_size=(camera.height, camera.width),
        fit_window=fit_window,
        view_id=args.view,
        n_init=config.n_gaussians,
        producer_version="mechanism-pilot",
        fit_config_digest=fit_digest,
    )
    rgb_path = frame / "rgb" / f"{args.view}.jpg"
    mask_path = frame / "mask" / f"mask_{args.view}.png"
    save_compact_view(
        output / f"{args.view}.rtgsv",
        observation,
        camera,
        calibration_sha256=file_sha256(calibration),
        source_rgb_name=rgb_path.name,
        source_rgb_sha256=file_sha256(rgb_path),
        alpha_crop=mask_crop,
        source_mask_name=mask_path.name,
        source_mask_sha256=file_sha256(mask_path),
        byte_cap=args.byte_cap,
    )
    metrics = {
        "schema": "rtgs.native_additive_fullres_pilot.v1",
        "evidence_status": "mechanism_only_not_protocol_result",
        "frame": frame.relative_to(ROOT).as_posix(),
        "view": args.view,
        "seed": args.seed,
        "canvas_height": camera.height,
        "canvas_width": camera.width,
        "fit_window": list(fit_window),
        "foreground_pixels": int(mask_crop.sum()),
        "fit_config": fit_config,
        "fit_config_digest": fit_digest,
        "history": history,
        "runtime": {
            "wall_seconds": elapsed,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "artifacts": [
            "target_native_crop.png",
            "fit_native_crop.png",
            "residual_x4_native_crop.png",
            "fit_native_full_canvas.png",
            f"{args.view}.rtgsv",
        ],
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), **metrics["runtime"], **history}, indent=2))
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=DEFAULT_FRAME)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--view", default="C0001")
    parser.add_argument("--gaussians", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, default=300700)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--byte-cap", type=int, default=4_194_304)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    run(_parse_args())
