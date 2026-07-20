#!/usr/bin/env python3
"""Exploratory C0001 StructSplat teacher-capacity screen.

The experiment compares the frozen full-frame 640/100 teacher against two current-source
masked/cropped fits. Source RGB is used only for Stage-1 fitting and isolated evaluation. Every
new fit is exported immediately as a lossless :class:`GaussianObservationField`; the winning
archive can therefore cross the RGB-denied Stage-1 boundary without retaining an image tensor.

This is a reproducible mechanism screen, not a sealed or default-changing benchmark.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.calibrated import _resize_image, _undistort
from rtgs.image2gs.fit import FitConfig, _crop_to_mask, fit_image

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008"
CALIBRATION = ROOT / "dataset/2025_03_07_stage_with_fabric/calibration_dome.json"
IMAGE = SCENE / "rgb/C0001.jpg"
MASK = SCENE / "mask/mask_C0001.png"
BASELINE = (
    ROOT / "runs/compact_point_training_20260716/reconstruction_inputs/teachers/0000.teacher.npz"
)
BASELINE_ACQUISITION = ROOT / "runs/compact_point_training_20260716/teacher_acquisition.json"
DEFAULT_OUT = ROOT / "runs/compact_stage1_mask_screen_20260717"
STRUCTSPLAT_ROOT = Path("~/Documents/structsplat").expanduser().resolve()
VIEW_ID = "C0001"
SEED = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def command_output(*argv: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(argv, cwd=cwd, text=True).strip()


def structsplat_source_binding() -> dict[str, Any]:
    import structsplat

    source_root = Path(structsplat.__file__).resolve().parent
    suffixes = {".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}
    files = sorted(
        path for path in source_root.rglob("*") if path.is_file() and path.suffix in suffixes
    )
    digest = hashlib.sha256()
    manifest: dict[str, str] = {}
    for path in files:
        relative = path.relative_to(source_root).as_posix()
        value = sha256_file(path)
        manifest[relative] = value
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    status = command_output("git", "status", "--porcelain=v1", cwd=STRUCTSPLAT_ROOT)
    tracked_diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--"], cwd=STRUCTSPLAT_ROOT
    )
    return {
        "version": importlib.metadata.version("structsplat"),
        "module_root": str(source_root),
        "repository_root": str(STRUCTSPLAT_ROOT),
        "git_revision": command_output("git", "rev-parse", "HEAD", cwd=STRUCTSPLAT_ROOT),
        "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "git_status_lines": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "provider_source_digest": digest.hexdigest(),
        "provider_source_file_count": len(files),
        "provider_source_manifest_sha256": canonical_hash(manifest),
    }


def repository_source_binding() -> dict[str, Any]:
    paths = [
        Path("benchmarks/compact_stage1_mask_screen.py"),
        Path("src/rtgs/core/observation2d.py"),
        Path("src/rtgs/data/calibrated.py"),
        Path("src/rtgs/image2gs/fit.py"),
        Path("src/rtgs/image2gs/structsplat_backend.py"),
    ]
    hashes = {path.as_posix(): sha256_file(ROOT / path) for path in paths}
    return {
        "git_revision": command_output("git", "rev-parse", "HEAD"),
        "files": hashes,
        "aggregate": canonical_hash(hashes),
    }


def camera_record() -> tuple[dict[str, Any], list[float]]:
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    record = next(item for item in payload["cameras"] if item["camera_id"].upper() == VIEW_ID)
    intrinsics = record["intrinsics"]
    matrix = intrinsics["camera_matrix"]
    width, height = (int(value) for value in intrinsics["resolution"])
    camera = {
        "width": width,
        "height": height,
        "fx": float(matrix[0]),
        "fy": float(matrix[4]),
        "cx": float(matrix[2]) + 0.5,
        "cy": float(matrix[5]) + 0.5,
    }
    distortion = [float(value) for value in intrinsics.get("distortion_coefficients", [])]
    return camera, distortion


def requested_config(*, n_start: int, n_max: int, iterations: int) -> FitConfig:
    return FitConfig(
        n_gaussians=n_start,
        max_gaussians=n_max,
        iterations=iterations,
        backend="structsplat",
        adaptive_density=False,
        growth_waves=1,
        relocate_fraction=0.0,
        structsplat_renderer="cuda_tiled",
        lr=1e-2,
        grad_init_mix=0.7,
        row_chunk=64,
        log_every=20,
        convergence_patience=0,
        convergence_tol=0.05,
        convergence_check_every=25,
        appearance_parameterization="weight_color_9p",
        freeze_geometry=False,
    )


def effective_external_configs(config: FitConfig, height: int, width: int) -> dict[str, Any]:
    """Mirror the source-bound adapter mapping and verify its exported digest later."""
    from structsplat.config import FitConfig as ExternalFitConfig
    from structsplat.config import InitConfig, StructureTensorConfig

    start = int(config.n_gaussians)
    maximum = start if config.max_gaussians is None else int(config.max_gaussians)
    add_total = maximum - start
    growth_count = max(1, math.ceil(add_total / config.growth_waves)) if add_total else 0
    growth_every = max(1, config.iterations // (config.growth_waves + 1)) if add_total else None
    feature_cap = 12.0 * max(height, width) / 160.0
    init = InitConfig(
        strategy="aniso_onedge",
        num_gaussians=start,
        seed=SEED,
        sampling_mode="wse",
        flank_offset_frac=0.0,
        scale_cap_mode="feature",
        scale_cap_max=feature_cap,
    )
    fit = ExternalFitConfig(
        iters=config.iterations,
        renderer="cuda_tiled",
        pixel_loss="l1",
        ssim_weight=0.3,
        log_every=max(1, config.log_every),
        split_every=(None if config.adaptive_density else growth_every),
        split_count=(0 if config.adaptive_density else growth_count),
        split_mode="residual_tensor_add",
        max_gaussians=maximum,
        adaptive_count=config.adaptive_density and add_total > 0,
        adaptive_growth_every=(growth_every or max(1, config.iterations)),
        adaptive_growth_count=max(1, growth_count),
        adaptive_split_mode="residual_tensor_add",
        adaptive_min_delta_psnr=config.convergence_tol,
        adaptive_patience=max(1, config.convergence_patience or 2),
        early_stop_patience=(config.convergence_patience or None),
        early_stop_min_delta=config.convergence_tol,
        early_stop_min_iters=max(config.convergence_check_every, config.iterations // 3),
        relocate_every=(
            growth_every if config.relocate_fraction > 0 and config.adaptive_density else None
        ),
        relocate_at_split=config.relocate_fraction > 0 and not config.adaptive_density,
        relocate_count=(math.ceil(growth_count * config.relocate_fraction) if growth_count else 0),
    )
    fit_dict = dataclasses.asdict(fit)
    return {
        "init": dataclasses.asdict(init),
        "structure_tensor": dataclasses.asdict(StructureTensorConfig()),
        "fit": fit_dict,
        "fit_config_digest": hashlib.sha256(
            json.dumps(fit_dict, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def observation_summary(field: GaussianObservationField) -> dict[str, Any]:
    tensors = {
        "means": field.means,
        "log_scales": field.log_scales,
        "rotations": field.rotations,
        "colors": field.colors,
        "amplitudes": field.amplitudes,
    }
    if field.mean_residuals is not None:
        tensors["mean_residuals"] = field.mean_residuals
    if field.color_grads is not None:
        tensors["color_grads"] = field.color_grads
    if field.filter_variance is not None:
        tensors["filter_variance"] = field.filter_variance
    return {
        "n_init_2d": field.n_init,
        "n_opt_2d": field.n,
        "width": field.width,
        "height": field.height,
        "fit_window": list(field.fit_window),
        "blend_mode": field.blend_mode,
        "epsilon": field.epsilon,
        "sigma_cutoff": field.sigma_cutoff,
        "support_fade_alpha": field.support_fade_alpha,
        "aa_dilation": field.aa_dilation,
        "view_id": field.view_id,
        "provider": field.provider,
        "producer_version": field.producer_version,
        "producer_source_digest": field.producer_source_digest,
        "fit_config_digest": field.fit_config_digest,
        "tensor_hashes": {name: tensor_hash(value) for name, value in tensors.items()},
    }


def render_observation(
    field: GaussianObservationField,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Render exactly on the fitted window and return an HWC CPU tensor."""
    from structsplat.gaussians import GaussianField
    from structsplat.render import render_field

    fit_x, fit_y, fit_width, fit_height = field.fit_window
    amplitudes = field.amplitudes.to(device)
    if torch.equal(amplitudes, torch.ones_like(amplitudes)):
        opacity_logits = None
    else:
        bounded = amplitudes.clamp(1e-6, 1.0 - 1e-6)
        opacity_logits = torch.logit(bounded)
    live = GaussianField(
        field.local_means().to(device),
        field.log_scales.to(device),
        field.rotations.to(device),
        field.colors.to(device),
        opacities=opacity_logits,
        color_grads=None if field.color_grads is None else field.color_grads.to(device),
        filter_variance=(
            None if field.filter_variance is None else field.filter_variance.to(device)
        ),
    )
    with torch.no_grad():
        rendered = render_field(
            live.means,
            live.conics(field.aa_dilation),
            live.colors,
            live.radii(field.sigma_cutoff, field.aa_dilation),
            fit_height,
            fit_width,
            mode="cuda_tiled",
            opacities=live.opacity_values(),
            scales=live.effective_scales(field.aa_dilation),
            rotations=live.rotations,
            support_fade=field.support_fade_alpha > 0.0,
            support_fade_alpha=field.support_fade_alpha,
            sigma_cutoff=field.sigma_cutoff,
            color_grads=live.color_grads,
        )
    return rendered.detach().cpu()


def support_union(field: GaussianObservationField) -> np.ndarray:
    """Exact union of StructSplat's clipped rounded support rectangles."""
    fit_x, fit_y, width, height = field.fit_window
    centers = field.support_pixels()
    centers[:, 0] -= fit_x
    centers[:, 1] -= fit_y
    radii = field.radii().long()
    lower = centers - radii
    upper = centers + radii
    x0 = lower[:, 0].clamp(0, width).cpu().numpy()
    y0 = lower[:, 1].clamp(0, height).cpu().numpy()
    x1 = (upper[:, 0] + 1).clamp(0, width).cpu().numpy()
    y1 = (upper[:, 1] + 1).clamp(0, height).cpu().numpy()
    diff = np.zeros((height + 1, width + 1), dtype=np.int32)
    np.add.at(diff, (y0, x0), 1)
    np.add.at(diff, (y1, x0), -1)
    np.add.at(diff, (y0, x1), -1)
    np.add.at(diff, (y1, x1), 1)
    return diff[:-1, :-1].cumsum(0).cumsum(1) > 0


def psnr(mse: float) -> float:
    return -10.0 * math.log10(max(float(mse), 1e-30))


def raw_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse = float((prediction - target).square().mean())
    clamped_mse = float((prediction.clamp(0, 1) - target).square().mean())
    return {
        "mse": mse,
        "psnr_db": psnr(mse),
        "clamped_mse": clamped_mse,
        "clamped_psnr_db": psnr(clamped_mse),
    }


def quantiles(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().float().reshape(-1)
    return {str(q): float(torch.quantile(flat, q)) for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)}


def evaluate(
    field: GaussianObservationField,
    *,
    rendered_crop: torch.Tensor,
    source_rgb: torch.Tensor,
    foreground: torch.Tensor,
    common_crop: tuple[int, int, int, int],
    seed: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    fit_x, fit_y, fit_width, fit_height = field.fit_window
    target_crop = source_rgb[fit_y : fit_y + fit_height, fit_x : fit_x + fit_width]
    mask_crop = foreground[fit_y : fit_y + fit_height, fit_x : fit_x + fit_width]
    masked_target_crop = target_crop * mask_crop[..., None]
    full_pixel_count = field.width * field.height

    crop_squared = (rendered_crop - masked_target_crop).square().sum()
    masked_full_mse = float(crop_squared / (3 * full_pixel_count))
    outside_rgb_squared = source_rgb.square().sum() - target_crop.square().sum()
    full_rgb_mse = float(
        ((rendered_crop - target_crop).square().sum() + outside_rgb_squared)
        / (3 * full_pixel_count)
    )
    crop_masked = raw_metrics(rendered_crop, masked_target_crop)
    crop_rgb = raw_metrics(rendered_crop, target_crop)
    foreground_metrics = raw_metrics(rendered_crop[mask_crop], target_crop[mask_crop])

    coverage = torch.from_numpy(support_union(field))
    integral = F.pad(mask_crop.to(torch.int64), (1, 0, 1, 0)).cumsum(0).cumsum(1)
    centers = field.support_pixels()
    center_x = centers[:, 0].clamp(0, field.width - 1)
    center_y = centers[:, 1].clamp(0, field.height - 1)
    center_on_foreground = foreground[center_y, center_x]
    radii = field.radii().long()
    local_centers = centers - centers.new_tensor([fit_x, fit_y])
    lower = local_centers - radii
    upper = local_centers + radii + 1
    x0 = lower[:, 0].clamp(0, fit_width)
    y0 = lower[:, 1].clamp(0, fit_height)
    x1 = upper[:, 0].clamp(0, fit_width)
    y1 = upper[:, 1].clamp(0, fit_height)
    foreground_in_rect = (
        integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    ) > 0

    generator = torch.Generator().manual_seed(seed)
    count = 8192
    sample_x = torch.randint(fit_width, (count,), generator=generator)
    sample_y = torch.randint(fit_height, (count,), generator=generator)
    native_xy = torch.stack([sample_x + fit_x, sample_y + fit_y], dim=-1).float() + 0.5
    with torch.no_grad():
        queried = field.query(native_xy, component_chunk=128).color
    parity = (queried - rendered_crop[sample_y, sample_x]).abs()

    common_x, common_y, common_width, common_height = common_crop
    if not (
        fit_x <= common_x
        and fit_y <= common_y
        and fit_x + fit_width >= common_x + common_width
        and fit_y + fit_height >= common_y + common_height
    ):
        raise RuntimeError("common comparison crop is not contained in an arm's fit window")
    full_render = torch.zeros(field.height, field.width, 3, dtype=rendered_crop.dtype)
    full_render[fit_y : fit_y + fit_height, fit_x : fit_x + fit_width] = rendered_crop
    common_render = full_render[
        common_y : common_y + common_height, common_x : common_x + common_width
    ]
    common_target_rgb = source_rgb[
        common_y : common_y + common_height, common_x : common_x + common_width
    ]
    common_mask = foreground[
        common_y : common_y + common_height, common_x : common_x + common_width
    ]
    common_target_masked = common_target_rgb * common_mask[..., None]
    coverage_common = coverage[
        common_y - fit_y : common_y - fit_y + common_height,
        common_x - fit_x : common_x - fit_x + common_width,
    ]
    preview_full = resize_hwc(full_render, height=768, width=888)
    preview_crop = resize_hwc(common_render, height=432, width=1188)

    scales = field.effective_variances().sqrt()
    metrics = {
        "masked_full_canvas": {
            "mse": masked_full_mse,
            "psnr_db": psnr(masked_full_mse),
        },
        "full_rgb_intent_mismatch": {
            "mse": full_rgb_mse,
            "psnr_db": psnr(full_rgb_mse),
            "note": "masked arms intentionally predict zero outside their fit window",
        },
        "masked_fit_crop": crop_masked,
        "rgb_fit_crop_intent_mismatch": crop_rgb,
        "common_mask_crop": raw_metrics(common_render, common_target_masked),
        "common_rgb_crop_intent_mismatch": raw_metrics(common_render, common_target_rgb),
        "foreground_rgb": foreground_metrics,
        "support": {
            "fit_crop_covered_fraction": float(coverage.float().mean()),
            "fit_crop_hole_fraction": float((~coverage).float().mean()),
            "foreground_covered_fraction": float(coverage[mask_crop].float().mean()),
            "foreground_hole_fraction": float((~coverage[mask_crop]).float().mean()),
            "common_mask_crop_covered_fraction": float(coverage_common.float().mean()),
            "common_mask_crop_hole_fraction": float((~coverage_common).float().mean()),
            "rectangle_area_sum": int(((x1 - x0).clamp_min(0) * (y1 - y0).clamp_min(0)).sum()),
        },
        "components": {
            "n_init_2d": field.n_init,
            "n_opt_2d": field.n,
            "centers_on_foreground": int(center_on_foreground.sum()),
            "centers_on_foreground_fraction": float(center_on_foreground.float().mean()),
            "support_rectangles_intersect_foreground": int(foreground_in_rect.sum()),
            "support_rectangles_intersect_foreground_fraction": float(
                foreground_in_rect.float().mean()
            ),
            "scale_px_quantiles": quantiles(scales),
            "radius_px_quantiles": quantiles(field.radii().float()),
            "color_outside_0_1_fraction": float(
                ((field.colors < 0) | (field.colors > 1)).float().mean()
            ),
        },
        "archive_query_cuda_raster_parity": {
            "sample_count": count,
            "max_abs_error": float(parity.max()),
            "mean_abs_error": float(parity.mean()),
        },
    }
    return metrics, preview_full, preview_crop


def resize_hwc(value: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    return F.interpolate(
        value.permute(2, 0, 1).unsqueeze(0),
        size=(height, width),
        mode="area",
    )[0].permute(1, 2, 0)


def save_rgb(path: Path, value: torch.Tensor) -> None:
    array = (value.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    Image.fromarray(array).save(path)


def labelled_contact_sheet(
    path: Path,
    *,
    labels: list[str],
    images: list[torch.Tensor],
    columns: int,
) -> None:
    if len(labels) != len(images):
        raise ValueError("labels and images must have equal length")
    converted = [
        Image.fromarray((value.clamp(0, 1) * 255).round().to(torch.uint8).numpy())
        for value in images
    ]
    cell_width = max(image.width for image in converted)
    cell_height = max(image.height for image in converted)
    label_height = 28
    rows = math.ceil(len(converted) / columns)
    canvas = Image.new("RGB", (columns * cell_width, rows * (cell_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(zip(labels, converted, strict=True)):
        x = (index % columns) * cell_width
        y = (index // columns) * (cell_height + label_height)
        draw.text((x + 8, y + 7), label, fill="black")
        canvas.paste(image, (x, y + label_height))
    canvas.save(path)


def run_fit(
    name: str,
    *,
    image: torch.Tensor,
    mask: torch.Tensor,
    config: FitConfig,
    device: torch.device,
    teacher_path: Path,
) -> tuple[GaussianObservationField, dict[str, Any]]:
    observations: list[GaussianObservationField] = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    _, history = fit_image(
        image.to(device),
        config,
        seed=SEED,
        mask=mask.to(device),
        observation_callback=observations.append,
        observation_view_id=VIEW_ID,
    )
    wall_seconds = time.perf_counter() - started
    if len(observations) != 1:
        raise RuntimeError(f"{name} exported {len(observations)} observations instead of one")
    teacher = observations[0].to("cpu")
    teacher.save_npz(teacher_path)
    runtime = {
        "wall_seconds": wall_seconds,
        "fit_seconds": history["fit_seconds"],
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "history": history,
    }
    return teacher, runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {out}")
    required_inputs = (CALIBRATION, IMAGE, MASK, BASELINE, BASELINE_ACQUISITION)
    if any(not path.is_file() for path in required_inputs):
        raise FileNotFoundError("one or more frozen experiment inputs are missing")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the native-resolution StructSplat screen requires CUDA")
    out.mkdir(parents=True)
    teacher_dir = out / "teachers"
    preview_dir = out / "previews"
    teacher_dir.mkdir()
    preview_dir.mkdir()

    try:
        camera, distortion = camera_record()
        width, height = camera["width"], camera["height"]
        source_rgb = _undistort(
            _resize_image(IMAGE, width, height),
            camera["fx"],
            camera["fy"],
            camera["cx"],
            camera["cy"],
            distortion,
        )
        foreground = (
            _undistort(
                _resize_image(MASK, width, height, mask=True),
                camera["fx"],
                camera["fy"],
                camera["cx"],
                camera["cy"],
                distortion,
                mask=True,
            )
            > 0.5
        )
        cropped_image, _, offset = _crop_to_mask(source_rgb, foreground)
        common_crop = (
            int(offset[0]),
            int(offset[1]),
            int(cropped_image.shape[1]),
            int(cropped_image.shape[0]),
        )

        source_binding = repository_source_binding()
        external_binding = structsplat_source_binding()
        acquisition = json.loads(BASELINE_ACQUISITION.read_text(encoding="utf-8"))
        baseline_record = next(
            item for item in acquisition["views"] if item["camera_id"] == VIEW_ID
        )
        if baseline_record["teacher_sha256"] != sha256_file(BASELINE):
            raise RuntimeError("frozen baseline teacher hash differs from its acquisition record")
        if (
            external_binding["provider_source_digest"]
            != baseline_record["external_structsplat"]["provider_source_digest"]
        ):
            raise RuntimeError("StructSplat sources drifted from the frozen baseline")

        configs = {
            "masked_640_100": requested_config(n_start=640, n_max=640, iterations=100),
            "masked_growth_1280_200": requested_config(n_start=640, n_max=1280, iterations=200),
        }
        effective = {
            name: effective_external_configs(config, common_crop[3], common_crop[2])
            for name, config in configs.items()
        }
        plan = {
            "artifact_type": "compact_stage1_mask_screen_plan_v1",
            "decision_bearing": False,
            "view_id": VIEW_ID,
            "seed": SEED,
            "winner_criterion": "raw foreground_rgb.psnr_db",
            "source_rgb_boundary": (
                "RGB is allowed only for Stage-1 fit and isolated evaluation; exported teachers "
                "contain no RGB tensor or path."
            ),
            "camera": camera,
            "distortion": distortion,
            "common_mask_crop": list(common_crop),
            "inputs": {
                "calibration": {
                    "path": str(CALIBRATION.relative_to(ROOT)),
                    "sha256": sha256_file(CALIBRATION),
                },
                "rgb": {"path": str(IMAGE.relative_to(ROOT)), "sha256": sha256_file(IMAGE)},
                "mask": {"path": str(MASK.relative_to(ROOT)), "sha256": sha256_file(MASK)},
                "baseline_teacher": {
                    "path": str(BASELINE.relative_to(ROOT)),
                    "sha256": sha256_file(BASELINE),
                },
                "undistorted_rgb_tensor_sha256": tensor_hash(source_rgb),
                "undistorted_mask_tensor_sha256": tensor_hash(foreground),
            },
            "repository": source_binding,
            "external_structsplat": external_binding,
            "environment": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "ld_preload": os.environ.get("LD_PRELOAD"),
            },
            "arms": {
                "full_frame_640_100_frozen": {
                    "mode": "load_frozen_archive",
                    "requested_config": acquisition_config(BASELINE_ACQUISITION),
                    "fit_config_digest": baseline_record["teacher_semantic_hashes"]["metadata"][
                        "fit_config_digest"
                    ],
                },
                **{
                    name: {
                        "mode": "fit_current_source_masked_crop",
                        "requested_config": dataclasses.asdict(config),
                        "effective_structsplat": effective[name],
                    }
                    for name, config in configs.items()
                },
            },
        }
        (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

        arms: dict[str, dict[str, Any]] = {}
        baseline_copy = teacher_dir / "full_frame_640_100_frozen.teacher.npz"
        shutil.copyfile(BASELINE, baseline_copy)
        baseline = GaussianObservationField.load_npz(baseline_copy, strict=True)
        arms["full_frame_640_100_frozen"] = {
            "teacher": baseline,
            "teacher_path": baseline_copy,
            "runtime": {"mode": "loaded_frozen_archive", "historical": baseline_record["history"]},
        }
        for name, config in configs.items():
            teacher_path = teacher_dir / f"{name}.teacher.npz"
            teacher, runtime = run_fit(
                name,
                image=source_rgb,
                mask=foreground,
                config=config,
                device=device,
                teacher_path=teacher_path,
            )
            if teacher.fit_config_digest != effective[name]["fit_config_digest"]:
                raise RuntimeError(f"{name} external FitConfig digest differs from adapter mapping")
            arms[name] = {"teacher": teacher, "teacher_path": teacher_path, "runtime": runtime}

        target_full = resize_hwc(source_rgb, height=768, width=888)
        masked_full = resize_hwc(source_rgb * foreground[..., None], height=768, width=888)
        crop_x, crop_y, crop_width, crop_height = common_crop
        target_crop = resize_hwc(
            source_rgb[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
            * foreground[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width, None],
            height=432,
            width=1188,
        )
        full_labels = ["source RGB (evaluation)", "masked target"]
        full_images = [target_full, masked_full]
        crop_labels = ["masked target crop"]
        crop_images = [target_crop]
        result_arms: dict[str, Any] = {}
        for index, (name, state) in enumerate(arms.items()):
            teacher = state["teacher"]
            rendered = render_observation(teacher, device=device)
            metrics, preview_full, preview_crop = evaluate(
                teacher,
                rendered_crop=rendered,
                source_rgb=source_rgb,
                foreground=foreground,
                common_crop=common_crop,
                seed=17072026 + index,
            )
            full_path = preview_dir / f"{name}_full.png"
            crop_path = preview_dir / f"{name}_mask_crop.png"
            save_rgb(full_path, preview_full)
            save_rgb(crop_path, preview_crop)
            full_labels.append(name)
            full_images.append(preview_full)
            crop_labels.append(name)
            crop_images.append(preview_crop)
            result_arms[name] = {
                "teacher": observation_summary(teacher),
                "teacher_path": str(state["teacher_path"].relative_to(out)),
                "teacher_sha256": sha256_file(state["teacher_path"]),
                "runtime": state["runtime"],
                "metrics": metrics,
                "previews": {
                    "full": str(full_path.relative_to(out)),
                    "mask_crop": str(crop_path.relative_to(out)),
                },
            }
            del rendered
            torch.cuda.empty_cache()

        full_contact = out / "full_target_vs_teachers.png"
        crop_contact = out / "mask_crop_target_vs_teachers.png"
        labelled_contact_sheet(
            full_contact, labels=full_labels, images=full_images, columns=len(full_images)
        )
        labelled_contact_sheet(
            crop_contact, labels=crop_labels, images=crop_images, columns=len(crop_images)
        )
        winner = max(
            result_arms,
            key=lambda name: result_arms[name]["metrics"]["foreground_rgb"]["psnr_db"],
        )
        winner_path = out / "winner.teacher.npz"
        shutil.copyfile(out / result_arms[winner]["teacher_path"], winner_path)

        import structsplat.cuda_render as cuda_render

        if repository_source_binding() != source_binding:
            raise RuntimeError("repository experiment sources changed during execution")
        if structsplat_source_binding() != external_binding:
            raise RuntimeError("StructSplat sources changed during execution")
        for binding in plan["inputs"].values():
            path_text = binding.get("path") if isinstance(binding, dict) else None
            expected_hash = binding.get("sha256") if isinstance(binding, dict) else None
            if (
                path_text is not None
                and expected_hash is not None
                and sha256_file(ROOT / path_text) != expected_hash
            ):
                raise RuntimeError(f"experiment input changed during execution: {path_text}")

        extension = getattr(cuda_render, "_EXT", None)
        extension_path = None if extension is None else Path(extension.__file__).resolve()
        result = {
            "artifact_type": "compact_stage1_mask_screen_result_v1",
            "status": "PASS",
            "decision_bearing": False,
            "plan_sha256": sha256_file(out / "plan.json"),
            "result_scope": (
                "exploratory one-view Stage-1 mechanism evidence; no default, 3D lift, "
                "novel-view, or end-to-end claim"
            ),
            "winner_criterion": "raw foreground_rgb.psnr_db",
            "winner": winner,
            "winner_teacher": "winner.teacher.npz",
            "winner_teacher_sha256": sha256_file(winner_path),
            "arms": result_arms,
            "contact_sheets": {
                "full": str(full_contact.relative_to(out)),
                "full_sha256": sha256_file(full_contact),
                "mask_crop": str(crop_contact.relative_to(out)),
                "mask_crop_sha256": sha256_file(crop_contact),
            },
            "loaded_structsplat_extension": (
                None
                if extension_path is None
                else {"path": str(extension_path), "sha256": sha256_file(extension_path)}
            ),
            "viewer": {
                "status": "NOT_APPLICABLE_STAGE1_ONLY",
                "reason": (
                    "rtgs view consumes 3D PLYs; this experiment ends at RGB-free 2D teachers"
                ),
            },
        }
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    except BaseException as error:
        failure = {
            "artifact_type": "compact_stage1_mask_screen_failure_v1",
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        (out / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise


def acquisition_config(path: Path) -> dict[str, Any]:
    plan = json.loads((path.parent / "CALIBRATED_PLAN.json").read_text(encoding="utf-8"))
    return plan["configuration"]["fit"]


if __name__ == "__main__":
    raise SystemExit(main())
