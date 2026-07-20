#!/usr/bin/env python3
"""Fit one truly mask-gated StructSplat field under a strict 168,000-byte cap.

This is a single-view feasibility example, not a dataset-wide conversion.  It differs from the
normal Stage-1 adapter in two important ways:

* the StructSplat initialization density is zero outside the foreground mask; and
* the optimization objective contains only foreground pixels and mask-normalized local SSIM.

The source mask is used only while preprocessing, fitting, and evaluating.  It is not serialized
in the resulting :class:`GaussianObservationField`.  Because normalized RGB splatting has no
separate alpha channel, the report and panel expose the raw unmasked render and outside-mask
leakage rather than hiding it with the source mask.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from rtgs.core.observation2d import GaussianObservationField
from stage1_capacity_sweep import (
    LOCAL_LIBSTDCXX_PRELOAD,
    ROOT,
    extension_binding,
    prepare_view,
    sha256_file,
    structsplat_source_binding,
    tensor_hash,
    weighted_error_metrics,
    weighted_ssim,
)

VIEW_ID = "C0014"
BYTE_CAP = 168_000
N_GAUSSIANS = 5_000
ITERATIONS = 1_000
RENDERER = "cuda_tiled"
SIGMA_CUTOFF = 3.0
SUPPORT_FADE_ALPHA = 1.0
SSIM_WEIGHT = 0.3
LOG_EVERY = 50
DEFAULT_OUT = ROOT / "runs/structsplat_masked_168kb_example_20260718"


def _gaussian_kernel(
    window: int,
    sigma: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window, device=device, dtype=dtype) - (window - 1) / 2
    kernel = torch.exp(-coords.square() / (2 * sigma * sigma))
    return kernel / kernel.sum()


def _separable_filter(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    channels = image.shape[1]
    radius = kernel.numel() // 2
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    value = F.conv2d(image, vertical, padding=(radius, 0), groups=channels)
    return F.conv2d(value, horizontal, padding=(0, radius), groups=channels)


class MaskedSSIM:
    """Differentiable SSIM whose local moments and output mean use foreground only."""

    def __init__(
        self,
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        window: int = 11,
        sigma: float = 1.5,
    ) -> None:
        if target.ndim != 3 or target.shape[-1] != 3:
            raise ValueError("target must have shape HxWx3")
        if mask.shape != target.shape[:2]:
            raise ValueError("mask must match target")
        self.mask = mask.to(target).clamp(0, 1)[None, None]
        self.target = target.permute(2, 0, 1).unsqueeze(0)
        self.kernel = _gaussian_kernel(
            window,
            sigma,
            device=target.device,
            dtype=target.dtype,
        )
        with torch.no_grad():
            self.normalizer = _separable_filter(self.mask, self.kernel).clamp_min(1e-6)
            self.target_mean = (
                _separable_filter(self.target * self.mask, self.kernel) / self.normalizer
            )
            self.target_variance = (
                _separable_filter(self.target.square() * self.mask, self.kernel) / self.normalizer
                - self.target_mean.square()
            ).clamp_min(0)

    def __call__(self, prediction: torch.Tensor) -> torch.Tensor:
        pred = prediction.permute(2, 0, 1).unsqueeze(0)
        pred_mean = _separable_filter(pred * self.mask, self.kernel) / self.normalizer
        pred_variance = (
            _separable_filter(pred.square() * self.mask, self.kernel) / self.normalizer
            - pred_mean.square()
        ).clamp_min(0)
        covariance = (
            _separable_filter(pred * self.target * self.mask, self.kernel) / self.normalizer
            - pred_mean * self.target_mean
        )
        c1, c2 = 0.01**2, 0.03**2
        similarity = ((2 * pred_mean * self.target_mean + c1) * (2 * covariance + c2)) / (
            (pred_mean.square() + self.target_mean.square() + c1)
            * (pred_variance + self.target_variance + c2)
        )
        return (similarity * self.mask).sum() / (self.mask.sum() * 3)


def _render_live(field: Any, *, height: int, width: int) -> torch.Tensor:
    from structsplat.render import render_field

    return render_field(
        field.means,
        field.conics(0.0),
        field.colors,
        field.radii(SIGMA_CUTOFF, 0.0),
        height,
        width,
        chunk=512,
        mode=RENDERER,
        opacities=field.opacity_values(),
        scales=field.effective_scales(0.0),
        rotations=field.rotations,
        support_fade=True,
        support_fade_alpha=SUPPORT_FADE_ALPHA,
        sigma_cutoff=SIGMA_CUTOFF,
        color_grads=field.color_grads,
    )


def _rounded_centers_inside(field: Any, mask: torch.Tensor) -> torch.Tensor:
    x = field.means[:, 0].detach().round().long().clamp(0, mask.shape[1] - 1)
    y = field.means[:, 1].detach().round().long().clamp(0, mask.shape[0] - 1)
    in_bounds = (
        (field.means[:, 0].detach() >= 0)
        & (field.means[:, 0].detach() <= mask.shape[1] - 1)
        & (field.means[:, 1].detach() >= 0)
        & (field.means[:, 1].detach() <= mask.shape[0] - 1)
    )
    return in_bounds & mask[y, x].bool()


@torch.no_grad()
def _project_centers_to_foreground(
    field: Any,
    mask: torch.Tensor,
    nearest_y: torch.Tensor,
    nearest_x: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    """Project invalid rounded centers to their nearest foreground pixel and clear momentum."""
    field.means[:, 0].clamp_(0, mask.shape[1] - 1)
    field.means[:, 1].clamp_(0, mask.shape[0] - 1)
    x = field.means[:, 0].round().long()
    y = field.means[:, 1].round().long()
    invalid = ~mask[y, x].bool()
    if bool(invalid.any()):
        bad_x = x[invalid]
        bad_y = y[invalid]
        field.means[invalid, 0] = nearest_x[bad_y, bad_x].to(field.means)
        field.means[invalid, 1] = nearest_y[bad_y, bad_x].to(field.means)
        state = optimizer.state.get(field.means, {})
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(key)
            if torch.is_tensor(value) and value.shape == field.means.shape:
                value[invalid] = 0
    return invalid


@torch.no_grad()
def _support_diagnostics(field: Any, mask: torch.Tensor) -> dict[str, Any]:
    """Measure exact finite-support activity on foreground/background pixels."""
    from structsplat.render import (
        _element_budget,
        _flat_tile_slices,
        _support_weight,
        _tile_bounds,
        _tile_coords,
    )

    height, width = mask.shape
    conics = field.conics(0.0)
    radii = field.radii(SIGMA_CUTOFF, 0.0)
    x0, y0, tx, counts = _tile_bounds(field.means, radii, height, width)
    total = torch.zeros(field.n, device=field.means.device, dtype=field.means.dtype)
    foreground = torch.zeros_like(total)
    denominator = torch.zeros(height * width, device=field.means.device, dtype=field.means.dtype)
    opacities = field.opacity_values()
    for start, end in _flat_tile_slices(counts, _element_budget(512)):
        gid, px, py = _tile_coords(
            x0,
            y0,
            tx,
            counts,
            start,
            end,
            field.means.device,
        )
        dx = px.to(field.means.dtype) - field.means[gid, 0]
        dy = py.to(field.means.dtype) - field.means[gid, 1]
        a, b, c = conics[gid, 0], conics[gid, 1], conics[gid, 2]
        q = a * dx.square() + 2 * b * dx * dy + c * dy.square()
        weights = _support_weight(
            q,
            SIGMA_CUTOFF,
            True,
            SUPPORT_FADE_ALPHA,
        )
        if opacities is not None:
            weights = weights * opacities[gid]
        total.index_add_(0, gid, weights)
        foreground.index_add_(0, gid, weights * mask[py, px].to(weights))
        denominator.index_add_(0, py * width + px, weights)
    foreground_fraction = foreground / total.clamp_min(1e-12)
    coverage = denominator.reshape(height, width) > 1e-8
    target = mask.bool()
    intersection = int((coverage & target).sum())
    union = int((coverage | target).sum())
    return {
        "components": {
            "count": int(field.n),
            "zero_total_activity": int((total <= 1e-12).sum()),
            "foreground_activity_fraction_mean": float(foreground_fraction.mean()),
            "foreground_activity_fraction_p05": float(torch.quantile(foreground_fraction, 0.05)),
            "foreground_activity_fraction_p50": float(torch.quantile(foreground_fraction, 0.50)),
            "rows_below_50pct_foreground_activity": int((foreground_fraction < 0.5).sum()),
            "rows_below_10pct_foreground_activity": int((foreground_fraction < 0.1).sum()),
        },
        "coverage_at_weight_gt_1e_8": {
            "foreground_pixels": int(target.sum()),
            "covered_pixels": int(coverage.sum()),
            "intersection_pixels": intersection,
            "union_pixels": union,
            "iou": intersection / max(union, 1),
            "foreground_recall": intersection / max(int(target.sum()), 1),
            "outside_covered_pixels": int((coverage & ~target).sum()),
        },
    }


def _resize_hwc(value: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    return F.interpolate(
        value.permute(2, 0, 1).unsqueeze(0),
        size=(height, width),
        mode="area",
    )[0].permute(1, 2, 0)


def _save_panel(
    path: Path,
    *,
    target: torch.Tensor,
    raw_render: torch.Tensor,
    mask: torch.Tensor,
    foreground_psnr: float,
    foreground_ssim: float,
) -> None:
    """Save target, raw archive playback, masked diagnostic, and unhidden raw error."""
    height, width = 420, 1066
    raw = raw_render.clamp(0, 1)
    masked_render = raw * mask[..., None]
    error = (raw - target).abs().mean(dim=-1, keepdim=True)
    scale = torch.quantile(error.reshape(-1), 0.99).clamp_min(1e-6)
    normalized = (error / scale).clamp(0, 1)
    error_rgb = torch.cat(
        [normalized, 0.2 * normalized, torch.zeros_like(normalized)],
        dim=-1,
    )
    cells = [
        _resize_hwc(target, height=height, width=width),
        _resize_hwc(raw, height=height, width=width),
        _resize_hwc(masked_render, height=height, width=width),
        _resize_hwc(error_rgb, height=height, width=width),
    ]
    labels = [
        "masked source target",
        "raw StructSplat archive render (NO mask applied)",
        (
            "foreground diagnostic only (source mask applied) | "
            f"PSNR {foreground_psnr:.3f} dB | wSSIM {foreground_ssim:.5f}"
        ),
        "raw-render absolute RGB error (red, normalized at p99)",
    ]
    label_height = 34
    canvas = Image.new("RGB", (width * len(cells), height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, value) in enumerate(zip(labels, cells, strict=True)):
        image = Image.fromarray((value.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy())
        x = index * width
        draw.text((x + 8, 9), label, fill="black")
        canvas.paste(image, (x, label_height))
    canvas.save(path)


def _archive_members(path: Path) -> list[str]:
    with np.load(path, allow_pickle=False) as data:
        return sorted(data.files)


def run(out: Path, *, device_text: str) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out}")
    device = torch.device(device_text)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the native StructSplat example requires a CUDA device")
    extension = extension_binding()
    source_binding = structsplat_source_binding()
    prepared = prepare_view(VIEW_ID)
    fit_x, fit_y, width, height = prepared.fit_window
    rgb_crop = prepared.rgb[fit_y : fit_y + height, fit_x : fit_x + width].contiguous()
    mask_cpu = prepared.mask_crop.bool().contiguous()
    masked_target = (rgb_crop * mask_cpu[..., None]).contiguous()
    if tensor_hash(masked_target) != prepared.input_record["masked_target_crop_tensor_sha256"]:
        raise RuntimeError("reconstructed masked target differs from frozen preprocessing")

    from structsplat import density as density_module
    from structsplat import structure_tensor as tensor_module
    from structsplat.config import FitConfig, InitConfig, StructureTensorConfig
    from structsplat.init import build_field

    init_config = InitConfig(
        strategy="aniso_onedge",
        num_gaussians=N_GAUSSIANS,
        seed=prepared.seed,
        sampling_mode="wse",
        flank_offset_frac=0.0,
        scale_cap_mode="feature",
        scale_cap_max=12.0 * max(height, width) / 160.0,
        background_fraction=0.0,
        background_grid=0,
    )
    tensor_config = StructureTensorConfig()
    fit_config = FitConfig(
        iters=ITERATIONS,
        renderer=RENDERER,
        pixel_loss="l1",
        ssim_weight=SSIM_WEIGHT,
        log_every=LOG_EVERY,
        support_fade=True,
        sigma_cutoff=SIGMA_CUTOFF,
        max_gaussians=N_GAUSSIANS,
    )
    fit_contract = {
        "initializer": dataclasses.asdict(init_config),
        "structure_tensor": dataclasses.asdict(tensor_config),
        "optimizer": {
            "name": "Adam",
            "iterations": ITERATIONS,
            "lr_means": fit_config.lr_means,
            "lr_scales": fit_config.lr_scales,
            "lr_rotation": fit_config.lr_rot,
            "lr_color": fit_config.lr_color,
        },
        "renderer": RENDERER,
        "sigma_cutoff": SIGMA_CUTOFF,
        "support_fade_alpha": SUPPORT_FADE_ALPHA,
        "objective": ("0.7 * foreground-only L1 + 0.3 * foreground-mask-normalized SSIM"),
        "hard_center_constraint": "nearest foreground pixel with Adam momentum reset",
    }
    fit_config_digest = hashlib.sha256(
        json.dumps(fit_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    masked_np = masked_target.numpy().astype(np.float32, copy=False)
    mask_np = mask_cpu.numpy().astype(np.float64, copy=False)
    tensor = tensor_module.compute(masked_np, tensor_config)
    density = density_module.density_from_tensor_and_image(
        masked_np,
        tensor,
        init_config,
        tensor_config,
    )
    density = np.maximum(np.asarray(density, dtype=np.float64), 0.0) * mask_np
    density_sum = float(density.sum())
    if not math.isfinite(density_sum) or density_sum <= 0:
        raise RuntimeError("masked StructSplat initialization density has no finite mass")
    density /= density_sum

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    field = build_field(
        rgb_crop.numpy().astype(np.float32, copy=False),
        init_config,
        tensor_config,
        density=density,
        tensor=tensor,
        device=str(device),
    )
    mask = mask_cpu.to(device)
    initial_inside = _rounded_centers_inside(field, mask)
    if not bool(initial_inside.all()):
        raise RuntimeError(
            f"masked initialization placed {int((~initial_inside).sum())} centers outside mask"
        )
    field.trainable()
    optimizer = torch.optim.Adam(
        field.parameter_groups(
            fit_config.lr_means,
            fit_config.lr_scales,
            fit_config.lr_rot,
            fit_config.lr_color,
            fit_config.lr_opacity,
        )
    )
    target = rgb_crop.to(device)
    mask_float = mask.to(target)
    masked_ssim = MaskedSSIM(target, mask_float)
    _, nearest_indices = distance_transform_edt(~mask_np.astype(bool), return_indices=True)
    nearest_y = torch.from_numpy(nearest_indices[0].copy()).to(device=device)
    nearest_x = torch.from_numpy(nearest_indices[1].copy()).to(device=device)
    history: dict[str, list[Any]] = {
        "iteration": [],
        "loss": [],
        "masked_l1": [],
        "masked_ssim": [],
        "foreground_psnr_db": [],
        "projected_center_updates": [],
    }
    projected_total = 0
    projected_rows = torch.zeros(field.n, dtype=torch.bool, device=device)
    lo, hi = math.log(0.35), math.log(max(height, width))
    for iteration in range(ITERATIONS):
        prediction = _render_live(field, height=height, width=width)
        error_map = (prediction - target).abs().mean(dim=-1)
        pixel_loss = (error_map * mask_float).sum() / mask_float.sum()
        similarity = masked_ssim(prediction)
        loss = (1.0 - SSIM_WEIGHT) * pixel_loss + SSIM_WEIGHT * (1.0 - similarity)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            field.log_scales.clamp_(lo, hi)
            if getattr(field, "scale_max", None) is not None:
                cap = torch.log(torch.clamp(field.scale_max, min=1e-3))
                torch.minimum(field.log_scales, cap, out=field.log_scales)
            projected = _project_centers_to_foreground(
                field,
                mask,
                nearest_y,
                nearest_x,
                optimizer,
            )
            projected_total += int(projected.sum())
            projected_rows |= projected
            if iteration % LOG_EVERY == 0 or iteration == ITERATIONS - 1:
                squared = (prediction - target).square().mean(dim=-1)
                foreground_mse = (squared * mask_float).sum() / mask_float.sum()
                foreground_psnr = -10.0 * torch.log10(foreground_mse.clamp_min(1e-30))
                history["iteration"].append(iteration)
                history["loss"].append(float(loss))
                history["masked_l1"].append(float(pixel_loss))
                history["masked_ssim"].append(float(similarity))
                history["foreground_psnr_db"].append(float(foreground_psnr))
                history["projected_center_updates"].append(int(projected.sum()))

    torch.cuda.synchronize(device)
    fit_seconds = time.perf_counter() - started
    final_inside = _rounded_centers_inside(field, mask)
    if not bool(final_inside.all()):
        raise RuntimeError(
            f"terminal field has {int((~final_inside).sum())} rounded centers outside mask"
        )
    live_render = _render_live(field, height=height, width=width).detach().cpu()
    support = _support_diagnostics(field, mask)

    from rtgs.image2gs.structsplat_backend import field_to_observation

    observation = field_to_observation(
        field,
        canvas_size=(prepared.camera["height"], prepared.camera["width"]),
        fit_window=prepared.fit_window,
        blend_mode="normalized",
        sigma_cutoff=SIGMA_CUTOFF,
        support_fade_alpha=SUPPORT_FADE_ALPHA,
        aa_dilation=0.0,
        view_id=VIEW_ID,
        n_init=N_GAUSSIANS,
        producer_version=source_binding["version"],
        producer_source_digest=source_binding["provider_source_digest"],
        fit_config_digest=fit_config_digest,
    ).to("cpu")

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        assert temporary is not None
        teacher_path = temporary / f"{VIEW_ID}.teacher.npz"
        panel_path = temporary / f"{VIEW_ID}_comparison.png"
        result_path = temporary / "result.json"
        observation.save_npz(teacher_path)
        teacher_bytes = teacher_path.stat().st_size
        if teacher_bytes > BYTE_CAP:
            raise RuntimeError(
                f"teacher archive is {teacher_bytes} bytes, above the {BYTE_CAP}-byte cap"
            )
        reloaded = GaussianObservationField.load_npz(teacher_path, strict=True)
        archive_render = _render_archive(reloaded, device=device)
        parity = (live_render - archive_render).abs()
        if float(parity.max()) > 1e-5 or float(parity.mean()) > 1e-6:
            raise RuntimeError(
                "strict archive reload changed native StructSplat playback: "
                f"max={float(parity.max())}, mean={float(parity.mean())}"
            )

        foreground = weighted_error_metrics(archive_render, masked_target, mask_cpu)
        fit_crop = weighted_error_metrics(
            archive_render,
            masked_target,
            torch.ones_like(mask_cpu, dtype=torch.float32),
        )
        foreground_ssim = weighted_ssim(
            archive_render.clamp(0, 1).to(device),
            masked_target.to(device),
            mask_float,
        )
        fit_crop_ssim = weighted_ssim(
            archive_render.clamp(0, 1).to(device),
            masked_target.to(device),
            torch.ones_like(mask_float),
        )
        foreground["clamped"]["weighted_ssim"] = foreground_ssim
        fit_crop["clamped"]["weighted_ssim"] = fit_crop_ssim
        outside = ~mask_cpu
        outside_rgb = archive_render.clamp(0, 1)[outside]
        outside_metrics = {
            "pixel_count": int(outside.sum()),
            "mean_absolute_rgb": float(outside_rgb.abs().mean()),
            "p99_absolute_rgb": float(torch.quantile(outside_rgb.abs(), 0.99)),
            "pixels_above_one_8bit_code_fraction": float(
                (outside_rgb.abs().amax(dim=-1) > (1.0 / 255.0)).float().mean()
            ),
        }
        _save_panel(
            panel_path,
            target=masked_target,
            raw_render=archive_render,
            mask=mask_cpu.to(archive_render),
            foreground_psnr=foreground["clamped"]["psnr_db"],
            foreground_ssim=foreground_ssim,
        )
        members = _archive_members(teacher_path)
        forbidden = [
            name
            for name in members
            if any(token in name.lower() for token in ("rgb", "image", "mask", "target"))
        ]
        if forbidden:
            raise RuntimeError(f"archive unexpectedly contains source payload members: {forbidden}")
        rgb_path = ROOT / prepared.input_record["rgb"]["path"]
        mask_path = ROOT / prepared.input_record["mask"]["path"]
        result = {
            "artifact_type": "structsplat_masked_168kb_example_v1",
            "status": "PASS",
            "decision_bearing": False,
            "scope": (
                "single masked C0014 feasibility example; no whole-dataset size or quality claim"
            ),
            "view_id": VIEW_ID,
            "archive": {
                "path": teacher_path.name,
                "sha256": sha256_file(teacher_path),
                "bytes": teacher_bytes,
                "strict_cap_bytes": BYTE_CAP,
                "margin_bytes": BYTE_CAP - teacher_bytes,
                "members": members,
                "contains_rgb_or_mask_payload": False,
                "n_gaussians": reloaded.n,
                "canvas_size": [reloaded.width, reloaded.height],
                "fit_window": list(reloaded.fit_window),
            },
            "source_payload": {
                "rgb_path": prepared.input_record["rgb"]["path"],
                "rgb_sha256": prepared.input_record["rgb"]["sha256"],
                "rgb_bytes": rgb_path.stat().st_size,
                "mask_path": prepared.input_record["mask"]["path"],
                "mask_sha256": prepared.input_record["mask"]["sha256"],
                "mask_bytes": mask_path.stat().st_size,
                "rgb_to_archive_ratio": rgb_path.stat().st_size / teacher_bytes,
                "rgb_plus_mask_to_archive_ratio": (
                    rgb_path.stat().st_size + mask_path.stat().st_size
                )
                / teacher_bytes,
                "serialized_after_conversion": False,
            },
            "preprocessing": {
                "undistort_rgb": "calibrated bilinear",
                "undistort_mask": "calibrated nearest then threshold_gt_0.5",
                "mask_crop": list(prepared.fit_window),
                "initialization_density": (
                    "StructSplat density on masked RGB, multiplied by binary mask and renormalized"
                ),
                "objective": ("0.7 * foreground-only L1 + 0.3 * foreground-mask-normalized SSIM"),
                "background_pixel_supervision": False,
                "initial_centers_outside_mask": int((~initial_inside).sum()),
                "final_centers_outside_mask": int((~final_inside).sum()),
                "center_projection_updates": projected_total,
                "unique_center_rows_projected": int(projected_rows.sum()),
            },
            "quality": {
                "foreground_rgb": foreground,
                "raw_render_vs_masked_crop": fit_crop,
                "outside_mask_raw_render": outside_metrics,
                "support": support,
                "native_live_vs_archive_render": {
                    "max_abs_error": float(parity.max()),
                    "mean_abs_error": float(parity.mean()),
                    "live_render_sha256": tensor_hash(live_render),
                    "archive_render_sha256": tensor_hash(archive_render),
                },
            },
            "fit": {
                "contract": fit_contract,
                "fit_config_digest": fit_config_digest,
                "fit_seconds": fit_seconds,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "history": history,
            },
            "bindings": {
                "structsplat": source_binding,
                "cuda_extension": extension,
                "masked_target_crop_tensor_sha256": tensor_hash(masked_target),
                "archive_render_tensor_sha256": tensor_hash(archive_render),
            },
            "preview": {
                "path": panel_path.name,
                "note": (
                    "The second cell is raw archive playback with no source mask. "
                    "The third cell applies the source mask only as a "
                    "foreground-quality diagnostic."
                ),
            },
        }
        result_path.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, out)
        temporary = None
        return result
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def _render_archive(
    observation: GaussianObservationField,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Rebuild StructSplat's native field and replay the exact archive semantics."""
    from structsplat.gaussians import GaussianField

    _, _, width, height = observation.fit_window
    amplitudes = observation.amplitudes.to(device)
    if torch.equal(amplitudes, torch.ones_like(amplitudes)):
        opacity_logits = None
    else:
        opacity_logits = torch.logit(amplitudes.clamp(1e-6, 1.0 - 1e-6))
    field = GaussianField(
        observation.local_means().to(device),
        observation.log_scales.to(device),
        observation.rotations.to(device),
        observation.colors.to(device),
        opacities=opacity_logits,
        color_grads=(
            None if observation.color_grads is None else observation.color_grads.to(device)
        ),
        filter_variance=(
            None if observation.filter_variance is None else observation.filter_variance.to(device)
        ),
    )
    return _render_live(field, height=height, width=width).detach().cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if LOCAL_LIBSTDCXX_PRELOAD.is_file() and not os.environ.get("LD_PRELOAD"):
        raise RuntimeError(
            "the current CUDA extension needs the system libstdc++; rerun with "
            f"LD_PRELOAD={LOCAL_LIBSTDCXX_PRELOAD}"
        )
    result = run(args.out.resolve(), device_text=args.device)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
