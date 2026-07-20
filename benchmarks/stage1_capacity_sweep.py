#!/usr/bin/env python3
"""Exploratory native-resolution StructSplat Stage-1 capacity/fidelity sweep.

The default protocol fits only C0014, a representative member of the frozen seven-view training
set.  C1004 is held out and cannot be named by the CLI.  RGB and masks are available only while
preparing, fitting, and evaluating Stage 1; each fitted field is immediately exported as a
lossless :class:`GaussianObservationField`.

The protocol is deliberately split into ``plan``, one-process-per-cell ``run``, and ``assemble``
actions.  That makes long CUDA cells resumable and keeps peak-memory measurements independent.
This remains exploratory Stage-1 evidence: it makes no 3D-lift or novel-view claim.
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
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.calibrated import _resize_image, _undistort
from rtgs.image2gs.fit import _crop_to_mask

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008"
CALIBRATION = ROOT / "dataset/2025_03_07_stage_with_fabric/calibration_dome.json"
DEFAULT_OUT = ROOT / "runs/stage1_capacity_sweep_C0014_20260717"
STRUCTSPLAT_ROOT = Path("~/Documents/structsplat").expanduser().resolve()

TRAIN_VIEWS = ("C0001", "C0008", "C0014", "C0021", "C0026", "C0031", "C0039")
DEFAULT_VIEWS = ("C0014",)
HELDOUT_VIEW = "C1004"
VIEW_SEEDS = {view_id: index for index, view_id in enumerate(TRAIN_VIEWS)}
RENDERER = "cuda_tiled"
QUERY_SAMPLE_COUNT = 8192
SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
# Preserve the raw schema-v1 native-coordinate quantization diagnostic, but require the optional
# crop-local residual to reconstruct the provider means exactly. The latter prevents a rounded
# finite-support boundary from changing after archival.
NATIVE_MEAN_ROUNDTRIP_MAX_ABS = 3e-4
LOCAL_MEAN_RECOVERY_MAX_ABS = 0.0
RENDER_PARITY_MAX_ABS = 1e-5
RENDER_PARITY_MEAN_ABS = 1e-6
# The v3/v4 failures calibrated the CPU reference equation against the exact archived CUDA
# renderer at 2k capped and 5k uncapped capacity. Full-field reference rendering confirmed the
# sparse 1.13e-4 v4 outlier is CUDA/reference arithmetic at one sampled pixel, not archive drift;
# only 2/24,576 scalars exceeded 1e-4 and mean error was 5.74e-8. A 5e-4 sparse maximum plus a
# 1e-6 mean gate remains 18x below the previously observed 0.009 finite-support semantic failure
# and below 0.128 of one 8-bit code value. This calibration is frozen before the v5 scale-cap pair.
QUERY_PARITY_MAX_ABS = 5e-4
QUERY_PARITY_MEAN_ABS = 1e-6
LOCAL_LIBSTDCXX_PRELOAD = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33")


@dataclass(frozen=True)
class Arm:
    """One count/iteration/initializer cell in the Stage-1 screen."""

    name: str
    initializer: str
    n_init_2d: int
    iterations: int
    max_2d: int | None = None
    adaptive: bool = False
    growth_waves: int = 1
    scale_cap_mode: str = "feature"
    phase: str = "core"

    def __post_init__(self) -> None:
        if self.initializer not in {"aniso_onedge", "quadtree_wse"}:
            raise ValueError(f"unsupported initializer: {self.initializer}")
        if self.scale_cap_mode not in {"feature", "none"}:
            raise ValueError(f"unsupported scale cap mode: {self.scale_cap_mode}")
        if self.n_init_2d < 16 or self.iterations <= 0:
            raise ValueError("arms require at least 16 Gaussians and one iteration")
        maximum = self.n_init_2d if self.max_2d is None else self.max_2d
        if maximum < self.n_init_2d:
            raise ValueError("max_2d cannot be below n_init_2d")
        if self.adaptive != (maximum > self.n_init_2d):
            raise ValueError("adaptive must be true exactly when max_2d exceeds n_init_2d")
        if self.growth_waves <= 0:
            raise ValueError("growth_waves must be positive")

    @property
    def n_max_2d(self) -> int:
        return self.n_init_2d if self.max_2d is None else self.max_2d

    def causal_axes(self) -> dict[str, Any]:
        """Axes used to verify the preregistered one-variable contrasts."""
        return {
            "initializer": self.initializer,
            "component_budget": (self.n_init_2d, self.n_max_2d),
            "iterations": self.iterations,
            "adaptive": self.adaptive,
            "scale_cap_mode": self.scale_cap_mode,
        }


ARMS: tuple[Arm, ...] = (
    Arm("current_aniso_n640_i100", "aniso_onedge", 640, 100),
    # Count control: compare with current at the same 100 iterations.
    Arm("count_aniso_n2000_i100", "aniso_onedge", 2000, 100),
    # Iteration control: compare with the preceding 2,000-component cell.
    Arm("budget_aniso_n2000_i500", "aniso_onedge", 2000, 500),
    # Initializer control: compare with the preceding cell.
    Arm("init_quadtree_n2000_i500", "quadtree_wse", 2000, 500),
    Arm("scale_quadtree_n5000_i1000", "quadtree_wse", 5000, 1000),
    Arm(
        "recipe_quadtree_n5000_i1000_nocap",
        "quadtree_wse",
        5000,
        1000,
        scale_cap_mode="none",
    ),
    Arm(
        "extended_quadtree_n10000_i2000",
        "quadtree_wse",
        10000,
        2000,
        phase="extended",
    ),
    Arm(
        "extended_adaptive_quadtree_n2000_to10000_i2000",
        "quadtree_wse",
        2000,
        2000,
        max_2d=10000,
        adaptive=True,
        growth_waves=8,
        phase="extended",
    ),
)
ARM_BY_NAME = {arm.name: arm for arm in ARMS}
PROFILES = {
    "current": (ARMS[0].name,),
    "core": tuple(arm.name for arm in ARMS if arm.phase == "core"),
    "extended": tuple(arm.name for arm in ARMS),
}
CAUSAL_CONTRASTS = (
    {
        "question": "component_count_at_100_iterations",
        "left": "current_aniso_n640_i100",
        "right": "count_aniso_n2000_i100",
        "changed_axis": "component_budget",
    },
    {
        "question": "optimization_iterations_at_2000_components",
        "left": "count_aniso_n2000_i100",
        "right": "budget_aniso_n2000_i500",
        "changed_axis": "iterations",
    },
    {
        "question": "initializer_at_2000_components_500_iterations",
        "left": "budget_aniso_n2000_i500",
        "right": "init_quadtree_n2000_i500",
        "changed_axis": "initializer",
    },
    {
        "question": "rtgs_feature_scale_cap_at_5000_components_1000_iterations",
        "left": "scale_quadtree_n5000_i1000",
        "right": "recipe_quadtree_n5000_i1000_nocap",
        "changed_axis": "scale_cap_mode",
    },
)


@dataclass
class PreparedView:
    view_id: str
    seed: int
    rgb: torch.Tensor
    foreground: torch.Tensor
    target_crop: torch.Tensor
    mask_crop: torch.Tensor
    fit_window: tuple[int, int, int, int]
    camera: dict[str, Any]
    distortion: list[float]
    input_record: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def command_output(*argv: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(argv, cwd=cwd, text=True).strip()


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(json_safe(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return json_safe(value.detach().cpu().item())
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("refusing to serialize non-finite evidence")
        return value
    return value


def resolve_views(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_VIEWS
    normalized = tuple(str(value).upper() for value in values)
    if normalized == ("ALL",):
        return TRAIN_VIEWS
    if "ALL" in normalized:
        raise ValueError("'all' cannot be combined with explicit views")
    if HELDOUT_VIEW in normalized:
        raise ValueError(f"{HELDOUT_VIEW} is held out and cannot be fit or selected")
    unknown = sorted(set(normalized) - set(TRAIN_VIEWS))
    if unknown:
        raise ValueError(f"views are outside the frozen training set: {unknown}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate views are not allowed")
    return normalized


def resolve_arms(profile: str, names: Sequence[str] | None) -> tuple[Arm, ...]:
    if names:
        unknown = sorted(set(names) - set(ARM_BY_NAME))
        if unknown:
            raise ValueError(f"unknown arms: {unknown}")
        if len(set(names)) != len(names):
            raise ValueError("duplicate arms are not allowed")
        return tuple(ARM_BY_NAME[name] for name in names)
    try:
        return tuple(ARM_BY_NAME[name] for name in PROFILES[profile])
    except KeyError as error:
        raise ValueError(f"unknown profile {profile!r}") from error


def validate_causal_contrasts() -> None:
    for contrast in CAUSAL_CONTRASTS:
        left = ARM_BY_NAME[contrast["left"]].causal_axes()
        right = ARM_BY_NAME[contrast["right"]].causal_axes()
        changed = {name for name in left if left[name] != right[name]}
        if changed != {contrast["changed_axis"]}:
            raise RuntimeError(
                f"contrast {contrast['question']} changes {sorted(changed)}, "
                f"expected only {contrast['changed_axis']}"
            )


def view_paths(view_id: str) -> tuple[Path, Path]:
    if view_id not in TRAIN_VIEWS:
        raise ValueError(f"{view_id} is not in the frozen training set")
    return SCENE / f"rgb/{view_id}.jpg", SCENE / f"mask/mask_{view_id}.png"


def calibration_records() -> dict[str, dict[str, Any]]:
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    records: dict[str, dict[str, Any]] = {}
    for record in payload["cameras"]:
        view_id = str(record["camera_id"]).upper()
        if view_id in records:
            raise RuntimeError(f"duplicate calibration record {view_id}")
        records[view_id] = record
    return records


def prepare_view(view_id: str) -> PreparedView:
    """Load and undistort exactly at calibrated native resolution, then make the mask crop."""
    if view_id not in TRAIN_VIEWS:
        raise ValueError(f"{view_id} is not in the frozen training set")
    record = calibration_records()[view_id]
    intrinsics = record["intrinsics"]
    matrix = intrinsics["camera_matrix"]
    width, height = (int(value) for value in intrinsics["resolution"])
    fx, fy = float(matrix[0]), float(matrix[4])
    cx, cy = float(matrix[2]) + 0.5, float(matrix[5]) + 0.5
    distortion = [float(value) for value in intrinsics.get("distortion_coefficients", [])]
    rgb_path, mask_path = view_paths(view_id)
    with Image.open(rgb_path) as source:
        if source.size != (width, height):
            raise RuntimeError(
                f"{view_id} image is {source.size}, calibration expects {(width, height)}"
            )
    rgb = _undistort(
        _resize_image(rgb_path, width, height),
        fx,
        fy,
        cx,
        cy,
        distortion,
    )
    foreground = (
        _undistort(
            _resize_image(mask_path, width, height, mask=True),
            fx,
            fy,
            cx,
            cy,
            distortion,
            mask=True,
        )
        > 0.5
    )
    rgb_crop, mask_crop, offset = _crop_to_mask(rgb, foreground)
    target_crop = (rgb_crop * mask_crop[..., None]).contiguous()
    fit_window = (
        int(offset[0]),
        int(offset[1]),
        int(target_crop.shape[1]),
        int(target_crop.shape[0]),
    )
    view = torch.tensor(record["extrinsics"]["view_matrix"], dtype=torch.float32).reshape(4, 4)
    camera = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
        "R": view[:3, :3].reshape(-1).tolist(),
        "t": view[:3, 3].tolist(),
    }
    input_record = {
        "view_id": view_id,
        "seed": VIEW_SEEDS[view_id],
        "rgb": {"path": rgb_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(rgb_path)},
        "mask": {
            "path": mask_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(mask_path),
        },
        "camera": camera,
        "distortion": distortion,
        "native_resolution": [width, height],
        "fit_window": list(fit_window),
        "full_canvas_pixels": width * height,
        "fit_crop_pixels": fit_window[2] * fit_window[3],
        "foreground_pixels_full": int(foreground.sum()),
        "foreground_pixels_crop": int(mask_crop.sum()),
        "undistorted_rgb_tensor_sha256": tensor_hash(rgb),
        "undistorted_mask_tensor_sha256": tensor_hash(foreground),
        "masked_target_crop_tensor_sha256": tensor_hash(target_crop),
    }
    return PreparedView(
        view_id=view_id,
        seed=VIEW_SEEDS[view_id],
        rgb=rgb,
        foreground=foreground,
        target_crop=target_crop,
        mask_crop=mask_crop,
        fit_window=fit_window,
        camera=camera,
        distortion=distortion,
        input_record=input_record,
    )


def _source_file_manifest(root: Path) -> tuple[dict[str, str], str]:
    suffixes = {".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in suffixes)
    manifest = {path.relative_to(root).as_posix(): sha256_file(path) for path in files}
    return manifest, canonical_hash(manifest)


def structsplat_source_binding() -> dict[str, Any]:
    """Bind the optional provider lazily, including dirty source and loaded extension."""
    import structsplat

    module_root = Path(structsplat.__file__).resolve().parent
    manifest, aggregate = _source_file_manifest(module_root)
    status = command_output("git", "status", "--porcelain=v1", cwd=STRUCTSPLAT_ROOT)
    tracked_diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--"], cwd=STRUCTSPLAT_ROOT
    )
    return {
        "version": importlib.metadata.version("structsplat"),
        "module_root": str(module_root),
        "repository_root": str(STRUCTSPLAT_ROOT),
        "git_revision": command_output("git", "rev-parse", "HEAD", cwd=STRUCTSPLAT_ROOT),
        "git_status_lines": status.splitlines(),
        "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "provider_source_digest": _provider_digest(module_root),
        "provider_source_manifest_sha256": aggregate,
        "provider_source_file_count": len(manifest),
    }


def _provider_digest(module_root: Path) -> str:
    """Match ``rtgs.image2gs.structsplat_backend._structsplat_provenance`` exactly."""
    suffixes = {".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}
    digest = hashlib.sha256()
    for path in sorted(
        candidate
        for candidate in module_root.rglob("*")
        if candidate.is_file() and candidate.suffix in suffixes
    ):
        digest.update(path.relative_to(module_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def repository_source_binding() -> dict[str, Any]:
    paths = (
        Path("benchmarks/stage1_capacity_sweep.py"),
        Path("src/rtgs/core/metrics.py"),
        Path("src/rtgs/core/observation2d.py"),
        Path("src/rtgs/data/calibrated.py"),
        Path("src/rtgs/image2gs/fit.py"),
        Path("src/rtgs/image2gs/structsplat_backend.py"),
    )
    manifest = {path.as_posix(): sha256_file(ROOT / path) for path in paths}
    return {
        "git_revision": command_output("git", "rev-parse", "HEAD"),
        "files": manifest,
        "aggregate_sha256": canonical_hash(manifest),
    }


def effective_configs(
    arm: Arm,
    *,
    crop_height: int,
    crop_width: int,
    seed: int,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Construct the exact external configs; importing StructSplat stays execution-local."""
    from structsplat.config import FitConfig, InitConfig, StructureTensorConfig

    add_total = arm.n_max_2d - arm.n_init_2d
    growth_count = max(1, math.ceil(add_total / arm.growth_waves)) if add_total else 0
    growth_every = max(1, arm.iterations // (arm.growth_waves + 1)) if add_total else None
    feature_cap = 12.0 * max(crop_height, crop_width) / 160.0
    init_cfg = InitConfig(
        strategy=arm.initializer,
        num_gaussians=arm.n_init_2d,
        seed=seed,
        sampling_mode="wse",
        flank_offset_frac=0.0,
        scale_cap_mode=arm.scale_cap_mode,
        scale_cap_max=feature_cap if arm.scale_cap_mode == "feature" else None,
    )
    tensor_cfg = StructureTensorConfig()
    fit_cfg = FitConfig(
        iters=arm.iterations,
        renderer=RENDERER,
        pixel_loss="l1",
        ssim_weight=0.3,
        log_every=20,
        split_every=(None if arm.adaptive else growth_every),
        split_count=(0 if arm.adaptive else growth_count),
        split_mode="residual_tensor_add",
        max_gaussians=arm.n_max_2d,
        adaptive_count=arm.adaptive and add_total > 0,
        adaptive_growth_every=(growth_every or max(1, arm.iterations)),
        adaptive_growth_count=max(1, growth_count),
        adaptive_split_mode="residual_tensor_add",
        adaptive_min_delta_psnr=0.05,
        adaptive_patience=2,
        early_stop_patience=None,
        early_stop_min_delta=0.05,
        early_stop_min_iters=max(25, arm.iterations // 3),
        relocate_every=None,
        relocate_at_split=False,
        relocate_count=0,
    )
    config_dict = {
        "arm": dataclasses.asdict(arm),
        "init": dataclasses.asdict(init_cfg),
        "structure_tensor": dataclasses.asdict(tensor_cfg),
        "fit": dataclasses.asdict(fit_cfg),
    }
    config_dict["digests"] = {
        "init_sha256": canonical_hash(config_dict["init"]),
        "structure_tensor_sha256": canonical_hash(config_dict["structure_tensor"]),
        "fit_sha256": canonical_hash(config_dict["fit"]),
        "combined_sha256": canonical_hash(config_dict),
    }
    return init_cfg, tensor_cfg, fit_cfg, config_dict


def _gaussian_kernel(
    window: int, sigma: float, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    coords = torch.arange(window, device=device, dtype=dtype) - (window - 1) / 2
    kernel = torch.exp(-(coords.square()) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def _separable_filter(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    channels = image.shape[1]
    radius = kernel.numel() // 2
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    value = F.conv2d(image, vertical, padding=(radius, 0), groups=channels)
    return F.conv2d(value, horizontal, padding=(0, radius), groups=channels)


@torch.no_grad()
def weighted_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    tile_rows: int = 256,
    window: int = SSIM_WINDOW,
    sigma: float = SSIM_SIGMA,
) -> float:
    """SSIM-map mean weighted at output-pixel centers, evaluated in row tiles.

    Inputs are HWC.  The same zero-padding convention as the repository and StructSplat metrics
    is retained.  Halo rows make the result invariant to ``tile_rows`` up to floating-point
    reduction order.
    """
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("prediction and target must have matching HxWx3 shapes")
    if weights.shape != prediction.shape[:2]:
        raise ValueError("weights must match the image height and width")
    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if tile_rows <= 0:
        raise ValueError("tile_rows must be positive")
    weights = weights.to(device=prediction.device, dtype=prediction.dtype).clamp(0, 1)
    weight_sum = weights.sum()
    if not bool(weight_sum > 0):
        raise ValueError("weights contain no positive pixels")
    kernel = _gaussian_kernel(window, sigma, device=prediction.device, dtype=prediction.dtype)
    radius = window // 2
    height = prediction.shape[0]
    numerator = prediction.new_zeros(())
    for y0 in range(0, height, tile_rows):
        y1 = min(height, y0 + tile_rows)
        source_y0 = max(0, y0 - radius)
        source_y1 = min(height, y1 + radius)
        pred = prediction[source_y0:source_y1].permute(2, 0, 1).unsqueeze(0)
        truth = target[source_y0:source_y1].permute(2, 0, 1).unsqueeze(0)
        mu_pred = _separable_filter(pred, kernel)
        mu_truth = _separable_filter(truth, kernel)
        mu_pred2 = mu_pred.square()
        mu_truth2 = mu_truth.square()
        mu_cross = mu_pred * mu_truth
        var_pred = _separable_filter(pred.square(), kernel) - mu_pred2
        var_truth = _separable_filter(truth.square(), kernel) - mu_truth2
        covariance = _separable_filter(pred * truth, kernel) - mu_cross
        c1, c2 = 0.01**2, 0.03**2
        similarity = ((2 * mu_cross + c1) * (2 * covariance + c2)) / (
            (mu_pred2 + mu_truth2 + c1) * (var_pred + var_truth + c2)
        )
        local_y0 = y0 - source_y0
        local_y1 = local_y0 + (y1 - y0)
        center = similarity[0, :, local_y0:local_y1]
        numerator += (center * weights[y0:y1].unsqueeze(0)).sum()
    return float(numerator / (weight_sum * prediction.shape[-1]))


def psnr_from_mse(mse: float) -> float:
    return -10.0 * math.log10(max(float(mse), 1e-30))


def weighted_error_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, Any]:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("prediction and target must have matching HxWx3 shapes")
    if weights.shape != prediction.shape[:2]:
        raise ValueError("weights must match prediction")
    weights = weights.to(prediction).clamp(0, 1)
    pixel_weight = float(weights.sum())
    if pixel_weight <= 0:
        raise ValueError("weights contain no positive pixels")
    squared = (prediction - target).square()
    mse = float((squared * weights[..., None]).sum() / (weights.sum() * 3))
    clamped_mse = float(
        ((prediction.clamp(0, 1) - target).square() * weights[..., None]).sum()
        / (weights.sum() * 3)
    )
    return {
        "weighted_pixel_count": pixel_weight,
        "weighted_scalar_count": pixel_weight * 3,
        "raw": {"mse": mse, "psnr_db": psnr_from_mse(mse)},
        "clamped": {
            "mse": clamped_mse,
            "psnr_db": psnr_from_mse(clamped_mse),
        },
    }


def validate_fit_outcome(
    arm: Arm,
    *,
    m_opt_2d: int,
    iterations_run: int,
) -> None:
    """Fail closed when a provider result violates the preregistered arm contract."""
    if arm.adaptive:
        if not arm.n_init_2d <= m_opt_2d <= arm.n_max_2d:
            raise RuntimeError(
                f"{arm.name} returned m_opt_2d={m_opt_2d}, outside "
                f"[{arm.n_init_2d}, {arm.n_max_2d}]"
            )
        if not 1 <= iterations_run <= arm.iterations:
            raise RuntimeError(
                f"{arm.name} ran {iterations_run} iterations, expected 1..{arm.iterations}"
            )
        return
    if m_opt_2d != arm.n_init_2d:
        raise RuntimeError(
            f"fixed-count arm {arm.name} returned {m_opt_2d} Gaussians, expected {arm.n_init_2d}"
        )
    if iterations_run != arm.iterations:
        raise RuntimeError(
            f"fixed-budget arm {arm.name} ran {iterations_run} iterations, "
            f"expected {arm.iterations}"
        )


def validate_semantic_parity(
    render_parity: torch.Tensor,
    query_parity: torch.Tensor,
    *,
    render_max_abs: float = RENDER_PARITY_MAX_ABS,
    render_mean_abs: float = RENDER_PARITY_MEAN_ABS,
    query_max_abs: float = QUERY_PARITY_MAX_ABS,
    query_mean_abs: float = QUERY_PARITY_MEAN_ABS,
) -> None:
    """Reject non-finite or materially lossy provider/archive/query conversions."""
    for name, value, threshold in (
        ("live_vs_archive_render", render_parity, render_max_abs),
        ("archive_query_vs_render", query_parity, query_max_abs),
    ):
        if value.numel() == 0 or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"{name} parity contains no finite comparison")
        maximum = float(value.max())
        if maximum > threshold:
            raise RuntimeError(
                f"{name} max_abs={maximum:.9g} exceeds preregistered threshold {threshold:.9g}"
            )
        mean_threshold = render_mean_abs if name == "live_vs_archive_render" else query_mean_abs
        mean = float(value.mean())
        if mean > mean_threshold:
            raise RuntimeError(
                f"{name} mean_abs={mean:.9g} exceeds preregistered threshold {mean_threshold:.9g}"
            )


def query_parity_samples(
    field: GaussianObservationField,
    *,
    sample_count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    """Mix uniform crop pixels with component support pixels for query/render parity."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    fit_x, fit_y, width, height = field.fit_window
    support_pixels = field.support_pixels()
    support_centers = field.support_centers()
    support_local = support_pixels - support_pixels.new_tensor([fit_x, fit_y])
    support_valid = (
        (support_local[:, 0] >= 0)
        & (support_local[:, 0] < width)
        & (support_local[:, 1] >= 0)
        & (support_local[:, 1] < height)
    )
    support_indices = torch.nonzero(support_valid, as_tuple=False).flatten()
    support_budget = min(sample_count // 2, int(support_indices.numel()))
    if support_indices.numel() > support_budget:
        order = torch.randperm(support_indices.numel(), generator=generator)[:support_budget]
        support_indices = support_indices[order]
    else:
        support_indices = support_indices[:support_budget]
    selected_local = support_local[support_indices]
    selected_xy = support_centers[support_indices]

    uniform_count = sample_count - support_budget
    uniform_x = torch.randint(width, (uniform_count,), generator=generator)
    uniform_y = torch.randint(height, (uniform_count,), generator=generator)
    uniform_xy = (
        torch.stack([uniform_x + fit_x, uniform_y + fit_y], dim=-1)
        .to(dtype=field.dtype, device=field.device)
        .add(0.5)
    )
    native_xy = torch.cat([selected_xy, uniform_xy], dim=0)
    sample_x = torch.cat([selected_local[:, 0], uniform_x], dim=0)
    sample_y = torch.cat([selected_local[:, 1], uniform_y], dim=0)
    return (
        native_xy,
        sample_x,
        sample_y,
        {
            "component_support_samples": support_budget,
            "uniform_crop_samples": uniform_count,
        },
    )


def render_observation(field: GaussianObservationField, *, device: torch.device) -> torch.Tensor:
    """Recreate a live provider field and render the exact archived native semantics."""
    from structsplat.gaussians import GaussianField
    from structsplat.render import render_field

    _, _, width, height = field.fit_window
    amplitudes = field.amplitudes.to(device)
    if torch.equal(amplitudes, torch.ones_like(amplitudes)):
        opacity_logits = None
    else:
        opacity_logits = torch.logit(amplitudes.clamp(1e-6, 1.0 - 1e-6))
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
    rendered = render_field(
        live.means,
        live.conics(field.aa_dilation),
        live.colors,
        live.radii(field.sigma_cutoff, field.aa_dilation),
        height,
        width,
        mode=RENDERER,
        opacities=live.opacity_values(),
        scales=live.effective_scales(field.aa_dilation),
        rotations=live.rotations,
        support_fade=field.support_fade_alpha > 0.0,
        support_fade_alpha=field.support_fade_alpha,
        sigma_cutoff=field.sigma_cutoff,
        color_grads=live.color_grads,
    )
    return rendered.detach().cpu()


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
        "m_init_2d": field.n_init,
        "m_opt_2d": field.n,
        "canvas_size": [field.width, field.height],
        "fit_window": list(field.fit_window),
        "view_id": field.view_id,
        "provider": field.provider,
        "producer_version": field.producer_version,
        "producer_source_digest": field.producer_source_digest,
        "fit_config_digest": field.fit_config_digest,
        "blend_mode": field.blend_mode,
        "epsilon": field.epsilon,
        "sigma_cutoff": field.sigma_cutoff,
        "support_fade_alpha": field.support_fade_alpha,
        "aa_dilation": field.aa_dilation,
        "tensor_hashes": {name: tensor_hash(value) for name, value in tensors.items()},
    }


def resize_hwc(value: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    return F.interpolate(
        value.permute(2, 0, 1).unsqueeze(0),
        size=(height, width),
        mode="area",
    )[0].permute(1, 2, 0)


def save_panel(
    path: Path,
    *,
    arm_name: str,
    target: torch.Tensor,
    prediction: torch.Tensor,
    foreground_psnr: float,
    foreground_ssim: float,
) -> None:
    height, width = 420, 1066
    target_small = resize_hwc(target, height=height, width=width)
    prediction_small = resize_hwc(prediction.clamp(0, 1), height=height, width=width)
    error = (prediction.clamp(0, 1) - target).abs().mean(dim=-1, keepdim=True)
    scale = torch.quantile(error.reshape(-1), 0.99).clamp_min(1e-6)
    normalized = (error / scale).clamp(0, 1)
    error_rgb = torch.cat([normalized, 0.2 * normalized, torch.zeros_like(normalized)], dim=-1)
    error_small = resize_hwc(error_rgb, height=height, width=width)
    cells = [target_small, prediction_small, error_small]
    labels = [
        "masked native target",
        f"{arm_name} | fg PSNR {foreground_psnr:.3f} dB | fg-wSSIM {foreground_ssim:.5f}",
        "absolute RGB error (red, normalized at p99)",
    ]
    label_height = 34
    canvas = Image.new("RGB", (width * 3, height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, value) in enumerate(zip(labels, cells, strict=True)):
        image = Image.fromarray((value.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy())
        x = index * width
        draw.text((x + 8, 9), label, fill="black")
        canvas.paste(image, (x, label_height))
    canvas.save(path)


def extension_binding() -> dict[str, str]:
    import structsplat.cuda_render as cuda_render

    try:
        extension = cuda_render._load_extension()
    except RuntimeError as error:
        preload_hint = (
            f" Prefix the command with LD_PRELOAD={LOCAL_LIBSTDCXX_PRELOAD}."
            if LOCAL_LIBSTDCXX_PRELOAD.is_file()
            else ""
        )
        raise RuntimeError(
            "StructSplat CUDA extension preflight failed before native fitting."
            f"{preload_hint} Provider error: {error}"
        ) from error
    path = Path(extension.__file__).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def environment_binding(device: torch.device | None = None) -> dict[str, Any]:
    result = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "ld_preload": os.environ.get("LD_PRELOAD"),
    }
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(device)
        result["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
    return result


def prepare_plan(
    out: Path,
    *,
    views: tuple[str, ...],
    arms: tuple[Arm, ...],
) -> dict[str, Any]:
    validate_causal_contrasts()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {out}")
    if HELDOUT_VIEW in views or any(view not in TRAIN_VIEWS for view in views):
        raise RuntimeError("plan contains a non-training or held-out view")
    if not CALIBRATION.is_file():
        raise FileNotFoundError(CALIBRATION)
    out.mkdir(parents=True)
    (out / "records").mkdir()
    (out / "teachers").mkdir()
    (out / "panels").mkdir()
    prepared: dict[str, PreparedView] = {}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("native-resolution StructSplat planning requires CUDA")
        plan_device = torch.device("cuda")
        loaded_extension = extension_binding()
        plan_environment = environment_binding(plan_device)
        for view_id in views:
            prepared[view_id] = prepare_view(view_id)
        external = structsplat_source_binding()
        arm_configs: dict[str, dict[str, Any]] = {}
        for view_id, view in prepared.items():
            arm_configs[view_id] = {}
            for arm in arms:
                _, _, _, config = effective_configs(
                    arm,
                    crop_height=view.fit_window[3],
                    crop_width=view.fit_window[2],
                    seed=view.seed,
                )
                arm_configs[view_id][arm.name] = config
        plan = {
            "artifact_type": "stage1_capacity_sweep_plan_v1",
            "decision_bearing": False,
            "scope": (
                "native-resolution masked Stage-1 representation fidelity only; no 3D lift, "
                "novel-view, or default claim"
            ),
            "training_view_universe": list(TRAIN_VIEWS),
            "selected_views": list(views),
            "heldout_view_excluded": HELDOUT_VIEW,
            "heldout_policy": (
                "C1004 is not accepted by view resolution and is never loaded, fit, ranked, "
                "or selected by this harness"
            ),
            "source_rgb_boundary": (
                "RGB/masks are used only for Stage-1 preprocessing, fitting, and isolated "
                "same-view evaluation; lossless teacher archives contain no RGB tensor."
            ),
            "arms": [dataclasses.asdict(arm) for arm in arms],
            "causal_contrasts": list(CAUSAL_CONTRASTS),
            "headline_metrics": {
                "foreground_psnr": (
                    "MSE weighted by binary foreground pixels and RGB channels; both raw and "
                    "display-clamped values are retained"
                ),
                "foreground_weighted_ssim": (
                    "11x11 sigma=1.5 zero-padded SSIM map on clamped prediction versus masked "
                    "target, averaged over channels and weighted at foreground output centers"
                ),
            },
            "semantic_parity_gates": {
                "crop_local_to_native_mean_roundtrip_max_abs_px": (NATIVE_MEAN_ROUNDTRIP_MAX_ABS),
                "residual_corrected_local_mean_recovery_max_abs_px": (LOCAL_MEAN_RECOVERY_MAX_ABS),
                "live_vs_archive_render_max_abs": RENDER_PARITY_MAX_ABS,
                "live_vs_archive_render_mean_abs": RENDER_PARITY_MEAN_ABS,
                "archive_query_vs_render_max_abs": QUERY_PARITY_MAX_ABS,
                "archive_query_vs_render_mean_abs": QUERY_PARITY_MEAN_ABS,
                "query_sampling": (
                    "up to half component support pixels, remainder uniform crop pixels"
                ),
                "precision_scope": (
                    "provider tensors remain float32; adding the native fit-window offset can "
                    "quantize schema-v1 native means by a fraction of one native pixel ULP, "
                    "while an integrity-covered crop-local residual preserves exact semantics"
                ),
            },
            "inputs": {
                "scene": SCENE.relative_to(ROOT).as_posix(),
                "calibration": {
                    "path": CALIBRATION.relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(CALIBRATION),
                },
                "views": {view_id: view.input_record for view_id, view in prepared.items()},
            },
            "effective_configs": arm_configs,
            "repository": repository_source_binding(),
            "external_structsplat": external,
            "environment_at_plan": plan_environment,
            "loaded_cuda_extension": loaded_extension,
        }
        write_json_exclusive(out / "plan.json", plan)
        return plan
    except BaseException:
        failure = {
            "artifact_type": "stage1_capacity_sweep_plan_failure_v1",
            "status": "FAIL",
            "traceback": traceback.format_exc(),
        }
        write_json_exclusive(out / "plan_failure.json", failure)
        raise


def _validate_plan_runtime(
    plan: dict[str, Any],
    *,
    view_id: str,
    arm: Arm,
    runtime_environment: Mapping[str, Any],
    loaded_extension: Mapping[str, str],
) -> PreparedView:
    if plan.get("artifact_type") != "stage1_capacity_sweep_plan_v1":
        raise RuntimeError("wrong plan artifact type")
    if plan["heldout_view_excluded"] != HELDOUT_VIEW:
        raise RuntimeError("held-out policy changed")
    if view_id == HELDOUT_VIEW or view_id not in plan["selected_views"]:
        raise RuntimeError(f"{view_id} is not a planned training view")
    planned_arms = {record["name"]: record for record in plan["arms"]}
    if planned_arms.get(arm.name) != dataclasses.asdict(arm):
        raise RuntimeError(f"{arm.name} differs from the planned arm")
    if repository_source_binding() != plan["repository"]:
        raise RuntimeError("realtime-gs experiment sources drifted after planning")
    if structsplat_source_binding() != plan["external_structsplat"]:
        raise RuntimeError("StructSplat sources drifted after planning")
    if dict(runtime_environment) != plan["environment_at_plan"]:
        raise RuntimeError("Python/CUDA environment differs from the frozen plan")
    if dict(loaded_extension) != plan["loaded_cuda_extension"]:
        raise RuntimeError("loaded StructSplat CUDA extension differs from the frozen plan")
    if sha256_file(CALIBRATION) != plan["inputs"]["calibration"]["sha256"]:
        raise RuntimeError("calibration changed after planning")
    prepared = prepare_view(view_id)
    if prepared.input_record != plan["inputs"]["views"][view_id]:
        raise RuntimeError(f"{view_id} calibrated input/preprocessing changed after planning")
    _, _, _, config = effective_configs(
        arm,
        crop_height=prepared.fit_window[3],
        crop_width=prepared.fit_window[2],
        seed=prepared.seed,
    )
    if config != plan["effective_configs"][view_id][arm.name]:
        raise RuntimeError(f"{view_id}/{arm.name} effective config changed after planning")
    return prepared


def run_cell(out: Path, *, view_id: str, arm_name: str, device_text: str) -> dict[str, Any]:
    if arm_name not in ARM_BY_NAME:
        raise ValueError(f"unknown arm {arm_name!r}")
    arm = ARM_BY_NAME[arm_name]
    plan_path = out / "plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    plan_sha256 = sha256_file(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    record_path = out / "records" / f"{view_id}__{arm_name}.json"
    teacher_path = out / "teachers" / f"{view_id}__{arm_name}.teacher.npz"
    panel_path = out / "panels" / f"{view_id}__{arm_name}.png"
    if any(path.exists() for path in (record_path, teacher_path, panel_path)):
        raise FileExistsError(
            f"refusing to overwrite completed or partial cell {view_id}/{arm_name}"
        )
    device = torch.device(device_text)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("native-resolution StructSplat fitting requires CUDA")
    try:
        loaded_extension = extension_binding()
        runtime_environment = environment_binding(device)
        prepared = _validate_plan_runtime(
            plan,
            view_id=view_id,
            arm=arm,
            runtime_environment=runtime_environment,
            loaded_extension=loaded_extension,
        )
        init_cfg, tensor_cfg, fit_cfg, config = effective_configs(
            arm,
            crop_height=prepared.fit_window[3],
            crop_width=prepared.fit_window[2],
            seed=prepared.seed,
        )
        from structsplat.fit import fit
        from structsplat.init import build_field

        from rtgs.image2gs.structsplat_backend import field_to_observation

        target = prepared.target_crop.to(device).contiguous()
        mask = prepared.mask_crop.to(device)
        image_np = prepared.target_crop.numpy().astype(np.float32, copy=False)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        wall_started = time.perf_counter()
        init_started = time.perf_counter()
        field = build_field(image_np, init_cfg, tensor_cfg, device=str(device))
        torch.cuda.synchronize(device)
        init_seconds = time.perf_counter() - init_started
        result = fit(field, target, fit_cfg, verbose=False)
        torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - wall_started
        fit_peak_allocated = int(torch.cuda.max_memory_allocated(device))
        fit_peak_reserved = int(torch.cuda.max_memory_reserved(device))

        producer_version = plan["external_structsplat"]["version"]
        producer_digest = plan["external_structsplat"]["provider_source_digest"]
        fit_digest = config["digests"]["fit_sha256"]
        observation = field_to_observation(
            result["field"],
            canvas_size=(prepared.camera["height"], prepared.camera["width"]),
            fit_window=prepared.fit_window,
            blend_mode="normalized",
            sigma_cutoff=float(fit_cfg.sigma_cutoff),
            support_fade_alpha=1.0 if fit_cfg.support_fade else 0.0,
            aa_dilation=float(fit_cfg.aa_dilation),
            view_id=view_id,
            n_init=arm.n_init_2d,
            producer_version=producer_version,
            producer_source_digest=producer_digest,
            fit_config_digest=fit_digest,
        ).to("cpu")
        fit_x, fit_y, _, _ = prepared.fit_window
        native_offset = observation.means.new_tensor([fit_x + 0.5, fit_y + 0.5])
        raw_mean_roundtrip = (
            observation.means - native_offset - result["field"].means.detach().to("cpu")
        ).abs()
        if (
            not bool(torch.isfinite(raw_mean_roundtrip).all())
            or float(raw_mean_roundtrip.max()) > NATIVE_MEAN_ROUNDTRIP_MAX_ABS
        ):
            raise RuntimeError(
                "raw crop-local/native mean roundtrip exceeds the preregistered float32 "
                f"precision budget: max_abs={float(raw_mean_roundtrip.max()):.9g}"
            )
        corrected_mean_recovery = (
            observation.local_means() - result["field"].means.detach().to("cpu")
        ).abs()
        if (
            not bool(torch.isfinite(corrected_mean_recovery).all())
            or float(corrected_mean_recovery.max()) > LOCAL_MEAN_RECOVERY_MAX_ABS
        ):
            raise RuntimeError(
                "residual-corrected crop-local means do not exactly recover the provider field: "
                f"max_abs={float(corrected_mean_recovery.max()):.9g}"
            )
        observation.save_npz(teacher_path)
        teacher_sha256 = sha256_file(teacher_path)
        reloaded = GaussianObservationField.load_npz(teacher_path, strict=True)
        if observation_summary(reloaded) != observation_summary(observation):
            raise RuntimeError("strict archive reload changed teacher semantics")
        iterations_run = int(result["iterations_run"])
        validate_fit_outcome(
            arm,
            m_opt_2d=reloaded.n,
            iterations_run=iterations_run,
        )

        live_render = result["render"].detach().cpu()
        archive_render = render_observation(reloaded, device=device)
        render_parity = (live_render - archive_render).abs()
        generator = torch.Generator(device="cpu").manual_seed(
            17072026 + 1009 * prepared.seed + list(ARM_BY_NAME).index(arm_name)
        )
        height, width = archive_render.shape[:2]
        native_xy, sample_x, sample_y, query_sampling = query_parity_samples(
            reloaded,
            sample_count=QUERY_SAMPLE_COUNT,
            generator=generator,
        )
        queried = reloaded.query(native_xy, component_chunk=256).color
        query_parity = (queried - archive_render[sample_y, sample_x]).abs()
        validate_semantic_parity(render_parity, query_parity)

        foreground = weighted_error_metrics(
            archive_render, prepared.target_crop, prepared.mask_crop
        )
        fit_crop = weighted_error_metrics(
            archive_render,
            prepared.target_crop,
            torch.ones_like(prepared.mask_crop, dtype=torch.float32),
        )
        foreground_ssim = weighted_ssim(
            archive_render.clamp(0, 1).to(device),
            target,
            mask,
        )
        fit_crop_ssim = weighted_ssim(
            archive_render.clamp(0, 1).to(device),
            target,
            torch.ones_like(mask, dtype=torch.float32),
        )
        foreground["clamped"]["weighted_ssim"] = foreground_ssim
        fit_crop["clamped"]["weighted_ssim"] = fit_crop_ssim

        save_panel(
            panel_path,
            arm_name=arm_name,
            target=prepared.target_crop,
            prediction=archive_render,
            foreground_psnr=foreground["clamped"]["psnr_db"],
            foreground_ssim=foreground_ssim,
        )
        history = result["history"]
        record = {
            "artifact_type": "stage1_capacity_sweep_cell_v1",
            "status": "PASS",
            "decision_bearing": False,
            "plan_sha256": plan_sha256,
            "view_id": view_id,
            "seed": prepared.seed,
            "arm": dataclasses.asdict(arm),
            "effective_config": config,
            "input": prepared.input_record,
            "teacher": observation_summary(reloaded),
            "teacher_path": teacher_path.relative_to(out).as_posix(),
            "teacher_sha256": teacher_sha256,
            "metrics": {
                "foreground_rgb": foreground,
                "masked_fit_crop": fit_crop,
                "pixel_counts": {
                    "full_canvas": prepared.camera["width"] * prepared.camera["height"],
                    "fit_crop": width * height,
                    "foreground": int(prepared.mask_crop.sum()),
                    "background_in_fit_crop": int((~prepared.mask_crop).sum()),
                },
                "native_live_vs_archive_render_parity": {
                    "compared_scalar_count": render_parity.numel(),
                    "required_max_abs": RENDER_PARITY_MAX_ABS,
                    "required_mean_abs": RENDER_PARITY_MEAN_ABS,
                    "max_abs_error": float(render_parity.max()),
                    "mean_abs_error": float(render_parity.mean()),
                    "live_render_sha256": tensor_hash(live_render),
                    "archive_render_sha256": tensor_hash(archive_render),
                },
                "raw_crop_local_to_native_mean_roundtrip": {
                    "compared_scalar_count": raw_mean_roundtrip.numel(),
                    "required_max_abs_px": NATIVE_MEAN_ROUNDTRIP_MAX_ABS,
                    "max_abs_px": float(raw_mean_roundtrip.max()),
                    "mean_abs_px": float(raw_mean_roundtrip.mean()),
                },
                "residual_corrected_local_mean_recovery": {
                    "compared_scalar_count": corrected_mean_recovery.numel(),
                    "required_max_abs_px": LOCAL_MEAN_RECOVERY_MAX_ABS,
                    "max_abs_px": float(corrected_mean_recovery.max()),
                    "mean_abs_px": float(corrected_mean_recovery.mean()),
                },
                "native_archive_query_vs_render_parity": {
                    "sample_count": QUERY_SAMPLE_COUNT,
                    "sampling": query_sampling,
                    "compared_scalar_count": query_parity.numel(),
                    "required_max_abs": QUERY_PARITY_MAX_ABS,
                    "required_mean_abs": QUERY_PARITY_MEAN_ABS,
                    "max_abs_error": float(query_parity.max()),
                    "mean_abs_error": float(query_parity.mean()),
                },
            },
            "optimization": {
                "m_init_2d": arm.n_init_2d,
                "m_opt_2d": reloaded.n,
                "iterations_requested": arm.iterations,
                "iterations_run": iterations_run,
                "adaptive_stop_reason": result.get("adaptive_stop_reason"),
                "final_provider_psnr_raw_masked_crop": float(result["psnr"]),
                "final_provider_ssim_raw_masked_crop": float(result["ssim"]),
                "final_provider_ms_ssim_raw_masked_crop": float(result["ms_ssim"]),
                "history": history,
            },
            "runtime": {
                "initializer_seconds": init_seconds,
                "fit_seconds": float(result["fit_seconds"]),
                "wall_seconds_through_fit": wall_seconds,
                "fit_peak_cuda_allocated_bytes": fit_peak_allocated,
                "fit_peak_cuda_reserved_bytes": fit_peak_reserved,
                "cell_peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cell_peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            },
            "panel": {
                "path": panel_path.relative_to(out).as_posix(),
                "sha256": sha256_file(panel_path),
            },
            "environment": runtime_environment,
            "external_structsplat": structsplat_source_binding(),
            "loaded_cuda_extension": loaded_extension,
        }
        if record["external_structsplat"] != plan["external_structsplat"]:
            raise RuntimeError("StructSplat source changed during the cell")
        if repository_source_binding() != plan["repository"]:
            raise RuntimeError("realtime-gs experiment source changed during the cell")
        if extension_binding() != loaded_extension:
            raise RuntimeError("loaded StructSplat CUDA extension changed during the cell")
        if environment_binding(device) != runtime_environment:
            raise RuntimeError("Python/CUDA environment changed during the cell")
        if sha256_file(plan_path) != plan_sha256:
            raise RuntimeError("plan changed during the cell")
        if prepare_view(view_id).input_record != prepared.input_record:
            raise RuntimeError("calibrated input changed during the cell")
        write_json_exclusive(record_path, record)
        return record
    except BaseException as error:
        if not record_path.exists():
            write_json_exclusive(
                record_path,
                {
                    "artifact_type": "stage1_capacity_sweep_cell_failure_v1",
                    "status": "FAIL",
                    "plan_sha256": plan_sha256,
                    "view_id": view_id,
                    "arm_name": arm_name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise


def assemble(out: Path, *, allow_incomplete: bool = False) -> dict[str, Any]:
    plan_path = out / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_sha256 = sha256_file(plan_path)
    expected = [
        (view_id, arm_record["name"])
        for view_id in plan["selected_views"]
        for arm_record in plan["arms"]
    ]
    possible_outputs = [
        out / "result.json",
        out / "partial_result.json",
        *(out / f"{view_id}_capacity_contact_sheet.png" for view_id in plan["selected_views"]),
    ]
    existing_outputs = [path for path in possible_outputs if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            f"refusing to overwrite assembled artifacts: {[str(path) for path in existing_outputs]}"
        )
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for view_id, arm_name in expected:
        path = out / "records" / f"{view_id}__{arm_name}.json"
        if not path.is_file():
            missing.append(f"{view_id}/{arm_name}")
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "PASS":
            missing.append(f"{view_id}/{arm_name} ({record.get('status')})")
            continue
        if record["plan_sha256"] != plan_sha256:
            raise RuntimeError(f"{view_id}/{arm_name} record has a different plan")
        if record["environment"] != plan["environment_at_plan"]:
            raise RuntimeError(f"{view_id}/{arm_name} environment differs from the plan")
        if record["loaded_cuda_extension"] != plan["loaded_cuda_extension"]:
            raise RuntimeError(f"{view_id}/{arm_name} CUDA extension differs from the plan")
        teacher = out / record["teacher_path"]
        panel = out / record["panel"]["path"]
        if sha256_file(teacher) != record["teacher_sha256"]:
            raise RuntimeError(f"{view_id}/{arm_name} teacher changed before assembly")
        if sha256_file(panel) != record["panel"]["sha256"]:
            raise RuntimeError(f"{view_id}/{arm_name} panel changed before assembly")
        records.append(record)
    if missing and not allow_incomplete:
        raise RuntimeError(f"incomplete sweep cells: {missing}")
    if not records:
        raise RuntimeError("no successful cells to assemble")

    by_view: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_view.setdefault(record["view_id"], []).append(record)
    contact_sheets: dict[str, dict[str, str]] = {}
    for view_id, view_records in by_view.items():
        panels = [
            Image.open(out / record["panel"]["path"]).convert("RGB") for record in view_records
        ]
        width = max(panel.width for panel in panels)
        height = max(panel.height for panel in panels)
        canvas = Image.new("RGB", (width, height * len(panels)), "white")
        for index, panel in enumerate(panels):
            canvas.paste(panel, (0, index * height))
        path = out / f"{view_id}_capacity_contact_sheet.png"
        with path.open("xb") as stream:
            canvas.save(stream, format="PNG")
        contact_sheets[view_id] = {
            "path": path.relative_to(out).as_posix(),
            "sha256": sha256_file(path),
        }

    rankings: dict[str, list[dict[str, Any]]] = {}
    for view_id, view_records in by_view.items():
        rankings[view_id] = sorted(
            (
                {
                    "arm": record["arm"]["name"],
                    "m_init_2d": record["optimization"]["m_init_2d"],
                    "m_opt_2d": record["optimization"]["m_opt_2d"],
                    "foreground_clamped_psnr_db": record["metrics"]["foreground_rgb"]["clamped"][
                        "psnr_db"
                    ],
                    "foreground_weighted_ssim": record["metrics"]["foreground_rgb"]["clamped"][
                        "weighted_ssim"
                    ],
                    "wall_seconds_through_fit": record["runtime"]["wall_seconds_through_fit"],
                    "fit_peak_cuda_reserved_bytes": record["runtime"][
                        "fit_peak_cuda_reserved_bytes"
                    ],
                    "cell_peak_cuda_reserved_bytes": record["runtime"][
                        "cell_peak_cuda_reserved_bytes"
                    ],
                }
                for record in view_records
            ),
            key=lambda item: (
                item["foreground_clamped_psnr_db"],
                item["foreground_weighted_ssim"],
            ),
            reverse=True,
        )
    result = {
        "artifact_type": "stage1_capacity_sweep_result_v1",
        "status": "INCOMPLETE" if missing else "PASS",
        "decision_bearing": False,
        "scope": plan["scope"],
        "plan_sha256": plan_sha256,
        "completed_cell_count": len(records),
        "expected_cell_count": len(expected),
        "missing_cells": missing,
        "selected_views": plan["selected_views"],
        "heldout_view_excluded": HELDOUT_VIEW,
        "rankings": rankings,
        "causal_contrasts": plan["causal_contrasts"],
        "contact_sheets": contact_sheets,
        "viewer": {
            "status": "NOT_APPLICABLE_STAGE1_ONLY",
            "reason": "rtgs view consumes a 3D PLY; this diagnostic ends at 2D teachers",
        },
    }
    result_path = out / ("partial_result.json" if missing else "result.json")
    write_json_exclusive(result_path, result)
    return result


def estimated_runtime_table() -> list[dict[str, Any]]:
    """Conservative planning estimates from the historical C0014 640/100 wall time.

    These are scheduling estimates only, not benchmark evidence.  The linear count model is
    intentionally conservative because support area and tiled occupancy do not scale linearly.
    """
    baseline_seconds = 16.693338783981744
    result = []
    for arm in ARMS:
        count_ratio = arm.n_max_2d / 640
        iteration_ratio = arm.iterations / 100
        seconds = baseline_seconds * count_ratio * iteration_ratio
        result.append(
            {
                "arm": arm.name,
                "phase": arm.phase,
                "conservative_seconds": seconds,
                "conservative_minutes": seconds / 60,
                "model": "historical_C0014_640x100_seconds * Nmax/640 * iterations/100",
            }
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    plan = subparsers.add_parser("plan", help="freeze inputs, configs, and source bindings")
    plan.add_argument("--out", type=Path, default=DEFAULT_OUT)
    plan.add_argument("--profile", choices=sorted(PROFILES), default="core")
    plan.add_argument("--arm", action="append", dest="arms")
    plan.add_argument("--view", action="append", dest="views")

    run = subparsers.add_parser("run", help="run exactly one planned CUDA cell")
    run.add_argument("--out", type=Path, default=DEFAULT_OUT)
    run.add_argument("--view", required=True, choices=TRAIN_VIEWS)
    run.add_argument("--arm", required=True, choices=sorted(ARM_BY_NAME))
    run.add_argument("--device", default="cuda")

    finish = subparsers.add_parser(
        "assemble", help="validate cells and build result/contact sheets"
    )
    finish.add_argument("--out", type=Path, default=DEFAULT_OUT)
    finish.add_argument("--allow-incomplete", action="store_true")

    subparsers.add_parser("list", help="print arms, contrasts, and conservative runtime estimates")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "list":
        print(
            json.dumps(
                {
                    "training_views": TRAIN_VIEWS,
                    "default_views": DEFAULT_VIEWS,
                    "heldout_view_excluded": HELDOUT_VIEW,
                    "profiles": PROFILES,
                    "arms": [dataclasses.asdict(arm) for arm in ARMS],
                    "causal_contrasts": CAUSAL_CONTRASTS,
                    "runtime_estimates": estimated_runtime_table(),
                },
                indent=2,
            )
        )
        return 0
    out = args.out.expanduser().resolve()
    if args.action == "plan":
        result = prepare_plan(
            out,
            views=resolve_views(args.views),
            arms=resolve_arms(args.profile, args.arms),
        )
    elif args.action == "run":
        (view_id,) = resolve_views([args.view])
        result = run_cell(out, view_id=view_id, arm_name=args.arm, device_text=args.device)
    elif args.action == "assemble":
        result = assemble(out, allow_incomplete=args.allow_incomplete)
    else:  # pragma: no cover - argparse makes this unreachable
        raise AssertionError(args.action)
    print(json.dumps(json_safe(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
