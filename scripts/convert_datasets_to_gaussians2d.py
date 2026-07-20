#!/usr/bin/env python3
"""Convert calibrated repository datasets into capped StructSplat view containers.

The conversion is resumable and transactional at the dataset boundary.  Existing ``.rtgsv``
files are skipped only after strict reload and source/calibration hash checks.  Source ``rgb/``
and ``mask/`` directories are removed only when every discovered calibrated view has a verified
container and every frame manifest reloads successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.calibrated import _camera_id, _find_calibration, _resize_image, _undistort
from rtgs.data.compact_views import (
    COMPACT_VIEW_BYTE_CAP,
    CompactDataset,
    CompactView,
    CompactViewTooLarge,
    file_sha256,
    save_compact_view,
    write_compact_dataset_manifest,
)
from rtgs.image2gs.fit import _crop_to_mask
from rtgs.image2gs.structsplat_backend import field_to_observation

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT / "dataset"
DEFAULT_CACHE_ROOT = ROOT / "runs/dataset_gaussians2d_conversion"
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
N_GAUSSIANS = 5_000
ITERATIONS = 1_000
RENDERER = "cuda_tiled"
SIGMA_CUTOFF = 3.0
SUPPORT_FADE_ALPHA = 1.0
SSIM_WEIGHT = 0.3
LOG_EVERY = 100
MIN_GAUSSIANS = 16


@dataclass(frozen=True, slots=True)
class SourceView:
    frame: Path
    calibration_path: Path
    calibration_sha256: str
    view_id: str
    rgb_path: Path
    rgb_sha256: str
    mask_path: Path | None
    mask_sha256: str | None
    camera_record: dict[str, Any]
    seed: int


@dataclass(frozen=True, slots=True)
class SourceFrame:
    path: Path
    calibration_path: Path
    calibration_sha256: str
    views: tuple[SourceView, ...]
    source_directories: tuple[Path, ...]
    source_files: tuple[tuple[Path, str], ...]


@dataclass(slots=True)
class PreparedView:
    source: SourceView
    camera: Camera
    rgb: torch.Tensor
    target: torch.Tensor
    alpha_crop: torch.Tensor | None
    fit_window: tuple[int, int, int, int]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stable_seed(frame: Path, view_id: str, dataset_root: Path) -> int:
    key = f"{frame.relative_to(dataset_root).as_posix()}:{view_id}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little") & 0x7FFF_FFFF


def _calibration_records(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: dict[str, dict[str, Any]] = {}
    for value in payload["cameras"]:
        view_id = str(value["camera_id"]).upper()
        if view_id in records:
            raise RuntimeError(f"duplicate calibration record {view_id} in {path}")
        records[view_id] = value
    return records


def discover_source_frames(dataset_root: str | Path) -> list[SourceFrame]:
    """Inventory all valid calibrated RGB views and authoritative masks."""
    dataset_root = Path(dataset_root).resolve()
    frames: list[SourceFrame] = []
    for rgb_dir in sorted(dataset_root.rglob("rgb")):
        if not rgb_dir.is_dir() or not rgb_dir.parent.name.startswith("frame_"):
            continue
        frame = rgb_dir.parent
        calibration_path = _find_calibration(frame).resolve()
        calibration_sha256 = file_sha256(calibration_path)
        records = _calibration_records(calibration_path)
        rgb_by_id: dict[str, Path] = {}
        for path in sorted(rgb_dir.iterdir()):
            view_id = _camera_id(path)
            if (
                not path.is_file()
                or path.suffix.lower() not in IMAGE_SUFFIXES
                or path.stem.lower().startswith("mask_")
                or view_id not in records
            ):
                continue
            assert view_id is not None
            if view_id in rgb_by_id:
                raise RuntimeError(f"duplicate RGB view {view_id} in {rgb_dir}")
            rgb_by_id[view_id] = path.resolve()
        if not rgb_by_id:
            raise RuntimeError(f"no calibrated RGB images in {rgb_dir}")
        views: list[SourceView] = []
        mask_dir = frame / "mask"
        for view_id, rgb_path in sorted(rgb_by_id.items()):
            png = mask_dir / f"mask_{view_id}.png"
            jpeg = mask_dir / f"mask_{view_id}.jpg"
            mask_path = png if png.is_file() else jpeg if jpeg.is_file() else None
            views.append(
                SourceView(
                    frame=frame.resolve(),
                    calibration_path=calibration_path,
                    calibration_sha256=calibration_sha256,
                    view_id=view_id,
                    rgb_path=rgb_path,
                    rgb_sha256=file_sha256(rgb_path),
                    mask_path=None if mask_path is None else mask_path.resolve(),
                    mask_sha256=None if mask_path is None else file_sha256(mask_path),
                    camera_record=records[view_id],
                    seed=_stable_seed(frame.resolve(), view_id, dataset_root),
                )
            )
        source_directories = [rgb_dir.resolve()]
        if mask_dir.is_dir():
            source_directories.append(mask_dir.resolve())
        known_hashes = {
            path: digest
            for view in views
            for path, digest in (
                (view.rgb_path, view.rgb_sha256),
                (view.mask_path, view.mask_sha256),
            )
            if path is not None and digest is not None
        }
        source_files = tuple(
            (path.resolve(), known_hashes.get(path.resolve()) or file_sha256(path))
            for directory in source_directories
            for path in sorted(directory.iterdir())
            if path.is_file()
        )
        frames.append(
            SourceFrame(
                path=frame.resolve(),
                calibration_path=calibration_path,
                calibration_sha256=calibration_sha256,
                views=tuple(views),
                source_directories=tuple(source_directories),
                source_files=source_files,
            )
        )
    if not frames:
        raise RuntimeError(f"no source RGB frames found below {dataset_root}")
    return frames


def _camera_from_source(source: SourceView) -> tuple[Camera, list[float], int, int]:
    intrinsics = source.camera_record["intrinsics"]
    width, height = (int(value) for value in intrinsics["resolution"])
    matrix = intrinsics["camera_matrix"]
    fx, fy = float(matrix[0]), float(matrix[4])
    cx, cy = float(matrix[2]) + 0.5, float(matrix[5]) + 0.5
    view = torch.tensor(
        source.camera_record["extrinsics"]["view_matrix"],
        dtype=torch.float32,
    ).reshape(4, 4)
    camera = Camera(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        R=view[:3, :3],
        t=view[:3, 3],
    )
    distortion = [float(value) for value in intrinsics.get("distortion_coefficients", [])]
    return camera, distortion, width, height


def prepare_source_view(source: SourceView) -> PreparedView:
    """Load and undistort a source view at the calibrated native resolution."""
    camera, distortion, width, height = _camera_from_source(source)
    with Image.open(source.rgb_path) as image:
        if image.size != (width, height):
            raise RuntimeError(
                f"{source.rgb_path} is {image.size}, calibration expects {(width, height)}"
            )
    rgb = _undistort(
        _resize_image(source.rgb_path, width, height),
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        distortion,
    ).contiguous()
    if source.mask_path is None:
        return PreparedView(
            source=source,
            camera=camera,
            rgb=rgb,
            target=rgb,
            alpha_crop=None,
            fit_window=(0, 0, width, height),
        )
    foreground = (
        _undistort(
            _resize_image(source.mask_path, width, height, mask=True),
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            distortion,
            mask=True,
        )
        > 0.5
    )
    rgb_crop, mask_crop, offset = _crop_to_mask(rgb, foreground)
    fit_window = (
        int(offset[0]),
        int(offset[1]),
        int(rgb_crop.shape[1]),
        int(rgb_crop.shape[0]),
    )
    return PreparedView(
        source=source,
        camera=camera,
        rgb=rgb_crop.contiguous(),
        target=rgb_crop.contiguous(),
        alpha_crop=mask_crop.bool().contiguous(),
        fit_window=fit_window,
    )


def _gaussian_kernel(
    window: int,
    sigma: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinates = torch.arange(window, device=device, dtype=dtype) - (window - 1) / 2
    kernel = torch.exp(-coordinates.square() / (2 * sigma * sigma))
    return kernel / kernel.sum()


def _separable_filter(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    channels = image.shape[1]
    radius = kernel.numel() // 2
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    value = F.conv2d(image, vertical, padding=(radius, 0), groups=channels)
    return F.conv2d(value, horizontal, padding=(0, radius), groups=channels)


class MaskedSSIM:
    """Differentiable local SSIM normalized over the active foreground."""

    def __init__(
        self,
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        window: int = 11,
        sigma: float = 1.5,
    ) -> None:
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


@torch.no_grad()
def _project_centers_to_foreground(
    field: Any,
    mask: torch.Tensor,
    nearest_y: torch.Tensor,
    nearest_x: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> int:
    field.means[:, 0].clamp_(0, mask.shape[1] - 1)
    field.means[:, 1].clamp_(0, mask.shape[0] - 1)
    x = field.means[:, 0].round().long()
    y = field.means[:, 1].round().long()
    invalid = ~mask[y, x].bool()
    count = int(invalid.sum())
    if count:
        bad_x = x[invalid]
        bad_y = y[invalid]
        field.means[invalid, 0] = nearest_x[bad_y, bad_x].to(field.means)
        field.means[invalid, 1] = nearest_y[bad_y, bad_x].to(field.means)
        state = optimizer.state.get(field.means, {})
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(key)
            if torch.is_tensor(value) and value.shape == field.means.shape:
                value[invalid] = 0
    return count


@torch.no_grad()
def _rounded_centers_inside(field: Any, mask: torch.Tensor) -> torch.Tensor:
    x = field.means[:, 0].round().long().clamp(0, mask.shape[1] - 1)
    y = field.means[:, 1].round().long().clamp(0, mask.shape[0] - 1)
    in_bounds = (
        (field.means[:, 0] >= 0)
        & (field.means[:, 0] <= mask.shape[1] - 1)
        & (field.means[:, 1] >= 0)
        & (field.means[:, 1] <= mask.shape[0] - 1)
    )
    return in_bounds & mask[y, x].bool()


@torch.no_grad()
def _component_activity(
    field: Any,
    mask: torch.Tensor | None,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Return opacity-weighted activity, restricted to foreground when present."""
    from structsplat.render import gaussian_activity

    if mask is None:
        activity = gaussian_activity(
            field.means,
            field.conics(0.0),
            field.radii(SIGMA_CUTOFF, 0.0),
            height,
            width,
            chunk=512,
            support_fade=True,
            sigma_cutoff=SIGMA_CUTOFF,
            support_fade_alpha=SUPPORT_FADE_ALPHA,
        )
    else:
        from structsplat.render import (
            _element_budget,
            _flat_tile_slices,
            _support_weight,
            _tile_bounds,
            _tile_coords,
        )

        conics = field.conics(0.0)
        radii = field.radii(SIGMA_CUTOFF, 0.0)
        x0, y0, tile_width, counts = _tile_bounds(
            field.means,
            radii,
            height,
            width,
        )
        activity = torch.zeros(
            field.n,
            dtype=field.means.dtype,
            device=field.means.device,
        )
        for start, end in _flat_tile_slices(counts, _element_budget(512)):
            component, px, py = _tile_coords(
                x0,
                y0,
                tile_width,
                counts,
                start,
                end,
                field.means.device,
            )
            dx = px.to(field.means.dtype) - field.means[component, 0]
            dy = py.to(field.means.dtype) - field.means[component, 1]
            a, b, c = (
                conics[component, 0],
                conics[component, 1],
                conics[component, 2],
            )
            q = a * dx.square() + 2 * b * dx * dy + c * dy.square()
            weights = _support_weight(
                q,
                SIGMA_CUTOFF,
                True,
                SUPPORT_FADE_ALPHA,
            )
            activity.index_add_(0, component, weights * mask[py, px].to(weights))
    opacity = field.opacity_values()
    if opacity is not None:
        activity = activity * opacity
    return activity


@lru_cache(maxsize=1)
def _implementation_binding() -> dict[str, str | None]:
    """Bind optional-provider and local conversion code used to produce teachers."""
    import structsplat

    try:
        version = importlib.metadata.version("structsplat")
    except importlib.metadata.PackageNotFoundError:
        version = None
    source_root = Path(structsplat.__file__).resolve().parent
    source_hash = hashlib.sha256()
    suffixes = {".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}
    for path in sorted(
        candidate
        for candidate in source_root.rglob("*")
        if candidate.is_file() and candidate.suffix in suffixes
    ):
        source_hash.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        source_hash.update(b"\0")
        source_hash.update(path.read_bytes())
        source_hash.update(b"\0")
    return {
        "structsplat_version": version,
        "structsplat_source_sha256": source_hash.hexdigest(),
        "converter_source_sha256": file_sha256(Path(__file__).resolve()),
        "adapter_source_sha256": file_sha256(ROOT / "src/rtgs/image2gs/structsplat_backend.py"),
    }


def _fit_setup(
    source: SourceView,
    *,
    height: int,
    width: int,
    n_gaussians: int,
    iterations: int,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    from structsplat.config import FitConfig, InitConfig, StructureTensorConfig

    masked = source.mask_path is not None
    init_config = InitConfig(
        strategy="aniso_onedge",
        num_gaussians=n_gaussians,
        seed=source.seed,
        sampling_mode="wse",
        flank_offset_frac=0.0,
        scale_cap_mode="feature",
        scale_cap_max=12.0 * max(height, width) / 160.0,
        background_fraction=0.0,
        background_grid=0,
    )
    tensor_config = StructureTensorConfig()
    fit_config = FitConfig(
        iters=iterations,
        renderer=RENDERER,
        pixel_loss="l1",
        ssim_weight=SSIM_WEIGHT,
        log_every=LOG_EVERY,
        support_fade=True,
        sigma_cutoff=SIGMA_CUTOFF,
        max_gaussians=n_gaussians,
    )
    contract = {
        "schema": "rtgs.dataset_structsplat_fit.v2",
        "provider": _implementation_binding(),
        "initializer": asdict(init_config),
        "structure_tensor": asdict(tensor_config),
        "fit_config": asdict(fit_config),
        "custom_loop": {
            "algorithm": "fixed_count_mask_aware_adam_v1",
            "objective": (
                "0.7*foreground_l1+0.3*foreground_mask_normalized_ssim"
                if masked
                else "0.7*full_l1+0.3*full_border_normalized_ssim"
            ),
            "center_constraint": (
                "nearest_foreground_pixel_with_momentum_reset" if masked else "canvas_clamp"
            ),
            "sigma_cutoff": SIGMA_CUTOFF,
            "support_fade_alpha": SUPPORT_FADE_ALPHA,
        },
        "fit_window_size": [width, height],
        "view_seed": source.seed,
    }
    return init_config, tensor_config, fit_config, contract


def _expected_fit_digest(
    source: SourceView,
    *,
    height: int,
    width: int,
    n_gaussians: int,
    iterations: int,
) -> str:
    _, _, _, contract = _fit_setup(
        source,
        height=height,
        width=width,
        n_gaussians=n_gaussians,
        iterations=iterations,
    )
    return hashlib.sha256(_canonical_json(contract)).hexdigest()


def fit_prepared_view(
    prepared: PreparedView,
    *,
    device: torch.device,
    n_gaussians: int,
    iterations: int,
) -> tuple[GaussianObservationField, np.ndarray]:
    """Fit one fixed-count native StructSplat field and return pruning activity."""
    try:
        from scipy.ndimage import distance_transform_edt
        from structsplat import density as density_module
        from structsplat import structure_tensor as tensor_module
        from structsplat.init import build_field
    except ImportError as error:
        raise RuntimeError(
            "dataset conversion requires the optional StructSplat and SciPy environments"
        ) from error

    height, width = prepared.rgb.shape[:2]
    masked = prepared.alpha_crop is not None
    init_config, tensor_config, fit_config, contract = _fit_setup(
        prepared.source,
        height=height,
        width=width,
        n_gaussians=n_gaussians,
        iterations=iterations,
    )
    structure_input = prepared.rgb
    mask_cpu = (
        torch.ones(height, width, dtype=torch.bool)
        if prepared.alpha_crop is None
        else prepared.alpha_crop
    )
    if masked:
        structure_input = structure_input * mask_cpu[..., None]
    image_np = structure_input.numpy().astype(np.float32, copy=False)
    tensor = tensor_module.compute(image_np, tensor_config)
    density = density_module.density_from_tensor_and_image(
        image_np,
        tensor,
        init_config,
        tensor_config,
    )
    density = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
    if masked:
        density *= mask_cpu.numpy().astype(np.float64, copy=False)
    density_sum = float(density.sum())
    if not math.isfinite(density_sum) or density_sum <= 0:
        raise RuntimeError("StructSplat initialization density has no finite mass")
    density /= density_sum

    field = build_field(
        prepared.rgb.numpy().astype(np.float32, copy=False),
        init_config,
        tensor_config,
        density=density,
        tensor=tensor,
        device=str(device),
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
    target = prepared.target.to(device)
    mask = mask_cpu.to(device)
    mask_float = mask.to(target)
    nearest_y = nearest_x = None
    if masked:
        _, nearest = distance_transform_edt(
            ~mask_cpu.numpy(),
            return_indices=True,
        )
        nearest_y = torch.from_numpy(nearest[0].copy()).to(device)
        nearest_x = torch.from_numpy(nearest[1].copy()).to(device)
        initial_projected = _project_centers_to_foreground(
            field,
            mask,
            nearest_y,
            nearest_x,
            optimizer,
        )
        initial_inside = _rounded_centers_inside(field, mask)
        if not bool(initial_inside.all()):
            raise RuntimeError(
                f"masked initialization placed {int((~initial_inside).sum())} "
                "rounded centers outside foreground"
            )
        if initial_projected:
            print(
                f"[{prepared.source.view_id}] projected {initial_projected} "
                "initial rounded centers into foreground",
                flush=True,
            )
    else:
        with torch.no_grad():
            field.means[:, 0].clamp_(0, width - 1)
            field.means[:, 1].clamp_(0, height - 1)
    similarity = MaskedSSIM(target, mask_float)
    low_scale = math.log(0.35)
    high_scale = math.log(max(height, width))
    started = time.perf_counter()
    for iteration in range(iterations):
        prediction = _render_live(field, height=height, width=width)
        absolute = (prediction - target).abs().mean(dim=-1)
        l1 = (absolute * mask_float).sum() / mask_float.sum()
        ssim = similarity(prediction)
        loss = (1.0 - SSIM_WEIGHT) * l1 + SSIM_WEIGHT * (1.0 - ssim)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            field.log_scales.clamp_(low_scale, high_scale)
            if getattr(field, "scale_max", None) is not None:
                cap = torch.log(torch.clamp(field.scale_max, min=1e-3))
                torch.minimum(field.log_scales, cap, out=field.log_scales)
            if masked:
                assert nearest_y is not None and nearest_x is not None
                projected = _project_centers_to_foreground(
                    field,
                    mask,
                    nearest_y,
                    nearest_x,
                    optimizer,
                )
            else:
                field.means[:, 0].clamp_(0, width - 1)
                field.means[:, 1].clamp_(0, height - 1)
                projected = 0
        if iteration % LOG_EVERY == 0 or iteration == iterations - 1:
            squared = (prediction - target).square().mean(dim=-1)
            mse = (squared * mask_float).sum() / mask_float.sum()
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-30))
            elapsed = time.perf_counter() - started
            print(
                f"[{prepared.source.view_id}] iter={iteration:04d}/{iterations - 1} "
                f"loss={float(loss.detach()):.6f} psnr={float(psnr.detach()):.3f} "
                f"projected={projected} elapsed={elapsed:.1f}s",
                flush=True,
            )
    torch.cuda.synchronize(device)
    if masked:
        terminal_inside = _rounded_centers_inside(field, mask)
        if not bool(terminal_inside.all()):
            raise RuntimeError(
                f"terminal field has {int((~terminal_inside).sum())} "
                "rounded centers outside foreground"
            )
    activity = (
        _component_activity(
            field,
            mask if masked else None,
            height=height,
            width=width,
        )
        .detach()
        .cpu()
        .numpy()
    )
    fit_config_digest = hashlib.sha256(_canonical_json(contract)).hexdigest()
    binding = _implementation_binding()
    observation = field_to_observation(
        field,
        canvas_size=(prepared.camera.height, prepared.camera.width),
        fit_window=prepared.fit_window,
        blend_mode="normalized",
        sigma_cutoff=SIGMA_CUTOFF,
        support_fade_alpha=SUPPORT_FADE_ALPHA,
        aa_dilation=0.0,
        view_id=prepared.source.view_id,
        n_init=n_gaussians,
        producer_version=binding["structsplat_version"],
        producer_source_digest=binding["structsplat_source_sha256"],
        fit_config_digest=fit_config_digest,
    ).to("cpu")
    return observation, np.asarray(activity, dtype=np.float64)


def _subset_observation(
    observation: GaussianObservationField,
    indices: np.ndarray,
) -> GaussianObservationField:
    index = torch.from_numpy(np.asarray(indices, dtype=np.int64))
    replacements: dict[str, Any] = {}
    for name in (
        "means",
        "log_scales",
        "rotations",
        "colors",
        "amplitudes",
        "mean_residuals",
        "color_grads",
        "filter_variance",
    ):
        value = getattr(observation, name)
        replacements[name] = None if value is None else value[index]
    return replace(observation, **replacements)


def _rank_components(activity: np.ndarray) -> np.ndarray:
    activity = np.nan_to_num(
        np.asarray(activity, dtype=np.float64),
        nan=-np.inf,
        posinf=np.finfo(np.float64).max,
        neginf=-np.inf,
    )
    return np.lexsort((np.arange(activity.size, dtype=np.int64), -activity))


def _cache_paths(
    cache_root: Path, source: SourceView, dataset_root: Path
) -> tuple[Path, Path, Path]:
    relative = source.frame.relative_to(dataset_root)
    base = cache_root / relative / source.view_id
    return (
        base.with_suffix(".candidate.npz"),
        base.with_suffix(".activity.npy"),
        base.with_suffix(".json"),
    )


def _cache_contract(
    source: SourceView,
    *,
    height: int,
    width: int,
    n_gaussians: int,
    iterations: int,
) -> dict[str, Any]:
    return {
        "schema": "rtgs.dataset_conversion_cache.v2",
        "view_id": source.view_id,
        "rgb_sha256": source.rgb_sha256,
        "mask_sha256": source.mask_sha256,
        "calibration_sha256": source.calibration_sha256,
        "fit_config_digest": _expected_fit_digest(
            source,
            height=height,
            width=width,
            n_gaussians=n_gaussians,
            iterations=iterations,
        ),
    }


def _load_cache(
    cache_root: Path,
    source: SourceView,
    dataset_root: Path,
    *,
    height: int,
    width: int,
    n_gaussians: int,
    iterations: int,
) -> tuple[GaussianObservationField, np.ndarray] | None:
    teacher_path, activity_path, metadata_path = _cache_paths(
        cache_root,
        source,
        dataset_root,
    )
    if not all(path.is_file() for path in (teacher_path, activity_path, metadata_path)):
        return None
    expected_contract = _cache_contract(
        source,
        height=height,
        width=width,
        n_gaussians=n_gaussians,
        iterations=iterations,
    )
    if json.loads(metadata_path.read_bytes()) != expected_contract:
        return None
    observation = GaussianObservationField.load_npz(teacher_path, strict=True)
    activity = np.load(activity_path, allow_pickle=False)
    if (
        observation.view_id != source.view_id
        or observation.n_init != n_gaussians
        or observation.fit_window[2:] != (width, height)
        or observation.fit_config_digest != expected_contract["fit_config_digest"]
        or activity.shape != (observation.n,)
        or not bool(np.isfinite(activity).all())
    ):
        return None
    return observation, np.asarray(activity, dtype=np.float64)


def _save_cache(
    cache_root: Path,
    source: SourceView,
    dataset_root: Path,
    observation: GaussianObservationField,
    activity: np.ndarray,
    *,
    height: int,
    width: int,
    n_gaussians: int,
    iterations: int,
) -> None:
    teacher_path, activity_path, metadata_path = _cache_paths(
        cache_root,
        source,
        dataset_root,
    )
    teacher_path.parent.mkdir(parents=True, exist_ok=True)
    observation.save_npz(teacher_path, overwrite=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{activity_path.name}.",
        suffix=".tmp",
        dir=activity_path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, np.asarray(activity, dtype=np.float64), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, activity_path)
    finally:
        temporary.unlink(missing_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{metadata_path.name}.",
        suffix=".tmp",
        dir=metadata_path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                json.dumps(
                    _cache_contract(
                        source,
                        height=height,
                        width=width,
                        n_gaussians=n_gaussians,
                        iterations=iterations,
                    ),
                    indent=2,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, metadata_path)
    finally:
        temporary.unlink(missing_ok=True)


def _camera_matches_source(camera: Camera, source: SourceView) -> bool:
    expected, _, _, _ = _camera_from_source(source)
    return (
        camera.fx == expected.fx
        and camera.fy == expected.fy
        and camera.cx == expected.cx
        and camera.cy == expected.cy
        and camera.width == expected.width
        and camera.height == expected.height
        and torch.equal(camera.R.cpu(), expected.R)
        and torch.equal(camera.t.cpu(), expected.t)
    )


def _bundle_matches_source(
    path: Path,
    source: SourceView,
    *,
    n_gaussians: int | None = None,
    iterations: int | None = None,
) -> bool:
    if (n_gaussians is None) != (iterations is None):
        raise ValueError("n_gaussians and iterations must occur together")
    if not path.is_file():
        return False
    try:
        view = CompactView.load(path)
    except (OSError, ValueError):
        return False
    expected_mask = (
        None
        if source.mask_path is None
        else {
            "name": source.mask_path.name,
            "sha256": source.mask_sha256,
        }
    )
    fit_matches = True
    if n_gaussians is not None and iterations is not None:
        _, _, fit_width, fit_height = view.observation.fit_window
        fit_matches = (
            view.observation.n_init == n_gaussians
            and view.observation.fit_config_digest
            == _expected_fit_digest(
                source,
                height=fit_height,
                width=fit_width,
                n_gaussians=n_gaussians,
                iterations=iterations,
            )
        )
    return (
        view.view_id == source.view_id
        and view.calibration_sha256 == source.calibration_sha256
        and _camera_matches_source(view.camera, source)
        and view.source
        == {
            "rgb": {"name": source.rgb_path.name, "sha256": source.rgb_sha256},
            "mask": expected_mask,
        }
        and (view.alpha is not None) == (source.mask_path is not None)
        and fit_matches
    )


def _save_with_backoff(
    output: Path,
    prepared: PreparedView,
    observation: GaussianObservationField,
    activity: np.ndarray,
) -> CompactView:
    order = _rank_components(activity)
    count = observation.n
    while count >= MIN_GAUSSIANS:
        indices = np.sort(order[:count])
        candidate = (
            observation if count == observation.n else _subset_observation(observation, indices)
        )
        try:
            save_compact_view(
                output,
                candidate,
                prepared.camera,
                calibration_sha256=prepared.source.calibration_sha256,
                source_rgb_name=prepared.source.rgb_path.name,
                source_rgb_sha256=prepared.source.rgb_sha256,
                alpha_crop=prepared.alpha_crop,
                source_mask_name=(
                    None if prepared.source.mask_path is None else prepared.source.mask_path.name
                ),
                source_mask_sha256=prepared.source.mask_sha256,
                byte_cap=COMPACT_VIEW_BYTE_CAP,
            )
            view = CompactView.load(output)
            print(
                f"[{prepared.source.view_id}] wrote {output} "
                f"({view.bytes} bytes, {view.observation.n} Gaussians, "
                f"alpha={view.alpha is not None})",
                flush=True,
            )
            return view
        except CompactViewTooLarge as error:
            output.unlink(missing_ok=True)
            bytes_per_component = max(error.actual_bytes / count, 1.0)
            remove = max(
                64, math.ceil((error.actual_bytes - error.byte_cap + 1024) / bytes_per_component)
            )
            next_count = max(MIN_GAUSSIANS, count - remove)
            next_count = max(MIN_GAUSSIANS, (next_count // 32) * 32)
            if next_count >= count:
                next_count = count - 1
            print(
                f"[{prepared.source.view_id}] {error}; retrying with "
                f"{next_count} activity-ranked components",
                flush=True,
            )
            count = next_count
    raise RuntimeError(f"could not fit {prepared.source.view_id} below the compact-view cap")


def convert_one(
    source: SourceView,
    *,
    dataset_root: Path,
    cache_root: Path,
    device_text: str,
    n_gaussians: int,
    iterations: int,
) -> CompactView:
    output = source.frame / "gaussians2d" / f"{source.view_id}.rtgsv"
    if _bundle_matches_source(
        output,
        source,
        n_gaussians=n_gaussians,
        iterations=iterations,
    ):
        view = CompactView.load(output)
        print(
            f"[{source.view_id}] verified existing {output} ({view.bytes} bytes); skipping",
            flush=True,
        )
        return view
    if output.exists():
        raise RuntimeError(f"existing compact view does not match its sources: {output}")
    prepared = prepare_source_view(source)
    cached = _load_cache(
        cache_root,
        source,
        dataset_root,
        height=prepared.fit_window[3],
        width=prepared.fit_window[2],
        n_gaussians=n_gaussians,
        iterations=iterations,
    )
    if cached is None:
        device = torch.device(device_text)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("native StructSplat dataset conversion requires a CUDA device")
        observation, activity = fit_prepared_view(
            prepared,
            device=device,
            n_gaussians=n_gaussians,
            iterations=iterations,
        )
        _save_cache(
            cache_root,
            source,
            dataset_root,
            observation,
            activity,
            height=prepared.fit_window[3],
            width=prepared.fit_window[2],
            n_gaussians=n_gaussians,
            iterations=iterations,
        )
    else:
        observation, activity = cached
        print(f"[{source.view_id}] reusing verified fitted-field cache", flush=True)
    return _save_with_backoff(output, prepared, observation, activity)


def _source_from_args(args: argparse.Namespace) -> SourceView:
    dataset_root = args.dataset_root.resolve()
    frame = args.frame.resolve()
    try:
        frame.relative_to(dataset_root)
    except ValueError as error:
        raise RuntimeError(f"worker frame is outside the dataset root: {frame}") from error
    calibration_path = _find_calibration(frame).resolve()
    calibration_sha256 = file_sha256(calibration_path)
    records = _calibration_records(calibration_path)
    view_id = str(args.view_id).upper()
    if view_id not in records:
        raise RuntimeError(f"worker view is not calibrated: {view_id}")
    matches = [
        path.resolve()
        for path in (frame / "rgb").iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and not path.stem.lower().startswith("mask_")
        and _camera_id(path) == view_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"worker view is not unique in frame: {args.view_id}")
    rgb_path = matches[0]
    mask_dir = frame / "mask"
    png = mask_dir / f"mask_{view_id}.png"
    jpeg = mask_dir / f"mask_{view_id}.jpg"
    mask_path = png if png.is_file() else jpeg if jpeg.is_file() else None
    return SourceView(
        frame=frame,
        calibration_path=calibration_path,
        calibration_sha256=calibration_sha256,
        view_id=view_id,
        rgb_path=rgb_path,
        rgb_sha256=file_sha256(rgb_path),
        mask_path=None if mask_path is None else mask_path.resolve(),
        mask_sha256=None if mask_path is None else file_sha256(mask_path),
        camera_record=records[view_id],
        seed=_stable_seed(frame, view_id, dataset_root),
    )


def _worker_command(
    source: SourceView,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--dataset-root",
        str(args.dataset_root.resolve()),
        "--cache-root",
        str(args.cache_root.resolve()),
        "--frame",
        str(source.frame),
        "--view-id",
        source.view_id,
        "--device",
        args.device,
        "--gaussians",
        str(args.gaussians),
        "--iterations",
        str(args.iterations),
    ]


def _ray_intersection(origins: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    directions = F.normalize(directions, dim=-1)
    identity = torch.eye(3, dtype=origins.dtype, device=origins.device)
    projectors = identity[None] - directions[:, :, None] * directions[:, None, :]
    matrix = projectors.sum(dim=0)
    right = (projectors @ origins[:, :, None]).sum(dim=0)[:, 0]
    return torch.linalg.lstsq(matrix, right).solution


def _bounds_from_views(views: Sequence[CompactView]) -> tuple[torch.Tensor, float]:
    cameras = [view.camera for view in views]
    positions = torch.stack([camera.position for camera in cameras])
    forwards = F.normalize(torch.stack([camera.R[2] for camera in cameras]), dim=-1)
    center = _ray_intersection(positions, forwards)
    alpha_views = [view for view in views if view.alpha is not None]
    if len(alpha_views) == len(views):
        origins: list[torch.Tensor] = []
        directions: list[torch.Tensor] = []
        boxes: list[tuple[CompactView, tuple[float, float, float, float]]] = []
        for view in views:
            assert view.alpha is not None
            foreground = torch.nonzero(view.alpha.crop_mask())
            if foreground.shape[0] < 16:
                continue
            origin_x, origin_y = view.alpha.origin
            centroid_vu = foreground.float().mean(dim=0) + 0.5
            centroid_uv = centroid_vu[[1, 0]]
            centroid_uv += centroid_uv.new_tensor([origin_x, origin_y])
            ray_origin, ray_direction = view.camera.pixel_rays(centroid_uv[None])
            origins.append(ray_origin)
            directions.append(ray_direction[0])
            minimum = foreground.amin(dim=0).float() + 0.5
            maximum = foreground.amax(dim=0).float() + 0.5
            boxes.append(
                (
                    view,
                    (
                        float(minimum[1] + origin_x),
                        float(minimum[0] + origin_y),
                        float(maximum[1] + origin_x),
                        float(maximum[0] + origin_y),
                    ),
                )
            )
        if len(origins) >= 2:
            center = _ray_intersection(torch.stack(origins), torch.stack(directions))
        radii: list[torch.Tensor] = []
        for view, box in boxes:
            distance = (view.camera.position - center).norm()
            u0, v0, u1, v1 = box
            half_width = 0.5 * (u1 - u0) / view.camera.fx
            half_height = 0.5 * (v1 - v0) / view.camera.fy
            radii.append(
                distance
                * torch.sqrt(
                    torch.as_tensor(
                        half_width * half_width + half_height * half_height,
                        dtype=distance.dtype,
                    )
                )
            )
        if radii:
            return center, float((2.4 * torch.stack(radii).median()).clamp_min(1e-3))
    distances = torch.stack([(camera.position - center).norm() for camera in cameras])
    return center, float((0.72 * distances.median()).clamp_min(1e-3))


def _write_frame_manifest(frame: SourceFrame) -> CompactDataset:
    paths = [frame.path / "gaussians2d" / f"{view.view_id}.rtgsv" for view in frame.views]
    views = [CompactView.load(path) for path in paths]
    bounds_hint = _bounds_from_views(views)
    write_compact_dataset_manifest(
        frame.path / "gaussians2d",
        name=frame.path.name,
        calibration_sha256=frame.calibration_sha256,
        view_paths=paths,
        bounds_hint=bounds_hint,
        overwrite=(frame.path / "gaussians2d/manifest.json").exists(),
    )
    dataset = CompactDataset.load(frame.path / "gaussians2d")
    expected_ids = [view.view_id for view in frame.views]
    if [view.view_id for view in dataset.views] != expected_ids:
        raise RuntimeError(f"frame manifest order mismatch for {frame.path}")
    return dataset


def _verify_current_source_hashes(frames: Sequence[SourceFrame]) -> None:
    checked_calibrations: set[Path] = set()
    for frame in frames:
        if frame.calibration_path not in checked_calibrations:
            if file_sha256(frame.calibration_path) != frame.calibration_sha256:
                raise RuntimeError(
                    f"calibration changed during conversion: {frame.calibration_path}"
                )
            checked_calibrations.add(frame.calibration_path)
        expected_files = dict(frame.source_files)
        actual_files = {
            path.resolve()
            for directory in frame.source_directories
            for path in directory.iterdir()
            if path.is_file()
        }
        if actual_files != set(expected_files):
            raise RuntimeError(
                f"source directory inventory changed during conversion: {frame.path}"
            )
        for path, expected_digest in expected_files.items():
            if file_sha256(path) != expected_digest:
                raise RuntimeError(f"source file changed during conversion: {path}")


def verify_source_conversion(
    frames: Sequence[SourceFrame],
    *,
    n_gaussians: int | None = None,
    iterations: int | None = None,
) -> list[CompactDataset]:
    """Verify complete source coverage, hashes, alpha, caps, and manifests."""
    if (n_gaussians is None) != (iterations is None):
        raise ValueError("n_gaussians and iterations must occur together")
    _verify_current_source_hashes(frames)
    datasets: list[CompactDataset] = []
    total = 0
    for frame in frames:
        dataset = CompactDataset.load(frame.path / "gaussians2d")
        expected_ids = [view.view_id for view in frame.views]
        if [view.view_id for view in dataset.views] != expected_ids:
            raise RuntimeError(f"compact view inventory mismatch for {frame.path}")
        for source, view in zip(frame.views, dataset.views, strict=True):
            if not _bundle_matches_source(
                view.path,
                source,
                n_gaussians=n_gaussians,
                iterations=iterations,
            ):
                raise RuntimeError(f"compact view no longer matches source: {view.path}")
            if view.bytes > COMPACT_VIEW_BYTE_CAP:
                raise RuntimeError(f"compact view exceeds byte cap: {view.path}")
        datasets.append(dataset)
        total += dataset.n_views
    expected_total = sum(len(frame.views) for frame in frames)
    if total != expected_total:
        raise RuntimeError(f"verified {total} views, expected {expected_total}")
    return datasets


def remove_verified_sources(
    frames: Sequence[SourceFrame],
    *,
    dataset_root: Path,
    n_gaussians: int | None = None,
    iterations: int | None = None,
) -> None:
    """Delete source directories only after the global conversion gate passes."""
    verify_source_conversion(
        frames,
        n_gaussians=n_gaussians,
        iterations=iterations,
    )
    roots = sorted(
        {directory for frame in frames for directory in frame.source_directories},
        key=lambda path: (len(path.parts), path.as_posix()),
        reverse=True,
    )
    for directory in roots:
        try:
            directory.relative_to(dataset_root)
        except ValueError as error:
            raise RuntimeError(
                f"refusing to remove source outside dataset root: {directory}"
            ) from error
        if directory.name not in {"rgb", "mask"} or not directory.is_dir():
            raise RuntimeError(f"refusing to remove unexpected source directory: {directory}")
    _verify_current_source_hashes(frames)
    for directory in roots:
        print(f"removing verified source directory {directory}", flush=True)
        shutil.rmtree(directory)
    remaining = [
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if remaining:
        raise RuntimeError(f"source image files remain after finalization: {remaining[:5]}")
    for frame in frames:
        CompactDataset.load(frame.path / "gaussians2d")


def run_conversion(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    frames = discover_source_frames(dataset_root)
    sources = [view for frame in frames for view in frame.views]
    if args.max_views is not None:
        sources = sources[: args.max_views]
    print(
        f"discovered {sum(len(frame.views) for frame in frames)} views across "
        f"{len(frames)} frames; processing {len(sources)}",
        flush=True,
    )
    for index, source in enumerate(sources, start=1):
        print(
            f"[{index}/{len(sources)}] {source.frame.relative_to(dataset_root)}/{source.view_id}",
            flush=True,
        )
        subprocess.run(_worker_command(source, args), check=True)
    processed = {(source.frame, source.view_id) for source in sources}
    for frame in frames:
        if all(
            (frame.path, view.view_id) in processed
            or _bundle_matches_source(
                frame.path / "gaussians2d" / f"{view.view_id}.rtgsv",
                view,
                n_gaussians=args.gaussians,
                iterations=args.iterations,
            )
            for view in frame.views
        ):
            _write_frame_manifest(frame)
    if args.max_views is not None:
        print("partial conversion complete; source deletion is disabled with --max-views")
        return
    if args.remove_sources:
        remove_verified_sources(
            frames,
            dataset_root=dataset_root,
            n_gaussians=args.gaussians,
            iterations=args.iterations,
        )
        print("conversion verified and original RGB/mask directories removed", flush=True)
    else:
        verify_source_conversion(
            frames,
            n_gaussians=args.gaussians,
            iterations=args.iterations,
        )
        print("conversion verified; rerun with --remove-sources to finalize", flush=True)


def verify_converted_after_source_removal(dataset_root: Path) -> int:
    directories = sorted(dataset_root.rglob("gaussians2d"))
    if not directories:
        raise RuntimeError("no compact datasets found")
    total = 0
    for directory in directories:
        dataset = CompactDataset.load(directory)
        total += dataset.n_views
        print(f"verified {directory}: {dataset.n_views} views", flush=True)
    remaining = [
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if remaining:
        raise RuntimeError(f"source images remain: {remaining[:5]}")
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert = subparsers.add_parser("convert")
    convert.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    convert.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    convert.add_argument("--device", default="cuda")
    convert.add_argument("--gaussians", type=int, default=N_GAUSSIANS)
    convert.add_argument("--iterations", type=int, default=ITERATIONS)
    convert.add_argument("--max-views", type=int)
    convert.add_argument("--remove-sources", action="store_true")

    worker = subparsers.add_parser("_worker")
    worker.add_argument("--dataset-root", type=Path, required=True)
    worker.add_argument("--cache-root", type=Path, required=True)
    worker.add_argument("--frame", type=Path, required=True)
    worker.add_argument("--view-id", required=True)
    worker.add_argument("--device", required=True)
    worker.add_argument("--gaussians", type=int, required=True)
    worker.add_argument("--iterations", type=int, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "_worker":
        source = _source_from_args(args)
        convert_one(
            source,
            dataset_root=args.dataset_root.resolve(),
            cache_root=args.cache_root.resolve(),
            device_text=args.device,
            n_gaussians=args.gaussians,
            iterations=args.iterations,
        )
        return
    if args.command == "verify":
        total = verify_converted_after_source_removal(args.dataset_root.resolve())
        print(f"verified {total} compact views", flush=True)
        return
    if args.gaussians < MIN_GAUSSIANS:
        raise ValueError(f"--gaussians must be at least {MIN_GAUSSIANS}")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.max_views is not None and args.max_views <= 0:
        raise ValueError("--max-views must be positive")
    run_conversion(args)


if __name__ == "__main__":
    main()
