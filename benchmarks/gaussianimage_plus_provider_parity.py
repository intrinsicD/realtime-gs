#!/usr/bin/env python3
"""Sealed GaussianImage++ direct-covariance provider parity experiment.

The importable portion of this module is CPU-only.  The official run invokes the
foreign CUDA renderer exclusively through ``gaussianimage_plus_native_worker.py``.
No source image is read and no image fitting is performed by this experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
STEM = "20260717_gaussianimage_plus_provider_parity"
PREREGISTRATION = Path(f"benchmarks/results/{STEM}_PREREG.md")
IMPLEMENTATION_REVIEW = Path(f"benchmarks/results/{STEM}_IMPLEMENTATION_REVIEW.md")
IMPLEMENTATION_REVIEW_ADDENDUM = Path(
    f"benchmarks/results/{STEM}_IMPLEMENTATION_REVIEW_ADDENDUM_1.md"
)
SEAL = ROOT / f"benchmarks/results/{STEM}_SEAL.json"
ATTEMPT = ROOT / f"benchmarks/results/{STEM}_ATTEMPT.json"
RESULT = ROOT / f"benchmarks/results/{STEM}_RESULT.json"
RUN_DIR = ROOT / "runs/gaussianimage_plus_provider_parity_20260717"

HARNESS = Path("benchmarks/gaussianimage_plus_provider_parity.py")
WORKER = Path("benchmarks/gaussianimage_plus_native_worker.py")
TEST = Path("tests/test_gaussianimage_plus_provider_parity.py")
SEALED_PATHS = (
    HARNESS,
    WORKER,
    TEST,
    PREREGISTRATION,
    IMPLEMENTATION_REVIEW,
    IMPLEMENTATION_REVIEW_ADDENDUM,
)

PREREGISTRATION_SHA256 = "8b8494bebd3829abdffaf00e7ed905137fc928b1d081e1ae567fb61b267bdae3"

EXTERNAL_REPO = Path("/home/alex/Documents/GaussianImage_plus")
EXTERNAL_COMMIT = "549cfaab2b400248f685c12782a180f3cfc038b0"
EXTERNAL_PYTHON = Path("/home/alex/Documents/structsplat/results/native_envs/image_gs/bin/python")
EXTERNAL_PREFIX = Path("/home/alex/Documents/structsplat/results/native_envs/image_gs")
EXTERNAL_CSRC = EXTERNAL_REPO / "gsplat/gsplat/csrc.so"
EXTERNAL_CSRC_SHA256 = "9b57b7e0531a50d87c529d3541fbf370f9d85455836ac0cf5414c01ce48ac222"
SYSTEM_LIBSTDCXX = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
SYSTEM_LIBSTDCXX_SHA256 = "1fd75fe70354a416d75aef22bcae68c47bd25d20e2d0568c30b1a9838cf62f11"
REAL_CHECKPOINT = Path(
    "/home/alex/Documents/structsplat/results/native_gaussianimage_plus_matched_proxy/"
    "cells/COCO_train2014_000000000009/s160_n640_seed0/native_logs/"
    "COCO_train2014_000000000009_s160/gaussian_model.pth.tar"
)
REAL_CHECKPOINT_SHA256 = "ad611facd72e813dece1b95c3268dbfd82f8af01cdb5ad67e1c7675cc670794b"
REAL_HEIGHT = 120
REAL_WIDTH = 160

BLOCK_SIZE = 16
MAX_TILE_POPULATION = BLOCK_SIZE * BLOCK_SIZE
ALPHA_CUTOFF = 1.0 / 255.0
PROJECTION_ATOL = 2e-6
PROJECTION_RTOL = 5e-6
IMAGE_ATOL = 1e-5
IMAGE_RTOL = 1e-5
MEAN_ABS_LIMIT = 1e-6

EXTERNAL_SOURCE_PATHS = (
    Path("models/gaussianimage_covariance.py"),
    Path("train.py"),
    Path("gsplat/gsplat/project_gaussians_2d_covariance.py"),
    Path("gsplat/gsplat/rasterize_sum_plus.py"),
    Path("gsplat/gsplat/utils.py"),
    Path("gsplat/gsplat/cuda/csrc/config.h"),
    Path("gsplat/gsplat/cuda/csrc/helpers.cuh"),
    Path("gsplat/gsplat/cuda/csrc/foward2d.cu"),
    Path("gsplat/gsplat/cuda/csrc/forward.cu"),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _exclusive_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
    return sha256_bytes(rendered.encode())


def write_terminal_result(
    path: Path,
    payload: dict[str, Any],
    *,
    attempt_sha256: str,
    seal_file_sha256: str,
) -> tuple[str, dict[str, Any]]:
    """Write exactly one finite result, reducing invalid payloads to a FAIL receipt."""

    try:
        canonical_json(payload)
    except (TypeError, ValueError, OverflowError) as error:
        payload = {
            "artifact_type": "gaussianimage_plus_provider_parity_result",
            "status": "FAIL",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "attempt_sha256": attempt_sha256,
            "seal_file_sha256": seal_file_sha256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "error_type": "TerminalPayloadValidationError",
            "cause_type": type(error).__name__,
            "error": str(error),
        }
        canonical_json(payload)
    return _exclusive_json(path, payload), payload


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    header = canonical_json({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
    return sha256_bytes(header.encode() + b"\0" + tensor.numpy().tobytes(order="C"))


def _as_float32_cpu(value: torch.Tensor | np.ndarray | list[float]) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device="cpu").contiguous()


@dataclass(frozen=True)
class DirectCovarianceField:
    """Pure-CPU representation of the GaussianImage++ covariance renderer inputs."""

    means: torch.Tensor
    covariances: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    background: torch.Tensor
    height: int
    width: int
    block_size: int = BLOCK_SIZE
    clip_coe: float = 3.0
    radius_clip: float = 1.0

    def __post_init__(self) -> None:
        tensors = {
            "means": _as_float32_cpu(self.means),
            "covariances": _as_float32_cpu(self.covariances),
            "colors": _as_float32_cpu(self.colors),
            "opacities": _as_float32_cpu(self.opacities),
            "background": _as_float32_cpu(self.background),
        }
        for name, value in tensors.items():
            object.__setattr__(self, name, value)
        n = tensors["means"].shape[0]
        expected = {
            "means": (n, 2),
            "covariances": (n, 3),
            "colors": (n, 3),
            "opacities": (n, 1),
            "background": (3,),
        }
        for name, shape in expected.items():
            value = tensors[name]
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        if n < 1:
            raise ValueError("at least one component is required")
        if self.height < 1 or self.width < 1:
            raise ValueError("height and width must be positive")
        if self.block_size != BLOCK_SIZE:
            raise ValueError("the frozen native binary requires 16x16 tiles")
        if self.clip_coe <= 0 or self.radius_clip < 0:
            raise ValueError("clip_coe must be positive and radius_clip non-negative")

    @property
    def n(self) -> int:
        return int(self.means.shape[0])

    @property
    def tile_bounds(self) -> tuple[int, int]:
        return (
            (self.width + self.block_size - 1) // self.block_size,
            (self.height + self.block_size - 1) // self.block_size,
        )

    def spd_mask(self) -> torch.Tensor:
        xx, xy, yy = self.covariances.unbind(dim=1)
        determinant = xx * yy - xy.square()
        return (determinant > 0.0) & (xx > 0.0) & (yy > 0.0)

    def filtered_spd(self) -> tuple[DirectCovarianceField, torch.Tensor]:
        mask = self.spd_mask()
        if not bool(mask.any()):
            raise ValueError("SPD filtering removed every component")
        return (
            DirectCovarianceField(
                means=self.means[mask],
                covariances=self.covariances[mask],
                colors=self.colors[mask],
                opacities=self.opacities[mask],
                background=self.background,
                height=self.height,
                width=self.width,
                block_size=self.block_size,
                clip_coe=self.clip_coe,
                radius_clip=self.radius_clip,
            ),
            mask,
        )

    def content_hash(self) -> str:
        payload = {
            "means": tensor_sha256(self.means),
            "covariances": tensor_sha256(self.covariances),
            "colors": tensor_sha256(self.colors),
            "opacities": tensor_sha256(self.opacities),
            "background": tensor_sha256(self.background),
            "height": self.height,
            "width": self.width,
            "block_size": self.block_size,
            "clip_coe": self.clip_coe,
            "radius_clip": self.radius_clip,
        }
        return canonical_hash(payload)


@dataclass(frozen=True)
class CPUProjection:
    xys: torch.Tensor
    conics: torch.Tensor
    radii: torch.Tensor
    hits: torch.Tensor
    tile_candidates: tuple[tuple[int, ...], ...]

    @property
    def max_tile_population(self) -> int:
        return max((len(ids) for ids in self.tile_candidates), default=0)


class TilePopulationError(RuntimeError):
    pass


def _tile_bbox(
    center_x: float,
    center_y: float,
    radius: int,
    tiles_x: int,
    tiles_y: int,
    block_size: int,
) -> tuple[int, int, int, int]:
    center_tx = center_x / block_size
    center_ty = center_y / block_size
    radius_t = radius / block_size
    # C/CUDA casts truncate toward zero.  Clamping after truncation is part of the ABI.
    minimum_x = min(max(0, math.trunc(center_tx - radius_t)), tiles_x)
    maximum_x = min(max(0, math.trunc(center_tx + radius_t + 1.0)), tiles_x)
    minimum_y = min(max(0, math.trunc(center_ty - radius_t)), tiles_y)
    maximum_y = min(max(0, math.trunc(center_ty + radius_t + 1.0)), tiles_y)
    return minimum_x, maximum_x, minimum_y, maximum_y


def project_cpu(field: DirectCovarianceField) -> CPUProjection:
    """Reproduce the bundled direct-covariance projection and circular tile bounds."""

    covariances = field.covariances
    xx, xy, yy = covariances.unbind(dim=1)
    determinant = xx * yy - xy.square()
    non_singular = determinant != 0.0

    conics = torch.zeros_like(covariances)
    inverse_determinant = torch.zeros_like(determinant)
    inverse_determinant[non_singular] = 1.0 / determinant[non_singular]
    # Preserve the CUDA kernel's reciprocal-then-multiply operation order.
    conics[non_singular, 0] = yy[non_singular] * inverse_determinant[non_singular]
    conics[non_singular, 1] = -xy[non_singular] * inverse_determinant[non_singular]
    conics[non_singular, 2] = xx[non_singular] * inverse_determinant[non_singular]

    half_trace = 0.5 * (xx + yy)
    root = torch.sqrt(torch.clamp(half_trace.square() - determinant, min=0.1))
    eigenvalue_1 = half_trace + root
    eigenvalue_2 = half_trace - root
    maximum_eigenvalue = torch.maximum(eigenvalue_1, eigenvalue_2)
    minimum_eigenvalue = torch.minimum(eigenvalue_1, eigenvalue_2)
    long_radius_float = torch.ceil(field.clip_coe * torch.sqrt(maximum_eigenvalue))
    short_radius_float = torch.ceil(field.clip_coe * torch.sqrt(minimum_eigenvalue))

    xys = torch.zeros_like(field.means)
    radii = torch.zeros(field.n, dtype=torch.int32)
    hits = torch.zeros(field.n, dtype=torch.int32)
    tiles_x, tiles_y = field.tile_bounds
    candidate_lists: list[list[int]] = [[] for _ in range(tiles_x * tiles_y)]

    for index in range(field.n):
        if not bool(non_singular[index]):
            continue
        short_radius = float(short_radius_float[index])
        # CUDA comparison with NaN is false.  This preserves the raw-checkpoint diagnostic;
        # provider-eligible fields are filtered to SPD before use.
        if short_radius < field.radius_clip:
            conics[index] = 0.0
            continue
        long_radius = float(long_radius_float[index])
        if not math.isfinite(long_radius):
            raise ValueError("native long-axis radius is non-finite")
        radius = math.trunc(long_radius)
        xys[index] = field.means[index]
        radii[index] = radius
        minimum_x, maximum_x, minimum_y, maximum_y = _tile_bbox(
            float(field.means[index, 0]),
            float(field.means[index, 1]),
            radius,
            tiles_x,
            tiles_y,
            field.block_size,
        )
        area = (maximum_x - minimum_x) * (maximum_y - minimum_y)
        if area <= 0:
            continue
        hits[index] = area
        for tile_y in range(minimum_y, maximum_y):
            for tile_x in range(minimum_x, maximum_x):
                candidate_lists[tile_y * tiles_x + tile_x].append(index)

    return CPUProjection(
        xys=xys,
        conics=conics,
        radii=radii,
        hits=hits,
        tile_candidates=tuple(tuple(ids) for ids in candidate_lists),
    )


def render_cpu(
    field: DirectCovarianceField,
    projection: CPUProjection | None = None,
) -> tuple[torch.Tensor, torch.Tensor, CPUProjection]:
    """Evaluate native integer-pixel additive semantics on CPU."""

    projection = project_cpu(field) if projection is None else projection
    if projection.max_tile_population > MAX_TILE_POPULATION:
        raise TilePopulationError(
            f"tile population {projection.max_tile_population} exceeds the frozen cap "
            f"{MAX_TILE_POPULATION}"
        )
    if int(projection.hits.sum()) < 1:
        raw = field.background.view(1, 1, 3).expand(field.height, field.width, 3).clone()
        return raw, raw.clamp(0.0, 1.0), projection

    raw = torch.zeros((field.height, field.width, 3), dtype=torch.float32)
    tiles_x, tiles_y = field.tile_bounds
    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            tile_id = tile_y * tiles_x + tile_x
            candidates = projection.tile_candidates[tile_id]
            if not candidates:
                continue
            y0 = tile_y * field.block_size
            y1 = min((tile_y + 1) * field.block_size, field.height)
            x0 = tile_x * field.block_size
            x1 = min((tile_x + 1) * field.block_size, field.width)
            pixel_y, pixel_x = torch.meshgrid(
                torch.arange(y0, y1, dtype=torch.float32),
                torch.arange(x0, x1, dtype=torch.float32),
                indexing="ij",
            )
            output = torch.zeros((y1 - y0, x1 - x0, 3), dtype=torch.float32)
            for gaussian_id in candidates:
                delta_x = projection.xys[gaussian_id, 0] - pixel_x
                delta_y = projection.xys[gaussian_id, 1] - pixel_y
                conic = projection.conics[gaussian_id]
                sigma = (
                    0.5 * (conic[0] * delta_x.square() + conic[2] * delta_y.square())
                    + conic[1] * delta_x * delta_y
                )
                alpha = torch.minimum(
                    torch.ones_like(sigma),
                    field.opacities[gaussian_id, 0] * torch.exp(-sigma),
                )
                alpha = torch.where(
                    (sigma >= 0.0) & (alpha >= ALPHA_CUTOFF),
                    alpha,
                    0.0,
                )
                output += alpha.unsqueeze(-1) * field.colors[gaussian_id]
            raw[y0:y1, x0:x1] = output
    return raw, raw.clamp(0.0, 1.0), projection


@dataclass(frozen=True)
class CheckpointAdapterResult:
    field: DirectCovarianceField
    checkpoint_num_gaussians: int
    reported_psnr: float | None
    reported_ms_ssim: float | None


def load_checkpoint_cpu(
    path: Path,
    *,
    expected_sha256: str,
    height: int,
    width: int,
    color_norm: bool,
    clip_coe: float = 3.0,
    radius_clip: float = 1.0,
) -> CheckpointAdapterResult:
    """Load a GaussianImage++ checkpoint without importing its repository."""

    payload = path.read_bytes()
    actual_sha256 = sha256_bytes(payload)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "checkpoint bytes differ from the expected SHA-256: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("gs"), dict):
        raise ValueError("checkpoint must contain a gs state dictionary")
    state = checkpoint["gs"]
    required = ("_xyz", "_cov2d", "_features_dc", "_opacity", "background")
    missing = [name for name in required if name not in state]
    if missing or "slv_bound" not in checkpoint:
        raise ValueError(f"checkpoint is missing required fields: {missing}")
    colors = torch.sigmoid(state["_features_dc"]) if color_norm else state["_features_dc"]
    field = DirectCovarianceField(
        means=state["_xyz"],
        covariances=state["_cov2d"] + checkpoint["slv_bound"],
        colors=colors,
        opacities=state["_opacity"],
        background=state["background"],
        height=height,
        width=width,
        block_size=BLOCK_SIZE,
        clip_coe=clip_coe,
        radius_clip=radius_clip,
    )
    if int(checkpoint.get("num_gs", field.n)) != field.n:
        raise ValueError("checkpoint num_gs disagrees with tensor shapes")
    return CheckpointAdapterResult(
        field=field,
        checkpoint_num_gaussians=field.n,
        reported_psnr=(float(checkpoint["psnr"]) if checkpoint.get("psnr") is not None else None),
        reported_ms_ssim=(
            float(checkpoint["ms-ssim"])
            if checkpoint.get("ms-ssim") is not None and math.isfinite(float(checkpoint["ms-ssim"]))
            else None
        ),
    )


def synthetic_fields() -> dict[str, DirectCovarianceField]:
    """Literal outcome-independent semantic fixtures for the official run."""

    base = {"height": 16, "width": 16, "background": torch.ones(3)}
    return {
        "overlap_upper_clamp": DirectCovarianceField(
            means=[[4.0, 5.0], [4.0, 5.0], [10.0, 8.0]],
            covariances=[[4.0, 0.0, 9.0], [4.0, 0.0, 9.0], [2.0, 0.5, 3.0]],
            colors=[[0.8, 0.1, 0.0], [0.6, 0.0, 0.2], [0.0, 0.3, 0.7]],
            opacities=torch.ones(3, 1),
            **base,
        ),
        "fractional_rotated_lower_clamp": DirectCovarianceField(
            means=[[7.25, 8.75], [10.5, 5.5]],
            covariances=[[6.0, 1.5, 3.0], [3.5, -0.75, 2.5]],
            colors=[[-0.4, 0.6, 1.2], [0.3, -0.5, 0.2]],
            opacities=[[0.8], [0.65]],
            **base,
        ),
        "cutoff": DirectCovarianceField(
            means=[[8.0, 8.0]],
            covariances=[[1.0, 0.0, 1.0]],
            colors=[[0.25, 0.5, 0.75]],
            opacities=[[1.0]],
            **base,
        ),
        "all_culled_background": DirectCovarianceField(
            means=[[1000.0, 1000.0]],
            covariances=[[1.0, 0.0, 1.0]],
            colors=[[0.2, 0.3, 0.4]],
            opacities=[[1.0]],
            height=16,
            width=16,
            background=[0.7, 0.8, 0.9],
        ),
        "one_hit_background_ignored": DirectCovarianceField(
            means=[[8.0, 8.0]],
            covariances=[[1.0, 0.0, 1.0]],
            colors=[[0.2, 0.3, 0.4]],
            opacities=[[1.0]],
            height=16,
            width=16,
            background=[0.7, 0.8, 0.9],
        ),
        "radius_clip": DirectCovarianceField(
            means=[[8.0, 8.0]],
            covariances=[[1.0, 0.0, 1.0]],
            colors=[[0.2, 0.3, 0.4]],
            opacities=[[1.0]],
            height=16,
            width=16,
            background=[0.7, 0.8, 0.9],
            radius_clip=4.0,
        ),
        "out_of_frame_intersection": DirectCovarianceField(
            means=[[-1.0, 8.0]],
            covariances=[[4.0, 0.0, 4.0]],
            colors=[[0.2, 0.3, 0.4]],
            opacities=[[1.0]],
            **base,
        ),
        "tile_cap_sentinel": DirectCovarianceField(
            means=torch.tensor([[8.0, 8.0]]).repeat(257, 1),
            covariances=torch.tensor([[1.0, 0.0, 1.0]]).repeat(257, 1),
            colors=torch.tensor([[0.001, 0.002, 0.003]]).repeat(257, 1),
            opacities=torch.ones(257, 1),
            **base,
        ),
    }


def validate_synthetic_semantics(
    name: str,
    raw: torch.Tensor,
    clamped: torch.Tensor,
    projection: CPUProjection,
) -> None:
    if name == "overlap_upper_clamp":
        if not float(raw[5, 4, 0]) > 1.0 or float(clamped[5, 4, 0]) != 1.0:
            raise RuntimeError("overlap fixture did not exercise final upper clamp")
    elif name == "fractional_rotated_lower_clamp":
        if not float(raw.min()) < 0.0 or float(clamped.min()) != 0.0:
            raise RuntimeError("fractional fixture did not exercise final lower clamp")
    elif name == "cutoff":
        if not float(raw[8, 11].abs().sum()) > 0.0:
            raise RuntimeError("3-pixel cutoff control was unexpectedly removed")
        if float(raw[8, 12].abs().sum()) != 0.0:
            raise RuntimeError("4-pixel cutoff sentinel was unexpectedly retained")
    elif name in {"all_culled_background", "radius_clip"}:
        expected = torch.tensor([0.7, 0.8, 0.9])
        if not torch.equal(raw[0, 0], expected):
            raise RuntimeError("all-culled branch did not return background")
    elif name == "one_hit_background_ignored":
        if not torch.equal(raw[0, 0], torch.zeros(3)):
            raise RuntimeError("non-empty render path unexpectedly added background")
    elif name == "out_of_frame_intersection" and int(projection.hits[0]) < 1:
        raise RuntimeError("out-of-frame component did not intersect the image tile")


def save_field_npz(path: Path, field: DirectCovarianceField) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "height": field.height,
        "width": field.width,
        "block_size": field.block_size,
        "clip_coe": field.clip_coe,
        "radius_clip": field.radius_clip,
        "content_hash": field.content_hash(),
    }
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            means=field.means.numpy(),
            covariances=field.covariances.numpy(),
            colors=field.colors.numpy(),
            opacities=field.opacities.numpy(),
            background=field.background.numpy(),
            metadata=np.asarray(canonical_json(metadata)),
        )
    return sha256_file(path)


def load_worker_output(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        output = {name: archive[name].copy() for name in archive.files if name != "metadata"}
        metadata = json.loads(str(archive["metadata"].item()))
    output["metadata"] = metadata
    return output


def _native_candidates(output: dict[str, Any], tile_count: int) -> tuple[tuple[int, ...], ...]:
    ids = output["gaussian_ids_sorted"].astype(np.int64, copy=False)
    bins = output["tile_bins"].astype(np.int64, copy=False)
    candidates = []
    for tile_id in range(tile_count):
        lower, upper = map(int, bins[tile_id])
        candidates.append(tuple(int(value) for value in ids[lower:upper]))
    return tuple(candidates)


def compare_worker_output(
    field: DirectCovarianceField,
    native: dict[str, Any],
    *,
    allow_tile_cap_sentinel: bool = False,
) -> dict[str, Any]:
    raw_cpu, clamped_cpu, projection = render_cpu(field)
    native_raw = torch.from_numpy(native["raw"])
    native_clamped = torch.from_numpy(native["clamped"])
    native_xys = torch.from_numpy(native["xys"])
    native_conics = torch.from_numpy(native["conics"])
    native_radii = torch.from_numpy(native["radii"])
    native_hits = torch.from_numpy(native["hits"])
    native_candidates = _native_candidates(native, len(projection.tile_candidates))
    parameter_tensors = {
        "means": field.means,
        "covariances": field.covariances,
        "colors": field.colors,
        "opacities": field.opacities,
        "background": field.background,
    }
    parameters_exact = all(
        torch.equal(expected, torch.from_numpy(native[name]))
        for name, expected in parameter_tensors.items()
    )

    xys_error = float((projection.xys - native_xys).abs().max())
    conics_error = float((projection.conics - native_conics).abs().max())
    conics_close = torch.allclose(
        projection.conics,
        native_conics,
        atol=PROJECTION_ATOL,
        rtol=PROJECTION_RTOL,
    )
    radii_equal = torch.equal(projection.radii, native_radii.to(torch.int32))
    hits_equal = torch.equal(projection.hits, native_hits.to(torch.int32))
    candidate_sets_equal = all(
        set(cpu_ids) == set(native_ids)
        for cpu_ids, native_ids in zip(projection.tile_candidates, native_candidates, strict=True)
    )
    raw_delta = (raw_cpu - native_raw).abs()
    clamped_delta = (clamped_cpu - native_clamped).abs()
    raw_close = torch.allclose(raw_cpu, native_raw, atol=IMAGE_ATOL, rtol=IMAGE_RTOL)
    clamped_close = torch.allclose(clamped_cpu, native_clamped, atol=IMAGE_ATOL, rtol=IMAGE_RTOL)
    passed = (
        xys_error <= PROJECTION_ATOL
        and parameters_exact
        and conics_close
        and radii_equal
        and hits_equal
        and candidate_sets_equal
        and raw_close
        and clamped_close
        and float(raw_delta.mean()) <= MEAN_ABS_LIMIT
        and projection.max_tile_population <= MAX_TILE_POPULATION
    )
    if allow_tile_cap_sentinel:
        passed = projection.max_tile_population > MAX_TILE_POPULATION
    return {
        "passed": bool(passed),
        "n_gaussians": field.n,
        "field_content_hash": field.content_hash(),
        "parameters_exact": bool(parameters_exact),
        "xys_max_abs": xys_error,
        "conics_max_abs": conics_error,
        "conics_allclose": bool(conics_close),
        "radii_exact": bool(radii_equal),
        "hits_exact": bool(hits_equal),
        "candidate_sets_exact": bool(candidate_sets_equal),
        "max_tile_population": projection.max_tile_population,
        "raw_max_abs": float(raw_delta.max()),
        "raw_mean_abs": float(raw_delta.mean()),
        "clamped_max_abs": float(clamped_delta.max()),
        "raw_allclose": bool(raw_close),
        "clamped_allclose": bool(clamped_close),
    }


def _git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status(path: Path) -> list[str]:
    return subprocess.run(
        ["git", "status", "--short"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def external_static_bindings() -> dict[str, Any]:
    required = [
        EXTERNAL_PYTHON,
        EXTERNAL_CSRC,
        SYSTEM_LIBSTDCXX,
        *(EXTERNAL_REPO / path for path in EXTERNAL_SOURCE_PATHS),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen external inputs: {missing}")
    commit = _git_commit(EXTERNAL_REPO)
    status = _git_status(EXTERNAL_REPO)
    if commit != EXTERNAL_COMMIT or status:
        raise RuntimeError(f"GaussianImage++ checkout drift: commit={commit}, status={status}")
    hashes = {str(path): sha256_file(EXTERNAL_REPO / path) for path in EXTERNAL_SOURCE_PATHS}
    fixed = {
        "repo": str(EXTERNAL_REPO),
        "commit": commit,
        "status": status,
        "source_hashes": hashes,
        "csrc": str(EXTERNAL_CSRC),
        "csrc_sha256": sha256_file(EXTERNAL_CSRC),
        "python": str(EXTERNAL_PYTHON),
        "python_prefix": str(EXTERNAL_PREFIX),
        "preload": str(SYSTEM_LIBSTDCXX),
        "preload_sha256": sha256_file(SYSTEM_LIBSTDCXX),
    }
    expected = {
        "csrc_sha256": EXTERNAL_CSRC_SHA256,
        "preload_sha256": SYSTEM_LIBSTDCXX_SHA256,
    }
    mismatch = {
        key: {"expected": value, "actual": fixed[key]}
        for key, value in expected.items()
        if fixed[key] != value
    }
    if mismatch:
        raise RuntimeError(f"external binary/input hash drift: {mismatch}")
    fixed["aggregate_sha256"] = canonical_hash(fixed)
    return fixed


def external_bindings() -> dict[str, Any]:
    """Bind seal-time external state, including the frozen checkpoint bytes."""

    fixed = {
        "static": external_static_bindings(),
        "checkpoint": {
            "path": str(REAL_CHECKPOINT),
            "sha256": sha256_file(REAL_CHECKPOINT),
        },
    }
    if fixed["checkpoint"]["sha256"] != REAL_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint hash differs from the frozen external input")
    fixed["aggregate_sha256"] = canonical_hash(fixed)
    return fixed


def verify_external_bindings_pre_attempt(sealed: dict[str, Any]) -> None:
    """Verify outcome-free bindings without reading any checkpoint byte."""

    body = dict(sealed)
    aggregate = body.pop("aggregate_sha256", None)
    if aggregate != canonical_hash(body):
        raise RuntimeError("sealed external binding aggregate mismatch")
    if body.get("static") != external_static_bindings():
        raise RuntimeError("static external inputs differ from the implementation seal")
    expected_checkpoint = {
        "path": str(REAL_CHECKPOINT),
        "sha256": REAL_CHECKPOINT_SHA256,
    }
    if body.get("checkpoint") != expected_checkpoint:
        raise RuntimeError("seal does not bind the frozen checkpoint identity")


def sealed_source_hashes() -> tuple[dict[str, str], str]:
    missing = [str(path) for path in SEALED_PATHS if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"sealed files are missing: {missing}")
    hashes = {str(path): sha256_file(ROOT / path) for path in SEALED_PATHS}
    return hashes, canonical_hash(hashes)


def verify_sealed_source_hashes(expected: dict[str, str], aggregate: str) -> None:
    actual, actual_aggregate = sealed_source_hashes()
    if actual != expected or actual_aggregate != aggregate:
        raise RuntimeError("provider parity sources differ from the implementation seal")


def assert_exact_command(phase: str) -> None:
    expected = [str(ROOT / ".venv/bin/python"), str(ROOT / HARNESS), phase]
    actual = [str(Path(sys.executable)), str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    if actual != expected:
        raise RuntimeError(f"official {phase} requires exact command {expected}, got {actual}")


def run_focused_verification() -> dict[str, Any]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        }
    )
    commands = (
        [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "pytest",
            "-q",
            str(TEST),
        ],
        [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "ruff",
            "check",
            str(HARNESS),
            str(WORKER),
            str(TEST),
        ],
        [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "ruff",
            "format",
            "--check",
            str(HARNESS),
            str(WORKER),
            str(TEST),
        ],
    )
    records = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        output = completed.stdout + completed.stderr
        records.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "output_sha256": sha256_bytes(output.encode()),
                "output": output,
            }
        )
        if completed.returncode != 0:
            raise RuntimeError(f"focused verification failed: {command}\n{output}")
    return {"passed": True, "commands": records}


def create_seal() -> dict[str, Any]:
    assert_exact_command("seal")
    if sha256_file(ROOT / PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("preregistration differs from its frozen hash")
    review = (ROOT / IMPLEMENTATION_REVIEW).read_text(encoding="utf-8")
    if "\nVerdict: `FAIL`\n" not in review:
        raise RuntimeError("historical implementation review must retain its exact FAIL verdict")
    addendum = (ROOT / IMPLEMENTATION_REVIEW_ADDENDUM).read_text(encoding="utf-8")
    if "\nVerdict: `PASS`\n" not in addendum:
        raise RuntimeError("implementation review addendum must contain an exact PASS verdict")
    existing = [str(path) for path in (SEAL, ATTEMPT, RESULT) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to seal over existing artifacts: {existing}")
    verification = run_focused_verification()
    hashes, aggregate = sealed_source_hashes()
    payload = {
        "artifact_type": "gaussianimage_plus_provider_parity_implementation_seal",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
        },
        "implementation_reviews": [
            {
                "path": str(IMPLEMENTATION_REVIEW),
                "sha256": sha256_file(ROOT / IMPLEMENTATION_REVIEW),
                "verdict": "FAIL",
            },
            {
                "path": str(IMPLEMENTATION_REVIEW_ADDENDUM),
                "sha256": sha256_file(ROOT / IMPLEMENTATION_REVIEW_ADDENDUM),
                "verdict": "PASS",
            },
        ],
        "source_hashes": hashes,
        "source_aggregate_sha256": aggregate,
        "external_bindings": external_bindings(),
        "verification": verification,
        "repository_revision": _git_commit(ROOT),
        "command": [sys.executable, *sys.argv],
    }
    payload["payload_sha256"] = canonical_hash(payload)
    return payload


def load_and_verify_seal() -> dict[str, Any]:
    seal = strict_json_load(SEAL)
    digest = seal.get("payload_sha256")
    body = dict(seal)
    body.pop("payload_sha256", None)
    if digest != canonical_hash(body):
        raise RuntimeError("seal self-digest mismatch")
    if seal.get("artifact_type") != "gaussianimage_plus_provider_parity_implementation_seal":
        raise RuntimeError("unexpected seal artifact type")
    if seal.get("verification", {}).get("passed") is not True:
        raise RuntimeError("seal does not bind passing CPU verification")
    verify_sealed_source_hashes(seal["source_hashes"], seal["source_aggregate_sha256"])
    verify_external_bindings_pre_attempt(seal["external_bindings"])
    seal["seal_file_sha256"] = sha256_file(SEAL)
    return seal


def worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("LD_LIBRARY_PATH", None)
    environment.pop("TORCH_EXTENSIONS_DIR", None)
    environment.update(
        {
            "LD_PRELOAD": str(SYSTEM_LIBSTDCXX),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(EXTERNAL_REPO / "gsplat"),
            "CUDA_MODULE_LOADING": "LAZY",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    return environment


def run_worker(
    *,
    kind: str,
    input_path: Path,
    expected_input_sha256: str,
    output_path: Path,
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    command = [
        str(EXTERNAL_PYTHON),
        str(ROOT / WORKER),
        "--kind",
        kind,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--repo",
        str(EXTERNAL_REPO),
        "--expected-prefix",
        str(EXTERNAL_PREFIX),
        "--expected-commit",
        EXTERNAL_COMMIT,
        "--expected-csrc-sha256",
        EXTERNAL_CSRC_SHA256,
        "--expected-preload-sha256",
        SYSTEM_LIBSTDCXX_SHA256,
        "--expected-input-sha256",
        expected_input_sha256,
    ]
    if kind == "checkpoint":
        if height is None or width is None:
            raise ValueError("checkpoint worker requires height and width")
        command.extend(["--height", str(height), "--width", str(width)])
    actual_input_sha256 = sha256_file(input_path)
    if actual_input_sha256 != expected_input_sha256:
        raise RuntimeError(
            "worker input differs from its point-of-use SHA-256 binding: "
            f"expected={expected_input_sha256}, actual={actual_input_sha256}"
        )
    completed = subprocess.run(
        command,
        cwd=EXTERNAL_REPO,
        env=worker_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    record = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_sha256": sha256_bytes(completed.stdout.encode()),
        "stderr_sha256": sha256_bytes(completed.stderr.encode()),
    }
    if completed.returncode != 0:
        raise RuntimeError(f"external CUDA worker failed: {record}")
    if not output_path.is_file():
        raise RuntimeError("external CUDA worker did not create its output")
    output_metadata = load_worker_output(output_path)["metadata"]
    if output_metadata.get("input_sha256") != expected_input_sha256:
        raise RuntimeError("external CUDA worker reported an unexpected input SHA-256")
    record["reported_input_sha256"] = output_metadata["input_sha256"]
    record["output_sha256"] = sha256_file(output_path)
    return record


def execute_official_run(seal: dict[str, Any]) -> dict[str, Any]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    synthetic_dir = RUN_DIR / "synthetic"
    checkpoint_dir = RUN_DIR / "checkpoint"
    synthetic_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    results: dict[str, Any] = {}
    worker_records: list[dict[str, Any]] = []

    for name, field in synthetic_fields().items():
        projection = project_cpu(field)
        if name == "tile_cap_sentinel":
            try:
                render_cpu(field, projection)
            except TilePopulationError:
                results[name] = {
                    "passed": projection.max_tile_population == 257,
                    "max_tile_population": projection.max_tile_population,
                    "expected_rejection": True,
                }
            else:
                raise RuntimeError("tile-cap sentinel was not rejected")
            continue
        raw_cpu, clamped_cpu, projection = render_cpu(field, projection)
        validate_synthetic_semantics(name, raw_cpu, clamped_cpu, projection)
        input_path = synthetic_dir / f"{name}_input.npz"
        output_path = synthetic_dir / f"{name}_native.npz"
        input_sha256 = save_field_npz(input_path, field)
        worker_record = run_worker(
            kind="field",
            input_path=input_path,
            expected_input_sha256=input_sha256,
            output_path=output_path,
        )
        worker_records.append(worker_record)
        comparison = compare_worker_output(field, load_worker_output(output_path))
        comparison["input_sha256"] = input_sha256
        comparison["native_output_sha256"] = sha256_file(output_path)
        results[name] = comparison

    expected_checkpoint_sha256 = seal["external_bindings"]["checkpoint"]["sha256"]
    if expected_checkpoint_sha256 != REAL_CHECKPOINT_SHA256:
        raise RuntimeError("seal checkpoint identity differs from the frozen experiment constant")
    adapted = load_checkpoint_cpu(
        REAL_CHECKPOINT,
        expected_sha256=expected_checkpoint_sha256,
        height=REAL_HEIGHT,
        width=REAL_WIDTH,
        color_norm=False,
    )
    raw_output_path = checkpoint_dir / "raw_native.npz"
    raw_worker_record = run_worker(
        kind="checkpoint",
        input_path=REAL_CHECKPOINT,
        expected_input_sha256=expected_checkpoint_sha256,
        output_path=raw_output_path,
        height=REAL_HEIGHT,
        width=REAL_WIDTH,
    )
    worker_records.append(raw_worker_record)
    raw_comparison = compare_worker_output(adapted.field, load_worker_output(raw_output_path))
    raw_comparison.update(
        {
            "diagnostic_only": True,
            "checkpoint_sha256": expected_checkpoint_sha256,
            "native_output_sha256": sha256_file(raw_output_path),
            "spd_components": int(adapted.field.spd_mask().sum()),
            "non_spd_components": int((~adapted.field.spd_mask()).sum()),
        }
    )
    results["raw_checkpoint_diagnostic"] = raw_comparison

    filtered, spd_mask = adapted.field.filtered_spd()
    filtered_input_path = checkpoint_dir / "filtered_input.npz"
    filtered_output_path = checkpoint_dir / "filtered_native.npz"
    filtered_input_sha256 = save_field_npz(filtered_input_path, filtered)
    filtered_worker_record = run_worker(
        kind="field",
        input_path=filtered_input_path,
        expected_input_sha256=filtered_input_sha256,
        output_path=filtered_output_path,
    )
    worker_records.append(filtered_worker_record)
    filtered_comparison = compare_worker_output(filtered, load_worker_output(filtered_output_path))
    raw_cpu, raw_clamped, _ = render_cpu(adapted.field)
    filtered_cpu, filtered_clamped, _ = render_cpu(filtered)
    filter_delta = (raw_clamped - filtered_clamped).abs()
    filtered_comparison.update(
        {
            "provider_eligible": True,
            "raw_n": adapted.field.n,
            "filtered_n": filtered.n,
            "removed_n": int((~spd_mask).sum()),
            "mask_sha256": tensor_sha256(spd_mask),
            "input_sha256": filtered_input_sha256,
            "native_output_sha256": sha256_file(filtered_output_path),
            "raw_to_filtered_clamped_max_abs": float(filter_delta.max()),
            "raw_to_filtered_clamped_mean_abs": float(filter_delta.mean()),
            "pixels_changed_gt_1e_6": int((filter_delta.amax(dim=-1) > 1e-6).sum()),
            "raw_cpu_sha256": tensor_sha256(raw_cpu),
            "filtered_cpu_sha256": tensor_sha256(filtered_cpu),
        }
    )
    results["filtered_checkpoint_provider"] = filtered_comparison

    all_pass = all(bool(result["passed"]) for result in results.values())
    return {
        "artifact_type": "gaussianimage_plus_provider_parity_result",
        "status": "PASS" if all_pass else "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "claim_scope": "renderer/checkpoint-adapter parity only; no image fit or quality claim",
        "seal_file_sha256": seal["seal_file_sha256"],
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "thresholds": {
            "projection_atol": PROJECTION_ATOL,
            "projection_rtol": PROJECTION_RTOL,
            "image_atol": IMAGE_ATOL,
            "image_rtol": IMAGE_RTOL,
            "mean_abs_limit": MEAN_ABS_LIMIT,
            "max_tile_population": MAX_TILE_POPULATION,
        },
        "results": results,
        "worker_records": worker_records,
        "external_bindings": seal["external_bindings"],
    }


def command_seal() -> None:
    payload = create_seal()
    digest = _exclusive_json(SEAL, payload)
    print(json.dumps({"seal": str(SEAL), "sha256": digest}, indent=2))


def command_run() -> None:
    assert_exact_command("run")
    if ATTEMPT.exists() or RESULT.exists():
        raise FileExistsError("official attempt/result already exists")
    seal = load_and_verify_seal()
    marker = {
        "artifact_type": "gaussianimage_plus_provider_parity_attempt",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seal_file_sha256": seal["seal_file_sha256"],
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "command": [sys.executable, *sys.argv],
    }
    marker_sha256 = _exclusive_json(ATTEMPT, marker)
    try:
        result = execute_official_run(seal)
        result["attempt_sha256"] = marker_sha256
    except Exception as error:
        result = {
            "artifact_type": "gaussianimage_plus_provider_parity_result",
            "status": "FAIL",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "attempt_sha256": marker_sha256,
            "seal_file_sha256": seal["seal_file_sha256"],
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    digest, result = write_terminal_result(
        RESULT,
        result,
        attempt_sha256=marker_sha256,
        seal_file_sha256=seal["seal_file_sha256"],
    )
    print(
        json.dumps(
            {"result": str(RESULT), "sha256": digest, "status": result["status"]},
            indent=2,
        )
    )
    if result["status"] != "PASS":
        raise SystemExit(1)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("seal", "run"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.phase == "seal":
        command_seal()
    else:
        command_run()


if __name__ == "__main__":
    main()
