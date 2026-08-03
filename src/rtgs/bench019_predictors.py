"""Fail-closed Stage-1 predictors for the pre-outcome StructSplat BENCH-019 lane.

The collector strictly reloads compact fields, queries their declared additive or normalized
equation, and compares deterministic pixel samples with adapter-bound source RGB and alpha.  A
normalized weight sum is reported only as support; it is never renamed alpha or density truth.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from rtgs.bench019 import ExportError, canonical_json, describe_artifact, load_json_object
from rtgs.bench019_adapters import (
    SafeTumArchive,
    derive_registered_depth_mask_png,
    materialized_calibration_payload,
    validate_source_adapter,
)
from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationIndex
from rtgs.data.compact_views import CompactDataset, CompactView

PREDICTOR_SCHEMA = "rtgs.structsplat_bench019.stage1_predictors.v1"
SUPPORT_DEFINITION = "exact_query_weight_sum_strictly_greater_than_zero_not_alpha"

SUPPORTED_PREDICTORS = (
    "train_sampled_foreground_rgb_mse",
    "train_sampled_foreground_psnr_db",
    "train_sampled_boundary_rgb_mae",
    "train_sampled_query_rgb_mae",
    "train_sampled_support_iou",
    "train_sampled_support_precision",
    "train_sampled_support_recall",
    "all_total_rows",
    "all_complete_field_bytes",
)
UNSUPPORTED_PREDICTORS = {
    "alpha_agreement": "field weight sums are support diagnostics, not source alpha",
    "field_conditioning": "no preregistered finite conditioning estimator exists",
    "lpips": "dense perceptual evaluation is outside the CPU point-query collector",
    "ms_ssim": "dense multiscale evaluation is outside the CPU point-query collector",
    "track_yield": "no source-backed correspondence or track artifact is available",
}

_FAMILY_CONTRACTS = {
    "gaussianimage_additive": {
        "id": "gaussianimage_additive",
        "provider": "native",
        "equation": "additive_sum",
        "blend_mode": "additive",
    },
    "structsplat_normalized_no_boundary": {
        "id": "structsplat_normalized_no_boundary",
        "provider": "structsplat",
        "equation": "normalized_weighted_sum",
        "blend_mode": "normalized",
    },
    "structsplat_normalized_mask_contained": {
        "id": "structsplat_normalized_mask_contained",
        "provider": "structsplat",
        "equation": "normalized_weighted_sum",
        "blend_mode": "normalized",
    },
}
FIELD_FAMILIES = tuple(_FAMILY_CONTRACTS)

_TOP_KEYS = frozenset(
    {
        "schema",
        "state",
        "capture_id",
        "role",
        "field_family",
        "source_adapter",
        "compact_field",
        "sample_policy",
        "support_definition",
        "unsupported_predictors",
        "views",
        "aggregates",
        "predictors",
        "semantic_digest",
    }
)
_FAMILY_KEYS = frozenset({"id", "provider", "equation", "blend_mode"})
_FIELD_KEYS = frozenset({"root", "manifest", "view_byte_cap", "complete_field_bytes", "files"})
_FIELD_FILE_KEYS = frozenset({"view_id", "artifact"})
_ARTIFACT_KEYS = frozenset({"path", "sha256", "bytes"})
_SAMPLE_POLICY_KEYS = frozenset(
    {
        "version",
        "seed",
        "sample_cap_per_stratum",
        "boundary_radius_px",
        "component_chunk",
        "tile_size",
        "max_index_entries",
        "max_index_candidates",
        "max_query_pairs",
        "view_byte_cap",
        "psnr_mse_floor",
        "requested_predictors",
    }
)
_VIEW_KEYS = frozenset(
    {"id", "ordinal", "split", "rows", "samples", "sufficient_statistics", "metrics"}
)
_SAMPLES_KEYS = frozenset({"query", "foreground", "boundary"})
_SAMPLE_KEYS = frozenset({"population", "count", "sha256"})
_STAT_KEYS = frozenset(
    {
        "query_rgb_absolute_error_sum",
        "query_rgb_value_count",
        "foreground_rgb_squared_error_sum",
        "foreground_rgb_value_count",
        "boundary_rgb_absolute_error_sum",
        "boundary_rgb_value_count",
        "support_true_positive",
        "support_false_positive",
        "support_false_negative",
        "support_true_negative",
    }
)
_METRIC_KEYS = frozenset(
    {
        "sampled_query_rgb_mae",
        "sampled_foreground_rgb_mse",
        "sampled_foreground_psnr_db",
        "sampled_boundary_rgb_mae",
        "sampled_support_iou",
        "sampled_support_precision",
        "sampled_support_recall",
    }
)
_AGGREGATE_KEYS = frozenset({"n_views", "total_rows", "sufficient_statistics", "metrics"})


@dataclass(frozen=True)
class PredictorConfig:
    """Deterministic, bounded CPU collection controls."""

    seed: int = 0
    sample_cap_per_stratum: int = 4096
    boundary_radius_px: int = 3
    component_chunk: int = 256
    tile_size: int = 16
    max_index_entries: int = GaussianObservationIndex.DEFAULT_MAX_ENTRIES
    max_index_candidates: int = GaussianObservationIndex.DEFAULT_MAX_CANDIDATES
    max_query_pairs: int = GaussianObservationIndex.DEFAULT_MAX_QUERY_PAIRS
    view_byte_cap: int = 8_388_608
    psnr_mse_floor: float = 1e-12

    def __post_init__(self) -> None:
        integer_values = {
            "seed": self.seed,
            "sample_cap_per_stratum": self.sample_cap_per_stratum,
            "boundary_radius_px": self.boundary_radius_px,
            "component_chunk": self.component_chunk,
            "tile_size": self.tile_size,
            "max_index_entries": self.max_index_entries,
            "max_index_candidates": self.max_index_candidates,
            "max_query_pairs": self.max_query_pairs,
            "view_byte_cap": self.view_byte_cap,
        }
        for name, value in integer_values.items():
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer at least {minimum}")
        if (
            isinstance(self.psnr_mse_floor, bool)
            or not isinstance(self.psnr_mse_floor, (int, float))
            or not math.isfinite(float(self.psnr_mse_floor))
            or not 0.0 < float(self.psnr_mse_floor) < 1.0
        ):
            raise ValueError("psnr_mse_floor must be finite and strictly between zero and one")


def _exact(value: object, keys: frozenset[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = set(value) if isinstance(value, dict) else set()
        raise ExportError(
            f"{label} keys are not exact "
            f"(missing={sorted(keys - actual)}, extra={sorted(actual - keys)})"
        )
    return value


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExportError(f"{label} must be a lowercase SHA-256")
    return value


def _artifact(value: object, *, label: str, verify_file: bool) -> dict[str, Any]:
    record = _exact(value, _ARTIFACT_KEYS, label=label)
    if not isinstance(record["path"], str) or not Path(record["path"]).is_absolute():
        raise ExportError(f"{label}.path must be absolute")
    _sha256(record["sha256"], label=f"{label}.sha256")
    if (
        isinstance(record["bytes"], bool)
        or not isinstance(record["bytes"], int)
        or record["bytes"] < 0
    ):
        raise ExportError(f"{label}.bytes must be a non-negative integer")
    if verify_file and describe_artifact(record["path"]) != record:
        raise ExportError(f"{label} differs from its bound file")
    return record


def _requested_predictors(requested: Sequence[str] | None) -> tuple[str, ...]:
    result = SUPPORTED_PREDICTORS if requested is None else tuple(requested)
    if not result:
        raise ExportError("at least one Stage-1 predictor must be requested")
    if any(not isinstance(item, str) for item in result) or len(set(result)) != len(result):
        raise ExportError("requested predictors must be unique strings")
    unavailable = [item for item in result if item in UNSUPPORTED_PREDICTORS]
    if unavailable:
        details = "; ".join(f"{item}: {UNSUPPORTED_PREDICTORS[item]}" for item in unavailable)
        raise ExportError(f"requested Stage-1 predictors are unavailable: {details}")
    unknown = [item for item in result if item not in SUPPORTED_PREDICTORS]
    if unknown:
        raise ExportError(f"requested Stage-1 predictors are unknown: {unknown}")
    return result


def _source_inventory(adapter: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item["artifact"] for item in adapter["source_artifacts"]}


def _camera_matches(camera: Camera, record: Mapping[str, Any]) -> bool:
    if (
        camera.width != record["width"]
        or camera.height != record["height"]
        or not math.isclose(camera.fx, float(record["fx"]), abs_tol=1e-9, rel_tol=0.0)
        or not math.isclose(camera.fy, float(record["fy"]), abs_tol=1e-9, rel_tol=0.0)
        or not math.isclose(camera.cx, float(record["cx"]), abs_tol=1e-9, rel_tol=0.0)
        or not math.isclose(camera.cy, float(record["cy"]), abs_tol=1e-9, rel_tol=0.0)
    ):
        return False
    rotation = camera.R.detach().cpu().double().numpy().reshape(-1)
    translation = camera.t.detach().cpu().double().numpy().reshape(-1)
    return bool(
        np.allclose(rotation, record["R"], atol=1e-6, rtol=0.0)
        and np.allclose(translation, record["t"], atol=1e-6, rtol=0.0)
    )


def _decode_png(payload: bytes, *, mode: str, label: str) -> np.ndarray:
    try:
        with PILImage.open(io.BytesIO(payload)) as image:
            image.load()
            return np.array(image.convert(mode), copy=True)
    except Exception as error:
        raise ExportError(f"cannot decode {label}") from error


def _decode_path(path: str | Path, *, mode: str, label: str) -> np.ndarray:
    try:
        with PILImage.open(path) as image:
            image.load()
            return np.array(image.convert(mode), copy=True)
    except Exception as error:
        raise ExportError(f"cannot decode {label}") from error


def _distorted_positions(
    xy: torch.Tensor,
    camera: Mapping[str, Any],
    distortion: Sequence[float],
) -> torch.Tensor:
    points = xy.to(dtype=torch.float32, device="cpu")
    x = (points[:, 0] - float(camera["cx"])) / float(camera["fx"])
    y = (points[:, 1] - float(camera["cy"])) / float(camera["fy"])
    coefficients = [float(item) for item in distortion]
    k1, k2, p1, p2, k3 = (coefficients + [0.0] * 5)[:5]
    r2 = x.square() + y.square()
    radial = 1.0 + k1 * r2 + k2 * r2.square() + k3 * r2.pow(3)
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x.square())
    yd = y * radial + p1 * (r2 + 2.0 * y.square()) + 2.0 * p2 * x * y
    return torch.stack(
        (
            float(camera["fx"]) * xd + float(camera["cx"]),
            float(camera["fy"]) * yd + float(camera["cy"]),
        ),
        dim=-1,
    )


def _sample_source_rgb(
    source: np.ndarray,
    xy: torch.Tensor,
    *,
    camera: Mapping[str, Any],
    distortion: Sequence[float],
) -> torch.Tensor:
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        raise ExportError("source RGB must decode to uint8 HxWx3")
    height, width = source.shape[:2]
    if (width, height) != (camera["width"], camera["height"]):
        raise ExportError("source RGB dimensions differ from the adapter camera")
    positions = _distorted_positions(xy, camera, distortion)
    index = positions - 0.5
    x0 = torch.floor(index[:, 0]).long()
    y0 = torch.floor(index[:, 1]).long()
    dx = index[:, 0] - x0
    dy = index[:, 1] - y0
    image = torch.from_numpy(source)
    result = torch.zeros((xy.shape[0], 3), dtype=torch.float32)
    for x_index, y_index, weight in (
        (x0, y0, (1.0 - dx) * (1.0 - dy)),
        (x0 + 1, y0, dx * (1.0 - dy)),
        (x0, y0 + 1, (1.0 - dx) * dy),
        (x0 + 1, y0 + 1, dx * dy),
    ):
        valid = (x_index >= 0) & (x_index < width) & (y_index >= 0) & (y_index < height)
        if bool(valid.any()):
            pixels = image[y_index[valid], x_index[valid]].to(torch.float32) / 255.0
            result[valid] += weight[valid, None] * pixels
    return result


def _derive_alpha_crop(
    source: np.ndarray,
    *,
    camera: Mapping[str, Any],
    distortion: Sequence[float],
    fit_window: tuple[int, int, int, int],
    row_chunk: int = 64,
) -> torch.Tensor:
    if source.ndim != 2 or source.dtype != np.uint8:
        raise ExportError("source mask must decode to uint8 HxW")
    height, width = source.shape
    if (width, height) != (camera["width"], camera["height"]):
        raise ExportError("source mask dimensions differ from the adapter camera")
    x0, y0, crop_width, crop_height = fit_window
    if max((abs(float(item)) for item in distortion), default=0.0) < 1e-12:
        return torch.from_numpy((source[y0 : y0 + crop_height, x0 : x0 + crop_width] > 127).copy())
    mask = torch.from_numpy(source)
    result = torch.zeros((crop_height, crop_width), dtype=torch.bool)
    xs = torch.arange(x0, x0 + crop_width, dtype=torch.float32) + 0.5
    for start in range(0, crop_height, row_chunk):
        stop = min(start + row_chunk, crop_height)
        ys = torch.arange(y0 + start, y0 + stop, dtype=torch.float32) + 0.5
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
        positions = _distorted_positions(xy, camera, distortion) - 0.5
        source_x = torch.round(positions[:, 0]).long()
        source_y = torch.round(positions[:, 1]).long()
        valid = (source_x >= 0) & (source_x < width) & (source_y >= 0) & (source_y < height)
        values = torch.zeros(xy.shape[0], dtype=torch.bool)
        if bool(valid.any()):
            values[valid] = mask[source_y[valid], source_x[valid]] > 127
        result[start:stop] = values.reshape(stop - start, crop_width)
    return result


def _boundary_band(alpha: torch.Tensor, radius: int) -> torch.Tensor:
    values = alpha.to(torch.float32)[None, None]
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(values, kernel, stride=1, padding=radius)[0, 0] > 0.5
    complement = F.pad(1.0 - values, (radius, radius, radius, radius), value=1.0)
    eroded = 1.0 - F.max_pool2d(complement, kernel, stride=1)[0, 0]
    return dilated & (eroded < 0.5)


def _choose_positions(population: int, count: int, *, token: str) -> torch.Tensor:
    if population <= 0:
        raise ExportError(f"sample stratum {token!r} is empty")
    selected_count = min(count, population)
    if selected_count == population:
        return torch.arange(population, dtype=torch.int64)
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "little") % population
    stride = int.from_bytes(digest[8:16], "little") % (population - 1) + 1
    while math.gcd(stride, population) != 1:
        stride = stride % (population - 1) + 1
    return (offset + stride * torch.arange(selected_count, dtype=torch.int64)) % population


def _choose_indices(
    candidates: torch.Tensor | int,
    count: int,
    *,
    token: str,
) -> tuple[int, torch.Tensor]:
    population = candidates if isinstance(candidates, int) else int(candidates.numel())
    positions = _choose_positions(population, count, token=token)
    chosen = positions if isinstance(candidates, int) else candidates[positions]
    return population, chosen


def _sample_set(
    candidates: torch.Tensor | int,
    *,
    cap: int,
    seed: int,
    view_id: str,
    stratum: str,
    fit_window: tuple[int, int, int, int],
    canvas_width: int,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    x0, y0, width, _height = fit_window
    population, chosen = _choose_indices(
        candidates,
        cap,
        token=f"rtgs-bench019-samples-v1\0{seed}\0{view_id}\0{stratum}",
    )
    local_x = chosen.remainder(width)
    local_y = torch.div(chosen, width, rounding_mode="floor")
    full_x = local_x + x0
    full_y = local_y + y0
    full_linear = full_y * canvas_width + full_x
    payload = np.asarray(full_linear.numpy(), dtype="<i8").tobytes()
    header = canonical_json(
        {"schema": "rtgs.bench019.sample_indices.v1", "view_id": view_id, "stratum": stratum}
    )
    digest = hashlib.sha256(header + b"\0" + payload).hexdigest()
    xy = torch.stack((full_x, full_y), dim=-1).to(dtype=dtype) + 0.5
    return (
        {"population": population, "count": int(chosen.numel()), "sha256": digest},
        chosen,
        xy,
    )


def _metrics(statistics: Mapping[str, Any], *, psnr_floor: float) -> dict[str, float]:
    foreground_mse = (
        float(statistics["foreground_rgb_squared_error_sum"])
        / statistics["foreground_rgb_value_count"]
    )
    true_positive = statistics["support_true_positive"]
    false_positive = statistics["support_false_positive"]
    false_negative = statistics["support_false_negative"]
    iou_denominator = true_positive + false_positive + false_negative
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    return {
        "sampled_query_rgb_mae": float(statistics["query_rgb_absolute_error_sum"])
        / statistics["query_rgb_value_count"],
        "sampled_foreground_rgb_mse": foreground_mse,
        "sampled_foreground_psnr_db": -10.0 * math.log10(max(foreground_mse, psnr_floor)),
        "sampled_boundary_rgb_mae": float(statistics["boundary_rgb_absolute_error_sum"])
        / statistics["boundary_rgb_value_count"],
        "sampled_support_iou": true_positive / iou_denominator if iou_denominator else 0.0,
        "sampled_support_precision": (
            true_positive / precision_denominator if precision_denominator else 0.0
        ),
        "sampled_support_recall": true_positive / recall_denominator if recall_denominator else 0.0,
    }


def _aggregate(
    views: Sequence[Mapping[str, Any]],
    *,
    psnr_floor: float,
) -> dict[str, Any]:
    if not views:
        raise ExportError("predictor aggregate cannot be empty")
    statistics: dict[str, int | float] = {}
    for key in _STAT_KEYS:
        statistics[key] = sum(view["sufficient_statistics"][key] for view in views)
    return {
        "n_views": len(views),
        "total_rows": sum(view["rows"] for view in views),
        "sufficient_statistics": statistics,
        "metrics": _metrics(statistics, psnr_floor=psnr_floor),
    }


class _AdapterSources:
    """One-view-at-a-time source decoder with a single safe TUM archive handle."""

    def __init__(self, adapter: Mapping[str, Any]):
        self.adapter = adapter
        self.sources = _source_inventory(adapter)
        self.archive: SafeTumArchive | None = None

    def __enter__(self) -> _AdapterSources:
        if self.adapter["source_kind"] == "tum_rgbd_archive":
            self.archive = SafeTumArchive(self.sources["official_archive"]["path"])
            self.archive.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.archive is not None:
            self.archive.__exit__(exc_type, exc_value, traceback)

    def calibration_sha256(self) -> str:
        if self.adapter["source_kind"] == "calibrated_multiview":
            return self.sources["calibration"]["sha256"]
        payload = canonical_json(materialized_calibration_payload(self.adapter)) + b"\n"
        return hashlib.sha256(payload).hexdigest()

    def read_view(
        self,
        adapter_view: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        view_id = adapter_view["id"]
        if self.adapter["source_kind"] == "calibrated_multiview":
            rgb_descriptor = self.sources[adapter_view["rgb_source"]["source_id"]]
            mask_descriptor = self.sources[adapter_view["mask_source"]["source_id"]]
            rgb = _decode_path(rgb_descriptor["path"], mode="RGB", label=f"view {view_id} RGB")
            mask = _decode_path(mask_descriptor["path"], mode="L", label=f"view {view_id} mask")
            expected_source = {
                "rgb": {
                    "name": Path(rgb_descriptor["path"]).name,
                    "sha256": rgb_descriptor["sha256"],
                },
                "mask": {
                    "name": Path(mask_descriptor["path"]).name,
                    "sha256": mask_descriptor["sha256"],
                },
            }
            return rgb, mask, expected_source

        assert self.archive is not None
        rgb_reference = adapter_view["rgb_source"]
        rgb_payload = self.archive.read_bound(rgb_reference, label=f"view {view_id} RGB")
        depth_reference = adapter_view["mask_source"]["depth_source"]
        depth_payload = self.archive.read_bound(depth_reference, label=f"view {view_id} depth")
        mask_payload = derive_registered_depth_mask_png(depth_payload, adapter_view["mask_source"])
        expected_source = {
            "rgb": {"name": f"{view_id}.png", "sha256": rgb_reference["sha256"]},
            "mask": {
                "name": f"mask_{view_id}.png",
                "sha256": hashlib.sha256(mask_payload).hexdigest(),
            },
        }
        return (
            _decode_png(rgb_payload, mode="RGB", label=f"view {view_id} RGB"),
            _decode_png(mask_payload, mode="L", label=f"view {view_id} mask"),
            expected_source,
        )


def _collect_view(
    compact_view: CompactView,
    adapter_view: Mapping[str, Any],
    *,
    sources: _AdapterSources,
    config: PredictorConfig,
) -> dict[str, Any]:
    view_id = adapter_view["id"]
    field = compact_view.observation
    if compact_view.view_id != view_id or not _camera_matches(
        compact_view.camera, adapter_view["camera"]
    ):
        raise ExportError(f"compact view {view_id} source/camera binding differs")
    if compact_view.alpha is None:
        raise ExportError(f"compact view {view_id} has no source-backed alpha")
    rgb, source_mask, expected_source = sources.read_view(adapter_view)
    if compact_view.source != expected_source:
        raise ExportError(f"compact view {view_id} source hash/name binding differs")
    expected_alpha = _derive_alpha_crop(
        source_mask,
        camera=adapter_view["camera"],
        distortion=adapter_view["preprocessing"]["distortion_coefficients"],
        fit_window=field.fit_window,
    )
    alpha = compact_view.alpha.crop_mask()
    if not torch.equal(alpha, expected_alpha):
        raise ExportError(
            f"compact view {view_id} alpha differs from its adapter-bound source mask"
        )

    x0, y0, width, height = field.fit_window
    population = width * height
    foreground_candidates = alpha.reshape(-1).nonzero(as_tuple=True)[0]
    boundary_candidates = (
        _boundary_band(alpha, config.boundary_radius_px).reshape(-1).nonzero(as_tuple=True)[0]
    )
    sample_inputs = {
        "query": population,
        "foreground": foreground_candidates,
        "boundary": boundary_candidates,
    }
    samples: dict[str, dict[str, Any]] = {}
    selected: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for stratum, candidates in sample_inputs.items():
        record, linear, xy = _sample_set(
            candidates,
            cap=config.sample_cap_per_stratum,
            seed=config.seed,
            view_id=view_id,
            stratum=stratum,
            fit_window=(x0, y0, width, height),
            canvas_width=field.width,
            dtype=field.dtype,
        )
        samples[stratum] = record
        selected[stratum] = (linear, xy)

    index = GaussianObservationIndex(
        field,
        tile_size=config.tile_size,
        max_entries=config.max_index_entries,
        max_candidates=config.max_index_candidates,
        max_query_pairs=config.max_query_pairs,
    )

    def evaluate(stratum: str, *, matte: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        linear, xy = selected[stratum]
        with torch.no_grad():
            query = index.query(xy, component_chunk=config.component_chunk)
        if not bool(torch.isfinite(query.color).all()) or not bool(
            torch.isfinite(query.weight_sum).all()
        ):
            raise ExportError(f"compact view {view_id} produced non-finite point-query values")
        target = _sample_source_rgb(
            rgb,
            xy,
            camera=adapter_view["camera"],
            distortion=adapter_view["preprocessing"]["distortion_coefficients"],
        )
        alpha_values = alpha.reshape(-1)[linear]
        if matte:
            target = target * alpha_values[:, None]
        return query.color.detach().cpu(), query.weight_sum.detach().cpu(), target

    query_color, query_weight, query_target = evaluate("query", matte=True)
    foreground_color, _foreground_weight, foreground_target = evaluate("foreground", matte=False)
    boundary_color, _boundary_weight, boundary_target = evaluate("boundary", matte=True)
    query_error = query_color.double() - query_target.double()
    foreground_error = foreground_color.double() - foreground_target.double()
    boundary_error = boundary_color.double() - boundary_target.double()
    query_truth = alpha.reshape(-1)[selected["query"][0]]
    query_support = query_weight > 0.0
    statistics = {
        "query_rgb_absolute_error_sum": float(query_error.abs().sum().item()),
        "query_rgb_value_count": int(query_error.numel()),
        "foreground_rgb_squared_error_sum": float(foreground_error.square().sum().item()),
        "foreground_rgb_value_count": int(foreground_error.numel()),
        "boundary_rgb_absolute_error_sum": float(boundary_error.abs().sum().item()),
        "boundary_rgb_value_count": int(boundary_error.numel()),
        "support_true_positive": int((query_support & query_truth).sum().item()),
        "support_false_positive": int((query_support & ~query_truth).sum().item()),
        "support_false_negative": int((~query_support & query_truth).sum().item()),
        "support_true_negative": int((~query_support & ~query_truth).sum().item()),
    }
    return {
        "id": view_id,
        "ordinal": adapter_view["ordinal"],
        "split": adapter_view["split"],
        "rows": field.n,
        "samples": samples,
        "sufficient_statistics": statistics,
        "metrics": _metrics(statistics, psnr_floor=config.psnr_mse_floor),
    }


def _collect(
    adapter_path: str | Path,
    compact_directory: str | Path,
    *,
    family_id: str,
    config: PredictorConfig,
    requested_predictors: Sequence[str] | None,
) -> dict[str, Any]:
    requested = _requested_predictors(requested_predictors)
    family = _FAMILY_CONTRACTS.get(family_id)
    if family is None:
        raise ExportError(f"unsupported Stage-1 field family {family_id!r}")
    adapter_file = Path(adapter_path).expanduser().resolve(strict=True)
    adapter = load_json_object(adapter_file, label="BENCH-019 source adapter")
    validate_source_adapter(adapter, verify_sources=True)
    dataset = CompactDataset.load(
        compact_directory,
        device="cpu",
        byte_cap=config.view_byte_cap,
        load_alpha=True,
    )
    if len(dataset.views) != len(adapter["views"]):
        raise ExportError("compact field view count differs from its source adapter")
    expected_ids = [view["id"] for view in adapter["views"]]
    if [view.view_id for view in dataset.views] != expected_ids:
        raise ExportError("compact field view order differs from its source adapter")
    if any(
        view.observation.provider != family["provider"]
        or view.observation.blend_mode != family["blend_mode"]
        for view in dataset.views
    ):
        raise ExportError("compact field semantics differ from the declared family equation")

    manifest = describe_artifact(dataset.path / "manifest.json")
    files = [
        {"view_id": view.view_id, "artifact": describe_artifact(view.path)}
        for view in dataset.views
    ]
    complete_field_bytes = manifest["bytes"] + sum(item["artifact"]["bytes"] for item in files)
    with _AdapterSources(adapter) as sources:
        expected_calibration = sources.calibration_sha256()
        if dataset.calibration_sha256 != expected_calibration:
            raise ExportError("compact field calibration digest differs from its source adapter")
        views = [
            _collect_view(
                compact_view,
                adapter_view,
                sources=sources,
                config=config,
            )
            for compact_view, adapter_view in zip(dataset.views, adapter["views"], strict=True)
        ]

    train_views = [view for view in views if view["split"] == "train"]
    heldout_views = [view for view in views if view["split"] == "heldout"]
    aggregates = {
        "train": _aggregate(train_views, psnr_floor=config.psnr_mse_floor),
        "heldout": _aggregate(heldout_views, psnr_floor=config.psnr_mse_floor),
        "all": _aggregate(views, psnr_floor=config.psnr_mse_floor),
    }
    all_predictors = _predictor_values(
        aggregates,
        complete_field_bytes=complete_field_bytes,
    )
    policy = {
        "version": "deterministic_modular_without_replacement_v1",
        **asdict(config),
        "requested_predictors": list(requested),
    }
    result = {
        "schema": PREDICTOR_SCHEMA,
        "state": "complete_development",
        "capture_id": adapter["capture_id"],
        "role": adapter["role"],
        "field_family": dict(family),
        "source_adapter": describe_artifact(adapter_file),
        "compact_field": {
            "root": str(dataset.path),
            "manifest": manifest,
            "view_byte_cap": config.view_byte_cap,
            "complete_field_bytes": complete_field_bytes,
            "files": files,
        },
        "sample_policy": policy,
        "support_definition": SUPPORT_DEFINITION,
        "unsupported_predictors": dict(UNSUPPORTED_PREDICTORS),
        "views": views,
        "aggregates": aggregates,
        "predictors": {name: all_predictors[name] for name in requested},
    }
    result["semantic_digest"] = _canonical_digest(result)
    return result


def _predictor_values(
    aggregates: Mapping[str, Any],
    *,
    complete_field_bytes: int,
) -> dict[str, int | float]:
    train_metrics = aggregates["train"]["metrics"]
    return {
        "train_sampled_foreground_rgb_mse": train_metrics["sampled_foreground_rgb_mse"],
        "train_sampled_foreground_psnr_db": train_metrics["sampled_foreground_psnr_db"],
        "train_sampled_boundary_rgb_mae": train_metrics["sampled_boundary_rgb_mae"],
        "train_sampled_query_rgb_mae": train_metrics["sampled_query_rgb_mae"],
        "train_sampled_support_iou": train_metrics["sampled_support_iou"],
        "train_sampled_support_precision": train_metrics["sampled_support_precision"],
        "train_sampled_support_recall": train_metrics["sampled_support_recall"],
        "all_total_rows": aggregates["all"]["total_rows"],
        "all_complete_field_bytes": complete_field_bytes,
    }


def _nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExportError(f"{label} must be a non-negative integer")
    return value


def _finite_nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExportError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ExportError(f"{label} must be finite and non-negative")
    return result


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExportError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ExportError(f"{label} must be finite")
    return result


def _config_from_policy(value: object) -> tuple[PredictorConfig, tuple[str, ...]]:
    policy = _exact(value, _SAMPLE_POLICY_KEYS, label="predictor sample_policy")
    if policy["version"] != "deterministic_modular_without_replacement_v1":
        raise ExportError("predictor sample policy version differs")
    requested = _requested_predictors(policy["requested_predictors"])
    if list(requested) != policy["requested_predictors"]:
        raise ExportError("predictor request order differs")
    try:
        config = PredictorConfig(**{key: policy[key] for key in asdict(PredictorConfig())})
    except (TypeError, ValueError) as error:
        raise ExportError(f"predictor sample policy is invalid: {error}") from error
    return config, requested


def _validate_statistics(
    value: object,
    *,
    samples: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    statistics = _exact(value, _STAT_KEYS, label=label)
    for key in (
        "query_rgb_absolute_error_sum",
        "foreground_rgb_squared_error_sum",
        "boundary_rgb_absolute_error_sum",
    ):
        _finite_nonnegative(statistics[key], label=f"{label}.{key}")
    for key in _STAT_KEYS - {
        "query_rgb_absolute_error_sum",
        "foreground_rgb_squared_error_sum",
        "boundary_rgb_absolute_error_sum",
    }:
        _nonnegative_integer(statistics[key], label=f"{label}.{key}")
    if statistics["query_rgb_value_count"] != samples["query"]["count"] * 3:
        raise ExportError(f"{label} query value count differs from its sample count")
    if statistics["foreground_rgb_value_count"] != samples["foreground"]["count"] * 3:
        raise ExportError(f"{label} foreground value count differs from its sample count")
    if statistics["boundary_rgb_value_count"] != samples["boundary"]["count"] * 3:
        raise ExportError(f"{label} boundary value count differs from its sample count")
    confusion = sum(
        statistics[key]
        for key in (
            "support_true_positive",
            "support_false_positive",
            "support_false_negative",
            "support_true_negative",
        )
    )
    if confusion != samples["query"]["count"]:
        raise ExportError(f"{label} support confusion differs from its sample count")
    return statistics


def _validate_metrics(
    value: object,
    *,
    statistics: Mapping[str, Any],
    psnr_floor: float,
    label: str,
) -> dict[str, Any]:
    metrics = _exact(value, _METRIC_KEYS, label=label)
    expected = _metrics(statistics, psnr_floor=psnr_floor)
    for key, expected_value in expected.items():
        actual = _finite(metrics[key], label=f"{label}.{key}")
        if key != "sampled_foreground_psnr_db" and actual < 0.0:
            raise ExportError(f"{label}.{key} must be non-negative")
        if key.startswith("sampled_support_") and actual > 1.0:
            raise ExportError(f"{label}.{key} must not exceed one")
        if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-12):
            raise ExportError(f"{label}.{key} differs from its sufficient statistics")
    return metrics


def validate_stage1_predictors(
    value: Mapping[str, Any],
    *,
    verify_files: bool = False,
) -> dict[str, Any]:
    """Validate a predictor artifact; optional file verification performs a full replay."""
    result = _exact(dict(value), _TOP_KEYS, label="BENCH-019 Stage-1 predictors")
    if result["schema"] != PREDICTOR_SCHEMA or result["state"] != "complete_development":
        raise ExportError("Stage-1 predictor schema/state is unsupported")
    if result["role"] != "development":
        raise ExportError("Stage-1 predictors must remain development-only")
    if not isinstance(result["capture_id"], str) or not result["capture_id"]:
        raise ExportError("Stage-1 predictor capture_id is invalid")
    payload = dict(result)
    recorded_digest = _sha256(
        payload.pop("semantic_digest"),
        label="Stage-1 predictor semantic_digest",
    )
    if recorded_digest != _canonical_digest(payload):
        raise ExportError("Stage-1 predictor semantic digest differs")

    family = _exact(result["field_family"], _FAMILY_KEYS, label="Stage-1 field family")
    expected_family = _FAMILY_CONTRACTS.get(family["id"])
    if expected_family is None or family != expected_family:
        raise ExportError("Stage-1 field family equation/provider differs")
    if result["support_definition"] != SUPPORT_DEFINITION:
        raise ExportError("Stage-1 support definition differs")
    if result["unsupported_predictors"] != UNSUPPORTED_PREDICTORS:
        raise ExportError("Stage-1 unavailable-predictor policy differs")
    config, requested = _config_from_policy(result["sample_policy"])

    adapter_artifact = _artifact(
        result["source_adapter"],
        label="Stage-1 source adapter",
        verify_file=verify_files,
    )
    adapter = load_json_object(adapter_artifact["path"], label="BENCH-019 source adapter")
    validate_source_adapter(adapter, verify_sources=verify_files)
    if adapter["capture_id"] != result["capture_id"] or adapter["role"] != result["role"]:
        raise ExportError("Stage-1 predictor identity differs from its source adapter")

    compact = _exact(result["compact_field"], _FIELD_KEYS, label="compact field")
    root_value = compact["root"]
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise ExportError("compact field root must be absolute")
    root = Path(root_value).resolve()
    manifest = _artifact(
        compact["manifest"], label="compact field manifest", verify_file=verify_files
    )
    if Path(manifest["path"]).resolve() != root / "manifest.json":
        raise ExportError("compact field manifest path is not canonical")
    if compact["view_byte_cap"] != config.view_byte_cap:
        raise ExportError("compact field byte cap differs from its sample policy")
    files = compact["files"]
    if not isinstance(files, list) or len(files) != len(adapter["views"]):
        raise ExportError("compact field file inventory is incomplete")
    file_records: list[dict[str, Any]] = []
    for index, (item, adapter_view) in enumerate(zip(files, adapter["views"], strict=True)):
        record = _exact(item, _FIELD_FILE_KEYS, label=f"compact field files[{index}]")
        if record["view_id"] != adapter_view["id"]:
            raise ExportError("compact field file order differs from its adapter")
        artifact = _artifact(
            record["artifact"],
            label=f"compact field file {record['view_id']}",
            verify_file=verify_files,
        )
        if Path(artifact["path"]).resolve() != root / f"{record['view_id']}.rtgsv":
            raise ExportError(f"compact field file {record['view_id']} path is not canonical")
        file_records.append(artifact)
    expected_bytes = manifest["bytes"] + sum(item["bytes"] for item in file_records)
    if compact["complete_field_bytes"] != expected_bytes:
        raise ExportError("complete field bytes differ from the bound manifest and view files")

    views = result["views"]
    if not isinstance(views, list) or len(views) != len(adapter["views"]):
        raise ExportError("Stage-1 predictor view inventory is incomplete")
    validated_views: list[dict[str, Any]] = []
    for index, (item, adapter_view) in enumerate(zip(views, adapter["views"], strict=True)):
        view = _exact(item, _VIEW_KEYS, label=f"predictor views[{index}]")
        if (
            view["id"] != adapter_view["id"]
            or view["ordinal"] != index
            or view["split"] != adapter_view["split"]
        ):
            raise ExportError("Stage-1 predictor view identity/split differs")
        rows = _nonnegative_integer(view["rows"], label=f"predictor view {view['id']} rows")
        if rows <= 0:
            raise ExportError(f"predictor view {view['id']} rows must be positive")
        samples = _exact(view["samples"], _SAMPLES_KEYS, label=f"view {view['id']} samples")
        for stratum, sample_value in samples.items():
            sample = _exact(
                sample_value,
                _SAMPLE_KEYS,
                label=f"view {view['id']} {stratum} samples",
            )
            population = _nonnegative_integer(
                sample["population"], label=f"view {view['id']} {stratum} population"
            )
            count = _nonnegative_integer(
                sample["count"], label=f"view {view['id']} {stratum} count"
            )
            if population <= 0 or count != min(population, config.sample_cap_per_stratum):
                raise ExportError(f"view {view['id']} {stratum} sample counts are invalid")
            _sha256(sample["sha256"], label=f"view {view['id']} {stratum} sample digest")
        statistics = _validate_statistics(
            view["sufficient_statistics"],
            samples=samples,
            label=f"view {view['id']} sufficient statistics",
        )
        _validate_metrics(
            view["metrics"],
            statistics=statistics,
            psnr_floor=config.psnr_mse_floor,
            label=f"view {view['id']} metrics",
        )
        validated_views.append(view)

    aggregates = result["aggregates"]
    if not isinstance(aggregates, dict) or set(aggregates) != {"train", "heldout", "all"}:
        raise ExportError("Stage-1 predictor aggregate splits differ")
    selections = {
        "train": [view for view in validated_views if view["split"] == "train"],
        "heldout": [view for view in validated_views if view["split"] == "heldout"],
        "all": validated_views,
    }
    for split, selected in selections.items():
        aggregate = _exact(aggregates[split], _AGGREGATE_KEYS, label=f"{split} aggregate")
        expected = _aggregate(selected, psnr_floor=config.psnr_mse_floor)
        if aggregate != expected:
            raise ExportError(f"{split} aggregate differs from its per-view sufficient statistics")

    all_predictors = _predictor_values(
        aggregates,
        complete_field_bytes=compact["complete_field_bytes"],
    )
    expected_predictors = {name: all_predictors[name] for name in requested}
    if result["predictors"] != expected_predictors:
        raise ExportError("Stage-1 predictors differ from the requested aggregate values")

    if verify_files:
        expected = _collect(
            adapter_artifact["path"],
            root,
            family_id=family["id"],
            config=config,
            requested_predictors=requested,
        )
        if expected != result:
            raise ExportError("Stage-1 predictor artifact differs from deterministic full replay")
    return {
        "capture_id": result["capture_id"],
        "field_family": family["id"],
        "views": len(views),
        "predictors": len(result["predictors"]),
    }


def build_stage1_predictors(
    adapter_path: str | Path,
    compact_directory: str | Path,
    *,
    family_id: str,
    config: PredictorConfig | None = None,
    requested_predictors: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Collect one complete development-only Stage-1 predictor artifact in memory."""
    result = _collect(
        adapter_path,
        compact_directory,
        family_id=family_id,
        config=config or PredictorConfig(),
        requested_predictors=requested_predictors,
    )
    validate_stage1_predictors(result, verify_files=False)
    return result


def write_stage1_predictors(
    value: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate and exclusively publish one canonical Stage-1 predictor artifact."""
    result = dict(value)
    summary = validate_stage1_predictors(result, verify_files=False)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json(result) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            output.unlink()
        raise
    return {"artifact": describe_artifact(output), **summary}


__all__ = [
    "PREDICTOR_SCHEMA",
    "FIELD_FAMILIES",
    "SUPPORTED_PREDICTORS",
    "SUPPORT_DEFINITION",
    "UNSUPPORTED_PREDICTORS",
    "PredictorConfig",
    "build_stage1_predictors",
    "validate_stage1_predictors",
    "write_stage1_predictors",
]
