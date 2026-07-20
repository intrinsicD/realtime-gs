#!/usr/bin/env python3
"""Exploratory bounds-only crossover for the seven-view compact initializer.

Arm A replays the camera-only fallback bounds.  Arm B injects object bounds derived only from
the exact same seven training masks.  Both arms use the same frozen compact teachers, one
component-center ray for every optimized 2D Gaussian, the same center-sampled occupancy proxy,
48 midpoint depth samples, and a fixed ``N_init^3D=835``.  Optional arm C reuses B's bounds and
keeps every eligible candidate; it is explicitly capacity-confounded.

RGB decoding is denied during candidate construction and lifting.  It is unlocked only after
all arm PLYs and a selection-complete receipt have been committed, for symmetric exact native
gsplat rendering and reporting-only evaluation.  C1004 is absent from every selection input.

This is an exploratory diagnostic, not a sealed or default-changing experiment.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import json
import math
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import (
    GaussianObservationField,
    GaussianObservationIndex,
    ObservationQuery,
)
from rtgs.data.calibrated import _object_bounds, _resize_image, _undistort
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.base import bilinear_sample
from rtgs.lift.compact_carve import (
    CompactCarveConfig,
    CompactCarveInitializer,
    _center_and_extent,
    _propose_anchors,
)

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008"
CALIBRATION = ROOT / "dataset/2025_03_07_stage_with_fabric/calibration_dome.json"
INPUT_BUNDLE = ROOT / "runs/compact_masked_bundle_640_20260717/reconstruction_inputs"
DEFAULT_OUT = ROOT / "runs/compact_bounds_crossover_20260717"

TRAIN_VIEWS = ("C0001", "C0008", "C0014", "C0021", "C0026", "C0031", "C0039")
HELDOUT_VIEW = "C1004"
N_INIT_3D = 835
EXPECTED_COMPONENT_CANDIDATES = 4_480
SAMPLES_PER_RAY = 48
SEED = 75_200
MASK_TARGET_COVERAGE = 0.999
MASK_STRENGTH = -math.log1p(-MASK_TARGET_COVERAGE)
NATIVE_SIZE = (5_328, 4_608)
CONTACT_TILE_WIDTH = 300


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(value))
    os.replace(temporary, path)


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _camera_record(camera: Camera) -> dict[str, Any]:
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


def _calibration_records() -> dict[str, Mapping[str, Any]]:
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    records: dict[str, Mapping[str, Any]] = {}
    for record in payload["cameras"]:
        view_id = str(record["camera_id"]).upper()
        if view_id in records:
            raise RuntimeError(f"duplicate calibration record for {view_id}")
        records[view_id] = record
    return records


def camera_from_calibration(view_id: str) -> tuple[Camera, list[float]]:
    record = _calibration_records()[view_id]
    intrinsics = record["intrinsics"]
    width, height = (int(value) for value in intrinsics["resolution"])
    matrix = intrinsics["camera_matrix"]
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


def validate_selection_views(
    view_names: Sequence[str],
    *,
    expected_train_views: Sequence[str] = TRAIN_VIEWS,
    heldout_view: str = HELDOUT_VIEW,
) -> None:
    """Fail closed before bounds or candidates exist if held-out data could leak."""
    normalized = tuple(str(name).upper() for name in view_names)
    expected = tuple(str(name).upper() for name in expected_train_views)
    heldout = str(heldout_view).upper()
    if heldout in normalized:
        raise ValueError(f"held-out view {heldout} must not occur in selection inputs")
    if heldout in expected:
        raise ValueError(f"held-out view {heldout} must not occur in the training protocol")
    if normalized != expected:
        raise ValueError(
            f"selection views differ from frozen training protocol: {normalized} != {expected}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("selection view names must be unique")


def load_training_masks(
    inputs: ReconstructionInputs,
) -> tuple[list[torch.Tensor], dict[str, dict[str, Any]]]:
    """Acquire only the seven selection masks, in the bound camera order."""
    validate_selection_views(inputs.view_names)
    calibration = _calibration_records()
    masks: list[torch.Tensor] = []
    records: dict[str, dict[str, Any]] = {}
    for view_id, camera in zip(inputs.view_names, inputs.cameras, strict=True):
        path = SCENE / "mask" / f"mask_{view_id}.png"
        distortion = [
            float(value)
            for value in calibration[view_id]["intrinsics"].get(
                "distortion_coefficients",
                [],
            )
        ]
        mask = _undistort(
            _resize_image(path, camera.width, camera.height, mask=True),
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            distortion,
            mask=True,
        ).float()
        masks.append(mask)
        records[view_id] = {
            "path": display_path(path),
            "sha256": sha256_file(path),
            "tensor_sha256": tensor_hash(mask),
            "foreground_fraction": float((mask > 0.5).float().mean()),
            "role": "training_bounds_and_center_proxy_only",
        }
    if HELDOUT_VIEW in records:
        raise RuntimeError("held-out mask entered bounds acquisition")
    return masks, records


def build_bounds_arm_inputs(
    inputs: ReconstructionInputs,
    masks: Sequence[torch.Tensor],
    *,
    heldout_view: str = HELDOUT_VIEW,
) -> dict[str, ReconstructionInputs]:
    """Construct the paired inputs while preserving every value except ``bounds_hint``."""
    validate_selection_views(
        inputs.view_names,
        expected_train_views=tuple(inputs.view_names),
        heldout_view=heldout_view,
    )
    if len(masks) != inputs.n_views:
        raise ValueError("bounds masks must contain exactly one tensor per training view")
    if inputs.bounds_hint is not None:
        raise ValueError("fallback arm requires a source bundle without an injected bounds hint")
    mask_center, mask_extent = _object_bounds(inputs.cameras, list(masks))
    mask_inputs = ReconstructionInputs(
        observations=inputs.observations,
        cameras=inputs.cameras,
        view_names=inputs.view_names,
        points=inputs.points,
        point_visibility=inputs.point_visibility,
        bounds_hint=(mask_center.detach().clone(), float(mask_extent)),
        name=f"{inputs.name}_same_seven_mask_bounds",
    )
    fallback_inputs = ReconstructionInputs(
        observations=inputs.observations,
        cameras=inputs.cameras,
        view_names=inputs.view_names,
        points=inputs.points,
        point_visibility=inputs.point_visibility,
        bounds_hint=None,
        name=f"{inputs.name}_camera_fallback_bounds",
    )
    for left, right in zip(
        fallback_inputs.observations,
        mask_inputs.observations,
        strict=True,
    ):
        if left is not right:
            raise RuntimeError("bounds crossover copied or changed a compact teacher")
    for left, right in zip(fallback_inputs.cameras, mask_inputs.cameras, strict=True):
        if left is not right:
            raise RuntimeError("bounds crossover copied or changed a camera")
    return {"A_fallback_bounds": fallback_inputs, "B_same7_mask_bounds": mask_inputs}


def bounds_record(inputs: ReconstructionInputs, config: CompactCarveConfig) -> dict[str, Any]:
    dtype = inputs.observations[0].dtype
    center, extent = _center_and_extent(inputs, dtype)
    half = float(extent) * config.bounds_scale
    lower = center - half
    upper = center + half
    return {
        "source": "bounds_hint" if inputs.bounds_hint is not None else "camera_fallback",
        "center": center.tolist(),
        "center_sha256": tensor_hash(center),
        "extent": float(extent),
        "bounds_scale": config.bounds_scale,
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "box_side_length": 2.0 * half,
        "box_volume": (2.0 * half) ** 3,
    }


def base_config(*, n_init_3d: int = N_INIT_3D) -> CompactCarveConfig:
    return CompactCarveConfig(
        n_init_3d=n_init_3d,
        candidate_multiplier=4,
        anchor_mode="component_centers",
        max_anchor_candidates=1_000_000,
        samples_per_ray=SAMPLES_PER_RAY,
        query_batch_size=4_096,
        query_component_chunk=256,
        max_query_pairs=1_048_576,
        tile_size=16,
        max_index_entries_per_view=16_000_000,
        max_candidates_per_tile=200_000,
        seed=SEED,
        bounds_scale=0.5,
        near=0.05,
        min_views=2,
        hull_fraction=0.85,
        coverage_scale=1.0,
        coverage_threshold=0.40,
        color_std_sigma=0.20,
        min_score=0.05,
        peak_radius_steps=3.0,
        init_opacity=0.1,
        sh_degree=0,
        max_anchor_rounds=8,
    )


def candidate_contract(
    inputs: ReconstructionInputs,
    config: CompactCarveConfig,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    view_ids, component_ids, xy, attempts, per_view = _propose_anchors(
        inputs,
        config,
        generator,
    )
    expected_xy = torch.cat([field.means for field in inputs.observations])
    expected_components = torch.cat(
        [torch.arange(field.n, dtype=torch.long) for field in inputs.observations]
    )
    expected_views = torch.cat(
        [
            torch.full((field.n,), index, dtype=torch.long)
            for index, field in enumerate(inputs.observations)
        ]
    )
    identity = {
        "xy": torch.equal(xy, expected_xy),
        "component_ids": torch.equal(component_ids, expected_components),
        "view_ids": torch.equal(view_ids, expected_views),
    }
    if not all(identity.values()):
        raise RuntimeError("component-center proposal did not preserve all teacher identities")
    return {
        "candidate_count": int(xy.shape[0]),
        "per_view": per_view,
        "attempt_count": attempts,
        "anchor_mode": config.anchor_mode,
        "samples_per_ray": config.samples_per_ray,
        "xy_sha256": tensor_hash(xy),
        "component_ids_sha256": tensor_hash(component_ids),
        "view_ids_sha256": tensor_hash(view_ids),
        "identity_checks": identity,
    }


def _field_mass(field: GaussianObservationField) -> torch.Tensor:
    return (
        field.amplitudes * (2.0 * math.pi) * field.effective_variances().prod(dim=1).sqrt()
    ).sum()


def build_center_proxies(
    inputs: ReconstructionInputs,
    masks: Sequence[torch.Tensor],
) -> tuple[list[GaussianObservationField], list[dict[str, Any]]]:
    """Copy teacher geometry and replace only amplitudes with mask-at-center scalars."""
    proxies: list[GaussianObservationField] = []
    records: list[dict[str, Any]] = []
    semantics = {
        "schema": "compact_bounds_crossover.center_proxy.v1",
        "mask_sample": "bilinear at each optimized component mean",
        "geometry": "bit-identical aliases of frozen teacher geometry",
        "colors": [1.0, 1.0, 1.0],
    }
    digest = canonical_hash(semantics)
    for view_id, field, mask in zip(
        inputs.view_names,
        inputs.observations,
        masks,
        strict=True,
    ):
        scalar = bilinear_sample(mask, field.means).clamp(0.0, 1.0)
        proxy = GaussianObservationField(
            width=field.width,
            height=field.height,
            means=field.means,
            log_scales=field.log_scales,
            rotations=field.rotations,
            colors=field.colors.new_ones((field.n, 3)),
            amplitudes=scalar,
            color_grads=None,
            filter_variance=field.filter_variance,
            blend_mode="normalized",
            epsilon=field.epsilon,
            sigma_cutoff=field.sigma_cutoff,
            support_fade_alpha=field.support_fade_alpha,
            aa_dilation=field.aa_dilation,
            view_id=view_id,
            fit_window=field.fit_window,
            n_init=field.n_init,
            provider="synthetic_fixture",
            producer_version="same-seven-mask-center-proxy-v1",
            producer_source_digest=canonical_hash(
                {
                    "view_id": view_id,
                    "teacher_means": tensor_hash(field.means),
                    "teacher_scales": tensor_hash(field.log_scales),
                    "teacher_rotations": tensor_hash(field.rotations),
                    "mask": tensor_hash(mask),
                }
            ),
            fit_config_digest=digest,
        )
        proxies.append(proxy)
        records.append(
            {
                "view_id": view_id,
                "components": field.n,
                "positive_center_count": int((scalar > 0.5).sum()),
                "positive_center_fraction": float((scalar > 0.5).float().mean()),
                "amplitude_sum": float(scalar.sum()),
                "amplitudes_sha256": tensor_hash(scalar),
            }
        )
    return proxies, records


class CenterProxyBackend:
    """Preserve exact compact color while replacing only its normalized coverage scalar."""

    def __init__(
        self,
        exact_field: GaussianObservationField,
        exact_index: GaussianObservationIndex,
        proxy_field: GaussianObservationField,
        proxy_index: GaussianObservationIndex,
    ):
        self.exact_field = exact_field
        self.exact_index = exact_index
        self.proxy_field = proxy_field
        self.proxy_index = proxy_index
        self._exact_area = float(exact_field.fit_window[2] * exact_field.fit_window[3])
        self._proxy_area = float(proxy_field.fit_window[2] * proxy_field.fit_window[3])
        self._exact_mass = _field_mass(exact_field)
        self._proxy_mass = _field_mass(proxy_field).clamp_min(torch.finfo(proxy_field.dtype).tiny)

    def query(self, xy: torch.Tensor, component_chunk: int = 4_096) -> ObservationQuery:
        exact = self.exact_index.query(xy, component_chunk=component_chunk)
        proxy = self.proxy_index.query(xy, component_chunk=component_chunk)
        relative_density = self._proxy_area * proxy.weight_sum / self._proxy_mass
        replacement = relative_density * self._exact_mass / self._exact_area
        return ObservationQuery(
            color=exact.color,
            numerator=exact.numerator,
            weight_sum=replacement,
            valid=exact.valid,
        )


def make_center_proxy_backends(
    inputs: ReconstructionInputs,
    proxies: Sequence[GaussianObservationField],
    config: CompactCarveConfig,
) -> list[CenterProxyBackend]:
    backends: list[CenterProxyBackend] = []
    for exact, proxy in zip(inputs.observations, proxies, strict=True):
        exact_index = GaussianObservationIndex(
            exact,
            tile_size=config.tile_size,
            max_entries=config.max_index_entries_per_view,
            max_candidates=config.max_candidates_per_tile,
        )
        proxy_index = GaussianObservationIndex(
            proxy,
            tile_size=config.tile_size,
            max_entries=config.max_index_entries_per_view,
            max_candidates=config.max_candidates_per_tile,
        )
        backends.append(CenterProxyBackend(exact, exact_index, proxy, proxy_index))
    return backends


@contextlib.contextmanager
def deny_rgb_during_selection():
    """Deny PIL decoding under the scene RGB directory during Stage 2."""
    from PIL import Image as PILImage

    original = PILImage.open
    rgb_root = (SCENE / "rgb").resolve()
    counter = {"rgb_open_attempts": 0}

    def guarded(path: Any, *args: Any, **kwargs: Any):
        try:
            Path(path).resolve().relative_to(rgb_root)
        except (TypeError, ValueError):
            return original(path, *args, **kwargs)
        counter["rgb_open_attempts"] += 1
        raise RuntimeError("RGB decoding is forbidden while compact lift arms are selected")

    PILImage.open = guarded
    try:
        yield counter
    finally:
        PILImage.open = original


def lineage_keys(result: Any) -> set[tuple[int, int]]:
    return set(
        zip(
            result.lineage.source_view_indices.tolist(),
            result.lineage.source_component_indices.tolist(),
            strict=True,
        )
    )


def initialization_record(
    result: Any,
    *,
    arm_dir: Path,
    output: Path,
    config: CompactCarveConfig,
    bounds: Mapping[str, Any],
    scientific_label: str,
) -> dict[str, Any]:
    initial_path = arm_dir / "gaussians_init.ply"
    final_path = arm_dir / "gaussians.ply"
    result.gaussians.save_ply(initial_path)
    result.gaussians.save_ply(final_path)
    if initial_path.read_bytes() != final_path.read_bytes():
        raise RuntimeError("unrefined crossover PLY copies differ")
    selected_keys = lineage_keys(result)
    mean_center = result.gaussians.means.mean(dim=0)
    return {
        "status": "PASS",
        "scientific_label": scientific_label,
        "config": dataclasses.asdict(config),
        "bounds": dict(bounds),
        "diagnostics": result.diagnostics,
        "n_gaussians": result.gaussians.n,
        "selected_unique_lineage_count": len(selected_keys),
        "selected_lineage_sha256": canonical_hash(sorted(selected_keys)),
        "means_sha256": tensor_hash(result.gaussians.means),
        "mean_center": mean_center.tolist(),
        "score_quantiles": {
            str(quantile): float(torch.quantile(result.scores, quantile))
            for quantile in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
        "depth_quantiles": {
            str(quantile): float(torch.quantile(result.depths, quantile))
            for quantile in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
        "artifacts": {
            "gaussians_init": initial_path.relative_to(output).as_posix(),
            "gaussians_init_sha256": sha256_file(initial_path),
            "gaussians": final_path.relative_to(output).as_posix(),
            "gaussians_sha256": sha256_file(final_path),
        },
    }


def compare_initializations(left: Any, right: Any) -> dict[str, Any]:
    left_keys = lineage_keys(left)
    right_keys = lineage_keys(right)
    shared = left_keys & right_keys
    left_rows = {
        (int(view), int(component)): index
        for index, (view, component) in enumerate(
            zip(
                left.lineage.source_view_indices,
                left.lineage.source_component_indices,
                strict=True,
            )
        )
    }
    right_rows = {
        (int(view), int(component)): index
        for index, (view, component) in enumerate(
            zip(
                right.lineage.source_view_indices,
                right.lineage.source_component_indices,
                strict=True,
            )
        )
    }
    if shared:
        ordered = sorted(shared)
        left_index = torch.tensor([left_rows[key] for key in ordered])
        right_index = torch.tensor([right_rows[key] for key in ordered])
        mean_delta = (left.gaussians.means[left_index] - right.gaussians.means[right_index]).norm(
            dim=-1
        )
        depth_delta = (left.depths[left_index] - right.depths[right_index]).abs()
        shared_metrics = {
            "mean_distance_mean": float(mean_delta.mean()),
            "mean_distance_median": float(mean_delta.median()),
            "mean_distance_max": float(mean_delta.max()),
            "depth_abs_mean": float(depth_delta.mean()),
            "depth_abs_median": float(depth_delta.median()),
            "depth_abs_max": float(depth_delta.max()),
        }
    else:
        shared_metrics = None
    return {
        "left_count": len(left_keys),
        "right_count": len(right_keys),
        "shared_lineage_count": len(shared),
        "lineage_jaccard": len(shared) / max(len(left_keys | right_keys), 1),
        "shared_lineage_metrics": shared_metrics,
    }


def _save_png(path: Path, color: torch.Tensor) -> dict[str, Any]:
    from PIL import Image as PILImage

    path.parent.mkdir(parents=True, exist_ok=True)
    uint8 = color.detach().clamp(0.0, 1.0).mul(255).round().to(torch.uint8).cpu().numpy()
    PILImage.fromarray(uint8).save(path)
    with PILImage.open(path) as image:
        dimensions = list(image.size)
    return {
        "path": display_path(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "dimensions": dimensions,
        "color_tensor_sha256": tensor_hash(color),
    }


def _mse_record(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    prediction = prediction.detach().cpu()
    target = target.detach().cpu()
    mask = mask.detach().cpu() > 0.5
    if prediction.shape != target.shape or target.shape[:2] != mask.shape:
        raise ValueError("evaluation tensors have inconsistent shapes")
    all_sse = 0.0
    all_count = 0
    foreground_sse = 0.0
    foreground_count = 0
    for start in range(0, target.shape[0], 64):
        difference = prediction[start : start + 64].double() - target[start : start + 64].double()
        active = mask[start : start + 64]
        all_sse += float(difference.square().sum())
        all_count += difference.numel()
        foreground_difference = difference[active]
        foreground_sse += float(foreground_difference.square().sum())
        foreground_count += foreground_difference.numel()
    if foreground_count == 0:
        raise RuntimeError("evaluation mask contains no foreground")
    all_mse = all_sse / all_count
    foreground_mse = foreground_sse / foreground_count
    return {
        "all": {
            "sse": all_sse,
            "scalar_count": all_count,
            "mse": all_mse,
            "psnr_db": -10.0 * math.log10(max(all_mse, 1e-30)),
        },
        "foreground": {
            "sse": foreground_sse,
            "scalar_count": foreground_count,
            "mse": foreground_mse,
            "psnr_db": -10.0 * math.log10(max(foreground_mse, 1e-30)),
        },
    }


def _load_evaluation_target(
    view_id: str,
    camera: Camera,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    image_path = SCENE / "rgb" / f"{view_id}.jpg"
    mask_path = SCENE / "mask" / f"mask_{view_id}.png"
    calibrated_camera, distortion = camera_from_calibration(view_id)
    scalar_delta = max(
        abs(float(getattr(camera, name)) - float(getattr(calibrated_camera, name)))
        for name in ("fx", "fy", "cx", "cy")
    )
    matrix_delta = max(
        float((camera.R - calibrated_camera.R).abs().max()),
        float((camera.t - calibrated_camera.t).abs().max()),
    )
    if scalar_delta > 1e-4 or matrix_delta > 1e-5:
        raise RuntimeError(f"evaluation camera {view_id} differs from calibration")
    image = _undistort(
        _resize_image(image_path, camera.width, camera.height),
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        distortion,
    ).float()
    mask = _undistort(
        _resize_image(mask_path, camera.width, camera.height, mask=True),
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        distortion,
        mask=True,
    ).float()
    return (
        image,
        mask,
        {
            "view_id": view_id,
            "role": "reporting_only_after_selection_complete",
            "calibration": {
                "path": display_path(CALIBRATION),
                "sha256": sha256_file(CALIBRATION),
            },
            "distortion_coefficients": distortion,
            "rgb": {"path": display_path(image_path), "sha256": sha256_file(image_path)},
            "mask": {"path": display_path(mask_path), "sha256": sha256_file(mask_path)},
            "decoded_rgb_tensor_sha256": tensor_hash(image),
            "decoded_mask_tensor_sha256": tensor_hash(mask),
            "camera": _camera_record(camera),
            "camera_scalar_max_abs_vs_calibration": scalar_delta,
            "camera_matrix_max_abs_vs_calibration": matrix_delta,
            "dimensions": [camera.width, camera.height],
        },
    )


def exact_native_evaluation(
    *,
    output: Path,
    arm_models: Mapping[str, Gaussians3D],
    train_inputs: ReconstructionInputs,
    selection_receipt_sha256: str,
    evaluation_targets: Mapping[
        str,
        tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]],
    ]
    | None = None,
) -> dict[str, Any]:
    """Symmetrically render all frozen arms; never feed metrics back into selection."""
    from rtgs.viewer import render_exact_snapshot

    validate_selection_views(train_inputs.view_names)
    heldout_camera, _ = camera_from_calibration(HELDOUT_VIEW)
    if (heldout_camera.width, heldout_camera.height) != NATIVE_SIZE:
        raise RuntimeError("held-out camera is not native resolution")
    views = [
        *zip(train_inputs.view_names, train_inputs.cameras, ["train"] * train_inputs.n_views),
        (HELDOUT_VIEW, heldout_camera, "heldout"),
    ]
    if any(view_id == HELDOUT_VIEW and role != "heldout" for view_id, _, role in views):
        raise RuntimeError("held-out view was mislabeled as training")
    if evaluation_targets is not None and set(evaluation_targets) != {
        view_id for view_id, _, _ in views
    }:
        raise ValueError("predecoded evaluation targets do not match the exact view protocol")
    records: list[dict[str, Any]] = []
    thumbnails: dict[tuple[str, str], Any] = {}
    for view_id, camera, role in views:
        if (camera.width, camera.height) != NATIVE_SIZE:
            raise RuntimeError(f"{view_id} is not native full resolution")
        if evaluation_targets is None:
            target, mask, target_record = _load_evaluation_target(view_id, camera)
        else:
            target, mask, source_record = evaluation_targets[view_id]
            target_record = dict(source_record)
            if (
                target_record.get("view_id") != view_id
                or tensor_hash(target) != target_record.get("decoded_rgb_tensor_sha256")
                or tensor_hash(mask) != target_record.get("decoded_mask_tensor_sha256")
            ):
                raise RuntimeError(f"predecoded evaluation target binding failed for {view_id}")
        thumbnails[(view_id, "target")] = _thumbnail(target)
        model_records: dict[str, Any] = {}
        for arm_name, model in arm_models.items():
            snapshot = render_exact_snapshot(
                model,
                camera,
                device="cuda",
                rasterizer="gsplat",
                packed=False,
                antialiased=False,
            )
            if (
                snapshot.backend != "rtgs.render.gsplat_backend.GsplatRasterizer"
                or not snapshot.device.startswith("cuda")
                or tuple(snapshot.color.shape[:2]) != (camera.height, camera.width)
            ):
                raise RuntimeError("exact native snapshot resolved the wrong execution path")
            color = snapshot.color.detach().cpu()
            render_path = output / "exact_native_gsplat" / arm_name / f"{view_id}.png"
            artifact = _save_png(render_path, color)
            metrics = _mse_record(color, target, mask)
            thumbnails[(view_id, arm_name)] = _thumbnail(color)
            model_records[arm_name] = {
                "n_gaussians": model.n,
                "backend": snapshot.backend,
                "device": snapshot.device,
                "packed": False,
                "antialiased": False,
                "render": artifact,
                "metrics": metrics,
            }
            del snapshot, color
            torch.cuda.empty_cache()
        records.append(
            {
                "view_id": view_id,
                "split": role,
                "target": target_record,
                "models": model_records,
            }
        )
        del target, mask
        gc.collect()
    contact = build_contact_sheet(
        output / "CONTACT_SHEET.png",
        view_names=[view_id for view_id, _, _ in views],
        arm_names=list(arm_models),
        thumbnails=thumbnails,
    )
    pooled: dict[str, dict[str, dict[str, float]]] = {}
    for arm_name in arm_models:
        pooled[arm_name] = {}
        for split in ("train", "heldout"):
            selected = [record for record in records if record["split"] == split]
            pooled[arm_name][split] = {}
            for stratum in ("all", "foreground"):
                sse = sum(
                    record["models"][arm_name]["metrics"][stratum]["sse"] for record in selected
                )
                count = sum(
                    record["models"][arm_name]["metrics"][stratum]["scalar_count"]
                    for record in selected
                )
                mse = sse / count
                pooled[arm_name][split][stratum] = {
                    "mse": mse,
                    "psnr_db": -10.0 * math.log10(max(mse, 1e-30)),
                }
    return {
        "artifact_type": "compact_bounds_crossover_reporting_evaluation_v1",
        "selection_receipt_sha256": selection_receipt_sha256,
        "selection_locked_before_any_rgb_decode": True,
        "metrics_used_for_selection": False,
        "heldout_view": HELDOUT_VIEW,
        "heldout_present_in_selection_inputs": False,
        "native_resolution": list(NATIVE_SIZE),
        "records": records,
        "pooled": pooled,
        "contact_sheet": contact,
    }


def _thumbnail(color: torch.Tensor):
    from PIL import Image as PILImage

    uint8 = color.detach().clamp(0.0, 1.0).mul(255).round().to(torch.uint8).cpu().numpy()
    image = PILImage.fromarray(uint8)
    height = max(1, round(CONTACT_TILE_WIDTH * image.height / image.width))
    return image.resize((CONTACT_TILE_WIDTH, height), PILImage.Resampling.LANCZOS)


def build_contact_sheet(
    path: Path,
    *,
    view_names: Sequence[str],
    arm_names: Sequence[str],
    thumbnails: Mapping[tuple[str, str], Any],
) -> dict[str, Any]:
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont

    row_names = ("target", *arm_names)
    sample = thumbnails[(view_names[0], "target")]
    label_width = 190
    header_height = 40
    row_height = sample.height + 10
    canvas = PILImage.new(
        "RGB",
        (
            label_width + CONTACT_TILE_WIDTH * len(view_names),
            header_height + row_height * len(row_names),
        ),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, view_id in enumerate(view_names):
        draw.text(
            (label_width + column * CONTACT_TILE_WIDTH + 8, 12),
            view_id,
            fill=(20, 20, 20),
            font=font,
        )
    for row, row_name in enumerate(row_names):
        y = header_height + row * row_height
        draw.text((8, y + 8), row_name, fill=(20, 20, 20), font=font)
        for column, view_id in enumerate(view_names):
            image = thumbnails[(view_id, row_name)]
            canvas.paste(image, (label_width + column * CONTACT_TILE_WIDTH, y))
    canvas.save(path)
    return {
        "path": display_path(path),
        "sha256": sha256_file(path),
        "dimensions": list(canvas.size),
        "rows": list(row_names),
        "columns": list(view_names),
    }


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def viewer_commands(
    output: Path,
    *,
    include_all_eligible: bool,
    port: int,
) -> dict[str, list[str]]:
    common = [
        ".venv/bin/rtgs",
        "view",
        "--scene",
        display_path(SCENE),
        "--downscale",
        "1",
        "--max-images",
        "8",
        "--rasterizer",
        "gsplat",
        "--device",
        "cuda",
        "--snapshot-dir",
        display_path(output / "viewer_snapshots"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-open",
    ]
    fixed = [
        common[0],
        common[1],
        "--gaussians",
        display_path(output / "arms/B_same7_mask_bounds/gaussians.ply"),
        "--initial",
        display_path(output / "arms/A_fallback_bounds/gaussians_init.ply"),
        *common[2:],
    ]
    commands = {"fixed_budget_A_vs_B": fixed}
    if include_all_eligible:
        commands["B_fixed_vs_C_all_eligible"] = [
            common[0],
            common[1],
            "--gaussians",
            display_path(output / "arms/C_same7_mask_bounds_all_eligible/gaussians.ply"),
            "--initial",
            display_path(output / "arms/B_same7_mask_bounds/gaussians_init.ply"),
            *common[2:],
        ]
    return commands


def smoke_viewer(command: Sequence[str], *, timeout: float = 180.0) -> dict[str, Any]:
    command = list(command)
    port = int(command[command.index("--port") + 1])
    env = dict(os.environ)
    env.setdefault("LD_PRELOAD", "/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    url = f"http://127.0.0.1:{port}"
    connected = False
    status = None
    body = b""
    started = time.monotonic()
    try:
        while time.monotonic() - started < timeout:
            if process.poll() is not None:
                break
            try:
                with urllib.request.urlopen(url, timeout=1.0) as response:
                    status = int(response.status)
                    body = response.read(1 << 20)
                lowered = body.lower()
                if status == 200 and (b"viser" in lowered or b"realtime-gs" in lowered):
                    connected = True
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(0.5)
    finally:
        process.terminate()
        try:
            output, _ = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate(timeout=30)
    return {
        "status": "PASS" if connected else "FAIL",
        "command": command,
        "url": url,
        "connected": connected,
        "http_status": status,
        "http_body_sha256": hashlib.sha256(body).hexdigest() if body else None,
        "stdout_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "returncode_after_termination": process.returncode,
    }


def source_binding() -> dict[str, Any]:
    paths = (
        Path("benchmarks/compact_bounds_crossover.py"),
        Path("src/rtgs/core/camera.py"),
        Path("src/rtgs/core/gaussians3d.py"),
        Path("src/rtgs/core/observation2d.py"),
        Path("src/rtgs/data/calibrated.py"),
        Path("src/rtgs/data/reconstruction_inputs.py"),
        Path("src/rtgs/lift/base.py"),
        Path("src/rtgs/lift/compact_carve.py"),
        Path("src/rtgs/render/gsplat_backend.py"),
        Path("src/rtgs/viewer.py"),
    )
    hashes = {path.as_posix(): sha256_file(ROOT / path) for path in paths}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {
        "git_revision": revision,
        "files": hashes,
        "aggregate": canonical_hash(hashes),
    }


def bundle_binding(directory: Path) -> dict[str, Any]:
    manifest = directory / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    teachers: dict[str, Any] = {}
    for record in payload["views"]:
        teacher = directory / record["teacher"]
        digest = sha256_file(teacher)
        if digest != record["teacher_sha256"]:
            raise RuntimeError(f"teacher binding mismatch for {record['view_id']}")
        teachers[record["view_id"]] = {
            "path": display_path(teacher),
            "sha256": digest,
            "components": record["n_opt_2d"],
        }
    return {
        "directory": display_path(directory),
        "manifest": display_path(manifest),
        "manifest_sha256": sha256_file(manifest),
        "semantic_digest": payload["semantic_digest"],
        "calibration_digest": payload["calibration_digest"],
        "teachers": teachers,
    }


def run(
    output: Path,
    *,
    input_bundle: Path = INPUT_BUNDLE,
    include_all_eligible: bool = True,
    render: bool = True,
    viewer_smoke: bool = True,
    viewer_port: int | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite crossover namespace: {output}")
    output.mkdir(parents=True)
    (output / "arms").mkdir()
    source_before = source_binding()
    bundle_before = bundle_binding(input_bundle)
    inputs = ReconstructionInputs.load(input_bundle, device="cpu", strict=True)
    validate_selection_views(inputs.view_names)
    if sum(field.n for field in inputs.observations) != EXPECTED_COMPONENT_CANDIDATES:
        raise RuntimeError("frozen compact bundle no longer contains exactly 4,480 components")
    masks, mask_bindings = load_training_masks(inputs)
    arms = build_bounds_arm_inputs(inputs, masks)
    config = base_config()
    candidate = candidate_contract(inputs, config)
    if (
        candidate["candidate_count"] != EXPECTED_COMPONENT_CANDIDATES
        or candidate["per_view"] != [640] * 7
        or candidate["samples_per_ray"] != SAMPLES_PER_RAY
    ):
        raise RuntimeError("candidate crossover contract differs from 4,480 x 48")
    proxies, proxy_records = build_center_proxies(inputs, masks)
    backends = make_center_proxy_backends(inputs, proxies, config)
    bounds = {name: bounds_record(arm_inputs, config) for name, arm_inputs in arms.items()}
    center_delta = torch.tensor(bounds["B_same7_mask_bounds"]["center"]) - torch.tensor(
        bounds["A_fallback_bounds"]["center"]
    )
    paired_bounds = {
        "center_shift_euclidean": float(center_delta.norm()),
        "B_extent_over_A_extent": (
            bounds["B_same7_mask_bounds"]["extent"] / bounds["A_fallback_bounds"]["extent"]
        ),
        "B_box_volume_over_A_box_volume": (
            bounds["B_same7_mask_bounds"]["box_volume"] / bounds["A_fallback_bounds"]["box_volume"]
        ),
    }
    plan = {
        "artifact_type": "compact_bounds_crossover_plan_v1",
        "decision_bearing": False,
        "scope": "exploratory calibrated bounds-only crossover",
        "source_binding": source_before,
        "input_bundle": bundle_before,
        "selection_views": list(inputs.view_names),
        "heldout_view": HELDOUT_VIEW,
        "heldout_excluded_from_selection": HELDOUT_VIEW not in inputs.view_names,
        "mask_bindings": mask_bindings,
        "center_proxy": {
            "records": proxy_records,
            "mask_target_coverage": MASK_TARGET_COVERAGE,
            "mask_strength": MASK_STRENGTH,
        },
        "shared_configuration": dataclasses.asdict(config),
        "candidate_contract": candidate,
        "bounds": bounds,
        "paired_bounds": paired_bounds,
        "arms": {
            "A_fallback_bounds": {
                "bounds": "camera-only compact_carve fallback",
                "capacity": N_INIT_3D,
                "causal_role": "control",
            },
            "B_same7_mask_bounds": {
                "bounds": "derived only from the exact seven training masks",
                "capacity": N_INIT_3D,
                "causal_role": "bounds-only intervention",
            },
            "C_same7_mask_bounds_all_eligible": {
                "enabled": include_all_eligible,
                "bounds": "identical to B",
                "capacity": "all eligible candidates; determined only after B lift",
                "causal_role": "capacity-confounded diagnostic; never interpret as bounds-only",
            },
        },
        "evaluation": {
            "enabled": render,
            "occurs_after_selection_receipt": True,
            "train_views": list(inputs.view_names),
            "heldout_view": HELDOUT_VIEW,
            "heldout_metrics_used_for_selection": False,
            "renderer": "rtgs gsplat exact native 5328x4608",
        },
    }
    plan["plan_sha256"] = canonical_hash(plan)
    write_json(output / "plan.json", plan)

    results: dict[str, Any] = {}
    objects: dict[str, Any] = {}
    with deny_rgb_during_selection() as denial:
        for arm_name in ("A_fallback_bounds", "B_same7_mask_bounds"):
            arm_dir = output / "arms" / arm_name
            arm_dir.mkdir()
            started = time.perf_counter()
            initialization = CompactCarveInitializer(config).initialize(
                arms[arm_name],
                backends=backends,
            )
            record = initialization_record(
                initialization,
                arm_dir=arm_dir,
                output=output,
                config=config,
                bounds=bounds[arm_name],
                scientific_label=(
                    "control: camera-only fallback bounds"
                    if arm_name.startswith("A_")
                    else "causal arm: same-seven-training-mask bounds only"
                ),
            )
            record["wall_seconds"] = time.perf_counter() - started
            results[arm_name] = record
            objects[arm_name] = initialization
            write_json(arm_dir / "result.json", record)
        eligible = int(results["B_same7_mask_bounds"]["diagnostics"]["eligible_candidate_count"])
        if include_all_eligible:
            arm_name = "C_same7_mask_bounds_all_eligible"
            arm_dir = output / "arms" / arm_name
            arm_dir.mkdir()
            all_config = dataclasses.replace(config, n_init_3d=eligible)
            started = time.perf_counter()
            initialization = CompactCarveInitializer(all_config).initialize(
                arms["B_same7_mask_bounds"],
                backends=backends,
            )
            record = initialization_record(
                initialization,
                arm_dir=arm_dir,
                output=output,
                config=all_config,
                bounds=bounds["B_same7_mask_bounds"],
                scientific_label=(
                    "capacity-confounded diagnostic: B bounds with every eligible placement"
                ),
            )
            record["wall_seconds"] = time.perf_counter() - started
            record["capacity_confounded"] = True
            results[arm_name] = record
            objects[arm_name] = initialization
            write_json(arm_dir / "result.json", record)
    if denial["rgb_open_attempts"]:
        raise RuntimeError("selection attempted forbidden RGB access")
    if bundle_binding(input_bundle) != bundle_before:
        raise RuntimeError("frozen compact bundle changed during selection")
    if source_binding() != source_before:
        raise RuntimeError("bound crossover source changed during selection")
    for view_id, record in mask_bindings.items():
        if view_id == HELDOUT_VIEW or sha256_file(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError("training mask binding changed or held-out mask leaked")

    paired = {
        "A_vs_B_fixed_budget": compare_initializations(
            objects["A_fallback_bounds"],
            objects["B_same7_mask_bounds"],
        )
    }
    if include_all_eligible:
        paired["B_fixed_vs_C_all_eligible"] = compare_initializations(
            objects["B_same7_mask_bounds"],
            objects["C_same7_mask_bounds_all_eligible"],
        )
        if not lineage_keys(objects["B_same7_mask_bounds"]).issubset(
            lineage_keys(objects["C_same7_mask_bounds_all_eligible"])
        ):
            raise RuntimeError("fixed-budget B is not a subset of all-eligible C")
    selection_receipt = {
        "artifact_type": "compact_bounds_crossover_selection_complete_v1",
        "plan_sha256": plan["plan_sha256"],
        "source_rgb_denial": denial,
        "heldout_view": HELDOUT_VIEW,
        "heldout_excluded": HELDOUT_VIEW not in inputs.view_names,
        "candidate_contract": candidate,
        "arms": results,
        "paired_diagnostics": paired,
    }
    selection_receipt["selection_receipt_sha256"] = canonical_hash(selection_receipt)
    write_json(output / "selection_complete.json", selection_receipt)

    evaluation = None
    if render:
        arm_models = {
            arm_name: initialization.gaussians for arm_name, initialization in objects.items()
        }
        evaluation = exact_native_evaluation(
            output=output,
            arm_models=arm_models,
            train_inputs=inputs,
            selection_receipt_sha256=selection_receipt["selection_receipt_sha256"],
        )
        write_json(output / "evaluation.json", evaluation)

    port = _free_local_port() if viewer_port is None else viewer_port
    commands = viewer_commands(
        output,
        include_all_eligible=include_all_eligible,
        port=port,
    )
    receipt = {
        "status": "DEFERRED",
        "commands": commands,
        "reason": "viewer smoke disabled by caller",
    }
    if viewer_smoke:
        smoke = smoke_viewer(commands["fixed_budget_A_vs_B"])
        receipt = {
            **smoke,
            "commands": commands,
            "smoked_comparison": "fixed_budget_A_vs_B",
        }
        if smoke["status"] != "PASS":
            raise RuntimeError(f"viewer smoke failed: {smoke}")
    write_json(output / "viewer_receipt.json", receipt)

    result = {
        "artifact_type": "compact_bounds_crossover_result_v1",
        "status": "PASS",
        "decision_bearing": False,
        "plan_sha256": plan["plan_sha256"],
        "selection_receipt_sha256": selection_receipt["selection_receipt_sha256"],
        "source_rgb_denial": denial,
        "heldout_exclusion": {
            "view_id": HELDOUT_VIEW,
            "present_in_selection_inputs": False,
            "present_in_mask_bounds": False,
            "used_for_selection": False,
            "evaluation_only_after_selection_complete": render,
        },
        "paired_bounds": paired_bounds,
        "candidate_contract": candidate,
        "arms": results,
        "paired_diagnostics": paired,
        "evaluation": evaluation,
        "viewer": receipt,
    }
    result["result_sha256"] = canonical_hash(result)
    write_json(output / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--input-bundle", type=Path, default=INPUT_BUNDLE)
    parser.add_argument(
        "--include-all-eligible",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--viewer-smoke",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--viewer-port", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        args.out.resolve(),
        input_bundle=args.input_bundle.resolve(),
        include_all_eligible=args.include_all_eligible,
        render=args.render,
        viewer_smoke=args.viewer_smoke,
        viewer_port=args.viewer_port,
    )
    summary = {
        "status": result["status"],
        "output": str(args.out.resolve()),
        "paired_bounds": result["paired_bounds"],
        "candidate_contract": result["candidate_contract"],
        "arms": {
            name: {
                "n_gaussians": record["n_gaussians"],
                "eligible": record["diagnostics"]["eligible_candidate_count"],
                "score_median": record["score_quantiles"]["0.5"],
            }
            for name, record in result["arms"].items()
        },
        "paired_diagnostics": result["paired_diagnostics"],
        "pooled_evaluation": (
            None if result["evaluation"] is None else result["evaluation"]["pooled"]
        ),
        "contact_sheet": (
            None if result["evaluation"] is None else result["evaluation"]["contact_sheet"]
        ),
        "viewer": result["viewer"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
