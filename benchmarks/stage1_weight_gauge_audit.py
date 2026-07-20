#!/usr/bin/env python3
"""Preregistered stage-1 weight/color gauge representation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.scene import SceneData
from rtgs.data.synthetic import make_synthetic_scene
from rtgs.depth.mock import GroundTruthDepth
from rtgs.image2gs.fit import FitConfig, fit_views
from rtgs.image2gs.renderer2d import render_gaussian_coverage_2d, render_gaussians_2d
from rtgs.lift.base import bilinear_sample, lift_view_at_depth, lift_view_from_depth_map
from rtgs.lift.carve import CarveLifter, _ray_box
from rtgs.lift.depth import DepthLifter
from rtgs.render.torch_ref import TorchRasterizer

ROOT = Path(__file__).resolve().parent.parent
PREREGISTRATION = Path("benchmarks/results/20260716_stage1_weight_gauge_PREREG.md")
PREREGISTRATION_SHA256 = "ec2bdaea7362649392da915af2d44e7aa47a8a1825546f8487f6afa3067b9489"
DEFAULT_SEAL = Path("benchmarks/results/20260716_stage1_weight_gauge_SEAL.json")
ATTEMPT = ROOT / "benchmarks/results/20260716_stage1_weight_gauge_AUDIT_ATTEMPT.json"
SEEDS = (0, 1, 2)
TRAIN_INDICES = (0, 1, 2, 4, 5, 6, 8, 9, 10)
HELD_OUT_INDICES = (3, 7, 11)
GAUGES = ("identity", "unit_weight", "peak_color")
TRANSFORMS = ("unit_weight", "peak_color")
WEIGHT_BINS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)
IMAGE_SIZE = 48
COMPONENTS_PER_VIEW = 150
SOURCE_RENDER_MAX_ABS = 5e-6
SOURCE_RENDER_REL_L1 = 1e-6
SOURCE_RENDER_MIN_PSNR = 100.0
TRANSFORM_MAX_ABS = 1e-7
TRANSFORM_MAX_REL = 1e-6
CHANGE_EPSILON = 1e-7
SEALED_PATHS = tuple(
    sorted(
        {
            PREREGISTRATION,
            Path("benchmarks/stage1_weight_gauge_audit.py"),
            Path("pyproject.toml"),
            *(path.relative_to(ROOT) for path in (ROOT / "src" / "rtgs").rglob("*.py")),
            *(path.relative_to(ROOT) for path in (ROOT / "tests").rglob("*.py")),
        },
        key=str,
    )
)


class AuditInvalid(RuntimeError):
    """Expected fail-closed invalidation carrying only evidence reached so far."""

    def __init__(self, stage: str, reason: str, evidence: dict[str, Any] | None = None):
        super().__init__(reason)
        self.stage = stage
        self.reason = reason
        self.evidence = evidence or {}


@dataclass
class PreparedSeed:
    """Runtime-only tensors plus their serializable provenance."""

    seed: int
    scene: SceneData
    fitted: list[Gaussians2D]
    gauges: dict[str, list[Gaussians2D]]
    preparation: dict[str, Any]
    transformations: dict[str, Any]


@dataclass
class LiftRuntime:
    """Runtime output and serialized evidence for one lift."""

    output: Gaussians3D
    keys: list[tuple[int, int, int]]
    renders: list[dict[str, torch.Tensor]]
    evidence: dict[str, Any]
    source_records: dict[tuple[int, int, int], dict[str, Any]] | None = None


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_hash(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def tensor_collection_hash(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def tensor_hash(name: str, value: torch.Tensor) -> str:
    return tensor_collection_hash([(name, value)])


def gaussians2d_hash(views: list[Gaussians2D]) -> str:
    return tensor_collection_hash(
        (f"view_{view}/{field}", getattr(gaussians, field))
        for view, gaussians in enumerate(views)
        for field in ("xy", "chol", "color", "weight")
    )


def gaussians3d_hash(gaussians: Gaussians3D) -> str:
    return tensor_collection_hash(
        (field, getattr(gaussians, field))
        for field in ("means", "quats", "log_scales", "opacity", "sh")
    )


def ordered_float64_sum(values: Iterable[float]) -> float:
    """Use one explicit, reviewable order for every serialized raw-sum pool."""
    total = 0.0
    for value in values:
        total = total + float(value)
    return total


def assert_finite_tree(value: Any, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"{context} contains a non-finite float")
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_tree(item, f"{context}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_tree(item, f"{context}/{index}")


def fit_config() -> FitConfig:
    return FitConfig(
        n_gaussians=150,
        max_gaussians=5_000,
        iterations=120,
        backend="native",
        adaptive_density=True,
        growth_waves=5,
        relocate_fraction=0.0,
        structsplat_renderer="auto",
        lr=0.01,
        grad_init_mix=0.7,
        row_chunk=64,
        log_every=50,
        convergence_patience=0,
        convergence_tol=0.05,
        convergence_check_every=25,
    )


def depth_kwargs() -> dict[str, Any]:
    return {
        "sh_degree": 0,
        "min_weight": 0.05,
        "init_opacity": 0.1,
        "normal_thickness": 0.15,
        "covariance_mode": "surface",
        "isotropic_sigma": None,
        "robust_depth_gradients": True,
        "merge": False,
        "merge_voxel_frac": 0.01,
    }


def carve_kwargs() -> dict[str, Any]:
    return {
        "grid_res": 48,
        "bounds_scale": 0.5,
        "min_views": 2,
        "hull_fraction": 0.85,
        "color_std_sigma": 0.20,
        "color_match_sigma": 0.35,
        "coverage_thresh": 0.40,
        "samples_per_ray": 64,
        "min_score": 0.05,
        "min_weight": 0.05,
        "merge": False,
        "merge_voxel_scale": 1.0,
        "init_opacity": 0.1,
        "sh_degree": 0,
    }


def source_key(seed: int, local_view: int, component: int) -> tuple[int, int, int]:
    return seed, local_view, component


def serialize_keys(keys: Iterable[tuple[int, int, int]]) -> list[list[int]]:
    return [list(key) for key in keys]


def camera_hashes(scene: SceneData) -> dict[str, Any]:
    scalars = [
        {
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy,
            "width": camera.width,
            "height": camera.height,
        }
        for camera in scene.cameras
    ]
    tensors = [
        (f"{index}/{field}", getattr(camera, field))
        for index, camera in enumerate(scene.cameras)
        for field in ("R", "t")
    ]
    return {
        "scalars": canonical_json_hash(scalars),
        "tensors": tensor_collection_hash(tensors),
        "per_view": [
            {
                "local_train_view_index": index,
                "scalar_sha256": canonical_json_hash(scalars[index]),
                "rotation_sha256": tensor_hash("R", camera.R),
                "translation_sha256": tensor_hash("t", camera.t),
            }
            for index, camera in enumerate(scene.cameras)
        ],
    }


def validate_fitted_views(views: list[Gaussians2D]) -> None:
    if len(views) != len(TRAIN_INDICES):
        raise AuditInvalid("preparation", "native fit did not return exactly nine views")
    for view_index, gaussian in enumerate(views):
        if gaussian.n != COMPONENTS_PER_VIEW:
            raise AuditInvalid(
                "preparation",
                f"view {view_index} returned {gaussian.n} rather than 150 components",
            )
        expected_shapes = {
            "xy": (COMPONENTS_PER_VIEW, 2),
            "chol": (COMPONENTS_PER_VIEW, 3),
            "color": (COMPONENTS_PER_VIEW, 3),
            "weight": (COMPONENTS_PER_VIEW,),
        }
        for field, expected_shape in expected_shapes.items():
            if tuple(getattr(gaussian, field).shape) != expected_shape:
                raise AuditInvalid(
                    "preparation",
                    f"view {view_index}/{field} shape differs from {expected_shape}",
                )
        for field in ("xy", "chol", "color", "weight"):
            if not bool(torch.isfinite(getattr(gaussian, field)).all()):
                raise AuditInvalid("preparation", f"view {view_index}/{field} is non-finite")
        if not bool((gaussian.chol[:, (0, 2)] > 0).all()):
            raise AuditInvalid("preparation", f"view {view_index} has non-positive diagonal")
        if not bool(((gaussian.color >= 0) & (gaussian.color <= 1)).all()):
            raise AuditInvalid("preparation", f"view {view_index} color is out of range")
        if not bool(((gaussian.weight >= 0) & (gaussian.weight <= 1)).all()):
            raise AuditInvalid("preparation", f"view {view_index} weight is out of range")
        x_valid = (gaussian.xy[:, 0] >= 0) & (gaussian.xy[:, 0] < IMAGE_SIZE)
        y_valid = (gaussian.xy[:, 1] >= 0) & (gaussian.xy[:, 1] < IMAGE_SIZE)
        if not bool((x_valid & y_valid).all()):
            raise AuditInvalid("preparation", f"view {view_index} center is out of bounds")


def fixed_bin_counts(values: torch.Tensor) -> list[int]:
    flat = values.detach().reshape(-1)
    counts = []
    for index, (lower, upper) in enumerate(zip(WEIGHT_BINS[:-1], WEIGHT_BINS[1:])):
        inside = (flat >= lower) & (
            (flat <= upper) if index == len(WEIGHT_BINS) - 2 else (flat < upper)
        )
        counts.append(int(inside.sum()))
    if sum(counts) != flat.numel():
        raise AuditInvalid("transform", "fixed bins did not partition all bounded values")
    return counts


def _amplitude_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    absolute = (actual - expected).abs()
    maximum_absolute = float(absolute.max()) if absolute.numel() else 0.0
    relative_mask = expected.abs() > 1e-8
    maximum_relative = (
        float((absolute[relative_mask] / expected[relative_mask].abs()).max())
        if bool(relative_mask.any())
        else 0.0
    )
    return maximum_absolute, maximum_relative


def construct_gauges(
    fitted: list[Gaussians2D], seed: int
) -> tuple[dict[str, list[Gaussians2D]], dict[str, Any]]:
    gauges: dict[str, list[Gaussians2D]] = {name: [] for name in GAUGES}
    per_view: dict[str, list[dict[str, Any]]] = {name: [] for name in GAUGES}
    for view_index, source in enumerate(fitted):
        amplitude = source.weight[:, None] * source.color
        peak = amplitude.max(dim=-1).values
        identity = source.detach()
        unit = Gaussians2D(
            xy=source.xy.detach().clone(),
            chol=source.chol.detach().clone(),
            color=amplitude.detach().clone(),
            weight=torch.ones_like(source.weight),
        )
        positive = peak > 0
        peak_color = torch.zeros_like(source.color)
        peak_color[positive] = amplitude[positive] / peak[positive, None]
        peak_gauge = Gaussians2D(
            xy=source.xy.detach().clone(),
            chol=source.chol.detach().clone(),
            color=peak_color,
            weight=torch.where(positive, peak, torch.zeros_like(peak)),
        )
        built = {"identity": identity, "unit_weight": unit, "peak_color": peak_gauge}
        for name, gauge in built.items():
            if not torch.equal(gauge.xy, source.xy) or not torch.equal(gauge.chol, source.chol):
                raise AuditInvalid("transform", f"{name} changed xy/chol bitwise")
            for field in ("xy", "chol", "color", "weight"):
                if not bool(torch.isfinite(getattr(gauge, field)).all()):
                    raise AuditInvalid("transform", f"{name}/{field} is non-finite")
            if not bool(((gauge.weight >= 0) & (gauge.weight <= 1)).all()):
                raise AuditInvalid("transform", f"{name} weight is out of range")
            if not bool(((gauge.color >= 0) & (gauge.color <= 1)).all()):
                raise AuditInvalid("transform", f"{name} color is out of range")
            max_abs, max_rel = _amplitude_errors(gauge.weight[:, None] * gauge.color, amplitude)
            record = {
                "seed": seed,
                "local_train_view_index": view_index,
                "original_view_index": TRAIN_INDICES[view_index],
                "component_count": source.n,
                "maximum_amplitude_absolute_error": max_abs,
                "maximum_amplitude_relative_error": max_rel,
                "weight_bin_counts": fixed_bin_counts(gauge.weight),
                "peak_color_bin_counts": fixed_bin_counts(gauge.color.max(dim=-1).values),
                "weight_changed_count": int(
                    ((gauge.weight - source.weight).abs() > CHANGE_EPSILON).sum()
                ),
                "any_color_changed_count": int(
                    ((gauge.color - source.color).abs() > CHANGE_EPSILON).any(dim=-1).sum()
                ),
                "joint_weight_color_changed_count": int(
                    (
                        ((gauge.weight - source.weight).abs() > CHANGE_EPSILON)
                        & ((gauge.color - source.color).abs() > CHANGE_EPSILON).any(dim=-1)
                    ).sum()
                ),
                "field_hash": tensor_collection_hash(
                    (field, getattr(gauge, field)) for field in ("xy", "chol", "color", "weight")
                ),
                "field_hashes": {
                    field: tensor_hash(field, getattr(gauge, field))
                    for field in ("xy", "chol", "color", "weight")
                },
                "amplitude_hash": tensor_hash("amplitude", gauge.weight[:, None] * gauge.color),
            }
            record["weight_changed_fraction"] = record["weight_changed_count"] / source.n
            record["any_color_changed_fraction"] = record["any_color_changed_count"] / source.n
            record["joint_weight_color_changed_fraction"] = (
                record["joint_weight_color_changed_count"] / source.n
            )
            if max_abs > TRANSFORM_MAX_ABS or max_rel > TRANSFORM_MAX_REL:
                raise AuditInvalid(
                    "transform",
                    f"{name} amplitude tolerance failed for seed {seed}/view {view_index}",
                    {"record": record},
                )
            gauges[name].append(gauge)
            per_view[name].append(record)
    aggregate = {}
    for name in GAUGES:
        total = sum(record["component_count"] for record in per_view[name])
        weight_changed = sum(record["weight_changed_count"] for record in per_view[name])
        color_changed = sum(record["any_color_changed_count"] for record in per_view[name])
        joint = sum(record["joint_weight_color_changed_count"] for record in per_view[name])
        aggregate[name] = {
            "component_count": total,
            "weight_changed_count": weight_changed,
            "weight_changed_fraction": weight_changed / total,
            "any_color_changed_count": color_changed,
            "any_color_changed_fraction": color_changed / total,
            "joint_weight_color_changed_count": joint,
            "joint_weight_color_changed_fraction": joint / total,
            "maximum_amplitude_absolute_error": max(
                record["maximum_amplitude_absolute_error"] for record in per_view[name]
            ),
            "maximum_amplitude_relative_error": max(
                record["maximum_amplitude_relative_error"] for record in per_view[name]
            ),
            "gauge_hash": gaussians2d_hash(gauges[name]),
        }
    return gauges, {"per_view": per_view, "aggregate": aggregate}


def prepare_seed(seed: int) -> PreparedSeed:
    full_scene = make_synthetic_scene(
        n_gaussians=40, n_cameras=12, image_size=IMAGE_SIZE, seed=seed
    )
    train_scene = full_scene.subset(list(TRAIN_INDICES), name_suffix="gauge-audit-train")
    del full_scene
    train_scene.validate()
    center, extent = train_scene.center_and_extent()
    original_bounds_hint = train_scene.bounds_hint
    bound_scene = SceneData(
        images=list(train_scene.images),
        cameras=list(train_scene.cameras),
        view_names=None if train_scene.view_names is None else list(train_scene.view_names),
        points=train_scene.points,
        point_visibility=train_scene.point_visibility,
        masks=train_scene.masks,
        gt_depths=train_scene.gt_depths,
        gt_gaussians=None,
        train_indices=list(range(len(TRAIN_INDICES))),
        test_indices=[],
        bounds_hint=original_bounds_hint,
        name=train_scene.name,
        _extent_cache=(center.detach().clone(), extent),
    )
    bound_scene.validate()
    rebound_center, rebound_extent = bound_scene.center_and_extent()
    if not torch.equal(center, rebound_center) or extent != rebound_extent:
        raise AuditInvalid("preparation", "bound scene changed center or extent")
    del train_scene
    if bound_scene.gt_depths is None:
        raise AuditInvalid("preparation", "synthetic training depths are absent")

    input_hashes = {
        "images": {
            "aggregate": tensor_collection_hash(
                (f"image_{index}", image) for index, image in enumerate(bound_scene.images)
            ),
            "per_view": [
                tensor_hash(f"image_{index}", image)
                for index, image in enumerate(bound_scene.images)
            ],
        },
        "cameras": camera_hashes(bound_scene),
        "depths": {
            "aggregate": tensor_collection_hash(
                (f"depth_{index}", depth) for index, depth in enumerate(bound_scene.gt_depths)
            ),
            "per_view": [
                tensor_hash(f"depth_{index}", depth)
                for index, depth in enumerate(bound_scene.gt_depths)
            ],
        },
        "masks": (
            {"is_null": True, "sha256": canonical_json_hash(None)}
            if bound_scene.masks is None
            else {
                "is_null": False,
                "aggregate": tensor_collection_hash(
                    (f"mask_{index}", mask) for index, mask in enumerate(bound_scene.masks)
                ),
                "per_view": [
                    tensor_hash(f"mask_{index}", mask)
                    for index, mask in enumerate(bound_scene.masks)
                ],
            }
        ),
        "points": (
            canonical_json_hash(None)
            if bound_scene.points is None
            else tensor_hash("points", bound_scene.points)
        ),
        "point_visibility": (
            {"is_null": True, "sha256": canonical_json_hash(None)}
            if bound_scene.point_visibility is None
            else {
                "is_null": False,
                "aggregate": tensor_collection_hash(
                    (f"visibility_{index}", indices)
                    for index, indices in enumerate(bound_scene.point_visibility)
                ),
                "per_view": [
                    tensor_hash(f"visibility_{index}", indices)
                    for index, indices in enumerate(bound_scene.point_visibility)
                ],
            }
        ),
        "original_bounds_hint": canonical_json_hash(
            None
            if original_bounds_hint is None
            else {
                "center": original_bounds_hint[0].tolist(),
                "extent": original_bounds_hint[1],
            }
        ),
        "bound_center": tensor_hash("center", center),
        "bound_extent": canonical_json_hash(float(extent)),
        "local_to_original_view_map": canonical_json_hash(list(TRAIN_INDICES)),
    }
    fitted, histories = fit_views(
        bound_scene.images,
        fit_config(),
        seed=seed,
        masks=bound_scene.masks,
    )
    validate_fitted_views(fitted)
    ordering = [
        [source_key(seed, view_index, component) for component in range(gaussian.n)]
        for view_index, gaussian in enumerate(fitted)
    ]
    fit_hashes = {
        "fields": gaussians2d_hash(fitted),
        "per_view_fields": [
            {
                field: tensor_hash(field, getattr(gaussian, field))
                for field in ("xy", "chol", "color", "weight")
            }
            for gaussian in fitted
        ],
        "history": canonical_json_hash(histories),
        "history_per_view": [canonical_json_hash(history) for history in histories],
        "source_order": canonical_json_hash(ordering),
    }
    source_aggregate = canonical_json_hash({"inputs": input_hashes, "fit": fit_hashes})
    preparation = {
        "seed": seed,
        "synthetic_config": {
            "n_gaussians": 40,
            "n_cameras": 12,
            "image_size": IMAGE_SIZE,
        },
        "split": {"train": list(TRAIN_INDICES), "held_out": list(HELD_OUT_INDICES)},
        "local_to_original_view_map": list(TRAIN_INDICES),
        "field_minimal_scene": {
            "gt_gaussians_is_none": bound_scene.gt_gaussians is None,
            "center": [float(value) for value in center],
            "extent_full_diameter": float(extent),
            "original_bounds_hint_is_none": original_bounds_hint is None,
            "bound_scene_bounds_hint_is_none": bound_scene.bounds_hint is None,
            "bound_scene_uses_frozen_extent_cache": True,
        },
        "fit_config": asdict(fit_config()),
        "input_hashes": input_hashes,
        "fit_hashes": fit_hashes,
        "fit_histories": histories,
        "source_order": [[list(key) for key in view] for view in ordering],
        "source_aggregate": source_aggregate,
    }
    try:
        gauges, transformations = construct_gauges(fitted, seed)
    except AuditInvalid as error:
        error.evidence = {"preparation": preparation, **error.evidence}
        raise
    if gaussians2d_hash(fitted) != fit_hashes["fields"]:
        raise AuditInvalid(
            "transform",
            "gauge construction mutated the fitted source",
            {"preparation": preparation},
        )
    return PreparedSeed(
        seed=seed,
        scene=bound_scene,
        fitted=fitted,
        gauges=gauges,
        preparation=preparation,
        transformations=transformations,
    )


def _source_psnr(delta: torch.Tensor) -> float:
    mse = float(delta.double().square().mean())
    return -10.0 * math.log10(max(mse, 1e-12))


def source_equivalence_prerequisite(prepared: list[PreparedSeed]) -> dict[str, Any]:
    """Finish every source render check before returning any downstream authority."""
    pending: list[dict[str, Any]] = []
    renders: list[tuple[str, torch.Tensor]] = []
    for item in prepared:
        for view_index in range(len(TRAIN_INDICES)):
            with torch.no_grad():
                identity = render_gaussians_2d(
                    item.gauges["identity"][view_index],
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    background=torch.zeros(3),
                    row_chunk=64,
                )
            if not bool(torch.isfinite(identity).all()):
                raise AuditInvalid(
                    "source_equivalence",
                    f"identity render is non-finite for seed {item.seed}/view {view_index}",
                    {"completed_checks": len(pending)},
                )
            renders.append((f"seed_{item.seed}/view_{view_index}/identity", identity))
            denominator = float(identity.double().abs().sum())
            if not math.isfinite(denominator) or denominator <= 0:
                raise AuditInvalid(
                    "source_equivalence",
                    f"identity denominator is invalid for seed {item.seed}/view {view_index}",
                    {"completed_checks": len(pending)},
                )
            for transform in TRANSFORMS:
                with torch.no_grad():
                    transformed = render_gaussians_2d(
                        item.gauges[transform][view_index],
                        IMAGE_SIZE,
                        IMAGE_SIZE,
                        background=torch.zeros(3),
                        row_chunk=64,
                    )
                if not bool(torch.isfinite(transformed).all()):
                    raise AuditInvalid(
                        "source_equivalence",
                        f"{transform} render is non-finite",
                        {
                            "seed": item.seed,
                            "view": view_index,
                            "completed_checks": len(pending),
                        },
                    )
                delta = transformed - identity
                numerator = float(delta.double().abs().sum())
                maximum = float(delta.abs().max())
                ratio = numerator / denominator
                psnr_value = _source_psnr(delta)
                record = {
                    "seed": item.seed,
                    "local_train_view_index": view_index,
                    "original_view_index": TRAIN_INDICES[view_index],
                    "transform": transform,
                    "maximum_absolute_rgb_error": maximum,
                    "raw_delta_l1": numerator,
                    "raw_identity_l1": denominator,
                    "delta_over_identity": ratio,
                    "psnr_db": psnr_value,
                }
                if (
                    maximum > SOURCE_RENDER_MAX_ABS
                    or ratio > SOURCE_RENDER_REL_L1
                    or psnr_value < SOURCE_RENDER_MIN_PSNR
                ):
                    invalid_record = {
                        key: value
                        for key, value in record.items()
                        if key not in {"raw_delta_l1", "raw_identity_l1"}
                    }
                    raise AuditInvalid(
                        "source_equivalence",
                        f"source equivalence failed for seed {item.seed}/view "
                        f"{view_index}/{transform}",
                        {
                            "completed_checks": len(pending),
                            "failure": invalid_record,
                            "thresholds": source_equivalence_thresholds(),
                        },
                    )
                pending.append(record)
                renders.append((f"seed_{item.seed}/view_{view_index}/{transform}", transformed))
    expected = len(SEEDS) * len(TRAIN_INDICES) * len(TRANSFORMS)
    if len(pending) != expected:
        raise AuditInvalid("source_equivalence", "global equivalence check count differs")
    render_hashes = {name: tensor_hash(name, render) for name, render in renders}
    return {
        "passed": True,
        "thresholds": source_equivalence_thresholds(),
        "completed_transform_view_checks": len(pending),
        "expected_transform_view_checks": expected,
        "per_transform_view": pending,
        "render_hashes": render_hashes,
        "render_aggregate": canonical_json_hash(render_hashes),
    }


def source_equivalence_thresholds() -> dict[str, float]:
    return {
        "maximum_absolute_rgb_error": SOURCE_RENDER_MAX_ABS,
        "delta_over_identity": SOURCE_RENDER_REL_L1,
        "minimum_psnr_db": SOURCE_RENDER_MIN_PSNR,
        "psnr_mse_floor": 1e-12,
    }


def run_after_source_equivalence(
    prepared: list[PreparedSeed],
    downstream: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Fail closed without invoking downstream work when source equivalence fails."""
    equivalence = source_equivalence_prerequisite(prepared)
    return downstream(equivalence)


def key_set_comparison(
    identity: Iterable[tuple[int, int, int]],
    transformed: Iterable[tuple[int, int, int]],
) -> dict[str, Any]:
    identity_set = set(identity)
    transformed_set = set(transformed)
    intersection = identity_set & transformed_set
    union = identity_set | transformed_set
    symmetric = identity_set ^ transformed_set
    return {
        "identity_count": len(identity_set),
        "transformed_count": len(transformed_set),
        "intersection_count": len(intersection),
        "union_count": len(union),
        "symmetric_difference_count": len(symmetric),
        "jaccard": len(intersection) / len(union) if union else 1.0,
        "set_disagreement": len(symmetric) / len(union) if union else 0.0,
        "identity_keys": serialize_keys(sorted(identity_set)),
        "transformed_keys": serialize_keys(sorted(transformed_set)),
        "intersection_keys": serialize_keys(sorted(intersection)),
        "union_keys": serialize_keys(sorted(union)),
        "symmetric_difference_keys": serialize_keys(sorted(symmetric)),
    }


def input_materiality_from_raw(
    *,
    joint_changed_count: int,
    source_component_count: int,
    retention_symmetric_difference_count: int,
    retention_union_count: int,
    coverage_delta_l1: float,
    coverage_reference_l1: float,
    coverage_crossing_count: int,
    coverage_pixel_count: int,
) -> dict[str, Any]:
    if source_component_count <= 0 or coverage_pixel_count <= 0:
        raise ValueError("input materiality denominators must be positive")
    if retention_union_count <= 0 or coverage_reference_l1 <= 0:
        raise ValueError("consumed-input denominators must be positive")
    joint_fraction = joint_changed_count / source_component_count
    retention_fraction = retention_symmetric_difference_count / retention_union_count
    coverage_ratio = coverage_delta_l1 / coverage_reference_l1
    crossing_fraction = coverage_crossing_count / coverage_pixel_count
    retention_gate = retention_symmetric_difference_count >= 10 and retention_fraction >= 0.01
    coverage_gate = coverage_ratio >= 0.01 and crossing_fraction >= 0.001
    material = joint_fraction >= 0.10 and (retention_gate or coverage_gate)
    return {
        "joint_weight_color_changed_count": joint_changed_count,
        "source_component_count": source_component_count,
        "joint_weight_color_changed_fraction": joint_fraction,
        "retention_symmetric_difference_count": retention_symmetric_difference_count,
        "retention_union_count": retention_union_count,
        "retention_symmetric_difference_fraction": retention_fraction,
        "coverage_delta_l1": coverage_delta_l1,
        "coverage_reference_l1": coverage_reference_l1,
        "coverage_delta_over_reference": coverage_ratio,
        "coverage_crossing_count": coverage_crossing_count,
        "coverage_pixel_count": coverage_pixel_count,
        "coverage_crossing_fraction": crossing_fraction,
        "criteria": {
            "joint_weight_color_changed_fraction_at_least_0_10": joint_fraction >= 0.10,
            "retention_key_count_at_least_10": retention_symmetric_difference_count >= 10,
            "retention_fraction_at_least_0_01": retention_fraction >= 0.01,
            "coverage_ratio_at_least_0_01": coverage_ratio >= 0.01,
            "coverage_crossing_fraction_at_least_0_001": crossing_fraction >= 0.001,
        },
        "retention_mechanism_gate": retention_gate,
        "coverage_mechanism_gate": coverage_gate,
        "input_consumption_material": material,
    }


def _retained_keys(
    views: list[Gaussians2D], seed: int
) -> tuple[list[list[tuple[int, int, int]]], list[tuple[int, int, int]]]:
    per_view = []
    pooled = []
    for view_index, gaussian in enumerate(views):
        indices = torch.where(gaussian.weight > 0.05)[0].tolist()
        keys = [source_key(seed, view_index, int(index)) for index in indices]
        per_view.append(keys)
        pooled.extend(keys)
    return per_view, pooled


def coverage_retention_seed(item: PreparedSeed) -> dict[str, Any]:
    maps: dict[str, list[torch.Tensor]] = {}
    map_hashes: dict[str, list[str]] = {}
    for gauge in GAUGES:
        maps[gauge] = []
        map_hashes[gauge] = []
        for view_index, gaussian in enumerate(item.gauges[gauge]):
            with torch.no_grad():
                coverage = render_gaussian_coverage_2d(
                    gaussian, IMAGE_SIZE, IMAGE_SIZE, row_chunk=64
                )
            if not bool(torch.isfinite(coverage).all()):
                raise AuditInvalid(
                    "coverage_retention",
                    f"coverage is non-finite for seed {item.seed}/{gauge}/{view_index}",
                )
            maps[gauge].append(coverage)
            map_hashes[gauge].append(tensor_hash("coverage", coverage))

    retained = {gauge: _retained_keys(item.gauges[gauge], item.seed) for gauge in GAUGES}
    comparisons = {}
    for transform in TRANSFORMS:
        per_view_coverage = []
        per_view_retention = []
        for view_index, (identity, changed) in enumerate(zip(maps["identity"], maps[transform])):
            delta = changed - identity
            reference_l1 = float(identity.double().abs().sum())
            delta_l1 = float(delta.double().abs().sum())
            crossing = (changed > 0.40) ^ (identity > 0.40)
            per_view_coverage.append(
                {
                    "local_train_view_index": view_index,
                    "original_view_index": TRAIN_INDICES[view_index],
                    "coverage_delta_l1": delta_l1,
                    "coverage_reference_l1": reference_l1,
                    "coverage_threshold_crossing_count": int(crossing.sum()),
                    "coverage_pixel_count": crossing.numel(),
                    "maximum_absolute_delta": float(delta.abs().max()),
                }
            )
            per_view_retention.append(
                {
                    "local_train_view_index": view_index,
                    "original_view_index": TRAIN_INDICES[view_index],
                    **key_set_comparison(
                        retained["identity"][0][view_index],
                        retained[transform][0][view_index],
                    ),
                }
            )
        coverage_delta = ordered_float64_sum(
            record["coverage_delta_l1"] for record in per_view_coverage
        )
        coverage_reference = ordered_float64_sum(
            record["coverage_reference_l1"] for record in per_view_coverage
        )
        crossing_count = sum(
            record["coverage_threshold_crossing_count"] for record in per_view_coverage
        )
        pixel_count = sum(record["coverage_pixel_count"] for record in per_view_coverage)
        if not math.isfinite(coverage_reference) or coverage_reference <= 0:
            raise AuditInvalid("coverage_retention", "coverage reference denominator is invalid")
        retention = key_set_comparison(retained["identity"][1], retained[transform][1])
        changed = item.transformations["aggregate"][transform]
        materiality = input_materiality_from_raw(
            joint_changed_count=changed["joint_weight_color_changed_count"],
            source_component_count=changed["component_count"],
            retention_symmetric_difference_count=retention["symmetric_difference_count"],
            retention_union_count=retention["union_count"],
            coverage_delta_l1=coverage_delta,
            coverage_reference_l1=coverage_reference,
            coverage_crossing_count=crossing_count,
            coverage_pixel_count=pixel_count,
        )
        comparisons[transform] = {
            "coverage": {
                "per_view": per_view_coverage,
                "coverage_delta_l1": coverage_delta,
                "coverage_reference_l1": coverage_reference,
                "coverage_delta_over_reference": coverage_delta / coverage_reference,
                "coverage_threshold_crossing_count": crossing_count,
                "coverage_pixel_count": pixel_count,
                "coverage_threshold_crossing_fraction": crossing_count / pixel_count,
                "maximum_absolute_delta": max(
                    record["maximum_absolute_delta"] for record in per_view_coverage
                ),
            },
            "retention": {"per_view": per_view_retention, "per_seed": retention},
            "input_materiality": materiality,
        }
    return {
        "seed": item.seed,
        "config": {"coverage_threshold": 0.40, "retention_min_weight_strict": 0.05},
        "coverage_map_hashes": map_hashes,
        "retained_keys": {
            gauge: {
                "per_view": [serialize_keys(keys) for keys in retained[gauge][0]],
                "per_seed": serialize_keys(retained[gauge][1]),
            }
            for gauge in GAUGES
        },
        "comparisons": comparisons,
    }


def pool_input_materiality(
    seeds: list[dict[str, Any]], prepared: list[PreparedSeed]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    prepared_by_seed = {item.seed: item for item in prepared}
    for transform in TRANSFORMS:
        identity_keys: list[tuple[int, int, int]] = []
        transformed_keys: list[tuple[int, int, int]] = []
        coverage_delta_values = []
        coverage_reference_values = []
        crossing_count = 0
        pixel_count = 0
        joint_count = 0
        source_count = 0
        seed_passes = []
        maximum_absolute_deltas = []
        for seed_record in seeds:
            comparison = seed_record["comparisons"][transform]
            identity_keys.extend(
                tuple(key) for key in seed_record["retained_keys"]["identity"]["per_seed"]
            )
            transformed_keys.extend(
                tuple(key) for key in seed_record["retained_keys"][transform]["per_seed"]
            )
            coverage = comparison["coverage"]
            coverage_delta_values.append(coverage["coverage_delta_l1"])
            coverage_reference_values.append(coverage["coverage_reference_l1"])
            crossing_count += coverage["coverage_threshold_crossing_count"]
            pixel_count += coverage["coverage_pixel_count"]
            maximum_absolute_deltas.append(coverage["maximum_absolute_delta"])
            seed_passes.append(comparison["input_materiality"]["input_consumption_material"])
            transformed = prepared_by_seed[seed_record["seed"]].transformations["aggregate"][
                transform
            ]
            joint_count += transformed["joint_weight_color_changed_count"]
            source_count += transformed["component_count"]
        retention = key_set_comparison(identity_keys, transformed_keys)
        pooled = input_materiality_from_raw(
            joint_changed_count=joint_count,
            source_component_count=source_count,
            retention_symmetric_difference_count=retention["symmetric_difference_count"],
            retention_union_count=retention["union_count"],
            coverage_delta_l1=ordered_float64_sum(coverage_delta_values),
            coverage_reference_l1=ordered_float64_sum(coverage_reference_values),
            coverage_crossing_count=crossing_count,
            coverage_pixel_count=pixel_count,
        )
        result[transform] = {
            "seed_passes": seed_passes,
            "seed_pass_count": sum(seed_passes),
            "retention": retention,
            "pooled_maximum_absolute_coverage_delta": max(maximum_absolute_deltas),
            "pooled_materiality": pooled,
            "material_consumed_input_effect": sum(seed_passes) >= 2
            and pooled["input_consumption_material"],
        }
    return result


def _frozen_renderer() -> TorchRasterizer:
    return TorchRasterizer(
        sh_color_activation="hard",
        kernel_support_mode="hard",
        visibility_margin_sigma=3.0,
    )


def render_lift_views(
    output: Gaussians3D, scene: SceneData
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, Any]]]:
    renderer = _frozen_renderer()
    runtime = []
    evidence = []
    with torch.no_grad():
        for view_index, camera in enumerate(scene.cameras):
            rendered = renderer.render(
                output,
                camera,
                background=torch.zeros(3, dtype=output.means.dtype),
                sh_degree=0,
            )
            tensors = {
                "color": rendered.color.detach(),
                "alpha": rendered.alpha.detach(),
                "depth": rendered.depth.detach(),
            }
            if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
                raise AuditInvalid("lift_render", f"non-finite lift render in view {view_index}")
            runtime.append(tensors)
            evidence.append(
                {
                    "local_train_view_index": view_index,
                    "original_view_index": TRAIN_INDICES[view_index]
                    if len(scene.cameras) == len(TRAIN_INDICES)
                    else view_index,
                    "hash": tensor_collection_hash(tensors.items()),
                    "color_l1": float(tensors["color"].double().abs().sum()),
                    "alpha_l1": float(tensors["alpha"].double().abs().sum()),
                    "accumulated_depth_l1": float(tensors["depth"].double().abs().sum()),
                }
            )
    return runtime, evidence


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).abs().max()) if actual.numel() else 0.0


def _assert_gaussian_parity(
    production: Gaussians3D,
    diagnostic: Gaussians3D,
    *,
    context: str,
    include_sh: bool = True,
) -> dict[str, float]:
    if production.n != diagnostic.n:
        raise AuditInvalid(context, f"{context} production/diagnostic counts differ")
    fields = {
        "means": (production.means, diagnostic.means),
        "covariance": (production.covariance(), diagnostic.covariance()),
        "opacity": (production.opacity, diagnostic.opacity),
    }
    if include_sh:
        fields["sh"] = (production.sh, diagnostic.sh)
    errors = {}
    for name, (actual, expected) in fields.items():
        errors[name] = _max_abs(actual, expected)
        if not torch.allclose(actual, expected, atol=2e-6, rtol=2e-5):
            raise AuditInvalid(context, f"{context} parity failed for {name}", {"errors": errors})
    return errors


def _depth_independent_reconstruction(
    item: PreparedSeed,
    gauge: str,
) -> tuple[list[tuple[int, int, int]], Gaussians3D | None, list[dict[str, Any]]]:
    if item.scene.gt_depths is None:
        raise AuditInvalid("depth", "metric training depths are absent")
    parts = []
    keys = []
    view_records = []
    for view_index, (gaussian, camera, depth_map) in enumerate(
        zip(item.gauges[gauge], item.scene.cameras, item.scene.gt_depths)
    ):
        depth_map = depth_map.to(gaussian.xy)
        z = bilinear_sample(depth_map, gaussian.xy)
        finite = torch.isfinite(z)
        depth_valid = z > 0.05
        strict_weight = gaussian.weight > 0.05
        mask_valid = torch.ones_like(strict_weight)
        if item.scene.masks is not None:
            mask_valid = (
                bilinear_sample(item.scene.masks[view_index].to(gaussian.xy), gaussian.xy) > 0.5
            )
        confidence = torch.ones_like(z)
        confidence_valid = confidence > 0.1
        valid = finite & depth_valid & strict_weight & mask_valid & confidence_valid
        indices = torch.where(valid)[0]
        view_keys = [source_key(item.seed, view_index, int(index)) for index in indices.tolist()]
        keys.extend(view_keys)
        view_records.append(
            {
                "local_train_view_index": view_index,
                "original_view_index": TRAIN_INDICES[view_index],
                "finite_depth_count": int(finite.sum()),
                "depth_above_0_05_count": int(depth_valid.sum()),
                "strict_weight_count": int(strict_weight.sum()),
                "optional_mask_valid_count": int(mask_valid.sum()),
                "confidence_above_0_1_count": int(confidence_valid.sum()),
                "emitted_count": len(view_keys),
                "emitted_keys": serialize_keys(view_keys),
            }
        )
        if indices.numel() == 0:
            continue
        selected = Gaussians2D(
            xy=gaussian.xy[valid],
            chol=gaussian.chol[valid],
            color=gaussian.color[valid],
            weight=gaussian.weight[valid],
        )
        z_valid = z[valid]
        parts.append(
            lift_view_from_depth_map(
                camera,
                selected,
                depth_map,
                z_valid,
                0,
                opacity=torch.full_like(z_valid, 0.1),
                normal_thickness=0.15,
                robust_depth_gradients=True,
            )
        )
    return keys, Gaussians3D.cat(parts) if parts else None, view_records


def run_depth_gauge(item: PreparedSeed, gauge: str) -> LiftRuntime:
    keys, diagnostic, mask_records = _depth_independent_reconstruction(item, gauge)
    if item.scene.gt_depths is None:
        raise AuditInvalid("depth", "training depths are absent")
    lifter = DepthLifter(
        backend=GroundTruthDepth(item.scene.gt_depths),
        **depth_kwargs(),
    )
    try:
        production = lifter.lift(item.gauges[gauge], item.scene)
    except ValueError as error:
        raise AuditInvalid(
            "depth_empty",
            f"Depth produced an empty output for seed {item.seed}/{gauge}: {error}",
            {"gauge": gauge, "mask_records": mask_records, "expected_keys": serialize_keys(keys)},
        ) from error
    if diagnostic is None:
        raise AuditInvalid(
            "depth_empty",
            f"independent Depth masks are empty for seed {item.seed}/{gauge}",
            {"gauge": gauge, "mask_records": mask_records},
        )
    parity = _assert_gaussian_parity(production, diagnostic, context="depth_order")
    if production.n != len(keys):
        raise AuditInvalid("depth_order", "Depth ordered source-key count differs")
    renders, render_evidence = render_lift_views(production, item.scene)
    evidence = {
        "gauge": gauge,
        "config": depth_kwargs(),
        "ordinary_lifter_call_count": 1,
        "independent_mask_records": mask_records,
        "ordered_output_keys": serialize_keys(keys),
        "output_count": production.n,
        "output_field_hash": gaussians3d_hash(production),
        "output_field_hashes": {
            field: tensor_hash(field, getattr(production, field))
            for field in ("means", "quats", "log_scales", "opacity", "sh")
        },
        "output_covariance_hash": tensor_hash("covariance", production.covariance()),
        "independent_order_parity_errors": parity,
        "independent_order_parity_tolerances": {"atol": 2e-6, "rtol": 2e-5},
        "renders": render_evidence,
    }
    return LiftRuntime(output=production, keys=keys, renders=renders, evidence=evidence)


def _render_raw_comparison(
    identity: list[dict[str, torch.Tensor]],
    transformed: list[dict[str, torch.Tensor]],
    scene: SceneData,
) -> dict[str, Any]:
    if len(identity) != len(transformed) or len(identity) != len(scene.images):
        raise AuditInvalid("lift_comparison", "lift-render view counts differ")
    per_view = []
    for view_index, (reference, changed, target) in enumerate(
        zip(identity, transformed, scene.images)
    ):
        color_delta = float((changed["color"] - reference["color"]).double().abs().sum())
        alpha_delta = float((changed["alpha"] - reference["alpha"]).double().abs().sum())
        depth_delta = float((changed["depth"] - reference["depth"]).double().abs().sum())
        color_signal = float(reference["color"].double().abs().sum())
        color_residual = float((reference["color"] - target).double().abs().sum())
        if color_signal <= 0 or color_residual <= 0:
            raise AuditInvalid("lift_comparison", "frozen render denominator is not positive")
        per_view.append(
            {
                "local_train_view_index": view_index,
                "original_view_index": TRAIN_INDICES[view_index]
                if len(scene.images) == len(TRAIN_INDICES)
                else view_index,
                "color_delta_l1": color_delta,
                "identity_color_signal_l1": color_signal,
                "identity_color_residual_l1": color_residual,
                "alpha_delta_l1": alpha_delta,
                "accumulated_depth_delta_l1": depth_delta,
            }
        )
    raw = {
        "render_delta_l1": ordered_float64_sum(record["color_delta_l1"] for record in per_view),
        "identity_signal_l1": ordered_float64_sum(
            record["identity_color_signal_l1"] for record in per_view
        ),
        "identity_residual_l1": ordered_float64_sum(
            record["identity_color_residual_l1"] for record in per_view
        ),
        "alpha_delta_l1": ordered_float64_sum(record["alpha_delta_l1"] for record in per_view),
        "accumulated_depth_delta_l1": ordered_float64_sum(
            record["accumulated_depth_delta_l1"] for record in per_view
        ),
    }
    if raw["identity_signal_l1"] <= 0 or raw["identity_residual_l1"] <= 0:
        raise AuditInvalid("lift_comparison", "pooled frozen render denominator is not positive")
    return {
        "per_view": per_view,
        **raw,
        "render_delta_over_signal": raw["render_delta_l1"] / raw["identity_signal_l1"],
        "render_delta_over_residual": raw["render_delta_l1"] / raw["identity_residual_l1"],
    }


def lift_materiality_from_raw(
    *,
    joint_changed_count: int,
    source_component_count: int,
    set_symmetric_difference_count: int,
    set_union_count: int,
    render_delta_l1: float,
    identity_signal_l1: float,
    identity_residual_l1: float,
) -> dict[str, Any]:
    if source_component_count <= 0 or set_union_count <= 0:
        raise ValueError("lift count denominators must be positive")
    if identity_signal_l1 <= 0 or identity_residual_l1 <= 0:
        raise ValueError("lift render denominators must be positive")
    joint_fraction = joint_changed_count / source_component_count
    set_disagreement = set_symmetric_difference_count / set_union_count
    signal_ratio = render_delta_l1 / identity_signal_l1
    residual_ratio = render_delta_l1 / identity_residual_l1
    set_gate = set_symmetric_difference_count >= 10 and set_disagreement >= 0.01
    render_gate = signal_ratio >= 0.001 and residual_ratio >= 0.01
    return {
        "joint_weight_color_changed_count": joint_changed_count,
        "source_component_count": source_component_count,
        "joint_weight_color_changed_fraction": joint_fraction,
        "set_symmetric_difference_count": set_symmetric_difference_count,
        "set_union_count": set_union_count,
        "set_disagreement": set_disagreement,
        "render_delta_l1": render_delta_l1,
        "identity_signal_l1": identity_signal_l1,
        "identity_residual_l1": identity_residual_l1,
        "render_delta_over_signal": signal_ratio,
        "render_delta_over_residual": residual_ratio,
        "criteria": {
            "joint_weight_color_changed_fraction_at_least_0_10": joint_fraction >= 0.10,
            "set_symmetric_difference_count_at_least_10": set_symmetric_difference_count >= 10,
            "set_disagreement_at_least_0_01": set_disagreement >= 0.01,
            "render_delta_over_signal_at_least_0_001": signal_ratio >= 0.001,
            "render_delta_over_residual_at_least_0_01": residual_ratio >= 0.01,
        },
        "set_mechanism_gate": set_gate,
        "render_mechanism_gate": render_gate,
        "lift_material": joint_fraction >= 0.10 and (set_gate or render_gate),
    }


def compare_lifts(
    identity: LiftRuntime,
    transformed: LiftRuntime,
    *,
    scene: SceneData,
    changed_counts: dict[str, Any],
    require_shared_geometry: bool,
) -> dict[str, Any]:
    sets = key_set_comparison(identity.keys, transformed.keys)
    identity_lookup = {key: index for index, key in enumerate(identity.keys)}
    transformed_lookup = {key: index for index, key in enumerate(transformed.keys)}
    shared = sorted(set(identity_lookup) & set(transformed_lookup))
    extent = scene.center_and_extent()[1]
    identity_covariances = identity.output.covariance()
    transformed_covariances = transformed.output.covariance()
    per_key = []
    for key in shared:
        left_index = identity_lookup[key]
        right_index = transformed_lookup[key]
        left_mean = identity.output.means[left_index]
        right_mean = transformed.output.means[right_index]
        left_covariance = identity_covariances[left_index]
        right_covariance = transformed_covariances[right_index]
        left_opacity = identity.output.opacity[left_index]
        right_opacity = transformed.output.opacity[right_index]
        left_sh = identity.output.sh[left_index]
        right_sh = transformed.output.sh[right_index]
        if require_shared_geometry:
            for name, left, right in (
                ("means", left_mean, right_mean),
                ("covariance", left_covariance, right_covariance),
                ("opacity", left_opacity, right_opacity),
            ):
                if not torch.allclose(left, right, atol=2e-6, rtol=2e-5):
                    raise AuditInvalid(
                        "depth_shared_geometry",
                        f"shared Depth key {key} changed {name}",
                    )
        covariance_denominator = float(left_covariance.double().flatten().norm())
        if covariance_denominator <= 0:
            raise AuditInvalid("lift_comparison", "shared covariance norm is not positive")
        record = {
            "key": list(key),
            "mean_maximum_absolute_delta": _max_abs(right_mean, left_mean),
            "center_displacement": float((right_mean - left_mean).double().norm()),
            "center_displacement_over_extent": float((right_mean - left_mean).double().norm())
            / extent,
            "covariance_maximum_absolute_delta": _max_abs(right_covariance, left_covariance),
            "covariance_frobenius_delta": float(
                (right_covariance - left_covariance).double().flatten().norm()
            ),
            "relative_covariance_frobenius_delta": float(
                (right_covariance - left_covariance).double().flatten().norm()
            )
            / covariance_denominator,
            "opacity_absolute_delta": float((right_opacity - left_opacity).double().abs()),
            "sh_frobenius_delta": float((right_sh - left_sh).double().flatten().norm()),
        }
        if identity.source_records is not None and transformed.source_records is not None:
            left_record = identity.source_records[key]
            right_record = transformed.source_records[key]
            record["tunnel_score_absolute_delta"] = abs(
                float(right_record["best_score"]) - float(left_record["best_score"])
            )
            record["selected_depth_absolute_delta"] = abs(
                float(right_record["selected_depth"]) - float(left_record["selected_depth"])
            )
        per_key.append(record)
    renders = _render_raw_comparison(identity.renders, transformed.renders, scene)
    materiality = lift_materiality_from_raw(
        joint_changed_count=changed_counts["joint_weight_color_changed_count"],
        source_component_count=changed_counts["component_count"],
        set_symmetric_difference_count=sets["symmetric_difference_count"],
        set_union_count=sets["union_count"],
        render_delta_l1=renders["render_delta_l1"],
        identity_signal_l1=renders["identity_signal_l1"],
        identity_residual_l1=renders["identity_residual_l1"],
    )
    return {
        "output_key_sets": sets,
        "shared_key_deltas": {
            "per_key": per_key,
            "shared_count": len(per_key),
            "maximum_center_displacement_over_extent": max(
                (record["center_displacement_over_extent"] for record in per_key),
                default=0.0,
            ),
            "maximum_mean_absolute_delta": max(
                (record["mean_maximum_absolute_delta"] for record in per_key), default=0.0
            ),
            "maximum_covariance_absolute_delta": max(
                (record["covariance_maximum_absolute_delta"] for record in per_key),
                default=0.0,
            ),
            "maximum_covariance_frobenius_delta": max(
                (record["covariance_frobenius_delta"] for record in per_key), default=0.0
            ),
            "maximum_relative_covariance_frobenius_delta": max(
                (record["relative_covariance_frobenius_delta"] for record in per_key),
                default=0.0,
            ),
            "maximum_opacity_absolute_delta": max(
                (record["opacity_absolute_delta"] for record in per_key), default=0.0
            ),
            "maximum_sh_frobenius_delta": max(
                (record["sh_frobenius_delta"] for record in per_key), default=0.0
            ),
        },
        "render_comparison": renders,
        "materiality": materiality,
        "shared_geometry_control_required": require_shared_geometry,
        "shared_geometry_control_passed": True if require_shared_geometry else None,
        "shared_geometry_control_tolerances": (
            {"atol": 2e-6, "rtol": 2e-5} if require_shared_geometry else None
        ),
    }


def run_depth_seed(item: PreparedSeed) -> tuple[dict[str, Any], dict[str, LiftRuntime]]:
    runtime = {gauge: run_depth_gauge(item, gauge) for gauge in GAUGES}
    comparisons = {
        transform: compare_lifts(
            runtime["identity"],
            runtime[transform],
            scene=item.scene,
            changed_counts=item.transformations["aggregate"][transform],
            require_shared_geometry=True,
        )
        for transform in TRANSFORMS
    }
    return {
        "seed": item.seed,
        "gauges": {gauge: runtime[gauge].evidence for gauge in GAUGES},
        "comparisons": comparisons,
    }, runtime


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach()
    if not bool(torch.isfinite(tensor).all()):
        raise AuditInvalid("carve_sidecar", "named Carve volume tensor is non-finite")
    if tensor.numel() == 0:
        return {"shape": list(tensor.shape), "count": 0}
    if tensor.dtype == torch.bool:
        return {
            "shape": list(tensor.shape),
            "count": tensor.numel(),
            "true_count": int(tensor.sum()),
        }
    as_double = tensor.double()
    return {
        "shape": list(tensor.shape),
        "count": tensor.numel(),
        "minimum": float(as_double.min()),
        "maximum": float(as_double.max()),
        "sum": float(as_double.sum()),
        "mean": float(as_double.mean()),
    }


def carve_sidecar(
    item: PreparedSeed,
    gauge: str,
    config: dict[str, Any] | None = None,
) -> tuple[
    Gaussians3D | None,
    list[tuple[int, int, int]],
    dict[tuple[int, int, int], dict[str, Any]],
    dict[str, Any],
]:
    """Duplicate current unmerged Carve arithmetic without invoking a second lifter."""
    cfg = carve_kwargs() if config is None else dict(config)
    if cfg.get("merge") is not False:
        raise ValueError("Carve gauge sidecar requires the frozen unmerged configuration")
    scene = item.scene
    views = item.gauges[gauge]
    center, extent = scene.center_and_extent()
    half = extent * cfg["bounds_scale"]
    lo = center - half
    voxel = 2.0 * half / cfg["grid_res"]
    grid = cfg["grid_res"]
    device = center.device
    axis = torch.arange(grid, dtype=torch.float32, device=device)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    idx3 = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    centers = lo[None, :] + (idx3 + 0.5) * voxel
    voxel_count = centers.shape[0]

    coverage_maps = []
    with torch.no_grad():
        for view_index, (gaussian, camera) in enumerate(zip(views, scene.cameras)):
            if scene.masks is not None:
                coverage_maps.append(scene.masks[view_index].to(gaussian.xy).float())
            else:
                coverage_maps.append(
                    render_gaussian_coverage_2d(gaussian, camera.height, camera.width)
                )

    n_seen = torch.zeros(voxel_count, device=device)
    n_covered = torch.zeros(voxel_count, device=device)
    color_sum = torch.zeros(voxel_count, 3, device=device)
    color_square_sum = torch.zeros(voxel_count, 3, device=device)
    for image, coverage, camera in zip(scene.images, coverage_maps, scene.cameras):
        uv, z = camera.project(centers)
        inside = (z > 0.05) & camera.in_image(uv, margin=-0.5)
        indices = inside.nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            continue
        covered = bilinear_sample(coverage, uv[indices]) > cfg["coverage_thresh"]
        colors = bilinear_sample(image, uv[indices])
        n_seen[indices] += 1
        n_covered[indices] += covered.float()
        color_sum[indices] += colors
        color_square_sum[indices] += colors**2

    seen_ok = n_seen >= cfg["min_views"]
    hull = seen_ok & (n_covered >= cfg["hull_fraction"] * n_seen) & (n_covered >= cfg["min_views"])
    count = n_seen.clamp_min(1.0)
    mean_color = color_sum / count[:, None]
    variance = (color_square_sum / count[:, None] - mean_color**2).clamp_min(0.0)
    standard_deviation = variance.mean(dim=-1).sqrt()
    consistency = hull.float() * torch.exp(
        -(standard_deviation**2) / (2 * cfg["color_std_sigma"] ** 2)
    )

    records: dict[tuple[int, int, int], dict[str, Any]] = {}
    parts = []
    emitted_keys = []
    samples = cfg["samples_per_ray"]
    for view_index, (gaussian, camera) in enumerate(zip(views, scene.cameras)):
        keep = gaussian.weight > cfg["min_weight"]
        if scene.masks is not None:
            keep &= bilinear_sample(scene.masks[view_index].to(gaussian.xy), gaussian.xy) > 0.5
        for component in range(gaussian.n):
            records[source_key(item.seed, view_index, component)] = {
                "key": list(source_key(item.seed, view_index, component)),
                "keep": bool(keep[component]),
                "valid_ray": None,
                "best_score": None,
                "best_idx": None,
                "selected_depth": None,
                "raw_score_weighted_depth_variance": None,
                "clamped_ray_sigma": None,
                "placed": None,
            }
        if int(keep.sum()) == 0:
            continue
        keep_indices = torch.where(keep)[0]
        selected = Gaussians2D(
            xy=gaussian.xy[keep],
            chol=gaussian.chol[keep],
            color=gaussian.color[keep],
            weight=gaussian.weight[keep],
        )
        origin, directions = camera.pixel_rays(selected.xy)
        t0, t1 = _ray_box(origin, directions, lo, lo + 2 * half)
        valid_ray = t1 > t0.clamp_min(0.05)
        t0 = t0.clamp_min(0.05)
        steps = torch.linspace(0.0, 1.0, samples, device=device)
        sample_depths = t0[:, None] + (t1 - t0).clamp_min(0.0)[:, None] * steps[None, :]
        points = origin.reshape(1, 1, 3) + sample_depths[:, :, None] * directions[:, None, :]
        voxels = torch.floor((points - lo) / voxel).long()
        in_grid = ((voxels >= 0) & (voxels < grid)).all(dim=-1) & valid_ray[:, None]
        voxels = voxels.clamp(0, grid - 1)
        flat = (voxels[..., 2] * grid + voxels[..., 1]) * grid + voxels[..., 0]
        score = consistency[flat] * in_grid.float()
        voxel_color = mean_color[flat]
        color_difference = ((voxel_color - selected.color[:, None, :]) ** 2).sum(-1)
        score = score * torch.exp(-color_difference / (2 * cfg["color_match_sigma"] ** 2))
        best_score, best_index = score.max(dim=1)
        placed = best_score > cfg["min_score"]
        selected_depth = torch.gather(sample_depths, 1, best_index[:, None])[:, 0]
        near_peak = (sample_depths - selected_depth[:, None]).abs() <= 3.0 * voxel
        local_weight = score * near_peak.float()
        weight_sum = local_weight.sum(dim=1).clamp_min(1e-8)
        mean_depth = (local_weight * sample_depths).sum(dim=1) / weight_sum
        depth_variance = (local_weight * (sample_depths - mean_depth[:, None]) ** 2).sum(
            dim=1
        ) / weight_sum
        sigma_ray = depth_variance.sqrt().clamp(0.25 * voxel, 2.0 * voxel)

        for local_index, original_index in enumerate(keep_indices.tolist()):
            key = source_key(item.seed, view_index, int(original_index))
            records[key].update(
                {
                    "valid_ray": bool(valid_ray[local_index]),
                    "best_score": float(best_score[local_index]),
                    "best_idx": int(best_index[local_index]),
                    "selected_depth": float(selected_depth[local_index]),
                    "raw_score_weighted_depth_variance": float(depth_variance[local_index]),
                    "clamped_ray_sigma": float(sigma_ray[local_index]),
                    "placed": bool(placed[local_index]),
                }
            )
        if int(placed.sum()) == 0:
            continue
        placed_indices = keep_indices[placed]
        view_keys = [
            source_key(item.seed, view_index, int(index)) for index in placed_indices.tolist()
        ]
        emitted_keys.extend(view_keys)
        sub = Gaussians2D(
            xy=selected.xy[placed],
            chol=selected.chol[placed],
            color=selected.color[placed],
            weight=selected.weight[placed],
        )
        parts.append(
            lift_view_at_depth(
                camera,
                sub,
                selected_depth[placed],
                sigma_ray[placed],
                cfg["sh_degree"],
                opacity=torch.full_like(selected_depth[placed], cfg["init_opacity"]),
            )
        )

    named_volumes = {
        "idx3": idx3,
        "centers": centers,
        "n_seen": n_seen,
        "n_covered": n_covered,
        "color_sum": color_sum,
        "color_square_sum": color_square_sum,
        "seen_ok": seen_ok,
        "hull": hull,
        "count": count,
        "mean_color": mean_color,
        "variance": variance,
        "standard_deviation": standard_deviation,
        "consistency": consistency,
    }
    coverage_hashes = [
        tensor_hash(f"coverage_{index}", coverage) for index, coverage in enumerate(coverage_maps)
    ]
    volume_hashes = {name: tensor_hash(name, value) for name, value in named_volumes.items()}
    serialized_source_records = [records[key] for key in sorted(records)]
    evidence = {
        "config": cfg,
        "bounds": {
            "center": [float(value) for value in center],
            "extent_full_diameter": float(extent),
            "half_extent": float(half),
            "lower": [float(value) for value in lo],
            "voxel_size": float(voxel),
        },
        "coverage_map_hashes": coverage_hashes,
        "coverage_map_aggregate": canonical_json_hash(coverage_hashes),
        "volume_hashes": volume_hashes,
        "volume_hash_aggregate": canonical_json_hash(volume_hashes),
        "volume_summaries": {name: tensor_summary(value) for name, value in named_volumes.items()},
        "source_records": serialized_source_records,
        "source_records_hash": canonical_json_hash(serialized_source_records),
        "source_counts": {
            "total": len(records),
            "keep": sum(record["keep"] for record in records.values()),
            "valid_ray": sum(record["valid_ray"] is True for record in records.values()),
            "placed": sum(record["placed"] is True for record in records.values()),
        },
        "ordered_emitted_keys": serialize_keys(emitted_keys),
    }
    return Gaussians3D.cat(parts) if parts else None, emitted_keys, records, evidence


def run_carve_gauge(
    item: PreparedSeed,
    gauge: str,
    config: dict[str, Any] | None = None,
) -> LiftRuntime:
    cfg = carve_kwargs() if config is None else dict(config)
    diagnostic, keys, records, sidecar_evidence = carve_sidecar(item, gauge, cfg)
    lifter = CarveLifter(**cfg)
    try:
        production = lifter.lift(item.gauges[gauge], item.scene)
    except ValueError as error:
        raise AuditInvalid(
            "carve_empty",
            f"Carve produced an empty output for seed {item.seed}/{gauge}: {error}",
            {"gauge": gauge, "sidecar": sidecar_evidence},
        ) from error
    if diagnostic is None:
        raise AuditInvalid(
            "carve_empty",
            f"Carve sidecar is empty for seed {item.seed}/{gauge}",
            {"gauge": gauge, "sidecar": sidecar_evidence},
        )
    parity = _assert_gaussian_parity(production, diagnostic, context="carve_sidecar")
    if production.n != len(keys):
        raise AuditInvalid("carve_sidecar", "Carve ordered emitted-key count differs")
    renders, render_evidence = render_lift_views(production, item.scene)
    evidence = {
        "gauge": gauge,
        "config": cfg,
        "ordinary_lifter_call_count": 1,
        "sidecar": sidecar_evidence,
        "output_count": production.n,
        "output_field_hash": gaussians3d_hash(production),
        "output_field_hashes": {
            field: tensor_hash(field, getattr(production, field))
            for field in ("means", "quats", "log_scales", "opacity", "sh")
        },
        "output_covariance_hash": tensor_hash("covariance", production.covariance()),
        "sidecar_parity_errors": parity,
        "sidecar_parity_tolerances": {"atol": 2e-6, "rtol": 2e-5},
        "renders": render_evidence,
    }
    return LiftRuntime(
        output=production,
        keys=keys,
        renders=renders,
        evidence=evidence,
        source_records=records,
    )


def run_carve_seed(item: PreparedSeed) -> tuple[dict[str, Any], dict[str, LiftRuntime]]:
    runtime = {gauge: run_carve_gauge(item, gauge) for gauge in GAUGES}
    comparisons = {
        transform: compare_lifts(
            runtime["identity"],
            runtime[transform],
            scene=item.scene,
            changed_counts=item.transformations["aggregate"][transform],
            require_shared_geometry=False,
        )
        for transform in TRANSFORMS
    }
    return {
        "seed": item.seed,
        "gauges": {gauge: runtime[gauge].evidence for gauge in GAUGES},
        "comparisons": comparisons,
    }, runtime


def pool_lift_backend(
    seed_records: list[dict[str, Any]], prepared: list[PreparedSeed]
) -> dict[str, Any]:
    prepared_by_seed = {item.seed: item for item in prepared}
    transforms = {}
    for transform in TRANSFORMS:
        identity_keys = []
        transformed_keys = []
        render_delta = []
        identity_signal = []
        identity_residual = []
        alpha_delta = []
        depth_delta = []
        joint_count = 0
        source_count = 0
        seed_passes = []
        for seed_record in seed_records:
            comparison = seed_record["comparisons"][transform]
            sets = comparison["output_key_sets"]
            identity_keys.extend(tuple(key) for key in sets["identity_keys"])
            transformed_keys.extend(tuple(key) for key in sets["transformed_keys"])
            renders = comparison["render_comparison"]
            render_delta.append(renders["render_delta_l1"])
            identity_signal.append(renders["identity_signal_l1"])
            identity_residual.append(renders["identity_residual_l1"])
            alpha_delta.append(renders["alpha_delta_l1"])
            depth_delta.append(renders["accumulated_depth_delta_l1"])
            seed_passes.append(comparison["materiality"]["lift_material"])
            transformed = prepared_by_seed[seed_record["seed"]].transformations["aggregate"][
                transform
            ]
            joint_count += transformed["joint_weight_color_changed_count"]
            source_count += transformed["component_count"]
        sets = key_set_comparison(identity_keys, transformed_keys)
        raw = {
            "render_delta_l1": ordered_float64_sum(render_delta),
            "identity_signal_l1": ordered_float64_sum(identity_signal),
            "identity_residual_l1": ordered_float64_sum(identity_residual),
            "alpha_delta_l1": ordered_float64_sum(alpha_delta),
            "accumulated_depth_delta_l1": ordered_float64_sum(depth_delta),
        }
        materiality = lift_materiality_from_raw(
            joint_changed_count=joint_count,
            source_component_count=source_count,
            set_symmetric_difference_count=sets["symmetric_difference_count"],
            set_union_count=sets["union_count"],
            render_delta_l1=raw["render_delta_l1"],
            identity_signal_l1=raw["identity_signal_l1"],
            identity_residual_l1=raw["identity_residual_l1"],
        )
        transforms[transform] = {
            "seed_passes": seed_passes,
            "seed_pass_count": sum(seed_passes),
            "pooled_output_key_sets": sets,
            "pooled_render_raw": raw,
            "pooled_materiality": materiality,
            "materially_gauge_dependent": sum(seed_passes) >= 2 and materiality["lift_material"],
        }
    qualifying = [
        transform for transform in TRANSFORMS if transforms[transform]["materially_gauge_dependent"]
    ]
    return {
        "transforms": transforms,
        "qualifying_transforms": qualifying,
        "backend_materially_gauge_dependent": bool(qualifying),
    }


def frozen_interpretation(depth_pass: bool, carve_pass: bool) -> str:
    if depth_pass and carve_pass:
        return (
            "Both tested boundaries are materially gauge-dependent in this narrow setup; "
            "the audit does not identify a correct replacement."
        )
    if depth_pass:
        return (
            "The retained/color Depth boundary is materially gauge-dependent here; this "
            "makes no claim about Carve geometry."
        )
    if carve_pass:
        return (
            "The Carve coverage/color-match placement boundary is materially "
            "gauge-dependent here; this makes no claim about Depth."
        )
    return (
        "The two tested gauges did not produce a material unoptimized-lift difference "
        "under this setup; this is not a universal invariance claim."
    )


def run_downstream_diagnostics(
    prepared: list[PreparedSeed], equivalence: dict[str, Any]
) -> dict[str, Any]:
    coverage_seeds = []
    for item in prepared:
        coverage_seeds.append(coverage_retention_seed(item))
    input_pool = pool_input_materiality(coverage_seeds, prepared)

    depth_seeds = []
    for item in prepared:
        depth_record, _ = run_depth_seed(item)
        depth_seeds.append(depth_record)
    depth_pool = pool_lift_backend(depth_seeds, prepared)

    carve_seeds = []
    for item in prepared:
        carve_record, _ = run_carve_seed(item)
        carve_seeds.append(carve_record)
    carve_pool = pool_lift_backend(carve_seeds, prepared)

    depth_pass = depth_pool["backend_materially_gauge_dependent"]
    carve_pass = carve_pool["backend_materially_gauge_dependent"]
    return {
        "artifact_type": "stage1_weight_gauge_audit",
        "source_equivalence": equivalence,
        "coverage_and_retention": {
            "seeds": coverage_seeds,
            "pooled": input_pool,
        },
        "depth": {"seeds": depth_seeds, "pooled": depth_pool},
        "carve": {"seeds": carve_seeds, "pooled": carve_pool},
        "decision": {
            "source_equivalence_passed": True,
            "depth_materially_gauge_dependent": depth_pass,
            "carve_materially_gauge_dependent": carve_pass,
            "interpretation": frozen_interpretation(depth_pass, carve_pass),
            "default_change_authorized": False,
            "phase_b_exists": False,
        },
    }


def invalid_payload(
    *,
    stage: str,
    reason: str,
    preparations: list[dict[str, Any]],
    transformations: list[dict[str, Any]],
    equivalence: dict[str, Any] | None,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "artifact_type": "stage1_weight_gauge_invalid",
        "invalid_stage": stage,
        "reason": reason,
        "preparations": preparations,
        "transformations": transformations,
        "evidence_reached_before_invalidation": evidence,
        "decision": None,
    }
    if equivalence is not None:
        payload["source_equivalence"] = equivalence
    if stage == "source_equivalence":
        forbidden = {"coverage_and_retention", "depth", "carve", "materiality"}
        if forbidden & payload.keys():  # pragma: no cover - construction invariant
            raise AssertionError("source-equivalence invalid payload contains downstream evidence")
    return payload


def execute_scientific_audit() -> dict[str, Any]:
    """Execute the frozen audit, returning a valid or fail-closed invalid payload."""
    prepared: list[PreparedSeed] = []
    try:
        for seed in SEEDS:
            print(f"preparing frozen seed {seed}", flush=True)
            prepared.append(prepare_seed(seed))
    except AuditInvalid as error:
        return invalid_payload(
            stage=error.stage,
            reason=error.reason,
            preparations=[item.preparation for item in prepared],
            transformations=[item.transformations for item in prepared],
            equivalence=None,
            evidence=error.evidence,
        )

    preparations = [item.preparation for item in prepared]
    transformations = [{"seed": item.seed, **item.transformations} for item in prepared]
    try:
        equivalence = source_equivalence_prerequisite(prepared)
    except AuditInvalid as error:
        return invalid_payload(
            stage=error.stage,
            reason=error.reason,
            preparations=preparations,
            transformations=transformations,
            equivalence=error.evidence,
            evidence=error.evidence,
        )
    try:
        return {
            **run_downstream_diagnostics(prepared, equivalence),
            "preparations": preparations,
            "transformations": transformations,
        }
    except AuditInvalid as error:
        return invalid_payload(
            stage=error.stage,
            reason=error.reason,
            preparations=preparations,
            transformations=transformations,
            equivalence=equivalence,
            evidence=error.evidence,
        )


def environment_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "device": "cpu",
    }


def environment_fingerprint(metadata: dict[str, Any]) -> dict[str, Any]:
    return dict(metadata)


def assert_official_environment(metadata: dict[str, Any]) -> None:
    expected = {
        "torch_num_threads": 4,
        "deterministic_algorithms": True,
        "cuda_visible_devices": "",
        "omp_num_threads": "4",
        "mkl_num_threads": "4",
        "device": "cpu",
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"official CPU environment differs: {mismatches}")


def git_metadata() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    return {
        "revision": revision,
        "dirty": bool(status.strip()),
        "status": status.splitlines(),
        "tracked_diff_sha256": sha256_bytes(diff),
    }


def source_hashes(paths: tuple[Path, ...] = SEALED_PATHS) -> tuple[dict[str, str], str]:
    missing = [str(path) for path in paths if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"sealed source files are missing: {missing}")
    hashes = {str(path): sha256_file(ROOT / path) for path in paths}
    return hashes, canonical_json_hash(hashes)


def loaded_source_hashes() -> tuple[dict[str, str], str]:
    paths: set[Path] = set()
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if source is None:
            continue
        path = Path(source).resolve()
        if path.suffix != ".py" or not path.is_relative_to(ROOT) or not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if relative.parts and relative.parts[0] != ".venv":
            paths.add(path)
    paths.update(
        {
            (ROOT / PREREGISTRATION).resolve(),
            (ROOT / "pyproject.toml").resolve(),
        }
    )
    hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(paths)}
    return hashes, canonical_json_hash(hashes)


def strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {value!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def fixed_protocol() -> dict[str, Any]:
    renderer = _frozen_renderer()
    if (
        renderer.sh_color_activation != "hard"
        or renderer.kernel_support_mode != "hard"
        or renderer.visibility_margin_sigma != 3.0
    ):
        raise AssertionError("frozen Torch renderer semantics differ")
    if sha256_file(ROOT / PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("preregistration hash differs from the bound protocol")
    return {
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
        },
        "seeds": list(SEEDS),
        "split": {"train": list(TRAIN_INDICES), "held_out": list(HELD_OUT_INDICES)},
        "fit_config": asdict(fit_config()),
        "gauges": list(GAUGES),
        "weight_bins": list(WEIGHT_BINS),
        "source_equivalence_thresholds": source_equivalence_thresholds(),
        "depth_config": depth_kwargs(),
        "carve_config": carve_kwargs(),
        "renderer": {
            "sh_color_activation": "hard",
            "kernel_support_mode": "hard",
            "visibility_margin_sigma": 3.0,
            "sh_degree": 0,
            "background": "black",
            "clamp": False,
        },
        "input_materiality_thresholds": {
            "joint_changed_fraction": 0.10,
            "retention_key_count": 10,
            "retention_fraction": 0.01,
            "coverage_delta_over_reference": 0.01,
            "coverage_crossing_fraction": 0.001,
            "seed_passes": 2,
        },
        "lift_materiality_thresholds": {
            "joint_changed_fraction": 0.10,
            "set_key_count": 10,
            "set_disagreement": 0.01,
            "render_delta_over_signal": 0.001,
            "render_delta_over_residual": 0.01,
            "seed_passes": 2,
        },
    }


VERIFICATION_COMMANDS = (
    (".venv/bin/python", "-m", "ruff", "check", "."),
    (".venv/bin/python", "-m", "ruff", "format", "--check", "."),
    (".venv/bin/python", "-m", "pytest", "-q", "-m", "not slow"),
    (".venv/bin/python", "scripts/docs_sync.py"),
    ("git", "diff", "--check"),
)
VERIFICATION_LITERAL_COMMANDS = (
    ".venv/bin/python -m ruff check .",
    ".venv/bin/python -m ruff format --check .",
    '.venv/bin/python -m pytest -q -m "not slow"',
    ".venv/bin/python scripts/docs_sync.py",
    "git diff --check",
)


def run_verification() -> dict[str, Any]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    results = []
    for frozen_command, literal_command in zip(
        VERIFICATION_COMMANDS, VERIFICATION_LITERAL_COMMANDS, strict=True
    ):
        command = list(frozen_command)
        print(f"verification: {' '.join(command)}", flush=True)
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        record = {
            "command": command,
            "literal_command": literal_command,
            "returncode": completed.returncode,
            "seconds": time.perf_counter() - started,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "stdout_sha256": sha256_bytes(completed.stdout.encode()),
            "stderr_sha256": sha256_bytes(completed.stderr.encode()),
            "combined_output_sha256": sha256_bytes(
                completed.stdout.encode() + b"\x00" + completed.stderr.encode()
            ),
        }
        results.append(record)
        if completed.returncode != 0:
            raise RuntimeError(
                f"verification failed: {' '.join(command)}\n"
                f"{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
            )
    return {"passed": True, "commands": results}


def create_seal() -> dict[str, Any]:
    environment = environment_metadata()
    assert_official_environment(environment)
    protocol = fixed_protocol()
    hashes_before, aggregate_before = source_hashes()
    verification = run_verification()
    hashes_after, aggregate_after = source_hashes()
    if hashes_after != hashes_before or aggregate_after != aggregate_before:
        raise RuntimeError("sealed implementation changed during verification")
    if fixed_protocol() != protocol:
        raise RuntimeError("frozen protocol changed during verification")
    return {
        "artifact_type": "stage1_weight_gauge_implementation_seal",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_metadata(),
        "sealed_paths": [str(path) for path in SEALED_PATHS],
        "source_hashes": hashes_after,
        "source_aggregate": aggregate_after,
        "protocol": protocol,
        "verification": verification,
        "environment": environment,
        "command": [sys.executable, *sys.argv],
    }


def load_and_verify_seal(path: Path) -> dict[str, Any]:
    payload = strict_json_load(path)
    if payload.get("artifact_type") != "stage1_weight_gauge_implementation_seal":
        raise ValueError(f"{path} is not a stage-1 weight-gauge implementation seal")
    expected_paths = [str(item) for item in SEALED_PATHS]
    if payload.get("sealed_paths") != expected_paths:
        raise RuntimeError("implementation seal path set differs from repository set")
    hashes, aggregate = source_hashes(tuple(Path(item) for item in expected_paths))
    if hashes != payload.get("source_hashes") or aggregate != payload.get("source_aggregate"):
        raise RuntimeError("implementation/protocol differs from sealed source aggregate")
    if payload.get("protocol") != fixed_protocol():
        raise RuntimeError("implementation seal protocol differs")
    verification = payload.get("verification", {})
    commands = verification.get("commands", [])
    if not verification.get("passed") or [record.get("command") for record in commands] != [
        list(command) for command in VERIFICATION_COMMANDS
    ]:
        raise RuntimeError("implementation seal lacks the exact passing verification sequence")
    if [record.get("literal_command") for record in commands] != list(
        VERIFICATION_LITERAL_COMMANDS
    ):
        raise RuntimeError("implementation seal literal verification commands differ")
    if any(record.get("returncode") != 0 for record in commands):
        raise RuntimeError("implementation seal records a failed verification command")
    for record in commands:
        stdout = record.get("stdout")
        stderr = record.get("stderr")
        if not isinstance(stdout, str) or not isinstance(stderr, str):
            raise RuntimeError("implementation seal lacks complete verification output")
        if record.get("stdout_sha256") != sha256_bytes(stdout.encode()):
            raise RuntimeError("implementation seal stdout hash differs")
        if record.get("stderr_sha256") != sha256_bytes(stderr.encode()):
            raise RuntimeError("implementation seal stderr hash differs")
        if record.get("combined_output_sha256") != sha256_bytes(
            stdout.encode() + b"\x00" + stderr.encode()
        ):
            raise RuntimeError("implementation seal combined output hash differs")
    current_environment = environment_metadata()
    assert_official_environment(current_environment)
    if environment_fingerprint(payload["environment"]) != environment_fingerprint(
        current_environment
    ):
        raise RuntimeError("current environment differs from implementation seal")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "source_aggregate": aggregate,
        "verification_sha256": canonical_json_hash(verification),
        "environment_fingerprint": environment_fingerprint(payload["environment"]),
    }


def verify_loaded_sources_against_seal(seal_path: Path) -> tuple[dict[str, str], str]:
    payload = strict_json_load(seal_path)
    sealed_hashes = payload["source_hashes"]
    loaded_hashes, aggregate = loaded_source_hashes()
    unexpected = sorted(set(loaded_hashes) - set(sealed_hashes))
    mismatched = sorted(
        path
        for path, digest in loaded_hashes.items()
        if path in sealed_hashes and sealed_hashes[path] != digest
    )
    if unexpected or mismatched:
        raise RuntimeError(
            "loaded repository sources are outside/different from seal: "
            f"unexpected={unexpected}, mismatched={mismatched}"
        )
    return loaded_hashes, aggregate


def companion_note_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_RESULT.md")


def invalid_output_path(prospective: Path) -> Path:
    suffix = "_audit.json"
    if not prospective.name.endswith(suffix):
        raise ValueError("prospective scientific output must end in '_audit.json'")
    return prospective.with_name(f"{prospective.name[: -len(suffix)]}_invalid.json")


def validate_official_output_path(prospective: Path) -> None:
    pattern = r"\d{8}T\d{6}Z_cpu_stage1_weight_gauge_audit\.json"
    if re.fullmatch(pattern, prospective.name) is None:
        raise ValueError("official output must use the frozen UTC audit filename")
    expected_parent = (ROOT / "benchmarks" / "results").resolve()
    actual_parent = (
        (ROOT / prospective).resolve().parent
        if not prospective.is_absolute()
        else prospective.resolve().parent
    )
    if actual_parent != expected_parent:
        raise ValueError("official output must be created under benchmarks/results")


def preflight_audit_outputs(prospective: Path) -> Path:
    invalid = invalid_output_path(prospective)
    prospective.parent.mkdir(parents=True, exist_ok=True)
    paths = (
        prospective,
        invalid,
        companion_note_path(prospective),
        companion_note_path(invalid),
    )
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to start because output possibilities exist: {existing}")
    return invalid


def claim_attempt(
    path: Path,
    *,
    prospective_output: Path,
    invalid_output: Path,
    inputs: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_type": "stage1_weight_gauge_once_only_attempt",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prospective_valid_output": str(prospective_output),
        "derived_invalid_output": str(invalid_output),
        "prospective_valid_note": str(companion_note_path(prospective_output)),
        "derived_invalid_note": str(companion_note_path(invalid_output)),
        "inputs": inputs,
        "command": [sys.executable, *sys.argv],
        "environment": environment_metadata(),
    }
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    except FileExistsError as error:
        raise RuntimeError(f"the preregistered audit has already been claimed by {path}") from error


def result_note(payload: dict[str, Any], output: Path, digest: str) -> str:
    lines = [
        f"# {payload['artifact_type']}",
        "",
        f"- Timestamp (UTC): `{payload['timestamp_utc']}`",
        f"- JSON artifact: `{output}`",
        f"- JSON SHA-256: `{digest}`",
        f"- Command: `{' '.join(payload['command'])}`",
    ]
    if payload["artifact_type"] == "stage1_weight_gauge_audit":
        decision = payload["decision"]
        lines.extend(
            [
                "",
                "## Frozen decision",
                "",
                f"- Depth materially gauge-dependent: "
                f"`{decision['depth_materially_gauge_dependent']}`",
                f"- Carve materially gauge-dependent: "
                f"`{decision['carve_materially_gauge_dependent']}`",
                "",
                "This representation-contract audit authorizes no default change.",
            ]
        )
    elif payload["artifact_type"] == "stage1_weight_gauge_invalid":
        lines.extend(
            [
                "",
                "## Fail-closed disposition",
                "",
                f"- Invalid stage: `{payload['invalid_stage']}`",
                "- No backend decision or default change is authorized.",
            ]
        )
    return "\n".join(lines) + "\n"


def write_artifact(output: Path, payload: dict[str, Any]) -> tuple[Path, str]:
    note = companion_note_path(output)
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
    digest = sha256_bytes(rendered.encode())
    with note.open("x", encoding="utf-8") as handle:
        handle.write(result_note(payload, output, digest))
    return note, digest


def run_bound_audit(seal_path: Path) -> dict[str, Any]:
    seal = load_and_verify_seal(seal_path)
    loaded_before, loaded_aggregate_before = verify_loaded_sources_against_seal(seal_path)
    started = time.perf_counter()
    payload = execute_scientific_audit()
    loaded_after, loaded_aggregate_after = verify_loaded_sources_against_seal(seal_path)
    if loaded_after != loaded_before or loaded_aggregate_after != loaded_aggregate_before:
        raise RuntimeError("loaded repository source set changed during the audit")
    if load_and_verify_seal(seal_path) != seal:
        raise RuntimeError("implementation seal changed during the audit")
    payload.update(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": git_metadata(),
            "seal": seal,
            "protocol": fixed_protocol(),
            "command": [sys.executable, *sys.argv],
            "environment": environment_metadata(),
            "loaded_source_hashes": loaded_after,
            "loaded_source_aggregate": loaded_aggregate_after,
            "wall_seconds": time.perf_counter() - started,
        }
    )
    assert_finite_tree(payload, "audit payload")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    seal = subparsers.add_parser("seal", help="verify and freeze the complete implementation")
    seal.add_argument("--output", type=Path, default=DEFAULT_SEAL)
    audit = subparsers.add_parser("audit", help="run the once-only representation audit")
    audit.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    audit.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if args.command_name == "seal":
        if args.output != DEFAULT_SEAL:
            raise ValueError("implementation seal must use the preregistered fixed path")
        note = companion_note_path(args.output)
        if args.output.exists() or note.exists():
            raise FileExistsError(f"refusing to overwrite {args.output} or {note}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = create_seal()
        output = args.output
    elif args.command_name == "audit":
        if args.seal != DEFAULT_SEAL:
            raise ValueError("audit must use the preregistered implementation-seal path")
        validate_official_output_path(args.output)
        invalid = preflight_audit_outputs(args.output)
        seal = load_and_verify_seal(args.seal)
        loaded_hashes, loaded_aggregate = verify_loaded_sources_against_seal(args.seal)
        # Re-hash all bound sources immediately before the atomic marker.
        if load_and_verify_seal(args.seal) != seal:
            raise RuntimeError("implementation seal changed before attempt claim")
        claim_attempt(
            ATTEMPT,
            prospective_output=args.output,
            invalid_output=invalid,
            inputs={
                "seal_sha256": seal["sha256"],
                "source_aggregate": seal["source_aggregate"],
                "loaded_source_aggregate": loaded_aggregate,
                "loaded_source_hashes_sha256": canonical_json_hash(loaded_hashes),
                "preregistration_sha256": PREREGISTRATION_SHA256,
            },
        )
        payload = run_bound_audit(args.seal)
        if payload["artifact_type"] == "stage1_weight_gauge_audit":
            output = args.output
        elif payload["artifact_type"] == "stage1_weight_gauge_invalid":
            output = invalid
        else:  # pragma: no cover - construction invariant
            raise RuntimeError("audit returned an unknown artifact type")
    else:  # pragma: no cover
        raise AssertionError(args.command_name)
    note, digest = write_artifact(output, payload)
    print(f"saved {output} (sha256={digest})", flush=True)
    print(f"saved {note}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
