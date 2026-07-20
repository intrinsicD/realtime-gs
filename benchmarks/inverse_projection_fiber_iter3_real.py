#!/usr/bin/env python3
"""Preregistered calibrated interaction for Iteration 3 Gaussian-fiber correspondence.

The runner deliberately stages access:

1. only the five development compact fields are decoded;
2. development-only bounds, anchors, geometry, plans, and appearance are frozen and hashed;
3. the two validation fields are released through the repository's strict full-bundle loader;
4. C1004 RGB/mask data is decoded only behind ``--release-report`` and only after the
   appearance freeze receipt verifies.

Real-data associations are diagnostics, never ground-truth correspondences.  Compact amplitudes
are not used as confidence or opacity; viewer PLYs retain an independent fixed opacity of 0.1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.core.sh import eval_sh_preactivation
from rtgs.data import reconstruction_inputs as reconstruction
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.fiber_correspondence import (
    CorrespondencePlan,
    FiberFitConfig,
    FiberFitResult,
    ObservationGaussians,
    exponential_schedule,
    fit_fiber_correspondence,
    pairwise_bhattacharyya_cost,
    safe_projection_geometry,
    validate_fiber_state,
)
from rtgs.lift.inverse_projection_fiber import InverseProjectionFiber
from rtgs.lift.source_anchored_sh import (
    SourceAnchoredSHFitConfig,
    SourceAnchoredSHFitResult,
    fit_source_anchored_sh,
)
from rtgs.lift.topology import radius_connected_components
from rtgs.render.projection import EWA_NEAR

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "rtgs.inverse-projection-fiber.iter3.real.v1"
SYNTHETIC_NAMESPACE = "rtgs.inverse-projection-fiber.iter3.synthetic.v1"
BUNDLE = ROOT / "runs/compact_masked_bundle_640_20260717/reconstruction_inputs"
SCENE = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008"
DEFAULT_OUT = ROOT / "runs/inverse_projection_fiber_iter3_real_20260717"
DEFAULT_RESULT = ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter3_REAL.json"
OFFICIAL_ATTEMPT = (
    ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter3_REAL_ATTEMPT.json"
)
SYNTHETIC_RESULT = (
    ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter3_SYNTHETIC_RESULT.json"
)
SYNTHETIC_ATTEMPT = (
    ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter3_SYNTHETIC_ATTEMPT.json"
)
SYNTHETIC_ARTIFACTS = ROOT / "runs/inverse_projection_fiber_iter3_synthetic_20260717"
PREREG = ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter3_PREREG.md"
PREREG_SHA256 = "59f0de21da20bb5785e2c5f14c89fc82114fed2d5945c704115d64b9fb3c27c8"
PREREG_DOCUMENTS = (
    (PREREG, PREREG_SHA256),
    (
        ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter3_PREREG_ADDENDUM_1.md",
        "f4ef57320edf1e099c24033753bf3e939d2c87fcf6b927b65bd5d6af213c91fc",
    ),
    (
        ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter3_PREREG_ADDENDUM_2.md",
        "2fbb29d2bdea86018009d1b3913820edda38de9f3881ae503eca9041c2c2eddc",
    ),
    (
        ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter3_PREREG_ADDENDUM_3.md",
        "69e55637cd97b88d40826daf3a18629a1de61b9efa328487f61c498411c98205",
    ),
)
FROZEN_MANIFEST_SHA256 = "6ed60cf3df1f9ca476dfabace18b8d868d63eebd041522313c5b4fd644ee2614"

SYNTHETIC_OFFICIAL_ROOT_TUPLES = (
    (37_688_011, 37_688_111, 37_688_211),
    (37_688_012, 37_688_112, 37_688_212),
    (37_688_013, 37_688_113, 37_688_213),
)
SYNTHETIC_ARM_NAMES = (
    "hardmin",
    "row",
    "uot_uniform",
    "uot_area",
    "oracle",
    "shuffled_view",
)
SYNTHETIC_FROZEN_CONFIG = {
    "outer_steps": 20,
    "geometry_steps": 2,
    "learning_rate": 0.025,
    "temperature_start": 2.0,
    "temperature_stop": 0.10,
    "residual_variance_start": 1.0,
    "residual_variance_stop": 0.05,
    "dustbin_cost": 4.0,
    "sinkhorn_iterations": 50,
    "marginal_penalty": 1.0,
    "sinkhorn_tolerance": 0.0,
}
SYNTHETIC_SOURCE_ROOT = ROOT
SYNTHETIC_EXECUTED_SOURCE_PATHS = (
    Path("benchmarks/inverse_projection_fiber_iter3.py"),
    Path("src/rtgs/core/camera.py"),
    Path("src/rtgs/core/gaussians3d.py"),
    Path("src/rtgs/data/synthetic.py"),
    Path("src/rtgs/lift/fiber_correspondence.py"),
    Path("src/rtgs/lift/inverse_projection_fiber.py"),
    Path("src/rtgs/lift/topology.py"),
    Path("src/rtgs/render/projection.py"),
)
REAL_SOURCE_ROOT = ROOT
REAL_EXECUTED_SOURCE_PATHS = (
    Path("benchmarks/inverse_projection_fiber_iter3_real.py"),
    Path("src/rtgs/core/camera.py"),
    Path("src/rtgs/core/gaussians3d.py"),
    Path("src/rtgs/core/observation2d.py"),
    Path("src/rtgs/core/sh.py"),
    Path("src/rtgs/data/reconstruction_inputs.py"),
    Path("src/rtgs/lift/fiber_correspondence.py"),
    Path("src/rtgs/lift/inverse_projection_fiber.py"),
    Path("src/rtgs/lift/source_anchored_sh.py"),
    Path("src/rtgs/lift/topology.py"),
    Path("src/rtgs/render/projection.py"),
)

DEVELOPMENT_IDS = ("C0001", "C0008", "C0014", "C0021", "C0026")
VALIDATION_IDS = ("C0031", "C0039")
REPORT_ID = "C1004"
EXPECTED_BUNDLE_IDS = (*DEVELOPMENT_IDS, *VALIDATION_IDS)
ARMS = ("uot_uniform", "uot_area")
N_ANCHORS_PER_VIEW = 128
GRID_SIZE = 8
PER_CELL = 2
N_TRACKS = len(DEVELOPMENT_IDS) * N_ANCHORS_PER_VIEW
FIXED_OPACITY = 0.1
NEAR_DEPTH = 0.05


@dataclass(frozen=True)
class DevelopmentLoad:
    inputs: ReconstructionInputs
    receipt: dict[str, Any]


@dataclass(frozen=True)
class Bounds:
    center: torch.Tensor
    extent: float
    lower: torch.Tensor
    upper: torch.Tensor
    semantic_sha256: str


@dataclass(frozen=True)
class AnchorSelection:
    source_view_indices: torch.Tensor
    source_component_indices: torch.Tensor
    source_means2d: torch.Tensor
    source_covariances2d: torch.Tensor
    depth_lower: torch.Tensor
    depth_upper: torch.Tensor
    initial_depths: torch.Tensor
    per_view_indices: tuple[torch.Tensor, ...]
    all_depth_lower: tuple[torch.Tensor, ...]
    all_depth_upper: tuple[torch.Tensor, ...]
    all_ray_valid: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class RealProblem:
    inputs: ReconstructionInputs
    bounds: Bounds
    anchors: AnchorSelection
    observations_uniform: tuple[ObservationGaussians, ...]
    observations_area: tuple[ObservationGaussians, ...]
    track_capacities_uniform: torch.Tensor
    track_capacities_area: torch.Tensor
    dilation: float


@dataclass
class ArmState:
    name: str
    fiber: InverseProjectionFiber
    fit: FiberFitResult
    elapsed_seconds: float
    peak_rss_kib: int
    initial_set_costs: tuple[float, ...]
    final_set_costs: tuple[float, ...]
    diagnostics: dict[str, Any]
    artifacts: dict[str, dict[str, Any]]


@dataclass
class AppearanceState:
    fit: SourceAnchoredSHFitResult
    source_colors: torch.Tensor
    coefficients: torch.Tensor
    dev_metrics: dict[str, Any]
    gaussians: Gaussians3D
    artifacts: dict[str, dict[str, Any]]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(b"\0")
    digest.update(_canonical_bytes(list(tensor.shape)))
    digest.update(b"\0")
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _descriptor(path: Path) -> dict[str, Any]:
    metadata = path.stat(follow_symlinks=False)
    return {
        "path": str(path),
        "bytes": metadata.st_size,
        "sha256": _sha256_file(path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    payload = _canonical_bytes(value)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)
    return _descriptor(path)


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    return np.ascontiguousarray(value.detach().cpu().numpy())


def _write_npz_exclusive(
    path: Path, arrays: dict[str, torch.Tensor | np.ndarray]
) -> dict[str, Any]:
    materialized = {name: _to_numpy(value) for name, value in arrays.items()}
    with path.open("xb") as stream:
        np.savez(stream, **materialized)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)
    result = _descriptor(path)
    semantic = hashlib.sha256()
    for name, array in sorted(materialized.items()):
        semantic.update(name.encode())
        semantic.update(b"\0")
        semantic.update(array.dtype.str.encode())
        semantic.update(b"\0")
        semantic.update(_canonical_bytes(list(array.shape)))
        semantic.update(b"\0")
        semantic.update(array.tobytes(order="C"))
    result["semantic_sha256"] = semantic.hexdigest()
    result["members"] = {
        name: {"dtype": array.dtype.str, "shape": list(array.shape)}
        for name, array in sorted(materialized.items())
    }
    return result


def _save_ply_exclusive(path: Path, gaussians: Gaussians3D) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    gaussians.detach().to("cpu").save_ply(path)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)
    return _descriptor(path)


def _source_hashes(root: Path, paths: tuple[Path, ...]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in paths:
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"executed-source path is not a safe relative path: {relative}")
        source = root / relative
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"executed source is not a regular non-symlink file: {source}")
        hashes[str(relative)] = _sha256_file(source)
    return hashes


def _synthetic_source_hashes() -> dict[str, str]:
    return _source_hashes(SYNTHETIC_SOURCE_ROOT, SYNTHETIC_EXECUTED_SOURCE_PATHS)


def _real_source_hashes() -> dict[str, str]:
    return _source_hashes(REAL_SOURCE_ROOT, REAL_EXECUTED_SOURCE_PATHS)


def _strict_manifest(
    directory: Path,
    *,
    expected_sha256: str | None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Validate the exact manifest without opening any compact teacher archive."""

    root = reconstruction._strict_bundle_root(directory)
    path = reconstruction._strict_regular_file(
        root, "manifest.json", label="reconstruction manifest"
    )
    payload = path.read_bytes()
    observed_sha256 = _sha256_bytes(payload)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise RuntimeError(
            "frozen compact-bundle manifest hash mismatch "
            f"(expected {expected_sha256}, observed {observed_sha256})"
        )
    manifest = reconstruction._strict_json_object(payload, label="reconstruction manifest")
    reconstruction._require_exact_keys(
        manifest,
        reconstruction._MANIFEST_KEYS,
        label="reconstruction manifest",
    )
    if manifest.get("schema") != reconstruction._SCHEMA:
        raise ValueError("unsupported reconstruction inputs manifest")
    digest_payload = dict(manifest)
    stored_digest = digest_payload.pop("semantic_digest", None)
    if stored_digest != _sha256_bytes(_canonical_bytes(digest_payload)):
        raise ValueError("reconstruction inputs semantic digest mismatch")
    records = manifest.get("views")
    if not isinstance(records, list) or not records:
        raise ValueError("strict reconstruction manifest requires non-empty views")
    validated_records = []
    for index, candidate in enumerate(records):
        record = reconstruction._require_exact_keys(
            candidate,
            reconstruction._VIEW_KEYS,
            label=f"view record {index}",
        )
        reconstruction._require_identifier(record["view_id"], label=f"view record {index} view_id")
        reconstruction._require_sha256(
            record["teacher_sha256"], label=f"view record {index} teacher_sha256"
        )
        reconstruction._validate_strict_camera_record(
            record["camera"], label=f"view record {index} camera"
        )
        if record["teacher"] != f"teachers/{index:04d}.teacher.npz":
            raise ValueError(f"view record {index} teacher path is not canonical")
        validated_records.append(record)
    view_ids = tuple(record["view_id"] for record in validated_records)
    if view_ids != EXPECTED_BUNDLE_IDS:
        raise RuntimeError(f"bundle roles are not the frozen exact IDs/order: {view_ids!r}")
    if manifest.get("geometry") is not None:
        raise RuntimeError("frozen compact bundle must declare geometry:null")
    calibration_payload = [
        {"view_id": record["view_id"], "camera": record["camera"]} for record in records
    ]
    if manifest.get("calibration_digest") != _sha256_bytes(_canonical_bytes(calibration_payload)):
        raise ValueError("calibration digest mismatch")
    return (
        root,
        manifest,
        {
            "manifest_path": str(path),
            "manifest_sha256": observed_sha256,
            "manifest_semantic_sha256": stored_digest,
            "manifest_bytes": len(payload),
            "view_ids": list(view_ids),
        },
    )


def _load_selected_inputs(
    directory: Path,
    selected_ids: tuple[str, ...],
    *,
    expected_manifest_sha256: str | None,
) -> DevelopmentLoad:
    """Strict-load only explicit compact fields and return an access receipt."""

    if len(set(selected_ids)) != len(selected_ids) or not set(selected_ids) <= set(
        EXPECTED_BUNDLE_IDS
    ):
        raise ValueError("selected compact IDs are not an exact unique frozen role subset")
    root, manifest, manifest_receipt = _strict_manifest(
        directory,
        expected_sha256=expected_manifest_sha256,
    )
    by_id = {record["view_id"]: (index, record) for index, record in enumerate(manifest["views"])}
    observations: list[GaussianObservationField] = []
    cameras = []
    archive_receipts: list[dict[str, Any]] = []
    limits = reconstruction.BundleLoadLimits()
    for view_id in selected_ids:
        index, record = by_id[view_id]
        reconstruction._require_exact_keys(
            record,
            reconstruction._VIEW_KEYS,
            label=f"view record {index}",
        )
        reconstruction._validate_strict_camera_record(
            record["camera"], label=f"view record {index} camera"
        )
        expected_relative = f"teachers/{index:04d}.teacher.npz"
        if record["teacher"] != expected_relative:
            raise ValueError(f"teacher path for {view_id} is not canonical")
        teacher_path = reconstruction._strict_regular_file(
            root,
            expected_relative,
            label=f"teacher archive {index}",
        )
        sha256 = _sha256_file(teacher_path)
        if sha256 != record["teacher_sha256"]:
            raise ValueError(f"teacher archive digest mismatch for {view_id}")
        stats = reconstruction._inspect_npz(
            teacher_path,
            expected_relative,
            limits=limits,
            allowed_members=reconstruction._TEACHER_ZIP_MEMBERS,
        )
        observation = GaussianObservationField.load_npz(
            teacher_path,
            device="cpu",
            strict=True,
        )
        if observation.view_id != view_id:
            raise ValueError(f"teacher view ID mismatch for {view_id}")
        if observation.n != int(record["n_opt_2d"]) or observation.n_init != record["n_init_2d"]:
            raise ValueError(f"teacher cardinality mismatch for {view_id}")
        observations.append(observation)
        cameras.append(reconstruction._camera_from_record(record["camera"]).to("cpu"))
        archive_receipts.append(
            {
                "view_id": view_id,
                "path": str(teacher_path),
                "sha256": sha256,
                "compressed_bytes": stats.compressed_bytes,
                "uncompressed_bytes": stats.uncompressed_bytes,
                "members": stats.member_count,
            }
        )
    inputs = ReconstructionInputs(
        observations=observations,
        cameras=cameras,
        view_names=list(selected_ids),
        points=None,
        point_visibility=None,
        bounds_hint=None,
        name=str(manifest["name"]),
    )
    return DevelopmentLoad(
        inputs=inputs,
        receipt={
            **manifest_receipt,
            "selected_ids": list(selected_ids),
            "manifest_metadata_ids_read": list(EXPECTED_BUNDLE_IDS),
            "materialized_camera_ids": list(selected_ids),
            "teacher_geometry_ids_opened": list(selected_ids),
            "teacher_archives_opened": archive_receipts,
            "validation_geometry_opened": any(
                view_id in VALIDATION_IDS for view_id in selected_ids
            ),
            "report_rgb_or_mask_opened": False,
        },
    )


def load_development_inputs(
    directory: Path = BUNDLE,
    *,
    expected_manifest_sha256: str | None = FROZEN_MANIFEST_SHA256,
) -> DevelopmentLoad:
    return _load_selected_inputs(
        directory,
        DEVELOPMENT_IDS,
        expected_manifest_sha256=expected_manifest_sha256,
    )


def compute_development_bounds(inputs: ReconstructionInputs) -> Bounds:
    """Reproduce compact-Carve's camera-axis fallback from development cameras only."""

    if tuple(inputs.view_names) != DEVELOPMENT_IDS:
        raise ValueError("bounds require the frozen development-only camera order")
    if inputs.points is not None or inputs.bounds_hint is not None:
        raise ValueError("camera-axis fallback requires no points or bounds hint")
    dtype = torch.float64
    positions = torch.stack([camera.position for camera in inputs.cameras]).to(dtype)
    forwards = torch.stack([camera.R[2] for camera in inputs.cameras]).to(dtype)
    forwards = torch.nn.functional.normalize(forwards, dim=-1)
    eye = torch.eye(3, dtype=dtype)
    projectors = eye[None] - forwards[:, :, None] * forwards[:, None, :]
    matrix = projectors.sum(dim=0)
    vector = (projectors @ positions[:, :, None]).sum(dim=0)[:, 0]
    rank_aware = torch.linalg.lstsq(matrix, vector, driver="gelsd")
    center = (
        torch.linalg.lstsq(matrix, vector, driver="gels").solution
        if int(rank_aware.rank.item()) == 3
        else rank_aware.solution
    )
    radius = 0.25 * (positions - center).norm(dim=-1).median().clamp_min(1e-3)
    extent = float((2.2 * radius).clamp_min(1e-3))
    lower = center - 0.5 * extent
    upper = center + 0.5 * extent
    semantic = {
        "source": "camera_axis_fallback",
        "development_ids": list(DEVELOPMENT_IDS),
        "camera_positions_sha256": _tensor_hash(positions),
        "camera_forwards_sha256": _tensor_hash(forwards),
        "center": center.tolist(),
        "extent": extent,
        "near_depth": NEAR_DEPTH,
        "lower": lower.tolist(),
        "upper": upper.tolist(),
    }
    return Bounds(
        center=center,
        extent=extent,
        lower=lower,
        upper=upper,
        semantic_sha256=_sha256_bytes(_canonical_bytes(semantic)),
    )


def ray_box_camera_z_intervals(
    camera: Camera,
    xy: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Intersect camera-z parameterized world rays with an AABB."""

    near, far, valid = _ray_box_camera_z_intervals_with_validity(camera, xy, lower, upper)
    if not bool(valid.all()):
        raise RuntimeError("an anchor ray has an invalid or empty development AABB interval")
    return near, far


def _ray_box_camera_z_intervals_with_validity(
    camera: Camera,
    xy: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return every camera-z interval and the prospective-addendum validity mask."""

    if xy.ndim != 2 or xy.shape[1] != 2 or not xy.is_floating_point():
        raise ValueError("xy must have floating shape (N,2)")
    if lower.shape != (3,) or upper.shape != (3,) or bool((upper <= lower).any()):
        raise ValueError("AABB bounds must be finite ordered three-vectors")
    # Use the exact inverse of the stored world-to-camera matrix. Camera.pixel_rays uses R.T,
    # which is geometrically intended but can perturb the camera-z parameter under float32
    # calibration round-off. The fiber uses this same exact inverse for its source equality.
    origins, directions = _exact_camera_z_world_rays(camera, xy)
    direction_safe = torch.where(
        directions.abs() < 1e-9,
        torch.full_like(directions, 1e-9),
        directions,
    )
    entry = (lower.to(xy)[None, :] - origins) / direction_safe
    exit = (upper.to(xy)[None, :] - origins) / direction_safe
    raw_near = torch.minimum(entry, exit).max(dim=-1).values
    near = raw_near.clamp_min(NEAR_DEPTH)
    far = torch.maximum(entry, exit).min(dim=-1).values
    valid = torch.isfinite(raw_near) & torch.isfinite(far) & (far > near) & (near >= NEAR_DEPTH)
    return near, far, valid


def _exact_camera_z_world_rays(
    camera: Camera,
    xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """World rays whose scalar is exactly source camera-z under stored calibration."""

    rotation_inverse = torch.linalg.inv(camera.R.to(xy))
    origin = (-camera.t.to(xy)) @ rotation_inverse.T
    direction_camera = torch.stack(
        [
            (xy[:, 0] - camera.cx) / camera.fx,
            (xy[:, 1] - camera.cy) / camera.fy,
            torch.ones_like(xy[:, 0]),
        ],
        dim=-1,
    )
    directions = direction_camera @ rotation_inverse.T
    return origin.expand(xy.shape[0], -1), directions


def select_grid_anchors(
    field: GaussianObservationField,
    *,
    count: int = N_ANCHORS_PER_VIEW,
    eligible: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact 8x8/two-per-cell selection over ray/AABB-valid components."""

    if count != N_ANCHORS_PER_VIEW:
        raise ValueError("official real interaction requires exactly 128 anchors per view")
    if field.n < count:
        raise ValueError("field has fewer components than the frozen anchor count")
    if eligible is None:
        eligible = torch.ones(field.n, dtype=torch.bool, device=field.device)
    if (
        eligible.shape != (field.n,)
        or eligible.dtype != torch.bool
        or eligible.device != field.device
    ):
        raise ValueError("eligible must be a boolean field-sized tensor")
    if int(eligible.sum()) < count:
        raise RuntimeError("fewer than 128 ray/AABB-valid components remain in one view")
    means = field.native_means(dtype=torch.float64)
    variances = field.effective_variances().to(torch.float64)
    area = variances.prod(dim=-1).sqrt()
    global_order = torch.argsort(area, descending=True, stable=True)
    fit_x, fit_y, fit_width, fit_height = field.fit_window
    inside = field.valid_domain(means.to(field.dtype)) & eligible
    cell_x = torch.floor((means[:, 0] - fit_x) * GRID_SIZE / fit_width).long()
    cell_y = torch.floor((means[:, 1] - fit_y) * GRID_SIZE / fit_height).long()
    cell = cell_y * GRID_SIZE + cell_x
    selected: list[int] = []
    used = torch.zeros(field.n, dtype=torch.bool)
    for cell_id in range(GRID_SIZE * GRID_SIZE):
        candidates = global_order[(cell[global_order] == cell_id) & inside[global_order]]
        for index in candidates[:PER_CELL].tolist():
            selected.append(int(index))
            used[index] = True
    if len(selected) < count:
        fill = global_order[eligible[global_order] & ~used[global_order]]
        for index in fill.tolist():
            selected.append(int(index))
            used[index] = True
            if len(selected) == count:
                break
    if len(selected) < count:
        raise RuntimeError("anchor fill failed to reach exactly 128 unique components")
    result = torch.tensor(selected[:count], dtype=torch.long)
    if result.numel() != count or int(torch.unique(result).numel()) != count:
        raise RuntimeError("anchor selection is not an exact unique 128-index set")
    return result


def build_real_problem(inputs: ReconstructionInputs, bounds: Bounds) -> RealProblem:
    if tuple(inputs.view_names) != DEVELOPMENT_IDS:
        raise ValueError("real problem requires the frozen development views")
    if any(field.n != 640 for field in inputs.observations):
        raise ValueError("frozen compact payload requires 640 observations per development view")
    dilations = {float(field.aa_dilation) for field in inputs.observations}
    if len(dilations) != 1:
        raise ValueError("all compact fields must share one exact AA dilation")
    dilation = dilations.pop()
    per_view: list[torch.Tensor] = []
    source_views: list[torch.Tensor] = []
    source_components: list[torch.Tensor] = []
    source_means: list[torch.Tensor] = []
    source_covariances: list[torch.Tensor] = []
    lower: list[torch.Tensor] = []
    upper: list[torch.Tensor] = []
    valid_counts: list[int] = []
    all_near_parts: list[torch.Tensor] = []
    all_far_parts: list[torch.Tensor] = []
    all_valid_parts: list[torch.Tensor] = []
    for view_index, (field, camera) in enumerate(
        zip(inputs.observations, inputs.cameras, strict=True)
    ):
        adapted = ObservationGaussians.from_field(
            field,
            dtype=torch.float64,
            capacity_mode="uniform",
        )
        all_near, all_far, valid = _ray_box_camera_z_intervals_with_validity(
            camera,
            adapted.means,
            bounds.lower,
            bounds.upper,
        )
        valid_counts.append(int(valid.sum()))
        all_near_parts.append(all_near)
        all_far_parts.append(all_far)
        all_valid_parts.append(valid)
        indices = select_grid_anchors(field, eligible=valid)
        per_view.append(indices)
        means = adapted.means[indices]
        near = all_near[indices]
        far = all_far[indices]
        source_views.append(torch.full((indices.numel(),), view_index, dtype=torch.long))
        source_components.append(indices)
        source_means.append(means)
        source_covariances.append(adapted.covariances[indices])
        lower.append(near)
        upper.append(far)
    source_view_indices = torch.cat(source_views)
    source_component_indices = torch.cat(source_components)
    source_means2d = torch.cat(source_means)
    source_covariances2d = torch.cat(source_covariances)
    depth_lower = torch.cat(lower)
    depth_upper = torch.cat(upper)
    initial_depths = 0.5 * (depth_lower + depth_upper)
    anchors = AnchorSelection(
        source_view_indices=source_view_indices,
        source_component_indices=source_component_indices,
        source_means2d=source_means2d,
        source_covariances2d=source_covariances2d,
        depth_lower=depth_lower,
        depth_upper=depth_upper,
        initial_depths=initial_depths,
        per_view_indices=tuple(per_view),
        all_depth_lower=tuple(all_near_parts),
        all_depth_upper=tuple(all_far_parts),
        all_ray_valid=tuple(all_valid_parts),
    )
    if source_view_indices.numel() != N_TRACKS or any(count < 128 for count in valid_counts):
        raise RuntimeError("real problem did not create exactly 640 source fibers")
    observations_uniform = tuple(
        ObservationGaussians.from_field(field, dtype=torch.float64, capacity_mode="uniform")
        for field in inputs.observations
    )
    observations_area = tuple(
        ObservationGaussians.from_field(field, dtype=torch.float64, capacity_mode="footprint_area")
        for field in inputs.observations
    )
    track_area = torch.linalg.det(source_covariances2d).sqrt()
    return RealProblem(
        inputs=inputs,
        bounds=bounds,
        anchors=anchors,
        observations_uniform=observations_uniform,
        observations_area=observations_area,
        track_capacities_uniform=torch.ones(N_TRACKS, dtype=torch.float64),
        track_capacities_area=track_area,
        dilation=dilation,
    )


def fresh_fiber(problem: RealProblem) -> InverseProjectionFiber:
    anchor = problem.anchors
    return InverseProjectionFiber(
        cameras=tuple(problem.inputs.cameras),
        source_view_indices=anchor.source_view_indices,
        source_component_indices=anchor.source_component_indices,
        source_means2d=anchor.source_means2d,
        source_covariances2d=anchor.source_covariances2d,
        initial_depths=anchor.initial_depths,
        depth_lower=anchor.depth_lower,
        depth_upper=anchor.depth_upper,
        dilation=problem.dilation,
    )


def official_fit_config() -> FiberFitConfig:
    return FiberFitConfig(
        temperatures=exponential_schedule(2.0, 0.10, 20),
        residual_variances=exponential_schedule(1.0, 0.05, 20),
        geometry_steps=2,
        learning_rate=0.025,
        assignment="unbalanced_sinkhorn",
        dustbin_cost=4.0,
        marginal_penalty=1.0,
        sinkhorn_iterations=50,
        sinkhorn_tolerance=0.0,
        track_batch_size=128,
        source_center_tolerance=1e-4,
        source_covariance_tolerance=1e-6,
    )


def symmetric_set_costs(
    fiber: InverseProjectionFiber,
    observations: tuple[ObservationGaussians, ...],
    *,
    cameras: tuple[Camera, ...] | list[Camera] | None = None,
    residual_variance: float = 0.05,
) -> tuple[float, ...]:
    """Symmetric Gaussian-set costs in explicitly supplied evaluation cameras.

    Fibers behind the EWA near plane or carrying non-finite projected geometry are excluded;
    otherwise EWA's near-clamped covariance would make an invalid projection look matchable.
    """

    active_cameras = tuple(fiber.cameras if cameras is None else cameras)
    if len(active_cameras) != len(observations):
        raise ValueError("evaluation cameras and observations must have equal length")
    values: list[float] = []
    with torch.no_grad():
        for camera, target in zip(active_cameras, observations, strict=True):
            projected = fiber.project(camera)
            valid = (
                (projected.depth > EWA_NEAR)
                & torch.isfinite(projected.means2d).all(dim=-1)
                & torch.isfinite(projected.covariances2d).all(dim=(-2, -1))
            )
            if not bool(valid.any()):
                raise RuntimeError("evaluation camera has no valid projected fiber")
            cost = pairwise_bhattacharyya_cost(
                projected.means2d[valid],
                projected.covariances2d[valid],
                target.means,
                target.covariances,
                residual_variance=residual_variance,
            )
            symmetric = 0.5 * (cost.amin(dim=1).mean() + cost.amin(dim=0).mean())
            values.append(float(symmetric))
    return tuple(values)


def _conditional_real_entropy(plan: CorrespondencePlan) -> torch.Tensor:
    real = plan.real_mass
    total = real.sum(dim=1)
    probability = real / total.clamp_min(torch.finfo(real.dtype).tiny)[:, None]
    entropy = -(probability * probability.clamp_min(torch.finfo(real.dtype).tiny).log()).sum(dim=1)
    return torch.where(total > 0, entropy, torch.zeros_like(entropy))


def _marginal_relative_error(
    realized: torch.Tensor,
    target: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if realized.shape != target.shape:
        raise RuntimeError(f"{name} realized and declared marginals have different shapes")
    absolute = (realized - target).abs()
    positive = target > 0
    if bool((absolute[~positive] != 0).any()):
        raise RuntimeError(f"{name} placed mass on a structurally zero declared marginal")
    relative = torch.zeros_like(absolute)
    relative[positive] = absolute[positive] / target[positive]
    return relative


def _uot_marginal_diagnostics(
    plan: CorrespondencePlan,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Return complete, diagnostic-only accounting for one augmented UOT plan."""

    if plan.method != "unbalanced_sinkhorn":
        raise RuntimeError(f"real interaction expected UOT, got {plan.method!r}")
    if plan.fixed_point_residual is None or not math.isfinite(plan.fixed_point_residual):
        raise RuntimeError("final UOT plan is missing a finite fixed-point residual")
    augmented = plan.augmented_mass
    targets = plan.augmented_target_marginals
    if augmented is None or targets is None:
        raise RuntimeError("final UOT plan is missing augmented mass or declared marginals")
    declared_row, declared_column = targets
    realized_row = augmented.sum(dim=1)
    realized_column = augmented.sum(dim=0)
    row_absolute = (realized_row - declared_row).abs()
    column_absolute = (realized_column - declared_column).abs()
    row_relative = _marginal_relative_error(realized_row, declared_row, name="row")
    column_relative = _marginal_relative_error(
        realized_column,
        declared_column,
        name="column",
    )
    declared_total = declared_row.sum()
    realized_total = augmented.sum()
    if not bool(torch.isfinite(declared_total)) or not bool(declared_total > 0):
        raise RuntimeError("declared augmented UOT mass is not finite and positive")
    if not bool(torch.isfinite(realized_total)) or not bool(realized_total > 0):
        raise RuntimeError("realized augmented UOT mass is not finite and positive")
    retention = realized_total / declared_total
    arrays = {
        "declared_augmented_row_target": declared_row,
        "declared_augmented_column_target": declared_column,
        "realized_augmented_row_total": realized_row,
        "realized_augmented_column_total": realized_column,
        "augmented_row_absolute_error": row_absolute,
        "augmented_column_absolute_error": column_absolute,
        "augmented_row_relative_error": row_relative,
        "augmented_column_relative_error": column_relative,
        "fixed_point_residual": declared_row.new_tensor([plan.fixed_point_residual]),
        "total_augmented_retention": retention.reshape(1),
    }
    summary = {
        "fixed_point_residual": plan.fixed_point_residual,
        "declared_augmented_total": float(declared_total),
        "realized_augmented_total": float(realized_total),
        "total_augmented_retention": float(retention),
        "row_max_absolute_error": float(row_absolute.max()),
        "column_max_absolute_error": float(column_absolute.max()),
        "max_absolute_marginal_error": float(
            torch.maximum(row_absolute.max(), column_absolute.max())
        ),
        "row_max_relative_error": float(row_relative.max()),
        "column_max_relative_error": float(column_relative.max()),
        "max_relative_marginal_error": float(
            torch.maximum(row_relative.max(), column_relative.max())
        ),
    }
    return summary, arrays


def validate_final_arm_state(
    fiber: InverseProjectionFiber,
    observations: tuple[ObservationGaussians, ...],
    plans: tuple[CorrespondencePlan, ...],
    *,
    track_capacities: torch.Tensor,
    config: FiberFitConfig,
) -> dict[str, Any]:
    """Fail closed on structural state/plan invariants, without outcome thresholds."""

    if config.assignment != "unbalanced_sinkhorn":
        raise RuntimeError("real interaction final validation requires UOT assignment")
    if len(plans) != len(fiber.cameras) or len(observations) != len(fiber.cameras):
        raise RuntimeError("final plans and observations must match the calibrated cameras")
    if track_capacities.shape != (fiber.n,):
        raise RuntimeError("final track capacities do not match the fiber cardinality")
    validate_fiber_state(fiber)
    depths = fiber.depths()
    _means, covariances = fiber.means_covariances()
    eigenvalues = torch.linalg.eigvalsh(covariances)
    if not bool(torch.isfinite(eigenvalues).all()) or not bool((eigenvalues > 0).all()):
        raise RuntimeError("final fiber covariance spectrum is not finite and positive")

    total_real_mass = 0.0
    supported_projection_counts: list[int] = []
    for view_index, (camera, target, plan) in enumerate(
        zip(fiber.cameras, observations, plans, strict=True)
    ):
        if plan.real_mass.shape != (fiber.n, target.n):
            raise RuntimeError("final UOT real-mass shape does not match tracks and observations")
        source = fiber.source_view_indices == view_index
        active = ~source
        if bool((plan.real_mass[source] != 0).any()):
            raise RuntimeError("source-view rows carried forbidden real correspondence mass")
        if bool((plan.track_dustbin_mass[source] != 0).any()):
            raise RuntimeError("source-view rows carried forbidden dustbin mass")
        if bool((plan.track_capacities[source] != 0).any()):
            raise RuntimeError("source-view rows retained forbidden UOT capacity")
        if not torch.equal(plan.track_capacities[active], track_capacities[active]):
            raise RuntimeError("active UOT track capacities changed during plan construction")
        if plan.observation_capacities is None or not torch.equal(
            plan.observation_capacities,
            target.capacities,
        ):
            raise RuntimeError("UOT observation capacities changed during plan construction")
        if config.sinkhorn_tolerance == 0 and plan.iterations != config.sinkhorn_iterations:
            raise RuntimeError(
                "fixed-iteration UOT plan did not execute the frozen iteration count"
            )
        _uot_marginal_diagnostics(plan)
        projected = fiber.project(camera)
        _safe_means, _safe_covariances, valid = safe_projection_geometry(camera, projected)
        supported = plan.real_mass.sum(dim=1) > 0
        if bool((supported & ~valid).any()):
            raise RuntimeError("a final supported projection is outside the shared valid domain")
        supported_projection_counts.append(int(supported.sum()))
        total_real_mass += float(plan.real_mass.sum())
    if not math.isfinite(total_real_mass) or total_real_mass < config.min_real_mass:
        raise RuntimeError("final UOT real mass is below the configured fail-closed minimum")

    lower_margin = depths - fiber.depth_lower
    upper_margin = fiber.depth_upper - depths
    return {
        "passed": True,
        "raw_parameter_finiteness": True,
        "strict_depth_interior": True,
        "world_covariances_spd": True,
        "minimum_depth_margin": float(torch.minimum(lower_margin, upper_margin).min().detach()),
        "minimum_covariance_eigenvalue": float(eigenvalues.min().detach()),
        "total_final_real_mass": total_real_mass,
        "supported_projection_counts": supported_projection_counts,
        "plan_count": len(plans),
    }


def summarize_plans(
    fiber: InverseProjectionFiber,
    plans: tuple[CorrespondencePlan, ...],
    *,
    extent: float,
) -> dict[str, Any]:
    support_counts = torch.zeros(fiber.n, dtype=torch.long)
    rows: list[dict[str, Any]] = []
    for view_index, plan in enumerate(plans):
        active = fiber.source_view_indices != view_index
        support = plan.track_support
        support_counts += (active & (support >= 0.20)).long()
        conditional = _conditional_real_entropy(plan)
        uot, _uot_arrays = _uot_marginal_diagnostics(plan)
        rows.append(
            {
                "view_id": DEVELOPMENT_IDS[view_index],
                "real_mass": float(plan.real_mass.sum()),
                "track_dustbin_mass": float(plan.track_dustbin_mass.sum()),
                "observation_dustbin_mass": (
                    None
                    if plan.observation_dustbin_mass is None
                    else float(plan.observation_dustbin_mass.sum())
                ),
                "dustbin_dustbin_mass": (
                    None if plan.dustbin_dustbin_mass is None else float(plan.dustbin_dustbin_mass)
                ),
                "mean_active_track_support": float(support[active].mean()),
                "mean_active_conditional_entropy": float(conditional[active].mean()),
                "sinkhorn_iterations": plan.iterations,
                "uot": uot,
            }
        )
    means, _ = fiber.means_covariances()
    retained = torch.arange(fiber.n, dtype=torch.long)
    components = radius_connected_components(
        means.detach(),
        retained,
        radius=0.01 * extent,
    )
    return {
        "per_view": rows,
        "support_view_histogram": torch.bincount(
            support_counts, minlength=len(DEVELOPMENT_IDS)
        ).tolist(),
        "support_view_counts": support_counts.tolist(),
        "effective_supported_track_count": int((support_counts >= 2).sum()),
        "effective_supported_track_fraction": float((support_counts >= 2).double().mean()),
        "cluster_radius": 0.01 * extent,
        "proximity_cluster_count": len(components),
        "proximity_cluster_sizes": [len(component) for component in components],
    }


def epipolar_residual_diagnostics(
    fiber: InverseProjectionFiber,
    observations: tuple[ObservationGaussians, ...],
    plans: tuple[CorrespondencePlan, ...],
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    """Association-weighted residuals split along the source epipolar direction."""

    means3d, _ = fiber.means_covariances()
    source_directions = means3d.new_empty((fiber.n, 3))
    for source_index, camera in enumerate(fiber.cameras):
        rows = (fiber.source_view_indices == source_index).nonzero(as_tuple=True)[0]
        _origins, directions = _exact_camera_z_world_rays(
            camera,
            fiber.source_means2d[rows],
        )
        source_directions[rows] = directions.to(means3d)
    reports: list[dict[str, Any]] = []
    array_parts: dict[str, torch.Tensor] = {}
    for view_index, (camera, target, plan) in enumerate(
        zip(fiber.cameras, observations, plans, strict=True)
    ):
        projected = fiber.project(camera)
        cam_points = camera.world_to_cam(means3d)
        cam_direction = source_directions @ camera.R.to(source_directions).T
        z = cam_points[:, 2].clamp_min(1e-9)
        du = camera.fx * (cam_direction[:, 0] * z - cam_points[:, 0] * cam_direction[:, 2])
        dv = camera.fy * (cam_direction[:, 1] * z - cam_points[:, 1] * cam_direction[:, 2])
        epipolar = torch.stack([du, dv], dim=-1) / z.square()[:, None]
        epi_norm = epipolar.norm(dim=-1)
        valid_direction = epi_norm > 1e-12
        epipolar = epipolar / epi_norm.clamp_min(1e-12)[:, None]
        flip = (epipolar[:, 0] < 0) | ((epipolar[:, 0].abs() <= 1e-12) & (epipolar[:, 1] < 0))
        epipolar = torch.where(flip[:, None], -epipolar, epipolar)
        perpendicular = torch.stack([-epipolar[:, 1], epipolar[:, 0]], dim=-1)
        residual = target.means[None, :, :] - projected.means2d[:, None, :]
        parallel = (residual * epipolar[:, None, :]).sum(dim=-1)
        normal = (residual * perpendicular[:, None, :]).sum(dim=-1)
        active = (fiber.source_view_indices != view_index) & valid_direction
        weight = plan.real_mass * active[:, None]
        mass = weight.sum()
        target_std = torch.sqrt(
            0.5 * torch.diagonal(target.covariances, dim1=-2, dim2=-1).sum(dim=-1)
        )
        pooled_std = torch.sqrt(
            (weight * target_std.square()[None, :]).sum()
            / mass.clamp_min(torch.finfo(weight.dtype).tiny)
        )
        mean_parallel = (weight * parallel).sum() / mass.clamp_min(torch.finfo(weight.dtype).tiny)
        mean_perpendicular = (weight * normal).sum() / mass.clamp_min(
            torch.finfo(weight.dtype).tiny
        )
        rms_parallel = torch.sqrt(
            (weight * parallel.square()).sum() / mass.clamp_min(torch.finfo(weight.dtype).tiny)
        )
        rms_perpendicular = torch.sqrt(
            (weight * normal.square()).sum() / mass.clamp_min(torch.finfo(weight.dtype).tiny)
        )
        normalized_systematic = mean_perpendicular.abs() / pooled_std.clamp_min(
            torch.finfo(weight.dtype).tiny
        )
        reports.append(
            {
                "view_id": DEVELOPMENT_IDS[view_index],
                "real_mass": float(mass),
                "valid_epipolar_track_count": int(active.sum()),
                "mean_parallel_px": float(mean_parallel),
                "mean_perpendicular_px": float(mean_perpendicular),
                "rms_parallel_px": float(rms_parallel),
                "rms_perpendicular_px": float(rms_perpendicular),
                "pooled_target_std_px": float(pooled_std),
                "systematic_perpendicular_in_target_std": float(normalized_systematic),
                "camera_or_model_mismatch_flag": bool(normalized_systematic > 0.25),
            }
        )
        prefix = f"view_{view_index:02d}"
        array_parts[f"{prefix}_epipolar_direction"] = epipolar.detach()
        array_parts[f"{prefix}_parallel_residual"] = parallel.detach()
        array_parts[f"{prefix}_perpendicular_residual"] = normal.detach()
        array_parts[f"{prefix}_transport_mass"] = weight.detach()
    return reports, array_parts


def _fiber_arrays(fiber: InverseProjectionFiber) -> dict[str, torch.Tensor]:
    means, covariances = fiber.means_covariances()
    return {
        "source_view_indices": fiber.source_view_indices,
        "source_component_indices": fiber.source_component_indices,
        "source_means2d": fiber.source_means2d,
        "source_covariances2d": fiber.source_covariances2d,
        "depth_lower": fiber.depth_lower,
        "depth_upper": fiber.depth_upper,
        "depth_logits": fiber.depth_logits,
        "depths": fiber.depths(),
        "cross": fiber.cross,
        "log_ray_scale": fiber.log_ray_scale,
        "means3d": means,
        "covariances3d": covariances,
    }


def _plan_arrays(plans: tuple[CorrespondencePlan, ...]) -> dict[str, torch.Tensor]:
    arrays: dict[str, torch.Tensor] = {}
    for view_index, plan in enumerate(plans):
        prefix = f"view_{view_index:02d}"
        arrays[f"{prefix}_real_mass"] = plan.real_mass
        arrays[f"{prefix}_track_dustbin_mass"] = plan.track_dustbin_mass
        arrays[f"{prefix}_track_capacities"] = plan.track_capacities
        arrays[f"{prefix}_track_support"] = plan.track_support
        arrays[f"{prefix}_track_entropy"] = plan.track_entropy
        arrays[f"{prefix}_conditional_real_entropy"] = _conditional_real_entropy(plan)
        if plan.observation_dustbin_mass is not None:
            arrays[f"{prefix}_observation_dustbin_mass"] = plan.observation_dustbin_mass
        if plan.observation_capacities is not None:
            arrays[f"{prefix}_observation_capacities"] = plan.observation_capacities
        if plan.dustbin_dustbin_mass is not None:
            arrays[f"{prefix}_dustbin_dustbin_mass"] = plan.dustbin_dustbin_mass.reshape(1)
        _uot_summary, uot_arrays = _uot_marginal_diagnostics(plan)
        for name, value in uot_arrays.items():
            arrays[f"{prefix}_{name}"] = value
    return arrays


def _arm_observations(
    problem: RealProblem,
    name: str,
) -> tuple[tuple[ObservationGaussians, ...], torch.Tensor]:
    if name == "uot_uniform":
        return problem.observations_uniform, problem.track_capacities_uniform
    if name == "uot_area":
        return problem.observations_area, problem.track_capacities_area
    raise ValueError(f"unknown real arm {name!r}")


def run_arm(
    problem: RealProblem,
    name: str,
    output: Path,
    *,
    config: FiberFitConfig | None = None,
) -> ArmState:
    observations, capacities = _arm_observations(problem, name)
    fiber = fresh_fiber(problem)
    initial_costs = symmetric_set_costs(fiber, observations)
    start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    fit = fit_fiber_correspondence(
        fiber,
        observations,
        config=official_fit_config() if config is None else config,
        track_capacities=capacities,
    )
    elapsed = time.perf_counter() - start
    peak_rss = max(start_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    fail_closed_validation = validate_final_arm_state(
        fiber,
        observations,
        fit.plans,
        track_capacities=capacities,
        config=official_fit_config() if config is None else config,
    )
    final_costs = symmetric_set_costs(fiber, observations)
    plan_summary = summarize_plans(fiber, fit.plans, extent=problem.bounds.extent)
    epipolar, epipolar_arrays = epipolar_residual_diagnostics(
        fiber,
        observations,
        fit.plans,
    )
    state_path = output / f"{name}_geometry.npz"
    plan_path = output / f"{name}_plans.npz"
    epipolar_path = output / f"{name}_epipolar.npz"
    ply_path = output / f"{name}.ply"
    artifacts = {
        "geometry": _write_npz_exclusive(state_path, _fiber_arrays(fiber)),
        "plans": _write_npz_exclusive(plan_path, _plan_arrays(fit.plans)),
        "epipolar": _write_npz_exclusive(epipolar_path, epipolar_arrays),
        "ply": _save_ply_exclusive(
            ply_path,
            fiber.as_gaussians(opacity=FIXED_OPACITY),
        ),
    }
    means, covariances = fiber.means_covariances()
    diagnostics = {
        "source_center_error": fit.source_center_error,
        "source_covariance_relative_error": fit.source_covariance_relative_error,
        "all_finite": bool(
            torch.isfinite(means).all()
            and torch.isfinite(covariances).all()
            and all(torch.isfinite(parameter).all() for parameter in fiber.parameters())
        ),
        "all_depths_strictly_bounded": bool(
            ((fiber.depths() > fiber.depth_lower) & (fiber.depths() < fiber.depth_upper)).all()
        ),
        "primitive_count": fiber.n,
        "fail_closed_validation": fail_closed_validation,
        "plan": plan_summary,
        "epipolar": epipolar,
        "history": [asdict(step) for step in fit.history],
    }
    return ArmState(
        name=name,
        fiber=fiber,
        fit=fit,
        elapsed_seconds=elapsed,
        peak_rss_kib=int(peak_rss),
        initial_set_costs=initial_costs,
        final_set_costs=final_costs,
        diagnostics=diagnostics,
        artifacts=artifacts,
    )


def _freeze_receipt(
    output: Path,
    name: str,
    descriptors: dict[str, dict[str, Any]],
    *,
    parent_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "namespace": NAMESPACE,
        "kind": name,
        "created_unix_ns": time.time_ns(),
        "parent": (
            None
            if parent_receipt is None
            else {"path": parent_receipt["path"], "sha256": parent_receipt["sha256"]}
        ),
        "files": descriptors,
    }
    path = output / f"{name}_freeze.json"
    descriptor = _write_json_exclusive(path, payload)
    return {"path": str(path), "sha256": descriptor["sha256"], "payload": payload}


def verify_freeze_receipt(receipt: dict[str, Any], *, expected_kind: str) -> None:
    path = Path(receipt["path"])
    if _sha256_file(path) != receipt["sha256"]:
        raise RuntimeError(f"{expected_kind} freeze receipt hash mismatch")
    payload = json.loads(path.read_bytes())
    if payload.get("namespace") != NAMESPACE or payload.get("kind") != expected_kind:
        raise RuntimeError(f"{expected_kind} freeze receipt identity mismatch")
    parent = payload.get("parent")
    if parent is not None:
        verify_freeze_receipt(parent, expected_kind="geometry")
    for descriptor in payload.get("files", {}).values():
        artifact = Path(descriptor["path"])
        if _sha256_file(artifact) != descriptor["sha256"]:
            raise RuntimeError(f"frozen artifact changed after {expected_kind}: {artifact}")


def _appearance_targets(
    fiber: InverseProjectionFiber,
    inputs: ReconstructionInputs,
    plans: tuple[CorrespondencePlan, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    means3d, _ = fiber.means_covariances()
    view_directions: list[torch.Tensor] = []
    target_colors: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for camera, field, plan in zip(
        inputs.cameras,
        inputs.observations,
        plans,
        strict=True,
    ):
        projected = fiber.project(camera)
        query = field.query(projected.means2d.to(field.dtype)).color.to(means3d)
        valid = field.valid_domain(projected.means2d.to(field.dtype)).to(means3d)
        view_directions.append(means3d - camera.position.to(means3d))
        target_colors.append(query)
        weights.append(plan.track_support.to(means3d) * valid)
    source_colors = means3d.new_empty((fiber.n, 3))
    source_directions = means3d.new_empty((fiber.n, 3))
    for source_index, (camera, field) in enumerate(
        zip(inputs.cameras, inputs.observations, strict=True)
    ):
        rows = (fiber.source_view_indices == source_index).nonzero(as_tuple=True)[0]
        source_query = field.query(fiber.source_means2d[rows].to(field.dtype))
        if not bool(source_query.valid.all()):
            raise RuntimeError("a selected source anchor lies outside its exact compact fit window")
        source_colors[rows] = source_query.color.to(means3d)
        source_directions[rows] = means3d[rows] - camera.position.to(means3d)
    return (
        source_directions,
        source_colors,
        torch.stack(view_directions),
        torch.stack(target_colors),
        torch.stack(weights),
    )


def _gaussians_with_sh(
    fiber: InverseProjectionFiber,
    coefficients: torch.Tensor,
) -> Gaussians3D:
    base = fiber.as_gaussians(opacity=FIXED_OPACITY).with_sh_degree(1)
    if coefficients.shape != (fiber.n, 4, 3):
        raise ValueError("degree-one SH coefficients must have shape (N,4,3)")
    return Gaussians3D(
        means=base.means,
        quats=base.quats,
        log_scales=base.log_scales,
        opacity=torch.full_like(base.opacity, FIXED_OPACITY),
        sh=coefficients.to(base.sh),
    )


def _appearance_metrics(
    coefficients: torch.Tensor,
    view_directions: torch.Tensor,
    target_colors: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, Any]:
    views, count = weights.shape
    flat_coefficients = coefficients[None].expand(views, -1, -1, -1).reshape(views * count, 4, 3)
    unit = view_directions.reshape(-1, 3)
    unit = unit / unit.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(unit.dtype).tiny)
    predicted = eval_sh_preactivation(1, flat_coefficients, unit).reshape(views, count, 3)
    squared = (predicted - target_colors).square().mean(dim=-1)
    denominator = weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
    weighted_rmse = torch.sqrt((weights * squared).sum() / denominator)
    per_view = []
    for view_index in range(views):
        local_denominator = weights[view_index].sum().clamp_min(torch.finfo(weights.dtype).tiny)
        per_view.append(
            float(torch.sqrt((weights[view_index] * squared[view_index]).sum() / local_denominator))
        )
    return {
        "weighted_query_color_rmse": float(weighted_rmse),
        "per_view_weighted_query_color_rmse": per_view,
        "negative_preactivation_fraction": float((predicted < 0).double().mean()),
        "prediction_min": float(predicted.min()),
        "prediction_max": float(predicted.max()),
    }


def fit_appearance(
    arm: ArmState,
    inputs: ReconstructionInputs,
    output: Path,
) -> AppearanceState:
    (
        source_directions,
        source_colors,
        view_directions,
        target_colors,
        weights,
    ) = _appearance_targets(arm.fiber, inputs, arm.fit.plans)
    fit = fit_source_anchored_sh(
        source_directions=source_directions,
        source_colors=source_colors,
        view_directions=view_directions,
        target_colors=target_colors,
        weights=weights.detach(),
        source_view_indices=arm.fiber.source_view_indices,
        config=SourceAnchoredSHFitConfig(
            degree=1,
            iterations=100,
            learning_rate=0.05,
        ),
    )
    coefficients = fit.model.coefficients().detach()
    gaussians = _gaussians_with_sh(arm.fiber, coefficients)
    coeff_path = output / "source_anchored_sh.npz"
    final_path = output / "gaussians.ply"
    artifacts = {
        "coefficients": _write_npz_exclusive(
            coeff_path,
            {
                "coefficients": coefficients,
                "source_preactivation_targets": source_colors,
                "source_directions": source_directions,
                "view_directions": view_directions,
                "target_colors": target_colors,
                "support_weights": weights,
                "losses": np.asarray(fit.losses, dtype=np.float64),
            },
        ),
        "final_ply": _save_ply_exclusive(final_path, gaussians),
    }
    metrics = _appearance_metrics(coefficients, view_directions, target_colors, weights)
    metrics.update(
        {
            "source_preactivation_max_abs_error": fit.source_max_abs_error,
            "source_constraint_semantics": (
                "exact SH preactivation only; post-activation rendered color is not claimed exact"
            ),
            "source_post_activation_color_claimed_exact": False,
            "opacity": FIXED_OPACITY,
            "amplitudes_used_as_opacity": False,
            "amplitudes_used_as_confidence": False,
        }
    )
    return AppearanceState(
        fit=fit,
        source_colors=source_colors,
        coefficients=coefficients,
        dev_metrics=metrics,
        gaussians=gaussians,
        artifacts=artifacts,
    )


def release_validation_inputs(
    directory: Path,
    freeze_receipt: dict[str, Any],
    *,
    expected_manifest_sha256: str | None = FROZEN_MANIFEST_SHA256,
) -> DevelopmentLoad:
    """Strict-load the complete bundle only after the model freeze verifies."""

    verify_freeze_receipt(freeze_receipt, expected_kind="appearance")
    _root, manifest, manifest_receipt = _strict_manifest(
        directory,
        expected_sha256=expected_manifest_sha256,
    )
    full = ReconstructionInputs.load(directory, device="cpu", strict=True)
    if tuple(full.view_names) != EXPECTED_BUNDLE_IDS:
        raise RuntimeError("strict validation release loaded unexpected view IDs")
    indices = [full.view_names.index(view_id) for view_id in VALIDATION_IDS]
    validation = ReconstructionInputs(
        observations=[full.observations[index] for index in indices],
        cameras=[full.cameras[index] for index in indices],
        view_names=list(VALIDATION_IDS),
        points=None,
        point_visibility=None,
        bounds_hint=None,
        name=full.name,
    )
    archive_stats = full.archive_stats
    return DevelopmentLoad(
        inputs=validation,
        receipt={
            **manifest_receipt,
            "release_kind": "validation_after_appearance_freeze",
            "appearance_freeze_sha256": freeze_receipt["sha256"],
            "all_opened_ids": list(EXPECTED_BUNDLE_IDS),
            "returned_ids": list(VALIDATION_IDS),
            "validation_teacher_sha256": {
                record["view_id"]: record["teacher_sha256"]
                for record in manifest["views"]
                if record["view_id"] in VALIDATION_IDS
            },
            "validation_geometry_opened": True,
            "report_rgb_or_mask_opened": False,
            "strict_total_compressed_bytes": (
                None if archive_stats is None else archive_stats.total_compressed_bytes
            ),
            "strict_total_uncompressed_bytes": (
                None if archive_stats is None else archive_stats.total_uncompressed_bytes
            ),
        },
    )


def _validation_geometry_metrics(
    initial: InverseProjectionFiber,
    arm: ArmState,
    validation: ReconstructionInputs,
) -> dict[str, Any]:
    observations = tuple(
        ObservationGaussians.from_field(field, dtype=torch.float64, capacity_mode="uniform")
        for field in validation.observations
    )
    initial_costs = symmetric_set_costs(
        initial,
        observations,
        cameras=validation.cameras,
    )
    final_costs = symmetric_set_costs(
        arm.fiber,
        observations,
        cameras=validation.cameras,
    )
    improvements = [
        (before - after) / max(before, torch.finfo(torch.float64).tiny)
        for before, after in zip(initial_costs, final_costs, strict=True)
    ]
    mean_initial = float(np.mean(initial_costs))
    mean_final = float(np.mean(final_costs))
    mean_improvement = (mean_initial - mean_final) / max(
        mean_initial, torch.finfo(torch.float64).tiny
    )
    return {
        "view_ids": list(VALIDATION_IDS),
        "initial_symmetric_set_costs": list(initial_costs),
        "final_symmetric_set_costs": list(final_costs),
        "per_view_relative_improvements": improvements,
        "mean_initial_symmetric_set_cost": mean_initial,
        "mean_final_symmetric_set_cost": mean_final,
        "mean_relative_improvement": mean_improvement,
    }


def _validation_appearance_metrics(
    arm: ArmState,
    appearance: AppearanceState,
    validation: ReconstructionInputs,
) -> dict[str, Any]:
    means3d, _ = arm.fiber.means_covariances()
    directions = []
    colors = []
    weights = []
    for camera, field in zip(validation.cameras, validation.observations, strict=True):
        projection = arm.fiber.project(camera)
        query = field.query(projection.means2d.to(field.dtype))
        directions.append(means3d - camera.position.to(means3d))
        colors.append(query.color.to(means3d))
        weights.append(query.valid.to(means3d))
    return _appearance_metrics(
        appearance.coefficients,
        torch.stack(directions),
        torch.stack(colors),
        torch.stack(weights),
    )


def geometry_gate(
    arm: ArmState,
    validation_metrics: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "source_center": arm.fit.source_center_error <= 1e-4,
        "source_covariance": arm.fit.source_covariance_relative_error <= 1e-6,
        "finite": arm.diagnostics["all_finite"],
        "strict_depth_bounds": arm.diagnostics["all_depths_strictly_bounded"],
        "validation_mean_improvement": validation_metrics["mean_relative_improvement"] >= 0.05,
        "validation_no_large_regression": all(
            improvement >= -0.10
            for improvement in validation_metrics["per_view_relative_improvements"]
        ),
        "supported_tracks": (arm.diagnostics["plan"]["effective_supported_track_fraction"] >= 0.50),
    }
    return {"pass": all(checks.values()), "checks": checks}


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
    }


def _protocol_receipt() -> dict[str, Any]:
    if not PREREG_DOCUMENTS:
        raise RuntimeError("Iteration 3 protocol document tuple is empty")
    documents: list[dict[str, str]] = []
    for path, expected in PREREG_DOCUMENTS:
        observed = _sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                "Iteration 3 protocol document hash mismatch: "
                f"{path}; expected {expected}, observed {observed}"
            )
        documents.append({"path": str(path), "sha256": observed})
    return {"base": documents[0], "addenda": documents[1:]}


def _input_arrays(problem: RealProblem) -> dict[str, torch.Tensor]:
    arrays: dict[str, torch.Tensor] = {
        "bounds_center": problem.bounds.center,
        "bounds_lower": problem.bounds.lower,
        "bounds_upper": problem.bounds.upper,
        "bounds_extent": torch.tensor([problem.bounds.extent], dtype=torch.float64),
        "near_depth": torch.tensor([NEAR_DEPTH], dtype=torch.float64),
        "source_view_indices": problem.anchors.source_view_indices,
        "source_component_indices": problem.anchors.source_component_indices,
        "source_means2d": problem.anchors.source_means2d,
        "source_covariances2d": problem.anchors.source_covariances2d,
        "depth_lower": problem.anchors.depth_lower,
        "depth_upper": problem.anchors.depth_upper,
        "initial_depths": problem.anchors.initial_depths,
    }
    for view_index, (camera, observation, selected) in enumerate(
        zip(
            problem.inputs.cameras,
            problem.observations_uniform,
            problem.anchors.per_view_indices,
            strict=True,
        )
    ):
        prefix = f"view_{view_index:02d}"
        arrays[f"{prefix}_camera_R"] = camera.R
        arrays[f"{prefix}_camera_t"] = camera.t
        arrays[f"{prefix}_camera_intrinsics"] = torch.tensor(
            [camera.fx, camera.fy, camera.cx, camera.cy],
            dtype=torch.float64,
        )
        arrays[f"{prefix}_means"] = observation.means
        arrays[f"{prefix}_covariances"] = observation.covariances
        arrays[f"{prefix}_selected_components"] = selected
        arrays[f"{prefix}_all_depth_lower"] = problem.anchors.all_depth_lower[view_index]
        arrays[f"{prefix}_all_depth_upper"] = problem.anchors.all_depth_upper[view_index]
        arrays[f"{prefix}_all_ray_valid"] = problem.anchors.all_ray_valid[view_index]
    return arrays


def _anchor_evidence_summary(problem: RealProblem, artifact: dict[str, Any]) -> dict[str, Any]:
    per_view = []
    for view_index, view_id in enumerate(DEVELOPMENT_IDS):
        valid = problem.anchors.all_ray_valid[view_index]
        per_view.append(
            {
                "view_id": view_id,
                "target_observation_count": int(valid.numel()),
                "valid_interval_count": int(valid.sum()),
                "valid_interval_fraction": float(valid.double().mean()),
                "non_intersecting_target_fraction": float((~valid).double().mean()),
                "selected_count": int(problem.anchors.per_view_indices[view_index].numel()),
                "selected_indices_sha256": _tensor_hash(
                    problem.anchors.per_view_indices[view_index]
                ),
                "eligibility_sha256": _tensor_hash(valid),
                "near_endpoints_sha256": _tensor_hash(problem.anchors.all_depth_lower[view_index]),
                "far_endpoints_sha256": _tensor_hash(problem.anchors.all_depth_upper[view_index]),
            }
        )
    return {
        "rule": "valid camera-z interval before stable 8x8/two-per-cell area selection",
        "near_depth": NEAR_DEPTH,
        "minimum_valid_per_view": N_ANCHORS_PER_VIEW,
        "per_view": per_view,
        "evidence_npz_semantic_sha256": artifact["semantic_sha256"],
    }


def _strict_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is not a regular non-symlink file: {path}")
    payload = path.read_bytes()

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return value, payload


def _require_exact_mapping_keys(
    value: Any,
    expected: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise RuntimeError(f"{label} keys mismatch: expected {sorted(expected)}, got {observed}")
    return value


def _validate_frozen_descriptor(
    descriptor: Any,
    *,
    expected_path: Path,
    label: str,
    exact_keys: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(descriptor, dict):
        raise RuntimeError(f"{label} descriptor is not an object")
    required = {"path", "bytes", "sha256"}
    if not required <= set(descriptor) or (
        exact_keys is not None and set(descriptor) != exact_keys
    ):
        raise RuntimeError(f"{label} descriptor fields are incomplete or unexpected")
    if descriptor["path"] != str(expected_path):
        raise RuntimeError(f"{label} descriptor path mismatch")
    if expected_path.is_symlink() or not expected_path.is_file():
        raise RuntimeError(f"{label} artifact is not a regular non-symlink file")
    if descriptor["bytes"] != expected_path.stat().st_size:
        raise RuntimeError(f"{label} descriptor byte count mismatch")
    if descriptor["sha256"] != _sha256_file(expected_path):
        raise RuntimeError(f"{label} descriptor hash mismatch")
    return descriptor


def _synthetic_finite_number(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise RuntimeError(f"{label} is not a finite JSON number")
    return float(value)


def _validate_synthetic_arm(name: str, value: Any) -> dict[str, Any]:
    arm = _require_exact_mapping_keys(
        value,
        {
            "association",
            "geometry",
            "transport_mass_diagnostics",
            "numerical_validity",
            "initial_state_sha256",
            "initial_state_matches_common",
            "history",
            "wall_time_seconds",
            "heldout",
        },
        label=f"synthetic arm {name}",
    )
    for field in (
        "association",
        "geometry",
        "transport_mass_diagnostics",
        "numerical_validity",
        "heldout",
    ):
        if not isinstance(arm[field], dict):
            raise RuntimeError(f"synthetic arm {name}.{field} is not an object")
    association = arm["association"]
    required_association = {
        "parent_purity",
        "parent_completeness",
        "outlier_track_dust_recall",
        "outlier_observation_dust_recall",
        "inlier_track_dust_false_positive_rate",
        "inlier_observation_dust_false_positive_rate",
    }
    if not required_association <= set(association):
        raise RuntimeError(f"synthetic arm {name} association metrics are incomplete")
    for metric in required_association:
        _synthetic_finite_number(
            association[metric],
            label=f"synthetic arm {name} association.{metric}",
        )
    geometry = arm["geometry"]
    required_geometry = {
        "finite_spd",
        "source_center_max_px",
        "source_covariance_relative_max",
        "depth_bound_incidence",
    }
    if not required_geometry <= set(geometry):
        raise RuntimeError(f"synthetic arm {name} geometry metrics are incomplete")
    if not isinstance(geometry["finite_spd"], bool):
        raise RuntimeError(f"synthetic arm {name} finite-SPD flag is not Boolean")
    for metric in ("source_center_max_px", "source_covariance_relative_max"):
        _synthetic_finite_number(
            geometry[metric],
            label=f"synthetic arm {name} geometry.{metric}",
        )
    incidence = geometry["depth_bound_incidence"]
    if isinstance(incidence, bool) or not isinstance(incidence, int) or incidence < 0:
        raise RuntimeError(f"synthetic arm {name} depth-bound incidence is invalid")
    if not isinstance(arm["numerical_validity"].get("pass"), bool):
        raise RuntimeError(f"synthetic arm {name} numerical-validity flag is not Boolean")
    if not isinstance(arm["initial_state_matches_common"], bool):
        raise RuntimeError(f"synthetic arm {name} common-initial-state flag is not Boolean")
    if not isinstance(arm["history"], list):
        raise RuntimeError(f"synthetic arm {name} history is not a list")
    if not isinstance(arm["initial_state_sha256"], str):
        raise RuntimeError(f"synthetic arm {name} initial-state hash is missing")
    return arm


def _validate_synthetic_root(
    value: Any,
    *,
    root_index: int,
    roots: tuple[int, int, int],
) -> dict[str, Any]:
    root = _require_exact_mapping_keys(
        value,
        {
            "roots",
            "input_receipt",
            "common_initial_state_sha256",
            "nonoracle_state_hashes_before_evaluator",
            "nonoracle_plan_hashes_before_evaluator",
            "frozen_state_hashes_before_heldout",
            "frozen_plan_hashes_before_heldout",
            "projection_validity_arm_semantics",
            "heldout_release",
            "arms",
            "artifacts",
            "peak_rss_kib",
        },
        label=f"synthetic root {root_index}",
    )
    if root["roots"] != list(roots):
        raise RuntimeError(f"synthetic root {root_index} identity mismatch")
    arms = _require_exact_mapping_keys(
        root["arms"],
        set(SYNTHETIC_ARM_NAMES),
        label=f"synthetic root {root_index} arms",
    )
    for name in SYNTHETIC_ARM_NAMES:
        _validate_synthetic_arm(name, arms[name])
    nonoracle = set(SYNTHETIC_ARM_NAMES) - {"oracle"}
    for field, expected in (
        ("nonoracle_state_hashes_before_evaluator", nonoracle),
        ("nonoracle_plan_hashes_before_evaluator", nonoracle),
        ("frozen_state_hashes_before_heldout", set(SYNTHETIC_ARM_NAMES)),
        ("frozen_plan_hashes_before_heldout", set(SYNTHETIC_ARM_NAMES)),
    ):
        _require_exact_mapping_keys(root[field], expected, label=f"root {root_index}.{field}")
    input_receipt = root["input_receipt"]
    if not isinstance(input_receipt, dict) or input_receipt.get("roots") != list(roots):
        raise RuntimeError(f"synthetic root {root_index} input receipt mismatch")
    view_receipts = input_receipt.get("view_receipts")
    if not isinstance(view_receipts, list) or len(view_receipts) != 5:
        raise RuntimeError(f"synthetic root {root_index} view receipts are incomplete")
    for view in view_receipts:
        if (
            not isinstance(view, dict)
            or isinstance(view.get("moment_mean_max_abs"), bool)
            or not isinstance(view.get("moment_mean_max_abs"), int | float)
            or not math.isfinite(float(view["moment_mean_max_abs"]))
            or isinstance(view.get("moment_covariance_max_abs"), bool)
            or not isinstance(view.get("moment_covariance_max_abs"), int | float)
            or not math.isfinite(float(view["moment_covariance_max_abs"]))
        ):
            raise RuntimeError(f"synthetic root {root_index} moment receipt is invalid")

    root_directory = SYNTHETIC_ARTIFACTS / f"root_{root_index}"
    expected_artifact_paths = {
        "initial_ply": root_directory / "gaussians_init.ply",
        "evidence_npz": root_directory / "evidence.npz",
        "root_summary_json": root_directory / "ROOT_RESULT.json",
        **{f"{name}_ply": root_directory / f"{name}.ply" for name in SYNTHETIC_ARM_NAMES},
    }
    artifacts = _require_exact_mapping_keys(
        root["artifacts"],
        set(expected_artifact_paths),
        label=f"synthetic root {root_index} artifacts",
    )
    for name, artifact_path in expected_artifact_paths.items():
        exact_keys = (
            {"path", "bytes", "sha256", "members"}
            if name == "evidence_npz"
            else {"path", "bytes", "sha256"}
        )
        _validate_frozen_descriptor(
            artifacts[name],
            expected_path=artifact_path,
            label=f"synthetic root {root_index} {name}",
            exact_keys=exact_keys,
        )
    root_file, _payload = _strict_json_object(
        expected_artifact_paths["root_summary_json"],
        label=f"synthetic root {root_index} summary",
    )
    expected_root_file = dict(root)
    expected_root_artifacts = dict(artifacts)
    expected_root_artifacts.pop("root_summary_json")
    expected_root_file["artifacts"] = expected_root_artifacts
    if root_file != expected_root_file:
        raise RuntimeError(f"synthetic root {root_index} summary/result mismatch")
    return root


def _transport_arm_is_accepted(root_results: list[dict[str, Any]], arm: str) -> bool:
    return all(
        root["arms"][arm]["association"]["parent_purity"] >= 0.90
        and root["arms"][arm]["association"]["parent_completeness"] >= 0.90
        and root["arms"][arm]["association"]["outlier_track_dust_recall"] >= 0.80
        and root["arms"][arm]["association"]["outlier_observation_dust_recall"] >= 0.80
        and root["arms"][arm]["association"]["inlier_track_dust_false_positive_rate"] <= 0.20
        and root["arms"][arm]["association"]["inlier_observation_dust_false_positive_rate"] <= 0.20
        and root["arms"][arm]["transport_mass_diagnostics"].get("pass") is True
        for root in root_results
    )


def _recompute_synthetic_validity_checks(
    root_results: list[dict[str, Any]],
) -> dict[str, bool]:
    area_geometry = [root["arms"]["uot_area"]["geometry"] for root in root_results]
    oracle_association = [root["arms"]["oracle"]["association"] for root in root_results]
    moment_splits_rederive = all(
        max(
            _synthetic_finite_number(
                view["moment_mean_max_abs"],
                label="synthetic moment mean residual",
            )
            for view in root["input_receipt"]["view_receipts"]
        )
        <= 1e-10
        and max(
            _synthetic_finite_number(
                view["moment_covariance_max_abs"],
                label="synthetic moment covariance residual",
            )
            for view in root["input_receipt"]["view_receipts"]
        )
        <= 1e-10
        for root in root_results
    )
    return {
        "all_arms_finite_spd": all(
            arm["geometry"]["finite_spd"] is True and arm["numerical_validity"]["pass"] is True
            for root in root_results
            for arm in root["arms"].values()
        ),
        "all_arms_share_common_initial_state": all(
            arm["initial_state_matches_common"] is True
            for root in root_results
            for arm in root["arms"].values()
        ),
        "uot_area_source_center_le_1e-8": all(
            _synthetic_finite_number(
                item["source_center_max_px"],
                label="synthetic uot_area source-center error",
            )
            <= 1e-8
            for item in area_geometry
        ),
        "uot_area_source_covariance_le_1e-8": all(
            _synthetic_finite_number(
                item["source_covariance_relative_max"],
                label="synthetic uot_area source-covariance error",
            )
            <= 1e-8
            for item in area_geometry
        ),
        "all_depths_strictly_bounded": all(
            arm["geometry"]["depth_bound_incidence"] == 0
            for root in root_results
            for arm in root["arms"].values()
        ),
        "moment_splits_rederive": moment_splits_rederive,
        "oracle_purity_ge_0_99": all(
            _synthetic_finite_number(
                item["parent_purity"],
                label="synthetic oracle parent purity",
            )
            >= 0.99
            for item in oracle_association
        ),
        "oracle_completeness_ge_0_99": all(
            _synthetic_finite_number(
                item["parent_completeness"],
                label="synthetic oracle parent completeness",
            )
            >= 0.99
            for item in oracle_association
        ),
    }


def _recompute_synthetic_real_release(
    root_results: list[dict[str, Any]],
    *,
    validity_pass: bool,
) -> dict[str, Any]:
    primary: str | None = None
    if validity_pass:
        area_accepted = _transport_arm_is_accepted(root_results, "uot_area")
        uniform_accepted = _transport_arm_is_accepted(root_results, "uot_uniform")
        primary = "uot_area" if area_accepted else "uot_uniform" if uniform_accepted else None
    permitted = validity_pass and primary is not None
    return {
        "permitted": permitted,
        "primary_transport_arm": primary,
        "paired_diagnostic_arms": list(ARMS),
        "reason": (
            f"synthetic validity passed and {primary} was scientifically accepted"
            if permitted
            else "no scientifically accepted transport arm"
            if validity_pass
            else "synthetic validity not established"
        ),
    }


def _assert_synthetic_release(path: Path | None) -> dict[str, Any]:
    """Attest the complete frozen official synthetic result before real fitting."""

    if path is None or path != SYNTHETIC_RESULT:
        raise RuntimeError("real interaction requires the exact frozen synthetic result path")
    result, result_bytes = _strict_json_object(path, label="synthetic result")
    _require_exact_mapping_keys(
        result,
        {
            "namespace",
            "status",
            "mode",
            "roots",
            "config",
            "arms",
            "root_results",
            "synthetic_gates",
            "real_release",
            "environment",
            "wall_time_seconds",
            "peak_rss_kib",
            "protocol",
            "source_hashes",
            "official_attempt",
        },
        label="synthetic result",
    )
    if result["namespace"] != SYNTHETIC_NAMESPACE or result["mode"] != "official":
        raise RuntimeError("synthetic result namespace/mode mismatch")
    expected_roots = [list(item) for item in SYNTHETIC_OFFICIAL_ROOT_TUPLES]
    if result["roots"] != expected_roots:
        raise RuntimeError("synthetic result root tuples mismatch")
    if result["config"] != SYNTHETIC_FROZEN_CONFIG:
        raise RuntimeError("synthetic result frozen config mismatch")
    if result["arms"] != list(SYNTHETIC_ARM_NAMES):
        raise RuntimeError("synthetic result does not contain the complete frozen arm suite")
    protocol = _protocol_receipt()
    if result["protocol"] != protocol:
        raise RuntimeError("synthetic result protocol receipt mismatch")
    source_hashes = _synthetic_source_hashes()
    if result["source_hashes"] != source_hashes:
        raise RuntimeError("synthetic result executed-source hashes mismatch")

    attempt = _require_exact_mapping_keys(
        result["official_attempt"],
        {"receipt", "descriptor"},
        label="synthetic official attempt",
    )
    attempt_descriptor = _validate_frozen_descriptor(
        attempt["descriptor"],
        expected_path=SYNTHETIC_ATTEMPT,
        label="synthetic official attempt",
        exact_keys={"path", "bytes", "sha256"},
    )
    attempt_file, attempt_bytes = _strict_json_object(
        SYNTHETIC_ATTEMPT,
        label="synthetic official attempt",
    )
    if attempt["receipt"] != attempt_file:
        raise RuntimeError("synthetic attempt receipt/file mismatch")
    receipt = _require_exact_mapping_keys(
        attempt_file,
        {
            "namespace",
            "status",
            "mode",
            "root_tuples",
            "all_nine_roots",
            "scene_roots",
            "depth_roots",
            "order_roots",
            "config",
            "arms",
            "protocol",
            "source_hashes",
            "artifacts_directory",
            "result_path",
            "environment_at_reservation",
        },
        label="synthetic attempt receipt",
    )
    flattened = [value for root in SYNTHETIC_OFFICIAL_ROOT_TUPLES for value in root]
    if (
        receipt["namespace"] != SYNTHETIC_NAMESPACE
        or receipt["status"] != "ATTEMPTED"
        or receipt["mode"] != "official"
        or receipt["root_tuples"] != expected_roots
        or receipt["all_nine_roots"] != flattened
        or receipt["scene_roots"] != [root[0] for root in SYNTHETIC_OFFICIAL_ROOT_TUPLES]
        or receipt["depth_roots"] != [root[1] for root in SYNTHETIC_OFFICIAL_ROOT_TUPLES]
        or receipt["order_roots"] != [root[2] for root in SYNTHETIC_OFFICIAL_ROOT_TUPLES]
        or receipt["config"] != SYNTHETIC_FROZEN_CONFIG
        or receipt["arms"] != list(SYNTHETIC_ARM_NAMES)
        or receipt["protocol"] != protocol
        or receipt["source_hashes"] != source_hashes
        or receipt["artifacts_directory"] != str(SYNTHETIC_ARTIFACTS)
        or receipt["result_path"] != str(SYNTHETIC_RESULT)
    ):
        raise RuntimeError("synthetic attempt receipt is inconsistent with the frozen result")
    if attempt_descriptor["sha256"] != _sha256_bytes(attempt_bytes):
        raise RuntimeError("synthetic attempt descriptor did not bind its exact bytes")

    raw_roots = result["root_results"]
    if not isinstance(raw_roots, list) or len(raw_roots) != len(SYNTHETIC_OFFICIAL_ROOT_TUPLES):
        raise RuntimeError("synthetic result does not contain all official root results")
    root_results = [
        _validate_synthetic_root(value, root_index=index, roots=roots)
        for index, (value, roots) in enumerate(
            zip(raw_roots, SYNTHETIC_OFFICIAL_ROOT_TUPLES, strict=True)
        )
    ]
    gates = _require_exact_mapping_keys(
        result["synthetic_gates"],
        {
            "eligible",
            "validity",
            "transport_mass_validity",
            "absolute_mechanism",
            "soft_assignment_gain",
            "capacity_attribution",
            "negative_control",
            "overall",
        },
        label="synthetic gates",
    )
    validity = _require_exact_mapping_keys(
        gates["validity"],
        {"checks", "pass"},
        label="synthetic validity gate",
    )
    recomputed_validity_checks = _recompute_synthetic_validity_checks(root_results)
    declared_validity_checks = _require_exact_mapping_keys(
        validity["checks"],
        set(recomputed_validity_checks),
        label="synthetic validity checks",
    )
    recomputed_validity_pass = all(recomputed_validity_checks.values())
    if declared_validity_checks != recomputed_validity_checks:
        raise RuntimeError("synthetic validity checks disagree with signed root evidence")
    if validity["pass"] is not recomputed_validity_pass:
        raise RuntimeError("synthetic validity pass flag disagrees with its recomputed checks")
    if gates["eligible"] is not True or not recomputed_validity_pass:
        raise RuntimeError("synthetic validity gate did not pass; real interaction is withheld")
    for name in (
        "transport_mass_validity",
        "absolute_mechanism",
        "soft_assignment_gain",
        "capacity_attribution",
        "negative_control",
    ):
        gate = gates[name]
        if not isinstance(gate, dict) or not {"checks", "pass"} <= set(gate):
            raise RuntimeError(f"synthetic gate {name} is structurally incomplete")
    overall = gates["overall"]
    if not isinstance(overall, dict) or isinstance(overall.get("pass"), bool) is False:
        raise RuntimeError("synthetic overall gate is structurally incomplete")
    expected_status = "PASS" if overall["pass"] else "FAIL"
    if result["status"] != expected_status:
        raise RuntimeError("synthetic result status disagrees with its overall gate")

    release = _require_exact_mapping_keys(
        result["real_release"],
        {"permitted", "primary_transport_arm", "paired_diagnostic_arms", "reason"},
        label="synthetic real release",
    )
    recomputed_release = _recompute_synthetic_real_release(
        root_results,
        validity_pass=recomputed_validity_pass,
    )
    if release != recomputed_release:
        raise RuntimeError("synthetic real release disagrees with area-first signed evidence")
    if recomputed_release["permitted"] is not True:
        raise RuntimeError("synthetic result did not issue an accepted-arm real release")
    primary = recomputed_release["primary_transport_arm"]
    return {
        "path": str(path),
        "bytes": len(result_bytes),
        "sha256": _sha256_bytes(result_bytes),
        "official_attempt_path": str(SYNTHETIC_ATTEMPT),
        "official_attempt_sha256": attempt_descriptor["sha256"],
        "protocol": protocol,
        "source_hashes": source_hashes,
        "roots": expected_roots,
        "arms": list(SYNTHETIC_ARM_NAMES),
        "eligible": True,
        "validity_pass": True,
        "primary_transport_arm": primary,
    }


def _resolve_primary_arm(
    synthetic_release: dict[str, Any],
    requested: Literal["uot_uniform", "uot_area"] | None,
) -> Literal["uot_uniform", "uot_area"]:
    released = synthetic_release.get("primary_transport_arm")
    if released not in ARMS:
        raise RuntimeError("synthetic release has no supported primary transport arm")
    if requested is not None and requested != released:
        raise RuntimeError("requested primary arm conflicts with the frozen synthetic disposition")
    return released


def _path_is_occupied(path: Path) -> bool:
    return os.path.lexists(path)


def _require_official_real_invocation(
    *,
    mode: str,
    confirm_official: bool,
    bundle: Path,
    output: Path,
    result_path: Path,
    synthetic_result: Path,
    scene: Path,
    expected_manifest_sha256: str | None,
) -> None:
    if mode != "official" or confirm_official is not True:
        raise RuntimeError("real interaction requires explicit official mode and confirmation")
    expected_paths = {
        "bundle": (bundle, BUNDLE),
        "output": (output, DEFAULT_OUT),
        "result": (result_path, DEFAULT_RESULT),
        "synthetic_result": (synthetic_result, SYNTHETIC_RESULT),
        "scene": (scene, SCENE),
    }
    mismatched = [
        name for name, (observed, expected) in expected_paths.items() if observed != expected
    ]
    if mismatched:
        raise RuntimeError(f"official real interaction path mismatch: {', '.join(mismatched)}")
    if expected_manifest_sha256 != FROZEN_MANIFEST_SHA256:
        raise RuntimeError("official real interaction requires the frozen bundle-manifest hash")


def _reserve_real_attempt(
    *,
    bundle: Path,
    output: Path,
    result_path: Path,
    synthetic_release: dict[str, Any],
    primary_arm: str,
    release_report: bool,
    protocol: dict[str, Any],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    occupied = [path for path in (OFFICIAL_ATTEMPT, result_path, output) if _path_is_occupied(path)]
    if occupied:
        raise RuntimeError(
            "official real attempt/result/output already exists: "
            + ", ".join(str(path) for path in occupied)
        )
    receipt = {
        "namespace": NAMESPACE,
        "status": "ATTEMPTED",
        "mode": "official",
        "confirmed": True,
        "bundle": str(bundle),
        "bundle_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "output_directory": str(output),
        "result_path": str(result_path),
        "synthetic_result": {
            "path": synthetic_release["path"],
            "bytes": synthetic_release["bytes"],
            "sha256": synthetic_release["sha256"],
            "official_attempt_path": synthetic_release["official_attempt_path"],
            "official_attempt_sha256": synthetic_release["official_attempt_sha256"],
        },
        "protocol": protocol,
        "executed_source_hashes": source_hashes,
        "development_ids": list(DEVELOPMENT_IDS),
        "validation_ids": list(VALIDATION_IDS),
        "report_id": REPORT_ID,
        "arms": list(ARMS),
        "primary_arm": primary_arm,
        "release_report": release_report,
        "fit_config": asdict(official_fit_config()),
        "appearance_config": asdict(
            SourceAnchoredSHFitConfig(degree=1, iterations=100, learning_rate=0.05)
        ),
        "created_unix_ns": time.time_ns(),
        "environment_at_reservation": _environment(),
    }
    descriptor = _write_json_exclusive(OFFICIAL_ATTEMPT, receipt)
    return {"receipt": receipt, "descriptor": descriptor}


def run_real_interaction(
    *,
    mode: Literal["official"],
    confirm_official: bool,
    bundle: Path,
    output: Path,
    result_path: Path,
    synthetic_result: Path,
    primary_arm: Literal["uot_uniform", "uot_area"] | None = None,
    release_report: bool = False,
    scene: Path = SCENE,
    expected_manifest_sha256: str | None = FROZEN_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Execute the frozen real interaction. Tests use helpers; this is the outcome path."""

    _require_official_real_invocation(
        mode=mode,
        confirm_official=confirm_official,
        bundle=bundle,
        output=output,
        result_path=result_path,
        synthetic_result=synthetic_result,
        scene=scene,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    protocol = _protocol_receipt()
    synthetic_release = _assert_synthetic_release(synthetic_result)
    primary_arm = _resolve_primary_arm(synthetic_release, primary_arm)
    source_hashes_start = _real_source_hashes()
    official_attempt = _reserve_real_attempt(
        bundle=bundle,
        output=output,
        result_path=result_path,
        synthetic_release=synthetic_release,
        primary_arm=primary_arm,
        release_report=release_report,
        protocol=protocol,
        source_hashes=source_hashes_start,
    )
    output.mkdir(parents=True, exist_ok=False)
    _fsync_directory(output.parent)
    development = load_development_inputs(
        bundle,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    bounds = compute_development_bounds(development.inputs)
    problem = build_real_problem(development.inputs, bounds)
    input_artifact = _write_npz_exclusive(
        output / "development_inputs.npz",
        _input_arrays(problem),
    )
    anchor_evidence = _anchor_evidence_summary(problem, input_artifact)
    initial = fresh_fiber(problem)
    initial_artifacts = {
        "state": _write_npz_exclusive(
            output / "geometry_init.npz",
            _fiber_arrays(initial),
        ),
        "ply": _save_ply_exclusive(
            output / "gaussians_init.ply",
            initial.as_gaussians(opacity=FIXED_OPACITY),
        ),
    }
    arms = {name: run_arm(problem, name, output, config=official_fit_config()) for name in ARMS}
    geometry_descriptors: dict[str, dict[str, Any]] = {
        "development_inputs": input_artifact,
        **{f"initial_{key}": value for key, value in initial_artifacts.items()},
    }
    for name, state in arms.items():
        for artifact_name, descriptor in state.artifacts.items():
            geometry_descriptors[f"{name}_{artifact_name}"] = descriptor
    geometry_freeze = _freeze_receipt(output, "geometry", geometry_descriptors)
    primary = arms[primary_arm]
    appearance = fit_appearance(primary, development.inputs, output)
    appearance_freeze = _freeze_receipt(
        output,
        "appearance",
        appearance.artifacts,
        parent_receipt=geometry_freeze,
    )
    validation = release_validation_inputs(
        bundle,
        appearance_freeze,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    validation_by_arm = {
        name: _validation_geometry_metrics(initial, state, validation.inputs)
        for name, state in arms.items()
    }
    validation_appearance = _validation_appearance_metrics(
        primary,
        appearance,
        validation.inputs,
    )
    gates = {name: geometry_gate(state, validation_by_arm[name]) for name, state in arms.items()}
    report = None
    if release_report:
        report = release_report_snapshot(
            scene,
            output,
            appearance.gaussians,
            appearance_freeze,
        )
    source_hashes_end = _real_source_hashes()
    if source_hashes_end != source_hashes_start:
        raise RuntimeError("executed real-runner sources changed during the official interaction")
    result = {
        "namespace": NAMESPACE,
        "mode": "official",
        "status": "PASS" if gates[primary_arm]["pass"] else "FAIL",
        "claim_boundary": (
            "bounded calibrated interaction on 640 hypotheses; inferred associations only"
        ),
        "config": {
            "development_ids": list(DEVELOPMENT_IDS),
            "validation_ids": list(VALIDATION_IDS),
            "report_id": REPORT_ID,
            "anchors_per_view": N_ANCHORS_PER_VIEW,
            "tracks": N_TRACKS,
            "primary_arm": primary_arm,
            "fit": asdict(official_fit_config()),
            "appearance": asdict(
                SourceAnchoredSHFitConfig(degree=1, iterations=100, learning_rate=0.05)
            ),
            "appearance_constraint": "exact_source_preactivation_only",
            "fixed_opacity": FIXED_OPACITY,
        },
        "environment": _environment(),
        "protocol": protocol,
        "official_attempt": official_attempt,
        "executed_source_hashes": {
            "start": source_hashes_start,
            "end": source_hashes_end,
            "stable": True,
        },
        "synthetic_release": synthetic_release,
        "input_access": development.receipt,
        "bounds": {
            "source": "camera_axis_fallback",
            "center": bounds.center.tolist(),
            "extent": bounds.extent,
            "lower": bounds.lower.tolist(),
            "upper": bounds.upper.tolist(),
            "near_depth": NEAR_DEPTH,
            "semantic_sha256": bounds.semantic_sha256,
        },
        "anchor_evidence": anchor_evidence,
        "geometry_freeze": geometry_freeze,
        "appearance_freeze": appearance_freeze,
        "validation_release": validation.receipt,
        "arms": {
            name: {
                "elapsed_seconds": state.elapsed_seconds,
                "peak_rss_kib": state.peak_rss_kib,
                "initial_development_set_costs": list(state.initial_set_costs),
                "final_development_set_costs": list(state.final_set_costs),
                "diagnostics": state.diagnostics,
                "validation": validation_by_arm[name],
                "gate": gates[name],
                "artifacts": state.artifacts,
            }
            for name, state in arms.items()
        },
        "appearance": {
            "development": appearance.dev_metrics,
            "validation": validation_appearance,
            "losses": list(appearance.fit.losses),
            "artifacts": appearance.artifacts,
        },
        "report_release": report,
        "viewer_command": (
            ".venv/bin/rtgs view "
            f"--gaussians {output / 'gaussians.ply'} "
            f"--initial {output / 'gaussians_init.ply'} "
            f"--scene {scene} --downscale 16 --device cpu --rasterizer torch --no-open"
        ),
    }
    result_descriptor = _write_json_exclusive(result_path, result)
    result["result_descriptor"] = result_descriptor
    return result


def release_report_snapshot(
    scene: Path,
    output: Path,
    gaussians: Gaussians3D,
    appearance_freeze: dict[str, Any],
    *,
    downscale: int = 16,
) -> dict[str, Any]:
    """Decode only C1004 and save an exact CPU Torch snapshot after model freeze."""

    verify_freeze_receipt(appearance_freeze, expected_kind="appearance")
    if downscale <= 0:
        raise ValueError("report downscale must be positive")
    from PIL import Image as PILImage

    from rtgs.data.calibrated import _find_calibration, _resize_image, _undistort
    from rtgs.render.torch_ref import TorchRasterizer

    frame = scene.resolve()
    rgb_candidates = tuple(
        frame / "rgb" / f"{REPORT_ID}{suffix}" for suffix in (".jpg", ".jpeg", ".png")
    )
    rgb_path = next((path for path in rgb_candidates if path.is_file()), None)
    if rgb_path is None:
        raise FileNotFoundError(f"missing explicit report-only RGB for {REPORT_ID}")
    mask_candidates = tuple(
        frame / "mask" / f"mask_{REPORT_ID}{suffix}" for suffix in (".png", ".jpg")
    )
    mask_path = next((path for path in mask_candidates if path.is_file()), None)
    if mask_path is None:
        raise FileNotFoundError(f"missing explicit report-only mask for {REPORT_ID}")
    calibration_path = _find_calibration(frame)
    calibration = json.loads(calibration_path.read_bytes())
    records = {str(record["camera_id"]).upper(): record for record in calibration["cameras"]}
    if REPORT_ID not in records:
        raise KeyError(f"calibration does not contain {REPORT_ID}")
    record = records[REPORT_ID]
    intrinsics = record["intrinsics"]
    calibration_width, calibration_height = map(int, intrinsics["resolution"])
    with PILImage.open(rgb_path) as source:
        source_width, source_height = source.size
    width = max(1, source_width // downscale)
    height = max(1, source_height // downscale)
    sx, sy = width / calibration_width, height / calibration_height
    matrix = intrinsics["camera_matrix"]
    fx, fy = float(matrix[0]) * sx, float(matrix[4]) * sy
    cx, cy = (float(matrix[2]) + 0.5) * sx, (float(matrix[5]) + 0.5) * sy
    view = torch.tensor(record["extrinsics"]["view_matrix"], dtype=torch.float32).view(4, 4)
    camera = Camera(fx, fy, cx, cy, width, height, view[:3, :3], view[:3, 3])
    distortion = intrinsics.get("distortion_coefficients", [])
    image = _undistort(
        _resize_image(rgb_path, width, height),
        fx,
        fy,
        cx,
        cy,
        distortion,
    )
    mask = _undistort(
        _resize_image(mask_path, width, height, mask=True),
        fx,
        fy,
        cx,
        cy,
        distortion,
        mask=True,
    )
    with torch.no_grad():
        rendered = TorchRasterizer().render(gaussians.to("cpu"), camera)
    active = mask > 0.5
    denominator = max(1, 3 * int(active.sum()))
    mse = float(((rendered.color - image).square() * active[:, :, None]).sum()) / denominator
    psnr = None if mse == 0 else -10.0 * math.log10(mse)
    snapshot = _write_npz_exclusive(
        output / "c1004_torch_snapshot.npz",
        {
            "rendered_color": rendered.color,
            "rendered_alpha": rendered.alpha,
            "rendered_depth": rendered.depth,
            "target_color": image,
            "target_mask": mask,
            "camera_R": camera.R,
            "camera_t": camera.t,
            "camera_intrinsics": torch.tensor([fx, fy, cx, cy]),
        },
    )
    return {
        "view_id": REPORT_ID,
        "release_after_appearance_freeze": appearance_freeze["sha256"],
        "rgb": _descriptor(rgb_path),
        "mask": _descriptor(mask_path),
        "calibration": _descriptor(calibration_path),
        "backend": "rtgs.render.torch_ref.TorchRasterizer",
        "background_rgb": [0.0, 0.0, 0.0],
        "sh_degree": 1,
        "downscale": downscale,
        "masked_mse": mse,
        "masked_psnr": psnr,
        "snapshot": snapshot,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("official",), required=True)
    parser.add_argument("--confirm-official", action="store_true")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--synthetic-result", type=Path, required=True)
    parser.add_argument("--primary-arm", choices=ARMS, default=None)
    parser.add_argument("--release-report", action="store_true")
    parser.add_argument("--scene", type=Path, default=SCENE)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_real_interaction(
        mode=args.mode,
        confirm_official=args.confirm_official,
        bundle=args.bundle,
        output=args.out,
        result_path=args.result,
        synthetic_result=args.synthetic_result,
        primary_arm=args.primary_arm,
        release_report=args.release_report,
        scene=args.scene,
    )


if __name__ == "__main__":
    main()
