#!/usr/bin/env python3
"""Frozen teacher-artifact fidelity/capacity crossover for compact 3D fitting.

The causal comparison changes only the frozen 2D teacher set:

* ``old_640``: seven 640-component StructSplat fields;
* ``new_2000``: the same seven cameras/windows with 2,000 components and exact mean residuals.

Both arms receive the exact bounds serialized by ``new_2000``.  Production
``CompactCarveInitializer`` consumes every component center exactly once, scores every ray with
``GaussianObservationIndex``, samples 48 midpoint depths, and selects ``N_init^3D=835``.  The two
initializations then receive identical fixed-topology, pixel-uniform CPU refinement.

No scene raster may be opened until both arms, histories, lineage archives, PLYs, and the atomic
selection receipt exist.  Native RGB/masks are reporting-only in the later ``evaluate`` phase.
The intervention is the complete frozen teacher artifact, not component count alone: the new
archives also carry exact mean residuals and a different fit configuration. There is deliberately
no outcome threshold; this is an exploratory causal diagnostic of the practical artifact upgrade.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import dataclasses
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
import platform
import secrets
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationIndex
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.compact_carve import (
    CompactCandidateAudit,
    CompactCarveConfig,
    CompactCarveInitializer,
    CompactInitializationResult,
    CompactRayDepthAuditBatch,
    _propose_anchors,
)
from rtgs.optim.compact_trainer import CompactTrainConfig, CompactTrainer

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008"
OLD_BUNDLE = ROOT / "runs/compact_masked_bundle_640_20260717/reconstruction_inputs"
NEW_BUNDLE = ROOT / "runs/stage1_capacity_bundle_all7_count2000_i100_20260717/reconstruction_inputs"
DEFAULT_OUT = ROOT / "runs/compact_teacher_capacity_crossover_20260717"

TRAIN_VIEWS = ("C0001", "C0008", "C0014", "C0021", "C0026", "C0031", "C0039")
HELDOUT_VIEW = "C1004"
NATIVE_SIZE = (5_328, 4_608)
OLD_COMPONENTS_PER_VIEW = 640
NEW_COMPONENTS_PER_VIEW = 2_000
OLD_CANDIDATES = len(TRAIN_VIEWS) * OLD_COMPONENTS_PER_VIEW
NEW_CANDIDATES = len(TRAIN_VIEWS) * NEW_COMPONENTS_PER_VIEW
N_INIT_3D = 835
NEXT_CAPACITY_ARM_N_INIT_3D = 2_610
SAMPLES_PER_RAY = 48
LIFT_SEED = 75_200
TRAIN_SEED = 75_201
BOUNDS_CENTER = (
    0.33366623520851135,
    0.12765459716320038,
    2.7440900802612305,
)
BOUNDS_EXTENT = 2.297786235809326
INDEX_ENTRY_CAP = 16_000_000
ABI_PRELOAD = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33")
CALIBRATION = ROOT / "dataset/2025_03_07_stage_with_fabric/calibration_dome.json"
SCHEMA = "compact_teacher_capacity_crossover"

SOURCE_PATHS = tuple(
    sorted(
        (
            *((path.relative_to(ROOT)) for path in (ROOT / "src/rtgs").rglob("*.py")),
            Path("benchmarks/compact_bounds_crossover.py"),
            Path("benchmarks/compact_teacher_capacity_crossover.py"),
        ),
        key=lambda path: path.as_posix(),
    )
)
MODEL_NAMES = ("old_init", "old_final", "new_init", "new_final")
EVALUATION_VIEWS = (*TRAIN_VIEWS, HELDOUT_VIEW)
EVALUATION_SPLITS = (*("train" for _ in TRAIN_VIEWS), "heldout")
VIEWER_MARKER_PREFIX = "RTGS_TEACHER_ARTIFACT_VIEW_BINDING="
SAMPLE_HASH_KEYS = (
    "view_index",
    "view_name",
    "sample_seed",
    "xy_sha256",
    "active_sha256",
    "inside_fit_window_sha256",
    "proposal_density_sha256",
    "joint_density_sha256",
    "target_density_sha256",
    "importance_sha256",
    "proposal_component_ids_sha256",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(b"\0")
    digest.update(canonical_bytes(list(array.shape)))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    return array_hash(value.detach().contiguous().cpu().numpy())


def display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def bound_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{path} contains duplicate key {key!r}")
            result[key] = value
        return result

    value = json.loads(path.read_bytes(), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_bytes_exclusive(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def _write_json_exclusive(path: Path, value: Any) -> str:
    return _write_bytes_exclusive(path, canonical_bytes(value))


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _save_ply_exclusive(path: Path, gaussians: Gaussians3D) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    try:
        gaussians.detach().to("cpu").save_ply(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _failure(out: Path, phase: str, error: BaseException) -> None:
    payload = {
        "artifact_type": f"{SCHEMA}_failure_v1",
        "phase": phase,
        "exception_type": type(error).__name__,
        "message": str(error)[:4_096],
        "timestamp_ns": time.time_ns(),
    }
    path = out / "failures" / f"{phase}_{payload['timestamp_ns']}.json"
    with contextlib.suppress(BaseException):
        _write_json_exclusive(path, payload)


def _ordinary_tree_hashes(directory: Path) -> dict[str, str]:
    try:
        root_mode = os.lstat(directory).st_mode
    except FileNotFoundError as error:
        raise ValueError(f"bundle does not exist: {directory}") from error
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise ValueError("bundle root must be an ordinary directory")
    hashes: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        mode = os.lstat(path).st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"bundle contains symlink {relative!r}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"bundle contains non-regular entry {relative!r}")
        hashes[relative] = sha256_file(path)
    if not hashes:
        raise ValueError("bundle tree is empty")
    return hashes


def bundle_binding(directory: Path) -> dict[str, Any]:
    hashes = _ordinary_tree_hashes(directory)
    return {
        "path": display_path(directory),
        "files": hashes,
        "aggregate_sha256": canonical_hash(hashes),
    }


def source_binding() -> dict[str, Any]:
    hashes = {path.as_posix(): sha256_file(ROOT / path) for path in SOURCE_PATHS}
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    return {
        "git_revision": revision,
        "files": hashes,
        "aggregate_sha256": canonical_hash(hashes),
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_driver_versions() -> list[str] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return [line.strip() for line in output.splitlines() if line.strip()]


def _cuda_runtime_binding() -> dict[str, Any]:
    available = torch.cuda.is_available()
    devices = []
    if available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "compute_capability": [properties.major, properties.minor],
                    "total_memory": properties.total_memory,
                    "multi_processor_count": properties.multi_processor_count,
                    "uuid": str(getattr(properties, "uuid", "")),
                }
            )
    return {
        "torch_cuda_build": torch.version.cuda,
        "available": available,
        "device_count": torch.cuda.device_count() if available else 0,
        "devices": devices,
        "driver_versions": _nvidia_driver_versions(),
    }


def environment_binding() -> dict[str, Any]:
    """Freeze scientific dependencies while keeping phase-specific preload state separate."""
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pillow": importlib.metadata.version("Pillow"),
        "gsplat": _distribution_version("gsplat"),
        "viser": _distribution_version("viser"),
        "cuda": _cuda_runtime_binding(),
        "device_policy": "cpu_selection_and_refinement",
        "required_evaluation_abi_preload": {
            "path": str(ABI_PRELOAD),
            "sha256": sha256_file(ABI_PRELOAD),
            "effective_value_bound_in_phase_receipt": True,
        },
    }


def _cpu_phase_runtime_record() -> dict[str, Any]:
    return {
        "effective_ld_preload": "",
        "policy": "plan_and_selection_require_no_preloaded_shared_libraries",
    }


def _assert_cpu_phase_runtime() -> dict[str, Any]:
    effective = os.environ.get("LD_PRELOAD", "")
    if effective:
        raise RuntimeError(
            f"teacher-artifact plan/selection requires an empty LD_PRELOAD; got {effective!r}"
        )
    return _cpu_phase_runtime_record()


def _camera_record(camera: Camera) -> dict[str, Any]:
    return {
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "width": camera.width,
        "height": camera.height,
        "R_sha256": tensor_hash(camera.R),
        "t_sha256": tensor_hash(camera.t),
    }


def lift_config() -> CompactCarveConfig:
    return CompactCarveConfig(
        n_init_3d=N_INIT_3D,
        candidate_multiplier=4,
        anchor_mode="component_centers",
        max_anchor_candidates=1_000_000,
        samples_per_ray=SAMPLES_PER_RAY,
        query_batch_size=4_096,
        query_component_chunk=256,
        max_query_pairs=1_048_576,
        tile_size=16,
        max_index_entries_per_view=INDEX_ENTRY_CAP,
        max_candidates_per_tile=200_000,
        seed=LIFT_SEED,
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


def train_config() -> CompactTrainConfig:
    return CompactTrainConfig(
        iterations=140,
        attempts_per_step=128,
        proposal_mode="pixel_uniform",
        schedule_mode="balanced_cycle",
        target_mode="uniform",
        uniform_fraction=1.0,
        seed=TRAIN_SEED,
        extent=BOUNDS_EXTENT,
        device="cpu",
        point_chunk=32,
        gaussian_chunk=64,
        outer_microbatch=32,
        query_component_chunk=64,
        teacher_tile_size=16,
        evaluation_chunk=256,
        checkpoints=(0, 28, 70, 140),
        evaluate_checkpoint_risks=False,
        sh_degree=0,
        max_index_entries_per_view=INDEX_ENTRY_CAP,
    )


def config_record() -> dict[str, Any]:
    refinement = dataclasses.asdict(train_config())
    refinement["checkpoints"] = list(refinement["checkpoints"])
    return {
        "lift": dataclasses.asdict(lift_config()),
        "refinement": refinement,
        "protocol": {
            "train_views": list(TRAIN_VIEWS),
            "heldout_view": HELDOUT_VIEW,
            "native_resolution": list(NATIVE_SIZE),
            "old_components_per_view": OLD_COMPONENTS_PER_VIEW,
            "new_components_per_view": NEW_COMPONENTS_PER_VIEW,
            "n_init_3d": N_INIT_3D,
            "n_opt_3d": N_INIT_3D,
            "samples_per_ray": SAMPLES_PER_RAY,
            "no_outcome_threshold": True,
            "interpretation": "descriptive exploratory causal comparison",
            "conditional_next_arm": {
                "enabled_in_this_plan": False,
                "n_init_3d": NEXT_CAPACITY_ARM_N_INIT_3D,
                "reason": (
                    "capacity-confounded follow-up only after interpreting the fixed-835 "
                    "teacher-artifact fidelity/capacity crossover"
                ),
            },
            "count_only_ablation": False,
            "intervention": (
                "complete old_640 versus new_2000 frozen teacher artifacts, including the "
                "new archive's exact mean-residual schema and fit configuration"
            ),
        },
    }


def _assert_exact_bounds(inputs: ReconstructionInputs) -> None:
    if inputs.bounds_hint is None:
        raise ValueError("new bundle must contain the frozen mask-derived bounds")
    center, extent = inputs.bounds_hint
    expected = torch.tensor(BOUNDS_CENTER, dtype=center.dtype, device=center.device)
    if not torch.equal(center, expected) or float(extent) != BOUNDS_EXTENT:
        raise ValueError("new bundle bounds differ from the frozen exact bounds")


def validate_paired_inputs(
    old: ReconstructionInputs,
    new: ReconstructionInputs,
) -> dict[str, Any]:
    """Validate every shared causal surface and the intended teacher differences."""
    old.validate()
    new.validate()
    if tuple(old.view_names) != TRAIN_VIEWS or tuple(new.view_names) != TRAIN_VIEWS:
        raise ValueError("paired bundles must contain the frozen seven training views in order")
    if HELDOUT_VIEW in old.view_names or HELDOUT_VIEW in new.view_names:
        raise ValueError("held-out C1004 must be absent from both selection bundles")
    if old.points is not None or new.points is not None:
        raise ValueError("teacher-artifact crossover forbids sparse points")
    if old.point_visibility is not None or new.point_visibility is not None:
        raise ValueError("teacher-artifact crossover forbids point visibility")
    if old.bounds_hint is not None:
        raise ValueError("old source bundle must retain its no-bounds acquisition state")
    _assert_exact_bounds(new)
    if old.n_opt_2d != [OLD_COMPONENTS_PER_VIEW] * len(TRAIN_VIEWS):
        raise ValueError("old bundle no longer contains exactly 7x640 components")
    if new.n_opt_2d != [NEW_COMPONENTS_PER_VIEW] * len(TRAIN_VIEWS):
        raise ValueError("new bundle no longer contains exactly 7x2000 components")

    cameras: list[dict[str, Any]] = []
    windows: list[list[int]] = []
    for view, (old_camera, new_camera, old_field, new_field) in enumerate(
        zip(old.cameras, new.cameras, old.observations, new.observations, strict=True)
    ):
        if _camera_record(old_camera) != _camera_record(new_camera):
            raise ValueError(f"camera mismatch in paired view {view}")
        if old_field.fit_window != new_field.fit_window:
            raise ValueError(f"fit-window mismatch in paired view {view}")
        if (old_field.width, old_field.height) != NATIVE_SIZE:
            raise ValueError(f"old view {view} is not native resolution")
        if (new_field.width, new_field.height) != NATIVE_SIZE:
            raise ValueError(f"new view {view} is not native resolution")
        if old_field.mean_residuals is not None:
            raise ValueError("old bundle unexpectedly contains exact mean residuals")
        if new_field.mean_residuals is None:
            raise ValueError("new bundle must preserve exact mean residuals in every view")
        shared_semantics = (
            "blend_mode",
            "epsilon",
            "sigma_cutoff",
            "support_fade_alpha",
            "aa_dilation",
            "provider",
            "producer_version",
            "producer_source_digest",
        )
        if any(getattr(old_field, name) != getattr(new_field, name) for name in shared_semantics):
            raise ValueError(f"renderer/provider semantics mismatch in paired view {view}")
        if (old_field.color_grads is None) != (new_field.color_grads is None):
            raise ValueError(f"color-gradient schema mismatch in paired view {view}")
        if (old_field.filter_variance is None) != (new_field.filter_variance is None):
            raise ValueError(f"filter-variance schema mismatch in paired view {view}")
        if old_field.n_init != OLD_COMPONENTS_PER_VIEW:
            raise ValueError(f"old n_init mismatch in paired view {view}")
        if new_field.n_init != NEW_COMPONENTS_PER_VIEW:
            raise ValueError(f"new n_init mismatch in paired view {view}")
        cameras.append(_camera_record(old_camera))
        windows.append(list(old_field.fit_window))

    tile_size = lift_config().tile_size
    old_entries = [
        GaussianObservationIndex.estimate_entries(field, tile_size) for field in old.observations
    ]
    new_entries = [
        GaussianObservationIndex.estimate_entries(field, tile_size) for field in new.observations
    ]
    if any(value > INDEX_ENTRY_CAP for value in (*old_entries, *new_entries)):
        raise ValueError("paired teacher index estimate exceeds the shared safety cap")
    return {
        "views": list(TRAIN_VIEWS),
        "heldout_absent": True,
        "points_absent": True,
        "cameras": cameras,
        "fit_windows": windows,
        "residual_contract": {
            "old_640": "absent_legacy_float32_means",
            "new_2000": "present_exact_crop_local_correction",
        },
        "shared_teacher_semantics": {
            "provider": old.observations[0].provider,
            "producer_version": old.observations[0].producer_version,
            "producer_source_digest": old.observations[0].producer_source_digest,
            "blend_mode": old.observations[0].blend_mode,
            "epsilon": old.observations[0].epsilon,
            "sigma_cutoff": old.observations[0].sigma_cutoff,
            "support_fade_alpha": old.observations[0].support_fade_alpha,
            "aa_dilation": old.observations[0].aa_dilation,
            "color_grads_present": old.observations[0].color_grads is not None,
            "filter_variance_present": old.observations[0].filter_variance is not None,
            "fit_config_digests": {
                "old_640": [field.fit_config_digest for field in old.observations],
                "new_2000": [field.fit_config_digest for field in new.observations],
            },
        },
        "bounds": {
            "source": "new_2000 serialized mask-derived bounds",
            "center": list(BOUNDS_CENTER),
            "extent": BOUNDS_EXTENT,
            "injected_bit_exactly_into_old": True,
        },
        "index": {
            "tile_size": tile_size,
            "shared_entry_cap": INDEX_ENTRY_CAP,
            "old_per_view_entries": old_entries,
            "new_per_view_entries": new_entries,
            "old_total_entries": sum(old_entries),
            "new_total_entries": sum(new_entries),
            "ordering_claimed": False,
        },
    }


def inject_new_bounds(
    old: ReconstructionInputs,
    new: ReconstructionInputs,
) -> ReconstructionInputs:
    validate_paired_inputs(old, new)
    assert new.bounds_hint is not None
    center, extent = new.bounds_hint
    result = ReconstructionInputs(
        observations=old.observations,
        cameras=old.cameras,
        view_names=list(old.view_names),
        points=None,
        point_visibility=None,
        bounds_hint=(center.detach().clone(), float(extent)),
        name=f"{old.name}-exact-new-bounds",
        archive_stats=old.archive_stats,
    )
    if not torch.equal(result.bounds_hint[0], center) or result.bounds_hint[1] != extent:
        raise RuntimeError("old arm did not receive exact new bounds")
    return result


def candidate_lineage(
    inputs: ReconstructionInputs,
    config: CompactCarveConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Materialize the exact production component-center proposal contract."""
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    view_ids, component_ids, xy, attempts, per_view = _propose_anchors(
        inputs,
        config,
        generator,
    )
    expected_view_ids = torch.cat(
        [
            torch.full((field.n,), view, dtype=torch.long)
            for view, field in enumerate(inputs.observations)
        ]
    )
    expected_component_ids = torch.cat(
        [torch.arange(field.n, dtype=torch.long) for field in inputs.observations]
    )
    expected_xy = torch.cat(
        [field.native_means(dtype=torch.float64) for field in inputs.observations]
    )
    if not torch.equal(view_ids, expected_view_ids):
        raise RuntimeError("production proposal did not preserve ordered source views")
    if not torch.equal(component_ids, expected_component_ids):
        raise RuntimeError("production proposal did not use every component exactly once")
    if xy.dtype != torch.float64 or not torch.equal(xy, expected_xy):
        raise RuntimeError("production proposal did not use exact float64 native means")
    arrays = {
        "source_view_indices": view_ids.numpy(),
        "source_component_indices": component_ids.numpy(),
        "source_xy": xy.numpy(),
    }
    record = {
        "candidate_count": int(view_ids.numel()),
        "per_view": per_view,
        "attempt_count": attempts,
        "anchor_mode": config.anchor_mode,
        "samples_per_ray": config.samples_per_ray,
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "sha256": array_hash(value),
            }
            for name, value in arrays.items()
        },
        "identity_checks": {
            "every_component_exactly_once": True,
            "ordered_view_lineage": True,
            "native_means_float64": True,
        },
        "candidate_complete_best_depth_audit": {
            "available": "pending production candidate_audit_callback",
            "depth_profile_available": False,
            "scoring_forked_or_duplicated": False,
        },
    }
    return arrays, record


def exact_scoring_backends(
    inputs: ReconstructionInputs,
    config: CompactCarveConfig,
) -> tuple[list[GaussianObservationIndex], list[dict[str, Any]]]:
    backends = [
        GaussianObservationIndex(
            field,
            tile_size=config.tile_size,
            max_entries=config.max_index_entries_per_view,
            max_candidates=config.max_candidates_per_tile,
        )
        for field in inputs.observations
    ]
    records = []
    for view, backend in enumerate(backends):
        if backend.field is not inputs.observations[view]:
            raise RuntimeError("exact index is not bound to its ordered teacher")
        if backend.stats.total_entries != backend.estimated_entries:
            raise RuntimeError("constructed exact index differs from its frozen entry estimate")
        records.append(
            {
                "view_index": view,
                "view_name": inputs.view_names[view],
                "backend": "GaussianObservationIndex",
                "total_entries": backend.stats.total_entries,
                "estimated_entries": backend.estimated_entries,
                "max_candidates_per_tile": backend.stats.max_candidates,
            }
        )
    return backends, records


class SceneRasterGuard:
    """Deny every ordinary file-open route into the calibrated scene during selection."""

    def __init__(self, scene: Path = SCENE):
        self.scene = scene.resolve()
        self.attempts: list[str] = []
        self.forbidden_import_attempts: list[str] = []
        self.forbidden_loader_attempts: list[str] = []
        self.negative_control_denials = 0
        self.loaded_forbidden_modules_at_entry: list[str] = []
        self.loaded_forbidden_modules_at_exit: list[str] = []
        self._probe = False
        self._restores: list[tuple[Any, str, Any]] = []

    @staticmethod
    def _forbidden_module(name: str) -> bool:
        return (
            name == "PIL"
            or name.startswith("PIL.")
            or name == "cv2"
            or name.startswith("cv2.")
            or name == "imageio"
            or name.startswith("imageio.")
            or name in {"rtgs.data.calibrated", "rtgs.data.scene"}
        )

    @classmethod
    def _loaded_forbidden_modules(cls) -> list[str]:
        return sorted(name for name in sys.modules if cls._forbidden_module(name))

    def _is_scene_path(self, value: Any) -> bool:
        if isinstance(value, int):
            return False
        try:
            path = Path(os.fspath(value)).resolve()
            path.relative_to(self.scene)
        except (TypeError, ValueError, OSError):
            return False
        return True

    def _check(self, value: Any) -> None:
        if self._is_scene_path(value):
            if self._probe:
                self.negative_control_denials += 1
            else:
                self.attempts.append(os.fspath(value))
            raise RuntimeError("scene raster/file access is forbidden before selection receipt")

    def _patch(self, owner: Any, name: str, replacement: Any) -> None:
        original = getattr(owner, name)
        self._restores.append((owner, name, original))
        setattr(owner, name, replacement)

    def __enter__(self) -> SceneRasterGuard:
        original_builtin_open = builtins.open
        original_io_open = io.open
        original_os_open = os.open
        original_path_open = Path.open
        original_import = builtins.__import__
        original_import_module = importlib.import_module

        def guarded_builtin(path: Any, *args: Any, **kwargs: Any):
            self._check(path)
            return original_builtin_open(path, *args, **kwargs)

        def guarded_io(path: Any, *args: Any, **kwargs: Any):
            self._check(path)
            return original_io_open(path, *args, **kwargs)

        def guarded_os(path: Any, *args: Any, **kwargs: Any):
            self._check(path)
            return original_os_open(path, *args, **kwargs)

        def guarded_path(path: Path, *args: Any, **kwargs: Any):
            self._check(path)
            return original_path_open(path, *args, **kwargs)

        def guarded_import(
            name: str,
            globals: Mapping[str, Any] | None = None,
            locals: Mapping[str, Any] | None = None,
            fromlist: Sequence[str] = (),
            level: int = 0,
        ):
            if self._forbidden_module(name):
                if self._probe:
                    self.negative_control_denials += 1
                else:
                    self.forbidden_import_attempts.append(name)
                raise ImportError("RGB-capable imports are forbidden before selection receipt")
            return original_import(name, globals, locals, fromlist, level)

        def guarded_import_module(name: str, package: str | None = None):
            if self._forbidden_module(name):
                if self._probe:
                    self.negative_control_denials += 1
                else:
                    self.forbidden_import_attempts.append(name)
                raise ImportError("RGB-capable imports are forbidden before selection receipt")
            return original_import_module(name, package)

        self.loaded_forbidden_modules_at_entry = self._loaded_forbidden_modules()
        self._patch(builtins, "open", guarded_builtin)
        self._patch(io, "open", guarded_io)
        self._patch(os, "open", guarded_os)
        self._patch(Path, "open", guarded_path)
        self._patch(builtins, "__import__", guarded_import)
        self._patch(importlib, "import_module", guarded_import_module)
        self._patch_loaded_decoder_surfaces()

        self._probe = True
        try:
            with contextlib.suppress(RuntimeError):
                guarded_builtin(self.scene / "negative-control.png", "rb")
            with contextlib.suppress(RuntimeError):
                guarded_os(self.scene / "negative-control.png", os.O_RDONLY)
            with contextlib.suppress(ImportError):
                guarded_import("PIL.Image")
            with contextlib.suppress(ImportError):
                guarded_import_module("rtgs.data.calibrated")
        finally:
            self._probe = False
        if self.negative_control_denials != 4:
            self.__exit__()
            raise RuntimeError("scene/RGB denial negative controls did not all fire")
        return self

    def _patch_loaded_decoder_surfaces(self) -> None:
        pil_image = sys.modules.get("PIL.Image")
        if pil_image is not None and hasattr(pil_image, "open"):
            original = pil_image.open

            def guarded_pil(path: Any, *args: Any, _original=original, **kwargs: Any):
                self._check(path)
                return _original(path, *args, **kwargs)

            self._patch(pil_image, "open", guarded_pil)

        cv2 = sys.modules.get("cv2")
        if cv2 is not None and hasattr(cv2, "imread"):
            original = cv2.imread

            def guarded_cv2(path: Any, *args: Any, _original=original, **kwargs: Any):
                self._check(path)
                return _original(path, *args, **kwargs)

            self._patch(cv2, "imread", guarded_cv2)

        for module_name in ("imageio", "imageio.v2", "imageio.v3"):
            module = sys.modules.get(module_name)
            if module is None or not hasattr(module, "imread"):
                continue
            original = module.imread

            def guarded_imageio(path: Any, *args: Any, _original=original, **kwargs: Any):
                self._check(path)
                return _original(path, *args, **kwargs)

            self._patch(module, "imread", guarded_imageio)

        calibrated = sys.modules.get("rtgs.data.calibrated")
        if calibrated is not None:
            for name in ("load_calibrated_scene", "_resize_image"):
                if not hasattr(calibrated, name):
                    continue

                def denied_loader(*args: Any, _name=name, **kwargs: Any):
                    del args, kwargs
                    if self._probe:
                        self.negative_control_denials += 1
                    else:
                        self.forbidden_loader_attempts.append(_name)
                    raise RuntimeError(
                        "calibrated image loaders are forbidden before selection receipt"
                    )

                self._patch(calibrated, name, denied_loader)

    def __exit__(self, *_: Any) -> None:
        for owner, name, original in reversed(self._restores):
            setattr(owner, name, original)
        self._restores.clear()
        self.loaded_forbidden_modules_at_exit = self._loaded_forbidden_modules()

    def record(self) -> dict[str, Any]:
        current_loaded = self._loaded_forbidden_modules()
        return {
            "scene_path_attempts": list(self.attempts),
            "scene_path_attempt_count": len(self.attempts),
            "forbidden_import_attempts": list(self.forbidden_import_attempts),
            "forbidden_loader_attempts": list(self.forbidden_loader_attempts),
            "negative_control_denials": self.negative_control_denials,
            "loaded_forbidden_modules_at_entry": self.loaded_forbidden_modules_at_entry,
            "loaded_forbidden_modules_at_receipt": current_loaded,
            "loaded_forbidden_modules_at_exit": self.loaded_forbidden_modules_at_exit,
            "passed": not self.attempts
            and not self.forbidden_import_attempts
            and not self.forbidden_loader_attempts
            and self.negative_control_denials == 4,
        }


@contextlib.contextmanager
def deny_scene_rasters(scene: Path = SCENE) -> Iterator[SceneRasterGuard]:
    with SceneRasterGuard(scene) as guard:
        yield guard


def _array_record(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "sha256": array_hash(array),
        }
        for name, array in arrays.items()
    }


def _selected_arrays(
    result: CompactInitializationResult,
    audit: CompactCandidateAudit,
) -> dict[str, np.ndarray]:
    selected = audit.selected_candidate_indices
    arrays = {
        "source_view_indices": result.lineage.source_view_indices.cpu().numpy(),
        "source_component_indices": result.lineage.source_component_indices.cpu().numpy(),
        "source_xy": result.lineage.source_xy.cpu().numpy(),
        "selected_depths": result.depths.cpu().numpy(),
        "selected_depth_sigmas": result.depth_sigmas.cpu().numpy(),
        "selected_ray_sigmas": result.ray_sigmas.cpu().numpy(),
        "selected_scores": result.scores.cpu().numpy(),
        "selected_means": result.gaussians.means.cpu().numpy(),
        "selected_valid": audit.candidate_valid_mask[selected].cpu().numpy(),
        "selected_eligible": audit.candidate_eligible_mask[selected].cpu().numpy(),
        "selected_candidate_indices": selected.cpu().numpy(),
    }
    if (
        len(
            set(
                zip(
                    arrays["source_view_indices"].tolist(),
                    arrays["source_component_indices"].tolist(),
                    strict=True,
                )
            )
        )
        != result.gaussians.n
    ):
        raise RuntimeError("selected lineage contains duplicate source components")
    return arrays


def _candidate_audit_arrays(audit: CompactCandidateAudit) -> dict[str, np.ndarray]:
    return {
        name: tensor.detach().contiguous().cpu().numpy() for name, tensor in audit.__dict__.items()
    }


def _validate_candidate_audit(
    audit: CompactCandidateAudit,
    candidate_arrays: Mapping[str, np.ndarray],
    result: CompactInitializationResult,
) -> dict[str, np.ndarray]:
    arrays = _candidate_audit_arrays(audit)
    for expected_name, audit_name in (
        ("source_view_indices", "candidate_source_view_indices"),
        ("source_component_indices", "candidate_source_component_indices"),
        ("source_xy", "candidate_source_xy"),
    ):
        if not np.array_equal(candidate_arrays[expected_name], arrays[audit_name]):
            raise RuntimeError(f"production candidate audit changed {expected_name}")
    selected = audit.selected_candidate_indices
    comparisons = (
        (
            audit.candidate_source_view_indices[selected],
            result.lineage.source_view_indices,
            "selected source views",
        ),
        (
            audit.candidate_source_component_indices[selected],
            result.lineage.source_component_indices,
            "selected source components",
        ),
        (audit.candidate_source_xy[selected], result.lineage.source_xy, "selected source XY"),
        (audit.candidate_best_depths[selected], result.depths, "selected depths"),
        (audit.candidate_depth_sigmas[selected], result.depth_sigmas, "selected depth sigmas"),
        (audit.candidate_best_means[selected], result.gaussians.means, "selected means"),
        (audit.candidate_best_scores[selected], result.scores, "selected scores"),
    )
    for actual, expected, label in comparisons:
        if not torch.equal(actual, expected):
            raise RuntimeError(f"candidate audit differs from initialization: {label}")
    if not bool(audit.candidate_valid_mask[selected].all()):
        raise RuntimeError("selected candidate audit contains an invalid ray")
    if not bool(audit.candidate_eligible_mask[selected].all()):
        raise RuntimeError("selected candidate audit contains an ineligible row")
    if int(audit.candidate_eligible_mask.sum()) != int(
        result.diagnostics["eligible_candidate_count"]
    ):
        raise RuntimeError("candidate eligible mask differs from initialization diagnostics")
    return arrays


def _validate_ray_depth_audit(
    batches: Sequence[CompactRayDepthAuditBatch],
    candidate_arrays: Mapping[str, np.ndarray],
    audit: CompactCandidateAudit,
    config: CompactCarveConfig,
) -> dict[str, np.ndarray]:
    """Assemble the streamed CxS profile and prove its exact production winner relation."""
    expected_start = 0
    tensor_parts: dict[str, list[torch.Tensor]] = {
        "source_view_indices": [],
        "source_component_indices": [],
        "source_xy": [],
        "valid_mask": [],
        "depths": [],
        "scores": [],
        "coverages": [],
        "color_variances": [],
        "n_seen": [],
        "n_covered": [],
        "consensus_colors": [],
    }
    field_map = {
        "source_view_indices": "candidate_source_view_indices",
        "source_component_indices": "candidate_source_component_indices",
        "source_xy": "candidate_source_xy",
        "valid_mask": "candidate_valid_mask",
        "depths": "depths",
        "scores": "scores",
        "coverages": "coverages",
        "color_variances": "color_variances",
        "n_seen": "n_seen",
        "n_covered": "n_covered",
        "consensus_colors": "consensus_colors",
    }
    for batch_index, batch in enumerate(batches):
        if batch.candidate_start != expected_start or batch.candidate_end <= expected_start:
            raise RuntimeError(f"ray-depth audit batch {batch_index} is not contiguous")
        count = batch.candidate_end - batch.candidate_start
        for output_name, field_name in field_map.items():
            tensor = getattr(batch, field_name)
            expected_shape = (
                (count, config.samples_per_ray, 3)
                if output_name == "consensus_colors"
                else (
                    (count, config.samples_per_ray)
                    if output_name
                    in {
                        "depths",
                        "scores",
                        "coverages",
                        "color_variances",
                        "n_seen",
                        "n_covered",
                    }
                    else ((count, 2) if output_name == "source_xy" else (count,))
                )
            )
            if (
                tuple(tensor.shape) != expected_shape
                or tensor.device.type != "cpu"
                or tensor.requires_grad
                or not tensor.is_contiguous()
            ):
                raise RuntimeError(f"ray-depth audit batch {batch_index} has invalid {output_name}")
            tensor_parts[output_name].append(tensor)
        expected_start = batch.candidate_end
    candidate_count = candidate_arrays["source_view_indices"].shape[0]
    if expected_start != candidate_count or not batches:
        raise RuntimeError("ray-depth audit does not cover the complete candidate ordering")
    tensors = {name: torch.cat(parts) for name, parts in tensor_parts.items()}
    for profile_name, candidate_name in (
        ("source_view_indices", "source_view_indices"),
        ("source_component_indices", "source_component_indices"),
        ("source_xy", "source_xy"),
    ):
        expected = torch.from_numpy(candidate_arrays[candidate_name])
        if not torch.equal(tensors[profile_name], expected):
            raise RuntimeError(f"ray-depth audit changed {profile_name}")
    if not torch.equal(tensors["valid_mask"], audit.candidate_valid_mask):
        raise RuntimeError("ray-depth audit validity differs from candidate audit")
    rows = torch.arange(candidate_count)
    winners = audit.candidate_best_depth_indices
    if not torch.equal(tensors["scores"].argmax(dim=1), winners):
        raise RuntimeError("ray-depth profile does not reproduce production winner indices")
    winner_checks = (
        (tensors["depths"][rows, winners], audit.candidate_best_depths, "depths"),
        (tensors["scores"][rows, winners], audit.candidate_best_scores, "scores"),
        (
            tensors["coverages"][rows, winners],
            audit.candidate_best_coverages,
            "coverages",
        ),
        (
            tensors["color_variances"][rows, winners],
            audit.candidate_best_color_variances,
            "color variances",
        ),
        (tensors["n_seen"][rows, winners], audit.candidate_best_n_seen, "n_seen"),
        (
            tensors["n_covered"][rows, winners],
            audit.candidate_best_n_covered,
            "n_covered",
        ),
        (
            tensors["consensus_colors"][rows, winners],
            audit.candidate_consensus_colors,
            "consensus colors",
        ),
    )
    for actual, expected, label in winner_checks:
        if not torch.equal(actual, expected):
            raise RuntimeError(f"ray-depth winner gather differs for {label}")
    without_best = tensors["scores"].clone()
    without_best[rows, winners] = -torch.inf
    second_best = without_best.max(dim=1).values
    if not torch.equal(second_best, audit.candidate_second_best_scores) or not torch.equal(
        audit.candidate_best_scores - second_best,
        audit.candidate_score_margins,
    ):
        raise RuntimeError("ray-depth profile does not reproduce score margins")
    arrays = {name: tensor.detach().contiguous().cpu().numpy() for name, tensor in tensors.items()}
    if arrays["scores"].shape != (candidate_count, config.samples_per_ray):
        raise RuntimeError("ray-depth profile changed the frozen Cx48 shape")
    return arrays


def _save_executed_sources(path: Path, binding: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with tarfile.open(temporary_path, "w") as archive:
            for relative, expected in sorted(binding["files"].items()):
                source = ROOT / relative
                if sha256_file(source) != expected:
                    raise RuntimeError(f"source changed before archival: {relative}")
                info = archive.gettarinfo(str(source), arcname=relative)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                with source.open("rb") as stream:
                    archive.addfile(info, stream)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return sha256_file(path)


def _verify_executed_sources_archive(
    path: Path,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay every archived source member against its canonical plan-bound digest."""
    mode = os.lstat(path).st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError("executed-source archive must be an ordinary file")
    expected = binding.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("source binding does not contain an exact file map")
    observed: dict[str, str] = {}
    try:
        with tarfile.open(path, "r:") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise ValueError("executed-source archive contains duplicate member names")
            if set(names) != set(expected):
                raise ValueError("executed-source archive member set differs from frozen source")
            for member in members:
                if not member.isfile() or member.linkname:
                    raise ValueError(
                        f"executed-source archive member is not a regular file: {member.name}"
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(
                        f"executed-source archive member cannot be read: {member.name}"
                    )
                digest = hashlib.sha256()
                size = 0
                while block := stream.read(1 << 20):
                    size += len(block)
                    digest.update(block)
                if size != member.size:
                    raise ValueError(f"executed-source archive member size changed: {member.name}")
                observed[member.name] = digest.hexdigest()
    except tarfile.TarError as error:
        raise ValueError("executed-source archive is not a valid plain tar") from error
    if observed != expected:
        raise ValueError("executed-source archive bytes differ from frozen source")
    return {
        "member_count": len(observed),
        "member_hashes_sha256": canonical_hash(observed),
        "source_aggregate_sha256": binding["aggregate_sha256"],
    }


def assert_paired_sampling(
    old_history: Mapping[str, Any],
    new_history: Mapping[str, Any],
) -> dict[str, Any]:
    old_steps = old_history.get("steps")
    new_steps = new_history.get("steps")
    if not isinstance(old_steps, list) or not isinstance(new_steps, list):
        raise ValueError("paired histories must contain step lists")
    if len(old_steps) != len(new_steps) or len(old_steps) != train_config().iterations:
        raise ValueError("paired histories do not contain the frozen iteration count")
    for index, (old, new) in enumerate(zip(old_steps, new_steps, strict=True), start=1):
        for key in SAMPLE_HASH_KEYS:
            if old.get(key) != new.get(key):
                raise RuntimeError(f"paired sampling differs at step {index} key {key}")
    return {
        "identical": True,
        "steps": len(old_steps),
        "keys": list(SAMPLE_HASH_KEYS),
        "aggregate_sha256": canonical_hash(
            [{key: step[key] for key in SAMPLE_HASH_KEYS} for step in old_steps]
        ),
    }


def _verify_frozen_plan(out: Path) -> tuple[dict[str, Any], str]:
    plan_path = out / "plan.json"
    plan = _strict_json(plan_path)
    digest = sha256_file(plan_path)
    if plan.get("artifact_type") != f"{SCHEMA}_plan_v1":
        raise ValueError("wrong crossover plan schema")
    if plan.get("plan_file_sha256") is not None:
        raise ValueError("plan must not contain a recursive file hash")
    if plan.get("source_binding") != source_binding():
        raise RuntimeError("bound repository sources changed after planning")
    if plan.get("environment_binding") != environment_binding():
        raise RuntimeError("bound CPU environment changed after planning")
    if plan.get("cpu_phase_runtime") != _cpu_phase_runtime_record():
        raise RuntimeError("frozen CPU-phase preload policy changed after planning")
    if plan.get("configuration") != config_record():
        raise RuntimeError("frozen crossover configuration changed after planning")
    expected_bundles = {
        "old_640": bundle_binding(bound_path(plan["bundles"]["old_640"]["path"])),
        "new_2000": bundle_binding(bound_path(plan["bundles"]["new_2000"]["path"])),
    }
    if expected_bundles != plan["bundles"]:
        raise RuntimeError("one or both frozen input bundle trees changed after planning")
    return plan, digest


def prepare_plan(
    out: Path,
    *,
    old_bundle: Path = OLD_BUNDLE,
    new_bundle: Path = NEW_BUNDLE,
) -> dict[str, Any]:
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to overwrite crossover namespace: {out}")
    out.mkdir(parents=True)
    try:
        cpu_phase_runtime = _assert_cpu_phase_runtime()
        old_binding = bundle_binding(old_bundle)
        new_binding = bundle_binding(new_bundle)
        old = ReconstructionInputs.load(old_bundle, device="cpu", strict=True)
        new = ReconstructionInputs.load(new_bundle, device="cpu", strict=True)
        paired = validate_paired_inputs(old, new)
        injected = inject_new_bounds(old, new)
        assert injected.bounds_hint is not None and new.bounds_hint is not None
        if not torch.equal(injected.bounds_hint[0], new.bounds_hint[0]):
            raise RuntimeError("planning bounds injection was not bit exact")

        plan = {
            "artifact_type": f"{SCHEMA}_plan_v1",
            "decision_bearing": False,
            "scope": "exploratory fixed-835 teacher-artifact fidelity/capacity crossover",
            "hypothesis": (
                "the complete new 2D teacher artifacts improve exact-index CompactCarve "
                "initialization and identical RGB-free fixed-topology refinement"
            ),
            "causal_estimand": (
                "new_2000 artifact bundle minus old_640 artifact bundle; not a component-count-"
                "only effect"
            ),
            "source_binding": source_binding(),
            "environment_binding": environment_binding(),
            "cpu_phase_runtime": cpu_phase_runtime,
            "configuration": config_record(),
            "bundles": {"old_640": old_binding, "new_2000": new_binding},
            "paired_input_contract": paired,
            "phases": ["plan", "select", "evaluate", "view-smoke"],
            "rgb_boundary": {
                "selection": "deny every file under calibrated scene",
                "unlock_condition": "atomic selection_receipt.json exists and verifies",
                "evaluation": "native RGB and masks reporting-only",
            },
            "evaluation": {
                "selection_metrics": False,
                "renderer": "rtgs exact native gsplat",
                "packed": False,
                "antialiased": False,
                "train_views": list(TRAIN_VIEWS),
                "heldout_view": HELDOUT_VIEW,
                "outcome_threshold": None,
            },
        }
        _write_json_exclusive(out / "plan.json", plan)
        return plan
    except BaseException as error:
        _failure(out, "plan", error)
        raise


def _arm_paths(out: Path, arm: str) -> dict[str, Path]:
    root = out / "arms" / arm
    return {
        "root": root,
        "candidate": root / "candidate_audit.npz",
        "ray_depth_profiles": root / "ray_depth_profiles.npz",
        "selected": root / "selected_lineage.npz",
        "initial": root / "gaussians_init.ply",
        "final": root / "gaussians.ply",
        "history": root / "history.json",
        "receipt": root / "receipt.json",
    }


def _select_arm(
    *,
    out: Path,
    arm: str,
    inputs: ReconstructionInputs,
    bundle_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = _arm_paths(out, arm)
    paths["root"].mkdir(parents=True, exist_ok=False)
    lift_cfg = lift_config()
    candidate_arrays, candidate = candidate_lineage(inputs, lift_cfg)
    expected = OLD_CANDIDATES if arm == "old_640" else NEW_CANDIDATES
    if candidate["candidate_count"] != expected:
        raise RuntimeError(f"{arm} candidate count differs from frozen {expected}")

    backends, index_records = exact_scoring_backends(inputs, lift_cfg)
    audits: list[CompactCandidateAudit] = []
    ray_depth_batches: list[CompactRayDepthAuditBatch] = []
    started = time.perf_counter()
    initialization = CompactCarveInitializer(lift_cfg).initialize(
        inputs,
        backends=backends,
        candidate_audit_callback=audits.append,
        ray_depth_audit_callback=ray_depth_batches.append,
    )
    lift_seconds = time.perf_counter() - started
    if len(audits) != 1:
        raise RuntimeError("production initializer did not emit exactly one candidate audit")
    full_candidate_arrays = _validate_candidate_audit(
        audits[0],
        candidate_arrays,
        initialization,
    )
    ray_depth_arrays = _validate_ray_depth_audit(
        ray_depth_batches,
        candidate_arrays,
        audits[0],
        lift_cfg,
    )
    ray_depth_archive_sha = _write_npz_exclusive(
        paths["ray_depth_profiles"],
        ray_depth_arrays,
    )
    candidate["candidate_complete_best_depth_audit"] = {
        "available": True,
        "source": "production CompactCarveInitializer candidate_audit_callback",
        "depth_profile_available": True,
        "scope": (
            "one production winner and eligibility decision per proposed ray, exactly gathered "
            "from the separately persisted complete 48-depth production profile"
        ),
        "scoring_forked_or_duplicated": False,
        "arrays": _array_record(full_candidate_arrays),
    }
    candidate["complete_ray_depth_profile"] = {
        "available": True,
        "source": "production CompactCarveInitializer ray_depth_audit_callback",
        "samples_per_ray": lift_cfg.samples_per_ray,
        "candidate_count": candidate["candidate_count"],
        "batch_count": len(ray_depth_batches),
        "contiguous_complete_order": True,
        "winner_gathers_exact": True,
        "arrays": _array_record(ray_depth_arrays),
    }
    candidate_archive_sha = _write_npz_exclusive(
        paths["candidate"],
        full_candidate_arrays,
    )
    if initialization.gaussians.n != N_INIT_3D:
        raise RuntimeError("CompactCarve changed the frozen initialization budget")
    selected_arrays = _selected_arrays(initialization, audits[0])
    selected_archive_sha = _write_npz_exclusive(paths["selected"], selected_arrays)
    initial_sha = _save_ply_exclusive(paths["initial"], initialization.gaussians)

    trainer = CompactTrainer(train_config())
    started = time.perf_counter()
    final, history = trainer.train(
        inputs,
        initialization.gaussians,
        bundle_path=bundle_path,
    )
    train_seconds = time.perf_counter() - started
    if final.n != N_INIT_3D or history["n_init_3d"] != N_INIT_3D:
        raise RuntimeError("fixed-topology refinement changed Gaussian cardinality")
    history_sha = _write_json_exclusive(paths["history"], history)
    final_sha = _save_ply_exclusive(paths["final"], final)
    record = {
        "status": "PASS",
        "arm": arm,
        "bundle_path": display_path(bundle_path),
        "teacher_components_per_view": inputs.n_opt_2d,
        "bounds_center_sha256": tensor_hash(inputs.bounds_hint[0]),
        "bounds_extent": inputs.bounds_hint[1],
        "candidate_contract": candidate,
        "index_backends": index_records,
        "initialization_diagnostics": initialization.diagnostics,
        "selected_arrays": _array_record(selected_arrays),
        "n_init_3d": initialization.gaussians.n,
        "n_opt_3d": final.n,
        "timing_seconds": {
            "lift": lift_seconds,
            "refinement": train_seconds,
        },
        "artifacts": {
            "candidate_audit": display_path(paths["candidate"]),
            "candidate_audit_sha256": candidate_archive_sha,
            "ray_depth_profiles": display_path(paths["ray_depth_profiles"]),
            "ray_depth_profiles_sha256": ray_depth_archive_sha,
            "selected_lineage": display_path(paths["selected"]),
            "selected_lineage_sha256": selected_archive_sha,
            "gaussians_init": display_path(paths["initial"]),
            "gaussians_init_sha256": initial_sha,
            "history": display_path(paths["history"]),
            "history_sha256": history_sha,
            "gaussians": display_path(paths["final"]),
            "gaussians_sha256": final_sha,
        },
    }
    _write_json_exclusive(paths["receipt"], record)
    return record, history


def run_selection(out: Path) -> dict[str, Any]:
    try:
        cpu_phase_runtime = _assert_cpu_phase_runtime()
        plan, plan_sha = _verify_frozen_plan(out)
        if (out / "selection_receipt.json").exists():
            raise FileExistsError("selection receipt already exists")
        old_path = bound_path(plan["bundles"]["old_640"]["path"])
        new_path = bound_path(plan["bundles"]["new_2000"]["path"])
        with deny_scene_rasters() as guard:
            old_source = ReconstructionInputs.load(old_path, device="cpu", strict=True)
            new = ReconstructionInputs.load(new_path, device="cpu", strict=True)
            paired = validate_paired_inputs(old_source, new)
            old = inject_new_bounds(old_source, new)

            old_record, old_history = _select_arm(
                out=out,
                arm="old_640",
                inputs=old,
                bundle_path=old_path,
            )
            new_record, new_history = _select_arm(
                out=out,
                arm="new_2000",
                inputs=new,
                bundle_path=new_path,
            )
            paired_sampling = assert_paired_sampling(old_history, new_history)
            if bundle_binding(old_path) != plan["bundles"]["old_640"]:
                raise RuntimeError("old bundle changed during selection")
            if bundle_binding(new_path) != plan["bundles"]["new_2000"]:
                raise RuntimeError("new bundle changed during selection")
            if source_binding() != plan["source_binding"]:
                raise RuntimeError("bound source changed during selection")
            source_archive = out / "artifacts" / "EXECUTED_SOURCES.tar"
            source_archive_sha = _save_executed_sources(
                source_archive,
                plan["source_binding"],
            )
            source_archive_verification = _verify_executed_sources_archive(
                source_archive,
                plan["source_binding"],
            )
            receipt = {
                "artifact_type": f"{SCHEMA}_selection_receipt_v1",
                "status": "PASS",
                "plan_sha256": plan_sha,
                "cpu_phase_runtime": cpu_phase_runtime,
                "paired_input_contract": paired,
                "source_denial_before_receipt": guard.record(),
                "heldout_present_in_selection": False,
                "selection_metrics_used": False,
                "paired_sampling": paired_sampling,
                "arms": {"old_640": old_record, "new_2000": new_record},
                "executed_sources": {
                    "path": display_path(source_archive),
                    "sha256": source_archive_sha,
                    "source_aggregate_sha256": plan["source_binding"]["aggregate_sha256"],
                    "verification": source_archive_verification,
                },
                "conditional_next_arm": plan["configuration"]["protocol"]["conditional_next_arm"],
            }
            receipt["selection_payload_sha256"] = canonical_hash(receipt)
            _write_json_exclusive(out / "selection_receipt.json", receipt)
        if guard.attempts:
            raise RuntimeError("selection attempted forbidden calibrated scene access")
        if not guard.record()["passed"]:
            raise RuntimeError("selection source-denial contract did not pass")
        return receipt
    except BaseException as error:
        _failure(out, "select", error)
        raise


def _verified_selection(out: Path) -> tuple[dict[str, Any], str]:
    plan, plan_sha = _verify_frozen_plan(out)
    path = out / "selection_receipt.json"
    selection = _strict_json(path)
    digest = sha256_file(path)
    if selection.get("artifact_type") != f"{SCHEMA}_selection_receipt_v1":
        raise ValueError("wrong selection receipt schema")
    if selection.get("status") != "PASS" or selection.get("plan_sha256") != plan_sha:
        raise ValueError("selection receipt does not bind the frozen plan")
    if selection.get("cpu_phase_runtime") != _cpu_phase_runtime_record() or selection.get(
        "cpu_phase_runtime"
    ) != plan.get("cpu_phase_runtime"):
        raise ValueError("selection receipt does not bind the CPU-phase preload policy")
    claimed = selection.get("selection_payload_sha256")
    payload = dict(selection)
    payload.pop("selection_payload_sha256", None)
    if claimed != canonical_hash(payload):
        raise ValueError("selection receipt payload hash mismatch")
    denial = selection.get("source_denial_before_receipt")
    if not isinstance(denial, dict) or not denial.get("passed"):
        raise ValueError("selection receipt records forbidden scene access")
    for arm in ("old_640", "new_2000"):
        artifacts = selection["arms"][arm]["artifacts"]
        for key, digest_key in (
            ("candidate_audit", "candidate_audit_sha256"),
            ("ray_depth_profiles", "ray_depth_profiles_sha256"),
            ("selected_lineage", "selected_lineage_sha256"),
            ("gaussians_init", "gaussians_init_sha256"),
            ("history", "history_sha256"),
            ("gaussians", "gaussians_sha256"),
        ):
            if sha256_file(_artifact_path_within(out, artifacts[key])) != artifacts[digest_key]:
                raise RuntimeError(f"committed {arm} artifact changed: {key}")
    executed = selection.get("executed_sources")
    if (
        not isinstance(executed, dict)
        or executed.get("source_aggregate_sha256") != plan["source_binding"]["aggregate_sha256"]
    ):
        raise ValueError("selection executed-source archive does not bind the frozen source")
    executed_path = _artifact_path_within(out, executed["path"])
    if sha256_file(executed_path) != executed.get("sha256"):
        raise RuntimeError("selection executed-source archive changed")
    if executed.get("verification") != _verify_executed_sources_archive(
        executed_path,
        plan["source_binding"],
    ):
        raise RuntimeError("selection executed-source semantic verification changed")
    return selection, digest


def _metric_deltas(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    pooled = evaluation["pooled"]

    def psnr(model: str, split: str, stratum: str) -> float:
        return float(pooled[model][split][stratum]["psnr_db"])

    result: dict[str, Any] = {}
    for split in ("train", "heldout"):
        result[split] = {}
        for stratum in ("all", "foreground"):
            result[split][stratum] = {
                "teacher_artifact_delta_init_new_minus_old_db": (
                    psnr("new_init", split, stratum) - psnr("old_init", split, stratum)
                ),
                "teacher_artifact_delta_final_new_minus_old_db": (
                    psnr("new_final", split, stratum) - psnr("old_final", split, stratum)
                ),
                "old_refinement_final_minus_init_db": (
                    psnr("old_final", split, stratum) - psnr("old_init", split, stratum)
                ),
                "new_refinement_final_minus_init_db": (
                    psnr("new_final", split, stratum) - psnr("new_init", split, stratum)
                ),
            }
    return result


def _assert_evaluation_abi() -> dict[str, Any]:
    expected = ABI_PRELOAD.resolve(strict=True)
    effective = os.environ.get("LD_PRELOAD", "")
    entries = [value for value in effective.split(":") if value]
    resolved = [Path(value).resolve() for value in entries]
    if len(entries) != 1 or resolved != [expected]:
        raise RuntimeError(
            "native gsplat evaluation requires the versioned libstdc++ as the first and sole "
            f"LD_PRELOAD entry: {expected}; got {effective!r}"
        )
    return {
        "effective_ld_preload": effective,
        "first_and_sole_entry": True,
        "required_path": str(expected),
        "required_sha256": sha256_file(expected),
        "environment": environment_binding(),
    }


def _evaluation_input_files() -> dict[str, str]:
    paths = [CALIBRATION]
    for view_id in EVALUATION_VIEWS:
        paths.extend(
            (
                SCENE / "rgb" / f"{view_id}.jpg",
                SCENE / "mask" / f"mask_{view_id}.png",
            )
        )
    return {display_path(path): sha256_file(path) for path in paths}


def _acquire_evaluation_targets(
    out: Path,
    *,
    inputs: ReconstructionInputs,
    selection_sha: str,
    runtime_abi: Mapping[str, Any],
) -> tuple[
    dict[str, tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]],
    dict[str, Any],
    str,
]:
    """Decode each exact reporting target once, then atomically lock its complete binding."""
    acquisition_path = out / "evaluation_acquisition.json"
    if acquisition_path.exists() or acquisition_path.is_symlink():
        raise FileExistsError("evaluation acquisition receipt already exists")
    from benchmarks import compact_bounds_crossover as bounds

    cameras = dict(zip(inputs.view_names, inputs.cameras, strict=True))
    heldout_camera, _ = bounds.camera_from_calibration(HELDOUT_VIEW)
    cameras[HELDOUT_VIEW] = heldout_camera
    files_before = _evaluation_input_files()
    targets: dict[str, tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    for view_id, split in zip(EVALUATION_VIEWS, EVALUATION_SPLITS, strict=True):
        target, mask, target_record = bounds._load_evaluation_target(
            view_id,
            cameras[view_id],
        )
        record = dict(target_record)
        record["split"] = split
        if (
            record.get("view_id") != view_id
            or record.get("decoded_rgb_tensor_sha256") != tensor_hash(target)
            or record.get("decoded_mask_tensor_sha256") != tensor_hash(mask)
        ):
            raise RuntimeError(f"decoded evaluation target did not self-bind for {view_id}")
        targets[view_id] = (target, mask, record)
        records.append(record)
    files_after = _evaluation_input_files()
    if files_after != files_before:
        raise RuntimeError("evaluation calibration/RGB/mask files changed during acquisition")
    receipt = {
        "artifact_type": f"{SCHEMA}_evaluation_acquisition_v1",
        "status": "PASS",
        "selection_receipt_file_sha256": selection_sha,
        "view_order": list(EVALUATION_VIEWS),
        "splits": list(EVALUATION_SPLITS),
        "calibration": {
            "path": display_path(CALIBRATION),
            "sha256": files_before[display_path(CALIBRATION)],
        },
        "input_files": files_before,
        "views": records,
        "source_aggregate_sha256": source_binding()["aggregate_sha256"],
        "runtime_abi": dict(runtime_abi),
        "decoded_once_per_view_in_this_phase": True,
    }
    receipt["acquisition_payload_sha256"] = canonical_hash(receipt)
    digest = _write_json_exclusive(acquisition_path, receipt)
    return targets, receipt, digest


def _verify_evaluation_acquisition(
    out: Path,
    *,
    selection_sha: str,
    targets: Mapping[str, tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]] | None = None,
    runtime_abi: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    path = out / "evaluation_acquisition.json"
    receipt = _strict_json(path)
    digest = sha256_file(path)
    claimed = receipt.get("acquisition_payload_sha256")
    payload = dict(receipt)
    payload.pop("acquisition_payload_sha256", None)
    if (
        receipt.get("artifact_type") != f"{SCHEMA}_evaluation_acquisition_v1"
        or receipt.get("status") != "PASS"
        or receipt.get("selection_receipt_file_sha256") != selection_sha
        or claimed != canonical_hash(payload)
    ):
        raise ValueError("evaluation acquisition receipt failed its frozen binding")
    if receipt.get("view_order") != list(EVALUATION_VIEWS) or receipt.get("splits") != list(
        EVALUATION_SPLITS
    ):
        raise ValueError("evaluation acquisition view protocol changed")
    if receipt.get("input_files") != _evaluation_input_files():
        raise RuntimeError("evaluation calibration/RGB/mask files changed after acquisition")
    if receipt.get("source_aggregate_sha256") != source_binding()["aggregate_sha256"]:
        raise RuntimeError("bound source changed after evaluation acquisition")
    if runtime_abi is not None and receipt.get("runtime_abi") != dict(runtime_abi):
        raise RuntimeError("native evaluation runtime changed after acquisition")
    records = receipt.get("views")
    if not isinstance(records, list) or len(records) != len(EVALUATION_VIEWS):
        raise ValueError("evaluation acquisition view records are incomplete")
    for index, (record, view_id, split) in enumerate(
        zip(records, EVALUATION_VIEWS, EVALUATION_SPLITS, strict=True)
    ):
        rgb_path = display_path(SCENE / "rgb" / f"{view_id}.jpg")
        mask_path = display_path(SCENE / "mask" / f"mask_{view_id}.png")
        if (
            not isinstance(record, dict)
            or record.get("view_id") != view_id
            or record.get("split") != split
            or record.get("calibration") != receipt.get("calibration")
            or record.get("rgb")
            != {"path": rgb_path, "sha256": receipt["input_files"].get(rgb_path)}
            or record.get("mask")
            != {"path": mask_path, "sha256": receipt["input_files"].get(mask_path)}
            or not isinstance(record.get("distortion_coefficients"), list)
            or not isinstance(record.get("camera"), dict)
            or not isinstance(record.get("decoded_rgb_tensor_sha256"), str)
            or len(record["decoded_rgb_tensor_sha256"]) != 64
            or not isinstance(record.get("decoded_mask_tensor_sha256"), str)
            or len(record["decoded_mask_tensor_sha256"]) != 64
        ):
            raise ValueError(f"evaluation acquisition record {index} changed")
        if targets is not None:
            target, mask, local_record = targets[view_id]
            if (
                tensor_hash(target) != record.get("decoded_rgb_tensor_sha256")
                or tensor_hash(mask) != record.get("decoded_mask_tensor_sha256")
                or dict(local_record) != record
            ):
                raise RuntimeError(f"decoded evaluation tensors changed for {view_id}")
    return receipt, digest


def run_evaluation(out: Path) -> dict[str, Any]:
    try:
        selection, selection_sha = _verified_selection(out)
        if any(
            (out / name).exists()
            for name in ("evaluation_acquisition.json", "evaluation.json", "result.json")
        ):
            raise FileExistsError("evaluation acquisition/result receipt already exists")
        abi = _assert_evaluation_abi()
        from benchmarks.compact_bounds_crossover import exact_native_evaluation

        models: dict[str, Gaussians3D] = {}
        for arm, prefix in (("old_640", "old"), ("new_2000", "new")):
            artifacts = selection["arms"][arm]["artifacts"]
            models[f"{prefix}_init"] = Gaussians3D.load_ply(ROOT / artifacts["gaussians_init"])
            models[f"{prefix}_final"] = Gaussians3D.load_ply(ROOT / artifacts["gaussians"])
        if any(model.n != N_INIT_3D for model in models.values()):
            raise RuntimeError("evaluation PLY changed fixed Gaussian cardinality")
        plan = _strict_json(out / "plan.json")
        new_bundle = bound_path(plan["bundles"]["new_2000"]["path"])
        inputs = ReconstructionInputs.load(new_bundle, device="cpu", strict=True)
        validate_paired_inputs(
            ReconstructionInputs.load(
                bound_path(plan["bundles"]["old_640"]["path"]),
                device="cpu",
                strict=True,
            ),
            inputs,
        )
        targets, acquisition, acquisition_sha = _acquire_evaluation_targets(
            out,
            inputs=inputs,
            selection_sha=selection_sha,
            runtime_abi=abi,
        )
        evaluation = exact_native_evaluation(
            output=out,
            arm_models=models,
            train_inputs=inputs,
            selection_receipt_sha256=selection_sha,
            evaluation_targets=targets,
        )
        _, final_selection_sha = _verified_selection(out)
        if final_selection_sha != selection_sha:
            raise RuntimeError("selection receipt changed during native evaluation")
        _verify_evaluation_acquisition(
            out,
            selection_sha=selection_sha,
            targets=targets,
            runtime_abi=abi,
        )
        if source_binding() != plan["source_binding"]:
            raise RuntimeError("bound source changed during native evaluation")
        if (
            bundle_binding(bound_path(plan["bundles"]["old_640"]["path"]))
            != plan["bundles"]["old_640"]
            or bundle_binding(bound_path(plan["bundles"]["new_2000"]["path"]))
            != plan["bundles"]["new_2000"]
        ):
            raise RuntimeError("frozen teacher bundles changed during native evaluation")
        evaluation["artifact_type"] = f"{SCHEMA}_native_gsplat_evaluation_v1"
        evaluation["selection_receipt_file_sha256"] = selection_sha
        evaluation["evaluation_acquisition"] = {
            "path": display_path(out / "evaluation_acquisition.json"),
            "sha256": acquisition_sha,
            "payload_sha256": acquisition["acquisition_payload_sha256"],
        }
        evaluation["descriptive_only"] = True
        evaluation["outcome_threshold"] = None
        evaluation["runtime_abi"] = abi
        evaluation["deltas"] = _metric_deltas(evaluation)
        evaluation["evaluation_payload_sha256"] = canonical_hash(evaluation)
        evaluation_sha = _write_json_exclusive(out / "evaluation.json", evaluation)
        result = {
            "artifact_type": f"{SCHEMA}_result_v1",
            "status": "PASS",
            "decision_bearing": False,
            "selection_receipt_file_sha256": selection_sha,
            "evaluation": {
                "path": display_path(out / "evaluation.json"),
                "sha256": evaluation_sha,
                "pooled": evaluation["pooled"],
                "deltas": evaluation["deltas"],
            },
            "viewer_command": viewer_command(out, port=8_881),
            "interpretation": (
                "descriptive teacher-artifact fidelity/capacity crossover; the intervention "
                "is not a component-count-only effect and has no default-changing threshold"
            ),
            "conditional_next_arm": selection["conditional_next_arm"],
        }
        result["result_payload_sha256"] = canonical_hash(result)
        _write_json_exclusive(out / "result.json", result)
        verified, verified_sha = _verified_evaluation(out, selection_sha)
        if verified_sha != evaluation_sha:
            raise RuntimeError("published native evaluation hash changed during final verification")
        return verified
    except BaseException as error:
        _failure(out, "evaluate", error)
        raise


def _artifact_path_within(out: Path, stored: str) -> Path:
    path = bound_path(stored)
    try:
        path.relative_to(out.resolve())
    except ValueError as error:
        raise ValueError(f"evaluation artifact escapes output namespace: {stored!r}") from error
    mode = os.lstat(path).st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"evaluation artifact is not an ordinary file: {stored!r}")
    return path


def _recompute_pooled(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pooled: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        pooled[model_name] = {}
        for split in ("train", "heldout"):
            selected = [record for record in records if record["split"] == split]
            pooled[model_name][split] = {}
            for stratum in ("all", "foreground"):
                sse = sum(
                    record["models"][model_name]["metrics"][stratum]["sse"] for record in selected
                )
                count = sum(
                    record["models"][model_name]["metrics"][stratum]["scalar_count"]
                    for record in selected
                )
                mse = sse / count
                pooled[model_name][split][stratum] = {
                    "mse": mse,
                    "psnr_db": -10.0 * math.log10(max(mse, 1e-30)),
                }
    return pooled


def _verify_native_artifacts(
    out: Path,
    evaluation: Mapping[str, Any],
    acquisition: Mapping[str, Any],
) -> None:
    records = evaluation.get("records")
    if not isinstance(records, list) or len(records) != len(EVALUATION_VIEWS):
        raise ValueError("native evaluation records are incomplete")
    acquisition_views = acquisition["views"]
    render_count = 0
    for index, (record, view_id, split) in enumerate(
        zip(records, EVALUATION_VIEWS, EVALUATION_SPLITS, strict=True)
    ):
        if (
            not isinstance(record, dict)
            or record.get("view_id") != view_id
            or record.get("split") != split
            or record.get("target") != acquisition_views[index]
        ):
            raise ValueError(f"native evaluation view record {index} changed")
        models = record.get("models")
        if not isinstance(models, dict) or set(models) != set(MODEL_NAMES):
            raise ValueError(f"native evaluation model set changed for {view_id}")
        for model_name in MODEL_NAMES:
            model = models[model_name]
            if (
                model.get("n_gaussians") != N_INIT_3D
                or model.get("backend") != "rtgs.render.gsplat_backend.GsplatRasterizer"
                or not str(model.get("device", "")).startswith("cuda")
                or model.get("packed") is not False
                or model.get("antialiased") is not False
            ):
                raise ValueError(f"native execution contract changed for {view_id}/{model_name}")
            render = model.get("render")
            if not isinstance(render, dict):
                raise ValueError(f"missing native render record for {view_id}/{model_name}")
            path = _artifact_path_within(out, render["path"])
            if (
                sha256_file(path) != render.get("sha256")
                or path.stat().st_size != render.get("bytes")
                or render.get("dimensions") != list(NATIVE_SIZE)
            ):
                raise RuntimeError(f"native render artifact changed for {view_id}/{model_name}")
            metrics = model.get("metrics")
            if not isinstance(metrics, dict) or set(metrics) != {"all", "foreground"}:
                raise ValueError(f"native metrics changed for {view_id}/{model_name}")
            for stratum in ("all", "foreground"):
                values = metrics[stratum]
                sse = values.get("sse")
                count = values.get("scalar_count")
                if (
                    not isinstance(sse, (int, float))
                    or not math.isfinite(float(sse))
                    or float(sse) < 0.0
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count <= 0
                ):
                    raise ValueError(f"invalid native metrics for {view_id}/{model_name}/{stratum}")
                mse = float(sse) / count
                if values.get("mse") != mse or values.get("psnr_db") != -10.0 * math.log10(
                    max(mse, 1e-30)
                ):
                    raise ValueError(
                        f"native metric arithmetic changed for {view_id}/{model_name}/{stratum}"
                    )
            render_count += 1
    if render_count != len(EVALUATION_VIEWS) * len(MODEL_NAMES):
        raise RuntimeError("native evaluation did not bind exactly 32 renders")
    if evaluation.get("pooled") != _recompute_pooled(records):
        raise ValueError(
            "native pooled metrics do not recompute from per-view sufficient statistics"
        )
    if evaluation.get("deltas") != _metric_deltas(evaluation):
        raise ValueError("native evaluation deltas do not recompute from pooled metrics")
    contact = evaluation.get("contact_sheet")
    if not isinstance(contact, dict):
        raise ValueError("native evaluation contact sheet record is missing")
    contact_path = _artifact_path_within(out, contact["path"])
    if (
        sha256_file(contact_path) != contact.get("sha256")
        or contact.get("rows") != ["target", *MODEL_NAMES]
        or contact.get("columns") != list(EVALUATION_VIEWS)
    ):
        raise RuntimeError("native evaluation contact sheet changed")


def _verified_evaluation(out: Path, selection_sha: str) -> tuple[dict[str, Any], str]:
    path = out / "evaluation.json"
    evaluation = _strict_json(path)
    digest = sha256_file(path)
    if evaluation.get("artifact_type") != f"{SCHEMA}_native_gsplat_evaluation_v1":
        raise ValueError("wrong native evaluation schema")
    if evaluation.get("selection_receipt_file_sha256") != selection_sha:
        raise ValueError("native evaluation does not bind the selection receipt")
    claimed = evaluation.get("evaluation_payload_sha256")
    payload = dict(evaluation)
    payload.pop("evaluation_payload_sha256", None)
    if claimed != canonical_hash(payload):
        raise ValueError("native evaluation payload hash mismatch")
    acquisition, acquisition_sha = _verify_evaluation_acquisition(
        out,
        selection_sha=selection_sha,
    )
    if evaluation.get("evaluation_acquisition") != {
        "path": display_path(out / "evaluation_acquisition.json"),
        "sha256": acquisition_sha,
        "payload_sha256": acquisition["acquisition_payload_sha256"],
    }:
        raise ValueError("native evaluation does not bind its acquisition receipt")
    _verify_native_artifacts(out, evaluation, acquisition)
    result = _strict_json(out / "result.json")
    result_claimed = result.get("result_payload_sha256")
    result_payload = dict(result)
    result_payload.pop("result_payload_sha256", None)
    if (
        result.get("artifact_type") != f"{SCHEMA}_result_v1"
        or result.get("status") != "PASS"
        or result_claimed != canonical_hash(result_payload)
        or result["evaluation"]["sha256"] != digest
        or result.get("selection_receipt_file_sha256") != selection_sha
        or result["evaluation"].get("path") != display_path(path)
        or result["evaluation"].get("pooled") != evaluation["pooled"]
        or result["evaluation"].get("deltas") != evaluation["deltas"]
        or result.get("viewer_command") != viewer_command(out, port=8_881)
    ):
        raise ValueError("result receipt does not bind the native evaluation")
    return evaluation, digest


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _assert_port_available(port: int) -> None:
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65_535:
        raise ValueError("viewer port must lie in [1,65535]")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(
                f"viewer port {port} is already occupied; refusing ambiguous HTTP evidence"
            ) from error


def _process_owns_listening_port(pid: int, port: int) -> bool:
    """Return whether ``pid`` owns a Linux listening socket on the requested port."""
    if pid <= 0 or port <= 0:
        return False
    inodes: set[str] = set()
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = table.read_text(encoding="ascii").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                local_port = int(fields[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if local_port == port:
                inodes.add(fields[9])
    if not inodes:
        return False
    try:
        descriptors = Path(f"/proc/{pid}/fd")
        for descriptor in descriptors.iterdir():
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in inodes:
                return True
    except OSError:
        return False
    return False


def _viewer_binding_digest(out: Path) -> str:
    payload = {
        "schema": f"{SCHEMA}_explicit_viewer_v1",
        "out": str(out.resolve()),
        "view_order": list(EVALUATION_VIEWS),
        "splits": list(EVALUATION_SPLITS),
        "models": {
            "initial": display_path(_arm_paths(out, "old_640")["final"]),
            "gaussians": display_path(_arm_paths(out, "new_2000")["final"]),
        },
        "settings": {
            "resolution": list(NATIVE_SIZE),
            "rasterizer": "gsplat",
            "packed": False,
            "antialiased": False,
            "device": "cuda",
        },
    }
    return canonical_hash(payload)


def _viewer_handshake_marker(out: Path, *, token: str, pid: int, port: int) -> str:
    return (
        f"{VIEWER_MARKER_PREFIX}binding={_viewer_binding_digest(out)} "
        f"token={token} pid={pid} port={port}"
    )


def viewer_command(
    out: Path,
    *,
    port: int,
    handshake_token: str | None = None,
) -> list[str]:
    command = [
        "env",
        f"LD_PRELOAD={ABI_PRELOAD}",
        ".venv/bin/python",
        "-m",
        "benchmarks.compact_teacher_capacity_crossover",
        "serve-viewer",
        "--out",
        str(out.resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if handshake_token is not None:
        command.extend(("--handshake-token", handshake_token))
    return command


def run_internal_viewer(
    out: Path,
    *,
    host: str,
    port: int,
    handshake_token: str | None = None,
) -> None:
    """Serve only the exact seven training views plus C1004 after all receipts verify."""
    if host != "127.0.0.1":
        raise ValueError("frozen viewer server must bind 127.0.0.1")
    _assert_port_available(port)
    if handshake_token is not None and (
        len(handshake_token) != 64
        or any(character not in "0123456789abcdef" for character in handshake_token)
    ):
        raise ValueError("viewer handshake token must be 32 random bytes in lowercase hex")
    _assert_evaluation_abi()
    selection, selection_sha = _verified_selection(out)
    _, _ = _verified_evaluation(out, selection_sha)
    acquisition, _ = _verify_evaluation_acquisition(out, selection_sha=selection_sha)
    plan = _strict_json(out / "plan.json")
    inputs = ReconstructionInputs.load(
        bound_path(plan["bundles"]["new_2000"]["path"]),
        device="cpu",
        strict=True,
    )
    from benchmarks import compact_bounds_crossover as bounds

    from rtgs.data.scene import SceneData
    from rtgs.viewer import launch_viewer

    cameras = dict(zip(inputs.view_names, inputs.cameras, strict=True))
    cameras[HELDOUT_VIEW] = bounds.camera_from_calibration(HELDOUT_VIEW)[0]
    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    ordered_cameras: list[Camera] = []
    for index, (view_id, split) in enumerate(zip(EVALUATION_VIEWS, EVALUATION_SPLITS, strict=True)):
        image, mask, record = bounds._load_evaluation_target(view_id, cameras[view_id])
        local = dict(record)
        local["split"] = split
        if local != acquisition["views"][index]:
            raise RuntimeError(f"viewer target differs from evaluation acquisition for {view_id}")
        images.append(image)
        masks.append(mask)
        ordered_cameras.append(cameras[view_id])
    scene = SceneData(
        images=images,
        cameras=ordered_cameras,
        view_names=list(EVALUATION_VIEWS),
        masks=masks,
        train_indices=list(range(len(TRAIN_VIEWS))),
        test_indices=[len(TRAIN_VIEWS)],
        bounds_hint=inputs.bounds_hint,
        name="teacher-artifact-crossover-explicit-eight-view",
    )
    scene.validate()
    old_artifacts = selection["arms"]["old_640"]["artifacts"]
    new_artifacts = selection["arms"]["new_2000"]["artifacts"]
    models = {
        "new_2000 final": Gaussians3D.load_ply(bound_path(new_artifacts["gaussians"])),
        "old_640 final": Gaussians3D.load_ply(bound_path(old_artifacts["gaussians"])),
    }
    if any(model.n != N_INIT_3D for model in models.values()):
        raise RuntimeError("viewer model cardinality differs from frozen fixed budget")
    if handshake_token is not None:
        print(
            _viewer_handshake_marker(
                out,
                token=handshake_token,
                pid=os.getpid(),
                port=port,
            ),
            flush=True,
        )
    launch_viewer(
        models,
        scene=scene,
        device="cuda",
        snapshot_rasterizer="gsplat",
        snapshot_packed=False,
        snapshot_antialiased=False,
        snapshot_dir=out / "viewer_snapshots",
        host=host,
        port=port,
        open_browser=False,
    )


def _smoke_explicit_viewer(
    out: Path,
    *,
    port: int,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Require a fresh child marker, live child PID, and child-owned listening socket."""
    _assert_port_available(port)
    token = secrets.token_hex(32)
    command = viewer_command(out, port=port, handshake_token=token)
    url = f"http://127.0.0.1:{port}"
    process: subprocess.Popen[str] | None = None
    marker = ""
    marker_observed = False
    listener_owner_verified = False
    connected = False
    status = None
    body = b""
    error: str | None = None
    output_text = ""
    log_descriptor, log_text = tempfile.mkstemp(
        prefix=".viewer-smoke.",
        suffix=".log",
        dir=out,
    )
    os.close(log_descriptor)
    log_path = Path(log_text)
    stream: Any | None = None
    try:
        stream = log_path.open("w+", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        marker = _viewer_handshake_marker(
            out,
            token=token,
            pid=process.pid,
            port=port,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            stream.flush()
            output_text = log_path.read_text(encoding="utf-8")
            marker_observed = marker in output_text
            if marker_observed:
                try:
                    with urllib.request.urlopen(url, timeout=2.0) as response:
                        status = int(response.status)
                        body = response.read(1 << 20)
                    listener_owner_verified = _process_owns_listening_port(
                        process.pid,
                        port,
                    )
                    connected = (
                        status == 200
                        and listener_owner_verified
                        and process.poll() is None
                        and (b"viser" in body.lower() or b"realtime-gs" in body.lower())
                    )
                    if connected:
                        break
                except (OSError, urllib.error.URLError) as caught:
                    error = str(caught)
            time.sleep(0.25)
    except BaseException as caught:
        error = f"{type(caught).__name__}: {caught}"
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        if stream is not None:
            stream.flush()
            stream.close()
        if log_path.is_file():
            output_text = log_path.read_text(encoding="utf-8")
            marker_observed = bool(marker) and marker in output_text
        log_path.unlink(missing_ok=True)
    process_pid = -1 if process is None else process.pid
    passed = connected and marker_observed and listener_owner_verified
    return {
        "status": "PASS" if passed else "FAIL",
        "command": command,
        "url": url,
        "connected": connected,
        "http_status": status,
        "http_body_sha256": hashlib.sha256(body).hexdigest() if body else None,
        "last_error": error,
        "process_handshake": {
            "token": token,
            "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "pid": process_pid,
            "port": port,
            "marker": marker,
            "marker_observed": marker_observed,
            "listener_owner_verified": listener_owner_verified,
        },
        "process_output_tail": output_text[-4_000:],
        "returncode_after_termination": None if process is None else process.returncode,
    }


def run_view_smoke(out: Path, *, port: int | None = None) -> dict[str, Any]:
    try:
        _, selection_sha = _verified_selection(out)
        _, evaluation_sha = _verified_evaluation(out, selection_sha)
        if (out / "viewer_receipt.json").exists():
            raise FileExistsError("viewer receipt already exists")
        smoke = _smoke_explicit_viewer(
            out,
            port=_free_port() if port is None else port,
        )
        handshake = smoke["process_handshake"]
        receipt = {
            "artifact_type": f"{SCHEMA}_viewer_receipt_v1",
            **smoke,
            "comparison": {
                "initial": "old_640 final",
                "gaussians": "new_2000 final",
            },
            "native_exact_evaluation_sha256": evaluation_sha,
            "selection_receipt_file_sha256": selection_sha,
            "evaluation_view_order": list(EVALUATION_VIEWS),
            "evaluation_splits": list(EVALUATION_SPLITS),
            "server_binding_digest": _viewer_binding_digest(out),
            "server_binding_marker_sha256": hashlib.sha256(
                handshake["marker"].encode()
            ).hexdigest(),
        }
        if receipt["status"] != "PASS":
            raise RuntimeError("viewer smoke did not pass")
        receipt["viewer_payload_sha256"] = canonical_hash(receipt)
        _write_json_exclusive(out / "viewer_receipt.json", receipt)
        return receipt
    except BaseException as error:
        _failure(out, "view-smoke", error)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--out", type=Path, default=DEFAULT_OUT)
    plan.add_argument("--old-bundle", type=Path, default=OLD_BUNDLE)
    plan.add_argument("--new-bundle", type=Path, default=NEW_BUNDLE)

    for phase in ("select", "evaluate"):
        command = subparsers.add_parser(phase)
        command.add_argument("--out", type=Path, default=DEFAULT_OUT)

    viewer = subparsers.add_parser("view-smoke")
    viewer.add_argument("--out", type=Path, default=DEFAULT_OUT)
    viewer.add_argument("--port", type=int, default=None)

    serve = subparsers.add_parser("serve-viewer", help=argparse.SUPPRESS)
    serve.add_argument("--out", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, required=True)
    serve.add_argument("--handshake-token", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out.resolve()
    if args.phase == "plan":
        result = prepare_plan(
            out,
            old_bundle=args.old_bundle.resolve(),
            new_bundle=args.new_bundle.resolve(),
        )
    elif args.phase == "select":
        result = run_selection(out)
    elif args.phase == "evaluate":
        result = run_evaluation(out)
    elif args.phase == "serve-viewer":
        run_internal_viewer(
            out,
            host=args.host,
            port=args.port,
            handshake_token=args.handshake_token,
        )
        return 0
    else:
        result = run_view_smoke(out, port=args.port)
    print(
        json.dumps(
            {
                "status": result.get("status", "PASS"),
                "phase": args.phase,
                "out": str(out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
