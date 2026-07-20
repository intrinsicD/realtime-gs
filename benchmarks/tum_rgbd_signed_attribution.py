#!/usr/bin/env python3
"""Sealed signed occlusion/rigidity attribution on two dynamic TUM RGB-D sequences.

The development phase audits ``fr3/sitting_xyz``.  Only a passing development artifact
authorizes the atomic, one-shot ``fr3/walking_xyz`` confirmation.  Visibility is constructed
from T-role depth only; V-role depth labels signed residuals after visibility is frozen.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import zlib
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from benchmarks import tum_rgbd_oriented_validity as base

from rtgs.core.camera import Camera

PREREGISTRATION = Path("benchmarks/results/20260715_tum_rgbd_signed_attribution_PREREG.md")
ACQUISITION_MANIFEST = Path(
    "benchmarks/results/20260715_tum_rgbd_signed_attribution_ACQUISITION.json"
)
CONFIRMATORY_SEAL = Path(
    "benchmarks/results/20260715_tum_rgbd_signed_attribution_CONFIRMATORY_SEAL.json"
)
PREDECODE_SEAL = Path("benchmarks/results/20260715_tum_rgbd_signed_attribution_PREDECODE_SEAL.json")
EXPERIMENT = "tum-rgbd-signed-occlusion-rigidity-attribution"
Phase = Literal["development", "confirmatory"]

SOURCES: dict[Phase, dict[str, Any]] = {
    "development": {
        "sequence": "rgbd_dataset_freiburg3_sitting_xyz",
        "url": (
            "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_xyz.tgz"
        ),
        "content_length": 775_406_859,
        "etag": '"2e37c50b-4c6b1a7e29cc0"',
        "last_modified": "Tue, 07 Aug 2012 19:03:55 GMT",
        "sha256": "05c071672cda22a668860a935124737a4eb4fa772cbad372e73d5a99ce4be205",
    },
    "confirmatory": {
        "sequence": "rgbd_dataset_freiburg3_walking_xyz",
        "url": (
            "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz"
        ),
        "content_length": 527_550_055,
        "etag": '"1f71c667-4c6b17936fb00"',
        "last_modified": "Tue, 07 Aug 2012 18:50:52 GMT",
        "sha256": "1459e9488ac0e61a2ec80dfbc35cfb77942f6d8eabded1c8d26a70be650d0e1d",
    },
}

CONFIG: dict[str, Any] = {
    "base_target_config_id": base.CONFIG_ID,
    "base_target_grid": {"rows": base.TARGET_ROWS, "columns": base.TARGET_COLUMNS},
    "dense_construction": {
        "stride": 8,
        "origin": 0,
        "explicit_sparse_union": True,
        "min_depth_m": 0.3,
        "max_depth_m": 5.0,
    },
    "visibility_depth_m": 0.020,
    "minimum_validation_views": 2,
    "signed_class": {"absolute_m": 0.050, "relative": 0.03},
    "temporal": {
        "near_max_normalized": 0.20,
        "far_min_normalized": 0.60,
        "translation_upper_edges_m": (0.10, 0.25, 0.50, 1.0),
        "rotation_upper_edges_deg": (5.0, 15.0, 30.0),
        "minimum_pairs_per_cell_stratum": 250,
        "minimum_targets_per_cell": 50,
        "minimum_cells": 4,
        "minimum_pairs_per_global_stratum": 1_000,
        "maximum_cell_weight": 0.25,
    },
    "occlusion_support": {
        "minimum_sparse_pairs": 20_000,
        "minimum_dense_retention": 0.70,
        "minimum_removed_fraction": 0.01,
        "minimum_removed_pairs": 1_000,
        "minimum_paired_targets": 100,
        "minimum_dense_supported_targets": 500,
    },
    "occlusion_effect": {
        "minimum_e_plus": 0.10,
        "minimum_positive_risk_ratio": 2.0,
        "minimum_positive_selectivity": 0.05,
        "minimum_positive_reduction_absolute": 0.01,
        "minimum_positive_reduction_relative": 0.15,
        "maximum_negative_increase": 0.01,
        "maximum_d90_ratio": 1.05,
    },
    "motion_effect": {
        "minimum_contradiction_difference": 0.05,
        "minimum_contradiction_ratio": 1.25,
        "minimum_d90_ratio": 1.25,
        "minimum_negative_difference": 0.02,
        "minimum_walking_temporal_delta": 0.03,
        "minimum_temporal_excess": 0.02,
    },
    "bootstrap": {"replicates": 1_000, "seed": 20_260_715},
    "reduction": "cpu-float64-target-balanced-linear-quantile",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


CONFIG_ID = _sha256_bytes(_canonical_json(CONFIG))


def _finite(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _interval(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    if tensor.numel() != CONFIG["bootstrap"]["replicates"]:
        raise AssertionError("bootstrap replicate count changed")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("bootstrap distribution is nonfinite")
    return {
        "p05": base._linear_quantile(tensor, 0.05),
        "p50": base._linear_quantile(tensor, 0.50),
        "p95": base._linear_quantile(tensor, 0.95),
    }


def _mean_finite(value: torch.Tensor, *, label: str) -> float:
    finite = value[torch.isfinite(value)]
    if finite.numel() == 0:
        raise ValueError(f"{label} population is empty")
    return _finite(float(finite.mean()), label=label)


def _git_metadata(root: Path) -> dict[str, Any]:
    def command(*args: str) -> str:
        return subprocess.check_output(args, cwd=root, text=True).strip()

    status = command("git", "status", "--short")
    return {
        "commit": command("git", "rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_sha256": _sha256_bytes(status.encode()),
    }


def _implementation_metadata(root: Path) -> dict[str, Any]:
    paths = [
        root / Path(__file__).resolve().relative_to(root),
        root / "benchmarks/tum_rgbd_oriented_validity.py",
        root / "src/rtgs/lift/surface.py",
        root / "src/rtgs/core/camera.py",
        root / "tests/test_tum_rgbd_signed_attribution.py",
        root / PREREGISTRATION,
        root / ACQUISITION_MANIFEST,
        root / "pyproject.toml",
    ]
    hashes = {str(path.relative_to(root)): base._sha256_file(path) for path in paths}
    return {"files": hashes, "aggregate_sha256": _sha256_bytes(_canonical_json(hashes))}


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pillow": base.PIL.__version__,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
    }


def _validate_predecode_seal(
    root: Path, implementation: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    payload = (root / PREDECODE_SEAL).read_bytes()
    seal = base._json_object_from_bytes(payload, label="signed-attribution predecode seal")
    if seal.get("schema_version") != 1 or seal.get("experiment") != EXPERIMENT:
        raise ValueError("predecode seal identity mismatch")
    if seal.get("config_id") != CONFIG_ID:
        raise ValueError("predecode seal configuration mismatch")
    if seal.get("implementation") != implementation:
        raise ValueError("runtime implementation differs from the predecode seal")
    if seal.get("source_sha256") != {phase: source["sha256"] for phase, source in SOURCES.items()}:
        raise ValueError("predecode seal source hashes differ")
    if seal.get("synthetic_test_status") != "21 passed":
        raise ValueError("predecode seal lacks the frozen focused-test result")
    return seal, _sha256_bytes(payload)


def _validate_source_archive_file(archive_path: Path, phase: Phase) -> tuple[str, os.stat_result]:
    source = SOURCES[phase]
    stat = archive_path.stat()
    if stat.st_size != source["content_length"]:
        raise ValueError("archive byte length differs from preregistered HTTP length")
    archive_sha256 = base._sha256_file(archive_path)
    if archive_sha256 != source["sha256"]:
        raise ValueError("archive SHA-256 differs from the first frozen download")
    return archive_sha256, stat


def _validate_acquisition_record(
    archive_path: Path, archive_stat: os.stat_result, phase: Phase
) -> tuple[str, dict[str, Any]]:
    root = Path(__file__).resolve().parents[1]
    payload = (root / ACQUISITION_MANIFEST).read_bytes()
    manifest = base._json_object_from_bytes(payload, label="signed-attribution acquisition")
    if manifest.get("schema_version") != 1 or set(manifest.get("sources", {})) != set(SOURCES):
        raise ValueError("acquisition manifest does not contain the exact frozen sources")
    record = manifest["sources"][phase]
    expected = SOURCES[phase]
    for key in ("sequence", "url", "content_length", "etag", "last_modified", "sha256"):
        if record.get(key) != expected[key]:
            raise ValueError(f"acquisition source {key} differs from the frozen source")
    if Path(record.get("archive_path", "")).resolve() != archive_path.resolve():
        raise ValueError("acquisition source path differs from the audited archive")
    if record.get("filesystem_mtime_utc_completion_proxy") != base._utc_from_posix_ns(
        archive_stat.st_mtime_ns
    ):
        raise ValueError("archive mtime differs from the frozen acquisition proxy")
    if "not server-observed retrieval timestamps" not in manifest.get("evidence_limit", ""):
        raise ValueError("acquisition timestamp limitation is missing")
    return _sha256_bytes(payload), record


def _load_source_metadata(
    archive_path: Path, phase: Phase
) -> tuple[
    base.TumTar,
    list[base.AssociatedFrame],
    list[base.SelectedFrame],
    dict[str, str],
    dict[str, Any],
    dict[str, frozenset[str]],
]:
    source = SOURCES[phase]
    archive_sha256, archive_stat = _validate_source_archive_file(archive_path, phase)
    acquisition_sha256, acquisition_record = _validate_acquisition_record(
        archive_path, archive_stat, phase
    )
    archive = base.TumTar(archive_path)
    archive.__enter__()
    try:
        if archive.prefix != source["sequence"]:
            raise ValueError("archive root differs from the frozen sequence")
        rgb_text, rgb_hash = archive.read_text("rgb.txt")
        depth_text, depth_hash = archive.read_text("depth.txt")
        pose_text, pose_hash = archive.read_text("groundtruth.txt")
        rgb = base._parse_timed_paths(rgb_text)
        depth = base._parse_timed_paths(depth_text)
        poses = base._parse_timed_poses(pose_text)
        rgb_members = frozenset(archive.full_member_name(item.path) for item in rgb)
        depth_members = frozenset(archive.full_member_name(item.path) for item in depth)
        if len(rgb_members) != len(rgb) or len(depth_members) != len(depth):
            raise AssertionError("manifest payload identities are not unique")
        if rgb_members & depth_members:
            raise ValueError("RGB and depth manifests alias archive members")
        frames, association_diagnostics = base._associate_frames(rgb, depth, poses)
        selected, selection_diagnostics = base._select_frames(frames)
        for item in selected:
            archive.full_member_name(item.frame.rgb.path)
            archive.full_member_name(item.frame.depth.path)
        selected_manifest = base._selected_manifest(selected)
        association_manifest = base._association_manifest(frames)
        hashes = {
            "rgb.txt": rgb_hash,
            "depth.txt": depth_hash,
            "groundtruth.txt": pose_hash,
            "association": _sha256_bytes(_canonical_json(association_manifest)),
            "split": _sha256_bytes(_canonical_json(selected_manifest)),
        }
        diagnostics = {
            "archive_sha256": archive_sha256,
            "local_file_mtime_utc": base._utc_from_posix_ns(archive_stat.st_mtime_ns),
            "local_file_mtime_semantics": "download_completion_proxy_not_network_retrieval_time",
            "acquisition_manifest": {
                "path": str(ACQUISITION_MANIFEST),
                "sha256": acquisition_sha256,
                "record": acquisition_record,
            },
            "archive_member_count": len(archive.member_names),
            "manifest_payload_members": {
                "rgb_count": len(rgb_members),
                "depth_count": len(depth_members),
                "rgb_set_sha256": _sha256_bytes(_canonical_json(sorted(rgb_members))),
                "depth_set_sha256": _sha256_bytes(_canonical_json(sorted(depth_members))),
                "rgb_depth_disjoint": True,
            },
            "association": association_diagnostics,
            "selection": selection_diagnostics,
        }
        return (
            archive,
            frames,
            selected,
            hashes,
            diagnostics,
            {"rgb": rgb_members, "depth": depth_members},
        )
    except BaseException:
        archive.__exit__(*sys.exc_info())
        raise


def _dense_construction_points(
    t_frames: list[base.SelectedFrame],
    cameras: dict[str, Camera],
    depths: dict[str, torch.Tensor],
    targets: base.TargetBundle,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Build the frozen stride-8 T-only set, explicitly unioned with sparse targets."""

    stride = CONFIG["dense_construction"]["stride"]
    rows, columns = torch.meshgrid(
        torch.arange(0, base.HEIGHT, stride, dtype=torch.long),
        torch.arange(0, base.WIDTH, stride, dtype=torch.long),
        indexing="ij",
    )
    flat_rows = rows.reshape(-1)
    flat_columns = columns.reshape(-1)
    pixel_centers = torch.stack(
        [flat_columns.to(torch.float64) + 0.5, flat_rows.to(torch.float64) + 0.5], dim=-1
    )
    points = [targets.points_world.detach().cpu().to(torch.float64)]
    counts: list[int] = []
    for selected in t_frames:
        view_id = selected.frame.view_id
        depth = depths[view_id].detach().cpu().to(torch.float64)[flat_rows, flat_columns]
        valid = (
            torch.isfinite(depth)
            & (depth >= CONFIG["dense_construction"]["min_depth_m"])
            & (depth <= CONFIG["dense_construction"]["max_depth_m"])
        )
        count = int(valid.sum())
        counts.append(count)
        points.append(cameras[view_id].unproject(pixel_centers[valid], depth[valid]))
    dense = torch.cat(points).detach().cpu().to(torch.float64)
    if not bool(torch.isfinite(dense).all()) or dense.shape[1:] != (3,):
        raise ValueError("dense T-only construction points must be finite 3D vectors")
    if dense.shape[0] != targets.points_world.shape[0] + sum(counts):
        raise AssertionError("dense construction count lost its explicit sparse union")
    if not torch.equal(dense[: targets.points_world.shape[0]], targets.points_world):
        raise AssertionError("dense construction does not begin with exact sparse target tensors")
    return dense, tuple(counts)


def _construction_visibility(
    target_points: torch.Tensor,
    construction_points: torch.Tensor,
    camera: Camera,
    *,
    visibility_depth_m: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return target in-frame/visible masks without accepting validation observations."""

    epsilon = (
        CONFIG["visibility_depth_m"] if visibility_depth_m is None else float(visibility_depth_m)
    )
    if epsilon < 0 or not math.isfinite(epsilon):
        raise ValueError("visibility tolerance must be finite and nonnegative")
    target_points = target_points.detach().cpu().to(torch.float64)
    construction_points = construction_points.detach().cpu().to(torch.float64)
    target_pixels, target_depth = camera.project(target_points)
    target_in_frame = (
        torch.isfinite(target_pixels).all(dim=-1)
        & torch.isfinite(target_depth)
        & (target_depth > 0)
        & (target_pixels[:, 0] >= 0)
        & (target_pixels[:, 0] < base.WIDTH)
        & (target_pixels[:, 1] >= 0)
        & (target_pixels[:, 1] < base.HEIGHT)
    )
    target_linear = torch.full((target_points.shape[0],), -1, dtype=torch.long)
    target_indices = torch.nonzero(target_in_frame, as_tuple=False).squeeze(-1)
    target_columns = torch.floor(target_pixels[target_indices, 0]).long()
    target_rows = torch.floor(target_pixels[target_indices, 1]).long()
    target_linear[target_indices] = target_rows * base.WIDTH + target_columns

    pixels, depth = camera.project(construction_points)
    in_frame = (
        torch.isfinite(pixels).all(dim=-1)
        & torch.isfinite(depth)
        & (depth > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < base.WIDTH)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < base.HEIGHT)
    )
    indices = torch.nonzero(in_frame, as_tuple=False).squeeze(-1)
    columns = torch.floor(pixels[indices, 0]).long()
    rows = torch.floor(pixels[indices, 1]).long()
    linear = rows * base.WIDTH + columns
    z_buffer = torch.full((base.HEIGHT * base.WIDTH,), float("inf"), dtype=torch.float64)
    z_buffer.scatter_reduce_(0, linear, depth[indices], reduce="amin", include_self=True)
    visible = torch.zeros_like(target_in_frame)
    visible[target_indices] = (
        target_depth[target_indices] <= z_buffer[target_linear[target_indices]] + epsilon
    )
    if bool(visible.any()) and not bool(target_in_frame[visible].all()):
        raise AssertionError("construction-visible target is not in-frame")
    return target_in_frame, visible, target_depth, target_pixels, target_linear


def _classify_signed(
    predicted_depth: torch.Tensor, observed_depth: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted_depth = predicted_depth.detach().cpu().to(torch.float64)
    observed_depth = observed_depth.detach().cpu().to(torch.float64)
    if predicted_depth.shape != observed_depth.shape:
        raise ValueError("predicted and observed depth shapes differ")
    valid = (
        torch.isfinite(predicted_depth)
        & (predicted_depth > 0)
        & torch.isfinite(observed_depth)
        & (observed_depth >= CONFIG["dense_construction"]["min_depth_m"])
        & (observed_depth <= CONFIG["dense_construction"]["max_depth_m"])
    )
    signed = predicted_depth - observed_depth
    relative = signed / observed_depth.clamp_min(CONFIG["dense_construction"]["min_depth_m"])
    tolerance = torch.maximum(
        torch.full_like(observed_depth, CONFIG["signed_class"]["absolute_m"]),
        CONFIG["signed_class"]["relative"] * observed_depth,
    )
    positive = valid & (signed > tolerance)
    negative = valid & (signed < -tolerance)
    if bool((positive & negative).any()):
        raise AssertionError("signed residual classes overlap")
    return signed, relative, tolerance, positive, negative


def _pose_pair_matrices(
    targets: base.TargetBundle,
    t_frames: list[base.SelectedFrame],
    v_frames: list[base.SelectedFrame],
    selected: list[base.SelectedFrame],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_count = targets.points_world.shape[0]
    gap = torch.empty((target_count, len(v_frames)), dtype=torch.float64)
    translation = torch.empty_like(gap)
    rotation = torch.empty_like(gap)
    source_span = max(item.frame.depth.timestamp_ns for item in selected) - min(
        item.frame.depth.timestamp_ns for item in selected
    )
    if source_span <= 0:
        raise ValueError("selected timestamp span must be positive")
    by_t_ordinal = {item.role_ordinal: item for item in t_frames}
    for v_index, validation in enumerate(v_frames):
        for t_ordinal in range(len(t_frames)):
            mask = targets.source_t_ordinals == t_ordinal
            source = by_t_ordinal[t_ordinal]
            gap[mask, v_index] = (
                abs(source.frame.depth.timestamp_ns - validation.frame.depth.timestamp_ns)
                / source_span
            )
            translation[mask, v_index] = float(
                (source.frame.pose.center - validation.frame.pose.center).norm()
            )
            rotation[mask, v_index] = base._rotation_distance_deg(
                source.frame.pose.quaternion_xyzw,
                validation.frame.pose.quaternion_xyzw,
            )
    return gap, translation, rotation


def _audit_signed(
    targets: base.TargetBundle,
    dense_points: torch.Tensor,
    v_frames: list[base.SelectedFrame],
    cameras: dict[str, Camera],
    depths: dict[str, torch.Tensor],
    t_frames: list[base.SelectedFrame],
    selected: list[base.SelectedFrame],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    count = targets.points_world.shape[0]
    n_validation = len(v_frames)
    shape = (count, n_validation)
    predicted = torch.full(shape, float("nan"), dtype=torch.float64)
    observed = torch.full_like(predicted, float("nan"))
    signed = torch.full_like(predicted, float("nan"))
    relative = torch.full_like(predicted, float("nan"))
    tolerance = torch.full_like(predicted, float("nan"))
    in_frame = torch.zeros(shape, dtype=torch.bool)
    sparse_visible = torch.zeros_like(in_frame)
    dense_visible = torch.zeros_like(in_frame)
    observed_valid = torch.zeros_like(in_frame)
    positive = torch.zeros_like(in_frame)
    negative = torch.zeros_like(in_frame)
    per_v: list[dict[str, Any]] = []
    target_points = targets.points_world.detach().cpu().to(torch.float64)

    for v_index, validation in enumerate(v_frames):
        view_id = validation.frame.view_id
        camera = cameras[view_id]
        sparse_in, sparse_mask, z_pred, pixels, linear = _construction_visibility(
            target_points, target_points, camera
        )
        dense_in, dense_mask, dense_pred, dense_pixels, dense_linear = _construction_visibility(
            target_points, dense_points, camera
        )
        if not torch.equal(sparse_in, dense_in):
            raise AssertionError("visibility arms disagree on target projection bounds")
        if not torch.equal(z_pred, dense_pred) or not torch.equal(pixels, dense_pixels):
            raise AssertionError("visibility arms changed target projections")
        if not torch.equal(linear, dense_linear):
            raise AssertionError("visibility arms changed target raster indices")
        if bool((dense_mask & ~sparse_mask).any()):
            raise AssertionError("dense visibility is not a subset of sparse visibility")
        target_indices = torch.nonzero(sparse_in, as_tuple=False).squeeze(-1)
        rows = torch.div(linear[target_indices], base.WIDTH, rounding_mode="floor")
        columns = linear[target_indices] % base.WIDTH
        z_obs = depths[view_id].detach().cpu().to(torch.float64)[rows, columns]
        full_observed = torch.full((count,), float("nan"), dtype=torch.float64)
        full_observed[target_indices] = z_obs
        d, r, tau, pos, neg = _classify_signed(z_pred, full_observed)
        valid = (
            sparse_in
            & torch.isfinite(full_observed)
            & (full_observed >= CONFIG["dense_construction"]["min_depth_m"])
            & (full_observed <= CONFIG["dense_construction"]["max_depth_m"])
        )
        predicted[:, v_index] = z_pred
        observed[:, v_index] = full_observed
        signed[:, v_index] = d
        relative[:, v_index] = r
        tolerance[:, v_index] = tau
        in_frame[:, v_index] = sparse_in
        sparse_visible[:, v_index] = sparse_mask
        dense_visible[:, v_index] = dense_mask
        observed_valid[:, v_index] = valid
        positive[:, v_index] = pos
        negative[:, v_index] = neg
        sparse_valid = sparse_mask & valid
        dense_valid = dense_mask & valid
        per_v.append(
            {
                "selected_ordinal": validation.selected_ordinal,
                "view_id": view_id,
                "targets": count,
                "out_of_frame": int((~sparse_in).sum()),
                "in_frame": int(sparse_in.sum()),
                "sparse_visible": int(sparse_mask.sum()),
                "dense_visible": int(dense_mask.sum()),
                "sparse_depth_valid": int(sparse_valid.sum()),
                "dense_depth_valid": int(dense_valid.sum()),
                "sparse_positive": int((sparse_valid & pos).sum()),
                "sparse_negative": int((sparse_valid & neg).sum()),
                "dense_positive": int((dense_valid & pos).sum()),
                "dense_negative": int((dense_valid & neg).sum()),
            }
        )
    gap, translation, rotation = _pose_pair_matrices(targets, t_frames, v_frames, selected)
    raw = {
        "predicted_depth_m": predicted,
        "observed_depth_m": observed,
        "signed_depth_m": signed,
        "signed_relative": relative,
        "contradiction_tolerance_m": tolerance,
        "in_frame": in_frame,
        "sparse_visible": sparse_visible,
        "dense_visible": dense_visible,
        "observed_depth_valid": observed_valid,
        "positive": positive,
        "negative": negative,
        "sparse_valid": sparse_visible & observed_valid,
        "dense_valid": dense_visible & observed_valid,
        "normalized_time_gap": gap,
        "translation_baseline_m": translation,
        "rotation_baseline_deg": rotation,
    }
    if bool((raw["dense_valid"] & ~raw["sparse_valid"]).any()):
        raise AssertionError("dense valid population is not nested under sparse")
    if not torch.equal(positive & negative, torch.zeros_like(positive)):
        raise AssertionError("positive and negative raw masks overlap")
    return raw, per_v


def _target_vectors(
    valid: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    signed_relative: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if valid.ndim != 2 or not (
        valid.shape == positive.shape == negative.shape == signed_relative.shape
    ):
        raise ValueError("target reduction matrices have incompatible shapes")
    counts = valid.sum(dim=1)
    supported = counts >= CONFIG["minimum_validation_views"]
    denominator = counts.clamp_min(1).to(torch.float64)
    positive_rate = ((positive & valid).sum(dim=1) / denominator).to(torch.float64)
    negative_rate = ((negative & valid).sum(dim=1) / denominator).to(torch.float64)
    positive_rate[~supported] = float("nan")
    negative_rate[~supported] = float("nan")
    contradiction_rate = positive_rate + negative_rate
    values = torch.where(
        valid, signed_relative.abs(), torch.full_like(signed_relative, float("nan"))
    )
    absolute_relative_median = torch.nanquantile(values, 0.5, dim=1, interpolation="linear")
    absolute_relative_median[~supported] = float("nan")
    return {
        "counts": counts,
        "supported": supported,
        "positive_rate": positive_rate,
        "negative_rate": negative_rate,
        "contradiction_rate": contradiction_rate,
        "absolute_relative_median": absolute_relative_median,
    }


def _arm_summary(
    name: str,
    valid: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    signed_relative: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    vectors = _target_vectors(valid, positive, negative, signed_relative)
    supported = vectors["supported"]
    if not bool(supported.any()):
        raise ValueError(f"{name} has no targets with two validation observations")
    pair_count = int(valid.sum())
    if pair_count == 0:
        raise ValueError(f"{name} has no depth-valid pairs")
    target_absrel = vectors["absolute_relative_median"][supported]
    summary = {
        "name": name,
        "depth_valid_pairs": pair_count,
        "supported_targets": int(supported.sum()),
        "support_fraction": float(supported.to(torch.float64).mean()),
        "P_plus": _mean_finite(vectors["positive_rate"], label=f"{name} P_plus"),
        "P_minus": _mean_finite(vectors["negative_rate"], label=f"{name} P_minus"),
        "C": _mean_finite(vectors["contradiction_rate"], label=f"{name} C"),
        "D50": base._linear_quantile(target_absrel, 0.50),
        "D90": base._linear_quantile(target_absrel, 0.90),
        "pair_weighted": {
            "P_plus": float((positive & valid).sum()) / pair_count,
            "P_minus": float((negative & valid).sum()) / pair_count,
            "C": float(((positive | negative) & valid).sum()) / pair_count,
        },
    }
    return summary, vectors


def _rates_with_minimum(
    mask: torch.Tensor, event: torch.Tensor, minimum: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    count = mask.sum(dim=1)
    keep = count >= minimum
    rate = (event & mask).sum(dim=1).to(torch.float64) / count.clamp_min(1)
    rate[~keep] = float("nan")
    return rate, keep


def _attribution_summary(
    raw: dict[str, torch.Tensor],
    arms: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    sparse_valid = raw["sparse_valid"]
    dense_valid = raw["dense_valid"]
    removed = sparse_valid & ~raw["dense_visible"]
    retained = dense_valid
    removed_positive, removed_has = _rates_with_minimum(removed, raw["positive"])
    retained_positive, retained_has = _rates_with_minimum(retained, raw["positive"])
    removed_negative, _ = _rates_with_minimum(removed, raw["negative"])
    retained_negative, _ = _rates_with_minimum(retained, raw["negative"])
    paired = removed_has & retained_has
    if not bool(paired.any()):
        e_plus = e_minus = positive_risk_ratio = None
    else:
        e_plus = float((removed_positive[paired] - retained_positive[paired]).mean())
        e_minus = float((removed_negative[paired] - retained_negative[paired]).mean())
        p_removed = float(removed_positive[paired].mean())
        p_retained = float(retained_positive[paired].mean())
        positive_risk_ratio = (p_removed + 0.001) / (p_retained + 0.001)
    sparse_count = int(sparse_valid.sum())
    dense_count = int(dense_valid.sum())
    removed_count = int(removed.sum())
    summary = {
        "sparse_depth_valid_pairs": sparse_count,
        "dense_depth_valid_pairs": dense_count,
        "removed_depth_valid_pairs": removed_count,
        "dense_retention": dense_count / sparse_count if sparse_count else 0.0,
        "removed_fraction": removed_count / sparse_count if sparse_count else 0.0,
        "paired_removed_retained_targets": int(paired.sum()),
        "E_plus": e_plus,
        "E_minus": e_minus,
        "positive_selectivity": (
            e_plus - max(e_minus, 0.0) if e_plus is not None and e_minus is not None else None
        ),
        "positive_risk_ratio": positive_risk_ratio,
        "P_plus_reduction": arms["sparse"]["P_plus"] - arms["dense_T"]["P_plus"],
        "P_minus_change_dense_minus_sparse": (
            arms["dense_T"]["P_minus"] - arms["sparse"]["P_minus"]
        ),
        "D90_ratio_dense_over_sparse": arms["dense_T"]["D90"] / arms["sparse"]["D90"],
        "removed_pair_rates": {
            "P_plus": (
                float((removed & raw["positive"]).sum()) / removed_count if removed_count else None
            ),
            "P_minus": (
                float((removed & raw["negative"]).sum()) / removed_count if removed_count else None
            ),
        },
        "positive_removed_recall": (
            float((removed & raw["positive"]).sum())
            / max(1, int((sparse_valid & raw["positive"]).sum()))
        ),
        "negative_removed_recall": (
            float((removed & raw["negative"]).sum())
            / max(1, int((sparse_valid & raw["negative"]).sum()))
        ),
    }
    vectors = {
        "paired": paired,
        "removed_positive_rate": removed_positive,
        "retained_positive_rate": retained_positive,
        "removed_negative_rate": removed_negative,
        "retained_negative_rate": retained_negative,
    }
    return summary, vectors


def _bootstrap_occlusion(
    attribution: dict[str, Any],
    attribution_vectors: dict[str, torch.Tensor],
    sparse_vectors: dict[str, torch.Tensor],
    dense_vectors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    if attribution["E_plus"] is None or not math.isfinite(attribution["E_plus"]):
        return {"estimable": False, "reason": "no targets have removed and retained pairs"}
    paired = attribution_vectors["paired"]
    paired_values = {
        name: value[paired].numpy()
        for name, value in attribution_vectors.items()
        if name != "paired"
    }
    common = sparse_vectors["supported"] & dense_vectors["supported"]
    common_values = {
        "sparse_positive": sparse_vectors["positive_rate"][common].numpy(),
        "dense_positive": dense_vectors["positive_rate"][common].numpy(),
        "sparse_negative": sparse_vectors["negative_rate"][common].numpy(),
        "dense_negative": dense_vectors["negative_rate"][common].numpy(),
    }
    if not paired_values["removed_positive_rate"].size or not common_values["sparse_positive"].size:
        return {"estimable": False, "reason": "bootstrap target populations are empty"}
    rng = np.random.default_rng(CONFIG["bootstrap"]["seed"])
    distributions: dict[str, list[float]] = {
        "E_plus": [],
        "log_positive_risk_ratio": [],
        "positive_selectivity": [],
        "P_plus_reduction_common_targets": [],
        "P_minus_sparse_minus_dense_common_targets": [],
    }
    for _ in range(CONFIG["bootstrap"]["replicates"]):
        p_index = rng.integers(
            0,
            paired_values["removed_positive_rate"].size,
            size=paired_values["removed_positive_rate"].size,
        )
        c_index = rng.integers(
            0, common_values["sparse_positive"].size, size=common_values["sparse_positive"].size
        )
        e_plus = float(
            np.mean(
                paired_values["removed_positive_rate"][p_index]
                - paired_values["retained_positive_rate"][p_index]
            )
        )
        e_minus = float(
            np.mean(
                paired_values["removed_negative_rate"][p_index]
                - paired_values["retained_negative_rate"][p_index]
            )
        )
        removed_rate = float(np.mean(paired_values["removed_positive_rate"][p_index]))
        retained_rate = float(np.mean(paired_values["retained_positive_rate"][p_index]))
        distributions["E_plus"].append(e_plus)
        distributions["log_positive_risk_ratio"].append(
            math.log((removed_rate + 0.001) / (retained_rate + 0.001))
        )
        distributions["positive_selectivity"].append(e_plus - max(e_minus, 0.0))
        distributions["P_plus_reduction_common_targets"].append(
            float(
                np.mean(
                    common_values["sparse_positive"][c_index]
                    - common_values["dense_positive"][c_index]
                )
            )
        )
        distributions["P_minus_sparse_minus_dense_common_targets"].append(
            float(
                np.mean(
                    common_values["sparse_negative"][c_index]
                    - common_values["dense_negative"][c_index]
                )
            )
        )
    intervals = {name: _interval(values) for name, values in distributions.items()}
    positive_names = (
        "E_plus",
        "log_positive_risk_ratio",
        "positive_selectivity",
        "P_plus_reduction_common_targets",
    )
    return {
        "estimable": True,
        "replicates": CONFIG["bootstrap"]["replicates"],
        "seed": CONFIG["bootstrap"]["seed"],
        "paired_target_count": int(paired.sum()),
        "common_supported_target_count": int(common.sum()),
        "intervals": intervals,
        "asserted_positive_lower_bounds_above_zero": {
            name: intervals[name]["p05"] > 0 for name in positive_names
        },
        "all_asserted_positive_signs": all(intervals[name]["p05"] > 0 for name in positive_names),
    }


def _occlusion_gate(
    attribution: dict[str, Any],
    arms: dict[str, dict[str, Any]],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    support = CONFIG["occlusion_support"]
    effect = CONFIG["occlusion_effect"]
    minimum_removed = max(
        support["minimum_removed_pairs"],
        math.ceil(support["minimum_removed_fraction"] * attribution["sparse_depth_valid_pairs"]),
    )

    def at_least(value: Any, threshold: float) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(value) and value >= threshold

    comparisons = {
        "minimum_sparse_pairs": (
            attribution["sparse_depth_valid_pairs"] >= support["minimum_sparse_pairs"]
        ),
        "minimum_dense_retention": (
            attribution["dense_retention"] >= support["minimum_dense_retention"]
        ),
        "minimum_removed_pairs": attribution["removed_depth_valid_pairs"] >= minimum_removed,
        "minimum_paired_targets": (
            attribution["paired_removed_retained_targets"] >= support["minimum_paired_targets"]
        ),
        "minimum_dense_supported_targets": (
            arms["dense_T"]["supported_targets"] >= support["minimum_dense_supported_targets"]
        ),
        "E_plus": at_least(attribution["E_plus"], effect["minimum_e_plus"]),
        "positive_risk_ratio": at_least(
            attribution["positive_risk_ratio"], effect["minimum_positive_risk_ratio"]
        ),
        "positive_selectivity": at_least(
            attribution["positive_selectivity"], effect["minimum_positive_selectivity"]
        ),
        "P_plus_reduction": (
            attribution["P_plus_reduction"]
            >= max(
                effect["minimum_positive_reduction_absolute"],
                effect["minimum_positive_reduction_relative"] * arms["sparse"]["P_plus"],
            )
        ),
        "P_minus_nonincrease": (
            attribution["P_minus_change_dense_minus_sparse"] <= effect["maximum_negative_increase"]
        ),
        "D90_safeguard": (
            attribution["D90_ratio_dense_over_sparse"] <= effect["maximum_d90_ratio"]
        ),
        "bootstrap_signs": bool(
            bootstrap.get("estimable") and bootstrap.get("all_asserted_positive_signs")
        ),
    }
    comparisons["all"] = all(comparisons.values())
    return {"minimum_removed_pairs_effective": minimum_removed, "comparisons": comparisons}


def _capped_weights(raw_weights: list[float], cap: float) -> list[float]:
    if not raw_weights or any(value <= 0 or not math.isfinite(value) for value in raw_weights):
        raise ValueError("pose-cell weights must be positive and finite")
    if cap * len(raw_weights) < 1.0 - 1e-12:
        raise ValueError("too few pose cells for the requested weight cap")
    remaining = set(range(len(raw_weights)))
    weights = [0.0] * len(raw_weights)
    remaining_mass = 1.0
    while remaining:
        scale = remaining_mass / sum(raw_weights[index] for index in remaining)
        newly_capped = [index for index in remaining if raw_weights[index] * scale > cap]
        if not newly_capped:
            for index in remaining:
                weights[index] = raw_weights[index] * scale
            break
        for index in newly_capped:
            weights[index] = cap
            remaining.remove(index)
            remaining_mass -= cap
    if not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-12):
        raise AssertionError("pose-cell weights do not sum to one")
    if max(weights) > cap + 1e-12:
        raise AssertionError("pose-cell weight cap was violated")
    return weights


def _bin_index(values: torch.Tensor, upper_edges: tuple[float, ...]) -> torch.Tensor:
    return torch.bucketize(
        values.detach().cpu().to(torch.float64),
        torch.tensor(upper_edges, dtype=torch.float64),
        right=False,
    )


def _pose_overall_cells(
    valid: torch.Tensor,
    contradiction: torch.Tensor,
    translation: torch.Tensor,
    rotation: torch.Tensor,
) -> list[dict[str, Any]]:
    temporal = CONFIG["temporal"]
    translation_bin = _bin_index(translation, temporal["translation_upper_edges_m"])
    rotation_bin = _bin_index(rotation, temporal["rotation_upper_edges_deg"])
    cells: list[dict[str, Any]] = []
    for t_bin in range(len(temporal["translation_upper_edges_m"]) + 1):
        for r_bin in range(len(temporal["rotation_upper_edges_deg"]) + 1):
            mask = valid & (translation_bin == t_bin) & (rotation_bin == r_bin)
            rates, has = _rates_with_minimum(mask, contradiction)
            cells.append(
                {
                    "translation_bin": t_bin,
                    "rotation_bin": r_bin,
                    "pairs": int(mask.sum()),
                    "targets": int(has.sum()),
                    "target_balanced_C": (
                        _mean_finite(rates, label="pose cell C") if bool(has.any()) else None
                    ),
                }
            )
    return cells


def _temporal_summary(
    valid: torch.Tensor,
    contradiction: torch.Tensor,
    gap: torch.Tensor,
    translation: torch.Tensor,
    rotation: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    temporal = CONFIG["temporal"]
    near = gap <= temporal["near_max_normalized"]
    far = gap >= temporal["far_min_normalized"]
    near_mask = valid & near
    far_mask = valid & far
    near_rate, near_has = _rates_with_minimum(near_mask, contradiction)
    far_rate, far_has = _rates_with_minimum(far_mask, contradiction)
    paired = near_has & far_has
    target_delta = torch.full((valid.shape[0],), float("nan"), dtype=torch.float64)
    target_delta[paired] = far_rate[paired] - near_rate[paired]
    temporal_delta = (
        _mean_finite(target_delta, label="temporal delta") if bool(paired.any()) else None
    )

    translation_bin = _bin_index(translation, temporal["translation_upper_edges_m"])
    rotation_bin = _bin_index(rotation, temporal["rotation_upper_edges_deg"])
    usable: list[dict[str, Any]] = []
    all_cells: list[dict[str, Any]] = []
    for t_bin in range(len(temporal["translation_upper_edges_m"]) + 1):
        for r_bin in range(len(temporal["rotation_upper_edges_deg"]) + 1):
            cell = (translation_bin == t_bin) & (rotation_bin == r_bin)
            cell_near = near_mask & cell
            cell_far = far_mask & cell
            cell_near_rate, cell_near_has = _rates_with_minimum(cell_near, contradiction)
            cell_far_rate, cell_far_has = _rates_with_minimum(cell_far, contradiction)
            cell_paired = cell_near_has & cell_far_has
            record = {
                "translation_bin": t_bin,
                "rotation_bin": r_bin,
                "near_pairs": int(cell_near.sum()),
                "far_pairs": int(cell_far.sum()),
                "near_targets": int(cell_near_has.sum()),
                "far_targets": int(cell_far_has.sum()),
                "paired_targets": int(cell_paired.sum()),
                "usable": False,
                "target_balanced_delta": None,
            }
            is_usable = (
                record["near_pairs"] >= temporal["minimum_pairs_per_cell_stratum"]
                and record["far_pairs"] >= temporal["minimum_pairs_per_cell_stratum"]
                and record["paired_targets"] >= temporal["minimum_targets_per_cell"]
            )
            if is_usable:
                record["usable"] = True
                record["target_balanced_delta"] = float(
                    (cell_far_rate[cell_paired] - cell_near_rate[cell_paired]).mean()
                )
                usable.append(record)
            all_cells.append(record)
    global_support = (
        int(near_mask.sum()) >= temporal["minimum_pairs_per_global_stratum"]
        and int(far_mask.sum()) >= temporal["minimum_pairs_per_global_stratum"]
    )
    pose_estimable = len(usable) >= temporal["minimum_cells"] and global_support
    if pose_estimable:
        weights = _capped_weights(
            [float(item["paired_targets"]) for item in usable],
            temporal["maximum_cell_weight"],
        )
        pose_delta = sum(
            weight * float(item["target_balanced_delta"]) for weight, item in zip(weights, usable)
        )
        for weight, item in zip(weights, usable):
            item["weight"] = weight
    else:
        pose_delta = None
    summary = {
        "near_pairs": int(near_mask.sum()),
        "far_pairs": int(far_mask.sum()),
        "paired_targets": int(paired.sum()),
        "near_target_balanced_C": (
            _mean_finite(near_rate, label="near C") if bool(near_has.any()) else None
        ),
        "far_target_balanced_C": (
            _mean_finite(far_rate, label="far C") if bool(far_has.any()) else None
        ),
        "temporal_delta": temporal_delta,
        "pose_conditioned": {
            "estimable": pose_estimable,
            "delta": pose_delta,
            "usable_cells": len(usable),
            "cells": all_cells,
        },
        "pose_overall_cells": _pose_overall_cells(valid, contradiction, translation, rotation),
    }
    return summary, target_delta


def _bootstrap_temporal(target_delta: torch.Tensor) -> dict[str, Any]:
    values = target_delta[torch.isfinite(target_delta)].numpy()
    if values.size == 0:
        return {"estimable": False, "reason": "no targets span near and far strata"}
    rng = np.random.default_rng(CONFIG["bootstrap"]["seed"])
    distribution = []
    for _ in range(CONFIG["bootstrap"]["replicates"]):
        indices = rng.integers(0, values.size, size=values.size)
        distribution.append(float(np.mean(values[indices])))
    return {
        "estimable": True,
        "target_count": int(values.size),
        "interval": _interval(distribution),
    }


def _motion_inputs(
    dense_vectors: dict[str, torch.Tensor], temporal_delta: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {
        "dense_contradiction_rate": dense_vectors["contradiction_rate"].detach().clone(),
        "dense_negative_rate": dense_vectors["negative_rate"].detach().clone(),
        "dense_absolute_relative_median": dense_vectors["absolute_relative_median"]
        .detach()
        .clone(),
        "temporal_target_delta": temporal_delta.detach().clone(),
    }


def _motion_bootstrap(
    sitting_inputs: dict[str, torch.Tensor],
    walking_inputs: dict[str, torch.Tensor],
) -> dict[str, Any]:
    def finite_numpy(inputs: dict[str, torch.Tensor], key: str) -> np.ndarray:
        value = inputs[key]
        return value[torch.isfinite(value)].detach().cpu().numpy()

    sitting = {key: finite_numpy(sitting_inputs, key) for key in sitting_inputs}
    walking = {key: finite_numpy(walking_inputs, key) for key in walking_inputs}
    required = (
        "dense_contradiction_rate",
        "dense_negative_rate",
        "dense_absolute_relative_median",
    )
    if any(sitting[key].size == 0 or walking[key].size == 0 for key in required):
        return {"estimable": False, "reason": "cross-sequence target population is empty"}
    rng = np.random.default_rng(CONFIG["bootstrap"]["seed"])
    distributions: dict[str, list[float]] = {
        "C_difference": [],
        "log_C_ratio": [],
        "log_D90_ratio": [],
        "P_minus_difference": [],
    }
    temporal_estimable = (
        sitting["temporal_target_delta"].size > 0 and walking["temporal_target_delta"].size > 0
    )
    if temporal_estimable:
        distributions["walking_temporal_delta"] = []
        distributions["temporal_excess"] = []
    for _ in range(CONFIG["bootstrap"]["replicates"]):
        s_common_size = sitting["dense_contradiction_rate"].size
        w_common_size = walking["dense_contradiction_rate"].size
        s_index = rng.integers(0, s_common_size, size=s_common_size)
        w_index = rng.integers(0, w_common_size, size=w_common_size)
        s_c = float(np.mean(sitting["dense_contradiction_rate"][s_index]))
        w_c = float(np.mean(walking["dense_contradiction_rate"][w_index]))
        distributions["C_difference"].append(w_c - s_c)
        distributions["log_C_ratio"].append(math.log(w_c / max(s_c, 0.01)))

        # Target arrays share support and length within each source.
        s_negative = sitting["dense_negative_rate"]
        w_negative = walking["dense_negative_rate"]
        s_absrel = sitting["dense_absolute_relative_median"]
        w_absrel = walking["dense_absolute_relative_median"]
        if not (
            s_negative.size == s_absrel.size == s_common_size
            and w_negative.size == w_absrel.size == w_common_size
        ):
            raise AssertionError("dense motion target vectors lost common support")
        distributions["P_minus_difference"].append(
            float(np.mean(w_negative[w_index]) - np.mean(s_negative[s_index]))
        )
        s_d90 = float(np.quantile(s_absrel[s_index], 0.90, method="linear"))
        w_d90 = float(np.quantile(w_absrel[w_index], 0.90, method="linear"))
        distributions["log_D90_ratio"].append(math.log(w_d90 / s_d90))
        if temporal_estimable:
            s_time = sitting["temporal_target_delta"]
            w_time = walking["temporal_target_delta"]
            s_t_index = rng.integers(0, s_time.size, size=s_time.size)
            w_t_index = rng.integers(0, w_time.size, size=w_time.size)
            s_delta = float(np.mean(s_time[s_t_index]))
            w_delta = float(np.mean(w_time[w_t_index]))
            distributions["walking_temporal_delta"].append(w_delta)
            distributions["temporal_excess"].append(w_delta - s_delta)
    return {
        "estimable": True,
        "replicates": CONFIG["bootstrap"]["replicates"],
        "seed": CONFIG["bootstrap"]["seed"],
        "temporal_estimable": temporal_estimable,
        "intervals": {name: _interval(values) for name, values in distributions.items()},
    }


def _standardized_pose_contrast(
    sitting_cells: list[dict[str, Any]], walking_cells: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(sitting_cells) != len(walking_cells):
        raise ValueError("pose sensitivity cell grids differ")
    supported: list[dict[str, Any]] = []
    for sitting, walking in zip(sitting_cells, walking_cells):
        if (sitting["translation_bin"], sitting["rotation_bin"]) != (
            walking["translation_bin"],
            walking["rotation_bin"],
        ):
            raise ValueError("pose sensitivity cell identities differ")
        if (
            sitting["pairs"] >= 250
            and walking["pairs"] >= 250
            and sitting["targets"] >= 50
            and walking["targets"] >= 50
        ):
            supported.append(
                {
                    "translation_bin": sitting["translation_bin"],
                    "rotation_bin": sitting["rotation_bin"],
                    "sitting_C": sitting["target_balanced_C"],
                    "walking_C": walking["target_balanced_C"],
                    "weight_basis": min(sitting["targets"], walking["targets"]),
                }
            )
    if len(supported) < CONFIG["temporal"]["minimum_cells"]:
        return {
            "estimable": False,
            "supported_cells": len(supported),
            "cells": supported,
        }
    weights = _capped_weights(
        [float(item["weight_basis"]) for item in supported],
        CONFIG["temporal"]["maximum_cell_weight"],
    )
    difference = 0.0
    for weight, cell in zip(weights, supported):
        cell["weight"] = weight
        difference += weight * (float(cell["walking_C"]) - float(cell["sitting_C"]))
    return {
        "estimable": True,
        "supported_cells": len(supported),
        "standardized_C_difference": difference,
        "cells": supported,
    }


def _motion_summary_and_gate(
    sitting_artifact: dict[str, Any],
    sitting_inputs: dict[str, torch.Tensor],
    walking_arms: dict[str, dict[str, Any]],
    walking_temporal: dict[str, Any],
    walking_inputs: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sitting_arms = sitting_artifact["metrics"]["arms"]
    sitting_temporal = sitting_artifact["metrics"]["temporal"]
    sitting_dense = sitting_arms["dense_T"]
    walking_dense = walking_arms["dense_T"]
    summary = {
        "C_difference": walking_dense["C"] - sitting_dense["C"],
        "C_ratio": walking_dense["C"] / max(sitting_dense["C"], 0.01),
        "D90_ratio": walking_dense["D90"] / sitting_dense["D90"],
        "P_minus_difference": walking_dense["P_minus"] - sitting_dense["P_minus"],
        "sitting_temporal_delta": sitting_temporal["temporal_delta"],
        "walking_temporal_delta": walking_temporal["temporal_delta"],
        "temporal_excess": (
            walking_temporal["temporal_delta"] - sitting_temporal["temporal_delta"]
            if walking_temporal["temporal_delta"] is not None
            and sitting_temporal["temporal_delta"] is not None
            else None
        ),
        "pose_standardized": _standardized_pose_contrast(
            sitting_temporal["pose_overall_cells"], walking_temporal["pose_overall_cells"]
        ),
    }
    bootstrap = _motion_bootstrap(sitting_inputs, walking_inputs)
    effect = CONFIG["motion_effect"]
    intervals = bootstrap.get("intervals", {})
    negative_discriminator = (
        summary["P_minus_difference"] >= effect["minimum_negative_difference"]
        and intervals.get("P_minus_difference", {}).get("p05", -math.inf) > 0
    )
    pose_temporal = walking_temporal["pose_conditioned"]
    temporal_discriminator = bool(
        summary["walking_temporal_delta"] is not None
        and summary["temporal_excess"] is not None
        and summary["walking_temporal_delta"] >= effect["minimum_walking_temporal_delta"]
        and summary["temporal_excess"] >= effect["minimum_temporal_excess"]
        and pose_temporal["estimable"]
        and pose_temporal["delta"] > 0
        and intervals.get("walking_temporal_delta", {}).get("p05", -math.inf) > 0
        and intervals.get("temporal_excess", {}).get("p05", -math.inf) > 0
    )
    comparisons = {
        "C_difference": (
            summary["C_difference"] >= effect["minimum_contradiction_difference"]
            and intervals.get("C_difference", {}).get("p05", -math.inf) > 0
        ),
        "C_ratio": (
            summary["C_ratio"] >= effect["minimum_contradiction_ratio"]
            and intervals.get("log_C_ratio", {}).get("p05", -math.inf) > 0
        ),
        "D90_ratio": (
            summary["D90_ratio"] >= effect["minimum_d90_ratio"]
            and intervals.get("log_D90_ratio", {}).get("p05", -math.inf) > 0
        ),
        "negative_discriminator": negative_discriminator,
        "temporal_discriminator": temporal_discriminator,
        "signed_or_temporal_discriminator": (negative_discriminator or temporal_discriminator),
    }
    comparisons["all"] = (
        comparisons["C_difference"]
        and comparisons["C_ratio"]
        and comparisons["D90_ratio"]
        and comparisons["signed_or_temporal_discriminator"]
    )
    return summary, bootstrap, {"comparisons": comparisons}


def _decode_tensor(value: Any, *, label: str) -> torch.Tensor:
    if not isinstance(value, dict):
        raise ValueError(f"{label} tensor encoding must be an object")
    if value.get("encoding") != "base64(zlib(raw-c-order))":
        raise ValueError(f"{label} tensor encoding is unsupported")
    dtype_by_name = {
        "torch.float64": np.dtype(np.float64),
        "torch.int64": np.dtype(np.int64),
        "torch.bool": np.dtype(np.bool_),
    }
    numpy_dtype = dtype_by_name.get(value.get("dtype"))
    if numpy_dtype is None:
        raise ValueError(f"{label} tensor dtype is unsupported")
    shape = value.get("shape")
    if not isinstance(shape, list) or any(
        type(dimension) is not int or dimension < 0 for dimension in shape
    ):
        raise ValueError(f"{label} tensor shape is invalid")
    try:
        raw = zlib.decompress(base64.b64decode(value["data"], validate=True))
    except (KeyError, ValueError, zlib.error) as error:
        raise ValueError(f"{label} tensor payload is invalid") from error
    if _sha256_bytes(raw) != value.get("raw_sha256"):
        raise ValueError(f"{label} tensor raw hash mismatch")
    expected_count = math.prod(shape)
    if len(raw) != expected_count * numpy_dtype.itemsize:
        raise ValueError(f"{label} tensor byte length differs from its shape")
    array = np.frombuffer(raw, dtype=numpy_dtype).reshape(shape).copy()
    return torch.from_numpy(array)


def _analysis_inputs_from_artifact(artifact: dict[str, Any]) -> dict[str, torch.Tensor]:
    encoded = artifact.get("analysis_inputs")
    expected = {
        "dense_contradiction_rate",
        "dense_negative_rate",
        "dense_absolute_relative_median",
        "temporal_target_delta",
    }
    if not isinstance(encoded, dict) or set(encoded) != expected:
        raise ValueError("development analysis input set is incomplete")
    return {
        name: _decode_tensor(value, label=f"development {name}") for name, value in encoded.items()
    }


def _tensor_entries(prefix: str, values: dict[str, torch.Tensor]) -> list[tuple[str, torch.Tensor]]:
    return [(f"{prefix}/{name}", value) for name, value in sorted(values.items())]


def _run_phase(
    archive_path: Path,
    phase: Phase,
    *,
    before_payload_decode: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if phase == "confirmatory" and before_payload_decode is None:
        raise ValueError("confirmatory audit requires an atomic attempt callback")
    if phase == "development" and before_payload_decode is not None:
        raise ValueError("development audit forbids a confirmatory attempt callback")
    archive, frames, selected, manifest_hashes, source_diagnostics, members = _load_source_metadata(
        archive_path, phase
    )
    try:
        t_frames = [item for item in selected if item.role == "T"]
        v_frames = [item for item in selected if item.role == "V"]
        h_frames = [item for item in selected if item.role == "H"]
        decoded_frames = [item for item in selected if item.role in {"T", "V"}]
        allowed_ordered = [
            archive.full_member_name(item.frame.depth.path) for item in decoded_frames
        ]
        allowed = set(allowed_ordered)
        h_depths = {archive.full_member_name(item.frame.depth.path) for item in h_frames}
        rgb_members = set(members["rgb"])
        depth_members = set(members["depth"])
        if len(allowed_ordered) != 56 or len(allowed) != 56 or len(h_depths) != 8:
            raise AssertionError("the frozen T/V/H split lost unique depth identities")
        if not allowed <= depth_members or not h_depths <= depth_members:
            raise AssertionError("selected payloads are not semantic depth members")
        if allowed & h_depths or allowed & rgb_members or h_depths & rgb_members:
            raise AssertionError("T/V, H-depth, and RGB capabilities overlap")
        if before_payload_decode is not None:
            before_payload_decode()
        archive.allowed_depth_members = allowed
        depths: dict[str, torch.Tensor] = {}
        cameras: dict[str, Camera] = {}
        decode_order = sorted(
            decoded_frames, key=lambda item: archive.member_offset(item.frame.depth.path)
        )
        for item in decode_order:
            view_id = item.frame.view_id
            if view_id in depths:
                raise ValueError("selected view IDs are not unique")
            depths[view_id] = archive.decode_depth(item.frame.depth.path)
            cameras[view_id] = base._camera_from_pose(item.frame.pose)
        base._validate_payload_access(archive, allowed)
        attempted = set(archive.attempted_depth_members)
        decoded = set(archive.decoded_depth_members)
        if attempted & rgb_members or decoded & rgb_members:
            raise AssertionError("an RGB payload was attempted or decoded")
        if attempted & h_depths or decoded & h_depths:
            raise AssertionError("an H-role depth payload was attempted or decoded")

        t_ids = {item.frame.view_id for item in t_frames}
        v_ids = {item.frame.view_id for item in v_frames}
        construction_depths = {
            view_id: depth for view_id, depth in depths.items() if view_id in t_ids
        }
        validation_depths = {
            view_id: depth for view_id, depth in depths.items() if view_id in v_ids
        }
        if set(construction_depths) != t_ids or set(validation_depths) != v_ids:
            raise AssertionError("backend capabilities differ from the exact role views")
        if set(construction_depths) & set(validation_depths):
            raise AssertionError("construction and validation capabilities overlap")
        targets = base._construct_targets(
            t_frames, cameras, base.TumDepthBackend(construction_depths)
        )
        target_entries = base._target_tensor_entries(targets)
        target_hash_before = base._tensor_hash(target_entries)
        dense_points, dense_counts = _dense_construction_points(
            t_frames, cameras, construction_depths, targets
        )
        dense_hash_before = base._tensor_hash([("dense_points_world", dense_points)])
        raw, per_v = _audit_signed(
            targets,
            dense_points,
            v_frames,
            cameras,
            validation_depths,
            t_frames,
            selected,
        )
        if target_hash_before != base._tensor_hash(base._target_tensor_entries(targets)):
            raise AssertionError("sparse target tensors mutated during audit")
        if dense_hash_before != base._tensor_hash([("dense_points_world", dense_points)]):
            raise AssertionError("dense construction tensor mutated during audit")

        sparse_summary, sparse_vectors = _arm_summary(
            "sparse",
            raw["sparse_valid"],
            raw["positive"],
            raw["negative"],
            raw["signed_relative"],
        )
        dense_summary, dense_vectors = _arm_summary(
            "dense_T",
            raw["dense_valid"],
            raw["positive"],
            raw["negative"],
            raw["signed_relative"],
        )
        arms = {"sparse": sparse_summary, "dense_T": dense_summary}
        attribution, attribution_vectors = _attribution_summary(raw, arms)
        occlusion_bootstrap = _bootstrap_occlusion(
            attribution,
            attribution_vectors,
            sparse_vectors,
            dense_vectors,
        )
        occlusion_gate = _occlusion_gate(attribution, arms, occlusion_bootstrap)
        contradiction = raw["positive"] | raw["negative"]
        temporal, temporal_target_delta = _temporal_summary(
            raw["dense_valid"],
            contradiction,
            raw["normalized_time_gap"],
            raw["translation_baseline_m"],
            raw["rotation_baseline_deg"],
        )
        temporal_bootstrap = _bootstrap_temporal(temporal_target_delta)
        analysis_inputs = _motion_inputs(dense_vectors, temporal_target_delta)

        selected_manifest = base._selected_manifest(selected)
        construction_manifest = [item for item in selected_manifest if item["role"] == "T"]
        construction_manifest_hash = _sha256_bytes(_canonical_json(construction_manifest))
        target_composite_hash = base._target_composite_hash(
            target_hash_before,
            construction_manifest_hash,
            config_id=CONFIG_ID,
        )
        return {
            "phase": phase,
            "source": {
                **SOURCES[phase],
                "archive_path": str(archive_path.resolve()),
                "manifest_hashes": manifest_hashes,
                **source_diagnostics,
            },
            "association_max_rgb_depth_delta_ns": max(frame.rgb_depth_delta_ns for frame in frames),
            "selected_frames": selected_manifest,
            "frozen_payload_allowlist": {
                "selected_order_members": allowed_ordered,
                "selected_order_sha256": _sha256_bytes(_canonical_json(allowed_ordered)),
                "count": len(allowed_ordered),
            },
            "attempted_payload_members": archive.attempted_depth_members,
            "decoded_payload_members": archive.decoded_depth_members,
            "rgb_payload_count": 0,
            "h_payload_count": 0,
            "backend_capabilities": {
                "construction_view_ids": sorted(construction_depths),
                "validation_view_ids": sorted(validation_depths),
                "disjoint": True,
            },
            "target": {
                "hash": target_composite_hash,
                "tensor_sha256": target_hash_before,
                "construction_manifest_sha256": construction_manifest_hash,
                "count": targets.points_world.shape[0],
                "per_t_counts": list(targets.per_t_counts),
                "bounds": base._bounds(targets.points_world),
                "serialized": {name: base._encoded_tensor(value) for name, value in target_entries},
            },
            "dense_construction": {
                "tensor_sha256": dense_hash_before,
                "count": dense_points.shape[0],
                "explicit_sparse_prefix_count": targets.points_world.shape[0],
                "stride_grid_count_per_t": list(dense_counts),
                "bounds": base._bounds(dense_points),
                "serialized_points_world": base._encoded_tensor(dense_points),
            },
            "audit_tensor_sha256": base._tensor_hash(_tensor_entries("audit", raw)),
            "audit_serialized": {name: base._encoded_tensor(value) for name, value in raw.items()},
            "analysis_inputs": {
                name: base._encoded_tensor(value) for name, value in analysis_inputs.items()
            },
            "metrics": {
                "arms": arms,
                "attribution": attribution,
                "temporal": temporal,
            },
            "bootstrap": {
                "occlusion": occlusion_bootstrap,
                "temporal": temporal_bootstrap,
            },
            "occlusion_gate": occlusion_gate,
            "diagnostics": {"per_validation_view": per_v},
        }
    finally:
        archive.__exit__(None, None, None)


def _write_bytes_exclusive(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    directory_descriptor = os.open(path.parent, directory_flags)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return _sha256_bytes(payload)


def _write_json_once(path: Path, value: dict[str, Any]) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return _write_bytes_exclusive(path, payload)


def _preflight_artifact_path(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite append-only artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.parent.is_dir() or not os.access(path.parent, os.W_OK):
        raise PermissionError(f"artifact parent is not writable: {path.parent}")


def _development_decision(
    development_artifact_path: Path,
    development_artifact_sha256: str,
    result: dict[str, Any],
    implementation: dict[str, Any],
) -> dict[str, Any]:
    authorized = bool(result["occlusion_gate"]["comparisons"]["all"])
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "config_id": CONFIG_ID,
        "status": "confirmation_authorized" if authorized else "development_stopped",
        "confirmation_authorized": authorized,
        "development_archive_sha256": SOURCES["development"]["sha256"],
        "confirmatory_archive_sha256": SOURCES["confirmatory"]["sha256"],
        "development_artifact": str(development_artifact_path.resolve()),
        "development_artifact_sha256": development_artifact_sha256,
        "implementation_aggregate_sha256": implementation["aggregate_sha256"],
        "preregistration_sha256": implementation["files"][str(PREREGISTRATION)],
        "development_occlusion_gate": result["occlusion_gate"],
        "confirmatory_predicates_sha256": _sha256_bytes(
            _canonical_json(
                {
                    "occlusion_support": CONFIG["occlusion_support"],
                    "occlusion_effect": CONFIG["occlusion_effect"],
                    "motion_effect": CONFIG["motion_effect"],
                    "bootstrap": CONFIG["bootstrap"],
                }
            )
        ),
    }


def _validate_development_decision(
    path: Path, root: Path, implementation: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], str]:
    payload = path.read_bytes()
    decision_sha256 = _sha256_bytes(payload)
    decision = base._json_object_from_bytes(payload, label="development decision")
    if decision.get("schema_version") != 1 or decision.get("experiment") != EXPERIMENT:
        raise ValueError("development decision identity mismatch")
    if decision.get("config_id") != CONFIG_ID:
        raise ValueError("development and confirmatory configurations differ")
    if not decision.get("confirmation_authorized"):
        raise ValueError("development gate did not authorize walking PNG decode")
    if decision.get("development_archive_sha256") != SOURCES["development"]["sha256"]:
        raise ValueError("development decision names the wrong sitting source")
    if decision.get("confirmatory_archive_sha256") != SOURCES["confirmatory"]["sha256"]:
        raise ValueError("development decision names the wrong walking source")
    if decision.get("implementation_aggregate_sha256") != implementation["aggregate_sha256"]:
        raise ValueError("implementation changed after development")
    if decision.get("preregistration_sha256") != implementation["files"][str(PREREGISTRATION)]:
        raise ValueError("preregistration changed after development")
    artifact_path = Path(decision.get("development_artifact", ""))
    if not artifact_path.is_absolute():
        artifact_path = root / artifact_path
    artifact_payload = artifact_path.read_bytes()
    if _sha256_bytes(artifact_payload) != decision.get("development_artifact_sha256"):
        raise ValueError("development artifact hash differs from its decision")
    artifact = base._json_object_from_bytes(
        artifact_payload, label="signed-attribution development artifact"
    )
    if (
        artifact.get("experiment") != EXPERIMENT
        or artifact.get("phase") != "development"
        or artifact.get("config_id") != CONFIG_ID
    ):
        raise ValueError("development artifact identity mismatch")
    if artifact.get("source", {}).get("archive_sha256") != SOURCES["development"]["sha256"]:
        raise ValueError("development artifact source mismatch")
    if (
        artifact.get("implementation", {}).get("aggregate_sha256")
        != implementation["aggregate_sha256"]
    ):
        raise ValueError("development artifact implementation mismatch")
    if artifact.get("occlusion_gate") != decision.get("development_occlusion_gate"):
        raise ValueError("development gate differs between artifact and decision")
    if not artifact.get("occlusion_gate", {}).get("comparisons", {}).get("all"):
        raise ValueError("development artifact itself does not authorize confirmation")
    expected_predicates = _sha256_bytes(
        _canonical_json(
            {
                "occlusion_support": CONFIG["occlusion_support"],
                "occlusion_effect": CONFIG["occlusion_effect"],
                "motion_effect": CONFIG["motion_effect"],
                "bootstrap": CONFIG["bootstrap"],
            }
        )
    )
    if decision.get("confirmatory_predicates_sha256") != expected_predicates:
        raise ValueError("confirmatory predicates changed after development")
    return decision, artifact, decision_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "confirmatory"), required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--decision-output",
        type=Path,
        help="required in development; append-only walking authorization decision",
    )
    parser.add_argument(
        "--decision",
        type=Path,
        help="required in confirmatory; frozen passing development decision",
    )
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    phase: Phase = args.phase
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if phase == "development":
        if args.decision_output is None or args.decision is not None:
            raise ValueError("development requires --decision-output and forbids --decision")
        _preflight_artifact_path(args.decision_output)
    else:
        if args.decision is None or args.decision_output is not None:
            raise ValueError("confirmatory requires --decision and forbids --decision-output")
    _preflight_artifact_path(args.output)
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)

    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    torch.set_num_threads(args.threads)
    implementation = _implementation_metadata(root)
    predecode_seal, predecode_seal_sha256 = _validate_predecode_seal(root, implementation)
    git = _git_metadata(root)
    started = datetime.now(timezone.utc)
    start_clock = time.perf_counter()
    development_decision = None
    development_artifact = None
    development_decision_sha256 = None
    confirmatory_seal_sha256 = None
    before_payload_decode: Callable[[], None] | None = None

    if phase == "confirmatory":
        (
            development_decision,
            development_artifact,
            development_decision_sha256,
        ) = _validate_development_decision(args.decision, root, implementation)

        def consume_confirmatory_attempt() -> None:
            nonlocal confirmatory_seal_sha256
            if confirmatory_seal_sha256 is not None:
                raise AssertionError("confirmatory attempt callback ran more than once")
            seal = {
                "schema_version": 1,
                "experiment": EXPERIMENT,
                "status": "started",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "confirmatory_archive_path": str(args.archive.resolve()),
                "confirmatory_archive_sha256": SOURCES["confirmatory"]["sha256"],
                "development_decision_path": str(args.decision.resolve()),
                "development_decision_sha256": development_decision_sha256,
                "requested_output": str(args.output.resolve()),
                "config_id": CONFIG_ID,
                "implementation_aggregate_sha256": implementation["aggregate_sha256"],
                "preregistration_sha256": implementation["files"][str(PREREGISTRATION)],
            }
            confirmatory_seal_sha256 = _write_json_once(root / CONFIRMATORY_SEAL, seal)

        before_payload_decode = consume_confirmatory_attempt

    result = _run_phase(args.archive, phase, before_payload_decode=before_payload_decode)
    completed = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "phase": phase,
        "config": CONFIG,
        "config_id": CONFIG_ID,
        "preregistration": str(PREREGISTRATION),
        "acquisition_manifest": str(ACQUISITION_MANIFEST),
        "implementation": implementation,
        "predecode_seal": {
            "path": str(PREDECODE_SEAL),
            "sha256": predecode_seal_sha256,
            "content": predecode_seal,
        },
        "git": git,
        "environment": _environment(),
        "started_utc": started.isoformat(),
        "completed_utc": completed.isoformat(),
        "runtime_seconds": time.perf_counter() - start_clock,
        **result,
    }
    if phase == "development":
        payload["derived_confirmation_authorized"] = bool(
            payload["occlusion_gate"]["comparisons"]["all"]
        )
    else:
        if confirmatory_seal_sha256 is None or development_artifact is None:
            raise AssertionError("confirmatory attempt was not sealed before PNG decode")
        sitting_inputs = _analysis_inputs_from_artifact(development_artifact)
        walking_inputs = _analysis_inputs_from_artifact(payload)
        motion_summary, motion_bootstrap, motion_gate = _motion_summary_and_gate(
            development_artifact,
            sitting_inputs,
            payload["metrics"]["arms"],
            payload["metrics"]["temporal"],
            walking_inputs,
        )
        payload["development_decision"] = {
            "path": str(args.decision.resolve()),
            "sha256": development_decision_sha256,
            "content": development_decision,
        }
        payload["confirmatory_attempt_seal"] = {
            "path": str(CONFIRMATORY_SEAL),
            "sha256": confirmatory_seal_sha256,
        }
        payload["motion_regime"] = motion_summary
        payload["bootstrap"]["motion_regime"] = motion_bootstrap
        payload["motion_gate"] = motion_gate
        walking_occlusion = payload["occlusion_gate"]["comparisons"]["all"]
        rigidity = motion_gate["comparisons"]["all"]
        if walking_occlusion and rigidity:
            classification = "TWO_MECHANISMS_SUPPORTED"
        elif walking_occlusion or rigidity:
            classification = "PARTIAL_ATTRIBUTION"
        else:
            classification = "ATTRIBUTION_REJECTED"
        payload["classification"] = classification

    artifact_sha256 = _write_json_once(args.output, payload)
    if phase == "development":
        decision = _development_decision(args.output, artifact_sha256, payload, implementation)
        _write_json_once(args.decision_output, decision)
    print(
        json.dumps(
            {
                "phase": phase,
                "output": str(args.output),
                "artifact_sha256": artifact_sha256,
                "occlusion_gate": payload["occlusion_gate"]["comparisons"],
                "classification": payload.get("classification"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
