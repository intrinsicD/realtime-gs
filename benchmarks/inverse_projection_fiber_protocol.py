#!/usr/bin/env python3
"""Generic protocol engine for reviewed inverse-projection-fiber experiments.

Identity, roots, receipt domains, and bound artifacts are supplied by a thin reviewed wrapper.
The scientific path uses only component-level projected Gaussian geometry: no RGB, alpha
compositing, opacity objective, spherical harmonics objective, visibility filtering, or topology
control enters it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from benchmarks.inverse_projection_fiber_transaction import (
    DEVELOPMENT_ONLY,
    MutationReport,
    OwnedMutationError,
    Ownership,
    PreparedJSON,
    ReceiptDomain,
    canonical_json_bytes,
    capture_entry,
    exchange_owned,
    open_directory,
    prepare_json,
    publish_exclusive,
    require_rename_exchange,
)
from torch import nn

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D, quat_to_rotmat
from rtgs.data.synthetic import make_gt_gaussians, make_ring_cameras
from rtgs.lift.inverse_projection_fiber import (
    FreeGaussianGeometry,
    InverseProjectionFiber,
    covariance_projection_design,
    hard_correspondence_loss,
    pairwise_center_cost,
    pairwise_conic_cost,
    pairwise_gaussian_geometry_cost,
    spd_affine_invariant_squared,
)
from rtgs.render.projection import project_covariances_ewa

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProtocolSpec:
    """Reviewed identity and evidence closure for one protocol execution.

    Scientific helpers below deliberately do not own an experiment identity.  A thin reviewed
    wrapper supplies this immutable object, which in turn supplies the one official and one
    development receipt domain.  Paths in ``declared_source_paths`` are repository-relative.
    """

    protocol_label: str
    official_domain: ReceiptDomain
    development_domain: ReceiptDomain
    preregistration: Path
    preregistration_review: Path
    verification_receipt: Path
    implementation_review: Path
    bound_historical_sha256: tuple[tuple[Path, str], ...]
    declared_source_paths: tuple[Path, ...]
    scene_roots: tuple[int, ...]
    initial_depth_roots: tuple[int, ...]
    verification_base: Path
    official_out: Path
    official_artifacts_dir: Path
    development_roots: tuple[int, ...]
    verification_test_paths: tuple[Path, ...]
    verification_scanner: Path

    def __post_init__(self) -> None:
        if len(self.scene_roots) != 3 or len(self.initial_depth_roots) != 3:
            raise ValueError("the frozen protocol requires exactly three paired roots")
        if len(set(self.scene_roots + self.initial_depth_roots)) != 6:
            raise ValueError("official roots must be pairwise distinct")
        if self.official_domain.label != "official":
            raise ValueError("official_domain must carry the official label")
        if self.development_domain.label != "development":
            raise ValueError("development_domain must carry the development label")
        if (
            self.official_domain.protocol_label != self.protocol_label
            or self.development_domain.protocol_label != self.protocol_label
        ):
            raise ValueError("both receipt domains must share the protocol label")
        official_roots = self.scene_roots + self.initial_depth_roots
        if self.official_domain.permitted_roots != official_roots:
            raise ValueError("official domain roots must exactly match ProtocolSpec roots")
        if len(self.official_domain.permitted_root_consumption_statuses) != 3:
            raise ValueError("official domain must define exactly three ordered lifecycle statuses")
        if not set(official_roots).issubset(self.development_domain.forbidden_roots):
            raise ValueError("development domain must prohibit every official root")
        if self.official_domain.namespace == self.development_domain.namespace:
            raise ValueError("official and development namespaces must differ")
        if len(set(self.declared_source_paths)) != len(self.declared_source_paths):
            raise ValueError("declared source paths must be unique")
        if not self.development_roots or any(
            type(root) is not int for root in self.development_roots
        ):
            raise ValueError("development roots must be a non-empty integer tuple")
        if set(self.development_roots) & set(official_roots):
            raise ValueError("development roots must not overlap official roots")
        if not self.verification_test_paths or len(set(self.verification_test_paths)) != len(
            self.verification_test_paths
        ):
            raise ValueError("verification test paths must be non-empty and unique")
        for path in (*self.verification_test_paths, self.verification_scanner):
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("verification source paths must be repository-relative")
        official_out = self.official_out.expanduser().resolve()
        official_artifacts_dir = self.official_artifacts_dir.expanduser().resolve()
        if (
            official_out == official_artifacts_dir
            or official_out in official_artifacts_dir.parents
            or official_artifacts_dir in official_out.parents
        ):
            raise ValueError("official output and artifact paths must be disjoint and non-nested")

    @property
    def bound_hashes(self) -> dict[Path, str]:
        return dict(self.bound_historical_sha256)


ARMS = (
    "free_center",
    "free_conic",
    "fiber_center",
    "fiber_conic",
    "oracle",
    "shuffled",
)
N_GAUSSIANS = 8
N_CAMERAS = 6
OPTIMIZATION_VIEWS = (0, 1, 2, 3)
HELDOUT_VIEWS = (4, 5)
N_HYPOTHESES = len(OPTIMIZATION_VIEWS) * N_GAUSSIANS
DEPTH_LOWER = 1.2
DEPTH_UPPER = 3.6
DILATION = 0.3
CONIC_WEIGHT = 0.25
FREE_SOURCE_WEIGHT = 25.0
LEARNING_RATE = 0.025
UPDATES = 400
CHECKPOINT_INTERVAL = 20
TRAIN_ASSOCIATION_DENOMINATOR = N_HYPOTHESES * 3
HELDOUT_ASSOCIATION_DENOMINATOR = N_HYPOTHESES * 2
TRACK_DENOMINATOR = N_HYPOTHESES

ERROR_METRICS = (
    "source_center_max_px",
    "source_covariance_relative_frobenius_max",
    "gt_center_distance_median_world",
    "gt_center_distance_p90_world",
    "gt_covariance_affine_invariant_median",
    "gt_covariance_affine_invariant_squared_median",
    "train_gt_projected_center_cost",
    "train_gt_projected_conic_cost",
    "train_gt_projected_geometry_cost",
    "heldout_gt_projected_center_cost",
    "heldout_gt_projected_conic_cost",
    "heldout_gt_projected_geometry_cost",
)
FRACTION_METRICS = (
    "train_association_accuracy",
    "heldout_association_accuracy",
    "correct_track_fraction",
    "consistent_track_fraction",
    "depth_fraction_within_1e-4_of_bound",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_bytes(list(tensor.shape)))
    digest.update(b"\0")
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_existing_parent(path: Path) -> None:
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise FileNotFoundError(
            f"artifact parent must be an existing real directory: {path.parent}"
        )


def _mkdir_exclusive_durable(path: Path) -> None:
    _require_existing_parent(path)
    path.mkdir(exist_ok=False)
    _fsync_directory(path.parent)


def _write_receipt_exclusive(
    path: Path,
    domain: ReceiptDomain,
    kind: str,
    receipt: dict[str, Any],
) -> PreparedJSON:
    """Prepare, retain, exclusively link, fsync, and verify one domain receipt."""

    _require_existing_parent(path)
    directory_fd = open_directory(path.parent)
    try:
        prepared = prepare_json(directory_fd, path.name, domain, kind, receipt)
        report = publish_exclusive(directory_fd, prepared)
        if not report.accepted or report.recovery_uncertainty or report.publication_error:
            raise RuntimeError(f"exclusive publication was not accepted for {path}")
        return prepared
    finally:
        os.close(directory_fd)


def _mutation_report_receipt(report: MutationReport) -> dict[str, Any]:
    def entry(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        ownership = value.ownership
        return {
            "name": value.name,
            "state": value.state,
            "sha256": None if ownership is None else ownership.sha256,
            "device": value.device if ownership is None else ownership.device,
            "inode": value.inode if ownership is None else ownership.inode,
            "mode": value.mode if ownership is None else ownership.mode,
            "error_class": value.error_class,
            "error_message": value.error_message,
        }

    return {
        "operation": report.operation,
        "target_name": report.target_name,
        "recovery_name": report.recovery_name,
        "last_observed": report.last_observed,
        "accepted": report.accepted,
        "public_entry": entry(report.public_entry),
        "recovery_entry": entry(report.recovery_entry),
        "displaced_entry": entry(report.displaced_entry),
        "events": list(report.events),
        "errors": list(report.errors),
        "recovery_uncertainty": report.recovery_uncertainty,
        "publication_error": report.publication_error,
    }


def _validate_durable_receipt_descriptor(
    descriptor: dict[str, Any],
    *,
    domain: ReceiptDomain,
    kind: str,
    expected_parent: Path,
) -> None:
    receipt = descriptor.get("receipt")
    if not isinstance(receipt, dict):
        raise RuntimeError("durable receipt descriptor has no receipt mapping")
    domain.validate_receipt(kind, receipt)
    path = Path(descriptor.get("path", ""))
    recovery_path = Path(descriptor.get("recovery_path", ""))
    expected_sha256 = descriptor.get("sha256")
    if path.parent != expected_parent or recovery_path.parent != expected_parent:
        raise RuntimeError("durable receipt descriptor escaped its expected directory")
    if _sha256_bytes(canonical_json_bytes(receipt)) != expected_sha256:
        raise RuntimeError("durable receipt descriptor hash does not match its receipt")
    directory_fd = open_directory(expected_parent)
    try:
        public = capture_entry(
            directory_fd,
            path.name,
            expected_sha256=expected_sha256,
        )
        recovery = capture_entry(
            directory_fd,
            recovery_path.name,
            expected_sha256=expected_sha256,
        )
    finally:
        os.close(directory_fd)
    if (
        public.ownership is None
        or recovery.ownership is None
        or not public.ownership.same_identity_and_hash(recovery.ownership)
    ):
        raise RuntimeError("public and retained receipt entries do not share ownership")


def _save_ply_exclusive(
    path: Path,
    gaussians: Gaussians3D,
    *,
    domain: ReceiptDomain,
) -> str:
    if not isinstance(domain, ReceiptDomain):
        raise TypeError("a receipt domain is required for every PLY producer")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    _require_existing_parent(path)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    try:
        gaussians.detach().to("cpu").save_ply(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)
    return _sha256_file(path)


def _git_output(*args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.rstrip("\n")


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _experiment_config(
    spec: ProtocolSpec,
    domain: ReceiptDomain,
    *,
    root_consumption_status: str,
) -> dict[str, Any]:
    payload = {
        "bound_historical_sha256": {
            path.as_posix(): digest for path, digest in spec.bound_historical_sha256
        },
        "arms": list(ARMS),
        "scene": {
            "n_gaussians": N_GAUSSIANS,
            "n_cameras": N_CAMERAS,
            "image_size": 64,
            "optimization_views": list(OPTIMIZATION_VIEWS),
            "heldout_views": list(HELDOUT_VIEWS),
            "visibility_filtering": False,
            "ewa_dilation_pixel_squared": DILATION,
            "hypothesis_order": "source view, then primitive",
        },
        "initial_depth": {
            "distribution": "torch.rand float64 uniform affine map",
            "lower": DEPTH_LOWER,
            "upper": DEPTH_UPPER,
            "strict_endpoints": True,
        },
        "paired_initialization": {
            "constructor_inputs": "byte-identical tensors",
            "fiber_realized_geometry": "byte-identical tensors",
            "free_realized_geometry": "float64 Cholesky round-trip within tolerance",
            "realized_mean_maximum_absolute": 0.0,
            "realized_covariance_relative_frobenius_maximum": 1e-14,
            "source_center_maximum_px": 1e-10,
            "source_covariance_relative_frobenius_maximum": 1e-12,
        },
        "root_stream": {
            "required_ambient_default_dtype": "torch.float32",
            "required_ambient_default_device": "cpu",
            "generate_before_float64_promotion": True,
        },
        "lifecycle": {
            "outputs": "exclusive, disjoint, non-nested",
            "post_root_failure": "durable INVALID terminal and aggregate receipts",
            "retry": False,
            "start_end_source_hashes_must_match": True,
        },
        "objective": {
            "center": "delta^T stop_gradient((P+Q)/2)^-1 delta",
            "conic": "sum(log(eig(P^-1/2 Q P^-1/2))^2)",
            "geometry": "center + 0.25*conic",
            "non_source_reduction": (
                "three non-source costs averaged per hypothesis, then 32 hypotheses averaged"
            ),
            "latent_tie_break": "torch.min first index",
            "free_source_weight": FREE_SOURCE_WEIGHT,
            "shuffled_target": "(GT_ID + 1 + target_view_index) mod 8",
        },
        "optimizer": {
            "name": "torch.optim.Adam",
            "dtype": "float64",
            "device": "cpu",
            "learning_rate": LEARNING_RATE,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "updates": UPDATES,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "schedule": None,
            "clipping": None,
            "early_stopping": False,
            "checkpoint_selection": False,
        },
        "evaluation": {
            "association_cost": "ordinary nearest center + 0.25*conic",
            "train_association_denominator": TRAIN_ASSOCIATION_DENOMINATOR,
            "heldout_association_denominator": HELDOUT_ASSOCIATION_DENOMINATOR,
            "correct_track_denominator": TRACK_DENOMINATOR,
            "consistent_track_denominator": TRACK_DENOMINATOR,
            "gt_covariance_primary_summary": ("median sqrt(sum(log(relative eigenvalue)^2))"),
            "gt_covariance_squared_diagnostic": ("median sum(log(relative eigenvalue)^2)"),
            "heldout_cost_for_gate_5": "mean heldout GT-ID center + 0.25*conic",
            "quantile": "torch.quantile float64 linear interpolation",
            "error_geometric_mean_floor": 1e-12,
        },
        "excluded": [
            "RGB",
            "opacity objective",
            "spherical harmonics objective",
            "visibility filtering",
            "occlusion",
            "global track partitioning",
            "merge",
            "split",
            "prune",
            "teleport",
        ],
    }
    roots: tuple[int, ...] = ()
    if domain is spec.official_domain:
        payload["scene_roots"] = list(spec.scene_roots)
        payload["initial_depth_roots"] = list(spec.initial_depth_roots)
        roots = spec.scene_roots + spec.initial_depth_roots
    return domain.make_receipt(
        "config",
        payload,
        root_consumption_status=root_consumption_status,
        roots=roots,
    )


def _loaded_local_source_paths(spec: ProtocolSpec) -> set[Path]:
    paths: set[Path] = set(spec.declared_source_paths)
    for module in tuple(sys.modules.values()):
        raw_path = getattr(module, "__file__", None)
        if raw_path is None:
            continue
        raw = Path(raw_path)
        if not raw.is_absolute():
            continue
        path = raw.resolve()
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            continue
        if relative.suffix == ".py":
            paths.add(relative)
    return paths


def _source_observation(spec: ProtocolSpec) -> dict[str, Any]:
    """Observe the closure once, retaining partial hashes and bounded per-path errors."""

    hashes: dict[str, str] = {}
    errors: dict[str, str] = {}
    try:
        paths = sorted(_loaded_local_source_paths(spec), key=lambda item: item.as_posix())
    except Exception as error:
        return {
            "hashes": hashes,
            "errors": {"<closure-enumeration>": f"{type(error).__name__}: {error}"},
        }
    for path in paths:
        label = path.as_posix()
        try:
            hashes[label] = _sha256_file(ROOT / path)
        except Exception as error:
            errors[label] = f"{type(error).__name__}: {error}"
    return {"hashes": hashes, "errors": errors}


def _provenance(
    spec: ProtocolSpec,
    domain: ReceiptDomain,
    *,
    root_consumption_status: str,
) -> dict[str, Any]:
    status = _git_output("status", "--short")
    source_observation = _source_observation(spec)
    payload = {
        "timestamp_ns": time.time_ns(),
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_default_device": str(torch.get_default_device()),
        "git_revision": _git_output("rev-parse", "HEAD"),
        "git_status_sha256": None if status is None else _sha256_bytes(status.encode("utf-8")),
        "git_dirty": status not in {None, ""},
        "source_observation": source_observation,
    }
    return domain.make_receipt(
        "provenance",
        payload,
        root_consumption_status=root_consumption_status,
        roots=(spec.scene_roots + spec.initial_depth_roots)
        if domain is spec.official_domain
        else (),
    )


def _promoted_gt_geometry(root: int) -> tuple[torch.Tensor, torch.Tensor]:
    gt = make_gt_gaussians(n=N_GAUSSIANS, seed=root)
    means = gt.means.to(dtype=torch.float64, device="cpu")
    rotations = quat_to_rotmat(gt.quats.to(dtype=torch.float64, device="cpu"))
    scales = gt.log_scales.to(dtype=torch.float64, device="cpu").exp()
    rotated_scales = rotations * scales[:, None, :]
    covariances = rotated_scales @ rotated_scales.transpose(-1, -2)
    return means, covariances


def _camera_receipt(camera: Camera, index: int) -> dict[str, Any]:
    values = {
        "index": index,
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "width": camera.width,
        "height": camera.height,
        "R_sha256": _tensor_sha256(camera.R),
        "t_sha256": _tensor_sha256(camera.t),
    }
    return {**values, "receipt_sha256": _sha256_bytes(_canonical_bytes(values))}


def _build_scene(scene_root: int, depth_root: int) -> dict[str, Any]:
    cameras = tuple(make_ring_cameras(n_cameras=N_CAMERAS, image_size=64))
    gt_means, gt_covariances = _promoted_gt_geometry(scene_root)
    target_means2d: list[torch.Tensor] = []
    target_covariances2d: list[torch.Tensor] = []
    target_depths: list[torch.Tensor] = []
    for camera in cameras:
        projection = project_covariances_ewa(
            gt_means,
            gt_covariances,
            camera,
            dilation=DILATION,
        )
        if not bool(torch.isfinite(projection.means2d).all()):
            raise RuntimeError("official projection means are non-finite")
        if not bool(torch.isfinite(projection.covariances2d).all()):
            raise RuntimeError("official projection covariances are non-finite")
        if not bool((projection.depth > 0).all()):
            raise RuntimeError("an official component has non-positive depth")
        if not bool((torch.linalg.eigvalsh(projection.covariances2d) > 0).all()):
            raise RuntimeError("an official component projection is not SPD")
        target_means2d.append(projection.means2d)
        target_covariances2d.append(projection.covariances2d)
        target_depths.append(projection.depth)

    source_view_indices = torch.arange(
        len(OPTIMIZATION_VIEWS),
        dtype=torch.long,
    ).repeat_interleave(N_GAUSSIANS)
    source_component_indices = torch.arange(N_GAUSSIANS, dtype=torch.long).repeat(
        len(OPTIMIZATION_VIEWS)
    )
    source_means2d = torch.cat(
        [target_means2d[view] for view in OPTIMIZATION_VIEWS],
        dim=0,
    )
    source_covariances2d = torch.cat(
        [target_covariances2d[view] for view in OPTIMIZATION_VIEWS],
        dim=0,
    )
    generator = torch.Generator(device="cpu").manual_seed(depth_root)
    initial_depths = DEPTH_LOWER + (DEPTH_UPPER - DEPTH_LOWER) * torch.rand(
        N_HYPOTHESES,
        generator=generator,
        dtype=torch.float64,
    )
    if bool((initial_depths <= DEPTH_LOWER).any()) or bool((initial_depths >= DEPTH_UPPER).any()):
        raise RuntimeError("the paired initial-depth generator produced an endpoint")
    input_hashes = {
        "gt_means_sha256": _tensor_sha256(gt_means),
        "gt_covariances_sha256": _tensor_sha256(gt_covariances),
        "cameras": [_camera_receipt(camera, index) for index, camera in enumerate(cameras)],
        "targets": [
            {
                "view_index": index,
                "means2d_sha256": _tensor_sha256(target_means2d[index]),
                "covariances2d_sha256": _tensor_sha256(target_covariances2d[index]),
                "depths_sha256": _tensor_sha256(target_depths[index]),
            }
            for index in range(len(cameras))
        ],
        "source_view_indices_sha256": _tensor_sha256(source_view_indices),
        "source_component_indices_sha256": _tensor_sha256(source_component_indices),
        "source_means2d_sha256": _tensor_sha256(source_means2d),
        "source_covariances2d_sha256": _tensor_sha256(source_covariances2d),
        "initial_depths_sha256": _tensor_sha256(initial_depths),
    }
    return {
        "scene_root": scene_root,
        "depth_root": depth_root,
        "cameras": cameras,
        "gt_means": gt_means,
        "gt_covariances": gt_covariances,
        "target_means2d": tuple(target_means2d),
        "target_covariances2d": tuple(target_covariances2d),
        "target_depths": tuple(target_depths),
        "source_view_indices": source_view_indices,
        "source_component_indices": source_component_indices,
        "source_means2d": source_means2d,
        "source_covariances2d": source_covariances2d,
        "initial_depths": initial_depths,
        "initial_depth_sha256": _tensor_sha256(initial_depths),
        "input_hashes": input_hashes,
    }


def _write_scene_input_receipt(
    scene: dict[str, Any],
    artifacts_dir: Path,
    *,
    domain: ReceiptDomain,
    root_consumption_status: str,
) -> dict[str, Any]:
    scene_directory = artifacts_dir / f"scene_{scene['scene_root']}"
    _mkdir_exclusive_durable(scene_directory)
    roots = (scene["scene_root"], scene["depth_root"]) if domain.label == "official" else ()
    receipt = domain.make_receipt(
        "raw_inputs",
        {
            "scene_root": scene["scene_root"],
            "initial_depth_root": scene["depth_root"],
            "input_hashes": scene["input_hashes"],
        },
        root_consumption_status=root_consumption_status,
        roots=roots,
        official_phase="raw_input_evidence" if domain.label == "official" else None,
    )
    receipt_path = scene_directory / "inputs.json"
    prepared = _write_receipt_exclusive(receipt_path, domain, "raw_inputs", receipt)
    return {
        "scene_root": scene["scene_root"],
        "initial_depth_root": scene["depth_root"],
        "path": str(receipt_path),
        "sha256": prepared.payload_sha256,
        "recovery_path": str(scene_directory / prepared.recovery_name),
        "input_hashes": scene["input_hashes"],
        "receipt": receipt,
    }


def _write_common_constructor_receipt(
    scene: dict[str, Any],
    raw_descriptor: dict[str, Any],
    artifacts_dir: Path,
    domain: ReceiptDomain,
    root_consumption_status: str,
) -> tuple[nn.Module | None, torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
    common_fiber: nn.Module | None = None
    common_means: torch.Tensor | None = None
    common_covariances: torch.Tensor | None = None
    try:
        common_fiber = _new_fiber(
            cameras=scene["cameras"],
            source_view_indices=scene["source_view_indices"],
            source_component_indices=scene["source_component_indices"],
            source_means2d=scene["source_means2d"],
            source_covariances2d=scene["source_covariances2d"],
            initial_depths=scene["initial_depths"],
        )
        with torch.no_grad():
            common_means, common_covariances = common_fiber.means_covariances()
        evidence: dict[str, Any] = {
            "pass": True,
            "initial_depths_sha256": scene["initial_depth_sha256"],
            "common_means_sha256": _tensor_sha256(common_means),
            "common_covariances_sha256": _tensor_sha256(common_covariances),
            "raw_receipt_path": raw_descriptor["path"],
            "raw_receipt_sha256": raw_descriptor["sha256"],
        }
    except Exception as error:
        evidence = {
            "pass": False,
            "initial_depths_sha256": scene["initial_depth_sha256"],
            "raw_receipt_path": raw_descriptor["path"],
            "raw_receipt_sha256": raw_descriptor["sha256"],
            "error_class": type(error).__name__,
            "error_message": str(error),
        }
    roots = (scene["scene_root"], scene["depth_root"]) if domain.label == "official" else ()
    receipt = domain.make_receipt(
        "common_constructor",
        {
            "scene_root": scene["scene_root"],
            "initial_depth_root": scene["depth_root"],
            **evidence,
        },
        root_consumption_status=root_consumption_status,
        roots=roots,
        official_phase="common_constructor" if domain.label == "official" else None,
    )
    path = artifacts_dir / f"scene_{scene['scene_root']}" / "common_constructor.json"
    prepared = _write_receipt_exclusive(path, domain, "common_constructor", receipt)
    descriptor = {
        "path": str(path),
        "sha256": prepared.payload_sha256,
        "recovery_path": str(path.parent / prepared.recovery_name),
        "receipt": receipt,
    }
    return common_fiber, common_means, common_covariances, descriptor


def _write_rank_receipt(
    scene: dict[str, Any],
    rank: dict[str, Any],
    artifacts_dir: Path,
    domain: ReceiptDomain,
    root_consumption_status: str,
) -> dict[str, Any]:
    roots = (scene["scene_root"], scene["depth_root"]) if domain.label == "official" else ()
    receipt = domain.make_receipt(
        "rank_sentinel",
        {
            "scene_root": scene["scene_root"],
            "initial_depth_root": scene["depth_root"],
            "rank": rank,
        },
        root_consumption_status=root_consumption_status,
        roots=roots,
        official_phase="rank_sentinel" if domain.label == "official" else None,
    )
    path = artifacts_dir / f"scene_{scene['scene_root']}" / "rank.json"
    prepared = _write_receipt_exclusive(path, domain, "rank_sentinel", receipt)
    return {
        "path": str(path),
        "sha256": prepared.payload_sha256,
        "recovery_path": str(path.parent / prepared.recovery_name),
        "receipt": receipt,
    }


def _build_and_write_initialization_receipt(
    arm: str,
    scene: dict[str, Any],
    common_means: torch.Tensor,
    common_covariances: torch.Tensor,
    artifacts_dir: Path,
    domain: ReceiptDomain,
    root_consumption_status: str,
) -> tuple[nn.Module | None, dict[str, Any]]:
    model: nn.Module | None = None
    try:
        model = _build_model(arm, scene, common_means, common_covariances)
        parameter_receipts = [
            {
                "name": name,
                "dtype": str(parameter.dtype),
                "device": str(parameter.device),
            }
            for name, parameter in model.named_parameters()
        ]
        cpu_float64 = bool(parameter_receipts) and all(
            item["dtype"] == "torch.float64" and item["device"] == "cpu"
            for item in parameter_receipts
        )
        equivalence = _initialization_equivalence(
            model,
            scene,
            common_means,
            common_covariances,
        )
        evidence: dict[str, Any] = {
            "pass": cpu_float64 and equivalence["pass"],
            "cpu_float64": cpu_float64,
            "parameters": parameter_receipts,
            "initialization_equivalence": equivalence,
        }
    except Exception as error:
        evidence = {
            "pass": False,
            "cpu_float64": False,
            "parameters": [],
            "error_class": type(error).__name__,
            "error_message": str(error),
        }
    roots = (scene["scene_root"], scene["depth_root"]) if domain.label == "official" else ()
    receipt = domain.make_receipt(
        "arm_initialization",
        {
            "scene_root": scene["scene_root"],
            "initial_depth_root": scene["depth_root"],
            "arm": arm,
            **evidence,
        },
        root_consumption_status=root_consumption_status,
        roots=roots,
        official_phase="arm_initialization" if domain.label == "official" else None,
    )
    arm_directory = artifacts_dir / f"scene_{scene['scene_root']}" / arm
    _mkdir_exclusive_durable(arm_directory)
    path = arm_directory / "initialization.json"
    prepared = _write_receipt_exclusive(path, domain, "arm_initialization", receipt)
    descriptor = {
        "path": str(path),
        "sha256": prepared.payload_sha256,
        "recovery_path": str(path.parent / prepared.recovery_name),
        "receipt": receipt,
    }
    return model, descriptor


def _rank_sentinel(scene: dict[str, Any]) -> dict[str, Any]:
    cameras: Sequence[Camera] = scene["cameras"]
    gt_means: torch.Tensor = scene["gt_means"]
    checks: list[dict[str, Any]] = []
    minimum_pair_ratio = math.inf
    minimum_triple_ratio = math.inf
    passed = True
    for primitive_index in range(N_GAUSSIANS):
        mean = gt_means[primitive_index]
        for view_indices in combinations(OPTIMIZATION_VIEWS, 2):
            selected_cameras = [cameras[index] for index in view_indices]
            design = covariance_projection_design(
                mean[None, :].expand(len(selected_cameras), 3),
                selected_cameras,
            )
            singular_values = torch.linalg.svdvals(design)
            tolerance = singular_values[0] * 1e-10
            rank = int((singular_values > tolerance).sum())
            ratio = float(singular_values[4] / singular_values[0])
            minimum_pair_ratio = min(minimum_pair_ratio, ratio)
            check_pass = rank == 5 and ratio >= 1e-8
            passed = passed and check_pass
            checks.append(
                {
                    "primitive_index": primitive_index,
                    "view_indices": list(view_indices),
                    "kind": "pair",
                    "rank": rank,
                    "expected_rank": 5,
                    "tolerance": float(tolerance),
                    "sigma_required_over_sigma_1": ratio,
                    "pass": check_pass,
                }
            )
        for view_indices in combinations(OPTIMIZATION_VIEWS, 3):
            selected_cameras = [cameras[index] for index in view_indices]
            design = covariance_projection_design(
                mean[None, :].expand(len(selected_cameras), 3),
                selected_cameras,
            )
            singular_values = torch.linalg.svdvals(design)
            tolerance = singular_values[0] * 1e-10
            rank = int((singular_values > tolerance).sum())
            ratio = float(singular_values[5] / singular_values[0])
            minimum_triple_ratio = min(minimum_triple_ratio, ratio)
            check_pass = rank == 6 and ratio >= 1e-8
            passed = passed and check_pass
            checks.append(
                {
                    "primitive_index": primitive_index,
                    "view_indices": list(view_indices),
                    "kind": "triple",
                    "rank": rank,
                    "expected_rank": 6,
                    "tolerance": float(tolerance),
                    "sigma_required_over_sigma_1": ratio,
                    "pass": check_pass,
                }
            )
    return {
        "scene_root": scene["scene_root"],
        "rank_tolerance": "sigma_max * 1e-10",
        "minimum_sigma_5_over_sigma_1_pairs": minimum_pair_ratio,
        "minimum_sigma_6_over_sigma_1_triples": minimum_triple_ratio,
        "minimum_required_ratio": 1e-8,
        "checks": checks,
        "pass": passed,
    }


def _new_fiber(
    *,
    cameras: Sequence[Camera],
    source_view_indices: torch.Tensor,
    source_component_indices: torch.Tensor,
    source_means2d: torch.Tensor,
    source_covariances2d: torch.Tensor,
    initial_depths: torch.Tensor,
    depth_lower: float = DEPTH_LOWER,
    depth_upper: float = DEPTH_UPPER,
) -> InverseProjectionFiber:
    return InverseProjectionFiber(
        cameras=cameras,
        source_view_indices=source_view_indices,
        source_component_indices=source_component_indices,
        source_means2d=source_means2d,
        source_covariances2d=source_covariances2d,
        initial_depths=initial_depths,
        depth_lower=depth_lower,
        depth_upper=depth_upper,
        dilation=DILATION,
    )


def _source_residuals(
    predicted_means: torch.Tensor,
    predicted_covariances: torch.Tensor,
    target_means: torch.Tensor,
    target_covariances: torch.Tensor,
) -> tuple[float, float]:
    center = torch.linalg.vector_norm(predicted_means - target_means, dim=-1)
    covariance = torch.linalg.matrix_norm(
        predicted_covariances - target_covariances,
        ord="fro",
        dim=(-2, -1),
    ) / torch.linalg.matrix_norm(target_covariances, ord="fro", dim=(-2, -1))
    return float(center.max().detach()), float(covariance.max().detach())


def _construction_sentinel() -> dict[str, Any]:
    camera = Camera(
        fx=71.0,
        fy=67.0,
        cx=32.0,
        cy=31.0,
        width=64,
        height=64,
        R=torch.eye(3, dtype=torch.float64),
        t=torch.tensor([0.125, -0.25, 0.5], dtype=torch.float64),
    )
    camera.R = camera.R.to(torch.float64)
    camera.t = camera.t.to(torch.float64)
    means2d = torch.tensor([[23.25, 39.5]], dtype=torch.float64)
    covariances2d = torch.tensor(
        [[[5.1, 0.7], [0.7, 2.9]]],
        dtype=torch.float64,
    )
    fiber = _new_fiber(
        cameras=(camera,),
        source_view_indices=torch.zeros(1, dtype=torch.long),
        source_component_indices=torch.zeros(1, dtype=torch.long),
        source_means2d=means2d,
        source_covariances2d=covariances2d,
        initial_depths=torch.tensor([2.2], dtype=torch.float64),
        depth_lower=1.0,
        depth_upper=4.0,
    )
    before_means, before_covariances, _ = fiber.source_projection()
    before_center, before_covariance = _source_residuals(
        before_means,
        before_covariances,
        means2d,
        covariances2d,
    )
    with torch.no_grad():
        fiber.depth_logits.add_(0.23)
        fiber.cross[0] = torch.tensor([0.17, -0.11], dtype=torch.float64)
        fiber.log_ray_scale.add_(0.19)
    after_means, after_covariances, _ = fiber.source_projection()
    after_center, after_covariance = _source_residuals(
        after_means,
        after_covariances,
        means2d,
        covariances2d,
    )
    maximum_center = max(before_center, after_center)
    maximum_covariance = max(before_covariance, after_covariance)
    return {
        "construction": "exact identity rotation, off-axis float64 source",
        "before": {
            "source_center_max_px": before_center,
            "source_covariance_relative_frobenius_max": before_covariance,
        },
        "after_deterministic_parameter_perturbation": {
            "source_center_max_px": after_center,
            "source_covariance_relative_frobenius_max": after_covariance,
        },
        "thresholds": {
            "source_center_max_px": 1e-8,
            "source_covariance_relative_frobenius_max": 1e-8,
        },
        "pass": maximum_center <= 1e-8 and maximum_covariance <= 1e-8,
    }


def _gradient_loss(
    fiber: InverseProjectionFiber,
    target_camera: Camera,
    target_mean: torch.Tensor,
    target_covariance: torch.Tensor,
    fixed_center_metric: torch.Tensor,
) -> torch.Tensor:
    projection = fiber.project(target_camera)
    delta = projection.means2d - target_mean
    solved = torch.linalg.solve(fixed_center_metric, delta.unsqueeze(-1))
    center = (delta.unsqueeze(-2) @ solved).squeeze()
    conic = pairwise_conic_cost(
        projection.covariances2d,
        target_covariance,
    )[0, 0]
    return center + CONIC_WEIGHT * conic


def _gradient_and_duplicate_sentinels() -> tuple[dict[str, Any], dict[str, Any]]:
    camera0 = Camera.look_at(
        torch.tensor([2.1, 0.3, 1.7]),
        torch.zeros(3),
        fov_x_deg=47.0,
        width=64,
        height=64,
    )
    camera1 = Camera.look_at(
        torch.tensor([-1.5, 0.6, 2.0]),
        torch.zeros(3),
        fov_x_deg=51.0,
        width=64,
        height=64,
    )
    gt_mean = torch.tensor([[0.18, -0.13, 0.07]], dtype=torch.float64)
    factor = torch.tensor(
        [[[0.13, 0.0, 0.0], [0.02, 0.08, 0.0], [-0.01, 0.015, 0.055]]],
        dtype=torch.float64,
    )
    gt_covariance = factor @ factor.transpose(-1, -2)
    source = project_covariances_ewa(
        gt_mean,
        gt_covariance,
        camera0,
        dilation=DILATION,
    )
    target = project_covariances_ewa(
        gt_mean,
        gt_covariance,
        camera1,
        dilation=DILATION,
    )
    fiber = _new_fiber(
        cameras=(camera0, camera1),
        source_view_indices=torch.zeros(1, dtype=torch.long),
        source_component_indices=torch.zeros(1, dtype=torch.long),
        source_means2d=source.means2d,
        source_covariances2d=source.covariances2d,
        initial_depths=torch.tensor([1.9], dtype=torch.float64),
    )
    with torch.no_grad():
        fiber.cross[0] = torch.tensor([0.12, -0.08], dtype=torch.float64)
        fiber.log_ray_scale.add_(0.15)
        baseline_projection = fiber.project(camera1)
        fixed_center_metric = 0.5 * (baseline_projection.covariances2d + target.covariances2d)

    loss = _gradient_loss(
        fiber,
        camera1,
        target.means2d,
        target.covariances2d,
        fixed_center_metric,
    )
    loss.backward()
    epsilon = 1e-6
    coordinates: tuple[tuple[str, torch.Tensor, tuple[int, ...]], ...] = (
        ("depth_logit", fiber.depth_logits, (0,)),
        ("cross_0", fiber.cross, (0, 0)),
        ("cross_1", fiber.cross, (0, 1)),
        ("ray_log_scale", fiber.log_ray_scale, (0,)),
    )
    checks: list[dict[str, Any]] = []
    for name, parameter, index in coordinates:
        if parameter.grad is None:
            raise RuntimeError(f"missing autograd gradient for {name}")
        autograd_value = float(parameter.grad[index])
        original = parameter[index].detach().clone()
        with torch.no_grad():
            parameter[index] = original + epsilon
        plus = float(
            _gradient_loss(
                fiber,
                camera1,
                target.means2d,
                target.covariances2d,
                fixed_center_metric,
            ).detach()
        )
        with torch.no_grad():
            parameter[index] = original - epsilon
        minus = float(
            _gradient_loss(
                fiber,
                camera1,
                target.means2d,
                target.covariances2d,
                fixed_center_metric,
            ).detach()
        )
        with torch.no_grad():
            parameter[index] = original
        finite_difference = (plus - minus) / (2.0 * epsilon)
        relative_error = abs(finite_difference - autograd_value) / max(
            1e-8,
            abs(finite_difference),
            abs(autograd_value),
        )
        checks.append(
            {
                "coordinate": name,
                "autograd": autograd_value,
                "finite_difference": finite_difference,
                "relative_error": relative_error,
                "threshold": 2e-4,
                "pass": relative_error <= 2e-4,
            }
        )

    projection = fiber.project(camera1)
    single_cost = pairwise_gaussian_geometry_cost(
        projection.means2d,
        projection.covariances2d,
        target.means2d,
        target.covariances2d,
        include_conic=True,
        conic_weight=CONIC_WEIGHT,
    )
    duplicate_cost = pairwise_gaussian_geometry_cost(
        projection.means2d,
        projection.covariances2d,
        torch.cat([target.means2d, target.means2d], dim=0),
        torch.cat([target.covariances2d, target.covariances2d], dim=0),
        include_conic=True,
        conic_weight=CONIC_WEIGHT,
    )
    single_loss, single_assignment = hard_correspondence_loss(single_cost)
    duplicate_loss, duplicate_assignment = hard_correspondence_loss(duplicate_cost)
    duplicate_pass = bool(torch.equal(single_loss, duplicate_loss)) and int(
        duplicate_assignment[0]
    ) == int(single_assignment[0])
    return (
        {
            "epsilon": epsilon,
            "construction": "off-axis anisotropic two-camera component geometry",
            "checks": checks,
            "maximum_relative_error": max(check["relative_error"] for check in checks),
            "pass": all(check["pass"] for check in checks),
        },
        {
            "single_loss": float(single_loss.detach()),
            "duplicate_loss": float(duplicate_loss.detach()),
            "bit_exact": bool(torch.equal(single_loss, duplicate_loss)),
            "single_assignment": int(single_assignment[0]),
            "duplicate_assignment": int(duplicate_assignment[0]),
            "pass": duplicate_pass,
        },
    )


def _geometry_state(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, InverseProjectionFiber):
        return model.means_covariances()
    if isinstance(model, FreeGaussianGeometry):
        return model.means, model.covariances()
    raise TypeError(type(model).__name__)


def _project_state(
    model: nn.Module,
    cameras: Sequence[Camera],
) -> tuple[torch.Tensor, torch.Tensor, list[Any]]:
    means, covariances = _geometry_state(model)
    projections = [
        project_covariances_ewa(
            means,
            covariances,
            camera,
            dilation=DILATION,
        )
        for camera in cameras
    ]
    return means, covariances, projections


def _arm_uses_conic(arm: str) -> bool:
    return arm not in {"free_center", "fiber_center"}


def _objective(model: nn.Module, arm: str, scene: dict[str, Any]) -> torch.Tensor:
    source_views: torch.Tensor = scene["source_view_indices"]
    source_components: torch.Tensor = scene["source_component_indices"]
    _, _, projections = _project_state(model, scene["cameras"])
    per_hypothesis_sum = scene["source_means2d"].new_zeros(N_HYPOTHESES)
    for target_view in OPTIMIZATION_VIEWS:
        rows = (source_views != target_view).nonzero(as_tuple=True)[0]
        projection = projections[target_view]
        cost = pairwise_gaussian_geometry_cost(
            projection.means2d[rows],
            projection.covariances2d[rows],
            scene["target_means2d"][target_view],
            scene["target_covariances2d"][target_view],
            include_conic=_arm_uses_conic(arm),
            conic_weight=CONIC_WEIGHT,
        )
        if arm in {"free_center", "free_conic", "fiber_center", "fiber_conic"}:
            selected = torch.min(cost, dim=1).values
        elif arm == "oracle":
            selected = cost.gather(1, source_components[rows, None]).squeeze(1)
        elif arm == "shuffled":
            shuffled = (source_components[rows] + 1 + target_view) % N_GAUSSIANS
            selected = cost.gather(1, shuffled[:, None]).squeeze(1)
        else:
            raise ValueError(arm)
        per_hypothesis_sum = per_hypothesis_sum.index_add(0, rows, selected)
    non_source = (per_hypothesis_sum / 3.0).mean()
    if not isinstance(model, FreeGaussianGeometry):
        return non_source

    source_costs = scene["source_means2d"].new_empty(N_HYPOTHESES)
    for source_view in OPTIMIZATION_VIEWS:
        rows = (source_views == source_view).nonzero(as_tuple=True)[0]
        projection = projections[source_view]
        cost = pairwise_gaussian_geometry_cost(
            projection.means2d[rows],
            projection.covariances2d[rows],
            scene["target_means2d"][source_view],
            scene["target_covariances2d"][source_view],
            include_conic=True,
            conic_weight=CONIC_WEIGHT,
        )
        source_costs[rows] = cost.gather(1, source_components[rows, None]).squeeze(1)
    return non_source + FREE_SOURCE_WEIGHT * source_costs.mean()


def _parameter_norms(model: nn.Module) -> dict[str, Any]:
    parameter_squared = 0.0
    gradient_squared = 0.0
    families: dict[str, Any] = {}
    all_gradients_finite = True
    missing_gradient_families: list[str] = []
    for name, parameter in model.named_parameters():
        parameter_l2 = float(torch.linalg.vector_norm(parameter.detach()))
        parameter_squared += parameter_l2 * parameter_l2
        if parameter.grad is None:
            gradient_l2 = None
            gradient_max_abs = None
            missing_gradient_families.append(name)
        else:
            gradient_l2 = float(torch.linalg.vector_norm(parameter.grad.detach()))
            gradient_max_abs = float(parameter.grad.detach().abs().max())
            gradient_squared += gradient_l2 * gradient_l2
            all_gradients_finite = all_gradients_finite and bool(
                torch.isfinite(parameter.grad).all()
            )
        families[name] = {
            "parameter_l2": parameter_l2,
            "parameter_max_abs": float(parameter.detach().abs().max()),
            "gradient_l2": gradient_l2,
            "gradient_max_abs": gradient_max_abs,
        }
    return {
        "families": families,
        "parameter_l2": math.sqrt(parameter_squared),
        "gradient_l2": math.sqrt(gradient_squared),
        "all_gradients_finite": all_gradients_finite,
        "missing_gradient_families": missing_gradient_families,
    }


def _source_projection(
    model: nn.Module,
    scene: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(model, InverseProjectionFiber):
        return model.source_projection()
    means, covariances = _geometry_state(model)
    projected_means = means.new_empty((N_HYPOTHESES, 2))
    projected_covariances = covariances.new_empty((N_HYPOTHESES, 2, 2))
    projected_depths = means.new_empty(N_HYPOTHESES)
    source_views: torch.Tensor = scene["source_view_indices"]
    for source_view in OPTIMIZATION_VIEWS:
        rows = (source_views == source_view).nonzero(as_tuple=True)[0]
        projection = project_covariances_ewa(
            means[rows],
            covariances[rows],
            scene["cameras"][source_view],
            dilation=DILATION,
        )
        projected_means[rows] = projection.means2d
        projected_covariances[rows] = projection.covariances2d
        projected_depths[rows] = projection.depth
    return projected_means, projected_covariances, projected_depths


def _checkpoint(
    model: nn.Module,
    arm: str,
    scene: dict[str, Any],
    step: int,
    loss: torch.Tensor,
) -> dict[str, Any]:
    with torch.no_grad():
        means, covariances, projections = _project_state(model, scene["cameras"])
        covariance_eigenvalues = torch.linalg.eigvalsh(covariances)
        source_views: torch.Tensor = scene["source_view_indices"]
        used_depths: list[torch.Tensor] = []
        for view_index in OPTIMIZATION_VIEWS:
            if isinstance(model, FreeGaussianGeometry):
                rows = torch.arange(N_HYPOTHESES, dtype=torch.long)
            else:
                rows = (source_views != view_index).nonzero(as_tuple=True)[0]
            used_depths.append(projections[view_index].depth[rows])
        all_used_depths = torch.cat(used_depths)
        if isinstance(model, InverseProjectionFiber):
            fiber_depths = model.depths()
            depths_in_bounds = bool(
                ((fiber_depths >= DEPTH_LOWER) & (fiber_depths <= DEPTH_UPPER)).all()
            )
            fiber_depth_min: float | None = float(fiber_depths.min())
            fiber_depth_max: float | None = float(fiber_depths.max())
        else:
            depths_in_bounds = True
            fiber_depth_min = None
            fiber_depth_max = None
        parameters_finite = all(
            bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
        )
        invariants = {
            "loss_finite": bool(torch.isfinite(loss)),
            "parameters_finite": parameters_finite,
            "means_finite": bool(torch.isfinite(means).all()),
            "covariances_finite": bool(torch.isfinite(covariances).all()),
            "covariance_eigenvalues_positive": bool((covariance_eigenvalues > 0).all()),
            "minimum_covariance_eigenvalue": float(covariance_eigenvalues.min()),
            "fiber_depths_in_closed_bounds": depths_in_bounds,
            "fiber_depth_min": fiber_depth_min,
            "fiber_depth_max": fiber_depth_max,
            "all_loss_projection_depths_positive": bool((all_used_depths > 0).all()),
            "minimum_loss_projection_depth": float(all_used_depths.min()),
        }
    norms = _parameter_norms(model)
    passed = all(
        (
            invariants["loss_finite"],
            invariants["parameters_finite"],
            invariants["means_finite"],
            invariants["covariances_finite"],
            invariants["covariance_eigenvalues_positive"],
            invariants["fiber_depths_in_closed_bounds"],
            invariants["all_loss_projection_depths_positive"],
            norms["all_gradients_finite"],
        )
    )
    return {
        "step": step,
        "loss": float(loss.detach()),
        "invariants": invariants,
        "norms": norms,
        "pass": passed,
        "arm": arm,
    }


def _materialize_gaussians(model: nn.Module) -> Gaussians3D:
    means, covariances = _geometry_state(model)
    colors = means.new_full((means.shape[0], 3), 0.5)
    opacity = means.new_full((means.shape[0],), 0.5)
    return Gaussians3D.from_means_covs(
        means,
        covariances,
        colors,
        opacity,
    )


def _evaluation_metrics(model: nn.Module, scene: dict[str, Any]) -> dict[str, Any]:
    with torch.no_grad():
        means, covariances, projections = _project_state(model, scene["cameras"])
        source_views: torch.Tensor = scene["source_view_indices"]
        source_components: torch.Tensor = scene["source_component_indices"]

        train_correct = 0
        heldout_correct = 0
        track_assignments: list[list[int]] = [[] for _ in range(N_HYPOTHESES)]
        projected_costs: dict[str, dict[str, list[torch.Tensor]]] = {
            "train": {"center": [], "conic": []},
            "heldout": {"center": [], "conic": []},
        }
        for view_index in (*OPTIMIZATION_VIEWS, *HELDOUT_VIEWS):
            if view_index in OPTIMIZATION_VIEWS:
                rows = (source_views != view_index).nonzero(as_tuple=True)[0]
                split = "train"
            else:
                rows = torch.arange(N_HYPOTHESES, dtype=torch.long)
                split = "heldout"
            projection = projections[view_index]
            center_cost = pairwise_center_cost(
                projection.means2d[rows],
                projection.covariances2d[rows],
                scene["target_means2d"][view_index],
                scene["target_covariances2d"][view_index],
            )
            conic_cost = pairwise_conic_cost(
                projection.covariances2d[rows],
                scene["target_covariances2d"][view_index],
            )
            geometry_cost = center_cost + CONIC_WEIGHT * conic_cost
            assignments = torch.min(geometry_cost, dim=1).indices
            correct = assignments == source_components[rows]
            if split == "train":
                train_correct += int(correct.sum())
            else:
                heldout_correct += int(correct.sum())
            for row, assignment in zip(rows.tolist(), assignments.tolist(), strict=True):
                track_assignments[row].append(assignment)
            target_ids = source_components[rows, None]
            projected_costs[split]["center"].append(center_cost.gather(1, target_ids).squeeze(1))
            projected_costs[split]["conic"].append(conic_cost.gather(1, target_ids).squeeze(1))

        if sum(len(track) for track in track_assignments) != N_HYPOTHESES * 5:
            raise RuntimeError("track evaluation did not produce exactly five associations")
        correct_track_count = 0
        consistent_track_count = 0
        for hypothesis_index, track in enumerate(track_assignments):
            if len(track) != 5:
                raise RuntimeError("a hypothesis does not have five evaluated associations")
            gt_id = int(source_components[hypothesis_index])
            correct_track_count += int(all(assignment == gt_id for assignment in track))
            consistent_track_count += int(all(assignment == track[0] for assignment in track))

        source_means, source_covariances, source_depths = _source_projection(model, scene)
        source_center_max, source_covariance_max = _source_residuals(
            source_means,
            source_covariances,
            scene["source_means2d"],
            scene["source_covariances2d"],
        )
        gt_means = scene["gt_means"][source_components]
        gt_covariances = scene["gt_covariances"][source_components]
        center_distances = torch.linalg.vector_norm(means - gt_means, dim=-1)
        covariance_squared = spd_affine_invariant_squared(covariances, gt_covariances)
        covariance_distance = covariance_squared.sqrt()

        train_center = torch.cat(projected_costs["train"]["center"])
        train_conic = torch.cat(projected_costs["train"]["conic"])
        heldout_center = torch.cat(projected_costs["heldout"]["center"])
        heldout_conic = torch.cat(projected_costs["heldout"]["conic"])
        if train_center.numel() != TRAIN_ASSOCIATION_DENOMINATOR:
            raise RuntimeError("incorrect train projected-cost denominator")
        if heldout_center.numel() != HELDOUT_ASSOCIATION_DENOMINATOR:
            raise RuntimeError("incorrect heldout projected-cost denominator")

        lower_distance = source_depths - DEPTH_LOWER
        upper_distance = DEPTH_UPPER - source_depths
        bound_distance = torch.minimum(
            (source_depths - DEPTH_LOWER).abs(),
            (source_depths - DEPTH_UPPER).abs(),
        )
        covariance_eigenvalues = torch.linalg.eigvalsh(covariances)
        covariance_condition = covariance_eigenvalues[:, -1] / covariance_eigenvalues[:, 0]
        return {
            "source_center_max_px": source_center_max,
            "source_covariance_relative_frobenius_max": source_covariance_max,
            "gt_center_distance_median_world": float(torch.quantile(center_distances, 0.5)),
            "gt_center_distance_p90_world": float(torch.quantile(center_distances, 0.9)),
            "gt_covariance_affine_invariant_median": float(
                torch.quantile(covariance_distance, 0.5)
            ),
            "gt_covariance_affine_invariant_squared_median": float(
                torch.quantile(covariance_squared, 0.5)
            ),
            "train_association_accuracy": train_correct / TRAIN_ASSOCIATION_DENOMINATOR,
            "heldout_association_accuracy": heldout_correct / HELDOUT_ASSOCIATION_DENOMINATOR,
            "correct_track_fraction": correct_track_count / TRACK_DENOMINATOR,
            "consistent_track_fraction": consistent_track_count / TRACK_DENOMINATOR,
            "train_gt_projected_center_cost": float(train_center.mean()),
            "train_gt_projected_conic_cost": float(train_conic.mean()),
            "train_gt_projected_geometry_cost": float(
                (train_center + CONIC_WEIGHT * train_conic).mean()
            ),
            "heldout_gt_projected_center_cost": float(heldout_center.mean()),
            "heldout_gt_projected_conic_cost": float(heldout_conic.mean()),
            "heldout_gt_projected_geometry_cost": float(
                (heldout_center + CONIC_WEIGHT * heldout_conic).mean()
            ),
            "depth_bound_margin": float(torch.minimum(lower_distance, upper_distance).min()),
            "depth_fraction_within_1e-4_of_bound": float((bound_distance <= 1e-4).double().mean()),
            "depth_min": float(source_depths.min()),
            "depth_max": float(source_depths.max()),
            "covariance_condition_number_p50": float(torch.quantile(covariance_condition, 0.5)),
            "covariance_condition_number_p95": float(torch.quantile(covariance_condition, 0.95)),
            "covariance_condition_number_max": float(covariance_condition.max()),
            "denominators": {
                "train_association": TRAIN_ASSOCIATION_DENOMINATOR,
                "heldout_association": HELDOUT_ASSOCIATION_DENOMINATOR,
                "correct_track": TRACK_DENOMINATOR,
                "consistent_track": TRACK_DENOMINATOR,
                "train_projected_cost": TRAIN_ASSOCIATION_DENOMINATOR,
                "heldout_projected_cost": HELDOUT_ASSOCIATION_DENOMINATOR,
            },
        }


def _train_arm(
    model: nn.Module,
    arm: str,
    scene: dict[str, Any],
    artifact_directory: Path,
    config_sha256: str,
    provenance_sha256: str,
    initialization_descriptor: dict[str, Any],
    *,
    domain: ReceiptDomain,
    root_consumption_status: str,
) -> dict[str, Any]:
    if not artifact_directory.is_dir() or artifact_directory.is_symlink():
        raise FileNotFoundError(
            "arm directory and durable initialization receipt must exist before training"
        )
    _validate_durable_receipt_descriptor(
        initialization_descriptor,
        domain=domain,
        kind="arm_initialization",
        expected_parent=artifact_directory,
    )
    initialization_receipt = initialization_descriptor["receipt"]
    if initialization_receipt.get("pass") is not True:
        raise RuntimeError("optimizer creation is forbidden after failed initialization")
    initial_means, initial_covariances = _geometry_state(model)
    initial_geometry_hashes = {
        "means_sha256": _tensor_sha256(initial_means),
        "covariances_sha256": _tensor_sha256(initial_covariances),
    }
    initial_path = artifact_directory / "gaussians_init.ply"
    initial_ply_sha256 = _save_ply_exclusive(
        initial_path,
        _materialize_gaussians(model),
        domain=domain,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    losses: list[float] = []
    checkpoints: list[dict[str, Any]] = []
    rss_before = _peak_rss_bytes()
    started = time.perf_counter()
    for step in range(UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _objective(model, arm, scene)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"{arm} produced a non-finite objective at step {step}")
        loss.backward()
        losses.append(float(loss.detach()))
        if step % CHECKPOINT_INTERVAL == 0:
            checkpoint = _checkpoint(model, arm, scene, step, loss)
            checkpoints.append(checkpoint)
            if not checkpoint["pass"]:
                raise RuntimeError(f"{arm} failed a validity sentinel at step {step}")
        if step == UPDATES:
            break
        optimizer.step()
    wall_seconds = time.perf_counter() - started
    rss_after = _peak_rss_bytes()
    if len(losses) != UPDATES + 1:
        raise RuntimeError("full loss trajectory does not contain 401 states")
    if [checkpoint["step"] for checkpoint in checkpoints] != list(
        range(0, UPDATES + 1, CHECKPOINT_INTERVAL)
    ):
        raise RuntimeError("checkpoint schedule differs from the frozen protocol")

    metrics = _evaluation_metrics(model, scene)
    final_means, final_covariances = _geometry_state(model)
    final_geometry_hashes = {
        "means_sha256": _tensor_sha256(final_means),
        "covariances_sha256": _tensor_sha256(final_covariances),
    }
    final_path = artifact_directory / "gaussians.ply"
    final_ply_sha256 = _save_ply_exclusive(
        final_path,
        _materialize_gaussians(model),
        domain=domain,
    )
    payload: dict[str, Any] = {
        "scene_root": scene["scene_root"],
        "initial_depth_root": scene["depth_root"],
        "arm": arm,
        "geometry": "free" if arm.startswith("free_") else "exact_source_fiber",
        "non_source_objective": (
            "latent_center"
            if arm in {"free_center", "fiber_center"}
            else "latent_geometry"
            if arm in {"free_conic", "fiber_conic"}
            else "oracle_geometry"
            if arm == "oracle"
            else "shuffled_geometry"
        ),
        "config_sha256": config_sha256,
        "provenance_sha256": provenance_sha256,
        "paired_initial_depth_sha256": scene["initial_depth_sha256"],
        "initialization_receipt": initialization_descriptor,
        "initial_geometry": initial_geometry_hashes,
        "final_geometry": final_geometry_hashes,
        "updates": UPDATES,
        "objective_evaluations": len(losses),
        "loss_trajectory": losses,
        "checkpoints": checkpoints,
        "metrics": metrics,
        "wall_time_seconds": wall_seconds,
        "process_peak_rss_bytes_so_far_before": rss_before,
        "process_peak_rss_bytes_so_far_after": rss_after,
        "process_peak_rss_bytes_so_far": rss_after,
        "artifacts": {
            "initial_ply": str(initial_path),
            "initial_ply_sha256": initial_ply_sha256,
            "final_ply": str(final_path),
            "final_ply_sha256": final_ply_sha256,
        },
    }
    roots = (scene["scene_root"], scene["depth_root"]) if domain.label == "official" else ()
    record = domain.make_receipt(
        "arm_result",
        payload,
        root_consumption_status=root_consumption_status,
        roots=roots,
        official_phase="arm_result" if domain.label == "official" else None,
    )
    receipt_path = artifact_directory / "result.json"
    record["artifacts"]["result_json"] = str(receipt_path)
    # The self hash cannot be embedded in the bytes it hashes.  Store the durable descriptor in
    # the aggregate; the arm receipt itself contains the public result path only.
    prepared = _write_receipt_exclusive(receipt_path, domain, "arm_result", record)
    record["artifacts"]["result_json_sha256"] = prepared.payload_sha256
    record["artifacts"]["result_json_recovery"] = str(receipt_path.parent / prepared.recovery_name)
    return record


def _aggregate(
    records: list[dict[str, Any]],
    scene_roots: tuple[int, ...],
) -> dict[str, Any]:
    by_arm = {
        arm: sorted(
            (record for record in records if record["arm"] == arm),
            key=lambda record: record["scene_root"],
        )
        for arm in ARMS
    }
    aggregate: dict[str, Any] = {}
    for arm, arm_records in by_arm.items():
        if len(arm_records) != len(scene_roots):
            raise RuntimeError(f"{arm} does not have all three replicates")
        metrics: dict[str, Any] = {}
        for key in ERROR_METRICS:
            raw = [float(record["metrics"][key]) for record in arm_records]
            metrics[key] = {
                "raw": raw,
                "arithmetic_mean": sum(raw) / len(raw),
                "geometric_mean_floor_1e-12": math.exp(
                    sum(math.log(max(value, 1e-12)) for value in raw) / len(raw)
                ),
            }
        for key in FRACTION_METRICS:
            raw = [float(record["metrics"][key]) for record in arm_records]
            metrics[key] = {
                "raw": raw,
                "arithmetic_mean": sum(raw) / len(raw),
            }
        for key in (
            "depth_bound_margin",
            "depth_min",
            "depth_max",
            "covariance_condition_number_p50",
            "covariance_condition_number_p95",
            "covariance_condition_number_max",
            "wall_time_seconds",
            "process_peak_rss_bytes_so_far",
        ):
            if key in {"wall_time_seconds", "process_peak_rss_bytes_so_far"}:
                raw = [float(record[key]) for record in arm_records]
            else:
                raw = [float(record["metrics"][key]) for record in arm_records]
            metrics[key] = {
                "raw": raw,
                "arithmetic_mean": sum(raw) / len(raw),
            }
        aggregate[arm] = {
            "replicate_roots": [record["scene_root"] for record in arm_records],
            "metrics": metrics,
        }
    return aggregate


def _scientific_gates(
    records: list[dict[str, Any]],
    sentinels: dict[str, Any],
    scene_roots: tuple[int, ...],
) -> dict[str, Any]:
    lookup = {(record["scene_root"], record["arm"]): record["metrics"] for record in records}
    all_checkpoint_sentinels = all(
        checkpoint["pass"] for record in records for checkpoint in record["checkpoints"]
    )
    gate1_pass = bool(sentinels["pass"]) and all_checkpoint_sentinels
    gate1 = {
        "status": "PASS" if gate1_pass else "FAIL",
        "development_and_rank_sentinels_pass": bool(sentinels["pass"]),
        "all_checkpoint_sentinels_pass": all_checkpoint_sentinels,
    }

    gate2_replicates: list[dict[str, Any]] = []
    for root in scene_roots:
        metrics = lookup[(root, "fiber_conic")]
        checks = {
            "source_center_max_px_le_1e-6": metrics["source_center_max_px"] <= 1e-6,
            "source_covariance_relative_max_le_1e-5": (
                metrics["source_covariance_relative_frobenius_max"] <= 1e-5
            ),
            "train_association_accuracy_ge_0.95": (metrics["train_association_accuracy"] >= 0.95),
            "heldout_association_accuracy_ge_0.95": (
                metrics["heldout_association_accuracy"] >= 0.95
            ),
            "correct_track_fraction_ge_0.90": metrics["correct_track_fraction"] >= 0.90,
            "gt_center_p90_le_0.05": metrics["gt_center_distance_p90_world"] <= 0.05,
        }
        gate2_replicates.append(
            {
                "scene_root": root,
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    gate2_pass = all(replicate["pass"] for replicate in gate2_replicates)
    gate2 = {
        "status": "PASS" if gate2_pass else "FAIL",
        "replicates": gate2_replicates,
    }

    gate3_replicates: list[dict[str, Any]] = []
    for root in scene_roots:
        fiber_p90 = lookup[(root, "fiber_conic")]["gt_center_distance_p90_world"]
        oracle_p90 = lookup[(root, "oracle")]["gt_center_distance_p90_world"]
        threshold = max(0.01, 1.25 * oracle_p90)
        gate3_replicates.append(
            {
                "scene_root": root,
                "fiber_conic_center_p90": fiber_p90,
                "oracle_center_p90": oracle_p90,
                "threshold": threshold,
                "pass": fiber_p90 <= threshold,
            }
        )
    gate3_pass = all(replicate["pass"] for replicate in gate3_replicates)
    gate3 = {
        "status": "PASS" if gate3_pass else "FAIL",
        "replicates": gate3_replicates,
    }

    fiber_p90_mean = sum(
        lookup[(root, "fiber_conic")]["gt_center_distance_p90_world"] for root in scene_roots
    ) / len(scene_roots)
    shuffled_p90_mean = sum(
        lookup[(root, "shuffled")]["gt_center_distance_p90_world"] for root in scene_roots
    ) / len(scene_roots)
    fiber_heldout_accuracy_mean = sum(
        lookup[(root, "fiber_conic")]["heldout_association_accuracy"] for root in scene_roots
    ) / len(scene_roots)
    shuffled_heldout_accuracy_mean = sum(
        lookup[(root, "shuffled")]["heldout_association_accuracy"] for root in scene_roots
    ) / len(scene_roots)
    relative_center_improvement = (shuffled_p90_mean - fiber_p90_mean) / max(
        shuffled_p90_mean, 0.01
    )
    heldout_accuracy_improvement = fiber_heldout_accuracy_mean - shuffled_heldout_accuracy_mean
    gate4_pass = relative_center_improvement >= 0.50 and heldout_accuracy_improvement >= 0.50
    gate4 = {
        "status": "PASS" if gate4_pass else "FAIL",
        "fiber_conic_center_p90_arithmetic_mean": fiber_p90_mean,
        "shuffled_center_p90_arithmetic_mean": shuffled_p90_mean,
        "relative_center_improvement": relative_center_improvement,
        "relative_center_improvement_threshold": 0.50,
        "fiber_conic_heldout_accuracy_arithmetic_mean": fiber_heldout_accuracy_mean,
        "shuffled_heldout_accuracy_arithmetic_mean": shuffled_heldout_accuracy_mean,
        "heldout_accuracy_improvement": heldout_accuracy_improvement,
        "heldout_accuracy_improvement_threshold": 0.50,
    }

    attribution_replicates: list[dict[str, Any]] = []
    for root in scene_roots:
        free_metrics = lookup[(root, "free_conic")]
        checks = {
            "free_source_center_max_px_le_0.05": (free_metrics["source_center_max_px"] <= 0.05),
            "free_source_covariance_relative_max_le_0.01": (
                free_metrics["source_covariance_relative_frobenius_max"] <= 0.01
            ),
        }
        attribution_replicates.append(
            {
                "scene_root": root,
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    attribution_valid = all(item["pass"] for item in attribution_replicates)
    gate6 = {
        "status": "PASS" if attribution_valid else "FAIL",
        "interpretation_if_failed": "soft-source failure; fiber/free attribution invalid",
        "replicates": attribution_replicates,
    }

    gate5_replicates: list[dict[str, Any]] = []
    for root in scene_roots:
        fiber_metrics = lookup[(root, "fiber_conic")]
        free_metrics = lookup[(root, "free_conic")]
        center_threshold = 1.05 * free_metrics["gt_center_distance_p90_world"] + 0.002
        heldout_threshold = 1.05 * free_metrics["heldout_gt_projected_geometry_cost"] + 0.01
        checks = {
            "fiber_center_p90_noninferior": (
                fiber_metrics["gt_center_distance_p90_world"] <= center_threshold
            ),
            "fiber_heldout_cost_noninferior": (
                fiber_metrics["heldout_gt_projected_geometry_cost"] <= heldout_threshold
            ),
        }
        gate5_replicates.append(
            {
                "scene_root": root,
                "fiber_center_p90": fiber_metrics["gt_center_distance_p90_world"],
                "free_center_p90": free_metrics["gt_center_distance_p90_world"],
                "center_threshold": center_threshold,
                "fiber_heldout_cost": fiber_metrics["heldout_gt_projected_geometry_cost"],
                "free_heldout_cost": free_metrics["heldout_gt_projected_geometry_cost"],
                "heldout_cost_threshold": heldout_threshold,
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    gate5_inequalities_pass = all(item["pass"] for item in gate5_replicates)
    if not attribution_valid:
        gate5_status = "UNINTERPRETABLE"
    else:
        gate5_status = "PASS" if gate5_inequalities_pass else "FAIL"
    gate5 = {
        "status": gate5_status,
        "attribution_valid": attribution_valid,
        "replicates": gate5_replicates,
    }

    overall_pass = all(
        (
            gate1_pass,
            gate2_pass,
            gate3_pass,
            gate4_pass,
            gate5_status == "PASS",
            attribution_valid,
        )
    )
    return {
        "gate_1_sentinels": gate1,
        "gate_2_absolute_fiber": gate2,
        "gate_3_oracle_proximity": gate3,
        "gate_4_shuffled_separation": gate4,
        "gate_5_free_noninferiority": gate5,
        "gate_6_free_attribution": gate6,
        "overall_status": "PASS" if overall_pass else "FAIL",
    }


def _build_model(
    arm: str,
    scene: dict[str, Any],
    common_initial_means: torch.Tensor,
    common_initial_covariances: torch.Tensor,
) -> nn.Module:
    if arm.startswith("free_"):
        return FreeGaussianGeometry(
            common_initial_means.detach().clone(),
            common_initial_covariances.detach().clone(),
        )
    return _new_fiber(
        cameras=scene["cameras"],
        source_view_indices=scene["source_view_indices"].detach().clone(),
        source_component_indices=scene["source_component_indices"].detach().clone(),
        source_means2d=scene["source_means2d"].detach().clone(),
        source_covariances2d=scene["source_covariances2d"].detach().clone(),
        initial_depths=scene["initial_depths"].detach().clone(),
    )


def _initialization_equivalence(
    model: nn.Module,
    scene: dict[str, Any],
    common_means: torch.Tensor,
    common_covariances: torch.Tensor,
) -> dict[str, Any]:
    require_byte_identity = isinstance(model, InverseProjectionFiber)
    with torch.no_grad():
        realized_means, realized_covariances = _geometry_state(model)
        mean_maximum_absolute = float((realized_means - common_means).abs().max())
        covariance_relative = torch.linalg.matrix_norm(
            realized_covariances - common_covariances,
            ord="fro",
            dim=(-2, -1),
        ) / torch.linalg.matrix_norm(common_covariances, ord="fro", dim=(-2, -1))
        covariance_relative_maximum = float(covariance_relative.max())
        source_means, source_covariances, _ = _source_projection(model, scene)
        source_center_maximum = float(
            torch.linalg.vector_norm(
                source_means - scene["source_means2d"],
                dim=-1,
            ).max()
        )
        source_covariance_relative = torch.linalg.matrix_norm(
            source_covariances - scene["source_covariances2d"],
            ord="fro",
            dim=(-2, -1),
        ) / torch.linalg.matrix_norm(
            scene["source_covariances2d"],
            ord="fro",
            dim=(-2, -1),
        )
        source_covariance_relative_maximum = float(source_covariance_relative.max())
    constructor_mean_sha256 = _tensor_sha256(common_means)
    constructor_covariance_sha256 = _tensor_sha256(common_covariances)
    realized_mean_sha256 = _tensor_sha256(realized_means)
    realized_covariance_sha256 = _tensor_sha256(realized_covariances)
    checks: dict[str, bool] = {
        "realized_mean_maximum_absolute_eq_0": mean_maximum_absolute == 0.0,
        "realized_covariance_relative_maximum_le_1e-14": (covariance_relative_maximum <= 1e-14),
        "source_center_maximum_px_le_1e-10": source_center_maximum <= 1e-10,
        "source_covariance_relative_maximum_le_1e-12": (
            source_covariance_relative_maximum <= 1e-12
        ),
    }
    if require_byte_identity:
        checks["fiber_realized_means_byte_identical"] = (
            realized_mean_sha256 == constructor_mean_sha256
        )
        checks["fiber_realized_covariances_byte_identical"] = (
            realized_covariance_sha256 == constructor_covariance_sha256
        )
    return {
        "contract": "fiber_byte_identity"
        if require_byte_identity
        else "free_numerical_equivalence",
        "constructor_input_means_sha256": constructor_mean_sha256,
        "constructor_input_covariances_sha256": constructor_covariance_sha256,
        "realized_means_sha256": realized_mean_sha256,
        "realized_covariances_sha256": realized_covariance_sha256,
        "realized_mean_maximum_absolute": mean_maximum_absolute,
        "realized_covariance_relative_frobenius_maximum": covariance_relative_maximum,
        "source_center_maximum_px": source_center_maximum,
        "source_covariance_relative_frobenius_maximum": (source_covariance_relative_maximum),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _require_combined_sentinels(sentinels: dict[str, Any]) -> None:
    if sentinels.get("pass") is not True:
        raise RuntimeError("a combined validity sentinel failed")


def _combined_sentinels_pass(sentinels: dict[str, Any]) -> bool:
    """Combine every frozen validity sentinel without silently dropping a family."""

    ranks = sentinels.get("rank")
    return bool(
        isinstance(ranks, list)
        and ranks
        and sentinels.get("construction", {}).get("pass") is True
        and sentinels.get("finite_difference", {}).get("pass") is True
        and sentinels.get("duplicate_hard_min", {}).get("pass") is True
        and all(rank.get("pass") is True for rank in ranks)
        and sentinels.get("checkpoint_count") == sentinels.get("checkpoint_expected_count")
        and sentinels.get("initialization_equivalence_pass") is True
        and sentinels.get("checkpoint_sentinels_pass") is True
    )


def _verification_forbidden_manifest(spec: ProtocolSpec) -> list[dict[str, Any]]:
    needles = (
        spec.official_domain.namespace,
        spec.official_domain.schema_family,
        *(str(root) for root in spec.scene_roots + spec.initial_depth_roots),
        *spec.official_domain.permitted_root_consumption_statuses,
    )
    if len(needles) != 11 or len(set(needles)) != len(needles):
        raise RuntimeError("verification requires exactly eleven unique forbidden literals")
    return [
        {
            "index": index,
            "length_bytes": len(needle.encode("utf-8")),
            "sha256": _sha256_bytes(needle.encode("utf-8")),
        }
        for index, needle in enumerate(needles)
    ]


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verification_environment(base: Path) -> dict[str, str]:
    """Return the complete, fixed environment used by verification subprocesses."""

    return {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": "",
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PATH": os.defpath,
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "src"))),
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUTF8": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "TMPDIR": str(base.parent),
        "TZ": "UTC",
    }


def _verification_pytest_args(
    spec: ProtocolSpec,
    base: Path,
    *,
    collect_only: bool,
) -> list[str]:
    """Return the exact pytest arguments, excluding the absolute interpreter path."""

    args = [
        "-m",
        "pytest",
        "-q",
        "-c",
        os.devnull,
        "--noconftest",
        f"--rootdir={ROOT}",
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
    ]
    if collect_only:
        args.append("--collect-only")
    args.extend(path.as_posix() for path in spec.verification_test_paths)
    if not collect_only:
        args.append(f"--basetemp={base}")
    return args


def _validate_collected_nodeids(spec: ProtocolSpec, nodeids: Any) -> tuple[str, ...]:
    if not isinstance(nodeids, list) or not nodeids:
        raise RuntimeError("verification receipt has no collected node-id manifest")
    if any(not isinstance(nodeid, str) or not nodeid for nodeid in nodeids):
        raise RuntimeError("verification receipt has a malformed collected node id")
    values = tuple(nodeids)
    if len(set(values)) != len(values):
        raise RuntimeError("verification receipt has duplicate collected node ids")
    prefixes = tuple(f"{path.as_posix()}::" for path in spec.verification_test_paths)
    represented = set()
    for nodeid in values:
        matches = [prefix for prefix in prefixes if nodeid.startswith(prefix)]
        if len(matches) != 1:
            raise RuntimeError("verification receipt collected a node outside the frozen files")
        represented.add(matches[0])
    if represented != set(prefixes):
        raise RuntimeError("verification receipt omitted a frozen test file")
    return values


def _validate_verification_receipt(spec: ProtocolSpec, verification: dict[str, Any]) -> None:
    spec.development_domain.validate_receipt("verification", verification)
    base = spec.verification_base.expanduser().resolve()
    command = verification.get("focused_pytest_command")
    collection_command = verification.get("collection_command")
    expected_command_tail = _verification_pytest_args(spec, base, collect_only=False)
    expected_collection_tail = _verification_pytest_args(spec, base, collect_only=True)
    if not isinstance(command, list) or len(command) < 2:
        raise RuntimeError("verification receipt has no exact pytest command")
    if not isinstance(collection_command, list) or len(collection_command) < 2:
        raise RuntimeError("verification receipt has no exact collection command")
    executable = Path(command[0]) if isinstance(command[0], str) else Path("")
    if (
        command[1:] != expected_command_tail
        or collection_command[0] != command[0]
        or collection_command[1:] != expected_collection_tail
        or not executable.is_absolute()
        or not executable.is_file()
        or Path(sys.executable).resolve() != executable.resolve()
    ):
        raise RuntimeError("verification receipt pytest command differs from the frozen command")
    nodeids = _validate_collected_nodeids(spec, verification.get("collection_nodeids"))
    expected_environment = _verification_environment(base)
    expected_source_manifest = _verification_source_manifest(spec)
    if (
        verification.get("status") != "PASS"
        or verification.get("verification_base") != str(base)
        or verification.get("official_roots_used") is not False
        or verification.get("development_roots_used") != list(spec.development_roots)
        or verification.get("pytest_status") != "PASS"
        or verification.get("pytest_returncode") != 0
        or type(verification.get("pytest_test_count")) is not int
        or verification["pytest_test_count"] <= 0
        or verification.get("collection_status") != "PASS"
        or verification.get("collection_returncode") != 0
        or verification.get("collection_test_count") != len(nodeids)
        or verification.get("pytest_test_count") != len(nodeids)
        or verification.get("collection_nodeids_sha256")
        != _sha256_bytes(_canonical_bytes(list(nodeids)))
        or not _is_sha256(verification.get("collection_stdout_sha256"))
        or verification.get("collection_stderr_sha256") != _sha256_bytes(b"")
        or verification.get("focused_pytest_cwd") != str(ROOT)
        or verification.get("verification_environment") != expected_environment
        or verification.get("pytest_python_realpath") != str(executable.resolve())
        or verification.get("pytest_python_sha256") != _sha256_file(executable.resolve())
        or verification.get("pytest_stderr_sha256") != _sha256_bytes(b"")
        or not _is_sha256(verification.get("pytest_stdout_sha256"))
        or verification.get("scanner_program") != str(ROOT / spec.verification_scanner)
        or verification.get("scanner_forbidden_needles") != _verification_forbidden_manifest(spec)
        or verification.get("scanner_stderr_sha256") != _sha256_bytes(b"")
        or verification.get("forbidden_match_count") != 0
        or verification.get("scan_error_count") != 0
        or verification.get("verification_source_stable") is not True
        or verification.get("verification_source_manifest") != expected_source_manifest
        or verification.get("verification_source_count") != len(expected_source_manifest)
        or verification.get("verification_source_manifest_sha256")
        != _source_manifest_sha256(expected_source_manifest)
    ):
        raise RuntimeError("verification receipt does not satisfy the frozen execution contract")

    scan = verification.get("scan_receipt")
    if not isinstance(scan, dict):
        raise RuntimeError("verification receipt has no nested scan receipt")
    spec.development_domain.validate_receipt("tree_scan", scan)
    scan_body = scan.get("scan")
    expected_base_hash = _sha256_bytes(os.fsencode(str(base)))
    if (
        scan.get("schema") != spec.development_domain.schema("tree_scan")
        or scan.get("namespace") != spec.development_domain.namespace
        or scan.get("root_consumption_status") != DEVELOPMENT_ONLY
        or scan.get("status") != "PASS"
        or not isinstance(scan_body, dict)
        or scan_body.get("base_path_sha256") != expected_base_hash
        or scan_body.get("scan_complete") is not True
        or scan_body.get("errors") != []
        or scan_body.get("forbidden_match_count") != 0
        or scan_body.get("forbidden_matches") != []
        or scan_body.get("forbidden_needles") != _verification_forbidden_manifest(spec)
    ):
        raise RuntimeError("verification nested scan does not satisfy the exact manifest")
    files = scan_body.get("files")
    if (
        not isinstance(files, list)
        or type(scan_body.get("file_count")) is not int
        or scan_body["file_count"] != len(files)
        or type(scan_body.get("directory_count")) is not int
        or scan_body["directory_count"] <= 0
        or type(scan_body.get("path_name_count")) is not int
        or scan_body["path_name_count"] < scan_body["file_count"]
        or scan_body.get("regular_files_sha256") != _sha256_bytes(_canonical_bytes(files))
    ):
        raise RuntimeError("verification nested scan counts or file hash are inconsistent")
    path_hex_values: list[str] = []
    for record in files:
        if not isinstance(record, dict) or set(record) != {"path_hex", "sha256", "size_bytes"}:
            raise RuntimeError("verification nested scan has a malformed file record")
        path_hex = record["path_hex"]
        if (
            not isinstance(path_hex, str)
            or len(path_hex) % 2 != 0
            or any(character not in "0123456789abcdef" for character in path_hex)
            or not _is_sha256(record["sha256"])
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] < 0
        ):
            raise RuntimeError("verification nested scan has an invalid file record")
        path_hex_values.append(path_hex)
    if path_hex_values != sorted(path_hex_values) or len(path_hex_values) != len(
        set(path_hex_values)
    ):
        raise RuntimeError("verification nested scan paths are not unique and sorted")
    if (
        verification.get("scan_file_count") != scan_body["file_count"]
        or verification.get("scan_directory_count") != scan_body["directory_count"]
        or verification.get("scan_regular_files_sha256") != scan_body["regular_files_sha256"]
        or verification.get("scanner_stdout_sha256") != _sha256_bytes(canonical_json_bytes(scan))
    ):
        raise RuntimeError("verification top-level scan evidence disagrees with the nested scan")


def _source_manifest(paths: set[Path]) -> list[dict[str, str]]:
    manifest = [
        {
            "path_sha256": _sha256_bytes(path.as_posix().encode("utf-8")),
            "file_sha256": _sha256_file(ROOT / path),
        }
        for path in sorted(paths, key=lambda item: item.as_posix())
    ]
    if len({record["path_sha256"] for record in manifest}) != len(manifest):
        raise RuntimeError("source-manifest path hash collision")
    return manifest


def _source_manifest_sha256(manifest: list[dict[str, str]]) -> str:
    return _sha256_bytes(_canonical_bytes(manifest))


def _verification_source_manifest(spec: ProtocolSpec) -> list[dict[str, str]]:
    paths = _loaded_local_source_paths(spec)
    paths.update(path for path, _digest in spec.bound_historical_sha256)
    paths.update(
        {
            spec.preregistration.relative_to(ROOT),
            spec.preregistration_review.relative_to(ROOT),
        }
    )
    paths.discard(spec.verification_receipt.relative_to(ROOT))
    paths.discard(spec.implementation_review.relative_to(ROOT))
    return _source_manifest(paths)


def _reviewed_source_manifest(spec: ProtocolSpec) -> list[dict[str, str]]:
    review_path = spec.implementation_review.relative_to(ROOT)
    reviewed_paths = _loaded_local_source_paths(spec)
    reviewed_paths.update(path for path, _digest in spec.bound_historical_sha256)
    reviewed_paths.update(
        {
            spec.preregistration.relative_to(ROOT),
            spec.preregistration_review.relative_to(ROOT),
            spec.verification_receipt.relative_to(ROOT),
        }
    )
    reviewed_paths.discard(review_path)
    return _source_manifest(reviewed_paths)


def _validate_implementation_review(spec: ProtocolSpec, review: dict[str, Any]) -> None:
    spec.development_domain.validate_receipt("implementation_review", review)
    expected_manifest = _reviewed_source_manifest(spec)
    if (
        review.get("status") != "PASS"
        or review.get("recommendation") != "PASS"
        or review.get("independent_review") is not True
        or review.get("reviewed_protocol_label") != spec.protocol_label
        or review.get("reviewed_source_manifest") != expected_manifest
        or review.get("reviewed_source_count") != len(expected_manifest)
        or review.get("reviewed_source_manifest_sha256")
        != _source_manifest_sha256(expected_manifest)
        or review.get("verification_receipt_sha256") != _sha256_file(spec.verification_receipt)
    ):
        raise RuntimeError("implementation review is not a current exact-source PASS artifact")


def _preflight_protocol(
    spec: ProtocolSpec,
    out: Path,
    artifacts_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """Fail closed before creating the artifact directory or attempting root transition."""

    require_rename_exchange()
    out = out.expanduser().resolve()
    artifacts_dir = artifacts_dir.expanduser().resolve()
    if out != spec.official_out.expanduser().resolve():
        raise ValueError("--out differs from the frozen one-shot official result path")
    if artifacts_dir != spec.official_artifacts_dir.expanduser().resolve():
        raise ValueError("--artifacts-dir differs from the frozen one-shot official artifact path")
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("official root stream requires ambient torch.float32 default dtype")
    if torch.get_default_device().type != "cpu":
        raise RuntimeError("official root stream requires ambient default device cpu")
    if out == artifacts_dir or out in artifacts_dir.parents or artifacts_dir in out.parents:
        raise ValueError("--out and --artifacts-dir must be disjoint and non-nested")
    for path, label in ((out, "--out"), (artifacts_dir, "--artifacts-dir")):
        if not path.parent.is_dir() or path.parent.is_symlink():
            raise FileNotFoundError(f"{label} parent must be an existing real directory")
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)

    required_paths = set(spec.declared_source_paths)
    required_paths.update(path for path, _digest in spec.bound_historical_sha256)
    required_paths.update(
        {
            spec.preregistration.relative_to(ROOT),
            spec.preregistration_review.relative_to(ROOT),
            spec.verification_receipt.relative_to(ROOT),
            spec.implementation_review.relative_to(ROOT),
        }
    )
    for required in sorted(required_paths, key=lambda item: item.as_posix()):
        path = ROOT / required
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"required source receipt is unavailable: {required}")
    for relative_path, expected_sha256 in spec.bound_historical_sha256:
        observed_sha256 = _sha256_file(ROOT / relative_path)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"bound historical protocol hash mismatch for {relative_path}: "
                f"expected {expected_sha256}, observed {observed_sha256}"
            )

    source_observation_before_review = _source_observation(spec)
    if source_observation_before_review["errors"]:
        raise RuntimeError("preflight source closure contains a read error")

    verification_bytes = spec.verification_receipt.read_bytes()
    verification = json.loads(verification_bytes)
    if verification_bytes != canonical_json_bytes(verification):
        raise RuntimeError("verification receipt is not canonical durable JSON")
    _validate_verification_receipt(spec, verification)
    review_bytes = spec.implementation_review.read_bytes()
    review = json.loads(review_bytes)
    if review_bytes != canonical_json_bytes(review):
        raise RuntimeError("implementation review is not canonical durable JSON")
    _validate_implementation_review(spec, review)
    source_observation_after_review = _source_observation(spec)
    if source_observation_after_review != source_observation_before_review:
        raise RuntimeError("source closure changed while verification and review were validated")
    return out, artifacts_dir, source_observation_after_review


def _artifact_hashes_protocol(artifacts_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    if not artifacts_dir.exists():
        return hashes
    for path in sorted(artifacts_dir.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"artifact tree contains a symlink: {path}")
        if path.is_file():
            hashes[path.relative_to(artifacts_dir).as_posix()] = _sha256_file(path)
        elif not path.is_dir():
            raise RuntimeError(f"artifact tree contains a non-regular entry: {path}")
    return hashes


def _root_status(
    spec: ProtocolSpec,
    *,
    root_transition_attempted: bool,
    official_generators_consumed: bool,
) -> str:
    not_started, transition_attempted, generators_consumed = (
        spec.official_domain.permitted_root_consumption_statuses
    )
    if official_generators_consumed:
        return generators_consumed
    if root_transition_attempted:
        return transition_attempted
    return not_started


def _safe_public_state_after_error(
    error: OwnedMutationError,
    prior: Ownership | None,
) -> tuple[Ownership | None, bool]:
    report = error.report
    if report.recovery_uncertainty or (report.publication_error and report.public_entry is None):
        return None, False
    public = report.public_entry
    if public is None or not public.is_regular or public.ownership is None:
        return None, False
    if prior is not None and public.ownership.same_identity_and_hash(prior):
        return prior, True
    return None, False


def _bounded_invalid_payload(
    spec: ProtocolSpec,
    *,
    phase: str,
    error: Exception,
    artifacts_dir: Path,
    records: list[dict[str, Any]],
    raw_descriptors: list[dict[str, Any]],
    common_descriptors: list[dict[str, Any]],
    rank_descriptors: list[dict[str, Any]],
    initialization_descriptors: list[dict[str, Any]],
    source_observation_start: dict[str, Any],
    first_source_observation: dict[str, Any] | None,
    root_transition_attempted: bool,
    official_generators_consumed: bool,
    mutation_reports: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    observation = (
        first_source_observation
        if first_source_observation is not None
        else _source_observation(spec)
    )
    try:
        partial_artifacts: dict[str, str] | None = _artifact_hashes_protocol(artifacts_dir)
        artifact_error: str | None = None
    except Exception as hash_error:
        partial_artifacts = None
        artifact_error = f"{type(hash_error).__name__}: {hash_error}"
    payload = {
        "status": "INVALID",
        "phase": phase,
        "error_class": type(error).__name__,
        "error_message": str(error),
        "root_transition_attempted": root_transition_attempted,
        "official_generators_consumed": official_generators_consumed,
        "completed_arms": [
            {"scene_root": record["scene_root"], "arm": record["arm"]} for record in records
        ],
        "raw_input_receipts": raw_descriptors,
        "common_constructor_receipts": common_descriptors,
        "rank_receipts": rank_descriptors,
        "arm_initialization_receipts": initialization_descriptors,
        "partial_artifact_sha256": partial_artifacts,
        "partial_artifact_hash_error": artifact_error,
        "source_observation_start": source_observation_start,
        "first_source_observation": observation,
        "source_drift": observation != source_observation_start,
        "mutation_reports": mutation_reports,
        "timestamp_ns": time.time_ns(),
    }
    return payload, observation


def _publish_invalid_protocol(
    spec: ProtocolSpec,
    *,
    out: Path,
    terminal_path: Path,
    lifecycle_path: Path,
    payload: dict[str, Any],
    root_consumption_status: str,
    result_ownership: Ownership | None,
    terminal_ownership: Ownership | None,
    lifecycle_ownership: Ownership | None,
    result_mutation_allowed: bool,
    terminal_mutation_allowed: bool,
    lifecycle_mutation_allowed: bool,
) -> dict[str, Any]:
    """Best-effort INVALID publication; colliders and uncertain names are never removed."""

    domain = spec.official_domain
    roots = spec.scene_roots + spec.initial_depth_roots
    failures: dict[str, str] = {}
    reports: dict[str, Any] = {}
    published: dict[str, str] = {}

    def attempt(
        label: str,
        path: Path,
        kind: str,
        receipt: dict[str, Any],
        ownership: Ownership | None,
        mutation_allowed: bool,
    ) -> Ownership | None:
        if not mutation_allowed:
            failures[label] = "public name is contested or uncertain; no further mutation permitted"
            return ownership
        directory_fd: int | None = None
        try:
            directory_fd = open_directory(path.parent)
            prepared = prepare_json(directory_fd, path.name, domain, kind, receipt)
            try:
                report = (
                    publish_exclusive(directory_fd, prepared)
                    if ownership is None
                    else exchange_owned(directory_fd, prepared, ownership)
                )
            except OwnedMutationError as mutation_error:
                reports[label] = _mutation_report_receipt(mutation_error.report)
                failures[label] = f"{type(mutation_error).__name__}: {mutation_error}"
                safe_ownership, _mutation_allowed = _safe_public_state_after_error(
                    mutation_error, ownership
                )
                return safe_ownership
            reports[label] = _mutation_report_receipt(report)
            published[label] = prepared.payload_sha256
            return prepared.ownership
        except Exception as publication_error:
            failures[label] = f"{type(publication_error).__name__}: {publication_error}"
            return ownership
        finally:
            if directory_fd is not None:
                os.close(directory_fd)

    aggregate_receipt = domain.make_receipt(
        "aggregate",
        payload,
        root_consumption_status=root_consumption_status,
        roots=roots,
        official_phase="invalid_publication",
        commit_state="INVALID",
    )
    result_ownership = attempt(
        "aggregate",
        out,
        "aggregate",
        aggregate_receipt,
        result_ownership,
        result_mutation_allowed,
    )
    terminal_receipt = domain.make_receipt(
        "terminal",
        {**payload, "aggregate_sha256": published.get("aggregate")},
        root_consumption_status=root_consumption_status,
        roots=roots,
        official_phase="invalid_publication",
        commit_state="INVALID",
    )
    terminal_ownership = attempt(
        "terminal",
        terminal_path,
        "terminal",
        terminal_receipt,
        terminal_ownership,
        terminal_mutation_allowed,
    )
    lifecycle_receipt = domain.make_receipt(
        "lifecycle",
        {
            "status": "INVALID",
            "phase": payload["phase"],
            "published_sha256": published,
            "publication_failures": failures,
            "mutation_reports": reports,
            "source_observation_start": payload["source_observation_start"],
            "first_source_observation": payload["first_source_observation"],
            "timestamp_ns": time.time_ns(),
        },
        root_consumption_status=root_consumption_status,
        roots=roots,
        official_phase="invalid_publication",
        commit_state="INVALID",
    )
    if lifecycle_ownership is None or not lifecycle_mutation_allowed:
        failures["lifecycle"] = "ownership unavailable; contested lifecycle not mutated"
    else:
        attempt(
            "lifecycle",
            lifecycle_path,
            "lifecycle",
            lifecycle_receipt,
            lifecycle_ownership,
            lifecycle_mutation_allowed,
        )

    fallback_name = f"invalid-publication-fallback-{uuid.uuid4().hex}.json"
    fallback_path = terminal_path.parent / fallback_name
    fallback_receipt = domain.make_receipt(
        "fallback_manifest",
        {
            **payload,
            "published_sha256": published,
            "publication_failures": failures,
            "publication_reports": reports,
        },
        root_consumption_status=root_consumption_status,
        roots=roots,
        official_phase="invalid_fallback",
        commit_state="INVALID",
    )
    try:
        fallback = _write_receipt_exclusive(
            fallback_path,
            domain,
            "fallback_manifest",
            fallback_receipt,
        )
        published["fallback_manifest"] = fallback.payload_sha256
    except Exception as fallback_error:
        failures["fallback_manifest"] = f"{type(fallback_error).__name__}: {fallback_error}"
    return {
        "published_sha256": published,
        "publication_failures": failures,
        "publication_reports": reports,
    }


def _validate_committed_protocol(
    spec: ProtocolSpec,
    *,
    out: Path,
    terminal_path: Path,
    lifecycle_path: Path,
    result: dict[str, Any],
    terminal: dict[str, Any],
    lifecycle: dict[str, Any],
    result_prepared: PreparedJSON,
    terminal_prepared: PreparedJSON,
    lifecycle_prepared: PreparedJSON,
    reports: tuple[MutationReport, MutationReport, MutationReport],
) -> dict[str, Any]:
    domain = spec.official_domain
    domain.validate_receipt("aggregate", result)
    domain.validate_receipt("terminal", terminal)
    domain.validate_receipt("lifecycle", lifecycle)
    for label, receipt, prepared in (
        ("aggregate", result, result_prepared),
        ("terminal", terminal, terminal_prepared),
        ("lifecycle", lifecycle, lifecycle_prepared),
    ):
        if _sha256_bytes(canonical_json_bytes(receipt)) != prepared.payload_sha256:
            raise RuntimeError(f"{label} receipt differs from its prepared public bytes")
    expected_roots = list(spec.scene_roots + spec.initial_depth_roots)
    for label, receipt, expected_commit_state in (
        ("aggregate", result, "UNCOMMITTED"),
        ("terminal", terminal, "UNCOMMITTED"),
        ("lifecycle", lifecycle, "COMMITTED"),
    ):
        if receipt.get("root_consumption_status") != domain.permitted_root_consumption_statuses[-1]:
            raise RuntimeError(f"{label} does not record consumed official generators")
        if receipt.get("roots") != expected_roots:
            raise RuntimeError(f"{label} does not carry the exact six official roots")
        if receipt.get("commit_state") != expected_commit_state:
            raise RuntimeError(f"{label} has an invalid commit state")
    for path, prepared in (
        (out, result_prepared),
        (terminal_path, terminal_prepared),
        (lifecycle_path, lifecycle_prepared),
    ):
        directory_fd = open_directory(path.parent)
        try:
            observed = capture_entry(
                directory_fd,
                path.name,
                expected_sha256=prepared.payload_sha256,
            )
        finally:
            os.close(directory_fd)
        if not observed.is_regular or observed.ownership is None:
            raise RuntimeError(f"committed path is not a durable regular file: {path}")
    status = result.get("status")
    if status not in {"PASS", "FAIL"}:
        raise RuntimeError("aggregate does not contain a terminal scientific status")
    if terminal.get("status") != status or lifecycle.get("status") != status:
        raise RuntimeError("aggregate, terminal, and lifecycle scientific status differ")
    shared_fields = (
        "namespace",
        "root_consumption_status",
        "source_observation_start",
        "source_observation_end",
    )
    for field in shared_fields:
        if result.get(field) != terminal.get(field) or result.get(field) != lifecycle.get(field):
            raise RuntimeError(f"committed receipts disagree on {field}")
    if result.get("source_observation_start") != result.get("source_observation_end"):
        raise RuntimeError("committed source observation changed")
    source_observation = result.get("source_observation_start")
    if not isinstance(source_observation, dict) or source_observation.get("errors") != {}:
        raise RuntimeError("committed source observation is incomplete or unreadable")
    if terminal.get("phase") != "complete":
        raise RuntimeError("terminal phase is not complete")
    if terminal.get("aggregate_sha256") != result_prepared.payload_sha256:
        raise RuntimeError("terminal aggregate hash mismatch")
    if lifecycle.get("aggregate_sha256") != result_prepared.payload_sha256:
        raise RuntimeError("lifecycle aggregate hash mismatch")
    if lifecycle.get("terminal_sha256") != terminal_prepared.payload_sha256:
        raise RuntimeError("lifecycle terminal hash mismatch")
    if any(
        not report.accepted or report.recovery_uncertainty or report.publication_error
        for report in reports
    ):
        raise RuntimeError("transaction validator found publication uncertainty")
    expected_operations = ("exclusive_link", "exclusive_link", "owned_exchange")
    expected_names = (out.name, terminal_path.name, lifecycle_path.name)
    for report, operation, target_name in zip(
        reports, expected_operations, expected_names, strict=True
    ):
        if report.operation != operation or report.target_name != target_name:
            raise RuntimeError("transaction validator found an invalid publication sequence")
    return domain.make_receipt(
        "commit_validation",
        {
            "status": status,
            "aggregate_path": str(out),
            "aggregate_sha256": result_prepared.payload_sha256,
            "terminal_path": str(terminal_path),
            "terminal_sha256": terminal_prepared.payload_sha256,
            "lifecycle_path": str(lifecycle_path),
            "lifecycle_sha256": lifecycle_prepared.payload_sha256,
            "publication_reports": [_mutation_report_receipt(report) for report in reports],
            "recovery_uncertainty": False,
            "publication_error": False,
            "timestamp_ns": time.time_ns(),
        },
        root_consumption_status=domain.permitted_root_consumption_statuses[-1],
        roots=spec.scene_roots + spec.initial_depth_roots,
        official_phase="commit_validation",
        commit_state="COMMITTED",
    )


def run_protocol(
    spec: ProtocolSpec,
    out: Path,
    artifacts_dir: Path,
) -> dict[str, Any]:
    out, artifacts_dir, reviewed_source_observation = _preflight_protocol(
        spec,
        out,
        artifacts_dir,
    )
    domain = spec.official_domain
    roots = spec.scene_roots + spec.initial_depth_roots
    roots_not_started, transition_status, generators_consumed_status = (
        domain.permitted_root_consumption_statuses
    )
    torch.use_deterministic_algorithms(True)

    config = _experiment_config(
        spec,
        domain,
        root_consumption_status=roots_not_started,
    )
    provenance = _provenance(
        spec,
        domain,
        root_consumption_status=roots_not_started,
    )
    source_observation_start: dict[str, Any] = provenance["source_observation"]
    if source_observation_start["errors"]:
        raise RuntimeError("start source closure contains a read error")
    if source_observation_start != reviewed_source_observation:
        raise RuntimeError("source closure changed after implementation-review preflight")

    _mkdir_exclusive_durable(artifacts_dir)
    config_path = artifacts_dir / "config.json"
    provenance_path = artifacts_dir / "provenance.json"
    lifecycle_path = artifacts_dir / "lifecycle.json"
    terminal_path = artifacts_dir / "terminal.json"
    validation_path = artifacts_dir / "commit_validation.json"
    config_prepared = _write_receipt_exclusive(config_path, domain, "config", config)
    provenance_prepared = _write_receipt_exclusive(
        provenance_path,
        domain,
        "provenance",
        provenance,
    )
    lifecycle = domain.make_receipt(
        "lifecycle",
        {
            "status": "PENDING",
            "phase": "pre_root",
            "config_sha256": config_prepared.payload_sha256,
            "provenance_sha256": provenance_prepared.payload_sha256,
            "timestamp_ns": time.time_ns(),
        },
        root_consumption_status=roots_not_started,
        roots=roots,
        official_phase="pre_root",
        commit_state="UNCOMMITTED",
    )
    lifecycle_prepared = _write_receipt_exclusive(
        lifecycle_path,
        domain,
        "lifecycle",
        lifecycle,
    )
    lifecycle_ownership: Ownership | None = lifecycle_prepared.ownership
    lifecycle_mutation_allowed = True
    artifacts_fd = open_directory(artifacts_dir)

    records: list[dict[str, Any]] = []
    raw_descriptors: list[dict[str, Any]] = []
    common_descriptors: list[dict[str, Any]] = []
    rank_descriptors: list[dict[str, Any]] = []
    initialization_descriptors: list[dict[str, Any]] = []
    paired_initial_inputs: list[dict[str, Any]] = []
    rank_values: list[dict[str, Any]] = []
    mutation_reports: list[dict[str, Any]] = []
    result_ownership: Ownership | None = None
    terminal_ownership: Ownership | None = None
    result_mutation_allowed = True
    terminal_mutation_allowed = True
    root_transition_attempted = False
    official_generators_consumed = False
    first_source_observation: dict[str, Any] | None = None
    phase = "development_sentinels"
    try:
        construction = _construction_sentinel()
        gradient, duplicate = _gradient_and_duplicate_sentinels()
        if not construction["pass"] or not gradient["pass"] or not duplicate["pass"]:
            raise RuntimeError("a development validity sentinel failed before official roots")

        phase = "root_transition"
        root_transition_receipt = domain.make_receipt(
            "lifecycle",
            {
                "status": "PENDING",
                "phase": "root_transition_attempted",
                "config_sha256": config_prepared.payload_sha256,
                "provenance_sha256": provenance_prepared.payload_sha256,
                "timestamp_ns": time.time_ns(),
            },
            root_consumption_status=transition_status,
            roots=roots,
            official_phase="root_transition_attempted",
            commit_state="UNCOMMITTED",
        )
        transition_prepared = prepare_json(
            artifacts_fd,
            lifecycle_path.name,
            domain,
            "lifecycle",
            root_transition_receipt,
        )
        root_transition_attempted = True
        assert lifecycle_ownership is not None
        try:
            transition_report = exchange_owned(
                artifacts_fd,
                transition_prepared,
                lifecycle_ownership,
            )
        except OwnedMutationError as mutation_error:
            mutation_reports.append(_mutation_report_receipt(mutation_error.report))
            lifecycle_ownership, lifecycle_mutation_allowed = _safe_public_state_after_error(
                mutation_error,
                lifecycle_ownership,
            )
            raise
        mutation_reports.append(_mutation_report_receipt(transition_report))
        lifecycle_ownership = transition_prepared.ownership

        phase = "generator_transition"
        generator_receipt = domain.make_receipt(
            "lifecycle",
            {
                "status": "PENDING",
                "phase": "official_generators_consumed",
                "config_sha256": config_prepared.payload_sha256,
                "provenance_sha256": provenance_prepared.payload_sha256,
                "timestamp_ns": time.time_ns(),
            },
            root_consumption_status=generators_consumed_status,
            roots=roots,
            official_phase="official_generators_consumed",
            commit_state="UNCOMMITTED",
        )
        generator_prepared = prepare_json(
            artifacts_fd,
            lifecycle_path.name,
            domain,
            "lifecycle",
            generator_receipt,
        )
        assert lifecycle_ownership is not None
        try:
            generator_report = exchange_owned(
                artifacts_fd,
                generator_prepared,
                lifecycle_ownership,
            )
        except OwnedMutationError as mutation_error:
            mutation_reports.append(_mutation_report_receipt(mutation_error.report))
            lifecycle_ownership, lifecycle_mutation_allowed = _safe_public_state_after_error(
                mutation_error,
                lifecycle_ownership,
            )
            raise
        mutation_reports.append(_mutation_report_receipt(generator_report))
        lifecycle_ownership = generator_prepared.ownership

        phase = "official_fitting"
        for scene_root, depth_root in zip(
            spec.scene_roots,
            spec.initial_depth_roots,
            strict=True,
        ):
            official_generators_consumed = True
            scene = _build_scene(scene_root, depth_root)
            raw_descriptor = _write_scene_input_receipt(
                scene,
                artifacts_dir,
                domain=domain,
                root_consumption_status=generators_consumed_status,
            )
            raw_descriptors.append(raw_descriptor)
            (
                _common_fiber,
                common_means,
                common_covariances,
                common_descriptor,
            ) = _write_common_constructor_receipt(
                scene,
                raw_descriptor,
                artifacts_dir,
                domain,
                generators_consumed_status,
            )
            common_descriptors.append(common_descriptor)
            rank = _rank_sentinel(scene)
            rank_values.append(rank)
            rank_descriptor = _write_rank_receipt(
                scene,
                rank,
                artifacts_dir,
                domain,
                generators_consumed_status,
            )
            rank_descriptors.append(rank_descriptor)
            if common_descriptor["receipt"].get("pass") is not True:
                raise RuntimeError("common constructor failed after durable error receipt")
            if rank.get("pass") is not True:
                raise RuntimeError("official rank sentinel failed after durable receipt")
            if common_means is None or common_covariances is None:
                raise RuntimeError("passing common receipt has no realized geometry")

            paired_record = {
                "scene_root": scene_root,
                "initial_depth_root": depth_root,
                "depths_sha256": scene["initial_depth_sha256"],
                "constructor_input_means_sha256": _tensor_sha256(common_means),
                "constructor_input_covariances_sha256": _tensor_sha256(common_covariances),
                "raw_input_receipt": raw_descriptor,
                "common_constructor_receipt": common_descriptor,
                "rank_receipt": rank_descriptor,
                "arms": [],
            }
            paired_initial_inputs.append(paired_record)
            for arm in ARMS:
                model, initialization_descriptor = _build_and_write_initialization_receipt(
                    arm,
                    scene,
                    common_means,
                    common_covariances,
                    artifacts_dir,
                    domain,
                    generators_consumed_status,
                )
                initialization_descriptors.append(initialization_descriptor)
                paired_record["arms"].append(initialization_descriptor)
                if initialization_descriptor["receipt"].get("pass") is not True:
                    raise RuntimeError(f"{arm} failed paired initialization after durable receipt")
                if model is None:
                    raise RuntimeError("passing initialization receipt has no model")
                record = _train_arm(
                    model,
                    arm,
                    scene,
                    artifacts_dir / f"scene_{scene_root}" / arm,
                    config_prepared.payload_sha256,
                    provenance_prepared.payload_sha256,
                    initialization_descriptor,
                    domain=domain,
                    root_consumption_status=generators_consumed_status,
                )
                records.append(record)

        phase = "aggregation"
        sentinels = {
            "construction": construction,
            "finite_difference": gradient,
            "duplicate_hard_min": duplicate,
            "rank": rank_values,
            "checkpoint_count": sum(len(record["checkpoints"]) for record in records),
            "checkpoint_expected_count": (
                len(spec.scene_roots) * len(ARMS) * (UPDATES // CHECKPOINT_INTERVAL + 1)
            ),
            "initialization_equivalence_pass": all(
                descriptor["receipt"].get("pass") is True
                for descriptor in initialization_descriptors
            ),
            "checkpoint_sentinels_pass": all(
                checkpoint["pass"] for record in records for checkpoint in record["checkpoints"]
            ),
        }
        sentinels["pass"] = _combined_sentinels_pass(sentinels)
        _require_combined_sentinels(sentinels)
        aggregate = _aggregate(records, spec.scene_roots)
        gates = _scientific_gates(records, sentinels, spec.scene_roots)
        scientific_status = gates["overall_status"]
        if scientific_status not in {"PASS", "FAIL"}:
            raise RuntimeError("scientific gates did not produce PASS or FAIL")

        source_hashes = source_observation_start["hashes"]
        result_payload = {
            "status": scientific_status,
            "config": config,
            "config_sha256": config_prepared.payload_sha256,
            "provenance": provenance,
            "provenance_sha256": provenance_prepared.payload_sha256,
            "preregistration": str(spec.preregistration),
            "preregistration_review": str(spec.preregistration_review),
            "verification_receipt": str(spec.verification_receipt),
            "implementation_review": str(spec.implementation_review),
            "preregistration_sha256": source_hashes[
                spec.preregistration.relative_to(ROOT).as_posix()
            ],
            "preregistration_review_sha256": source_hashes[
                spec.preregistration_review.relative_to(ROOT).as_posix()
            ],
            "verification_receipt_sha256": source_hashes[
                spec.verification_receipt.relative_to(ROOT).as_posix()
            ],
            "implementation_review_sha256": source_hashes[
                spec.implementation_review.relative_to(ROOT).as_posix()
            ],
            "raw_input_receipts": raw_descriptors,
            "common_constructor_receipts": common_descriptors,
            "rank_receipts": rank_descriptors,
            "arm_initialization_receipts": initialization_descriptors,
            "paired_initial_inputs": paired_initial_inputs,
            "validity_sentinels": sentinels,
            "records": records,
            "aggregate": aggregate,
            "scientific_gates": gates,
            "artifacts_directory": str(artifacts_dir),
            "artifact_sha256_before_final_preparation": _artifact_hashes_protocol(artifacts_dir),
            "process_peak_rss_bytes_so_far": _peak_rss_bytes(),
            "source_observation_start": source_observation_start,
            "source_observation_end": source_observation_start,
            "commit_validation_path": str(validation_path),
            "timestamp_ns": time.time_ns(),
        }
        result = domain.make_receipt(
            "aggregate",
            result_payload,
            root_consumption_status=generators_consumed_status,
            roots=roots,
            official_phase="final_preparation",
            commit_state="UNCOMMITTED",
        )
        result_sha256 = _sha256_bytes(canonical_json_bytes(result))
        terminal = domain.make_receipt(
            "terminal",
            {
                "status": scientific_status,
                "phase": "complete",
                "aggregate_path": str(out),
                "aggregate_sha256": result_sha256,
                "completed_arms": [
                    {"scene_root": record["scene_root"], "arm": record["arm"]} for record in records
                ],
                "source_observation_start": source_observation_start,
                "source_observation_end": source_observation_start,
                "commit_validation_path": str(validation_path),
                "timestamp_ns": time.time_ns(),
            },
            root_consumption_status=generators_consumed_status,
            roots=roots,
            official_phase="complete",
            commit_state="UNCOMMITTED",
        )
        terminal_sha256 = _sha256_bytes(canonical_json_bytes(terminal))
        final_lifecycle = domain.make_receipt(
            "lifecycle",
            {
                "status": scientific_status,
                "phase": "complete",
                "aggregate_path": str(out),
                "aggregate_sha256": result_sha256,
                "terminal_path": str(terminal_path),
                "terminal_sha256": terminal_sha256,
                "source_observation_start": source_observation_start,
                "source_observation_end": source_observation_start,
                "commit_validation_path": str(validation_path),
                "timestamp_ns": time.time_ns(),
            },
            root_consumption_status=generators_consumed_status,
            roots=roots,
            official_phase="complete",
            commit_state="COMMITTED",
        )

        phase = "final_preparation"
        output_fd = open_directory(out.parent)
        try:
            result_prepared = prepare_json(
                output_fd,
                out.name,
                domain,
                "aggregate",
                result,
            )
            terminal_prepared = prepare_json(
                artifacts_fd,
                terminal_path.name,
                domain,
                "terminal",
                terminal,
            )
            final_lifecycle_prepared = prepare_json(
                artifacts_fd,
                lifecycle_path.name,
                domain,
                "lifecycle",
                final_lifecycle,
            )
            if result_prepared.payload_sha256 != result_sha256:
                raise RuntimeError("prepared aggregate bytes differ from frozen candidate")
            if terminal_prepared.payload_sha256 != terminal_sha256:
                raise RuntimeError("prepared terminal bytes differ from frozen candidate")

            phase = "final_source_observation"
            first_source_observation = _source_observation(spec)
            if first_source_observation != source_observation_start:
                raise RuntimeError("scientific source closure changed or became unreadable")

            phase = "authoritative_publication"
            try:
                result_report = publish_exclusive(output_fd, result_prepared)
            except OwnedMutationError as mutation_error:
                mutation_reports.append(_mutation_report_receipt(mutation_error.report))
                result_ownership, result_mutation_allowed = _safe_public_state_after_error(
                    mutation_error, None
                )
                raise
            result_ownership = result_prepared.ownership
            mutation_reports.append(_mutation_report_receipt(result_report))
            try:
                terminal_report = publish_exclusive(artifacts_fd, terminal_prepared)
            except OwnedMutationError as mutation_error:
                mutation_reports.append(_mutation_report_receipt(mutation_error.report))
                terminal_ownership, terminal_mutation_allowed = _safe_public_state_after_error(
                    mutation_error, None
                )
                raise
            terminal_ownership = terminal_prepared.ownership
            mutation_reports.append(_mutation_report_receipt(terminal_report))
            assert lifecycle_ownership is not None
            try:
                lifecycle_report = exchange_owned(
                    artifacts_fd,
                    final_lifecycle_prepared,
                    lifecycle_ownership,
                )
            except OwnedMutationError as mutation_error:
                mutation_reports.append(_mutation_report_receipt(mutation_error.report))
                lifecycle_ownership, lifecycle_mutation_allowed = _safe_public_state_after_error(
                    mutation_error,
                    lifecycle_ownership,
                )
                raise
            lifecycle_ownership = final_lifecycle_prepared.ownership
            mutation_reports.append(_mutation_report_receipt(lifecycle_report))
        finally:
            os.close(output_fd)

        phase = "commit_validation"
        validation = _validate_committed_protocol(
            spec,
            out=out,
            terminal_path=terminal_path,
            lifecycle_path=lifecycle_path,
            result=result,
            terminal=terminal,
            lifecycle=final_lifecycle,
            result_prepared=result_prepared,
            terminal_prepared=terminal_prepared,
            lifecycle_prepared=final_lifecycle_prepared,
            reports=(result_report, terminal_report, lifecycle_report),
        )
        validation_prepared = _write_receipt_exclusive(
            validation_path,
            domain,
            "commit_validation",
            validation,
        )
        return {
            **result,
            "commit_validation": validation,
            "commit_validation_sha256": validation_prepared.payload_sha256,
        }
    except Exception as error:
        if root_transition_attempted:
            invalid_payload, first_source_observation = _bounded_invalid_payload(
                spec,
                phase=phase,
                error=error,
                artifacts_dir=artifacts_dir,
                records=records,
                raw_descriptors=raw_descriptors,
                common_descriptors=common_descriptors,
                rank_descriptors=rank_descriptors,
                initialization_descriptors=initialization_descriptors,
                source_observation_start=source_observation_start,
                first_source_observation=first_source_observation,
                root_transition_attempted=root_transition_attempted,
                official_generators_consumed=official_generators_consumed,
                mutation_reports=mutation_reports,
            )
            try:  # noqa: SIM105 - preserve original exception after best-effort INVALID
                _publish_invalid_protocol(
                    spec,
                    out=out,
                    terminal_path=terminal_path,
                    lifecycle_path=lifecycle_path,
                    payload=invalid_payload,
                    root_consumption_status=_root_status(
                        spec,
                        root_transition_attempted=root_transition_attempted,
                        official_generators_consumed=official_generators_consumed,
                    ),
                    result_ownership=result_ownership,
                    terminal_ownership=terminal_ownership,
                    lifecycle_ownership=lifecycle_ownership,
                    result_mutation_allowed=result_mutation_allowed,
                    terminal_mutation_allowed=terminal_mutation_allowed,
                    lifecycle_mutation_allowed=lifecycle_mutation_allowed,
                )
            except Exception:
                # The original exception is authoritative.  Prepared INVALID and recovery
                # entries, if any, remain as best-effort evidence.
                pass
        raise
    finally:
        os.close(artifacts_fd)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Exclusive aggregate JSON output path.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Exclusive directory for config, provenance, per-arm JSON, and PLY artifacts.",
    )
    return parser.parse_args(argv)
