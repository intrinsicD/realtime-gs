#!/usr/bin/env python3
"""Iteration 3 synthetic screen for capacity-aware exact-fiber correspondence.

The official roots and scientific schedule are frozen in the preregistration.  Development
mode accepts smaller schedules for implementation smoke tests, but refuses every official root.
Held-out cameras, projections, and labels are constructed only after all fitted geometry and
final correspondence plans have been frozen.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import quat_to_rotmat
from rtgs.data.synthetic import make_gt_gaussians
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
from rtgs.lift.inverse_projection_fiber import (
    InverseProjectionFiber,
    spd_affine_invariant_squared,
)
from rtgs.lift.topology import radius_connected_components
from rtgs.render.projection import project_covariances_ewa

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "rtgs.inverse-projection-fiber.iter3.synthetic.v1"
PREREG = ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter3_PREREG.md"
PREREG_SHA256 = "59f0de21da20bb5785e2c5f14c89fc82114fed2d5945c704115d64b9fb3c27c8"
PREREG_ADDENDA = (
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
OFFICIAL_RESULT = (
    ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter3_SYNTHETIC_RESULT.json"
)
OFFICIAL_ATTEMPT = (
    ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter3_SYNTHETIC_ATTEMPT.json"
)
OFFICIAL_ARTIFACTS = ROOT / "runs/inverse_projection_fiber_iter3_synthetic_20260717"
OFFICIAL_SCENE_ROOTS = (37_688_011, 37_688_012, 37_688_013)
OFFICIAL_DEPTH_ROOTS = (37_688_111, 37_688_112, 37_688_113)
OFFICIAL_ORDER_ROOTS = (37_688_211, 37_688_212, 37_688_213)
OFFICIAL_ROOT_TUPLES = tuple(
    zip(OFFICIAL_SCENE_ROOTS, OFFICIAL_DEPTH_ROOTS, OFFICIAL_ORDER_ROOTS, strict=True)
)
DEVELOPMENT_ROOT_TUPLES = (
    (3_768_801, 3_768_811, 3_768_821),
    (3_768_802, 3_768_812, 3_768_822),
    (3_768_803, 3_768_813, 3_768_823),
)

N_PARENTS = 8
N_FIT_VIEWS = 5
N_HELDOUT_VIEWS = 2
DEPTH_LOWER = 1.2
DEPTH_UPPER = 3.6
DILATION = 0.3
SPLIT_FRACTION = 0.20
OUTLIERS_PER_VIEW = 2
ARM_NAMES = ("hardmin", "row", "uot_uniform", "uot_area", "oracle", "shuffled_view")
CORRECT_SUPPORT_FRACTION = 0.20
UOT_MIN_MASS_RATIO = 0.20
UOT_MAX_MASS_RATIO = 5.0
UOT_MAX_MARGINAL_RELATIVE_ERROR = 4.0
UOT_MAX_FIXED_POINT_RESIDUAL = 0.05
EXECUTED_SOURCE_PATHS = (
    Path("benchmarks/inverse_projection_fiber_iter3.py"),
    Path("src/rtgs/core/camera.py"),
    Path("src/rtgs/core/gaussians3d.py"),
    Path("src/rtgs/data/synthetic.py"),
    Path("src/rtgs/lift/fiber_correspondence.py"),
    Path("src/rtgs/lift/inverse_projection_fiber.py"),
    Path("src/rtgs/lift/topology.py"),
    Path("src/rtgs/render/projection.py"),
)


@dataclass(frozen=True)
class ScientificConfig:
    outer_steps: int = 20
    geometry_steps: int = 2
    learning_rate: float = 0.025
    temperature_start: float = 2.0
    temperature_stop: float = 0.10
    residual_variance_start: float = 1.0
    residual_variance_stop: float = 0.05
    dustbin_cost: float = 4.0
    sinkhorn_iterations: int = 50
    marginal_penalty: float = 1.0
    sinkhorn_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.outer_steps <= 0 or self.geometry_steps <= 0:
            raise ValueError("outer_steps and geometry_steps must be positive")
        if self.sinkhorn_iterations <= 0:
            raise ValueError("sinkhorn_iterations must be positive")
        positive = (
            self.learning_rate,
            self.temperature_start,
            self.temperature_stop,
            self.residual_variance_start,
            self.residual_variance_stop,
            self.marginal_penalty,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("scientific schedule values must be finite and positive")
        if not math.isfinite(self.dustbin_cost) or self.dustbin_cost < 0:
            raise ValueError("dustbin_cost must be finite and non-negative")


FROZEN_CONFIG = ScientificConfig()


@dataclass(frozen=True)
class CandidateInputs:
    """The label-free fitting boundary shared by every non-oracle arm."""

    cameras: tuple[Camera, ...]
    observations: tuple[ObservationGaussians, ...]
    source_view_indices: torch.Tensor
    source_component_indices: torch.Tensor
    source_means2d: torch.Tensor
    source_covariances2d: torch.Tensor
    initial_depths: torch.Tensor


@dataclass(frozen=True)
class EvaluatorData:
    gt_means: torch.Tensor
    gt_covariances: torch.Tensor
    fit_parent_labels: tuple[torch.Tensor, ...]
    fit_child_indices: tuple[torch.Tensor, ...]
    source_parent_labels: torch.Tensor


@dataclass(frozen=True)
class HeldoutRecipe:
    order_generator_state: torch.Tensor


@dataclass(frozen=True)
class HeldoutData:
    cameras: tuple[Camera, ...]
    observations: tuple[ObservationGaussians, ...]
    parent_labels: tuple[torch.Tensor, ...]
    receipt: dict[str, Any]


@dataclass(frozen=True)
class RootInputs:
    roots: tuple[int, int, int]
    candidate: CandidateInputs
    evaluator: EvaluatorData
    heldout_recipe: HeldoutRecipe
    receipt: dict[str, Any]


@dataclass(frozen=True)
class ArmState:
    name: str
    model: InverseProjectionFiber
    plans: tuple[CorrespondencePlan, ...]
    history: tuple[dict[str, Any], ...]
    wall_time_seconds: float
    initial_state_sha256: str


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    header = f"{tensor.dtype}:{tuple(tensor.shape)}:".encode()
    return _sha256_bytes(header + tensor.numpy().tobytes(order="C"))


def _tensor_mapping_hash(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(_canonical_json(list(tensor.shape)))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _camera_receipt(camera: Camera) -> dict[str, Any]:
    return {
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "width": camera.width,
        "height": camera.height,
        "R_sha256": _tensor_hash(camera.R),
        "t_sha256": _tensor_hash(camera.t),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, payload: Any) -> dict[str, Any]:
    data = _canonical_json(payload)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)
    return {"path": str(path), "bytes": len(data), "sha256": _sha256_bytes(data)}


def _write_npz_exclusive(
    path: Path,
    arrays: dict[str, torch.Tensor | np.ndarray],
) -> dict[str, Any]:
    converted = {
        key: np.ascontiguousarray(
            value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        )
        for key, value in arrays.items()
    }
    with path.open("xb") as stream:
        np.savez(stream, **converted)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "members": {
            key: {"dtype": value.dtype.str, "shape": list(value.shape)}
            for key, value in sorted(converted.items())
        },
    }


def _validated_protocol_receipt() -> dict[str, Any]:
    documents = [(PREREG, PREREG_SHA256), *PREREG_ADDENDA]
    receipt: list[dict[str, str]] = []
    for path, expected in documents:
        observed = _sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                "Iteration 3 preregistration document hash mismatch: "
                f"{path}; expected {expected}, observed {observed}"
            )
        receipt.append({"path": str(path), "sha256": observed})
    return {"base": receipt[0], "addenda": receipt[1:]}


def _source_hashes() -> dict[str, str]:
    return {str(relative): _sha256_file(ROOT / relative) for relative in EXECUTED_SOURCE_PATHS}


def _environment_receipt() -> dict[str, Any]:
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "load_average_1m_5m_15m": list(os.getloadavg()),
    }


def _reserve_official_attempt(
    *,
    artifacts_dir: Path,
    result_path: Path,
    roots: tuple[tuple[int, int, int], ...],
    config: ScientificConfig,
    arm_names: tuple[str, ...],
    protocol: dict[str, Any],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    occupied = [path for path in (OFFICIAL_ATTEMPT, result_path, artifacts_dir) if path.exists()]
    if occupied:
        joined = ", ".join(str(path) for path in occupied)
        raise RuntimeError(f"official Iteration 3 attempt/result already exists: {joined}")
    flattened_roots = [value for root_tuple in roots for value in root_tuple]
    receipt = {
        "namespace": NAMESPACE,
        "status": "ATTEMPTED",
        "mode": "official",
        "root_tuples": [list(item) for item in roots],
        "all_nine_roots": flattened_roots,
        "scene_roots": list(OFFICIAL_SCENE_ROOTS),
        "depth_roots": list(OFFICIAL_DEPTH_ROOTS),
        "order_roots": list(OFFICIAL_ORDER_ROOTS),
        "config": dataclasses.asdict(config),
        "arms": list(arm_names),
        "protocol": protocol,
        "source_hashes": source_hashes,
        "artifacts_directory": str(artifacts_dir),
        "result_path": str(result_path),
        "environment_at_reservation": _environment_receipt(),
    }
    descriptor = _write_json_exclusive(OFFICIAL_ATTEMPT, receipt)
    return {"receipt": receipt, "descriptor": descriptor}


def _save_ply(path: Path, model: InverseProjectionFiber, labels: torch.Tensor) -> dict[str, Any]:
    palette = model.source_means2d.new_tensor(
        [
            [0.90, 0.20, 0.20],
            [0.20, 0.70, 0.25],
            [0.20, 0.35, 0.90],
            [0.90, 0.70, 0.15],
            [0.65, 0.25, 0.80],
            [0.10, 0.75, 0.75],
            [0.95, 0.45, 0.10],
            [0.45, 0.45, 0.45],
        ]
    )
    safe = labels.clamp_min(0)
    colors = palette[safe]
    colors = torch.where((labels >= 0)[:, None], colors, colors.new_tensor([0.05, 0.05, 0.05]))
    model.as_gaussians(colors=colors, opacity=0.35).save_ply(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _ring_camera(view: int) -> Camera:
    if view < 0 or view >= N_FIT_VIEWS + N_HELDOUT_VIEWS:
        raise ValueError("ring-camera index outside frozen domain")
    angle = 2.0 * math.pi * view / (N_FIT_VIEWS + N_HELDOUT_VIEWS)
    eye = torch.tensor([2.4 * math.cos(angle), 0.6 * math.sin(2.0 * angle), 2.4 * math.sin(angle)])
    return Camera.look_at(eye, torch.zeros(3), fov_x_deg=45.0, width=64, height=64)


def _moment_split(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    count: int,
    *,
    dilation: float = DILATION,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Split one Gaussian while preserving equal-weight first and second moments exactly."""
    if mean.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError("mean/covariance shapes must be (2,) and (2,2)")
    if count not in {1, 2, 3}:
        raise ValueError("moment split count must be one, two, or three")
    intrinsic = covariance - dilation * torch.eye(2, dtype=covariance.dtype)
    eigenvalues, eigenvectors = torch.linalg.eigh(intrinsic)
    if bool((eigenvalues <= 0).any()):
        raise ValueError("parent intrinsic covariance must be SPD")
    factor = eigenvectors @ torch.diag_embed(eigenvalues.sqrt())
    if count == 1:
        canonical = mean.new_zeros((1, 2))
    elif count == 2:
        canonical = mean.new_tensor([[-1.0, 0.0], [1.0, 0.0]])
    else:
        inverse_sqrt_two = 1.0 / math.sqrt(2.0)
        canonical = mean.new_tensor(
            [
                [math.sqrt(2.0), 0.0],
                [-inverse_sqrt_two, math.sqrt(1.5)],
                [-inverse_sqrt_two, -math.sqrt(1.5)],
            ]
        )
    offsets = SPLIT_FRACTION * (canonical @ factor.T)
    between = offsets.T @ offsets / count
    child_intrinsic = intrinsic - between
    child_covariance = child_intrinsic + dilation * torch.eye(2, dtype=covariance.dtype)
    children_mean = mean[None, :] + offsets
    children_covariance = child_covariance[None, :, :].expand(count, 2, 2).clone()
    reconstructed_mean = children_mean.mean(dim=0)
    centered = children_mean - reconstructed_mean
    reconstructed_covariance = children_covariance.mean(dim=0) + centered.T @ centered / count
    mean_error = float((reconstructed_mean - mean).abs().max())
    covariance_error = float((reconstructed_covariance - covariance).abs().max())
    minimum_intrinsic_eigenvalue = float(torch.linalg.eigvalsh(child_intrinsic).min())
    if minimum_intrinsic_eigenvalue <= 0:
        raise RuntimeError("moment split produced a non-SPD child intrinsic covariance")
    return (
        children_mean,
        children_covariance,
        {
            "mean_max_abs": mean_error,
            "covariance_max_abs": covariance_error,
            "minimum_child_intrinsic_eigenvalue": minimum_intrinsic_eigenvalue,
        },
    )


def _split_pattern(parent: int, view: int) -> int:
    return 1 + ((parent + view) % 3)


def _make_view_observations(
    parent_means: torch.Tensor,
    parent_covariances: torch.Tensor,
    *,
    view: int,
    generator: torch.Generator,
    add_outliers: bool,
) -> tuple[ObservationGaussians, torch.Tensor, torch.Tensor, dict[str, Any]]:
    means: list[torch.Tensor] = []
    covariances: list[torch.Tensor] = []
    labels: list[int] = []
    child_indices: list[int] = []
    split_counts: list[int] = []
    moment_checks: list[dict[str, float]] = []
    for parent in range(N_PARENTS):
        count = _split_pattern(parent, view)
        split_mean, split_covariance, check = _moment_split(
            parent_means[parent], parent_covariances[parent], count
        )
        means.append(split_mean)
        covariances.append(split_covariance)
        labels.extend([parent] * count)
        child_indices.extend(range(count))
        split_counts.append(count)
        moment_checks.append(check)
    if add_outliers:
        outlier_means = parent_means.new_tensor(
            [[6.0 + 0.4 * view, 7.0 + 0.3 * view], [57.0 - 0.5 * view, 55.0 - 0.2 * view]]
        )
        outlier_intrinsic = parent_means.new_tensor(
            [[[1.4, 0.12], [0.12, 0.9]], [[0.8, -0.08], [-0.08, 1.6]]]
        )
        outlier_covariances = outlier_intrinsic + DILATION * torch.eye(2, dtype=parent_means.dtype)
        means.append(outlier_means)
        covariances.append(outlier_covariances)
        labels.extend([-1, -1])
        child_indices.extend([-1, -1])
    all_means = torch.cat(means)
    all_covariances = torch.cat(covariances)
    all_labels = torch.tensor(labels, dtype=torch.long)
    all_children = torch.tensor(child_indices, dtype=torch.long)
    permutation = torch.randperm(all_means.shape[0], generator=generator)
    all_means = all_means[permutation]
    all_covariances = all_covariances[permutation]
    all_labels = all_labels[permutation]
    all_children = all_children[permutation]
    observation = ObservationGaussians(
        means=all_means,
        covariances=all_covariances,
        capacities=all_means.new_ones(all_means.shape[0]),
        dilation=DILATION,
        capacity_mode="uniform",
    )
    receipt = {
        "view": view,
        "split_counts": split_counts,
        "observation_count": observation.n,
        "outlier_count": OUTLIERS_PER_VIEW if add_outliers else 0,
        "permutation": permutation.tolist(),
        "means_sha256": _tensor_hash(all_means),
        "covariances_sha256": _tensor_hash(all_covariances),
        "moment_mean_max_abs": max(item["mean_max_abs"] for item in moment_checks),
        "moment_covariance_max_abs": max(item["covariance_max_abs"] for item in moment_checks),
        "minimum_child_intrinsic_eigenvalue": min(
            item["minimum_child_intrinsic_eigenvalue"] for item in moment_checks
        ),
    }
    return observation, all_labels, all_children, receipt


def _build_root_inputs(scene_root: int, depth_root: int, order_root: int) -> RootInputs:
    cameras = tuple(_ring_camera(view) for view in range(N_FIT_VIEWS))
    scene_generator = torch.Generator(device="cpu").manual_seed(scene_root)
    gt = make_gt_gaussians(n=N_PARENTS, seed=scene_root, generator=scene_generator)
    gt_means = gt.means.to(dtype=torch.float64, device="cpu")
    rotations = quat_to_rotmat(gt.quats.to(dtype=torch.float64, device="cpu"))
    scales = gt.log_scales.to(dtype=torch.float64, device="cpu").exp()
    factors = rotations * scales[:, None, :]
    gt_covariances = factors @ factors.transpose(-1, -2)
    order_generator = torch.Generator(device="cpu").manual_seed(order_root)
    observations: list[ObservationGaussians] = []
    fit_labels: list[torch.Tensor] = []
    child_indices: list[torch.Tensor] = []
    view_receipts: list[dict[str, Any]] = []
    for view, camera in enumerate(cameras):
        parent_projection = project_covariances_ewa(
            gt_means, gt_covariances, camera, dilation=DILATION
        )
        if not bool((parent_projection.depth > 0).all()):
            raise RuntimeError("synthetic parent projection has non-positive depth")
        observation, labels, children, receipt = _make_view_observations(
            parent_projection.means2d,
            parent_projection.covariances2d,
            view=view,
            generator=order_generator,
            add_outliers=True,
        )
        observations.append(observation)
        fit_labels.append(labels)
        child_indices.append(children)
        view_receipts.append(receipt)
    order_state = order_generator.get_state().clone()
    counts = [observation.n for observation in observations]
    source_view_indices = torch.cat(
        [torch.full((count,), view, dtype=torch.long) for view, count in enumerate(counts)]
    )
    source_component_indices = torch.cat(
        [torch.arange(count, dtype=torch.long) for count in counts]
    )
    source_means = torch.cat([observation.means for observation in observations])
    source_covariances = torch.cat([observation.covariances for observation in observations])
    depth_generator = torch.Generator(device="cpu").manual_seed(depth_root)
    initial_depths = DEPTH_LOWER + (DEPTH_UPPER - DEPTH_LOWER) * torch.rand(
        source_means.shape[0], generator=depth_generator, dtype=torch.float64
    )
    lower_tensor = initial_depths.new_tensor(DEPTH_LOWER)
    upper_tensor = initial_depths.new_tensor(DEPTH_UPPER)
    initial_depths = torch.where(
        initial_depths <= lower_tensor,
        torch.nextafter(lower_tensor, upper_tensor),
        initial_depths,
    )
    candidate = CandidateInputs(
        cameras=cameras,
        observations=tuple(observations),
        source_view_indices=source_view_indices,
        source_component_indices=source_component_indices,
        source_means2d=source_means,
        source_covariances2d=source_covariances,
        initial_depths=initial_depths,
    )
    evaluator = EvaluatorData(
        gt_means=gt_means,
        gt_covariances=gt_covariances,
        fit_parent_labels=tuple(fit_labels),
        fit_child_indices=tuple(child_indices),
        source_parent_labels=torch.cat(fit_labels),
    )
    receipt = {
        "roots": [scene_root, depth_root, order_root],
        "candidate_fields": sorted(CandidateInputs.__dataclass_fields__),
        "view_receipts": view_receipts,
        "cameras": [_camera_receipt(camera) for camera in cameras],
        "source_count": int(source_means.shape[0]),
        "initial_depths_sha256": _tensor_hash(initial_depths),
        "order_state_before_heldout_sha256": _tensor_hash(order_state),
        "heldout_materialized": False,
    }
    return RootInputs(
        roots=(scene_root, depth_root, order_root),
        candidate=candidate,
        evaluator=evaluator,
        heldout_recipe=HeldoutRecipe(order_generator_state=order_state),
        receipt=receipt,
    )


def _materialize_heldout(root: RootInputs) -> HeldoutData:
    generator = torch.Generator(device="cpu")
    generator.set_state(root.heldout_recipe.order_generator_state.clone())
    cameras: list[Camera] = []
    observations: list[ObservationGaussians] = []
    labels: list[torch.Tensor] = []
    receipts: list[dict[str, Any]] = []
    for view in range(N_FIT_VIEWS, N_FIT_VIEWS + N_HELDOUT_VIEWS):
        camera = _ring_camera(view)
        parent = project_covariances_ewa(
            root.evaluator.gt_means,
            root.evaluator.gt_covariances,
            camera,
            dilation=DILATION,
        )
        observation, parent_labels, _children, receipt = _make_view_observations(
            parent.means2d,
            parent.covariances2d,
            view=view,
            generator=generator,
            add_outliers=False,
        )
        cameras.append(camera)
        observations.append(observation)
        labels.append(parent_labels)
        receipts.append(receipt)
    return HeldoutData(
        cameras=tuple(cameras),
        observations=tuple(observations),
        parent_labels=tuple(labels),
        receipt={
            "materialized": True,
            "views": receipts,
            "order_state_after_sha256": _tensor_hash(generator.get_state()),
        },
    )


def _new_fiber(candidate: CandidateInputs) -> InverseProjectionFiber:
    return InverseProjectionFiber(
        cameras=candidate.cameras,
        source_view_indices=candidate.source_view_indices,
        source_component_indices=candidate.source_component_indices,
        source_means2d=candidate.source_means2d,
        source_covariances2d=candidate.source_covariances2d,
        initial_depths=candidate.initial_depths,
        depth_lower=DEPTH_LOWER,
        depth_upper=DEPTH_UPPER,
        dilation=DILATION,
    )


def _observations_with_capacity(
    observations: tuple[ObservationGaussians, ...], mode: Literal["uniform", "area"]
) -> tuple[ObservationGaussians, ...]:
    result: list[ObservationGaussians] = []
    for observation in observations:
        capacity = (
            observation.means.new_ones(observation.n)
            if mode == "uniform"
            else torch.linalg.det(observation.covariances).sqrt()
        )
        result.append(
            ObservationGaussians(
                means=observation.means,
                covariances=observation.covariances,
                capacities=capacity,
                dilation=observation.dilation,
                capacity_mode="uniform" if mode == "uniform" else "footprint_area",
            )
        )
    return tuple(result)


def _source_capacities(
    candidate: CandidateInputs, mode: Literal["uniform", "area"]
) -> torch.Tensor:
    if mode == "uniform":
        return candidate.source_means2d.new_ones(candidate.source_means2d.shape[0])
    return torch.linalg.det(candidate.source_covariances2d).sqrt()


def _source_errors(model: InverseProjectionFiber) -> tuple[float, float]:
    means, covariances, _depth = model.source_projection()
    center = torch.linalg.vector_norm(means - model.source_means2d, dim=-1).max()
    relative = torch.linalg.matrix_norm(
        covariances - model.source_covariances2d, ord="fro", dim=(-2, -1)
    ) / torch.linalg.matrix_norm(model.source_covariances2d, ord="fro", dim=(-2, -1))
    return float(center.detach()), float(relative.detach().max())


def _plan_hardmin(
    model: InverseProjectionFiber,
    observations: tuple[ObservationGaussians, ...],
    *,
    residual_variance: float,
) -> tuple[CorrespondencePlan, ...]:
    plans: list[CorrespondencePlan] = []
    with torch.no_grad():
        for view, (camera, target) in enumerate(zip(model.cameras, observations, strict=True)):
            projection = model.project(camera)
            projected_means, projected_covariances, valid = safe_projection_geometry(
                camera, projection
            )
            cost = pairwise_bhattacharyya_cost(
                projected_means,
                projected_covariances,
                target.means,
                target.covariances,
                residual_variance=residual_variance,
            )
            active = (model.source_view_indices != view) & valid
            real = cost.new_zeros(cost.shape)
            rows = active.nonzero(as_tuple=True)[0]
            columns = cost[rows].argmin(dim=1)
            real[rows, columns] = 1.0
            plans.append(
                CorrespondencePlan(
                    real_mass=real,
                    track_dustbin_mass=cost.new_zeros(model.n),
                    observation_dustbin_mass=None,
                    dustbin_dustbin_mass=None,
                    track_capacities=(model.source_view_indices != view).to(cost),
                    observation_capacities=None,
                    method="hardmin",
                    iterations=1,
                )
            )
    return tuple(plans)


def _plan_oracle(
    model: InverseProjectionFiber,
    observations: tuple[ObservationGaussians, ...],
    source_labels: torch.Tensor,
    target_labels: tuple[torch.Tensor, ...],
) -> tuple[CorrespondencePlan, ...]:
    plans: list[CorrespondencePlan] = []
    for view, target in enumerate(observations):
        real = target.means.new_zeros((model.n, target.n))
        track_dust = target.means.new_zeros(model.n)
        projection = model.project(model.cameras[view])
        _safe_means, _safe_covariances, valid = safe_projection_geometry(
            model.cameras[view], projection
        )
        non_source = model.source_view_indices != view
        active = non_source & valid
        track_dust[non_source & ~valid] = 1.0
        for row in active.nonzero(as_tuple=True)[0].tolist():
            label = int(source_labels[row])
            if label < 0:
                track_dust[row] = 1.0
                continue
            matches = target_labels[view] == label
            if not bool(matches.any()):
                track_dust[row] = 1.0
            else:
                real[row, matches] = 1.0 / int(matches.sum())
        observation_dust = (target_labels[view] < 0).to(target.means)
        plans.append(
            CorrespondencePlan(
                real_mass=real,
                track_dustbin_mass=track_dust,
                observation_dustbin_mass=observation_dust,
                dustbin_dustbin_mass=target.means.new_tensor(0.0),
                track_capacities=(model.source_view_indices != view).to(target.means),
                observation_capacities=target.means.new_ones(target.n),
                method="oracle",
                iterations=1,
            )
        )
    return tuple(plans)


def _mass_normalized_expected_cost(
    real_mass: torch.Tensor,
    cost: torch.Tensor,
) -> torch.Tensor | None:
    if real_mass.shape != cost.shape:
        raise ValueError("real_mass and cost must have the same shape")
    denominator = real_mass.sum()
    if not bool(torch.isfinite(denominator)) or float(denominator) < 0:
        raise ValueError("real_mass must have finite non-negative total mass")
    if float(denominator) == 0:
        return None
    return (real_mass * cost).sum() / denominator


def _weighted_geometry_update(
    model: InverseProjectionFiber,
    observations: tuple[ObservationGaussians, ...],
    plans: tuple[CorrespondencePlan, ...],
    optimizer: torch.optim.Optimizer,
    *,
    residual_variance: float,
) -> float:
    view_real_masses = [float(plan.real_mass.sum()) for plan in plans]
    if any(not math.isfinite(value) or value < 0 for value in view_real_masses):
        raise RuntimeError("custom correspondence planner produced invalid real mass")
    nonempty_view_count = sum(value > 0 for value in view_real_masses)
    if nonempty_view_count == 0:
        raise RuntimeError("custom correspondence planner transported no finite real mass")
    optimizer.zero_grad(set_to_none=True)
    loss = model.depth_logits.new_tensor(0.0)
    for camera, target, plan, view_mass in zip(
        model.cameras,
        observations,
        plans,
        view_real_masses,
        strict=True,
    ):
        if view_mass == 0:
            continue
        projection = model.project(camera)
        projected_means, projected_covariances, valid = safe_projection_geometry(camera, projection)
        supported = plan.real_mass.sum(dim=1) > 0
        if bool((supported & ~valid).any()):
            raise RuntimeError("a supported projection left the valid camera domain during M-step")
        cost = pairwise_bhattacharyya_cost(
            projected_means,
            projected_covariances,
            target.means,
            target.covariances,
            residual_variance=residual_variance,
        )
        contribution = _mass_normalized_expected_cost(plan.real_mass, cost)
        if contribution is None:  # guarded by view_mass above
            raise RuntimeError("non-empty view lost its real mass during objective evaluation")
        loss = loss + contribution
    loss = loss / nonempty_view_count
    loss.backward()
    if any(
        parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    ):
        raise RuntimeError("custom correspondence fit produced invalid gradients")
    optimizer.step()
    validate_fiber_state(model)
    return float(loss.detach())


def _fit_custom(
    model: InverseProjectionFiber,
    observations: tuple[ObservationGaussians, ...],
    config: ScientificConfig,
    planner: Any,
) -> tuple[tuple[CorrespondencePlan, ...], tuple[dict[str, Any], ...]]:
    temperatures = exponential_schedule(
        config.temperature_start, config.temperature_stop, config.outer_steps
    )
    residuals = exponential_schedule(
        config.residual_variance_start,
        config.residual_variance_stop,
        config.outer_steps,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, Any]] = []
    plans: tuple[CorrespondencePlan, ...] = ()
    for temperature, residual_variance in zip(temperatures, residuals, strict=True):
        del temperature  # the hard/oracle E-steps are temperature independent by definition
        plans = planner(model, residual_variance)
        loss = math.nan
        for _ in range(config.geometry_steps):
            loss = _weighted_geometry_update(
                model,
                observations,
                plans,
                optimizer,
                residual_variance=residual_variance,
            )
        center, covariance = _source_errors(model)
        history.append(
            {
                "residual_variance": residual_variance,
                "loss": loss,
                "real_mass": sum(float(plan.real_mass.sum()) for plan in plans),
                "source_center_error": center,
                "source_covariance_relative_error": covariance,
            }
        )
    validate_fiber_state(model)
    final_plans = planner(model, residuals[-1])
    validate_fiber_state(model)
    return final_plans, tuple(history)


def _fit_config(config: ScientificConfig, assignment: str) -> FiberFitConfig:
    return FiberFitConfig(
        temperatures=exponential_schedule(
            config.temperature_start, config.temperature_stop, config.outer_steps
        ),
        residual_variances=exponential_schedule(
            config.residual_variance_start,
            config.residual_variance_stop,
            config.outer_steps,
        ),
        geometry_steps=config.geometry_steps,
        learning_rate=config.learning_rate,
        assignment=assignment,
        dustbin_cost=config.dustbin_cost,
        marginal_penalty=config.marginal_penalty,
        sinkhorn_iterations=config.sinkhorn_iterations,
        sinkhorn_tolerance=config.sinkhorn_tolerance,
        track_batch_size=256,
        max_grad_norm=None,
        source_center_tolerance=1e-8,
        source_covariance_tolerance=1e-8,
    )


def _history_dicts(result: FiberFitResult) -> tuple[dict[str, Any], ...]:
    return tuple(dataclasses.asdict(item) for item in result.history)


def _run_arm(
    name: str,
    candidate: CandidateInputs,
    config: ScientificConfig,
    *,
    oracle_evaluator: EvaluatorData | None = None,
) -> ArmState:
    if name not in ARM_NAMES:
        raise ValueError(f"unknown arm {name!r}")
    if name != "oracle" and oracle_evaluator is not None:
        raise ValueError("evaluator labels may only be passed to the oracle arm")
    model = _new_fiber(candidate)
    initial_state_sha256 = _tensor_mapping_hash(dict(model.state_dict()))
    started = time.perf_counter()
    if name == "hardmin":
        observations = _observations_with_capacity(candidate.observations, "uniform")
        plans, history = _fit_custom(
            model,
            observations,
            config,
            lambda current, residual: _plan_hardmin(
                current, observations, residual_variance=residual
            ),
        )
    elif name == "oracle":
        if oracle_evaluator is None:
            raise ValueError("the oracle arm requires evaluator labels")
        observations = _observations_with_capacity(candidate.observations, "uniform")
        plans, history = _fit_custom(
            model,
            observations,
            config,
            lambda current, _residual: _plan_oracle(
                current,
                observations,
                oracle_evaluator.source_parent_labels,
                oracle_evaluator.fit_parent_labels,
            ),
        )
    else:
        capacity_mode: Literal["uniform", "area"] = (
            "area" if name in {"uot_area", "shuffled_view"} else "uniform"
        )
        base_observations = _observations_with_capacity(candidate.observations, capacity_mode)
        if name == "shuffled_view":
            observations = tuple(
                base_observations[(view + 1) % N_FIT_VIEWS] for view in range(N_FIT_VIEWS)
            )
        else:
            observations = base_observations
        assignment = "row_softmax" if name == "row" else "unbalanced_sinkhorn"
        result = fit_fiber_correspondence(
            model,
            observations,
            config=_fit_config(config, assignment),
            track_capacities=_source_capacities(candidate, capacity_mode),
        )
        plans = result.plans
        history = _history_dicts(result)
    validate_fiber_state(model)
    return ArmState(
        name=name,
        model=model,
        plans=plans,
        history=history,
        wall_time_seconds=time.perf_counter() - started,
        initial_state_sha256=initial_state_sha256,
    )


def _conditional_entropy(real_mass: torch.Tensor, active: torch.Tensor) -> list[float]:
    row_mass = real_mass.sum(dim=1)
    valid = active & (row_mass > torch.finfo(real_mass.dtype).tiny)
    if not bool(valid.any()):
        return []
    probabilities = real_mass[valid] / row_mass[valid, None]
    return (
        -(probabilities * probabilities.clamp_min(torch.finfo(real_mass.dtype).tiny).log()).sum(
            dim=1
        )
    ).tolist()


def _association_metrics(
    state: ArmState,
    root: RootInputs,
    target_parent_labels: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    source_labels = root.evaluator.source_parent_labels
    view_purity: list[float] = []
    view_real_mass: list[float] = []
    view_normalized_real_mass: list[float] = []
    view_entropy: list[float] = []
    track_outlier_recall: list[float] = []
    track_inlier_fpr: list[float] = []
    observation_outlier_recall: list[float] = []
    observation_inlier_fpr: list[float] = []
    realized_track_outlier_fraction: list[float] = []
    realized_track_inlier_fraction: list[float] = []
    realized_observation_outlier_fraction: list[float] = []
    realized_observation_inlier_fraction: list[float] = []
    correct_support = torch.zeros((N_PARENTS, N_FIT_VIEWS), dtype=torch.float64)
    parent_transport = torch.zeros((N_PARENTS, N_FIT_VIEWS), dtype=torch.float64)
    supported_views = torch.zeros(state.model.n, dtype=torch.long)
    for view, (plan, target_labels) in enumerate(
        zip(state.plans, target_parent_labels, strict=True)
    ):
        active = state.model.source_view_indices != view
        source_inlier = source_labels >= 0
        real = plan.real_mass
        same_parent = source_labels[:, None] == target_labels[None, :]
        same_parent &= source_inlier[:, None] & (target_labels >= 0)[None, :]
        same_parent &= active[:, None]
        correct = float(real[same_parent].sum())
        inlier_real = float(real[active & source_inlier].sum())
        view_purity.append(correct / inlier_real if inlier_real > 0 else 0.0)
        raw_real = float(real[active].sum())
        view_real_mass.append(raw_real)
        plan_total = (
            float(plan.augmented_mass.sum())
            if plan.augmented_mass is not None
            else float(plan.track_row_mass.sum())
        )
        view_normalized_real_mass.append(raw_real / plan_total if plan_total > 0 else 0.0)
        if plan.method == "unbalanced_sinkhorn":
            targets = plan.augmented_target_marginals
            if targets is None:
                raise RuntimeError("UOT plan is missing its declared target marginals")
            declared_track = targets[0][:-1]
            declared_observation = targets[1][:-1]
        else:
            declared_track = plan.track_capacities
            declared_observation = plan.observation_capacities
        row_total = plan.track_row_mass
        outlier_active = active & ~source_inlier
        inlier_active = active & source_inlier
        outlier_track_declared = float(declared_track[outlier_active].sum())
        inlier_track_declared = float(declared_track[inlier_active].sum())
        track_outlier_recall.append(
            float(plan.track_dustbin_mass[outlier_active].sum()) / outlier_track_declared
            if outlier_track_declared > 0
            else 0.0
        )
        track_inlier_fpr.append(
            float(plan.track_dustbin_mass[inlier_active].sum()) / inlier_track_declared
            if inlier_track_declared > 0
            else 0.0
        )
        outlier_track_realized = float(row_total[outlier_active].sum())
        inlier_track_realized = float(row_total[inlier_active].sum())
        realized_track_outlier_fraction.append(
            float(plan.track_dustbin_mass[outlier_active].sum()) / outlier_track_realized
            if outlier_track_realized > 0
            else 0.0
        )
        realized_track_inlier_fraction.append(
            float(plan.track_dustbin_mass[inlier_active].sum()) / inlier_track_realized
            if inlier_track_realized > 0
            else 0.0
        )
        if plan.observation_dustbin_mass is not None:
            column_total = real.sum(dim=0) + plan.observation_dustbin_mass
            target_outlier = target_labels < 0
            if declared_observation is None:
                raise RuntimeError("two-sided plan is missing declared observation capacity")
            outlier_declared = float(declared_observation[target_outlier].sum())
            inlier_declared = float(declared_observation[~target_outlier].sum())
            observation_outlier_recall.append(
                float(plan.observation_dustbin_mass[target_outlier].sum()) / outlier_declared
                if outlier_declared > 0
                else 0.0
            )
            observation_inlier_fpr.append(
                float(plan.observation_dustbin_mass[~target_outlier].sum()) / inlier_declared
                if inlier_declared > 0
                else 0.0
            )
            outlier_realized = float(column_total[target_outlier].sum())
            inlier_realized = float(column_total[~target_outlier].sum())
            realized_observation_outlier_fraction.append(
                float(plan.observation_dustbin_mass[target_outlier].sum()) / outlier_realized
                if outlier_realized > 0
                else 0.0
            )
            realized_observation_inlier_fraction.append(
                float(plan.observation_dustbin_mass[~target_outlier].sum()) / inlier_realized
                if inlier_realized > 0
                else 0.0
            )
        entropy = _conditional_entropy(real, active)
        view_entropy.append(float(np.mean(entropy)) if entropy else 0.0)
        row_real_fraction = real.sum(dim=1) / row_total.clamp_min(torch.finfo(real.dtype).tiny)
        supported_views += (active & (row_real_fraction >= 0.20)).long()
        for parent in range(N_PARENTS):
            rows = active & (source_labels == parent)
            declared = float(declared_track[rows].sum())
            columns = target_labels == parent
            if declared > 0:
                parent_transport[parent, view] = float(real[rows].sum()) / declared
                if bool(columns.any()):
                    correct_support[parent, view] = float(real[rows][:, columns].sum()) / declared
    completeness = float((correct_support >= CORRECT_SUPPORT_FRACTION).all(dim=1).double().mean())
    parent_mass = parent_transport.mean(dim=1)
    mean_parent_mass = parent_mass.mean()
    parent_cv = (
        float(parent_mass.std(unbiased=False) / mean_parent_mass)
        if float(mean_parent_mass) > 0
        else sys.float_info.max
    )
    track_outlier_mean = float(np.mean(track_outlier_recall))
    track_inlier_mean = float(np.mean(track_inlier_fpr))
    observation_outlier_mean = (
        float(np.mean(observation_outlier_recall)) if observation_outlier_recall else None
    )
    observation_inlier_mean = (
        float(np.mean(observation_inlier_fpr)) if observation_inlier_fpr else None
    )
    combined_outlier = [track_outlier_mean]
    combined_inlier = [track_inlier_mean]
    if observation_outlier_mean is not None:
        combined_outlier.append(observation_outlier_mean)
    if observation_inlier_mean is not None:
        combined_inlier.append(observation_inlier_mean)
    return {
        "parent_purity": float(np.mean(view_purity)),
        "parent_purity_by_view": view_purity,
        "parent_completeness": completeness,
        "capacity_normalized_correct_support_threshold": CORRECT_SUPPORT_FRACTION,
        "capacity_normalized_correct_support_by_parent_view": correct_support.tolist(),
        "outlier_dust_recall": float(np.mean(combined_outlier)),
        "dust_gate_denominator": "declared_target_marginal_capacity",
        "outlier_track_dust_recall": track_outlier_mean,
        "outlier_track_dust_recall_by_view": track_outlier_recall,
        "realized_conditional_outlier_track_dust_fraction_by_view": (
            realized_track_outlier_fraction
        ),
        "outlier_observation_dust_recall": observation_outlier_mean,
        "outlier_observation_dust_recall_by_view": observation_outlier_recall,
        "realized_conditional_outlier_observation_dust_fraction_by_view": (
            realized_observation_outlier_fraction
        ),
        "inlier_dust_false_positive_rate": float(np.mean(combined_inlier)),
        "inlier_track_dust_false_positive_rate": track_inlier_mean,
        "inlier_track_dust_false_positive_rate_by_view": track_inlier_fpr,
        "realized_conditional_inlier_track_dust_fraction_by_view": (realized_track_inlier_fraction),
        "inlier_observation_dust_false_positive_rate": observation_inlier_mean,
        "inlier_observation_dust_false_positive_rate_by_view": observation_inlier_fpr,
        "realized_conditional_inlier_observation_dust_fraction_by_view": (
            realized_observation_inlier_fraction
        ),
        "conditional_association_entropy": float(np.mean(view_entropy)),
        "conditional_association_entropy_by_view": view_entropy,
        "real_transported_mass": float(np.mean(view_normalized_real_mass)),
        "raw_real_transported_mass_by_view": view_real_mass,
        "normalized_real_transported_mass_by_view": view_normalized_real_mass,
        "parent_transported_mass": parent_mass.tolist(),
        "capacity_normalized_parent_transported_mass_by_view": parent_transport.tolist(),
        "parent_mass_coefficient_of_variation": parent_cv,
        "parent_mass_coefficient_of_variation_defined": float(mean_parent_mass) > 0,
        "support_view_histogram": torch.bincount(
            supported_views, minlength=N_FIT_VIEWS + 1
        ).tolist(),
    }


def _uot_mass_diagnostics(state: ArmState) -> dict[str, Any]:
    if any(plan.method != "unbalanced_sinkhorn" for plan in state.plans):
        return {"applicable": False, "pass": True, "views": []}
    views: list[dict[str, Any]] = []
    passed = True
    for view, plan in enumerate(state.plans):
        augmented = plan.augmented_mass
        targets = plan.augmented_target_marginals
        if augmented is None or targets is None or plan.fixed_point_residual is None:
            views.append({"view": view, "valid": False})
            passed = False
            continue
        target_row, target_column = targets
        realized_row = augmented.sum(dim=1)
        realized_column = augmented.sum(dim=0)
        declared_track = float(target_row[:-1].sum())
        declared_observation = float(target_column[:-1].sum())
        declared_augmented = float(target_row.sum())
        realized_track = float(realized_row[:-1].sum())
        realized_observation = float(realized_column[:-1].sum())
        realized_augmented = float(augmented.sum())
        ratios = {
            "track_row": realized_track / declared_track if declared_track > 0 else 0.0,
            "observation_column": (
                realized_observation / declared_observation if declared_observation > 0 else 0.0
            ),
            "augmented": (
                realized_augmented / declared_augmented if declared_augmented > 0 else 0.0
            ),
        }
        row_positive = target_row > 0
        column_positive = target_column > 0
        row_relative = (realized_row[row_positive] - target_row[row_positive]).abs() / target_row[
            row_positive
        ]
        column_relative = (
            realized_column[column_positive] - target_column[column_positive]
        ).abs() / target_column[column_positive]
        max_relative_error = float(torch.cat([row_relative, column_relative]).max())
        fixed_point_residual = float(plan.fixed_point_residual)
        checks = {
            "mass_ratios_in_range": all(
                UOT_MIN_MASS_RATIO <= value <= UOT_MAX_MASS_RATIO for value in ratios.values()
            ),
            "marginal_relative_error_bounded": (
                max_relative_error <= UOT_MAX_MARGINAL_RELATIVE_ERROR
            ),
            "fixed_point_residual_bounded": (fixed_point_residual <= UOT_MAX_FIXED_POINT_RESIDUAL),
        }
        view_pass = all(checks.values())
        passed &= view_pass
        views.append(
            {
                "view": view,
                "valid": True,
                "declared_track_row_mass": declared_track,
                "realized_track_row_mass": realized_track,
                "declared_observation_column_mass": declared_observation,
                "realized_observation_column_mass": realized_observation,
                "declared_augmented_mass": declared_augmented,
                "realized_augmented_mass": realized_augmented,
                "realized_to_declared_ratios": ratios,
                "max_marginal_relative_error": max_relative_error,
                "fixed_point_residual": fixed_point_residual,
                "checks": checks,
                "pass": view_pass,
            }
        )
    return {
        "applicable": True,
        "thresholds": {
            "minimum_mass_ratio": UOT_MIN_MASS_RATIO,
            "maximum_mass_ratio": UOT_MAX_MASS_RATIO,
            "maximum_marginal_relative_error": UOT_MAX_MARGINAL_RELATIVE_ERROR,
            "maximum_fixed_point_residual": UOT_MAX_FIXED_POINT_RESIDUAL,
        },
        "views": views,
        "pass": passed,
    }


def _geometry_metrics(state: ArmState, root: RootInputs) -> dict[str, Any]:
    means, covariances = state.model.means_covariances()
    labels = root.evaluator.source_parent_labels
    inlier = labels >= 0
    target_means = root.evaluator.gt_means[labels[inlier]]
    target_covariances = root.evaluator.gt_covariances[labels[inlier]]
    center = torch.linalg.vector_norm(means[inlier] - target_means, dim=-1)
    covariance = spd_affine_invariant_squared(covariances[inlier], target_covariances).sqrt()
    center_source, covariance_source = _source_errors(state.model)
    depths = state.model.depths()
    components = radius_connected_components(
        means.detach(), torch.arange(state.model.n), radius=0.01
    )
    pure = 0
    for component in components:
        component_labels = {int(labels[index]) for index in component}
        pure += int(len(component_labels) == 1)
    finite = bool(
        torch.isfinite(means).all()
        and torch.isfinite(covariances).all()
        and (torch.linalg.eigvalsh(covariances) > 0).all()
        and torch.isfinite(depths).all()
    )
    return {
        "center_error_p90": float(torch.quantile(center.detach(), 0.9)),
        "covariance_affine_invariant_median": float(torch.median(covariance.detach())),
        "source_center_max_px": center_source,
        "source_covariance_relative_max": covariance_source,
        "depth_min": float(depths.detach().min()),
        "depth_max": float(depths.detach().max()),
        "depth_bound_incidence": int(((depths <= DEPTH_LOWER) | (depths >= DEPTH_UPPER)).sum()),
        "proximity_cluster_count": len(components),
        "proximity_cluster_purity": pure / len(components) if components else 0.0,
        "finite_spd": finite,
    }


def _heldout_metrics(
    state: ArmState,
    root: RootInputs,
    heldout: HeldoutData,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    labels = root.evaluator.source_parent_labels
    inlier = labels >= 0
    correct: list[torch.Tensor] = []
    denominators: list[int] = []
    invalid_incidence: list[int] = []
    invalid_inlier_incidence: list[int] = []
    arrays: dict[str, torch.Tensor] = {}
    for view, (camera, target, target_labels) in enumerate(
        zip(heldout.cameras, heldout.observations, heldout.parent_labels, strict=True)
    ):
        projection = state.model.project(camera)
        projected_means, projected_covariances, valid = safe_projection_geometry(camera, projection)
        cost = pairwise_bhattacharyya_cost(
            projected_means,
            projected_covariances,
            target.means,
            target.covariances,
            residual_variance=FROZEN_CONFIG.residual_variance_stop,
        )
        assignment = torch.full((state.model.n,), -1, dtype=torch.long, device=cost.device)
        assignment[valid] = cost[valid].argmin(dim=1)
        evaluable = inlier & valid
        view_correct = target_labels[assignment[evaluable]] == labels[evaluable]
        correct.append(view_correct)
        denominators.append(int(evaluable.sum()))
        invalid_incidence.append(int((~valid).sum()))
        invalid_inlier_incidence.append(int((inlier & ~valid).sum()))
        arrays[f"heldout_view_{view}_cost"] = cost
        arrays[f"heldout_view_{view}_assignment"] = assignment
        arrays[f"heldout_view_{view}_projection_valid"] = valid
        arrays[f"heldout_view_{view}_evaluable"] = evaluable
    correctness = torch.cat(correct)
    denominator = int(correctness.numel())
    return {
        "heldout_parent_assignment_accuracy": (
            float(correctness.double().mean()) if denominator else 0.0
        ),
        "heldout_assignment_denominator": denominator,
        "heldout_assignment_denominator_by_view": denominators,
        "heldout_invalid_projection_incidence": sum(invalid_incidence),
        "heldout_invalid_projection_incidence_by_view": invalid_incidence,
        "heldout_invalid_inlier_projection_incidence": sum(invalid_inlier_incidence),
        "heldout_invalid_inlier_projection_incidence_by_view": invalid_inlier_incidence,
    }, arrays


def _model_arrays(prefix: str, model: InverseProjectionFiber) -> dict[str, torch.Tensor]:
    means, covariances = model.means_covariances()
    source_means, source_covariances, source_depths = model.source_projection()
    return {
        f"{prefix}_means": means.detach(),
        f"{prefix}_covariances": covariances.detach(),
        f"{prefix}_depth_logits": model.depth_logits.detach(),
        f"{prefix}_cross": model.cross.detach(),
        f"{prefix}_log_ray_scale": model.log_ray_scale.detach(),
        f"{prefix}_source_means": source_means.detach(),
        f"{prefix}_source_covariances": source_covariances.detach(),
        f"{prefix}_source_depths": source_depths.detach(),
    }


def _plan_arrays(prefix: str, plans: tuple[CorrespondencePlan, ...]) -> dict[str, torch.Tensor]:
    arrays: dict[str, torch.Tensor] = {}
    for view, plan in enumerate(plans):
        stem = f"{prefix}_view_{view}"
        arrays[f"{stem}_real_mass"] = plan.real_mass.detach()
        arrays[f"{stem}_track_dustbin_mass"] = plan.track_dustbin_mass.detach()
        arrays[f"{stem}_track_capacities"] = plan.track_capacities.detach()
        arrays[f"{stem}_fixed_point_residual"] = plan.real_mass.new_tensor(
            [-1.0 if plan.fixed_point_residual is None else plan.fixed_point_residual]
        )
        if plan.observation_dustbin_mass is not None:
            arrays[f"{stem}_observation_dustbin_mass"] = plan.observation_dustbin_mass.detach()
        if plan.dustbin_dustbin_mass is not None:
            arrays[f"{stem}_dustbin_dustbin_mass"] = plan.dustbin_dustbin_mass.detach().reshape(1)
        if plan.observation_capacities is not None:
            arrays[f"{stem}_observation_capacities"] = plan.observation_capacities.detach()
        target_marginals = plan.augmented_target_marginals
        if target_marginals is not None:
            arrays[f"{stem}_augmented_target_row_marginal"] = target_marginals[0].detach()
            arrays[f"{stem}_augmented_target_column_marginal"] = target_marginals[1].detach()
    return arrays


def _arm_numerical_validity(state: ArmState) -> dict[str, Any]:
    plan_tensors = _plan_arrays(state.name, state.plans)
    plans_finite = all(bool(torch.isfinite(value).all()) for value in plan_tensors.values())
    history_scalars = [
        float(value)
        for step in state.history
        for value in step.values()
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    history_finite = all(math.isfinite(value) for value in history_scalars)
    return {
        "plans_finite": plans_finite,
        "history_finite": history_finite,
        "pass": plans_finite and history_finite,
    }


def _candidate_input_arrays(candidate: CandidateInputs) -> dict[str, torch.Tensor]:
    arrays = {
        "source_view_indices": candidate.source_view_indices,
        "source_component_indices": candidate.source_component_indices,
        "source_means2d": candidate.source_means2d,
        "source_covariances2d": candidate.source_covariances2d,
        "initial_depths": candidate.initial_depths,
    }
    for view, observation in enumerate(candidate.observations):
        arrays[f"fit_view_{view}_means"] = observation.means
        arrays[f"fit_view_{view}_covariances"] = observation.covariances
    return arrays


def _released_evaluator_arrays(root: RootInputs) -> dict[str, torch.Tensor]:
    evaluator = root.evaluator
    arrays = {
        "source_parent_labels": evaluator.source_parent_labels,
        "gt_means": evaluator.gt_means,
        "gt_covariances": evaluator.gt_covariances,
    }
    for view in range(N_FIT_VIEWS):
        arrays[f"fit_view_{view}_parent_labels"] = evaluator.fit_parent_labels[view]
        arrays[f"fit_view_{view}_child_indices"] = evaluator.fit_child_indices[view]
    return arrays


def _released_target_labels(
    name: str,
    evaluator: EvaluatorData,
) -> tuple[torch.Tensor, ...]:
    if name == "shuffled_view":
        return tuple(
            evaluator.fit_parent_labels[(view + 1) % N_FIT_VIEWS] for view in range(N_FIT_VIEWS)
        )
    return evaluator.fit_parent_labels


def _run_root(
    root: RootInputs,
    root_directory: Path,
    config: ScientificConfig,
    arm_names: tuple[str, ...],
) -> dict[str, Any]:
    root_directory.mkdir(parents=True, exist_ok=False)
    initial = _new_fiber(root.candidate)
    common_initial_sha256 = _tensor_mapping_hash(dict(initial.state_dict()))
    neutral_labels = torch.full_like(root.candidate.source_view_indices, -1)
    artifacts: dict[str, Any] = {
        "initial_ply": _save_ply(root_directory / "gaussians_init.ply", initial, neutral_labels)
    }
    evidence: dict[str, torch.Tensor | np.ndarray] = _candidate_input_arrays(root.candidate)
    evidence.update(_model_arrays("common_initial", initial))
    states: dict[str, ArmState] = {}
    non_oracle_names = tuple(name for name in arm_names if name != "oracle")
    for name in non_oracle_names:
        if name not in ARM_NAMES:
            raise ValueError(f"unknown arm {name!r}")
        states[name] = _run_arm(
            name,
            root.candidate,
            config,
        )

    # This mechanical boundary binds every non-oracle outcome before invoking either the oracle
    # or any evaluator-label path. Invalid projections are structural zero rows for hardmin and
    # are routed to dust by the soft transport arms; both share the exact same validity mask.
    nonoracle_state_hashes_before_evaluator = {
        name: _tensor_mapping_hash(dict(states[name].model.state_dict()))
        for name in non_oracle_names
    }
    nonoracle_plan_hashes_before_evaluator = {
        name: _tensor_mapping_hash(_plan_arrays(name, states[name].plans))
        for name in non_oracle_names
    }

    if "oracle" in arm_names:
        states["oracle"] = _run_arm(
            "oracle",
            root.candidate,
            config,
            oracle_evaluator=root.evaluator,
        )
    frozen_state_hashes = dict(nonoracle_state_hashes_before_evaluator)
    frozen_plan_hashes = dict(nonoracle_plan_hashes_before_evaluator)
    if "oracle" in states:
        frozen_state_hashes["oracle"] = _tensor_mapping_hash(
            dict(states["oracle"].model.state_dict())
        )
        frozen_plan_hashes["oracle"] = _tensor_mapping_hash(
            _plan_arrays("oracle", states["oracle"].plans)
        )

    evidence.update(_released_evaluator_arrays(root))
    arm_summaries: dict[str, Any] = {}
    for name in arm_names:
        state = states[name]
        association = _association_metrics(
            state,
            root,
            _released_target_labels(name, root.evaluator),
        )
        geometry = _geometry_metrics(state, root)
        arm_summaries[name] = {
            "association": association,
            "geometry": geometry,
            "transport_mass_diagnostics": _uot_mass_diagnostics(state),
            "numerical_validity": _arm_numerical_validity(state),
            "initial_state_sha256": state.initial_state_sha256,
            "initial_state_matches_common": state.initial_state_sha256 == common_initial_sha256,
            "history": list(state.history),
            "wall_time_seconds": state.wall_time_seconds,
        }
        evidence.update(_model_arrays(name, state.model))
        evidence.update(_plan_arrays(name, state.plans))
        artifacts[f"{name}_ply"] = _save_ply(
            root_directory / f"{name}.ply", state.model, root.evaluator.source_parent_labels
        )

    # This is the first point at which held-out camera geometry and labels exist.
    heldout = _materialize_heldout(root)
    evidence["heldout_release_marker"] = np.array([1], dtype=np.int8)
    for view, observation in enumerate(heldout.observations):
        evidence[f"heldout_view_{view}_means"] = observation.means
        evidence[f"heldout_view_{view}_covariances"] = observation.covariances
        evidence[f"heldout_view_{view}_parent_labels"] = heldout.parent_labels[view]
    for name, state in states.items():
        heldout_summary, heldout_arrays = _heldout_metrics(state, root, heldout)
        arm_summaries[name]["heldout"] = heldout_summary
        evidence.update({f"{name}_{key}": value for key, value in heldout_arrays.items()})
    artifacts["evidence_npz"] = _write_npz_exclusive(root_directory / "evidence.npz", evidence)
    summary = {
        "roots": list(root.roots),
        "input_receipt": root.receipt,
        "common_initial_state_sha256": common_initial_sha256,
        "nonoracle_state_hashes_before_evaluator": nonoracle_state_hashes_before_evaluator,
        "nonoracle_plan_hashes_before_evaluator": nonoracle_plan_hashes_before_evaluator,
        "frozen_state_hashes_before_heldout": frozen_state_hashes,
        "frozen_plan_hashes_before_heldout": frozen_plan_hashes,
        "projection_validity_arm_semantics": {
            "shared_mask": True,
            "hardmin_invalid_row": "structural_zero",
            "soft_transport_invalid_row": "dust_routed",
        },
        "heldout_release": heldout.receipt,
        "arms": arm_summaries,
        "artifacts": artifacts,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    artifacts["root_summary_json"] = _write_json_exclusive(
        root_directory / "ROOT_RESULT.json", summary
    )
    return summary


def _mean_metric(results: list[dict[str, Any]], arm: str, metric: str) -> float:
    return float(np.mean([root["arms"][arm]["association"][metric] for root in results]))


def _transport_arm_accepted(results: list[dict[str, Any]], arm: str) -> bool:
    return bool(
        all(
            (association := root["arms"][arm]["association"])["parent_purity"] >= 0.90
            and association["parent_completeness"] >= 0.90
            and association["outlier_track_dust_recall"] >= 0.80
            and association["outlier_observation_dust_recall"] >= 0.80
            and association["inlier_track_dust_false_positive_rate"] <= 0.20
            and association["inlier_observation_dust_false_positive_rate"] <= 0.20
            and root["arms"][arm]["transport_mass_diagnostics"]["pass"]
            for root in results
        )
    )


def _synthetic_gates(results: list[dict[str, Any]]) -> dict[str, Any]:
    required_arms = set(ARM_NAMES)
    complete_suite = len(results) == 3 and all(
        required_arms.issubset(root["arms"]) for root in results
    )
    if not complete_suite:
        return {
            "eligible": False,
            "reason": "scientific gates require three roots and all six frozen arms",
            "overall": {"pass": False},
        }
    moment_ok = all(
        max(view["moment_mean_max_abs"] for view in root["input_receipt"]["view_receipts"]) <= 1e-10
        and max(
            view["moment_covariance_max_abs"] for view in root["input_receipt"]["view_receipts"]
        )
        <= 1e-10
        for root in results
    )
    all_finite = all(
        arm["geometry"]["finite_spd"] and arm["numerical_validity"]["pass"]
        for root in results
        for arm in root["arms"].values()
    )
    all_initial_states_paired = all(
        arm["initial_state_matches_common"] for root in results for arm in root["arms"].values()
    )
    area_geometry = [root["arms"]["uot_area"]["geometry"] for root in results]
    oracle_association = [root["arms"]["oracle"]["association"] for root in results]
    validity_checks = {
        "all_arms_finite_spd": all_finite,
        "all_arms_share_common_initial_state": all_initial_states_paired,
        "uot_area_source_center_le_1e-8": all(
            item["source_center_max_px"] <= 1e-8 for item in area_geometry
        ),
        "uot_area_source_covariance_le_1e-8": all(
            item["source_covariance_relative_max"] <= 1e-8 for item in area_geometry
        ),
        "all_depths_strictly_bounded": all(
            arm["geometry"]["depth_bound_incidence"] == 0
            for root in results
            for arm in root["arms"].values()
        ),
        "moment_splits_rederive": moment_ok,
        "oracle_purity_ge_0_99": all(item["parent_purity"] >= 0.99 for item in oracle_association),
        "oracle_completeness_ge_0_99": all(
            item["parent_completeness"] >= 0.99 for item in oracle_association
        ),
    }
    validity = {"checks": validity_checks, "pass": all(validity_checks.values())}
    area = [root["arms"]["uot_area"]["association"] for root in results]
    uniform = [root["arms"]["uot_uniform"]["association"] for root in results]
    hard = [root["arms"]["hardmin"]["association"] for root in results]
    row = [root["arms"]["row"]["association"] for root in results]
    shuffled = [root["arms"]["shuffled_view"]["association"] for root in results]
    transport_mass_checks = {
        "uot_uniform_all_roots": all(
            root["arms"]["uot_uniform"]["transport_mass_diagnostics"]["pass"] for root in results
        ),
        "uot_area_all_roots": all(
            root["arms"]["uot_area"]["transport_mass_diagnostics"]["pass"] for root in results
        ),
    }
    transport_mass = {
        "checks": transport_mass_checks,
        "pass": all(transport_mass_checks.values()),
    }
    absolute_checks = {
        "purity_all_roots_ge_0_90": all(item["parent_purity"] >= 0.90 for item in area),
        "completeness_all_roots_ge_0_90": all(item["parent_completeness"] >= 0.90 for item in area),
        "mean_track_outlier_dust_recall_ge_0_80": _mean_metric(
            results, "uot_area", "outlier_track_dust_recall"
        )
        >= 0.80,
        "mean_observation_outlier_dust_recall_ge_0_80": _mean_metric(
            results, "uot_area", "outlier_observation_dust_recall"
        )
        >= 0.80,
        "mean_track_inlier_dust_fpr_le_0_20": _mean_metric(
            results, "uot_area", "inlier_track_dust_false_positive_rate"
        )
        <= 0.20,
        "mean_observation_inlier_dust_fpr_le_0_20": _mean_metric(
            results, "uot_area", "inlier_observation_dust_false_positive_rate"
        )
        <= 0.20,
        "uot_area_mass_diagnostics_all_roots": transport_mass_checks["uot_area_all_roots"],
    }
    absolute = {"checks": absolute_checks, "pass": all(absolute_checks.values())}
    area_purity = np.array([item["parent_purity"] for item in area])
    hard_purity = np.array([item["parent_purity"] for item in hard])
    row_purity = np.array([item["parent_purity"] for item in row])
    soft_checks = {
        "mean_area_minus_hard_ge_0_15": float(np.mean(area_purity - hard_purity)) >= 0.15,
        "mean_area_minus_row_ge_0_05": float(np.mean(area_purity - row_purity)) >= 0.05,
        "area_beats_hard_all_roots": bool(np.all(area_purity > hard_purity)),
        "area_beats_row_all_roots": bool(np.all(area_purity > row_purity)),
    }
    soft = {"checks": soft_checks, "pass": all(soft_checks.values())}
    uniform_purity = np.array([item["parent_purity"] for item in uniform])
    area_cv = np.array([item["parent_mass_coefficient_of_variation"] for item in area])
    uniform_cv = np.array([item["parent_mass_coefficient_of_variation"] for item in uniform])
    purity_attribution = bool(
        np.mean(area_purity - uniform_purity) >= 0.03 and np.sum(area_purity > uniform_purity) >= 2
    )
    cv_attribution = bool(
        np.mean(area_cv) <= 0.8 * np.mean(uniform_cv) and np.sum(area_cv < uniform_cv) >= 2
    )
    capacity = {
        "checks": {
            "purity_route": purity_attribution,
            "parent_mass_cv_route": cv_attribution,
        },
        "pass": purity_attribution or cv_attribution,
    }
    shuffled_purity = np.array([item["parent_purity"] for item in shuffled])
    negative_checks = {
        "mean_area_minus_shuffled_ge_0_15": float(np.mean(area_purity - shuffled_purity)) >= 0.15,
        "shuffled_fails_floor_every_root": all(
            item["parent_purity"] < 0.90 or item["parent_completeness"] < 0.90 for item in shuffled
        ),
        "shuffled_mass_diagnostics_all_roots": all(
            root["arms"]["shuffled_view"]["transport_mass_diagnostics"]["pass"] for root in results
        ),
    }
    negative = {"checks": negative_checks, "pass": all(negative_checks.values())}
    overall_pass = all(
        gate["pass"] for gate in (validity, transport_mass, absolute, soft, capacity, negative)
    )
    return {
        "eligible": True,
        "validity": validity,
        "transport_mass_validity": transport_mass,
        "absolute_mechanism": absolute,
        "soft_assignment_gain": soft,
        "capacity_attribution": capacity,
        "negative_control": negative,
        "overall": {"pass": overall_pass},
    }


def _real_release(gates: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    validity = bool(gates.get("eligible") and gates.get("validity", {}).get("pass"))
    primary: str | None = None
    if validity:
        area_pass = _transport_arm_accepted(results, "uot_area")
        uniform_pass = _transport_arm_accepted(results, "uot_uniform")
        primary = "uot_area" if area_pass else "uot_uniform" if uniform_pass else None
    permitted = validity and primary is not None
    return {
        "permitted": permitted,
        "primary_transport_arm": primary,
        "paired_diagnostic_arms": ["uot_uniform", "uot_area"],
        "reason": (
            f"synthetic validity passed and {primary} was scientifically accepted"
            if permitted
            else "no scientifically accepted transport arm"
            if validity
            else "synthetic validity not established"
        ),
    }


def _resolve_roots(mode: str) -> tuple[tuple[int, int, int], ...]:
    if mode == "official":
        return OFFICIAL_ROOT_TUPLES
    if mode != "development":
        raise ValueError("mode must be 'development' or 'official'")
    official_values = set(OFFICIAL_SCENE_ROOTS + OFFICIAL_DEPTH_ROOTS + OFFICIAL_ORDER_ROOTS)
    if any(value in official_values for roots in DEVELOPMENT_ROOT_TUPLES for value in roots):
        raise RuntimeError("development roots overlap the frozen official roots")
    return DEVELOPMENT_ROOT_TUPLES


def run_experiment(
    *,
    mode: Literal["development", "official"],
    artifacts_dir: Path,
    result_path: Path,
    config: ScientificConfig = FROZEN_CONFIG,
    roots: tuple[tuple[int, int, int], ...] | None = None,
    arm_names: tuple[str, ...] = ARM_NAMES,
    confirm_official_roots: bool = False,
) -> dict[str, Any]:
    protocol = _validated_protocol_receipt()
    source_hashes = _source_hashes()
    selected_roots = _resolve_roots(mode) if roots is None else roots
    official_attempt: dict[str, Any] | None = None
    if mode == "official":
        if not confirm_official_roots:
            raise RuntimeError("official mode requires --confirm-official-roots")
        if selected_roots != OFFICIAL_ROOT_TUPLES:
            raise RuntimeError("official mode requires the exact preregistered root tuples")
        if config != FROZEN_CONFIG or arm_names != ARM_NAMES:
            raise RuntimeError("official mode requires the frozen schedule and complete arm suite")
        if artifacts_dir.resolve() != OFFICIAL_ARTIFACTS.resolve():
            raise RuntimeError("official mode requires the frozen artifact directory")
        if result_path.resolve() != OFFICIAL_RESULT.resolve():
            raise RuntimeError("official mode requires the frozen result path")
        # The exclusive, fsync'd attempt receipt is the first official write. Root construction
        # starts only after this durable non-reuse boundary has been established.
        official_attempt = _reserve_official_attempt(
            artifacts_dir=artifacts_dir,
            result_path=result_path,
            roots=selected_roots,
            config=config,
            arm_names=arm_names,
            protocol=protocol,
            source_hashes=source_hashes,
        )
    else:
        official_values = set(OFFICIAL_SCENE_ROOTS + OFFICIAL_DEPTH_ROOTS + OFFICIAL_ORDER_ROOTS)
        if any(value in official_values for root in selected_roots for value in root):
            raise RuntimeError("development mode refuses official roots")
    artifacts_dir.mkdir(parents=True, exist_ok=False)
    _fsync_directory(artifacts_dir.parent)
    started = time.perf_counter()
    root_results: list[dict[str, Any]] = []
    for index, root_values in enumerate(selected_roots):
        inputs = _build_root_inputs(*root_values)
        root_results.append(_run_root(inputs, artifacts_dir / f"root_{index}", config, arm_names))
    gates = _synthetic_gates(root_results)
    release = _real_release(gates, root_results)
    if _source_hashes() != source_hashes:
        raise RuntimeError("executed synthetic-runner sources changed during the experiment")
    payload = {
        "namespace": NAMESPACE,
        "status": "PASS" if gates.get("overall", {}).get("pass") else "FAIL",
        "mode": mode,
        "roots": [list(item) for item in selected_roots],
        "config": dataclasses.asdict(config),
        "arms": list(arm_names),
        "root_results": root_results,
        "synthetic_gates": gates,
        "real_release": release,
        "environment": _environment_receipt(),
        "wall_time_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "protocol": protocol,
        "source_hashes": source_hashes,
        "official_attempt": official_attempt,
    }
    descriptor = _write_json_exclusive(result_path, payload)
    payload["result_descriptor"] = descriptor
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("development", "official"), default="development")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--confirm-official-roots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.mode == "official":
        artifacts = OFFICIAL_ARTIFACTS if args.out is None else args.out
        result = OFFICIAL_RESULT if args.result is None else args.result
    else:
        if args.out is None or args.result is None:
            raise SystemExit("development mode requires --out and --result")
        artifacts = args.out
        result = args.result
    payload = run_experiment(
        mode=args.mode,
        artifacts_dir=artifacts,
        result_path=result,
        confirm_official_roots=args.confirm_official_roots,
    )
    print(json.dumps({"status": payload["status"], "result": str(result)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
