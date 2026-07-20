#!/usr/bin/env python3
"""Preregistered Carve moment-merge audit and gated exact-count ablation."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.metrics import masked_crop, masked_psnr, psnr, ssim
from rtgs.core.sh import sh_to_rgb
from rtgs.data.scene import SceneData
from rtgs.data.synthetic import make_synthetic_scene
from rtgs.image2gs.fit import FitConfig, fit_views
from rtgs.lift.carve import CarveLifter
from rtgs.lift.merge import merge_by_voxel
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.torch_ref import TorchRasterizer

ROOT = Path(__file__).resolve().parent.parent
BASE_PREREGISTRATION = Path("benchmarks/results/20260715_carve_merge_controls_PREREG.md")
BASE_PREREGISTRATION_SHA256 = "4eda7a69442bddc25cd5edce85125942f91adc52f3d62806f050a64b854b3efe"
PREREGISTRATION = Path("benchmarks/results/20260716_carve_merge_controls_iter2_PREREG.md")
PREREGISTRATION_SHA256 = "fd4361ab1a53a22760db72e99614abb04206c1b639602e0015d8debde91c1203"
FAILED_SEAL = Path("benchmarks/results/20260715_carve_merge_controls_SEAL.json")
FAILED_SEAL_SHA256 = "a802d14170944944e2cee0b766a44f635aad59bdf2412bd771629b06e4d0d923"
FAILED_SOURCE_AGGREGATE = "21ca5d47a4cad54c8cdf446339f174febc48018f6cee45b193569aebd40694cf"
FAILED_PHASE_A_ATTEMPT = Path(
    "benchmarks/results/20260715_carve_merge_controls_PHASE_A_ATTEMPT.json"
)
FAILED_PHASE_A_ATTEMPT_SHA256 = "4e784e9626bf9d3025be1e8ed2c362ba75471538a02b2babae93257af3cf7b5c"
FAILURE_AUDIT = Path(
    "benchmarks/results/20260715T225457Z_cpu_carve_merge_controls_FAILURE_AUDIT.md"
)
FAILURE_AUDIT_SHA256 = "861535cd6a99bca7ce4f49ddd66aefe3dd4965bcb40de3ed93d309363c5b7c5c"
FAILED_PHASE_A_OUTPUT = Path(
    "benchmarks/results/20260715T225457Z_cpu_carve_merge_controls_audit.json"
)
FAILED_PHASE_A_NOTE = Path(
    "benchmarks/results/20260715T225457Z_cpu_carve_merge_controls_audit_RESULT.md"
)
REPAIR_PATCH = Path("benchmarks/results/20260716_carve_merge_controls_iter2_repair.patch")
DEFAULT_SEAL = Path("benchmarks/results/20260716_carve_merge_controls_iter2_SEAL.json")
PHASE_A_ATTEMPT = (
    ROOT / "benchmarks/results/20260716_carve_merge_controls_iter2_PHASE_A_ATTEMPT.json"
)
PHASE_B_ATTEMPT = (
    ROOT / "benchmarks/results/20260716_carve_merge_controls_iter2_PHASE_B_ATTEMPT.json"
)
SEEDS = [0, 1, 2]
TRAIN_INDICES = [0, 1, 2, 4, 5, 6, 8, 9, 10]
TEST_INDICES = [3, 7, 11]
ARMS = ("moment", "voxel_representative", "global_budget_prune")
REPORTING_ARMS = (*ARMS, "raw_keep_all")
CHECKPOINT_STEPS = [0, 30, 60, 90, 120]
TRAINING_ORDER = {
    0: ["moment", "voxel_representative", "global_budget_prune"],
    1: ["voxel_representative", "global_budget_prune", "moment"],
    2: ["global_budget_prune", "moment", "voxel_representative"],
}
METRICS = (
    "psnr_fg",
    "psnr_full",
    "psnr_crop",
    "ssim_crop",
    "depth_rmse_over_extent",
    "alpha_iou",
    "foreground_coverage",
)
SEALED_PATHS = tuple(
    sorted(
        {
            BASE_PREREGISTRATION,
            PREREGISTRATION,
            FAILED_SEAL,
            FAILED_PHASE_A_ATTEMPT,
            FAILURE_AUDIT,
            REPAIR_PATCH,
            Path("benchmarks/carve_merge_controls_ablation.py"),
            Path("pyproject.toml"),
            *(path.relative_to(ROOT) for path in (ROOT / "src" / "rtgs").rglob("*.py")),
            *(path.relative_to(ROOT) for path in (ROOT / "tests").rglob("*.py")),
        },
        key=str,
    )
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def left_fold_float64(values: Iterable[float]) -> float:
    """Reproduce the frozen ordered Python-float materiality reduction exactly."""
    total = 0.0
    for value in values:
        total = total + float(value)
    return total


def canonical_json_hash(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def preregistration_bindings() -> list[dict[str, str]]:
    return [
        {"path": str(BASE_PREREGISTRATION), "sha256": BASE_PREREGISTRATION_SHA256},
        {"path": str(PREREGISTRATION), "sha256": PREREGISTRATION_SHA256},
    ]


def verify_retry_provenance() -> dict[str, Any]:
    fixed_hashes = {
        BASE_PREREGISTRATION: BASE_PREREGISTRATION_SHA256,
        PREREGISTRATION: PREREGISTRATION_SHA256,
        FAILED_SEAL: FAILED_SEAL_SHA256,
        FAILED_PHASE_A_ATTEMPT: FAILED_PHASE_A_ATTEMPT_SHA256,
        FAILURE_AUDIT: FAILURE_AUDIT_SHA256,
    }
    missing = [str(path) for path in fixed_hashes if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"Retry-2 provenance files are missing: {missing}")
    actual_hashes = {path: sha256_file(ROOT / path) for path in fixed_hashes}
    mismatches = {
        str(path): {"expected": fixed_hashes[path], "actual": actual_hashes[path]}
        for path in fixed_hashes
        if actual_hashes[path] != fixed_hashes[path]
    }
    if mismatches:
        raise RuntimeError(f"Retry-2 fixed provenance hashes differ: {mismatches}")
    failed_seal = json.loads((ROOT / FAILED_SEAL).read_text(encoding="utf-8"))
    if failed_seal.get("source_aggregate") != FAILED_SOURCE_AGGREGATE:
        raise RuntimeError("failed implementation-seal source aggregate differs")
    absent_targets = (FAILED_PHASE_A_OUTPUT, FAILED_PHASE_A_NOTE)
    existing = [str(path) for path in absent_targets if (ROOT / path).exists()]
    if existing:
        raise RuntimeError(f"failed-attempt targets must remain absent: {existing}")
    return {
        "fixed_hashes": {str(path): fixed_hashes[path] for path in fixed_hashes},
        "failed_source_aggregate": FAILED_SOURCE_AGGREGATE,
        "failed_targets_absent": [str(path) for path in absent_targets],
    }


def retry_attempt_bindings() -> dict[str, str]:
    verify_retry_provenance()
    return {
        "base_preregistration_sha256": BASE_PREREGISTRATION_SHA256,
        "retry_preregistration_sha256": PREREGISTRATION_SHA256,
        "failed_seal_sha256": FAILED_SEAL_SHA256,
        "failed_source_aggregate": FAILED_SOURCE_AGGREGATE,
        "failed_phase_a_attempt_sha256": FAILED_PHASE_A_ATTEMPT_SHA256,
        "failure_audit_sha256": FAILURE_AUDIT_SHA256,
    }


def tensor_collection_hash(items: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def gaussians_hash(gaussians: Gaussians3D) -> str:
    return tensor_collection_hash(
        [
            ("means", gaussians.means),
            ("quats", gaussians.quats),
            ("log_scales", gaussians.log_scales),
            ("opacity", gaussians.opacity),
            ("sh", gaussians.sh),
        ]
    )


def fitted_hash(gaussians2d) -> str:
    items = []
    for view, gaussians in enumerate(gaussians2d):
        for field in ("xy", "chol", "color", "weight"):
            items.append((f"view_{view}/{field}", getattr(gaussians, field)))
    return tensor_collection_hash(items)


def scene_hashes(scene: SceneData) -> dict[str, str]:
    camera_payload = []
    camera_tensors = []
    for index, camera in enumerate(scene.cameras):
        camera_payload.append(
            {
                "fx": camera.fx,
                "fy": camera.fy,
                "cx": camera.cx,
                "cy": camera.cy,
                "width": camera.width,
                "height": camera.height,
            }
        )
        camera_tensors.extend([(f"R_{index}", camera.R), (f"t_{index}", camera.t)])
    result = {
        "images": tensor_collection_hash(
            [(f"image_{index}", image) for index, image in enumerate(scene.images)]
        ),
        "camera_scalars": canonical_json_hash(camera_payload),
        "camera_tensors": tensor_collection_hash(camera_tensors),
    }
    if scene.points is not None:
        result["points"] = tensor_collection_hash([("points", scene.points)])
    if scene.gt_depths is not None:
        result["gt_depths"] = tensor_collection_hash(
            [(f"depth_{index}", depth) for index, depth in enumerate(scene.gt_depths)]
        )
    if scene.gt_gaussians is not None:
        result["gt_gaussians"] = gaussians_hash(scene.gt_gaussians)
    return result


def assert_finite_tree(value: Any, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"{context} contains non-finite float")
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_tree(item, f"{context}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_tree(item, f"{context}/{index}")


def assert_finite_gaussians(gaussians: Gaussians3D, context: str) -> None:
    for field in ("means", "quats", "log_scales", "opacity", "sh"):
        value = getattr(gaussians, field)
        if not bool(torch.isfinite(value).all()):
            raise AssertionError(f"{context}/{field} is non-finite")


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
        lr=1e-2,
        grad_init_mix=0.7,
        row_chunk=64,
        log_every=50,
        convergence_patience=0,
        convergence_tol=0.05,
        convergence_check_every=25,
    )


def carve_kwargs(*, merge: bool) -> dict[str, Any]:
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
        "merge": merge,
        "merge_voxel_scale": 1.0,
        "init_opacity": 0.1,
        "sh_degree": 0,
    }


def train_config(seed: int) -> TrainConfig:
    return TrainConfig(
        iterations=120,
        lr_means=1.6e-4,
        lr_quats=1e-3,
        lr_scales=5e-3,
        lr_opacity=5e-2,
        lr_sh=2.5e-3,
        lr_sh_rest=1.25e-4,
        ssim_lambda=0.2,
        rasterizer="torch",
        device="cpu",
        densify=False,
        eval_every=30,
        target_sh_degree=3,
        sh_degree_interval=30,
        use_masks=False,
        outside_alpha_lambda=0.01,
        mask_alpha_lambda=0.05,
        random_background=False,
        opacity_reg=None,
        scale_reg=None,
        packed=False,
        antialiased=False,
        sh_color_activation="hard",
        kernel_support_mode="hard",
        visibility_margin_sigma=3.0,
        validate_render_finite=True,
        seed=seed,
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
    keys = (
        "python",
        "torch",
        "platform",
        "processor",
        "cpu_count",
        "torch_num_threads",
        "torch_num_interop_threads",
        "deterministic_algorithms",
        "cuda_visible_devices",
        "omp_num_threads",
        "mkl_num_threads",
        "device",
    )
    return {key: metadata[key] for key in keys}


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
        key: {"expected": expected_value, "actual": metadata.get(key)}
        for key, expected_value in expected.items()
        if metadata.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"official CPU environment differs from preregistration: {mismatches}")


def git_metadata() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True
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
    paths.add((ROOT / BASE_PREREGISTRATION).resolve())
    paths.add((ROOT / PREREGISTRATION).resolve())
    paths.add((ROOT / "pyproject.toml").resolve())
    hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(paths)}
    return hashes, canonical_json_hash(hashes)


def _native_merge_weights(raw: Gaussians3D) -> torch.Tensor:
    return (raw.opacity * raw.scales.prod(dim=-1)).clamp_min(1e-12)


def _world_origin_groups(raw: Gaussians3D, voxel_size: float) -> tuple[torch.Tensor, torch.Tensor]:
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel_size must be finite and positive")
    keys = torch.floor(raw.means / voxel_size).long()
    unique_keys, groups = torch.unique(keys, dim=0, return_inverse=True)
    return unique_keys, groups


def construct_arms(
    raw: Gaussians3D, voxel_size: float
) -> tuple[dict[str, Gaussians3D], dict[str, Any]]:
    """Construct every frozen arm from one raw, ordered Carve tensor."""
    assert_finite_gaussians(raw, "raw Carve tensor")
    unique_keys, groups = _world_origin_groups(raw, voxel_size)
    count = int(unique_keys.shape[0])
    weights = _native_merge_weights(raw)
    audit_weights = (raw.opacity.double() * raw.log_scales.double().exp().prod(dim=-1)).clamp_min(
        1e-12
    )
    representative_indices = []
    for group_id in range(count):
        indices = torch.where(groups == group_id)[0]
        maximum = weights[indices].max()
        tied = indices[weights[indices] == maximum]
        representative_indices.append(int(tied.min()))
    representatives = torch.tensor(
        representative_indices, dtype=torch.long, device=raw.means.device
    )
    ranked = sorted(range(raw.n), key=lambda index: (-float(weights[index]), index))
    global_indices = torch.tensor(sorted(ranked[:count]), dtype=torch.long, device=raw.means.device)
    arms = {
        "moment": merge_by_voxel(raw, voxel_size, opacity_mode="mean"),
        "voxel_representative": raw.subset(representatives),
        "global_budget_prune": raw.subset(global_indices),
        "raw_keep_all": raw.detach(),
    }
    group_counts = torch.bincount(groups, minlength=count)
    metadata = {
        "voxel_size": float(voxel_size),
        "count": count,
        "unique_keys": unique_keys.cpu().tolist(),
        "group_ids": groups.cpu().tolist(),
        "group_counts": group_counts.cpu().tolist(),
        "weights": [float(value) for value in weights.double().cpu()],
        "audit_weights_float64": [float(value) for value in audit_weights.cpu()],
        "representative_indices": representatives.cpu().tolist(),
        "global_budget_indices": global_indices.cpu().tolist(),
        "hashes": {
            "unique_keys": tensor_collection_hash([("unique_keys", unique_keys)]),
            "group_ids": tensor_collection_hash([("group_ids", groups)]),
            "group_counts": tensor_collection_hash([("group_counts", group_counts)]),
            "weights": tensor_collection_hash([("weights", weights)]),
            "audit_weights_float64": tensor_collection_hash(
                [("audit_weights_float64", audit_weights)]
            ),
            "representative_indices": tensor_collection_hash(
                [("representative_indices", representatives)]
            ),
            "global_budget_indices": tensor_collection_hash(
                [("global_budget_indices", global_indices)]
            ),
        },
        "arm_hashes": {name: gaussians_hash(arm) for name, arm in arms.items()},
    }
    return arms, metadata


def _max_errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    absolute = (actual - expected).abs()
    relative = absolute / expected.abs().clamp_min(1e-12)
    return {
        "maximum_absolute_error": float(absolute.max()) if absolute.numel() else 0.0,
        "maximum_relative_error": float(relative.max()) if relative.numel() else 0.0,
    }


def covariance_float64_from_raw(raw: Gaussians3D) -> torch.Tensor:
    """Reconstruct raw covariance after promoting primitive parameters to float64."""
    quaternions = torch.nn.functional.normalize(raw.quats.double(), dim=-1)
    w, x, y, z = (
        quaternions[:, 0],
        quaternions[:, 1],
        quaternions[:, 2],
        quaternions[:, 3],
    )
    rotations = torch.stack(
        [
            torch.stack(
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - w * z),
                    2 * (x * z + w * y),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (x * y + w * z),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - w * x),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (x * z - w * y),
                    2 * (y * z + w * x),
                    1 - 2 * (x * x + y * y),
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )
    scales = raw.log_scales.double().exp()
    rotation_scales = rotations * scales[:, None, :]
    return rotation_scales @ rotation_scales.transpose(-1, -2)


def audit_arm_construction(
    raw: Gaussians3D,
    arms: dict[str, Gaussians3D],
    construction: dict[str, Any],
    *,
    parity: Gaussians3D,
) -> dict[str, Any]:
    """Recompute all frozen structural and normalized-moment identities."""
    count = int(construction["count"])
    groups = torch.tensor(construction["group_ids"], dtype=torch.long, device=raw.means.device)
    unique_keys = torch.tensor(
        construction["unique_keys"], dtype=torch.long, device=raw.means.device
    )
    representative_indices = torch.tensor(
        construction["representative_indices"], dtype=torch.long, device=raw.means.device
    )
    global_indices = torch.tensor(
        construction["global_budget_indices"], dtype=torch.long, device=raw.means.device
    )
    weights_native = _native_merge_weights(raw)
    moment = arms["moment"]
    representative = arms["voxel_representative"]
    global_prune = arms["global_budget_prune"]
    for name, arm in arms.items():
        assert_finite_gaussians(arm, name)
    if raw.n <= 0 or not (1 <= count <= raw.n):
        raise AssertionError("raw/group construction is internally impossible")
    if any(arms[name].n != count for name in ARMS):
        raise AssertionError("trained arms do not have the exact common count")
    if not torch.equal(torch.unique(groups), torch.arange(count, device=groups.device)):
        raise AssertionError("group IDs do not cover the canonical range")
    expected_keys, expected_groups = _world_origin_groups(raw, float(construction["voxel_size"]))
    if not torch.equal(unique_keys, expected_keys) or not torch.equal(groups, expected_groups):
        raise AssertionError("serialized world-origin groups differ on recomputation")
    if gaussians_hash(parity) != gaussians_hash(moment):
        raise AssertionError("Carve merge=True parity lift differs from the moment arm")

    expected_representatives = []
    for group_id in range(count):
        indices = torch.where(groups == group_id)[0]
        maximum = weights_native[indices].max()
        tied = indices[weights_native[indices] == maximum]
        expected_representatives.append(int(tied.min()))
    if representative_indices.tolist() != expected_representatives:
        raise AssertionError("representative tie rule differs from preregistration")
    ranked = sorted(range(raw.n), key=lambda index: (-float(weights_native[index]), index))
    expected_global = sorted(ranked[:count])
    if global_indices.tolist() != expected_global:
        raise AssertionError("global top-K tie rule differs from preregistration")
    if representative_indices.unique().numel() != count or global_indices.unique().numel() != count:
        raise AssertionError("control indices are not unique")
    if not torch.equal(groups[representative_indices], torch.arange(count, device=groups.device)):
        raise AssertionError("representatives do not belong to their emitted groups")
    for output, indices, name in (
        (representative, representative_indices, "voxel_representative"),
        (global_prune, global_indices, "global_budget_prune"),
    ):
        for field in ("means", "quats", "log_scales", "opacity", "sh"):
            if not torch.equal(getattr(output, field), getattr(raw, field)[indices]):
                raise AssertionError(f"{name}/{field} is not a bitwise raw subset")

    weights = (raw.opacity.double() * raw.log_scales.double().exp().prod(dim=-1)).clamp_min(1e-12)
    covariances = covariance_float64_from_raw(raw)
    expected_means = torch.empty(count, 3, dtype=torch.float64, device=raw.means.device)
    expected_covariances = torch.empty(count, 3, 3, dtype=torch.float64, device=raw.means.device)
    expected_sh = torch.empty(
        count, raw.sh.shape[1], 3, dtype=torch.float64, device=raw.means.device
    )
    expected_opacity = torch.empty(count, dtype=torch.float64, device=raw.means.device)
    center_dispersion = []
    color_dispersion = []
    covariance_dispersion = []
    colors = sh_to_rgb(raw.sh[:, 0]).double()
    for group_id in range(count):
        indices = torch.where(groups == group_id)[0]
        group_weights = weights[indices]
        denominator = group_weights.sum()
        mean = (raw.means[indices].double() * group_weights[:, None]).sum(dim=0) / denominator
        difference = raw.means[indices].double() - mean
        second = covariances[indices] + difference[:, :, None] * difference[:, None, :]
        covariance = (second * group_weights[:, None, None]).sum(dim=0) / denominator
        expected_means[group_id] = mean
        expected_covariances[group_id] = covariance
        expected_sh[group_id] = (raw.sh[indices].double() * group_weights[:, None, None]).sum(
            dim=0
        ) / denominator
        expected_opacity[group_id] = (
            raw.opacity[indices].double() * group_weights
        ).sum() / denominator
        center_dispersion.append(float(difference.norm(dim=-1).max()))
        color_dispersion.append(float(colors[indices].std(dim=0, unbiased=False).norm()))
        covariance_dispersion.append(
            float(
                (covariances[indices] - covariances[indices].mean(dim=0, keepdim=True))
                .flatten(1)
                .norm(dim=-1)
                .max()
            )
        )
    expected_opacity = expected_opacity.clamp(0.01, 0.995)
    actual_covariances = moment.covariance().double()
    comparisons = {
        "mean": _max_errors(moment.means.double(), expected_means),
        "covariance": _max_errors(actual_covariances, expected_covariances),
        "sh": _max_errors(moment.sh.double(), expected_sh),
        "opacity": _max_errors(moment.opacity.double(), expected_opacity),
    }
    for name, (actual, expected) in {
        "mean": (moment.means.double(), expected_means),
        "covariance": (actual_covariances, expected_covariances),
        "sh": (moment.sh.double(), expected_sh),
        "opacity": (moment.opacity.double(), expected_opacity),
    }.items():
        if not torch.allclose(actual, expected, atol=2e-6, rtol=2e-5):
            raise AssertionError(f"normalized moment identity failed for {name}")
    symmetry_error = float((actual_covariances - actual_covariances.transpose(-1, -2)).abs().max())
    minimum_eigenvalue = float(torch.linalg.eigvalsh(actual_covariances).min())
    lower = torch.empty_like(expected_means)
    upper = torch.empty_like(expected_means)
    for group_id in range(count):
        indices = torch.where(groups == group_id)[0]
        lower[group_id] = raw.means[indices].double().amin(dim=0)
        upper[group_id] = raw.means[indices].double().amax(dim=0)
    bounds_violation = float(
        torch.maximum(lower - moment.means.double(), moment.means.double() - upper)
        .clamp_min(0)
        .max()
    )
    if symmetry_error > 2e-6:
        raise AssertionError("moment covariance symmetry tolerance failed")
    if minimum_eigenvalue <= 0.0:
        raise AssertionError("moment covariance is not positive definite")
    if bounds_violation > 2e-6:
        raise AssertionError("moment mean lies outside a group raw-mean bound")

    group_counts = torch.bincount(groups, minlength=count)
    multi = group_counts > 1
    multi_group_count = int(multi.sum())
    raw_in_multi = int(group_counts[multi].sum())
    representative_set = set(representative_indices.tolist())
    global_set = set(global_indices.tolist())
    union = representative_set | global_set
    jaccard = len(representative_set & global_set) / len(union) if union else 1.0
    return {
        "structural_valid": True,
        "parity_hash_equal": True,
        "moment_identity_pass": True,
        "moment_errors": comparisons,
        "covariance_symmetry_max_abs": symmetry_error,
        "covariance_minimum_eigenvalue": minimum_eigenvalue,
        "mean_bounds_max_violation": bounds_violation,
        "raw_count": raw.n,
        "group_count": count,
        "compression_fraction": 1.0 - count / raw.n,
        "group_count_distribution": {
            "minimum": int(group_counts.min()),
            "maximum": int(group_counts.max()),
            "mean": float(group_counts.double().mean()),
            "singleton_groups": int((group_counts == 1).sum()),
            "multi_member_groups": multi_group_count,
            "raw_primitives_in_multi_member_groups": raw_in_multi,
            "raw_multi_member_fraction": raw_in_multi / raw.n,
        },
        "within_group_dispersion": {
            "center_max_per_group_mean": statistics.fmean(center_dispersion),
            "center_max": max(center_dispersion),
            "rgb_std_norm_per_group_mean": statistics.fmean(color_dispersion),
            "rgb_std_norm_max": max(color_dispersion),
            "covariance_frobenius_max_per_group_mean": statistics.fmean(covariance_dispersion),
            "covariance_frobenius_max": max(covariance_dispersion),
        },
        "selected_index_overlap": {
            "intersection": len(representative_set & global_set),
            "union": len(union),
            "jaccard": jaccard,
        },
    }


def truth_cache(scene: SceneData, indices: list[int]) -> dict[int, dict[str, Any]]:
    if scene.gt_gaussians is None:
        raise ValueError("synthetic GT Gaussians are required")
    renderer = TorchRasterizer()
    result = {}
    with torch.no_grad():
        for index in indices:
            output = renderer.render(scene.gt_gaussians, scene.cameras[index])
            for field in ("color", "alpha", "depth"):
                if not bool(torch.isfinite(getattr(output, field)).all()):
                    raise AssertionError(f"truth {field} is non-finite in view {index}")
            support = output.alpha > 0.05
            if not bool(support.any()):
                raise AssertionError(f"truth support is empty in view {index}")
            result[index] = {
                "color": output.color.detach(),
                "alpha": output.alpha.detach(),
                "depth": output.depth.detach(),
                "support": support.detach(),
                "hash": tensor_collection_hash(
                    [
                        ("color", output.color),
                        ("alpha", output.alpha),
                        ("depth", output.depth),
                        ("support", support),
                    ]
                ),
            }
    return result


def evaluate_views(
    scene: SceneData,
    gaussians: Gaussians3D,
    indices: list[int],
    truths: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    renderer = TorchRasterizer()
    _, extent = scene.center_and_extent()
    per_view = []
    with torch.no_grad():
        for index in indices:
            output = renderer.render(gaussians, scene.cameras[index])
            for field in ("color", "alpha", "depth"):
                if not bool(torch.isfinite(getattr(output, field)).all()):
                    raise AssertionError(f"prediction {field} is non-finite in view {index}")
            target = scene.images[index].clamp(0.0, 1.0)
            predicted = output.color.clamp(0.0, 1.0)
            support = truths[index]["support"]
            predicted_support = output.alpha > 0.05
            intersection = support & predicted_support
            union = support | predicted_support
            if not bool(intersection.any()):
                raise AssertionError(f"prediction has empty depth intersection in view {index}")
            predicted_depth = output.depth / output.alpha.clamp_min(1e-6)
            truth_depth = truths[index]["depth"] / truths[index]["alpha"].clamp_min(1e-6)
            predicted_crop = masked_crop(predicted, support.float())
            target_crop = masked_crop(target, support.float())
            support_values = int(support.sum()) * predicted.shape[-1]
            foreground_sse = float(
                (((predicted - target).double().square()) * support[..., None]).sum()
            )
            full_sse = float((predicted.double() - target.double()).square().sum())
            crop_sse = float((predicted_crop.double() - target_crop.double()).square().sum())
            depth_sse = float(
                (predicted_depth[intersection].double() - truth_depth[intersection].double())
                .square()
                .sum()
            )
            intersection_count = int(intersection.sum())
            union_count = int(union.sum())
            truth_support_count = int(support.sum())
            values = {
                "view": index,
                "psnr_fg": masked_psnr(predicted, target, support.float()),
                "psnr_full": psnr(predicted, target),
                "psnr_crop": psnr(predicted_crop, target_crop),
                "ssim_crop": float(ssim(predicted_crop, target_crop)),
                "depth_rmse_over_extent": float(
                    (predicted_depth[intersection] - truth_depth[intersection])
                    .square()
                    .mean()
                    .sqrt()
                    / extent
                ),
                "alpha_iou": float(intersection.sum() / union.sum().clamp_min(1)),
                "foreground_coverage": float(intersection.sum() / support.sum().clamp_min(1)),
                "truth_hash": truths[index]["hash"],
                "raw_numerators": {
                    "foreground_rgb_squared_error_sum": foreground_sse,
                    "foreground_rgb_value_count": support_values,
                    "full_rgb_squared_error_sum": full_sse,
                    "full_rgb_value_count": predicted.numel(),
                    "crop_rgb_squared_error_sum": crop_sse,
                    "crop_rgb_value_count": predicted_crop.numel(),
                    "depth_squared_error_sum": depth_sse,
                    "depth_intersection_pixel_count": intersection_count,
                    "alpha_intersection_pixel_count": intersection_count,
                    "alpha_union_pixel_count": union_count,
                    "truth_support_pixel_count": truth_support_count,
                    "scene_extent": float(extent),
                },
            }
            assert_finite_tree(values, f"view metrics/{index}")
            per_view.append(values)
    mean = {
        metric: statistics.fmean(float(view[metric]) for view in per_view) for metric in METRICS
    }
    return {"indices": list(indices), "per_view": per_view, "mean": mean}


def heldout_checkpoint_inputs(
    scene: SceneData, truths: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build the held-out-only capability captured by the checkpoint observer."""
    return [
        {
            "view": index,
            "camera": scene.cameras[index],
            "target": scene.images[index].detach().clone(),
            "support": truths[index]["support"].detach().clone(),
            "truth_hash": truths[index]["hash"],
        }
        for index in TEST_INDICES
    ]


def evaluate_checkpoint(gaussians: Gaussians3D, inputs: list[dict[str, Any]]) -> dict[str, Any]:
    renderer = TorchRasterizer()
    per_view = []
    with torch.no_grad():
        for item in inputs:
            index = int(item["view"])
            output = renderer.render(gaussians, item["camera"])
            if not bool(torch.isfinite(output.color).all()):
                raise AssertionError(f"checkpoint color is non-finite in view {index}")
            predicted = output.color.clamp(0.0, 1.0)
            target = item["target"].clamp(0.0, 1.0)
            support = item["support"]
            foreground_sse = float(
                (((predicted - target).double().square()) * support[..., None]).sum()
            )
            foreground_values = int(support.sum()) * predicted.shape[-1]
            full_sse = float((predicted.double() - target.double()).square().sum())
            per_view.append(
                {
                    "view": index,
                    "psnr_fg": masked_psnr(predicted, target, support.float()),
                    "psnr_full": psnr(predicted, target),
                    "truth_hash": item["truth_hash"],
                    "raw_numerators": {
                        "foreground_rgb_squared_error_sum": foreground_sse,
                        "foreground_rgb_value_count": foreground_values,
                        "full_rgb_squared_error_sum": full_sse,
                        "full_rgb_value_count": predicted.numel(),
                    },
                }
            )
    result = {
        "per_view": per_view,
        "mean_psnr_fg": statistics.fmean(float(view["psnr_fg"]) for view in per_view),
        "mean_psnr_full": statistics.fmean(float(view["psnr_full"]) for view in per_view),
    }
    assert_finite_tree(result, "checkpoint evaluation")
    return result


def materiality_totals_from_views(
    views: list[dict[str, Any]],
) -> tuple[float, dict[str, float]]:
    denominator = left_fold_float64(view["raw_residual_l1"] for view in views)
    numerators = {
        control: left_fold_float64(view["moment_control_color_l1"][control] for view in views)
        for control in ("voxel_representative", "global_budget_prune")
    }
    return denominator, numerators


def materiality_render_audit(scene: SceneData, arms: dict[str, Gaussians3D]) -> dict[str, Any]:
    renderer = TorchRasterizer()
    views = []
    with torch.no_grad():
        for index in TRAIN_INDICES:
            raw_output = renderer.render(arms["raw_keep_all"], scene.cameras[index])
            moment_output = renderer.render(arms["moment"], scene.cameras[index])
            target = scene.images[index]
            residual = float((raw_output.color - target).abs().double().sum())
            control_values = {}
            for control in ("voxel_representative", "global_budget_prune"):
                control_output = renderer.render(arms[control], scene.cameras[index])
                difference = float(
                    (moment_output.color - control_output.color).abs().double().sum()
                )
                control_values[control] = difference
            views.append(
                {
                    "view": index,
                    "raw_residual_l1": residual,
                    "moment_control_color_l1": control_values,
                }
            )
    denominator, numerators = materiality_totals_from_views(views)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise AssertionError("raw render residual denominator is non-finite or zero")
    ratios = {control: numerator / denominator for control, numerator in numerators.items()}
    result = {
        "views": views,
        "raw_residual_l1": denominator,
        "moment_control_color_l1": numerators,
        "ratios": ratios,
    }
    assert_finite_tree(result, "materiality render audit")
    return result


def make_scene(seed: int) -> SceneData:
    scene = make_synthetic_scene(n_gaussians=40, n_cameras=12, image_size=48, seed=seed)
    scene.train_indices = list(TRAIN_INDICES)
    scene.test_indices = list(TEST_INDICES)
    scene.validate()
    return scene


def prepare_seed(
    seed: int, *, include_parity: bool
) -> tuple[SceneData, Any, Gaussians3D, Gaussians3D | None, float, dict[str, Any]]:
    torch.manual_seed(seed)
    scene = make_scene(seed)
    train_scene = scene.subset(TRAIN_INDICES)
    started = time.perf_counter()
    gaussians2d, fit_history = fit_views(train_scene.images, fit_config(), seed=seed, masks=None)
    fit_seconds = time.perf_counter() - started
    raw = CarveLifter(**carve_kwargs(merge=False)).lift(gaussians2d, train_scene)
    parity = (
        CarveLifter(**carve_kwargs(merge=True)).lift(gaussians2d, train_scene)
        if include_parity
        else None
    )
    _, extent = train_scene.center_and_extent()
    voxel_size = extent / 48.0
    assert_finite_gaussians(raw, f"raw Carve seed {seed}")
    if parity is not None:
        assert_finite_gaussians(parity, f"parity Carve seed {seed}")
    metadata = {
        "seed": seed,
        "scene_hashes": scene_hashes(scene),
        "train_scene_hashes": scene_hashes(train_scene),
        "fitted_hash": fitted_hash(gaussians2d),
        "fit_history_hash": canonical_json_hash(fit_history),
        "fit_history": fit_history,
        "fit_seconds": fit_seconds,
        "fit_config": asdict(fit_config()),
        "carve_config": carve_kwargs(merge=False),
        "raw_hash": gaussians_hash(raw),
        "raw_order_hash": tensor_collection_hash(
            [("raw_index", torch.arange(raw.n, dtype=torch.int64)), ("means", raw.means)]
        ),
        "raw_count": raw.n,
        "parity_hash": None if parity is None else gaussians_hash(parity),
        "voxel_size": voxel_size,
        "split": {"train": TRAIN_INDICES, "held_out": TEST_INDICES},
    }
    assert_finite_tree(metadata, f"preparation/{seed}")
    return scene, gaussians2d, raw, parity, voxel_size, metadata


def phase_a_seed(seed: int) -> dict[str, Any]:
    scene, _, raw, parity, voxel_size, preparation = prepare_seed(seed, include_parity=True)
    if parity is None:  # pragma: no cover
        raise AssertionError("Phase A requires the parity lift")
    arms, construction = construct_arms(raw, voxel_size)
    construction_audit = audit_arm_construction(raw, arms, construction, parity=parity)
    truths = truth_cache(scene, list(range(scene.n_views)))
    initialization_metrics = {
        name: {
            "training": evaluate_views(scene, arm, TRAIN_INDICES, truths),
            "held_out": evaluate_views(scene, arm, TEST_INDICES, truths),
        }
        for name, arm in arms.items()
    }
    render_materiality = materiality_render_audit(scene, arms)
    record = {
        "seed": seed,
        "preparation": preparation,
        "construction": construction,
        "construction_audit": construction_audit,
        "truth_hashes": {str(index): truths[index]["hash"] for index in range(scene.n_views)},
        "initialization_metrics": initialization_metrics,
        "render_materiality": render_materiality,
    }
    record["gate"] = phase_a_seed_gate_from_raw_evidence(record)
    assert_finite_tree(record, f"Phase A seed {seed}")
    return record


def phase_a_seed_gate_from_raw_evidence(record: dict[str, Any]) -> dict[str, Any]:
    """Recompute every material/size gate from serialized arrays and per-view sums."""
    construction = record["construction"]
    _validate_construction_record(construction)
    groups = torch.tensor(construction["group_ids"], dtype=torch.int64)
    count = int(construction["count"])
    raw_count = int(groups.numel())
    if int(record["preparation"]["raw_count"]) != raw_count:
        raise RuntimeError("Phase-A preparation/raw group length differs")
    group_counts = torch.bincount(groups, minlength=count)
    multi = group_counts > 1
    multi_group_count = int(multi.sum())
    raw_in_multi = int(group_counts[multi].sum())
    representatives = set(int(index) for index in construction["representative_indices"])
    global_indices = set(int(index) for index in construction["global_budget_indices"])
    selected_union = representatives | global_indices
    jaccard = len(representatives & global_indices) / len(selected_union) if selected_union else 1.0
    views = record["render_materiality"]["views"]
    if [int(view.get("view", -1)) for view in views] != TRAIN_INDICES:
        raise RuntimeError("Phase-A render materiality does not cover ordered training views")
    residual, numerators = materiality_totals_from_views(views)
    if not math.isfinite(residual) or residual <= 0.0:
        raise RuntimeError("Phase-A raw render residual is non-finite or zero")
    ratios = {control: numerator / residual for control, numerator in numerators.items()}
    reported = record["render_materiality"]
    if float(reported["raw_residual_l1"]) != residual:
        raise RuntimeError("Phase-A reported residual differs from per-view raw sums")
    if reported["moment_control_color_l1"] != numerators:
        raise RuntimeError("Phase-A reported control numerator differs from per-view raw sums")
    if reported["ratios"] != ratios:
        raise RuntimeError("Phase-A reported render ratio differs from raw numerators")

    audit = record["construction_audit"]
    distribution = audit["group_count_distribution"]
    recomputed_summaries = {
        "raw_count": raw_count,
        "group_count": count,
        "compression_fraction": 1.0 - count / raw_count,
        "multi_member_groups": multi_group_count,
        "raw_primitives_in_multi_member_groups": raw_in_multi,
        "raw_multi_member_fraction": raw_in_multi / raw_count,
        "selected_intersection": len(representatives & global_indices),
        "selected_union": len(selected_union),
        "selected_jaccard": jaccard,
    }
    reported_summaries = {
        "raw_count": int(audit["raw_count"]),
        "group_count": int(audit["group_count"]),
        "compression_fraction": float(audit["compression_fraction"]),
        "multi_member_groups": int(distribution["multi_member_groups"]),
        "raw_primitives_in_multi_member_groups": int(
            distribution["raw_primitives_in_multi_member_groups"]
        ),
        "raw_multi_member_fraction": float(distribution["raw_multi_member_fraction"]),
        "selected_intersection": int(audit["selected_index_overlap"]["intersection"]),
        "selected_union": int(audit["selected_index_overlap"]["union"]),
        "selected_jaccard": float(audit["selected_index_overlap"]["jaccard"]),
    }
    if reported_summaries != recomputed_summaries:
        raise RuntimeError("Phase-A construction summaries differ from serialized raw evidence")
    identity_pass = bool(
        audit["structural_valid"]
        and audit["parity_hash_equal"]
        and audit["moment_identity_pass"]
        and float(audit["covariance_symmetry_max_abs"]) <= 2e-6
        and float(audit["covariance_minimum_eigenvalue"]) > 0.0
        and float(audit["mean_bounds_max_violation"]) <= 2e-6
    )
    criteria = {
        "structural_and_moment_identities_pass": identity_pass,
        "raw_count_at_least_500": raw_count >= 500,
        "group_count_at_least_100": count >= 100,
        "group_count_strictly_below_raw_count": count < raw_count,
        "compression_at_least_10_percent": 1.0 - count / raw_count >= 0.10,
        "multi_member_groups_at_least_50": multi_group_count >= 50,
        "raw_multi_member_fraction_at_least_15_percent": raw_in_multi / raw_count >= 0.15,
        "control_selected_index_jaccard_below_0_95": jaccard < 0.95,
        "moment_vs_voxel_representative_render_ratio_at_least_0_005": ratios["voxel_representative"]
        >= 0.005,
        "moment_vs_global_budget_prune_render_ratio_at_least_0_005": ratios["global_budget_prune"]
        >= 0.005,
    }
    return {"criteria": criteria, "pass": all(criteria.values())}


def phase_a_decision(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if [int(run.get("seed", -1)) for run in runs] != SEEDS:
        raise AssertionError("Phase A runs are missing, duplicated, or reordered")
    seed_gates = [phase_a_seed_gate_from_raw_evidence(run) for run in runs]
    return {
        "seed_passes": [gate["pass"] for gate in seed_gates],
        "seed_criteria": [gate["criteria"] for gate in seed_gates],
        "phase_b_authorized": all(gate["pass"] for gate in seed_gates),
    }


def isolated_schedule_probe(seed: int, arm_hashes: dict[str, str]) -> dict[str, Any]:
    if set(arm_hashes) != set(ARMS):
        raise ValueError("schedule probe requires all three trained arm hashes")
    records = {}
    for arm in ARMS:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        local = [
            int(torch.randint(0, len(TRAIN_INDICES), (1,), generator=generator, device="cpu"))
            for _ in range(120)
        ]
        records[arm] = {
            "arm_hash": arm_hashes[arm],
            "local_schedule": local,
            "global_schedule": [TRAIN_INDICES[index] for index in local],
            "local_schedule_hash": canonical_json_hash(local),
            "global_schedule_hash": canonical_json_hash([TRAIN_INDICES[index] for index in local]),
        }
    hashes = {record["local_schedule_hash"] for record in records.values()}
    if len(hashes) != 1:
        raise AssertionError("isolated schedule depends on arm/hash")
    return {
        "seed": seed,
        "arms": records,
        "all_arm_schedules_equal": True,
        "common_local_schedule_hash": next(iter(hashes)),
    }


def checkpoint_auc(checkpoints: list[dict[str, Any]]) -> float:
    steps = [int(checkpoint["step"]) for checkpoint in checkpoints]
    if steps != CHECKPOINT_STEPS:
        raise ValueError(f"checkpoint steps differ: {steps} != {CHECKPOINT_STEPS}")
    values = [float(checkpoint["metrics"]["mean_psnr_fg"]) for checkpoint in checkpoints]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("checkpoint curve contains a non-finite PSNR")
    area = sum(
        0.5 * (left + right) * (steps[index + 1] - steps[index])
        for index, (left, right) in enumerate(zip(values, values[1:]))
    )
    return area / 120.0


def parameter_count(gaussians: Gaussians3D, target_sh_degree: int = 3) -> int:
    expanded = gaussians.with_sh_degree(target_sh_degree)
    return sum(
        int(getattr(expanded, field).numel())
        for field in ("means", "quats", "log_scales", "opacity", "sh")
    )


def train_arm(
    *,
    seed: int,
    arm_name: str,
    initialization: Gaussians3D,
    full_scene: SceneData,
    truths: dict[int, dict[str, Any]],
    schedule_probe: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_inputs = heldout_checkpoint_inputs(full_scene, truths)
    checkpoints = [
        {
            "step": 0,
            "metrics": evaluate_checkpoint(initialization.with_sh_degree(3), checkpoint_inputs),
        }
    ]
    observed_steps = []

    def checkpoint_observer(snapshot: Gaussians3D, step: int) -> None:
        if step not in CHECKPOINT_STEPS[1:] or step in observed_steps:
            raise AssertionError(f"unexpected or duplicated callback step {step}")
        assert_finite_gaussians(snapshot, f"checkpoint snapshot {arm_name}/{seed}/{step}")
        if any(
            getattr(snapshot, field).requires_grad
            for field in ("means", "quats", "log_scales", "opacity", "sh")
        ):
            raise AssertionError("checkpoint callback received an attached tensor")
        observed_steps.append(step)
        checkpoints.append(
            {"step": step, "metrics": evaluate_checkpoint(snapshot, checkpoint_inputs)}
        )

    config = train_config(seed)
    started = time.perf_counter()
    final, history = Trainer(config).train(
        full_scene.subset(TRAIN_INDICES),
        initialization.detach(),
        checkpoint_callback=checkpoint_observer,
    )
    training_seconds = time.perf_counter() - started
    assert_finite_gaussians(final, f"final {arm_name}/{seed}")
    assert_finite_tree(history, f"history {arm_name}/{seed}")
    if observed_steps != CHECKPOINT_STEPS[1:]:
        raise AssertionError(f"callback checkpoint steps differ for {arm_name}/{seed}")
    if final.n != initialization.n:
        raise AssertionError(f"fixed topology changed primitive count for {arm_name}/{seed}")
    native_steps = [int(item[0]) for item in history["psnr"]]
    if native_steps != CHECKPOINT_STEPS[1:]:
        raise AssertionError(f"native checkpoint schedule differs for {arm_name}/{seed}")
    expected_local = schedule_probe["arms"][arm_name]["local_schedule"]
    actual_local = [int(index) for index in history["sampled_train_views"]]
    if actual_local != expected_local:
        raise AssertionError(f"official history differs from schedule probe for {arm_name}/{seed}")
    expected_counts = [[step, initialization.n] for step in CHECKPOINT_STEPS[1:]]
    actual_counts = [[int(step), int(count)] for step, count in history["n_gaussians"]]
    if actual_counts != expected_counts:
        raise AssertionError(f"checkpoint primitive counts differ for {arm_name}/{seed}")
    expected_sh = [[30, 0], [60, 1], [90, 2], [120, 3]]
    actual_sh = [[int(step), int(degree)] for step, degree in history["active_sh_degree"]]
    if actual_sh != expected_sh:
        raise AssertionError(f"active SH schedule differs for {arm_name}/{seed}")
    final_metrics = evaluate_views(full_scene, final, TEST_INDICES, truths)
    history["sampled_train_views_local"] = actual_local
    history["sampled_train_views"] = [TRAIN_INDICES[index] for index in actual_local]
    result = {
        "seed": seed,
        "arm": arm_name,
        "initialization_hash": gaussians_hash(initialization),
        "initial_gaussians": initialization.n,
        "parameter_count": parameter_count(initialization),
        "train_config": asdict(config),
        "training_seconds": training_seconds,
        "checkpoint_curve": checkpoints,
        "checkpoint_auc_psnr_fg_db": checkpoint_auc(checkpoints),
        "final_hash": gaussians_hash(final),
        "final_gaussians": final.n,
        "final_metrics": final_metrics,
        "history": history,
        "local_schedule_hash": canonical_json_hash(actual_local),
        "global_schedule_hash": canonical_json_hash(history["sampled_train_views"]),
    }
    assert_finite_tree(result, f"trained arm {arm_name}/{seed}")
    return result


def summarize_ablation(runs: list[dict[str, Any]]) -> dict[str, Any]:
    expected = [(seed, arm) for seed in SEEDS for arm in TRAINING_ORDER[seed]]
    identities = [(int(run.get("seed", -1)), run.get("arm")) for run in runs]
    if identities != expected:
        raise AssertionError("Phase B runs are missing, duplicated, or reordered")
    summary = {}
    for arm in ARMS:
        arm_runs = [run for run in runs if run["arm"] == arm]
        if [int(run["seed"]) for run in arm_runs] != SEEDS:
            raise AssertionError(f"seed order differs for {arm}")
        auc_samples = [float(run["checkpoint_auc_psnr_fg_db"]) for run in arm_runs]
        metrics = {}
        for metric in METRICS:
            samples = [float(run["final_metrics"]["mean"][metric]) for run in arm_runs]
            metrics[metric] = {
                "samples": samples,
                "mean": statistics.fmean(samples),
                "stdev": statistics.stdev(samples),
            }
        summary[arm] = {
            "checkpoint_auc_psnr_fg_db": {
                "samples": auc_samples,
                "mean": statistics.fmean(auc_samples),
                "stdev": statistics.stdev(auc_samples),
            },
            "final_metrics": metrics,
        }
    return summary


def ablation_decision(summary: dict[str, Any]) -> dict[str, Any]:
    moment_auc = summary["moment"]["checkpoint_auc_psnr_fg_db"]["samples"]
    comparisons = {}
    utility_criteria = {}
    safety_criteria = {}
    next_candidates = {}
    for control in ("voxel_representative", "global_budget_prune"):
        control_auc = summary[control]["checkpoint_auc_psnr_fg_db"]["samples"]
        auc_gains = [moment - baseline for moment, baseline in zip(moment_auc, control_auc)]
        utility = {
            "mean_auc_gain_at_least_0_10_db": statistics.fmean(auc_gains) >= 0.10,
            "auc_wins_at_least_two_seeds": sum(gain > 0.0 for gain in auc_gains) >= 2,
        }
        moment_metrics = summary["moment"]["final_metrics"]
        control_metrics = summary[control]["final_metrics"]
        psnr_deltas = [
            moment - baseline
            for moment, baseline in zip(
                moment_metrics["psnr_fg"]["samples"],
                control_metrics["psnr_fg"]["samples"],
            )
        ]
        ssim_deltas = [
            moment - baseline
            for moment, baseline in zip(
                moment_metrics["ssim_crop"]["samples"],
                control_metrics["ssim_crop"]["samples"],
            )
        ]
        control_depth = float(control_metrics["depth_rmse_over_extent"]["mean"])
        if control_depth <= 0.0:
            raise AssertionError(f"zero depth denominator for {control}")
        depth_regression = (
            float(moment_metrics["depth_rmse_over_extent"]["mean"]) - control_depth
        ) / control_depth
        alpha_delta = float(moment_metrics["alpha_iou"]["mean"]) - float(
            control_metrics["alpha_iou"]["mean"]
        )
        coverage_delta = float(moment_metrics["foreground_coverage"]["mean"]) - float(
            control_metrics["foreground_coverage"]["mean"]
        )
        safety = {
            "mean_final_psnr_regression_within_0_10_db": statistics.fmean(psnr_deltas) >= -0.10,
            "per_seed_final_psnr_regression_within_0_25_db": min(psnr_deltas) >= -0.25,
            "mean_ssim_regression_within_0_002": statistics.fmean(ssim_deltas) >= -0.002,
            "per_seed_ssim_regression_within_0_005": min(ssim_deltas) >= -0.005,
            "mean_depth_rmse_regression_within_2_percent": depth_regression <= 0.02,
            "mean_alpha_iou_regression_within_0_02": alpha_delta >= -0.02,
            "mean_foreground_coverage_regression_within_0_02": coverage_delta >= -0.02,
        }
        inverse_auc = [-gain for gain in auc_gains]
        moment_depth = float(moment_metrics["depth_rmse_over_extent"]["mean"])
        if moment_depth <= 0.0:
            raise AssertionError("zero moment depth denominator")
        inverse_depth_regression = (control_depth - moment_depth) / moment_depth
        inverse_safety = {
            "mean_final_psnr_regression_within_0_10_db": statistics.fmean(
                [-delta for delta in psnr_deltas]
            )
            >= -0.10,
            "per_seed_final_psnr_regression_within_0_25_db": min(-delta for delta in psnr_deltas)
            >= -0.25,
            "mean_ssim_regression_within_0_002": statistics.fmean([-delta for delta in ssim_deltas])
            >= -0.002,
            "per_seed_ssim_regression_within_0_005": min(-delta for delta in ssim_deltas) >= -0.005,
            "mean_depth_rmse_regression_within_2_percent": inverse_depth_regression <= 0.02,
            "mean_alpha_iou_regression_within_0_02": -alpha_delta >= -0.02,
            "mean_foreground_coverage_regression_within_0_02": -coverage_delta >= -0.02,
        }
        control_candidate_utility = {
            "mean_auc_gain_at_least_0_10_db": statistics.fmean(inverse_auc) >= 0.10,
            "auc_wins_at_least_two_seeds": sum(gain > 0.0 for gain in inverse_auc) >= 2,
        }
        comparisons[control] = {
            "auc_gains_db": auc_gains,
            "mean_auc_gain_db": statistics.fmean(auc_gains),
            "auc_seed_wins": sum(gain > 0.0 for gain in auc_gains),
            "final_psnr_deltas_db": psnr_deltas,
            "ssim_deltas": ssim_deltas,
            "depth_rmse_regression_fraction": depth_regression,
            "alpha_iou_delta": alpha_delta,
            "foreground_coverage_delta": coverage_delta,
            "utility_criteria": utility,
            "safety_criteria": safety,
        }
        utility_criteria[control] = utility
        safety_criteria[control] = safety
        next_candidates[control] = {
            "utility_criteria": control_candidate_utility,
            "safety_criteria": inverse_safety,
            "promising": all(control_candidate_utility.values()) and all(inverse_safety.values()),
        }
    utility_pass = all(
        value for criteria in utility_criteria.values() for value in criteria.values()
    )
    safety_pass = all(value for criteria in safety_criteria.values() for value in criteria.values())
    return {
        "comparisons": comparisons,
        "utility_pass": utility_pass,
        "safety_pass": safety_pass,
        "primary_hypothesis_pass": utility_pass and safety_pass,
        "control_next_candidate_diagnostics": next_candidates,
    }


def strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {value!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def verify_default_semantics() -> dict[str, Any]:
    carve = CarveLifter()
    config = TrainConfig()
    callback_default = inspect.signature(Trainer.train).parameters["checkpoint_callback"].default
    if callback_default is not None:
        raise AssertionError("Trainer checkpoint callback is not zero-default")
    if not carve.merge or carve.merge_voxel_scale != 1.0:
        raise AssertionError("Carve merge defaults differ from the frozen current behavior")
    if config.kernel_support_mode != "hard" or config.visibility_margin_sigma != 3.0:
        raise AssertionError("Trainer hard-render defaults differ from the frozen protocol")
    return {
        "carve_merge_default": carve.merge,
        "carve_merge_voxel_scale_default": carve.merge_voxel_scale,
        "trainer_checkpoint_callback_default_is_none": True,
        "kernel_support_mode": config.kernel_support_mode,
        "visibility_margin_sigma": config.visibility_margin_sigma,
        "base_preregistration_sha256": sha256_file(ROOT / BASE_PREREGISTRATION),
        "retry_preregistration_sha256": sha256_file(ROOT / PREREGISTRATION),
    }


def run_verification() -> dict[str, Any]:
    commands = (
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        [sys.executable, "-m", "pytest", "-q", "-m", "not slow"],
        [sys.executable, "scripts/docs_sync.py"],
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    results = []
    for command in commands:
        print(f"verification: {' '.join(command)}", flush=True)
        started = time.perf_counter()
        completed = subprocess.run(
            command, cwd=ROOT, env=environment, capture_output=True, text=True, check=False
        )
        result = {
            "command": command,
            "returncode": completed.returncode,
            "seconds": time.perf_counter() - started,
            "stdout_sha256": sha256_bytes(completed.stdout.encode()),
            "stderr_sha256": sha256_bytes(completed.stderr.encode()),
            "stdout_tail": completed.stdout[-4_000:],
            "stderr_tail": completed.stderr[-4_000:],
        }
        results.append(result)
        if completed.returncode != 0:
            raise RuntimeError(
                f"verification failed: {' '.join(command)}\n"
                f"{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
            )
    return {"passed": True, "commands": results}


def create_seal() -> dict[str, Any]:
    retry_provenance = verify_retry_provenance()
    environment = environment_metadata()
    assert_official_environment(environment)
    verification = run_verification()
    if verify_retry_provenance() != retry_provenance:
        raise RuntimeError("Retry-2 provenance changed during seal verification")
    hashes, aggregate = source_hashes()
    return {
        "artifact_type": "carve_merge_controls_iter2_implementation_seal",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_metadata(),
        "sealed_paths": [str(path) for path in SEALED_PATHS],
        "source_hashes": hashes,
        "source_aggregate": aggregate,
        "preregistrations": preregistration_bindings(),
        "retry_provenance": retry_provenance,
        "verification": verification,
        "default_semantics": verify_default_semantics(),
        "environment": environment,
        "command": [sys.executable, *sys.argv],
    }


def load_and_verify_seal(path: Path) -> dict[str, Any]:
    retry_provenance = verify_retry_provenance()
    payload = strict_json_load(path)
    if payload.get("artifact_type") != "carve_merge_controls_iter2_implementation_seal":
        raise ValueError(f"{path} is not a Carve merge-controls Retry-2 implementation seal")
    expected_paths = [str(item) for item in SEALED_PATHS]
    if payload.get("sealed_paths") != expected_paths:
        raise RuntimeError("implementation seal path set differs from repository set")
    hashes, aggregate = source_hashes(tuple(Path(item) for item in expected_paths))
    if hashes != payload.get("source_hashes") or aggregate != payload.get("source_aggregate"):
        raise RuntimeError("implementation/protocol differs from sealed source aggregate")
    if not payload.get("verification", {}).get("passed"):
        raise RuntimeError("implementation seal lacks passing verification")
    if payload.get("preregistrations") != preregistration_bindings():
        raise RuntimeError("implementation seal preregistration bindings differ")
    if payload.get("retry_provenance") != retry_provenance:
        raise RuntimeError("implementation seal Retry-2 provenance binding differs")
    current_environment = environment_metadata()
    assert_official_environment(current_environment)
    if environment_fingerprint(payload["environment"]) != environment_fingerprint(
        current_environment
    ):
        raise RuntimeError("current environment differs from implementation seal")
    if payload.get("default_semantics") != verify_default_semantics():
        raise RuntimeError("repository defaults differ from implementation seal")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "source_aggregate": aggregate,
        "verification_sha256": canonical_json_hash(payload["verification"]),
        "environment_fingerprint": environment_fingerprint(payload["environment"]),
        "retry_provenance_sha256": canonical_json_hash(retry_provenance),
    }


def verify_loaded_sources_against_seal(seal_path: Path) -> tuple[dict[str, str], str]:
    payload = strict_json_load(seal_path)
    sealed_hashes = payload["source_hashes"]
    loaded_hashes, loaded_aggregate = loaded_source_hashes()
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
    return loaded_hashes, loaded_aggregate


def run_phase_a(seal_path: Path, output: Path) -> dict[str, Any]:
    preflight_output(output)
    seal = load_and_verify_seal(seal_path)
    verify_default_semantics()
    claim_attempt(
        PHASE_A_ATTEMPT,
        phase="phase_a",
        output=output,
        inputs={
            "seal_sha256": seal["sha256"],
            "source_aggregate": seal["source_aggregate"],
            **retry_attempt_bindings(),
        },
    )
    started = time.perf_counter()
    runs = []
    for seed in SEEDS:
        print(f"Phase A: preparing seed {seed}", flush=True)
        runs.append(phase_a_seed(seed))
    decision = phase_a_decision(runs)
    loaded_hashes, loaded_aggregate = verify_loaded_sources_against_seal(seal_path)
    if load_and_verify_seal(seal_path) != seal:
        raise RuntimeError("implementation seal changed during Phase A")
    return {
        "artifact_type": "carve_merge_controls_iter2_phase_a_audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_metadata(),
        "seal": seal,
        "preregistrations": preregistration_bindings(),
        "command": [sys.executable, *sys.argv],
        "environment": environment_metadata(),
        "default_semantics": verify_default_semantics(),
        "loaded_source_hashes": loaded_hashes,
        "loaded_source_aggregate": loaded_aggregate,
        "split": {"train": TRAIN_INDICES, "held_out": TEST_INDICES},
        "runs": runs,
        "decision": decision,
        "wall_seconds": time.perf_counter() - started,
    }


def _validate_construction_record(construction: dict[str, Any]) -> None:
    count = int(construction["count"])
    keys = torch.tensor(construction["unique_keys"], dtype=torch.int64)
    groups = torch.tensor(construction["group_ids"], dtype=torch.int64)
    counts = torch.tensor(construction["group_counts"], dtype=torch.int64)
    weights = torch.tensor(construction["weights"], dtype=torch.float32)
    audit_weights = torch.tensor(construction["audit_weights_float64"], dtype=torch.float64)
    representatives = torch.tensor(construction["representative_indices"], dtype=torch.int64)
    global_indices = torch.tensor(construction["global_budget_indices"], dtype=torch.int64)
    if keys.shape != (count, 3) or counts.shape != (count,):
        raise RuntimeError("Phase-A group key/count shape differs")
    if groups.ndim != 1 or weights.shape != groups.shape:
        raise RuntimeError("Phase-A group/weight shape differs")
    if audit_weights.shape != groups.shape:
        raise RuntimeError("Phase-A float64 audit-weight shape differs")
    if representatives.shape != (count,) or global_indices.shape != (count,):
        raise RuntimeError("Phase-A selected-index shape differs")
    expected_hashes = {
        "unique_keys": tensor_collection_hash([("unique_keys", keys)]),
        "group_ids": tensor_collection_hash([("group_ids", groups)]),
        "group_counts": tensor_collection_hash([("group_counts", counts)]),
        "weights": tensor_collection_hash([("weights", weights)]),
        "audit_weights_float64": tensor_collection_hash([("audit_weights_float64", audit_weights)]),
        "representative_indices": tensor_collection_hash(
            [("representative_indices", representatives)]
        ),
        "global_budget_indices": tensor_collection_hash(
            [("global_budget_indices", global_indices)]
        ),
    }
    if construction.get("hashes") != expected_hashes:
        raise RuntimeError("Phase-A construction raw-array hashes differ")
    if counts.tolist() != torch.bincount(groups, minlength=count).tolist():
        raise RuntimeError("Phase-A serialized group counts differ")
    arm_hashes = construction.get("arm_hashes")
    if not isinstance(arm_hashes, dict) or set(arm_hashes) != set(REPORTING_ARMS):
        raise RuntimeError("Phase-A arm hashes are incomplete")
    if any(not isinstance(value, str) or len(value) != 64 for value in arm_hashes.values()):
        raise RuntimeError("Phase-A arm hash is malformed")


def validate_phase_a_audit(audit: dict[str, Any], seal: dict[str, Any]) -> None:
    if audit.get("artifact_type") != "carve_merge_controls_iter2_phase_a_audit":
        raise ValueError("Phase-B input is not a Carve merge-controls Retry-2 Phase-A audit")
    if audit.get("split") != {"train": TRAIN_INDICES, "held_out": TEST_INDICES}:
        raise RuntimeError("Phase-A split differs from preregistration")
    if audit.get("seal") != seal:
        raise RuntimeError("Phase-A seal binding differs from current seal")
    if audit.get("preregistrations") != preregistration_bindings():
        raise RuntimeError("Phase-A preregistration bindings differ")
    if environment_fingerprint(audit["environment"]) != seal["environment_fingerprint"]:
        raise RuntimeError("Phase-A environment differs from implementation seal")
    runs = audit.get("runs")
    if not isinstance(runs, list) or [run.get("seed") for run in runs] != SEEDS:
        raise RuntimeError("Phase-A runs are missing, duplicated, or reordered")
    for run in runs:
        seed = int(run["seed"])
        preparation = run.get("preparation", {})
        if preparation.get("fit_config") != asdict(fit_config()):
            raise RuntimeError(f"Phase-A fit config differs for seed {seed}")
        if preparation.get("carve_config") != carve_kwargs(merge=False):
            raise RuntimeError(f"Phase-A Carve config differs for seed {seed}")
        if preparation.get("split") != {"train": TRAIN_INDICES, "held_out": TEST_INDICES}:
            raise RuntimeError(f"Phase-A split metadata differs for seed {seed}")
        _validate_construction_record(run.get("construction", {}))
        recomputed_gate = phase_a_seed_gate_from_raw_evidence(run)
        if canonical_json_hash(recomputed_gate) != canonical_json_hash(run.get("gate")):
            raise RuntimeError(f"Phase-A frozen gate differs for seed {seed}")
    recomputed_decision = phase_a_decision(runs)
    if canonical_json_hash(recomputed_decision) != canonical_json_hash(audit.get("decision")):
        raise RuntimeError("Phase-A decision differs from recomputed frozen gate")


def verify_phase_a_review(path: Path, *, audit_path: Path, seal: dict[str, Any]) -> dict[str, str]:
    review = strict_json_load(path)
    expected = {
        "artifact_type": "carve_merge_controls_iter2_phase_a_scientist_review",
        "verdict": "pass",
        "phase_b_execution_clearance": True,
        "audit_sha256": sha256_file(audit_path),
        "seal_sha256": seal["sha256"],
        "source_aggregate": seal["source_aggregate"],
    }
    if set(review) != set(expected):
        raise RuntimeError("Phase-A scientist review has missing or unexpected keys")
    mismatches = {
        key: {"expected": value, "actual": review.get(key)}
        for key, value in expected.items()
        if review.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Phase-A scientist review is invalid or unbound: {mismatches}")
    return {"path": str(path), "sha256": sha256_file(path)}


def run_phase_b(
    audit_path: Path, seal_path: Path, review_path: Path, output: Path
) -> dict[str, Any]:
    preflight_output(output)
    seal = load_and_verify_seal(seal_path)
    audit_sha256 = sha256_file(audit_path)
    audit = strict_json_load(audit_path)
    validate_phase_a_audit(audit, seal)
    review = verify_phase_a_review(review_path, audit_path=audit_path, seal=seal)
    if not audit["decision"]["phase_b_authorized"]:
        raise RuntimeError("Phase A did not authorize exact-count training")
    claim_attempt(
        PHASE_B_ATTEMPT,
        phase="phase_b",
        output=output,
        inputs={
            "seal_sha256": seal["sha256"],
            "source_aggregate": seal["source_aggregate"],
            "audit_sha256": audit_sha256,
            "review_sha256": review["sha256"],
            **retry_attempt_bindings(),
        },
    )
    phase_a_lookup = {int(run["seed"]): run for run in audit["runs"]}
    runs = []
    recreations = []
    started = time.perf_counter()
    for seed in SEEDS:
        print(f"Phase B: recreating seed {seed}", flush=True)
        scene, _, raw, parity, voxel_size, preparation = prepare_seed(seed, include_parity=False)
        if parity is not None:  # pragma: no cover
            raise AssertionError("Phase B unexpectedly ran a parity recarve")
        phase_a_run = phase_a_lookup[seed]
        phase_a_preparation = phase_a_run["preparation"]
        for field in (
            "scene_hashes",
            "train_scene_hashes",
            "fitted_hash",
            "fit_history_hash",
            "raw_hash",
            "raw_order_hash",
            "raw_count",
            "voxel_size",
            "fit_config",
            "carve_config",
            "split",
        ):
            if canonical_json_hash(preparation[field]) != canonical_json_hash(
                phase_a_preparation[field]
            ):
                raise AssertionError(f"Phase-B recreation differs for seed {seed}/{field}")
        arms, construction = construct_arms(raw, voxel_size)
        if canonical_json_hash(construction) != canonical_json_hash(phase_a_run["construction"]):
            raise AssertionError(f"Phase-B arm construction differs for seed {seed}")
        trained_hashes = {name: construction["arm_hashes"][name] for name in ARMS}
        schedule_probe = isolated_schedule_probe(seed, trained_hashes)
        truths = truth_cache(scene, TEST_INDICES)
        count_set = {arms[name].n for name in ARMS}
        parameter_counts = {parameter_count(arms[name]) for name in ARMS}
        if count_set != {construction["count"]} or len(parameter_counts) != 1:
            raise AssertionError(f"exact count/parameter invariant failed for seed {seed}")
        recreation = {
            "seed": seed,
            "phase_a_preparation_hashes_verified": True,
            "phase_a_construction_hashes_verified": True,
            "preparation": preparation,
            "construction_hash": canonical_json_hash(construction),
            "schedule_probe": schedule_probe,
            "truth_hashes": {str(index): truths[index]["hash"] for index in TEST_INDICES},
            "common_gaussian_count": construction["count"],
            "common_parameter_count": next(iter(parameter_counts)),
            "training_order": TRAINING_ORDER[seed],
        }
        recreations.append(recreation)
        seed_runs = []
        for arm_name in TRAINING_ORDER[seed]:
            print(f"Phase B: training seed {seed}/{arm_name}", flush=True)
            arm_run = train_arm(
                seed=seed,
                arm_name=arm_name,
                initialization=arms[arm_name],
                full_scene=scene,
                truths=truths,
                schedule_probe=schedule_probe,
            )
            if arm_run["initialization_hash"] != trained_hashes[arm_name]:
                raise AssertionError(f"sealed initialization differs for seed {seed}/{arm_name}")
            seed_runs.append(arm_run)
            runs.append(arm_run)
        common_local_hashes = {run["local_schedule_hash"] for run in seed_runs}
        common_global_hashes = {run["global_schedule_hash"] for run in seed_runs}
        if len(common_local_hashes) != 1 or len(common_global_hashes) != 1:
            raise AssertionError(f"training view schedules differ across arms for seed {seed}")
        for history_field in ("active_sh_degree", "n_gaussians"):
            hashes = {canonical_json_hash(run["history"][history_field]) for run in seed_runs}
            if len(hashes) != 1:
                raise AssertionError(f"{history_field} differs across arms for seed {seed}")
    summary = summarize_ablation(runs)
    decision = ablation_decision(summary)
    loaded_hashes, loaded_aggregate = verify_loaded_sources_against_seal(seal_path)
    if load_and_verify_seal(seal_path) != seal:
        raise RuntimeError("implementation seal changed during Phase B")
    if sha256_file(audit_path) != audit_sha256:
        raise RuntimeError("Phase-A audit changed during Phase B")
    if sha256_file(review_path) != review["sha256"]:
        raise RuntimeError("Phase-A scientist review changed during Phase B")
    return {
        "artifact_type": "carve_merge_controls_iter2_phase_b_ablation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_metadata(),
        "seal": seal,
        "phase_a": {"path": str(audit_path), "sha256": audit_sha256},
        "phase_a_scientist_review": review,
        "preregistrations": preregistration_bindings(),
        "command": [sys.executable, *sys.argv],
        "environment": environment_metadata(),
        "loaded_source_hashes": loaded_hashes,
        "loaded_source_aggregate": loaded_aggregate,
        "split": {"train": TRAIN_INDICES, "held_out": TEST_INDICES},
        "recreations": recreations,
        "runs": runs,
        "summary": summary,
        "decision": decision,
        "wall_seconds": time.perf_counter() - started,
    }


def companion_note_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_RESULT.md")


def preflight_output(output: Path) -> None:
    note = companion_note_path(output)
    if output.exists() or note.exists():
        raise FileExistsError(f"refusing to start: {output} or {note} already exists")
    output.parent.mkdir(parents=True, exist_ok=True)


def claim_attempt(path: Path, *, phase: str, output: Path, inputs: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_type": "carve_merge_controls_iter2_once_only_attempt",
        "phase": phase,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output": str(output),
        "inputs": inputs,
        "command": [sys.executable, *sys.argv],
        "environment": environment_metadata(),
    }
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    except FileExistsError as error:
        raise RuntimeError(
            f"the preregistered {phase} attempt has already been claimed by {path}"
        ) from error


def result_note(payload: dict[str, Any], output: Path, digest: str) -> str:
    lines = [
        f"# {payload['artifact_type']}",
        "",
        f"- Timestamp (UTC): `{payload['timestamp_utc']}`",
        f"- JSON artifact: `{output}`",
        f"- JSON SHA-256: `{digest}`",
        f"- Command: `{' '.join(payload['command'])}`",
    ]
    if "seal" in payload:
        lines.append(f"- Implementation seal: `{payload['seal']['source_aggregate']}`")
    if payload["artifact_type"] == "carve_merge_controls_iter2_phase_a_audit":
        decision = payload["decision"]
        disposition = (
            "Phase B remains blocked until an independent scientist review binds and clears "
            "this exact audit."
            if decision["phase_b_authorized"]
            else "The frozen materiality/preservation gate failed, so Phase B is forbidden."
        )
        lines.extend(
            [
                "",
                "## Frozen gate decision",
                "",
                f"- Seed passes: `{decision['seed_passes']}`",
                f"- Phase B authorized: `{decision['phase_b_authorized']}`",
                "",
                disposition,
            ]
        )
    elif payload["artifact_type"] == "carve_merge_controls_iter2_phase_b_ablation":
        decision = payload["decision"]
        lines.extend(
            [
                "",
                "## Frozen outcome decision",
                "",
                f"- Primary hypothesis pass: `{decision['primary_hypothesis_pass']}`",
                f"- Utility pass: `{decision['utility_pass']}`",
                f"- Safety pass: `{decision['safety_pass']}`",
                "",
                "Checkpoint-AUC held-out foreground PSNR is primary. This remains a "
                "fixed-topology CPU synthetic result and cannot change a default.",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    seal = subparsers.add_parser("seal", help="verify and freeze the complete implementation")
    seal.add_argument("--output", type=Path, default=DEFAULT_SEAL)
    audit = subparsers.add_parser("audit", help="run the Phase-A construction audit")
    audit.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    audit.add_argument("--output", type=Path, required=True)
    ablate = subparsers.add_parser("ablate", help="run Phase B after bound authorization")
    ablate.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    ablate.add_argument("--audit", type=Path, required=True)
    ablate.add_argument("--phase-a-review", type=Path, required=True)
    ablate.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    expected_suffix = {
        "audit": "_cpu_carve_merge_controls_iter2_audit.json",
        "ablate": "_cpu_carve_merge_controls_iter2_ablation.json",
    }
    if args.command_name in expected_suffix and not args.output.name.endswith(
        expected_suffix[args.command_name]
    ):
        raise ValueError(
            f"official {args.command_name} output must end with "
            f"{expected_suffix[args.command_name]!r}"
        )
    if args.command_name == "seal" and args.output != DEFAULT_SEAL:
        raise ValueError("official seal must use the preregistered fixed path")
    preflight_output(args.output)
    if args.command_name == "seal":
        payload = create_seal()
    elif args.command_name == "audit":
        payload = run_phase_a(args.seal, args.output)
    elif args.command_name == "ablate":
        payload = run_phase_b(args.audit, args.seal, args.phase_a_review, args.output)
    else:  # pragma: no cover
        raise AssertionError(args.command_name)
    note, digest = write_artifact(args.output, payload)
    print(f"saved {args.output} (sha256={digest})", flush=True)
    print(f"saved {note}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
