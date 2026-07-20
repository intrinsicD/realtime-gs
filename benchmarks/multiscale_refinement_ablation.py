#!/usr/bin/env python3
"""Preregistered fixed-topology 24-to-48 multiscale refinement ablation."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import __version__ as pillow_version

from rtgs.core.camera import Camera
from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.metrics import ssim
from rtgs.data.scene import SceneData
from rtgs.data.synthetic import make_synthetic_scene
from rtgs.image2gs.fit import FitConfig, fit_views
from rtgs.lift.carve import CarveLifter
from rtgs.optim.trainer import (
    TrainConfig,
    Trainer,
    TrainStepControl,
    area_downsample_2x,
    downscale_camera,
)
from rtgs.render.torch_ref import TorchRasterizer

ROOT = Path(__file__).resolve().parent.parent
PREREGISTRATION = Path("benchmarks/results/20260716_multiscale_refinement_PREREG.md")
PREREGISTRATION_SHA256 = "b4c17da489a6e66950ec15ecda78dd7ddb063d65055e25bb4f76df6e8cca0a59"
IMPLEMENTATION_REVIEW = Path(
    "benchmarks/results/20260716_multiscale_refinement_IMPLEMENTATION_REVIEW.md"
)
DEFAULT_SEAL = ROOT / "benchmarks/results/20260716_multiscale_refinement_SEAL.json"
ATTEMPT = ROOT / "benchmarks/results/20260716_multiscale_refinement_ATTEMPT.json"
ARTIFACT_TYPE = "multiscale_refinement_ablation"
SEAL_ARTIFACT_TYPE = "multiscale_refinement_implementation_seal"
ATTEMPT_ARTIFACT_TYPE = "multiscale_refinement_once_only_attempt"
INVALID_ARTIFACT_TYPE = "multiscale_refinement_invalid_attempt"

SEEDS = (3, 4, 5)
TRAIN_INDICES = (0, 1, 2, 4, 5, 6, 8, 9, 10)
TEST_INDICES = (3, 7, 11)
ARMS = ("full", "camera_blocked", "pyramid_blocked", "camera_interleaved")
CHECKPOINT_STEPS = (0, 30, 60, 90, 120)
ARM_ORDERS = {
    3: ("full", "camera_blocked", "pyramid_blocked", "camera_interleaved"),
    4: ("camera_blocked", "pyramid_blocked", "camera_interleaved", "full"),
    5: ("pyramid_blocked", "camera_interleaved", "full", "camera_blocked"),
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
GAUSSIAN_FIELDS = ("means", "quats", "log_scales", "opacity", "sh")

SEALED_PATHS = tuple(
    sorted(
        {
            PREREGISTRATION,
            IMPLEMENTATION_REVIEW,
            Path("benchmarks/multiscale_refinement_ablation.py"),
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


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_json_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {value!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    header = canonical_json(
        {"dtype": str(value.dtype), "shape": list(value.shape), "stride": list(value.stride())}
    ).encode()
    return sha256_bytes(header + b"\0" + value.numpy().tobytes(order="C"))


def tensor_collection_hash(items: list[tuple[str, torch.Tensor]]) -> str:
    manifest = [{"name": name, "sha256": tensor_hash(value)} for name, value in items]
    return canonical_json_hash(manifest)


def camera_record(camera: Camera) -> dict[str, Any]:
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


def camera_hash(camera: Camera) -> str:
    return canonical_json_hash(camera_record(camera))


def gaussians_hashes(gaussians: Gaussians3D) -> dict[str, Any]:
    fields = {name: tensor_hash(getattr(gaussians, name)) for name in GAUSSIAN_FIELDS}
    return {
        "n": gaussians.n,
        "sh_degree": gaussians.sh_degree,
        "fields": fields,
        "aggregate": canonical_json_hash(fields),
    }


def gaussians2d_hashes(values: list[Gaussians2D]) -> dict[str, Any]:
    views = []
    for index, gaussians in enumerate(values):
        fields = {
            name: tensor_hash(getattr(gaussians, name))
            for name in ("xy", "chol", "color", "weight")
        }
        views.append(
            {
                "view": index,
                "n": gaussians.n,
                "fields": fields,
                "aggregate": canonical_json_hash(fields),
            }
        )
    return {"views": views, "aggregate": canonical_json_hash(views)}


def assert_finite_gaussians(gaussians: Gaussians3D, context: str) -> None:
    if gaussians.n <= 0:
        raise RuntimeError(f"{context} has no primitives")
    for name in GAUSSIAN_FIELDS:
        if not bool(torch.isfinite(getattr(gaussians, name)).all()):
            raise RuntimeError(f"{context} contains non-finite {name}")


def assert_finite_tree(value: Any, context: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"{context} contains a non-finite float")
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_tree(item, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_tree(item, f"{context}[{index}]")


def source_hashes(
    paths: tuple[Path, ...] = SEALED_PATHS, *, root: Path = ROOT
) -> tuple[dict[str, str], str]:
    missing = [str(path) for path in paths if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"sealed source files are missing: {missing}")
    hashes = {str(path): sha256_file(root / path) for path in paths}
    return hashes, canonical_json_hash(hashes)


def verify_source_manifest(
    paths: tuple[Path, ...],
    expected_hashes: dict[str, str],
    expected_aggregate: str,
    *,
    root: Path = ROOT,
) -> None:
    hashes, aggregate = source_hashes(paths, root=root)
    if hashes != expected_hashes or aggregate != expected_aggregate:
        raise RuntimeError("source manifest differs from sealed hashes")


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
        if not relative.parts or relative.parts[0] == ".venv":
            continue
        paths.add(path)
    for relative in (PREREGISTRATION, IMPLEMENTATION_REVIEW, Path("pyproject.toml")):
        paths.add((ROOT / relative).resolve())
    hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(paths)}
    return hashes, canonical_json_hash(hashes)


def git_metadata() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    return {
        "revision": revision,
        "dirty": bool(status.strip()),
        "status": status.splitlines(),
        "tracked_diff": diff,
        "tracked_diff_sha256": sha256_bytes(diff.encode()),
    }


def _loadavg() -> str | None:
    path = Path("/proc/loadavg")
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def environment_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pillow": pillow_version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "device": "cpu",
        "loadavg": _loadavg(),
    }


def environment_fingerprint(metadata: dict[str, Any]) -> dict[str, Any]:
    ignored = {"loadavg"}
    return {key: value for key, value in metadata.items() if key not in ignored}


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


def fit_config() -> FitConfig:
    return FitConfig(
        n_gaussians=150,
        max_gaussians=5000,
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


def carve_lifter() -> CarveLifter:
    return CarveLifter(
        grid_res=48,
        bounds_scale=0.5,
        min_views=2,
        hull_fraction=0.85,
        color_std_sigma=0.20,
        color_match_sigma=0.35,
        coverage_thresh=0.40,
        samples_per_ray=64,
        min_score=0.05,
        min_weight=0.05,
        merge=True,
        merge_voxel_scale=1.0,
        init_opacity=0.1,
        sh_degree=0,
    )


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
        density_strategy="classic",
        eval_every=30,
        target_sh_degree=0,
        sh_degree_interval=120,
        use_masks=False,
        outside_alpha_lambda=0.01,
        mask_alpha_lambda=0.05,
        random_background=False,
        opacity_reg=None,
        scale_reg=None,
        packed=False,
        antialiased=False,
        sh_color_activation="hard",
        collect_sh_color_diagnostics=False,
        kernel_support_mode="hard",
        collect_kernel_support_diagnostics=False,
        visibility_margin_sigma=3.0,
        validate_render_finite=True,
        quaternion_update_policy="current",
        seed=seed,
    )


def control_sequence(arm: str, *, iterations: int = 120) -> tuple[TrainStepControl, ...]:
    if arm not in ARMS:
        raise ValueError(f"unknown multiscale arm {arm!r}")
    if iterations != 120:
        raise ValueError("official arm schedules require exactly 120 iterations")
    full = TrainStepControl(1, 1)
    half_camera = TrainStepControl(2, 2)
    half_loss = TrainStepControl(1, 2)
    if arm == "full":
        return (full,) * 120
    if arm == "camera_blocked":
        return (half_camera,) * 60 + (full,) * 60
    if arm == "pyramid_blocked":
        return (half_loss,) * 60 + (full,) * 60
    return tuple(half_camera if step % 2 else full for step in range(1, 121))


def trainer_step_controls(arm: str) -> tuple[TrainStepControl, ...] | None:
    return None if arm == "full" else control_sequence(arm)


def schedule_record(arm: str) -> dict[str, Any]:
    controls = control_sequence(arm)
    sequence = [[item.render_downscale, item.loss_downscale] for item in controls]
    return {
        "arm": arm,
        "sequence": sequence,
        "sha256": canonical_json_hash(sequence),
        "render_scale_counts": {
            str(scale): sum(item.render_downscale == scale for item in controls) for scale in (1, 2)
        },
        "loss_scale_counts": {
            str(scale): sum(item.loss_downscale == scale for item in controls) for scale in (1, 2)
        },
        "last_step": sequence[-1],
    }


def exposure_record(arm: str, *, width: int = 48, height: int = 48) -> dict[str, Any]:
    render_pixels = 0
    loss_pixels = 0
    for control in control_sequence(arm):
        render_pixels += (height // control.render_downscale) * (width // control.render_downscale)
        loss_pixels += (height // control.loss_downscale) * (width // control.loss_downscale)
    baseline = 120 * height * width
    return {
        "render_pixels": render_pixels,
        "loss_pixels": loss_pixels,
        "full_render_pixels": baseline,
        "render_ratio": render_pixels / baseline,
    }


def frozen_schedules() -> dict[str, Any]:
    schedules = {arm: schedule_record(arm) for arm in ARMS}
    expected_exposures = {
        "full": (276480, 276480),
        "camera_blocked": (172800, 172800),
        "pyramid_blocked": (276480, 172800),
        "camera_interleaved": (172800, 172800),
    }
    for arm, expected in expected_exposures.items():
        exposure = exposure_record(arm)
        actual = (exposure["render_pixels"], exposure["loss_pixels"])
        if actual != expected:
            raise RuntimeError(f"frozen exposure mismatch for {arm}: {actual} != {expected}")
        schedules[arm]["exposure"] = exposure
    for seed, order in ARM_ORDERS.items():
        if set(order) != set(ARMS) or len(order) != len(ARMS):
            raise RuntimeError(f"invalid counterbalanced arm order for seed {seed}")
    return {
        "arms": schedules,
        "orders": {str(seed): list(order) for seed, order in ARM_ORDERS.items()},
    }


def probe_view_schedule(seed: int, *, n_views: int = 9, iterations: int = 120) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return [int(torch.randint(0, n_views, (1,), generator=generator)) for _ in range(iterations)]


def per_view_scale_counts(
    controls: tuple[TrainStepControl, ...], sampled_views: list[int]
) -> dict[str, Any]:
    if len(controls) != len(sampled_views):
        raise ValueError("controls and sampled views must have equal lengths")
    result: dict[str, Any] = {}
    for control, view in zip(controls, sampled_views, strict=True):
        record = result.setdefault(
            str(view), {"render": {"1": 0, "2": 0}, "loss": {"1": 0, "2": 0}}
        )
        record["render"][str(control.render_downscale)] += 1
        record["loss"][str(control.loss_downscale)] += 1
    return result


def _heldout_spatial_pixels(checkpoint: dict[str, Any], *, expected_step: int) -> int:
    if int(checkpoint.get("step", -1)) != expected_step:
        raise RuntimeError(f"held-out checkpoint step differs from {expected_step}")
    per_view = checkpoint.get("per_view", [])
    if [int(item.get("view", -1)) for item in per_view] != list(TEST_INDICES):
        raise RuntimeError(f"held-out view set differs at step {expected_step}")
    expected_rgb_values = 48 * 48 * 3
    for item in per_view:
        if int(item.get("evidence", {}).get("full_rgb_value_count", -1)) != expected_rgb_values:
            raise RuntimeError(
                f"held-out checkpoint is not a full 48x48 render at step {expected_step}"
            )
    return len(per_view) * 48 * 48


def runtime_exposure_accounting(
    arm_records: list[dict[str, Any]], step_zero: dict[str, Any]
) -> dict[str, Any]:
    """Recompute every optimization and evaluation exposure from runtime records."""
    if {record.get("arm") for record in arm_records} != set(ARMS) or len(arm_records) != len(ARMS):
        raise RuntimeError("runtime exposure accounting requires exactly the four frozen arms")
    manual_step_zero_pixels = _heldout_spatial_pixels(step_zero, expected_step=0)
    if manual_step_zero_pixels != 6912:
        raise RuntimeError("manual step-zero exposure differs from 3 * 48 * 48")

    by_arm: dict[str, Any] = {}
    for arm_record in arm_records:
        arm = str(arm_record["arm"])
        history = arm_record["history"]
        sampled_views = history.get("sampled_train_views", [])
        if len(sampled_views) != 120 or any(
            int(view) not in range(len(TRAIN_INDICES)) for view in sampled_views
        ):
            raise RuntimeError(f"optimization view exposure differs for {arm}")
        controls = control_sequence(arm)
        render_pixels = sum(
            (48 // control.render_downscale) * (48 // control.render_downscale)
            for control in controls
        )
        loss_pixels = sum(
            (48 // control.loss_downscale) * (48 // control.loss_downscale) for control in controls
        )
        expected_optimization = exposure_record(arm)
        if (render_pixels, loss_pixels) != (
            expected_optimization["render_pixels"],
            expected_optimization["loss_pixels"],
        ):
            raise RuntimeError(f"runtime optimization exposure differs for {arm}")
        if arm == "full":
            if "step_control_metadata" in history:
                raise RuntimeError("full baseline has unexpected step-control exposure metadata")
        else:
            metadata = history.get("step_control_metadata", {})
            if metadata.get("render_pixels") != render_pixels:
                raise RuntimeError(f"Trainer render exposure differs for {arm}")
            if metadata.get("loss_pixels") != loss_pixels:
                raise RuntimeError(f"Trainer loss exposure differs for {arm}")
            if metadata.get("sequence_sha256") != schedule_record(arm)["sha256"]:
                raise RuntimeError(f"Trainer schedule exposure binding differs for {arm}")
            expected_counts = per_view_scale_counts(controls, list(sampled_views))
            if metadata.get("per_view_scale_counts") != expected_counts:
                raise RuntimeError(f"Trainer per-view exposure differs for {arm}")

        native_steps = [int(item[0]) for item in history.get("psnr", [])]
        if native_steps != [30, 60, 90, 120]:
            raise RuntimeError(f"native evaluation exposure schedule differs for {arm}")
        native_evaluation_pixels = len(native_steps) * len(TRAIN_INDICES) * 48 * 48
        if native_evaluation_pixels != 82944:
            raise RuntimeError(f"native evaluation exposure differs for {arm}")

        checkpoints = arm_record.get("checkpoints", [])
        if len(checkpoints) != len(CHECKPOINT_STEPS):
            raise RuntimeError(f"held-out checkpoint count differs for {arm}")
        if checkpoints[0].get("record_sha256") != step_zero.get("record_sha256"):
            raise RuntimeError(f"manual step-zero checkpoint differs for {arm}")
        heldout_callback_pixels = sum(
            _heldout_spatial_pixels(checkpoint, expected_step=step)
            for step, checkpoint in zip(CHECKPOINT_STEPS[1:], checkpoints[1:], strict=True)
        )
        if heldout_callback_pixels != 27648:
            raise RuntimeError(f"held-out callback exposure differs for {arm}")

        by_arm[arm] = {
            "optimization": {
                "sampled_training_views": len(sampled_views),
                "render_pixels": render_pixels,
                "loss_pixels": loss_pixels,
                "full_render_reference_pixels": 276480,
                "render_ratio": render_pixels / 276480,
                "included_in_optimization_ratio": True,
            },
            "native_training_evaluation": {
                "checkpoint_steps": native_steps,
                "training_views_per_checkpoint": len(TRAIN_INDICES),
                "pixels_per_view": 48 * 48,
                "render_pixels": native_evaluation_pixels,
                "included_in_optimization_ratio": False,
            },
            "held_out_checkpoint_callbacks": {
                "checkpoint_steps": list(CHECKPOINT_STEPS[1:]),
                "held_out_views_per_checkpoint": len(TEST_INDICES),
                "pixels_per_view": 48 * 48,
                "render_pixels": heldout_callback_pixels,
                "included_in_optimization_ratio": False,
            },
        }
    record = {
        "unit": "spatial_pixels",
        "by_arm": by_arm,
        "shared_manual_step_zero": {
            "checkpoint_steps": [0],
            "held_out_views": len(TEST_INDICES),
            "pixels_per_view": 48 * 48,
            "render_pixels": manual_step_zero_pixels,
            "scope": "once_per_seed_shared_by_all_arms",
            "included_in_optimization_ratio": False,
        },
        "optimization_ratio_excludes": [
            "native_training_evaluation",
            "held_out_checkpoint_callbacks",
            "shared_manual_step_zero",
        ],
    }
    record["record_sha256"] = canonical_json_hash(record)
    return record


@dataclass(frozen=True)
class TruthView:
    view: int
    target: torch.Tensor
    camera: Camera
    color: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor
    support: torch.Tensor
    record: dict[str, Any]


def _hard_renderer() -> TorchRasterizer:
    return TorchRasterizer(
        sh_color_activation="hard",
        collect_sh_color_diagnostics=False,
        kernel_support_mode="hard",
        collect_kernel_support_diagnostics=False,
        visibility_margin_sigma=3.0,
    )


def construct_truth(full_scene: SceneData) -> tuple[TruthView, ...]:
    if full_scene.gt_gaussians is None or full_scene.gt_gaussians.sh_degree != 0:
        raise RuntimeError("official held-out truth requires degree-zero GT gaussians")
    renderer = _hard_renderer()
    truths = []
    with torch.no_grad():
        for view in TEST_INDICES:
            camera = full_scene.cameras[view]
            target = full_scene.images[view]
            output = renderer.render(
                full_scene.gt_gaussians, camera, background=torch.zeros(3), sh_degree=0
            )
            for name, value in (
                ("target", target),
                ("color", output.color),
                ("alpha", output.alpha),
                ("depth", output.depth),
            ):
                if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                    raise RuntimeError(f"held-out truth {name} is invalid in view {view}")
            support = output.alpha > 0.05
            if not bool(support.any()):
                raise RuntimeError(f"held-out truth support is empty in view {view}")
            record = {
                "view": view,
                "camera": camera_record(camera),
                "target_sha256": tensor_hash(target),
                "truth_color_sha256": tensor_hash(output.color),
                "truth_alpha_sha256": tensor_hash(output.alpha),
                "truth_depth_sha256": tensor_hash(output.depth),
                "truth_expected_depth_sha256": tensor_hash(
                    output.depth / output.alpha.clamp_min(1e-6)
                ),
                "truth_support_sha256": tensor_hash(support),
            }
            record["aggregate_sha256"] = canonical_json_hash(record)
            truths.append(
                TruthView(
                    view=view,
                    target=target.detach().clone(),
                    camera=camera,
                    color=output.color.detach().clone(),
                    alpha=output.alpha.detach().clone(),
                    depth=output.depth.detach().clone(),
                    support=support.detach().clone(),
                    record=record,
                )
            )
    return tuple(truths)


def _crop_bounds(mask: torch.Tensor) -> tuple[int, int, int, int]:
    yy, xx = torch.where(mask)
    if yy.numel() == 0:
        raise ValueError("truth support is empty")
    height, width = mask.shape
    margin = max(1, round(max(height, width) * 0.05))
    y0 = max(0, int(yy.min()) - margin)
    y1 = min(height, int(yy.max()) + 1 + margin)
    x0 = max(0, int(xx.min()) - margin)
    x1 = min(width, int(xx.max()) + 1 + margin)
    if y1 <= y0 or x1 <= x0:
        raise ValueError("truth-support crop is empty")
    return y0, y1, x0, x1


def recompute_metrics_from_evidence(evidence: dict[str, Any]) -> dict[str, float]:
    def positive_number(name: str) -> float:
        value = float(evidence[name])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return value

    def nonnegative_number(name: str) -> float:
        value = float(evidence[name])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return value

    foreground_count = positive_number("foreground_rgb_value_count")
    full_count = positive_number("full_rgb_value_count")
    crop_count = positive_number("crop_rgb_value_count")
    depth_count = positive_number("depth_intersection_pixel_count")
    union_count = positive_number("alpha_union_pixel_count")
    truth_count = positive_number("truth_support_pixel_count")
    foreground_mse = nonnegative_number("foreground_rgb_squared_error_sum") / foreground_count
    full_mse = nonnegative_number("full_rgb_squared_error_sum") / full_count
    crop_mse = nonnegative_number("crop_rgb_squared_error_sum") / crop_count
    depth_mse = nonnegative_number("depth_squared_error_sum") / depth_count
    intersection = positive_number("alpha_intersection_pixel_count")
    extent = positive_number("extent")
    result = {
        "psnr_fg": -10.0 * math.log10(max(foreground_mse, 1e-12)),
        "psnr_full": -10.0 * math.log10(max(full_mse, 1e-12)),
        "psnr_crop": -10.0 * math.log10(max(crop_mse, 1e-12)),
        "ssim_crop": float(evidence["ssim_crop"]),
        "depth_rmse_over_extent": math.sqrt(depth_mse) / extent,
        "alpha_iou": intersection / union_count,
        "foreground_coverage": intersection / truth_count,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("recomputed held-out metrics are non-finite")
    return result


def _metric_evidence(
    predicted_color: torch.Tensor,
    predicted_alpha: torch.Tensor,
    predicted_depth: torch.Tensor,
    truth: TruthView,
    *,
    extent: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    target = truth.target
    expected_shapes = ((48, 48, 3), (48, 48), (48, 48))
    values = (predicted_color, predicted_alpha, predicted_depth)
    if tuple(value.shape for value in values) != expected_shapes:
        raise ValueError("held-out renderer fields do not have the frozen 48x48 shapes")
    if target.shape != (48, 48, 3) or target.dtype != torch.float32:
        raise ValueError("held-out target does not have frozen float32 RGB shape")
    for value in (*values, target, truth.alpha, truth.depth):
        if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError("held-out metric input is not finite float32")

    support = truth.support
    predicted_support = predicted_alpha > 0.05
    intersection = support & predicted_support
    union = support | predicted_support
    if not bool(support.any()) or not bool(intersection.any()) or not bool(union.any()):
        raise ValueError("held-out truth/intersection/union support is empty")
    if not math.isfinite(extent) or extent <= 0.0:
        raise ValueError("scene extent must be finite and positive")

    color_error = predicted_color.double() - target.double()
    foreground_sse = (color_error.square() * support.double()[..., None]).sum(dtype=torch.float64)
    full_sse = color_error.square().sum(dtype=torch.float64)
    y0, y1, x0, x1 = _crop_bounds(support)
    float_support = support.to(torch.float32)[..., None]
    predicted_crop = (predicted_color * float_support)[y0:y1, x0:x1]
    target_crop = (target * float_support)[y0:y1, x0:x1]
    crop_error = predicted_crop.double() - target_crop.double()
    crop_sse = crop_error.square().sum(dtype=torch.float64)
    expected_depth = predicted_depth / predicted_alpha.clamp_min(1e-6)
    truth_expected_depth = truth.depth / truth.alpha.clamp_min(1e-6)
    depth_error = expected_depth.double() - truth_expected_depth.double()
    depth_sse = (depth_error.square() * intersection.double()).sum(dtype=torch.float64)
    intersection_count = int(intersection.sum())
    evidence = {
        "foreground_rgb_squared_error_sum": float(foreground_sse),
        "foreground_rgb_value_count": 3 * int(support.sum()),
        "full_rgb_squared_error_sum": float(full_sse),
        "full_rgb_value_count": 48 * 48 * 3,
        "crop_rgb_squared_error_sum": float(crop_sse),
        "crop_rgb_value_count": (y1 - y0) * (x1 - x0) * 3,
        "depth_squared_error_sum": float(depth_sse),
        "depth_intersection_pixel_count": intersection_count,
        "alpha_intersection_pixel_count": intersection_count,
        "alpha_union_pixel_count": int(union.sum()),
        "truth_support_pixel_count": int(support.sum()),
        "extent": float(extent),
        "ssim_crop": float(ssim(predicted_crop, target_crop, window_size=11)),
    }
    return evidence, recompute_metrics_from_evidence(evidence)


def evaluate_checkpoint(
    gaussians: Gaussians3D,
    truths: tuple[TruthView, ...],
    *,
    extent: float,
    step: int,
) -> dict[str, Any]:
    if step not in CHECKPOINT_STEPS:
        raise ValueError(f"unexpected held-out checkpoint step {step}")
    assert_finite_gaussians(gaussians, f"held-out checkpoint {step}")
    if gaussians.sh_degree != 0:
        raise ValueError("held-out checkpoint requires degree-zero gaussians")
    renderer = _hard_renderer()
    before = gaussians_hashes(gaussians)
    per_view = []
    with torch.no_grad():
        for truth in truths:
            if truth.camera.width != 48 or truth.camera.height != 48:
                raise ValueError("held-out evaluation camera is not full resolution")
            output = renderer.render(
                gaussians, truth.camera, background=torch.zeros(3), sh_degree=0
            )
            evidence, metrics = _metric_evidence(
                output.color, output.alpha, output.depth, truth, extent=extent
            )
            if canonical_json_hash(metrics) != canonical_json_hash(
                recompute_metrics_from_evidence(evidence)
            ):
                raise RuntimeError("held-out metric recomputation changed")
            record = {
                "view": truth.view,
                "step": step,
                "primitive_count": gaussians.n,
                "crop_bounds": list(_crop_bounds(truth.support)),
                "support_sha256": tensor_hash(truth.support),
                "intersection_sha256": tensor_hash(truth.support & (output.alpha > 0.05)),
                "union_sha256": tensor_hash(truth.support | (output.alpha > 0.05)),
                "renderer_fields": {
                    "color_sha256": tensor_hash(output.color),
                    "alpha_sha256": tensor_hash(output.alpha),
                    "depth_sha256": tensor_hash(output.depth),
                },
                "truth_sha256": truth.record["aggregate_sha256"],
                "evidence": evidence,
                "metrics": metrics,
            }
            record["record_sha256"] = canonical_json_hash(record)
            per_view.append(record)
    if gaussians_hashes(gaussians) != before:
        raise RuntimeError("held-out checkpoint evaluation mutated the Gaussian snapshot")
    means = {
        metric: statistics.fmean(float(record["metrics"][metric]) for record in per_view)
        for metric in METRICS
    }
    result = {
        "step": step,
        "per_view": per_view,
        "mean": means,
        "primitive_count": gaussians.n,
        "active_sh_degree": gaussians.sh_degree,
    }
    result["record_sha256"] = canonical_json_hash(result)
    return result


def validate_checkpoint_record(record: dict[str, Any]) -> None:
    expected_hash = record.get("record_sha256")
    without_hash = {key: value for key, value in record.items() if key != "record_sha256"}
    if expected_hash != canonical_json_hash(without_hash):
        raise ValueError("checkpoint record hash mismatch")
    if int(record["step"]) not in CHECKPOINT_STEPS:
        raise ValueError("checkpoint step is outside the frozen schedule")
    per_view = record["per_view"]
    if [int(item["view"]) for item in per_view] != list(TEST_INDICES):
        raise ValueError("checkpoint held-out views differ from the frozen split")
    for item in per_view:
        item_hash = item.get("record_sha256")
        item_without_hash = {key: value for key, value in item.items() if key != "record_sha256"}
        if item_hash != canonical_json_hash(item_without_hash):
            raise ValueError("per-view held-out record hash mismatch")
        metrics = recompute_metrics_from_evidence(item["evidence"])
        if canonical_json_hash(metrics) != canonical_json_hash(item["metrics"]):
            raise ValueError("per-view metric differs from raw evidence")
        evidence = item["evidence"]
        truth_count = int(evidence["truth_support_pixel_count"])
        intersection_count = int(evidence["alpha_intersection_pixel_count"])
        if int(evidence["foreground_rgb_value_count"]) != 3 * truth_count:
            raise ValueError("foreground RGB count differs from truth support")
        if int(evidence["full_rgb_value_count"]) != 48 * 48 * 3:
            raise ValueError("full RGB count differs from the frozen canvas")
        if int(evidence["depth_intersection_pixel_count"]) != intersection_count:
            raise ValueError("depth and alpha intersection counts differ")
        if int(evidence["alpha_union_pixel_count"]) < intersection_count:
            raise ValueError("alpha union is smaller than the intersection")
        if truth_count < intersection_count:
            raise ValueError("truth support is smaller than the intersection")
        y0, y1, x0, x1 = (int(value) for value in item["crop_bounds"])
        if int(evidence["crop_rgb_value_count"]) != (y1 - y0) * (x1 - x0) * 3:
            raise ValueError("crop RGB count differs from the serialized bounds")
        if int(item["step"]) != int(record["step"]):
            raise ValueError("per-view and checkpoint steps differ")
        if int(item["primitive_count"]) != int(record["primitive_count"]):
            raise ValueError("per-view and checkpoint primitive counts differ")
    means = {
        metric: statistics.fmean(float(item["metrics"][metric]) for item in per_view)
        for metric in METRICS
    }
    if canonical_json_hash(means) != canonical_json_hash(record["mean"]):
        raise ValueError("checkpoint arithmetic view means differ")
    if int(record["active_sh_degree"]) != 0 or int(record["primitive_count"]) <= 0:
        raise ValueError("checkpoint topology or SH degree differs from the frozen setup")


def normalized_auc(checkpoints: list[dict[str, Any]], metric: str) -> float:
    if [int(record["step"]) for record in checkpoints] != list(CHECKPOINT_STEPS):
        raise ValueError("AUC checkpoints differ from the frozen schedule")
    values = [float(record["mean"][metric]) for record in checkpoints]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("AUC contains non-finite values")
    area = 0.0
    for left, right, value_left, value_right in zip(
        CHECKPOINT_STEPS[:-1], CHECKPOINT_STEPS[1:], values[:-1], values[1:], strict=True
    ):
        area += (right - left) * (value_left + value_right) / 2.0
    return area / 120.0


def validate_reported_auc(checkpoints: list[dict[str, Any]], metric: str, reported: float) -> None:
    recomputed = normalized_auc(checkpoints, metric)
    if recomputed != float(reported):
        raise ValueError("reported AUC differs from frozen trapezoidal recomputation")


def scene_hashes(scene: SceneData) -> dict[str, Any]:
    center, extent = scene.center_and_extent()
    record = {
        "name": scene.name,
        "images": [tensor_hash(image) for image in scene.images],
        "cameras": [camera_hash(camera) for camera in scene.cameras],
        "view_names": scene.view_names,
        "points_sha256": None if scene.points is None else tensor_hash(scene.points),
        "point_visibility": (
            None
            if scene.point_visibility is None
            else [tensor_hash(value) for value in scene.point_visibility]
        ),
        "masks": None if scene.masks is None else [tensor_hash(mask) for mask in scene.masks],
        "train_indices": scene.train_indices,
        "test_indices": scene.test_indices,
        "bounds_hint": (
            None
            if scene.bounds_hint is None
            else {
                "center_sha256": tensor_hash(scene.bounds_hint[0]),
                "extent": float(scene.bounds_hint[1]),
            }
        ),
        "resolved_center_sha256": tensor_hash(center),
        "resolved_extent": float(extent),
    }
    record["aggregate_sha256"] = canonical_json_hash(record)
    return record


def verify_training_pyramid_invariants(scene: SceneData) -> dict[str, Any]:
    if scene.n_views != 9 or scene.training_views != list(range(9)) or scene.testing_views:
        raise ValueError("runtime pyramid invariants require the physical nine-view training set")
    if scene.points is None:
        raise ValueError("runtime pyramid invariants require frozen sparse points")
    finite_points = scene.points[torch.isfinite(scene.points).all(dim=-1)]
    if finite_points.numel() == 0:
        raise ValueError("runtime pyramid invariants require finite sparse points")
    test_points = finite_points[: min(32, finite_points.shape[0])]
    before = scene_hashes(scene)
    per_view = []
    maximum_projection_error = 0.0
    maximum_ray_error = 0.0
    maximum_direct_float64_pool_error = 0.0
    for view, (image, camera) in enumerate(zip(scene.images, scene.cameras, strict=True)):
        if image.shape != (48, 48, 3) or (camera.width, camera.height) != (48, 48):
            raise ValueError(f"training view {view} is not the frozen 48x48 resolution")
        half_camera = downscale_camera(camera, 2)
        half_image = area_downsample_2x(image)
        if half_image.shape != (24, 24, 3) or (half_camera.width, half_camera.height) != (24, 24):
            raise RuntimeError(f"training view {view} half-resolution dimensions differ")
        direct_float64 = image.to(torch.float64).reshape(24, 2, 24, 2, 3).sum(dim=(1, 3)) / 4.0
        pooled_float64 = half_image.to(torch.float64)
        direct_pool_error = float((pooled_float64 - direct_float64).abs().max())
        if not torch.allclose(
            pooled_float64,
            direct_float64,
            atol=1e-7,
            rtol=1e-6,
        ):
            raise RuntimeError(
                f"training view {view} area pool differs from direct float64 2x2 sum"
            )
        full_uv, full_depth = camera.project(test_points)
        half_uv, half_depth = half_camera.project(test_points)
        projection_error = float((half_uv - full_uv / 2).abs().max())
        if not torch.allclose(half_uv, full_uv / 2, atol=1e-6, rtol=1e-6):
            raise RuntimeError(f"camera projection scaling differs in training view {view}")
        if not torch.equal(half_depth, full_depth):
            raise RuntimeError(f"camera projection depth differs in training view {view}")
        low_pixels = image.new_tensor([[0.5, 0.5], [12.0, 12.0], [23.5, 23.5]])
        half_origin, half_rays = half_camera.pixel_rays(low_pixels)
        full_origin, full_rays = camera.pixel_rays(2 * low_pixels)
        ray_error = float((half_rays - full_rays).abs().max())
        if not torch.allclose(half_origin, full_origin, atol=1e-6, rtol=1e-6):
            raise RuntimeError(f"camera ray origin differs in training view {view}")
        if not torch.allclose(half_rays, full_rays, atol=1e-6, rtol=1e-6):
            raise RuntimeError(f"camera ray scaling differs in training view {view}")
        if not torch.equal(half_camera.R, camera.R) or not torch.equal(half_camera.t, camera.t):
            raise RuntimeError(f"camera extrinsics differ in training view {view}")
        if not torch.equal(half_camera.position, camera.position):
            raise RuntimeError(f"camera position differs in training view {view}")
        maximum_projection_error = max(maximum_projection_error, projection_error)
        maximum_ray_error = max(maximum_ray_error, ray_error)
        maximum_direct_float64_pool_error = max(
            maximum_direct_float64_pool_error, direct_pool_error
        )
        per_view.append(
            {
                "view": view,
                "full_camera_sha256": camera_hash(camera),
                "half_camera_sha256": camera_hash(half_camera),
                "full_image_sha256": tensor_hash(image),
                "half_image_sha256": tensor_hash(half_image),
                "direct_float64_pool_sha256": tensor_hash(direct_float64),
                "pooled_float64_sha256": tensor_hash(pooled_float64),
                "maximum_direct_float64_pool_error": direct_pool_error,
                "direct_float64_pool_tolerance": {"atol": 1e-7, "rtol": 1e-6},
                "direct_float64_pool_parity": True,
                "maximum_projection_error": projection_error,
                "maximum_ray_error": ray_error,
            }
        )
    after = scene_hashes(scene)
    if after != before:
        raise RuntimeError("runtime camera/pyramid checks mutated points, bounds, or view order")
    return {
        "training_scene_sha256": before["aggregate_sha256"],
        "test_point_count": int(test_points.shape[0]),
        "test_points_sha256": tensor_hash(test_points),
        "maximum_projection_error": maximum_projection_error,
        "maximum_ray_error": maximum_ray_error,
        "maximum_direct_float64_pool_error": maximum_direct_float64_pool_error,
        "direct_float64_pool_tolerance": {"atol": 1e-7, "rtol": 1e-6},
        "direct_float64_pool_parity": True,
        "views": per_view,
    }


def prepare_seed(seed: int) -> tuple[SceneData, SceneData, Gaussians3D, dict[str, Any]]:
    full_scene = make_synthetic_scene(n_gaussians=40, n_cameras=12, image_size=48, seed=seed)
    if full_scene.masks is not None:
        raise RuntimeError("official synthetic scene unexpectedly has masks")
    training_scene = full_scene.subset(list(TRAIN_INDICES), name_suffix="multiscale-train")
    training_scene.gt_depths = None
    training_scene.gt_gaussians = None
    training_scene.validate()
    if training_scene.training_views != list(range(9)) or training_scene.testing_views:
        raise RuntimeError("physical training subset has an invalid local split")
    before_fit = scene_hashes(training_scene)
    config = fit_config()
    fit_wall_started = time.perf_counter()
    fit_process_started = time.process_time()
    fitted, fit_history = fit_views(
        training_scene.images,
        config,
        seed=seed,
        masks=training_scene.masks,
    )
    fit_timing = {
        "wall_seconds": time.perf_counter() - fit_wall_started,
        "process_seconds": time.process_time() - fit_process_started,
    }
    if len(fitted) != 9 or len(fit_history) != 9:
        raise RuntimeError("stage-1 fit did not return exactly nine views")
    assert_finite_tree(fit_history, "fit history")
    fitted_hash = gaussians2d_hashes(fitted)
    fit_history_hash = canonical_json_hash(fit_history)
    pre_lift_binding = {
        "training_scene_hashes": before_fit,
        "local_to_original_views": list(TRAIN_INDICES),
        "fit_config": asdict(config),
        "fitted_hashes": fitted_hash,
        "fit_history_sha256": fit_history_hash,
    }
    pre_lift_binding["aggregate_sha256"] = canonical_json_hash(pre_lift_binding)
    lifter = carve_lifter()
    lift_wall_started = time.perf_counter()
    lift_process_started = time.process_time()
    initialization = lifter.lift(fitted, training_scene)
    lift_timing = {
        "wall_seconds": time.perf_counter() - lift_wall_started,
        "process_seconds": time.process_time() - lift_process_started,
    }
    assert_finite_gaussians(initialization, f"seed {seed} Carve initialization")
    if gaussians2d_hashes(fitted) != fitted_hash:
        raise RuntimeError("Carve lift mutated the frozen fitted 2D Gaussians")
    if initialization.sh_degree != 0:
        raise RuntimeError("Carve initialization is not degree zero")
    initialization = initialization.detach()
    initialization_hash = gaussians_hashes(initialization)
    after_lift = scene_hashes(training_scene)
    if before_fit != after_lift:
        raise RuntimeError("fit or lift mutated the physical training scene")
    preparation = {
        "seed": seed,
        "training_scene_hashes": before_fit,
        "local_to_original_views": list(TRAIN_INDICES),
        "fit_config": asdict(config),
        "fit_timing": fit_timing,
        "fitted_hashes": fitted_hash,
        "fit_history": fit_history,
        "fit_history_sha256": fit_history_hash,
        "pre_lift_binding": pre_lift_binding,
        "carve_config": {
            "grid_res": lifter.grid_res,
            "bounds_scale": lifter.bounds_scale,
            "min_views": lifter.min_views,
            "hull_fraction": lifter.hull_fraction,
            "color_std_sigma": lifter.color_std_sigma,
            "color_match_sigma": lifter.color_match_sigma,
            "coverage_thresh": lifter.coverage_thresh,
            "samples_per_ray": lifter.samples_per_ray,
            "min_score": lifter.min_score,
            "min_weight": lifter.min_weight,
            "merge": lifter.merge,
            "merge_voxel_scale": lifter.merge_voxel_scale,
            "init_opacity": lifter.init_opacity,
            "sh_degree": lifter.sh_degree,
        },
        "lift_timing": lift_timing,
        "initialization_hashes": initialization_hash,
    }
    preparation["aggregate_sha256"] = canonical_json_hash(preparation)
    return full_scene, training_scene, initialization, preparation


def _history_without_elapsed(history: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in history.items() if key != "elapsed"}


def run_seed(
    seed: int,
    schedules: dict[str, Any],
) -> dict[str, Any]:
    if schedules != frozen_schedules():
        raise RuntimeError("arm schedules were not frozen before seed preparation")
    full_scene, training_scene, initialization, preparation = prepare_seed(seed)
    pyramid_invariants = verify_training_pyramid_invariants(training_scene)
    truths = construct_truth(full_scene)
    truth_records = [truth.record for truth in truths]
    truth_aggregate = canonical_json_hash(truth_records)
    expected_views = probe_view_schedule(seed)
    expected_view_hash = canonical_json_hash(expected_views)
    _, extent = full_scene.center_and_extent()
    del full_scene
    step_zero = evaluate_checkpoint(initialization.detach(), truths, extent=extent, step=0)
    initialization_hash = gaussians_hashes(initialization)
    arms = []
    for arm in ARM_ORDERS[seed]:
        controls = control_sequence(arm)
        schedule = schedules["arms"][arm]
        if schedule["sha256"] != canonical_json_hash(
            [[item.render_downscale, item.loss_downscale] for item in controls]
        ):
            raise RuntimeError(f"source-frozen schedule hash differs for {arm}")
        arm_initialization = initialization.detach()
        if gaussians_hashes(arm_initialization) != initialization_hash:
            raise RuntimeError(f"fresh initialization clone differs for seed {seed}/{arm}")
        callback_records: list[dict[str, Any]] = []

        def checkpoint_observer(
            snapshot: Gaussians3D,
            step: int,
            *,
            _records: list[dict[str, Any]] = callback_records,
            _arm: str = arm,
        ) -> None:
            before = gaussians_hashes(snapshot)
            _records.append(evaluate_checkpoint(snapshot, truths, extent=extent, step=step))
            if gaussians_hashes(snapshot) != before:
                raise RuntimeError(f"held-out observer mutated seed {seed}/{_arm} at {step}")

        config = train_config(seed)
        wall_started = time.perf_counter()
        process_started = time.process_time()
        final, history = Trainer(config).train(
            training_scene,
            arm_initialization,
            checkpoint_callback=checkpoint_observer,
            step_controls=trainer_step_controls(arm),
        )
        timing = {
            "wall_seconds": time.perf_counter() - wall_started,
            "process_seconds": time.process_time() - process_started,
            "native_elapsed": history["elapsed"],
            "pyramid_setup_seconds": (
                0.0
                if arm == "full"
                else float(history["step_control_metadata"]["pyramid_setup_seconds"])
            ),
            "loadavg_after": _loadavg(),
        }
        assert_finite_gaussians(final, f"seed {seed}/{arm} final")
        assert_finite_tree(history, f"seed {seed}/{arm} history")
        if final.sh_degree != 0 or final.n != initialization.n:
            raise RuntimeError(f"fixed topology or degree changed for seed {seed}/{arm}")
        if history["sampled_train_views"] != expected_views:
            raise RuntimeError(f"sampled-view schedule differs for seed {seed}/{arm}")
        if [int(item[0]) for item in history["psnr"]] != [30, 60, 90, 120]:
            raise RuntimeError(f"native checkpoint schedule differs for seed {seed}/{arm}")
        if history["n_gaussians"] != [(step, initialization.n) for step in (30, 60, 90, 120)]:
            raise RuntimeError(f"primitive-count schedule differs for seed {seed}/{arm}")
        if history["active_sh_degree"] != [(step, 0) for step in (30, 60, 90, 120)]:
            raise RuntimeError(f"SH-degree schedule differs for seed {seed}/{arm}")
        if [int(record["step"]) for record in callback_records] != [30, 60, 90, 120]:
            raise RuntimeError(f"held-out callback schedule differs for seed {seed}/{arm}")
        if arm == "full":
            if "step_control_metadata" in history:
                raise RuntimeError("full baseline unexpectedly constructed step-control metadata")
        else:
            metadata = history["step_control_metadata"]
            exposure = schedule["exposure"]
            if metadata["sequence_sha256"] != schedule["sha256"]:
                raise RuntimeError(f"Trainer schedule hash differs for seed {seed}/{arm}")
            if metadata["render_pixels"] != exposure["render_pixels"]:
                raise RuntimeError(f"Trainer render exposure differs for seed {seed}/{arm}")
            if metadata["loss_pixels"] != exposure["loss_pixels"]:
                raise RuntimeError(f"Trainer loss exposure differs for seed {seed}/{arm}")
            expected_counts = per_view_scale_counts(controls, expected_views)
            if metadata["per_view_scale_counts"] != expected_counts:
                raise RuntimeError(f"per-view scale counts differ for seed {seed}/{arm}")
        checkpoints = [step_zero, *callback_records]
        for checkpoint in checkpoints:
            validate_checkpoint_record(checkpoint)
        arm_record = {
            "seed": seed,
            "arm": arm,
            "execution_position": len(arms),
            "initialization_hashes": initialization_hash,
            "train_config": asdict(config),
            "schedule": schedule,
            "sampled_train_views": expected_views,
            "sampled_train_views_sha256": expected_view_hash,
            "per_view_scale_counts": per_view_scale_counts(controls, expected_views),
            "timing": timing,
            "history": history,
            "final_hashes": gaussians_hashes(final),
            "checkpoints": checkpoints,
            "foreground_psnr_auc_db": normalized_auc(checkpoints, "psnr_fg"),
        }
        arm_record["record_sha256"] = canonical_json_hash(arm_record)
        arms.append(arm_record)
    exposure_accounting = runtime_exposure_accounting(arms, step_zero)
    return {
        "seed": seed,
        "arm_order": list(ARM_ORDERS[seed]),
        "preparation": preparation,
        "pyramid_invariants": pyramid_invariants,
        "truth": {"views": truth_records, "aggregate_sha256": truth_aggregate},
        "view_schedule_probe": {
            "views": expected_views,
            "sha256": expected_view_hash,
        },
        "initialization_checkpoint": step_zero,
        "exposure_accounting": exposure_accounting,
        "arms": arms,
    }


def summarize_runs(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    if [int(run["seed"]) for run in seed_runs] != list(SEEDS):
        raise ValueError("summary seed order differs from the frozen seeds")
    by_seed: dict[str, Any] = {}
    for seed_run in seed_runs:
        seed = int(seed_run["seed"])
        arms = {run["arm"]: run for run in seed_run["arms"]}
        if set(arms) != set(ARMS):
            raise ValueError(f"seed {seed} does not contain the four frozen arms")
        by_seed[str(seed)] = {}
        for arm in ARMS:
            checkpoints = arms[arm]["checkpoints"]
            optimization_exposure = seed_run["exposure_accounting"]["by_arm"][arm]["optimization"]
            by_seed[str(seed)][arm] = {
                "final": dict(checkpoints[-1]["mean"]),
                "foreground_psnr_auc_db": normalized_auc(checkpoints, "psnr_fg"),
                "render_pixels": int(optimization_exposure["render_pixels"]),
                "loss_pixels": int(optimization_exposure["loss_pixels"]),
                "render_ratio": float(optimization_exposure["render_ratio"]),
            }
    return {"by_seed": by_seed}


def _candidate_decision(summary: dict[str, Any], arm: str) -> dict[str, Any]:
    by_seed = summary["by_seed"]

    def values(candidate: str, metric: str) -> list[float]:
        return [float(by_seed[str(seed)][candidate]["final"][metric]) for seed in SEEDS]

    baseline_psnr = values("full", "psnr_fg")
    candidate_psnr = values(arm, "psnr_fg")
    final_psnr_deltas = [
        candidate - baseline
        for candidate, baseline in zip(candidate_psnr, baseline_psnr, strict=True)
    ]
    baseline_ssim = values("full", "ssim_crop")
    candidate_ssim = values(arm, "ssim_crop")
    ssim_deltas = [
        candidate - baseline
        for candidate, baseline in zip(candidate_ssim, baseline_ssim, strict=True)
    ]
    baseline_depth = statistics.fmean(values("full", "depth_rmse_over_extent"))
    candidate_depth = statistics.fmean(values(arm, "depth_rmse_over_extent"))
    if not math.isfinite(baseline_depth) or baseline_depth <= 0.0:
        raise ValueError("baseline depth RMSE mean must be finite and positive")
    depth_regression = (candidate_depth - baseline_depth) / baseline_depth
    alpha_iou_deltas = [
        candidate - baseline
        for candidate, baseline in zip(
            values(arm, "alpha_iou"), values("full", "alpha_iou"), strict=True
        )
    ]
    coverage_deltas = [
        candidate - baseline
        for candidate, baseline in zip(
            values(arm, "foreground_coverage"),
            values("full", "foreground_coverage"),
            strict=True,
        )
    ]
    alpha_delta = statistics.fmean(alpha_iou_deltas)
    coverage_delta = statistics.fmean(coverage_deltas)
    auc_deltas = [
        float(by_seed[str(seed)][arm]["foreground_psnr_auc_db"])
        - float(by_seed[str(seed)]["full"]["foreground_psnr_auc_db"])
        for seed in SEEDS
    ]
    noninferiority_criteria = {
        "mean_final_psnr_delta_at_least_minus_0_05_db": statistics.fmean(final_psnr_deltas)
        >= -0.05,
        "each_final_psnr_delta_at_least_minus_0_15_db": min(final_psnr_deltas) >= -0.15,
        "mean_final_ssim_delta_at_least_minus_0_002": statistics.fmean(ssim_deltas) >= -0.002,
        "each_final_ssim_delta_at_least_minus_0_005": min(ssim_deltas) >= -0.005,
        "depth_rmse_relative_regression_at_most_0_02": depth_regression <= 0.02,
        "mean_final_alpha_iou_delta_at_least_minus_0_02": alpha_delta >= -0.02,
        "mean_final_coverage_delta_at_least_minus_0_02": coverage_delta >= -0.02,
    }
    quality_noninferior = all(noninferiority_criteria.values())
    improvement_criteria = {
        "quality_noninferior": quality_noninferior,
        "mean_auc_delta_at_least_0_10_db": statistics.fmean(auc_deltas) >= 0.10,
        "positive_auc_delta_at_least_two_seeds": sum(delta > 0.0 for delta in auc_deltas) >= 2,
        "mean_final_psnr_delta_nonnegative": statistics.fmean(final_psnr_deltas) >= 0.0,
        "nonnegative_final_psnr_delta_at_least_two_seeds": sum(
            delta >= 0.0 for delta in final_psnr_deltas
        )
        >= 2,
    }
    return {
        "arm": arm,
        "final_foreground_psnr_deltas_db": final_psnr_deltas,
        "mean_final_foreground_psnr_delta_db": statistics.fmean(final_psnr_deltas),
        "final_crop_ssim_deltas": ssim_deltas,
        "mean_final_crop_ssim_delta": statistics.fmean(ssim_deltas),
        "depth_rmse_relative_regression": depth_regression,
        "final_alpha_iou_deltas": alpha_iou_deltas,
        "mean_final_alpha_iou_delta": alpha_delta,
        "final_foreground_coverage_deltas": coverage_deltas,
        "mean_final_foreground_coverage_delta": coverage_delta,
        "foreground_psnr_auc_deltas_db": auc_deltas,
        "mean_foreground_psnr_auc_delta_db": statistics.fmean(auc_deltas),
        "auc_seed_wins": sum(delta > 0.0 for delta in auc_deltas),
        "final_psnr_nonnegative_seed_count": sum(delta >= 0.0 for delta in final_psnr_deltas),
        "quality_noninferiority_criteria": noninferiority_criteria,
        "quality_noninferior": quality_noninferior,
        "quality_improvement_criteria": improvement_criteria,
        "quality_improvement": all(improvement_criteria.values()),
    }


def frozen_decisions(summary: dict[str, Any]) -> dict[str, Any]:
    candidates = {arm: _candidate_decision(summary, arm) for arm in ARMS if arm != "full"}
    by_seed = summary["by_seed"]
    blocked_interleaved_auc = [
        float(by_seed[str(seed)]["camera_blocked"]["foreground_psnr_auc_db"])
        - float(by_seed[str(seed)]["camera_interleaved"]["foreground_psnr_auc_db"])
        for seed in SEEDS
    ]
    blocked_interleaved_final = [
        float(by_seed[str(seed)]["camera_blocked"]["final"]["psnr_fg"])
        - float(by_seed[str(seed)]["camera_interleaved"]["final"]["psnr_fg"])
        for seed in SEEDS
    ]
    blocked_order_criteria = {
        "mean_auc_delta_at_least_0_05_db": statistics.fmean(blocked_interleaved_auc) >= 0.05,
        "blocked_auc_wins_at_least_two_seeds": sum(delta > 0.0 for delta in blocked_interleaved_auc)
        >= 2,
        "mean_final_psnr_delta_at_least_minus_0_05_db": statistics.fmean(blocked_interleaved_final)
        >= -0.05,
        "each_final_psnr_delta_at_least_minus_0_15_db": min(blocked_interleaved_final) >= -0.15,
        "both_camera_arms_quality_noninferior": candidates["camera_blocked"]["quality_noninferior"]
        and candidates["camera_interleaved"]["quality_noninferior"],
    }
    for arm in ("camera_blocked", "camera_interleaved"):
        runtime_exposures = [by_seed[str(seed)][arm] for seed in SEEDS]
        exposure_criteria = {
            "each_render_pixel_count_is_172800": all(
                item["render_pixels"] == 172800 for item in runtime_exposures
            ),
            "each_loss_pixel_count_is_172800": all(
                item["loss_pixels"] == 172800 for item in runtime_exposures
            ),
            "each_render_ratio_is_0_625": all(
                item["render_ratio"] == 0.625 for item in runtime_exposures
            ),
            "quality_noninferior": candidates[arm]["quality_noninferior"],
        }
        candidates[arm]["exposure_efficiency_criteria"] = exposure_criteria
        candidates[arm]["exposure_efficiency"] = all(exposure_criteria.values())
    candidates["pyramid_blocked"]["exposure_efficiency"] = False
    camera_improves = candidates["camera_blocked"]["quality_improvement"]
    pyramid_improves = candidates["pyramid_blocked"]["quality_improvement"]
    if camera_improves and pyramid_improves:
        mechanism = "shared_low_frequency_curriculum_plausible"
    elif camera_improves:
        mechanism = "renderer_scale_or_forward_difference_only"
    elif pyramid_improves:
        mechanism = "low_frequency_supervision_only"
    else:
        mechanism = "no_quality_improvement"
    return {
        "candidates": candidates,
        "blocked_vs_interleaved": {
            "foreground_psnr_auc_deltas_db": blocked_interleaved_auc,
            "mean_foreground_psnr_auc_delta_db": statistics.fmean(blocked_interleaved_auc),
            "blocked_auc_seed_wins": sum(delta > 0.0 for delta in blocked_interleaved_auc),
            "final_foreground_psnr_deltas_db": blocked_interleaved_final,
            "criteria": blocked_order_criteria,
            "blocked_order_attribution": all(blocked_order_criteria.values()),
        },
        "mechanism_classification": mechanism,
        "timing_used_in_any_gate": False,
    }


def validate_seed_runs(seed_runs: list[dict[str, Any]], schedules: dict[str, Any]) -> None:
    if [int(item["seed"]) for item in seed_runs] != list(SEEDS):
        raise ValueError("result seeds differ from preregistration")
    for seed_run in seed_runs:
        seed = int(seed_run["seed"])
        if seed_run["arm_order"] != list(ARM_ORDERS[seed]):
            raise ValueError(f"arm order differs for seed {seed}")
        expected_views = probe_view_schedule(seed)
        if seed_run["view_schedule_probe"]["views"] != expected_views:
            raise ValueError(f"view schedule probe differs for seed {seed}")
        if seed_run["view_schedule_probe"]["sha256"] != canonical_json_hash(expected_views):
            raise ValueError(f"view schedule hash differs for seed {seed}")
        if [arm["arm"] for arm in seed_run["arms"]] != list(ARM_ORDERS[seed]):
            raise ValueError(f"arm records differ from execution order for seed {seed}")
        for arm_record in seed_run["arms"]:
            record_hash = arm_record.get("record_sha256")
            without_hash = {
                key: value for key, value in arm_record.items() if key != "record_sha256"
            }
            if record_hash != canonical_json_hash(without_hash):
                raise ValueError(f"arm record hash mismatch for seed {seed}/{arm_record['arm']}")
            arm = arm_record["arm"]
            if arm_record["schedule"] != schedules["arms"][arm]:
                raise ValueError(f"schedule record differs for seed {seed}/{arm}")
            if arm_record["sampled_train_views"] != expected_views:
                raise ValueError(f"sampled views differ for seed {seed}/{arm}")
            for checkpoint in arm_record["checkpoints"]:
                validate_checkpoint_record(checkpoint)
            try:
                validate_reported_auc(
                    arm_record["checkpoints"],
                    "psnr_fg",
                    arm_record["foreground_psnr_auc_db"],
                )
            except ValueError as error:
                raise ValueError(f"AUC differs for seed {seed}/{arm}") from error
        recomputed_exposure = runtime_exposure_accounting(
            seed_run["arms"], seed_run["initialization_checkpoint"]
        )
        if seed_run.get("exposure_accounting") != recomputed_exposure:
            raise ValueError(f"runtime exposure accounting differs for seed {seed}")


def validate_result_payload(payload: dict[str, Any]) -> None:
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("result artifact type is invalid")
    schedules = frozen_schedules()
    if payload.get("schedules") != schedules:
        raise ValueError("result schedules differ from frozen schedules")
    seed_runs = payload["seed_runs"]
    validate_seed_runs(seed_runs, schedules)
    summary = summarize_runs(seed_runs)
    if canonical_json_hash(summary) != canonical_json_hash(payload["summary"]):
        raise ValueError("result summary differs from arithmetic recomputation")
    validate_reported_decisions(summary, payload["decisions"])
    verify_attempt_marker(ATTEMPT, payload["attempt"])


def validate_reported_decisions(summary: dict[str, Any], decisions: dict[str, Any]) -> None:
    recomputed = frozen_decisions(summary)
    if canonical_json_hash(recomputed) != canonical_json_hash(decisions):
        raise ValueError("result decisions differ from frozen gate recomputation")


def verify_preregistration() -> dict[str, str]:
    path = ROOT / PREREGISTRATION
    digest = sha256_file(path)
    if digest != PREREGISTRATION_SHA256:
        raise RuntimeError(
            f"multiscale preregistration hash differs: {digest} != {PREREGISTRATION_SHA256}"
        )
    return {"path": str(PREREGISTRATION), "sha256": digest}


def verify_implementation_review() -> dict[str, str]:
    path = ROOT / IMPLEMENTATION_REVIEW
    if not path.is_file():
        raise FileNotFoundError(f"independent implementation review is missing: {path}")
    text = path.read_text(encoding="utf-8")
    normalized = text.lower().replace("`", "").replace("*", "")
    if "verdict: pass" not in normalized:
        raise RuntimeError("independent implementation review does not contain 'Verdict: pass'")
    if "unresolved findings" not in normalized:
        raise RuntimeError("implementation review must explicitly address unresolved findings")
    return {"path": str(IMPLEMENTATION_REVIEW), "sha256": sha256_file(path)}


def verify_default_seam() -> dict[str, Any]:
    parameters = inspect.signature(Trainer.train).parameters
    if parameters["step_controls"].default is not None:
        raise RuntimeError("Trainer.train step_controls default is not None")
    if not TrainStepControl.__dataclass_params__.frozen:
        raise RuntimeError("TrainStepControl is not immutable")
    default = TrainStepControl()
    if default != TrainStepControl(1, 1):
        raise RuntimeError("TrainStepControl default is not full resolution")
    if train_config(3).densify or train_config(3).target_sh_degree != 0:
        raise RuntimeError("frozen refinement configuration is not fixed-topology degree zero")
    return {
        "step_controls_default_is_none": True,
        "train_step_control_is_frozen": True,
        "train_step_control_default": asdict(default),
        "frozen_train_config_seed3": asdict(train_config(3)),
        "schedules": frozen_schedules(),
    }


def verification_commands() -> tuple[tuple[str, ...], ...]:
    return (
        (".venv/bin/python", "-m", "ruff", "check", "."),
        (".venv/bin/python", "-m", "ruff", "format", "--check", "."),
        (".venv/bin/python", "-m", "pytest", "-q", "-m", "not slow"),
        (".venv/bin/python", "scripts/docs_sync.py"),
        ("git", "diff", "--check"),
    )


def run_verification() -> dict[str, Any]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    results = []
    for command in verification_commands():
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
        result = {
            "command": list(command),
            "returncode": completed.returncode,
            "seconds": time.perf_counter() - started,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "stdout_sha256": sha256_bytes(completed.stdout.encode()),
            "stderr_sha256": sha256_bytes(completed.stderr.encode()),
        }
        results.append(result)
        if completed.returncode != 0:
            raise RuntimeError(
                f"verification failed: {' '.join(command)}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
    return {"passed": True, "commands": results}


def capture_seal_snapshot() -> dict[str, Any]:
    """Capture every source-derived value that verification is intended to certify."""
    hashes, aggregate = source_hashes(SEALED_PATHS, root=ROOT)
    return {
        "sealed_paths": [str(path) for path in SEALED_PATHS],
        "source_hashes": hashes,
        "source_aggregate": aggregate,
        "preregistration": verify_preregistration(),
        "implementation_review": verify_implementation_review(),
        "default_seam": verify_default_seam(),
    }


def require_unchanged_seal_snapshot(before: dict[str, Any], after: dict[str, Any]) -> None:
    for field in (
        "sealed_paths",
        "source_hashes",
        "source_aggregate",
        "preregistration",
        "implementation_review",
        "default_seam",
    ):
        if before.get(field) != after.get(field):
            raise RuntimeError(f"seal snapshot drifted during verification: {field}")


def create_seal() -> dict[str, Any]:
    environment = environment_metadata()
    assert_official_environment(environment)
    before = capture_seal_snapshot()
    verification = run_verification()
    after = capture_seal_snapshot()
    require_unchanged_seal_snapshot(before, after)
    snapshot_sha256 = canonical_json_hash(before)
    return {
        "artifact_type": SEAL_ARTIFACT_TYPE,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_metadata(),
        "environment": environment,
        "preregistration": before["preregistration"],
        "implementation_review": before["implementation_review"],
        "default_seam": before["default_seam"],
        "verification": verification,
        "verification_snapshot": {
            "pre_verification_sha256": snapshot_sha256,
            "post_verification_sha256": canonical_json_hash(after),
            "unchanged": True,
        },
        "sealed_paths": before["sealed_paths"],
        "source_hashes": before["source_hashes"],
        "source_aggregate": before["source_aggregate"],
        "command": [sys.executable, *sys.argv],
    }


def load_and_verify_seal(path: Path) -> dict[str, Any]:
    if path.resolve() != DEFAULT_SEAL.resolve():
        raise ValueError("official run must use the fixed preregistered seal path")
    payload = strict_json_load(path)
    if payload.get("artifact_type") != SEAL_ARTIFACT_TYPE:
        raise ValueError(f"{path} is not a multiscale implementation seal")
    current_snapshot = capture_seal_snapshot()
    for field in (
        "sealed_paths",
        "source_hashes",
        "source_aggregate",
        "preregistration",
        "implementation_review",
        "default_seam",
    ):
        if payload.get(field) != current_snapshot[field]:
            raise RuntimeError(f"seal {field.replace('_', '-')} binding differs")
    snapshot_sha256 = canonical_json_hash(current_snapshot)
    expected_verification_snapshot = {
        "pre_verification_sha256": snapshot_sha256,
        "post_verification_sha256": snapshot_sha256,
        "unchanged": True,
    }
    if payload.get("verification_snapshot") != expected_verification_snapshot:
        raise RuntimeError("seal verification snapshot binding differs")
    paths = tuple(Path(item) for item in payload["sealed_paths"])
    verify_source_manifest(
        paths,
        payload.get("source_hashes", {}),
        payload.get("source_aggregate", ""),
        root=ROOT,
    )
    _, aggregate = source_hashes(paths, root=ROOT)
    verification = payload.get("verification", {})
    if not verification.get("passed"):
        raise RuntimeError("implementation seal lacks passing full verification")
    expected_commands = [list(command) for command in verification_commands()]
    recorded = verification.get("commands", [])
    if [item.get("command") for item in recorded] != expected_commands or any(
        item.get("returncode") != 0 for item in recorded
    ):
        raise RuntimeError("seal verification command sequence or status differs")
    for item in recorded:
        if item.get("stdout_sha256") != sha256_bytes(item.get("stdout", "").encode()):
            raise RuntimeError("seal verification stdout hash differs")
        if item.get("stderr_sha256") != sha256_bytes(item.get("stderr", "").encode()):
            raise RuntimeError("seal verification stderr hash differs")
    current_environment = environment_metadata()
    assert_official_environment(current_environment)
    if environment_fingerprint(payload["environment"]) != environment_fingerprint(
        current_environment
    ):
        raise RuntimeError("runtime environment differs from the implementation seal")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "source_aggregate": aggregate,
        "verification_sha256": canonical_json_hash(verification),
        "environment_fingerprint": environment_fingerprint(payload["environment"]),
        "implementation_review": payload["implementation_review"],
    }


def verify_loaded_sources_against_seal(seal_path: Path) -> tuple[dict[str, str], str]:
    seal = strict_json_load(seal_path)
    sealed_hashes = seal["source_hashes"]
    loaded_hashes, aggregate = loaded_source_hashes()
    unexpected = sorted(set(loaded_hashes) - set(sealed_hashes))
    mismatched = sorted(
        path
        for path, digest in loaded_hashes.items()
        if path in sealed_hashes and sealed_hashes[path] != digest
    )
    if unexpected or mismatched:
        raise RuntimeError(
            "loaded repository sources differ from seal: "
            f"unexpected={unexpected}, mismatched={mismatched}"
        )
    return loaded_hashes, aggregate


def companion_note_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_RESULT.md")


def preflight_output(output: Path) -> None:
    note = companion_note_path(output)
    if output.exists() or note.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {note}")
    output.parent.mkdir(parents=True, exist_ok=True)


def attempt_marker_binding(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = strict_json_load(path)
    if not isinstance(payload, dict) or payload.get("artifact_type") != ATTEMPT_ARTIFACT_TYPE:
        raise RuntimeError(f"{path} is not the frozen multiscale attempt marker")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_bytes(raw),
        "payload_sha256": canonical_json_hash(payload),
        "payload": payload,
    }


def verify_attempt_marker(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if Path(expected.get("path", "")).resolve() != path.resolve():
        raise RuntimeError("attempt marker path binding differs")
    actual = attempt_marker_binding(path)
    if actual != expected:
        raise RuntimeError("attempt marker payload or digest changed during the official run")
    return actual


def claim_attempt(path: Path, *, output: Path, seal: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_type": ATTEMPT_ARTIFACT_TYPE,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output": str(output),
        "seal_sha256": seal["sha256"],
        "source_aggregate": seal["source_aggregate"],
        "command": [sys.executable, *sys.argv],
        "environment": environment_metadata(),
    }
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    except FileExistsError as error:
        raise RuntimeError(
            f"the once-only multiscale attempt is already claimed by {path}"
        ) from error
    binding = attempt_marker_binding(path)
    if binding["sha256"] != sha256_bytes(rendered.encode()) or binding["payload"] != payload:
        raise RuntimeError("attempt marker changed during exclusive creation")
    return binding


def run_experiment(seal_path: Path, output: Path) -> dict[str, Any]:
    if seal_path.resolve() != DEFAULT_SEAL.resolve():
        raise ValueError("official run must use the fixed preregistered seal path")
    preflight_output(output)
    seal = load_and_verify_seal(seal_path)
    schedules = frozen_schedules()
    if ATTEMPT.exists():
        raise RuntimeError(f"the once-only multiscale attempt already exists: {ATTEMPT}")
    marker = claim_attempt(ATTEMPT, output=output, seal=seal)
    experiment_wall_started = time.perf_counter()
    experiment_process_started = time.process_time()
    seed_runs = []
    for seed in SEEDS:
        print(f"multiscale: preparing and running seed {seed}", flush=True)
        seed_runs.append(run_seed(seed, schedules))
    verify_attempt_marker(ATTEMPT, marker)
    summary = summarize_runs(seed_runs)
    decisions = frozen_decisions(summary)
    loaded_hashes, loaded_aggregate = verify_loaded_sources_against_seal(seal_path)
    if load_and_verify_seal(seal_path) != seal:
        raise RuntimeError("implementation seal changed during the official run")
    verify_attempt_marker(ATTEMPT, marker)
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv],
        "git": git_metadata(),
        "environment": environment_metadata(),
        "preregistration": verify_preregistration(),
        "seal": seal,
        "attempt": marker,
        "loaded_source_hashes": loaded_hashes,
        "loaded_source_aggregate": loaded_aggregate,
        "split": {"train": list(TRAIN_INDICES), "held_out": list(TEST_INDICES)},
        "schedules": schedules,
        "seed_runs": seed_runs,
        "summary": summary,
        "decisions": decisions,
        "timing": {
            "wall_seconds": time.perf_counter() - experiment_wall_started,
            "process_seconds": time.process_time() - experiment_process_started,
            "loadavg_final": _loadavg(),
            "descriptive_only": True,
        },
        "limitations": {
            "cpu_synthetic_fixed_topology_degree_zero_only": True,
            "no_wall_clock_speed_claim": True,
            "no_real_scene_cuda_density_or_full_sh_claim": True,
            "no_default_change": True,
            "requires_independent_results_audit": True,
        },
    }
    validate_result_payload(payload)
    assert_finite_tree(payload, "official result")
    verify_attempt_marker(ATTEMPT, marker)
    return payload


def result_note(payload: dict[str, Any], output: Path, digest: str) -> str:
    lines = [
        f"# {payload['artifact_type']}",
        "",
        f"- Timestamp (UTC): `{payload['timestamp_utc']}`",
        f"- JSON artifact: `{output}`",
        f"- JSON SHA-256: `{digest}`",
        f"- Command: `{' '.join(payload['command'])}`",
    ]
    if payload["artifact_type"] == ARTIFACT_TYPE:
        decisions = payload["decisions"]
        lines.extend(
            [
                "",
                "## Frozen decisions",
                "",
                f"- Camera blocked quality improvement: "
                f"`{decisions['candidates']['camera_blocked']['quality_improvement']}`",
                f"- Camera blocked exposure efficiency: "
                f"`{decisions['candidates']['camera_blocked']['exposure_efficiency']}`",
                f"- Blocked-order attribution: "
                f"`{decisions['blocked_vs_interleaved']['blocked_order_attribution']}`",
                f"- Mechanism classification: `{decisions['mechanism_classification']}`",
                "",
                "These are unaudited frozen-gate outputs. No quantitative or capability claim, "
                "documentation update, default change, or follow-up selection is permitted until "
                "the independent results audit passes.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "The atomic attempt was consumed by an invalid run. This artifact contains no "
                "success classification and cannot be resumed.",
            ]
        )
    return "\n".join(lines) + "\n"


def write_artifact(output: Path, payload: dict[str, Any]) -> tuple[Path, str]:
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    digest = sha256_bytes(rendered.encode())
    note = companion_note_path(output)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
    serialized_payload = strict_json_load(output)
    if canonical_json_hash(serialized_payload) != canonical_json_hash(payload):
        raise RuntimeError("serialized artifact differs from the in-memory payload")
    with note.open("x", encoding="utf-8") as handle:
        handle.write(result_note(serialized_payload, output, digest))
    return note, digest


def invalid_attempt_payload(error: Exception, *, output: Path, seal_path: Path) -> dict[str, Any]:
    return {
        "artifact_type": INVALID_ARTIFACT_TYPE,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv],
        "output": str(output),
        "seal_path": str(seal_path),
        "attempt": attempt_marker_binding(ATTEMPT),
        "error": {"type": type(error).__name__, "message": str(error)},
        "environment": environment_metadata(),
        "success_classification": None,
        "resume_permitted": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    seal = subparsers.add_parser("seal", help="verify and freeze the implementation")
    seal.add_argument("--output", type=Path, default=DEFAULT_SEAL)
    run = subparsers.add_parser("run", help="consume the once-only official attempt")
    run.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    run.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if args.command_name == "seal":
        output = args.output.resolve()
        if output != DEFAULT_SEAL.resolve():
            raise ValueError("official seal must use the fixed preregistered path")
        preflight_output(output)
        payload = create_seal()
        note, digest = write_artifact(output, payload)
    elif args.command_name == "run":
        output = args.output.resolve()
        expected = re.fullmatch(r"\d{8}T\d{6}Z_cpu_multiscale_refinement\.json", output.name)
        if output.parent != (ROOT / "benchmarks/results").resolve() or expected is None:
            raise ValueError("official output must be <UTC>_cpu_multiscale_refinement.json")
        marker_was_absent = not ATTEMPT.exists()
        try:
            payload = run_experiment(args.seal.resolve(), output)
        except Exception as error:
            if marker_was_absent and ATTEMPT.exists() and not output.exists():
                payload = invalid_attempt_payload(
                    error, output=output, seal_path=args.seal.resolve()
                )
                note, digest = write_artifact(output, payload)
                print(f"saved fail-closed {output} (sha256={digest})", flush=True)
                print(f"saved {note}", flush=True)
            raise
        verify_attempt_marker(ATTEMPT, payload["attempt"])
        note, digest = write_artifact(output, payload)
    else:  # pragma: no cover
        raise AssertionError(args.command_name)
    print(f"saved {output} (sha256={digest})", flush=True)
    print(f"saved {note}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
