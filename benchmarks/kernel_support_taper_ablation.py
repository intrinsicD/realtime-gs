#!/usr/bin/env python3
"""Preregistered hard kernel-support audit and gated C1 taper ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.metrics import masked_crop, masked_psnr, psnr, ssim
from rtgs.core.sh import C1, eval_sh_preactivation
from rtgs.data.scene import SceneData
from rtgs.data.synthetic import make_synthetic_scene
from rtgs.image2gs.fit import FitConfig, fit_views
from rtgs.lift.depth import DepthLifter
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.torch_ref import (
    KERNEL_SUPPORT_CUTOFF,
    KERNEL_SUPPORT_TAPER_WIDTH,
    TorchRasterizer,
    kernel_support_weight,
)

ROOT = Path(__file__).resolve().parent.parent
INCORPORATED_PREREGISTRATION = Path("benchmarks/results/20260715_sh_activation_PREREG.md")
INCORPORATED_PREREGISTRATION_SHA256 = (
    "5353c4aa37c13e280f0bf3761679424e0bb5e17b4e942a7ff36275e84be88c1f"
)
PREVIOUS_SUPPORT_PREREGISTRATION = Path(
    "benchmarks/results/20260715_kernel_support_taper_PREREG.md"
)
PREREGISTRATION = Path("benchmarks/results/20260715_kernel_support_taper_iter2_PREREG.md")
DEFAULT_SEAL = Path("benchmarks/results/20260715_kernel_support_taper_iter2_SEAL.json")
PHASE_A_ATTEMPT = (
    ROOT / "benchmarks/results/20260715_kernel_support_taper_iter2_PHASE_A_ATTEMPT.json"
)
PHASE_B_ATTEMPT = (
    ROOT / "benchmarks/results/20260715_kernel_support_taper_iter2_PHASE_B_ATTEMPT.json"
)
HISTORICAL_ARTIFACT_HASHES = {
    Path(
        "benchmarks/results/20260715_kernel_support_taper_SEAL.json"
    ): "f35827d362318d4eb55d637cdadb77c5a97deb68fd62e6f04e231e9c39702184",
    Path(
        "benchmarks/results/20260715T200803Z_cpu_kernel_support_taper_audit.json"
    ): "6380dc0b92043db608f6ba056c1cbaa2509e4eeba62a7b20e3f0fb7eacdde59c",
    Path(
        "benchmarks/results/20260715T200803Z_cpu_kernel_support_taper_audit_AUDIT.json"
    ): "3f8f404912d0ef1e30e605ef6f7c194d3ec07e38f517ca1493166ff03a182919",
    Path(
        "benchmarks/results/20260715_kernel_support_taper_PHASE_B_ATTEMPT.json"
    ): "0c3b1e96ab56680db64758c9e2ceb17a5c53bb5f950bfd416d1165b08433e3c1",
}
FAILED_PHASE_B_OUTPUT = Path(
    "benchmarks/results/20260715T201746Z_cpu_kernel_support_taper_ablation.json"
)
CONDITIONS = ("diffuse", "view_dependent")
CANDIDATE_ARMS = ("c1_taper", "hard_forward_c1_taper_gradient")
TRAIN_INDICES = [0, 1, 2, 4, 5, 6, 8, 9, 10]
TEST_INDICES = [3, 7, 11]
SEEDS = [0, 1, 2]

SEALED_PATHS = tuple(
    sorted(
        {
            PREREGISTRATION,
            PREVIOUS_SUPPORT_PREREGISTRATION,
            INCORPORATED_PREREGISTRATION,
            Path("benchmarks/kernel_support_taper_ablation.py"),
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


def canonical_json_hash(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def tensor_collection_hash(items: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for label, tensor in items:
        value = tensor.detach().cpu().contiguous()
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def gaussians_hash(gaussians: Gaussians3D) -> str:
    return tensor_collection_hash(
        [
            (field, getattr(gaussians, field))
            for field in ("means", "quats", "log_scales", "opacity", "sh")
        ]
    )


def fitted_hash(gaussians2d) -> str:
    values = []
    for view, gaussians in enumerate(gaussians2d):
        for field in ("xy", "chol", "color", "weight"):
            values.append((f"view{view}/{field}", getattr(gaussians, field)))
    return tensor_collection_hash(values)


def scene_hashes(scene: SceneData) -> dict[str, str]:
    camera_values = []
    for index, camera in enumerate(scene.cameras):
        camera_values.extend(
            [
                (f"camera{index}/R", camera.R),
                (f"camera{index}/t", camera.t),
                (f"camera{index}/K", camera.K),
                (
                    f"camera{index}/size",
                    torch.tensor([camera.width, camera.height], dtype=torch.int64),
                ),
            ]
        )
    values = {
        "images": tensor_collection_hash(
            [(f"image{index}", image) for index, image in enumerate(scene.images)]
        ),
        "depths": tensor_collection_hash(
            [(f"depth{index}", depth) for index, depth in enumerate(scene.gt_depths or [])]
        ),
        "cameras": tensor_collection_hash(camera_values),
        "gt_gaussians": gaussians_hash(scene.gt_gaussians),
        "points": tensor_collection_hash([("points", scene.points)]),
    }
    values["aggregate"] = canonical_json_hash(values)
    return values


def git_metadata() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    return {
        "revision": revision,
        "dirty": bool(status.strip()),
        "status": status.splitlines(),
        "tracked_diff_sha256": sha256_bytes(tracked_diff),
    }


def source_hashes(paths: tuple[Path, ...] = SEALED_PATHS) -> tuple[dict[str, str], str]:
    missing = [str(path) for path in paths if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"sealed source files are missing: {missing}")
    hashes = {str(path): sha256_file(ROOT / path) for path in paths}
    return hashes, canonical_json_hash(hashes)


def loaded_source_hashes() -> tuple[dict[str, str], str]:
    paths = set()
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if source is None:
            continue
        path = Path(source).resolve()
        if path.suffix != ".py" or not path.is_relative_to(ROOT) or not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        # Treat every loaded repo-owned Python file as outcome-relevant, while excluding
        # packages installed into the repository-local virtual environment.
        if relative.parts and relative.parts[0] != ".venv":
            paths.add(path)
    for relative in (
        INCORPORATED_PREREGISTRATION,
        PREVIOUS_SUPPORT_PREREGISTRATION,
        PREREGISTRATION,
        Path("pyproject.toml"),
    ):
        paths.add((ROOT / relative).resolve())
    hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(paths)}
    return hashes, canonical_json_hash(hashes)


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


def train_config(seed: int, mode: str, *, diagnostics: bool) -> TrainConfig:
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
        collect_sh_color_diagnostics=False,
        kernel_support_mode=mode,
        collect_kernel_support_diagnostics=diagnostics,
        validate_render_finite=True,
        seed=seed,
    )


def make_condition_scene(seed: int, condition: str) -> tuple[SceneData, dict[str, float]]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    base = make_synthetic_scene(n_gaussians=40, n_cameras=12, image_size=48, seed=seed)
    target_range = {"minimum": 0.0, "maximum": 1.0}
    if condition == "view_dependent":
        gt = base.gt_gaussians.with_sh_degree(1)
        gaussian_ids = torch.arange(gt.n)[:, None]
        channels = torch.arange(3)[None, :]
        signs = torch.where((gaussian_ids + channels) % 2 == 0, 1.0, -1.0)
        gt.sh[:, 3] = -0.12 * signs / C1
        values = []
        for camera in base.cameras:
            directions = torch.nn.functional.normalize(
                gt.means - camera.position.to(gt.means), dim=-1
            )
            values.append(eval_sh_preactivation(1, gt.sh, directions))
        target_values = torch.cat(values)
        target_range = {
            "minimum": float(target_values.min()),
            "maximum": float(target_values.max()),
        }
        if not bool(torch.isfinite(target_values).all()) or not bool(
            ((target_values >= 0.03) & (target_values <= 0.97)).all()
        ):
            raise AssertionError(f"view-dependent GT violates frozen raw range: {target_range}")

        renderer = TorchRasterizer()
        images, depths = [], []
        with torch.no_grad():
            for camera in base.cameras:
                output = renderer.render(gt, camera)
                images.append(output.color.clamp(0.0, 1.0))
                depths.append(
                    torch.where(
                        output.alpha > 0.05,
                        output.depth / output.alpha.clamp_min(1e-6),
                        0.0,
                    )
                )
        base = SceneData(
            images=images,
            cameras=base.cameras,
            points=base.points,
            gt_depths=depths,
            gt_gaussians=gt,
            name=f"{base.name}-view-dependent",
        )
    base.train_indices = list(TRAIN_INDICES)
    base.test_indices = list(TEST_INDICES)
    base.validate()
    return base, target_range


def prepare_seed(seed: int, condition: str) -> tuple[SceneData, Any, Gaussians3D, dict[str, Any]]:
    torch.manual_seed(seed)
    scene, target_range = make_condition_scene(seed, condition)
    train_scene = scene.subset(TRAIN_INDICES)
    started = time.perf_counter()
    gaussians2d, fit_history = fit_views(train_scene.images, fit_config(), seed=seed, masks=None)
    fit_seconds = time.perf_counter() - started
    lifter = DepthLifter(
        backend=None,
        sh_degree=0,
        min_weight=0.05,
        init_opacity=0.1,
        normal_thickness=0.15,
        covariance_mode="surface",
        isotropic_sigma=None,
        robust_depth_gradients=True,
        merge=True,
        merge_voxel_frac=0.01,
    )
    initialization = lifter.lift(gaussians2d, train_scene)
    assert_finite_gaussians(initialization, f"initialization {condition}/{seed}")
    metadata = {
        "target_preactivation_range": target_range,
        "scene_hashes": scene_hashes(scene),
        "fitted_hash": fitted_hash(gaussians2d),
        "initialization_hash": gaussians_hash(initialization),
        "initial_gaussians": initialization.n,
        "fit_seconds": fit_seconds,
        "fit_psnr": [float(item["final_psnr"]) for item in fit_history],
        "fit_config": asdict(fit_config()),
        "train_indices": list(scene.training_views),
        "test_indices": list(scene.testing_views),
    }
    assert_finite_tree(metadata, f"preparation {condition}/{seed}")
    return scene, gaussians2d, initialization, metadata


def evaluate_final(
    scene: SceneData,
    gaussians: Gaussians3D,
    *,
    mode: str = "hard",
) -> dict[str, Any]:
    renderer = TorchRasterizer(kernel_support_mode=mode)
    truth_renderer = TorchRasterizer()
    _, extent = scene.center_and_extent()
    per_view = []
    with torch.no_grad():
        for index in TEST_INDICES:
            predicted = renderer.render(gaussians, scene.cameras[index])
            truth = truth_renderer.render(scene.gt_gaussians, scene.cameras[index])
            for label, output in (("predicted", predicted), ("truth", truth)):
                for field in ("color", "alpha", "depth"):
                    if not bool(torch.isfinite(getattr(output, field)).all()):
                        raise AssertionError(
                            f"held-out {label} {field} is non-finite in view {index}"
                        )
            target = scene.images[index]
            predicted_color = predicted.color.clamp(0.0, 1.0)
            target_color = target.clamp(0.0, 1.0)
            truth_support = truth.alpha > 0.05
            predicted_support = predicted.alpha > 0.05
            intersection = truth_support & predicted_support
            union = truth_support | predicted_support
            if not bool(truth_support.any()):
                raise AssertionError(f"held-out view {index} has empty GT support")
            if not bool(intersection.any()):
                raise AssertionError(f"held-out view {index} has empty depth intersection")
            predicted_crop = masked_crop(predicted_color, truth_support.float())
            target_crop = masked_crop(target_color, truth_support.float())
            predicted_depth = predicted.depth / predicted.alpha.clamp_min(1e-6)
            truth_depth = truth.depth / truth.alpha.clamp_min(1e-6)
            values = {
                "view": index,
                "psnr_fg": masked_psnr(predicted_color, target_color, truth_support.float()),
                "psnr_full": psnr(predicted_color, target_color),
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
                "foreground_coverage": float(intersection.sum() / truth_support.sum().clamp_min(1)),
            }
            if not all(math.isfinite(value) for key, value in values.items() if key != "view"):
                raise AssertionError(f"non-finite held-out metric: {values}")
            per_view.append(values)
    metric_names = [key for key in per_view[0] if key != "view"]
    mean = {key: statistics.fmean(float(view[key]) for view in per_view) for key in metric_names}
    return {"per_view": per_view, "mean": mean, "kernel_support_mode": mode}


def step0_invariants(initialization: Gaussians3D, scene: SceneData) -> dict[str, float]:
    gaussians = initialization.with_sh_degree(3)
    hard_renderer = TorchRasterizer()
    smooth_renderer = TorchRasterizer(kernel_support_mode="c1_taper")
    control_renderer = TorchRasterizer(kernel_support_mode="hard_forward_c1_taper_gradient")
    q = torch.tensor(
        [0.0, 8.0, 11.999, 12.0, 13.0, 15.999, 16.0, 20.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    hard_weight = kernel_support_weight(q, "hard")
    smooth_weight = kernel_support_weight(q, "c1_taper")
    control_weight = kernel_support_weight(q, "hard_forward_c1_taper_gradient")
    if not torch.equal(control_weight, hard_weight):
        raise AssertionError("hard-forward taper control differs from hard")
    if not torch.equal(smooth_weight[q < 12.0], hard_weight[q < 12.0]):
        raise AssertionError("C1 taper changes the established q<12 kernel")
    if not torch.equal(smooth_weight[q >= 16.0], torch.zeros_like(smooth_weight[q >= 16.0])):
        raise AssertionError("C1 taper is not compact at q>=16")
    max_kernel_difference = float((smooth_weight - hard_weight).abs().max())
    if max_kernel_difference > math.exp(-6.0) + 1e-7:
        raise AssertionError("C1 taper exceeds the frozen maximum kernel deviation")
    smooth_gradient = torch.autograd.grad(smooth_weight.sum(), q, retain_graph=True)[0]
    control_gradient = torch.autograd.grad(control_weight.sum(), q)[0]
    gradient_max_error = float((smooth_gradient - control_gradient).abs().max())
    if gradient_max_error != 0.0:
        raise AssertionError("hard-forward control gradient differs from C1 taper")

    maximum_rendered_color_difference = 0.0
    maximum_rendered_alpha_difference = 0.0
    maximum_rendered_depth_difference = 0.0
    for index in TRAIN_INDICES:
        camera = scene.cameras[index]
        hard = hard_renderer.render(gaussians, camera)
        smooth = smooth_renderer.render(gaussians, camera)
        control = control_renderer.render(gaussians, camera)
        for label, output in (("hard", hard), ("smooth", smooth), ("control", control)):
            for field in ("color", "alpha", "depth"):
                if not bool(torch.isfinite(getattr(output, field)).all()):
                    raise AssertionError(f"step-zero {label} {field} is non-finite in view {index}")
        for field in ("color", "alpha", "depth"):
            if not torch.equal(getattr(control, field), getattr(hard, field)):
                raise AssertionError(
                    f"hard-forward control {field} differs from hard in view {index}"
                )
        maximum_rendered_color_difference = max(
            maximum_rendered_color_difference, float((smooth.color - hard.color).abs().max())
        )
        maximum_rendered_alpha_difference = max(
            maximum_rendered_alpha_difference, float((smooth.alpha - hard.alpha).abs().max())
        )
        maximum_rendered_depth_difference = max(
            maximum_rendered_depth_difference, float((smooth.depth - hard.depth).abs().max())
        )
    return {
        "maximum_kernel_difference": max_kernel_difference,
        "control_taper_gradient_max_error": gradient_max_error,
        "maximum_rendered_color_difference": maximum_rendered_color_difference,
        "maximum_rendered_alpha_difference": maximum_rendered_alpha_difference,
        "maximum_rendered_depth_difference": maximum_rendered_depth_difference,
    }


def _sum_field(records: list[dict[str, Any]], field: str) -> float:
    return math.fsum(float(record[field]) for record in records)


def aggregate_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    populated = [record for record in records if int(record["observation_count"]) > 0]
    if not populated:
        raise AssertionError("hard-support audit has no populated diagnostics")
    zero_invariants = (
        "q_below_tolerance_count",
        "hard_outside_kernel_nonzero_count",
        "hard_outside_qgrad_nonzero_count",
        "hard_active_qgrad_violation_count",
    )
    for field in zero_invariants:
        if any(int(record[field]) != 0 for record in populated):
            raise AssertionError(f"hard-support validity invariant failed: {field}")
    int_fields = (
        "observation_count",
        "eligible_count",
        "interior_count",
        "boundary_count",
        "annulus_count",
        "negative_q_count",
    )
    float_fields = (
        "local_upstream_l1",
        "local_upstream_sum",
        "annulus_upstream_l1",
        "annulus_upstream_sum",
        "recoverable_annulus_upstream_l1",
        "active_hard_qgrad_l1",
        "boundary_hard_qgrad_l1",
        "annulus_candidate_qgrad_l1",
        "recoverable_annulus_candidate_qgrad_l1",
        "local_kernel_loss_contribution_sum",
        "local_kernel_loss_contribution_l1",
    )
    summary = {field: sum(int(record[field]) for record in populated) for field in int_fields}
    summary.update({field: _sum_field(populated, field) for field in float_fields})
    required_positive = (
        "eligible_count",
        "local_upstream_l1",
        "annulus_upstream_l1",
        "active_hard_qgrad_l1",
        "boundary_hard_qgrad_l1",
    )
    if any(
        not math.isfinite(float(summary[field])) or float(summary[field]) <= 0.0
        for field in required_positive
    ):
        raise AssertionError("hard-support pooled denominator is zero or non-finite")
    q_bins = []
    for bin_index in range(len(populated[0]["q_bins"])):
        members = [record["q_bins"][bin_index] for record in populated]
        if any(
            member["low"] != members[0]["low"] or member["high"] != members[0]["high"]
            for member in members
        ):
            raise AssertionError("q-bin boundaries differ across records")
        q_bins.append(
            {
                "low": members[0]["low"],
                "high": members[0]["high"],
                "count": sum(int(member["count"]) for member in members),
                **{
                    field: _sum_field(members, field)
                    for field in (
                        "upstream_l1",
                        "upstream_sum",
                        "recoverable_upstream_l1",
                        "hard_qgrad_l1",
                        "candidate_qgrad_l1",
                        "recoverable_candidate_qgrad_l1",
                        "kernel_loss_contribution_sum",
                        "kernel_loss_contribution_l1",
                    )
                },
            }
        )
    sampled_views = sorted({int(record["view"]) for record in populated})
    summary.update(
        {
            "eligible_steps": len(populated),
            "steps_with_invalid_local_denominator": sum(
                not bool(record["ratio_denominators_valid"]) for record in populated
            ),
            "annulus_incidence": summary["annulus_count"] / summary["eligible_count"],
            "annulus_upstream_fraction": summary["annulus_upstream_l1"]
            / summary["local_upstream_l1"],
            "recoverable_annulus_fraction": summary["recoverable_annulus_upstream_l1"]
            / summary["annulus_upstream_l1"],
            "recovered_total_ratio": summary["recoverable_annulus_candidate_qgrad_l1"]
            / summary["active_hard_qgrad_l1"],
            "recovered_boundary_ratio": summary["recoverable_annulus_candidate_qgrad_l1"]
            / summary["boundary_hard_qgrad_l1"],
            "sampled_views": sampled_views,
            "all_training_views_sampled": sampled_views == TRAIN_INDICES,
            "q_min": min(float(record["q_min"]) for record in populated),
            "q_max": max(float(record["q_max"]) for record in populated),
            "hard_outside_kernel_max_abs": max(
                float(record["hard_outside_kernel_max_abs"]) for record in populated
            ),
            "hard_outside_qgrad_max_abs": max(
                float(record["hard_outside_qgrad_max_abs"]) for record in populated
            ),
            "hard_active_qgrad_max_abs_error": max(
                float(record["hard_active_qgrad_max_abs_error"]) for record in populated
            ),
            "hard_active_qgrad_max_rel_error": max(
                float(record["hard_active_qgrad_max_rel_error"]) for record in populated
            ),
            "q_bins": q_bins,
        }
    )
    return summary


def pool_audit_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    int_fields = (
        "observation_count",
        "eligible_count",
        "interior_count",
        "boundary_count",
        "annulus_count",
    )
    float_fields = (
        "local_upstream_l1",
        "annulus_upstream_l1",
        "recoverable_annulus_upstream_l1",
        "active_hard_qgrad_l1",
        "boundary_hard_qgrad_l1",
        "annulus_candidate_qgrad_l1",
        "recoverable_annulus_candidate_qgrad_l1",
    )
    pooled = {field: sum(int(summary[field]) for summary in summaries) for field in int_fields}
    pooled.update(
        {field: math.fsum(float(summary[field]) for summary in summaries) for field in float_fields}
    )
    sampled_views = sorted(
        {int(view) for summary in summaries for view in summary["sampled_views"]}
    )
    pooled.update(
        {
            "annulus_incidence": pooled["annulus_count"] / pooled["eligible_count"],
            "annulus_upstream_fraction": pooled["annulus_upstream_l1"]
            / pooled["local_upstream_l1"],
            "recoverable_annulus_fraction": pooled["recoverable_annulus_upstream_l1"]
            / pooled["annulus_upstream_l1"],
            "recovered_total_ratio": pooled["recoverable_annulus_candidate_qgrad_l1"]
            / pooled["active_hard_qgrad_l1"],
            "recovered_boundary_ratio": pooled["recoverable_annulus_candidate_qgrad_l1"]
            / pooled["boundary_hard_qgrad_l1"],
            "sampled_views": sampled_views,
            "all_training_views_sampled": sampled_views == TRAIN_INDICES,
        }
    )
    return pooled


def audit_gate(summary: dict[str, Any]) -> bool:
    return bool(
        summary["annulus_upstream_fraction"] >= 0.01
        and summary["recoverable_annulus_fraction"] >= 0.10
        and summary["recovered_total_ratio"] >= 0.001
        and summary["recovered_boundary_ratio"] >= 0.05
        and summary.get("all_training_views_sampled", True)
        and summary["eligible_count"] >= 100_000
    )


def audit_decision(runs: list[dict[str, Any]]) -> dict[str, Any]:
    diffuse = [run["diagnostic_summary"] for run in runs if run["condition"] == "diffuse"]
    if [run["seed"] for run in runs if run["condition"] == "diffuse"] != SEEDS:
        raise AssertionError("audit is missing preregistered diffuse seeds")
    seed_passes = [audit_gate(summary) for summary in diffuse]
    pooled = pool_audit_summaries(diffuse)
    pooled_pass = audit_gate(pooled)
    return {
        "seed_passes": seed_passes,
        "seed_pass_count": sum(seed_passes),
        "pooled": pooled,
        "pooled_pass": pooled_pass,
        "phase_b_authorized": sum(seed_passes) >= 2 and pooled_pass,
    }


def map_history_to_global_views(history: dict[str, Any]) -> None:
    local_schedule = [int(index) for index in history["sampled_train_views"]]
    history["sampled_train_views_local"] = local_schedule
    history["sampled_train_views"] = [TRAIN_INDICES[index] for index in local_schedule]
    for record in history["kernel_support_diagnostics"]:
        record["view_local"] = int(record["view"])
        record["view"] = TRAIN_INDICES[int(record["view"])]


def assert_finite_gaussians(gaussians: Gaussians3D, context: str) -> None:
    for field in ("means", "quats", "log_scales", "opacity", "sh"):
        if not bool(torch.isfinite(getattr(gaussians, field)).all()):
            raise AssertionError(f"{context} contains non-finite {field}")


def assert_finite_tree(value: Any, context: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AssertionError(f"{context} contains non-finite value {value}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_tree(child, f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_finite_tree(child, f"{context}[{index}]")
        return
    raise TypeError(f"{context} contains unsupported value type {type(value).__name__}")


def verify_retry_history() -> dict[str, str]:
    previous_protocol_hash = sha256_file(ROOT / PREVIOUS_SUPPORT_PREREGISTRATION)
    expected_previous_hash = "c78a74ea67a4a0d327b8ef884006dc8ad5781da9a632f557c2e9f370a8868a58"
    if previous_protocol_hash != expected_previous_hash:
        raise AssertionError("incorporated first support protocol differs from its frozen hash")
    verified = {}
    for relative, expected_hash in HISTORICAL_ARTIFACT_HASHES.items():
        path = ROOT / relative
        if not path.is_file():
            raise AssertionError(f"required retry-history artifact is missing: {relative}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise AssertionError(f"retry-history artifact hash differs: {relative}")
        verified[str(relative)] = actual_hash
    failed_note = companion_note_path(ROOT / FAILED_PHASE_B_OUTPUT)
    if (ROOT / FAILED_PHASE_B_OUTPUT).exists() or failed_note.exists():
        raise AssertionError("failed first Phase-B output namespace is unexpectedly occupied")
    verified[str(PREVIOUS_SUPPORT_PREREGISTRATION)] = previous_protocol_hash
    return verified


def verify_default_semantics() -> dict[str, Any]:
    q = torch.tensor([0.0, 11.999, 12.0, 16.0, 50.0], dtype=torch.float64)
    legacy = torch.exp(-0.5 * q.clamp_max(48.0)) * (q < 12.0)
    if not torch.equal(kernel_support_weight(q), legacy):
        raise AssertionError("default kernel support no longer matches the established hard form")
    renderer = TorchRasterizer()
    config = TrainConfig()
    if renderer.kernel_support_mode != "hard" or config.kernel_support_mode != "hard":
        raise AssertionError("renderer or trainer default kernel support is no longer hard")
    if KERNEL_SUPPORT_CUTOFF != 12.0 or KERNEL_SUPPORT_TAPER_WIDTH != 4.0:
        raise AssertionError("repository kernel-support constants differ from the preregistration")
    incorporated_hash = sha256_file(ROOT / INCORPORATED_PREREGISTRATION)
    if incorporated_hash != INCORPORATED_PREREGISTRATION_SHA256:
        raise AssertionError("incorporated SH protocol no longer matches its frozen hash")
    retry_history = verify_retry_history()
    return {
        "kernel_support_weight_default": "hard",
        "torch_rasterizer_default": renderer.kernel_support_mode,
        "train_config_default": config.kernel_support_mode,
        "cutoff": KERNEL_SUPPORT_CUTOFF,
        "taper_width": KERNEL_SUPPORT_TAPER_WIDTH,
        "incorporated_preregistration_sha256": incorporated_hash,
        "retry_history": retry_history,
    }


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
        raise RuntimeError(f"official CPU environment does not match preregistration: {mismatches}")


def strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {value!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def load_and_verify_seal(path: Path) -> dict[str, Any]:
    payload = strict_json_load(path)
    if payload.get("artifact_type") != "kernel_support_taper_implementation_seal":
        raise ValueError(f"{path} is not a kernel-support-taper implementation seal")
    expected_paths = [str(item) for item in SEALED_PATHS]
    if payload.get("sealed_paths") != expected_paths:
        raise RuntimeError("implementation seal path set differs from the frozen repository set")
    paths = tuple(Path(item) for item in payload["sealed_paths"])
    hashes, aggregate = source_hashes(paths)
    if hashes != payload["source_hashes"] or aggregate != payload["source_aggregate"]:
        raise RuntimeError("implementation/protocol differs from the sealed source aggregate")
    if not payload.get("verification", {}).get("passed"):
        raise RuntimeError("implementation seal does not contain passing verification")
    current_environment = environment_metadata()
    assert_official_environment(current_environment)
    if environment_fingerprint(payload["environment"]) != environment_fingerprint(
        current_environment
    ):
        raise RuntimeError("current execution environment differs from the implementation seal")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "source_aggregate": aggregate,
        "retry_history": verify_retry_history(),
        "verification_sha256": canonical_json_hash(payload["verification"]),
        "environment_fingerprint": environment_fingerprint(payload["environment"]),
    }


def verify_loaded_sources_against_seal(
    seal_path: Path,
) -> tuple[dict[str, str], str]:
    seal_payload = strict_json_load(seal_path)
    sealed_hashes = seal_payload["source_hashes"]
    loaded_hashes, loaded_aggregate = loaded_source_hashes()
    unexpected = sorted(set(loaded_hashes) - set(sealed_hashes))
    mismatched = sorted(
        path
        for path, digest in loaded_hashes.items()
        if path in sealed_hashes and sealed_hashes[path] != digest
    )
    if unexpected or mismatched:
        raise RuntimeError(
            f"loaded repository sources are outside/different from seal: "
            f"unexpected={unexpected}, mismatched={mismatched}"
        )
    return loaded_hashes, loaded_aggregate


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
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
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
    current_environment = environment_metadata()
    assert_official_environment(current_environment)
    verification = run_verification()
    hashes, aggregate = source_hashes()
    return {
        "artifact_type": "kernel_support_taper_implementation_seal",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_metadata(),
        "sealed_paths": [str(path) for path in SEALED_PATHS],
        "source_hashes": hashes,
        "source_aggregate": aggregate,
        "incorporated_preregistration": {
            "path": str(INCORPORATED_PREREGISTRATION),
            "sha256": sha256_file(ROOT / INCORPORATED_PREREGISTRATION),
        },
        "verification": verification,
        "default_semantics": verify_default_semantics(),
        "environment": current_environment,
        "command": [sys.executable, *sys.argv],
    }


def run_audit(seal_path: Path, attempt_output: Path) -> dict[str, Any]:
    preflight_output(attempt_output)
    seal = load_and_verify_seal(seal_path)
    verify_default_semantics()
    claim_attempt(
        PHASE_A_ATTEMPT,
        phase="phase_a",
        output=attempt_output,
        inputs={"seal_sha256": seal["sha256"]},
    )
    runs = []
    experiment_started = time.perf_counter()
    for condition in CONDITIONS:
        for seed in SEEDS:
            print(f"audit: preparing {condition} seed {seed}", flush=True)
            scene, _, initialization, preparation = prepare_seed(seed, condition)
            training_scene = scene.subset(TRAIN_INDICES)
            config = train_config(seed, "hard", diagnostics=True)
            started = time.perf_counter()
            final, history = Trainer(config).train(training_scene, initialization.detach())
            training_seconds = time.perf_counter() - started
            assert_finite_gaussians(final, f"hard audit {condition}/{seed}")
            map_history_to_global_views(history)
            assert_finite_tree(history, f"hard audit history {condition}/{seed}")
            if len(history["sampled_train_views"]) != 120:
                raise AssertionError("hard audit did not record the complete view schedule")
            if len(history["kernel_support_diagnostics"]) != 120:
                raise AssertionError("hard audit did not record every kernel diagnostic step")
            if final.n != initialization.n:
                raise AssertionError(f"hard audit changed primitive count for {condition}/{seed}")
            if [int(item[0]) for item in history["psnr"]] != [30, 60, 90, 120]:
                raise AssertionError(f"hard audit checkpoints differ for {condition}/{seed}")
            diagnostic_summary = aggregate_diagnostics(history["kernel_support_diagnostics"])
            hard_metrics = evaluate_final(scene, final, mode="hard")
            runs.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "preparation": preparation,
                    "train_config": asdict(config),
                    "training_seconds": training_seconds,
                    "final_hash": gaussians_hash(final),
                    "final_gaussians": final.n,
                    "hard_metrics": hard_metrics,
                    "diagnostic_summary": diagnostic_summary,
                    "history": history,
                    "schedule_hash": canonical_json_hash(history["sampled_train_views"]),
                }
            )
    decision = audit_decision(runs)
    loaded_hashes, loaded_aggregate = verify_loaded_sources_against_seal(seal_path)
    end_seal = load_and_verify_seal(seal_path)
    if seal != end_seal:
        raise RuntimeError("implementation seal changed during Phase A")
    return {
        "artifact_type": "kernel_support_taper_phase_a_audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_metadata(),
        "seal": seal,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": sha256_file(ROOT / PREREGISTRATION),
        },
        "incorporated_preregistration": {
            "path": str(INCORPORATED_PREREGISTRATION),
            "sha256": sha256_file(ROOT / INCORPORATED_PREREGISTRATION),
        },
        "command": [sys.executable, *sys.argv],
        "environment": environment_metadata(),
        "default_semantics": verify_default_semantics(),
        "loaded_source_hashes": loaded_hashes,
        "loaded_source_aggregate": loaded_aggregate,
        "split": {"train": TRAIN_INDICES, "held_out": TEST_INDICES},
        "runs": runs,
        "decision": decision,
        "wall_seconds": time.perf_counter() - experiment_started,
    }


def validate_phase_a_audit(audit: dict[str, Any], audit_path: Path, seal: dict[str, Any]) -> None:
    if audit.get("artifact_type") != "kernel_support_taper_phase_a_audit":
        raise ValueError("Phase-B input is not a kernel-support-taper Phase-A audit")
    if audit.get("split") != {"train": TRAIN_INDICES, "held_out": TEST_INDICES}:
        raise RuntimeError("Phase-A split differs from the preregistration")
    if audit.get("seal") != seal:
        raise RuntimeError("Phase-A seal binding differs from the current seal")
    expected_preregistration = {
        "path": str(PREREGISTRATION),
        "sha256": sha256_file(ROOT / PREREGISTRATION),
    }
    if audit.get("preregistration") != expected_preregistration:
        raise RuntimeError("Phase-A preregistration binding differs")
    if environment_fingerprint(audit["environment"]) != seal["environment_fingerprint"]:
        raise RuntimeError("Phase-A environment differs from the implementation seal")
    runs = audit.get("runs")
    if not isinstance(runs, list) or len(runs) != len(CONDITIONS) * len(SEEDS):
        raise RuntimeError("Phase-A artifact has an invalid run count")
    identities = [(run.get("condition"), run.get("seed")) for run in runs]
    expected_identities = [(condition, seed) for condition in CONDITIONS for seed in SEEDS]
    if identities != expected_identities:
        raise RuntimeError("Phase-A artifact has missing, duplicated, or reordered runs")
    for run in runs:
        seed = int(run["seed"])
        expected_config = asdict(train_config(seed, "hard", diagnostics=True))
        if run.get("train_config") != expected_config:
            raise RuntimeError(f"Phase-A config differs for {run['condition']}/{seed}")
        history = run.get("history", {})
        schedule = history.get("sampled_train_views")
        if not isinstance(schedule, list) or len(schedule) != 120:
            raise RuntimeError(f"Phase-A schedule is invalid for {run['condition']}/{seed}")
        if canonical_json_hash(schedule) != run.get("schedule_hash"):
            raise RuntimeError(f"Phase-A schedule hash differs for {run['condition']}/{seed}")
        recomputed_summary = aggregate_diagnostics(history.get("kernel_support_diagnostics", []))
        if canonical_json_hash(recomputed_summary) != canonical_json_hash(
            run.get("diagnostic_summary")
        ):
            raise RuntimeError(f"Phase-A diagnostic summary differs for {run['condition']}/{seed}")
    recomputed_decision = audit_decision(runs)
    if canonical_json_hash(recomputed_decision) != canonical_json_hash(audit.get("decision")):
        raise RuntimeError("Phase-A decision does not match recomputed frozen gate")


def verify_phase_a_review(
    path: Path,
    *,
    audit_path: Path,
    seal: dict[str, Any],
) -> dict[str, str]:
    review = strict_json_load(path)
    expected = {
        "artifact_type": "kernel_support_taper_phase_a_scientist_review",
        "verdict": "pass",
        "phase_b_execution_clearance": True,
        "audit_sha256": sha256_file(audit_path),
        "seal_sha256": seal["sha256"],
        "source_aggregate": seal["source_aggregate"],
    }
    if set(review) != set(expected):
        raise RuntimeError(
            "Phase-A scientist review has missing or unexpected keys: "
            f"expected={sorted(expected)}, actual={sorted(review)}"
        )
    mismatches = {
        key: {"expected": value, "actual": review.get(key)}
        for key, value in expected.items()
        if review.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Phase-A scientist review is invalid or unbound: {mismatches}")
    return {"path": str(path), "sha256": sha256_file(path)}


def _metric_samples(
    audit: dict[str, Any],
    candidate_runs: list[dict[str, Any]],
    condition: str,
    arm: str,
    metric: str,
) -> list[float]:
    if arm == "hard":
        selected = [run for run in audit["runs"] if run["condition"] == condition]
        return [float(run["hard_metrics"]["mean"][metric]) for run in selected]
    selected = [
        run for run in candidate_runs if run["condition"] == condition and run["arm"] == arm
    ]
    return [float(run["hard_metrics"]["mean"][metric]) for run in selected]


def summarize_ablation(
    audit: dict[str, Any], candidate_runs: list[dict[str, Any]]
) -> dict[str, Any]:
    metrics = (
        "psnr_fg",
        "psnr_full",
        "psnr_crop",
        "ssim_crop",
        "depth_rmse_over_extent",
        "alpha_iou",
        "foreground_coverage",
    )
    summary = {}
    for condition in CONDITIONS:
        summary[condition] = {}
        for arm in ("hard", *CANDIDATE_ARMS):
            summary[condition][arm] = {}
            for metric in metrics:
                samples = _metric_samples(audit, candidate_runs, condition, arm, metric)
                if len(samples) != len(SEEDS) or not all(math.isfinite(value) for value in samples):
                    raise AssertionError(f"invalid summary samples for {condition}/{arm}/{metric}")
                summary[condition][arm][metric] = {
                    "samples": samples,
                    "mean": statistics.fmean(samples),
                    "stdev": statistics.stdev(samples),
                }
    return summary


def ablation_decision(summary: dict[str, Any]) -> dict[str, Any]:
    def samples(condition: str, arm: str, metric: str) -> list[float]:
        return summary[condition][arm][metric]["samples"]

    def mean(condition: str, arm: str, metric: str) -> float:
        return float(summary[condition][arm][metric]["mean"])

    hard_psnr = samples("diffuse", "hard", "psnr_fg")
    smooth_psnr = samples("diffuse", "c1_taper", "psnr_fg")
    control_psnr = samples("diffuse", "hard_forward_c1_taper_gradient", "psnr_fg")
    smooth_gains = [candidate - hard for candidate, hard in zip(smooth_psnr, hard_psnr)]
    control_gains = [candidate - hard for candidate, hard in zip(control_psnr, hard_psnr)]
    smooth_mean_gain = statistics.fmean(smooth_gains)
    control_mean_gain = statistics.fmean(control_gains)
    smooth_seed_wins = sum(gain > 0.0 for gain in smooth_gains)
    control_seed_wins = sum(gain > 0.0 for gain in control_gains)

    hard_ssim = samples("diffuse", "hard", "ssim_crop")
    smooth_ssim = samples("diffuse", "c1_taper", "ssim_crop")
    ssim_deltas = [candidate - hard for candidate, hard in zip(smooth_ssim, hard_ssim)]
    hard_depth = mean("diffuse", "hard", "depth_rmse_over_extent")
    smooth_depth = mean("diffuse", "c1_taper", "depth_rmse_over_extent")
    if hard_depth <= 0.0:
        raise AssertionError("hard depth RMSE denominator is zero")
    depth_regression = (smooth_depth - hard_depth) / hard_depth
    alpha_iou_delta = mean("diffuse", "c1_taper", "alpha_iou") - mean(
        "diffuse", "hard", "alpha_iou"
    )
    coverage_delta = mean("diffuse", "c1_taper", "foreground_coverage") - mean(
        "diffuse", "hard", "foreground_coverage"
    )
    view_hard = samples("view_dependent", "hard", "psnr_fg")
    view_smooth = samples("view_dependent", "c1_taper", "psnr_fg")
    view_deltas = [candidate - hard for candidate, hard in zip(view_smooth, view_hard)]

    criteria = {
        "taper_mean_psnr_gain_at_least_0_10_db": smooth_mean_gain >= 0.10,
        "taper_psnr_wins_at_least_two_seeds": smooth_seed_wins >= 2,
        "mean_ssim_regression_within_0_002": statistics.fmean(ssim_deltas) >= -0.002,
        "per_seed_ssim_regression_within_0_005": min(ssim_deltas) >= -0.005,
        "depth_rmse_regression_within_2_percent": depth_regression <= 0.02,
        "alpha_iou_regression_within_0_02": alpha_iou_delta >= -0.02,
        "coverage_regression_within_0_02": coverage_delta >= -0.02,
        "view_dependent_mean_psnr_regression_within_0_10_db": statistics.fmean(view_deltas)
        >= -0.10,
        "view_dependent_per_seed_psnr_regression_within_0_25_db": min(view_deltas) >= -0.25,
        "hard_forward_control_wins_at_least_two_seeds": control_seed_wins >= 2,
        "hard_forward_control_preserves_half_mean_gain": control_mean_gain
        >= 0.5 * smooth_mean_gain,
    }
    return {
        "criteria": criteria,
        "taper_psnr_gains_db": smooth_gains,
        "taper_mean_psnr_gain_db": smooth_mean_gain,
        "taper_seed_wins": smooth_seed_wins,
        "control_psnr_gains_db": control_gains,
        "control_mean_psnr_gain_db": control_mean_gain,
        "control_seed_wins": control_seed_wins,
        "ssim_deltas": ssim_deltas,
        "depth_rmse_regression_fraction": depth_regression,
        "alpha_iou_delta": alpha_iou_delta,
        "foreground_coverage_delta": coverage_delta,
        "view_dependent_psnr_deltas_db": view_deltas,
        "primary_hypothesis_pass": all(criteria.values()),
    }


def run_ablation(
    audit_path: Path,
    seal_path: Path,
    review_path: Path,
    attempt_output: Path,
) -> dict[str, Any]:
    preflight_output(attempt_output)
    seal = load_and_verify_seal(seal_path)
    audit = strict_json_load(audit_path)
    validate_phase_a_audit(audit, audit_path, seal)
    review = verify_phase_a_review(review_path, audit_path=audit_path, seal=seal)
    if not audit["decision"]["phase_b_authorized"]:
        raise RuntimeError("Phase A did not authorize candidate arms")
    claim_attempt(
        PHASE_B_ATTEMPT,
        phase="phase_b",
        output=attempt_output,
        inputs={
            "seal_sha256": seal["sha256"],
            "audit_sha256": sha256_file(audit_path),
            "review_sha256": review["sha256"],
        },
    )

    audit_lookup = {(run["condition"], int(run["seed"])): run for run in audit["runs"]}
    candidate_runs = []
    experiment_started = time.perf_counter()
    for condition in CONDITIONS:
        for seed in SEEDS:
            print(f"ablation: recreating {condition} seed {seed}", flush=True)
            scene, _, initialization, preparation = prepare_seed(seed, condition)
            baseline = audit_lookup[(condition, seed)]
            for field in ("scene_hashes", "fitted_hash", "initialization_hash"):
                if preparation[field] != baseline["preparation"][field]:
                    raise AssertionError(
                        f"Phase-B recreation differs in {condition}/{seed}/{field}"
                    )
            invariants = step0_invariants(initialization, scene)
            training_scene = scene.subset(TRAIN_INDICES)
            for arm in CANDIDATE_ARMS:
                config = train_config(seed, arm, diagnostics=False)
                expected_common = dict(baseline["train_config"])
                expected_common["kernel_support_mode"] = arm
                expected_common["collect_kernel_support_diagnostics"] = False
                if asdict(config) != expected_common:
                    raise AssertionError(
                        "candidate config differs beyond kernel support for "
                        f"{condition}/{seed}/{arm}"
                    )
                started = time.perf_counter()
                final, history = Trainer(config).train(training_scene, initialization.detach())
                training_seconds = time.perf_counter() - started
                assert_finite_gaussians(final, f"candidate {condition}/{seed}/{arm}")
                map_history_to_global_views(history)
                assert_finite_tree(history, f"candidate history {condition}/{seed}/{arm}")
                if history["sampled_train_views"] != baseline["history"]["sampled_train_views"]:
                    raise AssertionError(
                        f"target-view schedule differs for {condition}/{seed}/{arm}"
                    )
                if canonical_json_hash(history["active_sh_degree"]) != canonical_json_hash(
                    baseline["history"]["active_sh_degree"]
                ):
                    raise AssertionError(f"SH-degree schedule differs for {condition}/{seed}/{arm}")
                if canonical_json_hash(history["n_gaussians"]) != canonical_json_hash(
                    baseline["history"]["n_gaussians"]
                ):
                    raise AssertionError(
                        f"primitive-count schedule differs for {condition}/{seed}/{arm}"
                    )
                if final.n != initialization.n or final.n != baseline["final_gaussians"]:
                    raise AssertionError(
                        f"final primitive count differs for {condition}/{seed}/{arm}"
                    )
                checkpoint_steps = [int(item[0]) for item in history["psnr"]]
                baseline_checkpoint_steps = [int(item[0]) for item in baseline["history"]["psnr"]]
                if checkpoint_steps != [30, 60, 90, 120] or (
                    checkpoint_steps != baseline_checkpoint_steps
                ):
                    raise AssertionError(
                        f"evaluation checkpoints differ for {condition}/{seed}/{arm}"
                    )
                if not all(math.isfinite(float(value)) for value in history["loss"]):
                    raise AssertionError(
                        f"non-finite training history for {condition}/{seed}/{arm}"
                    )
                candidate_runs.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "arm": arm,
                        "preparation_hashes_verified": True,
                        "step0_invariants": invariants,
                        "train_config": asdict(config),
                        "training_seconds": training_seconds,
                        "final_hash": gaussians_hash(final),
                        "final_gaussians": final.n,
                        "hard_metrics": evaluate_final(scene, final, mode="hard"),
                        "matched_metrics": evaluate_final(scene, final, mode=arm),
                        "history": history,
                        "schedule_hash": canonical_json_hash(history["sampled_train_views"]),
                    }
                )
    summary = summarize_ablation(audit, candidate_runs)
    decision = ablation_decision(summary)
    loaded_hashes, loaded_aggregate = verify_loaded_sources_against_seal(seal_path)
    if load_and_verify_seal(seal_path) != seal:
        raise RuntimeError("implementation seal changed during Phase B")
    return {
        "artifact_type": "kernel_support_taper_phase_b_ablation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_metadata(),
        "seal": seal,
        "phase_a": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        "phase_a_scientist_review": review,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": sha256_file(ROOT / PREREGISTRATION),
        },
        "incorporated_preregistration": {
            "path": str(INCORPORATED_PREREGISTRATION),
            "sha256": sha256_file(ROOT / INCORPORATED_PREREGISTRATION),
        },
        "command": [sys.executable, *sys.argv],
        "environment": environment_metadata(),
        "loaded_source_hashes": loaded_hashes,
        "loaded_source_aggregate": loaded_aggregate,
        "split": {"train": TRAIN_INDICES, "held_out": TEST_INDICES},
        "runs": candidate_runs,
        "summary": summary,
        "decision": decision,
        "wall_seconds": time.perf_counter() - experiment_started,
    }


def companion_note_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_RESULT.md")


def preflight_output(output: Path) -> None:
    note = companion_note_path(output)
    if output.exists() or note.exists():
        raise FileExistsError(f"refusing to start: {output} or {note} already exists")
    output.parent.mkdir(parents=True, exist_ok=True)


def claim_attempt(
    path: Path,
    *,
    phase: str,
    output: Path,
    inputs: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_type": "kernel_support_taper_once_only_attempt",
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
    artifact_type = payload["artifact_type"]
    lines = [
        f"# {artifact_type}",
        "",
        f"- Timestamp (UTC): `{payload['timestamp_utc']}`",
        f"- JSON artifact: `{output}`",
        f"- JSON SHA-256: `{digest}`",
        f"- Command: `{' '.join(payload['command'])}`",
    ]
    if "seal" in payload:
        lines.append(f"- Implementation seal: `{payload['seal']['source_aggregate']}`")
    if artifact_type == "kernel_support_taper_phase_a_audit":
        decision = payload["decision"]
        disposition = (
            "Phase B remains blocked until an independent scientist review binds this exact "
            "passing audit and grants the execution clearance required by the harness."
            if decision["phase_b_authorized"]
            else "The frozen gate failed, so Phase B is permanently forbidden under this protocol."
        )
        lines.extend(
            [
                "",
                "## Frozen gate decision",
                "",
                f"- Seed passes: `{decision['seed_passes']}`",
                f"- Pooled pass: `{decision['pooled_pass']}`",
                f"- Phase B authorized: `{decision['phase_b_authorized']}`",
                "",
                "This is a CPU synthetic mechanism audit. It is not a real-scene, CUDA, speed, "
                f"or default-change result. {disposition}",
            ]
        )
    elif artifact_type == "kernel_support_taper_phase_b_ablation":
        decision = payload["decision"]
        lines.extend(
            [
                "",
                "## Frozen outcome decision",
                "",
                f"- Primary hypothesis pass: `{decision['primary_hypothesis_pass']}`",
                f"- C1 taper mean foreground PSNR gain: "
                f"`{decision['taper_mean_psnr_gain_db']:.6f} dB`",
                f"- Hard-forward control mean gain: "
                f"`{decision['control_mean_psnr_gain_db']:.6f} dB`",
                "",
                "The common hard renderer defines primary evaluation. This result is limited to "
                "fixed-topology CPU synthetic depth-initialized refinement.",
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

    audit = subparsers.add_parser("audit", help="run the official hard-support incidence audit")
    audit.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    audit.add_argument("--output", type=Path, required=True)

    ablate = subparsers.add_parser("ablate", help="run candidate arms after an authorized audit")
    ablate.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    ablate.add_argument("--audit", type=Path, required=True)
    ablate.add_argument("--phase-a-review", type=Path, required=True)
    ablate.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if (
        args.command_name in {"audit", "ablate"}
        and "kernel_support_taper_iter2" not in args.output.name
    ):
        raise ValueError("retry outputs must use the kernel_support_taper_iter2 namespace")
    preflight_output(args.output)
    if args.command_name == "seal":
        payload = create_seal()
    elif args.command_name == "audit":
        payload = run_audit(args.seal, args.output)
    elif args.command_name == "ablate":
        payload = run_ablation(args.audit, args.seal, args.phase_a_review, args.output)
    else:  # pragma: no cover - argparse enforces the choices
        raise AssertionError(args.command_name)
    note, digest = write_artifact(args.output, payload)
    print(f"saved {args.output} (sha256={digest})", flush=True)
    print(f"saved {note}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
