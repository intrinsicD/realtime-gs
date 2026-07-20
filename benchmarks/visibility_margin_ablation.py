#!/usr/bin/env python3
"""Preregistered coarse visibility-margin audit and gated support-safe ablation."""

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

try:
    import benchmarks.kernel_support_taper_ablation as support_bench
except ModuleNotFoundError as error:
    if error.name not in {"benchmarks", "benchmarks.kernel_support_taper_ablation"}:
        raise
    import kernel_support_taper_ablation as support_bench

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.metrics import masked_crop, masked_psnr, psnr, ssim
from rtgs.data.scene import SceneData
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.torch_ref import KERNEL_SUPPORT_CUTOFF, TorchRasterizer, kernel_support_weight

ROOT = Path(__file__).resolve().parent.parent
PREREGISTRATION = Path("benchmarks/results/20260715_visibility_margin_iter2_PREREG.md")
PREVIOUS_PREREGISTRATION = Path("benchmarks/results/20260715_visibility_margin_PREREG.md")
PREVIOUS_PREREGISTRATION_SHA256 = "1a0d9ec8c211a678898a699650fab2e2ab4c146c4d82df801e40622ab551767a"
PREVIOUS_SEAL = Path("benchmarks/results/20260715_visibility_margin_SEAL.json")
PREVIOUS_SEAL_SHA256 = "92396fc86621d432cf0b53be6e37376578b2f4271ab979141c7bdc117f8b1b99"
PREVIOUS_ATTEMPT = Path("benchmarks/results/20260715_visibility_margin_PHASE_A_ATTEMPT.json")
PREVIOUS_ATTEMPT_SHA256 = "13f3b8515d2a8657c6cb12230c55f0c60d2c253694914ff8c91cfb056490c149"
FAILED_PHASE_A_OUTPUT = Path("benchmarks/results/20260715T211212Z_cpu_visibility_margin_audit.json")
FAILED_PHASE_A_NOTE = Path(
    "benchmarks/results/20260715T211212Z_cpu_visibility_margin_audit_RESULT.md"
)
INCORPORATED_PREREGISTRATION = Path("benchmarks/results/20260715_kernel_support_taper_PREREG.md")
INCORPORATED_PREREGISTRATION_SHA256 = (
    "c78a74ea67a4a0d327b8ef884006dc8ad5781da9a632f557c2e9f370a8868a58"
)
DEFAULT_SEAL = Path("benchmarks/results/20260715_visibility_margin_iter2_SEAL.json")
PHASE_A_ATTEMPT = ROOT / "benchmarks/results/20260715_visibility_margin_iter2_PHASE_A_ATTEMPT.json"
PHASE_B_ATTEMPT = ROOT / "benchmarks/results/20260715_visibility_margin_iter2_PHASE_B_ATTEMPT.json"
CONDITIONS = ("diffuse", "view_dependent")
SEEDS = [0, 1, 2]
TRAIN_INDICES = [0, 1, 2, 4, 5, 6, 8, 9, 10]
TEST_INDICES = [3, 7, 11]
CURRENT_MARGIN_SIGMA = 3.0
SUPPORT_SAFE_MARGIN_SIGMA = math.sqrt(KERNEL_SUPPORT_CUTOFF)
CANDIDATE_ARM = "support_safe"
Q_BINS = ((9.0, 10.0), (10.0, 11.0), (11.0, 12.0))

SEALED_PATHS = tuple(
    sorted(
        {
            PREREGISTRATION,
            PREVIOUS_PREREGISTRATION,
            PREVIOUS_SEAL,
            PREVIOUS_ATTEMPT,
            INCORPORATED_PREREGISTRATION,
            Path("benchmarks/visibility_margin_ablation.py"),
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
    for relative in (PREREGISTRATION, INCORPORATED_PREREGISTRATION, Path("pyproject.toml")):
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
        ["git", "diff", "--binary", "HEAD"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    return {
        "revision": revision,
        "dirty": bool(status.strip()),
        "status": status.splitlines(),
        "tracked_diff_sha256": sha256_bytes(diff),
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


def assert_finite_tree(value: Any, context: str) -> None:
    support_bench.assert_finite_tree(value, context)


def assert_finite_gaussians(gaussians: Gaussians3D, context: str) -> None:
    support_bench.assert_finite_gaussians(gaussians, context)


def train_config(seed: int, margin_sigma: float) -> TrainConfig:
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
        kernel_support_mode="hard",
        collect_kernel_support_diagnostics=False,
        visibility_margin_sigma=margin_sigma,
        validate_render_finite=True,
        seed=seed,
    )


def prepare_seed(seed: int, condition: str):
    return support_bench.prepare_seed(seed, condition)


def map_history_to_global_views(history: dict[str, Any]) -> None:
    local = [int(index) for index in history["sampled_train_views"]]
    history["sampled_train_views_local"] = local
    history["sampled_train_views"] = [TRAIN_INDICES[index] for index in local]


def _projection(
    gaussians: Gaussians3D, camera
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    means_cam = camera.world_to_cam(gaussians.means)
    z = means_cam[:, 2]
    in_front = z > 0.05
    uv, _ = camera.project(gaussians.means)
    cov_world = gaussians.covariance()
    r_wc = camera.R.to(gaussians.means)
    cov_cam = r_wc @ cov_world @ r_wc.T
    zs = z.clamp_min(0.05)
    jac = torch.zeros(gaussians.n, 2, 3, device=z.device, dtype=z.dtype)
    jac[:, 0, 0] = camera.fx / zs
    jac[:, 0, 2] = -camera.fx * means_cam[:, 0] / zs.square()
    jac[:, 1, 1] = camera.fy / zs
    jac[:, 1, 2] = -camera.fy * means_cam[:, 1] / zs.square()
    cov2d = jac @ cov_cam @ jac.transpose(-1, -2)
    cov2d = cov2d + 0.3 * torch.eye(2, device=z.device, dtype=z.dtype)
    eig_max = (
        0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
        + (0.25 * (cov2d[:, 0, 0] - cov2d[:, 1, 1]).square() + cov2d[:, 0, 1].square()).sqrt()
    )
    sigma = eig_max.clamp_min(1e-8).sqrt()
    return z, in_front, uv, cov2d, sigma


def _render_hash(output) -> str:
    return support_bench.tensor_collection_hash(
        [("color", output.color), ("alpha", output.alpha), ("depth", output.depth)]
    )


def audit_checkpoint(scene: SceneData, gaussians: Gaussians3D) -> dict[str, Any]:
    """Audit current versus support-safe visibility at one frozen checkpoint."""
    current_renderer = TorchRasterizer(visibility_margin_sigma=CURRENT_MARGIN_SIGMA)
    safe_renderer = TorchRasterizer(visibility_margin_sigma=SUPPORT_SAFE_MARGIN_SIGMA)
    views = []
    with torch.no_grad():
        for view_index in TRAIN_INDICES:
            camera = scene.cameras[view_index]
            z, in_front, uv, cov2d, sigma = _projection(gaussians, camera)
            for label, tensor in (
                ("projected depth", z),
                ("projected center", uv),
                ("projected covariance", cov2d),
                ("projected sigma", sigma),
            ):
                if not bool(torch.isfinite(tensor).all()):
                    raise AssertionError(f"{label} is non-finite in view {view_index}")
            current_visible = in_front & camera.in_image(
                uv, margin=(CURRENT_MARGIN_SIGMA * sigma).detach()
            )
            safe_visible = in_front & camera.in_image(
                uv, margin=(SUPPORT_SAFE_MARGIN_SIGMA * sigma).detach()
            )
            if bool((current_visible & ~safe_visible).any()):
                raise AssertionError("current-visible is not a subset of support-safe-visible")
            newly_admitted = safe_visible & ~current_visible
            if bool((newly_admitted & ~in_front).any()):
                raise AssertionError("support-safe margin changed the near-plane set")
            current_indices = current_visible.nonzero(as_tuple=True)[0]
            safe_indices = safe_visible.nonzero(as_tuple=True)[0]
            current_order = current_indices[torch.argsort(z[current_indices])]
            new_indices = (safe_visible & ~current_visible).nonzero(as_tuple=True)[0]
            new_order = new_indices[torch.argsort(z[new_indices])]
            safe_order = torch.cat((current_order, new_order))
            safe_order = safe_order[torch.argsort(z[safe_order], stable=True)]
            if safe_indices.numel() != safe_order.numel() or not bool(
                torch.isin(safe_indices, safe_order).all()
            ):
                raise AssertionError("support-safe depth order does not cover its visible set")
            if safe_order.numel() > 1 and bool((z[safe_order][1:] < z[safe_order][:-1]).any()):
                raise AssertionError("support-safe depth order is not monotone")
            safe_filtered = safe_order[torch.isin(safe_order, current_order)]
            if not torch.equal(safe_filtered, current_order):
                raise AssertionError("current depth order differs from filtered support-safe order")

            in_front_indices = in_front.nonzero(as_tuple=True)[0]
            if in_front_indices.numel() == 0:
                raise AssertionError(f"in-front set is empty in view {view_index}")
            means2d = uv[in_front_indices]
            cov = cov2d[in_front_indices]
            det = (cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1].square()).clamp_min(1e-12)
            i00 = cov[:, 1, 1] / det
            i01 = -cov[:, 0, 1] / det
            i11 = cov[:, 0, 0] / det
            inverse = torch.stack((i00, i01, i11), dim=-1)
            if not bool(torch.isfinite(inverse).all()):
                raise AssertionError(
                    f"projected covariance inverse is non-finite in view {view_index}"
                )
            xs = torch.arange(camera.width, device=z.device, dtype=z.dtype) + 0.5
            ys = torch.arange(camera.height, device=z.device, dtype=z.dtype) + 0.5
            pixels = torch.stack(
                [
                    xs[None, :].expand(camera.height, camera.width),
                    ys[:, None].expand(camera.height, camera.width),
                ],
                dim=-1,
            ).reshape(-1, 2)
            d = pixels[:, None, :] - means2d[None, :, :]
            q = (
                d[..., 0].square() * i00[None]
                + 2.0 * d[..., 0] * d[..., 1] * i01[None]
                + d[..., 1].square() * i11[None]
            )
            if not bool(torch.isfinite(q).all()):
                raise AssertionError(f"Mahalanobis q is non-finite in view {view_index}")
            support = q < KERNEL_SUPPORT_CUTOFF
            has_support = support.any(dim=0)
            local_safe = safe_visible[in_front_indices]
            if bool((has_support & ~local_safe).any()):
                raise AssertionError(
                    "an all-in-front support pair is outside support-safe visibility"
                )
            local_current = current_visible[in_front_indices]
            included = support & local_current[None]
            missed = support & ~local_current[None]
            if bool(missed.any()) and float(q[missed].min()) < 9.0 - 1e-5:
                raise AssertionError("newly admitted support pair violates the frozen q>=9 shell")
            opacity = gaussians.opacity[in_front_indices]
            kernel = kernel_support_weight(q, "hard")
            if not bool(torch.isfinite(kernel).all()):
                raise AssertionError(f"kernel is non-finite in view {view_index}")
            effective_mass = opacity[None] * kernel
            if not bool(torch.isfinite(effective_mass).all()):
                raise AssertionError(f"kernel/effective mass is non-finite in view {view_index}")
            included_count = int(included.sum())
            missed_count = int(missed.sum())
            support_count = included_count + missed_count
            included_mass = float(effective_mass[included].double().sum())
            missed_mass = float(effective_mass[missed].double().sum())
            missed_indices: list[int] = []
            if bool(missed.any()):
                local_missed = missed.any(dim=0).nonzero(as_tuple=True)[0]
                missed_indices = sorted(
                    int(in_front_indices[index]) for index in local_missed.tolist()
                )

            bins = []
            for low, high in Q_BINS:
                member = missed & (q >= low) & (q < high)
                bins.append(
                    {
                        "low": low,
                        "high": high,
                        "count": int(member.sum()),
                        "effective_mass": float(effective_mass[member].double().sum()),
                    }
                )

            current = current_renderer.render(gaussians, camera)
            safe = safe_renderer.render(gaussians, camera)
            if not torch.equal(current.visible, current_order):
                raise AssertionError("current renderer depth order differs from audit projection")
            if not torch.equal(safe.visible, safe_order):
                raise AssertionError("support-safe renderer depth order differs from retry rule")
            for label, output in (("current", current), ("support_safe", safe)):
                for field in ("color", "alpha", "depth"):
                    if not bool(torch.isfinite(getattr(output, field)).all()):
                        raise AssertionError(f"{label} {field} is non-finite in view {view_index}")
            target = scene.images[view_index].to(current.color)
            residual_l1 = float((current.color - target).abs().double().sum())
            render_delta_l1 = float((safe.color - current.color).abs().double().sum())
            current_objective = float(
                0.8 * (current.color - target).abs().mean()
                + 0.2 * (1.0 - ssim(current.color, target))
            )
            safe_objective = float(
                0.8 * (safe.color - target).abs().mean() + 0.2 * (1.0 - ssim(safe.color, target))
            )
            required_positive = (
                support_count,
                included_mass + missed_mass,
                residual_l1,
                current_objective,
            )
            if any(
                not math.isfinite(float(value)) or float(value) <= 0.0
                for value in required_positive
            ):
                raise AssertionError("coarse-margin audit denominator is zero or non-finite")
            views.append(
                {
                    "view": view_index,
                    "current_visible_gaussians": int(current_visible.sum()),
                    "support_safe_visible_gaussians": int(safe_visible.sum()),
                    "all_in_front_gaussians": int(in_front.sum()),
                    "newly_admitted_gaussians": int(newly_admitted.sum()),
                    "missed_gaussian_indices": missed_indices,
                    "included_pair_count": included_count,
                    "missed_pair_count": missed_count,
                    "support_pair_count": support_count,
                    "included_effective_mass": included_mass,
                    "missed_effective_mass": missed_mass,
                    "total_effective_mass": included_mass + missed_mass,
                    "residual_l1": residual_l1,
                    "render_delta_l1": render_delta_l1,
                    "current_objective": current_objective,
                    "support_safe_objective": safe_objective,
                    "objective_delta": safe_objective - current_objective,
                    "current_render_hash": _render_hash(current),
                    "support_safe_render_hash": _render_hash(safe),
                    "maximum_color_delta": float((safe.color - current.color).abs().max()),
                    "maximum_alpha_delta": float((safe.alpha - current.alpha).abs().max()),
                    "maximum_depth_delta": float((safe.depth - current.depth).abs().max()),
                    "q_bins": bins,
                }
            )
            assert_finite_tree(views[-1], f"coarse-margin audit view {view_index}")
    return {"views": views, "summary": aggregate_checkpoint_views(views)}


def aggregate_checkpoint_views(views: list[dict[str, Any]]) -> dict[str, Any]:
    if [int(view["view"]) for view in views] != TRAIN_INDICES:
        raise AssertionError("checkpoint audit does not contain all frozen training views")
    for view in views:
        indices = view.get("missed_gaussian_indices")
        if not isinstance(indices, list) or indices != sorted(set(int(item) for item in indices)):
            raise AssertionError("missed Gaussian exposure identities are invalid")
    integer_fields = ("included_pair_count", "missed_pair_count", "support_pair_count")
    float_fields = (
        "included_effective_mass",
        "missed_effective_mass",
        "total_effective_mass",
        "residual_l1",
        "render_delta_l1",
        "current_objective",
        "support_safe_objective",
    )
    result: dict[str, Any] = {
        field: sum(int(view[field]) for view in views) for field in integer_fields
    }
    result.update(
        {field: math.fsum(float(view[field]) for view in views) for field in float_fields}
    )
    result["objective_delta"] = result["support_safe_objective"] - result["current_objective"]
    if result["support_pair_count"] != result["included_pair_count"] + result["missed_pair_count"]:
        raise AssertionError("support-pair partition is inconsistent")
    if result["total_effective_mass"] <= 0.0 or result["residual_l1"] <= 0.0:
        raise AssertionError("checkpoint pooled denominator is zero")
    q_bins = []
    for index, (low, high) in enumerate(Q_BINS):
        members = [view["q_bins"][index] for view in views]
        if any(member["low"] != low or member["high"] != high for member in members):
            raise AssertionError("q-bin boundaries differ across views")
        q_bins.append(
            {
                "low": low,
                "high": high,
                "count": sum(int(member["count"]) for member in members),
                "effective_mass": math.fsum(float(member["effective_mass"]) for member in members),
            }
        )
    if sum(int(item["count"]) for item in q_bins) != result["missed_pair_count"]:
        raise AssertionError("missed pairs are not fully partitioned by the frozen q bins")
    result.update(
        {
            "missed_pair_fraction": result["missed_pair_count"] / result["support_pair_count"],
            "missed_effective_mass_fraction": result["missed_effective_mass"]
            / result["total_effective_mass"],
            "render_delta_over_residual": result["render_delta_l1"] / result["residual_l1"],
            "signed_relative_objective_change": result["objective_delta"]
            / result["current_objective"],
            "audited_views": [int(view["view"]) for view in views],
            "all_training_views_audited": True,
            "distinct_missed_gaussian_view_exposures": sum(
                len(view["missed_gaussian_indices"]) for view in views
            ),
            "missed_gaussian_view_exposures": [
                [int(view["view"]), int(index)]
                for view in views
                for index in view["missed_gaussian_indices"]
            ],
            "q_bins": q_bins,
        }
    )
    assert_finite_tree(result, "checkpoint aggregate")
    return result


def material_gate(summary: dict[str, Any]) -> bool:
    return bool(
        summary["missed_pair_fraction"] >= 0.0005
        and summary["missed_effective_mass_fraction"] >= 0.0005
        and summary["render_delta_over_residual"] >= 0.001
        and summary["missed_pair_count"] >= 100
        and summary["distinct_missed_gaussian_view_exposures"] >= 3
    )


def validity_gate(summary: dict[str, Any], *, minimum_support_pairs: int = 100_000) -> bool:
    return bool(
        summary["all_training_views_audited"]
        and summary["support_pair_count"] >= minimum_support_pairs
    )


def audit_gate(summary: dict[str, Any]) -> bool:
    """Combined per-seed gate retained for focused unit checks."""
    return material_gate(summary) and validity_gate(summary)


def pool_summaries(seed_summaries: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    summaries = [summary for _, summary in seed_summaries]
    integer_fields = ("included_pair_count", "missed_pair_count", "support_pair_count")
    float_fields = (
        "included_effective_mass",
        "missed_effective_mass",
        "total_effective_mass",
        "residual_l1",
        "render_delta_l1",
        "current_objective",
        "support_safe_objective",
    )
    pooled: dict[str, Any] = {
        field: sum(int(summary[field]) for summary in summaries) for field in integer_fields
    }
    exposure_keys = sorted(
        {
            (int(seed), int(exposure[0]), int(exposure[1]))
            for seed, summary in seed_summaries
            for exposure in summary["missed_gaussian_view_exposures"]
        }
    )
    pooled.update(
        {field: math.fsum(float(summary[field]) for summary in summaries) for field in float_fields}
    )
    pooled["objective_delta"] = pooled["support_safe_objective"] - pooled["current_objective"]
    pooled.update(
        {
            "missed_pair_fraction": pooled["missed_pair_count"] / pooled["support_pair_count"],
            "missed_effective_mass_fraction": pooled["missed_effective_mass"]
            / pooled["total_effective_mass"],
            "render_delta_over_residual": pooled["render_delta_l1"] / pooled["residual_l1"],
            "signed_relative_objective_change": pooled["objective_delta"]
            / pooled["current_objective"],
            "distinct_missed_gaussian_view_exposures": len(exposure_keys),
            "missed_gaussian_view_exposures": [list(item) for item in exposure_keys],
            "all_training_views_audited": all(
                bool(summary["all_training_views_audited"]) for summary in summaries
            ),
        }
    )
    assert_finite_tree(pooled, "pooled diffuse audit")
    return pooled


def audit_decision(runs: list[dict[str, Any]]) -> dict[str, Any]:
    diffuse = [run["final_incidence"]["summary"] for run in runs if run["condition"] == "diffuse"]
    identities = [int(run["seed"]) for run in runs if run["condition"] == "diffuse"]
    if identities != SEEDS:
        raise AssertionError("audit is missing preregistered diffuse seeds")
    seed_material_passes = [material_gate(summary) for summary in diffuse]
    seed_validity_passes = [validity_gate(summary) for summary in diffuse]
    seed_passes = [
        material and valid for material, valid in zip(seed_material_passes, seed_validity_passes)
    ]
    pooled = pool_summaries(list(zip(SEEDS, diffuse)))
    pooled_material_pass = material_gate(pooled)
    pooled_validity_pass = validity_gate(pooled, minimum_support_pairs=300_000)
    pooled_pass = pooled_material_pass and pooled_validity_pass
    return {
        "seed_material_passes": seed_material_passes,
        "seed_validity_passes": seed_validity_passes,
        "seed_passes": seed_passes,
        "seed_pass_count": sum(seed_passes),
        "pooled": pooled,
        "pooled_material_pass": pooled_material_pass,
        "pooled_validity_pass": pooled_validity_pass,
        "pooled_pass": pooled_pass,
        "phase_b_authorized": (
            all(seed_validity_passes)
            and sum(seed_material_passes) >= 2
            and pooled_material_pass
            and pooled_validity_pass
        ),
    }


def evaluate_final(
    scene: SceneData, gaussians: Gaussians3D, *, margin_sigma: float
) -> dict[str, Any]:
    renderer = TorchRasterizer(visibility_margin_sigma=margin_sigma)
    truth_renderer = TorchRasterizer(visibility_margin_sigma=CURRENT_MARGIN_SIGMA)
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
            target = scene.images[index].clamp(0.0, 1.0)
            color = predicted.color.clamp(0.0, 1.0)
            truth_support = truth.alpha > 0.05
            predicted_support = predicted.alpha > 0.05
            intersection = truth_support & predicted_support
            union = truth_support | predicted_support
            if not bool(truth_support.any()) or not bool(intersection.any()):
                raise AssertionError(f"held-out view {index} has empty evaluation support")
            predicted_crop = masked_crop(color, truth_support.float())
            target_crop = masked_crop(target, truth_support.float())
            predicted_depth = predicted.depth / predicted.alpha.clamp_min(1e-6)
            truth_depth = truth.depth / truth.alpha.clamp_min(1e-6)
            values = {
                "view": index,
                "psnr_fg": masked_psnr(color, target, truth_support.float()),
                "psnr_full": psnr(color, target),
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
    return {
        "per_view": per_view,
        "mean": {
            key: statistics.fmean(float(view[key]) for view in per_view) for key in metric_names
        },
        "visibility_margin_sigma": margin_sigma,
    }


def verify_synthetic_target_margin_invariance(scene: SceneData) -> dict[str, Any]:
    """Prove the frozen synthetic targets do not privilege the current margin."""
    current_renderer = TorchRasterizer(visibility_margin_sigma=CURRENT_MARGIN_SIGMA)
    safe_renderer = TorchRasterizer(visibility_margin_sigma=SUPPORT_SAFE_MARGIN_SIGMA)
    hashes = []
    if scene.gt_depths is None or len(scene.gt_depths) != 12:
        raise AssertionError("synthetic target invariant requires twelve stored GT depths")
    with torch.no_grad():
        for index, (camera, target, target_depth) in enumerate(
            zip(scene.cameras, scene.images, scene.gt_depths)
        ):
            current = current_renderer.render(scene.gt_gaussians, camera)
            safe = safe_renderer.render(scene.gt_gaussians, camera)
            maximum_errors = {}
            for field in ("color", "alpha", "depth"):
                maximum_errors[f"current_safe_{field}"] = float(
                    (getattr(current, field) - getattr(safe, field)).abs().max()
                )
                if not torch.equal(getattr(current, field), getattr(safe, field)):
                    raise AssertionError(
                        f"synthetic GT {field} changes with visibility margin in view {index}"
                    )
            generated_color = current.color.clamp(0.0, 1.0)
            generated_depth = torch.where(
                current.alpha > 0.05,
                current.depth / current.alpha.clamp_min(1e-6),
                0.0,
            )
            maximum_errors["stored_color"] = float((generated_color - target).abs().max())
            maximum_errors["stored_depth"] = float((generated_depth - target_depth).abs().max())
            if not torch.equal(generated_color, target):
                raise AssertionError(
                    f"synthetic target is not the exact current-margin GT render in view {index}"
                )
            if not torch.equal(generated_depth, target_depth):
                raise AssertionError(
                    f"synthetic depth is not the exact current-margin GT render in view {index}"
                )
            hashes.append(
                {
                    "view": index,
                    "target_sha256": support_bench.tensor_collection_hash([("target", target)]),
                    "target_depth_sha256": support_bench.tensor_collection_hash(
                        [("target_depth", target_depth)]
                    ),
                    "render_sha256": _render_hash(current),
                    "maximum_absolute_errors": maximum_errors,
                }
            )
    if len(hashes) != 12:
        raise AssertionError("synthetic target invariant did not cover all twelve views")
    return {"all_twelve_views_bit_exact": True, "views": hashes}


def verify_default_semantics() -> dict[str, Any]:
    from rtgs.render.base import DEFAULT_VISIBILITY_MARGIN_SIGMA

    renderer = TorchRasterizer()
    config = TrainConfig()
    if DEFAULT_VISIBILITY_MARGIN_SIGMA != CURRENT_MARGIN_SIGMA:
        raise AssertionError("repository default visibility margin differs from 3.0")
    if renderer.visibility_margin_sigma != CURRENT_MARGIN_SIGMA:
        raise AssertionError("Torch renderer default visibility margin differs from 3.0")
    if config.visibility_margin_sigma != CURRENT_MARGIN_SIGMA:
        raise AssertionError("trainer default visibility margin differs from 3.0")
    incorporated_hash = sha256_file(ROOT / INCORPORATED_PREREGISTRATION)
    if incorporated_hash != INCORPORATED_PREREGISTRATION_SHA256:
        raise AssertionError("incorporated support protocol differs from its frozen hash")
    return {
        "default_visibility_margin_sigma": CURRENT_MARGIN_SIGMA,
        "support_safe_visibility_margin_sigma": SUPPORT_SAFE_MARGIN_SIGMA,
        "kernel_support_cutoff": KERNEL_SUPPORT_CUTOFF,
        "incorporated_preregistration_sha256": incorporated_hash,
    }


def verify_retry_provenance() -> dict[str, Any]:
    expected_hashes = {
        str(PREVIOUS_PREREGISTRATION): PREVIOUS_PREREGISTRATION_SHA256,
        str(PREVIOUS_SEAL): PREVIOUS_SEAL_SHA256,
        str(PREVIOUS_ATTEMPT): PREVIOUS_ATTEMPT_SHA256,
    }
    actual_hashes = {path: sha256_file(ROOT / Path(path)) for path in expected_hashes}
    if actual_hashes != expected_hashes:
        raise RuntimeError(
            f"visibility-margin retry provenance differs: {actual_hashes} != {expected_hashes}"
        )
    unexpected_outputs = [
        str(path) for path in (FAILED_PHASE_A_OUTPUT, FAILED_PHASE_A_NOTE) if (ROOT / path).exists()
    ]
    if unexpected_outputs:
        raise RuntimeError(f"failed first-attempt outputs unexpectedly exist: {unexpected_outputs}")
    return {
        "incorporated_hashes": actual_hashes,
        "failed_phase_a_output": str(FAILED_PHASE_A_OUTPUT),
        "failed_phase_a_note": str(FAILED_PHASE_A_NOTE),
        "failed_outputs_absent": True,
    }


def strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {value!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


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
    current_environment = environment_metadata()
    assert_official_environment(current_environment)
    retry_provenance = verify_retry_provenance()
    verification = run_verification()
    hashes, aggregate = source_hashes()
    return {
        "artifact_type": "visibility_margin_iter2_implementation_seal",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_metadata(),
        "sealed_paths": [str(path) for path in SEALED_PATHS],
        "source_hashes": hashes,
        "source_aggregate": aggregate,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": sha256_file(ROOT / PREREGISTRATION),
        },
        "incorporated_preregistration": {
            "path": str(INCORPORATED_PREREGISTRATION),
            "sha256": sha256_file(ROOT / INCORPORATED_PREREGISTRATION),
        },
        "verification": verification,
        "default_semantics": verify_default_semantics(),
        "retry_provenance": retry_provenance,
        "environment": current_environment,
        "command": [sys.executable, *sys.argv],
    }


def load_and_verify_seal(path: Path) -> dict[str, Any]:
    payload = strict_json_load(path)
    if payload.get("artifact_type") != "visibility_margin_iter2_implementation_seal":
        raise ValueError(f"{path} is not a visibility-margin implementation seal")
    expected_paths = [str(item) for item in SEALED_PATHS]
    if payload.get("sealed_paths") != expected_paths:
        raise RuntimeError("implementation seal path set differs from the frozen repository set")
    paths = tuple(Path(item) for item in payload["sealed_paths"])
    hashes, aggregate = source_hashes(paths)
    if hashes != payload.get("source_hashes") or aggregate != payload.get("source_aggregate"):
        raise RuntimeError("implementation/protocol differs from the sealed source aggregate")
    if not payload.get("verification", {}).get("passed"):
        raise RuntimeError("implementation seal does not contain passing verification")
    if payload.get("retry_provenance") != verify_retry_provenance():
        raise RuntimeError("implementation seal retry provenance differs from current state")
    current_environment = environment_metadata()
    assert_official_environment(current_environment)
    if environment_fingerprint(payload["environment"]) != environment_fingerprint(
        current_environment
    ):
        raise RuntimeError("current execution environment differs from implementation seal")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "source_aggregate": aggregate,
        "verification_sha256": canonical_json_hash(payload["verification"]),
        "environment_fingerprint": environment_fingerprint(payload["environment"]),
    }


def verify_loaded_sources_against_seal(seal_path: Path) -> tuple[dict[str, str], str]:
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
            "loaded repository sources are outside/different from seal: "
            f"unexpected={unexpected}, mismatched={mismatched}"
        )
    return loaded_hashes, loaded_aggregate


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
            target_scene, _ = support_bench.make_condition_scene(seed, condition)
            target_margin_invariance = verify_synthetic_target_margin_invariance(target_scene)
            scene, _, initialization, preparation = prepare_seed(seed, condition)
            if support_bench.scene_hashes(target_scene) != preparation["scene_hashes"]:
                raise AssertionError(
                    f"pre-fit target scene differs from prepared scene for {condition}/{seed}"
                )
            initial_incidence = audit_checkpoint(scene, initialization.with_sh_degree(3))
            config = train_config(seed, CURRENT_MARGIN_SIGMA)
            started = time.perf_counter()
            final, history = Trainer(config).train(
                scene.subset(TRAIN_INDICES), initialization.detach()
            )
            training_seconds = time.perf_counter() - started
            assert_finite_gaussians(final, f"current-margin audit {condition}/{seed}")
            map_history_to_global_views(history)
            assert_finite_tree(history, f"current-margin history {condition}/{seed}")
            if len(history["sampled_train_views"]) != 120:
                raise AssertionError("current-margin audit did not record the complete schedule")
            if final.n != initialization.n:
                raise AssertionError(f"current-margin audit changed count for {condition}/{seed}")
            if [int(item[0]) for item in history["psnr"]] != [30, 60, 90, 120]:
                raise AssertionError(f"checkpoint schedule differs for {condition}/{seed}")
            final_incidence = audit_checkpoint(scene, final)
            runs.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "preparation": preparation,
                    "target_margin_invariance": target_margin_invariance,
                    "train_config": asdict(config),
                    "training_seconds": training_seconds,
                    "initial_incidence": initial_incidence,
                    "final_incidence": final_incidence,
                    "final_hash": support_bench.gaussians_hash(final),
                    "final_gaussians": final.n,
                    "current_metrics": evaluate_final(
                        scene, final, margin_sigma=CURRENT_MARGIN_SIGMA
                    ),
                    "forward_safe_metrics": evaluate_final(
                        scene, final, margin_sigma=SUPPORT_SAFE_MARGIN_SIGMA
                    ),
                    "history": history,
                    "schedule_hash": canonical_json_hash(history["sampled_train_views"]),
                }
            )
    decision = audit_decision(runs)
    loaded_hashes, loaded_aggregate = verify_loaded_sources_against_seal(seal_path)
    if load_and_verify_seal(seal_path) != seal:
        raise RuntimeError("implementation seal changed during Phase A")
    return {
        "artifact_type": "visibility_margin_iter2_phase_a_audit",
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


def _recompute_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return aggregate_checkpoint_views(checkpoint.get("views", []))


def _validate_target_invariance_record(record: dict[str, Any]) -> None:
    views = record.get("views")
    if record.get("all_twelve_views_bit_exact") is not True or not isinstance(views, list):
        raise RuntimeError("target-margin invariant is missing or did not pass")
    if [view.get("view") for view in views] != list(range(12)):
        raise RuntimeError("target-margin invariant does not cover all twelve ordered views")
    expected_errors = {
        "current_safe_color",
        "current_safe_alpha",
        "current_safe_depth",
        "stored_color",
        "stored_depth",
    }
    for view in views:
        errors = view.get("maximum_absolute_errors")
        if not isinstance(errors, dict) or set(errors) != expected_errors:
            raise RuntimeError("target-margin invariant error fields are incomplete")
        if any(float(value) != 0.0 for value in errors.values()):
            raise RuntimeError("target-margin invariant contains a nonzero exact-equality error")
        for field in ("target_sha256", "target_depth_sha256", "render_sha256"):
            digest = view.get(field)
            if not isinstance(digest, str) or len(digest) != 64:
                raise RuntimeError("target-margin invariant contains an invalid hash")


def validate_phase_a_audit(audit: dict[str, Any], seal: dict[str, Any]) -> None:
    if audit.get("artifact_type") != "visibility_margin_iter2_phase_a_audit":
        raise ValueError("Phase-B input is not a visibility-margin Phase-A audit")
    if audit.get("split") != {"train": TRAIN_INDICES, "held_out": TEST_INDICES}:
        raise RuntimeError("Phase-A split differs from preregistration")
    if audit.get("seal") != seal:
        raise RuntimeError("Phase-A seal binding differs from current seal")
    expected_prereg = {
        "path": str(PREREGISTRATION),
        "sha256": sha256_file(ROOT / PREREGISTRATION),
    }
    if audit.get("preregistration") != expected_prereg:
        raise RuntimeError("Phase-A preregistration binding differs")
    if environment_fingerprint(audit["environment"]) != seal["environment_fingerprint"]:
        raise RuntimeError("Phase-A environment differs from implementation seal")
    runs = audit.get("runs")
    if not isinstance(runs, list) or len(runs) != len(CONDITIONS) * len(SEEDS):
        raise RuntimeError("Phase-A artifact has an invalid run count")
    expected_ids = [(condition, seed) for condition in CONDITIONS for seed in SEEDS]
    if [(run.get("condition"), run.get("seed")) for run in runs] != expected_ids:
        raise RuntimeError("Phase-A artifact has missing, duplicated, or reordered runs")
    for run in runs:
        seed = int(run["seed"])
        if run.get("train_config") != asdict(train_config(seed, CURRENT_MARGIN_SIGMA)):
            raise RuntimeError(f"Phase-A config differs for {run['condition']}/{seed}")
        schedule = run.get("history", {}).get("sampled_train_views")
        if not isinstance(schedule, list) or len(schedule) != 120:
            raise RuntimeError(f"Phase-A schedule is invalid for {run['condition']}/{seed}")
        if canonical_json_hash(schedule) != run.get("schedule_hash"):
            raise RuntimeError(f"Phase-A schedule hash differs for {run['condition']}/{seed}")
        try:
            _validate_target_invariance_record(run.get("target_margin_invariance", {}))
        except RuntimeError as error:
            raise RuntimeError(
                f"Phase-A target-margin invariant is invalid for {run['condition']}/{seed}: {error}"
            ) from error
        for checkpoint_name in ("initial_incidence", "final_incidence"):
            checkpoint = run.get(checkpoint_name, {})
            recomputed = _recompute_checkpoint(checkpoint)
            if canonical_json_hash(recomputed) != canonical_json_hash(checkpoint.get("summary")):
                raise RuntimeError(
                    f"Phase-A {checkpoint_name} summary differs for {run['condition']}/{seed}"
                )
    recomputed_decision = audit_decision(runs)
    if canonical_json_hash(recomputed_decision) != canonical_json_hash(audit.get("decision")):
        raise RuntimeError("Phase-A decision does not match recomputed frozen gate")


def verify_phase_a_review(path: Path, *, audit_path: Path, seal: dict[str, Any]) -> dict[str, str]:
    review = strict_json_load(path)
    expected = {
        "artifact_type": "visibility_margin_iter2_phase_a_scientist_review",
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


METRICS = (
    "psnr_fg",
    "psnr_full",
    "psnr_crop",
    "ssim_crop",
    "depth_rmse_over_extent",
    "alpha_iou",
    "foreground_coverage",
)


def summarize_ablation(
    audit: dict[str, Any], candidate_runs: list[dict[str, Any]]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        summary[condition] = {}
        baseline = [run for run in audit["runs"] if run["condition"] == condition]
        candidates = [run for run in candidate_runs if run["condition"] == condition]
        if [int(run["seed"]) for run in baseline] != SEEDS or [
            int(run["seed"]) for run in candidates
        ] != SEEDS:
            raise AssertionError(f"summary run identities differ for {condition}")
        for arm, runs, metric_key in (
            ("current", baseline, "current_metrics"),
            ("current_forward_safe", baseline, "forward_safe_metrics"),
            (CANDIDATE_ARM, candidates, "matched_metrics"),
            ("support_safe_common_current", candidates, "common_current_metrics"),
        ):
            summary[condition][arm] = {}
            for metric in METRICS:
                samples = [float(run[metric_key]["mean"][metric]) for run in runs]
                if len(samples) != len(SEEDS) or not all(math.isfinite(x) for x in samples):
                    raise AssertionError(f"invalid samples for {condition}/{arm}/{metric}")
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

    hard_psnr = samples("diffuse", "current", "psnr_fg")
    safe_psnr = samples("diffuse", CANDIDATE_ARM, "psnr_fg")
    gains = [candidate - baseline for candidate, baseline in zip(safe_psnr, hard_psnr)]
    common_psnr = samples("diffuse", "support_safe_common_current", "psnr_fg")
    common_gains = [candidate - baseline for candidate, baseline in zip(common_psnr, hard_psnr)]
    forward_psnr = samples("diffuse", "current_forward_safe", "psnr_fg")
    forward_gains = [candidate - baseline for candidate, baseline in zip(forward_psnr, hard_psnr)]
    hard_ssim = samples("diffuse", "current", "ssim_crop")
    safe_ssim = samples("diffuse", CANDIDATE_ARM, "ssim_crop")
    ssim_deltas = [candidate - baseline for candidate, baseline in zip(safe_ssim, hard_ssim)]
    hard_depth = mean("diffuse", "current", "depth_rmse_over_extent")
    safe_depth = mean("diffuse", CANDIDATE_ARM, "depth_rmse_over_extent")
    if hard_depth <= 0.0:
        raise AssertionError("current depth RMSE denominator is zero")
    depth_regression = (safe_depth - hard_depth) / hard_depth
    alpha_delta = mean("diffuse", CANDIDATE_ARM, "alpha_iou") - mean(
        "diffuse", "current", "alpha_iou"
    )
    coverage_delta = mean("diffuse", CANDIDATE_ARM, "foreground_coverage") - mean(
        "diffuse", "current", "foreground_coverage"
    )
    view_hard = samples("view_dependent", "current", "psnr_fg")
    view_safe = samples("view_dependent", CANDIDATE_ARM, "psnr_fg")
    view_deltas = [candidate - baseline for candidate, baseline in zip(view_safe, view_hard)]
    view_common = samples("view_dependent", "support_safe_common_current", "psnr_fg")
    view_common_deltas = [
        candidate - baseline for candidate, baseline in zip(view_common, view_hard)
    ]
    view_forward = samples("view_dependent", "current_forward_safe", "psnr_fg")
    view_forward_deltas = [
        candidate - baseline for candidate, baseline in zip(view_forward, view_hard)
    ]
    criteria = {
        "mean_psnr_gain_at_least_0_10_db": statistics.fmean(gains) >= 0.10,
        "psnr_wins_at_least_two_seeds": sum(gain > 0.0 for gain in gains) >= 2,
        "mean_ssim_regression_within_0_002": statistics.fmean(ssim_deltas) >= -0.002,
        "per_seed_ssim_regression_within_0_005": min(ssim_deltas) >= -0.005,
        "depth_rmse_regression_within_2_percent": depth_regression <= 0.02,
        "alpha_iou_regression_within_0_02": alpha_delta >= -0.02,
        "coverage_regression_within_0_02": coverage_delta >= -0.02,
        "diffuse_common_current_mean_psnr_regression_within_0_10_db": statistics.fmean(common_gains)
        >= -0.10,
        "diffuse_common_current_per_seed_psnr_regression_within_0_25_db": min(common_gains)
        >= -0.25,
        "view_dependent_mean_psnr_regression_within_0_10_db": statistics.fmean(view_deltas)
        >= -0.10,
        "view_dependent_per_seed_psnr_regression_within_0_25_db": min(view_deltas) >= -0.25,
    }
    return {
        "criteria": criteria,
        "psnr_gains_db": gains,
        "mean_psnr_gain_db": statistics.fmean(gains),
        "psnr_seed_wins": sum(gain > 0.0 for gain in gains),
        "common_current_psnr_gains_db": common_gains,
        "common_current_mean_psnr_gain_db": statistics.fmean(common_gains),
        "forward_only_psnr_gains_db": forward_gains,
        "forward_only_mean_psnr_gain_db": statistics.fmean(forward_gains),
        "ssim_deltas": ssim_deltas,
        "depth_rmse_regression_fraction": depth_regression,
        "alpha_iou_delta": alpha_delta,
        "foreground_coverage_delta": coverage_delta,
        "view_dependent_psnr_deltas_db": view_deltas,
        "view_dependent_common_current_psnr_deltas_db": view_common_deltas,
        "view_dependent_forward_only_psnr_deltas_db": view_forward_deltas,
        "primary_hypothesis_pass": all(criteria.values()),
    }


def _step0_invariants(initialization: Gaussians3D, scene: SceneData) -> dict[str, Any]:
    gaussians = initialization.with_sh_degree(3)
    current = TorchRasterizer(visibility_margin_sigma=CURRENT_MARGIN_SIGMA)
    explicit = TorchRasterizer()
    safe = TorchRasterizer(visibility_margin_sigma=SUPPORT_SAFE_MARGIN_SIGMA)
    maxima = {"color": 0.0, "alpha": 0.0, "depth": 0.0}
    for index in TRAIN_INDICES:
        default_output = explicit.render(gaussians, scene.cameras[index])
        current_output = current.render(gaussians, scene.cameras[index])
        safe_output = safe.render(gaussians, scene.cameras[index])
        for field in maxima:
            if not torch.equal(getattr(default_output, field), getattr(current_output, field)):
                raise AssertionError(f"explicit current {field} differs from default at step zero")
            maxima[field] = max(
                maxima[field],
                float((getattr(safe_output, field) - getattr(current_output, field)).abs().max()),
            )
    return {
        "default_current_bit_exact": True,
        "maximum_support_safe_color_difference": maxima["color"],
        "maximum_support_safe_alpha_difference": maxima["alpha"],
        "maximum_support_safe_depth_difference": maxima["depth"],
    }


def run_ablation(
    audit_path: Path, seal_path: Path, review_path: Path, attempt_output: Path
) -> dict[str, Any]:
    preflight_output(attempt_output)
    seal = load_and_verify_seal(seal_path)
    audit = strict_json_load(audit_path)
    validate_phase_a_audit(audit, seal)
    review = verify_phase_a_review(review_path, audit_path=audit_path, seal=seal)
    if not audit["decision"]["phase_b_authorized"]:
        raise RuntimeError("Phase A did not authorize support-safe training")
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
    baseline_lookup = {(run["condition"], int(run["seed"])): run for run in audit["runs"]}
    candidate_runs = []
    experiment_started = time.perf_counter()
    for condition in CONDITIONS:
        for seed in SEEDS:
            print(f"ablation: recreating {condition} seed {seed}", flush=True)
            target_scene, _ = support_bench.make_condition_scene(seed, condition)
            target_margin_invariance = verify_synthetic_target_margin_invariance(target_scene)
            scene, _, initialization, preparation = prepare_seed(seed, condition)
            if support_bench.scene_hashes(target_scene) != preparation["scene_hashes"]:
                raise AssertionError(
                    f"pre-fit target scene differs from prepared scene for {condition}/{seed}"
                )
            baseline = baseline_lookup[(condition, seed)]
            if canonical_json_hash(target_margin_invariance) != canonical_json_hash(
                baseline["target_margin_invariance"]
            ):
                raise AssertionError(
                    f"target-margin invariant differs on recreation for {condition}/{seed}"
                )
            for field in ("scene_hashes", "fitted_hash", "initialization_hash"):
                if preparation[field] != baseline["preparation"][field]:
                    raise AssertionError(
                        f"Phase-B recreation differs in {condition}/{seed}/{field}"
                    )
            config = train_config(seed, SUPPORT_SAFE_MARGIN_SIGMA)
            expected = dict(baseline["train_config"])
            expected["visibility_margin_sigma"] = SUPPORT_SAFE_MARGIN_SIGMA
            if asdict(config) != expected:
                raise AssertionError("candidate config differs beyond visibility margin")
            invariants = _step0_invariants(initialization, scene)
            started = time.perf_counter()
            final, history = Trainer(config).train(
                scene.subset(TRAIN_INDICES), initialization.detach()
            )
            training_seconds = time.perf_counter() - started
            assert_finite_gaussians(final, f"support-safe candidate {condition}/{seed}")
            map_history_to_global_views(history)
            assert_finite_tree(history, f"support-safe history {condition}/{seed}")
            if history["sampled_train_views"] != baseline["history"]["sampled_train_views"]:
                raise AssertionError(f"target-view schedule differs for {condition}/{seed}")
            for history_field in ("active_sh_degree", "n_gaussians"):
                if canonical_json_hash(history[history_field]) != canonical_json_hash(
                    baseline["history"][history_field]
                ):
                    raise AssertionError(f"{history_field} schedule differs for {condition}/{seed}")
            if final.n != initialization.n or final.n != baseline["final_gaussians"]:
                raise AssertionError(f"primitive count differs for {condition}/{seed}")
            if [int(item[0]) for item in history["psnr"]] != [30, 60, 90, 120]:
                raise AssertionError(f"checkpoint schedule differs for {condition}/{seed}")
            candidate_runs.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "arm": CANDIDATE_ARM,
                    "preparation_hashes_verified": True,
                    "target_margin_invariance": target_margin_invariance,
                    "step0_invariants": invariants,
                    "train_config": asdict(config),
                    "training_seconds": training_seconds,
                    "final_hash": support_bench.gaussians_hash(final),
                    "final_gaussians": final.n,
                    "common_current_metrics": evaluate_final(
                        scene, final, margin_sigma=CURRENT_MARGIN_SIGMA
                    ),
                    "matched_metrics": evaluate_final(
                        scene, final, margin_sigma=SUPPORT_SAFE_MARGIN_SIGMA
                    ),
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
        "artifact_type": "visibility_margin_iter2_phase_b_ablation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_metadata(),
        "seal": seal,
        "phase_a": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        "phase_a_scientist_review": review,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": sha256_file(ROOT / PREREGISTRATION),
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


def claim_attempt(path: Path, *, phase: str, output: Path, inputs: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_type": "visibility_margin_once_only_attempt",
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
    if artifact_type == "visibility_margin_iter2_phase_a_audit":
        decision = payload["decision"]
        disposition = (
            "Phase B remains blocked until an independent scientist review binds this exact "
            "passing audit and grants execution clearance."
            if decision["phase_b_authorized"]
            else "The frozen gate failed, so Phase B is forbidden under this protocol."
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
                "This is a CPU synthetic final-state mechanism audit, not real-scene, CUDA, "
                f"speed, density-enabled, or default-change evidence. {disposition}",
            ]
        )
    elif artifact_type == "visibility_margin_iter2_phase_b_ablation":
        decision = payload["decision"]
        lines.extend(
            [
                "",
                "## Frozen outcome decision",
                "",
                f"- Primary hypothesis pass: `{decision['primary_hypothesis_pass']}`",
                f"- Mean matched-by-arm foreground PSNR gain: "
                f"`{decision['mean_psnr_gain_db']:.6f} dB`",
                "",
                "Matched-by-arm rendering defines the primary total-effect evaluation; the "
                "off-diagonal renders are attribution controls. This result is limited to "
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
    audit = subparsers.add_parser("audit", help="run current-margin incidence audit")
    audit.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    audit.add_argument("--output", type=Path, required=True)
    ablate = subparsers.add_parser("ablate", help="run support-safe arm after authorization")
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
        "audit": "_cpu_visibility_margin_iter2_audit.json",
        "ablate": "_cpu_visibility_margin_iter2_ablation.json",
    }
    if args.command_name in expected_suffix and not args.output.name.endswith(
        expected_suffix[args.command_name]
    ):
        raise ValueError(
            f"official {args.command_name} output must end with "
            f"{expected_suffix[args.command_name]!r}; retries require a new preregistered namespace"
        )
    if args.command_name == "seal" and args.output != DEFAULT_SEAL:
        raise ValueError("official seal must use the preregistered fixed path")
    preflight_output(args.output)
    if args.command_name == "seal":
        payload = create_seal()
    elif args.command_name == "audit":
        payload = run_audit(args.seal, args.output)
    elif args.command_name == "ablate":
        payload = run_ablation(args.audit, args.seal, args.phase_a_review, args.output)
    else:  # pragma: no cover
        raise AssertionError(args.command_name)
    note, digest = write_artifact(args.output, payload)
    print(f"saved {args.output} (sha256={digest})", flush=True)
    print(f"saved {note}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
