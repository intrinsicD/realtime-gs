#!/usr/bin/env python3
"""Mechanism screen for covariance estimates derived from beam-fusion correspondences.

The control is the unchanged covariance-intersection (CI) Beam Fusion result.  The two treatment
arms preserve every other initialized field exactly:

``track-lsq``
    Solve the Splat-SfM covariance projection equations on Beam Fusion's own contributor tracks,
    then project the symmetric result to a bounded SPD covariance.

``track-robust``
    Start from ``track-lsq`` and optimize a Cholesky-factorized SPD covariance against a robust,
    observation-whitened covariance reprojection loss.  Means, track assignments, colors, opacity,
    and Gaussian count remain frozen during this covariance-only fit.

All three initializations are then passed to the same 1,000-step, fixed-topology CPU 3DGS
refinement used by ``beam_convergence_dynamics.py``.  This is an all-fitted-view, downscale-32,
single-scene development diagnostic.  It is not held-out or production-default evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.compact_views import CompactDataset
from rtgs.lift.beam_fusion import (
    BeamFusionConfig,
    BeamFusionResult,
    _component_covariances_2d,
    fuse_gaussian_beams,
)
from rtgs.lift.splat_sfm import _spd_project, _triangulate_covariances

try:
    from benchmarks.beam_convergence_dynamics import (
        DOWNSCALE,
        EVAL_VIEWS,
        ITERATIONS,
        N_INIT,
        SEED,
        SELECTED_VIEWS,
        build_scene,
        run_arm,
        save_preview,
        selected_inputs,
    )
except ModuleNotFoundError:  # direct ``python benchmarks/beam_covariance_refit.py``
    from beam_convergence_dynamics import (  # type: ignore[no-redef]
        DOWNSCALE,
        EVAL_VIEWS,
        ITERATIONS,
        N_INIT,
        SEED,
        SELECTED_VIEWS,
        build_scene,
        run_arm,
        save_preview,
        selected_inputs,
    )

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d"
DEFAULT_OUT = ROOT / "runs/beam_covariance_refit_20260723"
DEFAULT_PROTOCOL = ROOT / "benchmarks/results/20260723_beam_covariance_refit_PREREG.md"

ROBUST_STEPS = 120
ROBUST_LR = 0.03
ROBUST_HUBER_DELTA = 0.25
ROBUST_PRIOR_WEIGHT = 1e-3
MIN_SIGMA_WORLD = 1e-4
MAX_SIGMA_EXTENT_FRACTION = 0.5

ARMS = ("ci", "track-lsq", "track-robust")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _projection_jacobians(
    points_world: torch.Tensor,
    cameras,
) -> torch.Tensor:
    """Return world-to-pixel covariance Jacobians with shape ``(T,V,2,3)``."""
    count = points_world.shape[0]
    jacobians = torch.zeros(
        count,
        len(cameras),
        2,
        3,
        dtype=torch.float64,
        device=points_world.device,
    )
    points = points_world.to(torch.float64)
    for view_index, camera in enumerate(cameras):
        cam_points = camera.world_to_cam(points)
        z = cam_points[:, 2].clamp_min(1e-8)
        j_cam = torch.zeros(count, 2, 3, dtype=torch.float64, device=points.device)
        j_cam[:, 0, 0] = camera.fx / z
        j_cam[:, 0, 2] = -camera.fx * cam_points[:, 0] / z.square()
        j_cam[:, 1, 1] = camera.fy / z
        j_cam[:, 1, 2] = -camera.fy * cam_points[:, 1] / z.square()
        jacobians[:, view_index] = j_cam @ camera.R.to(torch.float64)
    return jacobians


def _member_observations(
    result: BeamFusionResult,
    inputs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand Beam Fusion's CSR lineage into a masked ``(component, view)`` covariance table."""
    count = result.n_components
    n_views = inputs.n_views
    member_mask = torch.zeros(count, n_views, dtype=torch.bool)
    member_cov2d = torch.zeros(count, n_views, 2, 2, dtype=torch.float64)
    per_view_covariances = [
        _component_covariances_2d(observation) for observation in inputs.observations
    ]
    offsets = result.component_offsets.tolist()
    for component in range(count):
        start, end = offsets[component], offsets[component + 1]
        views = result.contributor_view_indices[start:end]
        splats = result.contributor_component_indices[start:end]
        if views.unique().numel() != views.numel():
            raise RuntimeError(f"beam component {component} contains duplicate contributor views")
        member_mask[component, views] = True
        for view, splat in zip(views.tolist(), splats.tolist(), strict=True):
            member_cov2d[component, view] = per_view_covariances[view][splat]
    if int(member_mask.sum()) != int(result.contributor_view_indices.numel()):
        raise RuntimeError("CSR contributor expansion changed the number of observation links")
    return member_mask, member_cov2d


def _pack_cholesky(covariances: torch.Tensor) -> torch.Tensor:
    lower = torch.linalg.cholesky(covariances.to(torch.float64))
    return torch.stack(
        [
            lower[:, 0, 0].log(),
            lower[:, 1, 0],
            lower[:, 1, 1].log(),
            lower[:, 2, 0],
            lower[:, 2, 1],
            lower[:, 2, 2].log(),
        ],
        dim=-1,
    )


def _unpack_cholesky(parameters: torch.Tensor) -> torch.Tensor:
    count = parameters.shape[0]
    lower = torch.zeros(count, 3, 3, dtype=parameters.dtype, device=parameters.device)
    lower[:, 0, 0] = parameters[:, 0].exp()
    lower[:, 1, 0] = parameters[:, 1]
    lower[:, 1, 1] = parameters[:, 2].exp()
    lower[:, 2, 0] = parameters[:, 3]
    lower[:, 2, 1] = parameters[:, 4]
    lower[:, 2, 2] = parameters[:, 5].exp()
    return lower @ lower.transpose(-1, -2)


def _project_covariances(
    jacobians: torch.Tensor,
    covariances: torch.Tensor,
) -> torch.Tensor:
    return jacobians @ covariances[:, None] @ jacobians.transpose(-1, -2)


def relative_reprojection_residuals(
    jacobians: torch.Tensor,
    covariances: torch.Tensor,
    member_mask: torch.Tensor,
    observed_covariances: torch.Tensor,
) -> torch.Tensor:
    """Per-observation relative Frobenius residuals, flattened over valid track members."""
    predicted = _project_covariances(jacobians, covariances.to(torch.float64))
    numerator = (predicted - observed_covariances).norm(dim=(-2, -1))
    denominator = observed_covariances.norm(dim=(-2, -1)).clamp_min(1e-12)
    return (numerator / denominator)[member_mask]


def whitened_reprojection_residuals(
    jacobians: torch.Tensor,
    covariances: torch.Tensor,
    member_mask: torch.Tensor,
    observed_covariances: torch.Tensor,
) -> torch.Tensor:
    """Per-observation RMS residual after whitening by the measured 2D covariance."""
    observed = observed_covariances.to(torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(observed)
    inverse_sqrt = (
        eigenvectors
        @ torch.diag_embed(eigenvalues.clamp_min(1e-12).rsqrt())
        @ eigenvectors.transpose(-1, -2)
    )
    predicted = _project_covariances(jacobians, covariances.to(torch.float64))
    normalized = inverse_sqrt @ predicted @ inverse_sqrt
    identity = torch.eye(2, dtype=torch.float64, device=jacobians.device)
    return ((normalized - identity).square().mean(dim=(-2, -1)).sqrt())[member_mask]


def robust_refit_from_jacobians(
    jacobians: torch.Tensor,
    member_mask: torch.Tensor,
    observed_covariances: torch.Tensor,
    initial_covariances: torch.Tensor,
    prior_covariances: torch.Tensor,
    *,
    steps: int = ROBUST_STEPS,
    learning_rate: float = ROBUST_LR,
    huber_delta: float = ROBUST_HUBER_DELTA,
    prior_weight: float = ROBUST_PRIOR_WEIGHT,
    min_sigma: float = MIN_SIGMA_WORLD,
    max_sigma: float,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Robustly fit SPD 3D covariances while keeping all geometry and assignments fixed."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0 or huber_delta <= 0 or prior_weight < 0:
        raise ValueError("invalid robust covariance optimizer parameters")
    parameters = torch.nn.Parameter(_pack_cholesky(initial_covariances).clone())
    optimizer = torch.optim.Adam([parameters], lr=learning_rate)
    observed = observed_covariances.to(torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(observed)
    inverse_sqrt = (
        eigenvectors
        @ torch.diag_embed(eigenvalues.clamp_min(1e-12).rsqrt())
        @ eigenvectors.transpose(-1, -2)
    )
    identity = torch.eye(2, dtype=torch.float64, device=jacobians.device)
    prior = prior_covariances.to(torch.float64)
    prior_norm = prior.norm(dim=(-2, -1)).clamp_min(1e-12)
    min_variance = min_sigma**2
    max_variance = max_sigma**2
    history: list[dict[str, float]] = []

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        covariances = _unpack_cholesky(parameters)
        predicted = _project_covariances(jacobians, covariances)
        normalized = inverse_sqrt @ predicted @ inverse_sqrt
        residual = (normalized - identity).square().mean(dim=(-2, -1)).sqrt()
        valid = residual[member_mask]
        huber = torch.where(
            valid <= huber_delta,
            0.5 * valid.square(),
            huber_delta * (valid - 0.5 * huber_delta),
        )
        data_loss = huber.mean()
        prior_loss = (
            ((covariances - prior) / prior_norm[:, None, None]).square().mean(dim=(-2, -1))
        ).mean()
        covariance_eigenvalues = torch.linalg.eigvalsh(covariances)
        bound_loss = (
            torch.relu(min_variance - covariance_eigenvalues).square().mean()
            + torch.relu(covariance_eigenvalues - max_variance).square().mean()
        ) / max(max_variance**2, 1e-24)
        loss = data_loss + prior_weight * prior_loss + 1e-3 * bound_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([parameters], max_norm=10.0)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "data_loss": float(data_loss.detach()),
                    "prior_loss": float(prior_loss.detach()),
                    "valid_whitened_residual_median": float(valid.detach().median()),
                    "valid_whitened_residual_mean": float(valid.detach().mean()),
                }
            )

    fitted = _unpack_cholesky(parameters.detach())
    return _spd_project(fitted, min_sigma=min_sigma, max_sigma=max_sigma), history


def _same_non_covariance_fields(first: Gaussians3D, second: Gaussians3D) -> bool:
    return (
        torch.equal(first.means, second.means)
        and torch.equal(first.opacity, second.opacity)
        and torch.equal(first.sh, second.sh)
        and first.n == second.n
    )


def _replace_covariances(base: Gaussians3D, covariances: torch.Tensor) -> Gaussians3D:
    converted = Gaussians3D.from_means_covs(
        means=base.means,
        covs=covariances.to(base.means),
        colors=torch.zeros(base.n, 3, dtype=base.means.dtype, device=base.means.device),
        opacity=base.opacity,
        sh_degree=base.sh_degree,
        min_scale=MIN_SIGMA_WORLD,
    )
    result = Gaussians3D(
        means=base.means.detach().clone(),
        quats=converted.quats.detach().clone(),
        log_scales=converted.log_scales.detach().clone(),
        opacity=base.opacity.detach().clone(),
        sh=base.sh.detach().clone(),
    )
    if not _same_non_covariance_fields(base, result):
        raise RuntimeError("covariance replacement changed a frozen initialization field")
    return result


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().to(torch.float64)
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p90": float(values.quantile(0.90)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def _covariance_diagnostics(
    covariance: torch.Tensor,
    jacobians: torch.Tensor,
    member_mask: torch.Tensor,
    observed_covariances: torch.Tensor,
) -> dict:
    eigenvalues = torch.linalg.eigvalsh(covariance.to(torch.float64)).clamp_min(0)
    sigmas = eigenvalues.sqrt()
    condition = eigenvalues[:, -1] / eigenvalues[:, 0].clamp_min(1e-24)
    relative = relative_reprojection_residuals(
        jacobians, covariance, member_mask, observed_covariances
    )
    whitened = whitened_reprojection_residuals(
        jacobians, covariance, member_mask, observed_covariances
    )
    return {
        "relative_frobenius_residual": _quantiles(relative),
        "whitened_rms_residual": _quantiles(whitened),
        "sigma_all_axes": _quantiles(sigmas.reshape(-1)),
        "sigma_min": _quantiles(sigmas[:, 0]),
        "sigma_max": _quantiles(sigmas[:, -1]),
        "condition_number": _quantiles(condition),
    }


def _beam_result(dataset: CompactDataset) -> tuple[BeamFusionResult, dict]:
    config = BeamFusionConfig(
        min_views=3,
        transverse_gate_sigma=3.0,
        max_color_distance=0.35,
        color_sigma=0.25,
        fold_in_gate_sigma=3.0,
        nms_voxel_size=float(dataset.bounds_hint[1]) / 100.0,
        init_opacity=0.10,
        source_chunk=256,
        max_components=N_INIT,
        seed_budget_multiplier=4,
    )
    started = time.perf_counter()
    inputs = selected_inputs(dataset)
    result = fuse_gaussian_beams(inputs, config)
    if result.n_components != N_INIT:
        raise RuntimeError(f"expected {N_INIT} beam components, got {result.n_components}")
    contributor_counts = result.component_offsets[1:] - result.component_offsets[:-1]
    receipt = {
        "elapsed_seconds": time.perf_counter() - started,
        "n_gaussians": result.n_components,
        "n_contributor_links": int(result.contributor_view_indices.numel()),
        "contributing_views": _quantiles(contributor_counts),
        "config": {
            "min_views": config.min_views,
            "transverse_gate_sigma": config.transverse_gate_sigma,
            "max_color_distance": config.max_color_distance,
            "color_sigma": config.color_sigma,
            "fold_in_gate_sigma": config.fold_in_gate_sigma,
            "nms_voxel_size": config.nms_voxel_size,
            "init_opacity": config.init_opacity,
            "source_chunk": config.source_chunk,
            "max_components": config.max_components,
            "seed_budget_multiplier": config.seed_budget_multiplier,
        },
    }
    return result, receipt


def build_initializations(
    dataset: CompactDataset,
) -> tuple[dict[str, Gaussians3D], dict]:
    """Build the frozen-field CI, track-LSQ, and track-robust initialization triplet."""
    result, beam_receipt = _beam_result(dataset)
    inputs = selected_inputs(dataset)
    member_mask, member_cov2d = _member_observations(result, inputs)
    means = result.gaussians.means
    jacobians = _projection_jacobians(means, inputs.cameras)
    _, extent = dataset.bounds_hint
    max_sigma = MAX_SIGMA_EXTENT_FRACTION * float(extent)

    ci_covariances = result.gaussians.covariance().to(torch.float64)
    started = time.perf_counter()
    raw_lsq, linear_residual = _triangulate_covariances(
        means, member_mask, member_cov2d, inputs.cameras
    )
    lsq_covariances = _spd_project(
        raw_lsq,
        min_sigma=MIN_SIGMA_WORLD,
        max_sigma=max_sigma,
    )
    lsq_seconds = time.perf_counter() - started
    raw_eigenvalues = torch.linalg.eigvalsh(raw_lsq)

    started = time.perf_counter()
    robust_covariances, robust_history = robust_refit_from_jacobians(
        jacobians,
        member_mask,
        member_cov2d,
        lsq_covariances,
        ci_covariances,
        max_sigma=max_sigma,
    )
    robust_seconds = time.perf_counter() - started

    inits = {
        "ci": result.gaussians.detach(),
        "track-lsq": _replace_covariances(result.gaussians, lsq_covariances),
        "track-robust": _replace_covariances(result.gaussians, robust_covariances),
    }
    for name, initialization in inits.items():
        if not _same_non_covariance_fields(inits["ci"], initialization):
            raise RuntimeError(f"{name} does not preserve the frozen non-covariance fields")

    diagnostics = {
        "beam": beam_receipt,
        "n_observation_links": int(member_mask.sum()),
        "max_sigma_world": max_sigma,
        "lsq": {
            "elapsed_seconds": lsq_seconds,
            "linear_system_residual": _quantiles(linear_residual),
            "raw_non_spd_count": int((raw_eigenvalues[:, 0] <= 0).sum()),
            "raw_eigenvalue_min": float(raw_eigenvalues.min()),
        },
        "robust": {
            "elapsed_seconds": robust_seconds,
            "steps": ROBUST_STEPS,
            "learning_rate": ROBUST_LR,
            "huber_delta": ROBUST_HUBER_DELTA,
            "ci_prior_weight": ROBUST_PRIOR_WEIGHT,
            "history": robust_history,
        },
        "arms": {
            name: _covariance_diagnostics(
                initialization.covariance(),
                jacobians,
                member_mask,
                member_cov2d,
            )
            for name, initialization in inits.items()
        },
        "frozen_field_assertions": {
            name: {
                "same_count": initialization.n == inits["ci"].n,
                "means_bit_exact": torch.equal(initialization.means, inits["ci"].means),
                "opacity_bit_exact": torch.equal(initialization.opacity, inits["ci"].opacity),
                "sh_bit_exact": torch.equal(initialization.sh, inits["ci"].sh),
            }
            for name, initialization in inits.items()
        },
    }
    return inits, diagnostics


def _write_viewer_manifest(out: Path) -> Path:
    manifest = {
        "schema": "rtgs.viewer-comparison.v1",
        "methods": [
            {
                "name": name,
                "initial": f"../../runs/{out.name}/{name}/gaussians_init.ply",
                "final": f"../../runs/{out.name}/{name}/gaussians_final.ply",
            }
            for name in ARMS
        ],
    }
    path = ROOT / "benchmarks/results/20260723_beam_covariance_refit_VIEWER.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    args = parser.parse_args()
    if not args.protocol.is_file():
        raise FileNotFoundError(
            f"frozen protocol missing: {args.protocol}; write it before reading Janelle outcomes"
        )
    if args.iterations != ITERATIONS:
        print(
            f"[warning] iterations={args.iterations} is a smoke override; frozen value is "
            f"{ITERATIONS}",
            flush=True,
        )

    torch.manual_seed(SEED)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dataset = CompactDataset.load(DATASET, device="cpu")
    scene = build_scene(dataset)
    initializations, covariance = build_initializations(dataset)
    reference_surface = initializations["ci"].means.detach().clone()
    records: dict[str, dict] = {}
    for arm in args.arms:
        records[arm] = run_arm(
            arm,
            initializations[arm],
            scene,
            reference_surface,
            False,
            out,
            args.iterations,
        )
        final = Gaussians3D.load_ply(out / arm / "gaussians_final.ply")
        save_preview(scene, initializations[arm], out / arm / "preview_init.png")
        save_preview(scene, final, out / arm / "preview_final.png")

    summary = {
        "schema": "rtgs.beam_covariance_refit.v1",
        "status": "complete" if set(args.arms) == set(ARMS) else "partial",
        "protocol": {
            "path": str(args.protocol.resolve().relative_to(ROOT)),
            "sha256": _sha256(args.protocol),
        },
        "dataset": str(DATASET.relative_to(ROOT)),
        "selected_global_views": list(SELECTED_VIEWS),
        "evaluation_local_views": list(EVAL_VIEWS),
        "downscale": DOWNSCALE,
        "n_init": N_INIT,
        "iterations": args.iterations,
        "fixed_topology": True,
        "seed": SEED,
        "covariance": covariance,
        "arms": {
            name: {
                "init_metrics": record["init_metrics"],
                "final_metrics": {
                    key.removeprefix("metric_"): value
                    for key, value in record["curve"][-1].items()
                    if key.startswith("metric_")
                },
                "final_n": record["final_n"],
                "elapsed_seconds": record["elapsed_seconds"],
                "loss_first_20": record["loss_first_20"],
                "loss_last_20": record["loss_last_20"],
            }
            for name, record in records.items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    viewer_manifest = _write_viewer_manifest(out)
    print(
        json.dumps(
            {
                name: {
                    "init_psnr_fg": arm["init_metrics"].get("psnr_fg"),
                    "init_alpha_iou": arm["init_metrics"].get("alpha_iou"),
                    "final_psnr_fg": arm["final_metrics"].get("psnr_fg"),
                    "final_alpha_iou": arm["final_metrics"].get("alpha_iou"),
                }
                for name, arm in summary["arms"].items()
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"[viewer] {viewer_manifest.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
