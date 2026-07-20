#!/usr/bin/env python3
"""Paired CPU ablation for bounded-ray depth-anchor semantics.

The harness reuses fitted 2D Gaussians across all arms, consumes training views only for
lifting, and evaluates clean, calibrated-corruption, and confidence-shuffled conditions
on a strict held-out split. Synthetic GT is used only to construct the controlled causal
instrument and to report geometry diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.data.synthetic import make_synthetic_scene
from rtgs.image2gs.fit import FitConfig, fit_views
from rtgs.lift.base import bilinear_sample
from rtgs.lift.depth import AlignedDepthPrior
from rtgs.lift.gradient import GradientLifter, _ray_box
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.torch_ref import TorchRasterizer

MODES = ("legacy", "normalized", "confidence", "thresholded")
CONDITIONS = ("clean", "corrupted", "shuffled")


def git_state(root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return revision, dirty


def source_hashes(root: Path) -> dict[str, str]:
    paths = (
        Path("benchmarks/depth_anchor_ablation.py"),
        Path("src/rtgs/lift/gradient.py"),
        Path("src/rtgs/lift/hybrid.py"),
    )
    return {str(path): hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def make_priors(
    clean_depths: list[torch.Tensor], condition: str, seed: int, block_size: int
) -> tuple[list[AlignedDepthPrior], list[torch.Tensor]]:
    """Build clean/corrupted priors and masks identifying the actual corrupted blocks."""
    priors = []
    corruption_masks = []
    for view_index, clean in enumerate(clean_depths):
        valid = torch.isfinite(clean) & (clean > 0.05)
        y, x = torch.meshgrid(
            torch.arange(clean.shape[0]), torch.arange(clean.shape[1]), indexing="ij"
        )
        corrupt = valid & (((x // block_size + y // block_size + view_index) % 3) == 0)
        depth = clean.clone()
        confidence = valid.to(clean)
        if condition != "clean":
            factor = 1.2 if view_index % 2 == 0 else 0.8
            depth = torch.where(corrupt, clean * factor, depth)
            confidence = torch.where(corrupt, torch.full_like(confidence, 0.05), confidence)
        if condition == "shuffled":
            valid_values = confidence[valid]
            generator = torch.Generator().manual_seed(seed * 10_000 + view_index)
            permutation = torch.randperm(valid_values.numel(), generator=generator)
            shuffled = confidence.clone()
            shuffled[valid] = valid_values[permutation]
            confidence = shuffled
        priors.append(AlignedDepthPrior(depth=depth, confidence=confidence))
        corruption_masks.append(corrupt)
    return priors, corruption_masks


def subset_gaussians(g2d: Gaussians2D, keep: torch.Tensor) -> Gaussians2D:
    return Gaussians2D(
        xy=g2d.xy[keep],
        chol=g2d.chol[keep],
        color=g2d.color[keep],
        weight=g2d.weight[keep],
    )


def retained_observations(
    gaussians2d: list[Gaussians2D], scene, min_weight: float
) -> list[tuple[int, torch.Tensor]]:
    """Reproduce GradientLifter's pre-merge filtering for source-ray diagnostics."""
    center, extent = scene.center_and_extent()
    half = 0.5 * extent
    retained = []
    for view_index, (g2d, camera) in enumerate(zip(gaussians2d, scene.cameras)):
        keep = g2d.weight > min_weight
        if scene.masks is not None:
            keep &= bilinear_sample(scene.masks[view_index].to(g2d.xy), g2d.xy) > 0.5
        selected = subset_gaussians(g2d, keep)
        if selected.n == 0:
            continue
        origin, direction = camera.pixel_rays(selected.xy)
        near, far = _ray_box(origin, direction, center - half, center + half)
        intersects = far > near.clamp_min(0.05)
        if bool(intersects.any()):
            retained.append((view_index, selected.xy[intersects]))
    return retained


def distribution(values: torch.Tensor) -> dict[str, float | int | None]:
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    return {
        "n": int(values.numel()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p90": float(torch.quantile(values, 0.9)),
    }


def source_depth_diagnostics(
    gaussians,
    gaussians2d: list[Gaussians2D],
    scene,
    clean_depths: list[torch.Tensor],
    priors: list[AlignedDepthPrior],
    corruption_masks: list[torch.Tensor],
    min_weight: float,
) -> dict[str, dict[str, float | int | None]]:
    errors = {name: [] for name in ("all", "corrupted", "uncorrupted", "low", "high")}
    center, extent = scene.center_and_extent()
    half = 0.5 * extent
    offset = 0
    for view_index, xy in retained_observations(gaussians2d, scene, min_weight):
        n = xy.shape[0]
        part = gaussians.means[offset : offset + n]
        if part.shape[0] != n:
            raise AssertionError("lifted output no longer matches retained source observations")
        _, predicted_depth = scene.cameras[view_index].project(part)
        gt_depth = bilinear_sample(clean_depths[view_index].to(xy), xy)
        confidence = bilinear_sample(priors[view_index].confidence.to(xy), xy)
        corrupted = bilinear_sample(corruption_masks[view_index].to(xy), xy) > 0.5
        origin, direction = scene.cameras[view_index].pixel_rays(xy)
        near, far = _ray_box(origin, direction, center - half, center + half)
        near = near.clamp_min(0.05)
        valid = torch.isfinite(gt_depth) & (gt_depth > near) & (gt_depth < far)
        relative_error = (predicted_depth - gt_depth).abs() / gt_depth.clamp_min(0.05)
        errors["all"].append(relative_error[valid])
        errors["corrupted"].append(relative_error[valid & corrupted])
        errors["uncorrupted"].append(relative_error[valid & ~corrupted])
        errors["low"].append(relative_error[valid & (confidence < 0.3)])
        errors["high"].append(relative_error[valid & (confidence > 0.7)])
        offset += n
    if offset != gaussians.n:
        raise AssertionError(f"source layout accounts for {offset} of {gaussians.n} gaussians")
    return {
        name: distribution(torch.cat(parts) if parts else gaussians.means.new_empty(0))
        for name, parts in errors.items()
    }


def held_out_geometry(scene, gaussians, indices: list[int]) -> dict[str, float]:
    renderer = TorchRasterizer()
    _, extent = scene.center_and_extent()
    depth_squared_error = 0.0
    depth_pixels = 0
    intersections = 0
    unions = 0
    true_pixels = 0
    with torch.no_grad():
        for index in indices:
            predicted = renderer.render(gaussians, scene.cameras[index])
            truth = renderer.render(scene.gt_gaussians, scene.cameras[index])
            predicted_fg = predicted.alpha > 0.05
            true_fg = truth.alpha > 0.05
            intersection = predicted_fg & true_fg
            union = predicted_fg | true_fg
            intersections += int(intersection.sum())
            unions += int(union.sum())
            true_pixels += int(true_fg.sum())
            if bool(intersection.any()):
                predicted_depth = predicted.depth / predicted.alpha.clamp_min(1e-8)
                true_depth = truth.depth / truth.alpha.clamp_min(1e-8)
                depth_squared_error += float(
                    (predicted_depth[intersection] - true_depth[intersection]).square().sum()
                )
                depth_pixels += int(intersection.sum())
    return {
        "depth_rmse_over_extent": (depth_squared_error / max(depth_pixels, 1)) ** 0.5 / extent,
        "alpha_iou": intersections / max(unions, 1),
        "foreground_coverage": intersections / max(true_pixels, 1),
    }


def nearest_gt_diagnostics(scene, gaussians) -> dict[str, float]:
    _, extent = scene.center_and_extent()
    distance = torch.cdist(gaussians.means, scene.gt_gaussians.means).min(dim=1).values / extent
    return {
        "median_over_extent": float(distance.median()),
        "p90_over_extent": float(torch.quantile(distance, 0.9)),
    }


def history_checkpoints(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    indices = sorted({0, min(9, len(values) - 1), min(29, len(values) - 1), len(values) - 1})
    return {str(index + 1): values[index] for index in indices}


def make_lifter(mode: str, args, seed: int, iterations: int | None = None) -> GradientLifter:
    return GradientLifter(
        iterations=args.lift_iters if iterations is None else iterations,
        rasterizer="torch",
        depth_jitter=args.depth_jitter,
        depth_prior_lambda=args.depth_prior_lambda,
        depth_anchor_mode=mode,
        depth_anchor_beta=args.depth_anchor_beta,
        depth_confidence_threshold=args.confidence_threshold,
        min_weight=args.min_weight,
        optimize_rotation=False,
        optimize_scale=False,
        merge=False,
        seed=seed,
    )


def assert_identical_initialization(gaussians2d, train_scene, priors, args, seed) -> None:
    controls = []
    for mode in args.modes:
        controls.append(
            make_lifter(mode, args, seed, iterations=0).lift_with_priors(
                gaussians2d, train_scene, priors
            )
        )
    reference = controls[0]
    for control in controls[1:]:
        for field in ("means", "quats", "log_scales", "opacity", "sh"):
            if not torch.equal(getattr(reference, field), getattr(control, field)):
                raise AssertionError(f"anchor arms differ at step 0 in {field}")


def evaluate_mode(
    mode,
    gaussians2d,
    scene,
    train_scene,
    clean_depths,
    priors,
    corruption_masks,
    condition,
    args,
    seed,
):
    lifter = make_lifter(mode, args, seed)
    started = time.perf_counter()
    initial = lifter.lift_with_priors(gaussians2d, train_scene, priors)
    lift_seconds = time.perf_counter() - started
    renderer = TorchRasterizer()
    nearest = nearest_gt_diagnostics(scene, initial)
    result = {
        "n_initial": initial.n,
        "lift_seconds": lift_seconds,
        "resolved_anchor_scale": lifter.resolved_depth_anchor_scale,
        "loss_history": history_checkpoints(lifter.history),
        "anchor_history": history_checkpoints(lifter.anchor_history),
        "initial_test": Trainer.evaluate_metrics(
            scene, initial, renderer, indices=scene.testing_views
        ),
        "initial_held_out_geometry": held_out_geometry(scene, initial, scene.testing_views),
        "initial_source_depth": source_depth_diagnostics(
            initial,
            gaussians2d,
            train_scene,
            clean_depths,
            priors,
            corruption_masks,
            args.min_weight,
        ),
        "initial_nearest_gt": nearest,
    }
    if condition == "corrupted" and args.refine_iters > 0:
        config = TrainConfig(
            iterations=args.refine_iters,
            device="cpu",
            rasterizer="torch",
            densify=False,
            target_sh_degree=0,
            eval_every=args.refine_iters,
            seed=seed,
        )
        started = time.perf_counter()
        final, _ = Trainer(config).train(scene, initial)
        result["refine_seconds"] = time.perf_counter() - started
        result["n_final"] = final.n
        result["final_test"] = Trainer.evaluate_metrics(
            scene, final, renderer, indices=scene.testing_views
        )
        result["final_held_out_geometry"] = held_out_geometry(scene, final, scene.testing_views)
        result["final_source_depth"] = source_depth_diagnostics(
            final,
            gaussians2d,
            train_scene,
            clean_depths,
            priors,
            corruption_masks,
            args.min_weight,
        )
        result["final_nearest_gt"] = nearest_gt_diagnostics(scene, final)
    return result


def metric_samples(runs, condition: str, mode: str, path: tuple[str, ...]) -> list[float]:
    samples = []
    for run in runs:
        value = run["conditions"][condition]["modes"][mode]
        for key in path:
            value = value[key]
        if value is not None:
            samples.append(float(value))
    return samples


def aggregate(samples: list[float]) -> dict[str, float | list[float] | None]:
    if not samples:
        return {"mean": None, "std": None, "samples": []}
    return {
        "mean": statistics.fmean(samples),
        "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "samples": samples,
    }


SUMMARY_PATHS = {
    "initial_test_psnr": ("initial_test", "psnr"),
    "initial_test_ssim": ("initial_test", "ssim"),
    "initial_depth_rmse_over_extent": ("initial_held_out_geometry", "depth_rmse_over_extent"),
    "initial_alpha_iou": ("initial_held_out_geometry", "alpha_iou"),
    "initial_coverage": ("initial_held_out_geometry", "foreground_coverage"),
    "source_low_p90": ("initial_source_depth", "low", "p90"),
    "source_corrupted_p90": ("initial_source_depth", "corrupted", "p90"),
    "nearest_gt_median": ("initial_nearest_gt", "median_over_extent"),
    "nearest_gt_p90": ("initial_nearest_gt", "p90_over_extent"),
    "lift_seconds": ("lift_seconds",),
}

FINAL_SUMMARY_PATHS = {
    "final_test_psnr": ("final_test", "psnr"),
    "final_test_ssim": ("final_test", "ssim"),
    "final_depth_rmse_over_extent": ("final_held_out_geometry", "depth_rmse_over_extent"),
    "final_alpha_iou": ("final_held_out_geometry", "alpha_iou"),
    "final_coverage": ("final_held_out_geometry", "foreground_coverage"),
    "final_source_corrupted_p90": ("final_source_depth", "corrupted", "p90"),
}


def summarize(runs, conditions: list[str], modes: list[str]) -> dict:
    summary = {}
    for condition in conditions:
        summary[condition] = {}
        for mode in modes:
            metrics = {
                name: aggregate(metric_samples(runs, condition, mode, path))
                for name, path in SUMMARY_PATHS.items()
            }
            if condition == "corrupted":
                metrics.update(
                    {
                        name: aggregate(metric_samples(runs, condition, mode, path))
                        for name, path in FINAL_SUMMARY_PATHS.items()
                    }
                )
            summary[condition][mode] = metrics
    return summary


def mean(summary, condition: str, mode: str, metric: str) -> float:
    value = summary[condition][mode][metric]["mean"]
    if value is None:
        raise ValueError(f"{condition}/{mode}/{metric} has no finite samples")
    return float(value)


def decision(summary) -> dict:
    init_delta = mean(summary, "corrupted", "confidence", "initial_test_psnr") - mean(
        summary, "corrupted", "legacy", "initial_test_psnr"
    )
    confidence_samples = summary["corrupted"]["confidence"]["initial_test_psnr"]["samples"]
    legacy_samples = summary["corrupted"]["legacy"]["initial_test_psnr"]["samples"]
    seed_wins = sum(c > legacy for c, legacy in zip(confidence_samples, legacy_samples))
    legacy_low = mean(summary, "corrupted", "legacy", "source_low_p90")
    confidence_low = mean(summary, "corrupted", "confidence", "source_low_p90")
    low_reduction = 100.0 * (legacy_low - confidence_low) / max(legacy_low, 1e-12)
    final_delta = mean(summary, "corrupted", "confidence", "final_test_psnr") - mean(
        summary, "corrupted", "legacy", "final_test_psnr"
    )
    clean_delta = mean(summary, "clean", "confidence", "initial_test_psnr") - mean(
        summary, "clean", "legacy", "initial_test_psnr"
    )
    threshold_delta = mean(summary, "corrupted", "confidence", "initial_test_psnr") - mean(
        summary, "corrupted", "thresholded", "initial_test_psnr"
    )
    threshold_low = mean(summary, "corrupted", "thresholded", "source_low_p90")
    threshold_relative = 100.0 * (confidence_low - threshold_low) / max(threshold_low, 1e-12)

    normalized_rmse = mean(summary, "corrupted", "normalized", "initial_depth_rmse_over_extent")
    confidence_rmse = mean(summary, "corrupted", "confidence", "initial_depth_rmse_over_extent")
    calibrated_gain = (normalized_rmse - confidence_rmse) / max(normalized_rmse, 1e-12)
    shuffled_normalized = mean(summary, "shuffled", "normalized", "initial_depth_rmse_over_extent")
    shuffled_confidence = mean(summary, "shuffled", "confidence", "initial_depth_rmse_over_extent")
    shuffled_gain = (shuffled_normalized - shuffled_confidence) / max(shuffled_normalized, 1e-12)
    primary_pass = (
        init_delta >= 0.25
        and seed_wins >= 2
        and low_reduction >= 15.0
        and final_delta >= 0.10
        and clean_delta >= -0.10
        and threshold_delta >= -0.10
        and threshold_relative <= 5.0
    )
    confidence_location_causal = calibrated_gain > 0 and shuffled_gain <= 0.5 * calibrated_gain
    return {
        "confidence_vs_legacy_corrupted_init_psnr_delta_db": init_delta,
        "confidence_vs_legacy_seed_wins": seed_wins,
        "confidence_vs_legacy_low_confidence_p90_reduction_percent": low_reduction,
        "confidence_vs_legacy_corrupted_final_psnr_delta_db": final_delta,
        "confidence_vs_legacy_clean_init_psnr_delta_db": clean_delta,
        "confidence_vs_thresholded_corrupted_init_psnr_delta_db": threshold_delta,
        "confidence_low_p90_vs_thresholded_percent": threshold_relative,
        "confidence_vs_normalized_calibrated_depth_rmse_gain_fraction": calibrated_gain,
        "confidence_vs_normalized_shuffled_depth_rmse_gain_fraction": shuffled_gain,
        "confidence_location_causal": confidence_location_causal,
        "primary_hypothesis_pass": primary_pass and confidence_location_causal,
    }


def run(args) -> dict:
    root = Path(__file__).resolve().parent.parent
    revision, dirty = git_state(root)
    torch.set_num_threads(args.threads)
    runs = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        scene = make_synthetic_scene(
            n_gaussians=args.gt_gaussians,
            n_cameras=args.views,
            image_size=args.image_size,
            seed=seed,
        )
        scene.test_indices = [
            index for index in range(args.views) if index % args.test_every == args.test_every - 1
        ]
        scene.train_indices = [
            index for index in range(args.views) if index not in scene.test_indices
        ]
        train_scene = scene.subset(scene.training_views)
        fit_config = FitConfig(
            n_gaussians=args.fit_gaussians,
            iterations=args.fit_iters,
            log_every=max(args.fit_iters, 1),
        )
        fit_started = time.perf_counter()
        gaussians2d, fit_history = fit_views(train_scene.images, fit_config, seed=seed)
        fit_seconds = time.perf_counter() - fit_started
        clean_depths = [depth.clone() for depth in train_scene.gt_depths]
        seed_result = {
            "seed": seed,
            "train_indices": scene.training_views,
            "test_indices": scene.testing_views,
            "fit_seconds": fit_seconds,
            "fit_psnr_mean": statistics.fmean(item["final_psnr"] for item in fit_history),
            "conditions": {},
        }
        for condition in args.conditions:
            priors, corruption_masks = make_priors(clean_depths, condition, seed, args.block_size)
            assert_identical_initialization(gaussians2d, train_scene, priors, args, seed)
            condition_result = {"step0_identical": True, "modes": {}}
            for mode in args.modes:
                condition_result["modes"][mode] = evaluate_mode(
                    mode,
                    gaussians2d,
                    scene,
                    train_scene,
                    clean_depths,
                    priors,
                    corruption_masks,
                    condition,
                    args,
                    seed,
                )
            seed_result["conditions"][condition] = condition_result
        runs.append(seed_result)
    summary = summarize(runs, args.conditions, args.modes)
    return {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_rev": revision,
            "git_dirty": dirty,
            "source_sha256": source_hashes(root),
            "torch": torch.__version__,
            "device": "cpu",
            "command": [sys.executable, *sys.argv],
            "config": {
                key: value for key, value in vars(args).items() if not isinstance(value, Path)
            },
        },
        "runs": runs,
        "summary": summary,
        "decision": decision(summary),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--test-every", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--gt-gaussians", type=int, default=40)
    parser.add_argument("--fit-gaussians", type=int, default=150)
    parser.add_argument("--fit-iters", type=int, default=120)
    parser.add_argument("--lift-iters", type=int, default=60)
    parser.add_argument("--refine-iters", type=int, default=60)
    parser.add_argument("--depth-jitter", type=float, default=0.02)
    parser.add_argument("--depth-prior-lambda", type=float, default=0.01)
    parser.add_argument("--depth-anchor-beta", type=float, default=0.05)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--min-weight", type=float, default=0.05)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required = set(MODES)
    if set(args.modes) != required or set(args.conditions) != set(CONDITIONS):
        raise ValueError("decision summary requires all preregistered modes and conditions")
    payload = run(args)
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
