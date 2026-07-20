#!/usr/bin/env python3
"""Exact sampled-confidence attribution test for bounded-ray depth anchoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from depth_anchor_ablation import (
    held_out_geometry,
    history_checkpoints,
    make_priors,
    nearest_gt_diagnostics,
    source_depth_diagnostics,
)
from rtgs.data.synthetic import make_synthetic_scene
from rtgs.image2gs.fit import FitConfig, fit_views
from rtgs.lift.gradient import GradientLifter
from rtgs.optim.trainer import Trainer
from rtgs.render.torch_ref import TorchRasterizer

ARMS = ("valid_uniform", "confidence", "confidence_shuffled")
PREREGISTRATION = Path("benchmarks/results/20260715_depth_anchor_attribution_PREREG.md")
PROTOCOL_AUDIT = Path("benchmarks/results/20260715_depth_anchor_AUDIT.md")


def git_metadata(root: Path) -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "revision": revision,
        "dirty": bool(status.strip()),
        "status": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
    }


def loaded_source_hashes(root: Path) -> tuple[dict[str, str], str]:
    """Hash every loaded repo-local Python source plus the frozen protocol inputs."""
    paths = set()
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if source is None:
            continue
        path = Path(source).resolve()
        if path.suffix == ".py" and path.is_relative_to(root) and path.is_file():
            paths.add(path)
    for relative in (PREREGISTRATION, PROTOCOL_AUDIT, Path("pyproject.toml")):
        path = (root / relative).resolve()
        if path.exists():
            paths.add(path)
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    aggregate = hashlib.sha256()
    for path, digest in hashes.items():
        aggregate.update(path.encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        aggregate.update(b"\n")
    return hashes, aggregate.hexdigest()


def make_lifter(
    mode: str,
    args: argparse.Namespace,
    seed: int,
    *,
    iterations: int | None = None,
    anchor_lambda: float | None = None,
) -> GradientLifter:
    return GradientLifter(
        iterations=args.lift_iters if iterations is None else iterations,
        rasterizer="torch",
        min_weight=args.min_weight,
        depth_jitter=args.depth_jitter,
        depth_prior_lambda=(args.depth_prior_lambda if anchor_lambda is None else anchor_lambda),
        depth_anchor_mode=mode,
        depth_anchor_beta=args.depth_anchor_beta,
        optimize_rotation=False,
        optimize_scale=False,
        merge=False,
        seed=seed,
    )


def assert_step0_and_weight_invariants(
    gaussians2d, train_scene, priors, args: argparse.Namespace, seed: int
) -> dict[str, object]:
    lifters = {}
    outputs = {}
    for arm in ARMS:
        lifter = make_lifter(arm, args, seed, iterations=0)
        outputs[arm] = lifter.lift_with_priors(gaussians2d, train_scene, priors)
        lifters[arm] = lifter

    reference = outputs[ARMS[0]]
    for arm in ARMS[1:]:
        for field in ("means", "quats", "log_scales", "opacity", "sh"):
            if not torch.equal(getattr(reference, field), getattr(outputs[arm], field)):
                raise AssertionError(f"{arm} differs from valid_uniform at step 0 in {field}")
    scales = {arm: lifters[arm].resolved_depth_anchor_scale for arm in ARMS}
    if len(set(scales.values())) != 1:
        raise AssertionError(f"resolved anchor stiffness differs across arms: {scales}")

    diagnostics = {arm: lifters[arm].anchor_weight_diagnostics for arm in ARMS}
    if len({len(items) for items in diagnostics.values()}) != 1:
        raise AssertionError("anchor diagnostic view counts differ across arms")
    per_view = []
    for view_items in zip(*(diagnostics[arm] for arm in ARMS)):
        by_arm = dict(zip(ARMS, view_items))
        reference_item = by_arm["confidence"]
        for arm, item in by_arm.items():
            for key in (
                "view_index",
                "valid_count",
                "valid_indices",
                "distinct_confidence_count",
                "confidence_sum",
                "confidence_square_sum",
            ):
                if item[key] != reference_item[key]:
                    raise AssertionError(f"{arm} changes per-view {key}")
        shuffled = by_arm["confidence_shuffled"]
        if any(int(item["invalid_nonzero_count"]) != 0 for item in by_arm.values()):
            raise AssertionError("an anchor arm assigned nonzero weight to an invalid prior")
        if shuffled["shuffle_multiset_exact"] is not True:
            raise AssertionError("sampled shuffle did not preserve the exact valid multiset")
        if (
            int(shuffled["distinct_confidence_count"]) > 1
            and shuffled["shuffle_location_changed"] is not True
        ):
            raise AssertionError("sampled shuffle left a non-constant confidence view unchanged")
        for moment in ("sum", "square_sum"):
            if shuffled[f"resolved_{moment}"] != shuffled[f"confidence_{moment}"]:
                raise AssertionError(f"sampled shuffle changed confidence {moment}")
        per_view.append({key: value for key, value in shuffled.items() if key != "valid_indices"})
    return {
        "step0_identical": True,
        "primitive_count": reference.n,
        "resolved_anchor_scale": next(iter(scales.values())),
        "per_view_sampled_shuffle": per_view,
    }


def assert_main_rng_invariant(
    gaussians2d, train_scene, priors, args: argparse.Namespace, seed: int
) -> bool:
    outputs = {}
    histories = {}
    for arm in ARMS:
        lifter = make_lifter(
            arm,
            args,
            seed,
            iterations=args.rng_check_iters,
            anchor_lambda=0.0,
        )
        outputs[arm] = lifter.lift_with_priors(gaussians2d, train_scene, priors)
        histories[arm] = lifter.history
    reference = outputs[ARMS[0]]
    for arm in ARMS[1:]:
        for field in ("means", "quats", "log_scales", "opacity", "sh"):
            if not torch.equal(getattr(reference, field), getattr(outputs[arm], field)):
                raise AssertionError(f"{arm} disturbed the main RNG schedule in {field}")
        if histories[arm] != histories[ARMS[0]]:
            raise AssertionError(f"{arm} disturbed the main RNG loss schedule")
    return True


def evaluate_arm(
    arm,
    gaussians2d,
    scene,
    train_scene,
    clean_depths,
    priors,
    corruption_masks,
    args,
    seed,
) -> dict[str, object]:
    lifter = make_lifter(arm, args, seed)
    started = time.perf_counter()
    result = lifter.lift_with_priors(gaussians2d, train_scene, priors)
    lift_seconds = time.perf_counter() - started
    renderer = TorchRasterizer()
    return {
        "n_initial": result.n,
        "lift_seconds": lift_seconds,
        "resolved_anchor_scale": lifter.resolved_depth_anchor_scale,
        "loss_history": history_checkpoints(lifter.history),
        "anchor_history": history_checkpoints(lifter.anchor_history),
        "weight_diagnostics": [
            {key: value for key, value in item.items() if key != "valid_indices"}
            for item in lifter.anchor_weight_diagnostics
        ],
        "initial_test": Trainer.evaluate_metrics(
            scene, result, renderer, indices=scene.testing_views
        ),
        "held_out_geometry": held_out_geometry(scene, result, scene.testing_views),
        "source_depth": source_depth_diagnostics(
            result,
            gaussians2d,
            train_scene,
            clean_depths,
            priors,
            corruption_masks,
            args.min_weight,
        ),
        "nearest_gt": nearest_gt_diagnostics(scene, result),
    }


SUMMARY_PATHS = {
    "test_psnr": ("initial_test", "psnr"),
    "test_ssim": ("initial_test", "ssim"),
    "heldout_depth_rmse": ("held_out_geometry", "depth_rmse_over_extent"),
    "alpha_iou": ("held_out_geometry", "alpha_iou"),
    "coverage": ("held_out_geometry", "foreground_coverage"),
    "source_corrupted_p90": ("source_depth", "corrupted", "p90"),
    "source_all_p90": ("source_depth", "all", "p90"),
    "nearest_gt_median": ("nearest_gt", "median_over_extent"),
    "nearest_gt_p90": ("nearest_gt", "p90_over_extent"),
    "lift_seconds": ("lift_seconds",),
}


def metric_samples(runs, arm: str, path: tuple[str, ...]) -> list[float]:
    samples = []
    for run in runs:
        value = run["arms"][arm]
        for key in path:
            value = value[key]
        if value is not None:
            samples.append(float(value))
    return samples


def aggregate(samples: list[float]) -> dict[str, object]:
    if not samples:
        return {"mean": None, "std": None, "samples": []}
    return {
        "mean": statistics.fmean(samples),
        "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "samples": samples,
    }


def summarize(runs) -> dict[str, object]:
    return {
        arm: {
            name: aggregate(metric_samples(runs, arm, path)) for name, path in SUMMARY_PATHS.items()
        }
        for arm in ARMS
    }


def metric_mean(summary, arm: str, metric: str) -> float:
    value = summary[arm][metric]["mean"]
    if value is None:
        raise ValueError(f"{arm}/{metric} has no finite samples")
    return float(value)


def lower_wins(summary, left: str, right: str, metric: str) -> int:
    left_samples = summary[left][metric]["samples"]
    right_samples = summary[right][metric]["samples"]
    return sum(
        left_value < right_value for left_value, right_value in zip(left_samples, right_samples)
    )


def decision(summary) -> dict[str, object]:
    uniform_rmse = metric_mean(summary, "valid_uniform", "heldout_depth_rmse")
    confidence_rmse = metric_mean(summary, "confidence", "heldout_depth_rmse")
    shuffled_rmse = metric_mean(summary, "confidence_shuffled", "heldout_depth_rmse")
    uniform_p90 = metric_mean(summary, "valid_uniform", "source_corrupted_p90")
    confidence_p90 = metric_mean(summary, "confidence", "source_corrupted_p90")
    shuffled_p90 = metric_mean(summary, "confidence_shuffled", "source_corrupted_p90")
    rmse_gain = (uniform_rmse - confidence_rmse) / max(uniform_rmse, 1e-12)
    shuffled_rmse_gain = (uniform_rmse - shuffled_rmse) / max(uniform_rmse, 1e-12)
    p90_gain = (uniform_p90 - confidence_p90) / max(uniform_p90, 1e-12)
    shuffled_p90_gain = (uniform_p90 - shuffled_p90) / max(uniform_p90, 1e-12)
    rmse_uniform_wins = lower_wins(summary, "confidence", "valid_uniform", "heldout_depth_rmse")
    p90_uniform_wins = lower_wins(summary, "confidence", "valid_uniform", "source_corrupted_p90")
    rmse_shuffle_wins = lower_wins(
        summary, "confidence", "confidence_shuffled", "heldout_depth_rmse"
    )
    p90_shuffle_wins = lower_wins(
        summary, "confidence", "confidence_shuffled", "source_corrupted_p90"
    )
    psnr_delta = metric_mean(summary, "confidence", "test_psnr") - metric_mean(
        summary, "valid_uniform", "test_psnr"
    )
    material_effect = (
        rmse_gain >= 0.02
        and p90_gain >= 0.15
        and rmse_uniform_wins >= 2
        and p90_uniform_wins >= 2
        and psnr_delta >= -0.10
    )
    attributed = (
        material_effect
        and rmse_shuffle_wins >= 2
        and p90_shuffle_wins >= 2
        and shuffled_rmse_gain <= 0.5 * rmse_gain
        and shuffled_p90_gain <= 0.5 * p90_gain
    )
    return {
        "confidence_vs_valid_uniform_depth_rmse_gain_fraction": rmse_gain,
        "confidence_vs_valid_uniform_corrupted_p90_gain_fraction": p90_gain,
        "confidence_vs_valid_uniform_depth_rmse_seed_wins": rmse_uniform_wins,
        "confidence_vs_valid_uniform_corrupted_p90_seed_wins": p90_uniform_wins,
        "confidence_vs_valid_uniform_psnr_delta_db": psnr_delta,
        "confidence_shuffled_depth_rmse_gain_fraction": shuffled_rmse_gain,
        "confidence_shuffled_corrupted_p90_gain_fraction": shuffled_p90_gain,
        "confidence_vs_shuffled_depth_rmse_seed_wins": rmse_shuffle_wins,
        "confidence_vs_shuffled_corrupted_p90_seed_wins": p90_shuffle_wins,
        "shuffle_erases_half_depth_rmse_gain": shuffled_rmse_gain <= 0.5 * rmse_gain,
        "shuffle_erases_half_corrupted_p90_gain": shuffled_p90_gain <= 0.5 * p90_gain,
        "material_effect_pass": material_effect,
        "confidence_location_attribution_pass": attributed,
        "stop_confidence_anchor_sweeps": not attributed,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parent.parent
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
        priors, corruption_masks = make_priors(clean_depths, "corrupted", seed, args.block_size)
        invariants = assert_step0_and_weight_invariants(
            gaussians2d, train_scene, priors, args, seed
        )
        invariants["lambda_zero_main_rng_identical"] = assert_main_rng_invariant(
            gaussians2d, train_scene, priors, args, seed
        )
        seed_result = {
            "seed": seed,
            "train_indices": scene.training_views,
            "test_indices": scene.testing_views,
            "fit_seconds": fit_seconds,
            "fit_psnr_mean": statistics.fmean(item["final_psnr"] for item in fit_history),
            "invariants": invariants,
            "arms": {},
        }
        for arm in ARMS:
            seed_result["arms"][arm] = evaluate_arm(
                arm,
                gaussians2d,
                scene,
                train_scene,
                clean_depths,
                priors,
                corruption_masks,
                args,
                seed,
            )
        counts = {arm: seed_result["arms"][arm]["n_initial"] for arm in ARMS}
        scales = {arm: seed_result["arms"][arm]["resolved_anchor_scale"] for arm in ARMS}
        if len(set(counts.values())) != 1:
            raise AssertionError(f"final primitive counts differ across arms: {counts}")
        if len(set(scales.values())) != 1:
            raise AssertionError(f"final anchor stiffness differs across arms: {scales}")
        runs.append(seed_result)
    summary = summarize(runs)
    source_hashes, source_tree_hash = loaded_source_hashes(root)
    return {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": git_metadata(root),
            "source_sha256": source_hashes,
            "source_tree_sha256": source_tree_hash,
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "command": [sys.executable, *sys.argv],
            "config": {
                key: value for key, value in vars(args).items() if not isinstance(value, Path)
            },
            "preregistration": str(PREREGISTRATION),
        },
        "runs": runs,
        "summary": summary,
        "decision": decision(summary),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--test-every", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--gt-gaussians", type=int, default=40)
    parser.add_argument("--fit-gaussians", type=int, default=150)
    parser.add_argument("--fit-iters", type=int, default=120)
    parser.add_argument("--lift-iters", type=int, default=60)
    parser.add_argument("--rng-check-iters", type=int, default=2)
    parser.add_argument("--depth-jitter", type=float, default=0.02)
    parser.add_argument("--depth-prior-lambda", type=float, default=0.01)
    parser.add_argument("--depth-anchor-beta", type=float, default=0.05)
    parser.add_argument("--min-weight", type=float, default=0.05)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def is_preregistered_config(args: argparse.Namespace) -> bool:
    expected = {
        "seeds": [0, 1, 2],
        "views": 12,
        "test_every": 4,
        "image_size": 48,
        "gt_gaussians": 40,
        "fit_gaussians": 150,
        "fit_iters": 120,
        "lift_iters": 60,
        "rng_check_iters": 2,
        "depth_jitter": 0.02,
        "depth_prior_lambda": 0.01,
        "depth_anchor_beta": 0.05,
        "min_weight": 0.05,
        "block_size": 8,
        "threads": 4,
    }
    return all(getattr(args, key) == value for key, value in expected.items())


def main() -> int:
    args = parse_args()
    if args.output is not None and not is_preregistered_config(args):
        raise ValueError("tracked output requires the exact preregistered configuration")
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
