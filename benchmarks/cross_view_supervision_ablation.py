#!/usr/bin/env python3
"""Paired LOSO versus matched-dropout supervision experiment."""

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
from rtgs.depth.base import DepthPrediction
from rtgs.image2gs.fit import FitConfig, fit_views
from rtgs.lift.depth import AlignedDepthPrior
from rtgs.lift.gradient import GradientLifter, _build_photometric_keep_masks
from rtgs.lift.hybrid import HybridLifter
from rtgs.optim.trainer import Trainer
from rtgs.render.torch_ref import TorchRasterizer

FAMILIES = ("gradient", "hybrid")
MODES = ("all", "leave_one_source_out", "matched_nonself_dropout")
PREREGISTRATION = Path("benchmarks/results/20260715_cross_view_supervision_PREREG.md")


class ListDepthBackend:
    """Return one frozen metric depth map per sequential HybridLifter call."""

    def __init__(self, depths: list[torch.Tensor]):
        self.depths = [depth.detach().clone() for depth in depths]
        self.index = 0

    def predict(self, image: torch.Tensor) -> DepthPrediction:
        if self.index >= len(self.depths):
            raise RuntimeError("depth backend received more calls than frozen views")
        depth = self.depths[self.index].to(image)
        self.index += 1
        return DepthPrediction(depth=depth, kind="metric", confidence=None)


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
    """Hash every loaded repository Python source plus frozen protocol inputs."""
    paths = set()
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if source is None:
            continue
        path = Path(source).resolve()
        if path.suffix == ".py" and path.is_relative_to(root) and path.is_file():
            paths.add(path)
    for relative in (PREREGISTRATION, Path("pyproject.toml")):
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


def fitted_tensor_hash(gaussians2d) -> str:
    items = []
    for view_index, gaussians in enumerate(gaussians2d):
        for field in ("xy", "chol", "color", "weight"):
            items.append((f"view{view_index}/{field}", getattr(gaussians, field)))
    return tensor_collection_hash(items)


def depth_tensor_hash(depths: list[torch.Tensor]) -> str:
    return tensor_collection_hash(
        [(f"view{view_index}/depth", depth) for view_index, depth in enumerate(depths)]
    )


def integer_sequence_hash(values: list[int]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def make_lifter(
    family: str,
    mode: str,
    args: argparse.Namespace,
    seed: int,
    corrupted_depths: list[torch.Tensor],
    *,
    iterations: int | None = None,
    lr: float | None = None,
):
    common = {
        "iterations": args.lift_iters if iterations is None else iterations,
        "lr": args.lr if lr is None else lr,
        "rasterizer": "torch",
        "min_weight": args.min_weight,
        "optimize_rotation": False,
        "optimize_scale": False,
        "merge": False,
        "seed": seed,
        "depth_anchor_mode": "legacy",
        "photometric_supervision_mode": mode,
    }
    if family == "gradient":
        return GradientLifter(
            depth_jitter=args.gradient_depth_jitter,
            depth_prior_lambda=args.gradient_depth_prior_lambda,
            **common,
        )
    if family == "hybrid":
        return HybridLifter(
            backend=ListDepthBackend(corrupted_depths),
            depth_prior_lambda=args.hybrid_depth_prior_lambda,
            merge_color_bin_size=None,
            depth_jitter=args.hybrid_depth_jitter,
            **common,
        )
    raise ValueError(f"unknown family {family!r}")


def core_lifter(lifter) -> GradientLifter:
    return lifter if isinstance(lifter, GradientLifter) else lifter.gradient


def assert_equal_fields(left, right, context: str) -> None:
    for field in ("means", "quats", "log_scales", "opacity", "sh"):
        if not torch.equal(getattr(left, field), getattr(right, field)):
            raise AssertionError(f"{context} differs in {field}")


def assert_family_invariants(
    family: str,
    gaussians2d,
    train_scene,
    corrupted_depths,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    zero_outputs = {}
    zero_cores = {}
    for mode in MODES:
        lifter = make_lifter(
            family,
            mode,
            args,
            seed,
            corrupted_depths,
            iterations=0,
        )
        zero_outputs[mode] = lifter.lift(gaussians2d, train_scene)
        zero_cores[mode] = core_lifter(lifter)
    reference = zero_outputs["all"]
    source_ids = zero_cores["all"].source_view_ids_before_merge
    if source_ids is None:
        raise AssertionError("source-view labels were not recorded")
    for mode in MODES[1:]:
        assert_equal_fields(reference, zero_outputs[mode], f"{family}/{mode} step 0")
        other_ids = zero_cores[mode].source_view_ids_before_merge
        if other_ids is None or not torch.equal(source_ids, other_ids):
            raise AssertionError(f"{family}/{mode} changes source-view labels")
        if (
            zero_cores[mode].source_view_ranges_before_merge
            != zero_cores["all"].source_view_ranges_before_merge
        ):
            raise AssertionError(f"{family}/{mode} changes source-view boundaries")

    masks = {}
    mask_hashes = {}
    diagnostics = {}
    for mode in MODES:
        mode_masks, mode_diagnostics = _build_photometric_keep_masks(
            source_ids, train_scene.training_views, mode, seed
        )
        masks[mode] = mode_masks
        diagnostics[mode] = mode_diagnostics
        mask_hashes[mode] = tensor_collection_hash(
            [
                (f"target{target_view}", mode_masks[target_view])
                for target_view in train_scene.training_views
            ]
        )
    for target_view in train_scene.training_views:
        own = source_ids == target_view
        own_count = int(own.sum())
        if own_count == 0:
            raise AssertionError(f"target view {target_view} has no retained own-source primitives")
        all_keep = masks["all"][target_view]
        loso_keep = masks["leave_one_source_out"][target_view]
        matched_keep = masks["matched_nonself_dropout"][target_view]
        if not bool(all_keep.all()) or not torch.equal(loso_keep, ~own):
            raise AssertionError(f"target view {target_view} has an invalid all/LOSO mask")
        if not bool(matched_keep[own].all()):
            raise AssertionError(f"matched dropout excludes own-source target {target_view}")
        if int((~loso_keep).sum()) != int((~matched_keep).sum()):
            raise AssertionError(f"target view {target_view} dropout counts differ")
    matched_exposure = torch.stack(
        [
            (~masks["matched_nonself_dropout"][target_view]).long()
            for target_view in train_scene.training_views
        ]
    ).sum(dim=0)
    if not bool((matched_exposure == 1).all()):
        raise AssertionError("matched dropout does not exclude every primitive exactly once")
    zero_supervision = {
        mode: {
            item["target_view"]: item
            for item in zero_cores[mode].photometric_supervision_diagnostics
        }
        for mode in MODES
    }
    for target_view in train_scene.training_views:
        loso_item = zero_supervision["leave_one_source_out"][target_view]
        matched_item = zero_supervision["matched_nonself_dropout"][target_view]
        if loso_item["excluded_opacity_sum"] != matched_item["excluded_opacity_sum"]:
            raise AssertionError(f"target view {target_view} excluded opacity sums differ")

    zero_lr_outputs = {}
    zero_lr_schedules = {}
    for mode in MODES:
        lifter = make_lifter(
            family,
            mode,
            args,
            seed,
            corrupted_depths,
            iterations=args.rng_check_iters,
            lr=0.0,
        )
        zero_lr_outputs[mode] = lifter.lift(gaussians2d, train_scene)
        zero_lr_schedules[mode] = core_lifter(lifter).target_view_history
    for mode in MODES[1:]:
        assert_equal_fields(
            zero_lr_outputs["all"],
            zero_lr_outputs[mode],
            f"{family}/{mode} zero-learning-rate RNG check",
        )
        if zero_lr_schedules[mode] != zero_lr_schedules["all"]:
            raise AssertionError(f"{family}/{mode} changes the target-view RNG schedule")

    return {
        "step0_identical": True,
        "zero_lr_outputs_identical": True,
        "zero_lr_target_schedule_identical": True,
        "primitive_count": reference.n,
        "source_view_ids_sha256": tensor_collection_hash([("source_view_ids", source_ids)]),
        "source_view_ranges": zero_cores["all"].source_view_ranges_before_merge,
        "mask_sha256": mask_hashes,
        "matched_exclusion_exposure_sha256": tensor_collection_hash(
            [("matched_exclusion_exposure", matched_exposure)]
        ),
        "matched_exclusion_exposure_exactly_once": True,
        "mask_diagnostics": diagnostics,
        "step0_supervision_diagnostics": {
            mode: zero_cores[mode].photometric_supervision_diagnostics for mode in MODES
        },
    }


def finite_result(result, core: GradientLifter, context: str) -> None:
    for field in ("means", "quats", "log_scales", "opacity", "sh"):
        if not bool(torch.isfinite(getattr(result, field)).all()):
            raise AssertionError(f"{context} has non-finite {field}")
    for name, values in (("loss", core.history), ("anchor", core.anchor_history)):
        if any(not math.isfinite(value) for value in values):
            raise AssertionError(f"{context} has non-finite {name} history")


def cross_only_training_l1(train_scene, result, source_view_ids: torch.Tensor) -> float:
    """Evaluate every arm with the same source-excluded train-view objective."""
    renderer = TorchRasterizer()
    losses = []
    with torch.no_grad():
        for target_view in train_scene.training_views:
            rendered = renderer.render(
                result.subset(source_view_ids != target_view),
                train_scene.cameras[target_view],
            )
            target = train_scene.images[target_view]
            if train_scene.masks is None:
                loss = (rendered.color - target).abs().mean()
            else:
                mask = train_scene.masks[target_view].to(target)[..., None]
                target = target * mask
                loss = ((rendered.color - target).abs() * (0.1 + 0.9 * mask)).mean()
            losses.append(float(loss))
    return statistics.fmean(losses)


def evaluate_arm(
    family: str,
    mode: str,
    gaussians2d,
    scene,
    train_scene,
    clean_depths,
    corrupted_depths,
    diagnostic_priors,
    corruption_masks,
    args,
    seed,
) -> dict[str, object]:
    lifter = make_lifter(family, mode, args, seed, corrupted_depths)
    started = time.perf_counter()
    result = lifter.lift(gaussians2d, train_scene)
    lift_seconds = time.perf_counter() - started
    core = core_lifter(lifter)
    finite_result(result, core, f"{family}/{mode}")
    source_view_ids = core.source_view_ids_before_merge
    if source_view_ids is None:
        raise AssertionError(f"{family}/{mode} did not retain source-view labels")
    renderer = TorchRasterizer()
    return {
        "n_initial": result.n,
        "lift_seconds": lift_seconds,
        "loss_history": history_checkpoints(core.history),
        "anchor_history": history_checkpoints(core.anchor_history),
        "target_view_history": core.target_view_history,
        "target_view_history_sha256": integer_sequence_hash(core.target_view_history),
        "rendered_count_history": core.rendered_count_history,
        "supervision_diagnostics": core.photometric_supervision_diagnostics,
        "initial_test": Trainer.evaluate_metrics(
            scene, result, renderer, indices=scene.testing_views
        ),
        "training": Trainer.evaluate_metrics(
            train_scene, result, renderer, indices=train_scene.training_views
        ),
        "cross_only_training_l1": cross_only_training_l1(train_scene, result, source_view_ids),
        "held_out_geometry": held_out_geometry(scene, result, scene.testing_views),
        "source_depth": source_depth_diagnostics(
            result,
            gaussians2d,
            train_scene,
            clean_depths,
            diagnostic_priors,
            corruption_masks,
            args.min_weight,
        ),
        "nearest_gt": nearest_gt_diagnostics(scene, result),
    }


SUMMARY_PATHS = {
    "test_psnr": ("initial_test", "psnr"),
    "test_ssim": ("initial_test", "ssim"),
    "train_psnr": ("training", "psnr"),
    "cross_only_training_l1": ("cross_only_training_l1",),
    "heldout_depth_rmse": ("held_out_geometry", "depth_rmse_over_extent"),
    "alpha_iou": ("held_out_geometry", "alpha_iou"),
    "coverage": ("held_out_geometry", "foreground_coverage"),
    "source_all_p90": ("source_depth", "all", "p90"),
    "source_corrupted_p90": ("source_depth", "corrupted", "p90"),
    "source_all_median": ("source_depth", "all", "median"),
    "nearest_gt_median": ("nearest_gt", "median_over_extent"),
    "nearest_gt_p90": ("nearest_gt", "p90_over_extent"),
    "lift_seconds": ("lift_seconds",),
}


def metric_samples(runs, family: str, mode: str, path: tuple[str, ...]) -> list[float]:
    samples = []
    for run in runs:
        value = run["families"][family][mode]
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
        family: {
            mode: {
                name: aggregate(metric_samples(runs, family, mode, path))
                for name, path in SUMMARY_PATHS.items()
            }
            for mode in MODES
        }
        for family in FAMILIES
    }


def metric_mean(summary, family: str, mode: str, metric: str) -> float:
    value = summary[family][mode][metric]["mean"]
    if value is None:
        raise ValueError(f"{family}/{mode}/{metric} has no finite samples")
    return float(value)


def lower_wins(summary, family: str, left: str, right: str, metric: str) -> int:
    left_samples = summary[family][left][metric]["samples"]
    right_samples = summary[family][right][metric]["samples"]
    return sum(
        left_value < right_value for left_value, right_value in zip(left_samples, right_samples)
    )


def metric_gain(summary, family: str, mode: str, metric: str) -> float:
    baseline = metric_mean(summary, family, "all", metric)
    value = metric_mean(summary, family, mode, metric)
    return (baseline - value) / max(abs(baseline), 1e-12)


def family_decision(summary, family: str) -> dict[str, object]:
    primary_metrics = ["heldout_depth_rmse", "source_all_p90"]
    if family == "hybrid":
        primary_metrics.append("source_corrupted_p90")
    gains = {
        metric: metric_gain(summary, family, "leave_one_source_out", metric)
        for metric in primary_metrics
    }
    matched_gains = {
        metric: metric_gain(summary, family, "matched_nonself_dropout", metric)
        for metric in primary_metrics
    }
    all_wins = {
        metric: lower_wins(summary, family, "leave_one_source_out", "all", metric)
        for metric in primary_metrics
    }
    matched_wins = {
        metric: lower_wins(
            summary,
            family,
            "leave_one_source_out",
            "matched_nonself_dropout",
            metric,
        )
        for metric in primary_metrics
    }
    psnr_delta = metric_mean(summary, family, "leave_one_source_out", "test_psnr") - metric_mean(
        summary, family, "all", "test_psnr"
    )
    coverage_delta = metric_mean(summary, family, "leave_one_source_out", "coverage") - metric_mean(
        summary, family, "all", "coverage"
    )
    iou_delta = metric_mean(summary, family, "leave_one_source_out", "alpha_iou") - metric_mean(
        summary, family, "all", "alpha_iou"
    )
    material = (
        gains["heldout_depth_rmse"] >= 0.02
        and gains["source_all_p90"] >= 0.10
        and all_wins["heldout_depth_rmse"] >= 2
        and all_wins["source_all_p90"] >= 2
        and psnr_delta >= -0.10
        and coverage_delta >= -0.02
        and iou_delta >= -0.02
    )
    if family == "hybrid":
        material = (
            material
            and gains["source_corrupted_p90"] >= 0.15
            and all_wins["source_corrupted_p90"] >= 2
        )
    attributed = material and all(
        matched_wins[metric] >= 2 and matched_gains[metric] <= 0.5 * gains[metric]
        for metric in primary_metrics
    )
    return {
        "loso_gain_fraction": gains,
        "matched_dropout_gain_fraction": matched_gains,
        "loso_vs_all_seed_wins": all_wins,
        "loso_vs_matched_seed_wins": matched_wins,
        "loso_vs_all_psnr_delta_db": psnr_delta,
        "loso_vs_all_coverage_delta": coverage_delta,
        "loso_vs_all_alpha_iou_delta": iou_delta,
        "material_effect_pass": material,
        "own_source_attribution_pass": attributed,
    }


def decision(summary) -> dict[str, object]:
    families = {family: family_decision(summary, family) for family in FAMILIES}
    passed = [family for family in FAMILIES if families[family]["own_source_attribution_pass"]]
    return {
        "families": families,
        "attributed_families": passed,
        "both_families_attributed": len(passed) == len(FAMILIES),
        "stop_loso_schedule_sweeps": not passed,
        "next_action": (
            "calibrated_real_replication"
            if len(passed) == len(FAMILIES)
            else "family_scoped_replication"
            if passed
            else "direct_world_frame_position_consistency"
        ),
        "production_default_change_authorized": False,
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
        clean_depths = [depth.detach().clone() for depth in train_scene.gt_depths]
        corrupted_priors, corruption_masks = make_priors(
            clean_depths, "corrupted", seed, args.block_size
        )
        corrupted_depths = [prior.depth.detach().clone() for prior in corrupted_priors]
        diagnostic_priors = [
            AlignedDepthPrior(
                depth=depth,
                confidence=(torch.isfinite(depth) & (depth > 0.05)).to(depth),
            )
            for depth in corrupted_depths
        ]
        seed_result = {
            "seed": seed,
            "global_train_indices": scene.training_views,
            "global_test_indices": scene.testing_views,
            "local_to_global_train_indices": scene.training_views,
            "fit_seconds": fit_seconds,
            "fit_psnr_mean": statistics.fmean(item["final_psnr"] for item in fit_history),
            "fitted_tensors_sha256": fitted_tensor_hash(gaussians2d),
            "corrupted_depths_sha256": depth_tensor_hash(corrupted_depths),
            "invariants": {},
            "families": {},
        }
        for family in FAMILIES:
            seed_result["invariants"][family] = assert_family_invariants(
                family,
                gaussians2d,
                train_scene,
                corrupted_depths,
                args,
                seed,
            )
            seed_result["families"][family] = {}
            for mode in MODES:
                seed_result["families"][family][mode] = evaluate_arm(
                    family,
                    mode,
                    gaussians2d,
                    scene,
                    train_scene,
                    clean_depths,
                    corrupted_depths,
                    diagnostic_priors,
                    corruption_masks,
                    args,
                    seed,
                )
            family_arms = seed_result["families"][family]
            counts = {mode: family_arms[mode]["n_initial"] for mode in MODES}
            if len(set(counts.values())) != 1:
                raise AssertionError(f"{family} final primitive counts differ: {counts}")
            schedules = {mode: family_arms[mode]["target_view_history"] for mode in MODES}
            if any(schedule != schedules["all"] for schedule in schedules.values()):
                raise AssertionError(f"{family} target-view schedules differ across modes")
            if args.output is not None and set(schedules["all"]) != set(train_scene.training_views):
                raise AssertionError(f"{family} did not visit every training view")
            expected_rendered = {
                mode: {
                    item["target_view"]: item["rendered_count"]
                    for item in family_arms[mode]["supervision_diagnostics"]
                }
                for mode in MODES
            }
            for mode in MODES:
                observed = family_arms[mode]["rendered_count_history"]
                expected = [expected_rendered[mode][view] for view in schedules[mode]]
                if observed != expected:
                    raise AssertionError(f"{family}/{mode} changed a frozen render mask")
        runs.append(seed_result)

    summary = summarize(runs)
    for family in FAMILIES:
        for mode in MODES:
            for metric in (
                "heldout_depth_rmse",
                "source_all_p90",
                "source_corrupted_p90",
                "test_psnr",
                "alpha_iou",
                "coverage",
            ):
                if len(summary[family][mode][metric]["samples"]) != len(runs):
                    raise AssertionError(f"{family}/{mode}/{metric} has incomplete samples")
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
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
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
    parser.add_argument("--lift-iters", type=int, default=90)
    parser.add_argument("--rng-check-iters", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gradient-depth-jitter", type=float, default=0.15)
    parser.add_argument("--gradient-depth-prior-lambda", type=float, default=0.001)
    parser.add_argument("--hybrid-depth-jitter", type=float, default=0.02)
    parser.add_argument("--hybrid-depth-prior-lambda", type=float, default=0.01)
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
        "lift_iters": 90,
        "rng_check_iters": 2,
        "lr": 0.1,
        "gradient_depth_jitter": 0.15,
        "gradient_depth_prior_lambda": 0.001,
        "hybrid_depth_jitter": 0.02,
        "hybrid_depth_prior_lambda": 0.01,
        "min_weight": 0.05,
        "block_size": 8,
        "threads": 4,
    }
    return all(getattr(args, key) == value for key, value in expected.items())


def main() -> int:
    args = parse_args()
    if args.output is not None and not is_preregistered_config(args):
        raise ValueError("tracked output requires the exact preregistered configuration")
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"refusing to overwrite official output {args.output}")
    payload = run(args)
    rendered = json.dumps(payload, indent=2, allow_nan=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
