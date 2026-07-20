#!/usr/bin/env python3
"""Paired CPU ablation for depth-lift covariance construction.

This deliberately sits outside the canonical all-lifter benchmark: it reuses one set of
2D fits per synthetic seed, tunes the scalar isotropic control on training views only,
and evaluates covariance modes on a strict held-out camera split.
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
import torch.nn.functional as F

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.data.synthetic import make_synthetic_scene
from rtgs.depth.base import DepthPrediction
from rtgs.image2gs.fit import FitConfig, fit_views
from rtgs.lift.base import bilinear_sample, minor_axis_sigma_ray
from rtgs.lift.depth import DepthLifter
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.torch_ref import TorchRasterizer

MODES = ("isotropic", "footprint", "surface")


class ListDepthBackend:
    """Serve a fixed list of metric maps in view order."""

    def __init__(self, depths: list[torch.Tensor]):
        self.depths = depths
        self.cursor = 0

    def predict(self, image: torch.Tensor) -> DepthPrediction:
        depth = self.depths[self.cursor]
        self.cursor += 1
        if self.cursor == len(self.depths):
            self.cursor = 0
        return DepthPrediction(depth=depth, kind="metric")


def git_state(root: Path) -> tuple[str, bool]:
    """Return short revision and whether the worktree differs."""
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
    """Bind a result to the exact research implementation, including dirty files."""
    paths = (
        Path("benchmarks/depth_covariance_ablation.py"),
        Path("src/rtgs/lift/base.py"),
        Path("src/rtgs/lift/depth.py"),
    )
    return {str(path): hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def perturb_depths(depths: list[torch.Tensor], seed: int) -> list[torch.Tensor]:
    """Apply deterministic masked 3x3 blur plus 2% multiplicative noise."""
    perturbed = []
    for view_index, depth in enumerate(depths):
        valid = depth > 0
        values = depth[None, None]
        weights = valid.to(depth)[None, None]
        numerator = F.avg_pool2d(values * weights, 3, stride=1, padding=1)
        denominator = F.avg_pool2d(weights, 3, stride=1, padding=1).clamp_min(1e-8)
        smooth = (numerator / denominator)[0, 0]
        generator = torch.Generator().manual_seed(seed * 10_000 + view_index)
        noise = 0.02 * torch.randn(depth.shape, generator=generator)
        noisy = (smooth * (1.0 + noise)).clamp_min(0.05)
        perturbed.append(torch.where(valid, noisy, torch.zeros_like(noisy)))
    return perturbed


def subset_gaussians(g2d: Gaussians2D, keep: torch.Tensor) -> Gaussians2D:
    return Gaussians2D(
        xy=g2d.xy[keep],
        chol=g2d.chol[keep],
        color=g2d.color[keep],
        weight=g2d.weight[keep],
    )


def isotropic_reference_sigma(
    gaussians2d: list[Gaussians2D], cameras, depths: list[torch.Tensor]
) -> float:
    """RMS minor-footprint sigma used only to center the train-only scalar sweep."""
    sigmas = []
    for g2d, camera, depth_map in zip(gaussians2d, cameras, depths):
        z = bilinear_sample(depth_map, g2d.xy)
        keep = torch.isfinite(z) & (z > 0.05) & (g2d.weight > 0.05)
        if bool(keep.any()):
            selected = subset_gaussians(g2d, keep)
            sigmas.append(minor_axis_sigma_ray(camera, selected, z[keep]))
    if not sigmas:
        raise ValueError("no valid depth samples for isotropic sigma reference")
    all_sigmas = torch.cat(sigmas)
    return float(all_sigmas.square().mean().sqrt())


def make_lifter(
    mode: str,
    depths: list[torch.Tensor],
    merge: bool,
    normal_thickness: float,
    isotropic_sigma: float | None,
    robust_depth_gradients: bool,
) -> DepthLifter:
    kwargs = {}
    if mode == "isotropic":
        kwargs["isotropic_sigma"] = isotropic_sigma
    return DepthLifter(
        backend=ListDepthBackend(depths),
        covariance_mode=mode,
        normal_thickness=normal_thickness,
        robust_depth_gradients=robust_depth_gradients,
        merge=merge,
        **kwargs,
    )


def render_diagnostics(scene, gaussians, indices: list[int]) -> dict[str, float]:
    """Held-out geometry/silhouette diagnostics available only on synthetic scenes."""
    renderer = TorchRasterizer()
    depth_sq_error = 0.0
    depth_pixels = 0
    ious = []
    coverages = []
    with torch.no_grad():
        for index in indices:
            pred = renderer.render(gaussians, scene.cameras[index])
            truth = renderer.render(scene.gt_gaussians, scene.cameras[index])
            pred_fg = pred.alpha > 0.05
            true_fg = truth.alpha > 0.05
            intersection = pred_fg & true_fg
            union = pred_fg | true_fg
            ious.append(float(intersection.sum() / union.sum().clamp_min(1)))
            coverages.append(float(intersection.sum() / true_fg.sum().clamp_min(1)))
            if bool(intersection.any()):
                pred_depth = pred.depth / pred.alpha.clamp_min(1e-8)
                true_depth = truth.depth / truth.alpha.clamp_min(1e-8)
                depth_sq_error += float(
                    (pred_depth[intersection] - true_depth[intersection]).square().sum()
                )
                depth_pixels += int(intersection.sum())
    return {
        "alpha_iou": statistics.fmean(ious),
        "foreground_coverage": statistics.fmean(coverages),
        "depth_rmse_intersection": (depth_sq_error / max(depth_pixels, 1)) ** 0.5,
    }


def covariance_diagnostics(gaussians, extent: float) -> dict[str, float]:
    evals = torch.linalg.eigvalsh(gaussians.covariance()).clamp_min(1e-12)
    condition = evals[:, -1] / evals[:, 0]
    scales = gaussians.scales
    return {
        "condition_median": float(condition.median()),
        "condition_p95": float(torch.quantile(condition, 0.95)),
        "condition_p99": float(torch.quantile(condition, 0.99)),
        "max_scale_over_extent": float(scales.max() / extent),
        "p99_scale_over_extent": float(torch.quantile(scales.reshape(-1), 0.99) / extent),
    }


def tune_isotropic_sigma(
    gaussians2d,
    train_scene,
    depths,
    factors: list[float],
    merge: bool,
    normal_thickness: float,
) -> tuple[float, dict[str, float], float]:
    reference = isotropic_reference_sigma(gaussians2d, train_scene.cameras, depths)
    renderer = TorchRasterizer()
    scores = {}
    for factor in factors:
        sigma = reference * factor
        init = make_lifter("isotropic", depths, merge, normal_thickness, sigma, False).lift(
            gaussians2d, train_scene
        )
        metrics = Trainer.evaluate_metrics(
            train_scene, init, renderer, indices=train_scene.training_views
        )
        scores[f"{factor:g}"] = metrics["psnr"]
    best_factor = max(factors, key=lambda factor: scores[f"{factor:g}"])
    return reference * best_factor, scores, reference


def evaluate_mode(
    mode,
    gaussians2d,
    scene,
    train_scene,
    depths,
    args,
    isotropic_sigma,
):
    renderer = TorchRasterizer()
    started = time.perf_counter()
    init = make_lifter(
        mode,
        depths,
        args.merge,
        args.normal_thickness,
        isotropic_sigma,
        args.robust_depth_gradients,
    ).lift(gaussians2d, train_scene)
    lift_seconds = time.perf_counter() - started
    _, extent = scene.center_and_extent()
    result = {
        "n_init": init.n,
        "lift_seconds": lift_seconds,
        "init_train": Trainer.evaluate_metrics(scene, init, renderer, indices=scene.training_views),
        "init_test": Trainer.evaluate_metrics(scene, init, renderer, indices=scene.testing_views),
        "init_test_diagnostics": render_diagnostics(scene, init, scene.testing_views),
        "covariance": covariance_diagnostics(init, extent),
    }
    final = init
    if args.refine_iters:
        train_config = TrainConfig(
            iterations=args.refine_iters,
            device="cpu",
            rasterizer="torch",
            densify=args.densify,
            target_sh_degree=0,
            eval_every=max(args.refine_iters // 3, 1),
            seed=args.current_seed,
        )
        started = time.perf_counter()
        final, history = Trainer(train_config).train(scene, init)
        result["refine_seconds"] = time.perf_counter() - started
        result["refine_history"] = {
            key: history[key]
            for key in ("psnr", "elapsed", "n_gaussians", "density_stats", "density_strategy")
        }
        result["n_final"] = final.n
        result["final_train"] = Trainer.evaluate_metrics(
            scene, final, renderer, indices=scene.training_views
        )
        result["final_test"] = Trainer.evaluate_metrics(
            scene, final, renderer, indices=scene.testing_views
        )
        result["final_test_diagnostics"] = render_diagnostics(scene, final, scene.testing_views)
    return result, init


def summarize(runs: list[dict], modes: list[str]) -> dict:
    summary = {}
    for mode in modes:
        mode_runs = [run["modes"][mode] for run in runs]
        values = {
            "init_test_psnr": [run["init_test"]["psnr"] for run in mode_runs],
            "init_test_ssim": [run["init_test"]["ssim"] for run in mode_runs],
            "condition_p99": [run["covariance"]["condition_p99"] for run in mode_runs],
        }
        if "final_test" in mode_runs[0]:
            values["final_test_psnr"] = [run["final_test"]["psnr"] for run in mode_runs]
            values["final_test_ssim"] = [run["final_test"]["ssim"] for run in mode_runs]
        summary[mode] = {
            name: {
                "mean": statistics.fmean(samples),
                "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                "samples": samples,
            }
            for name, samples in values.items()
        }
    return summary


def run(args) -> dict:
    root = Path(__file__).resolve().parent.parent
    revision, dirty = git_state(root)
    torch.set_num_threads(args.threads)
    runs = []
    for seed in args.seeds:
        args.current_seed = seed
        torch.manual_seed(seed)
        scene = make_synthetic_scene(
            n_gaussians=args.gt_gaussians,
            n_cameras=args.views,
            image_size=args.image_size,
            seed=seed,
        )
        if args.test_every < 2:
            raise ValueError("--test-every must be at least 2")
        scene.test_indices = [
            index for index in range(args.views) if index % args.test_every == args.test_every - 1
        ]
        if not scene.test_indices:
            raise ValueError("test split is empty; increase --views or change --test-every")
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
        gaussians2d, fit_history = fit_views(
            train_scene.images, fit_config, seed=seed, masks=train_scene.masks
        )
        fit_seconds = time.perf_counter() - fit_started
        clean_depths = [depth.clone() for depth in train_scene.gt_depths]
        depths = clean_depths if args.condition == "clean" else perturb_depths(clean_depths, seed)
        isotropic_sigma, isotropic_scores, reference_sigma = tune_isotropic_sigma(
            gaussians2d,
            train_scene,
            depths,
            args.isotropic_factors,
            args.merge,
            args.normal_thickness,
        )
        seed_result = {
            "seed": seed,
            "train_indices": scene.training_views,
            "test_indices": scene.testing_views,
            "fit_seconds": fit_seconds,
            "fit_psnr_mean": statistics.fmean(item["final_psnr"] for item in fit_history),
            "isotropic_reference_sigma": reference_sigma,
            "isotropic_selected_sigma": isotropic_sigma,
            "isotropic_train_sweep_psnr": isotropic_scores,
            "modes": {},
        }
        control = None
        for mode in args.modes:
            mode_result, init = evaluate_mode(
                mode,
                gaussians2d,
                scene,
                train_scene,
                depths,
                args,
                isotropic_sigma,
            )
            if not args.merge:
                if control is None:
                    control = init
                else:
                    if not torch.allclose(init.means, control.means):
                        raise AssertionError("covariance arms changed lifted means")
                    if not torch.equal(init.opacity, control.opacity) or init.n != control.n:
                        raise AssertionError("covariance arms changed opacity or primitive count")
                    if not torch.equal(init.sh, control.sh):
                        raise AssertionError("covariance arms changed spherical harmonics")
            seed_result["modes"][mode] = mode_result
        runs.append(seed_result)
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
                key: value
                for key, value in vars(args).items()
                if key != "current_seed" and not isinstance(value, Path)
            },
        },
        "runs": runs,
        "summary": summarize(runs, args.modes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--condition", choices=("clean", "blur-noise"), default="clean")
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--test-every", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--gt-gaussians", type=int, default=40)
    parser.add_argument("--fit-gaussians", type=int, default=150)
    parser.add_argument("--fit-iters", type=int, default=120)
    parser.add_argument("--refine-iters", type=int, default=0)
    parser.add_argument("--merge", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--densify", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--normal-thickness", type=float, default=0.15)
    parser.add_argument(
        "--robust-depth-gradients", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--isotropic-factors", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
