#!/usr/bin/env python3
"""Initialization-only compact-view comparison: balanced top-K vs dense all-Gaussian + merge.

This is the runnable entry point behind the "weak initialization" question. It scores an
image-free 3D initialization *before* any 3DGS optimization by rendering it through every
calibrated camera and comparing it to that view's exact 2D teacher render (the RGB-free surrogate
for the source image). Two placements are compared on equal footing:

  * ``top-K``  — the current balanced ``n_init_3d`` selection (the sparse default);
  * ``dense``  — retain one carve lift per supported 2D Gaussian across all views, then
                 deduplicate with the voxel-hash moment merge.

It saves an init-only metrics JSON, viewer-ready initial PLYs for both placements, and prints a
side-by-side ``rtgs view`` command. Run ``--synthetic`` for a fast, dependency-free smoke scene, or
``--bundle <dir>`` for a saved :class:`ReconstructionInputs` bundle (e.g. a calibrated frame).

Note: this reports numbers for whatever scene it is given. On the synthetic scene the absolute
values are only a mechanism check; a pipeline-quality decision must use a calibrated frame under
``dataset/`` and go through the results-audit skill.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import torch

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField, GaussianObservationIndex
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.compact_carve import (
    CompactCandidateAudit,
    CompactCarveConfig,
    CompactCarveInitializer,
    CompactInitializationResult,
    CompactPlacementProgress,
)
from rtgs.lift.compact_confidence_gate import (
    ClusterConfidenceConfig,
    gate_merged_initialization,
)
from rtgs.lift.compact_init_eval import (
    InitEvaluationProgress,
    evaluate_initialization,
    merge_initialization,
    prepare_evaluation_targets,
)
from rtgs.lift.compact_refine import LocalDepthRefineConfig, refine_initialization_depths
from rtgs.render import get_rasterizer
from rtgs.render.torch_ref import TorchRasterizer, TorchRenderProgress


def _placement_progress_callback(label: str, *, enabled: bool):
    if not enabled:
        return None

    def callback(progress: CompactPlacementProgress) -> None:
        if progress.phase == "index_built":
            message = (
                f"[{label}] indexes built in {progress.index_build_seconds:.2f}s; "
                f"{progress.total_ray_batches} ray batches"
            )
        elif progress.phase == "ray_batch":
            report_every = max(1, progress.total_ray_batches // 10)
            if (
                progress.completed_ray_batches not in {1, progress.total_ray_batches}
                and progress.completed_ray_batches % report_every != 0
            ):
                return
            message = (
                f"[{label}] placement {progress.completed_ray_batches}/"
                f"{progress.total_ray_batches} batches, {progress.sampled_points:,} points, "
                f"{progress.elapsed_seconds:.1f}s"
            )
        else:
            message = (
                f"[{label}] placement complete: {progress.sampled_points:,} points in "
                f"{progress.elapsed_seconds:.2f}s"
            )
        print(message, file=sys.stderr, flush=True)

    return callback


def _evaluation_progress_callback(label: str, *, enabled: bool):
    if not enabled:
        return None

    def callback(progress: InitEvaluationProgress) -> None:
        if progress.phase == "view_start":
            message = (
                f"[{label}] evaluating view {progress.view_index + 1}/"
                f"{progress.total_views} ({progress.view_name})"
            )
        else:
            visible = (
                "unknown"
                if progress.visible_gaussians is None
                else f"{progress.visible_gaussians:,}"
            )
            message = (
                f"[{label}] completed {progress.view_name} in {progress.view_seconds:.2f}s "
                f"({visible} visible; total {progress.elapsed_seconds:.1f}s)"
            )
        print(message, file=sys.stderr, flush=True)

    return callback


def _torch_render_progress_callback(*, enabled: bool):
    if not enabled:
        return None
    last_bucket = -1

    def callback(progress: TorchRenderProgress) -> None:
        nonlocal last_bucket
        bucket = min(10, progress.completed_rows * 10 // progress.total_rows)
        if bucket < last_bucket:
            last_bucket = -1  # a new view started
        if bucket == last_bucket and progress.completed_rows != progress.total_rows:
            return
        last_bucket = bucket
        print(
            f"[torch-render] {progress.completed_rows:,}/{progress.total_rows:,} rows, "
            f"{progress.visible_gaussians:,} visible, {progress.elapsed_seconds:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    return callback


def _synthetic_inputs(*, image_size: int = 48) -> ReconstructionInputs:
    """A small multi-view scene of colored point targets seen by several cameras."""
    targets = torch.tensor(
        [
            [-0.18, -0.16, 0.0],
            [0.18, -0.16, 0.0],
            [-0.18, 0.16, 0.05],
            [0.18, 0.16, -0.05],
            [0.0, 0.0, 0.12],
        ]
    )
    colors = torch.tensor(
        [
            [0.85, 0.15, 0.15],
            [0.15, 0.80, 0.20],
            [0.15, 0.25, 0.90],
            [0.80, 0.75, 0.15],
            [0.60, 0.20, 0.70],
        ]
    )

    def camera(x: float) -> Camera:
        return Camera.look_at(
            eye=torch.tensor([x, 0.15 * x, -3.0]),
            target=torch.zeros(3),
            width=image_size,
            height=image_size,
            fov_x_deg=55.0,
        )

    cameras = [camera(x) for x in (-0.9, -0.45, 0.0, 0.45, 0.9)]
    observations = []
    for index, cam in enumerate(cameras):
        means, depth = cam.project(targets)
        assert bool((depth > 0).all())
        observations.append(
            GaussianObservationField(
                width=image_size,
                height=image_size,
                means=means,
                log_scales=torch.log(torch.full((len(targets), 2), 1.7)),
                rotations=torch.tensor([0.0, 0.2, -0.15, 0.35, 0.1]),
                colors=colors,
                amplitudes=torch.tensor([0.9, 0.8, 0.7, 1.0, 0.85]),
                view_id=f"view-{index:02d}",
                n_init=len(targets),
            )
        )
    return ReconstructionInputs(
        observations=observations,
        cameras=cameras,
        view_names=[f"view-{index:02d}" for index in range(len(cameras))],
        bounds_hint=(torch.zeros(3), 1.2),
        name="synthetic-init-eval",
    )


def _build_indexes(inputs: ReconstructionInputs, config: CompactCarveConfig):
    return [
        GaussianObservationIndex(
            field,
            tile_size=config.tile_size,
            max_entries=config.max_index_entries_per_view,
            max_candidates=config.max_candidates_per_tile,
            max_query_pairs=config.max_query_pairs,
        )
        for field in inputs.observations
    ]


def run(
    inputs: ReconstructionInputs,
    *,
    n_init_3d: int,
    merge_voxel_size: float,
    out_dir: Path,
    seed: int = 0,
    samples_per_ray: int = 48,
    hull_fraction: float = 0.85,
    min_views: int = 2,
    coverage_scale: float = 1.0,
    coverage_threshold: float = 0.40,
    color_std_sigma: float = 0.20,
    min_score: float = 0.05,
    init_opacity: float = 0.5,
    refine: bool = False,
    rasterizer_name: str = "torch",
    render_device: str = "cpu",
    progress: bool = False,
    confidence_gate: bool = False,
) -> dict:
    shared = dict(
        candidate_multiplier=4,
        anchor_mode="component_centers",
        samples_per_ray=samples_per_ray,
        seed=seed,
        min_views=min_views,
        hull_fraction=hull_fraction,
        coverage_scale=coverage_scale,
        coverage_threshold=coverage_threshold,
        color_std_sigma=color_std_sigma,
        min_score=min_score,
        init_opacity=init_opacity,
    )
    topk_config = CompactCarveConfig(n_init_3d=n_init_3d, **shared)
    dense_config = CompactCarveConfig(n_init_3d=1, select_all_eligible=True, **shared)
    indexes = _build_indexes(inputs, topk_config)  # caps identical between the two configs
    device = torch.device(render_device)
    if rasterizer_name == "gsplat" and device.type != "cuda":
        raise ValueError("the gsplat evaluation backend requires a CUDA render device")
    if rasterizer_name == "torch":
        rasterizer = TorchRasterizer(
            progress_callback=_torch_render_progress_callback(enabled=progress)
        )
    else:
        rasterizer = get_rasterizer(rasterizer_name, device=device)
    stage_seconds: dict[str, float] = {}

    def timed(label: str, function):
        started = perf_counter()
        value = function()
        stage_seconds[label] = perf_counter() - started
        if progress:
            print(
                f"[{label}] completed in {stage_seconds[label]:.2f}s",
                file=sys.stderr,
                flush=True,
            )
        return value

    targets = timed(
        "evaluation_targets",
        lambda: prepare_evaluation_targets(inputs, backends=indexes),
    )

    topk = timed(
        "topk_placement",
        lambda: CompactCarveInitializer(topk_config).initialize(
            inputs,
            backends=indexes,
            progress_callback=_placement_progress_callback("top-K", enabled=progress),
        ),
    )
    topk_eval = timed(
        "topk_evaluation",
        lambda: evaluate_initialization(
            inputs,
            topk.gaussians.to(device),
            backends=indexes,
            rasterizer=rasterizer,
            progress_callback=_evaluation_progress_callback("top-K", enabled=progress),
            targets=targets,
        ),
    )

    dense_audits: list[CompactCandidateAudit] = []
    dense = timed(
        "dense_placement",
        lambda: CompactCarveInitializer(dense_config).initialize(
            inputs,
            backends=indexes,
            candidate_audit_callback=dense_audits.append if confidence_gate else None,
            progress_callback=_placement_progress_callback("dense", enabled=progress),
        ),
    )
    merge_started = perf_counter()
    merged, group = merge_initialization(dense, merge_voxel_size)
    stage_seconds["dense_merge"] = perf_counter() - merge_started
    dense_eval = timed(
        "dense_evaluation",
        lambda: evaluate_initialization(
            inputs,
            merged.to(device),
            backends=indexes,
            rasterizer=rasterizer,
            progress_callback=_evaluation_progress_callback("dense", enabled=progress),
            targets=targets,
        ),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    topk_ply = out_dir / "init_topk.ply"
    dense_ply = out_dir / "init_dense_merged.ply"
    topk.gaussians.save_ply(topk_ply)
    merged.save_ply(dense_ply)

    easy_gated_block: dict | None = None
    easy_ply: Path | None = None
    if confidence_gate:
        if len(dense_audits) != 1:
            raise RuntimeError("confidence gate requires exactly one dense candidate audit")
        gated = timed(
            "confidence_gate",
            lambda: gate_merged_initialization(
                inputs,
                dense,
                dense_audits[0],
                merged,
                group,
                merge_voxel_size=merge_voxel_size,
                config=ClusterConfidenceConfig(),
            ),
        )
        gated_eval = timed(
            "easy_gated_evaluation",
            lambda: evaluate_initialization(
                inputs,
                gated.gaussians.to(device),
                backends=indexes,
                rasterizer=rasterizer,
                progress_callback=_evaluation_progress_callback("easy-gated", enabled=progress),
                targets=targets,
            ),
        )
        easy_ply = out_dir / "init_easy_gated.ply"
        gated.gaussians.save_ply(easy_ply)
        easy_gated_block = {
            **gated_eval.as_dict(),
            "gate": gated.as_dict(),
            "ply": str(easy_ply),
        }

    cluster_view_multiplicity_histogram: dict[str, int] = {}
    for cluster in group.unique().tolist():
        multiplicity = int(dense.lineage.source_view_indices[group == cluster].unique().numel())
        key = str(multiplicity)
        cluster_view_multiplicity_histogram[key] = (
            cluster_view_multiplicity_histogram.get(key, 0) + 1
        )
    cross_view_clusters = sum(
        count
        for multiplicity, count in cluster_view_multiplicity_histogram.items()
        if int(multiplicity) > 1
    )

    refined_block: dict | None = None
    if refine:
        refined = timed(
            "dense_refine",
            lambda: refine_initialization_depths(
                inputs,
                dense,
                LocalDepthRefineConfig(init_opacity=init_opacity),
                backends=indexes,
            ),
        )
        refined_merged, _ = merge_initialization(
            CompactInitializationResult(
                gaussians=refined.gaussians,
                lineage=dense.lineage,
                depths=refined.refined_depths,
                depth_sigmas=dense.depth_sigmas,
                ray_sigmas=dense.ray_sigmas,
                scores=dense.scores,
                diagnostics=dense.diagnostics,
            ),
            merge_voxel_size,
        )
        refined_eval = timed(
            "dense_refined_evaluation",
            lambda: evaluate_initialization(
                inputs,
                refined_merged.to(device),
                backends=indexes,
                rasterizer=rasterizer,
                progress_callback=_evaluation_progress_callback("dense-refined", enabled=progress),
                targets=targets,
            ),
        )
        refined_ply = out_dir / "init_dense_refined_merged.ply"
        refined_merged.save_ply(refined_ply)
        refined_block = {
            "consensus_objective_initial": refined.initial_objective,
            "consensus_objective_refined": refined.refined_objective,
            "mean_absolute_depth_change": refined.diagnostics["mean_absolute_depth_change"],
            "ply": str(refined_ply),
            **refined_eval.as_dict(),
        }

    delta_mean_foreground_psnr = dense_eval.mean_foreground_psnr - topk_eval.mean_foreground_psnr
    per_view_foreground_psnr_delta = [
        {
            "view_index": dense_view.view_index,
            "view_name": dense_view.view_name,
            "dense_minus_topk_db": dense_view.foreground_psnr - topk_view.foreground_psnr,
        }
        for topk_view, dense_view in zip(topk_eval.per_view, dense_eval.per_view, strict=True)
    ]
    worst_view_delta = min(item["dense_minus_topk_db"] for item in per_view_foreground_psnr_delta)
    gaussian_count_ratio = dense_eval.n_gaussians / max(topk_eval.n_gaussians, 1)
    report = {
        "scene": inputs.name,
        "n_views": inputs.n_views,
        "n_opt_2d_total": int(sum(inputs.n_opt_2d)),
        "seed": seed,
        "merge_voxel_size": merge_voxel_size,
        "evaluation_backend": type(rasterizer).__name__,
        "render_device": str(device),
        "fit_window_rendering": True,
        "stage_seconds": stage_seconds,
        "topk": {
            "n_init_3d_requested": n_init_3d,
            **topk_eval.as_dict(),
        },
        "dense_merged": {
            "lifted": int(dense.gaussians.n),
            "cross_view_correspondence_clusters": cross_view_clusters,
            "cluster_view_multiplicity_histogram": (cluster_view_multiplicity_histogram),
            **dense_eval.as_dict(),
        },
        "easy_gated": easy_gated_block,
        "delta_mean_foreground_psnr": delta_mean_foreground_psnr,
        "per_view_foreground_psnr_delta": per_view_foreground_psnr_delta,
        "preregistered_e1_decision": {
            "mean_gain_at_least_0_5_db": delta_mean_foreground_psnr >= 0.5,
            "no_view_regresses_more_than_0_25_db": worst_view_delta >= -0.25,
            "gaussian_count_within_2x": gaussian_count_ratio <= 2.0,
            "worst_view_delta_db": worst_view_delta,
            "gaussian_count_ratio": gaussian_count_ratio,
            "dense_is_better_init": (
                delta_mean_foreground_psnr >= 0.5
                and worst_view_delta >= -0.25
                and gaussian_count_ratio <= 2.0
            ),
        },
        "dense_refined_merged": refined_block,
        "artifacts": {
            "topk_ply": str(topk_ply),
            "dense_merged_ply": str(dense_ply),
            "easy_gated_ply": None if easy_ply is None else str(easy_ply),
        },
        "viewer_command": (f".venv/bin/rtgs view --gaussians {dense_ply} --initial {topk_ply}"),
    }
    (out_dir / "init_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--synthetic", action="store_true", help="use the built-in smoke scene")
    source.add_argument("--bundle", type=Path, default=None, help="saved ReconstructionInputs dir")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--n-init-3d", type=int, default=None, help="top-K control budget")
    parser.add_argument("--merge-voxel", type=float, default=0.06, help="merge voxel size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rasterizer",
        choices=("torch", "gsplat"),
        default="torch",
        help="3D evaluation backend; torch remains the correctness default",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="3D evaluation device (default: cpu for torch, cuda for gsplat)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="also apply and evaluate the frozen correspondence-confidence gate",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="also run the correspondence-free local 4-dof depth refine (prototype)",
    )
    args = parser.parse_args(argv)

    if args.bundle is not None:
        inputs = ReconstructionInputs.load(args.bundle, strict=True)
        default_topk = max(1, int(sum(inputs.n_opt_2d) // 26))
    else:
        inputs = _synthetic_inputs()
        default_topk = 5
    n_init_3d = args.n_init_3d if args.n_init_3d is not None else default_topk
    out_dir = args.out or Path("benchmarks/results") / f"init_eval_{inputs.name}"
    render_device = args.device or ("cuda" if args.rasterizer == "gsplat" else "cpu")

    report = run(
        inputs,
        n_init_3d=n_init_3d,
        merge_voxel_size=args.merge_voxel,
        out_dir=out_dir,
        seed=args.seed,
        refine=args.refine,
        rasterizer_name=args.rasterizer,
        render_device=render_device,
        progress=not args.quiet,
        confidence_gate=args.gate,
    )
    print(json.dumps(report, indent=2))
    print(
        "\ntop-K   mean fg-PSNR {:.2f} dB over {} gaussians".format(
            report["topk"]["mean_foreground_psnr"], report["topk"]["n_gaussians"]
        )
    )
    print(
        "dense   mean fg-PSNR {:.2f} dB over {} gaussians "
        "({} cross-view correspondence clusters)".format(
            report["dense_merged"]["mean_foreground_psnr"],
            report["dense_merged"]["n_gaussians"],
            report["dense_merged"]["cross_view_correspondence_clusters"],
        )
    )
    print("Δ mean fg-PSNR (dense - top-K): {:+.2f} dB".format(report["delta_mean_foreground_psnr"]))
    if report["easy_gated"] is not None:
        easy = report["easy_gated"]
        print(
            "gate    mean fg-PSNR {:.2f} dB over {} easy gaussians "
            "({} dense clusters dropped)".format(
                easy["mean_foreground_psnr"],
                easy["n_gaussians"],
                easy["gate"]["dropped_count"],
            )
        )
    if report["dense_refined_merged"] is not None:
        refined = report["dense_refined_merged"]
        print(
            "refine  consensus objective {:.4f} -> {:.4f}, mean |Δdepth| {:.4f}; "
            "merged fg-PSNR {:.2f} dB over {} gaussians".format(
                refined["consensus_objective_initial"],
                refined["consensus_objective_refined"],
                refined["mean_absolute_depth_change"],
                refined["mean_foreground_psnr"],
                refined["n_gaussians"],
            )
        )
    print("\nviewer:", report["viewer_command"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
