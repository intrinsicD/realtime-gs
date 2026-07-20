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
from pathlib import Path

import torch

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField, GaussianObservationIndex
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.compact_carve import (
    CompactCarveConfig,
    CompactCarveInitializer,
    CompactInitializationResult,
)
from rtgs.lift.compact_init_eval import evaluate_initialization, merge_initialization
from rtgs.lift.compact_refine import LocalDepthRefineConfig, refine_initialization_depths


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

    topk = CompactCarveInitializer(topk_config).initialize(inputs, backends=indexes)
    topk_eval = evaluate_initialization(inputs, topk.gaussians, backends=indexes)

    dense = CompactCarveInitializer(dense_config).initialize(inputs, backends=indexes)
    merged, group = merge_initialization(dense, merge_voxel_size)
    dense_eval = evaluate_initialization(inputs, merged, backends=indexes)

    out_dir.mkdir(parents=True, exist_ok=True)
    topk_ply = out_dir / "init_topk.ply"
    dense_ply = out_dir / "init_dense_merged.ply"
    topk.gaussians.save_ply(topk_ply)
    merged.save_ply(dense_ply)

    cross_view_clusters = 0
    for cluster in group.unique().tolist():
        if dense.lineage.source_view_indices[group == cluster].unique().numel() > 1:
            cross_view_clusters += 1

    refined_block: dict | None = None
    if refine:
        # Correspondence-free local 4-dof refine between lift and merge (prototype). Reported
        # honestly: the consensus objective is optimized, but geometry may drift because
        # correspondence-free consensus rewards coverage, not the exact surface.
        refined = refine_initialization_depths(
            inputs, dense, LocalDepthRefineConfig(init_opacity=init_opacity), backends=indexes
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
        refined_eval = evaluate_initialization(inputs, refined_merged, backends=indexes)
        refined_ply = out_dir / "init_dense_refined_merged.ply"
        refined_merged.save_ply(refined_ply)
        refined_block = {
            "consensus_objective_initial": refined.initial_objective,
            "consensus_objective_refined": refined.refined_objective,
            "mean_absolute_depth_change": refined.diagnostics["mean_absolute_depth_change"],
            "ply": str(refined_ply),
            **refined_eval.as_dict(),
        }

    report = {
        "scene": inputs.name,
        "n_views": inputs.n_views,
        "n_opt_2d_total": int(sum(inputs.n_opt_2d)),
        "merge_voxel_size": merge_voxel_size,
        "topk": {
            "n_init_3d_requested": n_init_3d,
            **topk_eval.as_dict(),
        },
        "dense_merged": {
            "lifted": int(dense.gaussians.n),
            "cross_view_correspondence_clusters": cross_view_clusters,
            **dense_eval.as_dict(),
        },
        "delta_mean_foreground_psnr": (
            dense_eval.mean_foreground_psnr - topk_eval.mean_foreground_psnr
        ),
        "dense_refined_merged": refined_block,
        "artifacts": {"topk_ply": str(topk_ply), "dense_merged_ply": str(dense_ply)},
        "viewer_command": (f"rtgs view --gaussians {dense_ply} --initial {topk_ply}"),
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
        "--refine",
        action="store_true",
        help="also run the correspondence-free local 4-dof depth refine (prototype)",
    )
    args = parser.parse_args(argv)

    if args.bundle is not None:
        inputs = ReconstructionInputs.load(args.bundle)
        default_topk = max(1, int(sum(inputs.n_opt_2d) // 26))
    else:
        inputs = _synthetic_inputs()
        default_topk = 5
    n_init_3d = args.n_init_3d if args.n_init_3d is not None else default_topk
    out_dir = args.out or Path("benchmarks/results") / f"init_eval_{inputs.name}"

    report = run(
        inputs,
        n_init_3d=n_init_3d,
        merge_voxel_size=args.merge_voxel,
        out_dir=out_dir,
        seed=args.seed,
        refine=args.refine,
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
