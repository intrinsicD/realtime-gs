#!/usr/bin/env python3
"""Init-only screen: calibrated splat-SfM vs carve top-K vs dense all-Gaussian + merge.

Runs the SfM-style RGB-free initializer (:mod:`rtgs.lift.splat_sfm`) beside the two carve arms on
the same inputs, evaluates every initialization against each view's exact 2D teacher render, and
saves metrics JSON, viewer-ready PLYs, and a side-by-side ``rtgs view`` command.

``--synthetic`` runs the dependency-free smoke scene (mechanism check only). ``--bundle <dir>``
runs a saved :class:`ReconstructionInputs` bundle — the calibrated entry point. Init-only metrics
are a screen, not a downstream decision: E2 (2026-07-20) showed init-only rank can invert after
optimization, so any default claim needs the downstream protocol plus a results audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

if __package__ in (None, ""):  # direct script execution: make the repo root importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.compact_init_eval import _build_indexes, _synthetic_inputs

from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.beam_fusion import BeamFusionConfig, fuse_gaussian_beams
from rtgs.lift.compact_carve import CompactCarveConfig, CompactCarveInitializer
from rtgs.lift.compact_init_eval import (
    evaluate_initialization,
    merge_initialization,
    prepare_evaluation_targets,
)
from rtgs.lift.splat_sfm import SplatSfMConfig, structure_from_splats


def run(
    inputs: ReconstructionInputs,
    *,
    n_init_3d: int,
    merge_voxel_size: float,
    out_dir: Path,
    seed: int = 0,
    sfm_config: SplatSfMConfig | None = None,
    fusion_config: BeamFusionConfig | None = None,
) -> dict:
    shared = dict(
        candidate_multiplier=4,
        anchor_mode="component_centers",
        samples_per_ray=48,
        seed=seed,
        min_views=2,
        hull_fraction=0.85,
        coverage_scale=1.0,
        coverage_threshold=0.40,
        color_std_sigma=0.20,
        min_score=0.05,
        init_opacity=0.5,
    )
    topk_config = CompactCarveConfig(n_init_3d=n_init_3d, **shared)
    dense_config = CompactCarveConfig(n_init_3d=1, select_all_eligible=True, **shared)
    indexes = _build_indexes(inputs, topk_config)
    targets = prepare_evaluation_targets(inputs, backends=indexes)

    started = perf_counter()
    topk = CompactCarveInitializer(topk_config).initialize(inputs, backends=indexes)
    topk_seconds = perf_counter() - started

    started = perf_counter()
    dense = CompactCarveInitializer(dense_config).initialize(inputs, backends=indexes)
    dense_merged, _ = merge_initialization(dense, merge_voxel_size)
    dense_seconds = perf_counter() - started

    started = perf_counter()
    sfm = structure_from_splats(inputs, sfm_config)
    sfm_seconds = perf_counter() - started

    started = perf_counter()
    fused = fuse_gaussian_beams(inputs, fusion_config)
    fusion_seconds = perf_counter() - started

    arms = {
        "topk": (topk.gaussians, topk_seconds),
        "dense_merged": (dense_merged, dense_seconds),
        "splat_sfm": (sfm.gaussians, sfm_seconds),
        "beam_fusion": (fused.gaussians, fusion_seconds),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "scene": inputs.name,
        "n_views": inputs.n_views,
        "n_opt_2d_total": int(sum(inputs.n_opt_2d)),
        "arms": {},
        "splat_sfm_diagnostics": sfm.diagnostics,
        "beam_fusion_diagnostics": fused.diagnostics,
    }
    for name, (gaussians, seconds) in arms.items():
        evaluation = evaluate_initialization(inputs, gaussians, targets=targets)
        ply_path = out_dir / f"init_{name}.ply"
        gaussians.save_ply(ply_path)
        report["arms"][name] = {
            "placement_seconds": seconds,
            "ply": str(ply_path),
            **evaluation.as_dict(),
        }
    report["viewer_command"] = (
        f"rtgs view --gaussians {out_dir / 'init_splat_sfm.ply'} "
        f"--initial {out_dir / 'init_topk.ply'}"
    )
    (out_dir / "splat_sfm_screen.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--synthetic", action="store_true", help="use the built-in smoke scene")
    source.add_argument("--bundle", type=Path, default=None, help="saved ReconstructionInputs dir")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--n-init-3d", type=int, default=None, help="top-K control budget")
    parser.add_argument("--merge-voxel", type=float, default=0.06, help="dense merge voxel size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-track-views", type=int, default=2, help="splat-SfM minimum views per track"
    )
    parser.add_argument(
        "--max-reprojection-px", type=float, default=3.0, help="splat-SfM track gate"
    )
    args = parser.parse_args(argv)

    if args.bundle is not None:
        inputs = ReconstructionInputs.load(args.bundle)
        default_topk = max(1, int(sum(inputs.n_opt_2d) // 26))
    else:
        inputs = _synthetic_inputs()
        default_topk = 5
    out_dir = args.out or Path("benchmarks/results") / f"splat_sfm_screen_{inputs.name}"

    report = run(
        inputs,
        n_init_3d=args.n_init_3d if args.n_init_3d is not None else default_topk,
        merge_voxel_size=args.merge_voxel,
        out_dir=out_dir,
        seed=args.seed,
        sfm_config=SplatSfMConfig(
            min_views=args.min_track_views,
            max_reprojection_px=args.max_reprojection_px,
            init_opacity=0.5,  # match the carve arms so photometrics compare placements
        ),
        fusion_config=BeamFusionConfig(
            min_views=args.min_track_views,
            init_opacity=0.5,
        ),
    )
    print(json.dumps(report, indent=2))
    print()
    for name, arm in report["arms"].items():
        print(
            f"{name:>12}: n={arm['n_gaussians']:>6}  "
            f"fg-PSNR {arm['mean_foreground_psnr']:6.2f} dB  "
            f"SSIM {arm['mean_ssim']:.4f}  placement {arm['placement_seconds']:.2f}s"
        )
    diagnostics = report["splat_sfm_diagnostics"]
    print(
        f"\nsplat-SfM: {diagnostics['n_tracks']} tracks "
        f"(lengths {diagnostics['track_length_histogram']}), "
        f"mean reproj {diagnostics['mean_reprojection_px']:.3f}px, "
        f"unmatched/view {diagnostics['unmatched_per_view']}"
    )
    fusion = report["beam_fusion_diagnostics"]
    print(
        f"beam-fusion: {fusion['n_components']} components "
        f"(views {fusion['component_view_histogram']}), "
        f"unmatched/view {fusion['unmatched_per_view']}"
    )
    print("viewer:", report["viewer_command"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
