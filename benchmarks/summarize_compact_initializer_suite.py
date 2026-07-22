#!/usr/bin/env python3
"""Build the audited comparison payload from terminal compact-initializer suite artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "runs/all_initializers_frame00008_20260721"
DEFAULT_BEAM_RESULT = ROOT / "benchmarks/results/20260721_beam_fusion_full_frame00008_RESULT.json"
DEFAULT_OUTPUT = ROOT / "benchmarks/results/20260721_all_initializers_frame00008_RESULT.json"

APPLICABILITY = {
    "prospective_compact_arms": {
        "topk": "compact fields and calibrated cameras",
        "dense-merge": "compact fields and calibrated cameras",
        "easy-only": "compact fields and calibrated cameras",
        "splat-sfm": "compact fields and calibrated cameras",
        "field": "compact fields and calibrated cameras",
        "random": "camera-derived bounds only",
    },
    "historical_compact_anchor": {
        "beam-fusion": "compact fields and calibrated cameras",
    },
    "inapplicable_to_compact_only_bundle": {
        "gradient": "requires dense RGB photometric targets",
        "legacy-carve": "requires dense RGB/color-volume samples",
        "depth": "requires RGB depth inference or supplied depth maps",
        "hybrid": "requires depth plus dense RGB photometric targets",
        "classic-sfm": "requires sparse scene points plus RGB for point colors",
    },
    "not_public_arms": {
        "field-placement-fallback": "an emergency fallback is a failure, not an initializer arm",
    },
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _history_count(history: dict[str, Any], step: int) -> int | None:
    values = {int(record[0]): int(record[1]) for record in history.get("n_gaussians", [])}
    return values.get(step)


def _phase_summary(directory: Path) -> dict[str, Any]:
    history_path = directory / "training_history.json"
    fit_path = directory / "fit_complete.json"
    history = _read(history_path)
    fit = _read(fit_path)
    elapsed = history.get("elapsed", [])
    return {
        "directory": str(directory.relative_to(ROOT)),
        "fit_complete": _artifact(fit_path),
        "training_history": _artifact(history_path),
        "segment_optimizer_seconds": 0.0 if not elapsed else float(elapsed[-1][1]),
        "checkpoint_callback_seconds": float(history.get("checkpoint_callback_seconds", 0.0)),
        "peak_vram_gib": float(history.get("peak_vram_gb", 0.0)),
        "n_final_gaussians": int(fit["n_final_gaussians"]),
    }


def _prospective_arm(suite: Path, arm: str, record: dict[str, Any]) -> dict[str, Any]:
    if record.get("state") != "complete":
        return {
            "status": "failed",
            "error_type": record.get("error_type"),
            "error": record.get("error"),
        }
    parent = suite / arm / "fit_0_30000"
    terminal = ROOT / record["terminal_directory"]
    placement_path = parent / "placement.json"
    initial_path = parent / "initial_compact_metrics.json"
    config_path = parent / "config.json"
    initial_ply = parent / "gaussians_init.ply"
    placement = _read(placement_path)
    initial = _read(initial_path)
    config = _read(config_path)
    compact_path = terminal / "compact_metrics.json"
    selection_path = terminal / "model_selection.json"
    compact = _read(compact_path)
    selection = _read(selection_path)
    selected = selection["selected"]
    if compact["model_selection"]["selected_candidate_sha256"] != selected["sha256"]:
        raise RuntimeError(f"compact metrics do not bind selected model for {arm}")
    phase_directories = [ROOT / phase["directory"] for phase in record["phases"]]
    phases = [_phase_summary(directory) for directory in phase_directories]
    parent_history = _read(parent / "training_history.json")
    placement_summary = {
        key: value
        for key, value in placement.items()
        if key
        in {
            "initializer",
            "elapsed_seconds",
            "candidate_count",
            "input_component_count",
            "n_gaussians",
            "n_dense_eligible",
            "n_merged",
            "compact_carve_config",
            "merge_voxel_size",
            "merge_opacity_mode",
            "merge_weight_by_score",
            "cluster_histograms",
            "gate",
            "splat_sfm_config",
            "field_lift_config",
            "semantic_validation",
            "seed",
            "bounds_center",
            "bounds_extent",
            "sampling",
            "color",
            "isotropic_scale",
            "diagnostics",
        }
    }
    return {
        "status": "complete",
        "historical": False,
        "config": {
            key: config[key]
            for key in (
                "initializer",
                "fit_mode",
                "seed",
                "max_tracks",
                "depth_samples",
                "min_views",
                "dense_merge_voxel_size",
                "sfm_min_views",
                "sfm_source_chunk",
                "field_max_tracks",
                "field_max_train_views",
                "iterations",
                "densify_start",
                "densify_stop",
                "densify_every",
                "max_gaussians",
            )
        },
        "placement": placement_summary,
        "initialized_gaussians_3d": int(initial["n_gaussians"]),
        "initial_metrics_all_26_fitted_views": initial["train"],
        "downstream": {
            "gaussians_at_density_stop_15000": _history_count(parent_history, 15_000),
            "selected_gaussians": int(phases[-1]["n_final_gaussians"]),
            "selected_global_step": int(selected["global_step"]),
            "assessed_endpoint": max(int(item["global_step"]) for item in selection["candidates"]),
            "convergence": selection["convergence"],
            "selected_objective": float(selected["objective"]),
            "selected_metrics_all_26_fitted_views": compact["train"],
            "optimizer_seconds_total": sum(phase["segment_optimizer_seconds"] for phase in phases),
            "checkpoint_callback_seconds_total": sum(
                phase["checkpoint_callback_seconds"] for phase in phases
            ),
            "peak_vram_gib_max": max(phase["peak_vram_gib"] for phase in phases),
            "phases": phases,
        },
        "artifacts": {
            "config": _artifact(config_path),
            "placement": _artifact(placement_path),
            "initial_metrics": _artifact(initial_path),
            "initial_ply": _artifact(initial_ply),
            "compact_metrics": _artifact(compact_path),
            "model_selection": _artifact(selection_path),
            "selected_ply": _artifact(terminal / "gaussians_final.ply"),
        },
    }


def _historical_beam(path: Path) -> dict[str, Any]:
    source = _read(path)
    init = source["initializers"]["beam_fusion"]
    fit = source["downstream_fit"]
    selection_path = ROOT / fit["artifacts"]["model_selection"]
    selection = _read(selection_path)
    return {
        "status": "complete",
        "historical": True,
        "initialized_gaussians_3d": init["initialized_gaussians_3d"],
        "placement": {
            "elapsed_seconds": init["placement_seconds"],
            "parameters": init["parameters"],
            "view_pairs": init["view_pairs"],
            "evaluated_ray_pairs": init["evaluated_ray_pairs"],
            "gated_pair_seeds": init["gated_pair_seeds"],
            "unique_input_component_coverage_fraction": init[
                "unique_input_component_coverage_fraction"
            ],
        },
        "initial_metrics_all_26_fitted_views": {
            "psnr_fg": init["metrics_all_26_fitted_views"]["foreground_psnr_db"],
            "psnr_crop": init["metrics_all_26_fitted_views"]["crop_psnr_db"],
            "ssim_crop": init["metrics_all_26_fitted_views"]["crop_ssim"],
            "alpha_iou": init["metrics_all_26_fitted_views"]["alpha_iou"],
            "alpha_inside": init["metrics_all_26_fitted_views"]["alpha_inside"],
            "alpha_outside": init["metrics_all_26_fitted_views"]["alpha_outside"],
        },
        "downstream": {
            "gaussians_at_density_stop_15000": fit["gaussian_count"]["at_density_stop_15000"],
            "selected_gaussians": fit["gaussian_count"]["selected_final"],
            "selected_global_step": fit["selected_global_step"],
            "assessed_endpoint": fit["convergence"]["assessed_endpoint"],
            "convergence": fit["convergence"],
            "selected_objective": selection["selected"]["objective"],
            "selected_metrics_all_26_fitted_views": {
                "psnr_fg": fit["selected_compact_metrics_all_26_fitted_views"][
                    "foreground_psnr_db"
                ],
                "psnr_crop": fit["selected_compact_metrics_all_26_fitted_views"]["crop_psnr_db"],
                "ssim_crop": fit["selected_compact_metrics_all_26_fitted_views"]["crop_ssim"],
                "alpha_iou": fit["selected_compact_metrics_all_26_fitted_views"]["alpha_iou"],
                "alpha_inside": fit["selected_compact_metrics_all_26_fitted_views"]["alpha_inside"],
                "alpha_outside": fit["selected_compact_metrics_all_26_fitted_views"][
                    "alpha_outside"
                ],
            },
            "optimizer_seconds_total": fit["optimizer_elapsed_seconds"]["total"],
            "checkpoint_callback_seconds_total": fit["checkpoint_callback_seconds_total"],
            "peak_vram_gib_max": fit["peak_vram_gib_max"],
        },
        "artifacts": {"historical_result": _artifact(path)},
    }


def _decision(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    successful = {name: arm for name, arm in arms.items() if arm["status"] == "complete"}
    initial_rank = sorted(
        successful,
        key=lambda name: successful[name]["initial_metrics_all_26_fitted_views"]["psnr_fg"],
        reverse=True,
    )
    final_rank = sorted(
        successful,
        key=lambda name: successful[name]["downstream"]["selected_metrics_all_26_fitted_views"][
            "psnr_fg"
        ],
        reverse=True,
    )
    winner = final_rank[0]
    runner_up = final_rank[1]
    winner_arm = successful[winner]["downstream"]
    runner_arm = successful[runner_up]["downstream"]
    psnr_delta = (
        winner_arm["selected_metrics_all_26_fitted_views"]["psnr_fg"]
        - runner_arm["selected_metrics_all_26_fitted_views"]["psnr_fg"]
    )
    objective_reduction = (
        runner_arm["selected_objective"] - winner_arm["selected_objective"]
    ) / runner_arm["selected_objective"]
    material = psnr_delta >= 0.10 and objective_reduction >= 0.0025
    best_psnr = winner_arm["selected_metrics_all_26_fitted_views"]["psnr_fg"]
    best_objective = min(arm["downstream"]["selected_objective"] for arm in successful.values())
    equivalent = [
        name
        for name in final_rank
        if best_psnr
        - successful[name]["downstream"]["selected_metrics_all_26_fitted_views"]["psnr_fg"]
        <= 0.05
        and (successful[name]["downstream"]["selected_objective"] - best_objective) / best_objective
        <= 0.0025
    ]
    return {
        "initial_foreground_psnr_rank": initial_rank,
        "final_foreground_psnr_rank": final_rank,
        "quality_winner": winner,
        "runner_up": runner_up,
        "winner_minus_runner_up_foreground_psnr_db": psnr_delta,
        "winner_vs_runner_up_objective_reduction_fraction": objective_reduction,
        "material_quality_winner": material,
        "practical_equivalence_set": equivalent,
        "default_change_authorized": False,
        "heldout_evidence": False,
    }


def build(suite: Path, beam_result: Path, protocol: Path) -> dict[str, Any]:
    suite_status_path = suite / "suite_status.json"
    status = _read(suite_status_path)
    if not status.get("all_requested_arms_terminal"):
        raise RuntimeError("suite has not reached terminal state")
    arms = {arm: _prospective_arm(suite, arm, record) for arm, record in status["arms"].items()}
    arms["beam-fusion"] = _historical_beam(beam_result)
    payload = {
        "schema": "rtgs.all_initializers_frame00008.result.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": {
            "scene": "frame_00008",
            "development_only": True,
            "all_views_were_fit": True,
            "source_rgb_opened": False,
            "heldout_evidence": False,
            "native_initial_counts_retained": True,
            "quality_rank_is_count_confounded": True,
            "timings_are_nonportable_local_diagnostics": True,
        },
        "applicability": APPLICABILITY,
        "protocol": _artifact(protocol),
        "suite_status": _artifact(suite_status_path),
        "arms": arms,
        "decision": _decision(arms),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--beam-result", type=Path, default=DEFAULT_BEAM_RESULT)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "benchmarks/results/20260721_all_initializers_frame00008_PREREG.md",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build(
        args.suite.resolve(strict=True),
        args.beam_result.resolve(strict=True),
        args.protocol.resolve(strict=True),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(payload["decision"], indent=2))
    print(f"result -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
