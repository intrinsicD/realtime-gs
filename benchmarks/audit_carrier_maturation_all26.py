#!/usr/bin/env python3
"""Independent scientist-pass checks for the all-26-view carrier-maturation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/carrier_maturation_all26_frame00008_20260727"
DEFAULT_CRASH_PARTIAL = (
    ROOT / "runs/carrier_maturation_all26_frame00008_20260727_crash_partial_step005000"
)
DEFAULT_OUTPUT = ROOT / "benchmarks/results/20260727_carrier_maturation_all26_AUDIT.json"
COMPACT_ROOT = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d"

EXPECTED_VIEWS = (
    "C0001",
    "C0004",
    "C0005",
    "C0006",
    "C0008",
    "C0009",
    "C0012",
    "C0014",
    "C0018",
    "C0019",
    "C0020",
    "C0021",
    "C0022",
    "C0025",
    "C0026",
    "C0028",
    "C0029",
    "C0030",
    "C0031",
    "C0034",
    "C0037",
    "C0039",
    "C1000",
    "C1001",
    "C1002",
    "C1004",
)
EXPECTED_PROTOCOL_SHA256 = "2becc1cf1f48f3d78bf85f8859b778d861d2f0325732335dbe9aea1bc0e212ef"
EXPECTED_IMPLEMENTATION_SHA256 = {
    "benchmarks/carrier_maturation_all26.py": (
        "e2e1710038029462a50de55217429db6813131c7f483593a29436ac4c928e545"
    ),
    "benchmarks/full_compact_reconstruction.py": (
        "9ee2688f5d4f18c46790cd572118503bb11d83b86d821ffe749930b1eb8be722"
    ),
    "src/rtgs/lift/beam_fusion.py": (
        "575c12fdb59ad7a430178ed5899eb9d546cddc965f50617eeed0b40fe9ca2e12"
    ),
    "src/rtgs/lift/carrier_refinement.py": (
        "bbd50a727da29663e22415376c8cabf269bf561e929a76ae86a4360dbb5590f5"
    ),
    "src/rtgs/optim/carrier_schedule.py": (
        "97c089c683ba0f86bafa302f674613a34d222b8faca5d697a6366f9eb1a8d17c"
    ),
    "src/rtgs/optim/density.py": (
        "a18be4d1b425177b74db8fb4ef814e53f4b246450d3e8f5f149962328110680a"
    ),
    "src/rtgs/optim/trainer.py": (
        "228a5269d02fe33e8d5981c1fd83ee79211b24d485d10bd6be59b13ff2432fed"
    ),
    "src/rtgs/visualize.py": ("0cd822b475fe7a8cf4e8738d28d0ac27f1c06b4bb7cd30f5592b4eff2c5b63d7"),
}
EXPECTED_PHASES = (
    "fixed_topology",
    "clone_recovery_1",
    "clone_recovery_2",
    "clone_recovery_3",
    "higher_sh",
    "standard_growth",
    "standard_settle",
)
EXPECTED_STAGE_NAMES = (
    "beam",
    "covariance_repair",
    "opacity_repair",
    "sh0_repair",
    "fixed_topology_converged",
    "clone_wave_1",
    "clone_wave_1_converged",
    "clone_wave_2",
    "clone_wave_2_converged",
    "clone_wave_3",
    "clone_wave_3_converged",
    "higher_sh_converged",
    "standard_growth_complete",
    "standard_converged",
)
MISLABELLED_BOUNDARIES = (
    "clone_wave_1_converged",
    "clone_wave_2_converged",
    "clone_wave_3_converged",
    "higher_sh_converged",
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _load_list(path: Path) -> list[Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError(f"{path} does not contain a JSON array")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_tree(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _ply_vertex_count(path: Path) -> int:
    with path.open("rb") as stream:
        for raw_line in stream:
            line = raw_line.decode("ascii", errors="strict").strip()
            match = re.fullmatch(r"element vertex (\d+)", line)
            if match:
                return int(match.group(1))
            if line == "end_header":
                break
    raise ValueError(f"{path} has no PLY vertex count")


def _close(left: float, right: float, *, tolerance: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _source_hashes_exact(source_binding: dict[str, Any]) -> bool:
    if tuple(sorted(source_binding)) != EXPECTED_VIEWS:
        return False
    for view in EXPECTED_VIEWS:
        for kind in ("rgb", "mask"):
            record = source_binding[view][kind]
            path = Path(record["path"])
            if not path.is_file():
                return False
            if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
                return False
    return True


def _compact_hashes_exact(compact_manifest: dict[str, Any]) -> bool:
    records = compact_manifest.get("views", [])
    if tuple(record["view_id"] for record in records) != EXPECTED_VIEWS:
        return False
    for record in records:
        path = COMPACT_ROOT / record["path"]
        if record["n_gaussians"] != 5000 or not record["has_alpha"]:
            return False
        if not path.is_file():
            return False
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            return False
    return True


def _phase_protocol_exact(phases: list[dict[str, Any]]) -> bool:
    by_name = {phase["name"]: phase for phase in phases}
    if tuple(by_name) != EXPECTED_PHASES:
        return False
    plateau_specs = {
        "fixed_topology": (30000, 4000, 6, 0.01, 0),
        "clone_recovery_1": (15000, 3000, 5, 0.01, 0),
        "clone_recovery_2": (15000, 3000, 5, 0.01, 0),
        "clone_recovery_3": (15000, 3000, 5, 0.01, 0),
        "higher_sh": (15000, 3000, 5, 0.01, 3),
        "standard_settle": (40000, 5000, 6, 0.005, 3),
    }
    for name, (maximum, minimum, patience, delta, degree) in plateau_specs.items():
        config = by_name[name]["config"]
        if (
            config["iterations"] != maximum
            or config["plateau_min_iterations"] != minimum
            or config["plateau_patience_evals"] != patience
            or not _close(config["plateau_min_delta"], delta)
            or config["target_sh_degree"] != degree
            or config["densify"]
            or config["eval_every"] != 1000
            or config["rasterizer"] != "gsplat"
            or config["device"] != "cuda"
            or not config["packed"]
            or not config["antialiased"]
            or not config["stream_scene_from_cpu"]
        ):
            return False
    higher = by_name["higher_sh"]["config"]
    if any(higher[key] != 0 for key in ("lr_means", "lr_quats", "lr_scales", "lr_opacity")):
        return False
    growth = by_name["standard_growth"]["config"]
    density = growth["density"]
    if (
        growth["iterations"] != 30000
        or not growth["densify"]
        or growth["target_sh_degree"] != 3
        or growth["plateau_patience_evals"] is not None
        or density["start_iter"] != 500
        or density["stop_iter"] != 15000
        or density["every"] != 100
        or density["max_gaussians"] != 100000
        or not _close(density["grad_threshold"], 2e-4)
        or not _close(density["split_scale_frac"], 0.01)
        or not _close(density["split_factor"], 1.6)
        or not _close(density["prune_opacity"], 0.005)
        or not _close(density["prune_scale_frac"], 0.1)
        or density["opacity_reset_every"] != 3000
        or not _close(density["opacity_reset_value"], 0.011)
    ):
        return False
    settle = by_name["standard_settle"]["config"]
    expected_lrs = {
        "lr_means": 1e-7,
        "lr_quats": 1.5625e-5,
        "lr_scales": 7.8125e-5,
        "lr_opacity": 7.8125e-4,
        "lr_sh": 3.90625e-5,
        "lr_sh_rest": 1.953125e-6,
    }
    return all(_close(settle[key], value) for key, value in expected_lrs.items())


def _phase_histories_exact(run: Path, phases: list[dict[str, Any]]) -> bool:
    for phase in phases:
        history = _load_object(run / "phase_histories" / f"{phase['name']}.json")
        executed = phase["executed_iterations"]
        if history["executed_iterations"] != executed:
            return False
        if len(history["loss"]) != executed or len(history["loss_terms"]) != executed:
            return False
        if len(history["sampled_train_views"]) != executed:
            return False
        if not set(history["sampled_train_views"]) <= set(range(26)):
            return False
        if history["stop_reason"] != phase["stop_reason"]:
            return False
        if not _finite_tree(history):
            return False
    return True


def _metric_and_model_bindings_exact(run: Path, stages: list[dict[str, Any]]) -> bool:
    for stage in stages:
        metrics = _load_object(run / stage["metrics"])
        if metrics["name"] != stage["name"] or metrics["n_gaussians"] != stage["n_gaussians"]:
            return False
        model = run / stage["model"]
        if _sha256(model) != stage["model_sha256"]:
            return False
        if _ply_vertex_count(model) != stage["n_gaussians"]:
            return False
        for target, prefix in (
            ("native_masked_rgb", "native"),
            ("compact_teacher", "compact"),
        ):
            target_metrics = metrics[target]
            per_view = target_metrics["per_view"]
            if tuple(sorted(per_view)) != EXPECTED_VIEWS or target_metrics["view_count"] != 26:
                return False
            for metric in (
                "psnr_fg",
                "psnr_crop",
                "psnr_full",
                "ssim_crop",
                "alpha_iou",
                "alpha_inside",
                "alpha_outside",
                "lpips_crop_max512",
            ):
                recomputed = fmean(float(per_view[view][metric]) for view in EXPECTED_VIEWS)
                recorded = float(target_metrics["aggregate"][metric]["mean"])
                if not _close(recomputed, recorded):
                    return False
            if not _close(
                stage[f"{prefix}_psnr_fg_mean"],
                target_metrics["aggregate"]["psnr_fg"]["mean"],
            ):
                return False
    return True


def _checkpoint_bindings_exact(run: Path, checkpoints: list[dict[str, Any]]) -> bool:
    for checkpoint in checkpoints:
        path = run / checkpoint["artifact"]
        if _sha256(path) != checkpoint["sha256"]:
            return False
        if _ply_vertex_count(path) != checkpoint["n_gaussians"]:
            return False
        if f"{checkpoint['global_step']:06d}" not in path.name:
            return False
    return True


def _clone_waves_exact(clones: list[dict[str, Any]]) -> bool:
    if len(clones) != 3:
        return False
    expected = ((5000, 10000), (10000, 20000), (20000, 40000))
    for wave, (record, (n_before, n_after)) in enumerate(zip(clones, expected), start=1):
        stats = record["density_stats"]
        if (
            record["wave"] != wave
            or record["n_before"] != n_before
            or record["n_after"] != n_after
            or not record["parents_survive"]
            or record["max_abs_local_normal_displacement"] > 2e-7
            or len(stats) != 1
            or stats[0]["cloned"] != n_before
            or stats[0]["split"] != 0
            or stats[0]["pruned"] != 0
            or not record["config"]["clone_tangent_only"]
            or not record["config"]["clone_all"]
            or not record["config"]["clone_only"]
        ):
            return False
    return True


def audit(run: Path) -> dict[str, Any]:
    summary = _load_object(run / "summary.json")
    manifest = _load_object(run / "run_manifest.json")
    crash_partial_manifest = _load_object(DEFAULT_CRASH_PARTIAL / "run_manifest.json")
    source_binding = _load_object(run / "source_binding.json")
    compact_manifest = _load_object(COMPACT_ROOT / "manifest.json")
    comparison_manifest = _load_object(run / "comparison_manifest.json")
    clone_receipts = _load_list(run / "clone_receipts.json")
    repair = _load_object(run / "repair_diagnostics.json")
    index_html = (run / "index.html").read_text()
    phases = summary["phases"]
    stages = summary["stages"]
    phase_names = tuple(phase["name"] for phase in phases)
    stage_names = tuple(stage["name"] for stage in stages)
    phase_by_name = {phase["name"]: phase for phase in phases}
    stage_by_name = {stage["name"]: stage for stage in stages}

    implementation_hashes = {path: _sha256(ROOT / path) for path in EXPECTED_IMPLEMENTATION_SHA256}
    protocol_path = ROOT / manifest["protocol"]["path"]
    raw_source_exact = _source_hashes_exact(source_binding)
    compact_source_exact = _compact_hashes_exact(compact_manifest)

    phase_continuity = bool(phases) and phases[0]["global_start"] == 0
    for previous, current in zip(phases, phases[1:]):
        phase_continuity &= previous["global_end"] == current["global_start"]
    phase_continuity &= phases[-1]["global_end"] == summary["final"]["global_optimizer_updates"]
    phase_continuity &= (
        sum(phase["executed_iterations"] for phase in phases) == phases[-1]["global_end"]
    )

    density_history = _load_object(run / "phase_histories/standard_growth.json")
    density_events = density_history["density_stats"]
    expected_density_iterations = list(range(500, 15001, 100))
    density_schedule_exact = (
        [event["iteration"] for event in density_events] == expected_density_iterations
        and all(event["n_after"] <= 100000 for event in density_events)
        and phase_by_name["standard_growth"]["global_end"] - 105000 == 15000
    )

    convergence = {
        phase["name"]: {
            "requested_max_iterations": phase["requested_max_iterations"],
            "executed_iterations": phase["executed_iterations"],
            "stop_reason": phase["stop_reason"],
            "plateau_converged": (
                None if phase["plateau"] is None else phase["plateau"]["converged"]
            ),
            "best_step": None if phase["plateau"] is None else phase["plateau"]["best_step"],
            "stale_evals": (None if phase["plateau"] is None else phase["plateau"]["stale_evals"]),
        }
        for phase in phases
    }
    capped_nonconverged = [
        name
        for name in (
            "clone_recovery_1",
            "clone_recovery_2",
            "clone_recovery_3",
            "higher_sh",
        )
        if phase_by_name[name]["stop_reason"] == "max_iterations"
        and not phase_by_name[name]["plateau"]["converged"]
    ]

    stage_metrics = {
        stage["name"]: {
            "global_step": stage["global_step"],
            "n_gaussians": stage["n_gaussians"],
            "native_psnr_fg_mean": stage["native_psnr_fg_mean"],
            "compact_psnr_fg_mean": stage["compact_psnr_fg_mean"],
            "native_ssim_crop_mean": stage["native_ssim_crop_mean"],
            "native_lpips_mean": stage["native_lpips_mean"],
            "native_alpha_iou_mean": stage["native_alpha_iou_mean"],
            "carrier_survival_rate": stage["carrier_survival_rate"],
        }
        for stage in stages
    }
    sequential_deltas = {}
    for before, after in zip(stages, stages[1:]):
        sequential_deltas[f"{before['name']}->{after['name']}"] = {
            "native_psnr_fg_db": (after["native_psnr_fg_mean"] - before["native_psnr_fg_mean"]),
            "compact_psnr_fg_db": (after["compact_psnr_fg_mean"] - before["compact_psnr_fg_mean"]),
        }
    before_after_clone = {}
    for wave in range(1, 4):
        before_name = (
            "fixed_topology_converged" if wave == 1 else f"clone_wave_{wave - 1}_converged"
        )
        immediate_name = f"clone_wave_{wave}"
        recovered_name = f"clone_wave_{wave}_converged"
        before_after_clone[f"wave_{wave}"] = {
            "immediate_native_delta_db": (
                stage_by_name[immediate_name]["native_psnr_fg_mean"]
                - stage_by_name[before_name]["native_psnr_fg_mean"]
            ),
            "after_15000_updates_native_delta_db": (
                stage_by_name[recovered_name]["native_psnr_fg_mean"]
                - stage_by_name[before_name]["native_psnr_fg_mean"]
            ),
            "causal_attribution_valid": False,
        }

    historical = summary["historical_reference"]
    historical_initial = ROOT / historical["initial"]["path"]
    historical_final = ROOT / historical["selected_final"]["path"]
    historical_sources_exact = (
        _sha256(historical_initial) == historical["initial"]["sha256"]
        and _sha256(historical_final) == historical["selected_final"]["sha256"]
    )
    historical_final_psnr = historical["selected_final"]["compact_train"]["psnr_fg"]
    historical_delta = summary["final"]["compact_psnr_fg_mean"] - historical_final_psnr

    final_stage = stage_by_name["standard_converged"]
    final_metrics = summary["final"]
    root_models_exact = (
        _sha256(run / "gaussians_init.ply") == stage_by_name["beam"]["model_sha256"]
        and _sha256(run / "gaussians.ply") == final_stage["model_sha256"]
        and _ply_vertex_count(run / "gaussians.ply") == final_metrics["n_gaussians"]
    )
    repair_order_exact = (
        repair["diagnostics"]["covariance"]["means_frozen_bit_exact"]
        and repair["diagnostics"]["covariance"]["opacity_frozen_bit_exact"]
        and repair["diagnostics"]["covariance"]["appearance_frozen_bit_exact"]
        and repair["diagnostics"]["opacity"]["means_frozen_bit_exact"]
        and repair["diagnostics"]["opacity"]["covariance_frozen_bit_exact"]
        and repair["diagnostics"]["opacity"]["appearance_frozen_bit_exact"]
        and repair["diagnostics"]["appearance"]["means_frozen_bit_exact"]
        and repair["diagnostics"]["appearance"]["covariance_frozen_bit_exact"]
        and repair["diagnostics"]["appearance"]["opacity_frozen_bit_exact"]
    )

    invariants = {
        "summary_schema_and_status": (
            summary.get("schema") == "rtgs.carrier-maturation-all26.summary.v1"
            and summary.get("status") == "complete"
        ),
        "manifest_schema_and_status": (
            manifest.get("schema") == "rtgs.carrier-maturation-all26.run-manifest.v1"
            and manifest.get("status") == "complete"
        ),
        "reader_handoff_hash_and_nonconvergence_labels_exact": (
            _sha256(run / "index.html") == manifest["index_sha256"]
            and {
                method["name"]
                for method in comparison_manifest["methods"]
                if "not converged" in method["name"]
            }
            == {
                "5.1b · clone wave 1 cap endpoint (not converged)",
                "5.2b · clone wave 2 cap endpoint (not converged)",
                "5.3b · clone wave 3 cap endpoint (not converged)",
                "6 · higher SH cap endpoint (not converged)",
            }
            and "<b>Audit erratum.</b>" in index_html
            and "const labelErrata=[" in index_html
        ),
        "scope_explicitly_all_fitted_no_heldout": (
            tuple(summary["scope"]["fit_views"]) == EXPECTED_VIEWS
            and tuple(summary["scope"]["selection_views"]) == EXPECTED_VIEWS
            and tuple(summary["scope"]["reporting_views"]) == EXPECTED_VIEWS
            and summary["scope"]["held_out_views"] == []
            and summary["scope"]["view_count"] == 26
        ),
        "protocol_sha256_exact": (
            _sha256(protocol_path)
            == manifest["protocol"]["sha256"]
            == EXPECTED_PROTOCOL_SHA256
            == _sha256(run / "protocol.md")
        ),
        "implementation_sha256_exact": (
            manifest["implementation_sha256"]
            == EXPECTED_IMPLEMENTATION_SHA256
            == implementation_hashes
        ),
        "raw_rgb_and_mask_sources_hash_exact": raw_source_exact,
        "compact_manifest_view_order_and_archives_hash_exact": compact_source_exact,
        "phase_names_exact": phase_names == EXPECTED_PHASES,
        "stage_names_exact": stage_names == EXPECTED_STAGE_NAMES,
        "phase_protocol_configs_exact": _phase_protocol_exact(phases),
        "phase_ranges_contiguous_and_total_exact": phase_continuity,
        "phase_histories_complete_and_finite": _phase_histories_exact(run, phases),
        "metric_means_models_and_stage_hashes_exact": _metric_and_model_bindings_exact(run, stages),
        "checkpoint_hashes_counts_and_steps_exact": _checkpoint_bindings_exact(
            run, summary["checkpoints"]
        ),
        "clone_waves_all_rows_tangent_only_and_parent_preserving": _clone_waves_exact(
            clone_receipts
        ),
        "repair_order_and_frozen_fields_exact": repair_order_exact,
        "standard_density_schedule_and_15000_recovery_exact": density_schedule_exact,
        "root_initial_and_final_models_match_boundaries": root_models_exact,
        "historical_reference_hashes_exact": historical_sources_exact,
        "crash_partial_preserved_with_identical_frozen_bindings": (
            crash_partial_manifest["status"] == "started"
            and crash_partial_manifest["seed"] == manifest["seed"] == 27027
            and crash_partial_manifest["protocol"] == manifest["protocol"]
            and crash_partial_manifest["implementation_sha256"] == manifest["implementation_sha256"]
            and (
                DEFAULT_CRASH_PARTIAL / "checkpoints/fixed_topology/gaussians_step_005000.ply"
            ).is_file()
        ),
        "all_summary_numbers_finite": _finite_tree(summary),
    }

    claims = [
        {
            "claim": (
                "The all-26-view schedule produced a complete native-resolution fitted-view "
                "reconstruction."
            ),
            "kind_scope": "measured; one outcome-exposed scene, one seed, all cameras fitted",
            "disposition": "confirm",
            "evidence": final_metrics,
        },
        {
            "claim": "Every fixed-topology maturation phase converged before handover.",
            "kind_scope": "mechanism behavior under the frozen plateau rules",
            "disposition": "retire",
            "evidence": {
                "capped_nonconverged_phases": capped_nonconverged,
                "convergence": convergence,
            },
        },
        {
            "claim": "Uniform tangent clone waves caused the recovered quality gains.",
            "kind_scope": "causal stage attribution",
            "disposition": "unverified",
            "evidence": {
                "sequential_measurements": before_after_clone,
                "limitation": (
                    "each recovery includes 15,000 additional optimizer updates and no matched "
                    "no-clone continuation exists"
                ),
            },
        },
        {
            "claim": (
                "The final model is competitive with the prior all-view compact-target endpoint."
            ),
            "kind_scope": "contextual fitted-view compact-teacher score, not a randomized control",
            "disposition": "retire",
            "evidence": {
                "current_compact_psnr_fg_db": final_metrics["compact_psnr_fg_mean"],
                "historical_compact_psnr_fg_db": historical_final_psnr,
                "delta_db": historical_delta,
            },
        },
        {
            "claim": "The run establishes held-out or novel-camera reconstruction quality.",
            "kind_scope": "generalization",
            "disposition": "unverified",
            "evidence": {"limitation": summary["scope"]},
        },
        {
            "claim": "The run establishes a speed, VRAM, storage, or bandwidth advantage.",
            "kind_scope": "performance",
            "disposition": "unverified",
            "evidence": {
                "peak_phase_vram_gb": max(phase["peak_vram_gb"] for phase in phases),
                "limitation": (
                    "single execution, no idle-GPU protocol or repeats, and another GPU process "
                    "was observed during execution"
                ),
            },
        },
        {
            "claim": "The run establishes compact-only sufficiency or authorizes the paper claim.",
            "kind_scope": "paper/main capability claim",
            "disposition": "unverified",
            "evidence": {
                "limitation": (
                    "all optimization phases consume native masked RGB; Original-3DGS and "
                    "compressed-RGB SfM baselines are absent"
                )
            },
        },
    ]

    findings = [
        {
            "severity": "major",
            "finding": (
                "Three clone recoveries and the higher-SH phase reached their safety caps without "
                "the frozen plateau, so the intended convergence-between-stages behavior failed."
            ),
            "affected_phases": capped_nonconverged,
        },
        {
            "severity": "major",
            "finding": (
                "Raw stage identifiers still call four capped boundaries 'converged'; phase "
                "stop_reason and plateau.converged are authoritative."
            ),
            "affected_boundary_ids": list(MISLABELLED_BOUNDARIES),
            "remediation": (
                "Reader-facing viewer/page labels are corrected; raw identifiers are preserved "
                "with this erratum."
            ),
        },
        {
            "severity": "major",
            "finding": (
                "All 26 cameras were used for fitting, stopping, and reporting; the result has no "
                "held-out or novel-camera metric and cannot establish generalization."
            ),
        },
        {
            "severity": "major",
            "finding": (
                "The run has no matched immediate-standard or no-clone control, so sequential "
                "stage gains cannot be attributed causally to carrier repair or clone waves."
            ),
        },
        {
            "severity": "moderate",
            "finding": (
                "Covariance repair changed native fitted-view PSNR by -0.443 dB and opacity repair "
                "then changed it by +0.612 dB; local repair objectives do not establish endpoint "
                "value without matched ablations."
            ),
        },
        {
            "severity": "moderate",
            "finding": (
                "The model reached the 100,000-Gaussian hard cap by the first 5k growth "
                "checkpoint; "
                "59.5% of original carriers survive at the final endpoint and 88.78% retain at "
                "least one descendant."
            ),
        },
        {
            "severity": "moderate",
            "finding": (
                "The final compact-teacher score is 6.146 dB below the bound historical selected "
                "69k endpoint; targets and schedules differ, so this is contextual, not causal."
            ),
        },
        {
            "severity": "moderate",
            "finding": (
                "Timing and throughput are non-decisional because the single run had no idle-GPU "
                "protocol or repeats and another GPU process was observed."
            ),
        },
        {
            "severity": "moderate",
            "finding": (
                "The canonical result is a fresh exact V2 restart after the first V2 process died "
                "at step 5,000. Partial Beam/repair/training outcomes were therefore exposed, but "
                "the preserved partial manifest proves the protocol, implementation hashes, and "
                "seed were unchanged; this remains development evidence, not an independent "
                "confirmatory replication."
            ),
        },
    ]

    return {
        "schema": "rtgs.carrier-maturation-all26.audit.v1",
        "disposition": "VALID_DEVELOPMENT_RUN_INTERMEDIATE_CONVERGENCE_FAILED",
        "default_change_authorized": False,
        "paper_claim_authorized": False,
        "performance_claim_authorized": False,
        "run": str(run.relative_to(ROOT)),
        "source_hashes": {
            "summary": _sha256(run / "summary.json"),
            "manifest": _sha256(run / "run_manifest.json"),
            "protocol": _sha256(protocol_path),
            "compact_manifest": _sha256(COMPACT_ROOT / "manifest.json"),
            "implementation": implementation_hashes,
        },
        "chronology": {
            "v1_failure": (
                "stopped before Beam Fusion or model-quality metrics because the compact-target "
                "helper import failed"
            ),
            "first_v2_attempt": str(DEFAULT_CRASH_PARTIAL.relative_to(ROOT)),
            "first_v2_status": crash_partial_manifest["status"],
            "first_v2_latest_checkpoint_global_step": 5000,
            "canonical_v2_restart_after_partial_outcome_exposure": True,
            "protocol_implementation_and_seed_unchanged": (
                crash_partial_manifest["seed"] == manifest["seed"] == 27027
                and crash_partial_manifest["protocol"] == manifest["protocol"]
                and crash_partial_manifest["implementation_sha256"]
                == manifest["implementation_sha256"]
            ),
        },
        "invariants": invariants,
        "convergence": convergence,
        "capped_nonconverged_phases": capped_nonconverged,
        "mislabelled_boundary_ids": list(MISLABELLED_BOUNDARIES),
        "stage_metrics": stage_metrics,
        "sequential_deltas": sequential_deltas,
        "clone_wave_measurements": before_after_clone,
        "historical_context": {
            "selected_global_step": historical["selected_final"]["selected_global_step"],
            "historical_compact_psnr_fg_db": historical_final_psnr,
            "current_compact_psnr_fg_db": final_metrics["compact_psnr_fg_mean"],
            "current_minus_historical_db": historical_delta,
            "causal_comparison_valid": False,
        },
        "lineage": {
            "final_carrier_survival_rate": final_stage["carrier_survival_rate"],
            "roots_with_any_descendant_rate": _load_object(run / final_stage["metrics"])["lineage"][
                "roots_with_any_descendant_rate"
            ],
            "surviving_original_carriers": _load_object(run / final_stage["metrics"])["lineage"][
                "surviving_original_carriers"
            ],
        },
        "claims": claims,
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.run.resolve())
    failed_invariants = [name for name, passed in result["invariants"].items() if not passed]
    result["failed_invariants"] = failed_invariants
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 1 if failed_invariants else 0


if __name__ == "__main__":
    raise SystemExit(main())
