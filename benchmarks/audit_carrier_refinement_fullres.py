#!/usr/bin/env python3
"""Independent scientist-pass checks for the full-resolution carrier experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIMARY = ROOT / "runs/carrier_refinement_fullres_frame00008_20260727"
DEFAULT_REPEAT = ROOT / "runs/carrier_refinement_fullres_frame00008_20260727_repeat"
DEFAULT_RECOVERY = ROOT / "runs/carrier_refinement_fullres_frame00008_20260727_recovery"
DEFAULT_RECOVERY_REPEAT = (
    ROOT / "runs/carrier_refinement_fullres_frame00008_20260727_recovery_repeat"
)
DEFAULT_OUTPUT = ROOT / "benchmarks/results/20260727_carrier_refinement_fullres_AUDIT.json"

EXPECTED_ARMS = {
    "beam-only",
    "beam-standard",
    "carrier-schedule",
    "carrier-schedule-clone",
    "no-covariance",
    "no-opacity",
    "no-warmup",
    "split-immediately",
    "clone-only",
    "particle",
    "random-rgb",
    "random-jpeg-q50",
    "means-only",
}
EXPECTED_RECOVERY_ARMS = EXPECTED_ARMS - {"beam-only"}
EXPECTED_TRAIN = ["C0001", "C0014", "C0028"]
EXPECTED_VALIDATION = ["C0031", "C1000", "C1002"]
EXPECTED_SEALED = {"C1001", "C1004"}
MATERIALITY_DB = 0.25


def _load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(summary: dict) -> dict[str, dict]:
    records = {record["arm"]: record for record in summary["arms"]}
    if len(records) != len(summary["arms"]):
        raise ValueError("summary contains duplicate arm names")
    return records


def _history_density_events(run: Path, arm: str) -> list[dict]:
    history = _load(run / arm / "training_history.json")
    events = []
    if "phases" in history:
        for phase, phase_history in history["phases"].items():
            for event in phase_history.get("density_stats") or []:
                events.append({"phase": phase, **event})
    else:
        for event in history.get("density_stats") or []:
            events.append({"phase": "standard", **event})
    return events


def _sampled_views(run: Path, arm: str) -> list[int]:
    history = _load(run / arm / "training_history.json")
    if "phases" in history:
        result = []
        for phase_history in history["phases"].values():
            result.extend(phase_history["sampled_train_views"])
        return result
    return history.get("sampled_train_views", [])


def _initialization_hashes(run: Path, arms: set[str]) -> dict[str, str]:
    return {arm: _sha256(run / arm / "gaussians_init.ply") for arm in sorted(arms)}


def _repeat_deltas(first: dict[str, dict], second: dict[str, dict]) -> dict:
    result = {}
    for arm in sorted(first):
        first_metrics = first[arm]["final_metrics"]
        second_metrics = second[arm]["final_metrics"]
        result[arm] = {
            "psnr_fg": second_metrics["psnr_fg"] - first_metrics["psnr_fg"],
            "ssim_crop": second_metrics["ssim_crop"] - first_metrics["ssim_crop"],
            "lpips_crop_max512": (
                second_metrics["lpips_crop_max512"] - first_metrics["lpips_crop_max512"]
            ),
            "alpha_iou": second_metrics["alpha_iou"] - first_metrics["alpha_iou"],
            "n_final": second[arm]["n_final"] - first[arm]["n_final"],
        }
    return result


def _recovery_repeat_deltas(first: dict[str, dict], second: dict[str, dict]) -> dict:
    result = {}
    for arm in sorted(first):
        first_metrics = first[arm]["recovered_metrics"]
        second_metrics = second[arm]["recovered_metrics"]
        result[arm] = {
            "psnr_fg": second_metrics["psnr_fg"] - first_metrics["psnr_fg"],
            "ssim_crop": second_metrics["ssim_crop"] - first_metrics["ssim_crop"],
            "lpips_crop_max512": (
                second_metrics["lpips_crop_max512"] - first_metrics["lpips_crop_max512"]
            ),
            "alpha_iou": second_metrics["alpha_iou"] - first_metrics["alpha_iou"],
            "n_gaussians": second[arm]["n_gaussians"] - first[arm]["n_gaussians"],
        }
    return result


def _finite_tree(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def audit(
    primary: Path,
    repeat: Path,
    recovery: Path,
    recovery_repeat: Path,
) -> dict:
    primary_summary = _load(primary / "summary.json")
    repeat_summary = _load(repeat / "summary.json")
    recovery_summary = _load(recovery / "summary.json")
    recovery_repeat_summary = _load(recovery_repeat / "summary.json")
    primary_records = _records(primary_summary)
    repeat_records = _records(repeat_summary)
    recovery_records = _records(recovery_summary)
    recovery_repeat_records = _records(recovery_repeat_summary)

    source_primary = _load(primary / "source_binding.json")
    source_repeat = _load(repeat / "source_binding.json")
    source_recovery = _load(recovery / "source_binding.json")
    selected_source_ids = set(source_primary)
    split = primary_summary["split"]

    initialization_hashes = _initialization_hashes(primary, EXPECTED_ARMS)
    repaired_group = {
        "carrier-schedule",
        "carrier-schedule-clone",
        "no-warmup",
        "split-immediately",
        "clone-only",
        "particle",
    }
    repaired_hashes = {initialization_hashes[arm] for arm in repaired_group}
    random_hashes = {
        initialization_hashes["random-rgb"],
        initialization_hashes["random-jpeg-q50"],
    }
    beam_hashes = {
        initialization_hashes["beam-only"],
        initialization_hashes["beam-standard"],
    }

    final_births = {}
    for arm in sorted(EXPECTED_ARMS):
        events = _history_density_events(primary, arm)
        last = events[-1] if events else None
        final_births[arm] = {
            "last_density_iteration": None if last is None else last["iteration"],
            "last_event_newborns": (0 if last is None else last["cloned"] + 2 * last["split"]),
            "last_event_net_growth": (0 if last is None else last["n_after"] - last["n_before"]),
            "evaluated_immediately_after_nonempty_density_event": bool(
                last is not None and last["iteration"] == 160 and last["cloned"] + last["split"] > 0
            ),
        }
    invalid_parent_endpoints = sorted(
        arm
        for arm, record in final_births.items()
        if record["evaluated_immediately_after_nonempty_density_event"]
    )

    covariance = primary_summary["repair"]["full"]["covariance"]
    opacity = primary_summary["repair"]["full"]["opacity"]
    covariance_before = covariance["whitened_residual_before"]["median"]
    covariance_after = covariance["whitened_residual_after"]["median"]
    opacity_before = opacity["optical_density_residual_before"]["median"]
    opacity_after = opacity["optical_density_residual_after"]["median"]
    covariance_reduction = 1.0 - covariance_after / covariance_before
    opacity_reduction = 1.0 - opacity_after / opacity_before

    recovered = {arm: record["recovered_metrics"] for arm, record in recovery_records.items()}
    recovery_gates = {
        "carrier_value_delta_db": (
            recovered["carrier-schedule-clone"]["psnr_fg"] - recovered["beam-standard"]["psnr_fg"]
        ),
        "clone_contribution_delta_db": (
            recovered["carrier-schedule-clone"]["psnr_fg"]
            - recovered["carrier-schedule"]["psnr_fg"]
        ),
        "means_value_delta_db": (
            recovered["means-only"]["psnr_fg"] - recovered["random-rgb"]["psnr_fg"]
        ),
        "jpeg_q50_delta_db": (
            recovered["random-jpeg-q50"]["psnr_fg"] - recovered["random-rgb"]["psnr_fg"]
        ),
        "no_covariance_minus_full_db": (
            recovered["no-covariance"]["psnr_fg"] - recovered["carrier-schedule-clone"]["psnr_fg"]
        ),
        "no_opacity_minus_full_db": (
            recovered["no-opacity"]["psnr_fg"] - recovered["carrier-schedule-clone"]["psnr_fg"]
        ),
        "no_warmup_minus_full_db": (
            recovered["no-warmup"]["psnr_fg"] - recovered["carrier-schedule-clone"]["psnr_fg"]
        ),
        "particle_minus_clone_only_db": (
            recovered["particle"]["psnr_fg"] - recovered["clone-only"]["psnr_fg"]
        ),
    }
    full_survival = primary_records["carrier-schedule-clone"]["carrier_survival_rate"]
    recovery_gates.update(
        {
            "carrier_survival_rate": full_survival,
            "carrier_value_gate_pass": (
                recovery_gates["carrier_value_delta_db"] >= MATERIALITY_DB and full_survival >= 0.5
            ),
            "clone_contribution_gate_pass": (
                recovery_gates["clone_contribution_delta_db"] >= MATERIALITY_DB
            ),
            "means_value_gate_pass": (recovery_gates["means_value_delta_db"] >= MATERIALITY_DB),
        }
    )

    parent_repeat = _repeat_deltas(primary_records, repeat_records)
    recovery_repeat_deltas = _recovery_repeat_deltas(recovery_records, recovery_repeat_records)
    max_parent_psnr_repeat_delta = max(abs(record["psnr_fg"]) for record in parent_repeat.values())
    max_parent_count_repeat_delta = max(abs(record["n_final"]) for record in parent_repeat.values())
    max_recovery_psnr_repeat_delta = max(
        abs(record["psnr_fg"]) for record in recovery_repeat_deltas.values()
    )

    compact_bytes = primary_summary["ingestion"]["compact_train3_bytes"]
    raw_bytes = primary_records["random-rgb"]["storage"]["input_bytes"]
    jpeg_bytes = primary_records["random-jpeg-q50"]["storage"]["input_bytes"]
    accounting = {
        "compact_train3_bytes": compact_bytes,
        "raw_rgb_mask_train3_bytes": raw_bytes,
        "jpeg_q50_mask_train3_bytes": jpeg_bytes,
        "raw_to_compact_size_ratio": raw_bytes / compact_bytes,
        "jpeg_q50_to_compact_size_ratio": jpeg_bytes / compact_bytes,
        "raw_ingestion_mib_per_second": primary_summary["ingestion"][
            "raw_effective_ingestion_mib_per_second"
        ],
        "compact_load_mib_per_second": primary_summary["ingestion"][
            "compact_effective_load_mib_per_second"
        ],
        "bandwidth_comparison_valid": False,
        "bandwidth_limitation": (
            "raw timing covers six selected files plus decode/undistort; compact timing covers "
            "all 26 strict containers, so the throughputs are not a controlled speed comparison"
        ),
    }

    invariants = {
        "primary_schema": primary_summary.get("schema") == "rtgs.carrier-refinement-fullres.v1",
        "repeat_schema": repeat_summary.get("schema") == "rtgs.carrier-refinement-fullres.v1",
        "recovery_schema": recovery_summary.get("schema") == "rtgs.carrier-refinement-recovery.v1",
        "all_status_complete": all(
            summary.get("status") == "complete"
            for summary in (
                primary_summary,
                repeat_summary,
                recovery_summary,
                recovery_repeat_summary,
            )
        ),
        "arm_set_exact": set(primary_records) == set(repeat_records) == EXPECTED_ARMS,
        "recovery_arm_set_exact": (
            set(recovery_records) == set(recovery_repeat_records) == EXPECTED_RECOVERY_ARMS
        ),
        "resolution_native": primary_summary.get("resolution") == [5328, 4608],
        "split_exact": (
            list(split["train3"]) == EXPECTED_TRAIN
            and list(split["validation"]) == EXPECTED_VALIDATION
            and set(split["report_only_sealed"]) == EXPECTED_SEALED
        ),
        "sealed_ids_unread": not bool(EXPECTED_SEALED & selected_source_ids),
        "source_binding_exact_across_runs": (source_primary == source_repeat == source_recovery),
        "selected_sources_exact": selected_source_ids == set(EXPECTED_TRAIN + EXPECTED_VALIDATION),
        "repaired_initializations_bit_exact": len(repaired_hashes) == 1,
        "random_rgb_jpeg_initializations_bit_exact": len(random_hashes) == 1,
        "beam_only_standard_initializations_bit_exact": len(beam_hashes) == 1,
        "training_views_only": all(
            set(_sampled_views(primary, arm)) <= {0, 1, 2} for arm in EXPECTED_ARMS
        ),
        "optimized_parent_budgets_160": all(
            primary_records[arm]["iterations"] == 160 for arm in EXPECTED_ARMS - {"beam-only"}
        ),
        "beam_only_budget_zero": primary_records["beam-only"]["iterations"] == 0,
        "recovery_fixed_counts": all(
            record["n_gaussians"] == primary_records[arm]["n_final"]
            for arm, record in recovery_records.items()
        ),
        "recovery_parent_roundtrip_exact_psnr": all(
            record["reloaded_parent_psnr_abs_delta"] == 0 for record in recovery_records.values()
        ),
        "all_json_numbers_finite": all(
            _finite_tree(summary)
            for summary in (
                primary_summary,
                repeat_summary,
                recovery_summary,
                recovery_repeat_summary,
            )
        ),
    }

    claims = [
        {
            "claim": "fixed-track covariance repair lowers its own reprojection residual",
            "scope": "mechanical, CPU float64, Janelle train3 contributor links",
            "disposition": "confirm",
            "evidence": {
                "median_reduction_fraction": covariance_reduction,
                "gate": 0.25,
            },
        },
        {
            "claim": "fixed-track opacity repair meets its preregistered residual gate",
            "scope": "experimental 2D-amplitude proxy, not physical alpha",
            "disposition": "retire",
            "evidence": {
                "median_reduction_fraction": opacity_reduction,
                "gate": 0.25,
            },
        },
        {
            "claim": "the complete carrier schedule beats immediate Beam-to-standard",
            "scope": "full-resolution frame00008 development endpoint after uniform recovery",
            "disposition": "retire",
            "evidence": {
                "delta_db": recovery_gates["carrier_value_delta_db"],
                "gate_db": MATERIALITY_DB,
                "carrier_survival_rate": full_survival,
            },
        },
        {
            "claim": "the clone phase contributes material quality within the carrier schedule",
            "scope": "matched 200-update recovered development arms; topology timing also differs",
            "disposition": "narrow",
            "evidence": {
                "delta_db": recovery_gates["clone_contribution_delta_db"],
                "gate_db": MATERIALITY_DB,
            },
        },
        {
            "claim": "covariance repair helps downstream carrier refinement",
            "scope": "full-resolution frame00008 development",
            "disposition": "retire",
            "evidence": {
                "no_covariance_minus_full_db": recovery_gates["no_covariance_minus_full_db"]
            },
        },
        {
            "claim": "Beam means carry downstream value over matched random means",
            "scope": "simple-parameter, 200-update, single exposed scene control",
            "disposition": "confirm",
            "evidence": {
                "delta_db": recovery_gates["means_value_delta_db"],
                "gate_db": MATERIALITY_DB,
            },
        },
        {
            "claim": "JPEG q50 changes the matched random-control quality materially",
            "scope": "second-generation JPEG proxy, not original SfM 3DGS",
            "disposition": "retire",
            "evidence": {
                "delta_db": recovery_gates["jpeg_q50_delta_db"],
                "materiality_db": MATERIALITY_DB,
            },
        },
        {
            "claim": "compact captures alone are sufficient for a competitive final model",
            "scope": "paper main claim",
            "disposition": "unverified",
            "evidence": {
                "beam_only_psnr": primary_records["beam-only"]["final_metrics"]["psnr_fg"],
                "limitation": (
                    "all optimized carrier arms consume full-resolution RGB; original SfM "
                    "baselines are unavailable"
                ),
            },
        },
        {
            "claim": "time/VRAM/I/O numbers establish a performance advantage",
            "scope": "single 4090 process with no idle-GPU protocol or timing repeats",
            "disposition": "unverified",
            "evidence": accounting,
        },
    ]

    findings = [
        {
            "severity": "critical",
            "finding": (
                "Nine parent endpoints were evaluated immediately after a non-empty step-160 "
                "density event; they are invalid mature endpoints without recovery."
            ),
            "affected_arms": invalid_parent_endpoints,
            "remediation": (
                "preserve parent result; use the separately preregistered uniform 40-step "
                "fixed-topology recovery bundle for mature development comparisons"
            ),
        },
        {
            "severity": "major",
            "finding": (
                "The central recovered carrier-value contrast is negative despite high carrier "
                "survival; the proposed schedule is rejected on this scene."
            ),
            "delta_db": recovery_gates["carrier_value_delta_db"],
        },
        {
            "severity": "major",
            "finding": (
                "Covariance repair improves its local objective yet hurts downstream quality; "
                "local mechanism success is not global materiality."
            ),
            "no_covariance_minus_full_db": recovery_gates["no_covariance_minus_full_db"],
        },
        {
            "severity": "major",
            "finding": (
                "The run cannot test the paper's compact-only sufficiency claim because every "
                "optimized carrier arm consumes raw RGB, and the original SfM baseline is absent."
            ),
        },
        {
            "severity": "moderate",
            "finding": (
                "The parent replay is not bit-deterministic in topology-sensitive CUDA arms."
            ),
            "max_psnr_repeat_delta_db": max_parent_psnr_repeat_delta,
            "max_count_repeat_delta": max_parent_count_repeat_delta,
        },
        {
            "severity": "moderate",
            "finding": (
                "The separately restarted fixed-topology recovery is highly repeatable but is "
                "not an exact optimizer-state continuation."
            ),
            "max_psnr_repeat_delta_db": max_recovery_psnr_repeat_delta,
        },
        {
            "severity": "moderate",
            "finding": (
                "Raw and compact ingestion timings cover different view counts and processing, "
                "so no bandwidth speedup can be claimed from them."
            ),
        },
    ]
    return {
        "schema": "rtgs.carrier-refinement-fullres.audit.v1",
        "disposition": "REJECT_PROPOSED_CARRIER_SCHEDULE_ON_FRAME00008",
        "default_change_authorized": False,
        "paper_claim_authorized": False,
        "primary": str(primary.relative_to(ROOT)),
        "repeat": str(repeat.relative_to(ROOT)),
        "recovery": str(recovery.relative_to(ROOT)),
        "recovery_repeat": str(recovery_repeat.relative_to(ROOT)),
        "source_hashes": {
            "primary_summary": _sha256(primary / "summary.json"),
            "repeat_summary": _sha256(repeat / "summary.json"),
            "recovery_summary": _sha256(recovery / "summary.json"),
            "recovery_repeat_summary": _sha256(recovery_repeat / "summary.json"),
        },
        "invariants": invariants,
        "initialization_hashes": initialization_hashes,
        "final_density_events": final_births,
        "invalid_parent_endpoints": invalid_parent_endpoints,
        "repair_gates": {
            "covariance_median_before": covariance_before,
            "covariance_median_after": covariance_after,
            "covariance_reduction_fraction": covariance_reduction,
            "covariance_gate_pass": covariance_reduction >= 0.25,
            "opacity_median_before": opacity_before,
            "opacity_median_after": opacity_after,
            "opacity_reduction_fraction": opacity_reduction,
            "opacity_gate_pass": opacity_reduction >= 0.25,
        },
        "recovery_gates": recovery_gates,
        "parent_repeat_deltas": parent_repeat,
        "max_parent_psnr_repeat_delta_db": max_parent_psnr_repeat_delta,
        "max_parent_count_repeat_delta": max_parent_count_repeat_delta,
        "recovery_repeat_deltas": recovery_repeat_deltas,
        "max_recovery_psnr_repeat_delta_db": max_recovery_psnr_repeat_delta,
        "accounting": accounting,
        "claims": claims,
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--repeat", type=Path, default=DEFAULT_REPEAT)
    parser.add_argument("--recovery", type=Path, default=DEFAULT_RECOVERY)
    parser.add_argument(
        "--recovery-repeat",
        type=Path,
        default=DEFAULT_RECOVERY_REPEAT,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(
        args.primary.resolve(),
        args.repeat.resolve(),
        args.recovery.resolve(),
        args.recovery_repeat.resolve(),
    )
    failed_invariants = [name for name, passed in result["invariants"].items() if not passed]
    result["failed_invariants"] = failed_invariants
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 1 if failed_invariants else 0


if __name__ == "__main__":
    raise SystemExit(main())
