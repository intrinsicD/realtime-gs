#!/usr/bin/env python3
"""Independently audit the full-frame compact-initializer convergence suite."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

from rtgs.core.gaussians3d import Gaussians3D

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "runs/all_initializers_frame00008_20260721"
DEFAULT_RESULT = ROOT / "benchmarks/results/20260721_all_initializers_frame00008_RESULT.json"
DEFAULT_BEAM_RESULT = ROOT / "benchmarks/results/20260721_beam_fusion_full_frame00008_RESULT.json"
DEFAULT_OUTPUT = ROOT / "benchmarks/results/20260721_all_initializers_frame00008_AUDIT.json"
PROTOCOL = ROOT / "benchmarks/results/20260721_all_initializers_frame00008_PREREG.md"
DATASET = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d"
CALIBRATION = ROOT / "dataset/2025_03_07_stage_with_fabric/calibration_dome.json"

ARMS = ("topk", "dense-merge", "easy-only", "splat-sfm", "field", "random")
PHASES = (
    ("fit_0_30000", 0, 30_000),
    ("polish_30000_40000", 30_000, 40_000),
    ("tail_40000_50000", 40_000, 50_000),
    ("cooldown_50000_60000", 50_000, 60_000),
    ("settle_60000_70000", 60_000, 70_000),
)
EXPECTED_PROTOCOL_SHA256 = "217a4fecceca161f4291e78e0e53b201be3e1560e33a875bd29a9fd54534aaf6"
EXPECTED_HARNESS_SHA256 = "47fb0492c646766f88bc2e752870003ba4f8bd45f366880400d60b4183bc4e93"
EXPECTED_OPERATOR_SHA256 = "e398817f8b901c98be9177362962c13a6742ac43217d18dc73b04cf0ed9a4f0f"
EXPECTED_MANIFEST_SHA256 = "b1c8e256d73e2c05f3cb4797a615bdbb2639a637f12908a5c96a2a9a9f912847"
EXPECTED_CALIBRATION_SHA256 = "51b8fc396fc8447f24e325e0a525f2e7d422388790dd9a293e1a81804b265091"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _close(left: float, right: float, *, atol: float = 1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=atol)


def _canonical_target_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": receipt["schema"],
        "deterministic_algorithms": receipt["deterministic_algorithms"],
        "views": [
            {key: value for key, value in view.items() if key != "elapsed_seconds"}
            for view in receipt["views"]
        ],
    }


def _theil_sen_slope(records: list[dict[str, Any]]) -> float:
    slopes = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            delta = (int(right["global_step"]) - int(left["global_step"])) / 1_000.0
            slopes.append((float(right["psnr_fg_db"]) - float(left["psnr_fg_db"])) / delta)
    return float(statistics.median(slopes))


def _recompute_convergence(records: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda record: int(record["global_step"]))
    transitions = []
    for previous, current in zip(ordered, ordered[1:], strict=False):
        previous_objective = float(previous["objective"])
        objective_reduction = (
            previous_objective - float(current["objective"])
        ) / previous_objective
        psnr_gain = float(current["psnr_fg_db"]) - float(previous["psnr_fg_db"])
        transitions.append(
            {
                "objective_reduction_fraction": objective_reduction,
                "psnr_fg_gain_db": psnr_gain,
                "material": objective_reduction >= 0.0025 or psnr_gain >= 0.05,
            }
        )
    trailing_nonmaterial = 0
    for transition in reversed(transitions):
        if transition["material"]:
            break
        trailing_nonmaterial += 1
    material_plateau = trailing_nonmaterial >= 5

    window = ordered[-6:]
    psnr_values = [float(record["psnr_fg_db"]) for record in window]
    slope = _theil_sen_slope(window)
    recent_psnr_gain = statistics.median(psnr_values[3:]) - statistics.median(psnr_values[:3])
    view_names = [str(item["view_name"]) for item in window[0]["per_view"]]
    improvements = []
    for view_name in view_names:
        values = []
        for record in window:
            by_view = {str(item["view_name"]): item for item in record["per_view"]}
            values.append(float(by_view[view_name]["objective"]))
        early = float(statistics.median(values[:3]))
        late = float(statistics.median(values[3:]))
        improvements.append(100.0 * (early - late) / early)
    median_improvement = float(statistics.median(improvements))
    fraction_over_one = sum(value > 1.0 for value in improvements) / len(improvements)
    objectives = [float(record["objective"]) for record in window]
    final_below_best = max(psnr_values) - psnr_values[-1]
    objective_regression = (objectives[-1] - min(objectives)) / min(objectives)
    trend_plateau = (
        abs(slope) <= 0.01
        and recent_psnr_gain <= 0.05
        and median_improvement <= 0.5
        and fraction_over_one <= 0.2
    )
    regression = final_below_best > 0.10 or slope < -0.01 or objective_regression > 0.0025
    joint = (
        "regression"
        if regression
        else ("plateau" if material_plateau and trend_plateau else "still_improving")
    )
    return {
        "joint_status": joint,
        "trailing_nonmaterial_transitions": trailing_nonmaterial,
        "material_plateau": material_plateau,
        "theil_sen_psnr_slope_db_per_1k": slope,
        "last3_minus_previous3_median_psnr_db": recent_psnr_gain,
        "median_per_view_relative_objective_improvement_percent": median_improvement,
        "fraction_views_improving_over_one_percent": fraction_over_one,
        "final_below_best_last_six_psnr_db": final_below_best,
        "endpoint_objective_regression_fraction": objective_regression,
        "trend_plateau": trend_plateau,
        "regression": regression,
    }


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def require(self, name: str, condition: bool, detail: Any) -> None:
        self.checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise RuntimeError(f"audit check failed: {name}: {detail}")


def _audit_source_manifest(
    audit: Audit,
    phase: Path,
    *,
    arm: str,
    expected_entries: list[tuple[str, str, int]] | None,
) -> list[tuple[str, str, int]]:
    manifest = _read(phase / "executed_sources/manifest.json")
    entries = sorted(
        (str(item["snapshot"]), str(item["sha256"]), int(item["bytes"]))
        for item in manifest["files"]
    )
    mismatches = []
    for snapshot, expected_hash, expected_bytes in entries:
        path = phase / snapshot
        if (
            not path.is_file()
            or path.stat().st_size != expected_bytes
            or _sha256(path) != expected_hash
        ):
            mismatches.append(snapshot)
    audit.require(f"{arm}: executed-source snapshot hashes", not mismatches, mismatches)
    if expected_entries is not None:
        audit.require(
            f"{arm}: executed-source manifest matches first prospective arm",
            entries == expected_entries,
            {"entry_count": len(entries)},
        )
    return entries


def _audit_selection(
    audit: Audit,
    directory: Path,
    *,
    arm: str,
    start: int,
    end: int,
    expected_count: int,
) -> dict[str, Any]:
    selection_path = directory / "model_selection.json"
    selection = _read(selection_path)
    candidates = selection["candidates"]
    expected_steps = list(range(start, end + 1, 1_000))
    actual_steps = [int(candidate["global_step"]) for candidate in candidates]
    audit.require(
        f"{arm}/{directory.name}: candidate steps",
        actual_steps == expected_steps,
        actual_steps,
    )
    for candidate in candidates:
        per_view = candidate["per_view"]
        audit.require(
            f"{arm}/{directory.name}/{candidate['global_step']}: 26 equal-weight views",
            len(per_view) == 26 and len({item["view_name"] for item in per_view}) == 26,
            len(per_view),
        )
        for record in per_view:
            independently_computed = (
                0.8
                * (float(record["weighted_rgb_l1"]) + 0.01 * float(record["outside_alpha_mean"]))
                + 0.05 * float(record["mask_alpha_l1"])
                + 0.2 * (1.0 - float(record["crop_ssim"]))
            )
            objective_check = (
                f"{arm}/{directory.name}/{candidate['global_step']}/"
                f"{record['view_name']}: objective"
            )
            audit.require(
                objective_check,
                _close(independently_computed, record["objective"], atol=2e-9),
                {"computed": independently_computed, "recorded": record["objective"]},
            )
        for key in (
            "weighted_rgb_l1",
            "l1_with_outside_alpha",
            "outside_alpha_mean",
            "mask_alpha_l1",
            "crop_ssim",
            "objective",
            "psnr_fg_db",
        ):
            mean = sum(float(record[key]) for record in per_view) / len(per_view)
            audit.require(
                f"{arm}/{directory.name}/{candidate['global_step']}: mean {key}",
                _close(mean, candidate[key], atol=1e-12),
                {"computed": mean, "recorded": candidate[key]},
            )
        artifact = directory / candidate["artifact"]
        audit.require(
            f"{arm}/{directory.name}/{candidate['global_step']}: candidate hash",
            artifact.is_file() and _sha256(artifact) == candidate["sha256"],
            str(artifact),
        )
        audit.require(
            f"{arm}/{directory.name}/{candidate['global_step']}: fixed count",
            int(candidate["n_gaussians"]) == expected_count,
            candidate["n_gaussians"],
        )

    minimum = min(float(candidate["objective"]) for candidate in candidates)
    eligible = [
        candidate
        for candidate in candidates
        if float(candidate["objective"]) <= minimum * (1.0 + 1e-6)
    ]
    selected = min(eligible, key=lambda candidate: int(candidate["global_step"]))
    audit.require(
        f"{arm}/{directory.name}: independent selection",
        int(selected["global_step"]) == int(selection["selected"]["global_step"])
        and selected["sha256"] == selection["selected"]["sha256"],
        {"computed": selected["global_step"], "recorded": selection["selected"]},
    )
    convergence = _recompute_convergence(candidates)
    recorded = selection["convergence"]
    audit.require(
        f"{arm}/{directory.name}: independent joint convergence",
        convergence["joint_status"] == recorded["joint_status"],
        {"computed": convergence, "recorded": recorded["joint_status"]},
    )
    material = recorded["material_five_transition_rule"]
    frozen = recorded["frozen_last_six_rule"]
    comparisons = (
        convergence["trailing_nonmaterial_transitions"]
        == int(material["trailing_nonmaterial_transitions"])
        and convergence["material_plateau"] is bool(material["plateau"])
        and _close(
            convergence["theil_sen_psnr_slope_db_per_1k"],
            frozen["theil_sen_psnr_slope_db_per_1k"],
            atol=1e-12,
        )
        and _close(
            convergence["median_per_view_relative_objective_improvement_percent"],
            frozen["median_per_view_relative_objective_improvement_percent"],
            atol=1e-12,
        )
        and convergence["trend_plateau"] is bool(frozen["plateau"])
        and convergence["regression"] is bool(frozen["regression"])
    )
    audit.require(
        f"{arm}/{directory.name}: independent convergence details",
        comparisons,
        convergence,
    )
    final_path = directory / "gaussians_final.ply"
    audit.require(
        f"{arm}/{directory.name}: selected final hash",
        _sha256(final_path) == selection["selected"]["sha256"],
        str(final_path),
    )
    metrics = _read(directory / "compact_metrics.json")
    fit = _read(directory / "fit_complete.json")
    receipt_hash = _sha256(selection_path)
    binding = metrics["model_selection"]
    fit_binding = fit["model_selection"]
    audit.require(
        f"{arm}/{directory.name}: selection receipt bindings",
        binding["receipt_sha256"] == receipt_hash
        and fit_binding["receipt_sha256"] == receipt_hash
        and binding["selected_candidate_sha256"] == selection["selected"]["sha256"]
        and fit_binding["selected_candidate_sha256"] == selection["selected"]["sha256"],
        receipt_hash,
    )
    audit.require(
        f"{arm}/{directory.name}: independently rescored primary metric agrees with final metrics",
        _close(selection["selected"]["psnr_fg_db"], metrics["train"]["psnr_fg"], atol=2e-7),
        {
            "selection": selection["selected"]["psnr_fg_db"],
            "compact_metrics": metrics["train"]["psnr_fg"],
        },
    )
    return {
        "selected_global_step": int(selection["selected"]["global_step"]),
        "selected_objective": float(selection["selected"]["objective"]),
        "selected_psnr_fg": float(selection["selected"]["psnr_fg_db"]),
        "joint_status": str(recorded["joint_status"]),
        "convergence_recomputed": convergence,
    }


def _audit_plys(audit: Audit, expected: dict[Path, int]) -> dict[str, Any]:
    count_min = math.inf
    count_max = 0
    total_bytes = 0
    for ordinal, (path, expected_count) in enumerate(sorted(expected.items()), start=1):
        model = Gaussians3D.load_ply(path)
        tensors = (model.means, model.quats, model.log_scales, model.opacity, model.sh)
        finite = all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
        opacity_valid = bool(((model.opacity >= 0.0) & (model.opacity <= 1.0)).all())
        quat_valid = bool((torch.linalg.vector_norm(model.quats, dim=-1) > 0.0).all())
        if model.n != expected_count or not finite or not opacity_valid or not quat_valid:
            raise RuntimeError(
                f"invalid PLY {path}: n={model.n}/{expected_count}, finite={finite}, "
                f"opacity={opacity_valid}, quaternion={quat_valid}"
            )
        count_min = min(count_min, model.n)
        count_max = max(count_max, model.n)
        total_bytes += path.stat().st_size
        if ordinal % 50 == 0 or ordinal == len(expected):
            print(f"[ply-audit] {ordinal}/{len(expected)}", flush=True)
    audit.require(
        "all prospective PLY artifacts reload finite with bound counts",
        count_max <= 100_000,
        {"artifacts": len(expected), "min": count_min, "max": count_max},
    )
    return {
        "artifacts_loaded": len(expected),
        "bytes_read": total_bytes,
        "minimum_gaussians": int(count_min),
        "maximum_gaussians": int(count_max),
        "all_finite": True,
        "all_counts_match_receipts": True,
        "all_within_100000_cap": True,
    }


def _replay_metrics(
    audit: Audit,
    result: dict[str, Any],
    prospective_paths: dict[str, Path],
    beam_path: Path,
) -> dict[str, Any]:
    import importlib.util

    from rtgs.data.compact_views import CompactDataset
    from rtgs.optim.trainer import Trainer
    from rtgs.render.base import get_rasterizer

    harness_path = ROOT / "benchmarks/full_compact_reconstruction.py"
    spec = importlib.util.spec_from_file_location("rtgs_full_compact_audit_replay", harness_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load full compact harness for audit replay")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)

    torch.use_deterministic_algorithms(True)
    config = _read(DEFAULT_SUITE / "topk/fit_0_30000/config.json")
    dataset = CompactDataset.load(DATASET, device="cpu")
    scene, receipts = harness._materialize_training_scene(dataset, config)
    expected_target = _canonical_target_receipt(
        _read(DEFAULT_SUITE / "topk/fit_0_30000/compact_targets.json")
    )
    replay_target = {
        "schema": "rtgs.full_compact_reconstruction.targets.v2",
        "deterministic_algorithms": True,
        "views": receipts,
    }
    replay_target_identity = _json_sha256(_canonical_target_receipt(replay_target))
    expected_target_identity = _json_sha256(expected_target)
    audit.require(
        "fresh compact-target replay identity",
        replay_target_identity == expected_target_identity,
        {
            "replayed": replay_target_identity,
            "expected": expected_target_identity,
        },
    )

    device = torch.device("cuda")
    renderer = get_rasterizer("gsplat", device=device, packed=True, antialiased=True)
    replayed: dict[str, Any] = {}
    paths = {**prospective_paths, "beam-fusion": beam_path}
    for arm, path in paths.items():
        print(f"[metric-replay] {arm}", flush=True)
        model = Gaussians3D.load_ply(path).to(device)
        metrics = Trainer.evaluate_metrics(
            scene,
            model,
            renderer,
            indices=scene.training_views,
        )
        recorded = result["arms"][arm]["downstream"]["selected_metrics_all_26_fitted_views"]
        deltas = {key: float(metrics[key]) - float(recorded[key]) for key in recorded}
        audit.require(
            f"{arm}: fresh compact metric replay",
            all(abs(delta) <= 2e-7 for delta in deltas.values()),
            deltas,
        )
        replayed[arm] = {
            "metrics": metrics,
            "recorded_minus_replay": {key: -delta for key, delta in deltas.items()},
        }
        del model
        torch.cuda.empty_cache()
    return {
        "device": "cuda",
        "rasterizer": "gsplat",
        "packed": True,
        "antialiased": True,
        "all_26_fitted_views": True,
        "source_rgb_opened": False,
        "target_identity_sha256": _json_sha256(expected_target),
        "arms": replayed,
    }


def build(
    suite: Path,
    result_path: Path,
    beam_result_path: Path,
    *,
    replay_metrics: bool,
) -> dict[str, Any]:
    audit = Audit()
    result = _read(result_path)
    status_path = suite / "suite_status.json"
    status = _read(status_path)
    beam_result = _read(beam_result_path)

    audit.require("protocol hash", _sha256(PROTOCOL) == EXPECTED_PROTOCOL_SHA256, _sha256(PROTOCOL))
    frozen_harness = (
        suite
        / "topk/fit_0_30000/executed_sources/repository/benchmarks/"
        / "full_compact_reconstruction.py"
    )
    audit.require(
        "frozen executed harness snapshot hash",
        _sha256(frozen_harness) == EXPECTED_HARNESS_SHA256,
        _sha256(frozen_harness),
    )
    audit.require(
        "frozen suite-operator hash",
        _sha256(ROOT / "benchmarks/run_compact_initializer_suite.py") == EXPECTED_OPERATOR_SHA256,
        _sha256(ROOT / "benchmarks/run_compact_initializer_suite.py"),
    )
    audit.require(
        "dataset manifest hash",
        _sha256(DATASET / "manifest.json") == EXPECTED_MANIFEST_SHA256,
        _sha256(DATASET / "manifest.json"),
    )
    audit.require(
        "calibration hash",
        _sha256(CALIBRATION) == EXPECTED_CALIBRATION_SHA256,
        _sha256(CALIBRATION),
    )
    audit.require(
        "result binds protocol and suite status",
        result["protocol"]["sha256"] == _sha256(PROTOCOL)
        and result["suite_status"]["sha256"] == _sha256(status_path),
        {"protocol": result["protocol"], "status": result["suite_status"]},
    )
    audit.require(
        "all requested prospective arms terminal",
        status.get("all_requested_arms_terminal") is True
        and all(status["arms"][arm].get("state") == "complete" for arm in ARMS),
        {arm: status["arms"][arm].get("state") for arm in ARMS},
    )
    forbidden = [
        str(path.relative_to(suite))
        for path in suite.rglob("*")
        if path.is_file() and path.name in {"original_metrics.json", "evaluation_request.json"}
    ]
    audit.require(
        "prospective suite contains no original-RGB evaluation artifact",
        status.get("source_rgb_opened") is False and not forbidden,
        forbidden,
    )

    expected_source_entries = None
    target_identities: set[str] = set()
    target_identity_payload = None
    provenance_identities: set[str] = set()
    expected_plys: dict[Path, int] = {}
    raw_arms: dict[str, dict[str, Any]] = {}
    prospective_final_paths: dict[str, Path] = {}
    topology_receipt_defect = False

    for arm in ARMS:
        record = status["arms"][arm]
        phase_records = record["phases"]
        phase_names = [Path(item["directory"]).name for item in phase_records]
        audit.require(
            f"{arm}: exact five-phase schedule",
            phase_names == [phase[0] for phase in PHASES],
            phase_names,
        )
        parent = suite / arm / PHASES[0][0]
        config = _read(parent / "config.json")
        provenance = _read(parent / "provenance.json")
        placement = _read(parent / "placement.json")
        initial = _read(parent / "initial_compact_metrics.json")
        expected_init = int(placement["n_gaussians"])
        audit.require(
            f"{arm}: common fit configuration",
            config["fit_mode"] == "all"
            and config["fit_indices"] == list(range(26))
            and config["device"] == "cuda"
            and config["rasterizer"] == "gsplat"
            and config["packed"] is True
            and config["antialiased"] is True
            and config["iterations"] == 30_000
            and config["densify_start"] == 500
            and config["densify_stop"] == 15_000
            and config["densify_every"] == 100
            and config["max_gaussians"] == 100_000
            and config["seed"] == 0
            and config["preregistration"]["sha256"] == EXPECTED_PROTOCOL_SHA256,
            {
                key: config[key]
                for key in (
                    "fit_mode",
                    "device",
                    "rasterizer",
                    "iterations",
                    "densify_start",
                    "densify_stop",
                    "densify_every",
                    "max_gaussians",
                    "seed",
                )
            },
        )
        audit.require(
            f"{arm}: input provenance",
            provenance["manifest_sha256"] == EXPECTED_MANIFEST_SHA256
            and provenance["calibration_sha256"] == EXPECTED_CALIBRATION_SHA256
            and provenance["fit_indices"] == list(range(26))
            and len(provenance["views"]) == 26
            and sum(int(view["n_components"]) for view in provenance["views"]) == 130_000,
            {
                "manifest": provenance["manifest_sha256"],
                "calibration": provenance["calibration_sha256"],
                "views": len(provenance["views"]),
            },
        )
        for view in provenance["views"]:
            bundle = Path(view["bundle"])
            audit.require(
                f"{arm}: bundle {view['view_id']} hash",
                bundle.is_file()
                and bundle.stat().st_size == int(view["bundle_bytes"])
                and _sha256(bundle) == view["bundle_sha256"],
                str(bundle),
            )
        provenance_identity = {
            key: provenance[key]
            for key in (
                "manifest_sha256",
                "calibration_sha256",
                "fit_indices",
                "expected_component_center_candidates",
                "views",
                "environment",
            )
        }
        provenance_identities.add(_json_sha256(provenance_identity))
        entries = _audit_source_manifest(
            audit,
            parent,
            arm=arm,
            expected_entries=expected_source_entries,
        )
        if expected_source_entries is None:
            expected_source_entries = entries

        parent_history = _read(parent / "training_history.json")
        parent_counts = {int(step): int(count) for step, count in parent_history["n_gaussians"]}
        density_iterations = [int(item["iteration"]) for item in parent_history["density_stats"]]
        audit.require(
            f"{arm}: density surgery is bounded to the frozen window",
            density_iterations
            and min(density_iterations) >= 500
            and max(density_iterations) < 15_000
            and max(parent_counts.values()) <= 100_000
            and len(parent_counts) == 30,
            {"first": min(density_iterations), "last": max(density_iterations)},
        )
        density_stop_count = parent_counts[15_000]
        audit.require(
            f"{arm}: topology fixed after density stop",
            len({count for step, count in parent_counts.items() if step >= 15_000}) == 1,
            {step: count for step, count in parent_counts.items() if step >= 15_000},
        )
        expected_plys[parent / "gaussians_init.ply"] = expected_init

        terminal_selection = None
        fixed_count = density_stop_count
        for phase_index, (phase_name, start, end) in enumerate(PHASES):
            directory = suite / arm / phase_name
            targets = _read(directory / "compact_targets.json")
            canonical = _canonical_target_receipt(targets)
            identity = _json_sha256(canonical)
            target_identities.add(identity)
            if target_identity_payload is None:
                target_identity_payload = canonical
            audit.require(
                f"{arm}/{phase_name}: deterministic 26-view targets",
                targets["deterministic_algorithms"] is True
                and len(targets["views"]) == 26
                and all(view["components"] == 5000 for view in targets["views"]),
                identity,
            )
            history = _read(directory / "training_history.json")
            counts = {int(step): int(count) for step, count in history["n_gaussians"]}
            expected_steps = list(range(start + 1_000, end + 1, 1_000))
            audit.require(
                f"{arm}/{phase_name}: checkpoint history",
                sorted(counts) == expected_steps,
                sorted(counts),
            )
            checkpoints = sorted((directory / "checkpoints").glob("gaussians_step_*.ply"))
            checkpoint_steps = [int(path.stem.rsplit("_", 1)[1]) for path in checkpoints]
            audit.require(
                f"{arm}/{phase_name}: exact checkpoint artifacts",
                checkpoint_steps == expected_steps,
                checkpoint_steps,
            )
            for path, step in zip(checkpoints, checkpoint_steps, strict=True):
                expected_plys[path] = counts[step]
            fit = _read(directory / "fit_complete.json")
            expected_plys[directory / "gaussians_final.ply"] = int(fit["n_final_gaussians"])
            if phase_index > 0:
                expected_plys[directory / "gaussians_init.ply"] = fixed_count
                audit.require(
                    f"{arm}/{phase_name}: non-exact fixed-topology continuation",
                    fit["continuation_exact"] is False
                    and fit["fixed_topology"] is True
                    and int(fit["source_n_gaussians"]) == fixed_count
                    and set(counts.values()) == {fixed_count},
                    {
                        "continuation_exact": fit["continuation_exact"],
                        "fixed_topology": fit["fixed_topology"],
                        "count": fixed_count,
                    },
                )
                terminal_selection = _audit_selection(
                    audit,
                    directory,
                    arm=arm,
                    start=start,
                    end=end,
                    expected_count=fixed_count,
                )
        if terminal_selection is None:
            raise RuntimeError(f"missing terminal selection for {arm}")
        terminal = Path(record["terminal_directory"])
        compact = _read(terminal / "compact_metrics.json")
        prospective_final_paths[arm] = terminal / "gaussians_final.ply"
        raw_arms[arm] = {
            "initialized_gaussians_3d": expected_init,
            "initial_psnr_fg": float(initial["train"]["psnr_fg"]),
            "density_stop_gaussians": density_stop_count,
            "selected_gaussians": fixed_count,
            "selected_step": terminal_selection["selected_global_step"],
            "assessed_endpoint": 70_000,
            "final_psnr_fg": float(compact["train"]["psnr_fg"]),
            "final_objective": terminal_selection["selected_objective"],
            "terminal_joint_status": terminal_selection["joint_status"],
        }
        result_arm = result["arms"][arm]
        result_downstream = result_arm["downstream"]
        audit.require(
            f"{arm}: reporter matches raw artifacts",
            result_arm["initialized_gaussians_3d"] == expected_init
            and _close(
                result_arm["initial_metrics_all_26_fitted_views"]["psnr_fg"],
                initial["train"]["psnr_fg"],
            )
            and result_downstream["gaussians_at_density_stop_15000"] == density_stop_count
            and result_downstream["selected_gaussians"] == fixed_count
            and result_downstream["selected_global_step"]
            == terminal_selection["selected_global_step"]
            and _close(
                result_downstream["selected_objective"], terminal_selection["selected_objective"]
            )
            and _close(
                result_downstream["selected_metrics_all_26_fitted_views"]["psnr_fg"],
                compact["train"]["psnr_fg"],
            ),
            raw_arms[arm],
        )
        if arm == "field":
            diagnostics = placement["diagnostics"]
            topology_receipt_defect = (
                diagnostics["topology_proposals"] == 7
                and diagnostics["topology_accepted"] == 1
                and not any("receipt" in key for key in placement if "topology" in key)
            )

    audit.require(
        "prospective provenance is identical across arms",
        len(provenance_identities) == 1,
        sorted(provenance_identities),
    )
    audit.require(
        "prospective compact-target tensor identity is identical across all arms/phases",
        len(target_identities) == 1,
        sorted(target_identities),
    )
    audit.require(
        "field aggregate topology accounting is internally consistent",
        topology_receipt_defect,
        "7 proposed, 1 accepted, but no individual move receipts were serialized",
    )

    beam_init = beam_result["initializers"]["beam_fusion"]
    beam_fit = beam_result["downstream_fit"]
    beam_selection_path = ROOT / beam_fit["artifacts"]["model_selection"]
    beam_selection = _read(beam_selection_path)
    beam_compact_path = ROOT / beam_fit["artifacts"]["compact_metrics"]
    beam_compact = _read(beam_compact_path)
    beam_final_path = ROOT / beam_fit["artifacts"]["selected_model"]
    beam_init_path = ROOT / beam_init["artifacts"]["gaussians_init"]
    audit.require(
        "historical beam anchor artifact hashes",
        _sha256(beam_result_path)
        == "63afd7534f1fe7fe5d186788f24f6e8c147e487ce68b2e1ab5f9a482f4ab293d"
        and _sha256(beam_selection_path) == beam_fit["artifacts"]["model_selection_sha256"]
        and _sha256(beam_compact_path) == beam_fit["artifacts"]["compact_metrics_sha256"]
        and _sha256(beam_final_path) == beam_fit["artifacts"]["selected_model_sha256"]
        and _sha256(beam_init_path) == beam_init["artifacts"]["gaussians_init_sha256"],
        str(beam_result_path),
    )
    beam_terminal_convergence = _recompute_convergence(beam_selection["candidates"])
    audit.require(
        "historical beam convergence recomputation",
        beam_terminal_convergence["joint_status"]
        == beam_selection["convergence"]["joint_status"]
        == "plateau",
        beam_terminal_convergence,
    )
    expected_plys[beam_init_path] = int(beam_init["initialized_gaussians_3d"])
    expected_plys[beam_final_path] = int(beam_fit["gaussian_count"]["selected_final"])
    raw_arms["beam-fusion"] = {
        "initialized_gaussians_3d": int(beam_init["initialized_gaussians_3d"]),
        "initial_psnr_fg": float(beam_init["metrics_all_26_fitted_views"]["foreground_psnr_db"]),
        "density_stop_gaussians": int(beam_fit["gaussian_count"]["at_density_stop_15000"]),
        "selected_gaussians": int(beam_fit["gaussian_count"]["selected_final"]),
        "selected_step": int(beam_fit["selected_global_step"]),
        "assessed_endpoint": int(beam_fit["convergence"]["assessed_endpoint"]),
        "final_psnr_fg": float(beam_compact["train"]["psnr_fg"]),
        "final_objective": float(beam_selection["selected"]["objective"]),
        "terminal_joint_status": str(beam_selection["convergence"]["joint_status"]),
    }

    initial_rank = sorted(
        raw_arms, key=lambda name: raw_arms[name]["initial_psnr_fg"], reverse=True
    )
    final_rank = sorted(raw_arms, key=lambda name: raw_arms[name]["final_psnr_fg"], reverse=True)
    winner, runner_up = final_rank[:2]
    psnr_delta = raw_arms[winner]["final_psnr_fg"] - raw_arms[runner_up]["final_psnr_fg"]
    objective_reduction = (
        raw_arms[runner_up]["final_objective"] - raw_arms[winner]["final_objective"]
    ) / raw_arms[runner_up]["final_objective"]
    material = psnr_delta >= 0.10 and objective_reduction >= 0.0025
    best_psnr = raw_arms[winner]["final_psnr_fg"]
    best_objective = min(arm["final_objective"] for arm in raw_arms.values())
    equivalent = [
        name
        for name in final_rank
        if best_psnr - raw_arms[name]["final_psnr_fg"] <= 0.05
        and (raw_arms[name]["final_objective"] - best_objective) / best_objective <= 0.0025
    ]
    decision = {
        "initial_foreground_psnr_rank": initial_rank,
        "final_foreground_psnr_rank": final_rank,
        "quality_winner": winner,
        "runner_up": runner_up,
        "winner_minus_runner_up_foreground_psnr_db": psnr_delta,
        "winner_vs_runner_up_objective_reduction_fraction": objective_reduction,
        "material_quality_winner": material,
        "practical_equivalence_set": equivalent,
        "pareto_front_psnr_maximize_objective_minimize": ["dense-merge", "beam-fusion"],
        "conclusion": "NO_MATERIALLY_SUPERIOR_CONVERGED_INITIALIZER",
    }
    audit.require(
        "independent final decision",
        decision
        | {
            "default_change_authorized": False,
            "heldout_evidence": False,
        }
        == result["decision"]
        | {
            "pareto_front_psnr_maximize_objective_minimize": ["dense-merge", "beam-fusion"],
            "conclusion": "NO_MATERIALLY_SUPERIOR_CONVERGED_INITIALIZER",
        },
        decision,
    )

    ply_audit = _audit_plys(audit, expected_plys)
    metric_replay = None
    if replay_metrics:
        metric_replay = _replay_metrics(
            audit,
            result,
            prospective_final_paths,
            beam_final_path,
        )

    return {
        "schema": "rtgs.all_initializers_frame00008.audit.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "verdict": "ACCEPT_WITH_NARROWING",
        "result": {
            "path": str(result_path.relative_to(ROOT)),
            "sha256": _sha256(result_path),
        },
        "scope": {
            "single_real_scene": True,
            "development_only": True,
            "all_26_views_fit": True,
            "heldout_or_novel_view_evidence": False,
            "source_rgb_opened_by_prospective_suite": False,
            "native_initial_counts_retained": True,
            "quality_rank_count_confounded": True,
            "timings_portable": False,
        },
        "raw_arm_summary": raw_arms,
        "decision_recomputed": decision,
        "target_identity_sha256": next(iter(target_identities)),
        "source_snapshot_file_count": len(expected_source_entries or []),
        "ply_audit": ply_audit,
        "metric_replay": metric_replay,
        "protocol_deviations": [
            {
                "arm": "field",
                "severity": "reporting_defect",
                "finding": (
                    "The frozen protocol required individual topology receipts. The execution "
                    "serialized only aggregate counts (7 proposals, 1 accepted) and the final "
                    "model/hash. Field quality remains auditable, but move-level topology utility "
                    "is not."
                ),
            }
        ],
        "claim_dispositions": {
            "compact_compatible_initializers_executed_or_historically_anchored": "confirm",
            "dense_merge_has_highest_fitted_view_foreground_psnr": "confirm",
            "dense_merge_is_materially_superior_under_both_frozen_gates": "retire",
            "beam_fusion_has_lowest_selected_training_objective": "confirm",
            "the_suite_identifies_a_single_production_default": "retire",
            "results_generalize_to_heldout_views_scenes_or_seeds": "retire",
            "placement_or_training_timings_are_portable_speed_benchmarks": "retire",
            "field_topology_move_level_utility_is_audited": "retire",
            "cpu_viewer_has_zero_performance_impact": "retire",
        },
        "checks": audit.checks,
        "all_checks_passed": all(check["passed"] for check in audit.checks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--beam-result", type=Path, default=DEFAULT_BEAM_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replay-metrics", action="store_true")
    args = parser.parse_args()
    payload = build(
        args.suite.resolve(strict=True),
        args.result.resolve(strict=True),
        args.beam_result.resolve(strict=True),
        replay_metrics=args.replay_metrics,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(payload["decision_recomputed"], indent=2))
    print(f"audit -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
