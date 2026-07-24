#!/usr/bin/env python3
"""Independent results audit for the pooled structure-tensor WSE ablation."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import subprocess
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from PIL import Image

try:
    from benchmarks import audit_new_variants_frame00008 as prior_audit
except ModuleNotFoundError:
    import audit_new_variants_frame00008 as prior_audit

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.calibrated import load_calibrated_scene
from rtgs.optim.trainer import Trainer
from rtgs.render.base import get_rasterizer

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/pool_structure_wse_frame00008_20260724"
SUMMARY = RUN / "summary.json"
PRIOR_SUMMARY = ROOT / "runs/new_variants_frame00008_20260724_v3/summary.json"
RAW_SCENE = Path("/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008")
CALIBRATION = ROOT / "dataset/2025_03_07_stage_with_fabric/calibration_dome.json"
PROTOCOL_V1 = ROOT / "benchmarks/results/20260724_pool_structure_wse_frame00008_PREREG.md"
PROTOCOL_V2 = ROOT / "benchmarks/results/20260724_pool_structure_wse_frame00008_PREREG_V2.md"
PROTOCOL_V3 = ROOT / "benchmarks/results/20260724_pool_structure_wse_frame00008_PREREG_V3.md"
V2_FAILURE = (
    ROOT / "benchmarks/results/20260724_pool_structure_wse_frame00008_ATTEMPT_V2_FAILURE.json"
)
VIEWER = ROOT / "benchmarks/results/20260724_pool_structure_wse_frame00008_VIEWER.json"
VIEWER_RECEIPT = (
    ROOT / "benchmarks/results/20260724_pool_structure_wse_frame00008_VIEWER_RECEIPT.json"
)
OUTPUT = ROOT / "benchmarks/results/20260724_pool_structure_wse_frame00008_AUDIT.json"

ARMS = (
    "pool-gradient",
    "pool-structure-density",
    "pool-structure-wse",
)
VIEW_NAMES = ("C0001", "C0008", "C0014", "C0021", "C0026", "C0031", "C0039", "C1004")
EXPECTED = {
    "summary": "83c832b920a4603937112f4ff177ca8ac4d420dc58e72e97e847e7c896e176eb",
    "prior_summary": "f302b6eaaae6eac8dd7e0894b371f8860df03d047ebb73e037e72ee90be166e9",
    "harness": "9daef81cfcfdb12d2cc3afa786ffb5f798d6492660284250ec09a2ba8ad5efd1",
    "protocol_v1": "616f3691c90e714270dd9c20daf87d48571e8257db8f150594abc90694a9c03d",
    "protocol_v2": "e9de68c509d5b0ee21eddf08e55180834f357713351d23a46bc1172ce9fff4f6",
    "protocol_v3": "1f8859d2bcae7c71336794c4e815202f9f738fcf6d9769a476d4cf6610016416",
    "v2_failure": "c1f41b750ec77decc2ff276cfea4978b959a5e3a24eac91393e8ac7bec783d6d",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def _close(left: float, right: float, tolerance: float = 5e-5) -> bool:
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance


def _walk_artifacts(value: Any, output: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"} <= set(value):
            output.append(value)
        for item in value.values():
            _walk_artifacts(item, output)
    elif isinstance(value, list):
        for item in value:
            _walk_artifacts(item, output)


def _stage_gate(
    replayed: dict[str, dict[str, Any]],
    treatment: str,
    reference: str,
) -> dict[str, Any]:
    treatment_aggregate = replayed[treatment]["aggregate"]
    reference_aggregate = replayed[reference]["aggregate"]
    wins = sum(
        treatment_view["metrics"]["psnr_fg"] > reference_view["metrics"]["psnr_fg"]
        for treatment_view, reference_view in zip(
            replayed[treatment]["views"],
            replayed[reference]["views"],
            strict=True,
        )
    )
    result = {
        "treatment": treatment,
        "reference": reference,
        "mean_psnr_delta_db": (treatment_aggregate["psnr_fg"] - reference_aggregate["psnr_fg"]),
        "view_wins": wins,
        "outside_ratio": (
            treatment_aggregate["coverage_outside"] / reference_aggregate["coverage_outside"]
        ),
    }
    result["passed"] = (
        result["mean_psnr_delta_db"] >= 0.10
        and result["view_wins"] >= 5
        and result["outside_ratio"] <= 1.10
    )
    return result


def _downstream_gate(
    summary: dict[str, Any],
    treatment: str,
    reference: str,
) -> dict[str, Any]:
    treatment_metrics = summary["reconstruction"][treatment]["final_metrics"]
    reference_metrics = summary["reconstruction"][reference]["final_metrics"]
    result = {
        "treatment": treatment,
        "reference": reference,
        "heldout_psnr_fg_delta_db": (
            treatment_metrics["test"]["psnr_fg"] - reference_metrics["test"]["psnr_fg"]
        ),
        "heldout_alpha_iou_delta": (
            treatment_metrics["test"]["alpha_iou"] - reference_metrics["test"]["alpha_iou"]
        ),
        "train_psnr_fg_delta_db": (
            treatment_metrics["train"]["psnr_fg"] - reference_metrics["train"]["psnr_fg"]
        ),
    }
    result["passed"] = (
        result["heldout_psnr_fg_delta_db"] >= 0.10
        and result["heldout_alpha_iou_delta"] >= -0.01
        and result["train_psnr_fg_delta_db"] >= -0.25
    )
    return result


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("metric replay requires the frozen CUDA/gsplat path")

    summary = _read(SUMMARY)
    plan = _read(RUN / "plan.json")
    prior = _read(PRIOR_SUMMARY)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check(
        "official summary hash and complete combination schema",
        _sha256(SUMMARY) == EXPECTED["summary"]
        and summary.get("schema") == "rtgs.pool_structure_wse_frame00008.result.v1"
        and summary.get("status") == "complete",
        {"summary_sha256": _sha256(SUMMARY), "status": summary.get("status")},
    )

    failure = _read(V2_FAILURE)
    protocol_chain_valid = (
        _sha256(PROTOCOL_V1) == EXPECTED["protocol_v1"]
        and _sha256(PROTOCOL_V2) == EXPECTED["protocol_v2"]
        and _sha256(PROTOCOL_V3) == EXPECTED["protocol_v3"]
        and _sha256(V2_FAILURE) == EXPECTED["v2_failure"]
        and failure["out_directory_created"] is False
        and failure["scene_loaded"] is False
        and failure["arm_started"] is False
        and failure["outcome_seen"] is False
    )
    check(
        "append-only protocol chain preserves the pre-plan V2 import failure",
        protocol_chain_valid,
        {
            "v1": EXPECTED["protocol_v1"],
            "v2": EXPECTED["protocol_v2"],
            "v3": EXPECTED["protocol_v3"],
            "v2_failure": EXPECTED["v2_failure"],
        },
    )

    protocol_before_plan = PROTOCOL_V3.stat().st_mtime_ns <= (RUN / "plan.json").stat().st_mtime_ns
    check(
        "V3 predates the plan and the plan binds V3",
        protocol_before_plan
        and plan["protocol"]["sha256"] == EXPECTED["protocol_v3"]
        and summary["protocol"]["sha256"] == EXPECTED["protocol_v3"],
        {
            "protocol_mtime_ns": PROTOCOL_V3.stat().st_mtime_ns,
            "plan_mtime_ns": (RUN / "plan.json").stat().st_mtime_ns,
            "bound_sha256": plan["protocol"]["sha256"],
        },
    )

    source_hashes = summary["repository"]["source_files"]
    current_source_hashes = {relative: _sha256(ROOT / relative) for relative in source_hashes}
    git_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    check(
        "executed source, wrapper, and git revision remain bound",
        current_source_hashes == source_hashes
        and source_hashes["benchmarks/pool_structure_wse_frame00008.py"] == EXPECTED["harness"]
        and summary["repository"]["git_revision"] == git_revision,
        {
            "harness_sha256": current_source_hashes["benchmarks/pool_structure_wse_frame00008.py"],
            "git_revision": git_revision,
            "source_count": len(source_hashes),
        },
    )

    artifacts: list[dict[str, Any]] = []
    _walk_artifacts(summary, artifacts)
    unique_artifacts = {record["path"]: record for record in artifacts}
    artifact_failures = []
    for record in unique_artifacts.values():
        path = _path(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or _sha256(path) != record["sha256"]
        ):
            artifact_failures.append(record["path"])
    check(
        "all summary-bound artifacts exist with exact bytes and hashes",
        not artifact_failures,
        {"unique_artifact_count": len(unique_artifacts), "failures": artifact_failures},
    )

    scene = load_calibrated_scene(
        RAW_SCENE,
        calibration_path=CALIBRATION,
        downscale=16,
        max_images=8,
        test_every=8,
        load_masks=True,
        undistort=True,
    )
    loaded_hashes_match = scene.masks is not None
    if loaded_hashes_match:
        for name, image, mask in zip(scene.view_names, scene.images, scene.masks, strict=True):
            record = summary["scene"]["inputs"][name]
            loaded_hashes_match &= (
                prior_audit._tensor_hash(image) == record["loaded_rgb_tensor_sha256"]
                and prior_audit._tensor_hash(mask) == record["loaded_mask_tensor_sha256"]
            )
    check(
        "calibrated tensors and seven-train/one-held-out split replay",
        tuple(scene.view_names or ()) == VIEW_NAMES
        and scene.training_views == list(range(7))
        and scene.testing_views == [7]
        and loaded_hashes_match,
        {
            "view_names": scene.view_names,
            "training_views": scene.training_views,
            "testing_views": scene.testing_views,
            "loaded_tensor_hashes_match": loaded_hashes_match,
        },
    )

    configs = plan["arms"]
    structure_diff = {
        key
        for key in configs["pool-structure-density"]
        if configs["pool-structure-density"][key] != configs["pool-structure-wse"][key]
    }
    pool_controls_valid = all(
        config["pool"] is True
        and config["pool_capacity"] == 1_280
        and config["n_gaussians"] == 640
        and config["iterations"] == 300
        and config["pool_triage_every"] == 50
        and config["pool_prune_count"] == config["pool_spawn_count"] == 32
        for config in configs.values()
    )
    treatment_valid = (
        tuple(configs) == ARMS
        and pool_controls_valid
        and configs["pool-gradient"]["init_strategy"] == "gradient"
        and configs["pool-structure-density"]["init_strategy"] == "structure_tensor"
        and configs["pool-structure-density"]["structure_sampling"] == "density"
        and configs["pool-structure-wse"]["init_strategy"] == "structure_tensor"
        and configs["pool-structure-wse"]["structure_sampling"] == "wse"
        and structure_diff == {"structure_sampling"}
    )
    check(
        "effective arms isolate WSE and hold the pool lifecycle fixed",
        treatment_valid,
        {
            "structure_arm_config_differences": sorted(structure_diff),
            "all_pool_controls_match": pool_controls_valid,
        },
    )

    stage1_replayed: dict[str, dict[str, Any]] = {}
    stage1_valid = True
    no_heldout_fit = True
    stored_metric_max_error = 0.0
    cuda_torch_max_abs = 0.0
    cuda_torch_mean_abs = []
    pool_history_valid = True
    for arm in ARMS:
        records = summary["stage1"][arm]["views"]
        no_heldout_fit &= [record["view_name"] for record in records] == list(VIEW_NAMES[:-1])
        replayed_views = []
        for index, record in enumerate(records):
            model = Gaussians2D.load_npz(_path(record["fit"]["path"]))
            stage1_valid &= model.n == 640 and prior_audit._finite_2d(model)
            cuda_metrics, cuda_image = prior_audit._stage1_metrics(
                model.to("cuda"),
                scene.images[index].to("cuda"),
                scene.masks[index].to("cuda"),
                renderer="cuda",
            )
            torch_metrics, torch_image = prior_audit._stage1_metrics(
                model,
                scene.images[index],
                scene.masks[index],
                renderer="torch",
            )
            for key, reported in record["metrics"].items():
                stored_metric_max_error = max(
                    stored_metric_max_error,
                    abs(float(reported) - cuda_metrics[key]),
                )
            difference = (cuda_image - torch_image).abs()
            cuda_torch_max_abs = max(cuda_torch_max_abs, float(difference.max()))
            cuda_torch_mean_abs.append(float(difference.mean()))
            history = _read(_path(record["history"]["path"]))
            pool_history_valid &= (
                history["pool_capacity"] == 1_280
                and history["live_count"] == 640
                and history["stopped_iter"] == 299
            )
            replayed_views.append({"metrics": cuda_metrics, "torch_metrics": torch_metrics})
        aggregate = {
            key: fmean(view["metrics"][key] for view in replayed_views)
            for key in replayed_views[0]["metrics"]
        }
        stage1_valid &= all(
            _close(aggregate[key], float(summary["stage1"][arm]["aggregate"][key]))
            for key in aggregate
        )
        stage1_replayed[arm] = {"views": replayed_views, "aggregate": aggregate}
    check(
        "all 21 pooled stage-1 fits and metrics replay",
        stage1_valid and pool_history_valid and no_heldout_fit and stored_metric_max_error <= 5e-4,
        {
            "stored_metric_max_abs_error": stored_metric_max_error,
            "cuda_vs_torch_render_max_abs": cuda_torch_max_abs,
            "cuda_vs_torch_render_mean_abs_mean": fmean(cuda_torch_mean_abs),
            "pool_histories_valid": pool_history_valid,
            "heldout_fit_absent": no_heldout_fit,
        },
    )

    stage1_gates = {
        "wse_vs_density": _stage_gate(
            stage1_replayed,
            "pool-structure-wse",
            "pool-structure-density",
        ),
        "density_vs_gradient": _stage_gate(
            stage1_replayed,
            "pool-structure-density",
            "pool-gradient",
        ),
        "wse_vs_gradient": _stage_gate(
            stage1_replayed,
            "pool-structure-wse",
            "pool-gradient",
        ),
    }
    check(
        "all frozen stage-1 gates recompute without a passing treatment",
        not any(gate["passed"] for gate in stage1_gates.values()),
        stage1_gates,
    )

    renderer3d = get_rasterizer(
        "gsplat",
        device=torch.device("cuda"),
        packed=False,
        antialiased=True,
    )
    models: dict[str, dict[str, Gaussians3D]] = {}
    metric_replay_max_error = 0.0
    model_valid = True
    for arm in ARMS:
        record = summary["reconstruction"][arm]
        initial = Gaussians3D.load_npz(_path(record["artifacts"]["initial_npz"]["path"]))
        final = Gaussians3D.load_npz(_path(record["artifacts"]["final_npz"]["path"]))
        initial_ply = Gaussians3D.load_ply(_path(record["artifacts"]["initial_ply"]["path"]))
        final_ply = Gaussians3D.load_ply(_path(record["artifacts"]["final_ply"]["path"]))
        model_valid &= (
            initial.n == record["init_n_gaussians"] == initial_ply.n
            and final.n == record["final_n_gaussians"] == final_ply.n
            and prior_audit._finite_3d(initial)
            and prior_audit._finite_3d(final)
            and prior_audit._finite_3d(initial_ply)
            and prior_audit._finite_3d(final_ply)
        )
        models[arm] = {"initial": initial, "final": final}
        for state, model in (("init", initial), ("final", final)):
            replayed = {
                "train": Trainer.evaluate_metrics(
                    scene,
                    model.to("cuda"),
                    renderer3d,
                    indices=scene.training_views,
                ),
                "test": Trainer.evaluate_metrics(
                    scene,
                    model.to("cuda"),
                    renderer3d,
                    indices=scene.testing_views,
                ),
            }
            reported = record[f"{state}_metrics"]
            for split in ("train", "test"):
                for key, value in replayed[split].items():
                    metric_replay_max_error = max(
                        metric_replay_max_error,
                        abs(float(value) - float(reported[split][key])),
                    )
    check(
        "all 3D NPZ/PLY states are finite and exact metrics replay",
        model_valid and metric_replay_max_error <= 5e-5,
        {
            "metric_replay_max_abs_error": metric_replay_max_error,
            "counts": {
                arm: {
                    "initial": models[arm]["initial"].n,
                    "final": models[arm]["final"].n,
                }
                for arm in ARMS
            },
        },
    )

    history_valid = True
    history_detail = {}
    for arm in ARMS:
        history = _read(_path(summary["reconstruction"][arm]["training_history"]["path"]))
        counts = [(int(step), int(count)) for step, count in history["n_gaussians"]]
        density_steps = [
            step
            for (previous_step, previous_count), (step, count) in zip(
                counts,
                counts[1:],
                strict=False,
            )
            if previous_step < step and previous_count != count
        ]
        last_surgery = max(density_steps, default=0)
        recovery_steps = 2_000 - last_surgery
        history_valid &= (
            len(history["loss"]) == 2_000
            and len(history["sampled_train_views"]) == 2_000
            and set(history["sampled_train_views"]) <= set(range(7))
            and counts[-1][0] == 2_000
            and counts[-1][1] == models[arm]["final"].n
            and last_surgery <= 1_000
            and recovery_steps >= 1_000
            and "selected_step" not in history
        )
        history_detail[arm] = {
            "density_change_steps": density_steps,
            "last_surgery": last_surgery,
            "recovery_steps": recovery_steps,
            "sampled_view_ids": sorted(set(history["sampled_train_views"])),
        }
    check(
        "training histories are complete, train-only, final-policy, and recovered after density",
        history_valid,
        history_detail,
    )

    downstream_gates = {
        "wse_vs_density": _downstream_gate(
            summary,
            "pool-structure-wse",
            "pool-structure-density",
        ),
        "density_vs_gradient": _downstream_gate(
            summary,
            "pool-structure-density",
            "pool-gradient",
        ),
        "wse_vs_gradient": _downstream_gate(
            summary,
            "pool-structure-wse",
            "pool-gradient",
        ),
    }
    check(
        "frozen downstream gates replay with only WSE versus density passing",
        downstream_gates["wse_vs_density"]["passed"]
        and not downstream_gates["density_vs_gradient"]["passed"]
        and not downstream_gates["wse_vs_gradient"]["passed"],
        downstream_gates,
    )

    prior_anchor = prior["reconstruction"]["pool"]
    current_anchor = summary["reconstruction"]["pool-gradient"]
    anchor_drift = {
        "stage1_mean_psnr_fg_delta_db": (
            summary["stage1"]["pool-gradient"]["aggregate"]["psnr_fg"]
            - prior["stage1"]["pool"]["aggregate"]["psnr_fg"]
        ),
        "heldout_final_psnr_fg_delta_db": (
            current_anchor["final_metrics"]["test"]["psnr_fg"]
            - prior_anchor["final_metrics"]["test"]["psnr_fg"]
        ),
        "heldout_final_alpha_iou_delta": (
            current_anchor["final_metrics"]["test"]["alpha_iou"]
            - prior_anchor["final_metrics"]["test"]["alpha_iou"]
        ),
        "prior_final_count": prior_anchor["final_n_gaussians"],
        "current_final_count": current_anchor["final_n_gaussians"],
    }
    prior_config = dict(prior["stage1"]["pool"]["config"])
    current_config = dict(summary["stage1"]["pool-gradient"]["config"])
    current_config.pop("structure_sampling")
    check(
        (
            "prior pooled-gradient anchor is hash-bound and config-matched except the new "
            "default field"
        ),
        _sha256(PRIOR_SUMMARY) == EXPECTED["prior_summary"] and current_config == prior_config,
        anchor_drift,
    )

    viewer = _read(VIEWER)
    receipt = _read(VIEWER_RECEIPT)
    viewer_valid = (
        _sha256(VIEWER) == receipt["manifest"]["sha256"]
        and viewer["schema"] == "rtgs.viewer-comparison.v1"
        and [item["name"] for item in viewer["methods"]] == list(ARMS)
        and receipt["server"]["http_status"] == 200
        and receipt["server"]["response_bytes"] > 0
        and receipt["server"]["cuda_visible_devices"] == ""
        and receipt["server"]["nvidia_compute_processes_at_check"] == []
        and receipt["server"]["pid_owned_listening_socket"] is True
        and receipt["server"]["pid_stopped_after_stop"] is True
        and receipt["server"]["port_closed_after_stop"] is True
    )
    for item in viewer["methods"]:
        arm = item["name"]
        initial = (VIEWER.parent / item["initial"]).resolve()
        final = (VIEWER.parent / item["final"]).resolve()
        viewer_valid &= (
            initial
            == _path(summary["reconstruction"][arm]["artifacts"]["initial_ply"]["path"]).resolve()
            and final
            == _path(summary["reconstruction"][arm]["artifacts"]["final_ply"]["path"]).resolve()
            and _sha256(initial) == receipt["models"][arm]["initial_sha256"]
            and _sha256(final) == receipt["models"][arm]["final_sha256"]
        )
    check(
        "CPU viewer loaded all six bound models and shut down cleanly",
        viewer_valid,
        {
            "manifest_sha256": _sha256(VIEWER),
            "methods": [item["name"] for item in viewer["methods"]],
            "http_status": receipt["server"]["http_status"],
            "response_bytes": receipt["server"]["response_bytes"],
        },
    )

    visual_paths = [_path(record["path"]) for record in summary["comparison_visuals"].values()]
    for arm in ARMS:
        visual_paths.extend(
            _path(record["path"]) for record in summary["reconstruction"][arm]["visuals"].values()
        )
    visuals_valid = True
    visual_detail = {}
    for path in visual_paths:
        with Image.open(path) as image:
            visuals_valid &= image.width > 0 and image.height > 0
            visual_detail[str(path.relative_to(ROOT))] = {
                "size": [image.width, image.height],
                "frames": getattr(image, "n_frames", 1),
            }
    check(
        "all calibrated, cross-arm, and novel-view visuals decode",
        visuals_valid and len(visual_paths) == 15,
        {"visual_count": len(visual_paths), "artifacts": visual_detail},
    )

    claim_disposition = {
        "no_wse_density_control_is_a_matched_opt_in_seam": (
            "confirm_mechanism_and_effective_config_default_unchanged"
        ),
        "wse_improves_pooled_structure_stage1": (
            "retire_failed_primary_quality_gate_negative_delta_and_3_of_7_wins"
        ),
        "wse_improves_pooled_structure_downstream": (
            "narrow_single_scene_single_seed_gate_pass_only_not_balanced_or_robust"
        ),
        "pooled_structure_density_beats_pooled_gradient": (
            "retire_failed_stage1_and_downstream_psnr_gates"
        ),
        "pooled_structure_wse_beats_pooled_gradient": (
            "retire_failed_stage1_and_downstream_materiality_gates"
        ),
        "wse_by_pool_interaction": "not_tested_all_arms_are_pooled",
        "default_or_generalization_claim": "not_authorized",
        "performance_or_memory_claim": "not_authorized_contended_unrepeated_execution",
    }

    findings = [
        (
            "WSE lost its direct pooled stage-1 contrast: "
            f"{stage1_gates['wse_vs_density']['mean_psnr_delta_db']:+.4f} dB, "
            f"{stage1_gates['wse_vs_density']['view_wins']}/7 wins, outside ratio "
            f"{stage1_gates['wse_vs_density']['outside_ratio']:.4f}."
        ),
        (
            "WSE passed the direct downstream gate versus density with "
            f"{downstream_gates['wse_vs_density']['heldout_psnr_fg_delta_db']:+.4f} dB "
            "held-out foreground PSNR, "
            f"{downstream_gates['wse_vs_density']['heldout_alpha_iou_delta']:+.5f} alpha IoU, "
            "and "
            f"{downstream_gates['wse_vs_density']['train_psnr_fg_delta_db']:+.4f} dB train PSNR."
        ),
        (
            "Neither pooled structure arm beat pooled gradient at stage 1 or passed its "
            "downstream materiality gate; no combined method is a balanced winner."
        ),
        (
            "The nominally unchanged pooled-gradient anchor moved "
            f"{anchor_drift['heldout_final_psnr_fg_delta_db']:+.4f} dB held out and changed "
            f"count {anchor_drift['prior_final_count']}->{anchor_drift['current_final_count']} "
            "relative to the prior run, so the marginal +0.1137 dB WSE endpoint contrast is not "
            "replication-grade evidence."
        ),
    ]

    audit = {
        "schema": "rtgs.pool_structure_wse_frame00008.audit.v1",
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "summary": {
            "path": str(SUMMARY.relative_to(ROOT)),
            "sha256": _sha256(SUMMARY),
        },
        "scope": (
            "single-scene, single-seed development audit with seven training cameras and one "
            "held-out camera; WSE effect conditional on the frozen pool only"
        ),
        "stage1_gates": stage1_gates,
        "downstream_gates": downstream_gates,
        "prior_anchor_drift": anchor_drift,
        "claim_disposition": claim_disposition,
        "findings": findings,
        "checks": checks,
        "check_summary": {
            "total": len(checks),
            "passed": sum(item["passed"] for item in checks),
            "failed": sum(not item["passed"] for item in checks),
        },
        "commands": {
            "official": (
                "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 "
                ".venv-cuda/bin/python benchmarks/pool_structure_wse_frame00008.py "
                "--protocol benchmarks/results/"
                "20260724_pool_structure_wse_frame00008_PREREG_V3.md "
                "--out runs/pool_structure_wse_frame00008_20260724"
            ),
            "audit": (
                "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 "
                ".venv-cuda/bin/python benchmarks/audit_pool_structure_wse_frame00008.py"
            ),
            "viewer": receipt["command"],
        },
    }
    OUTPUT.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(audit["check_summary"], indent=2))
    print(json.dumps({"stage1": stage1_gates, "downstream": downstream_gates}, indent=2))
    return 1 if audit["check_summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
