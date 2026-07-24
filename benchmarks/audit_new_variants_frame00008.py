#!/usr/bin/env python3
"""Independent results audit for the 2026-07-24 Janelle new-variants experiment."""

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

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.metrics import image_metrics
from rtgs.data.calibrated import load_calibrated_scene
from rtgs.image2gs.renderer2d import render_gaussian_coverage_2d, render_gaussians_2d
from rtgs.optim.trainer import Trainer
from rtgs.render.base import get_rasterizer

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/new_variants_frame00008_20260724_v3"
SUMMARY = RUN / "summary.json"
RAW_SCENE = Path("/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008")
CALIBRATION = ROOT / "dataset/2025_03_07_stage_with_fabric/calibration_dome.json"
PROTOCOL_V1 = ROOT / "benchmarks/results/20260724_new_variants_frame00008_PREREG.md"
PROTOCOL_V2 = ROOT / "benchmarks/results/20260724_new_variants_frame00008_PREREG_V2.md"
PROTOCOL_V3 = ROOT / "benchmarks/results/20260724_new_variants_frame00008_PREREG_V3.md"
VIEWER = ROOT / "benchmarks/results/20260724_new_variants_frame00008_VIEWER.json"
VIEWER_RECEIPT = ROOT / "benchmarks/results/20260724_new_variants_frame00008_VIEWER_RECEIPT.json"
OUTPUT = ROOT / "benchmarks/results/20260724_new_variants_frame00008_AUDIT.json"

STAGE1_ARMS = ("baseline", "pool", "mask-containment", "structure-tensor")
REPORT_ARMS = STAGE1_ARMS + ("best-train-checkpoint",)
VIEW_NAMES = ("C0001", "C0008", "C0014", "C0021", "C0026", "C0031", "C0039", "C1004")
EXPECTED = {
    "summary": "f302b6eaaae6eac8dd7e0894b371f8860df03d047ebb73e037e72ee90be166e9",
    "harness": "d0f429352a28bdb1584cc30ff9b92a7a70b94c168966a19e4785876ea7cc1e8c",
    "protocol_v1": "eb8e053d823462485f0c1e7b11f269aae0b0efbe25f8136c3a22e37824b14a22",
    "protocol_v2": "2230b2f8a0b78918308f6fc850586e616e24447acb333fc002e8c243e0ba4b90",
    "protocol_v3": "649f93fca12c71437c7a44fd742935d1e7d446ade9ea7950913444ef8395816d",
    "attempt_v1_plan": "4184a61f0d8ef1a2a96568bfd79243a21ba892b593afb639403e3c6d5bd44e07",
    "attempt_v1_failure": "95d5dad253eba9dd46e70ecc9d7a2e64c4c2f5fc2a1508a87dd6a09150cb7642",
    "attempt_v2_plan": "e8849d3222d48b60962fbee68f8a759b83686710c69b5b1eceec34aa84bdcc9f",
    "attempt_v2_failure": "7d0841fdd600687755f3fc20ca50b48b6722f56f279db959a5e0bb18583bbac2",
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


def _tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _path(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def _close(left: float, right: float, tolerance: float = 5e-5) -> bool:
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance


def _all_fields_equal(left: Gaussians3D, right: Gaussians3D) -> bool:
    return all(
        torch.equal(getattr(left, name), getattr(right, name))
        for name in ("means", "quats", "log_scales", "opacity", "sh")
    )


def _finite_2d(model: Gaussians2D) -> bool:
    return (
        all(
            bool(torch.isfinite(value).all())
            for value in (model.xy, model.chol, model.color, model.weight)
        )
        and bool((model.chol[:, 0] > 0).all())
        and bool((model.chol[:, 2] > 0).all())
        and bool(((model.weight >= 0) & (model.weight <= 1)).all())
        and bool(((model.color >= 0) & (model.color <= 1)).all())
    )


def _finite_3d(model: Gaussians3D) -> bool:
    fields = (model.means, model.quats, model.log_scales, model.opacity, model.sh)
    return (
        all(bool(torch.isfinite(value).all()) for value in fields)
        and bool((model.quats.norm(dim=-1) > 0).all())
        and bool(((model.opacity >= 0) & (model.opacity <= 1)).all())
        and bool((model.scales > 0).all())
    )


def _stage1_metrics(
    model: Gaussians2D,
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    renderer: str,
) -> tuple[dict[str, float], torch.Tensor]:
    height, width = image.shape[:2]
    with torch.no_grad():
        prediction = render_gaussians_2d(
            model, height, width, row_chunk=64, renderer=renderer
        ).clamp(0.0, 1.0)
        coverage = render_gaussian_coverage_2d(model, height, width, row_chunk=64)
        foreground = mask > 0.5
        background = ~foreground
        values = image_metrics(prediction, image, mask)
        masked_target = image * mask[..., None]
        masked_mse = (prediction - masked_target).square().mean().clamp_min(1e-12)
        support = coverage > 0.10
        intersection = (support & foreground).sum()
        union = (support | foreground).sum().clamp_min(1)
        result = {
            "psnr_fg": float(values["psnr_fg"]),
            "psnr_crop": float(values["psnr_crop"]),
            "ssim_crop": float(values["ssim_crop"]),
            "masked_full_psnr": float(-10.0 * torch.log10(masked_mse)),
            "coverage_iou_at_0_1": float(intersection / union),
            "coverage_inside": float(coverage[foreground].mean()),
            "coverage_outside": (
                float(coverage[background].mean()) if bool(background.any()) else 0.0
            ),
            "foreground_support_recall_at_0_1": float(support[foreground].float().mean()),
            "n_gaussians": float(model.n),
        }
    return result, prediction.detach().cpu()


def _walk_artifacts(value: Any, output: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"} <= set(value):
            output.append(value)
        for item in value.values():
            _walk_artifacts(item, output)
    elif isinstance(value, list):
        for item in value:
            _walk_artifacts(item, output)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("metric replay requires the CUDA/gsplat execution path")
    summary = _read(SUMMARY)
    plan = _read(RUN / "plan.json")
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check(
        "official summary hash and complete schema",
        _sha256(SUMMARY) == EXPECTED["summary"]
        and summary.get("schema") == "rtgs.new_variants_frame00008.result.v1"
        and summary.get("status") == "complete",
        {"summary_sha256": _sha256(SUMMARY), "status": summary.get("status")},
    )
    check(
        "append-only protocol chain and failed attempts preserved",
        _sha256(PROTOCOL_V1) == EXPECTED["protocol_v1"]
        and _sha256(PROTOCOL_V2) == EXPECTED["protocol_v2"]
        and _sha256(PROTOCOL_V3) == EXPECTED["protocol_v3"]
        and _sha256(ROOT / "runs/new_variants_frame00008_20260724/plan.json")
        == EXPECTED["attempt_v1_plan"]
        and _sha256(ROOT / "runs/new_variants_frame00008_20260724/failure.json")
        == EXPECTED["attempt_v1_failure"]
        and _sha256(ROOT / "runs/new_variants_frame00008_20260724_v2/plan.json")
        == EXPECTED["attempt_v2_plan"]
        and _sha256(ROOT / "runs/new_variants_frame00008_20260724_v2/failure.json")
        == EXPECTED["attempt_v2_failure"],
        {
            "v1": EXPECTED["protocol_v1"],
            "v2": EXPECTED["protocol_v2"],
            "v3": EXPECTED["protocol_v3"],
        },
    )
    protocol_before_plan = PROTOCOL_V3.stat().st_mtime_ns <= (RUN / "plan.json").stat().st_mtime_ns
    check(
        "v3 protocol predates official plan and plan binds protocol",
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
    check(
        "executed source and git revision remain bound",
        current_source_hashes == source_hashes
        and source_hashes["benchmarks/new_variants_frame00008.py"] == EXPECTED["harness"]
        and summary["repository"]["git_revision"]
        == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        {
            "harness_sha256": current_source_hashes["benchmarks/new_variants_frame00008.py"],
            "git_revision": summary["repository"]["git_revision"],
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
                _tensor_hash(image) == record["loaded_rgb_tensor_sha256"]
                and _tensor_hash(mask) == record["loaded_mask_tensor_sha256"]
            )
    check(
        "calibrated view order, tensor hashes, and seven/one split replay",
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

    stage1_replayed: dict[str, dict[str, Any]] = {}
    stage1_valid = True
    stored_metric_max_error = 0.0
    cuda_torch_max_abs = 0.0
    cuda_torch_mean_abs = []
    no_heldout_fit = True
    for arm in STAGE1_ARMS:
        arm_records = summary["stage1"][arm]["views"]
        no_heldout_fit &= [record["view_name"] for record in arm_records] == list(VIEW_NAMES[:-1])
        replayed_views = []
        for index, record in enumerate(arm_records):
            model = Gaussians2D.load_npz(_path(record["fit"]["path"]))
            stage1_valid &= model.n == int(record["metrics"]["n_gaussians"])
            stage1_valid &= _finite_2d(model)
            cuda_metrics, cuda_image = _stage1_metrics(
                model.to("cuda"),
                scene.images[index].to("cuda"),
                scene.masks[index].to("cuda"),
                renderer="cuda",
            )
            torch_metrics, torch_image = _stage1_metrics(
                model,
                scene.images[index],
                scene.masks[index],
                renderer="torch",
            )
            for key, reported in record["metrics"].items():
                stored_metric_max_error = max(
                    stored_metric_max_error, abs(float(reported) - cuda_metrics[key])
                )
            difference = (cuda_image - torch_image).abs()
            cuda_torch_max_abs = max(cuda_torch_max_abs, float(difference.max()))
            cuda_torch_mean_abs.append(float(difference.mean()))
            replayed_views.append({"metrics": cuda_metrics, "torch_metrics": torch_metrics})
        aggregates = {
            key: fmean(view["metrics"][key] for view in replayed_views)
            for key in replayed_views[0]["metrics"]
        }
        aggregate_match = all(
            _close(aggregates[key], float(summary["stage1"][arm]["aggregate"][key]))
            for key in aggregates
        )
        stage1_valid &= aggregate_match
        stage1_replayed[arm] = {
            "views": replayed_views,
            "aggregate": aggregates,
        }
    check(
        "all 28 stage-1 fits replay with finite valid fields and metrics",
        stage1_valid and stored_metric_max_error <= 5e-4,
        {
            "stored_metric_max_abs_error": stored_metric_max_error,
            "cuda_vs_torch_render_max_abs": cuda_torch_max_abs,
            "cuda_vs_torch_render_mean_abs_mean": fmean(cuda_torch_mean_abs),
        },
    )
    check(
        "held-out C1004 has no stage-1 fit artifact",
        no_heldout_fit
        and not any((RUN / "stage1" / arm / "C1004.npz").exists() for arm in STAGE1_ARMS),
        {"stage1_view_names": list(VIEW_NAMES[:-1]), "heldout": "C1004"},
    )

    baseline_stage1 = stage1_replayed["baseline"]["aggregate"]
    stage1_gates: dict[str, Any] = {}
    for arm in ("pool", "structure-tensor"):
        treatment = stage1_replayed[arm]["aggregate"]
        wins = sum(
            treatment_view["metrics"]["psnr_fg"] > baseline_view["metrics"]["psnr_fg"]
            for treatment_view, baseline_view in zip(
                stage1_replayed[arm]["views"],
                stage1_replayed["baseline"]["views"],
                strict=True,
            )
        )
        stage1_gates[arm] = {
            "mean_psnr_delta_db": treatment["psnr_fg"] - baseline_stage1["psnr_fg"],
            "view_wins": wins,
            "outside_ratio": treatment["coverage_outside"] / baseline_stage1["coverage_outside"],
        }
    containment = stage1_replayed["mask-containment"]["aggregate"]
    stage1_gates["mask-containment"] = {
        "outside_reduction_fraction": 1.0
        - containment["coverage_outside"] / baseline_stage1["coverage_outside"],
        "mean_psnr_delta_db": containment["psnr_fg"] - baseline_stage1["psnr_fg"],
        "inside_ratio": containment["coverage_inside"] / baseline_stage1["coverage_inside"],
    }
    stage1_gates["pool"]["passed"] = (
        stage1_gates["pool"]["mean_psnr_delta_db"] >= 0.10
        and stage1_gates["pool"]["view_wins"] >= 5
        and stage1_gates["pool"]["outside_ratio"] <= 1.10
    )
    stage1_gates["structure-tensor"]["passed"] = (
        stage1_gates["structure-tensor"]["mean_psnr_delta_db"] >= 0.10
        and stage1_gates["structure-tensor"]["view_wins"] >= 5
        and stage1_gates["structure-tensor"]["outside_ratio"] <= 1.10
    )
    stage1_gates["mask-containment"]["passed"] = (
        stage1_gates["mask-containment"]["outside_reduction_fraction"] >= 0.20
        and stage1_gates["mask-containment"]["mean_psnr_delta_db"] >= -0.25
        and stage1_gates["mask-containment"]["inside_ratio"] >= 0.95
    )
    pool_mechanism = all(
        record["pool"]["capacity"] == 1_280
        and record["pool"]["live_count"] == 640
        and int(record["metrics"]["n_gaussians"]) == 640
        for record in summary["stage1"]["pool"]["views"]
    )
    check(
        "pool mechanism and all frozen stage-1 gates recomputed",
        pool_mechanism
        and stage1_gates["pool"]["passed"]
        and not stage1_gates["mask-containment"]["passed"]
        and not stage1_gates["structure-tensor"]["passed"],
        {"pool_mechanism": pool_mechanism, "gates": stage1_gates},
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
    for arm in REPORT_ARMS:
        record = summary["reconstruction"][arm]
        initial = Gaussians3D.load_npz(_path(record["artifacts"]["initial_npz"]["path"]))
        final = Gaussians3D.load_npz(_path(record["artifacts"]["final_npz"]["path"]))
        initial_ply = Gaussians3D.load_ply(_path(record["artifacts"]["initial_ply"]["path"]))
        final_ply = Gaussians3D.load_ply(_path(record["artifacts"]["final_ply"]["path"]))
        model_valid &= (
            initial.n == record["init_n_gaussians"] == initial_ply.n
            and final.n == record["final_n_gaussians"] == final_ply.n
            and _finite_3d(initial)
            and _finite_3d(final)
            and _finite_3d(initial_ply)
            and _finite_3d(final_ply)
        )
        models[arm] = {"initial": initial, "final": final}
        for state, model in (("init", initial), ("final", final)):
            replayed = {
                "train": Trainer.evaluate_metrics(
                    scene, model.to("cuda"), renderer3d, indices=scene.training_views
                ),
                "test": Trainer.evaluate_metrics(
                    scene, model.to("cuda"), renderer3d, indices=scene.testing_views
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
        "all NPZ/PLY models are finite with exact counts and 3D metrics replay",
        model_valid and metric_replay_max_error <= 5e-5,
        {
            "metric_replay_max_abs_error": metric_replay_max_error,
            "counts": {
                arm: {
                    "initial": models[arm]["initial"].n,
                    "final": models[arm]["final"].n,
                }
                for arm in REPORT_ARMS
            },
        },
    )

    baseline_history = _read(
        _path(summary["reconstruction"]["baseline"]["training_history"]["path"])
    )
    checkpoint_pair_exact = (
        _all_fields_equal(models["baseline"]["initial"], models["best-train-checkpoint"]["initial"])
        and _all_fields_equal(models["baseline"]["final"], models["best-train-checkpoint"]["final"])
        and summary["reconstruction"]["baseline"]["training_history"]["sha256"]
        == summary["reconstruction"]["best-train-checkpoint"]["training_history"]["sha256"]
    )
    selected_values = {int(step): float(value) for step, value in baseline_history["train_psnr"]}
    checkpoint_valid = (
        baseline_history["checkpoint_selection_views"] == list(range(7))
        and 7 not in baseline_history["checkpoint_selection_views"]
        and baseline_history["selected_step"]
        == max(selected_values, key=selected_values.__getitem__)
        and _close(
            baseline_history["selected_train_psnr"],
            max(selected_values.values()),
            tolerance=1e-8,
        )
        and set(baseline_history["sampled_train_views"]) <= set(range(7))
        and len(baseline_history["sampled_train_views"]) == 2_000
    )
    check(
        "best-checkpoint arm is an exact paired final-step selection with no test access",
        checkpoint_pair_exact and checkpoint_valid and baseline_history["selected_step"] == 2_000,
        {
            "selected_step": baseline_history["selected_step"],
            "selected_train_psnr": baseline_history["selected_train_psnr"],
            "selection_views": baseline_history["checkpoint_selection_views"],
            "models_bit_exact": checkpoint_pair_exact,
        },
    )

    history_valid = True
    history_detail = {}
    for arm in STAGE1_ARMS:
        history = _read(_path(summary["reconstruction"][arm]["training_history"]["path"]))
        density_steps = [int(item["iteration"]) for item in history["density_stats"]]
        recovery_steps = 2_000 - max(density_steps)
        final_count = int(history["n_gaussians"][-1][1])
        history_valid &= (
            len(history["loss"]) == 2_000
            and len(history["sampled_train_views"]) == 2_000
            and set(history["sampled_train_views"]) <= set(range(7))
            and history["n_gaussians"][-1][0] == 2_000
            and final_count == models[arm]["final"].n
            and recovery_steps >= 1_000
        )
        history_detail[arm] = {
            "density_steps": density_steps,
            "recovery_steps_after_last_surgery": recovery_steps,
            "final_count": final_count,
        }
    check(
        "training histories are complete with train-only sampling and post-surgery recovery",
        history_valid,
        history_detail,
    )

    baseline_final = summary["reconstruction"]["baseline"]["final_metrics"]
    downstream_gates = {}
    for arm in ("pool", "mask-containment", "structure-tensor"):
        final = summary["reconstruction"][arm]["final_metrics"]
        downstream_gates[arm] = {
            "heldout_psnr_fg_delta_db": final["test"]["psnr_fg"]
            - baseline_final["test"]["psnr_fg"],
            "heldout_alpha_iou_delta": final["test"]["alpha_iou"]
            - baseline_final["test"]["alpha_iou"],
            "train_psnr_fg_delta_db": final["train"]["psnr_fg"]
            - baseline_final["train"]["psnr_fg"],
        }
        downstream_gates[arm]["passed"] = (
            downstream_gates[arm]["heldout_psnr_fg_delta_db"] >= 0.10
            and downstream_gates[arm]["heldout_alpha_iou_delta"] >= -0.01
            and downstream_gates[arm]["train_psnr_fg_delta_db"] >= -0.25
        )
    checkpoint_final = summary["reconstruction"]["best-train-checkpoint"]["final_metrics"]
    downstream_gates["best-train-checkpoint"] = {
        "selected_step": baseline_history["selected_step"],
        "heldout_psnr_fg_delta_db": checkpoint_final["test"]["psnr_fg"]
        - baseline_final["test"]["psnr_fg"],
        "heldout_alpha_iou_delta": checkpoint_final["test"]["alpha_iou"]
        - baseline_final["test"]["alpha_iou"],
    }
    downstream_gates["best-train-checkpoint"]["passed"] = (
        baseline_history["selected_step"] < 2_000
        and downstream_gates["best-train-checkpoint"]["heldout_psnr_fg_delta_db"] >= 0.10
        and downstream_gates["best-train-checkpoint"]["heldout_alpha_iou_delta"] >= -0.01
    )
    check(
        "all frozen downstream gates recomputed",
        downstream_gates["pool"]["passed"]
        and downstream_gates["mask-containment"]["passed"]
        and not downstream_gates["structure-tensor"]["passed"]
        and not downstream_gates["best-train-checkpoint"]["passed"],
        downstream_gates,
    )

    viewer = _read(VIEWER)
    receipt = _read(VIEWER_RECEIPT)
    viewer_valid = (
        _sha256(VIEWER) == receipt["manifest"]["sha256"]
        and viewer["schema"] == "rtgs.viewer-comparison.v1"
        and [item["name"] for item in viewer["methods"]] == list(REPORT_ARMS)
        and receipt["server"]["http_status"] == 200
        and receipt["server"]["response_bytes"] > 0
        and receipt["server"]["cuda_visible_devices"] == ""
        and receipt["server"]["nvidia_compute_processes_at_check"] == []
        and receipt["server"]["port_closed_after_stop"] is True
        and receipt["server"]["pid_stopped_after_stop"] is True
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
        "comparison viewer loaded all ten bound CPU models and shut down cleanly",
        viewer_valid,
        {
            "manifest_sha256": _sha256(VIEWER),
            "methods": [item["name"] for item in viewer["methods"]],
            "http_status": receipt["server"]["http_status"],
            "response_bytes": receipt["server"]["response_bytes"],
        },
    )

    visual_paths = [_path(record["path"]) for record in summary["comparison_visuals"].values()]
    for arm in REPORT_ARMS:
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
        "calibrated, cross-arm, and novel-view visual artifacts decode",
        visuals_valid,
        {"visual_count": len(visual_paths), "artifacts": visual_detail},
    )

    claim_disposition = {
        "pool_stage1_quality": "confirm_single_scene_single_seed_development",
        "pool_downstream_heldout_gain": "confirm_single_scene_single_heldout_camera_development",
        "mask_containment_stage1_useful_at_weight_5": (
            "retire_failed_quality_and_inside_coverage_gates"
        ),
        "mask_containment_downstream_heldout_gain": (
            "confirm_surprising_single_scene_development_result_requires_replication"
        ),
        "structure_tensor_stage1_useful": "retire_failed_outside_coverage_guardrail",
        "structure_tensor_downstream_gain": "retire_below_material_psnr_floor",
        "best_train_checkpoint_improves_result": "retire_selected_final_step_exactly",
        "default_or_generalization_claim": "not_authorized",
        "performance_or_memory_claim": "not_authorized_contended_unrepeated_execution",
    }
    findings = [
        (
            "Pool recycling passed its frozen stage-1 gate: +"
            f"{stage1_gates['pool']['mean_psnr_delta_db']:.4f} dB, "
            f"{stage1_gates['pool']['view_wins']}/7 view wins, and "
            f"{(1.0 - stage1_gates['pool']['outside_ratio']) * 100.0:.2f}% less outside coverage."
        ),
        (
            "Pool also passed the held-out downstream gate: "
            f"{downstream_gates['pool']['heldout_psnr_fg_delta_db']:+.4f} dB foreground PSNR, "
            f"{downstream_gates['pool']['heldout_alpha_iou_delta']:+.5f} alpha IoU, and "
            f"{downstream_gates['pool']['train_psnr_fg_delta_db']:+.4f} dB train PSNR."
        ),
        (
            "Containment weight 5.0 cut outside stage-1 coverage by "
            f"{stage1_gates['mask-containment']['outside_reduction_fraction'] * 100.0:.2f}% "
            f"but lost {abs(stage1_gates['mask-containment']['mean_psnr_delta_db']):.4f} dB "
            "foreground PSNR, so the intended stage-1 gate failed."
        ),
        (
            "Despite that failed local gate, the containment arm passed the downstream gate with "
            f"{downstream_gates['mask-containment']['heldout_psnr_fg_delta_db']:+.4f} dB held-out "
            "foreground PSNR; this is a surprising development observation, not a selected weight."
        ),
        (
            "Structure initialization improved mean stage-1 foreground PSNR by "
            f"{stage1_gates['structure-tensor']['mean_psnr_delta_db']:.4f} dB but raised outside "
            f"coverage by {(stage1_gates['structure-tensor']['outside_ratio'] - 1.0) * 100.0:.2f}% "
            "and added only "
            f"{downstream_gates['structure-tensor']['heldout_psnr_fg_delta_db']:+.4f} dB held out."
        ),
        "Train-only checkpoint selection chose step 2000; selected and final models are bit-exact.",
    ]

    audit = {
        "schema": "rtgs.new_variants_frame00008.audit.v1",
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "summary": {
            "path": str(SUMMARY.relative_to(ROOT)),
            "sha256": _sha256(SUMMARY),
        },
        "scope": (
            "single-scene, single-seed development audit with seven training cameras and one "
            "held-out camera; quality evidence only, no default/generalization/performance claim"
        ),
        "stage1_gates": stage1_gates,
        "downstream_gates": downstream_gates,
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
                ".venv-cuda/bin/python benchmarks/new_variants_frame00008.py "
                "--protocol benchmarks/results/"
                "20260724_new_variants_frame00008_PREREG_V3.md "
                "--out runs/new_variants_frame00008_20260724_v3"
            ),
            "audit": ".venv-cuda/bin/python benchmarks/audit_new_variants_frame00008.py",
            "viewer": receipt["command"],
        },
    }
    OUTPUT.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(audit["check_summary"], indent=2))
    print(json.dumps({"stage1": stage1_gates, "downstream": downstream_gates}, indent=2))
    if audit["check_summary"]["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
