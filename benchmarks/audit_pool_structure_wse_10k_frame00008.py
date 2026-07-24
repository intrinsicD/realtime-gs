#!/usr/bin/env python3
"""Independent scientist pass for the pooled structure/WSE 10k checkpoint study."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urlsplit

import torch
from PIL import Image

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.calibrated import load_calibrated_scene
from rtgs.optim.trainer import Trainer
from rtgs.render.base import get_rasterizer

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/pool_structure_wse_10k_frame00008_20260724"
SUMMARY = RUN / "summary.json"
PARENT_RUN = ROOT / "runs/pool_structure_wse_frame00008_20260724"
PARENT_SUMMARY = PARENT_RUN / "summary.json"
PROTOCOL = ROOT / "benchmarks/results/20260724_pool_structure_wse_10k_frame00008_PREREG.md"
VIEWER = ROOT / "benchmarks/results/20260724_pool_structure_wse_10k_frame00008_VIEWER.json"
VIEWER_RECEIPT = (
    ROOT / "benchmarks/results/20260724_pool_structure_wse_10k_frame00008_VIEWER_RECEIPT.json"
)
RESULT_NOTE = ROOT / "benchmarks/results/20260724_pool_structure_wse_10k_frame00008_RESULT.md"
OUTPUT = ROOT / "benchmarks/results/20260724_pool_structure_wse_10k_frame00008_AUDIT.json"
RAW_SCENE = Path("/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008")
CALIBRATION = ROOT / "dataset/2025_03_07_stage_with_fabric/calibration_dome.json"

ARMS = (
    "pool-gradient",
    "pool-structure-density",
    "pool-structure-wse",
)
STEPS = (2_000, 4_000, 6_000, 8_000, 10_000)
VIEW_NAMES = ("C0001", "C0008", "C0014", "C0021", "C0026", "C0031", "C0039", "C1004")
CONTRASTS = {
    "density_vs_gradient": ("pool-structure-density", "pool-gradient"),
    "wse_vs_gradient": ("pool-structure-wse", "pool-gradient"),
    "wse_vs_density": ("pool-structure-wse", "pool-structure-density"),
}
EXPECTED = {
    "summary": "6fdabac92cd0bf1d4ad610f90083ecd75f0942c15cf16278809fa0ba46baf01b",
    "parent_summary": "83c832b920a4603937112f4ff177ca8ac4d420dc58e72e97e847e7c896e176eb",
    "protocol": "93d1ec116a1f23c88c970393809742574e6e92b1e95479d1f347db92c39eb3c8",
    "harness": "c163af7328cc7d3ad599101fcf4ddeb0ed90f97755cc82409883f4823c6d9c69",
    "shared_harness": "d0f429352a28bdb1584cc30ff9b92a7a70b94c168966a19e4785876ea7cc1e8c",
    "viewer": "1234020869f3945114cf645ccbc728c1281e49e79622e36823700d0bc1e6eab9",
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


def _walk_artifacts(value: Any, output: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"} <= set(value):
            output.append(value)
        for item in value.values():
            _walk_artifacts(item, output)
    elif isinstance(value, list):
        for item in value:
            _walk_artifacts(item, output)


def _finite(model: Gaussians3D) -> bool:
    return all(
        bool(torch.isfinite(getattr(model, name)).all())
        for name in ("means", "quats", "log_scales", "opacity", "sh")
    )


def _equal(left: Gaussians3D, right: Gaussians3D) -> bool:
    return all(
        torch.equal(getattr(left, name), getattr(right, name))
        for name in ("means", "quats", "log_scales", "opacity", "sh")
    )


def _checkpoint_gate(
    summary: dict[str, Any],
    treatment: str,
    reference: str,
    step: int,
) -> dict[str, Any]:
    treatment_metrics = summary["arms"][treatment]["checkpoints"][str(step)]["metrics"]
    reference_metrics = summary["arms"][reference]["checkpoints"][str(step)]["metrics"]
    result = {
        "treatment": treatment,
        "reference": reference,
        "step": step,
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


def _trajectory_average(
    summary: dict[str, Any],
    arm: str,
    split: str,
) -> dict[str, Any]:
    steps = (0, *STEPS)
    record = summary["arms"][arm]
    values = [float(record["init_metrics"][split]["psnr_fg"])]
    values.extend(
        float(record["checkpoints"][str(step)]["metrics"][split]["psnr_fg"]) for step in STEPS
    )
    average = (
        sum(
            (left_value + right_value) / 2 * (right_step - left_step)
            for left_step, right_step, left_value, right_value in zip(
                steps[:-1],
                steps[1:],
                values[:-1],
                values[1:],
                strict=True,
            )
        )
        / 10_000
    )
    return {"steps": list(steps), "values": values, "average_db": average}


class _PageLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        for key in ("href", "src"):
            if values.get(key):
                self.links.append(str(values[key]))


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite audit: {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("exact metric replay requires the frozen CUDA/gsplat path")

    summary = _read(SUMMARY)
    plan = _read(RUN / "plan.json")
    parent = _read(PARENT_SUMMARY)
    parent_plan = _read(PARENT_RUN / "plan.json")
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    summary_hash = _sha256(SUMMARY)
    check(
        "official result is complete and hash-bound",
        summary_hash == EXPECTED["summary"]
        and summary.get("schema") == "rtgs.pool_structure_wse_10k_frame00008.result.v1"
        and summary.get("status") == "complete"
        and tuple(summary.get("checkpoint_steps", ())) == STEPS
        and tuple(summary.get("arms", ())) == ARMS,
        {
            "summary_sha256": summary_hash,
            "status": summary.get("status"),
            "checkpoint_steps": summary.get("checkpoint_steps"),
        },
    )

    protocol_before_plan = PROTOCOL.stat().st_mtime_ns <= (RUN / "plan.json").stat().st_mtime_ns
    check(
        "protocol predates execution and is bound by plan and summary",
        _sha256(PROTOCOL) == EXPECTED["protocol"]
        and protocol_before_plan
        and plan["protocol"]["sha256"] == EXPECTED["protocol"]
        and summary["protocol"]["sha256"] == EXPECTED["protocol"],
        {
            "protocol_sha256": _sha256(PROTOCOL),
            "protocol_mtime_ns": PROTOCOL.stat().st_mtime_ns,
            "plan_mtime_ns": (RUN / "plan.json").stat().st_mtime_ns,
        },
    )

    git_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    source_hashes = summary["repository"]["source_files"]
    current_hashes = {relative: _sha256(ROOT / relative) for relative in source_hashes}
    check(
        "executed dirty-tree source remains exactly bound",
        source_hashes == current_hashes
        and source_hashes["benchmarks/pool_structure_wse_10k_frame00008.py"] == EXPECTED["harness"]
        and source_hashes["benchmarks/new_variants_frame00008.py"] == EXPECTED["shared_harness"]
        and summary["repository"]["git_revision"] == git_revision,
        {
            "source_count": len(source_hashes),
            "harness_sha256": source_hashes["benchmarks/pool_structure_wse_10k_frame00008.py"],
            "git_revision": git_revision,
        },
    )

    check(
        "audited parent summary and parent model records remain exact",
        _sha256(PARENT_SUMMARY) == EXPECTED["parent_summary"]
        and summary["parent"]["summary"]["sha256"] == EXPECTED["parent_summary"]
        and summary["parent"]["initial_npz"]
        == {arm: parent["reconstruction"][arm]["artifacts"]["initial_npz"] for arm in ARMS},
        {
            "parent_summary_sha256": _sha256(PARENT_SUMMARY),
            "parent_status": parent.get("status"),
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
        "all summary-bound artifacts retain exact bytes and hashes",
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
    tensors_match = scene.masks is not None
    if scene.masks is not None:
        for name, image, mask in zip(scene.view_names, scene.images, scene.masks, strict=True):
            record = summary["scene"]["inputs"][name]
            tensors_match &= (
                _tensor_hash(image) == record["loaded_rgb_tensor_sha256"]
                and _tensor_hash(mask) == record["loaded_mask_tensor_sha256"]
            )
    check(
        "calibrated tensors and seven-train/one-reporting-only split replay",
        tuple(scene.view_names) == VIEW_NAMES
        and scene.training_views == list(range(7))
        and scene.testing_views == [7]
        and tensors_match,
        {
            "view_names": scene.view_names,
            "training_views": scene.training_views,
            "testing_views": scene.testing_views,
            "loaded_tensor_hashes_match": tensors_match,
        },
    )

    current_config = dict(summary["training"])
    expected_config = dict(parent_plan["training_final"])
    expected_config["iterations"] = 10_000
    expected_config["schedule_iterations"] = 10_000
    check(
        "10k schedule changes only iterations and global schedule length",
        current_config == expected_config
        and current_config["iterations"] == 10_000
        and current_config["schedule_iterations"] == 10_000
        and current_config["eval_every"] == 100
        and current_config["checkpoint_policy"] == "final"
        and current_config["density"]["stop_iter"] == 1_000,
        {
            "iterations": current_config["iterations"],
            "schedule_iterations": current_config["schedule_iterations"],
            "eval_every": current_config["eval_every"],
            "density_stop_iter": current_config["density"]["stop_iter"],
        },
    )

    renderer = get_rasterizer(
        "gsplat",
        device=torch.device("cuda"),
        packed=False,
        antialiased=True,
    )
    model_valid = True
    parent_initials_exact = True
    metric_replay_max_error = 0.0
    model_detail: dict[str, Any] = {}
    for arm in ARMS:
        record = summary["arms"][arm]
        parent_record = parent["reconstruction"][arm]["artifacts"]
        parent_initial = Gaussians3D.load_npz(_path(parent_record["initial_npz"]["path"]))
        copied_initial_npz = RUN / "models" / arm / "gaussians_init.npz"
        copied_initial_ply = RUN / "models" / arm / "gaussians_init.ply"
        initial = Gaussians3D.load_npz(copied_initial_npz)
        initial_ply = Gaussians3D.load_ply(copied_initial_ply)
        parent_initials_exact &= (
            _sha256(copied_initial_npz) == parent_record["initial_npz"]["sha256"]
            and _sha256(copied_initial_ply) == parent_record["initial_ply"]["sha256"]
            and _equal(parent_initial, initial)
            and initial.n == initial_ply.n == record["init_n_gaussians"]
        )

        states: list[tuple[str, Gaussians3D, Gaussians3D, dict[str, Any]]] = []
        for step in STEPS:
            checkpoint = record["checkpoints"][str(step)]
            npz = Gaussians3D.load_npz(_path(checkpoint["artifacts"]["npz"]["path"]))
            ply = Gaussians3D.load_ply(_path(checkpoint["artifacts"]["ply"]["path"]))
            states.append((str(step), npz, ply, checkpoint["metrics"]))
        final = Gaussians3D.load_npz(_path(record["final_artifacts"]["npz"]["path"]))
        final_ply = Gaussians3D.load_ply(_path(record["final_artifacts"]["ply"]["path"]))
        model_valid &= (
            _finite(initial)
            and _finite(initial_ply)
            and _finite(final)
            and _finite(final_ply)
            and final.n == final_ply.n == record["final_n_gaussians"]
            and _equal(final, states[-1][1])
            and _sha256(_path(record["final_artifacts"]["npz"]["path"]))
            == record["checkpoints"]["10000"]["artifacts"]["npz"]["sha256"]
            and _sha256(_path(record["final_artifacts"]["ply"]["path"]))
            == record["checkpoints"]["10000"]["artifacts"]["ply"]["sha256"]
        )
        for step, npz, ply, _ in states:
            model_valid &= (
                _finite(npz)
                and _finite(ply)
                and npz.n == ply.n == record["checkpoints"][step]["n_gaussians"]
            )

        replay_states = [("init", initial, record["init_metrics"])]
        replay_states.extend((step, npz, metrics) for step, npz, _, metrics in states)
        for _, model, reported in replay_states:
            replayed = {
                "train": Trainer.evaluate_metrics(
                    scene,
                    model.to("cuda"),
                    renderer,
                    indices=scene.training_views,
                ),
                "test": Trainer.evaluate_metrics(
                    scene,
                    model.to("cuda"),
                    renderer,
                    indices=scene.testing_views,
                ),
            }
            for split in ("train", "test"):
                for key, value in replayed[split].items():
                    metric_replay_max_error = max(
                        metric_replay_max_error,
                        abs(float(value) - float(reported[split][key])),
                    )
        model_detail[arm] = {
            "initial_n": initial.n,
            "checkpoint_n": {step: npz.n for step, npz, _, _ in states},
            "final_equals_10k": _equal(final, states[-1][1]),
        }
    check(
        "copied initial states exactly equal audited parent states",
        parent_initials_exact,
        {
            arm: {
                "npz_sha256": _sha256(RUN / "models" / arm / "gaussians_init.npz"),
                "ply_sha256": _sha256(RUN / "models" / arm / "gaussians_init.ply"),
            }
            for arm in ARMS
        },
    )
    check(
        "all 36 checkpoint NPZ/PLY loads are finite and final equals 10k",
        model_valid,
        model_detail,
    )
    check(
        "initial and all 15 checkpoint metric records replay through exact gsplat",
        metric_replay_max_error <= 5e-5,
        {"metric_replay_max_abs_error": metric_replay_max_error},
    )

    histories: dict[str, dict[str, Any]] = {}
    history_valid = True
    sampled_sequences: list[list[int]] = []
    for arm in ARMS:
        record = summary["arms"][arm]
        history = _read(_path(record["training_history"]["path"]))
        sampled_sequences.append(history["sampled_train_views"])
        counts = [(0, record["init_n_gaussians"])]
        counts.extend((int(step), int(count)) for step, count in history["n_gaussians"])
        changes = [
            step
            for (_, previous_count), (step, count) in zip(counts, counts[1:], strict=False)
            if previous_count != count
        ]
        last_change = max(changes, default=0)
        density_iterations = [int(item["iteration"]) for item in history["density_stats"]]
        history_valid &= (
            len(history["loss"]) == 10_000
            and len(history["loss_terms"]) == 10_000
            and len(history["sampled_train_views"]) == 10_000
            and set(history["sampled_train_views"]) <= set(range(7))
            and len(history["psnr"]) == 100
            and [int(step) for step, _ in history["psnr"]] == list(range(100, 10_001, 100))
            and len(history["n_gaussians"]) == 100
            and counts[-1] == (10_000, record["final_n_gaussians"])
            and last_change <= 1_000
            and 10_000 - last_change >= 9_000
            and max(density_iterations, default=0) <= 900
            and history["schedule_iterations"] == 10_000
            and history["segment_iterations"] == 10_000
            and history["iteration_offset"] == 0
            and "selected_step" not in history
        )
        histories[arm] = {
            "density_count_change_steps": changes,
            "density_controller_iterations": density_iterations,
            "last_count_change_step": last_change,
            "recovery_steps": 10_000 - last_change,
            "sampled_view_ids": sorted(set(history["sampled_train_views"])),
            "means_lr_initial": history["means_lr_initial"],
            "means_lr_final": history["means_lr_final"],
        }
    history_valid &= sampled_sequences[0] == sampled_sequences[1] == sampled_sequences[2]
    check(
        "histories are complete, paired, train-only, final-policy, and recover 9k steps",
        history_valid,
        histories,
    )

    checkpoint_gates = {
        name: {str(step): _checkpoint_gate(summary, treatment, reference, step) for step in STEPS}
        for name, (treatment, reference) in CONTRASTS.items()
    }
    expected_gate_pattern = (
        not any(item["passed"] for item in checkpoint_gates["density_vs_gradient"].values())
        and all(item["passed"] for item in checkpoint_gates["wse_vs_gradient"].values())
        and all(item["passed"] for item in checkpoint_gates["wse_vs_density"].values())
    )
    sustained = {
        name: gates["8000"]["passed"] and gates["10000"]["passed"]
        for name, gates in checkpoint_gates.items()
    }
    check(
        "frozen checkpoint gates recompute with both WSE contrasts sustained",
        expected_gate_pattern
        and sustained
        == {
            "density_vs_gradient": False,
            "wse_vs_gradient": True,
            "wse_vs_density": True,
        },
        {"checkpoint_gates": checkpoint_gates, "sustained_long_run_positive": sustained},
    )

    trajectories = {
        arm: {split: _trajectory_average(summary, arm, split) for split in ("train", "test")}
        for arm in ARMS
    }
    trajectory_finite = all(
        math.isfinite(record["average_db"])
        for arm in trajectories.values()
        for record in arm.values()
    )
    heldout_peaks = {}
    for arm in ARMS:
        checkpoints = summary["arms"][arm]["checkpoints"]
        best_step = max(
            STEPS,
            key=lambda step: checkpoints[str(step)]["metrics"]["test"]["psnr_fg"],
        )
        heldout_peaks[arm] = {
            "best_reporting_step": best_step,
            "best_psnr_fg": checkpoints[str(best_step)]["metrics"]["test"]["psnr_fg"],
            "10k_psnr_fg": checkpoints["10000"]["metrics"]["test"]["psnr_fg"],
            "10k_minus_best_db": (
                checkpoints["10000"]["metrics"]["test"]["psnr_fg"]
                - checkpoints[str(best_step)]["metrics"]["test"]["psnr_fg"]
            ),
        }
    check(
        "preregistered trajectory averages are finite and all held-out peaks occur at 2k",
        trajectory_finite
        and all(item["best_reporting_step"] == 2_000 for item in heldout_peaks.values()),
        {"trajectories": trajectories, "heldout_checkpoint_peaks": heldout_peaks},
    )

    viewer = _read(VIEWER)
    viewer_receipt = _read(VIEWER_RECEIPT)
    viewer_valid = (
        _sha256(VIEWER) == EXPECTED["viewer"]
        and viewer_receipt["manifest"]["sha256"] == EXPECTED["viewer"]
        and viewer["schema"] == "rtgs.viewer-comparison.v1"
        and len(viewer["methods"]) == len(ARMS) * len(STEPS)
        and viewer_receipt["manifest"]["method_count"] == 15
        and viewer_receipt["manifest"]["model_count"] == 30
        and viewer_receipt["server"]["http_status"] == 200
        and viewer_receipt["server"]["response_bytes"] > 0
        and viewer_receipt["server"]["cuda_visible_devices"] == ""
        and viewer_receipt["server"]["nvidia_compute_processes_at_check"] == []
        and viewer_receipt["server"]["pid_owned_listening_socket"] is True
        and viewer_receipt["server"]["pid_stopped_after_stop"] is True
        and viewer_receipt["server"]["port_closed_after_stop"] is True
    )
    for manifest_record, receipt_record in zip(
        viewer["methods"], viewer_receipt["models"], strict=True
    ):
        initial = (VIEWER.parent / manifest_record["initial"]).resolve()
        final = (VIEWER.parent / manifest_record["final"]).resolve()
        viewer_valid &= (
            manifest_record["name"] == receipt_record["name"]
            and _sha256(initial) == receipt_record["initial_sha256"]
            and _sha256(final) == receipt_record["final_sha256"]
        )
    check(
        "CPU viewer loads all 30 manifest entries and shuts down cleanly",
        viewer_valid,
        {
            "manifest_sha256": _sha256(VIEWER),
            "method_count": len(viewer["methods"]),
            "http_status": viewer_receipt["server"]["http_status"],
            "response_bytes": viewer_receipt["server"]["response_bytes"],
        },
    )

    raster_visuals = [
        _path(summary["comparison_visuals"][key]["path"])
        for key in ("stage1_parent", "train_checkpoints", "heldout_checkpoints")
    ]
    for arm in ARMS:
        record = summary["arms"][arm]
        raster_visuals.extend(
            _path(record["checkpoints"][str(step)]["visuals"][split]["path"])
            for step in STEPS
            for split in ("train", "heldout")
        )
        raster_visuals.extend(
            _path(record["progress_visuals"][key]["path"]) for key in ("train_gif", "heldout_gif")
        )
    visuals_valid = len(raster_visuals) == 39 and len(set(raster_visuals)) == 39
    visual_detail = {}
    for path in raster_visuals:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            visuals_valid &= image.width > 0 and image.height > 0
            visual_detail[str(path.relative_to(ROOT))] = {
                "size": [image.width, image.height],
                "frames": getattr(image, "n_frames", 1),
            }
    svg_path = _path(summary["comparison_visuals"]["quality_trajectory"]["path"])
    svg_root = ET.parse(svg_path).getroot()
    polylines = [item for item in svg_root.iter() if item.tag.endswith("polyline")]
    visuals_valid &= svg_root.tag.endswith("svg") and len(polylines) == 6
    check(
        "all 39 raster visuals and six SVG trajectories decode",
        visuals_valid,
        {
            "raster_visual_count": len(raster_visuals),
            "svg_polyline_count": len(polylines),
            "raster_visuals": visual_detail,
        },
    )

    index_path = _path(summary["index_html"]["path"])
    page = index_path.read_text(encoding="utf-8")
    parser = _PageLinks()
    parser.feed(page)
    local_targets = []
    missing_targets = []
    deferred_targets = {RESULT_NOTE.resolve(), OUTPUT.resolve()}
    deferred_seen = set()
    for link in parser.links:
        clean, _ = urldefrag(link)
        parsed = urlsplit(clean)
        if not clean or parsed.scheme or parsed.netloc:
            continue
        target = (index_path.parent / parsed.path).resolve()
        local_targets.append(target)
        if target in deferred_targets:
            deferred_seen.add(target)
        elif not target.is_file():
            missing_targets.append(str(target))
    html_valid = (
        page.startswith("<!doctype html>")
        and page.endswith("</html>\n")
        and len(parser.links) == 60
        and page.count('class="checkpoint"') == 15
        and page.count("<tr>") == 19
        and not missing_targets
        and deferred_seen == deferred_targets
    )
    check(
        "required index is summary-bound with complete local links except audit self-reference",
        html_valid,
        {
            "index_sha256": _sha256(index_path),
            "link_count": len(parser.links),
            "unique_local_target_count": len(set(local_targets)),
            "missing_non_deferred_targets": missing_targets,
            "deferred_until_notes_written": [
                str(path.relative_to(ROOT)) for path in sorted(deferred_seen)
            ],
        },
    )

    parent_endpoint_deltas = {}
    for arm in ARMS:
        old = parent["reconstruction"][arm]["final_metrics"]
        new = summary["arms"][arm]["checkpoints"]["2000"]["metrics"]
        parent_endpoint_deltas[arm] = {
            "heldout_psnr_fg_delta_db": new["test"]["psnr_fg"] - old["test"]["psnr_fg"],
            "heldout_alpha_iou_delta": new["test"]["alpha_iou"] - old["test"]["alpha_iou"],
            "train_psnr_fg_delta_db": new["train"]["psnr_fg"] - old["train"]["psnr_fg"],
        }

    final_gates = {name: gates["10000"] for name, gates in checkpoint_gates.items()}
    findings = [
        (
            "WSE versus gradient passes every frozen checkpoint gate; at 10k the deltas are "
            f"{final_gates['wse_vs_gradient']['heldout_psnr_fg_delta_db']:+.4f} dB held-out "
            "foreground PSNR, "
            f"{final_gates['wse_vs_gradient']['heldout_alpha_iou_delta']:+.5f} alpha IoU, and "
            f"{final_gates['wse_vs_gradient']['train_psnr_fg_delta_db']:+.4f} dB train PSNR."
        ),
        (
            "WSE versus density also passes every checkpoint gate; its 10k held-out foreground "
            f"PSNR edge is {final_gates['wse_vs_density']['heldout_psnr_fg_delta_db']:+.4f} dB."
        ),
        (
            "Density without WSE fails against gradient at every checkpoint; its 10k held-out "
            f"foreground PSNR delta is "
            f"{final_gates['density_vs_gradient']['heldout_psnr_fg_delta_db']:+.4f} dB."
        ),
        (
            "All three held-out trajectories peak at the first 2k reporting snapshot and decline "
            "thereafter; no held-out-selected early-stop claim is authorized."
        ),
        (
            "The new 2k states are deliberately not parent replays because the means-LR schedule "
            "now spans 10k; parent endpoint differences therefore do not estimate nondeterminism."
        ),
    ]
    claim_disposition = {
        "wse_beats_density_under_pool_over_10k": (
            "confirm_single_scene_single_seed_sustained_development_observation"
        ),
        "wse_beats_pool_gradient_over_10k": (
            "confirm_single_scene_single_seed_sustained_downstream_observation"
        ),
        "density_without_wse_beats_pool_gradient": "retire_fails_all_checkpoint_gates",
        "training_beyond_2k_improves_heldout_quality": (
            "retire_all_three_reporting_trajectories_peak_at_2k"
        ),
        "best_checkpoint_is_2k": (
            "not_authorized_heldout_camera_is_reporting_only_and_no_validation_selection_ran"
        ),
        "stage1_or_combined_end_to_end_winner": (
            "not_reopened_parent_stage1_gates_still_bound_and_failed_for_structure_arms"
        ),
        "default_or_generalization_claim": "not_authorized_single_scene_seed_and_heldout_camera",
        "performance_or_memory_claim": "not_authorized_contended_unrepeated_execution",
    }

    audit_source = Path(__file__).resolve()
    audit = {
        "schema": "rtgs.pool_structure_wse_10k_frame00008.audit.v1",
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "audit_source": {
            "path": str(audit_source.relative_to(ROOT)),
            "sha256": _sha256(audit_source),
        },
        "summary": {
            "path": str(SUMMARY.relative_to(ROOT)),
            "sha256": summary_hash,
        },
        "scope": (
            "single-scene, single-seed 10k development trajectories with seven training cameras "
            "and one reporting-only held-out camera"
        ),
        "checkpoint_gates": checkpoint_gates,
        "sustained_long_run_positive": sustained,
        "trajectory_averages": trajectories,
        "heldout_checkpoint_peaks": heldout_peaks,
        "parent_2k_endpoint_deltas_not_repeatability_evidence": parent_endpoint_deltas,
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
                ".venv-cuda/bin/python benchmarks/pool_structure_wse_10k_frame00008.py "
                "--protocol benchmarks/results/"
                "20260724_pool_structure_wse_10k_frame00008_PREREG.md "
                "--out runs/pool_structure_wse_10k_frame00008_20260724"
            ),
            "audit": (
                "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 "
                ".venv-cuda/bin/python "
                "benchmarks/audit_pool_structure_wse_10k_frame00008.py"
            ),
            "viewer": viewer_receipt["command"],
        },
    }
    OUTPUT.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(audit["check_summary"], indent=2))
    print(
        json.dumps(
            {
                "sustained_long_run_positive": sustained,
                "final_gates": final_gates,
                "heldout_checkpoint_peaks": heldout_peaks,
            },
            indent=2,
        )
    )
    return 1 if audit["check_summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
