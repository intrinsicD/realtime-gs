#!/usr/bin/env python3
"""Independent scientist pass for the Janelle geometric-arena experiment."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
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
RUN = ROOT / "runs/geometric_arena_frame00008_20260724"
SUMMARY = RUN / "summary.json"
PLAN = RUN / "plan.json"
PROTOCOL = ROOT / "benchmarks/results/20260724_geometric_arena_frame00008_PREREG.md"
RESULT_NOTE = ROOT / "benchmarks/results/20260724_geometric_arena_frame00008_RESULT.md"
VIEWER = ROOT / "benchmarks/results/20260724_geometric_arena_frame00008_VIEWER.json"
VIEWER_RECEIPT = ROOT / "benchmarks/results/20260724_geometric_arena_frame00008_VIEWER_RECEIPT.json"
OUTPUT = ROOT / "benchmarks/results/20260724_geometric_arena_frame00008_AUDIT.json"
OUTPUT_NOTE = ROOT / "benchmarks/results/20260724_geometric_arena_frame00008_AUDIT.md"
RAW_SCENE = Path("/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008")
CALIBRATION = ROOT / "dataset/2025_03_07_stage_with_fabric/calibration_dome.json"

ARMS = ("dynamic-a", "geometric", "dynamic-b")
STEPS = (2_000, 4_000, 6_000, 8_000, 10_000)
EVENT_STEPS = (200, 300, 400, 500, 600, 700, 800, 900)
FIELDS = ("means", "quats", "log_scales", "opacity", "sh")
VIEW_NAMES = ("C0001", "C0008", "C0014", "C0021", "C0026", "C0031", "C0039", "C1004")


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


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


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


def _all_finite(model: Gaussians3D) -> bool:
    return all(bool(torch.isfinite(getattr(model, name)).all()) for name in FIELDS)


def _equal(left: Gaussians3D, right: Gaussians3D) -> bool:
    return all(torch.equal(getattr(left, name), getattr(right, name)) for name in FIELDS)


def _field_hashes(model: Gaussians3D) -> dict[str, str]:
    return {name: _tensor_hash(getattr(model, name)) for name in FIELDS}


def _close_metrics(
    recorded: dict[str, dict[str, float]],
    replayed: dict[str, dict[str, float]],
) -> tuple[bool, float]:
    differences = [
        abs(float(recorded[split][name]) - float(replayed[split][name]))
        for split in ("train", "test")
        for name in recorded[split]
    ]
    maximum = max(differences, default=0.0)
    return maximum <= 1e-6, maximum


def _performance_reduction(arms: dict[str, dict[str, Any]]) -> dict[str, float]:
    histories = {arm: arms[arm]["performance"] for arm in ARMS}
    controls = ("dynamic-a", "dynamic-b")
    dynamic_event_fastest = min(histories[arm]["density_event_total_seconds"] for arm in controls)
    dynamic_total_fastest = min(histories[arm]["native_elapsed_seconds"] for arm in controls)
    dynamic_non_density_worst = max(
        histories[arm]["approx_non_density_seconds"] for arm in controls
    )
    dynamic_allocated_worst = max(histories[arm]["peak_allocated_gib"] for arm in controls)
    dynamic_reserved_worst = max(histories[arm]["peak_reserved_gib"] for arm in controls)
    geometric = histories["geometric"]
    return {
        "density_event_ratio_to_faster_dynamic": (
            geometric["density_event_total_seconds"] / dynamic_event_fastest
        ),
        "native_elapsed_ratio_to_faster_dynamic": (
            geometric["native_elapsed_seconds"] / dynamic_total_fastest
        ),
        "non_density_ratio_to_worse_dynamic": (
            geometric["approx_non_density_seconds"] / dynamic_non_density_worst
        ),
        "peak_allocated_ratio_to_worse_dynamic": (
            geometric["peak_allocated_gib"] / dynamic_allocated_worst
        ),
        "peak_reserved_ratio_to_worse_dynamic": (
            geometric["peak_reserved_gib"] / dynamic_reserved_worst
        ),
    }


def _audit_disposition(summary: dict[str, Any]) -> dict[str, Any]:
    arms = summary["arms"]
    trajectories = {arm: arms[arm]["performance"]["density_trajectory"] for arm in ARMS}
    sampled_hashes = {arm: arms[arm]["performance"]["sampled_train_views_sha256"] for arm in ARMS}
    dynamic_counts = [int(arms[arm]["final_n_gaussians"]) for arm in ("dynamic-a", "dynamic-b")]
    validity = {
        "sampled_views_match": len(set(sampled_hashes.values())) == 1,
        "event_trajectories_match": len({_canonical_hash(trajectories[arm]) for arm in ARMS}) == 1,
        "dynamic_final_counts_match": len(set(dynamic_counts)) == 1,
    }
    dynamic_test_psnr = [
        float(arms[arm]["checkpoints"]["10000"]["metrics"]["test"]["psnr_fg"])
        for arm in ("dynamic-a", "dynamic-b")
    ]
    if not validity["sampled_views_match"]:
        disposition = "INVALID_RNG_STREAM_MISMATCH"
    elif not validity["dynamic_final_counts_match"] or not validity["event_trajectories_match"]:
        disposition = "INVALID_DYNAMIC_CONTROL_NONREPEATABILITY"
    else:
        performance = _performance_reduction(arms)
        disposition = (
            "RETAIN_OPT_IN_AUTHORIZE_SCALING_STUDY"
            if performance["density_event_ratio_to_faster_dynamic"] <= 0.80
            else "NEGATIVE_SYSTEMS_RESULT_KEEP_DYNAMIC_DEFAULT"
        )
    return {
        "validity": validity,
        "dynamic_final_counts": dynamic_counts,
        "dynamic_heldout_psnr_fg": dynamic_test_psnr,
        "dynamic_heldout_psnr_delta_db": abs(dynamic_test_psnr[0] - dynamic_test_psnr[1]),
        "disposition": disposition,
        "default_change_authorized": False,
    }


class _PageLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        for key in ("href", "src"):
            if values.get(key):
                self.links.append(str(values[key]))


def _note(audit: dict[str, Any]) -> str:
    summary = audit["raw_measurements"]
    performance = audit["independent_reduction"]["performance"]
    disposition = audit["independent_reduction"]["protocol_disposition"]
    density_ratio = performance["density_event_ratio_to_faster_dynamic"]
    elapsed_ratio = performance["native_elapsed_ratio_to_faster_dynamic"]
    lines = [
        "# Audit: geometric Stage-3 arena on Janelle `frame_00008`",
        "",
        "## Referee verdict",
        "",
        f"**`{disposition['disposition']}`.** The experiment completed and its artifacts replay, "
        "but the two supposedly identical dynamic controls did not reproduce one another. The "
        "frozen protocol therefore invalidates the arena comparison before a correctness or "
        "performance verdict.",
        "",
        "The producer summary labels this `REJECT_CURRENT_ARENA_CORRECTNESS`; that is a reduction "
        "precedence error. Dynamic-A and dynamic-B already disagree, so their variation cannot "
        "be attributed to arena storage. The producer JSON remains untouched; this audit is the "
        "authoritative disposition.",
        "",
        "## Claim disposition",
        "",
        "| Claim | Evidence | Disposition |",
        "|---|---|---|",
    ]
    for claim in audit["claim_table"]:
        lines.append(f"| {claim['claim']} | {claim['evidence']} | **{claim['disposition']}** |")
    lines.extend(
        [
            "",
            "## Raw measurements",
            "",
            "| arm | final N | native 10k s | density events ms | peak alloc MiB | "
            "held-out FG PSNR |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        record = summary[arm]
        lines.append(
            f"| {arm} | {record['final_n']} | {record['native_seconds']:.6f} | "
            f"{record['density_event_seconds'] * 1000:.3f} | "
            f"{record['peak_allocated_mib']:.3f} | "
            f"{record['heldout_psnr_fg']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"The arena/dynamic-A density-event ratio is **{density_ratio:.3f}×** "
            "against the faster dynamic control, and its native elapsed ratio is "
            f"**{elapsed_ratio:.3f}×**. These are "
            "descriptive only because dynamic control quality/counts drifted and the run has "
            "one arena observation.",
            "",
            "## First divergence",
            "",
            "All arms matched through the 300-step density event (422→815→1,378). At step 400, "
            "dynamic-A reached 2,098 rows, the arena 2,094, and dynamic-B 2,096. This symmetric "
            "control drift is consistent with end-to-end CUDA nondeterminism; it is not evidence "
            "that either storage policy is correct or incorrect. The mechanism-level transaction "
            "tests remain useful but cannot rescue the consumed end-to-end protocol.",
            "",
            "## Required next evidence",
            "",
            "Before another timing claim, freeze a real pre-event state and its accumulated "
            "selection tensors, then apply dynamic and arena topology transactions to that same "
            "payload for exact parity. For performance, use repeated fresh-process blocks on an "
            "idle named GPU, a warmup rule that excludes one-time kernel initialization, tolerant "
            "count/quality equivalence gates justified before access, and multiple scenes. Keep "
            "`dynamic` as the default.",
            "",
            "## Audit checks",
            "",
            f"{audit['check_summary']['passed']}/{audit['check_summary']['total']} audit checks "
            "passed. An audit check passing means the referee detected and disposed of the "
            "invalid result correctly; it does not mean the arena experiment passed.",
            "",
            f"- Machine-readable audit: `{OUTPUT.relative_to(ROOT)}`",
            f"- Results page: `{RUN.relative_to(ROOT)}/index.html`",
            f"- Viewer receipt: `{VIEWER_RECEIPT.relative_to(ROOT)}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    if OUTPUT.exists() or OUTPUT_NOTE.exists():
        raise FileExistsError("refusing to overwrite geometric-arena audit")
    if not torch.cuda.is_available():
        raise RuntimeError("metric replay requires CUDA/gsplat")

    summary = _read(SUMMARY)
    plan = _read(PLAN)
    viewer = _read(VIEWER)
    viewer_receipt = _read(VIEWER_RECEIPT)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check(
        "official result shape and completion state are exact",
        summary.get("schema") == "rtgs.geometric_arena_frame00008.result.v1"
        and summary.get("status") == "complete"
        and tuple(summary.get("arms", ())) == ARMS
        and tuple(summary.get("checkpoint_steps", ())) == STEPS,
        {
            "summary_sha256": _sha256(SUMMARY),
            "schema": summary.get("schema"),
            "status": summary.get("status"),
            "arms": list(summary.get("arms", ())),
            "steps": summary.get("checkpoint_steps"),
        },
    )

    plan_time = dt.datetime.fromisoformat(plan["created_utc"])
    worker_times = {
        arm: dt.datetime.fromisoformat(
            _read(_path(summary["arms"][arm]["worker"]["path"]))["completed_utc"]
        )
        for arm in ARMS
    }
    summary_time = dt.datetime.fromisoformat(summary["completed_utc"])
    chronology_valid = (
        _sha256(PROTOCOL) == plan["protocol"]["sha256"]
        and plan["protocol"] == summary["protocol"]
        and all(plan_time < worker_times[arm] < summary_time for arm in ARMS)
        and worker_times["dynamic-a"] < worker_times["geometric"] < worker_times["dynamic-b"]
    )
    check(
        "protocol predates ordered workers and remains hash-bound",
        chronology_valid,
        {
            "protocol_sha256": _sha256(PROTOCOL),
            "plan_created_utc": plan["created_utc"],
            "worker_completed_utc": {arm: worker_times[arm].isoformat() for arm in ARMS},
            "summary_completed_utc": summary["completed_utc"],
        },
    )

    source_hashes = {
        relative: _sha256(ROOT / relative) for relative in plan["repository"]["source_files"]
    }
    source_valid = (
        source_hashes == plan["repository"]["source_files"]
        and _canonical_hash(source_hashes) == plan["repository"]["source_aggregate_sha256"]
        and summary["repository"] == plan["repository"]
    )
    check(
        "dirty-tree executed source closure remains byte-exact",
        source_valid,
        {
            "git_revision": plan["repository"]["git_revision"],
            "source_count": len(source_hashes),
            "source_aggregate_sha256": _canonical_hash(source_hashes),
        },
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
    input_valid = (
        tuple(scene.view_names) == VIEW_NAMES
        and scene.training_views == list(range(7))
        and scene.testing_views == [7]
        and scene.masks is not None
        and _sha256(CALIBRATION) == plan["scene"]["calibration"]["sha256"]
        and _sha256(_path(plan["initialization"]["path"])) == plan["initialization"]["sha256"]
    )
    input_details = {}
    assert scene.masks is not None
    for name, image, mask in zip(scene.view_names, scene.images, scene.masks, strict=True):
        record = plan["scene"]["inputs"][name]
        rgb = _path(record["rgb"]["path"])
        mask_path = _path(record["mask"]["path"])
        matched = (
            _sha256(rgb) == record["rgb"]["sha256"]
            and _sha256(mask_path) == record["mask"]["sha256"]
            and _tensor_hash(image) == record["loaded_rgb_tensor_sha256"]
            and _tensor_hash(mask) == record["loaded_mask_tensor_sha256"]
        )
        input_valid &= matched
        input_details[name] = {"matched": matched}
    check(
        "calibrated inputs, tensors, masks, initialization, and held-out split replay",
        input_valid,
        {
            "view_names": scene.view_names,
            "train_indices": scene.training_views,
            "test_indices": scene.testing_views,
            "inputs": input_details,
        },
    )

    configs = plan["training_by_arm"]
    ignored = {
        "gaussian_storage_policy",
        "arena_growth_factor",
        "arena_initial_capacity",
    }
    normalized_configs = {
        arm: {key: value for key, value in configs[arm].items() if key not in ignored}
        for arm in ARMS
    }
    config_valid = (
        normalized_configs["dynamic-a"]
        == normalized_configs["geometric"]
        == normalized_configs["dynamic-b"]
        and configs["dynamic-a"]["gaussian_storage_policy"] == "dynamic"
        and configs["geometric"]["gaussian_storage_policy"] == "geometric"
        and configs["dynamic-b"]["gaussian_storage_policy"] == "dynamic"
        and all(configs[arm]["iterations"] == 10_000 for arm in ARMS)
        and all(configs[arm]["profile_density_events"] is True for arm in ARMS)
        and all(configs[arm]["checkpoint_policy"] == "final" for arm in ARMS)
    )
    check(
        "arms differ only in frozen storage configuration",
        config_valid,
        {
            "config_sha256": {arm: _canonical_hash(configs[arm]) for arm in ARMS},
            "storage_policy": {arm: configs[arm]["gaussian_storage_policy"] for arm in ARMS},
        },
    )

    artifact_records: list[dict[str, Any]] = []
    _walk_artifacts(summary, artifact_records)
    workers = {}
    for arm in ARMS:
        worker = _read(_path(summary["arms"][arm]["worker"]["path"]))
        workers[arm] = worker
        _walk_artifacts(worker, artifact_records)
    unique_artifacts = {record["path"]: record for record in artifact_records}
    artifact_failures = []
    for record in unique_artifacts.values():
        path = _path(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or _sha256(path) != record["sha256"]
        ):
            artifact_failures.append(record["path"])
    check(
        "all summary and worker artifacts remain hash/size exact",
        not artifact_failures,
        {
            "artifact_reference_count": len(artifact_records),
            "unique_artifact_count": len(unique_artifacts),
            "failures": artifact_failures,
        },
    )

    model_failures = []
    models: dict[str, dict[int, Gaussians3D]] = {}
    for arm in ARMS:
        models[arm] = {}
        for step in STEPS:
            record = summary["arms"][arm]["checkpoints"][str(step)]
            model = Gaussians3D.load_npz(_path(record["artifacts"]["npz"]["path"]))
            models[arm][step] = model
            valid = (
                model.n == int(record["n_gaussians"])
                and _field_hashes(model) == record["field_sha256"]
                and _all_finite(model)
                and bool((model.opacity >= 0).all())
                and bool((model.opacity <= 1).all())
            )
            if not valid:
                model_failures.append(f"{arm}@{step}")
        final = Gaussians3D.load_npz(_path(summary["arms"][arm]["final_artifacts"]["npz"]["path"]))
        if (
            not _equal(final, models[arm][10_000])
            or _field_hashes(final) != summary["arms"][arm]["final_field_sha256"]
            or final.n != int(summary["arms"][arm]["final_n_gaussians"])
        ):
            model_failures.append(f"{arm}@final")
    check(
        "all 15 saved states and three finals replay exact, finite, and bounded",
        not model_failures,
        {"failures": model_failures},
    )

    renderer = get_rasterizer(
        "gsplat",
        device=torch.device("cuda"),
        packed=False,
        antialiased=True,
    )
    metric_failures = []
    metric_max_abs = 0.0
    with torch.no_grad():
        for arm in ARMS:
            for step in STEPS:
                recorded = summary["arms"][arm]["checkpoints"][str(step)]["metrics"]
                model = models[arm][step].to("cuda")
                replayed = {
                    "train": Trainer.evaluate_metrics(
                        scene,
                        model,
                        renderer,
                        indices=scene.training_views,
                    ),
                    "test": Trainer.evaluate_metrics(
                        scene,
                        model,
                        renderer,
                        indices=scene.testing_views,
                    ),
                }
                matched, maximum = _close_metrics(recorded, replayed)
                metric_max_abs = max(metric_max_abs, maximum)
                if not matched:
                    metric_failures.append({"arm": arm, "step": step, "max_abs": maximum})
    check(
        "all 15 train/held-out metric records independently replay",
        not metric_failures,
        {
            "tolerance": 1e-6,
            "maximum_absolute_difference": metric_max_abs,
            "failures": metric_failures,
        },
    )

    histories = {
        arm: _read(_path(summary["arms"][arm]["training_history"]["path"])) for arm in ARMS
    }
    sampled_hashes = {arm: _canonical_hash(histories[arm]["sampled_train_views"]) for arm in ARMS}
    history_valid = (
        len(set(sampled_hashes.values())) == 1
        and all(
            tuple(item["iteration"] for item in histories[arm]["density_stats"]) == EVENT_STEPS
            for arm in ARMS
        )
        and all(
            all(item["event_seconds"] > 0 for item in histories[arm]["density_stats"])
            for arm in ARMS
        )
        and all(histories[arm]["cuda_memory_stats"].get("num_ooms") == 0 for arm in ARMS)
    )
    check(
        "training histories preserve RNG sequence and complete synchronized event accounting",
        history_valid,
        {
            "sampled_train_views_sha256": sampled_hashes,
            "event_steps": {
                arm: [item["iteration"] for item in histories[arm]["density_stats"]] for arm in ARMS
            },
            "event_milliseconds": {
                arm: [item["event_seconds"] * 1_000 for item in histories[arm]["density_stats"]]
                for arm in ARMS
            },
        },
    )

    arena_storage = histories["geometric"]["storage_diagnostics"]
    migrations = arena_storage["migrations"]
    arena_accounting_valid = (
        arena_storage["active_n"] == summary["arms"]["geometric"]["final_n_gaussians"]
        and arena_storage["capacity"] == 8_192
        and arena_storage["migration_count"] == len(migrations) == 4
        and [item["new_capacity"] for item in migrations] == [1_024, 2_048, 4_096, 8_192]
        and all(item["required_n"] <= item["new_capacity"] for item in migrations)
        and arena_storage["physical_parameter_adam_bytes"]
        == arena_storage["adam_parameter_row_bytes"] * arena_storage["capacity"]
    )
    check(
        "arena capacity and migration accounting are internally consistent",
        arena_accounting_valid,
        arena_storage,
    )

    trajectories = {arm: summary["arms"][arm]["performance"]["density_trajectory"] for arm in ARMS}
    first_divergence = next(
        (
            index
            for index in range(len(EVENT_STEPS))
            if len({trajectories[arm][index]["n_after"] for arm in ARMS}) > 1
        ),
        None,
    )
    dynamic_nonrepeatability = (
        summary["arms"]["dynamic-a"]["final_n_gaussians"]
        != summary["arms"]["dynamic-b"]["final_n_gaussians"]
        and trajectories["dynamic-a"] != trajectories["dynamic-b"]
        and first_divergence == 2
    )
    check(
        "audit detects preregistered dynamic-control nonrepeatability",
        dynamic_nonrepeatability,
        {
            "first_divergence_index": first_divergence,
            "first_divergence_step": (
                EVENT_STEPS[first_divergence] if first_divergence is not None else None
            ),
            "trajectories": trajectories,
        },
    )

    performance = _performance_reduction(summary["arms"])
    producer_performance = summary["decision"]["performance"]
    performance_matches = all(
        math.isclose(
            performance[key],
            float(producer_performance[key]),
            rel_tol=0,
            abs_tol=1e-12,
        )
        for key in performance
    )
    check(
        "raw timing/memory ratios recompute, while remaining descriptive",
        performance_matches,
        {
            "independent": performance,
            "producer": producer_performance,
            "timing_decisional": False,
            "reason": "dynamic controls differ by count, trajectory, and >0.02 dB held-out",
        },
    )

    audit_disposition = _audit_disposition(summary)
    producer_disposition = summary["decision"]["disposition"]
    disposition_valid = (
        audit_disposition["disposition"] == "INVALID_DYNAMIC_CONTROL_NONREPEATABILITY"
        and producer_disposition == "REJECT_CURRENT_ARENA_CORRECTNESS"
        and summary["decision"]["validity"]["dynamic_final_counts_match"] is False
        and summary["decision"]["default_change_authorized"] is False
    )
    check(
        "audit applies protocol validity before arena correctness and overrides producer label",
        disposition_valid,
        {
            "audit_disposition": audit_disposition["disposition"],
            "producer_disposition": producer_disposition,
            "producer_validity": summary["decision"]["validity"],
        },
    )

    viewer_valid = (
        viewer.get("schema") == "rtgs.viewer-comparison.v1"
        and len(viewer.get("methods", ())) == 15
        and viewer_receipt.get("passed") is True
        and viewer_receipt["manifest"]["sha256"] == _sha256(VIEWER)
        and viewer_receipt["manifest"]["method_count"] == 15
        and viewer_receipt["manifest"]["model_count"] == 30
        and viewer_receipt["server"]["http_status"] == 200
        and viewer_receipt["server"]["response_bytes"] > 0
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
        "CPU viewer loads all 30 manifest models and shuts down cleanly",
        viewer_valid,
        {
            "manifest_sha256": _sha256(VIEWER),
            "method_count": len(viewer["methods"]),
            "http_status": viewer_receipt["server"]["http_status"],
            "response_bytes": viewer_receipt["server"]["response_bytes"],
        },
    )

    raster_paths = [
        _path(summary["comparison_visuals"][key]["path"])
        for key in ("train_checkpoints", "heldout_checkpoints")
    ]
    for arm in ARMS:
        record = summary["arms"][arm]
        raster_paths.extend(
            _path(record["checkpoints"][str(step)]["visuals"][split]["path"])
            for step in STEPS
            for split in ("train", "heldout")
        )
        raster_paths.extend(
            _path(record["progress_visuals"][key]["path"]) for key in ("train_gif", "heldout_gif")
        )
    visual_failures = []
    visual_detail = {}
    for path in raster_paths:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                valid = image.width > 0 and image.height > 0
                visual_detail[str(path.relative_to(ROOT))] = {
                    "size": [image.width, image.height],
                    "frames": getattr(image, "n_frames", 1),
                }
            if not valid:
                visual_failures.append(str(path))
        except Exception as error:  # noqa: BLE001 - artifact referee records decoder failures
            visual_failures.append(f"{path}: {error}")
    check(
        "all 38 diagnostic raster/GIF artifacts decode",
        len(raster_paths) == 38 and len(set(raster_paths)) == 38 and not visual_failures,
        {
            "visual_count": len(raster_paths),
            "failures": visual_failures,
            "visuals": visual_detail,
        },
    )

    index_path = _path(summary["index_html"]["path"])
    page = index_path.read_text(encoding="utf-8")
    parser = _PageLinks()
    parser.feed(page)
    missing_targets = []
    local_targets = []
    deferred_seen = False
    for link in parser.links:
        clean, _fragment = urldefrag(link)
        parsed = urlsplit(clean)
        if not clean or parsed.scheme or parsed.netloc:
            continue
        target = (index_path.parent / parsed.path).resolve()
        local_targets.append(target)
        if target == OUTPUT_NOTE.resolve():
            deferred_seen = True
        elif not target.is_file():
            missing_targets.append(str(target))
    index_valid = (
        _sha256(index_path) == summary["index_html"]["sha256"]
        and page.startswith("<!doctype html>")
        and page.endswith("</html>")
        and deferred_seen
        and not missing_targets
    )
    check(
        "required index is result-bound with only the audit self-reference deferred",
        index_valid,
        {
            "index_sha256": _sha256(index_path),
            "reference_count": len(parser.links),
            "unique_local_target_count": len(set(local_targets)),
            "deferred_audit_note_seen": deferred_seen,
            "missing_non_deferred_targets": missing_targets,
        },
    )

    raw_measurements = {}
    for arm in ARMS:
        arm_record = summary["arms"][arm]
        perf = arm_record["performance"]
        raw_measurements[arm] = {
            "final_n": arm_record["final_n_gaussians"],
            "native_seconds": perf["native_elapsed_seconds"],
            "density_event_seconds": perf["density_event_total_seconds"],
            "peak_allocated_mib": perf["peak_allocated_gib"] * 1_024,
            "peak_reserved_mib": perf["peak_reserved_gib"] * 1_024,
            "heldout_psnr_fg": arm_record["checkpoints"]["10000"]["metrics"]["test"]["psnr_fg"],
            "heldout_alpha_iou": arm_record["checkpoints"]["10000"]["metrics"]["test"]["alpha_iou"],
        }

    allocated_ratio = performance["peak_allocated_ratio_to_worse_dynamic"]
    reserved_ratio = performance["peak_reserved_ratio_to_worse_dynamic"]
    claim_table = [
        {
            "claim": "The geometric arena preserves the dynamic end-to-end trajectory.",
            "kind_scope": "asserted correctness; one CUDA scene/seed",
            "evidence": (
                "Dynamic controls themselves end at 5,424 and 5,337; the arena ends at 5,395."
            ),
            "disposition": "NARROW — not testable under this invalid control bracket",
        },
        {
            "claim": "The arena materially reduces density-event latency.",
            "kind_scope": "measured systems claim; one arena run",
            "evidence": (
                f"{performance['density_event_ratio_to_faster_dynamic']:.3f}× the faster "
                "dynamic event total; frozen ≤0.80 gate not met."
            ),
            "disposition": "RETIRE for this run",
        },
        {
            "claim": "The arena improves end-to-end 10k time.",
            "kind_scope": "measured systems claim; one arena run",
            "evidence": (
                f"{performance['native_elapsed_ratio_to_faster_dynamic']:.3f}× the faster "
                "dynamic total; frozen ≤0.98 gate not met."
            ),
            "disposition": "RETIRE for this run",
        },
        {
            "claim": "Arena memory is non-inferior at this scale.",
            "kind_scope": "descriptive GPU memory; one scene",
            "evidence": (
                f"Peak allocated ratio {allocated_ratio:.3f}×; "
                f"reserved ratio {reserved_ratio:.3f}×."
            ),
            "disposition": "NARROW to this observed run",
        },
        {
            "claim": "The default may change from dynamic allocation.",
            "kind_scope": "production/default claim",
            "evidence": "Protocol forbids a default change; validity and speed gates fail.",
            "disposition": "RETIRE — keep dynamic default",
        },
    ]

    audit_source = Path(__file__).resolve()
    audit = {
        "schema": "rtgs.geometric_arena_frame00008.audit.v1",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "audit_source": {
            "path": str(audit_source.relative_to(ROOT)),
            "sha256": _sha256(audit_source),
            "bytes": audit_source.stat().st_size,
        },
        "official_summary": {
            "path": str(SUMMARY.relative_to(ROOT)),
            "sha256": _sha256(SUMMARY),
            "bytes": SUMMARY.stat().st_size,
        },
        "scope": (
            "single-scene, single-seed storage experiment with two fresh-process dynamic "
            "controls and one geometric arena observation"
        ),
        "claim_table": claim_table,
        "independent_reduction": {
            "protocol_disposition": audit_disposition,
            "producer_disposition": producer_disposition,
            "performance": performance,
            "timing_decisional": False,
            "default_change_authorized": False,
        },
        "raw_measurements": raw_measurements,
        "first_divergence": {
            "step": EVENT_STEPS[first_divergence] if first_divergence is not None else None,
            "n_after": {arm: trajectories[arm][first_divergence]["n_after"] for arm in ARMS}
            if first_divergence is not None
            else None,
        },
        "checks": checks,
        "check_summary": {
            "total": len(checks),
            "passed": sum(item["passed"] for item in checks),
            "failed": sum(not item["passed"] for item in checks),
        },
        "commands": {
            "official": (
                "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 "
                ".venv-cuda/bin/python benchmarks/geometric_arena_frame00008.py"
            ),
            "viewer": viewer_receipt["command"],
            "audit": (
                "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 "
                ".venv-cuda/bin/python benchmarks/audit_geometric_arena_frame00008.py"
            ),
        },
        "verification_context": {
            "focused_cpu": (
                ".venv/bin/pytest -q tests/test_gaussian_arena.py tests/test_optim.py "
                "-m 'not cuda' (passed)"
            ),
            "focused_cuda": (
                ".venv-cuda/bin/pytest -q tests/test_optim.py::"
                "test_geometric_arena_gsplat_default_trains_and_grows (passed)"
            ),
            "repository_verify": (
                "./scripts/verify.sh: lint/format passed; six unrelated historical tests "
                "failed because they freeze removed libstdc++.so.6.0.33; all remaining "
                "non-slow tests passed when those six were deselected"
            ),
            "git_diff_check": "passed",
        },
    }
    OUTPUT.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    OUTPUT_NOTE.write_text(_note(audit), encoding="utf-8")
    print(json.dumps(audit["check_summary"], indent=2))
    print(json.dumps(audit["independent_reduction"], indent=2))
    return 1 if audit["check_summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
