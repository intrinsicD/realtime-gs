#!/usr/bin/env python3
"""RTGS-016 report-only RGB evaluation and shared-v2 machine-source producer.

No optimizer lives in this process. The coordinator invokes phase evaluation only
when every frozen cell in that phase has a saved endpoint and passed input guard.
The shared contract renderer owns HTML; a distinct reviewer owns AUDIT records.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import platform
import resource
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "20260906_tomography_source_constraints_haelyn_dome"


def driver():
    path = Path(__file__).with_name(TASK_ID + ".py")
    spec = importlib.util.spec_from_file_location("_rtgs016_report_driver", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def cells(task, run, dataset_id):
    d = driver()
    output = []
    for arm in d.ARMS:
        for seed in task["seeds"]:
            path = run / "cells" / dataset_id / f"seed_{seed}" / arm
            summary = json.loads((path / "summary.json").read_text())
            if (
                summary["status"] != "completed"
                or not summary["input_boundary"]["passed"]
                or summary["dataset_id"] != dataset_id
                or summary["arm_id"] != arm
                or summary["seed"] != seed
            ):
                raise ValueError("incomplete or mismatched cell before report-only access")
            for name, digest in summary["models"].items():
                if d.sha(path / name) != digest:
                    raise ValueError(f"saved model changed: {path / name}")
            output.append((path, summary))
    return output


def external_phase(task, run, dataset_id):
    started = time.perf_counter()
    d = driver()
    measured = cells(task, run, dataset_id)  # gate precedes image-capable imports
    d.verify_external_seal(task)
    import numpy as np
    import torch
    from PIL import Image

    from rtgs.core.gaussians3d import Gaussians3D
    from rtgs.core.metrics import ssim
    from rtgs.data.calibrated import load_calibrated_scene
    from rtgs.data.compact_views import CompactDataset
    from rtgs.optim.compact_trainer import CompactTrainer

    torch.set_num_threads(1)
    dataset = next(x for x in task["datasets"] if x["id"] == dataset_id)
    heldout = task["splits"][dataset_id]["heldout"]
    production = json.loads((ROOT / dataset["production_manifest"]).read_text())
    scene = load_calibrated_scene(
        ROOT / dataset["frame_path"],
        calibration_path=ROOT / dataset["calibration"],
        downscale=production["downscale"],
        view_ids=heldout,
        load_masks=True,
    )
    compact = CompactDataset.load(
        (ROOT / dataset["compact_manifest"]).parent,
        view_ids=heldout,
        load_alpha=False,
        byte_cap=task["frozen_configuration"]["converter"]["view_byte_cap"],
    )
    if len(scene.images) != len(heldout) or scene.masks is None:
        raise ValueError("report-only heldout RGB/mask set is incomplete")
    rows = []
    for path, summary in measured:
        cfg = d.configs(task, summary["arm_id"], summary["seed"], dataset_id=dataset_id)[1]
        renderer = CompactTrainer(cfg)._renderer()
        for endpoint, filename in (
            ("initial", "gaussians_init.ply"),
            ("proxy", "gaussians_proxy.ply"),
            ("final", "gaussians.ply"),
        ):
            model = Gaussians3D.load_ply(path / filename)
            for index, view in enumerate(compact.views):
                camera = view.camera
                target = scene.images[index].float()
                mask = scene.masks[index] > 0.5
                if target.shape[:2] != (camera.height, camera.width):
                    raise ValueError("report-only camera and reference resolution differ")
                if not all(
                    torch.allclose(
                        getattr(camera, key),
                        getattr(scene.cameras[index], key),
                        atol=1e-5,
                        rtol=1e-6,
                    )
                    for key in ("K", "R", "t")
                ):
                    raise ValueError("report-only reference calibration differs")
                yy, xx = torch.meshgrid(
                    torch.arange(camera.height), torch.arange(camera.width), indexing="ij"
                )
                xy = torch.stack((xx.flatten(), yy.flatten()), -1).float() + 0.5
                chunks = []
                with torch.no_grad():
                    for start in range(0, len(xy), cfg.evaluation_chunk):
                        chunks.append(
                            renderer.render_points(
                                model,
                                camera,
                                xy[start : start + cfg.evaluation_chunk],
                                background=torch.zeros(3),
                                sh_degree=cfg.sh_degree,
                            ).color
                        )
                    prediction = torch.cat(chunks).reshape_as(target)
                    if not torch.isfinite(prediction).all() or not mask.any():
                        raise ValueError("nonfinite prediction or empty report-only foreground")
                    error = (prediction - target).square()
                    mse = float(error.mean())
                    foreground_mse = float(error[mask].mean())
                    # A perfect match is represented explicitly, not by an arbitrary PSNR cap.
                    row = {
                        "dataset_id": dataset_id,
                        "arm_id": summary["arm_id"],
                        "seed": summary["seed"],
                        "endpoint": endpoint,
                        "view_id": view.view_id,
                        "full_canvas_mse": mse,
                        "foreground_mse": foreground_mse,
                        "full_canvas_psnr": -10 * math.log10(mse) if mse > 0 else None,
                        "foreground_psnr": -10 * math.log10(foreground_mse)
                        if foreground_mse > 0
                        else None,
                        "ssim": float(ssim(prediction, target)),
                        "perfect_match": mse == 0,
                    }
                rows.append(row)
                if summary["seed"] == task["seeds"][0] and index == 0:
                    panels = [target, prediction, error.sqrt().clamp(0, 1)]
                    pixels = (
                        (torch.cat(panels, 1).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
                    )
                    preview = path / f"preview_{endpoint}.png"
                    if preview.exists():
                        raise FileExistsError(preview)
                    Image.fromarray(pixels).save(preview)
    d.verify_external_seal(task)
    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "dataset_id": dataset_id,
        "status": "pending_independent_audit",
        "rows": rows,
        "wall_seconds": time.perf_counter() - started,
        "peak_host_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "scope": (
            "Report-only CPU evaluation including imports, reference loading and "
            "previews; excludes process startup."
        ),
        "renderer": (
            "Same TorchPointRasterizer/configuration as native compact refinement; "
            "half-integer pixel centers, unclamped predictions for metrics."
        ),
        "mask_use": (
            "Foreground reporting only, including the maskless phase; no optimizer "
            "sees these masks."
        ),
    }
    d.write_new(run / "phases" / dataset_id / "external_evaluation.json", payload)
    print(f"phase report saved: {dataset_id}; pending independent audit", flush=True)


def aggregate(task, run):
    run = run.resolve()
    d = driver()
    measured = [
        item
        for dataset in task["frozen_configuration"]["phase_order"]
        for item in cells(task, run, dataset)
    ]
    summaries = [s for _, s in measured]
    representative = next(
        p
        for p, s in measured
        if s["dataset_id"] == "haelyn_masked"
        and s["arm_id"] == "soft"
        and s["seed"] == task["seeds"][0]
    )
    # The representative is fixed before outcomes, never a winning-seed selection.
    for name in ("gaussians_init.ply", "gaussians.ply"):
        if (run / name).exists():
            raise FileExistsError(run / name)
        shutil.copyfile(representative / name, run / name)
    meta = {
        "label": "Validation native teacher pixel MSE",
        "unit": "MSE",
        "group": "Quality",
        "direction": "lower",
    }
    history = {
        "schema_version": 2,
        "records": [r for s in summaries for r in s["records"]],
        "metric_metadata": {"native_teacher_mse": meta},
        "stage_markers": [r for s in summaries for r in s["stage_markers"]],
    }
    contract = d.module(ROOT / "scripts/experiment_contract.py", "_rtgs016_report_contract")
    if errors := contract._history_errors(history, task, completed=True):
        raise ValueError("; ".join(errors))
    d.write_new(run / "training_history.json", history)
    d.write_new(
        run / "gaussians.config.json",
        {
            "frozen": task["frozen_configuration"],
            "effective_cells": [
                {
                    "dataset_id": s["dataset_id"],
                    "arm_id": s["arm_id"],
                    "seed": s["seed"],
                    "effective": s["effective"],
                }
                for s in summaries
            ],
            "representative": str(representative.relative_to(run)),
            "training": {"packed": False, "antialiased": False},
        },
    )
    d.write_new(
        run / "input_boundary_receipt.json",
        {
            "schema_version": 1,
            "all_guards_passed": all(s["input_boundary"]["passed"] for s in summaries),
            "data_seal_sha256": d.sha(ROOT / task["data_seal"]),
            "external_evaluation_is_separate": True,
            "cells": [
                {
                    "dataset_id": s["dataset_id"],
                    "arm_id": s["arm_id"],
                    "seed": s["seed"],
                    "guard": s["input_boundary"],
                    "selected_input_binding": s["selected_input_binding"],
                }
                for s in summaries
            ],
        },
    )
    d.write_new(
        run / "resource_receipt.json",
        {
            "schema_version": 1,
            "protocol": task["resource_protocol"],
            "cells": [json.loads((p / "process_receipt.json").read_text()) for p, s in measured],
            "offline_preparation": [
                json.loads((ROOT / x["production_manifest"]).read_text()) for x in task["datasets"]
            ],
            "scope_note": (
                "Do not sum process peaks or label offline loop timers as process "
                "startup-inclusive time."
            ),
        },
    )
    import torch

    d.write_new(
        run / "environment.json",
        {
            "schema_version": 1,
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {"torch": torch.__version__},
            "device": {
                "type": "cpu",
                "name": platform.processor() or platform.machine(),
                "cuda": None,
            },
        },
    )
    lock = json.loads((run / "task.lock.json").read_text())
    groups = []
    paired = []
    for dataset in task["frozen_configuration"]["phase_order"]:
        for arm in d.ARMS:
            selected = [s for s in summaries if s["dataset_id"] == dataset and s["arm_id"] == arm]
            groups.append(
                {
                    "dataset_id": dataset,
                    "arm_id": arm,
                    "native_teacher_mse": statistics.median(
                        s["heldout_metrics"]["final"]["J_pixel"] for s in selected
                    ),
                    "proxy_teacher_mse": statistics.median(
                        s["heldout_metrics"]["proxy"]["J_pixel"] for s in selected
                    ),
                    "wall_seconds": statistics.median(
                        s["worker_endpoint_seconds"] for s in selected
                    ),
                    "observer_excluded_wall_seconds": statistics.median(
                        s["endpoint_excluding_validation_seconds"] for s in selected
                    ),
                    "peak_host_bytes": statistics.median(s["peak_host_bytes"] for s in selected),
                    "source_mean_drift": None
                    if arm == "beam_reference"
                    else statistics.median(s["source_mean_drift"] for s in selected),
                }
            )
        by_key = {(s["arm_id"], s["seed"]): s for s in summaries if s["dataset_id"] == dataset}
        for arm in ("soft", "free"):
            for seed in task["seeds"]:
                base, treatment = by_key["hard", seed], by_key[arm, seed]
                for endpoint in ("proxy", "final"):
                    paired.append(
                        {
                            "dataset_id": dataset,
                            "arm_id": arm,
                            "seed": seed,
                            "endpoint": endpoint,
                            "teacher_mse_difference_vs_hard": treatment["heldout_metrics"][
                                endpoint
                            ]["J_pixel"]
                            - base["heldout_metrics"][endpoint]["J_pixel"],
                            "wall_ratio_vs_hard": treatment["worker_endpoint_seconds"]
                            / base["worker_endpoint_seconds"],
                            "observer_excluded_wall_ratio_vs_hard": treatment[
                                "endpoint_excluding_validation_seconds"
                            ]
                            / base["endpoint_excluding_validation_seconds"],
                        }
                    )
    result = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "pending_independent_audit",
        "claim_boundary": task["claim_boundary"],
        "groups": groups,
        "paired": paired,
        "cells": summaries,
        "representative_policy": "first frozen seed, masked Haelyn, soft arm",
    }
    d.write_new(run / "cell_results.json", result)
    reference = next(
        g for g in groups if g["dataset_id"] == "haelyn_masked" and g["arm_id"] == "soft"
    )
    metrics = {
        key: reference[key]
        for key in ("native_teacher_mse", "wall_seconds", "peak_host_bytes", "source_mean_drift")
    }
    metadata = {
        m["id"]: {
            "label": "Masked Haelyn soft, seed median: " + m["label"],
            "unit": m["unit"],
            "direction": m["direction"],
            "group": "Reference condition",
        }
        for m in task["primary_metrics"]
    }
    charts = [
        {
            "id": chart,
            "title": title,
            "unit": unit,
            "values": [
                {"label": g["dataset_id"] + " / " + g["arm_id"], "value": g[key]} for g in groups
            ],
        }
        for chart, title, unit, key in (
            ("quality", "Per-condition final teacher MSE", "MSE", "native_teacher_mse"),
            ("resources", "Per-condition process peak memory", "bytes", "peak_host_bytes"),
            (
                "stage_runtime",
                "Per-condition reconstruction endpoint time",
                "seconds",
                "wall_seconds",
            ),
        )
    ]
    stage_values = []
    for group in groups:
        selected = [
            s
            for s in summaries
            if s["dataset_id"] == group["dataset_id"] and s["arm_id"] == group["arm_id"]
        ]
        for stage in task["stages"]:
            durations = []
            for summary in selected:
                bounds = {
                    m["boundary"]: m["wall_seconds"]
                    for m in summary["stage_markers"]
                    if m["stage"] == stage["id"]
                }
                durations.append(bounds["end"] - bounds["start"])
            stage_values.append(
                {
                    "label": group["dataset_id"] + " / " + group["arm_id"] + " / " + stage["label"],
                    "value": statistics.median(durations),
                }
            )
    charts[-1].update(
        title="Per-condition stage time including validation observers", values=stage_values
    )
    artifacts = [
        {"label": name.replace("_", " "), "path": name} for name in contract.REQUIRED_V2_ARTIFACTS
    ]
    artifacts.append({"label": "All cells and paired differences", "path": "cell_results.json"})
    artifacts.extend(
        {"label": str(p.relative_to(run)), "path": str(p.relative_to(run))}
        for p in sorted(run.glob("cells/**/preview_*.png"))
    )
    artifacts.extend(
        {"label": "Report-only " + dataset, "path": f"phases/{dataset}/external_evaluation.json"}
        for dataset in task["frozen_configuration"]["phase_order"]
    )
    payload = {
        "schema_version": 2,
        "report_template_version": 2,
        "task_id": TASK_ID,
        "summary": (
            "All frozen cells completed; per-capture paired results await independent audit."
        ),
        "decision": "pending_independent_audit",
        "claim_boundary": task["claim_boundary"],
        "metrics": metrics,
        "metric_metadata": metadata,
        "charts": charts,
        "artifacts": artifacts,
        "evidence": [
            {"label": suffix, "path": f"benchmarks/results/{TASK_ID}_{suffix}"}
            for suffix in contract.EVIDENCE_SUFFIXES
        ],
        "commands": {
            "reproduce": task["run_command"],
            "serve_report": [
                ".venv/bin/python",
                "-m",
                "http.server",
                "8765",
                "--directory",
                str(run.relative_to(ROOT)),
            ],
            "viewer": [
                ".venv/bin/rtgs",
                "view",
                "--gaussians",
                str((run / "gaussians.ply").relative_to(ROOT)),
                "--initial",
                str((run / "gaussians_init.ply").relative_to(ROOT)),
                "--no-open",
            ],
        },
        "notes": [
            (
                "Top-level metrics are the prospectively fixed masked-Haelyn soft "
                "condition; never pooled across captures."
            ),
            (
                "Each condition retains every seed; all causal paired differences are "
                "in cell_results.json."
            ),
            (
                "Maskless reconstruction consumes no masks. Foreground masks are used "
                "only by the separate report evaluator."
            ),
            (
                "Compact native risk is against finite-support 2D teachers; external "
                "RGB risks separately expose the conversion gap."
            ),
            (
                "No default change or confirmatory significance claim follows from "
                "this development screen."
            ),
        ],
    }
    if errors := contract._metric_errors_v2(payload, task, lock, completed=True):
        raise ValueError("; ".join(errors))
    d.write_new(run / "metrics.json", payload)
    d.write_new(ROOT / "benchmarks/results" / f"{TASK_ID}_RESULT.json", result)
    note = ROOT / "benchmarks/results" / f"{TASK_ID}_RESULT.md"
    with note.open("x") as stream:
        stream.write(
            (
                "# Captured-reference tomography comparison\n\nProducer execution "
                "completed; independent audit pending.\n\n"
            )
            + task["claim_boundary"]
            + (
                "\n\nEvery seed, paired difference and endpoint is retained in the "
                "adjacent RESULT.json. No arm is promoted by this note.\n"
            )
        )

    d.write_new(
        run / "run_receipt.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "completed",
            "started_at_utc": lock["started_at_utc"],
            "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exit_code": 0,
            "failure_phase": None,
            "message": (
                "Producer execution completed; results audit and browser validation pending."
            ),
        },
    )


def failure(task, run, error, phase):
    """Publish renderable v2 failure sources, preserving any earlier partial bytes."""
    run = run.resolve()
    d = driver()
    contract = d.module(ROOT / "scripts/experiment_contract.py", "_rtgs016_failure_contract")
    lock = json.loads((run / "task.lock.json").read_text())
    completed = [json.loads(p.read_text()) for p in sorted(run.glob("cells/**/summary.json"))]
    partial = [json.loads(p.read_text()) for p in sorted(run.glob("cells/**/failure.json"))]
    records = [r for s in completed + partial for r in s.get("records", [])]
    markers = [
        m for s in completed + partial if s.get("records") for m in s.get("stage_markers", [])
    ]
    history = {
        "schema_version": 2,
        "records": records,
        "stage_markers": markers,
        "metric_metadata": {
            "native_teacher_mse": {
                "label": "Validation native teacher MSE",
                "unit": "MSE",
                "group": "Quality",
                "direction": "lower",
            }
        }
        if records
        else {},
    }
    history_errors = contract._history_errors(history, task, completed=False)
    if history_errors:
        # Raw cell files retain invalid/nonfinite progress for diagnosis. Such values
        # cannot be drawn as valid fitting observations by the shared v2 renderer.
        history = {"schema_version": 2, "records": [], "stage_markers": [], "metric_metadata": {}}
    receipt = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "failed",
        "started_at_utc": lock["started_at_utc"],
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "exit_code": 1,
        "failure_phase": phase,
        "message": f"{type(error).__name__}: {error}",
    }
    environment = {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {"realtime-gs": lock.get("source_commit", "unknown")},
        "device": {"type": "cpu", "name": platform.processor() or platform.machine(), "cuda": None},
    }
    payloads = {
        "training_history.json": history,
        "gaussians.config.json": {"frozen": task["frozen_configuration"], "run_failed": True},
        "input_boundary_receipt.json": {
            "completed_cell_count": len(completed),
            "guards": [s["input_boundary"] for s in completed],
            "failure_phase": phase,
        },
        "resource_receipt.json": {
            "scope": "Failed run; raw process/cell receipts retained",
            "history_errors": history_errors,
            "process_receipts": [
                str(p.relative_to(run)) for p in sorted(run.glob("**/process_receipt.json"))
            ],
        },
        "environment.json": environment,
        "run_receipt.json": receipt,
    }
    metrics = {
        "schema_version": 2,
        "report_template_version": 2,
        "task_id": TASK_ID,
        "summary": f"Run aborted during {phase}; no comparison result is established.",
        "decision": "failed_no_result",
        "claim_boundary": task["claim_boundary"],
        "metrics": {},
        "metric_metadata": {},
        "charts": [],
        "evidence": [],
        "artifacts": [
            {"label": name.replace("_", " "), "path": name}
            for name in contract.REQUIRED_V2_FAILURE_ARTIFACTS
        ],
        "commands": {
            "reproduce": task["run_command"],
            "serve_report": [
                ".venv/bin/python",
                "-m",
                "http.server",
                "8765",
                "--directory",
                str(run.relative_to(ROOT)),
            ],
            "viewer": None,
        },
        "notes": [
            (
                "The first failure aborts the remaining cells. No retry or replacement "
                "is authorized by this task."
            ),
            "Partial artifacts are retained for diagnosis; this bundle is not results-bearing.",
        ],
    }
    if errors := contract._metric_errors_v2(metrics, task, lock, completed=False):
        raise ValueError("invalid failure source: " + "; ".join(errors)) from error
    if errors := contract._run_receipt_errors(receipt, task, lock):
        raise ValueError("invalid failure receipt: " + "; ".join(errors)) from error
    payloads["metrics.json"] = metrics
    for name, value in payloads.items():
        path = run / name
        if path.exists():
            # Preserve every previous partial source exactly; never overwrite it.
            saved = run / "failure_preserved" / name
            if saved.exists():
                raise FileExistsError(saved) from error
            saved.parent.mkdir(exist_ok=True)
            path.rename(saved)
        d.write_new(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    task = driver().validate_binding(args.task, args.run)
    if args.phase not in task["frozen_configuration"]["phase_order"]:
        raise ValueError("phase is not frozen")
    external_phase(task, args.run.resolve(), args.phase)


if __name__ == "__main__":
    main()
