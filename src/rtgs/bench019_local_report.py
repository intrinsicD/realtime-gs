"""Publish immutable local BENCH019 receipts into shared RTGS report sources.

This module reads already-completed cells. It never fits, renders, selects checkpoints,
opens raw images, or supplies an independent audit verdict. Stage-1 acquisition remains
one shared observation per frame/family; only measured downstream seeds are aggregated.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import platform
import statistics
import sys
from pathlib import Path

from rtgs import bench019 as B

QUALITY = ("heldout_foreground_psnr", "heldout_alpha_iou", "heldout_exterior_leakage")
HISTORY_META = {
    "refine_loss": {
        "label": "Training objective",
        "unit": "loss",
        "group": "Objective",
        "direction": "lower",
    },
    "refine_rows": {
        "label": "3D Gaussian rows",
        "unit": "gaussians",
        "group": "Capacity",
        "direction": "descriptive",
    },
}


def _read(path: Path) -> dict:
    return B.load_json_object(path)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _verified(base: Path, descriptor: dict) -> Path:
    path = (base / descriptor["path"]).resolve(strict=True)
    if path.stat().st_size != descriptor["bytes"] or B.sha256_file(path) != descriptor["sha256"]:
        raise ValueError(f"saved artifact changed: {path}")
    return path


def _cell_dir(run: Path, frame: str, family: str, seed: int, replicate="primary") -> Path:
    category = "warmup" if replicate == "warmup" else "downstream"
    return run / category / frame / family / f"seed_{seed}_{replicate}"


def _collect_cell(
    task: dict, run: Path, frame: str, family: str, seed: int, replicate="primary"
) -> dict:
    directory = _cell_dir(run, frame, family, seed, replicate)
    receipt = _read(directory / "receipt.json")
    evaluation = _read(directory / "evaluation" / "evaluation.json")
    history_path = _verified(directory, receipt["artifacts"]["history"])
    config_path = _verified(directory, receipt["artifacts"]["config"])
    final = _verified(directory, receipt["artifacts"]["final"]["npz"])
    for document in (receipt, evaluation):
        for key, expected in (
            ("task_id", task["task_id"]),
            ("frame_id", frame),
            ("family_id", family),
            ("seed", seed),
            ("status", "ok"),
        ):
            if document[key] != expected:
                raise ValueError(f"cell {key} differs from the matrix")
    if evaluation["model_sha256"] != B.sha256_file(final):
        raise ValueError("held-out evaluation does not bind the final saved model")
    expected_views = task["splits"][frame]["heldout"]
    if (
        evaluation["view_ids"] != expected_views
        or [row["view_id"] for row in evaluation["per_view"]] != expected_views
    ):
        raise ValueError("held-out evaluation view inventory differs from task")
    for name in QUALITY:
        key = name.removeprefix("heldout_")
        mean = statistics.mean(_number(row[key], key) for row in evaluation["per_view"])
        if not math.isclose(
            _number(evaluation["metrics"][name], name), mean, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("evaluation aggregate differs from per-view arithmetic mean")
    history = _read(history_path)
    if history["heldout_evaluation"] is not False:
        raise ValueError("training history must not contain held-out evaluation")
    if not set(history["sampled_training_view_ids"]) <= set(task["splits"][frame]["train"]):
        raise ValueError("training schedule contains excluded view ids")
    if receipt["metrics"]["final_gaussians"] != 256:
        raise ValueError("downstream count changed")
    metrics = {**receipt["metrics"], **evaluation["metrics"]}
    for item in task["primary_metrics"]:
        _number(metrics[item["id"]], item["id"])
    return {
        "frame_id": frame,
        "family_id": family,
        "seed": seed,
        "replicate": replicate,
        "directory": directory,
        "receipt": receipt,
        "evaluation": evaluation,
        "history": history,
        "config": _read(config_path),
        "metrics": metrics,
        "resource": _read(directory / "resource_receipt.json"),
    }


def training_history(task: dict, cells: list[dict]) -> dict:
    """Use actual native optimizer-clock checkpoints; do not interpolate timestamps."""
    if [stage["id"] for stage in task["stages"]] != ["refine"]:
        raise ValueError(
            "shared history must freeze only refine; acquisition histories are separate"
        )
    records, markers = [], []
    for cell in cells:
        native = cell["history"]["native_trainer"]
        if native["executed_iterations"] != 1000 or native["stop_reason"] != "max_iterations":
            raise ValueError("refinement did not finish its fixed horizon")
        elapsed = dict(native["elapsed"])
        counts = dict(native["n_gaussians"])
        if (
            not elapsed
            or 1000 not in elapsed
            or len(native["loss"]) != 1000
            or list(elapsed) != sorted(elapsed)
            or list(elapsed.values()) != sorted(elapsed.values())
        ):
            raise ValueError("native history lacks ordered actual checkpoint times")
        identity = {
            "dataset_id": cell["frame_id"],
            "arm_id": cell["family_id"],
            "seed": cell["seed"],
            "stage": "refine",
        }
        for boundary, step, wall in (("start", 0, 0.0), ("end", 1000, elapsed[1000])):
            markers.append(
                {
                    **identity,
                    "boundary": boundary,
                    "step": step,
                    "wall_seconds": wall,
                    "label": task["stages"][0]["label"],
                }
            )
        records.append(
            {
                **identity,
                "step": 0,
                "wall_seconds": 0.0,
                "split": "train",
                "metric_id": "refine_rows",
                "value": 256,
            }
        )
        for step, wall in elapsed.items():
            if type(step) is not int or not 1 <= step <= 1000 or counts[step] != 256:
                raise ValueError("invalid native checkpoint step or row budget")
            for metric, value in (
                ("refine_loss", native["loss"][step - 1]),
                ("refine_rows", counts[step]),
            ):
                records.append(
                    {
                        **identity,
                        "step": step,
                        "wall_seconds": wall,
                        "split": "train",
                        "metric_id": metric,
                        "value": _number(value, metric),
                    }
                )
    return {
        "schema_version": 2,
        "records": records,
        "metric_metadata": HISTORY_META,
        "stage_markers": markers,
    }


def paired_comparisons(task: dict, cells: list[dict]) -> list[dict]:
    """Frozen within-frame paired seed arithmetic; no independence-based interval."""
    indexed = {(c["frame_id"], c["family_id"], c["seed"]): c for c in cells}
    results = []
    for family in task["comparators"]:
        if family["id"] == "native_additive":
            continue
        frames = []
        for dataset in task["datasets"]:
            frame = dataset["id"]
            deltas, iou = [], []
            for seed in task["seeds"]:
                candidate = indexed[frame, family["id"], seed]["metrics"]
                base = indexed[frame, "native_additive", seed]["metrics"]
                deltas.append(candidate[QUALITY[0]] - base[QUALITY[0]])
                iou.append(candidate[QUALITY[1]] - base[QUALITY[1]])
            mean, mean_iou = statistics.mean(deltas), statistics.mean(iou)
            meets = (
                mean >= task["decision_policy"]["family_materiality_floor_db"]
                and all(value > 0 for value in deltas)
                and mean_iou >= -task["decision_policy"]["alpha_iou_maximum_loss"]
            )
            frames.append(
                {
                    "frame_id": frame,
                    "seeds": task["seeds"],
                    "paired_psnr_deltas_db": deltas,
                    "mean_psnr_delta_db": mean,
                    "paired_psnr_min_db": min(deltas),
                    "paired_psnr_max_db": max(deltas),
                    "paired_psnr_std_db": statistics.stdev(deltas),
                    "paired_alpha_iou_deltas": iou,
                    "mean_alpha_iou_delta": mean_iou,
                    "meets_frozen_materiality_rule": meets,
                }
            )
        results.append(
            {
                "family_id": family["id"],
                "baseline": "native_additive",
                "frames": frames,
                "meets_rule_in_both_frames": all(
                    row["meets_frozen_materiality_rule"] for row in frames
                ),
            }
        )
    return results


def _group_summary(task: dict, cells: list[dict]) -> list[dict]:
    groups = []
    for dataset in task["datasets"]:
        for family in task["comparators"]:
            rows = [
                c
                for c in cells
                if c["frame_id"] == dataset["id"] and c["family_id"] == family["id"]
            ]
            if [c["seed"] for c in rows] != task["seeds"]:
                raise ValueError("group does not contain each frozen measured seed exactly once")
            fields = [c["metrics"]["field_bytes"] for c in rows]
            if len(set(fields)) != 1:
                raise ValueError("shared field byte count changed between seeds")
            metrics = {name: statistics.mean(c["metrics"][name] for c in rows) for name in QUALITY}
            metrics.update(field_bytes=fields[0], final_gaussians=256)
            for name in (
                "wall_seconds",
                "peak_cuda_allocated_bytes",
                "lift_seconds",
                "refine_seconds",
                "evaluation_seconds",
            ):
                metrics[name] = statistics.median(c["metrics"][name] for c in rows)
            groups.append(
                {
                    "frame_id": dataset["id"],
                    "family_id": family["id"],
                    "metrics": metrics,
                    "seeds": [{"seed": c["seed"], "metrics": c["metrics"]} for c in rows],
                }
            )
    return groups


def _export_rows(task: dict, run: Path, protocol_path: Path, cells: list[dict]) -> dict:
    protocol = _read(protocol_path)
    identity = B.protocol_identity(protocol, protocol_base=protocol_path.parent, allow_review=False)
    families = {
        (frame["id"], family["id"]): (capture["id"], family)
        for capture in protocol["captures"]
        for frame in capture["frames"]
        for family in frame["families"]
    }
    directory = run / "downstream" / "bench019"
    directory.mkdir(exist_ok=False)
    rows, receipts = [], []
    for cell in cells:
        frame, family_id, seed = cell["frame_id"], cell["family_id"], cell["seed"]
        capture, family = families[frame, family_id]
        worker = cell["directory"]
        manifest = run / "inputs" / frame / family_id / "manifest.json"
        if cell["receipt"]["fields"]["manifest"]["sha256"] != B.sha256_file(manifest):
            raise ValueError("worker did not consume the bound family field manifest")
        binding = {
            "capture_id": capture,
            "frame_id": frame,
            "family_id": family_id,
            "seed": seed,
            "initializer": "field_sweep",
            "replicate_id": cell["replicate"],
        }
        stem = f"{frame}_{family_id}_{seed}_{cell['replicate']}"
        run_binding = directory / f"{stem}.run_binding.json"
        _write(
            run_binding,
            {
                "bench019": {
                    "schema": B.RUN_BINDING_SCHEMA,
                    **binding,
                    "field_manifest_sha256": B.sha256_file(manifest),
                    "field_semantic_digest": family["semantics"]["semantic_digest"],
                    "downstream_factor_digest": B.downstream_factor_digest(
                        protocol,
                        frame_id=frame,
                        seed=seed,
                        initializer="field_sweep",
                        protocol_base=protocol_path.parent,
                    ),
                },
                "worker_receipt": B.describe_artifact(worker / "receipt.json"),
                "evaluation_receipt": B.describe_artifact(
                    worker / "evaluation" / "evaluation.json"
                ),
            },
        )
        sources = {"run_receipt": B.describe_artifact(run_binding)}
        bindings = {"stage1": {}, "downstream": {}}
        for predictor in protocol["predictors"]:
            name = predictor["name"]
            bindings["stage1"][name] = {"source": "stage1_metrics", "pointer": f"/metrics/{name}"}
        for response in protocol["responses"]:
            name = response["name"]
            if name in cell["evaluation"]["metrics"]:
                source_name, path = "evaluation", worker / "evaluation" / "evaluation.json"
            elif name in cell["receipt"]["metrics"]:
                source_name, path = "worker", worker / "receipt.json"
            else:
                raise ValueError(f"unsupported frozen response {name}")
            sources[source_name] = B.describe_artifact(path)
            bindings["downstream"][name] = {"source": source_name, "pointer": f"/metrics/{name}"}
        first_view = cell["evaluation"]["per_view"][0]
        artifacts = {
            "field": B.describe_artifact(manifest),
            "history": B.describe_artifact(worker / "history.json"),
            "config": B.describe_artifact(worker / "config.json"),
        }
        for name in ("target", "reconstruction", "error"):
            path = _verified(worker / "evaluation", first_view["previews"][name])
            artifacts[name] = B.describe_artifact(path)
        source = directory / f"{stem}.source.json"
        _write(
            source,
            {
                "schema": B.SOURCE_SCHEMA,
                "protocol_digest": identity,
                "status": "ok",
                "error": "",
                "cell": binding,
                "sources": sources,
                "metric_bindings": bindings,
                "artifacts": artifacts,
            },
        )
        row, receipt = directory / f"{stem}.row.json", directory / f"{stem}.export.json"
        B.export_cell(protocol_path, source, row, receipt)
        rows.append(row)
        receipts.append(receipt)
    assembled, assembly = directory / "rows.jsonl", directory / "assembly.json"
    exported = B.assemble_rows(
        protocol_path, rows, assembled, assembly, export_receipt_paths=receipts
    )
    return {
        "rows": B.describe_artifact(assembled),
        "assembly": B.describe_artifact(assembly),
        "cell_count": len(exported),
        "contained_aa": "additional diagnostic, outside protocol rows",
    }


def _viewer(run: Path, initial: Path, final: Path) -> list[str]:
    root = run.parent.parent
    return [
        ".venv/bin/rtgs",
        "view",
        "--gaussians",
        final.relative_to(root).as_posix(),
        "--initial",
        initial.relative_to(root).as_posix(),
        "--no-open",
    ]


def _copy_new(source: Path, destination: Path) -> None:
    with source.open("rb") as reader, destination.open("xb") as writer:
        writer.write(reader.read())


def publish_result_sources(task: dict, run: Path, protocol_path: Path) -> dict:
    """Publish completed execution sources; independent audit and rendering remain pending."""
    run, protocol_path = Path(run).resolve(), Path(protocol_path).resolve()
    if run.name != task["task_id"] or run.parent.name != "runs":
        raise ValueError("publication requires the canonical task run root")
    lock = _read(run / "task.lock.json")
    if lock["task_id"] != task["task_id"] or lock["command"] != task["run_command"]:
        raise ValueError("task lock identity or command mismatch")
    if _read(run / "aa_replay.json")["status"] != "passed":
        raise ValueError("publication requires both preregistered A/A checks")
    cells = [
        _collect_cell(task, run, dataset["id"], family["id"], seed)
        for dataset in task["datasets"]
        for family in task["comparators"]
        for seed in task["seeds"]
    ]
    aa = task["aa_replay"]
    replays = [
        _collect_cell(task, run, aa["frame_id"], family, aa["seed"], "aa")
        for family in (aa["family_id"], aa["additional_family_id"])
    ]
    if len(cells) != 18 or len(replays) != 2:
        raise ValueError("publication expects 18 primary cells and two diagnostic replays")
    # Resolve and check all major sources before writing any aggregate result.
    groups = _group_summary(task, cells)
    history = training_history(task, cells)
    acquisitions, warmups = [], []
    for dataset in task["datasets"]:
        for family in task["comparators"]:
            frame_id, family_id = dataset["id"], family["id"]
            directory = run / "inputs" / frame_id / family_id
            production = _read(directory / "production.json")
            if production["status"] != "complete":
                raise ValueError("Stage-1 acquisition is incomplete")
            if production["train_views"] != task["splits"][frame_id]["train"]:
                raise ValueError("Stage-1 training split differs from the task")
            acquisitions.append(
                {
                    "frame_id": frame_id,
                    "family_id": family_id,
                    "production": production,
                    "directory": directory,
                    "resource": _read(directory / "resource_receipt.json"),
                }
            )
            warmup = _cell_dir(
                run, frame_id, family_id, task["resource_monitor"]["warmup_seed"], "warmup"
            )
            warmups.append(
                {
                    "frame_id": frame_id,
                    "family_id": family_id,
                    "receipt": B.describe_artifact(warmup / "receipt.json"),
                    "resource": _read(warmup / "resource_receipt.json"),
                }
            )
    bench019 = _export_rows(task, run, protocol_path, [*cells, replays[0]])
    _write(run / "training_history.json", history)
    _write(
        run / "configuration_aggregate.json",
        {
            "schema": "rtgs.bench019.local.configurations.v1",
            "task": task,
            "downstream_cells": [
                {
                    "frame_id": c["frame_id"],
                    "family_id": c["family_id"],
                    "seed": c["seed"],
                    "replicate": c["replicate"],
                    "configuration": c["config"],
                }
                for c in [*cells, *replays]
            ],
            "stage1_acquisitions": [
                {
                    "frame_id": a["frame_id"],
                    "family_id": a["family_id"],
                    "views": a["production"]["views"],
                }
                for a in acquisitions
            ],
        },
    )
    if (run / "gaussians.config.json").exists():
        if _read(run / "gaussians.config.json") != task:
            raise ValueError("existing effective task configuration differs")
    else:
        _write(run / "gaussians.config.json", task)
    boundary = {
        "schema": "rtgs.bench019.local.input_boundary.v1",
        "splits": task["splits"],
        "generated_inputs": B.describe_artifact(run / "generated_inputs.json"),
        "phase2_protocol": B.describe_artifact(protocol_path),
        "reconstruction": "Stage1/lift/refinement use training cameras only; lift has no RGB/masks",
        "reporting": "held-out images loaded only by separate final-model reporting workers",
        "cells": [
            {
                "frame_id": c["frame_id"],
                "family_id": c["family_id"],
                "seed": c["seed"],
                "replicate": c["replicate"],
                "fields": c["receipt"]["fields"],
                "bounds": c["receipt"]["bounds"],
                "evaluated_model_sha256": c["evaluation"]["model_sha256"],
                "heldout_views": c["evaluation"]["view_ids"],
            }
            for c in [*cells, *replays]
        ],
    }
    _write(run / "input_boundary_receipt.json", boundary)
    resources = {
        "schema": "rtgs.bench019.local.resource_aggregate.v1",
        "scope": task["resource_protocol"]["scope"],
        "stage1_acquisitions": [
            {
                "frame_id": a["frame_id"],
                "family_id": a["family_id"],
                "wall_seconds": a["production"]["wall_seconds"],
                "resource": a["resource"],
            }
            for a in acquisitions
        ],
        "primary_cells": [
            {
                "frame_id": c["frame_id"],
                "family_id": c["family_id"],
                "seed": c["seed"],
                "metrics": c["metrics"],
                "resource": c["resource"],
            }
            for c in cells
        ],
        "warmups_excluded": warmups,
        "aa_excluded": [{"family_id": c["family_id"], "resource": c["resource"]} for c in replays],
        "stage1_timing_replication": False,
        "timing_claim": "descriptive local observations; no speed or process-memory substitution",
        "nvml_unavailable_policy": "retain null and errors; no total-device proxy",
    }
    _write(run / "resource_receipt.json", resources)
    representative = next(
        c
        for c in cells
        if c["frame_id"] == task["datasets"][0]["id"]
        and c["family_id"] == "native_additive"
        and c["seed"] == task["seeds"][0]
    )
    initial = _verified(
        representative["directory"], representative["receipt"]["artifacts"]["initial"]["ply"]
    )
    final = _verified(
        representative["directory"], representative["receipt"]["artifacts"]["final"]["ply"]
    )
    _copy_new(initial, run / "gaussians_init.ply")
    _copy_new(final, run / "gaussians.ply")
    fallback = []
    for dataset in task["datasets"]:
        for seed in task["seeds"]:
            rows = [c for c in cells if c["frame_id"] == dataset["id"] and c["seed"] == seed]
            if len({json.dumps(c["receipt"]["bounds"], sort_keys=True) for c in rows}) != 1:
                raise ValueError("camera-only bounds changed across families")
            if len({c["receipt"]["fields"]["camera_geometry_sha256"] for c in rows}) != 1:
                raise ValueError("training camera geometry changed across families")
            fractions = {
                c["family_id"]: c["receipt"]["unsupported_midpoint_count"] / 256 for c in rows
            }
            fallback.append(
                {
                    "frame_id": dataset["id"],
                    "seed": seed,
                    "fractions": fractions,
                    "above_25_percent": any(value > 0.25 for value in fractions.values()),
                    "spread_above_10_percentage_points": max(fractions.values())
                    - min(fractions.values())
                    > 0.1,
                }
            )
    result = {
        "schema": "rtgs.bench019.local.result.v1",
        "task_id": task["task_id"],
        "execution_status": "completed",
        "audit_status": "pending_independent_audit",
        "surrogate_verdict": "not_evaluated_insufficient_scope",
        "claim_boundary": task["claim_boundary"],
        "groups": groups,
        "paired_comparisons": paired_comparisons(task, cells),
        "initializer_diagnostics": fallback,
        "bench019": bench019,
        "aa_replay": _read(run / "aa_replay.json"),
        "cells": [
            {
                "frame_id": c["frame_id"],
                "family_id": c["family_id"],
                "seed": c["seed"],
                "replicate": c["replicate"],
                "receipt": B.describe_artifact(c["directory"] / "receipt.json"),
                "evaluation": B.describe_artifact(
                    c["directory"] / "evaluation" / "evaluation.json"
                ),
            }
            for c in [*cells, *replays]
        ],
    }
    _write(run / "result_sources.json", result)
    metrics = _report_metrics(task, run, lock, cells, groups, acquisitions)
    _write(run / "metrics.json", metrics)
    root = run.parent.parent
    evidence_stem = root / "benchmarks" / "results" / task["task_id"]
    _write(Path(f"{evidence_stem}_RESULT.json"), result)
    lines = [
        f"# {task['task_id']} result",
        "",
        metrics["summary"],
        "",
        task["claim_boundary"],
        "",
        "Execution is complete. Independent audit, report rendering, "
        "and browser/viewer smoke remain pending.",
        "",
        "| Frame | Family | Foreground PSNR (mean3seeds,dB) | Alpha IoU | Wall (median3seeds,s) |",
        "|---|---|---:|---:|---:|",
    ]
    for group in groups:
        m = group["metrics"]
        lines.append(
            f"| {group['frame_id']} | {group['family_id']} | {m[QUALITY[0]]:.6f} "
            f"| {m[QUALITY[1]]:.6f} | {m['wall_seconds']:.6f} |"
        )
    lines += [
        "",
        "Stage-1 acquisition is shared once per frame/family; "
        "it is not three independent timing observations.",
        "",
        f"[Raw result sources](../../runs/{task['task_id']}/result_sources.json)",
        f"[Run metrics](../../runs/{task['task_id']}/metrics.json)",
        "",
    ]
    _text(Path(f"{evidence_stem}_RESULT.md"), "\n".join(lines))
    _write(
        run / "run_receipt.json",
        {
            "schema_version": 1,
            "task_id": task["task_id"],
            "status": "completed",
            "started_at_utc": lock["started_at_utc"],
            "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exit_code": 0,
            "failure_phase": None,
            "message": "Frozen primary matrix and A/A completed; "
            "independent outcome audit and browser gates pending.",
        },
    )
    return result


def _report_metrics(
    task: dict,
    run: Path,
    lock: dict,
    cells: list[dict],
    groups: list[dict],
    acquisitions: list[dict],
) -> dict:
    task_meta = {
        item["id"]: {
            "label": item["label"],
            "unit": item["unit"],
            "direction": item["direction"],
            "group": "Quality" if item["id"] in QUALITY else "Resources",
        }
        for item in task["primary_metrics"]
    }
    control = next(
        g
        for g in groups
        if g["frame_id"] == task["datasets"][0]["id"] and g["family_id"] == "native_additive"
    )
    metrics = {name: control["metrics"][name] for name in task_meta}
    metadata = {
        name: {**meta, "label": f"{control['frame_id']} / {control['family_id']}: {meta['label']}"}
        for name, meta in task_meta.items()
    }
    charts = [
        {
            "id": "quality",
            "title": "Foreground PSNR: mean of three paired seeds per frame/family",
            "unit": "dB",
            "values": [
                {"label": f"{g['frame_id']}/{g['family_id']}", "value": g["metrics"][QUALITY[0]]}
                for g in groups
            ],
        },
        {
            "id": "resources",
            "title": "Peak CUDA allocated: median of three measured workers",
            "unit": "bytes",
            "values": [
                {
                    "label": f"{g['frame_id']}/{g['family_id']}",
                    "value": g["metrics"]["peak_cuda_allocated_bytes"],
                }
                for g in groups
            ],
        },
        {
            "id": "stage_runtime",
            "title": "Shared acquisition once; downstream stages median of three seeds",
            "unit": "seconds",
            "values": [
                {
                    "label": f"{a['frame_id']}/{a['family_id']}/Stage1 acquisition once",
                    "value": a["production"]["wall_seconds"],
                }
                for a in acquisitions
            ]
            + [
                {
                    "label": f"{g['frame_id']}/{g['family_id']}/{stage} median3",
                    "value": g["metrics"][f"{stage}_seconds"],
                }
                for g in groups
                for stage in ("lift", "refine", "evaluation")
            ],
        },
    ]
    required = (
        "gaussians_init.ply",
        "gaussians.ply",
        "training_history.json",
        "gaussians.config.json",
        "input_boundary_receipt.json",
        "resource_receipt.json",
        "run_receipt.json",
        "environment.json",
    )
    artifacts = [{"label": name, "path": name} for name in required]
    artifacts += [
        {"label": name, "path": name}
        for name in ("configuration_aggregate.json", "result_sources.json", "aa_replay.json")
    ]
    # Real calibrated and novel-view previews are generated by the coordinator's
    # separate reporting workers. Link their actual outputs without manufacturing GIFs.
    for base in (run, run / "previews"):
        candidates = base.iterdir() if base == run else base.rglob("*")
        for path in sorted(candidates):
            if path.is_file() and (
                path.suffix in {".png", ".gif"} or path.name == "preview_receipt.json"
            ):
                artifacts.append(
                    {
                        "label": f"Representative preview: {path.name}",
                        "path": path.relative_to(run).as_posix(),
                    }
                )
    for cell in cells:
        base = cell["directory"].relative_to(run).as_posix()
        label = f"{cell['frame_id']}/{cell['family_id']}/seed{cell['seed']}"
        for name in (
            "receipt.json",
            "history.json",
            "initializer.json",
            "config.json",
            "gaussians.ply",
            "evaluation/evaluation.json",
        ):
            artifacts.append({"label": f"{label}/{name}", "path": f"{base}/{name}"})
        for view in cell["evaluation"]["per_view"]:
            for kind, descriptor in view["previews"].items():
                path = _verified(cell["directory"] / "evaluation", descriptor)
                artifacts.append(
                    {
                        "label": f"{label}/{view['view_id']}/{kind}",
                        "path": path.relative_to(run).as_posix(),
                    }
                )
    summaries = {}
    for dataset in task["datasets"]:
        frame = dataset["id"]
        frame_groups = [g for g in groups if g["frame_id"] == frame]
        frame_cells = [c for c in cells if c["frame_id"] == frame]
        frame_metrics, frame_meta = {}, {}
        for group in frame_groups:
            for name in task_meta:
                metric_id = f"{group['family_id']}_{name}"
                frame_metrics[metric_id] = group["metrics"][name]
                frame_meta[metric_id] = {
                    **task_meta[name],
                    "label": f"{group['family_id']}: {task_meta[name]['label']}",
                }
        curves = [
            {
                "id": "refine_objective",
                "title": "Actual refinement checkpoint objectives",
                "x_label": "Optimizer updates",
                "unit": "loss",
                "direction": "lower",
                "series": [
                    {
                        "label": f"{c['family_id']}/seed{c['seed']}",
                        "points": [
                            {"x": step, "value": c["history"]["native_trainer"]["loss"][step - 1]}
                            for step, _wall in c["history"]["native_trainer"]["elapsed"]
                        ],
                    }
                    for c in frame_cells
                ],
            }
        ]
        stage1_series, stage1_artifacts = [], []
        for acquisition in acquisitions:
            if acquisition["frame_id"] != frame:
                continue
            for view in acquisition["production"]["views"]:
                path = _verified(acquisition["directory"], view["history"])
                document = _read(path)
                stage1_series.append(
                    {
                        "label": f"{acquisition['family_id']}/{document['view_id']}/"
                        f"field seed {document['seed']}",
                        "points": [
                            {"x": point["step"], "value": point["pixel_l2"]}
                            for point in document["history"]
                        ],
                    }
                )
                stage1_artifacts.append(
                    {
                        "label": f"Original acquisition history {path.name}",
                        "path": path.relative_to(run).as_posix(),
                    }
                )
        curves.append(
            {
                "id": "stage1_acquisition",
                "title": "Stage1 per-view acquisition (shared once)",
                "x_label": "Per-view fitting updates",
                "unit": "pixel L2",
                "direction": "lower",
                "series": stage1_series,
            }
        )
        selected = next(
            c
            for c in frame_cells
            if c["family_id"] == "native_additive" and c["seed"] == task["seeds"][0]
        )
        initial = _verified(
            selected["directory"], selected["receipt"]["artifacts"]["initial"]["ply"]
        )
        final = _verified(selected["directory"], selected["receipt"]["artifacts"]["final"]["ply"])
        summaries[frame] = {
            "title": frame,
            "summary": "Three families, three paired downstream seeds; "
            "one exposed development frame.",
            "metrics": frame_metrics,
            "metric_metadata": frame_meta,
            "charts": [
                {
                    **chart,
                    "values": [
                        row for row in chart["values"] if row["label"].startswith(frame + "/")
                    ],
                }
                for chart in charts
            ],
            "curves": curves,
            "artifacts": [a for a in artifacts if f"/{frame}/" in a["path"]] + stage1_artifacts,
            "commands": {"viewer": _viewer(run, initial, final)},
            "notes": [
                "Quality is mean per-view then mean per-seed; "
                "wall/memory summaries are medians across primary seeds.",
                "Actual Stage1 field seeds are displayed separately; "
                "acquisition is not replicated across downstream seeds.",
            ],
        }
    return {
        "schema_version": 2,
        "report_template_version": 2,
        "task_id": task["task_id"],
        "summary": "Completed 18 primary cells and two A/A diagnostics on two exposed frames. "
        "See the distinct AUDIT evidence for acceptance. Broad BENCH019 surrogate verdict "
        "is not evaluated because one capture is insufficient.",
        "decision": "development_comparison_completed",
        "claim_boundary": task["claim_boundary"],
        "metrics": metrics,
        "metric_metadata": metadata,
        "charts": charts,
        "artifacts": artifacts,
        "dataset_summaries": summaries,
        "evidence": [
            {"label": suffix, "path": f"benchmarks/results/{task['task_id']}_{suffix}"}
            for suffix in ("RESULT.md", "RESULT.json", "AUDIT.md", "AUDIT.json")
        ],
        "commands": {
            "reproduce": lock["command"],
            "serve_report": [
                ".venv/bin/python",
                "-m",
                "http.server",
                "8765",
                "--directory",
                f"runs/{task['task_id']}",
            ],
            "viewer": _viewer(run, run / "gaussians_init.ply", run / "gaussians.ply"),
        },
        "notes": [
            "Top-level scalar cards describe the first-frame native control; "
            "charts and child pages show every family/frame.",
            "Quality: arithmetic mean of three per-view scores, then three primary seeds. "
            "Field bytes are counted once; resource/time groups show medians across three seeds.",
            "Shared history uses the native optimizer clock excluding checkpoint observers; "
            "cell wall includes serialization and is reported separately.",
            "Stage1 per-view histories retain actual acquisition seeds; "
            "warmups and A/A replays are excluded from primary aggregation.",
            "Raw per-view sums/counts and every primary model/preview are linked. "
            "Final checkpoint is frozen; no heldout selection.",
            "Any observed foreign workload makes timing exploratory; "
            "unavailable NVML process memory stays null.",
            "Clocks include lazy import, JIT, and cache setup when incurred. "
            "Separate warmup workers do not prove later compilation overhead is absent.",
            "Independent acceptance belongs to the canonical AUDIT evidence; "
            "execution completion and scientific materiality are recorded separately.",
        ],
    }


def publish_failure_sources(task: dict | None, run: Path, error: object) -> dict:
    """Preserve first-guard failure metadata, with minimal v2 sources when task is available.

    The coordinator owns the failed run_receipt.json. Existing outputs are never replaced;
    unavailable task context is explicitly recorded rather than reconstructed speculatively.
    """
    run = Path(run).resolve()
    if not run.is_dir():
        raise ValueError("failure publication requires an existing run directory")
    context = task
    required = {"task_id", "claim_boundary", "run_command", "datasets", "stages", "seeds"}
    if not isinstance(context, dict) or not required <= set(context):
        try:
            saved = _read(run / "gaussians.config.json")
        except (ValueError, OSError):
            saved = None
        context = saved if isinstance(saved, dict) and required <= set(saved) else None
    available = [
        B.describe_artifact(path)
        for path in sorted(run.rglob("*.json"))
        if path.name
        in {
            "receipt.json",
            "evaluation.json",
            "history.json",
            "production.json",
            "resource_receipt.json",
            "failure.json",
        }
    ]
    receipt = {
        "schema": "rtgs.bench019.local.failure_sources.v1",
        "status": "failed",
        "task_id": context["task_id"] if context else run.name,
        "task_context_available": context is not None,
        "error_type": type(error).__name__,
        "message": str(error),
        "available_partial_sources": available,
        "metrics_claim": "none; partial source receipts are preserved without aggregation",
        "v2_sources_written": context is not None and not (run / "metrics.json").exists(),
    }
    if (run / "failure_sources.json").exists():
        return _read(run / "failure_sources.json")
    _write(run / "failure_sources.json", receipt)
    if context is None or not receipt["v2_sources_written"]:
        return receipt
    placeholders = {
        "training_history.json": {
            "schema_version": 2,
            "records": [],
            "metric_metadata": {},
            "stage_markers": [],
        },
        "gaussians.config.json": context,
        "environment.json": {
            "schema_version": 1,
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {"python": platform.python_version()},
            "device": {
                "type": "unprobed",
                "name": "device inventory not reached before failure",
                "cuda": None,
            },
        },
        "input_boundary_receipt.json": {
            "schema": "rtgs.bench019.local.failed_boundary.v1",
            "status": "incomplete",
            "declared_splits": context.get("splits", {}),
            "verification": "failed or incomplete; no successful input-boundary claim",
        },
        "resource_receipt.json": {
            "schema": "rtgs.bench019.local.failed_resources.v1",
            "status": "incomplete",
            "aggregate_metrics": None,
            "available_partial_sources": available,
        },
    }
    for name, payload in placeholders.items():
        if not (run / name).exists():
            _write(run / name, payload)
    _write(
        run / "metrics.json",
        {
            "schema_version": 2,
            "report_template_version": 2,
            "task_id": context["task_id"],
            "summary": f"Execution failed: {error}",
            "decision": "failed_no_comparative_claim",
            "claim_boundary": context["claim_boundary"],
            "metrics": {},
            "metric_metadata": {},
            "charts": [],
            "evidence": [],
            "artifacts": [
                {"label": name, "path": name}
                for name in (
                    "training_history.json",
                    "gaussians.config.json",
                    "input_boundary_receipt.json",
                    "resource_receipt.json",
                    "run_receipt.json",
                    "environment.json",
                    "failure_sources.json",
                )
            ],
            "commands": {
                "reproduce": context["run_command"],
                "serve_report": [
                    ".venv/bin/python",
                    "-m",
                    "http.server",
                    "8765",
                    "--directory",
                    f"runs/{context['task_id']}",
                ],
                "viewer": None,
            },
            "notes": [
                "Empty metrics and charts make no result claim; "
                "available partial sources are linked.",
                "No model, timing, boundary-verification, "
                "or independent-audit result is fabricated.",
                "Missing environment/task records can still prevent the structural report gate.",
            ],
        },
    )
    return receipt
