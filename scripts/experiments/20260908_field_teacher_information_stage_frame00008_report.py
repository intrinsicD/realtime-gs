"""Task-owned report sources; shared experiment renderer owns HTML/Markdown/manifest.

Consumes saved cell records only. This module never fits models or selects checkpoints.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import shutil
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("rgb", "field_high", "field_low")
PREVIEWS = (
    "reconstruction_contact_sheet.png",
    "reconstruction.gif",
    "novel_orbit.gif",
    "novel_elevation.gif",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _once(path: Path, value: object) -> None:
    if path.exists():
        if read(path) != value:
            raise ValueError(f"refusing to replace existing evidence: {path}")
    else:
        write(path, value)


def _commands(task: dict, run: Path) -> dict:
    relative = run.relative_to(ROOT).as_posix()
    return {
        "reproduce": task["run_command"],
        "serve_report": [".venv/bin/python", "-m", "http.server", "8765", "--directory", relative],
        "viewer": [
            ".venv/bin/rtgs",
            "view",
            "--gaussians",
            f"{relative}/gaussians.ply",
            "--initial",
            f"{relative}/gaussians_init.ply",
            "--no-open",
            "--port",
            "8879",
        ],
    }


def _receipt(task: dict, run: Path, *, failed: str | None = None) -> None:
    lock = read(run / "task.lock.json")
    existing = run / "run_receipt.json"
    if existing.exists() and failed is None and read(existing).get("status") == "completed":
        return
    write(
        existing,
        {
            "schema_version": 1,
            "task_id": task["task_id"],
            "status": "failed" if failed else "completed",
            "started_at_utc": lock["started_at_utc"],
            "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exit_code": 1 if failed else 0,
            "failure_phase": "execution" if failed else None,
            "message": failed or "All frozen fitting cells and final evaluations completed.",
        },
    )


def _history(task: dict, cells: list[dict]) -> dict:
    metadata = {
        "loss_total": {
            "label": "Training objective",
            "unit": "loss",
            "group": "Fitting",
            "direction": "lower",
        },
        "n_gaussians": {
            "label": "Live 3D Gaussians",
            "unit": "count",
            "group": "Capacity",
            "direction": "descriptive",
        },
    }
    records, markers = [], []
    for cell in cells:
        receipt, history = cell["receipt"], cell["history"]
        common = {"dataset_id": "frame_00008", "arm_id": cell["arm"], "seed": cell["seed"]}
        for stage in task["stages"]:
            start, end = receipt["stage_intervals"][stage["id"]]
            first_step = 8000 if stage["id"] == "evaluate" else 0
            last_step = 8000 if stage["id"] in {"fit", "evaluate"} else 0
            for boundary, stamp, step in [("start", start, first_step), ("end", end, last_step)]:
                markers.append(
                    {
                        **common,
                        "stage": stage["id"],
                        "boundary": boundary,
                        "label": stage["label"],
                        "step": step,
                        "wall_seconds": stamp,
                    }
                )
        losses = dict(enumerate(history["loss"], start=1))
        counts = dict(history["n_gaussians"])
        observed = {row["step"]: row["run_seconds"] for row in history["checkpoint_observer"]}
        for step, _elapsed in history["elapsed"]:
            stamp = observed[step]
            for metric, values in [("loss_total", losses), ("n_gaussians", counts)]:
                if step in values:
                    records.append(
                        {
                            **common,
                            "stage": "fit",
                            "split": "train",
                            "step": step,
                            "wall_seconds": stamp,
                            "metric_id": metric,
                            "value": values[step],
                        }
                    )
    return {
        "schema_version": 2,
        "records": records,
        "metric_metadata": metadata,
        "stage_markers": markers,
    }


def _gates(task: dict, groups: dict, cells: list[dict], preparation: dict) -> dict:
    teachers = preparation["teacher_metrics"]["field_high"]
    teacher_psnr = mean(row["foreground_psnr"] for row in teachers)
    teacher_lpips = mean(row["crop_lpips"] for row in teachers)
    qualified = teacher_psnr >= 30 and teacher_lpips <= 0.08
    adequate_rows = []
    pairs = []
    for seed in task["seeds"]:
        a = next(c["evaluation"]["mean"] for c in cells if c["arm"] == "rgb" and c["seed"] == seed)
        b = next(
            c["evaluation"]["mean"] for c in cells if c["arm"] == "field_high" and c["seed"] == seed
        )
        adequate_rows.append(
            {
                "seed": seed,
                "numeric_pass": a["foreground_psnr"] >= 25
                and a["crop_lpips"] <= 0.15
                and a["alpha_iou"] >= 0.90,
            }
        )
        delta = {key: b[key] - a[key] for key in a}
        pairs.append(
            {
                "seed": seed,
                "delta_b_minus_a": delta,
                "numeric_pass": b["foreground_psnr"] >= a["foreground_psnr"] - 0.5
                and b["crop_lpips"] <= a["crop_lpips"] + 0.02
                and b["boundary_psnr"] >= a["boundary_psnr"] - 0.5
                and b["alpha_iou"] >= a["alpha_iou"] - 0.02,
            }
        )
    return {
        "teacher_qualified": qualified,
        "teacher_foreground_psnr": teacher_psnr,
        "teacher_crop_lpips": teacher_lpips,
        "a_numeric_adequacy": adequate_rows,
        "b_paired_numeric_gates": pairs,
        "numeric_prerequisites_pass": qualified
        and all(r["numeric_pass"] for r in adequate_rows)
        and all(r["numeric_pass"] for r in pairs),
        "visual_adequacy": "requires independent AUDIT record",
        "group_means": groups,
    }


def _row_means(rows: list[dict], names: list[str], keys: list[str]) -> dict:
    if [row["view_id"] for row in rows] != names:
        raise ValueError("metric rows must cover the frozen views exactly once in order")
    if not all(math.isfinite(row[key]) for row in rows for key in keys):
        raise ValueError("nonfinite per-view metric")
    return {key: mean(row[key] for row in rows) for key in keys}


def publish(task: dict, run: Path) -> dict:
    """Build repeatable report sources and once-only pre-audit result records."""
    run = run.resolve()
    preparation = read(run / "preparation.json")
    initialization = read(run / "initialization.json")
    metric_keys = [item["id"] for item in task["primary_metrics"]]
    for arm in ("field_high", "field_low"):
        _row_means(
            preparation["teacher_metrics"][arm],
            task["splits"]["frame_00008"]["train"],
            [key for key in metric_keys if key != "alpha_iou"],
        )
    cells = []
    for arm, seed in task["execution_order"]["cells"]:
        cell_dir = run / "cells" / arm / str(seed)
        cells.append(
            {
                "arm": arm,
                "seed": seed,
                "path": str(cell_dir.relative_to(run)),
                "receipt": read(cell_dir / "receipt.json"),
                "history": read(cell_dir / "history.json"),
                "evaluation": read(cell_dir / "evaluation.json"),
            }
        )
        cell = cells[-1]
        receipt = cell["receipt"]
        if (
            receipt["status"] != "completed"
            or receipt["arm"] != arm
            or receipt["seed"] != seed
            or cell["history"]["executed_iterations"] != 8000
        ):
            raise ValueError("incomplete or mismatched fitting cell")
        observed_means = _row_means(
            cell["evaluation"]["per_view"], task["splits"]["frame_00008"]["heldout"], metric_keys
        )
        if any(abs(observed_means[k] - cell["evaluation"]["mean"][k]) > 1e-10 for k in metric_keys):
            raise ValueError("saved evaluation mean differs from per-view rows")
        cell["evaluation"]["mean"] = observed_means
        receipt["stage_intervals"] = {
            **preparation["stage_intervals"],
            **initialization["stage_intervals"],
            **receipt["stage_intervals"],
            **cell["evaluation"]["stage_intervals"],
        }
    write(
        run / "gaussians.config.json",
        {
            "protocol": task,
            "cells": [
                {
                    "arm": c["arm"],
                    "seed": c["seed"],
                    "effective": read(run / c["path"] / "effective_config.json"),
                }
                for c in cells
            ],
        },
    )
    groups = {
        arm: {
            key: mean(c["evaluation"]["mean"][key] for c in cells if c["arm"] == arm)
            for key in metric_keys
        }
        for arm in ARMS
    }
    gates = _gates(task, groups, cells, preparation)
    write(
        run / "comparison.json",
        {
            "groups": groups,
            "gates": gates,
            "cells": [{k: v for k, v in c.items() if k != "history"} for c in cells],
        },
    )
    write(run / "training_history.json", _history(task, cells))
    selected = run / "cells" / "field_high" / "8101"
    for name in ("gaussians_init.ply", "gaussians.ply", *PREVIEWS):
        shutil.copyfile(selected / name, run / name)
    resources = {
        "scope": task["resource_protocol"]["scope"],
        "performance_inference": False,
        "gpu_contended": True,
        "cells": [c["receipt"] for c in cells],
    }
    write(run / "resource_receipt.json", resources)
    write(
        run / "input_boundary_receipt.json",
        {
            "task_id": task["task_id"],
            "split": task["splits"],
            "boundary": task["claim_boundary"],
            "preparation": preparation,
            "cell_receipts": [str(Path(c["path"]) / "receipt.json") for c in cells],
            "heldout_role": "Reporting-only after every final fitting model was saved.",
            "reference_initialization": (
                "Shared training RGB/mask visual hull; not field-only initialization."
            ),
        },
    )
    _receipt(task, run)
    metric_values, metadata = {}, {}
    for arm in ARMS:
        for spec in task["primary_metrics"]:
            key = spec["id"]
            metric_values[f"{arm}_{key}"] = groups[arm][key]
            metadata[f"{arm}_{key}"] = {
                "label": f"{arm}: {spec['label']}",
                "unit": spec["unit"],
                "group": arm,
                "direction": spec["direction"],
            }
    for spec in task["primary_metrics"]:
        key = spec["id"]
        metric_values[key] = groups["field_high"][key]
        metadata[key] = {
            "label": f"B primary: {spec['label']}",
            "unit": spec["unit"],
            "group": "B primary",
            "direction": spec["direction"],
        }
    if not gates["teacher_qualified"]:
        decision = (
            "High-capacity teacher failed its fidelity qualification; "
            "high-fidelity reconstruction remains unresolved."
        )
    elif not all(row["numeric_pass"] for row in gates["a_numeric_adequacy"]):
        decision = (
            "Photograph baseline failed the frozen adequacy gate; resolve baseline "
            "geometry/solver quality before interpreting field-only viability."
        )
    elif not gates["numeric_prerequisites_pass"]:
        decision = "Field target refinement failed the frozen paired noninferiority gates."
    else:
        decision = (
            "Numeric prerequisites pass; independent visual audit required "
            "before field-derived initialization experiments."
        )
    summary = (
        f"Nine paired development cells completed. Heldout foreground PSNR: "
        f"photographs {groups['rgb']['foreground_psnr']:.3f} dB, "
        f"100k fields {groups['field_high']['foreground_psnr']:.3f} dB, "
        f"low-budget fields {groups['field_low']['foreground_psnr']:.3f} dB."
    )
    evidence_stem = f"benchmarks/results/{task['task_id']}"
    evidence = [
        {"label": label, "path": f"{evidence_stem}_{suffix}"}
        for label, suffix in [
            ("Result note", "RESULT.md"),
            ("Machine result", "RESULT.json"),
            ("Independent audit", "AUDIT.md"),
            ("Machine audit", "AUDIT.json"),
        ]
    ]
    core = [
        "gaussians_init.ply",
        "gaussians.ply",
        "training_history.json",
        "gaussians.config.json",
        "input_boundary_receipt.json",
        "resource_receipt.json",
        "run_receipt.json",
        "environment.json",
        "comparison.json",
        "preparation.json",
        "source_snapshot/manifest.json",
        *PREVIEWS,
    ]
    artifacts = [{"label": name, "path": name} for name in core]
    for cell in cells:
        for name in ("receipt.json", "evaluation.json", "gaussians.ply", *PREVIEWS):
            value = f"{cell['path']}/{name}"
            artifacts.append({"label": value, "path": value})
    charts = [
        {
            "id": "quality",
            "title": "Heldout foreground PSNR",
            "unit": "dB",
            "values": [{"label": a, "value": groups[a]["foreground_psnr"]} for a in ARMS],
        },
        {
            "id": "resources",
            "title": "Peak allocated CUDA memory (descriptive)",
            "unit": "bytes",
            "values": [
                {"label": f"{c['arm']}/{c['seed']}", "value": c["receipt"]["peak_allocated_bytes"]}
                for c in cells
            ],
        },
        {
            "id": "stage_runtime",
            "title": "Recorded stage durations on shared GPU (descriptive)",
            "unit": "seconds",
            "values": [
                {"label": f"{c['arm']}/{c['seed']}/{stage}", "value": bounds[1] - bounds[0]}
                for c in cells
                for stage, bounds in c["receipt"]["stage_intervals"].items()
            ],
        },
    ]
    notes = [
        "Root preview is the prospectively selected B/8101 model; every cell remains linked.",
        "Fitting curves use recorded checkpoint clocks only. All stages share the run UTC origin; "
        "shared acquisition/init intervals recur across paired series and must not be summed "
        "as independent costs.",
        "Full native per-step loss arrays are retained in cell history.json; "
        "no synthetic timestamps are assigned to unsynchronized steps.",
        "C changes acquisition count, fit iterations and fit seed together; "
        "it is an existing low-budget family comparison.",
        "Fields are decoded to dense targets for this information diagnostic. "
        "It does not establish compact-process memory savings.",
        "No timing advantage can be inferred on this already contended GPU.",
    ]
    payload = {
        "schema_version": 2,
        "report_template_version": 2,
        "task_id": task["task_id"],
        "summary": summary,
        "decision": decision,
        "claim_boundary": task["claim_boundary"],
        "metrics": metric_values,
        "metric_metadata": metadata,
        "charts": charts,
        "artifacts": artifacts,
        "evidence": evidence,
        "commands": _commands(task, run),
        "notes": notes,
    }
    write(run / "metrics.json", payload)
    result = {
        "schema_version": 1,
        "task_id": task["task_id"],
        "summary": summary,
        "decision": decision,
        "claim_boundary": task["claim_boundary"],
        "groups": groups,
        "gates": gates,
        "command": task["run_command"],
        "source_lock": read(run / "task.lock.json"),
        "raw_comparison_path": str(run.relative_to(ROOT) / "comparison.json"),
    }
    _once(ROOT / f"{evidence_stem}_RESULT.json", result)
    result_md = ROOT / f"{evidence_stem}_RESULT.md"
    rows = [
        "| Arm | Foreground PSNR | Full PSNR | Boundary PSNR | Crop LPIPS | Alpha IoU |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        rows.append(
            "| " + arm + " | " + " | ".join(f"{groups[arm][k]:.6f}" for k in metric_keys) + " |"
        )
    result_text = (
        f"# {task['title']}\n\n{summary}\n\n{decision}\n\n"
        + "\n".join(rows)
        + f"\n\n{task['claim_boundary']}\n\n"
        + "Numeric gate computation precedes independent audit; "
        + "it does not assert visual adequacy. "
        + f"Raw paired values and source lock: `{evidence_stem}_RESULT.json`. "
        + "Reproduce argv is bound in that record. "
        + f"Report: `runs/{task['task_id']}/index.html`.\n"
    )
    if result_md.exists() and result_md.read_text() != result_text:
        raise ValueError("refusing to replace conflicting RESULT Markdown")
    if not result_md.exists():
        result_md.write_text(result_text)
    return result


def publish_failure(task: dict, run: Path, error: str) -> dict:
    """Preserve a failed run explicitly; do not manufacture metrics or evidence."""
    _receipt(task, run, failed=error)
    if not (run / "training_history.json").exists():
        write(
            run / "training_history.json",
            {"schema_version": 2, "records": [], "metric_metadata": {}, "stage_markers": []},
        )
    for name in ("gaussians.config.json", "input_boundary_receipt.json", "resource_receipt.json"):
        if not (run / name).exists():
            write(run / name, {"task_id": task["task_id"], "status": "failed", "message": error})
    payload = {
        "schema_version": 2,
        "report_template_version": 2,
        "task_id": task["task_id"],
        "summary": "Experiment execution stopped before completion.",
        "decision": error,
        "claim_boundary": task["claim_boundary"],
        "metrics": {},
        "metric_metadata": {},
        "charts": [],
        "artifacts": [],
        "evidence": [],
        "commands": {**_commands(task, run), "viewer": None},
        "notes": ["Failed execution is inspectable but not a results-bearing bundle."],
    }
    write(run / "metrics.json", payload)
    return payload
