"""Report sources for the RTGS-029 portfolio; the shared renderer owns the page.

Consumes saved cell records only. It never fits models or selects checkpoints. Gate functions
are pure so the frozen decision policy is unit-tested before execution.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import shutil
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
DATASET = "frame_00008"
CONDITIONS = (
    "nb_base",
    "nb_sh1",
    "nb_sh0",
    "nb_reg",
    "nb_30k_d6",
    "nb_30k_d6_lr",
    "nb_ds4",
    "nb_v11",
    "ph_base",
)
TREATMENTS = ("nb_sh1", "nb_sh0", "nb_reg", "nb_30k_d6", "nb_30k_d6_lr", "nb_ds4")
SELECTED = ("nb_base", 9561)
PREVIEWS = (
    "reconstruction_contact_sheet.png",
    "reconstruction.gif",
    "novel_orbit.gif",
    "novel_elevation.gif",
)
LONG = {"nb_30k_d6", "nb_30k_d6_lr"}


def condition_iterations(condition: str) -> int:
    return 30000 if condition in LONG else 8000


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


# --------------------------------------------------------------------------- decision policy


def _cell(cells: list[dict], condition: str, seed: int) -> dict:
    matches = [c for c in cells if c["condition"] == condition and c["seed"] == seed]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {condition}/{seed} cell")
    return matches[0]["evaluation"]["mean"]


def _paired(cells: list[dict], seeds: list[int], left: str, right: str) -> list[dict]:
    rows = []
    for seed in seeds:
        a, b = _cell(cells, left, seed), _cell(cells, right, seed)
        rows.append({"seed": seed, "delta": {key: a[key] - b[key] for key in a}})
    return rows


def _rule(cells, seeds, treatment, control, gain, lpips_margin, alpha_margin=None) -> dict:
    """Per-seed inclusive rule in written form; reject iff every seed loses at least ``gain``."""
    rows = []
    for seed in seeds:
        a, b = _cell(cells, treatment, seed), _cell(cells, control, seed)
        passed = (
            a["foreground_psnr"] >= b["foreground_psnr"] + gain
            and a["crop_lpips"] <= b["crop_lpips"] + lpips_margin
        )
        if alpha_margin is not None:
            passed = passed and a["outside_alpha_mass"] <= b["outside_alpha_mass"] + alpha_margin
        rows.append(
            {
                "seed": seed,
                "delta": {key: a[key] - b[key] for key in a},
                "pass": passed,
                "reverse": a["foreground_psnr"] <= b["foreground_psnr"] - gain,
            }
        )
    if all(row["pass"] for row in rows):
        verdict = "pass"
    elif all(row["reverse"] for row in rows):
        verdict = "reject"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "rows": rows}


def gates(task: dict, cells: list[dict]) -> dict:
    """Apply the frozen per-arm rule against nb_base, per paired seed, in written form."""
    seeds = task["seeds"]
    result = {arm: _rule(cells, seeds, arm, "nb_base", 0.1, 0.005, 0.005) for arm in TREATMENTS}
    result["view_sensitivity_descriptive"] = _paired(cells, seeds, "nb_base", "nb_v11")
    result["field_vs_photographs_descriptive"] = _paired(cells, seeds, "nb_base", "ph_base")
    result["visual_adequacy"] = "requires the independent AUDIT record"
    return result


def decision_text(result: dict) -> str:
    verdicts = ", ".join(f"{arm}: {result[arm]['verdict']}" for arm in TREATMENTS)
    return (
        f"Per-arm verdicts against nb_base ({verdicts}). This is a screen of six simultaneous "
        "comparisons; any pass needs a separately registered confirmation. No default changes."
    )


# --------------------------------------------------------------------------- report sources


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


def history(task: dict, cells: list[dict]) -> dict:
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
        receipt, trace = cell["receipt"], cell["history"]
        common = {"dataset_id": DATASET, "arm_id": cell["condition"], "seed": cell["seed"]}
        for stage in task["stages"]:
            start, end = receipt["stage_intervals"][stage["id"]]
            total = condition_iterations(cell["condition"])
            first = total if stage["id"] == "evaluate" else 0
            last = total if stage["id"] in {"fit", "evaluate"} else 0
            for boundary, stamp, step in [("start", start, first), ("end", end, last)]:
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
        losses = dict(enumerate(trace["loss"], start=1))
        counts = dict(trace["n_gaussians"])
        observed = {row["step"]: row["run_seconds"] for row in trace["checkpoint_observer"]}
        for step, _elapsed in trace["elapsed"]:
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


def row_means(rows: list[dict], names: list[str], keys: list[str]) -> dict:
    if [row["view_id"] for row in rows] != names:
        raise ValueError("metric rows must cover the frozen views exactly once in order")
    if not all(math.isfinite(row[key]) for row in rows for key in keys):
        raise ValueError("nonfinite per-view metric")
    return {key: mean(row[key] for row in rows) for key in keys}


def publish(task: dict, run: Path) -> dict:
    """Build repeatable report sources and once-only pre-audit result records."""
    run = run.resolve()
    if run != ROOT / "runs" / task["task_id"]:
        raise ValueError("only the canonical run root may publish evidence")
    preparation = read(run / "preparation.json")
    initialization = read(run / "initialization.json")
    metric_keys = [item["id"] for item in task["primary_metrics"]]
    heldout = task["splits"][DATASET]["heldout"]
    cells = []
    for condition, seed in task["execution_order"]["cells"]:
        cell_dir = run / "cells" / condition / str(seed)
        cell = {
            "condition": condition,
            "seed": seed,
            "path": str(cell_dir.relative_to(run)),
            "receipt": read(cell_dir / "receipt.json"),
            "history": read(cell_dir / "history.json"),
            "evaluation": read(cell_dir / "evaluation.json"),
        }
        receipt = cell["receipt"]
        if (
            receipt["status"] != "completed"
            or receipt["condition"] != condition
            or receipt["seed"] != seed
            or cell["history"]["executed_iterations"] != condition_iterations(condition)
        ):
            raise ValueError("incomplete or mismatched fitting cell")
        observed = row_means(cell["evaluation"]["per_view"], heldout, metric_keys)
        if any(abs(observed[k] - cell["evaluation"]["mean"][k]) > 1e-10 for k in metric_keys):
            raise ValueError("saved evaluation mean differs from per-view rows")
        cell["evaluation"]["mean"] = observed
        receipt["stage_intervals"] = {
            **preparation["stage_intervals"],
            **initialization["stage_intervals"],
            **receipt["stage_intervals"],
            **cell["evaluation"]["stage_intervals"],
        }
        cells.append(cell)
    write(
        run / "gaussians.config.json",
        {
            "protocol": task,
            "cells": [
                {
                    "condition": c["condition"],
                    "seed": c["seed"],
                    "effective": read(run / c["path"] / "effective_config.json"),
                }
                for c in cells
            ],
        },
    )
    groups = {
        condition: {
            key: mean(c["evaluation"]["mean"][key] for c in cells if c["condition"] == condition)
            for key in metric_keys
        }
        for condition in CONDITIONS
    }
    result_gates = gates(task, cells)
    decision = decision_text(result_gates)
    write(
        run / "comparison.json",
        {
            "groups": groups,
            "gates": result_gates,
            "teacher_mean_foreground_psnr": preparation["teacher_mean_foreground_psnr"],
            "cells": [{k: v for k, v in c.items() if k != "history"} for c in cells],
        },
    )
    write(run / "training_history.json", history(task, cells))
    selected = run / "cells" / SELECTED[0] / str(SELECTED[1])
    for name in ("gaussians_init.ply", "gaussians.ply", *PREVIEWS):
        shutil.copyfile(selected / name, run / name)
    write(
        run / "resource_receipt.json",
        {
            "scope": task["resource_protocol"]["scope"],
            "performance_inference": False,
            "cells": [c["receipt"] for c in cells],
        },
    )
    write(
        run / "input_boundary_receipt.json",
        {
            "task_id": task["task_id"],
            "split": task["splits"],
            "boundary": task["claim_boundary"],
            "preparation_access": preparation["access_guard"],
            "initialization_access": initialization["access_guard"],
            "cell_receipts": [str(Path(c["path"]) / "receipt.json") for c in cells],
            "heldout_role": "Reporting-only after every final fitting model was saved.",
            "mask_role": (
                "Training packed alpha supervises the masked objective; held-out masks are "
                "evaluation-only, so held-out alpha metrics are out-of-sample."
            ),
        },
    )
    _receipt(task, run)
    metric_values, metadata = {}, {}
    for condition in CONDITIONS:
        for spec in task["primary_metrics"]:
            key = spec["id"]
            metric_values[f"{condition}_{key}"] = groups[condition][key]
            metadata[f"{condition}_{key}"] = {
                "label": f"{condition}: {spec['label']}",
                "unit": spec["unit"],
                "group": condition,
                "direction": spec["direction"],
            }
    for spec in task["primary_metrics"]:
        key = spec["id"]
        metric_values[key] = groups[SELECTED[0]][key]
        metadata[key] = {
            "label": f"Primary {SELECTED[0]}: {spec['label']}",
            "unit": spec["unit"],
            "group": "primary",
            "direction": spec["direction"],
        }
    summary = (
        f"{len(cells)} paired development cells completed. Held-out foreground PSNR inside the "
        "mask: " + ", ".join(f"{c} {groups[c]['foreground_psnr']:.3f} dB" for c in CONDITIONS) + "."
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
        "initialization.json",
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
            "title": "Held-out foreground PSNR inside the mask",
            "unit": "dB",
            "values": [{"label": c, "value": groups[c]["foreground_psnr"]} for c in CONDITIONS],
        },
        {
            "id": "resources",
            "title": "Peak allocated CUDA memory (descriptive)",
            "unit": "bytes",
            "values": [
                {
                    "label": f"{c['condition']}/{c['seed']}",
                    "value": c["receipt"]["peak_allocated_bytes"],
                }
                for c in cells
            ],
        },
        {
            "id": "stage_runtime",
            "title": "Recorded stage durations (descriptive)",
            "unit": "seconds",
            "values": [
                {"label": f"{c['condition']}/{c['seed']}/{stage}", "value": bounds[1] - bounds[0]}
                for c in cells
                for stage, bounds in c["receipt"]["stage_intervals"].items()
            ],
        },
    ]
    notes = [
        f"Root preview is the prospectively selected {SELECTED[0]}/{SELECTED[1]} model.",
        "Colour metrics are computed only inside held-out masks; floaters use rendered alpha "
        "outside the 3-pixel-dilated held-out mask.",
        "Held-out masks and colour never enter fitting; alpha metrics are out-of-sample.",
        "All models are rendered at downscale 4 and box-averaged to the downscale-8 grid.",
        "Six simultaneous treatment comparisons: a pass is a screening signal only.",
        "Shared preparation/initialization intervals recur across series and must not be summed.",
        "No timing advantage is inferred on a local desktop GPU.",
    ]
    write(
        run / "metrics.json",
        {
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
        },
    )
    result = {
        "schema_version": 1,
        "task_id": task["task_id"],
        "summary": summary,
        "decision": decision,
        "claim_boundary": task["claim_boundary"],
        "groups": groups,
        "gates": result_gates,
        "command": task["run_command"],
        "source_lock": read(run / "task.lock.json"),
        "raw_comparison_path": str(run.relative_to(ROOT) / "comparison.json"),
    }
    _once(ROOT / f"{evidence_stem}_RESULT.json", result)
    rows = [
        "| Condition | " + " | ".join(metric_keys) + " |",
        "|---|" + "---:|" * len(metric_keys),
    ]
    for condition in CONDITIONS:
        values = " | ".join(f"{groups[condition][k]:.6f}" for k in metric_keys)
        rows.append(f"| {condition} | {values} |")
    text = (
        f"# {task['title']}\n\n{summary}\n\n{decision}\n\n"
        + "\n".join(rows)
        + f"\n\n{task['claim_boundary']}\n\n"
        + "Numeric gates precede the independent audit and do not assert visual adequacy. "
        + f"Raw paired values and source lock: `{evidence_stem}_RESULT.json`. "
        + f"Report: `runs/{task['task_id']}/index.html`.\n"
    )
    result_md = ROOT / f"{evidence_stem}_RESULT.md"
    if result_md.exists() and result_md.read_text() != text:
        raise ValueError("refusing to replace conflicting RESULT Markdown")
    if not result_md.exists():
        result_md.write_text(text)
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
