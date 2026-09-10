"""Task-owned diagnostic report sources; the shared renderer owns the final page.

This module consumes completed observations. It never fits or selects a model.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("means", "quats", "scales", "opacities", "sh0", "shN")
COMPONENTS = ("l1", "dssim", "total")
REGIONS = ("interior", "boundary", "exterior")
STATS = (
    "photo_norm",
    "field_norm",
    "difference_norm",
    "difference_over_photo_norm",
    "cosine",
    "max_absolute_difference",
)
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


def once(path: Path, value: object) -> None:
    if path.exists():
        if read(path) != value:
            raise ValueError(f"refusing to overwrite evidence: {path}")
    else:
        write(path, value)


def nullable_summary(values: list) -> dict:
    defined = [v for v in values if v is not None]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in defined):
        raise ValueError("nonfinite diagnostic statistic")
    return {
        "mean": mean(defined) if defined else None,
        "min": min(defined) if defined else None,
        "max": max(defined) if defined else None,
        "defined_count": len(defined),
        "undefined_count": len(values) - len(defined),
    }


def coverage(rows: list, expected: list[str]) -> None:
    if [r["view_id"] for r in rows] != expected:
        raise ValueError("diagnostic views must cover the frozen order exactly once")


def validate_comparisons(comparisons: dict, epsilon: float = 1e-12) -> None:
    if set(comparisons) != set(COMPONENTS):
        raise ValueError("comparison component coverage is incomplete")
    for component in COMPONENTS:
        if set(comparisons[component]) != set(GROUPS):
            raise ValueError("comparison group coverage is incomplete")
        for group in GROUPS:
            stats = comparisons[component][group]
            if not set(STATS) <= set(stats):
                raise ValueError("comparison statistic coverage is incomplete")
            for key in STATS:
                value = stats[key]
                null_allowed = key in {"cosine", "difference_over_photo_norm"}
                if value is None and null_allowed:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("invalid comparison statistic")
                if not math.isfinite(value):
                    raise ValueError("nonfinite comparison statistic")
                if key != "cosine" and value < 0:
                    raise ValueError("gradient norms/relative differences cannot be negative")
                if key == "cosine" and abs(value) > 1 + 1e-12:
                    raise ValueError("cosine outside [-1,1]")
            zero_photo = stats["photo_norm"] <= epsilon
            zero_field = stats["field_norm"] <= epsilon
            if (stats["difference_over_photo_norm"] is None) != zero_photo:
                raise ValueError("relative gradient null does not match reference norm")
            if (stats["cosine"] is None) != (zero_photo or zero_field):
                raise ValueError("gradient cosine null does not match parameter norms")


def validate_control(control: dict, state: dict, view: str) -> None:
    if control.get("state_id") != state["id"] or control.get("view_id") != view:
        raise ValueError("identity control state/view mismatch")
    if control.get("independent_render_graphs") is not True:
        raise ValueError("identity control lacks independent render graphs")
    expected = {"gradient_atol": 1e-8, "gradient_rtol": 1e-4, "loss_atol": 1e-7}
    if any(control.get(key) != value for key, value in expected.items()):
        raise ValueError("identity control tolerances changed")
    validate_comparisons(control["comparisons"])
    if set(control["losses"]) != set(COMPONENTS):
        raise ValueError("identity loss coverage incomplete")
    passed = True
    for component in COMPONENTS:
        loss = control["losses"][component]
        for key in ("photo_path", "teacher_path", "absolute_difference"):
            if not isinstance(loss[key], (float, int)) or not math.isfinite(loss[key]):
                raise ValueError("nonfinite identity loss")
        difference = abs(loss["photo_path"] - loss["teacher_path"])
        if abs(difference - loss["absolute_difference"]) > 1e-12:
            raise ValueError("identity loss difference disagrees with values")
        passed &= difference <= expected["loss_atol"]
        for group in GROUPS:
            stats = control["comparisons"][component][group]
            count, violations = stats["element_count"], stats["tolerance_violations"]
            excess = stats["max_tolerance_excess"]
            if not isinstance(count, int) or count < 1:
                raise ValueError("identity control element count is invalid")
            if not isinstance(violations, int) or not 0 <= violations <= count:
                raise ValueError("identity control violation count is invalid")
            if not isinstance(excess, (float, int)) or not math.isfinite(excess):
                raise ValueError("identity control tolerance excess is invalid")
            passed &= violations == 0 and excess <= 0
    if control.get("passed") is not True or not passed:
        raise ValueError("failed identity control cannot produce a completed result")


def summarize(task: dict, residuals: dict, states: list[dict]) -> dict:
    views = task["gradient_protocol"]["view_order"]
    coverage(residuals["rows"], views)
    if [s["state"]["id"] for s in states] != task["gradient_protocol"]["state_order"]:
        raise ValueError("diagnostics must cover the frozen state order exactly once")
    region_keys = (
        "signed_mean_rgb",
        "mae_region",
        "mse_region",
        "absolute_error_sum",
        "absolute_error_share_full_canvas",
        "full_canvas_l1_contribution",
        "photo_local_contrast",
        "field_local_contrast",
        "residual_lowpass_mse",
        "residual_highpass_mse",
        "pixel_count",
    )
    regions = {
        region: {
            key: nullable_summary([row["regions"][region][key] for row in residuals["rows"]])
            for key in region_keys
        }
        for region in REGIONS
    }
    summaries = []
    for state in states:
        coverage(state["rows"], views)
        if state.get("status") != "completed" or state.get("saved_state_unchanged") is not True:
            raise ValueError("state diagnostics are incomplete or parameters changed")
        if state["state"] != next(s for s in task["states"] if s["id"] == state["state"]["id"]):
            raise ValueError("saved state metadata differs from frozen protocol")
        validate_control(state["identity_control"], state["state"], views[0])
        validate_comparisons(state["aggregate"])
        for row in state["rows"]:
            validate_comparisons(row["comparisons"])
        groups = {
            component: {
                group: {
                    stat: nullable_summary(
                        [row["comparisons"][component][group][stat] for row in state["rows"]]
                    )
                    for stat in STATS
                }
                for group in GROUPS
            }
            for component in COMPONENTS
        }
        summaries.append(
            {
                "state": state["state"],
                "per_view_statistics": groups,
                "gradient_of_view_mean": state["aggregate"],
                "identity_control": state["identity_control"],
                "mean_losses": {
                    target: {
                        c: mean(row["losses"][target][c] for row in state["rows"])
                        for c in COMPONENTS
                    }
                    for target in ("rgb", "field_high")
                },
            }
        )
    primary = {
        "target_boundary_mae": regions["boundary"]["mae_region"]["mean"],
        "target_exterior_bias": regions["exterior"]["signed_mean_rgb"]["mean"],
    }
    for metric, stat in (
        ("means_total_cosine", "cosine"),
        ("means_total_relative_delta", "difference_over_photo_norm"),
    ):
        kind_means = []
        for kind in ("initial", "rgb_final", "field_high_final"):
            values = [
                s["per_view_statistics"]["total"]["means"][stat]["mean"]
                for s in summaries
                if s["state"]["kind"] == kind
            ]
            kind_means.append(nullable_summary(values)["mean"])
        primary[metric] = nullable_summary(kind_means)["mean"]
    if not all(v is not None and math.isfinite(v) for v in primary.values()):
        raise ValueError("a frozen primary metric is undefined; cannot silently invent a value")
    return {"regions": regions, "states": summaries, "primary_metrics": primary}


def history(task: dict, states: list[dict], execution: dict) -> dict:
    records, markers = [], []
    for state in states:
        for target in ("rgb", "field_high"):
            common = {
                "dataset_id": "frame_00008",
                "arm_id": f"{state['state']['kind']}_{target}",
                "seed": state["state"]["seed"],
            }
            for stage in task["stages"]:
                start, end = execution["stage_intervals"][stage["id"]]
                first = 22 if stage["id"] == "presentation" else 0
                last = 0 if stage["id"] == "residual" else 22
                for boundary, step, stamp in (("start", first, start), ("end", last, end)):
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
            for row in state["rows"]:
                records.append(
                    {
                        **common,
                        "stage": "gradient",
                        "split": "diagnostic",
                        "step": row["step"],
                        "wall_seconds": row["wall_seconds"],
                        "metric_id": "diagnostic_loss_total",
                        "value": row["losses"][target]["total"],
                    }
                )
    return {
        "schema_version": 2,
        "records": records,
        "stage_markers": markers,
        "metric_metadata": {
            "diagnostic_loss_total": {
                "label": "Fixed-state objective across measured views (no fitting)",
                "unit": "loss",
                "group": "Diagnostic",
                "direction": "descriptive",
            }
        },
    }


def heatmap(run: Path, summary: dict) -> None:
    """Write a small deterministic vector chart of within-state mean-view cosines."""
    width, height = 1070, 690
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" fill="#17212f">',
        '<text x="28" y="35" font-size="22">Target gradients at the same saved state</text>',
        '<text x="28" y="61" font-size="14">Total loss; mean of defined per-view cosines. '
        "1 = aligned, 0 = orthogonal, −1 = opposed.</text>",
    ]
    for col, group in enumerate(GROUPS):
        parts.append(f'<text x="{285 + 120 * col}" y="98" font-size="15">{group}</text>')
    for index, state in enumerate(summary["states"]):
        y = 116 + index * 54
        label = state["state"]["id"]
        parts.append(f'<text x="28" y="{y + 29}" font-size="14">{html.escape(label)}</text>')
        for col, group in enumerate(GROUPS):
            value = state["per_view_statistics"]["total"][group]["cosine"]["mean"]
            if value is None:
                colour, label = "#e8e8e8", "undefined"
            else:
                intensity = min(1.0, abs(value))
                pale = round(245 - 150 * intensity)
                colour = f"rgb({pale},{pale + 5},245)" if value >= 0 else f"rgb(245,{pale},{pale})"
                label = f"{value:.3f}"
            x = 272 + col * 120
            parts.extend(
                [
                    f'<rect x="{x}" y="{y}" width="111" height="44" rx="4" fill="{colour}"/>',
                    f'<text x="{x + 10}" y="{y + 28}" font-size="14">{label}</text>',
                ]
            )
    parts.extend(
        [
            '<text x="28" y="638" font-size="14">Quats = saved raw quaternion coordinates; '
            "scales = log scales; opacities = local logit derivative.</text>",
            '<text x="28" y="662" font-size="14">No optimization or cross-topology vector '
            "comparison. This does not identify the cause of the earlier failure.</text>",
            "</g></svg>",
        ]
    )
    (run / "gradient_directions.svg").write_text("\n".join(parts) + "\n")


def display(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.6f}"


def commands(task: dict) -> dict:
    relative = f"runs/{task['task_id']}"
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
            "8880",
        ],
    }


def publish(task: dict, run: Path) -> dict:
    residuals = read(run / "residuals.json")
    states = [read(run / "states" / s["id"] / "diagnostics.json") for s in task["states"]]
    execution = read(run / "execution.json")
    if (
        execution.get("status") != "completed"
        or execution.get("states") != task["gradient_protocol"]["state_order"]
        or execution.get("optimization_steps") != 0
        or execution.get("topology_updates") != 0
    ):
        raise ValueError("execution is incomplete or violates the no-update protocol")
    comparison = summarize(task, residuals, states)
    lock = read(run / "task.lock.json")
    summary = (
        "Completed training-only target residuals and common-state gradient measurements on "
        "all 22 cameras at nine predeclared saved states. All nine equal-target controls passed. "
        "Saved model snapshots were rendered without fitting or topology changes."
    )
    decision = (
        "Development diagnostic complete; interpret region and within-group gradients with "
        "the independent audit. These observations do not identify a causal remedy or reopen "
        "the failed reconstruction-quality prerequisites."
    )
    notes = [
        "Previously exposed capture; local first-order development diagnostics, "
        "not new reconstruction quality or physical-density evidence.",
        "Masks define regions only. Loss and gradients use the full cached training canvas. "
        "Raw-seal checks hash heldout bytes; the measurement worker never decodes them.",
        "Step is a measured-view ordinal. Parameters never change: the curves "
        "are not fitting histories. Global diagnostic stage intervals are repeated for context and "
        "must not be summed across series.",
        "Root PLYs and previews show the prospectively selected existing snapshots. "
        "PNG/GIF previews are display-clamped; numerical loss uses unclamped renderer output.",
        "Effective-opacity gradients are mapped by alpha*(1-alpha) to local logit coordinates. "
        "Historical raw logits and Adam updates are unavailable and are not reconstructed.",
        "Undefined cosine bars are omitted; raw summaries retain null counts. "
        "Cosines and relative deltas are null near zero. Parameter-group norms "
        "have different units; they cannot rank causal importance. Mean per-view statistics and "
        "statistics of the mean gradient are reported separately.",
        "Equal-target controls cover C0004 per state; they do not bound all views' CUDA noise. "
        "Shared-GPU resource measurements support no speed or memory advantage claim.",
        "Lowpass and highpass residual summaries are not an orthogonal energy partition. "
        "No regional gradient localization or intervention was performed.",
    ]
    metadata = {
        spec["id"]: {
            "label": spec["label"],
            "unit": spec["unit"],
            "group": "Descriptive diagnostic",
            "direction": spec["direction"],
        }
        for spec in task["primary_metrics"]
    }
    resource = execution["resource"]
    charts = [
        {
            "id": "quality",
            "title": "Within-state means-gradient cosine (mean over views)",
            "unit": "cosine",
            "values": [
                {
                    "label": s["state"]["id"],
                    "value": s["per_view_statistics"]["total"]["means"]["cosine"]["mean"],
                }
                for s in comparison["states"]
                if s["per_view_statistics"]["total"]["means"]["cosine"]["mean"] is not None
            ],
        },
        {
            "id": "resources",
            "title": "Diagnostic CUDA process peaks (descriptive)",
            "unit": "bytes",
            "values": [
                {"label": key, "value": resource[key]}
                for key in ("torch_cuda_max_memory_allocated", "torch_cuda_max_memory_reserved")
            ],
        },
        {
            "id": "stage_runtime",
            "title": "Diagnostic stage wall time (no speed inference)",
            "unit": "seconds",
            "values": [
                {
                    "label": stage["label"],
                    "value": execution["stage_intervals"][stage["id"]][1]
                    - execution["stage_intervals"][stage["id"]][0],
                }
                for stage in task["stages"]
            ],
        },
    ]
    artifact_paths = [
        "gaussians_init.ply",
        "gaussians.ply",
        *PREVIEWS,
        "gradient_directions.svg",
        "comparison.json",
        "residuals.json",
        "training_history.json",
        "gaussians.config.json",
        "execution.json",
        "environment.json",
        "run_receipt.json",
        "input_boundary_receipt.json",
        "resource_receipt.json",
        "source_snapshot/manifest.json",
    ]
    evidence = [
        {"label": label, "path": f"benchmarks/results/{task['task_id']}_{suffix}"}
        for label, suffix in [
            ("Result note", "RESULT.md"),
            ("Machine result", "RESULT.json"),
            ("Independent audit", "AUDIT.md"),
            ("Machine audit", "AUDIT.json"),
        ]
    ]
    metrics_payload = {
        "schema_version": 2,
        "report_template_version": 2,
        "task_id": task["task_id"],
        "summary": summary,
        "decision": decision,
        "claim_boundary": task["claim_boundary"],
        "metrics": comparison["primary_metrics"],
        "metric_metadata": metadata,
        "charts": charts,
        "artifacts": [{"label": p, "path": p} for p in artifact_paths],
        "evidence": evidence,
        "commands": commands(task),
        "notes": notes,
    }
    source_files = ["residuals.json", "execution.json"] + [
        f"states/{s['id']}/diagnostics.json" for s in task["states"]
    ]
    result = {
        "schema_version": 1,
        "task_id": task["task_id"],
        "summary": summary,
        "decision": decision,
        "claim_boundary": task["claim_boundary"],
        "source_lock": lock,
        "command": task["run_command"],
        "comparison": comparison,
        "residual_rows": residuals["rows"],
        "raw_diagnostics": states,
        "raw_file_hashes": {
            p: hashlib.sha256((run / p).read_bytes()).hexdigest() for p in source_files
        },
        "notes": notes,
    }
    stem = ROOT / "benchmarks/results" / task["task_id"]
    lines = [
        f"# {task['title']}",
        "",
        summary,
        "",
        decision,
        "",
        "| Saved state | Means cosine, mean over views | Relative gradient difference |",
        "|---|---:|---:|",
    ]
    for state in comparison["states"]:
        stats = state["per_view_statistics"]["total"]["means"]
        lines.append(
            f"| {state['state']['id']} | {display(stats['cosine']['mean'])} | "
            f"{display(stats['difference_over_photo_norm']['mean'])} |"
        )
    lines += [
        "",
        "| Region | Mean per-view MAE | Mean signed RGB bias | Mean absolute-error share |",
        "|---|---:|---:|---:|",
    ]
    for region, stats in comparison["regions"].items():
        lines.append(
            f"| {region} | {display(stats['mae_region']['mean'])} | "
            f"{display(stats['signed_mean_rgb']['mean'])} | "
            f"{display(stats['absolute_error_share_full_canvas']['mean'])} |"
        )
    lines += [
        "",
        task["claim_boundary"],
        "",
        *[f"- {note}" for note in notes],
        "",
        "Raw view rows, all six parameter groups, component statistics and source identities "
        "are retained in the companion RESULT JSON. Independent audit is separate.",
        "",
    ]
    path = Path(str(stem) + "_RESULT.md")
    text = "\n".join(lines)
    if path.exists() and path.read_text() != text:
        raise ValueError("refusing to replace conflicting result Markdown")
    json_path = Path(str(stem) + "_RESULT.json")
    if json_path.exists() and read(json_path) != result:
        raise ValueError("refusing to replace conflicting result JSON")
    # Validate both immutable records before any publication mutation.
    write(run / "comparison.json", comparison)
    write(run / "training_history.json", history(task, states, execution))
    write(
        run / "gaussians.config.json",
        {
            "protocol": task,
            "diagnostic_only": True,
            "optimizer_updates": 0,
            "model_role": "Existing RTGS-021 snapshots; no model fitted in this run.",
        },
    )
    heatmap(run, comparison)
    write(run / "metrics.json", metrics_payload)
    once(json_path, result)
    if not path.exists():
        path.write_text(text)
    write(
        run / "run_receipt.json",
        {
            "schema_version": 1,
            "task_id": task["task_id"],
            "status": "completed",
            "started_at_utc": lock["started_at_utc"],
            "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exit_code": 0,
            "failure_phase": None,
            "message": "Fixed-state diagnostics completed; no optimization.",
        },
    )
    return result


def publish_failure(task: dict, run: Path, error: str) -> None:
    """Make a truthful failure page while preserving raw observations and old results."""
    receipt_path = run / "run_receipt.json"
    if receipt_path.exists() and read(receipt_path).get("status") == "completed":
        raise ValueError("refusing to replace an existing completed receipt")
    lock = read(run / "task.lock.json")
    write(
        receipt_path,
        {
            "schema_version": 1,
            "task_id": task["task_id"],
            "status": "failed",
            "started_at_utc": lock["started_at_utc"],
            "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exit_code": 1,
            "failure_phase": "diagnostic_or_presentation",
            "message": error,
        },
    )
    for filename, payload in {
        "training_history.json": {
            "schema_version": 2,
            "records": [],
            "metric_metadata": {},
            "stage_markers": [],
        },
        "gaussians.config.json": {"protocol": task, "diagnostic_only": True},
        "input_boundary_receipt.json": {"status": "failed", "detail": error},
        "resource_receipt.json": {"status": "failed", "detail": error},
    }.items():
        if not (run / filename).exists():
            write(run / filename, payload)
    artifacts = [
        "training_history.json",
        "gaussians.config.json",
        "input_boundary_receipt.json",
        "resource_receipt.json",
        "run_receipt.json",
        "environment.json",
    ]
    failed_commands = commands(task)
    failed_commands["viewer"] = None
    write(
        run / "metrics.json",
        {
            "schema_version": 2,
            "report_template_version": 2,
            "task_id": task["task_id"],
            "summary": "Diagnostic did not produce a completed results-bearing bundle.",
            "decision": error,
            "claim_boundary": task["claim_boundary"],
            "metrics": {},
            "metric_metadata": {},
            "charts": [],
            "evidence": [],
            "artifacts": [{"label": name, "path": name} for name in artifacts],
            "commands": failed_commands,
            "notes": [
                "Raw observations and failure receipts remain in this run. No automatic retry, "
                "scientific success claim, or fabricated missing metric is permitted. "
                "Any partial diagnostic rows remain in the raw files; no fitting history exists."
            ],
        },
    )
