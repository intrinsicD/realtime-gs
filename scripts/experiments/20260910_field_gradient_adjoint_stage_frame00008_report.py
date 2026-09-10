"""Task-owned diagnostic report sources; the shared renderer owns the final page.

This module consumes completed observations. It never fits or selects a model.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import itertools
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
    "direct_delta_norm",
    "direct_delta_max_absolute",
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


def finite_nonnegative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def validate_zero_stats(stats: dict) -> None:
    count = stats.get("element_count")
    if not isinstance(count, int) or count < 1:
        raise ValueError("control element count is invalid")
    if stats.get("tolerance_violations") != 0:
        raise ValueError("control has tolerance violations")
    excess = stats.get("max_tolerance_excess")
    if not isinstance(excess, (int, float)) or not math.isfinite(excess) or excess > 0:
        raise ValueError("control tolerance excess is invalid")


def validate_control(control: dict, state: dict, view: str) -> None:
    if control.get("state_id") != state["id"] or control.get("view_id") != view:
        raise ValueError("identity control state/view mismatch")
    if control.get("common_render") is not True:
        raise ValueError("identity control lacks a common render")
    expected = {
        "parameter_delta_atol": 1e-8,
        "image_atol": 1e-8,
        "image_rtol": 1e-4,
        "loss_atol": 1e-7,
    }
    if any(control.get(key) != value for key, value in expected.items()):
        raise ValueError("identity control tolerances changed")
    for key in ("losses", "adjoints", "comparisons", "null_vjp"):
        if set(control[key]) != set(COMPONENTS):
            raise ValueError("identity component coverage incomplete")
    for component in COMPONENTS:
        loss = control["losses"][component]
        for key in ("photo_path", "teacher_path", "absolute_difference"):
            value = loss[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("nonfinite identity loss")
        difference = abs(loss["photo_path"] - loss["teacher_path"])
        if abs(difference - loss["absolute_difference"]) > 1e-12 or difference > 1e-7:
            raise ValueError("identity loss difference failed")
        validate_zero_stats(control["adjoints"][component])
        for category in ("comparisons", "null_vjp"):
            if set(control[category][component]) != set(GROUPS):
                raise ValueError("identity group coverage incomplete")
            for group in GROUPS:
                stats = control[category][component][group]
                validate_zero_stats(stats)
                if not finite_nonnegative(stats["max_absolute_difference"]) or (
                    stats["max_absolute_difference"] > 1e-8
                ):
                    raise ValueError("zero-effect propagated parameter delta failed")
    if control.get("passed") is not True:
        raise ValueError("failed identity control cannot produce a completed result")


RELIABILITY_FLAGS = (
    "effect_resolved_against_observed_repeats",
    "reference_resolved",
    "field_resolved",
)


def validate_reliability(reliability: dict, comparisons: dict) -> None:
    if set(reliability) != set(COMPONENTS):
        raise ValueError("repeatability component coverage incomplete")
    for component in COMPONENTS:
        if set(reliability[component]) != set(GROUPS):
            raise ValueError("repeatability group coverage incomplete")
        for group in GROUPS:
            record, stats = reliability[component][group], comparisons[component][group]
            p = record["photo_repeat_difference_norm"]
            d = record["delta_repeat_difference_norm"]
            noise = record["noise_proxy"]
            if not all(finite_nonnegative(v) for v in (p, d, noise)):
                raise ValueError("invalid observed-repeat noise")
            if not math.isclose(noise, p + d, rel_tol=1e-12, abs_tol=1e-18):
                raise ValueError("observed-repeat noise proxy is not the declared sum")
            flags = (
                stats["direct_delta_norm"] > 10 * noise,
                stats["photo_norm"] > 10 * p,
                stats["field_norm"] > 10 * noise,
            )
            if any(record[key] is not value for key, value in zip(RELIABILITY_FLAGS, flags)):
                raise ValueError("observed-repeat precision flag disagrees with frozen rule")


def precision_counts(rows: list, component: str, group: str) -> dict:
    records = [r["reliability"][component][group] for r in rows]
    output = {
        key: {
            "resolved_count": sum(r[key] for r in records),
            "unresolved_count": sum(not r[key] for r in records),
        }
        for key in RELIABILITY_FLAGS
    }
    joint = sum(all(r[k] for k in RELIABILITY_FLAGS) for r in records)
    output["all_three"] = {"resolved_count": joint, "unresolved_count": len(rows) - joint}
    return output


def validate_numerical_control(control: dict, task: dict) -> None:
    probe = task["gradient_protocol"]["repeatability_probe"]
    if (
        control.get("passed") is not True
        or control.get("state_id") != probe["state"]
        or (control.get("view_id") != probe["view"])
    ):
        raise ValueError("numerical control failed or used the wrong state/view")
    if (
        control.get("saved_state_unchanged") is not True
        or control.get("photo_unchanged") is not True
    ):
        raise ValueError("numerical control inputs changed")
    fixed_norms = control["fixed_vjp"]["fixed_adjoint_norms"]
    if set(fixed_norms) != set(COMPONENTS) or any(
        not finite_nonnegative(v) or v == 0 for v in fixed_norms.values()
    ):
        raise ValueError("fixed nonzero-adjoint probe has invalid adjoints")
    for kind in ("whole_chain", "fixed_image", "fixed_vjp"):
        part = control[kind]
        if len(part["repeats"]) != 6 or len(part["pairwise"]) != 15:
            raise ValueError("numerical repeat coverage incomplete")
        expected_pairs = list(itertools.combinations(range(6), 2))
        if [(r["a"], r["b"]) for r in part["pairwise"]] != expected_pairs:
            raise ValueError("numerical pairwise order or coverage incomplete")
    for kind in ("whole_chain", "fixed_image"):
        for row in control[kind]["pairwise"]:
            if not finite_nonnegative(row["render_max_abs"]) or row["render_max_abs"] > 1e-7:
                raise ValueError("render repeat gate failed")
            if set(row["loss_differences"]) != set(COMPONENTS):
                raise ValueError("numerical loss coverage incomplete")
            if any(not finite_nonnegative(v) or v > 1e-7 for v in row["loss_differences"].values()):
                raise ValueError("loss repeat gate failed")
    for kind in ("whole_chain", "fixed_image"):
        for row in control[kind]["pairwise"]:
            if set(row["adjoints"]) != set(COMPONENTS):
                raise ValueError("numerical adjoint coverage incomplete")
            for stats in row["adjoints"].values():
                validate_zero_stats(stats)
    if len(control["zero_controls"]) != 6:
        raise ValueError("numerical zero-control coverage incomplete")
    for row in control["zero_controls"]:
        validate_control(row, {"id": probe["state"]}, probe["view"])


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
            validate_reliability(row["reliability"], row["comparisons"])
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
                "observed_repeat_precision": {
                    c: {g: precision_counts(state["rows"], c, g) for g in GROUPS}
                    for c in COMPONENTS
                },
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
                last = 0 if stage["id"] in {"numerical_control", "residual"} else 22
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
    width, height = 1070, 760
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" fill="#17212f">',
        '<text x="28" y="35" font-size="22">Target gradients at the same saved state</text>',
        '<text x="28" y="61" font-size="14">Total loss; mean of defined per-view cosines. '
        "1 = aligned, 0 = orthogonal, −1 = opposed.</text>",
        '<text x="28" y="82" font-size="13">Each cell includes resolved/total views under '
        "all three observed-repeat flags, plus defined cosine counts; "
        "no accuracy guarantee.</text>",
    ]
    for col, group in enumerate(GROUPS):
        parts.append(f'<text x="{285 + 120 * col}" y="108" font-size="15">{group}</text>')
    for index, state in enumerate(summary["states"]):
        y = 126 + index * 58
        label = state["state"]["id"]
        parts.append(f'<text x="28" y="{y + 29}" font-size="14">{html.escape(label)}</text>')
        for col, group in enumerate(GROUPS):
            cosine = state["per_view_statistics"]["total"][group]["cosine"]
            value = cosine["mean"]
            if value is None:
                colour, label = "#e8e8e8", "undefined"
            else:
                intensity = min(1.0, abs(value))
                pale = round(245 - 150 * intensity)
                colour = f"rgb({pale},{pale + 5},245)" if value >= 0 else f"rgb(245,{pale},{pale})"
                label = f"{value:.3f}"
            counts = state["observed_repeat_precision"]["total"][group]["all_three"]
            total = counts["resolved_count"] + counts["unresolved_count"]
            precision = f"{counts['resolved_count']}/{total} resolved"
            defined = f"{cosine['defined_count']}/{total} defined"
            x = 272 + col * 120
            parts.extend(
                [
                    f'<rect x="{x}" y="{y}" width="111" height="53" rx="4" fill="{colour}"/>',
                    f'<text x="{x + 10}" y="{y + 19}" font-size="14">{label}</text>',
                    f'<text x="{x + 10}" y="{y + 34}" font-size="11">{precision}</text>',
                    f'<text x="{x + 10}" y="{y + 48}" font-size="11">{defined}</text>',
                ]
            )
    parts.extend(
        [
            '<text x="28" y="686" font-size="14">Quats = saved raw quaternion coordinates; '
            "scales = log scales; opacities = local logit derivative.</text>",
            '<text x="28" y="710" font-size="14">No optimization or cross-topology vector '
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


def verify_array_evidence(run: Path, numerical: dict, states: list[dict]) -> None:
    def verify(path_text: str, expected_hash: str, size: int | None = None) -> None:
        path = (run / path_text).resolve()
        if not path.is_relative_to(run.resolve()) or not path.is_file():
            raise ValueError("raw array evidence is missing or outside the run")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("raw array evidence checksum mismatch")
        if size is not None and path.stat().st_size != size:
            raise ValueError("raw array evidence byte count mismatch")

    def walk(value: object) -> None:
        if isinstance(value, dict):
            if "array_path" in value:
                verify(value["array_path"], value["sha256"], value["bytes"])
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for kind in ("whole_chain", "fixed_image", "fixed_vjp"):
        for repeat in numerical[kind]["repeats"]:
            if not {"array_path", "sha256", "bytes"} <= set(repeat):
                raise ValueError("numerical repeat lacks raw array evidence")
    if not {"array_path", "sha256", "bytes"} <= set(numerical["fixed_vjp"]["fixed_inputs"]):
        raise ValueError("fixed VJP inputs lack raw array evidence")
    controls = numerical["zero_controls"] + [s["identity_control"] for s in states]
    for control in controls:
        if not {"array_path", "sha256", "bytes"} <= set(control):
            raise ValueError("identity control lacks raw array evidence")
    for state in states:
        for row in state["rows"]:
            if not {"array_path", "sha256", "bytes"} <= set(row):
                raise ValueError("per-view repeats lack raw array evidence")
        verify(f"states/{state['state']['id']}/mean_gradients.npz", state["mean_gradients_sha256"])
    walk(numerical)
    walk(states)


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
    numerical = read(run / "numerical_control.json")
    validate_numerical_control(numerical, task)
    verify_array_evidence(run, numerical, states)
    comparison = summarize(task, residuals, states)
    comparison["numerical_control"] = numerical
    lock = read(run / "task.lock.json")
    summary = (
        "Completed training-only target residuals and common-state gradient measurements on "
        "all 22 cameras at nine predeclared saved states. The new photo-only numerical gate and "
        "all nine common-render zero-effect controls passed. "
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
        "Common-render gradients use g_field = g_photo + J^T(h_field - h_photo), with two "
        "nonzero VJP repeats per site. Float64 means derive from saved float32 component arrays.",
        "Zero-effect controls cover C0004 per state and test null/routing behavior only. "
        "They do not certify nonzero differences. The predecessor remains failed; ordinary "
        "parameter repeat differences are descriptive and never count as passing its gate.",
        "Resolved/total counts beside cosines use all three frozen observed-repeat flags. "
        "Two repeats and the ten-times rule provide sampled precision context, not a confidence "
        "interval or global accuracy bound. Weak effects and angular comparisons remain "
        "unresolved. "
        "Mean-gradient arrays have no independent aggregate precision certificate.",
        "Shared-GPU resources cover measurement/previews through receipts, before publication; "
        "administrative raw-seal checks and report generation are excluded. No speed claim.",
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
                    "label": (
                        f"{s['state']['id']} "
                        f"({s['observed_repeat_precision']['total']['means']['all_three']['resolved_count']}"
                        f"/{len(task['gradient_protocol']['view_order'])} resolved; "
                        f"{s['per_view_statistics']['total']['means']['cosine']['defined_count']}"
                        f"/{len(task['gradient_protocol']['view_order'])} defined)"
                    ),
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
        "numerical_control.json",
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
    source_files = ["residuals.json", "execution.json", "numerical_control.json"] + [
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
        "| Saved state | Means cosine, mean over views | Relative gradient difference | "
        "Resolved/total views | Defined/total cosines |",
        "|---|---:|---:|---:|---:|",
    ]
    for state in comparison["states"]:
        stats = state["per_view_statistics"]["total"]["means"]
        lines.append(
            f"| {state['state']['id']} | {display(stats['cosine']['mean'])} | "
            f"{display(stats['difference_over_photo_norm']['mean'])} | "
            f"{state['observed_repeat_precision']['total']['means']['all_three']['resolved_count']}"
            f"/{len(task['gradient_protocol']['view_order'])} | "
            f"{stats['cosine']['defined_count']}/{len(task['gradient_protocol']['view_order'])} |"
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
