"""Guard coverage and undefined-statistic handling in the fixed-state report."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


def report_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/experiments"
        / "20260909_field_target_gradient_stage_frame00008_report.py"
    )
    spec = importlib.util.spec_from_file_location("gradient_report_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    module = report_module()
    views = ["C0004", "C0005"]
    states = [
        {"id": f"{kind}_{seed}", "kind": kind, "seed": seed}
        for seed in [1, 2, 3]
        for kind in ["initial", "rgb_final", "field_high_final"]
    ]
    task = {
        "states": states,
        "gradient_protocol": {"view_order": views, "state_order": [s["id"] for s in states]},
    }
    region = {
        key: 0.1
        for key in (
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
    }
    residuals = {
        "rows": [
            {"view_id": v, "regions": {r: dict(region) for r in module.REGIONS}} for v in views
        ]
    }
    diagnostics = []
    for state in states:
        stats = {
            component: {group: {key: 0.5 for key in module.STATS} for group in module.GROUPS}
            for component in module.COMPONENTS
        }
        diagnostics.append(
            {
                "state": state,
                "status": "completed",
                "saved_state_unchanged": True,
                "identity_control": {
                    "state_id": state["id"],
                    "view_id": views[0],
                    "passed": True,
                    "gradient_atol": 1e-8,
                    "gradient_rtol": 1e-4,
                    "loss_atol": 1e-7,
                    "independent_render_graphs": True,
                    "losses": {
                        c: {"photo_path": 0.2, "teacher_path": 0.2, "absolute_difference": 0.0}
                        for c in module.COMPONENTS
                    },
                    "comparisons": {
                        c: {
                            g: {
                                "photo_norm": 0.5,
                                "field_norm": 0.5,
                                "difference_norm": 0.0,
                                "difference_over_photo_norm": 0.0,
                                "cosine": 1.0,
                                "max_absolute_difference": 0.0,
                                "element_count": 1,
                                "tolerance_violations": 0,
                                "max_tolerance_excess": 0.0,
                            }
                            for g in module.GROUPS
                        }
                        for c in module.COMPONENTS
                    },
                },
                "aggregate": copy.deepcopy(stats),
                "rows": [
                    {
                        "view_id": v,
                        "comparisons": copy.deepcopy(stats),
                        "losses": {
                            target: {c: 0.2 for c in module.COMPONENTS}
                            for target in ["rgb", "field_high"]
                        },
                    }
                    for v in views
                ],
            }
        )
    return module, task, residuals, diagnostics


def test_undefined_groups_stay_undefined_and_do_not_become_agreement():
    module, task, residuals, states = fixture()
    for row in states[0]["rows"]:
        zero_stats(row["comparisons"]["total"]["shN"])
    result = module.summarize(task, residuals, states)
    stats = result["states"][0]["per_view_statistics"]["total"]["shN"]["cosine"]
    assert stats == {
        "mean": None,
        "min": None,
        "max": None,
        "defined_count": 0,
        "undefined_count": 2,
    }
    assert result["primary_metrics"]["means_total_cosine"] == 0.5


@pytest.mark.parametrize("failure", ["duplicate_view", "missing_state", "metadata", "control"])
def test_incomplete_or_invalid_diagnostics_cannot_publish(failure):
    module, task, residuals, states = fixture()
    if failure == "duplicate_view":
        states[0]["rows"][1]["view_id"] = "C0004"
    elif failure == "missing_state":
        states.pop()
    elif failure == "metadata":
        states[0]["state"] = {**states[0]["state"], "seed": 999}
    else:
        states[0]["identity_control"]["passed"] = False
    with pytest.raises(ValueError):
        module.summarize(task, residuals, states)


def test_primary_averages_defined_views_then_seeds_then_state_kinds():
    module, task, residuals, states = fixture()
    states[0]["rows"][0]["comparisons"]["total"]["means"]["cosine"] = 1.0
    zero_stats(states[0]["rows"][1]["comparisons"]["total"]["means"])
    result = module.summarize(task, residuals, states)
    assert result["primary_metrics"]["means_total_cosine"] == pytest.approx((1 + 8 * 0.5) / 9)
    with pytest.raises(ValueError, match="nonfinite"):
        module.nullable_summary([float("nan")])


def zero_stats(stats):
    stats.update(
        photo_norm=0.0,
        field_norm=0.0,
        difference_norm=0.0,
        max_absolute_difference=0.0,
        cosine=None,
        difference_over_photo_norm=None,
    )


@pytest.mark.parametrize("failure", ["aggregate", "component", "group", "nonfinite", "false_flag"])
def test_incomplete_control_or_aggregate_is_rejected(failure):
    module, task, residuals, states = fixture()
    state = states[0]
    if failure == "aggregate":
        state["aggregate"] = {}
    elif failure == "component":
        state["identity_control"]["comparisons"].pop("dssim")
    elif failure == "group":
        state["identity_control"]["comparisons"]["l1"].pop("quats")
    elif failure == "nonfinite":
        state["aggregate"]["total"]["means"]["difference_norm"] = float("nan")
    else:
        state["identity_control"]["comparisons"]["l1"]["means"]["tolerance_violations"] = 1
    with pytest.raises(ValueError):
        module.summarize(task, residuals, states)


def publication_fixture(tmp_path, monkeypatch):
    module, minimal, residuals, states = fixture()
    root = Path(__file__).resolve().parents[1]
    task = json.loads(
        (
            root / "experiments/tasks/20260909_field_target_gradient_stage_frame00008.json"
        ).read_text()
    )
    task.update(minimal)
    task["seeds"] = [1, 2, 3]
    for state in states:
        for i, row in enumerate(state["rows"], 1):
            row.update(step=i, wall_seconds=2.0 + i)
    run = tmp_path / "runs" / task["task_id"]
    run.mkdir(parents=True)
    module.write(run / "residuals.json", residuals)
    for state in states:
        module.write(run / "states" / state["state"]["id"] / "diagnostics.json", state)
    execution = {
        "status": "completed",
        "states": task["gradient_protocol"]["state_order"],
        "optimization_steps": 0,
        "topology_updates": 0,
        "stage_intervals": {
            "residual": [0.1, 1.0],
            "gradient": [2.0, 30.0],
            "presentation": [31.0, 32.0],
        },
        "resource": {"torch_cuda_max_memory_allocated": 1, "torch_cuda_max_memory_reserved": 2},
    }
    module.write(run / "execution.json", execution)
    lock = {
        "task_id": task["task_id"],
        "command": task["run_command"],
        "started_at_utc": "2026-09-09T00:00:00+00:00",
    }
    module.write(run / "task.lock.json", lock)
    (tmp_path / "benchmarks/results").mkdir(parents=True)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module, task, run, residuals, states, lock


def test_full_publish_handles_zero_gradient_state_and_zero_error_shares(tmp_path, monkeypatch):
    from scripts import experiment_contract as contract

    module, task, run, residuals, states, lock = publication_fixture(tmp_path, monkeypatch)
    for row in states[0]["rows"]:
        for component in module.COMPONENTS:
            for group in module.GROUPS:
                zero_stats(row["comparisons"][component][group])
    for row in residuals["rows"]:
        for region in row["regions"].values():
            region["absolute_error_share_full_canvas"] = None
            for key in ("mae_region", "mse_region", "signed_mean_rgb", "absolute_error_sum"):
                region[key] = 0.0
    module.write(run / "residuals.json", residuals)
    module.write(run / "states" / states[0]["state"]["id"] / "diagnostics.json", states[0])
    module.publish(task, run)
    result = (tmp_path / f"benchmarks/results/{task['task_id']}_RESULT.md").read_text()
    assert "undefined" in result
    metrics = module.read(run / "metrics.json")
    assert len(metrics["charts"][0]["values"]) == 8
    assert contract._metric_errors_v2(metrics, task, lock, completed=True) == []
    assert (
        contract._history_errors(module.read(run / "training_history.json"), task, completed=True)
        == []
    )


@pytest.mark.parametrize("suffix", ["json", "md"])
def test_result_conflict_is_detected_before_publication(tmp_path, monkeypatch, suffix):
    module, task, run, _residuals, _states, _lock = publication_fixture(tmp_path, monkeypatch)
    conflict = tmp_path / f"benchmarks/results/{task['task_id']}_RESULT.{suffix}"
    conflict.write_text("{}" if suffix == "json" else "preserved evidence")
    before = {
        str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    }
    with pytest.raises(ValueError, match="conflicting"):
        module.publish(task, run)
    after = {
        str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    }
    assert after == before


def test_all_undefined_primary_is_preserved_as_failed_presentation(tmp_path, monkeypatch):
    from scripts import experiment_contract as contract

    module, task, run, _residuals, states, lock = publication_fixture(tmp_path, monkeypatch)
    for state in states:
        for row in state["rows"]:
            zero_stats(row["comparisons"]["total"]["means"])
        module.write(run / "states" / state["state"]["id"] / "diagnostics.json", state)
    with pytest.raises(ValueError, match="primary metric is undefined"):
        module.publish(task, run)
    assert not (tmp_path / f"benchmarks/results/{task['task_id']}_RESULT.json").exists()
    module.publish_failure(task, run, "primary metric is undefined")
    assert module.read(run / "run_receipt.json")["status"] == "failed"
    assert (
        contract._metric_errors_v2(module.read(run / "metrics.json"), task, lock, completed=False)
        == []
    )
    assert (
        module.read(run / "states" / states[0]["state"]["id"] / "diagnostics.json")["rows"][0][
            "comparisons"
        ]["total"]["means"]["cosine"]
        is None
    )
