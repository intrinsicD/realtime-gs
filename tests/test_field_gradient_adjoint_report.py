"""Guard coverage and undefined-statistic handling in the fixed-state report."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import pytest


def report_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/experiments"
        / "20260910_field_gradient_adjoint_stage_frame00008_report.py"
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
                    "parameter_delta_atol": 1e-8,
                    "image_atol": 1e-8,
                    "image_rtol": 1e-4,
                    "loss_atol": 1e-7,
                    "common_render": True,
                    "adjoints": {
                        c: {
                            "element_count": 1,
                            "tolerance_violations": 0,
                            "max_tolerance_excess": 0.0,
                        }
                        for c in module.COMPONENTS
                    },
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
                        "reliability": {
                            c: {
                                g: {
                                    "photo_repeat_difference_norm": 0.01,
                                    "delta_repeat_difference_norm": 0.01,
                                    "noise_proxy": 0.02,
                                    "effect_resolved_against_observed_repeats": True,
                                    "reference_resolved": True,
                                    "field_resolved": True,
                                }
                                for g in module.GROUPS
                            }
                            for c in module.COMPONENTS
                        },
                        "losses": {
                            target: {c: 0.2 for c in module.COMPONENTS}
                            for target in ["rgb", "field_high"]
                        },
                    }
                    for v in views
                ],
            }
        )
    for state in diagnostics:
        state["identity_control"]["null_vjp"] = copy.deepcopy(
            state["identity_control"]["comparisons"]
        )
    return module, task, residuals, diagnostics


def test_undefined_groups_stay_undefined_and_do_not_become_agreement():
    module, task, residuals, states = fixture()
    for row in states[0]["rows"]:
        zero_stats(row["comparisons"]["total"]["shN"])
    refresh_flags(module, states)
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
    refresh_flags(module, states)
    result = module.summarize(task, residuals, states)
    assert result["primary_metrics"]["means_total_cosine"] == pytest.approx((1 + 8 * 0.5) / 9)
    with pytest.raises(ValueError, match="nonfinite"):
        module.nullable_summary([float("nan")])


def zero_stats(stats):
    stats.update(
        photo_norm=0.0,
        field_norm=0.0,
        difference_norm=0.0,
        direct_delta_norm=0.0,
        direct_delta_max_absolute=0.0,
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
            root / "experiments/tasks/20260910_field_gradient_adjoint_stage_frame00008.json"
        ).read_text()
    )
    minimal["gradient_protocol"]["repeatability_probe"] = task["gradient_protocol"][
        "repeatability_probe"
    ]
    task.update(minimal)
    task["seeds"] = [1, 2, 3]
    for state in states:
        for i, row in enumerate(state["rows"], 1):
            row.update(step=i, wall_seconds=2.0 + i)
    run = tmp_path / "runs" / task["task_id"]
    run.mkdir(parents=True)
    module.write(run / "residuals.json", residuals)
    placeholder = run / "test_array.bin"
    placeholder.write_bytes(b"synthetic report fixture only")
    receipt = {
        "array_path": "test_array.bin",
        "bytes": placeholder.stat().st_size,
        "sha256": hashlib.sha256(placeholder.read_bytes()).hexdigest(),
    }
    for state in states:
        state["identity_control"].update(receipt)
        for row in state["rows"]:
            row.update(receipt)
        path = run / "states" / state["state"]["id"] / "mean_gradients.npz"
        path.parent.mkdir(parents=True)
        path.write_bytes(placeholder.read_bytes())
        state["mean_gradients_sha256"] = receipt["sha256"]
    for state in states:
        module.write(run / "states" / state["state"]["id"] / "diagnostics.json", state)
    execution = {
        "status": "completed",
        "states": task["gradient_protocol"]["state_order"],
        "optimization_steps": 0,
        "topology_updates": 0,
        "stage_intervals": {
            "numerical_control": [0.0, 0.1],
            "residual": [0.1, 1.0],
            "gradient": [2.0, 30.0],
            "presentation": [31.0, 32.0],
        },
        "resource": {"torch_cuda_max_memory_allocated": 1, "torch_cuda_max_memory_reserved": 2},
    }
    module.write(run / "execution.json", execution)
    numerical = numerical_fixture(module, task, states)
    for kind in ("whole_chain", "fixed_image", "fixed_vjp"):
        for repeat in numerical[kind]["repeats"]:
            repeat.update(receipt)
    numerical["fixed_vjp"]["fixed_inputs"] = dict(receipt)
    numerical["fixed_vjp"]["fixed_adjoint_norms"] = {c: 1.0 for c in module.COMPONENTS}
    module.write(run / "numerical_control.json", numerical)
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
    refresh_flags(module, states)
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
        refresh_flags(module, states)
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


def refresh_flags(module, states):
    for state in states:
        for row in state["rows"]:
            for c in module.COMPONENTS:
                for g in module.GROUPS:
                    r, s = row["reliability"][c][g], row["comparisons"][c][g]
                    r.update(
                        effect_resolved_against_observed_repeats=s["direct_delta_norm"]
                        > 10 * r["noise_proxy"],
                        reference_resolved=s["photo_norm"] > 10 * r["photo_repeat_difference_norm"],
                        field_resolved=s["field_norm"] > 10 * r["noise_proxy"],
                    )


def numerical_fixture(module, task, states):
    probe = task["gradient_protocol"]["repeatability_probe"]
    zero = copy.deepcopy(states[0]["identity_control"])
    zero.update(state_id=probe["state"], view_id=probe["view"])
    pair = {
        "render_max_abs": 0.0,
        "loss_differences": {c: 0.0 for c in module.COMPONENTS},
        "adjoints": zero["adjoints"],
    }
    part = {
        "repeats": [{} for _ in range(6)],
        "pairwise": [
            {**copy.deepcopy(pair), "a": a, "b": b} for a, b in itertools.combinations(range(6), 2)
        ],
    }
    return {
        "state_id": probe["state"],
        "view_id": probe["view"],
        "passed": True,
        "saved_state_unchanged": True,
        "photo_unchanged": True,
        "whole_chain": copy.deepcopy(part),
        "fixed_image": copy.deepcopy(part),
        "fixed_vjp": copy.deepcopy(part),
        "zero_controls": [copy.deepcopy(zero) for _ in range(6)],
    }


@pytest.mark.parametrize("failure", ["noise", "flag", "missing", "negative"])
def test_repeat_precision_validation_rejects_fabricated_resolution(failure):
    module, task, residuals, states = fixture()
    r = states[0]["rows"][0]["reliability"]["total"]["means"]
    if failure == "noise":
        r["noise_proxy"] = 0.0
    elif failure == "flag":
        r["effect_resolved_against_observed_repeats"] = False
    elif failure == "missing":
        states[0]["rows"][0]["reliability"].pop("l1")
    else:
        r["photo_repeat_difference_norm"] = -1.0
    with pytest.raises(ValueError):
        module.summarize(task, residuals, states)


def test_unresolved_counts_stay_beside_defined_cosines():
    module, task, residuals, states = fixture()
    r = states[0]["rows"][0]["reliability"]["total"]["means"]
    r.update(photo_repeat_difference_norm=1.0, noise_proxy=1.01)
    refresh_flags(module, states)
    result = module.summarize(task, residuals, states)
    state = result["states"][0]
    assert state["per_view_statistics"]["total"]["means"]["cosine"]["defined_count"] == 2
    assert state["observed_repeat_precision"]["total"]["means"]["all_three"] == {
        "resolved_count": 1,
        "unresolved_count": 1,
    }


@pytest.mark.parametrize("failure", ["repeat", "image", "render", "zero", "flag"])
def test_numerical_failure_cannot_publish(tmp_path, monkeypatch, failure):
    module, task, run, _, _, _ = publication_fixture(tmp_path, monkeypatch)
    control = module.read(run / "numerical_control.json")
    if failure == "repeat":
        control["fixed_vjp"]["repeats"].pop()
    elif failure == "image":
        control["fixed_image"]["pairwise"][0]["adjoints"]["l1"]["tolerance_violations"] = 1
    elif failure == "render":
        control["whole_chain"]["pairwise"][0]["render_max_abs"] = 1e-5
    elif failure == "zero":
        control["zero_controls"][0]["comparisons"]["l1"]["means"]["max_absolute_difference"] = 1e-6
    else:
        control["passed"] = False
    module.write(run / "numerical_control.json", control)
    with pytest.raises(ValueError):
        module.publish(task, run)
    assert not (run / "metrics.json").exists()


def test_raw_repeat_corruption_prevents_publication(tmp_path, monkeypatch):
    module, task, run, _, _, _ = publication_fixture(tmp_path, monkeypatch)
    (run / "test_array.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        module.publish(task, run)
    assert not (run / "metrics.json").exists()


@pytest.mark.parametrize("failure", ["fixed_loss", "unchanged", "duplicate_pair", "null_vjp"])
def test_numerical_gate_checks_underlying_evidence_not_only_passed_flag(
    tmp_path, monkeypatch, failure
):
    module, task, run, _, _, _ = publication_fixture(tmp_path, monkeypatch)
    control = module.read(run / "numerical_control.json")
    if failure == "fixed_loss":
        control["fixed_image"]["pairwise"][0]["loss_differences"]["dssim"] = 1.0
    elif failure == "unchanged":
        control["saved_state_unchanged"] = False
    elif failure == "duplicate_pair":
        control["fixed_vjp"]["pairwise"][1].update(a=0, b=1)
    else:
        control["zero_controls"][0]["null_vjp"]["l1"]["means"]["max_absolute_difference"] = 1.0
    module.write(run / "numerical_control.json", control)
    with pytest.raises(ValueError):
        module.publish(task, run)
    assert not (run / "metrics.json").exists()
