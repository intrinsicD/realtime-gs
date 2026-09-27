"""CPU contract tests for the RTGS-028 budget x opacity-reset driver and decision rules."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "20260927_color_budget_reset_stage_frame00008"
DRIVER = ROOT / "scripts/experiments" / f"{TASK_ID}.py"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _module(DRIVER, "color_budget_driver")


@pytest.fixture(scope="module")
def report():
    return _module(DRIVER.with_name(f"{TASK_ID}_report.py"), "color_budget_report_test")


@pytest.fixture(scope="module")
def task():
    return json.loads((ROOT / "experiments/tasks" / f"{TASK_ID}.json").read_text())


def test_tables_and_split_match_the_protocol(driver, report, task) -> None:
    driver.check_protocol_tables(task)
    assert tuple(driver.CONDITIONS) == report.CONDITIONS
    assert [m["id"] for m in task["primary_metrics"]] == list(driver.METRICS)
    split = task["splits"][driver.DATASET]
    assert len(split["train"]) == 22
    assert split["heldout"] == ["C0001", "C0018", "C0029", "C1002"]
    assert list(report.SELECTED) == [
        task["preview_policy"]["selected_model"].split()[0],
        task["seeds"][0],
    ]
    broken = copy.deepcopy(task)
    broken["resolved_training_configs"].pop("30k")
    with pytest.raises(RuntimeError, match="frozen budgets"):
        driver.check_protocol_tables(broken)


def test_budget_configs_round_trip_and_differ_only_where_frozen(driver, report, task) -> None:
    for seed in task["seeds"]:
        short = driver.train_config(task, "8k", seed)
        long = driver.train_config(task, "30k", seed)
        assert (short.iterations, short.density.stop_iter) == (8000, 6000)
        assert (long.iterations, long.density.stop_iter) == (30000, 15000)
        assert short.density.opacity_reset_every == long.density.opacity_reset_every == 3000
        a = task["resolved_training_configs"]["8k"][str(seed)]
        b = task["resolved_training_configs"]["30k"][str(seed)]
        assert {k for k in a if a[k] != b[k]} == {"iterations", "density"}
        assert {k for k in a["density"] if a["density"][k] != b["density"][k]} == {"stop_iter"}
    assert report.condition_iterations("nb_30k_rs") == 30000
    assert report.condition_iterations("ph_8k_up") == 8000


def test_intended_reset_schedule_per_budget(driver, task) -> None:
    from rtgs.optim.strategies import IntendedOpacityReset

    for budget, expected in (("8k", [3000]), ("30k", [3000, 6000, 9000, 12000])):
        config = driver.train_config(task, budget, task["seeds"][0])
        reset = IntendedOpacityReset.from_density(config.density)
        fired = [s - 1 for s in range(1, config.iterations + 1) if reset.due(s)]
        assert fired == expected


def _cells(values: dict[str, dict], seeds=(1, 2)) -> list[dict]:
    base = {
        "foreground_psnr": 24.0,
        "crop_lpips": 0.10,
        "outside_alpha_mass": 0.001,
        "floater_fraction": 0.001,
        "interior_alpha": 0.99,
    }
    return [
        {
            "condition": condition,
            "seed": seed,
            "evaluation": {"mean": {**base, **values.get(condition, {})}},
        }
        for seed in seeds
        for condition in ("nb_8k_up", "nb_8k_rs", "nb_30k_up", "nb_30k_rs", "ph_8k_up", "ph_30k_rs")
    ]


def test_frozen_rules_are_inclusive_per_seed(report) -> None:
    task = {"seeds": [1, 2]}
    both = report.gates(
        task,
        _cells(
            {
                "nb_30k_up": {"foreground_psnr": 24.25, "crop_lpips": 0.105},
                "nb_30k_rs": {
                    "foreground_psnr": 24.375,
                    "crop_lpips": 0.11,
                    "outside_alpha_mass": 0.006,
                },
            }
        ),
    )
    assert both["h1_budget"]["verdict"] == "pass"
    assert both["h2_reset"]["verdict"] == "pass"
    floaters = report.gates(
        task,
        _cells(
            {
                "nb_30k_up": {"foreground_psnr": 24.25},
                "nb_30k_rs": {"foreground_psnr": 24.5, "outside_alpha_mass": 0.0061},
            }
        ),
    )
    assert floaters["h2_reset"]["verdict"] == "inconclusive"
    worse = report.gates(task, _cells({"nb_30k_up": {"foreground_psnr": 23.75}}))
    assert worse["h1_budget"]["verdict"] == "reject"
    assert "H1 longer budget" in report.decision_text(worse)


def test_selftest_denies_forbidden_worker_opens() -> None:
    completed = subprocess.run(
        [sys.executable, str(DRIVER), "selftest"],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["field_worker_input_guard"] == "4 forbidden open probes denied"


def test_coordinator_refuses_a_consumed_run_root(driver, task, tmp_path) -> None:
    run = tmp_path / "run"
    (run / "targets").mkdir(parents=True)
    before = sorted(p.relative_to(run) for p in run.rglob("*"))
    with pytest.raises(RuntimeError, match="refusing re-entry"):
        driver.coordinate(tmp_path / "task.json", task, run)
    assert sorted(p.relative_to(run) for p in run.rglob("*")) == before
