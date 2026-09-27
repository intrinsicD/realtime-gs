"""CPU contract tests for the RTGS-026 relocation driver and its frozen decision rule."""

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
TASK_ID = "20260927_silhouette_relocation_stage_frame00008"
DRIVER = ROOT / "scripts/experiments" / f"{TASK_ID}.py"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _module(DRIVER, "relocation_driver")


@pytest.fixture(scope="module")
def report():
    return _module(DRIVER.with_name(f"{TASK_ID}_report.py"), "relocation_report_test")


@pytest.fixture(scope="module")
def task():
    return json.loads((ROOT / "experiments/tasks" / f"{TASK_ID}.json").read_text())


def test_tables_split_and_hull_views_match_the_protocol(driver, report, task) -> None:
    driver.check_protocol_tables(task)
    assert tuple(driver.CONDITIONS) == report.CONDITIONS
    assert [m["id"] for m in task["primary_metrics"]] == list(driver.METRICS)
    split = task["splits"][driver.DATASET]
    assert len(split["train"]) == 24 and split["heldout"] == ["C0001", "C0029"]
    assert sorted(task["hull"]["mask_views"]) == sorted(split["train"] + split["heldout"])
    assert list(report.SELECTED) == [
        task["preview_policy"]["selected_model"].split()[0],
        task["seeds"][0],
    ]
    broken = copy.deepcopy(task)
    broken["hull"]["mask_views"] = split["train"]
    with pytest.raises(RuntimeError, match="every calibrated view"):
        driver.check_protocol_tables(broken)
    overlap = copy.deepcopy(task)
    overlap["splits"][driver.DATASET]["train"].append("C0001")
    with pytest.raises(RuntimeError, match="overlap"):
        driver.check_protocol_tables(overlap)


def test_resolved_configs_round_trip_and_differ_only_in_seed(driver, task) -> None:
    configs = [driver.train_config(task, seed) for seed in task["seeds"]]
    assert all(c.use_masks and c.random_background and c.iterations == 8000 for c in configs)
    first, second = (task["resolved_training_configs"][str(s)] for s in task["seeds"][:2])
    assert {k for k in first if first[k] != second[k]} == {"seed"}


def _cells(on: dict, seeds=(1, 2)) -> list[dict]:
    base = {
        "foreground_psnr": 24.0,
        "crop_lpips": 0.10,
        "outside_alpha_mass": 0.001,
        "floater_fraction": 0.001,
        "interior_alpha": 0.99,
        "hull_rejected_fraction": 0.01,
    }
    cells = []
    for seed in seeds:
        for condition in ("nb_ms", "nb_ms_reloc", "ph_ms", "ph_ms_reloc"):
            values = {**base, **(on if condition == "nb_ms_reloc" else {})}
            cells.append({"condition": condition, "seed": seed, "evaluation": {"mean": values}})
    return cells


def test_relative_gate_is_inclusive_per_seed(report) -> None:
    task = {"seeds": [1, 2]}
    passing = report.gates(task, _cells({"foreground_psnr": 24.125, "crop_lpips": 0.105}))
    assert passing["h1_relocation"]["verdict"] == "pass"
    assert "production model" in report.decision_text(passing)
    lpips_fail = report.gates(task, _cells({"foreground_psnr": 24.5, "crop_lpips": 0.1051}))
    assert lpips_fail["h1_relocation"]["verdict"] == "inconclusive"
    small = report.gates(task, _cells({"foreground_psnr": 24.0625}))
    assert small["h1_relocation"]["verdict"] == "inconclusive"
    worse = report.gates(task, _cells({"foreground_psnr": 23.875}))
    assert worse["h1_relocation"]["verdict"] == "reject"
    assert "no production model" in report.decision_text(worse)


def test_production_requires_a_passing_gate(driver, task, tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "comparison.json").write_text(
        json.dumps({"gates": {"h1_relocation": {"verdict": "inconclusive"}}})
    )
    with pytest.raises(RuntimeError, match="passing frozen relocation gate"):
        driver.production(task, run)


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
