"""CPU contract tests for the RTGS-029 field-only improvement portfolio."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "20260928_field_only_portfolio_stage_frame00008"
DRIVER = ROOT / "scripts/experiments" / f"{TASK_ID}.py"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _module(DRIVER, "portfolio_driver")


@pytest.fixture(scope="module")
def report():
    return _module(DRIVER.with_name(f"{TASK_ID}_report.py"), "portfolio_report_test")


@pytest.fixture(scope="module")
def task():
    return json.loads((ROOT / "experiments/tasks" / f"{TASK_ID}.json").read_text())


def test_tables_split_and_view_subsets_match_the_protocol(driver, report, task) -> None:
    driver.check_protocol_tables(task)
    assert tuple(driver.CONDITIONS) == report.CONDITIONS
    assert driver.TREATMENTS == report.TREATMENTS
    assert [m["id"] for m in task["primary_metrics"]] == list(driver.METRICS)
    split = task["splits"][driver.DATASET]
    assert len(split["train"]) == 22 and len(task["view_subsets"]["half"]) == 11
    assert set(task["view_subsets"]["half"]) < set(split["train"])
    assert list(report.SELECTED) == [
        task["preview_policy"]["selected_model"].split()[0],
        task["seeds"][0],
    ]
    broken = copy.deepcopy(task)
    broken["view_subsets"]["half"] = split["train"][1::2]
    with pytest.raises(RuntimeError, match="view subsets"):
        driver.check_protocol_tables(broken)


def test_each_config_changes_exactly_its_factor(driver, task) -> None:
    seed = str(task["seeds"][0])
    configs = task["resolved_training_configs"]
    base = configs["base"][seed]

    def changed(config_id):
        other = configs[config_id][seed]
        return {k for k in base if base[k] != other[k]}

    assert changed("sh1") == changed("sh0") == {"target_sh_degree"}
    assert changed("reg") == {"opacity_reg", "scale_reg"}
    assert changed("it30k_d6") == {"iterations"}
    assert changed("it30k_d6_lr8k") == {"iterations", "means_lr_final_factor"}
    assert configs["it30k_d6"][seed]["density"]["stop_iter"] == 6000
    for config_id in configs:
        for s in task["seeds"]:
            driver.train_config(task, config_id, s)
    lr = driver.train_config(task, "it30k_d6_lr8k", task["seeds"][0])
    gamma = lr.means_lr_final_factor ** (1 / lr.iterations)
    assert gamma**8000 == pytest.approx(0.01, rel=1e-9)
    assert report_iterations_consistent(driver, task)


def report_iterations_consistent(driver, task) -> bool:
    report = _module(DRIVER.with_name(f"{TASK_ID}_report.py"), "portfolio_report_iters")
    for condition, (_family, config_id, _ds, _views) in driver.CONDITIONS.items():
        iterations = task["resolved_training_configs"][config_id][str(task["seeds"][0])]
        if report.condition_iterations(condition) != iterations["iterations"]:
            return False
    return True


def test_box_render_matches_quadrature_average(driver) -> None:
    class Stub:
        def render(self, model, camera):
            class Out:
                pass

            out = Out()
            out.color = torch.arange(4 * 6 * 3, dtype=torch.float32).reshape(4, 6, 3).cuda()
            out.alpha = torch.ones(4, 6, 1).cuda()
            return out

    if not torch.cuda.is_available():
        pytest.skip("box_render moves the camera to cuda:0")
    from rtgs.core.camera import Camera

    camera = Camera(4.0, 4.0, 3.0, 2.0, 6, 4, torch.eye(3), torch.zeros(3))
    color, alpha = driver.box_render(Stub(), None, camera)
    expected = torch.nn.functional.avg_pool2d(
        torch.arange(72, dtype=torch.float32).reshape(4, 6, 3).permute(2, 0, 1)[None], 2
    )[0].permute(1, 2, 0)
    assert torch.equal(color, expected) and alpha.shape == (2, 3)


def _cells(values: dict[str, dict], seeds=(1, 2)) -> list[dict]:
    base = {
        "foreground_psnr": 24.0,
        "crop_lpips": 0.10,
        "outside_alpha_mass": 0.001,
        "floater_fraction": 0.001,
        "interior_alpha": 0.99,
    }
    conditions = (
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
    return [
        {"condition": c, "seed": s, "evaluation": {"mean": {**base, **values.get(c, {})}}}
        for s in seeds
        for c in conditions
    ]


def test_per_arm_rule_is_inclusive_and_independent(report) -> None:
    task = {"seeds": [1, 2]}
    result = report.gates(
        task,
        _cells(
            {
                "nb_sh1": {"foreground_psnr": 24.125, "crop_lpips": 0.105},
                "nb_reg": {"foreground_psnr": 24.5, "outside_alpha_mass": 0.0061},
                "nb_ds4": {"foreground_psnr": 23.875},
            }
        ),
    )
    assert result["nb_sh1"]["verdict"] == "pass"
    assert result["nb_reg"]["verdict"] == "inconclusive"
    assert result["nb_ds4"]["verdict"] == "reject"
    assert result["nb_sh0"]["verdict"] == "inconclusive"
    assert "nb_sh1: pass" in report.decision_text(result)


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
    assert result["field_worker_input_guard"] == "5 forbidden open probes denied"


def test_coordinator_refuses_a_consumed_run_root(driver, task, tmp_path) -> None:
    run = tmp_path / "run"
    (run / "targets").mkdir(parents=True)
    before = sorted(p.relative_to(run) for p in run.rglob("*"))
    with pytest.raises(RuntimeError, match="refusing re-entry"):
        driver.coordinate(tmp_path / "task.json", task, run)
    assert sorted(p.relative_to(run) for p in run.rglob("*")) == before
