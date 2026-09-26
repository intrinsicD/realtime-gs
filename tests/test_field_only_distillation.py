"""CPU contract tests for the RTGS-025 field-only distillation driver and decision policy."""

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

from rtgs.core.camera import Camera

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "20260926_field_only_distillation_stage_frame00008"
DRIVER = ROOT / "scripts/experiments" / f"{TASK_ID}.py"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _module(DRIVER, "field_only_driver")


@pytest.fixture(scope="module")
def report():
    return _module(DRIVER.with_name(f"{TASK_ID}_report.py"), "field_only_report_test")


@pytest.fixture(scope="module")
def task():
    return json.loads((ROOT / "experiments/tasks" / f"{TASK_ID}.json").read_text())


def test_driver_and_report_tables_match_frozen_protocol(driver, report, task) -> None:
    driver.check_protocol_tables(task)
    assert tuple(driver.CONDITIONS) == report.CONDITIONS
    assert [m["id"] for m in task["primary_metrics"]] == list(driver.METRICS)
    assert list(report.SELECTED) == [
        task["preview_policy"]["selected_model"].split()[0],
        task["seeds"][0],
    ]
    assert task["preprocessing"]["mask_family"] in task["field_inputs"]
    broken = copy.deepcopy(task)
    broken["execution_order"]["cells"].pop()
    with pytest.raises(RuntimeError, match="exactly once"):
        driver.check_protocol_tables(broken)


def test_resolved_training_configs_round_trip_and_differ_only_in_mask_terms(driver, task) -> None:
    for seed in task["seeds"]:
        masked = driver.train_config(task, "masked_silhouette", seed)
        black = driver.train_config(task, "premultiplied_black", seed)
        assert masked.use_masks and masked.random_background and masked.seed == seed
        assert not black.use_masks and not black.random_background
        assert masked.iterations == black.iterations == 8000
        left = task["resolved_training_configs"]["masked_silhouette"][str(seed)]
        right = task["resolved_training_configs"]["premultiplied_black"][str(seed)]
        assert {k for k in left if left[k] != right[k]} == {"use_masks", "random_background"}


def test_colour_is_scored_only_inside_the_mask(driver) -> None:
    mask = torch.zeros(20, 20, dtype=torch.bool)
    mask[5:15, 5:15] = True
    reference = torch.full((20, 20, 3), 0.4)
    prediction = reference.clone()
    prediction[~mask] = 0.9
    alpha = mask.float()
    clean = driver.mask_scores(prediction, alpha, reference, mask)
    assert clean["foreground_psnr"] > 100
    assert clean["outside_alpha_mass"] == 0 and clean["floater_fraction"] == 0
    assert clean["interior_alpha"] == 1
    prediction[mask] = 0.5
    alpha[0, :] = 1.0  # a floater row outside the dilated band
    noisy = driver.mask_scores(prediction, alpha, reference, mask)
    assert abs(noisy["foreground_psnr"] - 20.0) < 1e-4
    assert noisy["floater_fraction"] == pytest.approx(20 / noisy["outside_pixels"])
    # A halo inside the 3-pixel band is not counted as a floater.
    halo = mask.float()
    halo[4, 5:15] = 1.0
    assert driver.mask_scores(reference, halo, reference, mask)["floater_fraction"] == 0


def test_random_initialization_is_seeded_and_inside_the_ball(driver) -> None:
    first = driver.random_initialization(torch.tensor([1.0, 2.0, 3.0]), 4.0, 500, 9261)
    again = driver.random_initialization(torch.tensor([1.0, 2.0, 3.0]), 4.0, 500, 9261)
    other = driver.random_initialization(torch.tensor([1.0, 2.0, 3.0]), 4.0, 500, 9262)
    assert torch.equal(first["means"], again["means"])
    assert not torch.equal(first["means"], other["means"])
    radius = (first["means"] - torch.tensor([1.0, 2.0, 3.0])).norm(dim=1)
    assert float(radius.max()) <= 2.0 + 1e-5 and first["scale"] > 0


def test_alpha_hull_recovers_a_centred_disc_silhouette(driver) -> None:
    cameras, masks = [], []
    for angle in torch.linspace(0, 2 * torch.pi, 7)[:-1]:
        eye = torch.tensor([3 * torch.cos(angle), 0.4, 3 * torch.sin(angle)])
        camera = Camera.look_at(eye=eye, target=torch.zeros(3), width=32, height=32)
        yy, xx = torch.meshgrid(torch.arange(32) + 0.5, torch.arange(32) + 0.5, indexing="ij")
        masks.append((((xx - 16) ** 2 + (yy - 16) ** 2) <= 36).float())
        cameras.append(camera)
    shell, voxel, occupied = driver.hull_candidates(cameras, masks, torch.zeros(3), 2.4, 24)
    assert occupied > len(shell) > 0 and voxel == pytest.approx(0.1)
    assert float(shell.norm(dim=1).max()) < 1.2
    with pytest.raises(RuntimeError, match="touches"):
        driver.hull_candidates(cameras, [torch.ones(32, 32)] * 6, torch.zeros(3), 0.5, 8)


def _cells(values: dict[str, dict]) -> list[dict]:
    base = {
        "foreground_psnr": 25.0,
        "crop_lpips": 0.10,
        "outside_alpha_mass": 0.010,
        "floater_fraction": 0.001,
        "interior_alpha": 0.99,
    }
    return [
        {
            "condition": condition,
            "seed": seed,
            "evaluation": {"mean": {**base, **values.get(condition, {})}},
        }
        for condition in (
            "nb_rand_ms",
            "nb_hull_ms",
            "nb_rand_pm",
            "mc_rand_ms",
            "gi_rand_ms",
            "ph_rand_ms",
            "ph_hull_ms",
        )
        for seed in (1, 2)
    ]


def test_frozen_gates_are_inclusive_per_seed_and_ordered(report) -> None:
    task = {"seeds": [1, 2]}
    boundary = {
        "nb_rand_ms": {"foreground_psnr": 24.5, "crop_lpips": 0.12, "outside_alpha_mass": 0.015},
        "mc_rand_ms": {"foreground_psnr": 24.25, "outside_alpha_mass": 0.010},
    }
    result = report.gates(task, _cells(boundary))
    assert result["g0_photo_reference"]["numeric_pass"]
    assert result["h1_main_path"]["verdict"] == "pass"
    assert result["h2_teacher_containment"]["verdict"] == "pass"
    worse = copy.deepcopy(boundary)
    worse["nb_rand_ms"]["outside_alpha_mass"] = 0.0151
    assert report.gates(task, _cells(worse))["h1_main_path"]["verdict"] == "fail"
    reverse = {"mc_rand_ms": {"foreground_psnr": 25.25}}
    assert report.gates(task, _cells(reverse))["h2_teacher_containment"]["verdict"] == "reject"
    mixed = _cells({})
    assert report.gates(task, mixed)["h2_teacher_containment"]["verdict"] == "inconclusive"
    failed_reference = _cells({"ph_rand_ms": {"foreground_psnr": 23.99}})
    result = report.gates(task, failed_reference)
    assert result["h1_main_path"]["verdict"] == "inconclusive"
    assert "photograph reference failed" in report.decision_text(result)


def test_history_uses_observed_checkpoint_clock(report) -> None:
    stages = ["prepare", "initialize", "fit", "evaluate"]
    task = {"stages": [{"id": stage, "label": stage} for stage in stages]}
    trace = {
        "loss": [0.4, 0.3],
        "n_gaussians": [[1, 10], [2, 12]],
        "elapsed": [[1, 0.2], [2, 0.4]],
        "checkpoint_observer": [{"step": 1, "run_seconds": 3.2}, {"step": 2, "run_seconds": 3.5}],
    }
    receipt = {"stage_intervals": dict(zip(stages, [[0, 1], [1, 2], [2, 9], [10, 11]]))}
    cell = {"condition": "nb_rand_ms", "seed": 1, "receipt": receipt, "history": trace}
    output = report.history(task, [cell])
    losses = [r for r in output["records"] if r["metric_id"] == "loss_total"]
    assert [r["value"] for r in losses] == [0.4, 0.3]
    assert [r["wall_seconds"] for r in losses] == [3.2, 3.5]
    with pytest.raises(ValueError, match="exactly once"):
        report.row_means([{"view_id": "C0001", "m": 1.0}], ["C0001", "C0002"], ["m"])


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
