"""Protect prerequisite ordering in the field-target development experiment."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


def _report():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/experiments/20260908_field_teacher_information_stage_frame00008_report.py"
    )
    spec = importlib.util.spec_from_file_location("field_information_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_paired_seed_and_teacher_fidelity_are_required():
    report = _report()
    task = {"seeds": [1, 2, 3]}
    quality = {
        "foreground_psnr": 28.0,
        "full_psnr": 35.0,
        "boundary_psnr": 25.0,
        "crop_lpips": 0.10,
        "alpha_iou": 0.95,
    }
    cells = [
        {"arm": arm, "seed": seed, "evaluation": {"mean": copy.deepcopy(quality)}}
        for seed in task["seeds"]
        for arm in ("rgb", "field_high")
    ]
    prep = {"teacher_metrics": {"field_high": [{"foreground_psnr": 32.0, "crop_lpips": 0.05}]}}
    assert report._gates(task, {}, cells, prep)["numeric_prerequisites_pass"]
    cells[-1]["evaluation"]["mean"]["boundary_psnr"] = 24.49
    assert not report._gates(task, {}, cells, prep)["numeric_prerequisites_pass"]
    cells[-1]["evaluation"]["mean"]["boundary_psnr"] = 25.0
    prep["teacher_metrics"]["field_high"][0]["foreground_psnr"] = 29.99
    assert not report._gates(task, {}, cells, prep)["numeric_prerequisites_pass"]
    prep["teacher_metrics"]["field_high"][0]["foreground_psnr"] = 32.0
    cells[0]["evaluation"]["mean"]["alpha_iou"] = 0.89
    assert not report._gates(task, {}, cells, prep)["numeric_prerequisites_pass"]


def test_inclusive_margin_and_complete_camera_coverage():
    report = _report()
    a = {
        "foreground_psnr": 28.0,
        "full_psnr": 35.0,
        "boundary_psnr": 25.0,
        "crop_lpips": 0.05,
        "alpha_iou": 0.95,
    }
    b = {**a, "crop_lpips": 0.07}
    cells = [
        {"arm": arm, "seed": 1, "evaluation": {"mean": quality}}
        for arm, quality in [("rgb", a), ("field_high", b)]
    ]
    prep = {"teacher_metrics": {"field_high": [{"foreground_psnr": 32.0, "crop_lpips": 0.04}]}}
    assert report._gates({"seeds": [1]}, {}, cells, prep)["numeric_prerequisites_pass"]
    with pytest.raises(ValueError, match="exactly once"):
        report._row_means([{"view_id": "C0001", "metric": 1.0}], ["C0001", "C0002"], ["metric"])
    with pytest.raises(ValueError, match="exactly once"):
        report._row_means([{"view_id": "C0001", "metric": 1.0}] * 2, ["C0001", "C0002"], ["metric"])


def test_native_flat_loss_uses_observed_checkpoint_clock():
    report = _report()
    stages = ["prepare", "initialize", "fit", "evaluate"]
    task = {"stages": [{"id": stage, "label": stage} for stage in stages]}
    history = {
        "loss": [0.4, 0.3],
        "n_gaussians": [[1, 10], [2, 12]],
        "elapsed": [[1, 0.2], [2, 0.4]],
        "checkpoint_observer": [{"step": 1, "run_seconds": 3.2}, {"step": 2, "run_seconds": 3.5}],
    }
    receipt = {"stage_intervals": dict(zip(stages, [[0, 1], [1, 2], [2, 9], [10, 11]]))}
    output = report._history(
        task, [{"arm": "rgb", "seed": 1, "receipt": receipt, "history": history}]
    )
    loss_rows = [r for r in output["records"] if r["metric_id"] == "loss_total"]
    assert [r["value"] for r in loss_rows] == [0.4, 0.3]
    assert [r["wall_seconds"] for r in loss_rows] == [3.2, 3.5]
