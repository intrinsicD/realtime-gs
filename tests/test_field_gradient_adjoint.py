"""Synthetic algebra, precision-flag and storage-boundary controls."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def driver():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/experiments/20260910_field_gradient_adjoint_stage_frame00008.py"
    )
    spec = importlib.util.spec_from_file_location("adjoint_test_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_distinct_target_chain_rule_and_zero_control(driver):
    result = driver.selftest()
    assert result["common_render_control"]["passed"]
    assert result["full_loss_linearity"] == "passed"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_distinct_target_chain_rule_and_zero_control(driver):
    result = driver.selftest("cuda:0")
    assert result["common_render_control"]["passed"]


def raw_fixture(driver, photo, delta):
    return {
        kind: [
            {
                component: {
                    group: torch.tensor([value, value], dtype=torch.float32)
                    for group in driver.GROUPS
                }
                for component in driver.BASE_COMPONENTS
            }
            for value in values
        ]
        for kind, values in (("photo", photo), ("delta", delta))
    }


def test_noise_flags_and_lossless_raw_reduction(driver, tmp_path):
    raw = raw_fixture(driver, [1.0, 1.2], [0.01, 0.015])
    result = driver.summarize_repeats(raw)
    flag = result["reliability"]["total"]["means"]
    assert not flag["effect_resolved_against_observed_repeats"]
    assert not flag["reference_resolved"]
    assert not flag["field_resolved"]
    path = tmp_path / "raw.npz"
    np.savez_compressed(path, **driver.raw_repeat_arrays(raw))
    with np.load(path) as archive:
        restored = {
            kind: [
                {
                    c: {
                        g: torch.from_numpy(archive[f"{kind}__r{r}__{c}__{g}"].copy())
                        for g in driver.GROUPS
                    }
                    for c in driver.BASE_COMPONENTS
                }
                for r in range(2)
            ]
            for kind in ("photo", "delta")
        }
        assert all(archive[key].dtype == np.float32 for key in archive.files)
    recomputed = driver.summarize_repeats(restored)
    assert recomputed["comparisons"] == result["comparisons"]
    assert recomputed["reliability"] == result["reliability"]
    clear = driver.summarize_repeats(raw_fixture(driver, [1.0, 1.0], [0.1, 0.1]))
    assert all(
        clear["reliability"]["total"]["means"][key]
        for key in (
            "reference_resolved",
            "field_resolved",
            "effect_resolved_against_observed_repeats",
        )
    )
    zero = driver.summarize_repeats(raw_fixture(driver, [0.0, 0.0], [0.0, 0.0]))
    assert zero["comparisons"]["total"]["means"]["cosine"] is None
    assert not zero["reliability"]["total"]["means"]["reference_resolved"]


def test_storage_guards_preserve_existing_files(driver, tmp_path, monkeypatch):
    task = {
        "execution_budget": {"max_output_bytes": 100_000, "min_free_disk_bytes": 12_000_000_000}
    }
    budget = driver.StorageBudget(tmp_path, task)
    monkeypatch.setattr(driver.shutil, "disk_usage", lambda _: SimpleNamespace(free=11_000_000_000))
    with pytest.raises(RuntimeError, match="storage budget"):
        budget.check(preflight=True)
    monkeypatch.setattr(driver.shutil, "disk_usage", lambda _: SimpleNamespace(free=20_000_000_000))
    old = tmp_path / "existing.txt"
    old.write_text("preserve")
    with pytest.raises(RuntimeError, match="storage budget"):
        budget.save(tmp_path / "oversized.npz", {"data": np.zeros(100_001, dtype=np.float32)})
    assert not (tmp_path / "oversized.npz").exists()
    assert old.read_text() == "preserve"


def test_photo_gate_phase_excludes_field_decoding(driver, tmp_path):
    task = {
        "source_run": "old",
        "cached_inputs": [{"path": "old/targets/field_high/C0004.npz"}],
        "gradient_protocol": {"repeatability_probe": {"state": "s", "view": "C0004"}},
        "states": [{"id": "s", "model": "old/model.npz"}],
    }
    guard = driver.PhaseGuard(task, tmp_path)
    guard.phase = "numerical_control"
    with pytest.raises(PermissionError, match="photo-only"):
        guard.audit("open", (str(tmp_path / "old/targets/field_high/C0004.npz"), "rb", 0))


def test_initial_inactive_sh_and_local_opacity_chain(driver):
    from rtgs.render.base import get_rasterizer

    saved, _, view = driver.OLD.synthetic_inputs()
    saved = saved.with_sh_degree(0)
    state = {"kind": "initial", "sh_degree": 0}
    renderer = get_rasterizer("torch", device="cpu")
    parameters, image = driver.render_graph(saved, state, view, renderer, "cpu")
    measured = driver.repeat_measurement(parameters, image, view)
    for component in driver.COMPONENTS:
        assert measured["means"]["delta"][component]["shN"].eq(0).all()
    expected = torch.sigmoid(torch.logit(saved.opacity.clamp(1e-4, 1 - 1e-4)))
    assert torch.equal(parameters["opacities"], expected)
