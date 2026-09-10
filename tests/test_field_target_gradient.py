"""Synthetic acceptance controls for the frozen target-gradient diagnostic."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def driver():
    path = Path(__file__).resolve().parents[1] / (
        "scripts/experiments/20260909_field_target_gradient_stage_frame00008.py"
    )
    spec = importlib.util.spec_from_file_location("field_target_gradient_test_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_rendered_components_opacity_and_identity(driver):
    result = driver.selftest("cpu")
    assert result["direct_full_gradient_control"]["passed"]
    assert result["identity_control"]["passed"]
    assert result["state_unchanged"]
    assert result["signal_deadline"] == "passed"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA renderer unavailable")
def test_cuda_rendered_components_opacity_and_identity(driver):
    result = driver.selftest("cuda:0")
    assert result["direct_full_gradient_control"]["passed"]
    assert result["identity_control"]["passed"]


def test_residual_spatial_statistics_against_numpy_windows(driver):
    rng = np.random.default_rng(6051)
    photo = rng.random((13, 15, 3)).astype(np.float32)
    field = rng.random((13, 15, 3)).astype(np.float32)
    mask = np.zeros((13, 15), dtype=bool)
    mask[2:11, 3:12] = True
    result = driver.residual_statistics(
        torch.from_numpy(photo), torch.from_numpy(field), torch.from_numpy(mask)
    )

    def naive_mean(image):
        windows = np.lib.stride_tricks.sliding_window_view(
            np.pad(image, ((3, 3), (3, 3), (0, 0)), mode="reflect"), (7, 7), axis=(0, 1)
        )
        return windows.mean(axis=(-2, -1))

    delta = field.astype(np.float64) - photo.astype(np.float64)
    lowpass = naive_mean(delta)
    contrast = np.sqrt(
        np.maximum(
            naive_mean(photo.astype(np.float64) ** 2) - naive_mean(photo.astype(np.float64)) ** 2, 0
        )
    )
    padded_mask = np.pad(mask, 3, constant_values=False)
    windows = np.lib.stride_tricks.sliding_window_view(padded_mask, (7, 7))
    boundary = windows.any(axis=(-2, -1)) ^ windows.all(axis=(-2, -1))
    partitions = {"interior": mask & ~boundary, "boundary": boundary, "exterior": ~mask & ~boundary}
    for name, selected in partitions.items():
        row = result["regions"][name]
        assert row["pixel_count"] == int(selected.sum())
        if not selected.any():
            assert row["mae_region"] is None
            continue
        np.testing.assert_allclose(
            row["signed_mean_per_channel"], delta[selected].mean(axis=0), rtol=1e-12, atol=1e-12
        )
        assert row["photo_local_contrast"] == pytest.approx(contrast[selected].mean(), abs=1e-12)
        assert row["residual_lowpass_mse"] == pytest.approx(
            (lowpass[selected] ** 2).mean(), abs=1e-12
        )
        assert row["residual_highpass_mse"] == pytest.approx(
            ((delta - lowpass)[selected] ** 2).mean(), abs=1e-12
        )
    assert sum(row["full_canvas_l1_contribution"] or 0 for row in result["regions"].values()) == (
        pytest.approx(np.abs(delta).mean(), abs=1e-12)
    )


def test_statistics_distinguish_mean_norm_from_norm_of_mean(driver):
    vectors = [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])]
    per_view = [driver.compare_gradients(value, value)["photo_norm"] for value in vectors]
    aggregate = driver.compare_gradients(
        torch.stack(vectors).double().mean(0), torch.stack(vectors).double().mean(0)
    )
    assert np.mean(per_view) == 1
    assert aggregate["photo_norm"] == 0
    assert aggregate["cosine"] is None
    assert aggregate["difference_over_photo_norm"] is None


def test_mapped_opacity_gradient_matches_independent_logit_graph(driver):
    from rtgs.render.base import get_rasterizer

    model, state, view = driver.synthetic_inputs()
    renderer = get_rasterizer("torch", device="cpu")
    measured = driver.evaluate_target(model, state, view, "rgb", renderer, "cpu")
    parameters = driver.fresh_parameters(model, state["kind"], "cpu")
    logits = torch.logit(model.opacity).detach().requires_grad_(True)
    parameters["opacities"] = torch.sigmoid(logits)
    prediction = renderer.render(driver.assemble(parameters), view["camera"], sh_degree=3).color
    loss = 0.8 * (prediction - view["rgb"]).abs().mean() + 0.2 * (
        1 - driver.ssim(prediction, view["rgb"])
    )
    (direct,) = torch.autograd.grad(loss, logits)
    torch.testing.assert_close(
        measured["gradients"]["total"]["opacities"], direct, atol=1e-7, rtol=1e-4
    )


def test_installed_input_guard_denies_real_file_opens(driver, tmp_path):
    code = """
import importlib.util
from pathlib import Path
import sys
spec = importlib.util.spec_from_file_location('guard_child', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
root = Path(sys.argv[2])
paths = ['dataset/rgb/C0001.jpg', 'old/heldout/C0001.npz', 'old/unlisted.json', 'old/allowed.npz']
for name in paths:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'synthetic')
guard = module.InputGuard({'source_run':'old', 'cached_inputs':[{'path':'old/allowed.npz'}]}, root)
guard.install()
assert (root / 'old/allowed.npz').read_bytes() == b'synthetic'
for name in paths[:3]:
    try:
        (root / name).read_bytes()
    except PermissionError:
        pass
    else:
        raise AssertionError(name)
try:
    (root / 'old/allowed.npz').write_bytes(b'changed')
except PermissionError:
    pass
else:
    raise AssertionError('write admitted')
assert len(guard.receipt['denied']) == 4
"""
    subprocess.run(
        [sys.executable, "-c", code, driver.__file__, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_environment_receipt_matches_shared_schema(driver, tmp_path, monkeypatch):
    path = Path(driver.__file__).resolve().parents[1] / "experiment_contract.py"
    spec = importlib.util.spec_from_file_location("gradient_environment_contract", path)
    contract = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = contract
    spec.loader.exec_module(contract)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    (tmp_path / "task.lock.json").write_text(json.dumps({"source_commit": "a" * 40}))
    driver.write_environment(tmp_path)
    environment = json.loads((tmp_path / "environment.json").read_text())
    assert contract._environment_errors(environment) == []
    assert "rtgs" in environment["packages"]
    details = json.loads((tmp_path / "environment_details.json").read_text())
    assert details["source_commit"] == "a" * 40
    assert details["available"] is False
