"""CPU contracts for phase-one acquisition; no local captures are opened."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from rtgs import bench019_local_inputs as inputs
from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.compact_views import CompactDataset, CompactView, save_compact_view


def _target(size=32):
    y, x = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
    rgb = torch.stack((x, y, 0.2 + x * y * 0.7), dim=-1)
    alpha = torch.zeros(size, size)
    alpha[4:-4, 4:-4] = 1
    alpha[3, 4:-4] = 0.25
    return rgb, alpha


def _camera(size):
    return Camera(20, 20, size / 2, size / 2, size, size, torch.eye(3), torch.zeros(3))


def _field(rows=8, size=32, *, amplitude=1.0, view_id="C0001"):
    return GaussianObservationField(
        width=size,
        height=size,
        means=torch.full((rows, 2), size / 2),
        log_scales=torch.zeros(rows, 2),
        rotations=torch.zeros(rows),
        colors=torch.full((rows, 3), 0.4),
        amplitudes=torch.full((rows,), amplitude),
        view_id=view_id,
        n_init=rows,
        provider="synthetic_fixture",
    )


def test_import_does_not_load_torch_or_structsplat():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import rtgs.bench019_local_inputs; "
            "assert 'torch' not in sys.modules; assert 'structsplat' not in sys.modules",
        ],
        check=True,
    )


def test_resolved_configs_freeze_count_objective_and_single_containment_axis():
    pytest.importorskip("structsplat")
    configs = inputs.resolved_stage1_configs()
    native = configs["native_additive"]["fit"]
    assert (native["n_gaussians"], native["iterations"], native["max_gaussians"]) == (
        512,
        1000,
        512,
    )
    assert not native["adaptive_density"] and not native["pool"]
    assert native["convergence_patience"] == native["mask_coverage_weight"] == 0
    normalized = configs["structsplat_normalized"]
    contained = copy.deepcopy(configs["structsplat_contained"])
    assert contained["fit"].pop("mask_contain") is True
    expected = copy.deepcopy(normalized)
    assert expected["fit"].pop("mask_contain") is False
    assert expected == contained
    fit = normalized["fit"]
    assert fit["pixel_loss"] == "l2" and fit["loss_weighting"] == "none"
    assert fit["ssim_weight"] == fit["geometry_loss_weight"] == fit["aa_dilation"] == 0
    assert fit["support_fade"] and fit["normalization_eps"] == 1e-8
    assert fit["split_every"] is fit["prune_every"] is fit["early_stop_patience"] is None
    assert normalized["init"]["scale_cap_mode"] == "none"


def test_common_target_keeps_soft_alpha_and_exact_margin():
    rgb, alpha = _target()
    crop, window = inputs.prepare_target(rgb, alpha)
    assert window == (2, 2, 28, 28)
    torch.testing.assert_close(crop, (rgb * alpha[..., None])[2:30, 2:30], rtol=0, atol=0)
    assert bool((crop[1, 2:-2] > 0).all())
    alpha[3, 4] = 0.5
    assert inputs.prepare_target(rgb, alpha)[1][1] == 1
    with pytest.raises(ValueError, match="foreground and exterior"):
        inputs.prepare_target(rgb, torch.ones_like(alpha))


@pytest.mark.parametrize("family", inputs.FAMILIES)
def test_tiny_fit_preserves_target_fixed_horizon_and_cold_observation(
    tmp_path, monkeypatch, family
):
    pytest.importorskip("structsplat")
    config = inputs.resolved_stage1_configs()[family]
    rgb, alpha = _target()
    target, window = inputs.prepare_target(rgb, alpha)
    x, y, width, height = window
    mask = alpha[y : y + height, x : x + width]
    target_before = target.clone()
    calls = []
    if family == "native_additive":
        import rtgs.image2gs.fit as fit_module

        config["fit"].update(n_gaussians=8, max_gaussians=8, iterations=3, native_renderer="torch")
        original = fit_module.fit_image

        def checked(image, cfg, **kwargs):
            assert kwargs["mask"] is None
            torch.testing.assert_close(image, target_before, rtol=0, atol=0)
            calls.append(True)
            return original(image, cfg, **kwargs)

        monkeypatch.setattr(fit_module, "fit_image", checked)
    else:
        import importlib

        fit_module = importlib.import_module("structsplat.fit")
        config["fit"].update(iters=3, max_gaussians=8, renderer="normalized", log_every=3)
        config["init"].update(num_gaussians=8, seed=72)
        original = fit_module.fit

        def checked(field, image, cfg, **kwargs):
            torch.testing.assert_close(image, target_before, rtol=0, atol=0)
            assert (kwargs["mask"] is not None) == (family == "structsplat_contained")
            assert cfg.pixel_loss == "l2" and cfg.ssim_weight == 0 and cfg.loss_weighting == "none"
            calls.append(True)
            return original(field, image, cfg, **kwargs)

        monkeypatch.setattr(fit_module, "fit", checked)
    field, history, source_render = inputs._fit_target(
        target,
        mask,
        family,
        config,
        72,
        (32, 32),
        window,
        "C0001",
        "a" * 64,
        history_steps=(0, 1, 3),
    )
    assert calls == [True]
    torch.testing.assert_close(target, target_before, rtol=0, atol=0)
    assert field.n == field.n_init == 8
    assert [row["step"] for row in history] == [0, 1, 3]
    assert all(row["rows"] == 8 for row in history)
    assert history[-1]["pixel_l2"] == pytest.approx(float((source_render - target).square().mean()))
    assert field.blend_mode == ("additive" if family == "native_additive" else "normalized")
    assert field.support_fade_alpha == (0 if family == "native_additive" else 1)
    path = tmp_path / "C0001.rtgsv"
    save_compact_view(
        path,
        field,
        _camera(32),
        calibration_sha256="c" * 64,
        source_rgb_name="C0001.jpg",
        source_rgb_sha256="d" * 64,
        source_mask_name="mask_C0001.png",
        source_mask_sha256="e" * 64,
        alpha_crop=mask > 0.5,
    )
    cold = CompactView.load(path).observation
    points = inputs._points(torch.arange(32 * 32), 32)
    before = inputs._query(field.to("cpu", dtype=torch.float64), points)
    after = inputs._query(cold.to("cpu", dtype=torch.float64), points)
    for name in ("color", "numerator", "weight_sum", "valid"):
        torch.testing.assert_close(getattr(before, name), getattr(after, name), rtol=0, atol=0)


def test_support_uses_weight_threshold_instead_of_crop_validity_or_alpha():
    rgb, alpha = _target()
    field = _field(amplitude=0.0)
    indices = torch.arange(32 * 32)
    stats = inputs.score_observation(field, rgb * alpha[..., None], alpha, indices, indices)
    assert stats["true_positive"] == stats["false_positive"] == 0
    assert stats["false_negative"] == int((alpha >= 0.5).sum())
    assert inputs._metrics(stats)["support_iou"] == 0


def test_predictor_samples_are_family_independent_and_fallback_is_explicit():
    alpha = torch.ones(64, 64)
    alpha[0] = alpha[-1] = alpha[:, 0] = alpha[:, -1] = 0
    a = inputs.predictor_samples(alpha, 191019, "frame_00008", "C0001", count=1)
    b = inputs.predictor_samples(alpha, 191019, "frame_00008", "C0001", count=1)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2]
    assert a[2]["complete_boundary_fallback"] and a[2]["boundary_count"] > 1


@pytest.mark.parametrize("fault", [None, "row_count", "raw_mutation"])
def test_phase_one_reads_only_training_views_and_writes_standard_manifest(
    tmp_path, monkeypatch, fault
):
    pytest.importorskip("structsplat")
    import rtgs.data.calibrated as calibrated

    frame = tmp_path / "frame"
    (frame / "rgb").mkdir(parents=True)
    (frame / "mask").mkdir()
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}")
    train = [f"C{i:04}" for i in range(1, 9)]
    for view in train:
        (frame / "rgb" / f"{view}.jpg").write_bytes(b"synthetic mock RGB")
        (frame / "mask" / f"mask_{view}.png").write_bytes(b"synthetic mock mask")
    task = {
        "status": "ready",
        "production": {
            "resolved_dataclasses": inputs.resolved_stage1_configs(),
            "downscale": 8,
            "count": 512,
            "iterations": 1000,
            "seed_base": 190000,
            "history_steps": list(inputs.HISTORY_STEPS),
        },
        "predictor_policy": {"seed": 191019, "samples_per_view": 2048},
        "splits": {"frame_00008": {"train": train, "heldout": ["C9999"]}},
        "datasets": [
            {"id": "frame_00008", "frame_path": str(frame), "calibration": str(calibration)}
        ],
    }
    rgb, alpha = _target(16)
    exposure = []

    def load(_frame, **kwargs):
        assert kwargs["view_ids"] == train and kwargs["test_every"] == 0
        assert kwargs["downscale"] == 8 and kwargs["undistort"]
        exposure.extend(kwargs["view_ids"])
        return SimpleNamespace(
            view_names=train,
            train_indices=list(range(8)),
            test_indices=[],
            images=[rgb] * 8,
            masks=[alpha] * 8,
            cameras=[_camera(16)] * 8,
        )

    def fit(target, mask, family, config, seed, canvas, window, view_id, source_digest):
        assert target.device.type == "cpu" and config["fit"]["iterations"] == 1000
        assert seed == 190000 + train.index(view_id)
        field = _field(rows=511 if fault == "row_count" else 512, size=16, view_id=view_id)
        field = replace(
            field,
            fit_window=window,
            provider="native",
            blend_mode="additive",
            sigma_cutoff=math.sqrt(12.0),
            fit_config_digest=inputs._digest(config),
            producer_source_digest=source_digest,
        )
        if fault == "raw_mutation":
            (frame / "rgb" / f"{train[0]}.jpg").write_bytes(b"changed source bytes")
        return (
            field,
            [
                {"step": step, "rows": 512, "pixel_l2": 1.0, "elapsed_seconds": float(step)}
                for step in inputs.HISTORY_STEPS
            ],
            target,
        )

    monkeypatch.setattr(calibrated, "load_calibrated_scene", load)
    monkeypatch.setattr(inputs, "_fit_target", fit)
    monkeypatch.setattr(inputs, "_source_binding", lambda: {"sha256": "f" * 64, "files": {}})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    if fault:
        with pytest.raises(ValueError, match="drift|changed during"):
            inputs.produce_frame_family(task, "frame_00008", "native_additive", tmp_path / "out")
        failure = json.loads((tmp_path / "out" / "failure.json").read_text())
        assert failure["status"] == "failed"
        assert not (tmp_path / "out" / "production.json").exists()
        assert "C9999" not in exposure
        return
    receipt = inputs.produce_frame_family(task, "frame_00008", "native_additive", tmp_path / "out")
    assert exposure == train
    assert receipt["status"] == "complete" and receipt["train_views"] == train
    dataset = CompactDataset.load(tmp_path / "out")
    assert [view.view_id for view in dataset.views] == train and dataset.bounds_hint is None
    assert all(view.observation.n == 512 for view in dataset.views)
    metrics = json.loads((tmp_path / "out" / "stage1_metrics.json").read_text())
    assert metrics["metrics"]["rows"] == 4096 and len(metrics["views"]) == 8
    assert all(view["cold_replay_max_abs"] == 0 for view in metrics["views"])
    with pytest.raises(FileExistsError):
        inputs.produce_frame_family(task, "frame_00008", "native_additive", tmp_path / "out")


def test_draft_and_count_drift_are_rejected_before_raw_access(tmp_path):
    with pytest.raises(ValueError, match="ready task"):
        inputs.produce_frame_family(
            {"status": "draft"}, "frame_00008", "native_additive", tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()
    pytest.importorskip("structsplat")
    task = {
        "status": "ready",
        "production": {
            "resolved_dataclasses": inputs.resolved_stage1_configs(),
            "downscale": 8,
            "count": 511,
            "iterations": 1000,
            "seed_base": 190000,
            "history_steps": list(inputs.HISTORY_STEPS),
        },
    }
    with pytest.raises(ValueError, match="fixed budget"):
        inputs.produce_frame_family(task, "frame_00008", "native_additive", tmp_path / "out")


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("family", inputs.FAMILIES)
def test_tiny_cuda_fit_uses_selected_renderer_and_fixed_checkpoint(family):
    pytest.importorskip("structsplat")
    config = inputs.resolved_stage1_configs()[family]
    rgb, alpha = _target()
    target, window = inputs.prepare_target(rgb, alpha)
    x, y, width, height = window
    if family == "native_additive":
        config["fit"].update(n_gaussians=8, max_gaussians=8, iterations=2)
    else:
        config["fit"].update(iters=2, max_gaussians=8, log_every=2)
        config["init"].update(num_gaussians=8, seed=72)
    field, history, source_render = inputs._fit_target(
        target,
        alpha[y : y + height, x : x + width],
        family,
        config,
        72,
        (32, 32),
        window,
        "C0001",
        "a" * 64,
        history_steps=(0, 1, 2),
    )
    torch.cuda.synchronize()
    assert field.n == 8 and [item["step"] for item in history] == [0, 1, 2]
    assert history[-1]["pixel_l2"] == pytest.approx(float((source_render - target).square().mean()))
    assert torch.isfinite(source_render).all()
