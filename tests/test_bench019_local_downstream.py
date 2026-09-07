"""Synthetic contract checks for the task-owned BENCH019 downstream worker."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

import rtgs.bench019_local_downstream as worker
from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.compact_views import save_compact_view, write_compact_dataset_manifest
from rtgs.data.scene import SceneData
from rtgs.lift.field_sweep import FieldSweepConfig
from rtgs.optim.trainer import TrainConfig
from rtgs.render.torch_ref import TorchRasterizer


def _task():
    return {
        "task_id": "synthetic_contract_only",
        "comparators": [{"id": name} for name in worker.FAMILY_BLEND],
        "seeds": [19001, 19002, 19003],
        "splits": {
            "frame": {
                "train": [f"C{index:04d}" for index in range(8)],
                "heldout": ["C0100", "C0101", "C0102"],
            }
        },
        "datasets": [
            {"id": "frame", "frame_path": "unused-frame", "calibration": "unused-calibration.json"}
        ],
        "production": {"downscale": 8, "undistort": True, "count": 512},
        "downstream_config": {
            "resolved_dataclasses": worker.resolved_configs(),
            "history_steps": list(worker.HISTORY_STEPS),
            "bounds_hint": None,
        },
        "environment_policy": {"cpu_threads": 2},
    }


def _camera(index):
    angle = index * math.pi / 4
    return Camera.look_at(
        torch.tensor([3 * math.sin(angle), 0.3, -3 * math.cos(angle)]),
        torch.zeros(3),
        width=16,
        height=12,
    )


def _scene(names):
    mask = torch.zeros(12, 16)
    mask[3:9, 4:12] = 1
    mask[3, 4:12] = 0.5
    return SceneData(
        images=[torch.full((12, 16, 3), 0.5 + index / 100) for index in range(len(names))],
        cameras=[_camera(index) for index in range(len(names))],
        masks=[mask.clone() for _ in names],
        view_names=list(names),
        train_indices=list(range(len(names))),
        test_indices=[],
        bounds_hint=(torch.zeros(3), 1.0),
    )


def _bundle(tmp_path, task, *, count=512, blend="normalized", bounds=None):
    directory = tmp_path / "fields"
    directory.mkdir()
    names = task["splits"]["frame"]["train"]
    calibration = hashlib.sha256(b"synthetic calibration").hexdigest()
    paths = []
    for index, name in enumerate(names):
        field = GaussianObservationField(
            width=16,
            height=12,
            means=torch.full((count, 2), 6.5),
            log_scales=torch.zeros(count, 2),
            rotations=torch.zeros(count),
            colors=torch.full((count, 3), 0.5),
            amplitudes=torch.full((count,), 0.02),
            view_id=name,
            n_init=count,
            blend_mode=blend,
        )
        path = directory / f"{name}.rtgsv"
        save_compact_view(
            path,
            field,
            _camera(index),
            calibration_sha256=calibration,
            source_rgb_name=f"{name}.jpg",
            source_rgb_sha256="a" * 64,
        )
        paths.append(path)
    write_compact_dataset_manifest(
        directory,
        name="synthetic",
        calibration_sha256=calibration,
        view_paths=paths,
        bounds_hint=bounds,
    )
    return directory


def _model(n=256):
    return Gaussians3D.from_means_covs(
        torch.zeros(n, 3),
        torch.eye(3).expand(n, 3, 3).clone() * 0.005,
        torch.full((n, 3), 0.6),
        torch.full((n,), 0.01),
        sh_degree=0,
    )


def test_full_configuration_is_seed_bound_without_hidden_resolution():
    first, second = worker.resolved_configs(19001), worker.resolved_configs(19002)
    assert set(first["field_sweep"]) == {item.name for item in fields(FieldSweepConfig)}
    assert set(first["trainer"]) == {item.name for item in fields(TrainConfig)}
    for key in first:
        assert first[key].pop("seed") == 19001
        assert second[key].pop("seed") == 19002
    assert first == second
    trainer = first["trainer"]
    assert not trainer["densify"] and not trainer["internal_checkpoint_evaluation"]
    assert not trainer["reset_cuda_peak_stats"]
    assert trainer["checkpoint_policy"] == "final"
    assert trainer["schedule_iterations"] == trainer["iterations"] == 1000
    assert first["field_sweep"]["n_init_3d"] == 256


def test_metric_formulas_use_soft_target_binary_regions_and_display_clamp():
    image = torch.ones(1, 4, 3)
    mask = torch.tensor([[1.0, 0.5, 0.0, 0.0]])
    color = torch.tensor([[[1.2, -1.0, 0.5], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]])
    alpha = torch.tensor([[0.5, 0.5, 0.25, 0.0]])
    result = worker.foreground_metrics(color, alpha, image, mask)
    assert result["foreground_psnr"] == pytest.approx(-10 * math.log10(1.25 / 6))
    assert result["alpha_iou"] == 1.0
    assert result["exterior_leakage"] == 0.125
    assert result["raw"] == {
        "foreground_squared_error_sum": 1.25,
        "foreground_rgb_value_count": 6,
        "foreground_pixel_count": 2,
        "alpha_intersection_pixels": 2,
        "alpha_union_pixels": 2,
        "exterior_alpha_sum": 0.25,
        "exterior_pixel_count": 2,
    }


@pytest.mark.parametrize("failure", ["nan", "empty_foreground", "empty_exterior", "invalid_alpha"])
def test_metrics_fail_closed_for_invalid_or_empty_regions(failure):
    image = torch.ones(1, 2, 3)
    mask = torch.tensor([[1.0, 0.0]])
    alpha = mask.clone()
    if failure == "nan":
        image[0, 0, 0] = torch.nan
    elif failure == "empty_foreground":
        mask.zero_()
    elif failure == "empty_exterior":
        mask.fill_(1)
    else:
        alpha[0, 0] = 1.1
    with pytest.raises(ValueError):
        worker.foreground_metrics(image, alpha, image, mask)


def test_perfect_foreground_has_finite_declared_floor():
    mask = torch.tensor([[1.0, 0.0]])
    image = torch.ones(1, 2, 3)
    result = worker.foreground_metrics(image * mask[..., None], mask, image, mask)
    assert result["foreground_psnr"] == 120.0


def test_training_loader_requests_only_training_ids_and_never_heldout_bounds(monkeypatch):
    task = _task()
    calls = []

    def load(_path, **kwargs):
        calls.append(kwargs)
        return _scene(kwargs["view_ids"])

    monkeypatch.setattr(worker, "load_calibrated_scene", load)
    result = worker._load_scene(task, "frame", split="train")
    assert result.n_views == 8
    assert calls[0]["view_ids"] == task["splits"]["frame"]["train"]
    assert calls[0]["test_every"] == 0
    assert calls[0]["downscale"] == 8 and calls[0]["undistort"]
    assert not set(calls[0]["view_ids"]) & set(task["splits"]["frame"]["heldout"])


def test_cold_field_loading_authenticates_count_semantics_and_complete_bytes(tmp_path):
    task = _task()
    directory = _bundle(tmp_path, task)
    dataset, receipt = worker.load_training_fields(
        task, "frame", "structsplat_normalized", directory
    )
    assert dataset.n_views == 8 and dataset.bounds_hint is None
    assert receipt["field_bytes"] == sum(path.stat().st_size for path in directory.iterdir())
    assert receipt["rows_per_view"] == [512] * 8
    assert not receipt["alpha_decoded"]
    query = dataset.views[0].observation.query(torch.tensor([[6.5, 6.5], [0.5, 0.5]])).color
    torch.testing.assert_close(query, torch.tensor([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="blend"):
        worker.load_training_fields(task, "frame", "native_additive", directory)
    # A changed archive is detected independently of its manifest's semantic digest.
    archive = directory / "C0000.rtgsv"
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="byte count mismatch"):
        worker.load_training_fields(task, "frame", "structsplat_normalized", directory)


@pytest.mark.parametrize("failure", ["count", "bounds"])
def test_cold_field_contract_rejects_unequal_count_and_mask_bounds(tmp_path, failure):
    task = _task()
    directory = _bundle(
        tmp_path,
        task,
        count=511 if failure == "count" else 512,
        bounds=(torch.zeros(3), 1.0) if failure == "bounds" else None,
    )
    with pytest.raises(ValueError, match="count|bounds"):
        worker.load_training_fields(task, "frame", "structsplat_normalized", directory)


def test_worker_saves_fixed_history_and_never_loads_heldout(tmp_path, monkeypatch):
    task = _task()
    directory = _bundle(tmp_path, task)
    seen_splits = []

    def load(_task, _frame, *, split):
        seen_splits.append(split)
        assert split == "train"
        return _scene(task["splits"]["frame"]["train"])

    class Initializer:
        def __init__(self, config):
            assert config.n_init_3d == 256 and config.seed == 19002

        def initialize(self, inputs):
            assert inputs.bounds_hint is None
            assert inputs.view_names == task["splits"]["frame"]["train"]
            return SimpleNamespace(
                gaussians=_model(),
                depths=torch.ones(256),
                depth_sigmas=torch.ones(256),
                ray_sigmas=torch.ones(256),
                scores=torch.zeros(256),
                lineage=SimpleNamespace(
                    source_view_indices=torch.zeros(256, dtype=torch.long),
                    source_component_indices=torch.arange(256),
                    source_xy=torch.zeros(256, 2),
                ),
                diagnostics={
                    "bounds_source": "camera_axis_fallback",
                    "bounds_center": [0, 0, 0],
                    "bounds_extent": 1.0,
                    "search_aabb_lower": [-0.5] * 3,
                    "search_aabb_upper": [0.5] * 3,
                    "supported_track_count": 128,
                    "rounds": [{"median_best_cost": float("inf")}],
                },
            )

    class Training:
        def __init__(self, config):
            assert config.seed == 19002 and config.iterations == 1000
            assert not config.internal_checkpoint_evaluation

        def train(self, scene, initial, *, checkpoint_callback, initialization_callback):
            assert not scene.testing_views
            initialization_callback(initial)
            for step in range(50, 1001, 50):
                checkpoint_callback(initial, step)
            return initial, {
                "executed_iterations": 1000,
                "stop_reason": "max_iterations",
                "psnr": [],
                "sampled_train_views": [0, 1] * 500,
                "loss": [0.1] * 1000,
            }

    monkeypatch.setattr(worker, "_cuda_start", lambda: torch.device("cpu"))
    monkeypatch.setattr(worker, "_load_scene", load)
    monkeypatch.setattr(worker, "FieldSweepInitializer", Initializer)
    monkeypatch.setattr(worker, "Trainer", Training)
    output = tmp_path / "cell"
    receipt = worker.run_cell(task, "frame", "structsplat_normalized", directory, 19002, output)
    assert receipt["metrics"]["final_gaussians"] == 256
    assert receipt["unsupported_midpoint_count"] == 128
    assert receipt["metrics"]["wall_seconds"] >= receipt["metrics"]["refine_seconds"]
    assert seen_splits == ["train"]
    history = json.loads((output / "history.json").read_text())
    assert [row["step"] for row in history["checkpoints"]] == list(worker.HISTORY_STEPS)
    assert not history["heldout_evaluation"]
    diagnostic = json.loads((output / "initializer.json").read_text())
    assert diagnostic["rounds"][0]["median_best_cost"] == {"nonfinite_diagnostic": "inf"}
    assert Gaussians3D.load_npz(output / "gaussians.npz").n == 256
    with pytest.raises(FileExistsError):
        worker.run_cell(task, "frame", "structsplat_normalized", directory, 19002, output)


def test_separate_evaluation_cold_model_uses_only_heldout_and_unpooled_mean(tmp_path, monkeypatch):
    task = _task()
    model_path = tmp_path / "model.npz"
    _model().save_npz(model_path)
    seen = []

    def load(_task, _frame, *, split):
        seen.append(split)
        assert split == "heldout"
        return _scene(task["splits"]["frame"][split])

    monkeypatch.setattr(worker, "_cuda_start", lambda: torch.device("cpu"))
    monkeypatch.setattr(worker, "_load_scene", load)
    monkeypatch.setattr(worker, "get_rasterizer", lambda *args, **kwargs: TorchRasterizer())
    output = tmp_path / "reporting"
    receipt = worker.evaluate_cell(
        task, "frame", "structsplat_normalized", 19001, model_path, output
    )
    assert seen == ["heldout"]
    assert receipt["view_ids"] == task["splits"]["frame"]["heldout"]
    rows = receipt["per_view"]
    psnr = sum(row["foreground_psnr"] for row in rows) / 3
    assert receipt["metrics"]["heldout_foreground_psnr"] == psnr
    pooled = -10 * math.log10(
        sum(row["raw"]["foreground_squared_error_sum"] for row in rows)
        / sum(row["raw"]["foreground_rgb_value_count"] for row in rows)
    )
    assert psnr != pytest.approx(pooled, abs=1e-8)
    assert len(list(output.glob("*.png"))) == 12
    assert json.loads((output / "evaluation.json").read_text())["status"] == "ok"


def test_worker_rejects_unfrozen_config_before_data_or_cuda(tmp_path, monkeypatch):
    task = copy.deepcopy(_task())
    task["downstream_config"]["resolved_dataclasses"]["trainer"]["iterations"] = 999

    def forbidden():
        pytest.fail("CUDA initialization reached before config rejection")

    monkeypatch.setattr(worker, "_cuda_start", forbidden)
    with pytest.raises(ValueError, match="dataclass"):
        worker.run_cell(
            task, "frame", "structsplat_normalized", tmp_path / "absent", 19001, tmp_path / "cell"
        )
    assert not (tmp_path / "cell").exists()


def test_cpu_runtime_does_not_silently_substitute_reference_backend(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no fallback"):
        worker._cuda_start()


def test_real_synthetic_sweep_preserves_exact_row_budget_and_camera_only_bounds(tmp_path):
    task = _task()
    directory = _bundle(tmp_path, task)
    dataset, _ = worker.load_training_fields(task, "frame", "structsplat_normalized", directory)
    config = FieldSweepConfig(**worker.resolved_configs(19001)["field_sweep"])
    initial = worker.FieldSweepInitializer(config).initialize(dataset.to_reconstruction_inputs())
    assert initial.gaussians.n == 256
    worker._validate_model(initial.gaussians)
    assert initial.diagnostics["bounds_source"] == "camera_axis_fallback"
    assert initial.diagnostics["depth_within_original_bounds"]
    assert initial.diagnostics["source_view_counted_as_neighbor"] is False
    assert 0 <= initial.diagnostics["supported_track_count"] <= 256


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_trainer_diagnostic_uses_real_backend_and_no_internal_evaluation():
    # Two updates exercise the actual selected backend and callback seam; this is a
    # synthetic mechanism check, not the 1000-step result-bearing protocol.
    config = worker.resolved_configs(19001)["trainer"]
    config.pop("density")
    config["iterations"] = 2
    trainer = worker.Trainer(TrainConfig(**config))
    scene = _scene(["synthetic_a", "synthetic_b", "synthetic_c"])
    checkpoints = []
    final, history = trainer.train(
        scene,
        _model(),
        initialization_callback=lambda model: checkpoints.append((0, model.n)),
        checkpoint_callback=lambda model, step: checkpoints.append((step, model.n)),
    )
    worker._validate_model(final)
    assert final.means.device.type == "cuda"
    assert history["executed_iterations"] == 2 and not history["psnr"]
    assert checkpoints == [(0, 256), (2, 256)]
    assert all(math.isfinite(value) for value in history["loss"])
