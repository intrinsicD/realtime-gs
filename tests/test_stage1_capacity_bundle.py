from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from benchmarks import stage1_capacity_bundle as bridge
from PIL import Image

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.calibrated import _resize_image, _undistort
from rtgs.data.reconstruction_inputs import ReconstructionInputs


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(bridge.canonical_bytes(value))


def _camera(index: int) -> Camera:
    angle = math.tau * index / len(bridge.TRAIN_VIEWS)
    eye = torch.tensor(
        [
            2.0 * math.cos(angle),
            2.0 * math.sin(angle),
            0.4 + 0.05 * index,
        ]
    )
    return Camera.look_at(
        eye,
        torch.zeros(3),
        width=16,
        height=16,
        fov_x_deg=55.0,
    )


def _field(
    view_id: str,
    *,
    count: int,
    fit_window: list[int],
) -> GaussianObservationField:
    x0, y0, width, height = fit_window
    local_x = torch.linspace(1.25, width - 1.25, count)
    local_y = torch.linspace(height - 1.5, 1.5, count)
    means = torch.stack([local_x + x0 + 0.5, local_y + y0 + 0.5], dim=-1)
    residuals = torch.stack(
        [
            torch.linspace(-2.0e-4, 2.0e-4, count),
            torch.linspace(1.5e-4, -1.5e-4, count),
        ],
        dim=-1,
    ).to(torch.float32)
    return GaussianObservationField(
        width=16,
        height=16,
        means=means.to(torch.float32),
        log_scales=torch.full((count, 2), -0.3),
        rotations=torch.linspace(-0.2, 0.2, count),
        colors=torch.linspace(0.1, 0.9, count)[:, None].expand(-1, 3).contiguous(),
        amplitudes=torch.linspace(0.4, 1.0, count),
        mean_residuals=residuals,
        fit_window=tuple(fit_window),
        view_id=view_id,
        n_init=16,
        provider="synthetic_fixture",
        producer_version="stage1-capacity-bundle-test",
        producer_source_digest="a" * 64,
        fit_config_digest="b" * 64,
    )


def _make_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    source = tmp_path / "source_sweep"
    for directory in (
        data_root / "mask",
        source / "records",
        source / "teachers",
        source / "panels",
    ):
        directory.mkdir(parents=True)
    monkeypatch.setattr(bridge, "DATA_ROOT", data_root)

    arm = {
        "name": "arbitrary_adaptive_arm",
        "initializer": "future_initializer",
        "n_init_2d": 16,
        "iterations": 37,
        "max_2d": 128,
        "adaptive": True,
        "growth_waves": 3,
        "scale_cap_mode": "future_policy",
        "phase": "test",
    }
    repository = {
        "git_revision": "1" * 40,
        "files": {"future/source.py": "2" * 64},
        "aggregate_sha256": "3" * 64,
    }
    external = {
        "version": "test",
        "provider_source_digest": "4" * 64,
        "provider_source_manifest_sha256": "5" * 64,
    }
    environment = {"python": "test", "torch": "test"}
    extension = {"path": "/test/extension.so", "sha256": "6" * 64}
    view_inputs: dict[str, dict] = {}
    configs: dict[str, dict] = {}
    fields: dict[str, GaussianObservationField] = {}

    for index, view_id in enumerate(bridge.TRAIN_VIEWS):
        camera = _camera(index)
        raw_mask = np.zeros((16, 16), dtype=np.uint8)
        raw_mask[4 : 12 + index % 2, 3 + index % 3 : 12] = 255
        mask_relative = f"mask/mask_{view_id}.png"
        mask_path = data_root / mask_relative
        Image.fromarray(raw_mask).save(mask_path)
        mask = (
            _undistort(
                _resize_image(mask_path, 16, 16, mask=True),
                camera.fx,
                camera.fy,
                camera.cx,
                camera.cy,
                [],
                mask=True,
            )
            > 0.5
        )
        fit_window = bridge._mask_fit_window(mask)
        fit_x, fit_y, fit_width, fit_height = fit_window
        view_inputs[view_id] = {
            "view_id": view_id,
            "seed": index,
            # Deliberately nonexistent: a successful bridge proves RGB was never accessed.
            "rgb": {
                "path": f"rgb/DO_NOT_OPEN_{view_id}.jpg",
                "sha256": "7" * 64,
            },
            "mask": {
                "path": mask_relative,
                "sha256": bridge.sha256_file(mask_path),
            },
            "camera": bridge._camera_record(camera),
            "distortion": [],
            "native_resolution": [16, 16],
            "fit_window": fit_window,
            "full_canvas_pixels": 256,
            "fit_crop_pixels": fit_width * fit_height,
            "foreground_pixels_full": int(mask.sum()),
            "foreground_pixels_crop": int(
                mask[fit_y : fit_y + fit_height, fit_x : fit_x + fit_width].sum()
            ),
            "undistorted_rgb_tensor_sha256": "8" * 64,
            "undistorted_mask_tensor_sha256": bridge.tensor_hash(mask),
            "masked_target_crop_tensor_sha256": "9" * 64,
        }
        configs[view_id] = {
            arm["name"]: {
                "requested": {"arbitrary": index},
                "digests": {"fit_sha256": "b" * 64},
            }
        }
        fields[view_id] = _field(
            view_id,
            count=index + 2,
            fit_window=fit_window,
        )

    plan = {
        "artifact_type": bridge.SOURCE_PLAN_TYPE,
        "decision_bearing": False,
        "scope": "synthetic Stage-1 fixture",
        "training_view_universe": list(bridge.TRAIN_VIEWS),
        "selected_views": list(bridge.TRAIN_VIEWS),
        "heldout_view_excluded": bridge.HELDOUT_VIEW,
        "arms": [arm],
        "inputs": {
            "scene": "synthetic",
            "calibration": {"path": "calibration.json", "sha256": "c" * 64},
            "views": view_inputs,
        },
        "effective_configs": configs,
        "repository": repository,
        "external_structsplat": external,
        "environment_at_plan": environment,
        "loaded_cuda_extension": extension,
    }
    plan_path = source / "plan.json"
    _write_json(plan_path, plan)
    plan_sha256 = bridge.sha256_file(plan_path)

    rankings: dict[str, list[dict]] = {}
    contact_sheets: dict[str, dict[str, str]] = {}
    for view_id in bridge.TRAIN_VIEWS:
        teacher_relative = f"teachers/{view_id}__{arm['name']}.teacher.npz"
        teacher_path = source / teacher_relative
        fields[view_id].save_npz(teacher_path)
        panel_relative = f"panels/{view_id}__{arm['name']}.png"
        panel_path = source / panel_relative
        panel_path.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic-panel-" + view_id.encode())
        record = {
            "artifact_type": bridge.SOURCE_CELL_TYPE,
            "status": "PASS",
            "plan_sha256": plan_sha256,
            "view_id": view_id,
            "arm": arm,
            "effective_config": configs[view_id][arm["name"]],
            "input": view_inputs[view_id],
            "teacher": bridge.observation_summary(fields[view_id]),
            "teacher_path": teacher_relative,
            "teacher_sha256": bridge.sha256_file(teacher_path),
            "optimization": {
                "m_init_2d": fields[view_id].n_init,
                "m_opt_2d": fields[view_id].n,
            },
            "panel": {
                "path": panel_relative,
                "sha256": bridge.sha256_file(panel_path),
            },
            "environment": environment,
            "external_structsplat": external,
            "loaded_cuda_extension": extension,
        }
        _write_json(source / "records" / f"{view_id}__{arm['name']}.json", record)
        contact_relative = f"{view_id}_capacity_contact_sheet.png"
        contact_path = source / contact_relative
        contact_path.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic-contact-" + view_id.encode())
        contact_sheets[view_id] = {
            "path": contact_relative,
            "sha256": bridge.sha256_file(contact_path),
        }
        rankings[view_id] = [
            {
                "arm": arm["name"],
                "m_init_2d": fields[view_id].n_init,
                "m_opt_2d": fields[view_id].n,
            }
        ]
    result = {
        "artifact_type": bridge.SOURCE_RESULT_TYPE,
        "status": "PASS",
        "plan_sha256": plan_sha256,
        "completed_cell_count": len(bridge.TRAIN_VIEWS),
        "expected_cell_count": len(bridge.TRAIN_VIEWS),
        "missing_cells": [],
        "selected_views": list(bridge.TRAIN_VIEWS),
        "heldout_view_excluded": bridge.HELDOUT_VIEW,
        "rankings": rankings,
        "contact_sheets": contact_sheets,
    }
    _write_json(source / "result.json", result)
    return source, data_root


def _rebind_source_plan(source: Path, mutate) -> None:
    plan_path = source / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    mutate(plan)
    _write_json(plan_path, plan)
    plan_sha256 = bridge.sha256_file(plan_path)
    arm_name = plan["arms"][0]["name"]
    for view_id in bridge.TRAIN_VIEWS:
        record_path = source / "records" / f"{view_id}__{arm_name}.json"
        if not record_path.exists():
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["plan_sha256"] = plan_sha256
        if view_id in plan["inputs"]["views"]:
            record["input"] = plan["inputs"]["views"][view_id]
        _write_json(record_path, record)
    result_path = source / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["plan_sha256"] = plan_sha256
    _write_json(result_path, result)


def test_bridge_preserves_variable_counts_residuals_bounds_and_denies_rgb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _ = _make_source(tmp_path, monkeypatch)
    original_resize = bridge._resize_image
    decoded: list[Path] = []

    def mask_only_resize(path, *args, **kwargs):
        candidate = Path(path)
        assert candidate.suffix.lower() == ".png"
        decoded.append(candidate)
        return original_resize(path, *args, **kwargs)

    monkeypatch.setattr(bridge, "_resize_image", mask_only_resize)
    out = tmp_path / "bridge"
    plan = bridge.prepare_plan(source, out, argv=["plan-test"])
    assert len(decoded) == len(bridge.TRAIN_VIEWS)
    assert plan["source_sweep"]["n_opt_2d"] == list(range(2, 9))

    monkeypatch.setattr(
        bridge,
        "_resize_image",
        lambda *_args, **_kwargs: pytest.fail("build decoded a raster"),
    )
    result = bridge.build_bundle(out, argv=["build-test"])
    assert result["n_opt_2d"] == list(range(2, 9))
    assert result["total_m_opt_2d"] == sum(range(2, 9))
    assert result["bundle"]["contains_dense_rgb_mask_or_source_path"] is False
    loaded = ReconstructionInputs.load(
        out / "reconstruction_inputs",
        device="cpu",
        strict=True,
    )
    assert loaded.bounds_hint is not None
    assert loaded.n_opt_2d == list(range(2, 9))
    assert all(field.mean_residuals is not None for field in loaded.observations)
    manifest = (out / "reconstruction_inputs" / "manifest.json").read_text().lower()
    assert all(token not in manifest for token in ("rgb", "mask", "image_path", "source_path"))


@pytest.mark.parametrize(
    "replacement",
    [
        list(bridge.TRAIN_VIEWS[:-1]),
        [*bridge.TRAIN_VIEWS, bridge.HELDOUT_VIEW],
    ],
)
def test_source_missing_or_extra_views_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: list[str],
) -> None:
    source, _ = _make_source(tmp_path, monkeypatch)
    _rebind_source_plan(
        source,
        lambda plan: plan.__setitem__("selected_views", replacement),
    )
    with pytest.raises(ValueError, match="select all seven training views"):
        bridge.verify_source_sweep(source)


@pytest.mark.parametrize("target", ["plan", "cell", "teacher"])
def test_source_plan_cell_and_teacher_hash_drift_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    source, _ = _make_source(tmp_path, monkeypatch)
    view_id = bridge.TRAIN_VIEWS[0]
    arm_name = "arbitrary_adaptive_arm"
    if target == "plan":
        with (source / "plan.json").open("ab") as stream:
            stream.write(b" ")
        expected = "different plan"
    elif target == "cell":
        record_path = source / "records" / f"{view_id}__{arm_name}.json"
        record = json.loads(record_path.read_text())
        record["plan_sha256"] = "0" * 64
        _write_json(record_path, record)
        expected = "different plan"
    else:
        with (source / "teachers" / f"{view_id}__{arm_name}.teacher.npz").open("ab") as stream:
            stream.write(b"tamper")
        expected = "teacher digest mismatch"
    with pytest.raises(ValueError, match=expected):
        bridge.verify_source_sweep(source)


def test_residual_array_semantic_drift_fails_even_with_updated_file_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _ = _make_source(tmp_path, monkeypatch)
    view_id = bridge.TRAIN_VIEWS[0]
    arm_name = "arbitrary_adaptive_arm"
    teacher_path = source / "teachers" / f"{view_id}__{arm_name}.teacher.npz"
    field = GaussianObservationField.load_npz(teacher_path, strict=True)
    teacher_path.unlink()
    changed = dataclasses.replace(
        field,
        mean_residuals=field.mean_residuals + torch.tensor([0.01, -0.01]),
    )
    changed.save_npz(teacher_path)
    record_path = source / "records" / f"{view_id}__{arm_name}.json"
    record = json.loads(record_path.read_text())
    record["teacher_sha256"] = bridge.sha256_file(teacher_path)
    _write_json(record_path, record)
    with pytest.raises(ValueError, match="semantics/residual arrays changed"):
        bridge.verify_source_sweep(source)


def test_mask_binding_cannot_alias_rgb_before_decoder_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _ = _make_source(tmp_path, monkeypatch)
    sweep = bridge.verify_source_sweep(source)
    first = bridge.TRAIN_VIEWS[0]
    sweep.plan["inputs"]["views"][first]["mask"]["path"] = sweep.plan["inputs"]["views"][first][
        "rgb"
    ]["path"]
    monkeypatch.setattr(
        bridge,
        "_resize_image",
        lambda *_args, **_kwargs: pytest.fail("RGB alias reached the decoder"),
    )
    with pytest.raises(ValueError, match="aliases or resembles source RGB"):
        bridge.acquire_bound_masks(sweep)


def test_source_binding_and_extra_cell_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _ = _make_source(tmp_path, monkeypatch)
    view_id = bridge.TRAIN_VIEWS[0]
    record_path = source / "records" / f"{view_id}__arbitrary_adaptive_arm.json"
    record = json.loads(record_path.read_text())
    record["external_structsplat"]["version"] = "drifted"
    _write_json(record_path, record)
    with pytest.raises(ValueError, match="StructSplat binding differs"):
        bridge.verify_source_sweep(source)

    source, _ = _make_source(tmp_path / "second", monkeypatch)
    (source / "records" / f"{bridge.HELDOUT_VIEW}__arbitrary_adaptive_arm.json").write_text("{}")
    with pytest.raises(ValueError, match="missing or extra views/cells"):
        bridge.verify_source_sweep(source)


def test_build_failure_is_append_only_and_does_not_commit_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _ = _make_source(tmp_path, monkeypatch)
    out = tmp_path / "bridge"
    bridge.prepare_plan(source, out, argv=["plan-test"])
    teacher = source / "teachers" / "C0001__arbitrary_adaptive_arm.teacher.npz"
    with teacher.open("ab") as stream:
        stream.write(b"tamper-after-plan")
    with pytest.raises(RuntimeError, match="source sweep tree changed"):
        bridge.build_bundle(out, argv=["build-test"])
    failures = sorted((out / "failures").glob("*.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text())["phase"] == "build"
    assert not (out / "reconstruction_inputs").exists()
