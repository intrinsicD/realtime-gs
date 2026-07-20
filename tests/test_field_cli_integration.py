"""Native pipeline and compact CLI integration for image-free field lifting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rtgs.cli import _field_lift_config, _field_split
from rtgs.cli import main as cli_main
from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.compact_views import CompactDataset, CompactView
from rtgs.data.field_inputs import SceneFits
from rtgs.lift import FieldLifter, get_lifter
from rtgs.lift.field_lifter import FieldLiftConfig
from rtgs.pipeline import run_field_pipeline


def _camera(offset: float) -> Camera:
    return Camera.look_at(
        torch.tensor([offset, 0.0, -3.0]),
        torch.zeros(3),
        width=12,
        height=10,
    )


def _observation(name: str, shift: float) -> GaussianObservationField:
    return GaussianObservationField(
        width=12,
        height=10,
        means=torch.tensor([[4.0 + shift, 4.0], [7.0 + shift, 5.5]]),
        log_scales=torch.log(torch.tensor([[0.9, 1.1], [1.2, 0.8]])),
        rotations=torch.tensor([0.15, -0.25]),
        colors=torch.tensor([[0.2, 0.5, 0.8], [0.8, 0.3, 0.15]]),
        amplitudes=torch.tensor([0.7, 0.45]),
        blend_mode="additive",
        view_id=name,
        provider="synthetic_fixture",
    )


def _dataset(tmp_path: Path, n_views: int = 3) -> CompactDataset:
    views = []
    for index in range(n_views):
        name = f"view-{index}"
        views.append(
            CompactView(
                observation=_observation(name, 0.1 * index),
                camera=_camera(-0.2 + 0.2 * index),
                alpha=None,
                calibration_sha256="a" * 64,
                source={},
                path=tmp_path / f"{name}.rtgsv",
                bytes=100,
                sha256=chr(ord("b") + index) * 64,
            )
        )
    return CompactDataset(
        views=views,
        name="compact-fixture",
        calibration_sha256="a" * 64,
        bounds_hint=(torch.zeros(3), 2.0),
        path=tmp_path,
    )


def _fast_config() -> FieldLiftConfig:
    return _field_lift_config(
        json.dumps(
            {
                "max_tracks": 2,
                "max_train_views": 2,
                "depth_samples": 2,
                "candidate_multiplier": 1,
                "min_views": 1,
                "topology_rounds": 0,
                "refit": {"iterations": 0, "appearance_start": 0},
            }
        )
    )


def test_field_lifter_is_registered() -> None:
    assert isinstance(get_lifter("field", config=_fast_config()), FieldLifter)


def test_run_field_pipeline_uses_explicit_split_without_images(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    fits = SceneFits.from_compact_dataset(
        dataset,
        train_view_indices=(1, 2),
        heldout_view_indices=(0,),
    )
    result = run_field_pipeline(fits, _fast_config())
    assert result.gaussians.n > 0
    assert result.optimized_view_indices == (1, 2)
    assert result.heldout_view_indices == (0,)
    assert not hasattr(fits, "images")


def test_lift_field_cli_loads_compact_data_and_writes_diagnostics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = _dataset(tmp_path)
    loaded: dict[str, object] = {}

    def fake_load(path, device="cpu", **_kwargs):
        loaded["path"] = Path(path)
        loaded["device"] = device
        return dataset

    monkeypatch.setattr(CompactDataset, "load", fake_load)
    out = tmp_path / "field.npz"
    rc = cli_main(
        [
            "lift-field",
            "--dataset",
            str(tmp_path / "gaussians2d"),
            "--heldout-stride",
            "2",
            "--field-args",
            json.dumps(
                {
                    "max_tracks": 2,
                    "max_train_views": 2,
                    "depth_samples": 2,
                    "candidate_multiplier": 1,
                    "min_views": 1,
                    "topology_rounds": 0,
                    "refit": {"iterations": 0, "appearance_start": 0},
                }
            ),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert loaded == {"path": tmp_path / "gaussians2d", "device": "cpu"}
    assert out.is_file()
    diagnostics = json.loads(out.with_suffix(".diagnostics.json").read_text())
    assert diagnostics["dataset"] == "compact-fixture"
    assert diagnostics["train_view_indices"] == [1]
    assert diagnostics["heldout_view_indices"] == [0, 2]
    assert diagnostics["semantic_validation"]["heldout"]["n_views"] == 2
    field_state = out.with_suffix(".field.npz")
    assert diagnostics["field_state"] == str(field_state)
    with np.load(field_state) as state:
        assert state["field_masses"].shape == (2,)
        assert state["covariance_free_mask"].shape == (2,)
        assert state["source_global_view_indices"].shape == (2,)
        assert state["fitting_visibility"].shape == (1, 2)
        assert state["correspondence_visibility"].shape == (3, 2)
        assert state["correspondence_0000"].shape[0] == 2
        assert state["correspondence_0002"].shape[0] == 2


def test_field_split_and_nested_config_validation() -> None:
    assert _field_split(2, 8) == ((0, 1), ())
    assert _field_split(5, 2) == ((1, 3), (0, 2, 4))
    config = _field_lift_config(
        '{"max_tracks": 3, "refit": {"iterations": 0, "appearance_start": 0}}'
    )
    assert config.max_tracks == 3
    assert config.refit.iterations == 0
