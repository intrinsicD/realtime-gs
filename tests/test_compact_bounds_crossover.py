from __future__ import annotations

import pytest
import torch
from benchmarks import compact_bounds_crossover as experiment

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.reconstruction_inputs import ReconstructionInputs


def _camera(eye: tuple[float, float, float], width: int = 8, height: int = 8) -> Camera:
    return Camera.look_at(
        torch.tensor(eye),
        torch.tensor([0.0, 0.0, 0.0]),
        width=width,
        height=height,
        fov_x_deg=60.0,
    )


def _field(view_id: str) -> GaussianObservationField:
    return GaussianObservationField(
        width=8,
        height=8,
        means=torch.tensor([[3.5, 3.5], [4.5, 4.5]]),
        log_scales=torch.zeros(2, 2),
        rotations=torch.zeros(2),
        colors=torch.full((2, 3), 0.5),
        amplitudes=torch.ones(2),
        view_id=view_id,
        n_init=2,
        provider="synthetic_fixture",
        producer_version="bounds-test",
        producer_source_digest="a" * 64,
        fit_config_digest="b" * 64,
    )


def _inputs(view_names: tuple[str, ...] = ("T0", "T1", "T2")) -> ReconstructionInputs:
    cameras = [
        _camera((2.0, 0.0, 0.5)),
        _camera((-1.0, 1.8, 0.5)),
        _camera((-1.0, -1.8, 0.5)),
    ][: len(view_names)]
    return ReconstructionInputs(
        observations=[_field(name) for name in view_names],
        cameras=cameras,
        view_names=list(view_names),
        points=None,
        point_visibility=None,
        bounds_hint=None,
        name="tiny_bounds_crossover",
    )


def _masks(count: int) -> list[torch.Tensor]:
    result = []
    for index in range(count):
        mask = torch.zeros(8, 8)
        mask[2:6, 2 + index % 2 : 6 + index % 2] = 1.0
        result.append(mask)
    return result


def test_bounds_arm_construction_changes_only_bounds_hint() -> None:
    inputs = _inputs()
    arms = experiment.build_bounds_arm_inputs(
        inputs,
        _masks(inputs.n_views),
        heldout_view="H0",
    )

    fallback = arms["A_fallback_bounds"]
    masked = arms["B_same7_mask_bounds"]
    assert fallback.bounds_hint is None
    assert masked.bounds_hint is not None
    center, extent = masked.bounds_hint
    assert center.shape == (3,)
    assert torch.isfinite(center).all()
    assert extent > 0
    assert fallback.observations == masked.observations == inputs.observations
    assert fallback.cameras == masked.cameras == inputs.cameras
    assert fallback.view_names == masked.view_names == inputs.view_names
    assert all(
        left is right
        for left, right in zip(fallback.observations, masked.observations, strict=True)
    )
    assert all(left is right for left, right in zip(fallback.cameras, masked.cameras, strict=True))

    config = experiment.base_config(n_init_3d=2)
    fallback_record = experiment.bounds_record(fallback, config)
    masked_record = experiment.bounds_record(masked, config)
    assert fallback_record["source"] == "camera_fallback"
    assert masked_record["source"] == "bounds_hint"
    assert fallback_record["center_sha256"] != masked_record["center_sha256"]


def test_heldout_view_is_rejected_before_bounds_construction() -> None:
    inputs = _inputs(("T0", "H0", "T2"))
    with pytest.raises(ValueError, match="held-out view H0"):
        experiment.build_bounds_arm_inputs(
            inputs,
            _masks(inputs.n_views),
            heldout_view="H0",
        )


def test_frozen_selection_protocol_excludes_c1004() -> None:
    experiment.validate_selection_views(experiment.TRAIN_VIEWS)
    leaked = (*experiment.TRAIN_VIEWS[:-1], experiment.HELDOUT_VIEW)
    with pytest.raises(ValueError, match="held-out view C1004"):
        experiment.validate_selection_views(leaked)


def test_component_center_candidate_contract_uses_every_component_once() -> None:
    inputs = _inputs()
    config = experiment.base_config(n_init_3d=2)
    contract = experiment.candidate_contract(inputs, config)
    assert contract["candidate_count"] == 6
    assert contract["per_view"] == [2, 2, 2]
    assert contract["attempt_count"] == 6
    assert contract["samples_per_ray"] == experiment.SAMPLES_PER_RAY
    assert all(contract["identity_checks"].values())
