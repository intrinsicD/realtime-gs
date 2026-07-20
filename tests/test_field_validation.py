from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.field_inputs import SceneFits
from rtgs.lift.field_validation import (
    FieldValidationConfig,
    validate_field_semantics,
)


def _camera(*, tx: float = 0.0) -> Camera:
    return Camera(
        R=torch.eye(3),
        t=torch.tensor([tx, 0.0, 0.0]),
        fx=24.0,
        fy=24.0,
        cx=7.5,
        cy=5.5,
        width=16,
        height=12,
    )


def _gaussians(*, opacity: float = 0.02) -> Gaussians3D:
    means = torch.tensor([[0.05, -0.03, 3.0]])
    covs = torch.diag_embed(torch.tensor([[0.018, 0.012, 0.025]]))
    colors = torch.tensor([[0.28, 0.46, 0.67]])
    return Gaussians3D.from_means_covs(
        means,
        covs,
        colors,
        opacity=torch.tensor([opacity]),
    )


def _template(
    *,
    blend_mode: str = "normalized",
) -> GaussianObservationField:
    return GaussianObservationField(
        width=16,
        height=12,
        means=torch.tensor([[7.5, 5.5]]),
        log_scales=torch.log(torch.tensor([[1.0, 1.0]])),
        rotations=torch.zeros(1),
        colors=torch.tensor([[0.0, 0.0, 0.0]]),
        amplitudes=torch.ones(1),
        provider="synthetic_fixture",
        view_id="template",
        sigma_cutoff=3.1,
        support_fade_alpha=0.35,
        aa_dilation=0.2,
        blend_mode=blend_mode,
        epsilon=2e-5,
        fit_window=(1, 1, 14, 10),
    )


def _projected_teacher(
    camera: Camera,
    *,
    blend_mode: str = "normalized",
    mass: float = 0.8,
) -> GaussianObservationField:
    from rtgs.lift.field_validation import _projected_field

    field = _projected_field(
        _gaussians(),
        camera,
        torch.tensor([mass]),
        _template(blend_mode=blend_mode),
    )
    assert field is not None
    return replace(field, view_id=None)


def _fits(
    observations: tuple[GaussianObservationField, ...],
    cameras: tuple[Camera, ...],
    *,
    train: tuple[int, ...] | None = None,
    heldout: tuple[int, ...] = (),
) -> SceneFits:
    if train is None:
        train = tuple(range(len(observations)))
    return SceneFits(
        observations=observations,
        cameras=cameras,
        view_names=tuple(f"view-{i}" for i in range(len(observations))),
        alphas=(None,) * len(observations),
        train_view_indices=train,
        heldout_view_indices=heldout,
    )


@pytest.mark.parametrize("blend_mode", ["normalized", "additive"])
def test_exact_query_semantics_match_projected_field(blend_mode: str) -> None:
    camera = _camera()
    teacher = _projected_teacher(camera, blend_mode=blend_mode)
    fits = _fits((teacher,), (camera,))

    result = validate_field_semantics(
        fits,
        (camera,),
        _gaussians(opacity=0.999),
        torch.tensor([0.8]),
        config=FieldValidationConfig(sample_cap_per_view=79, seed=4),
    )

    assert result.train.density_mse == pytest.approx(0.0, abs=1e-13)
    assert result.train.rgb_mse == pytest.approx(0.0, abs=1e-13)
    assert result.heldout is None


def test_affine_teacher_colors_are_queried_at_sample_locations() -> None:
    camera = _camera()
    constant = _projected_teacher(camera)
    affine = replace(
        constant,
        color_grads=torch.tensor([[[0.025, 0.01, -0.02], [-0.015, 0.02, 0.03]]]),
    )
    config = FieldValidationConfig(sample_cap_per_view=71, seed=12)

    constant_result = validate_field_semantics(
        _fits((constant,), (camera,)),
        (camera,),
        _gaussians(),
        torch.tensor([0.8]),
        config=config,
    )
    affine_result = validate_field_semantics(
        _fits((affine,), (camera,)),
        (camera,),
        _gaussians(),
        torch.tensor([0.8]),
        config=config,
    )

    assert constant_result.train.density_mse == pytest.approx(0.0, abs=1e-13)
    assert affine_result.train.density_mse == pytest.approx(0.0, abs=1e-13)
    assert constant_result.train.rgb_mse == pytest.approx(0.0, abs=1e-13)
    assert affine_result.train.rgb_mse > 1e-5


def test_sampling_is_bounded_and_deterministic() -> None:
    camera = _camera()
    teacher = _projected_teacher(camera)
    fits = _fits((teacher,), (camera,))
    kwargs = (_gaussians(), torch.tensor([0.8]))

    first = validate_field_semantics(
        fits,
        (camera,),
        *kwargs,
        config=FieldValidationConfig(sample_cap_per_view=23, seed=55),
    )
    second = validate_field_semantics(
        fits,
        (camera,),
        *kwargs,
        config=FieldValidationConfig(sample_cap_per_view=23, seed=55),
    )
    different_seed = validate_field_semantics(
        fits,
        (camera,),
        *kwargs,
        config=FieldValidationConfig(sample_cap_per_view=23, seed=56),
    )

    assert first == second
    assert first.per_view[0].n_samples == 23
    assert first.per_view[0].sample_sha256 != different_seed.per_view[0].sample_sha256


def test_heldout_views_are_evaluation_only() -> None:
    cameras = (_camera(), _camera(tx=-0.12))
    teachers = tuple(_projected_teacher(camera) for camera in cameras)
    base_fits = _fits(
        teachers,
        cameras,
        train=(0,),
        heldout=(1,),
    )
    poisoned_heldout = replace(
        teachers[1],
        colors=teachers[1].colors + torch.tensor([[0.3, -0.2, 0.1]]),
        amplitudes=teachers[1].amplitudes * 1.7,
    )
    poisoned_fits = _fits(
        (teachers[0], poisoned_heldout),
        cameras,
        train=(0,),
        heldout=(1,),
    )
    config = FieldValidationConfig(sample_cap_per_view=53, seed=8)
    args = (cameras, _gaussians(), torch.tensor([0.8]))

    base = validate_field_semantics(base_fits, *args, config=config)
    poisoned = validate_field_semantics(poisoned_fits, *args, config=config)

    assert base.train == poisoned.train
    assert base.per_view[0] == poisoned.per_view[0]
    assert base.heldout is not None
    assert poisoned.heldout is not None
    assert poisoned.heldout.density_mse > base.heldout.density_mse
    assert poisoned.heldout.rgb_mse > base.heldout.rgb_mse
    assert tuple(metric.split for metric in poisoned.per_view) == (
        "train",
        "heldout",
    )
