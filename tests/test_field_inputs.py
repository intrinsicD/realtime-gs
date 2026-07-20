"""CPU tests for the image-free field-lift input boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.compact_views import CompactDataset, CompactView, PackedAlpha
from rtgs.data.field_inputs import SceneFits
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.data.scene import SceneData


def _camera(offset: float = 0.0, *, width: int = 8, height: int = 6) -> Camera:
    return Camera.look_at(
        torch.tensor([offset, 0.0, -3.0]),
        torch.zeros(3),
        width=width,
        height=height,
    )


def _observation(name: str, *, width: int = 8, height: int = 6) -> GaussianObservationField:
    return GaussianObservationField(
        width=width,
        height=height,
        means=torch.tensor([[2.5, 2.5], [5.0, 3.5]]),
        log_scales=torch.log(torch.tensor([[0.8, 1.1], [1.2, 0.7]])),
        rotations=torch.tensor([0.25, -0.4]),
        colors=torch.tensor([[0.2, 0.4, 0.7], [0.8, 0.3, 0.1]]),
        amplitudes=torch.tensor([0.6, 0.4]),
        blend_mode="additive",
        view_id=name,
        fit_window=(1, 1, 4, 3),
        provider="synthetic_fixture",
    )


def _legacy_fit(shift: float = 0.0) -> Gaussians2D:
    return Gaussians2D(
        xy=torch.tensor([[2.5 + shift, 2.0], [5.0 + shift, 3.5]]),
        chol=torch.tensor([[1.2, 0.25, 0.7], [0.8, -0.1, 1.1]]),
        color=torch.tensor([[0.2, 0.3, 0.4], [0.7, 0.6, 0.5]]),
        weight=torch.tensor([0.8, 0.35]),
    )


def _effective_covariances(field: GaussianObservationField) -> torch.Tensor:
    variances = field.effective_variances()
    cosine = torch.cos(field.rotations)
    sine = torch.sin(field.rotations)
    rotation = torch.stack(
        [
            torch.stack([cosine, -sine], dim=-1),
            torch.stack([sine, cosine], dim=-1),
        ],
        dim=-2,
    )
    return rotation @ torch.diag_embed(variances) @ rotation.transpose(-1, -2)


def test_compact_adapter_preserves_packed_alpha_and_all_optional_evidence(tmp_path: Path) -> None:
    from rtgs.data import SceneFits as PublicSceneFits

    assert PublicSceneFits is SceneFits
    observations = (_observation("left"), _observation("right"))
    cameras = (_camera(-0.2), _camera(0.2))
    alpha = PackedAlpha(
        payload=b"\x2d\x03",
        shape=(3, 4),
        origin=(1, 1),
        foreground_count=6,
    )
    views = [
        CompactView(
            observation=observations[0],
            camera=cameras[0],
            alpha=alpha,
            calibration_sha256="a" * 64,
            source={},
            path=tmp_path / "left.rtgsv",
            bytes=100,
            sha256="b" * 64,
        ),
        CompactView(
            observation=observations[1],
            camera=cameras[1],
            alpha=None,
            calibration_sha256="a" * 64,
            source={},
            path=tmp_path / "right.rtgsv",
            bytes=100,
            sha256="c" * 64,
        ),
    ]
    bounds = (torch.tensor([0.1, -0.2, 0.3]), 2.5)
    dataset = CompactDataset(
        views=views,
        name="compact",
        calibration_sha256="a" * 64,
        bounds_hint=bounds,
        path=tmp_path,
    )
    priors = (torch.tensor([1.0, 1.5]), None)
    confidence = (torch.tensor([0.8, 0.25]), None)

    fits = SceneFits.from_compact_dataset(
        dataset,
        train_view_indices=(0,),
        heldout_view_indices=(1,),
        depth_priors=priors,
        depth_confidences=confidence,
        neighbors=((1,), (0,)),
    )

    assert fits.observations[0] is observations[0]
    assert fits.cameras[1] is cameras[1]
    assert fits.alphas[0] is alpha
    assert fits.alphas[1] is None
    assert fits.train_view_indices == (0,)
    assert fits.heldout_view_indices == (1,)
    assert fits.training_observations == (observations[0],)
    assert fits.depth_priors == priors
    assert fits.depth_confidences == confidence
    assert fits.neighbors == ((1,), (0,))
    assert fits.bounds_hint is bounds
    assert not fits.geometry_is_train_only
    assert not hasattr(fits, "images")


def test_reconstruction_adapter_carries_sparse_geometry_and_derives_split() -> None:
    points = torch.tensor([[0.0, 0.0, 0.0], [0.2, -0.1, 0.4]])
    visibility = [torch.tensor([0, 1]), torch.tensor([1])]
    inputs = ReconstructionInputs(
        observations=[_observation("left"), _observation("right")],
        cameras=[_camera(-0.2), _camera(0.2)],
        view_names=["left", "right"],
        points=points,
        point_visibility=visibility,
        bounds_hint=(torch.zeros(3), 2.0),
        name="sparse",
    )
    fits = SceneFits.from_reconstruction_inputs(
        inputs,
        heldout_view_indices=(1,),
        geometry_is_train_only=True,
    )

    assert fits.train_view_indices == (0,)
    assert fits.heldout_view_indices == (1,)
    assert fits.points is points
    assert fits.point_visibility == tuple(visibility)
    assert fits.alphas == (None, None)
    assert fits.geometry_is_train_only


def test_legacy_adapter_preserves_covariance_amplitude_masks_and_scene_split() -> None:
    fits2d = [_legacy_fit(), _legacy_fit(0.1), _legacy_fit(-0.1)]
    cameras = [_camera(-0.3), _camera(), _camera(0.3)]
    masks = [
        torch.ones(6, 8, dtype=torch.bool),
        torch.eye(6, 8, dtype=torch.bool),
        torch.zeros(6, 8, dtype=torch.bool),
    ]
    scene = SceneData(
        images=[torch.zeros(6, 8, 3) for _ in range(3)],
        cameras=cameras,
        view_names=["a", "b", "held"],
        masks=masks,
        train_indices=[0, 1],
        test_indices=[2],
        name="legacy",
    )

    fits = SceneFits.from_legacy(fits2d, scene)

    assert fits.train_view_indices == (0, 1)
    assert fits.heldout_view_indices == (2,)
    assert fits.view_names == ("a", "b", "held")
    assert all(field.blend_mode == "additive" for field in fits.observations)
    for source, field in zip(fits2d, fits.observations, strict=True):
        torch.testing.assert_close(field.means, source.xy, rtol=0, atol=0)
        torch.testing.assert_close(field.colors, source.color, rtol=0, atol=0)
        torch.testing.assert_close(field.amplitudes, source.weight, rtol=0, atol=0)
        torch.testing.assert_close(
            _effective_covariances(field),
            source.covariance(),
            rtol=2e-6,
            atol=2e-6,
        )
    assert all(torch.equal(actual, expected) for actual, expected in zip(fits.alphas, masks))
    assert all(actual is not expected for actual, expected in zip(fits.alphas, masks))
    assert not hasattr(fits, "images")


def test_training_only_legacy_fits_become_an_explicit_local_training_partition() -> None:
    scene = SceneData(
        images=[torch.zeros(6, 8, 3) for _ in range(3)],
        cameras=[_camera(-0.2), _camera(), _camera(0.2)],
        view_names=["a", "b", "held"],
        train_indices=[0, 1],
        test_indices=[2],
    )
    fits = SceneFits.from_legacy([_legacy_fit(), _legacy_fit()], scene)
    assert fits.view_names == ("a", "b")
    assert fits.train_view_indices == (0, 1)
    assert fits.heldout_view_indices == ()


def test_scene_fits_rejects_split_neighbor_and_prior_misalignment() -> None:
    observation = _observation("only")
    base = dict(
        observations=(observation,),
        cameras=(_camera(),),
        view_names=("only",),
        alphas=(None,),
        train_view_indices=(0,),
        heldout_view_indices=(),
    )
    with pytest.raises(ValueError, match="disjoint"):
        SceneFits(**{**base, "train_view_indices": (0,), "heldout_view_indices": (0,)})
    with pytest.raises(ValueError, match="non-self"):
        SceneFits(**base, neighbors=((0,),))
    with pytest.raises(ValueError, match="supplied together"):
        SceneFits(**base, depth_priors=(torch.ones(observation.n),))
    with pytest.raises(ValueError, match="cardinality"):
        SceneFits(
            **base,
            depth_priors=(torch.ones(observation.n + 1),),
            depth_confidences=(torch.ones(observation.n + 1),),
        )
    with pytest.raises(TypeError, match="geometry_is_train_only"):
        SceneFits(**base, geometry_is_train_only=1)
