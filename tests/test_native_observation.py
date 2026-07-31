"""Tests for freezing native additive Stage-1 fits as compact fields."""

from __future__ import annotations

import math

import pytest
import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.image2gs.native_observation import (
    native_gaussians_to_observation,
    native_observation_to_gaussians,
    render_native_observation_crop,
)
from rtgs.image2gs.renderer2d import render_gaussians_2d


def _gaussians(dtype: torch.dtype = torch.float32) -> Gaussians2D:
    return Gaussians2D(
        xy=torch.tensor([[0.625123, 1.875321], [4.25, 3.5]], dtype=dtype),
        chol=torch.tensor([[1.2, 0.2, 0.8], [0.7, -0.1, 1.1]], dtype=dtype),
        color=torch.tensor([[0.2, 0.4, 0.6], [0.7, 0.3, 0.1]], dtype=dtype),
        weight=torch.tensor([0.8, 0.35], dtype=dtype),
    )


def test_native_observation_preserves_additive_parameters_and_native_scale_means() -> None:
    source = _gaussians()
    field = native_gaussians_to_observation(
        source,
        canvas_size=(5000, 6000),
        fit_window=(4097, 3073, 8, 6),
        view_id="C0001",
        producer_version="fixture",
        producer_source_digest="a" * 64,
        fit_config_digest="b" * 64,
    )

    assert field.provider == "native"
    assert field.blend_mode == "additive"
    assert field.support_fade_alpha == 0.0
    assert field.aa_dilation == 0.0
    assert field.sigma_cutoff == pytest.approx(math.sqrt(12.0))
    assert field.fit_window == (4097, 3073, 8, 6)
    assert field.n_init == source.n
    assert field.mean_residuals is not None
    torch.testing.assert_close(field.colors, source.color, rtol=0, atol=0)
    torch.testing.assert_close(field.amplitudes, source.weight, rtol=0, atol=0)
    expected_means = source.xy.double() + torch.tensor([4097.0, 3073.0], dtype=torch.float64)
    torch.testing.assert_close(
        field.native_means(dtype=torch.float64), expected_means, rtol=0, atol=0
    )

    rotation = field.rotations
    cos, sin = torch.cos(rotation), torch.sin(rotation)
    axes = torch.stack(
        [
            torch.stack([cos, -sin], dim=-1),
            torch.stack([sin, cos], dim=-1),
        ],
        dim=-2,
    )
    covariance = axes @ torch.diag_embed(field.scales().square()) @ axes.transpose(-1, -2)
    torch.testing.assert_close(covariance, source.covariance(), rtol=2e-6, atol=2e-6)


def test_native_observation_round_trips_provider_and_provenance(tmp_path) -> None:
    path = tmp_path / "native.npz"
    field = native_gaussians_to_observation(
        _gaussians(torch.float64),
        canvas_size=(6, 8),
        view_id="C0001",
        producer_version="fixture",
        producer_source_digest="a" * 64,
        fit_config_digest="b" * 64,
    )
    field.save_npz(path)
    loaded = type(field).load_npz(path, strict=True)

    assert loaded.provider == "native"
    assert loaded.producer_version == "fixture"
    assert loaded.producer_source_digest == "a" * 64
    assert loaded.fit_config_digest == "b" * 64
    assert loaded.mean_residuals is None
    torch.testing.assert_close(loaded.means, field.means, rtol=0, atol=0)


def test_native_observation_replays_original_additive_crop_without_structsplat() -> None:
    source = _gaussians()
    field = native_gaussians_to_observation(
        source,
        canvas_size=(6, 8),
        fit_window=(0, 0, 8, 6),
        view_id="C0001",
    )

    recovered = native_observation_to_gaussians(field)
    torch.testing.assert_close(recovered.xy, source.xy, rtol=0, atol=1e-6)
    torch.testing.assert_close(
        recovered.covariance(),
        source.covariance(),
        rtol=3e-6,
        atol=3e-6,
    )
    torch.testing.assert_close(
        render_native_observation_crop(field, renderer="torch"),
        render_gaussians_2d(source, 6, 8, renderer="torch"),
        rtol=3e-6,
        atol=3e-6,
    )


def test_native_observation_rejects_crop_local_means_outside_fit_window() -> None:
    source = _gaussians()
    with pytest.raises(ValueError, match="inside the local fit_window"):
        native_gaussians_to_observation(
            source,
            canvas_size=(20, 20),
            fit_window=(2, 3, 4, 4),
            view_id="C0001",
        )
