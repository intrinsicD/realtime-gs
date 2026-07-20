"""CPU tests for block-fixed field visibility and per-view gains."""

from __future__ import annotations

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.lift.field_visibility import (
    center_transmittance_visibility,
    estimate_view_gains,
    solve_view_gains,
)


def _camera() -> Camera:
    return Camera(
        fx=16.0,
        fy=16.0,
        cx=16.0,
        cy=16.0,
        width=32,
        height=32,
        R=torch.eye(3),
        t=torch.zeros(3),
    )


def _gaussians() -> Gaussians3D:
    means = torch.tensor(
        [
            [0.0, 0.0, 2.0],  # front, centered
            [0.0, 0.0, 3.0],  # back, centered
            [10.0, 0.0, 2.0],  # center offscreen
            [0.0, 0.0, -1.0],  # behind camera
        ],
        dtype=torch.float64,
    )
    covariance = torch.diag(torch.tensor([0.1**2, 0.1**2, 0.05**2], dtype=torch.float64))
    return Gaussians3D.from_means_covs(
        means=means,
        covs=covariance[None].expand(4, -1, -1).clone(),
        colors=torch.full((4, 3), 0.5, dtype=torch.float64),
        opacity=torch.tensor([0.25, 0.7, 0.9, 0.8], dtype=torch.float64),
    )


def test_center_transmittance_occludes_back_and_zeros_invalid_centers() -> None:
    gaussians = _gaussians()
    gaussians.means.requires_grad_(True)
    result = center_transmittance_visibility(
        gaussians,
        (_camera(),),
        target_chunk_size=1,
    )

    assert result.weights.shape == (1, 4)
    assert torch.equal(result.valid, torch.tensor([[True, True, False, False]]))
    assert result.weights[0, 0].item() == pytest.approx(1.0)
    assert result.weights[0, 1].item() == pytest.approx(0.75, abs=1e-12)
    assert result.weights[0, 2].item() == 0.0
    assert result.weights[0, 3].item() == 0.0
    assert not result.weights.requires_grad
    assert not result.means2d.requires_grad


def test_visibility_is_chunk_invariant_permutation_equivariant_and_ties_do_not_occlude() -> None:
    camera = _camera()
    gaussians = _gaussians()
    one = center_transmittance_visibility(
        gaussians,
        (camera,),
        target_chunk_size=1,
    )
    all_at_once = center_transmittance_visibility(
        gaussians,
        (camera,),
        target_chunk_size=gaussians.n,
    )
    torch.testing.assert_close(one.weights, all_at_once.weights, rtol=0, atol=0)

    permutation = torch.tensor([2, 1, 3, 0])
    permuted = gaussians.subset(permutation)
    changed = center_transmittance_visibility(permuted, (camera,))
    inverse = torch.argsort(permutation)
    torch.testing.assert_close(changed.weights[:, inverse], one.weights, rtol=1e-15, atol=1e-15)
    assert torch.equal(changed.valid[:, inverse], one.valid)

    tied = Gaussians3D.from_means_covs(
        means=torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]], dtype=torch.float64),
        covs=torch.eye(3, dtype=torch.float64)[None].expand(2, -1, -1).clone() * 0.01,
        colors=torch.ones(2, 3, dtype=torch.float64),
        opacity=torch.tensor([0.9, 0.8], dtype=torch.float64),
    )
    tied_result = center_transmittance_visibility(tied, (camera,))
    torch.testing.assert_close(
        tied_result.weights,
        torch.ones_like(tied_result.weights),
        rtol=0,
        atol=0,
    )


def test_source_forcing_only_overrides_valid_source_projections() -> None:
    gaussians = _gaussians()
    sources = torch.zeros(gaussians.n, dtype=torch.long)
    result = center_transmittance_visibility(
        gaussians,
        (_camera(),),
        source_view_indices=sources,
        force_source_visible=True,
    )

    assert torch.equal(result.source_forced, torch.tensor([[True, True, False, False]]))
    assert torch.equal(result.weights, torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float64))
    with pytest.raises(ValueError, match="requires source_view_indices"):
        center_transmittance_visibility(
            gaussians,
            (_camera(),),
            force_source_visible=True,
        )


def test_scalar_and_rgb_gains_recover_known_scales_and_are_detached() -> None:
    generator = torch.Generator().manual_seed(812)
    predicted = torch.rand((2, 7, 3), generator=generator, dtype=torch.float64) + 0.1
    scalar_truth = torch.tensor([2.0, 0.5], dtype=torch.float64)
    scalar_target = predicted * scalar_truth[:, None, None]
    scalar = estimate_view_gains(
        predicted,
        scalar_target,
        mode="scalar",
        ridge=1e-12,
    )
    torch.testing.assert_close(scalar.gains, scalar_truth, rtol=1e-11, atol=1e-11)
    assert scalar.gains.shape == (2,)
    assert not scalar.gains.requires_grad

    rgb_truth = torch.tensor([[0.5, 1.5, 2.0], [3.0, 0.75, 1.25]], dtype=torch.float64)
    rgb_target = predicted * rgb_truth[:, None, :]
    weights = torch.linspace(0.1, 1.0, 14, dtype=torch.float64).reshape(2, 7)
    rgb = estimate_view_gains(
        predicted.requires_grad_(),
        rgb_target,
        weights=weights,
        mode="rgb",
        ridge=1e-12,
    )
    torch.testing.assert_close(rgb.gains, rgb_truth, rtol=2e-11, atol=2e-11)
    assert rgb.gains.shape == (2, 3)
    assert not rgb.gains.requires_grad


def test_gain_ridge_clamp_and_statistics_are_explicit_and_do_not_touch_opacity() -> None:
    gaussians = _gaussians()
    opacity_before = gaussians.opacity.clone()
    estimate = solve_view_gains(
        torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[100.0, 100.0, 100.0], [-3.0, -3.0, -3.0]], dtype=torch.float64),
        mode="scalar",
        ridge=0.5,
        prior_gain=1.0,
        min_gain=0.2,
        max_gain=4.0,
    )

    assert torch.equal(estimate.gains, torch.tensor([4.0, 0.2], dtype=torch.float64))
    assert torch.equal(estimate.clipped, torch.tensor([True, True]))
    assert torch.equal(gaussians.opacity, opacity_before)
    assert estimate.numerator.tolist() == [300.5, -8.5]
    assert estimate.denominator.tolist() == [3.5, 0.5]


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: center_transmittance_visibility(
                _gaussians(),
                (),
            ),
            "at least one camera",
        ),
        (
            lambda: center_transmittance_visibility(
                _gaussians(),
                (_camera(),),
                source_view_indices=torch.zeros(3, dtype=torch.long),
            ),
            "shape",
        ),
        (
            lambda: estimate_view_gains(
                torch.ones(2, 3, 3),
                torch.ones(2, 3, 2),
            ),
            "equal",
        ),
        (
            lambda: solve_view_gains(
                torch.ones(2, 3),
                torch.ones(2, 3),
                ridge=0.0,
            ),
            "ridge",
        ),
    ],
)
def test_visibility_and_gain_validation_fails_closed(call, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        call()
