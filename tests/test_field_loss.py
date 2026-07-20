"""CPU checks for decomposition-invariant analytic Gaussian field losses."""

from __future__ import annotations

import math

import pytest
import torch

from rtgs.lift.field_loss import (
    AnalyticGaussianField2D,
    field_l2,
    mixture_inner_product,
    peak_gaussian_product_integrals,
)


def _covariances(raw: torch.Tensor) -> torch.Tensor:
    cholesky = raw.new_zeros((raw.shape[0], 2, 2))
    cholesky[:, 0, 0] = raw[:, 0].exp()
    cholesky[:, 1, 0] = raw[:, 1]
    cholesky[:, 1, 1] = raw[:, 2].exp()
    return cholesky @ cholesky.transpose(-1, -2)


def _field(
    means: torch.Tensor,
    covariance_raw: torch.Tensor,
    density: torch.Tensor,
    rgb: torch.Tensor,
) -> AnalyticGaussianField2D:
    return AnalyticGaussianField2D(
        means=means,
        covariances=_covariances(covariance_raw),
        density_amplitudes=density,
        rgb_amplitudes=rgb,
    )


def test_peak_gaussian_product_integral_retains_peak_area() -> None:
    covariance = torch.tensor(
        [[[4.0, 0.75], [0.75, 2.0]]],
        dtype=torch.float64,
    )
    mean = torch.tensor([[1.25, -0.5]], dtype=torch.float64)
    actual = peak_gaussian_product_integrals(mean, covariance, mean, covariance)
    expected = math.pi * torch.linalg.det(covariance[0]).sqrt()
    assert actual.shape == (1, 1)
    assert torch.allclose(actual[0, 0], expected, atol=1e-12, rtol=1e-12)

    shifted = mean + torch.tensor([[2.0, -1.0]], dtype=torch.float64)
    summed = 2.0 * covariance[0]
    delta = mean[0] - shifted[0]
    attenuation = torch.exp(-0.5 * (delta @ torch.linalg.solve(summed, delta)))
    assert torch.allclose(
        peak_gaussian_product_integrals(mean, covariance, shifted, covariance)[0, 0],
        expected * attenuation,
        atol=1e-12,
        rtol=1e-12,
    )


def test_scalar_and_rgb_inner_products_match_unchunked_pairwise_sum() -> None:
    generator = torch.Generator().manual_seed(817)
    first_means = torch.randn(7, 2, generator=generator, dtype=torch.float64)
    second_means = torch.randn(5, 2, generator=generator, dtype=torch.float64)
    first_raw = torch.randn(7, 3, generator=generator, dtype=torch.float64) * 0.2
    second_raw = torch.randn(5, 3, generator=generator, dtype=torch.float64) * 0.2
    first_covariances = _covariances(first_raw)
    second_covariances = _covariances(second_raw)
    first_scalar = torch.rand(7, generator=generator, dtype=torch.float64)
    second_scalar = torch.rand(5, generator=generator, dtype=torch.float64)
    first_rgb = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    second_rgb = torch.randn(5, 3, generator=generator, dtype=torch.float64)
    kernel = peak_gaussian_product_integrals(
        first_means,
        first_covariances,
        second_means,
        second_covariances,
    )

    expected_scalar = (kernel * (first_scalar[:, None] * second_scalar[None, :])).sum()
    expected_rgb = (kernel * (first_rgb @ second_rgb.T)).sum()
    for chunk_size in (1, 2, 4, 32):
        assert torch.allclose(
            mixture_inner_product(
                first_means,
                first_covariances,
                first_scalar,
                second_means,
                second_covariances,
                second_scalar,
                chunk_size=chunk_size,
            ),
            expected_scalar,
            atol=1e-12,
            rtol=1e-12,
        )
        assert torch.allclose(
            mixture_inner_product(
                first_means,
                first_covariances,
                first_rgb,
                second_means,
                second_covariances,
                second_rgb,
                chunk_size=chunk_size,
            ),
            expected_rgb,
            atol=1e-12,
            rtol=1e-12,
        )


def test_field_l2_chunk_parity_for_density_and_rgb_numerator() -> None:
    generator = torch.Generator().manual_seed(991)
    first = _field(
        torch.randn(6, 2, generator=generator, dtype=torch.float64),
        torch.randn(6, 3, generator=generator, dtype=torch.float64) * 0.15,
        torch.rand(6, generator=generator, dtype=torch.float64),
        torch.randn(6, 3, generator=generator, dtype=torch.float64),
    )
    second = _field(
        torch.randn(5, 2, generator=generator, dtype=torch.float64),
        torch.randn(5, 3, generator=generator, dtype=torch.float64) * 0.15,
        torch.rand(5, generator=generator, dtype=torch.float64),
        torch.randn(5, 3, generator=generator, dtype=torch.float64),
    )
    reference = field_l2(first, second, chunk_size=32)
    for chunk_size in (1, 2, 3):
        actual = field_l2(first, second, chunk_size=chunk_size)
        assert torch.allclose(actual.density, reference.density, atol=2e-12, rtol=2e-12)
        assert torch.allclose(
            actual.rgb_numerator,
            reference.rgb_numerator,
            atol=2e-12,
            rtol=2e-12,
        )


def test_field_l2_gradients_match_central_finite_differences() -> None:
    raw = torch.tensor(
        [
            0.2,
            -0.35,
            math.log(0.8),
            0.15,
            math.log(1.1),
            0.65,
            -0.25,
            0.4,
            -0.1,
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = _field(
        torch.tensor([[0.3, -0.2], [1.1, 0.7]], dtype=torch.float64),
        torch.tensor(
            [[math.log(0.9), 0.1, math.log(0.7)], [0.0, -0.2, math.log(1.2)]],
            dtype=torch.float64,
        ),
        torch.tensor([0.75, 0.35], dtype=torch.float64),
        torch.tensor([[0.4, -0.2, 0.7], [0.1, 0.6, -0.3]], dtype=torch.float64),
    )

    def objective(parameters: torch.Tensor) -> torch.Tensor:
        candidate = _field(
            parameters[:2].reshape(1, 2),
            parameters[2:5].reshape(1, 3),
            parameters[5:6],
            parameters[6:].reshape(1, 3),
        )
        terms = field_l2(candidate, target, chunk_size=1)
        return 0.7 * terms.density + 1.3 * terms.rgb_numerator

    loss = objective(raw)
    (actual,) = torch.autograd.grad(loss, raw)
    epsilon = 1e-6
    expected = torch.empty_like(raw)
    with torch.no_grad():
        for index in range(raw.numel()):
            plus = raw.detach().clone()
            minus = raw.detach().clone()
            plus[index] += epsilon
            minus[index] -= epsilon
            expected[index] = (objective(plus) - objective(minus)) / (2.0 * epsilon)
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75])
def test_colocated_mass_split_and_permutation_leave_both_fields_exact(
    fraction: float,
) -> None:
    means = torch.tensor([[-0.4, 0.2], [1.3, -0.7]], dtype=torch.float64)
    raw_covariances = torch.tensor(
        [[math.log(0.8), 0.15, math.log(1.2)], [math.log(1.1), -0.2, math.log(0.7)]],
        dtype=torch.float64,
    )
    density = torch.tensor([0.8, 0.45], dtype=torch.float64)
    rgb = torch.tensor([[0.6, -0.2, 0.4], [0.1, 0.5, 0.7]], dtype=torch.float64)
    parent = _field(means, raw_covariances, density, rgb)

    split_means = torch.stack([means[0], means[0], means[1]])
    split_raw = torch.stack([raw_covariances[0], raw_covariances[0], raw_covariances[1]])
    split_density = torch.stack([fraction * density[0], (1.0 - fraction) * density[0], density[1]])
    split_rgb = torch.stack([fraction * rgb[0], (1.0 - fraction) * rgb[0], rgb[1]])
    split = _field(split_means, split_raw, split_density, split_rgb)
    permutation = torch.tensor([2, 1, 0])
    permuted = AnalyticGaussianField2D(
        means=split.means[permutation],
        covariances=split.covariances[permutation],
        density_amplitudes=split.density_amplitudes[permutation],
        rgb_amplitudes=split.rgb_amplitudes[permutation],
    )

    split_loss = field_l2(parent, split, chunk_size=1)
    permuted_loss = field_l2(parent, permuted, chunk_size=2)
    assert abs(float(split_loss.density)) <= 2e-14
    assert abs(float(split_loss.rgb_numerator)) <= 2e-14
    assert torch.allclose(
        split_loss.density,
        permuted_loss.density,
        atol=2e-14,
        rtol=0.0,
    )
    assert torch.allclose(
        split_loss.rgb_numerator,
        permuted_loss.rgb_numerator,
        atol=2e-14,
        rtol=0.0,
    )
