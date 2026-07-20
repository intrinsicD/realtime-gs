"""Exactness and autograd checks for tiled SSIM."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rtgs.core.metrics import ssim


def _reference_ssim(
    pred: torch.Tensor, target: torch.Tensor, window_size: int = 11
) -> torch.Tensor:
    """The original untiled separable implementation."""
    x = pred.permute(2, 0, 1)[None]
    y = target.permute(2, 0, 1)[None]
    coords = (
        torch.arange(window_size, dtype=torch.float32, device=pred.device) - (window_size - 1) / 2
    )
    kernel = torch.exp(-(coords**2) / (2 * 1.5**2))
    kernel = kernel / kernel.sum()
    channels = x.shape[1]
    radius = window_size // 2
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)

    def separable_filter(image: torch.Tensor) -> torch.Tensor:
        filtered = F.conv2d(image, vertical, padding=(radius, 0), groups=channels)
        return F.conv2d(filtered, horizontal, padding=(0, radius), groups=channels)

    mu_x = separable_filter(x)
    mu_y = separable_filter(y)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = separable_filter(x * x) - mu_x2
    sigma_y2 = separable_filter(y * y) - mu_y2
    sigma_xy = separable_filter(x * y) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    similarity = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return similarity.mean()


@pytest.mark.parametrize(
    ("height", "width", "tile_rows"),
    [
        (17, 19, 6),
        (18, 20, 7),
        (19, 18, 5),
        (18, 19, 6),
    ],
)
def test_tiled_ssim_matches_untiled_value_and_gradients(
    height: int, width: int, tile_rows: int
) -> None:
    generator = torch.Generator().manual_seed(height * 1000 + width)
    pred_data = torch.rand(height, width, 3, generator=generator)
    target_data = torch.rand(height, width, 3, generator=generator)
    pred_reference = pred_data.clone().requires_grad_()
    target_reference = target_data.clone().requires_grad_()
    pred_tiled = pred_data.clone().requires_grad_()
    target_tiled = target_data.clone().requires_grad_()

    reference = _reference_ssim(pred_reference, target_reference)
    tiled = ssim(pred_tiled, target_tiled, tile_rows=tile_rows)
    reference.backward()
    tiled.backward()

    assert torch.allclose(tiled, reference, atol=2e-7, rtol=2e-7)
    assert torch.allclose(pred_tiled.grad, pred_reference.grad, atol=2e-7, rtol=2e-5)
    assert torch.allclose(target_tiled.grad, target_reference.grad, atol=2e-7, rtol=2e-5)


def test_default_tiled_ssim_matches_untiled_reference_across_boundary() -> None:
    generator = torch.Generator().manual_seed(123)
    pred = torch.rand(259, 13, 3, generator=generator)
    target = torch.rand(259, 13, 3, generator=generator)

    expected = _reference_ssim(pred, target)
    actual = ssim(pred, target)

    assert torch.allclose(actual, expected, atol=2e-7, rtol=2e-7)


@pytest.mark.parametrize("tile_rows", [0, -1])
def test_tiled_ssim_rejects_nonpositive_tile_rows(tile_rows: int) -> None:
    image = torch.rand(8, 9, 3)
    with pytest.raises(ValueError, match="tile_rows"):
        ssim(image, image, tile_rows=tile_rows)
