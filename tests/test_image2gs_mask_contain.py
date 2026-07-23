"""Stage 1: opt-in soft mask containment (``FitConfig.mask_coverage_weight``)."""

import pytest
import torch

from rtgs.image2gs.fit import FitConfig, fit_image
from rtgs.image2gs.renderer2d import render_gaussian_coverage_2d


def _image_and_mask(h: int = 24, w: int = 24):
    torch.manual_seed(0)
    image = torch.rand(h, w, 3)
    mask = torch.zeros(h, w)
    mask[6:18, 6:18] = 1.0  # central foreground box
    return image, mask


def _outside_coverage(gaussians, mask, h: int, w: int) -> float:
    coverage = render_gaussian_coverage_2d(gaussians, h, w)
    outside = 1.0 - mask.clamp(0, 1)
    return float((coverage * outside).sum())


def test_mask_coverage_weight_off_by_default():
    assert FitConfig().mask_coverage_weight == 0.0


def test_mask_coverage_weight_reduces_outside_spill():
    image, mask = _image_and_mask()
    h, w = image.shape[:2]
    shared = dict(n_gaussians=40, iterations=120, log_every=120)
    free, _ = fit_image(image, FitConfig(mask_coverage_weight=0.0, **shared), seed=0, mask=mask)
    contained, _ = fit_image(
        image, FitConfig(mask_coverage_weight=5.0, **shared), seed=0, mask=mask
    )
    # Both return full-image coordinates; the penalty pulls coverage inside the mask.
    assert _outside_coverage(contained, mask, h, w) < _outside_coverage(free, mask, h, w)


def test_mask_coverage_weight_requires_a_mask():
    image, _ = _image_and_mask()
    with pytest.raises(ValueError, match="requires a mask"):
        fit_image(image, FitConfig(mask_coverage_weight=1.0, iterations=3), seed=0)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"mask_coverage_weight": -1.0}, "non-negative"),
        ({"mask_coverage_weight": 1.0, "batch_views": True}, "batch_views or the pool"),
        ({"mask_coverage_weight": 1.0, "pool": True}, "batch_views or the pool"),
        ({"mask_coverage_weight": 1.0, "backend": "structsplat"}, "native backend"),
    ],
)
def test_mask_coverage_weight_validation(overrides, match):
    image, mask = _image_and_mask()
    with pytest.raises(ValueError, match=match):
        fit_image(image, FitConfig(iterations=3, **overrides), seed=0, mask=mask)
