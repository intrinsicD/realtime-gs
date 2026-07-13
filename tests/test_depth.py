"""Depth backends and scale/shift alignment."""

import pytest
import torch

from rtgs.depth.align import (
    align_depth_to_points,
    align_inverse_depth_to_bounds,
    scale_shift_align,
)
from rtgs.depth.mock import ConstantDepth, GroundTruthDepth


def test_scale_shift_align_exact():
    pred = torch.rand(100) + 0.5
    ref = 2.5 * pred + 0.7
    s, b = scale_shift_align(pred, ref)
    assert abs(s - 2.5) < 1e-4
    assert abs(b - 0.7) < 1e-4


def test_scale_shift_align_robust_to_outliers():
    pred = torch.rand(200) + 0.5
    ref = 3.0 * pred + 0.2
    ref[:10] += 50.0  # gross outliers
    s, b = scale_shift_align(pred, ref, robust_iters=2)
    assert abs(s - 3.0) < 0.2
    assert abs(b - 0.2) < 0.3


def test_scale_shift_align_degenerate_flat_pred():
    pred = torch.full((50,), 2.0)
    ref = torch.rand(50) + 1.0
    s, b = scale_shift_align(pred, ref)
    assert s > 0  # falls back to scale-only, stays positive


def test_align_depth_to_points_recovers_gt(tiny_scene):
    """Distorting GT depth by a known affine map must be undone by alignment."""
    view = 0
    cam = tiny_scene.cameras[view]
    gt = tiny_scene.gt_depths[view]
    distorted = 0.4 * gt + 1.3
    aligned = align_depth_to_points(distorted, cam, tiny_scene.points, robust_iters=2)
    valid = gt > 0.05
    err = (aligned[valid] - gt[valid]).abs().median()
    assert err < 0.15, f"median depth error {err}"


def test_mock_backends(tiny_scene):
    gt = GroundTruthDepth(tiny_scene.gt_depths)
    for i in range(2):
        pred = gt.predict(tiny_scene.images[i])
        assert pred.kind == "metric"
        assert torch.equal(pred.depth, tiny_scene.gt_depths[i])
    const = ConstantDepth(2.0)
    pred = const.predict(tiny_scene.images[0])
    assert (pred.depth == 2.0).all()


def test_inverse_depth_bounds_alignment_without_sfm(tiny_scene):
    camera = tiny_scene.cameras[0]
    center, extent = tiny_scene.center_and_extent()
    inverse = torch.linspace(0.1, 1.0, 32 * 32).reshape(32, 32)
    depth = align_inverse_depth_to_bounds(inverse, camera, center, extent)
    assert torch.isfinite(depth).all() and (depth > 0).all()
    assert depth[-1, -1] < depth[0, 0]  # higher inverse depth remains nearer


def test_gt_backend_shape_mismatch(tiny_scene):
    gt = GroundTruthDepth([torch.zeros(4, 4)])
    with pytest.raises(ValueError):
        gt.predict(tiny_scene.images[0])
