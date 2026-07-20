"""CPU tests for the train-only fitted-center correspondence backend."""

from __future__ import annotations

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.lift.matching import (
    MatchLayout,
    PatchEpipolarMatcher,
    fundamental_matrix,
    get_position_matcher,
    symmetric_epipolar_distance,
)


def _parallel_camera(center_x: float, image_size: int = 40) -> Camera:
    """Forward-facing calibrated camera with a translated optical center."""
    return Camera(
        fx=30.0,
        fy=30.0,
        cx=image_size / 2,
        cy=image_size / 2,
        width=image_size,
        height=image_size,
        R=torch.eye(3),
        t=torch.tensor([-center_x, 0.0, 0.0]),
    )


def _stamp_patch(image: torch.Tensor, xy: torch.Tensor, identity: int) -> None:
    """Place a distinct 5x5 RGB patch at a half-integer pixel center."""
    center_x = int(float(xy[0] - 0.5))
    center_y = int(float(xy[1] - 0.5))
    yy, xx = torch.meshgrid(torch.arange(5), torch.arange(5), indexing="ij")
    base = image.new_tensor(
        [
            [0.15, 0.65, 0.25],
            [0.75, 0.20, 0.40],
            [0.30, 0.35, 0.85],
        ][identity]
    )
    pattern = base[None, None, :] * (0.65 + 0.04 * xx[..., None] + 0.03 * yy[..., None])
    image[center_y - 2 : center_y + 3, center_x - 2 : center_x + 3] = pattern


def _known_match_problem() -> tuple[list[torch.Tensor], list[Camera], MatchLayout]:
    """Two-view problem with three exact, geometrically well-conditioned matches."""
    cameras = [_parallel_camera(-0.4), _parallel_camera(0.4)]
    left_xy = torch.tensor([[14.5, 12.5], [20.5, 20.5], [26.5, 28.5]])
    # Parallel stereo shifts every z=2 point left by fx * baseline / z = 12 pixels.
    right_xy = left_xy - torch.tensor([12.0, 0.0])
    images = [torch.zeros(40, 40, 3), torch.zeros(40, 40, 3)]
    for identity in range(3):
        _stamp_patch(images[0], left_xy[identity], identity)
        _stamp_patch(images[1], right_xy[identity], identity)
    layout = MatchLayout(
        xy=torch.cat([left_xy, right_xy]),
        source_view_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
        source_ranges=[(0, 3), (3, 6)],
    )
    return images, cameras, layout


def test_fundamental_matrix_places_true_projections_on_epipolar_lines():
    left = Camera.look_at(
        torch.tensor([2.0, 0.2, 0.0]),
        torch.zeros(3),
        width=48,
        height=48,
        fov_x_deg=45.0,
    )
    right = Camera.look_at(
        torch.tensor([0.8, -0.1, 1.8]),
        torch.zeros(3),
        width=48,
        height=48,
        fov_x_deg=45.0,
    )
    points = torch.tensor([[-0.2, -0.1, 0.1], [0.15, 0.2, -0.15], [0.05, -0.25, 0.2]])
    left_xy, left_depth = left.project(points)
    right_xy, right_depth = right.project(points)
    assert bool((left_depth > 0).all() and (right_depth > 0).all())

    fundamental = fundamental_matrix(left, right, left_xy)
    distance = symmetric_epipolar_distance(left_xy, right_xy, fundamental)
    assert torch.allclose(fundamental.norm(), torch.tensor(1.0), atol=1e-6, rtol=0)
    assert torch.all(torch.diagonal(distance) < 1e-4)
    assert float(distance[0, 1]) > 0.1


def test_patch_matcher_builds_deterministic_graph_and_degree_exact_control():
    images, cameras, layout = _known_match_problem()
    matcher = PatchEpipolarMatcher()

    first = matcher.match(images, cameras, layout)
    second = matcher.match(images, cameras, layout)

    expected = torch.tensor([[0, 3], [1, 4], [2, 5]])
    expected_control = torch.tensor([[0, 4], [1, 5], [2, 3]])
    assert torch.equal(first.pairs, expected)
    assert torch.equal(first.shuffled_pairs, expected_control)
    assert torch.equal(first.pairs, second.pairs)
    assert torch.equal(first.shuffled_pairs, second.shuffled_pairs)
    assert torch.equal(first.confidence, second.confidence)
    assert bool((first.confidence >= 0.5).all())

    positive_degree = torch.bincount(first.pairs.flatten(), minlength=layout.xy.shape[0])
    control_degree = torch.bincount(first.shuffled_pairs.flatten(), minlength=layout.xy.shape[0])
    assert torch.equal(positive_degree, control_degree)
    assert not set(map(tuple, first.pairs.tolist())).intersection(
        map(tuple, first.shuffled_pairs.tolist())
    )
    assert first.diagnostics["edge_count"] == 3
    assert first.diagnostics["represented_node_fraction"] == 1.0
    assert first.diagnostics["degree_exact"] is True
    assert first.diagnostics["source_pair_counts_exact"] is True
    assert first.diagnostics["blocks"]["0-1"]["edge_count"] == 3


def test_patch_matcher_rejects_exact_descriptor_ties():
    cameras = [_parallel_camera(-0.4), _parallel_camera(0.4)]
    # Both rows have two epipolar-compatible, identically black candidates. The epsilon-stabilized
    # best/second ratio is one, so deterministic index tie-breaking cannot create a false match.
    left_xy = torch.tensor([[14.5, 18.5], [20.5, 18.5]])
    right_xy = left_xy - torch.tensor([12.0, 0.0])
    layout = MatchLayout(
        xy=torch.cat([left_xy, right_xy]),
        source_view_ids=torch.tensor([0, 0, 1, 1]),
        source_ranges=[(0, 2), (2, 4)],
    )
    images = [torch.zeros(40, 40, 3), torch.zeros(40, 40, 3)]

    with pytest.raises(ValueError, match="no derangeable"):
        PatchEpipolarMatcher().match(images, cameras, layout)


@pytest.mark.parametrize(
    ("layout", "message"),
    [
        (
            MatchLayout(
                xy=torch.zeros(3, 3),
                source_view_ids=torch.tensor([0, 0, 1]),
                source_ranges=[(0, 2), (2, 3)],
            ),
            "xy must have shape",
        ),
        (
            MatchLayout(
                xy=torch.zeros(3, 2),
                source_view_ids=torch.tensor([0, 0, 0]),
                source_ranges=[(0, 2), (2, 3)],
            ),
            "source ids disagree",
        ),
        (
            MatchLayout(
                xy=torch.zeros(3, 2),
                source_view_ids=torch.tensor([0, 0, 1]),
                source_ranges=[(0, 1), (2, 3)],
            ),
            "contiguous and exhaustive",
        ),
    ],
)
def test_match_layout_validation_rejects_inconsistent_inputs(layout, message):
    with pytest.raises(ValueError, match=message):
        layout.validate(2)


def test_position_matcher_factory_and_configuration_validation():
    assert isinstance(get_position_matcher("patch-epipolar"), PatchEpipolarMatcher)
    assert isinstance(get_position_matcher("cpu_patch_epipolar"), PatchEpipolarMatcher)
    with pytest.raises(ValueError, match="unknown position matcher"):
        get_position_matcher("learned_oracle")
    with pytest.raises(ValueError, match="patch_radius"):
        PatchEpipolarMatcher(patch_radius=-1)
    with pytest.raises(ValueError, match="max_ratio"):
        PatchEpipolarMatcher(max_ratio=1.0)
    with pytest.raises(ValueError, match="min_block_edges"):
        PatchEpipolarMatcher(min_block_edges=1)
