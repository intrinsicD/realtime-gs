"""CPU tests for source-preserving correspondence topology helpers."""

from __future__ import annotations

import itertools

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.lift.inverse_projection_fiber import InverseProjectionFiber
from rtgs.lift.topology import (
    cyclic_shift_grouped_scores,
    exact_assignment_loss,
    exact_linear_assignment,
    radius_connected_components,
    select_component_representatives,
)


def _brute_force(cost: torch.Tensor, fixed: dict[int, int] | None = None) -> tuple[int, ...]:
    fixed = fixed or {}
    candidates = []
    for permutation in itertools.permutations(range(cost.shape[0])):
        if any(permutation[row] != column for row, column in fixed.items()):
            continue
        total = sum(float(cost[row, column]) for row, column in enumerate(permutation))
        candidates.append((total, permutation))
    return min(candidates)[1]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_exact_assignment_matches_brute_force(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    cost = torch.rand((5, 5), generator=generator, dtype=torch.float64)
    expected = _brute_force(cost)
    actual = exact_linear_assignment(cost)
    assert tuple(actual.tolist()) == expected

    fixed = {1: 3, 4: 0}
    expected_fixed = _brute_force(cost, fixed)
    actual_fixed = exact_linear_assignment(cost, fixed=fixed)
    assert tuple(actual_fixed.tolist()) == expected_fixed


def test_exact_assignment_uses_lexicographic_ties_and_selected_gradients() -> None:
    tied = torch.zeros((4, 4), dtype=torch.float64, requires_grad=True)
    loss, assignment = exact_assignment_loss(tied, fixed={1: 2})
    assert tuple(assignment.tolist()) == (0, 2, 1, 3)
    loss.backward()
    expected = torch.zeros_like(tied)
    expected[torch.arange(4), assignment] = 0.25
    assert torch.equal(tied.grad, expected)


def test_exact_assignment_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="square"):
        exact_linear_assignment(torch.zeros((2, 3)))
    with pytest.raises(ValueError, match="unique columns"):
        exact_linear_assignment(torch.zeros((3, 3)), fixed={0: 1, 2: 1})
    with pytest.raises(ValueError, match="at most 16"):
        exact_linear_assignment(torch.zeros((17, 17)))


def test_components_representatives_and_grouped_shift_are_deterministic() -> None:
    means = torch.tensor(
        [[0.0, 0.0, 0.0], [0.006, 0.0, 0.0], [0.012, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    components = radius_connected_components(
        means,
        torch.tensor([3, 2, 0, 1]),
        radius=0.01,
    )
    assert components == ((0, 1, 2), (3,))
    scores = torch.tensor([0.2, 0.1, 0.1, 0.0], dtype=torch.float64)
    views = torch.tensor([1, 2, 0, 3])
    rows = torch.tensor([0, 0, 4, 0])
    representatives = select_component_representatives(components, scores, views, rows)
    assert representatives.tolist() == [2, 3]

    grouped_scores = torch.arange(16, dtype=torch.float64)
    grouped_views = torch.arange(2).repeat_interleave(8)
    grouped_rows = torch.arange(8).repeat(2)
    shifted = cyclic_shift_grouped_scores(
        grouped_scores,
        grouped_views,
        grouped_rows,
        shift=3,
        group_size=8,
    )
    assert shifted.tolist() == [3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10]


def test_fiber_subset_preserves_state_and_exact_source_projection() -> None:
    camera = Camera.look_at(
        torch.tensor([0.0, 0.0, 2.0]),
        torch.zeros(3),
        fov_x_deg=50.0,
        width=64,
        height=64,
    )
    means2d = torch.tensor([[20.0, 25.0], [31.0, 30.0], [42.0, 38.0]], dtype=torch.float64)
    covariances2d = torch.tensor(
        [
            [[3.0, 0.2], [0.2, 2.0]],
            [[2.5, -0.1], [-0.1, 3.5]],
            [[4.0, 0.3], [0.3, 2.8]],
        ],
        dtype=torch.float64,
    )
    fiber = InverseProjectionFiber(
        cameras=(camera,),
        source_view_indices=torch.zeros(3, dtype=torch.long),
        source_component_indices=torch.arange(3),
        source_means2d=means2d,
        source_covariances2d=covariances2d,
        initial_depths=torch.tensor([1.4, 1.7, 2.1], dtype=torch.float64),
        depth_lower=1.0,
        depth_upper=3.0,
        dilation=0.3,
    )
    with torch.no_grad():
        fiber.depth_logits.add_(torch.tensor([0.2, -0.1, 0.3]))
        fiber.cross.copy_(torch.tensor([[0.1, -0.2], [0.0, 0.3], [-0.1, 0.05]]))
        fiber.log_ray_scale.add_(torch.tensor([0.15, -0.05, 0.2]))
    indices = torch.tensor([2, 0], dtype=torch.long)
    child = fiber.subset(indices)
    parent_means, parent_covariances = fiber.means_covariances()
    child_means, child_covariances = child.means_covariances()
    assert torch.equal(child_means, parent_means[indices])
    assert torch.equal(child_covariances, parent_covariances[indices])
    projected_means, projected_covariances, _ = child.source_projection()
    assert torch.allclose(projected_means, means2d[indices], atol=1e-12, rtol=0.0)
    assert torch.allclose(projected_covariances, covariances2d[indices], atol=1e-12, rtol=0.0)
