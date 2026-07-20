"""CPU-first topology helpers for source-anchored Gaussian correspondence research."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch


def exact_linear_assignment(
    cost: torch.Tensor,
    *,
    fixed: Mapping[int, int] | None = None,
) -> torch.Tensor:
    """Return the exact minimum-cost square assignment with deterministic tie breaking.

    The subset dynamic program is deliberately a small-problem correctness reference.  It is
    dependency-free and exponential in ``N``; callers must not present more than 16 rows.
    Exact floating-point ties select the lexicographically smallest column tuple.
    """

    if cost.ndim != 2 or cost.shape[0] == 0 or cost.shape[0] != cost.shape[1]:
        raise ValueError("cost must be a non-empty square matrix")
    if cost.shape[0] > 16:
        raise ValueError("exact subset assignment supports at most 16 rows")
    if not cost.is_floating_point() or not bool(torch.isfinite(cost).all()):
        raise ValueError("cost must be finite and floating point")
    size = int(cost.shape[0])
    fixed_assignments = dict(fixed or {})
    if any(
        type(row) is not int or type(column) is not int for row, column in fixed_assignments.items()
    ):
        raise TypeError("fixed assignment keys and values must be integers")
    if any(
        row < 0 or row >= size or column < 0 or column >= size
        for row, column in fixed_assignments.items()
    ):
        raise IndexError("fixed assignment contains an unavailable row or column")
    if len(set(fixed_assignments.values())) != len(fixed_assignments):
        raise ValueError("fixed assignments must use unique columns")

    detached = cost.detach().to(dtype=torch.float64, device="cpu")
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for row in range(size):
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        permitted = (fixed_assignments[row],) if row in fixed_assignments else range(size)
        for mask, (total, columns) in states.items():
            for column in permitted:
                bit = 1 << column
                if mask & bit:
                    continue
                candidate = (total + float(detached[row, column]), columns + (column,))
                new_mask = mask | bit
                incumbent = next_states.get(new_mask)
                if (
                    incumbent is None
                    or candidate[0] < incumbent[0]
                    or (candidate[0] == incumbent[0] and candidate[1] < incumbent[1])
                ):
                    next_states[new_mask] = candidate
        states = next_states
        if not states:
            raise ValueError("fixed assignments admit no complete bijection")

    full_mask = (1 << size) - 1
    if full_mask not in states:
        raise RuntimeError("assignment solver did not produce a complete bijection")
    return torch.tensor(states[full_mask][1], dtype=torch.long, device=cost.device)


def exact_assignment_loss(
    cost: torch.Tensor,
    *,
    fixed: Mapping[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select with detached costs while retaining gradients through chosen matrix entries."""

    assignment = exact_linear_assignment(cost, fixed=fixed)
    rows = torch.arange(cost.shape[0], dtype=torch.long, device=cost.device)
    return cost[rows, assignment].mean(), assignment


def radius_connected_components(
    means: torch.Tensor,
    retained_indices: torch.Tensor,
    *,
    radius: float,
) -> tuple[tuple[int, ...], ...]:
    """Connected components under a strict Euclidean-radius graph."""

    if means.ndim != 2 or means.shape[1] != 3 or not means.is_floating_point():
        raise ValueError("means must have shape (N,3) and be floating point")
    if not bool(torch.isfinite(means).all()):
        raise ValueError("means must be finite")
    if retained_indices.ndim != 1 or retained_indices.dtype != torch.long:
        raise ValueError("retained_indices must be a one-dimensional long tensor")
    if retained_indices.device != means.device:
        raise ValueError("retained_indices and means must share a device")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if retained_indices.numel() == 0:
        return ()
    if int(retained_indices.min()) < 0 or int(retained_indices.max()) >= means.shape[0]:
        raise IndexError("retained_indices contain an unavailable row")
    if int(torch.unique(retained_indices).numel()) != int(retained_indices.numel()):
        raise ValueError("retained_indices must be unique")

    ordered = sorted(int(index) for index in retained_indices.tolist())
    parent = {index: index for index in ordered}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    for position, left in enumerate(ordered):
        for right in ordered[position + 1 :]:
            distance = torch.linalg.vector_norm(means[left] - means[right])
            if float(distance) < radius:
                union(left, right)

    groups: dict[int, list[int]] = {}
    for index in ordered:
        groups.setdefault(find(index), []).append(index)
    return tuple(tuple(group) for _root, group in sorted(groups.items()))


def select_component_representatives(
    components: Sequence[Sequence[int]],
    scores: torch.Tensor,
    source_views: torch.Tensor,
    source_rows: torch.Tensor,
) -> torch.Tensor:
    """Choose the minimum-score source anchor in every component deterministically."""

    count = int(scores.numel())
    if scores.shape != (count,) or not scores.is_floating_point():
        raise ValueError("scores must be a one-dimensional floating tensor")
    if source_views.shape != (count,) or source_rows.shape != (count,):
        raise ValueError("source metadata must match scores")
    if source_views.dtype != torch.long or source_rows.dtype != torch.long:
        raise ValueError("source metadata must use dtype long")
    if scores.device != source_views.device or scores.device != source_rows.device:
        raise ValueError("scores and source metadata must share a device")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite")
    representatives: list[int] = []
    seen: set[int] = set()
    for component in components:
        members = tuple(int(index) for index in component)
        if not members:
            raise ValueError("components must be non-empty")
        if any(index < 0 or index >= count for index in members):
            raise IndexError("component contains an unavailable index")
        if seen.intersection(members):
            raise ValueError("components must be disjoint")
        seen.update(members)
        representatives.append(
            min(
                members,
                key=lambda index: (
                    float(scores[index]),
                    int(source_views[index]),
                    int(source_rows[index]),
                    index,
                ),
            )
        )
    return torch.tensor(representatives, dtype=torch.long, device=scores.device)


def cyclic_shift_grouped_scores(
    scores: torch.Tensor,
    source_views: torch.Tensor,
    source_rows: torch.Tensor,
    *,
    shift: int,
    group_size: int,
) -> torch.Tensor:
    """Set q[s,i] = scores[s,(i+shift) mod group_size] for complete groups."""

    if scores.ndim != 1 or not scores.is_floating_point() or not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be a finite one-dimensional floating tensor")
    if source_views.shape != scores.shape or source_rows.shape != scores.shape:
        raise ValueError("source metadata must match scores")
    if source_views.dtype != torch.long or source_rows.dtype != torch.long:
        raise ValueError("source metadata must use dtype long")
    if scores.device != source_views.device or scores.device != source_rows.device:
        raise ValueError("scores and source metadata must share a device")
    if type(group_size) is not int or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    if type(shift) is not int or shift <= 0 or shift >= group_size:
        raise ValueError("shift must satisfy 0 < shift < group_size")

    shifted = torch.empty_like(scores)
    for view in sorted(int(item) for item in torch.unique(source_views).tolist()):
        indices = (source_views == view).nonzero(as_tuple=True)[0]
        if int(indices.numel()) != group_size:
            raise ValueError("every source-view group must be complete")
        lookup = {int(source_rows[index]): int(index) for index in indices.tolist()}
        if sorted(lookup) != list(range(group_size)):
            raise ValueError("source rows must be exactly 0..group_size-1 per view")
        for local_row in range(group_size):
            shifted[lookup[local_row]] = scores[lookup[(local_row + shift) % group_size]]
    return shifted


__all__ = [
    "cyclic_shift_grouped_scores",
    "exact_assignment_loss",
    "exact_linear_assignment",
    "radius_connected_components",
    "select_component_representatives",
]
