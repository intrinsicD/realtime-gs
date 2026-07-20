"""Deterministic train-view correspondence backends for fixed position constraints.

The CPU reference matcher deliberately consumes only training RGB images, calibrated cameras,
and the retained fitted-center layout.  It does not accept a :class:`SceneData` object, depth,
sparse points, or held-out data, which keeps synthetic ground truth out of graph construction.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Protocol

import torch

from rtgs.core.camera import Camera
from rtgs.lift.base import bilinear_sample


@dataclass(frozen=True)
class MatchLayout:
    """Retained fitted centers and their contiguous source-view layout."""

    xy: torch.Tensor
    source_view_ids: torch.Tensor
    source_ranges: list[tuple[int, int]]

    def validate(self, n_views: int) -> None:
        """Raise when the layout is not a canonical contiguous per-view partition."""
        if self.xy.ndim != 2 or self.xy.shape[1] != 2:
            raise ValueError("match layout xy must have shape (N, 2)")
        if self.source_view_ids.shape != (self.xy.shape[0],):
            raise ValueError("match layout source_view_ids must have shape (N,)")
        if self.source_view_ids.dtype != torch.long:
            raise ValueError("match layout source_view_ids must use torch.long")
        if len(self.source_ranges) != n_views:
            raise ValueError("match layout requires one source range per view")
        cursor = 0
        for view_index, (start, end) in enumerate(self.source_ranges):
            if start != cursor or end < start or end > self.xy.shape[0]:
                raise ValueError("match layout source ranges must be contiguous and exhaustive")
            if end > start and not bool((self.source_view_ids[start:end] == view_index).all()):
                raise ValueError("match layout source ids disagree with source ranges")
            cursor = end
        if cursor != self.xy.shape[0]:
            raise ValueError("match layout source ranges do not cover every retained center")


@dataclass(frozen=True)
class PositionMatchGraph:
    """A positive graph and exact-degree cyclic endpoint derangement."""

    pairs: torch.Tensor
    shuffled_pairs: torch.Tensor
    confidence: torch.Tensor
    diagnostics: dict[str, object]


class PositionMatcher(Protocol):
    """Build a detached fixed graph from training images, cameras, and fitted centers."""

    def match(
        self,
        images: list[torch.Tensor],
        cameras: list[Camera],
        layout: MatchLayout,
    ) -> PositionMatchGraph:
        """Return positive and degree-matched control edges."""
        ...


def _distribution(values: torch.Tensor) -> dict[str, float | int | None]:
    values = values.detach().double().flatten()
    if values.numel() == 0:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(values.numel()),
        "min": float(values.min()),
        "median": float(values.median()),
        "p90": float(torch.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind()
    zero = torch.zeros((), dtype=vector.dtype, device=vector.device)
    return torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )


def fundamental_matrix(left: Camera, right: Camera, like: torch.Tensor) -> torch.Tensor:
    """Return F such that ``x_right.T @ F @ x_left == 0`` for calibrated projections."""
    left_r = left.R.to(like)
    right_r = right.R.to(like)
    left_t = left.t.to(like)
    right_t = right.t.to(like)
    relative_r = right_r @ left_r.T
    relative_t = right_t - relative_r @ left_t
    essential = _skew(relative_t) @ relative_r
    left_k_inv = torch.linalg.inv(left.K.to(like))
    right_k_inv = torch.linalg.inv(right.K.to(like))
    fundamental = right_k_inv.T @ essential @ left_k_inv
    return fundamental / fundamental.norm().clamp_min(1e-12)


def symmetric_epipolar_distance(
    left_xy: torch.Tensor,
    right_xy: torch.Tensor,
    fundamental: torch.Tensor,
) -> torch.Tensor:
    """Maximum bidirectional point-to-epipolar-line distance for every pixel pair."""
    left_h = torch.cat([left_xy, torch.ones_like(left_xy[:, :1])], dim=-1)
    right_h = torch.cat([right_xy, torch.ones_like(right_xy[:, :1])], dim=-1)
    errors = (left_h @ fundamental.T) @ right_h.T
    right_lines = left_h @ fundamental.T
    left_lines = right_h @ fundamental
    right_distance = errors.abs() / right_lines[:, :2].norm(dim=-1).clamp_min(1e-12)[:, None]
    left_distance = errors.abs() / left_lines[:, :2].norm(dim=-1).clamp_min(1e-12)[None, :]
    return torch.maximum(left_distance, right_distance)


def _patch_descriptors(image: torch.Tensor, xy: torch.Tensor, radius: int) -> torch.Tensor:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("matcher images must have shape (H, W, 3)")
    offsets = torch.tensor(
        list(itertools.product(range(-radius, radius + 1), repeat=2)),
        dtype=xy.dtype,
        device=xy.device,
    )
    # itertools emits (dy, dx); image coordinates are (x, y).
    offsets = offsets[:, [1, 0]]
    samples = xy[:, None, :] + offsets[None, :, :]
    values = bilinear_sample(image.to(xy), samples.reshape(-1, 2))
    return values.reshape(xy.shape[0], -1)


def _best_two(cost: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic best indices, best values, and second values along ``dim``."""
    best_values, best_indices = cost.min(dim=dim)
    if cost.shape[dim] == 1:
        second_values = torch.full_like(best_values, torch.inf)
    else:
        second_values = torch.topk(cost, k=2, dim=dim, largest=False, sorted=True).values
        second_values = second_values.select(dim, 1)
    return best_indices, best_values, second_values


def _triangulation_diagnostics(
    left_camera: Camera,
    right_camera: Camera,
    left_xy: torch.Tensor,
    right_xy: torch.Tensor,
) -> dict[str, torch.Tensor]:
    left_origin, left_direction = left_camera.pixel_rays(left_xy)
    right_origin, right_direction = right_camera.pixel_rays(right_xy)
    left_origin = left_origin.to(left_xy).expand_as(left_direction)
    right_origin = right_origin.to(left_xy).expand_as(right_direction)
    left_direction = torch.nn.functional.normalize(left_direction, dim=-1)
    right_direction = torch.nn.functional.normalize(right_direction, dim=-1)
    origin_delta = left_origin - right_origin
    cosine = (left_direction * right_direction).sum(dim=-1)
    denominator = 1.0 - cosine.square()
    left_dot = (left_direction * origin_delta).sum(dim=-1)
    right_dot = (right_direction * origin_delta).sum(dim=-1)
    left_parameter = (cosine * right_dot - left_dot) / denominator.clamp_min(1e-12)
    right_parameter = (right_dot - cosine * left_dot) / denominator.clamp_min(1e-12)
    left_point = left_origin + left_parameter[:, None] * left_direction
    right_point = right_origin + right_parameter[:, None] * right_direction
    midpoint = 0.5 * (left_point + right_point)
    left_reprojection, left_depth = left_camera.project(midpoint)
    right_reprojection, right_depth = right_camera.project(midpoint)
    return {
        "denominator": denominator,
        "left_parameter": left_parameter,
        "right_parameter": right_parameter,
        "angle_deg": torch.rad2deg(torch.acos(cosine.abs().clamp(0.0, 1.0))),
        "ray_gap": (left_point - right_point).norm(dim=-1),
        "left_depth": left_depth,
        "right_depth": right_depth,
        "reprojection": torch.maximum(
            (left_reprojection - left_xy).norm(dim=-1),
            (right_reprojection - right_xy).norm(dim=-1),
        ),
    }


class PatchEpipolarMatcher:
    """Pure-Torch reciprocal raw-patch matcher with calibrated geometric filtering.

    Defaults are the frozen CPU reference settings used by the dense position-consistency
    experiment.  All accepted edges are consumed uniformly by the lifter; ``confidence`` is
    diagnostic only.
    """

    def __init__(
        self,
        *,
        patch_radius: int = 2,
        max_epipolar_distance: float = 2.0,
        max_ratio: float = 0.5,
        min_ray_angle_deg: float = 10.0,
        min_depth: float = 0.05,
        max_reprojection_error: float = 1.5,
        min_block_edges: int = 2,
        ratio_epsilon: float = 1e-6,
    ) -> None:
        if patch_radius < 0:
            raise ValueError("patch_radius must be non-negative")
        for name, value in (
            ("max_epipolar_distance", max_epipolar_distance),
            ("min_ray_angle_deg", min_ray_angle_deg),
            ("min_depth", min_depth),
            ("max_reprojection_error", max_reprojection_error),
            ("ratio_epsilon", ratio_epsilon),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(max_ratio) or not 0 < max_ratio < 1:
            raise ValueError("max_ratio must be finite and in (0, 1)")
        if min_block_edges < 2:
            raise ValueError("min_block_edges must be at least two for a derangement")
        self.patch_radius = patch_radius
        self.max_epipolar_distance = max_epipolar_distance
        self.max_ratio = max_ratio
        self.min_ray_angle_deg = min_ray_angle_deg
        self.min_depth = min_depth
        self.max_reprojection_error = max_reprojection_error
        self.min_block_edges = min_block_edges
        self.ratio_epsilon = ratio_epsilon

    def match(
        self,
        images: list[torch.Tensor],
        cameras: list[Camera],
        layout: MatchLayout,
    ) -> PositionMatchGraph:
        """Build one immutable positive graph and its cyclic degree-exact control."""
        if len(images) != len(cameras):
            raise ValueError("matcher requires one calibrated camera per image")
        layout.validate(len(images))
        descriptors = []
        for image, (start, end) in zip(images, layout.source_ranges):
            descriptors.append(_patch_descriptors(image, layout.xy[start:end], self.patch_radius))

        positive_blocks: list[torch.Tensor] = []
        shuffled_blocks: list[torch.Tensor] = []
        confidence_blocks: list[torch.Tensor] = []
        block_diagnostics: dict[str, dict[str, object]] = {}
        accepted_epipolar = []
        accepted_descriptor = []
        accepted_ratio = []
        accepted_angle = []
        accepted_reprojection = []
        accepted_ray_gap = []

        for left_view, right_view in itertools.combinations(range(len(images)), 2):
            left_start, left_end = layout.source_ranges[left_view]
            right_start, right_end = layout.source_ranges[right_view]
            if left_end == left_start or right_end == right_start:
                continue
            left_xy = layout.xy[left_start:left_end]
            right_xy = layout.xy[right_start:right_end]
            descriptor_distance = torch.cdist(descriptors[left_view], descriptors[right_view])
            fundamental = fundamental_matrix(cameras[left_view], cameras[right_view], left_xy)
            epipolar_distance = symmetric_epipolar_distance(left_xy, right_xy, fundamental)
            candidate = epipolar_distance <= self.max_epipolar_distance
            masked_distance = descriptor_distance.masked_fill(~candidate, torch.inf)

            forward_index, forward_best, forward_second = _best_two(masked_distance, dim=1)
            reverse_index, reverse_best, reverse_second = _best_two(masked_distance, dim=0)
            forward_ratio = (forward_best + self.ratio_epsilon) / (
                forward_second + self.ratio_epsilon
            )
            reverse_ratio = (reverse_best + self.ratio_epsilon) / (
                reverse_second + self.ratio_epsilon
            )
            left_local = torch.arange(left_xy.shape[0], device=left_xy.device)
            finite = torch.isfinite(forward_best)
            mutual = finite & (reverse_index[forward_index] == left_local)
            distinctive = (forward_ratio <= self.max_ratio) & (
                reverse_ratio[forward_index] <= self.max_ratio
            )
            selected_left = left_local[mutual & distinctive]
            selected_right = forward_index[selected_left]
            pre_geometry_count = int(selected_left.numel())
            if selected_left.numel():
                geometry = _triangulation_diagnostics(
                    cameras[left_view],
                    cameras[right_view],
                    left_xy[selected_left],
                    right_xy[selected_right],
                )
                geometry_keep = (
                    (geometry["denominator"] > 1e-8)
                    & (geometry["left_parameter"] > 0)
                    & (geometry["right_parameter"] > 0)
                    & (geometry["angle_deg"] >= self.min_ray_angle_deg)
                    & (geometry["left_depth"] > self.min_depth)
                    & (geometry["right_depth"] > self.min_depth)
                    & (geometry["reprojection"] <= self.max_reprojection_error)
                )
                selected_left = selected_left[geometry_keep]
                selected_right = selected_right[geometry_keep]
                geometry = {key: value[geometry_keep] for key, value in geometry.items()}
            else:
                geometry = {
                    name: left_xy.new_empty((0,))
                    for name in (
                        "denominator",
                        "left_parameter",
                        "right_parameter",
                        "angle_deg",
                        "ray_gap",
                        "left_depth",
                        "right_depth",
                        "reprojection",
                    )
                }

            if selected_left.numel() < self.min_block_edges:
                block_diagnostics[f"{left_view}-{right_view}"] = {
                    "candidate_count": int(candidate.sum()),
                    "reciprocal_distinctive_count": pre_geometry_count,
                    "accepted_before_block_floor": int(selected_left.numel()),
                    "edge_count": 0,
                }
                continue

            order_key = selected_left * max(right_xy.shape[0], 1) + selected_right
            order = torch.argsort(order_key, stable=True)
            selected_left = selected_left[order]
            selected_right = selected_right[order]
            positive = torch.stack(
                [selected_left + left_start, selected_right + right_start], dim=-1
            ).long()
            shuffled = torch.stack([positive[:, 0], torch.roll(positive[:, 1], shifts=-1)], dim=-1)
            selected_epipolar = epipolar_distance[selected_left, selected_right]
            selected_descriptor = descriptor_distance[selected_left, selected_right]
            selected_ratio = torch.maximum(
                forward_ratio[selected_left], reverse_ratio[selected_right]
            )
            confidence = (1.0 - selected_ratio).clamp(0.0, 1.0)
            positive_blocks.append(positive)
            shuffled_blocks.append(shuffled)
            confidence_blocks.append(confidence)
            accepted_epipolar.append(selected_epipolar)
            accepted_descriptor.append(selected_descriptor)
            accepted_ratio.append(selected_ratio)
            accepted_angle.append(geometry["angle_deg"][order])
            accepted_reprojection.append(geometry["reprojection"][order])
            accepted_ray_gap.append(geometry["ray_gap"][order])
            block_diagnostics[f"{left_view}-{right_view}"] = {
                "candidate_count": int(candidate.sum()),
                "reciprocal_distinctive_count": pre_geometry_count,
                "accepted_before_block_floor": int(selected_left.numel()),
                "edge_count": int(selected_left.numel()),
                "epipolar_distance_px": _distribution(selected_epipolar),
                "descriptor_l2": _distribution(selected_descriptor),
                "best_second_ratio": _distribution(selected_ratio),
                "ray_angle_deg": _distribution(geometry["angle_deg"][order]),
                "reprojection_error_px": _distribution(geometry["reprojection"][order]),
                "closest_line_gap": _distribution(geometry["ray_gap"][order]),
            }

        if not positive_blocks:
            raise ValueError("patch-epipolar matcher produced no derangeable view-pair blocks")
        pairs = torch.cat(positive_blocks).detach()
        shuffled_pairs = torch.cat(shuffled_blocks).detach()
        confidence = torch.cat(confidence_blocks).detach()
        if not bool((pairs[:, 0] < pairs[:, 1]).all()):
            raise AssertionError("matcher pairs are not canonical cross-source indices")
        if not bool((shuffled_pairs[:, 0] < shuffled_pairs[:, 1]).all()):
            raise AssertionError("shuffled matcher pairs are not canonical cross-source indices")
        if torch.unique(pairs, dim=0).shape[0] != pairs.shape[0]:
            raise AssertionError("matcher produced duplicate positive edges")
        if torch.unique(shuffled_pairs, dim=0).shape[0] != shuffled_pairs.shape[0]:
            raise AssertionError("matcher produced duplicate shuffled edges")
        if set(map(tuple, pairs.tolist())).intersection(map(tuple, shuffled_pairs.tolist())):
            raise AssertionError("cyclic control retains an exact positive edge")
        positive_degree = torch.bincount(pairs.flatten(), minlength=layout.xy.shape[0])
        shuffled_degree = torch.bincount(shuffled_pairs.flatten(), minlength=layout.xy.shape[0])
        if not torch.equal(positive_degree, shuffled_degree):
            raise AssertionError("cyclic control changes endpoint degree")
        positive_source_pairs = torch.stack(
            [layout.source_view_ids[pairs[:, 0]], layout.source_view_ids[pairs[:, 1]]], dim=-1
        )
        shuffled_source_pairs = torch.stack(
            [
                layout.source_view_ids[shuffled_pairs[:, 0]],
                layout.source_view_ids[shuffled_pairs[:, 1]],
            ],
            dim=-1,
        )
        if not torch.equal(positive_source_pairs, shuffled_source_pairs):
            raise AssertionError("cyclic control changes ordered camera-pair blocks")
        represented = positive_degree > 0
        nodes_per_view = torch.bincount(layout.source_view_ids[represented], minlength=len(images))

        def combined(values: list[torch.Tensor]) -> torch.Tensor:
            return torch.cat(values) if values else layout.xy.new_empty((0,))

        diagnostics: dict[str, object] = {
            "backend": "patch_epipolar",
            "config": {
                "patch_radius": self.patch_radius,
                "patch_side": 2 * self.patch_radius + 1,
                "patch_channels": "raw_rgb",
                "padding": "bilinear_sample_clamp",
                "descriptor_metric": "l2",
                "max_epipolar_distance_px": self.max_epipolar_distance,
                "epipolar_metric": "max_bidirectional_point_to_line",
                "max_bidirectional_best_second_ratio": self.max_ratio,
                "ratio_epsilon": self.ratio_epsilon,
                "missing_second_candidate": "positive_infinity",
                "mutual_tie_break": "lowest_retained_index_then_ratio_rejects_exact_ties",
                "min_acute_line_angle_deg": self.min_ray_angle_deg,
                "triangulation": "closest_infinite_lines_midpoint",
                "require_positive_line_parameters": True,
                "min_reprojected_depth": self.min_depth,
                "max_midpoint_reprojection_error_px": self.max_reprojection_error,
                "min_edges_per_camera_pair_block": self.min_block_edges,
                "edge_weighting": "uniform_confidence_diagnostic_only",
                "control": "right_endpoint_cyclic_shift_minus_one_per_sorted_block",
            },
            "edge_count": int(pairs.shape[0]),
            "represented_node_count": int(represented.sum()),
            "represented_node_fraction": float(represented.float().mean()),
            "view_pair_block_count": sum(
                int(block["edge_count"] > 0) for block in block_diagnostics.values()
            ),
            "represented_views": torch.nonzero(nodes_per_view > 0, as_tuple=False)
            .squeeze(-1)
            .tolist(),
            "represented_nodes_per_view": nodes_per_view.tolist(),
            "endpoint_degree": _distribution(positive_degree[represented]),
            "confidence": _distribution(confidence),
            "epipolar_distance_px": _distribution(combined(accepted_epipolar)),
            "descriptor_l2": _distribution(combined(accepted_descriptor)),
            "best_second_ratio": _distribution(combined(accepted_ratio)),
            "ray_angle_deg": _distribution(combined(accepted_angle)),
            "reprojection_error_px": _distribution(combined(accepted_reprojection)),
            "closest_line_gap": _distribution(combined(accepted_ray_gap)),
            "blocks": block_diagnostics,
            "degree_exact": True,
            "source_pair_counts_exact": True,
            "exact_edge_overlap_count": 0,
        }
        return PositionMatchGraph(
            pairs=pairs,
            shuffled_pairs=shuffled_pairs,
            confidence=confidence,
            diagnostics=diagnostics,
        )


def get_position_matcher(name: str, **kwargs: object) -> PositionMatcher:
    """Construct a position matcher without importing optional learned dependencies."""
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"patch_epipolar", "cpu_patch_epipolar"}:
        return PatchEpipolarMatcher(**kwargs)
    raise ValueError(f"unknown position matcher {name!r}")
