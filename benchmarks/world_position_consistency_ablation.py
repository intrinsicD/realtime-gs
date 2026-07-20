#!/usr/bin/env python3
"""Paired fixed-match world-frame position-consistency experiment."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from cross_view_supervision_ablation import (
    ListDepthBackend,
    assert_equal_fields,
    cross_only_training_l1,
    depth_tensor_hash,
    fitted_tensor_hash,
    git_metadata,
    integer_sequence_hash,
    tensor_collection_hash,
)
from depth_anchor_ablation import (
    held_out_geometry,
    history_checkpoints,
    make_priors,
    nearest_gt_diagnostics,
    source_depth_diagnostics,
)
from rtgs.data.synthetic import make_synthetic_scene
from rtgs.image2gs.fit import FitConfig, fit_views
from rtgs.lift.base import bilinear_sample
from rtgs.lift.depth import AlignedDepthPrior
from rtgs.lift.gradient import GradientLifter
from rtgs.lift.hybrid import HybridLifter
from rtgs.optim.trainer import Trainer
from rtgs.render.torch_ref import TorchRasterizer

FAMILIES = ("gradient", "hybrid")
MODES = ("none", "oracle_position", "degree_shuffled_position")
PREREGISTRATION = Path("benchmarks/results/20260715_world_position_consistency_PREREG.md")

# Literal CPU-reference semantics frozen in the preregistration. The parity check below fails if
# these ever drift from TorchRasterizer rather than silently changing oracle identity purity.
_NEAR = 0.05
_CUTOFF = 12.0
_DILATION = 0.3
_MAX_ALPHA = 0.999


@dataclass(frozen=True)
class OracleGraph:
    correct_pairs: torch.Tensor
    shuffled_pairs: torch.Tensor
    assigned_gt_ids: torch.Tensor
    diagnostics: dict[str, object]


def loaded_source_hashes(root: Path) -> tuple[dict[str, str], str]:
    """Hash every loaded repository Python source plus frozen protocol inputs."""
    paths = set()
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if source is None:
            continue
        path = Path(source).resolve()
        if path.suffix == ".py" and path.is_relative_to(root) and path.is_file():
            paths.add(path)
    for relative in (PREREGISTRATION, Path("pyproject.toml")):
        path = (root / relative).resolve()
        if path.exists():
            paths.add(path)
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    aggregate = hashlib.sha256()
    for path, digest in hashes.items():
        aggregate.update(path.encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        aggregate.update(b"\n")
    return hashes, aggregate.hexdigest()


def distribution(values: torch.Tensor) -> dict[str, float | int | None]:
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


def closest_points_on_rays(
    left_origin: torch.Tensor,
    left_direction: torch.Tensor,
    right_origin: torch.Tensor,
    right_direction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Return the closest points and nonnegative Euclidean parameters on two rays."""
    left_direction = left_direction / left_direction.norm().clamp_min(1e-12)
    right_direction = right_direction / right_direction.norm().clamp_min(1e-12)
    origin_delta = left_origin - right_origin
    cosine = torch.dot(left_direction, right_direction)
    denominator = 1.0 - cosine.square()
    if float(denominator) <= 1e-8:
        raise AssertionError("oracle graph contains a near-parallel ray pair")

    left_dot = torch.dot(left_direction, origin_delta)
    right_dot = torch.dot(right_direction, origin_delta)
    line_left = (cosine * right_dot - left_dot) / denominator
    line_right = (right_dot - cosine * left_dot) / denominator
    candidates: list[tuple[torch.Tensor, torch.Tensor, float, float]] = []
    if float(line_left) >= 0.0 and float(line_right) >= 0.0:
        candidates.append(
            (
                left_origin + line_left * left_direction,
                right_origin + line_right * right_direction,
                float(line_left),
                float(line_right),
            )
        )

    boundary_right = right_dot.clamp_min(0.0)
    candidates.append(
        (
            left_origin,
            right_origin + boundary_right * right_direction,
            0.0,
            float(boundary_right),
        )
    )
    boundary_left = (-left_dot).clamp_min(0.0)
    candidates.append(
        (
            left_origin + boundary_left * left_direction,
            right_origin,
            float(boundary_left),
            0.0,
        )
    )
    candidates.append((left_origin, right_origin, 0.0, 0.0))
    return min(candidates, key=lambda item: float((item[0] - item[1]).square().sum()))


def point_to_ray_distance(
    point: torch.Tensor, origin: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    """Euclidean distance from a point to the closest nonnegative point on a ray."""
    direction = direction / direction.norm().clamp_min(1e-12)
    parameter = torch.dot(point - origin, direction).clamp_min(0.0)
    return (point - (origin + parameter * direction)).norm()


def graph_ray_diagnostics(
    scene,
    source_xy: torch.Tensor,
    source_view_ids: torch.Tensor,
    pairs: torch.Tensor,
    edge_gt_ids: list[int] | list[tuple[int, int]],
    extent: float,
) -> dict[str, object]:
    """Measure whether frozen fitted-center rays geometrically support their edge labels."""
    gaps = []
    midpoint_left_gt = []
    midpoint_right_gt = []
    left_gt_to_left_ray = []
    right_gt_to_right_ray = []
    left_parameters = []
    right_parameters = []
    for edge_index, (left_index, right_index) in enumerate(pairs.tolist()):
        left_view = int(source_view_ids[left_index])
        right_view = int(source_view_ids[right_index])
        left_origin, left_direction = scene.cameras[left_view].pixel_rays(
            source_xy[left_index : left_index + 1]
        )
        right_origin, right_direction = scene.cameras[right_view].pixel_rays(
            source_xy[right_index : right_index + 1]
        )
        left_origin = left_origin.to(source_xy)
        right_origin = right_origin.to(source_xy)
        left_direction = left_direction[0]
        right_direction = right_direction[0]
        left_point, right_point, left_parameter, right_parameter = closest_points_on_rays(
            left_origin,
            left_direction,
            right_origin,
            right_direction,
        )
        midpoint = 0.5 * (left_point + right_point)
        identities = edge_gt_ids[edge_index]
        if isinstance(identities, int):
            left_gt_id = right_gt_id = identities
        else:
            left_gt_id, right_gt_id = identities
        left_gt = scene.gt_gaussians.means[left_gt_id]
        right_gt = scene.gt_gaussians.means[right_gt_id]
        gaps.append(float((left_point - right_point).norm() / extent))
        midpoint_left_gt.append(float((midpoint - left_gt).norm() / extent))
        midpoint_right_gt.append(float((midpoint - right_gt).norm() / extent))
        left_gt_to_left_ray.append(
            float(point_to_ray_distance(left_gt, left_origin, left_direction) / extent)
        )
        right_gt_to_right_ray.append(
            float(point_to_ray_distance(right_gt, right_origin, right_direction) / extent)
        )
        left_parameters.append(left_parameter / extent)
        right_parameters.append(right_parameter / extent)
    return {
        "closest_ray_distance_over_extent": distribution(torch.tensor(gaps)),
        "midpoint_to_left_assigned_gt_over_extent": distribution(torch.tensor(midpoint_left_gt)),
        "midpoint_to_right_assigned_gt_over_extent": distribution(torch.tensor(midpoint_right_gt)),
        "left_assigned_gt_to_endpoint_ray_over_extent": distribution(
            torch.tensor(left_gt_to_left_ray)
        ),
        "right_assigned_gt_to_endpoint_ray_over_extent": distribution(
            torch.tensor(right_gt_to_right_ray)
        ),
        "left_closest_parameter_over_extent": distribution(torch.tensor(left_parameters)),
        "right_closest_parameter_over_extent": distribution(torch.tensor(right_parameters)),
    }


def reference_contributions(gaussians, camera, samples: torch.Tensor) -> torch.Tensor:
    """Evaluate per-Gaussian alpha-compositing contributions at arbitrary pixel centers."""
    means_cam = camera.world_to_cam(gaussians.means)
    z = means_cam[:, 2]
    in_front = z > _NEAR
    uv, _ = camera.project(gaussians.means)

    covariance = gaussians.covariance()
    rotation = camera.R.to(covariance)
    covariance_cam = rotation @ covariance @ rotation.T
    safe_z = z.clamp_min(_NEAR)
    jacobian = torch.zeros(gaussians.n, 2, 3, device=samples.device, dtype=samples.dtype)
    jacobian[:, 0, 0] = camera.fx / safe_z
    jacobian[:, 0, 2] = -camera.fx * means_cam[:, 0] / safe_z.square()
    jacobian[:, 1, 1] = camera.fy / safe_z
    jacobian[:, 1, 2] = -camera.fy * means_cam[:, 1] / safe_z.square()
    covariance_2d = jacobian @ covariance_cam @ jacobian.transpose(-1, -2)
    covariance_2d = covariance_2d + _DILATION * torch.eye(
        2, device=samples.device, dtype=samples.dtype
    )

    eig_max = (
        0.5 * (covariance_2d[:, 0, 0] + covariance_2d[:, 1, 1])
        + (
            0.25 * (covariance_2d[:, 0, 0] - covariance_2d[:, 1, 1]).square()
            + covariance_2d[:, 0, 1].square()
        ).sqrt()
    )
    radii = 3.0 * eig_max.clamp_min(1e-8).sqrt()
    visible = in_front & camera.in_image(uv, margin=radii.detach())
    visible_indices = torch.nonzero(visible, as_tuple=False).squeeze(-1)
    contributions = samples.new_zeros((samples.shape[0], gaussians.n))
    if visible_indices.numel() == 0:
        return contributions

    visible_indices = visible_indices[torch.argsort(z[visible_indices])]
    means_2d = uv[visible_indices]
    covariance_visible = covariance_2d[visible_indices]
    opacity = gaussians.opacity[visible_indices]
    determinant = (
        covariance_visible[:, 0, 0] * covariance_visible[:, 1, 1]
        - covariance_visible[:, 0, 1].square()
    ).clamp_min(1e-12)
    inverse_00 = covariance_visible[:, 1, 1] / determinant
    inverse_01 = -covariance_visible[:, 0, 1] / determinant
    inverse_11 = covariance_visible[:, 0, 0] / determinant

    delta = samples[:, None, :] - means_2d[None, :, :]
    mahalanobis = (
        delta[..., 0].square() * inverse_00[None, :]
        + 2.0 * delta[..., 0] * delta[..., 1] * inverse_01[None, :]
        + delta[..., 1].square() * inverse_11[None, :]
    )
    gaussian = torch.exp(-0.5 * mahalanobis.clamp_max(4 * _CUTOFF)) * (mahalanobis < _CUTOFF)
    alpha = (opacity[None, :] * gaussian).clamp(0.0, _MAX_ALPHA)
    transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=1)
    transmittance = torch.cat([torch.ones_like(transmittance[:, :1]), transmittance[:, :-1]], dim=1)
    contributions[:, visible_indices] = alpha * transmittance
    return contributions


def assert_reference_contribution_parity(gaussians, camera) -> dict[str, float]:
    """Bind the local contribution helper to the semantics-defining renderer."""
    xs = torch.arange(camera.width, dtype=gaussians.means.dtype) + 0.5
    ys = torch.arange(camera.height, dtype=gaussians.means.dtype) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    samples = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    contributions = reference_contributions(gaussians, camera, samples)
    expected_alpha = contributions.sum(dim=1).reshape(camera.height, camera.width)
    _, z = camera.project(gaussians.means)
    expected_depth = (contributions @ z).reshape(camera.height, camera.width)
    rendered = TorchRasterizer().render(gaussians, camera)
    alpha_error = (expected_alpha - rendered.alpha).abs()
    depth_error = (expected_depth - rendered.depth).abs()
    if not torch.allclose(expected_alpha, rendered.alpha, atol=1e-6, rtol=1e-6):
        raise AssertionError("oracle contribution helper does not reproduce reference alpha")
    if not torch.allclose(expected_depth, rendered.depth, atol=1e-6, rtol=1e-6):
        raise AssertionError("oracle contribution helper does not reproduce reference depth")
    return {
        "alpha_max_abs_error": float(alpha_error.max()),
        "depth_max_abs_error": float(depth_error.max()),
    }


def build_oracle_graph(
    train_scene,
    initial_means: torch.Tensor,
    source_xy: torch.Tensor,
    source_view_ids: torch.Tensor,
    source_ranges: list[tuple[int, int]],
    args: argparse.Namespace,
) -> OracleGraph:
    """Build the frozen privileged GT-identity graph and degree-exact derangement."""
    if train_scene.gt_gaussians is None or train_scene.gt_depths is None:
        raise ValueError("oracle graph requires synthetic GT Gaussians and training depths")
    if source_xy.shape != (source_view_ids.numel(), 2):
        raise ValueError("retained source pixel layout is inconsistent")
    if initial_means.shape != (source_view_ids.numel(), 3):
        raise ValueError("step-zero world means are inconsistent with the retained layout")
    if len(source_ranges) != train_scene.n_views:
        raise ValueError("one retained source range is required per training view")

    _, extent = train_scene.center_and_extent()
    gt = train_scene.gt_gaussians
    representatives: list[dict[int, int]] = []
    representative_meta: list[dict[int, dict[str, float]]] = []
    parity = []
    per_view_counts = []
    all_pixel_distances = []
    all_depth_discrepancies = []
    all_contributions = []
    all_purities = []

    with torch.no_grad():
        for view_index, ((start, end), camera, gt_depth) in enumerate(
            zip(source_ranges, train_scene.cameras, train_scene.gt_depths)
        ):
            xy = source_xy[start:end]
            if xy.numel() == 0 or not bool((source_view_ids[start:end] == view_index).all()):
                raise AssertionError(f"view {view_index} has an invalid retained source layout")
            parity.append(assert_reference_contribution_parity(gt, camera))
            gt_uv, gt_z = camera.project(gt.means)
            sampled_expected_depth = bilinear_sample(gt_depth.to(gt_uv), gt_uv)
            eligible = (
                (gt_z > _NEAR)
                & camera.in_image(gt_uv)
                & torch.isfinite(sampled_expected_depth)
                & (sampled_expected_depth > _NEAR)
                & ((sampled_expected_depth - gt_z).abs() / extent <= args.match_depth_tolerance)
            )
            eligible_ids = torch.nonzero(eligible, as_tuple=False).squeeze(-1)
            if eligible_ids.numel() == 0:
                raise AssertionError(f"view {view_index} has no eligible oracle identities")

            distances = torch.cdist(xy, gt_uv[eligible_ids])
            fit_to_gt = distances.argmin(dim=1)
            gt_to_fit = distances.argmin(dim=0)
            columns = torch.arange(eligible_ids.numel())
            mutual = fit_to_gt[gt_to_fit] == columns
            selected_distance = distances[gt_to_fit, columns]
            accepted_columns = columns[mutual & (selected_distance <= args.match_pixel_tolerance)]
            contributions = reference_contributions(gt, camera, xy)
            view_representatives: dict[int, int] = {}
            view_meta: dict[int, dict[str, float]] = {}
            for column in accepted_columns.tolist():
                gt_id = int(eligible_ids[column])
                local_index = int(gt_to_fit[column])
                contribution_row = contributions[local_index]
                total_contribution = contribution_row.sum().clamp_min(1e-12)
                own_contribution = contribution_row[gt_id]
                purity = own_contribution / total_contribution
                if float(own_contribution) < args.match_min_contribution:
                    continue
                if float(purity) < args.match_min_purity:
                    continue
                if int(contribution_row.argmax()) != gt_id:
                    continue
                pixel_distance = float(distances[local_index, column])
                depth_discrepancy = float(
                    (sampled_expected_depth[gt_id] - gt_z[gt_id]).abs() / extent
                )
                view_representatives[gt_id] = start + local_index
                view_meta[gt_id] = {
                    "pixel_distance": pixel_distance,
                    "depth_discrepancy_over_extent": depth_discrepancy,
                    "contribution": float(own_contribution),
                    "purity": float(purity),
                }
                all_pixel_distances.append(pixel_distance)
                all_depth_discrepancies.append(depth_discrepancy)
                all_contributions.append(float(own_contribution))
                all_purities.append(float(purity))
            if len(view_representatives) < args.min_representatives_per_view:
                raise AssertionError(
                    f"view {view_index} has only {len(view_representatives)} oracle representatives"
                )
            representatives.append(view_representatives)
            representative_meta.append(view_meta)
            per_view_counts.append(len(view_representatives))

    correct_edges: list[tuple[int, int]] = []
    shuffled_edges: list[tuple[int, int]] = []
    correct_edge_gt_ids: list[int] = []
    shuffled_edge_gt_ids: list[tuple[int, int]] = []
    block_diagnostics: dict[str, dict[str, float | int]] = {}
    all_angles = []
    for left_view, right_view in itertools.combinations(range(train_scene.n_views), 2):
        common_ids = sorted(
            set(representatives[left_view]).intersection(representatives[right_view])
        )
        kept_ids = []
        kept_angles = []
        for gt_id in common_ids:
            left_index = representatives[left_view][gt_id]
            right_index = representatives[right_view][gt_id]
            _, left_direction = train_scene.cameras[left_view].pixel_rays(
                source_xy[left_index : left_index + 1]
            )
            _, right_direction = train_scene.cameras[right_view].pixel_rays(
                source_xy[right_index : right_index + 1]
            )
            left_direction = torch.nn.functional.normalize(left_direction, dim=-1)
            right_direction = torch.nn.functional.normalize(right_direction, dim=-1)
            line_cosine = (left_direction * right_direction).sum().abs().clamp(0.0, 1.0)
            angle = math.degrees(math.acos(float(line_cosine)))
            if angle >= args.match_min_ray_angle_deg:
                kept_ids.append(gt_id)
                kept_angles.append(angle)
        if len(kept_ids) < 2:
            continue
        left = [representatives[left_view][gt_id] for gt_id in kept_ids]
        right = [representatives[right_view][gt_id] for gt_id in kept_ids]
        shuffled_left = list(left)
        shuffled_right = right[1:] + right[:1]
        shuffled_ids = kept_ids[1:] + kept_ids[:1]
        if Counter(left) != Counter(shuffled_left) or Counter(right) != Counter(shuffled_right):
            raise AssertionError("shuffled graph changes a block endpoint multiset")
        correct_edges.extend(zip(left, right))
        shuffled_edges.extend(zip(shuffled_left, shuffled_right))
        correct_edge_gt_ids.extend(kept_ids)
        shuffled_edge_gt_ids.extend(zip(kept_ids, shuffled_ids))
        all_angles.extend(kept_angles)
        baseline = float(
            (
                train_scene.cameras[left_view].position - train_scene.cameras[right_view].position
            ).norm()
            / extent
        )
        block_diagnostics[f"{left_view}-{right_view}"] = {
            "edge_count": len(kept_ids),
            "camera_baseline_over_extent": baseline,
            "ray_angle_min_deg": min(kept_angles),
            "ray_angle_max_deg": max(kept_angles),
            "left_endpoint_multiset_exact": True,
            "right_endpoint_multiset_exact": True,
        }

    correct_pairs = torch.tensor(correct_edges, dtype=torch.long)
    shuffled_pairs = torch.tensor(shuffled_edges, dtype=torch.long)
    if correct_pairs.ndim != 2 or correct_pairs.shape[1] != 2:
        raise AssertionError("oracle graph is empty")
    if correct_pairs.shape != shuffled_pairs.shape:
        raise AssertionError("oracle and shuffled graphs differ in edge count")
    if not bool((correct_pairs[:, 0] < correct_pairs[:, 1]).all()):
        raise AssertionError("oracle graph indices are not canonical")
    if not bool((shuffled_pairs[:, 0] < shuffled_pairs[:, 1]).all()):
        raise AssertionError("shuffled graph indices are not canonical")
    if torch.unique(correct_pairs, dim=0).shape[0] != correct_pairs.shape[0]:
        raise AssertionError("oracle graph contains duplicate edges")
    if torch.unique(shuffled_pairs, dim=0).shape[0] != shuffled_pairs.shape[0]:
        raise AssertionError("shuffled graph contains duplicate edges")
    if set(map(tuple, correct_pairs.tolist())).intersection(map(tuple, shuffled_pairs.tolist())):
        raise AssertionError("shuffled graph retains an exact oracle edge")
    if any(left_id == right_id for left_id, right_id in shuffled_edge_gt_ids):
        raise AssertionError("shuffled graph retains a semantic GT-identity edge")

    correct_degree = torch.bincount(correct_pairs.flatten(), minlength=source_view_ids.numel())
    shuffled_degree = torch.bincount(shuffled_pairs.flatten(), minlength=source_view_ids.numel())
    if not torch.equal(correct_degree, shuffled_degree):
        raise AssertionError("shuffled graph changes endpoint degree")
    correct_source_pairs = Counter(
        zip(
            source_view_ids[correct_pairs[:, 0]].tolist(),
            source_view_ids[correct_pairs[:, 1]].tolist(),
        )
    )
    shuffled_source_pairs = Counter(
        zip(
            source_view_ids[shuffled_pairs[:, 0]].tolist(),
            source_view_ids[shuffled_pairs[:, 1]].tolist(),
        )
    )
    if correct_source_pairs != shuffled_source_pairs:
        raise AssertionError("shuffled graph changes source-view pair counts")

    represented = correct_degree > 0
    represented_views = sorted(torch.unique(source_view_ids[represented]).tolist())
    represented_gt_ids = sorted(set(correct_edge_gt_ids))
    if correct_pairs.shape[0] < args.min_graph_edges:
        raise AssertionError("oracle graph misses the frozen edge-count floor")
    if int(represented.sum()) < args.min_graph_nodes:
        raise AssertionError("oracle graph misses the frozen represented-node floor")
    if len(block_diagnostics) < args.min_graph_blocks:
        raise AssertionError("oracle graph misses the frozen view-pair-block floor")
    if len(represented_gt_ids) < args.min_graph_gt_ids:
        raise AssertionError("oracle graph misses the frozen represented-identity floor")
    if represented_views != list(range(train_scene.n_views)):
        raise AssertionError("oracle graph does not represent all training views")

    assigned_gt_ids = torch.full((source_view_ids.numel(),), -1, dtype=torch.long)
    for view_representatives in representatives:
        for gt_id, retained_index in view_representatives.items():
            if represented[retained_index]:
                assigned_gt_ids[retained_index] = gt_id
    if bool((assigned_gt_ids[represented] < 0).any()):
        raise AssertionError("represented graph node is missing its assigned GT identity")

    degree_values = correct_degree[represented]
    correct_initial_residual = (
        initial_means[correct_pairs[:, 0]].sub(initial_means[correct_pairs[:, 1]]).abs().sum(dim=-1)
        / extent
    )
    shuffled_initial_residual = (
        initial_means[shuffled_pairs[:, 0]]
        .sub(initial_means[shuffled_pairs[:, 1]])
        .abs()
        .sum(dim=-1)
        / extent
    )
    diagnostics: dict[str, object] = {
        "edge_count": int(correct_pairs.shape[0]),
        "represented_node_count": int(represented.sum()),
        "represented_node_fraction": float(represented.float().mean()),
        "represented_gt_identity_count": len(represented_gt_ids),
        "represented_gt_identities": represented_gt_ids,
        "view_pair_block_count": len(block_diagnostics),
        "represented_views": represented_views,
        "representatives_per_view": per_view_counts,
        "blocks": block_diagnostics,
        "correct_pairs": correct_pairs.tolist(),
        "degree_shuffled_pairs": shuffled_pairs.tolist(),
        "correct_edge_gt_ids": correct_edge_gt_ids,
        "degree_shuffled_edge_gt_ids": [list(pair) for pair in shuffled_edge_gt_ids],
        "represented_node_gt_assignments": [
            [int(index), int(assigned_gt_ids[index])]
            for index in torch.nonzero(represented, as_tuple=False).squeeze(-1).tolist()
        ],
        "pixel_distance": distribution(torch.tensor(all_pixel_distances)),
        "depth_discrepancy_over_extent": distribution(torch.tensor(all_depth_discrepancies)),
        "assigned_contribution": distribution(torch.tensor(all_contributions)),
        "assigned_purity": distribution(torch.tensor(all_purities)),
        "ray_angle_deg": distribution(torch.tensor(all_angles)),
        "step0_correct_edge_normalized_l1": distribution(correct_initial_residual),
        "step0_degree_shuffled_edge_normalized_l1": distribution(shuffled_initial_residual),
        "correct_edge_ray_geometry": graph_ray_diagnostics(
            train_scene,
            source_xy,
            source_view_ids,
            correct_pairs,
            correct_edge_gt_ids,
            extent,
        ),
        "degree_shuffled_edge_ray_geometry": graph_ray_diagnostics(
            train_scene,
            source_xy,
            source_view_ids,
            shuffled_pairs,
            shuffled_edge_gt_ids,
            extent,
        ),
        "endpoint_degree": distribution(degree_values),
        "reference_contribution_parity": parity,
        "correct_pairs_sha256": tensor_collection_hash([("position_pairs", correct_pairs)]),
        "shuffled_pairs_sha256": tensor_collection_hash([("position_pairs", shuffled_pairs)]),
        "endpoint_degree_sha256": tensor_collection_hash([("endpoint_degree", correct_degree)]),
        "assigned_gt_ids_sha256": tensor_collection_hash([("assigned_gt_ids", assigned_gt_ids)]),
        "semantic_edge_overlap_count": 0,
        "exact_edge_overlap_count": 0,
        "degree_exact": True,
        "source_pair_counts_exact": True,
        "block_endpoint_multisets_exact": True,
    }
    return OracleGraph(
        correct_pairs=correct_pairs,
        shuffled_pairs=shuffled_pairs,
        assigned_gt_ids=assigned_gt_ids,
        diagnostics=diagnostics,
    )


def pairs_for_mode(graph: OracleGraph, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode == "oracle_position":
        return graph.correct_pairs
    if mode == "degree_shuffled_position":
        return graph.shuffled_pairs
    raise ValueError(f"unknown position mode {mode!r}")


def make_lifter(
    family: str,
    mode: str,
    args: argparse.Namespace,
    seed: int,
    corrupted_depths: list[torch.Tensor],
    *,
    iterations: int | None = None,
    lr: float | None = None,
    position_lambda: float | None = None,
):
    active_lambda = (
        (0.0 if mode == "none" else args.position_lambda)
        if position_lambda is None
        else position_lambda
    )
    common = {
        "iterations": args.lift_iters if iterations is None else iterations,
        "lr": args.lr if lr is None else lr,
        "rasterizer": "torch",
        "min_weight": args.min_weight,
        "optimize_rotation": False,
        "optimize_scale": False,
        "merge": False,
        "seed": seed,
        "depth_anchor_mode": "legacy",
        "photometric_supervision_mode": "all",
        "position_consistency_lambda": active_lambda,
        "position_consistency_beta": args.position_beta,
    }
    if family == "gradient":
        return GradientLifter(
            depth_jitter=args.gradient_depth_jitter,
            depth_prior_lambda=args.gradient_depth_prior_lambda,
            **common,
        )
    if family == "hybrid":
        return HybridLifter(
            backend=ListDepthBackend(corrupted_depths),
            depth_prior_lambda=args.hybrid_depth_prior_lambda,
            merge_color_bin_size=None,
            depth_jitter=args.hybrid_depth_jitter,
            **common,
        )
    raise ValueError(f"unknown family {family!r}")


def core_lifter(lifter) -> GradientLifter:
    return lifter if isinstance(lifter, GradientLifter) else lifter.gradient


def execute_lift(lifter, gaussians2d, train_scene, pairs: torch.Tensor | None):
    if pairs is None:
        return lifter.lift(gaussians2d, train_scene)
    return lifter.lift_with_position_pairs(gaussians2d, train_scene, pairs)


def assert_layout_equal(reference: GradientLifter, other: GradientLifter, context: str) -> None:
    if not torch.equal(reference.source_view_ids_before_merge, other.source_view_ids_before_merge):
        raise AssertionError(f"{context} changes retained source IDs")
    if reference.source_view_ranges_before_merge != other.source_view_ranges_before_merge:
        raise AssertionError(f"{context} changes retained source ranges")
    if not torch.equal(reference.source_xy_before_merge, other.source_xy_before_merge):
        raise AssertionError(f"{context} changes retained source pixels")


def assert_family_invariants(
    family: str,
    gaussians2d,
    train_scene,
    corrupted_depths,
    graph: OracleGraph,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    _, extent = train_scene.center_and_extent()
    zero_outputs = {}
    zero_cores = {}
    for mode in MODES:
        lifter = make_lifter(
            family,
            mode,
            args,
            seed,
            corrupted_depths,
            iterations=0,
        )
        zero_outputs[mode] = execute_lift(
            lifter, gaussians2d, train_scene, pairs_for_mode(graph, mode)
        )
        zero_cores[mode] = core_lifter(lifter)
    reference_output = zero_outputs["none"]
    reference_core = zero_cores["none"]
    for mode in MODES[1:]:
        assert_equal_fields(reference_output, zero_outputs[mode], f"{family}/{mode} step 0")
        assert_layout_equal(reference_core, zero_cores[mode], f"{family}/{mode} step 0")
        if not torch.equal(
            reference_core.initial_ray_fractions_before_merge,
            zero_cores[mode].initial_ray_fractions_before_merge,
        ):
            raise AssertionError(f"{family}/{mode} changes initial ray fractions")
        if not torch.equal(
            reference_core.final_ray_fractions_before_merge,
            zero_cores[mode].final_ray_fractions_before_merge,
        ):
            raise AssertionError(f"{family}/{mode} changes step-zero final ray fractions")
    if reference_core.position_pairs_before_merge is not None:
        raise AssertionError(f"{family}/none unexpectedly records position pairs")
    if not torch.equal(
        zero_cores["oracle_position"].position_pairs_before_merge, graph.correct_pairs
    ):
        raise AssertionError(f"{family} changed the correct position graph")
    if not torch.equal(
        zero_cores["degree_shuffled_position"].position_pairs_before_merge,
        graph.shuffled_pairs,
    ):
        raise AssertionError(f"{family} changed the shuffled position graph")
    if (
        tensor_collection_hash(
            [("position_pairs", zero_cores["oracle_position"].position_pairs_before_merge)]
        )
        != graph.diagnostics["correct_pairs_sha256"]
    ):
        raise AssertionError(f"{family} correct position graph hash changed")
    if (
        tensor_collection_hash(
            [
                (
                    "position_pairs",
                    zero_cores["degree_shuffled_position"].position_pairs_before_merge,
                )
            ]
        )
        != graph.diagnostics["shuffled_pairs_sha256"]
    ):
        raise AssertionError(f"{family} shuffled position graph hash changed")

    zero_lambda_outputs = {}
    zero_lambda_cores = {}
    for mode in MODES:
        lifter = make_lifter(
            family,
            mode,
            args,
            seed,
            corrupted_depths,
            iterations=args.rng_check_iters,
            position_lambda=0.0,
        )
        zero_lambda_outputs[mode] = execute_lift(
            lifter, gaussians2d, train_scene, pairs_for_mode(graph, mode)
        )
        zero_lambda_cores[mode] = core_lifter(lifter)
    for mode in MODES[1:]:
        assert_equal_fields(
            zero_lambda_outputs["none"],
            zero_lambda_outputs[mode],
            f"{family}/{mode} zero-position-lambda check",
        )
        assert_layout_equal(
            zero_lambda_cores["none"],
            zero_lambda_cores[mode],
            f"{family}/{mode} zero-position-lambda check",
        )
        if zero_lambda_cores[mode].history != zero_lambda_cores["none"].history:
            raise AssertionError(f"{family}/{mode} changes total history at zero position lambda")
        if zero_lambda_cores[mode].anchor_history != zero_lambda_cores["none"].anchor_history:
            raise AssertionError(f"{family}/{mode} changes anchor history at zero position lambda")
        if (
            zero_lambda_cores[mode].target_view_history
            != zero_lambda_cores["none"].target_view_history
        ):
            raise AssertionError(f"{family}/{mode} changes target RNG at zero position lambda")
        if not torch.equal(
            zero_lambda_cores[mode].final_ray_fractions_before_merge,
            zero_lambda_cores["none"].final_ray_fractions_before_merge,
        ):
            raise AssertionError(f"{family}/{mode} changes rays at zero position lambda")

    return {
        "step0_identical": True,
        "zero_position_lambda_outputs_identical": True,
        "zero_position_lambda_histories_identical": True,
        "zero_position_lambda_target_schedule_identical": True,
        "primitive_count": reference_output.n,
        "source_view_ids_sha256": tensor_collection_hash(
            [("source_view_ids", reference_core.source_view_ids_before_merge)]
        ),
        "source_xy_sha256": tensor_collection_hash(
            [("source_xy", reference_core.source_xy_before_merge)]
        ),
        "source_view_ranges": reference_core.source_view_ranges_before_merge,
        "initial_ray_fractions_sha256": tensor_collection_hash(
            [("initial_ray_fractions", reference_core.initial_ray_fractions_before_merge)]
        ),
        "correct_pairs_sha256": graph.diagnostics["correct_pairs_sha256"],
        "shuffled_pairs_sha256": graph.diagnostics["shuffled_pairs_sha256"],
        "step0_correct_edge_residual": pair_residual_diagnostics(
            reference_output.means, graph.correct_pairs, extent
        ),
        "step0_degree_shuffled_edge_residual": pair_residual_diagnostics(
            reference_output.means, graph.shuffled_pairs, extent
        ),
    }


def finite_result(result, core: GradientLifter, context: str) -> None:
    for field in ("means", "quats", "log_scales", "opacity", "sh"):
        if not bool(torch.isfinite(getattr(result, field)).all()):
            raise AssertionError(f"{context} has non-finite {field}")
    for name, values in (
        ("total loss", core.history),
        ("anchor", core.anchor_history),
        ("position", core.position_history),
    ):
        if any(not math.isfinite(value) for value in values):
            raise AssertionError(f"{context} has non-finite {name} history")
    for name, fractions in (
        ("initial", core.initial_ray_fractions_before_merge),
        ("final", core.final_ray_fractions_before_merge),
    ):
        if fractions is None or not bool(torch.isfinite(fractions).all()):
            raise AssertionError(f"{context} has invalid {name} ray fractions")
        if not bool(((fractions > 0) & (fractions < 1)).all()):
            raise AssertionError(f"{context} leaves the bounded ray interval")


def pair_residual_diagnostics(
    means: torch.Tensor, pairs: torch.Tensor, extent: float
) -> dict[str, float | int | None]:
    residual = means[pairs[:, 0]].sub(means[pairs[:, 1]]).abs().sum(dim=-1) / extent
    return distribution(residual)


def assigned_gt_diagnostics(
    scene, means: torch.Tensor, assigned_gt_ids: torch.Tensor
) -> dict[str, float | int | None]:
    represented = assigned_gt_ids >= 0
    if scene.gt_gaussians is None or not bool(represented.any()):
        raise AssertionError("assigned-GT diagnostics require represented oracle identities")
    _, extent = scene.center_and_extent()
    distance = (means[represented] - scene.gt_gaussians.means[assigned_gt_ids[represented]]).norm(
        dim=-1
    ) / extent
    return distribution(distance)


def ray_fraction_diagnostics(fractions: torch.Tensor) -> dict[str, float | int]:
    values = fractions.detach().double()
    return {
        "count": int(values.numel()),
        "min": float(values.min()),
        "p01": float(torch.quantile(values, 0.01)),
        "median": float(values.median()),
        "p99": float(torch.quantile(values, 0.99)),
        "max": float(values.max()),
        "near_saturation_fraction": float((values <= 0.02).double().mean()),
        "far_saturation_fraction": float((values >= 0.98).double().mean()),
    }


def evaluate_arm(
    family: str,
    mode: str,
    gaussians2d,
    scene,
    train_scene,
    clean_depths,
    corrupted_depths,
    diagnostic_priors,
    corruption_masks,
    graph: OracleGraph,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    lifter = make_lifter(family, mode, args, seed, corrupted_depths)
    pairs = pairs_for_mode(graph, mode)
    started = time.perf_counter()
    result = execute_lift(lifter, gaussians2d, train_scene, pairs)
    lift_seconds = time.perf_counter() - started
    core = core_lifter(lifter)
    finite_result(result, core, f"{family}/{mode}")
    source_view_ids = core.source_view_ids_before_merge
    if source_view_ids is None:
        raise AssertionError(f"{family}/{mode} did not retain source-view labels")
    if mode == "none" and core.position_history:
        raise AssertionError(f"{family}/none unexpectedly computed a position history")
    if mode != "none" and len(core.position_history) != args.lift_iters:
        raise AssertionError(f"{family}/{mode} has an incomplete position history")
    renderer = TorchRasterizer()
    _, extent = scene.center_and_extent()
    return {
        "n_initial": result.n,
        "lift_seconds": lift_seconds,
        "loss_history": list(core.history),
        "loss_history_checkpoints": history_checkpoints(core.history),
        "anchor_history": list(core.anchor_history),
        "anchor_history_checkpoints": history_checkpoints(core.anchor_history),
        "position_history": list(core.position_history),
        "position_history_checkpoints": history_checkpoints(core.position_history),
        "target_view_history": core.target_view_history,
        "target_view_history_sha256": integer_sequence_hash(core.target_view_history),
        "rendered_count_history": core.rendered_count_history,
        "position_pairs_sha256": (
            None
            if core.position_pairs_before_merge is None
            else tensor_collection_hash([("position_pairs", core.position_pairs_before_merge)])
        ),
        "initial_ray_fractions": ray_fraction_diagnostics(core.initial_ray_fractions_before_merge),
        "final_ray_fractions": ray_fraction_diagnostics(core.final_ray_fractions_before_merge),
        "correct_edge_residual": pair_residual_diagnostics(
            result.means, graph.correct_pairs, extent
        ),
        "shuffled_edge_residual": pair_residual_diagnostics(
            result.means, graph.shuffled_pairs, extent
        ),
        "assigned_gt": assigned_gt_diagnostics(train_scene, result.means, graph.assigned_gt_ids),
        "initial_test": Trainer.evaluate_metrics(
            scene, result, renderer, indices=scene.testing_views
        ),
        "training": Trainer.evaluate_metrics(
            train_scene, result, renderer, indices=train_scene.training_views
        ),
        "cross_only_training_l1": cross_only_training_l1(train_scene, result, source_view_ids),
        "held_out_geometry": held_out_geometry(scene, result, scene.testing_views),
        "source_depth": source_depth_diagnostics(
            result,
            gaussians2d,
            train_scene,
            clean_depths,
            diagnostic_priors,
            corruption_masks,
            args.min_weight,
        ),
        "nearest_gt": nearest_gt_diagnostics(scene, result),
    }


SUMMARY_PATHS = {
    "test_psnr": ("initial_test", "psnr"),
    "test_ssim": ("initial_test", "ssim"),
    "train_psnr": ("training", "psnr"),
    "cross_only_training_l1": ("cross_only_training_l1",),
    "heldout_depth_rmse": ("held_out_geometry", "depth_rmse_over_extent"),
    "alpha_iou": ("held_out_geometry", "alpha_iou"),
    "coverage": ("held_out_geometry", "foreground_coverage"),
    "source_all_p90": ("source_depth", "all", "p90"),
    "source_corrupted_p90": ("source_depth", "corrupted", "p90"),
    "source_all_median": ("source_depth", "all", "median"),
    "nearest_gt_median": ("nearest_gt", "median_over_extent"),
    "nearest_gt_p90": ("nearest_gt", "p90_over_extent"),
    "correct_edge_median": ("correct_edge_residual", "median"),
    "correct_edge_p90": ("correct_edge_residual", "p90"),
    "shuffled_edge_p90": ("shuffled_edge_residual", "p90"),
    "assigned_gt_median": ("assigned_gt", "median"),
    "assigned_gt_p90": ("assigned_gt", "p90"),
    "near_bound_fraction": ("final_ray_fractions", "near_saturation_fraction"),
    "far_bound_fraction": ("final_ray_fractions", "far_saturation_fraction"),
    "lift_seconds": ("lift_seconds",),
}


def metric_samples(runs, family: str, mode: str, path: tuple[str, ...]) -> list[float]:
    samples = []
    for run in runs:
        value = run["families"][family][mode]
        for key in path:
            value = value[key]
        if value is not None:
            samples.append(float(value))
    return samples


def aggregate(samples: list[float]) -> dict[str, object]:
    if not samples:
        return {"mean": None, "std": None, "samples": []}
    return {
        "mean": statistics.fmean(samples),
        "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "samples": samples,
    }


def summarize(runs) -> dict[str, object]:
    return {
        family: {
            mode: {
                name: aggregate(metric_samples(runs, family, mode, path))
                for name, path in SUMMARY_PATHS.items()
            }
            for mode in MODES
        }
        for family in FAMILIES
    }


def metric_mean(summary, family: str, mode: str, metric: str) -> float:
    value = summary[family][mode][metric]["mean"]
    if value is None:
        raise ValueError(f"{family}/{mode}/{metric} has no finite samples")
    return float(value)


def lower_wins(summary, family: str, left: str, right: str, metric: str) -> int:
    left_samples = summary[family][left][metric]["samples"]
    right_samples = summary[family][right][metric]["samples"]
    return sum(
        left_value < right_value for left_value, right_value in zip(left_samples, right_samples)
    )


def metric_gain(summary, family: str, mode: str, metric: str) -> float:
    baseline = metric_mean(summary, family, "none", metric)
    value = metric_mean(summary, family, mode, metric)
    return (baseline - value) / max(abs(baseline), 1e-12)


def family_decision(summary, family: str) -> dict[str, object]:
    primary_metrics = ["heldout_depth_rmse", "source_all_p90"]
    if family == "hybrid":
        primary_metrics.append("source_corrupted_p90")
    gains = {
        metric: metric_gain(summary, family, "oracle_position", metric)
        for metric in primary_metrics
    }
    shuffled_gains = {
        metric: metric_gain(summary, family, "degree_shuffled_position", metric)
        for metric in primary_metrics
    }
    baseline_wins = {
        metric: lower_wins(summary, family, "oracle_position", "none", metric)
        for metric in primary_metrics
    }
    shuffled_wins = {
        metric: lower_wins(
            summary,
            family,
            "oracle_position",
            "degree_shuffled_position",
            metric,
        )
        for metric in primary_metrics
    }

    correct_edge_gain = metric_gain(summary, family, "oracle_position", "correct_edge_p90")
    correct_edge_wins = lower_wins(summary, family, "oracle_position", "none", "correct_edge_p90")
    assigned_gt_gain = metric_gain(summary, family, "oracle_position", "assigned_gt_p90")
    assigned_gt_wins = lower_wins(summary, family, "oracle_position", "none", "assigned_gt_p90")
    engagement = correct_edge_gain >= 0.25 and correct_edge_wins == 3
    local_geometry = assigned_gt_gain >= 0.20 and assigned_gt_wins >= 2

    psnr_delta = metric_mean(summary, family, "oracle_position", "test_psnr") - metric_mean(
        summary, family, "none", "test_psnr"
    )
    coverage_delta = metric_mean(summary, family, "oracle_position", "coverage") - metric_mean(
        summary, family, "none", "coverage"
    )
    iou_delta = metric_mean(summary, family, "oracle_position", "alpha_iou") - metric_mean(
        summary, family, "none", "alpha_iou"
    )
    material = (
        gains["heldout_depth_rmse"] >= 0.02
        and gains["source_all_p90"] >= 0.10
        and baseline_wins["heldout_depth_rmse"] >= 2
        and baseline_wins["source_all_p90"] >= 2
        and psnr_delta >= -0.10
        and coverage_delta >= -0.02
        and iou_delta >= -0.02
        and engagement
    )
    if family == "hybrid":
        material = (
            material
            and gains["source_corrupted_p90"] >= 0.15
            and baseline_wins["source_corrupted_p90"] >= 2
        )
    control_separation = all(
        shuffled_wins[metric] >= 2 and shuffled_gains[metric] <= 0.5 * gains[metric]
        for metric in primary_metrics
    )
    attributed = material and control_separation
    generic_graph_regularization = material and not control_separation
    return {
        "oracle_gain_fraction": gains,
        "degree_shuffled_gain_fraction": shuffled_gains,
        "oracle_vs_none_seed_wins": baseline_wins,
        "oracle_vs_shuffled_seed_wins": shuffled_wins,
        "correct_edge_p90_gain_fraction": correct_edge_gain,
        "correct_edge_p90_seed_wins": correct_edge_wins,
        "position_engagement_pass": engagement,
        "assigned_gt_p90_gain_fraction": assigned_gt_gain,
        "assigned_gt_p90_seed_wins": assigned_gt_wins,
        "local_geometry_mechanism_pass": local_geometry,
        "oracle_vs_none_psnr_delta_db": psnr_delta,
        "oracle_vs_none_coverage_delta": coverage_delta,
        "oracle_vs_none_alpha_iou_delta": iou_delta,
        "material_effect_pass": material,
        "degree_shuffled_control_separation_pass": control_separation,
        "generic_graph_regularization_detected": generic_graph_regularization,
        "correspondence_attribution_pass": attributed,
        "topology_utility_without_localization": attributed and not local_geometry,
    }


def decision(summary) -> dict[str, object]:
    families = {family: family_decision(summary, family) for family in FAMILIES}
    passed = [family for family in FAMILIES if families[family]["correspondence_attribution_pass"]]
    locally_working = [
        family
        for family in FAMILIES
        if families[family]["position_engagement_pass"]
        and families[family]["local_geometry_mechanism_pass"]
    ]
    generic = [
        family for family in FAMILIES if families[family]["generic_graph_regularization_detected"]
    ]
    if len(passed) == len(FAMILIES):
        next_action = "deployable_train_only_matcher"
    elif passed:
        next_action = "family_scoped_deployable_matcher"
    elif generic:
        next_action = "generic_graph_regularization_stop"
    elif locally_working:
        next_action = "denser_train_only_matcher_without_loss_sweep"
    elif any(families[family]["position_engagement_pass"] for family in FAMILIES):
        next_action = "local_plane_normal_consistency"
    else:
        next_action = "position_intervention_non_engaging"
    return {
        "families": families,
        "attributed_families": passed,
        "generic_graph_regularization_families": generic,
        "locally_working_families": locally_working,
        "both_families_attributed": len(passed) == len(FAMILIES),
        "stop_position_loss_hyperparameter_sweeps": True,
        "next_action": next_action,
        "production_default_change_authorized": False,
        "oracle_topology_is_deployable": False,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parent.parent
    torch.set_num_threads(args.threads)
    runs = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        scene = make_synthetic_scene(
            n_gaussians=args.gt_gaussians,
            n_cameras=args.views,
            image_size=args.image_size,
            seed=seed,
        )
        scene.test_indices = [
            index for index in range(args.views) if index % args.test_every == args.test_every - 1
        ]
        scene.train_indices = [
            index for index in range(args.views) if index not in scene.test_indices
        ]
        train_scene = scene.subset(scene.training_views)
        fit_config = FitConfig(
            n_gaussians=args.fit_gaussians,
            iterations=args.fit_iters,
            log_every=max(args.fit_iters, 1),
        )
        fit_started = time.perf_counter()
        gaussians2d, fit_history = fit_views(train_scene.images, fit_config, seed=seed)
        fit_seconds = time.perf_counter() - fit_started
        clean_depths = [depth.detach().clone() for depth in train_scene.gt_depths]
        corrupted_priors, corruption_masks = make_priors(
            clean_depths, "corrupted", seed, args.block_size
        )
        corrupted_depths = [prior.depth.detach().clone() for prior in corrupted_priors]
        diagnostic_priors = [
            AlignedDepthPrior(
                depth=depth,
                confidence=(torch.isfinite(depth) & (depth > _NEAR)).to(depth),
            )
            for depth in corrupted_depths
        ]

        layout_lifter = make_lifter(
            "gradient",
            "none",
            args,
            seed,
            corrupted_depths,
            iterations=0,
        )
        layout_output = layout_lifter.lift(gaussians2d, train_scene)
        layout_core = core_lifter(layout_lifter)
        graph = build_oracle_graph(
            train_scene,
            layout_output.means,
            layout_core.source_xy_before_merge,
            layout_core.source_view_ids_before_merge,
            layout_core.source_view_ranges_before_merge,
            args,
        )
        if layout_output.n != layout_core.source_xy_before_merge.shape[0]:
            raise AssertionError("layout output and retained source pixels differ")

        seed_result: dict[str, object] = {
            "seed": seed,
            "global_train_indices": scene.training_views,
            "global_test_indices": scene.testing_views,
            "local_to_global_train_indices": scene.training_views,
            "fit_seconds": fit_seconds,
            "fit_psnr_mean": statistics.fmean(item["final_psnr"] for item in fit_history),
            "fitted_tensors_sha256": fitted_tensor_hash(gaussians2d),
            "corrupted_depths_sha256": depth_tensor_hash(corrupted_depths),
            "layout_source_xy_sha256": tensor_collection_hash(
                [("source_xy", layout_core.source_xy_before_merge)]
            ),
            "layout_source_view_ids_sha256": tensor_collection_hash(
                [("source_view_ids", layout_core.source_view_ids_before_merge)]
            ),
            "oracle_graph": graph.diagnostics,
            "invariants": {},
            "families": {},
        }
        for family in FAMILIES:
            seed_result["invariants"][family] = assert_family_invariants(
                family,
                gaussians2d,
                train_scene,
                corrupted_depths,
                graph,
                args,
                seed,
            )
            seed_result["families"][family] = {}
            for mode in MODES:
                seed_result["families"][family][mode] = evaluate_arm(
                    family,
                    mode,
                    gaussians2d,
                    scene,
                    train_scene,
                    clean_depths,
                    corrupted_depths,
                    diagnostic_priors,
                    corruption_masks,
                    graph,
                    args,
                    seed,
                )
            family_arms = seed_result["families"][family]
            counts = {mode: family_arms[mode]["n_initial"] for mode in MODES}
            if len(set(counts.values())) != 1:
                raise AssertionError(f"{family} final primitive counts differ: {counts}")
            schedules = {mode: family_arms[mode]["target_view_history"] for mode in MODES}
            if any(schedule != schedules["none"] for schedule in schedules.values()):
                raise AssertionError(f"{family} target-view schedules differ across modes")
            if args.output is not None and set(schedules["none"]) != set(
                train_scene.training_views
            ):
                raise AssertionError(f"{family} did not visit every training view")
            for mode in MODES:
                if family_arms[mode]["rendered_count_history"] != [counts[mode]] * args.lift_iters:
                    raise AssertionError(f"{family}/{mode} did not render the inclusive full set")
        reference_invariants = seed_result["invariants"][FAMILIES[0]]
        for family in FAMILIES[1:]:
            family_invariants = seed_result["invariants"][family]
            for key in (
                "source_view_ids_sha256",
                "source_xy_sha256",
                "source_view_ranges",
                "correct_pairs_sha256",
                "shuffled_pairs_sha256",
            ):
                if family_invariants[key] != reference_invariants[key]:
                    raise AssertionError(f"cross-family retained layout differs in {key}")
        seed_result["cross_family_layout_and_graph_identical"] = True
        reference_schedule = seed_result["families"][FAMILIES[0]]["none"]["target_view_history"]
        all_counts = set()
        for family in FAMILIES:
            for mode in MODES:
                arm = seed_result["families"][family][mode]
                if arm["target_view_history"] != reference_schedule:
                    raise AssertionError("target-view schedule differs across families")
                all_counts.add(arm["n_initial"])
        if len(all_counts) != 1:
            raise AssertionError("final primitive count differs across families")
        seed_result["cross_family_schedule_and_count_identical"] = True
        runs.append(seed_result)

    summary = summarize(runs)
    for family in FAMILIES:
        for mode in MODES:
            for metric in (
                "heldout_depth_rmse",
                "source_all_p90",
                "source_corrupted_p90",
                "test_psnr",
                "alpha_iou",
                "coverage",
                "correct_edge_p90",
                "assigned_gt_p90",
            ):
                if len(summary[family][mode][metric]["samples"]) != len(runs):
                    raise AssertionError(f"{family}/{mode}/{metric} has incomplete samples")
    source_hashes, source_tree_hash = loaded_source_hashes(root)
    return {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": git_metadata(root),
            "source_sha256": source_hashes,
            "source_tree_sha256": source_tree_hash,
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
            "command": [sys.executable, *sys.argv],
            "config": {
                key: value for key, value in vars(args).items() if not isinstance(value, Path)
            },
            "preregistration": str(PREREGISTRATION),
            "literature_scope": {
                "mac_splat_arxiv": "2607.10792",
                "edgs_arxiv": "2504.13204",
                "position_only_repo_adaptation": True,
                "oracle_topology": True,
            },
        },
        "runs": runs,
        "summary": summary,
        "decision": decision(summary),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--test-every", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--gt-gaussians", type=int, default=40)
    parser.add_argument("--fit-gaussians", type=int, default=150)
    parser.add_argument("--fit-iters", type=int, default=120)
    parser.add_argument("--lift-iters", type=int, default=90)
    parser.add_argument("--rng-check-iters", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gradient-depth-jitter", type=float, default=0.15)
    parser.add_argument("--gradient-depth-prior-lambda", type=float, default=0.001)
    parser.add_argument("--hybrid-depth-jitter", type=float, default=0.02)
    parser.add_argument("--hybrid-depth-prior-lambda", type=float, default=0.01)
    parser.add_argument("--position-lambda", type=float, default=0.25)
    parser.add_argument("--position-beta", type=float, default=0.05)
    parser.add_argument("--match-pixel-tolerance", type=float, default=1.5)
    parser.add_argument("--match-depth-tolerance", type=float, default=0.10)
    parser.add_argument("--match-min-contribution", type=float, default=0.05)
    parser.add_argument("--match-min-purity", type=float, default=0.50)
    parser.add_argument("--match-min-ray-angle-deg", type=float, default=10.0)
    parser.add_argument("--min-graph-edges", type=int, default=100)
    parser.add_argument("--min-graph-nodes", type=int, default=75)
    parser.add_argument("--min-graph-blocks", type=int, default=24)
    parser.add_argument("--min-graph-gt-ids", type=int, default=20)
    parser.add_argument("--min-representatives-per-view", type=int, default=5)
    parser.add_argument("--min-weight", type=float, default=0.05)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def is_preregistered_config(args: argparse.Namespace) -> bool:
    expected = {
        "seeds": [0, 1, 2],
        "views": 12,
        "test_every": 4,
        "image_size": 48,
        "gt_gaussians": 40,
        "fit_gaussians": 150,
        "fit_iters": 120,
        "lift_iters": 90,
        "rng_check_iters": 2,
        "lr": 0.1,
        "gradient_depth_jitter": 0.15,
        "gradient_depth_prior_lambda": 0.001,
        "hybrid_depth_jitter": 0.02,
        "hybrid_depth_prior_lambda": 0.01,
        "position_lambda": 0.25,
        "position_beta": 0.05,
        "match_pixel_tolerance": 1.5,
        "match_depth_tolerance": 0.10,
        "match_min_contribution": 0.05,
        "match_min_purity": 0.50,
        "match_min_ray_angle_deg": 10.0,
        "min_graph_edges": 100,
        "min_graph_nodes": 75,
        "min_graph_blocks": 24,
        "min_graph_gt_ids": 20,
        "min_representatives_per_view": 5,
        "min_weight": 0.05,
        "block_size": 8,
        "threads": 4,
    }
    return all(getattr(args, key) == value for key, value in expected.items())


def main() -> int:
    args = parse_args()
    if args.output is not None and not is_preregistered_config(args):
        raise ValueError("tracked output requires the exact preregistered configuration")
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"refusing to overwrite official output {args.output}")
    payload = run(args)
    rendered = json.dumps(payload, indent=2, allow_nan=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
