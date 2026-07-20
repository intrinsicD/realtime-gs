#!/usr/bin/env python3
"""Paired Hybrid ablation for cross-view plane pulling and normal alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from rtgs.core.gaussians3d import quat_to_rotmat
from rtgs.data.synthetic import make_synthetic_scene
from rtgs.image2gs.fit import FitConfig, fit_views
from rtgs.lift.base import bilinear_sample, depth_map_gradients
from rtgs.lift.depth import AlignedDepthPrior
from rtgs.lift.hybrid import HybridLifter
from rtgs.lift.surface import OrientedPointTargets
from rtgs.optim.trainer import Trainer
from rtgs.render.torch_ref import TorchRasterizer

ARMS = (
    "thick_none",
    "surfel_none",
    "surfel_plane",
    "surfel_plane_normal",
    "surfel_plane_shuffled_normal",
)
SURFEL_ARMS = ARMS[1:]
PREREGISTRATION = Path("benchmarks/results/20260715_surface_plane_normal_PREREG.md")
_NEAR = 0.05


@dataclass(frozen=True)
class CandidatePool:
    points: torch.Tensor
    view_ids: torch.Tensor
    rows: torch.Tensor
    columns: torch.Tensor
    flat_pixel_indices: torch.Tensor
    per_view_counts: list[int]


@dataclass(frozen=True)
class FrozenTargetBundle:
    indices: torch.Tensor
    points: torch.Tensor
    normals: torch.Tensor
    alignment_normals: torch.Tensor
    shuffled_alignment_normals: torch.Tensor
    shuffle_permutation: torch.Tensor
    support_indices: torch.Tensor
    support_views: torch.Tensor
    support_rows: torch.Tensor
    support_columns: torch.Tensor
    support_squared_distances: torch.Tensor
    eigenvalues: torch.Tensor
    incidence: torch.Tensor
    intersection_depths: torch.Tensor
    own_prior_depths: torch.Tensor
    source_view_ids: torch.Tensor
    source_xy: torch.Tensor
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class CleanTargetAudit:
    points: torch.Tensor
    normals: torch.Tensor
    corruption: torch.Tensor
    diagnostics: dict[str, object]


@dataclass
class PreparedSeed:
    seed: int
    scene: Any
    train_scene: Any
    gaussians2d: Any
    fit_seconds: float
    fit_history: list[dict[str, float]]
    clean_depths: list[torch.Tensor]
    corrupted_depths: list[torch.Tensor]
    corruption_masks: list[torch.Tensor]
    diagnostic_priors: list[AlignedDepthPrior]
    layout_output: Any
    layout_core: Any
    targets: FrozenTargetBundle
    clean_audit: CleanTargetAudit
    seed_result: dict[str, object]


def distribution(values: torch.Tensor) -> dict[str, float | int | None]:
    values = values.detach().double().flatten()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "max": None,
        }
    return {
        "count": int(values.numel()),
        "min": float(values.min()),
        "p10": float(torch.quantile(values, 0.1)),
        "median": float(values.median()),
        "p90": float(torch.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def loaded_source_hashes(root: Path) -> tuple[dict[str, str], str]:
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


def target_tensor_hash(
    indices: torch.Tensor,
    points: torch.Tensor,
    normals: torch.Tensor,
    alignment_normals: torch.Tensor,
) -> str:
    return tensor_collection_hash(
        [
            ("indices", indices),
            ("points", points),
            ("plane_normals", normals),
            ("alignment_normals", alignment_normals),
        ]
    )


def make_candidate_pool(
    corrupted_depths: list[torch.Tensor],
    cameras,
    center: torch.Tensor,
    extent: float,
) -> CandidatePool:
    """Enumerate the frozen AABB-valid corrupted-depth pool in canonical order."""
    lo = center - 0.5 * extent
    hi = center + 0.5 * extent
    all_points = []
    all_views = []
    all_rows = []
    all_columns = []
    all_flat = []
    counts = []
    for view_index, (depth, camera) in enumerate(zip(corrupted_depths, cameras)):
        rows, columns = torch.meshgrid(
            torch.arange(depth.shape[0]),
            torch.arange(depth.shape[1]),
            indexing="ij",
        )
        xy = torch.stack([columns + 0.5, rows + 0.5], dim=-1).to(depth)
        flat_xy = xy.reshape(-1, 2)
        flat_depth = depth.reshape(-1)
        valid = torch.isfinite(flat_depth) & (flat_depth > _NEAR)
        points = camera.unproject(flat_xy, flat_depth)
        valid &= torch.isfinite(points).all(dim=-1)
        valid &= ((points >= lo.to(points)) & (points <= hi.to(points))).all(dim=-1)
        valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        points = points[valid_indices]
        flat_rows = rows.reshape(-1)[valid_indices]
        flat_columns = columns.reshape(-1)[valid_indices]
        counts.append(int(points.shape[0]))
        all_points.append(points)
        all_views.append(torch.full((points.shape[0],), view_index, dtype=torch.long))
        all_rows.append(flat_rows.long())
        all_columns.append(flat_columns.long())
        all_flat.append(valid_indices.long())
    return CandidatePool(
        points=torch.cat(all_points),
        view_ids=torch.cat(all_views),
        rows=torch.cat(all_rows),
        columns=torch.cat(all_columns),
        flat_pixel_indices=torch.cat(all_flat),
        per_view_counts=counts,
    )


def _canonicalize_normal(normal: torch.Tensor) -> torch.Tensor:
    pivot = int(normal.abs().argmax())
    return -normal if float(normal[pivot]) < 0.0 else normal


def make_alignment_shuffle(
    normals: torch.Tensor,
    target_source_ids: torch.Tensor,
    target_xy: torch.Tensor,
    retained_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Apply the frozen per-source spatial sort and half-group cyclic roll."""
    permutation = torch.empty(normals.shape[0], dtype=torch.long)
    groups = {}
    for view_index in sorted(torch.unique(target_source_ids).tolist()):
        slots = torch.nonzero(target_source_ids == view_index, as_tuple=False).squeeze(-1)
        ordered = sorted(
            slots.tolist(),
            key=lambda slot: (
                float(target_xy[slot, 1]),
                float(target_xy[slot, 0]),
                int(retained_indices[slot]),
            ),
        )
        ordered_t = torch.tensor(ordered, dtype=torch.long)
        shift = len(ordered) // 2
        if shift == 0:
            raise AssertionError(f"source view {view_index} cannot derange one target")
        source_slots = torch.roll(ordered_t, shifts=shift)
        if not torch.equal(source_slots.sort().values, slots.sort().values):
            raise AssertionError(f"source view {view_index} shuffle changes its normal multiset")
        permutation[ordered_t] = source_slots
        groups[str(view_index)] = {
            "count": len(ordered),
            "shift": shift,
            "ordered_target_slots": ordered,
            "source_target_slots": source_slots.tolist(),
        }
    if bool((permutation == torch.arange(permutation.numel())).any()):
        raise AssertionError("shuffled alignment normals retain a fixed slot")
    shuffled = normals[permutation]
    return shuffled, permutation, groups


def build_frozen_targets(
    corrupted_depths: list[torch.Tensor],
    cameras,
    source_xy: torch.Tensor,
    source_view_ids: torch.Tensor,
    source_ranges: list[tuple[int, int]],
    center: torch.Tensor,
    extent: float,
    args: argparse.Namespace,
) -> FrozenTargetBundle:
    """Construct global four-neighbor cross-view PCA planes from corrupted train depth."""
    if len(corrupted_depths) != len(cameras) or len(source_ranges) != len(cameras):
        raise ValueError("target construction requires one depth/range per train camera")
    pool = make_candidate_pool(corrupted_depths, cameras, center, extent)
    target_indices = []
    target_points = []
    target_normals = []
    support_indices = []
    support_views = []
    support_rows = []
    support_columns = []
    support_distances = []
    eigenvalues = []
    incidences = []
    intersections = []
    prior_depths = []
    half = 0.5 * extent
    lo = center - half
    hi = center + half

    with torch.no_grad():
        for retained_index in range(source_xy.shape[0]):
            source_view = int(source_view_ids[retained_index])
            xy = source_xy[retained_index : retained_index + 1]
            camera = cameras[source_view]
            prior_depth = bilinear_sample(corrupted_depths[source_view].to(xy), xy)[0]
            origin, direction = camera.pixel_rays(xy)
            safe_direction = torch.where(
                direction.abs() < 1e-9,
                torch.full_like(direction, 1e-9),
                direction,
            )
            ta = (lo.to(xy)[None] - origin[None]) / safe_direction
            tb = (hi.to(xy)[None] - origin[None]) / safe_direction
            near = torch.minimum(ta, tb).amax(dim=-1)[0].clamp_min(_NEAR)
            far = torch.maximum(ta, tb).amin(dim=-1)[0]
            if not bool(torch.isfinite(prior_depth)) or not bool(
                (prior_depth > near) & (prior_depth < far)
            ):
                continue
            query = camera.unproject(xy, prior_depth[None])[0]
            eligible = pool.view_ids != source_view
            eligible_indices = torch.nonzero(eligible, as_tuple=False).squeeze(-1)
            if eligible_indices.numel() < args.target_neighbors:
                continue
            squared = (pool.points[eligible_indices] - query).square().sum(dim=-1)
            order = torch.argsort(squared, stable=True)
            chosen = eligible_indices[order[: args.target_neighbors]]
            chosen_squared = squared[order[: args.target_neighbors]]
            chosen_views = pool.view_ids[chosen]
            if torch.unique(chosen_views).numel() < args.target_min_support_views:
                continue
            if float(chosen_squared[-1].sqrt()) > args.target_max_radius_frac * extent:
                continue
            neighbors = pool.points[chosen]
            point = neighbors.mean(dim=0)
            delta = neighbors - point
            covariance = delta.T @ delta / args.target_neighbors
            values, vectors = torch.linalg.eigh(covariance)
            if not bool(torch.isfinite(values).all()) or float(values.sum()) <= 0:
                continue
            if float(values[0] / values.sum()) > args.target_planarity_ratio:
                continue
            normal = _canonicalize_normal(vectors[:, 0])
            unit_ray = torch.nn.functional.normalize(direction[0], dim=0)
            incidence = torch.dot(normal, unit_ray).abs()
            if float(incidence) < args.target_min_incidence:
                continue
            denominator = torch.dot(normal, direction[0])
            intersection = torch.dot(normal, point - origin) / denominator
            if not bool(torch.isfinite(intersection)) or not bool(
                (intersection > near) & (intersection < far)
            ):
                continue
            target_indices.append(retained_index)
            target_points.append(point)
            target_normals.append(normal)
            support_indices.append(chosen)
            support_views.append(chosen_views)
            support_rows.append(pool.rows[chosen])
            support_columns.append(pool.columns[chosen])
            support_distances.append(chosen_squared)
            eigenvalues.append(values)
            incidences.append(incidence)
            intersections.append(intersection)
            prior_depths.append(prior_depth)

    if not target_indices:
        raise AssertionError("cross-view target builder produced no valid target")
    indices_t = torch.tensor(target_indices, dtype=torch.long)
    points_t = torch.stack(target_points).detach()
    normals_t = torch.stack(target_normals).detach()
    support_indices_t = torch.stack(support_indices).long().detach()
    support_views_t = torch.stack(support_views).long().detach()
    support_rows_t = torch.stack(support_rows).long().detach()
    support_columns_t = torch.stack(support_columns).long().detach()
    support_squared_t = torch.stack(support_distances).detach()
    eigenvalues_t = torch.stack(eigenvalues).detach()
    incidence_t = torch.stack(incidences).detach()
    intersection_t = torch.stack(intersections).detach()
    prior_depths_t = torch.stack(prior_depths).detach()
    target_source_ids = source_view_ids[indices_t]
    target_xy = source_xy[indices_t]
    shuffled, permutation, shuffle_groups = make_alignment_shuffle(
        normals_t, target_source_ids, target_xy, indices_t
    )
    separation = 1.0 - (normals_t * shuffled).sum(dim=-1).abs().clamp_max(1.0)
    selected_support_views = sorted(torch.unique(support_views_t).tolist())
    candidate_pool_views = sorted(torch.unique(pool.view_ids).tolist())
    query_views = sorted(torch.unique(target_source_ids).tolist())
    per_source = {str(view): int((target_source_ids == view).sum()) for view in range(len(cameras))}
    diagnostics: dict[str, object] = {
        "target_count": int(indices_t.numel()),
        "retained_count": int(source_xy.shape[0]),
        "coverage": float(indices_t.numel() / source_xy.shape[0]),
        "query_views": query_views,
        "candidate_pool_views": candidate_pool_views,
        "selected_support_views": selected_support_views,
        "targets_per_query_source": per_source,
        "candidate_count": int(pool.points.shape[0]),
        "candidate_counts_per_view": pool.per_view_counts,
        "farthest_neighbor_over_extent": distribution(support_squared_t[:, -1].sqrt() / extent),
        "eigenvalue_min_over_sum": distribution(eigenvalues_t[:, 0] / eigenvalues_t.sum(dim=-1)),
        "eigenvalue_mid_over_sum": distribution(eigenvalues_t[:, 1] / eigenvalues_t.sum(dim=-1)),
        "incidence": distribution(incidence_t),
        "intersection_depth": distribution(intersection_t),
        "own_prior_depth": distribution(prior_depths_t),
        "shuffle_separation": distribution(separation),
        "shuffle_groups": shuffle_groups,
        "shuffle_no_fixed_slots": True,
        "shuffle_per_source_normal_multiset_exact": True,
        "target_indices_sha256": tensor_collection_hash([("target_indices", indices_t)]),
        "target_points_sha256": tensor_collection_hash([("target_points", points_t)]),
        "target_normals_sha256": tensor_collection_hash([("target_normals", normals_t)]),
        "shuffled_normals_sha256": tensor_collection_hash([("alignment_normals", shuffled)]),
        "correct_targets_sha256": target_tensor_hash(indices_t, points_t, normals_t, normals_t),
        "shuffled_targets_sha256": target_tensor_hash(indices_t, points_t, normals_t, shuffled),
        "shuffle_permutation_sha256": tensor_collection_hash(
            [("shuffle_permutation", permutation)]
        ),
        "support_metadata_sha256": tensor_collection_hash(
            [
                ("support_indices", support_indices_t),
                ("support_views", support_views_t),
                ("support_rows", support_rows_t),
                ("support_columns", support_columns_t),
                ("support_squared_distances", support_squared_t),
                ("eigenvalues", eigenvalues_t),
                ("incidence", incidence_t),
                ("intersection_depths", intersection_t),
            ]
        ),
        "candidate_metadata_sha256": tensor_collection_hash(
            [
                ("candidate_points", pool.points),
                ("candidate_view_ids", pool.view_ids),
                ("candidate_rows", pool.rows),
                ("candidate_columns", pool.columns),
                ("candidate_flat_pixel_indices", pool.flat_pixel_indices),
            ]
        ),
        "raw": {
            "indices": indices_t.tolist(),
            "points": points_t.tolist(),
            "normals": normals_t.tolist(),
            "shuffled_alignment_normals": shuffled.tolist(),
            "shuffle_permutation": permutation.tolist(),
            "support_indices": support_indices_t.tolist(),
            "support_views": support_views_t.tolist(),
            "support_rows": support_rows_t.tolist(),
            "support_columns": support_columns_t.tolist(),
            "support_squared_distances": support_squared_t.tolist(),
            "eigenvalues": eigenvalues_t.tolist(),
            "incidence": incidence_t.tolist(),
            "intersection_depths": intersection_t.tolist(),
            "own_prior_depths": prior_depths_t.tolist(),
        },
    }
    if diagnostics["correct_targets_sha256"] == diagnostics["shuffled_targets_sha256"]:
        raise AssertionError("correct and shuffled target hashes are equal")
    return FrozenTargetBundle(
        indices=indices_t,
        points=points_t,
        normals=normals_t,
        alignment_normals=normals_t,
        shuffled_alignment_normals=shuffled,
        shuffle_permutation=permutation,
        support_indices=support_indices_t,
        support_views=support_views_t,
        support_rows=support_rows_t,
        support_columns=support_columns_t,
        support_squared_distances=support_squared_t,
        eigenvalues=eigenvalues_t,
        incidence=incidence_t,
        intersection_depths=intersection_t,
        own_prior_depths=prior_depths_t,
        source_view_ids=target_source_ids,
        source_xy=target_xy,
        diagnostics=diagnostics,
    )


def oriented_targets(bundle: FrozenTargetBundle, *, shuffled: bool) -> OrientedPointTargets:
    return OrientedPointTargets(
        indices=bundle.indices,
        points=bundle.points,
        plane_normals=bundle.normals,
        alignment_normals=(
            bundle.shuffled_alignment_normals if shuffled else bundle.alignment_normals
        ),
    )


def corruption_at_layout(
    corruption_masks: list[torch.Tensor],
    source_xy: torch.Tensor,
    source_view_ids: torch.Tensor,
    source_ranges: list[tuple[int, int]],
) -> torch.Tensor:
    values = torch.zeros(source_xy.shape[0], dtype=torch.bool)
    for view_index, (start, end) in enumerate(source_ranges):
        sampled = bilinear_sample(corruption_masks[view_index].to(source_xy), source_xy[start:end])
        if not bool((source_view_ids[start:end] == view_index).all()):
            raise AssertionError("retained source layout is inconsistent")
        values[start:end] = sampled > 0.5
    return values


def apply_structural_floors(
    bundle: FrozenTargetBundle,
    layout_output,
    source_xy: torch.Tensor,
    source_view_ids: torch.Tensor,
    source_ranges: list[tuple[int, int]],
    corruption_masks: list[torch.Tensor],
    n_views: int,
    args: argparse.Namespace,
) -> None:
    """Bind the train-input-only validity floors before any clean target audit."""
    corrupted_nodes = corruption_at_layout(
        corruption_masks, source_xy, source_view_ids, source_ranges
    )
    target_corrupted = corrupted_nodes[bundle.indices]
    scales = layout_output.scales[bundle.indices]
    sorted_scales = scales.sort(dim=-1).values
    scale_ratio = sorted_scales[:, 1] / sorted_scales[:, 0].clamp_min(1e-12)
    axis_indices = layout_output.log_scales[bundle.indices].argmin(dim=-1)
    eigen_mid_ratio = bundle.eigenvalues[:, 1] / bundle.eigenvalues.sum(dim=-1)
    separation = 1.0 - (bundle.normals * bundle.shuffled_alignment_normals).sum(
        dim=-1
    ).abs().clamp_max(1.0)
    expected_views = list(range(n_views))
    floor_checks = {
        "target_count": bundle.indices.numel() >= args.min_targets,
        "coverage": bundle.indices.numel() / source_xy.shape[0] >= args.min_target_coverage,
        "corrupted_target_count": int(target_corrupted.sum()) >= args.min_corrupted_targets,
        "all_query_views": bundle.diagnostics["query_views"] == expected_views,
        "all_candidate_pool_views": bundle.diagnostics["candidate_pool_views"] == expected_views,
        "all_selected_support_views": bundle.diagnostics["selected_support_views"]
        == expected_views,
        "targets_per_query_source": min(bundle.diagnostics["targets_per_query_source"].values())
        >= args.min_targets_per_source,
        "farthest_neighbor_p90": float(bundle.diagnostics["farthest_neighbor_over_extent"]["p90"])
        <= args.max_farthest_neighbor_p90_frac,
        "incidence_p10": float(torch.quantile(bundle.incidence, 0.1)) >= args.min_incidence_p10,
        "eigenvalue_mid_ratio_min": float(eigen_mid_ratio.min()) >= args.min_eigenvalue_mid_ratio,
        "scale_ratio_p10": float(torch.quantile(scale_ratio, 0.1)) >= args.min_scale_ratio_p10,
        "shuffle_separation_median": float(separation.median())
        >= args.min_shuffle_separation_median,
    }
    bundle.diagnostics.update(
        {
            "corrupted_retained_node_count": int(corrupted_nodes.sum()),
            "corrupted_target_count": int(target_corrupted.sum()),
            "corrupted_target_fraction_of_corrupted_nodes": float(
                target_corrupted.sum() / corrupted_nodes.sum().clamp_min(1)
            ),
            "second_over_shortest_scale_ratio": distribution(scale_ratio),
            "oriented_axis_indices": axis_indices.tolist(),
            "oriented_axis_indices_sha256": tensor_collection_hash(
                [("axis_indices", axis_indices)]
            ),
            "target_corruption_sha256": tensor_collection_hash(
                [("target_corruption", target_corrupted)]
            ),
            "floor_checks": floor_checks,
            "structural_floor_pass": all(floor_checks.values()),
        }
    )
    if not all(floor_checks.values()):
        failed = [name for name, passed in floor_checks.items() if not passed]
        raise AssertionError(f"oriented target construction misses frozen floors: {failed}")


def clean_surface_labels(
    bundle: FrozenTargetBundle,
    clean_depths: list[torch.Tensor],
    cameras,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct post-hash clean points/normals using the existing Jacobian convention."""
    points = torch.zeros_like(bundle.points)
    normals = torch.zeros_like(bundle.normals)
    labelable = torch.ones(bundle.indices.numel(), dtype=torch.bool)
    for view_index, (depth, camera) in enumerate(zip(clean_depths, cameras)):
        slots = torch.nonzero(bundle.source_view_ids == view_index, as_tuple=False).squeeze(-1)
        if slots.numel() == 0:
            continue
        xy = bundle.source_xy[slots]
        z = bilinear_sample(depth.to(xy), xy)
        gx, gy = depth_map_gradients(depth.to(xy), validity_aware=True)
        du = bilinear_sample(gx, xy)
        dv = bilinear_sample(gy, xy)
        q = torch.stack(
            [
                (xy[:, 0] - camera.cx) / camera.fx,
                (xy[:, 1] - camera.cy) / camera.fy,
                torch.ones_like(z),
            ],
            dim=-1,
        )
        ex = torch.zeros_like(q)
        ey = torch.zeros_like(q)
        ex[:, 0] = z / camera.fx
        ey[:, 1] = z / camera.fy
        j_u = ex + q * du[:, None]
        j_v = ey + q * dv[:, None]
        cross = torch.linalg.cross(j_u, j_v)
        normal_cam = torch.nn.functional.normalize(cross, dim=-1)
        normal_world = normal_cam @ camera.R.to(xy)
        point = camera.unproject(xy, z)
        valid = (
            torch.isfinite(z)
            & torch.isfinite(du)
            & torch.isfinite(dv)
            & (z > _NEAR)
            & torch.isfinite(point).all(dim=-1)
            & torch.isfinite(normal_world).all(dim=-1)
            & (cross.norm(dim=-1) > 1e-8)
        )
        points[slots] = point
        normals[slots] = normal_world
        labelable[slots] = valid
    return points, normals, labelable


def audit_targets_with_clean_depth(
    bundle: FrozenTargetBundle,
    clean_depths: list[torch.Tensor],
    cameras,
    corruption_masks: list[torch.Tensor],
    source_xy: torch.Tensor,
    source_view_ids: torch.Tensor,
    source_ranges: list[tuple[int, int]],
    extent: float,
    args: argparse.Namespace,
) -> CleanTargetAudit:
    """Apply the frozen clean target audit only after target/control hashes exist."""
    clean_points, clean_normals, labelable = clean_surface_labels(bundle, clean_depths, cameras)
    corruption = corruption_at_layout(corruption_masks, source_xy, source_view_ids, source_ranges)[
        bundle.indices
    ]
    plane_residual = ((clean_points - bundle.points) * bundle.normals).sum(dim=-1).abs() / extent
    cosine = (bundle.normals * clean_normals).sum(dim=-1).abs().clamp(0.0, 1.0)
    subsets = {
        "all": torch.ones(bundle.indices.numel(), dtype=torch.bool),
        "corrupted": corruption,
    }
    subset_diagnostics = {}
    audit_checks = {"every_target_labelable": bool(labelable.all())}
    for name, mask in subsets.items():
        evaluated = mask & labelable
        subset_diagnostics[name] = {
            "count": int(mask.sum()),
            "labelable_count": int(evaluated.sum()),
            "plane_residual_over_extent": distribution(plane_residual[evaluated]),
            "normal_cosine": distribution(cosine[evaluated]),
        }
        audit_checks[f"{name}_plane_p90"] = (
            bool(mask.any())
            and bool(labelable[mask].all())
            and float(torch.quantile(plane_residual[evaluated], 0.9)) <= args.max_clean_plane_p90
        )
        audit_checks[f"{name}_normal_cosine_median"] = (
            bool(mask.any())
            and bool(labelable[mask].all())
            and float(cosine[evaluated].median()) >= args.min_clean_normal_cosine
        )
    diagnostics: dict[str, object] = {
        "timing": "after_correct_and_shuffled_target_hashes_frozen",
        "subsets": subset_diagnostics,
        "checks": audit_checks,
        "pass": all(audit_checks.values()),
        "clean_points_sha256": tensor_collection_hash([("clean_points", clean_points)]),
        "clean_normals_sha256": tensor_collection_hash([("clean_normals", clean_normals)]),
        "labelable_sha256": tensor_collection_hash([("labelable", labelable)]),
        "clean_data_changed_targets": False,
    }
    return CleanTargetAudit(
        points=clean_points,
        normals=clean_normals,
        corruption=corruption,
        diagnostics=diagnostics,
    )


def arm_settings(arm: str, args: argparse.Namespace) -> tuple[float, float, float]:
    if arm == "thick_none":
        return args.thick_ray_thickness, 0.0, 0.0
    if arm == "surfel_none":
        return args.surfel_ray_thickness, 0.0, 0.0
    if arm == "surfel_plane":
        return args.surfel_ray_thickness, args.plane_lambda, 0.0
    if arm in ("surfel_plane_normal", "surfel_plane_shuffled_normal"):
        return args.surfel_ray_thickness, args.plane_lambda, args.normal_lambda
    raise ValueError(f"unknown surface arm {arm!r}")


def make_lifter(
    arm: str,
    args: argparse.Namespace,
    seed: int,
    corrupted_depths: list[torch.Tensor],
    *,
    iterations: int | None = None,
) -> HybridLifter:
    thickness, plane_lambda, normal_lambda = arm_settings(arm, args)
    return HybridLifter(
        backend=ListDepthBackend(corrupted_depths),
        iterations=args.lift_iters if iterations is None else iterations,
        lr=args.lr,
        lr_rotation=args.lr_rotation,
        rasterizer="torch",
        min_weight=args.min_weight,
        depth_jitter=args.depth_jitter,
        depth_prior_lambda=args.depth_prior_lambda,
        depth_anchor_mode="legacy",
        photometric_supervision_mode="all",
        optimize_rotation=True,
        optimize_scale=False,
        ray_thickness=thickness,
        merge=False,
        merge_color_bin_size=None,
        seed=seed,
        plane_consistency_lambda=plane_lambda,
        normal_consistency_lambda=normal_lambda,
    )


def targets_for_arm(arm: str, bundle: FrozenTargetBundle) -> OrientedPointTargets | None:
    if arm == "thick_none":
        return None
    return oriented_targets(bundle, shuffled=arm == "surfel_plane_shuffled_normal")


def execute_lift(
    lifter: HybridLifter,
    arm: str,
    gaussians2d,
    train_scene,
    bundle: FrozenTargetBundle,
):
    targets = targets_for_arm(arm, bundle)
    if targets is None:
        return lifter.lift(gaussians2d, train_scene)
    return lifter.lift_with_oriented_points(gaussians2d, train_scene, targets)


def stored_target_hash(targets: OrientedPointTargets | None) -> str | None:
    if targets is None:
        return None
    alignment = (
        targets.plane_normals if targets.alignment_normals is None else targets.alignment_normals
    )
    return target_tensor_hash(targets.indices, targets.points, targets.plane_normals, alignment)


def assert_layout_equal(reference, other, context: str) -> None:
    if not torch.equal(reference.source_xy_before_merge, other.source_xy_before_merge):
        raise AssertionError(f"{context} changes source pixels")
    if not torch.equal(reference.source_view_ids_before_merge, other.source_view_ids_before_merge):
        raise AssertionError(f"{context} changes source IDs")
    if reference.source_view_ranges_before_merge != other.source_view_ranges_before_merge:
        raise AssertionError(f"{context} changes source ranges")


def target_residual_diagnostics(
    gaussians,
    bundle: FrozenTargetBundle,
    axis_indices: torch.Tensor,
    extent: float,
) -> dict[str, object]:
    offsets = gaussians.means[bundle.indices] - bundle.points
    plane = (offsets * bundle.normals).sum(dim=-1).abs()
    rotations = quat_to_rotmat(gaussians.quats[bundle.indices])
    axes = rotations.gather(2, axis_indices[:, None, None].expand(-1, 3, 1)).squeeze(-1)
    correct_normal = 1.0 - (axes * bundle.normals).sum(dim=-1).abs().clamp_max(1.0)
    shuffled_normal = 1.0 - (axes * bundle.shuffled_alignment_normals).sum(dim=-1).abs().clamp_max(
        1.0
    )
    return {
        "plane_distance_over_extent": distribution(plane / extent),
        "correct_normal_loss": distribution(correct_normal),
        "shuffled_normal_loss": distribution(shuffled_normal),
        "selected_axis": axes,
    }


def quaternion_change_diagnostics(initial, final) -> dict[str, float | int | None]:
    cosine = (initial.quats * final.quats).sum(dim=-1).abs().clamp(0.0, 1.0)
    angle = torch.rad2deg(2.0 * torch.acos(cosine))
    return distribution(angle)


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


def local_clean_diagnostics(
    gaussians,
    bundle: FrozenTargetBundle,
    audit: CleanTargetAudit,
    cameras,
    axis_indices: torch.Tensor,
) -> dict[str, object]:
    predicted_depth = torch.zeros(bundle.indices.numel())
    clean_depth = torch.zeros(bundle.indices.numel())
    for view_index, camera in enumerate(cameras):
        slots = torch.nonzero(bundle.source_view_ids == view_index, as_tuple=False).squeeze(-1)
        if slots.numel() == 0:
            continue
        predicted_depth[slots] = camera.project(gaussians.means[bundle.indices[slots]])[1]
        clean_depth[slots] = camera.project(audit.points[slots])[1]
    relative = (predicted_depth - clean_depth).abs() / clean_depth.clamp_min(_NEAR)
    rotations = quat_to_rotmat(gaussians.quats[bundle.indices])
    axes = rotations.gather(2, axis_indices[:, None, None].expand(-1, 3, 1)).squeeze(-1)
    clean_normal_loss = 1.0 - (axes * audit.normals).sum(dim=-1).abs().clamp_max(1.0)
    all_mask = torch.ones_like(audit.corruption)
    return {
        "all_targeted_source_depth_relative": distribution(relative[all_mask]),
        "corrupted_targeted_source_depth_relative": distribution(relative[audit.corruption]),
        "uncorrupted_targeted_source_depth_relative": distribution(relative[~audit.corruption]),
        "selected_axis_clean_normal_loss": distribution(clean_normal_loss),
    }


def finite_result(result, core, context: str) -> None:
    for field in ("means", "quats", "log_scales", "opacity", "sh"):
        if not bool(torch.isfinite(getattr(result, field)).all()):
            raise AssertionError(f"{context} has non-finite {field}")
    for name, values in (
        ("total", core.history),
        ("anchor", core.anchor_history),
        ("plane", core.plane_history),
        ("normal", core.normal_history),
    ):
        if any(not math.isfinite(value) for value in values):
            raise AssertionError(f"{context} has non-finite {name} history")
    for name, values in (
        ("initial", core.initial_ray_fractions_before_merge),
        ("final", core.final_ray_fractions_before_merge),
    ):
        if values is None or not bool(torch.isfinite(values).all()):
            raise AssertionError(f"{context} has invalid {name} ray fractions")
        if not bool(((values > 0) & (values < 1)).all()):
            raise AssertionError(f"{context} leaves bounded ray fractions")


def assert_initialization_invariants(
    gaussians2d,
    train_scene,
    corrupted_depths,
    bundle: FrozenTargetBundle,
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, object], dict[str, Any]]:
    _, extent = train_scene.center_and_extent()
    outputs = {}
    cores = {}
    for arm in ARMS:
        lifter = make_lifter(arm, args, seed, corrupted_depths, iterations=0)
        outputs[arm] = execute_lift(lifter, arm, gaussians2d, train_scene, bundle)
        cores[arm] = lifter.gradient
    reference = outputs["surfel_none"]
    reference_core = cores["surfel_none"]
    for arm in SURFEL_ARMS[1:]:
        assert_equal_fields(reference, outputs[arm], f"{arm} step 0")
        assert_layout_equal(reference_core, cores[arm], f"{arm} step 0")
    for field in ("means", "opacity", "sh"):
        if not torch.equal(getattr(outputs["thick_none"], field), getattr(reference, field)):
            raise AssertionError(f"thick reference unexpectedly changes step-zero {field}")
    assert_layout_equal(reference_core, cores["thick_none"], "thick reference")

    correct_hashes = set()
    axis_hashes = set()
    for arm in SURFEL_ARMS:
        stored = cores[arm].oriented_targets_before_merge
        if stored is None:
            raise AssertionError(f"{arm} did not store oriented targets")
        stored_hash = stored_target_hash(stored)
        # Validation may renormalize unit eigenvectors by a final floating-point division; bind
        # arm equality separately from the pre-validation target hash.
        if arm != "surfel_plane_shuffled_normal":
            correct_hashes.add(stored_hash)
        axis = cores[arm].oriented_axis_indices_before_merge
        if axis is None:
            raise AssertionError(f"{arm} did not freeze shortest-axis indices")
        axis_hashes.add(tensor_collection_hash([("axis_indices", axis)]))
    if len(correct_hashes) != 1 or len(axis_hashes) != 1:
        raise AssertionError("surfel arms changed correct targets or axis indices")
    correct_stored = cores["surfel_none"].oriented_targets_before_merge
    shuffled_stored = cores["surfel_plane_shuffled_normal"].oriented_targets_before_merge
    for field in ("indices", "points", "plane_normals"):
        if not torch.equal(getattr(correct_stored, field), getattr(shuffled_stored, field)):
            raise AssertionError(f"shuffled-normal control changes stored {field}")
    if not torch.equal(
        shuffled_stored.alignment_normals,
        correct_stored.alignment_normals[bundle.shuffle_permutation],
    ):
        raise AssertionError("stored shuffled normals disagree with frozen permutation")

    with_targets = make_lifter(
        "surfel_none", args, seed, corrupted_depths, iterations=args.rng_check_iters
    )
    with_output = execute_lift(with_targets, "surfel_none", gaussians2d, train_scene, bundle)
    without_targets = make_lifter(
        "surfel_none", args, seed, corrupted_depths, iterations=args.rng_check_iters
    )
    without_output = without_targets.lift(gaussians2d, train_scene)
    assert_equal_fields(without_output, with_output, "zero-coefficient target check")
    without_core = without_targets.gradient
    with_core = with_targets.gradient
    for name in ("history", "anchor_history", "target_view_history", "rendered_count_history"):
        if getattr(without_core, name) != getattr(with_core, name):
            raise AssertionError(f"zero-coefficient targets change {name}")
    for name in ("initial_ray_fractions_before_merge", "final_ray_fractions_before_merge"):
        if not torch.equal(getattr(without_core, name), getattr(with_core, name)):
            raise AssertionError(f"zero-coefficient targets change {name}")

    initial_target_metrics = {}
    for arm in SURFEL_ARMS:
        initial_target_metrics[arm] = target_residual_diagnostics(
            outputs[arm],
            bundle,
            cores[arm].oriented_axis_indices_before_merge,
            extent,
        )
        initial_target_metrics[arm].pop("selected_axis")
    invariants = {
        "four_surfel_arms_step0_identical": True,
        "thick_differs_only_in_quaternions_and_scales": True,
        "zero_coefficient_targets_preserve_behavior": True,
        "surfel_optimizer_groups_identical": True,
        "optimizer_groups": [
            {"parameter": "bounded_ray_depth", "lr": args.lr},
            {"parameter": "quaternion", "lr": args.lr_rotation},
        ],
        "primitive_count": reference.n,
        "source_xy_sha256": tensor_collection_hash(
            [("source_xy", reference_core.source_xy_before_merge)]
        ),
        "source_view_ids_sha256": tensor_collection_hash(
            [("source_view_ids", reference_core.source_view_ids_before_merge)]
        ),
        "source_view_ranges": reference_core.source_view_ranges_before_merge,
        "correct_stored_targets_sha256": next(iter(correct_hashes)),
        "axis_indices_sha256": next(iter(axis_hashes)),
        "initial_target_metrics": initial_target_metrics,
        "thick_scales_sha256": tensor_collection_hash(
            [("log_scales", outputs["thick_none"].log_scales)]
        ),
        "surfel_scales_sha256": tensor_collection_hash([("log_scales", reference.log_scales)]),
    }
    return invariants, outputs


def evaluate_arm(
    arm: str,
    gaussians2d,
    scene,
    train_scene,
    clean_depths,
    corrupted_depths,
    diagnostic_priors,
    corruption_masks,
    bundle: FrozenTargetBundle,
    audit: CleanTargetAudit,
    initial_output,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    lifter = make_lifter(arm, args, seed, corrupted_depths)
    started = time.perf_counter()
    result = execute_lift(lifter, arm, gaussians2d, train_scene, bundle)
    lift_seconds = time.perf_counter() - started
    core = lifter.gradient
    finite_result(result, core, arm)
    _, extent = train_scene.center_and_extent()
    if arm in SURFEL_ARMS:
        if (
            len(core.plane_history) != args.lift_iters
            or len(core.normal_history) != args.lift_iters
        ):
            raise AssertionError(f"{arm} has incomplete surface histories")
        if core.oriented_axis_indices_before_merge is None:
            raise AssertionError(f"{arm} did not retain axis indices")
        axis_indices = core.oriented_axis_indices_before_merge
        target_diagnostics = target_residual_diagnostics(result, bundle, axis_indices, extent)
        target_diagnostics.pop("selected_axis")
        clean_local = local_clean_diagnostics(
            result, bundle, audit, train_scene.cameras, axis_indices
        )
        stored_hash = stored_target_hash(core.oriented_targets_before_merge)
        stored_axis_hash = tensor_collection_hash([("axis_indices", axis_indices)])
    else:
        if core.plane_history or core.normal_history:
            raise AssertionError("thick reference unexpectedly computed surface histories")
        axis_indices = initial_output.log_scales[bundle.indices].argmin(dim=-1)
        target_diagnostics = target_residual_diagnostics(result, bundle, axis_indices, extent)
        target_diagnostics.pop("selected_axis")
        clean_local = local_clean_diagnostics(
            result, bundle, audit, train_scene.cameras, axis_indices
        )
        stored_hash = None
        stored_axis_hash = None
    renderer = TorchRasterizer()
    return {
        "n_initial": result.n,
        "lift_seconds": lift_seconds,
        "loss_history": list(core.history),
        "loss_history_checkpoints": history_checkpoints(core.history),
        "anchor_history": list(core.anchor_history),
        "anchor_history_checkpoints": history_checkpoints(core.anchor_history),
        "plane_history": list(core.plane_history),
        "plane_history_checkpoints": history_checkpoints(core.plane_history),
        "normal_history": list(core.normal_history),
        "normal_history_checkpoints": history_checkpoints(core.normal_history),
        "target_view_history": core.target_view_history,
        "target_view_history_sha256": integer_sequence_hash(core.target_view_history),
        "rendered_count_history": core.rendered_count_history,
        "stored_targets_sha256": stored_hash,
        "stored_axis_indices_sha256": stored_axis_hash,
        "initial_ray_fractions": ray_fraction_diagnostics(core.initial_ray_fractions_before_merge),
        "final_ray_fractions": ray_fraction_diagnostics(core.final_ray_fractions_before_merge),
        "target_geometry": target_diagnostics,
        "clean_local": clean_local,
        "quaternion_change_deg": quaternion_change_diagnostics(initial_output, result),
        "scales": {
            "shortest": distribution(result.scales.sort(dim=-1).values[:, 0]),
            "second_over_shortest": distribution(
                result.scales.sort(dim=-1).values[:, 1]
                / result.scales.sort(dim=-1).values[:, 0].clamp_min(1e-12)
            ),
            "largest_over_shortest": distribution(
                result.scales.sort(dim=-1).values[:, 2]
                / result.scales.sort(dim=-1).values[:, 0].clamp_min(1e-12)
            ),
        },
        "initial_test": Trainer.evaluate_metrics(
            scene, result, renderer, indices=scene.testing_views
        ),
        "training": Trainer.evaluate_metrics(
            train_scene, result, renderer, indices=train_scene.training_views
        ),
        "cross_only_training_l1": cross_only_training_l1(
            train_scene, result, core.source_view_ids_before_merge
        ),
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
    "plane_p90": ("target_geometry", "plane_distance_over_extent", "p90"),
    "correct_normal_p90": ("target_geometry", "correct_normal_loss", "p90"),
    "shuffled_normal_p90": ("target_geometry", "shuffled_normal_loss", "p90"),
    "clean_target_all_p90": (
        "clean_local",
        "all_targeted_source_depth_relative",
        "p90",
    ),
    "clean_target_corrupted_p90": (
        "clean_local",
        "corrupted_targeted_source_depth_relative",
        "p90",
    ),
    "clean_local_normal_p90": (
        "clean_local",
        "selected_axis_clean_normal_loss",
        "p90",
    ),
    "lift_seconds": ("lift_seconds",),
}


def metric_samples(runs, arm: str, path: tuple[str, ...]) -> list[float]:
    samples = []
    for run in runs:
        value = run["arms"][arm]
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
        arm: {
            name: aggregate(metric_samples(runs, arm, path)) for name, path in SUMMARY_PATHS.items()
        }
        for arm in ARMS
    }


def metric_mean(summary, arm: str, metric: str) -> float:
    value = summary[arm][metric]["mean"]
    if value is None:
        raise ValueError(f"{arm}/{metric} has no finite samples")
    return float(value)


def lower_wins(summary, left: str, right: str, metric: str) -> int:
    return sum(
        left_value < right_value
        for left_value, right_value in zip(
            summary[left][metric]["samples"], summary[right][metric]["samples"]
        )
    )


def metric_gain(summary, arm: str, baseline: str, metric: str) -> float:
    baseline_value = metric_mean(summary, baseline, metric)
    value = metric_mean(summary, arm, metric)
    return (baseline_value - value) / max(abs(baseline_value), 1e-12)


def global_predicate(summary, arm: str, baseline: str) -> dict[str, object]:
    thresholds = {
        "heldout_depth_rmse": 0.02,
        "source_all_p90": 0.10,
        "source_corrupted_p90": 0.15,
    }
    gains = {metric: metric_gain(summary, arm, baseline, metric) for metric in thresholds}
    wins = {metric: lower_wins(summary, arm, baseline, metric) for metric in thresholds}
    psnr_delta = metric_mean(summary, arm, "test_psnr") - metric_mean(
        summary, baseline, "test_psnr"
    )
    coverage_delta = metric_mean(summary, arm, "coverage") - metric_mean(
        summary, baseline, "coverage"
    )
    iou_delta = metric_mean(summary, arm, "alpha_iou") - metric_mean(summary, baseline, "alpha_iou")
    passed = (
        all(gains[name] >= threshold for name, threshold in thresholds.items())
        and all(wins[name] >= 2 for name in thresholds)
        and psnr_delta >= -0.10
        and coverage_delta >= -0.02
        and iou_delta >= -0.02
    )
    return {
        "gain_fraction": gains,
        "seed_wins": wins,
        "test_psnr_delta_db": psnr_delta,
        "coverage_delta": coverage_delta,
        "alpha_iou_delta": iou_delta,
        "pass": passed,
    }


def decision(summary) -> dict[str, object]:
    plane_engagement_gain = metric_gain(summary, "surfel_plane", "surfel_none", "plane_p90")
    plane_engagement_wins = lower_wins(summary, "surfel_plane", "surfel_none", "plane_p90")
    plane_local_all_gain = metric_gain(
        summary, "surfel_plane", "surfel_none", "clean_target_all_p90"
    )
    plane_local_corrupt_gain = metric_gain(
        summary,
        "surfel_plane",
        "surfel_none",
        "clean_target_corrupted_p90",
    )
    plane_local_all_wins = lower_wins(
        summary, "surfel_plane", "surfel_none", "clean_target_all_p90"
    )
    plane_local_corrupt_wins = lower_wins(
        summary,
        "surfel_plane",
        "surfel_none",
        "clean_target_corrupted_p90",
    )
    plane_global = {
        baseline: global_predicate(summary, "surfel_plane", baseline)
        for baseline in ("surfel_none", "thick_none")
    }
    plane_pass = (
        plane_engagement_gain >= 0.25
        and plane_engagement_wins == 3
        and plane_local_all_gain >= 0.15
        and plane_local_corrupt_gain >= 0.20
        and plane_local_all_wins >= 2
        and plane_local_corrupt_wins >= 2
        and all(item["pass"] for item in plane_global.values())
    )

    combined_plane_gain = metric_gain(summary, "surfel_plane_normal", "surfel_none", "plane_p90")
    combined_plane_wins = lower_wins(summary, "surfel_plane_normal", "surfel_none", "plane_p90")
    combined_normal_gain = metric_gain(
        summary, "surfel_plane_normal", "surfel_plane", "correct_normal_p90"
    )
    combined_normal_wins = lower_wins(
        summary, "surfel_plane_normal", "surfel_plane", "correct_normal_p90"
    )
    combined_local_all_gain = metric_gain(
        summary, "surfel_plane_normal", "surfel_none", "clean_target_all_p90"
    )
    combined_local_corrupt_gain = metric_gain(
        summary,
        "surfel_plane_normal",
        "surfel_none",
        "clean_target_corrupted_p90",
    )
    combined_local_all_wins = lower_wins(
        summary, "surfel_plane_normal", "surfel_none", "clean_target_all_p90"
    )
    combined_local_corrupt_wins = lower_wins(
        summary,
        "surfel_plane_normal",
        "surfel_none",
        "clean_target_corrupted_p90",
    )
    combined_clean_normal_gain = metric_gain(
        summary, "surfel_plane_normal", "surfel_plane", "clean_local_normal_p90"
    )
    combined_clean_normal_wins = lower_wins(
        summary,
        "surfel_plane_normal",
        "surfel_plane",
        "clean_local_normal_p90",
    )
    combined_global = {
        baseline: global_predicate(summary, "surfel_plane_normal", baseline)
        for baseline in ("surfel_none", "thick_none")
    }

    shuffled = "surfel_plane_shuffled_normal"
    control_target_wins = lower_wins(summary, "surfel_plane_normal", shuffled, "correct_normal_p90")
    control_clean_wins = lower_wins(
        summary, "surfel_plane_normal", shuffled, "clean_local_normal_p90"
    )
    shuffled_normal_gain = metric_gain(summary, shuffled, "surfel_plane", "correct_normal_p90")
    normal_gain_separated = shuffled_normal_gain <= 0.5 * combined_normal_gain
    primary_global = (
        "heldout_depth_rmse",
        "source_all_p90",
        "source_corrupted_p90",
    )
    control_global_wins = {
        metric: lower_wins(summary, "surfel_plane_normal", shuffled, metric)
        for metric in primary_global
    }
    control_global_gain = {
        metric: metric_gain(summary, shuffled, "surfel_none", metric) for metric in primary_global
    }
    correct_global_gain = {
        metric: metric_gain(summary, "surfel_plane_normal", "surfel_none", metric)
        for metric in primary_global
    }
    global_control_separation = all(
        control_global_wins[metric] >= 2
        and control_global_gain[metric] <= 0.5 * correct_global_gain[metric]
        for metric in primary_global
    )
    control_pass = (
        control_target_wins == 3
        and control_clean_wins >= 2
        and normal_gain_separated
        and global_control_separation
    )
    combined_local_mechanism_pass = (
        combined_plane_gain >= 0.25
        and combined_plane_wins == 3
        and combined_normal_gain >= 0.25
        and combined_normal_wins == 3
        and combined_local_all_gain >= 0.15
        and combined_local_corrupt_gain >= 0.20
        and combined_local_all_wins >= 2
        and combined_local_corrupt_wins >= 2
        and combined_clean_normal_gain >= 0.20
        and combined_clean_normal_wins >= 2
    )
    combined_global_pass = all(item["pass"] for item in combined_global.values())
    thin_surface_initialization_only = bool(combined_global["thick_none"]["pass"]) and not bool(
        combined_global["surfel_none"]["pass"]
    )
    combined_pass = combined_local_mechanism_pass and combined_global_pass and control_pass

    if combined_pass:
        next_action = "hybrid_real_calibrated_plane_normal_replication"
    elif plane_pass:
        next_action = "retain_plane_reject_shortest_axis_normal"
    elif thin_surface_initialization_only:
        next_action = "thin_surface_covariance_initialization_not_surface_losses"
    elif combined_local_mechanism_pass and not combined_global_pass:
        next_action = "local_mechanism_global_insufficiency_close_synthetic_sweeps"
    elif combined_global_pass and not control_pass:
        next_action = "generic_surface_regularization_not_normal_attribution"
    else:
        next_action = "surface_losses_nonadvancing_close_fixed_adaptation"
    return {
        "plane_only": {
            "plane_engagement_gain_fraction": plane_engagement_gain,
            "plane_engagement_seed_wins": plane_engagement_wins,
            "clean_target_all_gain_fraction": plane_local_all_gain,
            "clean_target_all_seed_wins": plane_local_all_wins,
            "clean_target_corrupted_gain_fraction": plane_local_corrupt_gain,
            "clean_target_corrupted_seed_wins": plane_local_corrupt_wins,
            "global_predicates": plane_global,
            "pass": plane_pass,
        },
        "combined": {
            "plane_engagement_gain_fraction": combined_plane_gain,
            "plane_engagement_seed_wins": combined_plane_wins,
            "normal_engagement_gain_fraction": combined_normal_gain,
            "normal_engagement_seed_wins": combined_normal_wins,
            "clean_target_all_gain_fraction": combined_local_all_gain,
            "clean_target_all_seed_wins": combined_local_all_wins,
            "clean_target_corrupted_gain_fraction": combined_local_corrupt_gain,
            "clean_target_corrupted_seed_wins": combined_local_corrupt_wins,
            "clean_normal_gain_fraction": combined_clean_normal_gain,
            "clean_normal_seed_wins": combined_clean_normal_wins,
            "local_mechanism_pass": combined_local_mechanism_pass,
            "global_predicates": combined_global,
            "global_pass": combined_global_pass,
            "thin_surface_initialization_only": thin_surface_initialization_only,
            "control": {
                "target_normal_seed_wins": control_target_wins,
                "clean_normal_seed_wins": control_clean_wins,
                "shuffled_normal_gain_fraction": shuffled_normal_gain,
                "correct_normal_gain_fraction": combined_normal_gain,
                "normal_gain_separation_pass": normal_gain_separated,
                "global_seed_wins": control_global_wins,
                "shuffled_global_gain_fraction": control_global_gain,
                "correct_global_gain_fraction": correct_global_gain,
                "global_separation_pass": global_control_separation,
                "pass": control_pass,
            },
            "pass": combined_pass,
        },
        "thinning_only_cannot_rescue": True,
        "production_default_change_authorized": False,
        "rgb_only_gradient_in_scope": False,
        "next_action": next_action,
    }


def stopped_decision() -> dict[str, object]:
    return {
        "stopped_before_optimization": True,
        "stop_reason": "postfreeze_clean_target_audit_failure",
        "plane_only": {"pass": False},
        "combined": {"pass": False},
        "production_default_change_authorized": False,
        "next_action": "reject_four_point_cross_view_target_constructor",
    }


def prepare_seed(args: argparse.Namespace, seed: int) -> PreparedSeed:
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
    scene.train_indices = [index for index in range(args.views) if index not in scene.test_indices]
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

    layout_lifter = make_lifter("surfel_none", args, seed, corrupted_depths, iterations=0)
    layout_output = layout_lifter.lift(gaussians2d, train_scene)
    layout_core = layout_lifter.gradient
    source_xy = layout_core.source_xy_before_merge
    source_view_ids = layout_core.source_view_ids_before_merge
    source_ranges = layout_core.source_view_ranges_before_merge
    if source_xy is None or source_view_ids is None or layout_output.n != source_xy.shape[0]:
        raise AssertionError("zero-step Hybrid did not expose a canonical retained layout")
    center, extent = train_scene.center_and_extent()
    bundle = build_frozen_targets(
        corrupted_depths,
        train_scene.cameras,
        source_xy,
        source_view_ids,
        source_ranges,
        center,
        extent,
        args,
    )
    bundle.diagnostics.update(
        {
            "input_corrupted_depths_sha256": depth_tensor_hash(corrupted_depths),
            "input_source_xy_sha256": tensor_collection_hash([("source_xy", source_xy)]),
            "input_source_view_ids_sha256": tensor_collection_hash(
                [("source_view_ids", source_view_ids)]
            ),
            "input_source_ranges": source_ranges,
            "input_bounds_sha256": tensor_collection_hash(
                [
                    ("center", center),
                    ("extent", torch.tensor([extent], dtype=center.dtype)),
                ]
            ),
            "builder_received_clean_depth": False,
            "builder_received_corruption_mask": False,
            "builder_received_gt_or_heldout": False,
            "build_count": 1,
        }
    )
    apply_structural_floors(
        bundle,
        layout_output,
        source_xy,
        source_view_ids,
        source_ranges,
        corruption_masks,
        train_scene.n_views,
        args,
    )
    clean_audit = audit_targets_with_clean_depth(
        bundle,
        clean_depths,
        train_scene.cameras,
        corruption_masks,
        source_xy,
        source_view_ids,
        source_ranges,
        extent,
        args,
    )
    seed_result: dict[str, object] = {
        "seed": seed,
        "global_train_indices": scene.training_views,
        "global_test_indices": scene.testing_views,
        "local_to_global_train_indices": scene.training_views,
        "fit_seconds": fit_seconds,
        "fit_psnr_mean": statistics.fmean(item["final_psnr"] for item in fit_history),
        "fitted_tensors_sha256": fitted_tensor_hash(gaussians2d),
        "corrupted_depths_sha256": depth_tensor_hash(corrupted_depths),
        "target_construction": bundle.diagnostics,
        "postfreeze_clean_target_audit": clean_audit.diagnostics,
        "invariants": {},
        "arms": {},
    }
    return PreparedSeed(
        seed=seed,
        scene=scene,
        train_scene=train_scene,
        gaussians2d=gaussians2d,
        fit_seconds=fit_seconds,
        fit_history=fit_history,
        clean_depths=clean_depths,
        corrupted_depths=corrupted_depths,
        corruption_masks=corruption_masks,
        diagnostic_priors=diagnostic_priors,
        layout_output=layout_output,
        layout_core=layout_core,
        targets=bundle,
        clean_audit=clean_audit,
        seed_result=seed_result,
    )


def target_hashes_unchanged(bundle: FrozenTargetBundle) -> bool:
    return (
        target_tensor_hash(bundle.indices, bundle.points, bundle.normals, bundle.alignment_normals)
        == bundle.diagnostics["correct_targets_sha256"]
        and target_tensor_hash(
            bundle.indices,
            bundle.points,
            bundle.normals,
            bundle.shuffled_alignment_normals,
        )
        == bundle.diagnostics["shuffled_targets_sha256"]
    )


def evaluate_prepared_seed(prepared: PreparedSeed, args: argparse.Namespace) -> None:
    invariants, initial_outputs = assert_initialization_invariants(
        prepared.gaussians2d,
        prepared.train_scene,
        prepared.corrupted_depths,
        prepared.targets,
        args,
        prepared.seed,
    )
    prepared.seed_result["invariants"] = invariants
    for arm in ARMS:
        prepared.seed_result["arms"][arm] = evaluate_arm(
            arm,
            prepared.gaussians2d,
            prepared.scene,
            prepared.train_scene,
            prepared.clean_depths,
            prepared.corrupted_depths,
            prepared.diagnostic_priors,
            prepared.corruption_masks,
            prepared.targets,
            prepared.clean_audit,
            initial_outputs[arm],
            args,
            prepared.seed,
        )
        if not target_hashes_unchanged(prepared.targets):
            raise AssertionError(f"{arm} mutated frozen target tensors")

    arms = prepared.seed_result["arms"]
    counts = {arm: arms[arm]["n_initial"] for arm in ARMS}
    if len(set(counts.values())) != 1:
        raise AssertionError(f"arm primitive counts differ: {counts}")
    schedules = {arm: arms[arm]["target_view_history"] for arm in ARMS}
    if any(schedule != schedules[ARMS[0]] for schedule in schedules.values()):
        raise AssertionError("target-view schedules differ across surface arms")
    if args.output is not None and set(schedules[ARMS[0]]) != set(
        prepared.train_scene.training_views
    ):
        raise AssertionError("official schedule does not visit all training views")
    for arm in ARMS:
        if arms[arm]["rendered_count_history"] != [counts[arm]] * args.lift_iters:
            raise AssertionError(f"{arm} did not render the inclusive full primitive set")
    axis_hashes = {arms[arm]["stored_axis_indices_sha256"] for arm in SURFEL_ARMS}
    if len(axis_hashes) != 1:
        raise AssertionError("surfel arms changed frozen shortest-axis indices")
    correct_target_hashes = {
        arms[arm]["stored_targets_sha256"]
        for arm in SURFEL_ARMS
        if arm != "surfel_plane_shuffled_normal"
    }
    if len(correct_target_hashes) != 1:
        raise AssertionError("correct surfel arms changed stored target tensors")
    prepared.seed_result["cross_arm_schedule_count_target_invariants"] = True


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parent.parent
    torch.set_num_threads(args.threads)
    prepared = [prepare_seed(args, seed) for seed in args.seeds]
    runs = [item.seed_result for item in prepared]
    audit_pass_by_seed = [bool(item.clean_audit.diagnostics["pass"]) for item in prepared]
    clean_audit_gate = {
        "pass_by_seed": audit_pass_by_seed,
        "all_pass": all(audit_pass_by_seed),
        "stops_before_optimization_on_failure": True,
    }
    if clean_audit_gate["all_pass"]:
        for item in prepared:
            evaluate_prepared_seed(item, args)
        summary: dict[str, object] = summarize(runs)
        final_decision = decision(summary)
        for arm in ARMS:
            for metric in SUMMARY_PATHS:
                if len(summary[arm][metric]["samples"]) != len(runs):
                    raise AssertionError(f"{arm}/{metric} has incomplete samples")
    else:
        summary = {}
        final_decision = stopped_decision()

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
                "incremental_gaussian_triangulation_arxiv": "2607.10690",
                "fixed_cpu_repository_adaptation": True,
                "paper_code_reused": False,
                "hybrid_only": True,
            },
            "protocol_order": [
                "fit_physically_subset_training_images",
                "construct_corrupted_training_depth",
                "freeze_zero_step_hybrid_layout",
                "build_and_hash_cross_view_targets_once",
                "apply_structural_floors",
                "audit_frozen_targets_with_clean_synthetic_depth",
                "run_five_paired_hybrid_arms_only_if_audit_passes",
            ],
        },
        "clean_target_audit_gate": clean_audit_gate,
        "runs": runs,
        "summary": summary,
        "decision": final_decision,
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
    parser.add_argument("--lr-rotation", type=float, default=0.005)
    parser.add_argument("--depth-jitter", type=float, default=0.02)
    parser.add_argument("--depth-prior-lambda", type=float, default=0.01)
    parser.add_argument("--thick-ray-thickness", type=float, default=1.0)
    parser.add_argument("--surfel-ray-thickness", type=float, default=0.15)
    parser.add_argument("--plane-lambda", type=float, default=0.05)
    parser.add_argument("--normal-lambda", type=float, default=0.2)
    parser.add_argument("--target-neighbors", type=int, default=4)
    parser.add_argument("--target-min-support-views", type=int, default=2)
    parser.add_argument("--target-max-radius-frac", type=float, default=0.10)
    parser.add_argument("--target-planarity-ratio", type=float, default=0.05)
    parser.add_argument("--target-min-incidence", type=float, default=0.10)
    parser.add_argument("--min-targets", type=int, default=300)
    parser.add_argument("--min-target-coverage", type=float, default=0.23)
    parser.add_argument("--min-corrupted-targets", type=int, default=60)
    parser.add_argument("--min-targets-per-source", type=int, default=20)
    parser.add_argument("--max-farthest-neighbor-p90-frac", type=float, default=0.08)
    parser.add_argument("--min-incidence-p10", type=float, default=0.10)
    parser.add_argument("--min-eigenvalue-mid-ratio", type=float, default=0.01)
    parser.add_argument("--min-scale-ratio-p10", type=float, default=5.0)
    parser.add_argument("--min-shuffle-separation-median", type=float, default=0.25)
    parser.add_argument("--max-clean-plane-p90", type=float, default=0.10)
    parser.add_argument("--min-clean-normal-cosine", type=float, default=0.50)
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
        "lr_rotation": 0.005,
        "depth_jitter": 0.02,
        "depth_prior_lambda": 0.01,
        "thick_ray_thickness": 1.0,
        "surfel_ray_thickness": 0.15,
        "plane_lambda": 0.05,
        "normal_lambda": 0.2,
        "target_neighbors": 4,
        "target_min_support_views": 2,
        "target_max_radius_frac": 0.10,
        "target_planarity_ratio": 0.05,
        "target_min_incidence": 0.10,
        "min_targets": 300,
        "min_target_coverage": 0.23,
        "min_corrupted_targets": 60,
        "min_targets_per_source": 20,
        "max_farthest_neighbor_p90_frac": 0.08,
        "min_incidence_p10": 0.10,
        "min_eigenvalue_mid_ratio": 0.01,
        "min_scale_ratio_p10": 5.0,
        "min_shuffle_separation_median": 0.25,
        "max_clean_plane_p90": 0.10,
        "min_clean_normal_cosine": 0.50,
        "min_weight": 0.05,
        "block_size": 8,
        "threads": 4,
    }
    return all(getattr(args, key) == value for key, value in expected.items())


def main() -> int:
    args = parse_args()
    if args.output is not None and not is_preregistered_config(args):
        raise ValueError("tracked output requires the exact preregistered configuration")
    if args.output is not None and not PREREGISTRATION.exists():
        raise FileNotFoundError(f"tracked output requires preregistration {PREREGISTRATION}")
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"refusing to overwrite official output {args.output}")
    if args.output is not None:
        previous = [
            path
            for path in PREREGISTRATION.parent.glob("*_cpu_surface_plane_normal.json")
            if path != args.output
        ]
        if previous:
            raise FileExistsError(f"official surface artifact already exists: {previous[0]}")
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
