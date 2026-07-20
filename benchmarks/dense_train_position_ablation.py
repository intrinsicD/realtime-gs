#!/usr/bin/env python3
"""Paired dense train-only position-correspondence experiment."""

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
from typing import Any

import torch

from cross_view_supervision_ablation import (
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
from rtgs.lift.matching import MatchLayout, get_position_matcher
from rtgs.optim.trainer import Trainer
from rtgs.render.torch_ref import TorchRasterizer
from world_position_consistency_ablation import (
    assert_layout_equal,
    assert_reference_contribution_parity,
    core_lifter,
    distribution,
    execute_lift,
    finite_result,
    make_lifter,
    reference_contributions,
)

FAMILIES = ("gradient", "hybrid")
MODES = ("none", "dense_train_position", "degree_shuffled_position")
PREREGISTRATION = Path("benchmarks/results/20260715_dense_train_position_PREREG.md")

# The preceding sparse fixed-pair experiment's represented-node counts. These are immutable
# historical comparators, not inputs to matching or optimization.
SPARSE_REFERENCE_NODES = {0: 106, 1: 100, 2: 119}
_NEAR = 0.05


@dataclass(frozen=True)
class FixedTrainGraph:
    """A graph frozen exclusively from training RGB, cameras, and retained source pixels."""

    pairs: torch.Tensor
    shuffled_pairs: torch.Tensor
    confidence: torch.Tensor
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class GraphAudit:
    """Post-freeze synthetic GT audit; never passed into the matcher or graph control."""

    dominant_gt_ids: torch.Tensor
    diagnostics: dict[str, object]


@dataclass
class PreparedSeed:
    seed: int
    scene: Any
    train_scene: Any
    gaussians2d: Any
    fit_history: list[dict[str, float]]
    fit_seconds: float
    layout_output: Any
    layout_core: GradientLifter
    graph: FixedTrainGraph
    audit: GraphAudit
    seed_result: dict[str, object]


def json_ready(value: Any) -> Any:
    """Convert matcher diagnostics to strict JSON without changing numeric values."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"matcher diagnostic {type(value).__name__} is not JSON serializable")


def loaded_source_hashes(root: Path) -> tuple[dict[str, str], str]:
    """Hash loaded repository Python sources and the frozen preregistration."""
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


def image_hash(images: list[torch.Tensor]) -> str:
    return tensor_collection_hash(
        [(f"train_image_{index}", image) for index, image in enumerate(images)]
    )


def camera_hash(cameras) -> str:
    tensors = []
    for index, camera in enumerate(cameras):
        intrinsics = torch.tensor(
            [
                camera.fx,
                camera.fy,
                camera.cx,
                camera.cy,
                camera.width,
                camera.height,
            ],
            dtype=torch.float64,
        )
        tensors.extend(
            [
                (f"camera_{index}_intrinsics", intrinsics),
                (f"camera_{index}_R", camera.R),
                (f"camera_{index}_t", camera.t),
            ]
        )
    return tensor_collection_hash(tensors)


def source_pair_counter(
    pairs: torch.Tensor, source_view_ids: torch.Tensor
) -> Counter[tuple[int, int]]:
    return Counter(
        zip(
            source_view_ids[pairs[:, 0]].tolist(),
            source_view_ids[pairs[:, 1]].tolist(),
        )
    )


def pair_residual_diagnostics(
    means: torch.Tensor, pairs: torch.Tensor, extent: float
) -> dict[str, float | int | None]:
    residual = means[pairs[:, 0]].sub(means[pairs[:, 1]]).abs().sum(dim=-1) / extent
    return distribution(residual)


def pair_loss_covariates(
    means: torch.Tensor,
    pairs: torch.Tensor,
    extent: float,
    beta: float,
) -> dict[str, object]:
    residual = means[pairs[:, 0]].sub(means[pairs[:, 1]]).abs().sum(dim=-1) / extent
    huber = torch.where(
        residual <= beta,
        0.5 * residual.square(),
        beta * (residual - 0.5 * beta),
    )
    return {
        "normalized_l1": distribution(residual),
        "huber_delta": beta,
        "per_edge_huber": distribution(huber),
        "mean_huber": float(huber.mean()),
    }


def pair_ray_covariates(
    cameras,
    source_xy: torch.Tensor,
    source_view_ids: torch.Tensor,
    pairs: torch.Tensor,
) -> dict[str, object]:
    """Compute train-camera-only ray covariates after the graph is frozen."""
    angles = []
    gaps = []
    left_parameters = []
    right_parameters = []
    reprojection_errors = []
    left_depths = []
    right_depths = []
    for left_index, right_index in pairs.tolist():
        left_view = int(source_view_ids[left_index])
        right_view = int(source_view_ids[right_index])
        left_origin, left_direction = cameras[left_view].pixel_rays(
            source_xy[left_index : left_index + 1]
        )
        right_origin, right_direction = cameras[right_view].pixel_rays(
            source_xy[right_index : right_index + 1]
        )
        left_direction = torch.nn.functional.normalize(left_direction[0], dim=0)
        right_direction = torch.nn.functional.normalize(right_direction[0], dim=0)
        delta = left_origin - right_origin
        cosine = torch.dot(left_direction, right_direction)
        denominator = 1.0 - cosine.square()
        if float(denominator) <= 1e-8:
            raise AssertionError("frozen graph contains near-parallel rays")
        left_dot = torch.dot(left_direction, delta)
        right_dot = torch.dot(right_direction, delta)
        left_parameter = (cosine * right_dot - left_dot) / denominator
        right_parameter = (right_dot - cosine * left_dot) / denominator
        left_point = left_origin + left_parameter * left_direction
        right_point = right_origin + right_parameter * right_direction
        midpoint = 0.5 * (left_point + right_point)
        left_reprojected, left_depth = cameras[left_view].project(midpoint[None])
        right_reprojected, right_depth = cameras[right_view].project(midpoint[None])
        line_cosine = cosine.abs().clamp(0.0, 1.0)
        angles.append(math.degrees(math.acos(float(line_cosine))))
        gaps.append(float((left_point - right_point).norm()))
        left_parameters.append(float(left_parameter))
        right_parameters.append(float(right_parameter))
        reprojection_errors.append(
            max(
                float((left_reprojected[0] - source_xy[left_index]).norm()),
                float((right_reprojected[0] - source_xy[right_index]).norm()),
            )
        )
        left_depths.append(float(left_depth[0]))
        right_depths.append(float(right_depth[0]))
    return {
        "ray_angle_deg": distribution(torch.tensor(angles)),
        "closest_line_gap_world": distribution(torch.tensor(gaps)),
        "left_closest_line_parameter_world": distribution(torch.tensor(left_parameters)),
        "right_closest_line_parameter_world": distribution(torch.tensor(right_parameters)),
        "midpoint_reprojection_error_px": distribution(torch.tensor(reprojection_errors)),
        "midpoint_left_depth": distribution(torch.tensor(left_depths)),
        "midpoint_right_depth": distribution(torch.tensor(right_depths)),
    }


def layout_patch_descriptors(
    images: list[torch.Tensor],
    source_xy: torch.Tensor,
    source_ranges: list[tuple[int, int]],
    radius: int,
) -> torch.Tensor:
    offsets = torch.tensor(
        list(itertools.product(range(-radius, radius + 1), repeat=2)),
        dtype=source_xy.dtype,
        device=source_xy.device,
    )[:, [1, 0]]
    descriptors = []
    for image, (start, end) in zip(images, source_ranges):
        samples = source_xy[start:end, None, :] + offsets[None, :, :]
        values = bilinear_sample(image.to(source_xy), samples.reshape(-1, 2))
        descriptors.append(values.reshape(end - start, -1))
    return torch.cat(descriptors)


def pair_patch_epipolar_covariates(
    descriptors: torch.Tensor,
    cameras,
    source_xy: torch.Tensor,
    source_view_ids: torch.Tensor,
    pairs: torch.Tensor,
) -> dict[str, object]:
    descriptor_distance = (descriptors[pairs[:, 0]] - descriptors[pairs[:, 1]]).norm(dim=-1)
    epipolar_distances = []
    for left_index, right_index in pairs.tolist():
        left_view = int(source_view_ids[left_index])
        right_view = int(source_view_ids[right_index])
        left_camera = cameras[left_view]
        right_camera = cameras[right_view]
        relative_rotation = right_camera.R @ left_camera.R.T
        relative_translation = right_camera.t - relative_rotation @ left_camera.t
        tx, ty, tz = relative_translation
        skew = relative_translation.new_tensor([[0.0, -tz, ty], [tz, 0.0, -tx], [-ty, tx, 0.0]])
        fundamental = (
            torch.linalg.inv(right_camera.K).T
            @ skew
            @ relative_rotation
            @ torch.linalg.inv(left_camera.K)
        )
        fundamental = fundamental / fundamental.norm().clamp_min(1e-12)
        left_h = torch.cat([source_xy[left_index], source_xy.new_ones(1)])
        right_h = torch.cat([source_xy[right_index], source_xy.new_ones(1)])
        error = torch.dot(right_h, fundamental.to(source_xy) @ left_h).abs()
        right_line = fundamental.to(source_xy) @ left_h
        left_line = fundamental.to(source_xy).T @ right_h
        epipolar_distances.append(
            float(
                torch.maximum(
                    error / right_line[:2].norm().clamp_min(1e-12),
                    error / left_line[:2].norm().clamp_min(1e-12),
                )
            )
        )
    return {
        "descriptor_l2": distribution(descriptor_distance),
        "max_bidirectional_epipolar_distance_px": distribution(torch.tensor(epipolar_distances)),
    }


def _validate_pair_tensor(name: str, pairs: torch.Tensor, n_nodes: int) -> None:
    if pairs.dtype != torch.long or pairs.ndim != 2 or pairs.shape[1] != 2:
        raise AssertionError(f"{name} must be an (E,2) int64 tensor")
    if pairs.shape[0] == 0:
        raise AssertionError(f"{name} is empty")
    if int(pairs.min()) < 0 or int(pairs.max()) >= n_nodes:
        raise AssertionError(f"{name} contains an out-of-range endpoint")
    if not bool((pairs[:, 0] < pairs[:, 1]).all()):
        raise AssertionError(f"{name} endpoints are not canonical")
    if torch.unique(pairs, dim=0).shape[0] != pairs.shape[0]:
        raise AssertionError(f"{name} contains duplicate edges")


def freeze_train_graph(
    images: list[torch.Tensor],
    cameras,
    source_xy: torch.Tensor,
    source_view_ids: torch.Tensor,
    source_ranges: list[tuple[int, int]],
    args: argparse.Namespace,
    seed: int,
) -> FixedTrainGraph:
    """Build the graph once, with an API that cannot receive GT, depth, or held-out data."""
    layout = MatchLayout(
        xy=source_xy.detach().clone(),
        source_view_ids=source_view_ids.detach().clone(),
        source_ranges=list(source_ranges),
    )
    matcher = get_position_matcher(
        args.matcher,
        patch_radius=args.match_patch_radius,
        max_epipolar_distance=args.match_max_epipolar_error_px,
        max_ratio=args.match_ratio_threshold,
        min_ray_angle_deg=args.match_min_ray_angle_deg,
        min_depth=args.match_min_depth,
        max_reprojection_error=args.match_max_reprojection_error_px,
        min_block_edges=args.match_min_block_edges,
        ratio_epsilon=args.match_ratio_epsilon,
    )
    matched = matcher.match(images, cameras, layout)
    pairs = matched.pairs.detach().cpu().long().contiguous()
    shuffled_pairs = matched.shuffled_pairs.detach().cpu().long().contiguous()
    confidence = matched.confidence.detach().cpu().contiguous()
    n_nodes = source_view_ids.numel()
    _validate_pair_tensor("dense train graph", pairs, n_nodes)
    _validate_pair_tensor("degree-shuffled graph", shuffled_pairs, n_nodes)
    if pairs.shape != shuffled_pairs.shape:
        raise AssertionError("positive and shuffled graphs differ in edge count")
    if confidence.shape != (pairs.shape[0],) or not bool(torch.isfinite(confidence).all()):
        raise AssertionError("matcher confidence is incomplete or non-finite")
    if bool((source_view_ids[pairs[:, 0]] == source_view_ids[pairs[:, 1]]).any()) or bool(
        (source_view_ids[shuffled_pairs[:, 0]] == source_view_ids[shuffled_pairs[:, 1]]).any()
    ):
        raise AssertionError("matcher graph contains a same-source edge")

    positive_degree = torch.bincount(pairs.flatten(), minlength=n_nodes)
    shuffled_degree = torch.bincount(shuffled_pairs.flatten(), minlength=n_nodes)
    if not torch.equal(positive_degree, shuffled_degree):
        raise AssertionError("degree-shuffled graph changes endpoint degree")
    positive_source_pairs = source_pair_counter(pairs, source_view_ids)
    shuffled_source_pairs = source_pair_counter(shuffled_pairs, source_view_ids)
    if positive_source_pairs != shuffled_source_pairs:
        raise AssertionError("degree-shuffled graph changes source-view pair counts")

    blocks = {}
    for source_pair, edge_count in sorted(positive_source_pairs.items()):
        left_view, right_view = source_pair
        positive_mask = (source_view_ids[pairs[:, 0]] == left_view) & (
            source_view_ids[pairs[:, 1]] == right_view
        )
        shuffled_mask = (source_view_ids[shuffled_pairs[:, 0]] == left_view) & (
            source_view_ids[shuffled_pairs[:, 1]] == right_view
        )
        positive_block = pairs[positive_mask]
        shuffled_block = shuffled_pairs[shuffled_mask]
        left_exact = Counter(positive_block[:, 0].tolist()) == Counter(
            shuffled_block[:, 0].tolist()
        )
        right_exact = Counter(positive_block[:, 1].tolist()) == Counter(
            shuffled_block[:, 1].tolist()
        )
        if not left_exact or not right_exact:
            raise AssertionError(f"shuffled block {source_pair} changes endpoint multisets")
        blocks[f"{left_view}-{right_view}"] = {
            "edge_count": edge_count,
            "camera_baseline_world": float(
                (cameras[left_view].position - cameras[right_view].position).norm()
            ),
            "left_endpoint_multiset_exact": left_exact,
            "right_endpoint_multiset_exact": right_exact,
        }

    positive_edges = set(map(tuple, pairs.tolist()))
    shuffled_edges = set(map(tuple, shuffled_pairs.tolist()))
    exact_overlap = len(positive_edges.intersection(shuffled_edges))
    if exact_overlap:
        raise AssertionError("degree-shuffled graph retains exact positive edges")

    represented = positive_degree > 0
    represented_views = sorted(torch.unique(source_view_ids[represented]).tolist())
    nodes_per_view = {
        str(view): int((represented & (source_view_ids == view)).sum())
        for view in range(len(cameras))
    }
    sparse_reference = SPARSE_REFERENCE_NODES.get(seed)
    density_multiplier = (
        None if sparse_reference is None else float(represented.sum()) / sparse_reference
    )
    floor_checks = {
        "edge_count": int(pairs.shape[0]) >= args.min_graph_edges,
        "represented_node_count": int(represented.sum()) >= args.min_graph_nodes,
        "represented_node_fraction": float(represented.float().mean()) >= args.min_graph_coverage,
        "view_pair_block_count": len(blocks) >= args.min_graph_blocks,
        "represented_view_count": len(represented_views) >= args.min_graph_views,
        "represented_nodes_per_view": min(nodes_per_view.values()) >= args.min_graph_nodes_per_view,
        "sparse_density_multiplier": density_multiplier is None
        or density_multiplier >= args.min_sparse_density_multiplier,
    }
    diagnostics: dict[str, object] = {
        "source": "train_rgb_camera_patch_epipolar_matcher",
        "seed": seed,
        "edge_count": int(pairs.shape[0]),
        "represented_node_count": int(represented.sum()),
        "represented_node_fraction": float(represented.float().mean()),
        "view_pair_block_count": len(blocks),
        "represented_views": represented_views,
        "represented_nodes_per_view": nodes_per_view,
        "sparse_reference_node_count": sparse_reference,
        "sparse_reference_density_multiplier": density_multiplier,
        "blocks": blocks,
        "matcher": json_ready(matched.diagnostics),
        "confidence": distribution(confidence),
        "positive_pairs": pairs.tolist(),
        "degree_shuffled_pairs": shuffled_pairs.tolist(),
        "positive_confidence": confidence.tolist(),
        "degree_shuffled_confidence": confidence.tolist(),
        "positive_pairs_sha256": tensor_collection_hash([("position_pairs", pairs)]),
        "shuffled_pairs_sha256": tensor_collection_hash([("position_pairs", shuffled_pairs)]),
        "confidence_sha256": tensor_collection_hash([("confidence", confidence)]),
        "endpoint_degree_sha256": tensor_collection_hash([("endpoint_degree", positive_degree)]),
        "endpoint_degree": distribution(positive_degree[represented]),
        "input_train_images_sha256": image_hash(images),
        "input_train_cameras_sha256": camera_hash(cameras),
        "input_source_xy_sha256": tensor_collection_hash([("source_xy", source_xy)]),
        "input_source_view_ids_sha256": tensor_collection_hash(
            [("source_view_ids", source_view_ids)]
        ),
        "input_source_ranges": source_ranges,
        "floor_checks": floor_checks,
        "structural_floor_pass": all(floor_checks.values()),
        "degree_exact": True,
        "source_pair_counts_exact": True,
        "cross_source_edges_only": True,
        "block_endpoint_multisets_exact": True,
        "confidence_multiset_exact": True,
        "exact_edge_overlap_count": exact_overlap,
        "gt_or_depth_consumed_by_matcher": False,
        "heldout_data_consumed_by_matcher": False,
        "graph_build_count": 1,
    }
    return FixedTrainGraph(
        pairs=pairs,
        shuffled_pairs=shuffled_pairs,
        confidence=confidence,
        diagnostics=diagnostics,
    )


def attach_postfreeze_train_covariates(
    graph: FixedTrainGraph,
    images: list[torch.Tensor],
    cameras,
    source_xy: torch.Tensor,
    source_view_ids: torch.Tensor,
    source_ranges: list[tuple[int, int]],
    initial_means: torch.Tensor,
    extent: float,
    patch_radius: int,
    position_beta: float,
) -> None:
    """Record non-GT control covariates without changing the frozen graph."""
    descriptors = layout_patch_descriptors(images, source_xy, source_ranges, patch_radius)
    graph.diagnostics["postfreeze_train_covariates"] = {
        "positive_patch_epipolar": pair_patch_epipolar_covariates(
            descriptors, cameras, source_xy, source_view_ids, graph.pairs
        ),
        "shuffled_patch_epipolar": pair_patch_epipolar_covariates(
            descriptors, cameras, source_xy, source_view_ids, graph.shuffled_pairs
        ),
        "positive_ray_geometry": pair_ray_covariates(
            cameras, source_xy, source_view_ids, graph.pairs
        ),
        "shuffled_ray_geometry": pair_ray_covariates(
            cameras, source_xy, source_view_ids, graph.shuffled_pairs
        ),
        "step0_positive_edge_loss": pair_loss_covariates(
            initial_means, graph.pairs, extent, position_beta
        ),
        "step0_shuffled_edge_loss": pair_loss_covariates(
            initial_means,
            graph.shuffled_pairs,
            extent,
            position_beta,
        ),
    }


def edge_identity_audit(
    pairs: torch.Tensor, dominant_gt_ids: torch.Tensor
) -> dict[str, float | int | None]:
    left_ids = dominant_gt_ids[pairs[:, 0]]
    right_ids = dominant_gt_ids[pairs[:, 1]]
    both_labeled = (left_ids >= 0) & (right_ids >= 0)
    true_edge = both_labeled & (left_ids == right_ids)
    count = int(pairs.shape[0])
    return {
        "edge_count": count,
        "strict_true_edge_count": int(true_edge.sum()),
        "strict_precision": float(true_edge.float().mean()),
        "both_endpoints_labeled_count": int(both_labeled.sum()),
        "both_endpoints_labeled_fraction": float(both_labeled.float().mean()),
        "precision_among_labeled_pairs": (
            None if not bool(both_labeled.any()) else float(true_edge.sum() / both_labeled.sum())
        ),
        "unlabeled_endpoints_counted_false": True,
    }


def audit_frozen_graph_with_gt(
    train_scene,
    source_xy: torch.Tensor,
    source_view_ids: torch.Tensor,
    source_ranges: list[tuple[int, int]],
    graph: FixedTrainGraph,
    args: argparse.Namespace,
) -> GraphAudit:
    """Assign strict dominant GT contributors only after both graph tensors are fixed."""
    if train_scene.gt_gaussians is None:
        raise ValueError("the post-freeze audit requires synthetic GT Gaussians")
    gt = train_scene.gt_gaussians
    n_nodes = source_view_ids.numel()
    dominant_gt_ids = torch.full((n_nodes,), -1, dtype=torch.long)
    contribution = torch.zeros(n_nodes)
    purity = torch.zeros(n_nodes)
    parity = []
    with torch.no_grad():
        for view_index, ((start, end), camera) in enumerate(
            zip(source_ranges, train_scene.cameras)
        ):
            if not bool((source_view_ids[start:end] == view_index).all()):
                raise AssertionError(f"view {view_index} retained layout is inconsistent")
            parity.append(assert_reference_contribution_parity(gt, camera))
            values = reference_contributions(gt, camera, source_xy[start:end])
            total = values.sum(dim=1)
            strongest, strongest_ids = values.max(dim=1)
            strongest_purity = strongest / total.clamp_min(1e-12)
            accepted = (
                (total >= args.audit_min_total_contribution)
                & (strongest >= args.audit_min_contribution)
                & (strongest_purity >= args.audit_min_purity)
            )
            local_ids = torch.where(accepted, strongest_ids, torch.full_like(strongest_ids, -1))
            dominant_gt_ids[start:end] = local_ids.cpu()
            contribution[start:end] = strongest.cpu()
            purity[start:end] = strongest_purity.cpu()

    degree = torch.bincount(graph.pairs.flatten(), minlength=n_nodes)
    represented = degree > 0
    represented_labeled = represented & (dominant_gt_ids >= 0)
    positive = edge_identity_audit(graph.pairs, dominant_gt_ids)
    shuffled = edge_identity_audit(graph.shuffled_pairs, dominant_gt_ids)
    shuffled_precision = float(shuffled["strict_precision"])
    positive_precision = float(positive["strict_precision"])
    precision_ratio = None if shuffled_precision == 0.0 else positive_precision / shuffled_precision
    diagnostics: dict[str, object] = {
        "timing": "after_positive_and_control_graph_frozen",
        "dominant_identity_rule": {
            "minimum_total_contribution": args.audit_min_total_contribution,
            "minimum_contribution": args.audit_min_contribution,
            "minimum_purity": args.audit_min_purity,
            "unlabeled_edges_count_as_false": True,
        },
        "represented_node_count": int(represented.sum()),
        "represented_labeled_node_count": int(represented_labeled.sum()),
        "represented_unlabeled_node_count": int((represented & ~represented_labeled).sum()),
        "represented_labeled_node_fraction": float(
            represented_labeled.sum() / represented.sum().clamp_min(1)
        ),
        "represented_dominant_contribution": distribution(contribution[represented]),
        "represented_dominant_purity": distribution(purity[represented]),
        "accepted_dominant_contribution": distribution(contribution[represented_labeled]),
        "accepted_dominant_purity": distribution(purity[represented_labeled]),
        "positive_edges": positive,
        "degree_shuffled_edges": shuffled,
        "positive_to_shuffled_precision_ratio": precision_ratio,
        "reference_contribution_parity": parity,
        "dominant_gt_ids_sha256": tensor_collection_hash([("dominant_gt_ids", dominant_gt_ids)]),
        "gt_used_to_construct_or_select_edges": False,
    }
    return GraphAudit(dominant_gt_ids=dominant_gt_ids, diagnostics=diagnostics)


def pairs_for_mode(graph: FixedTrainGraph, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode == "dense_train_position":
        return graph.pairs
    if mode == "degree_shuffled_position":
        return graph.shuffled_pairs
    raise ValueError(f"unknown position mode {mode!r}")


def assert_family_invariants(
    family: str,
    gaussians2d,
    train_scene,
    corrupted_depths,
    graph: FixedTrainGraph,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    _, extent = train_scene.center_and_extent()
    zero_outputs = {}
    zero_cores = {}
    for mode in MODES:
        lifter = make_lifter(family, mode, args, seed, corrupted_depths, iterations=0)
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
    if not torch.equal(zero_cores["dense_train_position"].position_pairs_before_merge, graph.pairs):
        raise AssertionError(f"{family} changed the dense train graph")
    if not torch.equal(
        zero_cores["degree_shuffled_position"].position_pairs_before_merge,
        graph.shuffled_pairs,
    ):
        raise AssertionError(f"{family} changed the shuffled graph")

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
            raise AssertionError(f"{family}/{mode} changes history at zero position lambda")
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
        "positive_pairs_sha256": graph.diagnostics["positive_pairs_sha256"],
        "shuffled_pairs_sha256": graph.diagnostics["shuffled_pairs_sha256"],
        "step0_positive_edge_residual": pair_residual_diagnostics(
            reference_output.means, graph.pairs, extent
        ),
        "step0_degree_shuffled_edge_residual": pair_residual_diagnostics(
            reference_output.means, graph.shuffled_pairs, extent
        ),
    }


def dominant_gt_diagnostics(
    scene,
    means: torch.Tensor,
    dominant_gt_ids: torch.Tensor,
    represented: torch.Tensor,
) -> dict[str, float | int | None]:
    labeled = represented & (dominant_gt_ids >= 0)
    if scene.gt_gaussians is None or not bool(labeled.any()):
        raise AssertionError("dominant-GT diagnostics require labeled represented nodes")
    _, extent = scene.center_and_extent()
    distance = (means[labeled] - scene.gt_gaussians.means[dominant_gt_ids[labeled]]).norm(
        dim=-1
    ) / extent
    result = distribution(distance)
    result.update(
        {
            "represented_count": int(represented.sum()),
            "labeled_count": int(labeled.sum()),
            "unlabeled_count": int((represented & ~labeled).sum()),
        }
    )
    return result


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
    graph: FixedTrainGraph,
    audit: GraphAudit,
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
    represented = torch.bincount(graph.pairs.flatten(), minlength=audit.dominant_gt_ids.numel()) > 0
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
        "positive_edge_residual": pair_residual_diagnostics(result.means, graph.pairs, extent),
        "shuffled_edge_residual": pair_residual_diagnostics(
            result.means, graph.shuffled_pairs, extent
        ),
        "dominant_gt": dominant_gt_diagnostics(
            train_scene, result.means, audit.dominant_gt_ids, represented
        ),
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
    "positive_edge_median": ("positive_edge_residual", "median"),
    "positive_edge_p90": ("positive_edge_residual", "p90"),
    "shuffled_edge_p90": ("shuffled_edge_residual", "p90"),
    "dominant_gt_median": ("dominant_gt", "median"),
    "dominant_gt_p90": ("dominant_gt", "p90"),
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


def family_decision(summary, family: str, graph_quality_pass: bool) -> dict[str, object]:
    primary_metrics = ["heldout_depth_rmse", "source_all_p90"]
    if family == "hybrid":
        primary_metrics.append("source_corrupted_p90")
    gains = {
        metric: metric_gain(summary, family, "dense_train_position", metric)
        for metric in primary_metrics
    }
    shuffled_gains = {
        metric: metric_gain(summary, family, "degree_shuffled_position", metric)
        for metric in primary_metrics
    }
    baseline_wins = {
        metric: lower_wins(summary, family, "dense_train_position", "none", metric)
        for metric in primary_metrics
    }
    shuffled_wins = {
        metric: lower_wins(
            summary,
            family,
            "dense_train_position",
            "degree_shuffled_position",
            metric,
        )
        for metric in primary_metrics
    }

    edge_gain = metric_gain(summary, family, "dense_train_position", "positive_edge_p90")
    edge_wins = lower_wins(summary, family, "dense_train_position", "none", "positive_edge_p90")
    dominant_gain = metric_gain(summary, family, "dense_train_position", "dominant_gt_p90")
    dominant_wins = lower_wins(summary, family, "dense_train_position", "none", "dominant_gt_p90")
    engagement = edge_gain >= 0.25 and edge_wins == 3
    local_geometry = dominant_gain >= 0.20 and dominant_wins >= 2

    psnr_delta = metric_mean(summary, family, "dense_train_position", "test_psnr") - metric_mean(
        summary, family, "none", "test_psnr"
    )
    coverage_delta = metric_mean(summary, family, "dense_train_position", "coverage") - metric_mean(
        summary, family, "none", "coverage"
    )
    iou_delta = metric_mean(summary, family, "dense_train_position", "alpha_iou") - metric_mean(
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
    attributed = graph_quality_pass and material and control_separation
    generic_graph_regularization = graph_quality_pass and material and not control_separation
    return {
        "dense_train_gain_fraction": gains,
        "degree_shuffled_gain_fraction": shuffled_gains,
        "dense_train_vs_none_seed_wins": baseline_wins,
        "dense_train_vs_shuffled_seed_wins": shuffled_wins,
        "positive_edge_p90_gain_fraction": edge_gain,
        "positive_edge_p90_seed_wins": edge_wins,
        "position_engagement_pass": engagement,
        "dominant_gt_p90_gain_fraction": dominant_gain,
        "dominant_gt_p90_seed_wins": dominant_wins,
        "local_geometry_mechanism_pass": local_geometry,
        "dense_train_vs_none_psnr_delta_db": psnr_delta,
        "dense_train_vs_none_coverage_delta": coverage_delta,
        "dense_train_vs_none_alpha_iou_delta": iou_delta,
        "material_effect_pass": material,
        "degree_shuffled_control_separation_pass": control_separation,
        "graph_quality_pass": graph_quality_pass,
        "generic_graph_regularization_detected": generic_graph_regularization,
        "correspondence_attribution_pass": attributed,
        "topology_utility_without_localization": attributed and not local_geometry,
    }


def graph_gate(runs, args: argparse.Namespace) -> dict[str, object]:
    structural = [bool(run["train_match_graph"]["structural_floor_pass"]) for run in runs]
    positive_precision = [
        float(run["postfreeze_gt_audit"]["positive_edges"]["strict_precision"]) for run in runs
    ]
    shuffled_precision = [
        float(run["postfreeze_gt_audit"]["degree_shuffled_edges"]["strict_precision"])
        for run in runs
    ]
    per_seed_precision = [
        positive >= args.min_graph_precision
        and positive >= args.min_precision_ratio_vs_shuffled * shuffled
        for positive, shuffled in zip(positive_precision, shuffled_precision)
    ]
    return {
        "structural_floor_pass_by_seed": structural,
        "positive_strict_precision_by_seed": positive_precision,
        "degree_shuffled_strict_precision_by_seed": shuffled_precision,
        "precision_pass_by_seed": per_seed_precision,
        "all_structural_floors_pass": all(structural),
        "all_precision_gates_pass": all(per_seed_precision),
        "graph_quality_pass": all(structural) and all(per_seed_precision),
        "minimum_positive_precision": args.min_graph_precision,
        "minimum_positive_to_shuffled_factor": args.min_precision_ratio_vs_shuffled,
    }


def stopped_decision(gate: dict[str, object]) -> dict[str, object]:
    if not gate["all_structural_floors_pass"]:
        next_action = "matcher_structural_failure_pivot_plane_normal"
    else:
        next_action = "matcher_precision_failure_pivot_plane_normal"
    return {
        "experiment_stopped_before_position_optimization": True,
        "stop_reason": next_action,
        "families": {},
        "attributed_families": [],
        "generic_graph_regularization_families": [],
        "locally_working_families": [],
        "both_families_attributed": False,
        "stop_position_loss_hyperparameter_sweeps": True,
        "next_action": next_action,
        "production_default_change_authorized": False,
        "train_only_matcher_deployable": False,
    }


def decision(summary, gate: dict[str, object]) -> dict[str, object]:
    graph_quality_pass = bool(gate["graph_quality_pass"])
    families = {family: family_decision(summary, family, graph_quality_pass) for family in FAMILIES}
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
        next_action = "validate_train_only_matcher_on_real_data"
    elif passed:
        next_action = "family_scoped_real_data_matcher_validation"
    elif generic:
        next_action = "generic_graph_regularization_stop_position_branch"
    elif locally_working:
        next_action = "dense_precise_graph_global_utility_falsified_pivot_plane_normal"
    elif any(families[family]["position_engagement_pass"] for family in FAMILIES):
        next_action = "position_localization_failure_pivot_plane_normal"
    else:
        next_action = "position_intervention_non_engaging_pivot_plane_normal"
    return {
        "experiment_stopped_before_position_optimization": False,
        "families": families,
        "attributed_families": passed,
        "generic_graph_regularization_families": generic,
        "locally_working_families": locally_working,
        "both_families_attributed": len(passed) == len(FAMILIES),
        "stop_position_loss_hyperparameter_sweeps": True,
        "next_action": next_action,
        "production_default_change_authorized": False,
        "train_only_matcher_deployable": False,
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

    # Gradient layout construction does not accept a depth backend. Its retained pixel layout is
    # frozen before any corrupted-depth diagnostics are created.
    layout_lifter = make_lifter("gradient", "none", args, seed, [], iterations=0)
    layout_output = layout_lifter.lift(gaussians2d, train_scene)
    layout_core = core_lifter(layout_lifter)
    source_xy = layout_core.source_xy_before_merge
    source_view_ids = layout_core.source_view_ids_before_merge
    source_ranges = layout_core.source_view_ranges_before_merge
    if layout_output.n != source_xy.shape[0]:
        raise AssertionError("layout output and retained source pixels differ")

    graph = freeze_train_graph(
        train_scene.images,
        train_scene.cameras,
        source_xy,
        source_view_ids,
        source_ranges,
        args,
        seed,
    )
    _, extent = train_scene.center_and_extent()
    attach_postfreeze_train_covariates(
        graph,
        train_scene.images,
        train_scene.cameras,
        source_xy,
        source_view_ids,
        source_ranges,
        layout_output.means,
        extent,
        args.match_patch_radius,
        args.position_beta,
    )
    if not graph.diagnostics["structural_floor_pass"]:
        failed = [name for name, passed in graph.diagnostics["floor_checks"].items() if not passed]
        raise AssertionError(
            f"seed {seed} matcher graph misses preregistered structural floors: {failed}"
        )
    audit = audit_frozen_graph_with_gt(
        train_scene,
        source_xy,
        source_view_ids,
        source_ranges,
        graph,
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
        "layout_source_xy_sha256": tensor_collection_hash([("source_xy", source_xy)]),
        "layout_source_view_ids_sha256": tensor_collection_hash(
            [("source_view_ids", source_view_ids)]
        ),
        "train_match_graph": graph.diagnostics,
        "postfreeze_gt_audit": audit.diagnostics,
        "invariants": {},
        "families": {},
    }
    return PreparedSeed(
        seed=seed,
        scene=scene,
        train_scene=train_scene,
        gaussians2d=gaussians2d,
        fit_history=fit_history,
        fit_seconds=fit_seconds,
        layout_output=layout_output,
        layout_core=layout_core,
        graph=graph,
        audit=audit,
        seed_result=seed_result,
    )


def evaluate_prepared_seed(prepared: PreparedSeed, args: argparse.Namespace) -> None:
    seed = prepared.seed
    scene = prepared.scene
    train_scene = prepared.train_scene
    gaussians2d = prepared.gaussians2d
    graph = prepared.graph
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
    prepared.seed_result["corrupted_depths_sha256"] = depth_tensor_hash(corrupted_depths)

    for family in FAMILIES:
        prepared.seed_result["invariants"][family] = assert_family_invariants(
            family,
            gaussians2d,
            train_scene,
            corrupted_depths,
            graph,
            args,
            seed,
        )
        prepared.seed_result["families"][family] = {}
        for mode in MODES:
            prepared.seed_result["families"][family][mode] = evaluate_arm(
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
                prepared.audit,
                args,
                seed,
            )
        family_arms = prepared.seed_result["families"][family]
        counts = {mode: family_arms[mode]["n_initial"] for mode in MODES}
        if len(set(counts.values())) != 1:
            raise AssertionError(f"{family} final primitive counts differ: {counts}")
        schedules = {mode: family_arms[mode]["target_view_history"] for mode in MODES}
        if any(schedule != schedules["none"] for schedule in schedules.values()):
            raise AssertionError(f"{family} target-view schedules differ across modes")
        if args.output is not None and set(schedules["none"]) != set(train_scene.training_views):
            raise AssertionError(f"{family} did not visit every training view")
        for mode in MODES:
            expected_counts = [counts[mode]] * args.lift_iters
            if family_arms[mode]["rendered_count_history"] != expected_counts:
                raise AssertionError(f"{family}/{mode} did not render the inclusive full set")

    reference_invariants = prepared.seed_result["invariants"][FAMILIES[0]]
    for family in FAMILIES[1:]:
        family_invariants = prepared.seed_result["invariants"][family]
        for key in (
            "source_view_ids_sha256",
            "source_xy_sha256",
            "source_view_ranges",
            "positive_pairs_sha256",
            "shuffled_pairs_sha256",
        ):
            if family_invariants[key] != reference_invariants[key]:
                raise AssertionError(f"cross-family retained layout differs in {key}")
    prepared.seed_result["cross_family_layout_and_graph_identical"] = True
    reference_schedule = prepared.seed_result["families"][FAMILIES[0]]["none"][
        "target_view_history"
    ]
    all_counts = set()
    for family in FAMILIES:
        for mode in MODES:
            arm = prepared.seed_result["families"][family][mode]
            if arm["target_view_history"] != reference_schedule:
                raise AssertionError("target-view schedule differs across families")
            all_counts.add(arm["n_initial"])
    if len(all_counts) != 1:
        raise AssertionError("final primitive count differs across families")
    prepared.seed_result["cross_family_schedule_and_count_identical"] = True


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parent.parent
    torch.set_num_threads(args.threads)

    # Phase 1 freezes and audits every seed graph before any position-optimization arm runs.
    prepared = [prepare_seed(args, seed) for seed in args.seeds]
    runs = [item.seed_result for item in prepared]
    gate = graph_gate(runs, args)
    if gate["graph_quality_pass"]:
        for item in prepared:
            evaluate_prepared_seed(item, args)
        summary: dict[str, object] = summarize(runs)
        final_decision = decision(summary, gate)
        for family in FAMILIES:
            for mode in MODES:
                for metric in (
                    "heldout_depth_rmse",
                    "source_all_p90",
                    "source_corrupted_p90",
                    "test_psnr",
                    "alpha_iou",
                    "coverage",
                    "positive_edge_p90",
                    "dominant_gt_p90",
                ):
                    if len(summary[family][mode][metric]["samples"]) != len(runs):
                        raise AssertionError(f"{family}/{mode}/{metric} has incomplete samples")
    else:
        summary = {}
        final_decision = stopped_decision(gate)

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
                "roma_arxiv": "2305.15404",
                "position_only_repo_adaptation": True,
                "train_only_patch_epipolar_topology": True,
                "external_model_or_code_reused": False,
            },
            "protocol_order": [
                "fit_training_images",
                "freeze_retained_layout",
                "build_train_only_graph_once",
                "record_train_only_covariates",
                "audit_frozen_graph_with_synthetic_gt",
                "apply_graph_quality_stopping_gate",
                "run_paired_optimization_arms_if_gate_passes",
            ],
        },
        "graph_gate": gate,
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
    parser.add_argument("--gradient-depth-jitter", type=float, default=0.15)
    parser.add_argument("--gradient-depth-prior-lambda", type=float, default=0.001)
    parser.add_argument("--hybrid-depth-jitter", type=float, default=0.02)
    parser.add_argument("--hybrid-depth-prior-lambda", type=float, default=0.01)
    parser.add_argument("--position-lambda", type=float, default=0.25)
    parser.add_argument("--position-beta", type=float, default=0.05)
    parser.add_argument("--matcher", default="patch_epipolar")
    parser.add_argument("--match-patch-radius", type=int, default=2)
    parser.add_argument("--match-max-epipolar-error-px", type=float, default=2.0)
    parser.add_argument("--match-ratio-threshold", type=float, default=0.50)
    parser.add_argument("--match-ratio-epsilon", type=float, default=1e-6)
    parser.add_argument("--match-min-ray-angle-deg", type=float, default=10.0)
    parser.add_argument("--match-min-depth", type=float, default=0.05)
    parser.add_argument("--match-max-reprojection-error-px", type=float, default=1.5)
    parser.add_argument("--match-min-block-edges", type=int, default=2)
    parser.add_argument("--min-graph-coverage", type=float, default=0.175)
    parser.add_argument("--min-sparse-density-multiplier", type=float, default=1.85)
    parser.add_argument("--min-graph-views", type=int, default=9)
    parser.add_argument("--min-graph-nodes-per-view", type=int, default=16)
    parser.add_argument("--min-graph-edges", type=int, default=160)
    parser.add_argument("--min-graph-nodes", type=int, default=220)
    parser.add_argument("--min-graph-blocks", type=int, default=34)
    parser.add_argument("--audit-min-total-contribution", type=float, default=0.05)
    parser.add_argument("--audit-min-contribution", type=float, default=0.05)
    parser.add_argument("--audit-min-purity", type=float, default=0.50)
    parser.add_argument("--min-graph-precision", type=float, default=0.60)
    parser.add_argument("--min-precision-ratio-vs-shuffled", type=float, default=2.0)
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
        "matcher": "patch_epipolar",
        "match_patch_radius": 2,
        "match_max_epipolar_error_px": 2.0,
        "match_ratio_threshold": 0.50,
        "match_ratio_epsilon": 1e-6,
        "match_min_ray_angle_deg": 10.0,
        "match_min_depth": 0.05,
        "match_max_reprojection_error_px": 1.5,
        "match_min_block_edges": 2,
        "min_graph_coverage": 0.175,
        "min_sparse_density_multiplier": 1.85,
        "min_graph_views": 9,
        "min_graph_nodes_per_view": 16,
        "min_graph_edges": 160,
        "min_graph_nodes": 220,
        "min_graph_blocks": 34,
        "audit_min_total_contribution": 0.05,
        "audit_min_contribution": 0.05,
        "audit_min_purity": 0.50,
        "min_graph_precision": 0.60,
        "min_precision_ratio_vs_shuffled": 2.0,
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
