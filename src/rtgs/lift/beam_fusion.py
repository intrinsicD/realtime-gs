"""Tomographic Gaussian beam fusion: a density-based RGB-free 3D initializer.

The per-view 2D Gaussian fields are *projections* of an unknown 3D radiance density, so 3D
initialization can be posed as reconstruction-from-projections instead of correspondence search.
Gaussians make that tractable in closed form:

1. **Back-projection**: each 2D splat becomes an analytic elongated 3D Gaussian *beam* — its 2D
   covariance lifted to the ray-orthogonal tangent plane at an implied depth, plus a long
   along-ray variance covering the working depth range. No voxel grid.
2. **Pair seeding**: for splats in two views, the closest points between their center rays give
   implied depths in closed form; pairs are gated by transverse ray distance (measured against
   both beams' tangent footprints) and color agreement. The gate is the first-order Gaussian
   product weight, so negligible cross terms are never materialized.
3. **Fusion by covariance intersection**: surviving beams are fused with the CI rule
   ``Lambda = mean_k Lambda_k``, ``m = Lambda^{-1} mean_k(Lambda_k m_k)``. A naive Gaussian
   product (``Lambda = sum_k Lambda_k``) is *rejected by design*: the views are correlated
   observations of one physical splat, and the product double-counts every axis observed by more
   than one view (two orthogonal views of an isotropic blob would fuse to half its variance).
   CI is the standard consistent rule for correlated sources: exact on directions all views
   share, conservative (never overconfident) elsewhere.
4. **Greedy fold-in**: each fused component projects into the remaining views and absorbs the
   nearest splat within a pixel gate and color gate (at most one per view); contributors are the
   correspondence byproduct.
5. **Reduction**: exact contributor-signature dedupe, then per-voxel non-maximum suppression by
   weight — selection, not moment matching, so fused covariances survive.

Every output Gaussian is a positive-weight CI fusion of at least ``min_views`` beams with bounded
SPD covariance and complete contributor lineage; unmatched splats are reported per view for
densification.  Association emerges from density overlap rather than discrete matching — this is
deliberately a different family from both the consensus scoring (which drifts) and the SfM-style
discrete matcher.  CPU-first, deterministic, no RGB.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.compact_carve import _center_and_extent, _ray_box, _validate_cpu_inputs
from rtgs.lift.inverse_projection_fiber import _camera_ray_geometry


@dataclass(frozen=True)
class BeamFusionConfig:
    """Controls for beam back-projection, pair seeding, CI fusion, and reduction."""

    min_views: int = 2
    near: float = 0.05
    bounds_scale: float = 0.5
    transverse_gate_sigma: float = 3.0
    max_color_distance: float = 0.35
    color_sigma: float = 0.25
    fold_in_gate_sigma: float = 3.0
    weight_floor: float = 1e-4
    nms_voxel_size: float | None = None
    min_sigma_world: float = 1e-4
    max_sigma_world: float | None = None
    init_opacity: float = 0.1
    source_chunk: int = 1024
    pair_limit: int | None = None
    max_components: int | None = None
    seed_budget_multiplier: int = 4
    fold_in_chunk: int = 512
    max_seed_voxels: int = 2_000_000

    def __post_init__(self) -> None:
        if isinstance(self.min_views, bool) or not isinstance(self.min_views, int):
            raise TypeError("min_views must be an integer")
        if self.min_views < 2:
            raise ValueError("min_views must be at least two")
        for name in (
            "source_chunk",
            "seed_budget_multiplier",
            "fold_in_chunk",
            "max_seed_voxels",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.pair_limit is not None and (
            isinstance(self.pair_limit, bool)
            or not isinstance(self.pair_limit, int)
            or self.pair_limit <= 0
        ):
            raise ValueError("pair_limit must be a positive integer or None")
        if self.max_components is not None and (
            isinstance(self.max_components, bool)
            or not isinstance(self.max_components, int)
            or self.max_components <= 0
        ):
            raise ValueError("max_components must be a positive integer or None")
        for name in (
            "near",
            "bounds_scale",
            "transverse_gate_sigma",
            "max_color_distance",
            "color_sigma",
            "fold_in_gate_sigma",
            "min_sigma_world",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_floor) or self.weight_floor < 0:
            raise ValueError("weight_floor must be finite and non-negative")
        if self.nms_voxel_size is not None and (
            not math.isfinite(self.nms_voxel_size) or self.nms_voxel_size <= 0
        ):
            raise ValueError("nms_voxel_size must be finite and positive when supplied")
        if self.max_sigma_world is not None and (
            not math.isfinite(self.max_sigma_world) or self.max_sigma_world <= 0
        ):
            raise ValueError("max_sigma_world must be finite and positive when supplied")
        if not 0.0 < self.init_opacity < 1.0:
            raise ValueError("init_opacity must be in (0,1)")


@dataclass(frozen=True)
class BeamFusionResult:
    """CI-fused 3D initialization with complete contributor provenance."""

    gaussians: Gaussians3D
    component_offsets: torch.Tensor  # (C+1,), int64 CSR offsets into contributor arrays
    contributor_view_indices: torch.Tensor  # (M,), int64
    contributor_component_indices: torch.Tensor  # (M,), int64
    contributor_depths: torch.Tensor  # (M,), float64 implied camera-space beam depths
    component_weights: torch.Tensor  # (C,)
    unmatched_per_view: tuple[int, ...]
    diagnostics: dict[str, object]

    @property
    def n_components(self) -> int:
        return int(self.component_offsets.numel() - 1)


@dataclass(frozen=True)
class BeamFusionProgress:
    """Silent-by-default progress record for the bounded production path."""

    stage: str
    completed: int
    total: int
    evaluated_ray_pairs: int
    gated_seeds: int
    retained_seed_voxels: int
    elapsed_seconds: float


@dataclass(frozen=True)
class _BeamView:
    """Per-view float64 geometry shared by seeding, lifting, and fold-in."""

    means2d: torch.Tensor  # (N,2)
    covariances2d: torch.Tensor  # (N,2,2)
    colors: torch.Tensor  # (N,3)
    amplitudes: torch.Tensor  # (N,)
    sigmas_px: torch.Tensor  # (N,)
    origin: torch.Tensor  # (3,) world
    dirs_world: torch.Tensor  # (N,3), unit camera depth
    rays_cam: torch.Tensor  # (N,3), unit camera depth (camera frame)
    basis_cam: torch.Tensor  # (N,3,3), orthonormal camera-frame basis, column 2 = ray unit
    depth_lo: torch.Tensor  # (N,)
    depth_hi: torch.Tensor  # (N,)


def _component_covariances_2d(field: GaussianObservationField) -> torch.Tensor:
    variances = field.effective_variances().to(torch.float64)
    theta = field.rotations.to(torch.float64)
    cos, sin = theta.cos(), theta.sin()
    off = cos * sin * (variances[:, 0] - variances[:, 1])
    row0 = torch.stack(
        [cos.square() * variances[:, 0] + sin.square() * variances[:, 1], off], dim=-1
    )
    row1 = torch.stack(
        [off, sin.square() * variances[:, 0] + cos.square() * variances[:, 1]], dim=-1
    )
    return torch.stack([row0, row1], dim=-2)


def _prepare_views(inputs: ReconstructionInputs, config: BeamFusionConfig) -> list[_BeamView]:
    center, extent = _center_and_extent(inputs, torch.float64)
    half = extent * config.bounds_scale
    lower, upper = center - half, center + half
    views: list[_BeamView] = []
    for field, camera in zip(inputs.observations, inputs.cameras, strict=True):
        means2d = field.native_means(dtype=torch.float64)
        covariances = _component_covariances_2d(field)
        origin, dirs_world = camera.pixel_rays(means2d)
        rays_cam, basis_cam = _camera_ray_geometry(camera, means2d)
        t0, t1 = _ray_box(origin.expand_as(dirs_world), dirs_world, lower, upper)
        t0 = t0.clamp_min(config.near)
        views.append(
            _BeamView(
                means2d=means2d,
                covariances2d=covariances,
                colors=field.colors.to(torch.float64),
                amplitudes=field.amplitudes.to(torch.float64),
                sigmas_px=(0.5 * (covariances[:, 0, 0] + covariances[:, 1, 1])).sqrt(),
                origin=origin.to(torch.float64),
                dirs_world=dirs_world.to(torch.float64),
                rays_cam=rays_cam.to(torch.float64),
                basis_cam=basis_cam.to(torch.float64),
                depth_lo=t0,
                depth_hi=torch.maximum(t1, t0),
            )
        )
    return views


def _beam_precisions(
    view: _BeamView,
    camera,
    indices: torch.Tensor,
    depths: torch.Tensor,
    ray_half_length: torch.Tensor,
) -> torch.Tensor:
    """World-frame beam precision matrices for ``indices`` at implied camera depths.

    The 2D covariance is lifted to the ray-orthogonal tangent plane through the inverse of the
    projected tangent Jacobian (the fiber's exact construction) and inverted; the along-ray
    precision is ``1 / L^2`` for beam half-length ``L``.  The camera-frame quadratic form pulls
    back to world through the world-to-camera rotation.
    """
    rays = view.rays_cam[indices]
    basis = view.basis_cam[indices]
    points_cam = depths[:, None] * rays
    z = points_cam[:, 2].clamp_min(1e-8)
    jacobian = torch.zeros(indices.numel(), 2, 3, dtype=torch.float64)
    jacobian[:, 0, 0] = camera.fx / z
    jacobian[:, 0, 2] = -camera.fx * points_cam[:, 0] / z.square()
    jacobian[:, 1, 1] = camera.fy / z
    jacobian[:, 1, 2] = -camera.fy * points_cam[:, 1] / z.square()
    tangent_projection = jacobian @ basis[:, :, :2]
    inverse_projection = torch.linalg.inv(tangent_projection)
    tangent_cov = (
        inverse_projection @ view.covariances2d[indices] @ inverse_projection.transpose(-1, -2)
    )
    tangent_cov = 0.5 * (tangent_cov + tangent_cov.transpose(-1, -2))
    tangent_precision = torch.linalg.inv(tangent_cov)
    block = torch.zeros(indices.numel(), 3, 3, dtype=torch.float64)
    block[:, :2, :2] = tangent_precision
    block[:, 2, 2] = 1.0 / ray_half_length.square().clamp_min(1e-12)
    precision_cam = basis @ block @ basis.transpose(-1, -2)
    rotation = camera.R.to(torch.float64)
    return rotation.transpose(-1, -2) @ precision_cam @ rotation


def _ray_closest_points(
    o_u: torch.Tensor,
    d1: torch.Tensor,
    o_v: torch.Tensor,
    d2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form per-pair depths ``(s, t)`` and midpoint distance for two ray families."""
    w = o_v - o_u
    a = (d1 * d1).sum(dim=1)
    c = (d2 * d2).sum(dim=1)
    b = d1 @ d2.T
    e = d1 @ w
    f = d2 @ w
    denom = a[:, None] * c[None, :] - b.square()
    parallel = denom.abs() <= 1e-12 * a[:, None] * c[None, :]
    safe = torch.where(parallel, torch.ones_like(denom), denom)
    s = (c[None, :] * e[:, None] - b * f[None, :]) / safe
    t = (b * e[:, None] - a[:, None] * f[None, :]) / safe
    points_u = o_u[None, None, :] + s[..., None] * d1[:, None, :]
    points_v = o_v[None, None, :] + t[..., None] * d2[None, :, :]
    distance = (points_u - points_v).norm(dim=-1)
    distance = torch.where(parallel, torch.full_like(distance, torch.inf), distance)
    return s, t, distance


def _ci_fuse(precisions: torch.Tensor, means: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Equal-weight covariance intersection of ``(K,3,3)`` precisions and ``(K,3)`` means."""
    fused_precision = precisions.mean(dim=0)
    information = (precisions @ means[:, :, None]).mean(dim=0)
    fused_mean = torch.linalg.solve(fused_precision, information)[:, 0]
    return fused_precision, fused_mean


def _bounded_seed_candidates(
    inputs: ReconstructionInputs,
    views: list[_BeamView],
    config: BeamFusionConfig,
    pairs: list[tuple[int, int]],
    nms_voxel: float,
    started: float,
    progress_callback: Callable[[BeamFusionProgress], None] | None,
) -> tuple[dict[str, torch.Tensor], dict[str, int | list[int]]]:
    """Stream every ray pair while retaining only the strongest seed in each 3D voxel."""
    center, extent = _center_and_extent(inputs, torch.float64)
    half = float(extent) * config.bounds_scale
    lower = center - half
    cells_per_axis = (
        torch.ceil(torch.full((3,), 2.0 * half / nms_voxel, dtype=torch.float64))
        .to(torch.long)
        .clamp_min(1)
    )
    grid_shape = [int(value) for value in cells_per_axis.tolist()]
    n_cells = math.prod(grid_shape)
    if n_cells > config.max_seed_voxels:
        raise ValueError(
            "beam fusion seed voxel grid exceeds max_seed_voxels: "
            f"{n_cells} > {config.max_seed_voxels}; increase nms_voxel_size or the explicit cap"
        )

    sentinel = torch.iinfo(torch.long).max
    best_weight = torch.full((n_cells,), -torch.inf, dtype=torch.float64)
    best_rank = torch.full((n_cells,), sentinel, dtype=torch.long)
    best_u = torch.full((n_cells,), -1, dtype=torch.long)
    best_i = torch.full((n_cells,), -1, dtype=torch.long)
    best_v = torch.full((n_cells,), -1, dtype=torch.long)
    best_j = torch.full((n_cells,), -1, dtype=torch.long)
    best_s = torch.zeros(n_cells, dtype=torch.float64)
    best_t = torch.zeros(n_cells, dtype=torch.float64)

    n_views = inputs.n_views
    component_stride = max(view.means2d.shape[0] for view in views) + 1
    evaluated_ray_pairs = 0
    gated_seeds = 0
    retained_seed_voxels = 0
    peak_pair_matrix = 0
    nx, ny, nz = grid_shape

    for pair_index, (u, v) in enumerate(pairs):
        view_u, view_v = views[u], views[v]
        n_u, n_v = view_u.means2d.shape[0], view_v.means2d.shape[0]
        if n_u == 0 or n_v == 0:
            continue
        focal_u = math.sqrt(inputs.cameras[u].fx * inputs.cameras[u].fy)
        focal_v = math.sqrt(inputs.cameras[v].fx * inputs.cameras[v].fy)
        for start in range(0, n_u, config.source_chunk):
            stop = min(start + config.source_chunk, n_u)
            pair_matrix = (stop - start) * n_v
            evaluated_ray_pairs += pair_matrix
            peak_pair_matrix = max(peak_pair_matrix, pair_matrix)
            s, t, distance = _ray_closest_points(
                view_u.origin,
                view_u.dirs_world[start:stop],
                view_v.origin,
                view_v.dirs_world,
            )
            sigma_u = view_u.sigmas_px[start:stop, None] * s.clamp_min(1e-8) / focal_u
            sigma_v = view_v.sigmas_px[None, :] * t.clamp_min(1e-8) / focal_v
            gate = distance.square() / (sigma_u.square() + sigma_v.square()).clamp_min(1e-18)
            color_distance = torch.cdist(view_u.colors[start:stop], view_v.colors)
            valid = (
                (s >= view_u.depth_lo[start:stop, None])
                & (s <= view_u.depth_hi[start:stop, None])
                & (t >= view_v.depth_lo[None, :])
                & (t <= view_v.depth_hi[None, :])
                & (gate <= config.transverse_gate_sigma**2)
                & (color_distance <= config.max_color_distance)
            )
            local_i, j = valid.nonzero(as_tuple=True)
            if local_i.numel() == 0:
                continue
            gated_seeds += int(local_i.numel())
            i = local_i + start
            s_value = s[local_i, j]
            t_value = t[local_i, j]
            point_u = view_u.origin + s_value[:, None] * view_u.dirs_world[i]
            point_v = view_v.origin + t_value[:, None] * view_v.dirs_world[j]
            provisional = 0.5 * (point_u + point_v)
            cell = torch.floor((provisional - lower) / nms_voxel).to(torch.long)
            inside = (
                (cell[:, 0] >= 0)
                & (cell[:, 0] < nx)
                & (cell[:, 1] >= 0)
                & (cell[:, 1] < ny)
                & (cell[:, 2] >= 0)
                & (cell[:, 2] < nz)
            )
            if not bool(inside.all()):
                i = i[inside]
                j = j[inside]
                local_i = local_i[inside]
                s_value = s_value[inside]
                t_value = t_value[inside]
                cell = cell[inside]
                if i.numel() == 0:
                    continue
            keys = cell[:, 0] + nx * (cell[:, 1] + ny * cell[:, 2])
            selected_gate = gate[local_i, j]
            selected_color_distance = color_distance[local_i, j]
            weights = (
                0.5
                * (view_u.amplitudes[i] + view_v.amplitudes[j])
                * torch.exp(-0.5 * selected_gate)
                * torch.exp(-0.5 * selected_color_distance.square() / config.color_sigma**2)
            )
            ranks = ((u * component_stride + i) * n_views + v) * component_stride + j

            unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
            local_weight = torch.full((unique_keys.numel(),), -torch.inf, dtype=torch.float64)
            local_weight.scatter_reduce_(0, inverse, weights, reduce="amax", include_self=True)
            weight_winner = weights == local_weight[inverse]
            rank_candidates = torch.where(weight_winner, ranks, torch.full_like(ranks, sentinel))
            local_rank = torch.full((unique_keys.numel(),), sentinel, dtype=torch.long)
            local_rank.scatter_reduce_(
                0, inverse, rank_candidates, reduce="amin", include_self=True
            )
            winner_rows = (weight_winner & (ranks == local_rank[inverse])).nonzero(as_tuple=True)[0]
            row_by_group = torch.full((unique_keys.numel(),), -1, dtype=torch.long)
            row_by_group[inverse[winner_rows]] = winner_rows
            if bool((row_by_group < 0).any()):  # pragma: no cover - reduction invariant
                raise RuntimeError("beam fusion voxel reduction lost a local winner")

            old_weight = best_weight[unique_keys]
            old_rank = best_rank[unique_keys]
            replace = (local_weight > old_weight) | (
                (local_weight == old_weight) & (local_rank < old_rank)
            )
            groups = replace.nonzero(as_tuple=True)[0]
            if groups.numel() == 0:
                continue
            update_keys = unique_keys[groups]
            rows = row_by_group[groups]
            retained_seed_voxels += int((old_rank[groups] == sentinel).sum())
            best_weight[update_keys] = weights[rows]
            best_rank[update_keys] = ranks[rows]
            best_u[update_keys] = u
            best_i[update_keys] = i[rows]
            best_v[update_keys] = v
            best_j[update_keys] = j[rows]
            best_s[update_keys] = s_value[rows]
            best_t[update_keys] = t_value[rows]

        if progress_callback is not None:
            progress_callback(
                BeamFusionProgress(
                    stage="pair_seeding",
                    completed=pair_index + 1,
                    total=len(pairs),
                    evaluated_ray_pairs=evaluated_ray_pairs,
                    gated_seeds=gated_seeds,
                    retained_seed_voxels=retained_seed_voxels,
                    elapsed_seconds=time.perf_counter() - started,
                )
            )

    occupied = (best_rank != sentinel).nonzero(as_tuple=True)[0]
    if occupied.numel() == 0:
        raise ValueError(
            "beam fusion produced no gated pair intersections; loosen the transverse/color "
            "gates or supply views with more overlap"
        )
    # Stable lexicographic selection: weight descending, then (u,i,v,j) rank ascending.
    rank_order = torch.argsort(best_rank[occupied], stable=True)
    occupied = occupied[rank_order]
    weight_order = torch.argsort(best_weight[occupied], descending=True, stable=True)
    occupied = occupied[weight_order]
    assert config.max_components is not None
    seed_budget = min(int(occupied.numel()), config.max_components * config.seed_budget_multiplier)
    occupied = occupied[:seed_budget]
    seeds = {
        "u": best_u[occupied],
        "i": best_i[occupied],
        "v": best_v[occupied],
        "j": best_j[occupied],
        "s": best_s[occupied],
        "t": best_t[occupied],
        "weight": best_weight[occupied],
    }
    diagnostics: dict[str, int | list[int]] = {
        "evaluated_ray_pairs": evaluated_ray_pairs,
        "gated_seeds": gated_seeds,
        "retained_seed_voxels": retained_seed_voxels,
        "retained_seed_candidates": seed_budget,
        "peak_pair_matrix": peak_pair_matrix,
        "seed_grid_shape": grid_shape,
        "seed_grid_cells": n_cells,
    }
    return seeds, diagnostics


def _fuse_gaussian_beams_bounded(
    inputs: ReconstructionInputs,
    views: list[_BeamView],
    config: BeamFusionConfig,
    pairs: list[tuple[int, int]],
    nms_voxel: float,
    max_sigma: float,
    started: float,
    progress_callback: Callable[[BeamFusionProgress], None] | None,
) -> BeamFusionResult:
    """Production-sized beam fusion with bounded seed storage and vectorized fold-in."""
    seeding_started = time.perf_counter()
    seeds, seed_diagnostics = _bounded_seed_candidates(
        inputs,
        views,
        config,
        pairs,
        nms_voxel,
        started,
        progress_callback,
    )
    seeding_seconds = time.perf_counter() - seeding_started
    n_seeds = int(seeds["u"].numel())
    n_views = inputs.n_views
    rows = torch.arange(n_seeds, dtype=torch.long)
    point_u = views[0].origin.new_zeros((n_seeds, 3))
    point_v = views[0].origin.new_zeros((n_seeds, 3))
    source_colors = views[0].colors.new_zeros((n_seeds, 3))
    for view_index, view in enumerate(views):
        source_u = seeds["u"] == view_index
        if bool(source_u.any()):
            indices = seeds["i"][source_u]
            point_u[source_u] = view.origin + seeds["s"][source_u, None] * view.dirs_world[indices]
            source_colors[source_u] = view.colors[indices]
        source_v = seeds["v"] == view_index
        if bool(source_v.any()):
            indices = seeds["j"][source_v]
            point_v[source_v] = view.origin + seeds["t"][source_v, None] * view.dirs_world[indices]
    provisional = 0.5 * (point_u + point_v)

    contributors = torch.full((n_seeds, n_views), -1, dtype=torch.long)
    depths = torch.full((n_seeds, n_views), torch.nan, dtype=torch.float64)
    contributors[rows, seeds["u"]] = seeds["i"]
    contributors[rows, seeds["v"]] = seeds["j"]
    depths[rows, seeds["u"]] = seeds["s"]
    depths[rows, seeds["v"]] = seeds["t"]

    fold_started = time.perf_counter()
    for view_index, (view, camera) in enumerate(zip(views, inputs.cameras, strict=True)):
        eligible = ((seeds["u"] != view_index) & (seeds["v"] != view_index)).nonzero(as_tuple=True)[
            0
        ]
        for start in range(0, eligible.numel(), config.fold_in_chunk):
            selected_rows = eligible[start : start + config.fold_in_chunk]
            uv, depth = camera.project(provisional[selected_rows])
            distances = torch.cdist(uv.to(torch.float64), view.means2d)
            nearest_distance, nearest = distances.min(dim=1)
            color_gap = (view.colors[nearest] - source_colors[selected_rows]).norm(dim=1)
            accepted = (
                (depth.to(torch.float64) > config.near)
                & (nearest_distance <= config.fold_in_gate_sigma * view.sigmas_px[nearest])
                & (color_gap <= config.max_color_distance)
            )
            accepted_rows = selected_rows[accepted]
            contributors[accepted_rows, view_index] = nearest[accepted]
            depths[accepted_rows, view_index] = depth[accepted].to(torch.float64)
        if progress_callback is not None:
            progress_callback(
                BeamFusionProgress(
                    stage="fold_in",
                    completed=view_index + 1,
                    total=n_views,
                    evaluated_ray_pairs=int(seed_diagnostics["evaluated_ray_pairs"]),
                    gated_seeds=int(seed_diagnostics["gated_seeds"]),
                    retained_seed_voxels=int(seed_diagnostics["retained_seed_voxels"]),
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
    fold_in_seconds = time.perf_counter() - fold_started

    lengths = (contributors >= 0).sum(dim=1)
    eligible = lengths >= config.min_views
    if not bool(eligible.any()):
        raise ValueError(
            "beam fusion rejected every bounded seed at the min_views fold-in gate; loosen "
            "the gates or lower min_views"
        )
    contributors = contributors[eligible]
    depths = depths[eligible]
    seed_weights = seeds["weight"][eligible]
    lengths = lengths[eligible]
    n_components_prefilter = int(contributors.shape[0])

    fusion_started = time.perf_counter()
    precision_sum = torch.zeros(n_components_prefilter, 3, 3, dtype=torch.float64)
    information_sum = torch.zeros(n_components_prefilter, 3, dtype=torch.float64)
    for view_index, (view, camera) in enumerate(zip(views, inputs.cameras, strict=True)):
        component_rows = (contributors[:, view_index] >= 0).nonzero(as_tuple=True)[0]
        if component_rows.numel() == 0:
            continue
        indices = contributors[component_rows, view_index]
        view_depths = depths[component_rows, view_index]
        half_length = (
            0.5
            * (view.depth_hi[indices] - view.depth_lo[indices]).clamp_min(1e-6)
            * view.dirs_world[indices].norm(dim=1)
        )
        precisions = _beam_precisions(
            view,
            camera,
            indices,
            view_depths,
            half_length,
        )
        means = view.origin + view_depths[:, None] * view.dirs_world[indices]
        precision_sum.index_add_(0, component_rows, precisions)
        information_sum.index_add_(0, component_rows, (precisions @ means[:, :, None])[:, :, 0])
    divisor = lengths.to(torch.float64)[:, None, None]
    fused_precision = precision_sum / divisor
    fused_information = information_sum / lengths.to(torch.float64)[:, None]
    fused_mean = torch.linalg.solve(fused_precision, fused_information[:, :, None])[:, :, 0]
    component_weights = seed_weights * lengths.to(torch.float64)

    components: list[dict] = []
    for row in range(n_components_prefilter):
        signature = tuple(
            (view_index, int(contributors[row, view_index]))
            for view_index in range(n_views)
            if int(contributors[row, view_index]) >= 0
        )
        components.append(
            {
                "signature": signature,
                "depths": tuple(
                    float(depths[row, view_index])
                    for view_index in range(n_views)
                    if int(contributors[row, view_index]) >= 0
                ),
                "weight": float(component_weights[row]),
                "precision": fused_precision[row],
                "mean": fused_mean[row],
            }
        )

    by_signature: dict[tuple, dict] = {}
    for component in components:
        existing = by_signature.get(component["signature"])
        if existing is None or component["weight"] > existing["weight"]:
            by_signature[component["signature"]] = component
    unique = sorted(by_signature.values(), key=lambda item: (-item["weight"], item["signature"]))
    kept: list[dict] = []
    occupied: set[tuple[int, int, int]] = set()
    assert config.max_components is not None
    for component in unique:
        if component["weight"] < config.weight_floor:
            continue
        key = tuple(int(value) for value in torch.floor(component["mean"] / nms_voxel).tolist())
        if key in occupied:
            continue
        occupied.add(key)
        kept.append(component)
        if len(kept) == config.max_components:
            break
    if not kept:
        raise ValueError("beam fusion NMS removed every component; lower weight_floor")

    means = torch.stack([component["mean"] for component in kept])
    covariances = torch.linalg.inv(torch.stack([component["precision"] for component in kept]))
    covariances = 0.5 * (covariances + covariances.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariances)
    eigenvalues = eigenvalues.clamp(min=config.min_sigma_world**2, max=max_sigma**2)
    covariances = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)

    colors = []
    offsets = [0]
    contributor_views: list[int] = []
    contributor_components: list[int] = []
    contributor_depths: list[float] = []
    matched: dict[int, set[int]] = {view: set() for view in range(n_views)}
    for component in kept:
        color_sum = torch.zeros(3, dtype=torch.float64)
        amplitude_sum = 0.0
        for (view_index, splat_index), depth in zip(
            component["signature"], component["depths"], strict=True
        ):
            amplitude = float(views[view_index].amplitudes[splat_index])
            color_sum += amplitude * views[view_index].colors[splat_index]
            amplitude_sum += amplitude
            contributor_views.append(view_index)
            contributor_components.append(splat_index)
            contributor_depths.append(depth)
            matched[view_index].add(splat_index)
        offsets.append(len(contributor_views))
        colors.append(color_sum / max(amplitude_sum, 1e-12))
    unmatched = tuple(
        int(inputs.observations[view].n - len(matched[view])) for view in range(n_views)
    )

    dtype = inputs.observations[0].dtype
    gaussians = Gaussians3D.from_means_covs(
        means=means.to(dtype),
        covs=covariances.to(dtype),
        colors=torch.stack(colors).clamp(0.0, 1.0).to(dtype),
        opacity=torch.full((len(kept),), config.init_opacity, dtype=dtype),
        sh_degree=0,
    )
    kept_lengths = torch.tensor(
        [len(component["signature"]) for component in kept], dtype=torch.long
    )
    diagnostics: dict[str, object] = {
        "bounded_seed_mode": True,
        "n_views": n_views,
        "n_view_pairs": len(pairs),
        "n_seeds": int(seed_diagnostics["gated_seeds"]),
        "n_seed_voxels": int(seed_diagnostics["retained_seed_voxels"]),
        "n_seed_candidates_retained": n_seeds,
        "n_components_prefilter": n_components_prefilter,
        "n_components": len(kept),
        "component_view_histogram": {
            int(length): int((kept_lengths == length).sum())
            for length in kept_lengths.unique(sorted=True)
        },
        "nms_voxel_size": nms_voxel,
        "max_sigma_world": max_sigma,
        "unmatched_per_view": list(unmatched),
        "evaluated_ray_pairs": int(seed_diagnostics["evaluated_ray_pairs"]),
        "peak_pair_matrix": int(seed_diagnostics["peak_pair_matrix"]),
        "seed_grid_shape": seed_diagnostics["seed_grid_shape"],
        "seed_grid_cells": int(seed_diagnostics["seed_grid_cells"]),
        "seed_budget": config.max_components * config.seed_budget_multiplier,
        "timings_seconds": {
            "pair_seeding": seeding_seconds,
            "fold_in": fold_in_seconds,
            "fusion_and_reduction": time.perf_counter() - fusion_started,
            "total": time.perf_counter() - started,
        },
    }
    if progress_callback is not None:
        progress_callback(
            BeamFusionProgress(
                stage="complete",
                completed=1,
                total=1,
                evaluated_ray_pairs=int(seed_diagnostics["evaluated_ray_pairs"]),
                gated_seeds=int(seed_diagnostics["gated_seeds"]),
                retained_seed_voxels=int(seed_diagnostics["retained_seed_voxels"]),
                elapsed_seconds=time.perf_counter() - started,
            )
        )
    return BeamFusionResult(
        gaussians=gaussians,
        component_offsets=torch.tensor(offsets, dtype=torch.long),
        contributor_view_indices=torch.tensor(contributor_views, dtype=torch.long),
        contributor_component_indices=torch.tensor(contributor_components, dtype=torch.long),
        contributor_depths=torch.tensor(contributor_depths, dtype=torch.float64),
        component_weights=torch.tensor(
            [component["weight"] for component in kept], dtype=torch.float64
        ),
        unmatched_per_view=unmatched,
        diagnostics=diagnostics,
    )


def fuse_gaussian_beams(
    inputs: ReconstructionInputs,
    config: BeamFusionConfig | None = None,
    *,
    progress_callback: Callable[[BeamFusionProgress], None] | None = None,
) -> BeamFusionResult:
    """Run tomographic beam fusion and return the CI-fused 3D initialization."""
    if config is None:
        config = BeamFusionConfig()
    _validate_cpu_inputs(inputs)
    if config.min_views > inputs.n_views:
        raise ValueError("beam fusion min_views exceeds the number of views")

    views = _prepare_views(inputs, config)
    n_views = inputs.n_views
    _, extent = _center_and_extent(inputs, torch.float64)
    max_sigma = (
        config.max_sigma_world if config.max_sigma_world is not None else 0.5 * float(extent)
    )
    nms_voxel = (
        config.nms_voxel_size
        if config.nms_voxel_size is not None
        else max(1e-6, 0.01 * float(extent))
    )

    pairs = [(u, v) for u in range(n_views) for v in range(u + 1, n_views)]
    positions = torch.stack([camera.position for camera in inputs.cameras]).to(torch.float64)
    pairs.sort(key=lambda pair: (float((positions[pair[0]] - positions[pair[1]]).norm()), pair))
    if config.pair_limit is not None:
        pairs = pairs[: config.pair_limit]

    if config.max_components is not None:
        return _fuse_gaussian_beams_bounded(
            inputs,
            views,
            config,
            pairs,
            nms_voxel,
            max_sigma,
            time.perf_counter(),
            progress_callback,
        )

    # Stage 1-2: seed candidate components from gated pair intersections.
    seeds: list[tuple[int, int, int, int, float, float, float]] = []  # u,i,v,j,s,t,gate_weight
    for u, v in pairs:
        view_u, view_v = views[u], views[v]
        n_u, n_v = view_u.means2d.shape[0], view_v.means2d.shape[0]
        if n_u == 0 or n_v == 0:
            continue
        for start in range(0, n_u, config.source_chunk):
            stop = min(start + config.source_chunk, n_u)
            s, t, distance = _ray_closest_points(
                view_u.origin,
                view_u.dirs_world[start:stop],
                view_v.origin,
                view_v.dirs_world,
            )
            # First-order transverse footprints at the implied depths (pixels -> world).
            focal_u = math.sqrt(inputs.cameras[u].fx * inputs.cameras[u].fy)
            focal_v = math.sqrt(inputs.cameras[v].fx * inputs.cameras[v].fy)
            sigma_u = view_u.sigmas_px[start:stop, None] * s.clamp_min(1e-8) / focal_u
            sigma_v = view_v.sigmas_px[None, :] * t.clamp_min(1e-8) / focal_v
            gate = distance.square() / (sigma_u.square() + sigma_v.square()).clamp_min(1e-18)
            color_distance = torch.cdist(view_u.colors[start:stop], view_v.colors)
            valid = (
                (s >= view_u.depth_lo[start:stop, None])
                & (s <= view_u.depth_hi[start:stop, None])
                & (t >= view_v.depth_lo[None, :])
                & (t <= view_v.depth_hi[None, :])
                & (gate <= config.transverse_gate_sigma**2)
                & (color_distance <= config.max_color_distance)
            )
            for local_i, j in zip(*valid.nonzero(as_tuple=True), strict=True):
                i = int(local_i) + start
                weight = float(
                    0.5
                    * (view_u.amplitudes[i] + view_v.amplitudes[int(j)])
                    * math.exp(-0.5 * float(gate[local_i, j]))
                    * math.exp(
                        -0.5 * float(color_distance[local_i, j]) ** 2 / config.color_sigma**2
                    )
                )
                seeds.append((u, i, v, int(j), float(s[local_i, j]), float(t[local_i, j]), weight))
    if not seeds:
        raise ValueError(
            "beam fusion produced no gated pair intersections; loosen the transverse/color "
            "gates or supply views with more overlap"
        )
    seeds.sort(key=lambda seed: (-seed[6], seed[0], seed[1], seed[2], seed[3]))

    # Stage 3-4: CI-fuse each seed pair, then greedily fold in the remaining views.
    components: list[dict] = []
    for u, i, v, j, s, t, gate_weight in seeds:
        contributors = {u: (i, s), v: (j, t)}
        for w in range(n_views):
            if w in contributors:
                continue
            view_w = views[w]
            # Provisional midpoint from the seed pair locates the projection into view w.
            provisional = 0.5 * (
                views[u].origin
                + s * views[u].dirs_world[i]
                + views[v].origin
                + t * views[v].dirs_world[j]
            )
            uv, depth = inputs.cameras[w].project(provisional[None, :])
            if float(depth[0]) <= config.near:
                continue
            offsets = (view_w.means2d - uv[0]).norm(dim=1)
            candidate = int(offsets.argmin())
            pixel_gate = config.fold_in_gate_sigma * float(view_w.sigmas_px[candidate])
            color_gap = float((view_w.colors[candidate] - views[u].colors[i]).norm())
            if float(offsets[candidate]) <= pixel_gate and color_gap <= config.max_color_distance:
                contributors[w] = (candidate, float(depth[0]))
        if len(contributors) < config.min_views:
            continue
        member_views = sorted(contributors)
        precisions = []
        means = []
        for view_index in member_views:
            component, depth = contributors[view_index]
            view = views[view_index]
            index_tensor = torch.tensor([component], dtype=torch.long)
            depth_tensor = torch.tensor([depth], dtype=torch.float64)
            half_length = (
                0.5
                * (view.depth_hi[component] - view.depth_lo[component]).clamp_min(1e-6)
                * view.dirs_world[component].norm()
            )
            precisions.append(
                _beam_precisions(
                    view,
                    inputs.cameras[view_index],
                    index_tensor,
                    depth_tensor,
                    half_length[None],
                )[0]
            )
            means.append(view.origin + depth * view.dirs_world[component])
        fused_precision, fused_mean = _ci_fuse(torch.stack(precisions), torch.stack(means))
        signature = tuple((view_index, contributors[view_index][0]) for view_index in member_views)
        components.append(
            {
                "signature": signature,
                "depths": tuple(contributors[view_index][1] for view_index in member_views),
                "weight": gate_weight * len(member_views),
                "precision": fused_precision,
                "mean": fused_mean,
            }
        )
    if not components:
        raise ValueError(
            "beam fusion rejected every seed at the min_views fold-in gate; loosen the gates "
            "or lower min_views"
        )

    # Stage 5: exact signature dedupe, then per-voxel non-maximum suppression by weight.
    by_signature: dict[tuple, dict] = {}
    for component in components:
        existing = by_signature.get(component["signature"])
        if existing is None or component["weight"] > existing["weight"]:
            by_signature[component["signature"]] = component
    unique = sorted(by_signature.values(), key=lambda item: (-item["weight"], item["signature"]))
    kept: list[dict] = []
    occupied: dict[tuple[int, int, int], float] = {}
    for component in unique:
        if component["weight"] < config.weight_floor:
            continue
        key = tuple(int(x) for x in torch.floor(component["mean"] / nms_voxel).tolist())
        if key in occupied:
            continue
        occupied[key] = component["weight"]
        kept.append(component)
    if not kept:
        raise ValueError("beam fusion NMS removed every component; lower weight_floor")

    means = torch.stack([component["mean"] for component in kept])
    covariances = torch.linalg.inv(torch.stack([component["precision"] for component in kept]))
    covariances = 0.5 * (covariances + covariances.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariances)
    eigenvalues = eigenvalues.clamp(min=config.min_sigma_world**2, max=max_sigma**2)
    covariances = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)

    colors = []
    offsets = [0]
    contributor_views: list[int] = []
    contributor_components: list[int] = []
    contributor_depths: list[float] = []
    matched: dict[int, set[int]] = {view: set() for view in range(n_views)}
    for component in kept:
        color_sum = torch.zeros(3, dtype=torch.float64)
        amplitude_sum = 0.0
        for (view_index, splat_index), depth in zip(
            component["signature"], component["depths"], strict=True
        ):
            amplitude = float(views[view_index].amplitudes[splat_index])
            color_sum += amplitude * views[view_index].colors[splat_index]
            amplitude_sum += amplitude
            contributor_views.append(view_index)
            contributor_components.append(splat_index)
            contributor_depths.append(depth)
            matched[view_index].add(splat_index)
        offsets.append(len(contributor_views))
        colors.append(color_sum / max(amplitude_sum, 1e-12))
    unmatched = tuple(
        int(inputs.observations[view].n - len(matched[view])) for view in range(n_views)
    )

    dtype = inputs.observations[0].dtype
    gaussians = Gaussians3D.from_means_covs(
        means=means.to(dtype),
        covs=covariances.to(dtype),
        colors=torch.stack(colors).clamp(0.0, 1.0).to(dtype),
        opacity=torch.full((len(kept),), config.init_opacity, dtype=dtype),
        sh_degree=0,
    )
    lengths = torch.tensor([len(component["signature"]) for component in kept], dtype=torch.long)
    diagnostics: dict[str, object] = {
        "n_views": n_views,
        "n_view_pairs": len(pairs),
        "n_seeds": len(seeds),
        "n_components_prefilter": len(components),
        "n_components": len(kept),
        "component_view_histogram": {
            int(length): int((lengths == length).sum()) for length in lengths.unique(sorted=True)
        },
        "nms_voxel_size": nms_voxel,
        "max_sigma_world": max_sigma,
        "unmatched_per_view": list(unmatched),
    }
    return BeamFusionResult(
        gaussians=gaussians,
        component_offsets=torch.tensor(offsets, dtype=torch.long),
        contributor_view_indices=torch.tensor(contributor_views, dtype=torch.long),
        contributor_component_indices=torch.tensor(contributor_components, dtype=torch.long),
        contributor_depths=torch.tensor(contributor_depths, dtype=torch.float64),
        component_weights=torch.tensor(
            [component["weight"] for component in kept], dtype=torch.float64
        ),
        unmatched_per_view=unmatched,
        diagnostics=diagnostics,
    )


__all__ = [
    "BeamFusionConfig",
    "BeamFusionProgress",
    "BeamFusionResult",
    "fuse_gaussian_beams",
]
