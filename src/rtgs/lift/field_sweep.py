"""Fixed-anchor compact-field ray sweeps for controlled placement experiments.

The historical cost-volume prototype sampled RGB images and mixed an obsolete ray optimizer into
placement.  This module preserves only its useful mechanism: evaluate discrete camera-depth
hypotheses, exclude the source view from the robust neighbor aggregate, retain the best-agreeing
neighbor fraction, refine the bounded interval coarse-to-fine, and use peak width as ray
uncertainty.

All supervision comes from frozen :class:`~rtgs.core.observation2d.GaussianObservationField`
queries.  Source *component* identity chooses a deterministic ray and covariance only; the source
color is the complete queried field at that ray coordinate.  The three modes share the exact same
seeded fixed-anchor set so an experiment can isolate placement:

``bounded_midpoint``
    No field sweep. Place every anchor at the midpoint of its ray/AABB intersection.
``all_view_consensus``
    Coarse-to-fine sweep using all-view compact-field color variance.
``source_excluded_robust``
    Coarse-to-fine sweep using source-field color against the lowest-cost fixed fraction of
    neighboring views. The source view never counts as a neighbor.

This is CPU reference research code.  It is opt-in and does not change the production field-lift
default.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import ObservationQueryBackend
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.base import lift_covariance
from rtgs.lift.compact_carve import (
    CompactCarveConfig,
    CompactInitializationResult,
    CompactLineage,
    build_query_backends,
)

FieldSweepMode = Literal[
    "bounded_midpoint",
    "all_view_consensus",
    "source_excluded_robust",
]


@dataclass(frozen=True)
class FieldSweepConfig:
    """Bounded controls for fixed-anchor compact-field placement."""

    n_init_3d: int
    mode: FieldSweepMode = "source_excluded_robust"
    anchor_pool_multiplier: int = 2
    samples_per_round: int = 32
    coarse_to_fine_rounds: int = 3
    refine_ratio: float = 0.30
    robust_keep_fraction: float = 0.60
    min_neighbor_views: int = 2
    near: float = 0.05
    bounds_scale: float = 0.50
    coverage_scale: float = 1.0
    coverage_threshold: float = 0.40
    color_sigma: float = 0.20
    cost_tau: float = 0.01
    init_opacity: float = 0.10
    sh_degree: int = 0
    query_batch_size: int = 4096
    query_component_chunk: int = 256
    max_query_pairs: int = 1_048_576
    tile_size: int = 16
    max_index_entries_per_view: int = 16_000_000
    max_candidates_per_tile: int = 200_000
    max_index_entries_total: int | None = None
    max_index_bytes_total: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        integer_fields = (
            "n_init_3d",
            "anchor_pool_multiplier",
            "samples_per_round",
            "coarse_to_fine_rounds",
            "min_neighbor_views",
            "sh_degree",
            "query_batch_size",
            "query_component_chunk",
            "max_query_pairs",
            "tile_size",
            "max_index_entries_per_view",
            "max_candidates_per_tile",
            "seed",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.mode not in {
            "bounded_midpoint",
            "all_view_consensus",
            "source_excluded_robust",
        }:
            raise ValueError("mode is not a supported fixed-anchor placement")
        if self.n_init_3d <= 0:
            raise ValueError("n_init_3d must be positive")
        if self.anchor_pool_multiplier <= 0:
            raise ValueError("anchor_pool_multiplier must be positive")
        if self.samples_per_round < 2:
            raise ValueError("samples_per_round must be at least two")
        if self.coarse_to_fine_rounds <= 0:
            raise ValueError("coarse_to_fine_rounds must be positive")
        if self.min_neighbor_views <= 0:
            raise ValueError("min_neighbor_views must be positive")
        if self.sh_degree < 0 or self.sh_degree > 3:
            raise ValueError("sh_degree must be in [0,3]")
        if self.query_batch_size < self.samples_per_round:
            raise ValueError("query_batch_size must be at least samples_per_round")
        if (
            self.query_component_chunk <= 0
            or self.max_query_pairs < self.query_batch_size
            or self.tile_size <= 0
            or self.max_index_entries_per_view <= 0
            or self.max_candidates_per_tile <= 0
            or self.seed < 0
        ):
            raise ValueError("query/index/seed controls are outside their valid range")
        for name in ("max_index_entries_total", "max_index_bytes_total"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        finite_fields = (
            "refine_ratio",
            "robust_keep_fraction",
            "near",
            "bounds_scale",
            "coverage_scale",
            "coverage_threshold",
            "color_sigma",
            "cost_tau",
            "init_opacity",
        )
        for name in finite_fields:
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < self.refine_ratio <= 0.5:
            raise ValueError("refine_ratio must be in (0,0.5]")
        if not 0.0 < self.robust_keep_fraction <= 1.0:
            raise ValueError("robust_keep_fraction must be in (0,1]")
        if self.near <= 0 or self.bounds_scale <= 0 or self.coverage_scale <= 0:
            raise ValueError("near, bounds_scale, and coverage_scale must be positive")
        if not 0.0 <= self.coverage_threshold <= 1.0:
            raise ValueError("coverage_threshold must be in [0,1]")
        if self.color_sigma <= 0 or self.cost_tau <= 0:
            raise ValueError("color_sigma and cost_tau must be positive")
        if not 0.0 < self.init_opacity < 1.0:
            raise ValueError("init_opacity must be in (0,1)")


@dataclass(frozen=True)
class FieldSweepPointScores:
    """Field agreement for arbitrary ray samples."""

    score: torch.Tensor
    cost: torch.Tensor
    consensus_color: torch.Tensor
    coverage: torch.Tensor
    n_seen: torch.Tensor
    n_covered: torch.Tensor
    supported: torch.Tensor


def _validate_cpu_inputs(inputs: ReconstructionInputs) -> None:
    inputs.validate()
    dtypes = {field.dtype for field in inputs.observations}
    if len(dtypes) != 1:
        raise ValueError("all compact observations must share one dtype")
    if any(field.device.type != "cpu" for field in inputs.observations):
        raise ValueError("field sweep CPU reference requires CPU observation fields")
    if any(camera.R.device.type != "cpu" for camera in inputs.cameras):
        raise ValueError("field sweep CPU reference requires CPU cameras")


def _field_mass(field) -> torch.Tensor:
    return (
        field.amplitudes * (2.0 * math.pi) * field.effective_variances().prod(dim=1).sqrt()
    ).sum()


def _component_covariances(field, component_ids: torch.Tensor) -> torch.Tensor:
    variances = field.effective_variances()[component_ids]
    theta = field.rotations[component_ids]
    cos = theta.cos()
    sin = theta.sin()
    cov00 = cos.square() * variances[:, 0] + sin.square() * variances[:, 1]
    cov01 = cos * sin * (variances[:, 0] - variances[:, 1])
    cov11 = sin.square() * variances[:, 0] + cos.square() * variances[:, 1]
    return torch.stack(
        (
            torch.stack((cov00, cov01), dim=-1),
            torch.stack((cov01, cov11), dim=-1),
        ),
        dim=-2,
    )


def _balanced_quotas(capacities: list[int], total: int) -> list[int]:
    if total > sum(capacities):
        raise ValueError("n_init_3d exceeds the available compact components")
    quotas = [0] * len(capacities)
    remaining = total
    while remaining:
        progressed = False
        for view, capacity in enumerate(capacities):
            if quotas[view] >= capacity:
                continue
            quotas[view] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError("balanced anchor allocation made no progress")
    return quotas


def _select_anchors(
    inputs: ReconstructionInputs,
    config: FieldSweepConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select a seed-varying balanced subset from each view's top-mass pool."""

    quotas = _balanced_quotas(inputs.n_opt_2d, config.n_init_3d)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    source_views: list[torch.Tensor] = []
    source_components: list[torch.Tensor] = []
    source_xy: list[torch.Tensor] = []
    dtype = inputs.observations[0].dtype
    for view, (field, quota) in enumerate(zip(inputs.observations, quotas, strict=True)):
        if quota == 0:
            continue
        component_mass = (
            field.amplitudes * (2.0 * math.pi) * field.effective_variances().prod(dim=1).sqrt()
        )
        ranked = torch.argsort(component_mass, descending=True, stable=True)
        pool_count = min(field.n, max(quota, quota * config.anchor_pool_multiplier))
        pool = ranked[:pool_count]
        draw = torch.randperm(pool_count, generator=generator)[:quota]
        # Canonical output order follows mass rank, not the random draw order.
        chosen_rank = torch.sort(draw).values
        chosen = pool[chosen_rank]
        source_views.append(torch.full((quota,), view, dtype=torch.long))
        source_components.append(chosen)
        source_xy.append(field.native_means(chosen, dtype=dtype))
    return (
        torch.cat(source_views),
        torch.cat(source_components),
        torch.cat(source_xy),
    )


def _anchor_digest(
    source_views: torch.Tensor,
    source_components: torch.Tensor,
    source_xy: torch.Tensor,
) -> str:
    digest = hashlib.sha256(b"rtgs-fixed-field-sweep-anchors-v1\0")
    for value in (source_views, source_components, source_xy):
        cpu = value.detach().cpu().contiguous()
        digest.update(f"{tuple(cpu.shape)}:{cpu.dtype}\0".encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _search_bounds(
    inputs: ReconstructionInputs,
    dtype: torch.dtype,
    bounds_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, str]:
    if inputs.bounds_hint is not None:
        center, extent = inputs.bounds_hint
        center = center.to(dtype=dtype)
        source = "explicit_hint"
    else:
        positions = torch.stack([camera.position for camera in inputs.cameras]).to(dtype)
        forwards = torch.stack([camera.R[2] for camera in inputs.cameras]).to(dtype)
        forwards = torch.nn.functional.normalize(forwards, dim=-1)
        eye = torch.eye(3, dtype=dtype)
        projectors = eye[None] - forwards[:, :, None] * forwards[:, None, :]
        matrix = projectors.sum(dim=0)
        vector = (projectors @ positions[:, :, None]).sum(dim=0)[:, 0]
        center = torch.linalg.lstsq(matrix, vector, driver="gelsd").solution
        radius = 0.25 * (positions - center).norm(dim=-1).median().clamp_min(1e-3)
        extent = float((2.2 * radius).clamp_min(1e-3))
        source = "camera_axis_fallback"
    half = float(extent) * bounds_scale
    return center - half, center + half, center, float(extent), source


def _ray_box(
    origins: torch.Tensor,
    directions: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe = torch.where(
        directions.abs() < 1e-9,
        torch.full_like(directions, 1e-9),
        directions,
    )
    entry = (lower[None] - origins) / safe
    exit = (upper[None] - origins) / safe
    near = torch.minimum(entry, exit).amax(dim=-1)
    far = torch.maximum(entry, exit).amin(dim=-1)
    return near.clamp_min(0.0), far


def _query_views(
    inputs: ReconstructionInputs,
    points: torch.Tensor,
    config: FieldSweepConfig,
    backends: list[ObservationQueryBackend],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = points.shape[0]
    n_views = inputs.n_views
    colors = torch.empty((count, n_views, 3), dtype=points.dtype)
    coverage = torch.empty((count, n_views), dtype=points.dtype)
    seen = torch.empty((count, n_views), dtype=torch.bool)
    for view, (field, camera, backend) in enumerate(
        zip(inputs.observations, inputs.cameras, backends, strict=True)
    ):
        uv, depth = camera.project(points)
        component_chunk = min(
            config.query_component_chunk,
            max(1, config.max_query_pairs // max(count, 1)),
        )
        query = backend.query(uv, component_chunk=component_chunk)
        local_seen = (depth > config.near) & query.valid
        area = float(field.fit_window[2] * field.fit_window[3])
        mass = _field_mass(field).to(points).clamp_min(torch.finfo(points.dtype).tiny)
        relative_density = area * query.weight_sum.to(points) / mass
        local_coverage = 1.0 - torch.exp(-relative_density / config.coverage_scale)
        colors[:, view] = query.color.to(points)
        coverage[:, view] = local_coverage * local_seen
        seen[:, view] = local_seen
    return colors, coverage, seen


def _all_view_consensus_scores(
    colors: torch.Tensor,
    coverage: torch.Tensor,
    seen: torch.Tensor,
    config: FieldSweepConfig,
) -> FieldSweepPointScores:
    covered = seen & (coverage >= config.coverage_threshold)
    n_seen = seen.sum(dim=1)
    n_covered = covered.sum(dim=1)
    weights = coverage * seen
    weight_sum = weights.sum(dim=1).clamp_min(torch.finfo(colors.dtype).eps)
    consensus = (weights[..., None] * colors).sum(dim=1) / weight_sum[:, None]
    variance = (
        (weights * (colors - consensus[:, None]).square().mean(dim=-1)).sum(dim=1) / weight_sum
    ).clamp_min(0.0)
    minimum_total = min(config.min_neighbor_views + 1, colors.shape[1])
    required_fraction = torch.ceil(config.robust_keep_fraction * n_seen).to(torch.long)
    supported = (
        (n_seen >= minimum_total) & (n_covered >= minimum_total) & (n_covered >= required_fraction)
    )
    mean_coverage = weight_sum / n_seen.clamp_min(1).to(colors.dtype)
    score = mean_coverage * torch.exp(-variance / (2.0 * config.color_sigma * config.color_sigma))
    infinity = torch.full_like(variance, torch.inf)
    return FieldSweepPointScores(
        score=torch.where(supported, score, torch.zeros_like(score)),
        cost=torch.where(supported, variance, infinity),
        consensus_color=consensus,
        coverage=mean_coverage,
        n_seen=n_seen,
        n_covered=n_covered,
        supported=supported,
    )


def _source_excluded_robust_scores(
    colors: torch.Tensor,
    coverage: torch.Tensor,
    seen: torch.Tensor,
    source_views: torch.Tensor,
    config: FieldSweepConfig,
) -> FieldSweepPointScores:
    count, n_views, _channels = colors.shape
    if n_views <= config.min_neighbor_views:
        raise ValueError("robust sweep requires more views than min_neighbor_views")
    rows = torch.arange(count)
    source_color = colors[rows, source_views]
    source_coverage = coverage[rows, source_views]
    source_valid = seen[rows, source_views] & (source_coverage >= config.coverage_threshold)

    view_ids = torch.arange(n_views)
    neighbor = view_ids[None] != source_views[:, None]
    neighbor_seen = seen & neighbor
    covered = neighbor_seen & (coverage >= config.coverage_threshold)
    n_seen = neighbor_seen.sum(dim=1)
    n_covered = covered.sum(dim=1)
    keep = min(
        n_views - 1,
        max(
            config.min_neighbor_views,
            math.ceil(config.robust_keep_fraction * (n_views - 1)),
        ),
    )

    per_view_cost = (colors - source_color[:, None]).square().mean(dim=-1)
    per_view_cost = torch.where(
        covered,
        per_view_cost,
        torch.full_like(per_view_cost, torch.inf),
    )
    sorted_cost, sorted_indices = torch.sort(per_view_cost, dim=1, stable=True)
    kept_cost = sorted_cost[:, :keep]
    kept_indices = sorted_indices[:, :keep]
    kept_coverage = torch.gather(coverage, 1, kept_indices)
    kept_colors = torch.gather(
        colors,
        1,
        kept_indices[..., None].expand(-1, -1, 3),
    )
    supported = source_valid & (n_covered >= keep) & torch.isfinite(kept_cost).all(dim=1)
    robust_cost = kept_cost.mean(dim=1)
    neighbor_coverage = kept_coverage.mean(dim=1)
    color_weight = torch.cat((source_coverage[:, None], kept_coverage), dim=1)
    color_values = torch.cat((source_color[:, None], kept_colors), dim=1)
    consensus = (color_weight[..., None] * color_values).sum(dim=1) / color_weight.sum(
        dim=1, keepdim=True
    ).clamp_min(torch.finfo(colors.dtype).eps)
    score = (
        source_coverage
        * neighbor_coverage
        * torch.exp(-robust_cost / (2.0 * config.color_sigma * config.color_sigma))
    )
    infinity = torch.full_like(robust_cost, torch.inf)
    return FieldSweepPointScores(
        score=torch.where(supported, score, torch.zeros_like(score)),
        cost=torch.where(supported, robust_cost, infinity),
        consensus_color=consensus,
        coverage=neighbor_coverage,
        n_seen=n_seen,
        n_covered=n_covered,
        supported=supported,
    )


def score_field_sweep_points(
    inputs: ReconstructionInputs,
    points: torch.Tensor,
    source_views: torch.Tensor,
    config: FieldSweepConfig,
    backends: list[ObservationQueryBackend],
) -> FieldSweepPointScores:
    """Score world points under one of the two compact-field sweep objectives."""

    if config.mode == "bounded_midpoint":
        raise ValueError("bounded_midpoint does not define a field-sweep point score")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (Q,3)")
    if source_views.shape != (points.shape[0],) or source_views.dtype != torch.long:
        raise ValueError("source_views must be int64 with shape (Q,)")
    if source_views.numel() and (
        int(source_views.min()) < 0 or int(source_views.max()) >= inputs.n_views
    ):
        raise ValueError("source_views contains an out-of-range view")
    colors, coverage, seen = _query_views(inputs, points, config, backends)
    if config.mode == "all_view_consensus":
        return _all_view_consensus_scores(colors, coverage, seen, config)
    return _source_excluded_robust_scores(
        colors,
        coverage,
        seen,
        source_views,
        config,
    )


def _query_backend_config(config: FieldSweepConfig) -> CompactCarveConfig:
    return CompactCarveConfig(
        n_init_3d=config.n_init_3d,
        samples_per_ray=config.samples_per_round,
        query_batch_size=config.query_batch_size,
        query_component_chunk=config.query_component_chunk,
        max_query_pairs=config.max_query_pairs,
        tile_size=config.tile_size,
        max_index_entries_per_view=config.max_index_entries_per_view,
        max_candidates_per_tile=config.max_candidates_per_tile,
        max_index_entries_total=config.max_index_entries_total,
        max_index_bytes_total=config.max_index_bytes_total,
        min_views=1,
        near=config.near,
        coverage_scale=config.coverage_scale,
        coverage_threshold=config.coverage_threshold,
        init_opacity=config.init_opacity,
        sh_degree=config.sh_degree,
        seed=config.seed,
    )


def _source_colors(
    inputs: ReconstructionInputs,
    source_views: torch.Tensor,
    source_xy: torch.Tensor,
) -> torch.Tensor:
    result = torch.empty((source_views.shape[0], 3), dtype=source_xy.dtype)
    for view in source_views.unique(sorted=True).tolist():
        rows = source_views == view
        result[rows] = inputs.observations[view].query(source_xy[rows]).color.to(result)
    return result


class FieldSweepInitializer:
    """Place a fixed source-ray set by midpoint or compact-field coarse-to-fine sweep."""

    def __init__(self, config: FieldSweepConfig):
        self.config = config

    def initialize(
        self,
        inputs: ReconstructionInputs,
        backends: list[ObservationQueryBackend] | None = None,
    ) -> CompactInitializationResult:
        """Return a fixed-count initialization with arm-comparable anchor lineage."""

        _validate_cpu_inputs(inputs)
        if (
            self.config.mode == "source_excluded_robust"
            and inputs.n_views <= self.config.min_neighbor_views
        ):
            raise ValueError("robust sweep has too few neighboring compact views")
        dtype = inputs.observations[0].dtype
        source_views, source_components, source_xy = _select_anchors(inputs, self.config)
        anchor_sha256 = _anchor_digest(source_views, source_components, source_xy)

        lower_box, upper_box, center, extent, bounds_source = _search_bounds(
            inputs,
            dtype,
            self.config.bounds_scale,
        )
        origins = torch.empty((self.config.n_init_3d, 3), dtype=dtype)
        directions = torch.empty_like(origins)
        for view in source_views.unique(sorted=True).tolist():
            rows = source_views == view
            origin, direction = inputs.cameras[view].pixel_rays(source_xy[rows])
            origins[rows] = origin.to(dtype).expand(int(rows.sum()), -1)
            directions[rows] = direction.to(dtype)
        original_lower, original_upper = _ray_box(
            origins,
            directions,
            lower_box,
            upper_box,
        )
        original_lower = original_lower.clamp_min(self.config.near)
        ray_valid = original_upper > original_lower
        if not bool(ray_valid.all()):
            raise ValueError(
                "fixed-anchor field sweep found a source ray with no forward AABB intersection"
            )

        midpoint = 0.5 * (original_lower + original_upper)
        interval = original_upper - original_lower
        depths = midpoint.clone()
        depth_sigmas = (0.10 * interval).clamp_min(torch.finfo(dtype).eps)
        best_scores = torch.ones_like(depths)
        best_costs = torch.zeros_like(depths)
        best_colors = _source_colors(inputs, source_views, source_xy)
        supported = ray_valid.clone()
        round_diagnostics: list[dict[str, float | int]] = []

        if self.config.mode != "bounded_midpoint":
            if backends is None:
                backends = build_query_backends(
                    inputs.observations,
                    _query_backend_config(self.config),
                    device="cpu",
                )
            if len(backends) != inputs.n_views:
                raise ValueError("backends must contain one compact-field query backend per view")
            current_lower = original_lower.clone()
            current_upper = original_upper.clone()
            supported = torch.zeros_like(ray_valid)
            best_scores.zero_()
            best_costs.fill_(torch.inf)
            ray_batch = max(1, self.config.query_batch_size // self.config.samples_per_round)
            steps = torch.linspace(
                0.0,
                1.0,
                self.config.samples_per_round,
                dtype=dtype,
            )
            for round_index in range(self.config.coarse_to_fine_rounds):
                round_supported = torch.zeros_like(ray_valid)
                round_depth = depths.clone()
                round_sigma = depth_sigmas.clone()
                round_score = best_scores.clone()
                round_cost = best_costs.clone()
                round_color = best_colors.clone()
                for start in range(0, self.config.n_init_3d, ray_batch):
                    end = min(start + ray_batch, self.config.n_init_3d)
                    local_depths = (
                        current_lower[start:end, None]
                        + (current_upper[start:end] - current_lower[start:end])[:, None]
                        * steps[None]
                    )
                    points = origins[start:end, None] + (
                        local_depths[..., None] * directions[start:end, None]
                    )
                    repeated_sources = source_views[start:end, None].expand(
                        -1, self.config.samples_per_round
                    )
                    point_scores = score_field_sweep_points(
                        inputs,
                        points.reshape(-1, 3),
                        repeated_sources.reshape(-1),
                        self.config,
                        backends,
                    )
                    score = point_scores.score.reshape(end - start, -1)
                    cost = point_scores.cost.reshape(end - start, -1)
                    valid = point_scores.supported.reshape(end - start, -1)
                    color = point_scores.consensus_color.reshape(end - start, -1, 3)
                    selection_score = torch.where(
                        valid,
                        score,
                        torch.full_like(score, -1.0),
                    )
                    best_index = selection_score.argmax(dim=1)
                    rows = torch.arange(end - start)
                    local_supported = valid.any(dim=1)
                    local_best_depth = local_depths[rows, best_index]
                    local_best_score = score[rows, best_index]
                    local_best_cost = cost[rows, best_index]
                    local_best_color = color[rows, best_index]

                    relative_cost = cost - cost.amin(dim=1, keepdim=True)
                    logits = -relative_cost / self.config.cost_tau
                    weights = torch.exp(logits.clamp(min=-80.0, max=0.0)) * valid
                    weight_sum = weights.sum(dim=1).clamp_min(torch.finfo(dtype).eps)
                    mean = (weights * local_depths).sum(dim=1) / weight_sum
                    variance = (weights * (local_depths - mean[:, None]).square()).sum(
                        dim=1
                    ) / weight_sum
                    depth_step = (current_upper[start:end] - current_lower[start:end]) / (
                        self.config.samples_per_round - 1
                    )
                    local_sigma = variance.clamp_min(0.0).sqrt()
                    local_sigma = torch.maximum(local_sigma, 0.25 * depth_step)
                    local_sigma = torch.minimum(local_sigma, 2.0 * depth_step)

                    sl = slice(start, end)
                    round_supported[sl] = local_supported
                    round_depth[sl] = torch.where(
                        local_supported,
                        local_best_depth,
                        round_depth[sl],
                    )
                    round_sigma[sl] = torch.where(
                        local_supported,
                        local_sigma,
                        round_sigma[sl],
                    )
                    round_score[sl] = torch.where(
                        local_supported,
                        local_best_score,
                        round_score[sl],
                    )
                    round_cost[sl] = torch.where(
                        local_supported,
                        local_best_cost,
                        round_cost[sl],
                    )
                    round_color[sl] = torch.where(
                        local_supported[:, None],
                        local_best_color,
                        round_color[sl],
                    )

                supported |= round_supported
                depths = torch.where(round_supported, round_depth, depths)
                depth_sigmas = torch.where(round_supported, round_sigma, depth_sigmas)
                best_scores = torch.where(round_supported, round_score, best_scores)
                best_costs = torch.where(round_supported, round_cost, best_costs)
                best_colors = torch.where(
                    round_supported[:, None],
                    round_color,
                    best_colors,
                )
                finite_cost = best_costs[torch.isfinite(best_costs)]
                round_diagnostics.append(
                    {
                        "round": round_index,
                        "supported_tracks": int(round_supported.sum()),
                        "cumulative_supported_tracks": int(supported.sum()),
                        "median_best_cost": (
                            float(finite_cost.median()) if finite_cost.numel() else math.inf
                        ),
                    }
                )
                span = current_upper - current_lower
                half = self.config.refine_ratio * span
                proposed_lower = torch.maximum(original_lower, depths - half)
                proposed_upper = torch.minimum(original_upper, depths + half)
                minimum_width = (
                    interval.clamp_min(torch.finfo(dtype).eps)
                    / self.config.samples_per_round
                    * 0.25
                )
                proposed_lower = torch.minimum(
                    proposed_lower,
                    depths - 0.5 * minimum_width,
                )
                proposed_upper = torch.maximum(
                    proposed_upper,
                    depths + 0.5 * minimum_width,
                )
                proposed_lower = torch.maximum(proposed_lower, original_lower)
                proposed_upper = torch.minimum(proposed_upper, original_upper)
                current_lower = torch.where(
                    round_supported,
                    proposed_lower,
                    current_lower,
                )
                current_upper = torch.where(
                    round_supported,
                    proposed_upper,
                    current_upper,
                )

        means = origins + depths[:, None] * directions
        ray_sigmas = depth_sigmas * directions.norm(dim=-1)
        covariances = torch.empty((self.config.n_init_3d, 3, 3), dtype=dtype)
        for view in source_views.unique(sorted=True).tolist():
            rows = source_views == view
            covariance_2d = _component_covariances(
                inputs.observations[view],
                source_components[rows],
            ).to(dtype)
            covariances[rows] = lift_covariance(
                inputs.cameras[view],
                source_xy[rows],
                covariance_2d,
                depths[rows],
                ray_sigmas[rows],
            )
        opacity = torch.full(
            (self.config.n_init_3d,),
            self.config.init_opacity,
            dtype=dtype,
        )
        gaussians = Gaussians3D.from_means_covs(
            means,
            covariances,
            best_colors.clamp(0.0, 1.0),
            opacity,
            sh_degree=self.config.sh_degree,
        )
        finite_costs = best_costs[torch.isfinite(best_costs)]
        diagnostics: dict[str, object] = {
            "initializer": "fixed_anchor_field_sweep",
            "mode": self.config.mode,
            "n_init_3d": self.config.n_init_3d,
            "anchor_sha256": anchor_sha256,
            "anchor_pool_multiplier": self.config.anchor_pool_multiplier,
            "bounds_source": bounds_source,
            "bounds_center": center.tolist(),
            "bounds_extent": extent,
            "search_aabb_lower": lower_box.tolist(),
            "search_aabb_upper": upper_box.tolist(),
            "samples_per_round": self.config.samples_per_round,
            "coarse_to_fine_rounds": (
                0 if self.config.mode == "bounded_midpoint" else self.config.coarse_to_fine_rounds
            ),
            "robust_keep_fraction": self.config.robust_keep_fraction,
            "min_neighbor_views": self.config.min_neighbor_views,
            "robust_keep_view_count": (
                min(
                    inputs.n_views - 1,
                    max(
                        self.config.min_neighbor_views,
                        math.ceil(self.config.robust_keep_fraction * (inputs.n_views - 1)),
                    ),
                )
                if self.config.mode == "source_excluded_robust"
                else None
            ),
            "source_view_counted_as_neighbor": (
                False if self.config.mode == "source_excluded_robust" else None
            ),
            "supported_track_count": int(supported.sum()),
            "supported_track_fraction": float(supported.float().mean()),
            "best_cost_median": (float(finite_costs.median()) if finite_costs.numel() else None),
            "rounds": round_diagnostics,
            "depth_within_original_bounds": bool(
                ((depths >= original_lower) & (depths <= original_upper)).all()
            ),
        }
        return CompactInitializationResult(
            gaussians=gaussians,
            lineage=CompactLineage(
                source_view_indices=source_views,
                source_component_indices=source_components,
                source_xy=source_xy,
            ),
            depths=depths,
            depth_sigmas=depth_sigmas,
            ray_sigmas=ray_sigmas,
            scores=best_scores,
            diagnostics=diagnostics,
        )


__all__ = [
    "FieldSweepConfig",
    "FieldSweepInitializer",
    "FieldSweepMode",
    "FieldSweepPointScores",
    "score_field_sweep_points",
]
