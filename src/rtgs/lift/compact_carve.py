"""CPU reference initializer from RGB-free compact observation fields.

This is deliberately separate from :class:`rtgs.lift.carve.CarveLifter`, whose legacy
contract consumes ``SceneData.images``.  The initializer below consumes the typed, serializable
RGB-free seam: calibrated :class:`~rtgs.data.reconstruction_inputs.ReconstructionInputs`.  It
either samples a bounded number of source rays or uses one center ray per optimized 2D component,
then scores their depth tunnels by querying every frozen 2D teacher without materializing an RGB
image or a dense voxel grid.

Source component ids are proposal lineage.  They choose a ray and provide its 2D covariance, but
they do not select the supervision target: placement uses coverage-weighted queries to every
teacher along the proposed ray, and initialization stores the resulting consensus color after
clamping it to the 3D Gaussian RGB domain.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import (
    GaussianObservationField,
    GaussianObservationIndex,
    ObservationQueryBackend,
)
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.base import lift_covariance

AnchorMode = Literal["mass_random", "component_centers"]


@dataclass(frozen=True)
class CompactCarveConfig:
    """Controls for the sparse RGB-free initialization reference path."""

    n_init_3d: int
    candidate_multiplier: int = 4
    anchor_mode: AnchorMode = "mass_random"
    max_anchor_candidates: int = 1_000_000
    samples_per_ray: int = 48
    query_batch_size: int = 4096
    query_component_chunk: int = 256
    max_query_pairs: int = 1_048_576
    tile_size: int = 16
    max_index_entries_per_view: int = 16_000_000
    max_candidates_per_tile: int = 200_000
    seed: int = 0
    bounds_scale: float = 0.5
    near: float = 0.05
    min_views: int = 2
    hull_fraction: float = 0.85
    coverage_scale: float = 1.0
    coverage_threshold: float = 0.40
    color_std_sigma: float = 0.20
    min_score: float = 0.05
    peak_radius_steps: float = 3.0
    init_opacity: float = 0.1
    sh_degree: int = 0
    max_anchor_rounds: int = 8
    select_all_eligible: bool = False

    def __post_init__(self) -> None:
        integer_fields = (
            "n_init_3d",
            "candidate_multiplier",
            "max_anchor_candidates",
            "samples_per_ray",
            "query_batch_size",
            "query_component_chunk",
            "max_query_pairs",
            "tile_size",
            "max_index_entries_per_view",
            "max_candidates_per_tile",
            "seed",
            "min_views",
            "sh_degree",
            "max_anchor_rounds",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.n_init_3d <= 0:
            raise ValueError("n_init_3d must be positive")
        if self.candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive")
        if self.anchor_mode not in {"mass_random", "component_centers"}:
            raise ValueError("anchor_mode must be 'mass_random' or 'component_centers'")
        if self.max_anchor_candidates <= 0:
            raise ValueError("max_anchor_candidates must be positive")
        if self.samples_per_ray < 2:
            raise ValueError("samples_per_ray must be at least two")
        if self.query_batch_size < self.samples_per_ray:
            raise ValueError("query_batch_size must be at least samples_per_ray")
        if self.query_component_chunk <= 0:
            raise ValueError("query_component_chunk must be positive")
        if self.max_query_pairs < self.query_batch_size:
            raise ValueError("max_query_pairs must be at least query_batch_size")
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if self.max_index_entries_per_view <= 0:
            raise ValueError("max_index_entries_per_view must be positive")
        if self.max_candidates_per_tile <= 0:
            raise ValueError("max_candidates_per_tile must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not math.isfinite(self.bounds_scale) or self.bounds_scale <= 0:
            raise ValueError("bounds_scale must be finite and positive")
        if not math.isfinite(self.near) or self.near <= 0:
            raise ValueError("near must be finite and positive")
        if self.min_views <= 0:
            raise ValueError("min_views must be positive")
        if not 0.0 < self.hull_fraction <= 1.0:
            raise ValueError("hull_fraction must be in (0,1]")
        if not math.isfinite(self.coverage_scale) or self.coverage_scale <= 0:
            raise ValueError("coverage_scale must be finite and positive")
        if not 0.0 <= self.coverage_threshold <= 1.0:
            raise ValueError("coverage_threshold must be in [0,1]")
        if not math.isfinite(self.color_std_sigma) or self.color_std_sigma <= 0:
            raise ValueError("color_std_sigma must be finite and positive")
        if not math.isfinite(self.min_score) or self.min_score < 0:
            raise ValueError("min_score must be finite and non-negative")
        if not math.isfinite(self.peak_radius_steps) or self.peak_radius_steps <= 0:
            raise ValueError("peak_radius_steps must be finite and positive")
        if not 0.0 < self.init_opacity < 1.0:
            raise ValueError("init_opacity must be in (0,1)")
        if self.sh_degree < 0 or self.sh_degree > 3:
            raise ValueError("sh_degree must be in [0,3]")
        if self.max_anchor_rounds <= 0:
            raise ValueError("max_anchor_rounds must be positive")
        if not isinstance(self.select_all_eligible, bool):
            raise TypeError("select_all_eligible must be a bool")


@dataclass(frozen=True)
class CompactPointScores:
    """Global teacher-consensus statistics for arbitrary world points."""

    score: torch.Tensor  # (Q,)
    consensus_color: torch.Tensor  # (Q,3), deliberately unclamped
    color_variance: torch.Tensor  # (Q,), channel-mean weighted variance
    coverage: torch.Tensor  # (Q,), mean soft coverage over views that see the point
    n_seen: torch.Tensor  # (Q,), integer count
    n_covered: torch.Tensor  # (Q,), integer count


@dataclass(frozen=True)
class CompactLineage:
    """Hard ray/covariance provenance; never a supervision label or rendering assignment."""

    source_view_indices: torch.Tensor  # (N,), int64
    source_component_indices: torch.Tensor  # (N,), int64
    source_xy: torch.Tensor  # (N,2)


@dataclass(frozen=True)
class CompactCandidateAudit:
    """Detached, candidate-complete snapshot of one initialization decision.

    Every tensor whose name starts with ``candidate_`` has leading dimension ``C``, where
    ``C`` is the complete proposed-ray count. ``selected_candidate_indices`` indexes those
    tensors in the exact output order. The initializer gives callbacks detached clones so an
    audit sink may serialize or even mutate its snapshot without changing the returned result.
    """

    candidate_source_view_indices: torch.Tensor  # (C,), int64
    candidate_source_component_indices: torch.Tensor  # (C,), int64
    candidate_source_xy: torch.Tensor  # (C,2), exact ray coordinates
    candidate_best_depths: torch.Tensor  # (C,)
    candidate_depth_sigmas: torch.Tensor  # (C,)
    candidate_best_means: torch.Tensor  # (C,3)
    candidate_best_scores: torch.Tensor  # (C,)
    candidate_best_depth_indices: torch.Tensor  # (C,), int64
    candidate_best_coverages: torch.Tensor  # (C,)
    candidate_best_color_variances: torch.Tensor  # (C,)
    candidate_best_n_seen: torch.Tensor  # (C,), int64
    candidate_best_n_covered: torch.Tensor  # (C,), int64
    candidate_second_best_scores: torch.Tensor  # (C,)
    candidate_score_margins: torch.Tensor  # (C,), best minus second-best
    candidate_half_max_widths: torch.Tensor  # (C,), contiguous sampled depth width
    candidate_consensus_colors: torch.Tensor  # (C,3), deliberately unclamped
    candidate_valid_mask: torch.Tensor  # (C,), bool
    candidate_eligible_mask: torch.Tensor  # (C,), bool
    selected_candidate_indices: torch.Tensor  # (N,), int64


CompactCandidateAuditCallback = Callable[[CompactCandidateAudit], None]


@dataclass(frozen=True)
class CompactRayDepthAuditBatch:
    """Detached snapshot of every sampled depth for a contiguous candidate batch.

    ``candidate_start`` is inclusive and ``candidate_end`` is exclusive in the complete
    candidate ordering. Every sample tensor has shape ``(B,S)`` except consensus colors, which
    have shape ``(B,S,3)``. Candidate identity tensors have leading dimension ``B``. The score is
    the exact aggregate score used for winner selection, including the invalid-ray zero mask.
    ``half-max`` widths in :class:`CompactCandidateAudit` measure the contiguous run around the
    winning sample whose scores are at least half the winning score, inclusive of one depth cell.
    """

    candidate_start: int
    candidate_end: int
    candidate_source_view_indices: torch.Tensor  # (B,), int64
    candidate_source_component_indices: torch.Tensor  # (B,), int64
    candidate_source_xy: torch.Tensor  # (B,2)
    candidate_valid_mask: torch.Tensor  # (B,), bool
    depths: torch.Tensor  # (B,S)
    scores: torch.Tensor  # (B,S)
    coverages: torch.Tensor  # (B,S)
    color_variances: torch.Tensor  # (B,S)
    n_seen: torch.Tensor  # (B,S), int64
    n_covered: torch.Tensor  # (B,S), int64
    consensus_colors: torch.Tensor  # (B,S,3), deliberately unclamped


CompactRayDepthAuditCallback = Callable[[CompactRayDepthAuditBatch], None]


@dataclass(frozen=True)
class CompactInitializationResult:
    """Exact-budget 3D initialization and reproducibility diagnostics."""

    gaussians: Gaussians3D
    lineage: CompactLineage
    depths: torch.Tensor
    depth_sigmas: torch.Tensor
    ray_sigmas: torch.Tensor
    scores: torch.Tensor
    diagnostics: dict[str, object]

    @property
    def n_init_3d(self) -> int:
        return self.gaussians.n


@dataclass(frozen=True)
class CompactPlacementProgress:
    """Typed, silent-by-default progress record for the exact image-free placement loop.

    The library emits these; long-running consumers (benchmark, CLI) throttle their printing. The
    final counters are also persisted in :attr:`CompactInitializationResult.diagnostics` so a long
    silent regression cannot recur unobserved.
    """

    phase: str  # "index_built", "ray_batch", or "complete"
    completed_ray_batches: int
    total_ray_batches: int
    sampled_points: int
    completed_point_view_queries: int
    evaluated_pairs: int
    index_payload_bytes: tuple[int, ...]
    index_build_seconds: float
    elapsed_seconds: float
    points_per_second: float
    estimated_remaining_seconds: float
    current_pair_chunk: int
    peak_pair_chunk: int

    def format_line(self) -> str:
        """One-line human summary for progress printers."""
        payload_mib = sum(self.index_payload_bytes) / 1024**2
        return (
            f"[compact-placement {self.phase}] "
            f"batch {self.completed_ray_batches}/{self.total_ray_batches} "
            f"points {self.sampled_points} pairs {self.evaluated_pairs} "
            f"chunk {self.current_pair_chunk}/{self.peak_pair_chunk} "
            f"index {payload_mib:.1f}MiB build {self.index_build_seconds:.2f}s "
            f"elapsed {self.elapsed_seconds:.1f}s "
            f"~{self.estimated_remaining_seconds:.1f}s left "
            f"({self.points_per_second:.0f} pts/s)"
        )


CompactPlacementProgressCallback = Callable[[CompactPlacementProgress], None]


def make_placement_progress_printer(
    *,
    every_batches: int = 10,
    every_seconds: float = 30.0,
    printer: Callable[[str], None] = print,
) -> CompactPlacementProgressCallback:
    """Build a throttled progress callback that prints at a bounded batch/time interval.

    Always prints the ``index_built`` and ``complete`` phases; ``ray_batch`` phases print at most
    once every ``every_batches`` batches or ``every_seconds`` seconds, whichever comes first.
    """
    state = {"last_batch": 0, "last_time": 0.0}

    def callback(record: CompactPlacementProgress) -> None:
        due = (
            record.phase != "ray_batch"
            or record.completed_ray_batches - state["last_batch"] >= every_batches
            or record.elapsed_seconds - state["last_time"] >= every_seconds
        )
        if due:
            state["last_batch"] = record.completed_ray_batches
            state["last_time"] = record.elapsed_seconds
            printer(record.format_line())

    return callback


def build_query_backends(
    observations: Sequence[GaussianObservationField],
    config: CompactCarveConfig,
    device: str = "cpu",
) -> list[ObservationQueryBackend]:
    """Build one per-view observation query backend under the compact Carve caps.

    ``device="cpu"`` returns the exact CPU CSR indexes (identical to the defaults that
    :func:`score_world_points` and :class:`CompactCarveInitializer` build internally).
    ``device="cuda"`` wraps those same CPU-built indexes in the experimental GPU query
    backend (``rtgs.core.observation2d_cuda``); it accepts the seam's CPU query points and
    returns CPU results, so the placement pipeline itself is unchanged. The CPU path stays
    the correctness oracle.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    indexes = [
        GaussianObservationIndex(
            field,
            tile_size=config.tile_size,
            max_entries=config.max_index_entries_per_view,
            max_candidates=config.max_candidates_per_tile,
            max_query_pairs=config.max_query_pairs,
        )
        for field in observations
    ]
    if device == "cpu":
        return list(indexes)
    from rtgs.core.observation2d_cuda import GaussianObservationIndexCuda

    return [GaussianObservationIndexCuda(index) for index in indexes]


def score_world_points(
    inputs: ReconstructionInputs,
    points: torch.Tensor,
    config: CompactCarveConfig,
    backends: list[ObservationQueryBackend] | None = None,
) -> CompactPointScores:
    """Score world points using every compact teacher, with no source-lineage input.

    For view ``i``, the exact teacher returns color ``C_i`` and denominator ``D_i``.  The
    dimensionless relative density and soft coverage are

    ``r_i = A_i D_i / M_i`` and ``h_i = 1 - exp(-r_i / coverage_scale)``,

    where ``A_i`` is fitted-window area and ``M_i`` is analytic Gaussian-mixture mass.  Both
    ``r_i`` and the resulting hull test are invariant to an exact co-located split into identical
    components whose amplitudes sum to the original. Exact teacher colors retain normalized-blend
    epsilon and additive semantics.
    Built-in field/index backends are identity-checked against the ordered teacher; correspondence
    remains an explicit caller contract for third-party query backends.
    """
    _validate_cpu_inputs(inputs)
    points = torch.as_tensor(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (Q,3)")
    if not points.is_floating_point() or not bool(torch.isfinite(points).all()):
        raise ValueError("points must be finite floating-point values")
    if points.device.type != "cpu":
        raise ValueError("compact Carve CPU reference requires CPU points")
    if backends is None:
        backends = [
            GaussianObservationIndex(
                field,
                tile_size=config.tile_size,
                max_entries=config.max_index_entries_per_view,
                max_candidates=config.max_candidates_per_tile,
                max_query_pairs=config.max_query_pairs,
            )
            for field in inputs.observations
        ]
    if len(backends) != inputs.n_views:
        raise ValueError("backends must contain one query backend per view")
    for field, backend in zip(inputs.observations, backends, strict=True):
        if isinstance(backend, GaussianObservationIndex):
            if backend.field is not field:
                raise ValueError("indexed query backends must correspond to their ordered teacher")
            if (
                backend.tile_size != config.tile_size
                or backend.n_entries > config.max_index_entries_per_view
                or backend.max_candidates > config.max_candidates_per_tile
            ):
                raise ValueError("indexed query backend differs from compact Carve caps")
        if isinstance(backend, GaussianObservationField) and backend is not field:
            raise ValueError("field query backends must correspond to their ordered teacher")

    parts: list[CompactPointScores] = []
    for start in range(0, points.shape[0], config.query_batch_size):
        parts.append(
            _score_world_points_batch(
                inputs,
                points[start : start + config.query_batch_size],
                config,
                backends,
            )
        )
    if not parts:
        empty = points.new_empty((0,))
        return CompactPointScores(
            score=empty,
            consensus_color=points.new_empty((0, 3)),
            color_variance=empty,
            coverage=empty,
            n_seen=torch.empty(0, dtype=torch.long),
            n_covered=torch.empty(0, dtype=torch.long),
        )
    return CompactPointScores(
        score=torch.cat([part.score for part in parts]),
        consensus_color=torch.cat([part.consensus_color for part in parts]),
        color_variance=torch.cat([part.color_variance for part in parts]),
        coverage=torch.cat([part.coverage for part in parts]),
        n_seen=torch.cat([part.n_seen for part in parts]),
        n_covered=torch.cat([part.n_covered for part in parts]),
    )


class CompactCarveInitializer:
    """Sparse ray-tunnel initializer over the typed RGB-free reconstruction-input seam."""

    def __init__(self, config: CompactCarveConfig):
        self.config = config

    def initialize(
        self,
        inputs: ReconstructionInputs,
        backends: list[ObservationQueryBackend] | None = None,
        *,
        candidate_audit_callback: CompactCandidateAuditCallback | None = None,
        ray_depth_audit_callback: CompactRayDepthAuditCallback | None = None,
        progress_callback: CompactPlacementProgressCallback | None = None,
    ) -> CompactInitializationResult:
        """Produce ``n_init_3d`` all-view-scored Gaussians or fail on insufficient support.

        ``candidate_audit_callback`` is an opt-in observation hook. When supplied, it is called
        once with detached clones of every proposed candidate and the exact selected indices.
        ``ray_depth_audit_callback`` is a scalable observation hook called once per contiguous
        ray batch with detached clones of every sampled depth and its exact consensus statistics.
        ``progress_callback`` is silent by default; when supplied it receives a typed
        :class:`CompactPlacementProgress` record at index completion, after every ray batch, and at
        completion. Library code never prints; long-running consumers throttle their own output.
        """
        _validate_cpu_inputs(inputs)
        if self.config.min_views > inputs.n_views:
            raise ValueError("compact Carve min_views exceeds the number of compact views")
        dtype = inputs.observations[0].dtype
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        build_start = time.perf_counter()
        if backends is None:
            backends = [
                GaussianObservationIndex(
                    field,
                    tile_size=self.config.tile_size,
                    max_entries=self.config.max_index_entries_per_view,
                    max_candidates=self.config.max_candidates_per_tile,
                    max_query_pairs=self.config.max_query_pairs,
                )
                for field in inputs.observations
            ]
        if len(backends) != inputs.n_views:
            raise ValueError("backends must contain one query backend per view")
        for field, backend in zip(inputs.observations, backends, strict=True):
            if isinstance(backend, GaussianObservationIndex):
                if backend.field is not field:
                    raise ValueError(
                        "indexed query backends must correspond to their ordered teacher"
                    )
                if (
                    backend.tile_size != self.config.tile_size
                    or backend.n_entries > self.config.max_index_entries_per_view
                    or backend.max_candidates > self.config.max_candidates_per_tile
                ):
                    raise ValueError("indexed query backend differs from compact Carve caps")
            if isinstance(backend, GaussianObservationField) and backend is not field:
                raise ValueError("field query backends must correspond to their ordered teacher")

        index_build_seconds = time.perf_counter() - build_start
        index_payload_bytes = tuple(int(getattr(b, "payload_bytes", 0)) for b in backends)
        placement_start = time.perf_counter()

        def _placement_counters() -> tuple[int, int, int]:
            evaluated = sum(int(getattr(b, "total_pairs_evaluated", 0)) for b in backends)
            current_chunk = max(
                (int(getattr(b, "last_pair_chunk", 0)) for b in backends), default=0
            )
            peak_chunk = max((int(getattr(b, "peak_pair_chunk", 0)) for b in backends), default=0)
            return evaluated, current_chunk, peak_chunk

        view_ids, component_ids, xy, anchor_attempts, proposed_per_view = _propose_anchors(
            inputs,
            self.config,
            generator,
        )
        candidate_count = int(view_ids.numel())
        if candidate_count < self.config.n_init_3d:
            raise ValueError(
                "compact Carve proposed fewer anchors than n_init_3d; increase the random "
                "candidate budget or provide more optimized 2D components"
            )

        center, extent = _center_and_extent(inputs, dtype)
        half = extent * self.config.bounds_scale
        lo = center - half
        hi = center + half
        depths = torch.zeros(candidate_count, dtype=dtype)
        sigmas = torch.zeros(candidate_count, dtype=dtype)
        means = torch.zeros(candidate_count, 3, dtype=dtype)
        best_scores = torch.zeros(candidate_count, dtype=dtype)
        best_colors = torch.zeros(candidate_count, 3, dtype=dtype)
        valid_rays = torch.zeros(candidate_count, dtype=torch.bool)
        if candidate_audit_callback is not None:
            best_depth_indices = torch.zeros(candidate_count, dtype=torch.long)
            best_coverages = torch.zeros(candidate_count, dtype=dtype)
            best_color_variances = torch.zeros(candidate_count, dtype=dtype)
            best_n_seen = torch.zeros(candidate_count, dtype=torch.long)
            best_n_covered = torch.zeros(candidate_count, dtype=torch.long)
            second_best_scores = torch.zeros(candidate_count, dtype=dtype)
            score_margins = torch.zeros(candidate_count, dtype=dtype)
            half_max_widths = torch.zeros(candidate_count, dtype=dtype)

        ray_batch = max(1, self.config.query_batch_size // self.config.samples_per_ray)
        total_ray_batches = math.ceil(candidate_count / ray_batch)
        total_points = candidate_count * self.config.samples_per_ray
        steps = (
            torch.arange(self.config.samples_per_ray, dtype=dtype) + 0.5
        ) / self.config.samples_per_ray
        if progress_callback is not None:
            progress_callback(
                CompactPlacementProgress(
                    phase="index_built",
                    completed_ray_batches=0,
                    total_ray_batches=total_ray_batches,
                    sampled_points=0,
                    completed_point_view_queries=0,
                    evaluated_pairs=0,
                    index_payload_bytes=index_payload_bytes,
                    index_build_seconds=index_build_seconds,
                    elapsed_seconds=time.perf_counter() - placement_start,
                    points_per_second=0.0,
                    estimated_remaining_seconds=0.0,
                    current_pair_chunk=0,
                    peak_pair_chunk=0,
                )
            )
        completed_batches = 0
        for start in range(0, candidate_count, ray_batch):
            end = min(start + ray_batch, candidate_count)
            local_views = view_ids[start:end]
            local_xy = xy[start:end]
            # Balanced source ordering creates contiguous view runs; handle any boundary robustly.
            local_origins = torch.empty(end - start, 3, dtype=dtype)
            local_directions = torch.empty(end - start, 3, dtype=dtype)
            for view_index in local_views.unique(sorted=True).tolist():
                mask = local_views == view_index
                origin, direction = inputs.cameras[view_index].pixel_rays(local_xy[mask])
                local_origins[mask] = origin.to(dtype).expand(int(mask.sum()), -1)
                local_directions[mask] = direction.to(dtype)
            t0, t1 = _ray_box(local_origins, local_directions, lo, hi)
            t0 = t0.clamp_min(self.config.near)
            ray_valid = t1 > t0
            ts = t0[:, None] + (t1 - t0).clamp_min(0)[:, None] * steps[None, :]
            world = local_origins[:, None, :] + ts[:, :, None] * local_directions[:, None, :]
            point_scores = score_world_points(
                inputs,
                world.reshape(-1, 3),
                self.config,
                backends,
            )
            scores = point_scores.score.reshape(end - start, -1)
            scores = scores * ray_valid[:, None]
            sample_consensus_colors = point_scores.consensus_color.reshape(
                end - start,
                -1,
                3,
            )
            if candidate_audit_callback is not None or ray_depth_audit_callback is not None:
                sample_coverages = point_scores.coverage.reshape(end - start, -1)
                sample_color_variances = point_scores.color_variance.reshape(end - start, -1)
                sample_n_seen = point_scores.n_seen.reshape(end - start, -1)
                sample_n_covered = point_scores.n_covered.reshape(end - start, -1)
            if ray_depth_audit_callback is not None:
                ray_depth_audit_callback(
                    CompactRayDepthAuditBatch(
                        candidate_start=start,
                        candidate_end=end,
                        candidate_source_view_indices=local_views.detach().clone(),
                        candidate_source_component_indices=component_ids[start:end]
                        .detach()
                        .clone(),
                        candidate_source_xy=local_xy.detach().clone(),
                        candidate_valid_mask=ray_valid.detach().clone(),
                        depths=ts.detach().clone(),
                        scores=scores.detach().clone(),
                        coverages=sample_coverages.detach().clone(),
                        color_variances=sample_color_variances.detach().clone(),
                        n_seen=sample_n_seen.detach().clone(),
                        n_covered=sample_n_covered.detach().clone(),
                        consensus_colors=sample_consensus_colors.detach().clone(),
                    )
                )
            best_index = scores.argmax(dim=1)
            row = torch.arange(end - start)
            selected_depth = ts[row, best_index]
            selected_score = scores[row, best_index]
            selected_mean = world[row, best_index]
            selected_color = sample_consensus_colors[row, best_index]

            depth_step = (t1 - t0).clamp_min(0) / self.config.samples_per_ray
            near_peak = (
                ts - selected_depth[:, None]
            ).abs() <= self.config.peak_radius_steps * depth_step[:, None]
            local_weight = scores * near_peak
            weight_sum = local_weight.sum(dim=1).clamp_min(torch.finfo(dtype).eps)
            depth_mean = (local_weight * ts).sum(dim=1) / weight_sum
            depth_variance = (local_weight * (ts - depth_mean[:, None]).square()).sum(
                dim=1
            ) / weight_sum
            sigma = depth_variance.clamp_min(0).sqrt()
            sigma = torch.maximum(sigma, 0.25 * depth_step)
            sigma = torch.minimum(sigma, 2.0 * depth_step)

            sl = slice(start, end)
            depths[sl] = selected_depth
            sigmas[sl] = sigma
            means[sl] = selected_mean
            best_scores[sl] = selected_score
            best_colors[sl] = selected_color
            valid_rays[sl] = ray_valid
            if candidate_audit_callback is not None:
                best_depth_indices[sl] = best_index
                best_coverages[sl] = sample_coverages[row, best_index]
                best_color_variances[sl] = sample_color_variances[row, best_index]
                best_n_seen[sl] = sample_n_seen[row, best_index]
                best_n_covered[sl] = sample_n_covered[row, best_index]

                without_best = scores.clone()
                without_best[row, best_index] = -torch.inf
                local_second_best = without_best.max(dim=1).values
                second_best_scores[sl] = local_second_best
                score_margins[sl] = selected_score - local_second_best

                sample_indices = torch.arange(
                    self.config.samples_per_ray,
                    dtype=torch.long,
                )
                half_max = (scores >= 0.5 * selected_score[:, None]) & (selected_score[:, None] > 0)
                before_or_at_best = sample_indices[None, :] <= best_index[:, None]
                at_or_after_best = sample_indices[None, :] >= best_index[:, None]
                left_false = (
                    torch.where(
                        ~half_max & before_or_at_best,
                        sample_indices[None, :],
                        -1,
                    )
                    .max(dim=1)
                    .values
                )
                right_false = (
                    torch.where(
                        ~half_max & at_or_after_best,
                        sample_indices[None, :],
                        self.config.samples_per_ray,
                    )
                    .min(dim=1)
                    .values
                )
                half_max_cells = (right_false - left_false - 1).clamp_min(0)
                local_half_max_width = half_max_cells.to(dtype) * depth_step
                half_max_widths[sl] = torch.where(
                    ray_valid & (selected_score > 0),
                    local_half_max_width,
                    0.0,
                )

            completed_batches += 1
            if progress_callback is not None:
                elapsed = time.perf_counter() - placement_start
                sampled_points = end * self.config.samples_per_ray
                evaluated_pairs, current_chunk, peak_chunk = _placement_counters()
                rate = sampled_points / elapsed if elapsed > 0 else 0.0
                remaining = (total_points - sampled_points) / rate if rate > 0 else 0.0
                progress_callback(
                    CompactPlacementProgress(
                        phase="ray_batch",
                        completed_ray_batches=completed_batches,
                        total_ray_batches=total_ray_batches,
                        sampled_points=sampled_points,
                        completed_point_view_queries=sampled_points * inputs.n_views,
                        evaluated_pairs=evaluated_pairs,
                        index_payload_bytes=index_payload_bytes,
                        index_build_seconds=index_build_seconds,
                        elapsed_seconds=elapsed,
                        points_per_second=rate,
                        estimated_remaining_seconds=remaining,
                        current_pair_chunk=current_chunk,
                        peak_pair_chunk=peak_chunk,
                    )
                )

        placement_seconds = time.perf_counter() - placement_start
        final_evaluated_pairs, _, final_peak_chunk = _placement_counters()
        eligible = valid_rays & (best_scores > self.config.min_score)
        if int(eligible.sum()) < self.config.n_init_3d:
            raise ValueError(
                "compact Carve found fewer globally supported ray placements than n_init_3d; "
                "increase candidate_multiplier or loosen the preregistered support thresholds"
            )
        if self.config.select_all_eligible:
            # Retain every globally supported candidate (one lift per proposed 2D Gaussian) in
            # canonical candidate order; a downstream voxel merge deduplicates redundant views.
            selected = eligible.nonzero(as_tuple=True)[0]
        else:
            selected = _balanced_topk(
                best_scores,
                eligible,
                view_ids,
                self.config.n_init_3d,
                inputs.n_views,
            )
        n_selected = int(selected.numel())

        selected_covariances = torch.empty(
            n_selected,
            3,
            3,
            dtype=dtype,
        )
        selected_depth_sigmas = sigmas[selected].clone()
        selected_ray_sigmas = torch.empty_like(selected_depth_sigmas)
        for view_index in view_ids[selected].unique(sorted=True).tolist():
            output_mask = view_ids[selected] == view_index
            selected_indices = selected[output_mask]
            field = inputs.observations[view_index]
            covariance_2d = _component_covariances(
                field,
                component_ids[selected_indices],
            )
            _, selected_directions = inputs.cameras[view_index].pixel_rays(xy[selected_indices])
            selected_ray_sigmas[output_mask] = selected_depth_sigmas[
                output_mask
            ] * selected_directions.to(dtype).norm(dim=-1)
            selected_covariances[output_mask] = lift_covariance(
                inputs.cameras[view_index],
                xy[selected_indices],
                covariance_2d.to(xy),
                depths[selected_indices].to(xy),
                selected_ray_sigmas[output_mask].to(xy),
            ).to(dtype)

        unclamped_colors = best_colors[selected]
        colors = unclamped_colors.clamp(0.0, 1.0)
        opacity = torch.full(
            (n_selected,),
            self.config.init_opacity,
            dtype=dtype,
        )
        gaussians = Gaussians3D.from_means_covs(
            means=means[selected],
            covs=selected_covariances,
            colors=colors,
            opacity=opacity,
            sh_degree=self.config.sh_degree,
        )
        lineage = CompactLineage(
            source_view_indices=view_ids[selected].clone(),
            source_component_indices=component_ids[selected].clone(),
            source_xy=xy[selected].clone(),
        )
        diagnostics: dict[str, object] = {
            "n_init_2d": list(inputs.n_init_2d),
            "n_opt_2d": list(inputs.n_opt_2d),
            "n_init_3d": gaussians.n,
            "bounds_source": _bounds_source(inputs),
            "bounds_center": center.tolist(),
            "bounds_extent": extent,
            "search_aabb_lower": lo.tolist(),
            "search_aabb_upper": hi.tolist(),
            "anchor_mode": self.config.anchor_mode,
            "selection_mode": (
                "all_eligible" if self.config.select_all_eligible else "balanced_topk"
            ),
            "candidate_count": candidate_count,
            "proposed_candidates_per_view": proposed_per_view,
            "eligible_candidate_count": int(eligible.sum()),
            "anchor_attempt_count": anchor_attempts,
            "query_batch_size": self.config.query_batch_size,
            "query_component_chunk": self.config.query_component_chunk,
            "max_query_pairs": self.config.max_query_pairs,
            "teacher_backend_kinds": [type(backend).__name__ for backend in backends],
            "teacher_index_entries": [
                backend.n_entries if isinstance(backend, GaussianObservationIndex) else None
                for backend in backends
            ],
            "teacher_component_id_dtypes": [
                str(backend.component_id_dtype).removeprefix("torch.")
                if isinstance(backend, GaussianObservationIndex)
                else None
                for backend in backends
            ],
            "placement_evaluated_pairs": final_evaluated_pairs,
            "placement_peak_pair_chunk": final_peak_chunk,
            "placement_index_payload_bytes": list(index_payload_bytes),
            "placement_ray_batches": total_ray_batches,
            "placement_sampled_points": total_points,
            "color_clipped_fraction": float((colors != unclamped_colors).any(dim=1).float().mean()),
        }
        result = CompactInitializationResult(
            gaussians=gaussians,
            lineage=lineage,
            depths=depths[selected].clone(),
            depth_sigmas=selected_depth_sigmas,
            ray_sigmas=selected_ray_sigmas,
            scores=best_scores[selected].clone(),
            diagnostics=diagnostics,
        )
        if candidate_audit_callback is not None:
            candidate_audit_callback(
                CompactCandidateAudit(
                    candidate_source_view_indices=view_ids.detach().clone(),
                    candidate_source_component_indices=component_ids.detach().clone(),
                    candidate_source_xy=xy.detach().clone(),
                    candidate_best_depths=depths.detach().clone(),
                    candidate_depth_sigmas=sigmas.detach().clone(),
                    candidate_best_means=means.detach().clone(),
                    candidate_best_scores=best_scores.detach().clone(),
                    candidate_best_depth_indices=best_depth_indices.detach().clone(),
                    candidate_best_coverages=best_coverages.detach().clone(),
                    candidate_best_color_variances=best_color_variances.detach().clone(),
                    candidate_best_n_seen=best_n_seen.detach().clone(),
                    candidate_best_n_covered=best_n_covered.detach().clone(),
                    candidate_second_best_scores=second_best_scores.detach().clone(),
                    candidate_score_margins=score_margins.detach().clone(),
                    candidate_half_max_widths=half_max_widths.detach().clone(),
                    candidate_consensus_colors=best_colors.detach().clone(),
                    candidate_valid_mask=valid_rays.detach().clone(),
                    candidate_eligible_mask=eligible.detach().clone(),
                    selected_candidate_indices=selected.detach().clone(),
                )
            )
        if progress_callback is not None:
            progress_callback(
                CompactPlacementProgress(
                    phase="complete",
                    completed_ray_batches=completed_batches,
                    total_ray_batches=total_ray_batches,
                    sampled_points=total_points,
                    completed_point_view_queries=total_points * inputs.n_views,
                    evaluated_pairs=final_evaluated_pairs,
                    index_payload_bytes=index_payload_bytes,
                    index_build_seconds=index_build_seconds,
                    elapsed_seconds=placement_seconds,
                    points_per_second=(
                        total_points / placement_seconds if placement_seconds > 0 else 0.0
                    ),
                    estimated_remaining_seconds=0.0,
                    current_pair_chunk=0,
                    peak_pair_chunk=final_peak_chunk,
                )
            )
        return result


def _score_world_points_batch(
    inputs: ReconstructionInputs,
    points: torch.Tensor,
    config: CompactCarveConfig,
    backends: list[ObservationQueryBackend],
) -> CompactPointScores:
    count = points.shape[0]
    dtype = points.dtype
    weight_sum = torch.zeros(count, dtype=dtype)
    color_sum = torch.zeros(count, 3, dtype=dtype)
    color_square_sum = torch.zeros(count, dtype=dtype)
    n_seen = torch.zeros(count, dtype=torch.long)
    n_covered = torch.zeros(count, dtype=torch.long)

    for field, camera, backend in zip(
        inputs.observations,
        inputs.cameras,
        backends,
        strict=True,
    ):
        uv, depth = camera.project(points)
        component_chunk = min(
            config.query_component_chunk,
            max(1, config.max_query_pairs // max(count, 1)),
        )
        query = backend.query(uv, component_chunk=component_chunk)
        seen = (depth > config.near) & query.valid
        area = float(field.fit_window[2] * field.fit_window[3])
        mass = _field_mass(field).to(dtype).clamp_min(torch.finfo(dtype).tiny)
        relative_density = area * query.weight_sum.to(dtype) / mass
        soft_coverage = 1.0 - torch.exp(-relative_density / config.coverage_scale)
        soft_coverage = soft_coverage * seen
        color = query.color.to(dtype)
        weight_sum += soft_coverage
        color_sum += soft_coverage[:, None] * color
        color_square_sum += soft_coverage * color.square().sum(dim=1)
        n_seen += seen.long()
        n_covered += (seen & (soft_coverage >= config.coverage_threshold)).long()

    safe_weight = weight_sum.clamp_min(torch.finfo(dtype).eps)
    consensus = color_sum / safe_weight[:, None]
    variance = (color_square_sum / safe_weight - consensus.square().sum(dim=1)).clamp_min(0.0) / 3.0
    safe_seen = n_seen.clamp_min(1).to(dtype)
    coverage = weight_sum / safe_seen
    hull = (
        (n_seen >= config.min_views)
        & (n_covered >= config.min_views)
        & (n_covered.to(dtype) >= config.hull_fraction * n_seen.to(dtype))
    )
    consistency = torch.exp(-variance / (2.0 * config.color_std_sigma**2))
    score = hull.to(dtype) * coverage * consistency
    return CompactPointScores(
        score=score,
        consensus_color=consensus,
        color_variance=variance,
        coverage=coverage,
        n_seen=n_seen,
        n_covered=n_covered,
    )


def _propose_anchors(
    inputs: ReconstructionInputs,
    config: CompactCarveConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, list[int]]:
    """Return bounded source rays under the configured identity or mass policy.

    ``component_centers`` is the scalable, variable-``m_i`` control: every optimized 2D
    component contributes exactly one ray, independent of its analytic mass.  The explicit cap
    fails before allocating candidate tensors when a producer exports an unexpectedly large set.
    """
    if config.anchor_mode == "mass_random":
        proposed_per_view = _balanced_counts(
            config.n_init_3d * config.candidate_multiplier,
            inputs.n_views,
        )
    else:
        proposed_per_view = [field.n for field in inputs.observations]
    candidate_count = sum(proposed_per_view)
    if candidate_count > config.max_anchor_candidates:
        raise ValueError(
            "compact Carve anchor count exceeds max_anchor_candidates; subsample the 2D sets "
            "explicitly or raise the bounded cap"
        )

    source_views: list[torch.Tensor] = []
    source_components: list[torch.Tensor] = []
    source_points: list[torch.Tensor] = []
    anchor_attempts = 0
    for view_index, (field, count) in enumerate(
        zip(inputs.observations, proposed_per_view, strict=True)
    ):
        if count == 0:
            continue
        if config.anchor_mode == "mass_random":
            xy, component_ids, attempts = _sample_field_anchors(
                field,
                count,
                generator,
                config.max_anchor_rounds,
            )
        else:
            component_ids = torch.arange(field.n, dtype=torch.long)
            xy = field.native_means(dtype=torch.float64)
            attempts = field.n
        source_views.append(torch.full((count,), view_index, dtype=torch.long))
        source_components.append(component_ids)
        source_points.append(xy)
        anchor_attempts += attempts
    return (
        torch.cat(source_views),
        torch.cat(source_components),
        torch.cat(source_points),
        anchor_attempts,
        proposed_per_view,
    )


def _sample_field_anchors(
    field: GaussianObservationField,
    count: int,
    generator: torch.Generator,
    max_rounds: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Draw exact-support Gaussian anchors, retaining lineage but no RGB targets."""
    masses = field.amplitudes * (2.0 * math.pi) * field.effective_variances().prod(dim=1).sqrt()
    total_mass = masses.sum()
    if not bool(torch.isfinite(total_mass)) or float(total_mass) <= 0:
        raise ValueError("compact observation has no positive proposal mass")
    cumulative = masses.cumsum(0)
    accepted_xy: list[torch.Tensor] = []
    accepted_ids: list[torch.Tensor] = []
    attempts = 0
    remaining = count
    for _ in range(max_rounds):
        if remaining <= 0:
            break
        draw_count = max(remaining * 2, 8)
        uniforms = torch.rand(draw_count, generator=generator, dtype=field.dtype) * total_mass
        component_ids = torch.searchsorted(cumulative, uniforms, right=False).clamp_max(field.n - 1)
        normal = torch.randn(draw_count, 2, generator=generator, dtype=field.dtype)
        scales = field.effective_variances()[component_ids].sqrt()
        local = normal * scales
        theta = field.rotations[component_ids]
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        offsets = torch.stack(
            [
                cos * local[:, 0] - sin * local[:, 1],
                sin * local[:, 0] + cos * local[:, 1],
            ],
            dim=-1,
        )
        if field.mean_residuals is None:
            xy = field.means[component_ids] + offsets
        else:
            fit_x, fit_y, _, _ = field.fit_window
            origin = torch.tensor([fit_x + 0.5, fit_y + 0.5], dtype=torch.float64)
            xy = field.local_means(component_ids).double() + offsets.double() + origin
        base_weight = field.amplitudes[component_ids] * torch.exp(-0.5 * normal.square().sum(dim=1))
        exact_weight = field.component_weight(xy, component_ids)
        acceptance = (exact_weight / base_weight.clamp_min(torch.finfo(field.dtype).tiny)).clamp(
            0.0, 1.0
        )
        active = torch.rand(draw_count, generator=generator, dtype=field.dtype) < acceptance
        active_indices = active.nonzero(as_tuple=True)[0][:remaining]
        if active_indices.numel():
            accepted_xy.append(xy[active_indices])
            accepted_ids.append(component_ids[active_indices])
            remaining -= active_indices.numel()
        attempts += draw_count
    if remaining > 0:
        raise ValueError(
            "compact observation proposal could not draw enough in-window support anchors; "
            "increase max_anchor_rounds or inspect off-window components"
        )
    return torch.cat(accepted_xy), torch.cat(accepted_ids), attempts


def _field_mass(field: GaussianObservationField) -> torch.Tensor:
    return (
        field.amplitudes * (2.0 * math.pi) * field.effective_variances().prod(dim=1).sqrt()
    ).sum()


def _component_covariances(
    field: GaussianObservationField,
    component_ids: torch.Tensor,
) -> torch.Tensor:
    variances = field.effective_variances()[component_ids]
    theta = field.rotations[component_ids]
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    cov00 = cos.square() * variances[:, 0] + sin.square() * variances[:, 1]
    cov01 = cos * sin * (variances[:, 0] - variances[:, 1])
    cov11 = sin.square() * variances[:, 0] + cos.square() * variances[:, 1]
    return torch.stack(
        [
            torch.stack([cov00, cov01], dim=-1),
            torch.stack([cov01, cov11], dim=-1),
        ],
        dim=-2,
    )


def _balanced_counts(total: int, groups: int) -> list[int]:
    base, remainder = divmod(total, groups)
    return [base + int(index < remainder) for index in range(groups)]


def _balanced_topk(
    scores: torch.Tensor,
    eligible: torch.Tensor,
    view_ids: torch.Tensor,
    count: int,
    n_views: int,
) -> torch.Tensor:
    quotas = _balanced_counts(count, n_views)
    chosen: list[torch.Tensor] = []
    used = torch.zeros_like(eligible)
    for view_index, quota in enumerate(quotas):
        candidates = ((view_ids == view_index) & eligible).nonzero(as_tuple=True)[0]
        take = min(quota, candidates.numel())
        if take:
            order = torch.argsort(scores[candidates], descending=True, stable=True)
            local = candidates[order[:take]]
            chosen.append(local)
            used[local] = True
    selected_count = sum(item.numel() for item in chosen)
    if selected_count < count:
        candidates = (eligible & ~used).nonzero(as_tuple=True)[0]
        order = torch.argsort(scores[candidates], descending=True, stable=True)
        chosen.append(candidates[order[: count - selected_count]])
    return torch.cat(chosen)


def _ray_box(
    origins: torch.Tensor,
    directions: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    direction_safe = torch.where(
        directions.abs() < 1e-9,
        torch.full_like(directions, 1e-9),
        directions,
    )
    entry = (lo[None, :] - origins) / direction_safe
    exit = (hi[None, :] - origins) / direction_safe
    near = torch.minimum(entry, exit).max(dim=-1).values
    far = torch.maximum(entry, exit).min(dim=-1).values
    return near.clamp_min(0.0), far


def _center_and_extent(
    inputs: ReconstructionInputs,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, float]:
    if inputs.bounds_hint is not None:
        center, extent = inputs.bounds_hint
        return center.to(dtype=dtype), float(extent)
    positions = torch.stack([camera.position for camera in inputs.cameras]).to(dtype)
    finite = (
        inputs.points[torch.isfinite(inputs.points).all(dim=-1)].to(dtype)
        if inputs.points is not None
        else positions.new_empty((0, 3))
    )
    if finite.shape[0] >= 4:
        if finite.shape[0] >= 20:
            lo = torch.quantile(finite, 0.01, dim=0)
            hi = torch.quantile(finite, 0.99, dim=0)
        else:
            lo, hi = finite.amin(dim=0), finite.amax(dim=0)
        center = 0.5 * (lo + hi)
        radii = (finite - center).norm(dim=-1)
        radius = torch.quantile(radii, 0.99) if radii.numel() >= 20 else radii.max()
    else:
        forwards = torch.stack([camera.R[2] for camera in inputs.cameras]).to(dtype)
        forwards = torch.nn.functional.normalize(forwards, dim=-1)
        eye = torch.eye(3, dtype=dtype)
        projectors = eye[None] - forwards[:, :, None] * forwards[:, None, :]
        matrix = projectors.sum(dim=0)
        vector = (projectors @ positions[:, :, None]).sum(dim=0)[:, 0]
        center = _solve_projector_center(matrix, vector)
        radius = 0.25 * (positions - center).norm(dim=-1).median().clamp_min(1e-3)
    return center, float((2.2 * radius).clamp_min(1e-3))


def _bounds_source(inputs: ReconstructionInputs) -> str:
    if inputs.bounds_hint is not None:
        return "explicit_hint"
    if inputs.points is not None:
        finite = torch.isfinite(inputs.points).all(dim=-1)
        if int(finite.sum()) >= 4:
            return "sparse_points"
    return "camera_axis_fallback"


def _solve_projector_center(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Solve the camera-projector system deterministically, including rank-deficient rigs.

    PyTorch's default CPU least-squares driver is ``gelsy``.  It can return byte-distinct
    float32 solutions for an identical small system.  ``gels`` preserves the historical
    full-rank solution but assumes full rank, so first use rank-aware ``gelsd`` and retain its
    minimum-norm solution when the camera layout is degenerate.
    """
    rank_aware = torch.linalg.lstsq(matrix, vector, driver="gelsd")
    if int(rank_aware.rank.item()) == matrix.shape[-1]:
        return torch.linalg.lstsq(matrix, vector, driver="gels").solution
    return rank_aware.solution


def _validate_cpu_inputs(inputs: ReconstructionInputs) -> None:
    inputs.validate()
    dtypes = {field.dtype for field in inputs.observations}
    if len(dtypes) != 1:
        raise ValueError("all compact observations must share one dtype")
    if any(field.device.type != "cpu" for field in inputs.observations):
        raise ValueError("compact Carve CPU reference requires CPU observation fields")
    if any(camera.R.device.type != "cpu" for camera in inputs.cameras):
        raise ValueError("compact Carve CPU reference requires CPU cameras")


__all__ = [
    "AnchorMode",
    "CompactCandidateAudit",
    "CompactCandidateAuditCallback",
    "CompactCarveConfig",
    "CompactCarveInitializer",
    "CompactInitializationResult",
    "CompactLineage",
    "CompactPlacementProgress",
    "CompactPlacementProgressCallback",
    "CompactPointScores",
    "CompactRayDepthAuditBatch",
    "CompactRayDepthAuditCallback",
    "make_placement_progress_printer",
    "score_world_points",
]
