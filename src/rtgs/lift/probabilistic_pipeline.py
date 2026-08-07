"""Opt-in probabilistic compact-field pipeline and independent-half stability audit.

The main reconstruction delegates to :class:`rtgs.lift.field_lifter.FieldLifter`.  This module
adds only orchestration that would otherwise be easy to implement inconsistently: a leak-free
training-camera half split and a world-frame, mutual-nearest stability report.  Stability is not
accuracy or resolution and never consumes held-out fields for fitting or matching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from rtgs.data.field_inputs import SceneFits
from rtgs.lift.field_lifter import FieldLiftConfig, FieldLifter, FieldLiftResult


@dataclass(frozen=True)
class ProbabilisticFieldPipelineConfig:
    """Controls for the main field lift and optional independent-half audit."""

    lift: FieldLiftConfig = field(default_factory=FieldLiftConfig)
    independent_half_validation: bool = False
    minimum_half_views: int = 2
    match_radius_fraction: float = 0.10

    def __post_init__(self) -> None:
        if not isinstance(self.lift, FieldLiftConfig):
            raise TypeError("lift must be FieldLiftConfig")
        if not isinstance(self.independent_half_validation, bool):
            raise TypeError("independent_half_validation must be a bool")
        if isinstance(self.minimum_half_views, bool) or not isinstance(
            self.minimum_half_views, int
        ):
            raise TypeError("minimum_half_views must be an integer")
        if self.minimum_half_views <= 0:
            raise ValueError("minimum_half_views must be positive")
        if not math.isfinite(self.match_radius_fraction) or not 0 < self.match_radius_fraction <= 1:
            raise ValueError("match_radius_fraction must be finite and in (0,1]")


@dataclass(frozen=True)
class IndependentHalfStability:
    """World-frame agreement diagnostics for two disjoint training-camera fits."""

    first_train_views: tuple[int, ...]
    second_train_views: tuple[int, ...]
    first_count: int
    second_count: int
    mutual_matches: int
    first_matched_fraction: float
    second_matched_fraction: float
    match_radius: float
    center_median: float
    center_p90: float
    center_rmse: float
    covariance_relative_median: float


@dataclass(frozen=True)
class ProbabilisticFieldPipelineResult:
    """Main reconstruction plus optional independent-half outputs and stability."""

    reconstruction: FieldLiftResult
    half_reconstructions: tuple[FieldLiftResult, FieldLiftResult] | None
    stability: IndependentHalfStability | None


def _training_halves(
    train_view_indices: tuple[int, ...],
    *,
    minimum: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    first = train_view_indices[::2]
    second = train_view_indices[1::2]
    if len(first) < minimum or len(second) < minimum:
        raise ValueError(
            "independent-half validation requires at least "
            f"{2 * minimum} training views, got {len(train_view_indices)}"
        )
    return first, second


def _scene_with_training_half(fits: SceneFits, train: tuple[int, ...]) -> SceneFits:
    """Copy compact inputs while removing geometry that was not derived from this half."""

    train_set = set(train)
    heldout = tuple(index for index in range(fits.n_views) if index not in train_set)
    return SceneFits(
        observations=fits.observations,
        cameras=fits.cameras,
        view_names=fits.view_names,
        alphas=fits.alphas,
        train_view_indices=train,
        heldout_view_indices=heldout,
        # Even view-indexed priors may have been produced by a multi-view method.  Without
        # per-prior provenance proving half-local derivation, drop them with the shared bounds,
        # points, and neighbor graph instead of leaking common geometry across both fits.
        depth_priors=None,
        depth_confidences=None,
        neighbors=None,
        points=None,
        point_visibility=None,
        bounds_hint=None,
        geometry_is_train_only=False,
        name=f"{fits.name}-half-{'-'.join(map(str, train))}",
    )


def _validate_realized_fit_split(result: FieldLiftResult, fits: SceneFits) -> None:
    """Fail closed unless a realized fit used exactly its declared train/reporting partition."""

    optimized = set(result.optimized_view_indices)
    train = set(fits.train_view_indices)
    heldout = set(fits.heldout_view_indices)
    if optimized & heldout or not optimized <= train or set(result.heldout_view_indices) != heldout:
        raise RuntimeError("realized probabilistic-field fit violated its camera partition")


def _mutual_matches(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    radius_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    first64 = first.detach().cpu().to(torch.float64)
    second64 = second.detach().cpu().to(torch.float64)
    joined = torch.cat([first64, second64])
    extent = float(torch.linalg.vector_norm(joined.amax(0) - joined.amin(0)))
    radius = radius_fraction * max(extent, torch.finfo(torch.float64).eps)
    distances = torch.cdist(first64, second64)
    first_distance, first_to_second = distances.min(dim=1)
    _second_distance, second_to_first = distances.min(dim=0)
    first_indices = torch.arange(first64.shape[0])
    mutual = second_to_first[first_to_second] == first_indices
    accepted = mutual & (first_distance <= radius)
    left = first_indices[accepted]
    right = first_to_second[accepted]
    return left, right, first_distance[accepted], radius


def compare_independent_halves(
    first: FieldLiftResult,
    second: FieldLiftResult,
    *,
    first_train_views: tuple[int, ...],
    second_train_views: tuple[int, ...],
    match_radius_fraction: float,
) -> IndependentHalfStability:
    """Compare two calibrated-world outputs without assigning correctness."""

    left, right, center_distance, radius = _mutual_matches(
        first.gaussians.means,
        second.gaussians.means,
        radius_fraction=match_radius_fraction,
    )
    match_count = int(left.numel())
    if match_count:
        center_median = float(center_distance.median())
        center_p90 = float(torch.quantile(center_distance, 0.90))
        center_rmse = float(center_distance.square().mean().sqrt())
        first_covariance = first.gaussians.covariance().detach().cpu().to(torch.float64)[left]
        second_covariance = second.gaussians.covariance().detach().cpu().to(torch.float64)[right]
        denominator = (
            (
                torch.linalg.matrix_norm(first_covariance)
                * torch.linalg.matrix_norm(second_covariance)
            )
            .sqrt()
            .clamp_min(torch.finfo(torch.float64).tiny)
        )
        covariance_relative = torch.linalg.matrix_norm(first_covariance - second_covariance) / (
            denominator
        )
        covariance_relative_median = float(covariance_relative.median())
    else:
        center_median = math.inf
        center_p90 = math.inf
        center_rmse = math.inf
        covariance_relative_median = math.inf
    return IndependentHalfStability(
        first_train_views=first_train_views,
        second_train_views=second_train_views,
        first_count=first.gaussians.n,
        second_count=second.gaussians.n,
        mutual_matches=match_count,
        first_matched_fraction=match_count / first.gaussians.n,
        second_matched_fraction=match_count / second.gaussians.n,
        match_radius=radius,
        center_median=center_median,
        center_p90=center_p90,
        center_rmse=center_rmse,
        covariance_relative_median=covariance_relative_median,
    )


def run_probabilistic_field_pipeline(
    fits: SceneFits,
    config: ProbabilisticFieldPipelineConfig | None = None,
) -> ProbabilisticFieldPipelineResult:
    """Run the opt-in field pipeline and optional leak-free half-camera stability audit."""

    if not isinstance(fits, SceneFits):
        raise TypeError("fits must be SceneFits")
    config = ProbabilisticFieldPipelineConfig() if config is None else config
    if not isinstance(config, ProbabilisticFieldPipelineConfig):
        raise TypeError("config must be ProbabilisticFieldPipelineConfig")
    halves: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    if config.independent_half_validation:
        halves = _training_halves(
            fits.train_view_indices,
            minimum=config.minimum_half_views,
        )
    reconstruction = FieldLifter(config.lift).fit(fits)
    _validate_realized_fit_split(reconstruction, fits)
    if halves is None:
        return ProbabilisticFieldPipelineResult(reconstruction, None, None)
    first_views, second_views = halves
    first_fits = _scene_with_training_half(fits, first_views)
    second_fits = _scene_with_training_half(fits, second_views)
    first = FieldLifter(config.lift).fit(first_fits)
    _validate_realized_fit_split(first, first_fits)
    second = FieldLifter(config.lift).fit(second_fits)
    _validate_realized_fit_split(second, second_fits)
    stability = compare_independent_halves(
        first,
        second,
        first_train_views=first_views,
        second_train_views=second_views,
        match_radius_fraction=config.match_radius_fraction,
    )
    return ProbabilisticFieldPipelineResult(
        reconstruction=reconstruction,
        half_reconstructions=(first, second),
        stability=stability,
    )


__all__ = [
    "IndependentHalfStability",
    "ProbabilisticFieldPipelineConfig",
    "ProbabilisticFieldPipelineResult",
    "compare_independent_halves",
    "run_probabilistic_field_pipeline",
]
