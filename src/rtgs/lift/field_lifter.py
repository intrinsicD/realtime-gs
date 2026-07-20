"""Image-free field-level 2D-to-3D Gaussian lifting.

This is the Stage-2 orchestrator described in ``docs/DESIGN_field_lift.md``.  It reuses the
existing compact ray sweep for placement, rebuilds every selected primitive as an exact
source-projection fiber, performs decomposition-free analytic field refitting, and emits
post-hoc soft overlap correspondences.  Compact normalized teachers are never mislabeled as
additive RGB mixtures: optimization uses their underlying density/RGB-numerator proxy and the
separate validation module evaluates the frozen renderer semantics.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.compact_views import CompactDataset, PackedAlpha
from rtgs.data.field_inputs import SceneFits
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.data.scene import SceneData
from rtgs.lift.compact_carve import (
    CompactCarveConfig,
    CompactCarveInitializer,
    CompactInitializationResult,
    CompactLineage,
)
from rtgs.lift.field_loss import (
    AnalyticGaussianField2D,
    mixture_inner_product,
    peak_gaussian_product_integrals,
)
from rtgs.lift.field_refit import (
    FieldRefitConfig,
    FieldRefitResult,
    fit_field_fibers,
)
from rtgs.lift.field_topology import (
    DeterministicTopologyScheduler,
    FieldComponent,
    FieldComponentPayload,
    FieldTopologyState,
    MoveReceipt,
    SourceAnchor,
    SourceLineage,
    TopologyOps,
    propose_birth,
    propose_merge,
    propose_prune,
    propose_split,
)
from rtgs.lift.field_validation import (
    FieldValidationConfig,
    FieldValidationResult,
    validate_field_semantics,
)
from rtgs.lift.field_visibility import center_transmittance_visibility
from rtgs.lift.inverse_projection_fiber import InverseProjectionFiber
from rtgs.render.projection import EWA_DILATION, project_gaussians_ewa


@dataclass(frozen=True)
class FieldLiftConfig:
    """CPU-bounded controls for placement, refit, and correspondence output."""

    max_tracks: int = 128
    max_train_views: int = 8
    depth_samples: int = 32
    candidate_multiplier: int = 3
    min_views: int = 2
    robust_view_fraction: float = 0.60
    min_placement_score: float = 0.01
    projection_dilation: float = EWA_DILATION
    init_opacity: float = 0.10
    background_fraction: float = 0.05
    background_ray_scale: float = 2.0
    depth_prior_window: float = 0.25
    correspondence_dustbin: float = 1e-8
    topology_rounds: int = 1
    topology_merge_candidates: int = 4
    parsimony_per_component: float = 1e-4
    validation_sample_cap: int = 128
    seed: int = 0
    refit: FieldRefitConfig = field(default_factory=FieldRefitConfig)

    def __post_init__(self) -> None:
        for name in (
            "max_tracks",
            "max_train_views",
            "depth_samples",
            "candidate_multiplier",
            "min_views",
            "validation_sample_cap",
            "seed",
            "topology_rounds",
            "topology_merge_candidates",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if (
            self.max_tracks <= 0
            or self.max_train_views <= 0
            or self.depth_samples < 2
            or self.candidate_multiplier <= 0
            or self.min_views <= 0
            or self.validation_sample_cap <= 0
            or self.seed < 0
            or self.topology_rounds < 0
            or self.topology_merge_candidates <= 0
        ):
            raise ValueError("integer field-lift controls are outside their valid range")
        for name in (
            "robust_view_fraction",
            "min_placement_score",
            "projection_dilation",
            "init_opacity",
            "background_fraction",
            "background_ray_scale",
            "depth_prior_window",
            "correspondence_dustbin",
            "parsimony_per_component",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < self.robust_view_fraction <= 1.0:
            raise ValueError("robust_view_fraction must be in (0,1]")
        if self.min_placement_score < 0 or self.projection_dilation < 0:
            raise ValueError("placement score and projection dilation must be non-negative")
        if not 0.0 < self.init_opacity < 1.0:
            raise ValueError("init_opacity must be in (0,1)")
        if not 0.0 <= self.background_fraction < 1.0:
            raise ValueError("background_fraction must be in [0,1)")
        if self.background_ray_scale < 1.0:
            raise ValueError("background_ray_scale must be at least one")
        if not 0.0 < self.depth_prior_window < 1.0:
            raise ValueError("depth_prior_window must be in (0,1)")
        if self.correspondence_dustbin < 0:
            raise ValueError("correspondence_dustbin must be non-negative")
        if self.parsimony_per_component < 0:
            raise ValueError("parsimony_per_component must be non-negative")
        if not isinstance(self.refit, FieldRefitConfig):
            raise TypeError("refit must be FieldRefitConfig")


@dataclass(frozen=True)
class FieldPlacementResult:
    """Exact-fiber placement state before continuous fitting."""

    fiber: InverseProjectionFiber
    field_masses: torch.Tensor
    source_colors: torch.Tensor
    render_opacity: torch.Tensor
    scores: torch.Tensor
    source_global_view_indices: torch.Tensor
    background_mask: torch.Tensor
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class FieldLiftResult:
    """Complete image-free Stage-2 output and reproducibility diagnostics."""

    gaussians: Gaussians3D
    placement: FieldPlacementResult
    refit: FieldRefitResult
    correspondences: tuple[torch.Tensor, ...]
    correspondence_visibility: torch.Tensor
    topology_receipts: tuple[MoveReceipt, ...]
    semantic_validation: FieldValidationResult
    optimized_view_indices: tuple[int, ...]
    heldout_view_indices: tuple[int, ...]
    diagnostics: dict[str, object]


def _even_subset(indices: Sequence[int], maximum: int) -> tuple[int, ...]:
    values = tuple(indices)
    if len(values) <= maximum:
        return values
    positions = torch.linspace(0, len(values) - 1, maximum).round().long()
    return tuple(values[int(position)] for position in positions)


def _component_covariances(field: GaussianObservationField) -> torch.Tensor:
    variances = field.effective_variances()
    theta = field.rotations
    cos = theta.cos()
    sin = theta.sin()
    covariance = torch.stack(
        [
            cos.square() * variances[:, 0] + sin.square() * variances[:, 1],
            cos * sin * (variances[:, 0] - variances[:, 1]),
            cos * sin * (variances[:, 0] - variances[:, 1]),
            sin.square() * variances[:, 0] + cos.square() * variances[:, 1],
        ],
        dim=-1,
    )
    return covariance.reshape(-1, 2, 2)


def _analytic_field(
    field: GaussianObservationField,
    *,
    dtype: torch.dtype | None = None,
) -> AnalyticGaussianField2D:
    # Affine colors are intentionally reduced to their component-center numerator only in this
    # analytic proxy. Frozen teacher validation retains the full affine renderer semantics.
    target_dtype = field.dtype if dtype is None else dtype
    means = field.native_means(dtype=target_dtype)
    covariance = _component_covariances(field).to(dtype=target_dtype)
    density = field.amplitudes.to(dtype=target_dtype)
    return AnalyticGaussianField2D(
        means=means,
        covariances=covariance,
        density_amplitudes=density,
        rgb_amplitudes=density[:, None] * field.colors.to(dtype=target_dtype),
    )


def _axis_bounds(
    cameras: Sequence,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, float]:
    positions = torch.stack([camera.position for camera in cameras]).to(dtype)
    forwards = torch.stack([camera.R[2] for camera in cameras]).to(dtype)
    forwards = torch.nn.functional.normalize(forwards, dim=-1)
    eye = torch.eye(3, dtype=dtype)
    projectors = eye[None] - forwards[:, :, None] * forwards[:, None, :]
    matrix = projectors.sum(dim=0)
    vector = (projectors @ positions[:, :, None]).sum(dim=0)[:, 0]
    center = torch.linalg.lstsq(matrix, vector, driver="gelsd").solution
    radius = 0.5 * (positions - center).norm(dim=-1).median().clamp_min(1e-3)
    return center, float((2.2 * radius).clamp_min(1e-3))


def _frustum_consensus_bounds(
    cameras: Sequence,
    dtype: torch.dtype,
    *,
    min_views: int,
) -> tuple[torch.Tensor, float]:
    """Bound camera-only scenes by a deterministic coarse frustum intersection."""

    center, extent = _axis_bounds(cameras, dtype)
    coordinates = torch.linspace(-0.5, 0.5, 11, dtype=dtype)
    grid = torch.cartesian_prod(coordinates, coordinates, coordinates)
    points = center[None] + extent * grid
    support = torch.zeros(points.shape[0], dtype=torch.long)
    for camera in cameras:
        uv, depth = camera.project(points)
        support += ((depth > 0.05) & camera.in_image(uv)).long()
    required = min(max(min_views, 1), len(cameras))
    selected = points[support >= required]
    if selected.shape[0] < 8:
        return center, extent
    lo = selected.amin(dim=0)
    hi = selected.amax(dim=0)
    consensus_center = 0.5 * (lo + hi)
    consensus_extent = float((hi - lo).amax().clamp_min(1e-3))
    return consensus_center, consensus_extent


def _training_inputs(
    fits: SceneFits,
    selected: Sequence[int],
) -> tuple[ReconstructionInputs, str]:
    selected_tuple = tuple(selected)
    geometry_is_safe = fits.geometry_is_train_only or not fits.heldout_view_indices
    if geometry_is_safe and fits.bounds_hint is not None:
        bounds = fits.bounds_hint
        source = (
            "explicit_train_only_hint"
            if fits.geometry_is_train_only
            else "all_views_are_training_hint"
        )
    elif geometry_is_safe and fits.points is not None and fits.points.shape[0] >= 4:
        finite = fits.points[torch.isfinite(fits.points).all(dim=-1)]
        lo = torch.quantile(finite, 0.01, dim=0) if finite.shape[0] >= 20 else finite.amin(dim=0)
        hi = torch.quantile(finite, 0.99, dim=0) if finite.shape[0] >= 20 else finite.amax(dim=0)
        bounds = (0.5 * (lo + hi), float((hi - lo).amax().clamp_min(1e-3)))
        source = "sparse_point_percentiles"
    else:
        dtype = fits.observations[selected_tuple[0]].dtype
        bounds = _frustum_consensus_bounds(
            [fits.cameras[index] for index in selected_tuple],
            dtype,
            min_views=min(2, len(selected_tuple)),
        )
        source = "frustum_consensus"
    return (
        ReconstructionInputs(
            observations=[fits.observations[index] for index in selected_tuple],
            cameras=[fits.cameras[index] for index in selected_tuple],
            view_names=[fits.view_names[index] for index in selected_tuple],
            point_visibility=(
                None
                if not geometry_is_safe or fits.point_visibility is None
                else [fits.point_visibility[index] for index in selected_tuple]
            ),
            # Unverified geometry is retained by SceneFits for reporting but never enters a
            # held-out fit. The training-camera frustum above supplies the safe working volume.
            points=fits.points if geometry_is_safe else None,
            bounds_hint=bounds,
            name=fits.name,
        ),
        source,
    )


def _alpha_mask(
    alpha: PackedAlpha | torch.Tensor | None,
    field: GaussianObservationField,
) -> torch.Tensor | None:
    if alpha is None:
        return None
    if isinstance(alpha, PackedAlpha):
        return alpha.full_mask((field.height, field.width))
    return alpha.bool()


def _alpha_valid_sources(
    fits: SceneFits,
    global_views: torch.Tensor,
    components: torch.Tensor,
    xy: torch.Tensor,
) -> torch.Tensor:
    valid = torch.ones(global_views.shape[0], dtype=torch.bool)
    for global_view in global_views.unique(sorted=True).tolist():
        rows = global_views == global_view
        mask = _alpha_mask(fits.alphas[global_view], fits.observations[global_view])
        if mask is None:
            continue
        local_xy = xy[rows]
        x = local_xy[:, 0].floor().long().clamp(0, mask.shape[1] - 1)
        y = local_xy[:, 1].floor().long().clamp(0, mask.shape[0] - 1)
        valid[rows] = mask[y, x]
    return valid


def _fallback_initialization(
    inputs: ReconstructionInputs,
    *,
    count: int,
    opacity: float,
) -> CompactInitializationResult:
    """Deterministic random-in-bounds floor using high-mass component-center rays."""

    candidates: list[tuple[float, int, int]] = []
    for view, observation in enumerate(inputs.observations):
        order = torch.argsort(observation.amplitudes, descending=True, stable=True)
        candidates.extend(
            (float(observation.amplitudes[index]), view, int(index)) for index in order
        )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    chosen = candidates[:count]
    if not chosen:
        raise ValueError("field placement requires at least one positive source component")
    view_indices = torch.tensor([item[1] for item in chosen], dtype=torch.long)
    component_indices = torch.tensor([item[2] for item in chosen], dtype=torch.long)
    xy = torch.stack(
        [
            inputs.observations[view].native_means(dtype=torch.float32)[component]
            for _mass, view, component in chosen
        ]
    )
    assert inputs.bounds_hint is not None
    center, extent = inputs.bounds_hint
    means: list[torch.Tensor] = []
    depths: list[torch.Tensor] = []
    colors: list[torch.Tensor] = []
    covariances: list[torch.Tensor] = []
    for row, (_mass, view, component) in enumerate(chosen):
        camera = inputs.cameras[view]
        origin, direction = camera.pixel_rays(xy[row : row + 1])
        target_depth = camera.project(center[None].to(xy))[1][0].clamp_min(0.1)
        depth = target_depth + 0.0 * extent
        means.append(origin.to(xy) + depth * direction[0])
        depths.append(depth)
        observation = inputs.observations[view]
        colors.append(observation.colors[component].clamp(0.0, 1.0))
        covariance2d = _component_covariances(observation)[component : component + 1]
        from rtgs.lift.base import lift_covariance

        covariances.append(
            lift_covariance(
                camera,
                xy[row : row + 1],
                covariance2d.to(xy),
                depth.reshape(1),
                (0.1 * depth).reshape(1),
            )[0]
        )
    means_tensor = torch.stack(means)
    depths_tensor = torch.stack(depths)
    covariance_tensor = torch.stack(covariances)
    colors_tensor = torch.stack(colors).to(means_tensor)
    opacity_tensor = torch.full((len(chosen),), opacity, dtype=means_tensor.dtype)
    gaussians = Gaussians3D.from_means_covs(
        means_tensor,
        covariance_tensor,
        colors_tensor,
        opacity_tensor,
    )
    zeros = torch.zeros(len(chosen), dtype=means_tensor.dtype)
    return CompactInitializationResult(
        gaussians=gaussians,
        lineage=CompactLineage(view_indices, component_indices, xy),
        depths=depths_tensor,
        depth_sigmas=0.1 * depths_tensor,
        ray_sigmas=0.1 * depths_tensor,
        scores=zeros,
        diagnostics={"fallback": "deterministic_bounds_midpoint"},
    )


def _ray_depth_bounds(
    inputs: ReconstructionInputs,
    view_indices: torch.Tensor,
    xy: torch.Tensor,
    initial_depths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert inputs.bounds_hint is not None
    center, extent = inputs.bounds_hint
    lo = center.to(initial_depths) - 0.5 * extent
    hi = center.to(initial_depths) + 0.5 * extent
    lower = torch.empty_like(initial_depths)
    upper = torch.empty_like(initial_depths)
    for view in view_indices.unique(sorted=True).tolist():
        rows = view_indices == view
        origin, direction = inputs.cameras[view].pixel_rays(xy[rows])
        origin = origin.to(initial_depths)
        direction = direction.to(initial_depths)
        safe = torch.where(
            direction.abs() < 1e-9,
            torch.full_like(direction, 1e-9),
            direction,
        )
        entry = (lo[None] - origin[None]) / safe
        exit = (hi[None] - origin[None]) / safe
        local_lower = torch.minimum(entry, exit).amax(dim=-1).clamp_min(0.05)
        local_upper = torch.maximum(entry, exit).amin(dim=-1)
        invalid = local_upper <= local_lower
        local_lower = torch.where(invalid, 0.5 * initial_depths[rows], local_lower)
        local_upper = torch.where(invalid, 1.5 * initial_depths[rows], local_upper)
        lower[rows] = local_lower
        upper[rows] = local_upper
    margin = (upper - lower).clamp_min(1e-4) * 1e-4
    lower = torch.minimum(lower, initial_depths - margin).clamp_min(1e-4)
    upper = torch.maximum(upper, initial_depths + margin)
    return lower, upper


def _apply_depth_priors(
    fits: SceneFits,
    global_views: torch.Tensor,
    components: torch.Tensor,
    depths: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    window: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if fits.depth_priors is None or fits.depth_confidences is None:
        return depths, lower, upper
    result = depths.clone()
    lo = lower.clone()
    hi = upper.clone()
    for global_view in global_views.unique(sorted=True).tolist():
        prior = fits.depth_priors[global_view]
        confidence = fits.depth_confidences[global_view]
        if prior is None or confidence is None:
            continue
        rows = global_views == global_view
        component_ids = components[rows]
        local_prior = prior[component_ids].to(result)
        local_confidence = confidence[component_ids].to(result)
        result[rows] = (1.0 - local_confidence) * result[rows] + local_confidence * local_prior
        half = window * local_prior
        lo[rows] = torch.where(
            local_confidence > 0,
            torch.maximum(lo[rows], local_prior - half),
            lo[rows],
        )
        hi[rows] = torch.where(
            local_confidence > 0,
            torch.minimum(hi[rows], local_prior + half),
            hi[rows],
        )
    invalid = hi <= lo
    lo = torch.where(invalid, lower, lo)
    hi = torch.where(invalid, upper, hi)
    margin = (hi - lo).clamp_min(1e-4) * 1e-4
    result = torch.maximum(lo + margin, torch.minimum(result, hi - margin))
    return result, lo, hi


def _apply_sparse_point_priors(
    inputs: ReconstructionInputs,
    local_views: torch.Tensor,
    components: torch.Tensor,
    xy: torch.Tensor,
    depths: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    window: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Seed rays from nearest trusted visible SfM points inside the source footprint."""

    if inputs.points is None or inputs.point_visibility is None:
        return depths, lower, upper, 0
    result = depths.clone()
    lo = lower.clone()
    hi = upper.clone()
    anchored = 0
    for local_view in local_views.unique(sorted=True).tolist():
        rows = local_views == local_view
        visible_indices = inputs.point_visibility[local_view]
        if visible_indices.numel() == 0:
            continue
        points = inputs.points[visible_indices].to(result)
        projected, point_depths = inputs.cameras[local_view].project(points)
        projected = projected.to(result)
        point_depths = point_depths.to(result)
        valid_points = (point_depths > 0.05) & inputs.cameras[local_view].in_image(projected)
        if not bool(valid_points.any()):
            continue
        projected = projected[valid_points]
        point_depths = point_depths[valid_points]
        distances = torch.cdist(xy[rows].to(result), projected)
        nearest_distance, nearest_index = distances.min(dim=1)
        covariance = _component_covariances(inputs.observations[local_view])[components[rows]].to(
            result
        )
        footprint_sigma = torch.linalg.eigvalsh(covariance).amax(dim=-1).sqrt().clamp_min(0.5)
        confidence = torch.exp(-0.5 * (nearest_distance / footprint_sigma).square())
        confidence = torch.where(
            nearest_distance <= 3.0 * footprint_sigma,
            confidence,
            torch.zeros_like(confidence),
        )
        nearest_depth = point_depths[nearest_index]
        result[rows] = (1.0 - confidence) * result[rows] + confidence * nearest_depth
        half_window = window * nearest_depth / confidence.clamp_min(0.25)
        local_lo = torch.maximum(lo[rows], nearest_depth - half_window)
        local_hi = torch.minimum(hi[rows], nearest_depth + half_window)
        usable = (confidence > 0) & (local_hi > local_lo)
        lo[rows] = torch.where(usable, local_lo, lo[rows])
        hi[rows] = torch.where(usable, local_hi, hi[rows])
        anchored += int(usable.sum())
    margin = (hi - lo).clamp_min(1e-4) * 1e-4
    result = torch.maximum(lo + margin, torch.minimum(result, hi - margin))
    return result, lo, hi, anchored


def _place(
    fits: SceneFits,
    selected_global_views: tuple[int, ...],
    config: FieldLiftConfig,
) -> FieldPlacementResult:
    inputs, bounds_source = _training_inputs(fits, selected_global_views)
    total_components = sum(field.n for field in inputs.observations)
    count = min(config.max_tracks, total_components)
    carve_config = CompactCarveConfig(
        n_init_3d=count,
        candidate_multiplier=config.candidate_multiplier,
        anchor_mode="component_centers",
        samples_per_ray=config.depth_samples,
        query_batch_size=max(config.depth_samples, min(4096, count * config.depth_samples)),
        min_views=min(config.min_views, len(selected_global_views)),
        hull_fraction=config.robust_view_fraction,
        min_score=config.min_placement_score,
        init_opacity=config.init_opacity,
        seed=config.seed,
    )
    fallback_reason: str | None = None
    try:
        initialization = CompactCarveInitializer(carve_config).initialize(inputs)
    except ValueError as error:
        fallback_reason = str(error)
        initialization = _fallback_initialization(
            inputs,
            count=count,
            opacity=config.init_opacity,
        )

    global_map = torch.tensor(selected_global_views, dtype=torch.long)
    global_views = global_map[initialization.lineage.source_view_indices]
    valid_alpha = _alpha_valid_sources(
        fits,
        global_views,
        initialization.lineage.source_component_indices,
        initialization.lineage.source_xy,
    )
    if not bool(valid_alpha.any()):
        raise ValueError("lossless alpha rejected every field-placement source")
    selected_rows = valid_alpha.nonzero(as_tuple=True)[0]
    local_views = initialization.lineage.source_view_indices[selected_rows]
    global_views = global_views[selected_rows]
    components = initialization.lineage.source_component_indices[selected_rows]
    xy = initialization.lineage.source_xy[selected_rows]
    depths = initialization.depths[selected_rows]
    scores = initialization.scores[selected_rows]

    lower, upper = _ray_depth_bounds(inputs, local_views, xy, depths)
    depths, lower, upper, sparse_anchor_tracks = _apply_sparse_point_priors(
        inputs,
        local_views,
        components,
        xy,
        depths,
        lower,
        upper,
        config.depth_prior_window,
    )
    depths, lower, upper = _apply_depth_priors(
        fits,
        global_views,
        components,
        depths,
        lower,
        upper,
        config.depth_prior_window,
    )

    source_covariances = torch.stack(
        [
            _component_covariances(fits.observations[int(view)])[int(component)]
            for view, component in zip(global_views, components, strict=True)
        ]
    ).to(xy)
    minimum_variance = float(torch.linalg.eigvalsh(source_covariances).min())
    dilation = min(config.projection_dilation, 0.5 * minimum_variance)
    source_colors = torch.stack(
        [
            fits.observations[int(view)].colors[int(component)]
            for view, component in zip(global_views, components, strict=True)
        ]
    ).to(xy)
    field_masses = torch.stack(
        [
            fits.observations[int(view)].amplitudes[int(component)]
            for view, component in zip(global_views, components, strict=True)
        ]
    ).to(xy)
    render_opacity = torch.full_like(field_masses, config.init_opacity)
    fiber = InverseProjectionFiber(
        cameras=inputs.cameras,
        source_view_indices=local_views,
        source_component_indices=components,
        source_means2d=xy,
        source_covariances2d=source_covariances,
        initial_depths=depths,
        depth_lower=lower,
        depth_upper=upper,
        dilation=dilation,
    )

    background_mask = torch.zeros(fiber.n, dtype=torch.bool)
    if all(fits.alphas[index] is None for index in selected_global_views):
        background_count = min(
            fiber.n,
            int(round(config.background_fraction * fiber.n)),
        )
        if background_count:
            border_scores = torch.empty(fiber.n, dtype=xy.dtype)
            for local_view in local_views.unique(sorted=True).tolist():
                rows = local_views == local_view
                camera = inputs.cameras[local_view]
                centered = (xy[rows] - xy.new_tensor([camera.cx, camera.cy])) / xy.new_tensor(
                    [0.5 * camera.width, 0.5 * camera.height]
                )
                border_scores[rows] = centered.abs().amax(dim=-1)
            background_rows = torch.argsort(
                border_scores,
                descending=True,
                stable=True,
            )[:background_count]
            background_mask[background_rows] = True
            target_depth = lower + 0.90 * (upper - lower)
            fraction = (target_depth - lower) / (upper - lower)
            with torch.no_grad():
                fiber.depth_logits[background_rows] = torch.logit(
                    fraction[background_rows].clamp(1e-5, 1.0 - 1e-5)
                ).to(fiber.depth_logits)
                fiber.log_ray_scale[background_rows] += math.log(config.background_ray_scale)

    diagnostics: dict[str, object] = {
        "placement": dict(initialization.diagnostics),
        "placement_fallback_reason": fallback_reason,
        "bounds_source": bounds_source,
        "geometry_is_train_only": fits.geometry_is_train_only,
        "unverified_geometry_ignored": bool(
            fits.heldout_view_indices
            and not fits.geometry_is_train_only
            and (fits.bounds_hint is not None or fits.points is not None)
        ),
        "selected_global_views": list(selected_global_views),
        "alpha_rejected_sources": int((~valid_alpha).sum()),
        "sparse_depth_anchor_tracks": sparse_anchor_tracks,
        "projection_dilation": dilation,
        "background_tracks": int(background_mask.sum()),
        "n_tracks": fiber.n,
    }
    return FieldPlacementResult(
        fiber=fiber,
        field_masses=field_masses,
        source_colors=source_colors,
        render_opacity=render_opacity,
        scores=scores,
        source_global_view_indices=global_views,
        background_mask=background_mask,
        diagnostics=diagnostics,
    )


def _topology_state(
    placement: FieldPlacementResult,
    refit: FieldRefitResult,
) -> FieldTopologyState:
    fiber = refit.fiber
    depths = fiber.depths().detach()
    components: list[FieldComponent] = []
    for index in range(fiber.n):
        lineage = SourceLineage(
            int(fiber.source_view_indices[index]),
            int(fiber.source_component_indices[index]),
        )
        payload = FieldComponentPayload(
            source_lineage=(lineage,),
            source_anchor=SourceAnchor(
                lineage,
                tuple(float(value) for value in fiber.source_means2d[index]),
            ),
            depth=float(depths[index]),
            cross=tuple(float(value) for value in fiber.cross[index].detach()),
            log_ray_scale=float(fiber.log_ray_scale[index].detach()),
            density_mass=float(refit.field_masses[index]),
            source_color=tuple(float(value) for value in placement.source_colors[index]),
            render_opacity=float(refit.render_opacity[index]),
        )
        components.append(FieldComponent(index, payload))
    return FieldTopologyState(tuple(components))


def _birth_payload(
    state: FieldTopologyState,
    inputs: ReconstructionInputs,
    fits: SceneFits,
    selected_global_views: tuple[int, ...],
    config: FieldLiftConfig,
    residual_scores: dict[tuple[int, int], float] | None = None,
) -> FieldComponentPayload | None:
    used = {
        (lineage.view_index, lineage.component_index)
        for component in state.components
        for lineage in component.source_lineage
    }
    candidates: list[tuple[float, float, int, int]] = []
    for local_view, observation in enumerate(inputs.observations):
        for component in range(observation.n):
            if (local_view, component) not in used:
                global_view = selected_global_views[local_view]
                mask = _alpha_mask(fits.alphas[global_view], observation)
                if mask is not None:
                    xy = observation.native_means(component)
                    x = int(xy[0].floor().clamp(0, mask.shape[1] - 1))
                    y = int(xy[1].floor().clamp(0, mask.shape[0] - 1))
                    if not bool(mask[y, x]):
                        continue
                amplitude = float(observation.amplitudes[component])
                candidates.append(
                    (
                        (
                            amplitude
                            if residual_scores is None
                            else residual_scores.get((local_view, component), 0.0)
                        ),
                        amplitude,
                        local_view,
                        component,
                    )
                )
    if not candidates:
        return None
    _residual, _amplitude, local_view, component = min(
        candidates,
        key=lambda item: (-item[0], -item[1], item[2], item[3]),
    )
    observation = inputs.observations[local_view]
    xy = observation.native_means(component, dtype=torch.float64)
    view_tensor = torch.tensor([local_view], dtype=torch.long)
    depth_guess = inputs.cameras[local_view].project(inputs.bounds_hint[0][None])[1].clamp_min(0.1)
    lower, upper = _ray_depth_bounds(
        inputs,
        view_tensor,
        xy.reshape(1, 2),
        depth_guess,
    )
    global_view = selected_global_views[local_view]
    if fits.depth_priors is not None and fits.depth_priors[global_view] is not None:
        prior = fits.depth_priors[global_view][component]
        assert fits.depth_confidences is not None
        confidence = fits.depth_confidences[global_view][component]
        depth = float(
            ((1.0 - confidence) * depth_guess[0] + confidence * prior).clamp(lower[0], upper[0])
        )
    else:
        depth = float(0.5 * (lower[0] + upper[0]))
    covariance = _component_covariances(observation)[component]
    minor_sigma = float(torch.linalg.eigvalsh(covariance).min().sqrt())
    focal = 0.5 * (inputs.cameras[local_view].fx + inputs.cameras[local_view].fy)
    ray_sigma = max(depth * minor_sigma / focal, 1e-6)
    lineage = SourceLineage(local_view, component)
    return FieldComponentPayload(
        source_lineage=(lineage,),
        source_anchor=SourceAnchor(lineage, tuple(float(value) for value in xy)),
        depth=depth,
        cross=(0.0, 0.0),
        log_ray_scale=math.log(ray_sigma),
        density_mass=max(float(observation.amplitudes[component]), 1e-12),
        source_color=tuple(float(value) for value in observation.colors[component]),
        render_opacity=config.init_opacity,
    )


def _state_view_amplitudes(
    state: FieldTopologyState,
    base_state: FieldTopologyState,
    base_refit: FieldRefitResult,
    view: int,
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    """Map block-fixed visibility/gain evidence onto a candidate topology state."""

    base_columns = {
        component.stable_id: index for index, component in enumerate(base_state.components)
    }
    visibility = like.new_ones(len(state.components))
    for index, component in enumerate(state.components):
        source_index = base_columns.get(component.stable_id)
        if source_index is not None:
            visibility[index] = base_refit.visibility.weights[view, source_index].to(like)
    masses = like.new_tensor([component.density_mass for component in state.components])
    return masses * visibility * base_refit.gains[view].to(like)


def _projected_runnalls_scores(
    means: torch.Tensor,
    covariances: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pair indices and the Runnalls moment-merge KL upper bound in one view."""

    pairs = torch.triu_indices(means.shape[0], means.shape[0], offset=1)
    if pairs.shape[1] == 0:
        return pairs, means.new_empty(0)
    left, right = pairs
    left_weight = weights[left].clamp_min(0.0)
    right_weight = weights[right].clamp_min(0.0)
    total_weight = left_weight + right_weight
    safe_weight = total_weight.clamp_min(torch.finfo(means.dtype).tiny)
    merged_mean = (
        left_weight[:, None] * means[left] + right_weight[:, None] * means[right]
    ) / safe_weight[:, None]
    left_delta = means[left] - merged_mean
    right_delta = means[right] - merged_mean
    merged_covariance = (
        left_weight[:, None, None]
        * (covariances[left] + left_delta[:, :, None] * left_delta[:, None, :])
        + right_weight[:, None, None]
        * (covariances[right] + right_delta[:, :, None] * right_delta[:, None, :])
    ) / safe_weight[:, None, None]
    merged_logdet = torch.linalg.slogdet(merged_covariance).logabsdet
    left_logdet = torch.linalg.slogdet(covariances[left]).logabsdet
    right_logdet = torch.linalg.slogdet(covariances[right]).logabsdet
    score = 0.5 * (
        total_weight * merged_logdet - left_weight * left_logdet - right_weight * right_logdet
    )
    score = torch.where(total_weight > 0, score.clamp_min(0.0), torch.full_like(score, torch.inf))
    return pairs, score


def _runnalls_merge_pairs(
    state: FieldTopologyState,
    *,
    inputs: ReconstructionInputs,
    fits: SceneFits,
    selected_global_views: tuple[int, ...],
    previous: FieldPlacementResult,
    base_state: FieldTopologyState,
    base_refit: FieldRefitResult,
    maximum: int,
) -> tuple[tuple[int, int], ...]:
    candidate = _placement_from_topology(state, fits, selected_global_views, previous)
    pair_indices: torch.Tensor | None = None
    total_score: torch.Tensor | None = None
    for view, camera in enumerate(inputs.cameras):
        projection = candidate.fiber.project(camera)
        projected_means = projection.means2d.detach()
        projected_covariances = projection.covariances2d.detach()
        projected_depth = projection.depth.detach()
        amplitudes = _state_view_amplitudes(
            state,
            base_state,
            base_refit,
            view,
            like=projected_means,
        )
        valid = (projected_depth > 0.05) & camera.in_image(projected_means)
        amplitudes = amplitudes * valid
        local_pairs, score = _projected_runnalls_scores(
            projected_means,
            projected_covariances,
            amplitudes,
        )
        pair_indices = local_pairs
        total_score = score if total_score is None else total_score + score
    if pair_indices is None or total_score is None or total_score.numel() == 0:
        return ()
    order = torch.argsort(total_score, stable=True)
    result: list[tuple[int, int]] = []
    for pair_column in order.tolist():
        if not math.isfinite(float(total_score[pair_column])):
            continue
        left = state.components[int(pair_indices[0, pair_column])].stable_id
        right = state.components[int(pair_indices[1, pair_column])].stable_id
        result.append((left, right))
        if len(result) == maximum:
            break
    return tuple(result)


def _unexplained_source_scores(
    state: FieldTopologyState,
    *,
    inputs: ReconstructionInputs,
    fits: SceneFits,
    selected_global_views: tuple[int, ...],
    previous: FieldPlacementResult,
    base_state: FieldTopologyState,
    base_refit: FieldRefitResult,
) -> dict[tuple[int, int], float]:
    """Score source components by target self-energy not covered by the current field."""

    candidate = _placement_from_topology(state, fits, selected_global_views, previous)
    scores: dict[tuple[int, int], float] = {}
    for view, (camera, observation) in enumerate(
        zip(inputs.cameras, inputs.observations, strict=True)
    ):
        projection = candidate.fiber.project(camera)
        projected_means = projection.means2d.detach()
        projected_covariances = projection.covariances2d.detach()
        projected_depth = projection.depth.detach()
        predicted_amplitude = _state_view_amplitudes(
            state,
            base_state,
            base_refit,
            view,
            like=projected_means,
        )
        valid = (projected_depth > 0.05) & camera.in_image(projected_means)
        predicted_amplitude = predicted_amplitude * valid
        target = _analytic_field(observation, dtype=projected_means.dtype)
        cross_kernel = peak_gaussian_product_integrals(
            projected_means,
            projected_covariances,
            target.means,
            target.covariances,
        )
        cross = (
            cross_kernel * predicted_amplitude[:, None] * target.density_amplitudes[None, :]
        ).sum(dim=0)
        self_kernel = peak_gaussian_product_integrals(
            target.means,
            target.covariances,
            target.means,
            target.covariances,
        ).diagonal()
        self_energy = self_kernel * target.density_amplitudes.square()
        fraction = self_energy / (self_energy + cross).clamp_min(
            torch.finfo(self_energy.dtype).tiny
        )
        residual = target.density_amplitudes * fraction
        for component, value in enumerate(residual):
            scores[(view, component)] = float(value.detach())
    return scores


class _DefaultTopologyOps(TopologyOps):
    def __init__(
        self,
        *,
        inputs: ReconstructionInputs,
        fits: SceneFits,
        selected_global_views: tuple[int, ...],
        config: FieldLiftConfig,
        previous: FieldPlacementResult,
        base_state: FieldTopologyState,
        base_refit: FieldRefitResult,
    ) -> None:
        self.inputs = inputs
        self.fits = fits
        self.selected_global_views = selected_global_views
        self.config = config
        self.previous = previous
        self.base_state = base_state
        self.base_refit = base_refit

    def proposals(self, state: FieldTopologyState):
        proposals = []
        if len(state.components) > 1:
            prune = min(
                state.components,
                key=lambda component: (component.density_mass, component.stable_id),
            )
            proposals.append(propose_prune(state, prune.stable_id, tag="lowest-density-mass"))

            for left, right in _runnalls_merge_pairs(
                state,
                inputs=self.inputs,
                fits=self.fits,
                selected_global_views=self.selected_global_views,
                previous=self.previous,
                base_state=self.base_state,
                base_refit=self.base_refit,
                maximum=self.config.topology_merge_candidates,
            ):
                proposals.append(
                    propose_merge(
                        state,
                        left,
                        right,
                        tag="projected-runnalls-bound",
                    )
                )
        if state.components:
            split = max(
                state.components,
                key=lambda component: (component.density_mass, -component.stable_id),
            )
            proposals.append(propose_split(state, split.stable_id, tag="largest-density-mass"))
        residual_scores = _unexplained_source_scores(
            state,
            inputs=self.inputs,
            fits=self.fits,
            selected_global_views=self.selected_global_views,
            previous=self.previous,
            base_state=self.base_state,
            base_refit=self.base_refit,
        )
        birth = _birth_payload(
            state,
            self.inputs,
            self.fits,
            self.selected_global_views,
            self.config,
            residual_scores,
        )
        if birth is not None:
            proposals.append(propose_birth(state, birth, tag="unexplained-source-field"))
        return tuple(proposals)


def _placement_from_topology(
    state: FieldTopologyState,
    fits: SceneFits,
    selected_global_views: tuple[int, ...],
    previous: FieldPlacementResult,
) -> FieldPlacementResult:
    inputs, bounds_source = _training_inputs(fits, selected_global_views)
    dtype = previous.field_masses.dtype
    local_views = torch.tensor(
        [component.source_anchor.source.view_index for component in state.components],
        dtype=torch.long,
    )
    component_ids = torch.tensor(
        [component.source_anchor.source.component_index for component in state.components],
        dtype=torch.long,
    )
    xy = torch.tensor(
        [component.source_anchor.xy for component in state.components],
        dtype=dtype,
    )
    depths = torch.tensor(
        [component.depth for component in state.components],
        dtype=dtype,
    )
    lower, upper = _ray_depth_bounds(inputs, local_views, xy, depths)
    source_covariances = torch.stack(
        [
            _component_covariances(inputs.observations[int(view)])[int(component)]
            for view, component in zip(local_views, component_ids, strict=True)
        ]
    ).to(dtype)
    minimum_variance = float(torch.linalg.eigvalsh(source_covariances).min())
    dilation = min(previous.fiber.dilation, 0.5 * minimum_variance)
    fiber = InverseProjectionFiber(
        cameras=inputs.cameras,
        source_view_indices=local_views,
        source_component_indices=component_ids,
        source_means2d=xy,
        source_covariances2d=source_covariances,
        initial_depths=depths,
        depth_lower=lower,
        depth_upper=upper,
        dilation=dilation,
    )
    with torch.no_grad():
        fiber.cross.copy_(
            torch.tensor([component.cross for component in state.components], dtype=dtype)
        )
        fiber.log_ray_scale.copy_(
            torch.tensor(
                [component.log_ray_scale for component in state.components],
                dtype=dtype,
            )
        )
    global_map = torch.tensor(selected_global_views, dtype=torch.long)
    return FieldPlacementResult(
        fiber=fiber,
        field_masses=torch.tensor(
            [component.density_mass for component in state.components],
            dtype=dtype,
        ),
        source_colors=torch.tensor(
            [component.source_color for component in state.components],
            dtype=dtype,
        ),
        render_opacity=torch.tensor(
            [component.render_opacity for component in state.components],
            dtype=dtype,
        ),
        scores=torch.zeros(len(state.components), dtype=dtype),
        source_global_view_indices=global_map[local_views],
        background_mask=torch.zeros(len(state.components), dtype=torch.bool),
        diagnostics={
            **previous.diagnostics,
            "topology_rebuilt": True,
            "bounds_source": bounds_source,
            "n_tracks": len(state.components),
        },
    )


def _topology_density_objective(
    state: FieldTopologyState,
    *,
    inputs: ReconstructionInputs,
    fits: SceneFits,
    selected_global_views: tuple[int, ...],
    previous: FieldPlacementResult,
    base_state: FieldTopologyState,
    base_refit: FieldRefitResult,
    references: tuple[AnalyticGaussianField2D, ...],
    chunk_size: int,
) -> float:
    candidate = _placement_from_topology(
        state,
        fits,
        selected_global_views,
        previous,
    )
    base_columns = {
        component.stable_id: index for index, component in enumerate(base_state.components)
    }
    density_total = candidate.field_masses.new_zeros(())
    for view, (camera, target) in enumerate(zip(candidate.fiber.cameras, references, strict=True)):
        projection = candidate.fiber.project(camera)
        weights = candidate.field_masses.new_ones(len(state.components))
        for index, component in enumerate(state.components):
            source_index = base_columns.get(component.stable_id)
            if source_index is None:
                for base_component in base_state.components:
                    if (
                        component.source_anchor == base_component.source_anchor
                        and component.depth == base_component.depth
                        and component.cross == base_component.cross
                    ):
                        source_index = base_columns[base_component.stable_id]
                        break
            if source_index is not None:
                weights[index] = base_refit.visibility.weights[view, source_index]
        amplitude = (
            candidate.field_masses * weights * base_refit.gains[view].to(candidate.field_masses)
        )
        predicted_self = mixture_inner_product(
            projection.means2d,
            projection.covariances2d,
            amplitude,
            projection.means2d,
            projection.covariances2d,
            amplitude,
            chunk_size=chunk_size,
        )
        predicted_target = mixture_inner_product(
            projection.means2d,
            projection.covariances2d,
            amplitude,
            target.means,
            target.covariances,
            target.density_amplitudes,
            chunk_size=chunk_size,
        )
        scale = (
            target.density_amplitudes.sum().square().clamp_min(torch.finfo(amplitude.dtype).tiny)
        )
        density_total = density_total + (predicted_self - 2.0 * predicted_target) / scale
    return float((density_total / len(references)).detach())


def _run_topology(
    *,
    fits: SceneFits,
    selected_global_views: tuple[int, ...],
    placement: FieldPlacementResult,
    refit: FieldRefitResult,
    references: tuple[AnalyticGaussianField2D, ...],
    config: FieldLiftConfig,
) -> tuple[FieldPlacementResult, FieldRefitResult, tuple[MoveReceipt, ...]]:
    if config.topology_rounds == 0:
        return placement, refit, ()
    inputs, _bounds_source = _training_inputs(fits, selected_global_views)
    base_state = _topology_state(placement, refit)
    scheduler = DeterministicTopologyScheduler(
        lambda state: _topology_density_objective(
            state,
            inputs=inputs,
            fits=fits,
            selected_global_views=selected_global_views,
            previous=placement,
            base_state=base_state,
            base_refit=refit,
            references=references,
            chunk_size=config.refit.chunk_size,
        ),
        parsimony_per_component=config.parsimony_per_component,
    )
    operations = _DefaultTopologyOps(
        inputs=inputs,
        fits=fits,
        selected_global_views=selected_global_views,
        config=config,
        previous=placement,
        base_state=base_state,
        base_refit=refit,
    )
    schedule = scheduler.run(
        base_state,
        operations,
        max_rounds=config.topology_rounds,
    )
    if schedule.state is base_state:
        return placement, refit, schedule.receipts
    rebuilt = _placement_from_topology(
        schedule.state,
        fits,
        selected_global_views,
        placement,
    )
    polished = fit_field_fibers(
        fiber=rebuilt.fiber,
        reference_fields=references,
        cameras=rebuilt.fiber.cameras,
        source_colors=rebuilt.source_colors,
        field_masses=rebuilt.field_masses,
        render_opacity=rebuilt.render_opacity,
        config=config.refit,
    )
    return rebuilt, polished, schedule.receipts


def _correspondences(
    gaussians: Gaussians3D,
    field_masses: torch.Tensor,
    visibility: torch.Tensor,
    fits: SceneFits,
    *,
    dustbin: float,
) -> tuple[torch.Tensor, ...]:
    if visibility.shape != (fits.n_views, gaussians.n):
        raise ValueError(
            "visibility must have shape "
            f"({fits.n_views}, {gaussians.n}), got {tuple(visibility.shape)}"
        )
    if visibility.device != gaussians.means.device or visibility.dtype != gaussians.means.dtype:
        raise ValueError("visibility must share the Gaussian device and dtype")
    if (
        not bool(torch.isfinite(visibility).all())
        or bool((visibility < 0).any())
        or bool((visibility > 1).any())
    ):
        raise ValueError("visibility must be finite and lie in [0,1]")
    result: list[torch.Tensor] = []
    for view_index, (camera, observation) in enumerate(
        zip(fits.cameras, fits.observations, strict=True)
    ):
        projection = project_gaussians_ewa(gaussians, camera)
        target = _analytic_field(observation, dtype=gaussians.means.dtype)
        overlap = peak_gaussian_product_integrals(
            projection.means2d,
            projection.covariances2d,
            target.means,
            target.covariances,
        )
        overlap = overlap * field_masses[:, None] * target.density_amplitudes[None, :]
        valid = (projection.depth > 0.05) & camera.in_image(projection.means2d)
        overlap = overlap * valid[:, None] * visibility[view_index, :, None]
        result.append(
            overlap
            / (overlap.sum(dim=1, keepdim=True) + dustbin).clamp_min(
                torch.finfo(overlap.dtype).tiny
            )
        )
    return tuple(result)


class FieldLifter:
    """Registered legacy adapter plus the native compact ``SceneFits`` entry point."""

    def __init__(
        self,
        config: FieldLiftConfig | None = None,
        **config_kwargs: object,
    ) -> None:
        if config is not None and config_kwargs:
            raise ValueError("pass either FieldLiftConfig or keyword controls, not both")
        self.config = FieldLiftConfig(**config_kwargs) if config is None else config

    def fit(self, fits: SceneFits) -> FieldLiftResult:
        """Run the complete field lift on CPU without source images."""

        if not isinstance(fits, SceneFits):
            raise TypeError("fits must be SceneFits")
        original_device = fits.observations[0].device
        working = fits.to("cpu")
        selected = _even_subset(
            working.train_view_indices,
            self.config.max_train_views,
        )
        placement = _place(working, selected, self.config)
        references = tuple(
            _analytic_field(
                working.observations[index],
                dtype=placement.field_masses.dtype,
            )
            for index in selected
        )
        refit = fit_field_fibers(
            fiber=placement.fiber,
            reference_fields=references,
            cameras=placement.fiber.cameras,
            source_colors=placement.source_colors,
            field_masses=placement.field_masses,
            render_opacity=placement.render_opacity,
            config=self.config.refit,
        )
        placement, refit, topology_receipts = _run_topology(
            fits=working,
            selected_global_views=selected,
            placement=placement,
            refit=refit,
            references=references,
            config=self.config,
        )
        all_view_visibility = center_transmittance_visibility(
            refit.gaussians,
            working.cameras,
            source_view_indices=placement.source_global_view_indices,
            force_source_visible=self.config.refit.force_source_visible,
            dilation=placement.fiber.dilation,
        )
        correspondences = _correspondences(
            refit.gaussians,
            refit.field_masses,
            all_view_visibility.weights,
            working,
            dustbin=self.config.correspondence_dustbin,
        )
        validation_masses = all_view_visibility.weights * refit.field_masses[None, :]
        semantic_validation = validate_field_semantics(
            working,
            working.cameras,
            refit.gaussians,
            validation_masses,
            config=FieldValidationConfig(
                sample_cap_per_view=self.config.validation_sample_cap,
                seed=self.config.seed,
                component_chunk=self.config.refit.chunk_size,
            ),
        )
        ranks = [report.rank for report in refit.observability]
        finite_conditions = [
            report.condition_number
            for report in refit.observability
            if math.isfinite(report.condition_number)
        ]
        diagnostics = {
            **placement.diagnostics,
            "analytic_semantics": (
                "exact additive peak-Gaussian density/RGB numerator"
                if all(
                    working.observations[index].blend_mode == "additive"
                    and working.observations[index].support_fade_alpha == 0
                    for index in selected
                )
                else (
                    "untruncated density/RGB-numerator proxy; "
                    "validate normalized teacher separately"
                )
            ),
            "optimized_views": list(selected),
            "heldout_views": list(working.heldout_view_indices),
            "objective_initial": refit.objective_history[0],
            "objective_final": refit.objective_history[-1],
            "accepted_continuous_steps": refit.accepted_steps,
            "topology_proposals": len(topology_receipts),
            "topology_accepted": sum(int(receipt.accepted) for receipt in topology_receipts),
            "observability_rank_histogram": {
                str(rank): ranks.count(rank) for rank in sorted(set(ranks))
            },
            "observability_condition_max": (
                max(finite_conditions) if finite_conditions else math.inf
            ),
            "covariance_free_tracks": int(refit.covariance_free_mask.sum()),
            "covariance_pinned_tracks": int((~refit.covariance_free_mask).sum()),
            "source_projection_max_error": refit.source_projection_max_error,
            "source_color_max_error": refit.source_color_max_error,
            "correspondence_visibility_mean": float(all_view_visibility.weights.mean()),
            "semantic_train_density_mse": semantic_validation.train.density_mse,
            "semantic_train_rgb_mse": semantic_validation.train.rgb_mse,
            "semantic_heldout_density_mse": (
                None
                if semantic_validation.heldout is None
                else semantic_validation.heldout.density_mse
            ),
            "semantic_heldout_rgb_mse": (
                None if semantic_validation.heldout is None else semantic_validation.heldout.rgb_mse
            ),
        }
        if original_device.type != "cpu":
            gaussians = refit.gaussians.to(original_device)
            correspondence_output = tuple(plan.to(original_device) for plan in correspondences)
            correspondence_visibility = all_view_visibility.weights.to(original_device)
        else:
            gaussians = refit.gaussians
            correspondence_output = correspondences
            correspondence_visibility = all_view_visibility.weights
        return FieldLiftResult(
            gaussians=gaussians,
            placement=placement,
            refit=refit,
            correspondences=correspondence_output,
            correspondence_visibility=correspondence_visibility,
            topology_receipts=topology_receipts,
            semantic_validation=semantic_validation,
            optimized_view_indices=selected,
            heldout_view_indices=working.heldout_view_indices,
            diagnostics=diagnostics,
        )

    def lift_compact(
        self,
        dataset: CompactDataset,
        *,
        train_view_indices: Sequence[int] | None = None,
        heldout_view_indices: Sequence[int] | None = None,
    ) -> FieldLiftResult:
        return self.fit(
            SceneFits.from_compact_dataset(
                dataset,
                train_view_indices=train_view_indices,
                heldout_view_indices=heldout_view_indices,
            )
        )

    def lift(
        self,
        gaussians2d: list[Gaussians2D],
        scene: SceneData,
    ) -> Gaussians3D:
        """Legacy registry contract; the native compact entry is :meth:`fit`."""

        fits = SceneFits.from_legacy(gaussians2d, scene)
        return self.fit(fits).gaussians


__all__ = [
    "FieldLiftConfig",
    "FieldLiftResult",
    "FieldLifter",
    "FieldPlacementResult",
]
