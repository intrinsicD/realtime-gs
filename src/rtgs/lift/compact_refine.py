"""Correspondence-free local 4-dof refinement of a compact-carve initialization (prototype).

The compact carve chooses each lift's depth by an ``argmax`` over a coarse sampled ray, so the
initialized depth is only accurate to one depth cell.  This module lifts a
:class:`~rtgs.lift.compact_carve.CompactInitializationResult` into the exact 4-dof
:class:`~rtgs.lift.inverse_projection_fiber.InverseProjectionFiber` (one depth coordinate for the
mean plus the ray-scale of the covariance null space) and locally fine-tunes it against a smooth,
differentiable multi-view *consensus* objective — soft coverage times color agreement in the other
views — before the redundant per-view lifts are merged.

It is deliberately correspondence-free: the objective is the continuous form of the carve/visual-
hull consensus (no explicit cross-view Gaussian matches), so it inherits the same limitation as the
carve — it can only refine depth *locally*, where a lift already projects near the object in the
other views, and it cannot recover the through-depth covariance that a single view leaves
unconstrained.  Solving that needs the explicit fiber-correspondence path, which the repository's
2026-07-17/18 experiments found unsolved.  This is a tested CPU prototype, opt-in and off by
default; no calibrated-scene quality result is claimed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import (
    GaussianObservationIndex,
    ObservationQueryBackend,
)
from rtgs.core.sh import sh_to_rgb
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.compact_carve import CompactInitializationResult, _component_covariances, _field_mass
from rtgs.lift.inverse_projection_fiber import InverseProjectionFiber
from rtgs.render.projection import EWA_DILATION


@dataclass(frozen=True)
class LocalDepthRefineConfig:
    """Controls for the correspondence-free local depth refine."""

    iterations: int = 40
    learning_rate: float = 0.05
    depth_window: float = 0.5  # per-ray sigmoid window as a fraction of the initial depth
    color_sigma: float = 0.25  # color-agreement bandwidth in the consensus objective
    coverage_scale: float = 1.0
    near: float = 0.05
    refine_ray_scale: bool = False  # also optimize the covariance null ray-scale
    init_opacity: float = 0.1

    def __post_init__(self) -> None:
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int):
            raise TypeError("iterations must be an integer")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        for name in ("learning_rate", "color_sigma", "coverage_scale", "near"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < self.depth_window < 1.0:
            raise ValueError("depth_window must be in (0,1)")
        if not 0.0 < self.init_opacity < 1.0:
            raise ValueError("init_opacity must be in (0,1)")


@dataclass(frozen=True)
class LocalDepthRefineResult:
    """Refined geometry plus before/after diagnostics."""

    gaussians: Gaussians3D
    fiber: InverseProjectionFiber
    initial_depths: torch.Tensor
    refined_depths: torch.Tensor
    initial_objective: float
    refined_objective: float
    diagnostics: dict[str, object]


def build_fiber_from_initialization(
    inputs: ReconstructionInputs,
    result: CompactInitializationResult,
    *,
    depth_window: float,
    dilation: float = EWA_DILATION,
) -> InverseProjectionFiber:
    """Construct the exact inverse-projection fiber for a compact initialization.

    Each row keeps its source view/component lineage and the fitted 2D covariance; the depth
    coordinate is bounded to a ``depth_window`` fraction around the carve depth so refinement stays
    local.
    """
    lineage = result.lineage
    dtype = inputs.observations[0].dtype
    xy = lineage.source_xy.to(dtype)
    depths = result.depths.to(dtype)
    source_covariances = torch.empty(lineage.source_view_indices.numel(), 2, 2, dtype=dtype)
    for view_index in lineage.source_view_indices.unique(sorted=True).tolist():
        rows = (lineage.source_view_indices == view_index).nonzero(as_tuple=True)[0]
        field = inputs.observations[view_index]
        source_covariances[rows] = _component_covariances(
            field, lineage.source_component_indices[rows]
        ).to(dtype)
    minimum_variance = float(torch.linalg.eigvalsh(source_covariances).min())
    lower = (depths * (1.0 - depth_window)).clamp_min(depths.new_tensor(1e-4))
    upper = depths * (1.0 + depth_window)
    return InverseProjectionFiber(
        cameras=inputs.cameras,
        source_view_indices=lineage.source_view_indices.clone(),
        source_component_indices=lineage.source_component_indices.clone(),
        source_means2d=xy,
        source_covariances2d=source_covariances,
        initial_depths=depths,
        depth_lower=lower,
        depth_upper=upper,
        dilation=min(dilation, 0.5 * minimum_variance),
    )


def _consensus_objective(
    inputs: ReconstructionInputs,
    means: torch.Tensor,
    source_view_indices: torch.Tensor,
    source_colors: torch.Tensor,
    backends: list[ObservationQueryBackend],
    config: LocalDepthRefineConfig,
) -> torch.Tensor:
    """Smooth per-row multi-view consensus: soft coverage times color agreement, other views only.

    The source view of each row is an exact projection invariant and is excluded, so the gradient
    comes only from the views that must agree on the depth.
    """
    total = means.new_zeros(means.shape[0])
    for view_index, (camera, backend) in enumerate(zip(inputs.cameras, backends, strict=True)):
        field = inputs.observations[view_index]
        uv, depth = camera.project(means)
        query = backend.query(uv)
        seen = ((depth > config.near) & query.valid).to(means.dtype)
        area = float(field.fit_window[2] * field.fit_window[3])
        mass = _field_mass(field).to(means.dtype).clamp_min(torch.finfo(means.dtype).tiny)
        relative_density = area * query.weight_sum.to(means.dtype) / mass
        soft_coverage = 1.0 - torch.exp(-relative_density / config.coverage_scale)
        color_gap = (query.color.to(means.dtype) - source_colors).square().sum(dim=1)
        agreement = torch.exp(-color_gap / (2.0 * config.color_sigma**2))
        contribution = soft_coverage * agreement * seen
        # Exclude each row's own source view (exact invariant, no depth signal there).
        contribution = contribution * (source_view_indices != view_index).to(means.dtype)
        total = total + contribution
    return total


def refine_initialization_depths(
    inputs: ReconstructionInputs,
    result: CompactInitializationResult,
    config: LocalDepthRefineConfig | None = None,
    *,
    backends: list[ObservationQueryBackend] | None = None,
) -> LocalDepthRefineResult:
    """Locally refine carve depths against the smooth multi-view consensus objective.

    Returns refined :class:`Gaussians3D` (projection-consistent covariances rebuilt from the
    refined depth), the fiber, and before/after objective/geometry diagnostics.
    """
    if config is None:
        config = LocalDepthRefineConfig()
    if any(field.device.type != "cpu" for field in inputs.observations):
        raise ValueError("local depth refine is a CPU reference path")
    if backends is None:
        backends = [GaussianObservationIndex(field) for field in inputs.observations]
    if len(backends) != inputs.n_views:
        raise ValueError("backends must contain one query backend per view")

    fiber = build_fiber_from_initialization(inputs, result, depth_window=config.depth_window)
    source_colors = sh_to_rgb(result.gaussians.sh[:, 0]).to(fiber.source_means2d.dtype)
    source_views = result.lineage.source_view_indices

    parameters = [fiber.depth_logits]
    if config.refine_ray_scale:
        parameters.append(fiber.log_ray_scale)
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)

    initial_depths = fiber.depths().detach().clone()
    with torch.no_grad():
        initial_objective = float(
            _consensus_objective(
                inputs,
                fiber.means_covariances()[0],
                source_views,
                source_colors,
                backends,
                config,
            ).mean()
        )

    for _ in range(config.iterations):
        optimizer.zero_grad()
        means, _ = fiber.means_covariances()
        objective = _consensus_objective(
            inputs, means, source_views, source_colors, backends, config
        ).mean()
        (-objective).backward()
        optimizer.step()

    refined_depths = fiber.depths().detach().clone()
    with torch.no_grad():
        refined_objective = float(
            _consensus_objective(
                inputs,
                fiber.means_covariances()[0],
                source_views,
                source_colors,
                backends,
                config,
            ).mean()
        )
    with torch.no_grad():
        gaussians = fiber.as_gaussians(
            colors=source_colors.clamp(0.0, 1.0),
            opacity=config.init_opacity,
        )
    diagnostics = {
        "refine_iterations": config.iterations,
        "refine_ray_scale": config.refine_ray_scale,
        "mean_absolute_depth_change": float((refined_depths - initial_depths).abs().mean()),
        "objective_gain": refined_objective - initial_objective,
    }
    return LocalDepthRefineResult(
        gaussians=gaussians,
        fiber=fiber,
        initial_depths=initial_depths,
        refined_depths=refined_depths,
        initial_objective=initial_objective,
        refined_objective=refined_objective,
        diagnostics=diagnostics,
    )


__all__ = [
    "LocalDepthRefineConfig",
    "LocalDepthRefineResult",
    "build_fiber_from_initialization",
    "refine_initialization_depths",
]
