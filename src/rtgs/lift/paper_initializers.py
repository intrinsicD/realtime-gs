"""Matched RGB-free initializations for the paper's three compact reconstruction paths.

This module deliberately does not import ``SceneData`` or either image-backed trainer.  Every
initializer consumes the same train-only :class:`ReconstructionInputs`:

* calibration-bounded random points;
* calibrated Splat-SfM tracks from the frozen 2D Gaussian fields;
* unmodified tomographic Beam Fusion.

Splat-SfM and Beam Fusion may naturally return different counts.  The comparison uses the
largest count shared by both structural initializers (optionally capped by the protocol), keeps
the strongest rows deterministically, and draws exactly that many random rows.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.core.sh import rgb_to_sh
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.beam_fusion import (
    BeamFusionConfig,
    BeamFusionResult,
    fuse_gaussian_beams,
)
from rtgs.lift.compact_carve import _center_and_extent, _validate_cpu_inputs
from rtgs.lift.splat_sfm import SplatSfMConfig, SplatSfMResult, structure_from_splats


@dataclass(frozen=True)
class PaperInitializerConfig:
    """Frozen controls for the three matched initializations."""

    random_seed: int
    random_bounds_scale: float = 0.5
    init_opacity: float = 0.1
    max_starting_gaussians: int | None = None
    structural_components_per_view: int | None = None
    sfm: SplatSfMConfig = SplatSfMConfig()
    beam: BeamFusionConfig = BeamFusionConfig()

    def __post_init__(self) -> None:
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be an integer")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if not math.isfinite(self.random_bounds_scale) or self.random_bounds_scale <= 0:
            raise ValueError("random_bounds_scale must be finite and positive")
        if not 0.0 < self.init_opacity < 1.0:
            raise ValueError("init_opacity must be in (0,1)")
        if self.max_starting_gaussians is not None and (
            isinstance(self.max_starting_gaussians, bool)
            or not isinstance(self.max_starting_gaussians, int)
            or self.max_starting_gaussians <= 0
        ):
            raise ValueError("max_starting_gaussians must be a positive integer or None")
        if self.structural_components_per_view is not None and (
            isinstance(self.structural_components_per_view, bool)
            or not isinstance(self.structural_components_per_view, int)
            or self.structural_components_per_view <= 0
        ):
            raise ValueError("structural_components_per_view must be a positive integer or None")
        if self.sfm.init_opacity != self.init_opacity:
            raise ValueError("Splat-SfM init_opacity must match the shared init_opacity")
        if self.beam.init_opacity != self.init_opacity:
            raise ValueError("Beam Fusion init_opacity must match the shared init_opacity")


@dataclass(frozen=True)
class MatchedPaperInitializations:
    """Three exact-count initializations plus their selection evidence."""

    bounded_random: Gaussians3D
    splat_sfm: Gaussians3D
    beam_fusion: Gaussians3D
    splat_sfm_result: SplatSfMResult
    beam_fusion_result: BeamFusionResult
    splat_sfm_selected_rows: torch.Tensor
    beam_fusion_selected_rows: torch.Tensor
    structural_selected_rows: tuple[torch.Tensor, ...]
    receipt: dict

    @property
    def count(self) -> int:
        """Return the common starting count."""

        return self.bounded_random.n


def bounded_random_initialization(
    inputs: ReconstructionInputs,
    count: int,
    *,
    seed: int,
    bounds_scale: float = 0.5,
    opacity: float = 0.1,
) -> Gaussians3D:
    """Draw standard random 3DGS rows inside the camera-derived scene sphere.

    The rows use neutral gray degree-zero color and one isotropic scale based on the expected
    spacing at ``count``.  No compact-field query is used, keeping this a genuine lower-bound
    initialization rather than a hidden placement method.
    """

    _validate_cpu_inputs(inputs)
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not math.isfinite(bounds_scale) or bounds_scale <= 0:
        raise ValueError("bounds_scale must be finite and positive")
    if not 0.0 < opacity < 1.0:
        raise ValueError("opacity must be in (0,1)")

    dtype = inputs.observations[0].dtype
    center, extent = _center_and_extent(inputs, dtype)
    radius = float(extent) * bounds_scale
    generator = torch.Generator(device="cpu").manual_seed(seed)
    directions = torch.randn(count, 3, dtype=dtype, generator=generator)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    radial = radius * torch.rand(count, 1, dtype=dtype, generator=generator).pow(1.0 / 3.0)
    means = center + directions * radial
    scale = radius / count ** (1.0 / 3.0)
    quats = torch.zeros(count, 4, dtype=dtype)
    quats[:, 0] = 1.0
    colors = torch.full((count, 3), 0.5, dtype=dtype)
    sh = rgb_to_sh(colors)[:, None, :]
    return Gaussians3D(
        means=means,
        quats=quats,
        log_scales=torch.full((count, 3), math.log(max(scale, 1e-6)), dtype=dtype),
        opacity=torch.full((count,), opacity, dtype=dtype),
        sh=sh,
    )


def _track_lengths(result: SplatSfMResult) -> torch.Tensor:
    return result.track_offsets[1:] - result.track_offsets[:-1]


def _component_lengths(result: BeamFusionResult) -> torch.Tensor:
    return result.component_offsets[1:] - result.component_offsets[:-1]


def _subset_field(
    field: GaussianObservationField,
    rows: torch.Tensor,
) -> GaussianObservationField:
    return GaussianObservationField(
        width=field.width,
        height=field.height,
        means=field.means[rows],
        log_scales=field.log_scales[rows],
        rotations=field.rotations[rows],
        colors=field.colors[rows],
        amplitudes=field.amplitudes[rows],
        color_grads=None if field.color_grads is None else field.color_grads[rows],
        filter_variance=(None if field.filter_variance is None else field.filter_variance[rows]),
        blend_mode=field.blend_mode,
        epsilon=field.epsilon,
        sigma_cutoff=field.sigma_cutoff,
        support_fade_alpha=field.support_fade_alpha,
        aa_dilation=field.aa_dilation,
        view_id=field.view_id,
        fit_window=field.fit_window,
        n_init=field.n_init,
        provider=field.provider,
        producer_version=field.producer_version,
        producer_source_digest=field.producer_source_digest,
        fit_config_digest=field.fit_config_digest,
        mean_residuals=None if field.mean_residuals is None else field.mean_residuals[rows],
    )


def _structural_rows(field: GaussianObservationField, count: int) -> torch.Tensor:
    """Select integrated-mass leaders over a deterministic fitted-window grid."""

    if count >= field.n:
        return torch.arange(field.n, dtype=torch.long)
    bins = max(1, math.isqrt(count))
    fit_x, fit_y, fit_width, fit_height = field.fit_window
    means = field.native_means()
    tile_x = ((means[:, 0] - fit_x) * bins / fit_width).floor().long().clamp(0, bins - 1)
    tile_y = ((means[:, 1] - fit_y) * bins / fit_height).floor().long().clamp(0, bins - 1)
    tile_key = tile_y * bins + tile_x
    mass = field.amplitudes * field.scales().prod(dim=-1)

    score_order = torch.argsort(mass, descending=True, stable=True)
    tile_order = torch.argsort(tile_key[score_order], stable=True)
    grouped = score_order[tile_order]
    grouped_tiles = tile_key[grouped]
    first = torch.ones(grouped.numel(), dtype=torch.bool)
    first[1:] = grouped_tiles[1:] != grouped_tiles[:-1]
    selected = grouped[first]
    selected_mask = torch.zeros(field.n, dtype=torch.bool)
    selected_mask[selected] = True
    remaining = score_order[~selected_mask[score_order]]
    rows = torch.cat([selected, remaining[: count - selected.numel()]])
    if rows.numel() != count or torch.unique(rows).numel() != count:
        raise RuntimeError("structural component selection did not return a unique exact count")
    return rows.sort().values


def structural_initialization_inputs(
    inputs: ReconstructionInputs,
    components_per_view: int | None,
) -> tuple[ReconstructionInputs, tuple[torch.Tensor, ...]]:
    """Derive one shared bounded structural carrier from high-capacity teacher fields.

    This controls the quadratic candidate work of Splat-SfM and Beam Fusion only.  The returned
    row lists are auditable, and downstream compact optimization still queries the complete
    high-capacity teachers.
    """

    _validate_cpu_inputs(inputs)
    rows = tuple(
        (
            torch.arange(field.n, dtype=torch.long)
            if components_per_view is None
            else _structural_rows(field, min(components_per_view, field.n))
        )
        for field in inputs.observations
    )
    if components_per_view is None:
        return inputs, rows
    structural = ReconstructionInputs(
        observations=[
            _subset_field(field, selected)
            for field, selected in zip(inputs.observations, rows, strict=True)
        ],
        cameras=list(inputs.cameras),
        view_names=list(inputs.view_names),
        points=None,
        point_visibility=None,
        bounds_hint=inputs.bounds_hint,
        name=f"{inputs.name}-structural-{components_per_view}",
        archive_stats=None,
    )
    return structural, rows


def _splat_sfm_selection(result: SplatSfMResult, count: int) -> torch.Tensor:
    """Rank verified tracks without reading any downstream outcome."""

    lengths = _track_lengths(result)
    rows = sorted(
        range(result.n_tracks),
        key=lambda row: (
            -int(lengths[row]),
            float(result.track_reprojection_error[row]),
            float(result.track_covariance_residual[row]),
            -float(result.track_triangulation_angle_deg[row]),
            row,
        ),
    )[:count]
    return torch.tensor(rows, dtype=torch.long)


def _beam_selection(result: BeamFusionResult, count: int) -> torch.Tensor:
    """Rank Beam components by their initializer-native evidence only."""

    lengths = _component_lengths(result)
    rows = sorted(
        range(result.n_components),
        key=lambda row: (
            -float(result.component_weights[row]),
            -int(lengths[row]),
            row,
        ),
    )[:count]
    return torch.tensor(rows, dtype=torch.long)


def build_matched_paper_initializations(
    inputs: ReconstructionInputs,
    config: PaperInitializerConfig,
) -> MatchedPaperInitializations:
    """Construct and exact-count match the three paper-path initializations."""

    _validate_cpu_inputs(inputs)
    structural_inputs, structural_rows = structural_initialization_inputs(
        inputs,
        config.structural_components_per_view,
    )
    sfm = structure_from_splats(structural_inputs, config.sfm)
    beam = fuse_gaussian_beams(structural_inputs, config.beam)
    count = min(sfm.n_tracks, beam.n_components)
    if config.max_starting_gaussians is not None:
        count = min(count, config.max_starting_gaussians)
    if count <= 0:
        raise RuntimeError("matched paper initializer count is empty")

    sfm_rows = _splat_sfm_selection(sfm, count)
    beam_rows = _beam_selection(beam, count)
    bounded_random = bounded_random_initialization(
        inputs,
        count,
        seed=config.random_seed,
        bounds_scale=config.random_bounds_scale,
        opacity=config.init_opacity,
    )
    splat_sfm = sfm.gaussians.subset(sfm_rows)
    beam_fusion = beam.gaussians.subset(beam_rows)
    if not (bounded_random.n == splat_sfm.n == beam_fusion.n == count):
        raise RuntimeError("paper initializer exact-count match failed")

    receipt = {
        "schema": "rtgs.paper_initializers.v1",
        "common_count": count,
        "count_policy": "min(splat_sfm_tracks,beam_components,optional_protocol_cap)",
        "random_policy": "uniform_volume_camera_derived_sphere_neutral_gray",
        "splat_sfm_selection_policy": (
            "track_views_desc,reprojection_asc,covariance_residual_asc,"
            "triangulation_angle_desc,row_asc"
        ),
        "beam_selection_policy": "component_weight_desc,contributor_views_desc,row_asc",
        "source_counts": {
            "splat_sfm": sfm.n_tracks,
            "beam_fusion": beam.n_components,
        },
        "structural_input": {
            "policy": (
                "all_components"
                if config.structural_components_per_view is None
                else "grid_stratified_integrated_mass"
            ),
            "components_per_view_cap": config.structural_components_per_view,
            "full_components_per_view": [field.n for field in inputs.observations],
            "selected_components_per_view": [int(selected.numel()) for selected in structural_rows],
            "selected_rows": [selected.tolist() for selected in structural_rows],
            "downstream_teacher_is_full_field": True,
        },
        "selected_rows": {
            "splat_sfm": sfm_rows.tolist(),
            "beam_fusion": beam_rows.tolist(),
        },
        "config": asdict(config),
        "diagnostics": {
            "splat_sfm": sfm.diagnostics,
            "beam_fusion": beam.diagnostics,
        },
    }
    return MatchedPaperInitializations(
        bounded_random=bounded_random,
        splat_sfm=splat_sfm,
        beam_fusion=beam_fusion,
        splat_sfm_result=sfm,
        beam_fusion_result=beam,
        splat_sfm_selected_rows=sfm_rows,
        beam_fusion_selected_rows=beam_rows,
        structural_selected_rows=structural_rows,
        receipt=receipt,
    )


__all__ = [
    "MatchedPaperInitializations",
    "PaperInitializerConfig",
    "bounded_random_initialization",
    "build_matched_paper_initializations",
    "structural_initialization_inputs",
]
