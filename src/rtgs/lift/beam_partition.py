"""Masked 2D-density partitions for covariance-only Beam Fusion refits.

Beam Fusion already records the native 2D Gaussian that contributed to every retained 3D
Gaussian in every participating view.  This module treats the *unique* recorded source
Gaussians as fixed 2D anchors; it never projects a 3D mean into an image to discover or rematch
an anchor.

For each view, deterministic Gauss-Hermite samples of the complete source Gaussian mixture are
discarded outside the packed foreground mask and assigned to their nearest native anchor mean.
Those hard Voronoi responsibilities form a partition of unity over the sampled masked density.
The second moment of each partition is measured about its fixed anchor mean.  The resulting 2D
covariances can then replace the original transverse beam covariances while retaining the exact
CSR correspondence, implied beam depth, 3D mean, color, opacity, and Gaussian count.

This is an opt-in research mechanism.  It does not register or change a production initializer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.compact_views import PackedAlpha
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.beam_fusion import (
    BeamFusionConfig,
    BeamFusionResult,
    _beam_precisions,
    _component_covariances_2d,
    _prepare_views,
)
from rtgs.lift.compact_carve import _center_and_extent, _validate_cpu_inputs


@dataclass(frozen=True)
class MaskedPartitionConfig:
    """Numerical controls for the fixed-anchor masked density partition."""

    quadrature_order: int = 5
    assignment_chunk: int = 8_192
    min_partition_mass: float = 1e-12
    min_variance_px: float = 1e-6

    def __post_init__(self) -> None:
        if self.quadrature_order not in {3, 5, 7}:
            raise ValueError("quadrature_order must be one of 3, 5, or 7")
        if isinstance(self.assignment_chunk, bool) or not isinstance(self.assignment_chunk, int):
            raise TypeError("assignment_chunk must be an integer")
        if self.assignment_chunk <= 0:
            raise ValueError("assignment_chunk must be positive")
        for name in ("min_partition_mass", "min_variance_px"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class MaskedDensityPartition:
    """Per-view moments for unique native contributor anchors."""

    anchor_component_indices: torch.Tensor  # (A,), int64, sorted and unique
    covariances2d: torch.Tensor  # (A,2,2), moments about exact native anchor means
    area_matched_covariances2d: torch.Tensor  # (A,2,2), native shape with matched determinant
    masses: torch.Tensor  # (A,), masked integrated source density
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class BeamPartitionCovarianceResult:
    """Covariance-only Beam Fusion treatments and their per-view partitions."""

    area_matched: Gaussians3D
    full_moment: Gaussians3D
    partitions: tuple[MaskedDensityPartition, ...]
    diagnostics: dict[str, object]


def _gauss_hermite_2d(order: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard-normal quadrature points and normalized tensor-product weights."""
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    nodes = math.sqrt(2.0) * nodes
    weights = weights / math.sqrt(math.pi)
    grid_x, grid_y = np.meshgrid(nodes, nodes, indexing="xy")
    weight_x, weight_y = np.meshgrid(weights, weights, indexing="xy")
    points = torch.from_numpy(np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1))
    point_weights = torch.from_numpy((weight_x * weight_y).reshape(-1))
    return points.to(torch.float64), point_weights.to(torch.float64)


def _masked_quadrature_samples(
    field: GaussianObservationField,
    alpha: PackedAlpha,
    config: MaskedPartitionConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Sample every source Gaussian and retain only quadrature mass inside ``alpha``."""
    fit_x, fit_y, fit_width, fit_height = field.fit_window
    if alpha.origin != (fit_x, fit_y) or alpha.shape != (fit_height, fit_width):
        raise ValueError("packed alpha must exactly cover the observation fit window")

    means = field.native_means(dtype=torch.float64)
    covariances = _component_covariances_2d(field)
    standard_points, quadrature_weights = _gauss_hermite_2d(config.quadrature_order)
    cholesky = torch.linalg.cholesky(covariances)
    offsets = torch.einsum("nij,qj->nqi", cholesky, standard_points)
    samples = means[:, None, :] + offsets

    integrated_mass = (
        field.amplitudes.to(torch.float64)
        * (2.0 * math.pi)
        * torch.linalg.det(covariances).clamp_min(0).sqrt()
    )
    sample_mass = integrated_mass[:, None] * quadrature_weights[None, :]

    flat_samples = samples.reshape(-1, 2)
    flat_mass = sample_mass.reshape(-1)
    local_x = torch.floor(flat_samples[:, 0]).to(torch.long) - fit_x
    local_y = torch.floor(flat_samples[:, 1]).to(torch.long) - fit_y
    in_crop = (local_x >= 0) & (local_x < fit_width) & (local_y >= 0) & (local_y < fit_height)
    in_mask = torch.zeros_like(in_crop)
    crop_mask = alpha.crop_mask("cpu")
    valid = in_crop.nonzero(as_tuple=True)[0]
    in_mask[valid] = crop_mask[local_y[valid], local_x[valid]]
    retained = in_mask & (flat_mass > 0)

    retained_samples = flat_samples[retained]
    retained_mass = flat_mass[retained]
    diagnostics: dict[str, object] = {
        "source_components": field.n,
        "quadrature_order": config.quadrature_order,
        "quadrature_samples": int(flat_samples.shape[0]),
        "positive_masked_samples": int(retained.sum()),
        "integrated_mass_unmasked": float(integrated_mass.sum()),
        "quadrature_mass_inside_mask": float(retained_mass.sum()),
        "quadrature_masked_mass_fraction": float(
            retained_mass.sum() / integrated_mass.sum().clamp_min(1e-30)
        ),
    }
    return retained_samples, retained_mass, diagnostics


def partition_masked_gaussian_density(
    field: GaussianObservationField,
    alpha: PackedAlpha,
    anchor_component_indices: torch.Tensor,
    config: MaskedPartitionConfig | None = None,
) -> MaskedDensityPartition:
    """Partition one complete masked 2D field around exact native contributor anchors.

    Each positive quadrature sample receives a hard nearest-anchor responsibility, so
    responsibilities sum to one at every retained sample.  Covariances are fixed-anchor second
    moments rather than moments about a newly estimated cluster center.
    """
    if config is None:
        config = MaskedPartitionConfig()
    if field.device.type != "cpu":
        raise ValueError("masked density partition currently requires a CPU observation")
    anchors = torch.as_tensor(anchor_component_indices, dtype=torch.long, device="cpu")
    if anchors.ndim != 1 or anchors.numel() == 0:
        raise ValueError("anchor_component_indices must be a non-empty 1D tensor")
    if bool((anchors < 0).any()) or bool((anchors >= field.n).any()):
        raise ValueError("anchor_component_indices contain an out-of-range component")
    if anchors.unique().numel() != anchors.numel():
        raise ValueError("anchor_component_indices must be unique")
    if not torch.equal(anchors, anchors.sort().values):
        raise ValueError("anchor_component_indices must be sorted")

    samples, sample_mass, sample_diagnostics = _masked_quadrature_samples(field, alpha, config)
    if samples.shape[0] == 0:
        raise ValueError("foreground mask retained no positive source density")
    anchor_means = field.native_means(anchors, dtype=torch.float64)
    assignments = torch.empty(samples.shape[0], dtype=torch.long)
    for start in range(0, samples.shape[0], config.assignment_chunk):
        stop = min(start + config.assignment_chunk, samples.shape[0])
        assignments[start:stop] = torch.cdist(samples[start:stop], anchor_means).argmin(dim=1)

    masses = torch.zeros(anchors.numel(), dtype=torch.float64)
    moments = torch.zeros(anchors.numel(), 2, 2, dtype=torch.float64)
    masses.index_add_(0, assignments, sample_mass)
    deltas = samples - anchor_means[assignments]
    outer = deltas[:, :, None] * deltas[:, None, :]
    moments.index_add_(0, assignments, sample_mass[:, None, None] * outer)
    empty = masses < config.min_partition_mass
    if bool(empty.any()):
        empty_components = anchors[empty].tolist()
        raise ValueError(
            f"masked partition produced anchors without supported density: {empty_components[:16]}"
        )

    covariance = moments / masses[:, None, None]
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(config.min_variance_px)
    covariance = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)

    native_covariance = _component_covariances_2d(field)[anchors]
    determinant_ratio = torch.linalg.det(covariance).clamp_min(
        config.min_variance_px**2
    ) / torch.linalg.det(native_covariance).clamp_min(config.min_variance_px**2)
    area_matched = native_covariance * determinant_ratio.sqrt()[:, None, None]
    partition_mass = float(masses.sum())
    source_mass = float(sample_mass.sum())
    diagnostics = {
        **sample_diagnostics,
        "unique_anchors": int(anchors.numel()),
        "partition_mass": partition_mass,
        "partition_of_unity_absolute_error": abs(partition_mass - source_mass),
        "partition_of_unity_relative_error": abs(partition_mass - source_mass)
        / max(source_mass, 1e-30),
        "partition_mass_min": float(masses.min()),
        "partition_mass_median": float(masses.median()),
        "partition_mass_max": float(masses.max()),
        "area_scale_min": float(determinant_ratio.sqrt().min()),
        "area_scale_median": float(determinant_ratio.sqrt().median()),
        "area_scale_max": float(determinant_ratio.sqrt().max()),
    }
    return MaskedDensityPartition(
        anchor_component_indices=anchors,
        covariances2d=covariance,
        area_matched_covariances2d=area_matched,
        masses=masses,
        diagnostics=diagnostics,
    )


def _replace_covariances(
    base: Gaussians3D,
    covariances: torch.Tensor,
    *,
    min_sigma: float,
) -> Gaussians3D:
    converted = Gaussians3D.from_means_covs(
        means=base.means,
        covs=covariances.to(base.means),
        colors=torch.zeros(base.n, 3, dtype=base.means.dtype, device=base.means.device),
        opacity=base.opacity,
        sh_degree=base.sh_degree,
        min_scale=min_sigma,
    )
    result = Gaussians3D(
        means=base.means.detach().clone(),
        quats=converted.quats.detach().clone(),
        log_scales=converted.log_scales.detach().clone(),
        opacity=base.opacity.detach().clone(),
        sh=base.sh.detach().clone(),
    )
    if (
        not torch.equal(result.means, base.means)
        or not torch.equal(result.opacity, base.opacity)
        or not torch.equal(result.sh, base.sh)
        or result.n != base.n
    ):
        raise RuntimeError("covariance refit changed a frozen Beam Fusion field")
    return result


def _refit_beam_covariances(
    inputs: ReconstructionInputs,
    result: BeamFusionResult,
    beam_config: BeamFusionConfig,
    covariance_tables: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Repeat only Beam Fusion's CI precision average with substituted 2D covariances."""
    if len(covariance_tables) != inputs.n_views:
        raise ValueError("covariance_tables must contain one table per input view")
    views = _prepare_views(inputs, beam_config)
    prepared = []
    for view_index, (view, table) in enumerate(zip(views, covariance_tables, strict=True)):
        table = table.to(torch.float64)
        if table.shape != view.covariances2d.shape:
            raise ValueError(f"view {view_index} covariance table has the wrong shape")
        if not bool(torch.isfinite(table).all()) or not bool(
            (torch.linalg.eigvalsh(table) > 0).all()
        ):
            raise ValueError(f"view {view_index} covariance table must be finite and SPD")
        prepared.append(replace(view, covariances2d=table))

    count = result.n_components
    offsets = result.component_offsets.to(torch.long)
    lengths = offsets[1:] - offsets[:-1]
    n_links = int(lengths.sum())
    arrays = (
        result.contributor_view_indices,
        result.contributor_component_indices,
        result.contributor_depths,
    )
    if offsets.shape != (count + 1,) or any(array.shape != (n_links,) for array in arrays):
        raise ValueError("Beam Fusion CSR lineage is inconsistent")
    if bool((lengths <= 0).any()):
        raise ValueError("every Beam Fusion component must retain a contributor")
    component_rows = torch.repeat_interleave(torch.arange(count, dtype=torch.long), lengths)
    precision_sum = torch.zeros(count, 3, 3, dtype=torch.float64)

    for view_index, (view, camera) in enumerate(zip(prepared, inputs.cameras, strict=True)):
        links = (result.contributor_view_indices == view_index).nonzero(as_tuple=True)[0]
        if links.numel() == 0:
            continue
        rows = component_rows[links]
        indices = result.contributor_component_indices[links].to(torch.long)
        depths = result.contributor_depths[links].to(torch.float64)
        if not bool(torch.isfinite(depths).all()) or bool((depths <= 0).any()):
            raise ValueError("Beam Fusion contributor depths must be finite and positive")
        half_length = (
            0.5
            * (view.depth_hi[indices] - view.depth_lo[indices]).clamp_min(1e-6)
            * view.dirs_world[indices].norm(dim=1)
        )
        precisions = _beam_precisions(view, camera, indices, depths, half_length)
        precision_sum.index_add_(0, rows, precisions)

    fused_precision = precision_sum / lengths.to(torch.float64)[:, None, None]
    covariance = torch.linalg.inv(fused_precision)
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    _, extent = _center_and_extent(inputs, torch.float64)
    max_sigma = (
        beam_config.max_sigma_world
        if beam_config.max_sigma_world is not None
        else 0.5 * float(extent)
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp(
        min=beam_config.min_sigma_world**2,
        max=max_sigma**2,
    )
    return eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)


def refit_beam_covariances_from_masked_partitions(
    inputs: ReconstructionInputs,
    alphas: Sequence[PackedAlpha | None],
    result: BeamFusionResult,
    beam_config: BeamFusionConfig,
    partition_config: MaskedPartitionConfig | None = None,
) -> BeamPartitionCovarianceResult:
    """Build area-only and full-moment covariance treatments from Beam CSR anchors."""
    if partition_config is None:
        partition_config = MaskedPartitionConfig()
    _validate_cpu_inputs(inputs)
    if len(alphas) != inputs.n_views:
        raise ValueError("alphas must contain one entry per input view")
    if result.n_components != result.gaussians.n:
        raise ValueError("Beam Fusion result count is inconsistent")

    partitions = []
    native_tables = []
    area_tables = []
    full_tables = []
    contributor_links_per_view = []
    for view_index, (field, alpha) in enumerate(zip(inputs.observations, alphas, strict=True)):
        if alpha is None:
            raise ValueError(f"view {view_index} has no packed foreground mask")
        view_links = result.contributor_view_indices == view_index
        anchor_indices = torch.unique(
            result.contributor_component_indices[view_links],
            sorted=True,
        )
        if anchor_indices.numel() == 0:
            raise ValueError(f"view {view_index} has no retained Beam Fusion contributor")
        partition = partition_masked_gaussian_density(
            field,
            alpha,
            anchor_indices,
            partition_config,
        )
        native = _component_covariances_2d(field)
        area = native.clone()
        full = native.clone()
        area[anchor_indices] = partition.area_matched_covariances2d
        full[anchor_indices] = partition.covariances2d
        partitions.append(partition)
        native_tables.append(native)
        area_tables.append(area)
        full_tables.append(full)
        contributor_links_per_view.append(int(view_links.sum()))

    native_roundtrip = _refit_beam_covariances(inputs, result, beam_config, native_tables)
    area_covariances = _refit_beam_covariances(inputs, result, beam_config, area_tables)
    full_covariances = _refit_beam_covariances(inputs, result, beam_config, full_tables)
    base_covariances = result.gaussians.covariance().to(torch.float64)
    native_error = (native_roundtrip - base_covariances).norm(dim=(-2, -1)) / (
        base_covariances.norm(dim=(-2, -1)).clamp_min(1e-30)
    )
    if float(native_error.max()) > 1e-4:
        raise RuntimeError(
            "native Beam covariance round trip did not reproduce the stored CI covariance"
        )

    area = _replace_covariances(
        result.gaussians,
        area_covariances,
        min_sigma=beam_config.min_sigma_world,
    )
    full = _replace_covariances(
        result.gaussians,
        full_covariances,
        min_sigma=beam_config.min_sigma_world,
    )
    total_links = int(result.contributor_view_indices.numel())
    total_unique = sum(int(part.anchor_component_indices.numel()) for part in partitions)
    diagnostics: dict[str, object] = {
        "method": "native-anchor-hard-voronoi-masked-density",
        "uses_3d_projection_for_anchor_discovery": False,
        "quadrature_order": partition_config.quadrature_order,
        "assignment_chunk": partition_config.assignment_chunk,
        "n_components": result.n_components,
        "n_contributor_links": total_links,
        "n_unique_view_component_anchors": total_unique,
        "n_duplicate_contributor_links": total_links - total_unique,
        "contributor_links_per_view": contributor_links_per_view,
        "native_ci_roundtrip_relative_error": {
            "mean": float(native_error.mean()),
            "median": float(native_error.median()),
            "max": float(native_error.max()),
        },
        "views": [partition.diagnostics for partition in partitions],
        "frozen_fields": {
            "area_matched": {
                "same_count": area.n == result.gaussians.n,
                "means_bit_exact": torch.equal(area.means, result.gaussians.means),
                "opacity_bit_exact": torch.equal(area.opacity, result.gaussians.opacity),
                "sh_bit_exact": torch.equal(area.sh, result.gaussians.sh),
            },
            "full_moment": {
                "same_count": full.n == result.gaussians.n,
                "means_bit_exact": torch.equal(full.means, result.gaussians.means),
                "opacity_bit_exact": torch.equal(full.opacity, result.gaussians.opacity),
                "sh_bit_exact": torch.equal(full.sh, result.gaussians.sh),
            },
        },
    }
    return BeamPartitionCovarianceResult(
        area_matched=area,
        full_moment=full,
        partitions=tuple(partitions),
        diagnostics=diagnostics,
    )


__all__ = [
    "BeamPartitionCovarianceResult",
    "MaskedDensityPartition",
    "MaskedPartitionConfig",
    "partition_masked_gaussian_density",
    "refit_beam_covariances_from_masked_partitions",
]
