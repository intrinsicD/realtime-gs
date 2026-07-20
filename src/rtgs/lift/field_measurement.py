"""Synthetic measurement controls for field-level Gaussian lifting.

The field fitter deliberately has no component correspondence in its continuous objective.
For tests and diagnostics, however, projected ground-truth Gaussians provide parent labels for
free.  This module creates deliberately unequal per-view decompositions while retaining those
labels, aggregates oracle parents, and scores soft overlap plans without importing benchmark
code or RGB images.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.sh import eval_sh_preactivation
from rtgs.lift.field_loss import (
    AnalyticGaussianField2D,
    peak_gaussian_product_integrals,
)
from rtgs.render.projection import EWA_DILATION, project_gaussians_ewa


@dataclass(frozen=True)
class LabeledFieldView:
    """One analytic projected field plus its oracle latent-parent ids."""

    field: AnalyticGaussianField2D
    parent_ids: torch.Tensor

    def __post_init__(self) -> None:
        if self.parent_ids.shape != (self.field.n,) or self.parent_ids.dtype != torch.long:
            raise ValueError("parent_ids must have shape (field.n,) and dtype long")
        if self.parent_ids.device != self.field.means.device:
            raise ValueError("parent_ids and field must share a device")
        if self.parent_ids.numel() and bool((self.parent_ids < 0).any()):
            raise ValueError("parent_ids must be non-negative")


@dataclass(frozen=True)
class AssociationMetrics:
    """Mass-weighted purity and completeness for a soft track/observation plan."""

    purity: float
    completeness: float


def project_labeled_fields(
    gaussians: Gaussians3D,
    cameras: Sequence[Camera],
    *,
    split_counts: Sequence[int] | None = None,
    dilation: float = EWA_DILATION,
    split_fraction: float = 0.20,
) -> tuple[LabeledFieldView, ...]:
    """Project GT 3D Gaussians and moment-split them unequally across views.

    ``split_counts[v]`` is the number of children per latent parent in view ``v`` and must be
    one, two, or three.  Splits preserve amplitude, mean, and covariance moments, but not the
    entire mixture field.  This is intentional: it models fitted decomposition mismatch rather
    than serving as a field-invariance construction.
    """

    if not cameras:
        raise ValueError("at least one camera is required")
    counts = tuple(split_counts or (1 + index % 3 for index in range(len(cameras))))
    if len(counts) != len(cameras) or any(count not in {1, 2, 3} for count in counts):
        raise ValueError("split_counts must contain one value in {1,2,3} per view")
    if not 0.0 <= split_fraction < 0.5:
        raise ValueError("split_fraction must be in [0,0.5)")

    result: list[LabeledFieldView] = []
    for camera, count in zip(cameras, counts, strict=True):
        projection = project_gaussians_ewa(gaussians, camera, dilation=dilation)
        directions = torch.nn.functional.normalize(
            gaussians.means - camera.position.to(gaussians.means),
            dim=-1,
        )
        colors = eval_sh_preactivation(gaussians.sh_degree, gaussians.sh, directions)
        amplitude = gaussians.opacity.to(gaussians.means)
        field, parents = _moment_split_projection(
            projection.means2d,
            projection.covariances2d,
            amplitude,
            colors,
            count=count,
            split_fraction=split_fraction,
        )
        result.append(LabeledFieldView(field=field, parent_ids=parents))
    return tuple(result)


def _moment_split_projection(
    means: torch.Tensor,
    covariances: torch.Tensor,
    amplitudes: torch.Tensor,
    colors: torch.Tensor,
    *,
    count: int,
    split_fraction: float,
) -> tuple[AnalyticGaussianField2D, torch.Tensor]:
    parent_count = means.shape[0]
    if count == 1 or split_fraction == 0:
        parent_ids = torch.arange(parent_count, device=means.device, dtype=torch.long)
        return (
            AnalyticGaussianField2D(
                means=means,
                covariances=covariances,
                density_amplitudes=amplitudes,
                rgb_amplitudes=amplitudes[:, None] * colors,
            ),
            parent_ids,
        )

    eigenvalues, eigenvectors = torch.linalg.eigh(covariances)
    axis = eigenvectors[:, :, 0]
    minor = eigenvalues[:, 0]
    if count == 2:
        scalar_offsets = means.new_tensor([-1.0, 1.0])
        offset_scale = (minor * split_fraction).sqrt()
    else:
        scalar_offsets = means.new_tensor([-1.0, 0.0, 1.0])
        # The equal-weight covariance of {-s,0,+s} is 2s²/3.
        offset_scale = (1.5 * minor * split_fraction).sqrt()
    offsets = offset_scale[:, None, None] * scalar_offsets[None, :, None] * axis[:, None, :]
    child_means = means[:, None, :] + offsets
    offset_covariance = torch.einsum("nki,nkj->nij", offsets, offsets) / count
    child_covariance = covariances - offset_covariance
    child_covariances = child_covariance[:, None].expand(-1, count, -1, -1)
    child_amplitudes = amplitudes[:, None].expand(-1, count) / count
    child_colors = colors[:, None].expand(-1, count, -1)
    parent_ids = (
        torch.arange(parent_count, device=means.device, dtype=torch.long)[:, None]
        .expand(-1, count)
        .reshape(-1)
    )
    return (
        AnalyticGaussianField2D(
            means=child_means.reshape(-1, 2),
            covariances=child_covariances.reshape(-1, 2, 2),
            density_amplitudes=child_amplitudes.reshape(-1),
            rgb_amplitudes=(child_amplitudes[:, :, None] * child_colors).reshape(-1, 3),
        ),
        parent_ids,
    )


def oracle_parent_aggregate(view: LabeledFieldView) -> AnalyticGaussianField2D:
    """Moment-merge every oracle parent within one view."""

    if view.parent_ids.numel() == 0:
        return view.field
    parent_count = int(view.parent_ids.max()) + 1
    field = view.field
    means: list[torch.Tensor] = []
    covariances: list[torch.Tensor] = []
    density: list[torch.Tensor] = []
    rgb: list[torch.Tensor] = []
    for parent in range(parent_count):
        selected = view.parent_ids == parent
        mass = field.density_amplitudes[selected]
        total = mass.sum()
        if not bool(total > 0):
            raise ValueError("every oracle parent must have positive density mass")
        normalized = mass / total
        mean = (normalized[:, None] * field.means[selected]).sum(dim=0)
        delta = field.means[selected] - mean
        covariance = (
            normalized[:, None, None]
            * (field.covariances[selected] + delta[:, :, None] * delta[:, None, :])
        ).sum(dim=0)
        means.append(mean)
        covariances.append(covariance)
        density.append(total)
        rgb.append(field.rgb_amplitudes[selected].sum(dim=0))
    return AnalyticGaussianField2D(
        means=torch.stack(means),
        covariances=torch.stack(covariances),
        density_amplitudes=torch.stack(density),
        rgb_amplitudes=torch.stack(rgb),
    )


def normalized_overlap_plan(
    predicted: AnalyticGaussianField2D,
    observed: AnalyticGaussianField2D,
    *,
    dustbin: float = 0.0,
) -> torch.Tensor:
    """Return row-normalized product-kernel overlap as post-hoc correspondence."""

    if dustbin < 0:
        raise ValueError("dustbin must be non-negative")
    overlap = peak_gaussian_product_integrals(
        predicted.means,
        predicted.covariances,
        observed.means,
        observed.covariances,
    )
    overlap = overlap * predicted.density_amplitudes[:, None] * observed.density_amplitudes[None, :]
    denominator = overlap.sum(dim=1, keepdim=True) + float(dustbin)
    return overlap / denominator.clamp_min(torch.finfo(overlap.dtype).tiny)


def association_metrics(
    plan: torch.Tensor,
    predicted_parent_ids: torch.Tensor,
    observed_parent_ids: torch.Tensor,
) -> AssociationMetrics:
    """Score a soft plan against known parent membership.

    Purity asks what fraction of assigned mass agrees with each predicted row's parent.
    Completeness asks what fraction of every observed parent's incoming mass comes from the
    matching predicted parent.  They are identical for a balanced complete plan but deliberately
    differ under dustbins, missing tracks, or duplicate tracks.
    """

    if plan.ndim != 2:
        raise ValueError("plan must be a matrix")
    if predicted_parent_ids.shape != (plan.shape[0],) or observed_parent_ids.shape != (
        plan.shape[1],
    ):
        raise ValueError("parent id shapes must match plan rows and columns")
    if not bool(torch.isfinite(plan).all()) or bool((plan < 0).any()):
        raise ValueError("plan must be finite and non-negative")
    agreement = predicted_parent_ids[:, None] == observed_parent_ids[None, :]
    total = plan.sum().clamp_min(torch.finfo(plan.dtype).tiny)
    purity = (plan * agreement).sum() / total

    observed_total = plan.sum(dim=0)
    supported = observed_total > 0
    if bool(supported.any()):
        correct_per_observation = (plan * agreement).sum(dim=0)
        completeness = (correct_per_observation[supported] / observed_total[supported]).mean()
    else:
        completeness = plan.new_zeros(())
    return AssociationMetrics(
        purity=float(purity.detach()),
        completeness=float(completeness.detach()),
    )


__all__ = [
    "AssociationMetrics",
    "LabeledFieldView",
    "association_metrics",
    "normalized_overlap_plan",
    "oracle_parent_aggregate",
    "project_labeled_fields",
]
