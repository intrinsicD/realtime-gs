"""Deterministic correspondence-confidence gate for dense compact initialization.

The dense compact initializer lifts one candidate per supported 2D component.  A voxel merge
deduplicates those candidates, and its group map exposes a useful byproduct: which source rays
contributed to each merged Gaussian.  This module classifies only the high-confidence ("easy")
clusters using placement-time signals and cross-view reprojection consistency.  It consumes no
decoded RGB image and performs no render-quality measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.compact_carve import CompactCandidateAudit, CompactInitializationResult

_QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)


@dataclass(frozen=True)
class ClusterConfidenceConfig:
    """Frozen thresholds for the correspondence-confidence classifier."""

    min_view_multiplicity: int = 2
    max_spread_rms_voxels: float = 0.50
    max_half_max_width: float = 0.20
    min_best_n_covered: int = 2
    max_reprojection_residual_px: float = 16.0

    def __post_init__(self) -> None:
        if isinstance(self.min_view_multiplicity, bool) or not isinstance(
            self.min_view_multiplicity, int
        ):
            raise TypeError("min_view_multiplicity must be an integer")
        if isinstance(self.min_best_n_covered, bool) or not isinstance(
            self.min_best_n_covered, int
        ):
            raise TypeError("min_best_n_covered must be an integer")
        if self.min_view_multiplicity <= 0:
            raise ValueError("min_view_multiplicity must be positive")
        if self.min_best_n_covered <= 0:
            raise ValueError("min_best_n_covered must be positive")
        for name in (
            "max_spread_rms_voxels",
            "max_half_max_width",
            "max_reprojection_residual_px",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    def as_dict(self) -> dict[str, int | float]:
        return {
            "min_view_multiplicity": self.min_view_multiplicity,
            "max_spread_rms_voxels": self.max_spread_rms_voxels,
            "max_half_max_width": self.max_half_max_width,
            "min_best_n_covered": self.min_best_n_covered,
            "max_reprojection_residual_px": self.max_reprojection_residual_px,
        }


@dataclass(frozen=True)
class ClusterConfidenceRecord:
    """Signals and exact keep decision for one merged cluster."""

    cluster_index: int
    member_count: int
    view_multiplicity: int
    spread_rms_voxels: float
    spread_max_voxels: float
    half_max_width_max: float
    score_margin_min: float
    best_n_covered_min: int
    consensus_color_variance: float
    reprojection_rms_px: float
    reprojection_max_px: float
    kept: bool
    failures: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "cluster_index": self.cluster_index,
            "member_count": self.member_count,
            "view_multiplicity": self.view_multiplicity,
            "spread_rms_voxels": self.spread_rms_voxels,
            "spread_max_voxels": self.spread_max_voxels,
            "half_max_width_max": self.half_max_width_max,
            "score_margin_min": self.score_margin_min,
            "best_n_covered_min": self.best_n_covered_min,
            "consensus_color_variance": self.consensus_color_variance,
            "reprojection_rms_px": self.reprojection_rms_px,
            "reprojection_max_px": self.reprojection_max_px,
            "kept": self.kept,
            "failures": list(self.failures),
        }


@dataclass(frozen=True)
class ConfidenceGateResult:
    """Easy-only Gaussian subset plus a complete audit of every merged cluster."""

    gaussians: Gaussians3D
    keep_mask: torch.Tensor
    records: tuple[ClusterConfidenceRecord, ...]
    config: ClusterConfidenceConfig

    @property
    def kept_count(self) -> int:
        return int(self.keep_mask.sum())

    @property
    def dropped_count(self) -> int:
        return int(self.keep_mask.numel() - self.keep_mask.sum())

    def as_dict(self, *, include_records: bool = True) -> dict[str, object]:
        failure_histogram: dict[str, int] = {}
        for record in self.records:
            for failure in record.failures:
                failure_histogram[failure] = failure_histogram.get(failure, 0) + 1
        payload: dict[str, object] = {
            "config": self.config.as_dict(),
            "kept_count": self.kept_count,
            "dropped_count": self.dropped_count,
            "failure_histogram": failure_histogram,
            "signal_quantiles": _signal_quantiles(self.records),
        }
        if include_records:
            payload["records"] = [record.as_dict() for record in self.records]
        return payload


def _selected_audit_rows(
    dense: CompactInitializationResult,
    audit: CompactCandidateAudit,
) -> torch.Tensor:
    selected = audit.selected_candidate_indices
    if selected.ndim != 1 or selected.dtype != torch.long:
        raise ValueError("candidate audit selected indices must be a one-dimensional int64 tensor")
    if int(selected.numel()) != dense.gaussians.n:
        raise ValueError("candidate audit selection count does not match dense initialization")
    if selected.numel() and (
        int(selected.min()) < 0
        or int(selected.max()) >= int(audit.candidate_source_view_indices.numel())
    ):
        raise ValueError("candidate audit selected indices are out of range")
    selected = selected.cpu()
    comparisons = (
        (
            "source view",
            audit.candidate_source_view_indices[selected],
            dense.lineage.source_view_indices,
        ),
        (
            "source component",
            audit.candidate_source_component_indices[selected],
            dense.lineage.source_component_indices,
        ),
        ("source coordinate", audit.candidate_source_xy[selected], dense.lineage.source_xy),
        ("best mean", audit.candidate_best_means[selected], dense.gaussians.means),
    )
    for label, observed, expected in comparisons:
        if not torch.equal(observed.cpu(), expected.cpu()):
            raise ValueError(f"candidate audit selected {label} rows do not match dense lineage")
    return selected


def _validate_group(
    dense: CompactInitializationResult,
    merged: Gaussians3D,
    group: torch.Tensor,
) -> torch.Tensor:
    if group.ndim != 1 or group.dtype != torch.long or group.shape[0] != dense.gaussians.n:
        raise ValueError("group must be a one-dimensional int64 map for every dense Gaussian")
    group = group.cpu()
    if merged.n == 0:
        if group.numel() != 0:
            raise ValueError("non-empty group cannot map to an empty merged initialization")
        return group
    if group.numel() == 0 or int(group.min()) != 0 or int(group.max()) != merged.n - 1:
        raise ValueError("group must use every canonical merged cluster index")
    if int(group.unique().numel()) != merged.n:
        raise ValueError("group must map at least one dense Gaussian to every merged cluster")
    return group


def _signal_quantiles(
    records: tuple[ClusterConfidenceRecord, ...],
) -> dict[str, dict[str, float]]:
    fields = (
        "view_multiplicity",
        "member_count",
        "spread_rms_voxels",
        "spread_max_voxels",
        "half_max_width_max",
        "score_margin_min",
        "best_n_covered_min",
        "consensus_color_variance",
        "reprojection_rms_px",
        "reprojection_max_px",
    )
    if not records:
        return {field: {} for field in fields}
    quantiles: dict[str, dict[str, float]] = {}
    for field in fields:
        values = torch.tensor([float(getattr(record, field)) for record in records])
        quantiles[field] = {
            f"{quantile:g}": float(torch.quantile(values, quantile)) for quantile in _QUANTILES
        }
    return quantiles


def gate_merged_initialization(
    inputs: ReconstructionInputs,
    dense: CompactInitializationResult,
    audit: CompactCandidateAudit,
    merged: Gaussians3D,
    group: torch.Tensor,
    *,
    merge_voxel_size: float,
    config: ClusterConfidenceConfig | None = None,
) -> ConfidenceGateResult:
    """Keep merged clusters that pass every configured placement-confidence condition.

    The candidate audit is first matched back to the dense output exactly; this prevents signal
    rows from being applied to the wrong lineage.  All diagnostics are computed on detached CPU
    tensors, making this a deterministic CPU-first post-process even when later evaluation uses a
    GPU rasterizer.
    """
    if not math.isfinite(merge_voxel_size) or merge_voxel_size <= 0:
        raise ValueError("merge_voxel_size must be finite and positive")
    config = config or ClusterConfidenceConfig()
    selected = _selected_audit_rows(dense, audit)
    group = _validate_group(dense, merged, group)

    source_views = dense.lineage.source_view_indices.detach().cpu()
    source_xy = dense.lineage.source_xy.detach().cpu()
    dense_means = dense.gaussians.means.detach().cpu()
    merged_means = merged.means.detach().cpu()
    half_widths = audit.candidate_half_max_widths[selected].detach().cpu()
    score_margins = audit.candidate_score_margins[selected].detach().cpu()
    best_n_covered = audit.candidate_best_n_covered[selected].detach().cpu()
    consensus_colors = audit.candidate_consensus_colors[selected].detach().cpu()

    records: list[ClusterConfidenceRecord] = []
    keep_mask = torch.zeros(merged.n, dtype=torch.bool)
    for cluster_index in range(merged.n):
        members = (group == cluster_index).nonzero(as_tuple=True)[0]
        center = merged_means[cluster_index]
        distances = torch.linalg.vector_norm(dense_means[members] - center, dim=-1)
        reprojection_residuals: list[float] = []
        for row in members.tolist():
            view_index = int(source_views[row])
            projected, depth = inputs.cameras[view_index].project(center[None])
            residual = torch.linalg.vector_norm(projected[0].cpu() - source_xy[row])
            if not bool(torch.isfinite(residual)) or float(depth[0]) <= 0:
                residual = torch.tensor(torch.inf)
            reprojection_residuals.append(float(residual))

        reprojection = torch.tensor(reprojection_residuals)
        colors = consensus_colors[members]
        view_multiplicity = int(source_views[members].unique().numel())
        spread_rms_voxels = float(distances.square().mean().sqrt() / merge_voxel_size)
        spread_max_voxels = float(distances.max() / merge_voxel_size)
        half_max_width_max = float(half_widths[members].max())
        score_margin_min = float(score_margins[members].min())
        best_n_covered_min = int(best_n_covered[members].min())
        consensus_color_variance = float((colors - colors.mean(dim=0)).square().mean())
        reprojection_rms_px = float(reprojection.square().mean().sqrt())
        reprojection_max_px = float(reprojection.max())

        failures: list[str] = []
        if view_multiplicity < config.min_view_multiplicity:
            failures.append("view_multiplicity")
        if spread_rms_voxels > config.max_spread_rms_voxels:
            failures.append("spread_rms_voxels")
        if half_max_width_max > config.max_half_max_width:
            failures.append("half_max_width")
        if best_n_covered_min < config.min_best_n_covered:
            failures.append("best_n_covered")
        if reprojection_max_px > config.max_reprojection_residual_px:
            failures.append("reprojection_residual")
        kept = not failures
        keep_mask[cluster_index] = kept
        records.append(
            ClusterConfidenceRecord(
                cluster_index=cluster_index,
                member_count=int(members.numel()),
                view_multiplicity=view_multiplicity,
                spread_rms_voxels=spread_rms_voxels,
                spread_max_voxels=spread_max_voxels,
                half_max_width_max=half_max_width_max,
                score_margin_min=score_margin_min,
                best_n_covered_min=best_n_covered_min,
                consensus_color_variance=consensus_color_variance,
                reprojection_rms_px=reprojection_rms_px,
                reprojection_max_px=reprojection_max_px,
                kept=kept,
                failures=tuple(failures),
            )
        )

    return ConfidenceGateResult(
        gaussians=merged.subset(keep_mask.to(merged.means.device)),
        keep_mask=keep_mask,
        records=tuple(records),
        config=config,
    )


__all__ = [
    "ClusterConfidenceConfig",
    "ClusterConfidenceRecord",
    "ConfidenceGateResult",
    "gate_merged_initialization",
]
