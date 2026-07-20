"""Stage 2: lift per-view 2D gaussians into a world-space 3D gaussian set.

Variants (see docs/ARCHITECTURE.md): `gradient` (multi-view photometric descent),
`depth` (monocular depth backend), `carve` (voxel consistency carving), plus the
`sfm` and `random` baselines. Register new variants in ``_LIFTERS``.
"""

from rtgs.lift.base import CompactInitializer, Lifter
from rtgs.lift.baselines import RandomLifter, SfMLifter
from rtgs.lift.carve import CarveLifter
from rtgs.lift.compact_carve import (
    CompactCandidateAudit,
    CompactCandidateAuditCallback,
    CompactCarveConfig,
    CompactCarveInitializer,
    CompactInitializationResult,
    CompactLineage,
    CompactPointScores,
    CompactRayDepthAuditBatch,
    CompactRayDepthAuditCallback,
    score_world_points,
)
from rtgs.lift.depth import DepthLifter
from rtgs.lift.field_lifter import FieldLiftConfig, FieldLifter, FieldLiftResult
from rtgs.lift.gradient import GradientLifter
from rtgs.lift.hybrid import HybridLifter
from rtgs.lift.inverse_projection_fiber import (
    FreeGaussianGeometry,
    InverseProjectionFiber,
    covariance_projection_design,
    hard_correspondence_loss,
    pairwise_center_cost,
    pairwise_conic_cost,
    pairwise_gaussian_geometry_cost,
    spd_affine_invariant_squared,
)
from rtgs.lift.topology import (
    cyclic_shift_grouped_scores,
    exact_assignment_loss,
    exact_linear_assignment,
    radius_connected_components,
    select_component_representatives,
)

_LIFTERS = {
    "gradient": GradientLifter,
    "depth": DepthLifter,
    "hybrid": HybridLifter,
    "carve": CarveLifter,
    "field": FieldLifter,
    "sfm": SfMLifter,
    "random": RandomLifter,
}


def get_lifter(name: str, **kwargs) -> Lifter:
    """Instantiate a registered lifter by name, forwarding keyword arguments."""
    if name not in _LIFTERS:
        raise ValueError(f"unknown lifter '{name}' (available: {sorted(_LIFTERS)})")
    return _LIFTERS[name](**kwargs)


def lifter_names() -> list[str]:
    """Names of all registered lifting variants."""
    return sorted(_LIFTERS)


__all__ = [
    "CarveLifter",
    "CompactCandidateAudit",
    "CompactCandidateAuditCallback",
    "CompactCarveConfig",
    "CompactCarveInitializer",
    "CompactInitializationResult",
    "CompactInitializer",
    "CompactLineage",
    "CompactPointScores",
    "CompactRayDepthAuditBatch",
    "CompactRayDepthAuditCallback",
    "DepthLifter",
    "FieldLiftConfig",
    "FieldLiftResult",
    "FieldLifter",
    "GradientLifter",
    "HybridLifter",
    "InverseProjectionFiber",
    "Lifter",
    "FreeGaussianGeometry",
    "RandomLifter",
    "SfMLifter",
    "get_lifter",
    "covariance_projection_design",
    "cyclic_shift_grouped_scores",
    "exact_assignment_loss",
    "exact_linear_assignment",
    "hard_correspondence_loss",
    "lifter_names",
    "pairwise_center_cost",
    "pairwise_conic_cost",
    "pairwise_gaussian_geometry_cost",
    "radius_connected_components",
    "score_world_points",
    "select_component_representatives",
    "spd_affine_invariant_squared",
]
