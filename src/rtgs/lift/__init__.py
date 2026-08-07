"""Stage 2: lift compact per-view observations into world-space 3D Gaussians.

The package initializer must remain CPU-first and input-neutral.  In particular, importing a
compact lifter such as :mod:`rtgs.lift.beam_fusion` must not eagerly import legacy RGB-image
lifters or ``SceneData``.  Public compatibility attributes are therefore resolved lazily.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rtgs.lift.base import CompactInitializer, Lifter
    from rtgs.lift.baselines import RandomLifter, SfMLifter
    from rtgs.lift.carrier_refinement import (
        CarrierObservationTable,
        CarrierRepairConfig,
        CarrierRepairResult,
    )
    from rtgs.lift.carve import CarveLifter
    from rtgs.lift.compact_carve import (
        CompactCandidateAudit,
        CompactCandidateAuditCallback,
        CompactCarveConfig,
        CompactCarveInitializer,
        CompactInitializationResult,
        CompactLineage,
        CompactPlacementProgress,
        CompactPlacementProgressCallback,
        CompactPointScores,
        CompactRayDepthAuditBatch,
        CompactRayDepthAuditCallback,
    )
    from rtgs.lift.compact_confidence_gate import (
        ClusterConfidenceConfig,
        ClusterConfidenceRecord,
        ConfidenceGateResult,
    )
    from rtgs.lift.compact_init_eval import (
        InitEvaluation,
        InitEvaluationTarget,
        InitViewMetrics,
    )
    from rtgs.lift.compact_refine import (
        LocalDepthRefineConfig,
        LocalDepthRefineResult,
    )
    from rtgs.lift.depth import DepthLifter
    from rtgs.lift.field_lifter import (
        FieldAssociationConfig,
        FieldLiftConfig,
        FieldLifter,
        FieldLiftResult,
    )
    from rtgs.lift.gradient import GradientLifter
    from rtgs.lift.hybrid import HybridLifter
    from rtgs.lift.inverse_projection_fiber import (
        FreeGaussianGeometry,
        InverseProjectionFiber,
    )
    from rtgs.lift.probabilistic_pipeline import (
        IndependentHalfStability,
        ProbabilisticFieldPipelineConfig,
        ProbabilisticFieldPipelineResult,
    )

_LIFTER_MODULES = {
    "gradient": ("rtgs.lift.gradient", "GradientLifter"),
    "depth": ("rtgs.lift.depth", "DepthLifter"),
    "hybrid": ("rtgs.lift.hybrid", "HybridLifter"),
    "carve": ("rtgs.lift.carve", "CarveLifter"),
    "field": ("rtgs.lift.field_lifter", "FieldLifter"),
    "sfm": ("rtgs.lift.baselines", "SfMLifter"),
    "random": ("rtgs.lift.baselines", "RandomLifter"),
}

_LAZY_ATTRIBUTES = {
    "CompactInitializer": ("rtgs.lift.base", "CompactInitializer"),
    "Lifter": ("rtgs.lift.base", "Lifter"),
    "RandomLifter": ("rtgs.lift.baselines", "RandomLifter"),
    "SfMLifter": ("rtgs.lift.baselines", "SfMLifter"),
    "CarrierObservationTable": (
        "rtgs.lift.carrier_refinement",
        "CarrierObservationTable",
    ),
    "CarrierRepairConfig": ("rtgs.lift.carrier_refinement", "CarrierRepairConfig"),
    "CarrierRepairResult": ("rtgs.lift.carrier_refinement", "CarrierRepairResult"),
    "build_carrier_observation_table": (
        "rtgs.lift.carrier_refinement",
        "build_carrier_observation_table",
    ),
    "repair_beam_carriers": (
        "rtgs.lift.carrier_refinement",
        "repair_beam_carriers",
    ),
    "CarveLifter": ("rtgs.lift.carve", "CarveLifter"),
    "CompactCandidateAudit": ("rtgs.lift.compact_carve", "CompactCandidateAudit"),
    "CompactCandidateAuditCallback": (
        "rtgs.lift.compact_carve",
        "CompactCandidateAuditCallback",
    ),
    "CompactCarveConfig": ("rtgs.lift.compact_carve", "CompactCarveConfig"),
    "CompactCarveInitializer": (
        "rtgs.lift.compact_carve",
        "CompactCarveInitializer",
    ),
    "CompactInitializationResult": (
        "rtgs.lift.compact_carve",
        "CompactInitializationResult",
    ),
    "CompactLineage": ("rtgs.lift.compact_carve", "CompactLineage"),
    "CompactPlacementProgress": (
        "rtgs.lift.compact_carve",
        "CompactPlacementProgress",
    ),
    "CompactPlacementProgressCallback": (
        "rtgs.lift.compact_carve",
        "CompactPlacementProgressCallback",
    ),
    "CompactPointScores": ("rtgs.lift.compact_carve", "CompactPointScores"),
    "CompactRayDepthAuditBatch": (
        "rtgs.lift.compact_carve",
        "CompactRayDepthAuditBatch",
    ),
    "CompactRayDepthAuditCallback": (
        "rtgs.lift.compact_carve",
        "CompactRayDepthAuditCallback",
    ),
    "make_placement_progress_printer": (
        "rtgs.lift.compact_carve",
        "make_placement_progress_printer",
    ),
    "score_world_points": ("rtgs.lift.compact_carve", "score_world_points"),
    "ClusterConfidenceConfig": (
        "rtgs.lift.compact_confidence_gate",
        "ClusterConfidenceConfig",
    ),
    "ClusterConfidenceRecord": (
        "rtgs.lift.compact_confidence_gate",
        "ClusterConfidenceRecord",
    ),
    "ConfidenceGateResult": (
        "rtgs.lift.compact_confidence_gate",
        "ConfidenceGateResult",
    ),
    "gate_merged_initialization": (
        "rtgs.lift.compact_confidence_gate",
        "gate_merged_initialization",
    ),
    "InitEvaluation": ("rtgs.lift.compact_init_eval", "InitEvaluation"),
    "InitEvaluationTarget": (
        "rtgs.lift.compact_init_eval",
        "InitEvaluationTarget",
    ),
    "InitViewMetrics": ("rtgs.lift.compact_init_eval", "InitViewMetrics"),
    "dense_merged_initialization": (
        "rtgs.lift.compact_init_eval",
        "dense_merged_initialization",
    ),
    "evaluate_initialization": (
        "rtgs.lift.compact_init_eval",
        "evaluate_initialization",
    ),
    "merge_initialization": (
        "rtgs.lift.compact_init_eval",
        "merge_initialization",
    ),
    "prepare_evaluation_targets": (
        "rtgs.lift.compact_init_eval",
        "prepare_evaluation_targets",
    ),
    "render_teacher_image": (
        "rtgs.lift.compact_init_eval",
        "render_teacher_image",
    ),
    "LocalDepthRefineConfig": (
        "rtgs.lift.compact_refine",
        "LocalDepthRefineConfig",
    ),
    "LocalDepthRefineResult": (
        "rtgs.lift.compact_refine",
        "LocalDepthRefineResult",
    ),
    "build_fiber_from_initialization": (
        "rtgs.lift.compact_refine",
        "build_fiber_from_initialization",
    ),
    "refine_initialization_depths": (
        "rtgs.lift.compact_refine",
        "refine_initialization_depths",
    ),
    "DepthLifter": ("rtgs.lift.depth", "DepthLifter"),
    "FieldAssociationConfig": ("rtgs.lift.field_lifter", "FieldAssociationConfig"),
    "FieldLiftConfig": ("rtgs.lift.field_lifter", "FieldLiftConfig"),
    "FieldLiftResult": ("rtgs.lift.field_lifter", "FieldLiftResult"),
    "FieldLifter": ("rtgs.lift.field_lifter", "FieldLifter"),
    "GradientLifter": ("rtgs.lift.gradient", "GradientLifter"),
    "HybridLifter": ("rtgs.lift.hybrid", "HybridLifter"),
    "FreeGaussianGeometry": (
        "rtgs.lift.inverse_projection_fiber",
        "FreeGaussianGeometry",
    ),
    "InverseProjectionFiber": (
        "rtgs.lift.inverse_projection_fiber",
        "InverseProjectionFiber",
    ),
    "IndependentHalfStability": (
        "rtgs.lift.probabilistic_pipeline",
        "IndependentHalfStability",
    ),
    "ProbabilisticFieldPipelineConfig": (
        "rtgs.lift.probabilistic_pipeline",
        "ProbabilisticFieldPipelineConfig",
    ),
    "ProbabilisticFieldPipelineResult": (
        "rtgs.lift.probabilistic_pipeline",
        "ProbabilisticFieldPipelineResult",
    ),
    "covariance_projection_design": (
        "rtgs.lift.inverse_projection_fiber",
        "covariance_projection_design",
    ),
    "hard_correspondence_loss": (
        "rtgs.lift.inverse_projection_fiber",
        "hard_correspondence_loss",
    ),
    "pairwise_center_cost": (
        "rtgs.lift.inverse_projection_fiber",
        "pairwise_center_cost",
    ),
    "pairwise_conic_cost": (
        "rtgs.lift.inverse_projection_fiber",
        "pairwise_conic_cost",
    ),
    "pairwise_gaussian_geometry_cost": (
        "rtgs.lift.inverse_projection_fiber",
        "pairwise_gaussian_geometry_cost",
    ),
    "spd_affine_invariant_squared": (
        "rtgs.lift.inverse_projection_fiber",
        "spd_affine_invariant_squared",
    ),
    "cyclic_shift_grouped_scores": (
        "rtgs.lift.topology",
        "cyclic_shift_grouped_scores",
    ),
    "exact_assignment_loss": ("rtgs.lift.topology", "exact_assignment_loss"),
    "exact_linear_assignment": ("rtgs.lift.topology", "exact_linear_assignment"),
    "radius_connected_components": (
        "rtgs.lift.topology",
        "radius_connected_components",
    ),
    "select_component_representatives": (
        "rtgs.lift.topology",
        "select_component_representatives",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_ATTRIBUTES[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def get_lifter(name: str, **kwargs: Any) -> Any:
    """Instantiate a registered lifter by name, forwarding keyword arguments."""

    try:
        module_name, attribute_name = _LIFTER_MODULES[name]
    except KeyError as error:
        raise ValueError(f"unknown lifter {name!r} (available: {lifter_names()})") from error
    lifter_type = getattr(import_module(module_name), attribute_name)
    return lifter_type(**kwargs)


def lifter_names() -> list[str]:
    """Names of all registered lifting variants."""

    return sorted(_LIFTER_MODULES)


__all__ = [
    "CarveLifter",
    "CarrierObservationTable",
    "CarrierRepairConfig",
    "CarrierRepairResult",
    "CompactCandidateAudit",
    "CompactCandidateAuditCallback",
    "CompactCarveConfig",
    "CompactCarveInitializer",
    "ClusterConfidenceConfig",
    "ClusterConfidenceRecord",
    "ConfidenceGateResult",
    "CompactInitializationResult",
    "CompactInitializer",
    "CompactLineage",
    "CompactPlacementProgress",
    "CompactPlacementProgressCallback",
    "CompactPointScores",
    "CompactRayDepthAuditBatch",
    "CompactRayDepthAuditCallback",
    "DepthLifter",
    "FieldAssociationConfig",
    "InitEvaluation",
    "InitEvaluationTarget",
    "InitViewMetrics",
    "LocalDepthRefineConfig",
    "LocalDepthRefineResult",
    "FieldLiftConfig",
    "FieldLiftResult",
    "FieldLifter",
    "GradientLifter",
    "HybridLifter",
    "InverseProjectionFiber",
    "IndependentHalfStability",
    "Lifter",
    "FreeGaussianGeometry",
    "RandomLifter",
    "ProbabilisticFieldPipelineConfig",
    "ProbabilisticFieldPipelineResult",
    "SfMLifter",
    "get_lifter",
    "gate_merged_initialization",
    "build_fiber_from_initialization",
    "build_carrier_observation_table",
    "covariance_projection_design",
    "cyclic_shift_grouped_scores",
    "dense_merged_initialization",
    "evaluate_initialization",
    "exact_assignment_loss",
    "exact_linear_assignment",
    "hard_correspondence_loss",
    "lifter_names",
    "make_placement_progress_printer",
    "merge_initialization",
    "prepare_evaluation_targets",
    "pairwise_center_cost",
    "pairwise_conic_cost",
    "pairwise_gaussian_geometry_cost",
    "radius_connected_components",
    "refine_initialization_depths",
    "repair_beam_carriers",
    "render_teacher_image",
    "score_world_points",
    "select_component_representatives",
    "spd_affine_invariant_squared",
]
