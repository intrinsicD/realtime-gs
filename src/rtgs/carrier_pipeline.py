"""End-to-end compact 2D-Gaussian to 3D-Gaussian carrier pipeline.

This module is the hard image-free carrier boundary.  Its public input type cannot carry RGB,
dense masks, or :class:`SceneData`, and every later stage remains supervised by fitted 2D
Gaussian fields.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.beam_fusion import (
    BeamFusionConfig,
    BeamFusionResult,
    fuse_gaussian_beams,
)
from rtgs.lift.carrier_refinement import (
    CarrierRepairConfig,
    CarrierRepairResult,
    repair_beam_carriers,
)
from rtgs.optim.carrier_schedule import (
    CarrierOptimizationConfig,
    CarrierOptimizationResult,
    optimize_carriers,
)
from rtgs.optim.compact_trainer import CompactTrainer
from rtgs.render.projection import EWA_DILATION


@dataclass(frozen=True)
class CarrierPipelineConfig:
    """Configuration for the bounded compact-only carrier sequence."""

    beam: BeamFusionConfig = field(default_factory=BeamFusionConfig)
    repair: CarrierRepairConfig = field(default_factory=CarrierRepairConfig)
    optimization: CarrierOptimizationConfig = field(default_factory=CarrierOptimizationConfig)
    evaluate_boundaries: bool = False

    def __post_init__(self) -> None:
        if self.beam.min_views < 3:
            raise ValueError("compact carrier pipeline requires at least three contributor views")
        if self.beam.max_components is None:
            raise ValueError("compact carrier pipeline requires a bounded max_components")
        repair = self.repair
        if not (
            repair.repair_covariance
            and not repair.repair_opacity
            and not repair.repair_appearance
            and repair.covariance_residual_mode == "renderer_log"
            and repair.projection_dilation == EWA_DILATION
        ):
            raise ValueError(
                "compact carrier pipeline requires renderer-aware covariance repair only"
            )
        if not isinstance(self.evaluate_boundaries, bool):
            raise TypeError("evaluate_boundaries must be bool")


@dataclass(frozen=True)
class CarrierPipelineResult:
    """Products, compact evaluations, and receipts from every carrier boundary."""

    beam: BeamFusionResult
    repair: CarrierRepairResult
    optimization: CarrierOptimizationResult
    gaussians: Gaussians3D
    metrics: dict[str, float | int]
    evaluations: dict[str, dict]
    timings: dict[str, float]


def run_carrier_pipeline(
    inputs: ReconstructionInputs,
    config: CarrierPipelineConfig | None = None,
) -> CarrierPipelineResult:
    """Run the complete compact-only Beam → repair → optimize/contain sequence."""

    config = config or CarrierPipelineConfig()
    inputs.validate()
    timings: dict[str, float] = {}
    started_total = time.perf_counter()

    started = time.perf_counter()
    beam = fuse_gaussian_beams(inputs, config.beam)
    timings["beam_fusion"] = time.perf_counter() - started
    contributor_counts = beam.component_offsets[1:] - beam.component_offsets[:-1]
    if contributor_counts.numel() == 0 or int(contributor_counts.min()) < 3:
        raise RuntimeError("Beam Fusion violated the three-view carrier lineage invariant")
    if beam.gaussians.n > config.beam.max_components:
        raise RuntimeError("Beam Fusion exceeded the bounded carrier count")

    started = time.perf_counter()
    repair = repair_beam_carriers(beam, inputs, config.repair)
    timings["carrier_covariance_repair"] = time.perf_counter() - started
    if (
        repair.gaussians.n != beam.gaussians.n
        or not torch.equal(repair.gaussians.means, beam.gaussians.means)
        or not torch.equal(repair.gaussians.opacity, beam.gaussians.opacity)
        or not torch.equal(repair.gaussians.sh, beam.gaussians.sh)
    ):
        raise RuntimeError("covariance-only carrier repair changed a frozen field")

    started = time.perf_counter()
    optimization = optimize_carriers(
        inputs,
        repair.gaussians,
        config.optimization,
    )
    timings["compact_carrier_optimization"] = time.perf_counter() - started
    final = optimization.gaussians
    containment = optimization.diagnostics["containment"]
    metrics: dict[str, float | int] = {
        "beam_n_gaussians": beam.gaussians.n,
        "repaired_n_gaussians": repair.gaussians.n,
        "phase1_n_gaussians": optimization.phase_snapshots["phase1"].n,
        "containment_removed_gaussians": containment["removed"],
        "contained_n_gaussians": containment["n_after"],
        "final_n_gaussians": final.n,
        "input_views": inputs.n_views,
        "minimum_contributor_views": int(contributor_counts.min()),
    }

    evaluations: dict[str, dict] = {}
    if config.evaluate_boundaries:
        started = time.perf_counter()
        evaluator = CompactTrainer(config.optimization.template)
        boundaries = {
            "beam": beam.gaussians,
            "repaired": repair.gaussians,
            "phase1": optimization.phase_snapshots["phase1"],
            "contained": optimization.phase_snapshots["contained"],
            "final": final,
        }
        for name, model in boundaries.items():
            evaluation = evaluator.evaluate(inputs, model)
            evaluations[name] = evaluation
            metrics[f"{name}_J_pixel"] = float(evaluation["J_pixel"])
            metrics[f"{name}_J_area"] = float(evaluation["J_area"])
        timings["compact_boundary_evaluation"] = time.perf_counter() - started

    timings["total"] = time.perf_counter() - started_total
    return CarrierPipelineResult(
        beam=beam,
        repair=repair,
        optimization=optimization,
        gaussians=final,
        metrics=metrics,
        evaluations=evaluations,
        timings=timings,
    )


__all__ = [
    "CarrierPipelineConfig",
    "CarrierPipelineResult",
    "run_carrier_pipeline",
]
