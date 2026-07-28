"""Compact-only fixed-topology maturation for Beam Fusion carriers.

The accepted carrier path never consumes RGB, masks, or :class:`SceneData`.  It optimizes
rendered point samples against frozen fitted 2D Gaussian fields, removes centers outside the
fitting-view fitted-Gaussian visual hull, and freezes means afterward so containment cannot be
undone.

The bounded sequence is intentionally small:

1. degree-zero compact optimization with all parameter families trainable;
2. strict projected-center containment in every input view; and
3. degree-zero compact optimization with means frozen.

There is no clone, split, insertion, densification, opacity reset, higher-SH phase, or free birth.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field, replace

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.optim.compact_trainer import CompactTrainConfig, CompactTrainer
from rtgs.render.projection import EWA_NEAR


def _default_compact_template() -> CompactTrainConfig:
    return CompactTrainConfig(
        iterations=380,
        attempts_per_step=256,
        proposal_mode="area_gaussian",
        schedule_mode="balanced_cycle",
        target_mode="proposal_attempt",
        uniform_fraction=0.25,
        device="auto",
        point_chunk=256,
        gaussian_chunk=512,
        outer_microbatch=128,
        query_component_chunk=512,
        teacher_tile_size=16,
        evaluation_chunk=8192,
        checkpoints=(0, 190, 380),
        evaluate_checkpoint_risks=False,
        sh_degree=0,
        sh_color_activation="hard",
        kernel_support_mode="hard",
        visibility_margin_sigma=3.0,
        support_mode="none",
        support_loss_weight=0.0,
    )


@dataclass(frozen=True)
class CarrierOptimizationConfig:
    """Controls for the audited compact-only carrier sequence.

    ``template`` supplies learning rates, sampling cardinality, device, resource caps, and chunk
    sizes.  The sequence fixes the proposal/risk semantics, degree-zero representation, phase-2
    mean freeze, and absence of topology control.
    """

    # Frozen defaults selected by the three 2026-07-28 entries summarized in
    # docs/EXPERIMENTS.md; changing them requires a new preregistered experiment.
    phase1_iterations: int = 380
    phase2_iterations: int = 380
    containment_sigma: float = 3.0
    containment_near: float = EWA_NEAR
    containment_gaussian_chunk: int = 512
    containment_component_chunk: int = 512
    template: CompactTrainConfig = field(default_factory=_default_compact_template)

    def __post_init__(self) -> None:
        for name in ("phase1_iterations", "phase2_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("containment_gaussian_chunk", "containment_component_chunk"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("containment_sigma", "containment_near"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.containment_sigma != 3.0:
            raise ValueError("compact carrier policy fixes containment_sigma=3.0")
        if self.containment_near != EWA_NEAR:
            raise ValueError(f"compact carrier policy fixes containment_near={EWA_NEAR}")
        required = {
            "proposal_mode": "area_gaussian",
            "schedule_mode": "balanced_cycle",
            "target_mode": "proposal_attempt",
            "sh_degree": 0,
            "sh_color_activation": "hard",
            "kernel_support_mode": "hard",
            "support_mode": "none",
            "support_loss_weight": 0.0,
        }
        for name, expected in required.items():
            if getattr(self.template, name) != expected:
                raise ValueError(f"compact carrier template requires {name}={expected!r}")
        for name in (
            "lr_means",
            "lr_quats",
            "lr_scales",
            "lr_opacity",
            "lr_sh",
        ):
            if getattr(self.template, name) <= 0:
                raise ValueError(f"phase-1 compact carrier optimization requires positive {name}")

    @property
    def total_iterations(self) -> int:
        return self.phase1_iterations + self.phase2_iterations


@dataclass(frozen=True)
class CarrierOptimizationResult:
    """Final carrier model plus compact phase and containment receipts."""

    gaussians: Gaussians3D
    phase_histories: dict[str, dict]
    phase_snapshots: dict[str, Gaussians3D]
    diagnostics: dict[str, object]


def _resolved_device(requested: str) -> torch.device:
    return torch.device(
        "cuda"
        if requested == "auto" and torch.cuda.is_available()
        else ("cpu" if requested == "auto" else requested)
    )


def _checkpoints(iterations: int) -> tuple[int, ...]:
    return tuple(sorted({0, iterations // 2, iterations}))


def _phase_config(
    config: CarrierOptimizationConfig,
    *,
    phase: int,
) -> CompactTrainConfig:
    if phase not in {1, 2}:
        raise ValueError("carrier phase must be one or two")
    iterations = config.phase1_iterations if phase == 1 else config.phase2_iterations
    return replace(
        config.template,
        iterations=iterations,
        checkpoints=_checkpoints(iterations),
        # The audited paired-root policy used an isolated +1000 phase-2 RNG domain.
        seed=config.template.seed + (phase - 1) * 1000,
        lr_means=config.template.lr_means if phase == 1 else 0.0,
        lr_sh_rest=0.0,
        sh_degree=0,
        support_mode="none",
        support_loss_weight=0.0,
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _compact_geometry_inputs(
    inputs: ReconstructionInputs,
    device: torch.device,
) -> ReconstructionInputs:
    return ReconstructionInputs(
        observations=[
            field if field.device == device else field.to(device) for field in inputs.observations
        ],
        cameras=[
            camera if camera.R.device == device and camera.t.device == device else camera.to(device)
            for camera in inputs.cameras
        ],
        view_names=list(inputs.view_names),
        bounds_hint=None,
        name=f"{inputs.name}-containment",
    )


def prune_carrier_centers(
    inputs: ReconstructionInputs,
    gaussians: Gaussians3D,
    *,
    sigma: float = 3.0,
    near: float = EWA_NEAR,
    gaussian_chunk: int = 512,
    component_chunk: int = 512,
    device: str = "auto",
) -> tuple[Gaussians3D, dict[str, object]]:
    """Remove centers outside any input-view fitted-Gaussian support union.

    A row is retained only when its projected center has positive depth and nearest-component
    Mahalanobis distance at most ``sigma`` in every input view.  This proves projected-center
    membership in the fitted-view Gaussian visual hull; it does not prove physical surface
    occupancy or constrain the full infinite-support Gaussian footprint.
    """

    inputs.validate()
    if gaussians.n <= 0:
        raise ValueError("carrier containment requires at least one Gaussian")
    for name, value in (("sigma", sigma), ("near", near)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in (
        ("gaussian_chunk", gaussian_chunk),
        ("component_chunk", component_chunk),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    target_device = _resolved_device(device)
    working_inputs = _compact_geometry_inputs(inputs, target_device)
    model = gaussians.to(target_device)
    keep = torch.ones(model.n, dtype=torch.bool, device=target_device)
    violating_view_count = torch.zeros(
        model.n,
        dtype=torch.int64,
        device=target_device,
    )
    per_view: list[dict[str, object]] = []
    threshold = sigma * sigma
    with torch.no_grad():
        for view_name, field, camera in zip(
            working_inputs.view_names,
            working_inputs.observations,
            working_inputs.cameras,
            strict=True,
        ):
            squared_parts: list[torch.Tensor] = []
            depth_parts: list[torch.Tensor] = []
            for start in range(0, model.n, gaussian_chunk):
                end = min(start + gaussian_chunk, model.n)
                xy, depth = camera.project(model.means[start:end])
                squared_parts.append(
                    field.minimum_mahalanobis_squared(
                        xy.to(dtype=field.dtype),
                        component_chunk=component_chunk,
                    )
                )
                depth_parts.append(depth)
            squared = torch.cat(squared_parts)
            depth = torch.cat(depth_parts)
            valid = (squared <= threshold) & (depth > near)
            keep &= valid
            violating_view_count += ~valid
            per_view.append(
                {
                    "view_name": view_name,
                    "outside_rows": int((squared > threshold).sum()),
                    "behind_near_rows": int((depth <= near).sum()),
                    "minimum_q_max": float(squared.max()),
                }
            )
    if not bool(keep.any()):
        raise RuntimeError("strict carrier containment removed every Gaussian")

    output = model.subset(keep).to(gaussians.means.device)
    removed_rows = (~keep).nonzero(as_tuple=True)[0].detach().cpu()
    receipt = {
        "schema": "rtgs.compact-carrier-containment.v1",
        "criterion": (
            "positive depth and nearest positive-amplitude fitted 2D Gaussian "
            f"Mahalanobis q<={threshold:g} in every input view"
        ),
        "n_before": model.n,
        "n_after": output.n,
        "removed": int(removed_rows.numel()),
        "removed_fraction": float((~keep).float().mean()),
        "removed_rows": removed_rows.tolist(),
        "keep_mask_sha256": _tensor_sha256(keep),
        "violating_view_count_sha256": _tensor_sha256(violating_view_count),
        "max_violating_views_before": int(violating_view_count.max()),
        "sigma": sigma,
        "near": near,
        "input_views": inputs.n_views,
        "per_view_before": per_view,
        "projected_center_containment_after": True,
        "surface_occupancy_proven": False,
        "interior_visual_hull_floaters_detected": False,
    }
    return output, receipt


def _phase_resources(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def optimize_carriers(
    inputs: ReconstructionInputs,
    initialization: Gaussians3D,
    config: CarrierOptimizationConfig | None = None,
) -> CarrierOptimizationResult:
    """Run the complete image-free, fixed-topology carrier sequence."""

    config = config or CarrierOptimizationConfig()
    inputs.validate()
    if initialization.n <= 0:
        raise ValueError("carrier optimization requires at least one Gaussian")
    if any(field.sigma_cutoff != config.containment_sigma for field in inputs.observations):
        raise ValueError("compact carrier policy requires three-sigma fitted 2D Gaussian fields")
    device = _resolved_device(config.template.device)
    current = initialization.with_sh_degree(0)
    phase_snapshots: dict[str, Gaussians3D] = {"initial": current.detach().to("cpu")}
    phase_histories: dict[str, dict] = {}
    phase_seconds: dict[str, float] = {}
    phase_resources: dict[str, dict[str, int] | None] = {}

    phase1_config = _phase_config(config, phase=1)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    current, phase1_history = CompactTrainer(phase1_config).train(inputs, current)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    phase_seconds["phase1"] = time.perf_counter() - started
    phase_resources["phase1"] = _phase_resources(device)
    phase_histories["phase1"] = phase1_history
    phase_snapshots["phase1"] = current.detach().to("cpu")
    if phase1_history["n_init_3d"] != phase1_history["n_opt_3d"]:
        raise RuntimeError("compact carrier phase 1 changed topology")

    started = time.perf_counter()
    contained, containment = prune_carrier_centers(
        inputs,
        current,
        sigma=config.containment_sigma,
        near=config.containment_near,
        gaussian_chunk=config.containment_gaussian_chunk,
        component_chunk=config.containment_component_chunk,
        device=config.template.device,
    )
    phase_seconds["containment"] = time.perf_counter() - started
    phase_snapshots["contained"] = contained.detach().to("cpu")

    phase2_config = _phase_config(config, phase=2)
    if phase2_config.lr_means != 0.0:
        raise RuntimeError("carrier phase 2 must freeze means")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    final, phase2_history = CompactTrainer(phase2_config).train(inputs, contained)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    phase_seconds["phase2"] = time.perf_counter() - started
    phase_resources["phase2"] = _phase_resources(device)
    phase_histories["phase2"] = phase2_history
    phase_snapshots["phase2"] = final.detach().to("cpu")
    if phase2_history["n_init_3d"] != phase2_history["n_opt_3d"]:
        raise RuntimeError("compact carrier phase 2 changed topology")
    if phase2_history["optimizer_group_motion"]["means"]["max_abs"] != 0.0:
        raise RuntimeError("carrier phase 2 moved frozen means")

    rechecked, containment_recheck = prune_carrier_centers(
        inputs,
        final,
        sigma=config.containment_sigma,
        near=config.containment_near,
        gaussian_chunk=config.containment_gaussian_chunk,
        component_chunk=config.containment_component_chunk,
        device=config.template.device,
    )
    if containment_recheck["removed"] != 0 or rechecked.n != final.n:
        raise RuntimeError("carrier phase 2 violated projected-center containment")

    return CarrierOptimizationResult(
        gaussians=final.detach(),
        phase_histories=phase_histories,
        phase_snapshots=phase_snapshots,
        diagnostics={
            "schema": "rtgs.compact-carrier-optimization.v2",
            "supervision": "fitted_2d_gaussian_fields_only",
            "source_rgb_used": False,
            "source_masks_used": False,
            "packed_alpha_used": False,
            "fixed_topology_within_each_phase": True,
            "topology_growth": False,
            "initial_carriers": initialization.n,
            "contained_carriers": contained.n,
            "final_carriers": final.n,
            "total_iterations": config.total_iterations,
            "phase_seconds": phase_seconds,
            "total_seconds": sum(phase_seconds.values()),
            "phase_resources": phase_resources,
            "containment": containment,
            "containment_recheck": containment_recheck,
            "phase2_means_frozen_bit_exact": True,
            "projected_center_containment": True,
            "surface_occupancy_proven": False,
        },
    )


__all__ = [
    "CarrierOptimizationConfig",
    "CarrierOptimizationResult",
    "optimize_carriers",
    "prune_carrier_centers",
]
