"""ADR-002 phased maturation schedule for Beam Fusion carrier Gaussians.

The schedule deliberately separates representation repair from topology growth:

1. fixed-topology SH0 warmup;
2. protected-parent clone-only growth (or the explicit low-opacity particle variant);
3. higher-order appearance with geometry and opacity frozen;
4. ordinary 3DGS optimization after the carriers have stabilized.

Classic density surgery emits an opt-in immutable record through :class:`Trainer`; the lineage
tracker below uses it to measure original-carrier survival, descendant counts, displacement,
covariance drift, and opacity drift at every phase boundary.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.scene import SceneData
from rtgs.optim.trainer import TrainConfig, Trainer


@dataclass(frozen=True)
class CarrierOptimizationConfig:
    """Controls for the post-repair carrier-maturation schedule."""

    warmup_iterations: int = 60
    clone_iterations: int = 80
    higher_sh_iterations: int = 40
    standard_iterations: int = 120
    clone_every: int = 20
    clone_grad_threshold: float = 2e-4
    clone_growth_factor: float = 2.0
    clone_jitter_fraction: float = 0.10
    clone_all: bool = False
    clone_tangent_only: bool = False
    particle_generation: bool = False
    particle_child_opacity_scale: float = 0.20
    particle_prune_opacity: float = 0.02
    standard_every: int = 40
    standard_growth_factor: float = 4.0
    target_sh_degree: int = 3
    template: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        for name in (
            "warmup_iterations",
            "clone_iterations",
            "higher_sh_iterations",
            "standard_iterations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("clone_every", "standard_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "clone_growth_factor",
            "standard_growth_factor",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 1:
                raise ValueError(f"{name} must be finite and at least one")
        if not math.isfinite(self.clone_grad_threshold) or self.clone_grad_threshold < 0:
            raise ValueError("clone_grad_threshold must be finite and non-negative")
        if not math.isfinite(self.clone_jitter_fraction) or self.clone_jitter_fraction < 0:
            raise ValueError("clone_jitter_fraction must be finite and non-negative")
        if not isinstance(self.clone_all, bool):
            raise TypeError("clone_all must be bool")
        if not isinstance(self.clone_tangent_only, bool):
            raise TypeError("clone_tangent_only must be bool")
        if (
            not math.isfinite(self.particle_child_opacity_scale)
            or not 0 < self.particle_child_opacity_scale <= 1
        ):
            raise ValueError("particle_child_opacity_scale must be in (0,1]")
        if (
            not math.isfinite(self.particle_prune_opacity)
            or not 0 <= self.particle_prune_opacity < 1
        ):
            raise ValueError("particle_prune_opacity must be in [0,1)")
        if not 0 <= self.target_sh_degree <= 3:
            raise ValueError("target_sh_degree must be between zero and three")
        if not any(
            (
                self.warmup_iterations,
                self.clone_iterations,
                self.higher_sh_iterations,
                self.standard_iterations,
            )
        ):
            raise ValueError("the carrier schedule must contain at least one optimization step")

    @property
    def total_iterations(self) -> int:
        return (
            self.warmup_iterations
            + self.clone_iterations
            + self.higher_sh_iterations
            + self.standard_iterations
        )


@dataclass(frozen=True)
class CarrierOptimizationResult:
    """Final model and complete phase/lineage diagnostics."""

    gaussians: Gaussians3D
    phase_histories: dict[str, dict]
    phase_snapshots: dict[str, Gaussians3D]
    diagnostics: dict[str, object]


def _quantiles(values: torch.Tensor) -> dict[str, float] | None:
    values = values.detach().to(torch.float64).reshape(-1)
    if values.numel() == 0:
        return None
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p90": float(values.quantile(0.90)),
        "max": float(values.max()),
    }


class CarrierLineageTracker:
    """Track original carriers and descendants across classic density surgery."""

    def __init__(self, initial: Gaussians3D):
        self.initial_n = initial.n
        self.initial_means = initial.means.detach().cpu().to(torch.float64)
        self.initial_covariances = initial.covariance().detach().cpu().to(torch.float64)
        self.initial_opacity = initial.opacity.detach().cpu().to(torch.float64)
        self.root_ids = torch.arange(initial.n, dtype=torch.int64)
        self.generations = torch.zeros(initial.n, dtype=torch.int64)
        self.surgeries: list[dict[str, int]] = []

    def apply_surgery(
        self,
        keep_mask: torch.Tensor,
        parent_rows: torch.Tensor,
        iteration: int,
    ) -> None:
        """Replay one survivors-then-newborns row edit."""

        keep = keep_mask.detach().cpu().to(torch.bool)
        parents = parent_rows.detach().cpu().to(torch.int64)
        if keep.shape != self.root_ids.shape:
            raise RuntimeError("lineage keep mask does not match the current carrier rows")
        if parents.numel() and (
            int(parents.min()) < 0 or int(parents.max()) >= self.root_ids.numel()
        ):
            raise RuntimeError("lineage parent row is out of bounds")
        old_roots = self.root_ids
        old_generations = self.generations
        self.root_ids = torch.cat([old_roots[keep], old_roots[parents]])
        self.generations = torch.cat([old_generations[keep], old_generations[parents] + 1])
        self.surgeries.append(
            {
                "iteration": int(iteration),
                "n_before": int(keep.numel()),
                "n_after": int(self.root_ids.numel()),
                "removed": int((~keep).sum()),
                "newborns": int(parents.numel()),
            }
        )

    def snapshot(self, name: str, gaussians: Gaussians3D, iteration: int) -> dict[str, object]:
        """Summarize mechanism state at one phase boundary."""

        if gaussians.n != self.root_ids.numel():
            raise RuntimeError("lineage state and Gaussian row count disagree")
        means = gaussians.means.detach().cpu().to(torch.float64)
        covariances = gaussians.covariance().detach().cpu().to(torch.float64)
        opacity = gaussians.opacity.detach().cpu().to(torch.float64)
        roots = self.root_ids
        generation_zero = self.generations == 0
        surviving_root = roots[generation_zero]
        original_displacement = (
            (means[generation_zero] - self.initial_means[surviving_root]).norm(dim=-1)
            if bool(generation_zero.any())
            else means.new_empty(0)
        )
        all_displacement = (means - self.initial_means[roots]).norm(dim=-1)
        original_logdet_drift = (
            (
                torch.linalg.slogdet(covariances[generation_zero]).logabsdet
                - torch.linalg.slogdet(self.initial_covariances[surviving_root]).logabsdet
            ).abs()
            if bool(generation_zero.any())
            else means.new_empty(0)
        )
        original_opacity_drift = (
            (opacity[generation_zero] - self.initial_opacity[surviving_root]).abs()
            if bool(generation_zero.any())
            else opacity.new_empty(0)
        )
        descendant_counts = torch.bincount(roots, minlength=self.initial_n)
        return {
            "phase": name,
            "iteration": int(iteration),
            "n_gaussians": gaussians.n,
            "surviving_original_carriers": int(generation_zero.sum()),
            "carrier_survival_rate": float(generation_zero.sum() / self.initial_n),
            "roots_with_any_descendant": int((descendant_counts > 0).sum()),
            "roots_with_any_descendant_rate": float((descendant_counts > 0).sum() / self.initial_n),
            "newborn_rows": int((self.generations > 0).sum()),
            "max_generation": int(self.generations.max()) if self.generations.numel() else 0,
            "descendants_per_root": _quantiles(descendant_counts),
            "original_mean_displacement": _quantiles(original_displacement),
            "all_rows_displacement_from_root": _quantiles(all_displacement),
            "original_covariance_abs_logdet_drift": _quantiles(original_logdet_drift),
            "original_opacity_abs_drift": _quantiles(original_opacity_drift),
            "opacity": _quantiles(opacity),
        }


def _density_maximum(initial_n: int, factor: float) -> int:
    return max(initial_n, int(math.ceil(initial_n * factor)))


def _phase_config(
    schedule: CarrierOptimizationConfig,
    *,
    name: str,
    iterations: int,
    offset: int,
    initial_carriers: int,
) -> TrainConfig:
    template = schedule.template
    common = {
        "iterations": iterations,
        "iteration_offset": offset,
        "schedule_iterations": schedule.total_iterations,
        "device": template.device,
        "rasterizer": template.rasterizer,
        "eval_every": min(template.eval_every, iterations),
        "checkpoint_policy": "final",
        "seed": template.seed + offset,
    }
    if name == "warmup":
        return replace(
            template,
            **common,
            densify=False,
            target_sh_degree=0,
        )
    if name == "clone":
        particle = schedule.particle_generation
        density = replace(
            template.density,
            start_iter=offset + 1,
            stop_iter=offset + iterations,
            every=schedule.clone_every,
            grad_threshold=schedule.clone_grad_threshold,
            max_gaussians=_density_maximum(
                initial_carriers,
                schedule.clone_growth_factor,
            ),
            opacity_reset_every=0,
            clone_only=True,
            clone_all=schedule.clone_all,
            clone_jitter_fraction=(
                max(schedule.clone_jitter_fraction, 0.25)
                if particle
                else schedule.clone_jitter_fraction
            ),
            clone_tangent_only=schedule.clone_tangent_only,
            clone_child_opacity_scale=(schedule.particle_child_opacity_scale if particle else 1.0),
            prune_opacity=(
                schedule.particle_prune_opacity if particle else template.density.prune_opacity
            ),
            protect_first_n=initial_carriers,
        )
        return replace(
            template,
            **common,
            densify=True,
            density_strategy="classic",
            density=density,
            target_sh_degree=0,
        )
    if name == "higher-sh":
        return replace(
            template,
            **common,
            densify=False,
            lr_means=0.0,
            lr_quats=0.0,
            lr_scales=0.0,
            lr_opacity=0.0,
            target_sh_degree=schedule.target_sh_degree,
            sh_degree_interval=1,
        )
    if name == "standard":
        density = replace(
            template.density,
            start_iter=offset + 1,
            stop_iter=offset + iterations,
            every=schedule.standard_every,
            max_gaussians=_density_maximum(
                initial_carriers,
                schedule.standard_growth_factor,
            ),
            clone_only=False,
            clone_all=False,
            clone_jitter_fraction=0.0,
            clone_tangent_only=False,
            clone_child_opacity_scale=1.0,
            protect_first_n=0,
        )
        return replace(
            template,
            **common,
            densify=True,
            density_strategy="classic",
            density=density,
            target_sh_degree=schedule.target_sh_degree,
            sh_degree_interval=1,
        )
    raise ValueError(f"unknown carrier phase {name!r}")


def optimize_carriers(
    scene: SceneData,
    initialization: Gaussians3D,
    config: CarrierOptimizationConfig | None = None,
) -> CarrierOptimizationResult:
    """Run the complete ADR-002 phased carrier-maturation schedule."""

    config = config or CarrierOptimizationConfig()
    scene.validate()
    current = initialization.with_sh_degree(0)
    tracker = CarrierLineageTracker(current)
    phase_histories: dict[str, dict] = {}
    phase_snapshots: dict[str, Gaussians3D] = {"repaired": current.detach()}
    boundaries: list[dict[str, object]] = [tracker.snapshot("repaired", current, 0)]
    phase_seconds: dict[str, float] = {}
    offset = 0
    phase_specs = (
        ("warmup", config.warmup_iterations),
        ("clone", config.clone_iterations),
        ("higher-sh", config.higher_sh_iterations),
        ("standard", config.standard_iterations),
    )
    for name, iterations in phase_specs:
        if iterations == 0:
            continue
        phase_config = _phase_config(
            config,
            name=name,
            iterations=iterations,
            offset=offset,
            initial_carriers=initialization.n,
        )
        started = time.perf_counter()
        current, history = Trainer(phase_config).train(
            scene,
            current,
            density_surgery_callback=tracker.apply_surgery,
        )
        phase_seconds[name] = time.perf_counter() - started
        offset += iterations
        phase_histories[name] = history
        phase_snapshots[name] = current.detach()
        boundaries.append(tracker.snapshot(name, current, offset))

    peak_vram = max(
        (float(history.get("peak_vram_gb", 0.0)) for history in phase_histories.values()),
        default=0.0,
    )
    peak_reserved = max(
        (float(history.get("peak_vram_reserved_gb", 0.0)) for history in phase_histories.values()),
        default=0.0,
    )
    return CarrierOptimizationResult(
        gaussians=current.detach(),
        phase_histories=phase_histories,
        phase_snapshots=phase_snapshots,
        diagnostics={
            "schema": "rtgs.carrier-optimization.v1",
            "initial_carriers": initialization.n,
            "total_iterations": offset,
            "particle_generation": config.particle_generation,
            "phase_seconds": phase_seconds,
            "total_seconds": sum(phase_seconds.values()),
            "peak_vram_gb": peak_vram,
            "peak_vram_reserved_gb": peak_reserved,
            "phase_boundaries": boundaries,
            "density_surgeries": tracker.surgeries,
            "final_root_ids": tracker.root_ids.tolist(),
            "final_generations": tracker.generations.tolist(),
        },
    )


__all__ = [
    "CarrierLineageTracker",
    "CarrierOptimizationConfig",
    "CarrierOptimizationResult",
    "optimize_carriers",
]
