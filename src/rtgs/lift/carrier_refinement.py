"""ADR-002 carrier repair on Beam Fusion's fixed contributor tracks.

Beam Fusion provides useful 3D means and an explicit CSR mapping back to the fitted 2D Gaussian
components.  This module matures those carriers *before* dense-image 3DGS optimization:

* covariance repair fits a Cholesky-parameterized SPD covariance to associated 2D footprints;
* opacity repair fits optical density to associated 2D component amplitudes;
* appearance repair computes a robust amplitude-weighted SH0 color and removes higher bands.

No phase discovers correspondences or changes means, contributor assignments, or topology.  The
2D fitted amplitude is an experimental proxy for per-carrier opacity, not an identifiable physical
alpha measurement; that limitation is surfaced in the returned diagnostics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.sh import rgb_to_sh, sh_to_rgb
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.beam_fusion import (
    BeamFusionResult,
    _component_covariances_2d,
)
from rtgs.lift.compact_carve import _center_and_extent


@dataclass(frozen=True)
class CarrierRepairConfig:
    """Controls for the fixed-track carrier repair phases."""

    covariance_steps: int = 120
    covariance_learning_rate: float = 0.03
    covariance_huber_delta: float = 0.25
    covariance_prior_weight: float = 1e-3
    min_sigma_world: float = 1e-4
    max_sigma_extent_fraction: float = 0.5
    max_aspect_ratio: float = 100.0
    opacity_steps: int = 80
    opacity_learning_rate: float = 0.05
    opacity_huber_delta: float = 0.20
    opacity_prior_weight: float = 1e-3
    min_opacity: float = 0.005
    max_opacity: float = 0.995
    appearance_irls_steps: int = 5
    appearance_huber_delta: float = 0.10
    repair_covariance: bool = True
    repair_opacity: bool = True
    repair_appearance: bool = True

    def __post_init__(self) -> None:
        for name in ("covariance_steps", "opacity_steps", "appearance_irls_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "covariance_learning_rate",
            "covariance_huber_delta",
            "min_sigma_world",
            "max_sigma_extent_fraction",
            "max_aspect_ratio",
            "opacity_learning_rate",
            "opacity_huber_delta",
            "appearance_huber_delta",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("covariance_prior_weight", "opacity_prior_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_aspect_ratio < 1:
            raise ValueError("max_aspect_ratio must be at least one")
        if not 0 < self.min_opacity < self.max_opacity < 1:
            raise ValueError("opacity bounds must satisfy 0 < min < max < 1")


@dataclass(frozen=True)
class CarrierObservationTable:
    """One row per Beam Fusion contributor link."""

    carrier_indices: torch.Tensor
    view_indices: torch.Tensor
    component_indices: torch.Tensor
    jacobians: torch.Tensor
    covariances2d: torch.Tensor
    amplitudes: torch.Tensor
    colors: torch.Tensor

    @property
    def n_links(self) -> int:
        return int(self.carrier_indices.numel())


@dataclass(frozen=True)
class CarrierRepairResult:
    """Carrier snapshots after each independent repair phase."""

    beam: Gaussians3D
    covariance: Gaussians3D
    opacity: Gaussians3D
    appearance: Gaussians3D
    gaussians: Gaussians3D
    diagnostics: dict[str, object]


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().to(torch.float64).reshape(-1)
    if values.numel() == 0:
        raise ValueError("cannot summarize an empty tensor")
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p90": float(values.quantile(0.90)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def build_carrier_observation_table(
    result: BeamFusionResult,
    inputs: ReconstructionInputs,
) -> CarrierObservationTable:
    """Expand Beam Fusion CSR lineage into a differentiable fixed-track observation table."""

    if result.n_components != result.gaussians.n:
        raise ValueError("Beam Fusion component and Gaussian counts disagree")
    if result.contributor_view_indices.device.type != "cpu":
        raise ValueError("carrier repair is a CPU-first reference path")
    counts = result.component_offsets[1:] - result.component_offsets[:-1]
    carrier_indices = torch.repeat_interleave(
        torch.arange(result.n_components, dtype=torch.int64), counts
    )
    view_indices = result.contributor_view_indices.to(torch.int64)
    component_indices = result.contributor_component_indices.to(torch.int64)
    if carrier_indices.numel() != view_indices.numel():
        raise ValueError("Beam Fusion CSR offsets do not cover every contributor")
    if bool((counts < 2).any()):
        raise ValueError("each carrier must retain at least two contributor views")

    means = result.gaussians.means.to(torch.float64)
    jacobians = torch.empty(carrier_indices.numel(), 2, 3, dtype=torch.float64)
    covariances = torch.empty(carrier_indices.numel(), 2, 2, dtype=torch.float64)
    amplitudes = torch.empty(carrier_indices.numel(), dtype=torch.float64)
    colors = torch.empty(carrier_indices.numel(), 3, dtype=torch.float64)
    for view_index, (camera, observation) in enumerate(
        zip(inputs.cameras, inputs.observations, strict=True)
    ):
        rows = (view_indices == view_index).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        carriers = carrier_indices[rows]
        components = component_indices[rows]
        points_cam = camera.world_to_cam(means[carriers])
        depth = points_cam[:, 2]
        if bool((depth <= 0).any()):
            raise ValueError("a carrier contributor projects behind its associated camera")
        j_cam = torch.zeros(rows.numel(), 2, 3, dtype=torch.float64)
        j_cam[:, 0, 0] = camera.fx / depth
        j_cam[:, 0, 2] = -camera.fx * points_cam[:, 0] / depth.square()
        j_cam[:, 1, 1] = camera.fy / depth
        j_cam[:, 1, 2] = -camera.fy * points_cam[:, 1] / depth.square()
        jacobians[rows] = j_cam @ camera.R.to(torch.float64)
        covariances[rows] = _component_covariances_2d(observation)[components]
        amplitudes[rows] = observation.amplitudes[components].to(torch.float64)
        colors[rows] = observation.colors[components].to(torch.float64)
    if not bool(torch.isfinite(jacobians).all()):
        raise ValueError("carrier projection Jacobians are not finite")
    return CarrierObservationTable(
        carrier_indices=carrier_indices,
        view_indices=view_indices,
        component_indices=component_indices,
        jacobians=jacobians,
        covariances2d=covariances,
        amplitudes=amplitudes,
        colors=colors,
    )


def _pack_cholesky(covariances: torch.Tensor) -> torch.Tensor:
    lower = torch.linalg.cholesky(covariances.to(torch.float64))
    return torch.stack(
        [
            lower[:, 0, 0].log(),
            lower[:, 1, 0],
            lower[:, 1, 1].log(),
            lower[:, 2, 0],
            lower[:, 2, 1],
            lower[:, 2, 2].log(),
        ],
        dim=-1,
    )


def _unpack_cholesky(parameters: torch.Tensor) -> torch.Tensor:
    count = parameters.shape[0]
    lower = torch.zeros(count, 3, 3, dtype=parameters.dtype, device=parameters.device)
    lower[:, 0, 0] = parameters[:, 0].clamp(-30, 30).exp()
    lower[:, 1, 0] = parameters[:, 1]
    lower[:, 1, 1] = parameters[:, 2].clamp(-30, 30).exp()
    lower[:, 2, 0] = parameters[:, 3]
    lower[:, 2, 1] = parameters[:, 4]
    lower[:, 2, 2] = parameters[:, 5].clamp(-30, 30).exp()
    return lower @ lower.transpose(-1, -2)


def _project_links(
    table: CarrierObservationTable,
    covariances: torch.Tensor,
) -> torch.Tensor:
    selected = covariances[table.carrier_indices]
    return table.jacobians @ selected @ table.jacobians.transpose(-1, -2)


def covariance_reprojection_residuals(
    table: CarrierObservationTable,
    covariances: torch.Tensor,
) -> torch.Tensor:
    """Observation-whitened RMS covariance residual for every contributor link."""

    observed = table.covariances2d
    eigenvalues, eigenvectors = torch.linalg.eigh(observed)
    inverse_sqrt = (
        eigenvectors
        @ torch.diag_embed(eigenvalues.clamp_min(1e-12).rsqrt())
        @ eigenvectors.transpose(-1, -2)
    )
    normalized = inverse_sqrt @ _project_links(table, covariances) @ inverse_sqrt
    identity = torch.eye(2, dtype=normalized.dtype, device=normalized.device)
    return (normalized - identity).square().mean(dim=(-2, -1)).sqrt()


def _bounded_spd(
    covariances: torch.Tensor,
    *,
    min_sigma: float,
    max_sigma: float,
    max_aspect_ratio: float,
) -> torch.Tensor:
    symmetric = 0.5 * (covariances + covariances.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    eigenvalues = eigenvalues.clamp(min=min_sigma**2, max=max_sigma**2)
    eigenvalues[:, -1] = torch.minimum(
        eigenvalues[:, -1],
        eigenvalues[:, 0] * max_aspect_ratio**2,
    )
    bounded = (eigenvectors * eigenvalues[:, None, :]) @ eigenvectors.transpose(-1, -2)
    return 0.5 * (bounded + bounded.transpose(-1, -2))


def _replace_covariances(base: Gaussians3D, covariances: torch.Tensor) -> Gaussians3D:
    converted = Gaussians3D.from_means_covs(
        base.means,
        covariances.to(base.means),
        colors=torch.zeros(base.n, 3, dtype=base.means.dtype, device=base.means.device),
        opacity=base.opacity,
        sh_degree=base.sh_degree,
    )
    return Gaussians3D(
        means=base.means.detach().clone(),
        quats=converted.quats,
        log_scales=converted.log_scales,
        opacity=base.opacity.detach().clone(),
        sh=base.sh.detach().clone(),
    )


def refine_carrier_covariances(
    base: Gaussians3D,
    table: CarrierObservationTable,
    *,
    extent: float,
    config: CarrierRepairConfig,
) -> tuple[Gaussians3D, dict[str, object]]:
    """Fit bounded SPD covariances while means and every non-covariance field stay frozen."""

    max_sigma = config.max_sigma_extent_fraction * extent
    if max_sigma <= config.min_sigma_world:
        raise ValueError("resolved maximum sigma must exceed minimum sigma")
    initial = base.covariance().to(torch.float64)
    parameters = torch.nn.Parameter(_pack_cholesky(initial).clone())
    optimizer = torch.optim.Adam([parameters], lr=config.covariance_learning_rate)
    prior_norm = initial.norm(dim=(-2, -1)).clamp_min(1e-12)
    history: list[dict[str, float]] = []
    log_aspect_limit = math.log(config.max_aspect_ratio**2)
    log_min_variance = math.log(config.min_sigma_world**2)
    log_max_variance = math.log(max_sigma**2)

    before = covariance_reprojection_residuals(table, initial)
    for step in range(config.covariance_steps):
        optimizer.zero_grad(set_to_none=True)
        covariances = _unpack_cholesky(parameters)
        residual = covariance_reprojection_residuals(table, covariances)
        data_loss = F.huber_loss(
            residual,
            torch.zeros_like(residual),
            delta=config.covariance_huber_delta,
        )
        prior_loss = (
            ((covariances - initial) / prior_norm[:, None, None]).square().mean(dim=(-2, -1))
        ).mean()
        eigenvalues = torch.linalg.eigvalsh(covariances).clamp_min(1e-30)
        log_eigenvalues = eigenvalues.log()
        bound_loss = (
            torch.relu(log_min_variance - log_eigenvalues).square().mean()
            + torch.relu(log_eigenvalues - log_max_variance).square().mean()
            + torch.relu(log_eigenvalues[:, -1] - log_eigenvalues[:, 0] - log_aspect_limit)
            .square()
            .mean()
        )
        loss = data_loss + config.covariance_prior_weight * prior_loss + 1e-3 * bound_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([parameters], 10.0)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == config.covariance_steps:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "data_loss": float(data_loss.detach()),
                    "prior_loss": float(prior_loss.detach()),
                    "bound_loss": float(bound_loss.detach()),
                    "residual_median": float(residual.detach().median()),
                }
            )

    fitted = _bounded_spd(
        _unpack_cholesky(parameters.detach()),
        min_sigma=config.min_sigma_world,
        max_sigma=max_sigma,
        max_aspect_ratio=config.max_aspect_ratio,
    )
    after = covariance_reprojection_residuals(table, fitted)
    eigenvalues = torch.linalg.eigvalsh(fitted)
    output = _replace_covariances(base, fitted)
    if not (
        torch.equal(output.means, base.means)
        and torch.equal(output.opacity, base.opacity)
        and torch.equal(output.sh, base.sh)
    ):
        raise RuntimeError("covariance repair changed a frozen carrier field")
    return output, {
        "steps": config.covariance_steps,
        "learning_rate": config.covariance_learning_rate,
        "huber_delta": config.covariance_huber_delta,
        "prior_weight": config.covariance_prior_weight,
        "min_sigma_world": config.min_sigma_world,
        "max_sigma_world": max_sigma,
        "max_aspect_ratio": config.max_aspect_ratio,
        "whitened_residual_before": _quantiles(before),
        "whitened_residual_after": _quantiles(after),
        "sigma": _quantiles(eigenvalues.sqrt()),
        "aspect_ratio": _quantiles((eigenvalues[:, -1] / eigenvalues[:, 0]).sqrt()),
        "history": history,
        "means_frozen_bit_exact": True,
        "opacity_frozen_bit_exact": True,
        "appearance_frozen_bit_exact": True,
    }


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return value + torch.log(-torch.expm1(-value))


def opacity_optical_density_residuals(
    gaussians: Gaussians3D,
    table: CarrierObservationTable,
) -> torch.Tensor:
    """Absolute link residual in optical-density space."""

    predicted = -torch.log1p(-gaussians.opacity.to(torch.float64).clamp(1e-8, 1.0 - 1e-8))
    target = -torch.log1p(-table.amplitudes.clamp(1e-8, 1.0 - 1e-8))
    return (predicted[table.carrier_indices] - target).abs()


def refine_carrier_opacities(
    base: Gaussians3D,
    table: CarrierObservationTable,
    *,
    config: CarrierRepairConfig,
) -> tuple[Gaussians3D, dict[str, object]]:
    """Fit one optical density per carrier to its fixed associated 2D amplitudes."""

    initial_alpha = base.opacity.to(torch.float64).clamp(config.min_opacity, config.max_opacity)
    initial_tau = -torch.log1p(-initial_alpha)
    target_tau = -torch.log1p(-table.amplitudes.clamp(config.min_opacity, config.max_opacity))
    raw_tau = torch.nn.Parameter(_inverse_softplus(initial_tau).clone())
    optimizer = torch.optim.Adam([raw_tau], lr=config.opacity_learning_rate)
    history: list[dict[str, float]] = []
    before = opacity_optical_density_residuals(base, table)

    for step in range(config.opacity_steps):
        optimizer.zero_grad(set_to_none=True)
        tau = F.softplus(raw_tau)
        link_residual = tau[table.carrier_indices] - target_tau
        data_loss = F.huber_loss(
            link_residual,
            torch.zeros_like(link_residual),
            delta=config.opacity_huber_delta,
        )
        prior_loss = (((tau - initial_tau) / (1.0 + initial_tau)) ** 2).mean()
        loss = data_loss + config.opacity_prior_weight * prior_loss
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == config.opacity_steps:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "data_loss": float(data_loss.detach()),
                    "prior_loss": float(prior_loss.detach()),
                    "residual_median": float(link_residual.detach().abs().median()),
                }
            )

    opacity = (1.0 - torch.exp(-F.softplus(raw_tau.detach()))).clamp(
        config.min_opacity, config.max_opacity
    )
    output = Gaussians3D(
        means=base.means.detach().clone(),
        quats=base.quats.detach().clone(),
        log_scales=base.log_scales.detach().clone(),
        opacity=opacity.to(base.opacity),
        sh=base.sh.detach().clone(),
    )
    after = opacity_optical_density_residuals(output, table)
    if not (
        torch.equal(output.means, base.means)
        and torch.equal(output.quats, base.quats)
        and torch.equal(output.log_scales, base.log_scales)
        and torch.equal(output.sh, base.sh)
    ):
        raise RuntimeError("opacity repair changed a frozen carrier field")
    return output, {
        "steps": config.opacity_steps,
        "learning_rate": config.opacity_learning_rate,
        "huber_delta": config.opacity_huber_delta,
        "prior_weight": config.opacity_prior_weight,
        "target": "associated fitted-2d-component amplitude",
        "target_identifiability": (
            "experimental proxy only; fitted component amplitude is not identifiable physical "
            "per-carrier alpha"
        ),
        "optical_density_residual_before": _quantiles(before),
        "optical_density_residual_after": _quantiles(after),
        "opacity_before": _quantiles(base.opacity),
        "opacity_after": _quantiles(output.opacity),
        "history": history,
        "means_frozen_bit_exact": True,
        "covariance_frozen_bit_exact": True,
        "appearance_frozen_bit_exact": True,
    }


def appearance_link_residuals(
    gaussians: Gaussians3D,
    table: CarrierObservationTable,
) -> torch.Tensor:
    """RGB Euclidean residual for each fixed contributor link."""

    color = sh_to_rgb(gaussians.sh[:, 0]).to(torch.float64).clamp(0.0, 1.0)
    return (color[table.carrier_indices] - table.colors).norm(dim=-1)


def refine_carrier_appearance(
    base: Gaussians3D,
    table: CarrierObservationTable,
    *,
    config: CarrierRepairConfig,
) -> tuple[Gaussians3D, dict[str, object]]:
    """Robust amplitude-weighted IRLS fit of SH0 with all higher bands disabled."""

    count = base.n
    color = sh_to_rgb(base.sh[:, 0]).to(torch.float64).clamp(0.0, 1.0)
    base_weight = table.amplitudes.clamp_min(0.05)
    before = appearance_link_residuals(base, table)
    history: list[dict[str, float]] = []
    for step in range(config.appearance_irls_steps):
        residual = (color[table.carrier_indices] - table.colors).norm(dim=-1).clamp_min(1e-12)
        robust = (config.appearance_huber_delta / residual).clamp(max=1.0)
        weights = base_weight * robust
        weighted_sum = torch.zeros(count, 3, dtype=torch.float64)
        weight_sum = torch.zeros(count, dtype=torch.float64)
        weighted_sum.index_add_(
            0,
            table.carrier_indices,
            weights[:, None] * table.colors,
        )
        weight_sum.index_add_(0, table.carrier_indices, weights)
        color = weighted_sum / weight_sum.clamp_min(1e-12)[:, None]
        history.append(
            {
                "step": step + 1,
                "link_residual_mean": float(residual.mean()),
                "link_residual_median": float(residual.median()),
            }
        )
    sh = rgb_to_sh(color.clamp(0.0, 1.0).to(base.means))[:, None]
    output = Gaussians3D(
        means=base.means.detach().clone(),
        quats=base.quats.detach().clone(),
        log_scales=base.log_scales.detach().clone(),
        opacity=base.opacity.detach().clone(),
        sh=sh,
    )
    after = appearance_link_residuals(output, table)
    if not (
        torch.equal(output.means, base.means)
        and torch.equal(output.quats, base.quats)
        and torch.equal(output.log_scales, base.log_scales)
        and torch.equal(output.opacity, base.opacity)
    ):
        raise RuntimeError("appearance repair changed a frozen carrier field")
    return output, {
        "irls_steps": config.appearance_irls_steps,
        "huber_delta": config.appearance_huber_delta,
        "rgb_link_residual_before": _quantiles(before),
        "rgb_link_residual_after": _quantiles(after),
        "higher_sh_bands": "disabled",
        "means_frozen_bit_exact": True,
        "covariance_frozen_bit_exact": True,
        "opacity_frozen_bit_exact": True,
    }


def repair_beam_carriers(
    result: BeamFusionResult,
    inputs: ReconstructionInputs,
    config: CarrierRepairConfig | None = None,
) -> CarrierRepairResult:
    """Run ADR-002 covariance, opacity, and SH0 repairs in order."""

    config = config or CarrierRepairConfig()
    table = build_carrier_observation_table(result, inputs)
    beam = result.gaussians.detach()
    current = beam
    diagnostics: dict[str, object] = {
        "schema": "rtgs.carrier-repair.v1",
        "n_carriers": beam.n,
        "n_contributor_links": table.n_links,
        "mean_contributors_per_carrier": table.n_links / beam.n,
        "phase_seconds": {},
        "enabled": {
            "covariance": config.repair_covariance,
            "opacity": config.repair_opacity,
            "appearance": config.repair_appearance,
        },
    }

    started = time.perf_counter()
    if config.repair_covariance:
        current, covariance_diagnostics = refine_carrier_covariances(
            current,
            table,
            extent=float(_center_and_extent(inputs, torch.float64)[1]),
            config=config,
        )
        diagnostics["covariance"] = covariance_diagnostics
    diagnostics["phase_seconds"]["covariance"] = time.perf_counter() - started
    covariance = current.detach()

    started = time.perf_counter()
    if config.repair_opacity:
        current, opacity_diagnostics = refine_carrier_opacities(
            current,
            table,
            config=config,
        )
        diagnostics["opacity"] = opacity_diagnostics
    diagnostics["phase_seconds"]["opacity"] = time.perf_counter() - started
    opacity = current.detach()

    started = time.perf_counter()
    if config.repair_appearance:
        current, appearance_diagnostics = refine_carrier_appearance(
            current,
            table,
            config=config,
        )
        diagnostics["appearance"] = appearance_diagnostics
    diagnostics["phase_seconds"]["appearance"] = time.perf_counter() - started
    appearance = current.detach()
    diagnostics["total_seconds"] = sum(diagnostics["phase_seconds"].values())
    return CarrierRepairResult(
        beam=beam,
        covariance=covariance,
        opacity=opacity,
        appearance=appearance,
        gaussians=current.detach(),
        diagnostics=diagnostics,
    )


__all__ = [
    "CarrierObservationTable",
    "CarrierRepairConfig",
    "CarrierRepairResult",
    "appearance_link_residuals",
    "build_carrier_observation_table",
    "covariance_reprojection_residuals",
    "opacity_optical_density_residuals",
    "refine_carrier_appearance",
    "refine_carrier_covariances",
    "refine_carrier_opacities",
    "repair_beam_carriers",
]
