"""CPU-first soft correspondence for exact inverse-projection fibers.

This module deliberately keeps four quantities separate:

* Gaussian geometry determines pairwise compatibility.
* Caller-supplied capacities determine transport budget.
* A correspondence plan reports posterior association and explanation support.
* Rendering opacity is not represented here.

The fitting loop is generalized-EM-inspired rather than a convergent BCPD implementation.  Each
E-step builds a deterministic, detached correspondence plan; one or more M-steps then optimize
only the existing :class:`~rtgs.lift.inverse_projection_fiber.InverseProjectionFiber`
coordinates.  A track's source view is always excluded from soft supervision because that
projection is already an exact construction invariant.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.lift.inverse_projection_fiber import InverseProjectionFiber
from rtgs.render.projection import EWA_NEAR, EWAProjection

CapacityMode = Literal["uniform", "footprint_area"]
AssignmentMethod = Literal["row_softmax", "unbalanced_sinkhorn"]


def _validate_spd_batch(name: str, matrices: torch.Tensor) -> torch.Tensor:
    if matrices.ndim != 3 or matrices.shape[-2:] != (2, 2):
        raise ValueError(f"{name} must have shape (N,2,2)")
    if matrices.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if not matrices.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(matrices).all()):
        raise ValueError(f"{name} must be finite")
    symmetric = 0.5 * (matrices + matrices.transpose(-1, -2))
    scale = float(symmetric.detach().abs().amax().clamp_min(1))
    tolerance = 64.0 * torch.finfo(matrices.dtype).eps * scale
    if float((matrices - matrices.transpose(-1, -2)).detach().abs().amax()) > tolerance:
        raise ValueError(f"{name} must be symmetric")
    if bool((torch.linalg.eigvalsh(symmetric) <= 0).any()):
        raise ValueError(f"{name} must be positive definite")
    return symmetric


def _validate_cost(cost: torch.Tensor) -> None:
    if cost.ndim != 2 or cost.shape[0] == 0 or cost.shape[1] == 0:
        raise ValueError("cost must be a non-empty matrix")
    if not cost.is_floating_point():
        raise TypeError("cost must be floating point")
    if not bool(torch.isfinite(cost).all()):
        raise ValueError("cost must be finite")


def _capacity_tensor(
    value: torch.Tensor | float | None,
    *,
    count: int,
    like: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if value is None:
        result = like.new_ones(count)
    else:
        result = torch.as_tensor(value, dtype=like.dtype, device=like.device)
        try:
            result = result.expand(count).clone()
        except RuntimeError as error:
            raise ValueError(f"{name} must be scalar or have shape ({count},)") from error
    if result.shape != (count,):
        raise ValueError(f"{name} must be scalar or have shape ({count},)")
    if not bool(torch.isfinite(result).all()) or bool((result <= 0).any()):
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _candidate_mask(cost: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(cost, dtype=torch.bool)
    if mask.shape != cost.shape or mask.dtype != torch.bool or mask.device != cost.device:
        raise ValueError("candidate_mask must be boolean with the same shape/device as cost")
    return mask


def _finite_nonnegative_scalar(value: float, *, name: str) -> float:
    scalar = float(value)
    if not math.isfinite(scalar) or scalar < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return scalar


@dataclass(frozen=True)
class ObservationGaussians:
    """Frozen 2D geometry and explicit transport capacities for one camera view.

    ``capacities`` are transport budget, never confidence or rendering opacity.  Use
    :meth:`from_field` to preserve corrected native means and the field's complete effective
    covariance (filter variance plus AA dilation).
    """

    means: torch.Tensor
    covariances: torch.Tensor
    capacities: torch.Tensor
    dilation: float
    capacity_mode: str = "explicit"

    def __post_init__(self) -> None:
        if self.means.ndim != 2 or self.means.shape[1] != 2 or self.means.shape[0] == 0:
            raise ValueError("means must have shape (N,2) with N > 0")
        if not self.means.is_floating_point() or not bool(torch.isfinite(self.means).all()):
            raise ValueError("means must be finite floating point")
        covariance = _validate_spd_batch("covariances", self.covariances)
        if covariance.device != self.means.device or covariance.dtype != self.means.dtype:
            raise ValueError("means and covariances must share device and dtype")
        capacities = _capacity_tensor(
            self.capacities,
            count=self.means.shape[0],
            like=self.means[:, 0],
            name="capacities",
        )
        dilation = _finite_nonnegative_scalar(self.dilation, name="dilation")
        if not isinstance(self.capacity_mode, str) or not self.capacity_mode:
            raise ValueError("capacity_mode must be a non-empty string")
        object.__setattr__(self, "means", self.means.detach().clone())
        object.__setattr__(self, "covariances", covariance.detach().clone())
        object.__setattr__(self, "capacities", capacities.detach().clone())
        object.__setattr__(self, "dilation", dilation)

    @property
    def n(self) -> int:
        return int(self.means.shape[0])

    @classmethod
    def from_field(
        cls,
        field: GaussianObservationField,
        *,
        dtype: torch.dtype = torch.float64,
        capacity_mode: CapacityMode = "uniform",
        capacities: torch.Tensor | float | None = None,
    ) -> ObservationGaussians:
        """Adapt a native observation field without treating amplitude as confidence.

        If ``capacities`` is supplied it is preserved exactly (after dtype conversion), which is
        useful for known-mass synthetic splits.  Otherwise ``uniform`` assigns unit capacity and
        ``footprint_area`` assigns ``sqrt(det(covariance))``.
        """

        if not isinstance(field, GaussianObservationField):
            raise TypeError("field must be GaussianObservationField")
        if dtype not in {torch.float32, torch.float64}:
            raise TypeError("dtype must be torch.float32 or torch.float64")
        if capacity_mode not in {"uniform", "footprint_area"}:
            raise ValueError("capacity_mode must be 'uniform' or 'footprint_area'")

        means = field.native_means(dtype=dtype)
        variances = field.effective_variances().to(dtype=dtype)
        rotations = field.rotations.to(dtype=dtype)
        cos = rotations.cos()
        sin = rotations.sin()
        covariance = torch.stack(
            [
                cos.square() * variances[:, 0] + sin.square() * variances[:, 1],
                cos * sin * (variances[:, 0] - variances[:, 1]),
                cos * sin * (variances[:, 0] - variances[:, 1]),
                sin.square() * variances[:, 0] + cos.square() * variances[:, 1],
            ],
            dim=-1,
        ).reshape(-1, 2, 2)
        if capacities is None:
            if capacity_mode == "uniform":
                capacity = means.new_ones(means.shape[0])
            else:
                capacity = torch.linalg.det(covariance).sqrt()
            mode = capacity_mode
        else:
            capacity = _capacity_tensor(
                capacities,
                count=means.shape[0],
                like=means[:, 0],
                name="capacities",
            )
            mode = "explicit"
        return cls(
            means=means,
            covariances=covariance,
            capacities=capacity,
            dilation=field.aa_dilation,
            capacity_mode=mode,
        )


def pairwise_bhattacharyya_cost(
    first_means: torch.Tensor,
    first_covariances: torch.Tensor,
    second_means: torch.Tensor,
    second_covariances: torch.Tensor,
    *,
    residual_variance: float = 0.0,
) -> torch.Tensor:
    """Return all-pairs Bhattacharyya distances between 2D Gaussian geometry.

    ``residual_variance`` adds the same isotropic mismatch variance to both covariance batches.
    The covariance path remains differentiable; unlike the legacy center metric, it is not
    detached.
    """

    if first_means.ndim != 2 or first_means.shape[1] != 2 or first_means.shape[0] == 0:
        raise ValueError("first_means must have shape (N,2) with N > 0")
    if second_means.ndim != 2 or second_means.shape[1] != 2 or second_means.shape[0] == 0:
        raise ValueError("second_means must have shape (M,2) with M > 0")
    if (
        not first_means.is_floating_point()
        or not second_means.is_floating_point()
        or not bool(torch.isfinite(first_means).all())
        or not bool(torch.isfinite(second_means).all())
    ):
        raise ValueError("means must be finite floating point")
    if first_means.device != second_means.device or first_means.dtype != second_means.dtype:
        raise ValueError("mean batches must share device and dtype")
    first = _validate_spd_batch("first_covariances", first_covariances)
    second = _validate_spd_batch("second_covariances", second_covariances)
    if first.device != first_means.device or first.dtype != first_means.dtype:
        raise ValueError("first means and covariances must share device and dtype")
    if second.device != second_means.device or second.dtype != second_means.dtype:
        raise ValueError("second means and covariances must share device and dtype")

    sigma2 = _finite_nonnegative_scalar(residual_variance, name="residual_variance")
    if sigma2:
        identity = torch.eye(2, dtype=first.dtype, device=first.device)
        first = first + sigma2 * identity
        second = second + sigma2 * identity
    average = 0.5 * (first[:, None, :, :] + second[None, :, :, :])
    delta = first_means[:, None, :] - second_means[None, :, :]
    solved = torch.linalg.solve(average, delta.unsqueeze(-1))
    quadratic = 0.125 * (delta.unsqueeze(-2) @ solved).squeeze(-1).squeeze(-1)
    logdet_average = torch.linalg.slogdet(average).logabsdet
    logdet_first = torch.linalg.slogdet(first).logabsdet[:, None]
    logdet_second = torch.linalg.slogdet(second).logabsdet[None, :]
    determinant = 0.5 * (logdet_average - 0.5 * (logdet_first + logdet_second))
    return (quadratic + determinant).clamp_min(0)


@dataclass(frozen=True)
class CorrespondencePlan:
    """Transport output with posterior, capacity, and rendering semantics kept distinct."""

    real_mass: torch.Tensor
    track_dustbin_mass: torch.Tensor
    observation_dustbin_mass: torch.Tensor | None
    dustbin_dustbin_mass: torch.Tensor | None
    track_capacities: torch.Tensor
    observation_capacities: torch.Tensor | None
    method: str
    iterations: int
    fixed_point_residual: float | None = None

    def __post_init__(self) -> None:
        if self.real_mass.ndim != 2 or 0 in self.real_mass.shape:
            raise ValueError("real_mass must be a non-empty matrix")
        if not self.real_mass.is_floating_point():
            raise TypeError("correspondence masses must be floating point")
        rows, columns = self.real_mass.shape
        tensors = {
            "real_mass": self.real_mass,
            "track_dustbin_mass": self.track_dustbin_mass,
            "track_capacities": self.track_capacities,
        }
        if self.track_dustbin_mass.shape != (rows,) or self.track_capacities.shape != (rows,):
            raise ValueError("track masses and capacities must have shape (N,)")
        transport_fields = (
            self.observation_dustbin_mass,
            self.dustbin_dustbin_mass,
            self.observation_capacities,
        )
        if any(value is None for value in transport_fields) and not all(
            value is None for value in transport_fields
        ):
            raise ValueError(
                "observation dustbin, completion, and capacities must be all set or all None"
            )
        if self.observation_dustbin_mass is not None:
            if self.observation_dustbin_mass.shape != (columns,):
                raise ValueError("observation_dustbin_mass must have shape (M,)")
            if self.dustbin_dustbin_mass is None or self.dustbin_dustbin_mass.numel() != 1:
                raise ValueError("dustbin_dustbin_mass must be scalar")
            if self.observation_capacities is None or self.observation_capacities.shape != (
                columns,
            ):
                raise ValueError("observation_capacities must have shape (M,)")
            tensors.update(
                {
                    "observation_dustbin_mass": self.observation_dustbin_mass,
                    "dustbin_dustbin_mass": self.dustbin_dustbin_mass,
                    "observation_capacities": self.observation_capacities,
                }
            )
        for name, value in tensors.items():
            if value.dtype != self.real_mass.dtype or value.device != self.real_mass.device:
                raise ValueError(f"{name} must share real_mass dtype and device")
            if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
                raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("method must be a non-empty string")
        if not isinstance(self.iterations, int) or self.iterations < 0:
            raise ValueError("iterations must be a non-negative integer")
        if self.fixed_point_residual is not None:
            residual = float(self.fixed_point_residual)
            if not math.isfinite(residual) or residual < 0:
                raise ValueError("fixed_point_residual must be finite and non-negative")
            object.__setattr__(self, "fixed_point_residual", residual)

    @property
    def track_row_mass(self) -> torch.Tensor:
        return self.real_mass.sum(dim=1) + self.track_dustbin_mass

    @property
    def track_real_probability(self) -> torch.Tensor:
        denominator = self.track_row_mass.clamp_min(torch.finfo(self.real_mass.dtype).tiny)
        return self.real_mass / denominator[:, None]

    @property
    def track_dustbin_probability(self) -> torch.Tensor:
        denominator = self.track_row_mass.clamp_min(torch.finfo(self.real_mass.dtype).tiny)
        return self.track_dustbin_mass / denominator

    @property
    def track_support(self) -> torch.Tensor:
        """Fraction of each realized row marginal assigned to real observations."""

        return self.track_real_probability.sum(dim=1)

    @property
    def track_entropy(self) -> torch.Tensor:
        """Entropy over each row's real associations plus its dustbin."""

        probabilities = torch.cat(
            [self.track_real_probability, self.track_dustbin_probability[:, None]],
            dim=1,
        )
        log_probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        return -(probabilities * log_probabilities).sum(dim=1)

    @property
    def observation_support(self) -> torch.Tensor | None:
        if self.observation_dustbin_mass is None:
            return None
        real = self.real_mass.sum(dim=0)
        denominator = (real + self.observation_dustbin_mass).clamp_min(torch.finfo(real.dtype).tiny)
        return real / denominator

    @property
    def augmented_mass(self) -> torch.Tensor | None:
        if self.observation_dustbin_mass is None or self.dustbin_dustbin_mass is None:
            return None
        top = torch.cat([self.real_mass, self.track_dustbin_mass[:, None]], dim=1)
        bottom = torch.cat([self.observation_dustbin_mass, self.dustbin_dustbin_mass.reshape(1)])
        return torch.cat([top, bottom[None, :]], dim=0)

    @property
    def augmented_target_marginals(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the normalized augmented UOT targets in the plan's mass units."""

        if self.observation_capacities is None:
            return None
        track_total = self.track_capacities.sum()
        observation_total = self.observation_capacities.sum()
        normalizer = track_total + observation_total
        if float(normalizer) <= 0:
            zeros_row = torch.zeros(
                self.real_mass.shape[0] + 1,
                dtype=self.real_mass.dtype,
                device=self.real_mass.device,
            )
            zeros_column = torch.zeros(
                self.real_mass.shape[1] + 1,
                dtype=self.real_mass.dtype,
                device=self.real_mass.device,
            )
            return zeros_row, zeros_column
        row = torch.cat([self.track_capacities, observation_total.reshape(1)]) / normalizer
        column = torch.cat([self.observation_capacities, track_total.reshape(1)]) / normalizer
        return row, column


def row_softmax_plan(
    cost: torch.Tensor,
    *,
    temperature: float,
    dustbin_cost: float,
    track_capacities: torch.Tensor | float | None = None,
    candidate_mask: torch.Tensor | None = None,
) -> CorrespondencePlan:
    """One-sided soft assignment with one finite dustbin per track row."""

    _validate_cost(cost)
    temperature_value = float(temperature)
    if not math.isfinite(temperature_value) or temperature_value <= 0:
        raise ValueError("temperature must be finite and positive")
    dustbin = _finite_nonnegative_scalar(dustbin_cost, name="dustbin_cost")
    capacities = _capacity_tensor(
        track_capacities,
        count=cost.shape[0],
        like=cost[:, 0],
        name="track_capacities",
    )
    mask = _candidate_mask(cost, candidate_mask)
    negative_infinity = torch.full_like(cost, -torch.inf)
    real_logits = torch.where(mask, -cost / temperature_value, negative_infinity)
    dustbin_logits = cost.new_full((cost.shape[0], 1), -dustbin / temperature_value)
    probabilities = torch.softmax(torch.cat([real_logits, dustbin_logits], dim=1), dim=1)
    real_probability = probabilities[:, :-1]
    return CorrespondencePlan(
        real_mass=capacities[:, None] * real_probability,
        track_dustbin_mass=capacities * probabilities[:, -1],
        observation_dustbin_mass=None,
        dustbin_dustbin_mass=None,
        track_capacities=capacities,
        observation_capacities=None,
        method="row_softmax",
        iterations=1,
        fixed_point_residual=0.0,
    )


def unbalanced_sinkhorn_plan(
    cost: torch.Tensor,
    *,
    track_capacities: torch.Tensor | float,
    observation_capacities: torch.Tensor | float,
    temperature: float,
    marginal_penalty: float,
    dustbin_cost: float,
    iterations: int = 100,
    tolerance: float = 1e-7,
    candidate_mask: torch.Tensor | None = None,
) -> CorrespondencePlan:
    """Log-domain augmented unbalanced Sinkhorn transport.

    Real capacities ``a`` and ``b`` are augmented with dust-row mass ``sum(b)`` and dust-column
    mass ``sum(a)``, then normalized by ``sum(a)+sum(b)``.  Finite ``marginal_penalty`` relaxes
    both target marginals through generalized Sinkhorn updates.  Pass ``math.inf`` for balanced
    updates.  Invalid real pairs have exactly zero mass; every dustbin edge remains finite.
    """

    _validate_cost(cost)
    temperature_value = float(temperature)
    if not math.isfinite(temperature_value) or temperature_value <= 0:
        raise ValueError("temperature must be finite and positive")
    penalty = float(marginal_penalty)
    if math.isnan(penalty) or penalty <= 0:
        raise ValueError("marginal_penalty must be positive or math.inf")
    dustbin = _finite_nonnegative_scalar(dustbin_cost, name="dustbin_cost")
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    tolerance_value = float(tolerance)
    if not math.isfinite(tolerance_value) or tolerance_value < 0:
        raise ValueError("tolerance must be finite and non-negative")

    track = _capacity_tensor(
        track_capacities,
        count=cost.shape[0],
        like=cost[:, 0],
        name="track_capacities",
    )
    observation = _capacity_tensor(
        observation_capacities,
        count=cost.shape[1],
        like=cost[0],
        name="observation_capacities",
    )
    mask = _candidate_mask(cost, candidate_mask)
    track_total = track.sum()
    observation_total = observation.sum()
    normalizer = track_total + observation_total
    row_target = torch.cat([track, observation_total.reshape(1)]) / normalizer
    column_target = torch.cat([observation, track_total.reshape(1)]) / normalizer

    augmented_cost = cost.new_full((cost.shape[0] + 1, cost.shape[1] + 1), dustbin)
    augmented_cost[:-1, :-1] = cost
    augmented_mask = torch.ones_like(augmented_cost, dtype=torch.bool)
    augmented_mask[:-1, :-1] = mask
    log_kernel = torch.where(
        augmented_mask,
        -augmented_cost / temperature_value,
        torch.full_like(augmented_cost, -torch.inf),
    )
    log_row_target = row_target.log()
    log_column_target = column_target.log()
    relaxation = 1.0 if math.isinf(penalty) else penalty / (penalty + temperature_value)
    log_u = torch.zeros_like(row_target)
    log_v = torch.zeros_like(column_target)
    completed_iterations = iterations
    for iteration in range(iterations):
        previous_u = log_u
        previous_v = log_v
        log_u = relaxation * (log_row_target - torch.logsumexp(log_kernel + log_v[None, :], dim=1))
        log_v = relaxation * (
            log_column_target - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
        )
        if tolerance_value:
            change = torch.maximum(
                (log_u - previous_u).abs().amax(),
                (log_v - previous_v).abs().amax(),
            )
            if float(change.detach()) <= tolerance_value:
                completed_iterations = iteration + 1
                break
    next_u = relaxation * (log_row_target - torch.logsumexp(log_kernel + log_v[None, :], dim=1))
    next_v = relaxation * (log_column_target - torch.logsumexp(log_kernel + next_u[:, None], dim=0))
    fixed_point_residual = float(
        torch.maximum((next_u - log_u).abs().amax(), (next_v - log_v).abs().amax()).detach()
    )
    augmented = torch.exp(log_u[:, None] + log_kernel + log_v[None, :])
    return CorrespondencePlan(
        real_mass=augmented[:-1, :-1],
        track_dustbin_mass=augmented[:-1, -1],
        observation_dustbin_mass=augmented[-1, :-1],
        dustbin_dustbin_mass=augmented[-1, -1],
        track_capacities=track,
        observation_capacities=observation,
        method="unbalanced_sinkhorn",
        iterations=completed_iterations,
        fixed_point_residual=fixed_point_residual,
    )


def exponential_schedule(start: float, stop: float, steps: int) -> tuple[float, ...]:
    """Return an endpoint-exact deterministic exponential schedule."""

    start_value = float(start)
    stop_value = float(stop)
    if (
        not math.isfinite(start_value)
        or not math.isfinite(stop_value)
        or start_value <= 0
        or stop_value <= 0
    ):
        raise ValueError("schedule endpoints must be finite and positive")
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if steps == 1:
        return (start_value,)
    ratio = stop_value / start_value
    values = tuple(start_value * ratio ** (index / (steps - 1)) for index in range(steps))
    return (start_value, *values[1:-1], stop_value)


@dataclass(frozen=True)
class FiberFitConfig:
    """Deterministic alternating-fit controls."""

    temperatures: tuple[float, ...] = (4.0, 2.0, 1.0)
    residual_variances: tuple[float, ...] = (4.0, 2.0, 1.0)
    geometry_steps: int = 8
    learning_rate: float = 0.03
    assignment: AssignmentMethod = "unbalanced_sinkhorn"
    dustbin_cost: float = 8.0
    marginal_penalty: float = 8.0
    sinkhorn_iterations: int = 100
    sinkhorn_tolerance: float = 1e-7
    track_batch_size: int = 128
    min_real_mass: float = 1e-10
    max_grad_norm: float | None = 10.0
    source_center_tolerance: float = 1e-7
    source_covariance_tolerance: float = 1e-7

    def __post_init__(self) -> None:
        temperatures = tuple(float(value) for value in self.temperatures)
        residuals = tuple(float(value) for value in self.residual_variances)
        if not temperatures or len(temperatures) != len(residuals):
            raise ValueError("temperature and residual-variance schedules must have equal length")
        if any(not math.isfinite(value) or value <= 0 for value in temperatures):
            raise ValueError("temperatures must be finite and positive")
        if any(not math.isfinite(value) or value < 0 for value in residuals):
            raise ValueError("residual variances must be finite and non-negative")
        if not isinstance(self.geometry_steps, int) or self.geometry_steps <= 0:
            raise ValueError("geometry_steps must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.assignment not in {"row_softmax", "unbalanced_sinkhorn"}:
            raise ValueError("assignment must be 'row_softmax' or 'unbalanced_sinkhorn'")
        _finite_nonnegative_scalar(self.dustbin_cost, name="dustbin_cost")
        penalty = float(self.marginal_penalty)
        if math.isnan(penalty) or penalty <= 0:
            raise ValueError("marginal_penalty must be positive or math.inf")
        if not isinstance(self.sinkhorn_iterations, int) or self.sinkhorn_iterations <= 0:
            raise ValueError("sinkhorn_iterations must be a positive integer")
        if not math.isfinite(self.sinkhorn_tolerance) or self.sinkhorn_tolerance < 0:
            raise ValueError("sinkhorn_tolerance must be finite and non-negative")
        if not isinstance(self.track_batch_size, int) or self.track_batch_size <= 0:
            raise ValueError("track_batch_size must be a positive integer")
        if not math.isfinite(self.min_real_mass) or self.min_real_mass <= 0:
            raise ValueError("min_real_mass must be finite and positive")
        if self.max_grad_norm is not None and (
            not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0
        ):
            raise ValueError("max_grad_norm must be finite and positive when supplied")
        for name in ("source_center_tolerance", "source_covariance_tolerance"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        object.__setattr__(self, "temperatures", temperatures)
        object.__setattr__(self, "residual_variances", residuals)


@dataclass(frozen=True)
class FiberFitStep:
    """One completed frozen-plan outer iteration."""

    temperature: float
    residual_variance: float
    loss: float
    real_mass: float
    mean_track_support: float
    mean_track_entropy: float
    source_center_error: float
    source_covariance_relative_error: float


@dataclass(frozen=True)
class FiberFitResult:
    """Final correspondence plans and invariant-aware optimization history."""

    plans: tuple[CorrespondencePlan, ...]
    history: tuple[FiberFitStep, ...]
    source_center_error: float
    source_covariance_relative_error: float


def _source_errors(fiber: InverseProjectionFiber) -> tuple[float, float]:
    source_means, source_covariances, _ = fiber.source_projection()
    center = (source_means - fiber.source_means2d).norm(dim=-1).amax()
    denominator = (
        fiber.source_covariances2d.flatten(1)
        .norm(dim=-1)
        .clamp_min(torch.finfo(source_covariances.dtype).tiny)
    )
    covariance = (
        (source_covariances - fiber.source_covariances2d).flatten(1).norm(dim=-1) / denominator
    ).amax()
    return float(center.detach()), float(covariance.detach())


def _check_source_invariant(
    fiber: InverseProjectionFiber,
    config: FiberFitConfig,
) -> tuple[float, float]:
    center, covariance = _source_errors(fiber)
    if (
        not math.isfinite(center)
        or not math.isfinite(covariance)
        or center > config.source_center_tolerance
        or covariance > config.source_covariance_tolerance
    ):
        raise RuntimeError(
            "inverse-projection source invariant failed "
            f"(center={center:.6g}, covariance={covariance:.6g})"
        )
    return center, covariance


def validate_fiber_state(fiber: InverseProjectionFiber) -> None:
    """Fail closed unless every raw coordinate and materialized 3D Gaussian is valid."""

    parameters = tuple(fiber.parameters())
    if not parameters or any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
        raise RuntimeError("inverse-projection fiber has non-finite raw parameters")
    depths = fiber.depths()
    if not bool(torch.isfinite(depths).all()):
        raise RuntimeError("inverse-projection fiber has non-finite depths")
    if bool(((depths <= fiber.depth_lower) | (depths >= fiber.depth_upper)).any()):
        raise RuntimeError("inverse-projection fiber depth reached a bound")
    ray_variance_innovation = (2.0 * fiber.log_ray_scale).exp()
    if not bool(torch.isfinite(ray_variance_innovation).all()) or bool(
        (ray_variance_innovation <= 0).any()
    ):
        raise RuntimeError("inverse-projection fiber ray variance underflowed or overflowed")
    means, covariances = fiber.means_covariances()
    if not bool(torch.isfinite(means).all()) or not bool(torch.isfinite(covariances).all()):
        raise RuntimeError("inverse-projection fiber materialized non-finite geometry")
    if bool((torch.linalg.cholesky_ex(covariances).info != 0).any()):
        raise RuntimeError("inverse-projection fiber covariance is not positive definite")


def safe_projection_geometry(
    camera: Camera,
    projected: EWAProjection,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return finite/SPD cost inputs plus the renderer-compatible validity mask.

    Invalid rows receive harmless finite placeholders so callers can form a dense cost matrix,
    but they must use the returned mask when assigning or supervising correspondences.  The helper
    is public so experimental controls can share exactly the production candidate domain.
    """

    finite_means = torch.isfinite(projected.means2d).all(dim=-1)
    finite_covariances = torch.isfinite(projected.covariances2d).all(dim=(-2, -1))
    finite_depth = torch.isfinite(projected.depth)
    safe_means = torch.where(
        finite_means[:, None],
        projected.means2d,
        torch.zeros_like(projected.means2d),
    )
    identity = torch.eye(
        2,
        dtype=projected.covariances2d.dtype,
        device=projected.covariances2d.device,
    )
    safe_covariances = torch.where(
        finite_covariances[:, None, None],
        projected.covariances2d,
        identity,
    )
    safe_covariances = 0.5 * (safe_covariances + safe_covariances.transpose(-1, -2))
    positive_definite = torch.linalg.cholesky_ex(safe_covariances).info == 0
    safe_covariances = torch.where(
        positive_definite[:, None, None],
        safe_covariances,
        identity,
    )
    radius = 3.0 * torch.linalg.eigvalsh(safe_covariances)[..., -1].sqrt()
    visible = (projected.depth > EWA_NEAR) & camera.in_image(
        safe_means,
        margin=radius.detach(),
    )
    valid = finite_means & finite_covariances & finite_depth & positive_definite & visible
    return safe_means, safe_covariances, valid


def _detached_plan(
    plan: CorrespondencePlan,
) -> CorrespondencePlan:
    return CorrespondencePlan(
        real_mass=plan.real_mass.detach(),
        track_dustbin_mass=plan.track_dustbin_mass.detach(),
        observation_dustbin_mass=(
            None
            if plan.observation_dustbin_mass is None
            else plan.observation_dustbin_mass.detach()
        ),
        dustbin_dustbin_mass=(
            None if plan.dustbin_dustbin_mass is None else plan.dustbin_dustbin_mass.detach()
        ),
        track_capacities=plan.track_capacities.detach(),
        observation_capacities=(
            None if plan.observation_capacities is None else plan.observation_capacities.detach()
        ),
        method=plan.method,
        iterations=plan.iterations,
        fixed_point_residual=plan.fixed_point_residual,
    )


def _scatter_track_plan(
    plan: CorrespondencePlan,
    track_indices: torch.Tensor,
    *,
    track_count: int,
) -> CorrespondencePlan:
    """Restore full track rows after solving a view only over non-source tracks."""

    real_mass = plan.real_mass.new_zeros((track_count, plan.real_mass.shape[1]))
    track_dustbin_mass = plan.track_dustbin_mass.new_zeros(track_count)
    track_capacities = plan.track_capacities.new_zeros(track_count)
    real_mass[track_indices] = plan.real_mass
    track_dustbin_mass[track_indices] = plan.track_dustbin_mass
    track_capacities[track_indices] = plan.track_capacities
    return CorrespondencePlan(
        real_mass=real_mass,
        track_dustbin_mass=track_dustbin_mass,
        observation_dustbin_mass=plan.observation_dustbin_mass,
        dustbin_dustbin_mass=plan.dustbin_dustbin_mass,
        track_capacities=track_capacities,
        observation_capacities=plan.observation_capacities,
        method=plan.method,
        iterations=plan.iterations,
        fixed_point_residual=plan.fixed_point_residual,
    )


def _empty_view_plan(
    *,
    track_count: int,
    observation_count: int,
    like: torch.Tensor,
    method: AssignmentMethod,
) -> CorrespondencePlan:
    """Represent a view with no non-source tracks as absent evidence."""

    transport = method == "unbalanced_sinkhorn"
    return CorrespondencePlan(
        real_mass=like.new_zeros((track_count, observation_count)),
        track_dustbin_mass=like.new_zeros(track_count),
        observation_dustbin_mass=(like.new_zeros(observation_count) if transport else None),
        dustbin_dustbin_mass=(like.new_tensor(0.0) if transport else None),
        track_capacities=like.new_zeros(track_count),
        observation_capacities=(like.new_zeros(observation_count) if transport else None),
        method=method,
        iterations=0,
        fixed_point_residual=0.0,
    )


def _make_plan(
    cost: torch.Tensor,
    *,
    mask: torch.Tensor,
    track_capacities: torch.Tensor,
    observation_capacities: torch.Tensor,
    temperature: float,
    config: FiberFitConfig,
) -> CorrespondencePlan:
    if config.assignment == "row_softmax":
        return row_softmax_plan(
            cost,
            temperature=temperature,
            dustbin_cost=config.dustbin_cost,
            track_capacities=track_capacities,
            candidate_mask=mask,
        )
    return unbalanced_sinkhorn_plan(
        cost,
        track_capacities=track_capacities,
        observation_capacities=observation_capacities,
        temperature=temperature,
        marginal_penalty=config.marginal_penalty,
        dustbin_cost=config.dustbin_cost,
        iterations=config.sinkhorn_iterations,
        tolerance=config.sinkhorn_tolerance,
        candidate_mask=mask,
    )


def _e_step(
    fiber: InverseProjectionFiber,
    observations: Sequence[ObservationGaussians],
    track_capacities: torch.Tensor,
    *,
    temperature: float,
    residual_variance: float,
    config: FiberFitConfig,
) -> tuple[CorrespondencePlan, ...]:
    plans: list[CorrespondencePlan] = []
    with torch.no_grad():
        for view_index, (camera, target) in enumerate(
            zip(fiber.cameras, observations, strict=True)
        ):
            projected = fiber.project(camera)
            projected_means, projected_covariances, valid = safe_projection_geometry(
                camera, projected
            )
            non_source = fiber.source_view_indices != view_index
            track_indices = non_source.nonzero(as_tuple=True)[0]
            if track_indices.numel() == 0:
                plans.append(
                    _empty_view_plan(
                        track_count=fiber.n,
                        observation_count=target.n,
                        like=target.means[:, 0],
                        method=config.assignment,
                    )
                )
                continue
            cost = pairwise_bhattacharyya_cost(
                projected_means[track_indices],
                projected_covariances[track_indices],
                target.means,
                target.covariances,
                residual_variance=residual_variance,
            )
            mask = valid[track_indices, None].expand_as(cost)
            plans.append(
                _detached_plan(
                    _scatter_track_plan(
                        _make_plan(
                            cost,
                            mask=mask,
                            track_capacities=track_capacities[track_indices],
                            observation_capacities=target.capacities,
                            temperature=temperature,
                            config=config,
                        ),
                        track_indices,
                        track_count=fiber.n,
                    )
                )
            )
    return tuple(plans)


def _m_step(
    fiber: InverseProjectionFiber,
    observations: Sequence[ObservationGaussians],
    plans: Sequence[CorrespondencePlan],
    optimizer: torch.optim.Optimizer,
    *,
    residual_variance: float,
    config: FiberFitConfig,
) -> tuple[float, float]:
    view_masses = tuple(float(plan.real_mass.detach().sum()) for plan in plans)
    total_mass = sum(view_masses)
    if not math.isfinite(total_mass) or total_mass < config.min_real_mass:
        raise RuntimeError(
            f"transported real mass {total_mass:.6g} is below fail-closed minimum "
            f"{config.min_real_mass:.6g}"
        )

    optimizer.zero_grad(set_to_none=True)
    loss_value = 0.0
    active_view_count = sum(mass >= config.min_real_mass for mass in view_masses)
    if active_view_count == 0:
        raise RuntimeError("no camera view transported enough real mass")
    for camera, target, plan, view_mass in zip(
        fiber.cameras, observations, plans, view_masses, strict=True
    ):
        if view_mass < config.min_real_mass:
            continue
        active_batches: list[tuple[int, int]] = []
        for start in range(0, fiber.n, config.track_batch_size):
            stop = min(start + config.track_batch_size, fiber.n)
            if float(plan.real_mass[start:stop].sum()) > 0:
                active_batches.append((start, stop))
        if not active_batches:
            continue

        projected = fiber.project(camera)
        projected_means, projected_covariances, valid = safe_projection_geometry(camera, projected)
        supported = plan.real_mass.sum(dim=1) > 0
        if bool((supported & ~valid).any()):
            raise RuntimeError("a supported projection left the valid camera domain during M-step")
        for batch_index, (start, stop) in enumerate(active_batches):
            cost = pairwise_bhattacharyya_cost(
                projected_means[start:stop],
                projected_covariances[start:stop],
                target.means,
                target.covariances,
                residual_variance=residual_variance,
            )
            contribution = (plan.real_mass[start:stop] * cost).sum() / view_mass / active_view_count
            contribution.backward(retain_graph=batch_index + 1 < len(active_batches))
            loss_value += float(contribution.detach())

    parameters = tuple(parameter for parameter in fiber.parameters() if parameter.requires_grad)
    invalid_gradient = any(
        parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )
    if invalid_gradient:
        raise RuntimeError("fiber correspondence produced missing or non-finite gradients")
    if config.max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    validate_fiber_state(fiber)
    return loss_value, total_mass


def fit_fiber_correspondence(
    fiber: InverseProjectionFiber,
    observations: Sequence[ObservationGaussians],
    *,
    config: FiberFitConfig | None = None,
    track_capacities: torch.Tensor | float | None = None,
) -> FiberFitResult:
    """Fit fiber coordinates with alternating frozen soft-correspondence plans.

    The function mutates ``fiber`` in place.  It performs no automatic pruning, merging,
    splitting, opacity updates, or appearance fitting; final plan support and entropy are
    returned so an experiment can make those decisions explicitly.
    """

    if not isinstance(fiber, InverseProjectionFiber):
        raise TypeError("fiber must be InverseProjectionFiber")
    if config is None:
        config = FiberFitConfig()
    if not isinstance(config, FiberFitConfig):
        raise TypeError("config must be FiberFitConfig")
    if len(observations) != len(fiber.cameras):
        raise ValueError("observations must contain exactly one entry per fiber camera")
    for view_index, target in enumerate(observations):
        if not isinstance(target, ObservationGaussians):
            raise TypeError(f"observation {view_index} must be ObservationGaussians")
        if target.means.device != fiber.source_means2d.device:
            raise ValueError("fiber and observations must share a device")
        if target.means.dtype != fiber.source_means2d.dtype:
            raise ValueError("fiber and observations must share a dtype")
        if not math.isclose(target.dilation, fiber.dilation, rel_tol=0, abs_tol=1e-12):
            raise ValueError("observation and fiber AA dilation must agree")
    capacities = _capacity_tensor(
        track_capacities,
        count=fiber.n,
        like=fiber.source_means2d[:, 0],
        name="track_capacities",
    )
    validate_fiber_state(fiber)
    _check_source_invariant(fiber, config)
    optimizer = torch.optim.Adam(fiber.parameters(), lr=config.learning_rate)
    history: list[FiberFitStep] = []
    plans: tuple[CorrespondencePlan, ...] = ()

    for temperature, residual_variance in zip(
        config.temperatures,
        config.residual_variances,
        strict=True,
    ):
        _check_source_invariant(fiber, config)
        plans = _e_step(
            fiber,
            observations,
            capacities,
            temperature=temperature,
            residual_variance=residual_variance,
            config=config,
        )
        loss = math.nan
        real_mass = math.nan
        for _ in range(config.geometry_steps):
            loss, real_mass = _m_step(
                fiber,
                observations,
                plans,
                optimizer,
                residual_variance=residual_variance,
                config=config,
            )
            center_error, covariance_error = _check_source_invariant(fiber, config)
        diagnostic_support: list[torch.Tensor] = []
        diagnostic_entropy: list[torch.Tensor] = []
        diagnostic_capacity: list[torch.Tensor] = []
        for view_index, plan in enumerate(plans):
            cross_view = fiber.source_view_indices != view_index
            diagnostic_support.append(plan.track_support[cross_view])
            diagnostic_entropy.append(plan.track_entropy[cross_view])
            diagnostic_capacity.append(plan.track_capacities[cross_view])
        support = torch.cat(diagnostic_support)
        entropy = torch.cat(diagnostic_entropy)
        capacity = torch.cat(diagnostic_capacity)
        capacity_sum = capacity.sum().clamp_min(torch.finfo(capacity.dtype).tiny)
        history.append(
            FiberFitStep(
                temperature=temperature,
                residual_variance=residual_variance,
                loss=loss,
                real_mass=real_mass,
                mean_track_support=float((capacity * support).sum() / capacity_sum),
                mean_track_entropy=float((capacity * entropy).sum() / capacity_sum),
                source_center_error=center_error,
                source_covariance_relative_error=covariance_error,
            )
        )

    plans = _e_step(
        fiber,
        observations,
        capacities,
        temperature=config.temperatures[-1],
        residual_variance=config.residual_variances[-1],
        config=config,
    )
    validate_fiber_state(fiber)
    center_error, covariance_error = _check_source_invariant(fiber, config)
    return FiberFitResult(
        plans=plans,
        history=tuple(history),
        source_center_error=center_error,
        source_covariance_relative_error=covariance_error,
    )


__all__ = [
    "AssignmentMethod",
    "CapacityMode",
    "CorrespondencePlan",
    "FiberFitConfig",
    "FiberFitResult",
    "FiberFitStep",
    "ObservationGaussians",
    "exponential_schedule",
    "fit_fiber_correspondence",
    "pairwise_bhattacharyya_cost",
    "row_softmax_plan",
    "safe_projection_geometry",
    "unbalanced_sinkhorn_plan",
    "validate_fiber_state",
]
