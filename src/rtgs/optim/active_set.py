"""Priority-ranked active-set gradient masking for Stage-3 optimization.

Restricts per-iteration parameter updates to a top-fraction subset of Gaussian rows scored
only from statistics the training loop already produces: accumulated screen-space gradients
and visibility counts when a density controller supplies them, plus a selector-maintained
recent gradient-variance proxy and a topology age term. Masking zeroes inactive rows'
gradients after backward and before ``optimizer.step()``. Dense Adam kernels still execute
over all rows, so this mechanism targets update efficiency (quality per effective updated
row), never wall-clock acceleration.

Row identity across topology events is approximated by prefix preservation: when the row
count grows, new tail rows are treated as newborns; when it shrinks, per-row state is
truncated. The registered active-set protocol's instrumentation runs at fixed topology or
tolerates this approximation by design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_SELECTIONS = ("priority", "random")


@dataclass(frozen=True)
class ActiveSetConfig:
    """Opt-in active-set selection; ``fraction=1.0`` keeps dense updates untouched.

    ``weight_count``, ``weight_variance``, and ``weight_age`` are the protocol's
    ``lambda_v``, ``lambda_o``, and ``lambda_a`` rank weights; the accumulated-gradient
    rank always carries weight one.
    """

    fraction: float = 1.0
    refresh_every: int = 8
    selection: str = "priority"
    weight_count: float = 0.25
    weight_variance: float = 0.25
    weight_age: float = 0.25
    variance_beta: float = 0.9

    def __post_init__(self) -> None:
        if not math.isfinite(self.fraction) or not 0.0 < self.fraction <= 1.0:
            raise ValueError("active-set fraction must be within (0, 1]")
        if self.refresh_every < 1:
            raise ValueError("active-set refresh_every must be at least one")
        if self.selection not in _SELECTIONS:
            raise ValueError(f"active-set selection must be one of: {', '.join(_SELECTIONS)}")
        for label in ("weight_count", "weight_variance", "weight_age"):
            value = getattr(self, label)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"active-set {label} must be finite and non-negative")
        if not 0.0 <= self.variance_beta < 1.0:
            raise ValueError("active-set variance_beta must be within [0, 1)")

    @property
    def enabled(self) -> bool:
        return self.fraction < 1.0


def _ranks(values: torch.Tensor) -> torch.Tensor:
    """Dense ranks in [0, n): larger value, larger rank; deterministic ties by index."""
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(values.shape[0], device=values.device)
    return ranks.to(torch.float32)


class ActiveSetSelector:
    """Maintain the active-row mask and zero inactive gradients before each Adam step."""

    def __init__(self, config: ActiveSetConfig, n_initial: int, device: torch.device) -> None:
        if n_initial <= 0:
            raise ValueError("active-set selector requires at least one row")
        self.config = config
        self.device = device
        self.mask = torch.ones(n_initial, dtype=torch.bool, device=device)
        self.grad_ema = torch.zeros(n_initial, device=device)
        self.grad_sq_ema = torch.zeros(n_initial, device=device)
        self.visibility = torch.zeros(n_initial, device=device)
        self.birth = torch.zeros(n_initial, dtype=torch.long, device=device)
        self.refresh_count = 0
        self.cumulative_row_updates = 0
        self.cumulative_steps = 0

    @property
    def n(self) -> int:
        return int(self.mask.shape[0])

    def _resize(self, n: int, step: int) -> None:
        old_n = self.n
        if n == old_n:
            return
        if n < old_n:
            self.mask = self.mask[:n]
            self.grad_ema = self.grad_ema[:n]
            self.grad_sq_ema = self.grad_sq_ema[:n]
            self.visibility = self.visibility[:n]
            self.birth = self.birth[:n]
            return
        grown = n - old_n
        self.mask = torch.cat((self.mask, torch.ones(grown, dtype=torch.bool, device=self.device)))
        self.grad_ema = torch.cat((self.grad_ema, torch.zeros(grown, device=self.device)))
        self.grad_sq_ema = torch.cat((self.grad_sq_ema, torch.zeros(grown, device=self.device)))
        self.visibility = torch.cat((self.visibility, torch.zeros(grown, device=self.device)))
        self.birth = torch.cat(
            (self.birth, torch.full((grown,), step, dtype=torch.long, device=self.device))
        )

    @torch.no_grad()
    def observe(
        self,
        params: dict[str, torch.nn.Parameter],
        visible: torch.Tensor | None,
        step: int,
    ) -> None:
        """Fold this iteration's dense gradients and visibility into the selector state."""
        n = int(params["means"].shape[0])
        self._resize(n, step)
        grad = params["means"].grad
        if grad is not None:
            norm = grad.detach().norm(dim=-1)
            beta = self.config.variance_beta
            self.grad_ema.mul_(beta).add_(norm, alpha=1.0 - beta)
            self.grad_sq_ema.mul_(beta).add_(norm * norm, alpha=1.0 - beta)
        if visible is not None and visible.numel():
            rows = visible.to(device=self.device, dtype=torch.long)
            rows = rows[rows < n]
            self.visibility.index_add_(0, rows, torch.ones(rows.shape[0], device=self.device))

    @torch.no_grad()
    def maybe_refresh(
        self,
        step: int,
        params: dict[str, torch.nn.Parameter],
        *,
        grad2d: torch.Tensor | None = None,
        count: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> bool:
        """Re-select the active set on the configured cadence; returns True on refresh."""
        n = int(params["means"].shape[0])
        self._resize(n, step)
        if self.refresh_count and step % self.config.refresh_every != 0:
            return False
        k = max(1, int(math.ceil(self.config.fraction * n)))
        if self.config.selection == "random":
            perm = torch.randperm(n, generator=generator, device=self.device)
            chosen = perm[:k]
        else:
            grad_term = grad2d[:n].to(self.device) if grad2d is not None else self.grad_ema
            count_term = count[:n].to(self.device) if count is not None else self.visibility
            variance = (self.grad_sq_ema - self.grad_ema * self.grad_ema).clamp_min(0.0)
            age = (step - self.birth).to(torch.float32)
            score = (
                _ranks(grad_term)
                + self.config.weight_count * _ranks(count_term)
                + self.config.weight_variance * _ranks(variance)
                + self.config.weight_age * _ranks(-age)
            )
            chosen = torch.topk(score, k).indices
        mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        mask[chosen] = True
        self.mask = mask
        self.refresh_count += 1
        return True

    @torch.no_grad()
    def mask_gradients(self, params: dict[str, torch.nn.Parameter]) -> int:
        """Zero inactive rows' gradients on every parameter field; returns active rows."""
        inactive = ~self.mask
        for parameter in params.values():
            if parameter.grad is not None:
                parameter.grad[inactive] = 0
        active = int(self.mask.sum())
        self.cumulative_row_updates += active
        self.cumulative_steps += 1
        return active

    def diagnostics(self) -> dict[str, float | int | str]:
        return {
            "fraction": self.config.fraction,
            "selection": self.config.selection,
            "refresh_every": self.config.refresh_every,
            "refresh_count": self.refresh_count,
            "cumulative_row_updates": self.cumulative_row_updates,
            "cumulative_steps": self.cumulative_steps,
        }
