"""Geometrically growing storage for Stage-3 Gaussian parameters and Adam state.

The arena separates logical size from physical capacity.  Parameter leaves are zero-copy views
of a dense active prefix, so autograd, rendering, and Adam retain live-row shapes while the
underlying parameter and moment tensors grow only geometrically.  Topology edits are committed as
one ordered transaction and refresh the lightweight leaf views without rebuilding physical
storage when capacity is sufficient.

This module is CPU import-safe and contains no gsplat dependency.  The CUDA gsplat adapter owns
policy decisions and supplies clone/split masks plus split offsets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ArenaMigration:
    """One physical capacity growth event."""

    iteration: int
    old_capacity: int
    new_capacity: int
    active_n: int
    required_n: int
    copied_bytes: int


@dataclass(frozen=True)
class ArenaTopologyReceipt:
    """Summary of one ordered clone/split/prune transaction."""

    n_before: int
    n_after_growth: int
    n_after: int
    n_duplicate: int
    n_split: int
    n_prune: int
    pruned_to_budget: int
    capacity_before: int
    capacity_after: int


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("arena size must be positive")
    return 1 << (value - 1).bit_length()


class GeometricParameterArena:
    """Capacity-backed parameter leaves with capacity-backed Adam moments.

    ``params`` and ``optimizers`` are updated in place so callers retain their existing mapping
    objects.  Every active parameter is a leaf view sharing the first ``active_n`` rows of its
    physical storage.  Adam's state tensors are corresponding views into fixed-capacity moment
    storage, which keeps standard PyTorch Adam updates at the logical shape.
    """

    def __init__(
        self,
        params: dict[str, torch.nn.Parameter],
        optimizers: dict[str, torch.optim.Optimizer],
        *,
        max_capacity: int,
        growth_factor: float = 2.0,
        initial_capacity: int | None = None,
    ) -> None:
        if not params:
            raise ValueError("arena requires at least one parameter field")
        if set(params) != set(optimizers):
            raise ValueError("arena parameter and optimizer keys must match")
        counts = {int(parameter.shape[0]) for parameter in params.values()}
        if len(counts) != 1:
            raise ValueError("arena parameters must share one leading row count")
        active_n = counts.pop()
        if active_n <= 0:
            raise ValueError("arena requires at least one active Gaussian")
        if max_capacity < active_n:
            raise ValueError(f"arena max_capacity {max_capacity} is below active count {active_n}")
        if not math.isfinite(growth_factor) or growth_factor <= 1.0:
            raise ValueError("arena growth_factor must be finite and greater than one")
        if initial_capacity is None:
            capacity = min(max_capacity, _next_power_of_two(active_n))
        else:
            if isinstance(initial_capacity, bool) or not isinstance(initial_capacity, int):
                raise TypeError("arena initial_capacity must be an integer or None")
            if initial_capacity < active_n or initial_capacity > max_capacity:
                raise ValueError("arena initial_capacity must be between active_n and max_capacity")
            capacity = initial_capacity

        self.params = params
        self.optimizers = optimizers
        self.active_n = active_n
        self.capacity = capacity
        self.max_capacity = int(max_capacity)
        self.growth_factor = float(growth_factor)
        self.migrations: list[ArenaMigration] = []
        self._data: dict[str, torch.Tensor] = {}
        self._exp_avg: dict[str, torch.Tensor] = {}
        self._exp_avg_sq: dict[str, torch.Tensor] = {}
        self._steps: dict[str, torch.Tensor] = {}
        self._requires_grad: dict[str, bool] = {}

        for name, parameter in list(params.items()):
            optimizer = optimizers[name]
            if len(optimizer.param_groups) != 1:
                raise ValueError(f"arena optimizer '{name}' must have one parameter group")
            if optimizer.param_groups[0]["params"] != [parameter]:
                raise ValueError(f"arena optimizer '{name}' must own its matching parameter")
            previous = optimizer.state.pop(parameter, None)
            storage = parameter.detach().new_empty((capacity, *parameter.shape[1:]))
            storage[:active_n].copy_(parameter.detach())
            exp_avg = torch.zeros_like(storage)
            exp_avg_sq = torch.zeros_like(storage)
            if previous:
                old_avg = previous.get("exp_avg")
                old_avg_sq = previous.get("exp_avg_sq")
                if old_avg is not None:
                    exp_avg[:active_n].copy_(old_avg)
                if old_avg_sq is not None:
                    exp_avg_sq[:active_n].copy_(old_avg_sq)
                step = previous.get("step")
            else:
                step = None
            if step is None:
                # Standard non-capturable Adam keeps its scalar clock on CPU.
                step = torch.tensor(0.0, dtype=torch.float32)
            self._data[name] = storage
            self._exp_avg[name] = exp_avg
            self._exp_avg_sq[name] = exp_avg_sq
            self._steps[name] = step
            self._requires_grad[name] = bool(parameter.requires_grad)
            self._refresh_parameter(name)

    def _refresh_parameter(self, name: str) -> None:
        optimizer = self.optimizers[name]
        old = self.params.get(name)
        if old is not None:
            optimizer.state.pop(old, None)
        parameter = torch.nn.Parameter(
            self._data[name][: self.active_n],
            requires_grad=self._requires_grad[name],
        )
        self.params[name] = parameter
        optimizer.param_groups[0]["params"] = [parameter]
        optimizer.state[parameter] = {
            "step": self._steps[name],
            "exp_avg": self._exp_avg[name][: self.active_n],
            "exp_avg_sq": self._exp_avg_sq[name][: self.active_n],
        }

    def _refresh_all(self) -> None:
        for name in self.params:
            self._refresh_parameter(name)

    def active_state(self, name: str, key: str) -> torch.Tensor:
        """Return an active-prefix Adam moment view."""
        if key == "exp_avg":
            return self._exp_avg[name][: self.active_n]
        if key == "exp_avg_sq":
            return self._exp_avg_sq[name][: self.active_n]
        raise KeyError(key)

    def reserve(self, required_n: int, *, iteration: int) -> bool:
        """Ensure physical capacity and migrate all fields at most once."""
        required_n = int(required_n)
        if required_n <= self.capacity:
            return False
        if required_n > self.max_capacity:
            raise ValueError(
                f"arena requires {required_n} rows but max_capacity is {self.max_capacity}"
            )
        new_capacity = self.capacity
        while new_capacity < required_n:
            grown = max(new_capacity + 1, int(math.ceil(new_capacity * self.growth_factor)))
            new_capacity = min(self.max_capacity, max(required_n, grown))

        copied_bytes = 0
        for name in self.params:
            old_data = self._data[name]
            old_avg = self._exp_avg[name]
            old_avg_sq = self._exp_avg_sq[name]
            shape = (new_capacity, *old_data.shape[1:])
            new_data = old_data.new_empty(shape)
            new_avg = torch.zeros(shape, device=old_avg.device, dtype=old_avg.dtype)
            new_avg_sq = torch.zeros(shape, device=old_avg_sq.device, dtype=old_avg_sq.dtype)
            new_data[: self.active_n].copy_(old_data[: self.active_n])
            new_avg[: self.active_n].copy_(old_avg[: self.active_n])
            new_avg_sq[: self.active_n].copy_(old_avg_sq[: self.active_n])
            copied_bytes += (
                old_data[: self.active_n].numel() * old_data.element_size()
                + old_avg[: self.active_n].numel() * old_avg.element_size()
                + old_avg_sq[: self.active_n].numel() * old_avg_sq.element_size()
            )
            self._data[name] = new_data
            self._exp_avg[name] = new_avg
            self._exp_avg_sq[name] = new_avg_sq

        old_capacity = self.capacity
        self.capacity = new_capacity
        self._refresh_all()
        self.migrations.append(
            ArenaMigration(
                iteration=int(iteration),
                old_capacity=old_capacity,
                new_capacity=new_capacity,
                active_n=self.active_n,
                required_n=required_n,
                copied_bytes=copied_bytes,
            )
        )
        return True

    @torch.no_grad()
    def write_field(
        self,
        name: str,
        values: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
    ) -> None:
        """Write one already-ordered transaction result into physical storage."""
        count = int(values.shape[0])
        expected_tail = self._data[name].shape[1:]
        if values.shape[1:] != expected_tail:
            raise ValueError(f"arena field '{name}' value shape changed")
        if exp_avg.shape != values.shape or exp_avg_sq.shape != values.shape:
            raise ValueError(f"arena field '{name}' Adam state shape changed")
        if count > self.capacity:
            raise ValueError(f"arena write of {count} rows exceeds capacity {self.capacity}")
        self._data[name][:count].copy_(values)
        self._exp_avg[name][:count].copy_(exp_avg)
        self._exp_avg_sq[name][:count].copy_(exp_avg_sq)

    def set_active_n(self, active_n: int) -> None:
        """Publish a new dense-prefix logical size after all fields are written."""
        active_n = int(active_n)
        if not 0 < active_n <= self.capacity:
            raise ValueError(f"arena active_n must be in (0, {self.capacity}]")
        self.active_n = active_n
        self._refresh_all()

    @torch.no_grad()
    def reset_opacity(self, value: float) -> None:
        """Clamp active opacity logits and reset their Adam moments in place."""
        limit = torch.logit(torch.tensor(float(value))).item()
        self._data["opacities"][: self.active_n].clamp_(max=limit)
        self._exp_avg["opacities"][: self.active_n].zero_()
        self._exp_avg_sq["opacities"][: self.active_n].zero_()

    def diagnostics(self) -> dict[str, Any]:
        """Return JSON-safe physical storage diagnostics."""
        row_bytes = 0
        for value in self._data.values():
            row_bytes += math.prod(value.shape[1:]) * value.element_size() * 3
        return {
            "policy": "geometric",
            "active_n": self.active_n,
            "capacity": self.capacity,
            "max_capacity": self.max_capacity,
            "growth_factor": self.growth_factor,
            "adam_parameter_row_bytes": row_bytes,
            "physical_parameter_adam_bytes": row_bytes * self.capacity,
            "active_parameter_adam_bytes": row_bytes * self.active_n,
            "migration_count": len(self.migrations),
            "migration_copied_bytes": sum(item.copied_bytes for item in self.migrations),
            "migrations": [
                {
                    "iteration": item.iteration,
                    "old_capacity": item.old_capacity,
                    "new_capacity": item.new_capacity,
                    "active_n": item.active_n,
                    "required_n": item.required_n,
                    "copied_bytes": item.copied_bytes,
                }
                for item in self.migrations
            ],
        }


def _expanded_field(
    name: str,
    value: torch.Tensor,
    clone_rows: torch.Tensor,
    split_rows: torch.Tensor,
    survivor_mask: torch.Tensor,
    split_offsets: torch.Tensor,
    *,
    split_factor: float,
    revised_opacity: bool,
) -> torch.Tensor:
    """Construct gsplat DefaultStrategy's post-grow row order once."""
    survivors = value[survivor_mask]
    clones = value[clone_rows]
    source = value[split_rows]
    if name == "means":
        child0 = source + split_offsets[0]
        child1 = source + split_offsets[1]
    elif name == "scales":
        child = torch.log(torch.exp(source) / float(split_factor))
        child0 = child
        child1 = child
    elif name == "opacities" and revised_opacity:
        child = torch.logit(1.0 - torch.sqrt(1.0 - torch.sigmoid(source)))
        child0 = child
        child1 = child
    else:
        child0 = source
        child1 = source
    return torch.cat((survivors, clones, child0, child1), dim=0)


def _expanded_moment(
    value: torch.Tensor,
    survivor_mask: torch.Tensor,
    clone_count: int,
    split_count: int,
) -> torch.Tensor:
    tail = value.shape[1:]
    return torch.cat(
        (
            value[survivor_mask],
            value.new_zeros((clone_count, *tail)),
            value.new_zeros((split_count, *tail)),
            value.new_zeros((split_count, *tail)),
        ),
        dim=0,
    )


@torch.no_grad()
def apply_default_topology_transaction(
    arena: GeometricParameterArena,
    *,
    clone_mask: torch.Tensor,
    split_mask: torch.Tensor,
    split_offsets: torch.Tensor,
    split_factor: float,
    revised_opacity: bool,
    prune_opacity: float,
    prune_large_scale: float | None,
    max_gaussians: int,
    iteration: int,
) -> ArenaTopologyReceipt:
    """Apply one DefaultStrategy-equivalent grow/prune wave as a single transaction."""
    n_before = arena.active_n
    if clone_mask.shape != (n_before,) or clone_mask.dtype != torch.bool:
        raise ValueError("clone_mask must be a boolean active-row mask")
    if split_mask.shape != (n_before,) or split_mask.dtype != torch.bool:
        raise ValueError("split_mask must be a boolean active-row mask")
    if bool((clone_mask & split_mask).any()):
        raise ValueError("clone and split masks must be disjoint")
    clone_rows = torch.where(clone_mask)[0]
    split_rows = torch.where(split_mask)[0]
    survivor_mask = ~split_mask
    n_duplicate = int(clone_rows.numel())
    n_split = int(split_rows.numel())
    if split_offsets.shape != (2, n_split, 3):
        raise ValueError("split_offsets must have shape (2, n_split, 3)")

    expanded_opacity = _expanded_field(
        "opacities",
        arena.params["opacities"].detach(),
        clone_rows,
        split_rows,
        survivor_mask,
        split_offsets,
        split_factor=split_factor,
        revised_opacity=revised_opacity,
    )
    expanded_scales = _expanded_field(
        "scales",
        arena.params["scales"].detach(),
        clone_rows,
        split_rows,
        survivor_mask,
        split_offsets,
        split_factor=split_factor,
        revised_opacity=revised_opacity,
    )
    n_after_growth = int(expanded_opacity.shape[0])
    prune_mask = torch.sigmoid(expanded_opacity.flatten()) < float(prune_opacity)
    if prune_large_scale is not None:
        prune_mask |= torch.exp(expanded_scales).max(dim=-1).values > float(prune_large_scale)
    n_prune = int(prune_mask.sum().item())

    remaining = int((~prune_mask).sum().item())
    pruned_to_budget = max(remaining - int(max_gaussians), 0)
    if pruned_to_budget:
        eligible = torch.where(~prune_mask)[0]
        significance = torch.sigmoid(expanded_opacity[eligible]).flatten() * torch.exp(
            expanded_scales[eligible]
        ).prod(dim=-1)
        remove = eligible[torch.topk(significance, pruned_to_budget, largest=False).indices]
        prune_mask[remove] = True
    keep = ~prune_mask
    n_after = int(keep.sum().item())
    if n_after <= 0:
        raise RuntimeError("arena topology transaction would remove every Gaussian")

    capacity_before = arena.capacity
    arena.reserve(n_after, iteration=iteration)
    for name, parameter in list(arena.params.items()):
        expanded = _expanded_field(
            name,
            parameter.detach(),
            clone_rows,
            split_rows,
            survivor_mask,
            split_offsets,
            split_factor=split_factor,
            revised_opacity=revised_opacity,
        )
        exp_avg = _expanded_moment(
            arena.active_state(name, "exp_avg"),
            survivor_mask,
            n_duplicate,
            n_split,
        )
        exp_avg_sq = _expanded_moment(
            arena.active_state(name, "exp_avg_sq"),
            survivor_mask,
            n_duplicate,
            n_split,
        )
        arena.write_field(name, expanded[keep], exp_avg[keep], exp_avg_sq[keep])
    arena.set_active_n(n_after)
    return ArenaTopologyReceipt(
        n_before=n_before,
        n_after_growth=n_after_growth,
        n_after=n_after,
        n_duplicate=n_duplicate,
        n_split=n_split,
        n_prune=n_prune,
        pruned_to_budget=pruned_to_budget,
        capacity_before=capacity_before,
        capacity_after=arena.capacity,
    )
