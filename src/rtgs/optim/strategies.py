"""Lazy adapters for gsplat's CUDA density-control strategies.

The CPU/reference path keeps :class:`rtgs.optim.density.DensityController`.  This module
bridges the canonical gsplat parameter/optimizer contract only when a CUDA strategy is
explicitly selected, preserving CPU-first imports and backend pluggability.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from rtgs.optim.density import DensityConfig
from rtgs.render.base import RenderOutput

if TYPE_CHECKING:
    from rtgs.optim.arena import GeometricParameterArena


class GsplatStrategyController:
    """Drive gsplat DefaultStrategy or MCMCStrategy and record its state transitions."""

    def __init__(
        self,
        name: str,
        config: DensityConfig,
        scene_extent: float,
        params: dict[str, torch.nn.Parameter],
        optimizers: dict[str, torch.optim.Optimizer],
        *,
        arena: GeometricParameterArena | None = None,
        profile_events: bool = False,
    ) -> None:
        if params["means"].device.type != "cuda":
            raise RuntimeError(f"{name} density control requires CUDA")
        if config.every <= 0:
            raise ValueError("density every must be positive")
        if config.max_gaussians <= 0:
            raise ValueError("density max_gaussians must be positive")
        try:
            from gsplat.strategy import DefaultStrategy, MCMCStrategy
        except ImportError as exc:
            raise RuntimeError(
                f"{name} density control requires gsplat; install `pip install -e '.[cuda]'`"
            ) from exc

        self.name = name
        self.config = config
        self.arena = arena
        self.profile_events = bool(profile_events)
        self.stats: list[dict[str, Any]] = []
        if arena is not None and name != "gsplat-default":
            raise ValueError("geometric arena storage currently supports gsplat-default only")
        if name == "gsplat-default":
            reset_every = (
                config.opacity_reset_every
                if config.opacity_reset_every > 0
                else max(config.stop_iter + 1, 1_000_000_000)
            )
            self.strategy = DefaultStrategy(
                prune_opa=config.prune_opacity,
                grow_grad2d=config.grad_threshold,
                grow_scale3d=config.split_scale_frac,
                prune_scale3d=config.prune_scale_frac,
                refine_start_iter=config.start_iter,
                refine_stop_iter=config.stop_iter,
                reset_every=reset_every,
                refine_every=config.every,
                pause_refine_after_reset=config.every,
                absgrad=config.absgrad,
                revised_opacity=config.revised_opacity,
                verbose=False,
            )
            self.state = self.strategy.initialize_state(scene_scale=scene_extent)
            if arena is not None:
                device = params["means"].device
                self.state["grad2d"] = torch.zeros(arena.capacity, device=device)
                self.state["count"] = torch.zeros(arena.capacity, device=device)
        elif name == "gsplat-mcmc":
            self.strategy = MCMCStrategy(
                cap_max=config.max_gaussians,
                noise_lr=config.mcmc_noise_lr,
                refine_start_iter=config.start_iter,
                refine_stop_iter=config.stop_iter,
                refine_every=config.every,
                min_opacity=config.prune_opacity,
                verbose=False,
            )
            self.state = self.strategy.initialize_state()
        else:
            raise ValueError(f"unknown gsplat density strategy '{name}'")
        self.strategy.check_sanity(params, optimizers)

    def pre_backward(
        self,
        params: dict[str, torch.nn.Parameter],
        optimizers: dict[str, torch.optim.Optimizer],
        output: RenderOutput,
        step: int,
    ) -> None:
        """Run the strategy hook that retains screen-gradient state."""
        if output.strategy_info is None:
            raise RuntimeError(
                f"{self.name} requires raw gsplat rasterization metadata; use --rasterizer gsplat"
            )
        self.strategy.step_pre_backward(
            params=params,
            optimizers=optimizers,
            state=self.state,
            step=step,
            info=output.strategy_info,
        )

    def post_backward(
        self,
        params: dict[str, torch.nn.Parameter],
        optimizers: dict[str, torch.optim.Optimizer],
        output: RenderOutput,
        step: int,
        means_lr: float,
        packed: bool,
    ) -> None:
        """Run relocation/growth after Adam and enforce the configured hard budget."""
        assert output.strategy_info is not None
        scheduled = (
            step > self.config.start_iter
            and step < self.config.stop_iter
            and step % self.config.every == 0
        )
        profile_event = self.profile_events and scheduled
        if profile_event:
            torch.cuda.synchronize(params["means"].device)
            event_started = time.perf_counter()
        else:
            event_started = None
        before = int(params["means"].shape[0])
        dead_before = int(
            (torch.sigmoid(params["opacities"].detach()) <= self.config.prune_opacity).sum().item()
        )
        receipt = None
        if self.arena is not None:
            receipt = self._arena_default_step(
                params,
                optimizers,
                output,
                step,
                packed,
            )
            pruned_to_budget = 0 if receipt is None else receipt.pruned_to_budget
        elif self.name == "gsplat-default":
            self.strategy.step_post_backward(
                params=params,
                optimizers=optimizers,
                state=self.state,
                step=step,
                info=output.strategy_info,
                packed=packed,
            )
        else:
            self.strategy.step_post_backward(
                params=params,
                optimizers=optimizers,
                state=self.state,
                step=step,
                info=output.strategy_info,
                lr=means_lr,
            )

        if self.arena is None:
            pruned_to_budget = enforce_budget(
                params,
                optimizers,
                self.config.max_gaussians,
                strategy_state=self.state if self.name == "gsplat-default" else None,
            )
        after = int(params["means"].shape[0])
        event_seconds = None
        if event_started is not None:
            torch.cuda.synchronize(params["means"].device)
            event_seconds = time.perf_counter() - event_started
        if scheduled or pruned_to_budget:
            record = {
                "strategy": self.name,
                "storage_policy": "geometric" if self.arena is not None else "dynamic",
                "iteration": step,
                "n_before": before,
                "n_after": after,
                "dead_before": dead_before,
                "pruned_to_budget": pruned_to_budget,
            }
            if receipt is not None:
                record.update(
                    {
                        "n_after_growth": receipt.n_after_growth,
                        "n_duplicate": receipt.n_duplicate,
                        "n_split": receipt.n_split,
                        "n_prune": receipt.n_prune,
                        "capacity_before": receipt.capacity_before,
                        "capacity_after": receipt.capacity_after,
                    }
                )
            if event_seconds is not None:
                record["event_seconds"] = event_seconds
            self.stats.append(record)

    @torch.no_grad()
    def _arena_default_step(
        self,
        params: dict[str, torch.nn.Parameter],
        optimizers: dict[str, torch.optim.Optimizer],
        output: RenderOutput,
        step: int,
        packed: bool,
    ):
        """Apply DefaultStrategy policy through one arena topology transaction."""
        from gsplat.utils import normalized_quat_to_rotmat

        from rtgs.optim.arena import apply_default_topology_transaction

        assert self.arena is not None
        assert output.strategy_info is not None
        if step >= self.strategy.refine_stop_iter:
            return None

        # Reuse gsplat's exact gradient/radius accumulation; capacity-sized state accepts the
        # live gaussian ids while selection below is restricted to the active prefix.
        self.strategy._update_state(  # noqa: SLF001 - version-bound adapter by design
            params,
            self.state,
            output.strategy_info,
            packed=packed,
        )
        refine = (
            step > self.strategy.refine_start_iter
            and step % self.strategy.refine_every == 0
            and step % self.strategy.reset_every >= self.strategy.pause_refine_after_reset
        )
        receipt = None
        if refine:
            n = self.arena.active_n
            count = self.state["count"][:n]
            grads = self.state["grad2d"][:n] / count.clamp_min(1)
            is_grad_high = grads > self.strategy.grow_grad2d
            scales = torch.exp(params["scales"])
            is_small = (
                scales.max(dim=-1).values <= self.strategy.grow_scale3d * self.state["scene_scale"]
            )
            clone_mask = is_grad_high & is_small
            split_mask = is_grad_high & ~is_small
            split_rows = torch.where(split_mask)[0]
            if int(split_rows.numel()) > 0:
                split_scales = scales[split_rows]
                split_quats = F.normalize(params["quats"][split_rows], dim=-1)
                rotmats = normalized_quat_to_rotmat(split_quats)
                split_offsets = torch.einsum(
                    "nij,nj,bnj->bni",
                    rotmats,
                    split_scales,
                    torch.randn(2, len(split_rows), 3, device=params["means"].device),
                )
            else:
                split_offsets = params["means"].new_empty((2, 0, 3))
            prune_large_scale = (
                self.strategy.prune_scale3d * self.state["scene_scale"]
                if step > self.strategy.reset_every
                else None
            )
            receipt = apply_default_topology_transaction(
                self.arena,
                clone_mask=clone_mask,
                split_mask=split_mask,
                split_offsets=split_offsets,
                split_factor=1.6,
                revised_opacity=self.strategy.revised_opacity,
                prune_opacity=self.strategy.prune_opa,
                prune_large_scale=prune_large_scale,
                max_gaussians=self.config.max_gaussians,
                iteration=step,
            )
            if self.state["grad2d"].shape[0] < self.arena.capacity:
                device = self.state["grad2d"].device
                self.state["grad2d"] = torch.zeros(self.arena.capacity, device=device)
                self.state["count"] = torch.zeros(self.arena.capacity, device=device)
            else:
                self.state["grad2d"][: self.arena.active_n].zero_()
                self.state["count"][: self.arena.active_n].zero_()

        if step % self.strategy.reset_every == 0 and step > 0:
            self.arena.reset_opacity(self.strategy.prune_opa * 2.0)
        return receipt


@torch.no_grad()
def enforce_budget(
    params: dict[str, torch.nn.Parameter],
    optimizers: dict[str, torch.optim.Optimizer],
    max_gaussians: int,
    strategy_state: dict | None = None,
) -> int:
    """Remove the least significant rows and preserve per-parameter Adam state."""
    n = int(params["means"].shape[0])
    excess = max(n - max_gaussians, 0)
    if not excess:
        return 0
    opacity = torch.sigmoid(params["opacities"].detach())
    volume = torch.exp(params["scales"].detach()).prod(dim=-1)
    significance = opacity * volume
    remove = torch.topk(significance, excess, largest=False).indices
    keep = torch.ones(n, dtype=torch.bool, device=remove.device)
    keep[remove] = False

    for name, old in list(params.items()):
        new = torch.nn.Parameter(old.detach()[keep], requires_grad=old.requires_grad)
        optimizer = optimizers[name]
        state = optimizer.state.pop(old, None)
        if state is not None:
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == n:
                    state[key] = value[keep]
            optimizer.state[new] = state
        optimizer.param_groups[0]["params"] = [new]
        params[name] = new
    if strategy_state is not None:
        for key, value in list(strategy_state.items()):
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == n:
                strategy_state[key] = value[keep]
    return excess


def strategy_uses_absgrad(name: str, config: DensityConfig) -> bool:
    """Whether rasterization must produce absolute screen-space gradients."""
    return name in {"classic", "gsplat-default"} and config.absgrad


def validate_strategy_name(name: str) -> None:
    """Validate without importing gsplat."""
    choices = {"classic", "gsplat-default", "gsplat-mcmc"}
    if name not in choices:
        raise ValueError(f"unknown density strategy '{name}' (expected one of {sorted(choices)})")
