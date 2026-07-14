"""Lazy adapters for gsplat's CUDA density-control strategies.

The CPU/reference path keeps :class:`rtgs.optim.density.DensityController`.  This module
bridges the canonical gsplat parameter/optimizer contract only when a CUDA strategy is
explicitly selected, preserving CPU-first imports and backend pluggability.
"""

from __future__ import annotations

import torch

from rtgs.optim.density import DensityConfig
from rtgs.render.base import RenderOutput


class GsplatStrategyController:
    """Drive gsplat DefaultStrategy or MCMCStrategy and record its state transitions."""

    def __init__(
        self,
        name: str,
        config: DensityConfig,
        scene_extent: float,
        params: dict[str, torch.nn.Parameter],
        optimizers: dict[str, torch.optim.Optimizer],
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
        self.stats: list[dict[str, int | str]] = []
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
        before = int(params["means"].shape[0])
        dead_before = int(
            (torch.sigmoid(params["opacities"].detach()) <= self.config.prune_opacity).sum().item()
        )
        if self.name == "gsplat-default":
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

        pruned_to_budget = enforce_budget(
            params,
            optimizers,
            self.config.max_gaussians,
            strategy_state=self.state if self.name == "gsplat-default" else None,
        )
        after = int(params["means"].shape[0])
        scheduled = (
            step > self.config.start_iter
            and step < self.config.stop_iter
            and step % self.config.every == 0
        )
        if scheduled or pruned_to_budget:
            self.stats.append(
                {
                    "strategy": self.name,
                    "iteration": step,
                    "n_before": before,
                    "n_after": after,
                    "dead_before": dead_before,
                    "pruned_to_budget": pruned_to_budget,
                }
            )


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
