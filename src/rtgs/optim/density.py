"""Adaptive density control: screen-space-gradient driven clone/split/prune.

Faithful-but-compact port of the original 3DGS strategy (Kerbl et al. 2023; see also
gsplat's DefaultStrategy): gaussians whose average screen-space positional gradient
exceeds a threshold are densified — cloned if small, split if large — and low-opacity or
oversized gaussians are pruned. All edits preserve Adam moments (zeros for newborn
gaussians), matching the reference behavior.

Gradient normalization: means2d gradients arrive in pixel units; we scale by
max(W, H)/2 to express them in NDC-like units so the classic 2e-4 threshold applies.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rtgs.render.base import RenderOutput


@dataclass
class DensityConfig:
    """Density-control hyperparameters (iteration numbers are trainer iterations)."""

    start_iter: int = 60
    stop_iter: int = 10_000_000
    every: int = 40
    grad_threshold: float = 2e-4
    # Split (instead of clone) when the max scale exceeds this fraction of scene extent.
    split_scale_frac: float = 0.01
    split_factor: float = 1.6
    prune_opacity: float = 0.01
    # Prune when the max scale exceeds this fraction of scene extent.
    prune_scale_frac: float = 0.5
    max_gaussians: int = 100_000
    opacity_reset_every: int = 0  # 0 disables (short refinement runs don't need it)
    opacity_reset_value: float = 0.011


class DensityController:
    """Accumulates densification statistics and performs param/optimizer surgery."""

    def __init__(self, config: DensityConfig, n_gaussians: int, scene_extent: float):
        self.cfg = config
        self.extent = scene_extent
        self.grad_accum = torch.zeros(n_gaussians)
        self.count = torch.zeros(n_gaussians)
        self.stats: list[dict] = []

    def accumulate(self, out: RenderOutput, width: int, height: int) -> None:
        """Record screen-space positional gradients after loss.backward()."""
        if out.means2d is None or out.means2d.grad is None or out.visible is None:
            return
        norm = out.means2d.grad.norm(dim=-1) * (max(width, height) * 0.5)
        self.grad_accum[out.visible] += norm
        self.count[out.visible] += 1.0

    def step(
        self,
        iteration: int,
        params: dict[str, torch.Tensor],
        optimizer: torch.optim.Adam,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run one clone/split/prune (+optional opacity reset) round if scheduled."""
        cfg = self.cfg
        if iteration < cfg.start_iter or iteration > cfg.stop_iter or iteration % cfg.every != 0:
            return params

        n = params["means"].shape[0]
        avg_grad = self.grad_accum / self.count.clamp_min(1.0)
        scales = params["log_scales"].detach().exp()
        opacity = torch.sigmoid(params["opacity_logit"].detach())
        scale_max = scales.max(dim=-1).values

        densify = (avg_grad > cfg.grad_threshold) & (self.count > 0)
        if n >= cfg.max_gaussians:
            densify = torch.zeros_like(densify)
        is_large = scale_max > cfg.split_scale_frac * self.extent
        clone_mask = densify & ~is_large
        split_mask = densify & is_large
        prune_mask = (opacity < cfg.prune_opacity) | (
            scale_max > cfg.prune_scale_frac * self.extent
        )
        # Split replaces the original: the original is pruned, two children are added.
        keep_mask = ~(prune_mask | split_mask)

        extras: list[dict[str, torch.Tensor]] = []
        if bool(clone_mask.any()):
            extras.append({k: v.detach()[clone_mask].clone() for k, v in params.items()})
        if bool(split_mask.any()):
            for _ in range(2):
                child = {k: v.detach()[split_mask].clone() for k, v in params.items()}
                from rtgs.core.gaussians3d import quat_to_rotmat

                rot = quat_to_rotmat(child["quats"])
                s = child["log_scales"].exp()
                noise = torch.randn(s.shape, generator=generator) * s
                child["means"] = child["means"] + (rot @ noise[..., None])[..., 0]
                child["log_scales"] = child["log_scales"] - torch.log(
                    torch.tensor(cfg.split_factor)
                )
                extras.append(child)

        new_params = _edit_params(optimizer, params, keep_mask, extras)

        n_new = new_params["means"].shape[0]
        self.grad_accum = torch.zeros(n_new)
        self.count = torch.zeros(n_new)
        self.stats.append(
            {
                "iteration": iteration,
                "n_before": n,
                "n_after": n_new,
                "cloned": int(clone_mask.sum()),
                "split": int(split_mask.sum()),
                "pruned": int(prune_mask.sum()),
            }
        )

        if cfg.opacity_reset_every and iteration % cfg.opacity_reset_every == 0:
            with torch.no_grad():
                cap = torch.logit(torch.tensor(cfg.opacity_reset_value))
                new_params["opacity_logit"].clamp_max_(cap)
        return new_params


def _edit_params(
    optimizer: torch.optim.Adam,
    params: dict[str, torch.Tensor],
    keep_mask: torch.Tensor,
    extras: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Keep a subset of each param and append new rows, preserving Adam moments.

    Optimizer param groups must be registered one param per group, with the group's
    'name' matching the params dict key (the trainer guarantees this).
    """
    new_params: dict[str, torch.Tensor] = {}
    for group in optimizer.param_groups:
        name = group["name"]
        old = params[name]
        rows = [old.detach()[keep_mask]] + [e[name] for e in extras]
        new = torch.cat(rows).requires_grad_(True)

        state = optimizer.state.pop(old, None)
        if state is not None:
            for key in ("exp_avg", "exp_avg_sq"):
                buf = state[key]
                pads = [
                    torch.zeros((e[name].shape[0], *buf.shape[1:]), dtype=buf.dtype) for e in extras
                ]
                state[key] = torch.cat([buf[keep_mask]] + pads)
            optimizer.state[new] = state
        group["params"] = [new]
        new_params[name] = new
    return new_params
