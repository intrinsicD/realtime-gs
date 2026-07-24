"""Fixed-capacity 2D-gaussian pool with a free list for the native stage-1 fitter.

Opt-in allocation substrate (default off; enabled by ``FitConfig.pool``) that ports StructSplat's
pooled row lifecycle (ADR-0020) to realtime-gs stage 1. Parameter tensors are allocated once at a
fixed ``capacity`` and never resized during the fit. A boolean ``active`` mask marks live vs.
parked rows; the parked rows form the free list that triage events spend on spawns and refill by
parks.

Because rtgs's 2D compositor is *additive* (``rtgs.image2gs.renderer2d``: pixel = sum_i
weight_i * color_i * G_i, no normalizing denominator), a parked row is rendered by simply omitting
it from the live set: its forward contribution and its gradient are then exactly zero for every
parameter. Slot recycling writes rows in place and zeros only the touched rows' Adam moments, so the
optimizer is never rebuilt -- the contrast with the classic grow-by-concat / rebuild density path.

Phase 1 scaffold: the default native fit (fixed N, no add/remove) is unchanged, and the
densification/pruning policy here is deliberately minimal. Establishing the allocation mechanism is
the goal; tuning the policy against the fixed-N baseline is a separate benchmarked experiment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.metrics import masked_psnr, psnr
from rtgs.image2gs.fit import (
    _MIN_DIAG,
    _fit_loss,
    _softplus_inv,
    _validate_initial_gaussians,
)
from rtgs.image2gs.renderer2d import render_gaussians_2d

if TYPE_CHECKING:
    from rtgs.image2gs.fit import FitConfig

# Raw parameter names, in the order handed to the optimizer (mirrors the default
# ``weight_color_9p`` parameterization in ``rtgs.image2gs.fit``).
_RAW_NAMES = ("xy_raw", "diag_raw", "off_raw", "color_raw", "weight_raw")


def _raw_from_gaussians(g: Gaussians2D, wh: torch.Tensor) -> dict[str, torch.Tensor]:
    """Unconstrained parameters for ``g`` (mirrors ``fit._fit_native_from_initialization``)."""
    return {
        "xy_raw": torch.logit((g.xy / wh).clamp(1e-4, 1 - 1e-4)),
        "diag_raw": _softplus_inv(g.chol[:, [0, 2]] - _MIN_DIAG),
        "off_raw": g.chol[:, 1].clone(),
        "color_raw": torch.logit(g.color.clamp(1e-3, 1 - 1e-3)),
        "weight_raw": torch.logit(g.weight.clamp(1e-3, 1 - 1e-3)),
    }


class GaussianPool2D:
    """Fixed-capacity raw-parameter store with an ``active``/free-list row lifecycle.

    ``capacity`` rows are allocated once. The first ``g0.n`` are live (seeded from ``g0``); the rest
    are parked spares. Live rows are the only ones rendered, so parked rows cost nothing and receive
    no gradient. ``park``/``activate`` recycle rows in place and zero only the touched rows' Adam
    moments -- the raw tensors (and therefore the optimizer's parameters) are never reallocated.
    """

    def __init__(self, g0: Gaussians2D, wh: torch.Tensor, capacity: int):
        if capacity < g0.n:
            raise ValueError(f"pool capacity {capacity} is below the initial count {g0.n}")
        device, dtype = g0.xy.device, g0.xy.dtype
        self.capacity = int(capacity)
        self.wh = wh
        self._raw: dict[str, torch.Tensor] = {}
        seed = _raw_from_gaussians(g0, wh)
        for name in _RAW_NAMES:
            init = seed[name]
            buffer = torch.zeros((capacity, *init.shape[1:]), device=device, dtype=dtype)
            buffer[: g0.n] = init.detach()
            buffer.requires_grad_(True)
            self._raw[name] = buffer
        self.active = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.active[: g0.n] = True

    # -- introspection ---------------------------------------------------------------------------
    def params(self) -> list[torch.Tensor]:
        """The raw leaf tensors, in optimizer order."""
        return [self._raw[name] for name in _RAW_NAMES]

    @property
    def live_count(self) -> int:
        return int(self.active.sum())

    def active_indices(self) -> torch.Tensor:
        return self.active.nonzero(as_tuple=True)[0]

    def free_rows(self) -> torch.Tensor:
        """Parked-row indices (the free list), ascending -- the deterministic spend order."""
        return (~self.active).nonzero(as_tuple=True)[0]

    # -- rendering -------------------------------------------------------------------------------
    def _build_rows(self, rows: torch.Tensor) -> Gaussians2D:
        diag = torch.nn.functional.softplus(self._raw["diag_raw"][rows]) + _MIN_DIAG
        chol = torch.stack([diag[:, 0], self._raw["off_raw"][rows], diag[:, 1]], dim=-1)
        return Gaussians2D(
            xy=torch.sigmoid(self._raw["xy_raw"][rows]) * self.wh,
            chol=chol,
            color=torch.sigmoid(self._raw["color_raw"][rows]),
            weight=torch.sigmoid(self._raw["weight_raw"][rows]),
        )

    def build_active(self) -> Gaussians2D:
        """Differentiable live-only gaussians (parked rows are omitted -> zero contribution)."""
        return self._build_rows(self.active_indices())

    def to_gaussians2d(self) -> Gaussians2D:
        """Detached snapshot of the live rows."""
        with torch.no_grad():
            return self._build_rows(self.active_indices()).detach()

    def built_weights(self) -> torch.Tensor:
        """Detached ``(capacity,)`` amplitudes; parked rows report their stored (unused) value."""
        with torch.no_grad():
            return torch.sigmoid(self._raw["weight_raw"])

    # -- lifecycle -------------------------------------------------------------------------------
    def _zero_moments(self, optimizer: torch.optim.Optimizer, rows: torch.Tensor) -> None:
        for param in self.params():
            state = optimizer.state.get(param)
            if not state:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key][rows] = 0.0

    def park(self, rows: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        """Return ``rows`` to the free list; clears their optimizer momentum."""
        self.active[rows] = False
        self._zero_moments(optimizer, rows)

    def activate(
        self,
        rows: torch.Tensor,
        gaussians: Gaussians2D,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Write ``gaussians`` into free ``rows`` in place, mark them live, reset their moments."""
        raw = _raw_from_gaussians(gaussians, self.wh)
        with torch.no_grad():
            for name in _RAW_NAMES:
                self._raw[name].data[rows] = raw[name].to(self._raw[name])
        self.active[rows] = True
        self._zero_moments(optimizer, rows)


def _residual_spawn_gaussians(
    target: torch.Tensor,
    rendered: torch.Tensor,
    mask: torch.Tensor | None,
    count: int,
) -> Gaussians2D | None:
    """Seed up to ``count`` gaussians at the highest-error pixels (deterministic top-k)."""
    height, width = target.shape[:2]
    error = (target - rendered).abs().sum(dim=-1)
    if mask is not None:
        error = error * mask.to(error).clamp(0, 1)
    flat = error.reshape(-1)
    count = min(count, int((flat > 0).sum()))
    if count <= 0:
        return None
    idx = torch.topk(flat, count).indices
    py = (idx // width).to(target.dtype) + 0.5
    px = (idx % width).to(target.dtype) + 0.5
    scale = max((height * width / max(flat.numel(), 1)) ** 0.5 * 0.6, _MIN_DIAG + 0.1)
    chol = torch.zeros(count, 3, device=target.device, dtype=target.dtype)
    chol[:, 0] = scale
    chol[:, 2] = scale
    return Gaussians2D(
        xy=torch.stack([px, py], dim=-1),
        chol=chol,
        color=target.reshape(-1, 3)[idx].clamp(1e-3, 1 - 1e-3),
        weight=torch.full((count,), 0.5, device=target.device, dtype=target.dtype),
    )


def _triage(
    pool: GaussianPool2D,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    config: FitConfig,
    optimizer: torch.optim.Optimizer,
    height: int,
    width: int,
) -> None:
    """One in-place triage event: park the lowest-amplitude live rows, spawn at residual peaks."""
    # Park: lowest-amplitude live rows, keeping at least ``pool_min_live`` alive.
    prune = min(config.pool_prune_count, max(0, pool.live_count - config.pool_min_live))
    if prune > 0:
        live = pool.active_indices()
        weights = pool.built_weights()[live]
        victims = live[torch.topk(weights, prune, largest=False).indices]
        pool.park(victims, optimizer)
    # Spawn: fill free rows at the current highest-error pixels.
    free = pool.free_rows()
    budget = min(config.pool_spawn_count, int(free.numel()))
    if budget > 0:
        with torch.no_grad():
            rendered = render_gaussians_2d(
                pool.build_active(),
                height,
                width,
                row_chunk=config.row_chunk,
                renderer=config.native_renderer,
            )
        seeds = _residual_spawn_gaussians(target, rendered, mask, budget)
        if seeds is not None:
            pool.activate(free[: seeds.n], seeds, optimizer)


def fit_pooled_from_initialization(
    image: torch.Tensor,
    target: torch.Tensor,
    g0: Gaussians2D,
    config: FitConfig,
    mask: torch.Tensor | None,
    xy_offset: torch.Tensor,
) -> tuple[Gaussians2D, dict]:
    """Pooled native fit from a supplied initialization (see ``FitConfig.pool``).

    Reuses the caller's crop/mask/seed/initialization; returns detached live gaussians (restored to
    full-image coordinates) and the usual PSNR history dict.
    """
    height, width = image.shape[:2]
    _validate_initial_gaussians(g0, height, width)
    wh = image.new_tensor([width, height])
    capacity = config.pool_capacity or config.max_gaussians or (2 * config.n_gaussians)
    pool = GaussianPool2D(g0, wh, capacity)

    opt = torch.optim.Adam(pool.params(), lr=config.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config.iterations, eta_min=config.lr * 0.1
    )
    history: dict = {"psnr": [], "stopped_iter": config.iterations - 1}
    best_psnr = -float("inf")
    stale_checks = 0
    for it in range(config.iterations):
        opt.zero_grad()
        rendered = render_gaussians_2d(
            pool.build_active(),
            height,
            width,
            row_chunk=config.row_chunk,
            renderer=config.native_renderer,
        )
        loss = _fit_loss(rendered, target, mask)
        loss.backward()
        opt.step()
        sched.step()
        do_triage = (
            config.pool_triage_every
            and (it + 1) % config.pool_triage_every == 0
            and it + 1 < config.iterations
        )
        if do_triage:
            _triage(pool, target, mask, config, opt, height, width)
        if it % config.log_every == 0 or it == config.iterations - 1:
            with torch.no_grad():
                history["psnr"].append((it, psnr(rendered.clamp(0, 1), target)))
        if config.convergence_patience and (it + 1) % config.convergence_check_every == 0:
            with torch.no_grad():
                cur = psnr(rendered.clamp(0, 1), target)
            if cur > best_psnr + config.convergence_tol:
                best_psnr = cur
                stale_checks = 0
            else:
                stale_checks += 1
                if stale_checks >= config.convergence_patience:
                    history["stopped_iter"] = it
                    break

    result = pool.to_gaussians2d()
    with torch.no_grad():
        final = render_gaussians_2d(
            result, height, width, row_chunk=config.row_chunk, renderer=config.native_renderer
        )
        history["final_psnr_full"] = psnr(final.clamp(0, 1), target)
        history["final_psnr"] = (
            history["final_psnr_full"]
            if mask is None
            else masked_psnr(final.clamp(0, 1), image, mask)
        )
    history["live_count"] = pool.live_count
    history["pool_capacity"] = pool.capacity
    result.xy += xy_offset
    return result, history
