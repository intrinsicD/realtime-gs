"""Init-preserving densification: three growth channels with appearance-preserving operators.

This module implements ADR-YYYY §§1-4, 6, 7
(`docs/ADR-YYYY-init-preserving-densification.md`).  It exists because vanilla adaptive density
control destroys a good initialization by four distinct mechanisms:

1. **Split samples children from ``N(mu, Sigma)``** — position information is replaced by a draw
   from the parent's own uncertainty.
2. **Clone and split are not appearance-preserving** — a clone at the same position with copied
   ``alpha`` changes the composite from ``alpha`` to ``2 alpha - alpha^2`` at the peak, and a
   split divides scales by the ad-hoc 1.6.  Every event perturbs the render, so the iterations
   that follow are repair rather than progress.  With a cold init that is invisible; with a warm
   init it is exactly how the warm start leaks away.
3. **The opacity reset erases an appearance-matched init** by construction.
4. **The gradient criterion conflates need with conflict** — accumulated view-space position
   gradients fire on large, correctly-placed primitives that merely carry residual colour error.

The design invariant: *every topology operation either preserves the rendered image (within a
tested tolerance) or is explicitly an insertion of new capacity into a region the current model
does not explain.*  Growth changes capacity, never the current solution.

**One documented deviation from ADR-YYYY.**  The ADR places children along a continuous in-plane
residual direction ``d``.  Scales are stored as ``(q, log s)`` with the surfel frame in ``q``, so
a continuous ``d`` cannot be shrunk exactly in that parameterization without re-deriving ``q``,
which would break the frame inheritance the ADR requires.  ``d`` is therefore quantized to the
primitive's own dominant tangential axis: the offsets and the second-moment updates are then
*exact* in the stored parameterization (acceptance criterion A2), at the cost of angular
resolution in the placement.  The continuous residual direction and its angle to the chosen
axis are logged per event so the cost is measurable rather than assumed away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from rtgs.core.camera import Camera
from rtgs.optim.density import _edit_params
from rtgs.optim.init_trust import NULL_ROOT, TrustSchedule

_EPS = 1e-12


@dataclass
class InitPreservingConfig:
    """ADR-YYYY levels.  Every field is a level the E12 grid may set, not a free knob."""

    # §7 warmup: with an appearance-matched init the residual is meaningful at iteration 0, so
    # only the accumulation window is needed, not the vanilla 500+ warmup.
    start_iter: int = 200
    stop_iter: int = 10_000_000
    every: int = 100
    accumulation_window: int = 100
    # §4 error-driven trigger
    tile: int = 16
    noise_floor_quantile: float = 0.75
    min_cluster_neighbours: int = 1
    insert_transmittance: float = 0.9
    rim_sigma: float = 1.0
    support_sigma: float = 3.0
    # §1 per-channel budgets, as a fraction of the remaining primitive budget per round
    clone_fraction: float = 0.4
    split_fraction: float = 0.4
    insert_fraction: float = 0.2
    max_gaussians: int = 100_000
    # §2/§3 operators
    clone_offset: float = 0.5  # delta = clone_offset * sigma_d
    clone_shrink_floor: float = 0.3
    split_offset: float = 0.5  # d = split_offset * sigma_d
    # §6 selective relocation instead of the global opacity reset
    relocate_opacity: float = 0.005
    relocate_strikes: int = 3
    floater_opacity: float = 0.5
    prune_scale_frac: float = 0.1
    insert_opacity: float = 0.1
    # Comparison levels (E12): switching these on restores vanilla semantics for that factor.
    # ``vanilla_operators`` restores the vanilla *shape* semantics -- coincident clone with
    # copied alpha, split scales divided by 1.6 -- but not the ``N(mu, Sigma)`` position draw,
    # which is ADR-YYYY mechanism 1. The draw is measured by the ``classic`` density strategy,
    # which is literally the incumbent code path rather than a re-implementation of it (A5), so
    # this flag is a within-controller attribution control and not the E12 vanilla arm.
    vanilla_operators: bool = False
    gradient_trigger: bool = False
    opacity_reset_every: int = 0
    opacity_reset_value: float = 0.011

    def __post_init__(self) -> None:
        for name in (
            "start_iter",
            "every",
            "accumulation_window",
            "tile",
            "relocate_strikes",
            "max_gaussians",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("clone_fraction", "split_fraction", "insert_fraction"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        if not 0.0 < self.noise_floor_quantile < 1.0:
            raise ValueError("noise_floor_quantile must lie in (0,1)")
        if not 0.0 < self.insert_transmittance <= 1.0:
            raise ValueError("insert_transmittance must lie in (0,1]")
        for name in ("clone_offset", "split_offset", "clone_shrink_floor", "rim_sigma"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass
class _ViewResidual:
    """Per-view tile accumulators over the current window."""

    residual: torch.Tensor  # (Ty,Tx) mean absolute residual
    alpha: torch.Tensor  # (Ty,Tx) mean rendered alpha
    depth: float  # median rendered depth where alpha > 0.5, or nan
    camera: Camera
    count: int = 0
    color: torch.Tensor | None = None  # (Ty,Tx,3) mean target colour, for INSERT appearance


@dataclass
class InitDensityStats:
    """One densification round, recorded whether or not it grew anything."""

    iteration: int
    n_before: int
    n_after: int
    cloned: int
    split: int
    inserted: int
    relocated: int
    pruned_floaters: int
    pruned_oversize: int
    structured_tiles: int
    noise_floor: float
    residual_axis_angle_deg_median: float
    channel_budgets: dict = field(default_factory=dict)


class InitPreservingController:
    """Error-driven, appearance-preserving density control with three lineage-aware channels.

    The controller owns no learning-rate state; the trust schedule (ADR-YYYY §5) is a separate
    object which this controller keeps in sync through :meth:`TrustSchedule.reindex`.
    """

    def __init__(
        self,
        config: InitPreservingConfig,
        n_gaussians: int,
        scene_extent: float,
        *,
        device: torch.device | str = "cpu",
    ):
        self.cfg = config
        self.extent = float(scene_extent)
        self.device = torch.device(device)
        self.views: dict[int, _ViewResidual] = {}
        self.low_opacity_strikes = torch.zeros(n_gaussians, dtype=torch.int64, device=self.device)
        self.visible_count = torch.zeros(n_gaussians, dtype=torch.int64, device=self.device)
        self.grad_accum = torch.zeros(n_gaussians, device=self.device)
        self.grad_count = torch.zeros(n_gaussians, device=self.device)
        self.stats: list[dict] = []

    # -- accumulation ----------------------------------------------------------------------

    def accumulate(
        self,
        *,
        view: int,
        camera: Camera,
        prediction: torch.Tensor,
        target: torch.Tensor,
        alpha: torch.Tensor,
        depth: torch.Tensor | None = None,
        visible: torch.Tensor | None = None,
    ) -> None:
        """Fold one training render into the per-tile residual map (ADR-YYYY §4).

        Held-out views are never passed in by the trainer; the accumulator has no way to see
        them and no code path that would.
        """
        cfg = self.cfg
        with torch.no_grad():
            residual = (prediction.detach() - target.detach()).abs().mean(dim=-1)
            if alpha.shape[:2] != residual.shape:
                # A render/loss pyramid step whose resolutions differ carries no aligned
                # residual/alpha pair; nothing is accumulated rather than mis-registered.
                return
            tiles = _tile_mean(residual, cfg.tile)
            alpha_tiles = _tile_mean(alpha.detach().reshape(residual.shape), cfg.tile)
            colour_tiles = _tile_mean_rgb(target.detach(), cfg.tile)
            entry = self.views.get(view)
            if entry is None or entry.residual.shape != tiles.shape:
                entry = _ViewResidual(
                    residual=torch.zeros_like(tiles),
                    alpha=torch.zeros_like(alpha_tiles),
                    depth=float("nan"),
                    camera=camera,
                    color=torch.zeros_like(colour_tiles),
                )
                self.views[view] = entry
            weight = 1.0 / float(entry.count + 1)
            entry.residual.mul_(1.0 - weight).add_(tiles, alpha=weight)
            entry.alpha.mul_(1.0 - weight).add_(alpha_tiles, alpha=weight)
            entry.color.mul_(1.0 - weight).add_(colour_tiles, alpha=weight)
            entry.camera = camera
            entry.count += 1
            if depth is not None:
                solid = alpha.detach().reshape(-1) > 0.5
                if bool(solid.any()):
                    entry.depth = float(depth.detach().reshape(-1)[solid].median())
            if visible is not None and visible.numel():
                index = visible.to(self.visible_count.device)
                self.visible_count.index_add_(0, index, torch.ones_like(index))

    def accumulate_gradients(self, out, width: int, height: int) -> None:
        """Vanilla screen-space gradient statistic, retained only as the E12 S3 comparison."""
        if out.means2d is None or out.visible is None:
            return
        grad = out.means2d.grad
        if grad is None:
            return
        if grad.ndim == 3:
            grad = grad[0]
        visible = out.visible.to(self.grad_accum.device)
        rows = grad if grad.shape[0] == visible.shape[0] else grad[visible]
        norm = rows.norm(dim=-1) * (max(width, height) * 0.5)
        self.grad_accum.index_add_(0, visible, norm.to(self.grad_accum))
        self.grad_count.index_add_(0, visible, torch.ones_like(norm, device=self.grad_count.device))

    def _reset_window(self) -> None:
        self.views = {}
        self.grad_accum.zero_()
        self.grad_count.zero_()
        self.visible_count.zero_()

    # -- the round -------------------------------------------------------------------------

    def step(
        self,
        iteration: int,
        params: dict[str, torch.Tensor],
        optimizers: dict[str, torch.optim.Optimizer],
        *,
        trust: TrustSchedule | None = None,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        scheduled = cfg.start_iter <= iteration <= cfg.stop_iter and iteration % cfg.every == 0
        if not scheduled or not self.views:
            return params

        n = int(params["means"].shape[0])
        opacity = torch.sigmoid(params["opacities"].detach()).reshape(-1)
        sigma = params["scales"].detach().exp()

        floor, structured = self._structured_tiles()
        clone_rows, split_rows, insert_sites, angles = self._select_candidates(
            params, structured, floor
        )
        relocate_rows, floater_rows, oversize_rows = self._select_removals(opacity, sigma)

        budget = max(cfg.max_gaussians - n, 0)
        clone_cap = int(budget * cfg.clone_fraction)
        split_cap = int(budget * cfg.split_fraction)
        insert_cap = int(budget * cfg.insert_fraction) + int(relocate_rows.numel())
        clone_rows = clone_rows[:clone_cap]
        split_rows = split_rows[:split_cap]
        insert_sites = _truncate_sites(insert_sites, insert_cap)

        new_params, parent_rows, keep_mask, applied = self._apply(
            params,
            optimizers,
            clone_rows,
            split_rows,
            insert_sites,
            floater_rows,
            oversize_rows,
            relocate_rows,
        )
        if trust is not None:
            trust.reindex(keep_mask, parent_rows)
        n_after = int(new_params["means"].shape[0])
        if n_after > cfg.max_gaussians:
            raise RuntimeError(f"init-preserving control exceeded max_gaussians: {n_after}")

        if cfg.opacity_reset_every and iteration % cfg.opacity_reset_every == 0:
            with torch.no_grad():
                cap = torch.logit(new_params["opacities"].new_tensor(cfg.opacity_reset_value))
                new_params["opacities"].clamp_max_(cap)

        record = InitDensityStats(
            iteration=iteration,
            n_before=n,
            n_after=n_after,
            cloned=int(clone_rows.numel()),
            split=int(split_rows.numel()),
            inserted=int(applied["inserted"]),
            relocated=int(applied["relocated"]),
            pruned_floaters=int(floater_rows.numel()),
            pruned_oversize=int(oversize_rows.numel()),
            structured_tiles=int(sum(int(mask.sum()) for mask in structured.values())),
            noise_floor=float(floor),
            residual_axis_angle_deg_median=float(angles.median()) if angles.numel() else 0.0,
            channel_budgets={"clone": clone_cap, "split": split_cap, "insert": insert_cap},
        )
        self.stats.append(record.__dict__)

        self.low_opacity_strikes = _resize_int(self.low_opacity_strikes, keep_mask, n_after)
        self.visible_count = torch.zeros(n_after, dtype=torch.int64, device=self.device)
        self.grad_accum = torch.zeros(n_after, device=self.device)
        self.grad_count = torch.zeros(n_after, device=self.device)
        self._reset_window()
        return new_params

    # -- §4 trigger ------------------------------------------------------------------------

    def _structured_tiles(self) -> tuple[float, dict[int, torch.Tensor]]:
        """Tiles above the noise floor whose above-floor neighbours make them *structured*."""
        cfg = self.cfg
        pooled = torch.cat([entry.residual.reshape(-1) for entry in self.views.values()])
        floor = float(pooled.quantile(cfg.noise_floor_quantile))
        structured: dict[int, torch.Tensor] = {}
        for view, entry in self.views.items():
            above = entry.residual > floor
            neighbours = _neighbour_count(above)
            structured[view] = above & (neighbours >= cfg.min_cluster_neighbours)
        return floor, structured

    def _select_candidates(
        self,
        params: dict[str, torch.Tensor],
        structured: dict[int, torch.Tensor],
        floor: float,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict], torch.Tensor]:
        """Assign structured residual to CLONE (rim), SPLIT (interior), or INSERT (empty).

        The residual direction is accumulated in each primitive's **own tangent-frame
        coordinates**, not in image space: an image-space vector cannot be summed across
        cameras, and the operators need a direction expressed in the frame they inherit.
        """
        from rtgs.core.gaussians3d import Gaussians3D, quat_to_rotmat
        from rtgs.render.projection import project_gaussians_ewa

        cfg = self.cfg
        n = int(params["means"].shape[0])
        device = params["means"].device
        clone_score = torch.zeros(n, device=device)
        split_score = torch.zeros(n, device=device)
        clone_dir = torch.zeros(n, 2, device=device)
        split_dir = torch.zeros(n, 2, device=device)
        insert_sites: list[dict] = []

        probe = Gaussians3D(
            means=params["means"].detach(),
            quats=params["quats"].detach(),
            log_scales=params["scales"].detach(),
            opacity=torch.sigmoid(params["opacities"].detach()).reshape(-1),
            sh=params["sh0"].detach(),
        )
        rotations = quat_to_rotmat(params["quats"].detach())
        for view, mask in structured.items():
            entry = self.views[view]
            if not bool(mask.any()):
                continue
            camera = entry.camera.to(device)
            rows, cols = mask.nonzero(as_tuple=True)
            centre = torch.stack(
                [
                    (cols.to(torch.float32) + 0.5) * cfg.tile,
                    (rows.to(torch.float32) + 0.5) * cfg.tile,
                ],
                dim=-1,
            ).to(device)
            weight = (entry.residual[rows, cols] - floor).clamp_min(0.0).to(device)
            empty = entry.alpha[rows, cols].to(device) <= (1.0 - cfg.insert_transmittance)

            projection = project_gaussians_ewa(probe, camera, dilation=0.0)
            mean2d = projection.means2d
            cov2d = projection.covariances2d
            depth = projection.depth
            radius = torch.stack(
                [cov2d[:, 0, 0].clamp_min(0.0).sqrt(), cov2d[:, 1, 1].clamp_min(0.0).sqrt()],
                dim=-1,
            )
            alive = (depth > 0) & (radius.amax(dim=-1) > 0)
            if bool(alive.any()) and bool((~empty).any()):
                index = alive.nonzero(as_tuple=True)[0]
                # Tile assignment in each primitive's own sigma units: a tile belongs to the
                # nearest primitive whose footprint reaches it.
                delta = centre[~empty][:, None, :] - mean2d[index][None, :, :]
                units = (delta / radius[index][None, :, :].clamp_min(_EPS)).norm(dim=-1)
                best = units.argmin(dim=-1)
                distance = units.gather(1, best[:, None]).squeeze(1)
                reachable = distance <= cfg.support_sigma
                owner = index[best][reachable]
                offset = delta[torch.arange(delta.shape[0], device=device), best][reachable]
                score = weight[~empty][reachable]
                interior = distance[reachable] <= cfg.rim_sigma
                tangent = _image_offset_to_tangent(offset, owner, depth, rotations, camera)
                for target_score, target_dir, selector in (
                    (split_score, split_dir, interior),
                    (clone_score, clone_dir, ~interior),
                ):
                    if not bool(selector.any()):
                        continue
                    target_score.index_add_(0, owner[selector], score[selector])
                    target_dir.index_add_(
                        0, owner[selector], tangent[selector] * score[selector][:, None]
                    )
            if bool(empty.any()) and math.isfinite(entry.depth):
                order = torch.argsort(weight[empty], descending=True)
                insert_sites.append(
                    {
                        "camera": camera,
                        "uv": centre[empty][order],
                        "depth": entry.depth,
                        "color": entry.color[rows[empty], cols[empty]][order],
                        "score": weight[empty][order],
                    }
                )

        clone_rows = clone_score.nonzero(as_tuple=True)[0]
        clone_rows = clone_rows[torch.argsort(clone_score[clone_rows], descending=True)]
        split_rows = split_score.nonzero(as_tuple=True)[0]
        split_rows = split_rows[torch.argsort(split_score[split_rows], descending=True)]
        # A primitive that is both a rim and an interior candidate refines rather than extends:
        # the finer channel wins so the two budgets never double-count one primitive.
        clone_rows = clone_rows[~torch.isin(clone_rows, split_rows)]

        if cfg.gradient_trigger:
            average = self.grad_accum / self.grad_count.clamp_min(1.0)
            candidates = (average > 2e-4).nonzero(as_tuple=True)[0]
            large = params["scales"].detach().exp().amax(dim=-1) > 0.01 * self.extent
            clone_rows = candidates[~large[candidates]]
            split_rows = candidates[large[candidates]]

        self._clone_dir, self._split_dir = clone_dir, split_dir
        angles = _axis_quantization_angles(params, clone_rows, split_rows, clone_dir, split_dir)
        return clone_rows, split_rows, insert_sites, angles

    # -- §6 removals -----------------------------------------------------------------------

    def _select_removals(
        self, opacity: torch.Tensor, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        low = opacity < cfg.relocate_opacity
        self.low_opacity_strikes = torch.where(
            low, self.low_opacity_strikes + 1, torch.zeros_like(self.low_opacity_strikes)
        )
        relocate = (self.low_opacity_strikes >= cfg.relocate_strikes).nonzero(as_tuple=True)[0]
        # Floaters: high opacity with no support in any training view over the window.  They are
        # pruned directly instead of reset-and-hoped-away.
        floaters = ((opacity >= cfg.floater_opacity) & (self.visible_count == 0)).nonzero(
            as_tuple=True
        )[0]
        oversize = (sigma.amax(dim=-1) > cfg.prune_scale_frac * self.extent).nonzero(as_tuple=True)[
            0
        ]
        floaters = floaters[~torch.isin(floaters, relocate)]
        oversize = oversize[~torch.isin(oversize, relocate) & ~torch.isin(oversize, floaters)]
        return relocate, floaters, oversize

    # -- operators -------------------------------------------------------------------------

    def _apply(
        self,
        params: dict[str, torch.Tensor],
        optimizers: dict[str, torch.optim.Optimizer],
        clone_rows: torch.Tensor,
        split_rows: torch.Tensor,
        insert_sites: list[dict],
        floater_rows: torch.Tensor,
        oversize_rows: torch.Tensor,
        relocate_rows: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, int]]:
        cfg = self.cfg
        n = int(params["means"].shape[0])
        device = params["means"].device

        remove = torch.zeros(n, dtype=torch.bool, device=device)
        remove[floater_rows] = True
        remove[oversize_rows] = True
        # §6 relocation is destruction plus creation (the M3 lineage rule), so a relocated row
        # leaves here and comes back through the INSERT channel with ``NULL_ROOT``.
        remove[relocate_rows] = True
        remove[split_rows] = True  # split replaces the parent by two children
        clone_rows = clone_rows[~remove[clone_rows]]

        extras: list[dict[str, torch.Tensor]] = []
        parents: list[torch.Tensor] = []

        if clone_rows.numel():
            parent, child = _appearance_preserving_clone(params, clone_rows, self._clone_dir, cfg)
            with torch.no_grad():
                for name, value in parent.items():
                    params[name].detach()[clone_rows] = value
            extras.append(child)
            parents.append(clone_rows)
        if split_rows.numel():
            children = _appearance_preserving_split(params, split_rows, self._split_dir, cfg)
            for child in children:
                extras.append(child)
                parents.append(split_rows)

        inserted = 0
        if insert_sites:
            child = _insert_primitives(params, insert_sites, cfg)
            if child is not None:
                inserted = int(child["means"].shape[0])
                extras.append(child)
                parents.append(torch.full((inserted,), -1, dtype=torch.int64, device=device))

        keep_mask = ~remove
        new_params = _edit_params(optimizers, params, keep_mask, extras)
        parent_rows = _concat_parents(parents, device)
        relocated = min(int(relocate_rows.numel()), inserted)
        return new_params, parent_rows, keep_mask, {"inserted": inserted, "relocated": relocated}


# --------------------------------------------------------------------------------------------
# Operators (ADR-YYYY §2/§3)
# --------------------------------------------------------------------------------------------


def _image_offset_to_tangent(
    offset: torch.Tensor,
    owner: torch.Tensor,
    depth: torch.Tensor,
    rotations: torch.Tensor,
    camera: Camera,
) -> torch.Tensor:
    """Express an image-space offset in the owning primitive's ``(t1, t2)`` tangent coordinates.

    A displacement of ``du`` pixels at depth ``z`` is a world displacement
    ``R^T [du_x z / fx, du_y z / fy, 0]`` in the camera's own image plane; projecting that onto
    the primitive's two tangential axes gives a view-independent 2-vector that *can* be summed
    across cameras, which an image-space vector cannot.
    """
    rotation = camera.R.to(offset)
    world_x, world_y = rotation[0, :], rotation[1, :]
    z = depth[owner].to(offset)[:, None]
    displacement = offset[:, 0:1] * world_x[None, :] * z / float(camera.fx) + offset[
        :, 1:2
    ] * world_y[None, :] * z / float(camera.fy)
    frame = rotations[owner].to(offset)
    return torch.stack(
        [
            (displacement * frame[:, :, 0]).sum(-1),
            (displacement * frame[:, :, 1]).sum(-1),
        ],
        dim=-1,
    )


def _dominant_axis(
    params: dict[str, torch.Tensor], rows: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    """Index (0 or 1) of the tangential axis the accumulated residual direction favours.

    ``direction`` is the ADR's continuous in-plane ``d``, in ``(t1, t2)`` coordinates.  Rows
    with no directional evidence fall back to the larger tangential axis, which is the ADR's
    ``NULL_ROOT`` rule ("the residual axis falls back to the largest covariance axis").
    """
    scales = params["scales"].detach()[rows].exp()
    component = direction[rows].abs()
    larger = (scales[:, 1] > scales[:, 0]).to(torch.int64)
    favoured = (component[:, 1] > component[:, 0]).to(torch.int64)
    return torch.where(component.amax(dim=-1) > 0, favoured, larger)


def _quantization_angle_deg(direction: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Angle between the continuous in-plane residual direction and the axis it was snapped to."""
    norm = direction.norm(dim=-1)
    rows = torch.arange(direction.shape[0], device=direction.device)
    cosine = (direction[rows, axis].abs() / norm.clamp_min(_EPS)).clamp(0.0, 1.0)
    angle = torch.rad2deg(torch.arccos(cosine))
    return torch.where(norm > 0, angle, torch.zeros_like(angle))


def _axis_shrink(
    log_scales: torch.Tensor, axis: torch.Tensor, factor: torch.Tensor
) -> torch.Tensor:
    """Multiply exactly one stored axis by ``factor`` (exact in log space, A2)."""
    out = log_scales.clone()
    rows = torch.arange(out.shape[0], device=out.device)
    out[rows, axis] = out[rows, axis] + factor.log()
    return out


def _appearance_preserving_clone(
    params: dict[str, torch.Tensor],
    rows: torch.Tensor,
    direction2d: torch.Tensor,
    cfg: InitPreservingConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """§2: extend a surface by a deterministic tangential step; both halves stay appearance-matched.

    Rejected alternative: the vanilla clone at the identical position with copied ``alpha``.
    Coincident twins have identical gradients and separate only by numerical noise while
    locally doubling opacity — the repair that follows is what the opacity reset papers over.
    """
    from rtgs.core.gaussians3d import quat_to_rotmat

    parent = {name: value.detach()[rows].clone() for name, value in params.items()}
    if cfg.vanilla_operators:
        child = {name: value.clone() for name, value in parent.items()}
        return {}, child

    axis = _dominant_axis(params, rows, direction2d)
    scales = parent["scales"].exp()
    index = torch.arange(rows.numel(), device=rows.device)
    sigma_d = scales[index, axis]
    delta = cfg.clone_offset * sigma_d
    rotation = quat_to_rotmat(parent["quats"])
    unit = rotation[index, :, axis]  # column ``axis`` is that tangential direction in world

    shrunk = torch.sqrt(
        torch.maximum(
            sigma_d.square() - (0.5 * delta).square(), (cfg.clone_shrink_floor * sigma_d).square()
        )
    )
    factor = (shrunk / sigma_d.clamp_min(_EPS)).clamp_min(_EPS)
    log_scales = _axis_shrink(parent["scales"], axis, factor)
    opacity = torch.sigmoid(parent["opacities"])
    matched = 1.0 - torch.sqrt((1.0 - opacity).clamp_min(1e-9))
    logit = torch.logit(matched.clamp(1e-6, 1.0 - 1e-6))

    parent_update = {
        "scales": log_scales,
        "opacities": logit,
    }
    child = {name: value.clone() for name, value in parent.items()}
    child["means"] = parent["means"] + delta[:, None] * unit
    child["scales"] = log_scales.clone()
    child["opacities"] = logit.clone()
    return parent_update, child


def _appearance_preserving_split(
    params: dict[str, torch.Tensor],
    rows: torch.Tensor,
    direction2d: torch.Tensor,
    cfg: InitPreservingConfig,
) -> list[dict[str, torch.Tensor]]:
    """§3: refine a surface by two deterministic offsets; never a draw from ``N(mu, Sigma)``.

    ``sigma_d' = sqrt(sigma_d^2 - d^2)`` preserves the mixture second moment exactly and
    replaces the ad-hoc ``/1.6``, which over-shrinks and, combined with sampling, is mechanism
    1+2 of the ADR context section.
    """
    from rtgs.core.gaussians3d import quat_to_rotmat

    base = {name: value.detach()[rows].clone() for name, value in params.items()}
    if cfg.vanilla_operators:
        children = []
        for _ in range(2):
            child = {name: value.clone() for name, value in base.items()}
            child["scales"] = child["scales"] - math.log(1.6)
            children.append(child)
        return children

    axis = _dominant_axis(params, rows, direction2d)
    index = torch.arange(rows.numel(), device=rows.device)
    scales = base["scales"].exp()
    sigma_d = scales[index, axis]
    offset = cfg.split_offset * sigma_d
    rotation = quat_to_rotmat(base["quats"])
    unit = rotation[index, :, axis]

    shrunk = torch.sqrt((sigma_d.square() - offset.square()).clamp_min(_EPS))
    factor = (shrunk / sigma_d.clamp_min(_EPS)).clamp_min(_EPS)
    log_scales = _axis_shrink(base["scales"], axis, factor)
    opacity = torch.sigmoid(base["opacities"])
    matched = 1.0 - torch.sqrt((1.0 - opacity).clamp_min(1e-9))
    logit = torch.logit(matched.clamp(1e-6, 1.0 - 1e-6))

    children = []
    for sign in (1.0, -1.0):
        child = {name: value.clone() for name, value in base.items()}
        child["means"] = base["means"] + sign * offset[:, None] * unit
        child["scales"] = log_scales.clone()
        child["opacities"] = logit.clone()
        children.append(child)
    return children


def _insert_primitives(
    params: dict[str, torch.Tensor],
    sites: list[dict],
    cfg: InitPreservingConfig,
) -> dict[str, torch.Tensor] | None:
    """§1 INSERT: new capacity where transmittance says nothing is there to grow from.

    Depth comes from the view's median rendered surface depth: a ray whose transmittance is
    ~1 carries no depth of its own, so the scene's own depth scale in that camera is the only
    image-free estimate available, and the primitive is free to move from there.
    """
    device = params["means"].device
    dtype = params["means"].dtype
    means, colours = [], []
    for site in sites:
        camera = site["camera"].to(device)
        uv = site["uv"].reshape(-1, 2).to(device=device, dtype=torch.float32)
        if uv.shape[0] == 0:
            continue
        depth = torch.full((uv.shape[0],), float(site["depth"]), device=device)
        means.append(camera.unproject(uv, depth))
        colours.append(site["color"].reshape(-1, 3).to(device=device, dtype=torch.float32))
    if not means:
        return None
    means_tensor = torch.cat(means).to(dtype)
    colour_tensor = torch.cat(colours).to(dtype)
    count = means_tensor.shape[0]

    from rtgs.core.sh import rgb_to_sh

    sigma = params["scales"].detach().exp().median()
    child = {
        "means": means_tensor,
        "quats": torch.zeros((count, 4), device=device, dtype=params["quats"].dtype),
        "scales": torch.full(
            (count, 3), float(sigma.log()), device=device, dtype=params["scales"].dtype
        ),
        "opacities": torch.full(
            (count,) if params["opacities"].ndim == 1 else (count, 1),
            float(torch.logit(torch.tensor(cfg.insert_opacity))),
            device=device,
            dtype=params["opacities"].dtype,
        ),
        "sh0": torch.zeros(
            (count, *params["sh0"].shape[1:]), device=device, dtype=params["sh0"].dtype
        ),
        "shN": torch.zeros(
            (count, *params["shN"].shape[1:]), device=device, dtype=params["shN"].dtype
        ),
    }
    child["quats"][:, 0] = 1.0
    child["sh0"][:, 0] = rgb_to_sh(colour_tensor.clamp(0.0, 1.0)).to(params["sh0"].dtype)
    return child


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def _tile_mean(image: torch.Tensor, tile: int) -> torch.Tensor:
    height, width = image.shape[:2]
    ty, tx = math.ceil(height / tile), math.ceil(width / tile)
    padded = torch.zeros(ty * tile, tx * tile, device=image.device, dtype=image.dtype)
    counts = torch.zeros_like(padded)
    padded[:height, :width] = image
    counts[:height, :width] = 1.0
    total = padded.reshape(ty, tile, tx, tile).sum(dim=(1, 3))
    weight = counts.reshape(ty, tile, tx, tile).sum(dim=(1, 3)).clamp_min(1.0)
    return total / weight


def _tile_mean_rgb(image: torch.Tensor, tile: int) -> torch.Tensor:
    return torch.stack([_tile_mean(image[..., c], tile) for c in range(image.shape[-1])], dim=-1)


def _neighbour_count(mask: torch.Tensor) -> torch.Tensor:
    """Count of 4-connected neighbours that are also above the floor (the 'clustered' test)."""
    padded = torch.zeros(
        mask.shape[0] + 2, mask.shape[1] + 2, dtype=torch.int64, device=mask.device
    )
    padded[1:-1, 1:-1] = mask.to(torch.int64)
    return padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]


def _axis_quantization_angles(
    params: dict[str, torch.Tensor],
    clone_rows: torch.Tensor,
    split_rows: torch.Tensor,
    clone_dir: torch.Tensor,
    split_dir: torch.Tensor,
) -> torch.Tensor:
    """Per-event angle between the ADR's continuous ``d`` and the axis it was snapped to.

    This is the quantity that bounds the cost of the one documented deviation from ADR-YYYY
    (module docstring); a median near zero means the quantization changed almost nothing, and a
    median near 45 degrees means it discarded most of the directional evidence.
    """
    parts = []
    for rows, direction in ((clone_rows, clone_dir), (split_rows, split_dir)):
        if not rows.numel():
            continue
        axis = _dominant_axis(params, rows, direction)
        parts.append(_quantization_angle_deg(direction[rows], axis))
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


def _truncate_sites(sites: list[dict], cap: int) -> list[dict]:
    """Keep the highest-scoring insertion sites across views, up to the channel budget."""
    if cap <= 0 or not sites:
        return []
    remaining = cap
    out: list[dict] = []
    for site in sorted(sites, key=lambda entry: -float(entry["score"][0])):
        if remaining <= 0:
            break
        take = min(remaining, int(site["uv"].shape[0]))
        out.append(
            {
                "camera": site["camera"],
                "uv": site["uv"][:take],
                "depth": site["depth"],
                "color": site["color"][:take],
                "score": site["score"][:take],
            }
        )
        remaining -= take
    return out


def _concat_parents(parents: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Parent rows for each appended child, in append order, in **pre-surgery** indexing.

    Pre-surgery is the indexing :meth:`TrustSchedule.reindex` needs: a split parent is removed
    by its own event, so a post-surgery index for it would not exist and its children would
    silently lose the lineage the ADR requires them to inherit.
    """
    if not parents:
        return torch.zeros(0, dtype=torch.int64, device=device)
    return torch.cat([rows.to(device=device, dtype=torch.int64) for rows in parents])


def _resize_int(buffer: torch.Tensor, keep_mask: torch.Tensor, n_after: int) -> torch.Tensor:
    kept = buffer[keep_mask]
    if kept.numel() >= n_after:
        return kept[:n_after]
    pad = torch.zeros(n_after - kept.numel(), dtype=buffer.dtype, device=buffer.device)
    return torch.cat([kept, pad])


__all__ = [
    "NULL_ROOT",
    "InitDensityStats",
    "InitPreservingConfig",
    "InitPreservingController",
]
