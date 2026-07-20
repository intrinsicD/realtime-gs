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

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from rtgs.render.base import RenderOutput


@dataclass
class DensityConfig:
    """Density-control hyperparameters (iteration numbers are trainer iterations)."""

    start_iter: int = 60
    stop_iter: int = 10_000_000
    every: int = 40
    grad_threshold: float = 2e-4
    absgrad: bool = False
    # Split (instead of clone) when the max scale exceeds this fraction of scene extent.
    split_scale_frac: float = 0.01
    split_factor: float = 1.6
    prune_opacity: float = 0.005
    # Prune when the max scale exceeds this fraction of scene extent.
    prune_scale_frac: float = 0.1
    max_gaussians: int = 100_000
    opacity_reset_every: int = 3_000
    opacity_reset_value: float = 0.011
    revised_opacity: bool = True
    mcmc_noise_lr: float = 500_000.0


@dataclass(frozen=True)
class SelectedBirthNewborn:
    """Immutable physical-row lineage for one explicitly selected newborn."""

    new_row: int
    parent_row: int
    operator: Literal["clone", "split"]
    child_ordinal: int


@dataclass(frozen=True)
class SelectedBirthReceipt:
    """Immutable evidence returned by :func:`apply_selected_birth_surgery`.

    Raw split normals are represented as nested tuples so the frozen receipt contains no
    mutable tensor aliases.  ``raw_split_dtype`` and ``raw_split_shape`` make their exact
    tensor representation explicit, while the hashes bind dtype, shape, and native bytes.
    """

    schema: str
    n_before: int
    n_after: int
    net_growth: int
    max_gaussians: int
    scale_boundary: float
    split_scale_frac: float
    split_factor: float
    revised_opacity: bool
    selected_parent_rows: tuple[int, ...]
    clone_parent_rows: tuple[int, ...]
    split_parent_rows: tuple[int, ...]
    survivor_old_rows: tuple[int, ...]
    removed_parent_rows: tuple[int, ...]
    old_row_to_new_row: tuple[int, ...]
    new_row_to_old_row: tuple[int, ...]
    clone_new_rows: tuple[int, ...]
    split_child0_new_rows: tuple[int, ...]
    split_child1_new_rows: tuple[int, ...]
    newborns: tuple[SelectedBirthNewborn, ...]
    raw_split_dtype: str
    raw_split_shape: tuple[int, int]
    raw_split_child0_standard_normals: tuple[tuple[float, float, float], ...]
    raw_split_child1_standard_normals: tuple[tuple[float, float, float], ...]
    raw_split_child0_sha256: str
    raw_split_child1_sha256: str
    generator_state_before_sha256: str
    generator_state_after_sha256: str


class DensityController:
    """Accumulates densification statistics and performs param/optimizer surgery."""

    def __init__(
        self,
        config: DensityConfig,
        n_gaussians: int,
        scene_extent: float,
        device: torch.device | str = "cpu",
    ):
        if config.every <= 0:
            raise ValueError("density every must be positive")
        if config.max_gaussians <= 0:
            raise ValueError("density max_gaussians must be positive")
        self.cfg = config
        self.extent = scene_extent
        self.grad_accum = torch.zeros(n_gaussians, device=device)
        self.count = torch.zeros(n_gaussians, device=device)
        self.stats: list[dict] = []

    def accumulate(self, out: RenderOutput, width: int, height: int) -> None:
        """Record screen-space positional gradients after loss.backward()."""
        if out.means2d is None or out.visible is None:
            return
        grad = getattr(out.means2d, "absgrad", None) if self.cfg.absgrad else None
        grad = out.means2d.grad if grad is None else grad
        if grad is None:
            return
        if grad.ndim == 3:
            grad = grad[0]
        visible = out.visible.to(self.grad_accum.device)
        # torch_ref exposes only visible rows; gsplat exposes all N rows.
        visible_grad = grad if grad.shape[0] == visible.shape[0] else grad[visible]
        norm = visible_grad.norm(dim=-1) * (max(width, height) * 0.5)
        self.grad_accum.index_add_(0, visible, norm.to(self.grad_accum))
        self.count.index_add_(0, visible, torch.ones_like(norm, device=self.count.device))

    def step(
        self,
        iteration: int,
        params: dict[str, torch.Tensor],
        optimizer: torch.optim.Adam | dict[str, torch.optim.Optimizer],
        generator: torch.Generator | None = None,
        force_budget: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run one clone/split/prune (+optional opacity reset) round if scheduled."""
        cfg = self.cfg
        n = params["means"].shape[0]
        scheduled = cfg.start_iter <= iteration <= cfg.stop_iter and iteration % cfg.every == 0
        if not scheduled and not (force_budget and n > cfg.max_gaussians):
            return params

        avg_grad = self.grad_accum / self.count.clamp_min(1.0)
        scale_key = "scales" if "scales" in params else "log_scales"
        opacity_key = "opacities" if "opacities" in params else "opacity_logit"
        scales = params[scale_key].detach().exp()
        opacity = torch.sigmoid(params[opacity_key].detach())
        scale_max = scales.max(dim=-1).values

        densify = (avg_grad > cfg.grad_threshold) & (self.count > 0)
        is_large = scale_max > cfg.split_scale_frac * self.extent
        clone_mask = densify & ~is_large
        split_mask = densify & is_large
        prune_mask = (
            (opacity < cfg.prune_opacity) | (scale_max > cfg.prune_scale_frac * self.extent)
            if scheduled
            else torch.zeros_like(densify)
        )
        # Dense transferred initializations can already exceed the configured budget. Cull the
        # least significant remaining splats on the first scheduled round instead of preserving
        # an over-budget set forever.
        budget_excess = max(n - int(prune_mask.sum()) - cfg.max_gaussians, 0)
        if budget_excess:
            eligible = (~prune_mask).nonzero(as_tuple=True)[0]
            significance = opacity[eligible] * scales[eligible].prod(dim=-1)
            prune_mask[eligible[torch.topk(significance, budget_excess, largest=False).indices]] = (
                True
            )
        # A row selected for pruning must not also be cloned/split.  Counting such an overlap as
        # one unit of growth is wrong: after removing the parent, a split appends two children
        # and can transiently exceed the hard cap (and the intended VRAM bound).
        clone_mask &= ~prune_mask
        split_mask &= ~prune_mask
        # Each clone or split grows the surviving set by one. Respect the hard primitive
        # budget even when one density round has many candidates.
        growth_budget = max(cfg.max_gaussians - n + int(prune_mask.sum()), 0)
        candidates = (clone_mask | split_mask).nonzero(as_tuple=True)[0]
        if candidates.numel() > growth_budget:
            keep_candidates = (
                candidates[torch.topk(avg_grad[candidates], growth_budget).indices]
                if growth_budget
                else candidates[:0]
            )
            selected = torch.zeros_like(densify)
            selected[keep_candidates] = True
            clone_mask &= selected
            split_mask &= selected
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
                s = child[scale_key].exp()
                noise = (
                    torch.randn(s.shape, generator=generator, device=s.device, dtype=s.dtype) * s
                )
                child["means"] = child["means"] + (rot @ noise[..., None])[..., 0]
                child[scale_key] = child[scale_key] - torch.log(
                    child[scale_key].new_tensor(cfg.split_factor)
                )
                if cfg.revised_opacity:
                    child_opacity = 1.0 - torch.sqrt(
                        1.0 - torch.sigmoid(child[opacity_key]).clamp(max=1.0 - 1e-6)
                    )
                    child[opacity_key] = torch.logit(child_opacity.clamp_min(1e-6))
                extras.append(child)

        new_params = _edit_params(optimizer, params, keep_mask, extras)

        n_new = new_params["means"].shape[0]
        if n_new > cfg.max_gaussians:
            raise RuntimeError(
                f"density control violated max_gaussians: {n_new} > {cfg.max_gaussians}"
            )
        self.grad_accum = torch.zeros(n_new, device=params["means"].device)
        self.count = torch.zeros(n_new, device=params["means"].device)
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
                opacity_key = "opacities" if "opacities" in new_params else "opacity_logit"
                cap = torch.logit(new_params[opacity_key].new_tensor(cfg.opacity_reset_value))
                new_params[opacity_key].clamp_max_(cap)
        return new_params


def apply_selected_birth_surgery(
    params: dict[str, torch.Tensor],
    optimizer: torch.optim.Adam | dict[str, torch.optim.Optimizer],
    parent_rows: Sequence[int] | torch.Tensor,
    *,
    scene_extent: float,
    generator: torch.Generator,
    split_scale_frac: float = 0.01,
    split_factor: float = 1.6,
    revised_opacity: bool = True,
    max_gaussians: int = 100_000,
) -> tuple[dict[str, torch.Tensor], SelectedBirthReceipt]:
    """Apply one exact birth wave to an ordered, unique set of parent rows.

    Selected rows at or below the current scale boundary are cloned.  Rows above it are
    replaced by two split children.  Physical output order is survivors, clones in the
    caller's order, split child zero in the caller's order, then split child one in the
    caller's order.  The supplied generator is the only random source and is consumed by
    exactly two complete split-normal draws (child zero, then child one).

    This is an explicit research seam; the classic controller does not call it, so all
    established density-control defaults remain unchanged.
    """

    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be an isolated torch.Generator")
    if isinstance(max_gaussians, bool) or not isinstance(max_gaussians, int):
        raise TypeError("max_gaussians must be an integer")
    if max_gaussians <= 0:
        raise ValueError("max_gaussians must be positive")
    for name, value in (
        ("scene_extent", scene_extent),
        ("split_scale_frac", split_scale_frac),
        ("split_factor", split_factor),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real scalar")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not isinstance(revised_opacity, bool):
        raise TypeError("revised_opacity must be bool")

    n, scale_key, opacity_key = _validate_selected_birth_params(params)
    ordered_rows = _ordered_parent_rows(parent_rows, n)
    n_after = n + len(ordered_rows)
    if n_after > max_gaussians:
        raise ValueError(
            "selected birth surgery exceeds max_gaussians: "
            f"{n} + {len(ordered_rows)} > {max_gaussians}"
        )
    _validate_selected_birth_optimizer(optimizer, params)

    device = params["means"].device
    if torch.device(generator.device) != device:
        raise ValueError(
            "generator device must match selected-birth parameter device: "
            f"{generator.device} != {device}"
        )
    selected = torch.tensor(ordered_rows, dtype=torch.long, device=device)
    selected_scale_max = params[scale_key].detach()[selected].exp().max(dim=-1).values
    scale_boundary = float(split_scale_frac) * float(scene_extent)
    selected_is_split = selected_scale_max > scale_boundary
    clone_rows = selected[~selected_is_split]
    split_rows = selected[selected_is_split]
    clone_parent_rows = tuple(int(row) for row in clone_rows.detach().cpu().tolist())
    split_parent_rows = tuple(int(row) for row in split_rows.detach().cpu().tolist())

    split_set = set(split_parent_rows)
    survivor_old_rows = tuple(row for row in range(n) if row not in split_set)
    keep_mask = torch.ones(n, dtype=torch.bool, device=device)
    if split_rows.numel():
        keep_mask[split_rows] = False

    generator_state_before = generator.get_state().detach().clone()
    split_shape = (len(split_parent_rows), 3)
    raw_child0 = torch.randn(
        split_shape,
        generator=generator,
        device=device,
        dtype=params[scale_key].dtype,
    )
    raw_child1 = torch.randn(
        split_shape,
        generator=generator,
        device=device,
        dtype=params[scale_key].dtype,
    )
    generator_state_after = generator.get_state().detach().clone()

    extras: list[dict[str, torch.Tensor]] = []
    if clone_rows.numel():
        extras.append({name: value.detach()[clone_rows].clone() for name, value in params.items()})
    if split_rows.numel():
        split_base = {name: value.detach()[split_rows].clone() for name, value in params.items()}
        for raw_standard_normal in (raw_child0, raw_child1):
            child = {name: value.clone() for name, value in split_base.items()}
            from rtgs.core.gaussians3d import quat_to_rotmat

            rotation = quat_to_rotmat(child["quats"])
            native_scale = child[scale_key].exp()
            native_noise = raw_standard_normal * native_scale
            child["means"] = child["means"] + (rotation @ native_noise[..., None])[..., 0]
            child[scale_key] = child[scale_key] - torch.log(
                child[scale_key].new_tensor(split_factor)
            )
            if revised_opacity:
                child_opacity = 1.0 - torch.sqrt(
                    1.0 - torch.sigmoid(child[opacity_key]).clamp(max=1.0 - 1e-6)
                )
                child[opacity_key] = torch.logit(child_opacity.clamp_min(1e-6))
            extras.append(child)

    new_params = _edit_params(optimizer, params, keep_mask, extras)
    actual_n_after = int(new_params["means"].shape[0])
    if actual_n_after != n_after:
        raise RuntimeError(f"selected birth count invariant failed: {actual_n_after} != {n_after}")

    old_row_to_new_row = [-1] * n
    for new_row, old_row in enumerate(survivor_old_rows):
        old_row_to_new_row[old_row] = new_row
    clone_start = len(survivor_old_rows)
    split_child0_start = clone_start + len(clone_parent_rows)
    split_child1_start = split_child0_start + len(split_parent_rows)
    clone_new_rows = tuple(range(clone_start, split_child0_start))
    split_child0_new_rows = tuple(range(split_child0_start, split_child1_start))
    split_child1_new_rows = tuple(range(split_child1_start, n_after))
    new_row_to_old_row = (
        survivor_old_rows + clone_parent_rows + split_parent_rows + split_parent_rows
    )
    newborns = tuple(
        SelectedBirthNewborn(new_row, parent_row, "clone", 0)
        for new_row, parent_row in zip(clone_new_rows, clone_parent_rows, strict=True)
    ) + tuple(
        SelectedBirthNewborn(new_row, parent_row, "split", child_ordinal)
        for child_ordinal, new_rows in (
            (0, split_child0_new_rows),
            (1, split_child1_new_rows),
        )
        for new_row, parent_row in zip(new_rows, split_parent_rows, strict=True)
    )

    return new_params, SelectedBirthReceipt(
        schema="rtgs.selected_birth_receipt.v1",
        n_before=n,
        n_after=n_after,
        net_growth=len(ordered_rows),
        max_gaussians=max_gaussians,
        scale_boundary=scale_boundary,
        split_scale_frac=float(split_scale_frac),
        split_factor=float(split_factor),
        revised_opacity=revised_opacity,
        selected_parent_rows=ordered_rows,
        clone_parent_rows=clone_parent_rows,
        split_parent_rows=split_parent_rows,
        survivor_old_rows=survivor_old_rows,
        removed_parent_rows=split_parent_rows,
        old_row_to_new_row=tuple(old_row_to_new_row),
        new_row_to_old_row=new_row_to_old_row,
        clone_new_rows=clone_new_rows,
        split_child0_new_rows=split_child0_new_rows,
        split_child1_new_rows=split_child1_new_rows,
        newborns=newborns,
        raw_split_dtype=str(raw_child0.dtype),
        raw_split_shape=split_shape,
        raw_split_child0_standard_normals=_standard_normal_rows(raw_child0),
        raw_split_child1_standard_normals=_standard_normal_rows(raw_child1),
        raw_split_child0_sha256=_tensor_sha256(raw_child0),
        raw_split_child1_sha256=_tensor_sha256(raw_child1),
        generator_state_before_sha256=_tensor_sha256(generator_state_before),
        generator_state_after_sha256=_tensor_sha256(generator_state_after),
    )


def _validate_selected_birth_params(params: dict[str, torch.Tensor]) -> tuple[int, str, str]:
    if not isinstance(params, dict) or not params:
        raise ValueError("params must be a non-empty dictionary")
    for required in ("means", "quats"):
        if required not in params:
            raise ValueError(f"params is missing required field {required}")
    scale_keys = [name for name in ("scales", "log_scales") if name in params]
    opacity_keys = [name for name in ("opacities", "opacity_logit") if name in params]
    if len(scale_keys) != 1:
        raise ValueError("params must contain exactly one scales/log_scales field")
    if len(opacity_keys) != 1:
        raise ValueError("params must contain exactly one opacities/opacity_logit field")
    scale_key = scale_keys[0]
    opacity_key = opacity_keys[0]
    means = params["means"]
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape (N,3)")
    n = int(means.shape[0])
    if n <= 0:
        raise ValueError("selected birth surgery requires at least one Gaussian")
    if params["quats"].shape != (n, 4):
        raise ValueError("quats must have shape (N,4)")
    if params[scale_key].shape != (n, 3):
        raise ValueError(f"{scale_key} must have shape (N,3)")
    device = means.device
    for name, value in params.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"parameter {name} must be a tensor")
        if value.ndim == 0 or value.shape[0] != n:
            raise ValueError(f"parameter {name} has an inconsistent leading dimension")
        if value.device != device:
            raise ValueError("all selected-birth parameters must share one device")
        if not value.is_floating_point():
            raise TypeError(f"parameter {name} must be floating point")
        if not bool(torch.isfinite(value.detach()).all()):
            raise ValueError(f"parameter {name} contains non-finite values")
    return n, scale_key, opacity_key


def _ordered_parent_rows(
    parent_rows: Sequence[int] | torch.Tensor,
    n: int,
) -> tuple[int, ...]:
    if isinstance(parent_rows, torch.Tensor):
        if parent_rows.ndim != 1:
            raise ValueError("parent_rows must be a one-dimensional sequence")
        if parent_rows.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("parent_rows tensor must have integer dtype")
        rows = tuple(int(row) for row in parent_rows.detach().cpu().tolist())
    else:
        if isinstance(parent_rows, (str, bytes)) or not isinstance(parent_rows, Sequence):
            raise TypeError("parent_rows must be a sequence of integers")
        if any(isinstance(row, bool) or not isinstance(row, int) for row in parent_rows):
            raise TypeError("parent_rows must contain only integers")
        rows = tuple(parent_rows)
    if not rows:
        raise ValueError("parent_rows must be non-empty")
    if len(set(rows)) != len(rows):
        raise ValueError("parent_rows must be unique")
    if any(row < 0 or row >= n for row in rows):
        raise IndexError("parent_rows contains an out-of-range row")
    return rows


def _validate_selected_birth_optimizer(
    optimizer: torch.optim.Adam | dict[str, torch.optim.Optimizer],
    params: dict[str, torch.Tensor],
) -> None:
    if isinstance(optimizer, dict):
        if set(optimizer) != set(params):
            raise ValueError("optimizer dictionary keys must exactly match params")
        entries = [(name, optimizer[name], optimizer[name].param_groups) for name in params]
    elif isinstance(optimizer, torch.optim.Adam):
        entries = [("<joint>", optimizer, optimizer.param_groups)]
        group_names = [group.get("name") for group in optimizer.param_groups]
        if len(group_names) != len(set(group_names)) or set(group_names) != set(params):
            raise ValueError("Adam param-group names must exactly match params")
    else:
        raise TypeError("optimizer must be Adam or a dictionary of Adam optimizers")

    for entry_name, parameter_optimizer, groups in entries:
        if not isinstance(parameter_optimizer, torch.optim.Adam):
            raise TypeError(f"optimizer {entry_name} must be torch.optim.Adam")
        if isinstance(optimizer, dict):
            if len(groups) != 1 or groups[0].get("name") != entry_name:
                raise ValueError(f"optimizer {entry_name} must have one same-named parameter group")
            groups_to_check = groups
        else:
            groups_to_check = groups
        for group in groups_to_check:
            name = group.get("name")
            if name not in params:
                raise ValueError("optimizer group has an unknown or missing name")
            if len(group["params"]) != 1 or group["params"][0] is not params[name]:
                raise ValueError(f"optimizer group {name} does not reference params[{name!r}]")
            if bool(group.get("amsgrad", False)):
                raise ValueError("selected birth surgery requires amsgrad=False")
            parameter = params[name]
            state = parameter_optimizer.state.get(parameter)
            if not state:
                continue
            if "exp_avg" not in state or "exp_avg_sq" not in state:
                raise ValueError(f"optimizer state for {name} is incomplete")
            for moment_name in ("exp_avg", "exp_avg_sq"):
                if state[moment_name].shape != parameter.shape:
                    raise ValueError(f"optimizer {moment_name} for {name} has the wrong shape")


def _standard_normal_rows(
    tensor: torch.Tensor,
) -> tuple[tuple[float, float, float], ...]:
    rows = tensor.detach().cpu().tolist()
    return tuple((float(row[0]), float(row[1]), float(row[2])) for row in rows)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _edit_params(
    optimizer: torch.optim.Adam | dict[str, torch.optim.Optimizer],
    params: dict[str, torch.Tensor],
    keep_mask: torch.Tensor,
    extras: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Keep a subset of each param and append new rows, preserving Adam moments.

    Optimizer param groups must be registered one param per group, with the group's
    'name' matching the params dict key (the trainer guarantees this).
    """
    new_params: dict[str, torch.Tensor] = {}
    if isinstance(optimizer, dict):
        entries = [(name, optimizer[name], optimizer[name].param_groups[0]) for name in params]
    else:
        entries = [(group["name"], optimizer, group) for group in optimizer.param_groups]
    for name, parameter_optimizer, group in entries:
        old = params[name]
        rows = [old.detach()[keep_mask]] + [e[name] for e in extras]
        new = torch.cat(rows).requires_grad_(True)

        state = parameter_optimizer.state.pop(old, None)
        if state is not None:
            for key in ("exp_avg", "exp_avg_sq"):
                buf = state[key]
                pads = [
                    torch.zeros(
                        (e[name].shape[0], *buf.shape[1:]),
                        dtype=buf.dtype,
                        device=buf.device,
                    )
                    for e in extras
                ]
                state[key] = torch.cat([buf[keep_mask]] + pads)
            parameter_optimizer.state[new] = state
        group["params"] = [new]
        new_params[name] = new
    return new_params
