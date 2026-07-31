"""Classic 3DGS clone/split/prune control for compact-field supervision.

The compact trainer owns the sampled RGB-free loss while :class:`DensityController` owns the
established topology surgery.  This adapter joins the two through
``CompactTopologyController`` without importing ``SceneData`` or the dense RGB trainer.
"""

from __future__ import annotations

from dataclasses import asdict

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.optim.density import DensityConfig, DensityController
from rtgs.render.point_base import PointRenderOutput

_GROUP_ORDER = ("means", "quats", "scales", "opacities", "sh0", "shN")


class ClassicCompactDensityController:
    """Drive classic screen-gradient density control from compact point renders.

    The adapter deliberately delegates all parameter and Adam-state surgery to the same
    :class:`~rtgs.optim.density.DensityController` used by the reference dense trainer.  It adds
    only persistent identities and lineage required by the compact trainer's audited topology
    boundary.
    """

    def __init__(
        self,
        config: DensityConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.config = config or DensityConfig()
        self.seed = seed
        self._density: DensityController | None = None
        self._generator: torch.Generator | None = None
        self._persistent_ids: torch.Tensor | None = None
        self._initial_count: int | None = None
        self._next_id = 0
        self._lineage_by_id: dict[int, dict[str, int | str]] = {}
        self._events: list[dict[str, int | bool]] = []

    @property
    def persistent_ids(self) -> torch.Tensor:
        """Return one stable identity per current physical row."""

        if self._persistent_ids is None:
            raise RuntimeError("compact density controller is not bound")
        return self._persistent_ids

    def bind(
        self,
        params: dict[str, torch.Tensor],
        optimizers: dict[str, torch.optim.Optimizer],
        *,
        extent: float,
        n_views: int,
        attempts_per_step: int,
    ) -> None:
        """Bind one fresh trainer state."""

        if self._density is not None:
            raise RuntimeError("compact density controller cannot be rebound")
        if tuple(params) != _GROUP_ORDER or tuple(optimizers) != _GROUP_ORDER:
            raise ValueError("compact density parameter and optimizer groups are not canonical")
        if n_views <= 0 or attempts_per_step <= 0:
            raise ValueError("compact density training dimensions must be positive")
        count = int(params["means"].shape[0])
        if count <= 0:
            raise ValueError("compact density control requires a non-empty initialization")
        device = params["means"].device
        self._density = DensityController(
            self.config,
            count,
            float(extent),
            device=device,
        )
        self._generator = torch.Generator(device=device).manual_seed(self.seed)
        self._persistent_ids = torch.arange(count, dtype=torch.long, device=device)
        self._initial_count = count
        self._next_id = count

    def needs_compositing_color_basis(self, step: int) -> bool:
        """Classic ADC needs screen gradients, not a compositing-color VJP."""

        del step
        return False

    def observe_pre_backward(
        self,
        *,
        step: int,
        view_index: int,
        output: PointRenderOutput,
        point_loss: torch.Tensor,
        active: torch.Tensor,
        attempts: int,
    ) -> None:
        """The ordinary compact loss supplies every gradient needed by classic ADC."""

        del step, view_index, output, point_loss, active, attempts

    def observe_post_backward(
        self,
        *,
        step: int,
        view_index: int,
        output: PointRenderOutput,
        width: int,
        height: int,
    ) -> None:
        """Accumulate the retained screen-space mean gradients from one point microbatch."""

        del step, view_index
        if self._density is None:
            raise RuntimeError("compact density controller is not bound")
        self._density.accumulate(output, width, height)

    def after_step(
        self,
        *,
        step: int,
        params: dict[str, torch.Tensor],
        optimizers: dict[str, torch.optim.Optimizer],
        snapshot: Gaussians3D,
    ) -> dict[str, torch.Tensor]:
        """Apply a scheduled classic density transaction and update stable lineage."""

        del snapshot
        if self._density is None or self._generator is None:
            raise RuntimeError("compact density controller is not bound")
        old_ids = self.persistent_ids
        rows_before = int(old_ids.numel())
        stats_before = len(self._density.stats)
        self._density.last_surgery = None
        new_params = self._density.step(
            step,
            params,
            optimizers,
            generator=self._generator,
        )
        if len(self._density.stats) == stats_before:
            if new_params is not params:
                raise RuntimeError("unscheduled compact density step replaced parameters")
            return params

        if len(self._density.stats) != stats_before + 1:
            raise RuntimeError("compact density step emitted an unexpected number of records")
        surgery = self._density.last_surgery
        if surgery is None:
            raise RuntimeError("scheduled compact density step omitted its surgery map")
        keep_mask = surgery["keep_mask"]
        parent_rows = surgery["parent_rows"]
        if keep_mask.shape != (rows_before,) or keep_mask.dtype != torch.bool:
            raise RuntimeError("compact density keep mask is invalid")
        if parent_rows.ndim != 1 or parent_rows.dtype != torch.long:
            raise RuntimeError("compact density parent map is invalid")

        stat = self._density.stats[-1]
        cloned = int(stat["cloned"])
        split = int(stat["split"])
        pruned = int(stat["pruned"])
        expected_newborns = cloned + 2 * split
        if int(parent_rows.numel()) != expected_newborns:
            raise RuntimeError("compact density parent map disagrees with clone/split counts")
        survivor_ids = old_ids[keep_mask]
        newborn_ids = torch.arange(
            self._next_id,
            self._next_id + expected_newborns,
            dtype=torch.long,
            device=old_ids.device,
        )
        self._next_id += expected_newborns
        new_ids = torch.cat([survivor_ids, newborn_ids])
        if int(new_params["means"].shape[0]) != int(new_ids.numel()):
            raise RuntimeError("compact density persistent IDs disagree with parameter rows")

        parent_ids = old_ids[parent_rows].detach().cpu().tolist()
        newborn_values = newborn_ids.detach().cpu().tolist()
        operators = [("clone", 0)] * cloned + [("split", 0)] * split + [("split", 1)] * split
        for birth_id, parent_id, (operator, child_ordinal) in zip(
            newborn_values,
            parent_ids,
            operators,
            strict=True,
        ):
            self._lineage_by_id[int(birth_id)] = {
                "birth_id": int(birth_id),
                "parent_id": int(parent_id),
                "operator": operator,
                "child_ordinal": child_ordinal,
                "birth_step": int(step),
            }
        self._persistent_ids = new_ids
        self._events.append(
            {
                "step": int(step),
                "rows_before": rows_before,
                "rows_after": int(new_ids.numel()),
                "cloned": cloned,
                "split": split,
                "pruned": pruned,
                "opacity_reset": bool(
                    self.config.opacity_reset_every and step % self.config.opacity_reset_every == 0
                ),
            }
        )
        return new_params

    def history_record(self) -> dict:
        """Return the density schedule, events, and complete persistent newborn lineage."""

        if self._density is None or self._initial_count is None:
            raise RuntimeError("compact density controller is not bound")
        current_ids = set(int(value) for value in self.persistent_ids.detach().cpu().tolist())
        current_newborns = {identity for identity in current_ids if identity >= self._initial_count}
        if not current_newborns <= self._lineage_by_id.keys():
            raise RuntimeError("compact density lineage lost a current newborn")
        lineage = [
            {
                **self._lineage_by_id[identity],
                "survives_final": identity in current_ids,
            }
            for identity in sorted(self._lineage_by_id)
        ]
        return {
            "schema": "rtgs.classic_compact_density.v1",
            "seed": self.seed,
            "config": asdict(self.config),
            "events": list(self._events),
            "stats": list(self._density.stats),
            "lineage": lineage,
        }


__all__ = ["ClassicCompactDensityController"]
