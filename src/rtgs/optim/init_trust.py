"""Per-primitive trust schedule for an initialization the optimizer should not overwrite.

This module implements ADR-YYYY §5 (`docs/ADR-YYYY-init-preserving-densification.md`).

The standard 3DGS schedule is an optimizer *for bad initializations*: the high initial
position learning rate exists to move wrong means, and it is a destroyer of right ones.  The
trust schedule replaces "one learning rate per tensor" with "one learning rate multiplier per
primitive per parameter block", ramped smoothly from ``beta`` to 1:

```
trust(t) = beta + (1 - beta) * min(1, t / T_trust)          beta = 0.3, T_trust = 2000
```

applied to init-rooted primitives only.  Children inherit the *current* trust state of their
parent at creation time rather than a fresh ``beta`` — otherwise every densification event
would re-freeze a region the optimizer had already earned the right to move.

Which blocks are trusted encodes which parts of the init are measurements and which are
priors:

- **position** -> ``trust(t)``: means are the measured asset.
- **tangential log-scales** -> ``trust(t)``: measured from the 2D footprints (ADR-XXXX §2.1).
- **normal log-scale** -> ``1``: ``rho`` is a declared prior, not a measurement, and must be
  free to correct itself immediately; trusting it would defend our own guess against the data.
- **rotation** -> ``c_hat * trust(t) + (1 - c_hat)``: trust the frame only where the PCA
  planarity justified it.
- **opacity and SH** -> ``1``: appearance parameters carry the photometric fit, and throttling
  them starves the only error signal the trusted geometry receives.

It is **not a freeze**: ``beta > 0`` always, so a wrong init stays overridable by the data.

Mechanically, a per-row multiplier cannot be expressed as an Adam learning rate, because Adam
normalizes by the second moment: scaling the gradient changes nothing.  The multiplier is
therefore applied as an exact post-step interpolation ``p <- p_old + m * (p_new - p_old)``,
which reproduces a per-row learning rate exactly for any first-order optimizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

NULL_ROOT = -1

#: Parameter blocks whose trust multiplier is columnwise rather than uniform over the row.
_SCALE_BLOCK = "scales"


@dataclass(frozen=True)
class TrustConfig:
    """ADR-YYYY §5 levels.  ``enabled=False`` reproduces the incumbent trainer exactly."""

    enabled: bool = True
    beta: float = 0.3
    t_trust: int = 2000
    trust_means: bool = True
    trust_tangential_scales: bool = True
    trust_rotation: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be bool")
        for name in ("trust_means", "trust_tangential_scales", "trust_rotation"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if isinstance(self.t_trust, bool) or not isinstance(self.t_trust, int) or self.t_trust <= 0:
            raise ValueError("t_trust must be a positive integer")
        if not math.isfinite(self.beta) or not 0.0 < self.beta <= 1.0:
            raise ValueError("beta must lie in (0,1]: a zero beta would be a freeze, not trust")


class TrustSchedule:
    """Tracks per-primitive lineage and applies the §5 multipliers around an optimizer step.

    Usage inside a training loop::

        schedule.capture(params)          # before optimizer.step()
        for optimizer in optimizers: optimizer.step()
        schedule.apply(params, step)      # after optimizer.step()

    ``capture``/``apply`` are no-ops when the schedule is disabled, so the incumbent code path
    is bit-identical (ADR-YYYY §A5).
    """

    def __init__(
        self,
        config: TrustConfig,
        n: int,
        *,
        coherence: torch.Tensor | None = None,
        root_id: torch.Tensor | None = None,
        device: torch.device | str = "cpu",
    ):
        self.config = config
        device = torch.device(device)
        self.device = device
        if root_id is None:
            root_id = torch.arange(n, dtype=torch.int64, device=device)
        else:
            root_id = root_id.detach().to(device=device, dtype=torch.int64).clone()
        if root_id.shape != (n,):
            raise ValueError("root_id must have shape (N,)")
        self.root_id = root_id
        if coherence is None:
            coherence = torch.ones(n, dtype=torch.float32, device=device)
        else:
            coherence = coherence.detach().to(device=device, dtype=torch.float32).clone()
        if coherence.shape != (n,):
            raise ValueError("coherence must have shape (N,)")
        self.coherence = coherence.clamp(0.0, 1.0)
        # Trust age advances once per optimizer step; children copy their parent's age, which is
        # what "inherit the CURRENT trust state" means operationally.
        self.trust_age = torch.zeros(n, dtype=torch.float32, device=device)
        self._captured: dict[str, torch.Tensor] = {}

    # -- lineage ---------------------------------------------------------------------------

    @property
    def is_init_rooted(self) -> torch.Tensor:
        return self.root_id != NULL_ROOT

    def reindex(self, keep_mask: torch.Tensor, parent_rows: torch.Tensor | None) -> None:
        """Apply the same survivor/append surgery the parameter tensors just underwent.

        ``keep_mask`` selects survivors in their original order; ``parent_rows`` names the
        parent of each appended row, in append order.  ``NULL_ROOT`` parents (insertions and
        relocations) are passed as ``-1`` and produce ``NULL_ROOT`` children.
        """
        kept_root = self.root_id[keep_mask]
        kept_age = self.trust_age[keep_mask]
        kept_coherence = self.coherence[keep_mask]
        if parent_rows is None or parent_rows.numel() == 0:
            self.root_id, self.trust_age, self.coherence = kept_root, kept_age, kept_coherence
            return
        parent_rows = parent_rows.to(device=self.device, dtype=torch.int64)
        fresh = parent_rows < 0
        safe = parent_rows.clamp_min(0)
        child_root = torch.where(fresh, torch.full_like(safe, NULL_ROOT), self.root_id[safe])
        child_age = torch.where(fresh, torch.zeros_like(self.trust_age[safe]), self.trust_age[safe])
        child_coherence = torch.where(
            fresh, torch.zeros_like(self.coherence[safe]), self.coherence[safe]
        )
        self.root_id = torch.cat([kept_root, child_root])
        self.trust_age = torch.cat([kept_age, child_age])
        self.coherence = torch.cat([kept_coherence, child_coherence])

    # -- schedule --------------------------------------------------------------------------

    def trust(self) -> torch.Tensor:
        """(N,) per-primitive ``trust`` value; ``1`` for everything not init-rooted."""
        cfg = self.config
        ramp = (self.trust_age / float(cfg.t_trust)).clamp(0.0, 1.0)
        value = cfg.beta + (1.0 - cfg.beta) * ramp
        return torch.where(self.is_init_rooted, value, torch.ones_like(value))

    def multipliers(self) -> dict[str, torch.Tensor]:
        """Per-parameter multipliers, broadcastable against each parameter's shape."""
        cfg = self.config
        trust = self.trust()
        one = torch.ones_like(trust)
        out: dict[str, torch.Tensor] = {}
        if cfg.trust_means:
            out["means"] = trust[:, None]
        if cfg.trust_rotation:
            out["quats"] = (self.coherence * trust + (1.0 - self.coherence))[:, None]
        if cfg.trust_tangential_scales:
            # Columns are (sigma_t1, sigma_t2, sigma_n) in the ADR-XXXX surfel frame.  The
            # normal axis is a prior and must stay free to correct itself immediately.
            out[_SCALE_BLOCK] = torch.stack([trust, trust, one], dim=-1)
        return out

    # -- step hooks ------------------------------------------------------------------------

    def capture(self, params: dict[str, torch.Tensor]) -> None:
        if not self.config.enabled:
            return
        self._captured = {
            name: params[name].detach().clone() for name in self.multipliers() if name in params
        }

    def apply(self, params: dict[str, torch.Tensor], step: int) -> None:
        if not self.config.enabled:
            return
        multipliers = self.multipliers()
        with torch.no_grad():
            for name, previous in self._captured.items():
                parameter = params.get(name)
                if parameter is None or parameter.shape != previous.shape:
                    # A density event between capture and apply invalidates the pairing; the
                    # step is left untouched rather than silently mis-applied.
                    continue
                multiplier = multipliers[name].to(parameter.dtype)
                if multiplier.shape[0] != parameter.shape[0]:
                    raise RuntimeError(
                        "trust lineage is out of sync with the parameters "
                        f"({multiplier.shape[0]} vs {parameter.shape[0]}): the density "
                        "controller changed the primitive count without calling reindex()"
                    )
                parameter.copy_(previous + multiplier * (parameter.detach() - previous))
        self._captured = {}
        self.trust_age = self.trust_age + torch.where(
            self.is_init_rooted,
            torch.ones_like(self.trust_age),
            torch.zeros_like(self.trust_age),
        )

    def summary(self) -> dict[str, float]:
        trust = self.trust()
        return {
            "n": int(trust.numel()),
            "n_init_rooted": int(self.is_init_rooted.sum()),
            "trust_median": float(trust.median()) if trust.numel() else float("nan"),
            "trust_min": float(trust.min()) if trust.numel() else float("nan"),
            "coherence_median": float(self.coherence.median())
            if self.coherence.numel()
            else float("nan"),
        }


__all__ = ["NULL_ROOT", "TrustConfig", "TrustSchedule"]
