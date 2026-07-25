#!/usr/bin/env python3
"""Is the control's deficit the initialization's scale, or a density controller mismatched to it?

The scale-gradient diagnostic left the previous conclusion confounded: the control's primitives do
grow, but ~860 steps per doubling, while density control commits the whole budget between steps 20
and 500 and reads the scale as it is then.  "The initialization is under-scaled" and "the schedule
and the absolute split threshold are mismatched to it" predict the same outcome, and only the
first has been tested.

Four arms therefore share the **unchanged `ci` initialization** and differ only in
``DensityConfig`` — default, delayed (event-count matched), initializer-relative split boundary,
and both — against the unchanged ``surfel`` treatment.  Written to narrow the initialization
claim, not to defend it.

Also closes two caveats of the gradient diagnostic by logging, at every checkpoint: the growth
trajectory of identity-tracked surviving originals as a curve, their split-eligibility, and a full
gradient-telemetry pass at that point in parameter space rather than only at step 0.

Frozen protocol: ``benchmarks/results/20260724_beam_surfel_schedule_confound_PREREG.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.compact_views import CompactDataset
from rtgs.data.scene import SceneData
from rtgs.optim.density import DensityConfig, DensityController
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.base import get_rasterizer

try:
    from benchmarks.beam_surfel_init import GRAD_THRESHOLD, build_initializations, build_scene
    from benchmarks.beam_surfel_matched_capacity import MATCHED_BUDGET
    from benchmarks.beam_surfel_scale_gradient import measure_arm as gradient_telemetry
except ModuleNotFoundError:  # direct invocation
    from beam_surfel_init import (  # type: ignore[no-redef]
        GRAD_THRESHOLD,
        build_initializations,
        build_scene,
    )
    from beam_surfel_matched_capacity import MATCHED_BUDGET  # type: ignore[no-redef]
    from beam_surfel_scale_gradient import measure_arm as gradient_telemetry  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
FRAME = "frame_00009"
DEFAULT_OUT = ROOT / "runs/beam_surfel_schedule_confound_20260724"
DEFAULT_PROTOCOL = ROOT / "benchmarks/results/20260724_beam_surfel_schedule_confound_PREREG.md"

ARMS = ("ci-baseline", "ci-late", "ci-relsplit", "ci-late-relsplit", "surfel-baseline")
REPAIRED = ("ci-late", "ci-relsplit", "ci-late-relsplit")
SEEDS = (0, 1, 2)
ITERATIONS = 1_000
EVAL_EVERY = 25
TELEMETRY_EVERY = 250
DEFAULT_SPLIT_SCALE_FRAC = 0.01
DECISION_MARGIN_DB = 0.15
# Event-count matched with the default window: (500-20)/4 + 1 == (780-300)/4 + 1 == 121.
LATE_WINDOW = (300, 780)
BASE_WINDOW = (20, 500)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arm_spec(arm: str, relative_frac: float) -> tuple[str, tuple[int, int], float]:
    """(initialization key, density window, split_scale_frac) for a frozen arm name."""
    init = "surfel" if arm.startswith("surfel") else "ci"
    window = LATE_WINDOW if "late" in arm else BASE_WINDOW
    frac = relative_frac if "relsplit" in arm else DEFAULT_SPLIT_SCALE_FRAC
    return init, window, frac


def density_config(window: tuple[int, int], split_scale_frac: float, budget: int) -> DensityConfig:
    return DensityConfig(
        start_iter=window[0],
        stop_iter=window[1],
        every=4,
        grad_threshold=GRAD_THRESHOLD,
        absgrad=False,
        split_scale_frac=split_scale_frac,
        prune_opacity=0.005,
        prune_scale_frac=0.1,
        max_gaussians=budget,
        opacity_reset_every=100,
        opacity_reset_value=0.011,
    )


def train_config(seed: int, iterations: int, density: DensityConfig) -> TrainConfig:
    return TrainConfig(
        iterations=iterations,
        rasterizer="torch",
        device="cpu",
        densify=True,
        density_strategy="classic",
        density=density,
        eval_every=EVAL_EVERY,
        target_sh_degree=3,
        sh_degree_interval=33,
        use_masks=True,
        random_background=False,
        seed=seed,
    )


class TrackingController(DensityController):
    """Identity-carrying controller.

    A split replaces its parent, so a split row stops counting as an original.
    """

    instances: list[TrackingController] = []

    def __init__(self, config, n_gaussians, scene_extent, device="cpu"):
        super().__init__(config, n_gaussians, scene_extent, device=device)
        self.identity = torch.arange(n_gaussians)
        self.events: list[dict] = []
        TrackingController.instances.append(self)

    def step(self, iteration, params, optimizer, generator=None, force_budget=False):
        import rtgs.optim.density as density_module

        cfg = self.cfg
        scheduled = cfg.start_iter <= iteration <= cfg.stop_iter and iteration % cfg.every == 0
        boundary = cfg.split_scale_frac * self.extent
        eligible = None
        if scheduled:
            scale_max = params["scales"].detach().exp().max(dim=-1).values
            originals = self.identity >= 0
            if bool(originals.any()):
                eligible = float((scale_max[originals] > boundary).double().mean())

        captured: list[tuple[torch.Tensor, int]] = []
        original_edit = density_module._edit_params

        def spy(opt, params_, keep_mask, extras):
            captured.append((keep_mask.detach().clone(), sum(e["means"].shape[0] for e in extras)))
            return original_edit(opt, params_, keep_mask, extras)

        density_module._edit_params = spy
        try:
            new_params = super().step(
                iteration, params, optimizer, generator=generator, force_budget=force_budget
            )
        finally:
            density_module._edit_params = original_edit

        stats = self.stats[-1] if self.stats and self.stats[-1]["iteration"] == iteration else {}
        for keep_mask, n_extras in captured:
            self.identity = torch.cat(
                [self.identity[keep_mask], torch.full((n_extras,), -1, dtype=torch.long)]
            )
            self.events.append(
                {
                    "iteration": iteration,
                    "n_after": int(self.identity.shape[0]),
                    "originals_alive": int((self.identity >= 0).sum()),
                    "frac_originals_split_eligible": eligible,
                    **{k: stats[k] for k in ("cloned", "split", "pruned") if k in stats},
                }
            )
        return new_params


def run_cell(
    arm: str,
    seed: int,
    init: Gaussians3D,
    scene: SceneData,
    train_local: list[int],
    heldout_local: list[int],
    out: Path,
    iterations: int,
    density: DensityConfig,
) -> dict:
    import rtgs.optim.trainer as trainer_module

    cell = out / f"seed{seed}" / arm
    cell.mkdir(parents=True, exist_ok=True)
    renderer = get_rasterizer("torch", device=torch.device("cpu"))
    boundary = density.split_scale_frac * float(scene.bounds_hint[1])
    curve: list[dict] = []
    started = time.perf_counter()
    torch.manual_seed(seed)

    def checkpoint(snapshot: Gaussians3D, step: int) -> None:
        controller = TrackingController.instances[-1]
        identity = controller.identity
        originals = identity >= 0
        scale_max = snapshot.scales.amax(dim=-1).double()
        row = {
            "step": step,
            "n": snapshot.n,
            "originals_alive": int(originals.sum()),
            # Growth trajectory as a curve, not an inferred average rate.
            "original_sigma_max_median": (
                float(scale_max[originals].median()) if bool(originals.any()) else None
            ),
            "frac_originals_above_split_boundary": (
                float((scale_max[originals] > boundary).double().mean())
                if bool(originals.any())
                else None
            ),
            **{
                f"heldout_{k}": v
                for k, v in Trainer.evaluate_metrics(
                    scene, snapshot, renderer, heldout_local
                ).items()
            },
        }
        # Gradient telemetry away from step 0, at this point in parameter space.
        if step % TELEMETRY_EVERY == 0:
            row["telemetry"] = gradient_telemetry(snapshot, scene, train_local)
        curve.append(row)
        print(
            f"[seed{seed}/{arm}] step {step} n={snapshot.n} "
            f"orig_sigma={row['original_sigma_max_median'] or float('nan'):.5f} "
            f"split_elig={row['frac_originals_above_split_boundary'] or 0:.3f} "
            f"held_fg={row.get('heldout_psnr_fg', float('nan')):.3f}dB "
            f"({time.perf_counter() - started:.0f}s)",
            flush=True,
        )

    previous = trainer_module.DensityController
    trainer_module.DensityController = TrackingController
    TrackingController.instances.clear()
    try:
        final, _ = Trainer(train_config(seed, iterations, density)).train(
            scene, init, checkpoint_callback=checkpoint
        )
    finally:
        trainer_module.DensityController = previous
    final.save_ply(cell / "gaussians_final.ply")
    controller = TrackingController.instances[-1]
    events = [e for e in controller.events if e["frac_originals_split_eligible"] is not None]
    record = {
        "arm": arm,
        "seed": seed,
        "density": {
            "start_iter": density.start_iter,
            "stop_iter": density.stop_iter,
            "every": density.every,
            "split_scale_frac": density.split_scale_frac,
            "split_boundary_world_sigma": boundary,
            "max_gaussians": density.max_gaussians,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "final_n": final.n,
        "n_surviving_originals": int((controller.identity >= 0).sum()),
        "final_metrics": {
            "heldout": Trainer.evaluate_metrics(scene, final, renderer, heldout_local),
            "train": Trainer.evaluate_metrics(scene, final, renderer, train_local),
        },
        "n_density_events": len(controller.events),
        "split_eligible_first_event": events[0]["frac_originals_split_eligible"]
        if events
        else None,
        "split_eligible_last_event": events[-1]["frac_originals_split_eligible"]
        if events
        else None,
        "total_cloned": sum(e.get("cloned", 0) for e in controller.events),
        "total_split": sum(e.get("split", 0) for e in controller.events),
        "curve": curve,
    }
    (cell / "record.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def decide(cells: dict[str, dict[int, dict]], seeds: list[int]) -> dict:
    """Score S1/S2/S3 exactly as preregistered."""

    def held(arm: str, seed: int) -> float:
        return cells[arm][seed]["final_metrics"]["heldout"]["psnr_fg"]

    available = [a for a in REPAIRED if a in cells]
    means = {a: sum(held(a, s) for s in seeds) / len(seeds) for a in available}
    best = max(means, key=means.get) if means else None
    per_seed = []
    for seed in seeds:
        delta_to_treatment = held(best, seed) - held("surfel-baseline", seed)
        delta_to_baseline = held(best, seed) - held("ci-baseline", seed)
        per_seed.append(
            {
                "seed": seed,
                "best_repaired_arm": best,
                "repaired_minus_surfel_db": delta_to_treatment,
                "repaired_minus_ci_baseline_db": delta_to_baseline,
                "s1_cell": abs(delta_to_treatment) < DECISION_MARGIN_DB or delta_to_treatment >= 0,
                "s2_cell": -delta_to_treatment >= DECISION_MARGIN_DB,
                "s3_cell": delta_to_baseline >= DECISION_MARGIN_DB
                and -delta_to_treatment >= DECISION_MARGIN_DB,
            }
        )
    majority = len(seeds) // 2 + 1
    s1 = sum(c["s1_cell"] for c in per_seed) >= majority
    s3 = sum(c["s3_cell"] for c in per_seed) >= majority
    s2 = sum(c["s2_cell"] for c in per_seed) >= majority
    if s1:
        verdict = "S1_CONTROLLER_EXPLAINS_IT"
    elif s3:
        verdict = "S3_BOTH_CONTRIBUTE"
    elif s2:
        verdict = "S2_INITIALIZATION_EXPLAINS_IT"
    else:
        verdict = "INCONCLUSIVE_NO_MAJORITY"
    return {
        "margin_db": DECISION_MARGIN_DB,
        "best_repaired_arm": best,
        "mean_heldout_psnr_by_arm": {a: sum(held(a, s) for s in seeds) / len(seeds) for a in cells},
        "per_seed": per_seed,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--budget", type=int, default=MATCHED_BUDGET)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    if not args.protocol.is_file():
        raise FileNotFoundError(f"frozen protocol missing: {args.protocol}")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dataset = CompactDataset.load(
        ROOT / f"dataset/2025_03_07_stage_with_fabric/{FRAME}/gaussians2d", device="cpu"
    )
    scene, train_local, heldout_local, _ = build_scene(dataset)
    torch.manual_seed(0)
    initializations, _ = build_initializations(dataset)
    extent = float(dataset.bounds_hint[1])
    # Initializer-relative split boundary, computed from the arm's own initialization.
    relative_frac = {
        key: float(initializations[key].scales.amax(dim=-1).median()) / extent
        for key in ("ci", "surfel")
    }
    print(f"[policy] initializer-relative split_scale_frac: {relative_frac}", flush=True)

    cells: dict[str, dict[int, dict]] = {arm: {} for arm in args.arms}
    for seed in args.seeds:
        for arm in args.arms:
            init_key, window, frac = arm_spec(arm, relative_frac["ci"])
            cells[arm][seed] = run_cell(
                arm,
                seed,
                initializations[init_key],
                scene,
                train_local,
                heldout_local,
                out,
                args.iterations,
                density_config(window, frac, args.budget),
            )

    decision = (
        decide(cells, list(args.seeds))
        if {"ci-baseline", "surfel-baseline"} <= set(args.arms) and set(REPAIRED) & set(args.arms)
        else {"verdict": "NOT_EVALUATED_PARTIAL_ARMS"}
    )
    summary = {
        "schema": "rtgs.beam_surfel_schedule_confound.v1",
        "status": (
            "complete"
            if set(args.arms) == set(ARMS) and set(args.seeds) == set(SEEDS)
            else "partial"
        ),
        "protocol": {
            "path": str(args.protocol.resolve().relative_to(ROOT)),
            "sha256": _sha256(args.protocol),
        },
        "frame": FRAME,
        "matched_budget": args.budget,
        "iterations": args.iterations,
        "seeds": list(args.seeds),
        "default_split_scale_frac": DEFAULT_SPLIT_SCALE_FRAC,
        "initializer_relative_split_scale_frac": relative_frac,
        "base_window": BASE_WINDOW,
        "late_window": LATE_WINDOW,
        "decision": decision,
        "cells": {
            arm: {
                str(seed): {
                    k: rec[k]
                    for k in (
                        "density",
                        "final_metrics",
                        "final_n",
                        "n_surviving_originals",
                        "n_density_events",
                        "split_eligible_first_event",
                        "split_eligible_last_event",
                        "total_cloned",
                        "total_split",
                        "elapsed_seconds",
                        "curve",
                    )
                }
                for seed, rec in by_seed.items()
            }
            for arm, by_seed in cells.items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(decision, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
