#!/usr/bin/env python3
"""Post-hoc diagnostic: who carries the final image, the initialization or the newborns?

The matched-capacity result showed the cover-consistent initialization wins at an equal budget,
but not *why*.  Two very different stories fit the same endpoint:

* the initial Gaussians are refined into the answer and births only patch residual holes; or
* the initial Gaussians are vestigial and the quality is rebuilt by birth, in which case the
  initialization is only a seeding heuristic.

This tracks the physical identity of every initial row through clone/split/prune surgery and then
renders three subsets of the *same* final model on the held-out cameras: everything, surviving
originals only, and newborns only.  Alpha compositing is not additive, so the two subsets are not
an additive decomposition — but "originals alone already reach the endpoint" and "newborns alone
already reach the endpoint" are both directly falsifiable this way.

It also records what the classic controller can physically do to each arm: an original may only be
*cloned in place* while its largest scale sits below ``split_scale_frac * extent``, and may only be
*split* (children displaced by a draw from its own covariance) above it.

This is explicitly a **post-hoc diagnostic**, not a preregistered treatment comparison.  It can
localize a mechanism; it cannot select a default, a rule, or a hyperparameter.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.compact_views import CompactDataset
from rtgs.data.scene import SceneData
from rtgs.optim.density import DensityController
from rtgs.optim.trainer import Trainer
from rtgs.render.base import get_rasterizer

try:
    from benchmarks.beam_surfel_init import build_initializations, build_scene
    from benchmarks.beam_surfel_matched_capacity import MATCHED_BUDGET, train_config
except ModuleNotFoundError:  # direct ``python benchmarks/beam_surfel_birth_attribution.py``
    from beam_surfel_init import build_initializations, build_scene  # type: ignore[no-redef]
    from beam_surfel_matched_capacity import (  # type: ignore[no-redef]
        MATCHED_BUDGET,
        train_config,
    )

ROOT = Path(__file__).resolve().parents[1]
FRAME = "frame_00009"
DEFAULT_OUT = ROOT / "runs/beam_surfel_birth_attribution_20260724"
ARMS = ("ci", "surfel")
SEED = 0
ITERATIONS = 1_000
SPLIT_SCALE_FRAC = 0.01


class IdentityController(DensityController):
    """Classic controller that carries a physical-row identity vector through every edit.

    ``identity[i] >= 0`` means row ``i`` descends from initial row ``identity[i]`` *without* having
    been replaced: clones append ``-1`` rows and splits replace the parent with two ``-1`` rows, so
    an original that is split stops being counted as alive.  That is the strict reading, and it is
    the one that makes "the originals are still here" a falsifiable statement.
    """

    instances: list[IdentityController] = []

    def __init__(self, config, n_gaussians, scene_extent, device="cpu"):
        super().__init__(config, n_gaussians, scene_extent, device=device)
        self.identity = torch.arange(n_gaussians)
        self.events: list[dict] = []
        IdentityController.instances.append(self)

    def step(self, iteration, params, optimizer, generator=None, force_budget=False):
        import rtgs.optim.density as density_module

        cfg = self.cfg
        scheduled = cfg.start_iter <= iteration <= cfg.stop_iter and iteration % cfg.every == 0
        boundary = SPLIT_SCALE_FRAC * self.extent
        eligibility: dict | None = None
        if scheduled:
            scale_max = params["scales"].detach().exp().max(dim=-1).values
            originals = self.identity >= 0
            if bool(originals.any()):
                eligibility = {
                    "n_alive_originals": int(originals.sum()),
                    # Above the boundary the controller splits (children displaced within the
                    # primitive's own covariance); below it, it can only clone in place.
                    "frac_originals_split_eligible": float(
                        (scale_max[originals] > boundary).double().mean()
                    ),
                    "frac_newborns_split_eligible": (
                        float((scale_max[~originals] > boundary).double().mean())
                        if bool((~originals).any())
                        else None
                    ),
                }

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
                    "eligibility": eligibility,
                    **{k: stats[k] for k in ("cloned", "split", "pruned") if k in stats},
                }
            )
        return new_params


def subset_metrics(
    scene: SceneData, model: Gaussians3D, mask: torch.Tensor, views: list[int], renderer
) -> dict:
    if not bool(mask.any()):
        return {"n": 0}
    values = Trainer.evaluate_metrics(scene, model.subset(mask), renderer, views)
    return {"n": int(mask.sum()), **values}


def run_arm(arm, init, scene, train_local, heldout_local, out, iterations, budget) -> dict:
    import rtgs.optim.trainer as trainer_module

    arm_dir = out / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    renderer = get_rasterizer("torch", device=torch.device("cpu"))
    init_means = init.means.detach().clone()
    torch.manual_seed(SEED)

    previous = trainer_module.DensityController
    trainer_module.DensityController = IdentityController
    IdentityController.instances.clear()
    started = time.perf_counter()
    try:
        final, _ = Trainer(train_config(SEED, iterations, budget)).train(scene, init)
    finally:
        trainer_module.DensityController = previous
    controller = IdentityController.instances[-1]
    identity = controller.identity
    if identity.shape[0] != final.n:
        raise RuntimeError(f"identity desync: {identity.shape[0]} != {final.n}")
    final.save_ply(arm_dir / "gaussians_final.ply")

    is_original = identity >= 0
    survivors = identity[is_original]
    displacement = (final.means[is_original] - init_means[survivors]).norm(dim=-1).double()
    scales = final.scales.amax(dim=-1).double()

    def describe(values: torch.Tensor) -> dict:
        return {
            "median": float(values.median()),
            "p90": float(values.quantile(0.9)),
            "mean": float(values.mean()),
        }

    record = {
        "arm": arm,
        "seed": SEED,
        "budget": budget,
        "elapsed_seconds": time.perf_counter() - started,
        "n_initial": init.n,
        "final_n": final.n,
        "n_surviving_originals": int(is_original.sum()),
        "n_newborns": int((~is_original).sum()),
        "surviving_original_fraction": float(is_original.double().mean()),
        "displacement_of_survivors": describe(displacement),
        "displacement_over_initial_sigma": describe(
            displacement / init.scales.amax(dim=-1).double()[survivors]
        ),
        "final_scale_max_originals": describe(scales[is_original]),
        "final_scale_max_newborns": (
            describe(scales[~is_original]) if bool((~is_original).any()) else None
        ),
        "final_opacity_originals": describe(final.opacity[is_original].double()),
        "final_opacity_newborns": (
            describe(final.opacity[~is_original].double()) if bool((~is_original).any()) else None
        ),
        "heldout": {
            "all": subset_metrics(
                scene, final, torch.ones(final.n, dtype=torch.bool), heldout_local, renderer
            ),
            "originals_only": subset_metrics(scene, final, is_original, heldout_local, renderer),
            "newborns_only": subset_metrics(scene, final, ~is_original, heldout_local, renderer),
        },
        "train": {
            "all": subset_metrics(
                scene, final, torch.ones(final.n, dtype=torch.bool), train_local, renderer
            ),
            "originals_only": subset_metrics(scene, final, is_original, train_local, renderer),
            "newborns_only": subset_metrics(scene, final, ~is_original, train_local, renderer),
        },
        "density_events": controller.events,
    }
    (arm_dir / "record.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--budget", type=int, default=MATCHED_BUDGET)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dataset = CompactDataset.load(
        ROOT / f"dataset/2025_03_07_stage_with_fabric/{FRAME}/gaussians2d", device="cpu"
    )
    scene, train_local, heldout_local, _ = build_scene(dataset)
    torch.manual_seed(0)
    initializations, _ = build_initializations(dataset)

    records = {}
    for arm in args.arms:
        records[arm] = run_arm(
            arm,
            initializations[arm],
            scene,
            train_local,
            heldout_local,
            out,
            args.iterations,
            args.budget,
        )
        r = records[arm]
        print(
            f"[{arm}] final {r['final_n']} = {r['n_surviving_originals']} originals + "
            f"{r['n_newborns']} newborns | held-out FG PSNR "
            f"all {r['heldout']['all']['psnr_fg']:.3f} / "
            f"originals {r['heldout']['originals_only'].get('psnr_fg', float('nan')):.3f} / "
            f"newborns {r['heldout']['newborns_only'].get('psnr_fg', float('nan')):.3f}",
            flush=True,
        )

    summary = {
        "schema": "rtgs.beam_surfel_birth_attribution.v1",
        "kind": "post-hoc diagnostic; not a preregistered comparison; selects nothing",
        "frame": FRAME,
        "seed": SEED,
        "budget": args.budget,
        "iterations": args.iterations,
        "split_boundary_world_sigma": SPLIT_SCALE_FRAC * float(dataset.bounds_hint[1]),
        "arms": records,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
