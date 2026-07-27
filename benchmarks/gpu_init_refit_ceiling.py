#!/usr/bin/env python3
"""Do the Beam Fusion means carry the value?  A refit-ceiling 2x2.

Every initialization arm so far repaired the non-mean parameters with an *analytic* rule -- a cover
sigma at a fixed ratio, optionally an oriented frame, optionally a coverage-derived opacity.  All of
them landed small, and repairing more parameters made it worse.  Two explanations fit:

* **E1, the estimator is bad** -- the means are good and the analytic rule is a poor way to set
  scales, rotations, opacity and colour;
* **E2, the means do not carry the value** -- no setting of the other parameters helps much.

This harness separates them by replacing the analytic rule with a direct optimization of the
non-mean parameters against the training images, means held exactly frozen (``lr_means = 0.0``,
``densify = False``; verified to move means by exactly 0.0).  That is the *ceiling* of "good means,
everything else correct" -- no analytic estimator of the same parameters can beat it.

Crossing means (Beam Fusion / uniform-random) with how the non-mean parameters are set (analytic /
refit) is what makes the question decidable.  ``random-refit`` is the decisive arm: if the refit
lifts random points as much as it lifts Beam Fusion points, the refit is doing the work, not the
means.

The refit is spent *from* the step budget, never added to it, so a refit arm has to earn its head
start rather than be bought one.

Frozen protocol: ``benchmarks/results/20260725_init_refit_ceiling_PREREG.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.compact_views import CompactDataset
from rtgs.data.scene import SceneData
from rtgs.lift.beam_fusion import BeamFusionConfig, fuse_gaussian_beams
from rtgs.lift.surfel_init import (
    SurfelInitConfig,
    beam_contributor_footprints,
    estimate_local_surface_frames,
    reconcile_covariances,
)
from rtgs.optim.density import DensityConfig
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.base import get_rasterizer

try:
    from benchmarks.scene_builder import build_scene_at, train_inputs_at
except ModuleNotFoundError:  # direct invocation
    from scene_builder import build_scene_at, train_inputs_at  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00009/gaussians2d"
DEFAULT_OUT = ROOT / "runs/gpu_init_refit_ceiling"
DEFAULT_PROTOCOL = ROOT / "benchmarks/results/20260725_init_refit_ceiling_PREREG.md"

ARMS = ("ci", "cover-iso", "ci-refit", "random", "random-refit")
REFIT_ARMS = frozenset({"ci-refit", "random-refit"})
MODES = ("fixed", "density", "mcmc", "density-classic")
DOWNSCALE = 4
N_INIT = 5_000
ITERATIONS = 7_000
REFIT_STEPS = 300
EVAL_EVERY = 250
SEED = 0

TARGETS_DB = (19.0, 20.0, 21.0)
PRIMARY_TARGET_DB = 21.0
PRIMARY_MODE = "mcmc"
GATE_BETTER = 0.75
GATE_WORSE = 1.33
GATE_MATERIAL = 0.5
DEFAULT_CELLS = ("mcmc:3", "density:3")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _median_nn_spacing(means: torch.Tensor, chunk: int = 1024) -> float:
    best = torch.empty(means.shape[0], dtype=means.dtype)
    for start in range(0, means.shape[0], chunk):
        block = means[start : start + chunk]
        d = torch.cdist(block, means)
        d[torch.arange(block.shape[0]), torch.arange(start, start + block.shape[0])] = float("inf")
        best[start : start + block.shape[0]] = d.amin(dim=1)
    return float(best.median())


def build_random_arm(reference: Gaussians3D, n: int, generator: torch.Generator) -> Gaussians3D:
    """Uniform points in the Beam Fusion bounding box, grey colour, spacing-matched extent.

    Same construction as the cost-to-target protocol: the box is handed over for free so the
    comparison isolates the fine-grained distribution rather than knowing where the object is.
    """
    lo = reference.means.amin(dim=0)
    hi = reference.means.amax(dim=0)
    unit = torch.rand((n, 3), generator=generator, dtype=reference.means.dtype)
    means = lo + unit * (hi - lo)
    spacing = _median_nn_spacing(means)
    log_scales = torch.full((n, 3), math.log(spacing * 0.5), dtype=reference.log_scales.dtype)
    quats = torch.zeros((n, 4), dtype=reference.quats.dtype)
    quats[:, 0] = 1.0
    opacity = torch.full((n,), 0.10, dtype=reference.opacity.dtype)
    sh = torch.zeros((n, reference.sh.shape[1], 3), dtype=reference.sh.dtype)
    return Gaussians3D(means=means, quats=quats, log_scales=log_scales, opacity=opacity, sh=sh)


def build_initializations(dataset: CompactDataset, n_init: int) -> tuple[dict, dict]:
    inputs = train_inputs_at(dataset)
    extent = float(dataset.bounds_hint[1])
    beam_config = BeamFusionConfig(
        min_views=3,
        transverse_gate_sigma=3.0,
        max_color_distance=0.35,
        color_sigma=0.25,
        fold_in_gate_sigma=3.0,
        nms_voxel_size=extent / 100.0,
        init_opacity=0.10,
        source_chunk=256,
        max_components=n_init,
        seed_budget_multiplier=4,
    )
    started = time.perf_counter()
    beam = fuse_gaussian_beams(inputs, beam_config)
    elapsed = time.perf_counter() - started
    ci = beam.gaussians.detach()

    floor = beam_contributor_footprints(inputs, beam)
    frames = estimate_local_surface_frames(ci.means, SurfelInitConfig())
    cover_iso = reconcile_covariances(
        ci,
        SurfelInitConfig(isotropic=True, use_coverage_opacity=False, fixed_opacity=0.10),
        resolution_floor=floor,
        frames=frames,
    )

    generator = torch.Generator().manual_seed(SEED)
    random_arm = build_random_arm(ci, ci.n, generator)

    # The refit arms start from the same tensors as their non-refit counterparts; Phase A is what
    # makes them different, so any gap is attributable to the refit alone.
    inits = {
        "ci": ci,
        "cover-iso": cover_iso.gaussians,
        "ci-refit": ci,
        "random": random_arm,
        "random-refit": random_arm,
    }
    diagnostics = {
        "beam_elapsed_seconds": elapsed,
        "n_gaussians": ci.n,
        "refit_steps": REFIT_STEPS,
        "refit_arms": sorted(REFIT_ARMS),
        "random_arm": {
            "median_nn_spacing": _median_nn_spacing(random_arm.means),
            "beam_median_nn_spacing": _median_nn_spacing(ci.means),
        },
        "arm_scale_summary": {
            name: {
                "sigma_max_median": float(arm.scales.amax(dim=-1).median()),
                "sigma_min_median": float(arm.scales.amin(dim=-1).median()),
                "opacity_median": float(arm.opacity.median()),
            }
            for name, arm in inits.items()
        },
    }
    return inits, diagnostics


def refit_config(steps: int, total: int, rasterizer: str, device: str) -> TrainConfig:
    """Phase A: optimize everything except the means, no topology change."""
    return TrainConfig(
        iterations=steps,
        lr_means=0.0,  # verified: means move by exactly 0.0
        densify=False,
        rasterizer=rasterizer,
        device=device,
        eval_every=EVAL_EVERY,
        target_sh_degree=3,
        use_masks=True,
        random_background=False,
        seed=SEED,
        iteration_offset=0,
        schedule_iterations=total,
    )


def main_config(
    mode: str, steps: int, offset: int, total: int, budget: int, rasterizer: str, device: str
) -> TrainConfig:
    strategy = {
        "fixed": "classic",
        "density": "gsplat-default",
        "mcmc": "gsplat-mcmc",
        "density-classic": "classic",
    }[mode]
    return TrainConfig(
        iterations=steps,
        rasterizer=rasterizer,
        device=device,
        densify=mode != "fixed",
        density_strategy=strategy,
        density=DensityConfig(
            start_iter=500,
            stop_iter=int(total * 0.5),
            every=100,
            grad_threshold=2e-4,
            absgrad=False,
            prune_opacity=0.005,
            prune_scale_frac=0.1,
            max_gaussians=budget,
            opacity_reset_every=3_000,
            opacity_reset_value=0.011,
        ),
        eval_every=EVAL_EVERY,
        target_sh_degree=3,
        use_masks=True,
        random_background=False,
        seed=SEED,
        iteration_offset=offset,
        schedule_iterations=total,
    )


def steps_to_targets(curve: list[dict], targets=TARGETS_DB) -> dict:
    crossings = {}
    for target in targets:
        hit = next((p for p in curve if p["heldout_psnr_fg"] >= target), None)
        crossings[f"{target:.1f}"] = (
            None
            if hit is None
            else {
                "step": hit["step"],
                "n": hit["n"],
                "seconds": hit["seconds"],
                "train_seconds": hit["train_seconds"],
                "psnr_fg": hit["heldout_psnr_fg"],
            }
        )
    return crossings


def run_cell(
    arm: str, mode: str, budget: int, label: str, init: Gaussians3D, scene, split, out: Path, args
) -> dict:
    cell = out / label / arm
    cell.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    renderer = get_rasterizer(args.rasterizer, device=device)
    model = init.to(device)

    def metrics(current: Gaussians3D) -> dict:
        return {
            "heldout": Trainer.evaluate_metrics(scene, current, renderer, split.heldout_local),
            "train": Trainer.evaluate_metrics(scene, current, renderer, split.train_local),
            "heldout_extrapolative": Trainer.evaluate_metrics(
                scene, current, renderer, [split.extrapolative_local]
            ),
        }

    torch.manual_seed(SEED)
    init_metrics = metrics(model)
    model.save_ply(cell / "gaussians_init.ply")
    curve: list[dict] = []
    started = time.perf_counter()

    def checkpoint(snapshot: Gaussians3D, step: int) -> None:
        values = metrics(snapshot)
        elapsed = time.perf_counter() - started
        curve.append(
            {
                "step": step,
                "n": snapshot.n,
                "seconds": elapsed,
                **{f"heldout_{k}": v for k, v in values["heldout"].items()},
            }
        )
        print(
            f"[{label}/{arm}] step {step} n={snapshot.n} "
            f"held_fg={values['heldout'].get('psnr_fg', float('nan')):.3f}dB ({elapsed:.0f}s)",
            flush=True,
        )

    total = args.iterations
    refit = args.refit_steps if arm in REFIT_ARMS else 0
    refit_metrics = None
    # step -> training seconds, taken from the trainer's own history so that checkpoint-callback
    # time is excluded.  Evaluation cost is identical across arms and is not something an
    # initialization changes, so charging it to the arm would distort the time comparison.
    train_clock: dict[int, float] = {}
    refit_train_seconds = 0.0
    if refit:
        model, history_a = Trainer(refit_config(refit, total, args.rasterizer, args.device)).train(
            scene, model, checkpoint_callback=checkpoint
        )
        for step, seconds in history_a["elapsed"]:
            train_clock[int(step)] = float(seconds)
        if history_a["elapsed"]:
            refit_train_seconds = float(history_a["elapsed"][-1][1])
        refit_metrics = metrics(model)
        print(
            f"[{label}/{arm}] refit done ({refit_train_seconds:.1f}s train): "
            f"held_fg={refit_metrics['heldout'].get('psnr_fg', float('nan')):.3f}dB",
            flush=True,
        )

    final, history_b = Trainer(
        main_config(mode, total - refit, refit, total, budget, args.rasterizer, args.device)
    ).train(scene, model, checkpoint_callback=checkpoint)
    for step, seconds in history_b["elapsed"]:
        train_clock[int(step)] = refit_train_seconds + float(seconds)
    for point in curve:
        point["train_seconds"] = train_clock.get(point["step"])
    final.save_ply(cell / "gaussians_final.ply")

    record = {
        "arm": arm,
        "mode": mode,
        "label": label,
        "budget": budget,
        "refit_steps": refit,
        "elapsed_seconds": time.perf_counter() - started,
        "init_metrics": init_metrics,
        "post_refit_metrics": refit_metrics,
        "final_metrics": metrics(final),
        "final_n": final.n,
        "reached_budget": final.n >= budget,
        "steps_to_target": steps_to_targets(curve),
        "curve": curve,
    }
    (cell / "record.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def _ratio(treatment, control, field: str, censor_value: float) -> tuple[float, bool]:
    """Ratio on ``field``; a never-reached arm is charged the full run so a miss looks bad."""
    censored = treatment is None or control is None
    t = censor_value if treatment is None else treatment[field]
    c = censor_value if control is None else control[field]
    if t is None or c is None:
        return float("nan"), True
    return (float("inf") if c == 0 else t / c), censored


def _band(ratio: float) -> str:
    """Which side of the preregistered thresholds a ratio falls on."""
    if ratio != ratio:  # NaN
        return "unscored"
    if ratio <= GATE_BETTER:
        return "better"
    if ratio >= GATE_WORSE:
        return "worse"
    return "same"


def _dual(treatment, control, iterations: int, budget_seconds: float, names) -> dict:
    """Score one comparison on both co-primaries; disagreement is reported as split."""
    better, mid, worse = names
    step_ratio, step_censored = _ratio(treatment, control, "step", iterations)
    time_ratio, time_censored = _ratio(treatment, control, "train_seconds", budget_seconds)
    bands = (_band(step_ratio), _band(time_ratio))
    if "unscored" in bands:
        verdict = "unscored (missing timing)"
    elif bands[0] == bands[1]:
        verdict = {"better": better, "same": mid, "worse": worse}[bands[0]]
    else:
        verdict = f"split: steps {bands[0]}, time {bands[1]} — promotes nothing"
    return {
        "step_ratio": step_ratio,
        "train_seconds_ratio": time_ratio,
        "censored": step_censored or time_censored,
        "verdict": verdict,
    }


def score(cells: dict, iterations: int) -> dict:
    primary = cells.get(f"{PRIMARY_MODE}:3")
    if primary is None:
        return {"note": f"primary cell {PRIMARY_MODE}:3 not run; no gate scored"}
    key = f"{PRIMARY_TARGET_DB:.1f}"
    reached = {a: primary[a]["steps_to_target"][key] for a in primary}
    out = {"target_db": PRIMARY_TARGET_DB, "cell": f"{PRIMARY_MODE}:3", "reached": reached}

    # Censoring value for time: the slowest observed full run, so "never reached" is charged at
    # least as much as any arm that did finish.
    budget_seconds = max(
        (
            c["train_seconds"]
            for a in primary.values()
            for c in a["curve"]
            if c.get("train_seconds")
        ),
        default=0.0,
    )

    if "ci-refit" in reached and "cover-iso" in reached:
        out["R1_refit_vs_analytic"] = _dual(
            reached["ci-refit"],
            reached["cover-iso"],
            iterations,
            budget_seconds,
            (
                "G-R1a estimator was the problem (E1)",
                "G-R1b analytic rule was not the limit",
                "G-R1c refit hurts",
            ),
        )
    if "ci-refit" in reached and "random-refit" in reached:
        result = _dual(
            reached["ci-refit"],
            reached["random-refit"],
            iterations,
            budget_seconds,
            (
                "directional, sub-threshold",
                "G-R2b means are not material (E2)",
                "G-R2c random means are better",
            ),
        )
        material = (
            result["step_ratio"] <= GATE_MATERIAL and result["train_seconds_ratio"] <= GATE_MATERIAL
        ) or (reached["random-refit"] is None and reached["ci-refit"] is not None)
        if material:
            result["verdict"] = "G-R2a means are material"
        result["random_refit_never_reached"] = reached["random-refit"] is None
        out["R2_means_vs_random"] = result
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--downscale", type=int, default=DOWNSCALE)
    parser.add_argument("--n-init", type=int, default=N_INIT)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--refit-steps", type=int, default=REFIT_STEPS)
    parser.add_argument("--rasterizer", default="gsplat")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--cells", nargs="+", default=list(DEFAULT_CELLS))
    args = parser.parse_args()
    if not args.protocol.is_file():
        raise FileNotFoundError(f"frozen protocol missing: {args.protocol}")

    parsed_cells = []
    for spec in args.cells:
        mode, _, mult = spec.partition(":")
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r} in cell {spec!r}")
        parsed_cells.append((spec, mode, float(mult or 3)))

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dataset = CompactDataset.load(args.dataset, device="cpu")
    split = build_scene_at(dataset, downscale=args.downscale)
    scene: SceneData = split.scene
    print(
        f"[scene] {len(split.train_local)} train + {len(split.heldout_local)} held-out views, "
        f"first image {scene.images[0].shape[1]}x{scene.images[0].shape[0]}",
        flush=True,
    )
    torch.manual_seed(SEED)
    initializations, diagnostics = build_initializations(dataset, args.n_init)
    print(json.dumps(diagnostics["arm_scale_summary"], indent=2), flush=True)

    cells: dict[str, dict] = {}
    for label, mode, mult in parsed_cells:
        budget = int(round(mult * args.n_init))
        cells[label] = {}
        for arm in args.arms:
            cells[label][arm] = run_cell(
                arm, mode, budget, label, initializations[arm], scene, split, out, args
            )

    decision = score(cells, args.iterations)
    summary = {
        "schema": "rtgs.gpu_init_refit_ceiling.v1",
        "protocol": {"path": str(args.protocol), "sha256": _sha256(args.protocol)},
        "dataset": str(args.dataset),
        "downscale": args.downscale,
        "image_size": [int(scene.images[0].shape[1]), int(scene.images[0].shape[0])],
        "n_init": args.n_init,
        "iterations": args.iterations,
        "refit_steps": args.refit_steps,
        "eval_every": EVAL_EVERY,
        "rasterizer": args.rasterizer,
        "device": args.device,
        "seed": SEED,
        "targets_db": list(TARGETS_DB),
        "train_views": [scene.view_names[i] for i in split.train_local],
        "heldout_views": [scene.view_names[i] for i in split.heldout_local],
        "initialization_diagnostics": diagnostics,
        "decision": decision,
        "cells": cells,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(decision, indent=2), flush=True)
    print(f"[written] {out / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
