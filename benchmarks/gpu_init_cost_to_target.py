#!/usr/bin/env python3
"""Initialization as a cost reduction, not a quality ceiling.

``gpu_stage1_initialization.py`` returned its preregistered null: at a 3x budget and 7,000 steps
the cover-consistent covariance moved held-out foreground PSNR by +0.071 dB.  Two facts from that
same run say the null is about the *outcome variable* rather than the initialization -- the effect
was +0.271 dB under fixed topology where the controller cannot spend budget, and swapping the
controller (Default -> MCMC) moved the *control* arm by +0.62 dB, roughly 9x the largest
initialization effect.  Adaptive density control rebuilds the answer by birth, so a better seed
converges to the same ceiling; quality-at-a-fixed-step-count structurally cannot see the benefit.

This harness measures the quantity that was never measured: **cost to reach a target**, plus
quality under budgets too small for the controller to paper over the seed.

It also closes the repository's largest evidence gap.  Every arm of every previous initialization
experiment was Beam Fusion with bit-identical means, so "the lifted gaussians are a good
initialization" has never been tested against *not lifting*.  The ``random`` arm is that control.

Frozen protocol: ``benchmarks/results/20260725_init_cost_to_target_PREREG.md``.
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
DEFAULT_OUT = ROOT / "runs/gpu_init_cost_to_target"
DEFAULT_PROTOCOL = ROOT / "benchmarks/results/20260725_init_cost_to_target_PREREG.md"

ARMS = ("random", "ci", "cover-iso")
MODES = ("fixed", "density", "mcmc", "density-classic")
DOWNSCALE = 4
N_INIT = 5_000
ITERATIONS = 7_000
EVAL_EVERY = 250  # first-crossing resolution
SEED = 0

# Preregistered ladder and decision thresholds.
TARGETS_DB = (19.0, 20.0, 21.0)
PRIMARY_TARGET_DB = 21.0
PRIMARY_MODE = "density"
COST_GATE_BETTER = 0.75  # cover-iso vs ci: <= is G-C1
COST_GATE_WORSE = 1.33  # >= is G-C3
BASELINE_GATE_MATERIAL = 0.5  # ci vs random: <= is G-B1
# Default cells: (mode, budget multiplier).  3x is the primary; 1.0x/1.5x are the
# constrained-budget secondaries where the fixed-topology result predicts the effect lives.
DEFAULT_CELLS = ("density:3", "mcmc:3", "density:1", "density:1.5")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _median_nn_spacing(means: torch.Tensor, chunk: int = 1024) -> float:
    """Median nearest-neighbour distance, chunked so 5k x 5k never materializes at once."""
    best = torch.empty(means.shape[0], dtype=means.dtype)
    for start in range(0, means.shape[0], chunk):
        block = means[start : start + chunk]
        d = torch.cdist(block, means)
        d[torch.arange(block.shape[0]), torch.arange(start, start + block.shape[0])] = float("inf")
        best[start : start + block.shape[0]] = d.amin(dim=1)
    return float(best.median())


def build_random_arm(reference: Gaussians3D, n: int, generator: torch.Generator) -> Gaussians3D:
    """No-prior control: uniform points in the reference's own bounding box.

    The box comes from the Beam Fusion result rather than the full scene bounds on purpose -- it
    hands the baseline the object's location for free, so the comparison isolates whether the
    *fine-grained geometric distribution* matters rather than whether knowing where the object is
    matters.  That is the conservative direction: it makes the baseline harder to beat.

    Colour is grey (``rgb_to_sh(0.5) == 0``), i.e. no photometric prior at all.
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
    """Beam Fusion once (``ci``, ``cover-iso``) plus the no-prior ``random`` control."""
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

    inits = {"random": random_arm, "ci": ci, "cover-iso": cover_iso.gaussians}
    for name, arm in inits.items():
        if arm.n != ci.n:
            raise RuntimeError(f"{name} has {arm.n} gaussians, expected {ci.n}")

    diagnostics = {
        "beam_elapsed_seconds": elapsed,
        "n_gaussians": ci.n,
        "n_contributor_links": int(beam.contributor_view_indices.numel()),
        "cover_iso_rule": cover_iso.diagnostics,
        "random_arm": {
            "box_lo": [float(v) for v in ci.means.amin(dim=0)],
            "box_hi": [float(v) for v in ci.means.amax(dim=0)],
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


def train_config(mode: str, iterations: int, budget: int, rasterizer: str, device: str):
    strategy = {
        "fixed": "classic",
        "density": "gsplat-default",
        "mcmc": "gsplat-mcmc",
        "density-classic": "classic",
    }[mode]
    return TrainConfig(
        iterations=iterations,
        rasterizer=rasterizer,
        device=device,
        densify=mode != "fixed",
        density_strategy=strategy,
        density=DensityConfig(
            start_iter=500,
            stop_iter=int(iterations * 0.5),
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
    )


def steps_to_targets(curve: list[dict], targets=TARGETS_DB) -> dict:
    """First evaluated step whose held-out foreground PSNR reaches each target.

    First-crossing, not best-value: held-out quality decays after its peak under DefaultStrategy,
    and a crossing is unaffected by that later decline.
    """
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
                "psnr_fg": hit["heldout_psnr_fg"],
            }
        )
    return crossings


def run_cell(
    arm: str,
    mode: str,
    budget: int,
    label: str,
    init: Gaussians3D,
    scene: SceneData,
    split,
    out: Path,
    args,
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

    final, _ = Trainer(
        train_config(mode, args.iterations, budget, args.rasterizer, args.device)
    ).train(scene, model, checkpoint_callback=checkpoint)
    final.save_ply(cell / "gaussians_final.ply")

    record = {
        "arm": arm,
        "mode": mode,
        "label": label,
        "budget": budget,
        "elapsed_seconds": time.perf_counter() - started,
        "init_metrics": init_metrics,
        "final_metrics": metrics(final),
        "final_n": final.n,
        "reached_budget": final.n >= budget,
        "steps_to_target": steps_to_targets(curve),
        "curve": curve,
    }
    (cell / "record.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def _ratio(treatment: dict | None, control: dict | None, iterations: int) -> tuple[float, bool]:
    """Step ratio treatment/control; a miss counts as > the full run so it cannot look good."""
    censored = treatment is None or control is None
    t = iterations if treatment is None else treatment["step"]
    c = iterations if control is None else control["step"]
    return (float("inf") if c == 0 else t / c), censored


def score(cells: dict, iterations: int) -> dict:
    """Preregistered gates, evaluated only on the primary cell."""
    primary = cells.get(f"{PRIMARY_MODE}:3")
    if primary is None:
        return {"note": f"primary cell {PRIMARY_MODE}:3 not run; no gate scored"}
    key = f"{PRIMARY_TARGET_DB:.1f}"
    reached = {a: primary[a]["steps_to_target"][key] for a in primary}

    out = {"target_db": PRIMARY_TARGET_DB, "cell": f"{PRIMARY_MODE}:3", "reached": reached}

    if "cover-iso" in reached and "ci" in reached:
        ratio, censored = _ratio(reached["cover-iso"], reached["ci"], iterations)
        out["A_cover_iso_vs_ci"] = {
            "step_ratio": ratio,
            "censored": censored,
            "verdict": (
                "G-C1 init reduces cost"
                if ratio <= COST_GATE_BETTER
                else "G-C3 reversal"
                if ratio >= COST_GATE_WORSE
                else "G-C2 no cost benefit"
            ),
        }
    if "ci" in reached and "random" in reached:
        ratio, censored = _ratio(reached["ci"], reached["random"], iterations)
        material = ratio <= BASELINE_GATE_MATERIAL or (
            reached["random"] is None and reached["ci"] is not None
        )
        out["B_ci_vs_random"] = {
            "step_ratio": ratio,
            "censored": censored,
            "random_never_reached": reached["random"] is None,
            "verdict": (
                "G-B1 lifting is material"
                if material
                else "G-B3 lifting hurts"
                if ratio >= COST_GATE_WORSE
                else "G-B2 lifting is not material"
                if ratio >= COST_GATE_BETTER
                else "directional, sub-threshold"
            ),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--downscale", type=int, default=DOWNSCALE)
    parser.add_argument("--n-init", type=int, default=N_INIT)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--rasterizer", default="gsplat")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument(
        "--cells",
        nargs="+",
        default=list(DEFAULT_CELLS),
        help="mode:budget_multiplier pairs, e.g. density:3 mcmc:3 density:1",
    )
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
    scene = split.scene
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
        "schema": "rtgs.gpu_init_cost_to_target.v1",
        "protocol": {"path": str(args.protocol), "sha256": _sha256(args.protocol)},
        "dataset": str(args.dataset),
        "downscale": args.downscale,
        "image_size": [int(scene.images[0].shape[1]), int(scene.images[0].shape[0])],
        "n_init": args.n_init,
        "iterations": args.iterations,
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
