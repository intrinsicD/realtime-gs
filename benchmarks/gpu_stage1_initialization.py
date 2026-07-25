#!/usr/bin/env python3
"""Stage 1 on GPU: does the cover-consistent initialization survive the production stack?

Everything established so far is a downscale-32 CPU result using the reference rasterizer and the
classic density controller.  Three things could break it at production scale, and this harness is
built to find out:

* **the renderer** — gsplat's antialiasing filter is not the CPU reference's 0.3 px² floor, and
  the whole scale-suppression mechanism may not exist there (run ``gpu_dilation_probe.py`` first);
* **the resolution** — at downscale 32 the object spans ~40 px, so the silhouette band is a large
  fraction of it; the stage-0 decomposition must be re-measured here;
* **the topology controller** — gsplat's Default strategy differs from the CPU classic one, and
  MCMC/relocation replaces clone/split outright, which could erase the clone-in-place failure.

Arms differ only in quaternions, log-scales, and opacity; means, SH/colour, and count are
bit-identical.  The preregistered treatment is ``cover-iso`` — cover extent, **opacity left at the
initializer's own value** — because the CPU screen measured the derived opacity to buy nothing
downstream (+0.18% AUC, −0.046 dB) while causing the silhouette leak (initial outside alpha 0.0437
without it, 0.1851 with it).  ``cover-surfel`` and ``cover-surfel-op`` are labelled secondaries.

Frozen protocol: ``benchmarks/results/20260725_gpu_stage1_initialization_PREREG.md``.
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
    from benchmarks.residual_decomposition import evaluate_model as decompose_model
    from benchmarks.scene_builder import build_scene_at, train_inputs_at
except ModuleNotFoundError:  # direct invocation
    from residual_decomposition import evaluate_model as decompose_model  # type: ignore
    from scene_builder import build_scene_at, train_inputs_at  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00009/gaussians2d"
DEFAULT_OUT = ROOT / "runs/gpu_stage1_initialization"
DEFAULT_PROTOCOL = ROOT / "benchmarks/results/20260725_gpu_stage1_initialization_PREREG.md"

ARMS = ("ci", "cover-iso", "cover-surfel", "cover-surfel-op")
PREREGISTERED_TREATMENT = "cover-iso"
# ``density-classic`` uses the CPU classic controller so the whole path can be
# smoke-tested without a GPU; the frozen protocol uses ``density`` (gsplat Default).
MODES = ("fixed", "density", "mcmc", "density-classic")
DOWNSCALE = 4
N_INIT = 5_000
ITERATIONS = 7_000
EVAL_EVERY = 500
BUDGET_MULTIPLIER = 3  # matched hard cap = BUDGET_MULTIPLIER * N_INIT
DECISION_MARGIN_DB = 0.15
SEED = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_initializations(dataset: CompactDataset, n_init: int) -> tuple[dict, dict]:
    """Beam Fusion once, then four covariance/opacity arms over identical means and colours."""
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
    shared = {"resolution_floor": floor, "frames": frames}

    cover_iso = reconcile_covariances(
        ci,
        SurfelInitConfig(isotropic=True, use_coverage_opacity=False, fixed_opacity=0.10),
        **shared,
    )
    cover_surfel = reconcile_covariances(
        ci, SurfelInitConfig(use_coverage_opacity=False, fixed_opacity=0.10), **shared
    )
    cover_surfel_op = reconcile_covariances(ci, SurfelInitConfig(), **shared)

    inits = {
        "ci": ci,
        "cover-iso": cover_iso.gaussians,
        "cover-surfel": cover_surfel.gaussians,
        "cover-surfel-op": cover_surfel_op.gaussians,
    }
    for name, arm in inits.items():
        if arm.n != ci.n or not torch.equal(arm.means, ci.means) or not torch.equal(arm.sh, ci.sh):
            raise RuntimeError(f"{name} changed a frozen field")

    diagnostics = {
        "beam_elapsed_seconds": elapsed,
        "n_gaussians": ci.n,
        "n_contributor_links": int(beam.contributor_view_indices.numel()),
        "surfel_rule": cover_surfel_op.diagnostics,
        "cover_iso_rule": cover_iso.diagnostics,
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


def run_cell(
    arm: str,
    mode: str,
    init: Gaussians3D,
    scene: SceneData,
    split,
    out: Path,
    args,
) -> dict:
    cell = out / mode / arm
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
        curve.append(
            {
                "step": step,
                "n": snapshot.n,
                **{f"heldout_{k}": v for k, v in values["heldout"].items()},
            }
        )
        print(
            f"[{mode}/{arm}] step {step} n={snapshot.n} "
            f"held_fg={values['heldout'].get('psnr_fg', float('nan')):.3f}dB "
            f"({time.perf_counter() - started:.0f}s)",
            flush=True,
        )

    budget = args.budget or BUDGET_MULTIPLIER * init.n
    final, _ = Trainer(
        train_config(mode, args.iterations, budget, args.rasterizer, args.device)
    ).train(scene, model, checkpoint_callback=checkpoint)
    final.save_ply(cell / "gaussians_final.ply")

    # Stage-0 decomposition at this resolution: holes vs appearance vs boundary vs leak.
    decomposition = decompose_model(
        final.to("cpu"),
        scene,
        split.heldout_local,
        boundary_radius=args.boundary_radius,
        thresholds=(0.1, 0.3, 0.5),
    )
    record = {
        "arm": arm,
        "mode": mode,
        "budget": budget,
        "elapsed_seconds": time.perf_counter() - started,
        "init_metrics": init_metrics,
        "final_metrics": metrics(final),
        "final_n": final.n,
        "reached_budget": final.n >= budget,
        "curve": curve,
        "residual_decomposition": decomposition,
    }
    (cell / "record.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--downscale", type=int, default=DOWNSCALE)
    parser.add_argument("--n-init", type=int, default=N_INIT)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--budget", type=int, default=0, help="0 = 3x n_init")
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--rasterizer", default="gsplat")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["fixed", "density"])
    args = parser.parse_args()
    if not args.protocol.is_file():
        raise FileNotFoundError(f"frozen protocol missing: {args.protocol}")

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

    modes: dict[str, dict] = {}
    for mode in args.modes:
        modes[mode] = {}
        for arm in args.arms:
            modes[mode][arm] = run_cell(arm, mode, initializations[arm], scene, split, out, args)

    decision = {}
    for mode, arms in modes.items():
        if "ci" in arms and PREREGISTERED_TREATMENT in arms:
            control = arms["ci"]["final_metrics"]["heldout"]["psnr_fg"]
            treatment = arms[PREREGISTERED_TREATMENT]["final_metrics"]["heldout"]["psnr_fg"]
            decision[mode] = {
                "treatment": PREREGISTERED_TREATMENT,
                "delta_psnr_fg_db": treatment - control,
                "treatment_final_n": arms[PREREGISTERED_TREATMENT]["final_n"],
                "ci_final_n": arms["ci"]["final_n"],
                "passes": (treatment - control) >= DECISION_MARGIN_DB
                and arms[PREREGISTERED_TREATMENT]["final_n"] <= arms["ci"]["final_n"],
            }

    summary = {
        "schema": "rtgs.gpu_stage1_initialization.v1",
        "protocol": {"path": str(args.protocol), "sha256": _sha256(args.protocol)},
        "dataset": str(args.dataset),
        "downscale": args.downscale,
        "image_size": [int(scene.images[0].shape[1]), int(scene.images[0].shape[0])],
        "n_init": args.n_init,
        "iterations": args.iterations,
        "rasterizer": args.rasterizer,
        "device": args.device,
        "seed": SEED,
        "preregistered_treatment": PREREGISTERED_TREATMENT,
        "decision_margin_db": DECISION_MARGIN_DB,
        "train_views": [scene.view_names[i] for i in split.train_local],
        "heldout_views": [scene.view_names[i] for i in split.heldout_local],
        "initialization_diagnostics": diagnostics,
        "decision": decision,
        "modes": modes,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(decision, indent=2), flush=True)
    print(f"[written] {out / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
