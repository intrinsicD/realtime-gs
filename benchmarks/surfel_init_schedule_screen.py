#!/usr/bin/env python3
"""ADR-XXXX / ADR-YYYY development screen: init parameters and the schedule that keeps them.

Two screens over the same scenes and the same evaluation surface:

**Screen A -- init parameters (ADR-XXXX, PREREG E7/E8).**  Fixed topology, so the only thing that
varies is what the initialization *is*.  Arms:

| arm | what changes |
| --- | --- |
| ``ci`` | beam fusion as-is: covariance-intersection localization covariance, fixed alpha 0.10 |
| ``beam-cover`` | the incumbent cover-consistent reconciliation (``rtgs.lift.surfel_init``) |
| ``surfel-rho{0.05,0.1,0.25}`` | ADR-XXXX §1-§4 with the §3 transmittance solve on (E7 rho sweep) |
| ``surfel-rho0.1-fixedalpha`` | the same construction with §3 **off** (E8 contrast) |
| ``random-matched`` | uniform means in the beam bounding box, train-view colour (control) |

**Screen B -- the schedule (ADR-YYYY, PREREG E9/E12).**  A 2x3x2 factorial crossed with the
initialization, which is the whole point: the ADR's thesis is that the right schedule is
*init-conditioned*, and an interaction cannot be measured with one init.

| factor | levels |
| --- | --- |
| init | the Screen-A winner-independent anchor ``surfel-rho0.1`` vs ``random-matched`` |
| growth | ``none`` (fixed topology), ``classic`` (vanilla ADC), ``init-preserving`` (ADR-YYYY) |
| trust | off, on (ADR-YYYY §5) |

**Evidence status: DEVELOPMENT-ONLY.**  Stage frames ``frame_00008``/``frame_00009`` are assigned
"debugging and hypothesis generation only" by §1.2 of the master protocol: they are
outcome-exposed, because earlier experiments in this repository selected variants on them.
Nothing here may enter ``README.md``, ``docs/``, or ``ara/logic/claims.md`` as a method claim.
These frames are used because they are the **only** compact bundles in the repository that carry
packed alpha, and ADR-XXXX §3 solves against exactly that alpha: the screen cannot be run
anywhere else at all.

Splits are the mechanical, outcome-independent rule already frozen in
``benchmarks/init_value_masked_screen.py`` and are imported rather than re-derived.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from benchmarks.init_value_dev_screen import (
    Teacher,
    _sha256,
    build_random_pair,
    build_teachers,
)
from benchmarks.init_value_masked_screen import (
    REPORT_ONLY_SEALED,
    SCENE_PATHS,
    VALIDATION,
    build_masks,
    derive_splits,
    evaluate_views_masked,
    make_masked_scene,
)

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.compact_views import CompactDataset
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.beam_fusion import BeamFusionConfig, fuse_gaussian_beams
from rtgs.lift.surfel_init import (
    SurfelInitConfig,
    beam_contributor_footprints,
    estimate_local_surface_frames,
    reconcile_covariances,
)
from rtgs.lift.surfel_lift import AlphaTarget, SurfelLiftConfig, build_surfel_lift
from rtgs.optim.density import DensityConfig
from rtgs.optim.init_density import InitPreservingConfig
from rtgs.optim.init_trust import TrustConfig
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.base import get_rasterizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "runs/surfel_init_schedule_screen"

RHOS = (0.05, 0.1, 0.25)
ANCHOR_RHO = 0.1
INIT_ARMS = (
    "ci",
    "beam-cover",
    "surfel-rho0.05",
    "surfel-rho0.1",
    "surfel-rho0.25",
    "surfel-rho0.1-fixedalpha",
    "random-matched",
)
SCHEDULE_INITS = ("surfel-rho0.1", "random-matched")
GROWTH_LEVELS = ("none", "classic", "init-preserving")
TRUST_LEVELS = (False, True)

N_INIT = 2400
ITERATIONS = 1500
EVAL_EVERY = 100
DOWNSCALE = 4
SEEDS = (26001, 26002, 26003)
SCHEDULE_SEEDS = (26001, 26002)
MATERIALITY_DB = 0.25
COMPARATOR = "ci"
# The primitive budget for the growth arms.  Screen A is fixed topology at N_INIT; Screen B lets
# the growth channels spend up to this cap, which is the E4 machinery's variable, not a tuning.
MAX_GAUSSIANS = 6000


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


# --------------------------------------------------------------------------------------------
# Initializations
# --------------------------------------------------------------------------------------------


@dataclass
class InitArm:
    gaussians: Gaussians3D
    seconds: float
    coherence: torch.Tensor | None
    diagnostics: dict


def _train_inputs(dataset: CompactDataset, train_ids: tuple[str, ...]) -> ReconstructionInputs:
    complete = dataset.to_reconstruction_inputs()
    names = complete.view_names
    index = [names.index(view_id) for view_id in train_ids]
    return ReconstructionInputs(
        observations=[complete.observations[i] for i in index],
        cameras=[complete.cameras[i] for i in index],
        view_names=[names[i] for i in index],
        bounds_hint=complete.bounds_hint,
        name=f"{complete.name}-train",
    )


def _alpha_targets(dataset: CompactDataset, train_ids: tuple[str, ...]) -> list[AlphaTarget]:
    """The Stage-1 alpha composites ADR-XXXX §3 solves against, addressed by local view index."""
    by_id = {view.view_id: view for view in dataset.views}
    targets = []
    for local, view_id in enumerate(train_ids):
        view = by_id[view_id]
        if view.alpha is None:
            raise ValueError(f"view {view_id} carries no packed alpha; ADR-XXXX §3 is unavailable")
        targets.append(
            AlphaTarget(
                view_index=local,
                alpha=view.alpha.crop_mask().to(torch.float64),
                origin=view.alpha.origin,
            )
        )
    return targets


def build_structured_arms(
    dataset: CompactDataset,
    train_ids: tuple[str, ...],
    *,
    n_init: int,
    arms: tuple[str, ...],
    pixel_stride: int,
) -> tuple[dict[str, InitArm], Gaussians3D]:
    """Every arm whose construction is deterministic, built once and shared across seeds.

    Beam fusion and the ADR-XXXX lift are seed-independent by construction (ADR-XXXX A3), so
    rebuilding them per seed would only measure the same number three times.  The random control
    is the one arm that does depend on the seed and is built separately.
    """
    inputs = _train_inputs(dataset, train_ids)
    extent = float(dataset.bounds_hint[1])
    config = BeamFusionConfig(
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
    beam = fuse_gaussian_beams(inputs, config)
    fusion_seconds = time.perf_counter() - started
    ci = beam.gaussians.detach()

    out: dict[str, InitArm] = {}
    if "ci" in arms:
        out["ci"] = InitArm(ci, fusion_seconds, None, {"n": ci.n})

    if "beam-cover" in arms:
        started = time.perf_counter()
        floor = beam_contributor_footprints(inputs, beam)
        frames = estimate_local_surface_frames(ci.means, SurfelInitConfig())
        cover = reconcile_covariances(
            ci,
            SurfelInitConfig(isotropic=True, use_coverage_opacity=False, fixed_opacity=0.10),
            resolution_floor=floor,
            frames=frames,
        )
        out["beam-cover"] = InitArm(
            cover.gaussians,
            fusion_seconds + (time.perf_counter() - started),
            None,
            cover.diagnostics,
        )

    targets = None
    for rho in RHOS:
        name = f"surfel-rho{rho}"
        if name not in arms:
            continue
        if targets is None:
            targets = _alpha_targets(dataset, train_ids)
        started = time.perf_counter()
        result = build_surfel_lift(
            inputs,
            beam,
            SurfelLiftConfig(rho=rho, pixel_stride=pixel_stride),
            alpha_targets=targets,
        )
        print(f"  [init] {name} in {time.perf_counter() - started:.0f}s", flush=True)
        out[name] = InitArm(
            result.gaussians,
            fusion_seconds + (time.perf_counter() - started),
            result.frames.coherence.to(torch.float32),
            result.diagnostics,
        )

    fixed = f"surfel-rho{ANCHOR_RHO}-fixedalpha"
    if fixed in arms:
        started = time.perf_counter()
        result = build_surfel_lift(
            inputs, beam, SurfelLiftConfig(rho=ANCHOR_RHO, solve_opacity=False), alpha_targets=None
        )
        out[fixed] = InitArm(
            result.gaussians,
            fusion_seconds + (time.perf_counter() - started),
            result.frames.coherence.to(torch.float32),
            result.diagnostics,
        )
    return out, ci


def build_random_arm(
    reference: Gaussians3D,
    teachers: dict[str, Teacher],
    train_ids: tuple[str, ...],
    *,
    n_init: int,
    seed: int,
) -> InitArm:
    matched, _ = build_random_pair(reference, n_init, seed, teachers, train_ids)
    # The control reads the beam bounding box, so its reported time excludes deriving it: a
    # lower bound that favours the control.
    return InitArm(matched, 0.0, None, {"n": matched.n})


# --------------------------------------------------------------------------------------------
# Pre-training audit (ADR-XXXX A2): does the lifted state reproduce the Stage-1 render?
# --------------------------------------------------------------------------------------------


def audit_init(
    model: Gaussians3D,
    scene,
    renderer,
    train_local: list[int],
) -> dict:
    """Fitted-view agreement with Stage 1, before any optimizer has touched the state.

    ``alpha_iou`` is the ADR §A2 gate: the intersection over union of the rendered alpha and the
    ground-truth foreground at 0.5.  ``train_psnr`` is the fitted-view lift efficiency: how close
    the lifted 3D state renders to the 2D fit it was built from.
    """
    device = model.means.device
    per_view = {}
    with torch.no_grad():
        for index in train_local:
            camera = scene.cameras[index].to(device)
            mask = scene.masks[index].to(device) > 0.5
            out = renderer.render(model, camera)
            predicted = out.alpha.reshape(mask.shape) > 0.5
            intersection = float((predicted & mask).sum())
            union = float((predicted | mask).sum())
            per_view[scene.view_names[index]] = {
                "alpha_iou": intersection / union if union else 0.0,
                "alpha_mean": float(out.alpha.mean()),
            }
    return {
        "per_view": per_view,
        "alpha_iou": _median([v["alpha_iou"] for v in per_view.values()]),
        "alpha_mean": _median([v["alpha_mean"] for v in per_view.values()]),
    }


# --------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------


def train_config(
    *,
    iterations: int,
    rasterizer: str,
    device: str,
    seed: int,
    growth: str,
    trust: bool,
    coherence: torch.Tensor | None,
) -> TrainConfig:
    config = TrainConfig(
        iterations=iterations,
        rasterizer=rasterizer,
        device=device,
        densify=growth != "none",
        density_strategy="classic" if growth == "none" else growth,
        eval_every=EVAL_EVERY,
        target_sh_degree=3,
        use_masks=True,
        random_background=False,
        checkpoint_policy="final",
        seed=seed,
    )
    if growth == "classic":
        config.density = DensityConfig(max_gaussians=MAX_GAUSSIANS)
    elif growth == "init-preserving":
        config.init_density = InitPreservingConfig(max_gaussians=MAX_GAUSSIANS)
    if trust:
        config.trust = TrustConfig()
        config.init_coherence = coherence
    return config


def run_cell(
    *,
    label: str,
    arm: str,
    init: InitArm,
    scene,
    train_local: list[int],
    eval_local: list[int],
    seed: int,
    growth: str,
    trust: bool,
    out: Path,
    args,
) -> dict:
    key = f"{label}/{arm}/{growth}/{'trust' if trust else 'notrust'}/{seed}"
    cell = out / "cells" / key.replace("/", "__")
    cell.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    renderer = get_rasterizer(args.rasterizer, device=device)
    model = init.gaussians.to(device)
    n_initial = model.n

    torch.manual_seed(seed)
    audit = audit_init(model, scene, renderer, train_local)
    step0 = evaluate_views_masked(scene, model, renderer, eval_local)
    curve = [
        {
            "step": 0,
            "n": n_initial,
            "seconds": init.seconds,
            "psnr": step0["psnr"],
            "ssim": step0["ssim"],
        }
    ]
    if args.save_ply:
        model.save_ply(cell / "gaussians_init.ply")

    started = time.perf_counter()
    observer_seconds = 0.0

    def checkpoint(snapshot: Gaussians3D, step: int) -> None:
        nonlocal observer_seconds
        elapsed = time.perf_counter() - started - observer_seconds
        observer_started = time.perf_counter()
        values = evaluate_views_masked(scene, snapshot, renderer, eval_local)
        curve.append(
            {
                "step": step,
                "n": snapshot.n,
                "seconds": init.seconds + elapsed,
                "psnr": values["psnr"],
                "ssim": values["ssim"],
            }
        )
        observer_seconds += time.perf_counter() - observer_started

    config = train_config(
        iterations=args.iterations,
        rasterizer=args.rasterizer,
        device=args.device,
        seed=seed,
        growth=growth,
        trust=trust,
        coherence=init.coherence,
    )
    final, history = Trainer(config).train(scene, model, checkpoint_callback=checkpoint)
    if growth == "none" and final.n != n_initial:
        raise RuntimeError(f"fixed topology violated in {key}: {n_initial} -> {final.n}")
    if args.save_ply:
        final.save_ply(cell / "gaussians_final.ply")

    final_eval = evaluate_views_masked(scene, final.to(device), renderer, eval_local)
    train_eval = evaluate_views_masked(scene, final.to(device), renderer, train_local)
    record = {
        "label": label,
        "arm": arm,
        "growth": growth,
        "trust": trust,
        "seed": seed,
        "n_initial": n_initial,
        "n_final": int(final.n),
        "init_seconds": init.seconds,
        "end_to_end_seconds": curve[-1]["seconds"],
        "init_alpha_iou": audit["alpha_iou"],
        "init_alpha_mean": audit["alpha_mean"],
        "q_init": step0["psnr"],
        "q_final": final_eval["psnr"],
        "ssim_final": final_eval["ssim"],
        "train_psnr_final": train_eval["psnr"],
        "per_view_final": final_eval["per_view"],
        "density_stats": history.get("density_stats"),
        "trust_schedule": history.get("trust_schedule"),
        "peak_vram_gb": history.get("peak_vram_gb"),
        "curve": curve,
    }
    (cell / "record.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(
        f"[{key}] IoU0={record['init_alpha_iou']:.3f} Q0={record['q_init']:.2f} "
        f"Q={record['q_final']:.2f}dB n={record['n_final']} "
        f"t={record['end_to_end_seconds']:.0f}s",
        flush=True,
    )
    return record


# --------------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------------


def contrast(records: dict[str, dict[int, dict]], comparator: str, seeds) -> dict:
    """Paired per-seed contrasts against ``comparator``; no pooling across scenes."""
    out: dict[str, dict] = {}
    if comparator not in records:
        return out
    for arm, by_seed in records.items():
        if arm == comparator:
            continue
        deltas = [
            by_seed[s]["q_final"] - records[comparator][s]["q_final"]
            for s in seeds
            if s in by_seed and s in records[comparator]
        ]
        if not deltas:
            continue
        wins = sum(1 for d in deltas if d >= MATERIALITY_DB)
        out[arm] = {
            "delta_db": deltas,
            "delta_median": _median(deltas),
            "material_wins": wins,
            "seeds": len(deltas),
            "verdict": (
                "material gain"
                if wins >= max(1, len(deltas) - 1) and _median(deltas) >= MATERIALITY_DB
                else "material loss"
                if _median(deltas) <= -MATERIALITY_DB
                else f"inside the {MATERIALITY_DB} dB band"
            ),
        }
    return out


def interaction(schedule_records: dict) -> dict:
    """The E12 question in one number per factor: does the schedule effect depend on the init?

    ``effect[init][level]`` is that level's median held-out gain over ``growth=none, trust=off``
    for that init; the interaction contrast is the difference of those effects between the two
    inits.  A non-zero interaction is the ADR-YYYY thesis; a zero one refutes it on this scene.
    """
    effects: dict[str, dict[str, float]] = {}
    for init in schedule_records:
        baseline = schedule_records[init].get(("none", False))
        if baseline is None:
            continue
        effects[init] = {}
        for (growth, trust), values in schedule_records[init].items():
            if not values or not baseline:
                continue
            seeds = sorted(set(values) & set(baseline))
            if not seeds:
                continue
            effects[init][f"{growth}/{'trust' if trust else 'notrust'}"] = _median(
                [values[s]["q_final"] - baseline[s]["q_final"] for s in seeds]
            )
    contrasts = {}
    if len(effects) == 2:
        good, control = SCHEDULE_INITS
        if good in effects and control in effects:
            for level in effects[good]:
                if level in effects[control]:
                    contrasts[level] = effects[good][level] - effects[control][level]
    return {"effects": effects, "interaction_contrast_db": contrasts}


def environment() -> dict:
    def git(*cmd: str) -> str:
        try:
            return subprocess.check_output(["git", *cmd], cwd=ROOT, text=True).strip()
        except Exception:  # pragma: no cover - provenance is best effort
            return "unavailable"

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
        try:
            import gsplat

            info["gsplat"] = gsplat.__version__
        except Exception:  # pragma: no cover
            info["gsplat"] = "unavailable"
    return info


# --------------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scenes", nargs="+", default=list(SCENE_PATHS))
    parser.add_argument("--arms", nargs="+", default=list(INIT_ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--schedule-seeds", nargs="+", type=int, default=list(SCHEDULE_SEEDS))
    parser.add_argument("--n-init", type=int, default=N_INIT)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--downscale", type=int, default=DOWNSCALE)
    parser.add_argument(
        "--pixel-stride",
        type=int,
        default=4,
        help="ADR-XXXX §3 pixel subsample stride; 4 is the ADR default",
    )
    parser.add_argument("--rasterizer", default="gsplat")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher-device", default="cuda")
    parser.add_argument("--skip-schedule", action="store_true")
    parser.add_argument("--save-ply", action="store_true", default=True)
    parser.add_argument("--no-save-ply", dest="save_ply", action="store_false")
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()

    screen_a: dict[str, dict] = {}
    screen_b: dict[str, dict] = {}
    scene_meta: dict[str, dict] = {}

    for scene_key in args.scenes:
        dataset_path = ROOT / SCENE_PATHS[scene_key]
        dataset = CompactDataset.load(dataset_path, device="cpu")
        splits = derive_splits(dataset)
        train_ids = splits["train3"]
        needed = tuple(splits["train8"]) + tuple(VALIDATION)
        print(f"[scene] {scene_key}: {len(needed)} teachers", flush=True)
        teachers = build_teachers(
            dataset, needed, downscale=args.downscale, device=args.teacher_device
        )
        masks = build_masks(dataset, teachers, downscale=args.downscale)
        scene, train_local, eval_local = make_masked_scene(
            teachers, masks, train_ids, VALIDATION, dataset.bounds_hint, f"{scene_key}-v3"
        )
        scene_meta[scene_key] = {
            "dataset": str(dataset_path.relative_to(ROOT)),
            "manifest_sha256": _sha256(dataset_path / "manifest.json"),
            "splits": {k: list(v) if isinstance(v, tuple) else v for k, v in splits.items()},
            "validation": list(VALIDATION),
            "report_only_sealed": list(REPORT_ONLY_SEALED),
            "image_size": [
                teachers[needed[0]].camera.width,
                teachers[needed[0]].camera.height,
            ],
        }

        # ---- Screen A -------------------------------------------------------------------
        print(f"[screen A] {scene_key}: building initializations", flush=True)
        structured, reference = build_structured_arms(
            dataset,
            train_ids,
            n_init=args.n_init,
            arms=tuple(args.arms),
            pixel_stride=args.pixel_stride,
        )
        arm_cache: dict[int, dict[str, InitArm]] = {}
        records: dict[str, dict[int, dict]] = {arm: {} for arm in args.arms}
        for seed in args.seeds:
            arms = dict(structured)
            if "random-matched" in args.arms:
                arms["random-matched"] = build_random_arm(
                    reference, teachers, train_ids, n_init=args.n_init, seed=seed
                )
            arm_cache[seed] = arms
            for arm in args.arms:
                records[arm][seed] = run_cell(
                    label=scene_key,
                    arm=arm,
                    init=arms[arm],
                    scene=scene,
                    train_local=train_local,
                    eval_local=eval_local,
                    seed=seed,
                    growth="none",
                    trust=False,
                    out=out,
                    args=args,
                )
        screen_a[scene_key] = {
            "records": {
                arm: {str(s): r for s, r in by_seed.items()} for arm, by_seed in records.items()
            },
            "init_diagnostics": {
                arm: arm_cache[args.seeds[0]][arm].diagnostics for arm in args.arms
            },
            "contrast_vs_ci": contrast(records, COMPARATOR, args.seeds),
            "init_alpha_iou": {
                arm: _median([r["init_alpha_iou"] for r in by_seed.values()])
                for arm, by_seed in records.items()
                if by_seed
            },
        }

        # ---- Screen B -------------------------------------------------------------------
        if args.skip_schedule:
            continue
        print(f"[screen B] {scene_key}: schedule factorial", flush=True)
        schedule: dict[str, dict] = {}
        for init_name in SCHEDULE_INITS:
            if init_name not in args.arms:
                continue
            schedule[init_name] = {}
            for growth in GROWTH_LEVELS:
                for trust in TRUST_LEVELS:
                    if growth == "none" and trust is False:
                        # Already measured by Screen A; reuse rather than re-run.
                        schedule[init_name][(growth, trust)] = {
                            s: records[init_name][s] for s in args.schedule_seeds
                        }
                        continue
                    by_seed = {}
                    for seed in args.schedule_seeds:
                        by_seed[seed] = run_cell(
                            label=scene_key,
                            arm=init_name,
                            init=arm_cache[seed][init_name],
                            scene=scene,
                            train_local=train_local,
                            eval_local=eval_local,
                            seed=seed,
                            growth=growth,
                            trust=trust,
                            out=out,
                            args=args,
                        )
                    schedule[init_name][(growth, trust)] = by_seed
        screen_b[scene_key] = {
            "records": {
                init_name: {
                    f"{growth}/{'trust' if trust else 'notrust'}": {
                        str(s): r for s, r in by_seed.items()
                    }
                    for (growth, trust), by_seed in levels.items()
                }
                for init_name, levels in schedule.items()
            },
            "analysis": interaction(schedule),
        }

    summary = {
        "schema": "rtgs.surfel_init_schedule_screen.v1",
        "status": "development-only; Stage frames are outcome-exposed (protocol §1.2)",
        "adrs": [
            "docs/ADR-XXXX-surfel-lift.md",
            "docs/ADR-YYYY-init-preserving-densification.md",
        ],
        "config": {
            "scenes": args.scenes,
            "arms": args.arms,
            "seeds": args.seeds,
            "schedule_seeds": args.schedule_seeds,
            "n_init": args.n_init,
            "iterations": args.iterations,
            "downscale": args.downscale,
            "pixel_stride": args.pixel_stride,
            "max_gaussians": MAX_GAUSSIANS,
            "growth_levels": list(GROWTH_LEVELS),
            "trust_levels": list(TRUST_LEVELS),
            "materiality_db": MATERIALITY_DB,
            "comparator": COMPARATOR,
            "metric": "median per-view foreground-weighted PSNR on held-out views "
            "(target: each view's own compact 2D fit composited on black)",
        },
        "scenes": scene_meta,
        "screen_a": screen_a,
        "screen_b": screen_b,
        "environment": environment(),
        "wall_clock_seconds": time.perf_counter() - started_all,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"[done] {out / 'summary.json'} ({summary['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
