#!/usr/bin/env python3
"""Initialization Value Program — DEVELOPMENT screen (not the confirmatory phase).

This runs the closest executable approximation of section 7.1 of
``benchmarks/results/20260725_init_value_program_PREREG.md`` on the two frozen Karate validation
frames.  It is **development evidence only**.  Three deviations from the frozen screen are forced
by what this repository actually contains, and each is recorded in the summary:

1. **No ``colmap-sfm-fixed-pose`` arm.**  ``dataset/`` holds compact RGB-free 2D-gaussian fits;
   the original Karate RGB is absent (its hashes survive in
   ``benchmarks/results/20260716_compact_point_training_SEAL.json``).  Section 3.3 is fail-closed
   about this: a missing comparator is reported as baseline unavailability, never converted into a
   win.  Every comparison here is therefore against the repository's own default packaging
   (``ci``) and against random controls.
2. **``Qval`` is measured against the compact 2D self-reconstruction, not source RGB.**  Section
   8.1 names that surface the *2D self-reconstruction baseline*.  Held-out quality here means
   "reproduces the unseen view's own 2D fit", which is a proxy for, and an upper-bounded stand-in
   of, held-out RGB PSNR.  Every previous GPU experiment in this repository uses the same surface,
   so the numbers are comparable to them and to nothing else.
3. **Full-canvas metrics.**  Karate carries no packed alpha (verified: 0/30 and 0/32 views), which
   section 7.1 already anticipates.

Two additions beyond the frozen screen, both testing review finding B1
(``20260725_init_value_program_PREREG_REVIEW_02_FAIL.md``): the random control is split into
``random-grey`` (no photometric prior, the historical control) and ``random-matched`` (identical
means, colours sampled from the training teachers).  Their contrast measures how much of any
"initialization information" gain is really an appearance-initialization gain — the confound that
made the single unmatched random arm unusable as an information control.

Report-only views (``C1001,C1004,C1005``) are NOT opened by this harness.  Section 2.2 opens them
once, after a validation decision is frozen; this run does not freeze one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.metrics import psnr, ssim
from rtgs.core.sh import rgb_to_sh
from rtgs.data.compact_views import CompactDataset
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.data.scene import SceneData
from rtgs.lift.base import bilinear_sample
from rtgs.lift.beam_fusion import BeamFusionConfig, fuse_gaussian_beams
from rtgs.lift.surfel_init import (
    SurfelInitConfig,
    beam_contributor_footprints,
    estimate_local_surface_frames,
    reconcile_covariances,
)
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.base import get_rasterizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "runs/init_value_dev_screen"
DEFAULT_PROTOCOL = ROOT / "benchmarks/results/20260725_init_value_program_PREREG.md"
DEFAULT_REVIEW = ROOT / "benchmarks/results/20260725_init_value_program_PREREG_REVIEW_02_FAIL.md"

# Section 2.2, verbatim.  Views are addressed by ID, never by manifest index: a per-file hash
# cannot detect a role mis-assignment, so the IDs are the frozen object (review finding A2).
SCENES: dict[str, dict] = {
    "frame_00005": {
        "path": "dataset/karate/frame_00005/gaussians2d",
        "train3": ("C0005", "C0010", "C0021"),
        "train8_extra": ("C0022", "C0039", "C0025", "C0030", "C0037"),
        "validation": ("C0031", "C1000", "C1002"),
        "report_only_sealed": ("C1001", "C1004", "C1005"),
    },
    "frame_00060": {
        "path": "dataset/karate/frame_00060/gaussians2d",
        "train3": ("C0000", "C0037", "C0020"),
        "train8_extra": ("C0001", "C0024", "C0021", "C0039", "C0022"),
        "validation": ("C0031", "C1000", "C1002"),
        "report_only_sealed": ("C1001", "C1004", "C1005"),
    },
}

ARMS = ("beam-cover", "ci", "random-matched", "random-grey")
VIEWSETS = ("v3", "v8")
COUNTS = (2400, 5000)
SEEDS = (26001, 26002, 26003)

ITERATIONS = 1500
EVAL_EVERY = 100  # checkpoints j in {0,100,...,1500}
DOWNSCALE = 4
ANCHOR = ("v3", 2400)  # section 7.1 anchor
TAU_MARGIN_DB = 0.25  # section 4.2 materiality margin
TAU_PERSIST_DB = 0.10  # section 4.2 persistence band
TIME_RATIO_GATE = 0.75  # section 5.1.4
COMPARATOR = "ci"  # stands in for the unavailable colmap-sfm-fixed-pose arm


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _mad(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    med = statistics.median(values)
    return float(statistics.median([abs(v - med) for v in values]))


# --------------------------------------------------------------------------------------------
# Scene construction: exact GaussianObservationField queries at downscaled pixel centres.
# No source RGB is read; the compact-bundle contract stays intact.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Teacher:
    view_id: str
    image: torch.Tensor  # (H, W, 3) in [0,1]
    camera: Camera


def build_teachers(
    dataset: CompactDataset,
    view_ids: tuple[str, ...],
    *,
    downscale: int,
    device: str,
    query_chunk: int = 8192,
    component_chunk: int = 2048,
) -> dict[str, Teacher]:
    """Render each requested view's own 2D fit at ``downscale``; this is the supervision surface."""
    by_id = {view.view_id: view for view in dataset.views}
    teachers: dict[str, Teacher] = {}
    for view_id in view_ids:
        if view_id not in by_id:
            raise KeyError(f"view {view_id} is not in {dataset.name}")
        view = by_id[view_id]
        observation = view.observation.to(device)
        fit_x, fit_y, width, height = view.observation.fit_window
        ds_width, ds_height = width // downscale, height // downscale
        if ds_width <= 0 or ds_height <= 0:
            raise ValueError(f"downscale {downscale} collapses view {view_id}")
        ys = (torch.arange(ds_height, dtype=torch.float32, device=device) + 0.5) * downscale + fit_y
        xs = (torch.arange(ds_width, dtype=torch.float32, device=device) + 0.5) * downscale + fit_x
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        xy = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        colors = []
        for start in range(0, xy.shape[0], query_chunk):
            colors.append(
                observation.query(
                    xy[start : start + query_chunk], component_chunk=component_chunk
                ).color
            )
        image = torch.cat(colors).reshape(ds_height, ds_width, 3).clamp(0, 1).cpu().float()
        camera = view.camera
        teachers[view_id] = Teacher(
            view_id=view_id,
            image=image,
            camera=Camera(
                fx=camera.fx / downscale,
                fy=camera.fy / downscale,
                cx=(camera.cx - fit_x) / downscale,
                cy=(camera.cy - fit_y) / downscale,
                width=ds_width,
                height=ds_height,
                R=camera.R,
                t=camera.t,
            ),
        )
        print(f"  [teacher] {view_id} {ds_width}x{ds_height}", flush=True)
    return teachers


def make_scene(
    teachers: dict[str, Teacher],
    train_ids: tuple[str, ...],
    eval_ids: tuple[str, ...],
    bounds_hint,
    name: str,
) -> tuple[SceneData, list[int], list[int]]:
    """Scene ordered train-first; only ``train_ids`` are ever visible to an initializer."""
    order = list(train_ids) + list(eval_ids)
    scene = SceneData(
        images=[teachers[v].image for v in order],
        cameras=[teachers[v].camera for v in order],
        view_names=list(order),
        masks=None,  # Karate carries no packed alpha; section 7.1 full-canvas Qval
        train_indices=list(range(len(train_ids))),
        test_indices=list(range(len(train_ids), len(order))),
        bounds_hint=bounds_hint,
        name=name,
    )
    scene.validate()
    return scene, list(range(len(train_ids))), list(range(len(train_ids), len(order)))


# --------------------------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------------------------


def _median_nn_spacing(means: torch.Tensor, chunk: int = 1024) -> float:
    best = torch.empty(means.shape[0], dtype=means.dtype)
    for start in range(0, means.shape[0], chunk):
        block = means[start : start + chunk]
        d = torch.cdist(block, means)
        d[torch.arange(block.shape[0]), torch.arange(start, start + block.shape[0])] = float("inf")
        best[start : start + block.shape[0]] = d.amin(dim=1)
    return float(best.median())


def _colors_from_train_views(
    points: torch.Tensor, teachers: dict[str, Teacher], train_ids: tuple[str, ...]
) -> torch.Tensor:
    """Sample colour from the first TRAINING view a point projects into (grey fallback).

    Training views only.  This is the appearance information the ``random-matched`` arm is
    allowed to have, and it is exactly what ``beam-cover``/``ci`` also derive from train views.
    """
    colors = torch.full((points.shape[0], 3), 0.5, dtype=points.dtype)
    remaining = torch.ones(points.shape[0], dtype=torch.bool)
    for view_id in train_ids:
        if not bool(remaining.any()):
            break
        teacher = teachers[view_id]
        uv, z = teacher.camera.project(points)
        hit = remaining & (z > 0.05) & teacher.camera.in_image(uv, margin=-0.5)
        if bool(hit.any()):
            colors[hit] = bilinear_sample(teacher.image, uv[hit])
            remaining &= ~hit
    return colors


def build_random_pair(
    reference: Gaussians3D,
    n: int,
    seed: int,
    teachers: dict[str, Teacher],
    train_ids: tuple[str, ...],
) -> tuple[Gaussians3D, Gaussians3D]:
    """``(random-matched, random-grey)`` — identical means, differing only in colour.

    Means are uniform in the Beam Fusion result's own bounding box.  That hands both controls the
    object's location for free, which is the conservative direction: it makes them harder to beat
    and isolates whether the *fine-grained* distribution matters.  Both receive the candidate's
    initial opacity (0.10) and an isotropic extent from their own median nearest-neighbour
    spacing, so neither is handicapped by extent or budget.
    """
    generator = torch.Generator().manual_seed(seed)
    lo = reference.means.amin(dim=0)
    hi = reference.means.amax(dim=0)
    unit = torch.rand((n, 3), generator=generator, dtype=reference.means.dtype)
    means = lo + unit * (hi - lo)

    spacing = _median_nn_spacing(means)
    log_scales = torch.full((n, 3), math.log(spacing * 0.5), dtype=reference.log_scales.dtype)
    quats = torch.zeros((n, 4), dtype=reference.quats.dtype)
    quats[:, 0] = 1.0
    opacity = torch.full((n,), 0.10, dtype=reference.opacity.dtype)

    n_bands = reference.sh.shape[1]
    grey_sh = torch.zeros((n, n_bands, 3), dtype=reference.sh.dtype)  # rgb_to_sh(0.5) == 0
    matched_sh = torch.zeros((n, n_bands, 3), dtype=reference.sh.dtype)
    matched_sh[:, 0] = rgb_to_sh(_colors_from_train_views(means, teachers, train_ids))

    def build(sh: torch.Tensor) -> Gaussians3D:
        return Gaussians3D(
            means=means.clone(),
            quats=quats.clone(),
            log_scales=log_scales.clone(),
            opacity=opacity.clone(),
            sh=sh,
        )

    return build(matched_sh), build(grey_sh)


def build_structured_arms(
    dataset: CompactDataset, train_ids: tuple[str, ...], n_init: int
) -> tuple[dict[str, Gaussians3D], dict[str, float], dict]:
    """``ci`` (Beam Fusion as-is) and ``beam-cover`` (cover-consistent isotropic extent)."""
    complete = dataset.to_reconstruction_inputs()
    names = complete.view_names
    idx = [names.index(v) for v in train_ids]
    inputs = ReconstructionInputs(
        observations=[complete.observations[i] for i in idx],
        cameras=[complete.cameras[i] for i in idx],
        view_names=[names[i] for i in idx],
        bounds_hint=complete.bounds_hint,
        name=f"{complete.name}-train",
    )
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

    started = time.perf_counter()
    floor = beam_contributor_footprints(inputs, beam)
    frames = estimate_local_surface_frames(ci.means, SurfelInitConfig())
    cover = reconcile_covariances(
        ci,
        SurfelInitConfig(isotropic=True, use_coverage_opacity=False, fixed_opacity=0.10),
        resolution_floor=floor,
        frames=frames,
    )
    cover_seconds = time.perf_counter() - started

    arms = {"ci": ci, "beam-cover": cover.gaussians}
    seconds = {"ci": fusion_seconds, "beam-cover": fusion_seconds + cover_seconds}
    diagnostics = {
        "beam_fusion_seconds": fusion_seconds,
        "cover_reconcile_seconds": cover_seconds,
        "native_roots": ci.n,
        "requested": n_init,
        "insufficient_initial_points": ci.n < n_init,
        "cover_rule": cover.diagnostics,
    }
    return arms, seconds, diagnostics


# --------------------------------------------------------------------------------------------
# Evaluation: per-view metrics, median-aggregated (section 4.1).  Deliberately not
# Trainer.evaluate_metrics, which means over views (review finding B4).
# --------------------------------------------------------------------------------------------


def evaluate_views(scene: SceneData, model: Gaussians3D, renderer, indices: list[int]) -> dict:
    per_view = {}
    with torch.no_grad():
        for index in indices:
            target = scene.images[index].to(model.means.device)
            camera = scene.cameras[index].to(model.means.device)
            out = renderer.render(model, camera)
            pred = out.color.clamp(0, 1)
            per_view[scene.view_names[index]] = {
                "psnr": psnr(pred, target.clamp(0, 1)),
                "ssim": float(ssim(pred, target.clamp(0, 1))),
            }
    return {
        "per_view": per_view,
        "psnr": _median([v["psnr"] for v in per_view.values()]),
        "ssim": _median([v["ssim"] for v in per_view.values()]),
    }


def train_config(iterations: int, rasterizer: str, device: str, seed: int) -> TrainConfig:
    """Fixed topology: no birth, clone, split, merge, or prune.  Final count must equal N."""
    return TrainConfig(
        iterations=iterations,
        rasterizer=rasterizer,
        device=device,
        densify=False,
        eval_every=EVAL_EVERY,
        target_sh_degree=3,
        use_masks=False,
        random_background=False,
        checkpoint_policy="final",
        seed=seed,
    )


def run_cell(
    *,
    arm: str,
    init: Gaussians3D,
    init_seconds: float,
    scene: SceneData,
    train_local: list[int],
    eval_local: list[int],
    seed: int,
    label: str,
    out: Path,
    args,
) -> dict:
    cell = out / "cells" / label / arm / str(seed)
    cell.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    renderer = get_rasterizer(args.rasterizer, device=device)
    model = init.to(device)
    n_initial = model.n

    torch.manual_seed(seed)
    step0 = evaluate_views(scene, model, renderer, eval_local)
    curve = [
        {
            "step": 0,
            "n": n_initial,
            # End-to-end seconds (section 4.3): arm-specific preprocessing is charged.
            "seconds": init_seconds,
            "train_seconds": 0.0,
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
        train_elapsed = time.perf_counter() - started - observer_seconds
        observer_started = time.perf_counter()
        values = evaluate_views(scene, snapshot, renderer, eval_local)
        curve.append(
            {
                "step": step,
                "n": snapshot.n,
                "seconds": init_seconds + train_elapsed,
                "train_seconds": train_elapsed,
                "psnr": values["psnr"],
                "ssim": values["ssim"],
            }
        )
        observer_seconds += time.perf_counter() - observer_started

    final, history = Trainer(
        train_config(args.iterations, args.rasterizer, args.device, seed)
    ).train(scene, model, checkpoint_callback=checkpoint)

    if final.n != n_initial:
        raise RuntimeError(
            f"fixed topology violated in {label}/{arm}/{seed}: {n_initial} -> {final.n}"
        )
    if args.save_ply:
        final.save_ply(cell / "gaussians_final.ply")

    final_eval = evaluate_views(scene, final.to(device), renderer, eval_local)
    train_eval = evaluate_views(scene, final.to(device), renderer, train_local)
    record = {
        "arm": arm,
        "label": label,
        "seed": seed,
        "n_initial": n_initial,
        "n_final": int(final.n),
        "init_seconds": init_seconds,
        "train_seconds": curve[-1]["train_seconds"],
        "end_to_end_seconds": curve[-1]["seconds"],
        "q_init": step0["psnr"],
        "q_final": final_eval["psnr"],
        "ssim_final": final_eval["ssim"],
        "train_psnr_final": train_eval["psnr"],
        "per_view_final": final_eval["per_view"],
        "peak_vram_gb": history.get("peak_vram_gb"),
        "curve": curve,
    }
    (cell / "record.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(
        f"[{label}/{arm}/{seed}] Q1500={record['q_final']:.3f}dB "
        f"Q0={record['q_init']:.3f}dB e2e={record['end_to_end_seconds']:.1f}s "
        f"n={record['n_final']}",
        flush=True,
    )
    return record


# --------------------------------------------------------------------------------------------
# Decision arithmetic (sections 4.2 / 5.1, with review finding B2 applied)
# --------------------------------------------------------------------------------------------


def time_to_target(curve: list[dict], tau: float) -> dict:
    """First checkpoint reaching ``tau`` and holding within ``tau - 0.10`` for the next two.

    A crossing at the last two checkpoints cannot establish persistence and is right-censored; a
    run that never establishes persistence is right-censored at its observed end.  It is never
    assigned a finite time, zero, or the budget limit.
    """
    for position, point in enumerate(curve):
        if point["psnr"] < tau:
            continue
        following = curve[position + 1 : position + 3]
        if len(following) < 2:
            return {"reached": False, "censored": True, "reason": "crossing_too_late"}
        if all(p["psnr"] >= tau - TAU_PERSIST_DB for p in following):
            return {
                "reached": True,
                "censored": False,
                "step": point["step"],
                "seconds": point["seconds"],
            }
    return {"reached": False, "censored": True, "reason": "never_reached"}


def analyse_cell(records: dict[str, dict[int, dict]], seeds: tuple[int, ...]) -> dict:
    """Paired per-seed contrasts of every arm against the comparator."""
    if COMPARATOR not in records:
        return {"note": f"comparator {COMPARATOR} absent; no contrast scored"}
    out: dict[str, dict] = {}
    for arm in records:
        if arm == COMPARATOR:
            continue
        quality_deltas, time_pairs = [], []
        for seed in seeds:
            candidate = records[arm].get(seed)
            comparator = records[COMPARATOR].get(seed)
            if candidate is None or comparator is None:
                continue
            quality_deltas.append(candidate["q_final"] - comparator["q_final"])
            tau = comparator["q_final"] - TAU_MARGIN_DB
            t_candidate = time_to_target(candidate["curve"], tau)
            t_comparator = time_to_target(comparator["curve"], tau)
            evaluable = t_candidate["reached"] and t_comparator["reached"]
            time_pairs.append(
                {
                    "seed": seed,
                    "tau_db": tau,
                    "candidate": t_candidate,
                    "comparator": t_comparator,
                    # Review finding B2: comparator or joint censoring is NON-EVALUABLE, never a
                    # candidate success.  The frozen protocol scores it as a success; that rule
                    # would award a win whenever the comparator gains >0.25 dB over its last 200
                    # updates, which is a property of the optimizer schedule, not of any
                    # initializer.
                    "evaluable": evaluable,
                    "ratio": (
                        t_candidate["seconds"] / t_comparator["seconds"]
                        if evaluable and t_comparator["seconds"] > 0
                        else None
                    ),
                }
            )
        ratios = [p["ratio"] for p in time_pairs if p["ratio"] is not None]
        wins = sum(1 for d in quality_deltas if d >= TAU_MARGIN_DB)
        out[arm] = {
            "quality_delta_db": quality_deltas,
            "quality_delta_median": _median(quality_deltas) if quality_deltas else None,
            "quality_delta_mad": _mad(quality_deltas) if quality_deltas else None,
            "quality_material_wins": wins,
            "quality_seeds": len(quality_deltas),
            "quality_verdict": (
                "material gain"
                if quality_deltas
                and wins >= max(1, len(quality_deltas) - 1)
                and _median(quality_deltas) >= TAU_MARGIN_DB
                else "material loss"
                if quality_deltas and _median(quality_deltas) <= -TAU_MARGIN_DB
                else "inside the 0.25 dB band"
            ),
            "time_pairs": time_pairs,
            "time_evaluable_pairs": len(ratios),
            "time_ratio_median": _median(ratios) if ratios else None,
            "time_verdict": (
                "insufficient evaluable pairs"
                if len(ratios) < 2
                else "faster to target"
                if _median(ratios) <= TIME_RATIO_GATE
                else "slower to target"
                if _median(ratios) >= 1.0 / TIME_RATIO_GATE
                else "no material time difference"
            ),
        }
    return out


def environment() -> dict:
    def _git(*cmd: str) -> str:
        try:
            return subprocess.check_output(["git", *cmd], cwd=ROOT, text=True).strip()
        except Exception:  # pragma: no cover - provenance is best-effort
            return "unavailable"

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--scenes", nargs="+", default=list(SCENES))
    parser.add_argument("--viewsets", nargs="+", choices=VIEWSETS, default=list(VIEWSETS))
    parser.add_argument("--counts", nargs="+", type=int, default=list(COUNTS))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--downscale", type=int, default=DOWNSCALE)
    parser.add_argument("--rasterizer", default="gsplat")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher-device", default="cuda")
    parser.add_argument("--save-ply", action="store_true", default=True)
    parser.add_argument("--no-save-ply", dest="save_ply", action="store_false")
    args = parser.parse_args()

    if not args.protocol.is_file():
        raise FileNotFoundError(f"protocol missing: {args.protocol}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    started_all = time.perf_counter()
    cells: dict[str, dict] = {}
    scene_meta: dict[str, dict] = {}

    for scene_key in args.scenes:
        spec = SCENES[scene_key]
        dataset_path = ROOT / spec["path"]
        dataset = CompactDataset.load(dataset_path, device="cpu")
        train8 = tuple(spec["train3"]) + tuple(spec["train8_extra"])
        needed = train8 + tuple(spec["validation"])
        print(f"[scene] {scene_key}: building {len(needed)} teachers", flush=True)
        teachers = build_teachers(
            dataset, needed, downscale=args.downscale, device=args.teacher_device
        )
        first = teachers[needed[0]]
        scene_meta[scene_key] = {
            "dataset": str(dataset_path.relative_to(ROOT)),
            "manifest_sha256": _sha256(dataset_path / "manifest.json"),
            "n_views_total": dataset.n_views,
            "train3": list(spec["train3"]),
            "train8": list(train8),
            "validation": list(spec["validation"]),
            "report_only_sealed": list(spec["report_only_sealed"]),
            "image_size": [first.camera.width, first.camera.height],
            "has_packed_alpha": any(v.alpha is not None for v in dataset.views),
        }

        for viewset in args.viewsets:
            train_ids = tuple(spec["train3"]) if viewset == "v3" else train8
            for n_init in args.counts:
                label = f"{scene_key}/{viewset}/N{n_init}"
                print(f"[cell] {label}: initializing", flush=True)
                structured, init_seconds, diagnostics = build_structured_arms(
                    dataset, train_ids, n_init
                )
                if diagnostics["insufficient_initial_points"]:
                    print(f"[cell] {label}: insufficient_initial_points, skipped", flush=True)
                    cells[label] = {"status": "insufficient_initial_points", **diagnostics}
                    continue

                scene, train_local, eval_local = make_scene(
                    teachers,
                    train_ids,
                    tuple(spec["validation"]),
                    dataset.bounds_hint,
                    f"{scene_key}-{viewset}-N{n_init}",
                )

                records: dict[str, dict[int, dict]] = {arm: {} for arm in args.arms}
                for seed in args.seeds:
                    matched, grey = build_random_pair(
                        structured["ci"], n_init, seed, teachers, train_ids
                    )
                    inits = {
                        "ci": (structured["ci"], init_seconds["ci"]),
                        "beam-cover": (structured["beam-cover"], init_seconds["beam-cover"]),
                        # The random controls read the Beam Fusion bounding box, so their
                        # reported end-to-end time excludes deriving it: a lower bound that
                        # favours the control.
                        "random-matched": (matched, 0.0),
                        "random-grey": (grey, 0.0),
                    }
                    for arm in args.arms:
                        init, seconds = inits[arm]
                        records[arm][seed] = run_cell(
                            arm=arm,
                            init=init,
                            init_seconds=seconds,
                            scene=scene,
                            train_local=train_local,
                            eval_local=eval_local,
                            seed=seed,
                            label=label,
                            out=out,
                            args=args,
                        )

                cells[label] = {
                    "status": "complete",
                    "scene": scene_key,
                    "viewset": viewset,
                    "n_init": n_init,
                    "train_views": list(train_ids),
                    "validation_views": list(spec["validation"]),
                    "initialization": diagnostics,
                    "init_seconds": init_seconds,
                    "records": {a: {str(s): r for s, r in v.items()} for a, v in records.items()},
                    "contrasts": analyse_cell(records, tuple(args.seeds)),
                }

    summary = {
        "schema": "rtgs.init_value_dev_screen.v1",
        "status": "DEVELOPMENT-ONLY — NOT CONFIRMATORY",
        "protocol": {
            "path": str(args.protocol.relative_to(ROOT)),
            "sha256": _sha256(args.protocol),
        },
        "review": (
            {"path": str(args.review.relative_to(ROOT)), "sha256": _sha256(args.review)}
            if args.review.is_file()
            else None
        ),
        "deviations_from_frozen_screen": [
            "colmap-sfm-fixed-pose is UNAVAILABLE: no source RGB in the repository. Reported as "
            "baseline unavailability (section 3.3); the repository default `ci` stands in as the "
            "comparator and no COLMAP-relative claim is made.",
            "Qval is measured against the compact 2D self-reconstruction, not source RGB.",
            "Full-canvas metrics: Karate carries no packed alpha (section 7.1).",
            "Random control split into random-matched / random-grey to measure the appearance "
            "confound identified as review finding B1.",
            "Comparator/joint time censoring is scored NON-EVALUABLE, not a candidate success "
            "(review finding B2).",
            "Report-only views were not opened.",
        ],
        "anchor": {"viewset": ANCHOR[0], "n_init": ANCHOR[1]},
        "config": {
            "iterations": args.iterations,
            "eval_every": EVAL_EVERY,
            "checkpoints": list(range(0, args.iterations + 1, EVAL_EVERY)),
            "downscale": args.downscale,
            "topology": "fixed (densify=False; final count asserted equal to initial)",
            "rasterizer": args.rasterizer,
            "device": args.device,
            "seeds": list(args.seeds),
            "arms": list(args.arms),
            "comparator": COMPARATOR,
            "tau_rule": "tau(s,k) = Q_comparator(s,k,1500) - 0.25 dB",
        },
        "environment": environment(),
        "scenes": scene_meta,
        "total_seconds": time.perf_counter() - started_all,
        "cells": cells,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"[written] {out / 'summary.json'}", flush=True)
    print(f"[total] {summary['total_seconds'] / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
