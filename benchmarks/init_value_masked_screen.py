#!/usr/bin/env python3
"""Initialization Value Program — MASKED development screen (Stage frames).

This is the masked counterpart of ``benchmarks/init_value_dev_screen.py``.  That screen ran on the
Karate validation frames, which carry no packed alpha, so its endpoint was full-canvas PSNR.  The
only alpha-bearing compact bundles in this repository are the two Stage frames (26/26 views each),
so this screen runs there and scores the section 4.2 masked foreground endpoint.

**Evidence status: DEVELOPMENT-ONLY, and weaker than the Karate screen.**  Section 1.2 of the
master protocol assigns Stage frames ``frame_00008``/``frame_00009`` the role
"debugging and hypothesis generation only" — they are outcome-exposed: prior experiments in this
repository selected variants on them.  A masked number computed here is therefore *not* the masked
confirmatory endpoint of section 7.1; it is a mechanism screen that shows what the masked endpoint
does to the arm ordering.  Nothing here may enter ``README.md``, ``docs/``, or
``ara/logic/claims.md``, and no result here may be pooled with, or described as replication of, a
confirmatory effect.

What this screen adds over the Karate screen
--------------------------------------------
1. **A masked endpoint.**  Primary ``Q`` is median per-view foreground-weighted PSNR: RGB squared
   error weighted by the ground-truth alpha, summed and divided by ``3 * sum(alpha)``
   (``rtgs.core.metrics.masked_psnr``, which is section 4.2's definition verbatim).  Crop PSNR and
   crop SSIM on the foreground bounding box plus a 5% margin are reported alongside, and
   full-canvas PSNR against the masked target is kept as a diagnostic only.
2. **Supervision as an explicit factor.**  ``supervision=full`` trains exactly as the Karate screen
   did, on each view's own complete 2D fit.  ``supervision=masked`` composites the target onto a
   black background and uses the trainer's masked loss (weighted L1 plus the outside-alpha
   penalty).  Initialization is computed once per cell and shared by both modes, so supervision is
   a clean one-factor contrast: the arms start from bit-identical gaussians.

Everything else — arms, seeds, counts, viewsets, 1500 fixed-topology updates, downscale 4, the
0.25 dB materiality margin, the comparator, and the censoring rule from review finding B2 — is
inherited unchanged from the Karate screen so the two are directly comparable.

Splits are derived by a rule fixed before the run and independent of any outcome (see
``derive_splits``).  Stage frames have no frozen section 2.2 split; inventing one after seeing
results would be selection on outcome-exposed data, so the rule is mechanical and stated here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from benchmarks.init_value_dev_screen import (
    ARMS,
    COUNTS,
    DOWNSCALE,
    EVAL_EVERY,
    ITERATIONS,
    SEEDS,
    VIEWSETS,
    Teacher,
    _mad,
    _median,
    _sha256,
    analyse_cell,
    build_random_pair,
    build_structured_arms,
    build_teachers,
    environment,
)

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.metrics import image_metrics
from rtgs.data.compact_views import CompactDataset
from rtgs.data.scene import SceneData
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.base import get_rasterizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "runs/init_value_masked_screen"
DEFAULT_PROTOCOL = ROOT / "benchmarks/results/20260725_init_value_program_PREREG.md"
DEFAULT_REVIEW = ROOT / "benchmarks/results/20260725_init_value_program_PREREG_REVIEW_02_FAIL.md"
DEFAULT_KARATE_RUN = ROOT / "runs/init_value_dev_screen/summary.json"

STAGE = "dataset/2025_03_07_stage_with_fabric"
SCENE_PATHS = {
    "frame_00008": f"{STAGE}/frame_00008/gaussians2d",
    "frame_00009": f"{STAGE}/frame_00009/gaussians2d",
}

# Inherited verbatim from the frozen Karate protocol (section 2.2) rather than chosen on Stage.
# Both roles are ID-addressed; a per-file hash cannot detect a role mis-assignment (finding A2).
VALIDATION = ("C0031", "C1000", "C1002")
REPORT_ONLY_SEALED = ("C1001", "C1004")

SUPERVISIONS = ("full", "masked")
ANCHOR = ("v3", 2400)


def derive_splits(dataset: CompactDataset) -> dict:
    """Mechanical, outcome-independent split rule for a Stage frame.

    Sort the eligible view IDs, take eight evenly spaced views as ``train8``, and take every third
    of those as ``train3`` so ``train3`` is a subset spanning the same arc.  Validation and
    report-only IDs are removed from the pool first and never enter training.  Nothing in this
    function reads a metric.
    """
    ids = sorted(view.view_id for view in dataset.views)
    excluded = set(VALIDATION) | set(REPORT_ONLY_SEALED)
    missing = excluded - set(ids)
    if missing:
        raise KeyError(f"{dataset.name} is missing reserved views {sorted(missing)}")
    pool = [view_id for view_id in ids if view_id not in excluded]
    if len(pool) < 8:
        raise ValueError(f"{dataset.name} has too few eligible training views: {len(pool)}")
    n = len(pool)
    train8 = tuple(pool[(i * n) // 8] for i in range(8))
    if len(set(train8)) != 8:
        raise ValueError(f"{dataset.name} split rule produced duplicate training views")
    train3 = (train8[0], train8[3], train8[6])
    return {
        "train3": train3,
        "train8": train8,
        "train8_extra": tuple(v for v in train8 if v not in train3),
        "validation": VALIDATION,
        "report_only_sealed": REPORT_ONLY_SEALED,
        "pool_size": n,
        "rule": "sorted eligible IDs; train8 = pool[(i*n)//8] for i in 0..7; "
        "train3 = train8[0],[3],[6]; validation and report-only removed before selection",
    }


def downscaled_mask(view, *, downscale: int, height: int, width: int) -> torch.Tensor:
    """Nearest-neighbour alpha at exactly the sample centres ``build_teachers`` renders.

    ``build_teachers`` samples the observation field at ``(i + 0.5) * downscale`` in crop-local
    coordinates, and the packed alpha crop is validated at load time to coincide with the fit
    window, so the same integer floor lands on the same source pixel.  No resampling filter is
    applied: the mask stays a lossless subsample of the ground-truth alpha.
    """
    if view.alpha is None:
        raise ValueError(f"view {view.view_id} carries no packed alpha")
    crop = view.alpha.crop_mask()
    fit_height, fit_width = crop.shape
    ys = (((torch.arange(height, dtype=torch.float32) + 0.5) * downscale).long()).clamp(
        0, fit_height - 1
    )
    xs = (((torch.arange(width, dtype=torch.float32) + 0.5) * downscale).long()).clamp(
        0, fit_width - 1
    )
    return crop[ys][:, xs].to(torch.float32)


def build_masks(
    dataset: CompactDataset, teachers: dict[str, Teacher], *, downscale: int
) -> dict[str, torch.Tensor]:
    by_id = {view.view_id: view for view in dataset.views}
    masks = {}
    for view_id, teacher in teachers.items():
        height, width = teacher.image.shape[:2]
        mask = downscaled_mask(by_id[view_id], downscale=downscale, height=height, width=width)
        coverage = float(mask.mean())
        if coverage <= 0.0:
            raise ValueError(f"view {view_id} has an empty foreground after downscale")
        masks[view_id] = mask
    return masks


def make_masked_scene(
    teachers: dict[str, Teacher],
    masks: dict[str, torch.Tensor],
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
        masks=[masks[v] for v in order],
        view_names=list(order),
        train_indices=list(range(len(train_ids))),
        test_indices=list(range(len(train_ids), len(order))),
        bounds_hint=bounds_hint,
        name=name,
    )
    scene.validate()
    return scene, list(range(len(train_ids))), list(range(len(train_ids), len(order)))


def evaluate_views_masked(
    scene: SceneData, model: Gaussians3D, renderer, indices: list[int]
) -> dict:
    """Median-aggregated per-view masked metrics.  ``psnr`` is section 4.2's foreground PSNR."""
    per_view = {}
    with torch.no_grad():
        for index in indices:
            device = model.means.device
            target = scene.images[index].to(device).clamp(0, 1)
            mask = scene.masks[index].to(device)
            camera = scene.cameras[index].to(device)
            pred = renderer.render(model, camera).color.clamp(0, 1)
            values = image_metrics(pred, target, mask)
            per_view[scene.view_names[index]] = {
                "psnr": values["psnr_fg"],
                "psnr_crop": values["psnr_crop"],
                "ssim_crop": values["ssim_crop"],
                "psnr_full_masked_target": values["psnr_full"],
                "foreground_fraction": float(mask.mean()),
            }
    return {
        "per_view": per_view,
        "psnr": _median([v["psnr"] for v in per_view.values()]),
        "psnr_crop": _median([v["psnr_crop"] for v in per_view.values()]),
        "ssim": _median([v["ssim_crop"] for v in per_view.values()]),
        "psnr_full_masked_target": _median(
            [v["psnr_full_masked_target"] for v in per_view.values()]
        ),
    }


def train_config(supervision: str, *, iterations: int, rasterizer: str, device: str, seed: int):
    """Fixed topology.  ``supervision`` is the only field that differs between the two modes."""
    return TrainConfig(
        iterations=iterations,
        rasterizer=rasterizer,
        device=device,
        densify=False,
        eval_every=EVAL_EVERY,
        target_sh_degree=3,
        use_masks=(supervision == "masked"),
        random_background=False,
        checkpoint_policy="final",
        seed=seed,
    )


def run_cell(
    *,
    arm: str,
    supervision: str,
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
    cell = out / "cells" / label / supervision / arm / str(seed)
    cell.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    renderer = get_rasterizer(args.rasterizer, device=device)
    model = init.to(device)
    n_initial = model.n

    torch.manual_seed(seed)
    step0 = evaluate_views_masked(scene, model, renderer, eval_local)
    curve = [
        {
            "step": 0,
            "n": n_initial,
            "seconds": init_seconds,  # section 4.3: arm-specific preprocessing is charged
            "train_seconds": 0.0,
            "psnr": step0["psnr"],
            "psnr_crop": step0["psnr_crop"],
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
        values = evaluate_views_masked(scene, snapshot, renderer, eval_local)
        curve.append(
            {
                "step": step,
                "n": snapshot.n,
                "seconds": init_seconds + train_elapsed,
                "train_seconds": train_elapsed,
                "psnr": values["psnr"],
                "psnr_crop": values["psnr_crop"],
                "ssim": values["ssim"],
            }
        )
        observer_seconds += time.perf_counter() - observer_started

    final, history = Trainer(
        train_config(
            supervision,
            iterations=args.iterations,
            rasterizer=args.rasterizer,
            device=args.device,
            seed=seed,
        )
    ).train(scene, model, checkpoint_callback=checkpoint)

    if final.n != n_initial:
        raise RuntimeError(
            f"fixed topology violated in {label}/{supervision}/{arm}/{seed}: "
            f"{n_initial} -> {final.n}"
        )
    if args.save_ply:
        final.save_ply(cell / "gaussians_final.ply")

    final_eval = evaluate_views_masked(scene, final.to(device), renderer, eval_local)
    train_eval = evaluate_views_masked(scene, final.to(device), renderer, train_local)
    record = {
        "arm": arm,
        "supervision": supervision,
        "label": label,
        "seed": seed,
        "n_initial": n_initial,
        "n_final": int(final.n),
        "init_seconds": init_seconds,
        "train_seconds": curve[-1]["train_seconds"],
        "end_to_end_seconds": curve[-1]["seconds"],
        "q_init": step0["psnr"],
        "q_final": final_eval["psnr"],
        "q_crop_final": final_eval["psnr_crop"],
        "ssim_final": final_eval["ssim"],
        "psnr_full_masked_target_final": final_eval["psnr_full_masked_target"],
        "train_psnr_final": train_eval["psnr"],
        "per_view_final": final_eval["per_view"],
        "peak_vram_gb": history.get("peak_vram_gb"),
        "curve": curve,
    }
    (cell / "record.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(
        f"[{label}/{supervision}/{arm}/{seed}] Qfg@{args.iterations}={record['q_final']:.3f}dB "
        f"Qfg@0={record['q_init']:.3f}dB crop={record['q_crop_final']:.3f}dB "
        f"e2e={record['end_to_end_seconds']:.1f}s n={record['n_final']}",
        flush=True,
    )
    return record


def supervision_contrast(records: dict[str, dict[str, dict[int, dict]]], seeds) -> dict:
    """Paired per-(arm, seed) effect of masked supervision on the masked endpoint."""
    out = {}
    for arm in records.get("masked", {}):
        deltas = []
        for seed in seeds:
            masked = records["masked"][arm].get(seed)
            full = records.get("full", {}).get(arm, {}).get(seed)
            if masked is None or full is None:
                continue
            deltas.append(masked["q_final"] - full["q_final"])
        if deltas:
            out[arm] = {
                "delta_db": deltas,
                "delta_median": _median(deltas),
                "delta_mad": _mad(deltas),
                "seeds": len(deltas),
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--scenes", nargs="+", default=list(SCENE_PATHS))
    parser.add_argument("--viewsets", nargs="+", choices=VIEWSETS, default=list(VIEWSETS))
    parser.add_argument("--counts", nargs="+", type=int, default=list(COUNTS))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument(
        "--supervisions", nargs="+", choices=SUPERVISIONS, default=list(SUPERVISIONS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--downscale", type=int, default=DOWNSCALE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rasterizer", default="gsplat")
    parser.add_argument("--save-ply", action="store_true", default=True)
    parser.add_argument("--no-save-ply", dest="save_ply", action="store_false")
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()

    summary = {
        "schema": "rtgs.init_value_masked_screen.v1",
        "status": "DEVELOPMENT-ONLY — NOT CONFIRMATORY — OUTCOME-EXPOSED INPUTS",
        "protocol": {
            "path": str(args.protocol.relative_to(ROOT)),
            "sha256": _sha256(args.protocol),
        },
        "review": {"path": str(args.review.relative_to(ROOT)), "sha256": _sha256(args.review)},
        "companion_run": {
            "path": "runs/init_value_dev_screen/summary.json",
            "note": "unmasked Karate screen; identical arms, seeds, counts, viewsets, and margin",
            "sha256": (
                _sha256(DEFAULT_KARATE_RUN) if DEFAULT_KARATE_RUN.exists() else "unavailable"
            ),
        },
        "evidence_limits": [
            "Stage frames frame_00008/frame_00009 are section 1.2 pilot/development inputs: "
            "outcome-exposed, licensed for debugging and hypothesis generation only. This is NOT "
            "the section 7.1 masked confirmatory endpoint.",
            "No colmap-sfm-fixed-pose arm: no source RGB in the repository. The repository "
            "default `ci` stands in as comparator and no COLMAP-relative claim is made.",
            "Q is measured against each view's own compact 2D fit restricted to ground-truth "
            "alpha, not against source RGB.",
            "Splits are derived by a mechanical rule, not frozen by section 2.2, because Stage "
            "frames have no protocol split.",
            "Report-only views were not opened.",
        ],
        "anchor": {"viewset": ANCHOR[0], "n_init": ANCHOR[1]},
        "config": {
            "iterations": args.iterations,
            "eval_every": EVAL_EVERY,
            "checkpoints": [0] + list(range(EVAL_EVERY, args.iterations + 1, EVAL_EVERY)),
            "downscale": args.downscale,
            "topology": "fixed (densify=False; final count asserted equal to initial)",
            "rasterizer": args.rasterizer,
            "device": args.device,
            "seeds": list(args.seeds),
            "arms": list(args.arms),
            "supervisions": list(args.supervisions),
            "comparator": "ci",
            "primary_endpoint": "median per-view foreground-weighted PSNR (section 4.2)",
            "secondary": ["psnr_crop", "ssim_crop", "psnr_full_masked_target"],
            "tau_rule": "tau(s,k) = Q_comparator(s,k,final) - 0.25 dB",
        },
        "environment": environment(),
        "scenes": {},
        "cells": {},
    }

    def flush() -> None:
        summary["total_seconds"] = time.perf_counter() - started_all
        (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    for scene_name in args.scenes:
        path = ROOT / SCENE_PATHS[scene_name]
        dataset = CompactDataset.load(path, device="cpu")
        splits = derive_splits(dataset)
        manifest = path / "manifest.json"
        with_alpha = sum(1 for v in dataset.views if v.alpha is not None)
        summary["scenes"][scene_name] = {
            "dataset": SCENE_PATHS[scene_name],
            "manifest_sha256": _sha256(manifest),
            "n_views_total": len(dataset.views),
            "n_views_with_alpha": with_alpha,
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in splits.items()},
        }
        if with_alpha != len(dataset.views):
            raise RuntimeError(f"{scene_name}: {with_alpha}/{len(dataset.views)} views carry alpha")

        needed = tuple(splits["train8"]) + tuple(splits["validation"])
        teachers = build_teachers(dataset, needed, downscale=args.downscale, device=args.device)
        masks = build_masks(dataset, teachers, downscale=args.downscale)
        summary["scenes"][scene_name]["foreground_fraction"] = {
            v: round(float(m.mean()), 6) for v, m in masks.items()
        }
        flush()

        for viewset in args.viewsets:
            train_ids = tuple(splits["train3"]) if viewset == "v3" else tuple(splits["train8"])
            scene, train_local, eval_local = make_masked_scene(
                teachers,
                masks,
                train_ids,
                tuple(splits["validation"]),
                dataset.bounds_hint,
                f"{scene_name}-{viewset}",
            )
            for n_init in args.counts:
                label = f"{scene_name}/{viewset}/N{n_init}"
                print(f"[cell] {label}: initializing", flush=True)
                # One initialization per cell, shared by both supervision modes, so the
                # supervision contrast is a clean one-factor comparison.
                structured, structured_seconds, diagnostics = build_structured_arms(
                    dataset, train_ids, n_init
                )
                inits: dict[str, tuple[Gaussians3D, float]] = {
                    a: (structured[a], structured_seconds[a])
                    for a in ("ci", "beam-cover")
                    if a in args.arms
                }
                if "random-matched" in args.arms or "random-grey" in args.arms:
                    matched, grey = build_random_pair(
                        structured["beam-cover"], n_init, args.seeds[0], teachers, train_ids
                    )
                    if "random-matched" in args.arms:
                        inits["random-matched"] = (matched, 0.0)
                    if "random-grey" in args.arms:
                        inits["random-grey"] = (grey, 0.0)

                records: dict[str, dict[str, dict[int, dict]]] = {}
                for supervision in args.supervisions:
                    records[supervision] = {}
                    for arm in args.arms:
                        init, init_seconds = inits[arm]
                        records[supervision][arm] = {}
                        for seed in args.seeds:
                            records[supervision][arm][seed] = run_cell(
                                arm=arm,
                                supervision=supervision,
                                init=init,
                                init_seconds=init_seconds,
                                scene=scene,
                                train_local=train_local,
                                eval_local=eval_local,
                                seed=seed,
                                label=label,
                                out=out,
                                args=args,
                            )

                summary["cells"][label] = {
                    "status": "complete",
                    "scene": scene_name,
                    "viewset": viewset,
                    "n_init": n_init,
                    "train_views": list(train_ids),
                    "validation_views": list(splits["validation"]),
                    "initialization": diagnostics,
                    "records": {
                        s: {
                            a: {str(k): v for k, v in by_seed.items()}
                            for a, by_seed in arms.items()
                        }
                        for s, arms in records.items()
                    },
                    "contrasts": {
                        s: analyse_cell(arms, tuple(args.seeds)) for s, arms in records.items()
                    },
                    "supervision_contrast": supervision_contrast(records, tuple(args.seeds)),
                }
                flush()

    flush()
    print(f"[written] {out / 'summary.json'}")
    print(f"[total] {(time.perf_counter() - started_all) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
