#!/usr/bin/env python3
"""Uniform fixed-topology recovery for the full-resolution ADR-002 experiment.

The parent experiment evaluated several arms immediately after a non-empty final density event.
This follow-up loads every optimized saved model and applies the same 40-update, no-density
full-resolution polish.  It never overwrites the parent bundle and explicitly records the optimizer
restart and reloaded-PLY boundary.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch

try:
    from benchmarks.carrier_refinement_fullres import (
        ARMS,
        COMPACT_PATH,
        DEFAULT_RAW_FRAME,
        JPEG_QUALITY,
        SEED,
        LPIPSEvaluator,
        _bind_raw_sources,
        _evaluate,
        _jpeg_scene,
        _json,
        _load_raw_scene,
        _sha256,
        _viewer_manifest,
        derive_splits,
    )
except ModuleNotFoundError:  # direct ``python benchmarks/carrier_refinement_recovery.py``
    from carrier_refinement_fullres import (  # type: ignore[no-redef]
        ARMS,
        COMPACT_PATH,
        DEFAULT_RAW_FRAME,
        JPEG_QUALITY,
        SEED,
        LPIPSEvaluator,
        _bind_raw_sources,
        _evaluate,
        _jpeg_scene,
        _json,
        _load_raw_scene,
        _sha256,
        _viewer_manifest,
        derive_splits,
    )

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.compact_views import CompactDataset
from rtgs.optim.trainer import TrainConfig, Trainer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = ROOT / "runs/carrier_refinement_fullres_frame00008_20260727"
DEFAULT_OUT = ROOT / "runs/carrier_refinement_fullres_frame00008_20260727_recovery"
DEFAULT_PROTOCOL = ROOT / "benchmarks/results/20260727_carrier_refinement_recovery_PREREG.md"
RECOVERY_ITERATIONS = 40
ITERATION_OFFSET = 160
SCHEDULE_ITERATIONS = 200
RECOVERY_SEED = SEED + ITERATION_OFFSET
RECOVERY_ARMS = tuple(arm for arm in ARMS if arm != "beam-only")


def _config() -> TrainConfig:
    return TrainConfig(
        iterations=RECOVERY_ITERATIONS,
        iteration_offset=ITERATION_OFFSET,
        schedule_iterations=SCHEDULE_ITERATIONS,
        rasterizer="gsplat",
        device="cuda",
        densify=False,
        eval_every=RECOVERY_ITERATIONS,
        checkpoint_policy="final",
        target_sh_degree=3,
        sh_degree_interval=1,
        use_masks=True,
        random_background=False,
        packed=True,
        seed=RECOVERY_SEED,
    )


def _write_index(out: Path, summary: dict) -> None:
    rows = []
    cards = []
    for record in summary["arms"]:
        arm = html.escape(record["arm"])
        before = record["parent_metrics"]["psnr_fg"]
        after = record["recovered_metrics"]["psnr_fg"]
        rows.append(
            f"<tr><td><a href='{arm}/record.json'>{arm}</a></td>"
            f"<td>{before:.3f}</td><td>{after:.3f}</td><td>{after - before:+.3f}</td>"
            f"<td>{record['recovered_metrics']['ssim_crop']:.4f}</td>"
            f"<td>{record['recovered_metrics']['lpips_crop_max512']:.4f}</td>"
            f"<td>{record['n_gaussians']:,}</td><td>{record['training_seconds']:.2f}</td></tr>"
        )
        cards.append(
            f"<article><h3>{arm} · {after:.2f} dB</h3>"
            f"<a href='{arm}/comparison_recovered.png'><img "
            f"src='{arm}/comparison_recovered.png' alt='{arm} recovered'></a>"
            f"<p><a href='{arm}/gaussians.ply'>PLY</a> · "
            f"<a href='{arm}/metrics.json'>metrics</a></p></article>"
        )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Carrier recovery follow-up</title><style>
:root{{color-scheme:dark}}body{{max-width:1450px;margin:35px auto;padding:0 18px;
font:15px/1.5 system-ui;background:#091019;color:#edf4fb}}a{{color:#6ed8c8}}
.notice,table,article{{background:#121d29;border:1px solid #2c3c50;border-radius:12px}}
.notice{{padding:14px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;
border-bottom:1px solid #2c3c50;text-align:right}}th:first-child,td:first-child{{text-align:left}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:12px}}
article{{overflow:hidden}}article h3,article p{{padding:0 12px}}img{{display:block;width:100%}}
</style></head><body><h1>Uniform 40-step recovery after final density surgery</h1>
<p class="notice">Exploratory remediation of a scientist-pass finding. Every optimized parent arm
is reloaded from its saved PLY and receives exactly 40 native-resolution, fixed-topology updates.
This does not overwrite or retroactively validate the parent endpoints.</p>
<p><a href="summary.json">summary.json</a> ·
<a href="../carrier_refinement_fullres_frame00008_20260727/index.html">parent experiment</a> ·
<a href="../../benchmarks/results/20260727_carrier_refinement_recovery_PREREG.md">
frozen recovery protocol</a> · <a href="comparison_manifest.json">viewer manifest</a></p>
<table><thead><tr><th>Arm</th><th>Parent dB</th><th>Recovered dB</th><th>Δ dB</th>
<th>SSIM</th><th>LPIPS↓</th><th>N</th><th>Train s</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table><h2>Recovered C0031 comparisons</h2>
<div class="grid">{"".join(cards)}</div></body></html>"""
    (out / "index.html").write_text(page)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--raw-frame", type=Path, default=DEFAULT_RAW_FRAME)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=RECOVERY_ARMS,
        default=list(RECOVERY_ARMS),
    )
    args = parser.parse_args()
    if not args.protocol.is_file():
        raise FileNotFoundError("freeze the recovery protocol before execution")
    parent = args.parent.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to mix recovery results in {out}")
    out.mkdir(parents=True, exist_ok=True)
    parent_summary_path = parent / "summary.json"
    parent_summary = json.loads(parent_summary_path.read_text())
    if (
        parent_summary.get("schema") != "rtgs.carrier-refinement-fullres.v1"
        or parent_summary.get("status") != "complete"
    ):
        raise ValueError("parent summary is not a complete carrier experiment")
    parent_records = {record["arm"]: record for record in parent_summary["arms"]}
    selected_arms = tuple(args.arms)
    if any(arm not in parent_records for arm in selected_arms):
        raise ValueError("requested recovery arm is absent from the parent summary")

    dataset = CompactDataset.load(COMPACT_PATH, device="cpu")
    split = derive_splits(dataset)
    train_ids = tuple(split["train3"])
    validation_ids = tuple(split["validation"])
    source_binding = _bind_raw_sources(
        dataset,
        args.raw_frame.resolve(),
        (*train_ids, *validation_ids),
    )
    _json(out / "source_binding.json", source_binding)
    scene, raw_load_seconds = _load_raw_scene(
        args.raw_frame.resolve(),
        train_ids,
        validation_ids,
        dataset.bounds_hint,
        downscale=1,
    )
    jpeg_scene, _jpeg_bytes = _jpeg_scene(scene, JPEG_QUALITY)
    lpips_metric = LPIPSEvaluator(torch.device("cuda"))
    config = _config()
    records = []
    started_total = time.perf_counter()
    for arm in selected_arms:
        directory = out / arm
        directory.mkdir(parents=True)
        parent_model_path = parent / arm / "gaussians.ply"
        parent_model_sha256 = _sha256(parent_model_path)
        initialization = Gaussians3D.load_ply(parent_model_path)
        parent_metrics = parent_records[arm]["final_metrics"]
        reloaded_metrics = _evaluate(
            scene,
            initialization.to("cuda"),
            scene.testing_views,
            lpips_metric,
            preview_dir=directory,
            prefix="reloaded",
        )
        training_scene = jpeg_scene if arm == "random-jpeg-q50" else scene
        started = time.perf_counter()
        final, history = Trainer(config).train(training_scene, initialization)
        training_seconds = time.perf_counter() - started
        if final.n != initialization.n:
            raise RuntimeError(f"fixed-topology recovery changed {arm} count")
        if history.get("density_stats") is not None:
            raise RuntimeError(f"fixed-topology recovery emitted density stats for {arm}")
        recovered_metrics = _evaluate(
            scene,
            final.to("cuda"),
            scene.testing_views,
            lpips_metric,
            preview_dir=directory,
            prefix="recovered",
        )
        initialization.save_ply(directory / "gaussians_init.ply")
        final.save_ply(directory / "gaussians.ply")
        shutil.copy2(directory / "gaussians.ply", directory / "gaussians_final.ply")
        record = {
            "arm": arm,
            "parent_model": str(parent_model_path.relative_to(ROOT)),
            "parent_model_sha256": parent_model_sha256,
            "parent_metrics": parent_metrics,
            "reloaded_metrics": reloaded_metrics,
            "reloaded_parent_psnr_abs_delta": abs(
                reloaded_metrics["psnr_fg"] - parent_metrics["psnr_fg"]
            ),
            "recovered_metrics": recovered_metrics,
            "psnr_delta": recovered_metrics["psnr_fg"] - parent_metrics["psnr_fg"],
            "n_gaussians": final.n,
            "iterations": RECOVERY_ITERATIONS,
            "iteration_offset": ITERATION_OFFSET,
            "training_seconds": training_seconds,
            "peak_vram_gb": float(history.get("peak_vram_gb", 0.0)),
        }
        _json(directory / "record.json", record)
        _json(directory / "training_history.json", history)
        _json(
            directory / "gaussians.config.json",
            {
                "schema": "rtgs.carrier-recovery-config.v1",
                "config": asdict(config),
                "optimizer_restart": True,
                "source_boundary": "reloaded parent PLY",
            },
        )
        _json(
            directory / "metrics.json",
            {
                "metrics": {
                    "psnr": recovered_metrics["psnr_fg"],
                    "ssim": recovered_metrics["ssim_crop"],
                    "lpips": recovered_metrics["lpips_crop_max512"],
                    "alpha_iou": recovered_metrics["alpha_iou"],
                    "n_gaussians": final.n,
                    "training_seconds": training_seconds,
                },
                "details": recovered_metrics,
            },
        )
        records.append(record)
        print(
            f"[{arm}] {parent_metrics['psnr_fg']:.3f} -> "
            f"{recovered_metrics['psnr_fg']:.3f} dB "
            f"({record['psnr_delta']:+.3f}), n={final.n}",
            flush=True,
        )
        torch.cuda.empty_cache()

    summary = {
        "schema": "rtgs.carrier-refinement-recovery.v1",
        "status": ("complete" if set(selected_arms) == set(RECOVERY_ARMS) else "partial"),
        "evidence_scope": (
            "post-outcome exploratory recovery; not preregistered confirmatory evidence"
        ),
        "reason": "parent endpoints may contain newborns from a non-empty final density event",
        "parent_summary": str(parent_summary_path.relative_to(ROOT)),
        "parent_summary_sha256": _sha256(parent_summary_path),
        "protocol": {
            "path": str(args.protocol.resolve().relative_to(ROOT)),
            "sha256": _sha256(args.protocol),
        },
        "resolution": [5328, 4608],
        "split": split,
        "recovery_iterations": RECOVERY_ITERATIONS,
        "iteration_offset": ITERATION_OFFSET,
        "schedule_iterations": SCHEDULE_ITERATIONS,
        "seed": RECOVERY_SEED,
        "optimizer_restart": True,
        "density": "disabled",
        "raw_load_seconds": raw_load_seconds,
        "arms": records,
        "total_wall_seconds": time.perf_counter() - started_total,
    }
    _json(out / "summary.json", summary)
    _json(out / "comparison_manifest.json", _viewer_manifest(out, records))
    representative = out / "carrier-schedule-clone"
    if representative.is_dir():
        for name in (
            "gaussians_init.ply",
            "gaussians.ply",
            "metrics.json",
            "training_history.json",
            "gaussians.config.json",
            "preview_reloaded.png",
            "preview_recovered.png",
        ):
            target = (
                "preview_init.png"
                if name == "preview_reloaded.png"
                else ("preview_final.png" if name == "preview_recovered.png" else name)
            )
            shutil.copy2(representative / name, out / target)
    _write_index(out, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
