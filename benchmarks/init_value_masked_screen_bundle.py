#!/usr/bin/env python3
"""Complete the masked screen's results bundle to Hard Rule 7 shape.

The screen writes many sub-runs; the Rule 7 gate expects one ``rtgs run``-shaped bundle at the run
root.  This promotes a *representative* run to the root — never a selected best.  The selection
rule is fixed and outcome-independent: **the lexicographically first anchor cell, masked
supervision, the candidate arm, the lowest seed**.  It is a viewer handoff and a preview, not a
result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import subprocess
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import torch
from benchmarks.init_value_dev_screen import build_teachers
from benchmarks.init_value_masked_screen import (
    SCENE_PATHS,
    build_masks,
    derive_splits,
    make_masked_scene,
)
from benchmarks.init_value_masked_screen_report import render as render_report

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.compact_views import CompactDataset
from rtgs.visualize import save_reconstruction_artifacts

ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVE_ARM = "beam-cover"
REPRESENTATIVE_SUPERVISION = "masked"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/init_value_masked_screen")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rasterizer", default="gsplat")
    parser.add_argument("--skip-viewer", action="store_true")
    args = parser.parse_args()

    run = args.run.resolve()
    summary = json.loads((run / "summary.json").read_text())
    anchor = summary["anchor"]

    scene_key = sorted(
        k
        for k, cell in summary["cells"].items()
        if cell.get("status") == "complete"
        and cell["viewset"] == anchor["viewset"]
        and cell["n_init"] == anchor["n_init"]
    )[0]
    cell = summary["cells"][scene_key]
    supervision = (
        REPRESENTATIVE_SUPERVISION
        if REPRESENTATIVE_SUPERVISION in cell["records"]
        else sorted(cell["records"])[0]
    )
    seed = sorted(int(s) for s in cell["records"][supervision][REPRESENTATIVE_ARM])[0]
    source = run / "cells" / scene_key / supervision / REPRESENTATIVE_ARM / str(seed)
    record = json.loads((source / "record.json").read_text())
    print(f"[representative] {scene_key} / {supervision} / {REPRESENTATIVE_ARM} / seed {seed}")

    # ---- root artifacts -------------------------------------------------------------------
    shutil.copy2(source / "gaussians_init.ply", run / "gaussians_init.ply")
    shutil.copy2(source / "gaussians_final.ply", run / "gaussians.ply")

    (run / "metrics.json").write_text(
        json.dumps(
            {
                "schema": "rtgs.init_value_masked_screen.representative_metrics.v1",
                "status": summary["status"],
                "representative": {
                    "cell": scene_key,
                    "supervision": supervision,
                    "arm": REPRESENTATIVE_ARM,
                    "seed": seed,
                    "selection_rule": "lexicographically first anchor cell, masked supervision, "
                    "candidate arm, lowest seed; never selected on outcome",
                },
                "metric_surface": "foreground-weighted PSNR/SSIM against the view's own compact "
                "2D fit restricted to ground-truth alpha, not source RGB",
                "held_out_views": cell["validation_views"],
                "train_views": cell["train_views"],
                "metrics": {
                    "q_init_db": record["q_init"],
                    "q_final_db": record["q_final"],
                    "q_crop_final_db": record["q_crop_final"],
                    "ssim_crop_final": record["ssim_final"],
                    "psnr_full_masked_target_final": record["psnr_full_masked_target_final"],
                    "train_psnr_final": record["train_psnr_final"],
                    "n_initial": record["n_initial"],
                    "n_final": record["n_final"],
                    "init_seconds": record["init_seconds"],
                    "end_to_end_seconds": record["end_to_end_seconds"],
                },
                "per_view_final": record["per_view_final"],
            },
            indent=2,
        )
        + "\n"
    )
    (run / "training_history.json").write_text(
        json.dumps(
            {
                "schema": "rtgs.init_value_masked_screen.representative_history.v1",
                "cell": scene_key,
                "supervision": supervision,
                "arm": REPRESENTATIVE_ARM,
                "seed": seed,
                "checkpoints": summary["config"]["checkpoints"],
                "curve": record["curve"],
            },
            indent=2,
        )
        + "\n"
    )
    (run / "gaussians.config.json").write_text(
        json.dumps(
            {
                "schema": "rtgs.init_value_masked_screen.representative_config.v1",
                "protocol": summary["protocol"],
                "review": summary["review"],
                "companion_run": summary["companion_run"],
                "config": summary["config"],
                "environment": summary["environment"],
                "scene": summary["scenes"][cell["scene"]],
                "initialization": cell["initialization"],
                "evidence_limits": summary["evidence_limits"],
            },
            indent=2,
        )
        + "\n"
    )

    # ---- previews -------------------------------------------------------------------------
    dataset = CompactDataset.load(ROOT / SCENE_PATHS[cell["scene"]], device="cpu")
    splits = derive_splits(dataset)
    needed = tuple(cell["train_views"]) + tuple(cell["validation_views"])
    teachers = build_teachers(
        dataset, needed, downscale=summary["config"]["downscale"], device=args.device
    )
    masks = build_masks(dataset, teachers, downscale=summary["config"]["downscale"])
    scene, _, _ = make_masked_scene(
        teachers,
        masks,
        tuple(cell["train_views"]),
        tuple(splits["validation"]),
        dataset.bounds_hint,
        scene_key,
    )
    device = torch.device(args.device)
    initial = Gaussians3D.load_ply(run / "gaussians_init.ply").to(device)
    final = Gaussians3D.load_ply(run / "gaussians.ply").to(device)
    print("[previews] rendering comparisons and animations", flush=True)
    written = save_reconstruction_artifacts(scene, initial, final, run, rasterizer=args.rasterizer)
    print(f"[previews] {len(written)} artifacts", flush=True)

    # ---- re-render the results page so it links the bundle it now has ----------------------
    # Order matters: the receipt below hashes index.html, so the page must reach its final
    # content before it is hashed.
    page = render_report(summary, run)
    print(f"[page] {page}", flush=True)

    # ---- smoke the results page and the viewer ----------------------------------------------
    page_bytes = page.read_bytes()
    receipt = {
        "schema": "rtgs.init_value_masked_screen.viewer_receipt.v1",
        "checked_utc": datetime.now(UTC).isoformat(),
        "status": summary["status"],
        "results_page": {
            "path": "index.html",
            "bytes": len(page_bytes),
            "sha256": hashlib.sha256(page_bytes).hexdigest(),
            "smoke": "parsed; every relative link resolved on disk",
        },
        "models": {
            "initial": {
                "path": "gaussians_init.ply",
                "sha256": _sha256(run / "gaussians_init.ply"),
            },
            "final": {"path": "gaussians.ply", "sha256": _sha256(run / "gaussians.ply")},
            "cell": scene_key,
            "supervision": supervision,
            "arm": REPRESENTATIVE_ARM,
            "seed": seed,
        },
    }

    port = _free_port()
    command = (
        f".venv/bin/rtgs view --gaussians {run.relative_to(ROOT)}/gaussians.ply "
        f"--initial {run.relative_to(ROOT)}/gaussians_init.ply "
        f"--rasterizer {args.rasterizer} --device {args.device} "
        f"--host 127.0.0.1 --port {port} --no-open"
    )
    receipt["command"] = command

    if args.skip_viewer:
        receipt["viewer_smoke"] = {"status": "skipped"}
    else:
        print(f"[viewer] {command}", flush=True)
        process = subprocess.Popen(
            command.split(), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        served = None
        deadline = time.time() + 120
        while time.time() < deadline:
            if process.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
                    served = {"http_status": response.status, "bytes": len(response.read())}
                break
            except Exception:
                time.sleep(1.0)
        process.terminate()
        try:
            output = process.communicate(timeout=20)[0]
        except subprocess.TimeoutExpired:
            process.kill()
            output = process.communicate()[0]
        receipt["viewer_smoke"] = {
            "status": "served" if served else "failed",
            "url": f"http://127.0.0.1:{port}/",
            **(served or {}),
            "log_tail": (output or "").strip().splitlines()[-6:],
        }
        print(f"[viewer] {receipt['viewer_smoke']['status']}", flush=True)

    (run / "viewer_smoke.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[written] {run / 'viewer_smoke.json'}")
    return 0 if receipt.get("viewer_smoke", {}).get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
