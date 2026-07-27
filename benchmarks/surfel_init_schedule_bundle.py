#!/usr/bin/env python3
"""Complete the ADR-XXXX / ADR-YYYY screen's bundle to Hard Rule 7 shape.

The screen writes many sub-runs; Rule 7 expects one ``rtgs run``-shaped bundle at the run root.
This promotes a **representative** run, never a selected best.  The rule is fixed and
outcome-independent: *the lexicographically first scene, the anchor arm
``surfel-rho0.1`` under ``none/notrust``, the lowest seed*.  It is a viewer handoff and a
preview, not a result.
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
    VALIDATION,
    build_masks,
    derive_splits,
    make_masked_scene,
)
from benchmarks.surfel_init_schedule_report import render as render_report

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.compact_views import CompactDataset
from rtgs.visualize import save_reconstruction_artifacts

ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVE_ARM = "surfel-rho0.1"
REPRESENTATIVE_LEVEL = "none__notrust"


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


def _write_comparison_manifest(run: Path, summary: dict, scene_key: str) -> Path:
    """One synchronized WebGL comparison over every arm and every schedule level.

    This is the handoff the question "what did the init and the schedule actually do?" needs:
    the same orbit camera over each arm's initial and final state, so a difference is seen
    rather than inferred from a table.  Paths are relative to the manifest's own directory,
    which is where ``rtgs view --comparison-manifest`` resolves them.
    """
    methods = []
    seed = sorted(int(s) for s in summary["screen_a"][scene_key]["records"][REPRESENTATIVE_ARM])[0]
    for arm in summary["config"]["arms"]:
        cell = f"{scene_key}__{arm}__none__notrust__{seed}"
        if (run / "cells" / cell / "gaussians_final.ply").is_file():
            methods.append(
                {
                    "name": f"A:{arm}",
                    "initial": f"cells/{cell}/gaussians_init.ply",
                    "final": f"cells/{cell}/gaussians_final.ply",
                }
            )
    for init_name, levels in (
        summary.get("screen_b", {}).get(scene_key, {}).get("records", {}).items()
    ):
        for level in levels:
            cell = f"{scene_key}__{init_name}__{level.replace('/', '__')}__{seed}"
            if (run / "cells" / cell / "gaussians_final.ply").is_file():
                methods.append(
                    {
                        "name": f"B:{init_name}:{level}",
                        "initial": f"cells/{cell}/gaussians_init.ply",
                        "final": f"cells/{cell}/gaussians_final.ply",
                    }
                )
    path = run / "comparison.json"
    path.write_text(
        json.dumps({"schema": "rtgs.viewer-comparison.v1", "methods": methods}, indent=2) + "\n"
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/surfel_init_schedule_screen")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rasterizer", default="gsplat")
    parser.add_argument("--skip-viewer", action="store_true")
    parser.add_argument("--skip-previews", action="store_true")
    args = parser.parse_args()

    run = args.run.resolve()
    summary = json.loads((run / "summary.json").read_text())
    scene_key = sorted(summary["screen_a"])[0]
    by_seed = summary["screen_a"][scene_key]["records"][REPRESENTATIVE_ARM]
    seed = sorted(int(s) for s in by_seed)[0]
    record = by_seed[str(seed)]
    cell = run / "cells" / f"{scene_key}__{REPRESENTATIVE_ARM}__{REPRESENTATIVE_LEVEL}__{seed}"
    print(f"[representative] {scene_key} / {REPRESENTATIVE_ARM} / seed {seed}")

    shutil.copy2(cell / "gaussians_init.ply", run / "gaussians_init.ply")
    shutil.copy2(cell / "gaussians_final.ply", run / "gaussians.ply")

    (run / "metrics.json").write_text(
        json.dumps(
            {
                "schema": "rtgs.surfel_init_schedule_screen.representative_metrics.v1",
                "status": summary["status"],
                "representative": {
                    "scene": scene_key,
                    "arm": REPRESENTATIVE_ARM,
                    "level": "none/notrust",
                    "seed": seed,
                    "selection_rule": "lexicographically first scene, anchor arm, fixed "
                    "topology, lowest seed; never selected on outcome",
                },
                "metric_surface": "median per-view foreground-weighted PSNR against each view's "
                "own compact 2D fit composited on black; not source RGB",
                "metrics": {
                    "init_alpha_iou": record["init_alpha_iou"],
                    "init_alpha_mean": record["init_alpha_mean"],
                    "q_init_db": record["q_init"],
                    "q_final_db": record["q_final"],
                    "ssim_final": record["ssim_final"],
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
                "schema": "rtgs.surfel_init_schedule_screen.representative_history.v1",
                "scene": scene_key,
                "arm": REPRESENTATIVE_ARM,
                "seed": seed,
                "curve": record["curve"],
            },
            indent=2,
        )
        + "\n"
    )
    (run / "gaussians.config.json").write_text(
        json.dumps(
            {
                "schema": "rtgs.surfel_init_schedule_screen.representative_config.v1",
                "adrs": summary["adrs"],
                "config": summary["config"],
                "environment": summary["environment"],
                "scene": summary["scenes"][scene_key],
                "init_diagnostics": summary["screen_a"][scene_key]["init_diagnostics"][
                    REPRESENTATIVE_ARM
                ],
            },
            indent=2,
        )
        + "\n"
    )

    manifest = _write_comparison_manifest(run, summary, scene_key)
    print(f"[manifest] {manifest}")

    device = torch.device(args.device)
    if not args.skip_previews:
        dataset = CompactDataset.load(ROOT / SCENE_PATHS[scene_key], device="cpu")
        splits = derive_splits(dataset)
        needed = tuple(splits["train8"]) + tuple(VALIDATION)
        teachers = build_teachers(
            dataset, needed, downscale=summary["config"]["downscale"], device=args.device
        )
        masks = build_masks(dataset, teachers, downscale=summary["config"]["downscale"])
        scene, _, _ = make_masked_scene(
            teachers, masks, splits["train3"], VALIDATION, dataset.bounds_hint, scene_key
        )
        initial = Gaussians3D.load_ply(run / "gaussians_init.ply").to(device)
        final = Gaussians3D.load_ply(run / "gaussians.ply").to(device)
        print("[previews] rendering comparisons and animations", flush=True)
        written = save_reconstruction_artifacts(
            scene, initial, final, run, rasterizer=args.rasterizer
        )
        print(f"[previews] {len(written)} artifacts", flush=True)

    # Order matters: the receipt hashes index.html, so the page must be final before hashing.
    page = render_report(summary, run)
    print(f"[page] {page}", flush=True)

    page_bytes = page.read_bytes()
    receipt = {
        "schema": "rtgs.surfel_init_schedule_screen.viewer_receipt.v1",
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
            "scene": scene_key,
            "arm": REPRESENTATIVE_ARM,
            "seed": seed,
        },
    }

    port = _free_port()
    shown = run.relative_to(ROOT) if run.is_relative_to(ROOT) else run
    command = (
        f".venv/bin/rtgs view --gaussians {shown}/gaussians.ply "
        f"--initial {shown}/gaussians_init.ply "
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
