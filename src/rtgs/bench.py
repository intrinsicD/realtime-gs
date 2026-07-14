"""Benchmark harness: micro-benchmarks + variant comparison, tracked as JSON.

Invoked via ``python benchmarks/run.py`` (thin wrapper) or ``rtgs bench``. Results land
in ``benchmarks/results/<timestamp>_<device>.json``; ``--update-docs`` rewrites the
table between the BENCH markers in docs/BENCHMARKS.md. ``--smoke`` runs a minimal
configuration and writes nothing (CI).
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from rtgs.data.synthetic import make_synthetic_scene
from rtgs.image2gs.fit import FitConfig, fit_image
from rtgs.lift import lifter_names
from rtgs.optim.density import DensityConfig
from rtgs.optim.trainer import TrainConfig
from rtgs.pipeline import PipelineConfig, compare_lifters
from rtgs.render.torch_ref import TorchRasterizer

_BEGIN = "<!-- BENCH:BEGIN -->"
_END = "<!-- BENCH:END -->"


@dataclass
class BenchConfig:
    """Benchmark scene/workload sizes."""

    image_size: int = 48
    n_views: int = 12
    n_gt_gaussians: int = 40
    fit_gaussians: int = 150
    fit_iterations: int = 120
    refine_iterations: int = 150
    render_repeats: int = 3

    @staticmethod
    def quick() -> BenchConfig:
        return BenchConfig()

    @staticmethod
    def full() -> BenchConfig:
        return BenchConfig(
            image_size=96,
            n_views=20,
            n_gt_gaussians=80,
            fit_gaussians=400,
            fit_iterations=300,
            refine_iterations=600,
            render_repeats=10,
        )

    @staticmethod
    def smoke() -> BenchConfig:
        return BenchConfig(
            image_size=24,
            n_views=6,
            n_gt_gaussians=15,
            fit_gaussians=40,
            fit_iterations=25,
            refine_iterations=15,
            render_repeats=1,
        )


def _git_rev(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def run_benchmarks(config: BenchConfig, smoke: bool = False) -> dict:
    """Execute all benchmarks; returns {'meta': ..., 'results': ...}."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench_device = torch.device(device)
    scene = make_synthetic_scene(
        n_gaussians=config.n_gt_gaussians,
        n_cameras=config.n_views,
        image_size=config.image_size,
        seed=0,
    )
    results: dict = {}

    # Stage-1 fitting speed/quality.
    fit_cfg = FitConfig(n_gaussians=config.fit_gaussians, iterations=config.fit_iterations)
    t0 = time.perf_counter()
    _, hist = fit_image(scene.images[0].to(bench_device), fit_cfg, seed=0)
    if bench_device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    results["image2gs_fit"] = {
        "iters_per_s": (hist["stopped_iter"] + 1) / dt,
        "psnr": hist["final_psnr"],
        "seconds": dt,
    }

    # Reference renderer throughput.
    renderer = TorchRasterizer()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(config.render_repeats):
            for cam in scene.cameras:
                renderer.render(scene.gt_gaussians, cam)
    dt = time.perf_counter() - t0
    n_frames = config.render_repeats * len(scene.cameras)
    results["render_ref_cpu"] = {"fps": n_frames / dt, "frames": n_frames, "seconds": dt}

    # Variant comparison (init quality + refinement).
    variants = (
        ["depth", "hybrid", "gradient", "carve", "sfm", "random"] if not smoke else ["depth", "sfm"]
    )
    assert set(variants) <= set(lifter_names())
    pipe_cfg = PipelineConfig(
        fit=fit_cfg,
        train=TrainConfig(
            iterations=config.refine_iterations,
            density=DensityConfig(start_iter=40, every=40),
            eval_every=max(config.refine_iterations // 2, 1),
        ),
    )
    if bench_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    comparison = compare_lifters(scene, {v: {} for v in variants}, pipe_cfg)
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if bench_device.type == "cuda" else 0.0
    for name, res in comparison.items():
        results[f"lift_{name}"] = {
            "seconds": res.timings["lift"],
            "init_psnr": res.metrics["init_psnr"],
            "init_n_gaussians": res.metrics["init_n_gaussians"],
            "fit_seconds": res.timings["fit"],
        }
        results[f"e2e_{name}"] = {
            "init_psnr": res.metrics["init_psnr"],
            "final_psnr": res.metrics["final_psnr"],
            "final_n_gaussians": res.metrics["final_n_gaussians"],
            "refine_seconds": res.timings["refine"],
            "fit_seconds": res.timings["fit"],
            "lift_seconds": res.timings["lift"],
            "total_seconds": res.timings["total"],
            "peak_vram_mb": peak_vram,
            "psnr_curve": res.train_history.get("psnr", []),
            "seconds_curve": res.train_history.get("elapsed", []),
        }

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "scene": scene.name,
        "config": config.__dict__,
    }
    return {"meta": meta, "results": results}


def _format_table(payload: dict) -> str:
    meta = payload["meta"]
    lines = [
        f"_Last run: {meta['timestamp']} · device `{meta['device']}` · torch {meta['torch']}"
        f" · rev `{meta.get('git_rev', 'unknown')}` · scene `{meta['scene']}`_",
        "",
        "| benchmark | key numbers |",
        "| --- | --- |",
    ]
    for name, vals in payload["results"].items():
        nums = " · ".join(
            f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in vals.items()
        )
        lines.append(f"| `{name}` | {nums} |")
    return "\n".join(lines)


def update_docs(payload: dict, docs_path: Path) -> None:
    """Rewrite the block between the BENCH markers in docs/BENCHMARKS.md."""
    text = docs_path.read_text(encoding="utf-8")
    if _BEGIN not in text or _END not in text:
        raise ValueError(f"{docs_path} is missing the BENCH markers")
    head, rest = text.split(_BEGIN, 1)
    _, tail = rest.split(_END, 1)
    docs_path.write_text(
        head + _BEGIN + "\n" + _format_table(payload) + "\n" + _END + tail, encoding="utf-8"
    )


def find_repo_root(start: Path | None = None) -> Path | None:
    """Walk upward until a directory containing pyproject.toml is found."""
    p = (start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (shared by benchmarks/run.py and `rtgs bench`)."""
    import argparse

    parser = argparse.ArgumentParser(description="rtgs benchmark harness")
    parser.add_argument("--quick", action="store_true", help="CPU-sized configuration")
    parser.add_argument("--smoke", action="store_true", help="minimal run, writes nothing (CI)")
    parser.add_argument("--update-docs", action="store_true", help="rewrite docs/BENCHMARKS.md")
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.smoke:
        config = BenchConfig.smoke()
    elif args.quick:
        config = BenchConfig.quick()
    else:
        config = BenchConfig.full()

    payload = run_benchmarks(config, smoke=args.smoke)
    root = args.repo_root or find_repo_root()
    if root is not None:
        payload["meta"]["git_rev"] = _git_rev(root)

    print(json.dumps(payload, indent=2))
    if args.smoke:
        return 0
    if root is None:
        print("warning: repo root not found; results not saved")
        return 0

    out_dir = root / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = payload["meta"]["timestamp"].replace(":", "").replace("-", "").replace("+0000", "Z")
    out_path = out_dir / f"{stamp}_{payload['meta']['device']}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {out_path}")

    if args.update_docs:
        update_docs(payload, root / "docs" / "BENCHMARKS.md")
        print("updated docs/BENCHMARKS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
