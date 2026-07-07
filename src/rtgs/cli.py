"""rtgs command-line interface (argparse; see docs/ARCHITECTURE.md §CLI).

Commands: fit-images, lift, refine, run, render, bench. `--scene synthetic[:key=val,..]`
builds a procedural test scene; any other value is treated as a COLMAP dataset directory
(sparse/0 + images/).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _load_scene(spec: str, downscale: int = 1, max_images: int | None = None):
    from rtgs.data.synthetic import make_synthetic_scene

    if spec == "synthetic" or spec.startswith("synthetic:"):
        kwargs: dict = {}
        if ":" in spec:
            for kv in spec.split(":", 1)[1].split(","):
                k, v = kv.split("=")
                kwargs[k] = int(v)
        return make_synthetic_scene(**kwargs)
    from rtgs.data.colmap import load_colmap_scene

    return load_colmap_scene(spec, downscale=downscale, max_images=max_images)


def _save_gaussians(g, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".ply":
        g.save_ply(out)
    else:
        g.save_npz(out)
    print(f"saved {g.n} gaussians -> {out}")


def _load_gaussians(path: Path):
    from rtgs.core.gaussians3d import Gaussians3D

    if path.suffix == ".ply":
        return Gaussians3D.load_ply(path)
    return Gaussians3D.load_npz(path)


def _cmd_fit_images(args: argparse.Namespace) -> int:
    import numpy as np
    from PIL import Image as PILImage

    from rtgs.image2gs.fit import FitConfig, fit_image

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = FitConfig(n_gaussians=args.n_gaussians, iterations=args.iterations)
    paths = sorted(
        p for p in Path(args.images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not paths:
        print(f"no images found in {args.images}", file=sys.stderr)
        return 1
    for i, p in enumerate(paths):
        img = torch.from_numpy(
            np.asarray(PILImage.open(p).convert("RGB"), dtype=np.float32) / 255.0
        )
        g, hist = fit_image(img, cfg, seed=args.seed + i)
        g.save_npz(out_dir / f"{p.stem}.npz")
        print(f"{p.name}: {g.n} gaussians, PSNR {hist['final_psnr']:.2f} dB")
    return 0


def _cmd_lift(args: argparse.Namespace) -> int:
    from rtgs.core.gaussians2d import Gaussians2D
    from rtgs.lift import get_lifter

    scene = _load_scene(args.scene, downscale=args.downscale, max_images=args.max_images)
    fits_dir = Path(args.fits)
    g2ds = [Gaussians2D.load_npz(p) for p in sorted(fits_dir.glob("*.npz"))]
    if len(g2ds) != scene.n_views:
        print(f"found {len(g2ds)} fits for {scene.n_views} views", file=sys.stderr)
        return 1
    lifter = get_lifter(args.lifter, **json.loads(args.lifter_args))
    g3d = lifter.lift(g2ds, scene)
    _save_gaussians(g3d, Path(args.out))
    return 0


def _cmd_refine(args: argparse.Namespace) -> int:
    from rtgs.optim.trainer import TrainConfig, Trainer

    scene = _load_scene(args.scene, downscale=args.downscale, max_images=args.max_images)
    init = _load_gaussians(Path(args.init))
    cfg = TrainConfig(iterations=args.iterations, rasterizer=args.rasterizer)
    refined, history = Trainer(cfg).train(scene, init)
    print(f"PSNR: {history['psnr']}")
    _save_gaussians(refined, Path(args.out))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from rtgs.image2gs.fit import FitConfig
    from rtgs.optim.trainer import TrainConfig
    from rtgs.pipeline import PipelineConfig, run_pipeline

    scene = _load_scene(args.scene, downscale=args.downscale, max_images=args.max_images)
    cfg = PipelineConfig(
        fit=FitConfig(n_gaussians=args.n_gaussians, iterations=args.fit_iterations),
        lifter=args.lifter,
        lifter_kwargs=json.loads(args.lifter_args),
        train=TrainConfig(iterations=args.refine_iters, rasterizer=args.rasterizer),
        refine=args.refine_iters > 0,
        seed=args.seed,
    )
    result = run_pipeline(scene, cfg)
    print(json.dumps({"metrics": result.metrics, "timings": result.timings}, indent=2))
    if args.out:
        out = Path(args.out)
        _save_gaussians(result.gaussians, out / "gaussians.ply")
        (out / "metrics.json").write_text(
            json.dumps({"metrics": result.metrics, "timings": result.timings}, indent=2)
        )
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    import numpy as np
    from PIL import Image as PILImage

    from rtgs.render.base import get_rasterizer

    scene = _load_scene(args.scene, downscale=args.downscale, max_images=args.max_images)
    g = _load_gaussians(Path(args.gaussians))
    renderer = get_rasterizer(args.rasterizer)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i, cam in enumerate(scene.cameras):
            img = renderer.render(g, cam).color.clamp(0, 1)
            PILImage.fromarray((img.cpu().numpy() * 255).astype(np.uint8)).save(
                out_dir / f"view_{i:04d}.png"
            )
    print(f"rendered {scene.n_views} views -> {out_dir}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from rtgs.bench import main as bench_main

    return bench_main(args.bench_args)


def _add_scene_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--scene", required=True, help="'synthetic[:k=v,..]' or COLMAP dataset dir")
    p.add_argument("--downscale", type=int, default=1)
    p.add_argument("--max-images", type=int, default=None)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `rtgs` executable."""
    parser = argparse.ArgumentParser(prog="rtgs", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fit-images", help="stage 1: fit 2D gaussians per image")
    p.add_argument("--images", required=True, help="directory of images")
    p.add_argument("--out", required=True, help="output directory for per-image .npz")
    p.add_argument("--n-gaussians", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_fit_images)

    p = sub.add_parser("lift", help="stage 2: lift 2D fits into a 3D gaussian set")
    _add_scene_args(p)
    p.add_argument("--fits", required=True, help="directory of stage-1 .npz files")
    p.add_argument("--lifter", default="depth")
    p.add_argument("--lifter-args", default="{}", help="JSON kwargs for the lifter")
    p.add_argument("--out", required=True, help="output .npz or .ply")
    p.set_defaults(func=_cmd_lift)

    p = sub.add_parser("refine", help="stage 3: 3DGS optimization from an initialization")
    _add_scene_args(p)
    p.add_argument("--init", required=True, help="input .npz or .ply")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--rasterizer", default="auto")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_refine)

    p = sub.add_parser("run", help="end-to-end: fit -> lift -> refine")
    _add_scene_args(p)
    p.add_argument("--lifter", default="depth")
    p.add_argument("--lifter-args", default="{}")
    p.add_argument("--n-gaussians", type=int, default=512, help="2D gaussians per image")
    p.add_argument("--fit-iterations", type=int, default=300)
    p.add_argument("--refine-iters", type=int, default=200)
    p.add_argument("--rasterizer", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="optional output directory")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("render", help="render a saved gaussian set from scene cameras")
    _add_scene_args(p)
    p.add_argument("--gaussians", required=True)
    p.add_argument("--rasterizer", default="auto")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_render)

    p = sub.add_parser("bench", help="run the benchmark harness (rtgs.bench)")
    p.add_argument(
        "bench_args",
        nargs=argparse.REMAINDER,
        help="forwarded to rtgs.bench (e.g. --quick --update-docs)",
    )
    p.set_defaults(func=_cmd_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
