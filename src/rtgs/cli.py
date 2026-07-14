"""rtgs command-line interface (argparse; see docs/ARCHITECTURE.md §CLI).

Commands: fit-images, lift, refine, run, render, bench. `--scene synthetic[:key=val,..]`
builds a procedural test scene; calibrated frame directories and COLMAP datasets are detected.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _resolve_device(requested: str) -> torch.device:
    return torch.device(
        "cuda"
        if requested == "auto" and torch.cuda.is_available()
        else ("cpu" if requested == "auto" else requested)
    )


def _load_scene(spec: str, downscale: int = 1, max_images: int | None = None):
    from rtgs.data.synthetic import make_synthetic_scene

    if spec == "synthetic" or spec.startswith("synthetic:"):
        kwargs: dict = {}
        if ":" in spec:
            for kv in spec.split(":", 1)[1].split(","):
                k, v = kv.split("=")
                kwargs[k] = int(v)
        return make_synthetic_scene(**kwargs)
    path = Path(spec).expanduser()
    frame = path.parent if path.name == "rgb" else path
    if (frame / "rgb").is_dir():
        from rtgs.data.calibrated import load_calibrated_scene

        try:
            return load_calibrated_scene(frame, downscale=downscale, max_images=max_images)
        except FileNotFoundError:
            pass

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
    device = _resolve_device(args.device)
    cfg = FitConfig(
        n_gaussians=args.n_gaussians,
        max_gaussians=args.max_gaussians,
        iterations=args.iterations,
        backend=args.fit_backend,
        adaptive_density=args.adaptive_density,
        growth_waves=args.growth_waves,
        relocate_fraction=args.relocate_fraction,
        structsplat_renderer=args.structsplat_renderer,
    )
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
        g, hist = fit_image(img.to(device), cfg, seed=args.seed + i)
        g.save_npz(out_dir / f"{p.stem}.npz")
        print(f"{p.name}: {g.n} gaussians, PSNR {hist['final_psnr']:.2f} dB")
    return 0


def _cmd_lift(args: argparse.Namespace) -> int:
    from rtgs.lift import get_lifter

    scene = _load_scene(args.scene, downscale=args.downscale, max_images=args.max_images)
    g2ds = _load_2d_fits(Path(args.fits), scene, args.fit_format)
    train_indices = scene.training_views
    if len(g2ds) not in (scene.n_views, len(train_indices)):
        print(
            f"found {len(g2ds)} fits; expected {scene.n_views} all-view or "
            f"{len(train_indices)} train-only fits",
            file=sys.stderr,
        )
        return 1
    scene = scene.subset(train_indices)
    if len(g2ds) != len(train_indices):
        g2ds = [g2ds[i] for i in train_indices]
    device = _resolve_device(args.device)
    lifter = get_lifter(args.lifter, **json.loads(args.lifter_args))
    g3d = lifter.lift([g.to(device) for g in g2ds], scene.to(device))
    _save_gaussians(g3d, Path(args.out))
    return 0


def _cmd_refine(args: argparse.Namespace) -> int:
    from rtgs.optim.trainer import TrainConfig, Trainer

    scene = _load_scene(args.scene, downscale=args.downscale, max_images=args.max_images)
    init = _load_gaussians(Path(args.init))
    cfg = TrainConfig(
        iterations=args.iterations,
        rasterizer=args.rasterizer,
        device=args.device,
        densify=args.densify,
        density=_density_config(args),
    )
    refined, history = Trainer(cfg).train(scene, init)
    print(f"PSNR: {history['psnr']}")
    _save_gaussians(refined, Path(args.out))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from rtgs.image2gs.fit import FitConfig
    from rtgs.optim.density import DensityConfig
    from rtgs.optim.trainer import TrainConfig
    from rtgs.pipeline import PipelineConfig, run_pipeline

    scene = _load_scene(args.scene, downscale=args.downscale, max_images=args.max_images)
    cfg = PipelineConfig(
        fit=FitConfig(
            n_gaussians=args.n_gaussians,
            max_gaussians=args.max_gaussians,
            iterations=args.fit_iterations,
            backend=args.fit_backend,
            adaptive_density=args.adaptive_density,
            growth_waves=args.growth_waves,
            relocate_fraction=args.relocate_fraction,
            structsplat_renderer=args.structsplat_renderer,
        ),
        lifter=args.lifter,
        lifter_kwargs=json.loads(args.lifter_args),
        train=TrainConfig(
            iterations=args.refine_iters,
            rasterizer=args.rasterizer,
            densify=args.densify,
            density=DensityConfig(
                start_iter=args.densify_start,
                stop_iter=args.densify_stop,
                every=args.densify_every,
                max_gaussians=args.max_3d_gaussians,
            ),
        ),
        refine=args.refine_iters > 0,
        device=args.device,
        seed=args.seed,
    )
    fits = None if args.fits is None else _load_2d_fits(Path(args.fits), scene, args.fit_format)
    result = run_pipeline(scene, cfg, gaussians2d=fits)
    print(json.dumps({"metrics": result.metrics, "timings": result.timings}, indent=2))
    if args.out:
        out = Path(args.out)
        _save_gaussians(result.gaussians_init, out / "gaussians_init.ply")
        _save_gaussians(result.gaussians, out / "gaussians.ply")
        (out / "metrics.json").write_text(
            json.dumps({"metrics": result.metrics, "timings": result.timings}, indent=2)
        )
        if args.preview:
            from rtgs.visualize import save_reconstruction_artifacts

            artifacts = save_reconstruction_artifacts(
                scene,
                result.gaussians_init,
                result.gaussians,
                out,
                rasterizer=args.rasterizer,
            )
            print(f"visual reconstruction -> {artifacts['contact_sheet']}")
            print(f"turntable -> {artifacts['turntable']}")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    import numpy as np
    from PIL import Image as PILImage

    from rtgs.render.base import get_rasterizer

    scene = _load_scene(args.scene, downscale=args.downscale, max_images=args.max_images)
    device = _resolve_device(args.device)
    scene = scene.to(device)
    g = _load_gaussians(Path(args.gaussians)).to(device)
    renderer = get_rasterizer(args.rasterizer, device=g.means.device)
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


def _load_2d_fits(fits_dir: Path, scene, source: str):
    """Load per-view fits in scene order instead of unsafe lexicographic order."""
    from rtgs.image2gs.adapters import load_gaussians2d

    paths = sorted(fits_dir.expanduser().glob("*.npz"))
    if scene.view_names is not None:
        ordered = []
        missing = []
        for index, name in enumerate(scene.view_names):
            matches = [
                path for path in paths if path.stem == name or path.stem.startswith(name + "_")
            ]
            if not matches and index in scene.testing_views:
                missing.append(index)
                continue
            if len(matches) != 1:
                raise ValueError(
                    f"expected one 2D fit for view '{name}' in {fits_dir}, found {len(matches)}"
                )
            ordered.append(matches[0])
        if missing and len(ordered) != len(scene.training_views):
            raise ValueError("partial 2D fits must contain every training view")
        paths = ordered
    return [load_gaussians2d(path, source=source) for path in paths]


def _add_scene_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--scene",
        required=True,
        help="'synthetic[:k=v,..]', calibrated frame dir, or COLMAP dataset dir",
    )
    p.add_argument("--downscale", type=int, default=1)
    p.add_argument("--max-images", type=int, default=None)


def _density_config(args: argparse.Namespace):
    from rtgs.optim.density import DensityConfig

    return DensityConfig(
        start_iter=args.densify_start,
        stop_iter=args.densify_stop,
        every=args.densify_every,
        max_gaussians=args.max_3d_gaussians,
    )


def _add_density_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--densify", action=argparse.BooleanOptionalAction, default=True, help="3D density control"
    )
    p.add_argument("--densify-start", type=int, default=60)
    p.add_argument("--densify-stop", type=int, default=10_000_000)
    p.add_argument("--densify-every", type=int, default=40)
    p.add_argument("--max-3d-gaussians", type=int, default=100_000)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `rtgs` executable."""
    parser = argparse.ArgumentParser(prog="rtgs", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fit-images", help="stage 1: fit 2D gaussians per image")
    p.add_argument("--images", required=True, help="directory of images")
    p.add_argument("--out", required=True, help="output directory for per-image .npz")
    p.add_argument(
        "--initial-gaussians",
        "--n-gaussians",
        dest="n_gaussians",
        type=int,
        default=640,
        help="initial 2D Gaussian count (native backend keeps this fixed)",
    )
    p.add_argument("--iterations", type=int, default=300)
    _add_fit_backend_args(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    p.set_defaults(func=_cmd_fit_images)

    p = sub.add_parser("lift", help="stage 2: lift 2D fits into a 3D gaussian set")
    _add_scene_args(p)
    p.add_argument("--fits", required=True, help="directory of stage-1 .npz files")
    p.add_argument("--lifter", default="depth")
    p.add_argument(
        "--fit-format",
        choices=["auto", "native", "structsplat", "gaussianimage"],
        default="auto",
    )
    p.add_argument("--lifter-args", default="{}", help="JSON kwargs for the lifter")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    p.add_argument("--out", required=True, help="output .npz or .ply")
    p.set_defaults(func=_cmd_lift)

    p = sub.add_parser("refine", help="stage 3: 3DGS optimization from an initialization")
    _add_scene_args(p)
    p.add_argument("--init", required=True, help="input .npz or .ply")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--rasterizer", default="auto")
    _add_density_args(p)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_refine)

    p = sub.add_parser("run", help="end-to-end: fit -> lift -> refine")
    _add_scene_args(p)
    p.add_argument("--lifter", default="depth")
    p.add_argument("--lifter-args", default="{}")
    p.add_argument("--fits", default=None, help="optional native/StructSplat .npz directory")
    p.add_argument(
        "--fit-format",
        choices=["auto", "native", "structsplat", "gaussianimage"],
        default="auto",
    )
    p.add_argument(
        "--initial-gaussians",
        "--n-gaussians",
        dest="n_gaussians",
        type=int,
        default=640,
        help="initial 2D Gaussian count per image",
    )
    p.add_argument("--fit-iterations", type=int, default=300)
    _add_fit_backend_args(p)
    p.add_argument("--refine-iters", type=int, default=200)
    p.add_argument("--rasterizer", default="auto")
    _add_density_args(p)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="optional output directory")
    p.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="with --out, save comparisons and an animated reconstruction turntable",
    )
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("render", help="render a saved gaussian set from scene cameras")
    _add_scene_args(p)
    p.add_argument("--gaussians", required=True)
    p.add_argument("--rasterizer", default="auto")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
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


def _add_fit_backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--fit-backend",
        choices=["native", "structsplat"],
        default="native",
        help="optional StructSplat backend grows from the configured initial count",
    )
    p.add_argument(
        "--max-gaussians",
        type=int,
        default=5_000,
        help="maximum per-image count for StructSplat density growth",
    )
    p.add_argument(
        "--adaptive-density",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="grow until convergence/stall or max count; disable for a fixed growth schedule",
    )
    p.add_argument("--growth-waves", type=int, default=5)
    p.add_argument(
        "--relocate-fraction",
        type=float,
        default=0.0,
        help="StructSplat relocation ablation; matched evidence favors 0",
    )
    p.add_argument(
        "--structsplat-renderer",
        choices=["auto", "normalized", "cuda", "cuda_tiled"],
        default="auto",
    )


if __name__ == "__main__":
    raise SystemExit(main())
