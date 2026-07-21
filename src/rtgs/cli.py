"""rtgs command-line interface (argparse; see docs/ARCHITECTURE.md §CLI).

Commands: fit-images, lift, lift-field, refine, run, render, view, bench.
`--scene synthetic[:key=val,..]` builds a procedural test scene; calibrated frame directories
and COLMAP datasets are detected.
"""

from __future__ import annotations

import argparse
import json
import math
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
        native_renderer=args.native_renderer,
    )
    if args.save_observation_teachers and cfg.backend != "structsplat":
        print("--save-observation-teachers requires --fit-backend structsplat", file=sys.stderr)
        return 2
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
        observations = []
        g, hist = fit_image(
            img.to(device),
            cfg,
            seed=args.seed + i,
            observation_callback=(observations.append if args.save_observation_teachers else None),
            observation_view_id=(p.stem if args.save_observation_teachers else None),
        )
        g.save_npz(out_dir / f"{p.stem}.npz")
        if observations:
            observations[0].save_npz(out_dir / f"{p.stem}.teacher.npz")
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


def _field_split(n_views: int, heldout_stride: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return an explicit deterministic train/held-out partition for compact fields."""

    if heldout_stride < 2:
        raise ValueError("heldout stride must be at least 2")
    heldout = () if n_views < 3 else tuple(range(0, n_views, heldout_stride))
    train = tuple(index for index in range(n_views) if index not in set(heldout))
    if not train:
        raise ValueError("heldout split must leave at least one training view")
    return train, heldout


def _field_lift_config(payload: str):
    """Parse field-lift JSON, including its nested continuous-refit controls."""

    from rtgs.lift.field_lifter import FieldLiftConfig
    from rtgs.lift.field_refit import FieldRefitConfig

    values = json.loads(payload)
    if not isinstance(values, dict):
        raise ValueError("--field-args must decode to a JSON object")
    values = dict(values)
    refit_values = values.pop("refit", None)
    if refit_values is not None:
        if not isinstance(refit_values, dict):
            raise ValueError("--field-args refit must be a JSON object")
        values["refit"] = FieldRefitConfig(**refit_values)
    return FieldLiftConfig(**values)


def _json_safe(value):
    """Convert diagnostics to strict JSON without non-finite numeric extensions."""

    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save_field_state(result, out: Path) -> Path:
    """Persist field semantics that are not representable in a standard 3DGS PLY/NPZ."""

    import numpy as np

    fiber = result.refit.fiber
    payload = {
        "field_masses": result.refit.field_masses.detach().cpu().numpy(),
        "render_opacity": result.refit.render_opacity.detach().cpu().numpy(),
        "source_global_view_indices": (
            result.placement.source_global_view_indices.detach().cpu().numpy()
        ),
        "source_local_view_indices": fiber.source_view_indices.detach().cpu().numpy(),
        "source_component_indices": fiber.source_component_indices.detach().cpu().numpy(),
        "source_means2d": fiber.source_means2d.detach().cpu().numpy(),
        "source_covariances2d": fiber.source_covariances2d.detach().cpu().numpy(),
        "depths": fiber.depths().detach().cpu().numpy(),
        "cross": fiber.cross.detach().cpu().numpy(),
        "log_ray_scale": fiber.log_ray_scale.detach().cpu().numpy(),
        "fitting_visibility": result.refit.visibility.weights.detach().cpu().numpy(),
        "correspondence_visibility": (result.correspondence_visibility.detach().cpu().numpy()),
        "gains": result.refit.gains.detach().cpu().numpy(),
        "covariance_free_mask": result.refit.covariance_free_mask.detach().cpu().numpy(),
        "optimized_view_indices": np.asarray(result.optimized_view_indices, dtype=np.int64),
        "heldout_view_indices": np.asarray(result.heldout_view_indices, dtype=np.int64),
    }
    for view, correspondence in enumerate(result.correspondences):
        payload[f"correspondence_{view:04d}"] = correspondence.detach().cpu().numpy()
    state_path = out.with_suffix(".field.npz")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(state_path, **payload)
    return state_path


def _cmd_lift_field(args: argparse.Namespace) -> int:
    """Lift a compact dataset without loading any reference image or mask file."""

    from rtgs.data.compact_views import CompactDataset
    from rtgs.data.field_inputs import SceneFits
    from rtgs.pipeline import run_field_pipeline

    dataset = CompactDataset.load(Path(args.dataset).expanduser(), device="cpu")
    train, heldout = _field_split(dataset.n_views, args.heldout_stride)
    fits = SceneFits.from_compact_dataset(
        dataset,
        train_view_indices=train,
        heldout_view_indices=heldout,
    )
    result = run_field_pipeline(fits, _field_lift_config(args.field_args))
    out = Path(args.out).expanduser()
    _save_gaussians(result.gaussians, out)
    state_path = _save_field_state(result, out)
    from dataclasses import asdict

    diagnostics = {
        "dataset": dataset.name,
        "train_view_indices": list(train),
        "heldout_view_indices": list(heldout),
        "field_state": str(state_path),
        "diagnostics": result.diagnostics,
        "semantic_validation": {
            "train": asdict(result.semantic_validation.train),
            "heldout": (
                None
                if result.semantic_validation.heldout is None
                else asdict(result.semantic_validation.heldout)
            ),
            "per_view": [asdict(metrics) for metrics in result.semantic_validation.per_view],
        },
    }
    diagnostics_path = out.with_suffix(".diagnostics.json")
    diagnostics_path.write_text(json.dumps(_json_safe(diagnostics), indent=2, allow_nan=False))
    print(f"field-lift state -> {state_path}")
    print(f"field-lift diagnostics -> {diagnostics_path}")
    return 0


def _cmd_refine(args: argparse.Namespace) -> int:
    from rtgs.optim.trainer import TrainConfig, Trainer

    scene = _load_scene(args.scene, downscale=args.downscale, max_images=args.max_images)
    init = _load_gaussians(Path(args.init))
    cfg = _train_config(args, args.iterations, TrainConfig)
    refined, history = Trainer(cfg).train(scene, init)
    metrics = _split_metrics(scene, refined, cfg)
    print(json.dumps({"metrics": metrics, "training": _history_summary(history)}, indent=2))
    out = Path(args.out)
    _save_gaussians(refined, out)
    out.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    _save_training_config(out.with_suffix(".config.json"), cfg)
    if args.preview:
        from rtgs.visualize import save_reconstruction_artifacts

        init_path = out.with_name("gaussians_init.ply")
        if not init_path.exists():
            init.save_ply(init_path)
        save_reconstruction_artifacts(
            scene,
            init,
            refined,
            out.parent,
            rasterizer=args.rasterizer,
            packed=args.packed,
            antialiased=args.antialiased,
        )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from rtgs.image2gs.fit import FitConfig
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
            native_renderer=args.native_renderer,
            batch_views=args.batch_views,
        ),
        lifter=args.lifter,
        lifter_kwargs=json.loads(args.lifter_args),
        train=_train_config(args, args.refine_iters, TrainConfig),
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
        (out / "training_history.json").write_text(json.dumps(result.train_history, indent=2))
        _save_training_config(out / "gaussians.config.json", cfg.train)
        if args.preview:
            from rtgs.visualize import save_reconstruction_artifacts

            artifacts = save_reconstruction_artifacts(
                scene,
                result.gaussians_init,
                result.gaussians,
                out,
                rasterizer=args.rasterizer,
                packed=args.packed,
                antialiased=args.antialiased,
            )
            print(f"visual reconstruction -> {artifacts['contact_sheet']}")
            print(f"calibrated camera path -> {artifacts['turntable']}")
            print(f"novel orbit -> {artifacts['novel_orbit']}")
            print(f"novel elevation path -> {artifacts['novel_elevation']}")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    import numpy as np
    from PIL import Image as PILImage

    from rtgs.render.base import get_rasterizer

    scene = _load_scene(args.scene, downscale=args.downscale, max_images=args.max_images)
    device = _resolve_device(args.device)
    scene = scene.to(device)
    gaussians_path = Path(args.gaussians).expanduser()
    g = _load_gaussians(gaussians_path).to(device)
    saved_packed, saved_antialiased = _saved_render_options(gaussians_path)
    packed = saved_packed if args.packed is None else args.packed
    antialiased = saved_antialiased if args.antialiased is None else args.antialiased
    renderer = get_rasterizer(
        args.rasterizer,
        device=g.means.device,
        packed=packed,
        antialiased=antialiased,
    )
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


def _cmd_view(args: argparse.Namespace) -> int:
    from rtgs.viewer import launch_viewer

    final_path = Path(args.gaussians).expanduser()
    models = {"final": _load_gaussians(final_path)}
    initial_path = None if args.initial is None else Path(args.initial).expanduser()
    if initial_path is None and final_path.name == "gaussians.ply":
        candidate = final_path.with_name("gaussians_init.ply")
        if candidate.is_file():
            initial_path = candidate
    if initial_path is not None:
        models["initial"] = _load_gaussians(initial_path)

    scene = None
    if args.scene is not None:
        scene = _load_scene(args.scene, downscale=args.downscale, max_images=args.max_images)
    snapshot_dir = None if args.snapshot_dir is None else Path(args.snapshot_dir).expanduser()
    saved_packed, saved_antialiased = _saved_render_options(final_path)
    packed = saved_packed if args.packed is None else args.packed
    antialiased = saved_antialiased if args.antialiased is None else args.antialiased
    launch_viewer(
        models,
        scene=scene,
        device=_resolve_device(args.device),
        snapshot_rasterizer=args.rasterizer,
        snapshot_packed=packed,
        snapshot_antialiased=antialiased,
        snapshot_dir=snapshot_dir,
        host=args.host,
        port=args.port,
        max_viewer_gaussians=args.max_viewer_gaussians,
        open_browser=args.open,
    )
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

    absgrad = args.density_absgrad
    if absgrad is None:
        absgrad = args.density_strategy == "gsplat-default"
    grad_threshold = args.density_grad_threshold
    if grad_threshold is None:
        grad_threshold = 8e-4 if absgrad else 2e-4
    return DensityConfig(
        start_iter=args.densify_start,
        stop_iter=args.densify_stop,
        every=args.densify_every,
        grad_threshold=grad_threshold,
        absgrad=absgrad,
        prune_opacity=args.prune_opacity,
        prune_scale_frac=args.prune_scale_frac,
        max_gaussians=args.max_3d_gaussians,
        opacity_reset_every=args.opacity_reset_every,
        revised_opacity=args.revised_opacity,
        mcmc_noise_lr=args.mcmc_noise_lr,
    )


def _train_config(args: argparse.Namespace, iterations: int, cls=None):
    if cls is None:
        from rtgs.optim.trainer import TrainConfig

        cls = TrainConfig
    return cls(
        iterations=iterations,
        rasterizer=args.rasterizer,
        device=args.device,
        densify=args.densify,
        density_strategy=args.density_strategy,
        density=_density_config(args),
        target_sh_degree=args.target_sh_degree,
        sh_degree_interval=args.sh_degree_interval,
        random_background=args.random_background,
        mask_alpha_lambda=args.mask_alpha_lambda,
        opacity_reg=args.opacity_reg,
        scale_reg=args.scale_reg,
        packed=args.packed,
        antialiased=args.antialiased,
        eval_every=args.eval_every,
    )


def _split_metrics(scene, gaussians, config) -> dict:
    from rtgs.optim.trainer import Trainer
    from rtgs.render.base import get_rasterizer

    renderer = get_rasterizer(
        config.rasterizer,
        device=gaussians.means.device,
        packed=config.packed,
        antialiased=config.antialiased,
        sh_color_activation=config.sh_color_activation,
        sh_smu1_mu=config.sh_smu1_mu,
        kernel_support_mode=config.kernel_support_mode,
        visibility_margin_sigma=config.visibility_margin_sigma,
    )

    if not scene.testing_views:
        return {
            "train": Trainer.evaluate_metrics(
                scene, gaussians, renderer, indices=scene.training_views
            )
        }
    return {
        "test": Trainer.evaluate_metrics(scene, gaussians, renderer, indices=scene.testing_views),
        "train": Trainer.evaluate_metrics(scene, gaussians, renderer, indices=scene.training_views),
    }


def _history_summary(history: dict) -> dict:
    return {
        "density_strategy": history["density_strategy"],
        "resolved_sh_degree_interval": history["resolved_sh_degree_interval"],
        "peak_vram_gb": history["peak_vram_gb"],
        "final_n_gaussians": history["n_gaussians"][-1][1] if history["n_gaussians"] else None,
    }


def _save_training_config(path: Path, config) -> None:
    from dataclasses import asdict

    path.write_text(json.dumps({"training": asdict(config)}, indent=2))


def _saved_render_options(gaussians_path: Path) -> tuple[bool, bool]:
    """Load packed/antialiased flags saved beside a reconstruction, if present."""
    path = gaussians_path.expanduser()
    candidates = [path.with_suffix(".config.json"), path.parent / "training_config.json"]
    config_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if config_path is None:
        return False, False
    try:
        payload = json.loads(config_path.read_text())
        training = payload.get("training", payload)
        return bool(training.get("packed", False)), bool(training.get("antialiased", False))
    except (OSError, TypeError, ValueError):
        return False, False


def _add_density_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--densify", action=argparse.BooleanOptionalAction, default=True, help="3D density control"
    )
    p.add_argument("--densify-start", type=int, default=60)
    p.add_argument("--densify-stop", type=int, default=10_000_000)
    p.add_argument("--densify-every", type=int, default=40)
    p.add_argument("--max-3d-gaussians", type=int, default=100_000)
    p.add_argument(
        "--density-strategy",
        choices=["classic", "gsplat-default", "gsplat-mcmc"],
        default="classic",
    )
    p.add_argument("--density-grad-threshold", type=float, default=None)
    p.add_argument("--density-absgrad", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--prune-opacity", type=float, default=0.005)
    p.add_argument("--prune-scale-frac", type=float, default=0.1)
    p.add_argument("--opacity-reset-every", type=int, default=3_000)
    p.add_argument("--revised-opacity", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mcmc-noise-lr", type=float, default=500_000.0)


def _add_training_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--target-sh-degree", type=int, choices=range(4), default=3)
    p.add_argument(
        "--sh-degree-interval",
        type=int,
        default=None,
        help="steps between SH bands; default adapts for short runs and caps at 1000",
    )
    p.add_argument(
        "--random-background",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="with masks, randomize the composited background to discourage floaters",
    )
    p.add_argument("--mask-alpha-lambda", type=float, default=0.05)
    p.add_argument("--opacity-reg", type=float, default=None)
    p.add_argument("--scale-reg", type=float, default=None)
    p.add_argument("--packed", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--antialiased", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--eval-every",
        type=int,
        default=250,
        help="evaluate held-out views every N steps (final evaluation always runs)",
    )


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
    p.add_argument(
        "--save-observation-teachers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also save lossless RGB-free .teacher.npz fields (StructSplat only)",
    )
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

    p = sub.add_parser(
        "lift-field",
        help="stage 2: lift a compact Gaussian field dataset without source images",
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="compact frame directory or its gaussians2d directory",
    )
    p.add_argument(
        "--heldout-stride",
        type=int,
        default=8,
        help="hold out views 0, N, 2N, ... when at least three views exist (N >= 2)",
    )
    p.add_argument(
        "--field-args",
        default="{}",
        help="JSON FieldLiftConfig controls; nested refit config is supported",
    )
    p.add_argument("--out", required=True, help="output .npz or .ply")
    p.set_defaults(func=_cmd_lift_field)

    p = sub.add_parser("refine", help="stage 3: 3DGS optimization from an initialization")
    _add_scene_args(p)
    p.add_argument("--init", required=True, help="input .npz or .ply")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--rasterizer", default="auto")
    _add_density_args(p)
    _add_training_args(p)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save calibrated and novel-view visual diagnostics beside the output",
    )
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
    p.add_argument(
        "--batch-views",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="fit all training views jointly in one fused stage-1 optimization (native only)",
    )
    p.add_argument("--refine-iters", type=int, default=200)
    p.add_argument("--rasterizer", default="auto")
    _add_density_args(p)
    _add_training_args(p)
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
    p.add_argument("--packed", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--antialiased", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_render)

    p = sub.add_parser("view", help="open an interactive browser viewer for saved gaussians")
    p.add_argument("--gaussians", required=True, help="final reconstruction .npz or .ply")
    p.add_argument(
        "--initial",
        default=None,
        help="optional initialization .npz or .ply (auto-detected beside gaussians.ply)",
    )
    p.add_argument(
        "--scene",
        default=None,
        help="optional scene for calibrated cameras, references, and exact snapshots",
    )
    p.add_argument("--downscale", type=int, default=1)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument(
        "--max-viewer-gaussians",
        type=int,
        default=None,
        help="optional WebGL transfer/display cap (does not alter saved reconstruction)",
    )
    p.add_argument(
        "--rasterizer",
        choices=["auto", "torch", "gsplat"],
        default="auto",
        help="backend for exact calibrated-camera snapshots",
    )
    p.add_argument(
        "--antialiased",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="match antialiased training (default: auto-detect saved config)",
    )
    p.add_argument(
        "--packed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="match packed rendering (default: auto-detect saved config)",
    )
    p.add_argument("--device", default="auto", help="snapshot device: auto, cpu, cuda, cuda:N")
    p.add_argument(
        "--snapshot-dir", default=None, help="optionally save exact snapshots as PNG files"
    )
    p.add_argument("--host", default="127.0.0.1", help="viewer bind address")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="open the viewer URL in a local browser",
    )
    p.set_defaults(func=_cmd_view)

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
    p.add_argument(
        "--native-renderer",
        choices=["torch", "cuda", "auto"],
        default="torch",
        help="native-backend renderer; 'cuda' is the experimental stage-1 extension",
    )


if __name__ == "__main__":
    raise SystemExit(main())
