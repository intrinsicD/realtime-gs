#!/usr/bin/env python3
"""Cover-consistent surfel initialization as a Beam Fusion covariance/opacity treatment.

Beam Fusion's covariance-intersection rule estimates *where a component is*; the renderer needs
*how much surface a primitive covers*.  This harness screens five arms that share bit-identical
means, SH/color, and count and differ only in quaternions, log-scales, and opacity, isolating
optical thickness from spatial extent from surface orientation.  It runs on the fresh
``frame_00009`` root with three cameras that enter neither Beam Fusion nor refinement.

Frozen protocol: ``benchmarks/results/20260724_beam_surfel_init_PREREG.md``.

Development-scope deviations, stated so no result is over-read: 8 of 26 views, downscale-32
point-sampled compact teachers, 800 initial components, the pure-Torch CPU reference rasterizer
with the classic density controller instead of CUDA gsplat, and a single scene and seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.compact_views import CompactDataset
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.data.scene import SceneData
from rtgs.lift.beam_fusion import BeamFusionConfig, fuse_gaussian_beams
from rtgs.lift.surfel_init import (
    SurfelInitConfig,
    beam_contributor_footprints,
    coverage_opacity,
    effective_cover_sigma,
    estimate_local_surface_frames,
    reconcile_covariances,
)
from rtgs.optim.density import DensityConfig, DensityController
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.base import get_rasterizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAME = "frame_00009"
DEFAULT_OUT = ROOT / "runs/beam_surfel_init_20260724"
DEFAULT_PROTOCOL = ROOT / "benchmarks/results/20260724_beam_surfel_init_PREREG.md"

TRAIN_VIEWS = (0, 3, 6, 9, 12, 15, 18, 21)
HELDOUT_VIEWS = (1, 13, 25)
EXTRAPOLATIVE_HELDOUT = 25
DOWNSCALE = 32
N_INIT = 800
ITERATIONS = 1_000
EVAL_EVERY = 25
SEED = 0
# Frozen in the 20260723 convergence-dynamics screen for this rasterizer and resolution.
GRAD_THRESHOLD = 3e-3
ARMS = ("ci", "ci-op", "cover-iso", "cover-iso-op", "surfel")
MODES = ("fixed", "adc")


def _display(path: Path) -> str:
    """Repository-relative path when possible, absolute otherwise."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def density_config() -> DensityConfig:
    return DensityConfig(
        start_iter=20,
        stop_iter=500,
        every=4,
        grad_threshold=GRAD_THRESHOLD,
        absgrad=False,
        prune_opacity=0.005,
        prune_scale_frac=0.1,
        max_gaussians=8_000,
        opacity_reset_every=100,
        opacity_reset_value=0.011,
    )


def train_config(densify: bool, iterations: int) -> TrainConfig:
    return TrainConfig(
        iterations=iterations,
        rasterizer="torch",
        device="cpu",
        densify=densify,
        density_strategy="classic",
        density=density_config(),
        eval_every=EVAL_EVERY,
        target_sh_degree=3,
        sh_degree_interval=33,
        use_masks=True,
        random_background=False,
        seed=SEED,
    )


# --------------------------------------------------------------------------------------- data


def build_scene(dataset: CompactDataset) -> tuple[SceneData, list[int], list[int], int]:
    """Compact-teacher scene over train + held-out views; only train views are optimized."""
    order = list(TRAIN_VIEWS) + list(HELDOUT_VIEWS)
    images, masks, cameras, names = [], [], [], []
    for global_index in order:
        view = dataset.views[global_index]
        observation = view.observation
        fit_x, fit_y, width, height = observation.fit_window
        ds_width, ds_height = width // DOWNSCALE, height // DOWNSCALE
        ys = (torch.arange(ds_height, dtype=torch.float32) + 0.5) * DOWNSCALE + fit_y
        xs = (torch.arange(ds_width, dtype=torch.float32) + 0.5) * DOWNSCALE + fit_x
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        xy = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        colors = []
        for start in range(0, xy.shape[0], 8192):
            colors.append(observation.query(xy[start : start + 8192], component_chunk=2048).color)
        image = torch.cat(colors).reshape(ds_height, ds_width, 3).clamp(0, 1)
        if view.alpha is None:
            raise RuntimeError(f"screen requires packed alpha for {view.view_id}")
        alpha = view.alpha.crop_mask("cpu").float()
        alpha = (
            alpha[: ds_height * DOWNSCALE, : ds_width * DOWNSCALE]
            .reshape(ds_height, DOWNSCALE, ds_width, DOWNSCALE)
            .mean(dim=(1, 3))
        )
        images.append((image * alpha[..., None]).float())
        masks.append(alpha)
        camera = view.camera
        cameras.append(
            Camera(
                fx=camera.fx / DOWNSCALE,
                fy=camera.fy / DOWNSCALE,
                cx=(camera.cx - fit_x) / DOWNSCALE,
                cy=(camera.cy - fit_y) / DOWNSCALE,
                width=ds_width,
                height=ds_height,
                R=camera.R,
                t=camera.t,
            )
        )
        names.append(view.view_id)
        print(f"[teacher] {view.view_id} {ds_width}x{ds_height}", flush=True)
    train_local = list(range(len(TRAIN_VIEWS)))
    heldout_local = list(range(len(TRAIN_VIEWS), len(order)))
    extrapolative_local = len(TRAIN_VIEWS) + list(HELDOUT_VIEWS).index(EXTRAPOLATIVE_HELDOUT)
    scene = SceneData(
        images=images,
        cameras=cameras,
        view_names=names,
        masks=masks,
        train_indices=train_local,
        test_indices=heldout_local,
        bounds_hint=dataset.bounds_hint,
        name=f"{dataset.name}-surfel-ds{DOWNSCALE}",
    )
    scene.validate()
    return scene, train_local, heldout_local, extrapolative_local


def train_inputs(dataset: CompactDataset) -> ReconstructionInputs:
    complete = dataset.to_reconstruction_inputs()
    return ReconstructionInputs(
        observations=[complete.observations[i] for i in TRAIN_VIEWS],
        cameras=[complete.cameras[i] for i in TRAIN_VIEWS],
        view_names=[complete.view_names[i] for i in TRAIN_VIEWS],
        bounds_hint=complete.bounds_hint,
        name=f"{complete.name}-surfel-train",
    )


# ------------------------------------------------------------------------------- initializations


def _replace_opacity(gaussians: Gaussians3D, opacity: torch.Tensor) -> Gaussians3D:
    return Gaussians3D(
        means=gaussians.means.detach().clone(),
        quats=gaussians.quats.detach().clone(),
        log_scales=gaussians.log_scales.detach().clone(),
        opacity=opacity.to(gaussians.opacity.dtype),
        sh=gaussians.sh.detach().clone(),
    )


def build_initializations(dataset: CompactDataset) -> tuple[dict[str, Gaussians3D], dict]:
    """Beam Fusion once, then five covariance/opacity arms over identical means and colors."""
    inputs = train_inputs(dataset)
    extent = float(dataset.bounds_hint[1])
    beam_config = BeamFusionConfig(
        min_views=3,
        transverse_gate_sigma=3.0,
        max_color_distance=0.35,
        color_sigma=0.25,
        fold_in_gate_sigma=3.0,
        nms_voxel_size=extent / 100.0,
        init_opacity=0.10,
        source_chunk=256,
        max_components=N_INIT,
        seed_budget_multiplier=4,
    )
    started = time.perf_counter()
    beam = fuse_gaussian_beams(inputs, beam_config)
    beam_seconds = time.perf_counter() - started
    ci = beam.gaussians.detach()

    surfel_config = SurfelInitConfig()
    floor = beam_contributor_footprints(inputs, beam)
    frames = estimate_local_surface_frames(ci.means, surfel_config)

    surfel = reconcile_covariances(ci, surfel_config, resolution_floor=floor, frames=frames)
    isotropic_op = reconcile_covariances(
        ci,
        SurfelInitConfig(isotropic=True),
        resolution_floor=floor,
        frames=frames,
    )
    isotropic_fixed = reconcile_covariances(
        ci,
        SurfelInitConfig(isotropic=True, use_coverage_opacity=False, fixed_opacity=0.10),
        resolution_floor=floor,
        frames=frames,
    )
    ci_sigma = effective_cover_sigma(ci)
    ci_opacity, ci_overlap = coverage_opacity(ci_sigma, frames.spacing, surfel_config)

    inits = {
        "ci": ci,
        "ci-op": _replace_opacity(ci, ci_opacity),
        "cover-iso": isotropic_fixed.gaussians,
        "cover-iso-op": isotropic_op.gaussians,
        "surfel": surfel.gaussians,
    }
    for name, arm in inits.items():
        if arm.n != ci.n:
            raise RuntimeError(f"{name} changed the component count")
        if not torch.equal(arm.means, ci.means):
            raise RuntimeError(f"{name} changed a frozen mean")
        if not torch.equal(arm.sh, ci.sh):
            raise RuntimeError(f"{name} changed a frozen SH coefficient")

    def summarize(values: torch.Tensor) -> dict[str, float]:
        values = values.detach().to(torch.float64)
        return {
            "p10": float(values.quantile(0.10)),
            "median": float(values.median()),
            "p90": float(values.quantile(0.90)),
            "mean": float(values.mean()),
        }

    views_per_component = (beam.component_offsets[1:] - beam.component_offsets[:-1]).double()
    diagnostics = {
        "beam": {
            "elapsed_seconds": beam_seconds,
            "n_gaussians": ci.n,
            "mean_contributing_views": float(views_per_component.mean()),
            "n_contributor_links": int(beam.contributor_view_indices.numel()),
            "unmatched_per_view": list(beam.unmatched_per_view),
            "config": {
                "min_views": beam_config.min_views,
                "transverse_gate_sigma": beam_config.transverse_gate_sigma,
                "max_color_distance": beam_config.max_color_distance,
                "color_sigma": beam_config.color_sigma,
                "fold_in_gate_sigma": beam_config.fold_in_gate_sigma,
                "nms_voxel_size": beam_config.nms_voxel_size,
                "init_opacity": beam_config.init_opacity,
                "source_chunk": beam_config.source_chunk,
                "max_components": beam_config.max_components,
                "seed_budget_multiplier": beam_config.seed_budget_multiplier,
            },
        },
        "surfel_rule": surfel.diagnostics,
        "resolution_floor_sigma": summarize(floor),
        "ci_effective_cover_sigma": summarize(ci_sigma),
        "ci_overlap_kernel_units": summarize(ci_overlap),
        "ci_coverage_opacity": summarize(ci_opacity),
        "arm_scale_summary": {
            name: {
                "sigma_min": summarize(arm.scales.amin(dim=-1)),
                "sigma_max": summarize(arm.scales.amax(dim=-1)),
                "opacity": summarize(arm.opacity),
            }
            for name, arm in inits.items()
        },
        "frozen_field_assertions": {
            name: {
                "same_count": arm.n == ci.n,
                "means_bit_exact": bool(torch.equal(arm.means, ci.means)),
                "sh_bit_exact": bool(torch.equal(arm.sh, ci.sh)),
            }
            for name, arm in inits.items()
        },
    }
    return inits, diagnostics


# ------------------------------------------------------------------------------------ training


class ParticipationController(DensityController):
    """Classic controller that records how far the *original* rows get into densification.

    Participation is measured by the controller's own selection criterion — the accumulated
    screen-space positional gradient against ``grad_threshold`` — evaluated on surviving
    original rows immediately before each scheduled round.  That is the quantity that decides
    clone/split, and it is read from public controller state rather than reconstructed from
    ``density.py``'s internal masks, so it cannot silently drift from the implementation.  The
    realized ``cloned``/``split``/``pruned`` totals are reported alongside from ``self.stats``.
    """

    instances: list[ParticipationController] = []

    def __init__(self, config, n_gaussians, scene_extent, device="cpu"):
        super().__init__(config, n_gaussians, scene_extent, device=device)
        self.identity = torch.arange(n_gaussians)
        self.qualified_originals: set[int] = set()
        self.first_event_gradient: dict[str, float] | None = None
        self.events: list[dict] = []
        ParticipationController.instances.append(self)

    def step(self, iteration, params, optimizer, generator=None, force_budget=False):
        import rtgs.optim.density as density_module

        cfg = self.cfg
        scheduled = cfg.start_iter <= iteration <= cfg.stop_iter and iteration % cfg.every == 0
        n_before = int(params["means"].shape[0])
        gradient_record: dict[str, float] | None = None
        if scheduled:
            avg_grad = self.grad_accum / self.count.clamp_min(1.0)
            original_rows = (self.identity >= 0) & (self.count > 0)
            if bool(original_rows.any()):
                original_grad = avg_grad[original_rows]
                above = original_grad > cfg.grad_threshold
                self.qualified_originals.update(
                    int(row) for row in self.identity[original_rows][above].tolist()
                )
                gradient_record = {
                    "n_original_rows": int(original_rows.sum()),
                    "median": float(original_grad.median()),
                    "p90": float(original_grad.quantile(0.9)),
                    "frac_above_threshold": float(above.double().mean()),
                }
                if self.first_event_gradient is None:
                    self.first_event_gradient = {"iteration": iteration, **gradient_record}

        captured: list[tuple[torch.Tensor, int]] = []
        original_edit = density_module._edit_params

        def spy(opt, params_, keep_mask, extras):
            captured.append((keep_mask.detach().clone(), sum(e["means"].shape[0] for e in extras)))
            return original_edit(opt, params_, keep_mask, extras)

        density_module._edit_params = spy
        try:
            new_params = super().step(
                iteration, params, optimizer, generator=generator, force_budget=force_budget
            )
        finally:
            density_module._edit_params = original_edit

        stats = self.stats[-1] if self.stats and self.stats[-1]["iteration"] == iteration else {}
        for keep_mask, n_extras in captured:
            self.identity = torch.cat(
                [self.identity[keep_mask], torch.full((n_extras,), -1, dtype=torch.long)]
            )
            self.events.append(
                {
                    "iteration": iteration,
                    "n_before": n_before,
                    "n_after": int(self.identity.shape[0]),
                    "originals_alive": int((self.identity >= 0).sum()),
                    "original_gradient": gradient_record,
                    **{key: stats[key] for key in ("cloned", "split", "pruned") if key in stats},
                }
            )
        return new_params


def run_arm(
    name: str,
    mode: str,
    init: Gaussians3D,
    scene: SceneData,
    train_local: list[int],
    heldout_local: list[int],
    extrapolative_local: int,
    out: Path,
    iterations: int,
) -> dict:
    import rtgs.optim.trainer as trainer_module

    arm_dir = out / mode / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    renderer = get_rasterizer("torch", device=torch.device("cpu"))
    densify = mode == "adc"

    def metrics(model: Gaussians3D) -> dict:
        return {
            "train": Trainer.evaluate_metrics(scene, model, renderer, train_local),
            "heldout": Trainer.evaluate_metrics(scene, model, renderer, heldout_local),
            "heldout_extrapolative": Trainer.evaluate_metrics(
                scene, model, renderer, [extrapolative_local]
            ),
        }

    init_metrics = metrics(init)
    init.save_ply(arm_dir / "gaussians_init.ply")
    curve: list[dict] = []
    started = time.perf_counter()

    def checkpoint(snapshot: Gaussians3D, step: int) -> None:
        values = metrics(snapshot)
        controller = ParticipationController.instances[-1] if densify else None
        row = {
            "step": step,
            "n": snapshot.n,
            "originals_alive": (
                int((controller.identity >= 0).sum()) if controller is not None else snapshot.n
            ),
            "opacity_median": float(snapshot.opacity.median()),
            "sigma_max_median": float(snapshot.scales.amax(dim=-1).median()),
            **{f"train_{k}": v for k, v in values["train"].items()},
            **{f"heldout_{k}": v for k, v in values["heldout"].items()},
        }
        curve.append(row)
        print(
            f"[{mode}/{name}] step {step} n={snapshot.n} "
            f"train_fg={values['train'].get('psnr_fg', float('nan')):.2f}dB "
            f"held_fg={values['heldout'].get('psnr_fg', float('nan')):.2f}dB "
            f"held_aIoU={values['heldout'].get('alpha_iou', float('nan')):.3f} "
            f"({time.perf_counter() - started:.0f}s)",
            flush=True,
        )

    previous = trainer_module.DensityController
    trainer_module.DensityController = ParticipationController
    ParticipationController.instances.clear()
    try:
        final, history = Trainer(train_config(densify, iterations)).train(
            scene, init, checkpoint_callback=checkpoint
        )
    finally:
        trainer_module.DensityController = previous
    final.save_ply(arm_dir / "gaussians_final.ply")

    controller = ParticipationController.instances[-1] if densify else None
    participation: dict = {}
    if controller is not None:
        alive = int((controller.identity >= 0).sum())
        participation = {
            "n_initial": init.n,
            "n_initial_alive": alive,
            "n_initial_qualified": len(controller.qualified_originals),
            "frac_initial_qualified": len(controller.qualified_originals) / max(init.n, 1),
            "first_event_gradient": controller.first_event_gradient,
            "n_density_events": len(controller.events),
            "total_cloned": sum(event.get("cloned", 0) for event in controller.events),
            "total_split": sum(event.get("split", 0) for event in controller.events),
            "total_pruned": sum(event.get("pruned", 0) for event in controller.events),
            "events": controller.events,
        }

    psnr_curve = [row["heldout_psnr_fg"] for row in curve if "heldout_psnr_fg" in row]
    initial_psnr = init_metrics["heldout"].get("psnr_fg")
    auc_samples = ([initial_psnr] if initial_psnr is not None else []) + psnr_curve
    record = {
        "arm": name,
        "mode": mode,
        "elapsed_seconds": time.perf_counter() - started,
        "init_metrics": init_metrics,
        "final_metrics": metrics(final),
        "final_n": final.n,
        "heldout_psnr_auc": (sum(auc_samples) / len(auc_samples)) if auc_samples else None,
        "curve": curve,
        "participation": participation,
        "loss_first_20": history["loss"][:20],
        "loss_last_20": history["loss"][-20:],
    }
    (arm_dir / "record.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def save_preview(scene: SceneData, gaussians: Gaussians3D, path: Path, view: int) -> None:
    from PIL import Image

    renderer = get_rasterizer("torch", device=torch.device("cpu"))
    with torch.no_grad():
        out = renderer.render(gaussians, scene.cameras[view])
    row = torch.cat([scene.images[view], out.color.clamp(0, 1)], dim=1)
    Image.fromarray((row * 255).byte().numpy()).save(path)


# ------------------------------------------------------------------------------------- reports


def write_viewer_manifest(out: Path, arms: list[str], modes: list[str]) -> Path:
    """One synchronized comparison over every (mode, arm) initial/final pair.

    Paths are relative to the manifest's own directory (`benchmarks/results/`), which is where
    `rtgs view --comparison-manifest` resolves them from.
    """
    manifest = {
        "schema": "rtgs.viewer-comparison.v1",
        "methods": [
            {
                "name": f"{mode}:{name}",
                "initial": f"../../runs/{out.name}/{mode}/{name}/gaussians_init.ply",
                "final": f"../../runs/{out.name}/{mode}/{name}/gaussians_final.ply",
            }
            for mode in modes
            for name in arms
        ],
    }
    path = ROOT / "benchmarks/results/20260724_beam_surfel_init_VIEWER.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return path


def write_index(out: Path, summary: dict) -> Path:
    rows = []
    for mode, arms in summary["modes"].items():
        for name, arm in arms.items():
            init = arm["init_metrics"]["heldout"]
            final = arm["final_metrics"]["heldout"]
            rows.append(
                "<tr><td>{mode}</td><td>{name}</td><td>{ip:.4f}</td><td>{ii:.5f}</td>"
                "<td>{io:.5f}</td><td>{auc}</td><td>{fp:.4f}</td><td>{fi:.5f}</td>"
                "<td>{fo:.5f}</td><td>{n}</td>"
                '<td><a href="{mode}/{name}/preview_init.png">init</a> / '
                '<a href="{mode}/{name}/preview_final.png">final</a></td></tr>'.format(
                    mode=mode,
                    name=name,
                    ip=init.get("psnr_fg", float("nan")),
                    ii=init.get("alpha_iou", float("nan")),
                    io=init.get("alpha_outside", float("nan")),
                    auc=(
                        f"{arm['heldout_psnr_auc']:.4f}"
                        if arm.get("heldout_psnr_auc") is not None
                        else "-"
                    ),
                    fp=final.get("psnr_fg", float("nan")),
                    fi=final.get("alpha_iou", float("nan")),
                    fo=final.get("alpha_outside", float("nan")),
                    n=arm["final_n"],
                )
            )
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>Beam surfel initialization — {summary["frame"]}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 72rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.6rem; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
caption {{ text-align: left; font-weight: 600; padding-bottom: 0.5rem; }}
</style>
<h1>Cover-consistent surfel initialization</h1>
<p>Root <code>{summary["frame"]}</code>; {summary["n_init"]} initial Gaussians;
{summary["iterations"]} steps; seed {summary["seed"]}. All metrics below are on the
<strong>held-out</strong> cameras {summary["heldout_view_names"]}, which entered neither Beam
Fusion nor refinement. Protocol:
<a href="../../benchmarks/results/20260724_beam_surfel_init_PREREG.md">preregistration</a>.</p>
<table>
<caption>Held-out foreground PSNR (dB), alpha IoU, alpha outside the mask</caption>
<tr><th>mode</th><th>arm</th><th>init PSNR</th><th>init IoU</th><th>init outside</th>
<th>PSNR AUC</th><th>final PSNR</th><th>final IoU</th><th>final outside</th><th>final N</th>
<th>previews</th></tr>
{chr(10).join(rows)}
</table>
<p>This is one scene, one seed, and a CPU reference rasterizer at downscale
{summary["downscale"]}; no default change is authorized by it.</p>
"""
    path = out / "index.html"
    path.write_text(html)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--frame", default=DEFAULT_FRAME)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args()
    if not args.protocol.is_file():
        raise FileNotFoundError(f"frozen protocol missing: {args.protocol}")
    if args.iterations != ITERATIONS:
        print(
            f"[warning] iterations={args.iterations} is a smoke override; frozen value is "
            f"{ITERATIONS}",
            flush=True,
        )

    torch.manual_seed(SEED)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dataset_path = ROOT / f"dataset/2025_03_07_stage_with_fabric/{args.frame}/gaussians2d"
    dataset = CompactDataset.load(dataset_path, device="cpu")
    scene, train_local, heldout_local, extrapolative_local = build_scene(dataset)
    initializations, diagnostics = build_initializations(dataset)
    print(json.dumps(diagnostics["surfel_rule"], indent=2), flush=True)

    modes: dict[str, dict] = {}
    for mode in args.modes:
        modes[mode] = {}
        for name in args.arms:
            record = run_arm(
                name,
                mode,
                initializations[name],
                scene,
                train_local,
                heldout_local,
                extrapolative_local,
                out,
                args.iterations,
            )
            modes[mode][name] = record
            final = Gaussians3D.load_ply(out / mode / name / "gaussians_final.ply")
            save_preview(
                scene,
                initializations[name],
                out / mode / name / "preview_init.png",
                heldout_local[0],
            )
            save_preview(scene, final, out / mode / name / "preview_final.png", heldout_local[0])

    summary = {
        "schema": "rtgs.beam_surfel_init.v1",
        "status": (
            "complete"
            if set(args.arms) == set(ARMS) and set(args.modes) == set(MODES)
            else "partial"
        ),
        "protocol": {
            "path": str(args.protocol.resolve().relative_to(ROOT)),
            "sha256": _sha256(args.protocol),
        },
        "dataset": str(dataset_path.relative_to(ROOT)),
        "frame": args.frame,
        "train_view_indices": list(TRAIN_VIEWS),
        "heldout_view_indices": list(HELDOUT_VIEWS),
        "train_view_names": [scene.view_names[i] for i in train_local],
        "heldout_view_names": [scene.view_names[i] for i in heldout_local],
        "extrapolative_heldout": scene.view_names[extrapolative_local],
        "downscale": DOWNSCALE,
        "n_init": N_INIT,
        "iterations": args.iterations,
        "seed": SEED,
        "grad_threshold": GRAD_THRESHOLD,
        "diagnostics": diagnostics,
        "modes": {
            mode: {
                name: {
                    key: record[key]
                    for key in (
                        "init_metrics",
                        "final_metrics",
                        "final_n",
                        "heldout_psnr_auc",
                        "participation",
                        "elapsed_seconds",
                        "curve",
                    )
                }
                for name, record in arms.items()
            }
            for mode, arms in modes.items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    index = write_index(out, summary)
    viewer = write_viewer_manifest(out, args.arms, args.modes)
    print(
        json.dumps(
            {
                mode: {
                    name: {
                        "init_heldout_psnr_fg": arm["init_metrics"]["heldout"].get("psnr_fg"),
                        "init_heldout_alpha_iou": arm["init_metrics"]["heldout"].get("alpha_iou"),
                        "init_heldout_alpha_outside": arm["init_metrics"]["heldout"].get(
                            "alpha_outside"
                        ),
                        "heldout_psnr_auc": arm["heldout_psnr_auc"],
                        "final_heldout_psnr_fg": arm["final_metrics"]["heldout"].get("psnr_fg"),
                        "final_heldout_alpha_iou": arm["final_metrics"]["heldout"].get("alpha_iou"),
                        "final_n": arm["final_n"],
                        "frac_initial_qualified": arm["participation"].get(
                            "frac_initial_qualified"
                        ),
                    }
                    for name, arm in arms.items()
                }
                for mode, arms in summary["modes"].items()
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"[index] {_display(index)}", flush=True)
    print(f"[viewer] {_display(viewer)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
