#!/usr/bin/env python3
"""Convergence-dynamics probe: does 3DGS refinement destroy a beam-fusion initialization?

The 20260721 full-frame runs recorded only initialization and endpoint metrics, so they cannot
say *what happened in between*: whether ordinary density-controlled refinement preserved the
beam-fusion surface film, or first dismantled it and re-derived geometry from clones. This CPU
probe replays the production interaction at reduced scale on the checked-in real bundle and logs
the trajectory itself:

- per-checkpoint foreground PSNR / alpha coverage (does quality dip below the init?);
- identity-tracked survival of the initial Gaussians through clone/split/prune surgery;
- displacement of surviving originals from their initial positions;
- symmetric center distance between the live model and the frozen beam-fusion surface;
- opacity distribution around the scheduled opacity resets.

Arms: {beam-fusion, random} x {classic ADC, fixed topology}, identical loss and schedule.

Faithful-to-production choices: real `frame_00008` compact bundle; exact frozen-field teachers
(``GaussianObservationField.query``) with packed-alpha masking; the 20260721 beam configuration
(min 3 views, 3-sigma gates, 0.35/0.25 color, extent/100 NMS voxel, opacity 0.10); the harness
loss (masked L1 + 0.2 D-SSIM + 0.05 mask-alpha + 0.01 outside-alpha, black background); the
harness prune rules (0.005/0.1) and reset value 0.011; the whole 30k/500-15k/100/3k schedule
scaled by 1/30 to 1000/20-500/4/100.

Known deviations (development diagnostics only, no default-change evidence): 8 of 26 views;
downscale-32 point-sampled teachers (beam splats project sub-pixel here, harsher than native);
800 initial components; pure-torch reference rasterizer with the classic CPU controller instead
of CUDA gsplat DefaultStrategy; the absgrad threshold recalibrated for this resolution (see
``GRAD_THRESHOLD``); all selected views are fitted, four of them evaluated (no held-out views).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.compact_views import CompactDataset
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.data.scene import SceneData
from rtgs.lift.baselines import _isotropic
from rtgs.lift.beam_fusion import BeamFusionConfig, fuse_gaussian_beams
from rtgs.optim.density import DensityConfig, DensityController
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.base import get_rasterizer

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d"

SELECTED_VIEWS = (0, 3, 6, 9, 12, 15, 18, 21)
DOWNSCALE = 32
N_INIT = 800
ITERATIONS = 1_000
EVAL_EVERY = 25
EVAL_VIEWS = (0, 2, 4, 6)
SNAPSHOT_STEPS = (75, 100, 125, 200, 300, 400, 500, 750, 1_000)
SEED = 0
# 20260721 GPU schedule (30000 steps, densify 500-15000 every 100, reset every 3000) / 30.
# The torch reference rasterizer exposes plain screen-gradient norms (no absgrad), and gradient
# magnitudes shift with resolution, so the GPU threshold does not transfer literally; the value
# below was calibrated (frozen before the measured arms, identical for every arm) so that the
# densification window reproduces the GPU run's ~x8.8 topology growth without hitting the cap:
# 8e-4 cloned ~880/event (cap-bound within 60 steps), 4e-3 ~34/event, 8e-3 ~3/event; 3e-3 gives
# ~65 newborns/event over the first 22 events, the GPU run's regime of ~50/event.
GRAD_THRESHOLD = 3e-3


def density_config(grad_threshold: float) -> DensityConfig:
    return DensityConfig(
        start_iter=20,
        stop_iter=500,
        every=4,
        grad_threshold=grad_threshold,
        absgrad=False,
        prune_opacity=0.005,
        prune_scale_frac=0.1,
        max_gaussians=8_000,
        opacity_reset_every=100,
        opacity_reset_value=0.011,
    )


def _train_config(densify: bool, iterations: int, grad_threshold: float) -> TrainConfig:
    return TrainConfig(
        iterations=iterations,
        rasterizer="torch",
        device="cpu",
        densify=densify,
        density_strategy="classic",
        density=density_config(grad_threshold),
        eval_every=EVAL_EVERY,
        target_sh_degree=3,
        sh_degree_interval=33,  # 1000 * (1000/30000), so every band still trains
        use_masks=True,
        random_background=False,
        seed=SEED,
    )


def build_scene(dataset: CompactDataset) -> SceneData:
    """Exact frozen-field teachers point-sampled at downscaled pixel centers, alpha-masked."""
    images, masks, cameras, names = [], [], [], []
    for global_index in SELECTED_VIEWS:
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
            raise RuntimeError(f"probe requires packed alpha for {view.view_id}")
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
    scene = SceneData(
        images=images,
        cameras=cameras,
        view_names=names,
        masks=masks,
        train_indices=list(range(len(names))),
        test_indices=[],
        bounds_hint=dataset.bounds_hint,
        name=f"{dataset.name}-dynamics-ds{DOWNSCALE}",
    )
    scene.validate()
    return scene


def selected_inputs(dataset: CompactDataset) -> ReconstructionInputs:
    complete = dataset.to_reconstruction_inputs()
    return ReconstructionInputs(
        observations=[complete.observations[i] for i in SELECTED_VIEWS],
        cameras=[complete.cameras[i] for i in SELECTED_VIEWS],
        view_names=[complete.view_names[i] for i in SELECTED_VIEWS],
        bounds_hint=complete.bounds_hint,
        name=f"{complete.name}-dynamics",
    )


def place_beam(dataset: CompactDataset) -> tuple[Gaussians3D, dict]:
    config = BeamFusionConfig(
        min_views=3,
        transverse_gate_sigma=3.0,
        max_color_distance=0.35,
        color_sigma=0.25,
        fold_in_gate_sigma=3.0,
        nms_voxel_size=float(dataset.bounds_hint[1]) / 100.0,
        init_opacity=0.10,
        source_chunk=256,
        max_components=N_INIT,
        seed_budget_multiplier=4,
    )
    started = time.perf_counter()
    result = fuse_gaussian_beams(selected_inputs(dataset), config)
    views_per_component = (result.component_offsets[1:] - result.component_offsets[:-1]).float()
    receipt = {
        "initializer": "beam-fusion",
        "elapsed_seconds": time.perf_counter() - started,
        "n_gaussians": result.gaussians.n,
        "mean_contributing_views": float(views_per_component.mean()),
        "world_sigma_median": float(result.gaussians.log_scales.exp().median()),
    }
    return result.gaussians.detach(), receipt


def place_random(dataset: CompactDataset) -> tuple[Gaussians3D, dict]:
    center, extent = dataset.bounds_hint
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    directions = torch.randn(N_INIT, 3, generator=generator)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    radii = 0.5 * float(extent) * torch.rand(N_INIT, 1, generator=generator).pow(1.0 / 3.0)
    points = center + directions * radii
    scale = torch.full((N_INIT,), 0.5 * float(extent) / N_INIT ** (1.0 / 3.0))
    gaussians = _isotropic(points, scale, torch.full((N_INIT, 3), 0.5), 0.10)
    return gaussians.detach(), {
        "initializer": "random",
        "isotropic_scale": float(scale[0]),
        "n_gaussians": gaussians.n,
    }


class TrackingController(DensityController):
    """Classic controller that additionally tracks the fate of the initial rows."""

    instances: list[TrackingController] = []

    def __init__(self, config, n_gaussians, scene_extent, device="cpu"):
        super().__init__(config, n_gaussians, scene_extent, device=device)
        self.identity = torch.arange(n_gaussians)
        self.events: list[dict] = []
        TrackingController.instances.append(self)

    def step(self, iteration, params, optimizer, generator=None, force_budget=False):
        import rtgs.optim.density as density_module

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
        for keep_mask, n_extras in captured:
            originals_before = int((self.identity >= 0).sum())
            self.identity = torch.cat(
                [self.identity[keep_mask], torch.full((n_extras,), -1, dtype=torch.long)]
            )
            cfg = self.cfg
            scheduled = cfg.start_iter <= iteration <= cfg.stop_iter and iteration % cfg.every == 0
            event = {
                "iteration": iteration,
                "originals_before": originals_before,
                "originals_after": int((self.identity >= 0).sum()),
                "opacity_reset": bool(
                    scheduled
                    and cfg.opacity_reset_every
                    and iteration % cfg.opacity_reset_every == 0
                ),
            }
            if self.stats and self.stats[-1]["iteration"] == iteration:
                event.update(self.stats[-1])
            self.events.append(event)
        return new_params


def _chamfer(a: torch.Tensor, b: torch.Tensor) -> tuple[float | None, float | None]:
    """Mean and p90 nearest-neighbor distance from each row of ``a`` to ``b``."""
    if a.numel() == 0 or b.numel() == 0:
        # JSON has no NaN value.  An empty confident set is expected immediately after an
        # opacity reset, so record the distance as unavailable instead of emitting non-standard
        # JSON tokens that strict downstream parsers reject.
        return None, None
    nearest = torch.cdist(a, b).min(dim=1).values
    return float(nearest.mean()), float(nearest.quantile(0.9))


def run_arm(
    name: str,
    init: Gaussians3D,
    scene: SceneData,
    reference_surface: torch.Tensor,
    densify: bool,
    out: Path,
    iterations: int = ITERATIONS,
    grad_threshold: float = GRAD_THRESHOLD,
) -> dict:
    import rtgs.optim.trainer as trainer_module

    arm_dir = out / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    renderer = get_rasterizer("torch", device=torch.device("cpu"))
    eval_indices = list(EVAL_VIEWS)
    init_metrics = Trainer.evaluate_metrics(scene, init, renderer, eval_indices)
    init.save_ply(arm_dir / "gaussians_init.ply")
    init_means = init.means.detach().clone()
    curve: list[dict] = []
    started = time.perf_counter()

    def checkpoint(snapshot: Gaussians3D, step: int) -> None:
        controller = TrackingController.instances[-1] if densify else None
        identity = (
            controller.identity.clone() if controller is not None else torch.arange(snapshot.n)
        )
        if identity.shape[0] != snapshot.n:
            raise RuntimeError(f"identity desync at step {step}")
        survivor_rows = identity >= 0
        displacement = (
            (snapshot.means[survivor_rows] - init_means[identity[survivor_rows]])
            .norm(dim=-1)
            .float()
        )
        opacity = snapshot.opacity
        confident = snapshot.means[opacity > 0.02]
        cur_to_ref, cur_to_ref_p90 = _chamfer(confident, reference_surface)
        ref_to_cur, _ = _chamfer(reference_surface, confident)
        metrics = Trainer.evaluate_metrics(scene, snapshot, renderer, eval_indices)
        row = {
            "step": step,
            "n": snapshot.n,
            "survivors": int(survivor_rows.sum()),
            "displacement_mean": float(displacement.mean()) if displacement.numel() else None,
            "displacement_p90": (
                float(displacement.quantile(0.9)) if displacement.numel() else None
            ),
            "survivor_opacity_mean": (
                float(opacity[survivor_rows].mean()) if bool(survivor_rows.any()) else None
            ),
            "newborn_opacity_mean": (
                float(opacity[~survivor_rows].mean()) if bool((~survivor_rows).any()) else None
            ),
            "frac_opacity_gt_half": float((opacity > 0.5).float().mean()),
            "frac_opacity_lt_02": float((opacity < 0.02).float().mean()),
            "n_confident": int(confident.shape[0]),
            "chamfer_current_to_surface_mean": cur_to_ref,
            "chamfer_current_to_surface_p90": cur_to_ref_p90,
            "chamfer_surface_to_current_mean": ref_to_cur,
            **{f"metric_{key}": value for key, value in metrics.items()},
        }
        curve.append(row)
        if step in SNAPSHOT_STEPS:
            snapshot.save_ply(arm_dir / f"gaussians_{step:05d}.ply")
        print(
            f"[{name}] step {step} n={snapshot.n} surv={row['survivors']} "
            f"fg={metrics.get('psnr_fg', float('nan')):.2f}dB "
            f"aIoU={metrics.get('alpha_iou', float('nan')):.3f} "
            f"({time.perf_counter() - started:.0f}s)",
            flush=True,
        )

    previous_controller = trainer_module.DensityController
    trainer_module.DensityController = TrackingController
    TrackingController.instances.clear()
    try:
        final, history = Trainer(_train_config(densify, iterations, grad_threshold)).train(
            scene, init, checkpoint_callback=checkpoint
        )
    finally:
        trainer_module.DensityController = previous_controller
    final.save_ply(arm_dir / "gaussians_final.ply")
    controller = TrackingController.instances[-1] if densify else None
    record = {
        "arm": name,
        "densify": densify,
        "elapsed_seconds": time.perf_counter() - started,
        "init_metrics": init_metrics,
        "final_n": final.n,
        "curve": curve,
        "density_events": controller.events if controller is not None else [],
        "loss_first_20": history["loss"][:20],
        "loss_last_20": history["loss"][-20:],
    }
    (arm_dir / "dynamics.json").write_text(json.dumps(record, indent=2, allow_nan=False))
    return record


def save_preview(scene: SceneData, gaussians: Gaussians3D, path: Path, view: int = 0) -> None:
    from PIL import Image

    renderer = get_rasterizer("torch", device=torch.device("cpu"))
    with torch.no_grad():
        out = renderer.render(gaussians, scene.cameras[view])
    row = torch.cat([scene.images[view], out.color.clamp(0, 1)], dim=1)
    Image.fromarray((row * 255).byte().numpy()).save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/beam_convergence_dynamics")
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["beam-adc", "random-adc", "beam-fixed", "random-fixed"],
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=ITERATIONS,
        help="Smoke-test override; the frozen protocol value is the default.",
    )
    parser.add_argument(
        "--grad-threshold",
        type=float,
        default=GRAD_THRESHOLD,
        help="Calibration override; the frozen protocol value is the default.",
    )
    args = parser.parse_args()
    torch.manual_seed(SEED)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    dataset = CompactDataset.load(DATASET, device="cpu")
    scene = build_scene(dataset)
    beam, beam_receipt = place_beam(dataset)
    random_init, random_receipt = place_random(dataset)
    print(f"[place] beam {beam_receipt}", flush=True)
    reference_surface = beam.means.detach().clone()
    inits = {"beam": beam, "random": random_init}
    records = {}
    for arm in args.arms:
        kind, mode = arm.rsplit("-", 1)
        record = run_arm(
            arm,
            inits[kind],
            scene,
            reference_surface,
            mode == "adc",
            out,
            args.iterations,
            args.grad_threshold,
        )
        records[arm] = record
        final = Gaussians3D.load_ply(out / arm / "gaussians_final.ply")
        save_preview(scene, final, out / arm / "preview_final.png")
        save_preview(scene, inits[kind], out / arm / "preview_init.png")
    summary = {
        "schema": "rtgs.beam_convergence_dynamics.v1",
        "dataset": str(DATASET.relative_to(ROOT)),
        "selected_global_views": SELECTED_VIEWS,
        "downscale": DOWNSCALE,
        "n_init": N_INIT,
        "iterations": args.iterations,
        "density": {
            "start": density_config(args.grad_threshold).start_iter,
            "stop": density_config(args.grad_threshold).stop_iter,
            "every": density_config(args.grad_threshold).every,
            "grad_threshold": args.grad_threshold,
            "absgrad": False,
            "opacity_reset_every": density_config(args.grad_threshold).opacity_reset_every,
            "max_gaussians": density_config(args.grad_threshold).max_gaussians,
        },
        "seed": SEED,
        "placements": {"beam": beam_receipt, "random": random_receipt},
        "arms": {
            name: {
                "init_psnr_fg": record["init_metrics"].get("psnr_fg"),
                "init_alpha_iou": record["init_metrics"].get("alpha_iou"),
                "init_alpha_inside": record["init_metrics"].get("alpha_inside"),
                "final_psnr_fg": record["curve"][-1].get("metric_psnr_fg"),
                "final_alpha_iou": record["curve"][-1].get("metric_alpha_iou"),
                "final_n": record["final_n"],
                "final_survivors": record["curve"][-1]["survivors"],
                "elapsed_seconds": record["elapsed_seconds"],
            }
            for name, record in records.items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary["arms"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
