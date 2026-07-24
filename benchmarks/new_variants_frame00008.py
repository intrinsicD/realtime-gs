#!/usr/bin/env python3
"""Compare the four 2026-07-23 opt-in variants on Janelle ``frame_00008``.

This is a prospective, single-scene development experiment.  Four stage-1 fits isolate the
unchanged native baseline, fixed-capacity pool/free-list recycling, soft mask containment, and
feature-aware structure-tensor initialization.  Every fit is lifted through the same carve
configuration and refined through the same CUDA/gsplat schedule.  The baseline refinement also
enables train-only best-checkpoint selection; its captured final iterate and selected checkpoint
form a paired stage-3 comparison with an identical optimization trajectory.

The calibrated loader selects eight evenly spaced cameras.  Its eighth local camera is reporting
only: stage-1 fitting, lifting, training, and checkpoint selection use the other seven cameras.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from PIL import Image, ImageDraw

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.metrics import image_metrics
from rtgs.data.calibrated import load_calibrated_scene
from rtgs.image2gs.fit import FitConfig, fit_image
from rtgs.image2gs.renderer2d import render_gaussian_coverage_2d, render_gaussians_2d
from rtgs.lift.carve import CarveLifter
from rtgs.optim.density import DensityConfig
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.base import get_rasterizer
from rtgs.visualize import save_reconstruction_artifacts

ROOT = Path(__file__).resolve().parents[1]
RAW_SCENE = Path("/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008")
CALIBRATION = ROOT / "dataset/2025_03_07_stage_with_fabric/calibration_dome.json"
COMPACT_MANIFEST = (
    ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d/manifest.json"
)
DEFAULT_PROTOCOL = ROOT / "benchmarks/results/20260724_new_variants_frame00008_PREREG_V3.md"
DEFAULT_OUT = ROOT / "runs/new_variants_frame00008_20260724_v3"
VIEWER_MANIFEST = ROOT / "benchmarks/results/20260724_new_variants_frame00008_VIEWER.json"

SEED = 0
DOWNSCALE = 16
MAX_IMAGES = 8
TEST_EVERY = 8
STAGE1_ARMS = ("baseline", "pool", "mask-containment", "structure-tensor")
REPORT_ARMS = STAGE1_ARMS + ("best-train-checkpoint",)
REPRESENTATIVE_TRAIN_VIEW = "C0014"
TRAIN_ITERATIONS = 2_000
EVAL_EVERY = 100

SOURCE_FILES = (
    "benchmarks/new_variants_frame00008.py",
    "src/rtgs/data/calibrated.py",
    "src/rtgs/image2gs/fit.py",
    "src/rtgs/image2gs/pool.py",
    "src/rtgs/image2gs/renderer2d.py",
    "src/rtgs/image2gs/structure_init.py",
    "src/rtgs/lift/carve.py",
    "src/rtgs/optim/trainer.py",
    "src/rtgs/optim/density.py",
    "src/rtgs/optim/strategies.py",
    "src/rtgs/render/gsplat_backend.py",
    "src/rtgs/visualize.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _command_output(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display_path = str(resolved.relative_to(ROOT))
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _input_path(view_name: str, kind: str) -> Path:
    if kind == "rgb":
        return RAW_SCENE / "rgb" / f"{view_name}.jpg"
    candidates = (
        RAW_SCENE / "mask" / f"mask_{view_name}.png",
        RAW_SCENE / "mask" / f"mask_{view_name}.jpg",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    return path


def _repository_binding() -> dict[str, Any]:
    hashes = {relative: _sha256(ROOT / relative) for relative in SOURCE_FILES}
    status = _command_output("git", "status", "--porcelain=v1")
    tracked_diff = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--"], cwd=ROOT)
    return {
        "git_revision": _command_output("git", "rev-parse", "HEAD"),
        "git_branch": _command_output("git", "branch", "--show-current"),
        "git_status_lines": status.splitlines(),
        "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "source_files": hashes,
        "source_aggregate_sha256": _canonical_hash(hashes),
    }


def _fit_configs() -> dict[str, FitConfig]:
    common = {
        "n_gaussians": 640,
        "max_gaussians": None,
        "iterations": 300,
        "backend": "native",
        "native_renderer": "cuda",
        "lr": 1e-2,
        "grad_init_mix": 0.7,
        "row_chunk": 64,
        "log_every": 50,
        "convergence_patience": 0,
        "appearance_parameterization": "weight_color_9p",
    }
    return {
        "baseline": FitConfig(**common),
        "pool": FitConfig(
            **common,
            pool=True,
            pool_capacity=1_280,
            pool_triage_every=50,
            pool_prune_count=32,
            pool_spawn_count=32,
            pool_min_live=1,
        ),
        # 5.0 is the only strength already exercised by the mechanism test.  It is frozen here as
        # a development treatment, not selected as a default.
        "mask-containment": FitConfig(**common, mask_coverage_weight=5.0),
        "structure-tensor": FitConfig(**common, init_strategy="structure_tensor"),
    }


def _carve_config() -> dict[str, Any]:
    return {
        "grid_res": 32,
        "bounds_scale": 0.5,
        "min_views": 2,
        "hull_fraction": 0.85,
        "color_std_sigma": 0.20,
        "color_match_sigma": 0.35,
        "coverage_thresh": 0.40,
        "samples_per_ray": 48,
        "min_score": 0.05,
        "min_weight": 0.05,
        "merge": True,
        "merge_voxel_scale": 1.0,
        "init_opacity": 0.10,
        "sh_degree": 0,
    }


def _density_config() -> DensityConfig:
    return DensityConfig(
        start_iter=100,
        stop_iter=1_000,
        every=100,
        grad_threshold=8e-4,
        absgrad=True,
        split_scale_frac=0.01,
        split_factor=1.6,
        prune_opacity=0.005,
        prune_scale_frac=0.1,
        max_gaussians=20_000,
        opacity_reset_every=1_000,
        opacity_reset_value=0.011,
        revised_opacity=True,
    )


def _train_config(checkpoint_policy: str) -> TrainConfig:
    return TrainConfig(
        iterations=TRAIN_ITERATIONS,
        rasterizer="gsplat",
        device="cuda",
        densify=True,
        density_strategy="gsplat-default",
        density=_density_config(),
        eval_every=EVAL_EVERY,
        checkpoint_policy=checkpoint_policy,
        target_sh_degree=3,
        sh_degree_interval=250,
        use_masks=True,
        outside_alpha_lambda=0.01,
        mask_alpha_lambda=0.05,
        random_background=True,
        packed=False,
        antialiased=True,
        seed=SEED,
    )


def _stage1_metrics(
    fit: Gaussians2D,
    image: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor]:
    height, width = image.shape[:2]
    with torch.no_grad():
        rendered = render_gaussians_2d(fit, height, width, row_chunk=64, renderer="cuda").clamp(
            0.0, 1.0
        )
        coverage = render_gaussian_coverage_2d(fit, height, width, row_chunk=64)
        foreground = mask > 0.5
        background = ~foreground
        masked_target = image * mask[..., None]
        values = image_metrics(rendered, image, mask)
        masked_mse = (rendered - masked_target).square().mean().clamp_min(1e-12)
        support = coverage > 0.10
        intersection = (support & foreground).sum()
        union = (support | foreground).sum().clamp_min(1)
        metrics = {
            "psnr_fg": float(values["psnr_fg"]),
            "psnr_crop": float(values["psnr_crop"]),
            "ssim_crop": float(values["ssim_crop"]),
            "masked_full_psnr": float(-10.0 * torch.log10(masked_mse)),
            "coverage_iou_at_0_1": float(intersection / union),
            "coverage_inside": float(coverage[foreground].mean()),
            "coverage_outside": (
                float(coverage[background].mean()) if bool(background.any()) else 0.0
            ),
            "foreground_support_recall_at_0_1": float(support[foreground].float().mean()),
            "n_gaussians": float(fit.n),
        }
    return metrics, rendered.detach().cpu()


def _aggregate_stage1(records: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = tuple(records[0]["metrics"])
    return {
        name: fmean(float(record["metrics"][name]) for record in records) for name in metric_names
    }


def _save_gaussians3d(directory: Path, initial: Gaussians3D, final: Gaussians3D) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    init_npz = directory / "gaussians_init.npz"
    init_ply = directory / "gaussians_init.ply"
    final_npz = directory / "gaussians_final.npz"
    final_ply = directory / "gaussians_final.ply"
    initial.save_npz(init_npz)
    initial.save_ply(init_ply)
    final.save_npz(final_npz)
    final.save_ply(final_ply)
    return {
        "initial_npz": _artifact(init_npz),
        "initial_ply": _artifact(init_ply),
        "final_npz": _artifact(final_npz),
        "final_ply": _artifact(final_ply),
    }


def _evaluate_3d(scene, model: Gaussians3D, renderer) -> dict[str, dict[str, float]]:
    device_model = model.to("cuda")
    return {
        "train": Trainer.evaluate_metrics(
            scene, device_model, renderer, indices=scene.training_views
        ),
        "test": Trainer.evaluate_metrics(
            scene, device_model, renderer, indices=scene.testing_views
        ),
    }


def _to_pil(value: torch.Tensor) -> Image.Image:
    array = value.detach().cpu().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
    return Image.fromarray(array)


def _labeled_panel(value: torch.Tensor, label: str, header: int = 28) -> Image.Image:
    image = _to_pil(value)
    panel = Image.new("RGB", (image.width, image.height + header), "white")
    panel.paste(image, (0, header))
    ImageDraw.Draw(panel).text((6, 8), label, fill="black")
    return panel


def _horizontal_sheet(panels: list[Image.Image]) -> Image.Image:
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width
    return sheet


def _vertical_sheet(rows: list[Image.Image]) -> Image.Image:
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    return sheet


def _stage1_contact_sheet(
    path: Path,
    scene,
    fits: dict[str, list[Gaussians2D]],
    rendered: dict[str, dict[str, torch.Tensor]],
) -> None:
    index = scene.view_names.index(REPRESENTATIVE_TRAIN_VIEW)
    mask = scene.masks[index]
    target = scene.images[index] * mask[..., None]
    panels = [_labeled_panel(target, f"{REPRESENTATIVE_TRAIN_VIEW} masked target")]
    for arm in STAGE1_ARMS:
        image = rendered[arm][REPRESENTATIVE_TRAIN_VIEW]
        panels.append(_labeled_panel(image, f"{arm} · N={fits[arm][index].n}"))
    _horizontal_sheet(panels).save(path)


def _reconstruction_contact_sheet(
    path: Path,
    scene,
    view_index: int,
    initializations: dict[str, Gaussians3D],
    finals: dict[str, Gaussians3D],
    renderer,
) -> None:
    target = scene.images[view_index]
    if scene.masks is not None:
        target = target * scene.masks[view_index][..., None]
    camera = scene.cameras[view_index].to("cuda")
    rows = []
    with torch.no_grad():
        for arm in REPORT_ARMS:
            initial = renderer.render(initializations[arm].to("cuda"), camera).color.clamp(0, 1)
            final = renderer.render(finals[arm].to("cuda"), camera).color.clamp(0, 1)
            error = (final.cpu() - target).abs().mul(4.0).clamp(0, 1)
            panels = [
                _labeled_panel(target, f"{arm} · target"),
                _labeled_panel(initial, f"{arm} · initial"),
                _labeled_panel(final, f"{arm} · final"),
                _labeled_panel(error, f"{arm} · |error|×4"),
            ]
            rows.append(_horizontal_sheet(panels))
    _vertical_sheet(rows).save(path)


def _write_viewer_manifest(out: Path) -> None:
    methods = []
    for arm in REPORT_ARMS:
        directory = out / "models" / arm
        methods.append(
            {
                "name": arm,
                "initial": os.path.relpath(
                    directory / "gaussians_init.ply", VIEWER_MANIFEST.parent
                ),
                "final": os.path.relpath(directory / "gaussians_final.ply", VIEWER_MANIFEST.parent),
            }
        )
    _write_json(
        VIEWER_MANIFEST,
        {
            "schema": "rtgs.viewer-comparison.v1",
            "methods": methods,
        },
    )


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("gsplat", "numpy", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": packages,
        "ld_preload": os.environ.get("LD_PRELOAD"),
    }


class _FinalSnapshotCapture:
    """Capture only the final evaluation checkpoint without retaining earlier GPU tensors."""

    def __init__(self) -> None:
        self.snapshot: Gaussians3D | None = None

    def __call__(self, snapshot: Gaussians3D, step: int) -> None:
        if step == TRAIN_ITERATIONS:
            self.snapshot = snapshot.to("cpu")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out = args.out.expanduser().resolve()
    protocol = args.protocol.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {out}")
    if not protocol.is_file():
        raise FileNotFoundError(f"frozen protocol is missing: {protocol}")
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen experiment requires CUDA")
    if not RAW_SCENE.is_dir() or not CALIBRATION.is_file() or not COMPACT_MANIFEST.is_file():
        raise FileNotFoundError("one or more frozen Janelle inputs are missing")

    out.mkdir(parents=True)
    try:
        scene = load_calibrated_scene(
            RAW_SCENE,
            calibration_path=CALIBRATION,
            downscale=DOWNSCALE,
            max_images=MAX_IMAGES,
            test_every=TEST_EVERY,
            load_masks=True,
            undistort=True,
        )
        if scene.masks is None:
            raise RuntimeError("the frozen experiment requires complete masks")
        if scene.view_names != [
            "C0001",
            "C0008",
            "C0014",
            "C0021",
            "C0026",
            "C0031",
            "C0039",
            "C1004",
        ]:
            raise RuntimeError(f"selected calibrated view order drifted: {scene.view_names}")
        if scene.training_views != list(range(7)) or scene.testing_views != [7]:
            raise RuntimeError("frozen seven-train/one-test split drifted")

        repository = _repository_binding()
        configs = _fit_configs()
        input_records = {}
        for name, image, mask in zip(scene.view_names, scene.images, scene.masks, strict=True):
            rgb_path = _input_path(name, "rgb")
            mask_path = _input_path(name, "mask")
            input_records[name] = {
                "rgb": _artifact(rgb_path),
                "mask": _artifact(mask_path),
                "loaded_rgb_tensor_sha256": _tensor_hash(image),
                "loaded_mask_tensor_sha256": _tensor_hash(mask),
            }
        plan = {
            "schema": "rtgs.new_variants_frame00008.plan.v1",
            "created_utc": dt.datetime.now(dt.UTC).isoformat(),
            "scope": (
                "single-scene, single-seed development comparison; one untouched calibrated "
                "test camera; no default or performance claim"
            ),
            "protocol": _artifact(protocol),
            "repository": repository,
            "environment": _environment(),
            "seed": SEED,
            "scene": {
                "raw_path": str(RAW_SCENE),
                "calibration": _artifact(CALIBRATION),
                "compact_manifest_context_only": _artifact(COMPACT_MANIFEST),
                "downscale": DOWNSCALE,
                "max_images": MAX_IMAGES,
                "test_every": TEST_EVERY,
                "view_names": scene.view_names,
                "train_indices": scene.training_views,
                "test_indices": scene.testing_views,
                "resolution": [scene.cameras[0].width, scene.cameras[0].height],
                "inputs": input_records,
            },
            "arms": {name: dataclasses.asdict(config) for name, config in configs.items()},
            "carve": _carve_config(),
            "training_final": dataclasses.asdict(_train_config("final")),
            "training_paired_checkpoint": dataclasses.asdict(_train_config("best_train_psnr")),
        }
        plan_path = out / "plan.json"
        _write_json(plan_path, plan)
        plan_hash = _sha256(plan_path)

        fits: dict[str, list[Gaussians2D]] = {}
        stage1_records: dict[str, Any] = {}
        rendered_stage1: dict[str, dict[str, torch.Tensor]] = {}
        train_scene = scene.subset(scene.training_views)
        for arm in STAGE1_ARMS:
            print(f"[stage1] {arm}", flush=True)
            arm_dir = out / "stage1" / arm
            arm_dir.mkdir(parents=True)
            arm_fits: list[Gaussians2D] = []
            arm_records = []
            rendered_stage1[arm] = {}
            for local_index, (view_name, image, mask) in enumerate(
                zip(
                    train_scene.view_names,
                    train_scene.images,
                    train_scene.masks,
                    strict=True,
                )
            ):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                image_cuda = image.to("cuda")
                mask_cuda = mask.to("cuda")
                torch.cuda.synchronize()
                started = time.perf_counter()
                fit, history = fit_image(
                    image_cuda,
                    configs[arm],
                    seed=SEED + local_index,
                    mask=mask_cuda,
                )
                torch.cuda.synchronize()
                wall_seconds = time.perf_counter() - started
                metrics, rendered = _stage1_metrics(fit, image_cuda, mask_cuda)
                fit_cpu = fit.to("cpu")
                fit_path = arm_dir / f"{view_name}.npz"
                fit_cpu.save_npz(fit_path)
                history_path = arm_dir / f"{view_name}.history.json"
                _write_json(history_path, history)
                record = {
                    "view_name": view_name,
                    "seed": SEED + local_index,
                    "metrics": metrics,
                    "history_final_psnr": float(history["final_psnr"]),
                    "history_final_psnr_full": float(history["final_psnr_full"]),
                    "history_stopped_iter": int(history["stopped_iter"]),
                    "wall_seconds_nondecisional": wall_seconds,
                    "peak_cuda_allocated_bytes_nondecisional": int(
                        torch.cuda.max_memory_allocated()
                    ),
                    "fit": _artifact(fit_path),
                    "history": _artifact(history_path),
                }
                if arm == "pool":
                    record["pool"] = {
                        "capacity": int(history["pool_capacity"]),
                        "live_count": int(history["live_count"]),
                        "triage_steps": list(
                            range(
                                configs[arm].pool_triage_every,
                                configs[arm].iterations,
                                configs[arm].pool_triage_every,
                            )
                        ),
                    }
                arm_records.append(record)
                arm_fits.append(fit_cpu)
                rendered_stage1[arm][view_name] = rendered
                print(
                    f"  {view_name}: N={fit_cpu.n} FG={metrics['psnr_fg']:.4f} dB "
                    f"outside={metrics['coverage_outside']:.6f}",
                    flush=True,
                )
                del image_cuda, mask_cuda, fit
            fits[arm] = arm_fits
            stage1_records[arm] = {
                "config": dataclasses.asdict(configs[arm]),
                "views": arm_records,
                "aggregate": _aggregate_stage1(arm_records),
                "wall_seconds_nondecisional": sum(
                    float(record["wall_seconds_nondecisional"]) for record in arm_records
                ),
            }

        stage1_sheet = out / "stage1_contact_sheet.png"
        # ``fits`` is train-only, and C0014 is local training index 2.
        _stage1_contact_sheet(stage1_sheet, train_scene, fits, rendered_stage1)

        renderer = get_rasterizer(
            "gsplat",
            device=torch.device("cuda"),
            packed=False,
            antialiased=True,
        )
        initializations: dict[str, Gaussians3D] = {}
        finals: dict[str, Gaussians3D] = {}
        reconstruction_records: dict[str, Any] = {}

        for arm in STAGE1_ARMS:
            print(f"[lift] {arm}", flush=True)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            lift_started = time.perf_counter()
            lifter = CarveLifter(**_carve_config())
            initial = lifter.lift(
                [fit.to("cuda") for fit in fits[arm]],
                train_scene.to("cuda"),
            ).detach()
            torch.cuda.synchronize()
            lift_seconds = time.perf_counter() - lift_started
            initial_cpu = initial.to("cpu")
            initializations[arm] = initial_cpu
            initial_metrics = _evaluate_3d(scene, initial_cpu, renderer)
            print(
                f"  init N={initial_cpu.n} "
                f"test FG={initial_metrics['test'].get('psnr_fg', float('nan')):.4f} dB",
                flush=True,
            )

            capture_final = _FinalSnapshotCapture()
            policy = "best_train_psnr" if arm == "baseline" else "final"
            config = _train_config(policy)
            print(f"[refine] {arm} policy={policy}", flush=True)
            selected, history = Trainer(config).train(
                scene,
                initial,
                checkpoint_callback=capture_final,
            )
            if capture_final.snapshot is None:
                raise RuntimeError(f"{arm} did not capture exactly one final snapshot")
            final_cpu = capture_final.snapshot
            selected_cpu = selected.to("cpu")
            # The returned final and callback final must be the same effective tensors.
            if policy == "final" and any(
                not torch.equal(getattr(selected_cpu, name), getattr(final_cpu, name))
                for name in ("means", "quats", "log_scales", "opacity", "sh")
            ):
                raise RuntimeError(f"{arm} callback and returned final snapshot differ")
            finals[arm] = final_cpu
            final_metrics = _evaluate_3d(scene, final_cpu, renderer)
            history_path = out / "models" / arm / "training_history.json"
            _write_json(history_path, history)
            artifacts = _save_gaussians3d(out / "models" / arm, initial_cpu, final_cpu)
            visual = save_reconstruction_artifacts(
                scene,
                initial_cpu.to("cuda"),
                final_cpu.to("cuda"),
                out / "models" / arm,
                rasterizer="gsplat",
                packed=False,
                antialiased=True,
                max_comparisons=8,
                max_animation_frames=12,
            )
            reconstruction_records[arm] = {
                "source_stage1_arm": arm,
                "checkpoint_policy": policy,
                "lift_seconds_nondecisional": lift_seconds,
                "init_n_gaussians": initial_cpu.n,
                "final_n_gaussians": final_cpu.n,
                "init_metrics": initial_metrics,
                "final_metrics": final_metrics,
                "training_history": _artifact(history_path),
                "selected_step": history.get("selected_step", TRAIN_ITERATIONS),
                "selected_train_psnr": history.get("selected_train_psnr"),
                "checkpoint_selection_views": history.get("checkpoint_selection_views"),
                "artifacts": artifacts,
                "visuals": {name: _artifact(Path(path)) for name, path in visual.items()},
            }
            print(
                f"  final N={final_cpu.n} "
                f"test FG={final_metrics['test'].get('psnr_fg', float('nan')):.4f} dB",
                flush=True,
            )

            if arm == "baseline":
                checkpoint_arm = "best-train-checkpoint"
                initializations[checkpoint_arm] = initial_cpu
                finals[checkpoint_arm] = selected_cpu
                selected_metrics = _evaluate_3d(scene, selected_cpu, renderer)
                checkpoint_artifacts = _save_gaussians3d(
                    out / "models" / checkpoint_arm,
                    initial_cpu,
                    selected_cpu,
                )
                selected_visual = save_reconstruction_artifacts(
                    scene,
                    initial_cpu.to("cuda"),
                    selected_cpu.to("cuda"),
                    out / "models" / checkpoint_arm,
                    rasterizer="gsplat",
                    packed=False,
                    antialiased=True,
                    max_comparisons=8,
                    max_animation_frames=12,
                )
                reconstruction_records[checkpoint_arm] = {
                    "source_stage1_arm": "baseline",
                    "checkpoint_policy": "best_train_psnr",
                    "paired_trajectory_with": "baseline",
                    "init_n_gaussians": initial_cpu.n,
                    "final_n_gaussians": selected_cpu.n,
                    "init_metrics": initial_metrics,
                    "final_metrics": selected_metrics,
                    "training_history": _artifact(history_path),
                    "selected_step": int(history["selected_step"]),
                    "selected_train_psnr": float(history["selected_train_psnr"]),
                    "checkpoint_selection_views": history["checkpoint_selection_views"],
                    "artifacts": checkpoint_artifacts,
                    "visuals": {
                        name: _artifact(Path(path)) for name, path in selected_visual.items()
                    },
                }
                print(
                    f"  selected step={history['selected_step']} "
                    f"test FG={selected_metrics['test'].get('psnr_fg', float('nan')):.4f} dB",
                    flush=True,
                )

            del initial, selected
            torch.cuda.empty_cache()

        if set(initializations) != set(REPORT_ARMS) or set(finals) != set(REPORT_ARMS):
            raise RuntimeError("report arm assembly is incomplete")
        train_sheet = out / "reconstruction_train_C0014.png"
        test_sheet = out / "reconstruction_heldout_C1004.png"
        _reconstruction_contact_sheet(
            train_sheet,
            scene,
            scene.view_names.index(REPRESENTATIVE_TRAIN_VIEW),
            initializations,
            finals,
            renderer,
        )
        _reconstruction_contact_sheet(
            test_sheet,
            scene,
            scene.testing_views[0],
            initializations,
            finals,
            renderer,
        )
        _write_viewer_manifest(out)

        current_source_hashes = {relative: _sha256(ROOT / relative) for relative in SOURCE_FILES}
        if current_source_hashes != repository["source_files"]:
            raise RuntimeError("bound experiment source changed during execution")
        if _sha256(protocol) != plan["protocol"]["sha256"]:
            raise RuntimeError("frozen protocol changed during execution")
        for view_name, record in input_records.items():
            if _sha256(_input_path(view_name, "rgb")) != record["rgb"]["sha256"]:
                raise RuntimeError(f"RGB input changed during execution: {view_name}")
            if _sha256(_input_path(view_name, "mask")) != record["mask"]["sha256"]:
                raise RuntimeError(f"mask input changed during execution: {view_name}")

        summary = {
            "schema": "rtgs.new_variants_frame00008.result.v1",
            "status": "complete",
            "completed_utc": dt.datetime.now(dt.UTC).isoformat(),
            "scope": plan["scope"],
            "plan": {**_artifact(plan_path), "sha256": plan_hash},
            "protocol": plan["protocol"],
            "repository": repository,
            "scene": plan["scene"],
            "stage1": stage1_records,
            "reconstruction": reconstruction_records,
            "comparison_visuals": {
                "stage1": _artifact(stage1_sheet),
                "train": _artifact(train_sheet),
                "heldout": _artifact(test_sheet),
            },
            "viewer_manifest": _artifact(VIEWER_MANIFEST),
            "timing_policy": (
                "wall times and peak allocations describe this contended local execution only; "
                "they are not performance evidence"
            ),
        }
        summary_path = out / "summary.json"
        _write_json(summary_path, summary)
        print(f"[complete] {summary_path.relative_to(ROOT)}", flush=True)
        print(f"[viewer] {VIEWER_MANIFEST.relative_to(ROOT)}", flush=True)
        return 0
    except BaseException as error:
        _write_json(
            out / "failure.json",
            {
                "schema": "rtgs.new_variants_frame00008.failure.v1",
                "failed_utc": dt.datetime.now(dt.UTC).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
