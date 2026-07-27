#!/usr/bin/env python3
"""Full-resolution ADR-002 carrier-refinement experiment on masked Janelle frame 00008.

This is a single-scene, single-seed, outcome-exposed development experiment.  It exercises every
runnable baseline and ablation named by ``docs/PAPER_PLAN_beam_fusion.md`` at the native
5328x4608 resolution.  The original-SfM 3DGS cells fail closed when no COLMAP model is available;
matched random-initialization RGB/JPEG controls are reported separately and never relabelled.

The harness requires an already-written preregistration, binds the raw RGB/mask files to the hashes
stored inside the compact captures, keeps report-only cameras sealed, saves a calibrated result
bundle for every arm, and generates an HTML overview plus an orbit-viewer comparison manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.metrics import image_metrics, masked_crop
from rtgs.data.calibrated import load_calibrated_scene
from rtgs.data.compact_views import CompactDataset, file_sha256
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.data.scene import SceneData
from rtgs.lift.beam_fusion import BeamFusionConfig, BeamFusionResult, fuse_gaussian_beams
from rtgs.lift.carrier_refinement import CarrierRepairConfig, repair_beam_carriers
from rtgs.optim.carrier_schedule import (
    CarrierLineageTracker,
    CarrierOptimizationConfig,
    optimize_carriers,
)
from rtgs.optim.density import DensityConfig
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.base import get_rasterizer

ROOT = Path(__file__).resolve().parents[1]
COMPACT_PATH = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d"
DEFAULT_RAW_FRAME = Path("/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008")
DEFAULT_OUT = ROOT / "runs/carrier_refinement_fullres_frame00008_20260727"
DEFAULT_PROTOCOL = ROOT / "benchmarks/results/20260727_carrier_refinement_fullres_PREREG_V4.md"

VALIDATION = ("C0031", "C1000", "C1002")
REPORT_ONLY_SEALED = ("C1001", "C1004")

SEED = 27027
N_INIT = 2400
ITERATIONS = 160
WARMUP_ITERATIONS = 30
CLONE_ITERATIONS = 40
HIGHER_SH_ITERATIONS = 30
STANDARD_ITERATIONS = 60
MAX_GAUSSIANS = 9600
JPEG_QUALITY = 50
LPIPS_MAX_SIZE = 512

ARMS = (
    "beam-only",
    "beam-standard",
    "carrier-schedule",
    "carrier-schedule-clone",
    "no-covariance",
    "no-opacity",
    "no-warmup",
    "split-immediately",
    "clone-only",
    "particle",
    "random-rgb",
    "random-jpeg-q50",
    "means-only",
)

ARM_DESCRIPTIONS = {
    "beam-only": "Beam Fusion carrier state without dense-image optimization",
    "beam-standard": "unrepaired Beam Fusion handed immediately to ordinary 3DGS",
    "carrier-schedule": "repair + warm-up + higher SH + standard handover, no clone phase",
    "carrier-schedule-clone": "complete ADR-002 repair and maturation schedule",
    "no-covariance": "complete schedule with covariance repair disabled",
    "no-opacity": "complete schedule with opacity repair disabled",
    "no-warmup": "complete schedule with fixed-topology warm-up disabled",
    "split-immediately": "repaired carriers handed immediately to classic split/prune",
    "clone-only": "repaired carriers mature through protected clones; no standard handover",
    "particle": "low-opacity local carrier children with aggressive unsuccessful-child pruning",
    "random-rgb": "matched random initialization trained on original full-resolution RGB",
    "random-jpeg-q50": "same random initialization trained on second-generation JPEG quality 50",
    "means-only": "Beam means with global color, isotropic covariance, and fixed opacity",
}


def derive_splits(dataset: CompactDataset) -> dict:
    """Apply the frozen outcome-independent Stage split rule without benchmark imports."""

    ids = sorted(view.view_id for view in dataset.views)
    excluded = set(VALIDATION) | set(REPORT_ONLY_SEALED)
    missing = excluded - set(ids)
    if missing:
        raise KeyError(f"{dataset.name} is missing reserved views {sorted(missing)}")
    pool = [view_id for view_id in ids if view_id not in excluded]
    if len(pool) < 8:
        raise ValueError(f"{dataset.name} has too few eligible training views: {len(pool)}")
    count = len(pool)
    train8 = tuple(pool[(index * count) // 8] for index in range(8))
    train3 = (train8[0], train8[3], train8[6])
    return {
        "train3": train3,
        "train8": train8,
        "train8_extra": tuple(view_id for view_id in train8 if view_id not in train3),
        "validation": VALIDATION,
        "report_only_sealed": REPORT_ONLY_SEALED,
        "pool_size": count,
        "rule": (
            "sorted eligible IDs; train8 = pool[(i*n)//8] for i in 0..7; "
            "train3 = train8[0],[3],[6]; validation and report-only removed before selection"
        ),
    }


@dataclass(frozen=True)
class ArmRun:
    final: Gaussians3D
    history: dict
    diagnostics: dict
    training_seconds: float
    peak_vram_gb: float
    peak_vram_reserved_gb: float
    iterations: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _environment() -> dict:
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_status_porcelain": _git(["status", "--porcelain"]),
    }
    if torch.cuda.is_available():
        result.update(
            {
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
        )
    try:
        import gsplat

        result["gsplat"] = gsplat.__version__
    except ImportError:
        result["gsplat"] = None
    try:
        import lpips

        result["lpips_module"] = str(Path(lpips.__file__).resolve())
    except ImportError:
        result["lpips_module"] = None
    return result


def _train_inputs(dataset: CompactDataset, train_ids: tuple[str, ...]) -> ReconstructionInputs:
    complete = dataset.to_reconstruction_inputs()
    positions = [complete.view_names.index(view_id) for view_id in train_ids]
    return ReconstructionInputs(
        observations=[complete.observations[index] for index in positions],
        cameras=[complete.cameras[index] for index in positions],
        view_names=[complete.view_names[index] for index in positions],
        bounds_hint=complete.bounds_hint,
        name=f"{complete.name}-carrier-fullres-train",
    )


def _raw_source_paths(raw_frame: Path, view) -> tuple[Path, Path]:
    rgb_record = view.source["rgb"]
    mask_record = view.source["mask"]
    if mask_record is None:
        raise ValueError(f"{view.view_id} has no source mask binding")
    return (
        raw_frame / "rgb" / rgb_record["name"],
        raw_frame / "mask" / mask_record["name"],
    )


def _bind_raw_sources(
    dataset: CompactDataset,
    raw_frame: Path,
    selected_ids: tuple[str, ...],
) -> dict:
    views = {view.view_id: view for view in dataset.views}
    receipt = {}
    for view_id in selected_ids:
        view = views[view_id]
        rgb, mask = _raw_source_paths(raw_frame, view)
        actual_rgb = file_sha256(rgb)
        actual_mask = file_sha256(mask)
        expected_rgb = view.source["rgb"]["sha256"]
        expected_mask = view.source["mask"]["sha256"]
        if actual_rgb != expected_rgb or actual_mask != expected_mask:
            raise RuntimeError(f"raw source digest mismatch for {view_id}")
        receipt[view_id] = {
            "rgb": {
                "path": str(rgb),
                "bytes": rgb.stat().st_size,
                "sha256": actual_rgb,
            },
            "mask": {
                "path": str(mask),
                "bytes": mask.stat().st_size,
                "sha256": actual_mask,
            },
        }
    return receipt


def _load_raw_scene(
    raw_frame: Path,
    train_ids: tuple[str, ...],
    validation_ids: tuple[str, ...],
    bounds_hint,
    *,
    downscale: int,
) -> tuple[SceneData, float]:
    order = (*train_ids, *validation_ids)
    started = time.perf_counter()
    scene = load_calibrated_scene(
        raw_frame,
        downscale=downscale,
        test_every=0,
        load_masks=True,
        undistort=True,
        view_ids=order,
    )
    elapsed = time.perf_counter() - started
    if scene.masks is None:
        raise RuntimeError("full-resolution masked experiment requires every selected mask")
    scene.train_indices = list(range(len(train_ids)))
    scene.test_indices = list(range(len(train_ids), len(order)))
    scene.bounds_hint = bounds_hint
    scene.name = "janelle-frame00008-fullres-carriers"
    scene.validate()
    return scene, elapsed


def _jpeg_scene(scene: SceneData, quality: int) -> tuple[SceneData, int]:
    images = list(scene.images)
    encoded_bytes = 0
    for index in scene.training_views:
        array = (images[index].clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()
        buffer = io.BytesIO()
        Image.fromarray(array, mode="RGB").save(
            buffer,
            format="JPEG",
            quality=quality,
            optimize=False,
            progressive=False,
            subsampling=2,
        )
        payload = buffer.getvalue()
        encoded_bytes += len(payload)
        with Image.open(io.BytesIO(payload)) as decoded:
            images[index] = (
                torch.from_numpy(np.array(decoded.convert("RGB"), dtype=np.float32, copy=True))
                / 255.0
            )
    compressed = SceneData(
        images=images,
        cameras=scene.cameras,
        view_names=scene.view_names,
        masks=scene.masks,
        train_indices=scene.train_indices,
        test_indices=scene.test_indices,
        bounds_hint=scene.bounds_hint,
        name=f"{scene.name}-jpeg-q{quality}",
    )
    compressed.validate()
    return compressed, encoded_bytes


def _beam(dataset: CompactDataset, inputs: ReconstructionInputs) -> tuple[BeamFusionResult, float]:
    extent = float(dataset.bounds_hint[1])
    config = BeamFusionConfig(
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
    result = fuse_gaussian_beams(inputs, config)
    elapsed = time.perf_counter() - started
    if result.n_components != N_INIT:
        raise RuntimeError(f"expected {N_INIT} Beam Fusion carriers, got {result.n_components}")
    return result, elapsed


def _simple_initializations(
    beam: Gaussians3D,
    inputs: ReconstructionInputs,
    bounds_hint,
) -> tuple[Gaussians3D, Gaussians3D]:
    center, extent = bounds_hint
    sigma = float(beam.scales.median())
    covariance = torch.eye(3, dtype=beam.means.dtype)[None].expand(beam.n, 3, 3).clone() * sigma**2
    global_color = torch.cat([field.colors for field in inputs.observations]).mean(dim=0)
    colors = global_color[None].expand(beam.n, 3).clamp(0, 1)
    opacity = torch.full((beam.n,), 0.10, dtype=beam.means.dtype)
    means_only = Gaussians3D.from_means_covs(
        beam.means,
        covariance,
        colors,
        opacity,
        sh_degree=0,
    )
    generator = torch.Generator().manual_seed(SEED)
    random_means = center.to(beam.means) + (
        torch.rand(beam.n, 3, generator=generator) - 0.5
    ) * float(extent)
    random = Gaussians3D.from_means_covs(
        random_means,
        covariance,
        colors,
        opacity,
        sh_degree=0,
    )
    return means_only, random


def _template(iterations: int) -> TrainConfig:
    return TrainConfig(
        iterations=iterations,
        rasterizer="gsplat",
        device="cuda",
        densify=True,
        density_strategy="classic",
        density=DensityConfig(
            start_iter=60,
            stop_iter=iterations,
            every=40,
            max_gaussians=MAX_GAUSSIANS,
            opacity_reset_every=0,
        ),
        eval_every=max(1, iterations),
        target_sh_degree=3,
        sh_degree_interval=1,
        use_masks=True,
        random_background=False,
        checkpoint_policy="final",
        packed=True,
        seed=SEED,
    )


def _run_standard(
    scene: SceneData,
    initialization: Gaussians3D,
    iterations: int,
    *,
    track_carriers: bool,
) -> ArmRun:
    tracker = CarrierLineageTracker(initialization) if track_carriers else None
    initial_boundary = None if tracker is None else tracker.snapshot("initial", initialization, 0)
    started = time.perf_counter()
    final, history = Trainer(_template(iterations)).train(
        scene,
        initialization,
        density_surgery_callback=None if tracker is None else tracker.apply_surgery,
    )
    elapsed = time.perf_counter() - started
    diagnostics = {}
    if tracker is not None:
        assert initial_boundary is not None
        diagnostics = {
            "phase_boundaries": [
                initial_boundary,
                tracker.snapshot("standard", final, iterations),
            ],
            "density_surgeries": tracker.surgeries,
        }
    return ArmRun(
        final=final,
        history=history,
        diagnostics=diagnostics,
        training_seconds=elapsed,
        peak_vram_gb=float(history.get("peak_vram_gb", 0.0)),
        peak_vram_reserved_gb=float(history.get("peak_vram_reserved_gb", 0.0)),
        iterations=iterations,
    )


def _run_schedule(
    scene: SceneData,
    initialization: Gaussians3D,
    *,
    warmup: int,
    clone: int,
    higher: int,
    standard: int,
    particle: bool = False,
) -> ArmRun:
    template = _template(max(1, warmup + clone + higher + standard))
    result = optimize_carriers(
        scene,
        initialization,
        CarrierOptimizationConfig(
            warmup_iterations=warmup,
            clone_iterations=clone,
            higher_sh_iterations=higher,
            standard_iterations=standard,
            clone_every=20,
            clone_grad_threshold=2e-4,
            clone_growth_factor=2.5,
            clone_jitter_fraction=0.10,
            particle_generation=particle,
            particle_child_opacity_scale=0.20,
            particle_prune_opacity=0.02,
            standard_every=40,
            standard_growth_factor=4.0,
            target_sh_degree=3,
            template=template,
        ),
    )
    history = {
        "schema": "rtgs.carrier-phased-history.v1",
        "phases": result.phase_histories,
    }
    return ArmRun(
        final=result.gaussians,
        history=history,
        diagnostics=result.diagnostics,
        training_seconds=float(result.diagnostics["total_seconds"]),
        peak_vram_gb=float(result.diagnostics["peak_vram_gb"]),
        peak_vram_reserved_gb=float(result.diagnostics["peak_vram_reserved_gb"]),
        iterations=warmup + clone + higher + standard,
    )


class LPIPSEvaluator:
    """Lazy, frozen AlexNet LPIPS evaluated on a bounded masked crop."""

    def __init__(self, device: torch.device):
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError(
                "LPIPS is preregistered and required; install the 'metrics' extra"
            ) from error
        self.module_path = str(Path(lpips.__file__).resolve())
        self.model = lpips.LPIPS(net="alex", verbose=False).eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        pred = masked_crop(prediction.clamp(0, 1), mask)
        truth = masked_crop(target.clamp(0, 1), mask)
        height, width = pred.shape[:2]
        scale = min(1.0, LPIPS_MAX_SIZE / max(height, width))
        size = (max(32, round(height * scale)), max(32, round(width * scale)))
        pred_nchw = pred.permute(2, 0, 1)[None]
        truth_nchw = truth.permute(2, 0, 1)[None]
        if pred_nchw.shape[-2:] != size:
            pred_nchw = F.interpolate(pred_nchw, size=size, mode="bilinear", align_corners=False)
            truth_nchw = F.interpolate(truth_nchw, size=size, mode="bilinear", align_corners=False)
        with torch.no_grad():
            return float(self.model(2 * pred_nchw - 1, 2 * truth_nchw - 1).reshape(()))


def _median(values: list[float]) -> float:
    return float(torch.tensor(values, dtype=torch.float64).median())


def _save_rgb(path: Path, tensor: torch.Tensor, max_width: int = 1000) -> None:
    image = tensor.detach().cpu().clamp(0, 1)
    array = (image.numpy() * 255.0).round().astype(np.uint8)
    pil = Image.fromarray(array, mode="RGB")
    if pil.width > max_width:
        height = round(pil.height * max_width / pil.width)
        pil = pil.resize((max_width, height), Image.Resampling.LANCZOS)
    pil.save(path, optimize=True)


def _save_comparison(
    path: Path,
    target: torch.Tensor,
    prediction: torch.Tensor,
    alpha: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    target = target * mask[..., None]
    residual = (prediction - target).abs().clamp(0, 0.25) * 4
    alpha_rgb = alpha[..., None].expand(-1, -1, 3)
    sheet = torch.cat([target, prediction, residual, alpha_rgb], dim=1)
    _save_rgb(path, sheet, max_width=1800)


def _evaluate(
    scene: SceneData,
    model: Gaussians3D,
    indices: list[int],
    lpips_metric: LPIPSEvaluator,
    *,
    preview_dir: Path | None,
    prefix: str,
) -> dict:
    renderer = get_rasterizer("gsplat", device=model.means.device, packed=True)
    per_view = {}
    with torch.no_grad():
        for position, index in enumerate(indices):
            target = scene.images[index].to(model.means.device)
            mask = scene.masks[index].to(model.means.device)
            output = renderer.render(model, scene.cameras[index].to(model.means.device))
            values = image_metrics(output.color, target, mask)
            foreground = mask > 0.5
            predicted = output.alpha > 0.5
            union = (foreground | predicted).sum().clamp_min(1)
            values.update(
                {
                    "lpips_crop_max512": lpips_metric(output.color, target, mask),
                    "alpha_iou": float((foreground & predicted).sum() / union),
                    "alpha_inside": float(output.alpha[foreground].mean()),
                    "alpha_outside": float(output.alpha[~foreground].mean()),
                }
            )
            view_name = scene.view_names[index]
            per_view[view_name] = values
            if position == 0 and preview_dir is not None:
                _save_rgb(preview_dir / f"preview_{prefix}.png", output.color)
                _save_comparison(
                    preview_dir / f"comparison_{prefix}.png",
                    target,
                    output.color,
                    output.alpha,
                    mask,
                )
    aggregate_keys = (
        "psnr_fg",
        "psnr_crop",
        "ssim_crop",
        "lpips_crop_max512",
        "alpha_iou",
        "alpha_inside",
        "alpha_outside",
    )
    return {
        **{key: _median([values[key] for values in per_view.values()]) for key in aggregate_keys},
        "aggregation": "median across validation views",
        "per_view": per_view,
    }


def _arm_initialization(
    arm: str,
    *,
    beam: Gaussians3D,
    repaired: Gaussians3D,
    no_covariance: Gaussians3D,
    no_opacity: Gaussians3D,
    means_only: Gaussians3D,
    random: Gaussians3D,
) -> Gaussians3D:
    if arm in {"beam-only", "beam-standard"}:
        return beam.detach()
    if arm == "no-covariance":
        return no_covariance.detach()
    if arm == "no-opacity":
        return no_opacity.detach()
    if arm in {
        "carrier-schedule",
        "carrier-schedule-clone",
        "no-warmup",
        "split-immediately",
        "clone-only",
        "particle",
    }:
        return repaired.detach()
    if arm == "means-only":
        return means_only.detach()
    if arm in {"random-rgb", "random-jpeg-q50"}:
        return random.detach()
    raise ValueError(arm)


def _run_arm(
    arm: str,
    scene: SceneData,
    jpeg_scene: SceneData,
    initialization: Gaussians3D,
    iterations: int,
) -> ArmRun:
    training_scene = jpeg_scene if arm == "random-jpeg-q50" else scene
    if arm == "beam-only":
        return ArmRun(
            final=initialization,
            history={"density_stats": None, "note": "no dense-image optimization"},
            diagnostics={
                "phase_boundaries": [
                    CarrierLineageTracker(initialization).snapshot("beam-only", initialization, 0)
                ]
            },
            training_seconds=0.0,
            peak_vram_gb=0.0,
            peak_vram_reserved_gb=0.0,
            iterations=0,
        )
    if arm in {
        "beam-standard",
        "split-immediately",
        "random-rgb",
        "random-jpeg-q50",
        "means-only",
    }:
        return _run_standard(
            training_scene,
            initialization,
            iterations,
            track_carriers=arm in {"beam-standard", "split-immediately", "means-only"},
        )
    if iterations != ITERATIONS:
        raise ValueError("phased arms require the frozen official iteration count")
    if arm == "carrier-schedule":
        return _run_schedule(
            training_scene,
            initialization,
            warmup=WARMUP_ITERATIONS,
            clone=0,
            higher=HIGHER_SH_ITERATIONS,
            standard=STANDARD_ITERATIONS + CLONE_ITERATIONS,
        )
    if arm in {"carrier-schedule-clone", "no-covariance", "no-opacity"}:
        return _run_schedule(
            training_scene,
            initialization,
            warmup=WARMUP_ITERATIONS,
            clone=CLONE_ITERATIONS,
            higher=HIGHER_SH_ITERATIONS,
            standard=STANDARD_ITERATIONS,
        )
    if arm == "no-warmup":
        return _run_schedule(
            training_scene,
            initialization,
            warmup=0,
            clone=CLONE_ITERATIONS,
            higher=HIGHER_SH_ITERATIONS,
            standard=STANDARD_ITERATIONS + WARMUP_ITERATIONS,
        )
    if arm == "clone-only":
        return _run_schedule(
            training_scene,
            initialization,
            warmup=WARMUP_ITERATIONS,
            clone=CLONE_ITERATIONS + STANDARD_ITERATIONS,
            higher=HIGHER_SH_ITERATIONS,
            standard=0,
        )
    if arm == "particle":
        return _run_schedule(
            training_scene,
            initialization,
            warmup=WARMUP_ITERATIONS,
            clone=CLONE_ITERATIONS + STANDARD_ITERATIONS,
            higher=HIGHER_SH_ITERATIONS,
            standard=0,
            particle=True,
        )
    raise ValueError(arm)


def _storage_for_arm(
    arm: str,
    *,
    compact_bytes: int,
    raw_bytes: int,
    jpeg_bytes: int,
    mask_bytes: int,
) -> dict:
    if arm == "random-jpeg-q50":
        representation = "second-generation JPEG q50 RGB + masks"
        input_bytes = jpeg_bytes + mask_bytes
    elif arm in {"random-rgb"}:
        representation = "original JPEG RGB + masks"
        input_bytes = raw_bytes
    elif arm == "means-only":
        representation = "compact captures for means + full-resolution RGB optimization"
        input_bytes = compact_bytes + raw_bytes
    elif arm == "beam-only":
        representation = "compact 2D Gaussian captures only"
        input_bytes = compact_bytes
    else:
        representation = "compact captures + full-resolution RGB maturation"
        input_bytes = compact_bytes + raw_bytes
    return {
        "representation": representation,
        "input_bytes": input_bytes,
        "compact_bytes": compact_bytes,
        "raw_rgb_and_mask_bytes": raw_bytes,
        "jpeg_q50_bytes": jpeg_bytes,
        "mask_bytes": mask_bytes,
    }


def _write_arm_bundle(
    directory: Path,
    initialization: Gaussians3D,
    run: ArmRun,
    init_metrics: dict,
    final_metrics: dict,
    record: dict,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    initialization.save_ply(directory / "gaussians_init.ply")
    run.final.save_ply(directory / "gaussians.ply")
    shutil.copy2(directory / "gaussians.ply", directory / "gaussians_final.ply")
    metrics = {
        "psnr": final_metrics["psnr_fg"],
        "ssim": final_metrics["ssim_crop"],
        "lpips": final_metrics["lpips_crop_max512"],
        "alpha_iou": final_metrics["alpha_iou"],
        "psnr_crop": final_metrics["psnr_crop"],
        "init_psnr": init_metrics["psnr_fg"],
        "n_gaussians": run.final.n,
        "training_seconds": run.training_seconds,
        "peak_vram_gb": run.peak_vram_gb,
        "details": final_metrics,
    }
    _json(directory / "metrics.json", metrics)
    _json(directory / "training_history.json", run.history)
    _json(
        directory / "gaussians.config.json",
        {
            "arm": record["arm"],
            "description": record["description"],
            "iterations": run.iterations,
            "seed": SEED,
            "native_resolution": record["resolution"],
            "training": asdict(_template(max(1, run.iterations))),
            "diagnostics": run.diagnostics,
        },
    )
    _json(directory / "record.json", record)


def _viewer_manifest(out: Path, records: list[dict]) -> dict:
    return {
        "schema": "rtgs.viewer-comparison.v1",
        "methods": [
            {
                "name": record["arm"],
                "initial": f"{record['arm']}/gaussians_init.ply",
                "final": f"{record['arm']}/gaussians.ply",
            }
            for record in records
        ],
    }


def _write_index(out: Path, summary: dict) -> None:
    records = summary["arms"]
    best_psnr = max(record["final_metrics"]["psnr_fg"] for record in records)
    rows = []
    cards = []
    bars = []
    colors = ("#68d5c4", "#ffca6e", "#9fa8ff", "#ff8f9d", "#80b7ff")
    for index, record in enumerate(records):
        arm = html.escape(record["arm"])
        metrics = record["final_metrics"]
        survival = record.get("carrier_survival_rate")
        survival_text = "—" if survival is None else f"{100 * survival:.1f}%"
        rows.append(
            "<tr>"
            f"<td><span class='dot' style='--c:{colors[index % len(colors)]}'></span>"
            f"<a href='{arm}/record.json'>{arm}</a></td>"
            f"<td>{record['iterations']}</td><td>{record['n_final']:,}</td>"
            f"<td>{metrics['psnr_fg']:.3f}</td><td>{metrics['ssim_crop']:.4f}</td>"
            f"<td>{metrics['lpips_crop_max512']:.4f}</td>"
            f"<td>{metrics['alpha_iou']:.4f}</td><td>{survival_text}</td>"
            f"<td>{record['training_seconds']:.1f}</td>"
            f"<td>{record['peak_vram_gb']:.2f}</td>"
            f"<td>{record['storage']['input_bytes'] / 2**20:.1f}</td></tr>"
        )
        cards.append(
            f"<article class='card'><header><strong>{arm}</strong>"
            f"<span>{metrics['psnr_fg']:.2f} dB</span></header>"
            f"<a href='{arm}/comparison_final.png'><img loading='lazy' "
            f"src='{arm}/comparison_final.png' alt='{arm} comparison'></a>"
            f"<footer><a href='{arm}/gaussians.ply'>PLY</a> · "
            f"<a href='{arm}/metrics.json'>metrics</a> · "
            f"<a href='{arm}/comparison_init.png'>initial</a></footer></article>"
        )
        width = 100 * metrics["psnr_fg"] / best_psnr
        bars.append(
            f"<div class='barrow'><span>{arm}</span><div class='track'>"
            f"<i style='width:{width:.2f}%'></i></div><b>{metrics['psnr_fg']:.2f}</b></div>"
        )
    train_ids = ", ".join(summary["split"]["train3"])
    validation_ids = ", ".join(summary["split"]["validation"])
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Janelle carrier-refinement full-resolution experiment</title>
<style>
:root{{color-scheme:dark;--bg:#080d13;--panel:#111a25;--line:#29384b;--text:#eef5fc;
--muted:#9eacbf;--accent:#68d5c4;--warm:#ffca6e}}*{{box-sizing:border-box}}
body{{margin:0;background:radial-gradient(circle at 12% 0,#203950,var(--bg) 42rem);
color:var(--text);font:15px/1.5 system-ui,sans-serif}}a{{color:var(--accent)}}
.shell{{width:min(1500px,calc(100% - 32px));margin:auto}}.hero{{padding:48px 0 20px}}
.eyebrow{{color:var(--accent);text-transform:uppercase;font-size:12px;font-weight:800;
letter-spacing:.14em}}h1{{font-size:clamp(32px,5vw,58px);line-height:1.03;margin:7px 0}}
.lead,.muted{{color:var(--muted)}}.notice{{padding:14px 16px;background:#201c12;
border:1px solid #675834;border-radius:12px}}nav{{position:sticky;top:0;z-index:3;
display:flex;gap:15px;flex-wrap:wrap;padding:12px 0;background:rgba(8,13,19,.92);
backdrop-filter:blur(12px)}}section{{margin:28px 0}}.panel{{overflow:auto;padding:17px;
border:1px solid var(--line);background:rgba(17,26,37,.94);border-radius:14px}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{
padding:9px 11px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}.dot{{display:inline-block;width:10px;height:10px;
margin-right:8px;border-radius:50%;background:var(--c)}}.grid{{display:grid;
grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}}.card{{
border:1px solid var(--line);
background:var(--panel);border-radius:12px;overflow:hidden}}.card header{{display:flex;
justify-content:space-between;padding:10px 12px;gap:8px}}.card header span,.card footer{{
color:var(--muted);font-size:12px}}.card img{{display:block;width:100%;background:#000}}
.card footer{{padding:8px 12px}}.barrow{{display:grid;grid-template-columns:190px 1fr 55px;
gap:10px;align-items:center;margin:7px 0}}.track{{height:12px;border-radius:8px;background:#253244;
overflow:hidden}}.track i{{display:block;height:100%;
background:linear-gradient(90deg,#55bfb1,#89eadb)}}
.links{{display:flex;flex-wrap:wrap;gap:15px}}footer.page{{padding:25px 0 42px;
border-top:1px solid var(--line);color:var(--muted)}}
</style></head><body><main class="shell"><section class="hero">
<div class="eyebrow">realtime-gs · ADR-002 development evidence</div>
<h1>Carrier refinement at native Janelle resolution</h1>
<p class="lead">All runnable paper-plan arms and ablations on masked frame 00008 at
{summary["resolution"][0]}×{summary["resolution"][1]}. Train: {html.escape(train_ids)}.
Validation: {html.escape(validation_ids)}.</p>
<p class="notice"><strong>Scope:</strong> outcome-exposed, single-scene, single-seed development
evidence. No COLMAP model exists for this capture, so “Original 3DGS” and its compressed-RGB
counterpart are unavailable; random RGB/JPEG controls are shown under their honest labels.</p>
</section><nav><a href="#quality">Quality</a><a href="#visuals">Visuals</a>
<a href="#provenance">Provenance</a><a href="summary.json">summary.json</a>
<a href="comparison_manifest.json">orbit-viewer manifest</a></nav>
<section id="quality"><h2>Held-out masked quality and resources</h2><div class="panel"><table>
<thead><tr><th>Arm</th><th>Steps</th><th>Final N</th><th>FG PSNR</th><th>Crop SSIM</th>
<th>LPIPS↓</th><th>α-IoU</th><th>Carrier survival</th><th>Train s</th>
<th>Peak GiB</th><th>Input MiB</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<p class="muted">Headline values are medians over three held-out cameras. LPIPS uses the masked
foreground crop resized to at most {LPIPS_MAX_SIZE}px. Input MiB counts each arm's declared
representation; compact+RGB arms charge both.</p></div></section>
<section><h2>Foreground PSNR overview</h2><div class="panel">{"".join(bars)}</div></section>
<section id="visuals"><h2>Held-out C0031 comparisons</h2><div class="grid">
{"".join(cards)}</div><p class="muted">Panels: masked target · prediction · 4× absolute error ·
predicted alpha. Click any card for the full saved overview image.</p></section>
<section id="provenance"><h2>Protocol and receipts</h2><div class="panel links">
<a href="../../benchmarks/results/20260727_carrier_refinement_fullres_PREREG.md">
frozen protocol</a><a href="run_manifest.json">run manifest</a>
<a href="source_binding.json">source hashes</a><a href="repair_diagnostics.json">
carrier repair diagnostics</a><a href="viewer_receipt.json">live viewer receipt</a></div></section>
<footer class="page">Generated from saved per-arm artifacts. Quantitative interpretation is
subject to the independent scientist-pass audit.</footer></main></body></html>"""
    (out / "index.html").write_text(page)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--raw-frame", type=Path, default=DEFAULT_RAW_FRAME)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--downscale", type=int, default=1)
    args = parser.parse_args()
    if not args.protocol.is_file():
        raise FileNotFoundError("write the frozen protocol before reading any outcomes")
    if args.iterations <= 0 or args.downscale <= 0:
        raise ValueError("iterations and downscale must be positive")
    if args.iterations != ITERATIONS:
        print("[warning] non-preregistered smoke iteration override", flush=True)
    if args.downscale != 1:
        print("[warning] non-preregistered resolution override", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("the full-resolution experiment requires CUDA gsplat")

    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to mix results in non-empty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    selected_arms = tuple(args.arms)
    started_total = time.perf_counter()
    torch.manual_seed(SEED)

    compact_started = time.perf_counter()
    dataset = CompactDataset.load(COMPACT_PATH, device="cpu")
    compact_load_seconds = time.perf_counter() - compact_started
    split = derive_splits(dataset)
    train_ids = tuple(split["train3"])
    validation_ids = tuple(split["validation"])
    selected_ids = (*train_ids, *validation_ids)
    source_binding = _bind_raw_sources(dataset, args.raw_frame.resolve(), selected_ids)
    _json(out / "source_binding.json", source_binding)
    scene, raw_load_seconds = _load_raw_scene(
        args.raw_frame.resolve(),
        train_ids,
        validation_ids,
        dataset.bounds_hint,
        downscale=args.downscale,
    )
    expected_resolution = (5328 // args.downscale, 4608 // args.downscale)
    actual_resolution = (scene.cameras[0].width, scene.cameras[0].height)
    if actual_resolution != expected_resolution:
        raise RuntimeError(
            f"unexpected selected resolution {actual_resolution}, expected {expected_resolution}"
        )
    jpeg_scene, jpeg_bytes = _jpeg_scene(scene, JPEG_QUALITY)
    inputs = _train_inputs(dataset, train_ids)

    implementation_files = (
        Path(__file__).resolve(),
        ROOT / "src/rtgs/lift/carrier_refinement.py",
        ROOT / "src/rtgs/optim/carrier_schedule.py",
        ROOT / "src/rtgs/optim/density.py",
        ROOT / "src/rtgs/optim/trainer.py",
    )
    manifest = {
        "schema": "rtgs.carrier-refinement-fullres.run-manifest.v1",
        "status": "started",
        "protocol": {
            "path": str(args.protocol.resolve().relative_to(ROOT)),
            "sha256": _sha256(args.protocol),
        },
        "implementation_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in implementation_files
        },
        "arms": list(selected_arms),
        "official_arms": list(ARMS),
        "seed": SEED,
        "iterations": args.iterations,
        "downscale": args.downscale,
        "resolution": list(actual_resolution),
        "train_ids": list(train_ids),
        "validation_ids": list(validation_ids),
        "report_only_sealed": list(REPORT_ONLY_SEALED),
        "environment": _environment(),
    }
    _json(out / "run_manifest.json", manifest)

    beam_result, beam_seconds = _beam(dataset, inputs)
    repair_config = CarrierRepairConfig()
    full_repair = repair_beam_carriers(beam_result, inputs, repair_config)
    no_covariance_repair = repair_beam_carriers(
        beam_result,
        inputs,
        replace(repair_config, repair_covariance=False),
    )
    no_opacity_repair = repair_beam_carriers(
        beam_result,
        inputs,
        replace(repair_config, repair_opacity=False),
    )
    means_only, random = _simple_initializations(
        beam_result.gaussians,
        inputs,
        dataset.bounds_hint,
    )
    repair_diagnostics = {
        "beam_seconds": beam_seconds,
        "beam": beam_result.diagnostics,
        "full": full_repair.diagnostics,
        "no_covariance": no_covariance_repair.diagnostics,
        "no_opacity": no_opacity_repair.diagnostics,
    }
    _json(out / "repair_diagnostics.json", repair_diagnostics)

    device = torch.device("cuda")
    lpips_metric = LPIPSEvaluator(device)
    views_by_id = {view.view_id: view for view in dataset.views}
    compact_bytes = sum(views_by_id[view_id].bytes for view_id in train_ids)
    raw_rgb_bytes = sum(source_binding[view_id]["rgb"]["bytes"] for view_id in train_ids)
    mask_bytes = sum(source_binding[view_id]["mask"]["bytes"] for view_id in train_ids)
    raw_bytes = raw_rgb_bytes + mask_bytes
    records: list[dict] = []
    for arm in selected_arms:
        directory = out / arm
        directory.mkdir(parents=True)
        initialization = _arm_initialization(
            arm,
            beam=beam_result.gaussians,
            repaired=full_repair.gaussians,
            no_covariance=no_covariance_repair.gaussians,
            no_opacity=no_opacity_repair.gaussians,
            means_only=means_only,
            random=random,
        )
        print(f"[{arm}] evaluating initialization", flush=True)
        init_metrics = _evaluate(
            scene,
            initialization.to(device),
            scene.testing_views,
            lpips_metric,
            preview_dir=directory,
            prefix="init",
        )
        print(f"[{arm}] running {ARM_DESCRIPTIONS[arm]}", flush=True)
        run = _run_arm(arm, scene, jpeg_scene, initialization, args.iterations)
        print(f"[{arm}] evaluating final", flush=True)
        final_metrics = _evaluate(
            scene,
            run.final.to(device),
            scene.testing_views,
            lpips_metric,
            preview_dir=directory,
            prefix="final",
        )
        boundaries = run.diagnostics.get("phase_boundaries", [])
        survival = float(boundaries[-1]["carrier_survival_rate"]) if boundaries else None
        storage = _storage_for_arm(
            arm,
            compact_bytes=compact_bytes,
            raw_bytes=raw_bytes,
            jpeg_bytes=jpeg_bytes,
            mask_bytes=mask_bytes,
        )
        preprocessing_seconds = (
            0.0
            if arm.startswith("random")
            else beam_seconds
            + (
                0.0
                if arm in {"beam-only", "beam-standard", "means-only"}
                else float(
                    (
                        no_covariance_repair.diagnostics
                        if arm == "no-covariance"
                        else (
                            no_opacity_repair.diagnostics
                            if arm == "no-opacity"
                            else full_repair.diagnostics
                        )
                    )["total_seconds"]
                )
            )
        )
        record = {
            "arm": arm,
            "description": ARM_DESCRIPTIONS[arm],
            "seed": SEED,
            "iterations": run.iterations,
            "resolution": list(actual_resolution),
            "n_initial": initialization.n,
            "n_final": run.final.n,
            "init_metrics": init_metrics,
            "final_metrics": final_metrics,
            "training_seconds": run.training_seconds,
            "preprocessing_seconds": preprocessing_seconds,
            "end_to_end_seconds": preprocessing_seconds + run.training_seconds,
            "peak_vram_gb": run.peak_vram_gb,
            "peak_vram_reserved_gb": run.peak_vram_reserved_gb,
            "carrier_survival_rate": survival,
            "lineage": run.diagnostics,
            "storage": storage,
        }
        _write_arm_bundle(
            directory,
            initialization,
            run,
            init_metrics,
            final_metrics,
            record,
        )
        records.append(record)
        print(
            f"[{arm}] PSNR={final_metrics['psnr_fg']:.3f} "
            f"SSIM={final_metrics['ssim_crop']:.4f} "
            f"LPIPS={final_metrics['lpips_crop_max512']:.4f} n={run.final.n} "
            f"time={run.training_seconds:.1f}s",
            flush=True,
        )
        del run
        torch.cuda.empty_cache()

    raw_selected_bytes = sum(
        entry[kind]["bytes"] for entry in source_binding.values() for kind in ("rgb", "mask")
    )
    summary = {
        "schema": "rtgs.carrier-refinement-fullres.v1",
        "status": "complete" if set(selected_arms) == set(ARMS) else "partial",
        "evidence_scope": (
            "single-scene single-seed outcome-exposed development evidence; not confirmatory"
        ),
        "protocol": manifest["protocol"],
        "manifest": "run_manifest.json",
        "dataset": str(COMPACT_PATH.relative_to(ROOT)),
        "raw_frame": str(args.raw_frame.resolve()),
        "resolution": list(actual_resolution),
        "downscale": args.downscale,
        "split": split,
        "seed": SEED,
        "n_init": N_INIT,
        "iterations": args.iterations,
        "baseline_availability": {
            "original_3dgs_sfm": {
                "status": "unavailable",
                "reason": "no COLMAP sparse point model or COLMAP executable is present",
            },
            "original_3dgs_compressed_rgb": {
                "status": "unavailable",
                "reason": "the matched original-SfM initialization is unavailable",
            },
            "proxy_controls": ["random-rgb", "random-jpeg-q50"],
        },
        "metrics": {
            "aggregation": "median across C0031, C1000, C1002",
            "psnr": "foreground-weighted masked PSNR at native resolution",
            "ssim": "SSIM on masked foreground crop plus 5% full-image margin",
            "lpips": (
                f"AlexNet LPIPS on the same masked crop, bilinear resized to max {LPIPS_MAX_SIZE}px"
            ),
        },
        "ingestion": {
            "compact_load_seconds_all_26_views": compact_load_seconds,
            "raw_load_decode_undistort_seconds_selected_6_views": raw_load_seconds,
            "raw_selected_bytes": raw_selected_bytes,
            "raw_effective_ingestion_mib_per_second": (
                raw_selected_bytes / 2**20 / raw_load_seconds
            ),
            "compact_train3_bytes": compact_bytes,
            "compact_effective_load_mib_per_second": (
                sum(view.bytes for view in dataset.views) / 2**20 / compact_load_seconds
            ),
            "jpeg_q50_train3_bytes": jpeg_bytes,
        },
        "repair": repair_diagnostics,
        "arms": records,
        "environment": manifest["environment"],
        "total_wall_seconds": time.perf_counter() - started_total,
    }
    _json(out / "summary.json", summary)
    comparison = _viewer_manifest(out, records)
    _json(out / "comparison_manifest.json", comparison)

    representative = out / "carrier-schedule-clone"
    if representative.is_dir():
        for source, target in (
            ("gaussians_init.ply", "gaussians_init.ply"),
            ("gaussians.ply", "gaussians.ply"),
            ("metrics.json", "metrics.json"),
            ("training_history.json", "training_history.json"),
            ("gaussians.config.json", "gaussians.config.json"),
            ("preview_init.png", "preview_init.png"),
            ("preview_final.png", "preview_final.png"),
        ):
            shutil.copy2(representative / source, out / target)
    _write_index(out, summary)
    manifest["status"] = summary["status"]
    manifest["summary_sha256"] = _sha256(out / "summary.json")
    _json(out / "run_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
