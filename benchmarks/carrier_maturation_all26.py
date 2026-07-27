#!/usr/bin/env python3
"""All-26-view ADR-002 carrier-maturation experiment on masked Janelle frame 00008.

This is deliberately an all-fitted-view diagnostic. Every compact teacher and every corresponding
native-resolution masked RGB view participates in reconstruction, optimization, stopping, and
reporting. It therefore measures fit dynamics and mechanism behavior, not generalization.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from rtgs.core.gaussians3d import Gaussians3D, quat_to_rotmat
from rtgs.core.metrics import image_metrics, masked_crop
from rtgs.data.calibrated import load_calibrated_scene
from rtgs.data.compact_views import CompactDataset, file_sha256
from rtgs.data.scene import SceneData
from rtgs.lift.beam_fusion import (
    BeamFusionConfig,
    BeamFusionProgress,
    fuse_gaussian_beams,
)
from rtgs.lift.carrier_refinement import CarrierRepairConfig, repair_beam_carriers
from rtgs.optim.carrier_schedule import CarrierLineageTracker
from rtgs.optim.density import DensityConfig, DensityController
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.base import get_rasterizer
from rtgs.visualize import save_reconstruction_artifacts

ROOT = Path(__file__).resolve().parents[1]
COMPACT_PATH = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d"
DEFAULT_RAW_FRAME = Path("/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008")
DEFAULT_OUT = ROOT / "runs/carrier_maturation_all26_frame00008_20260727"
DEFAULT_PROTOCOL = ROOT / "benchmarks/results/20260727_carrier_maturation_all26_PREREG.md"
HISTORICAL_INITIAL = (
    ROOT / "runs/beam_fusion_full_frame00008_fit_20260721/initial_compact_metrics.json"
)
HISTORICAL_FINAL = (
    ROOT / "runs/beam_fusion_full_frame00008_settle_60000_70000_20260721/compact_metrics.json"
)

SEED = 27_027
N_CARRIERS = 5_000
CLONE_WAVES = 3
CLONE_JITTER_FRACTION = 0.10
EVAL_EVERY = 1_000
CHECKPOINT_EVERY = 5_000
MAX_GAUSSIANS = 100_000
LPIPS_MAX_SIZE = 512
VISUAL_DOWNSCALE = 8
EXPECTED_RESOLUTION = (5_328, 4_608)
PREVIEW_VIEW = "C0031"


@dataclass(frozen=True)
class PlateauPhase:
    max_iterations: int
    min_iterations: int
    patience_evals: int
    min_delta_db: float


WARMUP = PlateauPhase(30_000, 4_000, 6, 0.01)
CLONE_RECOVERY = PlateauPhase(15_000, 3_000, 5, 0.01)
HIGHER_SH = PlateauPhase(15_000, 3_000, 5, 0.01)
STANDARD_GROWTH_ITERATIONS = 30_000
STANDARD_SETTLE = PlateauPhase(40_000, 5_000, 6, 0.005)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _environment() -> dict[str, Any]:
    environment: dict[str, Any] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "git_head": _git("rev-parse", "HEAD"),
        "git_status_porcelain": _git("status", "--porcelain"),
    }
    if torch.cuda.is_available():
        environment.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
        )
    try:
        import gsplat

        environment["gsplat"] = gsplat.__version__
    except ImportError:
        environment["gsplat"] = None
    return environment


def _source_binding(dataset: CompactDataset, raw_frame: Path) -> dict[str, Any]:
    receipt: dict[str, Any] = {}
    for view in dataset.views:
        mask_record = view.source["mask"]
        rgb_record = view.source["rgb"]
        if mask_record is None or rgb_record is None:
            raise RuntimeError(f"{view.view_id} lacks a raw RGB/mask source binding")
        rgb = raw_frame / "rgb" / rgb_record["name"]
        mask = raw_frame / "mask" / mask_record["name"]
        rgb_hash = file_sha256(rgb)
        mask_hash = file_sha256(mask)
        if rgb_hash != rgb_record["sha256"] or mask_hash != mask_record["sha256"]:
            raise RuntimeError(f"raw source digest mismatch for {view.view_id}")
        receipt[view.view_id] = {
            "rgb": {"path": str(rgb), "bytes": rgb.stat().st_size, "sha256": rgb_hash},
            "mask": {"path": str(mask), "bytes": mask.stat().st_size, "sha256": mask_hash},
        }
    return receipt


def _load_raw_scene(
    dataset: CompactDataset,
    raw_frame: Path,
    *,
    downscale: int,
) -> SceneData:
    view_ids = [view.view_id for view in dataset.views]
    scene = load_calibrated_scene(
        raw_frame,
        downscale=downscale,
        test_every=0,
        load_masks=True,
        undistort=True,
        view_ids=view_ids,
    )
    if scene.view_names != view_ids:
        raise RuntimeError("raw loader did not retain the exact compact-view order")
    if scene.masks is None:
        raise RuntimeError("the all-26 experiment requires a mask for every view")
    scene.train_indices = list(range(scene.n_views))
    scene.test_indices = []
    scene.bounds_hint = dataset.bounds_hint
    scene.name = (
        "janelle-frame00008-all26-native"
        if downscale == 1
        else f"janelle-frame00008-all26-downscale{downscale}"
    )
    scene.validate()
    return scene


def _compact_metric_scene(dataset: CompactDataset) -> tuple[SceneData, list[dict[str, Any]]]:
    # Reuse the source-bound deterministic StructSplat replay used by the historical all-view
    # Beam experiment. This scene is evaluation-only; native masked RGB remains the optimizer
    # target throughout the present run.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from benchmarks.full_compact_reconstruction import _materialize_training_scene

    return _materialize_training_scene(
        dataset,
        {
            "fit_indices": list(range(len(dataset.views))),
            "validation_indices": [],
            "device": "cuda",
            "structsplat_renderer": "reference",
            "structsplat_chunk": 4096,
            "fit_mode": "all",
        },
    )


def _beam_config(dataset: CompactDataset) -> BeamFusionConfig:
    return BeamFusionConfig(
        min_views=3,
        transverse_gate_sigma=3.0,
        max_color_distance=0.35,
        color_sigma=0.25,
        fold_in_gate_sigma=3.0,
        nms_voxel_size=float(dataset.bounds_hint[1]) / 100.0,
        init_opacity=0.10,
        source_chunk=256,
        max_components=N_CARRIERS,
        seed_budget_multiplier=4,
    )


def _beam_progress() -> Any:
    last_reported = [-math.inf]

    def callback(record: BeamFusionProgress) -> None:
        if (
            record.stage == "complete"
            or record.elapsed_seconds - last_reported[0] >= 30.0
            or (record.stage == "pair_seeding" and record.completed % 10 == 0)
        ):
            last_reported[0] = record.elapsed_seconds
            print(
                f"[beam:{record.stage}] {record.completed}/{record.total} "
                f"pairs={record.evaluated_ray_pairs:,} gated={record.gated_seeds:,} "
                f"voxels={record.retained_seed_voxels:,} "
                f"elapsed={record.elapsed_seconds:.1f}s",
                flush=True,
            )

    return callback


class LPIPSEvaluator:
    """Frozen AlexNet LPIPS on a bounded masked foreground crop."""

    def __init__(self, device: torch.device):
        try:
            import lpips
        except ImportError as error:  # pragma: no cover - required official environment
            raise RuntimeError("LPIPS is required for the frozen protocol") from error
        self.model = lpips.LPIPS(net="alex", verbose=False).eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        prediction = masked_crop(prediction.clamp(0, 1), mask)
        target = masked_crop(target.clamp(0, 1), mask)
        height, width = prediction.shape[:2]
        scale = min(1.0, LPIPS_MAX_SIZE / max(height, width))
        size = (max(32, round(height * scale)), max(32, round(width * scale)))
        prediction = prediction.permute(2, 0, 1)[None]
        target = target.permute(2, 0, 1)[None]
        if prediction.shape[-2:] != size:
            prediction = F.interpolate(prediction, size=size, mode="bilinear", align_corners=False)
            target = F.interpolate(target, size=size, mode="bilinear", align_corners=False)
        with torch.no_grad():
            return float(self.model(2 * prediction - 1, 2 * target - 1).reshape(()))


def _aggregate(per_view: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = tuple(next(iter(per_view.values())))
    aggregate: dict[str, dict[str, float]] = {}
    for key in keys:
        values = torch.tensor(
            [record[key] for record in per_view.values()],
            dtype=torch.float64,
        )
        aggregate[key] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return aggregate


@torch.no_grad()
def _evaluate_scene(
    scene: SceneData,
    gaussians: Gaussians3D,
    lpips_metric: LPIPSEvaluator,
    *,
    target_kind: str,
) -> dict[str, Any]:
    device = torch.device("cuda")
    model = gaussians.to(device)
    renderer = get_rasterizer(
        "gsplat",
        device=device,
        packed=True,
        antialiased=True,
    )
    if scene.masks is None or scene.view_names is None:
        raise RuntimeError("boundary evaluation requires named masked views")
    per_view: dict[str, dict[str, float]] = {}
    started = time.perf_counter()
    for position, index in enumerate(scene.training_views):
        target = scene.images[index].to(device)
        mask = scene.masks[index].to(device)
        output = renderer.render(
            model,
            scene.cameras[index].to(device),
            background=None,
            sh_degree=model.sh_degree,
        )
        if not all(
            bool(torch.isfinite(getattr(output, field)).all())
            for field in ("color", "alpha", "depth")
        ):
            raise RuntimeError(f"non-finite {target_kind} boundary render at view {index}")
        metrics = image_metrics(output.color, target, mask)
        foreground = mask > 0.5
        predicted = output.alpha > 0.5
        union = (foreground | predicted).sum().clamp_min(1)
        metrics.update(
            {
                "lpips_crop_max512": lpips_metric(output.color, target, mask),
                "alpha_iou": float((foreground & predicted).sum() / union),
                "alpha_inside": float(output.alpha[foreground].mean()),
                "alpha_outside": (
                    float(output.alpha[~foreground].mean()) if bool((~foreground).any()) else 0.0
                ),
            }
        )
        per_view[scene.view_names[index]] = metrics
        print(
            f"[metric:{target_kind}] {position + 1:02d}/{len(scene.training_views):02d} "
            f"{scene.view_names[index]} fg={metrics['psnr_fg']:.3f}dB",
            flush=True,
        )
        del target, mask, output
    result = {
        "target_kind": target_kind,
        "view_count": len(per_view),
        "aggregation": {
            "mean": "arithmetic mean over all 26 fitted views",
            "median": "sample median over all 26 fitted views",
        },
        "aggregate": _aggregate(per_view),
        "per_view": per_view,
        "elapsed_seconds": time.perf_counter() - started,
    }
    del model, renderer
    torch.cuda.empty_cache()
    return result


def _save_rgb(path: Path, tensor: torch.Tensor, *, max_width: int = 1_800) -> None:
    image = tensor.detach().cpu().clamp(0, 1)
    array = (image.numpy() * 255.0).round().astype(np.uint8)
    pil = Image.fromarray(array, mode="RGB")
    if pil.width > max_width:
        pil = pil.resize(
            (max_width, round(pil.height * max_width / pil.width)),
            Image.Resampling.LANCZOS,
        )
    pil.save(path, optimize=True)


@torch.no_grad()
def _save_stage_preview(path: Path, scene: SceneData, gaussians: Gaussians3D) -> None:
    if scene.view_names is None or scene.masks is None:
        raise RuntimeError("preview scene must be named and masked")
    index = scene.view_names.index(PREVIEW_VIEW)
    device = torch.device("cuda")
    model = gaussians.to(device)
    target = scene.images[index].to(device)
    mask = scene.masks[index].to(device)
    output = get_rasterizer("gsplat", device=device, packed=True, antialiased=True).render(
        model,
        scene.cameras[index].to(device),
        background=None,
        sh_degree=model.sh_degree,
    )
    reference = target * mask[..., None]
    error = (output.color - reference).abs().clamp(0, 0.25) * 4.0
    alpha = output.alpha[..., None].expand(-1, -1, 3)
    _save_rgb(path, torch.cat([reference, output.color, error, alpha], dim=1))
    del model, target, mask, output
    torch.cuda.empty_cache()


def _headline(metrics: dict[str, Any], key: str) -> float:
    return float(metrics["aggregate"][key]["mean"])


def _stage_slug(order: int, name: str) -> str:
    return f"{order:02d}_{name.replace('-', '_')}"


def _save_boundary(
    *,
    out: Path,
    order: int,
    name: str,
    label: str,
    global_step: int,
    gaussians: Gaussians3D,
    raw_scene: SceneData,
    compact_scene: SceneData,
    visual_scene: SceneData,
    lpips_metric: LPIPSEvaluator,
    tracker: CarrierLineageTracker,
    stage_records: list[dict[str, Any]],
) -> dict[str, Any]:
    slug = _stage_slug(order, name)
    directory = out / "stages" / slug
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite stage boundary: {directory}")
    directory.mkdir(parents=True)
    model_path = directory / "gaussians.ply"
    gaussians.save_ply(model_path)
    _save_stage_preview(directory / "preview.png", visual_scene, gaussians)
    print(f"[boundary:{name}] evaluating native masked RGB", flush=True)
    raw_metrics = _evaluate_scene(
        raw_scene,
        gaussians,
        lpips_metric,
        target_kind="native_masked_rgb",
    )
    print(f"[boundary:{name}] evaluating compact teacher replay", flush=True)
    compact_metrics = _evaluate_scene(
        compact_scene,
        gaussians,
        lpips_metric,
        target_kind="compact_teacher_replay",
    )
    lineage = tracker.snapshot(name, gaussians, global_step)
    metrics = {
        "schema": "rtgs.carrier-maturation.boundary.v1",
        "order": order,
        "name": name,
        "label": label,
        "global_step": global_step,
        "n_gaussians": gaussians.n,
        "sh_degree": gaussians.sh_degree,
        "model_sha256": _sha256(model_path),
        "native_masked_rgb": raw_metrics,
        "compact_teacher": compact_metrics,
        "lineage": lineage,
    }
    _json(directory / "metrics.json", metrics)
    record = {
        "order": order,
        "name": name,
        "label": label,
        "global_step": global_step,
        "n_gaussians": gaussians.n,
        "sh_degree": gaussians.sh_degree,
        "model": str(model_path.relative_to(out)),
        "model_sha256": metrics["model_sha256"],
        "preview": str((directory / "preview.png").relative_to(out)),
        "metrics": str((directory / "metrics.json").relative_to(out)),
        "native_psnr_fg_mean": _headline(raw_metrics, "psnr_fg"),
        "native_ssim_crop_mean": _headline(raw_metrics, "ssim_crop"),
        "native_lpips_mean": _headline(raw_metrics, "lpips_crop_max512"),
        "native_alpha_iou_mean": _headline(raw_metrics, "alpha_iou"),
        "compact_psnr_fg_mean": _headline(compact_metrics, "psnr_fg"),
        "compact_ssim_crop_mean": _headline(compact_metrics, "ssim_crop"),
        "compact_alpha_iou_mean": _headline(compact_metrics, "alpha_iou"),
        "carrier_survival_rate": lineage["carrier_survival_rate"],
    }
    stage_records.append(record)
    print(
        f"[boundary:{name}] N={gaussians.n:,} "
        f"raw={record['native_psnr_fg_mean']:.3f}dB "
        f"compact={record['compact_psnr_fg_mean']:.3f}dB",
        flush=True,
    )
    return record


def _parameters(gaussians: Gaussians3D) -> dict[str, torch.nn.Parameter]:
    return {
        "means": torch.nn.Parameter(gaussians.means.detach().clone()),
        "quats": torch.nn.Parameter(gaussians.quats.detach().clone()),
        "scales": torch.nn.Parameter(gaussians.log_scales.detach().clone()),
        "opacities": torch.nn.Parameter(
            torch.logit(gaussians.opacity.detach().clamp(1e-6, 1.0 - 1e-6))
        ),
        "sh0": torch.nn.Parameter(gaussians.sh[:, :1].detach().clone()),
        "shN": torch.nn.Parameter(gaussians.sh[:, 1:].detach().clone()),
    }


def _clone_all_tangent_wave(
    gaussians: Gaussians3D,
    *,
    extent: float,
    seed: int,
    global_step: int,
    tracker: CarrierLineageTracker,
) -> tuple[Gaussians3D, dict[str, Any]]:
    parent_count = gaussians.n
    params = _parameters(gaussians)
    optimizers = {
        name: torch.optim.Adam(
            [{"params": [parameter], "lr": 1e-3, "name": name}],
            eps=1e-15,
        )
        for name, parameter in params.items()
    }
    config = DensityConfig(
        start_iter=1,
        stop_iter=1,
        every=1,
        grad_threshold=0.0,
        split_scale_frac=1.0,
        prune_opacity=0.0,
        prune_scale_frac=1_000_000.0,
        max_gaussians=2 * parent_count,
        opacity_reset_every=0,
        clone_only=True,
        clone_all=True,
        clone_jitter_fraction=CLONE_JITTER_FRACTION,
        clone_tangent_only=True,
        clone_child_opacity_scale=1.0,
        protect_first_n=parent_count,
    )
    controller = DensityController(
        config,
        parent_count,
        scene_extent=extent,
        device=gaussians.means.device,
    )
    changed = controller.step(
        1,
        params,
        optimizers,
        generator=torch.Generator(device=gaussians.means.device).manual_seed(seed),
    )
    if controller.last_surgery is None:
        raise RuntimeError("clone-all wave did not produce a density surgery")
    surgery = controller.last_surgery
    tracker.apply_surgery(
        surgery["keep_mask"],
        surgery["parent_rows"],
        global_step,
    )
    output = Gaussians3D(
        means=changed["means"].detach(),
        quats=changed["quats"].detach(),
        log_scales=changed["scales"].detach(),
        opacity=torch.sigmoid(changed["opacities"].detach()),
        sh=torch.cat([changed["sh0"].detach(), changed["shN"].detach()], dim=1),
    )
    if output.n != 2 * parent_count:
        raise RuntimeError(f"clone wave produced {output.n}, expected {2 * parent_count}")
    if not torch.equal(output.means[:parent_count], gaussians.means):
        raise RuntimeError("clone wave changed a parent mean")
    children = output.means[parent_count:]
    delta_world = children - gaussians.means
    rotation = quat_to_rotmat(gaussians.quats)
    delta_local = (rotation.transpose(-1, -2) @ delta_world[..., None])[..., 0]
    shortest = gaussians.scales.argmin(dim=-1, keepdim=True)
    normal_displacement = delta_local.gather(-1, shortest).abs()
    inherited = {
        "quats": torch.equal(output.quats[parent_count:], gaussians.quats),
        "log_scales": torch.equal(output.log_scales[parent_count:], gaussians.log_scales),
        "opacity": torch.allclose(
            output.opacity[parent_count:],
            gaussians.opacity,
            rtol=1e-6,
            atol=1e-7,
        ),
        "sh": torch.equal(output.sh[parent_count:], gaussians.sh),
    }
    if not all(inherited.values()):
        raise RuntimeError("clone child failed an inheritance check")
    receipt = {
        "schema": "rtgs.carrier-maturation.clone-wave.v1",
        "global_step": global_step,
        "seed": seed,
        "config": asdict(config),
        "n_before": parent_count,
        "n_after": output.n,
        "density_stats": controller.stats,
        "shortest_axis_interpretation": "local Gaussian normal",
        "max_abs_local_normal_displacement": float(normal_displacement.max()),
        "rms_world_displacement": float(delta_world.square().sum(dim=-1).mean().sqrt()),
        "inherited_fields": inherited,
        "parents_survive": True,
    }
    return output, receipt


def _base_config(
    *,
    iterations: int,
    seed: int,
    target_sh_degree: int,
    plateau: PlateauPhase | None,
) -> TrainConfig:
    return TrainConfig(
        iterations=iterations,
        rasterizer="gsplat",
        device="cuda",
        densify=False,
        density_strategy="classic",
        eval_every=EVAL_EVERY,
        checkpoint_policy="final",
        target_sh_degree=target_sh_degree,
        sh_degree_interval=1,
        use_masks=True,
        random_background=False,
        packed=True,
        antialiased=True,
        validate_render_finite=True,
        seed=seed,
        opacity_logit_epsilon=1e-6,
        plateau_patience_evals=(None if plateau is None else plateau.patience_evals),
        plateau_min_delta=0.0 if plateau is None else plateau.min_delta_db,
        plateau_min_iterations=0 if plateau is None else plateau.min_iterations,
        stream_scene_from_cpu=True,
        record_train_metrics=True,
    )


def _phase_config(name: str, seed: int) -> TrainConfig:
    if name == "fixed_topology":
        return _base_config(
            iterations=WARMUP.max_iterations,
            seed=seed,
            target_sh_degree=0,
            plateau=WARMUP,
        )
    if name.startswith("clone_recovery_"):
        return _base_config(
            iterations=CLONE_RECOVERY.max_iterations,
            seed=seed,
            target_sh_degree=0,
            plateau=CLONE_RECOVERY,
        )
    if name == "higher_sh":
        return replace(
            _base_config(
                iterations=HIGHER_SH.max_iterations,
                seed=seed,
                target_sh_degree=3,
                plateau=HIGHER_SH,
            ),
            lr_means=0.0,
            lr_quats=0.0,
            lr_scales=0.0,
            lr_opacity=0.0,
        )
    if name == "standard_growth":
        density = DensityConfig(
            start_iter=500,
            stop_iter=15_000,
            every=100,
            grad_threshold=2e-4,
            absgrad=False,
            split_scale_frac=0.01,
            split_factor=1.6,
            prune_opacity=0.005,
            prune_scale_frac=0.1,
            max_gaussians=MAX_GAUSSIANS,
            opacity_reset_every=3_000,
            opacity_reset_value=0.011,
            revised_opacity=True,
        )
        return replace(
            _base_config(
                iterations=STANDARD_GROWTH_ITERATIONS,
                seed=seed,
                target_sh_degree=3,
                plateau=None,
            ),
            densify=True,
            density_strategy="classic",
            density=density,
        )
    if name == "standard_settle":
        return replace(
            _base_config(
                iterations=STANDARD_SETTLE.max_iterations,
                seed=seed,
                target_sh_degree=3,
                plateau=STANDARD_SETTLE,
            ),
            lr_means=1e-7,
            lr_quats=1.5625e-5,
            lr_scales=7.8125e-5,
            lr_opacity=7.8125e-4,
            lr_sh=3.90625e-5,
            lr_sh_rest=1.953125e-6,
            means_lr_final_factor=1.0,
        )
    raise ValueError(name)


def _run_phase(
    *,
    name: str,
    label: str,
    scene: SceneData,
    current: Gaussians3D,
    config: TrainConfig,
    global_start: int,
    out: Path,
    tracker: CarrierLineageTracker,
    phase_records: list[dict[str, Any]],
    trajectory: dict[str, list[dict[str, Any]]],
    checkpoint_records: list[dict[str, Any]],
) -> tuple[Gaussians3D, int, dict[str, Any]]:
    phase_checkpoint_dir = out / "checkpoints" / name
    phase_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint(snapshot: Gaussians3D, local_step: int) -> None:
        if local_step % CHECKPOINT_EVERY != 0:
            return
        global_step = global_start + local_step
        path = phase_checkpoint_dir / f"gaussians_step_{global_step:06d}.ply"
        snapshot.save_ply(path)
        checkpoint_records.append(
            {
                "phase": name,
                "global_step": global_step,
                "n_gaussians": snapshot.n,
                "artifact": str(path.relative_to(out)),
                "sha256": _sha256(path),
            }
        )

    def density_surgery(
        keep_mask: torch.Tensor,
        parent_rows: torch.Tensor,
        local_step: int,
    ) -> None:
        tracker.apply_surgery(keep_mask, parent_rows, global_start + local_step)

    started = time.perf_counter()
    final, history = Trainer(config).train(
        scene,
        current,
        checkpoint_callback=checkpoint,
        density_surgery_callback=density_surgery,
    )
    elapsed = time.perf_counter() - started
    executed = int(history["executed_iterations"])
    global_end = global_start + executed
    for local_index, (loss, terms) in enumerate(
        zip(history["loss"], history["loss_terms"], strict=True),
        start=1,
    ):
        trajectory["updates"].append(
            {
                "global_step": global_start + local_index,
                "phase": name,
                "phase_step": local_index,
                "loss": loss,
                **terms,
            }
        )
    for metric in history.get("train_metrics", []):
        trajectory["evaluations"].append(
            {
                **metric,
                "global_step": global_start + int(metric["step"]),
                "phase": name,
                "phase_step": int(metric["step"]),
            }
        )
    for density in history.get("density_stats") or []:
        trajectory["topology"].append(
            {
                **density,
                "global_step": global_start + int(density["iteration"]),
                "phase": name,
            }
        )
    record = {
        "name": name,
        "label": label,
        "global_start": global_start,
        "global_end": global_end,
        "requested_max_iterations": config.iterations,
        "executed_iterations": executed,
        "stop_reason": history["stop_reason"],
        "plateau": history.get("plateau"),
        "training_seconds": elapsed,
        "n_before": current.n,
        "n_after": final.n,
        "peak_vram_gb": history.get("peak_vram_gb", 0.0),
        "config": asdict(config),
    }
    phase_records.append(record)
    trajectory["phases"].append(
        {
            "name": name,
            "label": label,
            "start": global_start,
            "end": global_end,
            "stop_reason": history["stop_reason"],
        }
    )
    _json(out / "phase_histories" / f"{name}.json", history)
    print(
        f"[phase:{name}] steps={executed:,} stop={history['stop_reason']} "
        f"N={current.n:,}->{final.n:,} elapsed={elapsed:.1f}s",
        flush=True,
    )
    torch.cuda.empty_cache()
    return final, global_end, history


def _historical_reference() -> dict[str, Any]:
    if not HISTORICAL_INITIAL.is_file() or not HISTORICAL_FINAL.is_file():
        return {"available": False}
    initial = json.loads(HISTORICAL_INITIAL.read_text(encoding="utf-8"))
    final = json.loads(HISTORICAL_FINAL.read_text(encoding="utf-8"))
    return {
        "available": True,
        "interpretation": (
            "Historical all-26 fitted-view compact-teacher optimization; contextual comparison "
            "only, not a randomized control or held-out result."
        ),
        "initial": {
            "path": str(HISTORICAL_INITIAL.relative_to(ROOT)),
            "sha256": _sha256(HISTORICAL_INITIAL),
            "n_gaussians": initial["n_gaussians"],
            "compact_train": initial["train"],
        },
        "selected_final": {
            "path": str(HISTORICAL_FINAL.relative_to(ROOT)),
            "sha256": _sha256(HISTORICAL_FINAL),
            "selected_global_step": final["model_selection"]["selected_global_step"],
            "compact_train": final["train"],
        },
    }


def _viewer_manifest(stage_records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "rtgs.viewer-comparison.v1",
        "methods": [
            {
                "name": record["label"],
                "initial": "gaussians_init.ply",
                "final": record["model"],
            }
            for record in stage_records
        ],
    }


def _write_index(out: Path, summary: dict[str, Any]) -> None:
    stages = summary["stages"]
    rows = []
    cards = []
    for record in stages:
        rows.append(
            "<tr>"
            f"<td>{record['order']:02d}</td>"
            f"<td><a href='{html.escape(record['metrics'])}'>"
            f"{html.escape(record['label'])}</a></td>"
            f"<td>{record['global_step']:,}</td>"
            f"<td>{record['n_gaussians']:,}</td>"
            f"<td>{record['sh_degree']}</td>"
            f"<td>{record['native_psnr_fg_mean']:.3f}</td>"
            f"<td>{record['native_ssim_crop_mean']:.5f}</td>"
            f"<td>{record['native_lpips_mean']:.5f}</td>"
            f"<td>{record['native_alpha_iou_mean']:.5f}</td>"
            f"<td>{record['compact_psnr_fg_mean']:.3f}</td>"
            f"<td>{record['compact_ssim_crop_mean']:.5f}</td>"
            f"<td>{100 * record['carrier_survival_rate']:.1f}%</td>"
            "</tr>"
        )
        cards.append(
            "<article class='card'>"
            f"<header><b>{record['order']:02d} · {html.escape(record['label'])}</b>"
            f"<span>{record['native_psnr_fg_mean']:.2f} dB raw · "
            f"{record['compact_psnr_fg_mean']:.2f} dB compact</span></header>"
            f"<a href='{html.escape(record['preview'])}'><img loading='lazy' "
            f"src='{html.escape(record['preview'])}' alt='{html.escape(record['label'])}'></a>"
            f"<footer><a href='{html.escape(record['model'])}'>PLY</a> · "
            f"<a href='{html.escape(record['metrics'])}'>metrics</a> · "
            f"N={record['n_gaussians']:,}</footer></article>"
        )
    phases = "".join(
        "<tr>"
        f"<td>{html.escape(record['label'])}</td>"
        f"<td>{record['global_start']:,}–{record['global_end']:,}</td>"
        f"<td>{record['executed_iterations']:,}</td>"
        f"<td>{html.escape(record['stop_reason'])}</td>"
        f"<td>{record['n_before']:,}→{record['n_after']:,}</td>"
        f"<td>{record['training_seconds']:.1f}</td>"
        "</tr>"
        for record in summary["phases"]
    )
    final = stages[-1]
    historical = summary["historical_reference"]
    historical_text = (
        f"The prior compact-target run started at "
        f"{historical['initial']['compact_train']['psnr_fg']:.3f} dB and selected "
        f"{historical['selected_final']['compact_train']['psnr_fg']:.3f} dB at step "
        f"{historical['selected_final']['selected_global_step']:,}. It optimized compact "
        "teacher replays, while this run optimizes native masked RGB; only the compact column "
        "is target-comparable."
        if historical.get("available")
        else "The historical compact-target receipt was unavailable."
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>All-26 Beam carrier maturation</title>
<style>
:root{{--bg:#071018;--panel:#101d29;--line:#294054;--ink:#edf6fb;--muted:#9db1c1;
--cyan:#5fe0d0;--amber:#ffc96b;--red:#ff8b91}}*{{box-sizing:border-box}}
body{{margin:0;background:radial-gradient(circle at 15% 0,#1d3b50,var(--bg) 44rem);
color:var(--ink);font:14px/1.5 Inter,ui-sans-serif,system-ui,sans-serif}}
a{{color:var(--cyan)}}.shell{{width:min(1580px,calc(100% - 28px));margin:auto}}
.hero{{padding:48px 0 22px}}.eyebrow{{color:var(--cyan);font-size:12px;font-weight:800;
letter-spacing:.14em;text-transform:uppercase}}h1{{font-size:clamp(34px,5vw,62px);
line-height:1.02;margin:7px 0 12px}}h2{{margin-top:34px}}.lead,.muted{{color:var(--muted)}}
.warning{{border:1px solid #755f31;background:#241f14;border-radius:12px;padding:13px 15px}}
nav{{display:flex;gap:16px;flex-wrap:wrap;position:sticky;top:0;z-index:3;padding:12px 0;
background:rgba(7,16,24,.92);backdrop-filter:blur(10px)}}.stats{{display:grid;
grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:20px 0}}
.stat,.panel,.card{{background:rgba(16,29,41,.95);border:1px solid var(--line);
border-radius:14px}}.stat{{padding:14px}}.stat b{{display:block;font-size:25px}}
.stat span{{color:var(--muted)}}.panel{{padding:16px;overflow:auto}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;
white-space:nowrap}}th:nth-child(2),td:nth-child(2),th:first-child,td:first-child{{text-align:left}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:12px}}
.card{{overflow:hidden}}.card header{{display:flex;justify-content:space-between;gap:9px;
padding:10px 12px}}.card header span,.card footer{{color:var(--muted);font-size:12px}}
.card img{{display:block;width:100%;background:#000}}.card footer{{padding:9px 12px}}
.chartgrid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:12px}}
canvas{{width:100%;height:270px;background:#09131c;border-radius:9px}}
.links{{display:flex;gap:15px;flex-wrap:wrap}}footer.page{{color:var(--muted);
border-top:1px solid var(--line);margin-top:36px;padding:24px 0 44px}}
</style></head><body><main class="shell"><section class="hero">
<div class="eyebrow">ADR-002 · full-resolution all-view diagnostic</div>
<h1>Beam carriers → repaired 3DGS</h1>
<p class="lead">Every one of the 26 Janelle cameras participates. Native optimization and raw
metrics are {EXPECTED_RESOLUTION[0]}×{EXPECTED_RESOLUTION[1]}; compact-teacher metrics preserve
comparability with the older Beam run.</p>
<p class="warning"><b>No held-out views.</b> These are fitted-view dynamics and visual-quality
diagnostics, not generalization evidence. Clone selection is deliberately uniform; error-driven
selection is deferred.</p></section>
<nav><a href="#boundaries">Boundaries</a><a href="#curves">Curves</a>
<a href="#visuals">Visuals</a><a href="#receipts">Receipts</a></nav>
<section class="stats"><div class="stat"><b>{final["native_psnr_fg_mean"]:.3f} dB</b>
<span>final native FG PSNR mean</span></div><div class="stat">
<b>{final["compact_psnr_fg_mean"]:.3f} dB</b><span>final compact FG PSNR mean</span></div>
<div class="stat"><b>{final["native_ssim_crop_mean"]:.5f}</b>
<span>final native crop SSIM</span></div><div class="stat">
<b>{final["n_gaussians"]:,}</b><span>final Gaussians</span></div></section>
<section><h2>Why the older result looked better</h2><div class="panel">
{html.escape(historical_text)}</div></section>
<section id="boundaries"><h2>Every requested boundary</h2><div class="panel"><table>
<thead><tr><th>#</th><th>Boundary</th><th>Step</th><th>N</th><th>SH</th>
<th>Raw PSNR</th><th>Raw SSIM</th><th>Raw LPIPS↓</th><th>Raw α-IoU</th>
<th>Compact PSNR</th><th>Compact SSIM</th><th>Carrier survival</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table><p class="muted">All displayed metrics are arithmetic means
over the same 26 fitted cameras; each linked JSON also includes medians, extrema, and per-view
values.</p></div></section>
<section><h2>Optimization phases</h2><div class="panel"><table><thead><tr>
<th>Phase</th><th>Global steps</th><th>Updates</th><th>Stop</th><th>Topology</th><th>Seconds</th>
</tr></thead><tbody>{phases}</tbody></table></div></section>
<section id="curves"><h2>Metrics over time</h2><div class="chartgrid">
<div class="panel"><h3>Total loss per optimizer update</h3><canvas id="loss"></canvas></div>
<div class="panel"><h3>Native fitted-view foreground PSNR</h3><canvas id="psnr"></canvas></div>
<div class="panel"><h3>Native fitted-view crop SSIM</h3><canvas id="ssim"></canvas></div>
<div class="panel"><h3>Gaussian count</h3><canvas id="count"></canvas></div>
</div><p class="muted">Background bands are phases; amber vertical rules are uniform tangent
clone waves. Full per-update terms and bounded evaluation metrics are in
<a href="trajectory.json">trajectory.json</a>.</p></section>
<section id="visuals"><h2>Visual boundary sequence · {PREVIEW_VIEW}</h2>
<p class="muted">Each strip is target · prediction · 4× absolute error · predicted alpha.
Visual strips use 1/{VISUAL_DOWNSCALE} resolution; the table above is native resolution.</p>
<div class="grid">{"".join(cards)}</div></section>
<section id="receipts"><h2>Models, protocol, and receipts</h2><div class="panel links">
<a href="gaussians_init.ply">Beam PLY</a><a href="gaussians.ply">final PLY</a>
<a href="metrics.json">metrics.json</a><a href="training_history.json">training history</a>
<a href="summary.json">summary.json</a><a href="protocol.md">frozen protocol</a>
<a href="comparison_manifest.json">orbit-viewer manifest</a>
<a href="reconstruction_contact_sheet.png">contact sheet</a>
<a href="novel_orbit.gif">novel orbit</a><a href="novel_elevation.gif">novel elevation</a>
</div></section><footer class="page">Generated entirely from the saved artifact bundle. Read the
scientist audit before promoting any quantitative claim.</footer></main>
<script>
const colors={{line:'#5fe0d0',phase:'rgba(67,103,128,.15)',clone:'#ffc96b',
grid:'rgba(157,177,193,.18)',text:'#9db1c1'}};
function compact(points,max=2200){{if(points.length<=max)return points;
const stride=Math.ceil(points.length/max);
return points.filter((_,i)=>i%stride===0||i===points.length-1)}}
function chart(canvas,points,phases,clones,ykey){{
 const dpr=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;
 canvas.width=w*dpr;canvas.height=h*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);
 const pad={{l:55,r:15,t:12,b:30}},pw=w-pad.l-pad.r,ph=h-pad.t-pad.b;
 if(!points.length)return;const xs=points.map(p=>p.global_step),ys=points.map(p=>p[ykey]);
 const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
 const xr=x=>pad.l+(x-xmin)/Math.max(1,xmax-xmin)*pw;
 const yr=y=>pad.t+ph-(y-ymin)/Math.max(1e-12,ymax-ymin)*ph;
 phases.forEach((p,i)=>{{c.fillStyle=i%2?colors.phase:'rgba(67,103,128,.08)';
 c.fillRect(xr(Math.max(xmin,p.start)),pad.t,
 Math.max(0,xr(Math.min(xmax,p.end))-xr(Math.max(xmin,p.start))),ph)}});
 c.strokeStyle=colors.grid;c.fillStyle=colors.text;c.font='11px system-ui';
 for(let i=0;i<5;i++){{const y=pad.t+ph*i/4;c.beginPath();c.moveTo(pad.l,y);c.lineTo(w-pad.r,y);
 c.stroke();const v=ymax-(ymax-ymin)*i/4;c.fillText(v.toFixed(ykey==='ssim_crop'?4:3),4,y+4)}}
 clones.forEach(p=>{{if(p.global_step<xmin||p.global_step>xmax)return;c.strokeStyle=colors.clone;
 c.beginPath();c.moveTo(xr(p.global_step),pad.t);c.lineTo(xr(p.global_step),pad.t+ph);c.stroke()}});
 c.strokeStyle=colors.line;c.lineWidth=1.7;c.beginPath();compact(points).forEach((p,i)=>{{
 const x=xr(p.global_step),y=yr(p[ykey]);if(i)c.lineTo(x,y);else c.moveTo(x,y)}});c.stroke();
 c.fillStyle=colors.text;c.fillText(xmin.toLocaleString(),pad.l,h-8);
 const right=xmax.toLocaleString();c.fillText(right,w-pad.r-c.measureText(right).width,h-8);
}}
fetch('trajectory.json').then(r=>r.json()).then(t=>{{
 chart(document.querySelector('#loss'),t.updates,t.phases,t.clone_events,'loss');
 chart(document.querySelector('#psnr'),t.evaluations,t.phases,t.clone_events,'psnr_fg');
 chart(document.querySelector('#ssim'),t.evaluations,t.phases,t.clone_events,'ssim_crop');
 const counts=t.topology.map(p=>({{global_step:p.global_step,n_after:p.n_after}}));
 t.clone_events.forEach(p=>counts.push({{global_step:p.global_step,n_after:p.n_after}}));
 counts.sort((a,b)=>a.global_step-b.global_step);
 chart(document.querySelector('#count'),counts,t.phases,t.clone_events,'n_after');
}});
</script></body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--raw-frame", type=Path, default=DEFAULT_RAW_FRAME)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.protocol.is_file():
        raise FileNotFoundError("freeze the preregistration before starting the run")
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen full-resolution protocol requires CUDA")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to mix outputs in non-empty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.protocol, out / "protocol.md")
    started_total = time.perf_counter()
    torch.manual_seed(SEED)

    implementation_files = (
        Path(__file__).resolve(),
        ROOT / "src/rtgs/lift/beam_fusion.py",
        ROOT / "src/rtgs/lift/carrier_refinement.py",
        ROOT / "src/rtgs/optim/carrier_schedule.py",
        ROOT / "src/rtgs/optim/density.py",
        ROOT / "src/rtgs/optim/trainer.py",
        ROOT / "src/rtgs/visualize.py",
        ROOT / "benchmarks/full_compact_reconstruction.py",
    )
    manifest: dict[str, Any] = {
        "schema": "rtgs.carrier-maturation-all26.run-manifest.v1",
        "status": "started",
        "protocol": {
            "path": str(args.protocol.resolve().relative_to(ROOT)),
            "sha256": _sha256(args.protocol),
        },
        "implementation_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in implementation_files
        },
        "seed": SEED,
        "argv": sys.argv,
        "environment": _environment(),
    }
    _json(out / "run_manifest.json", manifest)

    print("[input] loading and binding all 26 compact captures", flush=True)
    dataset = CompactDataset.load(COMPACT_PATH, device="cpu")
    if len(dataset.views) != 26:
        raise RuntimeError(f"expected 26 compact views, found {len(dataset.views)}")
    view_ids = [view.view_id for view in dataset.views]
    inputs = dataset.to_reconstruction_inputs()
    source_binding = _source_binding(dataset, args.raw_frame.resolve())
    _json(out / "source_binding.json", source_binding)
    print("[input] loading native masked RGB scene", flush=True)
    raw_scene = _load_raw_scene(dataset, args.raw_frame.resolve(), downscale=1)
    resolution = (raw_scene.cameras[0].width, raw_scene.cameras[0].height)
    if resolution != EXPECTED_RESOLUTION:
        raise RuntimeError(f"native resolution is {resolution}, expected {EXPECTED_RESOLUTION}")
    print("[input] loading 1/8-resolution visual scene", flush=True)
    visual_scene = _load_raw_scene(
        dataset,
        args.raw_frame.resolve(),
        downscale=VISUAL_DOWNSCALE,
    )
    print("[input] replaying compact teachers for target-comparable metrics", flush=True)
    compact_scene, compact_receipts = _compact_metric_scene(dataset)
    _json(out / "compact_target_receipts.json", compact_receipts)

    beam_config = _beam_config(dataset)
    print("[beam] fusing 130,000 compact 2D Gaussians into 5,000 carriers", flush=True)
    beam_started = time.perf_counter()
    beam_result = fuse_gaussian_beams(
        inputs,
        beam_config,
        progress_callback=_beam_progress(),
    )
    beam_seconds = time.perf_counter() - beam_started
    if beam_result.n_components != N_CARRIERS:
        raise RuntimeError(
            f"Beam Fusion returned {beam_result.n_components}, expected {N_CARRIERS}"
        )
    _json(
        out / "beam_receipt.json",
        {
            "config": asdict(beam_config),
            "seconds": beam_seconds,
            "diagnostics": beam_result.diagnostics,
            "n_input_components": sum(inputs.n_opt_2d),
            "n_carriers": beam_result.n_components,
        },
    )

    repair_config = CarrierRepairConfig()
    print("[repair] covariance -> opacity -> robust SH0", flush=True)
    repair_result = repair_beam_carriers(beam_result, inputs, repair_config)
    _json(
        out / "repair_diagnostics.json",
        {
            "config": asdict(repair_config),
            "diagnostics": repair_result.diagnostics,
        },
    )
    tracker = CarrierLineageTracker(repair_result.beam)
    lpips_metric = LPIPSEvaluator(torch.device("cuda"))
    stages: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    clone_receipts: list[dict[str, Any]] = []
    trajectory: dict[str, list[dict[str, Any]]] = {
        "updates": [],
        "evaluations": [],
        "topology": [],
        "phases": [],
        "clone_events": [],
    }
    stage_arguments = {
        "out": out,
        "raw_scene": raw_scene,
        "compact_scene": compact_scene,
        "visual_scene": visual_scene,
        "lpips_metric": lpips_metric,
        "tracker": tracker,
        "stage_records": stages,
    }
    for order, (name, label, snapshot) in enumerate(
        (
            ("beam", "0 · Beam Fusion", repair_result.beam),
            ("covariance_repair", "1 · covariance repaired", repair_result.covariance),
            ("opacity_repair", "2 · opacity repaired", repair_result.opacity),
            ("sh0_repair", "3 · robust SH0/color repaired", repair_result.appearance),
        )
    ):
        _save_boundary(
            order=order,
            name=name,
            label=label,
            global_step=0,
            gaussians=snapshot,
            **stage_arguments,
        )

    current = repair_result.gaussians.to("cuda")
    global_step = 0
    current, global_step, _ = _run_phase(
        name="fixed_topology",
        label="4 · fixed topology to train plateau",
        scene=raw_scene,
        current=current,
        config=_phase_config("fixed_topology", SEED + 10),
        global_start=global_step,
        out=out,
        tracker=tracker,
        phase_records=phases,
        trajectory=trajectory,
        checkpoint_records=checkpoints,
    )
    _save_boundary(
        order=4,
        name="fixed_topology_converged",
        label="4 · means/covariance/opacity/SH0 converged",
        global_step=global_step,
        gaussians=current,
        **stage_arguments,
    )

    order = 5
    for wave in range(1, CLONE_WAVES + 1):
        current, clone_receipt = _clone_all_tangent_wave(
            current,
            extent=float(dataset.bounds_hint[1]),
            seed=SEED + 100 + wave,
            global_step=global_step,
            tracker=tracker,
        )
        clone_receipt["wave"] = wave
        clone_receipts.append(clone_receipt)
        trajectory["clone_events"].append(
            {
                "wave": wave,
                "global_step": global_step,
                "n_before": clone_receipt["n_before"],
                "n_after": clone_receipt["n_after"],
            }
        )
        trajectory["topology"].append(
            {
                "phase": f"clone_wave_{wave}",
                "global_step": global_step,
                "iteration": 0,
                "n_before": clone_receipt["n_before"],
                "n_after": clone_receipt["n_after"],
                "cloned": clone_receipt["n_before"],
                "split": 0,
                "pruned": 0,
            }
        )
        _save_boundary(
            order=order,
            name=f"clone_wave_{wave}",
            label=f"5.{wave}a · tangent clone wave {wave}",
            global_step=global_step,
            gaussians=current,
            **stage_arguments,
        )
        order += 1
        phase_name = f"clone_recovery_{wave}"
        current, global_step, _ = _run_phase(
            name=phase_name,
            label=f"5.{wave}b · clone wave {wave} recovery to train plateau",
            scene=raw_scene,
            current=current,
            config=_phase_config(phase_name, SEED + 20 + wave),
            global_start=global_step,
            out=out,
            tracker=tracker,
            phase_records=phases,
            trajectory=trajectory,
            checkpoint_records=checkpoints,
        )
        _save_boundary(
            order=order,
            name=f"clone_wave_{wave}_converged",
            label=f"5.{wave}b · clone wave {wave} converged",
            global_step=global_step,
            gaussians=current,
            **stage_arguments,
        )
        order += 1

    _json(out / "clone_receipts.json", clone_receipts)
    current = current.with_sh_degree(3)
    current, global_step, _ = _run_phase(
        name="higher_sh",
        label="6 · higher SH appearance to train plateau",
        scene=raw_scene,
        current=current,
        config=_phase_config("higher_sh", SEED + 30),
        global_start=global_step,
        out=out,
        tracker=tracker,
        phase_records=phases,
        trajectory=trajectory,
        checkpoint_records=checkpoints,
    )
    _save_boundary(
        order=order,
        name="higher_sh_converged",
        label="6 · higher SH appearance converged",
        global_step=global_step,
        gaussians=current,
        **stage_arguments,
    )
    order += 1

    current, global_step, _ = _run_phase(
        name="standard_growth",
        label="7a · standard 3DGS growth and recovery",
        scene=raw_scene,
        current=current,
        config=_phase_config("standard_growth", SEED + 40),
        global_start=global_step,
        out=out,
        tracker=tracker,
        phase_records=phases,
        trajectory=trajectory,
        checkpoint_records=checkpoints,
    )
    _save_boundary(
        order=order,
        name="standard_growth_complete",
        label="7a · standard 3DGS growth complete",
        global_step=global_step,
        gaussians=current,
        **stage_arguments,
    )
    order += 1

    current, global_step, _ = _run_phase(
        name="standard_settle",
        label="7b · standard fixed-topology settle to train plateau",
        scene=raw_scene,
        current=current,
        config=_phase_config("standard_settle", SEED + 41),
        global_start=global_step,
        out=out,
        tracker=tracker,
        phase_records=phases,
        trajectory=trajectory,
        checkpoint_records=checkpoints,
    )
    _save_boundary(
        order=order,
        name="standard_converged",
        label="7b · final standard 3DGS convergence",
        global_step=global_step,
        gaussians=current,
        **stage_arguments,
    )

    first_model = out / stages[0]["model"]
    final_model = out / stages[-1]["model"]
    shutil.copy2(first_model, out / "gaussians_init.ply")
    shutil.copy2(final_model, out / "gaussians.ply")
    final_metrics = json.loads((out / stages[-1]["metrics"]).read_text(encoding="utf-8"))
    metrics = {
        "metrics": {
            "native_psnr_fg_mean": stages[-1]["native_psnr_fg_mean"],
            "native_ssim_crop_mean": stages[-1]["native_ssim_crop_mean"],
            "native_lpips_mean": stages[-1]["native_lpips_mean"],
            "native_alpha_iou_mean": stages[-1]["native_alpha_iou_mean"],
            "compact_psnr_fg_mean": stages[-1]["compact_psnr_fg_mean"],
            "compact_ssim_crop_mean": stages[-1]["compact_ssim_crop_mean"],
            "compact_alpha_iou_mean": stages[-1]["compact_alpha_iou_mean"],
            "n_gaussians": stages[-1]["n_gaussians"],
            "global_optimizer_updates": global_step,
        },
        "details": final_metrics,
        "scope": "all 26 views fitted; no held-out cameras",
    }
    _json(out / "metrics.json", metrics)
    training_history = {
        "schema": "rtgs.carrier-maturation-all26.history.v1",
        "phases": phases,
        "phase_history_files": [
            str(path.relative_to(out)) for path in sorted((out / "phase_histories").glob("*.json"))
        ],
        "checkpoints": checkpoints,
        "clone_receipts": clone_receipts,
        "total_optimizer_updates": global_step,
    }
    _json(out / "training_history.json", training_history)
    _json(out / "trajectory.json", trajectory)
    config_receipt = {
        "schema": "rtgs.carrier-maturation-all26.config.v1",
        "seed": SEED,
        "dataset": str(COMPACT_PATH.relative_to(ROOT)),
        "raw_frame": str(args.raw_frame.resolve()),
        "view_ids": view_ids,
        "view_count": len(view_ids),
        "native_resolution": list(EXPECTED_RESOLUTION),
        "fit_scope": "all views train/selection/reporting; no held-out views",
        "beam": asdict(beam_config),
        "repair": asdict(repair_config),
        "clone_waves": {
            "count": CLONE_WAVES,
            "count_schedule": [N_CARRIERS, 10_000, 20_000, 40_000],
            "selection": "every current row",
            "jitter_fraction": CLONE_JITTER_FRACTION,
            "jitter_axes": "two axes orthogonal to shortest covariance axis",
            "child_opacity_scale": 1.0,
            "parents_survive": True,
        },
        "phase_configs": {record["name"]: record["config"] for record in phases},
    }
    _json(out / "gaussians.config.json", config_receipt)
    _json(out / "comparison_manifest.json", _viewer_manifest(stages))

    print("[preview] writing calibrated and novel-view artifacts at 1/8 resolution", flush=True)
    save_reconstruction_artifacts(
        visual_scene,
        repair_result.beam.to("cuda"),
        current,
        out,
        rasterizer="gsplat",
        packed=True,
        antialiased=True,
        max_comparisons=6,
        max_animation_frames=24,
    )
    historical = _historical_reference()
    summary = {
        "schema": "rtgs.carrier-maturation-all26.summary.v1",
        "status": "complete",
        "scope": {
            "view_count": 26,
            "view_ids": view_ids,
            "fit_views": view_ids,
            "selection_views": view_ids,
            "reporting_views": view_ids,
            "held_out_views": [],
            "interpretation": (
                "all-fitted-view mechanism and reconstruction diagnostic; no generalization claim"
            ),
        },
        "native_resolution": list(EXPECTED_RESOLUTION),
        "visual_downscale": VISUAL_DOWNSCALE,
        "stages": stages,
        "phases": phases,
        "clone_receipts": clone_receipts,
        "checkpoints": checkpoints,
        "historical_reference": historical,
        "final": metrics["metrics"],
        "total_seconds": time.perf_counter() - started_total,
        "protocol": manifest["protocol"],
    }
    _json(out / "summary.json", summary)
    _write_index(out, summary)
    manifest.update(
        {
            "status": "complete",
            "summary_sha256": _sha256(out / "summary.json"),
            "metrics_sha256": _sha256(out / "metrics.json"),
            "training_history_sha256": _sha256(out / "training_history.json"),
            "trajectory_sha256": _sha256(out / "trajectory.json"),
            "index_sha256": _sha256(out / "index.html"),
            "gaussians_init_sha256": _sha256(out / "gaussians_init.ply"),
            "gaussians_final_sha256": _sha256(out / "gaussians.ply"),
            "total_seconds": summary["total_seconds"],
        }
    )
    _json(out / "run_manifest.json", manifest)
    print(
        f"[complete] {out} raw={stages[-1]['native_psnr_fg_mean']:.3f}dB "
        f"compact={stages[-1]['compact_psnr_fg_mean']:.3f}dB "
        f"N={stages[-1]['n_gaussians']:,} steps={global_step:,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
