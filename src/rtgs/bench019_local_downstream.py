"""Task-owned BENCH019 fixed-field worker and separate held-out reporting seam.

The coordinator owns prospective approval, raw/generated seals, fresh processes, NVML,
and the report. Importing this module neither accesses data nor initializes CUDA. Fitting
loads only the frozen training cameras; held-out images are opened only by evaluate_cell.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.calibrated import load_calibrated_scene
from rtgs.data.compact_views import CompactDataset, file_sha256
from rtgs.data.scene import SceneData
from rtgs.lift.field_sweep import FieldSweepConfig, FieldSweepInitializer
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.base import get_rasterizer

REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORY_STEPS = (0, 100, 250, 500, 750, 1000)
FAMILY_BLEND = {
    "native_additive": "additive",
    "structsplat_normalized": "normalized",
    "structsplat_contained": "normalized",
}


def resolved_configs(seed: int = 0) -> dict:
    """Full dataclass snapshots; seed zero is the task's template, bound per cell."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    sweep = FieldSweepConfig(n_init_3d=256, seed=seed)
    trainer = TrainConfig(
        iterations=1000,
        rasterizer="gsplat",
        device="cuda:0",
        densify=False,
        target_sh_degree=0,
        sh_degree_interval=1000,
        schedule_iterations=1000,
        opacity_reg=0.0,
        scale_reg=0.0,
        checkpoint_policy="final",
        stream_scene_from_cpu=True,
        internal_checkpoint_evaluation=False,
        reset_cuda_peak_stats=False,
        seed=seed,
    )
    return {"field_sweep": asdict(sweep), "trainer": asdict(trainer)}


def _json_write(path: Path, payload: object) -> None:
    # Exclusive creation makes accidental replays into an existing cell fail closed.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _descriptor(path: Path, root: Path) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _diagnostic_json(value: object) -> object:
    """Preserve an unsupported sweep's infinite cost explicitly, never as a metric."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_diagnostic": str(value)}
    if isinstance(value, dict):
        return {key: _diagnostic_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_json(item) for item in value]
    return value


def _validate_task(task: Mapping, frame_id: str, family_id: str, seed: int) -> None:
    if family_id not in FAMILY_BLEND or family_id not in {
        item["id"] for item in task["comparators"]
    }:
        raise ValueError("family is outside the frozen matrix")
    if seed not in [*task["seeds"], 19999] or type(seed) is not int:
        raise ValueError("seed is outside measured/warmup matrix")
    split = task["splits"][frame_id]
    train, heldout = split["train"], split["heldout"]
    if (
        len(train) != 8
        or len(heldout) != 3
        or len(set(train)) != 8
        or len(set(heldout)) != 3
        or set(train) & set(heldout)
    ):
        raise ValueError("expected disjoint eight-training/three-held-out split")
    production = task["production"]
    if (
        production["downscale"] != 8
        or production["undistort"] is not True
        or production["count"] != 512
    ):
        raise ValueError("production geometry/count differs from frozen worker")
    downstream = task["downstream_config"]
    if downstream["resolved_dataclasses"] != resolved_configs():
        raise ValueError("full downstream dataclass template differs from worker")
    if downstream["history_steps"] != list(HISTORY_STEPS):
        raise ValueError("checkpoint schedule differs from frozen worker")
    if downstream["bounds_hint"] is not None:
        raise ValueError("field sweep requires camera-only bounds")


def _frame(task: Mapping, frame_id: str) -> Mapping:
    matches = [item for item in task["datasets"] if item["id"] == frame_id]
    if len(matches) != 1:
        raise ValueError("frame must identify exactly one frozen dataset")
    return matches[0]


def _load_scene(task: Mapping, frame_id: str, *, split: str) -> SceneData:
    if split not in {"train", "heldout"}:
        raise ValueError("unknown split")
    dataset = _frame(task, frame_id)
    names = task["splits"][frame_id][split]
    scene = load_calibrated_scene(
        REPO_ROOT / dataset["frame_path"],
        calibration_path=REPO_ROOT / dataset["calibration"],
        view_ids=names,
        downscale=8,
        undistort=True,
        test_every=0,
        load_masks=True,
    )
    if scene.view_names != names or scene.testing_views:
        raise ValueError("calibrated loader returned a different camera split")
    if scene.masks is None or scene.training_views != list(range(len(names))):
        raise ValueError("calibrated scene lacks complete masks or local indices")
    for image, mask in zip(scene.images, scene.masks, strict=True):
        if (
            not torch.isfinite(image).all()
            or not torch.isfinite(mask).all()
            or bool(((mask < 0) | (mask > 1)).any())
            or not bool((mask >= 0.5).any())
            or not bool((mask < 0.5).any())
        ):
            raise ValueError("scene has nonfinite values, invalid masks, or empty scoring regions")
    return scene


def _camera_record(camera) -> dict:
    return {
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "width": camera.width,
        "height": camera.height,
        "R": camera.R.cpu().tolist(),
        "t": camera.t.cpu().tolist(),
    }


def load_training_fields(
    task: Mapping, frame_id: str, family_id: str, input_dir: str | Path
) -> tuple[CompactDataset, dict]:
    """Cold-load exact teachers without decoding alpha or opening excluded archives."""
    names = task["splits"][frame_id]["train"]
    dataset = CompactDataset.load(input_dir, device="cpu", load_alpha=False, view_ids=names)
    manifest_path = dataset.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if [view["view_id"] for view in manifest["views"]] != names:
        raise ValueError("generated manifest must contain exactly the training split")
    if dataset.bounds_hint is not None:
        raise ValueError("generated fields must carry no mask-derived bounds")
    for view in dataset.views:
        field = view.observation
        if field.n != 512 or field.n_init != 512:
            raise ValueError("generated field count must be exactly 512")
        if field.blend_mode != FAMILY_BLEND[family_id]:
            raise ValueError("generated field blend semantics mismatch")
    geometry = [_camera_record(view.camera) for view in dataset.views]
    receipt = {
        "manifest": {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
            "bytes": manifest_path.stat().st_size,
        },
        "semantic_digest": manifest["semantic_digest"],
        "field_bytes": manifest_path.stat().st_size + sum(view.bytes for view in dataset.views),
        "camera_geometry_sha256": _digest(geometry),
        "view_ids": list(names),
        "rows_per_view": [view.observation.n for view in dataset.views],
        "blend_modes": [view.observation.blend_mode for view in dataset.views],
        "alpha_decoded": False,
        "bounds_hint": None,
    }
    return dataset, receipt


def _cuda_start() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("BENCH019 downstream requires CUDA; no fallback is permitted")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_model(model: Gaussians3D) -> None:
    if model.n != 256 or model.sh_degree != 0:
        raise ValueError("successful downstream models require exactly 256 SH0 rows")
    for name in ("means", "quats", "log_scales", "opacity", "sh"):
        if not bool(torch.isfinite(getattr(model, name)).all()):
            raise ValueError(f"model contains nonfinite {name}")


def _save_model(model: Gaussians3D, directory: Path, stem: str) -> dict:
    _validate_model(model)
    npz, ply = directory / f"{stem}.npz", directory / f"{stem}.ply"
    model.save_npz(npz)
    model.save_ply(ply)
    cold = Gaussians3D.load_npz(npz)
    for name in ("means", "quats", "log_scales", "opacity", "sh"):
        if not torch.equal(getattr(cold, name), getattr(model, name).detach().cpu()):
            raise ValueError(f"saved model cold replay differs in {name}")
    return {"npz": _descriptor(npz, directory), "ply": _descriptor(ply, directory)}


def run_cell(
    task: Mapping,
    frame_id: str,
    family_id: str,
    input_dir: str | Path,
    seed: int,
    output_dir: str | Path,
) -> dict:
    """Run one fresh training-only cell. The coordinator catches and receipts failures.

    output_dir must not exist. Call in a fresh process after both prospective gates;
    receipt.json and history.json are immutable JSON. Reporting is a separate invocation.
    """
    _validate_task(task, frame_id, family_id, seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(task["environment_policy"]["cpu_threads"])
    device = _cuda_start()
    started = time.perf_counter()
    fields, field_receipt = load_training_fields(task, frame_id, family_id, input_dir)
    load_seconds = time.perf_counter() - started
    configurations = resolved_configs(seed)
    sweep = FieldSweepConfig(**configurations["field_sweep"])
    # Instantiate nested dataclasses from defaults, then check their complete frozen snapshot.
    train = TrainConfig(
        **{key: value for key, value in configurations["trainer"].items() if key != "density"}
    )
    if asdict(train) != configurations["trainer"]:
        raise ValueError("trainer nested configuration did not resolve exactly")
    _json_write(
        output / "config.json",
        {
            "schema": "rtgs.bench019.local.downstream_config.v1",
            "task_id": task["task_id"],
            "frame_id": frame_id,
            "family_id": family_id,
            "seed": seed,
            "resolved_dataclasses": configurations,
            "history_steps": list(HISTORY_STEPS),
            "split": task["splits"][frame_id],
            "fields": field_receipt,
        },
    )
    lift_started = time.perf_counter()
    initial = FieldSweepInitializer(sweep).initialize(fields.to_reconstruction_inputs())
    _synchronize(device)
    lift_seconds = time.perf_counter() - lift_started
    if initial.diagnostics["bounds_source"] != "camera_axis_fallback":
        raise ValueError("initializer did not use camera-only bounds")
    _validate_model(initial.gaussians)
    diagnostics = dict(initial.diagnostics)
    supported = diagnostics["supported_track_count"]
    diagnostics["unsupported_midpoint_count"] = 256 - supported
    diagnostics["unsupported_midpoint_fraction"] = (256 - supported) / 256
    diagnostics["fallback_fraction_above_25_percent"] = (256 - supported) / 256 > 0.25
    _json_write(output / "initializer.json", _diagnostic_json(diagnostics))
    np.savez_compressed(
        output / "lineage.npz",
        **{
            name: getattr(initial.lineage, name).cpu().numpy()
            for name in ("source_view_indices", "source_component_indices", "source_xy")
        },
        depths=initial.depths.cpu().numpy(),
        depth_sigmas=initial.depth_sigmas.cpu().numpy(),
        ray_sigmas=initial.ray_sigmas.cpu().numpy(),
        scores=initial.scores.cpu().numpy(),
    )
    raw_initial = _save_model(initial.gaussians, output, "gaussians_lift")
    rgb_started = time.perf_counter()
    scene = _load_scene(task, frame_id, split="train")
    rgb_load_seconds = time.perf_counter() - rgb_started
    if [_camera_record(camera) for camera in scene.cameras] != [
        _camera_record(view.camera) for view in fields.views
    ]:
        raise ValueError("RGB training cameras differ from frozen field cameras")
    checkpoints = []

    def observe(model: Gaussians3D, step: int) -> None:
        if step not in HISTORY_STEPS:
            return
        artifacts = _save_model(model, output, f"checkpoint_{step:04d}")
        checkpoints.append({"step": step, "n_gaussians": model.n, "artifacts": artifacts})

    refine_started = time.perf_counter()
    final, native_history = Trainer(train).train(
        scene,
        initial.gaussians,
        checkpoint_callback=observe,
        initialization_callback=lambda model: observe(model, 0),
    )
    _synchronize(device)
    refine_seconds = time.perf_counter() - refine_started
    if (
        native_history["executed_iterations"] != 1000
        or native_history["stop_reason"] != "max_iterations"
        or native_history["psnr"]
        or [item["step"] for item in checkpoints] != list(HISTORY_STEPS)
    ):
        raise ValueError("trainer violated fixed horizon/checkpoints/no-held-out-evaluation")
    final_artifacts = _save_model(final, output, "gaussians")
    _synchronize(device)
    wall_seconds = time.perf_counter() - started
    history = {
        "schema": "rtgs.bench019.local.downstream_history.v1",
        "step_unit": "optimizer updates",
        "loss_unit": "dimensionless",
        "elapsed_unit": "seconds",
        "count_unit": "gaussians",
        "heldout_evaluation": False,
        "checkpoints": checkpoints,
        "native_trainer": native_history,
        "sampled_training_view_ids": [
            scene.view_names[index] for index in native_history["sampled_train_views"]
        ],
    }
    _json_write(output / "history.json", history)
    metrics = {
        "field_bytes": field_receipt["field_bytes"],
        "field_load_seconds": load_seconds,
        "rgb_load_seconds": rgb_load_seconds,
        "lift_seconds": lift_seconds,
        "refine_seconds": refine_seconds,
        "wall_seconds": wall_seconds,
        "final_gaussians": final.n,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
    }
    receipt = {
        "schema": "rtgs.bench019.local.downstream_receipt.v1",
        "status": "ok",
        "task_id": task["task_id"],
        "frame_id": frame_id,
        "family_id": family_id,
        "seed": seed,
        "initializer": "field_sweep",
        "metrics": metrics,
        "fields": field_receipt,
        "split_sha256": _digest(task["splits"][frame_id]),
        "bounds": {
            key: diagnostics[key]
            for key in (
                "bounds_source",
                "bounds_center",
                "bounds_extent",
                "search_aabb_lower",
                "search_aabb_upper",
            )
        },
        "unsupported_midpoint_count": 256 - supported,
        "artifacts": {
            "lift": raw_initial,
            "initial": checkpoints[0]["artifacts"],
            "final": final_artifacts,
            "config": _descriptor(output / "config.json", output),
            "history": _descriptor(output / "history.json", output),
        },
        "resource_scope": "generated field load through final model save; reporting excluded",
        "refine_clock": "Trainer call including fixed checkpoint serialization",
        "torch_peak_reset": "once before field loading; Trainer reset disabled",
    }
    _json_write(output / "receipt.json", receipt)
    return receipt


def foreground_metrics(
    color: torch.Tensor, alpha: torch.Tensor, image: torch.Tensor, mask: torch.Tensor
) -> dict:
    """Frozen full-canvas metric formulas, including auditable raw sums and counts."""
    if (
        color.shape != image.shape
        or color.ndim != 3
        or color.shape[-1] != 3
        or alpha.shape != color.shape[:2]
        or mask.shape != alpha.shape
    ):
        raise ValueError("render/image/mask shapes do not match")
    if any(not bool(torch.isfinite(value).all()) for value in (color, alpha, image, mask)):
        raise ValueError("nonfinite rendered or target values")
    if bool(((mask < 0) | (mask > 1)).any()) or bool(((alpha < 0) | (alpha > 1)).any()):
        raise ValueError("alpha/mask must be probabilities in [0,1]")
    foreground = mask >= 0.5
    exterior = ~foreground
    pixels, outside_pixels = int(foreground.sum()), int(exterior.sum())
    if not pixels or not outside_pixels:
        raise ValueError("foreground and exterior scoring regions must both be nonempty")
    prediction = color.to(torch.float64).clamp(0, 1)
    target = (image.to(torch.float64) * mask.to(torch.float64)[..., None]).clamp(0, 1)
    squared_sum = float((prediction - target).square()[foreground].sum())
    values = pixels * 3
    mse = squared_sum / values
    binary_alpha = alpha >= 0.5
    intersection = int((binary_alpha & foreground).sum())
    union = int((binary_alpha | foreground).sum())
    outside_sum = float(alpha.to(torch.float64)[exterior].sum())
    return {
        "foreground_psnr": -10.0 * math.log10(max(mse, 1e-12)),
        "alpha_iou": intersection / union,
        "exterior_leakage": outside_sum / outside_pixels,
        "raw": {
            "foreground_squared_error_sum": squared_sum,
            "foreground_rgb_value_count": values,
            "foreground_pixel_count": pixels,
            "alpha_intersection_pixels": intersection,
            "alpha_union_pixels": union,
            "exterior_alpha_sum": outside_sum,
            "exterior_pixel_count": outside_pixels,
        },
    }


def _save_preview(path: Path, image: torch.Tensor) -> None:
    from PIL import Image

    array = image.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
    Image.fromarray(array).save(path)


def evaluate_cell(
    task: Mapping,
    frame_id: str,
    family_id: str,
    seed: int,
    model_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Render one saved final model in a separate, reporting-only fresh process.

    No model selection occurs here. The coordinator invokes this only after the choices
    and all input fields are frozen, and keeps these results out of fitting histories.
    """
    _validate_task(task, frame_id, family_id, seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(task["environment_policy"]["cpu_threads"])
    device = _cuda_start()
    started = time.perf_counter()
    model = Gaussians3D.load_npz(model_path)
    _validate_model(model)
    model = model.to(device)
    scene = _load_scene(task, frame_id, split="heldout")
    config = resolved_configs(seed)["trainer"]
    renderer = get_rasterizer(
        "gsplat",
        device=device,
        packed=config["packed"],
        antialiased=config["antialiased"],
        sh_color_activation=config["sh_color_activation"],
        sh_smu1_mu=config["sh_smu1_mu"],
        kernel_support_mode=config["kernel_support_mode"],
        visibility_margin_sigma=config["visibility_margin_sigma"],
    )
    rows = []
    with torch.no_grad():
        for name, camera, image, mask in zip(
            scene.view_names, scene.cameras, scene.images, scene.masks, strict=True
        ):
            rendered = renderer.render(model, camera.to(device), sh_degree=0)
            color, alpha = rendered.color.cpu(), rendered.alpha.cpu()
            metrics = foreground_metrics(color, alpha, image, mask)
            target = (image * mask[..., None]).clamp(0, 1)
            preview_paths = {}
            for kind, value in (
                ("target", target),
                ("reconstruction", color),
                ("error", (color.clamp(0, 1) - target).abs()),
                ("alpha", alpha),
            ):
                path = output / f"{name}_{kind}.png"
                _save_preview(path, value)
                preview_paths[kind] = _descriptor(path, output)
            rows.append({"view_id": name, **metrics, "previews": preview_paths})
    _synchronize(device)
    metrics = {
        "heldout_foreground_psnr": sum(row["foreground_psnr"] for row in rows) / len(rows),
        "heldout_alpha_iou": sum(row["alpha_iou"] for row in rows) / len(rows),
        "heldout_exterior_leakage": sum(row["exterior_leakage"] for row in rows) / len(rows),
        "evaluation_seconds": time.perf_counter() - started,
    }
    receipt = {
        "schema": "rtgs.bench019.local.evaluation.v1",
        "status": "ok",
        "task_id": task["task_id"],
        "frame_id": frame_id,
        "family_id": family_id,
        "seed": seed,
        "model_sha256": file_sha256(model_path),
        "model_path": str(Path(model_path).resolve()),
        "checkpoint_policy": "final",
        "view_ids": list(scene.view_names),
        "per_view": rows,
        "metrics": metrics,
        "aggregation": "arithmetic mean of per-view metrics, not pooled foreground MSE",
        "foreground_rule": "source soft alpha >= 0.5; RGB target = image * soft alpha on black",
        "predicted_alpha_rule": "rendered 3D compositing alpha >= 0.5; no 2D support proxy",
        "psnr_mse_floor": 1e-12,
    }
    _json_write(output / "evaluation.json", receipt)
    return receipt
