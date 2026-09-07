"""Task-owned, training-only acquisition for the local BENCH-019 comparison.

Imports stay light; optional fitters and torch are loaded only at execution. This
module writes phase-one inputs, never downstream models or held-out metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

FAMILIES = ("native_additive", "structsplat_normalized", "structsplat_contained")
HISTORY_STEPS = (0, 100, 250, 500, 750, 1000)
SCORING = {
    "canvas": "full calibrated downscale8 canvas",
    "target": "float32 processed RGB times soft source alpha, on black",
    "display": "exact observation RGB clamped to [0,1] only for scoring",
    "foreground": "source alpha >= 0.5",
    "foreground_psnr": (
        "-10*log10(max(foreground_squared_error_sum / foreground_value_count,1e-12))"
    ),
    "boundary": "3x3 binary morphological gradient with exterior padding=background",
    "boundary_mae": "boundary_absolute_error_sum / boundary_value_count",
    "support_iou": "true_positive / (true_positive + false_positive + false_negative)",
    "support": "query.valid and query.weight_sum > 1e-8; support threshold, not alpha",
    "aggregation": "pool sufficient statistics over training views, then evaluate formulas",
    "sampling": "2048 uniform full-canvas draws with replacement; same samples across families",
    "empty_boundary_sample": "use complete boundary, recorded separately",
    "cold_replay": "same pixel samples, CPU float64, max_abs <= 1e-10; valid exact",
}


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _file(path):
    path = Path(path)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def resolved_stage1_configs():
    """Full dataclasses; only InitConfig.seed is replaced by the frozen view seed."""
    from structsplat.config import FitConfig, InitConfig, StructureTensorConfig

    from rtgs.image2gs.fit import FitConfig as NativeFitConfig

    native = NativeFitConfig(
        n_gaussians=512,
        max_gaussians=512,
        iterations=1000,
        backend="native",
        adaptive_density=False,
        native_renderer="cuda",
        convergence_patience=0,
        batch_views=False,
        pool=False,
        mask_coverage_weight=0.0,
    )
    init = InitConfig(
        strategy="aniso_onedge",
        num_gaussians=512,
        sampling_mode="wse",
        scale_cap_mode="none",
        seed=0,
    )
    result = {"native_additive": {"fit": asdict(native), "init": None, "structure_tensor": None}}
    for family in FAMILIES[1:]:
        fit = FitConfig(
            iters=1000,
            renderer="cuda",
            pixel_loss="l2",
            ssim_weight=0.0,
            loss_weighting="none",
            mask_contain=family == "structsplat_contained",
            support_fade=True,
            normalization_eps=1e-8,
            aa_dilation=0.0,
            covariance_filter_mode="none",
            color_basis="constant",
            color_solve_schedule="none",
            split_mode="none",
            split_every=None,
            split_count=0,
            prune_every=None,
            relocate_every=None,
            relocate_count=0,
            adaptive_count=False,
            max_gaussians=512,
            early_stop_patience=None,
            triage_every=None,
            checkpoint_policy="terminal",
            compute_lpips=False,
        )
        result[family] = {
            "fit": asdict(fit),
            "init": asdict(init),
            "structure_tensor": asdict(StructureTensorConfig()),
        }
    return result


def prepare_target(image, alpha):
    """Return the shared soft-matted crop and (x,y,width,height), without recropping."""
    import torch

    if image.ndim != 3 or image.shape[-1] != 3 or alpha.shape != image.shape[:2]:
        raise ValueError("RGB/alpha dimensions do not match")
    if not bool(torch.isfinite(image).all() and torch.isfinite(alpha).all()):
        raise ValueError("RGB/alpha must be finite")
    if bool(((image < 0) | (image > 1)).any() or ((alpha < 0) | (alpha > 1)).any()):
        raise ValueError("RGB/alpha must be in [0,1]")
    foreground = alpha >= 0.5
    yy, xx = foreground.nonzero(as_tuple=True)
    if not len(xx) or bool(foreground.all()):
        raise ValueError("source mask requires foreground and exterior")
    height, width = alpha.shape
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    mx, my = math.ceil((x1 - x0) * 0.05), math.ceil((y1 - y0) * 0.05)
    x0, x1 = max(0, x0 - mx), min(width, x1 + mx)
    y0, y1 = max(0, y0 - my), min(height, y1 + my)
    target = (image * alpha[..., None]).contiguous()
    return target[y0:y1, x0:x1].contiguous(), (x0, y0, x1 - x0, y1 - y0)


def _tensor_digest(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return _digest(
        {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    )


def _query(field, points):
    import torch

    from rtgs.core.observation2d import ObservationQuery

    outputs = [field.query(part) for part in points.split(512)]
    return ObservationQuery(
        *(
            torch.cat([getattr(item, key) for item in outputs])
            for key in ("color", "numerator", "weight_sum", "valid")
        )
    )


def predictor_samples(alpha, seed, frame_id, view_id, count=2048):
    """Fixed full-canvas samples, plus a separately labelled empty-boundary fallback."""
    import torch
    import torch.nn.functional as functional

    foreground = alpha.cpu() >= 0.5
    values = foreground.float()[None, None]
    dilated = functional.max_pool2d(values, 3, stride=1, padding=1)[0, 0] > 0
    complement = functional.pad(1 - values, (1, 1, 1, 1), value=1)
    eroded = 1 - functional.max_pool2d(complement, 3, stride=1)[0, 0]
    boundary = dilated & (eroded < 0.5)
    token = _digest({"seed": seed, "frame": frame_id, "view": view_id})
    generator = torch.Generator().manual_seed(int(token[:16], 16) % (2**63 - 1))
    indices = torch.randint(alpha.numel(), (count,), generator=generator)
    if not bool(foreground.flatten()[indices].any()):
        raise ValueError("fixed sample has no foreground")
    boundary_indices = indices[boundary.flatten()[indices]]
    fallback = not len(boundary_indices)
    if fallback:
        boundary_indices = boundary.flatten().nonzero().flatten()
    if not len(boundary_indices):
        raise ValueError("source mask has no boundary")
    return (
        indices,
        boundary_indices,
        {
            "seed": seed,
            "token": token,
            "count": count,
            "indices_sha256": _tensor_digest(indices),
            "boundary_indices_sha256": _tensor_digest(boundary_indices),
            "boundary_count": len(boundary_indices),
            "complete_boundary_fallback": fallback,
        },
    )


def _points(indices, width):
    import torch

    return torch.stack((indices % width, indices // width), dim=-1).double() + 0.5


def score_observation(field, target, alpha, indices, boundary_indices):
    """Return additive sufficient statistics, using full-canvas exact observations."""
    import torch

    cpu = field.to("cpu", dtype=torch.float64)
    target = target.cpu().double().reshape(-1, 3)
    foreground = (alpha.cpu() >= 0.5).flatten()
    query = _query(cpu, _points(indices, field.width))
    valid = query.valid & (query.weight_sum > 1e-8)
    errors = query.color.clamp(0, 1) - target[indices]
    inside = foreground[indices]
    boundary_query = _query(cpu, _points(boundary_indices, field.width))
    boundary_error = boundary_query.color.clamp(0, 1) - target[boundary_indices]
    return {
        "foreground_squared_error_sum": float(errors[inside].square().sum()),
        "foreground_value_count": int(inside.sum()) * 3,
        "boundary_absolute_error_sum": float(boundary_error.abs().sum()),
        "boundary_value_count": len(boundary_indices) * 3,
        "true_positive": int((valid & inside).sum()),
        "false_positive": int((valid & ~inside).sum()),
        "false_negative": int((~valid & inside).sum()),
    }


def _metrics(stats):
    mse = stats["foreground_squared_error_sum"] / stats["foreground_value_count"]
    union = stats["true_positive"] + stats["false_positive"] + stats["false_negative"]
    return {
        "foreground_psnr": -10 * math.log10(max(mse, 1e-12)),
        "boundary_mae": stats["boundary_absolute_error_sum"] / stats["boundary_value_count"],
        "support_iou": stats["true_positive"] / union,
    }


def _fit_target(
    target,
    alpha,
    family,
    config,
    seed,
    canvas,
    window,
    view_id,
    source_digest,
    *,
    history_steps=HISTORY_STEPS,
):
    """Uninterrupted fitting through existing APIs; tests may pass tiny CPU configs."""
    import torch

    renderer = config["fit"].get("native_renderer", config["fit"].get("renderer"))
    if renderer == "cuda":
        target, alpha = target.to("cuda:0"), alpha.to("cuda:0")
    started = time.perf_counter()
    history = []
    count = config["fit"].get(
        "n_gaussians", config["init"]["num_gaussians"] if config["init"] else 0
    )

    def record(step, rendered, rows):
        if rows != count:
            raise ValueError("Stage-1 row count drift")
        if not bool(torch.isfinite(rendered).all()):
            raise ValueError("nonfinite source render")
        history.append(
            {
                "step": step,
                "elapsed_seconds": time.perf_counter() - started,
                "rows": rows,
                "pixel_l2": float((rendered - target).square().mean()),
                "state": "post-update; step0 after fit-entry geometry projection",
            }
        )

    provenance = {
        "canvas_size": canvas,
        "fit_window": window,
        "view_id": view_id,
        "n_init": count,
        "producer_source_digest": source_digest,
        "fit_config_digest": _digest(config),
    }
    if family == "native_additive":
        from rtgs.image2gs.fit import FitConfig, fit_image
        from rtgs.image2gs.native_observation import native_gaussians_to_observation
        from rtgs.image2gs.renderer2d import render_gaussians_2d

        cfg = FitConfig(**config["fit"])

        def diagnostic(event):
            if event.gaussians.n != count:
                raise ValueError("Stage-1 row count drift")
            if event.event in ("initial", "checkpoint") and event.step in history_steps:
                record(event.step, event.rendered, event.gaussians.n)

        field, raw_history = fit_image(
            target,
            cfg,
            seed=seed,
            mask=None,
            diagnostic_callback=diagnostic,
            diagnostic_steps=history_steps,
        )
        observation = native_gaussians_to_observation(field, **provenance)
        final_render = render_gaussians_2d(
            field, *target.shape[:2], row_chunk=cfg.row_chunk, renderer=cfg.native_renderer
        )
        if raw_history["stopped_iter"] != cfg.iterations - 1:
            raise ValueError("native fit stopped before the fixed horizon")
    else:
        from structsplat.config import FitConfig, InitConfig, StructureTensorConfig
        from structsplat.fit import _MaskConstraint, _render, fit
        from structsplat.init import build_field

        from rtgs.image2gs.structsplat_backend import field_to_observation

        cfg = FitConfig(**config["fit"])
        init = InitConfig(**{**config["init"], "seed": seed})
        field = build_field(
            target.detach().cpu().numpy(),
            init,
            StructureTensorConfig(**config["structure_tensor"]),
            device=target.device,
        )
        constraint = None
        mask = None
        if cfg.mask_contain:
            mask = (alpha >= 0.5).detach().cpu().numpy()
            constraint = _MaskConstraint.from_mask(
                mask,
                target.device,
                target.dtype,
                cfg.sigma_cutoff,
                cfg.mask_margin,
                aa_dilation=cfg.aa_dilation,
                cap_mode=cfg.mask_cap_mode,
                undercoverage_band=cfg.mask_undercoverage_band,
            )
            constraint.apply(field, cfg)
        with torch.no_grad():
            record(0, _render(field, cfg, *target.shape[:2]), field.n)

        def observer(live, step, _pre_update_loss):
            if live.n != count:
                raise ValueError("Stage-1 row count drift")
            if step in history_steps:
                record(step, _render(live, cfg, *target.shape[:2]), live.n)

        fit_info = fit(
            field,
            target,
            cfg,
            verbose=False,
            mask=mask,
            mask_constraint_override=constraint,
            iteration_observer=observer,
            observer_every=1,
        )
        field = fit_info["field"]
        if fit_info["stopped_early"] or fit_info["iterations_run"] != cfg.iters:
            raise ValueError("StructSplat fit stopped before the fixed horizon")
        observation = field_to_observation(
            field,
            **provenance,
            epsilon=cfg.normalization_eps,
            sigma_cutoff=cfg.sigma_cutoff,
            support_fade_alpha=1.0,
            aa_dilation=cfg.aa_dilation,
        )
        final_render = _render(field, cfg, *target.shape[:2])
    if observation.n != count or [row["step"] for row in history] != list(history_steps):
        raise ValueError("Stage-1 count or temporal checkpoint drift")
    return observation.to("cpu"), history, final_render.detach().cpu()


def _source_binding():
    import structsplat

    import rtgs

    files = {}
    for name, package in (("rtgs", rtgs), ("structsplat", structsplat)):
        root = Path(package.__file__).parent
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".cpp", ".cu", ".h", ".cuh"}:
                files[f"{name}/{path.relative_to(root)}"] = _file(path)["sha256"]
    return {"sha256": _digest(files), "files": files}


def _save_previews(directory, target, field):
    import numpy as np
    import torch
    from PIL import Image

    x, y, width, height = field.fit_window
    indices = torch.arange(height * width)
    points = _points(indices, width) + torch.tensor([x, y], dtype=torch.float64)
    query = _query(field.to("cpu", dtype=torch.float64), points)
    prediction = query.color.reshape(height, width, 3).clamp(0, 1).float()
    result = {}
    for name, tensor in (
        ("target", target),
        ("reconstruction", prediction),
        ("absolute_error", (prediction - target).abs()),
    ):
        path = directory / f"{name}.png"
        Image.fromarray(np.rint(tensor.numpy().clip(0, 1) * 255).astype(np.uint8)).save(path)
        result[name] = _file(path)
    return result


def produce_frame_family(task, frame_id, family_id, output_dir):
    """Generate immutable training inputs after the coordinator approves phase one."""
    import torch

    from rtgs.data.calibrated import load_calibrated_scene
    from rtgs.data.compact_views import (
        CompactDataset,
        CompactView,
        save_compact_view,
        write_compact_dataset_manifest,
    )

    if task.get("status") != "ready" or family_id not in FAMILIES:
        raise ValueError("phase one requires a ready task and a declared family")
    production = task["production"]
    configs = resolved_stage1_configs()
    if production.get("resolved_dataclasses") != configs:
        raise ValueError("resolved Stage-1 configs differ from the frozen task")
    if (
        production["downscale"],
        production["count"],
        production["iterations"],
        production["seed_base"],
        production["history_steps"],
    ) != (8, 512, 1000, 190000, list(HISTORY_STEPS)):
        raise ValueError("Stage-1 fixed budget or seed drift")
    if (
        task["predictor_policy"]["seed"] != 191019
        or task["predictor_policy"]["samples_per_view"] != 2048
    ):
        raise ValueError("Stage-1 predictor policy drift")
    split = task["splits"][frame_id]
    train = sorted(split["train"])
    if len(train) != 8 or len(set(train)) != 8 or set(train) & set(split["heldout"]):
        raise ValueError("Stage-1 requires eight distinct training-only views")
    dataset = next(item for item in task["datasets"] if item["id"] == frame_id)
    frame = Path(dataset["frame_path"]).resolve()
    calibration = Path(dataset["calibration"]).resolve()
    raw_paths = [calibration] + [
        path
        for view in train
        for path in (frame / "rgb" / f"{view}.jpg", frame / "mask" / f"mask_{view}.png")
    ]
    raw = [_file(path) for path in raw_paths]
    source = _source_binding()
    task_digest = _digest(task)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("explicit CUDA Stage-1 requires an available device")
        scene = load_calibrated_scene(
            frame,
            calibration_path=calibration,
            downscale=8,
            test_every=0,
            load_masks=True,
            undistort=True,
            view_ids=train,
        )
        if (
            scene.view_names != train
            or scene.train_indices != list(range(8))
            or scene.test_indices
            or scene.masks is None
        ):
            raise ValueError("training loader split/mask drift")
        _write(output / "source_binding.json", source)
        records = []
        paths = []
        for index, view_id in enumerate(train):
            image, alpha, camera = scene.images[index], scene.masks[index], scene.cameras[index]
            crop, window = prepare_target(image, alpha)
            # The compact schema records a >.5 binary mask. Calibrated nearest
            # 8-bit PNG alpha has no exactly-.5 value, so >= and > must coincide.
            if bool((alpha == 0.5).any()):
                raise ValueError("source alpha threshold disagrees with compact mask metadata")
            x, y, width, height = window
            crop_alpha = alpha[y : y + height, x : x + width]
            seed = 190000 + 100 * sorted(task["splits"]).index(frame_id) + index
            config = json.loads(json.dumps(configs[family_id]))
            config["seed"] = seed
            if config["init"] is not None:
                config["init"]["seed"] = seed
            indices, boundary_indices, sampling = predictor_samples(
                alpha, 191019, frame_id, view_id
            )
            field, history, source_render = _fit_target(
                crop,
                crop_alpha,
                family_id,
                config,
                seed,
                tuple(alpha.shape),
                window,
                view_id,
                source["sha256"],
            )
            path = output / f"{view_id}.rtgsv"
            save_compact_view(
                path,
                field,
                camera,
                calibration_sha256=raw[0]["sha256"],
                source_rgb_name=f"{view_id}.jpg",
                source_rgb_sha256=raw[1 + 2 * index]["sha256"],
                source_mask_name=f"mask_{view_id}.png",
                source_mask_sha256=raw[2 + 2 * index]["sha256"],
                alpha_crop=crop_alpha >= 0.5,
            )
            loaded = CompactView.load(path).observation
            expected_provider = "native" if family_id == "native_additive" else "structsplat"
            expected_blend = "additive" if family_id == "native_additive" else "normalized"
            if (
                loaded.n_init != 512
                or loaded.provider != expected_provider
                or loaded.blend_mode != expected_blend
                or loaded.fit_window != window
                or loaded.epsilon != 1e-8
                or loaded.aa_dilation != 0.0
                or loaded.support_fade_alpha != (0.0 if family_id == "native_additive" else 1.0)
                or loaded.sigma_cutoff
                != (math.sqrt(12.0) if family_id == "native_additive" else 3.0)
                or loaded.color_grads is not None
                or loaded.filter_variance is not None
                or loaded.fit_config_digest != _digest(config)
                or loaded.producer_source_digest != source["sha256"]
            ):
                raise ValueError("cold field provenance or family semantics drift")
            points = _points(indices, camera.width)
            before = _query(field.to("cpu", dtype=torch.float64), points)
            after = _query(loaded.to("cpu", dtype=torch.float64), points)
            error = max(
                float((getattr(before, key) - getattr(after, key)).abs().max())
                for key in ("color", "numerator", "weight_sum")
            )
            if error > 1e-10 or not torch.equal(before.valid, after.valid) or loaded.n != 512:
                raise ValueError("cold serialization replay/count drift")
            source_full = torch.zeros_like(image)
            source_full[y : y + height, x : x + width] = source_render
            source_sample = source_full.reshape(-1, 3)[indices].double()
            stats = score_observation(
                loaded, image * alpha[..., None], alpha, indices, boundary_indices
            )
            history_path = output / f"{view_id}.history.json"
            _write(
                history_path,
                {
                    "frame_id": frame_id,
                    "family_id": family_id,
                    "view_id": view_id,
                    "seed": seed,
                    "stage": "stage1",
                    "split": "train",
                    "history": history,
                },
            )
            records.append(
                {
                    "view_id": view_id,
                    "seed": seed,
                    "fit_window": list(window),
                    "field": _file(path),
                    "field_rows": loaded.n,
                    "config": config,
                    "config_sha256": _digest(config),
                    "target_sha256": _tensor_digest(crop),
                    "soft_alpha_sha256": _tensor_digest(alpha),
                    "sampling": sampling,
                    "statistics": stats,
                    "metrics": _metrics(stats),
                    "history": _file(history_path),
                    "cold_replay_max_abs": error,
                    "source_renderer_vs_observation": {
                        "scope": "unclamped RGB at same full-canvas samples; descriptive",
                        "max_abs": float((before.color - source_sample).abs().max()),
                        "mean_abs": float((before.color - source_sample).abs().mean()),
                    },
                }
            )
            paths.append(path)
            if index == 0:
                records[-1]["previews"] = _save_previews(output, crop, loaded)
        manifest = write_compact_dataset_manifest(
            output,
            name=f"{frame_id}_{family_id}",
            calibration_sha256=raw[0]["sha256"],
            view_paths=paths,
            bounds_hint=None,
        )
        dataset_check = CompactDataset.load(output)
        if [view.view_id for view in dataset_check.views] != train:
            raise ValueError("cold manifest training split drift")
        statistics = {
            key: sum(record["statistics"][key] for record in records)
            for key in records[0]["statistics"]
        }
        field_bytes = manifest.stat().st_size + sum(path.stat().st_size for path in paths)
        metrics = {
            "schema": "rtgs.bench019.local.stage1_metrics.v1",
            "frame_id": frame_id,
            "family_id": family_id,
            "split": "train",
            "scoring": SCORING,
            "statistics": statistics,
            "metrics": {
                **_metrics(statistics),
                "field_bytes": field_bytes,
                "rows": 512 * len(train),
            },
            "views": records,
        }
        _write(output / "stage1_metrics.json", metrics)
        if (
            raw != [_file(path) for path in raw_paths]
            or source != _source_binding()
            or task_digest != _digest(task)
        ):
            raise ValueError("raw input/source/task changed during Stage-1 acquisition")
        receipt = {
            "schema": "rtgs.bench019.local.production.v1",
            "status": "complete",
            "frame_id": frame_id,
            "family_id": family_id,
            "train_views": train,
            "task_sha256": task_digest,
            "source_sha256": source["sha256"],
            "raw_inputs": raw,
            "manifest": _file(manifest),
            "stage1_metrics": _file(output / "stage1_metrics.json"),
            "config_template_sha256": _digest(configs[family_id]),
            "wall_seconds": time.perf_counter() - started,
            "timing_scope": (
                "one acquisition; includes diagnostics/serialization/previews; descriptive"
            ),
            "views": records,
        }
        _write(output / "production.json", receipt)
        return receipt
    except Exception as exc:
        _write(
            output / "failure.json",
            {
                "status": "failed",
                "type": type(exc).__name__,
                "message": str(exc),
                "frame_id": frame_id,
                "family_id": family_id,
                "task_sha256": task_digest,
            },
        )
        raise
