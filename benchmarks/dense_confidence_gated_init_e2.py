#!/usr/bin/env python3
"""Sealed E2 comparison: top-K vs dense-all vs easy-only under matched density control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.calibrated import load_calibrated_scene
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.data.scene import SceneData
from rtgs.data.synthetic import make_synthetic_scene
from rtgs.lift.compact_carve import _center_and_extent
from rtgs.lift.compact_init_eval import (
    InitEvaluationTarget,
    evaluate_initialization,
    prepare_evaluation_targets,
)
from rtgs.optim.density import DensityConfig
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render import get_rasterizer

ROOT = Path(__file__).resolve().parent.parent
PREREGISTRATION = ROOT / "benchmarks/results/20260720_dense_confidence_gated_init_e2_PREREG.md"
SOURCE_FRAME = Path("/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008")
CALIBRATION = SOURCE_FRAME.parent / "calibration_dome.json"
BUNDLE = ROOT / "runs/compact_masked_bundle_640_20260717/reconstruction_inputs"
I1_RUN = ROOT / "runs/dense_confidence_gated_init_i1_20260720"
DEFAULT_OUT = ROOT / "runs/dense_confidence_gated_init_e2_20260720"

TRAIN_VIEWS = ("C0001", "C0008", "C0014", "C0021", "C0026", "C0031", "C0039")
VALIDATION_VIEW = "C1002"
HELDOUT_VIEW = "C1004"
DOWNSCALE = 8
MAIN_SEED = 20_260_720
REPEAT_SEED = 20_260_721
MAX_GAUSSIANS = 2_319
EXPECTED_INITIAL_HASHES = {
    "topk": "d83ee1e764ee6bc0d1cf7696e848df91b0a92d33ad5c9932c9e1138e8564e9fb",
    "dense_all": "56ce5f1ac3a321f6912506dc4e2c8484c1c3b9d5930eb140b84253faf106cff7",
    "easy_only": "1d3205755d67e6e3badd48a9d41a1329a38898e6e6178150cac25aadc57b6a9f",
}
INITIAL_PATHS = {
    "topk": I1_RUN / "init_topk.ply",
    "dense_all": I1_RUN / "init_dense_merged.ply",
    "easy_only": I1_RUN / "init_easy_gated.ply",
}
EXECUTIONS = (
    ("topk", "topk", MAIN_SEED),
    ("dense_all", "dense_all", MAIN_SEED),
    ("easy_only", "easy_only", MAIN_SEED),
    ("topk_repeat", "topk", REPEAT_SEED),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_config(seed: int) -> TrainConfig:
    """Return the exact preregistered E2 optimizer and density schedule."""
    return TrainConfig(
        iterations=300,
        rasterizer="gsplat",
        device="cuda",
        densify=True,
        density_strategy="gsplat-default",
        density=DensityConfig(
            start_iter=25,
            stop_iter=275,
            every=25,
            grad_threshold=8e-4,
            absgrad=True,
            split_scale_frac=0.01,
            prune_opacity=0.005,
            prune_scale_frac=0.1,
            max_gaussians=MAX_GAUSSIANS,
            opacity_reset_every=0,
            revised_opacity=True,
        ),
        eval_every=50,
        target_sh_degree=3,
        sh_degree_interval=75,
        use_masks=True,
        random_background=True,
        mask_alpha_lambda=0.05,
        packed=False,
        antialiased=False,
        validate_render_finite=True,
        seed=seed,
    )


def _source_file(view_name: str, role: str) -> Path:
    if role == "rgb":
        path = SOURCE_FRAME / "rgb" / f"{view_name}.jpg"
    elif role == "mask":
        path = SOURCE_FRAME / "mask" / f"mask_{view_name}.png"
    else:
        raise ValueError(f"unknown source role {role!r}")
    return path


def source_manifest() -> dict:
    """Hash-bind every selected source without decoding the late-held-out tensors."""
    selected = (*TRAIN_VIEWS, VALIDATION_VIEW, HELDOUT_VIEW)
    files = []
    for view_name in selected:
        for role in ("rgb", "mask"):
            path = _source_file(view_name, role)
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append(
                {
                    "view_name": view_name,
                    "role": role,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    if not CALIBRATION.is_file():
        raise FileNotFoundError(CALIBRATION)
    return {
        "schema": "rtgs.dense_confidence_e2_sources.v1",
        "calibration": {
            "path": str(CALIBRATION),
            "sha256": sha256_file(CALIBRATION),
            "bytes": CALIBRATION.stat().st_size,
        },
        "files": files,
        "train_views": list(TRAIN_VIEWS),
        "validation_view": VALIDATION_VIEW,
        "heldout_view": HELDOUT_VIEW,
        "downscale": DOWNSCALE,
    }


def _load_selected_scene(view_names: tuple[str, ...]) -> SceneData:
    """Load only named calibrated views through the repository's canonical decoder."""
    with tempfile.TemporaryDirectory(prefix="rtgs-e2-scene-") as temporary:
        frame = Path(temporary) / "frame_00008"
        rgb = frame / "rgb"
        masks = frame / "mask"
        rgb.mkdir(parents=True)
        masks.mkdir()
        for view_name in view_names:
            (rgb / f"{view_name}.jpg").symlink_to(_source_file(view_name, "rgb"))
            (masks / f"mask_{view_name}.png").symlink_to(_source_file(view_name, "mask"))
        scene = load_calibrated_scene(
            frame,
            calibration_path=CALIBRATION,
            downscale=DOWNSCALE,
            test_every=0,
        )
    if tuple(scene.view_names or ()) != tuple(sorted(view_names)):
        raise RuntimeError("calibrated loader returned an unexpected view order")
    return scene


def load_train_validation_scene(bounds_hint: tuple[torch.Tensor, float]) -> SceneData:
    """Materialize the seven optimization views and C1002, but never C1004."""
    names = (*TRAIN_VIEWS, VALIDATION_VIEW)
    scene = _load_selected_scene(names)
    if tuple(scene.view_names or ()) != names:
        raise RuntimeError("training/validation views do not match the frozen order")
    scene.train_indices = list(range(len(TRAIN_VIEWS)))
    scene.test_indices = [len(TRAIN_VIEWS)]
    # The compact bundle's training-only bounds prevent validation geometry from affecting LRs.
    scene.bounds_hint = (bounds_hint[0].detach().clone(), float(bounds_hint[1]))
    scene.name = "frame_00008_e2_train_validation"
    scene.validate()
    return scene


def load_heldout_scene() -> SceneData:
    """Late-release decoder for the single untouched C1004 view."""
    scene = _load_selected_scene((HELDOUT_VIEW,))
    if tuple(scene.view_names or ()) != (HELDOUT_VIEW,):
        raise RuntimeError("held-out loader returned an unexpected view")
    scene.train_indices = []
    scene.test_indices = [0]
    scene.name = "frame_00008_e2_heldout"
    scene.validate()
    return scene


def _metric_renderer(device: torch.device):
    return get_rasterizer(
        "gsplat",
        device=device,
        packed=False,
        antialiased=False,
    )


def _evaluate_rgb(
    scene: SceneData,
    gaussians: Gaussians3D,
    indices: list[int],
) -> dict[str, float]:
    renderer = _metric_renderer(gaussians.means.device)
    return Trainer.evaluate_metrics(scene, gaussians, renderer, indices=indices)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _checkpoint_progress(label: str, started: float):
    def callback(snapshot: Gaussians3D, step: int) -> None:
        print(
            f"[{label}] checkpoint {step}/300: {snapshot.n:,} gaussians, "
            f"{time.perf_counter() - started:.1f}s wall",
            file=sys.stderr,
            flush=True,
        )

    return callback


def _run_arm(
    *,
    execution_name: str,
    init_name: str,
    seed: int,
    scene: SceneData,
    compact_inputs: ReconstructionInputs,
    compact_targets: tuple[InitEvaluationTarget, ...],
    out_dir: Path,
) -> tuple[Gaussians3D, dict]:
    init_path = INITIAL_PATHS[init_name]
    init = Gaussians3D.load_ply(init_path)
    execution_dir = out_dir / execution_name
    execution_dir.mkdir(parents=True, exist_ok=False)
    init_copy = execution_dir / "initial.ply"
    init.save_ply(init_copy)
    config = frozen_config(seed)
    _write_json(execution_dir / "config.json", asdict(config))

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    initial_cuda = init.to("cuda")
    validation_initial = _evaluate_rgb(scene, initial_cuda, scene.testing_views)
    started = time.perf_counter()
    final_cuda, history = Trainer(config).train(
        scene,
        init,
        checkpoint_callback=_checkpoint_progress(execution_name, started),
    )
    wall_seconds = time.perf_counter() - started
    final_path = execution_dir / "final.ply"
    final_cuda.save_ply(final_path)
    _write_json(execution_dir / "history.json", history)

    train_metrics = _evaluate_rgb(scene, final_cuda, scene.training_views)
    validation_metrics = _evaluate_rgb(scene, final_cuda, scene.testing_views)
    compact_metrics = evaluate_initialization(
        compact_inputs,
        final_cuda,
        rasterizer=_metric_renderer(final_cuda.means.device),
        targets=compact_targets,
    ).as_dict()
    native_seconds = float(history["elapsed"][-1][1])
    trajectory = [
        {
            "step": int(psnr_item[0]),
            "validation_foreground_psnr": float(psnr_item[1]),
            "native_elapsed_seconds": float(elapsed_item[1]),
            "n_gaussians": int(count_item[1]),
        }
        for psnr_item, elapsed_item, count_item in zip(
            history["psnr"],
            history["elapsed"],
            history["n_gaussians"],
            strict=True,
        )
    ]
    record = {
        "execution": execution_name,
        "initialization": init_name,
        "seed": seed,
        "initial_count": init.n,
        "final_count": final_cuda.n,
        "wall_seconds": wall_seconds,
        "native_optimization_seconds": native_seconds,
        "peak_vram_gb": history["peak_vram_gb"],
        "initial_ply": str(init_copy),
        "initial_ply_sha256": sha256_file(init_copy),
        "final_ply": str(final_path),
        "final_ply_sha256": sha256_file(final_path),
        "history": str(execution_dir / "history.json"),
        "history_sha256": sha256_file(execution_dir / "history.json"),
        "validation_initial_rgb": validation_initial,
        "train_final_rgb": train_metrics,
        "validation_final_rgb": validation_metrics,
        "compact_train_final": compact_metrics,
        "trajectory": trajectory,
        "density_stats": history["density_stats"],
        "viewer_command": (
            f".venv/bin/rtgs view --gaussians {final_path} --initial {init_copy} "
            f"--scene {SOURCE_FRAME} --downscale {DOWNSCALE} --rasterizer gsplat "
            "--device cuda --no-open"
        ),
    }
    _write_json(execution_dir / "preheldout_metrics.json", record)
    final_cpu = final_cuda.to("cpu")
    del final_cuda, initial_cuda
    torch.cuda.empty_cache()
    return final_cpu, record


def decide_e2(executions: dict[str, dict]) -> dict:
    """Apply the exact repeat-calibrated held-out decision rule."""
    required = {"topk", "dense_all", "easy_only", "topk_repeat"}
    if set(executions) != required:
        raise ValueError(f"decision requires executions {sorted(required)}")
    topk_psnr = float(executions["topk"]["heldout_final_rgb"]["psnr_fg"])
    repeat_psnr = float(executions["topk_repeat"]["heldout_final_rgb"]["psnr_fg"])
    control_envelope = abs(topk_psnr - repeat_psnr)
    allowed_deficit = min(0.1, control_envelope)
    competitors = ("topk", "dense_all")
    best_name = max(
        competitors,
        key=lambda name: (
            float(executions[name]["heldout_final_rgb"]["psnr_fg"]),
            -int(executions[name]["final_count"]),
            -float(executions[name]["native_optimization_seconds"]),
        ),
    )
    best = executions[best_name]
    easy = executions["easy_only"]
    quality_within = float(easy["heldout_final_rgb"]["psnr_fg"]) >= (
        float(best["heldout_final_rgb"]["psnr_fg"]) - allowed_deficit
    )
    count_within = int(easy["final_count"]) <= int(best["final_count"])
    time_within = float(easy["native_optimization_seconds"]) <= float(
        best["native_optimization_seconds"]
    )
    return {
        "control_repeat_envelope_db": control_envelope,
        "allowed_quality_deficit_db": allowed_deficit,
        "best_competitor": best_name,
        "easy_quality_within_repeat_calibrated_band": quality_within,
        "easy_primitive_count_no_greater": count_within,
        "easy_native_time_no_greater": time_within,
        "easy_only_wins": quality_within and count_within and time_within,
    }


def run_cpu_smoke(out_dir: Path) -> dict:
    """Mechanism-only CPU smoke of three cardinalities with classic density control."""
    scene = make_synthetic_scene(n_gaussians=6, n_cameras=3, image_size=12, seed=17)
    assert scene.gt_gaussians is not None
    smoke_config = TrainConfig(
        iterations=4,
        rasterizer="torch",
        device="cpu",
        densify=True,
        density_strategy="classic",
        density=DensityConfig(
            start_iter=1,
            stop_iter=3,
            every=1,
            grad_threshold=0.0,
            max_gaussians=6,
            opacity_reset_every=0,
        ),
        eval_every=2,
        target_sh_degree=0,
        ssim_lambda=0.0,
        use_masks=False,
        random_background=False,
        seed=31,
    )
    rows = {"easy_only": 2, "topk": 4, "dense_all": 6}
    result = {}
    for name, count in rows.items():
        torch.manual_seed(31)
        final, history = Trainer(smoke_config).train(
            scene,
            scene.gt_gaussians.subset(torch.arange(count)),
        )
        if final.n > 6:
            raise RuntimeError("CPU smoke violated the matched primitive cap")
        result[name] = {
            "initial_count": count,
            "final_count": final.n,
            "density_stats": history["density_stats"],
            "final_psnr": Trainer.evaluate(scene, final),
        }
    _write_json(out_dir / "cpu_smoke.json", result)
    return result


def run(out_dir: Path) -> dict:
    if out_dir.exists():
        raise FileExistsError(f"E2 output namespace already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    smoke = run_cpu_smoke(out_dir)
    manifest = source_manifest()
    manifest["preregistration"] = {
        "path": str(PREREGISTRATION),
        "sha256": sha256_file(PREREGISTRATION),
    }
    manifest["initializations"] = {}
    for name, path in INITIAL_PATHS.items():
        actual = sha256_file(path)
        if actual != EXPECTED_INITIAL_HASHES[name]:
            raise RuntimeError(f"{name} initialization hash changed")
        manifest["initializations"][name] = {
            "path": str(path),
            "sha256": actual,
        }
    _write_json(out_dir / "source_manifest.json", manifest)

    compact_inputs = ReconstructionInputs.load(BUNDLE, strict=True)
    compact_targets = prepare_evaluation_targets(compact_inputs)
    training_bounds = _center_and_extent(compact_inputs, torch.float32)
    scene = load_train_validation_scene(training_bounds)
    finals: dict[str, Gaussians3D] = {}
    records: dict[str, dict] = {}
    for execution_name, init_name, seed in EXECUTIONS:
        print(
            f"[E2] starting {execution_name}: init={init_name}, seed={seed}",
            file=sys.stderr,
            flush=True,
        )
        final, record = _run_arm(
            execution_name=execution_name,
            init_name=init_name,
            seed=seed,
            scene=scene,
            compact_inputs=compact_inputs,
            compact_targets=compact_targets,
            out_dir=out_dir,
        )
        finals[execution_name] = final
        records[execution_name] = record

    preheldout = {
        "schema": "rtgs.dense_confidence_e2_preheldout.v1",
        "heldout_materialized": False,
        "completed_executions": list(records),
        "execution_artifacts": {
            name: {
                "final_ply_sha256": record["final_ply_sha256"],
                "history_sha256": record["history_sha256"],
                "final_count": record["final_count"],
                "native_optimization_seconds": record["native_optimization_seconds"],
            }
            for name, record in records.items()
        },
    }
    _write_json(out_dir / "PREHELDOUT.json", preheldout)

    heldout_scene = load_heldout_scene()
    for execution_name, init_name, _seed in EXECUTIONS:
        final_cuda = finals[execution_name].to("cuda")
        init_cuda = Gaussians3D.load_ply(INITIAL_PATHS[init_name]).to("cuda")
        records[execution_name]["heldout_initial_rgb"] = _evaluate_rgb(
            heldout_scene, init_cuda, [0]
        )
        records[execution_name]["heldout_final_rgb"] = _evaluate_rgb(heldout_scene, final_cuda, [0])
        del final_cuda, init_cuda
        torch.cuda.empty_cache()
        _write_json(
            out_dir / execution_name / "complete_metrics.json",
            records[execution_name],
        )

    decision = decide_e2(records)
    init_report = json.loads((I1_RUN / "init_eval.json").read_text(encoding="utf-8"))
    report = {
        "schema": "rtgs.dense_confidence_e2_result.v1",
        "preregistration_sha256": manifest["preregistration"]["sha256"],
        "source_manifest_sha256": sha256_file(out_dir / "source_manifest.json"),
        "cpu_smoke": smoke,
        "split": {
            "train": list(TRAIN_VIEWS),
            "validation": VALIDATION_VIEW,
            "heldout": HELDOUT_VIEW,
            "heldout_materialized_after_all_optimization": True,
            "downscale": DOWNSCALE,
        },
        "config": asdict(frozen_config(MAIN_SEED)),
        "init_only_compact_metrics": {
            "topk": init_report["topk"],
            "dense_all": init_report["dense_merged"],
            "easy_only": init_report["easy_gated"],
        },
        "executions": records,
        "decision": decision,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "pid": os.getpid(),
        },
    }
    _write_json(out_dir / "e2_result.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="run only the tiny CPU/classic density-control mechanism check",
    )
    args = parser.parse_args(argv)
    if args.smoke_only:
        args.out.mkdir(parents=True, exist_ok=True)
        print(json.dumps(run_cpu_smoke(args.out), indent=2))
        return 0
    report = run(args.out)
    summary = {
        name: {
            "heldout_psnr_fg": record["heldout_final_rgb"]["psnr_fg"],
            "heldout_ssim_crop": record["heldout_final_rgb"]["ssim_crop"],
            "heldout_alpha_iou": record["heldout_final_rgb"]["alpha_iou"],
            "final_count": record["final_count"],
            "native_seconds": record["native_optimization_seconds"],
        }
        for name, record in report["executions"].items()
    }
    print(json.dumps({"executions": summary, "decision": report["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
