#!/usr/bin/env python3
"""Development driver for the RTGS-007 three-path full-resolution paper pipeline.

The task remains draft until its complete schedule and prospective review are frozen.  These
subcommands provide the mechanism-only path needed to finish that protocol:

* ``initialize`` loads only compact fields/calibration and writes exact-count Random,
  Splat-SfM, and Beam Fusion starts;
* ``train`` gives one saved start to the common compact-field trainer with classic
  clone/split/prune density control;
* ``train-standard`` natively replays each sealed field crop and runs the established ordinary
  30k full-crop gsplat recipe with DefaultStrategy density control;
* ``present`` renders the three endpoints from calibrated cameras and writes the synchronized
  interactive-viewer manifest.

No subcommand reads source RGB or masks. Every training worker denies image suffixes and
image-capable loaders. The sparse compact worker additionally denies ``SceneData`` and the dense
trainer; ``train-standard`` permits them only for tensors replayed directly from sealed fields.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import dataclasses
import importlib
import io
import json
import os
import resource
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "20260730_paper_three_path_fullres_stage_frames00008_00009"
DEFAULT_TASK = ROOT / "experiments/tasks" / f"{TASK_ID}.json"
DEFAULT_OUTPUT = ROOT / ".scratch" / TASK_ID / "development_demo"
ARMS = ("bounded_random", "splat_sfm", "beam_fusion")
ARM_LABELS = {
    "bounded_random": "Bounded Random",
    "splat_sfm": "Splat-SfM",
    "beam_fusion": "Beam Fusion",
}
IMAGE_SUFFIXES = frozenset({".bmp", ".exr", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
IMAGE_FORBIDDEN_MODULES = (
    "PIL",
    "cv2",
    "imageio",
    "rtgs.data.calibrated",
)
DENSE_FORBIDDEN_MODULES = (
    "rtgs.data.scene",
    "rtgs.optim.trainer",
)

# Mechanism settings.  They are deliberately mirrored into the task only after the high-capacity
# teachers and one bounded topology pilot pass.  Until then every output is development evidence.
INITIALIZER_CONFIG = {
    "random_bounds_scale": 0.5,
    "init_opacity": 0.1,
    "max_starting_gaussians": 5_000,
    "structural_components_per_view": 2_000,
    "splat_sfm": {
        "min_views": 2,
        "source_chunk": 256,
        "seed_pair_limit": 24,
    },
    "beam_fusion": {
        "min_views": 3,
        "source_chunk": 256,
        "pair_limit": 24,
        "max_components": 5_000,
        "seed_budget_multiplier": 4,
    },
}
TRAIN_CONFIG = {
    "iterations": 10_000,
    "attempts_per_step": 128,
    "proposal_mode": "area_gaussian",
    "schedule_mode": "balanced_cycle",
    "target_mode": "uniform",
    "uniform_fraction": 0.25,
    "device": "cuda:0",
    "lr_means": 1.6e-4,
    "lr_quats": 1.0e-3,
    "lr_scales": 5.0e-3,
    "lr_opacity": 5.0e-2,
    "lr_sh": 2.5e-3,
    "lr_sh_rest": 1.25e-4,
    "point_chunk": 64,
    "gaussian_chunk": 1_024,
    "outer_microbatch": 128,
    "query_component_chunk": 512,
    "teacher_tile_size": 16,
    "evaluation_chunk": 1_024,
    "checkpoints": (
        0,
        100,
        250,
        500,
        1_000,
        2_000,
        3_500,
        5_000,
        6_500,
        8_000,
        10_000,
    ),
    "evaluate_checkpoint_risks": False,
    "sh_degree": 3,
}
DENSITY_CONFIG = {
    "start_iter": 60,
    "stop_iter": 8_000,
    "every": 40,
    "grad_threshold": 2e-4,
    "absgrad": False,
    "split_scale_frac": 0.01,
    "split_factor": 1.6,
    "prune_opacity": 0.005,
    "prune_scale_frac": 0.1,
    "max_gaussians": 100_000,
    "opacity_reset_every": 3_000,
    "opacity_reset_value": 0.011,
    "revised_opacity": True,
}
QUERY_INDEX_CONFIG = {
    "tile_size": 16,
    "max_entries": 16_000_000,
    "max_candidates": 200_000,
    "max_query_pairs": 1_048_576,
}
POINT_RENDER_CONFIG = {
    "backend": "gsplat_microcamera",
    "packed": True,
    "antialiased": False,
    "kernel_support_mode": "hard",
}
STANDARD_3DGS_CONFIG = {
    "iterations": 30_000,
    "eval_every": 1_000,
    "device": "cuda:0",
    "rasterizer": "gsplat",
    "density_strategy": "gsplat-default",
    "densify_start": 500,
    "densify_stop": 15_000,
    "densify_every": 100,
    "grad_threshold": 8e-4,
    "max_gaussians": 100_000,
    "prune_opacity": 0.005,
    "prune_scale_frac": 0.1,
    "target_sh_degree": 3,
    "sh_degree_interval": 1_000,
    "packed": True,
    "antialiased": True,
    "use_masks": True,
    "random_background": False,
    "stream_scene_from_cpu": True,
    "checkpoints": (1_000, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000),
}


class NoImageGuard:
    """Live source-image boundary with optional generated-dense training support."""

    def __init__(self, *, allow_generated_dense: bool = False) -> None:
        self.forbidden_modules = (
            IMAGE_FORBIDDEN_MODULES
            if allow_generated_dense
            else (*IMAGE_FORBIDDEN_MODULES, *DENSE_FORBIDDEN_MODULES)
        )
        self.negative_control_expected = 2 if allow_generated_dense else 4
        self.denied_paths = 0
        self.denied_imports = 0
        self.negative_control_denials = 0
        self._probing = False
        self._open = builtins.open
        self._io_open = io.open
        self._os_open = os.open
        self._import = builtins.__import__
        self._import_module = importlib.import_module

    def _forbidden_module(self, name: str) -> bool:
        return any(name == root or name.startswith(f"{root}.") for root in self.forbidden_modules)

    @staticmethod
    def _forbidden_path(value: object) -> bool:
        if isinstance(value, int):
            return False
        try:
            path = Path(os.fspath(value))
        except TypeError:
            return False
        return path.suffix.lower() in IMAGE_SUFFIXES

    @staticmethod
    def _resolved_name(
        name: str,
        globals_value: Mapping[str, Any] | None,
        level: int,
    ) -> str:
        if level <= 0:
            return name
        package = None if globals_value is None else globals_value.get("__package__")
        if not isinstance(package, str) or not package:
            return name
        try:
            return importlib_util.resolve_name("." * level + name, package)
        except (ImportError, ValueError):
            return name

    def _deny_path(self, value: object) -> None:
        if not self._forbidden_path(value):
            return
        if self._probing:
            self.negative_control_denials += 1
        else:
            self.denied_paths += 1
        raise PermissionError("compact reconstruction denies every image-file open")

    def _guarded_open(self, value: object, *args: object, **kwargs: object) -> Any:
        self._deny_path(value)
        return self._open(value, *args, **kwargs)

    def _guarded_io_open(self, value: object, *args: object, **kwargs: object) -> Any:
        self._deny_path(value)
        return self._io_open(value, *args, **kwargs)

    def _guarded_os_open(self, value: object, *args: object, **kwargs: object) -> int:
        self._deny_path(value)
        return self._os_open(value, *args, **kwargs)

    def _guarded_import(
        self,
        name: str,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        resolved = self._resolved_name(name, globals, level)
        candidates = (
            resolved,
            *(f"{resolved}.{item}" for item in (fromlist or ()) if item != "*"),
        )
        if any(self._forbidden_module(candidate) for candidate in candidates):
            if self._probing:
                self.negative_control_denials += 1
            else:
                self.denied_imports += 1
            raise ImportError("compact reconstruction denies image/dense-pipeline imports")
        return self._import(name, globals, locals, fromlist, level)

    def _guarded_import_module(self, name: str, package: str | None = None) -> Any:
        try:
            resolved = importlib_util.resolve_name(name, package) if name.startswith(".") else name
        except (ImportError, ValueError):
            resolved = name
        if self._forbidden_module(resolved):
            if self._probing:
                self.negative_control_denials += 1
            else:
                self.denied_imports += 1
            raise ImportError("compact reconstruction denies image/dense-pipeline imports")
        return self._import_module(name, package)

    def __enter__(self) -> NoImageGuard:
        loaded = sorted(name for name in sys.modules if self._forbidden_module(name))
        if loaded:
            raise RuntimeError(f"forbidden modules loaded before compact boundary: {loaded}")
        builtins.open = self._guarded_open
        io.open = self._guarded_io_open
        os.open = self._guarded_os_open
        builtins.__import__ = self._guarded_import
        importlib.import_module = self._guarded_import_module
        self._probing = True
        try:
            with contextlib.suppress(PermissionError):
                builtins.open(ROOT / "negative-control.png", "rb")
            with contextlib.suppress(ImportError):
                builtins.__import__("PIL.Image")
            if self.negative_control_expected == 4:
                with contextlib.suppress(ImportError):
                    importlib.import_module("rtgs.data.calibrated")
                with contextlib.suppress(ImportError):
                    importlib.import_module("rtgs.optim.trainer")
        finally:
            self._probing = False
        if self.negative_control_denials != self.negative_control_expected:
            self.__exit__()
            raise RuntimeError("compact reconstruction negative controls did not all fire")
        return self

    def __exit__(self, *exc: object) -> None:
        builtins.open = self._open
        io.open = self._io_open
        os.open = self._os_open
        builtins.__import__ = self._import
        importlib.import_module = self._import_module

    def record(self) -> dict[str, object]:
        loaded = sorted(name for name in sys.modules if self._forbidden_module(name))
        return {
            "passed": (
                not loaded
                and self.denied_paths == 0
                and self.denied_imports == 0
                and self.negative_control_denials == self.negative_control_expected
            ),
            "forbidden_modules_at_exit": loaded,
            "image_open_attempts": self.denied_paths,
            "forbidden_import_attempts": self.denied_imports,
            "negative_control_denials": self.negative_control_denials,
            "negative_control_expected": self.negative_control_expected,
        }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False).encode("utf-8"))
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _task_and_dataset(task_path: Path, dataset_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    task = _load_json(task_path.resolve())
    if task.get("task_id") != TASK_ID or task.get("status") != "draft":
        raise ValueError("development driver requires the matching draft task")
    matches = [dataset for dataset in task["datasets"] if dataset["id"] == dataset_id]
    if len(matches) != 1:
        raise ValueError(f"task does not contain exactly one dataset {dataset_id!r}")
    return task, matches[0]


def _byte_cap(task: dict[str, Any]) -> int:
    return int(task["frozen_configuration"]["stage1_native_fullres"]["byte_cap"])


def _subset_inputs(inputs: Any, names: list[str]) -> Any:
    from rtgs.data.reconstruction_inputs import ReconstructionInputs

    lookup = {name: index for index, name in enumerate(inputs.view_names)}
    if len(lookup) != len(inputs.view_names) or any(name not in lookup for name in names):
        raise ValueError("frozen split differs from the compact manifest")
    indices = [lookup[name] for name in names]
    return ReconstructionInputs(
        observations=[inputs.observations[index] for index in indices],
        cameras=[inputs.cameras[index] for index in indices],
        view_names=list(names),
        points=None,
        point_visibility=None,
        bounds_hint=inputs.bounds_hint,
        name=f"{inputs.name}-{'-'.join(names[:1])}-{len(names)}views",
        archive_stats=None,
    )


def _load_split_inputs(
    task: dict[str, Any],
    dataset: dict[str, Any],
) -> tuple[Any, Any, Any]:
    from rtgs.data.compact_views import CompactDataset

    directory = (ROOT / dataset["compact_manifest"]).parent
    compact = CompactDataset.load(
        directory,
        device="cpu",
        byte_cap=_byte_cap(task),
        load_alpha=False,
    )
    all_inputs = compact.to_reconstruction_inputs()
    split = task["splits"][dataset["id"]]
    train = _subset_inputs(all_inputs, split["train"])
    heldout = _subset_inputs(all_inputs, split["heldout"])
    if set(train.view_names) & set(heldout.view_names):
        raise RuntimeError("train and held-out compact inputs overlap")
    return compact, train, heldout


def _initialize(args: argparse.Namespace) -> int:
    task, dataset = _task_and_dataset(args.task, args.dataset_id)
    output = args.output.resolve() / args.dataset_id / f"seed_{args.seed}" / "initializations"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite initialization output: {output}")
    output.mkdir(parents=True)
    guard = NoImageGuard()
    with guard:
        import numpy as np
        import torch

        from rtgs.lift.beam_fusion import BeamFusionConfig
        from rtgs.lift.compact_carve import _center_and_extent
        from rtgs.lift.paper_initializers import (
            PaperInitializerConfig,
            build_matched_paper_initializations,
        )
        from rtgs.lift.splat_sfm import SplatSfMConfig

        _compact, train, _heldout = _load_split_inputs(task, dataset)
        common_center, common_extent = _center_and_extent(train, torch.float64)
        config = PaperInitializerConfig(
            random_seed=args.seed,
            random_bounds_scale=INITIALIZER_CONFIG["random_bounds_scale"],
            init_opacity=INITIALIZER_CONFIG["init_opacity"],
            max_starting_gaussians=INITIALIZER_CONFIG["max_starting_gaussians"],
            structural_components_per_view=INITIALIZER_CONFIG["structural_components_per_view"],
            sfm=SplatSfMConfig(
                init_opacity=INITIALIZER_CONFIG["init_opacity"],
                **INITIALIZER_CONFIG["splat_sfm"],
            ),
            beam=BeamFusionConfig(
                init_opacity=INITIALIZER_CONFIG["init_opacity"],
                **INITIALIZER_CONFIG["beam_fusion"],
            ),
        )
        started = time.perf_counter()
        result = build_matched_paper_initializations(train, config)
        elapsed = time.perf_counter() - started
        models = {
            "bounded_random": result.bounded_random,
            "splat_sfm": result.splat_sfm,
            "beam_fusion": result.beam_fusion,
        }
        for arm, model in models.items():
            arm_directory = output.parent / arm
            arm_directory.mkdir(parents=True)
            model.save_npz(arm_directory / "gaussians_init.npz")
            model.save_ply(arm_directory / "gaussians_init.ply")
        np.savez_compressed(
            output / "initializer_lineage.npz",
            splat_sfm_selected_rows=result.splat_sfm_selected_rows.numpy(),
            beam_fusion_selected_rows=result.beam_fusion_selected_rows.numpy(),
            splat_sfm_track_offsets=result.splat_sfm_result.track_offsets.numpy(),
            splat_sfm_member_view_indices=(result.splat_sfm_result.member_view_indices.numpy()),
            splat_sfm_member_component_indices=(
                result.splat_sfm_result.member_component_indices.numpy()
            ),
            beam_component_offsets=result.beam_fusion_result.component_offsets.numpy(),
            beam_contributor_view_indices=(
                result.beam_fusion_result.contributor_view_indices.numpy()
            ),
            beam_contributor_component_indices=(
                result.beam_fusion_result.contributor_component_indices.numpy()
            ),
            beam_component_weights=result.beam_fusion_result.component_weights.numpy(),
        )
        guard_record = guard.record()
        if not guard_record["passed"]:
            raise RuntimeError(f"initialization input boundary failed: {guard_record}")
        _write_json_atomic(
            output / "receipt.json",
            {
                "schema": "rtgs.paper_three_path_initialization.v1",
                "evidence_status": "mechanism_only_task_still_draft",
                "task_id": TASK_ID,
                "dataset_id": args.dataset_id,
                "seed": args.seed,
                "train_views": train.view_names,
                "heldout_views_excluded": task["splits"][args.dataset_id]["heldout"],
                "config": INITIALIZER_CONFIG,
                "elapsed_seconds": elapsed,
                "common_count": result.count,
                "common_training_geometry": {
                    "policy": "camera_axis_fallback_from_frozen_train_cameras",
                    "center": common_center.tolist(),
                    "extent": common_extent,
                },
                "initializer_receipt": result.receipt,
                "input_boundary": guard_record,
                "torch_seed": torch.initial_seed(),
            },
        )
    print(f"three exact-count initializations ({result.count:,}) -> {output.parent}")
    return 0


def _query_indexes(inputs: Any) -> list[Any]:
    from rtgs.core.observation2d_cuda import GaussianObservationIndexCuda

    return [
        GaussianObservationIndexCuda.from_field(
            field,
            **QUERY_INDEX_CONFIG,
            device=field.device,
        )
        for field in inputs.observations
    ]


def _sampled_evaluation(
    inputs: Any,
    model: Any,
    *,
    seed: int,
    samples_per_view: int,
) -> dict[str, Any]:
    import torch

    from rtgs.render.gsplat_points import GsplatPointRasterizer

    indexes = _query_indexes(inputs)
    renderer = GsplatPointRasterizer(
        antialiased=POINT_RENDER_CONFIG["antialiased"],
        kernel_support_mode=POINT_RENDER_CONFIG["kernel_support_mode"],
    )
    per_view = []
    with torch.no_grad():
        for view_index, (field, camera, index) in enumerate(
            zip(inputs.observations, inputs.cameras, indexes, strict=True)
        ):
            generator = torch.Generator(device=model.means.device).manual_seed(
                seed + 10_000 + view_index
            )
            unit = torch.rand(
                samples_per_view,
                2,
                generator=generator,
                device=model.means.device,
                dtype=model.means.dtype,
            )
            fit_x, fit_y, fit_width, fit_height = field.fit_window
            xy = unit * unit.new_tensor([fit_width, fit_height])
            xy = xy + unit.new_tensor([fit_x, fit_y])
            target = index.query(xy).color
            predicted = renderer.render_points(model, camera, xy).color
            mse = float((predicted - target).square().mean())
            per_view.append(
                {
                    "view_id": inputs.view_names[view_index],
                    "samples": samples_per_view,
                    "uniform_fit_window_mse": mse,
                }
            )
    return {
        "schema": "rtgs.sampled_compact_evaluation.v1",
        "samples_per_view": samples_per_view,
        "equal_view_uniform_fit_window_mse": sum(
            item["uniform_fit_window_mse"] for item in per_view
        )
        / len(per_view),
        "per_view": per_view,
    }


def _crop_camera(camera: Any, fit_window: tuple[int, int, int, int]) -> Any:
    from rtgs.core.camera import Camera

    fit_x, fit_y, width, height = fit_window
    return Camera(
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx - fit_x,
        cy=camera.cy - fit_y,
        width=width,
        height=height,
        R=camera.R,
        t=camera.t,
    )


def _materialize_native_training_scene(
    task: dict[str, Any],
    dataset: dict[str, Any],
    *,
    device: str,
) -> tuple[Any, list[dict[str, Any]]]:
    """Replay sealed native fields into ordinary 3DGS crop tensors without source images."""

    import torch

    from rtgs.data.compact_views import CompactDataset
    from rtgs.data.scene import SceneData
    from rtgs.image2gs.native_observation import render_native_observation_crop

    compact_directory = (ROOT / dataset["compact_manifest"]).parent
    compact = CompactDataset.load(
        compact_directory,
        device="cpu",
        byte_cap=_byte_cap(task),
        load_alpha=True,
    )
    split = task["splits"][dataset["id"]]
    ordered_names = [*split["train"], *split["heldout"]]
    lookup = {view.view_id: view for view in compact.views}
    if len(lookup) != len(compact.views) or any(name not in lookup for name in ordered_names):
        raise RuntimeError("standard 3DGS scene differs from the frozen compact split")

    images = []
    masks = []
    cameras = []
    records = []
    target_device = torch.device(device)
    for index, name in enumerate(ordered_names, start=1):
        view = lookup[name]
        if view.alpha is None:
            raise RuntimeError(f"native field {name} has no bundled alpha")
        observation = view.observation.to(target_device)
        started = time.perf_counter()
        target = render_native_observation_crop(
            observation,
            renderer="cuda",
            row_chunk=64,
        )
        alpha = view.alpha.crop_mask(target_device).float()
        if target.shape[:2] != alpha.shape:
            raise RuntimeError(f"native target and alpha differ for {name}")
        unclamped_min = float(target.min())
        unclamped_max = float(target.max())
        target = (target * alpha[..., None]).clamp(0.0, 1.0).float()
        images.append(target.cpu())
        masks.append(alpha.cpu())
        cameras.append(_crop_camera(view.camera, observation.fit_window))
        records.append(
            {
                "view_id": name,
                "provider": observation.provider,
                "blend_mode": observation.blend_mode,
                "components": observation.n,
                "fit_window": observation.fit_window,
                "shape": list(target.shape),
                "unclamped_min": unclamped_min,
                "unclamped_max": unclamped_max,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        print(
            f"native teacher {index:02d}/{len(ordered_names):02d}: "
            f"{name} {target.shape[1]}x{target.shape[0]}",
            flush=True,
        )
        del observation, target, alpha
        torch.cuda.empty_cache()

    train_count = len(split["train"])
    scene = SceneData(
        images=images,
        cameras=cameras,
        view_names=ordered_names,
        masks=masks,
        train_indices=list(range(train_count)),
        test_indices=list(range(train_count, len(ordered_names))),
        bounds_hint=compact.bounds_hint,
        name=f"{compact.name}-native-additive-standard-3dgs",
    )
    scene.validate()
    return scene, records


def _train_standard(args: argparse.Namespace) -> int:
    """Run the established full-image 3DGS recipe on native field replays."""

    if args.arm not in ARMS:
        raise ValueError(f"unknown arm {args.arm!r}")
    if args.iterations <= 0:
        raise ValueError("standard 3DGS iterations must be positive")
    task, dataset = _task_and_dataset(args.task, args.dataset_id)
    cell = args.output.resolve() / args.dataset_id / f"seed_{args.seed}" / args.arm
    init_path = cell / "gaussians_init.npz"
    if not init_path.is_file():
        raise FileNotFoundError(f"run initialize first: {init_path}")
    final_path = cell / "gaussians.ply"
    if final_path.exists():
        raise FileExistsError(f"refusing to overwrite trained endpoint: {final_path}")
    checkpoints = cell / "checkpoints"
    checkpoints.mkdir()

    guard = NoImageGuard(allow_generated_dense=True)
    with guard:
        import torch

        from rtgs.core.gaussians3d import Gaussians3D
        from rtgs.optim.density import DensityConfig
        from rtgs.optim.trainer import TrainConfig, Trainer
        from rtgs.render.base import get_rasterizer

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        materialize_started = time.perf_counter()
        scene, target_records = _materialize_native_training_scene(
            task,
            dataset,
            device=STANDARD_3DGS_CONFIG["device"],
        )
        materialize_seconds = time.perf_counter() - materialize_started
        init = Gaussians3D.load_npz(init_path)
        density = DensityConfig(
            start_iter=STANDARD_3DGS_CONFIG["densify_start"],
            stop_iter=STANDARD_3DGS_CONFIG["densify_stop"],
            every=STANDARD_3DGS_CONFIG["densify_every"],
            grad_threshold=STANDARD_3DGS_CONFIG["grad_threshold"],
            absgrad=True,
            prune_opacity=STANDARD_3DGS_CONFIG["prune_opacity"],
            prune_scale_frac=STANDARD_3DGS_CONFIG["prune_scale_frac"],
            max_gaussians=STANDARD_3DGS_CONFIG["max_gaussians"],
        )
        train_config = TrainConfig(
            iterations=args.iterations,
            eval_every=STANDARD_3DGS_CONFIG["eval_every"],
            rasterizer=STANDARD_3DGS_CONFIG["rasterizer"],
            device=STANDARD_3DGS_CONFIG["device"],
            density_strategy=STANDARD_3DGS_CONFIG["density_strategy"],
            density=density,
            target_sh_degree=STANDARD_3DGS_CONFIG["target_sh_degree"],
            sh_degree_interval=STANDARD_3DGS_CONFIG["sh_degree_interval"],
            use_masks=STANDARD_3DGS_CONFIG["use_masks"],
            random_background=STANDARD_3DGS_CONFIG["random_background"],
            packed=STANDARD_3DGS_CONFIG["packed"],
            antialiased=STANDARD_3DGS_CONFIG["antialiased"],
            validate_render_finite=True,
            stream_scene_from_cpu=STANDARD_3DGS_CONFIG["stream_scene_from_cpu"],
            seed=args.seed,
        )
        saved_checkpoints = set(STANDARD_3DGS_CONFIG["checkpoints"])

        def checkpoint_callback(snapshot: Any, step: int) -> None:
            if step in saved_checkpoints:
                snapshot.save_ply(checkpoints / f"gaussians_step_{step:06d}.ply")

        started = time.perf_counter()
        final, history = Trainer(train_config).train(
            scene,
            init,
            checkpoint_callback=checkpoint_callback,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        final.save_npz(cell / "gaussians.npz")
        final.save_ply(final_path)
        renderer = get_rasterizer(
            "gsplat",
            device="cuda",
            packed=True,
            antialiased=True,
        )
        final_cuda = final.to("cuda")
        train_metrics = Trainer.evaluate_metrics(
            scene,
            final_cuda,
            renderer,
            indices=scene.training_views,
        )
        heldout_metrics = Trainer.evaluate_metrics(
            scene,
            final_cuda,
            renderer,
            indices=scene.testing_views,
        )
        structsplat_modules = sorted(
            name for name in sys.modules if name == "structsplat" or name.startswith("structsplat.")
        )
        if structsplat_modules:
            raise RuntimeError(
                f"standard native-field replay unexpectedly loaded StructSplat: "
                f"{structsplat_modules}"
            )
        guard_record = guard.record()
        if not guard_record["passed"]:
            raise RuntimeError(f"standard training input boundary failed: {guard_record}")
        _write_json_atomic(cell / "training_history.json", history)
        density_events = history.get("density_stats", [])
        _write_json_atomic(
            cell / "summary.json",
            {
                "schema": "rtgs.paper_three_path_standard_3dgs.v1",
                "evidence_status": "development_task_still_draft",
                "task_id": TASK_ID,
                "dataset_id": args.dataset_id,
                "seed": args.seed,
                "arm": args.arm,
                "trainer": "ordinary_full_crop_3dgs_from_native_field_replay",
                "source_rgb_opened": False,
                "structsplat_used": False,
                "structsplat_modules_at_exit": structsplat_modules,
                "train_views": [scene.view_names[index] for index in scene.training_views],
                "heldout_views": [scene.view_names[index] for index in scene.testing_views],
                "initial_gaussians": init.n,
                "final_gaussians": final.n,
                "completed_iterations": args.iterations,
                "elapsed_seconds": elapsed,
                "teacher_materialization_seconds": materialize_seconds,
                "teacher_materialization": target_records,
                "train_metrics": train_metrics,
                "heldout_metrics": heldout_metrics,
                "topology_control": {
                    "schema": "rtgs.gsplat_default_density.v1",
                    "policy": "DefaultStrategy clone/split/prune/opacity-reset",
                    "events": density_events,
                    "operation_counts_available": False,
                },
                "standard_3dgs_config": {
                    **STANDARD_3DGS_CONFIG,
                    "iterations": args.iterations,
                },
                "density_config": dataclasses.asdict(density),
                "input_boundary": guard_record,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            },
        )
    print(f"{args.arm} standard 3DGS: {init.n:,} -> {final.n:,} in {elapsed:.1f}s")
    return 0


def _train(args: argparse.Namespace) -> int:
    if args.arm not in ARMS:
        raise ValueError(f"unknown arm {args.arm!r}")
    task, dataset = _task_and_dataset(args.task, args.dataset_id)
    cell = args.output.resolve() / args.dataset_id / f"seed_{args.seed}" / args.arm
    init_path = cell / "gaussians_init.npz"
    if not init_path.is_file():
        raise FileNotFoundError(f"run initialize first: {init_path}")
    final_path = cell / "gaussians.ply"
    if final_path.exists():
        raise FileExistsError(f"refusing to overwrite trained endpoint: {final_path}")
    checkpoints = cell / "checkpoints"
    checkpoints.mkdir()
    guard = NoImageGuard()
    with guard:
        import torch

        from rtgs.core.gaussians3d import Gaussians3D
        from rtgs.lift.compact_carve import _center_and_extent
        from rtgs.optim.compact_density import ClassicCompactDensityController
        from rtgs.optim.compact_trainer import CompactTrainConfig, CompactTrainer
        from rtgs.optim.density import DensityConfig
        from rtgs.render.gsplat_points import GsplatPointRasterizer

        _compact, train_cpu, heldout_cpu = _load_split_inputs(task, dataset)
        _common_center, common_extent = _center_and_extent(train_cpu, torch.float64)
        train = train_cpu.to("cuda")
        init = Gaussians3D.load_npz(init_path)
        train_config = CompactTrainConfig(
            seed=args.seed,
            extent=common_extent,
            **TRAIN_CONFIG,
        )
        density_config = DensityConfig(**DENSITY_CONFIG)
        trainer = CompactTrainer(
            train_config,
            point_rasterizer=GsplatPointRasterizer(
                absgrad=density_config.absgrad,
                antialiased=POINT_RENDER_CONFIG["antialiased"],
                kernel_support_mode=POINT_RENDER_CONFIG["kernel_support_mode"],
            ),
        )
        train_indexes = _query_indexes(train)

        def checkpoint_callback(snapshot: Any, step: int) -> None:
            snapshot.save_ply(checkpoints / f"gaussians_step_{step:06d}.ply")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        final, history = trainer.train(
            train,
            init,
            query_backends=train_indexes,
            proposal_query_backends=train_indexes,
            checkpoint_callback=checkpoint_callback,
            topology_controller=ClassicCompactDensityController(
                density_config,
                seed=args.seed,
            ),
            stop_after_step=args.stop_after_step,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        final.save_npz(cell / "gaussians.npz")
        final.save_ply(final_path)
        heldout = heldout_cpu.to("cuda")
        sampled = _sampled_evaluation(
            heldout,
            final.to("cuda"),
            seed=args.seed,
            samples_per_view=args.evaluation_samples,
        )
        guard_record = guard.record()
        if not guard_record["passed"]:
            raise RuntimeError(f"training input boundary failed: {guard_record}")
        _write_json_atomic(cell / "training_history.json", history)
        _write_json_atomic(
            cell / "summary.json",
            {
                "schema": "rtgs.paper_three_path_training.v1",
                "evidence_status": "mechanism_only_task_still_draft",
                "task_id": TASK_ID,
                "dataset_id": args.dataset_id,
                "seed": args.seed,
                "arm": args.arm,
                "train_views": train.view_names,
                "heldout_views": heldout.view_names,
                "initial_gaussians": init.n,
                "final_gaussians": final.n,
                "completed_iterations": history.get(
                    "completed_iterations",
                    TRAIN_CONFIG["iterations"],
                ),
                "elapsed_seconds": elapsed,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "topology_control": history["topology_control"],
                "sampled_heldout_evaluation": sampled,
                "train_config": TRAIN_CONFIG,
                "common_training_extent": common_extent,
                "density_config": DENSITY_CONFIG,
                "query_index_config": QUERY_INDEX_CONFIG,
                "point_render_config": POINT_RENDER_CONFIG,
                "input_boundary": guard_record,
                "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            },
        )
    print(f"{args.arm}: {init.n:,} -> {final.n:,} Gaussians in {elapsed:.1f}s")
    return 0


def _present(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    from PIL import Image

    from rtgs.core.gaussians3d import Gaussians3D
    from rtgs.data.compact_views import CompactDataset
    from rtgs.render.base import get_rasterizer

    task, dataset = _task_and_dataset(args.task, args.dataset_id)
    base = args.output.resolve() / args.dataset_id / f"seed_{args.seed}"
    presentation = base / "presentation"
    if presentation.exists():
        raise FileExistsError(f"refusing to overwrite presentation: {presentation}")
    presentation.mkdir()
    compact_directory = (ROOT / dataset["compact_manifest"]).parent
    scene_directory = (ROOT / dataset["frame_path"]).resolve()
    compact = CompactDataset.load(
        compact_directory,
        byte_cap=_byte_cap(task),
        load_alpha=False,
    )
    lookup = {view.view_id: view for view in compact.views}
    if args.view_id not in lookup:
        raise ValueError(f"unknown presentation view {args.view_id!r}")
    split = task["splits"][args.dataset_id]
    view_role = (
        "Trainingskamera" if args.view_id in split["train"] else "Held-out-Kamera (nicht optimiert)"
    )
    camera = lookup[args.view_id].camera.to("cuda")
    renderer = get_rasterizer(
        "gsplat",
        device="cuda",
        packed=True,
        antialiased=STANDARD_3DGS_CONFIG["antialiased"],
    )
    viewer_methods = []
    cards = []
    for arm in ARMS:
        directory = base / arm
        initial_path = directory / "gaussians_init.ply"
        final_path = directory / "gaussians.ply"
        if not initial_path.is_file() or not final_path.is_file():
            raise FileNotFoundError(f"missing {arm} initial/final model")
        state_images = {}
        for state, model_path in (("initial", initial_path), ("final", final_path)):
            model = Gaussians3D.load_ply(model_path).to("cuda")
            with torch.no_grad():
                color = renderer.render(model, camera).color.clamp(0.0, 1.0)
            image_path = presentation / f"{arm}_{args.view_id}_{state}_native.png"
            Image.fromarray((color.cpu().numpy() * 255).round().astype(np.uint8)).save(image_path)
            state_images[state] = image_path.name
            del model, color
            torch.cuda.empty_cache()
        summary = _load_json(directory / "summary.json")
        density_events = summary["topology_control"]["events"]
        has_operation_counts = all(
            {"cloned", "split", "pruned"} <= set(event) for event in density_events
        )
        if has_operation_counts:
            density_detail = (
                f"Clone {sum(event['cloned'] for event in density_events):,} · "
                f"Split {sum(event['split'] for event in density_events):,} · "
                f"Prune {sum(event['pruned'] for event in density_events):,} · "
                f"Opacity reset {sum(bool(event['opacity_reset']) for event in density_events):,}"
            )
        else:
            density_detail = (
                "gsplat DefaultStrategy: clone + split + prune + opacity reset "
                "(dynamic backend exposes count transitions, not per-operation totals)"
            )
        cards.append(
            {
                "arm": arm,
                "label": ARM_LABELS[arm],
                "initial_image": state_images["initial"],
                "final_image": state_images["final"],
                "initial": summary["initial_gaussians"],
                "final": summary["final_gaussians"],
                "events": len(density_events),
                "density_detail": density_detail,
                "width": camera.width,
                "height": camera.height,
            }
        )
        viewer_methods.append(
            {
                "name": ARM_LABELS[arm],
                "initial": os.path.relpath(initial_path, presentation),
                "final": os.path.relpath(final_path, presentation),
            }
        )
    _write_json_atomic(
        presentation / "viewer_comparison.json",
        {"schema": "rtgs.viewer-comparison.v1", "methods": viewer_methods},
    )
    field_path = compact_directory / "qa" / f"{args.view_id}_field_native_full_canvas.png"
    field_href = os.path.relpath(field_path, presentation)
    card_html = "".join(
        f"""<article><h2>{card["label"]}</h2>
<div class="pair"><div><h3>Initialisierung</h3>
<a href="{card["initial_image"]}"><img src="{card["initial_image"]}"
alt="{card["arm"]} initialization"></a></div>
<div><h3>Endzustand · 30.000</h3>
<a href="{card["final_image"]}"><img src="{card["final_image"]}"
alt="{card["arm"]} final render"></a></div></div>
<p>{card["initial"]:,} → {card["final"]:,} Gaussians ·
{card["events"]} density transactions</p>
<p>{card["density_detail"]} ·
<a href="{card["initial_image"]}">Initialisierung</a> und
<a href="{card["final_image"]}">Endzustand</a> mit jeweils
{card["width"]}×{card["height"]} nativen Pixeln</p></article>"""
        for card in cards
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RTGS: drei native Full-Resolution-Pfade</title><style>
body{{font-family:system-ui;background:#111;color:#eee;margin:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:20px}}
article{{background:#1d1d1d;padding:14px;border-radius:8px}} img{{width:100%;height:auto}}
a{{color:#8cc8ff}} .boundary{{border-left:4px solid #e6a23c;padding-left:12px}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.pipeline{{display:grid;grid-template-columns:minmax(220px,1fr) auto minmax(240px,1fr) auto
minmax(260px,1.2fr);gap:12px;align-items:stretch;margin:22px 0}}
.node{{background:#1d1d1d;border:1px solid #444;border-radius:8px;padding:14px}}
.node strong{{display:block;margin-bottom:6px}}
.arrow{{align-self:center;font-size:28px;color:#8cc8ff}}
.branches{{display:grid;gap:8px}} .branches div{{background:#252525;border-radius:6px;padding:8px}}
@media(max-width:980px){{.pipeline{{grid-template-columns:1fr}}.arrow{{transform:rotate(90deg);
justify-self:center}}.pair{{grid-template-columns:1fr}}}}
</style></head><body><h1>Native 2D-Gaussian-Felder → drei vollständig gefittete 3DGS-Pfade</h1>
<p class="boundary">Entwicklungsnachweis, kein freigegebenes Experimentresultat. Splat-SfM ist
der aus den 2D-Feldern abgeleitete Strukturpfad; für diese Aufnahme existiert kein COLMAP-Modell.
Die Ergebniskarten verwenden Native-Field-Replay → gewöhnliches 3DGS und sind deshalb kein
Compact-/VRAM-Nachweis.</p>
<section class="pipeline" aria-label="current code pipeline">
<div class="node"><strong>1 · Aufnahme</strong>RGB + Maske + kalibrierte Kameras<br>
5328×4608 natives Canvas</div><div class="arrow">→</div>
<div class="node"><strong>2 · Eingefrorener 2D-Teacher</strong>100.000 native additive
Gaussians/Kamera<br>2.000 Fit-Updates · kein StructSplat</div><div class="arrow">→</div>
<div class="node"><strong>3 · Drei Initialisierungen</strong><div class="branches">
<div>Bounded Random</div><div>Splat-SfM (feldbasiert)</div><div>Beam Fusion (Tomografie)</div>
</div></div>
</section>
<section class="pipeline" aria-label="shared downstream fit">
<div class="node"><strong>Nur die Initialisierung unterscheidet sich</strong>Exakt gleiche
Startanzahl und gleicher, kamerabasierter Trainingsraum</div><div class="arrow">→</div>
<div class="node"><strong>4 · Gemeinsamer Standard-3DGS-Fit</strong>Native Field-Replays,
kein Source-RGB · 30.000 Full-Crop-Updates · SH-Grad 3</div><div class="arrow">→</div>
<div class="node"><strong>5 · Volle Densification</strong>DefaultStrategy:
clone + split + prune · Opacity-Reset bis Schritt 15.000 · Hard-Cap 100.000 ·
Render in nativer Auflösung</div>
</section>
<p class="boundary"><strong>Separater geschützter Compact-Pfad:</strong> direkte indizierte
Field-Queries → gepackte Ein-Pixel-gsplat-Kameras → CompactTrainer → klassische Density-Control.
Sein erster 10k-Entwicklungslauf erreichte 100.000 Splats, fiel aber in der nativen visuellen
Prüfung als dunkel/weich durch. Dieser Pfad bleibt blockiert und wird durch die Karten unten
nicht stillschweigend ersetzt.</p>
<article><h2>Eingefrorenes additives 2D-Feld · {args.view_id} · {view_role}</h2>
<a href="{field_href}"><img src="{field_href}" alt="2D Gaussian field"></a>
<p><a href="{field_href}">{camera.width}×{camera.height} native Pixel öffnen</a> ·
100.000 Gaussians · kein StructSplat. Der Browser skaliert die Vorschau nur ans Fenster;
der Link selbst bleibt in voller Dateiauflösung.</p></article>
<div class="grid">{card_html}</div>
<p>Interaktiver Modellumschalter mit allen kalibrierten Kameras in voller Auflösung:
<code>rtgs view --comparison-manifest {presentation / "viewer_comparison.json"}
--scene {scene_directory} --downscale 1 --rasterizer gsplat --packed --antialiased
--device cuda:0 --open</code></p>
</body></html>"""
    (presentation / "index.html").write_text(html, encoding="utf-8")
    print(f"presentation -> {presentation / 'index.html'}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser("initialize")
    train = subparsers.add_parser("train")
    train_standard = subparsers.add_parser("train-standard")
    present = subparsers.add_parser("present")
    for subparser in (initialize, train, train_standard, present):
        subparser.add_argument("--task", type=Path, default=DEFAULT_TASK)
        subparser.add_argument("--dataset-id", default="frame_00008")
        subparser.add_argument("--seed", type=int, default=300701)
        subparser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    train.add_argument("--arm", choices=ARMS, required=True)
    train.add_argument("--stop-after-step", type=int)
    train.add_argument("--evaluation-samples", type=int, default=256)
    train_standard.add_argument("--arm", choices=ARMS, required=True)
    train_standard.add_argument(
        "--iterations",
        type=int,
        default=STANDARD_3DGS_CONFIG["iterations"],
    )
    present.add_argument("--view-id", default="C0014")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "initialize":
        return _initialize(args)
    if args.command == "train":
        return _train(args)
    if args.command == "train-standard":
        return _train_standard(args)
    if args.command == "present":
        return _present(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
