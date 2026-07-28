#!/usr/bin/env python3
"""Compact-only Beam/carrier stage ablation on frame 00008.

The executable boundary is deliberately narrower than the older carrier-maturation harness:
every target is queried directly from a frozen 2D Gaussian field, and no RGB, alpha, mask, dense
teacher replay, ``SceneData``, or dense-image trainer is reachable.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import hashlib
import importlib
import io
import json
import math
import os
import platform
import resource
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianPointProposal, ObservationSamples
from rtgs.data.compact_views import CompactDataset
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.beam_fusion import BeamFusionConfig, BeamFusionResult, fuse_gaussian_beams
from rtgs.lift.carrier_refinement import CarrierRepairConfig, repair_beam_carriers
from rtgs.optim.compact_trainer import CompactTrainConfig, CompactTrainer
from rtgs.render.projection import EWA_NEAR
from rtgs.render.torch_points import TorchPointRasterizer

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d"
PREREGISTRATION = ROOT / "benchmarks/results/20260728_compact_only_carrier_stage_ablation_PREREG.md"
DEFAULT_OUT = ROOT / "runs/compact_only_carrier_stage_ablation_20260728"

TRAIN_ROOTS = (280701, 280702, 280703)
EVALUATION_ROOTS = (281701, 281702, 281703)
HELDOUT_INDICES = (7, 15, 23)
ITERATIONS = 380
ATTEMPTS = 256
EVALUATION_ATTEMPTS = 8192
CHECKPOINTS = (0, 95, 190, 380)
N_CARRIERS = 5000
IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".exr", ".gif"}
)


@dataclass(frozen=True)
class Arm:
    repair_key: str
    trainable: tuple[str, ...]
    support: bool = False


ALL_FAMILIES = ("means", "quats", "scales", "opacities", "sh0")
ARMS: dict[str, Arm] = {
    "beam_all": Arm("beam", ALL_FAMILIES),
    "legacy_C_all": Arm("legacy_C", ALL_FAMILIES),
    "legacy_O_all": Arm("legacy_O", ALL_FAMILIES),
    "legacy_A_all": Arm("legacy_A", ALL_FAMILIES),
    "legacy_CO_all": Arm("legacy_CO", ALL_FAMILIES),
    "legacy_CA_all": Arm("legacy_CA", ALL_FAMILIES),
    "legacy_OA_all": Arm("legacy_OA", ALL_FAMILIES),
    "legacy_COA_all": Arm("legacy_COA", ALL_FAMILIES),
    "corrected_C_all": Arm("corrected_C", ALL_FAMILIES),
    "corrected_CA_all": Arm("corrected_CA", ALL_FAMILIES),
    "beam_means_only": Arm("beam", ("means",)),
    "beam_means_fixed": Arm(
        "beam",
        ("quats", "scales", "opacities", "sh0"),
    ),
    "beam_geometry_only": Arm("beam", ("means", "quats", "scales")),
    "beam_appearance_only": Arm("beam", ("opacities", "sh0")),
    "beam_all_support": Arm("beam", ALL_FAMILIES, support=True),
    "corrected_CA_all_support": Arm(
        "corrected_CA",
        ALL_FAMILIES,
        support=True,
    ),
}

REPAIR_SPECS: dict[str, dict[str, str | bool]] = {
    "legacy_C": {"covariance": "legacy", "opacity": False, "appearance": "none"},
    "legacy_O": {"covariance": "none", "opacity": True, "appearance": "none"},
    "legacy_A": {"covariance": "none", "opacity": False, "appearance": "legacy"},
    "legacy_CO": {"covariance": "legacy", "opacity": True, "appearance": "none"},
    "legacy_CA": {"covariance": "legacy", "opacity": False, "appearance": "legacy"},
    "legacy_OA": {"covariance": "none", "opacity": True, "appearance": "legacy"},
    "legacy_COA": {"covariance": "legacy", "opacity": True, "appearance": "legacy"},
    "corrected_C": {"covariance": "corrected", "opacity": False, "appearance": "none"},
    "corrected_CA": {
        "covariance": "corrected",
        "opacity": False,
        "appearance": "corrected",
    },
}


class NoImageGuard:
    """Live denial boundary for source images and image-capable repository paths."""

    def __init__(self) -> None:
        self.denied_paths = 0
        self.denied_imports = 0
        self.negative_control_denials = 0
        self._probing = False
        self._open = builtins.open
        self._io_open = io.open
        self._os_open = os.open
        self._import = builtins.__import__
        self._import_module = importlib.import_module

    @staticmethod
    def _forbidden_module(name: str) -> bool:
        return (
            name == "PIL"
            or name.startswith("PIL.")
            or name == "cv2"
            or name.startswith("cv2.")
            or name == "imageio"
            or name.startswith("imageio.")
            or name in {"rtgs.data.calibrated", "rtgs.data.scene", "rtgs.optim.trainer"}
        )

    @staticmethod
    def _forbidden_path(value: object) -> bool:
        if isinstance(value, int):
            return False
        try:
            path = Path(os.fspath(value))
        except TypeError:
            return False
        return path.suffix.lower() in IMAGE_SUFFIXES

    def _deny_path(self, value: object) -> None:
        if self._forbidden_path(value):
            if self._probing:
                self.negative_control_denials += 1
            else:
                self.denied_paths += 1
            raise PermissionError("compact-only experiment denies every image-file open")

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
        if self._forbidden_module(name):
            if self._probing:
                self.negative_control_denials += 1
            else:
                self.denied_imports += 1
            raise ImportError("compact-only experiment denies image/dense-trainer imports")
        return self._import(name, globals, locals, fromlist, level)

    def _guarded_import_module(self, name: str, package: str | None = None) -> Any:
        if self._forbidden_module(name):
            if self._probing:
                self.negative_control_denials += 1
            else:
                self.denied_imports += 1
            raise ImportError("compact-only experiment denies image/dense-trainer imports")
        return self._import_module(name, package)

    def __enter__(self) -> NoImageGuard:
        forbidden_loaded = sorted(name for name in sys.modules if self._forbidden_module(name))
        if forbidden_loaded:
            raise RuntimeError(
                f"forbidden modules loaded before no-image boundary: {forbidden_loaded}"
            )
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
            with contextlib.suppress(ImportError):
                importlib.import_module("rtgs.optim.trainer")
        finally:
            self._probing = False
        if self.negative_control_denials != 3:
            self.__exit__()
            raise RuntimeError("no-image negative controls did not all fire")
        return self

    def __exit__(self, *exc: object) -> None:
        builtins.open = self._open
        io.open = self._io_open
        os.open = self._os_open
        builtins.__import__ = self._import
        importlib.import_module = self._import_module

    def record(self) -> dict[str, object]:
        forbidden_loaded = sorted(name for name in sys.modules if self._forbidden_module(name))
        return {
            "passed": not forbidden_loaded
            and self.denied_paths == 0
            and self.denied_imports == 0
            and self.negative_control_denials == 3,
            "forbidden_modules_at_exit": forbidden_loaded,
            "image_open_attempts": self.denied_paths,
            "forbidden_import_attempts": self.denied_imports,
            "negative_control_denials": self.negative_control_denials,
        }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _gaussians_sha256(value: Gaussians3D) -> str:
    return _canonical_hash(
        {
            name: _tensor_sha256(getattr(value, name))
            for name in ("means", "quats", "log_scales", "opacity", "sh")
        }
    )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _domain_seed(root: int, view_name: str, kind: str) -> int:
    payload = f"rtgs.compact-carrier-stage.eval.v1\0{root}\0{view_name}\0{kind}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def _split_indices(count: int) -> dict[str, tuple[int, ...]]:
    if count != 26:
        raise ValueError(f"expected 26 compact views, found {count}")
    heldout = tuple(HELDOUT_INDICES)
    pool = tuple(index for index in range(count) if index not in heldout)
    validation = tuple(pool[position] for position in range(4, len(pool), 5))
    fitting = tuple(index for index in pool if index not in validation)
    if len(fitting) != 19 or len(validation) != 4 or len(heldout) != 3:
        raise RuntimeError("camera split cardinality changed")
    return {"fit": fitting, "validation": validation, "heldout": heldout}


def _inputs(dataset: CompactDataset, indices: Sequence[int], *, name: str) -> ReconstructionInputs:
    return ReconstructionInputs(
        observations=[dataset.views[index].observation for index in indices],
        cameras=[dataset.views[index].camera for index in indices],
        view_names=[dataset.views[index].view_id for index in indices],
        bounds_hint=dataset.bounds_hint,
        name=name,
    )


def _subset(
    inputs: ReconstructionInputs, indices: Sequence[int], *, name: str
) -> ReconstructionInputs:
    return ReconstructionInputs(
        observations=[inputs.observations[index] for index in indices],
        cameras=[inputs.cameras[index] for index in indices],
        view_names=[inputs.view_names[index] for index in indices],
        bounds_hint=inputs.bounds_hint,
        name=name,
    )


def _beam_config(extent: float) -> BeamFusionConfig:
    return BeamFusionConfig(
        min_views=3,
        transverse_gate_sigma=3.0,
        max_color_distance=0.35,
        color_sigma=0.25,
        fold_in_gate_sigma=3.0,
        nms_voxel_size=extent / 100.0,
        init_opacity=0.10,
        source_chunk=256,
        max_components=N_CARRIERS,
        seed_budget_multiplier=4,
    )


def _save_beam(out: Path, beam: BeamFusionResult, seconds: float, config: BeamFusionConfig) -> None:
    beam.gaussians.save_npz(out / "beam_gaussians.npz")
    _atomic_npz(
        out / "beam_lineage.npz",
        {
            "component_offsets": beam.component_offsets.numpy(),
            "contributor_view_indices": beam.contributor_view_indices.numpy(),
            "contributor_component_indices": beam.contributor_component_indices.numpy(),
            "contributor_depths": beam.contributor_depths.numpy(),
            "component_weights": beam.component_weights.numpy(),
            "unmatched_per_view": np.asarray(beam.unmatched_per_view, dtype=np.int64),
        },
    )
    _atomic_json(
        out / "beam_receipt.json",
        {
            "config": asdict(config),
            "seconds": seconds,
            "diagnostics": beam.diagnostics,
            "gaussians_semantic_sha256": _gaussians_sha256(beam.gaussians),
            "gaussians_file_sha256": _file_sha256(out / "beam_gaussians.npz"),
            "lineage_file_sha256": _file_sha256(out / "beam_lineage.npz"),
        },
    )


def _load_beam(out: Path) -> BeamFusionResult:
    gaussians = Gaussians3D.load_npz(out / "beam_gaussians.npz")
    with np.load(out / "beam_lineage.npz", allow_pickle=False) as archive:
        return BeamFusionResult(
            gaussians=gaussians,
            component_offsets=torch.from_numpy(archive["component_offsets"].copy()),
            contributor_view_indices=torch.from_numpy(archive["contributor_view_indices"].copy()),
            contributor_component_indices=torch.from_numpy(
                archive["contributor_component_indices"].copy()
            ),
            contributor_depths=torch.from_numpy(archive["contributor_depths"].copy()),
            component_weights=torch.from_numpy(archive["component_weights"].copy()),
            unmatched_per_view=tuple(int(value) for value in archive["unmatched_per_view"]),
            diagnostics=json.loads((out / "beam_receipt.json").read_text())["diagnostics"],
        )


def _repair_config(spec: Mapping[str, str | bool]) -> CarrierRepairConfig:
    covariance = str(spec["covariance"])
    appearance = str(spec["appearance"])
    corrected_covariance = covariance == "corrected"
    return CarrierRepairConfig(
        repair_covariance=covariance != "none",
        repair_opacity=bool(spec["opacity"]),
        repair_appearance=appearance != "none",
        covariance_residual_mode=("renderer_log" if corrected_covariance else "legacy_whitened"),
        projection_dilation=0.3 if corrected_covariance else 0.0,
        appearance_weight_mode=("uniform_view" if appearance == "corrected" else "amplitude"),
    )


def _prepare_repairs(
    out: Path,
    beam: BeamFusionResult,
    fit_inputs: ReconstructionInputs,
) -> dict[str, Gaussians3D]:
    repaired: dict[str, Gaussians3D] = {"beam": beam.gaussians}
    repair_root = out / "repairs"
    repair_root.mkdir(exist_ok=True)
    for key, spec in REPAIR_SPECS.items():
        model_path = repair_root / f"{key}.npz"
        receipt_path = repair_root / f"{key}.json"
        if model_path.exists() and receipt_path.exists():
            repaired[key] = Gaussians3D.load_npz(model_path)
            continue
        print(f"[repair:{key}] {spec}", flush=True)
        config = _repair_config(spec)
        started = time.perf_counter()
        result = repair_beam_carriers(beam, fit_inputs, config)
        seconds = time.perf_counter() - started
        result.gaussians.save_npz(model_path)
        _atomic_json(
            receipt_path,
            {
                "spec": dict(spec),
                "config": asdict(config),
                "seconds": seconds,
                "diagnostics": result.diagnostics,
                "semantic_sha256": _gaussians_sha256(result.gaussians),
                "file_sha256": _file_sha256(model_path),
            },
        )
        repaired[key] = result.gaussians
    return repaired


def _sample_arrays(samples: ObservationSamples) -> dict[str, np.ndarray]:
    return {
        "xy": samples.xy.cpu().numpy(),
        "active": samples.active.cpu().numpy(),
        "inside": samples.inside_fit_window.cpu().numpy(),
        "component": samples.proposal_component_ids.cpu().numpy(),
        "proposal_density": samples.proposal_density.cpu().numpy(),
    }


def _prepare_bank(
    out: Path,
    inputs: ReconstructionInputs,
    root: int,
) -> tuple[Path, dict[str, dict[str, torch.Tensor]]]:
    path = out / "banks" / f"evaluation_{root}.npz"
    metadata_path = path.with_suffix(".json")
    if not path.exists():
        arrays: dict[str, np.ndarray] = {}
        metadata: dict[str, object] = {"root": root, "views": []}
        for view_index, (view_name, field) in enumerate(
            zip(inputs.view_names, inputs.observations, strict=True)
        ):
            proposal = GaussianPointProposal(field)
            view_record: dict[str, object] = {
                "view_index": view_index,
                "view_name": view_name,
            }
            for kind, uniform_fraction in (("uniform", 1.0), ("proposal", 0.25)):
                seed = _domain_seed(root, view_name, kind)
                samples = proposal.sample(
                    EVALUATION_ATTEMPTS,
                    uniform_fraction=uniform_fraction,
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                )
                if kind == "uniform" and not bool(samples.active.all()):
                    raise RuntimeError("uniform evaluation bank contains an inactive attempt")
                for tensor_name, value in _sample_arrays(samples).items():
                    arrays[f"v{view_index:02d}_{kind}_{tensor_name}"] = value
                view_record[kind] = {
                    "seed": seed,
                    "active_count": int(samples.active.sum()),
                    "xy_sha256": _tensor_sha256(samples.xy),
                    "active_sha256": _tensor_sha256(samples.active),
                }
            metadata["views"].append(view_record)
        _atomic_npz(path, arrays)
        metadata["file_sha256"] = _file_sha256(path)
        _atomic_json(metadata_path, metadata)
    if _file_sha256(path) != json.loads(metadata_path.read_text())["file_sha256"]:
        raise RuntimeError("evaluation bank digest changed")
    loaded: dict[str, dict[str, torch.Tensor]] = {}
    with np.load(path, allow_pickle=False) as archive:
        for view_index, view_name in enumerate(inputs.view_names):
            loaded[view_name] = {}
            for kind in ("uniform", "proposal"):
                loaded[view_name][f"{kind}_xy"] = torch.from_numpy(
                    archive[f"v{view_index:02d}_{kind}_xy"].copy()
                )
                loaded[view_name][f"{kind}_active"] = torch.from_numpy(
                    archive[f"v{view_index:02d}_{kind}_active"].copy()
                )
    return path, loaded


def _evaluate_views(
    gaussians: Gaussians3D,
    inputs: ReconstructionInputs,
    banks: Mapping[str, Mapping[str, torch.Tensor]],
    indices: Sequence[int],
) -> dict[str, object]:
    device = gaussians.means.device
    renderer = TorchPointRasterizer(point_chunk=8192, gaussian_chunk=512)
    per_view: list[dict[str, object]] = []
    with torch.no_grad():
        for index in indices:
            field = inputs.observations[index]
            camera = inputs.cameras[index]
            name = inputs.view_names[index]
            values: dict[str, float | int] = {"view_index": index, "view_name": name}
            for kind, metric in (("uniform", "J_U"), ("proposal", "J_Q")):
                xy = banks[name][f"{kind}_xy"].to(device)
                active = banks[name][f"{kind}_active"].to(device)
                target = field.query(xy, component_chunk=512).color
                prediction = renderer.render_points(
                    gaussians,
                    camera,
                    xy,
                    background=torch.zeros(3, device=device),
                    sh_degree=0,
                ).color
                point_loss = (prediction - target).square().mean(dim=-1)
                if kind == "proposal":
                    point_loss = point_loss * active.to(point_loss)
                values[metric] = float(point_loss.double().mean())
                values[f"{kind}_active_count"] = int(active.sum())
            per_view.append(values)
    return {
        "J_U": sum(float(record["J_U"]) for record in per_view) / len(per_view),
        "J_Q": sum(float(record["J_Q"]) for record in per_view) / len(per_view),
        "per_view": per_view,
    }


def _support_diagnostics(
    gaussians: Gaussians3D,
    inputs: ReconstructionInputs,
    split_indices: Mapping[str, Sequence[int]],
) -> dict[str, object]:
    per_view: list[dict[str, object]] = []
    violation_by_gaussian = torch.zeros(
        gaussians.n,
        dtype=torch.int64,
        device=gaussians.means.device,
    )
    with torch.no_grad():
        for view_index, (field, camera) in enumerate(
            zip(inputs.observations, inputs.cameras, strict=True)
        ):
            violations: list[torch.Tensor] = []
            depth_parts: list[torch.Tensor] = []
            for start in range(0, gaussians.n, 512):
                end = min(start + 512, gaussians.n)
                xy, depth = camera.project(gaussians.means[start:end])
                squared = field.minimum_mahalanobis_squared(xy, component_chunk=512)
                normalized = (
                    torch.relu(torch.sqrt(squared.clamp_min(1e-12)) - field.sigma_cutoff)
                    / field.sigma_cutoff
                )
                violations.append(normalized)
                depth_parts.append(depth)
            violation = torch.cat(violations)
            depth = torch.cat(depth_parts)
            outside = violation > 0
            violation_by_gaussian += outside
            per_view.append(
                {
                    "view_index": view_index,
                    "view_name": inputs.view_names[view_index],
                    "outside_fraction": float(outside.float().mean()),
                    "violation_mean": float(violation.mean()),
                    "violation_p95": float(torch.quantile(violation, 0.95)),
                    "behind_near_fraction": float((depth <= EWA_NEAR).float().mean()),
                }
            )

    split_records: dict[str, object] = {}
    for split, indices in split_indices.items():
        selected = [per_view[index] for index in indices]
        split_records[split] = {
            key: sum(float(record[key]) for record in selected) / len(selected)
            for key in (
                "outside_fraction",
                "violation_mean",
                "violation_p95",
                "behind_near_fraction",
            )
        }
    return {
        "per_view": per_view,
        "splits": split_records,
        "violating_view_count": {
            "mean": float(violation_by_gaussian.float().mean()),
            "median": float(violation_by_gaussian.float().median()),
            "p95": float(torch.quantile(violation_by_gaussian.float(), 0.95)),
            "max": int(violation_by_gaussian.max()),
            "all_views_clear_fraction": float((violation_by_gaussian == 0).float().mean()),
        },
    }


def _learning_rates(arm: Arm) -> dict[str, float]:
    enabled = set(arm.trainable)
    return {
        "lr_means": 1.6e-4 if "means" in enabled else 0.0,
        "lr_quats": 1e-3 if "quats" in enabled else 0.0,
        "lr_scales": 5e-3 if "scales" in enabled else 0.0,
        "lr_opacity": 5e-2 if "opacities" in enabled else 0.0,
        "lr_sh": 2.5e-3 if "sh0" in enabled else 0.0,
        "lr_sh_rest": 0.0,
    }


def _train_config(arm: Arm, seed: int, extent: float) -> CompactTrainConfig:
    return CompactTrainConfig(
        iterations=ITERATIONS,
        attempts_per_step=ATTEMPTS,
        proposal_mode="area_gaussian",
        schedule_mode="balanced_cycle",
        target_mode="proposal_attempt",
        uniform_fraction=0.25,
        seed=seed,
        extent=extent,
        device="cuda:0",
        **_learning_rates(arm),
        point_chunk=256,
        gaussian_chunk=512,
        outer_microbatch=128,
        query_component_chunk=512,
        teacher_tile_size=16,
        evaluation_chunk=8192,
        checkpoints=CHECKPOINTS,
        evaluate_checkpoint_risks=False,
        sh_degree=0,
        sh_color_activation="hard",
        kernel_support_mode="hard",
        visibility_margin_sigma=3.0,
        support_mode="teacher_ellipse_center" if arm.support else "none",
        support_loss_weight=0.01 if arm.support else 0.0,
        support_samples_per_step=256,
        support_component_chunk=512,
        support_sigma=3.0,
        support_near=EWA_NEAR,
    )


def _teacher_bytes(inputs: ReconstructionInputs) -> int:
    total = 0
    for field in inputs.observations:
        for name in (
            "means",
            "log_scales",
            "rotations",
            "colors",
            "amplitudes",
            "mean_residuals",
            "color_grads",
            "filter_variance",
        ):
            value = getattr(field, name)
            if value is not None:
                total += value.numel() * value.element_size()
    return total


def _gaussian_parameter_bytes(gaussians: Gaussians3D) -> int:
    return sum(
        getattr(gaussians, name).numel() * getattr(gaussians, name).element_size()
        for name in ("means", "quats", "log_scales", "opacity", "sh")
    )


def _run_arm(
    out: Path,
    arm_name: str,
    arm: Arm,
    seed: int,
    initialization: Gaussians3D,
    fit_inputs: ReconstructionInputs,
    all_inputs: ReconstructionInputs,
    banks: Mapping[str, Mapping[str, torch.Tensor]],
    split_indices: Mapping[str, Sequence[int]],
    extent: float,
) -> dict[str, object]:
    directory = out / "arms" / f"seed_{seed}" / arm_name
    result_path = directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("status") != "complete":
            raise RuntimeError(f"incomplete arm receipt already exists: {result_path}")
        return result
    directory.mkdir(parents=True, exist_ok=False)
    config = _train_config(arm, seed, extent)
    snapshots: dict[int, Gaussians3D] = {}

    def capture(snapshot: Gaussians3D, step: int) -> None:
        if step in snapshots:
            raise RuntimeError("duplicate compact checkpoint")
        snapshots[step] = snapshot.to("cpu")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    final, history = CompactTrainer(config).train(
        fit_inputs,
        initialization,
        checkpoint_callback=capture,
    )
    torch.cuda.synchronize(0)
    train_seconds = time.perf_counter() - started
    train_peak_allocated = torch.cuda.max_memory_allocated(0)
    train_peak_reserved = torch.cuda.max_memory_reserved(0)
    if tuple(sorted(snapshots)) != CHECKPOINTS:
        raise RuntimeError("compact checkpoint set changed")
    checkpoint_metrics: dict[str, object] = {}
    for step in CHECKPOINTS:
        model = snapshots[step].to("cuda:0")
        checkpoint_metrics[str(step)] = {
            "fit": _evaluate_views(model, all_inputs, banks, split_indices["fit"]),
            "validation": _evaluate_views(
                model,
                all_inputs,
                banks,
                split_indices["validation"],
            ),
        }
        del model
    final_cpu = final.to("cpu")
    support = _support_diagnostics(final, all_inputs, split_indices)
    final_cpu.save_npz(directory / "gaussians_final.npz")
    _atomic_json(directory / "history.json", history)
    frozen_motion = {
        family: history["optimizer_group_motion"][family]["max_abs"]
        for family in history["frozen_parameter_groups"]
    }
    if any(value != 0.0 for value in frozen_motion.values()):
        raise RuntimeError("a zero-learning-rate parameter family moved")
    result = {
        "schema": "rtgs.compact-only-carrier-stage.arm.v1",
        "status": "complete",
        "arm": arm_name,
        "arm_spec": asdict(arm),
        "seed": seed,
        "config": asdict(config),
        "initial_semantic_sha256": _gaussians_sha256(initialization),
        "final_semantic_sha256": _gaussians_sha256(final_cpu),
        "final_file_sha256": _file_sha256(directory / "gaussians_final.npz"),
        "history_file_sha256": _file_sha256(directory / "history.json"),
        "checkpoint_metrics": checkpoint_metrics,
        "support": support,
        "frozen_motion": frozen_motion,
        "resources": {
            "train_seconds": train_seconds,
            "train_peak_cuda_allocated_bytes": train_peak_allocated,
            "train_peak_cuda_reserved_bytes": train_peak_reserved,
            "arm_peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "arm_peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "peak_rss_bytes": _peak_rss_bytes(),
            "fit_teacher_tensor_bytes": _teacher_bytes(fit_inputs),
            "evaluation_teacher_tensor_bytes": _teacher_bytes(all_inputs),
            "initial_3d_parameter_bytes": _gaussian_parameter_bytes(initialization),
            "final_3d_parameter_bytes": _gaussian_parameter_bytes(final_cpu),
            "point_chunk": config.point_chunk,
            "gaussian_chunk": config.gaussian_chunk,
            "outer_microbatch": config.outer_microbatch,
            "teacher_query_component_chunk": config.query_component_chunk,
            "support_component_chunk": (config.support_component_chunk if arm.support else 0),
        },
    }
    _atomic_json(result_path, result)
    print(
        f"[arm:{arm_name}:{seed}] "
        f"val_Q={checkpoint_metrics[str(ITERATIONS)]['validation']['J_Q']:.8f} "
        f"val_U={checkpoint_metrics[str(ITERATIONS)]['validation']['J_U']:.8f} "
        f"outside={support['splits']['validation']['outside_fraction']:.5f} "
        f"{train_seconds:.1f}s",
        flush=True,
    )
    return result


def _geometric_mean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _fraction_ratio(
    summaries: Mapping[str, Mapping[str, object]],
    numerator: str,
    denominator: str,
    metric: str,
) -> dict[str, object]:
    """Ratio summary for non-negative fractions, including exact-zero outcomes."""

    values: list[float | None] = []
    for left, right in zip(
        summaries[numerator][metric],
        summaries[denominator][metric],
        strict=True,
    ):
        left_value = float(left)
        right_value = float(right)
        if left_value < 0 or right_value < 0:
            raise ValueError("fraction ratios require non-negative values")
        if right_value == 0:
            values.append(1.0 if left_value == 0 else None)
        else:
            values.append(left_value / right_value)
    finite = [value for value in values if value is not None]
    geometric = (
        None
        if len(finite) != len(values)
        else (0.0 if any(value == 0 for value in finite) else _geometric_mean(finite))
    )
    return {
        "numerator": numerator,
        "denominator": denominator,
        "metric": metric,
        "per_seed": values,
        "geometric": geometric,
        "denominator_zero_count": sum(
            float(right) == 0 for right in summaries[denominator][metric]
        ),
    }


def _log_auc(record: Mapping[str, object], split: str, metric: str) -> float:
    x = [step / ITERATIONS for step in CHECKPOINTS]
    y = [
        math.log(
            max(
                float(record["checkpoint_metrics"][str(step)][split][metric]),
                1e-12,
            )
        )
        for step in CHECKPOINTS
    ]
    return sum(
        0.5 * (y[index] + y[index + 1]) * (x[index + 1] - x[index]) for index in range(len(x) - 1)
    )


def _arm_summary(records: Mapping[int, Mapping[str, Mapping[str, object]]], arm: str) -> dict:
    final_q = [
        float(records[seed][arm]["checkpoint_metrics"][str(ITERATIONS)]["validation"]["J_Q"])
        for seed in TRAIN_ROOTS
    ]
    final_u = [
        float(records[seed][arm]["checkpoint_metrics"][str(ITERATIONS)]["validation"]["J_U"])
        for seed in TRAIN_ROOTS
    ]
    auc_q = [_log_auc(records[seed][arm], "validation", "J_Q") for seed in TRAIN_ROOTS]
    outside = [
        float(records[seed][arm]["support"]["splits"]["validation"]["outside_fraction"])
        for seed in TRAIN_ROOTS
    ]
    behind_near = [
        float(records[seed][arm]["support"]["splits"]["validation"]["behind_near_fraction"])
        for seed in TRAIN_ROOTS
    ]
    return {
        "validation_J_Q": final_q,
        "validation_J_U": final_u,
        "validation_log_auc_J_Q": auc_q,
        "validation_outside_fraction": outside,
        "validation_behind_near_fraction": behind_near,
        "geometric_validation_J_Q": _geometric_mean(final_q),
        "geometric_validation_J_U": _geometric_mean(final_u),
        "mean_validation_log_auc_J_Q": sum(auc_q) / len(auc_q),
        "mean_validation_outside_fraction": sum(outside) / len(outside),
        "mean_validation_behind_near_fraction": sum(behind_near) / len(behind_near),
    }


def _ratio(
    summaries: Mapping[str, Mapping[str, object]],
    numerator: str,
    denominator: str,
    metric: str,
) -> dict[str, object]:
    values = [
        float(left) / float(right)
        for left, right in zip(
            summaries[numerator][metric],
            summaries[denominator][metric],
            strict=True,
        )
    ]
    return {
        "numerator": numerator,
        "denominator": denominator,
        "metric": metric,
        "per_seed": values,
        "geometric": _geometric_mean(values),
        "wins": sum(value < 1.0 for value in values),
    }


def _analysis(
    records: Mapping[int, Mapping[str, Mapping[str, object]]],
) -> dict[str, object]:
    summaries = {arm: _arm_summary(records, arm) for arm in ARMS}
    ranking = sorted(ARMS, key=lambda arm: summaries[arm]["geometric_validation_J_Q"])
    repair_pairs = {
        "C": (
            ("legacy_C_all", "beam_all"),
            ("legacy_CO_all", "legacy_O_all"),
            ("legacy_CA_all", "legacy_A_all"),
            ("legacy_COA_all", "legacy_OA_all"),
        ),
        "O": (
            ("legacy_O_all", "beam_all"),
            ("legacy_CO_all", "legacy_C_all"),
            ("legacy_OA_all", "legacy_A_all"),
            ("legacy_COA_all", "legacy_CA_all"),
        ),
        "A": (
            ("legacy_A_all", "beam_all"),
            ("legacy_CA_all", "legacy_C_all"),
            ("legacy_OA_all", "legacy_O_all"),
            ("legacy_COA_all", "legacy_CO_all"),
        ),
    }
    repair_marginals: dict[str, object] = {}
    for factor, pairs in repair_pairs.items():
        per_seed_q = []
        per_seed_u = []
        for seed_index in range(len(TRAIN_ROOTS)):
            q_ratios = [
                float(summaries[on]["validation_J_Q"][seed_index])
                / float(summaries[off]["validation_J_Q"][seed_index])
                for on, off in pairs
            ]
            u_ratios = [
                float(summaries[on]["validation_J_U"][seed_index])
                / float(summaries[off]["validation_J_U"][seed_index])
                for on, off in pairs
            ]
            per_seed_q.append(_geometric_mean(q_ratios))
            per_seed_u.append(_geometric_mean(u_ratios))
        repair_marginals[factor] = {
            "per_seed_J_Q_ratio": per_seed_q,
            "per_seed_J_U_ratio": per_seed_u,
            "geometric_J_Q_ratio": _geometric_mean(per_seed_q),
            "geometric_J_U_ratio": _geometric_mean(per_seed_u),
            "wins": sum(value < 1.0 for value in per_seed_q),
            "necessary_gate": _geometric_mean(per_seed_q) <= 0.95
            and all(value < 1.0 for value in per_seed_q)
            and _geometric_mean(per_seed_u) <= 1.05,
        }
    simple = {
        arm: {
            "J_Q": _ratio(summaries, arm, "beam_all", "validation_J_Q"),
            "J_U": _ratio(summaries, arm, "beam_all", "validation_J_U"),
        }
        for arm in (
            "beam_means_only",
            "beam_means_fixed",
            "beam_geometry_only",
            "beam_appearance_only",
        )
    }
    for value in simple.values():
        value["noninferior"] = (
            value["J_Q"]["geometric"] <= 1.02 and value["J_U"]["geometric"] <= 1.02
        )
    support_pairs = {
        "beam": ("beam_all_support", "beam_all"),
        "corrected_CA": ("corrected_CA_all_support", "corrected_CA_all"),
    }
    support: dict[str, object] = {}
    for name, (on, off) in support_pairs.items():
        q = _ratio(summaries, on, off, "validation_J_Q")
        u = _ratio(summaries, on, off, "validation_J_U")
        outside = _fraction_ratio(summaries, on, off, "validation_outside_fraction")
        near_delta = [
            float(left) - float(right)
            for left, right in zip(
                summaries[on]["validation_behind_near_fraction"],
                summaries[off]["validation_behind_near_fraction"],
                strict=True,
            )
        ]
        outside_geometric = outside["geometric"]
        support[name] = {
            "J_Q": q,
            "J_U": u,
            "outside": outside,
            "behind_near_delta": near_delta,
            "success_gate": (
                outside_geometric is not None
                and outside_geometric <= 0.5
                and q["geometric"] <= 1.05
                and u["geometric"] <= 1.05
                and max(near_delta) <= 0.0
            ),
        }
    corrected = {
        "C": {
            "J_Q": _ratio(summaries, "corrected_C_all", "legacy_C_all", "validation_J_Q"),
            "J_U": _ratio(summaries, "corrected_C_all", "legacy_C_all", "validation_J_U"),
        },
        "CA": {
            "J_Q": _ratio(summaries, "corrected_CA_all", "legacy_CA_all", "validation_J_Q"),
            "J_U": _ratio(summaries, "corrected_CA_all", "legacy_CA_all", "validation_J_U"),
        },
    }
    for value in corrected.values():
        value["replacement_gate"] = (
            value["J_Q"]["geometric"] <= 0.98 and value["J_U"]["geometric"] <= 1.02
        )
    simpler_finalists = [arm for arm, value in simple.items() if value["noninferior"]]
    support_ranking = sorted(
        ("beam_all_support", "corrected_CA_all_support"),
        key=lambda arm: summaries[arm]["geometric_validation_J_Q"],
    )
    finalists = sorted(
        {
            "beam_baseline",
            ranking[0],
            support_ranking[0],
            *simpler_finalists,
        }
    )
    return {
        "summaries": summaries,
        "validation_ranking": ranking,
        "repair_marginals": repair_marginals,
        "simple_parameter_arms": simple,
        "support": support,
        "corrected_vs_legacy": corrected,
        "heldout_finalists": finalists,
    }


def _heldout_unlock(
    out: Path,
    analysis: Mapping[str, object],
    records: Mapping[int, Mapping[str, Mapping[str, object]]],
    beam: Gaussians3D,
    all_inputs: ReconstructionInputs,
    banks_by_root: Mapping[int, Mapping[str, Mapping[str, torch.Tensor]]],
    heldout: Sequence[int],
) -> dict[str, object]:
    path = out / "heldout_unlock.json"
    if path.exists():
        return json.loads(path.read_text())
    results: dict[str, object] = {}
    for finalist in analysis["heldout_finalists"]:
        results[finalist] = {}
        for seed, evaluation_root in zip(TRAIN_ROOTS, EVALUATION_ROOTS, strict=True):
            if finalist == "beam_baseline":
                model = beam.to("cuda:0")
            else:
                model_path = out / "arms" / f"seed_{seed}" / finalist / "gaussians_final.npz"
                model = Gaussians3D.load_npz(model_path).to("cuda:0")
            results[finalist][str(seed)] = _evaluate_views(
                model,
                all_inputs,
                banks_by_root[evaluation_root],
                heldout,
            )
            del model
    receipt = {
        "schema": "rtgs.compact-only-carrier-stage.heldout.v1",
        "selection_sha256": _canonical_hash(analysis),
        "finalists": analysis["heldout_finalists"],
        "results": results,
    }
    _atomic_json(path, receipt)
    return receipt


def run(out: Path) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen calibrated development run requires cuda:0")
    out.mkdir(parents=True, exist_ok=True)
    guard = NoImageGuard()
    started = time.perf_counter()
    with guard:
        dataset = CompactDataset.load(DATASET, device="cpu", load_alpha=False)
        splits = _split_indices(len(dataset.views))
        all_cpu = _inputs(dataset, range(len(dataset.views)), name="frame00008-compact-all")
        fit_cpu = _inputs(dataset, splits["fit"], name="frame00008-compact-fit")
        extent = float(dataset.bounds_hint[1])
        split_names = {
            split: [dataset.views[index].view_id for index in indices]
            for split, indices in splits.items()
        }
        manifest_path = out / "run_manifest.json"
        manifest = {
            "schema": "rtgs.compact-only-carrier-stage.run.v1",
            "status": "running",
            "preregistration": {
                "path": str(PREREGISTRATION.relative_to(ROOT)),
                "sha256": _file_sha256(PREREGISTRATION),
            },
            "dataset": {
                "path": str(DATASET.relative_to(ROOT)),
                "manifest_sha256": _file_sha256(DATASET / "manifest.json"),
                "view_count": len(dataset.views),
                "view_ids": [view.view_id for view in dataset.views],
                "container_bytes": sum(view.bytes for view in dataset.views),
                "container_sha256": {view.view_id: view.sha256 for view in dataset.views},
                "packed_alpha_accessed": False,
            },
            "splits": split_names,
            "split_indices": {key: list(value) for key, value in splits.items()},
            "train_roots": list(TRAIN_ROOTS),
            "evaluation_roots": list(EVALUATION_ROOTS),
            "arms": {name: asdict(value) for name, value in ARMS.items()},
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            },
        }
        _atomic_json(manifest_path, manifest)

        if (out / "beam_gaussians.npz").exists():
            beam = _load_beam(out)
        else:
            config = _beam_config(extent)
            print(
                f"[beam] {len(splits['fit'])} fit views -> {N_CARRIERS} carriers",
                flush=True,
            )
            beam_started = time.perf_counter()
            beam = fuse_gaussian_beams(fit_cpu, config)
            beam_seconds = time.perf_counter() - beam_started
            if beam.n_components != N_CARRIERS:
                raise RuntimeError(f"Beam returned {beam.n_components}, expected {N_CARRIERS}")
            _save_beam(out, beam, beam_seconds, config)

        repaired = _prepare_repairs(out, beam, fit_cpu)
        all_gpu = all_cpu.to("cuda:0")
        fit_gpu = _subset(
            all_gpu,
            splits["fit"],
            name="frame00008-compact-fit-gpu",
        )
        banks_by_root: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
        for root in EVALUATION_ROOTS:
            bank_path, bank = _prepare_bank(out, all_cpu, root)
            banks_by_root[root] = bank
            print(f"[bank:{root}] {bank_path.relative_to(out)}", flush=True)

        records: dict[int, dict[str, dict[str, object]]] = {}
        for seed, evaluation_root in zip(TRAIN_ROOTS, EVALUATION_ROOTS, strict=True):
            records[seed] = {}
            for arm_name, arm in ARMS.items():
                records[seed][arm_name] = _run_arm(
                    out,
                    arm_name,
                    arm,
                    seed,
                    repaired[arm.repair_key],
                    fit_gpu,
                    all_gpu,
                    banks_by_root[evaluation_root],
                    splits,
                    extent,
                )

        analysis = _analysis(records)
        _atomic_json(out / "analysis.json", analysis)
        heldout = _heldout_unlock(
            out,
            analysis,
            records,
            beam.gaussians,
            all_gpu,
            banks_by_root,
            splits["heldout"],
        )
        guard_record = guard.record()
        if not guard_record["passed"]:
            raise RuntimeError(f"no-image boundary failed: {guard_record}")
        result = {
            "schema": "rtgs.compact-only-carrier-stage.result.v1",
            "status": "complete",
            "scope": "single-scene compact-only development",
            "splits": split_names,
            "records": {
                str(seed): {
                    arm: {
                        "result": str(
                            (out / "arms" / f"seed_{seed}" / arm / "result.json").relative_to(out)
                        ),
                        "result_sha256": _file_sha256(
                            out / "arms" / f"seed_{seed}" / arm / "result.json"
                        ),
                    }
                    for arm in ARMS
                }
                for seed in TRAIN_ROOTS
            },
            "analysis": analysis,
            "heldout": heldout,
            "no_image_boundary": guard_record,
            "resources": {
                "total_seconds": time.perf_counter() - started,
                "peak_rss_bytes": _peak_rss_bytes(),
                "teacher_tensor_bytes": _teacher_bytes(all_gpu),
                "compact_container_bytes": sum(view.bytes for view in dataset.views),
            },
        }
        _atomic_json(out / "result.json", result)
        manifest["status"] = "complete"
        manifest["result_sha256"] = _file_sha256(out / "result.json")
        manifest["no_image_boundary"] = guard_record
        manifest["total_seconds"] = result["resources"]["total_seconds"]
        _atomic_json(manifest_path, manifest)
    print(
        f"[complete] winner={analysis['validation_ranking'][0]} "
        f"elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    return run(args.out.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
