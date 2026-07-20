#!/usr/bin/env python3
"""Preregistered quaternion radial-gauge optimizer audit and gated ablation."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from rtgs.core.gaussians3d import Gaussians3D, quat_to_rotmat
from rtgs.core.metrics import ssim
from rtgs.data.scene import SceneData
from rtgs.data.synthetic import make_synthetic_scene
from rtgs.depth.mock import GroundTruthDepth
from rtgs.image2gs.fit import FitConfig, fit_views
from rtgs.lift.depth import DepthLifter
from rtgs.optim.trainer import TrainConfig, Trainer
from rtgs.render.torch_ref import TorchRasterizer

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_PREREGISTRATION = Path("benchmarks/results/20260716_quaternion_gauge_PREREG.md")
ORIGINAL_PREREGISTRATION_SHA256 = "f1ba26d2520e6f78731b404babe0e091f2341d16ab5e30607b25ba32692c764e"
PREREGISTRATION = Path("benchmarks/results/20260716_quaternion_gauge_iter2_PREREG.md")
PREREGISTRATION_SHA256 = "fe201606b878cb29b4502a283dde78a30c3d2dab9a0efa091c83be3b95bfe4f3"
IMPLEMENTATION_REVIEW = Path(
    "benchmarks/results/20260716_quaternion_gauge_iter2_IMPLEMENTATION_REVIEW.md"
)
IMPLEMENTATION_REVIEW_ADDENDUM = Path(
    "benchmarks/results/20260716_quaternion_gauge_iter2_IMPLEMENTATION_REVIEW_ADDENDUM_1.md"
)
DEFAULT_SEAL = Path("benchmarks/results/20260716_quaternion_gauge_iter2_SEAL.json")
PHASE_A_ATTEMPT = ROOT / "benchmarks/results/20260716_quaternion_gauge_iter2_PHASE_A_ATTEMPT.json"
PHASE_B_ATTEMPT = ROOT / "benchmarks/results/20260716_quaternion_gauge_iter2_PHASE_B_ATTEMPT.json"
HARNESS_PATH = Path("benchmarks/quaternion_gauge_ablation.py")
FOCUSED_TEST_PATHS = (
    Path("tests/test_quaternion_gauge.py"),
    Path("tests/test_quaternion_gauge_ablation.py"),
)
CONSUMED_ATTEMPT_BINDINGS = {
    Path("benchmarks/results/20260716_quaternion_gauge_SEAL.json"): (
        "146193dc0783b01d5fada9608e276845a1aea6e8e44ba4ed53772adc47ef4ad8"
    ),
    Path("benchmarks/results/20260716_quaternion_gauge_PHASE_A_ATTEMPT.json"): (
        "c6a7c663edff15114c11b714ed6342e1ebd1e72b535a565e6d3861ce9e7868dc"
    ),
    Path("benchmarks/results/20260716T015517Z_cpu_quaternion_gauge_invalid.json"): (
        "8381979a9b6fba958e34d8a2d2e4210dc783ede808edd2fa88faddf3b4b53739"
    ),
    Path("benchmarks/results/20260716T015517Z_cpu_quaternion_gauge_invalid_RESULT.md"): (
        "34adccfe91650cd821dc99c0f6c4cdf7e5668ac4b89faa0e4ad4466c95d56a61"
    ),
    Path("benchmarks/results/20260716T015517Z_cpu_quaternion_gauge_invalid_AUDIT.md"): (
        "7528d22e0daa909f8f67e8d73b0269de5f9b4bf21b1677a0d2341361be1ecd8d"
    ),
}
CONSUMED_ATTEMPT_FORBIDDEN_PATHS = (
    Path("benchmarks/results/20260716T015517Z_cpu_quaternion_gauge_audit.json"),
    Path("benchmarks/results/20260716T015517Z_cpu_quaternion_gauge_audit_RESULT.md"),
    Path("benchmarks/results/20260716T015517Z_cpu_quaternion_gauge_audit_AUDIT.md"),
    Path("benchmarks/results/20260716T015517Z_cpu_quaternion_gauge_audit_AUDIT.json"),
    Path("benchmarks/results/20260716_quaternion_gauge_PHASE_B_ATTEMPT.json"),
)
CONSUMED_ATTEMPT_FORBIDDEN_GLOBS = (
    "????????T??????Z_cpu_quaternion_gauge_ablation.json",
    "????????T??????Z_cpu_quaternion_gauge_ablation_RESULT.md",
)

SEEDS = (0, 1, 2)
TRAIN_INDICES = (0, 1, 2, 4, 5, 6, 8, 9, 10)
HELD_OUT_INDICES = (3, 7, 11)
RADIAL_SCALES = (0.25, 1.0, 4.0)
PHASE_A_POLICIES = (
    "current",
    "entry_canonical",
    "unit_retraction",
    "tangent_displacement_retraction",
    "gradient_projection_current",
)
CANONICALIZING_POLICIES = (
    "entry_canonical",
    "unit_retraction",
    "tangent_displacement_retraction",
)
PHASE_B_ARMS = ("current", "unit_retraction", "tangent_displacement_retraction")
PHASE_A_CHECKPOINTS = (0, 10, 20, 30, 40)
PHASE_B_CHECKPOINTS = (0, 30, 60, 90, 120)
TRAINING_ORDER = {
    0: PHASE_B_ARMS,
    1: ("unit_retraction", "tangent_displacement_retraction", "current"),
    2: ("tangent_displacement_retraction", "current", "unit_retraction"),
}
UTC_PREFIX_RE = re.compile(r"^\d{8}T\d{6}Z_cpu_quaternion_gauge_iter2$")
MIN_QUATERNION_NORM = 1e-8
ACTIVE_NORM = 1e-12
SCIENTIST_REVIEW_KEYS = frozenset(
    {
        "artifact_type",
        "verdict",
        "phase_b_execution_clearance",
        "phase_a_sha256",
        "human_audit_sha256",
        "seal_sha256",
        "phase_a_attempt_sha256",
        "source_aggregate",
    }
)
ARTIFACT_BINDING_KEYS = frozenset(
    {
        "original_preregistration_path",
        "original_preregistration_sha256",
        "preregistration_path",
        "preregistration_sha256",
        "implementation_review_path",
        "implementation_review_sha256",
        "seal_path",
        "seal_sha256",
        "seal_file_sha256",
        "source_aggregate",
        "retry_provenance_sha256",
        "attempt_marker_path",
        "attempt_marker_sha256",
    }
)


def _sealed_paths() -> tuple[Path, ...]:
    return tuple(
        sorted(
            {
                ORIGINAL_PREREGISTRATION,
                PREREGISTRATION,
                IMPLEMENTATION_REVIEW,
                IMPLEMENTATION_REVIEW_ADDENDUM,
                HARNESS_PATH,
                Path("pyproject.toml"),
                *FOCUSED_TEST_PATHS,
                *(path.relative_to(ROOT) for path in (ROOT / "src" / "rtgs").rglob("*.py")),
                *(path.relative_to(ROOT) for path in (ROOT / "tests").rglob("*.py")),
            },
            key=str,
        )
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def seal_payload_digest(payload: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "sha256"}
    return canonical_json_hash(unsigned)


def strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {value!r} in {path}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    assert_finite_tree(value, str(path))
    return value


def assert_finite_tree(value: Any, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context} contains a non-finite float")
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_tree(item, f"{context}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_tree(item, f"{context}/{index}")


def tensor_collection_hash(items: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        tensor = value.detach().contiguous().cpu()
        for token in (name, str(tensor.dtype), json.dumps(list(tensor.shape))):
            encoded = token.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        raw = tensor.numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def gaussians_hash(gaussians: Gaussians3D) -> str:
    return tensor_collection_hash(
        [
            ("means", gaussians.means),
            ("quats", gaussians.quats),
            ("log_scales", gaussians.log_scales),
            ("opacity", gaussians.opacity),
            ("sh", gaussians.sh),
        ]
    )


def fitted_hash(fitted: list[Any]) -> str:
    items: list[tuple[str, torch.Tensor]] = []
    for view, gaussians in enumerate(fitted):
        for field in ("xy", "chol", "color", "weight"):
            items.append((f"view_{view}/{field}", getattr(gaussians, field)))
    return tensor_collection_hash(items)


def covariance_float64(gaussians: Gaussians3D) -> torch.Tensor:
    quaternions = gaussians.quats.detach().to(dtype=torch.float64)
    log_scales = gaussians.log_scales.detach().to(dtype=torch.float64)
    rotations = quat_to_rotmat(quaternions)
    scales = log_scales.exp()
    rs = rotations * scales[:, None, :]
    return rs @ rs.transpose(-1, -2)


def relative_row_frobenius(
    candidate: torch.Tensor, reference: torch.Tensor
) -> tuple[float, float, float]:
    candidate64 = candidate.detach().to(dtype=torch.float64)
    reference64 = reference.detach().to(dtype=torch.float64)
    if candidate64.shape != reference64.shape or candidate64.ndim < 2:
        raise ValueError("relative Frobenius tensors must have matching row shapes")
    numerator = float(torch.linalg.matrix_norm(candidate64 - reference64, ord="fro").sum().cpu())
    denominator = float(torch.linalg.matrix_norm(reference64, ord="fro").sum().cpu())
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("relative Frobenius denominator must be finite and positive")
    return numerator, denominator, numerator / max(denominator, 1e-18)


def linear_quantile(values: list[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile input or probability is invalid")
    tensor = torch.tensor(values, dtype=torch.float64)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("quantile input must be finite")
    return float(torch.quantile(tensor, quantile, interpolation="linear"))


def normalized_auc(steps: list[int], values: list[float]) -> float:
    if len(steps) != len(values) or len(steps) < 2:
        raise ValueError("AUC requires matching checkpoint/value sequences")
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError("AUC checkpoints must be strictly increasing")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("AUC values must be finite")
    integral = 0.0
    for left, right, a, b in zip(steps, steps[1:], values, values[1:]):
        integral += (right - left) * (a + b) / 2.0
    return integral / (steps[-1] - steps[0])


def psnr_from_sse(sse: float, count: int) -> float:
    if not math.isfinite(sse) or sse < 0.0 or count <= 0:
        raise ValueError("PSNR raw evidence is invalid")
    return -10.0 * math.log10(max(sse / count, 1e-12))


def require_close(stored: Any, derived: float, context: str, *, atol: float = 1e-12) -> None:
    """Fail closed when a serialized scalar summary differs from raw evidence."""
    if not isinstance(stored, (int, float)) or not math.isfinite(float(stored)):
        raise ValueError(f"{context} is not a finite numeric scalar")
    if not math.isclose(float(stored), float(derived), rel_tol=0.0, abs_tol=atol):
        raise ValueError(f"{context} differs from raw evidence")


def require_tensor_equal(stored: Any, derived: torch.Tensor, context: str) -> None:
    """Compare JSON-restored tensor evidence using the producer tensor's exact dtype."""
    restored = torch.tensor(stored, dtype=derived.dtype)
    if restored.shape != derived.shape or not torch.equal(restored, derived.cpu()):
        raise ValueError(f"{context} differs from raw evidence")


def require_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{context} is not a SHA-256 digest")
    return value


def hamilton_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 4:
        raise ValueError("Hamilton product requires matching (N,4) tensors")
    lw, lv = left[:, :1], left[:, 1:]
    rw, rv = right[:, :1], right[:, 1:]
    scalar = lw * rw - (lv * rv).sum(dim=-1, keepdim=True)
    vector = lw * rv + rw * lv + torch.linalg.cross(lv, rv)
    return torch.cat([scalar, vector], dim=-1)


def require_valid_quaternions(value: torch.Tensor, context: str) -> torch.Tensor:
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError(f"{context} must have shape (N,4)")
    norms = torch.linalg.vector_norm(value, dim=-1)
    if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(norms).all()):
        raise ValueError(f"{context} contains non-finite values")
    if bool((norms <= MIN_QUATERNION_NORM).any()):
        raise ValueError(f"{context} contains an invalid zero/near-zero row")
    return norms


def removed_gradient_diagnostics_float64(
    quaternions: torch.Tensor, raw_gradient: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Derive removed-gradient evidence after promoting raw float32 inputs."""
    if (
        quaternions.ndim != 2
        or quaternions.shape[1] != 4
        or raw_gradient.shape != quaternions.shape
    ):
        raise ValueError("removed-gradient inputs must have matching (N,4) shapes")
    if quaternions.dtype != torch.float32 or raw_gradient.dtype != torch.float32:
        raise ValueError("removed-gradient inputs must be raw float32 tensors")
    if quaternions.device != raw_gradient.device:
        raise ValueError("removed-gradient inputs must share a device")
    if not bool(torch.isfinite(quaternions).all()) or not bool(torch.isfinite(raw_gradient).all()):
        raise ValueError("removed-gradient inputs must be finite")

    q64 = quaternions.detach().to(dtype=torch.float64)
    g64 = raw_gradient.detach().to(dtype=torch.float64)
    q_norm64 = torch.linalg.vector_norm(q64, dim=-1)
    if bool((q_norm64 <= MIN_QUATERNION_NORM).any()):
        raise ValueError("removed-gradient quaternion norm is zero/near-zero")
    unit64 = F.normalize(q64, dim=-1)
    signed_dot64 = (unit64 * g64).sum(dim=-1)
    numerator64 = signed_dot64.abs()
    denominator64 = torch.linalg.vector_norm(g64, dim=-1).clamp_min(ACTIVE_NORM)
    fraction64 = numerator64 / denominator64
    projected64 = g64 - unit64 * signed_dot64.unsqueeze(-1)
    return {
        "numerator": numerator64,
        "denominator": denominator64,
        "fraction": fraction64,
        "projected_gradient": projected64,
    }


def fit_config() -> FitConfig:
    return FitConfig(
        n_gaussians=150,
        max_gaussians=5_000,
        iterations=120,
        backend="native",
        adaptive_density=True,
        growth_waves=5,
        relocate_fraction=0.0,
        structsplat_renderer="auto",
        lr=0.01,
        grad_init_mix=0.7,
        row_chunk=64,
        log_every=50,
        convergence_patience=0,
        convergence_tol=0.05,
        convergence_check_every=25,
    )


def depth_lifter(train_scene: SceneData) -> DepthLifter:
    if train_scene.gt_depths is None:
        raise ValueError("frozen Depth lift requires training metric depths")
    return DepthLifter(
        backend=GroundTruthDepth(train_scene.gt_depths),
        sh_degree=0,
        min_weight=0.05,
        init_opacity=0.1,
        normal_thickness=0.15,
        covariance_mode="surface",
        isotropic_sigma=None,
        robust_depth_gradients=True,
        merge=True,
        merge_voxel_frac=0.01,
    )


def depth_lifter_config() -> dict[str, Any]:
    return {
        "backend": "GroundTruthDepth(metric, training-only, cursor=0)",
        "sh_degree": 0,
        "min_weight": 0.05,
        "init_opacity": 0.1,
        "normal_thickness": 0.15,
        "covariance_mode": "surface",
        "isotropic_sigma": None,
        "robust_depth_gradients": True,
        "merge": True,
        "merge_voxel_frac": 0.01,
    }


def train_config(seed: int, arm: str) -> TrainConfig:
    if arm not in PHASE_B_ARMS:
        raise ValueError(f"unknown Phase-B arm {arm!r}")
    return TrainConfig(
        iterations=120,
        lr_means=1.6e-4,
        lr_quats=1e-3,
        lr_scales=5e-3,
        lr_opacity=5e-2,
        lr_sh=2.5e-3,
        lr_sh_rest=1.25e-4,
        ssim_lambda=0.2,
        rasterizer="torch",
        device="cpu",
        densify=False,
        eval_every=30,
        target_sh_degree=3,
        sh_degree_interval=30,
        use_masks=False,
        outside_alpha_lambda=0.01,
        mask_alpha_lambda=0.05,
        random_background=False,
        opacity_reg=None,
        scale_reg=None,
        packed=False,
        antialiased=False,
        sh_color_activation="hard",
        kernel_support_mode="hard",
        visibility_margin_sigma=3.0,
        validate_render_finite=True,
        quaternion_update_policy=arm,
        seed=seed,
    )


def scene_hashes(scene: SceneData) -> dict[str, str]:
    camera_scalars = []
    camera_tensors = []
    for index, camera in enumerate(scene.cameras):
        camera_scalars.append(
            {
                "fx": camera.fx,
                "fy": camera.fy,
                "cx": camera.cx,
                "cy": camera.cy,
                "width": camera.width,
                "height": camera.height,
            }
        )
        camera_tensors.extend(((f"R_{index}", camera.R), (f"t_{index}", camera.t)))
    center, extent = scene.center_and_extent()
    result = {
        "images": tensor_collection_hash(
            [(f"image_{index}", image) for index, image in enumerate(scene.images)]
        ),
        "camera_scalars": canonical_json_hash(camera_scalars),
        "camera_tensors": tensor_collection_hash(camera_tensors),
        "bounds": canonical_json_hash(
            {"center": [float(item) for item in center.double()], "extent": float(extent)}
        ),
    }
    if scene.gt_depths is not None:
        result["gt_depths"] = tensor_collection_hash(
            [(f"depth_{index}", depth) for index, depth in enumerate(scene.gt_depths)]
        )
    if scene.points is not None:
        result["points"] = tensor_collection_hash([("points", scene.points)])
    return result


def strip_training_capability(full_scene: SceneData) -> SceneData:
    train_scene = full_scene.subset(list(TRAIN_INDICES), name_suffix="quaternion-gauge-train")
    train_scene.gt_gaussians = None
    train_scene.test_indices = []
    if train_scene.n_views != 9 or train_scene.training_views != list(range(9)):
        raise RuntimeError("physical training subset is invalid")
    return train_scene


def environment_metadata() -> dict[str, Any]:
    optional_modules = (
        "gsplat",
        "structsplat",
        "rtgs.image2gs.structsplat_backend",
    )
    dependency_versions = {}
    for distribution in ("numpy", "torch", "pillow", "pytest", "ruff"):
        try:
            dependency_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[distribution] = None
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "optional_modules_loaded": {module: module in sys.modules for module in optional_modules},
        "dependency_versions": dependency_versions,
        "device": "cpu",
    }


def environment_fingerprint(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "python",
        "torch",
        "platform",
        "processor",
        "cpu_count",
        "torch_num_threads",
        "torch_num_interop_threads",
        "deterministic_algorithms",
        "cuda_visible_devices",
        "omp_num_threads",
        "mkl_num_threads",
        "optional_modules_loaded",
        "dependency_versions",
        "device",
    )
    return {key: metadata[key] for key in keys}


def assert_official_environment(metadata: dict[str, Any]) -> None:
    expected = {
        "torch_num_threads": 4,
        "deterministic_algorithms": True,
        "cuda_visible_devices": "",
        "omp_num_threads": "4",
        "mkl_num_threads": "4",
        "device": "cpu",
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    loaded = [name for name, present in metadata["optional_modules_loaded"].items() if present]
    if loaded:
        mismatches["optional_modules_loaded"] = {"expected": [], "actual": loaded}
    if mismatches:
        raise RuntimeError(f"official CPU environment differs: {mismatches}")


def git_metadata() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    return {
        "revision": revision,
        "dirty": bool(status.strip()),
        "status": status.splitlines(),
        "tracked_diff_sha256": sha256_bytes(diff),
    }


def source_hashes(paths: tuple[Path, ...] | None = None) -> tuple[dict[str, str], str]:
    selected = _sealed_paths() if paths is None else paths
    missing = [str(path) for path in selected if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"sealed source files are missing: {missing}")
    hashes = {str(path): sha256_file(ROOT / path) for path in selected}
    return hashes, canonical_json_hash(hashes)


def loaded_source_hashes() -> tuple[dict[str, str], str]:
    paths: set[Path] = set()
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if source is None:
            continue
        path = Path(source).resolve()
        if path.suffix == ".py" and path.is_relative_to(ROOT) and path.is_file():
            relative = path.relative_to(ROOT)
            if not relative.parts or relative.parts[0] != ".venv":
                paths.add(path)
    paths.update(
        {
            (ROOT / ORIGINAL_PREREGISTRATION).resolve(),
            (ROOT / PREREGISTRATION).resolve(),
            (ROOT / IMPLEMENTATION_REVIEW).resolve(),
            (ROOT / IMPLEMENTATION_REVIEW_ADDENDUM).resolve(),
            (ROOT / HARNESS_PATH).resolve(),
            (ROOT / "pyproject.toml").resolve(),
        }
    )
    hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(paths)}
    return hashes, canonical_json_hash(hashes)


def verify_preregistration() -> None:
    original_actual = sha256_file(ROOT / ORIGINAL_PREREGISTRATION)
    if original_actual != ORIGINAL_PREREGISTRATION_SHA256:
        raise RuntimeError(
            "original quaternion preregistration hash differs: "
            f"{original_actual} != {ORIGINAL_PREREGISTRATION_SHA256}"
        )
    actual = sha256_file(ROOT / PREREGISTRATION)
    if actual != PREREGISTRATION_SHA256:
        raise RuntimeError(
            f"quaternion preregistration hash differs: {actual} != {PREREGISTRATION_SHA256}"
        )


def verify_consumed_attempt_provenance() -> dict[str, Any]:
    artifacts: dict[str, str] = {}
    for path, expected in CONSUMED_ATTEMPT_BINDINGS.items():
        resolved = ROOT / path
        if not resolved.is_file():
            raise FileNotFoundError(f"consumed-attempt artifact is missing: {path}")
        actual = sha256_file(resolved)
        if actual != expected:
            raise RuntimeError(
                f"consumed-attempt artifact hash differs for {path}: {actual} != {expected}"
            )
        artifacts[str(path)] = actual

    present = [str(path) for path in CONSUMED_ATTEMPT_FORBIDDEN_PATHS if (ROOT / path).exists()]
    results = ROOT / "benchmarks/results"
    glob_matches = sorted(
        {
            str(path.relative_to(ROOT))
            for pattern in CONSUMED_ATTEMPT_FORBIDDEN_GLOBS
            for path in results.glob(pattern)
        }
    )
    if present or glob_matches:
        raise RuntimeError(
            "consumed invalid attempt unexpectedly exposes a valid/Phase-B artifact: "
            f"paths={present}, globs={glob_matches}"
        )
    return {
        "original_preregistration": {
            "path": str(ORIGINAL_PREREGISTRATION),
            "sha256": ORIGINAL_PREREGISTRATION_SHA256,
        },
        "consumed_artifacts": artifacts,
        "required_absent_paths": [str(path) for path in CONSUMED_ATTEMPT_FORBIDDEN_PATHS],
        "required_absent_globs": list(CONSUMED_ATTEMPT_FORBIDDEN_GLOBS),
        "verified_absent": True,
    }


def verify_implementation_review() -> dict[str, str]:
    path = ROOT / IMPLEMENTATION_REVIEW
    if not path.is_file():
        raise FileNotFoundError(
            f"independent implementation review is missing: {IMPLEMENTATION_REVIEW}"
        )
    content = path.read_text(encoding="utf-8")
    if "Verdict: PASS" not in content.splitlines():
        raise RuntimeError("independent implementation review lacks exact 'Verdict: PASS'")
    return {"path": str(IMPLEMENTATION_REVIEW), "sha256": sha256_file(path)}


def verify_implementation_review_addendum() -> dict[str, str]:
    path = ROOT / IMPLEMENTATION_REVIEW_ADDENDUM
    if not path.is_file():
        raise FileNotFoundError(
            "independent implementation-review addendum is missing: "
            f"{IMPLEMENTATION_REVIEW_ADDENDUM}"
        )
    content = path.read_text(encoding="utf-8")
    if "Verdict: PASS" not in content.splitlines():
        raise RuntimeError("independent implementation-review addendum lacks exact 'Verdict: PASS'")
    return {"path": str(IMPLEMENTATION_REVIEW_ADDENDUM), "sha256": sha256_file(path)}


def validate_implementation_review_bindings(
    payload: dict[str, Any],
    implementation_review: dict[str, str],
    implementation_review_addendum: dict[str, str],
) -> None:
    if payload.get("implementation_review") != implementation_review:
        raise RuntimeError("seal implementation-review binding differs")
    if payload.get("implementation_review_addendum") != implementation_review_addendum:
        raise RuntimeError("seal implementation-review addendum binding differs")


def verify_default_semantics() -> dict[str, Any]:
    config = TrainConfig()
    signature = inspect.signature(Trainer.train).parameters
    expected_none = (
        "checkpoint_callback",
        "step_controls",
        "initialization_callback",
        "quaternion_step_callback",
    )
    if config.quaternion_update_policy != "current":
        raise RuntimeError("Trainer quaternion policy default is not current")
    if any(signature[name].default is not None for name in expected_none):
        raise RuntimeError("Trainer research callback/control default is not None")
    return {
        "quaternion_update_policy": config.quaternion_update_policy,
        "none_defaults": list(expected_none),
        "kernel_support_mode": config.kernel_support_mode,
        "visibility_margin_sigma": config.visibility_margin_sigma,
        "original_preregistration_sha256": sha256_file(ROOT / ORIGINAL_PREREGISTRATION),
        "preregistration_sha256": sha256_file(ROOT / PREREGISTRATION),
    }


def run_verification() -> dict[str, Any]:
    python = ".venv/bin/python"
    commands = (
        [python, "-m", "ruff", "check", "."],
        [python, "-m", "ruff", "format", "--check", "."],
        [python, "-m", "pytest", "-q", "-m", "not slow"],
        [python, "scripts/docs_sync.py"],
        ["git", "diff", "--check"],
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    results = []
    for command in commands:
        started = time.perf_counter()
        completed = subprocess.run(
            command, cwd=ROOT, env=environment, capture_output=True, text=True, check=False
        )
        record = {
            "command": command,
            "returncode": completed.returncode,
            "seconds": time.perf_counter() - started,
            "stdout_sha256": sha256_bytes(completed.stdout.encode()),
            "stderr_sha256": sha256_bytes(completed.stderr.encode()),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        results.append(record)
        if completed.returncode != 0:
            raise RuntimeError(
                f"verification failed: {' '.join(command)}\n"
                f"{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
            )
    return {"passed": True, "commands": results}


def create_seal() -> dict[str, Any]:
    verify_preregistration()
    retry_provenance_before = verify_consumed_attempt_provenance()
    implementation_review_before = verify_implementation_review()
    implementation_review_addendum_before = verify_implementation_review_addendum()
    environment = environment_metadata()
    assert_official_environment(environment)
    hashes_before, aggregate_before = source_hashes()
    verification = run_verification()
    verify_preregistration()
    retry_provenance = verify_consumed_attempt_provenance()
    implementation_review = verify_implementation_review()
    implementation_review_addendum = verify_implementation_review_addendum()
    hashes, aggregate = source_hashes()
    if (
        retry_provenance != retry_provenance_before
        or implementation_review != implementation_review_before
        or implementation_review_addendum != implementation_review_addendum_before
        or hashes != hashes_before
        or aggregate != aggregate_before
    ):
        raise RuntimeError("sealed source/review snapshot changed during full verification")
    payload: dict[str, Any] = {
        "artifact_type": "quaternion_gauge_iter2_implementation_seal",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_metadata(),
        "sealed_paths": [str(path) for path in _sealed_paths()],
        "source_hashes": hashes,
        "source_aggregate": aggregate,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
        },
        "original_preregistration": {
            "path": str(ORIGINAL_PREREGISTRATION),
            "sha256": ORIGINAL_PREREGISTRATION_SHA256,
        },
        "retry_provenance": retry_provenance,
        "implementation_review": implementation_review,
        "implementation_review_addendum": implementation_review_addendum,
        "verification": verification,
        "default_semantics": verify_default_semantics(),
        "effective_configurations": {
            "synthetic_scene": {
                "n_gaussians": 40,
                "n_cameras": 12,
                "image_size": 48,
                "seeds": list(SEEDS),
                "training_indices": list(TRAIN_INDICES),
                "held_out_indices": list(HELD_OUT_INDICES),
            },
            "fit": asdict(fit_config()),
            "depth_lifter": depth_lifter_config(),
            "phase_a": {
                "policies": list(PHASE_A_POLICIES),
                "radial_scales": list(RADIAL_SCALES),
                "checkpoints": list(PHASE_A_CHECKPOINTS),
                "iterations": 40,
            },
            "phase_b": {arm: asdict(train_config(0, arm)) for arm in PHASE_B_ARMS},
        },
        "environment": environment,
        "command": [sys.executable, *sys.argv],
    }
    payload["sha256"] = seal_payload_digest(payload)
    return payload


def load_and_verify_seal(path: Path) -> dict[str, Any]:
    verify_preregistration()
    implementation_review = verify_implementation_review()
    implementation_review_addendum = verify_implementation_review_addendum()
    payload = strict_json_load(path)
    if payload.get("artifact_type") != "quaternion_gauge_iter2_implementation_seal":
        raise ValueError(f"{path} is not a quaternion-gauge implementation seal")
    digest = payload.get("sha256")
    if not isinstance(digest, str) or digest != seal_payload_digest(payload):
        raise RuntimeError("seal canonical payload digest differs")
    expected_paths = [str(item) for item in _sealed_paths()]
    if payload.get("sealed_paths") != expected_paths:
        raise RuntimeError("seal path set differs from current repository path set")
    hashes, aggregate = source_hashes(tuple(Path(item) for item in expected_paths))
    if hashes != payload.get("source_hashes") or aggregate != payload.get("source_aggregate"):
        raise RuntimeError("implementation/protocol differs from sealed source")
    if payload.get("preregistration") != {
        "path": str(PREREGISTRATION),
        "sha256": PREREGISTRATION_SHA256,
    }:
        raise RuntimeError("seal preregistration binding differs")
    if payload.get("original_preregistration") != {
        "path": str(ORIGINAL_PREREGISTRATION),
        "sha256": ORIGINAL_PREREGISTRATION_SHA256,
    }:
        raise RuntimeError("seal original-preregistration binding differs")
    retry_provenance = verify_consumed_attempt_provenance()
    if payload.get("retry_provenance") != retry_provenance:
        raise RuntimeError("seal consumed-attempt provenance binding differs")
    validate_implementation_review_bindings(
        payload, implementation_review, implementation_review_addendum
    )
    if not payload.get("verification", {}).get("passed"):
        raise RuntimeError("seal lacks passing full verification")
    if payload.get("default_semantics") != verify_default_semantics():
        raise RuntimeError("sealed defaults differ from current defaults")
    current_environment = environment_metadata()
    assert_official_environment(current_environment)
    if environment_fingerprint(payload["environment"]) != environment_fingerprint(
        current_environment
    ):
        raise RuntimeError("current environment differs from implementation seal")
    return {
        "path": str(path),
        "sha256": digest,
        "file_sha256": sha256_file(path),
        "source_hashes": hashes,
        "source_aggregate": aggregate,
        "implementation_review": implementation_review,
        "implementation_review_addendum": implementation_review_addendum,
        "retry_provenance": retry_provenance,
        "environment_fingerprint": environment_fingerprint(payload["environment"]),
        "verification_sha256": canonical_json_hash(payload["verification"]),
    }


def verify_loaded_sources_against_seal(seal_path: Path) -> tuple[dict[str, str], str]:
    payload = strict_json_load(seal_path)
    sealed = payload["source_hashes"]
    loaded, aggregate = loaded_source_hashes()
    unexpected = sorted(set(loaded) - set(sealed))
    mismatched = sorted(
        path for path, digest in loaded.items() if path in sealed and sealed[path] != digest
    )
    if unexpected or mismatched:
        raise RuntimeError(
            "loaded repository source differs from seal: "
            f"unexpected={unexpected}, mismatched={mismatched}"
        )
    return loaded, aggregate


def exclusive_atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(temporary),
            -100,
            os.fsencode(path),
            1,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise FileExistsError(f"refusing to overwrite append-only path {path}")
            raise OSError(error_number, os.strerror(error_number), path)
    finally:
        if temporary.exists():
            temporary.unlink()


def tensor_values(value: torch.Tensor) -> Any:
    """Return detached CPU values while refusing non-finite scientific evidence."""
    tensor = value.detach().cpu()
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise ValueError("refusing to serialize a non-finite tensor")
    return tensor.tolist()


def scale_key(scale: float) -> str:
    return {0.25: "0.25", 1.0: "1", 4.0: "4"}[scale]


def renderer() -> TorchRasterizer:
    return TorchRasterizer(
        sh_color_activation="hard",
        kernel_support_mode="hard",
        visibility_margin_sigma=3.0,
    )


def gaussian_with_quaternions(template: Gaussians3D, quaternions: torch.Tensor) -> Gaussians3D:
    if quaternions.shape != template.quats.shape:
        raise ValueError("replacement quaternion tensor has the wrong shape")
    return Gaussians3D(
        means=template.means,
        quats=quaternions,
        log_scales=template.log_scales,
        opacity=template.opacity,
        sh=template.sh,
    )


def render_hash(outputs: list[Any]) -> str:
    fields: list[tuple[str, torch.Tensor]] = []
    for view, output in enumerate(outputs):
        fields.extend(
            (
                (f"view_{view}/color", output.color),
                (f"view_{view}/alpha", output.alpha),
                (f"view_{view}/depth", output.depth),
            )
        )
    return tensor_collection_hash(fields)


def render_views(
    gaussians: Gaussians3D,
    cameras: list[Any],
    *,
    sh_degree: int,
) -> list[Any]:
    rasterizer = renderer()
    background = gaussians.means.new_zeros(3)
    with torch.no_grad():
        outputs = [
            rasterizer.render(
                gaussians,
                camera,
                background=background,
                sh_degree=sh_degree,
            )
            for camera in cameras
        ]
    for view, output in enumerate(outputs):
        for field in ("color", "alpha", "depth"):
            if not bool(torch.isfinite(getattr(output, field)).all()):
                raise ValueError(f"view {view} has a non-finite {field} render")
    return outputs


def phase_a_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.8 * (prediction - target).abs().mean() + 0.2 * (1.0 - ssim(prediction, target))


def phase_a_schedule(seed: int) -> list[int]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [int(torch.randint(0, 9, (1,), generator=generator, device="cpu")) for _ in range(40)]


def phase_b_schedule(seed: int) -> dict[str, list[int]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    local = [int(torch.randint(0, 9, (1,), generator=generator, device="cpu")) for _ in range(120)]
    return {
        "local_positions": local,
        "original_view_indices": [TRAIN_INDICES[index] for index in local],
    }


def validate_fitted(fitted: list[Any], histories: list[dict[str, Any]]) -> None:
    if len(fitted) != 9 or len(histories) != 9:
        raise ValueError("native fit did not return exactly nine views and histories")
    for view, gaussians in enumerate(fitted):
        if gaussians.n != 150:
            raise ValueError(f"fit view {view} has {gaussians.n} rather than 150 rows")
        expected = {
            "xy": (150, 2),
            "chol": (150, 3),
            "color": (150, 3),
            "weight": (150,),
        }
        for field, shape in expected.items():
            value = getattr(gaussians, field)
            if value.shape != shape or value.dtype != torch.float32:
                raise ValueError(f"fit view {view} {field} has an invalid shape/dtype")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"fit view {view} {field} is non-finite")
        if bool((gaussians.chol[:, (0, 2)] <= 0).any()):
            raise ValueError(f"fit view {view} has a non-positive Cholesky diagonal")
        if bool(
            (gaussians.xy[:, 0] < 0).any()
            or (gaussians.xy[:, 0] > 48).any()
            or (gaussians.xy[:, 1] < 0).any()
            or (gaussians.xy[:, 1] > 48).any()
        ):
            raise ValueError(f"fit view {view} center is outside the frozen image bounds")
        if bool(((gaussians.color < 0) | (gaussians.color > 1)).any()):
            raise ValueError(f"fit view {view} color is outside [0,1]")
        if bool(((gaussians.weight < 0) | (gaussians.weight > 1)).any()):
            raise ValueError(f"fit view {view} weight is outside [0,1]")
    assert_finite_tree(histories, "fit histories")


def validate_gaussians(
    gaussians: Gaussians3D,
    context: str,
    *,
    minimum_rows: int = 1,
) -> None:
    if gaussians.n < minimum_rows:
        raise ValueError(f"{context} has fewer than {minimum_rows} rows")
    for field in ("means", "quats", "log_scales", "opacity", "sh"):
        value = getattr(gaussians, field)
        if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{context} {field} is not finite float32")
    if bool(((gaussians.opacity < 0) | (gaussians.opacity > 1)).any()):
        raise ValueError(f"{context} opacity is outside [0,1]")
    require_valid_quaternions(gaussians.quats, f"{context} quaternions")
    eigenvalues = torch.linalg.eigvalsh(covariance_float64(gaussians))
    if not bool(torch.isfinite(eigenvalues).all()) or bool((eigenvalues <= 0).any()):
        raise ValueError(f"{context} covariance is not strictly positive definite")


def training_hashes(
    scene: SceneData,
    fitted: list[Any],
    histories: list[dict[str, Any]],
    initialization: Gaussians3D,
) -> dict[str, Any]:
    covariance = covariance_float64(initialization)
    field_hashes = {
        field: tensor_collection_hash([(field, getattr(initialization, field))])
        for field in ("means", "quats", "log_scales", "opacity", "sh")
    }
    result: dict[str, Any] = {
        "scene": scene_hashes(scene),
        "fit_fields": fitted_hash(fitted),
        "fit_histories": canonical_json_hash(histories),
        "fit_order": canonical_json_hash(list(range(9))),
        "initialization_fields": field_hashes,
        "initialization": gaussians_hash(initialization),
        "initialization_covariance": tensor_collection_hash([("covariance_float64", covariance)]),
    }
    result["aggregate"] = canonical_json_hash(result)
    return result


def diagnostic_selection(initialization: Gaussians3D) -> dict[str, Any]:
    covariance = covariance_float64(initialization)
    eigenvalues = torch.linalg.eigvalsh(covariance)
    if not bool(torch.isfinite(eigenvalues).all()) or bool((eigenvalues <= 0).any()):
        raise ValueError("diagnostic selection encountered invalid covariance eigenvalues")
    anisotropy = eigenvalues[:, 2] / eigenvalues[:, 0].clamp_min(1e-18)
    eligible = [index for index, value in enumerate(anisotropy.tolist()) if value >= 2.0]
    if len(eligible) < 128:
        raise ValueError(f"only {len(eligible)} anisotropic rows are diagnostic-eligible")
    ranked = sorted(eligible, key=lambda index: (-float(anisotropy[index]), index))
    selected = sorted(ranked[:128])
    return {
        "eigenvalues": tensor_values(eigenvalues),
        "anisotropy": tensor_values(anisotropy),
        "eligible_indices": eligible,
        "selected_indices": selected,
        "eigenvalues_sha256": tensor_collection_hash([("eigenvalues", eigenvalues)]),
        "anisotropy_sha256": tensor_collection_hash([("anisotropy", anisotropy)]),
        "selected_sha256": tensor_collection_hash(
            [("selected", torch.tensor(selected, dtype=torch.int64))]
        ),
    }


def perturb_diagnostic(target: Gaussians3D, seed: int) -> tuple[Gaussians3D, dict[str, Any]]:
    normalized = F.normalize(target.quats, dim=-1)
    generator = torch.Generator(device="cpu").manual_seed(50_000 + seed)
    raw_axes = torch.randn(128, 3, generator=generator, dtype=torch.float32)
    axis_norms = torch.linalg.vector_norm(raw_axes, dim=-1)
    if (
        not bool(torch.isfinite(raw_axes).all())
        or not bool(torch.isfinite(axis_norms).all())
        or bool((axis_norms < 1e-6).any())
    ):
        raise ValueError("frozen perturbation contains an invalid axis without redraw")
    axes = raw_axes / axis_norms[:, None]
    half_angle = math.radians(20.0) / 2.0
    delta = torch.cat(
        [
            torch.full((128, 1), math.cos(half_angle), dtype=torch.float32),
            axes * math.sin(half_angle),
        ],
        dim=-1,
    )
    perturbed_quats = F.normalize(hamilton_product(delta, normalized), dim=-1)
    perturbed = gaussian_with_quaternions(target, perturbed_quats)
    return perturbed, {
        "seed": 50_000 + seed,
        "angle_degrees": 20.0,
        "raw_axes": tensor_values(raw_axes),
        "raw_axis_norms": tensor_values(axis_norms),
        "axes": tensor_values(axes),
        "delta_quaternions": tensor_values(delta),
        "perturbed_quaternions": tensor_values(perturbed_quats),
        "raw_axes_sha256": tensor_collection_hash([("raw_axes", raw_axes)]),
        "perturbed_sha256": tensor_collection_hash([("perturbed_quaternions", perturbed_quats)]),
    }


def prepare_seed(seed: int, *, retain_evaluator: bool) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError(f"seed {seed} is outside the frozen set")
    full_scene = make_synthetic_scene(
        n_gaussians=40,
        n_cameras=12,
        image_size=48,
        seed=seed,
    )
    train_scene = strip_training_capability(full_scene)
    evaluator = None
    if retain_evaluator:
        if full_scene.gt_gaussians is None:
            raise ValueError("fresh Phase-B scene lacks synthetic GT")
        evaluator = {
            "cameras": [full_scene.cameras[index] for index in HELD_OUT_INDICES],
            "gt_gaussians": full_scene.gt_gaussians.detach(),
            "original_indices": list(HELD_OUT_INDICES),
        }
    del full_scene
    train_scene.validate()

    fitted, histories = fit_views(
        train_scene.images,
        fit_config(),
        seed=seed,
        masks=train_scene.masks,
    )
    validate_fitted(fitted, histories)
    initialization = depth_lifter(train_scene).lift(fitted, train_scene)
    validate_gaussians(initialization, "full Depth initialization", minimum_rows=256)
    hashes = training_hashes(train_scene, fitted, histories, initialization)
    return {
        "seed": seed,
        "scene": train_scene,
        "fitted": fitted,
        "fit_histories": histories,
        "initialization": initialization,
        "training_hashes": hashes,
        "evaluator": evaluator,
    }


def prepare_phase_a_seed(seed: int) -> dict[str, Any]:
    prepared = prepare_seed(seed, retain_evaluator=False)
    selection = diagnostic_selection(prepared["initialization"])
    selected_tensor = torch.tensor(selection["selected_indices"], dtype=torch.int64)
    target = prepared["initialization"].subset(selected_tensor).detach()
    validate_gaussians(target, "diagnostic target", minimum_rows=128)
    perturbed, perturbation = perturb_diagnostic(target, seed)
    targets = render_views(target, prepared["scene"].cameras, sh_degree=0)
    schedule = phase_a_schedule(seed)
    prepared.update(
        {
            "selection": selection,
            "target": target,
            "perturbed": perturbed,
            "perturbation": perturbation,
            "targets": targets,
            "target_hashes": {
                "fields": gaussians_hash(target),
                "covariance": tensor_collection_hash([("covariance", covariance_float64(target))]),
                "renders": render_hash(targets),
            },
            "schedule": schedule,
            "schedule_sha256": canonical_json_hash(schedule),
        }
    )
    return prepared


def gaussians_from_serialized_fields(
    fields: dict[str, Any],
    context: str,
    *,
    expected_rows: int | None = None,
    minimum_rows: int = 1,
) -> Gaussians3D:
    expected = {"means", "quats", "log_scales", "opacity", "sh"}
    if set(fields) != expected:
        raise ValueError(f"{context} field set differs")
    gaussians = Gaussians3D(
        means=torch.tensor(fields["means"], dtype=torch.float32),
        quats=torch.tensor(fields["quats"], dtype=torch.float32),
        log_scales=torch.tensor(fields["log_scales"], dtype=torch.float32),
        opacity=torch.tensor(fields["opacity"], dtype=torch.float32),
        sh=torch.tensor(fields["sh"], dtype=torch.float32),
    )
    validate_gaussians(gaussians, context, minimum_rows=minimum_rows)
    if expected_rows is not None and gaussians.n != expected_rows:
        raise ValueError(f"{context} does not contain exactly {expected_rows} rows")
    return gaussians


def validate_phase_a_preparation(preparation: dict[str, Any], seed: int) -> dict[str, Any]:
    if preparation.get("seed") != seed:
        raise ValueError(f"seed {seed}: preparation seed differs")
    if preparation.get("fit_config") != asdict(fit_config()):
        raise ValueError(f"seed {seed}: fit configuration differs")
    if preparation.get("depth_lifter_config") != depth_lifter_config():
        raise ValueError(f"seed {seed}: Depth-lifter configuration differs")
    if preparation.get("training_original_indices") != list(TRAIN_INDICES):
        raise ValueError(f"seed {seed}: training-view identity differs")
    initialization_n = preparation.get("initialization_n")
    if not isinstance(initialization_n, int) or initialization_n < 256:
        raise ValueError(f"seed {seed}: initialization row count is invalid")
    initialization = gaussians_from_serialized_fields(
        preparation["initialization_fields"],
        f"seed {seed} serialized full initialization",
        expected_rows=initialization_n,
        minimum_rows=256,
    )
    training_hashes = preparation["training_hashes"]
    if not isinstance(training_hashes, dict):
        raise ValueError(f"seed {seed}: training hashes are invalid")
    training_without_aggregate = {
        key: value for key, value in training_hashes.items() if key != "aggregate"
    }
    if training_hashes.get("aggregate") != canonical_json_hash(training_without_aggregate):
        raise ValueError(f"seed {seed}: training aggregate hash differs")
    if training_hashes.get("initialization") != gaussians_hash(initialization):
        raise ValueError(f"seed {seed}: initialization field hash differs")
    if training_hashes.get("initialization_covariance") != tensor_collection_hash(
        [("covariance_float64", covariance_float64(initialization))]
    ):
        raise ValueError(f"seed {seed}: initialization covariance hash differs")
    expected_field_hashes = {
        field: tensor_collection_hash([(field, getattr(initialization, field))])
        for field in ("means", "quats", "log_scales", "opacity", "sh")
    }
    if training_hashes.get("initialization_fields") != expected_field_hashes:
        raise ValueError(f"seed {seed}: initialization per-field hashes differ")
    for field in ("scene", "fit_fields", "fit_histories", "fit_order"):
        value = training_hashes.get(field)
        if isinstance(value, dict):
            for name, digest in value.items():
                require_sha256(digest, f"seed {seed} training {field}/{name}")
        else:
            require_sha256(value, f"seed {seed} training {field}")
    target = gaussians_from_serialized_fields(
        preparation["target_fields"],
        f"seed {seed} serialized diagnostic target",
        expected_rows=128,
        minimum_rows=128,
    )
    target_covariance = covariance_float64(target)
    require_tensor_equal(
        preparation["target_covariance"], target_covariance, f"seed {seed} target covariance"
    )
    target_hashes = preparation["target_hashes"]
    if target_hashes.get("fields") != gaussians_hash(target):
        raise ValueError(f"seed {seed}: target field hash differs")
    if target_hashes.get("covariance") != tensor_collection_hash(
        [("covariance", target_covariance)]
    ):
        raise ValueError(f"seed {seed}: target covariance hash differs")
    require_sha256(target_hashes.get("renders"), f"seed {seed} target render hash")

    selection = preparation["selection"]
    eigenvalues = torch.linalg.eigvalsh(covariance_float64(initialization))
    require_tensor_equal(
        selection["eigenvalues"], eigenvalues, f"seed {seed} diagnostic eigenvalues"
    )
    if bool((eigenvalues <= 0).any()):
        raise ValueError(f"seed {seed}: diagnostic eigenvalue evidence is non-positive")
    anisotropy = eigenvalues[:, 2] / eigenvalues[:, 0].clamp_min(1e-18)
    require_tensor_equal(selection["anisotropy"], anisotropy, f"seed {seed} diagnostic anisotropy")
    eligible = [index for index, value in enumerate(anisotropy.tolist()) if value >= 2.0]
    ranked = sorted(eligible, key=lambda index: (-float(anisotropy[index]), index))
    selected = sorted(ranked[:128])
    if selection.get("eligible_indices") != eligible:
        raise ValueError(f"seed {seed}: diagnostic eligibility differs from row evidence")
    if len(eligible) < 128 or selection.get("selected_indices") != selected:
        raise ValueError(f"seed {seed}: diagnostic selection differs from frozen ranking")
    if selection.get("eigenvalues_sha256") != tensor_collection_hash(
        [("eigenvalues", eigenvalues)]
    ):
        raise ValueError(f"seed {seed}: diagnostic eigenvalue hash differs")
    if selection.get("anisotropy_sha256") != tensor_collection_hash([("anisotropy", anisotropy)]):
        raise ValueError(f"seed {seed}: diagnostic anisotropy hash differs")
    if selection.get("selected_sha256") != tensor_collection_hash(
        [("selected", torch.tensor(selected, dtype=torch.int64))]
    ):
        raise ValueError(f"seed {seed}: diagnostic selection hash differs")
    selected_initialization = initialization.subset(torch.tensor(selected, dtype=torch.int64))
    for field in ("means", "quats", "log_scales", "opacity", "sh"):
        if not torch.equal(getattr(target, field), getattr(selected_initialization, field)):
            raise ValueError(
                f"seed {seed}: diagnostic target {field} differs from selected initialization"
            )

    perturbation = preparation["perturbation"]
    if perturbation.get("seed") != 50_000 + seed:
        raise ValueError(f"seed {seed}: perturbation seed differs")
    require_close(
        perturbation.get("angle_degrees"), 20.0, f"seed {seed} perturbation angle", atol=0.0
    )
    raw_axes = torch.tensor(perturbation["raw_axes"], dtype=torch.float32)
    if raw_axes.shape != (128, 3) or not bool(torch.isfinite(raw_axes).all()):
        raise ValueError(f"seed {seed}: raw perturbation axes are invalid")
    axis_norms = torch.linalg.vector_norm(raw_axes, dim=-1)
    if bool((axis_norms < 1e-6).any()):
        raise ValueError(f"seed {seed}: perturbation axis is degenerate")
    axes = raw_axes / axis_norms[:, None]
    half_angle = math.radians(20.0) / 2.0
    delta = torch.cat(
        [
            torch.full((128, 1), math.cos(half_angle), dtype=torch.float32),
            axes * math.sin(half_angle),
        ],
        dim=-1,
    )
    perturbed = F.normalize(hamilton_product(delta, F.normalize(target.quats, dim=-1)), dim=-1)
    for key, derived in (
        ("raw_axis_norms", axis_norms),
        ("axes", axes),
        ("delta_quaternions", delta),
        ("perturbed_quaternions", perturbed),
    ):
        require_tensor_equal(perturbation[key], derived, f"seed {seed} perturbation {key}")
    if perturbation.get("raw_axes_sha256") != tensor_collection_hash([("raw_axes", raw_axes)]):
        raise ValueError(f"seed {seed}: raw perturbation-axis hash differs")
    if perturbation.get("perturbed_sha256") != tensor_collection_hash(
        [("perturbed_quaternions", perturbed)]
    ):
        raise ValueError(f"seed {seed}: perturbed-quaternion hash differs")

    expected_schedule = phase_a_schedule(seed)
    if preparation.get("schedule") != expected_schedule:
        raise ValueError(f"seed {seed}: preparation schedule differs")
    if preparation.get("schedule_sha256") != canonical_json_hash(expected_schedule):
        raise ValueError(f"seed {seed}: preparation schedule hash differs")
    return {"target": target, "perturbed_quaternions": perturbed}


def _equivalence_record(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    numerator, denominator, relative = relative_row_frobenius(candidate, reference)
    return {
        "max_absolute_error": float((candidate - reference).abs().max()),
        "frobenius_numerator": numerator,
        "frobenius_denominator": denominator,
        "relative_frobenius_error": relative,
    }


def representation_prerequisites(prepared: dict[str, Any]) -> dict[str, Any]:
    perturbed = prepared["perturbed"]
    representations = {scale_key(scale): perturbed.quats * scale for scale in RADIAL_SCALES}
    representations["antipode"] = -perturbed.quats
    reference_quaternions = representations["1"]
    reference_gaussians = gaussian_with_quaternions(perturbed, reference_quaternions)
    reference_rotations = quat_to_rotmat(reference_quaternions.to(torch.float64))
    reference_covariance = covariance_float64(reference_gaussians)
    reference_renders = render_views(reference_gaussians, prepared["scene"].cameras, sh_degree=0)
    records: dict[str, Any] = {}
    failures: list[str] = []
    for key, quaternions in representations.items():
        require_valid_quaternions(quaternions, f"representation {key}")
        gaussians = gaussian_with_quaternions(perturbed, quaternions)
        rotations = quat_to_rotmat(quaternions.to(torch.float64))
        covariance = covariance_float64(gaussians)
        outputs = render_views(gaussians, prepared["scene"].cameras, sh_degree=0)
        rotation_record = _equivalence_record(rotations, reference_rotations)
        covariance_record = _equivalence_record(covariance, reference_covariance)
        views = []
        color_numerator = 0.0
        color_denominator = 0.0
        for view, (output, reference) in enumerate(zip(outputs, reference_renders, strict=True)):
            numerator = float((output.color - reference.color).abs().to(torch.float64).sum())
            denominator = float(reference.color.abs().to(torch.float64).sum())
            if not math.isfinite(denominator) or denominator <= 0.0:
                raise ValueError("representation render color denominator is not positive")
            color_numerator += numerator
            color_denominator += denominator
            views.append(
                {
                    "view": view,
                    "color_max_absolute_error": float((output.color - reference.color).abs().max()),
                    "alpha_max_absolute_error": float((output.alpha - reference.alpha).abs().max()),
                    "depth_max_absolute_error": float((output.depth - reference.depth).abs().max()),
                    "color_l1_numerator": numerator,
                    "color_l1_denominator": denominator,
                    "color_l1_relative_error": numerator / denominator,
                }
            )
        record = {
            "quaternion_sha256": tensor_collection_hash([("quaternions", quaternions)]),
            "rotation_sha256": tensor_collection_hash([("rotations", rotations)]),
            "covariance_sha256": tensor_collection_hash([("covariance", covariance)]),
            "render_sha256": render_hash(outputs),
            "rotation": rotation_record,
            "covariance": covariance_record,
            "views": views,
            "pooled_color_l1_numerator": color_numerator,
            "pooled_color_l1_denominator": color_denominator,
            "pooled_color_l1_relative_error": color_numerator / color_denominator,
        }
        records[key] = record
        if rotation_record["max_absolute_error"] > 2e-12:
            failures.append(f"{key}: rotation absolute equivalence")
        if rotation_record["relative_frobenius_error"] > 2e-12:
            failures.append(f"{key}: rotation relative equivalence")
        if covariance_record["max_absolute_error"] > 2e-12:
            failures.append(f"{key}: covariance absolute equivalence")
        if covariance_record["relative_frobenius_error"] > 2e-12:
            failures.append(f"{key}: covariance relative equivalence")
        for view_record in views:
            if (
                max(
                    view_record["color_max_absolute_error"],
                    view_record["alpha_max_absolute_error"],
                    view_record["depth_max_absolute_error"],
                )
                > 5e-6
            ):
                failures.append(f"{key}: view {view_record['view']} render absolute equivalence")
            if view_record["color_l1_relative_error"] > 1e-6:
                failures.append(f"{key}: view {view_record['view']} render relative equivalence")
        if record["pooled_color_l1_relative_error"] > 1e-6:
            failures.append(f"{key}: render relative equivalence")
    return {
        "passed": not failures,
        "failures": failures,
        "reference": "1",
        "representations": records,
    }


def _gradient_row_record(quaternions: torch.Tensor, gradients: torch.Tensor) -> dict[str, Any]:
    quaternions64 = quaternions.to(torch.float64)
    gradients64 = gradients.to(torch.float64)
    q_norm = torch.linalg.vector_norm(quaternions64, dim=-1)
    gradient_norm = torch.linalg.vector_norm(gradients64, dim=-1)
    dot = (quaternions64 * gradients64).sum(dim=-1)
    active = gradient_norm > ACTIVE_NORM
    residual = torch.zeros_like(gradient_norm)
    residual[active] = dot[active].abs() / (q_norm[active] * gradient_norm[active])
    return {
        "quaternions": tensor_values(quaternions),
        "gradients": tensor_values(gradients),
        "quaternion_norms": tensor_values(q_norm),
        "gradient_norms": tensor_values(gradient_norm),
        "dot_q_gradient": tensor_values(dot),
        "active": tensor_values(active),
        "tangent_residual": tensor_values(residual),
        "active_count": int(active.sum()),
        "max_active_tangent_residual": float(residual[active].max()),
        "quaternion_sha256": tensor_collection_hash([("quaternions", quaternions)]),
        "gradient_sha256": tensor_collection_hash([("gradients", gradients)]),
    }


def gradient_prerequisites(prepared: dict[str, Any]) -> dict[str, Any]:
    first_view = prepared["schedule"][0]
    gradients: dict[str, torch.Tensor] = {}
    records: dict[str, Any] = {}
    failures: list[str] = []
    for scale in RADIAL_SCALES:
        key = scale_key(scale)
        q = torch.nn.Parameter((prepared["perturbed"].quats * scale).detach().clone())
        output = renderer().render(
            gaussian_with_quaternions(prepared["perturbed"], q),
            prepared["scene"].cameras[first_view],
            background=q.new_zeros(3),
            sh_degree=0,
        )
        loss = phase_a_loss(output.color, prepared["targets"][first_view].color)
        if not bool(torch.isfinite(loss)) or float(loss.detach()) <= 0.0:
            raise ValueError(f"scale {key} has a non-finite/non-positive step-zero loss")
        loss.backward()
        if q.grad is None or not bool(torch.isfinite(q.grad).all()):
            raise ValueError(f"scale {key} has an invalid step-zero gradient")
        gradient = q.grad.detach().clone()
        gradients[key] = gradient
        record = _gradient_row_record(q.detach(), gradient)
        record["loss"] = float(loss.detach())
        records[key] = record
        if record["active_count"] < 32:
            failures.append(f"{key}: fewer than 32 active step-zero gradients")
        if record["max_active_tangent_residual"] > 1e-5:
            failures.append(f"{key}: tangent residual exceeds tolerance")

    unit_q = prepared["perturbed"].quats
    unit_gradient = gradients["1"].to(torch.float64)
    unit_gradient_norm = torch.linalg.vector_norm(unit_gradient, dim=-1)
    removed_diagnostics = removed_gradient_diagnostics_float64(unit_q, gradients["1"])
    removed_numerator = removed_diagnostics["numerator"]
    removed_denominator = removed_diagnostics["denominator"]
    removed_ratio = removed_diagnostics["fraction"]
    projected_gradient = removed_diagnostics["projected_gradient"]
    if not bool(torch.isfinite(projected_gradient).all()):
        raise ValueError("explicit step-zero tangent projection is non-finite")
    records["1"]["removed_gradient"] = {
        "numerator": tensor_values(removed_numerator),
        "denominator": tensor_values(removed_denominator),
        "ratio": tensor_values(removed_ratio),
        "max_ratio": float(removed_ratio.max()),
        "projected_gradient": tensor_values(projected_gradient),
        "projected_gradient_sha256": tensor_collection_hash(
            [("projected_gradient", projected_gradient)]
        ),
    }
    if float(removed_ratio.max()) > 1e-5:
        failures.append("unit: explicit projection removed-gradient fraction exceeds tolerance")

    scaled_records: dict[str, Any] = {}
    for scale in (0.25, 4.0):
        key = scale_key(scale)
        numerator = torch.linalg.vector_norm(
            scale * gradients[key].to(torch.float64) - unit_gradient, dim=-1
        )
        denominator = unit_gradient_norm.clamp_min(ACTIVE_NORM)
        difference = numerator / denominator
        scaled_records[key] = {
            "numerator": tensor_values(numerator),
            "denominator": tensor_values(denominator),
            "difference": tensor_values(difference),
            "max_difference": float(difference.max()),
        }
        if float(difference.max()) > 5e-4:
            failures.append(f"{key}: scaled gradient identity exceeds tolerance")
    return {
        "passed": not failures,
        "failures": failures,
        "scheduled_view": first_view,
        "scales": records,
        "scaled_gradient_identities": scaled_records,
    }


def phase_a_prerequisites(prepared: dict[str, Any]) -> dict[str, Any]:
    representation = representation_prerequisites(prepared)
    gradients = gradient_prerequisites(prepared)
    failures = [*representation["failures"], *gradients["failures"]]
    return {
        "passed": not failures,
        "failures": failures,
        "representation": representation,
        "step_zero_gradients": gradients,
    }


def non_quaternion_hash(gaussians: Gaussians3D) -> str:
    return tensor_collection_hash(
        [
            ("means", gaussians.means),
            ("log_scales", gaussians.log_scales),
            ("opacity", gaussians.opacity),
            ("sh", gaussians.sh),
        ]
    )


def orientation_errors(candidate: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    candidate_unit = F.normalize(candidate.detach().to(torch.float64), dim=-1)
    target_unit = F.normalize(target.detach().to(torch.float64), dim=-1)
    dots = (candidate_unit * target_unit).sum(dim=-1).abs().clamp(0.0, 1.0)
    values = 2.0 * torch.acos(dots)
    raw = [float(value) for value in values]
    return {
        "rows": raw,
        "mean": statistics.fmean(raw),
        "median": linear_quantile(raw, 0.5),
        "p90": linear_quantile(raw, 0.9),
        "maximum": max(raw),
    }


def physical_equivalence(
    candidate: Gaussians3D,
    reference: Gaussians3D,
    cameras: list[Any],
) -> dict[str, Any]:
    candidate_covariance = covariance_float64(candidate)
    reference_covariance = covariance_float64(reference)
    covariance_record = _equivalence_record(candidate_covariance, reference_covariance)
    candidate_renders = render_views(candidate, cameras, sh_degree=0)
    reference_renders = render_views(reference, cameras, sh_degree=0)
    views = []
    pooled_numerator = 0.0
    pooled_denominator = 0.0
    for view, (output, target) in enumerate(zip(candidate_renders, reference_renders, strict=True)):
        numerator = float((output.color - target.color).abs().to(torch.float64).sum())
        denominator = float(target.color.abs().to(torch.float64).sum())
        if denominator <= 0.0 or not math.isfinite(denominator):
            raise ValueError("physical render equivalence denominator is invalid")
        pooled_numerator += numerator
        pooled_denominator += denominator
        views.append(
            {
                "view": view,
                "color_max_absolute_error": float((output.color - target.color).abs().max()),
                "alpha_max_absolute_error": float((output.alpha - target.alpha).abs().max()),
                "depth_max_absolute_error": float((output.depth - target.depth).abs().max()),
                "color_l1_numerator": numerator,
                "color_l1_denominator": denominator,
                "color_l1_relative_error": numerator / denominator,
            }
        )
    passed = (
        covariance_record["max_absolute_error"] <= 2e-12
        and covariance_record["relative_frobenius_error"] <= 2e-12
        and all(
            max(
                view["color_max_absolute_error"],
                view["alpha_max_absolute_error"],
                view["depth_max_absolute_error"],
            )
            <= 5e-6
            for view in views
        )
        and all(view["color_l1_relative_error"] <= 1e-6 for view in views)
        and pooled_numerator / pooled_denominator <= 1e-6
    )
    return {
        "passed": passed,
        "covariance": covariance_record,
        "views": views,
        "pooled_color_l1_numerator": pooled_numerator,
        "pooled_color_l1_denominator": pooled_denominator,
        "pooled_color_l1_relative_error": pooled_numerator / pooled_denominator,
        "candidate_covariance_sha256": tensor_collection_hash(
            [("covariance", candidate_covariance)]
        ),
        "reference_covariance_sha256": tensor_collection_hash(
            [("covariance", reference_covariance)]
        ),
        "candidate_render_sha256": render_hash(candidate_renders),
        "reference_render_sha256": render_hash(reference_renders),
    }


def phase_a_checkpoint(
    template: Gaussians3D,
    quaternions: torch.Tensor,
    target: Gaussians3D,
    targets: list[Any],
    cameras: list[Any],
    step: int,
    *,
    normalize_audit: bool,
) -> dict[str, Any]:
    gaussians = gaussian_with_quaternions(template, quaternions.detach().clone())
    validate_gaussians(gaussians, f"Phase-A checkpoint {step}", minimum_rows=128)
    outputs = render_views(gaussians, cameras, sh_degree=0)
    per_view = []
    pooled_sse = 0.0
    pooled_count = 0
    losses = []
    for view, (output, target_output) in enumerate(zip(outputs, targets, strict=True)):
        error = output.color.to(torch.float64) - target_output.color.to(torch.float64)
        sse = float(error.square().sum())
        count = error.numel()
        loss = float(phase_a_loss(output.color, target_output.color))
        pooled_sse += sse
        pooled_count += count
        losses.append(loss)
        per_view.append(
            {
                "view": view,
                "color_sse": sse,
                "color_count": count,
                "color_mse": sse / count,
                "color_psnr": psnr_from_sse(sse, count),
                "loss": loss,
                "color_sha256": tensor_collection_hash([("color", output.color)]),
                "alpha_sha256": tensor_collection_hash([("alpha", output.alpha)]),
                "depth_sha256": tensor_collection_hash([("depth", output.depth)]),
            }
        )
    covariance = covariance_float64(gaussians)
    target_covariance = covariance_float64(target)
    cov_num, cov_den, cov_rel = relative_row_frobenius(covariance, target_covariance)
    norms = torch.linalg.vector_norm(quaternions.detach().to(torch.float64), dim=-1)
    result: dict[str, Any] = {
        "step": step,
        "per_view": per_view,
        "pooled_color_sse": pooled_sse,
        "pooled_color_count": pooled_count,
        "pooled_color_mse": pooled_sse / pooled_count,
        "pooled_color_psnr": psnr_from_sse(pooled_sse, pooled_count),
        "mean_loss": statistics.fmean(losses),
        "covariance_error": {
            "numerator": cov_num,
            "denominator": cov_den,
            "relative": cov_rel,
        },
        "orientation_error": orientation_errors(quaternions, target.quats),
        "raw_quaternion_norms": tensor_values(norms),
        "raw_quaternions": tensor_values(quaternions),
        "covariance": tensor_values(covariance),
        "field_hash": gaussians_hash(gaussians),
        "field_shapes": {
            field: list(getattr(gaussians, field).shape)
            for field in ("means", "quats", "log_scales", "opacity", "sh")
        },
        "non_quaternion_hash": non_quaternion_hash(gaussians),
        "quaternion_hash": tensor_collection_hash([("quaternions", quaternions)]),
        "covariance_hash": tensor_collection_hash([("covariance", covariance)]),
        "render_hash": render_hash(outputs),
    }
    if normalize_audit:
        normalized = gaussian_with_quaternions(gaussians, F.normalize(gaussians.quats, dim=-1))
        result["normalized_copy_audit"] = physical_equivalence(normalized, gaussians, cameras)
    return result


def optimizer_state_record(
    optimizer: torch.optim.Optimizer, parameter: torch.Tensor
) -> dict[str, Any]:
    """Serialize the one-parameter Adam ordering and raw state behind its audit hash."""
    if len(optimizer.param_groups) != 1 or optimizer.param_groups[0]["params"] != [parameter]:
        raise RuntimeError("Phase-A optimizer parameter ordering is not the frozen singleton")
    state = optimizer.state[parameter]
    values: list[tuple[str, torch.Tensor]] = []
    entries = []
    for key in sorted(state):
        value = state[key]
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        detached = value.detach().clone()
        values.append((str(key), detached))
        entries.append(
            {
                "key": str(key),
                "dtype": str(detached.dtype),
                "shape": list(detached.shape),
                "values": tensor_values(detached),
            }
        )
    parameter_hash = tensor_collection_hash([("parameter", parameter.detach())])
    return {
        "parameter_count": 1,
        "parameter_shape": list(parameter.shape),
        "parameter_sha256": parameter_hash,
        "ordered_parameter_sha256": [parameter_hash],
        "state_entries": entries,
        "state_sha256": tensor_collection_hash(values),
    }


def optimizer_state_hash(optimizer: torch.optim.Optimizer, parameter: torch.Tensor) -> str:
    return optimizer_state_record(optimizer, parameter)["state_sha256"]


def _torch_dtype(name: str) -> torch.dtype:
    choices = {
        "torch.bool": torch.bool,
        "torch.int64": torch.int64,
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
    }
    if name not in choices:
        raise ValueError(f"unsupported serialized optimizer dtype {name!r}")
    return choices[name]


def validate_optimizer_state_record(
    record: dict[str, Any], parameter: torch.Tensor, context: str
) -> dict[str, Any]:
    if record.get("parameter_count") != 1:
        raise ValueError(f"{context} parameter count differs")
    if record.get("parameter_shape") != list(parameter.shape):
        raise ValueError(f"{context} parameter shape differs")
    parameter_hash = tensor_collection_hash([("parameter", parameter)])
    if record.get("parameter_sha256") != parameter_hash:
        raise ValueError(f"{context} parameter hash differs")
    if record.get("ordered_parameter_sha256") != [parameter_hash]:
        raise ValueError(f"{context} parameter ordering/hash differs")
    entries = record.get("state_entries")
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        raise ValueError(f"{context} optimizer state schema differs")
    keys = [item.get("key") for item in entries]
    if keys != ["exp_avg", "exp_avg_sq", "step"]:
        raise ValueError(f"{context} optimizer state ordering differs")
    values: list[tuple[str, torch.Tensor]] = []
    canonical_entries = []
    for item in entries:
        if set(item) != {"key", "dtype", "shape", "values"}:
            raise ValueError(f"{context} optimizer state entry schema differs")
        value = torch.tensor(item["values"], dtype=_torch_dtype(item["dtype"]))
        if list(value.shape) != item["shape"]:
            raise ValueError(f"{context} optimizer state shape differs")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{context} optimizer state is non-finite")
        values.append((item["key"], value))
        canonical_entries.append(item)
    expected_shapes = [list(parameter.shape), list(parameter.shape), []]
    if [item["shape"] for item in entries] != expected_shapes:
        raise ValueError(f"{context} Adam state tensor shapes differ")
    state_hash = tensor_collection_hash(values)
    if record.get("state_sha256") != state_hash:
        raise ValueError(f"{context} optimizer state hash differs")
    return {"entries": canonical_entries, "sha256": state_hash}


def phase_a_optimizer_config() -> dict[str, Any]:
    return {
        "name": "torch.optim.Adam",
        "lr": 1e-3,
        "betas": [0.9, 0.999],
        "eps": 1e-15,
        "weight_decay": 0,
        "amsgrad": False,
        "foreach": False,
        "fused": False,
        "capturable": False,
        "differentiable": False,
    }


def _phase_a_step_record(
    q_old: torch.Tensor,
    gradient: torch.Tensor,
    q_star: torch.Tensor,
    q_new: torch.Tensor,
    *,
    step: int,
    view: int,
    loss: float,
    policy: str,
    optimizer_before_policy: dict[str, Any],
    optimizer_after_policy: dict[str, Any],
    projection: dict[str, Any] | None,
) -> dict[str, Any]:
    q_old64 = q_old.to(torch.float64)
    gradient64 = gradient.to(torch.float64)
    q_star64 = q_star.to(torch.float64)
    q_new64 = q_new.to(torch.float64)
    q_norm = torch.linalg.vector_norm(q_old64, dim=-1)
    gradient_norm = torch.linalg.vector_norm(gradient64, dim=-1)
    dot = (q_old64 * gradient64).sum(dim=-1)
    active_gradient = gradient_norm > ACTIVE_NORM
    tangent = torch.zeros_like(gradient_norm)
    tangent[active_gradient] = dot[active_gradient].abs() / (
        q_norm[active_gradient] * gradient_norm[active_gradient]
    )
    displacement = q_star64 - q_old64
    displacement_norm = torch.linalg.vector_norm(displacement, dim=-1)
    active_displacement = displacement_norm > ACTIVE_NORM
    radial_numerator = (F.normalize(q_old64, dim=-1) * displacement).sum(dim=-1).abs()
    radial_fraction = torch.zeros_like(displacement_norm)
    radial_fraction[active_displacement] = (
        radial_numerator[active_displacement] / displacement_norm[active_displacement]
    )
    pre_norm = q_norm
    star_norm = torch.linalg.vector_norm(q_star64, dim=-1)
    post_norm = torch.linalg.vector_norm(q_new64, dim=-1)
    old_unit = F.normalize(q_old64, dim=-1)
    new_unit = F.normalize(q_new64, dim=-1)
    angular = 2.0 * torch.acos((old_unit * new_unit).sum(dim=-1).abs().clamp(0.0, 1.0))
    raw_displacement = q_star - q_old
    diagnostic_arrays = [
        ("q_old_norm", pre_norm),
        ("gradient_norm", gradient_norm),
        ("dot_q_gradient", dot),
        ("gradient_active", active_gradient),
        ("tangent_residual", tangent),
        ("displacement_norm", displacement_norm),
        ("displacement_active", active_displacement),
        ("radial_displacement_numerator", radial_numerator),
        ("radial_displacement_fraction", radial_fraction),
        ("q_star_norm", star_norm),
        ("q_new_norm", post_norm),
        ("effective_lr_scale", (1.0 / star_norm - 1.0).abs()),
        ("physical_angular_step", angular),
    ]
    return {
        "step": step,
        "scheduled_view": view,
        "loss": loss,
        "finite": True,
        "policy": policy,
        "q_old": tensor_values(q_old),
        "gradient": tensor_values(gradient),
        "q_star": tensor_values(q_star),
        "adam_displacement": tensor_values(raw_displacement),
        "q_new": tensor_values(q_new),
        "q_old_norm": tensor_values(pre_norm),
        "gradient_norm": tensor_values(gradient_norm),
        "dot_q_gradient": tensor_values(dot),
        "gradient_active": tensor_values(active_gradient),
        "tangent_residual": tensor_values(tangent),
        "active_gradient_count": int(active_gradient.sum()),
        "inactive_gradient_count": int((~active_gradient).sum()),
        "displacement_norm": tensor_values(displacement_norm),
        "displacement_active": tensor_values(active_displacement),
        "radial_displacement_numerator": tensor_values(radial_numerator),
        "radial_displacement_fraction": tensor_values(radial_fraction),
        "active_displacement_count": int(active_displacement.sum()),
        "inactive_displacement_count": int((~active_displacement).sum()),
        "q_star_norm": tensor_values(star_norm),
        "q_new_norm": tensor_values(post_norm),
        "effective_lr_scale": tensor_values((1.0 / star_norm - 1.0).abs()),
        "physical_angular_step": tensor_values(angular),
        "projection": projection,
        "optimizer_before_policy": optimizer_before_policy,
        "optimizer_after_policy": optimizer_after_policy,
        "optimizer_state_before_policy": optimizer_before_policy["state_sha256"],
        "optimizer_state_after_policy": optimizer_after_policy["state_sha256"],
        "optimizer_state_unchanged_by_policy": (
            optimizer_before_policy["state_sha256"] == optimizer_after_policy["state_sha256"]
        ),
        "q_old_sha256": tensor_collection_hash([("q_old", q_old)]),
        "gradient_sha256": tensor_collection_hash([("gradient", gradient)]),
        "q_star_sha256": tensor_collection_hash([("q_star", q_star)]),
        "displacement_sha256": tensor_collection_hash([("adam_displacement", raw_displacement)]),
        "q_new_sha256": tensor_collection_hash([("q_new", q_new)]),
        "diagnostic_arrays_sha256": tensor_collection_hash(diagnostic_arrays),
    }


def run_phase_a_arm(prepared: dict[str, Any], policy: str, scale: float) -> dict[str, Any]:
    if policy not in PHASE_A_POLICIES or scale not in RADIAL_SCALES:
        raise ValueError("unknown Phase-A arm")
    raw = prepared["perturbed"].quats * scale
    if policy in CANONICALIZING_POLICIES:
        raw = F.normalize(raw, dim=-1)
    require_valid_quaternions(raw, "Phase-A arm initialization")
    q = torch.nn.Parameter(raw.detach().clone())
    optimizer = torch.optim.Adam(
        [q],
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-15,
        weight_decay=0,
        amsgrad=False,
        foreach=False,
        fused=False,
        capturable=False,
        differentiable=False,
    )
    checkpoints = {
        "0": phase_a_checkpoint(
            prepared["perturbed"],
            q,
            prepared["target"],
            prepared["targets"],
            prepared["scene"].cameras,
            0,
            normalize_audit=policy == "current",
        )
    }
    construction = physical_equivalence(
        gaussian_with_quaternions(prepared["perturbed"], q.detach()),
        prepared["perturbed"],
        prepared["scene"].cameras,
    )
    steps = []
    for step, view in enumerate(prepared["schedule"], start=1):
        require_valid_quaternions(q.detach(), f"Phase-A {policy} step {step} q_old")
        optimizer.zero_grad(set_to_none=True)
        output = renderer().render(
            gaussian_with_quaternions(prepared["perturbed"], q),
            prepared["scene"].cameras[view],
            background=q.new_zeros(3),
            sh_degree=0,
        )
        loss_tensor = phase_a_loss(output.color, prepared["targets"][view].color)
        if not bool(torch.isfinite(loss_tensor)):
            raise ValueError(f"Phase-A {policy} step {step} loss is non-finite")
        loss_tensor.backward()
        if q.grad is None or not bool(torch.isfinite(q.grad).all()):
            raise ValueError(f"Phase-A {policy} step {step} gradient is invalid")
        q_old = q.detach().clone()
        raw_gradient = q.grad.detach().clone()
        projection = None
        if policy == "gradient_projection_current":
            unit = F.normalize(q_old, dim=-1)
            removed_diagnostics = removed_gradient_diagnostics_float64(q_old, raw_gradient)
            numerator = removed_diagnostics["numerator"]
            denominator = removed_diagnostics["denominator"]
            ratio = removed_diagnostics["fraction"]
            projection = {
                "removed_numerator": tensor_values(numerator),
                "removed_denominator": tensor_values(denominator),
                "removed_fraction": tensor_values(ratio),
                "diagnostic_sha256": tensor_collection_hash(
                    [
                        ("removed_numerator", numerator),
                        ("removed_denominator", denominator),
                        ("removed_fraction", ratio),
                    ]
                ),
            }
            projected = raw_gradient - unit * (unit * raw_gradient).sum(dim=-1, keepdim=True)
            with torch.no_grad():
                q.grad.copy_(projected)
        optimizer.step()
        q_star = q.detach().clone()
        require_valid_quaternions(q_star, f"Phase-A {policy} step {step} q_star")
        optimizer_before = optimizer_state_record(optimizer, q)
        if policy == "unit_retraction":
            with torch.no_grad():
                q.copy_(F.normalize(q_star, dim=-1))
        elif policy == "tangent_displacement_retraction":
            if bool(((torch.linalg.vector_norm(q_old, dim=-1) - 1.0).abs() > 2e-5).any()):
                raise ValueError("tangent-displacement q_old is not unit length")
            displacement = q_star - q_old
            tangent = displacement - q_old * (q_old * displacement).sum(dim=-1, keepdim=True)
            with torch.no_grad():
                q.copy_(F.normalize(q_old + tangent, dim=-1))
        optimizer_after = optimizer_state_record(optimizer, q)
        q_new = q.detach().clone()
        require_valid_quaternions(q_new, f"Phase-A {policy} step {step} q_new")
        if policy in ("unit_retraction", "tangent_displacement_retraction") and bool(
            ((torch.linalg.vector_norm(q_new, dim=-1) - 1.0).abs() > 2e-5).any()
        ):
            raise ValueError(f"Phase-A {policy} step {step} violates unit norm")
        record = _phase_a_step_record(
            q_old,
            raw_gradient,
            q_star,
            q_new,
            step=step,
            view=view,
            loss=float(loss_tensor.detach()),
            policy=policy,
            optimizer_before_policy=optimizer_before,
            optimizer_after_policy=optimizer_after,
            projection=projection,
        )
        if record["active_gradient_count"] < 32:
            raise ValueError(f"Phase-A {policy} step {step} has too few active gradients")
        if record["active_displacement_count"] < 32:
            raise ValueError(f"Phase-A {policy} step {step} has too few active displacements")
        if not record["optimizer_state_unchanged_by_policy"]:
            raise ValueError(f"Phase-A {policy} step {step} mutated Adam state")
        steps.append(record)
        if step in PHASE_A_CHECKPOINTS:
            checkpoints[str(step)] = phase_a_checkpoint(
                prepared["perturbed"],
                q,
                prepared["target"],
                prepared["targets"],
                prepared["scene"].cameras,
                step,
                normalize_audit=policy == "current",
            )
    psnr_values = [checkpoints[str(step)]["pooled_color_psnr"] for step in PHASE_A_CHECKPOINTS]
    return {
        "policy": policy,
        "radial_scale": scale,
        "initial_quaternions": tensor_values(raw),
        "initial_quaternion_sha256": tensor_collection_hash([("initial_quaternions", raw)]),
        "non_quaternion_hash": non_quaternion_hash(prepared["perturbed"]),
        "target_hashes": prepared["target_hashes"],
        "schedule": prepared["schedule"],
        "schedule_sha256": prepared["schedule_sha256"],
        "optimizer": phase_a_optimizer_config(),
        "construction_equivalence": construction,
        "steps": steps,
        "checkpoints": checkpoints,
        "self_target_auc_db": normalized_auc(list(PHASE_A_CHECKPOINTS), psnr_values),
    }


def run_phase_a_seed_arms(prepared: dict[str, Any]) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {}
    for policy in PHASE_A_POLICIES:
        arms[policy] = {}
        for scale in RADIAL_SCALES:
            arms[policy][scale_key(scale)] = run_phase_a_arm(prepared, policy, scale)
    return arms


def derive_phase_a_checkpoint(
    checkpoint: dict[str, Any],
    *,
    expected_step: int | None = None,
    template: Gaussians3D | None = None,
    target: Gaussians3D | None = None,
) -> dict[str, Any]:
    if expected_step is not None and checkpoint.get("step") != expected_step:
        raise ValueError(f"Phase-A checkpoint {expected_step} step identity differs")
    views = checkpoint.get("per_view")
    if not isinstance(views, list) or len(views) != 9:
        raise ValueError("Phase-A checkpoint lacks nine per-view raw reductions")
    if [view.get("view") for view in views] != list(range(9)):
        raise ValueError("Phase-A checkpoint per-view order differs")
    pooled_sse = 0.0
    pooled_count = 0
    losses = []
    for view in views:
        sse = float(view["color_sse"])
        count = int(view["color_count"])
        psnr = psnr_from_sse(sse, count)
        require_close(view["color_mse"], sse / count, "Phase-A per-view MSE")
        require_close(view["color_psnr"], psnr, "Phase-A per-view PSNR")
        if not math.isfinite(float(view["loss"])):
            raise ValueError("Phase-A per-view loss is non-finite")
        for field in ("color_sha256", "alpha_sha256", "depth_sha256"):
            require_sha256(view.get(field), f"Phase-A per-view {field}")
        pooled_sse += sse
        pooled_count += count
        losses.append(float(view["loss"]))
    require_close(checkpoint["pooled_color_sse"], pooled_sse, "Phase-A pooled color SSE")
    if checkpoint.get("pooled_color_count") != pooled_count:
        raise ValueError("Phase-A pooled color count differs from per-view evidence")
    require_close(
        checkpoint["pooled_color_mse"], pooled_sse / pooled_count, "Phase-A pooled color MSE"
    )
    pooled_psnr = psnr_from_sse(pooled_sse, pooled_count)
    require_close(checkpoint["pooled_color_psnr"], pooled_psnr, "Phase-A pooled color PSNR")
    require_close(checkpoint["mean_loss"], statistics.fmean(losses), "Phase-A mean loss")

    if template is not None or target is not None:
        if template is None or target is None:
            raise ValueError("Phase-A checkpoint reconstruction inputs are incomplete")
        quaternions = torch.tensor(checkpoint["raw_quaternions"], dtype=torch.float32)
        candidate = gaussian_with_quaternions(template, quaternions)
        validate_gaussians(candidate, "serialized Phase-A checkpoint", minimum_rows=128)
        if candidate.n != 128:
            raise ValueError("serialized Phase-A checkpoint topology differs")
        norms = torch.linalg.vector_norm(quaternions.to(torch.float64), dim=-1)
        require_tensor_equal(checkpoint["raw_quaternion_norms"], norms, "Phase-A raw norms")
        covariance = covariance_float64(candidate)
        require_tensor_equal(checkpoint["covariance"], covariance, "Phase-A covariance")
        if checkpoint.get("quaternion_hash") != tensor_collection_hash(
            [("quaternions", quaternions)]
        ):
            raise ValueError("Phase-A checkpoint quaternion hash differs")
        if checkpoint.get("covariance_hash") != tensor_collection_hash(
            [("covariance", covariance)]
        ):
            raise ValueError("Phase-A checkpoint covariance hash differs")
        if checkpoint.get("field_hash") != gaussians_hash(candidate):
            raise ValueError("Phase-A checkpoint field hash differs")
        if checkpoint.get("non_quaternion_hash") != non_quaternion_hash(candidate):
            raise ValueError("Phase-A checkpoint non-quaternion hash differs")
        expected_shapes = {
            field: list(getattr(candidate, field).shape)
            for field in ("means", "quats", "log_scales", "opacity", "sh")
        }
        if checkpoint.get("field_shapes") != expected_shapes:
            raise ValueError("Phase-A checkpoint field shapes differ")
        require_sha256(checkpoint.get("render_hash"), "Phase-A checkpoint render hash")
        cov_num, cov_den, cov_rel = relative_row_frobenius(covariance, covariance_float64(target))
        covariance_error = checkpoint["covariance_error"]
        for key, value in (
            ("numerator", cov_num),
            ("denominator", cov_den),
            ("relative", cov_rel),
        ):
            require_close(
                covariance_error[key], value, f"Phase-A checkpoint covariance error {key}"
            )
        derived_orientation = orientation_errors(quaternions, target.quats)
        if checkpoint.get("orientation_error") != derived_orientation:
            raise ValueError("Phase-A checkpoint orientation summary differs from raw quaternions")
    return {"pooled_color_psnr": pooled_psnr, "pooled_color_sse": pooled_sse}


def derive_phase_a_auc(arm: dict[str, Any]) -> float:
    checkpoints = arm["checkpoints"]
    if set(map(int, checkpoints)) != set(PHASE_A_CHECKPOINTS):
        raise ValueError("Phase-A checkpoint set differs while deriving AUC")
    psnr_values = [
        derive_phase_a_checkpoint(checkpoints[str(step)])["pooled_color_psnr"]
        for step in PHASE_A_CHECKPOINTS
    ]
    derived_auc = normalized_auc(list(PHASE_A_CHECKPOINTS), psnr_values)
    require_close(arm.get("self_target_auc_db"), derived_auc, "stored Phase-A AUC")
    return derived_auc


def derive_phase_a_step_metrics(
    step: dict[str, Any],
    *,
    expected_policy: str | None = None,
    expected_step: int | None = None,
    expected_view: int | None = None,
    expected_q_old: torch.Tensor | None = None,
) -> dict[str, Any]:
    q_old32 = torch.tensor(step["q_old"], dtype=torch.float32)
    gradient32 = torch.tensor(step["gradient"], dtype=torch.float32)
    q_star32 = torch.tensor(step["q_star"], dtype=torch.float32)
    q_new32 = torch.tensor(step["q_new"], dtype=torch.float32)
    if (
        q_old32.shape != gradient32.shape
        or q_old32.shape != q_star32.shape
        or q_old32.shape != q_new32.shape
        or q_old32.ndim != 2
        or q_old32.shape[1] != 4
    ):
        raise ValueError("stored Phase-A raw step arrays have invalid shapes")
    if not all(
        bool(torch.isfinite(value).all()) for value in (q_old32, gradient32, q_star32, q_new32)
    ):
        raise ValueError("stored Phase-A raw step arrays are non-finite")
    if expected_policy is not None and step.get("policy") != expected_policy:
        raise ValueError("stored Phase-A step policy differs")
    policy = step["policy"]
    if policy not in PHASE_A_POLICIES:
        raise ValueError("stored Phase-A step policy is unknown")
    if expected_step is not None and step.get("step") != expected_step:
        raise ValueError("stored Phase-A step order differs")
    if expected_view is not None and step.get("scheduled_view") != expected_view:
        raise ValueError("stored Phase-A schedule position differs")
    if expected_q_old is not None and not torch.equal(q_old32, expected_q_old):
        raise ValueError("stored Phase-A optimizer endpoint ordering differs")
    if step.get("finite") is not True or not math.isfinite(float(step["loss"])):
        raise ValueError("stored Phase-A finite/loss evidence differs")

    q_old = q_old32.to(torch.float64)
    gradient = gradient32.to(torch.float64)
    q_star = q_star32.to(torch.float64)
    q_new = q_new32.to(torch.float64)
    q_old_norm = torch.linalg.vector_norm(q_old, dim=-1)
    gradient_norm = torch.linalg.vector_norm(gradient, dim=-1)
    dot = (q_old * gradient).sum(dim=-1)
    gradient_active = gradient_norm > ACTIVE_NORM
    tangent = torch.zeros_like(gradient_norm)
    tangent[gradient_active] = dot[gradient_active].abs() / (
        q_old_norm[gradient_active] * gradient_norm[gradient_active]
    )
    displacement = q_star - q_old
    raw_displacement32 = q_star32 - q_old32
    displacement_norm = torch.linalg.vector_norm(displacement, dim=-1)
    displacement_active = displacement_norm > ACTIVE_NORM
    radial_numerator = (F.normalize(q_old, dim=-1) * displacement).sum(dim=-1).abs()
    radial_fraction = torch.zeros_like(displacement_norm)
    radial_fraction[displacement_active] = (
        radial_numerator[displacement_active] / displacement_norm[displacement_active]
    )
    star_norm = torch.linalg.vector_norm(q_star, dim=-1)
    new_norm = torch.linalg.vector_norm(q_new, dim=-1)
    effective_lr = (1.0 / star_norm - 1.0).abs()
    angular = 2.0 * torch.acos(
        (F.normalize(q_old, dim=-1) * F.normalize(q_new, dim=-1)).sum(dim=-1).abs().clamp(0.0, 1.0)
    )
    arrays = {
        "q_old_norm": q_old_norm,
        "gradient_norm": gradient_norm,
        "dot_q_gradient": dot,
        "gradient_active": gradient_active,
        "tangent_residual": tangent,
        "displacement_norm": displacement_norm,
        "displacement_active": displacement_active,
        "radial_displacement_numerator": radial_numerator,
        "radial_displacement_fraction": radial_fraction,
        "q_star_norm": star_norm,
        "q_new_norm": new_norm,
        "effective_lr_scale": effective_lr,
        "physical_angular_step": angular,
    }
    require_tensor_equal(step["adam_displacement"], raw_displacement32, "Phase-A displacement")
    for key, value in arrays.items():
        require_tensor_equal(step[key], value, f"Phase-A step {key}")
    counts = {
        "active_gradient_count": int(gradient_active.sum()),
        "inactive_gradient_count": int((~gradient_active).sum()),
        "active_displacement_count": int(displacement_active.sum()),
        "inactive_displacement_count": int((~displacement_active).sum()),
    }
    for key, value in counts.items():
        if step.get(key) != value:
            raise ValueError(f"stored Phase-A {key} differs from raw arrays")

    hashes = {
        "q_old_sha256": tensor_collection_hash([("q_old", q_old32)]),
        "gradient_sha256": tensor_collection_hash([("gradient", gradient32)]),
        "q_star_sha256": tensor_collection_hash([("q_star", q_star32)]),
        "displacement_sha256": tensor_collection_hash([("adam_displacement", raw_displacement32)]),
        "q_new_sha256": tensor_collection_hash([("q_new", q_new32)]),
        "diagnostic_arrays_sha256": tensor_collection_hash(list(arrays.items())),
    }
    for key, value in hashes.items():
        if step.get(key) != value:
            raise ValueError(f"stored Phase-A {key} differs from raw arrays")

    optimizer_before = validate_optimizer_state_record(
        step["optimizer_before_policy"], q_star32, "Phase-A pre-policy optimizer"
    )
    optimizer_after = validate_optimizer_state_record(
        step["optimizer_after_policy"], q_new32, "Phase-A post-policy optimizer"
    )
    if optimizer_before != optimizer_after:
        raise ValueError("stored Phase-A optimizer state changed across policy")
    if step.get("optimizer_state_before_policy") != optimizer_before["sha256"]:
        raise ValueError("stored Phase-A pre-policy optimizer hash differs")
    if step.get("optimizer_state_after_policy") != optimizer_after["sha256"]:
        raise ValueError("stored Phase-A post-policy optimizer hash differs")
    if step.get("optimizer_state_unchanged_by_policy") is not True:
        raise ValueError("stored Phase-A optimizer-state equality flag differs")

    if policy in ("current", "entry_canonical", "gradient_projection_current"):
        expected_q_new = q_star32
    elif policy == "unit_retraction":
        expected_q_new = F.normalize(q_star32, dim=-1)
    else:
        if bool(((torch.linalg.vector_norm(q_old32, dim=-1) - 1.0).abs() > 2e-5).any()):
            raise ValueError("stored tangent-displacement q_old is not unit length")
        delta = q_star32 - q_old32
        tangent_step = delta - q_old32 * (q_old32 * delta).sum(dim=-1, keepdim=True)
        expected_q_new = F.normalize(q_old32 + tangent_step, dim=-1)
    if not torch.equal(q_new32, expected_q_new):
        raise ValueError("stored Phase-A policy equation differs from raw endpoints")
    if policy == "gradient_projection_current":
        derive_projection_removed_fractions(step)
    elif step.get("projection") is not None:
        raise ValueError("non-projection Phase-A step contains projection evidence")
    return {
        "effective_lr_scale": [float(value) for value in effective_lr],
        "radial_displacement_fraction": [
            float(value) for value in radial_fraction[displacement_active]
        ],
        "active_count": int(displacement_active.sum()),
        "q_new": q_new32,
    }


def derive_projection_removed_fractions(step: dict[str, Any]) -> list[float]:
    if step.get("policy") != "gradient_projection_current" or step.get("projection") is None:
        raise ValueError("projection evidence requested from a non-projection step")
    q_old = torch.tensor(step["q_old"], dtype=torch.float32)
    gradient = torch.tensor(step["gradient"], dtype=torch.float32)
    removed_diagnostics = removed_gradient_diagnostics_float64(q_old, gradient)
    numerator = removed_diagnostics["numerator"]
    denominator = removed_diagnostics["denominator"]
    fraction = removed_diagnostics["fraction"]
    projection = step["projection"]
    stored = (
        torch.tensor(projection["removed_numerator"], dtype=torch.float64),
        torch.tensor(projection["removed_denominator"], dtype=torch.float64),
        torch.tensor(projection["removed_fraction"], dtype=torch.float64),
    )
    if not all(
        torch.equal(candidate, expected)
        for candidate, expected in zip(stored, (numerator, denominator, fraction), strict=True)
    ):
        raise ValueError("stored projected-gradient fractions differ from raw q/gradient")
    expected_hash = tensor_collection_hash(
        [
            ("removed_numerator", numerator),
            ("removed_denominator", denominator),
            ("removed_fraction", fraction),
        ]
    )
    if projection.get("diagnostic_sha256") != expected_hash:
        raise ValueError("stored projected-gradient diagnostic hash differs")
    return [float(value) for value in fraction]


def derive_physical_equivalence(
    record: dict[str, Any],
    *,
    candidate: Gaussians3D | None = None,
    reference: Gaussians3D | None = None,
    context: str,
) -> bool:
    """Derive construction validity from row-level reductions, not its stored pass flag."""
    covariance = record["covariance"]
    for field in ("max_absolute_error", "frobenius_numerator", "relative_frobenius_error"):
        require_close(covariance[field], float(covariance[field]), f"{context} covariance {field}")
        if covariance[field] < 0.0:
            raise ValueError(f"{context} covariance {field} is negative")
    if covariance["frobenius_denominator"] <= 0.0 or not math.isfinite(
        float(covariance["frobenius_denominator"])
    ):
        raise ValueError(f"{context} covariance denominator is invalid")
    require_close(
        covariance["relative_frobenius_error"],
        covariance["frobenius_numerator"] / covariance["frobenius_denominator"],
        f"{context} covariance relative error",
        atol=1e-15,
    )
    if candidate is not None or reference is not None:
        if candidate is None or reference is None:
            raise ValueError(f"{context} covariance inputs are incomplete")
        derived_covariance = _equivalence_record(
            covariance_float64(candidate), covariance_float64(reference)
        )
        for key, value in derived_covariance.items():
            require_close(covariance[key], value, f"{context} covariance {key}", atol=1e-15)
        if record.get("candidate_covariance_sha256") != tensor_collection_hash(
            [("covariance", covariance_float64(candidate))]
        ):
            raise ValueError(f"{context} candidate covariance hash differs")
        if record.get("reference_covariance_sha256") != tensor_collection_hash(
            [("covariance", covariance_float64(reference))]
        ):
            raise ValueError(f"{context} reference covariance hash differs")
    require_sha256(record.get("candidate_render_sha256"), f"{context} candidate render hash")
    require_sha256(record.get("reference_render_sha256"), f"{context} reference render hash")

    views = record["views"]
    if [view.get("view") for view in views] != list(range(9)):
        raise ValueError(f"{context} render view order differs")
    pooled_numerator = 0.0
    pooled_denominator = 0.0
    for view in views:
        numerator = view["color_l1_numerator"]
        denominator = view["color_l1_denominator"]
        for field in (
            "color_max_absolute_error",
            "alpha_max_absolute_error",
            "depth_max_absolute_error",
            "color_l1_numerator",
        ):
            require_close(view[field], float(view[field]), f"{context} view {field}")
            if view[field] < 0.0:
                raise ValueError(f"{context} view {field} is negative")
        if denominator <= 0.0 or not math.isfinite(float(denominator)):
            raise ValueError(f"{context} view denominator is invalid")
        require_close(
            view["color_l1_relative_error"],
            numerator / denominator,
            f"{context} view {view['view']} relative error",
            atol=1e-15,
        )
        pooled_numerator += numerator
        pooled_denominator += denominator
    require_close(
        record["pooled_color_l1_numerator"],
        pooled_numerator,
        f"{context} pooled numerator",
    )
    require_close(
        record["pooled_color_l1_denominator"],
        pooled_denominator,
        f"{context} pooled denominator",
    )
    require_close(
        record["pooled_color_l1_relative_error"],
        pooled_numerator / pooled_denominator,
        f"{context} pooled relative error",
        atol=1e-15,
    )
    passed = (
        covariance["max_absolute_error"] <= 2e-12
        and covariance["relative_frobenius_error"] <= 2e-12
        and all(
            max(
                view["color_max_absolute_error"],
                view["alpha_max_absolute_error"],
                view["depth_max_absolute_error"],
            )
            <= 5e-6
            for view in views
        )
        and all(view["color_l1_relative_error"] <= 1e-6 for view in views)
        and pooled_numerator / pooled_denominator <= 1e-6
    )
    if record.get("passed") is not None and record["passed"] is not passed:
        raise ValueError(f"{context} stored pass flag differs from raw reductions")
    return passed


def _stored_physical_equivalence_passes(record: dict[str, Any]) -> bool:
    return derive_physical_equivalence(record, context="stored physical equivalence")


def recompute_prerequisite_validity(
    prerequisite: dict[str, Any], preparation: dict[str, Any] | None = None
) -> dict[str, Any]:
    failures: list[str] = []
    prepared = None
    if preparation is not None:
        prepared = validate_phase_a_preparation(preparation, preparation["seed"])

    representation = prerequisite["representation"]
    if representation.get("reference") != "1":
        raise ValueError("stored representation reference differs")
    representations = representation["representations"]
    if set(representations) != {"0.25", "1", "4", "antipode"}:
        raise ValueError("representation key set differs")
    reference_rotation = None
    reference_covariance = None
    if prepared is not None:
        perturbed = prepared["perturbed_quaternions"]
        reference_gaussians = gaussian_with_quaternions(prepared["target"], perturbed)
        reference_rotation = quat_to_rotmat(perturbed.to(torch.float64))
        reference_covariance = covariance_float64(reference_gaussians)
    for key, record in representations.items():
        for field in ("rotation", "covariance"):
            item = record[field]
            for summary in (
                "max_absolute_error",
                "frobenius_numerator",
                "relative_frobenius_error",
            ):
                require_close(item[summary], float(item[summary]), f"{key} {field} {summary}")
                if item[summary] < 0.0:
                    raise ValueError(f"{key}: {field} {summary} is negative")
            if item["frobenius_denominator"] <= 0.0 or not math.isfinite(
                float(item["frobenius_denominator"])
            ):
                raise ValueError(f"{key}: {field} denominator invalid")
            require_close(
                item["relative_frobenius_error"],
                item["frobenius_numerator"] / item["frobenius_denominator"],
                f"{key} {field} relative reduction",
                atol=1e-15,
            )
        if prepared is not None:
            quaternions = {
                "0.25": 0.25 * prepared["perturbed_quaternions"],
                "1": prepared["perturbed_quaternions"],
                "4": 4.0 * prepared["perturbed_quaternions"],
                "antipode": -prepared["perturbed_quaternions"],
            }[key]
            rotations = quat_to_rotmat(quaternions.to(torch.float64))
            covariance = covariance_float64(
                gaussian_with_quaternions(prepared["target"], quaternions)
            )
            for field, derived in (
                ("rotation", _equivalence_record(rotations, reference_rotation)),
                ("covariance", _equivalence_record(covariance, reference_covariance)),
            ):
                for summary, value in derived.items():
                    require_close(
                        record[field][summary],
                        value,
                        f"{key} {field} {summary}",
                        atol=1e-15,
                    )
            expected_hashes = {
                "quaternion_sha256": tensor_collection_hash([("quaternions", quaternions)]),
                "rotation_sha256": tensor_collection_hash([("rotations", rotations)]),
                "covariance_sha256": tensor_collection_hash([("covariance", covariance)]),
            }
            for field, value in expected_hashes.items():
                if record.get(field) != value:
                    raise ValueError(f"{key}: {field} differs from reconstructed representation")
        require_sha256(record.get("render_sha256"), f"{key} representation render hash")
        views = record["views"]
        if [view.get("view") for view in views] != list(range(9)):
            raise ValueError(f"{key}: representation render view order differs")
        pooled_numerator = 0.0
        pooled_denominator = 0.0
        for view in views:
            denominator = view["color_l1_denominator"]
            numerator = view["color_l1_numerator"]
            for field in (
                "color_max_absolute_error",
                "alpha_max_absolute_error",
                "depth_max_absolute_error",
                "color_l1_numerator",
            ):
                require_close(view[field], float(view[field]), f"{key} view {field}")
                if view[field] < 0.0:
                    raise ValueError(f"{key}: view {field} is negative")
            if denominator <= 0.0 or not math.isfinite(float(denominator)):
                raise ValueError(f"{key}: view denominator invalid")
            require_close(
                view["color_l1_relative_error"],
                numerator / denominator,
                f"{key} view relative render error",
                atol=1e-15,
            )
            pooled_numerator += numerator
            pooled_denominator += denominator
            if (
                max(
                    view["color_max_absolute_error"],
                    view["alpha_max_absolute_error"],
                    view["depth_max_absolute_error"],
                )
                > 5e-6
            ):
                failures.append(f"{key}: view {view['view']} render absolute equivalence")
            if view["color_l1_relative_error"] > 1e-6:
                failures.append(f"{key}: view {view['view']} render relative equivalence")
        require_close(
            record["pooled_color_l1_numerator"], pooled_numerator, f"{key} pooled numerator"
        )
        require_close(
            record["pooled_color_l1_denominator"],
            pooled_denominator,
            f"{key} pooled denominator",
        )
        require_close(
            record["pooled_color_l1_relative_error"],
            pooled_numerator / pooled_denominator,
            f"{key} pooled relative render error",
            atol=1e-15,
        )
        for field in ("rotation", "covariance"):
            item = record[field]
            if item["max_absolute_error"] > 2e-12:
                failures.append(f"{key}: {field} absolute equivalence")
            if item["relative_frobenius_error"] > 2e-12:
                failures.append(f"{key}: {field} relative equivalence")
        if pooled_denominator <= 0.0:
            raise ValueError(f"{key}: pooled render denominator invalid")
        if pooled_numerator / pooled_denominator > 1e-6:
            failures.append(f"{key}: render relative equivalence")

    representation_passed = not failures
    if representation.get("passed") is not representation_passed:
        raise ValueError("stored representation validity flag differs from raw evidence")
    if representation.get("failures") != failures:
        raise ValueError("stored representation failure summary differs from raw evidence")

    representation_failures = list(failures)
    gradient_failures: list[str] = []
    gradient_evidence = prerequisite["step_zero_gradients"]
    scale_records = gradient_evidence["scales"]
    if set(scale_records) != {"0.25", "1", "4"}:
        raise ValueError("step-zero gradient scale set differs")
    if preparation is not None:
        expected_schedule = phase_a_schedule(preparation["seed"])
        if gradient_evidence.get("scheduled_view") != expected_schedule[0]:
            raise ValueError("step-zero gradient scheduled view differs")
    reconstructed: dict[str, torch.Tensor] = {}
    reconstructed_raw: dict[str, torch.Tensor] = {}
    for key in ("0.25", "1", "4"):
        record = scale_records[key]
        quaternions32 = torch.tensor(record["quaternions"], dtype=torch.float32)
        gradients32 = torch.tensor(record["gradients"], dtype=torch.float32)
        if quaternions32.shape != (128, 4) or gradients32.shape != (128, 4):
            raise ValueError(f"{key}: step-zero gradient row shape differs")
        quaternions = quaternions32.to(torch.float64)
        gradients = gradients32.to(torch.float64)
        gradient_norm = torch.linalg.vector_norm(gradients, dim=-1)
        q_norm = torch.linalg.vector_norm(quaternions, dim=-1)
        active = gradient_norm > ACTIVE_NORM
        residual = torch.zeros_like(gradient_norm)
        dot = (quaternions * gradients).sum(dim=-1)
        residual[active] = dot[active].abs() / (q_norm[active] * gradient_norm[active])
        for field, value in (
            ("quaternion_norms", q_norm),
            ("gradient_norms", gradient_norm),
            ("dot_q_gradient", dot),
            ("active", active),
            ("tangent_residual", residual),
        ):
            require_tensor_equal(record[field], value, f"{key} step-zero {field}")
        if record.get("active_count") != int(active.sum()):
            raise ValueError(f"{key}: stored active step-zero count differs")
        require_close(
            record["max_active_tangent_residual"],
            float(residual[active].max()),
            f"{key} maximum active tangent residual",
        )
        if record.get("quaternion_sha256") != tensor_collection_hash(
            [("quaternions", quaternions32)]
        ):
            raise ValueError(f"{key}: step-zero quaternion hash differs")
        if record.get("gradient_sha256") != tensor_collection_hash([("gradients", gradients32)]):
            raise ValueError(f"{key}: step-zero gradient hash differs")
        if not math.isfinite(float(record["loss"])) or float(record["loss"]) <= 0.0:
            raise ValueError(f"{key}: step-zero loss is invalid")
        reconstructed[key] = gradients
        reconstructed_raw[key] = gradients32
        if int(active.sum()) < 32:
            gradient_failures.append(f"{key}: fewer than 32 active step-zero gradients")
        if float(residual[active].max()) > 1e-5:
            gradient_failures.append(f"{key}: tangent residual exceeds tolerance")
    unit_gradient = reconstructed["1"]
    unit_q = torch.tensor(scale_records["1"]["quaternions"], dtype=torch.float32)
    removed_diagnostics = removed_gradient_diagnostics_float64(unit_q, reconstructed_raw["1"])
    removed_numerator = removed_diagnostics["numerator"]
    removed_denominator = removed_diagnostics["denominator"]
    removed = removed_diagnostics["fraction"]
    expected_projection = removed_diagnostics["projected_gradient"]
    removed_record = scale_records["1"]["removed_gradient"]
    for field, value in (
        ("numerator", removed_numerator),
        ("denominator", removed_denominator),
        ("ratio", removed),
        ("projected_gradient", expected_projection),
    ):
        require_tensor_equal(removed_record[field], value, f"unit removed-gradient {field}")
    require_close(
        removed_record["max_ratio"], float(removed.max()), "unit removed-gradient max ratio"
    )
    if removed_record.get("projected_gradient_sha256") != tensor_collection_hash(
        [("projected_gradient", expected_projection)]
    ):
        raise ValueError("unit projected-gradient hash differs")
    if float(removed.max()) > 1e-5:
        gradient_failures.append(
            "unit: explicit projection removed-gradient fraction exceeds tolerance"
        )
    scaled_records = gradient_evidence["scaled_gradient_identities"]
    if set(scaled_records) != {"0.25", "4"}:
        raise ValueError("scaled-gradient identity key set differs")
    for key, scale in (("0.25", 0.25), ("4", 4.0)):
        numerator = torch.linalg.vector_norm(scale * reconstructed[key] - unit_gradient, dim=-1)
        denominator = torch.linalg.vector_norm(unit_gradient, dim=-1).clamp_min(ACTIVE_NORM)
        difference = numerator / denominator
        record = scaled_records[key]
        for field, value in (
            ("numerator", numerator),
            ("denominator", denominator),
            ("difference", difference),
        ):
            require_tensor_equal(record[field], value, f"{key} scaled-gradient {field}")
        require_close(
            record["max_difference"],
            float(difference.max()),
            f"{key} scaled-gradient maximum",
        )
        if float(difference.max()) > 5e-4:
            gradient_failures.append(f"{key}: scaled gradient identity exceeds tolerance")
    gradient_passed = not gradient_failures
    if gradient_evidence.get("passed") is not gradient_passed:
        raise ValueError("stored gradient validity flag differs from raw evidence")
    if gradient_evidence.get("failures") != gradient_failures:
        raise ValueError("stored gradient failure summary differs from raw evidence")

    all_failures = [*representation_failures, *gradient_failures]
    passed = not all_failures
    if prerequisite.get("passed") is not passed:
        raise ValueError("stored prerequisite validity flag differs from raw evidence")
    if prerequisite.get("failures") != all_failures:
        raise ValueError("stored prerequisite failure summary differs from raw evidence")
    return {"passed": passed, "failures": all_failures}


def _covariance_pair_reduction(
    left: list[Any], right: list[Any], denominator_covariance: list[Any]
) -> dict[str, float]:
    left_tensor = torch.tensor(left, dtype=torch.float64)
    right_tensor = torch.tensor(right, dtype=torch.float64)
    denominator_tensor = torch.tensor(denominator_covariance, dtype=torch.float64)
    numerator = float(torch.linalg.matrix_norm(left_tensor - right_tensor, ord="fro").sum())
    denominator = float(torch.linalg.matrix_norm(denominator_tensor, ord="fro").sum())
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("stored covariance denominator is invalid")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "relative": numerator / denominator,
    }


def _phase_a_seed_decision(seed_record: dict[str, Any]) -> dict[str, Any]:
    arms = seed_record["arms"]["current"]
    auc_values = [derive_phase_a_auc(arms[scale_key(scale)]) for scale in RADIAL_SCALES]
    gauge_auc_spread = max(auc_values) - min(auc_values)
    denominator = arms["1"]["checkpoints"]["40"]["covariance"]
    pair_records = {}
    for left, right in (("0.25", "1"), ("0.25", "4"), ("1", "4")):
        pair_records[f"{left}_vs_{right}"] = _covariance_pair_reduction(
            arms[left]["checkpoints"]["40"]["covariance"],
            arms[right]["checkpoints"]["40"]["covariance"],
            denominator,
        )
    gauge_cov_spread = max(item["relative"] for item in pair_records.values())
    unit_steps = [derive_phase_a_step_metrics(step) for step in arms["1"]["steps"]]
    effective_lr = [float(value) for step in unit_steps for value in step["effective_lr_scale"]]
    radial_fraction = [
        float(value) for step in unit_steps for value in step["radial_displacement_fraction"]
    ]
    effective_p90 = linear_quantile(effective_lr, 0.9)
    radial_median = linear_quantile(radial_fraction, 0.5)
    tests = {
        "gauge_auc_spread": gauge_auc_spread >= 0.05,
        "gauge_cov_spread": gauge_cov_spread >= 0.001,
        "unit_effective_lr_p90": effective_p90 >= 0.01,
        "unit_radial_fraction_median": radial_median >= 0.10,
    }
    return {
        "seed": seed_record["seed"],
        "gauge_auc_spread_db": gauge_auc_spread,
        "gauge_cov_spread": gauge_cov_spread,
        "gauge_covariance_pairs": pair_records,
        "unit_effective_lr_values": effective_lr,
        "unit_effective_lr_p90": effective_p90,
        "unit_radial_fraction_values": radial_fraction,
        "unit_radial_fraction_median": radial_median,
        "tests": tests,
        "ambient_gauge_material": all(tests.values()),
    }


def _phase_a_invariants(seed_records: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    prerequisite_recomputations = {}
    canonical_collapse: dict[str, Any] = {}
    projection_per_seed: dict[str, Any] = {}
    pooled_projection: list[float] = []
    for seed_record in seed_records:
        seed = seed_record["seed"]
        prepared = validate_phase_a_preparation(seed_record["preparation"], seed)
        target = prepared["target"]
        perturbed = gaussian_with_quaternions(target, prepared["perturbed_quaternions"])
        prerequisite = recompute_prerequisite_validity(
            seed_record["prerequisites"], seed_record["preparation"]
        )
        prerequisite_recomputations[str(seed)] = prerequisite
        if not prerequisite["passed"]:
            failures.extend(f"seed {seed}: {item}" for item in prerequisite["failures"])
        arms = seed_record["arms"]
        if set(arms) != set(PHASE_A_POLICIES):
            failures.append(f"seed {seed}: policy set differs")
            continue
        expected_schedule = phase_a_schedule(seed)
        expected_nonquat = non_quaternion_hash(target)
        expected_targets = seed_record["preparation"]["target_hashes"]
        for policy in PHASE_A_POLICIES:
            if set(arms[policy]) != {"0.25", "1", "4"}:
                failures.append(f"seed {seed} {policy}: radial set differs")
                continue
            for scale, arm in arms[policy].items():
                if arm["policy"] != policy or scale_key(arm["radial_scale"]) != scale:
                    failures.append(f"seed {seed} {policy}/{scale}: arm identity differs")
                if arm["optimizer"] != phase_a_optimizer_config():
                    failures.append(f"seed {seed} {policy}/{scale}: optimizer config differs")
                initial = torch.tensor(arm["initial_quaternions"], dtype=torch.float32)
                radial_scale = float(arm["radial_scale"])
                expected_initial = prepared["perturbed_quaternions"] * radial_scale
                if policy in CANONICALIZING_POLICIES:
                    expected_initial = F.normalize(expected_initial, dim=-1)
                if initial.shape != (128, 4) or not torch.equal(initial, expected_initial):
                    failures.append(f"seed {seed} {policy}/{scale}: initial quaternion formula")
                if arm.get("initial_quaternion_sha256") != tensor_collection_hash(
                    [("initial_quaternions", initial)]
                ):
                    failures.append(f"seed {seed} {policy}/{scale}: initial quaternion hash")
                if arm["schedule"] != expected_schedule or arm.get(
                    "schedule_sha256"
                ) != canonical_json_hash(expected_schedule):
                    failures.append(f"seed {seed} {policy}/{scale}: schedule/hash differs")
                if len(arm["steps"]) != 40 or set(map(int, arm["checkpoints"])) != set(
                    PHASE_A_CHECKPOINTS
                ):
                    failures.append(f"seed {seed} {policy}/{scale}: trajectory incomplete")
                candidate = gaussian_with_quaternions(target, initial)
                if not derive_physical_equivalence(
                    arm["construction_equivalence"],
                    candidate=candidate,
                    reference=perturbed,
                    context=f"seed {seed} {policy}/{scale} construction",
                ):
                    failures.append(f"seed {seed} {policy}/{scale}: step-zero equivalence")
                if (
                    arm["non_quaternion_hash"] != expected_nonquat
                    or arm["target_hashes"] != expected_targets
                ):
                    failures.append(f"seed {seed} {policy}/{scale}: shared inputs differ")
                previous = initial
                checkpoint_quaternions = {0: initial}
                simulated_q = torch.nn.Parameter(initial.detach().clone())
                simulated_optimizer = torch.optim.Adam(
                    [simulated_q],
                    lr=1e-3,
                    betas=(0.9, 0.999),
                    eps=1e-15,
                    weight_decay=0,
                    amsgrad=False,
                    foreach=False,
                    fused=False,
                    capturable=False,
                    differentiable=False,
                )
                for step_index, (step, scheduled_view) in enumerate(
                    zip(arm["steps"], expected_schedule, strict=True), start=1
                ):
                    derived_step = derive_phase_a_step_metrics(
                        step,
                        expected_policy=policy,
                        expected_step=step_index,
                        expected_view=scheduled_view,
                        expected_q_old=previous,
                    )
                    raw_gradient = torch.tensor(step["gradient"], dtype=torch.float32)
                    optimizer_gradient = raw_gradient
                    if policy == "gradient_projection_current":
                        unit = F.normalize(simulated_q.detach(), dim=-1)
                        optimizer_gradient = raw_gradient - unit * (unit * raw_gradient).sum(
                            dim=-1, keepdim=True
                        )
                    simulated_optimizer.zero_grad(set_to_none=True)
                    simulated_q.grad = optimizer_gradient.detach().clone()
                    simulated_optimizer.step()
                    simulated_star = simulated_q.detach().clone()
                    if not torch.equal(
                        simulated_star, torch.tensor(step["q_star"], dtype=torch.float32)
                    ):
                        raise ValueError(
                            f"seed {seed} {policy}/{scale} step {step_index}: Adam equation differs"
                        )
                    expected_before = optimizer_state_record(simulated_optimizer, simulated_q)
                    if step["optimizer_before_policy"] != expected_before:
                        raise ValueError(
                            f"seed {seed} {policy}/{scale} step {step_index}: "
                            "pre-policy Adam state differs"
                        )
                    if policy == "unit_retraction":
                        with torch.no_grad():
                            simulated_q.copy_(F.normalize(simulated_star, dim=-1))
                    elif policy == "tangent_displacement_retraction":
                        simulated_old = torch.tensor(step["q_old"], dtype=torch.float32)
                        displacement = simulated_star - simulated_old
                        tangent = displacement - simulated_old * (simulated_old * displacement).sum(
                            dim=-1, keepdim=True
                        )
                        with torch.no_grad():
                            simulated_q.copy_(F.normalize(simulated_old + tangent, dim=-1))
                    expected_after = optimizer_state_record(simulated_optimizer, simulated_q)
                    if step["optimizer_after_policy"] != expected_after:
                        raise ValueError(
                            f"seed {seed} {policy}/{scale} step {step_index}: "
                            "post-policy Adam state differs"
                        )
                    if not torch.equal(simulated_q.detach(), derived_step["q_new"]):
                        raise ValueError(
                            f"seed {seed} {policy}/{scale} step {step_index}: trajectory differs"
                        )
                    if step["active_gradient_count"] < 32 or derived_step["active_count"] < 32:
                        failures.append(f"seed {seed} {policy}/{scale}: active rows fail")
                    previous = derived_step["q_new"]
                    if step_index in PHASE_A_CHECKPOINTS:
                        checkpoint_quaternions[step_index] = previous
                for checkpoint_step in PHASE_A_CHECKPOINTS:
                    checkpoint = arm["checkpoints"][str(checkpoint_step)]
                    if len(checkpoint.get("per_view", [])) != 9:
                        failures.append(
                            f"seed {seed} {policy}/{scale}: checkpoint view count differs"
                        )
                    derive_phase_a_checkpoint(
                        checkpoint,
                        expected_step=checkpoint_step,
                        template=target,
                        target=target,
                    )
                    stored_q = torch.tensor(checkpoint["raw_quaternions"], dtype=torch.float32)
                    if not torch.equal(stored_q, checkpoint_quaternions[checkpoint_step]):
                        failures.append(
                            f"seed {seed} {policy}/{scale}: checkpoint endpoint ordering differs"
                        )
                if policy == "current":
                    for checkpoint_step, checkpoint in arm["checkpoints"].items():
                        raw_q = torch.tensor(checkpoint["raw_quaternions"], dtype=torch.float32)
                        raw_gaussians = gaussian_with_quaternions(target, raw_q)
                        normalized = gaussian_with_quaternions(target, F.normalize(raw_q, dim=-1))
                        if not derive_physical_equivalence(
                            checkpoint["normalized_copy_audit"],
                            candidate=normalized,
                            reference=raw_gaussians,
                            context=(
                                f"seed {seed} current/{scale} checkpoint {checkpoint_step} "
                                "normalized-copy"
                            ),
                        ):
                            failures.append(
                                f"seed {seed} current/{scale}: normalized-copy audit fails"
                            )

        canonical_collapse[str(seed)] = {}
        for policy in CANONICALIZING_POLICIES:
            radial_arms = arms[policy]
            covariance_pairs = []
            for checkpoint in PHASE_A_CHECKPOINTS:
                checkpoint_key = str(checkpoint)
                denominator = radial_arms["1"]["checkpoints"][checkpoint_key]["covariance"]
                for left, right in (("0.25", "1"), ("0.25", "4"), ("1", "4")):
                    covariance_pairs.append(
                        {
                            "checkpoint": checkpoint,
                            "pair": [left, right],
                            **_covariance_pair_reduction(
                                radial_arms[left]["checkpoints"][checkpoint_key]["covariance"],
                                radial_arms[right]["checkpoints"][checkpoint_key]["covariance"],
                                denominator,
                            ),
                        }
                    )
            aucs = [derive_phase_a_auc(radial_arms[key]) for key in ("0.25", "1", "4")]
            maximum_covariance = max(item["relative"] for item in covariance_pairs)
            auc_spread = max(aucs) - min(aucs)
            passed = maximum_covariance <= 1e-6 and auc_spread <= 0.001
            canonical_collapse[str(seed)][policy] = {
                "covariance_pairs": covariance_pairs,
                "maximum_relative_covariance_difference": maximum_covariance,
                "auc_values_db": aucs,
                "auc_spread_db": auc_spread,
                "passed": passed,
            }
            if not passed:
                failures.append(f"seed {seed} {policy}: radial replicas do not collapse")

        seed_projection = []
        for scale in ("0.25", "1", "4"):
            for step in arms["gradient_projection_current"][scale]["steps"]:
                seed_projection.extend(derive_projection_removed_fractions(step))
        seed_p99 = linear_quantile(seed_projection, 0.99)
        projection_per_seed[str(seed)] = {
            "values": seed_projection,
            "p99": seed_p99,
            "passed": seed_p99 <= 1e-5,
        }
        pooled_projection.extend(seed_projection)
        if seed_p99 > 1e-5:
            failures.append(f"seed {seed}: projected-gradient p99 fails")
    pooled_p99 = linear_quantile(pooled_projection, 0.99)
    if pooled_p99 > 1e-5:
        failures.append("pooled projected-gradient p99 fails")
    return {
        "passed": not failures,
        "failures": failures,
        "prerequisites": prerequisite_recomputations,
        "canonical_replica_collapse": canonical_collapse,
        "gradient_projection": {
            "per_seed": projection_per_seed,
            "pooled_values": pooled_projection,
            "pooled_p99": pooled_p99,
            "passed": pooled_p99 <= 1e-5
            and all(item["passed"] for item in projection_per_seed.values()),
        },
    }


def _pooled_phase_a_decision(seed_records: list[dict[str, Any]]) -> dict[str, Any]:
    pooled_checkpoints: dict[str, dict[str, Any]] = {}
    aucs = {}
    for scale in ("0.25", "1", "4"):
        values = []
        pooled_checkpoints[scale] = {}
        for checkpoint in PHASE_A_CHECKPOINTS:
            sse = sum(
                seed["arms"]["current"][scale]["checkpoints"][str(checkpoint)]["pooled_color_sse"]
                for seed in seed_records
            )
            count = sum(
                seed["arms"]["current"][scale]["checkpoints"][str(checkpoint)]["pooled_color_count"]
                for seed in seed_records
            )
            psnr = psnr_from_sse(sse, count)
            pooled_checkpoints[scale][str(checkpoint)] = {
                "color_sse": sse,
                "color_count": count,
                "psnr": psnr,
            }
            values.append(psnr)
        aucs[scale] = normalized_auc(list(PHASE_A_CHECKPOINTS), values)
    auc_spread = max(aucs.values()) - min(aucs.values())

    covariance_pairs = {}
    for left, right in (("0.25", "1"), ("0.25", "4"), ("1", "4")):
        per_seed = [
            _covariance_pair_reduction(
                seed["arms"]["current"][left]["checkpoints"]["40"]["covariance"],
                seed["arms"]["current"][right]["checkpoints"]["40"]["covariance"],
                seed["arms"]["current"]["1"]["checkpoints"]["40"]["covariance"],
            )
            for seed in seed_records
        ]
        numerator = sum(item["numerator"] for item in per_seed)
        denominator = sum(item["denominator"] for item in per_seed)
        covariance_pairs[f"{left}_vs_{right}"] = {
            "per_seed": per_seed,
            "numerator": numerator,
            "denominator": denominator,
            "relative": numerator / denominator,
        }
    cov_spread = max(item["relative"] for item in covariance_pairs.values())
    derived_steps = [
        derive_phase_a_step_metrics(step)
        for seed in seed_records
        for step in seed["arms"]["current"]["1"]["steps"]
    ]
    effective_values = [
        float(value) for step in derived_steps for value in step["effective_lr_scale"]
    ]
    radial_values = [
        float(value) for step in derived_steps for value in step["radial_displacement_fraction"]
    ]
    effective_p90 = linear_quantile(effective_values, 0.9)
    radial_median = linear_quantile(radial_values, 0.5)
    tests = {
        "gauge_auc_spread": auc_spread >= 0.05,
        "gauge_cov_spread": cov_spread >= 0.001,
        "unit_effective_lr_p90": effective_p90 >= 0.01,
        "unit_radial_fraction_median": radial_median >= 0.10,
    }
    return {
        "pooled_checkpoints": pooled_checkpoints,
        "radial_auc_db": aucs,
        "gauge_auc_spread_db": auc_spread,
        "gauge_covariance_pairs": covariance_pairs,
        "gauge_cov_spread": cov_spread,
        "unit_effective_lr_values": effective_values,
        "unit_effective_lr_p90": effective_p90,
        "unit_radial_fraction_values": radial_values,
        "unit_radial_fraction_median": radial_median,
        "tests": tests,
        "ambient_gauge_material": all(tests.values()),
    }


def recompute_phase_a_decision(payload: dict[str, Any]) -> dict[str, Any]:
    seed_records = payload["seeds"]
    if [seed["seed"] for seed in seed_records] != list(SEEDS):
        raise ValueError("Phase-A seed order differs from preregistration")
    invariants = _phase_a_invariants(seed_records)
    per_seed = [_phase_a_seed_decision(seed) for seed in seed_records]
    pooled = _pooled_phase_a_decision(seed_records)
    material_seed_count = sum(item["ambient_gauge_material"] for item in per_seed)
    authorized = (
        invariants["passed"] and material_seed_count >= 2 and pooled["ambient_gauge_material"]
    )
    return {
        "invariants": invariants,
        "per_seed": per_seed,
        "material_seed_count": material_seed_count,
        "pooled": pooled,
        "phase_b_authorized": authorized,
        "outcome": (
            "ambient_quaternion_gauge_material"
            if authorized
            else "ambient_quaternion_gauge_not_material"
        ),
    }


def heldout_truth(evaluator: dict[str, Any]) -> list[dict[str, Any]]:
    gt_gaussians = evaluator["gt_gaussians"]
    outputs = render_views(gt_gaussians, evaluator["cameras"], sh_degree=0)
    truth = []
    for local_view, (original_view, output) in enumerate(
        zip(evaluator["original_indices"], outputs, strict=True)
    ):
        support = output.alpha > 0.05
        if int(support.sum()) == 0:
            raise ValueError(f"held-out view {original_view} has empty truth support")
        expected_depth = output.depth / output.alpha.clamp_min(1e-6)
        truth.append(
            {
                "local_view": local_view,
                "original_view": original_view,
                "color_tensor": output.color.detach().clone(),
                "alpha_tensor": output.alpha.detach().clone(),
                "depth_tensor": output.depth.detach().clone(),
                "expected_depth_tensor": expected_depth.detach().clone(),
                "support_tensor": support.detach().clone(),
                "color": tensor_values(output.color),
                "alpha": tensor_values(output.alpha),
                "accumulated_depth": tensor_values(output.depth),
                "expected_depth": tensor_values(expected_depth),
                "support": tensor_values(support),
                "support_count": int(support.sum()),
                "sha256": tensor_collection_hash(
                    [
                        ("color", output.color),
                        ("alpha", output.alpha),
                        ("accumulated_depth", output.depth),
                        ("expected_depth", expected_depth),
                        ("support", support),
                    ]
                ),
            }
        )
    return truth


def serializable_truth(truth: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tensor_keys = {
        "color_tensor",
        "alpha_tensor",
        "depth_tensor",
        "expected_depth_tensor",
        "support_tensor",
    }
    return [{key: value for key, value in view.items() if key not in tensor_keys} for view in truth]


def phase_b_checkpoint(
    gaussians: Gaussians3D,
    evaluator: dict[str, Any],
    truth: list[dict[str, Any]],
    *,
    step: int,
    wall_seconds: float,
) -> dict[str, Any]:
    validate_gaussians(gaussians, f"Phase-B checkpoint {step}")
    outputs = render_views(gaussians, evaluator["cameras"], sh_degree=3)
    per_view = []
    full_sse = 0.0
    full_count = 0
    foreground_sse = 0.0
    foreground_count = 0
    ssim_values = []
    depth_squared_error = 0.0
    depth_count = 0
    intersection = 0
    union = 0
    covered_truth = 0
    truth_count = 0
    for output, target in zip(outputs, truth, strict=True):
        support = target["support_tensor"]
        target_color = target["color_tensor"]
        error = output.color.to(torch.float64) - target_color.to(torch.float64)
        view_full_sse = float(error.square().sum())
        view_full_count = error.numel()
        foreground_error = error[support]
        view_foreground_sse = float(foreground_error.square().sum())
        view_foreground_count = foreground_error.numel()
        view_ssim = float(ssim(output.color, target_color))
        expected_depth = output.depth / output.alpha.clamp_min(1e-6)
        depth_error = (
            expected_depth.to(torch.float64) - target["expected_depth_tensor"].to(torch.float64)
        )[support]
        predicted_support = output.alpha > 0.05
        view_intersection = int((predicted_support & support).sum())
        view_union = int((predicted_support | support).sum())
        view_truth_count = int(support.sum())
        full_sse += view_full_sse
        full_count += view_full_count
        foreground_sse += view_foreground_sse
        foreground_count += view_foreground_count
        ssim_values.append(view_ssim)
        depth_squared_error += float(depth_error.square().sum())
        depth_count += depth_error.numel()
        intersection += view_intersection
        union += view_union
        covered_truth += view_intersection
        truth_count += view_truth_count
        per_view.append(
            {
                "local_view": target["local_view"],
                "original_view": target["original_view"],
                "full_color_sse": view_full_sse,
                "full_color_count": view_full_count,
                "full_color_psnr": psnr_from_sse(view_full_sse, view_full_count),
                "foreground_color_sse": view_foreground_sse,
                "foreground_color_count": view_foreground_count,
                "foreground_color_psnr": psnr_from_sse(view_foreground_sse, view_foreground_count),
                "ssim": view_ssim,
                "predicted_color": tensor_values(output.color),
                "predicted_alpha": tensor_values(output.alpha),
                "predicted_accumulated_depth": tensor_values(output.depth),
                "predicted_expected_depth": tensor_values(expected_depth),
                "truth_color": target["color"],
                "truth_alpha": target["alpha"],
                "truth_accumulated_depth": target["accumulated_depth"],
                "truth_expected_depth": target["expected_depth"],
                "truth_support": target["support"],
                "depth_squared_error": float(depth_error.square().sum()),
                "depth_count": depth_error.numel(),
                "alpha_intersection_count": view_intersection,
                "alpha_union_count": view_union,
                "covered_truth_count": view_intersection,
                "truth_support_count": view_truth_count,
                "predicted_sha256": tensor_collection_hash(
                    [
                        ("color", output.color),
                        ("alpha", output.alpha),
                        ("accumulated_depth", output.depth),
                        ("expected_depth", expected_depth),
                    ]
                ),
                "truth_sha256": target["sha256"],
            }
        )
    if foreground_count <= 0 or depth_count <= 0 or union <= 0 or truth_count <= 0:
        raise ValueError("Phase-B pooled held-out denominator is invalid")
    extent = evaluator["scene_extent"]
    normalized_depth_rmse = math.sqrt(depth_squared_error / depth_count) / extent
    covariance = covariance_float64(gaussians)
    return {
        "step": step,
        "wall_seconds": wall_seconds,
        "per_view": per_view,
        "pooled": {
            "full_color_sse": full_sse,
            "full_color_count": full_count,
            "full_color_psnr": psnr_from_sse(full_sse, full_count),
            "foreground_color_sse": foreground_sse,
            "foreground_color_count": foreground_count,
            "foreground_color_psnr": psnr_from_sse(foreground_sse, foreground_count),
            "ssim_sum": sum(ssim_values),
            "ssim_count": len(ssim_values),
            "ssim": statistics.fmean(ssim_values),
            "depth_squared_error": depth_squared_error,
            "depth_count": depth_count,
            "scene_extent": extent,
            "normalized_depth_rmse": normalized_depth_rmse,
            "alpha_intersection_count": intersection,
            "alpha_union_count": union,
            "alpha_iou": intersection / union,
            "covered_truth_count": covered_truth,
            "truth_support_count": truth_count,
            "truth_support_coverage": covered_truth / truth_count,
        },
        "raw_quaternion_norms": tensor_values(
            torch.linalg.vector_norm(gaussians.quats.to(torch.float64), dim=-1)
        ),
        "effective_quaternions": tensor_values(gaussians.quats),
        "fields": {
            field: tensor_values(getattr(gaussians, field))
            for field in ("means", "quats", "log_scales", "opacity", "sh")
        },
        "covariance": tensor_values(covariance),
        "field_hash": gaussians_hash(gaussians),
        "field_shapes": {
            field: list(getattr(gaussians, field).shape)
            for field in ("means", "quats", "log_scales", "opacity", "sh")
        },
        "non_quaternion_hash": non_quaternion_hash(gaussians),
        "covariance_hash": tensor_collection_hash([("covariance", covariance)]),
        "render_hash": render_hash(outputs),
    }


def run_phase_b_arm(
    prepared: dict[str, Any],
    evaluator: dict[str, Any],
    truth: list[dict[str, Any]],
    arm: str,
) -> dict[str, Any]:
    seed = prepared["seed"]
    config = train_config(seed, arm)
    common_input = prepared["initialization"].with_sh_degree(config.target_sh_degree)
    common_input_hash = gaussians_hash(common_input)
    checkpoints: dict[str, Any] = {}
    quaternion_steps = []
    initialization_captures = []
    arm_started = time.perf_counter()

    def capture_initialization(snapshot: Gaussians3D) -> None:
        initialization_captures.append(gaussians_hash(snapshot))
        expected_quaternions = (
            common_input.quats if arm == "current" else F.normalize(common_input.quats, dim=-1)
        )
        if not torch.equal(snapshot.quats, expected_quaternions):
            raise RuntimeError(f"{arm} effective step-zero quaternion is not the stored entry")
        if non_quaternion_hash(snapshot) != non_quaternion_hash(common_input):
            raise RuntimeError(f"{arm} entry changed a non-quaternion field")
        checkpoints["0"] = phase_b_checkpoint(
            snapshot,
            evaluator,
            truth,
            step=0,
            wall_seconds=time.perf_counter() - arm_started,
        )

    def capture_quaternion_step(
        q_old: torch.Tensor,
        q_star: torch.Tensor,
        q_new: torch.Tensor,
        step: int,
    ) -> None:
        for context, value in (("q_old", q_old), ("q_star", q_star), ("q_new", q_new)):
            require_valid_quaternions(value, f"Phase-B {arm} step {step} {context}")
        new_norm = torch.linalg.vector_norm(q_new.to(torch.float64), dim=-1)
        if arm != "current" and bool(((new_norm - 1.0).abs() > 2e-5).any()):
            raise RuntimeError(f"Phase-B {arm} step {step} violates unit norm")
        quaternion_steps.append(
            {
                "step": step,
                "q_old": tensor_values(q_old),
                "q_star": tensor_values(q_star),
                "q_new": tensor_values(q_new),
                "q_old_norm": tensor_values(
                    torch.linalg.vector_norm(q_old.to(torch.float64), dim=-1)
                ),
                "q_star_norm": tensor_values(
                    torch.linalg.vector_norm(q_star.to(torch.float64), dim=-1)
                ),
                "q_new_norm": tensor_values(new_norm),
                "q_old_sha256": tensor_collection_hash([("q_old", q_old)]),
                "q_star_sha256": tensor_collection_hash([("q_star", q_star)]),
                "q_new_sha256": tensor_collection_hash([("q_new", q_new)]),
            }
        )

    def capture_checkpoint(snapshot: Gaussians3D, step: int) -> None:
        if str(step) in checkpoints:
            raise RuntimeError(f"Phase-B {arm} duplicate checkpoint {step}")
        checkpoints[str(step)] = phase_b_checkpoint(
            snapshot,
            evaluator,
            truth,
            step=step,
            wall_seconds=time.perf_counter() - arm_started,
        )

    refined, history = Trainer(config).train(
        prepared["scene"],
        prepared["initialization"].detach(),
        initialization_callback=capture_initialization,
        quaternion_step_callback=capture_quaternion_step,
        checkpoint_callback=capture_checkpoint,
    )
    validate_gaussians(refined, f"Phase-B {arm} final", minimum_rows=prepared["initialization"].n)
    expected_schedule = phase_b_schedule(seed)
    if history["sampled_train_views"] != expected_schedule["local_positions"]:
        raise RuntimeError(f"Phase-B {arm} sampled-view history differs from probe")
    if len(initialization_captures) != 1:
        raise RuntimeError(f"Phase-B {arm} did not capture exactly one effective entry")
    if len(quaternion_steps) != 120:
        raise RuntimeError(f"Phase-B {arm} quaternion observer is incomplete")
    if set(map(int, checkpoints)) != set(PHASE_B_CHECKPOINTS):
        raise RuntimeError(f"Phase-B {arm} held-out checkpoints are incomplete")
    if refined.n != prepared["initialization"].n:
        raise RuntimeError(f"Phase-B {arm} topology changed")
    if any(not math.isfinite(float(value)) for value in history["loss"]):
        raise RuntimeError(f"Phase-B {arm} training loss is non-finite")
    foreground_psnr = [
        checkpoints[str(step)]["pooled"]["foreground_color_psnr"] for step in PHASE_B_CHECKPOINTS
    ]
    return {
        "arm": arm,
        "config": asdict(config),
        "common_input_hash": common_input_hash,
        "common_input_raw_initialization_hash": gaussians_hash(prepared["initialization"]),
        "effective_parameter_hash": checkpoints["0"]["field_hash"],
        "effective_quaternion_hash": tensor_collection_hash(
            [
                (
                    "effective_quaternions",
                    torch.tensor(checkpoints["0"]["effective_quaternions"], dtype=torch.float32),
                )
            ]
        ),
        "schedule_probe": expected_schedule,
        "sampled_train_views": history["sampled_train_views"],
        "sampled_original_views": [
            TRAIN_INDICES[index] for index in history["sampled_train_views"]
        ],
        "quaternion_steps": quaternion_steps,
        "checkpoints": checkpoints,
        "heldout_foreground_auc_db": normalized_auc(list(PHASE_B_CHECKPOINTS), foreground_psnr),
        "history": history,
        "final_field_hash": gaussians_hash(refined),
        "final_n": refined.n,
        "validity": {
            "finite": True,
            "complete_checkpoints": True,
            "exact_schedule": True,
            "fixed_topology": True,
            "candidate_norm_invariant": arm == "current"
            or all(
                max(abs(value - 1.0) for value in step["q_new_norm"]) <= 2e-5
                for step in quaternion_steps
            ),
        },
    }


def phase_b_step_zero_invariant(arms: dict[str, Any]) -> dict[str, Any]:
    reference = arms["current"]["checkpoints"]["0"]
    records = {}
    failures = []
    for arm in PHASE_B_ARMS:
        candidate = arms[arm]["checkpoints"]["0"]
        covariance_equal = torch.allclose(
            torch.tensor(candidate["covariance"], dtype=torch.float64),
            torch.tensor(reference["covariance"], dtype=torch.float64),
            atol=5e-6,
            rtol=2e-5,
        )
        render_equal = []
        for candidate_view, reference_view in zip(
            candidate["per_view"], reference["per_view"], strict=True
        ):
            fields = {}
            for field in (
                "predicted_color",
                "predicted_alpha",
                "predicted_accumulated_depth",
            ):
                fields[field] = torch.allclose(
                    torch.tensor(candidate_view[field], dtype=torch.float32),
                    torch.tensor(reference_view[field], dtype=torch.float32),
                    atol=5e-6,
                    rtol=2e-5,
                )
            render_equal.append(fields)
        non_quaternion_equal = candidate["non_quaternion_hash"] == reference["non_quaternion_hash"]
        shape_equal = (
            len(candidate["raw_quaternion_norms"]) == len(reference["raw_quaternion_norms"])
            and candidate["field_shapes"] == reference["field_shapes"]
        )
        passed = (
            covariance_equal
            and all(all(fields.values()) for fields in render_equal)
            and non_quaternion_equal
            and shape_equal
        )
        records[arm] = {
            "covariance_allclose": covariance_equal,
            "render_allclose": render_equal,
            "non_quaternion_bit_identical": non_quaternion_equal,
            "shape_and_count_equal": shape_equal,
            "passed": passed,
        }
        if not passed:
            failures.append(arm)
    return {"passed": not failures, "failures": failures, "arms": records}


def run_phase_b_seed(prepared: dict[str, Any]) -> dict[str, Any]:
    evaluator = prepared["evaluator"]
    if evaluator is None:
        raise RuntimeError("Phase-B evaluator capability was not retained")
    _, scene_extent = prepared["scene"].center_and_extent()
    evaluator["scene_extent"] = scene_extent
    truth = heldout_truth(evaluator)
    arms = {}
    for arm in TRAINING_ORDER[prepared["seed"]]:
        arms[arm] = run_phase_b_arm(prepared, evaluator, truth, arm)
    step_zero = phase_b_step_zero_invariant(arms)
    if not step_zero["passed"]:
        raise RuntimeError("Phase-B step-zero physical invariant failed")
    return {
        "seed": prepared["seed"],
        "training_order": list(TRAINING_ORDER[prepared["seed"]]),
        "training_hashes": prepared["training_hashes"],
        "scene_extent": scene_extent,
        "truth": serializable_truth(truth),
        "truth_aggregate_sha256": canonical_json_hash(serializable_truth(truth)),
        "step_zero_invariant": step_zero,
        "arms": arms,
    }


def validate_phase_b_truth(truth: Any) -> list[dict[str, Any]]:
    if not isinstance(truth, list) or len(truth) != 3:
        raise ValueError("Phase-B serialized truth is incomplete")
    expected_keys = {
        "local_view",
        "original_view",
        "color",
        "alpha",
        "accumulated_depth",
        "expected_depth",
        "support",
        "support_count",
        "sha256",
    }
    for index, view in enumerate(truth):
        if not isinstance(view, dict) or set(view) != expected_keys:
            raise ValueError(f"Phase-B truth view {index} schema differs")
        if view["local_view"] != index or view["original_view"] != HELD_OUT_INDICES[index]:
            raise ValueError(f"Phase-B truth view {index} identity differs")
        color = torch.tensor(view["color"], dtype=torch.float32)
        alpha = torch.tensor(view["alpha"], dtype=torch.float32)
        depth = torch.tensor(view["accumulated_depth"], dtype=torch.float32)
        expected_depth = torch.tensor(view["expected_depth"], dtype=torch.float32)
        support = torch.tensor(view["support"], dtype=torch.bool)
        height_width = color.shape[:2]
        if (
            color.ndim != 3
            or color.shape[-1] != 3
            or alpha.shape != height_width
            or depth.shape != height_width
            or expected_depth.shape != height_width
            or support.shape != height_width
        ):
            raise ValueError(f"Phase-B truth view {index} tensor shape differs")
        if not all(bool(value.isfinite().all()) for value in (color, alpha, depth, expected_depth)):
            raise ValueError(f"Phase-B truth view {index} is non-finite")
        require_tensor_equal(
            view["expected_depth"],
            depth / alpha.clamp_min(1e-6),
            f"Phase-B truth view {index} expected depth",
        )
        require_tensor_equal(view["support"], alpha > 0.05, f"Phase-B truth view {index} support")
        support_count = int(support.sum())
        if type(view["support_count"]) is not int or view["support_count"] != support_count:
            raise ValueError(f"Phase-B truth view {index} support count differs")
        if support_count <= 0:
            raise ValueError(f"Phase-B truth view {index} support is empty")
        expected_hash = tensor_collection_hash(
            [
                ("color", color),
                ("alpha", alpha),
                ("accumulated_depth", depth),
                ("expected_depth", expected_depth),
                ("support", support),
            ]
        )
        if view["sha256"] != expected_hash:
            raise ValueError(f"Phase-B truth view {index} hash differs")
    return truth


def derive_phase_b_checkpoint(
    record: dict[str, Any],
    *,
    expected_step: int,
    expected_truth: list[dict[str, Any]],
    expected_scene_extent: float,
) -> dict[str, Any]:
    expected_record_keys = {
        "step",
        "wall_seconds",
        "per_view",
        "pooled",
        "raw_quaternion_norms",
        "effective_quaternions",
        "fields",
        "covariance",
        "field_hash",
        "field_shapes",
        "non_quaternion_hash",
        "covariance_hash",
        "render_hash",
    }
    if set(record) != expected_record_keys:
        raise ValueError(f"Phase-B checkpoint {expected_step} schema differs")
    if record.get("step") != expected_step:
        raise ValueError(f"Phase-B checkpoint {expected_step} step identity differs")
    if (
        not math.isfinite(float(record.get("wall_seconds", float("nan"))))
        or float(record["wall_seconds"]) < 0.0
    ):
        raise ValueError(f"Phase-B checkpoint {expected_step} wall time is invalid")
    views = record.get("per_view")
    if not isinstance(views, list) or len(views) != 3:
        raise ValueError(f"Phase-B checkpoint {expected_step} per-view evidence is incomplete")
    if [view.get("local_view") for view in views] != [0, 1, 2] or [
        view.get("original_view") for view in views
    ] != list(HELD_OUT_INDICES):
        raise ValueError(f"Phase-B checkpoint {expected_step} held-out view order differs")
    if not math.isfinite(float(expected_scene_extent)) or expected_scene_extent <= 0.0:
        raise ValueError("Phase-B seed scene extent is invalid")

    pooled = {
        "full_color_sse": 0.0,
        "full_color_count": 0,
        "foreground_color_sse": 0.0,
        "foreground_color_count": 0,
        "ssim_sum": 0.0,
        "ssim_count": 0,
        "depth_squared_error": 0.0,
        "depth_count": 0,
        "alpha_intersection_count": 0,
        "alpha_union_count": 0,
        "covered_truth_count": 0,
        "truth_support_count": 0,
    }
    render_tensors: list[tuple[str, torch.Tensor]] = []
    expected_view_keys = {
        "local_view",
        "original_view",
        "full_color_sse",
        "full_color_count",
        "full_color_psnr",
        "foreground_color_sse",
        "foreground_color_count",
        "foreground_color_psnr",
        "ssim",
        "predicted_color",
        "predicted_alpha",
        "predicted_accumulated_depth",
        "predicted_expected_depth",
        "truth_color",
        "truth_alpha",
        "truth_accumulated_depth",
        "truth_expected_depth",
        "truth_support",
        "depth_squared_error",
        "depth_count",
        "alpha_intersection_count",
        "alpha_union_count",
        "covered_truth_count",
        "truth_support_count",
        "predicted_sha256",
        "truth_sha256",
    }
    for index, view in enumerate(views):
        if set(view) != expected_view_keys:
            raise ValueError(f"Phase-B checkpoint {expected_step} view {index} schema differs")
        predicted_color = torch.tensor(view["predicted_color"], dtype=torch.float32)
        predicted_alpha = torch.tensor(view["predicted_alpha"], dtype=torch.float32)
        predicted_depth = torch.tensor(view["predicted_accumulated_depth"], dtype=torch.float32)
        predicted_expected = torch.tensor(view["predicted_expected_depth"], dtype=torch.float32)
        truth_color = torch.tensor(view["truth_color"], dtype=torch.float32)
        truth_alpha = torch.tensor(view["truth_alpha"], dtype=torch.float32)
        truth_depth = torch.tensor(view["truth_accumulated_depth"], dtype=torch.float32)
        truth_expected = torch.tensor(view["truth_expected_depth"], dtype=torch.float32)
        truth_support = torch.tensor(view["truth_support"], dtype=torch.bool)
        height_width = predicted_color.shape[:2]
        if (
            predicted_color.ndim != 3
            or predicted_color.shape[-1] != 3
            or truth_color.shape != predicted_color.shape
            or predicted_alpha.shape != height_width
            or predicted_depth.shape != height_width
            or predicted_expected.shape != height_width
            or truth_alpha.shape != height_width
            or truth_depth.shape != height_width
            or truth_expected.shape != height_width
            or truth_support.shape != height_width
        ):
            raise ValueError(
                f"Phase-B checkpoint {expected_step} view {index} tensor shape differs"
            )
        tensors = (
            predicted_color,
            predicted_alpha,
            predicted_depth,
            predicted_expected,
            truth_color,
            truth_alpha,
            truth_depth,
            truth_expected,
        )
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError(f"Phase-B checkpoint {expected_step} view {index} is non-finite")
        require_tensor_equal(
            view["predicted_expected_depth"],
            predicted_depth / predicted_alpha.clamp_min(1e-6),
            f"Phase-B checkpoint {expected_step} view {index} predicted expected depth",
        )
        require_tensor_equal(
            view["truth_expected_depth"],
            truth_depth / truth_alpha.clamp_min(1e-6),
            f"Phase-B checkpoint {expected_step} view {index} truth expected depth",
        )
        require_tensor_equal(
            view["truth_support"],
            truth_alpha > 0.05,
            f"Phase-B checkpoint {expected_step} view {index} truth support",
        )
        truth_view = expected_truth[index]
        for field, tensor in (
            ("color", truth_color),
            ("alpha", truth_alpha),
            ("accumulated_depth", truth_depth),
            ("expected_depth", truth_expected),
            ("support", truth_support),
        ):
            require_tensor_equal(
                truth_view[field], tensor, f"Phase-B checkpoint truth {index}/{field}"
            )

        error = predicted_color.to(torch.float64) - truth_color.to(torch.float64)
        full_sse = float(error.square().sum())
        full_count = error.numel()
        foreground_error = error[truth_support]
        foreground_sse = float(foreground_error.square().sum())
        foreground_count = foreground_error.numel()
        view_ssim = float(ssim(predicted_color, truth_color))
        depth_error = (predicted_expected.to(torch.float64) - truth_expected.to(torch.float64))[
            truth_support
        ]
        depth_sse = float(depth_error.square().sum())
        depth_count = depth_error.numel()
        predicted_support = predicted_alpha > 0.05
        intersection = int((predicted_support & truth_support).sum())
        union = int((predicted_support | truth_support).sum())
        truth_count = int(truth_support.sum())
        numeric = {
            "full_color_sse": full_sse,
            "full_color_count": full_count,
            "full_color_psnr": psnr_from_sse(full_sse, full_count),
            "foreground_color_sse": foreground_sse,
            "foreground_color_count": foreground_count,
            "foreground_color_psnr": psnr_from_sse(foreground_sse, foreground_count),
            "ssim": view_ssim,
            "depth_squared_error": depth_sse,
            "depth_count": depth_count,
            "alpha_intersection_count": intersection,
            "alpha_union_count": union,
            "covered_truth_count": intersection,
            "truth_support_count": truth_count,
        }
        for field, value in numeric.items():
            if isinstance(value, int):
                if type(view.get(field)) is not int or view[field] != value:
                    raise ValueError(
                        f"Phase-B checkpoint {expected_step} view {index} {field} differs"
                    )
            else:
                require_close(
                    view[field],
                    value,
                    f"Phase-B checkpoint {expected_step} view {index} {field}",
                )
        predicted_hash = tensor_collection_hash(
            [
                ("color", predicted_color),
                ("alpha", predicted_alpha),
                ("accumulated_depth", predicted_depth),
                ("expected_depth", predicted_expected),
            ]
        )
        truth_hash = tensor_collection_hash(
            [
                ("color", truth_color),
                ("alpha", truth_alpha),
                ("accumulated_depth", truth_depth),
                ("expected_depth", truth_expected),
                ("support", truth_support),
            ]
        )
        if view.get("predicted_sha256") != predicted_hash or view.get("truth_sha256") != truth_hash:
            raise ValueError(f"Phase-B checkpoint {expected_step} view {index} hash differs")
        if expected_truth[index].get("sha256") != truth_hash:
            raise ValueError(f"Phase-B serialized truth {index} hash differs")
        render_tensors.extend(
            (
                (f"view_{index}/color", predicted_color),
                (f"view_{index}/alpha", predicted_alpha),
                (f"view_{index}/depth", predicted_depth),
            )
        )
        for key in pooled:
            if key == "ssim_sum":
                pooled[key] += view_ssim
            elif key == "ssim_count":
                pooled[key] += 1
            elif key in numeric:
                pooled[key] += numeric[key]

    stored_pooled = record["pooled"]
    expected_pooled_keys = {
        *pooled,
        "full_color_psnr",
        "foreground_color_psnr",
        "ssim",
        "scene_extent",
        "normalized_depth_rmse",
        "alpha_iou",
        "truth_support_coverage",
    }
    if set(stored_pooled) != expected_pooled_keys:
        raise ValueError(f"Phase-B checkpoint {expected_step} pooled schema differs")
    for key, value in pooled.items():
        if isinstance(value, int):
            if type(stored_pooled.get(key)) is not int or stored_pooled[key] != value:
                raise ValueError(f"Phase-B checkpoint {expected_step} pooled {key} differs")
        else:
            require_close(
                stored_pooled[key], value, f"Phase-B checkpoint {expected_step} pooled {key}"
            )
    if (
        pooled["full_color_count"] <= 0
        or pooled["foreground_color_count"] <= 0
        or pooled["depth_count"] <= 0
        or pooled["alpha_union_count"] <= 0
        or pooled["truth_support_count"] <= 0
        or pooled["ssim_count"] <= 0
    ):
        raise ValueError(f"Phase-B checkpoint {expected_step} pooled denominator is invalid")
    require_close(
        stored_pooled["scene_extent"],
        expected_scene_extent,
        f"Phase-B checkpoint {expected_step} scene extent",
        atol=0.0,
    )
    derived_pooled = {
        "full_color_psnr": psnr_from_sse(pooled["full_color_sse"], pooled["full_color_count"]),
        "foreground_color_psnr": psnr_from_sse(
            pooled["foreground_color_sse"], pooled["foreground_color_count"]
        ),
        "ssim": pooled["ssim_sum"] / pooled["ssim_count"],
        "normalized_depth_rmse": math.sqrt(pooled["depth_squared_error"] / pooled["depth_count"])
        / expected_scene_extent,
        "alpha_iou": pooled["alpha_intersection_count"] / pooled["alpha_union_count"],
        "truth_support_coverage": pooled["covered_truth_count"] / pooled["truth_support_count"],
    }
    for key, value in derived_pooled.items():
        require_close(stored_pooled[key], value, f"Phase-B checkpoint {expected_step} pooled {key}")

    effective_quaternions = torch.tensor(record["effective_quaternions"], dtype=torch.float32)
    fields = gaussians_from_serialized_fields(
        record["fields"], f"Phase-B checkpoint {expected_step} fields", minimum_rows=256
    )
    if not torch.equal(effective_quaternions, fields.quats):
        raise ValueError(f"Phase-B checkpoint {expected_step} effective quaternions differ")
    n = fields.n
    norms = torch.linalg.vector_norm(effective_quaternions.to(torch.float64), dim=-1)
    require_tensor_equal(
        record["raw_quaternion_norms"], norms, f"Phase-B checkpoint {expected_step} raw norms"
    )
    covariance = covariance_float64(fields)
    require_tensor_equal(
        record["covariance"], covariance, f"Phase-B checkpoint {expected_step} covariance"
    )
    if record.get("covariance_hash") != tensor_collection_hash([("covariance", covariance)]):
        raise ValueError(f"Phase-B checkpoint {expected_step} covariance hash differs")
    if record.get("render_hash") != tensor_collection_hash(render_tensors):
        raise ValueError(f"Phase-B checkpoint {expected_step} render hash differs")
    if record["field_hash"] != gaussians_hash(fields):
        raise ValueError(f"Phase-B checkpoint {expected_step} field hash differs")
    if record["non_quaternion_hash"] != non_quaternion_hash(fields):
        raise ValueError(f"Phase-B checkpoint {expected_step} non-quaternion hash differs")
    shapes = {
        field: list(getattr(fields, field).shape)
        for field in ("means", "quats", "log_scales", "opacity", "sh")
    }
    if record["field_shapes"] != shapes or shapes["sh"] != [n, 16, 3]:
        raise ValueError(f"Phase-B checkpoint {expected_step} field shapes differ")
    return {
        "foreground_psnr": derived_pooled["foreground_color_psnr"],
        "full_psnr": derived_pooled["full_color_psnr"],
        "ssim": derived_pooled["ssim"],
        "normalized_depth_rmse": derived_pooled["normalized_depth_rmse"],
        "alpha_iou": derived_pooled["alpha_iou"],
        "coverage": derived_pooled["truth_support_coverage"],
        "n": n,
        "quaternions": effective_quaternions,
        "non_quaternion_hash": record["non_quaternion_hash"],
        "field_hash": record["field_hash"],
        "wall_seconds": float(record["wall_seconds"]),
    }


def _phase_b_arm_metrics(
    arm: dict[str, Any],
    *,
    expected_arm: str,
    seed: int,
    expected_truth: list[dict[str, Any]],
    expected_scene_extent: float,
) -> dict[str, Any]:
    if arm.get("arm") != expected_arm:
        raise ValueError(f"Phase-B {expected_arm} arm identity differs")
    if arm.get("config") != asdict(train_config(seed, expected_arm)):
        raise ValueError(f"Phase-B {expected_arm} configuration differs")
    checkpoints = arm["checkpoints"]
    expected_checkpoint_keys = {str(step) for step in PHASE_B_CHECKPOINTS}
    if set(checkpoints) != expected_checkpoint_keys:
        raise ValueError("Phase-B checkpoint set differs")
    foreground_psnr = []
    checkpoint_metrics = {}
    checkpoint_raw = {}
    for step in PHASE_B_CHECKPOINTS:
        derived = derive_phase_b_checkpoint(
            checkpoints[str(step)],
            expected_step=step,
            expected_truth=expected_truth,
            expected_scene_extent=expected_scene_extent,
        )
        checkpoint_raw[str(step)] = derived
        foreground_psnr.append(derived["foreground_psnr"])
        checkpoint_metrics[str(step)] = {
            "foreground_psnr": derived["foreground_psnr"],
            "full_psnr": derived["full_psnr"],
            "ssim": derived["ssim"],
        }
    wall_times = [checkpoint_raw[str(step)]["wall_seconds"] for step in PHASE_B_CHECKPOINTS]
    if any(right < left for left, right in zip(wall_times, wall_times[1:])):
        raise ValueError(f"Phase-B {expected_arm} checkpoint wall times are not monotonic")
    expected_schedule = phase_b_schedule(seed)
    if arm.get("schedule_probe") != expected_schedule:
        raise ValueError(f"Phase-B {expected_arm} schedule probe differs")
    if arm.get("sampled_train_views") != expected_schedule["local_positions"]:
        raise ValueError(f"Phase-B {expected_arm} sampled schedule differs")
    if arm.get("sampled_original_views") != expected_schedule["original_view_indices"]:
        raise ValueError(f"Phase-B {expected_arm} original-view schedule differs")
    schedule_matches = arm["sampled_train_views"] == arm["schedule_probe"]["local_positions"]
    quaternion_steps = arm["quaternion_steps"]
    if not isinstance(quaternion_steps, list) or [
        step.get("step") for step in quaternion_steps
    ] != list(range(1, 121)):
        raise ValueError(f"Phase-B {expected_arm} quaternion step order differs")
    n = checkpoint_raw["0"]["n"]
    norm_valid = True
    previous = checkpoint_raw["0"]["quaternions"]
    expected_step_keys = {
        "step",
        "q_old",
        "q_star",
        "q_new",
        "q_old_norm",
        "q_star_norm",
        "q_new_norm",
        "q_old_sha256",
        "q_star_sha256",
        "q_new_sha256",
    }
    for step_index, step in enumerate(quaternion_steps, start=1):
        if set(step) != expected_step_keys:
            raise ValueError(f"Phase-B {expected_arm} step {step_index} schema differs")
        raw = {
            field: torch.tensor(step[field], dtype=torch.float32)
            for field in ("q_old", "q_star", "q_new")
        }
        if any(
            value.shape != (n, 4) or not bool(torch.isfinite(value).all()) for value in raw.values()
        ):
            raise ValueError(f"Phase-B {expected_arm} step {step_index} raw endpoint differs")
        if not torch.equal(raw["q_old"], previous):
            raise ValueError(f"Phase-B {expected_arm} step {step_index} trajectory differs")
        for field in ("q_old", "q_star", "q_new"):
            norms = torch.linalg.vector_norm(raw[field].to(torch.float64), dim=-1)
            require_tensor_equal(
                step[f"{field}_norm"],
                norms,
                f"Phase-B {expected_arm} step {step_index} {field} norms",
            )
            if step[f"{field}_sha256"] != tensor_collection_hash([(field, raw[field])]):
                raise ValueError(f"Phase-B {expected_arm} step {step_index} {field} hash differs")
        if expected_arm == "current":
            expected_new = raw["q_star"]
        elif expected_arm == "unit_retraction":
            expected_new = F.normalize(raw["q_star"], dim=-1)
        else:
            if bool(((torch.linalg.vector_norm(raw["q_old"], dim=-1) - 1.0).abs() > 2e-5).any()):
                raise ValueError(
                    f"Phase-B {expected_arm} step {step_index} q_old is not unit length"
                )
            displacement = raw["q_star"] - raw["q_old"]
            tangent = displacement - raw["q_old"] * (raw["q_old"] * displacement).sum(
                dim=-1, keepdim=True
            )
            expected_new = F.normalize(raw["q_old"] + tangent, dim=-1)
        if not torch.equal(raw["q_new"], expected_new):
            raise ValueError(f"Phase-B {expected_arm} step {step_index} policy equation differs")
        if expected_arm != "current" and bool(
            ((torch.linalg.vector_norm(raw["q_new"], dim=-1) - 1.0).abs() > 2e-5).any()
        ):
            norm_valid = False
        if step_index in PHASE_B_CHECKPOINTS and not torch.equal(
            raw["q_new"], checkpoint_raw[str(step_index)]["quaternions"]
        ):
            raise ValueError(f"Phase-B {expected_arm} checkpoint {step_index} endpoint differs")
        previous = raw["q_new"]
    if arm.get("effective_parameter_hash") != checkpoints["0"]["field_hash"]:
        raise ValueError(f"Phase-B {expected_arm} effective parameter hash differs")
    effective_quaternions = torch.tensor(
        checkpoints["0"]["effective_quaternions"], dtype=torch.float32
    )
    if arm.get("effective_quaternion_hash") != tensor_collection_hash(
        [("effective_quaternions", effective_quaternions)]
    ):
        raise ValueError(f"Phase-B {expected_arm} effective quaternion hash differs")
    for field in ("common_input_hash", "common_input_raw_initialization_hash"):
        require_sha256(arm.get(field), f"Phase-B {expected_arm} {field}")
    if arm.get("final_n") != n or any(item["n"] != n for item in checkpoint_raw.values()):
        raise ValueError(f"Phase-B {expected_arm} fixed topology differs")
    if arm.get("final_field_hash") != checkpoints["120"]["field_hash"]:
        raise ValueError(f"Phase-B {expected_arm} final field hash differs")
    history = arm.get("history")
    expected_history_keys = {
        "loss",
        "loss_terms",
        "psnr",
        "elapsed",
        "n_gaussians",
        "active_sh_degree",
        "sampled_train_views",
        "sh_color_diagnostics",
        "kernel_support_diagnostics",
        "density_stats",
        "density_strategy",
        "resolved_sh_degree_interval",
        "peak_vram_gb",
    }
    if not isinstance(history, dict) or set(history) != expected_history_keys:
        raise ValueError(f"Phase-B {expected_arm} history schema differs")
    if history["sampled_train_views"] != arm["sampled_train_views"]:
        raise ValueError(f"Phase-B {expected_arm} history schedule differs")
    if len(history.get("loss", [])) != 120 or any(
        not math.isfinite(float(value)) for value in history["loss"]
    ):
        raise ValueError(f"Phase-B {expected_arm} history loss evidence differs")
    loss_terms = history["loss_terms"]
    if len(loss_terms) != 120 or any(
        set(item) != {"l1", "alpha", "opacity_reg", "scale_reg"}
        or any(not math.isfinite(float(value)) for value in item.values())
        for item in loss_terms
    ):
        raise ValueError(f"Phase-B {expected_arm} history loss-term evidence differs")
    eval_steps = [30, 60, 90, 120]
    for field in ("psnr", "elapsed", "n_gaussians", "active_sh_degree"):
        values = history[field]
        if len(values) != 4 or [int(item[0]) for item in values] != eval_steps:
            raise ValueError(f"Phase-B {expected_arm} history {field} schedule differs")
    if any(not math.isfinite(float(item[1])) for item in history["psnr"]):
        raise ValueError(f"Phase-B {expected_arm} history PSNR differs")
    elapsed = [float(item[1]) for item in history["elapsed"]]
    if any(value < 0.0 or not math.isfinite(value) for value in elapsed) or any(
        right < left for left, right in zip(elapsed, elapsed[1:])
    ):
        raise ValueError(f"Phase-B {expected_arm} history elapsed time differs")
    if [int(item[1]) for item in history["n_gaussians"]] != [n] * 4:
        raise ValueError(f"Phase-B {expected_arm} history topology differs")
    if [int(item[1]) for item in history["active_sh_degree"]] != [0, 1, 2, 3]:
        raise ValueError(f"Phase-B {expected_arm} history SH schedule differs")
    if (
        history["sh_color_diagnostics"]
        or history["kernel_support_diagnostics"]
        or history["density_stats"] is not None
        or history["density_strategy"] != "none"
        or history["resolved_sh_degree_interval"] != 30
        or float(history["peak_vram_gb"]) != 0.0
    ):
        raise ValueError(f"Phase-B {expected_arm} history frozen controls differ")
    derived_validity = {
        "finite": True,
        "complete_checkpoints": set(checkpoints) == expected_checkpoint_keys,
        "exact_schedule": schedule_matches,
        "fixed_topology": arm["final_n"] == n
        and all(item["n"] == n for item in checkpoint_raw.values()),
        "candidate_norm_invariant": expected_arm == "current" or norm_valid,
    }
    if arm.get("validity") != derived_validity:
        raise ValueError(f"Phase-B {expected_arm} stored validity summary differs")
    auc = normalized_auc(list(PHASE_B_CHECKPOINTS), foreground_psnr)
    require_close(arm.get("heldout_foreground_auc_db"), auc, "stored Phase-B AUC")
    final = checkpoint_raw["120"]
    return {
        "checkpoints": checkpoint_metrics,
        "heldout_foreground_auc_db": auc,
        "final_foreground_psnr": checkpoint_metrics["120"]["foreground_psnr"],
        "final_ssim": checkpoint_metrics["120"]["ssim"],
        "final_normalized_depth_rmse": final["normalized_depth_rmse"],
        "final_alpha_iou": final["alpha_iou"],
        "final_truth_support_coverage": final["coverage"],
        "validity_passed": all(derived_validity.values()),
    }


def recompute_phase_b_decision(payload: dict[str, Any]) -> dict[str, Any]:
    seed_records = payload["seeds"]
    if [seed["seed"] for seed in seed_records] != list(SEEDS):
        raise ValueError("Phase-B seed order differs")
    metrics: dict[str, dict[str, Any]] = {}
    failures = []
    for seed_record in seed_records:
        seed = seed_record["seed"]
        if seed_record["training_order"] != list(TRAINING_ORDER[seed]):
            raise ValueError(f"seed {seed}: training order differs")
        truth = validate_phase_b_truth(seed_record["truth"])
        if seed_record.get("truth_aggregate_sha256") != canonical_json_hash(truth):
            raise ValueError(f"seed {seed}: truth aggregate hash differs")
        scene_extent = seed_record.get("scene_extent")
        if (
            not isinstance(scene_extent, (int, float))
            or not math.isfinite(float(scene_extent))
            or float(scene_extent) <= 0.0
        ):
            raise ValueError(f"seed {seed}: scene extent is invalid")
        training_hashes = seed_record.get("training_hashes")
        expected_training_keys = {
            "scene",
            "fit_fields",
            "fit_histories",
            "fit_order",
            "initialization_fields",
            "initialization",
            "initialization_covariance",
            "aggregate",
        }
        if not isinstance(training_hashes, dict) or set(training_hashes) != expected_training_keys:
            raise ValueError(f"seed {seed}: training hash schema differs")
        training_without_aggregate = {
            key: value for key, value in training_hashes.items() if key != "aggregate"
        }
        if training_hashes["aggregate"] != canonical_json_hash(training_without_aggregate):
            raise ValueError(f"seed {seed}: training aggregate hash differs")
        for field in (
            "fit_fields",
            "fit_histories",
            "fit_order",
            "initialization",
            "initialization_covariance",
        ):
            require_sha256(training_hashes[field], f"seed {seed} training {field}")
        if not isinstance(training_hashes["scene"], dict) or not isinstance(
            training_hashes["initialization_fields"], dict
        ):
            raise ValueError(f"seed {seed}: nested training hashes differ")
        for group in ("scene", "initialization_fields"):
            for name, digest in training_hashes[group].items():
                require_sha256(digest, f"seed {seed} training {group}/{name}")
        if set(seed_record["arms"]) != set(PHASE_B_ARMS):
            raise ValueError(f"seed {seed}: Phase-B arm set differs")
        metrics[str(seed)] = {}
        for arm in PHASE_B_ARMS:
            arm_metrics = _phase_b_arm_metrics(
                seed_record["arms"][arm],
                expected_arm=arm,
                seed=seed,
                expected_truth=truth,
                expected_scene_extent=float(scene_extent),
            )
            metrics[str(seed)][arm] = arm_metrics
            if not arm_metrics["validity_passed"]:
                failures.append(f"seed {seed} {arm}: validity fails")
        arms = seed_record["arms"]
        if len({arms[arm]["common_input_hash"] for arm in PHASE_B_ARMS}) != 1:
            raise ValueError(f"seed {seed}: common effective input hash differs across arms")
        if len({arms[arm]["common_input_raw_initialization_hash"] for arm in PHASE_B_ARMS}) != 1:
            raise ValueError(f"seed {seed}: raw initialization hash differs across arms")
        if any(
            arms[arm]["common_input_raw_initialization_hash"] != training_hashes["initialization"]
            for arm in PHASE_B_ARMS
        ):
            raise ValueError(f"seed {seed}: arm raw initialization hash differs from seed input")
        current_zero = arms["current"]["checkpoints"]["0"]
        if arms["current"]["common_input_hash"] != current_zero["field_hash"]:
            raise ValueError(f"seed {seed}: current effective entry is not the common input")
        current_q = torch.tensor(current_zero["effective_quaternions"], dtype=torch.float32)
        for candidate in PHASE_B_ARMS[1:]:
            candidate_zero = arms[candidate]["checkpoints"]["0"]
            candidate_q = torch.tensor(candidate_zero["effective_quaternions"], dtype=torch.float32)
            if not torch.equal(candidate_q, F.normalize(current_q, dim=-1)):
                raise ValueError(f"seed {seed} {candidate}: entry normalization differs")
            if candidate_zero["non_quaternion_hash"] != current_zero["non_quaternion_hash"]:
                raise ValueError(f"seed {seed} {candidate}: entry non-quaternion fields differ")
        derived_step_zero = phase_b_step_zero_invariant(arms)
        if seed_record.get("step_zero_invariant") != derived_step_zero:
            raise ValueError(f"seed {seed}: stored step-zero invariant differs")
        if not derived_step_zero["passed"]:
            failures.append(f"seed {seed}: step-zero invariant fails")

    candidate_decisions = {}
    for candidate in PHASE_B_ARMS[1:]:
        auc_delta = [
            metrics[str(seed)][candidate]["heldout_foreground_auc_db"]
            - metrics[str(seed)]["current"]["heldout_foreground_auc_db"]
            for seed in SEEDS
        ]
        utility_tests = {
            "mean_delta_at_least_0.05": statistics.fmean(auc_delta) >= 0.05,
            "positive_in_at_least_two_seeds": sum(value > 0.0 for value in auc_delta) >= 2,
            "no_seed_below_minus_0.15": min(auc_delta) >= -0.15,
        }
        final_psnr_delta = [
            metrics[str(seed)][candidate]["final_foreground_psnr"]
            - metrics[str(seed)]["current"]["final_foreground_psnr"]
            for seed in SEEDS
        ]
        final_ssim_delta = [
            metrics[str(seed)][candidate]["final_ssim"]
            - metrics[str(seed)]["current"]["final_ssim"]
            for seed in SEEDS
        ]
        depth_regression = []
        alpha_regression = []
        coverage_regression = []
        for seed in SEEDS:
            current = metrics[str(seed)]["current"]
            candidate_metrics = metrics[str(seed)][candidate]
            denominator = current["final_normalized_depth_rmse"]
            if denominator == 0.0:
                raise ValueError("current Phase-B depth RMSE is zero")
            depth_regression.append(
                (candidate_metrics["final_normalized_depth_rmse"] - denominator) / denominator
            )
            alpha_regression.append(
                current["final_alpha_iou"] - candidate_metrics["final_alpha_iou"]
            )
            coverage_regression.append(
                current["final_truth_support_coverage"]
                - candidate_metrics["final_truth_support_coverage"]
            )
        safety_tests = {
            "mean_final_foreground_psnr_delta": statistics.fmean(final_psnr_delta) >= -0.05,
            "no_final_foreground_psnr_delta_below_minus_0.15": min(final_psnr_delta) >= -0.15,
            "mean_final_ssim_delta": statistics.fmean(final_ssim_delta) >= -0.002,
            "no_final_ssim_delta_below_minus_0.005": min(final_ssim_delta) >= -0.005,
            "mean_depth_rmse_regression": statistics.fmean(depth_regression) <= 0.02,
            "no_depth_rmse_regression_above_0.05": max(depth_regression) <= 0.05,
            "mean_alpha_iou_regression": statistics.fmean(alpha_regression) <= 0.01,
            "no_alpha_iou_regression_above_0.03": max(alpha_regression) <= 0.03,
            "mean_coverage_regression": statistics.fmean(coverage_regression) <= 0.01,
            "no_coverage_regression_above_0.03": max(coverage_regression) <= 0.03,
            "all_validity_invariants": not failures,
        }
        utility_passed = all(utility_tests.values())
        safety_passed = all(safety_tests.values())
        candidate_decisions[candidate] = {
            "auc_delta_by_seed": auc_delta,
            "mean_auc_delta_db": statistics.fmean(auc_delta),
            "utility_tests": utility_tests,
            "utility_passed": utility_passed,
            "final_foreground_psnr_delta_by_seed": final_psnr_delta,
            "final_ssim_delta_by_seed": final_ssim_delta,
            "relative_normalized_depth_rmse_regression_by_seed": depth_regression,
            "alpha_iou_regression_by_seed": alpha_regression,
            "coverage_regression_by_seed": coverage_regression,
            "safety_tests": safety_tests,
            "safety_passed": safety_passed,
            "passed": utility_passed and safety_passed,
        }

    passing = [
        candidate for candidate in PHASE_B_ARMS[1:] if candidate_decisions[candidate]["passed"]
    ]
    preference = None
    selected = "current"
    if passing == ["unit_retraction"]:
        selected = "unit_retraction"
    elif passing == ["tangent_displacement_retraction"]:
        selected = "tangent_displacement_retraction"
    elif len(passing) == 2:
        tangent_advantage = [
            metrics[str(seed)]["tangent_displacement_retraction"]["heldout_foreground_auc_db"]
            - metrics[str(seed)]["unit_retraction"]["heldout_foreground_auc_db"]
            for seed in SEEDS
        ]
        tangent_preferred = (
            statistics.fmean(tangent_advantage) >= 0.03
            and sum(value > 0.0 for value in tangent_advantage) >= 2
        )
        preference = {
            "tangent_minus_unit_auc_by_seed": tangent_advantage,
            "mean_tangent_minus_unit_auc_db": statistics.fmean(tangent_advantage),
            "strict_tangent_wins": sum(value > 0.0 for value in tangent_advantage),
            "tangent_preferred": tangent_preferred,
        }
        selected = "tangent_displacement_retraction" if tangent_preferred else "unit_retraction"
    return {
        "validity": {"passed": not failures, "failures": failures},
        "metrics": metrics,
        "candidates": candidate_decisions,
        "passing_candidates": passing,
        "preference": preference,
        "confirmatory_candidate": selected if selected != "current" else None,
        "outcome": (
            "retain_current" if selected == "current" else f"confirmatory_candidate_{selected}"
        ),
        "production_default_change_authorized": False,
    }


def relative_to_root(path: Path) -> Path:
    resolved = path if path.is_absolute() else ROOT / path
    return resolved.resolve()


def require_default_seal_path(path: Path) -> Path:
    resolved = relative_to_root(path)
    expected = relative_to_root(DEFAULT_SEAL)
    if resolved != expected:
        raise ValueError(f"sole implementation seal path is {DEFAULT_SEAL}")
    return resolved


def json_text(payload: dict[str, Any]) -> str:
    assert_finite_tree(payload, "artifact")
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_artifact(output: Path, payload: dict[str, Any], note: str) -> tuple[Path, Path]:
    output = relative_to_root(output)
    note_path = output.with_name(f"{output.stem}_RESULT.md")
    exclusive_atomic_write(output, json_text(payload))
    exclusive_atomic_write(note_path, note.rstrip() + "\n")
    return output, note_path


def preflight_paths(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if relative_to_root(path).exists()]
    if existing:
        raise FileExistsError(f"refusing an append-only namespace with existing paths: {existing}")


def phase_a_output_paths(prefix: Path) -> dict[str, Path]:
    expected_parent = (ROOT / "benchmarks/results").resolve()
    if relative_to_root(prefix).parent != expected_parent:
        raise ValueError("Phase-A output-prefix must be directly under benchmarks/results")
    if not UTC_PREFIX_RE.fullmatch(prefix.name):
        raise ValueError(
            "Phase-A output-prefix basename must match YYYYMMDDTHHMMSSZ_cpu_quaternion_gauge_iter2"
        )
    return {
        "audit_json": Path(f"{prefix}_audit.json"),
        "audit_note": Path(f"{prefix}_audit_RESULT.md"),
        "invalid_json": Path(f"{prefix}_invalid.json"),
        "invalid_note": Path(f"{prefix}_invalid_RESULT.md"),
    }


def phase_b_output_paths(output: Path) -> dict[str, Path]:
    expected_parent = (ROOT / "benchmarks/results").resolve()
    if relative_to_root(output).parent != expected_parent:
        raise ValueError("Phase-B output must be directly under benchmarks/results")
    pattern = re.compile(r"^\d{8}T\d{6}Z_cpu_quaternion_gauge_iter2_ablation\.json$")
    if not pattern.fullmatch(output.name):
        raise ValueError(
            "Phase-B output basename must match "
            "YYYYMMDDTHHMMSSZ_cpu_quaternion_gauge_iter2_ablation.json"
        )
    return {
        "json": output,
        "note": output.with_name(f"{output.stem}_RESULT.md"),
    }


def attempt_marker(
    *,
    artifact_type: str,
    marker_path: Path,
    seal: dict[str, Any],
    outputs: dict[str, Path],
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loaded_hashes, loaded_aggregate = verify_loaded_sources_against_seal(
        relative_to_root(Path(seal["path"]))
    )
    payload = {
        "artifact_type": artifact_type,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
        },
        "seal": seal,
        "outputs": {key: str(value) for key, value in outputs.items()},
        "inputs": inputs or {},
        "sealed_source_aggregate": seal["source_aggregate"],
        "loaded_source_hashes": loaded_hashes,
        "loaded_source_aggregate": loaded_aggregate,
        "environment": environment_metadata(),
        "command": [sys.executable, *sys.argv],
    }
    exclusive_atomic_write(relative_to_root(marker_path), json_text(payload))
    return {
        "path": str(marker_path.relative_to(ROOT) if marker_path.is_absolute() else marker_path),
        "sha256": sha256_file(relative_to_root(marker_path)),
        "payload": payload,
    }


def preparation_evidence(prepared: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": prepared["seed"],
        "training_hashes": prepared["training_hashes"],
        "fit_config": asdict(fit_config()),
        "depth_lifter_config": depth_lifter_config(),
        "initialization_n": prepared["initialization"].n,
        "initialization_fields": {
            field: tensor_values(getattr(prepared["initialization"], field))
            for field in ("means", "quats", "log_scales", "opacity", "sh")
        },
        "training_original_indices": list(TRAIN_INDICES),
        "selection": prepared["selection"],
        "perturbation": prepared["perturbation"],
        "target_hashes": prepared["target_hashes"],
        "target_fields": {
            field: tensor_values(getattr(prepared["target"], field))
            for field in ("means", "quats", "log_scales", "opacity", "sh")
        },
        "target_covariance": tensor_values(covariance_float64(prepared["target"])),
        "schedule": prepared["schedule"],
        "schedule_sha256": prepared["schedule_sha256"],
    }


def artifact_bindings(seal: dict[str, Any], marker: dict[str, Any]) -> dict[str, Any]:
    return {
        "original_preregistration_path": str(ORIGINAL_PREREGISTRATION),
        "original_preregistration_sha256": ORIGINAL_PREREGISTRATION_SHA256,
        "preregistration_path": str(PREREGISTRATION),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "implementation_review_path": seal["implementation_review"]["path"],
        "implementation_review_sha256": seal["implementation_review"]["sha256"],
        "seal_path": seal["path"],
        "seal_sha256": seal["sha256"],
        "seal_file_sha256": seal["file_sha256"],
        "source_aggregate": seal["source_aggregate"],
        "retry_provenance_sha256": canonical_json_hash(seal["retry_provenance"]),
        "attempt_marker_path": marker["path"],
        "attempt_marker_sha256": marker["sha256"],
    }


def validate_artifact_bindings(bindings: Any, seal: dict[str, Any]) -> None:
    if not isinstance(bindings, dict) or set(bindings) != ARTIFACT_BINDING_KEYS:
        raise RuntimeError("Phase-A artifact binding key set differs")
    expected = {
        "original_preregistration_path": str(ORIGINAL_PREREGISTRATION),
        "original_preregistration_sha256": ORIGINAL_PREREGISTRATION_SHA256,
        "preregistration_path": str(PREREGISTRATION),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "implementation_review_path": seal["implementation_review"]["path"],
        "implementation_review_sha256": seal["implementation_review"]["sha256"],
        "seal_path": seal["path"],
        "seal_sha256": seal["sha256"],
        "seal_file_sha256": seal["file_sha256"],
        "source_aggregate": seal["source_aggregate"],
        "retry_provenance_sha256": canonical_json_hash(seal["retry_provenance"]),
    }
    mismatched = [key for key, value in expected.items() if bindings.get(key) != value]
    if mismatched:
        raise RuntimeError(f"Phase-A artifact binding differs: {mismatched}")
    require_sha256(bindings["attempt_marker_sha256"], "Phase-A attempt-marker binding")


def verify_runtime_bindings(seal_path: Path, marker: dict[str, Any]) -> dict[str, Any]:
    seal = load_and_verify_seal(relative_to_root(seal_path))
    marker_path = relative_to_root(Path(marker["path"]))
    if sha256_file(marker_path) != marker["sha256"]:
        raise RuntimeError("attempt marker changed during execution")
    verify_loaded_sources_against_seal(relative_to_root(seal_path))
    return seal


def phase_a_note(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    return (
        "# Quaternion radial-gauge Phase A\n\n"
        f"Artifact type: `{payload['artifact_type']}`.\n\n"
        f"Frozen materiality outcome: `{decision['outcome']}`.\n\n"
        f"Phase B authorized by the mechanical gate: "
        f"`{str(decision['phase_b_authorized']).lower()}`. Independent scientist review "
        "is still mandatory before any Phase-B execution.\n"
    )


def invalid_phase_a_note(payload: dict[str, Any]) -> str:
    return (
        "# Quaternion radial-gauge Phase A invalid\n\n"
        "The consumed Phase-A attempt failed a frozen preparation, algebra, representation, "
        "gradient, or construction invariant. No materiality conclusion is authorized.\n\n"
        f"Failure: `{payload['failure']['message']}`.\n"
    )


def run_phase_a(seal_path: Path, output_prefix: Path) -> tuple[Path, Path]:
    seal_path = require_default_seal_path(seal_path)
    paths = phase_a_output_paths(output_prefix)
    preflight_paths([*paths.values(), PHASE_A_ATTEMPT])
    seal = load_and_verify_seal(relative_to_root(seal_path))
    marker = attempt_marker(
        artifact_type="quaternion_gauge_iter2_phase_a_attempt",
        marker_path=PHASE_A_ATTEMPT,
        seal=seal,
        outputs=paths,
        inputs={"output_prefix": str(output_prefix)},
    )
    prepared_seeds: list[dict[str, Any]] = []
    seed_evidence: list[dict[str, Any]] = []
    try:
        for seed in SEEDS:
            prepared = prepare_phase_a_seed(seed)
            prerequisites = phase_a_prerequisites(prepared)
            prepared["prerequisites"] = prerequisites
            prepared_seeds.append(prepared)
            seed_evidence.append(
                {
                    "seed": seed,
                    "preparation": preparation_evidence(prepared),
                    "prerequisites": prerequisites,
                }
            )
        prerequisite_failures = [
            f"seed {seed['seed']}: {failure}"
            for seed in seed_evidence
            for failure in seed["prerequisites"]["failures"]
        ]
        if (
            len(prepared_seeds) != len(SEEDS)
            or [prepared["seed"] for prepared in prepared_seeds] != list(SEEDS)
            or [evidence["seed"] for evidence in seed_evidence] != list(SEEDS)
        ):
            raise RuntimeError("Phase-A prepared-seed count/order differs before arm execution")
        if prerequisite_failures:
            invalid = {
                "artifact_type": "quaternion_gauge_iter2_phase_a_invalid",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "bindings": artifact_bindings(seal, marker),
                "environment": environment_metadata(),
                "failure": {
                    "stage": "global_prerequisites",
                    "message": "; ".join(prerequisite_failures),
                },
                "seeds": seed_evidence,
            }
            verify_runtime_bindings(seal_path, marker)
            return write_artifact(paths["invalid_json"], invalid, invalid_phase_a_note(invalid))
    except (ValueError, RuntimeError) as error:
        invalid = {
            "artifact_type": "quaternion_gauge_iter2_phase_a_invalid",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bindings": artifact_bindings(seal, marker),
            "environment": environment_metadata(),
            "failure": {"stage": "preparation_or_prerequisite", "message": str(error)},
            "seeds": seed_evidence,
        }
        verify_runtime_bindings(seal_path, marker)
        return write_artifact(paths["invalid_json"], invalid, invalid_phase_a_note(invalid))

    try:
        for prepared, evidence in zip(prepared_seeds, seed_evidence, strict=True):
            evidence["arms"] = run_phase_a_seed_arms(prepared)
        provisional = {"seeds": seed_evidence}
        invariants = _phase_a_invariants(seed_evidence)
        if not invariants["passed"]:
            invalid = {
                "artifact_type": "quaternion_gauge_iter2_phase_a_invalid",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "bindings": artifact_bindings(seal, marker),
                "environment": environment_metadata(),
                "failure": {
                    "stage": "post_optimizer_construction_invariants",
                    "message": "; ".join(invariants["failures"]),
                },
                "seeds": [
                    {
                        "seed": seed["seed"],
                        "preparation": seed["preparation"],
                        "prerequisites": seed["prerequisites"],
                    }
                    for seed in seed_evidence
                ],
            }
            verify_runtime_bindings(seal_path, marker)
            return write_artifact(paths["invalid_json"], invalid, invalid_phase_a_note(invalid))
        decision = recompute_phase_a_decision(provisional)
    except (ValueError, RuntimeError) as error:
        invalid = {
            "artifact_type": "quaternion_gauge_iter2_phase_a_invalid",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bindings": artifact_bindings(seal, marker),
            "environment": environment_metadata(),
            "failure": {"stage": "optimizer_or_reduction", "message": str(error)},
            "seeds": [
                {
                    "seed": seed["seed"],
                    "preparation": seed["preparation"],
                    "prerequisites": seed["prerequisites"],
                }
                for seed in seed_evidence
            ],
        }
        verify_runtime_bindings(seal_path, marker)
        return write_artifact(paths["invalid_json"], invalid, invalid_phase_a_note(invalid))

    payload = {
        "artifact_type": "quaternion_gauge_iter2_phase_a_audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bindings": artifact_bindings(seal, marker),
        "environment": environment_metadata(),
        "effective_configurations": {
            "fit": asdict(fit_config()),
            "depth_lifter": depth_lifter_config(),
            "phase_a_policies": list(PHASE_A_POLICIES),
            "radial_scales": list(RADIAL_SCALES),
            "checkpoints": list(PHASE_A_CHECKPOINTS),
        },
        "seeds": seed_evidence,
        "decision": decision,
        "interpretation_boundary": (
            "mechanism-only CPU synthetic Depth-surface fixed-topology evidence; no default claim"
        ),
    }
    if recompute_phase_a_decision(payload) != payload["decision"]:
        raise RuntimeError("Phase-A independent decision recomputation differs before write")
    verify_runtime_bindings(seal_path, marker)
    return write_artifact(paths["audit_json"], payload, phase_a_note(payload))


def validate_phase_a_attempt_payload(
    marker: dict[str, Any], seal: dict[str, Any], phase_a_path: Path
) -> None:
    if marker.get("artifact_type") != "quaternion_gauge_iter2_phase_a_attempt":
        raise ValueError("Phase-A marker artifact type differs")
    if marker.get("preregistration") != {
        "path": str(PREREGISTRATION),
        "sha256": PREREGISTRATION_SHA256,
    }:
        raise RuntimeError("Phase-A marker preregistration binding differs")
    if marker.get("seal") != seal:
        raise RuntimeError("Phase-A marker seal binding differs")
    if marker.get("sealed_source_aggregate") != seal["source_aggregate"]:
        raise RuntimeError("Phase-A marker full-sealed source binding differs")
    loaded = marker.get("loaded_source_hashes")
    if not isinstance(loaded, dict) or not loaded:
        raise RuntimeError("Phase-A marker loaded-source path set is missing")
    mandatory_loaded = {
        str(ORIGINAL_PREREGISTRATION),
        str(PREREGISTRATION),
        str(IMPLEMENTATION_REVIEW),
        str(IMPLEMENTATION_REVIEW_ADDENDUM),
        str(HARNESS_PATH),
        "pyproject.toml",
    }
    missing_mandatory = sorted(mandatory_loaded - set(loaded))
    if missing_mandatory:
        raise RuntimeError(
            f"Phase-A marker omits mandatory loaded-source paths: {missing_mandatory}"
        )
    sealed = seal.get("source_hashes")
    if not isinstance(sealed, dict):
        raise RuntimeError("verified seal source hashes are missing")
    unexpected = sorted(set(loaded) - set(sealed))
    mismatched = sorted(
        path for path, digest in loaded.items() if path in sealed and sealed[path] != digest
    )
    if unexpected or mismatched:
        raise RuntimeError(
            "Phase-A marker loaded-source subset differs from its sealed domain: "
            f"unexpected={unexpected}, mismatched={mismatched}"
        )
    loaded_aggregate = canonical_json_hash(loaded)
    if marker.get("loaded_source_aggregate") != loaded_aggregate:
        raise RuntimeError("Phase-A marker loaded-source aggregate differs from its path set")
    phase_a_path = relative_to_root(phase_a_path)
    suffix = "_audit.json"
    if not phase_a_path.name.endswith(suffix):
        raise ValueError("Phase-A artifact does not use the valid audit suffix")
    prefix = phase_a_path.with_name(phase_a_path.name[: -len(suffix)])
    stored_inputs = marker.get("inputs")
    if not isinstance(stored_inputs, dict) or set(stored_inputs) != {"output_prefix"}:
        raise RuntimeError("Phase-A marker output-prefix binding is missing or ambiguous")
    if relative_to_root(Path(stored_inputs["output_prefix"])) != prefix:
        raise RuntimeError("Phase-A marker output-prefix binding differs")
    expected_outputs = phase_a_output_paths(prefix)
    stored_outputs = marker.get("outputs")
    if not isinstance(stored_outputs, dict) or set(stored_outputs) != set(expected_outputs):
        raise RuntimeError("Phase-A marker output key set differs")
    for key, expected in expected_outputs.items():
        if relative_to_root(Path(stored_outputs[key])) != relative_to_root(expected):
            raise RuntimeError(f"Phase-A marker output binding differs for {key}")
    if relative_to_root(Path(stored_outputs["audit_json"])) != phase_a_path:
        raise RuntimeError("Phase-A marker does not bind the supplied audit artifact")


def validate_phase_a_authorization(
    seal_path: Path, phase_a_path: Path, review_path: Path
) -> dict[str, Any]:
    seal_path = require_default_seal_path(seal_path)
    phase_a_path = relative_to_root(phase_a_path)
    review_path = relative_to_root(review_path)
    seal = load_and_verify_seal(seal_path)
    phase_a = strict_json_load(phase_a_path)
    if phase_a.get("artifact_type") != "quaternion_gauge_iter2_phase_a_audit":
        raise ValueError("Phase-B input is not a valid Phase-A audit artifact")
    recomputed = recompute_phase_a_decision(phase_a)
    if recomputed != phase_a.get("decision") or not recomputed["phase_b_authorized"]:
        raise RuntimeError("Phase-A raw evidence does not independently authorize Phase B")
    bindings = phase_a.get("bindings", {})
    validate_artifact_bindings(bindings, seal)
    attempt_path = relative_to_root(Path(bindings["attempt_marker_path"]))
    if attempt_path != PHASE_A_ATTEMPT.resolve():
        raise RuntimeError("Phase-A artifact names a noncanonical attempt marker")
    if sha256_file(attempt_path) != bindings.get("attempt_marker_sha256"):
        raise RuntimeError("Phase-A attempt-marker binding differs")
    marker_payload = strict_json_load(attempt_path)
    validate_phase_a_attempt_payload(marker_payload, seal, phase_a_path)
    human_audit = phase_a_path.with_name(f"{phase_a_path.stem}_AUDIT.md")
    derived_review = phase_a_path.with_name(f"{phase_a_path.stem}_AUDIT.json")
    if review_path != derived_review:
        raise ValueError("supplied machine review path is not the derived Phase-A review path")
    if not human_audit.is_file() or not review_path.is_file():
        raise FileNotFoundError("derived human and machine scientist reviews must both exist")
    review = strict_json_load(review_path)
    if set(review) != SCIENTIST_REVIEW_KEYS:
        raise ValueError("machine scientist review key set is not exact")
    expected_review = {
        "artifact_type": "quaternion_gauge_iter2_phase_a_scientist_review",
        "verdict": "pass",
        "phase_b_execution_clearance": True,
        "phase_a_sha256": sha256_file(phase_a_path),
        "human_audit_sha256": sha256_file(human_audit),
        "seal_sha256": seal["sha256"],
        "phase_a_attempt_sha256": sha256_file(attempt_path),
        "source_aggregate": seal["source_aggregate"],
    }
    if review != expected_review:
        raise RuntimeError("machine scientist review does not exactly bind authorized inputs")
    return {
        "seal": seal,
        "phase_a": phase_a,
        "phase_a_path": phase_a_path,
        "phase_a_sha256": sha256_file(phase_a_path),
        "phase_a_attempt_path": attempt_path,
        "phase_a_attempt_sha256": sha256_file(attempt_path),
        "human_audit_path": human_audit,
        "human_audit_sha256": sha256_file(human_audit),
        "review_path": review_path,
        "review_sha256": sha256_file(review_path),
    }


def phase_b_note(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    return (
        "# Quaternion retraction Phase B\n\n"
        f"Frozen outcome: `{decision['outcome']}`.\n\n"
        f"Confirmatory candidate: `{decision['confirmatory_candidate']}`.\n\n"
        "This CPU synthetic result cannot change a production default. An independent results "
        "audit, a preregistered real/calibrated test, and CUDA/gsplat parity remain required.\n"
    )


def verify_authorization_inputs_unchanged(authorization: dict[str, Any]) -> None:
    checks = {
        authorization["phase_a_path"]: authorization["phase_a_sha256"],
        authorization["phase_a_attempt_path"]: authorization["phase_a_attempt_sha256"],
        authorization["human_audit_path"]: authorization["human_audit_sha256"],
        authorization["review_path"]: authorization["review_sha256"],
    }
    mismatches = [str(path) for path, digest in checks.items() if sha256_file(path) != digest]
    if mismatches:
        raise RuntimeError(f"Phase-B authorization input changed during execution: {mismatches}")


def run_phase_b(
    seal_path: Path,
    phase_a_path: Path,
    review_path: Path,
    output: Path,
) -> tuple[Path, Path]:
    seal_path = require_default_seal_path(seal_path)
    paths = phase_b_output_paths(output)
    preflight_paths([*paths.values(), PHASE_B_ATTEMPT])
    authorization = validate_phase_a_authorization(seal_path, phase_a_path, review_path)
    inputs = {
        "phase_a_path": str(authorization["phase_a_path"]),
        "phase_a_sha256": authorization["phase_a_sha256"],
        "phase_a_attempt_path": str(authorization["phase_a_attempt_path"]),
        "phase_a_attempt_sha256": authorization["phase_a_attempt_sha256"],
        "human_audit_path": str(authorization["human_audit_path"]),
        "human_audit_sha256": authorization["human_audit_sha256"],
        "machine_review_path": str(authorization["review_path"]),
        "machine_review_sha256": authorization["review_sha256"],
    }
    marker = attempt_marker(
        artifact_type="quaternion_gauge_iter2_phase_b_attempt",
        marker_path=PHASE_B_ATTEMPT,
        seal=authorization["seal"],
        outputs=paths,
        inputs=inputs,
    )
    phase_a_by_seed = {
        seed["seed"]: seed["preparation"]["training_hashes"]
        for seed in authorization["phase_a"]["seeds"]
    }
    seed_records = []
    for seed in SEEDS:
        prepared = prepare_seed(seed, retain_evaluator=True)
        if prepared["training_hashes"] != phase_a_by_seed[seed]:
            raise RuntimeError(f"seed {seed} fresh Phase-B training hashes differ from Phase A")
        seed_records.append(run_phase_b_seed(prepared))
    provisional = {"seeds": seed_records}
    decision = recompute_phase_b_decision(provisional)
    payload = {
        "artifact_type": "quaternion_gauge_iter2_phase_b_ablation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bindings": {
            **artifact_bindings(authorization["seal"], marker),
            **inputs,
        },
        "environment": environment_metadata(),
        "effective_configurations": {arm: asdict(train_config(0, arm)) for arm in PHASE_B_ARMS},
        "seeds": seed_records,
        "decision": decision,
        "interpretation_boundary": (
            "CPU synthetic Depth-surface fixed-topology only; confirmatory candidate at most"
        ),
    }
    if recompute_phase_b_decision(payload) != decision:
        raise RuntimeError("Phase-B independent decision recomputation differs before write")
    verify_runtime_bindings(seal_path, marker)
    verify_authorization_inputs_unchanged(authorization)
    return write_artifact(paths["json"], payload, phase_b_note(payload))


def configure_official_runtime() -> None:
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    seal = subparsers.add_parser("seal", help="verify and bind the frozen implementation")
    seal.add_argument("--output", type=Path, required=True)
    audit = subparsers.add_parser("audit", help="run the frozen Phase-A mechanism audit")
    audit.add_argument("--seal", type=Path, required=True)
    audit.add_argument("--output-prefix", type=Path, required=True)
    run = subparsers.add_parser("run", help="run authorized fresh Phase-B joint refinement")
    run.add_argument("--seal", type=Path, required=True)
    run.add_argument("--phase-a", type=Path, required=True)
    run.add_argument("--review", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_official_runtime()
    if args.action == "seal":
        output = relative_to_root(args.output)
        expected = relative_to_root(DEFAULT_SEAL)
        if output != expected:
            raise ValueError(f"sole seal output is {DEFAULT_SEAL}")
        note = output.with_name(f"{output.stem}_RESULT.md")
        preflight_paths([output, note])
        payload = create_seal()
        write_artifact(
            output,
            payload,
            "# Quaternion radial-gauge implementation seal\n\n"
            "The frozen implementation, independent implementation review, sources, environment, "
            "and full verification are bound by this append-only seal.\n",
        )
    elif args.action == "audit":
        run_phase_a(args.seal, args.output_prefix)
    else:
        run_phase_b(args.seal, args.phase_a, args.review, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
