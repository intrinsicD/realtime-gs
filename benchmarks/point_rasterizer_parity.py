#!/usr/bin/env python3
"""Sealed CPU point-rasterizer parity and discrete-risk mechanism experiment."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import (
    GaussianObservationField,
    GaussianPixelProposal,
    fixed_attempt_mean,
)
from rtgs.core.sh import eval_sh_preactivation, rgb_to_sh
from rtgs.render.torch_points import TorchPointRasterizer
from rtgs.render.torch_ref import (
    _DILATION,
    _MAX_ALPHA,
    _NEAR,
    TorchRasterizer,
    kernel_support_weight,
)

ROOT = Path(__file__).resolve().parent.parent
PREREGISTRATION = Path("benchmarks/results/20260716_point_rasterizer_parity_PREREG.md")
PREREGISTRATION_SHA256 = "afc9d036ad1c037a5cb3eab7fd5b19f97d37d920f520cb5c51bf37f41f989916"
IMPLEMENTATION_REVIEW = Path(
    "benchmarks/results/20260716_point_rasterizer_parity_IMPLEMENTATION_REVIEW.md"
)
SEAL = ROOT / "benchmarks/results/20260716_point_rasterizer_parity_SEAL.json"
ATTEMPT = ROOT / "benchmarks/results/20260716_point_rasterizer_parity_ATTEMPT.json"
RESULT = ROOT / "benchmarks/results/20260716_point_rasterizer_parity_RESULT.json"
AUDIT = ROOT / "benchmarks/results/20260716_point_rasterizer_parity_AUDIT.md"
RUN_DIR = ROOT / "runs/point_rasterizer_parity_20260716"
CALIBRATED_RESULT = RUN_DIR / "calibrated_parity.json"
CALIBRATION = ROOT / "dataset/2025_03_07_stage_with_fabric/calibration_dome.json"
CALIBRATED_PLY = ROOT / "runs/dataset_viewer_fullres_20260716/gaussians_init.ply"

OFFICIAL_SEEDS = (91301, 91302, 91303)
POINT_CHUNKS = (1, 7, 4096)
GAUSSIAN_CHUNKS = (1, 3, 4096)
FORWARD_ATOL = 2e-6
FORWARD_RTOL = 2e-5
GRADIENT_ATOL = 4e-6
GRADIENT_RTOL = 5e-5
FIELD_NAMES = ("means", "quats", "log_scales", "opacity", "sh")
FROZEN_ANCHOR_HASHES = {
    "src/rtgs/render/torch_ref.py": (
        "61716787329e85a186982f81c2a89cb270255473ca26688c409191a1b53bd86e"
    ),
    "src/rtgs/render/base.py": "1175cf359e2800ff3a518849b43c4d9a6fd6dccc3dfb7c24459f13e9f81ca0b9",
    "src/rtgs/core/gaussians3d.py": (
        "d417a4a103ae7ea1e3f4a7799c2b709597014b8966acb0e72b2bd447a0ad0ba5"
    ),
    "src/rtgs/core/camera.py": "1e6a42c7cd9fa14b2ffff19808e6e88c106df4562d30fc18b0ca107c00072ac2",
    "src/rtgs/core/sh.py": "554f3a25e25c7312248a98c15685e9bf805c85a81a96f56e13e1481619eb4687",
    "src/rtgs/lift/compact_carve.py": (
        "87efa40b4e5ac40684367e723a57a46a2f08b8613d225124daf3144cff1afa83"
    ),
    "src/rtgs/data/reconstruction_inputs.py": (
        "2f93b571760c61d8fce6ecc5bfcfe103ecbce2049d4c15c3c43c33132577376b"
    ),
}

SEALED_PATHS = tuple(
    sorted(
        {
            PREREGISTRATION,
            IMPLEMENTATION_REVIEW,
            Path("benchmarks/point_rasterizer_parity.py"),
            Path("pyproject.toml"),
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


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {value!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    header = canonical_json({"dtype": str(tensor.dtype), "shape": list(tensor.shape)}).encode()
    return sha256_bytes(header + b"\0" + tensor.numpy().tobytes(order="C"))


def assert_finite_tree(value: Any, context: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"{context} contains a non-finite float")
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_tree(item, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_tree(item, f"{context}[{index}]")


def environment_metadata() -> dict[str, Any]:
    loadavg = Path("/proc/loadavg")
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "device": "cpu",
        "loadavg": loadavg.read_text().strip() if loadavg.is_file() else None,
    }


def environment_fingerprint(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key != "loadavg"}


def assert_official_environment(metadata: dict[str, Any]) -> None:
    expected = {
        "torch_num_threads": 4,
        "deterministic_algorithms": True,
        "cuda_visible_devices": "",
        "omp_num_threads": "4",
        "mkl_num_threads": "4",
        "device": "cpu",
    }
    mismatch = {
        key: {"expected": expected_value, "actual": metadata.get(key)}
        for key, expected_value in expected.items()
        if metadata.get(key) != expected_value
    }
    if mismatch:
        raise RuntimeError(f"official environment differs from preregistration: {mismatch}")


def git_metadata() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    return {
        "revision": revision,
        "dirty": bool(status.strip()),
        "status": status.splitlines(),
        "tracked_diff_sha256": sha256_bytes(diff.encode()),
    }


def source_hashes() -> tuple[dict[str, str], str]:
    missing = [str(path) for path in SEALED_PATHS if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"sealed source files are missing: {missing}")
    hashes = {str(path): sha256_file(ROOT / path) for path in SEALED_PATHS}
    return hashes, canonical_hash(hashes)


def verify_source_hashes(expected: dict[str, str], aggregate: str) -> None:
    actual, actual_aggregate = source_hashes()
    if actual != expected or actual_aggregate != aggregate:
        raise RuntimeError("repository sources differ from the implementation seal")


def run_preseal_verification(
    expected_hashes: dict[str, str], expected_aggregate: str
) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    commands = (
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_point_render.py",
            "tests/test_observation2d.py",
            "tests/test_point_rasterizer_parity.py",
        ],
        ["./scripts/verify.sh"],
    )
    records = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        output = completed.stdout + completed.stderr
        records.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "output_sha256": sha256_bytes(output.encode()),
                "output": output,
            }
        )
        if completed.returncode != 0:
            raise RuntimeError(f"preseal verification failed: {command}\n{output}")
        verify_source_hashes(expected_hashes, expected_aggregate)
    return {"passed": True, "commands": records}


def _exclusive_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
    return sha256_bytes(rendered.encode())


def create_seal() -> dict[str, Any]:
    if sha256_file(ROOT / PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("preregistration differs from the frozen hash")
    for relative, expected in FROZEN_ANCHOR_HASHES.items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"frozen dense-anchor source drifted: {relative}")
    review = (ROOT / IMPLEMENTATION_REVIEW).read_text(encoding="utf-8")
    if re.search(r"verdict\s*:\s*`?PASS`?", review, flags=re.IGNORECASE) is None:
        raise RuntimeError("implementation review does not contain a PASS verdict")
    paths = (SEAL, ATTEMPT, RESULT, AUDIT, CALIBRATED_RESULT)
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to seal over existing artifacts: {existing}")
    hashes, aggregate = source_hashes()
    verification = run_preseal_verification(hashes, aggregate)
    environment = environment_metadata()
    assert_official_environment(environment)
    payload: dict[str, Any] = {
        "artifact_type": "point_rasterizer_parity_implementation_seal",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration": {"path": str(PREREGISTRATION), "sha256": PREREGISTRATION_SHA256},
        "implementation_review": {
            "path": str(IMPLEMENTATION_REVIEW),
            "sha256": sha256_file(ROOT / IMPLEMENTATION_REVIEW),
        },
        "source_hashes": hashes,
        "source_aggregate": aggregate,
        "verification": verification,
        "git": git_metadata(),
        "environment": environment,
        "command": [sys.executable, *sys.argv],
    }
    payload["seal_payload_sha256"] = canonical_hash(payload)
    return payload


def load_and_verify_seal() -> dict[str, Any]:
    seal = strict_json_load(SEAL)
    if not isinstance(seal, dict):
        raise RuntimeError("seal must be a JSON object")
    digest = seal.get("seal_payload_sha256")
    body = dict(seal)
    body.pop("seal_payload_sha256", None)
    if digest != canonical_hash(body):
        raise RuntimeError("seal self-digest mismatch")
    if seal.get("artifact_type") != "point_rasterizer_parity_implementation_seal":
        raise RuntimeError("unexpected seal artifact type")
    if seal.get("verification", {}).get("passed") is not True:
        raise RuntimeError("seal does not bind passing preseal verification")
    if seal.get("preregistration", {}).get("sha256") != PREREGISTRATION_SHA256:
        raise RuntimeError("seal binds the wrong preregistration")
    verify_source_hashes(seal["source_hashes"], seal["source_aggregate"])
    verify_git_binding(seal)
    seal["seal_file_sha256"] = sha256_file(SEAL)
    return seal


def verify_git_binding(seal: dict[str, Any]) -> None:
    current_git = git_metadata()
    for key in ("revision", "tracked_diff_sha256"):
        if current_git[key] != seal["git"][key]:
            raise RuntimeError(f"git {key} differs from the implementation seal")


def pixel_centers(camera: Camera) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(camera.height, dtype=torch.float32) + 0.5,
        torch.arange(camera.width, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    return torch.stack([x, y], dim=-1).reshape(-1, 2)


def official_fixture(seed: int) -> tuple[Gaussians3D, Camera]:
    """Literal preregistered fixture; do not call before the atomic attempt marker."""
    if seed not in OFFICIAL_SEEDS:
        raise ValueError("not a frozen official seed")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    camera = Camera.look_at(
        torch.tensor([0.0, 0.0, -3.0]),
        torch.zeros(3),
        fov_x_deg=55.0,
        width=19,
        height=15,
    )
    means = torch.empty(11, 3, dtype=torch.float32)
    means[0] = torch.tensor([0.0, 0.0, -0.65])
    means[1] = torch.tensor([0.0, 0.0, 0.35])
    means[2] = torch.tensor([-0.25, 0.12, 0.0])
    means[3] = torch.tensor([0.31, -0.18, 0.0])
    means[4:9, 0] = 1.2 * torch.rand(5, generator=generator) - 0.6
    means[4:9, 1] = 0.8 * torch.rand(5, generator=generator) - 0.4
    means[4:9, 2] = 1.2 * torch.rand(5, generator=generator) - 0.6
    means[9] = torch.tensor([2.5, 0.0, 0.0])
    means[10] = torch.tensor([0.0, 0.0, -3.2])
    quats = torch.randn(11, 4, generator=generator)
    log_scales = torch.log(0.04 + 0.14 * torch.rand(11, 3, generator=generator))
    opacity = 0.12 + 0.76 * torch.rand(11, generator=generator)
    sh = 0.04 * torch.randn(11, 9, 3, generator=generator)
    sh[:, 0] = rgb_to_sh(0.15 + 0.70 * torch.rand(11, 3, generator=generator))
    return Gaussians3D(means, quats, log_scales, opacity, sh), camera


def leaf_clone(source: Gaussians3D) -> Gaussians3D:
    return Gaussians3D(
        **{
            name: getattr(source, name).detach().clone().requires_grad_(True)
            for name in FIELD_NAMES
        }
    )


def error_record(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual - expected).abs()
    relative = difference / expected.abs().clamp_min(1e-12)
    return {
        "maximum_absolute_error": float(difference.max()) if difference.numel() else 0.0,
        "maximum_relative_error": float(relative.max()) if relative.numel() else 0.0,
    }


def require_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    context: str,
) -> dict[str, float]:
    record = error_record(actual, expected)
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise RuntimeError(f"{context} parity failed: {record}")
    return record


def fixture_boundary_audit(
    gaussians: Gaussians3D,
    camera: Camera,
    query_xy: torch.Tensor,
) -> dict[str, Any]:
    """Assert the frozen fixture is away from every relevant hard boundary."""
    means_cam = camera.world_to_cam(gaussians.means)
    z = means_cam[:, 2]
    uv, _ = camera.project(gaussians.means)
    cov_world = gaussians.covariance()
    rotation = camera.R.to(gaussians.means.device)
    cov_cam = rotation @ cov_world @ rotation.T
    safe_z = z.clamp_min(_NEAR)
    jacobian = torch.zeros(gaussians.n, 2, 3, dtype=gaussians.means.dtype)
    jacobian[:, 0, 0] = camera.fx / safe_z
    jacobian[:, 0, 2] = -camera.fx * means_cam[:, 0] / safe_z.square()
    jacobian[:, 1, 1] = camera.fy / safe_z
    jacobian[:, 1, 2] = -camera.fy * means_cam[:, 1] / safe_z.square()
    cov2d = jacobian @ cov_cam @ jacobian.transpose(-1, -2)
    cov2d = cov2d + _DILATION * torch.eye(2)
    eig_max = (
        0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
        + (0.25 * (cov2d[:, 0, 0] - cov2d[:, 1, 1]).square() + cov2d[:, 0, 1].square()).sqrt()
    )
    radii = 3.0 * eig_max.clamp_min(1e-8).sqrt()
    in_front = z > _NEAR
    visible_mask = in_front & camera.in_image(uv, margin=radii.detach())
    visible = visible_mask.nonzero(as_tuple=True)[0]

    if set(visible.tolist()) != set(range(9)):
        raise RuntimeError(f"fixture visibility differs from rows 0..8: {visible.tolist()}")
    if bool(in_front[10]) or bool(visible_mask[9]) or bool(visible_mask[10]):
        raise RuntimeError("fixture outside/behind rows do not satisfy their frozen roles")
    if float(gaussians.quats.norm(dim=1).min()) <= 1e-4:
        raise RuntimeError("fixture contains a degenerate quaternion")
    center_error = float((uv[0] - torch.tensor([9.5, 7.5])).abs().max())
    if not torch.equal(uv[0], uv[1]) or center_error > 1e-6:
        raise RuntimeError("rows 0 and 1 do not share the frozen center")
    if z[2] != z[3]:
        raise RuntimeError("rows 2 and 3 do not have equal camera depth")

    sorted_visible = visible[torch.argsort(z[visible])]
    cov_visible = cov2d[sorted_visible]
    determinant = (
        cov_visible[:, 0, 0] * cov_visible[:, 1, 1] - cov_visible[:, 0, 1].square()
    ).clamp_min(1e-12)
    i00 = cov_visible[:, 1, 1] / determinant
    i01 = -cov_visible[:, 0, 1] / determinant
    i11 = cov_visible[:, 0, 0] / determinant
    delta = query_xy[:, None, :] - uv[sorted_visible][None, :, :]
    q = (
        delta[..., 0].square() * i00[None]
        + 2.0 * delta[..., 0] * delta[..., 1] * i01[None]
        + delta[..., 1].square() * i11[None]
    )
    q_boundary_distances = {
        "12": float((q - 12.0).abs().min()),
        "16": float((q - 16.0).abs().min()),
    }
    alpha = (gaussians.opacity[sorted_visible][None] * kernel_support_weight(q, "hard")).clamp(
        0.0, _MAX_ALPHA
    )
    positive_alpha = alpha[alpha > 0]
    alpha_cap_distance = float((_MAX_ALPHA - positive_alpha).abs().min())
    directions = torch.nn.functional.normalize(
        gaussians.means[sorted_visible] - camera.position, dim=-1
    )
    sh_floor_distances = {
        str(degree): float(
            eval_sh_preactivation(degree, gaussians.sh[sorted_visible], directions).abs().min()
        )
        for degree in (0, 2)
    }
    near_distance = float((z - _NEAR).abs().min())
    envelope_slacks = torch.stack(
        [
            uv[:, 0] - (0.5 - radii),
            camera.width - 0.5 + radii - uv[:, 0],
            uv[:, 1] - (0.5 - radii),
            camera.height - 0.5 + radii - uv[:, 1],
        ],
        dim=1,
    )
    cull_boundary_distance = float(envelope_slacks[in_front].abs().min())
    distances = {
        "q_hard_support_12": q_boundary_distances["12"],
        "q_taper_support_16": q_boundary_distances["16"],
        "positive_alpha_cap": alpha_cap_distance,
        "sh_floor_degree_0": sh_floor_distances["0"],
        "sh_floor_degree_2": sh_floor_distances["2"],
        "near_plane": near_distance,
        "cull_envelope": cull_boundary_distance,
    }
    if min(distances.values()) <= 1e-5:
        raise RuntimeError(f"official fixture lies on a hard boundary: {distances}")
    return {
        "visible": sorted_visible.tolist(),
        "visible_hash": tensor_hash(sorted_visible),
        "minimum_boundary_distances": distances,
        "means_hash": tensor_hash(gaussians.means),
        "quats_hash": tensor_hash(gaussians.quats),
        "log_scales_hash": tensor_hash(gaussians.log_scales),
        "opacity_hash": tensor_hash(gaussians.opacity),
        "sh_hash": tensor_hash(gaussians.sh),
    }


def compare_point_to_dense(
    gaussians: Gaussians3D,
    camera: Camera,
    xy: torch.Tensor,
    *,
    point_chunk: int,
    gaussian_chunk: int,
    background: torch.Tensor | None,
    sh_degree: int | None,
    sh_color_activation: str = "hard",
    kernel_support_mode: str = "hard",
) -> dict[str, Any]:
    dense = TorchRasterizer(
        row_chunk=4,
        sh_color_activation=sh_color_activation,
        kernel_support_mode=kernel_support_mode,
    ).render(gaussians, camera, background, sh_degree)
    point = TorchPointRasterizer(
        point_chunk=point_chunk,
        gaussian_chunk=gaussian_chunk,
        sh_color_activation=sh_color_activation,
        kernel_support_mode=kernel_support_mode,
    ).render_points(gaussians, camera, xy, background, sh_degree)
    flat = (xy[:, 1] - 0.5).long() * camera.width + (xy[:, 0] - 0.5).long()
    if not torch.equal(point.visible, dense.visible):
        raise RuntimeError("point and dense visible indices/order differ")
    return {
        "color": require_close(
            point.color,
            dense.color.reshape(-1, 3)[flat],
            atol=FORWARD_ATOL,
            rtol=FORWARD_RTOL,
            context="color",
        ),
        "alpha": require_close(
            point.alpha,
            dense.alpha.reshape(-1)[flat],
            atol=FORWARD_ATOL,
            rtol=FORWARD_RTOL,
            context="alpha",
        ),
        "depth": require_close(
            point.depth,
            dense.depth.reshape(-1)[flat],
            atol=FORWARD_ATOL,
            rtol=FORWARD_RTOL,
            context="depth",
        ),
        "visible": point.visible.tolist() if point.visible is not None else None,
    }


def run_forward_gates(fixtures: dict[int, tuple[Gaussians3D, Camera]]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    backgrounds = {"black": None, "nonblack": torch.tensor([0.13, 0.29, 0.47])}
    for seed, (gaussians, camera) in fixtures.items():
        xy = pixel_centers(camera)
        for background_name, background in backgrounds.items():
            for degree in (0, 2):
                for point_chunk in POINT_CHUNKS:
                    for gaussian_chunk in GAUSSIAN_CHUNKS:
                        cases.append(
                            {
                                "seed": seed,
                                "background": background_name,
                                "sh_degree": degree,
                                "point_chunk": point_chunk,
                                "gaussian_chunk": gaussian_chunk,
                                "errors": compare_point_to_dense(
                                    gaussians,
                                    camera,
                                    xy,
                                    point_chunk=point_chunk,
                                    gaussian_chunk=gaussian_chunk,
                                    background=background,
                                    sh_degree=degree,
                                ),
                            }
                        )

    gaussians, camera = fixtures[91301]
    xy = pixel_centers(camera)
    supplemental = []
    modes = (
        ("smu1", "hard"),
        ("hard_forward_smu1_negative_gradient", "hard"),
        ("hard", "c1_taper"),
        ("hard", "hard_forward_c1_taper_gradient"),
    )
    for activation, kernel in modes:
        supplemental.append(
            {
                "sh_color_activation": activation,
                "kernel_support_mode": kernel,
                "errors": compare_point_to_dense(
                    gaussians,
                    camera,
                    xy,
                    point_chunk=7,
                    gaussian_chunk=3,
                    background=torch.tensor([0.13, 0.29, 0.47]),
                    sh_degree=2,
                    sh_color_activation=activation,
                    kernel_support_mode=kernel,
                ),
            }
        )
    return {"primary_case_count": len(cases), "primary_cases": cases, "supplemental": supplemental}


def gradient_loss(
    color: torch.Tensor,
    alpha: torch.Tensor,
    depth: torch.Tensor,
    wc: torch.Tensor,
    wa: torch.Tensor,
    wd: torch.Tensor,
) -> torch.Tensor:
    count = alpha.shape[0]
    return (
        (color * wc).sum() / (3 * count)
        + 0.17 * (alpha * wa).sum() / count
        + 0.03 * (depth * wd).sum() / count
    )


def dense_gradient_anchor(
    source: Gaussians3D, camera: Camera, seed: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    gaussians = leaf_clone(source)
    coefficients = torch.Generator(device="cpu").manual_seed(seed + 100000)
    wc = torch.randn(285, 3, generator=coefficients, dtype=torch.float32)
    wa = torch.randn(285, generator=coefficients, dtype=torch.float32)
    wd = torch.randn(285, generator=coefficients, dtype=torch.float32)
    for name, value in (("wc", wc), ("wa", wa), ("wd", wd)):
        if not bool(torch.isfinite(value).all()) or not bool((value != 0).any()):
            raise RuntimeError(f"gradient coefficient {name} is vacuous")
    output = TorchRasterizer(row_chunk=4).render(
        gaussians,
        camera,
        background=torch.tensor([0.13, 0.29, 0.47]),
        sh_degree=2,
    )
    loss = gradient_loss(
        output.color.reshape(-1, 3),
        output.alpha.reshape(-1),
        output.depth.reshape(-1),
        wc,
        wa,
        wd,
    )
    loss.backward()
    if output.means2d is None or output.means2d.grad is None:
        raise RuntimeError("dense anchor did not retain means2d gradients")
    gradients: dict[str, torch.Tensor] = {}
    maxima: dict[str, float] = {}
    for name in FIELD_NAMES:
        gradient = getattr(gaussians, name).grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError(f"dense anchor has invalid {name} gradient")
        gradients[name] = gradient.detach().clone()
        maxima[name] = float(gradient.abs().max())
    gradients["means2d"] = output.means2d.grad.detach().clone()
    maxima["means2d"] = float(output.means2d.grad.abs().max())
    if any(value <= 1e-10 for value in maxima.values()):
        raise RuntimeError(f"dense gradient anchor is vacuous: {maxima}")
    return gradients, {
        "loss": float(loss.detach()),
        "maximum_absolute_gradient": maxima,
        "coefficient_hashes": {"wc": tensor_hash(wc), "wa": tensor_hash(wa), "wd": tensor_hash(wd)},
    }


def run_gradient_gates(fixtures: dict[int, tuple[Gaussians3D, Camera]]) -> dict[str, Any]:
    records = []
    background = torch.tensor([0.13, 0.29, 0.47])
    for seed, (source, camera) in fixtures.items():
        anchor, anchor_record = dense_gradient_anchor(source, camera, seed)
        coefficient_generator = torch.Generator(device="cpu").manual_seed(seed + 100000)
        wc = torch.randn(285, 3, generator=coefficient_generator, dtype=torch.float32)
        wa = torch.randn(285, generator=coefficient_generator, dtype=torch.float32)
        wd = torch.randn(285, generator=coefficient_generator, dtype=torch.float32)
        seed_cases = []
        for point_chunk in POINT_CHUNKS:
            for gaussian_chunk in GAUSSIAN_CHUNKS:
                gaussians = leaf_clone(source)
                output = TorchPointRasterizer(
                    point_chunk=point_chunk, gaussian_chunk=gaussian_chunk
                ).render_points(gaussians, camera, pixel_centers(camera), background, sh_degree=2)
                loss = gradient_loss(output.color, output.alpha, output.depth, wc, wa, wd)
                loss.backward()
                if output.means2d is None or output.means2d.grad is None:
                    raise RuntimeError("point renderer did not retain means2d gradients")
                errors = {}
                for name in FIELD_NAMES:
                    gradient = getattr(gaussians, name).grad
                    if gradient is None or not bool(torch.isfinite(gradient).all()):
                        raise RuntimeError(f"point renderer has invalid {name} gradient")
                    errors[name] = require_close(
                        gradient,
                        anchor[name],
                        atol=GRADIENT_ATOL,
                        rtol=GRADIENT_RTOL,
                        context=f"{name} gradient",
                    )
                errors["means2d"] = require_close(
                    output.means2d.grad,
                    anchor["means2d"],
                    atol=GRADIENT_ATOL,
                    rtol=GRADIENT_RTOL,
                    context="means2d gradient",
                )
                seed_cases.append(
                    {
                        "point_chunk": point_chunk,
                        "gaussian_chunk": gaussian_chunk,
                        "loss": float(loss.detach()),
                        "errors": errors,
                    }
                )
        records.append({"seed": seed, "dense_anchor": anchor_record, "cases": seed_cases})
    return {"seed_count": len(records), "cases_per_seed": 9, "seeds": records}


def run_global_compositor_gate(source: Gaussians3D, camera: Camera) -> dict[str, Any]:
    renderer = TorchPointRasterizer(point_chunk=1, gaussian_chunk=1)
    xy = torch.tensor([[9.5, 7.5]], dtype=torch.float32)
    baseline = renderer.render_points(source, camera, xy, sh_degree=2)
    changed_sh = source.sh.detach().clone()
    changed_sh[0].zero_()
    changed_sh[0, 0] = rgb_to_sh(torch.tensor([0.95, 0.05, 0.15], dtype=torch.float32))
    changed_opacity = source.opacity.detach().clone()
    changed_opacity[0] = 0.77
    changed = Gaussians3D(
        source.means.detach().clone(),
        source.quats.detach().clone(),
        source.log_scales.detach().clone(),
        changed_opacity,
        changed_sh,
    )
    intervention = renderer.render_points(changed, camera, xy, sh_degree=2)
    color_change = float((intervention.color - baseline.color).abs().max())
    if color_change < 1e-4:
        raise RuntimeError(f"near-Gaussian intervention was not material: {color_change}")
    dense_changed = TorchRasterizer().render(changed, camera, sh_degree=2)
    dense_color = dense_changed.color[7, 9][None]
    dense_alpha = dense_changed.alpha[7, 9][None]
    dense_depth = dense_changed.depth[7, 9][None]
    parity = {
        "color": require_close(
            intervention.color,
            dense_color,
            atol=FORWARD_ATOL,
            rtol=FORWARD_RTOL,
            context="intervention color",
        ),
        "alpha": require_close(
            intervention.alpha,
            dense_alpha,
            atol=FORWARD_ATOL,
            rtol=FORWARD_RTOL,
            context="intervention alpha",
        ),
        "depth": require_close(
            intervention.depth,
            dense_depth,
            atol=FORWARD_ATOL,
            rtol=FORWARD_RTOL,
            context="intervention depth",
        ),
    }

    permutation = torch.arange(source.n)
    permutation[0], permutation[1] = permutation[1].clone(), permutation[0].clone()
    reversed_output = renderer.render_points(source.subset(permutation), camera, xy, sh_degree=2)
    reversal = {
        "color": require_close(
            reversed_output.color,
            baseline.color,
            atol=FORWARD_ATOL,
            rtol=FORWARD_RTOL,
            context="distinct-depth reversal color",
        ),
        "alpha": require_close(
            reversed_output.alpha,
            baseline.alpha,
            atol=FORWARD_ATOL,
            rtol=FORWARD_RTOL,
            context="distinct-depth reversal alpha",
        ),
        "depth": require_close(
            reversed_output.depth,
            baseline.depth,
            atol=FORWARD_ATOL,
            rtol=FORWARD_RTOL,
            context="distinct-depth reversal depth",
        ),
    }
    parameters = inspect.signature(renderer.render_points).parameters
    forbidden = sorted({"proposal", "proposer", "lineage", "component_ids"} & set(parameters))
    if forbidden:
        raise RuntimeError(
            f"point renderer exposes forbidden local assignment arguments: {forbidden}"
        )
    return {
        "synthetic_proposer_metadata_only": 1,
        "changed_global_gaussian": 0,
        "maximum_color_change": color_change,
        "intervention_parity": parity,
        "distinct_depth_reversal": reversal,
        "forbidden_api_arguments": forbidden,
    }


def run_empty_gates(source: Gaussians3D, camera: Camera) -> dict[str, Any]:
    behind = Gaussians3D(
        means=torch.tensor([0.0, 0.0, -3.2]).repeat(source.n, 1),
        quats=source.quats.detach().clone(),
        log_scales=source.log_scales.detach().clone(),
        opacity=source.opacity.detach().clone(),
        sh=source.sh.detach().clone(),
    )
    xy = torch.tensor([[0.5, 0.5], [9.5, 7.5]], dtype=torch.float32)
    background = torch.tensor([0.13, 0.29, 0.47])
    point = TorchPointRasterizer(point_chunk=1, gaussian_chunk=1).render_points(
        behind, camera, xy, background, sh_degree=2
    )
    dense = TorchRasterizer(row_chunk=4).render(behind, camera, background, sh_degree=2)
    flat = torch.tensor([0, 7 * camera.width + 9])
    if point.visible is None or point.visible.numel() != 0 or point.means2d is not None:
        raise RuntimeError("empty-visible metadata contract failed")
    if not torch.equal(point.color, background[None].expand(2, 3)):
        raise RuntimeError("empty-visible point colors are not exact background")
    if not torch.equal(point.alpha, torch.zeros(2)) or not torch.equal(point.depth, torch.zeros(2)):
        raise RuntimeError("empty-visible alpha/depth are not exact zeros")
    require_close(
        point.color,
        dense.color.reshape(-1, 3)[flat],
        atol=0.0,
        rtol=0.0,
        context="empty-visible color",
    )
    empty_xy = torch.empty(0, 2, dtype=torch.float32)
    empty = TorchPointRasterizer().render_points(source, camera, empty_xy, background, sh_degree=2)
    if empty.color.shape != (0, 3) or empty.alpha.shape != (0,) or empty.depth.shape != (0,):
        raise RuntimeError("empty-query output shapes differ from the contract")
    if empty.visible is None or empty.visible.numel() == 0 or empty.means2d is None:
        raise RuntimeError("empty-query did not preserve camera visibility metadata")
    return {
        "empty_visible_color_exact": True,
        "empty_visible_zero_alpha_depth_exact": True,
        "empty_query_shapes": [[0, 3], [0], [0]],
        "empty_query_visible_count": int(empty.visible.numel()),
    }


def run_continuous_coordinate_gates(
    fixtures: dict[int, tuple[Gaussians3D, Camera]],
) -> dict[str, Any]:
    frozen_xy = torch.tensor(
        [[0.75, 0.75], [4.125, 3.875], [18.25, 14.25], [2.2, 5.4]],
        dtype=torch.float32,
    )
    records = []
    for seed, (source, camera) in fixtures.items():
        gaussians = leaf_clone(source)
        xy = frozen_xy.detach().clone().requires_grad_(True)
        output = TorchPointRasterizer(point_chunk=3, gaussian_chunk=2).render_points(
            gaussians,
            camera,
            xy,
            background=torch.tensor([0.13, 0.29, 0.47]),
            sh_degree=2,
        )
        tensors = (output.color, output.alpha, output.depth)
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise RuntimeError(f"continuous output is non-finite for seed {seed}")
        loss = output.color.sum() + output.alpha.sum() + output.depth.sum()
        loss.backward()
        if xy.grad is None or not bool(torch.isfinite(xy.grad).all()):
            raise RuntimeError(f"continuous coordinate gradient is invalid for seed {seed}")
        field_gradient_finite = {}
        for name in FIELD_NAMES:
            gradient = getattr(gaussians, name).grad
            field_gradient_finite[name] = gradient is not None and bool(
                torch.isfinite(gradient).all()
            )
            if not field_gradient_finite[name]:
                raise RuntimeError(f"continuous query has invalid {name} gradient")
        records.append(
            {
                "seed": seed,
                "xy_hash": tensor_hash(xy),
                "xy_gradient_hash": tensor_hash(xy.grad),
                "maximum_xy_gradient": float(xy.grad.abs().max()),
                "field_gradients_finite": field_gradient_finite,
            }
        )
    return {"coordinates": frozen_xy.tolist(), "seeds": records}


def official_teacher() -> tuple[GaussianObservationField, torch.Tensor]:
    """Literal preregistered estimator fixture; call only after the attempt marker."""
    teacher = GaussianObservationField(
        width=5,
        height=4,
        fit_window=(1, 1, 3, 2),
        means=torch.tensor([[2.5, 1.5], [3.2, 2.1]], dtype=torch.float64),
        log_scales=torch.log(torch.tensor([[0.8, 1.1], [1.2, 0.7]], dtype=torch.float64)),
        rotations=torch.tensor([0.0, 0.3], dtype=torch.float64),
        colors=torch.tensor([[0.2, 0.7, 1.1], [0.9, -0.1, 0.4]], dtype=torch.float64),
        amplitudes=torch.tensor([0.7, 0.4], dtype=torch.float64),
        epsilon=0.2,
        sigma_cutoff=3.0,
        support_fade_alpha=0.4,
    )
    losses = torch.tensor([0.0, 1 / 16, 1 / 4, 9 / 16, 1.0, 25 / 16], dtype=torch.float64)
    return teacher, losses


def sample_flat_indices(samples: Any, field: GaussianObservationField) -> torch.Tensor:
    fit_x, fit_y, fit_width, _ = field.fit_window
    x = (samples.xy[:, 0] - 0.5).long() - fit_x
    y = (samples.xy[:, 1] - 0.5).long() - fit_y
    return y * fit_width + x


def run_discrete_risk_gate() -> dict[str, Any]:
    teacher, losses = official_teacher()
    proposal = GaussianPixelProposal(teacher)
    pixels = teacher.pixel_centers()
    eta = 0.20
    count = pixels.shape[0]
    if count != 6 or proposal.pixel_count != 6:
        raise RuntimeError("official discrete pixel count differs from six")
    exact = losses.mean()
    if not torch.equal(exact, torch.tensor(55 / 96, dtype=torch.float64)):
        raise RuntimeError("literal loss table does not have exact risk 55/96")
    q = eta / count + (1.0 - eta) * proposal.gaussian_probability(pixels)
    target = torch.full_like(q, 1.0 / count)
    importance = target / q
    enumerated = (q * importance * losses).sum()
    require_close(
        enumerated[None],
        exact[None],
        atol=1e-12,
        rtol=1e-12,
        context="enumerated discrete expectation",
    )
    null_probability = 1.0 - q.sum()
    variance = (q * (target * losses / q).square()).sum() - exact.square()
    invariants = {
        "exact_positive": float(exact) > 0.0,
        "loss_std_positive": float(losses.std(unbiased=True)) > 0.0,
        "q_finite_positive": bool(torch.isfinite(q).all() and (q > 0).all()),
        "q_nonuniform": float(q.max() - q.min()) > 0.0,
        "variance_positive": float(variance) > 0.0,
        "null_probability_positive": float(null_probability) > 0.0,
        "importance_bounded": float(importance.max()) <= 1.0 / eta + 1e-12,
    }
    if not all(invariants.values()):
        raise RuntimeError(f"discrete estimator nonvacuity invariant failed: {invariants}")

    seed_records = []
    differences = []
    branch_counts = {"uniform": 0, "gaussian": 0, "gaussian_accepted": 0, "gaussian_rejected": 0}
    per_seed_limit = 6.0 * math.sqrt(float(variance) / 512) + 1e-12
    for seed in range(92000, 92064):
        samples = proposal.sample(
            512,
            uniform_fraction=eta,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )
        indices = sample_flat_indices(samples, teacher)
        values = losses[indices]
        expected_q = q[indices]
        require_close(
            samples.proposal_density,
            expected_q,
            atol=1e-12,
            rtol=1e-12,
            context="sample proposal probability",
        )
        expected_target = samples.active.to(torch.float64) / count
        if not torch.equal(samples.target_density, expected_target):
            raise RuntimeError("sample target probabilities differ from active u")
        expected_importance = torch.where(
            samples.active, expected_target / expected_q, torch.zeros_like(expected_target)
        )
        require_close(
            samples.importance,
            expected_importance,
            atol=1e-12,
            rtol=1e-12,
            context="sample importance",
        )
        estimate = fixed_attempt_mean(values, samples)
        starts = (0, 1, 8, 39)
        sizes = (1, 7, 31, 473)
        micro_sum = torch.zeros((), dtype=torch.float64)
        for start, size in zip(starts, sizes):
            micro_sum = (
                micro_sum
                + (values[start : start + size] * samples.importance[start : start + size]).sum()
            )
        micro_estimate = micro_sum / 512
        require_close(
            micro_estimate[None],
            estimate[None],
            atol=2e-12,
            rtol=2e-12,
            context="microchunk fixed-attempt estimate",
        )
        difference = float(estimate - exact)
        if abs(difference) > per_seed_limit:
            raise RuntimeError(
                f"sampling seed {seed} exceeds six analytic standard errors: {difference}"
            )
        gaussian = samples.proposal_component_ids >= 0
        uniform = ~gaussian
        branch_counts["uniform"] += int(uniform.sum())
        branch_counts["gaussian"] += int(gaussian.sum())
        branch_counts["gaussian_accepted"] += int((gaussian & samples.active).sum())
        branch_counts["gaussian_rejected"] += int((gaussian & ~samples.active).sum())
        differences.append(difference)
        seed_records.append(
            {
                "seed": seed,
                "estimate": float(estimate),
                "difference": difference,
                "micro_estimate": float(micro_estimate),
                "active_count": int(samples.active.sum()),
                "sample_xy_hash": tensor_hash(samples.xy),
                "sample_importance_hash": tensor_hash(samples.importance),
            }
        )
    if any(value <= 0 for value in branch_counts.values()):
        raise RuntimeError(f"not every proposal branch/outcome was observed: {branch_counts}")
    pooled_difference = sum(differences) / len(differences)
    pooled_limit = 3.0 * math.sqrt(float(variance) / (64 * 512)) + 1e-12
    if abs(pooled_difference) > pooled_limit:
        raise RuntimeError(
            f"pooled estimator bias exceeds three analytic standard errors: {pooled_difference}"
        )
    return {
        "risk_measure": "discrete_pixels",
        "pixel_count": count,
        "exact_risk": float(exact),
        "losses": losses.tolist(),
        "proposal_probabilities": q.tolist(),
        "importance": importance.tolist(),
        "null_probability": float(null_probability),
        "analytic_one_attempt_variance": float(variance),
        "invariants": invariants,
        "branch_counts": branch_counts,
        "per_seed_six_se_limit": per_seed_limit,
        "pooled_three_se_limit": pooled_limit,
        "pooled_difference": pooled_difference,
        "seeds": seed_records,
        "proposal_state": {
            "rectangle_lower": proposal.rectangle_lower.tolist(),
            "rectangle_sizes": proposal.rectangle_sizes.tolist(),
            "component_areas": proposal.component_areas.tolist(),
            "component_masses": proposal.component_masses.tolist(),
            "envelope_mass": float(proposal.envelope_mass),
        },
    }


def execute_phase_a() -> dict[str, Any]:
    started = time.perf_counter()
    fixtures = {seed: official_fixture(seed) for seed in OFFICIAL_SEEDS}
    arbitrary = torch.tensor(
        [[0.75, 0.75], [4.125, 3.875], [18.25, 14.25], [2.2, 5.4]],
        dtype=torch.float32,
    )
    fixture_records = {}
    for seed, (gaussians, camera) in fixtures.items():
        fixture_records[str(seed)] = fixture_boundary_audit(
            gaussians, camera, torch.cat([pixel_centers(camera), arbitrary])
        )
    payload = {
        "artifact_type": "point_rasterizer_parity_result",
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "claim_boundary": (
            "CPU selected-pixel renderer equation/gradient parity and an exact discrete "
            "fixed-attempt Gaussian proposal only; no speed, memory, CUDA, quality, training, "
            "density-control, or default claim"
        ),
        "fixtures": fixture_records,
        "forward": run_forward_gates(fixtures),
        "gradients": run_gradient_gates(fixtures),
        "global_compositor": run_global_compositor_gate(*fixtures[91301]),
        "empty_contracts": run_empty_gates(*fixtures[91301]),
        "continuous_coordinates": run_continuous_coordinate_gates(fixtures),
        "discrete_risk": run_discrete_risk_gate(),
        "thresholds": {
            "forward_atol": FORWARD_ATOL,
            "forward_rtol": FORWARD_RTOL,
            "gradient_atol": GRADIENT_ATOL,
            "gradient_rtol": GRADIENT_RTOL,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    assert_finite_tree(payload, "phase_a")
    return payload


def namespace_siblings() -> list[Path]:
    results = ROOT / "benchmarks" / "results"
    siblings: set[Path] = set()
    for pattern in (
        "20260716_point_rasterizer_parity_ATTEMPT*",
        "20260716_point_rasterizer_parity_RESULT*",
    ):
        siblings.update(results.glob(pattern))
    return sorted(siblings)


def claim_attempt(seal: dict[str, Any], environment: dict[str, Any]) -> tuple[dict[str, Any], str]:
    siblings = namespace_siblings()
    if siblings or ATTEMPT.exists() or RESULT.exists() or AUDIT.exists():
        raise FileExistsError(
            "once-only experiment namespace is already claimed: "
            f"{[str(path) for path in (*siblings, AUDIT) if path.exists()]}"
        )
    payload = {
        "artifact_type": "point_rasterizer_parity_once_only_attempt",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "seal_file_sha256": seal["seal_file_sha256"],
        "seal_payload_sha256": seal["seal_payload_sha256"],
        "source_aggregate": seal["source_aggregate"],
        "command": [sys.executable, *sys.argv],
        "environment_fingerprint": environment_fingerprint(environment),
        "result_path": str(RESULT.relative_to(ROOT)),
    }
    digest = _exclusive_json(ATTEMPT, payload)
    return payload, digest


def run_once() -> int:
    environment = environment_metadata()
    assert_official_environment(environment)
    seal = load_and_verify_seal()
    if environment_fingerprint(environment) != environment_fingerprint(seal["environment"]):
        raise RuntimeError("run environment differs from the sealed environment")
    attempt, attempt_digest = claim_attempt(seal, environment)
    try:
        phase_a = execute_phase_a()
        verify_source_hashes(seal["source_hashes"], seal["source_aggregate"])
        verify_git_binding(seal)
        if sha256_file(ATTEMPT) != attempt_digest:
            raise RuntimeError("attempt marker changed during Phase A")
        payload = {
            **phase_a,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "seal_file_sha256": seal["seal_file_sha256"],
            "seal_payload_sha256": seal["seal_payload_sha256"],
            "source_aggregate": seal["source_aggregate"],
            "attempt_sha256": attempt_digest,
            "attempt": attempt,
            "environment": environment,
            "git": git_metadata(),
            "command": [sys.executable, *sys.argv],
        }
    except Exception as error:
        marker_digest = sha256_file(ATTEMPT)
        marker_intact = marker_digest == attempt_digest
        payload = {
            "artifact_type": "point_rasterizer_parity_result",
            "status": "FAIL",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "traceback": traceback.format_exc(),
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "seal_file_sha256": seal["seal_file_sha256"],
            "seal_payload_sha256": seal["seal_payload_sha256"],
            "source_aggregate": seal["source_aggregate"],
            "attempt_sha256": marker_digest,
            "expected_attempt_sha256": attempt_digest,
            "attempt_marker_intact": marker_intact,
            "environment": environment,
            "command": [sys.executable, *sys.argv],
        }
        _exclusive_json(RESULT, payload)
        raise
    assert_finite_tree(payload, "official result")
    digest = _exclusive_json(RESULT, payload)
    print(f"saved {RESULT.relative_to(ROOT)} (sha256={digest})", flush=True)
    return 0


def load_calibrated_camera_only(
    calibration_path: Path,
    *,
    camera_id: str = "C0001",
    downscale: int = 16,
) -> Camera:
    """Load one ideal pinhole camera from calibration JSON without touching image files."""
    if downscale < 1:
        raise ValueError("downscale must be positive")
    calibration = strict_json_load(calibration_path)
    record = next(
        (item for item in calibration["cameras"] if item["camera_id"].upper() == camera_id),
        None,
    )
    if record is None:
        raise ValueError(f"calibration has no camera {camera_id}")
    intrinsics = record["intrinsics"]
    calibration_width, calibration_height = map(int, intrinsics["resolution"])
    width = max(1, calibration_width // downscale)
    height = max(1, calibration_height // downscale)
    sx = width / calibration_width
    sy = height / calibration_height
    matrix = intrinsics["camera_matrix"]
    fx = float(matrix[0]) * sx
    fy = float(matrix[4]) * sy
    cx = (float(matrix[2]) + 0.5) * sx
    cy = (float(matrix[5]) + 0.5) * sy
    view = torch.tensor(record["extrinsics"]["view_matrix"], dtype=torch.float32).view(4, 4)
    return Camera(fx, fy, cx, cy, width, height, view[:3, :3], view[:3, 3])


def validate_audit_text(
    text: str,
    *,
    preregistration_sha256: str,
    seal_sha256: str,
    result_sha256: str,
) -> str:
    match = re.search(
        r"(?:verdict\s*:|#{1,6}\s*verdict\s*\n)\s*[*_`]*(PASS|QUALIFIED)",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise RuntimeError("independent audit does not authorize calibrated interaction")
    bindings = {
        "preregistration": preregistration_sha256,
        "seal": seal_sha256,
        "result": result_sha256,
    }
    missing = [name for name, digest in bindings.items() if digest not in text]
    if missing:
        raise RuntimeError(f"independent audit is missing current artifact bindings: {missing}")
    return match.group(1).upper()


def audit_verdict(seal: dict[str, Any]) -> str:
    return validate_audit_text(
        AUDIT.read_text(encoding="utf-8"),
        preregistration_sha256=PREREGISTRATION_SHA256,
        seal_sha256=seal["seal_file_sha256"],
        result_sha256=sha256_file(RESULT),
    )


def calibrated_sample_indices(width: int, height: int) -> tuple[torch.Tensor, str, int | None]:
    pixel_count = width * height
    if pixel_count <= 0:
        raise ValueError("calibrated image dimensions must be positive")
    if pixel_count <= 4096:
        return torch.arange(pixel_count), "all_pixel_centers", None
    generator = torch.Generator(device="cpu").manual_seed(93001)
    return (
        torch.randint(pixel_count, (4096,), generator=generator),
        "uniform_with_replacement",
        93001,
    )


def run_calibrated() -> int:
    environment = environment_metadata()
    assert_official_environment(environment)
    seal = load_and_verify_seal()
    if environment_fingerprint(environment) != environment_fingerprint(seal["environment"]):
        raise RuntimeError("calibrated environment differs from the sealed environment")
    result = strict_json_load(RESULT)
    if result.get("status") != "PASS" or result.get("seal_file_sha256") != seal["seal_file_sha256"]:
        raise RuntimeError("calibrated interaction requires the sealed PASS result")
    verdict = audit_verdict(seal)
    if CALIBRATED_RESULT.exists():
        raise FileExistsError(f"refusing to overwrite {CALIBRATED_RESULT}")
    if not CALIBRATION.is_file() or not CALIBRATED_PLY.is_file():
        raise FileNotFoundError("calibration JSON or calibrated PLY is missing")

    camera = load_calibrated_camera_only(CALIBRATION, camera_id="C0001", downscale=16)
    gaussians = Gaussians3D.load_ply(CALIBRATED_PLY)
    flat, sample_mode, sample_seed = calibrated_sample_indices(camera.width, camera.height)
    xy = torch.stack(
        [
            torch.remainder(flat, camera.width).to(torch.float32) + 0.5,
            torch.div(flat, camera.width, rounding_mode="floor").to(torch.float32) + 0.5,
        ],
        dim=-1,
    )
    started = time.perf_counter()
    dense = TorchRasterizer(row_chunk=32).render(gaussians, camera)
    point = TorchPointRasterizer(point_chunk=512, gaussian_chunk=256).render_points(
        gaussians, camera, xy
    )
    if not torch.equal(point.visible, dense.visible):
        raise RuntimeError("calibrated point/dense visible order differs")
    errors = {
        "color": require_close(
            point.color,
            dense.color.reshape(-1, 3)[flat],
            atol=FORWARD_ATOL,
            rtol=FORWARD_RTOL,
            context="calibrated color",
        ),
        "alpha": require_close(
            point.alpha,
            dense.alpha.reshape(-1)[flat],
            atol=FORWARD_ATOL,
            rtol=FORWARD_RTOL,
            context="calibrated alpha",
        ),
        "depth": require_close(
            point.depth,
            dense.depth.reshape(-1)[flat],
            atol=FORWARD_ATOL,
            rtol=FORWARD_RTOL,
            context="calibrated depth",
        ),
    }
    verify_source_hashes(seal["source_hashes"], seal["source_aggregate"])
    verify_git_binding(seal)
    payload = {
        "artifact_type": "point_rasterizer_calibrated_interaction",
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "audit_verdict": verdict,
        "audit_path": str(AUDIT.relative_to(ROOT)),
        "audit_sha256": sha256_file(AUDIT),
        "seal_file_sha256": seal["seal_file_sha256"],
        "seal_payload_sha256": seal["seal_payload_sha256"],
        "phase_a_result_sha256": sha256_file(RESULT),
        "calibration_path": str(CALIBRATION.relative_to(ROOT)),
        "calibration_sha256": sha256_file(CALIBRATION),
        "ply_path": str(CALIBRATED_PLY.relative_to(ROOT)),
        "ply_sha256": sha256_file(CALIBRATED_PLY),
        "camera_id": "C0001",
        "downscale": 16,
        "camera": {
            "width": camera.width,
            "height": camera.height,
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy,
            "R_sha256": tensor_hash(camera.R),
            "t_sha256": tensor_hash(camera.t),
        },
        "gaussian_count": gaussians.n,
        "visible_count": int(point.visible.numel()) if point.visible is not None else 0,
        "sample_count": xy.shape[0],
        "sample_mode": sample_mode,
        "sample_seed": sample_seed,
        "sample_flat_indices_sha256": tensor_hash(flat),
        "errors": errors,
        "source_rgb_decoded": False,
        "source_masks_decoded": False,
        "quality_metrics_computed": False,
        "wall_seconds": time.perf_counter() - started,
        "environment": environment,
        "command": [sys.executable, *sys.argv],
    }
    assert_finite_tree(payload, "calibrated result")
    digest = _exclusive_json(CALIBRATED_RESULT, payload)
    print(f"saved {CALIBRATED_RESULT.relative_to(ROOT)} (sha256={digest})", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("seal")
    subparsers.add_parser("run")
    subparsers.add_parser("calibrated")
    return parser.parse_args()


def assert_exact_command(command: str) -> None:
    expected_python = (ROOT / ".venv/bin/python").resolve()
    if Path(sys.executable).resolve() != expected_python or sys.argv[1:] != [command]:
        raise RuntimeError(
            f"{command} must use the preregistered .venv/bin/python command without extra args"
        )


def main() -> int:
    args = parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    assert_exact_command(args.command)
    if args.command == "seal":
        payload = create_seal()
        digest = _exclusive_json(SEAL, payload)
        print(f"saved {SEAL.relative_to(ROOT)} (sha256={digest})", flush=True)
        return 0
    if args.command == "run":
        return run_once()
    if args.command == "calibrated":
        return run_calibrated()
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
