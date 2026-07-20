#!/usr/bin/env python3
"""Isolated CUDA worker for the GaussianImage++ provider parity experiment.

This worker only decodes a Gaussian checkpoint or a compact NPZ field and renders it.
It never opens source RGB, imports a trainer, optimizes parameters, or fits an image.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

EXPECTED_TORCH = "2.9.0+cu128"
EXPECTED_TORCH_CUDA = "12.8"
BLOCK_SIZE = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def git_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_status(repo: Path) -> list[str]:
    return subprocess.run(
        ["git", "status", "--short"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def verify_process(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    repo = args.repo.resolve()
    if git_commit(repo) != args.expected_commit or git_status(repo):
        raise RuntimeError("GaussianImage++ checkout differs from the frozen clean commit")
    if Path(sys.prefix).resolve() != args.expected_prefix.resolve():
        raise RuntimeError(
            f"unexpected Python prefix {sys.prefix}; expected {args.expected_prefix}"
        )
    if torch.__version__ != EXPECTED_TORCH or torch.version.cuda != EXPECTED_TORCH_CUDA:
        raise RuntimeError(
            f"unexpected torch ABI {torch.__version__}/{torch.version.cuda}; "
            f"expected {EXPECTED_TORCH}/{EXPECTED_TORCH_CUDA}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the isolated worker")

    preload_parts = [value for value in os.environ.get("LD_PRELOAD", "").split(":") if value]
    if len(preload_parts) != 1:
        raise RuntimeError("worker requires exactly one LD_PRELOAD entry")
    preload = Path(preload_parts[0])
    if not preload.is_file() or sha256_file(preload) != args.expected_preload_sha256:
        raise RuntimeError("system libstdc++ preload hash differs from the seal")

    # Import the binary explicitly.  A failure here is fatal and cannot enter gsplat's
    # write-producing JIT fallback.
    import gsplat
    import gsplat.csrc as csrc

    expected_module_root = (repo / "gsplat/gsplat").resolve()
    module_path = Path(gsplat.__file__).resolve()
    csrc_path = Path(csrc.__file__).resolve()
    if module_path.parent != expected_module_root:
        raise RuntimeError(f"foreign gsplat module imported from {module_path}")
    if csrc_path != expected_module_root / "csrc.so":
        raise RuntimeError(f"foreign gsplat extension imported from {csrc_path}")
    if sha256_file(csrc_path) != args.expected_csrc_sha256:
        raise RuntimeError("GaussianImage++ csrc hash differs from the seal")
    metadata = {
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "repo": str(repo),
        "repo_commit": args.expected_commit,
        "gsplat_module": str(module_path),
        "csrc": str(csrc_path),
        "csrc_sha256": args.expected_csrc_sha256,
        "preload": str(preload.resolve()),
        "preload_sha256": args.expected_preload_sha256,
    }
    return csrc, metadata


def load_field_npz(payload: bytes) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {
            name: torch.from_numpy(archive[name].copy()).to(dtype=torch.float32)
            for name in ("means", "covariances", "colors", "opacities", "background")
        }
    return arrays, metadata


def load_checkpoint(
    payload: bytes, *, height: int, width: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    state = checkpoint["gs"]
    arrays = {
        "means": state["_xyz"].to(dtype=torch.float32),
        "covariances": (state["_cov2d"] + checkpoint["slv_bound"]).to(dtype=torch.float32),
        "colors": state["_features_dc"].to(dtype=torch.float32),
        "opacities": state["_opacity"].to(dtype=torch.float32),
        "background": state["background"].to(dtype=torch.float32),
    }
    metadata = {
        "height": height,
        "width": width,
        "block_size": BLOCK_SIZE,
        "clip_coe": 3.0,
        "radius_clip": 1.0,
        "checkpoint_num_gaussians": int(checkpoint["num_gs"]),
    }
    if arrays["means"].shape[0] != metadata["checkpoint_num_gaussians"]:
        raise RuntimeError("checkpoint num_gs disagrees with state tensors")
    return arrays, metadata


def validate_arrays(arrays: dict[str, torch.Tensor], metadata: dict[str, Any]) -> None:
    n = arrays["means"].shape[0]
    expected = {
        "means": (n, 2),
        "covariances": (n, 3),
        "colors": (n, 3),
        "opacities": (n, 1),
        "background": (3,),
    }
    for name, shape in expected.items():
        value = arrays[name]
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise RuntimeError(f"invalid worker field {name}: {tuple(value.shape)}")
    if int(metadata["block_size"]) != BLOCK_SIZE:
        raise RuntimeError("the frozen native extension requires 16x16 blocks")
    if int(metadata["height"]) < 1 or int(metadata["width"]) < 1:
        raise RuntimeError("invalid render dimensions")


def render_native(
    arrays: dict[str, torch.Tensor], metadata: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from gsplat.project_gaussians_2d_covariance import project_gaussians_2d_covariance
    from gsplat.rasterize_sum_plus import rasterize_gaussians_plus
    from gsplat.utils import bin_and_sort_gaussians, compute_cumulative_intersects

    device = torch.device("cuda:0")
    values = {name: value.to(device=device).contiguous() for name, value in arrays.items()}
    height = int(metadata["height"])
    width = int(metadata["width"])
    block_size = int(metadata["block_size"])
    tile_bounds = (
        (width + block_size - 1) // block_size,
        (height + block_size - 1) // block_size,
        1,
    )
    tile_count = tile_bounds[0] * tile_bounds[1]
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    xys, depths, radii, conics, hits = project_gaussians_2d_covariance(
        values["means"],
        values["covariances"],
        height,
        width,
        tile_bounds,
        clip_coe=float(metadata["clip_coe"]),
        radius_clip=float(metadata["radius_clip"]),
    )
    num_intersects, cumulative_hits = compute_cumulative_intersects(hits)
    if num_intersects > 0:
        _, _, _, gaussian_ids_sorted, native_tile_bins = bin_and_sort_gaussians(
            values["means"].shape[0],
            num_intersects,
            xys,
            depths,
            radii,
            cumulative_hits,
            tile_bounds,
            float(metadata["radius_clip"]),
        )
        if native_tile_bins.shape[0] < tile_count:
            raise RuntimeError("native tile-bin allocation is smaller than the image tile grid")
        tile_bins = native_tile_bins[:tile_count]
    else:
        gaussian_ids_sorted = torch.empty(0, dtype=torch.int32, device=device)
        tile_bins = torch.zeros((tile_count, 2), dtype=torch.int32, device=device)
    raw = rasterize_gaussians_plus(
        xys,
        depths,
        radii,
        conics,
        hits,
        values["colors"],
        values["opacities"],
        height,
        width,
        block_size,
        block_size,
        background=values["background"],
        radius_clip=float(metadata["radius_clip"]),
    )
    clamped = raw.clamp(0.0, 1.0)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    output = {
        "means": values["means"].cpu().numpy(),
        "covariances": values["covariances"].cpu().numpy(),
        "colors": values["colors"].cpu().numpy(),
        "opacities": values["opacities"].cpu().numpy(),
        "background": values["background"].cpu().numpy(),
        "xys": xys.cpu().numpy(),
        "conics": conics.cpu().numpy(),
        "radii": radii.cpu().numpy(),
        "hits": hits.cpu().numpy(),
        "gaussian_ids_sorted": gaussian_ids_sorted.cpu().numpy(),
        "tile_bins": tile_bins.cpu().numpy(),
        "raw": raw.cpu().numpy(),
        "clamped": clamped.cpu().numpy(),
    }
    run_metadata = {
        **metadata,
        "n_gaussians": int(values["means"].shape[0]),
        "num_intersects": int(num_intersects),
        "tile_count": tile_count,
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    return output, run_metadata


def save_output(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite worker output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary worker output already exists: {temporary}")
    with temporary.open("xb") as stream:
        np.savez_compressed(
            stream,
            **arrays,
            metadata=np.asarray(canonical_json(metadata)),
        )
    os.replace(temporary, path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("field", "checkpoint"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-prefix", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-csrc-sha256", required=True)
    parser.add_argument("--expected-preload-sha256", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    _, process_metadata = verify_process(args)
    input_payload = args.input.read_bytes()
    input_sha256 = sha256_bytes(input_payload)
    if input_sha256 != args.expected_input_sha256:
        raise RuntimeError(
            "worker input differs from its expected SHA-256 before decode: "
            f"expected={args.expected_input_sha256}, actual={input_sha256}"
        )
    if args.kind == "field":
        if args.height is not None or args.width is not None:
            raise ValueError("field dimensions must come from the sealed NPZ")
        arrays, input_metadata = load_field_npz(input_payload)
    else:
        if args.height is None or args.width is None:
            raise ValueError("checkpoint mode requires height and width")
        arrays, input_metadata = load_checkpoint(
            input_payload,
            height=args.height,
            width=args.width,
        )
    validate_arrays(arrays, input_metadata)
    output, run_metadata = render_native(arrays, input_metadata)
    metadata = {
        **process_metadata,
        **run_metadata,
        "kind": args.kind,
        "input": str(args.input.resolve()),
        "input_sha256": input_sha256,
    }
    save_output(args.output, output, metadata)
    print(json.dumps({"output": str(args.output), "metadata": metadata}, allow_nan=False))


if __name__ == "__main__":
    main()
