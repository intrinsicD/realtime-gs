#!/usr/bin/env python3
"""Does the CPU screen's dilation mechanism exist under gsplat? Measure it, do not assume.

The CPU reference renders ``Sigma_2D = J Sigma_3D J^T + 0.3 I`` px², a 0.5477 px low-pass floor.
That floor is why an under-sized primitive supplies only ~12% of its own rendered footprint and
why its scale gradient is an order of magnitude weaker than a correctly sized one's.  gsplat
applies its own antialiasing filter, and the entire mechanism identified on CPU may or may not
transfer.  This probe measures the floor empirically for whichever backend is available.

Method: one isotropic Gaussian at a fixed depth, world scale swept so the *intrinsic* projected
sigma spans well below to well above one pixel.  For a single primitive the rendered alpha is
exactly ``o * exp(-r^2 / 2 sigma^2)``, so the alpha-weighted second moment gives the rendered
sigma in closed form: ``sigma_rendered = sqrt(M2 / 2)`` with ``M2 = sum(r^2 alpha) / sum(alpha)``.

Reported per backend: the rendered-versus-intrinsic curve, the fitted floor
(``sigma_rendered^2 - sigma_intrinsic^2`` where the intrinsic term is negligible), and the
suppression factor ``d sigma_rendered / d log sigma_intrinsic`` relative to ``sigma_rendered`` —
the quantity that decides how much of the scale gradient survives.

Measurement only.  Selects nothing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.render.base import get_rasterizer

DEPTH = 3.0
IMAGE = 256
FOCAL = 600.0
OPACITY = 0.5
# Intrinsic projected sigmas to probe, in pixels; spans far below to far above the CPU floor.
TARGET_SIGMA_PX = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)


def probe_camera(device: torch.device) -> Camera:
    return Camera(
        fx=FOCAL,
        fy=FOCAL,
        cx=IMAGE / 2.0,
        cy=IMAGE / 2.0,
        width=IMAGE,
        height=IMAGE,
        R=torch.eye(3, device=device),
        t=torch.zeros(3, device=device),
    )


def single_gaussian(world_sigma: float, device: torch.device, dtype=torch.float32) -> Gaussians3D:
    """One isotropic Gaussian on the optical axis at DEPTH, white, fixed opacity."""
    means = torch.tensor([[0.0, 0.0, DEPTH]], device=device, dtype=dtype)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
    log_scales = torch.full((1, 3), math.log(world_sigma), device=device, dtype=dtype)
    sh = torch.zeros(1, 1, 3, device=device, dtype=dtype)
    from rtgs.core.sh import rgb_to_sh

    sh[:, 0] = rgb_to_sh(torch.ones(1, 3, device=device, dtype=dtype))
    return Gaussians3D(
        means=means,
        quats=quats,
        log_scales=log_scales,
        opacity=torch.full((1,), OPACITY, device=device, dtype=dtype),
        sh=sh,
    )


def rendered_sigma_px(alpha: torch.Tensor) -> float:
    """Alpha-weighted second moment of a single rendered primitive -> sigma in pixels."""
    height, width = alpha.shape
    ys = torch.arange(height, device=alpha.device, dtype=torch.float64) + 0.5
    xs = torch.arange(width, device=alpha.device, dtype=torch.float64) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    weight = alpha.double()
    total = weight.sum()
    if float(total) <= 0:
        return float("nan")
    center_x = (weight * grid_x).sum() / total
    center_y = (weight * grid_y).sum() / total
    radius_sq = (grid_x - center_x).square() + (grid_y - center_y).square()
    second_moment = (weight * radius_sq).sum() / total
    return float((second_moment / 2.0).sqrt())


def probe_backend(name: str, device: torch.device) -> dict:
    renderer = get_rasterizer(name, device=device)
    camera = probe_camera(device)
    rows = []
    for target_px in TARGET_SIGMA_PX:
        # Thin-lens scaling: sigma_px = sigma_world * focal / depth.
        world_sigma = target_px * DEPTH / FOCAL
        model = single_gaussian(world_sigma, device)
        with torch.no_grad():
            out = renderer.render(model, camera)
        rendered = rendered_sigma_px(out.alpha.detach().float().cpu())
        rows.append(
            {
                "intrinsic_sigma_px": target_px,
                "world_sigma": world_sigma,
                "rendered_sigma_px": rendered,
                "implied_floor_variance_px2": (
                    rendered**2 - target_px**2 if rendered == rendered else None
                ),
                "peak_alpha": float(out.alpha.max()),
            }
        )
    # Suppression: d sigma_rendered / d log sigma_intrinsic, normalized by sigma_rendered.
    for index, row in enumerate(rows):
        if 0 < index < len(rows) - 1:
            lo, hi = rows[index - 1], rows[index + 1]
            d_log = math.log(hi["intrinsic_sigma_px"]) - math.log(lo["intrinsic_sigma_px"])
            d_sigma = hi["rendered_sigma_px"] - lo["rendered_sigma_px"]
            row["measured_suppression"] = (d_sigma / d_log) / row["rendered_sigma_px"]
        else:
            row["measured_suppression"] = None
        # Analytic prediction if the backend is an additive-variance floor.
        floor = row["implied_floor_variance_px2"]
        row["analytic_suppression_if_additive_floor"] = (
            row["intrinsic_sigma_px"] ** 2 / (row["intrinsic_sigma_px"] ** 2 + floor)
            if floor is not None and floor > 0
            else None
        )
    floors = [
        r["implied_floor_variance_px2"]
        for r in rows
        if r["implied_floor_variance_px2"] is not None and r["intrinsic_sigma_px"] <= 0.3
    ]
    return {
        "backend": name,
        "device": str(device),
        "rows": rows,
        "floor_variance_px2_estimate": sum(floors) / len(floors) if floors else None,
        "floor_sigma_px_estimate": (
            (sum(floors) / len(floors)) ** 0.5 if floors and sum(floors) > 0 else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/gpu_dilation_probe/summary.json"))
    parser.add_argument(
        "--backends", nargs="+", default=None, help="default: torch, and gsplat if available"
    )
    args = parser.parse_args()

    backends = args.backends
    if backends is None:
        backends = ["torch"]
        if torch.cuda.is_available():
            backends.append("gsplat")

    results = {}
    for name in backends:
        device = torch.device("cuda" if name == "gsplat" else "cpu")
        try:
            results[name] = probe_backend(name, device)
        except Exception as error:  # noqa: BLE001 - a missing backend is a reportable outcome
            results[name] = {"backend": name, "error": f"{type(error).__name__}: {error}"}
            print(f"[{name}] unavailable: {error}", flush=True)
            continue
        summary = results[name]
        print(f"\n=== {name} ({summary['device']})")
        floor = summary["floor_sigma_px_estimate"]
        print(
            f"  estimated screen-space floor: {floor:.4f} px"
            if floor
            else "  estimated floor: not resolvable"
        )
        print(f"  {'intrinsic px':>13} {'rendered px':>12} {'suppression':>12}")
        for row in summary["rows"]:
            suppression = row["measured_suppression"]
            print(
                f"  {row['intrinsic_sigma_px']:>13.3f} {row['rendered_sigma_px']:>12.4f} "
                f"{suppression if suppression is not None else float('nan'):>12.4f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "schema": "rtgs.gpu_dilation_probe.v1",
                "kind": "measurement; selects nothing",
                "geometry": {
                    "depth": DEPTH,
                    "focal": FOCAL,
                    "image": IMAGE,
                    "opacity": OPACITY,
                },
                "cpu_reference_floor_variance_px2": 0.3,
                "backends": results,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
