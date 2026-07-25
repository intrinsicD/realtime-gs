#!/usr/bin/env python3
"""Stage 0: what is the remaining held-out error actually made of?

Before spending a stage on "fill internal holes", measure whether internal holes are the
dominant residual at all.  This decomposes the held-out reconstruction error of a saved model
into four disjoint image regions and reports how much of the total error lives in each:

``interior_holes``
    inside the foreground mask, eroded by the boundary width, where rendered alpha is below
    threshold — geometry is missing.
``interior_covered``
    inside the eroded mask with adequate alpha — geometry is present, so any error here is
    **appearance**, not coverage.
``boundary``
    the band straddling the silhouette — where a cover rule is expected to be wrong because the
    surface curves away from the camera.
``exterior``
    outside the dilated mask — leakage.

The point of the split is that the three failure modes need different tools.  Holes need
birth/split; interior appearance error needs SH, colour, and fine optimization; boundary error
needs a silhouette-aware rule.  Aiming the wrong tool at the dominant term wastes a stage.

Two habits are built in because single-threshold summaries misled this project before: the hole
test is reported over an **alpha-threshold sweep** rather than only at 0.5, and interior holes
get a connected-component analysis, because "many sub-pixel pinholes" and "a few large missing
patches" look identical in a fraction and call for different interventions.

Post-hoc diagnostic over saved artifacts.  No optimization, no selection, no default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.compact_views import CompactDataset
from rtgs.data.scene import SceneData
from rtgs.render.base import get_rasterizer

try:
    from benchmarks.scene_builder import build_scene_at
except ModuleNotFoundError:  # direct invocation
    from scene_builder import build_scene_at  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAME = "frame_00009"
DEFAULT_RUN = ROOT / "runs/beam_surfel_init_20260724"
ALPHA_THRESHOLDS = (0.1, 0.3, 0.5)
PRIMARY_ALPHA_THRESHOLD = 0.5
BOUNDARY_RADIUS_PX = 2


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation by a (2r+1) square structuring element."""
    if radius <= 0:
        return mask
    padded = mask[None, None].float()
    return F.max_pool2d(padded, kernel_size=2 * radius + 1, stride=1, padding=radius)[0, 0] > 0.5


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return ~_dilate(~mask, radius)


def connected_components(mask: torch.Tensor, max_iterations: int = 512) -> torch.Tensor:
    """Label 4/8-connected components by iterative max-propagation (deterministic, no scipy).

    Returns a label image where background is -1.  Labels are seeded from the flat pixel index,
    so the final label of a component is the largest seed index it contains; only the partition
    matters, not the label values.
    """
    height, width = mask.shape
    labels = torch.where(
        mask,
        torch.arange(height * width, device=mask.device).reshape(height, width),
        torch.full((height, width), -1, device=mask.device, dtype=torch.long),
    )
    for _ in range(max_iterations):
        spread = F.max_pool2d(labels[None, None].float(), 3, stride=1, padding=1)[0, 0].long()
        updated = torch.where(mask, spread, torch.full_like(labels, -1))
        if torch.equal(updated, labels):
            break
        labels = updated
    return labels


def component_sizes(mask: torch.Tensor) -> list[int]:
    if not bool(mask.any()):
        return []
    labels = connected_components(mask)
    present = labels[labels >= 0]
    return torch.bincount(present)[torch.bincount(present) > 0].tolist()


def decompose_view(
    color: torch.Tensor,
    alpha: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    boundary_radius: int,
    alpha_threshold: float,
) -> dict:
    """Disjoint region decomposition of one view's L1 error."""
    hard = mask > 0.5
    interior = _erode(hard, boundary_radius)
    exterior = ~_dilate(hard, boundary_radius)
    boundary = ~interior & ~exterior

    covered = alpha >= alpha_threshold
    holes = interior & ~covered
    interior_covered = interior & covered

    # The trainer's target is the masked image, so compare against the same thing.
    error = (color - target * mask[..., None]).abs().sum(dim=-1)
    total = float(error.sum())

    def region(selector: torch.Tensor) -> dict:
        count = int(selector.sum())
        return {
            "pixels": count,
            "l1_sum": float(error[selector].sum()) if count else 0.0,
            "l1_share_of_total": (float(error[selector].sum()) / total) if total > 0 else 0.0,
            "mean_alpha": float(alpha[selector].mean()) if count else 0.0,
        }

    sizes = component_sizes(holes)
    sizes_sorted = sorted(sizes, reverse=True)
    return {
        "total_l1": total,
        "n_pixels": int(alpha.numel()),
        "interior_pixels": int(interior.sum()),
        "regions": {
            "interior_holes": region(holes),
            "interior_covered": region(interior_covered),
            "boundary": region(boundary),
            "exterior": region(exterior),
        },
        "hole_fraction_of_interior": (
            float(holes.sum() / interior.sum()) if bool(interior.any()) else 0.0
        ),
        "hole_components": {
            "count": len(sizes_sorted),
            "largest": sizes_sorted[0] if sizes_sorted else 0,
            "median": sizes_sorted[len(sizes_sorted) // 2] if sizes_sorted else 0,
            "n_single_pixel": sum(1 for s in sizes_sorted if s == 1),
            "frac_hole_pixels_in_largest": (
                sizes_sorted[0] / sum(sizes_sorted) if sizes_sorted else 0.0
            ),
        },
    }


def evaluate_model(
    model: Gaussians3D,
    scene: SceneData,
    views: list[int],
    *,
    boundary_radius: int,
    thresholds: tuple[float, ...],
    rasterizer: str = "torch",
) -> dict:
    renderer = get_rasterizer(rasterizer, device=model.means.device)
    by_threshold: dict[str, dict] = {}
    for threshold in thresholds:
        per_view = []
        with torch.no_grad():
            for index in views:
                out = renderer.render(model, scene.cameras[index].to(model.means.device))
                per_view.append(
                    decompose_view(
                        out.color.detach().cpu(),
                        out.alpha.detach().cpu(),
                        scene.images[index],
                        scene.masks[index],
                        boundary_radius=boundary_radius,
                        alpha_threshold=threshold,
                    )
                )
        total = sum(v["total_l1"] for v in per_view)
        aggregate = {
            name: {
                "l1_share_of_total": (
                    sum(v["regions"][name]["l1_sum"] for v in per_view) / total if total else 0.0
                ),
                "pixels": sum(v["regions"][name]["pixels"] for v in per_view),
                "mean_alpha": sum(v["regions"][name]["mean_alpha"] for v in per_view)
                / len(per_view),
            }
            for name in ("interior_holes", "interior_covered", "boundary", "exterior")
        }
        by_threshold[str(threshold)] = {
            "aggregate_l1_share": aggregate,
            "hole_fraction_of_interior": sum(v["hole_fraction_of_interior"] for v in per_view)
            / len(per_view),
            "hole_components": {
                "total_count": sum(v["hole_components"]["count"] for v in per_view),
                "largest": max(v["hole_components"]["largest"] for v in per_view),
                "n_single_pixel": sum(v["hole_components"]["n_single_pixel"] for v in per_view),
                "mean_frac_in_largest": sum(
                    v["hole_components"]["frac_hole_pixels_in_largest"] for v in per_view
                )
                / len(per_view),
            },
            "per_view": per_view,
        }
    return by_threshold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--frame", default=DEFAULT_FRAME)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["adc/ci/gaussians_final.ply", "adc/surfel/gaussians_final.ply"],
        help="paths relative to --run, or absolute .ply paths",
    )
    parser.add_argument("--boundary-radius", type=int, default=BOUNDARY_RADIUS_PX)
    parser.add_argument("--downscale", type=int, default=32)
    parser.add_argument("--rasterizer", default="torch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    dataset = CompactDataset.load(
        ROOT / f"dataset/2025_03_07_stage_with_fabric/{args.frame}/gaussians2d", device="cpu"
    )
    split = build_scene_at(dataset, downscale=args.downscale)
    scene, heldout_local = split.scene, split.heldout_local

    results = {}
    for spec in args.models:
        path = Path(spec)
        if not path.is_absolute():
            path = args.run / spec
        if not path.is_file():
            raise FileNotFoundError(f"model not found: {path}")
        model = Gaussians3D.load_ply(path).to(args.device)
        results[spec] = {
            "path": str(path),
            "n_gaussians": model.n,
            "by_alpha_threshold": evaluate_model(
                model,
                scene,
                heldout_local,
                boundary_radius=args.boundary_radius,
                thresholds=ALPHA_THRESHOLDS,
                rasterizer=args.rasterizer,
            ),
        }
        primary = results[spec]["by_alpha_threshold"][str(PRIMARY_ALPHA_THRESHOLD)]
        share = primary["aggregate_l1_share"]
        print(f"\n=== {spec}  (N={model.n})")
        print(f"  held-out L1 error share by region (alpha threshold {PRIMARY_ALPHA_THRESHOLD}):")
        for name in ("interior_holes", "interior_covered", "boundary", "exterior"):
            print(
                f"    {name:<20} {100 * share[name]['l1_share_of_total']:6.2f}%  "
                f"({share[name]['pixels']} px)"
            )
        print(f"  interior hole fraction: {100 * primary['hole_fraction_of_interior']:.3f}%")
        hc = primary["hole_components"]
        print(
            f"  hole components: {hc['total_count']} total, largest {hc['largest']} px, "
            f"{hc['n_single_pixel']} single-pixel, "
            f"{100 * hc['mean_frac_in_largest']:.1f}% of hole pixels in the largest"
        )

    summary = {
        "schema": "rtgs.residual_decomposition.v1",
        "kind": "post-hoc diagnostic over saved artifacts; selects nothing",
        "frame": args.frame,
        "heldout_views": [scene.view_names[i] for i in heldout_local],
        "boundary_radius_px": args.boundary_radius,
        "downscale": args.downscale,
        "alpha_thresholds": list(ALPHA_THRESHOLDS),
        "primary_alpha_threshold": PRIMARY_ALPHA_THRESHOLD,
        "models": results,
    }
    out = args.out or (args.run / "residual_decomposition.json")
    out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"\n[written] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
