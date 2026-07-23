#!/usr/bin/env python3
"""Post-hoc step-0 alpha bottleneck probe for Beam partition initializations.

This intentionally performs no fitting and changes no geometry.  It renders the saved CI,
partition-area, and full-partition initial PLYs with a frozen global opacity multiplier sweep,
then records alpha threshold curves.  The probe is exploratory because its factors and questions
were chosen after the covariance experiment outcomes were visible.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.compact_views import CompactDataset
from rtgs.optim.trainer import Trainer
from rtgs.render.base import get_rasterizer

try:
    from benchmarks.beam_convergence_dynamics import EVAL_VIEWS, build_scene
except ModuleNotFoundError:  # direct ``python benchmarks/beam_partition_opacity_probe.py``
    from beam_convergence_dynamics import EVAL_VIEWS, build_scene  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d"
DEFAULT_RUN = ROOT / "runs/beam_partition_covariance_20260723"
ARMS = ("ci", "pou-area", "pou-full")
OPACITY_FACTORS = (0.5, 1.0, 2.0, 4.0, 8.0)
ALPHA_THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _with_opacity_factor(base: Gaussians3D, factor: float) -> Gaussians3D:
    return Gaussians3D(
        means=base.means,
        quats=base.quats,
        log_scales=base.log_scales,
        opacity=(base.opacity * factor).clamp(max=0.99),
        sh=base.sh,
    )


def _precision_recall(
    alphas: list[torch.Tensor],
    masks: list[torch.Tensor],
    threshold: float,
) -> dict[str, float]:
    recalls = []
    precisions = []
    for alpha, mask in zip(alphas, masks, strict=True):
        foreground = mask > 0.5
        predicted = alpha > threshold
        intersection = (foreground & predicted).sum()
        recalls.append(float(intersection / foreground.sum().clamp_min(1)))
        precisions.append(float(intersection / predicted.sum().clamp_min(1)))
    return {
        "recall": sum(recalls) / len(recalls),
        "precision": sum(precisions) / len(precisions),
    }


def build(run: Path) -> dict:
    dataset = CompactDataset.load(DATASET, device="cpu")
    scene = build_scene(dataset)
    renderer = get_rasterizer("torch", device=torch.device("cpu"))
    evaluation_views = list(EVAL_VIEWS)
    evaluation_masks = [scene.masks[index] for index in evaluation_views]
    arms = {}
    for arm in ARMS:
        path = run / arm / "gaussians_init.ply"
        base = Gaussians3D.load_ply(path)
        factors = {}
        baseline_alphas = None
        for factor in OPACITY_FACTORS:
            model = _with_opacity_factor(base, factor)
            metrics = Trainer.evaluate_metrics(
                scene,
                model,
                renderer,
                evaluation_views,
            )
            with torch.no_grad():
                alphas = [
                    renderer.render(model, scene.cameras[index]).alpha for index in evaluation_views
                ]
            factor_key = f"{factor:g}"
            factors[factor_key] = {
                "opacity_factor": factor,
                "effective_uniform_opacity": float(model.opacity[0]),
                "psnr_fg": metrics["psnr_fg"],
                "alpha_iou_at_0_5": metrics["alpha_iou"],
                "alpha_inside": metrics["alpha_inside"],
                "alpha_outside": metrics["alpha_outside"],
                **_precision_recall(alphas, evaluation_masks, 0.5),
            }
            if factor == 1.0:
                baseline_alphas = alphas
        assert baseline_alphas is not None
        threshold_curve = {
            f"{threshold:g}": {
                "threshold": threshold,
                **_precision_recall(
                    baseline_alphas,
                    evaluation_masks,
                    threshold,
                ),
            }
            for threshold in ALPHA_THRESHOLDS
        }
        arms[arm] = {
            "initial_ply": str(path.relative_to(ROOT)),
            "initial_ply_sha256": _sha256(path),
            "n_gaussians": base.n,
            "uniform_base_opacity": float(base.opacity[0]),
            "opacity_factors": factors,
            "baseline_threshold_curve": threshold_curve,
        }
    return {
        "schema": "rtgs.beam_partition_opacity_probe.v1",
        "status": "complete_exploratory_posthoc",
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "source_sha256": _sha256(Path(__file__)),
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_manifest_sha256": _sha256(DATASET / "manifest.json"),
        "base_run": str(run.relative_to(ROOT)),
        "base_summary_sha256": _sha256(run / "summary.json"),
        "evaluation_local_views": evaluation_views,
        "all_evaluation_views_were_fitted": True,
        "rasterizer": "torch",
        "device": "cpu",
        "no_optimization": True,
        "geometry_and_appearance_frozen": True,
        "alpha_iou_threshold": 0.5,
        "opacity_factors": list(OPACITY_FACTORS),
        "alpha_thresholds": list(ALPHA_THRESHOLDS),
        "arms": arms,
        "scope_warning": (
            "Post-hoc diagnostic on fitted views; it can localize a mechanism but cannot select "
            "an opacity rule, support a default change, or establish held-out utility."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    out = args.out or run / "opacity_probe.json"
    result = build(run)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
