#!/usr/bin/env python3
"""Where does the scale gradient go, and why does an under-sized primitive not grow?

The birth-attribution diagnostic showed the control's initial primitives sit below the split
boundary and are cloned in place rather than split.  The obvious next question is why plain
gradient descent does not simply grow them until they cover the surface.  This measures the
gradient the optimizer actually receives, rather than inferring it from outcomes.

Two candidate mechanisms, both directly testable:

1. **Dilation absorption.** The reference rasterizer renders ``Sigma_2D = J Sigma_3D J^T + 0.3 I``
   (px^2), a 0.548 px low-pass floor.  What the loss can see is
   ``sigma_2D = sqrt(sigma_own^2 + 0.3)``, so ``d sigma_2D / d log sigma_own = sigma_own^2 /
   sigma_2D``: the *relative* sensitivity of the rendered footprint to a primitive's own scale is
   ``sigma_own^2 / (sigma_own^2 + 0.3)``.  Far below the floor that factor collapses toward zero
   and the rendered blob is the dilation kernel, nearly independent of the primitive's own scale.
   Position is not suppressed this way — a dilated blob still translates at full strength when its
   mean moves — so the mechanism predicts a *specific asymmetry*, not uniform weakness.

2. **Sign inconsistency across views.** Adam normalizes magnitude, so a small but consistently
   signed gradient still yields a full learning-rate step.  A gradient that flips sign between
   training views does not: ``exp_avg`` cancels while ``exp_avg_sq`` does not.  Per-view
   ``mean / RMS`` over the view set is the quantity that decides whether Adam can move a
   parameter at all, and it is measured here per Gaussian.

Post-hoc diagnostic on the frozen initializations.  No optimization, no selection, no default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.metrics import ssim
from rtgs.data.compact_views import CompactDataset
from rtgs.data.scene import SceneData
from rtgs.render.projection import EWA_DILATION, project_covariances_ewa
from rtgs.render.torch_ref import TorchRasterizer

try:
    from benchmarks.beam_surfel_init import build_initializations, build_scene
except ModuleNotFoundError:  # direct invocation
    from beam_surfel_init import build_initializations, build_scene  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
FRAME = "frame_00009"
DEFAULT_OUT = ROOT / "runs/beam_surfel_scale_gradient_20260724"
ARMS = ("ci", "ci-op", "cover-iso", "cover-iso-op", "surfel")
SSIM_LAMBDA = 0.2
MASK_ALPHA_LAMBDA = 0.05
OUTSIDE_ALPHA_LAMBDA = 0.01


def describe(values: torch.Tensor) -> dict:
    values = values.detach().double()
    return {
        "p10": float(values.quantile(0.10)),
        "median": float(values.median()),
        "p90": float(values.quantile(0.90)),
        "mean": float(values.mean()),
    }


def projected_sigma_px(model: Gaussians3D, scene: SceneData, views: list[int]) -> dict:
    """Intrinsic (undilated) projected sigma in pixels, and the dilation suppression factor."""
    intrinsic, suppression = [], []
    for index in views:
        camera = scene.cameras[index]
        projection = project_covariances_ewa(model.means, model.covariance(), camera, dilation=0.0)
        cov = projection.covariances2d.double()
        # Geometric-mean sigma of the projected ellipse, in pixels.
        determinant = (cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]).clamp_min(1e-30)
        sigma = determinant.sqrt().sqrt()
        visible = projection.depth.double() > 0
        intrinsic.append(sigma[visible])
        # d sigma_2D / d log sigma_own = sigma_own^2 / sigma_2D, relative to sigma_2D itself.
        suppression.append((sigma.square() / (sigma.square() + EWA_DILATION))[visible])
    return {
        "intrinsic_sigma_px": describe(torch.cat(intrinsic)),
        "dilation_suppression_factor": describe(torch.cat(suppression)),
        "frac_below_dilation_floor": float(
            (torch.cat(intrinsic) < EWA_DILATION**0.5).double().mean()
        ),
        "dilation_sigma_px": EWA_DILATION**0.5,
    }


def view_loss(
    params: dict[str, torch.Tensor],
    scene: SceneData,
    index: int,
    renderer: TorchRasterizer,
) -> torch.Tensor:
    """The trainer's masked loss for one view, rebuilt over differentiable leaves."""
    model = Gaussians3D(
        means=params["means"],
        quats=params["quats"],
        log_scales=params["scales"],
        opacity=torch.sigmoid(params["opacities"]),
        sh=params["sh"],
    )
    out = renderer.render(model, scene.cameras[index])
    target = scene.images[index]
    mask = scene.masks[index]
    weights = mask + (1.0 - mask)  # trainer weights are unity under a hard mask
    target_for_loss = target * mask[..., None]
    l1 = ((out.color - target_for_loss).abs() * weights[..., None]).mean()
    l1 = l1 + OUTSIDE_ALPHA_LAMBDA * (out.alpha * (1.0 - mask)).mean()
    alpha_loss = (out.alpha - mask).abs().mean()
    loss = (1 - SSIM_LAMBDA) * l1 + MASK_ALPHA_LAMBDA * alpha_loss
    loss = loss + SSIM_LAMBDA * (1.0 - ssim(out.color, target_for_loss))
    return loss


def measure_arm(model: Gaussians3D, scene: SceneData, views: list[int]) -> dict:
    renderer = TorchRasterizer()
    per_view_scale: list[torch.Tensor] = []
    per_view_mean: list[torch.Tensor] = []
    per_view_opacity: list[torch.Tensor] = []
    for index in views:
        params = {
            "means": model.means.detach().clone().requires_grad_(True),
            "quats": model.quats.detach().clone().requires_grad_(True),
            "scales": model.log_scales.detach().clone().requires_grad_(True),
            "opacities": torch.logit(model.opacity.detach().clamp(1e-6, 1 - 1e-6))
            .clone()
            .requires_grad_(True),
            "sh": model.sh.detach().clone().requires_grad_(True),
        }
        loss = view_loss(params, scene, index, renderer)
        grads = torch.autograd.grad(loss, list(params.values()), allow_unused=True)
        named = dict(zip(params.keys(), grads, strict=True))
        # Isotropic component of the log-scale gradient: the part that grows or shrinks a
        # primitive rather than reshaping it.
        per_view_scale.append(named["scales"].sum(dim=-1).detach())
        per_view_mean.append(named["means"].norm(dim=-1).detach())
        per_view_opacity.append(named["opacities"].detach())

    scale = torch.stack(per_view_scale)  # (V, N)
    means = torch.stack(per_view_mean)
    opacity = torch.stack(per_view_opacity)

    def consistency(values: torch.Tensor) -> torch.Tensor:
        """|mean| / RMS over views: 1.0 = same sign every view, 0 = pure cancellation."""
        rms = values.square().mean(dim=0).sqrt().clamp_min(1e-30)
        return values.mean(dim=0).abs() / rms

    return {
        "n_views": len(views),
        "scale_grad_abs_per_view": describe(scale.abs().mean(dim=0)),
        "scale_grad_consistency": describe(consistency(scale)),
        "scale_grad_signed_mean": describe(scale.mean(dim=0)),
        "frac_scale_grad_points_to_growth": float((scale.mean(dim=0) < 0).double().mean()),
        "mean_grad_abs_per_view": describe(means.mean(dim=0)),
        "opacity_grad_consistency": describe(consistency(opacity)),
        "scale_over_position_grad_ratio": describe(
            scale.abs().mean(dim=0) / means.mean(dim=0).clamp_min(1e-30)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    dataset = CompactDataset.load(
        ROOT / f"dataset/2025_03_07_stage_with_fabric/{FRAME}/gaussians2d", device="cpu"
    )
    scene, train_local, _, _ = build_scene(dataset)
    torch.manual_seed(0)
    initializations, _ = build_initializations(dataset)

    arms = {}
    for arm in args.arms:
        model = initializations[arm]
        arms[arm] = {
            "world_sigma_max": describe(model.scales.amax(dim=-1)),
            "opacity": describe(model.opacity),
            **projected_sigma_px(model, scene, train_local),
            **measure_arm(model, scene, train_local),
        }
        a = arms[arm]
        print(
            f"[{arm}] intrinsic sigma {a['intrinsic_sigma_px']['median']:.4f} px "
            f"({100 * a['frac_below_dilation_floor']:.1f}% below the "
            f"{a['dilation_sigma_px']:.3f} px floor) "
            f"| suppression {a['dilation_suppression_factor']['median']:.4f} "
            f"| scale-grad consistency {a['scale_grad_consistency']['median']:.4f} "
            f"| |dL/dlog s| / |dL/dmu| {a['scale_over_position_grad_ratio']['median']:.4f}",
            flush=True,
        )

    summary = {
        "schema": "rtgs.beam_surfel_scale_gradient.v1",
        "kind": "post-hoc diagnostic on frozen initializations; no optimization, selects nothing",
        "frame": FRAME,
        "ewa_dilation_variance_px2": EWA_DILATION,
        "train_views": train_local,
        "arms": arms,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
