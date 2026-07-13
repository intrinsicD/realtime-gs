"""Stage 3: standard 3DGS optimization loop (L1 + D-SSIM, per-group LRs, density control).

Faithful to the original 3DGS recipe at small scale: Adam with per-parameter-group
learning rates (means LR scaled by scene extent and exponentially decayed), loss
(1-lambda)*L1 + lambda*(1-SSIM), adaptive density control between configurable
iterations. Rasterizer-agnostic: uses the reference renderer on CPU and gsplat on GPU
through the same interface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.metrics import psnr, ssim
from rtgs.data.scene import SceneData
from rtgs.optim.density import DensityConfig, DensityController
from rtgs.render.base import get_rasterizer


@dataclass
class TrainConfig:
    """Refinement hyperparameters (3DGS defaults, scaled for short runs)."""

    iterations: int = 200
    lr_means: float = 1.6e-4  # multiplied by scene extent, decayed to 1% over the run
    lr_quats: float = 1e-3
    lr_scales: float = 5e-3
    lr_opacity: float = 5e-2
    lr_sh: float = 2.5e-3
    ssim_lambda: float = 0.2
    rasterizer: str = "auto"
    device: str = "auto"
    densify: bool = True
    density: DensityConfig = field(default_factory=DensityConfig)
    eval_every: int = 50
    target_sh_degree: int = 3
    sh_degree_interval: int = 1_000
    use_masks: bool = True
    outside_alpha_lambda: float = 0.01
    seed: int = 0


class Trainer:
    """Optimizes a Gaussians3D initialization against a scene's posed images."""

    def __init__(self, config: TrainConfig | None = None):
        self.config = config or TrainConfig()

    def train(self, scene: SceneData, init: Gaussians3D) -> tuple[Gaussians3D, dict]:
        """Run the optimization; returns (refined gaussians, history dict)."""
        cfg = self.config
        device = _resolve_device(cfg.device)
        scene = scene.to(device)
        init = init.with_sh_degree(max(init.sh_degree, cfg.target_sh_degree)).to(device)
        gen = torch.Generator(device=device).manual_seed(cfg.seed)
        _, extent = scene.center_and_extent()
        renderer = get_rasterizer(cfg.rasterizer)

        params: dict[str, torch.Tensor] = {
            "means": init.means.detach().clone().requires_grad_(True),
            "quats": init.quats.detach().clone().requires_grad_(True),
            "log_scales": init.log_scales.detach().clone().requires_grad_(True),
            "opacity_logit": torch.logit(init.opacity.detach().clamp(1e-4, 1 - 1e-4))
            .clone()
            .requires_grad_(True),
            "sh": init.sh.detach().clone().requires_grad_(True),
        }
        lrs = {
            "means": cfg.lr_means * extent,
            "quats": cfg.lr_quats,
            "log_scales": cfg.lr_scales,
            "opacity_logit": cfg.lr_opacity,
            "sh": cfg.lr_sh,
        }
        optimizer = torch.optim.Adam(
            [{"params": [p], "lr": lrs[name], "name": name} for name, p in params.items()],
            eps=1e-15,
        )
        means_gamma = 0.01 ** (1.0 / max(cfg.iterations, 1))
        controller = (
            DensityController(cfg.density, init.n, extent, device=device) if cfg.densify else None
        )

        def build() -> Gaussians3D:
            return Gaussians3D(
                means=params["means"],
                quats=params["quats"],
                log_scales=params["log_scales"],
                opacity=torch.sigmoid(params["opacity_logit"]),
                sh=params["sh"],
            )

        history: dict = {
            "loss": [],
            "psnr": [],
            "elapsed": [],
            "n_gaussians": [],
            "active_sh_degree": [],
            "density_stats": None,
        }
        train_views = scene.training_views
        if not train_views:
            raise ValueError("scene has no training views")

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for it in range(cfg.iterations):
            view_pos = int(torch.randint(0, len(train_views), (1,), generator=gen, device=device))
            v = train_views[view_pos]
            target = scene.images[v]
            active_degree = (
                cfg.target_sh_degree
                if cfg.sh_degree_interval <= 0
                else min(cfg.target_sh_degree, it // cfg.sh_degree_interval)
            )
            out = renderer.render(build(), scene.cameras[v], sh_degree=active_degree)
            target_for_loss = target
            if cfg.use_masks and scene.masks is not None:
                mask = scene.masks[v].to(target.dtype).clamp(0, 1)
                target_for_loss = target * mask[..., None]
                weights = 0.1 + 0.9 * mask
                l1 = ((out.color - target_for_loss).abs() * weights[..., None]).mean()
                l1 = l1 + cfg.outside_alpha_lambda * (out.alpha * (1.0 - mask)).mean()
            else:
                l1 = (out.color - target).abs().mean()
            loss = (1 - cfg.ssim_lambda) * l1
            if cfg.ssim_lambda > 0:
                loss = loss + cfg.ssim_lambda * (1.0 - ssim(out.color, target_for_loss))
            optimizer.zero_grad()
            loss.backward()

            if controller is not None:
                controller.accumulate(out, scene.cameras[v].width, scene.cameras[v].height)
            optimizer.step()
            history["loss"].append(float(loss.detach()))

            # Exponential means-LR decay.
            for group in optimizer.param_groups:
                if group["name"] == "means":
                    group["lr"] *= means_gamma

            if controller is not None:
                params = controller.step(it + 1, params, optimizer, generator=gen)

            if (it + 1) % cfg.eval_every == 0 or it == cfg.iterations - 1:
                history["psnr"].append((it + 1, self.evaluate(scene, build(), renderer)))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                history["elapsed"].append((it + 1, time.perf_counter() - started))
                history["n_gaussians"].append((it + 1, params["means"].shape[0]))
                history["active_sh_degree"].append((it + 1, active_degree))

        if controller is not None:
            history["density_stats"] = controller.stats
        return build().detach(), history

    @staticmethod
    def evaluate(
        scene: SceneData,
        gaussians: Gaussians3D,
        renderer=None,
        indices: list[int] | None = None,
    ) -> float:
        """Mean PSNR over held-out views when available, otherwise all views."""
        renderer = renderer or get_rasterizer("auto")
        if indices is None:
            indices = scene.testing_views or list(range(scene.n_views))
        if not indices:
            raise ValueError("cannot evaluate an empty view split")
        device = gaussians.means.device
        vals = []
        with torch.no_grad():
            for index in indices:
                image = scene.images[index].to(device)
                cam = scene.cameras[index].to(device)
                out = renderer.render(gaussians, cam)
                if scene.masks is not None:
                    image = image * scene.masks[index].to(device)[..., None]
                vals.append(psnr(out.color.clamp(0, 1), image))
        return float(sum(vals) / len(vals))


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device
