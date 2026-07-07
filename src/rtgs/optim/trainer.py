"""Stage 3: standard 3DGS optimization loop (L1 + D-SSIM, per-group LRs, density control).

Faithful to the original 3DGS recipe at small scale: Adam with per-parameter-group
learning rates (means LR scaled by scene extent and exponentially decayed), loss
(1-lambda)*L1 + lambda*(1-SSIM), adaptive density control between configurable
iterations. Rasterizer-agnostic: uses the reference renderer on CPU and gsplat on GPU
through the same interface.
"""

from __future__ import annotations

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
    densify: bool = True
    density: DensityConfig = field(default_factory=DensityConfig)
    eval_every: int = 50
    seed: int = 0


class Trainer:
    """Optimizes a Gaussians3D initialization against a scene's posed images."""

    def __init__(self, config: TrainConfig | None = None):
        self.config = config or TrainConfig()

    def train(self, scene: SceneData, init: Gaussians3D) -> tuple[Gaussians3D, dict]:
        """Run the optimization; returns (refined gaussians, history dict)."""
        cfg = self.config
        gen = torch.Generator().manual_seed(cfg.seed)
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
        controller = DensityController(cfg.density, init.n, extent) if cfg.densify else None

        def build() -> Gaussians3D:
            return Gaussians3D(
                means=params["means"],
                quats=params["quats"],
                log_scales=params["log_scales"],
                opacity=torch.sigmoid(params["opacity_logit"]),
                sh=params["sh"],
            )

        history: dict = {"loss": [], "psnr": [], "n_gaussians": [], "density_stats": None}
        n_views = scene.n_views

        for it in range(cfg.iterations):
            v = int(torch.randint(0, n_views, (1,), generator=gen))
            target = scene.images[v]
            out = renderer.render(build(), scene.cameras[v])
            l1 = (out.color - target).abs().mean()
            loss = (1 - cfg.ssim_lambda) * l1
            if cfg.ssim_lambda > 0:
                loss = loss + cfg.ssim_lambda * (1.0 - ssim(out.color, target))
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
                history["n_gaussians"].append((it + 1, params["means"].shape[0]))

        if controller is not None:
            history["density_stats"] = controller.stats
        return build().detach(), history

    @staticmethod
    def evaluate(scene: SceneData, gaussians: Gaussians3D, renderer=None) -> float:
        """Mean PSNR over all views."""
        renderer = renderer or get_rasterizer("auto")
        vals = []
        with torch.no_grad():
            for image, cam in zip(scene.images, scene.cameras):
                out = renderer.render(gaussians, cam)
                vals.append(psnr(out.color.clamp(0, 1), image))
        return float(sum(vals) / len(vals))
