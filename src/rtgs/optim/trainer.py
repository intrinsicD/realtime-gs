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
from rtgs.core.metrics import image_metrics, masked_crop, ssim
from rtgs.data.scene import SceneData
from rtgs.optim.density import DensityConfig, DensityController
from rtgs.render.base import get_rasterizer


@dataclass
class TrainConfig:
    """Refinement hyperparameters with CPU-classic and CUDA gsplat strategies."""

    iterations: int = 200
    lr_means: float = 1.6e-4  # multiplied by scene extent, decayed to 1% over the run
    lr_quats: float = 1e-3
    lr_scales: float = 5e-3
    lr_opacity: float = 5e-2
    lr_sh: float = 2.5e-3  # degree-0/DC band
    lr_sh_rest: float = 2.5e-3 / 20.0
    ssim_lambda: float = 0.2
    rasterizer: str = "auto"
    device: str = "auto"
    densify: bool = True
    density_strategy: str = "classic"
    density: DensityConfig = field(default_factory=DensityConfig)
    eval_every: int = 50
    target_sh_degree: int = 3
    # None preserves the standard 1k interval for long runs but scales it down so a short run
    # still trains every requested band.
    sh_degree_interval: int | None = None
    use_masks: bool = True
    outside_alpha_lambda: float = 0.01
    mask_alpha_lambda: float = 0.05
    random_background: bool = True
    opacity_reg: float | None = None  # MCMC default: 0.01; otherwise 0
    scale_reg: float | None = None  # MCMC default: 0.01; otherwise 0
    packed: bool = False
    antialiased: bool = False
    seed: int = 0


class Trainer:
    """Optimizes a Gaussians3D initialization against a scene's posed images."""

    def __init__(self, config: TrainConfig | None = None):
        self.config = config or TrainConfig()

    def train(self, scene: SceneData, init: Gaussians3D) -> tuple[Gaussians3D, dict]:
        """Run the optimization; returns (refined gaussians, history dict)."""
        from rtgs.optim.strategies import (
            GsplatStrategyController,
            enforce_budget,
            strategy_uses_absgrad,
            validate_strategy_name,
        )

        cfg = self.config
        if cfg.eval_every <= 0:
            raise ValueError("eval_every must be positive")
        device = _resolve_device(cfg.device)
        scene = scene.to(device)
        init = init.with_sh_degree(max(init.sh_degree, cfg.target_sh_degree)).to(device)
        gen = torch.Generator(device=device).manual_seed(cfg.seed)
        _, extent = scene.center_and_extent()
        strategy_name = cfg.density_strategy
        if cfg.densify:
            validate_strategy_name(strategy_name)
        needs_absgrad = cfg.densify and strategy_uses_absgrad(strategy_name, cfg.density)
        renderer = get_rasterizer(
            cfg.rasterizer,
            device=device,
            packed=cfg.packed,
            absgrad=needs_absgrad,
            antialiased=cfg.antialiased,
        )

        params: dict[str, torch.nn.Parameter] = {
            "means": torch.nn.Parameter(init.means.detach().clone()),
            "quats": torch.nn.Parameter(init.quats.detach().clone()),
            "scales": torch.nn.Parameter(init.log_scales.detach().clone()),
            "opacities": torch.nn.Parameter(
                torch.logit(init.opacity.detach().clamp(1e-4, 1 - 1e-4)).clone()
            ),
            "sh0": torch.nn.Parameter(init.sh[:, :1].detach().clone()),
            "shN": torch.nn.Parameter(init.sh[:, 1:].detach().clone()),
        }
        lrs = {
            "means": cfg.lr_means * extent,
            "quats": cfg.lr_quats,
            "scales": cfg.lr_scales,
            "opacities": cfg.lr_opacity,
            "sh0": cfg.lr_sh,
            "shN": cfg.lr_sh_rest,
        }
        optimizers: dict[str, torch.optim.Optimizer] = {
            name: torch.optim.Adam(
                [{"params": [parameter], "lr": lrs[name], "name": name}], eps=1e-15
            )
            for name, parameter in params.items()
        }
        means_gamma = 0.01 ** (1.0 / max(cfg.iterations, 1))
        classic_controller = None
        gsplat_controller = None
        if cfg.densify and strategy_name == "classic":
            classic_controller = DensityController(cfg.density, init.n, extent, device=device)
            if init.n > cfg.density.max_gaussians:
                params = classic_controller.step(
                    0, params, optimizers, generator=gen, force_budget=True
                )
        elif cfg.densify:
            if cfg.rasterizer == "torch" or device.type != "cuda":
                raise RuntimeError(f"{strategy_name} requires --rasterizer gsplat and CUDA")
            enforce_budget(params, optimizers, cfg.density.max_gaussians)
            gsplat_controller = GsplatStrategyController(
                strategy_name, cfg.density, extent, params, optimizers
            )

        def build() -> Gaussians3D:
            return Gaussians3D(
                means=params["means"],
                quats=params["quats"],
                log_scales=params["scales"],
                opacity=torch.sigmoid(params["opacities"]),
                sh=torch.cat([params["sh0"], params["shN"]], dim=1),
            )

        sh_interval = _resolve_sh_interval(cfg)
        opacity_reg = (
            0.01 if strategy_name == "gsplat-mcmc" and cfg.opacity_reg is None else cfg.opacity_reg
        ) or 0.0
        scale_reg = (
            0.01 if strategy_name == "gsplat-mcmc" and cfg.scale_reg is None else cfg.scale_reg
        ) or 0.0
        history: dict = {
            "loss": [],
            "loss_terms": [],
            "psnr": [],
            "elapsed": [],
            "n_gaussians": [],
            "active_sh_degree": [],
            "density_stats": None,
            "density_strategy": strategy_name if cfg.densify else "none",
            "resolved_sh_degree_interval": sh_interval,
            "peak_vram_gb": 0.0,
        }
        train_views = scene.training_views
        if not train_views:
            raise ValueError("scene has no training views")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for it in range(cfg.iterations):
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            view_pos = int(torch.randint(0, len(train_views), (1,), generator=gen, device=device))
            v = train_views[view_pos]
            target = scene.images[v]
            active_degree = min(cfg.target_sh_degree, it // sh_interval)
            mask = None
            background = None
            if cfg.use_masks and scene.masks is not None:
                mask = scene.masks[v].to(target.dtype).clamp(0, 1)
                if cfg.random_background:
                    background = torch.rand(3, generator=gen, device=device)
            out = renderer.render(
                build(), scene.cameras[v], background=background, sh_degree=active_degree
            )
            if gsplat_controller is not None:
                gsplat_controller.pre_backward(params, optimizers, out, it)
            target_for_loss = target
            alpha_loss = target.new_zeros(())
            if mask is not None:
                outside = target.new_zeros(3) if background is None else background.to(target)
                target_for_loss = target * mask[..., None] + outside * (1.0 - mask[..., None])
                weights = 0.1 + 0.9 * mask
                l1 = ((out.color - target_for_loss).abs() * weights[..., None]).mean()
                l1 = l1 + cfg.outside_alpha_lambda * (out.alpha * (1.0 - mask)).mean()
                alpha_loss = (out.alpha - mask).abs().mean()
            else:
                l1 = (out.color - target).abs().mean()
            loss = (1 - cfg.ssim_lambda) * l1 + cfg.mask_alpha_lambda * alpha_loss
            if cfg.ssim_lambda > 0:
                if mask is not None:
                    pred_for_ssim = masked_crop(out.color, mask)
                    target_for_ssim = masked_crop(target_for_loss, mask)
                else:
                    pred_for_ssim = out.color
                    target_for_ssim = target_for_loss
                loss = loss + cfg.ssim_lambda * (1.0 - ssim(pred_for_ssim, target_for_ssim))
            if opacity_reg:
                loss = loss + opacity_reg * torch.sigmoid(params["opacities"]).mean()
            if scale_reg:
                loss = loss + scale_reg * torch.exp(params["scales"]).mean()
            loss.backward()

            if classic_controller is not None:
                classic_controller.accumulate(out, scene.cameras[v].width, scene.cameras[v].height)
            for optimizer in optimizers.values():
                optimizer.step()
            history["loss"].append(float(loss.detach()))
            history["loss_terms"].append(
                {
                    "l1": float(l1.detach()),
                    "alpha": float(alpha_loss.detach()),
                    "opacity_reg": opacity_reg,
                    "scale_reg": scale_reg,
                }
            )

            # Exponential means-LR decay.
            optimizers["means"].param_groups[0]["lr"] *= means_gamma

            if classic_controller is not None:
                params = classic_controller.step(it + 1, params, optimizers, generator=gen)
            elif gsplat_controller is not None:
                gsplat_controller.post_backward(
                    params,
                    optimizers,
                    out,
                    it,
                    optimizers["means"].param_groups[0]["lr"],
                    cfg.packed,
                )

            if (it + 1) % cfg.eval_every == 0 or it == cfg.iterations - 1:
                history["psnr"].append((it + 1, self.evaluate(scene, build(), renderer)))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                history["elapsed"].append((it + 1, time.perf_counter() - started))
                history["n_gaussians"].append((it + 1, params["means"].shape[0]))
                history["active_sh_degree"].append((it + 1, active_degree))

        if classic_controller is not None:
            history["density_stats"] = classic_controller.stats
        elif gsplat_controller is not None:
            history["density_stats"] = gsplat_controller.stats
        if device.type == "cuda":
            history["peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
        return build().detach(), history

    @staticmethod
    def evaluate(
        scene: SceneData,
        gaussians: Gaussians3D,
        renderer=None,
        indices: list[int] | None = None,
    ) -> float:
        """Mean foreground PSNR when masks exist, otherwise ordinary mean PSNR."""
        metrics = Trainer.evaluate_metrics(scene, gaussians, renderer, indices)
        return metrics["psnr_fg"] if "psnr_fg" in metrics else metrics["psnr"]

    @staticmethod
    def evaluate_metrics(
        scene: SceneData,
        gaussians: Gaussians3D,
        renderer=None,
        indices: list[int] | None = None,
    ) -> dict[str, float]:
        """Average explicit image metrics over a split, preferring held-out views."""
        renderer = renderer or get_rasterizer("auto", device=gaussians.means.device)
        if indices is None:
            indices = scene.testing_views or list(range(scene.n_views))
        if not indices:
            raise ValueError("cannot evaluate an empty view split")
        device = gaussians.means.device
        totals: dict[str, float] = {}
        with torch.no_grad():
            for index in indices:
                image = scene.images[index].to(device)
                cam = scene.cameras[index].to(device)
                out = renderer.render(gaussians, cam)
                mask = None if scene.masks is None else scene.masks[index].to(device)
                values = image_metrics(out.color, image, mask)
                if mask is not None:
                    foreground = mask > 0.5
                    predicted = out.alpha > 0.5
                    intersection = (foreground & predicted).sum()
                    union = (foreground | predicted).sum().clamp_min(1)
                    values["alpha_iou"] = float(intersection / union)
                    values["alpha_inside"] = float(out.alpha[foreground].mean())
                    background = ~foreground
                    values["alpha_outside"] = (
                        float(out.alpha[background].mean()) if bool(background.any()) else 0.0
                    )
                for key, value in values.items():
                    totals[key] = totals.get(key, 0.0) + value
        return {key: value / len(indices) for key, value in totals.items()}


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _resolve_sh_interval(config: TrainConfig) -> int:
    """Resolve a schedule that activates all bands in short runs and matches long-run 3DGS."""
    if config.target_sh_degree < 0 or config.target_sh_degree > 3:
        raise ValueError("target_sh_degree must be between 0 and 3")
    if config.sh_degree_interval is not None:
        if config.sh_degree_interval <= 0:
            raise ValueError("sh_degree_interval must be positive or None for auto")
        return config.sh_degree_interval
    if config.target_sh_degree == 0:
        return max(config.iterations, 1)
    return min(1_000, max(1, config.iterations // (config.target_sh_degree + 1)))
