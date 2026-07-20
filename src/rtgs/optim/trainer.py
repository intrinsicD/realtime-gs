"""Stage 3: standard 3DGS optimization loop (L1 + D-SSIM, per-group LRs, density control).

Faithful to the original 3DGS recipe at small scale: Adam with per-parameter-group
learning rates (means LR scaled by scene extent and exponentially decayed), loss
(1-lambda)*L1 + lambda*(1-SSIM), adaptive density control between configurable
iterations. Rasterizer-agnostic: uses the reference renderer on CPU and gsplat on GPU
through the same interface.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.metrics import image_metrics, masked_crop, ssim
from rtgs.core.sh import DEFAULT_SMU1_MU
from rtgs.data.scene import SceneData
from rtgs.optim.density import DensityConfig, DensityController
from rtgs.render.base import (
    DEFAULT_VISIBILITY_MARGIN_SIGMA,
    KernelSupportDiagnostics,
    SHColorDiagnostics,
    get_rasterizer,
)
from rtgs.render.torch_ref import KERNEL_SUPPORT_CUTOFF, KERNEL_SUPPORT_TAPER_WIDTH


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
    # Research controls for the post-SH nonnegative color floor. The standard hard floor
    # remains the default renderer semantics.
    sh_color_activation: str = "hard"
    sh_smu1_mu: float = DEFAULT_SMU1_MU
    collect_sh_color_diagnostics: bool = False
    # Opt-in reference-renderer controls for the compact EWA support boundary.
    kernel_support_mode: str = "hard"
    collect_kernel_support_diagnostics: bool = False
    visibility_margin_sigma: float = DEFAULT_VISIBILITY_MARGIN_SIGMA
    validate_render_finite: bool = False
    # Opt-in quaternion optimizer research seam. Ambient Adam remains the exact default.
    quaternion_update_policy: str = "current"
    seed: int = 0
    # Opt-in segmented-run coordinates. ``iterations`` remains the number of steps executed
    # by this invocation; schedules use ``schedule_iterations`` and global step numbers.
    iteration_offset: int = 0
    schedule_iterations: int | None = None
    # Endpoint multiplier for the exponential means-LR schedule. The historical 1% endpoint
    # remains the exact default; 1.0 provides an opt-in constant means LR for polish segments.
    means_lr_final_factor: float = 0.01
    # Clamp used before opacity logit initialization. The historical value remains default;
    # continuation/polish runs may opt into a tighter, explicitly receipted boundary.
    opacity_logit_epsilon: float = 1e-4


@dataclass(frozen=True)
class TrainStepControl:
    """Immutable per-step render and supervision resolution control."""

    render_downscale: int = 1
    loss_downscale: int = 1


def area_downsample_2x(image: torch.Tensor) -> torch.Tensor:
    """Downsample a float32 ``(H,W)`` or ``(H,W,C)`` tensor by exact 2x2 area pooling."""
    if image.dtype != torch.float32 or image.ndim not in (2, 3):
        raise ValueError("area downsampling requires a float32 (H,W) or (H,W,C) tensor")
    height, width = image.shape[:2]
    if height % 2 or width % 2:
        raise ValueError("area downsampling requires even image dimensions")
    if image.ndim == 2:
        pooled = F.avg_pool2d(image[None, None], kernel_size=2, stride=2, padding=0)
        return pooled[0, 0]
    channel_first = image.permute(2, 0, 1)[None]
    pooled = F.avg_pool2d(channel_first, kernel_size=2, stride=2, padding=0)
    return pooled[0].permute(1, 2, 0)


def downscale_camera(camera: Camera, downscale: int) -> Camera:
    """Return the exact edge-origin pinhole camera for downscale 1 or 2."""
    if type(downscale) is not int or downscale not in (1, 2):
        raise ValueError("camera downscale must be 1 or 2")
    if downscale == 1:
        return camera
    if camera.width % 2 or camera.height % 2:
        raise ValueError("camera downsampling requires even image dimensions")
    return Camera(
        fx=camera.fx / 2,
        fy=camera.fy / 2,
        cx=camera.cx / 2,
        cy=camera.cy / 2,
        width=camera.width // 2,
        height=camera.height // 2,
        R=camera.R,
        t=camera.t,
    )


def _control_sequence_hash(controls: tuple[TrainStepControl, ...]) -> str:
    payload = [[item.render_downscale, item.loss_downscale] for item in controls]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _prepare_step_controls(
    scene: SceneData,
    config: TrainConfig,
    step_controls: Sequence[TrainStepControl],
) -> tuple[
    tuple[TrainStepControl, ...],
    dict[int, list[torch.Tensor]],
    dict[int, list[Camera]],
    dict,
]:
    """Materialize, validate, hash, and construct an immutable training pyramid."""
    started = time.perf_counter()
    controls = tuple(step_controls)
    if len(controls) != config.iterations:
        raise ValueError("step_controls length must equal TrainConfig.iterations")
    for index, control in enumerate(controls):
        if not isinstance(control, TrainStepControl):
            raise TypeError(f"step_controls[{index}] must be a TrainStepControl")
        render_scale = control.render_downscale
        loss_scale = control.loss_downscale
        if type(render_scale) is not int or render_scale not in (1, 2):
            raise ValueError(f"step_controls[{index}] render_downscale must be 1 or 2")
        if type(loss_scale) is not int or loss_scale not in (1, 2):
            raise ValueError(f"step_controls[{index}] loss_downscale must be 1 or 2")
        if loss_scale < render_scale or loss_scale % render_scale:
            raise ValueError(
                f"step_controls[{index}] loss_downscale must be a multiple of render_downscale"
            )
    non_unit = any(
        control.render_downscale != 1 or control.loss_downscale != 1 for control in controls
    )
    if non_unit and config.densify:
        raise ValueError("non-unit step_controls require density control to be disabled")
    if non_unit and config.use_masks and scene.masks is not None:
        raise ValueError("non-unit step_controls do not support masked training")

    required_loss_scales = {control.loss_downscale for control in controls}
    required_render_scales = {control.render_downscale for control in controls}
    image_pyramid = {1: scene.images}
    camera_pyramid = {1: scene.cameras}
    if 2 in required_loss_scales:
        image_pyramid[2] = [area_downsample_2x(image) for image in scene.images]
    if 2 in required_render_scales:
        camera_pyramid[2] = [downscale_camera(camera, 2) for camera in scene.cameras]

    for view, (image, camera) in enumerate(zip(scene.images, scene.cameras, strict=True)):
        if image.dtype != torch.float32 or image.shape != (camera.height, camera.width, 3):
            raise ValueError(f"view {view} must have a matching float32 (H,W,3) image")
        for scale in required_loss_scales:
            target = image_pyramid[scale][view]
            if target.shape != (camera.height // scale, camera.width // scale, 3):
                raise RuntimeError(f"loss pyramid shape mismatch for view {view} at scale {scale}")
        for scale in required_render_scales:
            scaled = camera_pyramid[scale][view]
            if (scaled.height, scaled.width) != (camera.height // scale, camera.width // scale):
                raise RuntimeError(
                    f"camera pyramid shape mismatch for view {view} at scale {scale}"
                )

    metadata = {
        "sequence": [
            {
                "render_downscale": control.render_downscale,
                "loss_downscale": control.loss_downscale,
            }
            for control in controls
        ],
        "sequence_sha256": _control_sequence_hash(controls),
        "pyramid_setup_seconds": time.perf_counter() - started,
        "render_pixels": 0,
        "loss_pixels": 0,
        "per_view_scale_counts": {},
    }
    return controls, image_pyramid, camera_pyramid, metadata


class Trainer:
    """Optimizes a Gaussians3D initialization against a scene's posed images."""

    def __init__(self, config: TrainConfig | None = None):
        self.config = config or TrainConfig()

    def train(
        self,
        scene: SceneData,
        init: Gaussians3D,
        *,
        checkpoint_callback: Callable[[Gaussians3D, int], None] | None = None,
        step_controls: Sequence[TrainStepControl] | None = None,
        initialization_callback: Callable[[Gaussians3D], None] | None = None,
        quaternion_step_callback: (
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], None] | None
        ) = None,
    ) -> tuple[Gaussians3D, dict]:
        """Run optimization and return ``(refined, history)``.

        ``checkpoint_callback`` is an opt-in research observer. At each normal evaluation
        checkpoint it receives an isolated detached Gaussian clone and the completed step while
        autograd is disabled; mutating that clone cannot affect training. The default ``None``
        path does not construct snapshots or change the established optimizer/render sequence.

        ``step_controls`` is an opt-in fixed-topology research seam. A non-``None`` sequence is
        materialized and validated before optimization; the default ``None`` path does not build
        an image or camera pyramid and retains the established full-resolution behavior.

        ``initialization_callback`` receives an isolated effective-parameter snapshot after the
        one quaternion entry policy and before optimizer construction. The quaternion step
        observer receives isolated ``(q_old, q_star, q_new, completed_step)`` clones after policy
        application and before history, decay, density control, callback, or evaluation.
        """
        from rtgs.optim.strategies import (
            GsplatStrategyController,
            enforce_budget,
            strategy_uses_absgrad,
            validate_strategy_name,
        )

        cfg = self.config
        schedule_iterations = _resolve_schedule_iterations(cfg)
        means_lr_final_factor = _resolve_means_lr_final_factor(cfg)
        opacity_logit_epsilon = _resolve_opacity_logit_epsilon(cfg)
        quaternion_policy = _validate_quaternion_update_policy(cfg.quaternion_update_policy)
        if quaternion_policy != "current" and cfg.densify:
            raise ValueError("non-current quaternion_update_policy requires densify=False")
        if cfg.eval_every <= 0:
            raise ValueError("eval_every must be positive")
        device = _resolve_device(cfg.device)
        scene = scene.to(device)
        controls = None
        image_pyramid = None
        camera_pyramid = None
        control_metadata = None
        if step_controls is not None:
            controls, image_pyramid, camera_pyramid, control_metadata = _prepare_step_controls(
                scene, cfg, step_controls
            )
        init = init.with_sh_degree(max(init.sh_degree, cfg.target_sh_degree)).to(device)
        entry_quats = init.quats
        if quaternion_policy != "current":
            _require_valid_quaternion_rows(entry_quats, "quaternion entry input")
            entry_quats = F.normalize(entry_quats, dim=-1)
            _require_unit_quaternion_rows(entry_quats, "quaternion entry output")
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
            sh_color_activation=cfg.sh_color_activation,
            sh_smu1_mu=cfg.sh_smu1_mu,
            collect_sh_color_diagnostics=cfg.collect_sh_color_diagnostics,
            kernel_support_mode=cfg.kernel_support_mode,
            collect_kernel_support_diagnostics=cfg.collect_kernel_support_diagnostics,
            visibility_margin_sigma=cfg.visibility_margin_sigma,
        )

        params: dict[str, torch.nn.Parameter] = {
            "means": torch.nn.Parameter(init.means.detach().clone()),
            "quats": torch.nn.Parameter(entry_quats.detach().clone()),
            "scales": torch.nn.Parameter(init.log_scales.detach().clone()),
            "opacities": torch.nn.Parameter(
                torch.logit(
                    init.opacity.detach().clamp(
                        opacity_logit_epsilon,
                        1.0 - opacity_logit_epsilon,
                    )
                ).clone()
            ),
            "sh0": torch.nn.Parameter(init.sh[:, :1].detach().clone()),
            "shN": torch.nn.Parameter(init.sh[:, 1:].detach().clone()),
        }
        if initialization_callback is not None:
            snapshot = Gaussians3D(
                means=params["means"],
                quats=params["quats"],
                log_scales=params["scales"],
                opacity=torch.sigmoid(params["opacities"]),
                sh=torch.cat([params["sh0"], params["shN"]], dim=1),
            ).detach()
            with torch.no_grad():
                initialization_callback(snapshot)
        lrs = {
            "means": cfg.lr_means * extent,
            "quats": cfg.lr_quats,
            "scales": cfg.lr_scales,
            "opacities": cfg.lr_opacity,
            "sh0": cfg.lr_sh,
            "shN": cfg.lr_sh_rest,
        }
        means_gamma = means_lr_final_factor ** (1.0 / schedule_iterations)
        if cfg.iteration_offset > 0:
            lrs["means"] *= means_gamma**cfg.iteration_offset
        means_lr_initial = lrs["means"]
        optimizers: dict[str, torch.optim.Optimizer] = {
            name: torch.optim.Adam(
                [{"params": [parameter], "lr": lrs[name], "name": name}], eps=1e-15
            )
            for name, parameter in params.items()
        }
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
            "sampled_train_views": [],
            "sh_color_diagnostics": [],
            "kernel_support_diagnostics": [],
            "density_stats": None,
            "density_strategy": strategy_name if cfg.densify else "none",
            "resolved_sh_degree_interval": sh_interval,
            "iteration_offset": cfg.iteration_offset,
            "segment_iterations": cfg.iterations,
            "schedule_iterations": schedule_iterations,
            "means_lr_initial": means_lr_initial,
            "means_lr_final": means_lr_initial,
            "means_lr_final_factor": means_lr_final_factor,
            "means_lr_gamma": means_gamma,
            "opacity_logit_epsilon": opacity_logit_epsilon,
            "peak_vram_gb": 0.0,
        }
        if control_metadata is not None:
            history["step_control_metadata"] = control_metadata
        train_views = scene.training_views
        if not train_views:
            raise ValueError("scene has no training views")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        callback_seconds = 0.0
        started = time.perf_counter()
        for local_it in range(cfg.iterations):
            global_it = cfg.iteration_offset + local_it
            completed_step = global_it + 1
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            view_pos = int(torch.randint(0, len(train_views), (1,), generator=gen, device=device))
            v = train_views[view_pos]
            history["sampled_train_views"].append(int(v))
            control = None if controls is None else controls[local_it]
            if control is None:
                target = scene.images[v]
                render_camera = scene.cameras[v]
            else:
                target = image_pyramid[control.loss_downscale][v]
                render_camera = camera_pyramid[control.render_downscale][v]
                control_metadata["render_pixels"] += render_camera.height * render_camera.width
                control_metadata["loss_pixels"] += target.shape[0] * target.shape[1]
                view_counts = control_metadata["per_view_scale_counts"].setdefault(
                    str(v),
                    {
                        "render": {"1": 0, "2": 0},
                        "loss": {"1": 0, "2": 0},
                    },
                )
                view_counts["render"][str(control.render_downscale)] += 1
                view_counts["loss"][str(control.loss_downscale)] += 1
            active_degree = min(cfg.target_sh_degree, global_it // sh_interval)
            mask = None
            background = None
            if cfg.use_masks and scene.masks is not None:
                mask = scene.masks[v].to(target.dtype).clamp(0, 1)
                if cfg.random_background:
                    background = torch.rand(3, generator=gen, device=device)
            out = renderer.render(
                build(), render_camera, background=background, sh_degree=active_degree
            )
            if cfg.validate_render_finite:
                for field_name in ("color", "alpha", "depth"):
                    if not bool(torch.isfinite(getattr(out, field_name)).all()):
                        raise RuntimeError(
                            f"training render contains non-finite {field_name} "
                            f"at iteration {completed_step}"
                        )
            if gsplat_controller is not None:
                gsplat_controller.pre_backward(params, optimizers, out, global_it)
            color_for_loss = out.color
            if control is not None and control.loss_downscale != control.render_downscale:
                color_for_loss = area_downsample_2x(color_for_loss)
            if color_for_loss.shape != target.shape:
                raise RuntimeError(
                    f"training render/target shape mismatch at iteration {completed_step}: "
                    f"{tuple(color_for_loss.shape)} != {tuple(target.shape)}"
                )
            target_for_loss = target
            alpha_loss = target.new_zeros(())
            if mask is not None:
                outside = target.new_zeros(3) if background is None else background.to(target)
                target_for_loss = target * mask[..., None] + outside * (1.0 - mask[..., None])
                weights = 0.1 + 0.9 * mask
                l1 = ((color_for_loss - target_for_loss).abs() * weights[..., None]).mean()
                l1 = l1 + cfg.outside_alpha_lambda * (out.alpha * (1.0 - mask)).mean()
                alpha_loss = (out.alpha - mask).abs().mean()
            else:
                l1 = (color_for_loss - target).abs().mean()
            loss = (1 - cfg.ssim_lambda) * l1 + cfg.mask_alpha_lambda * alpha_loss
            if cfg.ssim_lambda > 0:
                if mask is not None:
                    pred_for_ssim = masked_crop(color_for_loss, mask)
                    target_for_ssim = masked_crop(target_for_loss, mask)
                else:
                    pred_for_ssim = color_for_loss
                    target_for_ssim = target_for_loss
                loss = loss + cfg.ssim_lambda * (1.0 - ssim(pred_for_ssim, target_for_ssim))
            if opacity_reg:
                loss = loss + opacity_reg * torch.sigmoid(params["opacities"]).mean()
            if scale_reg:
                loss = loss + scale_reg * torch.exp(params["scales"]).mean()
            loss.backward()

            if cfg.collect_sh_color_diagnostics:
                if out.sh_color_diagnostics is None:
                    history["sh_color_diagnostics"].append(
                        {
                            "iteration": completed_step,
                            "view": int(v),
                            "active_sh_degree": active_degree,
                            "observation_count": 0,
                        }
                    )
                else:
                    history["sh_color_diagnostics"].append(
                        _summarize_sh_color_diagnostics(
                            out.sh_color_diagnostics,
                            iteration=completed_step,
                            view=int(v),
                            active_sh_degree=active_degree,
                            smu1_mu=cfg.sh_smu1_mu,
                        )
                    )

            if cfg.collect_kernel_support_diagnostics:
                if out.kernel_support_diagnostics is None:
                    history["kernel_support_diagnostics"].append(
                        _empty_kernel_support_summary(
                            iteration=completed_step,
                            view=int(v),
                            active_sh_degree=active_degree,
                        )
                    )
                else:
                    history["kernel_support_diagnostics"].append(
                        _summarize_kernel_support_diagnostics(
                            out.kernel_support_diagnostics,
                            iteration=completed_step,
                            view=int(v),
                            active_sh_degree=active_degree,
                        )
                    )
                    # Outcome-sized graph tensors are valid for this backward pass only.
                    out.kernel_support_diagnostics = None

            if classic_controller is not None:
                classic_controller.accumulate(out, scene.cameras[v].width, scene.cameras[v].height)
            observe_quaternion_step = (
                quaternion_policy != "current" or quaternion_step_callback is not None
            )
            if not observe_quaternion_step:
                # Preserve the established default expression and optimizer order exactly.
                for optimizer in optimizers.values():
                    optimizer.step()
            else:
                q_old = None
                for name, optimizer in optimizers.items():
                    if name == "quats":
                        q_old = params["quats"].detach().clone()
                    optimizer.step()
                if q_old is None:  # pragma: no cover - fixed optimizer schema
                    raise RuntimeError("quaternion optimizer is missing")
                q_star = params["quats"].detach().clone()
                _require_valid_quaternion_rows(q_old, "pre-Adam quaternion")
                _require_valid_quaternion_rows(q_star, "post-Adam quaternion")
                if quaternion_policy != "current":
                    _apply_quaternion_update_policy_(
                        params["quats"], q_old, q_star, quaternion_policy
                    )
                q_new = params["quats"].detach().clone()
                _require_valid_quaternion_rows(q_new, "post-policy quaternion")
                if quaternion_step_callback is not None:
                    with torch.no_grad():
                        quaternion_step_callback(q_old, q_star, q_new, completed_step)
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
                params = classic_controller.step(completed_step, params, optimizers, generator=gen)
            elif gsplat_controller is not None:
                gsplat_controller.post_backward(
                    params,
                    optimizers,
                    out,
                    global_it,
                    optimizers["means"].param_groups[0]["lr"],
                    cfg.packed,
                )

            if completed_step % cfg.eval_every == 0 or local_it == cfg.iterations - 1:
                history["psnr"].append((completed_step, self.evaluate(scene, build(), renderer)))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if checkpoint_callback is None:
                    elapsed = time.perf_counter() - started
                else:
                    elapsed = time.perf_counter() - started - callback_seconds
                history["elapsed"].append((completed_step, elapsed))
                history["n_gaussians"].append((completed_step, params["means"].shape[0]))
                history["active_sh_degree"].append((completed_step, active_degree))
                if checkpoint_callback is not None:
                    # The observer receives only an isolated snapshot and the completed step.
                    # Snapshot construction and observer work are excluded from later native
                    # elapsed checkpoints as well as the current checkpoint above.
                    observer_started = time.perf_counter()
                    snapshot = build().detach()
                    with torch.no_grad():
                        checkpoint_callback(snapshot, completed_step)
                    callback_seconds += time.perf_counter() - observer_started

        history["means_lr_final"] = optimizers["means"].param_groups[0]["lr"]
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


def _summarize_sh_color_diagnostics(
    diagnostics: SHColorDiagnostics,
    *,
    iteration: int,
    view: int,
    active_sh_degree: int,
    smu1_mu: float,
) -> dict:
    """Reduce retained SH-color gradients without serializing outcome-sized tensors."""
    preactivation = diagnostics.preactivation
    activated = diagnostics.activated
    if preactivation.grad is None or activated.grad is None:
        raise RuntimeError("SH color diagnostics require retained gradients after backward")

    x = preactivation.detach().double().reshape(-1)
    upstream = activated.grad.detach().double().reshape(-1)
    raw_gradient = preactivation.grad.detach().double().reshape(-1)
    if x.shape != upstream.shape or x.shape != raw_gradient.shape:
        raise RuntimeError("SH color diagnostic tensors have inconsistent shapes")
    if not bool(torch.isfinite(x).all()):
        raise RuntimeError("SH color preactivation contains non-finite values")
    if not bool(torch.isfinite(upstream).all()) or not bool(torch.isfinite(raw_gradient).all()):
        raise RuntimeError("SH color diagnostics contain non-finite gradients")

    upstream_l1 = upstream.abs()
    negative = x < 0.0
    positive = x > 0.0
    recoverable = negative & (upstream < 0.0)
    derivative = 0.5 * (1.0 + x / torch.sqrt(x.square() + float(smu1_mu) ** 2))

    def scalar_sum(values: torch.Tensor) -> float:
        return float(values.sum(dtype=torch.float64).cpu())

    def selected_sum(values: torch.Tensor, mask: torch.Tensor) -> float:
        return scalar_sum(values[mask]) if bool(mask.any()) else 0.0

    def selected_max(values: torch.Tensor, mask: torch.Tensor) -> float:
        return float(values[mask].max().cpu()) if bool(mask.any()) else 0.0

    negative_bins = (
        (None, -0.10),
        (-0.10, -0.05),
        (-0.05, -0.02),
        (-0.02, -0.01),
        (-0.01, 0.0),
    )
    positive_bins = (
        (0.0, 0.01),
        (0.01, 0.02),
        (0.02, 0.05),
        (0.05, 0.10),
        (0.10, None),
    )

    def margin_bin(low: float | None, high: float | None) -> dict:
        mask = torch.ones_like(x, dtype=torch.bool)
        if low is not None:
            mask &= x >= low
        if high is not None:
            mask &= x < high
        return {
            "low": low,
            "high": high,
            "count": int(mask.sum().cpu()),
            "upstream_l1": selected_sum(upstream_l1, mask),
            "recoverable_l1": selected_sum(upstream_l1, mask & recoverable),
            "smu1_retained_l1": selected_sum(upstream_l1 * derivative, mask),
        }

    channels = []
    x_matrix = preactivation.detach().double().reshape(-1, 3)
    upstream_matrix = activated.grad.detach().double().reshape(-1, 3)
    derivative_matrix = 0.5 * (1.0 + x_matrix / torch.sqrt(x_matrix.square() + float(smu1_mu) ** 2))
    for channel in range(3):
        channel_x = x_matrix[:, channel]
        channel_upstream = upstream_matrix[:, channel]
        channel_l1 = channel_upstream.abs()
        channel_negative = channel_x < 0.0
        channel_recoverable = channel_negative & (channel_upstream < 0.0)
        channels.append(
            {
                "channel": channel,
                "observation_count": int(channel_x.numel()),
                "negative_count": int(channel_negative.sum().cpu()),
                "upstream_l1": scalar_sum(channel_l1),
                "blocked_l1": selected_sum(channel_l1, channel_negative),
                "recoverable_l1": selected_sum(channel_l1, channel_recoverable),
                "smu1_recovered_l1": selected_sum(
                    channel_l1 * derivative_matrix[:, channel], channel_recoverable
                ),
            }
        )

    negative_magnitude = -x[negative]
    return {
        "iteration": iteration,
        "view": view,
        "active_sh_degree": active_sh_degree,
        "visible_gaussian_count": int(diagnostics.gaussian_indices.numel()),
        "observation_count": int(x.numel()),
        "negative_count": int(negative.sum().cpu()),
        "zero_count": int((x == 0.0).sum().cpu()),
        "upstream_l1": scalar_sum(upstream_l1),
        "blocked_l1": selected_sum(upstream_l1, negative),
        "recoverable_l1": selected_sum(upstream_l1, recoverable),
        "smu1_recovered_l1": selected_sum(upstream_l1 * derivative, recoverable),
        "positive_upstream_l1": selected_sum(upstream_l1, positive),
        "positive_smu1_retained_l1": selected_sum(upstream_l1 * derivative, positive),
        "negative_raw_gradient_nonzero_count": int(((raw_gradient != 0.0) & negative).sum().cpu()),
        "negative_raw_gradient_max_abs": selected_max(raw_gradient.abs(), negative),
        "positive_raw_upstream_max_abs_error": selected_max(
            (raw_gradient - upstream).abs(), positive
        ),
        "negative_magnitude": {
            "count": int(negative_magnitude.numel()),
            "mean": scalar_sum(negative_magnitude) / max(int(negative_magnitude.numel()), 1),
            "max": float(negative_magnitude.max().cpu()) if negative_magnitude.numel() else 0.0,
        },
        "negative_margin_bins": [margin_bin(low, high) for low, high in negative_bins],
        "positive_margin_bins": [margin_bin(low, high) for low, high in positive_bins],
        "channels": channels,
    }


_KERNEL_Q_BINS = (
    (0.0, 2.0),
    (2.0, 4.0),
    (4.0, 6.0),
    (6.0, 8.0),
    (8.0, 10.0),
    (10.0, 12.0),
    (12.0, 13.0),
    (13.0, 14.0),
    (14.0, 15.0),
    (15.0, 16.0),
)


def _kernel_bin_template(low: float, high: float) -> dict:
    return {
        "low": low,
        "high": high,
        "count": 0,
        "upstream_l1": 0.0,
        "upstream_sum": 0.0,
        "recoverable_upstream_l1": 0.0,
        "hard_qgrad_l1": 0.0,
        "candidate_qgrad_l1": 0.0,
        "recoverable_candidate_qgrad_l1": 0.0,
        # First-order contribution at the kernel value consumed by this hard render.
        "kernel_loss_contribution_sum": 0.0,
        "kernel_loss_contribution_l1": 0.0,
    }


def _empty_kernel_support_summary(*, iteration: int, view: int, active_sh_degree: int) -> dict:
    """Return the additive schema used when the hard render has no visible primitives."""
    return {
        "iteration": iteration,
        "view": view,
        "active_sh_degree": active_sh_degree,
        "visible_gaussian_count": 0,
        "chunk_count": 0,
        "observation_count": 0,
        "eligible_count": 0,
        "interior_count": 0,
        "boundary_count": 0,
        "annulus_count": 0,
        "local_upstream_l1": 0.0,
        "local_upstream_sum": 0.0,
        "annulus_upstream_l1": 0.0,
        "annulus_upstream_sum": 0.0,
        "recoverable_annulus_upstream_l1": 0.0,
        "active_hard_qgrad_l1": 0.0,
        "boundary_hard_qgrad_l1": 0.0,
        "annulus_candidate_qgrad_l1": 0.0,
        "recoverable_annulus_candidate_qgrad_l1": 0.0,
        "local_kernel_loss_contribution_sum": 0.0,
        "local_kernel_loss_contribution_l1": 0.0,
        "annulus_incidence": 0.0,
        "annulus_upstream_fraction": 0.0,
        "recoverable_annulus_fraction": 0.0,
        "recovered_total_ratio": 0.0,
        "recovered_boundary_ratio": 0.0,
        "ratio_denominators_valid": False,
        "q_min": 0.0,
        "q_max": 0.0,
        "q_below_tolerance_count": 0,
        "negative_q_count": 0,
        "hard_outside_kernel_nonzero_count": 0,
        "hard_outside_kernel_max_abs": 0.0,
        "hard_outside_qgrad_nonzero_count": 0,
        "hard_outside_qgrad_max_abs": 0.0,
        "hard_active_qgrad_violation_count": 0,
        "hard_active_qgrad_max_abs_error": 0.0,
        "hard_active_qgrad_max_rel_error": 0.0,
        "q_bins": [_kernel_bin_template(low, high) for low, high in _KERNEL_Q_BINS],
    }


def _summarize_kernel_support_diagnostics(
    diagnostics: KernelSupportDiagnostics,
    *,
    iteration: int,
    view: int,
    active_sh_degree: int,
) -> dict:
    """Reduce retained hard-kernel gradients into additive float64 audit statistics."""
    if len(diagnostics.q_chunks) != len(diagnostics.kernel_chunks):
        raise RuntimeError("kernel-support diagnostic chunk counts do not match")

    result = _empty_kernel_support_summary(
        iteration=iteration,
        view=view,
        active_sh_degree=active_sh_degree,
    )
    result["visible_gaussian_count"] = int(diagnostics.gaussian_indices.numel())
    result["chunk_count"] = len(diagnostics.q_chunks)
    q_min = float("inf")
    q_max = float("-inf")
    cutoff = float(KERNEL_SUPPORT_CUTOFF)
    width = float(KERNEL_SUPPORT_TAPER_WIDTH)

    def scalar_sum(values: torch.Tensor) -> float:
        return float(values.sum(dtype=torch.float64).cpu())

    def selected_sum(values: torch.Tensor, mask: torch.Tensor) -> float:
        return scalar_sum(values[mask]) if bool(mask.any()) else 0.0

    def selected_max(values: torch.Tensor, mask: torch.Tensor) -> float:
        return float(values[mask].max().cpu()) if bool(mask.any()) else 0.0

    for q_tensor, kernel_tensor in zip(
        diagnostics.q_chunks, diagnostics.kernel_chunks, strict=True
    ):
        if q_tensor.grad is None or kernel_tensor.grad is None:
            raise RuntimeError(
                "kernel-support diagnostics require retained gradients after backward"
            )
        q = q_tensor.detach().double().reshape(-1)
        kernel = kernel_tensor.detach().double().reshape(-1)
        upstream = kernel_tensor.grad.detach().double().reshape(-1)
        raw_qgrad = q_tensor.grad.detach().double().reshape(-1)
        if not (q.shape == kernel.shape == upstream.shape == raw_qgrad.shape):
            raise RuntimeError("kernel-support diagnostic tensors have inconsistent shapes")
        if not bool(torch.isfinite(q).all()) or not bool(torch.isfinite(kernel).all()):
            raise RuntimeError("kernel-support diagnostics contain non-finite forward values")
        if not bool(torch.isfinite(upstream).all()) or not bool(torch.isfinite(raw_qgrad).all()):
            raise RuntimeError("kernel-support diagnostics contain non-finite gradients")

        result["observation_count"] += int(q.numel())
        if q.numel():
            q_min = min(q_min, float(q.min().cpu()))
            q_max = max(q_max, float(q.max().cpu()))

        upstream_l1 = upstream.abs()
        hard_region = q < cutoff
        interior = (q >= 0.0) & hard_region
        boundary = (q >= 8.0) & hard_region
        annulus = (q >= cutoff) & (q < cutoff + width)
        # q is mathematically nonnegative; use the literal I union A definition in ratios.
        eligible = interior | annulus
        recoverable = upstream < 0.0
        outside = q >= cutoff

        exp_term = torch.exp(-0.5 * q)
        hard_derivative = torch.where(hard_region, -0.5 * exp_term, 0.0)
        t = (q - cutoff) / width
        taper = 1.0 - 3.0 * t.square() + 2.0 * t.pow(3)
        taper_derivative = (-6.0 * t + 6.0 * t.square()) / width
        tail_derivative = exp_term * (-0.5 * taper + taper_derivative)
        candidate_derivative = torch.where(
            hard_region,
            -0.5 * exp_term,
            torch.where(annulus, tail_derivative, 0.0),
        )
        hard_qgrad_l1 = (upstream * hard_derivative).abs()
        candidate_qgrad_l1 = (upstream * candidate_derivative).abs()
        loss_contribution = upstream * kernel

        result["eligible_count"] += int(eligible.sum().cpu())
        result["interior_count"] += int(interior.sum().cpu())
        result["boundary_count"] += int(boundary.sum().cpu())
        result["annulus_count"] += int(annulus.sum().cpu())
        result["local_upstream_l1"] += selected_sum(upstream_l1, eligible)
        result["local_upstream_sum"] += selected_sum(upstream, eligible)
        result["annulus_upstream_l1"] += selected_sum(upstream_l1, annulus)
        result["annulus_upstream_sum"] += selected_sum(upstream, annulus)
        result["recoverable_annulus_upstream_l1"] += selected_sum(
            upstream_l1, annulus & recoverable
        )
        result["active_hard_qgrad_l1"] += selected_sum(hard_qgrad_l1, interior)
        result["boundary_hard_qgrad_l1"] += selected_sum(hard_qgrad_l1, boundary)
        result["annulus_candidate_qgrad_l1"] += selected_sum(candidate_qgrad_l1, annulus)
        result["recoverable_annulus_candidate_qgrad_l1"] += selected_sum(
            candidate_qgrad_l1, annulus & recoverable
        )
        result["local_kernel_loss_contribution_sum"] += selected_sum(loss_contribution, eligible)
        result["local_kernel_loss_contribution_l1"] += selected_sum(
            loss_contribution.abs(), eligible
        )

        result["q_below_tolerance_count"] += int((q < -1e-6).sum().cpu())
        result["negative_q_count"] += int((q < 0.0).sum().cpu())
        result["hard_outside_kernel_nonzero_count"] += int(((kernel != 0.0) & outside).sum().cpu())
        result["hard_outside_kernel_max_abs"] = max(
            result["hard_outside_kernel_max_abs"], selected_max(kernel.abs(), outside)
        )
        result["hard_outside_qgrad_nonzero_count"] += int(
            ((raw_qgrad != 0.0) & outside).sum().cpu()
        )
        result["hard_outside_qgrad_max_abs"] = max(
            result["hard_outside_qgrad_max_abs"], selected_max(raw_qgrad.abs(), outside)
        )
        expected_qgrad = upstream * hard_derivative
        qgrad_abs_error = (raw_qgrad - expected_qgrad).abs()
        qgrad_rel_error = qgrad_abs_error / expected_qgrad.abs().clamp_min(1e-30)
        qgrad_violation = hard_region & (qgrad_abs_error > 1e-6 + 1e-5 * expected_qgrad.abs())
        result["hard_active_qgrad_violation_count"] += int(qgrad_violation.sum().cpu())
        result["hard_active_qgrad_max_abs_error"] = max(
            result["hard_active_qgrad_max_abs_error"],
            selected_max(qgrad_abs_error, hard_region),
        )
        result["hard_active_qgrad_max_rel_error"] = max(
            result["hard_active_qgrad_max_rel_error"],
            selected_max(qgrad_rel_error, hard_region),
        )

        for row in result["q_bins"]:
            bin_mask = (q >= row["low"]) & (q < row["high"])
            row["count"] += int(bin_mask.sum().cpu())
            row["upstream_l1"] += selected_sum(upstream_l1, bin_mask)
            row["upstream_sum"] += selected_sum(upstream, bin_mask)
            row["recoverable_upstream_l1"] += selected_sum(upstream_l1, bin_mask & recoverable)
            row["hard_qgrad_l1"] += selected_sum(hard_qgrad_l1, bin_mask)
            row["candidate_qgrad_l1"] += selected_sum(candidate_qgrad_l1, bin_mask)
            row["recoverable_candidate_qgrad_l1"] += selected_sum(
                candidate_qgrad_l1, bin_mask & recoverable
            )
            row["kernel_loss_contribution_sum"] += selected_sum(loss_contribution, bin_mask)
            row["kernel_loss_contribution_l1"] += selected_sum(loss_contribution.abs(), bin_mask)

    diagnostics.q_chunks.clear()
    diagnostics.kernel_chunks.clear()
    result["q_min"] = 0.0 if result["observation_count"] == 0 else q_min
    result["q_max"] = 0.0 if result["observation_count"] == 0 else q_max

    def ratio(numerator: float | int, denominator: float | int) -> float:
        return float(numerator) / float(denominator) if float(denominator) > 0.0 else 0.0

    result["annulus_incidence"] = ratio(result["annulus_count"], result["eligible_count"])
    result["annulus_upstream_fraction"] = ratio(
        result["annulus_upstream_l1"], result["local_upstream_l1"]
    )
    result["recoverable_annulus_fraction"] = ratio(
        result["recoverable_annulus_upstream_l1"], result["annulus_upstream_l1"]
    )
    result["recovered_total_ratio"] = ratio(
        result["recoverable_annulus_candidate_qgrad_l1"], result["active_hard_qgrad_l1"]
    )
    result["recovered_boundary_ratio"] = ratio(
        result["recoverable_annulus_candidate_qgrad_l1"],
        result["boundary_hard_qgrad_l1"],
    )
    result["ratio_denominators_valid"] = all(
        result[key] > 0.0
        for key in (
            "eligible_count",
            "local_upstream_l1",
            "annulus_upstream_l1",
            "active_hard_qgrad_l1",
            "boundary_hard_qgrad_l1",
        )
    )
    return result


_QUATERNION_UPDATE_POLICIES = (
    "current",
    "unit_retraction",
    "tangent_displacement_retraction",
)
_MIN_QUATERNION_NORM = 1e-8
_UNIT_QUATERNION_TOLERANCE = 2e-5


def _validate_quaternion_update_policy(policy: str) -> str:
    if policy not in _QUATERNION_UPDATE_POLICIES:
        choices = ", ".join(_QUATERNION_UPDATE_POLICIES)
        raise ValueError(f"quaternion_update_policy must be one of {choices}; got {policy!r}")
    return policy


def _require_valid_quaternion_rows(quaternions: torch.Tensor, context: str) -> None:
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise RuntimeError(f"{context} must have shape (N,4)")
    norms = torch.linalg.vector_norm(quaternions, dim=-1)
    if not bool(torch.isfinite(quaternions).all()) or not bool(torch.isfinite(norms).all()):
        raise RuntimeError(f"{context} contains non-finite values")
    if bool((norms <= _MIN_QUATERNION_NORM).any()):
        raise RuntimeError(f"{context} contains a norm at or below {_MIN_QUATERNION_NORM}")


def _require_unit_quaternion_rows(quaternions: torch.Tensor, context: str) -> None:
    _require_valid_quaternion_rows(quaternions, context)
    norms = torch.linalg.vector_norm(quaternions, dim=-1)
    if bool(((norms - 1.0).abs() > _UNIT_QUATERNION_TOLERANCE).any()):
        raise RuntimeError(
            f"{context} differs from unit norm by more than {_UNIT_QUATERNION_TOLERANCE}"
        )


def _apply_quaternion_update_policy_(
    parameter: torch.nn.Parameter,
    q_old: torch.Tensor,
    q_star: torch.Tensor,
    policy: str,
) -> None:
    """Apply one frozen post-Adam quaternion policy without touching optimizer state."""
    policy = _validate_quaternion_update_policy(policy)
    if policy == "current":
        return
    if parameter.shape != q_old.shape or parameter.shape != q_star.shape:
        raise RuntimeError("quaternion policy tensors have inconsistent shapes")
    if not torch.equal(parameter.detach(), q_star):
        raise RuntimeError("q_star is not the actual post-Adam quaternion parameter")
    _require_unit_quaternion_rows(q_old, "pre-Adam candidate quaternion")
    _require_valid_quaternion_rows(q_star, "post-Adam candidate quaternion")
    if policy == "unit_retraction":
        proposed = F.normalize(q_star, dim=-1)
    else:
        delta = q_star - q_old
        radial = (q_old * delta).sum(dim=-1, keepdim=True)
        delta_tangent = delta - q_old * radial
        proposed = F.normalize(q_old + delta_tangent, dim=-1)
    _require_unit_quaternion_rows(proposed, "post-policy candidate quaternion")
    with torch.no_grad():
        parameter.copy_(proposed)


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
    schedule_iterations = _resolve_schedule_iterations(config)
    if config.target_sh_degree == 0:
        return schedule_iterations
    return min(1_000, max(1, schedule_iterations // (config.target_sh_degree + 1)))


def _resolve_schedule_iterations(config: TrainConfig) -> int:
    """Validate segmented-run coordinates and return the global schedule length."""

    for name, value in (
        ("iterations", config.iterations),
        ("iteration_offset", config.iteration_offset),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if config.iterations < 0:
        raise ValueError("iterations must be nonnegative")
    if config.iteration_offset < 0:
        raise ValueError("iteration_offset must be nonnegative")
    segmented = config.iteration_offset != 0 or config.schedule_iterations is not None
    if segmented and config.iterations == 0:
        raise ValueError("segmented-run iterations must be positive")
    total = (
        max(config.iterations, 1)
        if config.schedule_iterations is None
        else config.schedule_iterations
    )
    if isinstance(total, bool) or not isinstance(total, int):
        raise TypeError("schedule_iterations must be an integer or None")
    if total <= 0:
        raise ValueError("schedule_iterations must be positive")
    if config.iteration_offset + config.iterations > total:
        raise ValueError("iteration_offset + iterations must not exceed schedule_iterations")
    return total


def _resolve_means_lr_final_factor(config: TrainConfig) -> float:
    """Validate and return the global means-LR endpoint multiplier."""

    value = config.means_lr_final_factor
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("means_lr_final_factor must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("means_lr_final_factor must be a finite positive number")
    return value


def _resolve_opacity_logit_epsilon(config: TrainConfig) -> float:
    """Validate and return the open-interval opacity clamp used before ``logit``."""

    value = config.opacity_logit_epsilon
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("opacity_logit_epsilon must be a finite number in (0, 0.5)")
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 0.5:
        raise ValueError("opacity_logit_epsilon must be a finite number in (0, 0.5)")
    return value
