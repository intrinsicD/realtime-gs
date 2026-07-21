"""Fused multi-view stage-1 fitting: optimize every view's 2D gaussians in one batch.

Per-view fitting is embarrassingly parallel, but the serial ``fit_views`` loop leaves a GPU
mostly idle at production image sizes (each render launches small kernels). This module fits
all views jointly: identical per-view initialization (same ``seed + i`` generators), stacked
raw parameters, one batched render per iteration via
:func:`rtgs.image2gs.renderer2d.render_gaussians_2d_batched`, and a summed per-view MSE loss.
Because the views share no parameters, gradients — and therefore per-parameter Adam updates
and the cosine schedule — match the serial path exactly up to float summation order, so
per-view results agree with serial fits to within normal float drift (parity is tested with
tolerances, not bitwise).

Opt-in via ``FitConfig.batch_views``; the serial path remains the default and the semantics
preregistered harnesses rely on. Unsupported serial features raise instead of silently
diverging: masks (per-view crops change geometry), the StructSplat backend (own loop), and
convergence early stopping (per-view stop iterations differ).
"""

from __future__ import annotations

import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.metrics import psnr
from rtgs.image2gs.fit import (
    _CANDIDATE_APPEARANCE,
    _CURRENT_APPEARANCE,
    _MIN_DIAG,
    FitConfig,
    _softplus_inv,
    _validate_fit_controls,
    _validate_initial_gaussians,
    _validate_raw_parameters,
    init_gaussians_2d,
)
from rtgs.image2gs.renderer2d import render_gaussians_2d_batched


def fit_views_batched(
    images: list[torch.Tensor],
    config: FitConfig | None = None,
    seed: int = 0,
    masks: list[torch.Tensor] | None = None,
) -> tuple[list[Gaussians2D], list[dict]]:
    """Fit every view jointly in one fused optimization; API mirrors ``fit_views``.

    Returns per-view detached gaussians and per-view history dicts with the same keys as the
    serial path (``psnr``, ``stopped_iter``, ``final_psnr_full``, ``final_psnr``). Raw
    non-finite parameter errors report first-dimension rows, which here index views.
    """
    config = config or FitConfig()
    _validate_fit_controls(config, None, ())
    if config.backend != "native":
        raise ValueError("batch_views requires the native backend")
    if masks is not None and any(mask is not None for mask in masks):
        raise ValueError("batch_views does not support masks; use serial fit_views")
    if config.convergence_patience:
        raise ValueError("batch_views does not support convergence early stopping")
    if not images:
        return [], []
    height, width = images[0].shape[:2]
    for image in images:
        if image.shape != images[0].shape:
            raise ValueError("batch_views requires equally sized views")

    inits = []
    for i, image in enumerate(images):
        generator = torch.Generator(device=image.device).manual_seed(seed + i)
        g0 = init_gaussians_2d(image, config.n_gaussians, config.grad_init_mix, generator)
        _validate_initial_gaussians(g0, height, width)
        inits.append(g0)
    n_views = len(inits)
    targets = torch.stack(images)
    xy0 = torch.stack([g.xy for g in inits])
    chol0 = torch.stack([g.chol for g in inits])
    color0 = torch.stack([g.color for g in inits])
    weight0 = torch.stack([g.weight for g in inits])

    # Raw (unconstrained) parameters, stacked (B, N, ...); transforms match the serial path
    # elementwise, so each view's raw state equals its serial counterpart.
    wh = targets.new_tensor([width, height])
    xy_raw = torch.logit((xy0 / wh).clamp(1e-4, 1 - 1e-4))
    diag_raw = _softplus_inv(chol0[..., [0, 2]] - _MIN_DIAG)
    off_raw = chol0[..., 1].clone()
    color_raw = torch.logit(color0.clamp(1e-3, 1 - 1e-3))
    weight_raw = torch.logit(weight0.clamp(1e-3, 1 - 1e-3))
    if config.appearance_parameterization == _CURRENT_APPEARANCE:
        appearance_raw = {"color_raw": color_raw, "weight_raw": weight_raw}
    else:
        assert config.appearance_parameterization == _CANDIDATE_APPEARANCE
        common_amplitude = torch.sigmoid(weight_raw)[..., None] * torch.sigmoid(color_raw)
        appearance_raw = {"amplitude_raw": torch.logit(common_amplitude)}
    raw_parameters = {
        "xy_raw": xy_raw,
        "diag_raw": diag_raw,
        "off_raw": off_raw,
        **appearance_raw,
    }
    _validate_raw_parameters(raw_parameters)
    geometry_names = ("xy_raw", "diag_raw", "off_raw")
    appearance_names = tuple(appearance_raw)
    optimizer_param_names = (
        appearance_names if config.freeze_geometry else geometry_names + appearance_names
    )
    for name, parameter in raw_parameters.items():
        parameter.requires_grad_(name in optimizer_param_names)
    params = [raw_parameters[name] for name in optimizer_param_names]

    def build() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_raw_parameters(raw_parameters)
        diag = torch.nn.functional.softplus(diag_raw) + _MIN_DIAG
        chol = torch.stack([diag[..., 0], off_raw, diag[..., 1]], dim=-1)
        if config.appearance_parameterization == _CURRENT_APPEARANCE:
            color = torch.sigmoid(color_raw)
            weight = torch.sigmoid(weight_raw)
        else:
            color = torch.sigmoid(appearance_raw["amplitude_raw"])
            weight = torch.ones_like(color[..., 0])
        return torch.sigmoid(xy_raw) * wh, chol, color, weight

    opt = torch.optim.Adam(params, lr=config.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config.iterations, eta_min=config.lr * 0.1
    )
    histories: list[dict] = [
        {"psnr": [], "stopped_iter": config.iterations - 1} for _ in range(n_views)
    ]
    for it in range(config.iterations):
        opt.zero_grad()
        xy, chol, color, weight = build()
        rendered = render_gaussians_2d_batched(
            xy,
            chol,
            color,
            weight,
            height,
            width,
            row_chunk=config.row_chunk,
            renderer=config.native_renderer,
        )
        # Sum of per-view MSE means: each view's gradient equals its serial-fit gradient.
        loss = (rendered - targets).square().mean(dim=(1, 2, 3)).sum()
        loss.backward()
        opt.step()
        sched.step()
        if it % config.log_every == 0 or it == config.iterations - 1:
            with torch.no_grad():
                for b in range(n_views):
                    histories[b]["psnr"].append((it, psnr(rendered[b].clamp(0, 1), targets[b])))

    with torch.no_grad():
        xy, chol, color, weight = build()
        final = render_gaussians_2d_batched(
            xy,
            chol,
            color,
            weight,
            height,
            width,
            row_chunk=config.row_chunk,
            renderer=config.native_renderer,
        )
        for b in range(n_views):
            histories[b]["final_psnr_full"] = psnr(final[b].clamp(0, 1), targets[b])
            histories[b]["final_psnr"] = histories[b]["final_psnr_full"]
    results = [
        Gaussians2D(xy=xy[b], chol=chol[b], color=color[b], weight=weight[b]).detach()
        for b in range(n_views)
    ]
    return results, histories
