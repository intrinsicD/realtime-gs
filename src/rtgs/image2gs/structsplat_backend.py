"""Optional StructSplat stage-1 backend with progressive residual density control.

StructSplat is imported lazily so the base package and CPU tests never require it.  The policy
uses feature-aware anisotropic WSE placement, then tensor-aligned residual growth until quality
stalls or a configurable maximum is reached.  A fixed-budget schedule can reproduce the fair
320-to-640 experiment. Relocation is an ablation and defaults off because it hurt that result.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.metrics import masked_psnr
from rtgs.core.observation2d import GaussianObservationField

if TYPE_CHECKING:
    from rtgs.image2gs.fit import FitConfig as RTGSFitConfig


def fit_image_structsplat(
    image: torch.Tensor,
    config: RTGSFitConfig,
    seed: int | None = None,
    mask: torch.Tensor | None = None,
    *,
    observation_callback: Callable[[GaussianObservationField], None] | None = None,
    observation_view_id: str | None = None,
) -> tuple[Gaussians2D, dict]:
    """Fit one image using StructSplat's progressive compact representation."""
    try:
        from structsplat.config import FitConfig, InitConfig, StructureTensorConfig
        from structsplat.fit import fit
        from structsplat.init import build_field
    except ImportError as exc:
        raise RuntimeError(
            "StructSplat backend is optional; install its MIT-licensed package into this "
            "environment (for example: pip install -e ~/Documents/structsplat)"
        ) from exc

    if config.n_gaussians < 16:
        raise ValueError("StructSplat initial budget must be at least 16")
    if config.growth_waves < 1:
        raise ValueError("StructSplat growth_waves must be positive")
    if not 0.0 <= config.relocate_fraction <= 1.0:
        raise ValueError("StructSplat relocate_fraction must be in [0, 1]")

    canvas_height, canvas_width = image.shape[:2]
    xy_offset = image.new_zeros(2)
    if mask is not None:
        from rtgs.image2gs.fit import _crop_to_mask

        image, mask, xy_offset = _crop_to_mask(image, mask)
    target = image
    if mask is not None:
        if mask.shape != image.shape[:2]:
            raise ValueError("mask size does not match image")
        target = image * mask.to(image).clamp(0, 1)[..., None]
    target = target.contiguous()
    image_np = target.detach().cpu().numpy().astype(np.float32, copy=False)
    start_budget = int(config.n_gaussians)
    max_budget = start_budget if config.max_gaussians is None else int(config.max_gaussians)
    if max_budget < start_budget:
        raise ValueError("StructSplat max_gaussians cannot be below the initial count")
    add_total = max_budget - start_budget
    growth_count = max(1, math.ceil(add_total / config.growth_waves)) if add_total else 0
    growth_every = max(1, config.iterations // (config.growth_waves + 1)) if add_total else None
    reference_side = 160.0
    feature_cap = 12.0 * max(image.shape[:2]) / reference_side
    renderer = config.structsplat_renderer
    if renderer == "auto":
        renderer = "cuda" if image.device.type == "cuda" else "normalized"

    init_cfg = InitConfig(
        strategy="aniso_onedge",
        num_gaussians=start_budget,
        seed=0 if seed is None else seed,
        sampling_mode="wse",
        flank_offset_frac=0.0,
        scale_cap_mode="feature",
        scale_cap_max=feature_cap,
    )
    fit_cfg = FitConfig(
        iters=config.iterations,
        renderer=renderer,
        pixel_loss="l1",
        ssim_weight=0.3,
        log_every=max(1, config.log_every),
        split_every=(None if config.adaptive_density else growth_every),
        split_count=(0 if config.adaptive_density else growth_count),
        split_mode="residual_tensor_add",
        max_gaussians=max_budget,
        adaptive_count=config.adaptive_density and add_total > 0,
        adaptive_growth_every=(growth_every or max(1, config.iterations)),
        adaptive_growth_count=max(1, growth_count),
        adaptive_split_mode="residual_tensor_add",
        adaptive_min_delta_psnr=config.convergence_tol,
        adaptive_patience=max(1, config.convergence_patience or 2),
        early_stop_patience=(config.convergence_patience or None),
        early_stop_min_delta=config.convergence_tol,
        early_stop_min_iters=max(config.convergence_check_every, config.iterations // 3),
        relocate_every=(
            growth_every if config.relocate_fraction > 0 and config.adaptive_density else None
        ),
        relocate_at_split=config.relocate_fraction > 0 and not config.adaptive_density,
        relocate_count=(math.ceil(growth_count * config.relocate_fraction) if growth_count else 0),
    )
    field = build_field(
        image_np,
        init_cfg,
        StructureTensorConfig(),
        device=str(image.device),
    )
    result = fit(field, target, fit_cfg, verbose=False)
    g2d = field_to_gaussians2d(result["field"])
    g2d.xy += xy_offset
    if observation_callback is not None:
        normalized_renderers = {
            "normalized",
            "cuda",
            "cuda_normalized",
            "cuda_block_reduce",
            "cuda_tiled",
            "cuda_tiled_normalized",
        }
        if renderer not in normalized_renderers:
            raise ValueError(
                f"cannot export observation for unmapped StructSplat renderer {renderer!r}"
            )
        fit_height, fit_width = image.shape[:2]
        fit_window = (
            int(xy_offset[0].item()),
            int(xy_offset[1].item()),
            fit_width,
            fit_height,
        )
        producer_version, producer_source_digest, fit_config_digest = _structsplat_provenance(
            fit_cfg
        )
        observation_callback(
            field_to_observation(
                result["field"],
                canvas_size=(canvas_height, canvas_width),
                fit_window=fit_window,
                blend_mode="normalized",
                sigma_cutoff=float(fit_cfg.sigma_cutoff),
                support_fade_alpha=1.0 if fit_cfg.support_fade else 0.0,
                aa_dilation=float(fit_cfg.aa_dilation),
                view_id=observation_view_id,
                n_init=start_budget,
                producer_version=producer_version,
                producer_source_digest=producer_source_digest,
                fit_config_digest=fit_config_digest,
            )
        )
    foreground_psnr = (
        float(result["psnr"])
        if mask is None
        else masked_psnr(result["render"].clamp(0, 1), image, mask)
    )
    history = {
        "psnr": list(zip(result["history"]["iter"], result["history"]["psnr"])),
        "final_psnr": foreground_psnr,
        "final_psnr_full": float(result["psnr"]),
        "stopped_iter": int(result["iterations_run"]) - 1,
        "n_gaussians": g2d.n,
        "start_gaussians": start_budget,
        "max_gaussians": max_budget,
        "adaptive_stop_reason": result.get("adaptive_stop_reason"),
        "split_events": result["history"]["split_events"],
        "relocate_events": result["history"]["relocate_events"],
        "fit_seconds": float(result["fit_seconds"]),
        "backend": "structsplat",
        "observation_exported": observation_callback is not None,
    }
    return g2d, history


def field_to_gaussians2d(field) -> Gaussians2D:
    """Convert a live StructSplat ``GaussianField`` without a disk round-trip."""
    means = field.means.detach() + 0.5  # StructSplat uses integer-centered pixels.
    scales = (
        field.effective_scales().detach()
        if hasattr(field, "effective_scales")
        else field.log_scales.detach().exp()
    )
    angles = field.rotations.detach().reshape(-1)
    c, s = torch.cos(angles), torch.sin(angles)
    rotation = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], dim=-2)
    rs = rotation * scales[:, None, :]
    covariance = rs @ rs.transpose(-1, -2)
    eye = torch.eye(2, dtype=covariance.dtype, device=covariance.device)
    cholesky = torch.linalg.cholesky(covariance + 1e-8 * eye)
    chol = torch.stack([cholesky[:, 0, 0], cholesky[:, 1, 0], cholesky[:, 1, 1]], dim=-1)
    opacity = field.opacity_values() if hasattr(field, "opacity_values") else None
    weight = (
        torch.ones(means.shape[0], device=means.device) if opacity is None else opacity.detach()
    )
    return Gaussians2D(
        xy=means,
        chol=chol,
        color=field.colors.detach().clamp(0, 1),
        weight=weight.clamp(0, 1),
    )


def field_to_observation(
    field,
    *,
    canvas_size: tuple[int, int],
    fit_window: tuple[int, int, int, int] | None = None,
    blend_mode: str = "normalized",
    epsilon: float = 1e-8,
    sigma_cutoff: float = 3.0,
    support_fade_alpha: float = 0.0,
    aa_dilation: float = 0.0,
    view_id: str | None = None,
    n_init: int | None = None,
    producer_version: str | None = None,
    producer_source_digest: str | None = None,
    fit_config_digest: str | None = None,
) -> GaussianObservationField:
    """Freeze a live StructSplat field without clamping or materializing its semantics.

    ``canvas_size`` is ``(height, width)`` in the calibrated source view. ``fit_window`` is
    ``(x, y, width, height)`` in that canvas and binds masked/cropped fits to their actual
    renderer domain. The optional StructSplat dependency remains outside the core contract: this
    adapter uses only the live object's public tensor attributes.
    """
    canvas_height, canvas_width = canvas_size
    if fit_window is None:
        fit_window = (0, 0, canvas_width, canvas_height)
    fit_x, fit_y, _, _ = fit_window
    offset = field.means.new_tensor([fit_x + 0.5, fit_y + 0.5])
    native_means = field.means + offset
    mean_residuals = None
    if field.means.dtype == torch.float32:
        # Adding a native-resolution crop offset in float32 can erase low-order mean bits. Store
        # the exact correction in crop-local coordinates while retaining schema-v1 native means.
        mean_residuals = field.means - (native_means - offset)
    opacity = field.opacity_values() if hasattr(field, "opacity_values") else None
    amplitudes = field.means.new_ones(field.means.shape[0]) if opacity is None else opacity
    return GaussianObservationField(
        width=canvas_width,
        height=canvas_height,
        means=native_means,
        log_scales=field.log_scales,
        rotations=field.rotations.reshape(-1),
        colors=field.colors,
        amplitudes=amplitudes,
        mean_residuals=mean_residuals,
        color_grads=getattr(field, "color_grads", None),
        filter_variance=getattr(field, "filter_variance", None),
        blend_mode=blend_mode,
        epsilon=epsilon,
        sigma_cutoff=sigma_cutoff,
        support_fade_alpha=support_fade_alpha,
        aa_dilation=aa_dilation,
        view_id=view_id,
        fit_window=fit_window,
        n_init=n_init,
        producer_version=producer_version,
        producer_source_digest=producer_source_digest,
        fit_config_digest=fit_config_digest,
    )


def _structsplat_provenance(config) -> tuple[str | None, str, str]:
    """Hash the exact optional dependency sources and effective fit configuration."""
    import structsplat

    try:
        version = importlib.metadata.version("structsplat")
    except importlib.metadata.PackageNotFoundError:
        version = None
    source_root = Path(structsplat.__file__).resolve().parent
    source_hash = hashlib.sha256()
    suffixes = {".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}
    for path in sorted(
        candidate
        for candidate in source_root.rglob("*")
        if candidate.is_file() and candidate.suffix in suffixes
    ):
        source_hash.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        source_hash.update(b"\0")
        source_hash.update(path.read_bytes())
        source_hash.update(b"\0")
    config_bytes = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    return version, source_hash.hexdigest(), hashlib.sha256(config_bytes).hexdigest()
