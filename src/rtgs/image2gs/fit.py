"""Fit a set of 2D gaussians to an image by gradient descent (stage 1).

Recipe distilled from GaussianImage (ECCV 2024) and Image-GS (SIGGRAPH 2025), see
docs/RESEARCH.md §2: Cholesky covariance parametrization, accumulated-summation blending,
plain L2 loss (best PSNR per their ablations), positions initialized by sampling the image
gradient magnitude mixed with a uniform floor.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.metrics import masked_psnr, psnr
from rtgs.image2gs.renderer2d import render_gaussians_2d

if TYPE_CHECKING:
    from rtgs.core.observation2d import GaussianObservationField

# Minimum Cholesky diagonal in pixels; keeps covariances well-conditioned.
_MIN_DIAG = 0.3
_CURRENT_APPEARANCE = "weight_color_9p"
_CANDIDATE_APPEARANCE = "unit_weight_bounded_8p"
_APPEARANCE_PARAMETERIZATIONS = {_CURRENT_APPEARANCE, _CANDIDATE_APPEARANCE}


@dataclass
class FitConfig:
    """Hyperparameters for per-image 2D gaussian fitting."""

    # Initial count (and the fixed count for the native backend). StructSplat can grow this with
    # convergence-aware density control up to the independently configurable maximum.
    n_gaussians: int = 640
    max_gaussians: int | None = 5_000
    iterations: int = 300
    backend: str = "native"  # native or structsplat (optional MIT dependency)
    adaptive_density: bool = True
    growth_waves: int = 5
    relocate_fraction: float = 0.0
    structsplat_renderer: str = "auto"
    # Renderer for the native backend: "torch" (reference), "cuda" (experimental extension,
    # CUDA tensors only), or "auto" (cuda when the images are on a CUDA device).
    native_renderer: str = "torch"
    # Fit all views of a scene jointly in one fused optimization (native backend only; same
    # per-view initialization seeds and loss as the serial path, batched renders). Opt-in:
    # serial per-view fitting remains the default and the preregistered-harness semantics.
    batch_views: bool = False
    lr: float = 1e-2
    # Fraction of position-sampling probability taken from the gradient magnitude; the
    # rest is uniform (Image-GS uses a 0.3 uniform floor).
    grad_init_mix: float = 0.7
    row_chunk: int = 64
    log_every: int = 50
    # Convergence-based early stopping (step 1: "fit until convergence"). PSNR is checked
    # every ``convergence_check_every`` iterations; if the best PSNR has not improved by
    # ``convergence_tol`` dB for ``convergence_patience`` consecutive checks, fitting stops.
    # patience=0 disables early stopping (fixed ``iterations``).
    convergence_patience: int = 0
    convergence_tol: float = 0.05
    convergence_check_every: int = 25
    # Research seam for comparing the established product gauge with an identifiable,
    # bounded RGB amplitude. The production default deliberately remains unchanged.
    appearance_parameterization: str = _CURRENT_APPEARANCE
    # Benchmark-only mechanism control. Geometry is omitted from the optimizer when enabled.
    freeze_geometry: bool = False
    # Fixed-capacity pool + free list (opt-in, default off; see rtgs.image2gs.pool). Preallocates
    # ``pool_capacity`` rows once and recycles parked rows via periodic triage (park lowest-weight,
    # spawn at residual peaks) without reallocating the optimizer. Native backend + default
    # appearance only; ``pool_capacity=None`` derives from ``max_gaussians`` or ``2*n_gaussians``.
    pool: bool = False
    pool_capacity: int | None = None
    pool_triage_every: int = 50
    pool_prune_count: int = 32
    pool_spawn_count: int = 32
    pool_min_live: int = 1


@dataclass(frozen=True)
class NativeFitDiagnostic:
    """Detached snapshot of one event in a native fit.

    ``initial`` is the step-zero checkpoint. ``pre_update`` is emitted after backward and
    before Adam for a state with ``step`` completed updates. ``post_update`` is emitted after
    Adam and the scheduler. Requested positive steps additionally emit ``checkpoint`` with a
    freshly built and rendered post-update state. All tensors and nested optimizer containers
    are cloned, and the callback runs behind the saved/restored CPU torch RNG state used by this
    CPU benchmark seam.
    """

    event: str
    step: int
    appearance_parameterization: str
    geometry_frozen: bool
    initial_gaussians: Gaussians2D | None
    initializer_rng_state_before: torch.Tensor | None
    initializer_rng_state_after: torch.Tensor | None
    raw_parameters: dict[str, torch.Tensor]
    gradients: dict[str, torch.Tensor | None]
    gaussians: Gaussians2D
    target: torch.Tensor
    rendered: torch.Tensor | None
    loss: torch.Tensor | None
    optimizer_param_names: tuple[str, ...]
    optimizer_state: dict[str, dict[str, Any]]
    optimizer_param_groups: tuple[dict[str, Any], ...]
    optimizer_defaults: dict[str, Any]
    scheduler_state: dict[str, Any]
    lr_used: float | None
    next_lr: float


NativeFitDiagnosticCallback = Callable[[NativeFitDiagnostic], None]
ObservationCallback = Callable[["GaussianObservationField"], None]


class NonFiniteRawParameterError(ValueError):
    """Non-finite native raw parameter with detached evidence for fail-closed callers."""

    def __init__(self, parameter_name: str, row_indices: list[int], raw_tensor: torch.Tensor):
        self.parameter_name = parameter_name
        self.row_indices = tuple(row_indices)
        self.raw_tensor = raw_tensor.detach().clone()
        super().__init__(f"non-finite raw rows in {parameter_name}: {row_indices}")


def _softplus_inv(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def _gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    """(H,W) mean-channel finite-difference gradient magnitude."""
    gray = image.mean(dim=-1)
    gx = torch.zeros_like(gray)
    gy = torch.zeros_like(gray)
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
    gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
    return torch.sqrt(gx**2 + gy**2)


def init_gaussians_2d(
    image: torch.Tensor,
    n: int,
    grad_mix: float = 0.7,
    generator: torch.Generator | None = None,
) -> Gaussians2D:
    """Initialize N gaussians: positions ~ gradient magnitude + uniform, colors sampled
    from the image, isotropic scale from the average area budget per gaussian."""
    h, w = image.shape[:2]
    device = image.device
    grad = _gradient_magnitude(image).reshape(-1)
    uniform = torch.ones_like(grad) / grad.numel()
    prob = grad_mix * grad / grad.sum().clamp_min(1e-8) + (1 - grad_mix) * uniform
    idx = torch.multinomial(prob, n, replacement=True, generator=generator)
    yy = (idx // w).float() + 0.5
    xx = (idx % w).float() + 0.5
    xy = torch.stack([xx, yy], dim=-1)
    # Jitter within the pixel to break ties from replacement sampling.
    xy = xy + (torch.rand(n, 2, generator=generator, device=device) - 0.5)
    xy[:, 0] = xy[:, 0].clamp(0.0, w - 1e-3)
    xy[:, 1] = xy[:, 1].clamp(0.0, h - 1e-3)

    scale0 = max((h * w / n) ** 0.5 * 0.6, _MIN_DIAG + 0.1)
    chol = torch.zeros(n, 3, device=device, dtype=image.dtype)
    chol[:, 0] = scale0
    chol[:, 2] = scale0

    color = image.reshape(-1, 3)[idx].clone()
    weight = torch.full((n,), 0.5, device=device, dtype=image.dtype)
    return Gaussians2D(xy=xy, chol=chol, color=color, weight=weight)


def fit_image(
    image: torch.Tensor,
    config: FitConfig | None = None,
    seed: int | None = None,
    mask: torch.Tensor | None = None,
    *,
    diagnostic_callback: NativeFitDiagnosticCallback | None = None,
    diagnostic_steps: Collection[int] = (),
    observation_callback: ObservationCallback | None = None,
    observation_view_id: str | None = None,
) -> tuple[Gaussians2D, dict]:
    """Fit 2D gaussians to an (H, W, 3) image in [0,1].

    Returns the fitted (detached) gaussians and a history dict with 'psnr' (list of
    (iteration, value)) and 'final_psnr'.
    """
    config = config or FitConfig()
    _validate_fit_controls(config, diagnostic_callback, diagnostic_steps)
    if config.backend == "structsplat":
        if config.appearance_parameterization != _CURRENT_APPEARANCE:
            raise ValueError(
                "unit_weight_bounded_8p is native-only and cannot be used with StructSplat"
            )
        if config.freeze_geometry or diagnostic_callback is not None or diagnostic_steps:
            raise ValueError("native fit research controls cannot be used with StructSplat")
        from rtgs.image2gs.structsplat_backend import fit_image_structsplat

        return fit_image_structsplat(
            image,
            config,
            seed=seed,
            mask=mask,
            observation_callback=observation_callback,
            observation_view_id=observation_view_id,
        )
    if config.backend != "native":
        raise ValueError("fit backend must be 'native' or 'structsplat'")
    if observation_callback is not None or observation_view_id is not None:
        raise ValueError("lossless observation export currently requires the StructSplat backend")
    xy_offset = image.new_zeros(2)
    if mask is not None:
        image, mask, xy_offset = _crop_to_mask(image, mask)
    h, w = image.shape[:2]
    target = image
    if mask is not None:
        if mask.shape != image.shape[:2]:
            raise ValueError("mask size does not match image")
        target = image * mask.to(image).clamp(0, 1)[..., None]
    gen = None
    rng_state_before = None
    rng_state_after = None
    if seed is not None:
        gen = torch.Generator(device=image.device).manual_seed(seed)
        if diagnostic_callback is not None:
            rng_state_before = gen.get_state().clone()

    g0 = init_gaussians_2d(target, config.n_gaussians, config.grad_init_mix, gen)
    if diagnostic_callback is not None and gen is not None:
        rng_state_after = gen.get_state().clone()

    if config.pool:
        from rtgs.image2gs.pool import fit_pooled_from_initialization

        return fit_pooled_from_initialization(image, target, g0, config, mask, xy_offset)

    return _fit_native_from_initialization(
        image,
        target,
        g0,
        config,
        mask,
        xy_offset,
        diagnostic_callback=diagnostic_callback,
        diagnostic_steps=diagnostic_steps,
        initializer_rng_state_before=rng_state_before,
        initializer_rng_state_after=rng_state_after,
    )


def fit_image_from_initialization(
    image: torch.Tensor,
    initial_gaussians: Gaussians2D,
    config: FitConfig | None = None,
    mask: torch.Tensor | None = None,
    *,
    diagnostic_callback: NativeFitDiagnosticCallback | None = None,
    diagnostic_steps: Collection[int] = (),
) -> tuple[Gaussians2D, dict]:
    """Fit through the production-native path from a supplied initialization.

    This research seam lets paired arms share one sampled initialization. With a mask,
    ``initial_gaussians`` must use the cropped fitting coordinates used by :func:`fit_image`;
    returned centers are restored to full-image coordinates as usual.
    """
    config = config or FitConfig()
    _validate_fit_controls(config, diagnostic_callback, diagnostic_steps)
    if config.backend != "native":
        raise ValueError("fit_image_from_initialization supports only the native backend")
    xy_offset = image.new_zeros(2)
    if mask is not None:
        image, mask, xy_offset = _crop_to_mask(image, mask)
    target = image
    if mask is not None:
        if mask.shape != image.shape[:2]:
            raise ValueError("mask size does not match image")
        target = image * mask.to(image).clamp(0, 1)[..., None]
    if config.pool:
        from rtgs.image2gs.pool import fit_pooled_from_initialization

        return fit_pooled_from_initialization(
            image, target, initial_gaussians, config, mask, xy_offset
        )

    return _fit_native_from_initialization(
        image,
        target,
        initial_gaussians,
        config,
        mask,
        xy_offset,
        diagnostic_callback=diagnostic_callback,
        diagnostic_steps=diagnostic_steps,
    )


def _fit_native_from_initialization(
    image: torch.Tensor,
    target: torch.Tensor,
    g0: Gaussians2D,
    config: FitConfig,
    mask: torch.Tensor | None,
    xy_offset: torch.Tensor,
    *,
    diagnostic_callback: NativeFitDiagnosticCallback | None,
    diagnostic_steps: Collection[int],
    initializer_rng_state_before: torch.Tensor | None = None,
    initializer_rng_state_after: torch.Tensor | None = None,
) -> tuple[Gaussians2D, dict]:
    """Shared implementation used by ordinary initialization and paired research arms."""
    h, w = image.shape[:2]
    _validate_initial_gaussians(g0, h, w)

    # Raw (unconstrained) parameters.
    wh = image.new_tensor([w, h])
    xy_raw = torch.logit((g0.xy / wh).clamp(1e-4, 1 - 1e-4))
    diag_raw = _softplus_inv(g0.chol[:, [0, 2]] - _MIN_DIAG)
    off_raw = g0.chol[:, 1].clone()
    color_raw = torch.logit(g0.color.clamp(1e-3, 1 - 1e-3))
    weight_raw = torch.logit(g0.weight.clamp(1e-3, 1 - 1e-3))
    if config.appearance_parameterization == _CURRENT_APPEARANCE:
        appearance_raw = {"color_raw": color_raw, "weight_raw": weight_raw}
    else:
        common_amplitude = torch.sigmoid(weight_raw)[:, None] * torch.sigmoid(color_raw)
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

    def build() -> Gaussians2D:
        _validate_raw_parameters(raw_parameters)
        diag = torch.nn.functional.softplus(diag_raw) + _MIN_DIAG
        chol = torch.stack([diag[:, 0], off_raw, diag[:, 1]], dim=-1)
        if config.appearance_parameterization == _CURRENT_APPEARANCE:
            color = torch.sigmoid(color_raw)
            weight = torch.sigmoid(weight_raw)
        else:
            color = torch.sigmoid(appearance_raw["amplitude_raw"])
            weight = torch.ones_like(color[:, 0])
        return Gaussians2D(
            xy=torch.sigmoid(xy_raw) * wh,
            chol=chol,
            color=color,
            weight=weight,
        )

    opt = torch.optim.Adam(params, lr=config.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config.iterations, eta_min=config.lr * 0.1
    )
    requested_steps = frozenset(int(step) for step in diagnostic_steps)
    if diagnostic_callback is not None:
        initial = build()
        initial_render = render_gaussians_2d(
            initial, h, w, row_chunk=config.row_chunk, renderer=config.native_renderer
        )
        initial_loss = _fit_loss(initial_render, target, mask)
        _emit_diagnostic(
            diagnostic_callback,
            event="initial",
            step=0,
            config=config,
            initial_gaussians=g0,
            initializer_rng_state_before=initializer_rng_state_before,
            initializer_rng_state_after=initializer_rng_state_after,
            raw_parameters=raw_parameters,
            built=initial,
            target=target,
            rendered=initial_render,
            loss=initial_loss,
            optimizer_param_names=optimizer_param_names,
            optimizer=opt,
            scheduler=sched,
            lr_used=None,
        )
    history: dict = {"psnr": [], "stopped_iter": config.iterations - 1}
    best_psnr = -float("inf")
    stale_checks = 0
    for it in range(config.iterations):
        opt.zero_grad()
        rendered = render_gaussians_2d(
            build(), h, w, row_chunk=config.row_chunk, renderer=config.native_renderer
        )
        loss = _fit_loss(rendered, target, mask)
        loss.backward()
        lr_used = float(opt.param_groups[0]["lr"])
        if diagnostic_callback is not None:
            _emit_diagnostic(
                diagnostic_callback,
                event="pre_update",
                step=it,
                config=config,
                initial_gaussians=None,
                initializer_rng_state_before=None,
                initializer_rng_state_after=None,
                raw_parameters=raw_parameters,
                built=build(),
                target=target,
                rendered=rendered,
                loss=loss,
                optimizer_param_names=optimizer_param_names,
                optimizer=opt,
                scheduler=sched,
                lr_used=lr_used,
            )
        opt.step()
        sched.step()
        if diagnostic_callback is not None:
            post = build()
            _emit_diagnostic(
                diagnostic_callback,
                event="post_update",
                step=it + 1,
                config=config,
                initial_gaussians=None,
                initializer_rng_state_before=None,
                initializer_rng_state_after=None,
                raw_parameters=raw_parameters,
                built=post,
                target=target,
                rendered=None,
                loss=None,
                optimizer_param_names=optimizer_param_names,
                optimizer=opt,
                scheduler=sched,
                lr_used=lr_used,
            )
            if it + 1 in requested_steps:
                checkpoint_render = render_gaussians_2d(
                    post, h, w, row_chunk=config.row_chunk, renderer=config.native_renderer
                )
                checkpoint_loss = _fit_loss(checkpoint_render, target, mask)
                _emit_diagnostic(
                    diagnostic_callback,
                    event="checkpoint",
                    step=it + 1,
                    config=config,
                    initial_gaussians=None,
                    initializer_rng_state_before=None,
                    initializer_rng_state_after=None,
                    raw_parameters=raw_parameters,
                    built=post,
                    target=target,
                    rendered=checkpoint_render,
                    loss=checkpoint_loss,
                    optimizer_param_names=optimizer_param_names,
                    optimizer=opt,
                    scheduler=sched,
                    lr_used=lr_used,
                )
        if it % config.log_every == 0 or it == config.iterations - 1:
            with torch.no_grad():
                history["psnr"].append((it, psnr(rendered.clamp(0, 1), target)))
        # Convergence check: stop once quality plateaus (step 1 "until convergence").
        if config.convergence_patience and (it + 1) % config.convergence_check_every == 0:
            with torch.no_grad():
                cur = psnr(rendered.clamp(0, 1), target)
            if cur > best_psnr + config.convergence_tol:
                best_psnr = cur
                stale_checks = 0
            else:
                stale_checks += 1
                if stale_checks >= config.convergence_patience:
                    history["stopped_iter"] = it
                    break

    result = build().detach()
    with torch.no_grad():
        final = render_gaussians_2d(
            result, h, w, row_chunk=config.row_chunk, renderer=config.native_renderer
        )
        history["final_psnr_full"] = psnr(final.clamp(0, 1), target)
        history["final_psnr"] = (
            history["final_psnr_full"]
            if mask is None
            else masked_psnr(final.clamp(0, 1), image, mask)
        )
    result.xy += xy_offset
    return result, history


def _fit_loss(
    rendered: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    if mask is None:
        return torch.nn.functional.mse_loss(rendered, target)
    weights = 0.1 + 0.9 * mask.to(rendered).clamp(0, 1)
    return (((rendered - target) ** 2) * weights[..., None]).mean()


def _validate_fit_controls(
    config: FitConfig,
    diagnostic_callback: NativeFitDiagnosticCallback | None,
    diagnostic_steps: Collection[int],
) -> None:
    if config.appearance_parameterization not in _APPEARANCE_PARAMETERIZATIONS:
        choices = ", ".join(sorted(_APPEARANCE_PARAMETERIZATIONS))
        raise ValueError(f"appearance_parameterization must be one of: {choices}")
    if config.native_renderer not in ("torch", "cuda", "auto"):
        raise ValueError("native_renderer must be 'torch', 'cuda', or 'auto'")
    steps = tuple(diagnostic_steps)
    if diagnostic_callback is None and steps:
        raise ValueError("diagnostic_steps require diagnostic_callback")
    if any(isinstance(step, bool) or not isinstance(step, int) for step in steps):
        raise ValueError("diagnostic_steps must contain integers")
    if any(step < 0 or step > config.iterations for step in steps):
        raise ValueError("diagnostic_steps must lie in [0, iterations]")
    if config.pool:
        if config.backend != "native":
            raise ValueError("pool fitting requires the native backend")
        if config.appearance_parameterization != _CURRENT_APPEARANCE:
            raise ValueError("pool fitting supports only the default appearance parameterization")
        if config.freeze_geometry:
            raise ValueError("pool fitting is incompatible with freeze_geometry")
        if diagnostic_callback is not None or steps:
            raise ValueError("pool fitting does not support native fit diagnostics")
        if config.pool_capacity is not None and config.pool_capacity < config.n_gaussians:
            raise ValueError("pool_capacity must be at least n_gaussians")
        if config.pool_min_live < 1:
            raise ValueError("pool_min_live must be at least 1")
        for name in ("pool_triage_every", "pool_prune_count", "pool_spawn_count"):
            if getattr(config, name) < 0:
                raise ValueError(f"{name} must be non-negative")


def _validate_raw_parameters(raw_parameters: dict[str, torch.Tensor]) -> None:
    for name, value in raw_parameters.items():
        finite = torch.isfinite(value)
        if bool(finite.all()):
            continue
        rows = torch.nonzero(~finite.reshape(value.shape[0], -1).all(dim=1)).flatten().tolist()
        raise NonFiniteRawParameterError(name, rows, value)


def _validate_initial_gaussians(g0: Gaussians2D, height: int, width: int) -> None:
    """Reject malformed initializer fields before clamps can hide invalid values."""
    expected_shapes = {
        "xy": (g0.n, 2),
        "chol": (g0.n, 3),
        "color": (g0.n, 3),
        "weight": (g0.n,),
    }
    for name, expected_shape in expected_shapes.items():
        value = getattr(g0, name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
            actual_shape = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
            raise ValueError(
                f"initial Gaussians2D {name} shape must be {expected_shape}, got {actual_shape}"
            )
        if not value.is_floating_point():
            raise ValueError(f"initial Gaussians2D {name} must be floating point")
        finite = torch.isfinite(value)
        if not bool(finite.all()):
            rows = torch.nonzero(~finite.reshape(g0.n, -1).all(dim=1)).flatten().tolist()
            raise ValueError(f"non-finite initial rows in {name}: {rows}")

    xy_in_bounds = (
        (g0.xy[:, 0] >= 0) & (g0.xy[:, 0] < width) & (g0.xy[:, 1] >= 0) & (g0.xy[:, 1] < height)
    )
    if not bool(xy_in_bounds.all()):
        rows = torch.nonzero(~xy_in_bounds).flatten().tolist()
        raise ValueError(f"initial Gaussians2D xy rows outside image bounds: {rows}")

    valid_diagonal = (g0.chol[:, 0] > _MIN_DIAG) & (g0.chol[:, 2] > _MIN_DIAG)
    if not bool(valid_diagonal.all()):
        rows = torch.nonzero(~valid_diagonal).flatten().tolist()
        raise ValueError(
            f"initial Gaussians2D Cholesky diagonal must exceed {_MIN_DIAG}: rows {rows}"
        )

    for name in ("color", "weight"):
        value = getattr(g0, name)
        in_range = (value >= 0) & (value <= 1)
        valid_rows = in_range.reshape(g0.n, -1).all(dim=1)
        if not bool(valid_rows.all()):
            rows = torch.nonzero(~valid_rows).flatten().tolist()
            raise ValueError(f"initial Gaussians2D {name} rows outside [0,1]: {rows}")


def _clone_diagnostic_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_diagnostic_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_diagnostic_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_diagnostic_value(item) for item in value)
    return value


def _emit_diagnostic(
    callback: NativeFitDiagnosticCallback,
    *,
    event: str,
    step: int,
    config: FitConfig,
    initial_gaussians: Gaussians2D | None,
    initializer_rng_state_before: torch.Tensor | None,
    initializer_rng_state_after: torch.Tensor | None,
    raw_parameters: dict[str, torch.Tensor],
    built: Gaussians2D,
    target: torch.Tensor,
    rendered: torch.Tensor | None,
    loss: torch.Tensor | None,
    optimizer_param_names: tuple[str, ...],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    lr_used: float | None,
) -> None:
    name_by_id = {id(raw_parameters[name]): name for name in optimizer_param_names}
    optimizer_state = {
        name: _clone_diagnostic_value(optimizer.state.get(raw_parameters[name], {}))
        for name in optimizer_param_names
    }
    param_groups = []
    for group in optimizer.param_groups:
        cloned_group = {
            key: (
                tuple(name_by_id[id(parameter)] for parameter in value)
                if key == "params"
                else _clone_diagnostic_value(value)
            )
            for key, value in group.items()
        }
        param_groups.append(cloned_group)
    snapshot = NativeFitDiagnostic(
        event=event,
        step=step,
        appearance_parameterization=config.appearance_parameterization,
        geometry_frozen=config.freeze_geometry,
        initial_gaussians=None if initial_gaussians is None else initial_gaussians.detach(),
        initializer_rng_state_before=(
            None
            if initializer_rng_state_before is None
            else initializer_rng_state_before.detach().clone()
        ),
        initializer_rng_state_after=(
            None
            if initializer_rng_state_after is None
            else initializer_rng_state_after.detach().clone()
        ),
        raw_parameters={name: value.detach().clone() for name, value in raw_parameters.items()},
        gradients={
            name: None if value.grad is None else value.grad.detach().clone()
            for name, value in raw_parameters.items()
        },
        gaussians=built.detach(),
        target=target.detach().clone(),
        rendered=None if rendered is None else rendered.detach().clone(),
        loss=None if loss is None else loss.detach().clone(),
        optimizer_param_names=tuple(optimizer_param_names),
        optimizer_state=optimizer_state,
        optimizer_param_groups=tuple(param_groups),
        optimizer_defaults=_clone_diagnostic_value(optimizer.defaults),
        scheduler_state=_clone_diagnostic_value(scheduler.state_dict()),
        lr_used=lr_used,
        next_lr=float(optimizer.param_groups[0]["lr"]),
    )
    rng_state = torch.random.get_rng_state()
    grad_enabled = torch.is_grad_enabled()
    try:
        callback(snapshot)
    finally:
        torch.random.set_rng_state(rng_state)
        torch.set_grad_enabled(grad_enabled)


def fit_views(
    images: list[torch.Tensor],
    config: FitConfig | None = None,
    seed: int = 0,
    masks: list[torch.Tensor] | None = None,
) -> tuple[list[Gaussians2D], list[dict]]:
    """Fit every view of a scene independently (embarrassingly parallel across images).

    With ``config.batch_views`` every view is fitted jointly in one fused optimization
    (identical per-view initialization seeds and loss; see ``rtgs.image2gs.batched``).
    """
    if config is not None and config.batch_views:
        from rtgs.image2gs.batched import fit_views_batched

        return fit_views_batched(images, config, seed=seed, masks=masks)
    results, histories = [], []
    for i, image in enumerate(images):
        mask = None if masks is None else masks[i]
        g, hist = fit_image(image, config, seed=seed + i, mask=mask)
        results.append(g)
        histories.append(hist)
    return results, histories


def _crop_to_mask(
    image: torch.Tensor, mask: torch.Tensor, margin_fraction: float = 0.05
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Crop a masked fit to useful pixels and return its original-image xy offset."""
    if mask.shape != image.shape[:2]:
        raise ValueError("mask size does not match image")
    foreground = torch.nonzero(mask > 0.5)
    if foreground.numel() == 0:
        raise ValueError("cannot fit an empty foreground mask")
    height, width = image.shape[:2]
    margin = max(2, round(max(height, width) * margin_fraction))
    y0 = max(0, int(foreground[:, 0].min()) - margin)
    y1 = min(height, int(foreground[:, 0].max()) + 1 + margin)
    x0 = max(0, int(foreground[:, 1].min()) - margin)
    x1 = min(width, int(foreground[:, 1].max()) + 1 + margin)
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1], image.new_tensor([x0, y0])
