"""Rasterizer protocol and backend registry.

The reference implementation (`torch_ref`) defines the semantics; fast backends
(`gsplat`) must match it (parity tests marked `cuda`). Pipeline code only ever talks to
this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.sh import DEFAULT_SMU1_MU

DEFAULT_VISIBILITY_MARGIN_SIGMA = 3.0


def _validate_visibility_margin_sigma(value: float) -> float:
    """Normalize the Torch coarse-culling margin and reject invalid values."""
    margin = float(value)
    if not torch.isfinite(torch.tensor(margin)) or margin <= 0.0:
        raise ValueError("visibility_margin_sigma must be finite and positive")
    return margin


@dataclass
class SHColorDiagnostics:
    """Visible SH colors before/after activation, retained for gradient audits."""

    preactivation: torch.Tensor  # (V,3), shifted SH color before the floor
    activated: torch.Tensor  # (V,3), color consumed by compositing
    gaussian_indices: torch.Tensor  # (V,), rows in the input Gaussian set


@dataclass
class KernelSupportDiagnostics:
    """Per-chunk kernel inputs/outputs retained until one training backward pass.

    The reference renderer is row-chunked, so preserving chunk boundaries avoids an
    outcome-sized concatenation in the forward pass.  The trainer reduces these tensors
    immediately after backward and clears both lists.
    """

    q_chunks: list[torch.Tensor]  # each (P,V), squared Mahalanobis distance
    kernel_chunks: list[torch.Tensor]  # each (P,V), kernel consumed by compositing
    gaussian_indices: torch.Tensor  # (V,), rows in the input Gaussian set


@dataclass
class RenderOutput:
    """Result of rasterizing one view.

    ``depth`` is alpha-weighted accumulated depth (divide by ``alpha`` for expected
    depth). ``means2d`` is the projected-center tensor participating in the autograd
    graph — after ``loss.backward()`` its ``.grad`` holds screen-space positional
    gradients (the densification signal); ``visible`` maps its rows to gaussian indices.
    """

    color: torch.Tensor  # (H, W, 3)
    alpha: torch.Tensor  # (H, W)
    depth: torch.Tensor  # (H, W)
    means2d: torch.Tensor | None = None  # (V,2) or (1,N,2), retained screen centers
    visible: torch.Tensor | None = None  # (V,) indices into the input gaussian set
    sh_color_diagnostics: SHColorDiagnostics | None = None
    kernel_support_diagnostics: KernelSupportDiagnostics | None = None
    # Backend-native metadata used by optional optimization strategies. Pipeline code must
    # otherwise remain backend-agnostic; torch_ref leaves this as None.
    strategy_info: dict | None = None


class Rasterizer(Protocol):
    """Renders a 3D gaussian set from a camera."""

    def render(
        self,
        gaussians: Gaussians3D,
        camera: Camera,
        background: torch.Tensor | None = None,
        sh_degree: int | None = None,
    ) -> RenderOutput:
        """Rasterize; ``sh_degree`` limits SH evaluation (None = use all bands)."""
        ...


def get_rasterizer(
    name: str = "auto",
    device: torch.device | str | None = None,
    *,
    packed: bool = False,
    absgrad: bool = False,
    antialiased: bool = False,
    sh_color_activation: str = "hard",
    sh_smu1_mu: float = DEFAULT_SMU1_MU,
    collect_sh_color_diagnostics: bool = False,
    kernel_support_mode: str = "hard",
    collect_kernel_support_diagnostics: bool = False,
    visibility_margin_sigma: float = DEFAULT_VISIBILITY_MARGIN_SIGMA,
) -> Rasterizer:
    """Return a rasterizer backend: 'torch' (reference), 'gsplat' (CUDA), or 'auto'.

    'auto' picks gsplat when both the package and a CUDA device are available,
    otherwise the reference implementation. Supplying ``device`` also prevents a
    CUDA-capable host from selecting gsplat for explicitly CPU-resident data.
    """
    if name == "auto":
        requested = None if device is None else torch.device(device)
        wants_cuda = requested is None or requested.type == "cuda"
        name = "gsplat" if wants_cuda and _gsplat_available() else "torch"
    visibility_margin_sigma = _validate_visibility_margin_sigma(visibility_margin_sigma)
    if name == "torch":
        from rtgs.render.torch_ref import TorchRasterizer

        return TorchRasterizer(
            sh_color_activation=sh_color_activation,
            sh_smu1_mu=sh_smu1_mu,
            collect_sh_color_diagnostics=collect_sh_color_diagnostics,
            kernel_support_mode=kernel_support_mode,
            collect_kernel_support_diagnostics=collect_kernel_support_diagnostics,
            visibility_margin_sigma=visibility_margin_sigma,
        )
    if name == "gsplat":
        from rtgs.render.gsplat_backend import GsplatRasterizer

        return GsplatRasterizer(
            packed=packed,
            absgrad=absgrad,
            antialiased=antialiased,
            sh_color_activation=sh_color_activation,
            sh_smu1_mu=sh_smu1_mu,
            collect_sh_color_diagnostics=collect_sh_color_diagnostics,
            kernel_support_mode=kernel_support_mode,
            collect_kernel_support_diagnostics=collect_kernel_support_diagnostics,
            visibility_margin_sigma=visibility_margin_sigma,
        )
    raise ValueError(f"unknown rasterizer '{name}' (expected 'auto', 'torch' or 'gsplat')")


def _gsplat_available() -> bool:
    try:
        import gsplat  # noqa: F401
        import torch as _torch

        return _torch.cuda.is_available()
    except ImportError:
        return False
