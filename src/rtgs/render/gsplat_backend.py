"""CUDA rasterization via gsplat (Apache-2.0, nerfstudio-project/gsplat).

Lazy import: this module loads without gsplat installed; constructing the backend
without gsplat + CUDA raises with an actionable message. Semantics must match
``rtgs.render.torch_ref`` (parity test: tests/test_render.py::test_gsplat_parity,
marked cuda).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.sh import (
    DEFAULT_SMU1_MU,
    SH_COLOR_ACTIVATIONS,
    activate_sh_color,
    eval_sh_preactivation,
)
from rtgs.render.base import (
    DEFAULT_VISIBILITY_MARGIN_SIGMA,
    RenderOutput,
    _validate_visibility_margin_sigma,
)


class GsplatRasterizer:
    """Rasterizer backed by gsplat.rasterization (requires CUDA)."""

    def __init__(
        self,
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
    ) -> None:
        visibility_margin_sigma = _validate_visibility_margin_sigma(visibility_margin_sigma)
        if visibility_margin_sigma != DEFAULT_VISIBILITY_MARGIN_SIGMA:
            raise NotImplementedError(
                "non-default visibility margins are defined only by the torch reference backend"
            )
        if kernel_support_mode != "hard":
            raise NotImplementedError(
                "non-hard kernel support modes are defined only by the torch reference backend"
            )
        if collect_kernel_support_diagnostics:
            raise NotImplementedError(
                "kernel-support diagnostics are defined only by the torch reference backend"
            )
        try:
            import gsplat
        except ImportError as e:
            raise RuntimeError(
                "gsplat is not installed; run `pip install -e '.[cuda]'` on a CUDA machine "
                "or use get_rasterizer('torch')"
            ) from e
        if not hasattr(gsplat, "rasterization"):
            raise RuntimeError(
                "the imported gsplat package does not expose the supported rasterization API; "
                "install gsplat>=1.4 in this environment"
            )
        _remove_shadowed_gsplat_editable_finders(gsplat)
        if not torch.cuda.is_available():
            raise RuntimeError("gsplat backend requires a CUDA device; use get_rasterizer('torch')")
        if sh_color_activation not in SH_COLOR_ACTIVATIONS:
            choices = ", ".join(SH_COLOR_ACTIVATIONS)
            raise ValueError(
                f"unknown SH color activation '{sh_color_activation}' (expected {choices})"
            )
        if not torch.isfinite(torch.tensor(sh_smu1_mu)) or sh_smu1_mu <= 0:
            raise ValueError("sh_smu1_mu must be finite and positive")
        if collect_sh_color_diagnostics:
            raise NotImplementedError(
                "SH color-gradient diagnostics are currently defined only by the torch "
                "reference backend"
            )
        self.packed = packed
        self.absgrad = absgrad
        self.antialiased = antialiased
        self.sh_color_activation = sh_color_activation
        self.sh_smu1_mu = float(sh_smu1_mu)
        self.visibility_margin_sigma = visibility_margin_sigma

    def render(
        self,
        gaussians: Gaussians3D,
        camera: Camera,
        background: torch.Tensor | None = None,
        sh_degree: int | None = None,
    ) -> RenderOutput:
        """Rasterize one view with gsplat; returns the same contract as torch_ref."""
        from gsplat import rasterization

        g = gaussians
        device = g.means.device
        degree = g.sh_degree if sh_degree is None else min(sh_degree, g.sh_degree)
        viewmat = camera.viewmat.to(device)[None]
        k = camera.K.to(device)[None]
        bg = None
        if background is not None:
            bg = background.to(device)[None]

        if self.sh_color_activation == "hard":
            colors_arg = g.sh[:, : (degree + 1) ** 2, :]
            raster_sh_degree: int | None = degree
        else:
            directions = torch.nn.functional.normalize(g.means - camera.position.to(device), dim=-1)
            preactivation = eval_sh_preactivation(degree, g.sh, directions)
            activated = activate_sh_color(
                preactivation,
                self.sh_color_activation,
                smu1_mu=self.sh_smu1_mu,
            )
            colors_arg = activated
            raster_sh_degree = None

        colors, alphas, meta = rasterization(
            means=g.means,
            quats=g.quats,
            scales=g.scales,
            opacities=g.opacity,
            colors=colors_arg,
            viewmats=viewmat,
            Ks=k,
            width=camera.width,
            height=camera.height,
            sh_degree=raster_sh_degree,
            backgrounds=bg,
            render_mode="RGB+D",
            packed=self.packed,
            absgrad=self.absgrad,
            rasterize_mode="antialiased" if self.antialiased else "classic",
        )
        color = colors[0, :, :, :3]
        depth = colors[0, :, :, 3]
        alpha = alphas[0, :, :, 0]
        means2d = meta.get("means2d")
        # Keep gsplat's original tensor: the renderer graph uses this leaf directly, while a
        # post-hoc ``means2d[0]`` view does not receive ``.grad``.  DensityController handles
        # the leading camera dimension.
        if means2d is not None and means2d.requires_grad:
            means2d.retain_grad()
        radii = meta.get("radii")
        if self.packed and "gaussian_ids" in meta:
            visible = torch.unique(meta["gaussian_ids"])
        else:
            visible = None if radii is None else _visible_indices(radii)
        return RenderOutput(
            color=color,
            alpha=alpha,
            depth=depth,
            means2d=means2d,
            visible=visible,
            strategy_info=meta,
        )


def _remove_shadowed_gsplat_editable_finders(gsplat_module: object) -> tuple[str, ...]:
    """Ignore stale editable gsplat submodules that shadow the imported package.

    A system-site-packages environment can contain an older editable ``gsplat`` finder even
    when a compatible wheel wins the top-level import.  That finder is otherwise allowed to
    satisfy the wheel's missing ``gsplat.csrc`` submodule from the unrelated checkout, mixing
    two binary APIs in one process.  Remove only finders that demonstrably resolve ``csrc``
    outside the root of the already-imported package; matching editable installs are retained.
    """
    module_file = getattr(gsplat_module, "__file__", None)
    module_path = getattr(gsplat_module, "__path__", None)
    if module_file is None or module_path is None:
        return ()
    package_root = Path(module_file).resolve().parent
    removed: list[str] = []
    for finder in tuple(sys.meta_path):
        if not getattr(finder, "__module__", "").startswith("__editable___gsplat_"):
            continue
        try:
            spec = finder.find_spec("gsplat.csrc", module_path)
        except (AttributeError, ImportError, TypeError, ValueError):
            continue
        origin = None if spec is None else getattr(spec, "origin", None)
        if origin is None:
            continue
        resolved = Path(origin).resolve()
        if resolved.parent == package_root or package_root in resolved.parents:
            continue
        sys.meta_path.remove(finder)
        removed.append(str(resolved))
    return tuple(removed)


def _visible_indices(radii: torch.Tensor) -> torch.Tensor:
    """Normalize gsplat's scalar/axis radius layouts to Gaussian indices."""
    # gsplat versions expose either [C,N] scalar radii or [C,N,2] axis radii.
    r = radii[0] if radii.ndim >= 2 and radii.shape[0] == 1 else radii
    visible = (r > 0).all(-1) if r.ndim == 2 else r > 0
    return visible.nonzero(as_tuple=True)[0]
