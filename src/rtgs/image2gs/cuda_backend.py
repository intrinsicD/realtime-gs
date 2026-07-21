"""CUDA backend for the batched stage-1 2D splatting renderer (experimental).

Owns a small PyTorch CUDA extension (JIT-compiled from ``rtgs/image2gs/cuda/``) that matches
``rtgs.image2gs.renderer2d``'s additive accumulated compositor: half-pixel centers, hard
Mahalanobis cutoff, ``out += weight * color * exp(-0.5 q)``. The pure-torch renderer remains
the correctness anchor; this path is opt-in via ``renderer="cuda"`` and must pass the parity
tests in ``tests/test_renderer2d_cuda.py`` on a GPU box before any default changes.

Nothing here imports CUDA at module import time (hard rule 1); the extension is built lazily
on first use and every entry point fails with an actionable error on CPU-only machines.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from rtgs.core.gaussians2d import chol_covariance, chol_inverse_covariance
from rtgs.image2gs.renderer2d import _CUTOFF, _support_rects

_EXT = None


def _load_extension():
    global _EXT
    if _EXT is not None:
        return _EXT
    if not torch.cuda.is_available():
        raise RuntimeError("the rtgs stage-1 CUDA renderer requires torch.cuda.is_available()")
    try:
        from torch.utils.cpp_extension import load
    except Exception as exc:  # pragma: no cover - depends on local torch install
        raise RuntimeError(
            "the rtgs stage-1 CUDA renderer requires torch.utils.cpp_extension"
        ) from exc

    root = Path(__file__).resolve().parent / "cuda"
    sources = [str(root / "renderer2d_ext.cpp"), str(root / "renderer2d_ext.cu")]
    try:
        _EXT = load(
            name="rtgs_renderer2d_ext",
            sources=sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=os.environ.get("RTGS_CUDA_VERBOSE", "0") == "1",
        )
    except Exception as exc:  # pragma: no cover - toolchain/environment dependent
        raise RuntimeError(
            "the rtgs stage-1 CUDA renderer extension failed to build or load. "
            "Check CUDA_HOME, nvcc, PyTorch CUDA version, and libstdc++ compatibility. "
            f"Original error: {exc}"
        ) from exc
    return _EXT


class _Render2DBatched(Function):
    @staticmethod
    def forward(ctx, xy, conics, colors, weights, rects, view_index, n_views, height, width):
        ext = _load_extension()
        xy = xy.contiguous()
        conics = conics.contiguous()
        colors = colors.contiguous()
        weights = weights.contiguous()
        rects = rects.contiguous()
        view_index = view_index.contiguous()
        out, den = ext.forward(
            xy,
            conics,
            colors,
            weights,
            rects,
            view_index,
            int(n_views),
            int(height),
            int(width),
            float(_CUTOFF),
        )
        ctx.save_for_backward(xy, conics, colors, weights, rects, view_index)
        ctx.dims = (int(n_views), int(height), int(width))
        return out, den

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out, grad_den):
        xy, conics, colors, weights, rects, view_index = ctx.saved_tensors
        n_views, height, width = ctx.dims
        need_xy, need_conics, need_colors, need_weights = ctx.needs_input_grad[:4]
        if not (need_xy or need_conics or need_colors or need_weights):
            return (None,) * 9
        if grad_out is None:
            grad_out = xy.new_zeros((n_views, height, width, 3))
        if grad_den is None:
            grad_den = xy.new_zeros((n_views, height, width))
        ext = _load_extension()
        grad_xy, grad_conics, grad_colors, grad_weights = ext.backward(
            grad_out.contiguous(),
            grad_den.contiguous(),
            xy,
            conics,
            colors,
            weights,
            rects,
            view_index,
            n_views,
            height,
            width,
            float(_CUTOFF),
        )
        return (
            grad_xy if need_xy else None,
            grad_conics if need_conics else None,
            grad_colors if need_colors else None,
            grad_weights if need_weights else None,
            None,
            None,
            None,
            None,
            None,
        )


def render_batched_cuda(
    xy: torch.Tensor,
    chol: torch.Tensor,
    color: torch.Tensor,
    weight: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Render (B, N) stacked gaussians to (B, H, W, 3) with the CUDA extension.

    Shapes are validated by the ``renderer2d`` caller; this checks device and dtype only.
    Gradients flow to ``xy``/``color``/``weight`` directly and to ``chol`` through the
    differentiable conic construction. Accumulation uses atomics and is not bit-exact
    across runs (parity tests use tolerances, like the gsplat backend).
    """
    if not xy.is_cuda:
        raise RuntimeError(
            "the rtgs stage-1 CUDA renderer requires CUDA tensors; "
            "move the fit to a CUDA device or use renderer='torch'"
        )
    if any(t.dtype != torch.float32 for t in (xy, chol, color, weight)):
        raise RuntimeError("the rtgs stage-1 CUDA renderer supports float32 tensors only")
    n_views, n = xy.shape[:2]
    xy_flat = xy.reshape(-1, 2)
    inv_cov = chol_inverse_covariance(chol).reshape(-1, 2, 2)
    conics = torch.stack([inv_cov[:, 0, 0], inv_cov[:, 0, 1], inv_cov[:, 1, 1]], dim=-1)
    with torch.no_grad():
        cov = chol_covariance(chol).reshape(-1, 2, 2)
        x0, x1, y0, y1, _, counts = _support_rects(xy_flat.detach(), cov, height, width)
        x1 = torch.where(counts > 0, x1, x0 - 1)
        rects = torch.stack([x0, x1, y0, y1], dim=-1).int()
        view_index = torch.arange(n_views, device=xy.device, dtype=torch.int32).repeat_interleave(n)
    out, _ = _Render2DBatched.apply(
        xy_flat,
        conics,
        color.reshape(-1, 3),
        weight.reshape(-1),
        rects,
        view_index,
        n_views,
        height,
        width,
    )
    return out
