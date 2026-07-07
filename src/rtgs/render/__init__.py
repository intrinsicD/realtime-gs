"""Rasterization backends behind a common protocol (torch reference / gsplat CUDA)."""

from rtgs.render.base import RenderOutput, get_rasterizer

__all__ = ["RenderOutput", "get_rasterizer"]
