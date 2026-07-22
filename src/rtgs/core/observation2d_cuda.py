"""CUDA backend for indexed observation-field (compact-teacher) queries (experimental).

Implements the :class:`rtgs.core.observation2d.ObservationQueryBackend` protocol on the GPU by
wrapping a CPU-built :class:`GaussianObservationIndex`: the exact CSR arrays and the field's
derived component tensors (conics, support centers/radii, precomputed cos/sin and clamped
color scales) are uploaded verbatim, and a one-thread-per-point kernel binary-searches the
point's tile and accumulates its CSR row sequentially in ascending component order — the same
canonical order the CPU pair stream evaluates, with no atomics, so results are deterministic
across runs. The CPU index remains the correctness oracle; parity tests live in
``tests/test_observation2d_cuda.py`` and self-skip without a GPU.

This path is inference-only (queries with ``requires_grad`` inputs are rejected); gradient
work stays on the CPU/torch path. Accepts CPU or CUDA query points and returns results on the
caller's device, so it drops into ``score_world_points``/``CompactCarveInitializer`` through
their existing ``backends`` parameter without any pipeline changes.

Nothing here imports CUDA at module import time (hard rule 1); the extension is built lazily
on first use and every entry point fails with an actionable error on CPU-only machines.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch

from rtgs.core.observation2d import (
    GaussianObservationIndex,
    ObservationQuery,
)

_EXT = None


def _load_extension():
    global _EXT
    if _EXT is not None:
        return _EXT
    if not torch.cuda.is_available():
        raise RuntimeError(
            "the rtgs observation-query CUDA backend requires torch.cuda.is_available()"
        )
    try:
        from torch.utils.cpp_extension import load
    except Exception as exc:  # pragma: no cover - depends on local torch install
        raise RuntimeError(
            "the rtgs observation-query CUDA backend requires torch.utils.cpp_extension"
        ) from exc

    root = Path(__file__).resolve().parent / "cuda"
    sources = [str(root / "observation2d_ext.cpp"), str(root / "observation2d_ext.cu")]
    try:
        _EXT = load(
            name="rtgs_observation2d_ext",
            sources=sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=os.environ.get("RTGS_CUDA_VERBOSE", "0") == "1",
        )
    except Exception as exc:  # pragma: no cover - toolchain/environment dependent
        raise RuntimeError(
            "the rtgs observation-query CUDA extension failed to build or load. "
            "Check CUDA_HOME, nvcc, PyTorch CUDA version, and libstdc++ compatibility. "
            f"Original error: {exc}"
        ) from exc
    return _EXT


class GaussianObservationIndexCuda:
    """GPU implementation of the ``query``/``query_weight_sum`` backend protocol.

    Wraps an already-built CPU :class:`GaussianObservationIndex` so index construction — caps,
    tile membership, and CSR ordering — is literally the CPU code path. ``component_chunk`` is
    accepted for protocol compatibility but ignored: the kernel streams whole CSR rows per
    point, and the pair-progress counters record one chunk per query.
    """

    def __init__(self, index: GaussianObservationIndex, device: torch.device | str = "cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "the rtgs observation-query CUDA backend requires torch.cuda.is_available()"
            )
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("GaussianObservationIndexCuda requires a CUDA device")
        field = index.field
        if field.dtype != torch.float32:
            raise ValueError("the CUDA observation-query backend supports float32 fields only")
        if index.component_id_dtype != torch.int32:
            raise ValueError(
                "the CUDA observation-query backend requires int32 component ids; "
                "the int64 fallback seam is CPU-only"
            )
        self.cpu_index = index
        self.field = field
        self.tile_size = index.tile_size
        self.tiles_x = index.tiles_x
        self.tiles_y = index.tiles_y
        self.n_entries = index.n_entries
        self.estimated_entries = index.estimated_entries
        self.max_candidates = index.max_candidates
        self.max_query_pairs = index.max_query_pairs
        self.component_id_dtype = index.component_id_dtype
        self.payload_bytes = index.payload_bytes

        self._tile_keys = index.tile_keys.to(self.device)
        self._tile_offsets = index.tile_offsets.to(self.device)
        self._component_ids = index.component_ids.to(self.device)

        # Displacements mirror _paired_displacements: legacy fields subtract the stored means
        # directly; mean-residual fields subtract local means from origin-shifted coordinates.
        if field.mean_residuals is None:
            query_means = field.means
            self._off = (0.0, 0.0)
        else:
            query_means = field.local_means()
            fit_x, fit_y, _, _ = field.fit_window
            self._off = (fit_x + 0.5, fit_y + 0.5)
        self._query_means = query_means.to(self.device).contiguous()
        self._conics = field.conics().to(self.device).contiguous()
        self._amplitudes = field.amplitudes.to(self.device).contiguous()
        self._colors = field.colors.to(self.device).contiguous()
        self._support_centers = field.support_centers().to(self.device).contiguous()
        self._support_radii = field.radii().to(torch.float32).to(self.device).contiguous()
        # cos/sin and clamped color scales are precomputed with torch on the CPU so their
        # values match the reference bit-for-bit before upload.
        rotations = field.rotations
        self._rot_cs = (
            torch.stack([torch.cos(rotations), torch.sin(rotations)], dim=-1)
            .to(self.device)
            .contiguous()
        )
        self._color_scales = field.color_scales().clamp_min(1e-6).to(self.device).contiguous()
        if field.color_grads is None:
            self._color_grads = torch.empty(0, 6, device=self.device)
        else:
            self._color_grads = field.color_grads.reshape(-1, 6).to(self.device).contiguous()
        fit_x, fit_y, fit_width, fit_height = field.fit_window
        self._fit_bounds = (
            float(fit_x),
            float(fit_y),
            float(fit_x + fit_width),
            float(fit_y + fit_height),
        )
        if field.support_fade_alpha > 0.0:
            self._fade_floor = field.support_fade_alpha * math.exp(-0.5 * field.sigma_cutoff**2)
        else:
            self._fade_floor = 0.0
        self._normalize = field.blend_mode == "normalized"

        # Progress counters, mirroring the CPU index (one chunk per query on this backend).
        self.total_pairs_evaluated = 0
        self.total_query_points = 0
        self.peak_pair_chunk = 0
        self.last_pair_chunk = 0

    @classmethod
    def from_field(
        cls,
        field,
        tile_size: int = 16,
        *,
        max_entries: int | None = GaussianObservationIndex.DEFAULT_MAX_ENTRIES,
        max_candidates: int | None = GaussianObservationIndex.DEFAULT_MAX_CANDIDATES,
        max_query_pairs: int | None = GaussianObservationIndex.DEFAULT_MAX_QUERY_PAIRS,
        device: torch.device | str = "cuda",
    ) -> GaussianObservationIndexCuda:
        """Build the CPU CSR index (enforcing its caps), then wrap it for GPU queries."""
        index = GaussianObservationIndex(
            field,
            tile_size=tile_size,
            max_entries=max_entries,
            max_candidates=max_candidates,
            max_query_pairs=max_query_pairs,
        )
        return cls(index, device=device)

    def _validate_xy(self, xy: torch.Tensor) -> torch.Tensor:
        xy = torch.as_tensor(xy, dtype=torch.float32)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (S,2)")
        if xy.requires_grad:
            raise RuntimeError(
                "the CUDA observation-query backend is inference-only; "
                "use the CPU index for gradient queries"
            )
        if not bool(torch.isfinite(xy).all()):
            raise ValueError("xy must be finite")
        return xy

    def _run(self, xy: torch.Tensor, want_color: bool):
        ext = _load_extension()
        xy_dev = xy.to(self.device).contiguous()
        fit_x0, fit_y0, fit_x1, fit_y1 = self._fit_bounds
        color, numerator, weight_sum = ext.query(
            xy_dev,
            self._tile_keys,
            self._tile_offsets,
            self._component_ids,
            self._query_means,
            self._conics,
            self._amplitudes,
            self._colors,
            self._support_centers,
            self._support_radii,
            self._rot_cs,
            self._color_scales,
            self._color_grads,
            self.tiles_x,
            self.tile_size,
            self._off[0],
            self._off[1],
            fit_x0,
            fit_y0,
            fit_x1,
            fit_y1,
            self._fade_floor,
            self._normalize,
            self.field.epsilon,
            want_color,
        )
        self._update_counters(xy_dev)
        return color, numerator, weight_sum, xy_dev

    def _update_counters(self, xy_dev: torch.Tensor) -> None:
        fit_x0, fit_y0, fit_x1, fit_y1 = self._fit_bounds
        valid = (
            (xy_dev[:, 0] >= fit_x0)
            & (xy_dev[:, 0] < fit_x1)
            & (xy_dev[:, 1] >= fit_y0)
            & (xy_dev[:, 1] < fit_y1)
        )
        if not self._tile_keys.numel() or not bool(valid.any()):
            self.last_pair_chunk = 0
            return
        valid_xy = xy_dev[valid]
        keys = (
            torch.floor(valid_xy[:, 1] / self.tile_size).long() * self.tiles_x
            + torch.floor(valid_xy[:, 0] / self.tile_size).long()
        )
        position = torch.searchsorted(self._tile_keys, keys)
        clamped = position.clamp_max(self._tile_keys.numel() - 1)
        found = (position < self._tile_keys.numel()) & (self._tile_keys[clamped] == keys)
        rows = position[found]
        pairs = int((self._tile_offsets[rows + 1] - self._tile_offsets[rows]).sum())
        self.total_query_points += int(found.sum())
        self.total_pairs_evaluated += pairs
        self.last_pair_chunk = pairs
        self.peak_pair_chunk = max(self.peak_pair_chunk, pairs)

    def query(self, xy: torch.Tensor, component_chunk: int = 4096) -> ObservationQuery:
        """Query colors on the GPU; results return on the caller's device."""
        if component_chunk <= 0:
            raise ValueError("component_chunk must be positive")
        xy = self._validate_xy(xy)
        target = xy.device
        color, numerator, weight_sum, xy_dev = self._run(xy, want_color=True)
        fit_x0, fit_y0, fit_x1, fit_y1 = self._fit_bounds
        valid = (
            (xy_dev[:, 0] >= fit_x0)
            & (xy_dev[:, 0] < fit_x1)
            & (xy_dev[:, 1] >= fit_y0)
            & (xy_dev[:, 1] < fit_y1)
        )
        return ObservationQuery(
            color=color.to(target),
            numerator=numerator.to(target),
            weight_sum=weight_sum.to(target),
            valid=valid.to(target),
        )

    def query_weight_sum(self, xy: torch.Tensor, component_chunk: int = 4096) -> torch.Tensor:
        """Query only the normalized-renderer denominator on the GPU."""
        if component_chunk <= 0:
            raise ValueError("component_chunk must be positive")
        xy = self._validate_xy(xy)
        target = xy.device
        _, _, weight_sum, _ = self._run(xy, want_color=False)
        return weight_sum.to(target)
