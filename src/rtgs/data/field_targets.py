"""Decode frozen compact 2D-Gaussian fields into dense photometric training targets.

This is the image-free target seam of the RTGS-025 main path: a compact view (2D field, camera,
optional packed alpha) is decoded on a downscaled pixel grid and used exactly like an image by an
alpha-compositing 3DGS trainer. No source photograph is read.

Decoding uses the exact tile-overlap CSR index (:class:`GaussianObservationIndex`) instead of an
all-component scan, and its CUDA twin when available; only quadrature sites inside the fitted
window are queried, and colors outside it are zero. Each target pixel averages
``supersample x supersample`` evenly spaced full-resolution sites (box prefilter), matching the
quadrature used by the RTGS-021 decoder for ``supersample=2``.

An optional on-disk cache stores decoded targets keyed by the compact view's SHA-256 and the
decode parameters, so repeated experiments skip decoding. Caches derived from private captures
belong in untracked local directories. CUDA is imported lazily (hard rule 1).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField, GaussianObservationIndex
from rtgs.data.compact_views import CompactView, PackedAlpha

DecodeBackend = Literal["auto", "cuda", "index", "reference"]

_CACHE_SCHEMA = "rtgs.field_target.v1"


@dataclass(frozen=True)
class DecodedFieldTarget:
    """A dense target decoded from one compact view on a downscaled grid."""

    view_id: str
    camera: Camera  # downscaled pinhole camera matching ``color``
    color: torch.Tensor  # (H,W,3) float32 in [0,1], zero outside the fitted window
    coverage: torch.Tensor  # (H,W) float32 fraction of sites inside the fitted window
    alpha: torch.Tensor | None  # (H,W) float32 packed-alpha fraction, or None
    downscale: int
    supersample: int
    backend: str
    source_sha256: str


def downscale_pinhole(camera: Camera, factor: int) -> Camera:
    """Exact edge-origin pinhole camera for an integer downscale that divides the canvas."""
    if type(factor) is not int or factor < 1:
        raise ValueError("downscale must be a positive integer")
    if factor == 1:
        return camera
    if camera.width % factor or camera.height % factor:
        raise ValueError(
            f"downscale {factor} must divide the canvas {camera.width}x{camera.height}"
        )
    return Camera(
        fx=camera.fx / factor,
        fy=camera.fy / factor,
        cx=camera.cx / factor,
        cy=camera.cy / factor,
        width=camera.width // factor,
        height=camera.height // factor,
        R=camera.R,
        t=camera.t,
    )


def quadrature_sites(width: int, height: int, factor: int, supersample: int) -> torch.Tensor:
    """(H*W, s*s, 2) full-resolution edge-origin sites for a ``factor``-downscaled grid.

    Rows follow the downscaled pixels in y/x order; sites within a pixel are y-major.
    """
    if supersample < 1:
        raise ValueError("supersample must be positive")
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    base = torch.stack((x, y), -1).to(torch.float32).reshape(-1, 1, 2)
    steps = (torch.arange(supersample, dtype=torch.float32) + 0.5) / supersample
    oy, ox = torch.meshgrid(steps, steps, indexing="ij")
    offsets = torch.stack((ox, oy), -1).reshape(1, -1, 2)
    return (base + offsets) * factor


def _resolve_backend(field: GaussianObservationField, backend: DecodeBackend) -> str:
    if backend not in ("auto", "cuda", "index", "reference"):
        raise ValueError(f"unknown decode backend: {backend}")
    if backend != "auto":
        return backend
    if torch.cuda.is_available() and field.dtype == torch.float32:
        return "cuda"
    return "index"


def make_query_backend(
    field: GaussianObservationField,
    backend: DecodeBackend = "auto",
    *,
    tile_size: int = 16,
    max_entries: int | None = None,
    max_candidates: int | None = None,
):
    """Return ``(query_object, resolved_name)`` implementing ``query(xy)``.

    ``max_entries``/``max_candidates`` default to unbounded here: decoding is an offline
    preprocessing step whose index size is proportional to the stored field.
    """
    name = _resolve_backend(field, backend)
    if name == "reference":
        return field, name
    source = field if field.device.type == "cpu" else field.to("cpu")
    index = GaussianObservationIndex(
        source, tile_size=tile_size, max_entries=max_entries, max_candidates=max_candidates
    )
    if name == "index":
        return index, name
    from rtgs.core.observation2d_cuda import GaussianObservationIndexCuda

    return GaussianObservationIndexCuda(index), name


def decode_field_grid(
    field: GaussianObservationField,
    *,
    downscale: int,
    supersample: int = 2,
    backend: DecodeBackend = "auto",
    query_chunk: int = 65536,
    tile_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Decode ``field`` to ``(color (H,W,3), coverage (H,W), backend)`` on the CPU.

    Sites outside the fitted window are not queried and contribute zero color.
    """
    if field.width % downscale or field.height % downscale:
        raise ValueError("downscale must divide the field canvas")
    width, height = field.width // downscale, field.height // downscale
    sites = quadrature_sites(width, height, downscale, supersample).reshape(-1, 2)
    fit_x, fit_y, fit_width, fit_height = field.fit_window
    inside = (
        (sites[:, 0] >= fit_x)
        & (sites[:, 0] < fit_x + fit_width)
        & (sites[:, 1] >= fit_y)
        & (sites[:, 1] < fit_y + fit_height)
    )
    values = torch.zeros(sites.shape[0], 3, dtype=torch.float32)
    selected = torch.nonzero(inside).squeeze(1)
    query_backend, name = make_query_backend(field, backend, tile_size=tile_size)
    device = field.device if name == "reference" else torch.device("cpu")
    with torch.no_grad():
        for chunk in selected.split(query_chunk):
            xy = sites[chunk].to(device=device, dtype=field.dtype)
            values[chunk] = query_backend.query(xy).color.to("cpu", torch.float32)
    samples = supersample * supersample
    color = values.clamp(0.0, 1.0).reshape(height, width, samples, 3).mean(2)
    coverage = inside.to(torch.float32).reshape(height, width, samples).mean(2)
    return color, coverage, name


def decode_alpha_grid(
    alpha: PackedAlpha,
    canvas_size: tuple[int, int],
    *,
    downscale: int,
    supersample: int = 2,
) -> torch.Tensor:
    """Fraction of quadrature sites inside the packed alpha, on the downscaled grid."""
    canvas_height, canvas_width = canvas_size
    full = alpha.full_mask(canvas_size)
    width, height = canvas_width // downscale, canvas_height // downscale
    sites = quadrature_sites(width, height, downscale, supersample).reshape(-1, 2)
    columns = sites[:, 0].floor().long().clamp(0, canvas_width - 1)
    rows = sites[:, 1].floor().long().clamp(0, canvas_height - 1)
    samples = supersample * supersample
    return full[rows, columns].to(torch.float32).reshape(height, width, samples).mean(2)


def _cache_path(cache_dir: Path, view: CompactView, downscale: int, supersample: int) -> Path:
    key = json.dumps(
        {
            "schema": _CACHE_SCHEMA,
            "sha256": view.sha256,
            "downscale": downscale,
            "supersample": supersample,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    return cache_dir / f"{view.view_id}_{digest}.pt"


def decode_compact_view(
    view: CompactView,
    *,
    downscale: int,
    supersample: int = 2,
    backend: DecodeBackend = "auto",
    cache_dir: str | Path | None = None,
    query_chunk: int = 65536,
) -> DecodedFieldTarget:
    """Decode one compact view (field, camera, optional alpha) into a dense target.

    With ``cache_dir`` a previously decoded target for the same view bytes and decode
    parameters is loaded instead of recomputed. The cache key does not include the backend:
    CUDA and CPU index results agree to float32 rounding.
    """
    field = view.observation
    if (field.width, field.height) != (view.camera.width, view.camera.height):
        raise ValueError("compact view camera and field canvas differ")
    camera = downscale_pinhole(view.camera, downscale)
    path = None
    if cache_dir is not None:
        path = _cache_path(Path(cache_dir), view, downscale, supersample)
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["sha256"] != view.sha256:
                raise RuntimeError(f"field target cache mismatch: {path}")
            return DecodedFieldTarget(
                view_id=view.view_id,
                camera=camera,
                color=payload["color"],
                coverage=payload["coverage"],
                alpha=payload["alpha"],
                downscale=downscale,
                supersample=supersample,
                backend=payload["backend"],
                source_sha256=view.sha256,
            )
    color, coverage, name = decode_field_grid(
        field,
        downscale=downscale,
        supersample=supersample,
        backend=backend,
        query_chunk=query_chunk,
    )
    alpha = None
    if view.alpha is not None:
        alpha = decode_alpha_grid(
            view.alpha, (field.height, field.width), downscale=downscale, supersample=supersample
        )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "schema": _CACHE_SCHEMA,
                "sha256": view.sha256,
                "color": color,
                "coverage": coverage,
                "alpha": alpha,
                "backend": name,
            },
            temporary,
        )
        temporary.replace(path)
    return DecodedFieldTarget(
        view_id=view.view_id,
        camera=camera,
        color=color,
        coverage=coverage,
        alpha=alpha,
        downscale=downscale,
        supersample=supersample,
        backend=name,
        source_sha256=view.sha256,
    )
