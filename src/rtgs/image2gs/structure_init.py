"""Feature-aware, oriented 2D-gaussian initialization (torch-free, NumPy only).

A feature-aware alternative to the gradient-magnitude baseline in
``rtgs.image2gs.fit.init_gaussians_2d``. The whole pipeline is deliberately importable
without torch (init-time math is pure NumPy), following the StructSplat convention that
structure-tensor / density / sampling math never touches autograd:

    image (H,W,3) in [0,1]
      -> structure tensor  J = G_rho * (grad I grad I^T)     (energy / orientation / coherence)
      -> density PMF        from tensor energy (floor so flats still get coverage)
      -> anisotropic WSE    exactly-N blue-noise subset of density-drawn candidates
                            (Weighted Sample Elimination, Yuksel EGSR 2015; per-point radius
                             for density adaptivity + per-point metric for anisotropy)
      -> oriented covariance built from the local tensor and packed as rtgs Cholesky factors

The structure-tensor / density / WSE cores are ported and adapted from StructSplat's
``structure_tensor.py``, ``density.py`` and ``sampling.py``. Only the WSE core needed for
one-shot init is carried over (the progressive ordering and the alternative ablation samplers
are intentionally omitted).

Coordinate / packing conventions follow rtgs: ``xy`` is ``(x, y)`` in pixel coordinates with
pixel ``j`` centered at ``j + 0.5`` (so ``xy`` lands in ``[0, width) x [0, height)``), and the
covariance is stored as a packed lower-triangular Cholesky factor ``chol = (l11, l21, l22)``
with ``Sigma = L L^T`` and positive diagonal comfortably above
``rtgs.image2gs.fit._MIN_DIAG``. Build a ``rtgs.core.gaussians2d.Gaussians2D`` from the
returned ``xy`` / ``chol`` (with separately sampled color / weight) on the torch side; color is
not produced here because rtgs samples color separately.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

# Minimum Cholesky diagonal in pixels. Mirrors ``rtgs.image2gs.fit._MIN_DIAG``; redefined here
# so this module never imports the torch-dependent fit module and stays torch-free.
_MIN_DIAG = 0.3
# Safety margin kept above _MIN_DIAG for the built covariance diagonal (matches fit.py's
# ``_MIN_DIAG + 0.1`` isotropic-scale floor).
_DIAG_MARGIN = 0.1

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)  # Rec.709 luma weights
# Reference-energy median floor (StructSplat INIT-005 robustness guard): on a near-blank noisy
# image percentile(energy, 99) is itself noise-scaled, so a fraction-of-p99 reference collapses.
# Flooring at a multiple of the *median* keeps the reference meaningful; on a structured image the
# median is dominated by the flat (~0) background so the floor never binds.
_ENERGY_REF_FLOOR_MULT = 8.0


@dataclass
class StructureInitConfig:
    """Seed-independent knobs for feature-aware oriented init.

    All randomness is carried by the explicit ``rng`` passed to :func:`structure_init`; nothing
    here is a seed. Defaults are sensible for tiny-to-medium images.
    """

    # --- structure tensor -------------------------------------------------------------------
    grad_sigma: float = 1.0
    """Gaussian pre-smoothing of the luma image before differentiation (noise regularization)."""
    rho: float = 2.0
    """G_rho: Gaussian integration scale of the outer-product tensor field ``grad I grad I^T``."""
    gradient_operator: str = "central"
    """Local gradient operator: ``"central"`` (``np.gradient``) or ``"sobel"``."""

    # --- density PMF ------------------------------------------------------------------------
    density_base: float = 0.1
    """Uniform floor mixed into the density so flat regions still get some coverage (in [0,1])."""
    density_power: float = 1.0
    """Shaping exponent: density ~ ``(energy / ref) ** density_power``."""

    # --- anisotropic Weighted Sample Elimination --------------------------------------------
    sampling_mode: str = "wse"
    """Final placement: ``"wse"`` or matched ``"density"`` without sample elimination."""
    wse_oversample: float = 4.0
    """Draw ``ceil(wse_oversample * N)`` density candidates, then eliminate down to exactly N."""
    wse_alpha: float = 8.0
    """WSE crowding-weight exponent (Yuksel)."""

    # --- oriented covariance ----------------------------------------------------------------
    base_scale_mult: float = 0.6
    """Base std = ``base_scale_mult * sqrt(H*W / N)`` (matches fit.py's isotropic scale)."""
    min_axis_ratio: float = 0.25
    """Floor on the minor/major axis ratio at full coherence; caps elongation at ``1/this``."""
    anisotropy_strength: float = 1.0
    """In [0,1]; how strongly coherence drives elongation (0 keeps every gaussian isotropic)."""

    def __post_init__(self) -> None:
        if self.sampling_mode not in ("wse", "density"):
            raise ValueError(
                f"sampling_mode must be 'wse' or 'density', got {self.sampling_mode!r}"
            )
        if self.gradient_operator not in ("central", "sobel"):
            raise ValueError(
                f"gradient_operator must be 'central' or 'sobel', got {self.gradient_operator!r}"
            )
        if self.grad_sigma < 0.0 or self.rho < 0.0:
            raise ValueError("grad_sigma and rho must be >= 0")
        if not 0.0 <= self.density_base <= 1.0:
            raise ValueError(f"density_base must be in [0, 1], got {self.density_base}")
        if self.density_power <= 0.0:
            raise ValueError(f"density_power must be > 0, got {self.density_power}")
        if self.wse_oversample < 1.0:
            raise ValueError(f"wse_oversample must be >= 1, got {self.wse_oversample}")
        if self.wse_alpha <= 0.0:
            raise ValueError(f"wse_alpha must be > 0, got {self.wse_alpha}")
        if self.base_scale_mult <= 0.0:
            raise ValueError(f"base_scale_mult must be > 0, got {self.base_scale_mult}")
        if not 0.0 < self.min_axis_ratio <= 1.0:
            raise ValueError(f"min_axis_ratio must be in (0, 1], got {self.min_axis_ratio}")
        if not 0.0 <= self.anisotropy_strength <= 1.0:
            raise ValueError(
                f"anisotropy_strength must be in [0, 1], got {self.anisotropy_strength}"
            )


@dataclass
class StructureInitResult:
    """Output of :func:`structure_init`.

    ``xy`` / ``chol`` are the initializer proper; the ``energy`` / ``orientation`` /
    ``coherence`` / ``density`` maps are the intermediate per-pixel fields, exposed for tests and
    diagnostics. ``orientation`` is the *across-edge* (gradient) angle in radians; the edge
    tangent an edge gaussian elongates along is ``orientation + pi/2``.
    """

    xy: np.ndarray  # (N, 2) float32 centers in pixels, (x, y)
    chol: np.ndarray  # (N, 3) float32 packed Cholesky (l11, l21, l22), Sigma = L L^T
    energy: np.ndarray  # (H, W) float32 tensor energy lam1 + lam2 (the density signal)
    orientation: np.ndarray  # (H, W) float32 across-edge (gradient) angle, radians
    coherence: np.ndarray  # (H, W) float32 ((lam1 - lam2)/(lam1 + lam2))^2 in [0, 1]
    density: np.ndarray  # (H, W) float32 normalized sampling PMF

    @property
    def n(self) -> int:
        """Number of gaussians."""
        return int(self.xy.shape[0])


@dataclass
class _StructureTensor:
    """Per-pixel structure-tensor fields (private helper container)."""

    energy: np.ndarray  # lam1 + lam2
    angle: np.ndarray  # across-edge (gradient) direction, radians
    coherence: np.ndarray  # ((lam1 - lam2)/(lam1 + lam2))^2 in [0, 1]


__all__ = ["StructureInitConfig", "StructureInitResult", "structure_init"]


# ---------------------------------------------------------------------------------------------
# Structure tensor (ported from StructSplat structure_tensor.py, luma color space)
# ---------------------------------------------------------------------------------------------
def _to_luma(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        return img
    return img[..., :3] @ _LUMA


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def _conv1d(a: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """Separable 1D convolution along ``axis`` with reflect padding."""
    r = len(k) // 2
    pad = [(r, r) if ax == axis else (0, 0) for ax in range(a.ndim)]
    ap = np.pad(a, pad, mode="reflect")
    out = np.zeros_like(a)
    for i, w in enumerate(k):
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(i, i + a.shape[axis])
        out += w * ap[tuple(sl)]
    return out


def _conv2d(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Small reflect-padded 2D cross-correlation for fixed gradient kernels."""
    kh, kw = k.shape
    rh, rw = kh // 2, kw // 2
    ap = np.pad(a, ((rh, rh), (rw, rw)), mode="reflect")
    out = np.zeros_like(a, dtype=np.float32)
    for y in range(kh):
        for x in range(kw):
            out += np.float32(k[y, x]) * ap[y : y + a.shape[0], x : x + a.shape[1]]
    return out


def _gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return a
    k = _gaussian_kernel(sigma)
    return _conv1d(_conv1d(a, k, 0), k, 1)


def _gradients(g: np.ndarray, operator: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(iy, ix)`` gradients (d/drow, d/dcol) using a selectable operator."""
    if operator == "central":
        iy, ix = np.gradient(g)
        return iy.astype(np.float32), ix.astype(np.float32)
    # Sobel: _conv2d is a cross-correlation, so the +1 column sits at +x for d/dx to match
    # np.gradient's sign convention (the tensor J is sign-invariant either way).
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
    ky = kx.T
    return _conv2d(g, ky), _conv2d(g, kx)


def _structure_tensor(img: np.ndarray, cfg: StructureInitConfig) -> _StructureTensor:
    """Closed-form 2x2 eigen-analysis of J = G_rho * (grad I grad I^T)."""
    g = _gaussian_blur(_to_luma(img), cfg.grad_sigma)
    iy, ix = _gradients(g, cfg.gradient_operator)

    jxx = _gaussian_blur(ix * ix, cfg.rho)
    jxy = _gaussian_blur(ix * iy, cfg.rho)
    jyy = _gaussian_blur(iy * iy, cfg.rho)

    half = 0.5 * (jxx + jyy)
    diff = 0.5 * (jxx - jyy)
    r = np.sqrt(diff * diff + jxy * jxy)
    lam1 = half + r
    lam2 = np.clip(half - r, 0.0, None)

    # major-eigenvector (gradient / across-edge) direction of J
    angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    energy = lam1 + lam2
    coherence = ((lam1 - lam2) / (energy + 1e-12)) ** 2
    return _StructureTensor(
        energy=energy.astype(np.float32),
        angle=angle.astype(np.float32),
        coherence=coherence.astype(np.float32),
    )


# ---------------------------------------------------------------------------------------------
# Density PMF (ported from StructSplat density.py)
# ---------------------------------------------------------------------------------------------
def _energy_reference(energy: np.ndarray) -> float:
    """Robust reference energy: ``max(percentile99, floor_mult * median) + eps``."""
    e = np.maximum(np.asarray(energy, dtype=np.float64), 0.0)
    med = float(np.median(e)) if e.size else 0.0
    return float(max(np.percentile(e, 99.0), _ENERGY_REF_FLOOR_MULT * med)) + 1e-12


def _density_pmf(energy: np.ndarray, cfg: StructureInitConfig) -> np.ndarray:
    """Normalized per-pixel sampling PMF from tensor energy (float64, sums to 1)."""
    e = np.maximum(energy.astype(np.float64), 0.0)
    ref = _energy_reference(e)
    d = np.clip(e / ref, 0.0, 1.0) ** cfg.density_power
    d = cfg.density_base + (1.0 - cfg.density_base) * d
    s = d.sum()
    if not s > 0.0:  # degenerate (all-zero) feature with density_base == 0
        return np.full(d.shape, 1.0 / d.size, dtype=np.float64)
    return d / s


def _sample_candidates(density: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``m`` sub-pixel candidate positions ~ density. Returns (m, 2) as (x, y) float64.

    Integer pixel ``j`` is the footprint ``[j, j+1)`` (center ``j + 0.5``, rtgs convention);
    ``floor`` of a candidate recovers its source pixel for the per-candidate tensor lookups.
    """
    h, w = density.shape
    flat = density.ravel().astype(np.float64)
    total = flat.sum()
    flat = np.full(flat.shape, 1.0 / flat.size) if not total > 0.0 else flat / total
    idx = rng.choice(flat.size, size=m, replace=True, p=flat)
    iy, ix = np.divmod(idx, w)
    jitter = rng.random((m, 2))
    xs = np.clip(ix + jitter[:, 0], 0.0, w - 1e-3)
    ys = np.clip(iy + jitter[:, 1], 0.0, h - 1e-3)
    return np.stack([xs, ys], axis=1)


# ---------------------------------------------------------------------------------------------
# Anisotropic Weighted Sample Elimination (ported from StructSplat sampling.py; WSE core only)
# ---------------------------------------------------------------------------------------------
def _anisotropy_metric(angle: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """Unit-area metric tensors M (K,2,2) whose unit ball is an ellipse.

    ``angle`` is the across-edge (gradient) direction; ``ratio >= 1`` is major/minor. Across-edge
    spacing is made small and along-edge spacing large (dense across an edge, sparse along it)
    with equal ellipse area so kept counts stay comparable to the isotropic case.
    """
    ratio = np.maximum(ratio, 1.0)
    s_across = 1.0 / np.sqrt(ratio)  # small spacing across the edge
    s_along = np.sqrt(ratio)  # large spacing along the edge
    c, s = np.cos(angle), np.sin(angle)
    a_across = 1.0 / s_across**2
    a_along = 1.0 / s_along**2
    k = angle.shape[0]
    metric = np.empty((k, 2, 2), dtype=np.float64)
    metric[:, 0, 0] = a_across * c * c + a_along * s * s
    metric[:, 0, 1] = a_across * c * s - a_along * s * c
    metric[:, 1, 0] = metric[:, 0, 1]
    metric[:, 1, 1] = a_across * s * s + a_along * c * c
    return metric


def _metric_min_eigenvalue(metric: np.ndarray) -> np.ndarray:
    """Smaller eigenvalue of each symmetric 2x2 metric (vectorized)."""
    half = 0.5 * (metric[:, 0, 0] + metric[:, 1, 1])
    diff = 0.5 * (metric[:, 0, 0] - metric[:, 1, 1])
    r = np.sqrt(diff * diff + metric[:, 0, 1] ** 2)
    return half - r


def _neighbor_pairs(
    points: np.ndarray, r_i: np.ndarray, metric: np.ndarray | None, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All ordered pairs (recv, ctrb) with nonzero WSE weight, and their contributions.

    A pair contributes iff ``pair_dist(recv, ctrb) < 2*r_i[recv]`` (Euclidean, or Mahalanobis in
    the averaged pair metric). Pair discovery is vectorized over bounded grid-cell offsets; the
    per-receiver Euclidean reach bound accounts for how far the averaged-metric distance can
    undershoot the Euclidean one.
    """
    m = points.shape[0]
    if metric is None:
        reach = 2.0 * r_i
    else:
        lam_min = np.maximum(_metric_min_eigenvalue(metric), 1e-12)
        reach = 2.0 * r_i * np.minimum(np.sqrt(2.0 / lam_min), 1.0 / np.sqrt(lam_min.min()))
    reach_max = float(reach.max())

    cell = max(float(np.median(2.0 * r_i)), reach_max / 16.0)
    mn = points.min(axis=0)
    gxy = np.floor((points - mn) / cell).astype(np.int64)
    ncx = int(gxy[:, 0].max()) + 1
    ncy = int(gxy[:, 1].max()) + 1
    cid = gxy[:, 1] * ncx + gxy[:, 0]
    order = np.argsort(cid, kind="stable")
    cid_sorted = cid[order]
    occupied, starts = np.unique(cid_sorted, return_index=True)
    counts = np.diff(np.append(starts, m))
    k = int(np.ceil(reach_max / cell))

    idx = np.arange(m, dtype=np.int64)
    recv_all, ctrb_all, w_all = [], [], []
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            ring = cell * np.hypot(max(abs(dx) - 1, 0), max(abs(dy) - 1, 0))
            near = reach >= ring
            if not near.any():
                continue
            src0 = idx[near]
            gx = gxy[near, 0] + dx
            gy = gxy[near, 1] + dy
            ok = (gx >= 0) & (gx < ncx) & (gy >= 0) & (gy < ncy)
            if not ok.any():
                continue
            src = src0[ok]
            pc = gy[ok] * ncx + gx[ok]
            pos = np.searchsorted(occupied, pc)
            in_range = pos < len(occupied)
            if not in_range.any():
                continue
            pc_in = pc[in_range]
            pos_in = pos[in_range]
            matched = occupied[pos_in] == pc_in
            if not matched.any():
                continue
            src = src[in_range][matched]
            ptr = starts[pos_in[matched]]
            cnt = counts[pos_in[matched]]
            recv = np.repeat(src, cnt)
            ends = np.cumsum(cnt)
            within = np.arange(int(ends[-1])) - np.repeat(ends - cnt, cnt)
            ctrb = order[np.repeat(ptr, cnt) + within]

            dv = points[recv] - points[ctrb]
            if metric is None:
                d2 = dv[:, 0] ** 2 + dv[:, 1] ** 2
            else:
                mm = 0.5 * (metric[recv] + metric[ctrb])
                d2 = (
                    mm[:, 0, 0] * dv[:, 0] ** 2
                    + 2.0 * mm[:, 0, 1] * dv[:, 0] * dv[:, 1]
                    + mm[:, 1, 1] * dv[:, 1] ** 2
                )
            two_r = 2.0 * r_i[recv]
            keep = (d2 < two_r * two_r) & (recv != ctrb)
            if not keep.any():
                continue
            d = np.sqrt(np.maximum(d2[keep], 0.0))
            recv_all.append(recv[keep])
            ctrb_all.append(ctrb[keep])
            w_all.append((1.0 - d / two_r[keep]) ** alpha)

    if not recv_all:
        e = np.empty(0, dtype=np.int64)
        return e, e.copy(), np.empty(0, dtype=np.float64)
    return np.concatenate(recv_all), np.concatenate(ctrb_all), np.concatenate(w_all)


def _wse_eliminate(
    points: np.ndarray, n: int, r_i: np.ndarray, metric: np.ndarray | None, alpha: float
) -> np.ndarray:
    """Indices (n,) of a blue-noise subset of ``points`` via greedy weighted elimination.

    Removes the most-crowded surviving sample until exactly ``n`` remain. Deterministic given
    the candidate set (heap ties break on sample index).
    """
    m = points.shape[0]
    target_n = min(max(int(n), 0), m)
    if target_n >= m:
        return np.arange(m, dtype=np.int64)
    points = np.asarray(points, dtype=np.float64)
    r_i = np.asarray(r_i, dtype=np.float64)

    recv, ctrb, w = _neighbor_pairs(points, r_i, metric, alpha)
    weights = np.bincount(recv, weights=w, minlength=m).tolist()

    # CSR keyed by CONTRIBUTOR: removing x decrements every receiver x crowds. Plain Python
    # lists keep the sequential greedy loop free of NumPy scalar boxing.
    by_ctrb = np.argsort(ctrb, kind="stable")
    recv_of = recv[by_ctrb].tolist()
    w_of = w[by_ctrb].tolist()
    indptr = np.searchsorted(ctrb[by_ctrb], np.arange(m + 1, dtype=np.int64)).tolist()

    version = [0] * m
    alive = [True] * m
    heap = [(-weights[i], i, 0) for i in range(m)]
    heapq.heapify(heap)
    push, pop = heapq.heappush, heapq.heappop

    remaining = m
    while remaining > target_n:
        _negw, i, ver = pop(heap)
        if not alive[i] or ver != version[i]:
            continue  # stale entry
        alive[i] = False
        remaining -= 1
        for e in range(indptr[i], indptr[i + 1]):
            j = recv_of[e]
            if not alive[j]:
                continue
            weights[j] -= w_of[e]
            version[j] += 1
            push(heap, (-weights[j], j, version[j]))
    return np.nonzero(alive)[0]


# ---------------------------------------------------------------------------------------------
# Oriented covariance -> packed Cholesky
# ---------------------------------------------------------------------------------------------
def _aniso_ratio(coherence: np.ndarray, cfg: StructureInitConfig) -> np.ndarray:
    """Per-point major/minor axis ratio (>= 1) from coherence.

    ``minor/major = 1 - anisotropy_strength * (1 - min_axis_ratio) * coherence`` in
    ``[min_axis_ratio, 1]``; ``ratio = major/minor`` is thus in ``[1, 1/min_axis_ratio]``.
    """
    coh = np.clip(np.asarray(coherence, dtype=np.float64), 0.0, 1.0)
    minor_over_major = 1.0 - cfg.anisotropy_strength * (1.0 - cfg.min_axis_ratio) * coh
    return 1.0 / np.maximum(minor_over_major, cfg.min_axis_ratio)


def _oriented_chol(
    theta: np.ndarray, coherence: np.ndarray, base: float, cfg: StructureInitConfig
) -> np.ndarray:
    """Packed Cholesky (N,3) for oriented gaussians elongated ALONG ``theta`` (the tangent).

    Equal-area scaling: ``sigma_major = base*sqrt(ratio)`` along ``theta`` and
    ``sigma_minor = base/sqrt(ratio)`` across it, so ``det(Sigma) = base^4`` is ratio-invariant.
    Both Cholesky diagonal entries are >= ``sigma_minor`` >= ``_MIN_DIAG + _DIAG_MARGIN``.
    """
    ratio = _aniso_ratio(coherence, cfg)
    floor = _MIN_DIAG + _DIAG_MARGIN
    sigma_major = np.maximum(base * np.sqrt(ratio), floor)  # along theta (edge tangent)
    sigma_minor = np.maximum(base / np.sqrt(ratio), floor)  # across the edge
    var_maj = sigma_major**2
    var_min = sigma_minor**2

    c, s = np.cos(theta), np.sin(theta)
    sxx = var_maj * c * c + var_min * s * s
    sxy = (var_maj - var_min) * c * s
    l11 = np.sqrt(sxx)  # >= sigma_minor > 0
    l21 = sxy / l11
    l22 = (sigma_major * sigma_minor) / l11  # = sqrt(det)/l11 >= sigma_minor > 0
    return np.stack([l11, l21, l22], axis=1).astype(np.float32)


def _target_radius(density_at_candidate: np.ndarray, n: int) -> np.ndarray:
    """Per-candidate isotropic WSE radius from the local target areal density (Yuksel r_max).

    ``lambda_i = n * pmf_i`` is the expected kept points per unit (pixel) area at candidate ``i``;
    ``r_i = 1 / sqrt(2*sqrt(3) * lambda_i)`` is the max-Poisson-disk radius for that density, so
    denser (higher-energy) regions get smaller radii and pack more tightly.
    """
    lam = np.maximum(n * np.asarray(density_at_candidate, dtype=np.float64), 1e-12)
    return 1.0 / np.sqrt(2.0 * math.sqrt(3.0) * lam)


# ---------------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------------
def structure_init(
    image: np.ndarray,
    n: int,
    *,
    rng: np.random.Generator,
    config: StructureInitConfig | None = None,
) -> StructureInitResult:
    """Feature-aware oriented initialization of ``n`` 2D gaussians for one image.

    Fits a structure tensor, derives a density PMF from its energy, places exactly ``n`` points,
    and builds an oriented anisotropic covariance at each point (edge gaussians elongate along the
    local edge tangent). The default ``sampling_mode="wse"`` uses anisotropic Weighted Sample
    Elimination. The matched ``"density"`` control draws the identical oversampled candidate
    stream but keeps its first ``n`` points, isolating removal of WSE without changing the tensor,
    density, candidate RNG, orientation, covariance, or count. Deterministic given ``rng``;
    torch-free.

    Args:
        image: ``(H, W, 3)`` (or ``(H, W)``) float array in ``[0, 1]``.
        n: exact number of gaussians to place (> 0).
        rng: NumPy generator carrying all randomness (thread a fresh one for reproducibility).
        config: optional :class:`StructureInitConfig`; defaults are used when omitted.

    Returns:
        :class:`StructureInitResult` with ``xy`` (N,2 float32 pixel coords), ``chol`` (N,3 float32
        packed Cholesky), and the ``energy`` / ``orientation`` / ``coherence`` / ``density`` maps.
    """
    cfg = config or StructureInitConfig()
    img = np.asarray(image, dtype=np.float32)
    if img.ndim not in (2, 3):
        raise ValueError(f"image must be (H,W) or (H,W,3), got shape {img.shape}")
    if not isinstance(n, (int, np.integer)) or int(n) <= 0:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    n = int(n)
    h, w = img.shape[:2]

    tensor = _structure_tensor(img, cfg)
    density = _density_pmf(tensor.energy, cfg)  # float64 (H, W)

    m = max(int(math.ceil(cfg.wse_oversample * n)), n)
    cand = _sample_candidates(density, m, rng)  # (m, 2) (x, y)
    ix = np.clip(np.floor(cand[:, 0]).astype(np.int64), 0, w - 1)
    iy = np.clip(np.floor(cand[:, 1]).astype(np.int64), 0, h - 1)

    dens_c = density[iy, ix]
    angle_c = tensor.angle[iy, ix].astype(np.float64)
    coh_c = tensor.coherence[iy, ix].astype(np.float64)

    if cfg.sampling_mode == "wse":
        r_i = _target_radius(dens_c, n)
        metric = _anisotropy_metric(angle_c, _aniso_ratio(coh_c, cfg))
        keep = _wse_eliminate(cand, n, r_i, metric, cfg.wse_alpha)
    else:
        # Matched no-WSE control: retain the prefix of the same density-drawn oversample used by
        # the WSE arm. Candidates after N are deliberately generated but ignored so both modes
        # consume an identical candidate RNG stream and differ only in subset selection.
        keep = np.arange(n, dtype=np.int64)

    xy = cand[keep]
    xy[:, 0] = np.clip(xy[:, 0], 0.0, w - 1e-3)
    xy[:, 1] = np.clip(xy[:, 1], 0.0, h - 1e-3)

    base = cfg.base_scale_mult * math.sqrt(h * w / n)
    theta = angle_c[keep] + 0.5 * np.pi  # edge tangent = across-edge angle + 90 degrees
    chol = _oriented_chol(theta, coh_c[keep], base, cfg)

    return StructureInitResult(
        xy=xy.astype(np.float32),
        chol=chol,
        energy=tensor.energy.astype(np.float32),
        orientation=tensor.angle.astype(np.float32),
        coherence=tensor.coherence.astype(np.float32),
        density=density.astype(np.float32),
    )
