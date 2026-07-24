"""Tests for the torch-free feature-aware oriented init (rtgs.image2gs.structure_init).

Small synthetic images, seeded ``np.random.default_rng(0)``, loose floors (not snapshots):
structure-tensor orientation/energy, exact-N blue-noise density-adaptive WSE placement, and
SPD oriented Cholesky elongated along the edge tangent.
"""

from __future__ import annotations

import numpy as np
import pytest

from rtgs.image2gs.structure_init import (
    _MIN_DIAG,
    StructureInitConfig,
    StructureInitResult,
    structure_init,
)


def _vertical_edge(size: int = 48) -> np.ndarray:
    """A vertical step edge at column ``size//2`` (gradient horizontal, tangent vertical)."""
    img = np.full((size, size, 3), 0.2, dtype=np.float32)
    img[:, size // 2 :] = 0.8
    return img


def _horizontal_edge(size: int = 48) -> np.ndarray:
    """A horizontal step edge at row ``size//2`` (gradient vertical, tangent horizontal)."""
    img = np.full((size, size, 3), 0.2, dtype=np.float32)
    img[size // 2 :, :] = 0.8
    return img


def _sigma_from_chol(chol: np.ndarray) -> np.ndarray:
    """(N,2,2) covariances Sigma = L L^T from packed (l11, l21, l22)."""
    n = chol.shape[0]
    lmat = np.zeros((n, 2, 2), dtype=np.float64)
    lmat[:, 0, 0] = chol[:, 0]
    lmat[:, 1, 0] = chol[:, 1]
    lmat[:, 1, 1] = chol[:, 2]
    return lmat @ np.transpose(lmat, (0, 2, 1))


def _pixel_index(xy: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    ix = np.clip(np.floor(xy[:, 0]).astype(int), 0, size - 1)
    iy = np.clip(np.floor(xy[:, 1]).astype(int), 0, size - 1)
    return ix, iy


# ---------------------------------------------------------------------------------------------
# Structure tensor: orientation and energy
# ---------------------------------------------------------------------------------------------
def test_energy_peaks_on_edges():
    size = 48
    res = structure_init(_vertical_edge(size), 120, rng=np.random.default_rng(0))
    center = size // 2
    energy = res.energy
    edge_energy = energy[:, center - 1 : center + 1].mean()
    flat_energy = energy[:, : center // 2].mean()  # left flat region
    assert edge_energy > 10.0 * (flat_energy + 1e-8)


def test_orientation_across_edge_is_horizontal_for_vertical_edge():
    size = 48
    res = structure_init(_vertical_edge(size), 120, rng=np.random.default_rng(0))
    # High-energy (edge) pixels: across-edge angle should point along x -> cos^2 ~ 1.
    strong = res.energy > 0.25 * res.energy.max()
    assert strong.sum() > 0
    cos2 = np.cos(res.orientation[strong]) ** 2
    assert cos2.mean() > 0.8


def test_orientation_across_edge_is_vertical_for_horizontal_edge():
    size = 48
    res = structure_init(_horizontal_edge(size), 120, rng=np.random.default_rng(0))
    # Across-edge angle should point along y -> sin^2 ~ 1.
    strong = res.energy > 0.25 * res.energy.max()
    assert strong.sum() > 0
    sin2 = np.sin(res.orientation[strong]) ** 2
    assert sin2.mean() > 0.8


# ---------------------------------------------------------------------------------------------
# WSE placement: exact count, in-bounds, blue noise, density-adaptive
# ---------------------------------------------------------------------------------------------
def test_exact_count_and_in_bounds():
    size = 48
    n = 150
    res = structure_init(_vertical_edge(size), n, rng=np.random.default_rng(0))
    assert isinstance(res, StructureInitResult)
    assert res.xy.shape == (n, 2)
    assert res.chol.shape == (n, 3)
    assert res.n == n
    assert res.xy.dtype == np.float32
    assert res.chol.dtype == np.float32
    assert (res.xy[:, 0] >= 0.0).all() and (res.xy[:, 0] < size).all()
    assert (res.xy[:, 1] >= 0.0).all() and (res.xy[:, 1] < size).all()
    assert np.isfinite(res.xy).all() and np.isfinite(res.chol).all()


def _min_pairwise_distance(pts: np.ndarray) -> float:
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return float(np.sqrt(d2.min()))


def test_wse_is_blue_noise_spaced():
    size = 48
    n = 150
    res = structure_init(_vertical_edge(size), n, rng=np.random.default_rng(0))
    wse_min = _min_pairwise_distance(res.xy.astype(np.float64))

    # A uniform-random N-subset of the same image has much smaller minimum spacing
    # (birthday-paradox clumping). Blue noise must beat it, and never coincide.
    rng = np.random.default_rng(1)
    rand_pts = rng.uniform(0.0, size, size=(n, 2))
    rand_min = _min_pairwise_distance(rand_pts)
    assert wse_min > 0.25  # no near-coincident points
    assert wse_min > 2.0 * rand_min  # clearly better spaced than random


def test_density_adaptive_denser_near_edge():
    size = 48
    n = 200
    res = structure_init(_vertical_edge(size), n, rng=np.random.default_rng(0))
    center = size // 2
    x = res.xy[:, 0]
    band = 2.5
    edge_count = int((np.abs(x - center) <= band).sum())
    flat_count = int((np.abs(x - center / 2) <= band).sum())  # equal-width flat band
    assert edge_count > flat_count


# ---------------------------------------------------------------------------------------------
# Oriented Cholesky: SPD, diagonal floor, elongation along the tangent
# ---------------------------------------------------------------------------------------------
def test_chol_is_spd_with_diagonal_above_min_diag():
    size = 48
    res = structure_init(_vertical_edge(size), 150, rng=np.random.default_rng(0))
    l11, l22 = res.chol[:, 0], res.chol[:, 2]
    assert (l11 > _MIN_DIAG).all()
    assert (l22 > _MIN_DIAG).all()

    sigma = _sigma_from_chol(res.chol)
    eigvals = np.linalg.eigvalsh(sigma)
    assert (eigvals > 0.0).all()  # symmetric positive definite
    # symmetry sanity
    assert np.allclose(sigma[:, 0, 1], sigma[:, 1, 0])


def test_edge_gaussians_elongate_along_tangent():
    size = 48
    res = structure_init(_vertical_edge(size), 200, rng=np.random.default_rng(0))
    sigma = _sigma_from_chol(res.chol)
    sxx = sigma[:, 0, 0]
    syy = sigma[:, 1, 1]

    center = size // 2
    ix, iy = _pixel_index(res.xy, size)
    coh = res.coherence[iy, ix]
    # Near the vertical edge with clear coherence -> elongated vertically (Sigma_yy > Sigma_xx).
    near_edge = (np.abs(res.xy[:, 0] - center) <= 2.0) & (coh > 0.3)
    assert near_edge.sum() >= 5
    frac_vertical = float((syy[near_edge] > sxx[near_edge]).mean())
    assert frac_vertical > 0.7
    assert syy[near_edge].mean() > sxx[near_edge].mean()


def test_flat_image_gives_near_isotropic_gaussians():
    size = 40
    flat = np.full((size, size, 3), 0.5, dtype=np.float32)
    res = structure_init(flat, 100, rng=np.random.default_rng(0))
    sigma = _sigma_from_chol(res.chol)
    eigvals = np.linalg.eigvalsh(sigma)  # ascending
    axis_ratio = eigvals[:, 1] / eigvals[:, 0]
    # No structure -> coherence ~ 0 -> ~isotropic covariances.
    assert np.median(axis_ratio) < 1.3
    assert (res.chol[:, 0] > _MIN_DIAG).all() and (res.chol[:, 2] > _MIN_DIAG).all()


# ---------------------------------------------------------------------------------------------
# Determinism and config
# ---------------------------------------------------------------------------------------------
def test_deterministic_given_rng():
    img = _vertical_edge(48)
    a = structure_init(img, 120, rng=np.random.default_rng(0))
    b = structure_init(img, 120, rng=np.random.default_rng(0))
    assert np.array_equal(a.xy, b.xy)
    assert np.array_equal(a.chol, b.chol)


def test_anisotropy_strength_zero_is_isotropic():
    size = 48
    cfg = StructureInitConfig(anisotropy_strength=0.0)
    res = structure_init(_vertical_edge(size), 150, rng=np.random.default_rng(0), config=cfg)
    sigma = _sigma_from_chol(res.chol)
    eigvals = np.linalg.eigvalsh(sigma)
    axis_ratio = eigvals[:, 1] / eigvals[:, 0]
    assert np.allclose(axis_ratio, 1.0, atol=1e-3)


def test_invalid_config_and_args_raise():
    with pytest.raises(ValueError):
        StructureInitConfig(gradient_operator="prewitt")
    with pytest.raises(ValueError):
        StructureInitConfig(min_axis_ratio=0.0)
    with pytest.raises(ValueError):
        StructureInitConfig(wse_oversample=0.5)
    with pytest.raises(ValueError):
        structure_init(_vertical_edge(16), 0, rng=np.random.default_rng(0))
