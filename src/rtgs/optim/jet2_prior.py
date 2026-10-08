"""v3 jet prior: 2-jet (shape-operator) consistency on a clone-invariant, frozen neighbourhood.

Plan: ``docs/TASK_jet2_synthetic_sphere_ellipsoid.md`` (rev. 3). Same convention as v1
(``jet_prior.py``) and SplatDiffuseLBO §11: frame ``(e1, e2, nu)`` from ``splat_frames``, outward
``S = +I/R`` on a sphere, osculating surface ``z = -1/2 s^T S s``.

At the first active step (after densification) each splat freezes ``sig_i`` = its largest
tangential scale (detached). Every ``refresh_every`` steps the pairs ``|c_j - c_i| < 3 sig_i``
are recomputed from detached centres. Per step, all weights are detached:

    g_ij  = exp(-|c_j - c_i|^2 / 2 sig_i^2)                  isotropic (needle splats keep S_22)
    rho_j = sum_k g_jk over all pairs of j, self and zero-offset copies included
    a_ij  = o_j g_ij / rho_j  for pairs with |c_j - c_i| > 1e-6 sig_i, else 0

``rho`` makes uniform replication (every splat copied k times) leave every residual unchanged:
copies of a neighbour share its weight, copies of the splat itself sit at zero offset and are
excluded from the fit. With ``s_ij, z_ij`` the tangential/normal coordinates of ``c_j - c_i`` in
frame ``i`` and ``n_ij = (e1_i, e2_i) . nu_j`` (``nu_j`` flipped to the hemisphere of ``nu_i``,
detached sign), ``S_i`` (symmetric 2x2, detached) solves ``min sum_j a_ij |n_ij - S s_ij|^2``;
``order=1`` sets ``S = 0``. Residuals, normalised by ``A_i = sum_j a_ij``:

    r_nu_i = sum_j a_ij |n_ij - S_i s_ij|^2 / A_i
    r_c_i  = sum_j a_ij (z_ij + 1/2 s_ij^T S_i s_ij)^2 / (sig_i^2 A_i)

Fail-closed support gate (detached, identical for both orders so the arms see the same splats):
neighbour mass ``A_i >= min_mass`` (``A_i`` is the share of the local kernel mass held by other
splats: 0 when isolated, unchanged by uniform replication, unlike a count) and design conditioning
``lambda_min / trace >= min_conditioning`` of the 3x3 normal matrix for ``(S11, S12, S22)``.
Masked splats contribute 0; the loss is the mean over all splats; ``stats`` logs the masked
fraction. Gradients reach means and quats only (scales/opacity detached).
"""

from __future__ import annotations

import torch

from rtgs.optim.jet_prior import splat_frames

ZERO_OFFSET = 1e-6  # relative to sig_i: pairs closer than this are copies of the splat itself


def exact_radius_pairs(
    points: torch.Tensor, radius: torch.Tensor, chunk: int = 1024
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rows, cols)`` with ``|c_col - c_row| < radius[row]``, self and exact copies always kept.

    ``torch.cdist``'s default matmul path can report a nonzero distance for identical float32
    points, dropping self pairs of tiny splats (then ``rho = 0``); the direct mode is exact.
    """
    points = points.detach()
    rows, cols = [], []
    for start in range(0, points.shape[0], chunk):
        block = slice(start, start + chunk)
        d = torch.cdist(points[block], points, compute_mode="donot_use_mm_for_euclid_dist")
        r, c = (d < radius[block, None]).nonzero(as_tuple=True)
        rows.append(r + start)
        cols.append(c)
    return torch.cat(rows), torch.cat(cols)  # ponytail: O(N^2) scan per refresh; grid if N >> 1e5


def jet2_residuals(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    opacities: torch.Tensor,
    sig: torch.Tensor,
    pairs: tuple[torch.Tensor, torch.Tensor],
    *,
    order: int = 2,
    min_mass: float = 0.5,
    min_conditioning: float = 0.02,
    ridge: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Per-splat ``(r_nu, r_c, info)``; residuals already masked. ``info``: ``S``, ``ok``."""
    if order not in (1, 2):
        raise ValueError("jet order must be 1 or 2")
    rows, cols = pairs
    n = means.shape[0]
    sig = sig.detach()
    frames = splat_frames(quats, log_scales)
    e, nu = frames[:, :2], frames[:, 2]
    delta = means[cols] - means[rows]  # (P,3)
    with torch.no_grad():
        dist2 = delta.detach().square().sum(-1)
        g = torch.exp(-dist2 / (2 * sig[rows] ** 2))
        rho = means.new_zeros(n).index_add_(0, rows, g)
        far = dist2 > (ZERO_OFFSET * sig[rows]) ** 2
        a = torch.where(far, opacities.detach()[cols] * g / rho[cols], torch.zeros_like(g))
        total = means.new_zeros(n).index_add_(0, rows, a)
        sign = torch.sign((nu[cols] * nu[rows]).sum(-1).detach())
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    nu_j = nu[cols] * sign[:, None]
    s = torch.einsum("pd,pcd->pc", delta, e[rows])  # (P,2)
    z = (delta * nu[rows]).sum(-1)
    nij = torch.einsum("pd,pcd->pc", nu_j, e[rows])
    with torch.no_grad():
        s0, s1 = s[:, 0].detach(), s[:, 1].detach()
        zero = torch.zeros_like(s0)
        r1 = torch.stack([s0, s1, zero], -1)  # n_1 = S11 s0 + S12 s1
        r2 = torch.stack([zero, s0, s1], -1)  # n_2 = S12 s0 + S22 s1
        outer = a[:, None, None] * (r1[:, :, None] * r1[:, None] + r2[:, :, None] * r2[:, None])
        normal = means.new_zeros(n, 3, 3).index_add_(0, rows, outer).double()
        trace = normal.diagonal(dim1=1, dim2=2).sum(-1)
        conditioning = torch.linalg.eigvalsh(normal)[:, 0] / trace.clamp_min(1e-300)
        ok = (total > 0) & (total >= min_mass) & (conditioning >= min_conditioning)
        if order == 2:
            nd = nij.detach()
            rhs_p = a[:, None] * (r1 * nd[:, :1] + r2 * nd[:, 1:])
            rhs = means.new_zeros(n, 3).index_add_(0, rows, rhs_p).double()
            eye = torch.eye(3, dtype=normal.dtype, device=normal.device)
            reg = ridge * trace.clamp_min(1e-300)[:, None, None] * eye
            x = torch.linalg.solve(normal + reg, rhs[..., None])[..., 0]
            x = torch.where(ok[:, None], x, torch.zeros_like(x)).to(means.dtype)
        else:
            x = means.new_zeros(n, 3)
        shape = torch.stack([torch.stack([x[:, 0], x[:, 1]], -1), x[:, 1:]], -2)  # (N,2,2)
    predicted = torch.einsum("pab,pb->pa", shape[rows], s)
    height = z + 0.5 * (s * predicted).sum(-1)
    # rejected rows get unit denominators: masking after a 0/0 would leave NaN gradients
    denom = torch.where(ok, total, torch.ones_like(total))
    sig2 = torch.where(ok, sig.square(), torch.ones_like(sig))
    mask = ok.to(means.dtype)
    r_nu = means.new_zeros(n).index_add(0, rows, a * (nij - predicted).square().sum(-1))
    r_c = means.new_zeros(n).index_add(0, rows, a * height.square())
    r_nu = r_nu / denom * mask
    r_c = r_c / (denom * sig2) * mask
    return r_nu, r_c, {"S": shape, "ok": ok, "mass": total}


class Jet2Prior:
    """Loss holder: zero before ``start``; freezes ``sig`` at the first active step."""

    def __init__(
        self,
        weight: float,
        order: int = 2,
        start: int = 7500,
        refresh_every: int = 50,
        radius: float = 3.0,
        min_mass: float = 0.5,
        min_conditioning: float = 0.02,
    ) -> None:
        if order not in (1, 2):
            raise ValueError("jet order must be 1 or 2")
        self.weight, self.order, self.start = float(weight), order, int(start)
        self.refresh_every, self.radius = int(refresh_every), float(radius)
        self.min_mass, self.min_conditioning = min_mass, min_conditioning
        self.sig: torch.Tensor | None = None
        self._pairs: tuple[torch.Tensor, torch.Tensor] | None = None
        self.stats: dict = {}

    def __call__(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        step: int,
        opacities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if step < self.start:
            return means.new_zeros(())
        if opacities is None:
            raise ValueError("Jet2Prior weights neighbours by opacity; pass opacities")
        if self.sig is None:
            self.sig = log_scales.detach().max(dim=1).values.exp().clamp_min(1e-12)
        elif self.sig.shape[0] != means.shape[0]:
            raise RuntimeError("the splat count changed after sig was frozen (densification on?)")
        if self._pairs is None or (step - self.start) % self.refresh_every == 0:
            self._pairs = exact_radius_pairs(means, self.radius * self.sig)
        r_nu, r_c, info = jet2_residuals(
            means,
            quats,
            log_scales,
            opacities,
            self.sig,
            self._pairs,
            order=self.order,
            min_mass=self.min_mass,
            min_conditioning=self.min_conditioning,
        )
        self.stats = {
            "step": step,
            "masked_fraction": float(1.0 - info["ok"].float().mean()),
            "r_nu": float(r_nu.detach().mean()),
            "r_c": float(r_c.detach().mean()),
            "pairs_per_splat": self._pairs[0].numel() / means.shape[0],
        }
        return (r_nu + r_c).mean()
