"""Jet-consistency (Weingarten) prior on splat normals and centres (opt-in, pure torch).

For splat ``i`` with centre ``c_i``, normal ``nu_i`` (the smallest-scale axis of its rotation)
and tangent axes ``(e1, e2)`` (the two other axes), neighbours ``j`` (the k nearest centres)
give ``d_j = c_j - c_i``, tangential coordinates ``s_j = (e1.d_j, e2.d_j)``, height
``z_j = nu_i.d_j``, normal tangent components ``a_j = (e1.nu_j, e2.nu_j) = P_i(nu_j - nu_i)`` and
weights ``g_j = exp(-|d_j|^2 / 2 h_i^2)`` with ``h_i`` the mean distance to the 6 nearest centres.

``S_i`` (symmetric 2x2) solves ``min sum_j g_j |a_j - S s_j|^2`` in closed form and is detached.
With this convention ``S = I/R`` on a sphere with outward normals and the osculating surface is
``z = -1/2 s^T S s`` (the SplatDiffuseLBO §11.2 code convention), so the residuals are

    r_nu = sum_j g_j |a_j - S_i s_j|^2 / sum_j g_j
    r_c  = sum_j g_j (z_j + 1/2 s_j^T S_i s_j)^2 / (h_i^2 sum_j g_j)

and the loss is ``mean_i (r_nu + r_c)``. ``order=1`` fixes ``S = 0`` (parallel normals and
co-planar neighbours). Splat normals are unoriented, so each neighbour normal is flipped to the
hemisphere of ``nu_i`` (a detached sign).
"""

from __future__ import annotations

import torch

from rtgs.core.gaussians3d import quat_to_rotmat


def knn_indices(points: torch.Tensor, k: int, chunk: int = 1024) -> torch.Tensor:
    """(N,k) indices of the k nearest other points (exact, chunked brute force)."""
    points = points.detach()
    n = points.shape[0]
    k = min(k, n - 1)
    out = torch.empty(n, k, dtype=torch.long, device=points.device)
    for start in range(0, n, chunk):
        distances = torch.cdist(points[start : start + chunk], points)
        rows = torch.arange(distances.shape[0], device=points.device)
        distances[rows, rows + start] = float("inf")
        out[start : start + chunk] = distances.topk(k, largest=False).indices
    return out  # ponytail: O(N^2) brute force; swap for a grid/KD search if N >> 1e5


def splat_frames(quats: torch.Tensor, log_scales: torch.Tensor) -> torch.Tensor:
    """(N,3,3) rows (e1, e2, nu): the two larger axes then the smallest-scale axis."""
    rot = quat_to_rotmat(quats)  # columns are the scale axes
    order = log_scales.detach().argsort(dim=1, descending=True)
    return torch.gather(rot, 2, order[:, None, :].expand(-1, 3, -1)).transpose(1, 2)


def jet_residuals(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    neighbours: torch.Tensor,
    *,
    order: int = 2,
    n_bandwidth: int = 6,
    ridge: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-splat ``(r_nu, r_c, S)``; neighbours are (N,k) indices sorted by distance."""
    if order not in (1, 2):
        raise ValueError("jet order must be 1 or 2")
    frames = splat_frames(quats, log_scales)  # (N,3,3)
    e, nu = frames[:, :2], frames[:, 2]  # (N,2,3), (N,3)
    delta = means[neighbours] - means[:, None]  # (N,k,3)
    h = delta.detach()[:, :n_bandwidth].norm(dim=-1).mean(1).clamp_min(1e-12)  # (N,)
    g = torch.exp(-delta.detach().square().sum(-1) / (2 * h[:, None] ** 2))  # (N,k)
    gsum = g.sum(1).clamp_min(1e-30)
    nu_j = nu[neighbours]  # (N,k,3)
    sign = torch.sign((nu_j * nu[:, None]).sum(-1).detach())
    nu_j = nu_j * torch.where(sign == 0, torch.ones_like(sign), sign)[..., None]
    s = torch.einsum("nkd,ncd->nkc", delta, e)  # (N,k,2)
    z = torch.einsum("nkd,nd->nk", delta, nu)
    a = torch.einsum("nkd,ncd->nkc", nu_j, e)
    if order == 1:
        shape = s.new_zeros(s.shape[0], 2, 2)
    else:
        with torch.no_grad():
            s0, s1, a0, a1 = s[..., 0], s[..., 1], a[..., 0], a[..., 1]
            zero = torch.zeros_like(s0)
            # Rows of the 2-equation design for x = (S11, S12, S22).
            r1 = torch.stack([s0, s1, zero], -1)
            r2 = torch.stack([zero, s0, s1], -1)
            normal = torch.einsum("nk,nki,nkj->nij", g, r1, r1) + torch.einsum(
                "nk,nki,nkj->nij", g, r2, r2
            )
            rhs = torch.einsum("nk,nki->ni", g, r1 * a0[..., None] + r2 * a1[..., None])
            scale = normal.diagonal(dim1=1, dim2=2).sum(-1).clamp_min(1e-30)
            normal = normal + ridge * scale[:, None, None] * torch.eye(3, device=s.device)
            x = torch.linalg.solve(normal.double(), rhs.double()[..., None])[..., 0].to(s.dtype)
            shape = torch.stack([torch.stack([x[:, 0], x[:, 1]], -1), x[:, 1:]], -2)
    predicted = torch.einsum("nij,nkj->nki", shape, s)
    r_nu = (g * (a - predicted).square().sum(-1)).sum(1) / gsum
    height = z + 0.5 * (s * predicted).sum(-1)
    r_c = (g * height.square()).sum(1) / (h.square() * gsum)
    return r_nu, r_c, shape


class JetPrior:
    """Loss holder; refreshes the k-NN list every ``refresh_every`` steps and on topology change."""

    def __init__(self, weight: float, order: int = 2, k: int = 16, refresh_every: int = 50) -> None:
        if order not in (1, 2):
            raise ValueError("jet order must be 1 or 2")
        self.weight, self.order, self.k, self.refresh_every = float(weight), order, k, refresh_every
        self._neighbours: torch.Tensor | None = None
        self._key: tuple | None = None
        self.refreshes = 0

    def __call__(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        step: int,
        opacities: torch.Tensor | None = None,  # unused; FieldPrior weights by opacity
    ) -> torch.Tensor:
        # Densification replaces the parameter tensors; any change invalidates the indices.
        key = (means.shape[0], means.data_ptr())
        if self._neighbours is None or key != self._key or step % self.refresh_every == 0:
            self._neighbours = knn_indices(means, self.k)
            self._key = key
            self.refreshes += 1
        r_nu, r_c, _ = jet_residuals(means, quats, log_scales, self._neighbours, order=self.order)
        return (r_nu + r_c).mean()


# --------------------------------------------------------------------------- v2.1: field prior
#
# Task 20261008_jet_field_prior_2dgs_tosca_cat0 (PREREG v2.1). The residual is measured against
# the splat *field*, not against neighbour lists. With a_ij = (w_j / rho_j) g_ij,
#   g_ij = exp(-|c_j - c_i|^2 / 2 (1.5 h0_i)^2),  rho_j = sum_k g_jk (self incl.),  w_j = opacity,
# over the fixed-radius pairs |c_j - c_i| < 3 h0_i (self included):
#   m_i = sum_j a_ij c_j / sum_j a_ij                         (sheet centroid)
#   C_i = sum_j a_ij (Sigma_j + (c_j - m_i)(c_j - m_i)^T)     (sheet PCA, centred at m_i)
# n_i = smallest eigenvector of C_i; n_i and m_i are detached. Because a clone of j doubles
# rho_j and halves each copy's weight, C_i, m_i and n_i are *invariant under uniform
# replication at fixed bandwidth* (every splat duplicated, h0 unchanged); non-uniform
# duplication does change them. Teacher guard: the residual of splat i counts only where the
# field estimate is well posed, eigengap (l2 - l1) / l3 >= 0.1 and effective support
# sum_j a_ij / max_j a_ij >= 4 (masked residuals are 0, the mean stays over all splats).
# Residuals: r_nu = 1 - (nu_i . n_i)^2 (treatment, terms="normal"); r_c = ((c_i - m_i) . n_i)^2
# / h0_i^2 is an ablation (terms="normal+centre"): on a curved sheet m_i lies on the concave
# side, so r_c pulls correctly placed centres inward (a shrinkage force, see the tests).
# Gradients flow into quats_i (through nu_i) and means_i (r_c only); h0_i = mean distance to
# the 6 nearest centres, frozen at the first active step (the end of densification).

FIELD_TERMS = ("normal", "normal+centre")


def mean_neighbour_distance(points: torch.Tensor, k: int = 6) -> torch.Tensor:
    """(N,) mean distance to the k nearest other points."""
    points = points.detach()
    return (points[knn_indices(points, k)] - points[:, None]).norm(dim=-1).mean(1)


def radius_pairs(
    points: torch.Tensor, radius: torch.Tensor, chunk: int = 1024
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rows, cols)`` of every pair with ``|c_col - c_row| < radius[row]``, self included."""
    points = points.detach()
    rows, cols = [], []
    for start in range(0, points.shape[0], chunk):
        block = slice(start, start + chunk)
        near = torch.cdist(points[block], points) < radius[block, None]
        r, c = near.nonzero(as_tuple=True)
        rows.append(r + start)
        cols.append(c)
    return torch.cat(rows), torch.cat(cols)  # ponytail: O(N^2) scan every refresh; grid if N >> 1e5


def field_estimate(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    opacities: torch.Tensor,
    h0: torch.Tensor,
    pairs: tuple[torch.Tensor, torch.Tensor],
    bandwidth: float = 1.5,
    min_gap: float = 0.1,
    min_support: float = 4.0,
) -> dict[str, torch.Tensor]:
    """Detached field normal ``n``, centroid ``m``, guard mask ``ok``, ``gap`` and ``support``."""
    rows, cols = pairs
    n = means.shape[0]
    with torch.no_grad():
        c = means.detach()
        g = torch.exp(-(c[cols] - c[rows]).square().sum(-1) / (2 * (bandwidth * h0[rows]) ** 2))
        rho = torch.zeros_like(h0).index_add_(0, rows, g)
        a = opacities.detach()[cols] / rho[cols] * g
        total = means.new_zeros(n).index_add_(0, rows, a).clamp_min(1e-30)
        peak = means.new_zeros(n).index_reduce_(0, rows, a, "amax", include_self=False)
        centroid = means.new_zeros(n, 3).index_add_(0, rows, a[:, None] * c[cols]) / total[:, None]
        rot = quat_to_rotmat(quats.detach())
        cov = torch.einsum("nij,nj,nkj->nik", rot, torch.exp(2 * log_scales.detach()), rot)
        d = c[cols] - centroid[rows]
        moment = cov[cols] + d[:, :, None] * d[:, None, :]
        tensor = means.new_zeros(n, 3, 3).index_add_(0, rows, a[:, None, None] * moment)
        values, vectors = torch.linalg.eigh(tensor.double())
        normal = vectors[..., 0].to(means.dtype)
        gap = ((values[:, 1] - values[:, 0]) / values[:, 2].clamp_min(1e-300)).to(means.dtype)
        support = total / peak.clamp_min(1e-30)
        ok = (gap >= min_gap) & (support >= min_support)
    return {"normal": normal, "centroid": centroid, "ok": ok, "gap": gap, "support": support}


def field_residuals(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    opacities: torch.Tensor,
    h0: torch.Tensor,
    pairs: tuple[torch.Tensor, torch.Tensor],
    bandwidth: float = 1.5,
    min_gap: float = 0.1,
    min_support: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Per-splat ``(r_nu, r_c, field)``, residuals already masked by the teacher guard."""
    field = field_estimate(
        means, quats, log_scales, opacities, h0, pairs, bandwidth, min_gap, min_support
    )
    nu = splat_frames(quats, log_scales)[:, 2]
    normal, mask = field["normal"], field["ok"].to(means.dtype)
    r_nu = (1.0 - (nu * normal).sum(-1).square()) * mask
    r_c = ((means - field["centroid"]) * normal).sum(-1).square() / h0.square() * mask
    return r_nu, r_c, field


class FieldPrior:
    """v2.1 loss holder: off before ``start``; freezes ``h0`` at the first active step.

    ``stats`` holds the last step's guard and term diagnostics; every ``log_every`` steps it also
    records the median per-splat gradient norm of the weighted prior w.r.t. means and quats.
    """

    def __init__(
        self,
        weight: float,
        start: int = 7500,
        terms: str = "normal",
        refresh_every: int = 50,
        radius: float = 3.0,
        bandwidth: float = 1.5,
        n_bandwidth: int = 6,
        min_gap: float = 0.1,
        min_support: float = 4.0,
        log_every: int = 500,
    ) -> None:
        if terms not in FIELD_TERMS:
            raise ValueError(f"terms must be one of {FIELD_TERMS}")
        self.weight, self.start, self.terms = float(weight), int(start), terms
        self.refresh_every, self.radius, self.bandwidth = int(refresh_every), radius, bandwidth
        self.n_bandwidth, self.min_gap, self.min_support = n_bandwidth, min_gap, min_support
        self.log_every = int(log_every)
        self.h0: torch.Tensor | None = None
        self._pairs: tuple[torch.Tensor, torch.Tensor] | None = None
        self.refreshes = 0
        self.stats: dict = {}
        self.log: list[dict] = []

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
            raise ValueError("FieldPrior weights the field by opacity; pass opacities")
        if self.h0 is None:
            self.h0 = mean_neighbour_distance(means, self.n_bandwidth).clamp_min(1e-12)
        elif self.h0.shape[0] != means.shape[0]:
            raise RuntimeError("the splat count changed after h0 was frozen (densification on?)")
        if self._pairs is None or (step - self.start) % self.refresh_every == 0:
            self._pairs = radius_pairs(means, self.radius * self.h0)
            self.refreshes += 1
        r_nu, r_c, field = field_residuals(
            means,
            quats,
            log_scales,
            opacities,
            self.h0,
            self._pairs,
            self.bandwidth,
            self.min_gap,
            self.min_support,
        )
        loss = (r_nu + r_c).mean() if self.terms == "normal+centre" else r_nu.mean()
        self.stats = {
            "step": step,
            "guarded_fraction": float(1.0 - field["ok"].float().mean()),
            "r_nu": float(r_nu.detach().mean()),
            "r_c": float(r_c.detach().mean()),
            "pairs_per_splat": self._pairs[0].numel() / means.shape[0],
        }
        if (step - self.start) % self.log_every == 0 and loss.requires_grad:
            inputs = [t for t in (means, quats) if t.requires_grad]
            grads = torch.autograd.grad(
                self.weight * loss, inputs, retain_graph=True, allow_unused=True
            )
            for tensor, grad in zip(inputs, grads, strict=True):
                name = "means" if tensor is means else "quats"
                value = 0.0 if grad is None else float(grad.norm(dim=-1).median())
                self.stats[f"prior_grad_{name}_median"] = value
            self.log.append(dict(self.stats))
        return loss
