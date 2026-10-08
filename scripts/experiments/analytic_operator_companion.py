"""LBO spectrum of a fitted splat PLY with the SplatDiffuseLBO operator (runs in the companion env).

    ~/miniconda3/bin/python3.12 scripts/experiments/analytic_operator_companion.py \
        <gaussians.ply> <a1> <a2> <a3> <guard_minutes> <out.json>

The companion's §16 operator on the opacity > 0.3 subset: ``splat_lbo_scene.splatset`` widths
(``σ_n = max(σ_min, sqrt(σ_t1 σ_t2)/50)``, i.e. ``sqrt(σ_t1 σ_t2)/50`` for 2DGS surfels), the
companion estimator ``shape_estimated`` for the osculating splats, numba sparse assembly with
``measure="closed"`` (``splat_lbo_measure_v3.NUMBA``), zero-row removal, ``mcheck`` / shift-invert
``eig`` (λ_0..λ_31) / ``structure`` (gate G15a), the reduced solve as report-only when
cond(M) > 1e13, and per-mode n90 (DOFs carrying 90 % of vᵀMv) as the localisation diagnostic.
Built from the companion's functions, not ``splat_lbo_cat.run`` (no K/M cache, no cat reference).
Normals are oriented by the analytic outward gradient ``x/a^2`` instead of Hoppe propagation.

Status in the JSON: ``valid`` (exit 0), ``invalid`` (a structural gate failed, exit 1),
``timeout`` (whole-process wall guard, exit 4; unevaluated, not failed). RSS above the companion's
20 GB watchdog aborts with exit 3 and no JSON.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

COMPANION = Path.home() / "Documents/SplatDiffuseLBO"
ARGS = [str(Path(a).resolve()) if i in (0, 5) else a for i, a in enumerate(sys.argv[1:])]
sys.path.insert(0, str(COMPANION))
os.chdir(COMPANION)

import splat_lbo_cat as CAT  # noqa: E402  (imports splat_lbo_check_3d first: pins BLAS threads)

np = CAT.np  # numpy only after the companion pinned its BLAS threads


def write(out: str, payload: dict) -> None:
    Path(out).write_text(json.dumps(payload, indent=1) + "\n")


def n90(K, M, lam, V) -> list[int]:
    rows = []
    for k in range(V.shape[1]):
        v = V[:, k]
        c = np.abs(v * (M @ v))
        rows.append(int(np.searchsorted(np.cumsum(np.sort(c)[::-1]) / c.sum(), 0.9)) + 1)
    return rows


def main(ply: str, axes: list[float], guard_min: float, out: str) -> int:
    started = time.time()
    base = {"ply": ply, "axes": axes, "guard_minutes": guard_min, "operator": CAT.V.NUMBA}

    def expire():
        write(out, {**base, "status": "timeout", "wall_s": time.time() - started})
        os._exit(4)

    timer = threading.Timer(guard_min * 60, expire)
    timer.daemon = True
    timer.start()
    CAT.numba.set_num_threads(8)
    threading.Thread(target=CAT.watchdog, daemon=True).start()

    S, _ = CAT.SC.load(ply, CAT.OPMIN, oriented=False)
    outward = S["c"] / np.asarray(axes) ** 2
    flip = (S["nu"] * outward).sum(1) < 0
    S["nu"][flip] *= -1
    S["e2"] = np.cross(S["nu"], S["e1"])
    shape = CAT.C.shape_estimated(S)
    w = S["w"]
    t = time.time()
    K, M, mass, _, _ = CAT.S3.assemble(
        S,
        w,
        shape=shape,
        measure="closed",
        deadline=started + guard_min * 60,
        mem_limit=CAT.MEM_LIMIT,
        sparse=True,  # CSR at every N (the companion switches to dense below 4000)
        log=print,
        **CAT.V.NUMBA,
    )
    t_asm = time.time() - t
    nbrs = CAT.S3.STATS["nbrs"]
    N = len(S["c"])
    graph = CAT.csr_matrix(
        (
            np.ones(sum(len(J) for J in nbrs)),
            (np.repeat(np.arange(N), [len(J) for J in nbrs]), np.concatenate(nbrs)),
        ),
        shape=(N, N),
    )
    ncomp = int(CAT.csgraph.connected_components(graph, directed=False)[0])
    keep = np.asarray(abs(M).sum(1)).ravel() != 0
    Kr, Mr = K[keep][:, keep], M[keep][:, keep]
    mm = CAT.mcheck(Mr, "analytic")
    lam, V, info, _ = CAT.eig(Kr, Mr, "analytic")
    result = CAT.structure(Kr, mm, ncomp, lam, "analytic")
    payload = {
        **base,
        "n_subset": int(N),
        "flipped_to_outward": int(flip.sum()),
        "nzero_rows": int((~keep).sum()),
        "lam": [float(x) for x in lam],
        "area": float(M.sum()),
        "quadrature_mass_error": float(mass / w.sum() - 1),
        "n90": n90(Kr, Mr, lam, V),
        "k1": float(result["k1"]),
        "mmin": float(result["mmin"]),
        "cond_M": float(result["cond"]),
        "nzero": int(result["nzero"]),
        "ncomp": ncomp,
        "t_assembly_s": t_asm,
        "lu_time_s": float(info["lu_time"]),
        "eig_time_s": float(info["eig_time"]),
        "nnz": int(K.nnz),
        "pairs": float(CAT.S3.STATS["pairs"]),
        "gates": {g: all(v) for g, v in CAT.GATES.items()},
        "sigma_n_rule": "max(sigma_min, sqrt(s1 s2)/50)",
        "estimator": "splat_lbo_curved.shape_estimated",
    }
    if mm["cond"] > 1e13:
        lr, _, nk, tr = CAT.reduced(Kr, Mr)
        payload["reduced_report_only"] = {"lam": [float(x) for x in lr], "kept": nk, "t_s": tr}
    payload["status"] = "valid" if all(payload["gates"].values()) else "invalid"
    payload["wall_s"] = time.time() - started
    payload["peak_rss_bytes"] = int(CAT.rss())
    write(out, payload)
    timer.cancel()
    return 0 if payload["status"] == "valid" else 1


if __name__ == "__main__":
    if len(ARGS) != 6:
        raise SystemExit(__doc__)
    sys.exit(main(ARGS[0], [float(x) for x in ARGS[1:4]], float(ARGS[4]), ARGS[5]))
