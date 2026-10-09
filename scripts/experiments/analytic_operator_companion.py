"""LBO spectrum of a fitted splat PLY with the SplatDiffuseLBO operator (runs in the companion env).

    ~/miniconda3/bin/python3.12 scripts/experiments/analytic_operator_companion.py \
        <gaussians.ply> <a1> <a2> <a3> <guard_minutes> <out.json> [flat]

The companion's §16 operator, configuration **v2** (§16.3, the companion's rule for trained
splats), on the opacity > 0.3 subset: ``splat_lbo_scene.splatset`` widths
(``σ_n = max(σ_min, sqrt(σ_t1 σ_t2)/50)``, i.e. ``sqrt(σ_t1 σ_t2)/50`` for 2DGS surfels), trial
width ``σ_χ,i = 0.75 h_i`` (h_i = mean distance to the 6 nearest centres), the companion
estimator ``shape_estimated`` with ``S_i = 0`` where ``|S_i| σ_t1,i > 1`` (``flat``: ``S ≡ 0``,
the companion's v2flat control), numba sparse assembly with
``measure="closed"`` (``splat_lbo_measure_v3.NUMBA``), zero-row removal, ``mcheck`` / shift-invert
``eig`` (λ_0..λ_31) / ``structure`` (gate G15a), the reduced solve as report-only when
cond(M) > 1e13, and per-mode n90 (DOFs carrying 90 % of vᵀMv) as the localisation diagnostic.
Built from the companion's functions, not ``splat_lbo_cat.run`` (no K/M cache, no cat reference).
Normals are oriented by the analytic outward gradient ``x/a^2`` instead of Hoppe propagation.

Comparator on the same subset, computed first so it survives an operator timeout: Zhou & Lähner's
point-cloud Laplacian (``robust_laplacian.point_cloud_laplacian`` on the centres, defaults
``n_neighbors=30``, ``mollify_factor=1e-5``), λ_0..λ_31 by the same shift-invert rule.

Status in the JSON, always with ``wall_s`` and ``peak_rss_bytes``: ``valid`` (gate G15a passed,
exit 0), ``invalid`` (G15a failed, exit 1), ``timeout`` (assembly deadline or whole-process wall
guard, exit 4), ``memory`` (the companion's 16 GB assembly limit, exit 5), ``error`` (any other
exception, exit 6). ``timeout`` and ``memory`` are unevaluated, not failed. RSS above the
companion's 20 GB watchdog aborts with exit 3 and no JSON (the caller records it).
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
FLAT = ARGS[6:] == ["flat"]
ARGS = ARGS[:6]
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
    base = {
        "ply": ply,
        "axes": axes,
        "guard_minutes": guard_min,
        "operator": CAT.V.NUMBA,
        "configuration": "v2flat" if FLAT else "v2",
    }

    def expire():
        write(
            out,
            {
                **base,
                "status": "timeout",
                "wall_s": time.time() - started,
                "peak_rss_bytes": int(CAT.rss()),
            },
        )
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
    S["sx"] = 0.75 * S["h"]  # §16.3 (1): trial PU from the centre layout
    shape = CAT.C.shape_estimated(S)
    validity = CAT.S3.shape_norm(shape) * S["s1"]
    shape[validity > 1] = 0.0  # §16.3 (2): the 2-jet only where it is a jet
    if FLAT:
        shape[:] = 0.0
    base["jet_invalid_fraction"] = float(np.mean(validity > 1))
    t = time.time()
    import robust_laplacian  # after the companion pinned its BLAS threads

    L, Mz = robust_laplacian.point_cloud_laplacian(S["c"])
    lam_zl, _, _, _ = CAT.eig(L.tocsr(), Mz.tocsr(), "zhou-laehner")
    base["zhou_laehner"] = {
        "lam": [float(x) for x in lam_zl],
        "area": float(Mz.sum()),
        "params": {"n_neighbors": 30, "mollify_factor": 1e-5},
        "wall_s": time.time() - t,
    }
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
    begun = time.time()
    try:
        sys.exit(main(ARGS[0], [float(x) for x in ARGS[1:4]], float(ARGS[4]), ARGS[5]))
    except (TimeoutError, MemoryError, Exception) as error:  # noqa: BLE001 - every exit path reports
        kind = {TimeoutError: ("timeout", 4), MemoryError: ("memory", 5)}.get(
            type(error), ("error", 6)
        )
        write(
            ARGS[5],
            {
                "ply": ARGS[0],
                "status": kind[0],
                "error": repr(error),
                "wall_s": time.time() - begun,
                "peak_rss_bytes": int(CAT.rss()),
            },
        )
        sys.exit(kind[1])
