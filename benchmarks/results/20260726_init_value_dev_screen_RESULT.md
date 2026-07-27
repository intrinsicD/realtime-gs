# Initialization Value Program — development screen (RESULT)

> **PROTOCOL WITHDRAWN AFTER THIS RUN (2026-07-26).** This screen cites
> `20260725_init_value_program_PREREG.md` at sha256 `e8f75dd7…`, which was an uncommitted rewrite
> the author withdrew the same day; it is preserved at
> `20260726_init_value_program_REWRITE_WITHDRAWN.md`. The governing protocol is the committed
> version at `7d62f50` (sha256 `2336b636…`), whose M0–M5 / E0–E11 program this run does not
> follow. The measurements below stand as measurements; they are governed by nothing and settle
> no preregistered question.

**Status: DEVELOPMENT-ONLY — NOT CONFIRMATORY.**
Karate is a validation/development scene under §1.2 of the master protocol
(`20260725_init_value_program_PREREG.md`, SHA-256
`e8f75dd72425abcb5b2edf33a88ba7c456b69576b97500083fae064ebae5f8cc`), whose outcome use is
"implementation validation and a fixed screen; never a public method claim". Nothing in this
document may enter `README.md`, `docs/`, or `ara/logic/claims.md`. The confirmatory phase remains
blocked: review 02 (`..._REVIEW_02_FAIL.md`) is still FAIL/bounded-amendment-required, and §2.3
confirmation still requires three newly acquired mask-bearing scenes.

This screen was run *because* the protocol is blocked, not in place of unblocking it. Its purpose
was to exercise the harness and to measure review finding B1 — whether the historical single random
arm is an information control — before the confirmatory design is frozen.

## Run identity

| Field | Value |
| --- | --- |
| Run dir | `runs/init_value_dev_screen/` |
| Summary | `summary.json` SHA-256 `bc255e7cf8cd286907ceaddaf38c6a4d401af770e4280419979e77971f9a00fd` |
| Results page | `index.html` SHA-256 `712a638ac2f4c26aadb260c5417e9797612fa0272aec217ffb6248db712ea776` |
| Harness | `benchmarks/init_value_dev_screen.py`, report `..._report.py`, bundle `..._bundle.py` |
| Source | `d6876bc1f1e7` (dirty working tree) |
| Environment | torch 2.9.0+cu128, gsplat 1.5.3, CUDA 12.8, RTX 3050 |
| Design | 2 scenes × 2 viewsets (v3/v8) × 2 counts (N=2400/5000) × 4 arms × 3 seeds = 96 runs |
| Optimization | fixed topology (densify off, final count asserted equal to initial), 1500 updates, downscale 4 |
| Anchor cell | v3 train views, N=2400 |
| Wall clock | 39.6 min |

Held-out quality `Q` is full-canvas PSNR against each unseen view's **own compact 2D fit**, not
against source RGB. Karate carries no packed alpha, so no masked metric exists here.

### Deviations from the frozen screen

- `colmap-sfm-fixed-pose` is **unavailable** — the repository holds no source RGB. Reported as
  baseline unavailability under §3.3; the repository default `ci` stands in as comparator and **no
  COLMAP-relative claim is made or licensed**.
- The random control was split into `random-matched` (matched to beam-cover on opacity, extent,
  and train-view colour) and `random-grey` (unmatched appearance), to measure B1 directly.
- Comparator/joint time censoring is scored NON-EVALUABLE, not a candidate success (B2).
- Report-only sealed views (`C1001`, `C1004`, `C1005`) were **not** opened.

## Anchor result (v3, N=2400) — seed-median held-out Q at 1500 updates

| Scene | Arm | Q@0 | Q@1500 | ΔQ vs `ci` | Wins (0.25 dB) | T@τ | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frame_00005 | beam-cover | 8.69 | 13.036 | +0.206 | 1/3 | 0.86× | inside the 0.25 dB band |
| frame_00005 | ci | 6.51 | 12.839 | — | — | — | reference |
| frame_00005 | random-matched | 7.92 | 12.722 | −0.219 | 0/3 | 0.97× | inside the band |
| frame_00005 | random-grey | 8.21 | 13.057 | +0.144 | 0/3 | 0.76× | inside the band |
| frame_00060 | beam-cover | 8.59 | 12.257 | +0.485 | 3/3 | 0.72× | material gain |
| frame_00060 | ci | 7.00 | 11.711 | — | — | — | reference |
| frame_00060 | random-matched | 8.13 | 11.450 | −0.433 | 1/3 | 0.64× | material loss |
| frame_00060 | random-grey | 8.35 | 11.670 | −0.182 | 0/3 | 0.94× | inside the band |

The two anchor scenes disagree. `beam-cover` is a material gain over the comparator on
`frame_00060` (3/3 seeds, and reaches the comparator's final quality in 0.72× the end-to-end
seconds) and is inside the materiality band on `frame_00005`. One anchor scene out of two is not
a result; it is a reason the confirmatory design needs more than one scene.

## Finding 1 — the historical random arm was not an information control (B1 confirmed)

`random-matched` and `random-grey` share identical means and differ **only** in appearance, so
their difference isolates the confound. Appearance effect = matched − grey:

| Cell | random-grey | random-matched | appearance effect | beam-cover − matched | beam-cover − grey |
| --- | --- | --- | --- | --- | --- |
| frame_00005/v3/N2400 | 13.057 | 12.722 | −0.335 | +0.314 | −0.021 |
| frame_00005/v3/N5000 | 13.187 | 12.474 | −0.713 | +0.513 | −0.200 |
| frame_00005/v8/N2400 | 13.666 | 13.434 | −0.232 | −0.285 | −0.516 |
| frame_00005/v8/N5000 | 13.851 | 13.177 | −0.675 | −0.024 | −0.699 |
| frame_00060/v3/N2400 | 11.670 | 11.450 | −0.220 | +0.808 | +0.588 |
| frame_00060/v3/N5000 | 11.534 | 11.231 | −0.303 | +0.619 | +0.315 |
| frame_00060/v8/N2400 | 13.083 | 13.350 | +0.268 | +0.762 | +1.030 |
| frame_00060/v8/N5000 | 12.701 | 13.090 | +0.389 | +0.235 | +0.624 |

Appearance alone moves held-out Q by up to 0.71 dB — larger than the 0.25 dB materiality margin in
5 of 8 cells — and it **changes sign between scenes**: on `frame_00005` the *unmatched grey* arm is
better, on `frame_00060`/v8 the *matched* arm is better. The candidate's measured margin therefore
depends on which random arm it is compared against: on `frame_00005` beam-cover beats
`random-matched` in 2 of 4 cells but never beats `random-grey`. Review finding B1 is confirmed
empirically, not just by argument: a single random arm cannot separate initialization *information*
from initialization *appearance*, and the amended protocol must keep both control arms.

## Finding 2 — the iteration-0 advantage is not the effect

`beam-cover` starts 1.7–2.6 dB above `ci` at step 0 in every cell, and that lead is almost entirely
gone by 1500 updates (final spread ≤ 1.0 dB, often ≤ 0.3 dB). Q@0 is not a proxy for Q@1500 and
must not be reported as evidence of initialization value. Under fixed topology, on these scenes,
1500 updates is enough for the comparator to close most of a 2 dB initialization gap.

## Finding 3 — the candidate's direction is scene-dependent, not count-dependent

`beam-cover − random-grey` is negative in all four `frame_00005` cells (−0.02 to −0.70) and
positive in all four `frame_00060` cells (+0.32 to +1.03). Changing view count (v3→v8) or primitive
count (2400→5000) does not flip the direction within a scene; changing scene does. Any confirmatory
design that pools scenes will average a sign change, and three scenes is the bare minimum to detect
one.

## What this screen does not establish

- No comparison against structure-from-motion of any kind.
- No held-out RGB quality — the target surface is each view's own 2D fit, so these numbers bound
  agreement with Stage 1, not with the camera.
- No geometry or coverage claim (no independent reference surface).
- No claim about density control — topology is fixed and birth is disabled.
- No benchmarked speed — one consumer GPU, no interleaving, no repeats beyond 3 seeds.

## Bundle

`python scripts/check_results_bundle.py runs/init_value_dev_screen` → OK. The representative
bundle at the run root is `frame_00005/v3/N2400 / beam-cover / seed 26001`, selected by a fixed
outcome-independent rule (lexicographically first anchor cell, candidate arm, lowest seed). It is a
viewer handoff and a preview, **not** a result. Viewer receipt: `viewer_smoke.json` (served).

```
.venv/bin/rtgs view --gaussians runs/init_value_dev_screen/gaussians.ply \
    --initial runs/init_value_dev_screen/gaussians_init.ply \
    --rasterizer gsplat --device cuda --host 127.0.0.1 --port 43783 --no-open
```

## Next actions

1. Amend prereg §§3.1, 4.2, 5.1.6, 6.8 for review findings B1–B4, keeping **both** random arms per
   Finding 1; re-hash and obtain a third non-author review.
2. Recover Karate source RGB and calibration against the 62 file hashes in
   `20260716_compact_point_training_SEAL.json`, which is what blocks the COLMAP comparator.
3. Commit the master protocol, both reviews, the two 2026-07-25 pilot preregistrations, and these
   harnesses so the outcome-access ledger is anchored in history.

## Environment note

The cached gsplat CUDA extension in `~/.cache/torch_extensions/py312_cu128/gsplat_cuda/` links the
system `libstdc++` (needs `CXXABI_1.3.15`), while `.venv/bin/python` resolves torch from
`~/miniconda3` whose `libstdc++` stops at `CXXABI_1.3.13`. GPU runs therefore need
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`. This bit after a reboot and is unrelated to
the screen's results.
