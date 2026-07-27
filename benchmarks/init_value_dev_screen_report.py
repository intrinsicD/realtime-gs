#!/usr/bin/env python3
"""Render the development screen's ``summary.json`` into a self-contained ``index.html``.

Relative links only, no external assets, so the page works straight out of the run directory and
satisfies the results-page half of Hard Rule 7.  It reports what the run measured and, just as
importantly, what the run could not measure.
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path

ARM_STYLE = {
    "beam-cover": ("#2563eb", "candidate"),
    "ci": ("#d97706", "comparator (repository default)"),
    "random-matched": ("#059669", "control, appearance matched"),
    "random-grey": ("#6b7280", "control, no photometric prior"),
}
ARM_ORDER = ("beam-cover", "ci", "random-matched", "random-grey")


def median(values):
    return float(statistics.median(values)) if values else float("nan")


def mad(values):
    if len(values) < 2:
        return 0.0
    med = statistics.median(values)
    return float(statistics.median([abs(v - med) for v in values]))


def fmt(value, digits=3, suffix=""):
    if value is None:
        return "—"
    if isinstance(value, float) and value != value:
        return "—"
    return f"{value:.{digits}f}{suffix}"


def signed(value, digits=3, suffix=""):
    if value is None:
        return "—"
    return f"{value:+.{digits}f}{suffix}"


def line_chart(series, *, x_key, y_key, x_label, y_label, width=560, height=300):
    """Inline SVG multi-series line chart. ``series`` is [(name, [points])]."""
    pad_l, pad_r, pad_t, pad_b = 58, 14, 14, 42
    xs = [p[x_key] for _, points in series for p in points]
    ys = [p[y_key] for _, points in series for p in points]
    if not xs or not ys:
        return "<p class='muted'>no data</p>"
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 == x0:
        x1 = x0 + 1
    span = y1 - y0
    y0 -= span * 0.08 or 0.5
    y1 += span * 0.08 or 0.5

    def sx(v):
        return pad_l + (v - x0) / (x1 - x0) * (width - pad_l - pad_r)

    def sy(v):
        return height - pad_b - (v - y0) / (y1 - y0) * (height - pad_t - pad_b)

    parts = [
        f"<svg viewBox='0 0 {width} {height}' class='chart' role='img' "
        f"aria-label='{html.escape(y_label)} versus {html.escape(x_label)}'>"
    ]
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        value = y0 + fraction * (y1 - y0)
        y = sy(value)
        parts.append(
            f"<line x1='{pad_l}' y1='{y:.1f}' x2='{width - pad_r}' y2='{y:.1f}' class='grid'/>"
            f"<text x='{pad_l - 8}' y='{y + 4:.1f}' class='tick' text-anchor='end'>"
            f"{value:.2f}</text>"
        )
    for fraction in (0, 0.5, 1):
        value = x0 + fraction * (x1 - x0)
        x = sx(value)
        parts.append(
            f"<text x='{x:.1f}' y='{height - pad_b + 18}' class='tick' text-anchor='middle'>"
            f"{value:.0f}</text>"
        )
    parts.append(
        f"<line x1='{pad_l}' y1='{height - pad_b}' x2='{width - pad_r}' y2='{height - pad_b}' "
        f"class='axis'/>"
        f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' y2='{height - pad_b}' class='axis'/>"
        f"<text x='{width / 2:.0f}' y='{height - 6}' class='axlabel' text-anchor='middle'>"
        f"{html.escape(x_label)}</text>"
        f"<text x='14' y='{height / 2:.0f}' class='axlabel' text-anchor='middle' "
        f"transform='rotate(-90 14 {height / 2:.0f})'>{html.escape(y_label)}</text>"
    )
    for name, points in series:
        color = ARM_STYLE.get(name, ("#111827", ""))[0]
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{sx(p[x_key]):.1f},{sy(p[y_key]):.1f}"
            for i, p in enumerate(points)
        )
        parts.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2'/>")
    parts.append("</svg>")
    legend = " ".join(
        f"<span class='key'><i style='background:{ARM_STYLE.get(n, ('#111827', ''))[0]}'></i>"
        f"{html.escape(n)}</span>"
        for n, _ in series
    )
    return f"<figure>{''.join(parts)}<figcaption class='legend'>{legend}</figcaption></figure>"


def mean_curve(records_for_arm):
    """Seed-median curve: median PSNR and median cumulative seconds at each checkpoint."""
    seeds = sorted(records_for_arm)
    if not seeds:
        return []
    length = min(len(records_for_arm[s]["curve"]) for s in seeds)
    curve = []
    for i in range(length):
        curve.append(
            {
                "step": records_for_arm[seeds[0]]["curve"][i]["step"],
                "psnr": median([records_for_arm[s]["curve"][i]["psnr"] for s in seeds]),
                "seconds": median([records_for_arm[s]["curve"][i]["seconds"] for s in seeds]),
            }
        )
    return curve


def cell_rows(cell):
    rows = []
    for arm in ARM_ORDER:
        records = cell["records"].get(arm)
        if not records:
            continue
        q = [r["q_final"] for r in records.values()]
        q0 = [r["q_init"] for r in records.values()]
        secs = [r["end_to_end_seconds"] for r in records.values()]
        ssims = [r["ssim_final"] for r in records.values()]
        contrast = cell["contrasts"].get(arm) if isinstance(cell.get("contrasts"), dict) else None
        rows.append(
            {
                "arm": arm,
                "q0": median(q0),
                "q": median(q),
                "q_mad": mad(q),
                "ssim": median(ssims),
                "seconds": median(secs),
                "delta": contrast["quality_delta_median"] if contrast else None,
                "wins": (
                    f"{contrast['quality_material_wins']}/{contrast['quality_seeds']}"
                    if contrast
                    else "—"
                ),
                "verdict": contrast["quality_verdict"] if contrast else "reference arm",
                "time_ratio": contrast["time_ratio_median"] if contrast else None,
                "time_verdict": contrast["time_verdict"] if contrast else "—",
                "evaluable": contrast["time_evaluable_pairs"] if contrast else None,
            }
        )
    return rows


def render(summary: dict, out: Path) -> Path:
    cells = summary["cells"]
    complete = {k: v for k, v in cells.items() if v.get("status") == "complete"}
    anchor_key = summary["anchor"]
    anchor_cells = {
        k: v
        for k, v in complete.items()
        if v["viewset"] == anchor_key["viewset"] and v["n_init"] == anchor_key["n_init"]
    }

    css = """
:root{--fg:#111827;--muted:#6b7280;--bg:#ffffff;--panel:#f8fafc;--line:#e5e7eb;--warn:#b45309;
--warnbg:#fffbeb}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1.25rem 4rem;font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',
Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg)}
main{max-width:1080px;margin:0 auto}
h1{font-size:1.7rem;margin:0 0 .35rem;letter-spacing:-.01em}
h2{font-size:1.2rem;margin:2.5rem 0 .75rem;padding-bottom:.35rem;
border-bottom:1px solid var(--line)}
h3{font-size:1rem;margin:1.75rem 0 .5rem}
p{margin:.6rem 0}
.sub{color:var(--muted);margin:0 0 1.25rem}
.banner{background:var(--warnbg);border:1px solid #fcd34d;border-left:4px solid var(--warn);
padding:.85rem 1rem;border-radius:6px;margin:1rem 0 1.5rem}
.banner strong{color:var(--warn)}
table{border-collapse:collapse;width:100%;font-size:.9rem;margin:.6rem 0 1rem}
th,td{text-align:right;padding:.42rem .6rem;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
thead th{font-weight:600;color:var(--muted);font-size:.8rem;text-transform:uppercase;
letter-spacing:.04em;border-bottom:1px solid #d1d5db}
tbody tr:hover{background:var(--panel)}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.86em}
.muted{color:var(--muted)}
.win{color:#047857;font-weight:600}
.loss{color:#b91c1c;font-weight:600}
.flat{color:var(--muted)}
.grid-2{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:1.25rem}
figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:8px;
padding:.6rem}
.chart{width:100%;height:auto;display:block}
.grid{stroke:#e5e7eb;stroke-width:1}
.axis{stroke:#9ca3af;stroke-width:1}
.tick{font-size:10px;fill:#6b7280}
.axlabel{font-size:11px;fill:#374151}
.legend{display:flex;flex-wrap:wrap;gap:.9rem;font-size:.8rem;color:var(--muted);
padding:.4rem .2rem 0}
.key i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:.35rem}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:1rem 1.15rem}
ul{margin:.5rem 0 .5rem 1.1rem;padding:0}li{margin:.25rem 0}
.scroll{overflow-x:auto}
dl{display:grid;grid-template-columns:auto 1fr;gap:.2rem 1rem;margin:.5rem 0;font-size:.88rem}
dt{color:var(--muted)}dd{margin:0}
@media(prefers-color-scheme:dark){
:root{--fg:#e5e7eb;--muted:#9ca3af;--bg:#0b0f14;--panel:#111826;--line:#1f2937;--warnbg:#231a06}
thead th{border-bottom-color:#374151}
.grid{stroke:#1f2937}.axis{stroke:#4b5563}.tick{fill:#9ca3af}.axlabel{fill:#d1d5db}
.win{color:#34d399}.loss{color:#f87171}}
"""

    def delta_class(value, threshold=0.25):
        if value is None:
            return "flat"
        if value >= threshold:
            return "win"
        if value <= -threshold:
            return "loss"
        return "flat"

    parts: list[str] = []
    parts.append("<main>")
    parts.append("<h1>Initialization Value Program — development screen</h1>")
    parts.append(
        f"<p class='sub'>{html.escape(summary['config']['topology'])} · "
        f"{summary['config']['iterations']} updates · seeds "
        f"{', '.join(str(s) for s in summary['config']['seeds'])} · "
        f"{html.escape(summary['environment'].get('gpu', 'CPU'))}</p>"
    )

    parts.append(
        "<div class='banner'><strong>DEVELOPMENT-ONLY — NOT CONFIRMATORY.</strong> "
        "This is not the frozen validation screen and not the confirmatory phase. The "
        "<code>colmap-sfm-fixed-pose</code> comparator could not run: this repository holds no "
        "source RGB, so <em>no COLMAP-relative claim of any kind is licensed by this page</em>. "
        "Held-out quality is measured against each unseen view's own compact 2D fit (the "
        "<em>2D self-reconstruction baseline</em>), not against source RGB. Nothing here may "
        "enter <code>README.md</code>, <code>docs/</code>, or <code>ara/logic/claims.md</code>."
        "</div>"
    )

    parts.append("<h2>What was run</h2>")
    env = summary["environment"]
    review_line = "—"
    if summary.get("review"):
        review_line = (
            f"{summary['review']['path']} <span class='mono muted'>"
            f"{summary['review']['sha256'][:16]}…</span>"
        )
    parts.append("<div class='panel'><dl>")
    for label, value in (
        (
            "Protocol",
            f"{summary['protocol']['path']} <span class='mono muted'>"
            f"{summary['protocol']['sha256'][:16]}…</span>",
        ),
        ("Review", review_line),
        ("Anchor", f"{anchor_key['viewset']} train views, N={anchor_key['n_init']}"),
        (
            "Checkpoints",
            f"{summary['config']['checkpoints'][0]}…"
            f"{summary['config']['checkpoints'][-1]} step "
            f"{summary['config']['eval_every']}",
        ),
        ("Target rule", html.escape(summary["config"]["tau_rule"])),
        ("Rasterizer", f"{summary['config']['rasterizer']} · {summary['config']['device']}"),
        (
            "Environment",
            f"torch {env['torch']} · gsplat {env.get('gsplat', '—')} · CUDA {env.get('cuda', '—')}",
        ),
        (
            "Source",
            f"<span class='mono'>{env['git_head'][:12]}</span>"
            f"{' (dirty)' if env['git_dirty'] else ''}",
        ),
        ("Wall clock", f"{summary['total_seconds'] / 60:.1f} min"),
    ):
        parts.append(f"<dt>{label}</dt><dd>{value}</dd>")
    parts.append("</dl></div>")

    parts.append("<h3>Deviations from the frozen screen</h3><ul>")
    for item in summary["deviations_from_frozen_screen"]:
        parts.append(f"<li>{html.escape(item)}</li>")
    parts.append("</ul>")

    parts.append(
        "<h3>Scenes and splits</h3><div class='scroll'><table><thead><tr>"
        "<th>Scene</th><th>Views</th><th>Alpha</th><th>Resolution</th>"
        "<th>Train (3)</th><th>Train (+5 → 8)</th><th>Validation</th>"
        "<th>Report-only</th></tr></thead><tbody>"
    )
    for key, meta in summary["scenes"].items():
        parts.append(
            f"<tr><td class='mono'>{key}</td><td>{meta['n_views_total']}</td>"
            f"<td>{'yes' if meta['has_packed_alpha'] else 'none'}</td>"
            f"<td>{meta['image_size'][0]}×{meta['image_size'][1]}</td>"
            f"<td class='mono'>{','.join(meta['train3'])}</td>"
            f"<td class='mono'>{','.join(meta['train8'][3:])}</td>"
            f"<td class='mono'>{','.join(meta['validation'])}</td>"
            f"<td class='mono muted'>{','.join(meta['report_only_sealed'])} (sealed)</td></tr>"
        )
    parts.append("</tbody></table></div>")

    # ---------------------------------------------------------------- headline
    parts.append("<h2>Anchor result (3 train views, N=2400)</h2>")
    parts.append(
        "<p>Seed-median held-out Q at 1500 updates, paired per seed against the comparator. "
        "The material margin is 0.25 dB; “wins” counts seeds clearing it.</p>"
    )
    for key, cell in anchor_cells.items():
        parts.append(f"<h3>{html.escape(key)}</h3>")
        parts.append(
            "<div class='scroll'><table><thead><tr><th>Arm</th><th>Q at step 0</th>"
            "<th>Q at 1500</th><th>MAD</th><th>ΔQ vs ci</th><th>Wins</th>"
            "<th>SSIM</th><th>End-to-end s</th><th>Time to target</th>"
            "<th>Verdict</th></tr></thead><tbody>"
        )
        for row in cell_rows(cell):
            parts.append(
                f"<tr><td class='mono'>{row['arm']}</td>"
                f"<td>{fmt(row['q0'], 2)}</td><td><strong>{fmt(row['q'], 3)}</strong></td>"
                f"<td class='muted'>{fmt(row['q_mad'], 3)}</td>"
                f"<td class='{delta_class(row['delta'])}'>{signed(row['delta'])}</td>"
                f"<td>{row['wins']}</td><td>{fmt(row['ssim'], 4)}</td>"
                f"<td>{fmt(row['seconds'], 1)}</td>"
                f"<td>{fmt(row['time_ratio'], 2, '×') if row['time_ratio'] else '—'} "
                f"<span class='muted'>{html.escape(str(row['time_verdict']))}</span></td>"
                f"<td>{html.escape(str(row['verdict']))}</td></tr>"
            )
        parts.append("</tbody></table></div>")

        series_step = [
            (arm, mean_curve(cell["records"][arm])) for arm in ARM_ORDER if arm in cell["records"]
        ]
        parts.append("<div class='grid-2'>")
        parts.append(
            line_chart(
                series_step,
                x_key="step",
                y_key="psnr",
                x_label="optimizer updates",
                y_label="held-out Q (dB)",
            )
        )
        parts.append(
            line_chart(
                series_step,
                x_key="seconds",
                y_key="psnr",
                x_label="end-to-end seconds (Stage 1 charged)",
                y_label="held-out Q (dB)",
            )
        )
        parts.append("</div>")

    # ---------------------------------------------------------------- B1 panel
    parts.append("<h2>Is the random control an information control?</h2>")
    parts.append(
        "<p>Review finding B1 says the frozen protocol's single random arm differs from the "
        "candidate in appearance as well as placement, so a win could come from colour rather "
        "than from initialization information. These two arms share identical means and differ "
        "<em>only</em> in colour, which measures that confound directly.</p>"
    )
    parts.append(
        "<div class='scroll'><table><thead><tr><th>Cell</th>"
        "<th>random-grey Q1500</th><th>random-matched Q1500</th>"
        "<th>appearance effect</th><th>beam-cover Q1500</th>"
        "<th>beam-cover − random-matched</th><th>beam-cover − random-grey</th>"
        "</tr></thead><tbody>"
    )
    for key, cell in sorted(complete.items()):
        records = cell["records"]
        if not {"random-grey", "random-matched", "beam-cover"} <= set(records):
            continue
        grey = median([r["q_final"] for r in records["random-grey"].values()])
        matched = median([r["q_final"] for r in records["random-matched"].values()])
        beam = median([r["q_final"] for r in records["beam-cover"].values()])
        parts.append(
            f"<tr><td class='mono'>{html.escape(key)}</td>"
            f"<td>{fmt(grey)}</td><td>{fmt(matched)}</td>"
            f"<td class='{delta_class(matched - grey)}'>{signed(matched - grey)}</td>"
            f"<td>{fmt(beam)}</td>"
            f"<td class='{delta_class(beam - matched)}'>{signed(beam - matched)}</td>"
            f"<td class='{delta_class(beam - grey)}'>{signed(beam - grey)}</td></tr>"
        )
    parts.append("</tbody></table></div>")
    parts.append(
        "<p class='muted'>If the last two columns disagree materially, the historical "
        "single-random-arm design was measuring appearance initialization and calling it "
        "initialization information.</p>"
    )

    # ---------------------------------------------------------------- all cells
    parts.append("<h2>Every cell</h2>")
    parts.append(
        "<p>Descriptive by design: the anchor is the decision cell, and the other cells only "
        "show whether the direction survives a change of view count or primitive count.</p>"
    )
    parts.append(
        "<div class='scroll'><table><thead><tr><th>Cell</th><th>Arm</th>"
        "<th>Q1500</th><th>MAD</th><th>ΔQ vs ci</th><th>Wins</th><th>SSIM</th>"
        "<th>Init s</th><th>End-to-end s</th><th>N</th><th>Record</th>"
        "</tr></thead><tbody>"
    )
    for key, cell in sorted(complete.items()):
        for row in cell_rows(cell):
            arm = row["arm"]
            seed0 = sorted(cell["records"][arm])[0]
            link = f"cells/{key}/{arm}/{seed0}/record.json"
            parts.append(
                f"<tr><td class='mono'>{html.escape(key)}</td><td class='mono'>{arm}</td>"
                f"<td>{fmt(row['q'])}</td><td class='muted'>{fmt(row['q_mad'])}</td>"
                f"<td class='{delta_class(row['delta'])}'>{signed(row['delta'])}</td>"
                f"<td>{row['wins']}</td><td>{fmt(row['ssim'], 4)}</td>"
                f"<td>{fmt(cell['init_seconds'].get(arm, 0.0), 1)}</td>"
                f"<td>{fmt(row['seconds'], 1)}</td><td>{cell['n_init']}</td>"
                f"<td><a href='{html.escape(link)}'>json</a></td></tr>"
            )
    for key, cell in sorted(cells.items()):
        if cell.get("status") == "insufficient_initial_points":
            parts.append(
                f"<tr><td class='mono'>{html.escape(key)}</td><td colspan='10' class='loss'>"
                f"insufficient_initial_points — {cell['native_roots']} of "
                f"{cell['requested']} requested</td></tr>"
            )
    parts.append("</tbody></table></div>")

    # ---------------------------------------------------------------- per-seed
    parts.append("<h2>Per-seed values at the anchor</h2>")
    parts.append(
        "<div class='scroll'><table><thead><tr><th>Cell</th><th>Arm</th>"
        + "".join(f"<th>seed {s}</th>" for s in summary["config"]["seeds"])
        + "<th>median</th></tr></thead><tbody>"
    )
    for key, cell in sorted(anchor_cells.items()):
        for arm in ARM_ORDER:
            if arm not in cell["records"]:
                continue
            values = [
                cell["records"][arm][str(s)]["q_final"]
                for s in summary["config"]["seeds"]
                if str(s) in cell["records"][arm]
            ]
            parts.append(
                f"<tr><td class='mono'>{html.escape(key)}</td><td class='mono'>{arm}</td>"
                + "".join(f"<td>{fmt(v)}</td>" for v in values)
                + f"<td><strong>{fmt(median(values))}</strong></td></tr>"
            )
    parts.append("</tbody></table></div>")

    # ---------------------------------------------------------------- boundary
    parts.append("<h2>What this page does not establish</h2>")
    parts.append(
        "<div class='panel'><ul>"
        "<li>No comparison against structure-from-motion of any kind. The comparator arm could "
        "not be built, and section 3.3 forbids converting that absence into a win.</li>"
        "<li>No held-out RGB quality. The target surface is each view's own 2D fit, so these "
        "numbers bound agreement with Stage 1, not with the camera.</li>"
        "<li>No confirmatory evidence. Karate is a development scene that has already been used "
        "for smoke and development runs; the frozen protocol requires three newly acquired "
        "mask-bearing scenes for any pipeline claim.</li>"
        "<li>No geometry or coverage claim: there is no independent reference surface here.</li>"
        "<li>No claim about dynamic densification. Topology is fixed and birth is disabled, "
        "which is the regime where an initialization effect is easiest to see.</li>"
        "<li>Timing is one consumer GPU, one run per cell, no repeats and no interleaving; treat "
        "seconds as indicative, not as a benchmarked speed.</li>"
        "</ul></div>"
    )

    parts.append("<h2>Artifacts</h2><ul>")
    parts.append("<li><a href='summary.json'>summary.json</a> — every number on this page</li>")
    parts.append("<li><a href='console.log'>console.log</a> — full run log</li>")
    parts.append(
        "<li><code>cells/&lt;scene&gt;/&lt;viewset&gt;/N&lt;count&gt;/&lt;arm&gt;/"
        "&lt;seed&gt;/</code> — per-run <code>record.json</code>, "
        "<code>gaussians_init.ply</code>, <code>gaussians_final.ply</code></li>"
    )
    parts.append("</ul>")

    bundle = [
        ("gaussians.ply", "representative final gaussians (viewer handoff)"),
        ("gaussians_init.ply", "representative initial gaussians"),
        ("metrics.json", "representative per-view metrics"),
        ("training_history.json", "representative quality curve"),
        ("gaussians.config.json", "protocol, config, and environment of the representative run"),
        ("viewer_smoke.json", "results-page and viewer smoke receipt"),
    ]
    present = [(name, note) for name, note in bundle if (out / name).exists()]
    if present:
        parts.append("<h2>Representative bundle</h2>")
        parts.append(
            "<div class='panel'><p>One run promoted to the run root for viewer handoff, chosen "
            "by a fixed outcome-independent rule (lexicographically first anchor cell, candidate "
            "arm, lowest seed). It is a handoff and a preview, not a result — the numbers on this "
            "page come from all 96 sub-runs.</p></div>"
        )
        parts.append("<ul>")
        for name, note in present:
            parts.append(f"<li><a href='{name}'>{name}</a> — {note}</li>")
        parts.append("</ul>")

    previews = [
        p.name for p in sorted(out.glob("*.png")) + sorted(out.glob("*.gif")) if p.is_file()
    ]
    if previews:
        parts.append("<h2>Previews</h2><ul>")
        for name in previews:
            parts.append(f"<li><a href='{name}'>{name}</a></li>")
        parts.append("</ul>")

    parts.append("</main>")

    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Initialization Value Program — development screen</title>"
        f"<style>{css}</style></head><body>{''.join(parts)}</body></html>"
    )
    path = out / "index.html"
    path.write_text(page)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/init_value_dev_screen"))
    args = parser.parse_args()
    run = args.run.resolve()
    summary = json.loads((run / "summary.json").read_text())
    path = render(summary, run)
    print(f"[written] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
