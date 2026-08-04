# BENCH-019 development source adapters

These exact-key manifests are the development-only source boundary for RTGS-010. They were
generated from `../structsplat_bench019_capture_portfolio.json` without fitting a field, running a
reconstruction, opening confirmation payloads, or producing a BENCH-019 outcome.

| Capture | Adapter SHA-256 | Selection evidence |
|---|---|---|
| Janelle Stage `frame_00008` | `a47fbce3551c29a7c294aa7db3186405705784b901a5983de8989b218f0d196e` | exact portfolio order; 26 views; held-out ordinals 7/15/23 |
| TUM `fr1/xyz` | `539dec4d279a124e1a6b8c58afbb80d6104cf69e250c1f3e931fbe9d0c334280` | 789 associated triples; 77 pose keyframes; endpoint-preserving 26-view selection |
| TUM `fr1/rpy` | `9e264a536721eeb6714a5393726884a03c6641bfed5fb088bbdd310f5cbdcc71` | 687 associated triples; 148 pose keyframes; endpoint-preserving 26-view selection |

The SHA-256 values above describe the adapter files, not their internal `semantic_digest` fields.
Each manifest additionally binds the source portfolio and every source artifact by absolute path,
bytes, and SHA-256. Replay all source bindings with:

```bash
for adapter in experiments/data/bench019_adapters/*.adapter.json; do
  PYTHONPATH=src .venv/bin/python \
    scripts/experiments/rtgs010_bench019_adapters.py verify-adapter \
    --adapter "$adapter" --verify-sources
done
```

The TUM policy uses strict `<20 ms` RGB/depth association, at-most-20-ms pose interpolation,
inclusive `0.08 m OR 8 degree` keyframes, endpoint-preserving integer half-up selection,
the [officially recommended ROS-default calibration for pre-registered RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)
in the repository half-integer convention, and an inclusive `[0.3, 5.0] m` registered-depth mask.
Materialization is development-only and exclusive-new. Archive validation admits only exact
ordinary file/directory member types with no parsed sparse metadata; materialized replay rejects
symlinks, special nodes, and undeclared lexical entries.
Karate is intentionally absent until a source-backed mask policy is approved.
