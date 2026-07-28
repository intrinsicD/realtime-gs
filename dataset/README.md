# Local calibrated data

Compact `.rtgsv` bundles and calibration records are repository fixtures. Raw `rgb/` and `mask/`
payloads are local data and are ignored to keep hundreds of megabytes out of Git.

For the stage capture, the canonical image-supervised selection is:

- RGB: `rgb/C####.jpg`
- mask: `mask/mask_C####.png`
- calibration: the nearest `calibration_dome.json`
- compact input: `gaussians2d/manifest.json` plus its declared `.rtgsv` files

PNG is the canonical mask because JPEG compression changes boundary values. The current PNGs are
8-bit soft masks (all 256 levels occur in sampled files), not binary masks; training may retain
that alpha, while binary IoU/leakage metrics must freeze and report their threshold. JPEG mask
copies may remain as local source backups but are never selected by a task or data seal. A file
beginning with `mask_` does not belong in `rgb/`.

Do not infer an experiment cohort from whatever files happen to be present. Every official task
names the exact dataset paths, camera split, patterns, and data seal under `experiments/`.
