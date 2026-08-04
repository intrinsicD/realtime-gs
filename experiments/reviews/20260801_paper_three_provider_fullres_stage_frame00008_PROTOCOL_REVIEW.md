# Prospective Protocol Review

- Task ID: `20260801_paper_three_provider_fullres_stage_frame00008`
- Protocol SHA-256: `eb78742c579930b6c04d4bd538ae33c6287ad0fb7f59190ab33adaef777a4fc1`
- Reviewer: `Hegel`
- Verdict: `rejected`
- Outcome Access: `none`

## Scope

This V2 review covers only the frozen development protocol, its 85-file compact-input seal, the
reviewed source boundary, image-free construction/training/evaluation controls, resource and
failure receipts, presentation/viewer requirements, and convergence/compression definitions. It
does not assess an RTGS-008 outcome. Even after repair, this one outcome-exposed frame could support
only a source-bound within-frame description of provider-native fidelity and visible signatures;
it cannot support provider superiority, generalization, compact-VRAM superiority, a production
default, Original 3DGS/COLMAP performance, or paper-level quality.

## Checks

- Verified a clean `rtgs/008-review-v2` worktree at review package
  `41ae4b4e0c20fa8d0b02015896ea902cdad08a56`; implementation checkpoint
  `40eb4c0b809c85c4e8d3669b49a34bab4860266d` is its ancestor by two metadata-only commits.
- Recomputed protocol digest
  `eb78742c579930b6c04d4bd538ae33c6287ad0fb7f59190ab33adaef777a4fc1` and independently
  rehashed all 85 unique regular sealed files: 78 `.rtgsv` plus seven JSON files, exactly
  24,831,997 bytes, data-seal SHA-256
  `645e2991232232db4425daa14a10fe5239263532d18060cf530859857b23507a`.
- Recomputed every frozen source hash. The six registered hashes match exactly, and the reviewed
  checkpoint is an ancestor of the review package.
- Reloaded all 78 compact fields with alpha disabled under the live no-image guard. Every provider
  reproduced the frozen 26-view, disjoint 23-train/3-heldout partition and 5328-by-4608 canvas;
  GaussianImage remained native/additive at 11,000 rows, mask-contained remained
  StructSplat/normalized at 11,000 rows, and no-boundary remained StructSplat/normalized at
  5,000--8,592 rows. All four negative controls fired and no real image/import denial occurred.
- Audited every Goodall correction. The hypothesis is narrowed to provider-native fidelity;
  canonical workers select one initializer; resource fields and raw repeats are frozen; root
  contact-sheet/reconstruction/orbit/elevation artifacts are required; the viewer command consumes
  a nine-method comparison manifest; worker and in-matrix root failures have structured paths; and
  provider-conditioned fit windows, proposal distributions, and realized coordinates are explicit.
- Independently exercised the frozen NVML sampler without a CUDA allocation. Its three-sample idle
  guard passed, recorded no foreign compute process, and emitted the frozen driver, device,
  background-memory, utilization, and sampling fields.
- Audited held-out convergence as a descriptive, no-stopping endpoint: fixed 1,024-sample
  provider-native coordinates per view/checkpoint, final-to-best ratio, stable-band step, final-tail
  relative change, and a prospectively fixed convergence rule. Audited compression as sealed
  provider `.rtgsv` bytes divided by final `gaussians.npz` bytes, with both byte counts retained and
  no rate-distortion interpretation.
- Passed 12 narrow initializer/driver tests and 53 broader outcome-free experiment-contract,
  compact-view, initializer, density, and driver tests. The final bound-environment
  `./scripts/verify.sh` gate also passed after the rejected review metadata was installed.
- Ran adversarial pure/mocked guards without creating a run. They proved that the current source
  binding reads only its six registered files, that several behavior-bearing dependencies are not
  read by that guard, and that a synthetic `validate-data` failure exits `_orchestrate` without
  invoking `_publish_failed_run`.

## Findings

The V2 protocol remains blocked. Goodall's hypothesis, single-arm construction, resource schema,
root presentation, synchronized viewer, and sampling-wording corrections are substantively
present, and the convergence/compression additions are sufficiently precise. Three fail-closed
gaps remain:

1. **Blocking / high -- the prospective source binding is not exact.** The runtime guard hashes
   only `pyproject.toml`, the contract/driver, `paper_initializers.py`, and two tests while accepting
   any descendant of `40eb4c0...`. Canonical behavior also depends on unbound files including
   `compact_views.py`, `beam_fusion.py`, `splat_sfm.py`, `compact_trainer.py`,
   `compact_density.py`, `gsplat_points.py`, `cli.py`, and `viewer.py`. A clean descendant can alter
   one of those dependencies, receive a matching current `source_commit` at `init-run`, retain all
   six registered hashes, and pass `_source_binding_passes`. Bind the complete behavior-bearing
   source tree (or an equivalent exact reviewed tree digest with only enumerated review metadata
   excluded) before execution.
2. **Blocking / high -- structured root failure publication starts too late.** `_orchestrate`
   performs run binding, sealed-data validation, existing-output checks, and environment recording
   before its failure-publication `try`. The independent negative control made `validate-data`
   fail after a valid mocked binding and observed zero `_publish_failed_run` calls. A post-`init-run`
   data drift or environment/preflight failure can therefore leave an official attempt without the
   frozen run receipt, source records, and structured failure evidence. Enclose every trusted
   post-initialization phase in the one-shot failure publisher while preserving no-overwrite rules.
3. **Blocking / high -- the canonical input-boundary receipt has an unguarded load.**
   `_canonical_worker` performs its first `CompactDataset.load` for provider-semantic inspection
   before either guarded `_initialize_arm` or guarded `_train`, and it has no outer
   `NoImageGuard`. The later receipt nevertheless asserts `source_rgb_or_mask_opened: false`.
   The exact current loader replayed safely, but the live canonical evidence cannot establish that
   assertion for the whole worker, especially while the loader is outside the source hash set.
   Guard that initial load (or remove it in favor of already guarded evidence) and bind the guard
   record into the receipt.

No downstream execution should begin until a human explicitly authorizes a bounded third technical
revision and a fresh outcome-unseen prospective review approves its new digest.

## Protected Actions Not Taken

The reviewer did not invoke `init-run`, invoke the experiment's `run` command, create or enumerate
an RTGS-008 run directory, open any RTGS-008 run/result artifact or downstream outcome path, train a
cell, render a downstream model, or inspect a downstream metric, preview, report, or result. Checks
were limited to frozen protocol/source/input files, compact-input QA, outcome-free unit and
contract tests, a no-allocation NVML smoke, static inspection, and synthetic negative controls.
