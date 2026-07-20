# Three-iteration R&D prompt

## Reusable controller prompt

> Act as the research engineer for `realtime-gs`. Read `CLAUDE.md`, `docs/RESEARCH.md`,
> `docs/ROADMAP.md`, and `docs/EXPERIMENTS.md` before changing code. Preserve unrelated worktree
> changes. Query the Scholar Inbox digest for recent, relevant papers and distinguish paper-stated
> results from implementation ideas you infer. Choose one unresolved question that is falsifiable,
> CPU-testable, and small enough to measure rigorously.
>
> Run exactly three evidence-driven iterations. Before each iteration, write its hypothesis,
> control arms, deterministic seeds, train/held-out split, primary metric, minimum meaningful effect,
> and falsification rule. Make the smallest pluggable implementation needed; keep CUDA optional and
> pipeline logic backend-independent. Reuse identical inputs across arms. Record exact commands,
> effective configuration, revision/dirty state, wall time, quality, geometry, and primitive count.
> Treat negative and null results as first-class results.
>
> After iteration 1, revise the next prompt from the observed mechanism rather than defending the
> original hypothesis. After iteration 2, replicate across seeds and isolate any suspected failure
> mode. In iteration 3, try to falsify the surviving explanation under a realistic perturbation and
> one production-path interaction. Do not change a default from a one-seed result or a sub-threshold
> effect. Reject invalid protocols and rerun them explicitly.
>
> Each iteration may begin with synthetic mechanism evidence, but it is not complete until the
> surviving path is exercised on a calibrated scene under `dataset/` with frozen train/validation/
> held-out roles. Save initial/final PLYs and preview artifacts, launch `rtgs view`, and include the
> exact viewer command in the handoff. Use exact rasterizer metrics for decisions; use the WebGL
> viewer to inspect geometry and failure modes rather than as quantitative evidence.
>
> After every iteration, append `docs/EXPERIMENTS.md` and preserve machine-readable evidence under
> `benchmarks/results/`. At the end, update research/architecture/roadmap documentation, run
> `./scripts/verify.sh`, and report supported, refuted, and unresolved conclusions separately. If a
> choice cannot be resolved from repository rules or evidence and would materially change the
> research objective, pause and ask the researcher for direction.

This controller prompt was executed on 2026-07-14 with the repository's
`.claude/skills/experiment/SKILL.md` workflow. Each iteration-specific revision below was written
before the next implementation or experiment, and every numerical result is preserved under
`benchmarks/results/` and summarized in `docs/EXPERIMENTS.md`.

## Literature grounding

The initial covariance question follows the depth-to-covariance trail already documented in
`docs/RESEARCH.md`: pixelSplat and SplaTAM use depth-scaled pixel footprints, while no cited method
provides a closed-form along-ray dimension for a fitted 2D Gaussian. A 2026-07-14 Scholar Inbox
sweep added two closely related mechanism checks. DP-GS propagates only high-confidence depth under
normal guidance and explicitly treats abnormal depth edges; Incremental Gaussian Triangulation
constrains Gaussians to local planes and aligns their shortest axes with surface normals. Those
papers do not establish which covariance construction wins here, but they strengthen the need to
separate valid local surface slope from invalid depth boundaries and to test surface-aligned
primitives rather than assume them.

## Iteration 1 — falsifiable screen

> Act as the repository's research engineer. Choose one unresolved, falsifiable CPU-first
> question from the roadmap; state the hypothesis and controls before editing; make the smallest
> pluggable change needed to expose the variants; reuse identical fitted 2D Gaussians; enforce a
> train/held-out split; compare initialization and fixed-budget refinement with deterministic
> seeds; log negative results; change no default on one-seed evidence; and finish with tests,
> docs-sync, and an evidence-based next prompt.

Applied question: does depth-gradient-aware covariance beat a no-gradient, globally isotropic
ray-thickness control, and how does the current surface-Jacobian construction compare?
The predeclared primary hypothesis required footprint to improve mean held-out initialization by
at least 0.25 dB, win at least two of three seeds, and retain at least 0.10 dB after refinement.

The seed-0 screen falsified the expected footprint win. Train-selected isotropic reached
20.95 dB held-out initialization PSNR, footprint 20.45 dB, and surface 19.58 dB. Surface's p99
covariance condition number was about 843k and its largest scale was 2.64 times the scene extent.

## Iteration 2 — diagnose and replicate

> Treat the one-seed ranking as a mechanism-finding result, not a conclusion. Quantify whether
> invalid zero-depth boundaries cause the surface arm's scale explosion; introduce one opt-in,
> validity-aware finite-difference rule with no held-out tuning; preserve the old behavior as the
> control; then run paired clean-depth seeds 0–2 with identical fits, held-out cameras, no
> merge/density, and short equal refinement. The claim survives only if it replicates across
> seeds and after refinement.

The raw three-seed result replicated the failure. Validity-aware gradients then improved mean
surface initialization from 19.74 to 21.20 dB and reduced p99 condition from about 1.76 million
to 721. After 60 equal refinement steps, footprint/isotropic/surface converged to
26.25/26.21/26.04 dB, so the covariance ranking was not decisive.

## Iteration 3 — stress and production interaction

> Try to falsify the clean-depth result. Freeze the validity rule and test blurred depth plus
> deterministic 2% multiplicative noise across seeds 0–2. Then exercise normal voxel merging and
> one short density-control event. Treat a ranking reversal or sub-0.25 dB effect as inconclusive;
> retain modes rather than selecting a winner. Adopt only a mechanism-level correctness fix if it
> consistently removes invalid-boundary failures without harming noisy-depth behavior.

The noisy-depth causal run reversed the clean initialization/refinement ordering. A first
80-step production run was rejected because density surgery occurred on the final step with no
recovery. The corrected 100-step run allowed 20 recovery steps: train-tuned isotropic won all
three seeds after merge+density (27.05 dB mean), ahead of footprint (26.49) and surface (26.28).

## Decision

Validity-aware gradients are now the default because they fix a measured invalid-depth boundary
failure across seeds and remain stable under perturbed depth. The covariance mode remains
`surface`; `footprint` and explicitly sized `isotropic` stay available through `DepthLifter` for
real-depth validation. No general ranking is claimed from synthetic scenes.
