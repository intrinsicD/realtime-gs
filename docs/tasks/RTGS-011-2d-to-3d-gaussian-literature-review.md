# Current Task

## Title

State-of-the-art review of 2D-Gaussian-to-3D-Gaussian reconstruction

## Task ID

RTGS-011

## Role Assignment

- Driver: Codex-literature-driver
- Reviewer: repository-owner
- Turn: none

## Mode

Explore

## Risk

Protected

## Maturity

- Target: Not applicable
- Reached: Not applicable

## Goal

Produce a deep, current, primary-source-backed literature review of methods that transform fitted
or predicted 2D Gaussian fields into 3D Gaussian fields, and derive evidence-calibrated design
recommendations for quality, wall-clock performance, and convergence under masked and unmasked
inputs.

## Motivation

The repository's existing `docs/RESEARCH.md` is broad and experiment-chronological. It does not
provide a self-contained taxonomy or a protocol-aware comparison focused on the exact 2D-to-3D
Gaussian transformation requested by the owner. The field also changes quickly, so the July 2026
survey and manuscript novelty ledger need a fresh, adversarial pass before related-work or design
claims are reused.

## Success Criteria

- A dedicated Markdown review defines the problem and disambiguates fitted image Gaussians,
  pixel-aligned predicted splats, 2D surfels, and ordinary image-to-3DGS methods.
- It surveys direct and adjacent primary literature through 2026-08-04, records the search scope,
  and links each substantive external claim to a paper, supplement, project page, or official
  repository.
- Evidence tables separate input assumptions, masking, geometry source, output representation,
  optimization, reported runtime/convergence, quality evidence, and implementation availability.
- Cross-paper performance numbers retain their dataset/hardware/protocol context; missing or
  incomparable evidence is explicit.
- The synthesis gives separate masked and unmasked recommended pipelines, Pareto choices for
  quality/speed/convergence, failure modes, ablations, and cheapest falsifying experiments for
  the repository.
- The repository documentation index points to the new review, links pass, the diff is reviewed,
  and `./scripts/verify.sh` passes.

## Constraints

- Documentation and task-state changes only; no implementation, benchmark, protected run, claim
  promotion, or default change.
- Use primary sources for technical and quantitative claims and record a dated search cutoff.
- Describe reported results as authors' reports, not independently reproduced facts.
- Preserve the user's unrelated `.idea/` changes.

## Non-Goals

- Running experiments or choosing a production default.
- Claiming universal state of the art from incomparable benchmark protocols.
- Treating masks as ground-truth geometry or conflating 2DGS surfels with 2D image-Gaussian
  decompositions.

## Selected Skills

- rtgs-core
- rtgs-task-workflow
- rtgs-research-ideation
- rtgs-review
- realtime-gs-results-audit
- rtgs-docs-sync
- rtgs-verify
- research-manager

## Experiment Contract

None

## Current Evidence

- Starting state: commit `36630c7fef14c0907134d2f3c532be3da4a0c43e`; active task record was
  the unchanged template.
- Unrelated pre-existing worktree changes: `.idea/rtgs.iml`, `.idea/vcs.xml`, and
  `.idea/pyLspTools.xml`; they are outside this task and will be preserved.
- Existing context: `docs/RESEARCH.md`, `docs/DESIGN_field_lift.md`,
  `docs/20260725_claims_and_questions.md`, `docs/ROADMAP.md`, `docs/EXPERIMENTS.md`, relevant
  ADRs, and the ARA ledger.
- Primary-source search cutoff: 2026-08-04. The closest direct method is G²SR
  (`arXiv:2607.14470`); it closes the old broad novelty claim. The narrower gap concerns arbitrary
  independently fitted teacher fields, especially field-only lifting after source RGB is removed.
- New artifacts: `docs/LITERATURE_REVIEW_2D_TO_3D_GAUSSIANS.md` and
  `docs/RESEARCH_PORTFOLIO_2D_TO_3D_GAUSSIANS.md`.
- Portfolio validator and self-test pass; all 37 unique external URLs extracted by the final link
  command returned HTTP 200; `docs_sync`, `check_ara`, and `check_agent_workflow` pass before the
  final full verification.
- `./scripts/verify.sh` passes after the completed review: Ruff check/format, the complete non-slow
  CPU test selection, docs sync, ARA, script layout, agent workflow, and experiment contracts are
  green. The two emitted PyTorch warnings pre-existed this documentation-only diff.
- Per `docs/tasks/README.md`, a self-reviewed provisional task remains in the active slot until a
  distinct reviewer or human accepts it; `check_agent_workflow.py` rejects provisional archives.
  RTGS-011 therefore remains active and provisionally accepted rather than fabricating independent
  acceptance.
- Follow-up search driver (2026-08-04): the owner requested a Scholar Inbox and cross-domain pass
  spanning tomography, 2D↔3D probabilistic models, astronomy, mixture registration, navigation,
  silhouette reconstruction, and validation. This reopens the documentation review without
  authorizing experiments, implementation, or claim promotion.
- Scholar Inbox discovery was filtered through primary sources. The strongest novelty threats are
  astronomy MGE, Panaretos random tomography, and cryo-EM mixed-dimensional GMMs; the strongest
  transfer candidates are projective mixture transport, forward-operator calibration,
  nonlinearity-triggered split/KL merge, probability-map masks, independent-half validation, and
  reversible view/region scheduling.
- GPS-Gaussian is now classified explicitly as an adjacent masked, human-specific, depth-supervised
  pixel-parameter-map route rather than a converter of sparse independently fitted 2D fields.
- After the extension, the focused review is 11,571 words and the portfolio is 6,945 words. All 56
  unique external URLs resolve with HTTP 2xx/3xx; the portfolio validator/self-test, `git
  diff --check`, docs/ARA/layout/workflow checks, and `./scripts/verify.sh` pass. The canonical
  verification ran CPU-only, completed with exit 0, and emitted the same two pre-existing PyTorch
  warnings recorded by the earlier pass.

## Minimal Plan

1. Completed: map repository terminology, evidence boundaries, unresolved novelty claims, and existing
   negative results.
2. Completed: search direct and adjacent primary literature with a reproducible query log and verify the
   closest methods in full text or official sources.
3. Completed: extend the review through Scholar Inbox discovery and primary-source verification of
   cross-domain novelty threats and transferable mechanisms.
4. Completed: rerun citation/claim audit, documentation checks, full verification, and self-review.
5. Completed: restore provisional acceptance after the extension passes review. Independent/human
   acceptance and archival remain a later workflow action.

## Status

Accepted

## Human Decisions

### Accept the literature deliverable as the basis for implementation

#### Question

Does the owner accept RTGS-011's documentation deliverable so implementation can proceed in a
separate protected task without treating any proposed mechanism as scientifically validated?

#### Options

- Accept the documentation and open a new implementation/experiment task.
- Keep RTGS-011 provisional and do not begin implementation.

#### Recommendation

Accept only the documentation deliverable. Preserve every synthesized pipeline component as an
untested hypothesis until prospective experiment review and outcome audit.

#### Decision

Accepted. The owner's 2026-08-05 instruction to lay out and implement the pipeline, then create its
experiment, adopts the review as the design input. It does not affirm novelty, quality, speed,
convergence, or a production default.

#### Date

2026-08-05

## Handoff Log

Append Driver handoffs, Reviewer verdicts, and session-completion entries in chronological order.
Use `###` for entries and `####` for their fields so entries remain nested below this section.
Never delete earlier entries. On terminal closeout, archive the complete record as
`docs/tasks/<task-id>-<slug>.md`, change the archived `Turn` to `none`, and reset this file to the
unchanged template.

### 2026-08-04 scientist pass — external literature claims

#### Scope

Adversarial claim-language and source-binding pass over the new review, portfolio, README index,
and the corrected initialization section in `docs/RESEARCH.md`. This task did not run a repository
experiment or benchmark; every performance number is an external author's reported result and is
explicitly labeled as such.

#### Claim inventory and disposition

| Claim family | Kind and scope | Evidence | Source/protocol bound? | Disposition |
| --- | --- | --- | --- | --- |
| G²SR directly creates 3D surfels from tracked image-plane Gaussian splats | External asserted capability; 2–3 posed RGB, 2026 preprint | `https://arxiv.org/abs/2607.14470` | Yes; detector, flow, calibration, thin-surface prior, and no arbitrary-field input are stated | Confirm with scope |
| G²SR geometry/memory/rate and rendering tables | External author-reported RTX 4090/384×512 results | G²SR Tables I–III | Yes; detailed 68.9–91.1/s values, valid-pixel alpha threshold, and coverage retained | Narrow; note abstract's 69–89/s summary and lower coverage/LPIPS |
| EDGS reaches original-3DGS quality in 15% time and reports 35% lower LPIPS when trained longer | External author-reported A100 result | CVPR 2026 paper / `arXiv:2504.13204`, Fig. 1 and Tables 1–2 | Yes; initialization is included and its ~120 s/15 GB cost is retained | Correct stale 25% wording to 15%; separate ratio from consumer-GPU wall-clocks |
| Stage-1 fitting speed/quality | External paper results on paper-specific datasets/hardware | GaussianImage, Image-GS, Instant-GI, SGI primary papers | Yes; highlighted-image versus aggregate results distinguished | Confirm as Stage-1 only; no geometry-transfer inference |
| Masked-object quality, mask ablation, count, and runtime | External paper results | GaussianObject and `arXiv:2603.14316` | Yes; visual-hull/repair and object/full-scene scope differences retained | Narrow; no causal attribution of full-pipeline gap to hull alone |
| Feed-forward and fast-training performance | External paper/project reports across incomparable protocols | Primary CVF/arXiv/project links in review | Yes at paper level, not cross-paper reproducible | Confirm only as protocol-specific Pareto examples; recent preprints labeled |
| Two-view covariance rank is generically five; three generic views recover rank six | Analytic assertion for known mean/Jacobians | Projection derivation in review plus local float64 numerical rank check | Yes; 100 random trials gave ranks 3/5/6/6 for 1/2/3/4 views | Confirm as linearized identifiability result, not real fitted-field evidence |
| Arbitrary fitted-field and field-only converter not located | Dated search-qualified absence statement | Search protocol and source inventory through 2026-08-04 | Search-bound, not proof of absence | Narrow; retire old broad novelty claim because G²SR closes it |

#### Corrections made

- Corrected EDGS from the stale 25% figure to its paper's 15% headline and separated this from
  consumer-GPU wall-clock references.
- Corrected the ellipse precursor to Mai, Hung, and Chesi with DOI
  `10.1016/j.patcog.2009.07.003`.
- Corrected the G²SR author list and exposed the detailed-table versus abstract throughput range.
- Replaced “proves,” “best,” and universal-ranking language with author-reported,
  protocol-qualified wording.
- Added explicit coverage, valid-pixel, model/input, preprocessing, and code-availability caveats.

#### Checks executed

- Reopened the primary PDF text for G²SR, EDGS, GaussianImage, Image-GS, Instant-GI,
  GaussianObject, FastGS, and the probabilistic object preprint and checked every retained headline
  value against its table or prose.
- Ran a float64 numerical projection-operator audit over 100 random trials per view count; generic
  rank was 3 with one view, 5 with two, and 6 with three or four.
- Extracted external URLs from both new artifacts and checked redirects/status in parallel; all 37
  unique URLs returned HTTP 200.
- Ran the portfolio validator and self-test, `git diff --check`, `scripts/docs_sync.py`,
  `scripts/check_ara.py`, and `scripts/check_agent_workflow.py` successfully.

#### Remaining evidence boundary

No paper result was reproduced locally, no CUDA/GPU benchmark ran, no repository claim was added to
the ARA ledger, and no default or maturity changed. The recommended combined pipeline and every
portfolio candidate remain hypotheses requiring prospective protected experiments.

### Handoff

#### Objective

Review the completed literature synthesis for exact problem taxonomy, primary-source fidelity,
masked/unmasked coverage, protocol-aware performance comparisons, calibrated novelty language,
and repository workflow consistency.

#### Reviewed state

Base commit `36630c7fef14c0907134d2f3c532be3da4a0c43e`; documentation-only worktree state bound by:

- `README.md`: `07d6fafa8548cd4efd7ac1f84792a1486ca8487b431e945dd8cd8f143b6b82bf`;
- `docs/RESEARCH.md`: `d57cdea273af733eea02fd7ef439dc1bdf6e372630578cef31091de27472231b`;
- focused review: `f20cc082cfd0547c7f2a868d40153abb474fd3d3395f90f0b6435ab4faa4401c`;
- research portfolio: `3722bc29d8351a53dcf0f934a79ef6556087a5bf098583bdffc4cd7a6b96d2e5`.

The pre-existing `.idea/` modifications are outside the reviewed task diff and remain untouched.

#### Changes

- Added an 8,000-plus-word focused review with formal identifiability analysis, a direct/precursor/
  adjacent taxonomy, quantitative protocol tables, masked and unmasked pipelines, performance and
  convergence guidance, a Pareto matrix, evaluation design, and decisive experiments.
- Added a structured research portfolio with eleven candidates/transfers, three new-evidence
  programs, novelty threats, null hypotheses, killing tests, abandonment rules, and a recommended
  first experiment.
- Linked both artifacts from README and `docs/RESEARCH.md`.
- Corrected the stale broad novelty statement for G²SR and the EDGS 25%→15% headline.

#### Evidence

- Primary papers and official sources were searched through 2026-08-04; quantitative headlines
  were reopened in source PDF text during the scientist pass.
- All extracted external links returned HTTP 200.
- The research-portfolio validator and its self-test pass.
- `git diff --check` and `./scripts/verify.sh` pass.

#### Assumptions

- Author-reported external values are useful when their source protocol and non-reproduced status
  are explicit.
- Search-qualified absence is acceptable as a dated gap statement but not as proof of novelty.
- Documentation-only work does not require a repository experiment contract or ARA result row.

#### Uncertainties

- Several July/August 2026 methods are preprints and may change after review or code release.
- Terminology outside “Gaussian Splatting” may hide additional mixture-tomography or compressed-
  domain precedents.
- The synthesized hybrid pipeline has not been implemented or measured as a combination.

#### Review Focus

Challenge the covariance-rank derivation, arbitrary-field gap language, G²SR/EDGS protocol
boundaries, valid-pixel coverage caveat, and whether any adjacent image-to-3DGS work is accidentally
presented as direct fitted-field lifting.

#### Protected actions not taken

No benchmark, experiment, source-code change, default change, maturity promotion, commit, branch,
push, or external publication was performed.

#### Recommended Next Action

If the owner wants implementation, open a separate protected task for the portfolio's exact
field-shape utility and identifiability experiment before building a large learned converter.

### Review

#### Verdict

Accepted

#### Self-reviewed

Yes

#### Correctness

The direct/precursor/adjacent distinctions are explicit, G²SR is treated as the closest direct
method without conflating it with arbitrary pre-fitted fields, and the mathematical two-view
covariance null direction was independently checked numerically. Mask semantics, alpha semantics,
coverage, and single-view ambiguity are stated correctly at the review's evidence level.

#### Evidence Quality

Primary sources bind every retained headline. External figures are identified as authors' reports,
hardware/dataset/timing boundaries are kept near the claims, and recent preprints are labeled. The
scientist pass corrected three source/protocol errors before closeout. No external result is
presented as locally reproduced.

#### Simplicity

Two focused artifacts are justified: the long review is a durable synthesis, while the portfolio
must follow the repository's executable idea-card schema. README and the chronological research
survey only receive links and necessary corrections.

#### Missing Cases

No dynamic/4D reconstruction, generative text-to-3D, SLAM, or exhaustive license audit is included;
these are outside the requested static field-lifting scope. The review does not benchmark methods
under one implementation, so its Pareto guidance remains protocol-aware rather than a leaderboard.

#### Required Changes

None for provisional self-reviewed closeout.

#### Optional Improvements

An independent reviewer can later promote the task from provisional status. Re-run the dated search
before manuscript submission because G²SR, ATSplat, QuerySplat, and the probabilistic object method
are fast-moving preprints.

### 2026-08-04 Scholar Inbox and cross-domain extension — scientist pass

#### Scope

Adversarial source and transfer audit over the follow-up search requested by the owner. Scholar
Inbox was used as a discovery surface; technical claims were retained only after reopening primary
papers or official sources. The extension remained documentation-only and did not run a repository
experiment, GPU benchmark, or protected data phase.

#### Claim inventory and disposition

| Claim family | Kind and scope | Primary evidence | Disposition |
| --- | --- | --- | --- |
| GPS-Gaussian converts 2D parameter maps to 3DGS in real time | External capability; masked human, two rectified views, amortized learned prior | CVPR 2024 paper, `arXiv:2312.02155`, project/code | Confirm as adjacent only; retain foreground matting, scan-depth supervision, source RGB, 40K+100K training, RTX 3090 timing boundaries |
| MGE analytically deprojects fitted 2D Gaussian brightness mixtures | External analytic result under a prescribed intrinsic family/view geometry | Cappellari 2002, `astro-ph/0201430`, equations 6–10 | Confirm as restricted prior art; retain non-uniqueness and `cos²(i) < q'²` feasibility condition |
| Random tomography recovers 3D radial-mixture center shape from 2D profiles | External statistical consistency result | Panaretos 2009, `arXiv:0909.0349`; Panaretos & Konis 2011 | Confirm; retain isotropic/random additive profiles, radial kernels, distinct-weight labeling in the base method, Gram recovery, and orthogonal gauge |
| e2gmm is mixed-dimensional 3D-GMM↔2D-image precedent | External peer-reviewed capability | Nature Methods 2021, DOI `10.1038/s41592-021-01220-5` | Confirm; restrict to known-orientation cryo-EM density images and many observations |
| CT Gaussian work provides correct projection and warm-start lessons | External tomography mechanisms and author-reported performance | R²-Gaussian, exact Gaussian ray tracing, FaCT-GS | Confirm only as a donor; additive X-ray line integrals and protocol-specific timing cannot be transferred to alpha radiance |
| Mixture OT/EM improves field association | Synthesized hypothesis from theory/registration | Delon–Desolneux, JRMPC, CPD | Keep as falsifiable transfer; no recipient performance claim |
| Nonlinearity-directed splits, KL merges, support mass, ordered views, and freezing improve the pipeline | Synthesized mechanisms | Kulik–LeGrand, Runnalls, OSEM, Manifold-GS, incremental online GS | Keep as ablation candidates; explicitly record broken alpha/visibility correspondence and native baselines |
| Probability masks and independent-half reconstruction improve evidence quality | External donor mechanisms plus recipient protocol hypothesis | Tabb CVPR 2013; Scheres–Chen Nature Methods 2012 | Confirm donor behavior; do not call camera-half agreement a resolution metric |

#### Corrections and novelty disposition

- Retired the broad implication that 2D-Gaussian-mixture deprojection is unexplored: MGE, random
  tomography, and cryo-EM GMM work are material prior art.
- Preserved the narrower gap: arbitrary independently fitted compact radiance fields under few-view
  perspective geometry, partial visibility, alpha/color semantics, and a demonstrated advantage
  over point/depth initialization.
- Classified GPS-Gaussian as a strong masked feed-forward speed/quality reference, not a sparse
  fitted-field converter or an unmasked general-scene method.
- Separated exact donor mechanisms, speculative transfers, and recent Scholar Inbox preprints so no
  component paper's result is inherited by the proposed hybrid pipeline.

#### Checks executed

- Reopened GPS-Gaussian PDF passages for >25 FPS at 2K, 27 ms source + 0.8 ms per novel view,
  40K depth pretraining + 100K joint training / about 15 hours, foreground matting, and depth
  supervision.
- Reopened MGE equations and the non-unique/feasible-deprojection text; reopened Panaretos' radial
  mixture, Gram inversion, distinct-weight labeling, and consistency passages.
- Reopened R²-Gaussian's additive line-integral composition, covariance-related integration bias,
  FDK warm start, and paper-specific runtime claims.
- Checked all 56 unique external URLs from the two artifacts; every URL returned HTTP 2xx/3xx after
  replacing a bot-blocked publisher link with the author's arXiv version.
- Portfolio validator and self-test, `git diff --check`, docs sync, ARA, script-layout,
  agent-workflow, experiment-contract, Ruff, format, and the complete non-slow CPU suite pass under
  `./scripts/verify.sh` (exit 0). No CUDA/GPU test or timing run was performed.

#### Reviewed state

- focused review SHA-256:
  `997606ab73c1df13b7a2f87b03adaa47680ed25a63d3c2d79911a6590dd44363`;
- research portfolio SHA-256:
  `6b15a7c0fbe3760082669614a356b90377db009f06f0b0f2fc7b9b4675a03f78`.

#### Remaining evidence boundary

No cross-domain transfer was implemented or measured. The proposed mixture transport,
nonlinearity-aware topology controller, geometric support mass, independent-half protocol, and
ordered/freezing schedule remain hypotheses with explicit killing tests. Recent 2026 preprints may
change, and the dated search is not proof that no further precedent exists.

### Handoff — cross-domain extension

#### Objective

Review whether the Scholar Inbox extension correctly distinguishes exact prior art, transferable
mechanisms, and pipeline hypotheses while preserving masked/unmasked, quality/speed/convergence,
and protocol boundaries.

#### Changes

- Added GPS-Gaussian to the implementation snapshot and pixel-aligned family with its input,
  training, mask, and runtime constraints.
- Added a cross-domain prior-art/mechanism section covering astronomy, random tomography, cryo-EM,
  CT, mixture transport/registration, navigation/tracking, probability silhouettes, validation,
  ordered subsets, and online freezing.
- Extended the architecture, ablations, metrics, novelty ledger, and portfolio with recipient
  mappings, broken correspondences, native baselines, predictions, counter-analogies, and killing
  tests.

#### Protected actions not taken

No source-code change, experiment, benchmark, claim promotion, default/maturity change, commit,
branch, push, or external publication was performed. The user's unrelated `.idea/` changes remain
untouched.

#### Recommended Next Action

Before implementing a large learned model, run the portfolio's restricted analytic sanity cases
and exact synthetic field-shape/association study. Gate mixture transport and full covariance on
measured precision×coverage and held-out geometry; gate every convergence donor on wall-clock
quality–time AUC against its native baseline.

### Review — cross-domain extension

#### Verdict

Accepted

#### Self-reviewed

Yes

#### Correctness

The strongest older precedents now bound novelty, GPS-Gaussian is correctly adjacent, additive
density and alpha radiance are not conflated, and every transfer states where its causal analogy
breaks.

#### Evidence Quality

Externally reported numbers remain attributed and protocol-bound; exact analytic assumptions were
reopened in primary text; recent Scholar Inbox results are labeled as preprints. No synthesized
pipeline claim is presented as repository evidence.

#### Simplicity

The existing review and portfolio were extended rather than creating another artifact. The
recommended system remains staged, and expensive transfers must earn inclusion in isolated killing
tests.

#### Missing Cases

The pass is not an exhaustive review of dynamic/4D, generative, or SLAM systems, nor a proof that
all astronomical/statistical tomography precedents were found. Those areas are outside the static
field-lifting question unless a later search identifies a closer functional match.

#### Required Changes

None for provisional self-reviewed closeout.

### Review

#### Verdict

Accepted

#### Self-reviewed

No

#### Correctness

The owner accepted the documentation as design input by explicitly requesting that the synthesized
pipeline be laid out, implemented, and assigned an experiment. This workflow acceptance does not
validate any novelty, quality, speed, convergence, or production claim from the synthesis.

#### Evidence Quality

No new evidence was introduced. The external-source boundaries and the earlier self-review remain
unchanged; all transferred mechanisms enter RTGS-012 as untested hypotheses.

#### Simplicity

RTGS-011 closes as documentation. Implementation and experimental work move to the separate
protected RTGS-012 record instead of changing the scope of the archived review.

#### Missing Cases

The owner acceptance is not an independent scientific audit and does not authorize outcome access
for RTGS-012's protected experiment.

#### Required Changes

None for the documentation handoff.

#### Optional Improvements

Re-run the dated literature search before manuscript submission; this acceptance only closes the
current documentation handoff.
