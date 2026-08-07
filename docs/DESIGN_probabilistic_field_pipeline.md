# Probabilistic compact-field lifting pipeline

Status: opt-in research implementation under RTGS-012. Nothing in this document is a general or
positive quality, speed, convergence, novelty, or production-default claim.

This document turns the literature synthesis into an executable pipeline contract. It is designed
for calibrated per-view `GaussianObservationField` inputs, with optional masks, after source images
may already have been discarded. The implementation extends the existing field lifter rather than
creating a second renderer or optimizer stack.

## 1. Scope and exact boundary

Inputs:

- calibrated cameras;
- one immutable 2D Gaussian field per camera;
- an explicit disjoint train/held-out camera partition;
- optional binary or floating support masks;
- optional train-only bounds, depth priors, or sparse points.

Outputs:

- a persistent `Gaussians3D` field;
- geometric density/support mass kept separate from render opacity;
- source-lineage, visibility, observability, association, topology, and schedule diagnostics;
- optional independent-half reconstructions and a world-frame stability report.

The fitting path may use only training cameras and their compact fields. Held-out compact fields
are opened only after fitting for reporting. Source RGB, `SceneData`, a learned optical-flow model,
and the RGB trainer are outside this pipeline.

## 2. Pipeline at a glance

```text
per-view 2D Gaussian fields + cameras + optional support probabilities
                              |
                              v
0. strict input/split validation and train-only geometry boundary
   optional experiment-only deterministic field-component cap
                              |
                              v
0b. fixed-anchor feasibility: retain only source rays entering the train-only AABB
                              |
                              v
1. support-aware placement: compact carve or source-excluded field sweep
   exact all-source rejection -> optional task-scoped whole-cell retry unmasked
                              |
                              v
2. exact source-projection fibers for mean and covariance
                              |
              optional, opt-in shared-latent transport
                              |
                              v
3. projection-gated per-view UOT plans + source-view exclusion
   (fit on a cloned fiber; fail or roll back the whole stage)
                              |
                              v
4. visibility/gain-aware analytic field refit
   geometry first -> appearance later -> mandatory full-view cleanup
                              |
                              v
5. transactional topology proposals
   prune | Runnalls-screened representative merge |
   projection-nonlinearity ray-depth split | residual birth
                              |
                              v
6. frozen-teacher semantic validation and post-hoc correspondences
                              |
                              v
7. optional independent-half camera refits and stability-only comparison
```

The transport stage is not a globally coupled multi-marginal OT solver. It is a shared-latent,
per-view unbalanced-transport approximation: all views update one persistent 3D fiber set, but each
view has its own transport marginal. This naming is deliberate.

## 3. Stage contracts and what to expect

| Stage | Mechanism | Expected benefit if the hypothesis is right | Cost | Expected failure / diagnostic | Fallback |
| --- | --- | --- | --- | --- | --- |
| 0. Input boundary | `SceneFits` validates cameras, fields, masks, and a complete train/held-out partition | Prevents silent held-out leakage and provider-semantic drift | Negligible | Invalid dimensions, non-finite tensors, incomplete split, untrusted geometry | Fail closed |
| 0a. Bounded teacher proxy | Optional `target_component_cap` keeps an 8x8-stratified mass-area subset, then fills the budget globally by mass times footprint area | Makes 5k-100k-component teachers tractable for dense CPU association and exact mixture kernels | One deterministic sort per field; later stages scale with the cap | A 512-component proxy may omit low-mass detail and is not a complete-field quality estimate | Default `None` keeps every component; always report original/used counts and the selection digest |
| 0b. Forward-AABB anchor eligibility | Fixed-anchor sweep computes its train-only search AABB first, rejects component-center rays with no positive-depth intersection, then performs the same capacity-aware balanced, seeded top-mass-pool draw over eligible rays | Prevents an impossible source ray from aborting placement while preserving exact source projection and arm-comparable seeded lineage | One vectorized ray/AABB test per capped 2D component | A camera may contribute fewer anchors; the requested global track count may exceed all eligible rays | Report candidate/eligible/rejected counts per view; fail closed if the eligible total cannot meet `n_init_3d` |
| 1. Support-aware placement | Existing compact carve or source-excluded field sweep; a floating mask supplies geometric support probability | Masks should reduce obvious foreground/background proposals; unmasked inputs keep all proposals | Dominated by field queries | Hard masks delete thin structure; soft masks suppress valid low-probability boundary components; either can reject every placement source | Use `hard` or `none`; in the protected all-dataset retry, only the exact empty-source error permits one whole-cell unmasked retry, which is labeled operability-only |
| 2. Exact source fiber | One depth coordinate and three covariance-null coordinates preserve the selected source 2D Gaussian exactly | Removes source-view drift and exposes what other views must identify | Linear in tracks | A fitted 2D fragment may not correspond to one physical 3D primitive | Keep field-level objective; do not interpret lineage as truth |
| 3. Shared-latent UOT | Bhattacharyya component costs, a finite projection gate, explicit track/observation capacities, and dustbins | May improve cross-view support when fields have missing or unequal components | Pairwise per active view plus Sinkhorn iterations | Diffuse mass, wrong split/merge identity, or a geometry step leaving the valid camera domain | Entire stage runs on a clone; configured `raise` or explicit `rollback`. In the completion experiment, rollback preserves the untouched placement but the required-transport hard gate still records the candidate cell as failed |
| 4. Analytic field refit | Additive density/RGB-numerator objective with block-fixed visibility, view gain, observability gate, source-exact SH | Should refine geometry without requiring component identity | Quadratic mixture kernels, chunked | Proxy can disagree with normalized finite-support teachers; narrow baselines leave covariance null directions | Pin unobservable covariance coordinates and validate the frozen teacher separately |
| 4a. Progressive views | Greedy camera-baseline order grows the active loss set, then all views run for a frozen cleanup interval | May reduce early cost and contradictory gradients | Lower early per-step view count | Early subset bias or no wall-clock gain | `view_schedule="all"`; final cleanup makes the accelerator reversible |
| 5. Split | Six covariance cubature points measure disagreement between nonlinear perspective projection and the local affine/EWA Gaussian; the selected fiber is split along source-ray depth | Spend components only where one Gaussian is a poor projected approximation | Extra projections per topology round | The score may select a component whose split does not improve the actual field | Exact penalized field objective rejects it; native largest-mass split remains a control |
| 5. Merge | Candidate pairs are ranked by summed projected Runnalls KL upper bounds | Remove redundant components at bounded search cost | Pairwise score for capped candidates | Fibers from different cameras cannot be moment-averaged without losing their source invariant | Keep one representative fiber; exact field objective and parsimony decide acceptance |
| 5. Birth/prune | Residual teacher energy proposes birth; lowest density mass proposes prune | Repair missing modes and remove unsupported mass | One residual scan per round | Birth follows decomposition noise; prune removes thin structure | Exact transactional acceptance; set topology rounds to zero |
| 6. Semantics | Immutable teacher query equation is sampled independently from the analytic proxy | Detects proxy/renderer disagreement | Bounded sample cap | Good analytic loss with worse true teacher replay | Do not promote the analytic arm; use sampled metrics as the decision surface |
| 7. Independent halves | Alternating training-camera halves are fit independently in the same calibrated world frame and matched mutually | Exposes camera-subset instability and initialization dependence | Approximately two extra fits | Different coverage can disagree even when either result is locally valid | Report as stability only, never accuracy or resolution |

## 4. Masked and unmasked inputs

### 4.1 No mask

`mask_mode="none"` ignores all supplied alpha during placement. Background and missing components
must be handled through placement consensus, visibility, transport dustbins, and residual birth.
Expect lower precision near clutter and a larger topology burden. This is the required control for
any mask benefit.

### 4.2 Hard mask

`mask_mode="hard"` preserves the incumbent behavior: a source component is retained only when its
sampled mask value is nonzero. Retained components receive support probability one. This is fast
and appropriate for trusted object mattes, but brittle at thin boundaries.

### 4.3 Probability mask

`mask_mode="probability"` requires floating masks in `[0,1]` or accepts binary masks as the
degenerate case. The sampled value scales geometric density/support mass after a frozen minimum
support floor. It does not become render opacity, correspondence confidence, or an RGB weight.
The protected mechanism experiment therefore injects frozen false-positive and false-negative
rates into train-only nuisance/foreground support on actual synthetic Gaussian fields, runs the
production placement/mass/opacity/refit path, and evaluates a clean held-out Gaussian field.
Without that field-level test, probability masking would be only an untested reinterpretation of
alpha.

### 4.4 Exact empty-support fallback

A mask can be internally valid yet assign zero support to every deterministically selected
placement source. The strict library behavior remains to raise
`ValueError: support-mask policy rejected every field-placement source`; no default changes. The
all-dataset support-fallback experiment wraps that exact exception at its task-specific worker
boundary and retries the entire cell once with `mask_mode="none"`. For the independent-half seed,
the primary fit and both halves are all rerun unmasked so one report never mixes support semantics.
The worker resets Torch to the frozen cell seed before the retry, making the unmasked pass
equivalent to a deterministic fresh configuration apart from measured retry overhead.

Expect the fallback to recover an orbitable 3D result when the 2D Gaussian fields and cameras are
otherwise usable, at the cost of a second placement attempt and loss of mask-conditioned
precision. The rejected attempt remains inside process wall time. Every successful cell reports
requested/effective mask modes, retry count, number of unmasked fits, and a binary fallback curve.
If an unmasked retry produces a model that subsequently fails a hard invariant, its preserved PLYs,
failure record, boundary receipt, resource receipt, requested/effective config, report entry, and
orbit-viewer label all retain the same explicit fallback record and rejected status. Such a model is
inspectable failure evidence, never an accepted masked reconstruction. A successful seed from the
same arm does not hide it: every preserved rejection receives a seed-qualified, presentation-only
viewer entry and an explicit per-cell report note, while the accepted seed remains the arm's
representative model.
Such a cell is evidence that the pipeline can operate without the supplied support—not evidence
that hard or probability masking worked, and not evidence that unmasked quality is favorable. A
different exception, a repeated empty-support exception, or any failed retry still aborts the
root.

### 4.5 Fixed-anchor feasibility is not a support fallback

Fixed-anchor placement has a separate geometric domain: every selected 2D component defines a
camera ray, but that ray may never enter the train-only search AABB at positive depth. Such a ray
cannot produce a valid source-exact 3D anchor anywhere in the frozen search volume. The initializer
therefore computes the AABB before the seeded draw and makes forward intersection an eligibility
condition. It then allocates the requested count with the existing deterministic capacity-aware
round-robin quotas and samples from each eligible top-mass pool with the frozen seed.

This filter does not read held-out fields, masks, source RGB, quality metrics, or arm identity; it
is applied identically to native and candidate arms. It is not an outcome-contingent retry, does
not change mask semantics, and does not place an invalid ray outside the AABB. Every successful
fit records total and per-view candidate, eligible, and rejected counts plus policy version
`forward_search_aabb_intersection_v1`. If the eligible total is smaller than the requested 3D
track count, placement still fails closed instead of silently reducing the budget.

## 5. Geometry and covariance expectations

- One view fixes a source ray and three of six covariance combinations.
- Two generic views identify the mean but generically leave one covariance null coordinate.
- Three well-conditioned views can identify all six covariance coordinates in the local affine
  model.
- Full rank does not imply stability: condition number and PSD projection must be reported.
- The local affine covariance solve is a baseline/oracle for small Gaussians, not an exact
  perspective density inverse.
- The nonlinear cubature score measures when that approximation is suspect; it is not itself an
  error guarantee.

The protected experiment therefore separates center recovery, source-footprint lifting,
oracle-tracked five-sigma-point surfels, and rank-aware full covariance. Association corruption is
introduced only in a separate factorial so it cannot be credited to the covariance solver.

## 6. Association semantics

Four quantities remain distinct:

1. geometric compatibility cost;
2. observation/track transport capacity;
3. posterior transported mass and dustbin mass;
4. rendering opacity.

The finite projection gate is evaluated after projecting the current 3D fiber into a target view.
It is therefore stronger than an image-only epipolar-line gate after depth has been initialized,
but it can also reject the correct component when placement is wrong. The experiment compares:

- field-level refit with no component association;
- row-softmax with a dustbin;
- projection-gated unbalanced Sinkhorn with uniform capacities;
- projection-gated unbalanced Sinkhorn with explicit geometric mass capacities;
- a shuffled-candidate negative control.

The previous Iteration-3 transport result remains negative evidence. This implementation earns a
new run only because it adds stage-level transactionality, a finite projection gate, explicit
failure policy, native field-level control, and separate exact/refitted-field conditions.

## 7. Topology semantics

`density_mass` is the additive field/support quantity. `render_opacity` controls alpha compositing.
Splits conserve density mass and partition optical thickness; merges sum density mass and use the
alpha-union formula. A representative source fiber is retained during merge because averaging
coordinates from different camera fibers is not geometrically meaningful.

Every proposal is provisional. The scheduler evaluates the actual configured analytic field
objective plus a per-component parsimony cost and accepts at most one deterministic best move per
round. The sampled nonlinearity score only chooses a proposal; it cannot accept it.

## 8. Scheduling and convergence

The progressive schedule orders fitting cameras greedily by camera-center separation. It starts
with a frozen number of views, grows the active set across the early geometry/refit steps, and uses
all fitting views for the last `full_view_cleanup_iterations`. Appearance still activates only at
`appearance_start`. Report:

- objective versus optimizer step and elapsed time;
- active-view count per step;
- accepted/rejected steps and learning-rate backoffs;
- time to each quality threshold;
- final full-view and held-out semantic metrics.

Fewer early view evaluations do not establish faster convergence. The only acceptable speed claim
is lower wall-clock time-to-quality at matched hardware, implementation, stopping rule, and final
quality.

## 9. Independent-half stability

The wrapper partitions only the original training cameras into alternating original-order halves.
Each half sees the other half and the original held-out cameras only after fitting. Outputs remain
in the calibrated world frame, so the diagnostic uses mutual-nearest centers and reports:

- component counts;
- mutual-match count and matched fraction in each direction;
- median, p90, and RMS center distance;
- relative covariance discrepancy on matched pairs.

Agreement can be caused by a shared initializer or strong prior; disagreement can be caused by
coverage rather than error. The report is therefore a stability audit, not ground-truth geometry,
resolution, or correctness.

## 10. Public implementation surface

The opt-in controls are:

- `FieldLiftConfig.mask_mode` and `mask_probability_floor`;
- optional `FieldLiftConfig.target_component_cap` for explicitly approximate bounded studies;
- optional `FieldLiftConfig.association: FieldAssociationConfig`;
- `FiberFitConfig.max_pair_cost` for the finite projection gate;
- `FieldLiftConfig.topology_split_mode`, `topology_split_min_score`, and
  `topology_split_max_relative_depth`;
- `FieldRefitConfig.view_schedule`, `progressive_start_views`, and
  `full_view_cleanup_iterations`;
- `ProbabilisticFieldPipelineConfig.independent_half_validation` and stability controls;
- `run_probabilistic_field_pipeline(SceneFits, config)`.

All incumbent defaults disable the new behavior: no target cap is applied, hard masks retain their
old semantics, no transport stage runs, largest-density split selection remains in place, all
views are active for every refit step, and independent halves are not computed.

## 11. Protected experiment

The completed all-dataset successor is
`experiments/tasks/20260805_probabilistic_field_pipeline_association_rollback_mixed.json`. It
preserves the original task and its failed infrastructure/input/completion/support-fallback/AABB
predecessors as immutable chronology, changes only the candidate association failure policy to
the existing transactional rollback mode, and freezes two evidence levels:

1. exact synthetic mechanism cases with known parent identity, exact and independently
   re-componentized fields, baseline angle/noise/aspect sweeps, field-level mask corruption with
   train-only nuisance Gaussians, and a transport negative control that shuffles the actual
   candidate geometry/gate before scoring against unchanged labels;
2. a development-only, three-seed native-control versus all-candidate interaction on every one of
   the eleven `gaussians2d*` field sets sealed under `dataset/` on 2026-08-05. These include Stage
   normalized/additive/full-resolution variants and two unmasked karate frames. Every calibrated
   view is reduced to the preregistered deterministic 512-component proxy before dense fitting.

Primary decisions are made separately for shape, association, masking, topology, and scheduling.
Both calibrated arms run even when an isolated gate fails, so failure remains visible; a combined
pipeline result can never rescue a failed mechanism gate. Each dataset receives a canonical child
`index.html` containing every final metric across seeds, optimizer/stage curves, preview artifacts,
and its two-model orbit-viewer command. The explicit unmasked-support fallback is also shown as a
per-seed curve and aggregate fraction, and fallback cells are excluded from mask-mode
interpretation. The run could not be initialized until a distinct prospective reviewer approved
the exact digest with `Outcome Access: none`. That review approved protocol
`e57d58112fd6f95467e8ddacdb4daad7fc9d83ed48b8b9f336a32b1966a92e87` before initialization;
the successor then completed once and was independently audited before report rendering.

The task digest also contains a SHA-256 over the task driver, experiment-contract code, and every
`src/rtgs/**/*.py` byte; development run locking additionally includes untracked-file content
digests. Timed schedule repeats run in fresh guarded processes with the same five-iteration
full-view cleanup declared by the task. Every hard source mean/covariance, transport
finite/mass/fixed-point/dustbin, candidate-gate, split mass/optical-thickness, held-out isolation,
and final-active-view invariant is measured and must pass before the calibrated warmup starts.
The first protected candidate runs used `failure_policy="raise"`. The association-rollback
successor uses the existing transactional `failure_policy="rollback"` only inside the candidate
association clone so an M-step `RuntimeError` or `ValueError` can be represented safely. A rollback
must carry exact typed failure diagnostics and still fails the unchanged required-transport gate;
it cannot produce a successful cell, impute a metric, substitute the native arm, or support a
mechanism claim.

Calibrated workers publish atomically from a temporary directory and preserve structured JSON
failure receipts. Aggregate models, previews, metrics, histories, per-dataset results, and producer
evidence are all built in an unpublished staging directory and committed only after completeness
checks; the completed run receipt is the final marker. Resource receipts cover compact loading,
fit, serialization, and directory publication, and retain input/output bytes plus per-arm/dataset
repeat minima, medians, and maxima. Aggregate previews/report rendering and browser smoke remain
outside the fitted-worker timing boundary.

### 11.1 Audited outcome

The independent audit accepts only a bounded development result. All 483 synthetic cells and hard
invariants passed their integrity checks. Rank-aware covariance recovery passed its exact-field
decision at `3/3` seeds, and probability support passed its controlled corrupted-mask decision at
`3/3`. Field-mass association, projection-nonlinearity topology selection, and progressive-view
scheduling each failed their isolated rule at `0/3`; these tested variants are retired and cannot
be rescued by the combined calibrated arm. In particular, progressive scheduling reduced
fresh-process refit time by 15.8–17.4% but violated the frozen 1% endpoint guard at every seed.

The calibrated matrix contains 66 explicit terminals. Native succeeds `33/33`; the all-candidate
arm succeeds `26/33`; seven candidate cells fail hard gates and contribute no summary,
quality/runtime point, imputation, or native substitution. Two of the 59 successes use the exact
whole-cell unmasked fallback and establish unmasked operability only. The original exception from
the single rollback-consistent failure and failed/half per-fit AABB arrays were not serialized, so
rollback and AABB conclusions retain the audit's provenance narrowing. Timings and RSS are
host-local diagnostics only.

The root report and all eleven dataset `index.html` pages passed a Chromium smoke test with all
`586/586` local targets reachable. All eleven WebGL2 orbit viewers reached the explicit Gaussian
renderer-ready state, produced non-background framebuffer pixels, and responded to orbit input.
This presentation check establishes report/viewer operability, not visual quality. The canonical
audit is
`benchmarks/results/20260805_probabilistic_field_pipeline_association_rollback_mixed_AUDIT.md`.

## 12. Stop rules

- Drop full-covariance recovery if it does not improve both known-parent 3D covariance error and
  held-out projected covariance error over source-footprint lifting in frozen four-view full-rank
  cases for at least two seeds.
- Drop transport if it does not improve precision times the common minimum accepted-parent
  coverage over both no-association and row-softmax controls, beat the actually shuffled candidate
  negative by `0.10`, or pass every transport/dustbin validity gate.
- Keep the largest-mass topology control if the nonlinearity score does not improve held-out field
  error at matched final count and wall clock.
- Drop probability masks if the field-level corruption test is dominated by hard or unmasked
  controls in precision, coverage, held-out density MSE, and held-out RGB-numerator MSE in the
  frozen number of strata/seeds.
- Drop progressive views unless fresh-process refit time improves by at least `10%` for two seeds
  while both held-out endpoints remain within `1%` after the identical five-step all-view cleanup.
- Never promote independent-half agreement by itself.

## 13. Evidence ladder

| Level | What it may establish |
| --- | --- |
| Deterministic unit tests | Shape, invariant, rollback, isolation, and serialization contracts |
| Exact synthetic experiment | Mechanism feasibility and observability under known generation |
| Re-componentized synthetic fields | Sensitivity to decomposition and correspondence mismatch |
| Calibrated development micro-capture | Pipeline operability and development-only utility |
| Multi-scene reviewed experiment | A bounded quality/convergence statement |
| Controlled GPU benchmark | A hardware-specific performance statement |

This task reached pipeline integration and the first four evidence levels only within the exact
audited boundaries above. It did not reach a multi-scene quality/convergence conclusion or a
controlled GPU performance conclusion.
