# Independent preregistration review: compact residual-responsibility birth allocation

Verdict: PASS

Unresolved findings: none

## Scope and chronology

I independently reviewed the complete frozen preregistration before implementation. Its SHA-256
is `cb384fb560cffae23550b6b4975a3fb439c0a05bb6997a079696830587b11bb9`.

This was an outcome-free document and executability review. I did not implement a seam, create or
run the future harness, invoke any official or focused seed, construct a schedule, sample, score,
selection, split draw, evaluation bank, training state, seal, attempt marker, or result, inspect an
outcome, or edit the preregistration or repository source. This review file is the only file
written by the reviewer.

The current source surfaces inspected for executability were:

- `src/rtgs/render/point_base.py`,
  SHA-256 `252e66eda091a7b9a769155889e11a2ed3f905a5bdf984164e842820c11203f7`;
- `src/rtgs/render/torch_points.py`,
  SHA-256 `c9b6441addbe19cb06f2bf65ec6140be4c61110c4c8f4704c149183d9c8b3696`;
- `src/rtgs/optim/compact_trainer.py`,
  SHA-256 `81a2b538f68623c39e2d17a513b3d43b41a0c7a6ea8a5f72355dd326e288378c`;
  and
- `src/rtgs/optim/density.py`,
  SHA-256 `d56d650eaf0cb758b53111a158f3721b8be69b31292a1785e3f33430e686d375`.

## Review rationale

- The estimand is isolated: all three arms receive the same 32-parent quota, clone/split mix,
  count trajectory, optimizer budget, samples, and recovery horizon. The shuffled-label arm is a
  valid within-stratum allocation null, while the gradient arm is the ordinary screen-gradient
  comparator. The document correctly excludes fixed-versus-birth, pruning, optimal-count,
  source-RGB, novel-view, scaling, and default claims.
- The literal compositor equations match the current point renderer's camera-wide depth order,
  hard-support alpha, exclusive transmittance, and contribution weights. Exposing the exact
  activated sorted color basis behind a default-false request makes the specified single VJP
  executable without materializing an attempts-by-visible diagnostic. The frozen contraction,
  inactive-attempt handling, global-index mapping, alpha identities, empty-visibility branch, and
  off/on parity tests are sufficient fail-closed mechanism checks.
- Score arithmetic, equal-view aggregation, native-float32 gradient comparator, support
  eligibility, stable ties, four matched strata, permutation semantics, and all Phase-A
  distinction gates are literal and independently recomputable. The gates test executability and
  treatment separation without being misrepresented as utility evidence.
- The current six-Adam compact trainer can admit the specified default-off observer/controller
  hook at the frozen post-backward and post-step boundaries. The current density surgery provides
  compatible clone/split field arithmetic and Adam-moment editing. The preregistration additionally
  fixes persistent identities, append order, isolated split draws, optimizer clocks, variable-row
  motion accounting, and exact pre/post-surgery checkpoints needed to remove current
  fixed-cardinality ambiguities.
- Phase B uses replay equality, fresh immutable banks, matched arm streams, fixed checkpoint
  arithmetic, two independently meaningful comparators, uniform-risk safety, and explicit
  population guards. Its decision table covers positive, partial, trade-off, valid-negative, and
  protocol-failure outcomes without post-hoc rescue.
- Official seeds are phase-separated and protected by exclusive attempt markers. Independent
  Phase-A audit authorization, complete source/input/runtime binding, RGB denial, immutable
  artifacts, no retry or resume, and fresh-namespace-on-failure rules make the lifecycle
  auditable and fail closed.

The protocol is therefore sufficiently specified, causal at its stated matched-birth-allocation
scope, and executable against the reviewed repository. This PASS authorizes implementation and
focused nonofficial verification only; it does not authorize Phase A, Phase B, or any scientific
claim without the later implementation review, seal, markers, and audits required by the
preregistration.
