# Independent audit: sparse point-rasterizer Phase A

## Verdict: `PASS`

The preregistered CPU mechanism gates pass. This authorizes the bounded calibrated
interaction check. It does not establish compact refinement, reconstruction quality,
speed, memory reduction, CUDA/gsplat parity, density control, or a production default.

### Exact artifact bindings

- Preregistration SHA-256: `afc9d036ad1c037a5cb3eab7fd5b19f97d37d920f520cb5c51bf37f41f989916`
- Implementation review SHA-256: `1b23fff1fdf654bec34730da036085eca4ef7be9e1a1179b12ab2225e8d94111`
- Seal file SHA-256: `51d9a5c75397568f311325064943671a3c2fa5a00c2743d9e1ae6d3e00b1801d`
- Seal payload SHA-256: `0e7dd5a891fa7e25beb8083e12b3468b1b23658a17d680bd799042453636bccd`
- Attempt SHA-256: `5ba74a7b4c3c35ef39215e1f4c221d979a27d2e10c1acac31e25d024c13c922a`
- Result SHA-256: `1abbdec0fd0fb71a3aa746430ca7f84b08999476951eb5386852d804cbfd4d85`
- Harness SHA-256: `89b0cda4de01ef3c2b5898c22096e86c725b97af2c7271f7d56c4e4ad0cd645a`
- Sealed source aggregate: `30876dfef84a1170e17b349ee27be44524c187547a4ee4f5a4a7de6c304bb449`

All 89 files in the seal manifest currently match their recorded hashes. The seal
self-digest, canonical source aggregate, embedded attempt, attempt digest,
preregistration binding, seal binding, result binding, git revision
`2dddca4aff59702341af9faceefa76ad2505dd83`, and tracked-diff SHA-256
`80073943e80d6aad7962df7981ade92555a276415389c2b9a4e6aa72c698027f`
independently recomputed correctly.

## Claim table

| # | Claim | Kind and scope | Evidence | Independent disposition |
|---|---|---|---|---|
| 1 | Sparse CPU point rendering matches dense CPU rendering at all frozen pixel centers. | Measured; synthetic CPU; three official seeds | RESULT `forward` | **Confirm.** All 108 required arms are present. Maximum absolute errors were color `5.9604645e-08`, alpha `1.1920929e-07`, and depth `2.3841858e-07`, all below `2e-6`. Visible order matched in every arm. |
| 2 | Sparse parameter and retained `means2d` gradients match the dense anchor. | Measured; synthetic CPU; three seeds x nine chunk pairs | RESULT `gradients` | **Confirm.** All 27 arms are present. Maximum absolute errors were means `1.8626451e-09`, quaternions `9.3132257e-10`, log-scales `3.4924597e-10`, opacity `9.3132257e-10`, SH `4.6566129e-10`, and `means2d` `2.3283064e-10`, all below `4e-6`. Every dense gradient family was nonvacuous. Larger relative errors on near-zero quaternion entries do not violate the preregistered combined `allclose` gate because their absolute errors are far below tolerance. |
| 3 | The sparse path retains the global depth compositor and does not filter by proposer/lineage. | Measured invariant plus source inspection | RESULT `global_compositor`; `torch_points.py` | **Confirm.** The non-proposer near-Gaussian intervention changed color by `0.3537486792` versus the `1e-4` floor, remained dense-equivalent, distinct-depth input reversal was exact, and the API exposes no forbidden proposal/lineage argument. |
| 4 | Empty visibility/query contracts are correct. | Measured API contract; synthetic CPU | RESULT `empty_contracts` | **Confirm.** Background and zero alpha/depth were exact; empty-query shapes were `(0,3)`, `(0,)`, `(0,)`, while visibility metadata remained present. |
| 5 | Frozen arbitrary continuous coordinates produce finite outputs and coordinate gradients. | Measured finiteness only; synthetic CPU | RESULT `continuous_coordinates` | **Narrow to the literal gate.** Outputs and gradient tensors were finite, but every seed reported maximum `xy` gradient exactly `0.0`. This does not fail the preregistration, which required finiteness only, but it supplies no evidence of an active/nonzero off-grid coordinate derivative. |
| 6 | The discrete Gaussian/rejection proposal gives the exact uniform finite-pixel expectation and fixed-attempt microchunk normalization. | Proven analytic identity plus measured Monte Carlo control; float64 tiny fixture | RESULT `discrete_risk` | **Confirm.** Recomputed exact risk and enumerated expectation both equal `55/96 = 0.5729166667`. Analytic variance is `0.7259830384`; null probability is `0.3714907127`. Maximum per-seed error was `0.0864502052 < 0.2259329157`; pooled error was `0.0075645968 < 0.0141208072`; maximum microchunk discrepancy was `2.2204460e-16 < 2e-12`. All 64 seeds and all proposal branches were present. |
| 7 | Pair temporaries are bounded by both chunk controls and proposal state is component-scaled. | Source/test invariant, not empirical memory measurement | Sealed source; focused tests | **Confirm as an implementation invariant only.** This is not a measured RAM, speed, asymptotic end-to-end, or GPU claim. |
| 8 | The method works on the calibrated 835-Gaussian dataset/viewer path. | Not yet executed | Future `calibrated_parity.json` and viewer artifacts | **Unverified.** Phase A authorizes this next check but cannot substitute for it. |

## Recomputed controls

- Fixture minimum hard-boundary distances were `0.0199385`, `0.00181580`, and
  `0.00868988`, all above `1e-5`.
- The 108 forward cases exactly cover three seeds, two backgrounds, two SH degrees,
  three point chunks, and three Gaussian chunks.
- The 27 gradient cases exactly cover three seeds and all nine chunk pairs.
- Four supplemental nondefault activation/kernel cases are present and below the
  forward tolerance.
- Discrete branch totals reconcile exactly: `6462` uniform plus `26306` Gaussian
  equals `32768` attempts; `14156` accepted plus `12150` rejected equals all Gaussian
  attempts.
- Proposal probabilities are finite, positive, and nonuniform; importance values equal
  `(1/6)/q_p` and remain below the frozen bound.
- The five expected Phase-A namespace artifacts existed before this audit; no alternate
  attempt or result sibling exists.

## Chronology and lifecycle

Filesystem chronology agrees with the protocol: the final preregistration amendment
preceded point-renderer, harness, and test creation; implementation review preceded
sealing; seal time was `2026-07-16T18:07:28Z`; the exclusive attempt marker was created
at `18:07:57Z`; the result file followed approximately 14.4 seconds later.

The RESULT's `timestamp_utc` equals the attempt second because it is captured at Phase-A
execution start, not at file completion. This is a metadata limitation, not a lifecycle
violation; wall time and exclusive file creation preserve the ordering.

The official one-shot command was not rerun and no official fixture constructor was
called during this audit.

## Commands executed

- Pure read-only Python recomputation over the existing SEAL, ATTEMPT, and RESULT JSON
  files.
- SHA-256 and canonical seal/source-manifest verification.
- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_point_render.py tests/test_observation2d.py tests/test_point_rasterizer_parity.py`
  - `61 passed`.
- `git diff --check` - passed.
- A second full `./scripts/verify.sh` was started but deliberately stopped during pytest
  and is not counted as audit evidence. The sealed pre-run full verification completed
  successfully, and its entire sealed source manifest remains unchanged.

## Remaining evidence required

The next authorized step is the preregistered calibrated CPU parity command, followed by
the viewer smoke test. Even if both pass, they establish only real-input renderer
integration. Compact fixed-topology refinement, sampled convergence, quality, memory,
throughput, CUDA/gsplat parity, and density-control behavior each require separate
experiments and evidence.

---

# Calibrated interaction audit addendum

## Verdict: `PASS`

The preregistered calibrated CPU interaction passes for its narrow scope: dense and
sparse renderers agreed on the frozen sampled pixels for the existing 835-Gaussian PLY
and calibrated C0001 camera.

### Exact bindings

- Calibrated artifact SHA-256: `d8779c15224881f3f61a2bdb11cffd1ab28d009e594bc6ac3e036e7d73c7bdf4`
- Phase-A audit SHA-256 at calibrated execution: `1dbfad77dff1bad4fbd7d4a8624243be7fecc4676287960fbfd27a5c5babbf67`
- Preregistration SHA-256: `afc9d036ad1c037a5cb3eab7fd5b19f97d37d920f520cb5c51bf37f41f989916`
- Seal SHA-256: `51d9a5c75397568f311325064943671a3c2fa5a00c2743d9e1ae6d3e00b1801d`
- Phase-A result SHA-256: `1abbdec0fd0fb71a3aa746430ca7f84b08999476951eb5386852d804cbfd4d85`
- PLY SHA-256: `0bed5a18609d560371f621634aaae915ea3e6ac0f834584f729c616c9821059d`
- Calibration SHA-256: `51b8fc396fc8447f24e325e0a525f2e7d422388790dd9a293e1a81804b265091`

All bindings and all 89 sealed source hashes independently match.

### Recomputed result

| Metric | Maximum absolute error | Maximum relative error | Gate |
|---|---:|---:|---|
| Color | `8.9406967e-08` | `4.0525060e-07` | PASS |
| Alpha | `1.7881393e-07` | `1.9879731e-07` | PASS |
| Depth | `4.7683716e-07` | `2.3579189e-07` | PASS |

Every absolute error is below the frozen `2e-6` tolerance.

The calibrated input checks also reconcile:

- PLY header: exactly `835` vertices.
- Reported Gaussian/visible counts: `835 / 835`.
- C0001 source resolution: `5328 x 4608`.
- Downscale 16 camera: `333 x 288`, `fx=286.3144477394796`,
  `fy=286.28346545453036`, `cx=166.847932318391`,
  `cy=140.70854297941912`.
- Independently reconstructed camera rotation and translation hashes match the artifact.
- Image domain: `95,904` pixels.
- Sampling: `4,096` uniform draws with replacement, seed `93001`.
- Independently reproduced sample hash:
  `26fe9a9b490ecac6c12e92a5cfecc4c4dc33c726fb243fd86fc00ccfd2e78330`.
- The sample contains `3,998` unique pixels and `98` repeated draws.
- Recorded wall time: `1.3903779359825421` seconds.

The sealed calibrated route reads the calibration JSON and PLY directly. It contains no
PIL, `load_calibrated_scene`, `SceneData`, RGB, or mask-loading call; focused tests fail
if those loaders are touched. Thus the no-source-RGB/no-mask interaction claim is
supported.

### Limitations

- Parity is established only for the frozen 4,096 replacement draws, not every
  calibrated pixel.
- Exact visible-order equality is enforced by the sealed fail-closed harness, but the
  JSON stores only `visible_count=835`, not the ordering or its hash.
- Runtime combines dense rendering, point rendering, parity checks, and provenance
  verification. It is descriptive and supports no speed claim.
- No RGB quality metric, reconstruction improvement, optimization, memory reduction,
  CUDA/gsplat parity, or density-control result was measured.
- The PLY contains its existing Gaussian appearance parameters; "no RGB decoded" refers
  to source images during this interaction.
- Viewer startup/snapshot evidence remains separate and was not audited here.

The calibrated command and render were not rerun during this audit.

---

## Viewer evidence addendum

### Verdict: `PASS`

The existing viewer evidence supports the preregistered integration-smoke claim only.

- Live `127.0.0.1:8767` returned HTTP `200` with the Viser client.
- The running process argv and `viewer.log` exactly match the frozen command.
- Viewer log SHA-256: `27ad42821cbbef30de5cfce8d7c53d22f89104f0f99e0dce45c9a2cc63443d8b`.
- Snapshot: `viewer_snapshots/final_camera_0000.png`.
- Snapshot SHA-256: `5392c6d4c03a6965dd043291de7fa2e89e53823a618e2a81d2dd0b32aa8df209`.
- PNG verification passed: RGB, `333 x 288`, 8-bit/channel, non-interlaced, `12,357`
  bytes.
- The logged status--`final`, camera index `0`, `835` splats, Torch, CPU, and filename
  `final_camera_0000.png`--is internally consistent with the sealed CLI/viewer source: the
  CLI names the supplied PLY `final`; the count slider defaults to all 835 splats; camera index
  0 is initially selected; the exact snapshot uses the requested Torch renderer on CPU; and the
  saved filename follows `{name}_camera_{index:04d}.png`.

The viewer action was not rerun.

Limitations:

- This proves viewer startup, UI-driven snapshot integration, and a decodable output file--not
  visual quality.
- Camera index `0` is scene camera `C0000`, whereas calibrated parity used `C0001`; the viewer
  receipt is not another C0001 parity measurement.
- The viewer intentionally loads and displays scene RGB references. No RGB-free viewer claim is
  allowed.
- HTTP `200` verifies the live web client endpoint, not every websocket/UI control.
- The log is an external receipt rather than an automatically sealed action trace; its status is
  corroborated by the live process, source behavior, and matching PNG hash, but the original click
  was not independently replayed.
