# Roadmap

## M0 — Infrastructure (done)
- [x] Agent workflow: CLAUDE.md/AGENTS.md, skills (verify, bench, docs-sync, experiment)
- [x] Verification: ruff + pytest (CPU) + docs_sync, mirrored in CI
- [x] Benchmark harness with tracked JSON results and auto-updated docs table

## M1 — Pipeline v1 on CPU (done)
- [x] Core containers (2D/3D gaussians, cameras, SH, metrics) with PLY/NPZ IO
- [x] Differentiable 2D accumulated splatting + per-image fitting (gradient-magnitude init)
- [x] Reference 3D rasterizer (EWA projection, depth-sorted alpha compositing)
- [x] Lifting variants: `gradient`, `depth`, `carve` (+ `sfm`/`random` baselines)
- [x] 3DGS refinement loop with adaptive density control
- [x] Synthetic ground-truthed scenes; COLMAP text/binary parsing
- [x] Calibrated object-capture JSON, distortion/masks, held-out split, external 2D adapters
- [x] End-to-end tests and variant-comparison benchmark

## M2 — GPU validation
- [x] gsplat backend parity test green on RTX 4090; auto backend respects explicit CPU devices
- [x] Interactive Viser viewer with initialization/final controls, calibrated cameras, and exact
      gsplat snapshots
- [x] Depth Anything V2 Small smoke test and bounds alignment on a calibrated Janelle capture
- [x] Optional StructSplat CUDA stage-1 backend with configurable progressive density growth
- [x] Wire gsplat Default and MCMC/relocation strategies as alternatives to
      `rtgs.optim.density`, including AbsGS/revised-opacity controls and hard budgets
- [ ] Benchmark on MipNeRF-360 `garden`/`bicycle` @ 7k iters: init-PSNR and
      time-to-quality vs SfM init (protocol in docs/RESEARCH.md §7)
- [ ] Fit-time target: stage 1+2 < 30 s for 200 images @ 1080p on one consumer GPU

## M3 — Research questions (log answers in docs/EXPERIMENTS.md)
- [x] First calibrated-capture comparison: compact `carve` wins Janelle frame 00008 at 1/16;
      repeat at higher resolution and on more scenes before treating this as general
- [x] Synthetic depth-covariance ablation: per-Gaussian footprint variance did not consistently
      beat a train-tuned isotropic σ; validity-aware gradients fixed the raw surface failure
- [ ] Repeat the covariance ablation on calibrated held-out views with real monocular depth
- [x] Audit the hard post-SH nonnegative color floor before testing SMU-family replacements: all
      three view-dependent CPU synthetic seeds and the pool failed the frozen incidence and
      recoverable-gradient gates, so Phase B is permanently closed without training either
      candidate and the hard activation remains default
- [x] Separately audit the hard `q<12` raster-support cutoff, then test the frozen `C=12`, `W=4`
      C1 taper and its hard-forward attribution control: the adjacent annulus passed every local
      mechanism gate, but Phase B lost diffuse foreground PSNR in all seeds (means -0.014483 dB
      and -0.018470 dB), so the taper branch is rejected without tuning and the hard kernel remains
      default
- [x] Audit the detached image-intersection visibility cull against the support-safe `sqrt(12)`
      envelope: all validity checks passed, but the diffuse pool missed only 4/2,480,463 support
      pairs (mass fraction 1.646359e-8) across two exposures, so every material gate failed,
      Phase B is forbidden, and the 3-sigma default remains unchanged without margin tuning
- [x] Ablate leave-one-source-view-out supervision in `gradient`/`hybrid` with a globally balanced
      count/opacity-matched non-self control: both families failed material geometry gates, so
      inclusive supervision remains default and LOSO schedule sweeps are closed on this setup
- [x] Compare confidence-weighted and robust normalized bounded-ray anchors
      (DP-GS/NoDrift3R-inspired): the preregistered synthetic test failed its PSNR/depth criteria,
      so `legacy` remains default and the new modes stay opt-in
- [x] Repair the confidence attribution control with valid-prior-uniform and exact sampled-weight
      shuffle arms: confidence improved held-out depth RMSE by only 1.15% and worsened corrupted
      p90 by 0.77%, so the preregistered stopping rule closes further anchor-weight sweeps
- [ ] Derive depth confidence from training-view geometric/photometric consistency and test it with
      actual monocular depth on calibrated held-out views; defer until cross-view supervision shows
      a material geometry signal, and never tune it on the test cameras
- [x] Test one direct robust world-frame position-consistency term between fixed train-view matches
      while depths remain ray-bounded: represented primitives localized strongly, but the sparse
      oracle graph missed global materiality gates, so loss hyperparameter sweeps are closed
- [x] Test one denser train-only correspondence graph with the frozen position loss and pluggable
      matching: raw patch/epipolar coverage reached 17.99%-19.10%, but strict semantic precision
      was only 9.04%-11.76%, so the preregistered gate stopped before optimization and closes this
      matcher/position branch without threshold tuning
- [x] Complete the three-iteration exact inverse-projection-fiber correspondence study: hardmin
      missed modes, post-hoc contraction could not restore them, and the final capacity-aware UOT
      attempt was consumed after one complete root. That root rejected both transport arms and
      real-data release; exact fibers remain research-only and should anchor stable/moment-merged
      tracks rather than every independently fitted fragment
- [ ] If exact-fiber correspondence is revisited as a newly authorized question, first require an
      oracle-topology ceiling with dynamic source-side moment aggregation, transactional
      projection-valid M-steps, per-arm failure receipts, and a calibrated outlier model on fresh
      development roots; do not resume this loop or unlock the withheld real bundle
- [x] Test local plane pulling plus shortest-axis normal alignment against detached train-depth
      oriented points, initially scoped to Hybrid: the four-neighbor corrupted-depth constructor
      passed structural floors but failed the frozen clean-plane audit in all seeds, so no loss arm
      ran and this constructor is closed without threshold tuning
- [x] Add an independently justified pluggable oriented-point backend for actual calibrated metric
      RGB-D normals and audit it before optimization: the API and real TUM harness are complete,
      but `fr1/desk` failed transferred surface/depth p90 and low-tail normal gates, so Phase B is
      withheld and no production backend/default is enabled
- [ ] If oriented supervision is revisited, preregister an occlusion/rigidity attribution audit on
      new development/confirmatory sequences with signed discrepancies and construction-only
      visibility controls; do not tune on the consumed desk case or skip the ordinary-depth control
- [x] Audit Carve at the exact post-merge count against within-voxel representative and global
      prune controls: production-scale grouping merged only 2.34%-2.68% of primitives and failed
      every seed's preregistered materiality floors, so refinement was withheld and merge utility
      remains untested; do not tune the consumed grid scale merely to force collisions
- [x] Audit the Stage-1 `weight*color` representation contract under product-preserving gauges:
      source RGB remained equivalent while both unmerged Depth and Carve changed materially in
      3/3 seeds and the pool; this validates the interface problem but authorizes no replacement
- [x] Test a gauge-invariant Stage-1-to-lifter boundary with a preregistered factorial: the
      mechanism and evidence gates passed, and observed source color was a positive factorial
      attribution signal in both backends, but in the three-seed deterministic CPU-synthetic scope
      the full invariant-scalar/observed-color repair gained +3.127 dB for Depth and lost -2.205 dB
      for Carve; retain the current boundary, do not select the color-only arm post hoc, and make
      no default change
- [x] Test one exact fixed-topology 24-to-48 multiscale schedule with blocked and interleaved
      controls: all candidate AUC/final-PSNR deltas were negative, so close this schedule without
      tuning and do not combine it with density or gauge interventions
- [x] Attempt a preregistered quaternion radial-gauge optimizer audit: both Phase-A attempts
      failed closed before any optimizer/materiality result, with Retry-2 proving the inherited
      `2e-12` covariance contract incompatible with native float32 canonicalization; keep the
      current policy and require a precision-feasibility gate before any fresh retry
- [x] Complete and run the frozen Stage-1 fit-time parameterization comparison between the
      current nine-parameter `weight*color` gauge and a bounded unit-weight eight-parameter arm.
      The independently audited once-only CPU-synthetic result was valid, but the candidate lost
      in all appearance-only and joint seeds: mean final-PSNR differences were -1.796 dB and
      -1.502 dB respectively. Material local null-direction motion was present, but the frozen
      curve/interference and joint non-inferiority gates failed. Retain the current default and
      close this exact bounded candidate without tuning or downstream/default claims
- [x] Add a library callback that freezes live normalized StructSplat fields without clamping;
      verify the captured equation against the CPU reference renderer on complete fixture pixel
      grids; add correctness-first dense/tile CPU point queries and a continuous-area proposal;
      and serialize slotted camera/teacher bundles with no declared RGB, mask, or source-path
      fields. The strict loader enforces exact keys, restricted identifiers, archive/member byte
      ceilings, ordinary contained files, and symlink rejection; generic callers must still release
      any separately retained `SceneData`
- [x] Add a standalone CPU compact-Carve mechanism that consumes the serialized bundle, samples
      an independently configured $N_{\mathrm{init}}^{3D}$ output budget from a fixed candidate
      pool, and scores coverage-weighted all-view compact fields without source-image or
      dense-voxel materialization. Lineage remains hard for ray proposal and initial covariance,
      not supervision/rendering; this path is unbenchmarked and not in the production CLI/pipeline
- [x] Implement the separate image-free field-lift research path: `SceneFits` preserves compact
      teachers and optional `PackedAlpha` under an explicit complete train/held-out split;
      additive density/RGB-numerator product-kernel refit, observability, visibility/gain,
      transactional topology, frozen-teacher semantic validation, `field` registry,
      `run_field_pipeline`, and `rtgs lift-field` are CPU-tested. This closes implementation only;
      calibrated quality, topology utility, performance, and default selection remain open
- [ ] Establish and validate train-only provenance for compact sparse points and object bounds;
      current `ReconstructionInputs.from_scene` may retain geometry derived before a held-out split.
      `FieldLifter` is safe meanwhile: it ignores unverified points/bounds whenever held-out views
      exist and requires an explicit `geometry_is_train_only` attestation to consume them
- [x] Add and independently audit a CPU point-rasterizer contract with camera-global compositing,
      frozen-synthetic pixel-center forward/gradient parity, bounded point/Gaussian pair chunks,
      a separately proven discrete-pixel proposal, a no-RGB 835-Gaussian sampled calibrated
      interaction, and a separate RGB-loading viewer smoke
- [x] Add sampled field-supervised fixed-topology 3D optimization and compare uniform,
      continuous-area, and discrete-pixel proposals under matched attempts. The independently
      audited result was `NO_GLOBAL_SAMPLING_WIN`: discrete-pixel lost all three seeds and
      continuous-area was non-inferior but missed materiality. Keep uniform as the conservative
      baseline; make no compact-proposal default change
- [x] Exercise seven full-resolution compact teachers through strict serialization, compact-Carve,
      and RGB-denied 835-Gaussian fixed-topology refinement. Preserve the calibrated lifecycle as a
      terminal failure at its first exact snapshot; its optimization and held-out numbers are only
      phase-local diagnostics
- [x] Move exact native gsplat snapshots into a fresh preload-inheriting spawn worker with strict
      artifact/result routing. This routing change alone is not complete ABI/CUDA validation: only
      a new preregistered namespace may validate the repaired complete lifecycle
- [x] Bind the resolved preload-library path/hash, required CXXABI symbol/version, and actual loaded
      default-namespace mapping; also require the launched viewer PID to own the exact listening
      socket before HTTP success. Focused tests and separately labelled diagnostics pass
- [x] Exercise the repaired worker with real spawned gsplat/CUDA renders in the fresh iter3
      lifecycle. All 28 native-resolution arm/view renders and the exact-socket live viewer passed;
      this validates the repaired visualization lifecycle only, not training-backend parity or
      source-RGB quality
- [x] Preflight original compact inputs before whole-input/working-set transfer and construct a
      teacher/camera-only device-tensor working set. Omit optional global geometry and count non-CPU
      tile overlaps in bounded device-native chunks
- [ ] Make compact proposal selection explicit before CLI/pipeline integration; the experimental
      config's pre-result `pixel_gaussian` convenience default is not validated by the sampling
      result. This fixed-topology `CompactTrainer` question is separate from `FieldLifter`
- [ ] Add density control toward explicit variable $N_{\mathrm{opt}}^{3D}$. The audited
      fixed-topology proposal-target result authorizes a fresh matched-count one-wave allocation
      experiment; do not wire a path into the CLI/pipeline before independent mechanism and utility
      audits. This remains a `CompactTrainer` question and is not closed by field topology moves
- [ ] Implement the preregistered compact residual-responsibility birth-allocation Phase A. Run its
      matched 835-to-867 Phase B only if literal point-compositor responsibility is attributable,
      sufficiently distinct from gradient and shuffled-parent controls, and independently cleared.
      The older CPU synthetic RGB preregistration is not the execution protocol for this follow-up
- [ ] Add aggregate device-byte/index budgets, replace eager Python overlap lists with CSR/lazy
      storage, add indexed CUDA compact-teacher queries, and bound backward activation memory before
      making a production-scale claim. (Flattened CPU CSR observation queries landed 2026-07-20 with
      exact parity and a tracked micro-benchmark; the device-byte/index budgets and
      backward-memory bound remain. 2026-07-21: an indexed CUDA query backend landed —
      `rtgs.core.observation2d_cuda.GaussianObservationIndexCuda` wraps the CPU-built CSR index,
      is inference-only and atomics-free, plugs into the existing `backends` seam via
      `rtgs.lift.compact_carve.build_query_backends(..., device="cuda")`, and has self-skipping
      GPU parity tests plus a conditional arm in the tracked micro-benchmark; it is not yet
      verified or timed on GPU hardware, so the CPU index remains the default and the oracle.
      2026-07-22: the two remaining mechanisms landed — opt-in aggregate
      `max_index_entries_total`/`max_index_bytes_total` budgets on `CompactCarveConfig`
      (preflighted before CSR allocation, re-checked on built or supplied backends, defaults
      unbounded) and opt-in `checkpoint_pair_chunks` on the CSR `query`/`query_weight_sum`
      (bounds backward activation memory by recomputing one pair chunk at a time; identical
      values/gradients, saved activation bytes measured at 0.48x baseline on the CPU toy case).
      What remains for this bullet is the production-scale confirmatory run itself, not
      mechanism work.)
- [x] Execute and independently audit the calibrated dense confidence-gated initialization chain.
      E1 found a +1.9714 dB all-view init-only gain for dense+merge but failed the 2× count gate at
      13.48×. I1 reproduced the frozen 60/2,319 easy-only classifier. E2 then rejected easy-only
      under its exact 300-step matched-cap schedule: C1004 foreground PSNR was 12.7332 dB versus
      dense-all's 14.9079 dB, outside the 0.0071 dB repeat envelope, although easy-only remained
      smaller and was still growing. Keep balanced top-K as the default. I2/E3 remains closed
      because the aggregate deficit was not localized to hard-dropped regions; any longer
      budget-filling or localization test needs a fresh preregistration. Full chain and records:
      `docs/TASK_DENSE_CONFIDENCE_GATED_INIT.md`
- [ ] Evaluate the two new correspondence-family initializers — structure-from-splats
      (`rtgs.lift.splat_sfm`, discrete epipolar matching + exact covariance triangulation) and
      tomographic beam fusion (`rtgs.lift.beam_fusion`, covariance-intersection Gaussian products
      with exact centers and conservative covariances) — the calibrated RGB-free SfM analog
      with exact mechanism recovery on the EWA fixture (centers ~1e-7, covariances ~6e-7 relative,
      identical-color twins disambiguated by geometry). Run
      `benchmarks/splat_sfm_screen.py --bundle` on the calibrated seven-view bundle to measure real
      track yield/reprojection/unmatched fractions, then include it as an arm in the next
      preregistered matched-budget downstream experiment (count-matched, budget-filling growth,
      longer horizon; E2's short matched-cap schedule is not reusable). Segmentation-mismatched
      match rates and downstream utility are unmeasured; no default change
- [x] Implement depth-seeded bounded-ray hybrid B→A; evaluate uncertainty and shorter schedules
- [x] Initial density ablation: a short 15k-capped schedule beats no-density and unrestricted
      growth on Janelle; repeat across scenes and compare gsplat MCMC/teleportation
- [x] Progressive/error-driven stage-1 allocation via StructSplat residual/tensor growth;
      compare `quadtree_wse` and GaussianImage at matched wall-clock/count
- [ ] Add LPIPS-VGG to held-out evaluation
- [x] Add alpha-IoU/leakage metrics and both interpolated-orbit and elevation-varying
      novel-view geometry diagnostics
- [x] Run full 26-view Janelle frame at 1/8 and 1/4 resolution with quality/VRAM curves
- [ ] Repeat clean time-to-quality runs on an otherwise idle GPU and add a train-only validation
      checkpoint policy (never select checkpoints on the held-out test views)
- [ ] Feed-forward multi-view init (VGGT/MASt3R pointmaps) as a fourth variant

## M4 — Real-time ambitions
- [ ] CUDA kernel for stage-1 fitting (batched per-image, GaussianImage reports ~2k it/s)
  - 2026-07-21: opt-in seams landed — fused `batch_views` multi-view fitting
    (`rtgs.image2gs.batched`, CPU-verified against serial) and a JIT-compiled CUDA extension
    for the native compositor (`rtgs.image2gs.cuda_backend`, modeled on StructSplat's exact
    renderer) with self-skipping GPU parity tests. The kernel is not yet verified on GPU
    hardware; `torch` stays the default until the parity tests and a tracked GPU benchmark
    pass on a GPU box.
- [ ] Streaming/incremental mode: add images to an existing scene without full re-fit
- [ ] Investigate skipping stage 3 entirely for preview-quality output
