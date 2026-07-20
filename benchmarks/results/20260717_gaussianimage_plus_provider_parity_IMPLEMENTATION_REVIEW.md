# Implementation review: GaussianImage++ provider parity

Date: 2026-07-17T02:51:52+02:00 (Europe/Berlin)

Verdict: `FAIL`

The CPU reference math and isolated-worker design are substantially consistent with the frozen
GaussianImage++ sources, but the official lifecycle is not ready to seal. The official `run`
command reads the frozen checkpoint before creating its one-shot attempt marker, then lacks a
post-marker assertion that the decoded checkpoint still has the sealed hash. The preregistration
also calls the implemented dense tile raster a sparse query. No seal or official run is authorized
from this review.

## Exact reviewed files

- `benchmarks/gaussianimage_plus_provider_parity.py` — SHA-256
  `32a24019b582ceea087d303cfb4ea9bf3a164d10df22d3b382aa8e262c9c64af`
- `benchmarks/gaussianimage_plus_native_worker.py` — SHA-256
  `b59a78a5187acaf4160b26834b616da53b792d6a45f2b222c3b0b409a6709cdb`
- `tests/test_gaussianimage_plus_provider_parity.py` — SHA-256
  `98ab7e3862377fd74b30d70ad354a60fbc5a72ee481c3213db89225783412b4a`
- `benchmarks/results/20260717_gaussianimage_plus_provider_parity_PREREG.md` — SHA-256
  `d3e443ab32cc1fb52281192eefdba659387c050aae66cc7dffe53deb0d8f21a7`

The review file is intentionally not self-hashed. A repaired implementation must receive a new
review whose hashes match the repaired files before a seal can be created.

## Blocking findings

1. **The checkpoint is read before the attempt marker.** `command_run()` calls
   `load_and_verify_seal()` before `_exclusive_json(ATTEMPT, marker)`. Seal verification calls
   `external_bindings()`, whose `sha256_file(REAL_CHECKPOINT)` reads every checkpoint byte. This
   directly contradicts the frozen lifecycle statement that the attempt marker is created before
   “reading the checkpoint.” A failed or changed checkpoint can therefore be inspected, repaired,
   and retried without consuming the one-shot attempt. Move checkpoint-byte verification behind
   exclusive attempt creation; pre-marker checks must be limited to outcome-free seal/source
   verification explicitly allowed by the preregistration. Add a source-order regression that
   proves checkpoint hashing and decoding both occur after marker creation.

2. **The sealed checkpoint identity is not enforced at point of use.** After the marker,
   `load_checkpoint_cpu(REAL_CHECKPOINT, ...)` decodes the file without asserting
   `REAL_CHECKPOINT_SHA256`. Checkpoint-mode `run_worker()` does not pass an expected checkpoint
   hash, and the worker merely reports the hash of whatever file it opened. The later
   `checkpoint_sha256` result field is descriptive and is not part of any pass gate. A replacement
   made after pre-marker seal verification but before both post-marker loads could therefore pass
   CPU/native parity while the result continues to carry the seal's original external bindings.
   Hash-bind the exact bytes immediately after the marker and before decode, pass the expected hash
   into checkpoint-mode workers, verify it before `torch.load`, and require the worker-reported
   input hash to equal the post-marker bound hash.

3. **The advertised sparse-query endpoint is not implemented.** The preregistered first aim says
   “sparse integer-pixel query,” but `render_cpu()` allocates a complete `H x W x 3` image and
   traverses every pixel in each occupied tile. All comparisons are dense full-image raster
   comparisons; there is no coordinate-list query API or sparse-query fixture. Change the frozen
   aim/claim wording to “pure-CPU dense tile reference renderer” unless an actual sparse integer
   coordinate query and corresponding native endpoint are added and tested.

4. **A non-finite parity failure can consume the attempt without a terminal result.** The
   `try/except` covers `execute_official_run()` but not `_exclusive_json(RESULT, result)`. If a
   native mismatch produces a NaN or infinity in a reported error metric, `allow_nan=False` raises
   during serialization after the attempt exists, leaving no failure result. The frozen protocol
   requires finite standard JSON and an exclusive terminal result. Validate/sanitize the complete
   payload inside the caught lifecycle, and ensure any serialization-validation failure is reduced
   to a finite failure receipt written once. Add a CPU test injecting non-finite native comparison
   data and asserting a terminal failure artifact rather than an orphan attempt.

## Semantics that passed source review

- The checkpoint adapter uses `_xyz`, `_cov2d + slv_bound`, raw `_features_dc` for the frozen
  `color_norm=false` arm, `_opacity`, and `background`, matching the reviewed model/checkpoint
  path.
- The CPU projection reproduces float32 direct-covariance inversion, the frozen `max(0.1, ...)`
  eigenvalue discriminant floor, long-axis circular tile bounds, C/CUDA truncation toward zero,
  short-radius rejection, integer radii, and integer pixel coordinates.
- The additive renderer uses
  `0.5*(a*dx^2+c*dy^2)+b*dx*dy`, `min(1, opacity*exp(-sigma))`, discards only
  `sigma < 0` or `alpha < 1/255`, adds colors without transmittance/normalization, and clamps once
  after the sum. The all-culled wrapper background and non-empty black-background behavior match
  the foreign Python/CUDA path.
- The foreign three-channel kernel unconditionally stops after its first 256-candidate batch.
  Rejecting the 257-component sentinel before worker dispatch and requiring at most 256 candidates
  for rendered arms is the correct bounded policy. Candidate-set, radii, hit-count, conic, raw,
  and clamped-image gates are otherwise appropriate for the frozen fields.
- The raw 639-component checkpoint remains explicitly diagnostic. The provider-bearing arm applies
  the exact stored-float32 predicate `xx > 0`, `yy > 0`, and `xx*yy-xy*xy > 0`, preserves order,
  removes rather than repairs the 13 invalid components, serializes the filtered tensors without
  pickle, and asks the isolated worker to render that exact field.
- The worker is render-only, checks the clean foreign commit, Python prefix, Torch/CUDA labels,
  sole preloaded `libstdc++`, exact `csrc.so` location/hash, and explicitly imports the bundled
  binary before invoking the lazy CUDA API. Neither harness nor worker imports an image decoder,
  trainer, or source RGB. Checkpoint colors remain RGB-derived compact data, not source-image
  access or source-quality evidence.
- Overall status includes every rendered synthetic arm, the rejection sentinel, the raw diagnostic,
  and the SPD-filtered provider arm. The claim limits correctly exclude fitting quality, calibrated
  data, memory/speed, StructSplat comparison, and downstream 3D benefit.

## External state and reviewed semantic bindings

- GaussianImage++ clean commit:
  `549cfaab2b400248f685c12782a180f3cfc038b0`
- Bundled `gsplat/gsplat/csrc.so`:
  `9b57b7e0531a50d87c529d3541fbf370f9d85455836ac0cf5414c01ce48ac222`
- Frozen checkpoint:
  `ad611facd72e813dece1b95c3268dbfd82f8af01cdb5ad67e1c7675cc670794b`
- `/usr/lib/x86_64-linux-gnu/libstdc++.so.6`:
  `1fd75fe70354a416d75aef22bcae68c47bd25d20e2d0568c30b1a9838cf62f11`
- `models/gaussianimage_covariance.py`:
  `c3a14856e8939bc5c42a48ddbed3ea7f1d5efb01b837f51886a7ebfce0d4ec18`
- `train.py`:
  `33e6eb9a40ad1c549171feb3b1bfc9211fe1ad38ca83602aed069b943494cbab`
- `gsplat/gsplat/project_gaussians_2d_covariance.py`:
  `bbfcacac8e3cec552e1b930122aa0a1bae3a72d6d3a4c25b7ea0165321f668c5`
- `gsplat/gsplat/rasterize_sum_plus.py`:
  `44a72f97128e22700087ea9002a44330ad1aea303d82729a030b01479a3bca80`
- `gsplat/gsplat/utils.py`:
  `89e49653e5f184e8600ba06fe4001b94daff7947591aabecde7e243ce1cf7472`
- `gsplat/gsplat/cuda/csrc/config.h`:
  `ef4cb25f4b3e008901b06170ce8ab3a2d112e59fa914cfa67db6790adfaeb01d`
- `gsplat/gsplat/cuda/csrc/helpers.cuh`:
  `292f5f60fcdf79a59bf1b92400bd97b87b8b07c826af7c2e719f451b2576a792`
- `gsplat/gsplat/cuda/csrc/foward2d.cu`:
  `fa69684a48c8e7a931977e6aa3b9ec0e05a91f87626e508b54b87e12dd909a04`
- `gsplat/gsplat/cuda/csrc/forward.cu`:
  `2e64ebb100fc71b008546761db01a8e8b2b2baf556838195be5d7d288fc48162`

The foreign checkout was clean when reviewed. Read-only CPU preflight recovered 639 components,
626 SPD-eligible components, 13 removed components, maximum CPU tile population 134, and finite
raw/clamped fields; these are consistency diagnostics, not experiment outcomes or pass criteria.

## Checks executed

```text
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m pytest -q tests/test_gaussianimage_plus_provider_parity.py
=> 19 passed

.venv/bin/python -m ruff check benchmarks/gaussianimage_plus_provider_parity.py \
  benchmarks/gaussianimage_plus_native_worker.py \
  tests/test_gaussianimage_plus_provider_parity.py
=> All checks passed

.venv/bin/python -m ruff format --check benchmarks/gaussianimage_plus_provider_parity.py \
  benchmarks/gaussianimage_plus_native_worker.py \
  tests/test_gaussianimage_plus_provider_parity.py
=> 3 files already formatted
```

No official experiment, seal, attempt marker, result, worker output, source RGB read, or viewer
artifact was created during this review.
