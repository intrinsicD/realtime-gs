# Result: registered RGB-D oriented-point transfer rejected before optimization

The official development artifact is
`20260715T143959Z_cpu_tum_rgbd_oriented_validity_xyz.json` (8,165,745 bytes; SHA-256
`384bc416bcd9c8539aa85a16da533fd2d024124bd549e80f337d6951eb206834`). The sole
confirmatory artifact is `20260715T144052Z_cpu_tum_rgbd_oriented_validity_desk.json`
(6,640,262 bytes; SHA-256
`febc92898cb2c8f60ea4bad4c699017067e79e9bee325b210b0edb636b2b32a7`). The frozen
threshold manifest has SHA-256
`e3ad0adfcd7909bcf64dabaa829dee87dab57b5cb52da7d0ae2f5ff5f9264324`; the durable
confirmatory-attempt seal has SHA-256
`ade8c68318388343d5d306becbcbe2e3b055d3ec536e2e1f692f4804eb863978`.

## Official commands and decision

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:src .venv/bin/python \
  benchmarks/tum_rgbd_oriented_validity.py \
  --phase development \
  --archive /home/alex/.cache/rtgs/tum-rgbd/rgbd_dataset_freiburg1_xyz.tgz \
  --output benchmarks/results/20260715T143959Z_cpu_tum_rgbd_oriented_validity_xyz.json \
  --thresholds-output benchmarks/results/20260715_tum_rgbd_oriented_validity_THRESHOLDS.json \
  --threads 4

CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:src .venv/bin/python \
  benchmarks/tum_rgbd_oriented_validity.py \
  --phase confirmatory \
  --archive /home/alex/.cache/rtgs/tum-rgbd/rgbd_dataset_freiburg1_desk.tgz \
  --output benchmarks/results/20260715T144052Z_cpu_tum_rgbd_oriented_validity_desk.json \
  --thresholds benchmarks/results/20260715_tum_rgbd_oriented_validity_THRESHOLDS.json \
  --threads 4
```

The registered-depth backend produced broad eligible/support populations on both official TUM
sequences, but cross-view geometric consistency did not transfer to `fr1/desk`. Three of the nine
frozen gates failed. The preregistered decision is therefore **stop**: do not run the plane/normal
optimization phase and do not tune this backend, target grid, visibility rule, or thresholds on
desk.

| metric | `fr1/xyz` development | `fr1/desk` confirmatory | frozen desk threshold | gate |
| --- | ---: | ---: | ---: | :---: |
| eligible fraction `A` | 0.70036 | 0.68441 | >= 0.49026 | pass |
| minimum T-view eligibility `A_min` | 0.64417 | 0.59750 | >= 0.32208 | pass |
| two-V oriented support `S` | 0.86686 | 0.80359 | >= 0.52012 | pass |
| p10 T-view support `S_10` | 0.73198 | 0.63593 | >= 0.36599 | pass |
| symmetric point-to-plane p90 `R90` | 0.02497 m | **0.20211 m** | <= 0.04245 m | **fail** |
| relative depth p90 `D90` | 0.03104 | **0.25186** | <= 0.05000 | **fail** |
| median normal cosine `C50` | 0.91516 | 0.88720 | >= 0.76516 | pass |
| p10 normal cosine `C10` | 0.67197 | **0.50262** | >= 0.52197 | **fail** |
| free-space contradiction fraction `F` | 0.04503 | 0.06247 | <= 0.07755 | pass |

## Population and tail diagnosis

Development constructed 40,341 eligible targets and confirmatory constructed 39,422, from the
same frozen 57,600 grid locations. On desk, 33,048 targets had at least two depth-valid validation
views and 31,679 had at least two oriented-valid views. All eight validation views contributed
thousands of globally supported pairs. The failure is therefore not explained by an empty or
single-view audit population.

The desk distributions are strongly heavy-tailed:

| target-level distribution | p50 | p75 | p90 | p95 |
| --- | ---: | ---: | ---: | ---: |
| relative depth residual | 0.02139 | 0.04402 | 0.25186 | 0.71073 |
| symmetric surface residual | 0.01542 m | 0.03189 m | 0.20211 m | 0.53669 m |

Of supported desk targets, 7,392/33,048 (22.37%) exceeded 5% median relative-depth error,
6,279/31,679 (19.82%) exceeded the transferred 42.45 mm surface limit, and 3,518/31,679
(11.11%) fell below the transferred low-normal-cosine limit. Validation ordinals 3 and 27 had the
largest pair-level depth/surface tails, but several other validation views also had large p90s.
This is descriptive evidence of a distributed tail, not proof of its cause. The frozen artifact
does not contain semantic motion labels or signed behind-surface residuals, so it cannot separate
occlusion, scene motion, approximate calibration, and sparse construction-only visibility after
the fact.

## Isolation and provenance

- The official TUM archive SHA-256 values are
  `a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98`
  (`fr1/xyz`) and
  `e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c`
  (`fr1/desk`). The acquisition record transparently identifies local filesystem timestamps as
  proxies because no contemporaneous network-request timestamp log survived.
- Each phase decoded exactly the 56 selected `T`/`V` depth members once. It decoded zero RGB and
  zero sealed `H` payloads. Construction received a T-only backend; audit received a disjoint
  V-only backend.
- Target identity binds float64 points/normals, the complete 48x1,200 eligibility mask, source
  pixels/ordinals, calibrated construction poses, and configuration. The development/desk target
  hashes are `86fbe1be8dc79ff3323ec009432b7858f011cab07e25947c7f0d4a223f1d7868`
  and `5af71d94aa5e1647f5580b8a0c732c933f8dfd830b29847b4a5376f72d34d69f`.
- Both phases used implementation aggregate SHA-256
  `119d3c07e0910c538b8e19a7d7c8ab299a992b54a0a1b71c434b6d8a81136c67` at revision
  `2dddca4aff59702341af9faceefa76ad2505dd83` in a dirty worktree. The confirmatory threshold
  validator re-derived every threshold from the exact frozen development bytes before the atomic
  one-shot seal was consumed.

## Conclusion and next admissible evidence

Reject this exact registered-depth target/visibility protocol as a transferable prerequisite for
plane pulling and shortest-axis normal alignment. It validates that the pluggable backend can
produce plentiful metric oriented points, but it does not validate their required cross-view tail
consistency on a second real sequence. The generic backend/canonicalization API and zero-default
surface losses remain useful, CPU-tested research infrastructure; production behavior is
unchanged.

Do not run the withheld Phase B, relax p90 gates, remove difficult validation views, add V-depth
visibility filtering, or reuse desk to select a denser grid. If this branch is revisited, the next
admissible experiment is a separately preregistered occlusion/rigidity attribution audit on new
development and confirmatory sequences. It should record signed depth discrepancies, compare
construction-only visibility mechanisms without filtering on confirmatory residuals, and retain an
ordinary extra-depth anchor as the mandatory utility control. Until that evidence exists, make no
utility claim for the oriented losses.
