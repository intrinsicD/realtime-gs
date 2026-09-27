# RTGS-026 first protected run: failed closed at the frozen decoder-parity gate

Run `runs/20260927_silhouette_relocation_stage_frame00008/` (local, git-ignored), locked at commit `f88f3d0`, one `run` invocation.
Preflight passed; `prepare` aborted at training view C0026 (training index 13) because the frozen
decoder-parity rule requires CUDA vs CPU-index agreement <= 2e-5 on 512 fixed-seed in-window sites
(seed 926213); the maximum was 1.889e-4 at one site (1 of 512), while CPU index vs reference was
1.79e-7. No initialization, fitting cell, held-out access or outcome occurred.

Driver diagnosis after the failure (same 512 sites): the teacher uses normalized blending and the
offending site has weight sum 4.19e-7 (CPU) versus 4.24e-7 (CUDA), i.e. a near-empty region where
the normalization amplifies float32 accumulation-order differences between the CUDA kernel and the
CPU CSR stream. Per the approved review's condition 3 the root is consumed; a repeat needs a new
task id and prospective review. Files: failure receipt, run receipt, preflight, environment, launch
context and the prepare log tail.
