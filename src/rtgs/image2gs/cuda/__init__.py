"""CUDA extension sources for the batched stage-1 2D splatting renderer.

The ``.cpp``/``.cu`` files here are compiled on first use via ``torch.utils.cpp_extension``
(see ``rtgs.image2gs.cuda_backend``). Nothing in this package imports CUDA at import time.
"""
