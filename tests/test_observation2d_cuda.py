"""Indexed observation-query CUDA backend: CPU-safe guards and GPU parity (self-skipping).

The CPU CSR index is the oracle. Because the GPU kernel accumulates each point's CSR row
sequentially (no atomics), parity tolerances only absorb FMA contraction, and repeated
queries must be bit-identical.
"""

import pytest
import torch

from rtgs.core.observation2d import GaussianObservationField, GaussianObservationIndex
from rtgs.core.observation2d_cuda import GaussianObservationIndexCuda
from rtgs.lift.compact_carve import CompactCarveConfig, build_query_backends

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
cpu_only = pytest.mark.skipif(torch.cuda.is_available(), reason="exercises the CPU-only guard")


def _field(
    n: int = 80,
    canvas: int = 64,
    *,
    grads: bool = True,
    fade: float = 0.4,
    blend: str = "normalized",
    residuals: bool = False,
    seed: int = 0,
) -> GaussianObservationField:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    kwargs = {
        "width": canvas,
        "height": canvas,
        "means": torch.rand(n, 2, generator=gen) * (canvas - 2.0) + 1.0,
        "log_scales": torch.log(torch.rand(n, 2, generator=gen) * 4.5 + 1.0),
        "rotations": (torch.rand(n, generator=gen) - 0.5) * 2.0,
        "colors": torch.rand(n, 3, generator=gen) * 1.4 - 0.2,
        "amplitudes": torch.rand(n, generator=gen) * 0.9 + 0.1,
        "support_fade_alpha": fade,
        "blend_mode": blend,
        "fit_window": (3, 2, canvas - 5, canvas - 4),
    }
    if grads:
        kwargs["color_grads"] = torch.randn(n, 2, 3, generator=gen) / 8.0
    if residuals:
        kwargs["mean_residuals"] = torch.randn(n, 2, generator=gen) / 64.0
    return GaussianObservationField(**kwargs)


def _points(field: GaussianObservationField, count: int = 512, seed: int = 1) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # Spread beyond the fit window so invalid and empty-tile points are exercised too.
    return torch.rand(count, 2, generator=gen) * (field.width + 6.0) - 3.0


def test_module_imports_without_cuda():
    assert hasattr(GaussianObservationIndexCuda, "query")
    assert hasattr(GaussianObservationIndexCuda, "query_weight_sum")


@cpu_only
def test_wrapper_requires_cuda():
    index = GaussianObservationIndex(_field(n=12))
    with pytest.raises(RuntimeError, match="cuda.is_available"):
        GaussianObservationIndexCuda(index)


def test_build_query_backends_cpu_matches_internal_defaults():
    fields = [_field(n=16, seed=0), _field(n=20, seed=1)]
    config = CompactCarveConfig(n_init_3d=8)
    backends = build_query_backends(fields, config, device="cpu")
    assert len(backends) == 2
    for field, backend in zip(fields, backends):
        assert isinstance(backend, GaussianObservationIndex)
        assert backend.field is field
        assert backend.tile_size == config.tile_size


def test_build_query_backends_rejects_bad_device():
    with pytest.raises(ValueError, match="'cpu' or 'cuda'"):
        build_query_backends([_field(n=8)], CompactCarveConfig(n_init_3d=8), device="tpu")


@cpu_only
def test_build_query_backends_cuda_without_gpu():
    with pytest.raises(RuntimeError, match="cuda.is_available"):
        build_query_backends([_field(n=8)], CompactCarveConfig(n_init_3d=8), device="cuda")


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize(
    "variant",
    [
        {},
        {"grads": False, "fade": 0.0},
        {"blend": "additive"},
        {"residuals": True},
    ],
    ids=["grads_fade", "plain", "additive", "residuals"],
)
def test_cuda_query_matches_cpu_index(variant):
    field = _field(**variant)
    index = GaussianObservationIndex(field)
    cuda_index = GaussianObservationIndexCuda(index)
    xy = _points(field)
    reference = index.query(xy)
    fast = cuda_index.query(xy)
    assert fast.color.device.type == "cpu"
    assert torch.equal(reference.valid, fast.valid)
    assert torch.allclose(fast.weight_sum, reference.weight_sum, atol=2e-5, rtol=2e-5)
    assert torch.allclose(fast.numerator, reference.numerator, atol=2e-5, rtol=2e-5)
    assert torch.allclose(fast.color, reference.color, atol=2e-5, rtol=2e-5)
    weight_sum = cuda_index.query_weight_sum(xy)
    assert torch.allclose(weight_sum, reference.weight_sum, atol=2e-5, rtol=2e-5)


@pytest.mark.cuda
@requires_cuda
def test_cuda_query_is_deterministic_and_serves_cuda_points():
    field = _field()
    cuda_index = GaussianObservationIndexCuda(GaussianObservationIndex(field))
    xy = _points(field).cuda()
    first = cuda_index.query(xy)
    second = cuda_index.query(xy)
    assert first.color.is_cuda
    # Sequential per-point accumulation, no atomics: repeat queries are bit-identical.
    assert torch.equal(first.color, second.color)
    assert torch.equal(first.weight_sum, second.weight_sum)


@pytest.mark.cuda
@requires_cuda
def test_cuda_counters_match_cpu_pair_stream():
    field = _field()
    index = GaussianObservationIndex(field)
    cuda_index = GaussianObservationIndexCuda(GaussianObservationIndex(field))
    xy = _points(field)
    index.query(xy)
    cuda_index.query(xy)
    assert cuda_index.total_pairs_evaluated == index.total_pairs_evaluated
    assert cuda_index.total_query_points == index.total_query_points


@pytest.mark.cuda
@requires_cuda
def test_cuda_query_rejects_gradient_inputs():
    field = _field(n=16)
    cuda_index = GaussianObservationIndexCuda(GaussianObservationIndex(field))
    xy = _points(field, count=8).requires_grad_(True)
    with pytest.raises(RuntimeError, match="inference-only"):
        cuda_index.query(xy)
