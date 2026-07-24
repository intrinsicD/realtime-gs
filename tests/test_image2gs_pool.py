"""Stage 1: fixed-capacity gaussian pool + free list (opt-in native fitting)."""

import pytest
import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.image2gs.fit import FitConfig, fit_image
from rtgs.image2gs.pool import GaussianPool2D, fit_pooled_from_initialization
from rtgs.image2gs.renderer2d import render_gaussians_2d


def _smooth_image(h: int = 24, w: int = 24) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    return torch.stack([xx, yy, 0.5 * (xx + yy)], dim=-1).clamp(0, 1)


def _example_gaussians(n: int, h: int = 24, w: int = 24) -> Gaussians2D:
    torch.manual_seed(0)
    xy = torch.rand(n, 2) * torch.tensor([w - 1e-3, h - 1e-3])
    chol = torch.zeros(n, 3)
    chol[:, 0] = 2.0 + torch.rand(n)
    chol[:, 2] = 2.0 + torch.rand(n)
    return Gaussians2D(
        xy=xy, chol=chol, color=torch.rand(n, 3), weight=torch.rand(n).clamp(0.1, 0.9)
    )


def _pool(n: int, capacity: int, h: int = 24, w: int = 24) -> GaussianPool2D:
    g0 = _example_gaussians(n, h, w)
    return GaussianPool2D(g0, torch.tensor([float(w), float(h)]), capacity)


def test_capacity_must_cover_initial_count():
    with pytest.raises(ValueError, match="capacity"):
        _pool(n=8, capacity=4)


def test_parked_rows_are_omitted_exactly_additive():
    # Additive compositor: rendering the live subset plus the parked subset recovers the full set,
    # so parking removes exactly (and only) the parked rows' contribution.
    pool = _pool(n=8, capacity=8)
    all_rows = torch.arange(8)
    parked = torch.tensor([1, 3, 5])
    live = torch.tensor([0, 2, 4, 6, 7])
    r_all = render_gaussians_2d(pool._build_rows(all_rows), 24, 24)
    r_live = render_gaussians_2d(pool._build_rows(live), 24, 24)
    r_parked = render_gaussians_2d(pool._build_rows(parked), 24, 24)
    assert torch.allclose(r_live + r_parked, r_all, atol=1e-6)

    optimizer = torch.optim.Adam(pool.params(), lr=1e-2)
    pool.park(parked, optimizer)
    assert torch.allclose(render_gaussians_2d(pool.build_active(), 24, 24), r_live, atol=1e-6)


def test_parked_rows_receive_zero_gradient():
    pool = _pool(n=8, capacity=8)
    optimizer = torch.optim.Adam(pool.params(), lr=1e-2)
    parked = torch.tensor([1, 3, 5])
    pool.park(parked, optimizer)
    render_gaussians_2d(pool.build_active(), 24, 24).sum().backward()
    for param in pool.params():
        assert param.grad is not None
        assert torch.count_nonzero(param.grad[parked]) == 0
    # At least one live row carries a gradient (the pool is actually optimizing something).
    assert torch.count_nonzero(pool.params()[0].grad) > 0


def test_free_list_recycles_without_reallocation():
    pool = _pool(n=4, capacity=6)
    optimizer = torch.optim.Adam(pool.params(), lr=1e-2)
    tensor_ids = [id(param) for param in pool.params()]
    assert pool.live_count == 4
    assert pool.free_rows().tolist() == [4, 5]

    pool.park(torch.tensor([0]), optimizer)
    assert pool.live_count == 3
    assert pool.free_rows().tolist() == [0, 4, 5]

    seed = _example_gaussians(1)
    pool.activate(pool.free_rows()[:1], seed, optimizer)
    assert pool.live_count == 4
    assert pool.free_rows().tolist() == [4, 5]
    # The raw parameter tensors were never reallocated.
    assert [id(param) for param in pool.params()] == tensor_ids


def test_optimizer_is_not_rebuilt_and_recycled_moments_reset():
    pool = _pool(n=8, capacity=8)
    optimizer = torch.optim.Adam(pool.params(), lr=1e-2)
    # One real step so Adam populates momentum buffers.
    render_gaussians_2d(pool.build_active(), 24, 24).sum().backward()
    optimizer.step()
    before = [id(param) for param in optimizer.param_groups[0]["params"]]

    rows = torch.tensor([2, 5])
    pool.park(rows, optimizer)
    pool.activate(rows, _example_gaussians(2), optimizer)

    after = [id(param) for param in optimizer.param_groups[0]["params"]]
    assert before == after  # same parameter objects -> optimizer never rebuilt
    for param in pool.params():
        state = optimizer.state[param]
        assert torch.count_nonzero(state["exp_avg"][rows]) == 0
        assert torch.count_nonzero(state["exp_avg_sq"][rows]) == 0


def test_pool_is_off_by_default():
    assert FitConfig().pool is False


def test_pool_without_triage_matches_native_fit():
    # With triage disabled the pool is a faithful reparameterization of the native fit: the parked
    # spare rows contribute nothing, so the live rows follow the identical optimization.
    image = _smooth_image()
    shared = dict(n_gaussians=16, iterations=40, log_every=40)
    native, native_hist = fit_image(image, FitConfig(**shared), seed=0)
    pooled, pooled_hist = fit_image(
        image, FitConfig(pool=True, pool_capacity=32, pool_triage_every=0, **shared), seed=0
    )
    assert pooled.n == native.n == 16
    assert torch.allclose(pooled.xy, native.xy, atol=1e-5)
    assert torch.allclose(pooled.color, native.color, atol=1e-5)
    assert pooled_hist["final_psnr"] == pytest.approx(native_hist["final_psnr"], abs=1e-4)


def test_pooled_fit_with_triage_grows_and_improves():
    image = _smooth_image()
    config = FitConfig(
        pool=True,
        n_gaussians=16,
        pool_capacity=48,
        iterations=120,
        pool_triage_every=30,
        pool_prune_count=2,
        pool_spawn_count=8,
        log_every=20,
    )
    gaussians, history = fit_image(image, config, seed=0)

    assert isinstance(gaussians, Gaussians2D)
    assert history["pool_capacity"] == 48
    assert history["live_count"] == gaussians.n
    assert 16 < gaussians.n <= 48  # spawning at residual peaks grew the live set past the seed
    for field in (gaussians.xy, gaussians.chol, gaussians.color, gaussians.weight):
        assert torch.isfinite(field).all()
    # Optimization made progress (a floor, not a snapshot of the untuned triage policy).
    assert history["final_psnr"] > history["psnr"][0][1]


def test_pooled_fit_is_deterministic():
    image = _smooth_image()
    config = FitConfig(pool=True, n_gaussians=16, pool_capacity=32, iterations=25, log_every=25)
    g1, _ = fit_image(image, config, seed=0)
    g2, _ = fit_image(image, config, seed=0)
    assert torch.allclose(g1.xy, g2.xy)
    assert torch.allclose(g1.color, g2.color)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"backend": "structsplat"}, "native backend"),
        ({"freeze_geometry": True}, "freeze_geometry"),
        ({"pool_capacity": 4, "n_gaussians": 8}, "pool_capacity"),
    ],
)
def test_pool_validation_rejects_incompatible_configs(overrides, match):
    image = _smooth_image()
    config = FitConfig(pool=True, iterations=5, **overrides)
    with pytest.raises(ValueError, match=match):
        fit_image(image, config, seed=0)


def test_pooled_fit_accepts_supplied_initialization():
    image = _smooth_image()
    g0 = _example_gaussians(12)
    config = FitConfig(pool=True, pool_capacity=24, iterations=15, log_every=15)
    target = image
    gaussians, history = fit_pooled_from_initialization(
        image, target, g0, config, mask=None, xy_offset=image.new_zeros(2)
    )
    assert gaussians.n == history["live_count"]
    assert history["pool_capacity"] == 24
