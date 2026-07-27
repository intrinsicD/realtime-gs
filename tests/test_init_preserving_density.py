"""ADR-YYYY acceptance criteria: event invariance, second moments, lineage, guards, isolation.

A1 runs the vanilla operators through the *same* harness and records their event error, because
a tolerance nobody compared against a baseline is a number, not a test.
"""

from __future__ import annotations

import math

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.metrics import psnr
from rtgs.optim.init_density import (
    InitPreservingConfig,
    InitPreservingController,
    _appearance_preserving_clone,
    _appearance_preserving_split,
)
from rtgs.optim.init_trust import NULL_ROOT, TrustConfig, TrustSchedule
from rtgs.render.torch_ref import TorchRasterizer


def _camera() -> Camera:
    rotation = torch.eye(3)
    return Camera(
        fx=180.0,
        fy=180.0,
        cx=64.0,
        cy=64.0,
        width=128,
        height=128,
        R=rotation,
        t=torch.tensor([0.0, 0.0, 2.0]),
    )


def _state(n: int = 24, seed: int = 7) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    means = (torch.rand((n, 3), generator=generator) - 0.5) * 0.6
    means[:, 2] = 0.0
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0
    log_scales = torch.log(
        torch.tensor([0.05, 0.03, 0.004]).expand(n, 3)
        * (0.8 + 0.4 * torch.rand((n, 3), generator=generator))
    )
    opacities = torch.logit(torch.full((n,), 0.4))
    sh0 = torch.rand((n, 1, 3), generator=generator) * 0.4
    shN = torch.zeros(n, 15, 3)
    return {
        "means": torch.nn.Parameter(means),
        "quats": torch.nn.Parameter(quats),
        "scales": torch.nn.Parameter(log_scales),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(shN),
    }


def _render(params: dict[str, torch.Tensor], camera: Camera) -> torch.Tensor:
    model = Gaussians3D(
        means=params["means"].detach(),
        quats=params["quats"].detach(),
        log_scales=params["scales"].detach(),
        opacity=torch.sigmoid(params["opacities"].detach()),
        sh=torch.cat([params["sh0"].detach(), params["shN"].detach()], dim=1),
    )
    return TorchRasterizer().render(model, camera).color.clamp(0, 1)


def _apply_event(
    params: dict[str, torch.Tensor],
    rows: torch.Tensor,
    direction: torch.Tensor,
    cfg: InitPreservingConfig,
    operator: str,
) -> dict[str, torch.Tensor]:
    """Return the post-event parameter set for one operator, without optimizer surgery."""
    updated = {name: value.detach().clone() for name, value in params.items()}
    if operator == "clone":
        parent, child = _appearance_preserving_clone(params, rows, direction, cfg)
        for name, value in parent.items():
            updated[name][rows] = value
        return {name: torch.cat([updated[name], child[name]]) for name in updated}
    children = _appearance_preserving_split(params, rows, direction, cfg)
    keep = torch.ones(params["means"].shape[0], dtype=torch.bool)
    keep[rows] = False
    return {
        name: torch.cat([updated[name][keep], children[0][name], children[1][name]])
        for name in updated
    }


# -- A1: event invariance ----------------------------------------------------------------------


@pytest.mark.parametrize("operator", ["clone", "split"])
def test_a1_event_invariance_beats_the_vanilla_baseline(operator):
    camera = _camera()
    params = _state()
    rows = torch.arange(params["means"].shape[0])
    direction = torch.zeros(params["means"].shape[0], 2)
    direction[:, 0] = 1.0

    before = _render(params, camera)
    preserving = _render(
        _apply_event(params, rows, direction, InitPreservingConfig(), operator), camera
    )
    vanilla = _render(
        _apply_event(
            params, rows, direction, InitPreservingConfig(vanilla_operators=True), operator
        ),
        camera,
    )
    preserving_db = psnr(preserving, before)
    vanilla_db = psnr(vanilla, before)
    # Measured on this fixture: clone 42.6 dB vs vanilla 28.3 dB; split 47.2 dB vs 32.8 dB.
    # The 40 dB figure is the ADR-YYYY A1 gate, and the vanilla arm is what it protects against.
    assert preserving_db > vanilla_db, (preserving_db, vanilla_db)
    assert preserving_db >= 40.0, preserving_db


# -- A2: second-moment preservation -------------------------------------------------------------


def test_a2_split_preserves_the_mixture_second_moment_exactly():
    params = _state()
    rows = torch.arange(params["means"].shape[0])
    direction = torch.zeros(params["means"].shape[0], 2)
    direction[:, 1] = 1.0  # snap to the t2 axis
    cfg = InitPreservingConfig()
    children = _appearance_preserving_split(params, rows, direction, cfg)

    sigma = params["scales"].detach().exp()[:, 1]
    offset = cfg.split_offset * sigma
    expected = (sigma.square() - offset.square()).sqrt()
    for child in children:
        assert torch.allclose(child["scales"][:, 1].exp(), expected, atol=1e-6)
        # Second moment of the two-child mixture about the parent mean: sigma'^2 + d^2.
        recovered = (child["scales"][:, 1].exp().square() + offset.square()).sqrt()
        assert float((recovered.log() - sigma.log()).abs().max()) < 1e-4
    # The untouched axes are inherited bit-exactly.
    for axis in (0, 2):
        assert torch.equal(children[0]["scales"][:, axis], params["scales"].detach()[:, axis])


def test_a2_clone_shrink_has_a_needle_floor():
    params = _state()
    rows = torch.arange(params["means"].shape[0])
    direction = torch.zeros(params["means"].shape[0], 2)
    direction[:, 0] = 1.0
    cfg = InitPreservingConfig(clone_offset=8.0, clone_shrink_floor=0.3)
    _, child = _appearance_preserving_clone(params, rows, direction, cfg)
    sigma = params["scales"].detach().exp()[:, 0]
    assert torch.allclose(child["scales"][:, 0].exp(), 0.3 * sigma, atol=1e-6)


def test_appearance_matched_opacity_is_the_peak_composite_inverse():
    params = _state()
    rows = torch.arange(params["means"].shape[0])
    direction = torch.zeros(params["means"].shape[0], 2)
    _, child = _appearance_preserving_clone(params, rows, direction, InitPreservingConfig())
    matched = torch.sigmoid(child["opacities"])
    composite = 1.0 - (1.0 - matched) ** 2
    assert torch.allclose(composite, torch.sigmoid(params["opacities"].detach()), atol=1e-5)


def test_split_offsets_are_deterministic_and_never_sampled():
    params = _state()
    rows = torch.arange(params["means"].shape[0])
    direction = torch.zeros(params["means"].shape[0], 2)
    torch.manual_seed(1)
    first = _appearance_preserving_split(params, rows, direction, InitPreservingConfig())
    torch.manual_seed(999)
    second = _appearance_preserving_split(params, rows, direction, InitPreservingConfig())
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a["means"], b["means"])
    # The two children straddle the parent exactly.
    assert torch.allclose(
        0.5 * (first[0]["means"] + first[1]["means"]), params["means"].detach(), atol=1e-6
    )


# -- A3: lineage integrity ------------------------------------------------------------------------


def test_a3_children_inherit_root_and_trust_state_relocations_do_not():
    schedule = TrustSchedule(TrustConfig(beta=0.3, t_trust=10), 4)
    schedule.trust_age = torch.tensor([5.0, 5.0, 5.0, 5.0])
    keep = torch.tensor([True, True, False, True])
    parents = torch.tensor([0, 1, -1])  # two inherited children plus one insertion
    schedule.reindex(keep, parents)
    assert schedule.root_id.tolist() == [0, 1, 3, 0, 1, NULL_ROOT]
    assert schedule.trust_age.tolist() == [5.0, 5.0, 5.0, 5.0, 5.0, 0.0]
    trust = schedule.trust()
    assert float(trust[-1]) == 1.0  # NULL_ROOT is never throttled
    assert float(trust[0]) == pytest.approx(0.3 + 0.7 * 0.5)


def test_trust_multipliers_follow_the_declared_asymmetry():
    coherence = torch.tensor([0.0, 1.0])
    schedule = TrustSchedule(TrustConfig(beta=0.25, t_trust=100), 2, coherence=coherence)
    multipliers = schedule.multipliers()
    assert torch.allclose(multipliers["means"].reshape(-1), torch.tensor([0.25, 0.25]))
    # sigma_n is a prior, never trusted; the tangential axes are measurements.
    assert torch.allclose(multipliers["scales"][:, 2], torch.ones(2))
    assert torch.allclose(multipliers["scales"][:, :2], torch.full((2, 2), 0.25))
    # Rotation trust is gated by planarity: a non-planar neighbourhood gets no throttle.
    assert float(multipliers["quats"][0]) == pytest.approx(1.0)
    assert float(multipliers["quats"][1]) == pytest.approx(0.25)
    assert "opacities" not in multipliers and "sh0" not in multipliers


def test_trust_is_an_exact_per_row_learning_rate():
    params = {"means": torch.nn.Parameter(torch.zeros(2, 3))}
    schedule = TrustSchedule(TrustConfig(beta=0.5, t_trust=1_000_000), 2)
    schedule.root_id = torch.tensor([0, NULL_ROOT])
    schedule.capture(params)
    with torch.no_grad():
        params["means"].add_(torch.ones(2, 3))
    schedule.apply(params, 1)
    assert torch.allclose(params["means"].detach()[0], torch.full((3,), 0.5))
    assert torch.allclose(params["means"].detach()[1], torch.ones(3))


def test_disabled_trust_is_a_no_op():
    params = {"means": torch.nn.Parameter(torch.zeros(2, 3))}
    schedule = TrustSchedule(TrustConfig(enabled=False), 2)
    schedule.capture(params)
    with torch.no_grad():
        params["means"].add_(torch.ones(2, 3))
    schedule.apply(params, 1)
    assert torch.equal(params["means"].detach(), torch.ones(2, 3))


def test_trust_config_rejects_a_freeze():
    with pytest.raises(ValueError):
        TrustConfig(beta=0.0)
    with pytest.raises(ValueError):
        TrustConfig(t_trust=0)


# -- controller end to end -------------------------------------------------------------------------


def _controller_round(cfg: InitPreservingConfig, iteration: int = 100):
    camera = _camera()
    params = _state()
    optimizers = {
        name: torch.optim.Adam([{"params": [parameter], "lr": 1e-3, "name": name}], eps=1e-15)
        for name, parameter in params.items()
    }
    controller = InitPreservingController(cfg, params["means"].shape[0], 1.0)
    schedule = TrustSchedule(TrustConfig(), params["means"].shape[0])
    prediction = _render(params, camera)
    target = torch.zeros_like(prediction)
    target[40:90, 40:90] = 0.9  # a structured block the model does not explain
    model = Gaussians3D(
        means=params["means"].detach(),
        quats=params["quats"].detach(),
        log_scales=params["scales"].detach(),
        opacity=torch.sigmoid(params["opacities"].detach()),
        sh=torch.cat([params["sh0"].detach(), params["shN"].detach()], dim=1),
    )
    out = TorchRasterizer().render(model, camera)
    for _ in range(3):
        controller.accumulate(
            view=0,
            camera=camera,
            prediction=prediction,
            target=target,
            alpha=out.alpha,
            depth=out.depth,
            visible=out.visible,
        )
    new_params = controller.step(iteration, params, optimizers, trust=schedule)
    return controller, new_params, schedule


def test_controller_grows_through_named_channels_and_logs_them():
    controller, new_params, schedule = _controller_round(
        InitPreservingConfig(start_iter=1, every=1, max_gaussians=400)
    )
    assert controller.stats, "a scheduled round must record a stat entry even if it grows nothing"
    record = controller.stats[-1]
    assert set(record["channel_budgets"]) == {"clone", "split", "insert"}
    assert record["n_after"] == new_params["means"].shape[0]
    assert record["cloned"] + record["split"] + record["inserted"] > 0
    assert schedule.root_id.numel() == new_params["means"].shape[0]
    for name, value in new_params.items():
        assert value.shape[0] == record["n_after"], name
        assert value.requires_grad


def test_controller_never_exceeds_the_primitive_budget():
    controller, new_params, _ = _controller_round(
        InitPreservingConfig(start_iter=1, every=1, max_gaussians=30)
    )
    assert new_params["means"].shape[0] <= 30


def test_controller_is_inert_outside_its_schedule():
    controller, new_params, _ = _controller_round(
        InitPreservingConfig(start_iter=1000, every=1000), iteration=100
    )
    assert not controller.stats
    assert new_params["means"].shape[0] == 24


def test_gradient_trigger_level_remains_selectable():
    cfg = InitPreservingConfig(start_iter=1, every=1, gradient_trigger=True, max_gaussians=400)
    controller, _, _ = _controller_round(cfg)
    assert controller.stats  # the comparison level must stay implemented and runnable


def test_opacity_reset_level_remains_selectable():
    cfg = InitPreservingConfig(start_iter=1, every=1, opacity_reset_every=1, max_gaussians=400)
    _, new_params, _ = _controller_round(cfg)
    assert float(torch.sigmoid(new_params["opacities"]).max()) <= 0.011 + 1e-6


def test_config_rejects_out_of_range_levels():
    with pytest.raises(ValueError):
        InitPreservingConfig(noise_floor_quantile=1.0)
    with pytest.raises(ValueError):
        InitPreservingConfig(clone_fraction=1.5)
    with pytest.raises(ValueError):
        InitPreservingConfig(every=0)


def test_null_root_primitives_use_the_largest_axis_fallback():
    from rtgs.optim.init_density import _dominant_axis

    params = _state()
    rows = torch.arange(params["means"].shape[0])
    no_evidence = torch.zeros(params["means"].shape[0], 2)
    axis = _dominant_axis(params, rows, no_evidence)
    scales = params["scales"].detach().exp()
    assert torch.equal(axis, (scales[:, 1] > scales[:, 0]).to(torch.int64))


def test_quantization_angle_is_reported():
    from rtgs.optim.init_density import _quantization_angle_deg

    direction = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    axis = torch.tensor([0, 0, 0])
    angles = _quantization_angle_deg(direction, axis)
    assert float(angles[0]) == pytest.approx(45.0, abs=1e-4)
    assert float(angles[1]) == pytest.approx(0.0, abs=1e-4)
    assert float(angles[2]) == 0.0


# -- A5: profile isolation --------------------------------------------------------------------


def test_classic_controller_records_its_surgery_for_the_lineage_tracker():
    from rtgs.optim.density import DensityConfig, DensityController

    params = _state()
    optimizers = {
        name: torch.optim.Adam([{"params": [parameter], "lr": 1e-3, "name": name}], eps=1e-15)
        for name, parameter in params.items()
    }
    controller = DensityController(DensityConfig(start_iter=1, every=1), 24, 1.0)
    controller.grad_accum = torch.full((24,), 1.0)
    controller.count = torch.ones(24)
    schedule = TrustSchedule(TrustConfig(), 24)
    new_params = controller.step(1, params, optimizers, generator=torch.Generator())
    surgery = controller.last_surgery
    assert surgery is not None
    schedule.reindex(surgery["keep_mask"], surgery["parent_rows"])
    assert schedule.root_id.numel() == new_params["means"].shape[0]
    # Split parents are removed by their own event; their children must still inherit the root.
    assert bool((schedule.root_id >= 0).all())


def test_trust_rejects_the_gsplat_strategies_it_cannot_track(tiny_scene):
    from rtgs.optim.trainer import TrainConfig, Trainer

    cfg = TrainConfig(
        iterations=1,
        rasterizer="torch",
        device="cpu",
        densify=True,
        density_strategy="gsplat-default",
        trust=TrustConfig(),
    )
    with pytest.raises(ValueError, match="reports its surgery"):
        Trainer(cfg).train(tiny_scene, tiny_scene.gt_gaussians)


def test_a5_default_trainer_config_leaves_both_features_off():
    from rtgs.optim.trainer import TrainConfig

    cfg = TrainConfig()
    assert cfg.init_density is None
    assert cfg.trust is None
    assert cfg.density_strategy == "classic"


def test_a5_disabled_profile_reproduces_the_incumbent_bit_for_bit(tiny_scene):
    from rtgs.optim.trainer import TrainConfig, Trainer

    scene, gaussians = tiny_scene, tiny_scene.gt_gaussians
    base = TrainConfig(iterations=6, rasterizer="torch", device="cpu", densify=False, seed=3)
    profiled = TrainConfig(
        iterations=6,
        rasterizer="torch",
        device="cpu",
        densify=False,
        seed=3,
        trust=TrustConfig(enabled=False),
    )
    first, _ = Trainer(base).train(scene, gaussians)
    second, _ = Trainer(profiled).train(scene, gaussians)
    assert torch.equal(first.means, second.means)
    assert torch.equal(first.log_scales, second.log_scales)


def test_hexagonal_cover_sanity_is_unaffected():
    """The incumbent surfel_init rule must keep working; this profile does not replace it."""
    from rtgs.lift.surfel_init import hexagonal_cover_ripple

    assert hexagonal_cover_ripple(0.5) < hexagonal_cover_ripple(0.4)
    assert math.isfinite(hexagonal_cover_ripple(0.55))
