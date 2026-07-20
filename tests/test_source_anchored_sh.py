import torch

from rtgs.core.sh import eval_sh_preactivation
from rtgs.lift.source_anchored_sh import (
    SourceAnchoredSH,
    SourceAnchoredSHFitConfig,
    fit_source_anchored_sh,
    real_sh_basis,
)


def _directions(count: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(90210)
    value = torch.randn(count, 3, generator=generator, dtype=dtype)
    return torch.nn.functional.normalize(value, dim=-1)


def test_real_sh_basis_matches_repository_evaluator_through_degree_three():
    directions = _directions(7)
    generator = torch.Generator().manual_seed(17)
    for degree in range(4):
        coefficients = torch.randn(
            7,
            (degree + 1) ** 2,
            3,
            generator=generator,
            dtype=torch.float64,
        )
        expected = eval_sh_preactivation(degree, coefficients, directions)
        actual = (real_sh_basis(degree, directions)[:, :, None] * coefficients).sum(dim=1) + 0.5
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-14)


def test_source_constraint_is_exact_after_arbitrary_optimizer_updates():
    directions = _directions(11)
    colors = torch.linspace(-0.2, 1.2, 33, dtype=torch.float64).reshape(11, 3)
    model = SourceAnchoredSH(
        degree=2,
        source_directions=directions,
        source_colors=colors,
    )
    optimizer = torch.optim.Adam([model.free], lr=0.2)
    other = torch.roll(directions, shifts=1, dims=0)
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        model.preactivation(other).square().mean().backward()
        optimizer.step()
        torch.testing.assert_close(
            model.source_preactivation(),
            colors,
            rtol=0.0,
            atol=2e-15,
        )


def test_fit_improves_non_source_colors_and_never_uses_source_weight():
    count, views = 8, 4
    source_view = torch.arange(count) % views
    directions = torch.stack(
        [torch.roll(_directions(count), view, dims=0) for view in range(views)]
    )
    true_coefficients = torch.randn(
        count,
        4,
        3,
        generator=torch.Generator().manual_seed(44),
        dtype=torch.float64,
    )
    target = torch.stack(
        [eval_sh_preactivation(1, true_coefficients, directions[view]) for view in range(views)]
    )
    rows = torch.arange(count)
    source_directions = directions[source_view, rows]
    source_colors = target[source_view, rows]
    weights = torch.ones(views, count, dtype=torch.float64)
    # Deliberately absurd source weights and colors would dominate an implementation that leaked
    # self-view evidence. The exact source target remains the explicit source_colors tensor.
    target = target.clone()
    target[source_view, rows] += 1000.0
    weights[source_view, rows] = 1e9
    result = fit_source_anchored_sh(
        source_directions=source_directions,
        source_colors=source_colors,
        view_directions=directions,
        target_colors=target,
        weights=weights,
        source_view_indices=source_view.long(),
        config=SourceAnchoredSHFitConfig(iterations=180, learning_rate=0.05),
    )
    assert result.losses[-1] < 0.02 * result.losses[0]
    assert result.source_max_abs_error <= 2e-15
    torch.testing.assert_close(
        result.model.source_preactivation(),
        source_colors,
        rtol=0.0,
        atol=2e-15,
    )


def test_degree_zero_has_no_trainable_null_direction():
    directions = _directions(3)
    colors = torch.tensor(
        [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
        dtype=torch.float64,
    )
    model = SourceAnchoredSH(
        degree=0,
        source_directions=directions,
        source_colors=colors,
    )
    with torch.no_grad():
        model.free.normal_()
    first = model.coefficients()
    with torch.no_grad():
        model.free.add_(10.0)
    second = model.coefficients()
    torch.testing.assert_close(first, second, rtol=0.0, atol=1e-14)
    torch.testing.assert_close(model.source_preactivation(), colors, rtol=0.0, atol=1e-14)
