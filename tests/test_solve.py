"""The GPTQ stage, on tensors small enough to reason about.

Skipped without torch, like tests/test_convert.py: the dev requirements do not carry it.
"""

import pytest

torch = pytest.importorskip("torch")

from pipeline import solve  # noqa: E402


def test_codes_land_on_the_grid_the_delegate_runs():
    torch.manual_seed(0)
    weight = torch.randn(8, 64)
    codes, scales = solve.solve_weight(weight, torch.eye(64))
    # Symmetric int4: XNNPACK reads codes in -8..7 with one positive scale per group.
    assert int(codes.min()) >= -8
    assert int(codes.max()) <= 7
    assert bool((scales > 0).all())
    assert codes.shape == (8, 64)
    assert scales.shape == (8, 64 // solve.GROUP)


def test_scales_are_exactly_what_the_file_can_store():
    # ExecuTorch 1.4 writes block scales as bf16. Solving against a scale the .pte cannot
    # hold would match the codes to a weight the runtime never sees.
    torch.manual_seed(1)
    _, scales = solve.solve_weight(torch.randn(4, 96), torch.eye(96))
    assert torch.equal(scales, solve.bf16_storage(scales))


def test_a_diagonal_hessian_gives_plain_rounding():
    # Not a defect: with no correlation between inputs there is no error worth pushing onto
    # later columns, so GPTQ degenerates to rounding onto the same grid. This pins that, so a
    # future change that accidentally ignores the Hessian does not look like an improvement.
    torch.manual_seed(2)
    weight = torch.randn(6, 64)
    flat = solve.dequantise(*solve.solve_weight(weight, torch.eye(64)))
    weighted = torch.eye(64)
    weighted[0, 0] = 1000.0
    scaled = solve.dequantise(*solve.solve_weight(weight, weighted))
    assert torch.equal(flat, scaled)


def test_a_correlated_hessian_beats_rounding_on_its_own_measure():
    # The property GPTQ is for: when inputs are correlated, pushing each column's error onto
    # the columns not yet rounded lowers the output error the Hessian measures.
    torch.manual_seed(3)
    weight = torch.randn(16, 128)
    x = torch.randn(512, 128) @ torch.randn(128, 128)  # correlated inputs
    hessian = x.T @ x / x.shape[0]

    solved = solve.dequantise(*solve.solve_weight(weight, hessian))
    # Round to nearest on the same grid and the same per-group scales.
    codes, scales = solve.solve_weight(weight, torch.eye(128))
    rounded = solve.dequantise(codes, scales)

    def output_error(candidate):
        delta = candidate - weight
        return float((delta @ hessian @ delta.T).trace())

    assert output_error(solved) < output_error(rounded)


def test_a_group_that_does_not_divide_the_row_is_refused():
    with pytest.raises(ValueError, match="divisible"):
        solve.solve_weight(torch.randn(4, 70), torch.eye(70))


def test_dequantise_reverses_the_grouping():
    codes = torch.full((2, 64), 3, dtype=torch.int8)
    scales = torch.full((2, 2), 0.5)
    out = solve.dequantise(codes, scales)
    assert out.shape == (2, 64)
    assert torch.allclose(out, torch.full((2, 64), 1.5))


def test_the_activation_rounding_matches_what_the_delegate_does():
    # Per token, asymmetric, int8: the values come back on a grid of at most 256 steps.
    torch.manual_seed(4)
    x = torch.randn(3, 40)
    q = solve.fake_act(x)
    assert q.shape == x.shape
    for row_in, row_out in zip(x, q, strict=True):
        assert len(torch.unique(row_out)) <= 256
        assert float((row_out - row_in).abs().max()) < float(row_in.abs().max())


def test_injection_refuses_codes_that_do_not_match_the_model():
    # A solve run against a different checkpoint, or a renamed module, would otherwise
    # export cleanly, be labelled GPTQ and carry weights that were quietly rounded.
    from pipeline import export_with_codes

    class Fake:
        def named_modules(self):
            return []

    injected, left = export_with_codes.inject(Fake(), {"layers.0.attention.wq": {"qdata": None, "scale": None}})
    assert injected == 0
    assert left == []


def test_injection_checks_the_grid_before_writing():
    from pipeline import export_with_codes

    class Weight:
        def __init__(self):
            self.qdata = torch.zeros(4, 32, dtype=torch.int8)
            self.scale = torch.ones(4, 1)
            self.zero_point = torch.zeros(4, 1)

    class Module:
        def __init__(self):
            self.weight = Weight()

        def named_modules(self):
            return [("linear", self)]

    # A negative scale is not representable in XNNPACK's blockwise int4 format.
    bad = {"linear": {"qdata": torch.zeros(4, 32, dtype=torch.int8), "scale": -torch.ones(4, 1)}}
    with pytest.raises(ValueError, match="not positive"):
        export_with_codes.inject(Module(), bad)

    # Nor is a code outside -8..7.
    off_grid = {"linear": {"qdata": torch.full((4, 32), 9, dtype=torch.int8), "scale": torch.ones(4, 1)}}
    with pytest.raises(ValueError, match="int4 grid"):
        export_with_codes.inject(Module(), off_grid)


def test_injection_refuses_a_scale_that_is_not_finite():
    # `inf > 0` is True, so a positivity check alone passes an infinite scale straight
    # through; bf16 stores it, and every weight in that group is inf at runtime.
    from pipeline import export_with_codes

    class Weight:
        def __init__(self):
            self.qdata = torch.zeros(4, 32, dtype=torch.int8)
            self.scale = torch.ones(4, 1)
            self.zero_point = torch.zeros(4, 1)

    class Module:
        def __init__(self):
            self.weight = Weight()

        def named_modules(self):
            return [("linear", self)]

    for bad in (torch.full((4, 1), float("inf")), torch.full((4, 1), float("nan"))):
        codes = {"linear": {"qdata": torch.zeros(4, 32, dtype=torch.int8), "scale": bad}}
        with pytest.raises(ValueError, match="not finite"):
            export_with_codes.inject(Module(), codes)
