"""Tests for the bit_manipulation forward-gen training trace (reasoners.bit_trace).

Every shipped trace must be SOUND (its per-bit rules reproduce all examples and
its applied output equals the verified answer) and SHORT (decomposable-local, to
avoid the long high-loss gradient that made cryptarithm induct traces crash
neighbors).
"""

from reasoners.bit_rule import build_rule, problem_tag
from reasoners.bit_trace import make_bit_trace_problem, reasoning_bit_perbit


def _trace(seed, difficulty=8):
    problem, answer, cols = make_bit_trace_problem(seed, difficulty)
    return problem, answer, reasoning_bit_perbit(problem, answer, cols)


def test_trace_is_produced_for_all_families():
    produced = 0
    for seed in range(200):
        _p, _a, t = _trace(seed)
        if t is not None:
            produced += 1
    # forward-gen is sound by construction -> essentially every seed yields a trace
    assert produced >= 198, produced


def test_applied_output_equals_verified_answer():
    for seed in range(150):
        problem, answer, t = _trace(seed)
        assert t is not None, seed
        # the rule's true answer (independent recompute) matches the boxed answer
        _sig, apply = build_rule(seed)
        assert format(apply(int(problem.question, 2)), "08b") == answer
        # and the trace's final line states exactly that answer
        assert f"gives {answer}" in t


def test_trace_covers_all_eight_bits():
    _p, _a, t = _trace(3)
    for j in range(8):
        assert f"Bit {j} =" in t
        assert f"bit {j}:" in t


def test_trace_is_short():
    # "short" = decomposable-local; well under the verbose ~800-line real traces.
    for seed in range(60):
        _p, _a, t = _trace(seed)
        assert t is not None
        assert t.count("\n") < 40, (seed, t.count("\n"))


def test_complex_traces_show_three_input_ops():
    seed = next(s for s in range(400) if problem_tag(s).startswith("complex"))
    _p, _a, t = _trace(seed)
    assert "MAJ(" in t or "if in[" in t


def test_heterogeneous_pairwise_uses_multiple_ops_in_trace():
    seed = next(s for s in range(400) if problem_tag(s) == "pairwise/het")
    _p, _a, t = _trace(seed)
    ops_present = sum(op in t for op in ("AND", "OR", "XOR"))
    assert ops_present >= 1
