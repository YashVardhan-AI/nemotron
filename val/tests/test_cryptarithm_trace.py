"""Tests for the genuine-deduction CoT renderer (Phase 4.2).

In val/tests/ with the rest of the repo's tests (no reasoners/tests/ dir).
"""

from reasoners.cryptarithm_deduce_core import solve_problem
from reasoners.cryptarithm_trace import (
    make_trace_problem,
    reasoning_cryptarithm_arith,
)


def _is_arith_query(problem) -> bool:
    """True iff the solver recovered a digit map (arithmetic query, not the pure
    concat shortcut) -- only then are all demos verified numerically."""
    _ans, (mapping, _op) = solve_problem(
        {
            "examples": [
                {"input_value": e.input_value, "output_value": e.output_value}
                for e in problem.examples
            ],
            "question": problem.question,
        }
    )
    return bool(mapping)


def test_trace_ends_with_correct_boxed_answer():
    for seed in range(22):
        problem, answer = make_trace_problem(seed, 4)
        trace = reasoning_cryptarithm_arith(problem, answer)
        assert trace is not None, seed
        assert trace.rstrip().endswith("\\boxed{" + answer + "}"), seed


def test_trace_is_sound_no_mismatch_lines():
    # every claimed verification reproduces the demo (soundness).
    for seed in range(22):
        problem, answer = make_trace_problem(seed, 5)
        trace = reasoning_cryptarithm_arith(problem, answer)
        assert "MISMATCH" not in trace, seed


def test_arith_query_traces_include_every_demo():
    checked = 0
    for seed in range(22):
        problem, answer = make_trace_problem(seed, 5)
        if not _is_arith_query(problem):
            continue  # concat-shortcut queries narrate the concat reasoning only
        trace = reasoning_cryptarithm_arith(problem, answer)
        for ex in problem.examples:
            assert ex.input_value in trace, (seed, ex.input_value)
            assert ex.output_value in trace, (seed, ex.output_value)
        checked += 1
    assert checked > 0  # the sweep actually exercised arith queries


def test_trace_shows_genuine_deduction_markers():
    saw_struct = saw_prune = False
    for seed in range(22):
        problem, answer = make_trace_problem(seed, 5)
        t = reasoning_cryptarithm_arith(problem, answer)
        assert "s0 s1 op s3 s4" in t  # structure recognition, every trace
        low = t.lower()
        if "concatenation" in low:
            saw_struct = True
        if "rules out" in low or "rules it out" in low:
            saw_prune = True
    assert saw_struct  # concat resolved structurally somewhere
    assert saw_prune  # arithmetic ops ruled out by length/value somewhere


def test_trace_names_the_query_operation():
    words = {
        "add": "sum",
        "abs_diff": "absolute difference",
        "mul": "product",
        "concat": "concatenation",
        "rev_concat": "reverse concatenation",
    }
    for seed in range(22):
        problem, answer = make_trace_problem(seed, 4)
        data = {
            "examples": [
                {"input_value": e.input_value, "output_value": e.output_value}
                for e in problem.examples
            ],
            "question": problem.question,
        }
        _ans, (_map, op_info) = solve_problem(data)
        qname = op_info[problem.question[2]]
        trace = reasoning_cryptarithm_arith(problem, answer).lower()
        assert words[qname] in trace, (seed, qname)


def test_trace_within_token_budget():
    for seed in range(22):
        problem, answer = make_trace_problem(seed, 5)
        trace = reasoning_cryptarithm_arith(problem, answer)
        assert len(trace) < 12000, (seed, len(trace))
