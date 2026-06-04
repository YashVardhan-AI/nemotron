"""Tests for the genuine-deduction CoT renderer (Phase 4.2).

In val/tests/ with the rest of the repo's tests (no reasoners/tests/ dir).
"""

from reasoners.cryptarithm_deduce_core import solve_problem
from reasoners.cryptarithm_trace import (
    make_trace_problem,
    reasoning_cryptarithm_arith,
    reasoning_cryptarithm_induct,
    reasoning_cryptarithm_propagate,
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


# --- Phase 4.3: compute-and-correct renderer --------------------------------


def test_propagate_ends_with_correct_boxed_answer_and_is_sound():
    for seed in range(22):
        problem, answer = make_trace_problem(seed, 5)
        trace = reasoning_cryptarithm_propagate(problem, answer)
        assert trace is not None, seed
        assert trace.rstrip().endswith("\\boxed{" + answer + "}"), seed
        assert "MISMATCH" not in trace
        assert len(trace) < 12000


def test_propagate_agrees_with_deduction_renderer():
    # both styles must reach the same boxed answer for the same problem.
    for seed in range(22):
        problem, answer = make_trace_problem(seed, 4)
        a = reasoning_cryptarithm_arith(problem, answer)
        b = reasoning_cryptarithm_propagate(problem, answer)
        assert a.rstrip().endswith("\\boxed{" + answer + "}")
        assert b.rstrip().endswith("\\boxed{" + answer + "}")


def test_propagate_shows_a_correction():
    saw_correction = False
    for seed in range(22):
        problem, answer = make_trace_problem(seed, 5)
        low = reasoning_cryptarithm_propagate(problem, answer).lower()
        if "wrong" in low and "correct" in low:
            saw_correction = True
    assert saw_correction  # the compute-and-correct step fires on arith queries


# --- Phase 4.4: induction-search CoT renderer --------------------------------


def _induct_arith_seeds(lo=0, hi=40):
    """Seeds whose query is arithmetic (digit map recovered) -- the induct style
    targets these; concat-shortcut seeds fall back to the deduce style."""
    out = []
    for seed in range(lo, hi):
        problem, answer = make_trace_problem(seed, 4)
        if _is_arith_query(problem):
            out.append((seed, problem, answer))
    return out


def test_induct_ends_with_correct_boxed_answer_and_is_sound():
    seeds = _induct_arith_seeds()
    assert seeds  # the sweep actually exercises arith queries
    for seed, problem, answer in seeds:
        trace = reasoning_cryptarithm_induct(problem, answer)
        assert trace is not None, seed
        assert trace.rstrip().endswith("\\boxed{" + answer + "}"), seed
        assert "MISMATCH" not in trace, seed


def test_induct_shows_state_blocks_and_backtracking_in_sweep():
    saw_state = saw_backtrack = False
    for _seed, problem, answer in _induct_arith_seeds():
        t = reasoning_cryptarithm_induct(problem, answer)
        if "STATE" in t:
            saw_state = True
        if "backtrack" in t.lower():
            saw_backtrack = True
    assert saw_state  # inductive scratchpad re-emits state
    assert saw_backtrack  # genuine search appears somewhere


def test_induct_includes_round_trip_injectivity_check():
    for _seed, problem, answer in _induct_arith_seeds():
        low = reasoning_cryptarithm_induct(problem, answer).lower()
        assert "injectiv" in low or "all distinct" in low


def test_induct_arithmetic_is_not_atomic():
    saw_columns = False
    for _seed, problem, answer in _induct_arith_seeds():
        if "carry" in reasoning_cryptarithm_induct(problem, answer).lower():
            saw_columns = True
    assert saw_columns


def test_induct_within_token_budget():
    for _seed, problem, answer in _induct_arith_seeds():
        t = reasoning_cryptarithm_induct(problem, answer)
        assert len(t) < 16000, (len(t),)
