"""Soundness tests for the instrumented cryptarithm solver port.

The solver is the AUTHORITY that recovers the rule for forward-generated
problems; the CoT renderers narrate its decision log. So the gate here is:
on well-determined generated problems it (a) recovers the planted answer and
(b) returns a rule consistent with the query. (Tests live in val/tests/ with the
rest of the repo's tests; there is no reasoners/tests/ dir.)
"""

from reasoners.cryptarithm_deduce_core import (
    is_uniquely_solvable,
    sample_solvable,
    solve_problem,
)


def _data(examples, q_input):
    return {
        "examples": [{"input_value": i, "output_value": o} for i, o in examples],
        "question": q_input,
    }


def test_sample_solvable_is_always_uniquely_recovered():
    # The generation contract: every emitted instance is uniquely deducible AND
    # the public solver recovers exactly the planted answer.
    for seed in range(40):
        _rule, examples, q_input, q_answer = sample_solvable(seed, 4)
        assert is_uniquely_solvable(examples, q_input, q_answer), seed
        ans, _info = solve_problem(_data(examples, q_input))
        assert ans == q_answer, (seed, ans, q_answer)


def test_recovered_map_is_injective():
    for seed in range(25):
        _rule, examples, q_input, _q_answer = sample_solvable(seed, 4)
        _ans, (mapping, _op_info) = solve_problem(_data(examples, q_input))
        if mapping:  # arith path returns a digit map (concat shortcut returns {})
            assert len(set(mapping.values())) == len(mapping)  # injective


def test_trace_mode_returns_decision_log_ending_in_answer():
    for seed in range(25):
        _rule, examples, q_input, q_answer = sample_solvable(seed, 4)
        ans, _info, log = solve_problem(_data(examples, q_input), trace=True)
        assert ans == q_answer
        assert isinstance(log, list) and log
        kinds = {r["kind"] for r in log}
        assert "op" in kinds  # operator resolution always present
        query_records = [r for r in log if r["kind"] == "query"]
        assert query_records and query_records[-1]["answer"] == q_answer


def test_backward_compatible_two_tuple_without_trace():
    _rule, examples, q_input, _q_answer = sample_solvable(1, 4)
    result = solve_problem(_data(examples, q_input))
    assert len(result) == 2  # (answer, (mapping, op_info)) -- drop-in for investigator
