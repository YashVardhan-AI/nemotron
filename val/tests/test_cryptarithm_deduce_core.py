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
    search_with_log,
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


def _arith_only(examples):
    """The arithmetic demos (the search operates on these, like _solve)."""
    from reasoners.cryptarithm_deduce_core import _is_concat

    out = []
    for i, o in examples:
        ex = (i[0], i[1], i[2], i[3], i[4], tuple(o))
        if not _is_concat(ex):
            out.append(ex)
    return out


def _reencode_query(found_map, op_info, q):
    """Apply op_info[q[2]] to the query under found_map; return the answer string
    or None if a needed digit has no glyph."""
    from reasoners.cryptarithm_deduce_core import _OP_NAMES, _OPS, num_to_digits

    left = 10 * found_map[q[0]] + found_map[q[1]]
    right = 10 * found_map[q[3]] + found_map[q[4]]
    idx = _OP_NAMES.index(op_info[q[2]])
    val = _OPS[idx](left, right)
    if idx >= 3:
        rd = (val // 1000, (val // 100) % 10, (val // 10) % 10, val % 10)
    else:
        rd = num_to_digits(val)
    d2s = {d: s for s, d in found_map.items()}
    if any(d not in d2s for d in rd):
        return None
    return "".join(d2s[d] for d in rd)


def test_search_recovers_a_sound_map_reproducing_the_answer():
    # The narrated search must reach an INJECTIVE map that reproduces the planted
    # query answer (soundness). It need not equal solve_problem's arbitrary first
    # map -- the renderer guards strict equality separately.
    checked = 0
    for seed in range(40):
        _r, examples, q_input, q_answer = sample_solvable(seed, 4)
        data = _data(examples, q_input)
        _ans, (mapping, op_info) = solve_problem(data)
        if not mapping:
            continue  # concat-shortcut query: no digit search to narrate
        q = tuple(q_input)
        result = search_with_log(
            _arith_only(examples), q, op_info, planted_answer=q_answer
        )
        assert result is not None, seed
        found_map, _log = result
        assert len(set(found_map.values())) == len(found_map), seed  # injective
        assert _reencode_query(found_map, op_info, q) == q_answer, seed
        checked += 1
    assert checked > 0


def test_search_usually_matches_the_authoritative_map():
    # When the map is uniquely pinned (the common case for "well" determinacy),
    # the search should find the SAME map solve_problem did. Require a strong
    # majority so the renderer rarely has to skip a seed.
    agree = total = 0
    for seed in range(40):
        _r, examples, q_input, q_answer = sample_solvable(seed, 4)
        data = _data(examples, q_input)
        _ans, (mapping, op_info) = solve_problem(data)
        if not mapping:
            continue
        q = tuple(q_input)
        result = search_with_log(
            _arith_only(examples), q, op_info, planted_answer=q_answer
        )
        if result is None:
            total += 1
            continue
        agree += int(result[0] == mapping)
        total += 1
    assert total > 0
    assert agree >= 0.6 * total, (agree, total)


def test_search_log_is_bounded_and_well_formed():
    for seed in range(40):
        _r, examples, q_input, q_answer = sample_solvable(seed, 4)
        data = _data(examples, q_input)
        _ans, (mapping, op_info) = solve_problem(data)
        if not mapping:
            continue
        q = tuple(q_input)
        result = search_with_log(
            _arith_only(examples), q, op_info, planted_answer=q_answer
        )
        if result is None:
            continue
        _m, log = result
        assert isinstance(log, list) and log
        assert len(log) <= 200, (seed, len(log))
        kinds = {r["kind"] for r in log}
        assert kinds <= {"assign", "forced", "reject", "backtrack", "solution"}
        assert log[-1]["kind"] == "solution"


def test_search_emits_backtracking_somewhere_in_the_sweep():
    saw_backtrack = False
    for seed in range(40):
        _r, examples, q_input, q_answer = sample_solvable(seed, 4)
        data = _data(examples, q_input)
        _ans, (mapping, op_info) = solve_problem(data)
        if not mapping:
            continue
        q = tuple(q_input)
        result = search_with_log(
            _arith_only(examples), q, op_info, planted_answer=q_answer
        )
        if result is None:
            continue
        if any(r["kind"] == "backtrack" for r in result[1]):
            saw_backtrack = True
    assert saw_backtrack  # genuine search, not a straight-line read-off
