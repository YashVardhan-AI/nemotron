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


def _arith_seeds():
    """(seed, examples, q_input, q_answer, mapping, op_info) for arith-query seeds."""
    out = []
    for seed in range(40):
        _r, examples, q_input, q_answer = sample_solvable(seed, 4)
        data = _data(examples, q_input)
        _ans, (mapping, op_info) = solve_problem(data)
        if mapping:
            out.append((seed, examples, q_input, q_answer, mapping, op_info))
    return out


def test_search_recovers_a_sound_map_reproducing_the_answer():
    checked = 0
    for seed, examples, q_input, q_answer, mapping, op_info in _arith_seeds():
        q = tuple(q_input)
        result = search_with_log(_arith_only(examples), q, op_info, mapping)
        assert result is not None, seed
        found_map, _log = result
        assert len(set(found_map.values())) == len(found_map), seed  # injective
        assert _reencode_query(found_map, op_info, q) == q_answer, seed
        checked += 1
    assert checked > 0


def test_search_log_is_bounded_and_well_formed():
    for seed, examples, q_input, _q_answer, mapping, op_info in _arith_seeds():
        q = tuple(q_input)
        result = search_with_log(_arith_only(examples), q, op_info, mapping)
        assert result is not None
        _m, log = result
        assert isinstance(log, list) and log
        assert len(log) <= 80, (seed, len(log))  # winning-path is SHORT
        kinds = {r["kind"] for r in log}
        assert kinds <= {"assign", "forced", "reject", "backtrack", "solution"}
        assert log[-1]["kind"] == "solution"


def _replay_to_state(log):
    """Replay the log; assert the mapping stays injective at EVERY step. Return it."""
    mapping: dict[str, int] = {}
    for rec in log:
        k = rec["kind"]
        if k in ("assign", "forced"):
            g, d = rec["glyph"], rec["digit"]
            assert d not in mapping.values(), (k, g, d, dict(mapping))
            assert g not in mapping, (k, g, dict(mapping))
            mapping[g] = d
        elif k == "backtrack":
            mapping.pop(rec["glyph"], None)
    return mapping


def test_search_log_replays_to_injective_states_and_matches_solution():
    checked = 0
    for seed, examples, q_input, _q_answer, mapping, op_info in _arith_seeds():
        q = tuple(q_input)
        result = search_with_log(_arith_only(examples), q, op_info, mapping)
        assert result is not None
        found_map, log = result
        replayed = _replay_to_state(log)
        solution = next(r for r in log if r["kind"] == "solution")
        assert replayed == solution["mapping"], seed
        assert replayed == found_map, seed
        checked += 1
    assert checked > 0


def test_search_log_shows_forced_deduction_and_ruling_out():
    saw_forced = saw_ruling = False
    for seed, examples, q_input, _q_answer, mapping, op_info in _arith_seeds():
        q = tuple(q_input)
        result = search_with_log(_arith_only(examples), q, op_info, mapping)
        assert result is not None
        log = result[1]
        if any(r["kind"] == "forced" for r in log):
            saw_forced = True
        if any(r["kind"] in ("reject", "backtrack") for r in log):
            saw_ruling = True
    assert saw_forced  # genuine deduction: glyphs forced by demos
    assert saw_ruling  # alternatives ruled out somewhere (light backtrack)
