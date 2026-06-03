"""Tests for the shared cryptarithm rule core + val generator.

Phase 1 (rule core) tests live here; the Phase 2 val-generator tests are appended
in the same file. Both exercise the same shared `sample_problem` source of truth.
"""

from reasoners.cryptarithm_rule import (
    BASE_OPERATORS,
    OP_NAMES,
    WRAPPERS,
    build_rule,
    num_to_digits,
    render_prompt,
    sample_problem,
)


def test_build_rule_is_deterministic():
    a, b = build_rule(7), build_rule(7)
    assert a.digit_to_sym == b.digit_to_sym
    assert a.op_of == b.op_of


def test_digit_sym_is_injective_over_ten_digits():
    rule = build_rule(3)
    assert set(rule.digit_to_sym) == set(range(10))
    assert len(set(rule.digit_to_sym.values())) == 10  # all-different glyphs
    for d, s in rule.digit_to_sym.items():
        assert rule.sym_to_digit[s] == d


def test_every_operator_glyph_assigned_one_of_five_ops():
    for seed in range(20):
        rule = build_rule(seed)
        assert set(BASE_OPERATORS) <= set(rule.op_of)  # +,-,* always candidates
        assert all(name in OP_NAMES for name in rule.op_of.values())
    assert set(OP_NAMES) == {"add", "abs_diff", "mul", "concat", "rev_concat"}


def test_num_to_digits():
    assert num_to_digits(0) == (0,)
    assert num_to_digits(25) == (2, 5)
    assert num_to_digits(137) == (1, 3, 7)


def test_concat_result_is_four_digits():
    rule = build_rule(5)
    rule.op_of["+"] = "concat"  # force concat to check padding: 05 concat 12 -> 0512
    assert rule.result_digits("+", 5, 12) == (0, 5, 1, 2)


def test_arith_round_trips_through_encoding():
    rule = build_rule(11)
    rule.op_of["+"] = "add"
    inp, out = rule.encode_example("+", (1, 2), (1, 3))  # 12 + 13 = 25
    assert tuple(rule.sym_to_digit[c] for c in out) == (2, 5)
    assert inp[2] == "+" and len(inp) == 5


def test_some_rules_use_a_tail_operator_glyph():
    # distribution match: operators are not always +,-,* (long tail exists)
    assert any(set(build_rule(s).op_of) - set(BASE_OPERATORS) for s in range(80))


def _witnessed_digit_glyphs(examples):
    w = set()
    for inp, out in examples:
        w |= {inp[0], inp[1], inp[3], inp[4]} | set(out)
    return w


def test_sample_problem_well_determined_and_self_consistent():
    for seed in range(30):
        rule, examples, q_input, q_answer = sample_problem(seed, 4)
        assert len(examples) == 4  # exact difficulty, no appended cover demos
        # query operator is witnessed in a demo
        assert any(i[2] == q_input[2] for i, _ in examples)
        # every digit-glyph needed to read the query AND its answer is witnessed
        needed = {q_input[0], q_input[1], q_input[3], q_input[4]} | set(q_answer)
        assert needed <= _witnessed_digit_glyphs(examples)
        # the answer reproduces under the rule (self-consistency)
        left = (rule.sym_to_digit[q_input[0]], rule.sym_to_digit[q_input[1]])
        right = (rule.sym_to_digit[q_input[3]], rule.sym_to_digit[q_input[4]])
        _, ans = rule.encode_example(q_input[2], left, right)
        assert ans == q_answer


def test_difficulty_controls_demo_count():
    _, ex3, _, _ = sample_problem(1, 3)
    _, ex5, _, _ = sample_problem(1, 5)
    assert len(ex3) == 3 and len(ex5) == 5


def test_sample_problem_is_deterministic():
    a = sample_problem(9, 4)
    b = sample_problem(9, 4)
    assert (a[1], a[2], a[3]) == (b[1], b[2], b[3])


def test_render_prompt_default_is_exact_real_wrapper():
    _, examples, q_input, _ = sample_problem(0, 4)
    prompt = render_prompt(examples, q_input, 0)
    assert prompt.startswith(
        "In Alice's Wonderland, a secret set of transformation rules"
    )
    assert "Now, determine the result for:" in prompt
    assert prompt.rstrip().endswith(q_input)
    for i, o in examples:
        assert f"{i} = {o}" in prompt


def test_render_prompt_paraphrases_preserve_demos_and_query_structure():
    _, examples, q_input, _ = sample_problem(2, 4)
    assert len(WRAPPERS) >= 2  # at least one paraphrase exists
    for w in range(len(WRAPPERS)):
        prompt = render_prompt(examples, q_input, w)
        for i, o in examples:
            assert f"{i} = {o}" in prompt  # demo structure untouched
        assert prompt.rstrip().endswith(q_input)
