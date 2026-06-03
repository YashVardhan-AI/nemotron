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


def test_near_determinacy_is_self_consistent_and_exact_count():
    for seed in range(40):
        rule, examples, q_input, q_answer = sample_problem(seed, 4, determinacy="near")
        assert len(examples) == 4
        needed = {q_input[0], q_input[1], q_input[3], q_input[4]} | set(q_answer)
        assert needed <= _witnessed_digit_glyphs(examples)
        left = (rule.sym_to_digit[q_input[0]], rule.sym_to_digit[q_input[1]])
        right = (rule.sym_to_digit[q_input[3]], rule.sym_to_digit[q_input[4]])
        _, ans = rule.encode_example(q_input[2], left, right)
        assert ans == q_answer


_EASY_OPS = {"add", "concat", "rev_concat"}
_HARD_OPS = {"mul", "abs_diff"}


def test_easy_profile_biases_query_to_easy_ops_and_small_operands():
    saw_easy = False
    for seed in range(120):
        rule = build_rule(seed)
        if not any(name in _EASY_OPS for name in rule.op_of.values()):
            continue  # rule has no easy op -> sampler falls back, skip
        _, _, q_input, _ = sample_problem(seed, 4, profile="easy")
        assert rule.op_of[q_input[2]] in _EASY_OPS  # query op is always easy
        digits = [rule.sym_to_digit[q_input[i]] for i in (0, 1, 3, 4)]
        assert max(digits) <= 5  # easy operand range
        saw_easy = True
    assert saw_easy  # the sweep actually exercised the easy branch


def test_hard_profile_biases_query_to_hard_ops():
    saw_hard = False
    for seed in range(120):
        rule = build_rule(seed)
        if not any(name in _HARD_OPS for name in rule.op_of.values()):
            continue
        _, _, q_input, _ = sample_problem(seed, 4, profile="hard")
        assert rule.op_of[q_input[2]] in _HARD_OPS
        saw_hard = True
    assert saw_hard


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


# --- Phase 2: val generator -------------------------------------------------

from reasoners.store_types import Problem  # noqa: E402
from val.generators.cryptarithm import generate, rule_signature  # noqa: E402


def _check_problem_is_self_consistent(p: Problem):
    seed = int(p.id.rsplit("-", 1)[1])
    rule = build_rule(seed)
    for ex in p.examples:  # every demo reproduces under the rule
        iv = ex.input_value
        left = (rule.sym_to_digit[iv[0]], rule.sym_to_digit[iv[1]])
        right = (rule.sym_to_digit[iv[3]], rule.sym_to_digit[iv[4]])
        _, out = rule.encode_example(iv[2], left, right)
        assert out == ex.output_value
    q = p.question  # the query answer reproduces under the rule
    left = (rule.sym_to_digit[q[0]], rule.sym_to_digit[q[1]])
    right = (rule.sym_to_digit[q[3]], rule.sym_to_digit[q[4]])
    _, ans = rule.encode_example(q[2], left, right)
    assert ans == p.answer


def test_generate_shape_and_metadata():
    p = generate(0, 4)
    assert p.category == "cryptarithm_deduce"
    assert p.id == "val-cryptarithm-0"
    assert len(p.examples) == 4
    assert len(p.question) == 5
    rule = build_rule(0)
    assert p.question[2] in rule.operators  # operator glyph (may be a tail glyph)
    assert p.prompt.startswith(
        "In Alice's Wonderland, a secret set of transformation rules"
    )
    assert "Now, determine the result for:" in p.prompt


def test_generate_is_self_consistent_and_well_determined():
    for seed in range(25):
        p = generate(seed, 4)
        _check_problem_is_self_consistent(p)
        assert any(ex.input_value[2] == p.question[2] for ex in p.examples)
        demo_glyphs = set()
        for ex in p.examples:
            iv = ex.input_value
            demo_glyphs |= {iv[0], iv[1], iv[3], iv[4]} | set(ex.output_value)
        needed = {p.question[0], p.question[1], p.question[3], p.question[4]} | set(
            p.answer
        )
        assert needed <= demo_glyphs


def test_rule_signature_is_stable_and_distinct():
    assert rule_signature(5) == rule_signature(5)
    assert rule_signature(5) != rule_signature(6)


def test_generate_difficulty_controls_demo_count():
    assert len(generate(1, 3).examples) == 3
    assert len(generate(1, 5).examples) == 5
