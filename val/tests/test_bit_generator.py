from val.generators import GENERATORS
from val.generators.bit_manipulation import build_rule, generate, rule_signature


def test_registered():
    assert "bit_manipulation" in GENERATORS


def test_deterministic_given_seed():
    a = generate(seed=3, difficulty=8)
    b = generate(seed=3, difficulty=8)
    assert a.prompt == b.prompt and a.answer == b.answer


def test_different_seeds_differ():
    # Across a small range, at least one pair of rules differs.
    sigs = {rule_signature(s) for s in range(8)}
    assert len(sigs) > 1


def test_examples_count_and_category():
    p = generate(seed=3, difficulty=10)
    assert len(p.examples) == 10
    assert p.category == "bit_manipulation"


def test_inputs_and_outputs_are_8bit_binary():
    p = generate(seed=5, difficulty=8)
    for ex in p.examples:
        assert len(ex.input_value) == 8 and set(ex.input_value) <= {"0", "1"}
        assert len(ex.output_value) == 8 and set(ex.output_value) <= {"0", "1"}
    assert len(p.answer) == 8 and set(p.answer) <= {"0", "1"}


def test_answer_recoverable_from_rule():
    p = generate(seed=5, difficulty=8)
    _sig, apply = build_rule(seed=5)
    expected = format(apply(int(p.question, 2)), "08b")
    assert expected == p.answer


def test_examples_satisfy_the_rule():
    p = generate(seed=9, difficulty=8)
    _sig, apply = build_rule(seed=9)
    for ex in p.examples:
        assert format(apply(int(ex.input_value, 2)), "08b") == ex.output_value


def test_prompt_uses_real_template_markers():
    p = generate(seed=3, difficulty=8)
    assert p.prompt.startswith(
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit"
    )
    assert "Here are some examples of input -> output:" in p.prompt
    assert "Now, determine the output for:" in p.prompt
