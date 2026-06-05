from val.grading import (
    extract_final_answer,
    extract_final_answer_braceaware,
    verify,
    verify_strict,
)


def test_extract_boxed():
    assert extract_final_answer(r"The answer is \boxed{42}") == "42"


def test_extract_last_nonempty_boxed():
    assert extract_final_answer(r"\boxed{} then \boxed{7}") == "7"


def test_extract_final_answer_phrase():
    assert extract_final_answer("The final answer is: 3.14") == "3.14"


def test_extract_trailing_number():
    assert extract_final_answer("Just a number 100 in text") == "100"


def test_extract_none():
    assert extract_final_answer(None) == "NOT_FOUND"


def test_verify_numeric_tolerance():
    assert verify("24.64", "24.6401") is True


def test_verify_case_insensitive_text():
    assert verify("XLVII", "xlvii") is True


def test_verify_binary_is_float_lenient():
    # Real-grader gotcha: binary strings go through float(), so two DIFFERENT
    # binary strings within 1% compare EQUAL. This pins that behavior.
    assert verify("11011011", "11011010") is True


def test_verify_strict_binary_is_exact():
    assert verify_strict("11011011", "11011010") is False
    assert verify_strict("11011011", "11011011") is True


def test_braceaware_recovers_answer_with_closing_brace():
    # The verbatim grader truncates at the first `}`; the diagnostic recovers it.
    text = r"so the result is \boxed{a}b}"
    assert extract_final_answer(text) == "a"  # the leaderboard cap
    assert extract_final_answer_braceaware(text) == "a}b"  # true answer


def test_braceaware_matches_verbatim_on_normal_answers():
    assert extract_final_answer_braceaware(r"x = \boxed{42}") == "42"


def test_braceaware_takes_last_boxed():
    assert extract_final_answer_braceaware(r"\boxed{1} then \boxed{x}y}") == "x}y"


def test_braceaware_falls_back_without_boxed():
    assert extract_final_answer_braceaware("The final answer is: 7") == "7"
    assert extract_final_answer_braceaware(None) == "NOT_FOUND"
