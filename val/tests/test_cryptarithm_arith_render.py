"""Soundness of the reversed-digit, carry-annotated arithmetic renderer.

The rendered text is for the CoT; the gate is that the renderer's returned VALUE
always equals Python's, and that the text shows per-digit (not atomic) work.
"""

from reasoners.cryptarithm_arith_render import long_arith


def test_value_matches_python_for_all_ops():
    for a in range(0, 100, 7):
        for b in range(0, 100, 5):
            assert long_arith("add", a, b)[0] == a + b, (a, b)
            assert long_arith("abs_diff", a, b)[0] == abs(a - b), (a, b)
            assert long_arith("mul", a, b)[0] == a * b, (a, b)


def test_text_is_per_digit_not_atomic():
    # The product 36*47 must NOT appear as the atomic string "36 * 47 = 1692".
    _val, text = long_arith("mul", 36, 47)
    assert "1692" in text  # the final value is stated
    assert "36 * 47 = 1692" not in text  # but not as a single atomic step
    assert "carry" in text.lower()  # carries are written out


def test_add_shows_reversed_columns_with_carry():
    _val, text = long_arith("add", 21, 8)
    low = text.lower()
    assert "29" in text
    assert "carry" in low
