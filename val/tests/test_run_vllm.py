from reasoners.store_types import Problem
from val.run_vllm import cap_per_category


def _p(pid: str, category: str) -> Problem:
    return Problem(
        id=pid,
        category=category,
        examples=[],
        question="q",
        answer="a",
        prompt="p",
    )


def test_cap_none_returns_input_unchanged():
    ps = [_p("1", "a"), _p("2", "b")]
    assert cap_per_category(ps, None) is ps


def test_cap_above_length_returns_input_unchanged():
    ps = [_p("1", "a")]
    assert cap_per_category(ps, 5) is ps


def test_cap_is_round_robin_balanced_across_categories():
    ps = [_p(f"a{i}", "alpha") for i in range(5)]
    ps += [_p(f"b{i}", "beta") for i in range(5)]
    out = cap_per_category(ps, 4)
    cats = [p.category for p in out]
    assert len(out) == 4
    assert cats.count("alpha") == 2  # round-robin -> even split
    assert cats.count("beta") == 2


def test_cap_preserves_original_order_within_category():
    ps = [_p(f"a{i}", "alpha") for i in range(5)]
    out = cap_per_category(ps, 3)
    assert [p.id for p in out] == ["a0", "a1", "a2"]


def test_cap_drains_remaining_when_one_category_is_small():
    ps = [_p("a0", "alpha"), _p("b0", "beta"), _p("b1", "beta"), _p("b2", "beta")]
    out = cap_per_category(ps, 3)
    # alpha exhausts after 1; round-robin keeps pulling from beta to reach 3.
    assert len(out) == 3
    assert [p.id for p in out] == ["a0", "b0", "b1"]
