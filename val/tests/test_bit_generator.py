import pytest

from val.generators import GENERATORS
from val.generators.bit_manipulation import (
    build_rule,
    columns,
    generate,
    rule_family,
    rule_signature,
)


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


def test_generate_rejects_impossible_difficulty():
    # 8-bit pool has only 256 values; difficulty+1 > 256 must raise, not hang.
    with pytest.raises(ValueError):
        generate(seed=1, difficulty=256)


def test_rule_families_match_real_grammar():
    # Only the real-distribution families are generated (no affine/xor-mask/perm),
    # and pairwise 2-input boolean is the plurality (~65%).
    fams = [rule_family(s) for s in range(400)]
    assert set(fams) <= {"pairwise", "rot", "complex"}
    counts = {f: fams.count(f) for f in set(fams)}
    assert counts["pairwise"] == max(counts.values())
    assert counts["pairwise"] / len(fams) > 0.5


# --- recalibration properties (per-column heterogeneity, matching real) -------


def _pair_ops_in(cols):
    return [c[1] for c in cols if c[0] == "pair"]


def _pair_operands_in(cols):
    return [(c[2], c[3]) for c in cols if c[0] == "pair"]


def test_pairwise_problems_mix_multiple_ops():
    # Real: 42.6% of problems mix >=2 distinct binary ops across the 8 bits.
    # The recalibrated generator must clear that (draws ops independently/column).
    seeds = [s for s in range(600) if rule_family(s) == "pairwise"]
    mixed = sum(1 for s in seeds if len(set(_pair_ops_in(columns(s)))) >= 2)
    assert mixed / len(seeds) > 0.4


def test_pairwise_operands_vary_by_column():
    # Real: 48.2% have operand offsets that vary by column (not one global shift).
    seeds = [s for s in range(600) if rule_family(s) == "pairwise"]
    varied = sum(1 for s in seeds if len(set(_pair_operands_in(columns(s)))) >= 2)
    assert varied / len(seeds) > 0.4


def test_complex_problems_have_a_three_input_column():
    seeds = [s for s in range(600) if rule_family(s) == "complex"]
    assert seeds, "expected some complex-family seeds in range"
    for s in seeds:
        assert any(c[0] in ("maj", "choice", "tt3") for c in columns(s))


def test_complex_family_includes_arbitrary_truth_tables():
    # The complex tail must include arbitrary 3-var truth tables (TT3), not only
    # the named MAJ/CHOICE ops — real data has both.
    seeds = [s for s in range(600) if rule_family(s) == "complex"]
    assert any(any(c[0] == "tt3" for c in columns(s)) for s in seeds)


def test_routing_includes_arbitrary_permutation():
    # The routing family is mostly constant-k rotation, but ~10% are arbitrary
    # permutations (matching the 2/20 observed in real routing problems).
    rot_seeds = [s for s in range(800) if rule_family(s) == "rot"]

    def is_rotation(cols):
        sources = [c[1] for c in cols]  # all "route" columns
        diffs = {(sources[j] - j) % 8 for j in range(8)}
        return len(diffs) == 1  # constant offset => rotation

    assert any(not is_rotation(columns(s)) for s in rot_seeds)


def test_three_input_columns_genuinely_use_three_inputs():
    # A complex column's output must be able to change when only the THIRD
    # operand bit flips — i.e. it really depends on 3 inputs, not 2.
    from val.generators.bit_manipulation import _eval_column

    seed = next(
        s for s in range(600) if any(c[0] in ("maj", "choice") for c in columns(s))
    )
    col = next(c for c in columns(seed) if c[0] in ("maj", "choice"))
    _kind, p, q, r = col
    # Build two inputs differing only at bit r; find a case where output differs.
    differs = False
    for x in range(256):
        if (x >> (7 - r)) & 1:
            continue
        y = x | (1 << (7 - r))
        if _eval_column(col, x) != _eval_column(col, y):
            differs = True
            break
    assert differs
