import pytest

from val.generators import GENERATORS
from val.generators.bit_manipulation import (
    build_rule,
    columns,
    generate,
    problem_tag,
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


# --- recalibration: an in-distribution MIXTURE of hom/het, not all-or-nothing --


def _pair_ops_in(cols):
    return [c[1] for c in cols if c[0] == "pair"]


def _is_global_stride_pairs(cols):
    if any(c[0] != "pair" for c in cols):
        return False
    a, b = cols[0][2], cols[0][3]
    return all(c[2] == (a + j) % 8 and c[3] == (b + j) % 8 for j, c in enumerate(cols))


def test_heterogeneity_is_a_calibrated_mixture():
    # Real: ~43% of problems mix >=2 ops across the 8 bits (NOT ~0%, NOT ~100%).
    # An all-heterogeneous generator over-corrected the absolute difficulty.
    seeds = [s for s in range(800) if rule_family(s) == "pairwise"]
    mixed = sum(1 for s in seeds if len(set(_pair_ops_in(columns(s)))) >= 2)
    rate = mixed / len(seeds)
    assert 0.30 < rate < 0.62, rate


def test_operand_offsets_vary_in_a_mixture():
    # Real: ~48% vary operand offset by column (not a single global stride).
    seeds = [s for s in range(800) if rule_family(s) == "pairwise"]
    non_stride = sum(1 for s in seeds if not _is_global_stride_pairs(columns(s)))
    rate = non_stride / len(seeds)
    assert 0.30 < rate < 0.62, rate


def test_homogeneous_pairwise_is_a_single_global_rule():
    # A 'hom' pairwise problem must be exactly one op at a global stride -- the
    # easy case; this is what keeps the absolute difficulty realistic.
    hom = [
        s
        for s in range(800)
        if problem_tag(s) == "pairwise/hom" and rule_family(s) == "pairwise"
    ]
    assert hom
    for s in hom[:50]:
        cols = columns(s)
        assert len(set(_pair_ops_in(cols))) == 1  # single op
        assert _is_global_stride_pairs(cols)  # global stride


def test_complex_problems_have_a_three_input_column():
    seeds = [s for s in range(800) if rule_family(s) == "complex"]
    assert seeds
    for s in seeds:
        assert any(c[0] in ("maj", "choice") for c in columns(s))


def test_no_arbitrary_truth_tables():
    # TT3 (arbitrary 3-var tables) is ~0.7% of real columns and breaks
    # determinability -- the recalibrated generator must not emit it.
    for s in range(800):
        assert all(c[0] != "tt3" for c in columns(s))


def test_routing_includes_arbitrary_permutation():
    rot_seeds = [s for s in range(800) if rule_family(s) == "rot"]

    def is_rotation(cols):
        diffs = {(c[1] - j) % 8 for j, c in enumerate(cols)}
        return len(diffs) == 1

    assert any(not is_rotation(columns(s)) for s in rot_seeds)


def test_three_input_columns_genuinely_use_three_inputs():
    from val.generators.bit_manipulation import _eval_column

    seed = next(
        s for s in range(800) if any(c[0] in ("maj", "choice") for c in columns(s))
    )
    col = next(c for c in columns(seed) if c[0] in ("maj", "choice"))
    _kind, p, q, r = col
    differs = False
    for x in range(256):
        if (x >> (7 - r)) & 1:
            continue
        y = x | (1 << (7 - r))
        if _eval_column(col, x) != _eval_column(col, y):
            differs = True
            break
    assert differs
