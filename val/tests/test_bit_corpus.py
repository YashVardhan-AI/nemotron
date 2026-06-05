"""Tokenizer-free corpus-row generation for the bit_manipulation lever.

Tests build_bit_examples (the pure, no-tokenizer half of the corpus integration).
The tokenizing build_bit_rows + corpus.main() need the HF tokenizer and are
user-run; this guards the row content: distinct, sound, weak-tier-only, and no
validation rule leaked.
"""

from corpus import BIT_SEED_OFFSET, BIT_TARGET_TAGS, build_bit_examples
from reasoners.bit_rule import build_rule, problem_tag, rule_signature


def test_examples_are_distinct_sound_and_boxed():
    ex = build_bit_examples(40, holdout=set())
    assert len(ex) == 40
    pids = [e[0] for e in ex]
    prompts = [e[1] for e in ex]
    assert len(set(pids)) == 40  # distinct ids
    assert len(set(prompts)) == 40  # no exact-copy duplication
    for pid, prompt, completion, answer in ex:
        assert completion.rstrip().endswith("\\boxed{" + answer + "}<|im_end|>")
        assert "</think>" in completion
        assert prompt.startswith("In Alice's Wonderland")
        assert pid.startswith("bit-")
        # the boxed answer is the rule's true output (sound by construction)
        seed = int(pid.rsplit("-", 1)[1])
        _sig, apply = build_rule(seed)
        assert format(apply(int(prompt.rsplit(": ", 1)[1], 2)), "08b") == answer


def test_only_weak_tier_families_are_generated():
    ex = build_bit_examples(60, holdout=set())
    for pid, *_ in ex:
        seed = int(pid.rsplit("-", 1)[1])
        assert problem_tag(seed) in BIT_TARGET_TAGS


def test_seeds_are_disjoint_from_val_range():
    ex = build_bit_examples(15, holdout=set())
    used_seeds = [int(pid.rsplit("-", 1)[1]) for pid, *_ in ex]
    assert min(used_seeds) >= BIT_SEED_OFFSET


def test_holdout_signatures_are_skipped():
    # reserve the signatures of the first few weak-tier seeds at the offset
    reserved = set()
    seed = BIT_SEED_OFFSET
    while len(reserved) < 3:
        if problem_tag(seed) in BIT_TARGET_TAGS:
            reserved.add(rule_signature(seed))
        seed += 1
    ex = build_bit_examples(20, holdout=reserved)
    assert len(ex) == 20  # still hits the count by skipping ahead
    for _pid, prompt, _c, _a in ex:
        # none of the produced rows use a reserved signature
        seed = int(_pid.rsplit("-", 1)[1])
        assert rule_signature(seed) not in reserved
