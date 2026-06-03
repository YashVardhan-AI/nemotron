"""Phase 5: tokenizer-free corpus-row generation for cryptarithm_deduce.

Tests build_cryptarithm_examples (the pure, no-tokenizer half of the corpus
integration). The tokenizing build_cryptarithm_rows + corpus.main() need the HF
tokenizer and are user-run; this guards the row content: distinct, correct, and
no validation rule leaked. (In val/tests/ with the rest of the repo's tests.)
"""

from corpus import build_cryptarithm_examples
from val.generators.cryptarithm import rule_signature


def test_examples_are_distinct_and_correctly_boxed():
    for style in ("deduce", "propagate"):
        ex = build_cryptarithm_examples(25, style, holdout=set())
        assert len(ex) == 25
        pids = [e[0] for e in ex]
        prompts = [e[1] for e in ex]
        assert len(set(pids)) == 25  # distinct ids
        assert len(set(prompts)) == 25  # no exact-copy duplication
        for pid, prompt, completion, answer in ex:
            assert completion.rstrip().endswith("\\boxed{" + answer + "}<|im_end|>")
            assert "</think>" in completion
            assert prompt.startswith("In Alice's Wonderland")
            assert pid.startswith(f"cryptarithm-{style}-")


def test_mixed_style_uses_both_renderers():
    ex = build_cryptarithm_examples(20, "mixed", holdout=set())
    # mixed picks the deduction renderer on even seeds, propagate on odd; both
    # styles have a distinct opening line.
    deduce = sum(c.startswith("I need to deduce") for _, _, c, _ in ex)
    propagate = sum(c.startswith("I'll work out") for _, _, c, _ in ex)
    assert deduce > 0 and propagate > 0


def test_holdout_signatures_are_skipped():
    off = 1_000_000  # CRYPT_SEED_OFFSET
    reserved = {rule_signature(off), rule_signature(off + 1), rule_signature(off + 2)}
    ex = build_cryptarithm_examples(20, "deduce", holdout=reserved)
    used_seeds = {int(pid.rsplit("-", 1)[1]) for pid, _, _, _ in ex}
    assert used_seeds.isdisjoint({off, off + 1, off + 2})
    assert len(ex) == 20  # still produces the full count by skipping ahead


def test_seeds_are_disjoint_from_val_range():
    # training seeds start at the high offset, never overlapping val's 0..N.
    ex = build_cryptarithm_examples(15, "deduce", holdout=set())
    used_seeds = [int(pid.rsplit("-", 1)[1]) for pid, _, _, _ in ex]
    assert min(used_seeds) >= 1_000_000
