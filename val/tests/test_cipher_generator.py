from val.generators import GENERATORS
from val.generators.cipher import generate, query_needs_vocab, rule_signature


def test_registered():
    assert "cipher" in GENERATORS


def test_deterministic_given_seed():
    a = generate(seed=7, difficulty=3)
    b = generate(seed=7, difficulty=3)
    assert a.prompt == b.prompt
    assert a.answer == b.answer


def test_different_seeds_differ():
    assert (
        generate(seed=1, difficulty=3).prompt != generate(seed=2, difficulty=3).prompt
    )


def test_examples_count_matches_difficulty():
    p = generate(seed=7, difficulty=4)
    assert len(p.examples) == 4
    assert p.category == "cipher"


def test_answer_is_recoverable_from_rule():
    # The query ciphertext decrypts (per the examples' substitution) to the answer.
    p = generate(seed=11, difficulty=5)
    sig = rule_signature(seed=11)  # plaintext-letter -> ciphertext-letter alphabet
    decrypt = {sig[i]: chr(ord("a") + i) for i in range(26)}
    recovered = "".join(decrypt.get(ch, ch) for ch in p.question)
    assert recovered == p.answer


def test_prompt_uses_real_template_markers():
    p = generate(seed=7, difficulty=3)
    assert p.prompt.startswith(
        "In Alice's Wonderland, secret encryption rules are used on text."
    )
    assert "Now, decrypt the following text:" in p.prompt
    assert " -> " in p.prompt


def test_needs_vocab_is_deterministic():
    p = generate(seed=7, difficulty=4)
    assert query_needs_vocab(p) == query_needs_vocab(p)


def test_needs_vocab_rate_matches_real_distribution():
    # Real cipher: ~62% of queries need vocab (a letter unseen in the examples).
    # At the real difficulty (3-5 examples) the synthetic rate must land in the
    # same regime, else the eval is easier/harder than real on the load-bearing
    # skill. Measured ~59% in the audit -> assert a generous band around it.
    ps = [generate(s, difficulty=4) for s in range(400)]
    rate = sum(query_needs_vocab(p) for p in ps) / len(ps)
    assert 0.45 < rate < 0.75, rate


def test_needs_vocab_definition():
    # If every query letter is shown in some example ciphertext -> not needs-vocab.
    from reasoners.store_types import Example, Problem

    seen_only = Problem(
        id="x",
        category="cipher",
        examples=[Example(input_value="abc", output_value="xyz")],
        question="ab",
        answer="xy",
        prompt="",
    )
    assert query_needs_vocab(seen_only) is False
    missing = Problem(
        id="y",
        category="cipher",
        examples=[Example(input_value="abc", output_value="xyz")],
        question="abz",  # 'z' never appears in any example ciphertext
        answer="xy?",
        prompt="",
    )
    assert query_needs_vocab(missing) is True
