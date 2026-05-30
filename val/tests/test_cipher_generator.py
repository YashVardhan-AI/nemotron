from val.generators import GENERATORS
from val.generators.cipher import generate, rule_signature


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
