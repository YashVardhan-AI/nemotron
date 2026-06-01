"""Reference generator: a brand-new substitution-cipher decryption problem."""

import random

from reasoners.cipher import _load_wonderland
from reasoners.store_types import Example, Problem
from val.generators import register

# Use the EXACT 77-word Wonderland vocabulary the real cipher problems draw from
# (reasoners/wonderland.txt). The real task is solvable at ~100% because ~47% of
# queries contain a cipher letter not shown in the 3-5 examples, so they can only
# be decoded by recognizing the plaintext word against this fixed vocab. An earlier
# 35-word list with 5 out-of-vocab words (gate/guards/opens/river/whispers) made
# those queries unsolvable -> a misleadingly hard ~78% (an OOD artifact).
_WORDS = _load_wonderland()

_PROMPT_HEADER = "In Alice's Wonderland, secret encryption rules are used on text."


def _alphabet(seed: int) -> str:
    """Plaintext-letter (a..z) -> ciphertext-letter mapping, as a 26-char string."""
    letters = list("abcdefghijklmnopqrstuvwxyz")
    random.Random(seed).shuffle(letters)
    return "".join(letters)


def rule_signature(seed: int) -> str:
    return _alphabet(seed)


def _encrypt(text: str, alphabet: str) -> str:
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(alphabet[ord(ch) - ord("a")])
        else:
            out.append(ch)
    return "".join(out)


def _sentence(rng: random.Random) -> str:
    n = rng.randint(3, 5)
    return " ".join(rng.choice(_WORDS) for _ in range(n))


def generate(seed: int, difficulty: int) -> Problem:
    rng = random.Random(seed)
    alphabet = _alphabet(seed)

    plain_examples = [_sentence(rng) for _ in range(difficulty)]
    plain_query = _sentence(rng)

    example_lines = []
    examples: list[Example] = []
    for plain in plain_examples:
        cipher = _encrypt(plain, alphabet)
        example_lines.append(f"{cipher} -> {plain}")
        examples.append(Example(input_value=cipher, output_value=plain))

    cipher_query = _encrypt(plain_query, alphabet)
    prompt = (
        f"{_PROMPT_HEADER}\nHere are some examples:\n"
        + "\n".join(example_lines)
        + f"\nNow, decrypt the following text: {cipher_query}"
    )

    return Problem(
        id=f"val-cipher-{seed}",
        category="cipher",
        examples=examples,
        question=cipher_query,
        answer=plain_query,
        prompt=prompt,
    )


register("cipher", generate, rule_signature)
