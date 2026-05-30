"""Reference generator: a brand-new substitution-cipher decryption problem."""

import random

from reasoners.store_types import Example, Problem
from val.generators import register

# Themed vocabulary mirroring the real cipher problems' Wonderland distribution.
_WORDS = [
    "queen",
    "dragon",
    "castle",
    "secret",
    "near",
    "valley",
    "discovers",
    "dreams",
    "inside",
    "student",
    "creates",
    "magical",
    "door",
    "golden",
    "follows",
    "princess",
    "reads",
    "mysterious",
    "cat",
    "imagines",
    "book",
    "wizard",
    "the",
    "guards",
    "hidden",
    "garden",
    "river",
    "mountain",
    "whispers",
    "ancient",
    "key",
    "opens",
    "silver",
    "gate",
    "forest",
]

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
