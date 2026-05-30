"""Second generator: brand-new 8-bit transformation rules.

Rule families (all deterministic, answer correct by construction):
  affine : out = (a*x + b) % 256        (a odd)
  xor    : out = x ^ mask
  rotl   : out = rotate-left(x, k)       (8-bit)
  perm   : out bit j = in bit perm[j]    (MSB-first indexing)
"""

import random
from collections.abc import Callable

from reasoners.store_types import Example, Problem
from val.generators import register

_HEADER = (
    "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit "
    "binary numbers. The transformation involves operations like bit shifts, "
    "rotations, XOR, AND, OR, NOT, and possibly majority or choice functions."
)

_RULE_TYPES = ["affine", "xor", "rotl", "perm"]


def build_rule(seed: int) -> tuple[str, Callable[[int], int]]:
    """Return (canonical_signature, apply) for the rule selected by *seed*."""
    rng = random.Random(seed)
    rule_type = rng.choice(_RULE_TYPES)

    if rule_type == "affine":
        a = rng.choice([3, 5, 7, 9, 11, 13, 15])
        b = rng.randint(1, 255)
        signature = f"affine:a={a},b={b}"

        def apply(x: int) -> int:
            return (a * x + b) & 0xFF

    elif rule_type == "xor":
        mask = rng.randint(1, 255)
        signature = f"xor:{mask}"

        def apply(x: int) -> int:
            return x ^ mask

    elif rule_type == "rotl":
        k = rng.randint(1, 7)
        signature = f"rotl:{k}"

        def apply(x: int) -> int:
            return ((x << k) | (x >> (8 - k))) & 0xFF

    else:  # perm
        perm = list(range(8))
        rng.shuffle(perm)
        signature = "perm:" + ",".join(str(i) for i in perm)

        def apply(x: int) -> int:
            in_bits = [(x >> (7 - i)) & 1 for i in range(8)]
            out_bits = [in_bits[perm[j]] for j in range(8)]
            out = 0
            for j in range(8):
                out |= out_bits[j] << (7 - j)
            return out

    return signature, apply


def rule_signature(seed: int) -> str:
    return build_rule(seed)[0]


def _distinct_inputs(seed: int, count: int) -> list[int]:
    if count > 256:
        raise ValueError(f"count={count} exceeds the 8-bit value pool (256)")
    rng = random.Random(seed * 1000 + 1)  # separate stream from the rule
    seen: list[int] = []
    while len(seen) < count:
        v = rng.randint(0, 255)
        if v not in seen:
            seen.append(v)
    return seen


def generate(seed: int, difficulty: int) -> Problem:
    _signature, apply = build_rule(seed)
    values = _distinct_inputs(seed, difficulty + 1)
    example_ints, query_int = values[:difficulty], values[difficulty]

    example_lines: list[str] = []
    examples: list[Example] = []
    for x in example_ints:
        x_bits = format(x, "08b")
        y_bits = format(apply(x), "08b")
        example_lines.append(f"{x_bits} -> {y_bits}")
        examples.append(Example(input_value=x_bits, output_value=y_bits))

    query_bits = format(query_int, "08b")
    answer_bits = format(apply(query_int), "08b")
    prompt = (
        f"{_HEADER}\n\nHere are some examples of input -> output:\n"
        + "\n".join(example_lines)
        + f"\n\nNow, determine the output for: {query_bits}"
    )

    return Problem(
        id=f"val-bit_manipulation-{seed}",
        category="bit_manipulation",
        examples=examples,
        question=query_bits,
        answer=answer_bits,
        prompt=prompt,
    )


register("bit_manipulation", generate, rule_signature)
