"""Generator: new 8-bit rules drawn from the REAL bit_manipulation grammar.

Empirically (1,602 train problems, and the solver's 85% rule_found rate), real
bit rules are overwhelmingly **per-output-bit functions of a few input bits**:
  ~65%  pairwise 2-input boolean   out[j] = op(in[a+j], in[b+j]), op in
                                   {AND, OR, XOR, AND-NOT, OR-NOT, XOR-NOT}
  ~20%  rotation / shift (+NOT)    out[j] = in[(j+k) % 8]   (the routing bulk)
  ~15%  complex 3-input            majority / choice of three input bits
                                   (the hard tail the 2-input solver can't express)
Affine-mod-256, xor-mask and arbitrary permutations are ~0% of real data, so we
do NOT generate them (an earlier version did, which made the eval out-of-distribution).
All rules are stride-structured (offset + j) so they're well-determined from a
handful of demos, matching the solver's run-based problems. Answers are correct
by construction.
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

_PAIR_OPS = ("AND", "OR", "XOR", "AND-NOT", "OR-NOT", "XOR-NOT")


def _bit(x: int, p: int) -> int:
    return (x >> (7 - p)) & 1


def _pack(bits: list[int]) -> int:
    out = 0
    for j in range(8):
        out |= bits[j] << (7 - j)
    return out


def _pair_apply(op: str, a: int, b: int) -> Callable[[int], int]:
    base = op.split("-")[0]
    neg = op.endswith("-NOT")

    def apply(x: int) -> int:
        bits = []
        for j in range(8):
            u = _bit(x, (a + j) % 8)
            v = _bit(x, (b + j) % 8)
            if neg:
                v = 1 - v
            if base == "AND":
                r = u & v
            elif base == "OR":
                r = u | v
            else:  # XOR
                r = u ^ v
            bits.append(r)
        return _pack(bits)

    return apply


def _rot_apply(k: int, inv: bool) -> Callable[[int], int]:
    def apply(x: int) -> int:
        bits = [_bit(x, (j + k) % 8) for j in range(8)]
        if inv:
            bits = [1 - b for b in bits]
        return _pack(bits)

    return apply


def _complex_apply(kind: str, a: int, b: int, c: int) -> Callable[[int], int]:
    def apply(x: int) -> int:
        bits = []
        for j in range(8):
            p = _bit(x, (a + j) % 8)
            q = _bit(x, (b + j) % 8)
            s = _bit(x, (c + j) % 8)
            if kind == "MAJ":
                r = 1 if (p + q + s) >= 2 else 0
            else:  # CHOICE: p selects q else s
                r = q if p else s
            bits.append(r)
        return _pack(bits)

    return apply


def build_rule(seed: int) -> tuple[str, Callable[[int], int]]:
    """Return (canonical_signature, apply) for the rule selected by *seed*.

    Family weights match the real distribution: ~65% pairwise, ~20% rotation,
    ~15% complex (3-input majority/choice).
    """
    rng = random.Random(seed)
    roll = rng.random()
    if roll < 0.65:
        op = rng.choice(_PAIR_OPS)
        a = rng.randint(0, 7)
        b = (a + rng.randint(1, 7)) % 8  # b != a
        return f"pairwise:{op},a={a},b={b}", _pair_apply(op, a, b)
    if roll < 0.85:
        k = rng.randint(1, 7)
        inv = rng.random() < 0.3
        return f"rot:k={k},inv={int(inv)}", _rot_apply(k, inv)
    kind = rng.choice(("MAJ", "CHOICE"))
    a = rng.randint(0, 7)
    b = (a + rng.randint(1, 7)) % 8
    c = (a + rng.randint(1, 7)) % 8
    return f"complex:{kind},a={a},b={b},c={c}", _complex_apply(kind, a, b, c)


def rule_signature(seed: int) -> str:
    return build_rule(seed)[0]


def rule_family(seed: int) -> str:
    """Coarse family label ('pairwise' / 'rot' / 'complex') for breakdowns."""
    return rule_signature(seed).split(":", 1)[0]


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
    signature, apply = build_rule(seed)
    family = signature.split(":", 1)[0]
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
        id=f"val-bit_manipulation-{family}-{seed}",
        category="bit_manipulation",
        examples=examples,
        question=query_bits,
        answer=answer_bits,
        prompt=prompt,
    )


register("bit_manipulation", generate, rule_signature)
