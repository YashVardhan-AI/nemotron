"""Generator: new 8-bit rules drawn from the REAL bit_manipulation grammar.

Empirically (1,602 train problems + the solver's 85% rule_found rate), real bit
rules are **per-output-bit functions of a few input bits**, and crucially they are
HETEROGENEOUS across the 8 columns:
  - 42.6% of real problems mix >=2 distinct binary ops across the 8 output bits
  - 48.2% have operand offsets that VARY by column (not one global `(a+j,b+j)`)
  - identity-routing `out[j]=in[p]` (p!=j) is the single most common per-bit op
  - the complex tail includes arbitrary 3-var truth tables, not just MAJ/CHOICE

Coarse family mix (calibrated to real): ~65% pairwise-2input / ~20% routing
(rotation, occasionally an arbitrary permutation) / ~15% complex (>=1 three-input
column: majority / choice / arbitrary truth table). Affine-mod-256, xor-mask and
global permutations as a *family* are ~0% of real data, so we don't emit them.

The earlier version of this generator applied ONE global op at ONE global offset
across all 8 bits. That made the eval pattern-matchable from a single column and
INSENSITIVE TO REGRESSIONS (a model that kept only the global-shift heuristic
scored high). This version builds an INDEPENDENT rule per output bit with
arbitrary operands, so inducing the rule requires genuine per-column reasoning.
Answers are correct by construction; example sets may under-determine some columns
(real data does too) but the answer key always applies the ground-truth rule.
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

# A "column rule" is a tuple (kind, *params) describing how one output bit is
# computed from input bits:
#   ("const", v)            -> v
#   ("route", p, neg)       -> in[p]   (xor neg)         (identity routing / NOT)
#   ("pair", op, p, q)      -> op(in[p], in[q])          (2-input boolean)
#   ("maj", p, q, r)        -> majority(in[p],in[q],in[r])
#   ("choice", p, q, r)     -> in[q] if in[p] else in[r]
#   ("tt3", p, q, r, table) -> arbitrary 3-var truth table (table in 1..254)


def _bit(x: int, p: int) -> int:
    return (x >> (7 - p)) & 1


def _pack(bits: list[int]) -> int:
    out = 0
    for j in range(8):
        out |= bits[j] << (7 - j)
    return out


def _eval_column(col: tuple, x: int) -> int:
    kind = col[0]
    if kind == "const":
        return col[1]
    if kind == "route":
        _, p, neg = col
        b = _bit(x, p)
        return 1 - b if neg else b
    if kind == "pair":
        _, op, p, q = col
        u, v = _bit(x, p), _bit(x, q)
        if op.endswith("-NOT"):
            v = 1 - v
        base = op.split("-")[0]
        if base == "AND":
            return u & v
        if base == "OR":
            return u | v
        return u ^ v
    if kind == "maj":
        _, p, q, r = col
        return 1 if (_bit(x, p) + _bit(x, q) + _bit(x, r)) >= 2 else 0
    if kind == "choice":
        _, p, q, r = col
        return _bit(x, q) if _bit(x, p) else _bit(x, r)
    if kind == "tt3":
        _, p, q, r, table = col
        idx = (_bit(x, p) << 2) | (_bit(x, q) << 1) | _bit(x, r)
        return (table >> idx) & 1
    raise ValueError(f"unknown column kind: {kind}")


def _rand_pair_column(rng: random.Random) -> tuple:
    op = rng.choice(_PAIR_OPS)
    p = rng.randint(0, 7)
    q = rng.randint(0, 7)
    while q == p:
        q = rng.randint(0, 7)
    return ("pair", op, p, q)


def _rand_route_column(rng: random.Random) -> tuple:
    return ("route", rng.randint(0, 7), rng.random() < 0.3)


def _rand_three_column(rng: random.Random) -> tuple:
    kind = rng.choices(("choice", "maj", "tt3"), weights=(0.45, 0.20, 0.35))[0]
    p, q, r = rng.sample(range(8), 3)
    if kind == "maj":
        return ("maj", p, q, r)
    if kind == "choice":
        return ("choice", p, q, r)
    return ("tt3", p, q, r, rng.randint(1, 254))  # exclude const 0/255


def _build_columns(rng: random.Random, profile: str) -> list[tuple]:
    cols: list[tuple] = []
    if profile == "pairwise":
        # Heterogeneous: each bit an independent boolean of arbitrary operands,
        # mostly 2-input with a sprinkle of identity-routing / const columns.
        for _ in range(8):
            roll = rng.random()
            if roll < 0.12:
                cols.append(_rand_route_column(rng))
            elif roll < 0.20:
                cols.append(("const", rng.randint(0, 1)))
            else:
                cols.append(_rand_pair_column(rng))
    elif profile == "rot":
        glob_neg = rng.random() < 0.3
        if rng.random() < 0.90:  # coherent rotation (the routing bulk)
            k = rng.randint(1, 7)
            cols = [("route", (j + k) % 8, glob_neg) for j in range(8)]
        else:  # arbitrary permutation (the rare routing tail)
            perm = list(range(8))
            rng.shuffle(perm)
            cols = [("route", perm[j], glob_neg) for j in range(8)]
    else:  # complex: >=1 genuine 3-input column, the rest 2-input / routing
        three = set(rng.sample(range(8), rng.randint(1, 4)))
        for j in range(8):
            if j in three:
                cols.append(_rand_three_column(rng))
            elif rng.random() < 0.15:
                cols.append(_rand_route_column(rng))
            else:
                cols.append(_rand_pair_column(rng))
    return cols


def _col_sig(col: tuple) -> str:
    return ",".join(str(x) for x in col)


def _profile_and_columns(seed: int) -> tuple[str, list[tuple]]:
    rng = random.Random(seed)
    roll = rng.random()
    profile = "pairwise" if roll < 0.65 else ("rot" if roll < 0.85 else "complex")
    return profile, _build_columns(rng, profile)


def columns(seed: int) -> list[tuple]:
    """The 8 per-output-bit column rules for *seed* (for tests/breakdowns)."""
    return _profile_and_columns(seed)[1]


def build_rule(seed: int) -> tuple[str, Callable[[int], int]]:
    """Return (canonical_signature, apply) for the rule selected by *seed*.

    Family weights match the real distribution: ~65% pairwise, ~20% routing,
    ~15% complex. Each output bit is an INDEPENDENT function of arbitrary input
    bits (per-column heterogeneity), matching measured real micro-structure.
    """
    profile, cols = _profile_and_columns(seed)
    signature = f"{profile}:" + ";".join(_col_sig(c) for c in cols)

    def apply(x: int) -> int:
        return _pack([_eval_column(c, x) for c in cols])

    return signature, apply


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
