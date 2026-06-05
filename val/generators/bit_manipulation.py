"""Generator: new 8-bit rules drawn from the REAL bit_manipulation grammar.

Real bit rules are per-output-bit functions of a few input bits. The key
calibration fact (measured on 1,602 train problems) is that they are a MIXTURE
of homogeneous and heterogeneous problems, NOT uniformly one or the other:
  - ~43% mix >=2 distinct binary ops across the 8 output bits
  - ~48% have operand offsets that vary by column (not one global `(a+j,b+j)`)
  - the rest are HOMOGENEOUS: a single op applied at a global stride (the easy
    case a global-shift heuristic solves)

So each problem gets a single `het` coin (~46%): heterogeneous problems use an
independent op + arbitrary operands per column; homogeneous problems use one
global op at a global stride. This matches the measured marginals AND keeps the
absolute difficulty in line with the real test (an all-heterogeneous generator
over-corrected to ~47%, matching the solver-FAILURE floor rather than the test).

Coarse family mix (calibrated): ~65% pairwise-2input / ~20% routing (rotation,
rarely an arbitrary permutation) / ~15% complex (>=1 three-input majority/choice
column). Affine-mod-256, xor-mask and arbitrary 3-var truth tables are ~0% of
real data (TT3 was ~0.7% of columns and made problems under-determined), so we
don't emit them. Answers are correct by construction.
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
_HET_PROB = 0.46  # P(problem is heterogeneous); matches ~43% mix-ops / ~48% vary-offset

# A "column rule" is a tuple (kind, *params) describing how one output bit is
# computed from input bits:
#   ("route", p, neg)   -> in[p]   (xor neg)            (identity routing / NOT)
#   ("pair", op, p, q)  -> op(in[p], in[q])             (2-input boolean)
#   ("maj", p, q, r)    -> majority(in[p],in[q],in[r])
#   ("choice", p, q, r) -> in[q] if in[p] else in[r]


def _bit(x: int, p: int) -> int:
    return (x >> (7 - p)) & 1


def _pack(bits: list[int]) -> int:
    out = 0
    for j in range(8):
        out |= bits[j] << (7 - j)
    return out


def _eval_column(col: tuple, x: int) -> int:
    kind = col[0]
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
    raise ValueError(f"unknown column kind: {kind}")


def _rand_distinct_pair(rng: random.Random) -> tuple[int, int]:
    p = rng.randint(0, 7)
    q = rng.randint(0, 7)
    while q == p:
        q = rng.randint(0, 7)
    return p, q


def _build_columns(rng: random.Random, profile: str, het: bool) -> list[tuple]:
    if profile == "rot":
        glob_neg = rng.random() < 0.3
        if rng.random() < 0.90:  # coherent rotation (the routing bulk)
            k = rng.randint(1, 7)
            return [("route", (j + k) % 8, glob_neg) for j in range(8)]
        perm = list(range(8))  # arbitrary permutation (the rare routing tail)
        rng.shuffle(perm)
        return [("route", perm[j], glob_neg) for j in range(8)]

    # pairwise / complex share a global op + stride used by the HOMOGENEOUS case.
    global_op = rng.choice(_PAIR_OPS)
    a = rng.randint(0, 7)
    b = (a + rng.randint(1, 7)) % 8
    c = (a + rng.randint(1, 7)) % 8
    three: set[int] = set()
    if profile == "complex":
        three = set(rng.sample(range(8), rng.randint(1, 3)))

    cols: list[tuple] = []
    for j in range(8):
        if j in three:  # a genuine 3-input column (majority / choice)
            kind = rng.choice(("maj", "choice"))
            if het:
                p, q, r = rng.sample(range(8), 3)
            else:
                p, q, r = (a + j) % 8, (b + j) % 8, (c + j) % 8
            cols.append((kind, p, q, r))
        else:  # a 2-input column
            if het:
                op = rng.choice(_PAIR_OPS)
                p, q = _rand_distinct_pair(rng)
            else:
                op, p, q = global_op, (a + j) % 8, (b + j) % 8
            cols.append(("pair", op, p, q))
    return cols


def _col_sig(col: tuple) -> str:
    return ",".join(str(x) for x in col)


def _profile_and_columns(seed: int) -> tuple[str, bool, list[tuple]]:
    rng = random.Random(seed)
    roll = rng.random()
    profile = "pairwise" if roll < 0.65 else ("rot" if roll < 0.85 else "complex")
    het = profile != "rot" and rng.random() < _HET_PROB
    return profile, het, _build_columns(rng, profile, het)


def columns(seed: int) -> list[tuple]:
    """The 8 per-output-bit column rules for *seed* (for tests/breakdowns)."""
    return _profile_and_columns(seed)[2]


def problem_tag(seed: int) -> str:
    """Family + homogeneity, e.g. 'pairwise/het' — the stratum for breakdowns.

    The het stratum is where a heterogeneous-induction weakness (or a regression)
    shows; hom problems are the easy global-rule case. rot is always 'hom'.
    """
    profile, het, _ = _profile_and_columns(seed)
    return f"{profile}/{'het' if het else 'hom'}"


def build_rule(seed: int) -> tuple[str, Callable[[int], int]]:
    """Return (canonical_signature, apply) for the rule selected by *seed*.

    Family weights match real: ~65% pairwise, ~20% routing, ~15% complex; each
    problem is homogeneous (one global op+stride) or heterogeneous (independent
    per-column op + arbitrary operands) per the ~46% measured rate.
    """
    profile, het, cols = _profile_and_columns(seed)
    signature = f"{profile}:{'het' if het else 'hom'}:" + ";".join(
        _col_sig(c) for c in cols
    )

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
