"""Shared rule core for the cryptarithm arithmetic-cipher family.

A rule = (injective symbol<->digit map) x (per-operator-glyph operation, one of
FIVE: add/abs_diff/mul/concat/rev_concat) x re-encoding of the result digits back
to symbols. Semantics ported from investigators/cryptarithm_deduce.py (the
authoritative full-family solver). Calibrated to the verified real
cryptarithm_deduce distribution (see docs/superpowers/plans/2026-06-02-
cryptarithm-p1.md): 5-char inputs `s0 s1 <op> s3 s4`, operands are two-digit
numbers, operator semantics are deduced per problem.

Pure: no I/O, no CoT. Imported by val/generators/cryptarithm.py and (later)
reasoners/cryptarithm_trace.py so the eval and the training data share one
definition -- the measured val number and the trained data cannot drift.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Glyphs that encode digits 0-9. Operator glyphs may *also* be drawn from this
# pool (real operators overlap digit-symbols; the role is purely positional).
SYMBOLS = list("!\"#$%&'()/:<>?@[]^`{|}\\")
# The common operators (always candidates); a long tail of other glyphs is added
# per-rule with TAIL_OPERATOR_PROB so the model does not overfit to {+,-,*}.
BASE_OPERATORS = ("+", "-", "*")
TAIL_OPERATOR_PROB = 0.15
OP_NAMES = ("add", "abs_diff", "mul", "concat", "rev_concat")

_OPS = {
    "add": lambda a, b: a + b,
    "abs_diff": lambda a, b: abs(a - b),
    "mul": lambda a, b: a * b,
    "concat": lambda a, b: a * 100 + b,
    "rev_concat": lambda a, b: b * 100 + a,
}
# Sampling weights: keep all five, lean toward the arithmetic ops (the headroom).
_OP_WEIGHTS = {"add": 3, "abs_diff": 3, "mul": 3, "concat": 2, "rev_concat": 1}

# --- prompt wrappers (instantiates super-plan lever #4: template diversity) ---
# Index 0 is the EXACT real wrapper; the val generator always uses it so the
# measured number reflects the real format. The training emitter samples a small
# fraction of paraphrases. Paraphrases vary only surrounding prose, never the
# `s0 s1 op s3 s4` token structure, the ` = ` joiner, or the trailing query.
WRAPPERS = [
    (
        "In Alice's Wonderland, a secret set of transformation rules is applied "
        "to equations. Below are a few examples:",
        "Now, determine the result for: ",
    ),
    (
        "In Wonderland, a hidden set of transformation rules is applied to each "
        "equation. Study these examples:",
        "Using the same rules, determine the result for: ",
    ),
    (
        "A secret set of rules transforms equations in Wonderland. Here are some "
        "examples:",
        "Now find the result for: ",
    ),
]


def num_to_digits(n: int) -> tuple[int, ...]:
    """Digits of *n*, most-significant first (ported from the investigator)."""
    if n == 0:
        return (0,)
    d: list[int] = []
    while n > 0:
        d.append(n % 10)
        n //= 10
    return tuple(reversed(d))


@dataclass
class CryptRule:
    digit_to_sym: dict[int, str]
    sym_to_digit: dict[str, int]
    op_of: dict[str, str]  # operator glyph -> op name in OP_NAMES

    @property
    def operators(self) -> tuple[str, ...]:
        return tuple(self.op_of)

    def apply(self, op: str, left: int, right: int) -> int:
        return _OPS[self.op_of[op]](left, right)

    def result_digits(self, op: str, left: int, right: int) -> tuple[int, ...]:
        """Result digits for operands *left*/*right* (full integers)."""
        name = self.op_of[op]
        val = _OPS[name](left, right)
        if name in ("concat", "rev_concat"):  # always 4 digits, zero-padded
            return (val // 1000, (val // 100) % 10, (val // 10) % 10, val % 10)
        return num_to_digits(val)

    def encode_output(self, digits: tuple[int, ...]) -> str:
        return "".join(self.digit_to_sym[d] for d in digits)

    def encode_example(
        self, op: str, left: tuple[int, int], right: tuple[int, int]
    ) -> tuple[str, str]:
        """Return (input_value, output_value) for digit-pairs *left*/*right*."""
        lnum, rnum = 10 * left[0] + left[1], 10 * right[0] + right[1]
        inp = (
            self.digit_to_sym[left[0]]
            + self.digit_to_sym[left[1]]
            + op
            + self.digit_to_sym[right[0]]
            + self.digit_to_sym[right[1]]
        )
        return inp, self.encode_output(self.result_digits(op, lnum, rnum))


def build_rule(seed: int) -> CryptRule:
    rng = random.Random(seed)
    glyphs = rng.sample(SYMBOLS, 10)  # 10 distinct glyphs, one per digit
    digit_to_sym = {d: glyphs[d] for d in range(10)}
    sym_to_digit = {s: d for d, s in digit_to_sym.items()}

    operators = list(BASE_OPERATORS)
    if rng.random() < TAIL_OPERATOR_PROB:
        # A tail operator drawn from the full glyph pool (may also encode a digit;
        # parsing stays positional). Keep distinct from the base operators.
        operators.append(rng.choice(SYMBOLS))

    names = list(_OP_WEIGHTS)
    weights = [_OP_WEIGHTS[n] for n in names]
    op_of = {op: rng.choices(names, weights=weights)[0] for op in operators}
    return CryptRule(digit_to_sym, sym_to_digit, op_of)


# Curriculum profiles: bias the QUERY operator + operand range (used in Phase 6).
# Default = the measured real mix (all ops, full operand range).
_PROFILE_OPS = {
    "easy": ("add", "concat", "rev_concat"),
    "hard": ("mul", "abs_diff"),
}
_PROFILE_RANGE = {"easy": (0, 5), "hard": (0, 9)}


def sample_problem(
    seed: int,
    difficulty: int = 4,
    *,
    determinacy: str = "well",
    profile: str | None = None,
    salt: int = 0,
) -> tuple[CryptRule, list[tuple[str, str]], str, str]:
    """Sample one solvable instance for this rule. Single source of truth for
    both the val generator and the training emitter (no drift).

    Returns (rule, examples, q_input, q_answer) where examples is a list of
    (input_value, output_value). Guarantees, by construction (no appended cover
    demos -> exactly *difficulty* demos): (a) the query operator appears in a
    demo, and (b) every digit-glyph in q_input operands + q_answer is witnessed
    in a demo operand/answer position (so symbol->digit is deducible).

    determinacy: "well" (default) over-specifies the map; "near" packs the needed
    digits with no redundant filler (mild uncertainty -- ~10-15% of training).
    profile: None (real mix) | "easy" | "hard" (curriculum, Phase 6).
    salt: re-rolls the instance (operands/demos) while keeping the SAME rule (so
    rule_signature(seed) stays stable for holdout). Used by the solver-in-the-loop
    sampler (cryptarithm_deduce_core.sample_solvable) to resample until the
    instance is uniquely deducible.
    """
    rng = random.Random(seed * 7919 + 1 + salt * 104729)  # own stream; salt re-rolls
    rule = build_rule(seed)
    ops = rule.operators

    lo_d, hi_d = _PROFILE_RANGE.get(profile or "", (0, 9))

    def digit() -> int:
        return rng.randint(lo_d, hi_d)

    # Choose the query operator (curriculum may bias toward easy/hard ops).
    if profile in _PROFILE_OPS:
        pref = [op for op in ops if rule.op_of[op] in _PROFILE_OPS[profile]] or list(
            ops
        )
    else:
        pref = list(ops)
    q_op = rng.choice(pref)
    q_left = (digit(), digit())
    q_right = (digit(), digit())
    q_input, q_answer = rule.encode_example(q_op, q_left, q_right)

    # Digits that MUST be witnessed so the query is readable + its answer codable.
    needed = sorted(
        {q_left[0], q_left[1], q_right[0], q_right[1]}
        | {rule.sym_to_digit[g] for g in q_answer}
    )

    total_slots = difficulty * 4  # each demo contributes 4 operand-digit slots
    if len(needed) > total_slots:  # only possible at difficulty < 3; clamp safely
        needed = needed[:total_slots]
    digit_seq = list(needed)
    if determinacy != "near":
        # "well": pad with random filler -> redundant witnessing (over-specified).
        while len(digit_seq) < total_slots:
            digit_seq.append(digit())
    else:
        # "near": minimal filler, only enough to fill the demo slots once.
        while len(digit_seq) < total_slots:
            digit_seq.append(rng.choice(needed) if needed else digit())
    rng.shuffle(digit_seq)

    triples: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    for i in range(difficulty):
        chunk = digit_seq[i * 4 : i * 4 + 4]
        left = (chunk[0], chunk[1])
        right = (chunk[2], chunk[3])
        # First two demos witness the query op (one demo can leave it ambiguous
        # between ops of equal result length; two varied demos pin it).
        op = q_op if i < 2 else rng.choice(ops)
        triples.append((op, left, right))
    rng.shuffle(triples)  # so the query-operator demo is not always first

    examples = [rule.encode_example(op, lo, ro) for op, lo, ro in triples]
    return rule, examples, q_input, q_answer


def render_prompt(
    examples: list[tuple[str, str]], q_input: str, wrapper_index: int = 0
) -> str:
    """Build the prompt from demo pairs. wrapper_index 0 = the exact real format
    (val default); higher indices are light paraphrases (training diversity)."""
    header, lead = WRAPPERS[wrapper_index % len(WRAPPERS)]
    lines = "\n".join(f"{i} = {o}" for i, o in examples)
    return f"{header}\n{lines}\n{lead}{q_input}"
