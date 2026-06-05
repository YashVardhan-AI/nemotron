"""Acceptance test for the bit-generator recalibration (spec Phase 0.2).

The OLD generator applied ONE global op at ONE global offset across all 8 bits,
so a model that only learned "find the single global rule" scored high — the eval
could not see a regression. This test proves the RECALIBRATED generator is
regression-sensitive WITHOUT a GPU, by scoring two synthetic predictors:

  * oracle              — knows the true rule (upper bound)
  * global-heuristic    — fits the best single OLD-style global rule to the
                          examples and applies it (the behavior the old eval
                          rewarded; a stand-in for a model that kept only that
                          skill)

A trustworthy eval must: (a) let the oracle score ~100% on the NEW generator,
(b) let the global-heuristic still ace OLD-style problems, but (c) drop the
global-heuristic substantially on the NEW generator. (c) is the property that
was missing.
"""

import random

from reasoners.store_types import Example, Problem
from val.generators.bit_manipulation import (
    _bit,
    _pack,
    build_rule,
    generate,
)

_PAIR_OPS = ("AND", "OR", "XOR", "AND-NOT", "OR-NOT", "XOR-NOT")


def _apply_pair(op, a, b, x):
    base, neg = op.split("-")[0], op.endswith("-NOT")
    bits = []
    for j in range(8):
        u, v = _bit(x, (a + j) % 8), _bit(x, (b + j) % 8)
        if neg:
            v = 1 - v
        bits.append(u & v if base == "AND" else u | v if base == "OR" else u ^ v)
    return _pack(bits)


def _apply_rot(k, inv, x):
    bits = [_bit(x, (j + k) % 8) for j in range(8)]
    if inv:
        bits = [1 - b for b in bits]
    return _pack(bits)


def _apply_complex(kind, a, b, c, x):
    bits = []
    for j in range(8):
        p, q, s = _bit(x, (a + j) % 8), _bit(x, (b + j) % 8), _bit(x, (c + j) % 8)
        bits.append((1 if p + q + s >= 2 else 0) if kind == "MAJ" else (q if p else s))
    return bits and _pack(bits)


def _old_hypotheses():
    """The OLD generator's entire rule space: one global op/offset per word."""
    hyps = []
    for op in _PAIR_OPS:
        for a in range(8):
            for b in range(8):
                if b != a:
                    hyps.append((lambda x, op=op, a=a, b=b: _apply_pair(op, a, b, x)))
    for k in range(1, 8):
        for inv in (0, 1):
            hyps.append((lambda x, k=k, inv=inv: _apply_rot(k, inv, x)))
    for kind in ("MAJ", "CHOICE"):
        for a in range(8):
            for b in range(8):
                for c in range(8):
                    hyps.append(
                        lambda x, kind=kind, a=a, b=b, c=c: _apply_complex(
                            kind, a, b, c, x
                        )
                    )
    return hyps


_OLD_HYPS = _old_hypotheses()


def _bits_match(u: int, v: int) -> int:
    return sum(((u >> i) & 1) == ((v >> i) & 1) for i in range(8))


def global_heuristic_predict(problem: Problem) -> str:
    """Pick the single global OLD-style rule best fitting the examples; apply it.

    This is the most a model that learned ONLY the old distribution could do.
    """
    ex = [(int(e.input_value, 2), int(e.output_value, 2)) for e in problem.examples]
    q = int(problem.question, 2)
    best_fn, best_score = None, -1
    for fn in _OLD_HYPS:
        score = sum(_bits_match(fn(x), y) for x, y in ex)
        if score > best_score:
            best_fn, best_score = fn, score
    return format(best_fn(q), "08b")


def oracle_predict(problem: Problem, seed: int) -> str:
    _sig, apply = build_rule(seed)
    return format(apply(int(problem.question, 2)), "08b")


def _old_style_problem(seed: int, difficulty: int = 8) -> Problem:
    """A problem whose rule IS a single global pairwise op (old distribution)."""
    rng = random.Random(90_000 + seed)
    op = rng.choice(_PAIR_OPS)
    a = rng.randint(0, 7)
    b = (a + rng.randint(1, 7)) % 8
    vals, seen = [], set()
    vrng = random.Random(seed * 13 + 7)
    while len(vals) < difficulty + 1:
        v = vrng.randint(0, 255)
        if v not in seen:
            seen.add(v)
            vals.append(v)
    examples = [
        Example(format(x, "08b"), format(_apply_pair(op, a, b, x), "08b"))
        for x in vals[:difficulty]
    ]
    q = vals[difficulty]
    return Problem(
        id=f"old-{seed}",
        category="bit_manipulation",
        examples=examples,
        question=format(q, "08b"),
        answer=format(_apply_pair(op, a, b, q), "08b"),
        prompt="",
    )


def _acc(predicted, problems):
    return sum(p.answer == pred for p, pred in zip(problems, predicted)) / len(problems)


def test_recalibrated_generator_is_regression_sensitive():
    seeds = range(120)
    new_problems = [generate(s, difficulty=8) for s in seeds]

    oracle = [oracle_predict(p, s) for s, p in zip(seeds, new_problems)]
    heur_new = [global_heuristic_predict(p) for p in new_problems]
    old_problems = [_old_style_problem(s) for s in seeds]
    heur_old = [global_heuristic_predict(p) for p in old_problems]

    acc_oracle = _acc(oracle, new_problems)
    acc_heur_new = _acc(heur_new, new_problems)
    acc_heur_old = _acc(heur_old, old_problems)

    # (a) the eval is solvable: a correct reasoner aces the NEW generator.
    assert acc_oracle > 0.99, acc_oracle
    # (b) the global-heuristic is competent on the OLD distribution it models.
    assert acc_heur_old > 0.85, acc_heur_old
    # (c) THE KEY PROPERTY: the same heuristic drops sharply on the NEW
    #     generator — i.e. a regressed/global-only model now scores lower.
    assert acc_heur_new < 0.60, acc_heur_new
    assert acc_heur_old - acc_heur_new > 0.30, (acc_heur_old, acc_heur_new)
