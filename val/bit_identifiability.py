"""Bit_manipulation identifiability ceiling — model-free root-cause probe.

Answers the question behind HET bit's ~47% accuracy: is it an INFORMATION cap
(the 8 worked examples don't uniquely determine the per-output-bit ops, so even an
oracle that knows the rule class caps near 47%) or a MODEL-error gap (the info IS
there; the LoRA just fails the 8-way independent per-column search)?

Why per-column, and only for HET:
  * HOMOGENEOUS rules COUPLE the 8 output bits (one global op + a sliding stride,
    or one rotation k), so 8 cols x 8 examples = 64 constraints pin a few global
    params -> massively over-determined. A per-column-independent analysis is the
    WRONG model for hom and wildly understates it (and the model already solves hom
    at 85-95%, so it is not the question).
  * HETEROGENEOUS rules draw each output bit's op + operands INDEPENDENTLY
    (reasoners.bit_rule._build_columns, het branch). So the query output is
    determined iff EACH column's query bit is determined, under that column's
    generative class (pair; or pair|maj|choice in complex -- never route, which
    only appears in the always-hom rot profile). Exact-match needs all 8 at once.

Method: reconstruct each problem's gold per-bit columns + the EXACT example/query
inputs the eval uses (shared reasoners.bit_rule core -> cannot drift from
val/kaggle_newrule_eval_standalone.py). For each output bit, enumerate the column
class, keep hypotheses consistent with all worked examples, check whether they
AGREE on the query bit. ceiling = % of problems with all 8 bits pinned.

Run:  uv run python val/bit_identifiability.py [N] [--sweep]
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reasoners.bit_rule import (  # noqa: E402
    PAIR_OPS,
    columns,
    distinct_inputs,
    eval_column,
    problem_tag,
)

BIT_DIFFICULTY = 8  # matches val/kaggle_newrule_eval_standalone.py (real avg 8.6)

# Measured model accuracy on the het-arm eval we are explaining, for side-by-side.
MODEL_ACC = {"pairwise/het": 46.7, "complex/het": 42.9}

# Column classes (which generative kinds an oracle considers per column).
PAIR_ONLY = frozenset({"pair"})  # the true class for pairwise/het (generous)
HET_CLASS = frozenset({"pair", "maj", "choice"})  # profile-agnostic het class
FULL_CLASS = frozenset({"route", "pair", "maj", "choice"})


def build_hypotheses(kinds: frozenset[str]) -> list[tuple[object, ...]]:
    """All column hypotheses of the requested kinds, over the generator's support
    (distinct positions)."""
    h: list[tuple[object, ...]] = []
    if "route" in kinds:
        for p in range(8):
            h.append(("route", p, False))
            h.append(("route", p, True))
    if "pair" in kinds:
        for op in PAIR_OPS:
            for p in range(8):
                for q in range(8):
                    if q != p:
                        h.append(("pair", op, p, q))
    if "maj" in kinds:
        for p, q, r in combinations(range(8), 3):  # maj symmetric -> unordered
            h.append(("maj", p, q, r))
    if "choice" in kinds:
        for p in range(8):
            for q in range(8):
                for r in range(8):
                    if len({p, q, r}) == 3:
                        h.append(("choice", p, q, r))
    return h


def _bits(x: int) -> tuple[int, ...]:
    return tuple((x >> (7 - p)) & 1 for p in range(8))


def _fast(col: tuple[object, ...], bits: tuple[int, ...]) -> int:
    """Fast column eval on a precomputed bit tuple (parity-checked vs eval_column)."""
    k = col[0]
    if k == "route":
        b = bits[col[1]]  # type: ignore[index]
        return 1 - b if col[2] else b
    if k == "pair":
        _, op, p, q = col
        u, v = bits[p], bits[q]  # type: ignore[index]
        if op[-4:] == "-NOT":  # type: ignore[index]
            v = 1 - v
        base = op[0]  # type: ignore[index]  # 'A' / 'O' / 'X'
        if base == "A":
            return u & v
        if base == "O":
            return u | v
        return u ^ v
    if k == "maj":
        _, p, q, r = col
        return 1 if bits[p] + bits[q] + bits[r] >= 2 else 0  # type: ignore[index]
    _, p, q, r = col  # choice
    return bits[q] if bits[p] else bits[r]  # type: ignore[index]


def _column_determined(
    col: tuple[object, ...],
    hyps: list[tuple[object, ...]],
    ex_inputs: list[int],
    q_input: int,
) -> bool:
    """Is the query bit of `col` uniquely pinned by the examples, under `hyps`?"""
    gold_ex = tuple(eval_column(col, x) for x in ex_inputs)
    gold_q = eval_column(col, q_input)
    q_bits = _bits(q_input)
    ex_bits = [_bits(x) for x in ex_inputs]
    q_vals = set()
    gold_seen = False
    for hh in hyps:
        if tuple(_fast(hh, b) for b in ex_bits) == gold_ex:
            q_vals.add(_fast(hh, q_bits))
            if _fast(hh, q_bits) == gold_q:
                gold_seen = True
    if not gold_seen:
        raise AssertionError("gold column not representable in this class")
    return len(q_vals) == 1


def analyze(seed: int, difficulty: int, hyps: list[tuple[object, ...]]) -> int:
    """Number of query bits NOT pinned (0 => problem fully solvable by oracle)."""
    cols = columns(seed)
    values = distinct_inputs(seed, difficulty + 1)
    ex_inputs, q_input = values[:difficulty], values[difficulty]
    return sum(
        0 if _column_determined(c, hyps, ex_inputs, q_input) else 1 for c in cols
    )


def _validate_fast(rng: random.Random) -> None:
    h = build_hypotheses(FULL_CLASS)
    for _ in range(20000):
        col = rng.choice(h)
        x = rng.randint(0, 255)
        if _fast(col, _bits(x)) != eval_column(col, x):
            raise AssertionError(f"_fast disagrees with eval_column on {col} x={x}")


def _validate_full_observation() -> None:
    """With ALL 256 inputs observed, every column must be fully determined (0
    ambiguous) -- proves the consistency machinery is sound, class-agnostically."""
    all_inputs = list(range(256))
    hyps = build_hypotheses(FULL_CLASS)
    for seed in range(10):
        cols = columns(seed)
        for c in cols:
            # query = any held-out is impossible (all observed); use input 0 as
            # the 'query' but it is also in examples, so it is trivially pinned.
            # Instead test: consistent set's outputs agree on EVERY input.
            gold_full = tuple(eval_column(c, x) for x in all_inputs)
            survivors = [
                hh
                for hh in hyps
                if tuple(_fast(hh, _bits(x)) for x in all_inputs) == gold_full
            ]
            # every survivor computes the identical function on all 256 inputs.
            for x in (3, 77, 200):
                vals = {_fast(hh, _bits(x)) for hh in survivors}
                if len(vals) != 1:
                    raise AssertionError(
                        f"full-obs seed={seed} col={c} x={x} ambiguous {vals}"
                    )


import math  # noqa: E402


def verify(stored: str, pred: str) -> bool:
    """The REAL competition grader (lenient): binary strings parsed as decimals,
    within 1% count equal. Verbatim from val/kaggle_newrule_eval_standalone.py."""
    stored, pred = stored.strip(), pred.strip()
    try:
        return math.isclose(float(stored), float(pred), rel_tol=1e-2, abs_tol=1e-5)
    except (ValueError, TypeError):
        return pred.lower() == stored.lower()


def _query_dist(
    col: tuple[object, ...],
    hyps: list[tuple[object, ...]],
    ex_bits: list[tuple[int, ...]],
    q_bits: tuple[int, ...],
    gold_ex: tuple[int, ...],
) -> dict[int, int]:
    """{query_bit_value: #consistent hypotheses} for one column under `hyps`."""
    dist: dict[int, int] = {0: 0, 1: 0}
    for hh in hyps:
        if tuple(_fast(hh, b) for b in ex_bits) == gold_ex:
            dist[_fast(hh, q_bits)] += 1
    return dist


def het_grader_ceilings(n: int, difficulty: int, rng: random.Random):
    """Per het stratum, under the GENERATIVELY-CORRECT class (pairwise/het->pair,
    complex/het->pair|maj|choice): exact ceiling, plus three grader-honest numbers
    under the REAL lenient grader -- MAP-oracle, determine+random-guess (expected),
    and a fully-random-8-bit baseline. Compare all to model lenient/strict."""
    h_pair = build_hypotheses(PAIR_ONLY)
    h_het = build_hypotheses(HET_CLASS)
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for seed in range(n):
        tag = problem_tag(seed)
        if not tag.endswith("/het"):
            continue
        hyps = h_pair if tag == "pairwise/het" else h_het
        cols = columns(seed)
        values = distinct_inputs(seed, difficulty + 1)
        ex_inputs, q_input = values[:difficulty], values[difficulty]
        ex_bits = [_bits(x) for x in ex_inputs]
        q_bits = _bits(q_input)
        gold_bits = [eval_column(c, q_input) for c in cols]
        gold_str = "".join(str(b) for b in gold_bits)

        dists = []
        ambiguous = []
        for j, c in enumerate(cols):
            gold_ex = tuple(eval_column(c, x) for x in ex_inputs)
            d = _query_dist(c, hyps, ex_bits, q_bits, gold_ex)
            present = [v for v in (0, 1) if d[v] > 0]
            dists.append(d)
            ambiguous.append(len(present) > 1)

        exact_solvable = not any(ambiguous)
        # MAP-oracle: determined bits = gold; ambiguous bits = argmax count.
        map_bits = [
            gold_bits[j]
            if not ambiguous[j]
            else (0 if dists[j][0] >= dists[j][1] else 1)
            for j in range(8)
        ]
        map_str = "".join(str(b) for b in map_bits)
        # determine + random-guess (expected over uniform guesses of ambiguous bits).
        amb_idx = [j for j in range(8) if ambiguous[j]]
        k = len(amb_idx)
        if k == 0:
            rg_lenient = 1.0
        else:
            hit = 0
            for mask in range(1 << k):
                b = list(gold_bits)
                for t, j in enumerate(amb_idx):
                    b[j] = (mask >> t) & 1
                if verify(gold_str, "".join(str(x) for x in b)):
                    hit += 1
            rg_lenient = hit / (1 << k)
        # fully-random 8-bit baseline (no inference at all).
        rnd_hits = 0
        for _ in range(40):
            rb = "".join(str(rng.randint(0, 1)) for _ in range(8))
            rnd_hits += int(verify(gold_str, rb))
        acc[tag]["exact"].append(int(exact_solvable))
        acc[tag]["map_lenient"].append(int(verify(gold_str, map_str)))
        acc[tag]["rg_lenient"].append(rg_lenient)
        acc[tag]["rand8_lenient"].append(rnd_hits / 40)
    out: dict[str, dict[str, float]] = {}
    for tag, d in acc.items():
        out[tag] = {key: 100.0 * sum(v) / len(v) for key, v in d.items()}
        out[tag]["n"] = float(len(d["exact"]))
    return out


def het_ceilings(n: int, difficulty: int) -> dict[str, dict[str, float]]:
    """Per-het-stratum ceiling. pairwise/het uses PAIR_ONLY (its true, generous
    class) AND HET_CLASS (profile-agnostic); complex/het uses HET_CLASS."""
    h_pair = build_hypotheses(PAIR_ONLY)
    h_het = build_hypotheses(HET_CLASS)
    agg: dict[str, dict[str, list[int]]] = {
        "pairwise/het@pair_only": {"amb": [], "solv": []},
        "pairwise/het@het_class": {"amb": [], "solv": []},
        "complex/het@het_class": {"amb": [], "solv": []},
    }
    for seed in range(n):
        tag = problem_tag(seed)
        if not tag.endswith("/het"):
            continue
        if tag == "pairwise/het":
            for key, hyps in (
                ("pairwise/het@pair_only", h_pair),
                ("pairwise/het@het_class", h_het),
            ):
                amb = analyze(seed, difficulty, hyps)
                agg[key]["amb"].append(amb)
                agg[key]["solv"].append(int(amb == 0))
        elif tag == "complex/het":
            amb = analyze(seed, difficulty, h_het)
            agg["complex/het@het_class"]["amb"].append(amb)
            agg["complex/het@het_class"]["solv"].append(int(amb == 0))
    out: dict[str, dict[str, float]] = {}
    for key, d in agg.items():
        nn = len(d["solv"])
        if nn == 0:
            continue
        mean_amb = sum(d["amb"]) / nn
        out[key] = {
            "n": nn,
            "ceiling_pct": 100.0 * sum(d["solv"]) / nn,
            "mean_ambig": mean_amb,
            "per_col_determined_pct": 100.0 * (1 - mean_amb / 8),
        }
    return out


def _print(out: dict[str, dict[str, float]], difficulty: int) -> None:
    print(f"\n{'=' * 82}")
    print(f"HET oracle identifiability ceiling  (difficulty={difficulty} examples)")
    print("ceiling = % of HET problems where ALL 8 query bits are pinned (oracle EM)")
    print("=" * 82)
    print(
        f"{'stratum @ class':<26} {'N':>5} {'model%':>7} {'ceiling%':>9} "
        f"{'gap_pp':>7} {'mean_amb':>9} {'per-col det%':>12}"
    )
    print("-" * 82)
    for key in sorted(out):
        s = out[key]
        base = key.split("@")[0]
        m = MODEL_ACC.get(base)
        mstr = f"{m:>7.1f}" if m is not None else f"{'--':>7}"
        gap = f"{s['ceiling_pct'] - m:>7.1f}" if m is not None else f"{'--':>7}"
        print(
            f"{key:<26} {int(s['n']):>5} {mstr} {s['ceiling_pct']:>9.1f} {gap} "
            f"{s['mean_ambig']:>9.2f} {s['per_col_determined_pct']:>12.1f}"
        )
    print(
        "\nREAD: ceiling ~= model -> INFO-CAP (conjunction of 8 per-col ambiguities; "
        "stop).\n      ceiling >> model -> MODEL-ERROR (bit-het SFT has headroom)."
    )


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if args else 1500
    do_sweep = "--sweep" in sys.argv[1:]

    _validate_fast(random.Random(12345))
    print("[validate] _fast matches eval_column on 20k random (col,x) pairs  OK")
    _validate_full_observation()
    print("[validate] full-observation -> every column determined  OK")
    print(
        f"[classes] |pair_only|={len(build_hypotheses(PAIR_ONLY))}  "
        f"|het_class|={len(build_hypotheses(HET_CLASS))}"
    )

    out = het_ceilings(n, BIT_DIFFICULTY)
    _print(out, BIT_DIFFICULTY)

    g = het_grader_ceilings(n, BIT_DIFFICULTY, random.Random(7))
    print(f"\n{'=' * 82}")
    print("GRADER-HONEST het ceilings under the REAL (lenient) competition grader")
    print("MAP=oracle determines + picks most-likely ambiguous bit; rg=determine+")
    print("random-guess (expected); rand8=no inference. model_lenient/strict measured.")
    print("=" * 82)
    print(
        f"{'stratum':<14} {'N':>5} {'modelLen':>9} {'modelStr':>9} {'exact':>7} "
        f"{'MAPlen':>7} {'rgLen':>7} {'rand8':>7}"
    )
    print("-" * 82)
    model_strict = {"pairwise/het": 3.3, "complex/het": 0.0}
    for tag in sorted(g):
        s = g[tag]
        ml = MODEL_ACC.get(tag, float("nan"))
        ms = model_strict.get(tag, float("nan"))
        print(
            f"{tag:<14} {int(s['n']):>5} {ml:>9.1f} {ms:>9.1f} {s['exact']:>7.1f} "
            f"{s['map_lenient']:>7.1f} {s['rg_lenient']:>7.1f} {s['rand8_lenient']:>7.1f}"
        )
    print(
        "\nREAD: if MAP/rg ~= model_lenient -> het is capped UNDER THE REAL GRADER too "
        "(stop).\n      if rand8 ~= model_lenient -> the 46.7% is mostly grader "
        "leniency, not skill."
    )

    if do_sweep:
        print(f"\n{'=' * 82}")
        print("HET difficulty sweep (real avg = 8.6 examples) — ceiling%")
        print("=" * 82)
        print(
            f"{'difficulty':>10} {'pw/het@pair':>14} {'pw/het@het':>14} "
            f"{'cx/het@het':>14}"
        )
        for d in (6, 7, 8, 9, 10, 12):
            s = het_ceilings(n, d)
            a = s.get("pairwise/het@pair_only", {}).get("ceiling_pct", float("nan"))
            b = s.get("pairwise/het@het_class", {}).get("ceiling_pct", float("nan"))
            c = s.get("complex/het@het_class", {}).get("ceiling_pct", float("nan"))
            print(f"{d:>10} {a:>14.1f} {b:>14.1f} {c:>14.1f}")


if __name__ == "__main__":
    main()
