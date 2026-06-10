"""Render correct forced-deduction CoT for cryptarithm ("equation_symbolic")
puzzles, driven by the verified symbolic solver
(`kaggle-nemotron-equation-symbolic-main`).

Background (see memory `cryptarithm-symbolic-solver-found`): the solver cracks
~100% of train cryptarithm gold-conditioned, but stores only the *program*
(glyph->digit map, per-operator op, mode, base) in `solver_results.parquet` —
no rendered reasoning text. This module is the missing renderer: program ->
natural-language CoT ending in a tight `\\boxed{answer}`.

One entry point — `render(prompt, mapping, ops, mode, base, gold, style)` — with a
selectable Step-2 (an A/B over the digit-deduction shape). Steps 1/3/4 (operator
id, verify every example, apply to query) are identical across styles; only Step 2
differs:
  * "lean"   — state the matched glyph->digit map directly. Universal (every base,
               every mode), base/mode-aware throughout.
  * "assert" — base-10 std/alice: the MRV forced/guess scratchpad from the solver's
               `_derive_order` (the map is gold-forced, so it mostly reads as
               "only {d} left"); falls back to "lean" elsewhere.
  * "derive" — base-10 std/alice: the GENUINE constraint-propagation deduction
               (each glyph 0-9 narrowed per example, intersected to singletons);
               falls back to "lean" elsewhere.
  * "derive_inductive" — like "derive" but BOUNDED: exactly one fixed-shape row
               per glyph (a step-count-INDEPENDENT inductive scratchpad — the row
               count tracks the glyph count, never the search depth), and result
               magnitudes are spelled least-significant-digit-first in the reversed
               modes. Falls back to a "lean" map where `_derive_order` is unsound,
               but KEEPS the LSB-first encoding annotations everywhere.
  * "derive_search" — the LONG arm. Mirrors `reasoning_equation_numeric` (the
               student's closest analog, learned ~87.5%): map FIRST, then an
               enumerate-and-test that tries candidate OPERATIONS on the known digits
               and shows wrong/match for each (Stream-of-Search shape). Universal —
               the op-search works in every base/mode — so it is the long, high-budget
               counterpart to the short bounded styles above.
`_derive_order` is sound only for base==10 + std/alice (it hardcodes base 10, flips
only for `alice`, and zeroes domains on concat ops — so concat ops are stripped
before the call and other regimes use "lean").

CRITICAL: we do NOT use the solver's `apply_query_trace` (it hardcodes base 10
and only flips for `alice`). All arithmetic here goes through the base/mode-aware
semantics below, mirroring the verified `_verify_solver.py`.

Glyphs in the *reasoning* are space-separated to defeat the Nemotron BPE merge
that fuses the operator into an operand (see `cryptarithm-tokenizer-merge`); the
final `\\boxed{}` is kept tight/unspaced because the grader compares against the
unspaced gold.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = (
    Path(__file__).resolve().parent.parent
    / "kaggle-nemotron-equation-symbolic-main"
    / "kaggle-nemotron-equation-symbolic-main"
    / "src"
)
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import solver_eq_symbolic as S  # noqa: E402

OPERATIONS = S.OPERATIONS
SIGNED_OPS = S.SIGNED_OPS
CONCAT_OPS = {"concat_fwd", "concat_rev"}


# ─────────────────────── operator descriptions ───────────────────────
# Short human phrase + a symbolic-expression template for each op. Values are
# always computed from OPERATIONS (never re-implemented here) so the narrative
# can never disagree with the solver's own semantics.

_OP_PHRASE = {
    "add": "add the two numbers",
    "sub": "subtract (left minus right)",
    "rsub": "subtract (right minus left)",
    "absdiff": "take the absolute difference",
    "neg_absdiff": "take the absolute difference (written with a leading sign glyph)",
    "mul": "multiply the two numbers",
    "gcd": "take the greatest common divisor",
    "lcm": "take the least common multiple",
    "fdiv": "floor-divide (left by right)",
    "rdiv": "floor-divide (right by left)",
    "mod": "take the remainder (left mod right)",
    "rmod": "take the remainder (right mod left)",
    "min": "take the smaller number",
    "max": "take the larger number",
    "add_m1": "add then subtract 1",
    "add_p1": "add then add 1",
    "add_m2": "add then subtract 2",
    "add_p2": "add then add 2",
    "mul_m1": "multiply then subtract 1",
    "mul_p1": "multiply then add 1",
    "mul_m2": "multiply then subtract 2",
    "mul_p2": "multiply then add 2",
    "absdiff_m1": "absolute difference then subtract 1",
    "absdiff_p1": "absolute difference then add 1",
    "absdiff_m2": "absolute difference then subtract 2",
    "absdiff_p2": "absolute difference then add 2",
    "sub_m1": "subtract (left minus right) then subtract 1",
    "sub_p1": "subtract (left minus right) then add 1",
    "rsub_m1": "subtract (right minus left) then subtract 1",
    "rsub_p1": "subtract (right minus left) then add 1",
    "mul_half": "multiply then halve",
    "mul_double": "multiply then double",
    "sq_diff": "square the difference",
    "sq_sum": "square the sum",
    "mul_plus_a": "multiply then add the left number",
    "mul_plus_b": "multiply then add the right number",
    "mul_minus_a": "multiply then subtract the left number",
    "mul_minus_b": "multiply then subtract the right number",
    "a2_plus_b": "square the left number then add the right",
    "a_plus_b2": "square the right number then add the left",
    "xor": "bitwise XOR",
    "band": "bitwise AND",
    "bor": "bitwise OR",
    "sub_signed": "subtract (left minus right; may be negative)",
    "rsub_signed": "subtract (right minus left; may be negative)",
    "concat_fwd": "concatenate the operand glyphs in order",
    "concat_rev": "concatenate the operand glyphs reversed",
}


def _expr(ot: str, a: int, b: int) -> str:
    """Symbolic expression string for display (value comes from OPERATIONS)."""
    t = {
        "add": f"{a}+{b}",
        "sub": f"{a}-{b}",
        "rsub": f"{b}-{a}",
        "absdiff": f"|{a}-{b}|",
        "neg_absdiff": f"|{a}-{b}|",
        "mul": f"{a}*{b}",
        "gcd": f"gcd({a},{b})",
        "lcm": f"lcm({a},{b})",
        "fdiv": f"{a}//{b}",
        "rdiv": f"{b}//{a}",
        "mod": f"{a} mod {b}",
        "rmod": f"{b} mod {a}",
        "min": f"min({a},{b})",
        "max": f"max({a},{b})",
        "add_m1": f"{a}+{b}-1",
        "add_p1": f"{a}+{b}+1",
        "add_m2": f"{a}+{b}-2",
        "add_p2": f"{a}+{b}+2",
        "mul_m1": f"{a}*{b}-1",
        "mul_p1": f"{a}*{b}+1",
        "mul_m2": f"{a}*{b}-2",
        "mul_p2": f"{a}*{b}+2",
        "absdiff_m1": f"|{a}-{b}|-1",
        "absdiff_p1": f"|{a}-{b}|+1",
        "absdiff_m2": f"|{a}-{b}|-2",
        "absdiff_p2": f"|{a}-{b}|+2",
        "sub_m1": f"{a}-{b}-1",
        "sub_p1": f"{a}-{b}+1",
        "rsub_m1": f"{b}-{a}-1",
        "rsub_p1": f"{b}-{a}+1",
        "mul_half": f"({a}*{b})//2",
        "mul_double": f"{a}*{b}*2",
        "sq_diff": f"({a}-{b})^2",
        "sq_sum": f"({a}+{b})^2",
        "mul_plus_a": f"{a}*{b}+{a}",
        "mul_plus_b": f"{a}*{b}+{b}",
        "mul_minus_a": f"{a}*{b}-{a}",
        "mul_minus_b": f"{a}*{b}-{b}",
        "a2_plus_b": f"{a}^2+{b}",
        "a_plus_b2": f"{a}+{b}^2",
        "xor": f"{a} XOR {b}",
        "band": f"{a} AND {b}",
        "bor": f"{a} OR {b}",
        "sub_signed": f"{a}-{b}",
        "rsub_signed": f"{b}-{a}",
    }
    return t.get(ot, f"{ot}({a},{b})")


# ─────────────────────── base/mode-aware semantics ───────────────────────


def _reversed_mode(mode: str) -> bool:
    return S._is_reversed_digit_mode(mode)


def _glyph_sep(s: str) -> str:
    """Space-separate glyphs so BPE can't fuse operator into operand."""
    return " ".join(list(s))


def _two_digit(d0: int, d1: int, base: int, reversed_mode: bool) -> int:
    """Value of the 2-glyph operand. In a reversed-digit mode the operand digits
    are read least-significant-first, i.e. the two digits are swapped."""
    val = d0 * base + d1
    if reversed_mode:
        val = (val % base) * base + (val // base)
    return val


def _encode_mag(
    mag: int, base: int, reversed_mode: bool, rev_map: dict[int, str], target_len: int
) -> str | None:
    """Encode a non-negative magnitude into glyphs at a fixed length."""
    digs = S._int_to_base_digits(mag, base)
    if len(digs) > target_len:
        return None
    digs = [0] * (target_len - len(digs)) + digs
    if reversed_mode:
        digs = digs[::-1]
    out = []
    for d in digs:
        if d not in rev_map:
            return None
        out.append(rev_map[d])
    return "".join(out)


def _digit_order_note(
    mag: int, base: int, reversed_mode: bool, rev_map: dict[int, str], target_len: int
) -> str:
    """Compact LSB-first annotation for a result magnitude: in the reversed modes
    the result's digits are written least-significant-digit first, the step the
    student most often gets wrong. Spelling that order out explicitly turns the
    encoding into a local next-token function (the replicated reversed-digit win).

    Returns '' for standard mode (digit order is obvious there) or whenever it
    cannot reproduce the encoding exactly (defense in depth — never emit a note
    that disagrees with the verified `_encode_mag`)."""
    if not reversed_mode:
        return ""
    digs = S._int_to_base_digits(mag, base)
    if len(digs) > target_len:
        return ""
    digs = [0] * (target_len - len(digs)) + digs  # most-significant first, padded
    write_order = digs[::-1]  # reversed mode → least-significant first
    for d in write_order:
        if d not in rev_map:
            return ""
    ds = " ".join(str(d) for d in write_order)
    return f"  [digits least-significant first: {ds}]"


def _op_value(ot: str, L: int, R: int):
    """Return (signed: bool, mag: int) or None if the op is invalid for these
    operands. Mirrors _verify_solver.apply_eq's sign handling exactly."""
    fn = OPERATIONS.get(ot)
    if fn is None:
        return None
    v = fn(L, R)
    if v is None:
        return None
    if ot in SIGNED_OPS:
        return (v < 0), (-v if v < 0 else v)
    if ot == "neg_absdiff":
        return True, v
    if v < 0:
        return None
    return False, v


def _apply_eq(c0, c1, opc, c3, c4, ot, mapping, base, reversed_mode):
    """Compute one equation's right-hand side from the program. Returns a dict of
    intermediates + the encoded glyph string, or None if inconsistent."""
    rev_map = {d: g for g, d in mapping.items()}
    if ot in CONCAT_OPS:
        if ot == "concat_fwd":
            encoded = c0 + c1 + c3 + c4
        else:
            encoded = c3 + c4 + c0 + c1
        return {
            "concat": True,
            "encoded": encoded,
            "L_letters": c0 + c1,
            "R_letters": c3 + c4,
        }
    for c in (c0, c1, c3, c4):
        if c not in mapping:
            return None
    L = _two_digit(mapping[c0], mapping[c1], base, reversed_mode)
    R = _two_digit(mapping[c3], mapping[c4], base, reversed_mode)
    sv = _op_value(ot, L, R)
    if sv is None:
        return None
    signed, mag = sv
    return {
        "concat": False,
        "L": L,
        "R": R,
        "signed": signed,
        "mag": mag,
        "sign_glyph": opc if signed else "",
        "expr": _expr(ot, L, R),
        "rev_map": rev_map,
        "reversed_mode": reversed_mode,
        "base": base,
    }


def _encode_query(qrec, gold: str):
    """Given the query computation dict and gold answer, produce the encoded
    answer string and confirm it reproduces gold. Returns the string or None."""
    if qrec is None:
        return None
    if qrec["concat"]:
        return qrec["encoded"] if qrec["encoded"] == gold else None
    sign = qrec["sign_glyph"]
    if sign:
        if not gold.startswith(sign):
            return None
        mag_syms = gold[len(sign) :]
    else:
        mag_syms = gold
    enc = _encode_mag(
        qrec["mag"], qrec["base"], qrec["reversed_mode"], qrec["rev_map"], len(mag_syms)
    )
    if enc is None:
        return None
    full = sign + enc
    return full if full == gold else None


# ─────────────────────────── shared render bits ───────────────────────────


def _parse(prompt: str):
    return S.AliceEquationSolver._parse(prompt)


def _eligible_examples(examples, ops, mapping):
    """Examples the program can actually verify — mirrors the solver's own filter
    in `_derive_order` (lines 957-969): operator must be resolved, and every
    operand + result glyph (after stripping a leading sign glyph) must be a
    content symbol in the map. Examples whose result reuses the operator glyph or
    whose operator was never resolved are set aside, exactly as the solver does."""
    out = []
    for lhs, rhs in examples:
        if len(lhs) < 5:
            continue
        oc = lhs[2]
        if oc not in ops:
            continue
        has_sign = len(rhs) > 1 and rhs[0] == oc
        res_str = rhs[1:] if has_sign else rhs
        chars = (lhs[0], lhs[1], lhs[3], lhs[4]) + tuple(res_str)
        if not all(c in mapping for c in chars):
            continue
        out.append((lhs, rhs))
    return out


def _intro_lines(base: int, reversed_mode: bool, symbols: list[str]) -> list[str]:
    lines = [
        "I need to deduce Alice's secret rule from the worked examples, then "
        "apply it to the query.",
        "Each equation reads `g0 g1 OP g3 g4 = result`: the glyphs `g0 g1` form a "
        "two-digit number and `g3 g4` form another, in a secret base where every "
        "distinct glyph is one digit; `OP` is an operator glyph; the right side "
        "spells the result's digits in the same glyph alphabet.",
        f"Step 0 - the distinct content glyphs are {{ {' '.join(symbols)} }}: "
        f"that is {len(symbols)} glyphs, so the numbers are written in base {base}.",
    ]
    if reversed_mode:
        lines.append(
            "Note: in this rule the digits of every number (operands and result) "
            "are written least-significant-digit first, i.e. reversed."
        )
    return lines


def _verify_line(c0, c1, opc, c3, c4, rhs, rec, annotate=False) -> str:
    inp = _glyph_sep(c0 + c1 + opc + c3 + c4)
    if rec["concat"]:
        return f"  {inp} = {_glyph_sep(rec['encoded'])}  (matches {_glyph_sep(rhs)})"
    body = (
        f"{rec['L']} {opc} {rec['R']} -> {rec['expr']} = "
        f"{rec['sign_glyph']}{rec['mag']}"
    )
    note = ""
    if annotate:
        sign = rec["sign_glyph"]
        note = _digit_order_note(
            rec["mag"],
            rec["base"],
            rec["reversed_mode"],
            rec["rev_map"],
            len(rhs) - len(sign),
        )
    return f"  {inp}:  {body}  ->  {_glyph_sep(rhs)}{note}"


def _verify_block(eligible, ops, mapping, base, reversed_mode, annotate=False):
    """Render a verify line per eligible example, confirming each reproduces its
    rhs. Returns the list of lines, or None if any eligible example fails (a real
    semantic mismatch → the whole trace is dropped to preserve correctness)."""
    out = []
    for lhs, rhs in eligible:
        opc = lhs[2]
        ot = ops[opc]
        rec = _apply_eq(
            lhs[0], lhs[1], opc, lhs[3], lhs[4], ot, mapping, base, reversed_mode
        )
        if rec is None:
            return None
        if rec["concat"]:
            if rec["encoded"] != rhs:
                return None
        else:
            sign = rec["sign_glyph"]
            if sign and not rhs.startswith(sign):
                return None
            mag_syms = rhs[len(sign) :] if sign else rhs
            enc = _encode_mag(
                rec["mag"], base, reversed_mode, rec["rev_map"], len(mag_syms)
            )
            if enc is None or sign + enc != rhs:
                return None
        out.append(
            _verify_line(lhs[0], lhs[1], opc, lhs[3], lhs[4], rhs, rec, annotate)
        )
    return out


def _operator_id_lines(examples, ops, mapping, base, reversed_mode) -> list[str]:
    """One justification line per operator glyph: name it and check it on the
    first example that uses it."""
    lines = []
    by_op = {}
    for lhs, rhs in _eligible_examples(examples, ops, mapping):
        if lhs[2] not in by_op:
            by_op[lhs[2]] = (lhs, rhs)
    for opc in sorted(by_op):
        ot = ops[opc]
        lhs, rhs = by_op[opc]
        rec = _apply_eq(
            lhs[0], lhs[1], opc, lhs[3], lhs[4], ot, mapping, base, reversed_mode
        )
        phrase = _OP_PHRASE.get(ot, ot)
        if rec is None:
            lines.append(f"  Operator `{opc}`: {phrase}.")
        elif rec["concat"]:
            lines.append(
                f"  Operator `{opc}`: {phrase} -> "
                f"{_glyph_sep(lhs)} gives {_glyph_sep(rec['encoded'])}, "
                f"matching {_glyph_sep(rhs)}."
            )
        else:
            lines.append(
                f"  Operator `{opc}`: {phrase} ({rec['expr']} = "
                f"{rec['sign_glyph']}{rec['mag']}), which spells {_glyph_sep(rhs)}."
            )
    return lines


def _map_line(mapping: dict[str, int]) -> str:
    items = sorted(mapping.items(), key=lambda kv: kv[1])
    return "  " + ", ".join(f"{g}={d}" for g, d in items)


def _query_lines(
    query, ops, mapping, base, reversed_mode, gold, annotate=False
) -> tuple | None:
    if not query or len(query) < 5:
        return None
    c0, c1, opc, c3, c4 = query[0], query[1], query[2], query[3], query[4]
    ot = ops.get(opc)
    if ot is None:
        return None
    rec = _apply_eq(c0, c1, opc, c3, c4, ot, mapping, base, reversed_mode)
    enc = _encode_query(rec, gold)
    if enc is None:
        return None
    inp = _glyph_sep(query)
    if rec["concat"]:
        line = f"  {inp}: {_OP_PHRASE.get(ot, ot)} -> {_glyph_sep(enc)}."
    else:
        note = (
            _digit_order_note(
                rec["mag"],
                rec["base"],
                rec["reversed_mode"],
                rec["rev_map"],
                len(enc) - len(rec["sign_glyph"]),
            )
            if annotate
            else ""
        )
        line = (
            f"  {inp}: {rec['L']} {opc} {rec['R']} -> {rec['expr']} = "
            f"{rec['sign_glyph']}{rec['mag']}{note} -> {_glyph_sep(enc)}."
        )
    return [f"Step 4 - apply the rule to the query {inp}:", line], enc


# ─────────────────── digit-deduction Step 2 (base-10 std/alice) ───────────────────


def _derive(prompt, mapping, ops, mode, gold):
    """Run the solver's `_derive_order` and sanity-check it reconstructs the known
    map. Returns (order, trace) or None. Caller MUST gate to base==10 + mode in
    {standard, alice} (the only regime where `_derive_order` is sound).

    Concat operators are stripped before the call: `_derive_order` feeds every
    example through `compute_v`, and a concat op (OPERATIONS[...] is None) yields an
    EMPTY feasible set that zeroes its glyphs' domains — breaking the propagation
    for mixed-concat problems. Concat examples carry no digit constraint anyway
    (they just copy glyphs), so dropping them is sound; glyphs that appear only in
    concat examples simply commit as guesses via the distinct-digit rule."""
    arith_ops = {oc: ot for oc, ot in ops.items() if ot not in CONCAT_OPS}
    if not arith_ops:
        return None
    solver = S.AliceEquationSolver(prompt, answer_hint=gold)
    try:
        order, trace = solver._derive_order(mapping, arith_ops, mode)
    except (KeyError, ValueError, IndexError):
        return None
    if not order or len(order) != len(mapping) or not trace:
        return None
    for step in trace:
        if mapping.get(step["sym"]) != step["digit"]:
            return None
        if step["digit"] not in step["domain"]:
            return None
    return order, trace


def _step2_assert(trace, mapping) -> list[str]:
    """ASSERT style: state the MRV commit order with forced/guess labels. The map
    is fully forced by gold-conditioning, so this mostly reads as `only {d} left`."""
    lines = [
        "Step 2 - pin down each glyph's digit by elimination. I track the still-"
        "possible digits for each glyph and commit the most-constrained one first; "
        "each commit removes that digit from the others (all glyphs are distinct):"
    ]
    for step in trace:
        sym, dom, digit = step["sym"], step["domain"], step["digit"]
        if step["kind"] == "forced":
            lines.append(
                f"  `{sym}` has only {{{digit}}} left -> forced `{sym}` = {digit}."
            )
        else:
            dom_str = ",".join(str(d) for d in dom)
            lines.append(
                f"  `{sym}` is down to {{{dom_str}}}; the examples make {digit} the "
                f"consistent choice -> `{sym}` = {digit}."
            )
    return lines


def _step2_derive(trace, mapping) -> list[str]:
    """DERIVE style: render the genuine constraint propagation that finds the map.
    `trace[0]['per_example_feasibility']` holds the from-scratch propagation
    (determined={}, domains 0-9): each example keeps only the digit values that
    reproduce its result, intersected across examples until the digits are pinned."""
    pef = trace[0]["per_example_feasibility"]
    lines = [
        "Step 2 - work out each glyph's digit by elimination. Every glyph starts as "
        "any digit 0-9; for each example I keep only the values that make its "
        "arithmetic spell the shown result, intersecting the survivors across examples:"
    ]
    for p in pef["passes"]:
        for ex in p["examples"]:
            if not ex["narrowed"]:
                continue
            narrowed = [
                c
                for c in ex["syms_unknown"]
                if ex["intersected_after"][c] != ex["running_before_this_ex"][c]
                and len(ex["intersected_after"][c]) < 10
            ]
            if not narrowed:
                continue
            parts = "; ".join(
                f"`{c}` -> {{{','.join(str(d) for d in ex['intersected_after'][c])}}}"
                for c in narrowed
            )
            lines.append(
                f"  from {_glyph_sep(ex['lhs'])} = {_glyph_sep(ex['rhs'])}: {parts}"
            )
    final = pef["final"]
    singles = [s for s in sorted(mapping) if len(final[s]) == 1]
    multi = [s for s in sorted(mapping) if len(final[s]) > 1]
    if singles:
        lines.append(
            "  These intersections pin: "
            + ", ".join(f"{s}={final[s][0]}" for s in singles)
            + "."
        )
    if multi:
        plural = "s" if len(multi) > 1 else ""
        lines.append(
            f"  The remaining glyph{plural} "
            + ", ".join(
                f"`{s}` (from {{{','.join(str(d) for d in final[s])}}})" for s in multi
            )
            + " resolve by the distinct-digit rule and consistency with the examples: "
            + ", ".join(f"{s}={mapping[s]}" for s in multi)
            + "."
        )
    return lines


def _binding_example(step):
    """The equation that drops this glyph's running domain to a single value, read
    from the solver's own per-example forward-checking. Returns (lhs, rhs) of the
    first example (document order) after which `intersected_after[sym] == [digit]` —
    the constraint that *completes* the glyph's deduction — or None when the glyph is
    pinned only jointly / by the distinct-digit rule (no single binding example)."""
    pef = step.get("per_example_feasibility")
    if not pef:
        return None
    target = [step["digit"]]
    for p in pef["passes"]:
        for ex in p["examples"]:
            if ex["intersected_after"].get(step["sym"]) == target:
                return ex["lhs"], ex["rhs"]
    return None


def _step2_derive_inductive(trace, header=None) -> list[str]:
    """BOUNDED, step-count-INDEPENDENT map deduction: exactly one fixed-shape row
    per glyph, in the solver's MRV commit order. Each row is the SAME local
    state-update — (digits already fixed) intersected with (what the examples still
    allow) -> commit the most-constrained glyph — so the row count tracks the glyph
    count (<=10), never the search depth. This is the inductive-scratchpad fix for
    the variable-length `_step2_derive` (which risks overfitting trace length OOD
    per 2406.06467).

    To keep it a genuine *search* (not a bare assertion of the answer), each forced
    row cites the specific equation that completes the glyph's deduction (the binding
    example). Every value comes straight from the solver's own forward-checking:
    `step['domain']` is the post-propagation domain at commit, `determined_before`
    the digits already fixed, the binder its per-example intersection — so each row
    is correct by construction.

    `header` overrides the section header (so derive_search can label this Step 1)."""
    lines = [
        header
        or (
            "Step 2 - pin down the glyph -> digit map one glyph at a time. I keep the "
            "set of digits already fixed; for the next glyph I read the examples in "
            "order, keeping only the digits still consistent (every glyph is a distinct "
            "digit), and commit the most-constrained glyph each round:"
        )
    ]
    for step in trace:
        sym, digit = step["sym"], step["digit"]
        used = sorted(step["determined_before"].values())
        used_str = ",".join(str(d) for d in used) if used else "none yet"
        dom_str = ",".join(str(d) for d in step["domain"])
        if step["kind"] == "forced":
            binder = _binding_example(step)
            if binder is not None:
                lhs, rhs = binder
                lines.append(
                    f"  `{sym}`: with {{{used_str}}} already used, working the examples "
                    f"in order, {_glyph_sep(lhs)} = {_glyph_sep(rhs)} leaves only "
                    f"{{{dom_str}}} for `{sym}` -> forced `{sym}` = {digit}."
                )
            else:
                lines.append(
                    f"  `{sym}`: with {{{used_str}}} already used, the examples together "
                    f"leave only {{{dom_str}}} for `{sym}` -> forced `{sym}` = {digit}."
                )
        else:
            lines.append(
                f"  `{sym}`: with {{{used_str}}} already used, the examples leave "
                f"{{{dom_str}}} for `{sym}`, of which {digit} is the value consistent "
                f"with every example -> `{sym}` = {digit}."
            )
    return lines


# ───────────── operator enumerate-and-test (derive_search, long) ─────────────
# Mirrors the proven LONG shape of reasoning_equation_numeric (the student's closest
# analog task, learned ~87.5%): once the digits are known, IDENTIFY each operator by
# trying candidate operations and showing wrong/match for each. Common ops are always
# enumerated (equation tries all); rare ops are appended only up to the true op so the
# match always appears without listing all 47. Every test reuses the verified
# base/mode-aware semantics (_op_value/_encode_mag), so it can never disagree.

_COMMON_SEARCH_OPS = [
    "add",
    "sub",
    "rsub",
    "absdiff",
    "mul",
    "concat_fwd",
    "concat_rev",
    "mod",
    "rmod",
    "gcd",
    "lcm",
    "fdiv",
    "rdiv",
    "min",
    "max",
]
_RARE_SEARCH_OPS = [
    "add_p1",
    "add_m1",
    "add_p2",
    "add_m2",
    "mul_p1",
    "mul_m1",
    "mul_p2",
    "mul_m2",
    "sub_p1",
    "sub_m1",
    "rsub_p1",
    "rsub_m1",
    "absdiff_p1",
    "absdiff_m1",
    "absdiff_p2",
    "absdiff_m2",
    "mul_double",
    "mul_half",
    "sq_diff",
    "sq_sum",
    "mul_plus_a",
    "mul_plus_b",
    "mul_minus_a",
    "mul_minus_b",
    "a2_plus_b",
    "a_plus_b2",
    "neg_absdiff",
    "sub_signed",
    "rsub_signed",
]


def _candidate_sequence(true_ot: str) -> list[str]:
    """Common ops (always shown, equation-style), then rare ops up to and INCLUDING
    the true op so the match always appears without dumping all 47 candidates."""
    seq = list(_COMMON_SEARCH_OPS)
    if true_ot not in seq:
        for ot in _RARE_SEARCH_OPS:
            seq.append(ot)
            if ot == true_ot:
                break
        if true_ot not in seq:  # safety: op outside both lists
            seq.append(true_ot)
    return seq


def _candidate_on_example(lhs, rhs, cand, mapping, base, reversed_mode, rev_map):
    """Apply candidate op `cand` to one example; return (ok, shown). ok is True iff it
    reproduces rhs exactly. `shown` is the human arithmetic string. All values come
    from the shared verified semantics — never a re-implemented operation."""
    c0, c1, opc, c3, c4 = lhs[0], lhs[1], lhs[2], lhs[3], lhs[4]
    if cand in CONCAT_OPS:
        enc = (c0 + c1 + c3 + c4) if cand == "concat_fwd" else (c3 + c4 + c0 + c1)
        return (enc == rhs, f"glue the glyphs -> {_glyph_sep(enc)}")
    if any(c not in mapping for c in (c0, c1, c3, c4)):
        return (False, "an operand glyph is not in the map")
    left = _two_digit(mapping[c0], mapping[c1], base, reversed_mode)
    right = _two_digit(mapping[c3], mapping[c4], base, reversed_mode)
    sv = _op_value(cand, left, right)
    if sv is None:
        return (False, f"{_expr(cand, left, right)} is undefined/negative here")
    signed, mag = sv
    if signed:
        if not (rhs and rhs[0] == opc):
            return (
                False,
                f"{_expr(cand, left, right)} = -{mag}, but no sign glyph here",
            )
        enc_mag = _encode_mag(mag, base, reversed_mode, rev_map, len(rhs) - 1)
        enc = (opc + enc_mag) if enc_mag is not None else None
    else:
        enc = _encode_mag(mag, base, reversed_mode, rev_map, len(rhs))
    shown = f"{_expr(cand, left, right)} = {'-' if signed else ''}{mag}"
    if enc is None:
        return (False, f"{shown}, which doesn't fit the {len(rhs)}-glyph result")
    return (enc == rhs, f"{shown} -> spells {_glyph_sep(enc)}")


def _eval_candidate(cand, exs, mapping, base, reversed_mode, rev_map):
    """Test candidate across ALL of an operator's examples (it is the rule only if it
    matches every one). Returns (all_ok, shown) where shown is the first FAILING
    example's arithmetic, or the first example's if all pass."""
    first = None
    for lhs, rhs in exs:
        ok, shown = _candidate_on_example(
            lhs, rhs, cand, mapping, base, reversed_mode, rev_map
        )
        if first is None:
            first = shown
        if not ok:
            return (False, shown)
    return (True, first)


def _op_search_block(examples, ops, mapping, base, reversed_mode) -> list[str] | None:
    """Equation_numeric-style enumerate-and-test operator identification. Returns the
    lines, or None if the named (true) op fails to match its own examples (defensive:
    that should never happen since the program reproduces every gold)."""
    rev_map = {d: g for g, d in mapping.items()}
    by_op: dict[str, list] = {}
    for lhs, rhs in _eligible_examples(examples, ops, mapping):
        by_op.setdefault(lhs[2], []).append((lhs, rhs))
    if not by_op:
        return None
    lines = [
        "Step 2 - now identify each operator: with the digits known, try candidate "
        "operations and keep the one matching every example (a negative result is "
        "written with the operator glyph in front):"
    ]
    for opc in sorted(by_op):
        exs = by_op[opc]
        true_ot = ops[opc]
        shown_exs = ", ".join(
            f"{_glyph_sep(lhs)} = {_glyph_sep(rhs)}" for lhs, rhs in exs
        )
        lines.append(f"  Operator `{opc}`  [{shown_exs}]:")
        for cand in _candidate_sequence(true_ot):
            ok, shown = _eval_candidate(
                cand, exs, mapping, base, reversed_mode, rev_map
            )
            if cand == true_ot:
                if not ok:
                    return None  # program/render inconsistency -> drop the trace
                tag = "MATCHES every example -> this is the operator"
            elif ok:
                tag = "also fits (kept the named operator below)"
            else:
                tag = "no"
            lines.append(f"    {_OP_PHRASE.get(cand, cand)}: {shown} -> {tag}")
    return lines


# ─────────────────────────── public renderer ───────────────────────────


def render_with_tier(prompt, mapping, ops, mode, base, gold, style="assert"):
    """Like `render`, but returns (text, actual_tier) so callers can report which
    Step-2 form was really used. actual_tier is "assert"/"derive" only when the
    deduction genuinely rendered (base-10 std/alice + `_derive_order` sound),
    otherwise "lean". Returns None if the program fails to reproduce gold."""
    examples, query = _parse(prompt)
    reversed_mode = _reversed_mode(mode)
    symbols = sorted(mapping.keys())
    annotate = style in ("derive_inductive", "derive_search")

    eligible = _eligible_examples(examples, ops, mapping)
    if not eligible:
        return None
    verify_lines = _verify_block(eligible, ops, mapping, base, reversed_mode, annotate)
    if verify_lines is None:
        return None
    q = _query_lines(query, ops, mapping, base, reversed_mode, gold, annotate)
    if q is None:
        return None
    query_block, _enc = q

    # derive_search: the LONG, equation_numeric-style arm — map (bounded) FIRST, then
    # an enumerate-and-test operator id on the known digits, then verify + apply.
    if style == "derive_search":
        op_lines = _op_search_block(examples, ops, mapping, base, reversed_mode)
        if op_lines is None:
            return None
        map_step = None
        if base == 10 and mode in ("standard", "alice"):
            d = _derive(prompt, mapping, ops, mode, gold)
            if d is not None:
                _order, trace = d
                map_step = _step2_derive_inductive(
                    trace,
                    header=(
                        "Step 1 - first recover the glyph -> digit map. I keep the "
                        "digits already fixed and, for the next glyph, keep only the "
                        "values the examples still allow (every glyph is a distinct "
                        "digit), committing the most-constrained glyph each round:"
                    ),
                )
        if map_step is None:
            map_step = [
                "Step 1 - the glyph -> digit map consistent with the examples is:",
                _map_line(mapping),
            ]
        lines = _intro_lines(base, reversed_mode, symbols)
        lines.append("")
        lines += map_step
        lines.append("")
        lines += op_lines
        lines.append("")
        lines.append("Step 3 - check the rule against every example:")
        lines += verify_lines
        lines.append("")
        lines += query_block
        lines.append("")
        lines.append(f"So the answer is \\boxed{{{gold}}}")
        return "\n".join(lines), "derive_search"

    step2, actual = None, "lean"
    if (
        style in ("assert", "derive", "derive_inductive")
        and base == 10
        and mode in ("standard", "alice")
    ):
        d = _derive(prompt, mapping, ops, mode, gold)
        if d is not None:
            _order, trace = d
            if style == "assert":
                step2 = _step2_assert(trace, mapping)
            elif style == "derive":
                step2 = _step2_derive(trace, mapping)
            else:
                step2 = _step2_derive_inductive(trace)
            actual = style
    if step2 is None:
        step2 = [
            "Step 2 - reading the matched digits, the glyph -> digit map is:",
            _map_line(mapping),
        ]

    lines = _intro_lines(base, reversed_mode, symbols)
    lines.append("")
    lines.append("Step 1 - identify each operator glyph by testing it on an example:")
    lines += _operator_id_lines(examples, ops, mapping, base, reversed_mode)
    lines.append("")
    lines += step2
    lines.append("")
    lines.append(
        f"Step 3 - check the {'completed map' if actual != 'lean' else 'rule'} "
        "against every example:"
    )
    lines += verify_lines
    lines.append("")
    lines += query_block
    lines.append("")
    lines.append(f"So the answer is \\boxed{{{gold}}}")
    return "\n".join(lines), actual


def render(prompt, mapping, ops, mode, base, gold, style="assert") -> str | None:
    """Render a correct CoT trace ending in `\\boxed{gold}`, or None if the program
    fails to reproduce gold (so a bad row can never reach training).

    style:
      "lean"   — state the matched map, verify every example, apply (universal).
      "assert" — base-10 std/alice: MRV forced/guess scratchpad; else lean.
      "derive" — base-10 std/alice: genuine constraint-propagation deduction; else
                 lean. (the A/B counterpart to "assert" — see _step2_derive)
      "derive_inductive" — bounded one-row-per-glyph map + LSB-first encoding
                 annotations (reversed modes); the step-count-independent recast of
                 "derive" (see _step2_derive_inductive). Falls back to lean map.
      "derive_search" — LONG, equation_numeric-style: map FIRST (bounded), then an
                 enumerate-and-test operator id on the known digits showing wrong/match
                 per candidate (the proven ~87.5% shape), then verify + apply. Universal
                 (op-search works in every base/mode); map uses derive_inductive where
                 sound, else lean. (see _op_search_block)
    The assert/derive/derive_inductive/lean styles share Step 1 (operator id), Step 3
    (verify), Step 4 (apply); derive_search reorders to map -> op-search -> verify ->
    apply."""
    r = render_with_tier(prompt, mapping, ops, mode, base, gold, style)
    return r[0] if r is not None else None


def render_lean(prompt, mapping, ops, mode, base, gold) -> str | None:
    """Universal base/mode-aware renderer (no digit-deduction scratchpad)."""
    return render(prompt, mapping, ops, mode, base, gold, style="lean")
