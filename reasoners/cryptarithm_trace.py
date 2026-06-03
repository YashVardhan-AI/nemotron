"""Genuine-deduction CoT for the arithmetic-cipher cryptarithm family.

Unlike reasoners/cryptarithm.py (which narrates an already-known concatenation
rule), this teaches *deduction*: recognize the 5-char structure, resolve each
operator (concat/rev_concat structurally; arithmetic narrowed by output length
and confirmed by value -- with the wrong candidates explicitly ruled out), read
off the consistent glyph->digit map, verify every example, then apply to the
query. The trace is rendered from the decision log of the instrumented solver
(reasoners/cryptarithm_deduce_core), so the problem is uniquely deducible and the
boxed answer is correct by construction. Every numeric claim is recomputed here,
so each step is sound.
"""

from __future__ import annotations

from reasoners.cryptarithm_deduce_core import sample_solvable, solve_problem
from reasoners.cryptarithm_rule import num_to_digits, render_prompt
from reasoners.store_types import Example, Problem

_OP_WORD = {
    "add": "sum",
    "abs_diff": "absolute difference",
    "mul": "product",
    "concat": "concatenation",
    "rev_concat": "reverse concatenation",
}
_OP_FN = {
    "add": lambda a, b: a + b,
    "abs_diff": lambda a, b: abs(a - b),
    "mul": lambda a, b: a * b,
    "concat": lambda a, b: a * 100 + b,
    "rev_concat": lambda a, b: b * 100 + a,
}
_ARITH = ("add", "abs_diff", "mul")
_MAX_LEN = {"add": 3, "abs_diff": 2, "mul": 4}  # max result digits per arith op


def make_trace_problem(seed: int, difficulty: int) -> tuple[Problem, str]:
    """A uniquely-deducible problem + its verified answer, from the SHARED
    solver-filtered sampler (same instances the val generator measures)."""
    _rule, raw, q_input, q_answer = sample_solvable(seed, difficulty)
    examples = [Example(input_value=i, output_value=o) for i, o in raw]
    problem = Problem(
        id=f"train-cryptarithm-{seed}",
        category="cryptarithm_deduce",
        examples=examples,
        question=q_input,
        answer=q_answer,
        prompt=render_prompt(raw, q_input, wrapper_index=0),
    )
    return problem, q_answer


def _data(problem: Problem) -> dict:
    return {
        "examples": [
            {"input_value": e.input_value, "output_value": e.output_value}
            for e in problem.examples
        ],
        "question": problem.question,
    }


def _expr(name: str, left: int, right: int) -> str:
    if name == "add":
        return f"{left} + {right}"
    if name == "abs_diff":
        return f"|{left} - {right}|"
    if name == "mul":
        return f"{left} * {right}"
    if name == "concat":
        return f"concat({left}, {right})"
    return f"rev_concat({left}, {right})"


def _result_digits(name: str, left: int, right: int) -> tuple[int, ...]:
    val = _OP_FN[name](left, right)
    if name in ("concat", "rev_concat"):
        return (val // 1000, (val // 100) % 10, (val // 10) % 10, val % 10)
    return num_to_digits(val)


def _wordlist(names: list[str]) -> str:
    words = [_OP_WORD[n] for n in names]
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + " and " + words[-1]


def _resolve_op_line(op_rec: dict, verifies: list[dict], maprec: dict | None) -> str:
    glyph, name, how = op_rec["op"], op_rec["name"], op_rec["how"]
    out_len = op_rec["out_len"]
    if how == "structural":
        order = "in order" if name == "concat" else "in reverse order"
        return (
            f"- Operator '{glyph}': every output is the input glyphs {order}, "
            f"so it is the {_OP_WORD[name]}."
        )
    if maprec is None or name not in _ARITH:
        return f"- Operator '{glyph}' is arithmetic: the {_OP_WORD[name]}."
    v = next(
        (x for x in verifies if x["inp"][2] == glyph and not x.get("structural")),
        None,
    )
    if v is None:
        return f"- Operator '{glyph}' is the {_OP_WORD[name]}."
    left, right, out = v["left"], v["right"], v["out"]
    d2s = maprec["digit_to_sym"]
    ruled_len = [n for n in _ARITH if out_len > _MAX_LEN[n]]
    feasible = [n for n in _ARITH if out_len <= _MAX_LEN[n]]
    parts = [
        f"- Operator '{glyph}': example {v['inp']} = {out} has a {out_len}-digit output"
    ]
    if ruled_len:
        parts.append(f", which rules out the {_wordlist(ruled_len)}")
    parts.append(". Testing the rest: ")
    tests = []
    for n in feasible:
        enc = "".join(d2s.get(d, "?") for d in _result_digits(n, left, right))
        if n == name:
            tests.append(
                f"{_OP_WORD[n]} gives {_expr(n, left, right)} = "
                f"{_OP_FN[n](left, right)} -> {enc} (matches)"
            )
        else:
            tests.append(f"{_OP_WORD[n]} -> {enc} (rules it out)")
    parts.append("; ".join(tests))
    parts.append(f". So '{glyph}' is the {_OP_WORD[name]}.")
    return "".join(parts)


def _verify_line(v: dict) -> str:
    if v.get("structural"):
        order = "in order" if v["name"] == "concat" else "reversed"
        return (
            f"{v['inp']} = {v['out']}: '{v['inp'][2]}' is the {_OP_WORD[v['name']]}, "
            f"output is the operand glyphs {order}. matches."
        )
    val = _OP_FN[v["name"]](v["left"], v["right"])
    status = "matches" if v["ok"] else "MISMATCH"
    return (
        f"{v['inp']} = {v['out']}: {_expr(v['name'], v['left'], v['right'])} = {val} "
        f"-> {v['check']}. {status}."
    )


def _apply_line(qrec: dict) -> str:
    name, q, ans = qrec["name"], qrec["inp"], qrec["answer"]
    if "left" in qrec:
        val = _OP_FN[name](qrec["left"], qrec["right"])
        return f"{q}: {_expr(name, qrec['left'], qrec['right'])} = {val} -> {ans}."
    order = "in order" if name == "concat" else "reversed"
    return (
        f"{q}: '{q[2]}' is the {_OP_WORD[name]}, so the answer is the operand "
        f"glyphs {order} -> {ans}."
    )


def reasoning_cryptarithm_arith(problem: Problem, answer: str) -> str | None:
    """Render a genuine-deduction CoT, or None if the solver disagrees (which
    should not happen for sample_solvable instances)."""
    ans, (_mapping, _op_info), log = solve_problem(_data(problem), trace=True)
    if ans != answer:
        return None

    # Skip operators the solver did not resolve (arith glyphs irrelevant to a
    # concat-shortcut query) -- they are not needed to answer it.
    op_recs = [r for r in log if r["kind"] == "op" and r["name"] is not None]
    maprec = next((r for r in log if r["kind"] == "map"), None)
    verifies = [r for r in log if r["kind"] == "verify"]
    qrec = next(r for r in log if r["kind"] == "query")

    lines = [
        "I need to deduce the secret rule from the examples, then apply it.",
        (
            "Each input is `s0 s1 op s3 s4`: two glyphs form a two-digit number, "
            "then an operator glyph, then two more glyphs form a second two-digit "
            "number. The output spells the result's digits in the same secret "
            "alphabet. So I must find (a) what each operator does and (b) each "
            "glyph's digit."
        ),
        "",
        "Step 1 - identify each operator:",
    ]
    lines += [_resolve_op_line(r, verifies, maprec) for r in op_recs]
    lines.append("")
    if maprec:
        lines.append("Step 2 - the consistent glyph->digit map:")
        d2s = maprec["digit_to_sym"]
        lines.append("  " + ", ".join(f"{d2s[d]}={d}" for d in range(10) if d in d2s))
        lines.append("")
    lines.append("Step 3 - check the rule against every example:")
    lines += ["  " + _verify_line(v) for v in verifies]
    lines.append("")
    lines.append(f"Step 4 - apply to the query {problem.question}:")
    lines.append("  " + _apply_line(qrec))
    lines.append("")
    lines.append(f"So the answer is \\boxed{{{answer}}}")
    return "\n".join(lines)


def reasoning_cryptarithm_propagate(problem: Problem, answer: str) -> str | None:
    """Alternative CoT style (Phase 4.3): leaner, single-pass-executable
    compute-and-correct. States the map inline, shows ONE sound wrong-then-fixed
    operator guess, then checks every example and applies. Same boxed answer as
    reasoning_cryptarithm_arith -- the two styles are A/B'd on val (Phase 6)."""
    ans, (_mapping, _op_info), log = solve_problem(_data(problem), trace=True)
    if ans != answer:
        return None

    maprec = next((r for r in log if r["kind"] == "map"), None)
    verifies = [r for r in log if r["kind"] == "verify"]
    qrec = next(r for r in log if r["kind"] == "query")
    q = problem.question
    qop = q[2]
    qname = qrec["name"]

    lines = [
        "I'll work out the secret rule by computing the examples and fixing "
        "whatever does not check out.",
        "Each input `s0 s1 op s3 s4` is two two-digit numbers joined by an "
        "operator glyph; the output spells the result in the same secret alphabet.",
    ]

    if maprec is not None and "left" in qrec:
        d2s = maprec["digit_to_sym"]
        v = next(
            (x for x in verifies if x["inp"][2] == qop and not x.get("structural")),
            None,
        )
        if v is not None:
            out_len = len(v["out"])
            feasible = [n for n in _ARITH if out_len <= _MAX_LEN[n]]
            alt = next((n for n in feasible if n != qname), None)
            if alt is not None:
                enc_alt = "".join(
                    d2s.get(d, "?") for d in _result_digits(alt, v["left"], v["right"])
                )
                lines.append(
                    f"Operator '{qop}': first try the {_OP_WORD[alt]} on {v['inp']}: "
                    f"{_expr(alt, v['left'], v['right'])} = "
                    f"{_OP_FN[alt](v['left'], v['right'])} -> {enc_alt}, but the "
                    f"example shows {v['out']} - wrong. The {_OP_WORD[qname]} gives "
                    f"{_expr(qname, v['left'], v['right'])} = "
                    f"{_OP_FN[qname](v['left'], v['right'])} -> {v['out']} - correct."
                )
            else:
                lines.append(
                    f"Operator '{qop}' must be the {_OP_WORD[qname]}: it is the only "
                    f"operation whose result has {out_len} digits and matches "
                    f"{v['inp']} = {v['out']}."
                )
        lines.append(
            "Reading the matching glyphs, the map is: "
            + ", ".join(f"{d2s[d]}={d}" for d in range(10) if d in d2s)
            + "."
        )
        lines.append("Checking every example:")
        lines += ["  " + _verify_line(vv) for vv in verifies]
    else:
        order = "in order" if qname == "concat" else "reversed"
        lines.append(
            f"Operator '{qop}': I first expect arithmetic, but in every example the "
            f"output is just the input glyphs {order} - so it is the "
            f"{_OP_WORD[qname]}, no digit values needed."
        )
        lines += ["  " + _verify_line(vv) for vv in verifies]

    lines.append("Apply: " + _apply_line(qrec))
    lines.append(f"\\boxed{{{answer}}}")
    return "\n".join(lines)
