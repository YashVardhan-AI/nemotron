"""Forward-gen training CoT for the bit_manipulation lever.

Targets the families the corpus does NOT teach well: the complex 3-input tier
(majority/choice — the solver `reasoners/bit_manipulation.py` cannot even express
it, so no good real traces exist) and heterogeneous per-column pairwise rules.

Unlike cryptarithm, bit is DECOMPOSABLE-LOCAL: each output bit is an independent
function of a few input bits with NO global coupling, so a greedy decoder can
emit one bit's reasoning and forget it. The trace exploits exactly that: deduce
each output bit as a local function, verify it reproduces the full column across
EVERY example (sound by construction), then apply each bit to the query with the
substituted values shown. Kept SHORT (~one line per bit) to avoid the long
high-loss gradient that made the cryptarithm induct traces crash neighbors.
"""

from __future__ import annotations

from reasoners.bit_rule import bit_at, columns, eval_column, sample_problem
from reasoners.store_types import Problem


def make_bit_trace_problem(
    seed: int, difficulty: int
) -> tuple[Problem, str, list[tuple]]:
    """A new-rule bit problem + its verified answer + the rule columns, from the
    SHARED sampler (same instances the val generator measures)."""
    p = sample_problem(seed, difficulty)
    return p, p.answer, columns(seed)


def _expr(col: tuple) -> str:
    """Symbolic form of a column rule, e.g. 'in[2] XOR (NOT in[5])'."""
    kind = col[0]
    if kind == "route":
        _, p, neg = col
        return f"NOT in[{p}]" if neg else f"in[{p}]"
    if kind == "pair":
        _, op, p, q = col
        base = op.split("-")[0]
        rhs = f"(NOT in[{q}])" if op.endswith("-NOT") else f"in[{q}]"
        return f"in[{p}] {base} {rhs}"
    if kind == "maj":
        _, p, q, r = col
        return f"MAJ(in[{p}], in[{q}], in[{r}])"
    _, p, q, r = col  # choice
    return f"in[{q}] if in[{p}]=1 else in[{r}]"


def _apply_expr(col: tuple, x: int) -> str:
    """Substituted computation of a column rule on input *x*, ending in '= bit'."""
    kind = col[0]
    if kind == "route":
        _, p, neg = col
        b = bit_at(x, p)
        return f"NOT {b} = {1 - b}" if neg else f"in[{p}]={b}"
    if kind == "pair":
        _, op, p, q = col
        base = op.split("-")[0]
        u, v = bit_at(x, p), bit_at(x, q)
        vv = 1 - v if op.endswith("-NOT") else v
        r = (u & vv) if base == "AND" else (u | vv) if base == "OR" else (u ^ vv)
        vstr = f"(NOT {v})={vv}" if op.endswith("-NOT") else f"{v}"
        return f"{u} {base} {vstr} = {r}"
    if kind == "maj":
        _, p, q, r = col
        a, b, c = bit_at(x, p), bit_at(x, q), bit_at(x, r)
        return f"MAJ({a},{b},{c}) = {1 if a + b + c >= 2 else 0}"
    _, p, q, r = col  # choice
    sel = bit_at(x, p)
    chosen = q if sel else r
    return f"in[{p}]={sel} so take in[{chosen}]={bit_at(x, chosen)}"


def reasoning_bit_perbit(
    problem: Problem, answer: str, cols: list[tuple]
) -> str | None:
    """Per-bit deduce -> verify-on-every-example -> apply CoT.

    Returns None (skip) if the rule does not reproduce the answer (should never
    happen for forward-gen; a defensive gate so only sound traces ship).
    """
    ex = [(int(e.input_value, 2), e.output_value) for e in problem.examples]
    q = int(problem.question, 2)

    lines = [
        "Each output bit is an independent function of the input bits "
        "(positions 0-7, left to right). I deduce each bit's rule, verify it "
        "reproduces that column in every example, then apply it to the query.",
        "",
    ]
    for j in range(8):
        col = cols[j]
        predicted = [eval_column(col, xi) for xi, _ in ex]
        actual = [int(yb[j]) for _, yb in ex]
        if predicted != actual:
            return None  # unsound; skip
        cols_str = "".join(str(b) for b in predicted)
        lines.append(
            f"Bit {j} = {_expr(col)} -> column {cols_str} matches all examples."
        )

    lines.append("")
    lines.append(f"Apply to the query {problem.question}:")
    out_bits: list[str] = []
    for j in range(8):
        col = cols[j]
        r = eval_column(col, q)
        out_bits.append(str(r))
        lines.append(f"  bit {j}: {_apply_expr(col, q)} -> {r}")

    out = "".join(out_bits)
    if out != answer:
        return None  # unsound; skip
    lines.append("")
    lines.append(f"Reading the bits in order gives {out}.")
    return "\n".join(lines)
