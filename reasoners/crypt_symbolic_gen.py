"""P2: rich SYNTHETIC cryptarithm generator (distribution-matched, holdout-safe).

The repo's existing forward-gen generator (`cryptarithm_rule` /
`cryptarithm_deduce_core`) is the OLD NARROW MODEL — 5 ops, base 10, standard order
only (the 12% slice the symbolic-solver breakthrough escaped). This module is the
RICH replacement: it samples a program from the EMPIRICAL marginals of the 725
solver-solved real puzzles (base/mode/op-type/operator-count/example-count), builds a
puzzle by REUSING the verified renderer semantics (`_op_value`/`_encode_mag`), and
keeps it ONLY if the rich solver recovers the planted answer by PURE INFERENCE (no
gold hint) — which both guarantees a well-posed instance and bounds the synthetic set
to the solver's ~72% learnable region.

Pipeline (per puzzle): sample program -> build examples+query -> compute golds via the
shared semantics -> unhinted-solve learnability filter -> prompt-hash dedup vs real
train -> render (`derive_search` by default) -> round-trip gate. `generate_dataset`
returns puzzles + a stats dict; `build_crypt_symbolic_examples` is the corpus emitter
(drop-in shape match for `corpus.build_cryptarithm_examples`).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from reasoners import crypt_symbolic_trace as R  # also puts the solver src on sys.path

import solver_eq_symbolic as S  # noqa: E402  (sys.path set by the import above)

_PARQUET = (
    Path(__file__).resolve().parent.parent
    / "kaggle-nemotron-equation-symbolic-main"
    / "kaggle-nemotron-equation-symbolic-main"
    / "data"
    / "solver_results.parquet"
)

# Real glyph inventories (measured from the 725 solved puzzles).
CONTENT_GLYPHS = list("!\"#$%&'()/:<>?@[\\]^`{|}")
OP_GLYPHS = list("!\"#$%&'()*+-/:<>?@[\\]^`{|}")
RESULT_LEN_CAP = 4  # real result lengths top out at 4 glyphs
HEADER = (
    "In Alice's Wonderland, a secret set of transformation rules is applied to "
    "equations. Below are a few examples:"
)


def load_marginals(parquet: Path | None = None) -> dict:
    """Empirical marginals from the gold-conditioned, solver-correct real pool."""
    import pandas as pd

    df = pd.read_parquet(parquet or _PARQUET)
    pool = df[df["conditioned_on_answer"] & df["solver_correct"]]
    pool = pool[pool["solver_mode"].notna() & pool["solver_radix"].notna()]
    base, mode, optypes, n_ops, n_ex = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    for _, r in pool.iterrows():
        base[int(r["solver_radix"])] += 1
        mode[str(r["solver_mode"])] += 1
        ops = json.loads(r["solver_ops"])
        n_ops[len(ops)] += 1
        for ot in ops.values():
            optypes[ot] += 1
        n_ex[sum(1 for ln in r["prompt"].split("\n") if " = " in ln)] += 1
    arith = Counter({k: v for k, v in optypes.items() if k not in R.CONCAT_OPS})
    return {
        "base": base,
        "mode": mode,
        "optypes": optypes,
        "arith_optypes": arith,
        "n_ops": n_ops,
        "n_ex": n_ex,
    }


def _wchoice(rng: random.Random, counter: Counter):
    """Weighted pick from a Counter."""
    items = list(counter.items())
    total = sum(w for _, w in items)
    x = rng.random() * total
    acc = 0.0
    for k, w in items:
        acc += w
        if x <= acc:
            return k
    return items[-1][0]


def _compute_rhs(c0, c1, opc, c3, c4, ot, mapping, base, reversed_mode):
    """RHS glyph string for one equation at NATURAL length, or None if the op is
    invalid for these operands or the result overflows RESULT_LEN_CAP. Reuses the
    renderer's verified semantics — never re-implements an operation."""
    if ot in R.CONCAT_OPS:
        return (c0 + c1 + c3 + c4) if ot == "concat_fwd" else (c3 + c4 + c0 + c1)
    rev_map = {d: g for g, d in mapping.items()}
    left = R._two_digit(mapping[c0], mapping[c1], base, reversed_mode)
    right = R._two_digit(mapping[c3], mapping[c4], base, reversed_mode)
    sv = R._op_value(ot, left, right)
    if sv is None:
        return None
    signed, mag = sv
    natural_len = max(1, len(S._int_to_base_digits(mag, base)))
    if natural_len > RESULT_LEN_CAP:
        return None
    enc = R._encode_mag(mag, base, reversed_mode, rev_map, natural_len)
    if enc is None:
        return None
    return (opc + enc) if signed else enc


def _gen_example(rng, mapping, opc, ot, base, reversed_mode, content, want):
    """One example for operator `opc`, preferring operand glyphs from `want` (the
    not-yet-covered content glyphs) so the whole puzzle uses all `base` glyphs.
    Consumes `want` only on success; restores it on a failed attempt. Returns
    (lhs, rhs) or None after a bounded number of operand retries."""
    for _ in range(24):
        taken, slots = [], []
        for _ in range(4):
            if want:
                g = want.pop()
                taken.append(g)
                slots.append(g)
            else:
                slots.append(rng.choice(content))
        c0, c1, c3, c4 = slots
        rhs = _compute_rhs(c0, c1, opc, c3, c4, ot, mapping, base, reversed_mode)
        if rhs is not None:
            return (c0 + c1 + opc + c3 + c4, rhs)
        want.extend(taken)  # invalid combo: restore coverage glyphs, retry
    return None


def generate_puzzle(rng: random.Random, marg: dict):
    """Sample one distribution-matched puzzle, or None if construction fails.

    Returns dict(prompt, mapping, ops, mode, base, gold, query)."""
    base = _wchoice(rng, marg["base"])
    mode = _wchoice(rng, marg["mode"])
    reversed_mode = S._is_reversed_digit_mode(mode)
    if base > len(CONTENT_GLYPHS):
        return None
    content = rng.sample(CONTENT_GLYPHS, base)
    digits = list(range(base))
    rng.shuffle(digits)
    mapping = dict(zip(content, digits))

    n_ops = min(_wchoice(rng, marg["n_ops"]), 3)
    op_pool = [g for g in OP_GLYPHS if g not in mapping]
    if len(op_pool) < n_ops:
        return None
    op_glyphs = rng.sample(op_pool, n_ops)
    ops = {og: _wchoice(rng, marg["optypes"]) for og in op_glyphs}
    if all(ops[og] in R.CONCAT_OPS for og in op_glyphs):
        ops[op_glyphs[0]] = _wchoice(rng, marg["arith_optypes"])

    n_ex = max(_wchoice(rng, marg["n_ex"]), n_ops)
    want = list(content)
    rng.shuffle(want)
    # assign operators to example slots: each operator at least once, rest random
    slot_ops = list(op_glyphs)
    while len(slot_ops) < n_ex:
        slot_ops.append(rng.choice(op_glyphs))
    rng.shuffle(slot_ops)

    examples = []
    for opc in slot_ops:
        ex = _gen_example(
            rng, mapping, opc, ops[opc], base, reversed_mode, content, want
        )
        if ex is None:
            return None
        examples.append(ex)
    if want:  # not all glyphs covered -> base would read low; reject
        return None

    # query: prefer an arithmetic operator (the answer then needs the digit map)
    arith_glyphs = [g for g in op_glyphs if ops[g] not in R.CONCAT_OPS]
    qop = rng.choice(arith_glyphs or op_glyphs)
    gold = None
    for _ in range(24):
        q = [rng.choice(content) for _ in range(4)]
        g = _compute_rhs(
            q[0], q[1], qop, q[2], q[3], ops[qop], mapping, base, reversed_mode
        )
        if g is not None and all(ch in mapping for ch in g if ch != qop):
            query = q[0] + q[1] + qop + q[2] + q[3]
            gold = g
            break
    if gold is None:
        return None

    ex_lines = "\n".join(f"{lhs} = {rhs}" for lhs, rhs in examples)
    prompt = f"{HEADER}\n{ex_lines}\nNow, determine the result for: {query}"
    return {
        "prompt": prompt,
        "mapping": mapping,
        "ops": ops,
        "mode": mode,
        "base": base,
        "gold": gold,
        "query": query,
    }


def is_learnable(prompt: str, gold: str) -> bool:
    """True iff the rich solver recovers `gold` by PURE INFERENCE (no answer hint).
    The learnability/well-posedness gate — the same oracle behind the ~72%
    pure-inference figure.

    We call the 'normal'-level path (concat + the 22-op arithmetic search that the
    real puzzles inhabit) directly, NOT `solve()`, to skip its automatic escalation to
    the 38-op 'deep' search: a rejected random puzzle would otherwise pay a full deep
    search every time. Puzzles unsolvable at 'normal' are simply rejected (the real
    distribution is normal-level), keeping the filter fast and the kept set clean."""
    try:
        solver = S.AliceEquationSolver(prompt)  # 'normal': ~22 op-types
        ans, _ = solver._try_concat()
        if ans is None:
            ans, _ = solver._try_arithmetic()
    except (ValueError, KeyError, IndexError, RecursionError, ZeroDivisionError):
        return False
    return ans == gold


def generate_dataset(
    n: int,
    seed: int,
    marg: dict | None = None,
    seen_prompts: set[str] | None = None,
    guard: int = 60,
):
    """Generate `n` learnable, deduped synthetic puzzles. Returns (puzzles, stats)."""
    marg = marg or load_marginals()
    rng = random.Random(seed)
    seen = set(seen_prompts or set())
    out: list[dict] = []
    stats = Counter()
    cap = n * guard
    while len(out) < n and stats["attempts"] < cap:
        stats["attempts"] += 1
        pz = generate_puzzle(rng, marg)
        if pz is None:
            stats["build_fail"] += 1
            continue
        if pz["prompt"] in seen:
            stats["dup"] += 1
            continue
        stats["solver_calls"] += 1
        if not is_learnable(pz["prompt"], pz["gold"]):
            stats["learnable_fail"] += 1
            continue
        seen.add(pz["prompt"])
        out.append(pz)
    stats["kept"] = len(out)
    return out, stats


def build_crypt_symbolic_examples(
    n: int,
    style: str = "derive_search",
    seed: int = 90001,
    marg: dict | None = None,
    seen_prompts: set[str] | None = None,
) -> list[tuple[str, str, str, str]]:
    """Corpus emitter: (problem_id, prompt, completion, answer) for `n` synthetic
    puzzles whose `style` trace round-trips to gold. Shape-matches
    `corpus.build_cryptarithm_examples`."""
    puzzles, _stats = generate_dataset(n, seed, marg, seen_prompts)
    rows: list[tuple[str, str, str, str]] = []
    for i, pz in enumerate(puzzles):
        reasoning = R.render(
            pz["prompt"],
            pz["mapping"],
            pz["ops"],
            pz["mode"],
            pz["base"],
            pz["gold"],
            style,
        )
        if reasoning is None:
            continue
        completion = f"{reasoning}\n</think>\n\\boxed{{{pz['gold']}}}<|im_end|>"
        rows.append(
            (f"crypt-sym-{style}-{seed}-{i}", pz["prompt"], completion, pz["gold"])
        )
    return rows
