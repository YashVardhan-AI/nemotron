# Cryptarithm Induction-Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **IMPLEMENTED 2026-06-04 — with one MAJOR redesign of Task 2 (read before trusting Task 2 below).** Task 2's original raw-DFS `_DigitSearch` was scrapped: measurement showed the injective-map DFS is a ~200-step brute grind for ~all instances (unusable as training data). It was replaced by a **winning-path narration** — `search_with_log(examples, query, op_info, mapping)` (note the signature: 4th arg is the authoritative solution `mapping` from `solve_problem`, NOT `planted_answer`). It narrates the path *toward* that known map: each glyph becomes known either FORCED by a demo (uniquely determined given current knowledge) or GUESSED (seed digit, with genuine immediate-contradiction backtracks). Measured logs: min 9 / median 14 / max 20 records. Record kinds: `forced | assign | reject | backtrack | solution`; the log replays to an injective map == the solution. Tasks 1, 3, 4, 5 were built as written (Task 3 calls `search_with_log(..., mapping)`). The accurate as-built summary is the addendum in `docs/superpowers/plans/2026-06-02-cryptarithm-p1.md`. Treat the Task 2 code blocks below as superseded.

**Goal:** Add a third cryptarithm_deduce CoT style, `reasoning_cryptarithm_induct`, that narrates a GENUINE search (operator elimination + most-constrained-first digit assignment with explicit backtracking + re-emitted state + round-trip verify + reversed-digit arithmetic), rendered from an instrumented solver search log — fixing the "verify-only read-off" root cause documented in `[[cryptarithm-trace-research]]`.

**Architecture:** Three small additions, each behind the existing safety net (renderer returns `None` if the narrated answer disagrees with the authoritative `solve_problem`, so a wrong trace is never emitted). (1) A reversed-digit arithmetic renderer (`_long_arith`) so every multi-digit op is written column-wise with carries, never atomically. (2) An instrumented digit-search narrator (`search_with_log`) in the solver core that, given the operators already pinned by `solve_problem`, replays a most-constrained-variable injective-map search and records a bounded guess/propagate/contradiction/backtrack decision log. (3) A new renderer that turns that log + the existing op/verify/query records into the research-backed trace schema. Then wire the style into `corpus.py` and the val A/B.

**Tech Stack:** Python 3.11, `uv run` (never bare python), `pytest`, `ruff`, `mypy`. Tests live in `val/tests/` (there is no `reasoners/tests/`). The authoritative solver is `reasoners/cryptarithm_deduce_core.py`; the shared rule core is `reasoners/cryptarithm_rule.py`; styles are consumed in `corpus.py::_crypt_renderer` (~line 194).

---

## File Structure

- `reasoners/cryptarithm_arith_render.py` — **Create.** Pure helpers that render an integer op (add/abs_diff/mul) as a reversed-digit, carry-annotated, single-token-spaced column computation. No I/O, no imports from the solver. One responsibility: make arithmetic *executable-looking* at greedy decode (Faith-and-Fate / Goat / tokenization findings).
- `reasoners/cryptarithm_deduce_core.py` — **Modify.** Add `search_with_log(arith_examples, query, op_info, planted_answer=None)` and a private `_DigitSearch` class. Does NOT change `_solve`, `solve_problem`, `sample_solvable`, or any existing record kind (those stay drop-in).
- `reasoners/cryptarithm_trace.py` — **Modify.** Add `reasoning_cryptarithm_induct(problem, answer)` next to the existing two styles, reusing `_OP_WORD`, `_OP_FN`, `_data`, `_resolve_op_line`, `_apply_line`.
- `corpus.py` — **Modify.** Add an `"induct"` branch to `_crypt_renderer` (~line 200).
- `val/tests/test_cryptarithm_arith_render.py` — **Create.** Soundness of the arithmetic renderer.
- `val/tests/test_cryptarithm_deduce_core.py` — **Modify.** Add tests for `search_with_log`.
- `val/tests/test_cryptarithm_trace.py` — **Modify.** Add tests for the new style.

---

## Task 1: Reversed-digit arithmetic renderer

**Files:**
- Create: `reasoners/cryptarithm_arith_render.py`
- Test: `val/tests/test_cryptarithm_arith_render.py`

- [ ] **Step 1: Write the failing test**

```python
# val/tests/test_cryptarithm_arith_render.py
"""Soundness of the reversed-digit, carry-annotated arithmetic renderer.

The rendered text is for the CoT; the gate is that the renderer's returned VALUE
always equals Python's, and that the text shows per-digit (not atomic) work.
"""

from reasoners.cryptarithm_arith_render import long_arith


def test_value_matches_python_for_all_ops():
    for a in range(0, 100, 7):
        for b in range(0, 100, 5):
            assert long_arith("add", a, b)[0] == a + b, (a, b)
            assert long_arith("abs_diff", a, b)[0] == abs(a - b), (a, b)
            assert long_arith("mul", a, b)[0] == a * b, (a, b)


def test_text_is_per_digit_not_atomic():
    # The product 36*47 must NOT appear as the atomic string "36 * 47 = 1692".
    _val, text = long_arith("mul", 36, 47)
    assert "1692" in text  # the final value is stated
    assert "36 * 47 = 1692" not in text  # but not as a single atomic step
    assert "carry" in text.lower()  # carries are written out


def test_add_shows_reversed_columns_with_carry():
    _val, text = long_arith("add", 21, 8)
    low = text.lower()
    assert "29" in text
    assert "carry" in low
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest val/tests/test_cryptarithm_arith_render.py -v`
Expected: FAIL with `ModuleNotFoundError: reasoners.cryptarithm_arith_render`.

- [ ] **Step 3: Write minimal implementation**

```python
# reasoners/cryptarithm_arith_render.py
"""Render an integer op as a reversed-digit, carry-annotated column computation.

Why reversed (least-significant-first) with explicit carries: the literature
(Faith and Fate; "What Algorithms can Transformers Learn?"; Goat; Tokenization
counts) shows transformers do multi-digit arithmetic reliably only when each
output digit is a constant-token function of aligned operand digits + the prior
carry, and when digits are single tokens. So we never emit `36 * 47 = 1692`
atomically; we write the columns. See [[cryptarithm-trace-research]].

Pure: no I/O. Returns (value, text) where value ALWAYS equals Python's op (the
caller gates the whole trace on the boxed answer, but this keeps each line sound).
"""

from __future__ import annotations


def _digits_lsb(n: int) -> list[int]:
    """Digits of n, least-significant-first. _digits_lsb(0) == [0]."""
    if n == 0:
        return [0]
    out: list[int] = []
    while n > 0:
        out.append(n % 10)
        n //= 10
    return out


def _spaced_msb(digits_lsb: list[int]) -> str:
    """Space-separated digits, most-significant-first (one token each)."""
    return " ".join(str(d) for d in reversed(digits_lsb)) or "0"


def _render_columns(symbol: str, a: int, b: int, da: list[int], db: list[int]) -> str:
    width = max(len(da), len(db))
    da = da + [0] * (width - len(da))
    db = db + [0] * (width - len(db))
    carry = 0
    res: list[int] = []
    lines = [f"  {a} {symbol} {b}, least-significant digit first:"]
    for i in range(width):
        s = da[i] + db[i] + carry
        d, new_carry = s % 10, s // 10
        lines.append(
            f"    col{i}: {da[i]} {symbol} {db[i]} + carry {carry} = {s} "
            f"-> digit {d}, carry {new_carry}"
        )
        carry = new_carry
        res.append(d)
    if carry:
        res.append(carry)
    lines.append(f"    result digits (reversed): {_spaced_msb(res)} = {a + b}")
    return "\n".join(lines)


def _render_sub(a: int, b: int) -> tuple[int, str]:
    hi, lo = (a, b) if a >= b else (b, a)
    dh, dl = _digits_lsb(hi), _digits_lsb(lo)
    width = max(len(dh), len(dl))
    dh = dh + [0] * (width - len(dh))
    dl = dl + [0] * (width - len(dl))
    borrow = 0
    res: list[int] = []
    lines = [f"  |{a} - {b}| = {hi} - {lo}, least-significant digit first:"]
    for i in range(width):
        x = dh[i] - dl[i] - borrow
        if x < 0:
            x += 10
            new_borrow = 1
        else:
            new_borrow = 0
        lines.append(
            f"    col{i}: {dh[i]} - {dl[i]} - borrow {borrow} = {x} "
            f"-> digit {x}, borrow {new_borrow}"
        )
        borrow = new_borrow
        res.append(x)
    while len(res) > 1 and res[-1] == 0:
        res.pop()
    lines.append(f"    result digits (reversed): {_spaced_msb(res)} = {abs(a - b)}")
    return abs(a - b), "\n".join(lines)


def _render_mul(a: int, b: int) -> tuple[int, str]:
    # Single-digit partial products + shift-add (Goat decomposition).
    db = _digits_lsb(b)
    lines = [f"  mul {a} * {b}: sum of single-digit partial products:"]
    for i, d in enumerate(db):
        p = a * d * (10**i)
        lines.append(f"    {a} * {d} (digit {i} of {b}) shifted x10^{i} = {p}")
    lines.append(f"    sum of partials = {a * b}")
    return a * b, "\n".join(lines)


def long_arith(op: str, a: int, b: int) -> tuple[int, str]:
    """(value, rendered-columns-text) for op in {add, abs_diff, mul}."""
    if op == "add":
        return a + b, _render_columns("+", a, b, _digits_lsb(a), _digits_lsb(b))
    if op == "abs_diff":
        return _render_sub(a, b)
    if op == "mul":
        return _render_mul(a, b)
    raise ValueError(f"long_arith does not render op {op!r}")
```

Note: delete the dead `_render_add` scaffold above before finishing the step — only `long_arith`, `_render_columns`, `_render_sub`, `_render_mul`, `_digits_lsb`, `_spaced_msb` are kept. (It is shown here only to make the column logic explicit; `_render_columns` is the real implementation.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest val/tests/test_cryptarithm_arith_render.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Format, lint, commit**

```bash
uv run --frozen ruff format reasoners/cryptarithm_arith_render.py val/tests/test_cryptarithm_arith_render.py
uv run --frozen ruff check reasoners/cryptarithm_arith_render.py val/tests/test_cryptarithm_arith_render.py --fix
git add reasoners/cryptarithm_arith_render.py val/tests/test_cryptarithm_arith_render.py
git commit -m "feat(cryptarithm): reversed-digit carry-annotated arithmetic renderer"
```

---

## Task 2: Instrument the solver — bounded most-constrained-first digit search log

**Files:**
- Modify: `reasoners/cryptarithm_deduce_core.py` (append after `is_uniquely_solvable`, do not touch existing functions)
- Test: `val/tests/test_cryptarithm_deduce_core.py`

**Design:** Operators are already pinned by the authoritative `solve_problem` (its `op_info`), so operator choice is rendered forward (no backtrack) by the existing `op` records. This task searches ONLY the injective digit map, given fixed ops — the genuinely search-requiring half (the "hybrid backtracking" refinement: clean for ops, search for digits). `search_with_log` runs a most-constrained-variable backtracking search over the glyphs in the arithmetic demos, forward-checking each demo's equation as soon as its operand glyphs are bound, and records a compact decision log. It STOPS at the first complete consistent assignment (instances are uniquely solvable, so the first solution is THE map). It is bounded (node + record caps); on cap or no-solution it returns `None`, and the renderer skips that seed. The returned map is asserted equal to `solve_problem`'s map in tests, so the search is verified faithful.

- [ ] **Step 1: Write the failing test**

```python
# append to val/tests/test_cryptarithm_deduce_core.py
from reasoners.cryptarithm_deduce_core import search_with_log


def _arith_only(examples):
    """The arithmetic demos (the search operates on these, like _solve)."""
    from reasoners.cryptarithm_deduce_core import _is_concat
    out = []
    for i, o in examples:
        ex = (i[0], i[1], i[2], i[3], i[4], tuple(o))
        if not _is_concat(ex):
            out.append(ex)
    return out


def test_search_recovers_the_authoritative_map():
    # Where solve_problem returns a digit map (arith query), the search must find
    # the SAME injective map -- proving the narrated search is faithful.
    checked = 0
    for seed in range(40):
        _r, examples, q_input, q_answer = sample_solvable(seed, 4)
        data = _data(examples, q_input)
        ans, (mapping, op_info) = solve_problem(data)
        if not mapping:
            continue  # concat-shortcut query: no digit search to narrate
        q = (q_input[0], q_input[1], q_input[2], q_input[3], q_input[4])
        result = search_with_log(_arith_only(examples), q, op_info, planted_answer=q_answer)
        assert result is not None, seed
        found_map, log = result
        assert found_map == mapping, (seed, found_map, mapping)
        checked += 1
    assert checked > 0


def test_search_log_is_bounded_and_well_formed():
    for seed in range(40):
        _r, examples, q_input, q_answer = sample_solvable(seed, 4)
        data = _data(examples, q_input)
        _ans, (mapping, op_info) = solve_problem(data)
        if not mapping:
            continue
        q = tuple(q_input)
        result = search_with_log(_arith_only(examples), q, op_info, planted_answer=q_answer)
        if result is None:
            continue
        _m, log = result
        assert isinstance(log, list) and log
        assert len(log) <= 200, (seed, len(log))  # bounded for the token budget
        kinds = {r["kind"] for r in log}
        assert kinds <= {"assign", "forced", "reject", "backtrack", "solution"}
        assert log[-1]["kind"] == "solution"


def test_search_emits_backtracking_somewhere_in_the_sweep():
    saw_backtrack = False
    for seed in range(40):
        _r, examples, q_input, q_answer = sample_solvable(seed, 4)
        data = _data(examples, q_input)
        _ans, (mapping, op_info) = solve_problem(data)
        if not mapping:
            continue
        q = tuple(q_input)
        result = search_with_log(_arith_only(examples), q, op_info, planted_answer=q_answer)
        if result is None:
            continue
        if any(r["kind"] == "backtrack" for r in result[1]):
            saw_backtrack = True
    assert saw_backtrack  # genuine search, not a straight-line read-off
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest val/tests/test_cryptarithm_deduce_core.py::test_search_recovers_the_authoritative_map -v`
Expected: FAIL with `ImportError: cannot import name 'search_with_log'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to reasoners/cryptarithm_deduce_core.py

_SEARCH_NODE_CAP = 200_000
_SEARCH_LOG_CAP = 200


class _DigitSearch:
    """Most-constrained-variable injective-map search over arithmetic demos with
    operators already fixed (op_info). Records a compact decision log and stops at
    the first complete consistent assignment. Bounded; returns None on cap.
    """

    def __init__(self, examples, op_info):
        # examples: list of (s0,s1,op,s3,s4,rsyms-tuple), all NON-concat.
        self.examples = examples
        self.op_info = op_info
        self.mapping: dict[str, int] = {}
        self.used: set[int] = set()
        self.log: list[dict] = []
        self.nodes = 0
        self.capped = False
        # Variable order: glyphs by descending occurrence (most-constrained-first
        # proxy -- a glyph in many demos prunes fastest).
        counts: Counter = Counter()
        for s0, s1, _op, s3, s4, rsyms in examples:
            for g in (s0, s1, s3, s4, *rsyms):
                counts[g] += 1
        self.order = [g for g, _ in counts.most_common()]

    def _emit(self, rec):
        if len(self.log) < _SEARCH_LOG_CAP:
            self.log.append(rec)

    def _demo_check(self, ex) -> bool | None:
        """True/False if ex is fully bound and (in)consistent; None if not yet
        fully bound (cannot check)."""
        s0, s1, op, s3, s4, rsyms = ex
        if any(g not in self.mapping for g in (s0, s1, s3, s4)):
            return None
        left = 10 * self.mapping[s0] + self.mapping[s1]
        right = 10 * self.mapping[s3] + self.mapping[s4]
        name = self.op_info.get(op)
        if name is None:
            return None
        idx = _OP_NAMES.index(name)
        val = _OPS[idx](left, right)
        rd = _op_result_digits(name, left, right)
        if len(rd) != len(rsyms):
            return False
        for rs, rdig in zip(rsyms, rd):
            if rs in self.mapping:
                if self.mapping[rs] != rdig:
                    return False
            elif rdig in self.used:
                return False  # injectivity: needed digit already taken
        return True

    def _bind_outputs(self, ex):
        """Bind any output glyphs forced by a now-consistent demo; return the list
        of (glyph, digit) bound so they can be undone."""
        s0, s1, op, s3, s4, rsyms = ex
        left = 10 * self.mapping[s0] + self.mapping[s1]
        right = 10 * self.mapping[s3] + self.mapping[s4]
        rd = _op_result_digits(self.op_info[op], left, right)
        bound = []
        for rs, rdig in zip(rsyms, rd):
            if rs not in self.mapping:
                self.mapping[rs] = rdig
                self.used.add(rdig)
                bound.append((rs, rdig))
                self._emit({"kind": "forced", "glyph": rs, "digit": rdig,
                            "by": ex[0] + ex[1] + ex[2] + ex[3] + ex[4]})
        return bound

    def _undo(self, bound):
        for g, d in reversed(bound):
            del self.mapping[g]
            self.used.discard(d)

    def _next_var(self):
        for g in self.order:
            if g not in self.mapping:
                return g
        return None

    def search(self) -> bool:
        self.nodes += 1
        if self.nodes > _SEARCH_NODE_CAP:
            self.capped = True
            return False
        # Forward-check every fully-bound demo; bind forced outputs.
        all_bound = []
        for ex in self.examples:
            ok = self._demo_check(ex)
            if ok is False:
                self._undo(all_bound)
                return False
            if ok is True:
                all_bound += self._bind_outputs(ex)
        var = self._next_var()
        if var is None:
            self._emit({"kind": "solution", "mapping": dict(self.mapping)})
            return True
        for d in range(10):
            if d in self.used:
                continue
            self.mapping[var] = d
            self.used.add(d)
            self._emit({"kind": "assign", "glyph": var, "digit": d})
            if self.search():
                return True
            del self.mapping[var]
            self.used.discard(d)
            self._emit({"kind": "reject", "glyph": var, "digit": d,
                        "reason": "no consistent completion"})
            self._emit({"kind": "backtrack", "glyph": var})
        self._undo(all_bound)
        return False


def search_with_log(examples, query, op_info, planted_answer=None):
    """Replay a most-constrained-first injective-map search over the arithmetic
    demos (operators fixed by op_info) and return (mapping, decision_log), or None
    if the search is capped or finds no solution. The mapping is asserted faithful
    by tests against solve_problem. `query`/`planted_answer` are accepted for
    symmetry with the renderer and reserved for future query-time narration.
    """
    s = _DigitSearch(list(examples), dict(op_info))
    if not s.search() or s.capped:
        return None
    return dict(s.mapping), s.log
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest val/tests/test_cryptarithm_deduce_core.py -v`
Expected: PASS (all existing tests + the 3 new ones).

- [ ] **Step 5: Type-check, format, lint, commit**

```bash
uv run --frozen mypy reasoners/cryptarithm_deduce_core.py
uv run --frozen ruff format reasoners/cryptarithm_deduce_core.py val/tests/test_cryptarithm_deduce_core.py
uv run --frozen ruff check reasoners/cryptarithm_deduce_core.py val/tests/test_cryptarithm_deduce_core.py --fix
git add reasoners/cryptarithm_deduce_core.py val/tests/test_cryptarithm_deduce_core.py
git commit -m "feat(cryptarithm): instrument bounded MRV digit-search decision log"
```

---

## Task 3: The `reasoning_cryptarithm_induct` renderer

**Files:**
- Modify: `reasoners/cryptarithm_trace.py` (add the new function + small private helpers; reuse existing `_OP_WORD`, `_data`, `_resolve_op_line`, `_result_digits`)
- Test: `val/tests/test_cryptarithm_trace.py`

**Design:** Render the research schema: (1) decode demos with space-separated glyphs; (2) Step 1 operator elimination via the existing `op` records (`_resolve_op_line`); (3) Step 2 narrate the `search_with_log` decision log as assign/forced/reject/backtrack lines, re-emitting a STATE block after each commit (inductive scratchpad); (4) Step 3 round-trip VERIFY — re-apply the op to every demo using `long_arith` and assert injectivity; (5) Step 4 apply to the query with `long_arith` and re-encode; `\boxed`. Returns `None` if `solve_problem` yields no digit map (pure concat-shortcut query — fall back to the existing `reasoning_cryptarithm_arith` for those, see Task 4 wiring) OR if `search_with_log` returns `None` OR if the boxed answer disagrees (the existing safety net).

- [ ] **Step 1: Write the failing test**

```python
# append to val/tests/test_cryptarithm_trace.py
from reasoners.cryptarithm_trace import reasoning_cryptarithm_induct


def _induct_arith_seeds(lo=0, hi=40):
    """Seeds whose query is arithmetic (digit map recovered) -- the induct style
    targets these; concat-shortcut seeds fall back to the deduce style."""
    out = []
    for seed in range(lo, hi):
        problem, answer = make_trace_problem(seed, 4)
        if _is_arith_query(problem):
            out.append((seed, problem, answer))
    return out


def test_induct_ends_with_correct_boxed_answer_and_is_sound():
    seeds = _induct_arith_seeds()
    assert seeds  # the sweep actually exercises arith queries
    for seed, problem, answer in seeds:
        trace = reasoning_cryptarithm_induct(problem, answer)
        assert trace is not None, seed
        assert trace.rstrip().endswith("\\boxed{" + answer + "}"), seed
        assert "MISMATCH" not in trace, seed


def test_induct_shows_state_blocks_and_backtracking_in_sweep():
    saw_state = saw_backtrack = False
    for _seed, problem, answer in _induct_arith_seeds():
        t = reasoning_cryptarithm_induct(problem, answer)
        if "STATE" in t:
            saw_state = True
        if "backtrack" in t.lower():
            saw_backtrack = True
    assert saw_state   # inductive scratchpad re-emits state
    assert saw_backtrack  # genuine search appears somewhere


def test_induct_includes_round_trip_injectivity_check():
    for _seed, problem, answer in _induct_arith_seeds():
        low = reasoning_cryptarithm_induct(problem, answer).lower()
        assert "injectiv" in low or "all distinct" in low


def test_induct_arithmetic_is_not_atomic():
    # at least one arith query must show carry-annotated columns, never just
    # "a * b = c" with nothing else.
    saw_columns = False
    for _seed, problem, answer in _induct_arith_seeds():
        if "carry" in reasoning_cryptarithm_induct(problem, answer).lower():
            saw_columns = True
    assert saw_columns


def test_induct_within_token_budget():
    for _seed, problem, answer in _induct_arith_seeds():
        t = reasoning_cryptarithm_induct(problem, answer)
        assert len(t) < 16000, (len(t),)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest val/tests/test_cryptarithm_trace.py::test_induct_ends_with_correct_boxed_answer_and_is_sound -v`
Expected: FAIL with `ImportError: cannot import name 'reasoning_cryptarithm_induct'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add near the top imports of reasoners/cryptarithm_trace.py
from reasoners.cryptarithm_arith_render import long_arith
from reasoners.cryptarithm_deduce_core import (
    _is_concat,
    search_with_log,
    solve_problem,
)
# (sample_solvable / solve_problem already imported; add the two new names.)


# add at the end of reasoners/cryptarithm_trace.py
def _arith_demos(problem) -> list[tuple]:
    out = []
    for e in problem.examples:
        iv, ov = e.input_value, e.output_value
        ex = (iv[0], iv[1], iv[2], iv[3], iv[4], tuple(ov))
        if not _is_concat(ex):
            out.append(ex)
    return out


def _state_block(mapping: dict, op_info: dict) -> str:
    m = ", ".join(f"{g}={mapping[g]}" for g in sorted(mapping)) or "(none yet)"
    ops = ", ".join(f"{g}->{op_info[g]}" for g in sorted(op_info))
    return f"STATE | map: {{{m}}} | ops: {{{ops}}}"


def _render_search(log: list[dict], op_info: dict) -> list[str]:
    mapping: dict[str, int] = {}
    lines: list[str] = []
    for rec in log:
        k = rec["kind"]
        if k == "assign":
            mapping[rec["glyph"]] = rec["digit"]
            lines.append(f"  guess {rec['glyph']} = {rec['digit']}.")
            lines.append("  " + _state_block(mapping, op_info))
        elif k == "forced":
            mapping[rec["glyph"]] = rec["digit"]
            lines.append(
                f"  forced {rec['glyph']} = {rec['digit']} by demo {rec['by']}."
            )
            lines.append("  " + _state_block(mapping, op_info))
        elif k == "reject":
            lines.append(
                f"  {rec['glyph']} = {rec['digit']} fails ({rec['reason']})."
            )
        elif k == "backtrack":
            mapping.pop(rec["glyph"], None)
            lines.append(f"  backtrack on {rec['glyph']}.")
        elif k == "solution":
            lines.append("  complete assignment reached.")
    return lines


def _verify_roundtrip(problem, mapping, op_info) -> list[str]:
    d2s = {d: s for s, d in mapping.items()}
    lines = ["Step 3 - verify the recovered rule against EVERY demo:"]
    for e in problem.examples:
        iv, ov = e.input_value, e.output_value
        ex = (iv[0], iv[1], iv[2], iv[3], iv[4], tuple(ov))
        if _is_concat(ex):
            lines.append(f"  {iv} = {ov}: concatenation, glyphs copied. matches.")
            continue
        name = op_info[iv[2]]
        left = 10 * mapping[iv[0]] + mapping[iv[1]]
        right = 10 * mapping[iv[3]] + mapping[iv[4]]
        val, cols = long_arith(name, left, right)
        enc = "".join(d2s[d] for d in _result_digits(name, left, right))
        status = "matches" if enc == ov else "MISMATCH"
        lines.append(f"  {iv} = {ov}: {_OP_WORD[name]}")
        lines.append(cols)
        lines.append(f"    encode -> {enc}. {status}.")
    digs = list(mapping.values())
    inj = "all distinct" if len(set(digs)) == len(digs) else "INJECTIVITY VIOLATION"
    lines.append(f"  injectivity round-trip: digits {sorted(digs)} - {inj}.")
    return lines


def _apply_induct(problem, answer, mapping, op_info) -> list[str]:
    q = problem.question
    name = op_info[q[2]]
    d2s = {d: s for s, d in mapping.items()}
    left = 10 * mapping[q[0]] + mapping[q[1]]
    right = 10 * mapping[q[3]] + mapping[q[4]]
    _val, cols = long_arith(name, left, right)
    enc = "".join(d2s[d] for d in _result_digits(name, left, right))
    return [
        f"Step 4 - apply to the query {q}:",
        f"  decode {q[0]}{q[1]} = {left}, {q[3]}{q[4]} = {right}; "
        f"operator '{q[2]}' is the {_OP_WORD[name]}.",
        cols,
        f"  encode the result digits back to glyphs -> {enc}.",
    ]


def reasoning_cryptarithm_induct(problem: Problem, answer: str) -> str | None:
    """Induction-search CoT: operator elimination (forward) + most-constrained-
    first digit search with explicit backtracking + re-emitted STATE + round-trip
    verify + reversed-digit arithmetic. Returns None for pure concat-shortcut
    queries (no digit map to search -- caller falls back to the deduce style), if
    the search caps, or if the boxed answer disagrees with the authority.
    """
    data = _data(problem)
    ans, (mapping, op_info), log = solve_problem(data, trace=True)
    if ans != answer or not mapping:
        return None
    q = tuple(problem.question)
    searched = search_with_log(_arith_demos(problem), q, op_info, planted_answer=answer)
    if searched is None:
        return None
    found_map, search_log = searched
    if found_map != mapping:  # faithfulness guard
        return None

    op_recs = [r for r in log if r["kind"] == "op" and r["name"] is not None]
    verifies = [r for r in log if r["kind"] == "verify"]
    maprec = next((r for r in log if r["kind"] == "map"), None)

    lines = [
        "I must DEDUCE two things from the examples: what each operator glyph "
        "does, and which digit each symbol stands for. I will search, not assume.",
        (
            "Each input is `s0 s1 op s3 s4`: two glyphs form a two-digit number, "
            "an operator glyph, then two more glyphs form a second two-digit "
            "number; the output spells the result's digits in the same secret "
            "alphabet."
        ),
        "",
        "Step 1 - pin each operator by elimination:",
    ]
    lines += [_resolve_op_line(r, verifies, maprec) for r in op_recs]
    lines.append("")
    lines.append("Step 2 - search the injective digit map (most-constrained first):")
    lines += _render_search(search_log, op_info)
    lines.append("")
    lines += _verify_roundtrip(problem, mapping, op_info)
    lines.append("")
    lines += _apply_induct(problem, answer, mapping, op_info)
    lines.append("")
    lines.append(f"So the answer is \\boxed{{{answer}}}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest val/tests/test_cryptarithm_trace.py -v`
Expected: PASS (all existing trace tests + the 5 new induct tests). If `test_induct_within_token_budget` fails on a rare deep-search seed, lower `_SEARCH_LOG_CAP` in `cryptarithm_deduce_core.py` from 200 to 120 and re-run (the renderer already returns None when capped, so those seeds are skipped, not emitted wrong).

- [ ] **Step 5: Type-check, format, lint, commit**

```bash
uv run --frozen mypy reasoners/cryptarithm_trace.py
uv run --frozen ruff format reasoners/cryptarithm_trace.py val/tests/test_cryptarithm_trace.py
uv run --frozen ruff check reasoners/cryptarithm_trace.py val/tests/test_cryptarithm_trace.py --fix
git add reasoners/cryptarithm_trace.py val/tests/test_cryptarithm_trace.py
git commit -m "feat(cryptarithm): induction-search CoT style (reasoning_cryptarithm_induct)"
```

---

## Task 4: Wire the `induct` style into the corpus builder

**Files:**
- Modify: `corpus.py:194-210` (`_crypt_renderer`)
- Test: `val/tests/test_cryptarithm_trace.py` (one integration test that the corpus renderer dispatches the new style and falls back cleanly)

**Design:** Add an `"induct"` branch. Because `reasoning_cryptarithm_induct` returns `None` for concat-shortcut queries by design, the `"induct"` renderer must fall back to `reasoning_cryptarithm_arith` for those so no row is silently dropped. Implement the fallback as a wrapper closure.

- [ ] **Step 1: Write the failing test**

```python
# append to val/tests/test_cryptarithm_trace.py
def test_corpus_induct_renderer_dispatches_and_falls_back():
    from corpus import _crypt_renderer
    render = _crypt_renderer("induct", seed=0)
    produced = 0
    for seed in range(12):
        problem, answer = make_trace_problem(seed, 4)
        trace = render(problem, answer)
        assert trace is not None, seed  # induct OR fallback always yields a trace
        assert trace.rstrip().endswith("\\boxed{" + answer + "}"), seed
        produced += 1
    assert produced == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest val/tests/test_cryptarithm_trace.py::test_corpus_induct_renderer_dispatches_and_falls_back -v`
Expected: FAIL with `ValueError: unknown cryptarithm trace style: 'induct'`.

- [ ] **Step 3: Write minimal implementation**

```python
# in corpus.py _crypt_renderer, add before the final `raise ValueError`:
    if style == "induct":
        from reasoners.cryptarithm_trace import (
            reasoning_cryptarithm_arith,
            reasoning_cryptarithm_induct,
        )

        def _induct_or_fallback(problem, answer):
            t = reasoning_cryptarithm_induct(problem, answer)
            return t if t is not None else reasoning_cryptarithm_arith(problem, answer)

        return _induct_or_fallback
```

(Place it alongside the existing `deduce`/`propagate`/`mixed` branches; keep the imports inside the function as the existing branches do.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest val/tests/test_cryptarithm_trace.py::test_corpus_induct_renderer_dispatches_and_falls_back -v`
Expected: PASS.

- [ ] **Step 5: Format, lint, commit**

```bash
uv run --frozen ruff format corpus.py val/tests/test_cryptarithm_trace.py
uv run --frozen ruff check corpus.py val/tests/test_cryptarithm_trace.py --fix
git add corpus.py val/tests/test_cryptarithm_trace.py
git commit -m "feat(corpus): wire induct cryptarithm style with concat fallback"
```

---

## Task 5: Full suite + plan-doc handoff note

**Files:**
- Modify: `docs/superpowers/plans/2026-06-02-cryptarithm-p1.md` (append a short note recording the new arm)

- [ ] **Step 1: Run the full val test suite**

Run: `uv run pytest val/tests -v`
Expected: PASS (all prior tests + the new ones). Investigate any failure before proceeding — do not edit tests to pass.

- [ ] **Step 2: Record the new arm in the Phase-6 A/B doc**

Append to `docs/superpowers/plans/2026-06-02-cryptarithm-p1.md`:

```markdown
## Addendum 2026-06-04: induction-search style (`induct`)
A third trace style `reasoning_cryptarithm_induct` was added (research basis:
[[cryptarithm-trace-research]]) to fix the verify-only read-off root cause:
operator elimination (forward) + instrumented most-constrained-first digit search
with explicit backtracking (`cryptarithm_deduce_core.search_with_log`) + re-emitted
STATE blocks + round-trip injectivity verify + reversed-digit carry-annotated
arithmetic (`cryptarithm_arith_render.long_arith`). Wired into `corpus.py`
`_crypt_renderer` as `style="induct"` (falls back to `deduce` for concat-shortcut
queries). Phase-6 A/B is now THREE arms: deduce | propagate | induct. NEXT: set
`CRYPT_STYLE="induct"` in corpus.py, `uv run corpus.py`, train, then re-measure on
`val/kaggle_newrule_eval_standalone.py` (N=50, difficulty 4) vs the 6% frozen
baseline.
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-06-02-cryptarithm-p1.md
git commit -m "docs(cryptarithm): record induct trace style as Phase-6 A/B arm 3"
```

---

## Handoff / measurement (user-run, not a coding task)

The accuracy payoff is measured on GPU/Kaggle, not locally:
1. In `corpus.py`, set `CRYPT_STYLE = "induct"` (line ~52); keep `CRYPT_REPLACE_REAL=True`.
2. `uv run corpus.py` → adds `cryptarithm-induct-*` rows, drops the concat-only real traces.
3. Set the training filter to the 9 reasoning categories, `uv run python -m train_sft`, then `adapter converter.ipynb`.
4. Re-run `val/kaggle_newrule_eval_standalone.py` (N=50, difficulty 4) and compare cryptarithm_deduce vs the 6.0% frozen baseline. A/B all three styles if budget allows (the doc records the arms).

## Notes for the executor
- NEVER use bare `python`/`pip` — `uv run` / `uv add` only (see `[[use-uv-run-python]]`).
- The renderer's `None`-on-disagreement safety net is load-bearing: a seed where the search caps or disagrees is SKIPPED, never emitted wrong. That is by design — do not "fix" it by forcing a trace.
- Do not modify `_solve`, `solve_problem`, `sample_solvable`, or existing record kinds; Task 2 only APPENDS `search_with_log` + `_DigitSearch`.
- Operators are rendered forward (no backtrack) because they are pinned by length + cross-demo; only the digit map is searched. This is the deliberate "hybrid backtracking" design from the research, not an omission.
```
