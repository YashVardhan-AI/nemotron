"""Windows-safe, instrumented port of investigators/cryptarithm_deduce.py.

Recovers the (glyph->digit map, operator-glyph->operation) rule from the demos by
backtracking search with a NODE CAP instead of signal.SIGALRM (which is
Linux-only). With trace=True, `solve_problem` also returns a structured decision
log that reasoners/cryptarithm_trace.py turns into induction CoT.

This is the AUTHORITY that recovers the planted rule for forward-generated
problems (reasoners/cryptarithm_rule.py constructs them; this verifies them). The
search semantics are ported verbatim from the investigator (confirmed against the
659 real problems + solved investigations/*.txt); only the timeout mechanism
changed. The op index order matches reasoners.cryptarithm_rule.OP_NAMES.
"""

from __future__ import annotations

from collections import Counter

from reasoners.cryptarithm_rule import OP_NAMES, num_to_digits, sample_problem

# Index order matches OP_NAMES: 0 add, 1 abs_diff, 2 mul, 3 concat, 4 rev_concat.
_OPS = [
    lambda a, b: a + b,
    lambda a, b: abs(a - b),
    lambda a, b: a * b,
    lambda a, b: a * 100 + b,
    lambda a, b: b * 100 + a,
]
_OP_NAMES = list(OP_NAMES)

_NODE_CAP = 2_000_000  # safety net replacing the investigator's 30s SIGALRM


def _is_concat(ex) -> bool:
    """True if the result spells s0s1s3s4 (concat) or s3s4s0s1 (rev_concat)."""
    s0, s1, _op, s3, s4, rsyms = ex
    return rsyms == (s0, s1, s3, s4) or rsyms == (s3, s4, s0, s1)


class _Solver:
    """Backtracking CSP solver (ported from the investigator, node-capped)."""

    def __init__(self, examples, query, unique=True):
        self.examples = examples
        self.query = query
        self.unique = unique
        self.mapping: dict[str, int] = {}
        self.used: set[int] = set()
        self.op_assign: dict[str, int] = {}
        self.answers: Counter = Counter()
        self.answer_info: dict[str, tuple[dict, dict]] = {}
        self.max_solutions = 200
        self.nodes = 0
        self.capped = False

    def solve(self):
        self._process(0)
        if self.answers:
            best, best_count = self.answers.most_common(1)[0]
            total = sum(self.answers.values())
            if not self.unique and total > 1 and best_count < total * 0.3:
                return None, ({}, {})
            return best, self.answer_info.get(best, ({}, {}))
        return None, ({}, {})

    def solve_counted(self):
        """Like solve() but also returns the number of DISTINCT consistent query
        answers found -- 1 means the query is uniquely determined by the demos."""
        self._process(0)
        if self.answers:
            best, _ = self.answers.most_common(1)[0]
            return best, self.answer_info.get(best, ({}, {})), len(self.answers)
        return None, ({}, {}), 0

    def _process(self, idx):
        if len(self.answers) >= self.max_solutions:
            return
        self.nodes += 1
        if self.nodes > _NODE_CAP:
            self.capped = True
            return
        if idx == len(self.examples):
            self._compute_query()
            return

        s0, s1, op_sym, s3, s4, rsyms = self.examples[idx]
        rlen = len(rsyms)

        # Filter ops by result-length feasibility (abs_diff<=2, add<=3, mul<=4,
        # concat/rev_concat == 4).
        feasible_ops = []
        if rlen <= 3:
            feasible_ops.append(0)  # add
        if rlen <= 2:
            feasible_ops.append(1)  # abs_diff
        if rlen <= 4:
            feasible_ops.append(2)  # mul
        if rlen == 4:
            feasible_ops.extend([3, 4])  # concat, rev_concat

        for d0 in self._vals(s0):
            n0 = self._assign(s0, d0)
            if n0 is None:
                continue
            for d1 in self._vals(s1):
                n1 = self._assign(s1, d1)
                if n1 is None:
                    continue
                lv = d0 * 10 + d1
                for d3 in self._vals(s3):
                    n3 = self._assign(s3, d3)
                    if n3 is None:
                        continue
                    for d4 in self._vals(s4):
                        n4 = self._assign(s4, d4)
                        if n4 is None:
                            continue
                        rv = d3 * 10 + d4

                        ops_to_try = (
                            [self.op_assign[op_sym]]
                            if op_sym in self.op_assign
                            else feasible_ops
                        )

                        for op_id in ops_to_try:
                            result_val = _OPS[op_id](lv, rv)
                            if op_id >= 3:
                                if result_val < 0 or result_val >= 10000:
                                    continue
                                rd = (
                                    result_val // 1000,
                                    (result_val // 100) % 10,
                                    (result_val // 10) % 10,
                                    result_val % 10,
                                )
                            else:
                                rd = num_to_digits(result_val)
                            if len(rd) != rlen:
                                continue

                            assigns = []
                            ok = True
                            for rs, rdig in zip(rsyms, rd):
                                ns = self._assign(rs, rdig)
                                if ns is None:
                                    ok = False
                                    break
                                assigns.append((rs, ns))

                            if ok:
                                op_new = op_sym not in self.op_assign
                                if op_new:
                                    self.op_assign[op_sym] = op_id
                                self._process(idx + 1)
                                if op_new:
                                    del self.op_assign[op_sym]

                            for rs, ns in reversed(assigns):
                                self._undo(rs, ns)

                            if len(self.answers) >= self.max_solutions:
                                self._undo(s4, n4)
                                self._undo(s3, n3)
                                self._undo(s1, n1)
                                self._undo(s0, n0)
                                return

                        self._undo(s4, n4)
                    self._undo(s3, n3)
                self._undo(s1, n1)
            self._undo(s0, n0)

    def _vals(self, sym):
        if sym in self.mapping:
            return (self.mapping[sym],)
        if self.unique:
            return tuple(d for d in range(10) if d not in self.used)
        return range(10)

    def _assign(self, sym, dig):
        if sym in self.mapping:
            return False if self.mapping[sym] == dig else None
        if self.unique and dig in self.used:
            return None
        self.mapping[sym] = dig
        if self.unique:
            self.used.add(dig)
        return True

    def _undo(self, sym, was_new):
        if was_new is True:
            if self.unique:
                self.used.discard(self.mapping[sym])
            del self.mapping[sym]

    def _compute_query(self):
        qs0, qs1, qop, qs3, qs4 = self.query
        for s in (qs0, qs1, qs3, qs4):
            if s not in self.mapping:
                return

        ql = self.mapping[qs0] * 10 + self.mapping[qs1]
        qr = self.mapping[qs3] * 10 + self.mapping[qs4]
        if qop in self.op_assign:
            op_candidates = [self.op_assign[qop]]
        else:
            op_candidates = range(len(_OP_NAMES))

        d2s: dict[int, str] = {}
        for s, d in self.mapping.items():
            if d not in d2s:
                d2s[d] = s

        for op_id in op_candidates:
            result_val = _OPS[op_id](ql, qr)
            if op_id >= 3:
                if result_val < 0 or result_val >= 10000:
                    continue
                rd = (
                    result_val // 1000,
                    (result_val // 100) % 10,
                    (result_val // 10) % 10,
                    result_val % 10,
                )
            else:
                rd = num_to_digits(result_val)

            parts = []
            ok = True
            for d in rd:
                if d not in d2s:
                    ok = False
                    break
                parts.append(d2s[d])
            if not ok:
                continue

            ans = "".join(parts)
            self.answers[ans] += 1
            if ans not in self.answer_info:
                op_info = {k: _OP_NAMES[v] for k, v in self.op_assign.items()}
                op_info[qop] = _OP_NAMES[op_id]
                self.answer_info[ans] = (dict(self.mapping), op_info)


def _solve(data):
    """Port of the investigator's solve_problem: returns (answer, (map, op))."""
    examples = []
    for e in data["examples"]:
        inp = e["input_value"]
        out = e["output_value"]
        examples.append((inp[0], inp[1], inp[2], inp[3], inp[4], tuple(out)))

    q = data["question"]
    query = (q[0], q[1], q[2], q[3], q[4])

    concat_ops = set()
    nonconcat_ops = set()
    for ex in examples:
        if _is_concat(ex):
            concat_ops.add(ex[2])
        else:
            nonconcat_ops.add(ex[2])

    q_op = query[2]
    if q_op in concat_ops and q_op not in nonconcat_ops:
        for ex in examples:
            if ex[2] == q_op and _is_concat(ex):
                s0, s1, _, s3, s4, rsyms = ex
                if rsyms == (s0, s1, s3, s4):
                    return query[0] + query[1] + query[3] + query[4], (
                        {},
                        {q_op: "concat"},
                    )
                return query[3] + query[4] + query[0] + query[1], (
                    {},
                    {q_op: "rev_concat"},
                )
        return query[0] + query[1] + query[3] + query[4], ({}, {q_op: "concat"})

    arith_examples = [ex for ex in examples if not _is_concat(ex)]

    solver = _Solver(arith_examples, query, unique=True)
    ans, info = solver.solve()
    if ans is not None:
        return ans, info

    solver2 = _Solver(arith_examples, query, unique=False)
    return solver2.solve()


# --- decision log (consumed by reasoners/cryptarithm_trace.py) ---------------


def _struct_dir(iv: str, ov: str) -> str | None:
    """Concat direction readable from glyphs alone (no digit values), else None."""
    rsyms = tuple(ov)
    if rsyms == (iv[0], iv[1], iv[3], iv[4]):
        return "concat"
    if rsyms == (iv[3], iv[4], iv[0], iv[1]):
        return "rev_concat"
    return None


def _op_result_digits(op_name: str, left: int, right: int) -> tuple[int, ...]:
    idx = _OP_NAMES.index(op_name)
    val = _OPS[idx](left, right)
    if idx >= 3:
        return (val // 1000, (val // 100) % 10, (val // 10) % 10, val % 10)
    return num_to_digits(val)


def _build_log(data, mapping, op_info, answer):
    """Render the recovered rule into structured deduction records. Every numeric
    claim is recomputed here, so the log is sound regardless of the search path.
    Record kinds: op | map | verify | query (see reasoners/cryptarithm_trace.py).
    """
    examples = data["examples"]
    question = data["question"]
    log: list[dict] = []

    # 1) operator resolution -- structural for concat/rev_concat (readable off the
    # demos with no digit values), else arithmetic (narrowed by output length).
    by_op: dict[str, list[dict]] = {}
    for e in examples:
        by_op.setdefault(e["input_value"][2], []).append(e)
    for op_glyph, demos in by_op.items():
        dirs = {_struct_dir(d["input_value"], d["output_value"]) for d in demos}
        if None not in dirs and len(dirs) == 1:  # all demos same concat direction
            name, how = dirs.pop(), "structural"
        else:
            name, how = op_info.get(op_glyph), "length"
        log.append(
            {
                "kind": "op",
                "op": op_glyph,
                "name": name,
                "how": how,
                "out_len": len(demos[0]["output_value"]),
            }
        )

    # 2) recovered digit map (may be empty on the pure-concat shortcut path).
    if mapping:
        digit_to_sym = {d: s for s, d in mapping.items()}
        log.append(
            {"kind": "map", "sym_to_digit": dict(mapping), "digit_to_sym": digit_to_sym}
        )

    # 3) verify every demo: concat structurally (no map), arithmetic numerically.
    d2s = {d: s for s, d in mapping.items()}
    for e in examples:
        iv, ov = e["input_value"], e["output_value"]
        direction = _struct_dir(iv, ov)
        if direction is not None:
            log.append(
                {
                    "kind": "verify",
                    "inp": iv,
                    "out": ov,
                    "name": direction,
                    "structural": True,
                    "ok": True,
                }
            )
            continue
        name = op_info.get(iv[2])
        if name is None or not all(iv[i] in mapping for i in (0, 1, 3, 4)):
            continue
        left = 10 * mapping[iv[0]] + mapping[iv[1]]
        right = 10 * mapping[iv[3]] + mapping[iv[4]]
        digits = _op_result_digits(name, left, right)
        check = "".join(d2s.get(d, "?") for d in digits)
        log.append(
            {
                "kind": "verify",
                "inp": iv,
                "out": ov,
                "left": left,
                "right": right,
                "name": name,
                "digits": digits,
                "check": check,
                "ok": check == ov,
                "structural": False,
            }
        )

    # 4) apply to the query.
    qname = op_info.get(question[2])
    q_rec = {"kind": "query", "inp": question, "name": qname, "answer": answer}
    if mapping and all(question[i] in mapping for i in (0, 1, 3, 4)):
        left = 10 * mapping[question[0]] + mapping[question[1]]
        right = 10 * mapping[question[3]] + mapping[question[4]]
        q_rec.update(
            {
                "left": left,
                "right": right,
                "digits": _op_result_digits(qname, left, right),
            }
        )
    log.append(q_rec)
    return log


def solve_problem(data, trace: bool = False):
    """Solve one cryptarithm_deduce problem.

    data: {"examples": [{"input_value", "output_value"}...], "question": str}.
    Returns (answer, (mapping, op_info)) -- drop-in compatible with the
    investigator. With trace=True returns (answer, (mapping, op_info), log).
    """
    answer, (mapping, op_info) = _solve(data)
    if not trace:
        return answer, (mapping, op_info)
    log = _build_log(data, mapping, op_info, answer) if answer is not None else []
    return answer, (mapping, op_info), log


def _solve_counted(data):
    """(answer, info, n_distinct_answers). n_distinct == 1 => uniquely solvable.

    Uses the strict injective solver only (no non-unique fallback) -- for
    GENERATION we want instances that are unambiguously deducible.
    """
    examples = [
        (
            e["input_value"][0],
            e["input_value"][1],
            e["input_value"][2],
            e["input_value"][3],
            e["input_value"][4],
            tuple(e["output_value"]),
        )
        for e in data["examples"]
    ]
    q = data["question"]
    query = (q[0], q[1], q[2], q[3], q[4])

    concat_ops = {ex[2] for ex in examples if _is_concat(ex)}
    nonconcat_ops = {ex[2] for ex in examples if not _is_concat(ex)}
    if query[2] in concat_ops and query[2] not in nonconcat_ops:
        ans, info = _solve(data)  # concat shortcut: unique by construction
        return ans, info, 1

    arith = [ex for ex in examples if not _is_concat(ex)]
    return _Solver(arith, query, unique=True).solve_counted()


def is_uniquely_solvable(examples, q_input, q_answer) -> bool:
    """True iff the demos pin the query to exactly the planted answer."""
    data = {
        "examples": [{"input_value": i, "output_value": o} for i, o in examples],
        "question": q_input,
    }
    ans, _info, n = _solve_counted(data)
    return ans == q_answer and n == 1


def sample_solvable(
    seed: int,
    difficulty: int = 4,
    *,
    determinacy: str = "well",
    profile: str | None = None,
    max_salt: int = 40,
):
    """Forward-generate a UNIQUELY-deducible instance for rule `seed`.

    Resamples the instance (operands/demos) via `salt` -- keeping the SAME rule,
    so rule_signature(seed) stays stable for holdout -- until the solver recovers
    exactly the planted answer as the unique consistent answer. This is the
    single source of truth for BOTH the val generator and the training emitter
    (no drift): they call this, not the raw constructor. Returns the first
    uniquely-solvable instance, or the last attempt if none qualifies (rare; the
    caller should treat that as a skip).
    """
    last = None
    for salt in range(max_salt):
        rule, examples, q_input, q_answer = sample_problem(
            seed, difficulty, determinacy=determinacy, profile=profile, salt=salt
        )
        last = (rule, examples, q_input, q_answer)
        if is_uniquely_solvable(examples, q_input, q_answer):
            return rule, examples, q_input, q_answer
    return last
