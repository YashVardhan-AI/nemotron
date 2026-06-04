# Cryptarithm P1 — Forward-Generator + Clean Val Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift `cryptarithm_deduce` (the #1 measured lever: ~7.5% test weight, ~8% solver coverage, ~0% on new rules) by building a calibrated forward-generator that emits unlimited verified arithmetic-cipher problems with induction/self-verify CoT, plus a clean new-rule **val** generator so the gain is measurable.

**Architecture:** *Generate, don't solve.* A shared rule core samples a rule = (symbol↔digit bijection) × (operator-glyph → arithmetic semantics) × (digit→symbol re-encoding), then emits demo pairs + a query whose answer is correct *by construction*. Two thin consumers wrap the core: (A) `val/generators/cryptarithm.py` (new-rule eval, mirrors the real prompt, registers a holdout signature) and (B) a training emitter that attaches a CoT trace and feeds the SFT corpus, replacing the ×12 exact-copy duplication of ~54 real cryptarithm traces. Both calibrate to the **measured** real distribution (format, operator alphabet incl. the long tail, 3–5 demos, glyph alphabet) — the recurring lesson from cipher/bit.

**This is a full multi-arm plan, not a single shot** (revised after a "best possible plan?" review). The four pillars beyond the base lever:
1. **Don't bet on one CoT template.** Three trace styles are produced and **A/B'd on the val harness** (Phase 6): a deduction-CSP-log renderer, a cleaner *constraint-propagation / compute-and-correct* renderer (likely more executable in one greedy pass), and a **teacher-LLM rejection-sampled** variant (natural traces, kept only when `\boxed{}` matches the constructed answer). val picks the winner — we do not assume the CSP log is best.
2. **Distribution match is pass-1, not deferred** — the cipher/bit lesson cost us three times. Pass-1 includes long-tail operator glyphs and a small fraction of near-determined instances.
3. **Two corpus arms:** a size-matched ablation (isolates the diversity effect) *and* a scale-up + curriculum arm (the induction skill plausibly needs thousands of graded instances).
4. **RL stays in the super plan (Phase 7 = hooks only):** RL (#7 RFT → #9 DPO → #10 GRPO, gated on plateau) is a *global, cross-category* lever already owned by `you-are-acting-as-deep-fountain.md`. This plan does not re-define it; it only exposes the cryptarithm hooks RL needs (unlimited verified prompts + the `compare_answer` reward).

**Tech Stack:** Python 3, `uv` (never pip), `pytest`, `ruff`, the existing `val/` harness (`val/generators` self-registering protocol, `reasoners/store_types.py` `Problem`/`Example`), `reasoners/cryptarithm.py` (CoT shape reference).

---

## Relationship to the super plan (consistency pass — no double-definition)

This plan is the **cryptarithm-specific instantiation** of levers in `C:\Users\porus\.claude\plans\you-are-acting-as-deep-fountain.md` (execution-sequence **step 1**). To avoid the RL-style misscope a second time, the mapping is explicit:

| Super-plan lever (scope) | This plan |
|---|---|
| #2 forward-gen (global method) | Phases 1/4/5 — applied to cryptarithm |
| #3 induction/self-verify/backtrack CoT (all traces) | Phase 4, deepened into a **3-style bake-off** (Phase 6). The winning style should feed back into the global #3 rewrite. |
| #4 hygiene + **prompt-template diversity** (global) | Phase 5 holdout + distribution-match; wrapper-diversity decision below. |
| #5 **scale unique set + retune steps/epochs** (global hyperparams) | **Scope split:** this plan scales only *cryptarithm data volume* (Phase 5 Arm B). The **global step/epoch sweep stays in super-plan #5** — Phase 6 only watches the weighted score for regressions, it does NOT own the step sweep. |
| #6 teacher LLM (label *real* hard instances; incl. code-exec rule discovery) | **Complementary, not duplicate:** Task 4.4 uses the teacher for *trace-style diversity on synthetic, already-answered problems* (rejection-sampled). The super plan's #6 use — teacher-with-code to *discover* rules for **real** unsolved cryptarithm instances — is a different capability and stays in the super plan; reference it if Phase 6 shows real-instance coverage is still the gap. |
| #7/#9/#10 RL (global, cross-category, gated on plateau) | **NOT owned here** — Phase 7 exposes hooks only. |

**Closes a super-plan Open Item:** "build val generators for the remaining categories (cryptarithm/…)". Phase 2 delivers the cryptarithm one → when this lands, tick that item in the super plan.

**Prompt-wrapper diversity decision (reconciling super-plan #4 with our exact-format calibration):** the real wrapper (`In Alice's Wonderland…`) is the category's verified signature and the safest exact match, so it is the **dominant (~80–85%)** form. To instantiate #4's robustness intent without risking the positional 5-char parse, emit a **small (~15–20%) fraction with light paraphrase/reordering of the example & query phrasing only** (never altering the `s0 s1 op s3 s4` token structure or the `=`/`\boxed{}` mechanics). Both the val generator and the training emitter draw wrappers from the same pool (no drift); the val default stays the exact real wrapper so the measured number reflects the real format.

---

## Calibration facts (measured + VERIFIED 2026-06-03 — the design is pinned to these)

**Authoritative source: `investigators/cryptarithm_deduce.py`** (a working full-family solver), confirmed against its solved `investigations/*.txt` outputs and 659 real problems. ⚠️ An earlier draft of this plan assumed a *fixed* mapping `+`=abs-diff/`-`=add/`*`=mul — a brute fit **falsified** it (explains 1/25 problems). The real rule:

- **Prompt template (verbatim):**
  `In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:\n` + each demo as `{input_value} = {output_value}` joined by `\n` + `\nNow, determine the result for: {question}`
- **Input value:** exactly 5 chars = `s0 s1 op s3 s4` (left symbol-pair, operator symbol at index 2, right symbol-pair). The operator is *just the 3rd symbol* and can be ANY glyph (usually `+`/`-`/`*` ≈600 each, plus a long tail). Operator glyphs also appear as digit-symbols elsewhere — role is positional.
- **Symbol↔digit map:** each distinct symbol → a **UNIQUE digit 0–9 (injective, all-different)**. left = `10·d(s0)+d(s1)`, right = `10·d(s3)+d(s4)`. Digits 0–9 are NEVER shown literally.
- **FIVE operations, each operator-symbol assigned one, DEDUCED PER PROBLEM (not a fixed glyph→op mapping):** `add`=a+b, `abs_diff`=|a−b|, `mul`=a·b, `concat`=a·100+b, `rev_concat`=b·100+a. A problem may use several operators; the same glyph means different ops in different problems.
- **Result re-encoding:** arith ops → `num_to_digits` (natural length 1–4); concat/rev_concat → exactly 4 digits (zero-padded). Each result digit re-encoded to its symbol via the same map → `output_value`.
- **Operation frequency among solved real problems:** concat 49, mul 29, abs_diff 24, add 23, rev_concat 11 (≈ 36/21/18/17/8 %; biased toward easily-solved concat). Generator target weights: keep all five, lean toward the three arithmetic ops (the real headroom).
- **Symbol alphabet (glyphs that encode digits):** `! " # $ % & ' ( ) / : < > ? @ [ \ ] ^ ` { | }` (operators `+ - *` may also appear as symbols).
- **Operator-glyph distribution (PASS-1 requirement, not a follow-up):** real operators are usually `+`/`-`/`*` (~600 each) but there is a long tail of *other* glyphs at position 2, and `+`/`-`/`*` themselves **also appear as digit-symbols** elsewhere (role is purely positional). cipher and bit both lost points to OOD generators — so distribution match is in pass 1. The core therefore: (a) samples the operator glyph mostly from `{+,-,*}` (weighted ~majority) but with a calibrated tail drawn from the full alphabet; (b) since an operator glyph may *also* encode a digit, parsing is **purely positional** (index 2 is the operator, indices 0/1/3/4 are operands) — do NOT keep operators disjoint from digit-glyphs. The `op_of` map is keyed by the actual sampled operator glyph, not a fixed `{+,-,*}` set.
- **Determinacy mix (PASS-1):** real `_deduce` problems include near-determined cases, not only the easily-solved ones. Pass-1 emits a small fraction (~10–15%) of *near-determined* instances (one glyph pinned only by the answer, or query op witnessed exactly once) alongside the well-determined majority, so the model learns to commit under mild uncertainty rather than only on over-specified inputs. Fully-underdetermined (ambiguous) instances stay out (that is the `_guess` sibling's failure mode).
- **Demos per problem:** min 3, max 5, mean 4.09 → val `difficulty` default = 4.
- **Why ~0% on new rules:** training CoT came from `reasoners/cryptarithm.py`, which expresses ONLY concat/rev_concat (~8% coverage). The model **only ever saw concatenation traces** — never the arithmetic family. Closing that is the lever.

Port the 5 `OPS`, `num_to_digits`, and concat 4-digit padding **directly from `investigators/cryptarithm_deduce.py`** so generated semantics match real exactly. (That file uses `signal.SIGALRM` — Linux-only; port the pure logic, not the harness.)

---

## File structure

- **Create** `reasoners/cryptarithm_rule.py` — shared, dependency-light rule core (sampling, encode/decode, compute, self-verify). Imported by BOTH consumers so the val number and the training data share one definition (DRY). No I/O, no CoT.
- **Create** `val/generators/cryptarithm.py` — val consumer: `generate(seed, difficulty)`, `rule_signature(seed)`, `register(...)`. Mirrors `val/generators/cipher.py`.
- **Modify** `val/generators/__init__.py:32` — add `cryptarithm` to the self-registering import.
- **Create** `val/tests/test_cryptarithm_generator.py` — structural + self-consistency tests.
- **Create** `reasoners/cryptarithm_deduce_core.py` — Windows-safe, *instrumented* port of `investigators/cryptarithm_deduce.py` (no `signal.SIGALRM`; node-cap instead). Adds a `trace=True` mode that records the actual decisions the search makes (operator hypotheses tried, digit assignments forced, prunes/backtracks, final recovered map). This is what makes the CoT a *real* deduction rather than a clairvoyant read-off.
- **Create** `reasoners/cryptarithm_trace.py` — training consumer: renders the decision log into CoT. Two deterministic renderers: `reasoning_cryptarithm_arith` (deduction-CSP log) and `reasoning_cryptarithm_propagate` (compute-and-correct, single-pass-executable). Both reach the same boxed answer.
- **Create** `reasoners/cryptarithm_teacher.py` — offline teacher-LLM trace generator with a rejection gate (keep only traces whose `\boxed{}` matches the constructed answer). Not imported by val. Optional arm; economics measured in Phase 4.4.
- **Create** `reasoners/tests/test_cryptarithm_deduce_core.py`, `reasoners/tests/test_cryptarithm_trace.py`, `reasoners/tests/test_cryptarithm_teacher.py` — solver-soundness, trace-correctness (every demo verified, deduction markers, token budget), and rejection-gate (mocked teacher) tests.
- **Modify** the training-corpus build entry point — **LOCATED (Phase 0.2, 2026-06-03):** `corpus.py` `main()`, immediately after the real-problem reasoning-file loop (~`corpus.py:227`), before the augmentations block. Add `build_cryptarithm_rows(n, style, holdout)` returning `CorpusEntry` objects (schema `corpus.py:87-96`: problem_id/category/tokens/mask/masked_token_count/unmasked_token_count/answer/included) appended to `entries` before the sort+index-write (`corpus.py:279-285`). CoT today: `reasoners/cryptarithm.py:36` `reasoning_cryptarithm` (concat-only), mapped at `reasoning.py:53`, written to `reasoning/{id}.txt` (`reasoning.py:186-188`) then tokenized at `corpus.py:185-194`. **NOTE: the ×12 duplication is ALREADY GONE** (removed in `f42e3fbcf6`) — current corpus has each of the ~54 real cryptarithm traces once. So Phase 5 = *ADD N distinct generated rows* (no copies to remove); the new rows must carry tokens+mask in the same format (reuse `corpus.py`'s tokenizer helper rather than re-implementing).

---

## Phase 0 — Grounding

### Task 0.1: Confirm the rule family against real data — ✅ DONE 2026-06-03

Verified via a brute-fit probe + reading `investigators/cryptarithm_deduce.py` and solved `investigations/*.txt`. Outcome recorded in "Calibration facts" above: the family is injective symbol→digit × per-problem-deduced operator semantics over FIVE ops {add, abs_diff, mul, concat, rev_concat} × re-encode. The earlier fixed-mapping hypothesis was falsified (1/25 fit). No commit (no code changed). Example confirmed: problem `0f6436da`, `+`=add, `>#+>|` → `12+13=25` → `#%`.

### Task 0.2: Locate the training-corpus build entry point

**Files:** none (investigation).

- [ ] **Step 1:** Identify where `cryptarithm_deduce` training rows + CoT traces are assembled into the SFT corpus (the path that produced snapshot `training/sft/04-08-16-14`). Read `reasoning.py`, `corpus.py`, `generate_csv.py`, `train_sft.py` and trace how `reasoners.cryptarithm.reasoning_cryptarithm` output reaches the corpus, and where the ×12 exact-copy duplication of the ~54 real cryptarithm traces is introduced.

- [ ] **Step 2: Record** the exact file + function + insertion point in the "File structure" section above (replace the "located in Phase 0" note). This unblocks Phase 4–5. **No commit.**

---

## Phase 1 — Shared rule core (`reasoners/cryptarithm_rule.py`)

**Files:**
- Create: `reasoners/cryptarithm_rule.py`
- Test: `val/tests/test_cryptarithm_generator.py` (core tests live with the val tests; the core has no separate package test dir requirement)

### Task 1: Rule sampling + encode/decode/compute (FIVE ops, per-problem semantics)

> **Design revisions (supersede the illustrative code blocks below — apply these when implementing):**
> 1. **Operators are positional glyphs, not a fixed `{+,-,*}` set.** Replace the fixed `OPERATORS = ("+","-","*")` with a per-rule sampled operator glyph: mostly `+`/`-`/`*` but with a calibrated long tail from the full alphabet (`OPERATOR_WEIGHTS` ≈ {`+`:1, `-`:1, `*`:1, plus a small uniform tail}). An operator glyph **may also encode a digit** — do not exclude it from `SYMBOLS`. `op_of` is keyed by whatever glyphs the rule actually uses at position 2; parsing is purely positional (idx 2 = operator). Tests assert each used operator glyph maps to one of the five ops, and that a rule can place an operator glyph that is also a digit-glyph.
> 2. **`sample_problem` gains a `determinacy` knob.** Default emits well-determined instances (current guard); with probability ~0.10–0.15 emit a *near-determined* instance (drop cover-demos so one needed glyph is witnessed exactly once, or the query op appears exactly once) — still uniquely solvable, but not over-specified. Add a test that the near-determined branch is still self-consistent and uniquely solvable by the Phase-4.1 solver.
> 3. **`sample_problem` gains a `difficulty_profile`/curriculum hook** (used in Phase 6): a profile selects the operator mix and operand ranges (easy = concat + small add/abs_diff; hard = mul + multi-operator). Default profile = the measured real mix.
> 4. **Shared prompt-wrapper pool (instantiates super-plan #4).** Add `WRAPPERS` (the exact real header + a few light paraphrases) and a `render_prompt(examples, q_input, wrapper_seed)` helper *in the rule core*, so the val generator and the training emitter draw from the **same** pool (no drift). Default `wrapper_seed=0` = the exact real wrapper (val uses this so the measured number reflects the real format); training samples ~15–20% paraphrases. Paraphrases vary only surrounding prose/ordering — never the `s0 s1 op s3 s4` structure, the `=` joiner, or `\boxed{}`. Add a test that every wrapper still contains the demos verbatim and a parseable query.

- [ ] **Step 1: Write the failing test** (append to `val/tests/test_cryptarithm_generator.py`):

```python
from reasoners.cryptarithm_rule import (
    OP_NAMES,
    OPERATORS,
    build_rule,
    num_to_digits,
)


def test_build_rule_is_deterministic():
    a, b = build_rule(7), build_rule(7)
    assert a.digit_to_sym == b.digit_to_sym
    assert a.op_of == b.op_of


def test_digit_sym_is_an_injective_map_over_all_ten_digits():
    rule = build_rule(3)
    assert set(rule.digit_to_sym) == set(range(10))
    assert len(set(rule.digit_to_sym.values())) == 10  # all-different glyphs
    for d, s in rule.digit_to_sym.items():
        assert rule.sym_to_digit[s] == d


def test_every_operator_glyph_is_assigned_one_of_the_five_ops():
    rule = build_rule(0)
    assert set(rule.op_of) == set(OPERATORS)
    assert all(name in OP_NAMES for name in rule.op_of.values())
    assert set(OP_NAMES) == {"add", "abs_diff", "mul", "concat", "rev_concat"}


def test_num_to_digits_matches_investigator():
    assert num_to_digits(0) == (0,)
    assert num_to_digits(25) == (2, 5)
    assert num_to_digits(137) == (1, 3, 7)


def test_concat_result_is_always_four_digits():
    rule = build_rule(5)
    rule.op_of["+"] = "concat"  # force concat to check padding
    digits = rule.result_digits("+", 5, 12)  # 05 concat 12 -> 0512
    assert digits == (0, 5, 1, 2)


def test_arith_round_trips_through_encoding():
    rule = build_rule(11)
    rule.op_of["+"] = "add"
    inp, out = rule.encode_example("+", left=(1, 2), right=(1, 3))  # 12 + 13 = 25
    # decode the output glyphs back to digits via the same map
    assert tuple(rule.sym_to_digit[c] for c in out) == (2, 5)
    assert inp[2] == "+" and len(inp) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --frozen pytest val/tests/test_cryptarithm_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reasoners.cryptarithm_rule'`.

- [ ] **Step 3: Write minimal implementation** (`reasoners/cryptarithm_rule.py`) — semantics ported verbatim from `investigators/cryptarithm_deduce.py`:

```python
"""Shared rule core for the cryptarithm arithmetic-cipher family.

A rule = (injective symbol<->digit map) x (per-operator-glyph operation, one of
FIVE: add/abs_diff/mul/concat/rev_concat) x re-encoding of the result digits
back to symbols. Semantics ported from investigators/cryptarithm_deduce.py (the
authoritative full-family solver). Calibrated to the verified real
cryptarithm_deduce distribution (see docs/superpowers/plans/2026-06-02-
cryptarithm-p1.md): 5-char inputs `s0s1<op>s3s4`, operands are two-digit
numbers, operator semantics are deduced per problem. Pure: no I/O, no CoT.
Imported by val/generators/cryptarithm.py and reasoners/cryptarithm_trace.py so
the eval and the training data share one definition.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Glyphs that encode digits. Operator glyphs '+','-','*' are kept disjoint from
# this set so positional parsing is unambiguous (real data overlaps them, but
# disjoint is a faithful, cleaner subset).
SYMBOLS = list("!\"#$%&'()/:<>?@[]^`{|}\\")
OPERATORS = ("+", "-", "*")
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


def num_to_digits(n: int) -> tuple[int, ...]:
    """Digits of *n*, most-significant first (ported from the investigator)."""
    if n == 0:
        return (0,)
    d = []
    while n > 0:
        d.append(n % 10)
        n //= 10
    return tuple(reversed(d))


@dataclass
class CryptRule:
    digit_to_sym: dict[int, str]
    sym_to_digit: dict[str, int]
    op_of: dict[str, str]  # operator glyph -> op name in OP_NAMES

    def apply(self, op: str, left: int, right: int) -> int:
        return _OPS[self.op_of[op]](left, right)

    def result_digits(self, op: str, left: int, right: int) -> tuple[int, ...]:
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
    names = list(_OP_WEIGHTS)
    weights = [_OP_WEIGHTS[n] for n in names]
    op_of = {op: rng.choices(names, weights=weights)[0] for op in OPERATORS}
    return CryptRule(digit_to_sym, sym_to_digit, op_of)


def sample_problem(
    seed: int, difficulty: int
) -> tuple[CryptRule, list[tuple[str, str]], str, str]:
    """Sample one WELL-DETERMINED instance for this rule. Single source of truth
    for both the val generator and the training emitter (avoids drift between the
    measured number and the trained data).

    Returns (rule, examples, q_input, q_answer) where examples is a list of
    (input_value, output_value). Guarantees: (a) the query operator appears in a
    demo, and (b) every glyph in q_input + q_answer is witnessed in the demos
    (so symbol->digit and the operation are deducible).
    """
    rng = random.Random(seed)
    rule = build_rule(seed)

    def pair() -> tuple[int, int]:
        return rng.randint(0, 9), rng.randint(0, 9)

    q_op = rng.choice(OPERATORS)
    q_input, q_answer = rule.encode_example(q_op, pair(), pair())

    # First demo uses the query operator (so its semantics are always witnessed);
    # the rest pick any operator. Shuffle so position carries no signal.
    triples = [(q_op, pair(), pair())] + [
        (rng.choice(OPERATORS), pair(), pair()) for _ in range(difficulty - 1)
    ]
    rng.shuffle(triples)
    examples = [rule.encode_example(op, lo, ro) for op, lo, ro in triples]

    needed = set(q_input + q_answer) - set(OPERATORS)

    def witnessed() -> set[str]:
        return set("".join(i + o for i, o in examples)) - set(OPERATORS)

    # Append cover demos (always using the query operator) until every needed
    # glyph is witnessed, or a bounded number of tries elapse.
    for _ in range(12):
        if needed <= witnessed() and any(i[2] == q_op for i, _ in examples):
            break
        examples.append(rule.encode_example(q_op, pair(), pair()))
    return rule, examples, q_input, q_answer
```

A test for the sampler (append to `val/tests/test_cryptarithm_generator.py`):

```python
from reasoners.cryptarithm_rule import sample_problem


def test_sample_problem_is_well_determined():
    for seed in range(25):
        rule, examples, q_input, q_answer = sample_problem(seed, 4)
        ops = set("+-*")
        witnessed = set("".join(i + o for i, o in examples)) - ops
        assert (set(q_input + q_answer) - ops) <= witnessed
        assert any(i[2] == q_input[2] for i, _ in examples)
        # answer reproduces under the rule
        _, ans = rule.encode_example(
            q_input[2],
            (rule.sym_to_digit[q_input[0]], rule.sym_to_digit[q_input[1]]),
            (rule.sym_to_digit[q_input[3]], rule.sym_to_digit[q_input[4]]),
        )
        assert ans == q_answer
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --frozen pytest val/tests/test_cryptarithm_generator.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Format + lint**

Run: `uv run --frozen ruff format reasoners/cryptarithm_rule.py val/tests/test_cryptarithm_generator.py` then `uv run --frozen ruff check reasoners/cryptarithm_rule.py val/tests/test_cryptarithm_generator.py`
Expected: no errors. (Note: the repo hook forbids the literal `except Exception` in `.py` files — none used here.)

- [ ] **Step 6: Commit**

```
git add reasoners/cryptarithm_rule.py val/tests/test_cryptarithm_generator.py
git commit -m "feat(cryptarithm): shared arithmetic-cipher rule core (sample/encode/decode)"
```

---

## Phase 2 — Val generator (`val/generators/cryptarithm.py`)

**Files:**
- Create: `val/generators/cryptarithm.py`
- Modify: `val/generators/__init__.py:32`
- Test: `val/tests/test_cryptarithm_generator.py`

### Task 2: Generate a well-determined new-rule problem

A "deduce" instance must be **well-determined**: every operator used in the query also appears in the demos, and every glyph in the query operands+answer appears somewhere in the demos (so the map is inducible). The generator enforces this.

- [ ] **Step 1: Write the failing test** (append):

```python
from reasoners.store_types import Problem
from val.generators.cryptarithm import generate, rule_signature


def _check_problem_is_self_consistent(p: Problem):
    from reasoners.cryptarithm_rule import build_rule

    seed = int(p.id.rsplit("-", 1)[1])
    rule = build_rule(seed)
    # every demo reproduces under the rule
    for ex in p.examples:
        op = ex.input_value[2]
        left = (rule.sym_to_digit[ex.input_value[0]], rule.sym_to_digit[ex.input_value[1]])
        right = (rule.sym_to_digit[ex.input_value[3]], rule.sym_to_digit[ex.input_value[4]])
        _, out = rule.encode_example(op, left, right)
        assert out == ex.output_value
    # the query answer reproduces under the rule
    q = p.question
    op = q[2]
    left = (rule.sym_to_digit[q[0]], rule.sym_to_digit[q[1]])
    right = (rule.sym_to_digit[q[3]], rule.sym_to_digit[q[4]])
    _, ans = rule.encode_example(op, left, right)
    assert ans == p.answer


def test_generate_shape_and_metadata():
    p = generate(0, 4)
    assert p.category == "cryptarithm_deduce"
    assert p.id == "val-cryptarithm-0"
    assert len(p.examples) == 4
    assert len(p.question) == 5 and p.question[2] in "+-*"
    assert p.prompt.startswith("In Alice's Wonderland, a secret set of transformation rules")
    assert "Now, determine the result for:" in p.prompt


def test_generate_is_self_consistent_and_well_determined():
    for seed in range(25):
        p = generate(seed, 4)
        _check_problem_is_self_consistent(p)
        # well-determined: query operator appears in a demo
        assert any(ex.input_value[2] == p.question[2] for ex in p.examples)
        # every glyph needed to read the query AND produce its answer is witnessed
        ops = {"+", "-", "*"}
        demo_glyphs = set(
            "".join(ex.input_value + ex.output_value for ex in p.examples)
        ) - ops
        needed = set(p.question + p.answer) - ops
        assert needed <= demo_glyphs


def test_rule_signature_is_stable_and_distinct():
    assert rule_signature(5) == rule_signature(5)
    assert rule_signature(5) != rule_signature(6)


def test_difficulty_controls_demo_count():
    assert len(generate(1, 3).examples) == 3
    assert len(generate(1, 5).examples) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --frozen pytest val/tests/test_cryptarithm_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'val.generators.cryptarithm'`.

- [ ] **Step 3: Write minimal implementation** (`val/generators/cryptarithm.py`):

```python
"""Reference generator: a brand-new arithmetic-cipher cryptarithm problem.

Calibrated to the measured real cryptarithm_deduce distribution (see
docs/superpowers/plans/2026-06-02-cryptarithm-p1.md): 5-char inputs
`a0a1<op>b0b1`, operators '+'/'-'/'*' remapped to integer ops, operands are
two-digit numbers, results re-encoded to glyphs. The category is ~0% solved by
the deterministic solver (it only handles concatenation), so this is the clean
new-rule signal on the #1 lever. Generated instances are well-determined: every
query operator and glyph is witnessed in the demos.
"""

from reasoners.cryptarithm_rule import OPERATORS, build_rule, sample_problem
from reasoners.store_types import Example, Problem
from val.generators import register

_PROMPT_HEADER = (
    "In Alice's Wonderland, a secret set of transformation rules is applied "
    "to equations. Below are a few examples:"
)


def rule_signature(seed: int) -> str:
    rule = build_rule(seed)
    smap = ",".join(f"{d}{rule.digit_to_sym[d]}" for d in range(10))
    osem = ",".join(f"{op}{rule.op_of[op]}" for op in OPERATORS)
    return f"{smap}|{osem}"


def generate(seed: int, difficulty: int) -> Problem:
    _rule, raw, q_input, q_answer = sample_problem(seed, difficulty)
    examples = [Example(input_value=i, output_value=o) for i, o in raw]
    example_lines = "\n".join(f"{e.input_value} = {e.output_value}" for e in examples)
    prompt = (
        f"{_PROMPT_HEADER}\n{example_lines}\n"
        f"Now, determine the result for: {q_input}"
    )
    return Problem(
        id=f"val-cryptarithm-{seed}",
        category="cryptarithm_deduce",
        examples=examples,
        question=q_input,
        answer=q_answer,
        prompt=prompt,
    )


register("cryptarithm_deduce", generate, rule_signature)
```

**Deducibility guard (required for a well-determined "deduce" instance).** The query is deducible only if (a) the query operator appears in ≥1 demo (so its operation is known), and (b) every glyph the solver must *read or produce* — the query operand glyphs AND the query **answer** glyphs — is witnessed in the demos (so its digit is pinned). With five ops this matters more: `concat` emits 4 answer glyphs including a possible leading-zero glyph. Implement the guard in Step 3: after assembling demos, compute `needed = set(q_input + q_answer) - set(OPERATORS)` and `demo_glyphs = set("".join(e.input_value + e.output_value for e in examples)) - set(OPERATORS)`; if `needed - demo_glyphs` is non-empty, append extra demos (operands drawn from the missing digits, using the query operator) until covered. The test `test_generate_is_self_consistent_and_well_determined` asserts `needed <= demo_glyphs`.

- [ ] **Step 4: Wire up self-registration** — edit `val/generators/__init__.py:32`:

```python
from val.generators import bit_manipulation, cipher, cryptarithm  # noqa: E402,F401
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --frozen pytest val/tests/test_cryptarithm_generator.py -v`
Expected: PASS (all Phase 1 + Phase 2 tests).

- [ ] **Step 6: Run the full val suite (no regressions)**

Run: `uv run --frozen pytest val/tests -v`
Expected: PASS (previous 44 + the new cryptarithm tests).

- [ ] **Step 7: Format + lint**

Run: `uv run --frozen ruff format val/generators/cryptarithm.py val/generators/__init__.py val/tests/test_cryptarithm_generator.py` then `uv run --frozen ruff check val/generators/cryptarithm.py val/generators/__init__.py val/tests/test_cryptarithm_generator.py`
Expected: no errors.

- [ ] **Step 8: Commit**

```
git add val/generators/cryptarithm.py val/generators/__init__.py val/tests/test_cryptarithm_generator.py
git commit -m "feat(val): clean new-rule cryptarithm_deduce generator (arithmetic-cipher family)"
```

---

## Phase 3 — Measure the clean baseline

The GPU/Kaggle run is user-executed; this phase confirms the generator is *eval-ready* and documents the run.

### Task 3: Confirm eval wiring + difficulty default

**Files:**
- Modify (if needed): `val/run_vllm.py` (already iterates `GENERATORS`; cryptarithm auto-included once registered)
- Modify (if needed): `val/kaggle_newrule_eval_standalone.py` (self-contained Kaggle cell — add an inlined cryptarithm generator mirroring Phase 2)

- [ ] **Step 1:** Confirm `build_synthetic_valset` picks up `cryptarithm_deduce` automatically (it loops `sorted(GENERATORS.items())`; no per-category difficulty override needed — real mean demos ≈ 4 = the default). No code change expected; verify by:

Run: `uv run python -c "import val.generators as g; print(sorted(g.GENERATORS))"`
Expected: includes `'cryptarithm_deduce'`.

- [ ] **Step 2:** Port the Phase 2 generator into `val/kaggle_newrule_eval_standalone.py` as an inlined `generate_cryptarithm(seed, difficulty)` (the standalone cell has no repo imports — inline `SYMBOLS`, `OPERATORS`, the rule core, and the generator), add `CRYPTARITHM_DIFFICULTY = 4`, and a run line `problems += [generate_cryptarithm(s, CRYPTARITHM_DIFFICULTY) for s in range(PER_CATEGORY)]`. Keep byte-identical rule logic to `reasoners/cryptarithm_rule.py` so local and Kaggle numbers agree.

- [ ] **Step 3: Format + lint + local sanity**

Run: `uv run --frozen ruff format val/kaggle_newrule_eval_standalone.py` then `uv run --frozen ruff check val/kaggle_newrule_eval_standalone.py`
Expected: no errors.

- [ ] **Step 4: Commit**

```
git add val/run_vllm.py val/kaggle_newrule_eval_standalone.py
git commit -m "feat(val): wire cryptarithm_deduce into synthetic new-rule eval"
```

- [ ] **Step 5 (USER-RUN on Kaggle):** Run the standalone eval; record the **clean** cryptarithm_deduce new-rule accuracy in `improvement-roadmap` memory. This number replaces the biased ~0% floor. **Expectation:** the *current* adapter scores low on this clean set (it was trained on ×12 copies of ~54 concatenation-shaped traces, not the arithmetic family) — this is the headroom Phase 4–5 closes. If it is already high, re-examine whether the real arithmetic family differs from our generated semantics (Phase 0 mismatch) before investing in Phase 4.

---

## Phase 4 — Training emitter with genuine-deduction CoT

**Files:**
- Create: `reasoners/cryptarithm_deduce_core.py` (instrumented, Windows-safe solver port)
- Create: `reasoners/cryptarithm_trace.py`
- Test: `reasoners/tests/test_cryptarithm_deduce_core.py`
- Test: `reasoners/tests/test_cryptarithm_trace.py`

**Design decision (user-confirmed): the CoT teaches GENUINE DEDUCTION, not verify-only.** The roadmap's verified root cause of ~0% is "CoT teaches application, not induction." A trace that merely *states* the digit map (read off `build_rule(seed)`) and then verifies it reproduces that failure — the model sees the answer handed over, never the recovery. For generalization to **new** rules at greedy inference (the only "search" available is in-CoT self-verification/backtracking), the trace must demonstrate the *recovery process*:

1. **Structure recognition** — "each input is `s0 s1 op s3 s4`: two 2-digit numbers joined by an operator glyph; the output spells the result's digits in the same secret alphabet."
2. **Operator resolution** — for each operator glyph: detect `concat`/`rev_concat` **structurally** (output glyphs == `(s0,s1,s3,s4)` or `(s3,s4,s0,s1)` — no digit values needed); else it is arithmetic and the **output length** narrows it (abs_diff ≤2, add ≤3, mul ≤4 digits), confirmed numerically once digits are pinned.
3. **Digit constraint propagation** — pin glyph→digit from the arithmetic demos, showing forced deductions **and at least one ruled-out guess** ("if `@`=3 then demo 2 needs `!`=12 — impossible, discard"), modeling self-correction.
4. **Full verification** — re-check every demo numerically (decode → op → encode → matches).
5. **Apply to query** → `\boxed{}`.

To keep every claimed deduction **sound** (not hand-waved), the trace is rendered from the *actual decision log of an instrumented solver* run on the generated problem — not from peeking at the planted rule. Because `sample_problem` guarantees well-determinedness, the solver recovers exactly the planted map, so the log is bounded and the boxed answer is correct by construction. Mirror the formatting conventions of `reasoners/cryptarithm.py` (glyph quoting, `\boxed{}` ending) so traces blend with existing data. **Token budget:** cap the rendered trace (well under the 7680-gen / 8192-ctx limit); limit explicit backtracks to ≤2 per problem.

### Task 4.1: Instrumented, Windows-safe solver port

**Files:**
- Create: `reasoners/cryptarithm_deduce_core.py`
- Test: `reasoners/tests/test_cryptarithm_deduce_core.py`

Port the search from `investigators/cryptarithm_deduce.py` **without** `signal.SIGALRM` (Windows-incompatible) — use a node-cap (the probe lesson from [[verify-assumptions-against-real-data]]: incremental constraint checks + most-constrained-first ordering + node cap; naive backtracking hung 13 min). Add a `trace: bool` flag: when set, `solve_problem` returns `(answer, info, decision_log)` where `decision_log` is a list of structured records — `{"kind": "op_struct"|"op_len"|"assign"|"prune"|"final", ...}` — capturing operator resolutions, forced digit assignments, prunes/backtracks, and the recovered map. Without the flag it behaves exactly like the investigator (drop-in).

- [ ] **Step 1: Failing test** (`reasoners/tests/test_cryptarithm_deduce_core.py`): for `seed in range(50)`, build a problem via `sample_problem(seed, 4)`, feed its demos+query to the ported solver, and assert it (a) returns the **planted answer**, and (b) recovers the **planted glyph→digit map and op_of** for every glyph it resolves. This is the soundness gate — the narrator can only be faithful if the solver it narrates is correct.
- [ ] **Step 2: Run → fail** (`ModuleNotFoundError`).
- [ ] **Step 3: Implement** the port (pure logic from the investigator + node-cap + decision_log). Reuse `reasoners.cryptarithm_rule` constants where possible (`OP_NAMES`, `num_to_digits`) to avoid a second source of truth.
- [ ] **Step 4: Run → pass** (50/50 recovered).
- [ ] **Step 5: Format + lint** (`uv run --frozen ruff format/check`).
- [ ] **Step 6: Commit** — `feat(cryptarithm): Windows-safe instrumented deduce solver (decision log)`.

### Task 4.2: Render the decision log into a deduction CoT

- [ ] **Step 1: Write the failing test** (`reasoners/tests/test_cryptarithm_trace.py`):

```python
from reasoners.cryptarithm_rule import build_rule
from reasoners.cryptarithm_trace import make_trace_problem, reasoning_cryptarithm_arith


def test_trace_ends_with_correct_boxed_answer():
    problem, answer = make_trace_problem(seed=0, difficulty=4)
    trace = reasoning_cryptarithm_arith(problem, answer)
    assert trace is not None
    assert trace.rstrip().endswith("\\boxed{" + answer + "}")


def test_trace_verifies_every_demo():
    problem, answer = make_trace_problem(seed=3, difficulty=5)
    trace = reasoning_cryptarithm_arith(problem, answer)
    # each demo's input and output appear in the verification body
    for ex in problem.examples:
        assert ex.input_value in trace
        assert ex.output_value in trace


def test_trace_names_each_demos_operation():
    problem, answer = make_trace_problem(seed=7, difficulty=5)
    rule = build_rule(7)
    words = {
        "add": "sum",
        "abs_diff": "absolute difference",
        "mul": "product",
        "concat": "concatenation",
        "rev_concat": "reverse concatenation",
    }
    trace = reasoning_cryptarithm_arith(problem, answer).lower()
    for op in {ex.input_value[2] for ex in problem.examples}:
        assert words[rule.op_of[op]] in trace


def test_trace_shows_genuine_deduction_not_just_verification():
    """Guard against regressing to verify-only: the trace must contain the
    recovery process, not merely a stated map + checks."""
    seen_struct = seen_len = seen_prune = 0
    for seed in range(40):
        problem, answer = make_trace_problem(seed, 5)
        t = reasoning_cryptarithm_arith(problem, answer).lower()
        # structure recognition is always present
        assert "s0" in t and "operator" in t
        # at least some problems exercise each deduction mode across the sweep
        if "concatenat" in t:  # structural operator resolution
            seen_struct += 1
        if "digit" in t and ("at most" in t or "so it is" in t):  # length narrowing
            seen_len += 1
        if "discard" in t or "rules out" in t or "impossible" in t:  # a prune
            seen_prune += 1
    assert seen_struct > 0  # concat/rev_concat resolved structurally
    assert seen_prune > 0  # at least one ruled-out guess shown somewhere


def test_trace_stays_within_token_budget():
    # well under the 7680-gen / 8192-ctx limit; ~4 chars/token heuristic
    for seed in range(40):
        problem, answer = make_trace_problem(seed, 5)
        trace = reasoning_cryptarithm_arith(problem, answer)
        assert len(trace) < 12000  # ~3000 tokens, generous headroom
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --frozen pytest reasoners/tests/test_cryptarithm_trace.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reasoners.cryptarithm_trace'`.

- [ ] **Step 3: Write the implementation** (`reasoners/cryptarithm_trace.py`). It renders the **decision log from Task 4.1's instrumented solver** — it does NOT read `build_rule(seed)` to state the map. (`build_rule` is used only by the test, to assert the recovered map is correct.)

```python
"""Genuine-deduction CoT for the arithmetic-cipher cryptarithm family.

Unlike reasoners/cryptarithm.py (which narrates an already-known concatenation
rule), this teaches *deduction*: recognize the 5-char structure, resolve each
operator's operation (concat/rev_concat structurally; arithmetic by output
length, confirmed numerically), propagate digit constraints from the demos
(showing forced assignments AND >=1 ruled-out guess), verify every demo, then
apply to the query. The narration is rendered from the ACTUAL decision log of
the instrumented solver (reasoners/cryptarithm_deduce_core.solve_problem with
trace=True) run on this problem, so every claimed deduction is sound. Built on
the shared rule core so the problem is well-determined and the boxed answer is
correct by construction.
"""

from __future__ import annotations

from reasoners.cryptarithm_deduce_core import solve_problem
from reasoners.cryptarithm_rule import sample_problem
from reasoners.store_types import Example, Problem

_OP_WORD = {
    "add": "sum",
    "abs_diff": "absolute difference",
    "mul": "product",
    "concat": "concatenation",
    "rev_concat": "reverse concatenation",
}
_PROMPT_HEADER = (
    "In Alice's Wonderland, a secret set of transformation rules is applied "
    "to equations. Below are a few examples:"
)


def make_trace_problem(seed: int, difficulty: int) -> tuple[Problem, str]:
    """A well-determined problem + its verified answer, from the SHARED sampler
    (same instances the val generator measures — no drift)."""
    _rule, raw, q_input, q_answer = sample_problem(seed, difficulty)
    examples = [Example(input_value=i, output_value=o) for i, o in raw]
    prompt = (
        f"{_PROMPT_HEADER}\n"
        + "\n".join(f"{e.input_value} = {e.output_value}" for e in examples)
        + f"\nNow, determine the result for: {q_input}"
    )
    problem = Problem(
        id=f"train-cryptarithm-{seed}",
        category="cryptarithm_deduce",
        examples=examples,
        question=q_input,
        answer=q_answer,
        prompt=prompt,
    )
    return problem, q_answer


def _render(record: dict) -> str:
    """Turn one decision-log record into a sentence. Record kinds:
    - op_struct: {op, direction}            -> "operator 'X' just concatenates ..."
    - op_len:    {op, out_len, candidates}  -> "output is N digits, so 'X' is +/-/* ..."
    - assign:    {sym, digit, why}          -> "demo ... forces glyph 's'=d because ..."
    - prune:     {sym, digit, conflict}     -> "if 's'=d then ... — impossible, discard."
    - verify:    {inp, out, left, right, op_word, digits, check, ok}
    - final:     {map, op_of}               -> recovered rule summary
    """
    ...  # one f-string branch per kind, prose mirroring reasoners/cryptarithm.py


def reasoning_cryptarithm_arith(problem: Problem, answer: str) -> str | None:
    data = {
        "examples": [
            {"input_value": e.input_value, "output_value": e.output_value}
            for e in problem.examples
        ],
        "question": problem.question,
    }
    predicted, _info, log = solve_problem(data, trace=True)
    if predicted != answer:  # well-determined => should never happen; guard anyway
        return None
    lines = [
        "I must deduce the secret rule from the examples, then apply it.",
        (
            "Each input is `s0 s1 op s3 s4`: two glyphs forming a 2-digit number, "
            "an operator glyph, then two glyphs forming a second 2-digit number. "
            "The output spells the result's digits in the same secret alphabet. "
            "So I need (a) what each operator does and (b) each glyph's digit."
        ),
        "",
    ]
    lines += [_render(r) for r in log]
    lines.append(f"\nSo the answer is \\boxed{{{answer}}}")
    return "\n".join(lines)
```

The `_render` branches are the only prose-authoring work; everything they state is backed by a real solver decision. Keep backtracks readable but bounded (Task 4.1 caps logged prunes at ≤2). If a future calibration uses non-`+/-/*` operator glyphs (see Calibration facts), `_render` needs no change — it reads the glyph from the record.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --frozen pytest reasoners/tests/test_cryptarithm_trace.py -v`
Expected: PASS (5 tests — boxed answer, every demo verified, op naming, genuine-deduction markers, token budget).

- [ ] **Step 5: Format + lint**

Run: `uv run --frozen ruff format reasoners/cryptarithm_trace.py reasoners/tests/test_cryptarithm_trace.py` then `uv run --frozen ruff check reasoners/cryptarithm_trace.py reasoners/tests/test_cryptarithm_trace.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```
git add reasoners/cryptarithm_trace.py reasoners/tests/test_cryptarithm_trace.py
git commit -m "feat(cryptarithm): genuine-deduction CoT rendered from instrumented solver"
```

### Task 4.3: Alternative renderer — constraint-propagation / compute-and-correct

**Files:** extend `reasoners/cryptarithm_trace.py` with a second renderer `reasoning_cryptarithm_propagate(problem, answer)`; test in `reasoners/tests/test_cryptarithm_trace.py`.

**Rationale:** a faithful CSP-search log narrates backtracking over a constraint store — something a greedy autoregressive model cannot maintain across one pass, so it may teach a procedure the model can't *execute* on a new problem. This renderer teaches a leaner, single-pass-executable strategy from the SAME decision log: (1) resolve each operator by output length/structure; (2) **read forced digits directly** from the demos where the arithmetic pins them (no enumeration narrated); (3) **compute each demo and, on any mismatch, correct the one offending glyph** (compute-and-correct — the realistic in-CoT self-repair the model CAN do greedily); (4) apply to query → `\boxed{}`.

- [ ] **Step 1:** Failing test — `reasoning_cryptarithm_propagate` ends with the correct boxed answer, references every demo, shows ≥1 explicit correction (`"recompute"`/`"correct"`/`"actually"`), and stays in budget. Both renderers reach the same answer for all seeds 0–40.
- [ ] **Step 2–4:** implement from the log (reuse `_render` helpers; different assembly), run → pass.
- [ ] **Step 5–6:** format/lint, commit — `feat(cryptarithm): compute-and-correct CoT renderer (single-pass-executable style)`.

### Task 4.4: Teacher-LLM rejection-sampled traces (don't overfit one template)

**Files:** `reasoners/cryptarithm_teacher.py` (offline generator; **not** imported by val); output `corpus/cryptarithm_teacher_traces.jsonl`; test `reasoners/tests/test_cryptarithm_teacher.py` (parse/format + the rejection gate, with the teacher call mocked).

**Rationale:** two deterministic renderers still share an authored surface form; the model can learn the form, not the skill (the original ~0% failure). We own the ground-truth answer, so we can use a strong teacher (external LLM is allowed for data-gen per the roadmap) to produce **natural, varied** reasoning, then **rejection-sample**: keep a trace only if its extracted `\boxed{}` equals the constructed answer (reuse `extract_final_answer`/`compare_answer` from `val/grading.py`). This is the roadmap's lever #6 applied to the #1 category.

- [ ] **Step 1:** Define the teacher prompt: give the demos + query (NOT the rule), ask for step-by-step induction ending in `\boxed{}`. Sampling (temp>0) is fine for data-gen (eval is greedy; only inference is). Batch over held-in seeds (skip `holdout_rules.json`).
- [ ] **Step 2:** Rejection gate — extract the boxed answer, `compare_answer` against the constructed answer; **drop** non-matching traces (log accept rate). No accept-rate floor assumed — measure it.
- [ ] **Step 3:** Test the gate with a mocked teacher (correct trace kept, wrong-answer trace dropped, malformed trace dropped). No live LLM call in tests.
- [ ] **Step 4–5:** format/lint, commit — `feat(cryptarithm): teacher-LLM rejection-sampled trace generator`.
- [ ] **Step 6 (USER-RUN):** run the teacher over the seed range; record accept rate + cost. This arm is optional if accept rate is too low to be economical — Phase 6 decides whether it earns its place.

---

## Phase 5 — Integrate into the training corpus (replace the ×12 exact copies)

**Files:**
- Modify: the corpus-build entry point recorded in Phase 0 Task 0.2.

### Task 5: Emit N distinct verified cryptarithm_deduce traces

- [ ] **Step 1:** At the insertion point from Phase 0, replace the duplicated ~54 cryptarithm_deduce traces (×12 exact copies ≈ 627 rows) with `N` **distinct** rule-core problems + traces. Make the builder take two parameters: `style ∈ {deduce, propagate, teacher, mixed}` (which renderer from Phase 4) and `n`/`curriculum`. Hold out the val rule signatures: skip any `seed` whose `rule_signature(seed)` is in `holdout_rules.json`. Build **two corpus arms** so Phase 6 can attribute the gain:
  - **Arm A — size-matched ablation:** `n ≈ 600` (same row budget, unchanged category weighting) → isolates the *diversity + arithmetic-coverage + CoT-style* effect against the frozen baseline. One arm per trace `style`.
  - **Arm B — scale + curriculum:** `n` large (e.g. 3–8k) drawn across a curriculum (`difficulty_profile` from Phase 1: concat/small-add → abs_diff/mul → multi-operator; demo counts 3–5; ~10–15% near-determined). Tests the hypothesis that the induction skill needs volume, not just diversity. Re-tune category weighting if `n` shifts the mix.

- [ ] **Step 2: Write the failing test** (in the corpus-build module's test file, or a new `reasoners/tests/test_cryptarithm_corpus.py`): assert the emitted cryptarithm_deduce rows are **distinct** (no exact-duplicate prompts) and that each row's boxed answer equals the constructed answer. Concrete assertion:

```python
import json


def test_emitted_cryptarithm_rows_are_distinct_and_correct():
    holdout = set(json.load(open("holdout_rules.json")))  # never leak val rules
    for style in ("deduce", "propagate"):
        rows = build_cryptarithm_rows(n=200, style=style, holdout=holdout)
        prompts = [r["prompt"] for r in rows]
        assert len(set(prompts)) == len(prompts)  # no exact-copy duplication
        for r in rows:
            assert r["completion"].rstrip().endswith("\\boxed{" + r["answer"] + "}")
            # no holdout leakage: none of the emitted rules are val rules
            assert r["rule_signature"] not in holdout
```

- [ ] **Step 3: Run test to verify it fails**, then implement `build_cryptarithm_rows`, then **Step 4: run to verify it passes** (commands mirror prior phases; exact module path from Phase 0).

- [ ] **Step 5: Verify no val leakage**

Run: `uv run --frozen pytest reasoners/tests/ val/tests/ -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```
git add <corpus-build-file> <its-test>
git commit -m "feat(cryptarithm): replace x12 duplicate traces with N distinct verified arithmetic instances"
```

- [ ] **Step 7 (USER-RUN):** Retrain on the size-matched Arm-A corpus (Tinker), convert (`adapter converter.ipynb` — settled DEAD lever, ship as-is), re-run the Phase 3 eval, record the clean cryptarithm_deduce accuracy + overall weighted estimate in `improvement-roadmap`. This is the first ablation point (diversity + arithmetic coverage, one trace style) vs the frozen baseline. The bake-off across styles/arms is Phase 6.

---

## Phase 6 — Experiment: trace-style bake-off + scale ablation (USER-RUN on Kaggle)

Each training run is expensive, so order them to learn the most per run and stop early when a factor is settled. **Change one factor at a time**; measure on the Phase-2 clean val holdout (the leaderboard-predictive harness) + the real snapshot-complement holdout.

- [ ] **Step 1 — trace style (Arm A, fixed n≈600):** train+eval `deduce`, `propagate`, and (if economical) `teacher`, plus a `mixed` arm. Pick the winner by clean cryptarithm_deduce accuracy. Hypothesis to test, not assume: `propagate`/`mixed` ≥ `deduce` because they are single-pass-executable. Record per-style numbers.
- [ ] **Step 2 — scale (Arm B, winning style):** train+eval at n ≈ 1.5k / 4k / 8k with the curriculum. Plot accuracy vs n to find the point of diminishing returns; watch overall weighted score for regressions in saturated categories (the cryptarithm rows displace others — keep category weighting balanced).
- [ ] **Step 3 — record** the best (style, n, curriculum) config + the new weighted estimate in `improvement-roadmap`. If clean accuracy is still low, run the Phase-4.4 completeness check: are failures induction (wrong map) or arithmetic (right map, bad compute)? That diagnosis routes the next investment.

---

## Phase 7 — RL is a GLOBAL lever, owned by the super plan (NOT defined here)

**Correction (2026-06-03): an earlier draft defined a standalone GRPO/DPO phase here — that was misscoped.** RL is already in the super plan (`C:\Users\porus\.claude\plans\you-are-acting-as-deep-fountain.md`) as **cross-category** levers, deliberately sequenced **#7 RFT/STaR → #9 DPO/KTO → #10 GRPO**, with GRPO explicitly gated on *"only once data + RFT plateau"* (high cost / unstable on a 30B MoE). The verifier reward (`compare_answer`) applies to every category, so RL must not be re-defined inside a single-category plan, and must not lead with GRPO. Do NOT duplicate it here.

**What THIS plan contributes to that global RL phase (the only cryptarithm-specific hooks):**
- **Unlimited verified RL prompts:** `sample_problem` yields problems with known-correct answers — exactly the prompt+gold pairs RFT/DPO/GRPO need, with no labeling cost, held out from `holdout_rules.json`.
- **The reward is already the val grader:** `compare_answer(extract_final_answer(gen), gold)` (`val/grading.py`) — reuse verbatim so RL optimizes the scored metric.
- **On-policy data for RFT/STaR (the first RL step):** harvest the SFT model's own *correct* greedy/sampled rollouts on forward-gen problems → re-SFT. This captures the model's working reasoning paths (better aligned than templated traces). For cryptarithm specifically, weigh RFT against just generating more forward-gen data — they overlap, so let the super plan's val-gated sequence decide the ratio.

When the super plan reaches its RL tier, point it at these hooks. Sequencing, method choice, and go/no-go stay in the super plan.

---

## Self-review notes

- **Spec coverage:** Lever #1 (cryptarithm_deduce forward-gen, Phases 1/4/5) + clean val generator (Phases 2/3) + **three CoT styles** (genuine-deduction, compute-and-correct, teacher-LLM rejection-sampled — Phase 4) **A/B'd on val** (Phase 6) + distribution-match in pass 1 (long-tail operators + near-determined mix — Calibration facts) + scale/curriculum arm (Phase 5 Arm B / Phase 6 Step 2) + data hygiene/no-leakage (Phase 5 holdout skip) + calibration-to-real (Phase 0). These are the cryptarithm-specific instantiations of super-plan levers #2/#3/#4/#5/#6. **RL (#7/#9/#10) stays GLOBAL in the super plan** — Phase 7 here only exposes cryptarithm hooks (verified prompts + the verifier reward); it does NOT define or own RL, and does not lead with GRPO. Out of scope here (separate plans): bit complex 3-input (lever #1).
- **Type consistency:** `CryptRule.digit_to_sym`/`sym_to_digit`/`op_of`, `build_rule(seed)`, `sample_problem(seed, difficulty)`, `apply`/`result_digits`/`encode_output`/`encode_example`, `num_to_digits`, `OPERATORS`/`OP_NAMES`/`SYMBOLS` used identically across Phases 1/2/4. **`sample_problem` is the single generation source** for both the val generator (Phase 2) and the training emitter (Phase 4) — no drift. `generate(seed, difficulty)`/`rule_signature(seed)` match the `GeneratorSpec` protocol in `val/generators/__init__.py`. `Problem`/`Example` fields match `reasoners/store_types.py`. Dependency direction is clean: `val.generators.cryptarithm`, `reasoners.cryptarithm_trace`, and `reasoners.cryptarithm_deduce_core` import the core `reasoners.cryptarithm_rule`; `cryptarithm_trace` also imports `cryptarithm_deduce_core` (reasoners never imports val).
- **Why deduction over verify-only (user-confirmed):** a verify-only trace states the map (read off `build_rule`) then checks it — this reproduces the roadmap's verified ~0% root cause ("CoT teaches application, not induction"). The chosen design renders the *real* decision log of the instrumented solver (Task 4.1), so the model sees operator resolution + digit constraint propagation + a ruled-out guess (self-correction — the only "search" available at greedy inference). Soundness is guaranteed because Task 4.1's solver is gated to recover the planted rule on all sampled seeds before any trace is rendered.
- **Open risks:** (R2 distribution mismatch) Task 0.1 done — rule family verified against `investigators/cryptarithm_deduce.py`; semantics ported verbatim. Remaining: the *operation mix* (concat/arith ratio) is calibrated only to the solver-biased solved set — Phase 3 Step 5 sanity-checks the current adapter's clean score before scaling. Operator-glyph long tail + near-determined mix are now **in pass 1** (Calibration facts), removing the cipher/bit OOD risk. (CoT-form risk) a CSP-search log may not be single-pass-executable — mitigated by the `propagate`/compute-and-correct renderer and a val bake-off (Phase 6 Step 1) that lets data decide rather than assumption. (Template overfit) two deterministic renderers share an authored surface form — mitigated by the teacher-LLM rejection-sampled variant (Task 4.4). (CoT cost) deduction traces are longer — capped backtracks ≤2 + a token-budget test. (Teacher economics) accept rate unknown — measured, not assumed; arm is optional if uneconomical. (RL scope) RL is NOT owned by this plan — it is a global super-plan lever (#7 RFT → #9 DPO → #10 GRPO, gated on plateau); Phase 7 here only exposes cryptarithm hooks, avoiding the earlier mistake of duplicating it and leading with GRPO. (Determinism) `random.Random(seed)` only — no `Date.now`/global RNG. (Hook constraints) no `except Exception`, no `&&` chaining, no grep/find in Bash.
```
## Addendum 2026-06-04: induction-search style (`induct`) — IMPLEMENTED

A third trace style `reasoning_cryptarithm_induct` was added (research basis:
memory `cryptarithm-trace-research`) to fix the verify-only read-off root cause.
Built TDD via subagent-driven development; full build plan at
`docs/superpowers/plans/2026-06-04-cryptarithm-induction-trace.md`.

Pipeline of the new style:
- `reasoners/cryptarithm_arith_render.py` `long_arith(op,a,b)` — renders add/abs_diff/mul
  digit-by-digit, reversed (least-significant-first) with explicit carries/borrows and
  single-digit partial products for mul (never atomic `a*b=c`).
- `reasoners/cryptarithm_deduce_core.py` `search_with_log(examples, query, op_info, mapping)`
  — builds a SHORT, replayable WINNING-PATH decision log toward the authoritative
  solution map: each glyph becomes known either FORCED by a demo (uniquely determined
  given current knowledge) or GUESSED (seed digit, with genuine ruled-out alternatives /
  light backtracking). Measured logs: min 9 / median 14 / max 20 records (the raw DFS it
  replaced was a ~200-step brute grind — unusable as training data). Record kinds:
  forced | assign | reject | backtrack | solution; replays to an injective map == solution.
- `reasoners/cryptarithm_trace.py` `reasoning_cryptarithm_induct(problem, answer)` —
  Step 1 operator elimination, Step 2 the narrated winning-path search with re-emitted
  `STATE |` blocks (inductive scratchpad), Step 3 round-trip verify of every demo with
  reversed-digit arithmetic + an injectivity check, Step 4 apply to the query + re-encode,
  then `\boxed`. Returns None for pure concat-shortcut queries.
- `corpus.py` `_crypt_renderer` gained `style="induct"` (falls back to `deduce` for
  concat-shortcut queries so no row is dropped).

Phase-6 A/B is now THREE arms: deduce | propagate | induct. NEXT (user-run, GPU/Kaggle):
set `CRYPT_STYLE="induct"` in corpus.py (keep `CRYPT_REPLACE_REAL=True`), `uv run corpus.py`
(adds `cryptarithm-induct-*`/fallback rows, drops the concat-only real traces), set the
train filter to the 9 reasoning categories, `uv run python -m train_sft`, convert via
`adapter converter.ipynb`, then re-measure cryptarithm_deduce on
`val/kaggle_newrule_eval_standalone.py` (N=50, difficulty 4) vs the 6.0% frozen baseline.
