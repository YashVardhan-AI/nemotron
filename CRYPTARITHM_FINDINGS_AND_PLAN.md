# Cryptarithm: Findings & Complete Plan to 0.86 → 0.89

_Session 2026-06-08/09. Goal: maximize train-set accuracy of a **rank-32 LoRA** on
**Nemotron-3 30B-A3B**, decode **temperature=0, max_tokens=7680, single greedy sample**.
Baseline 0.86; current LB top 0.89._

> **⚠️ §0–§6 below are PARTLY SUPERSEDED. The Phase-0/1 renderer was built and arm A
> was trained — it FAILED (0.84, −2pp). See the 2026-06-10 update immediately below;
> the revised plan replaces §6.**

---

## 🔄 2026-06-10 UPDATE — Arm A trained & FAILED; revised research-backed plan

**What we did since §6:** built the symbolic renderer (`reasoners/crypt_symbolic_trace.py`,
both `assert` and `derive` styles), wired `corpus.py` (`CRYPT_SYMBOLIC=1`,
`CRYPT_SYMBOLIC_STYLE`), validated 725/725 traces round-trip + budget, built **arm A
(assert)**, set `train_sft.py` to 1 epoch, trained.

**Result: 0.86 → 0.84 (NET −2pp).** Eval breakdown: cryptarithm **arithmetic still ~0%**
(add/mul/absdiff/rev_concat all 0%), only **concat 40%** — *regressed* from the old
concat-fallback's ~100% on concat. So the `assert` trace did **not** teach the search
**and** erased the working concat strategy — exactly the `corpus.py:88` warning ("the
assertion trace cannot teach an unlearnable deduction and erases the working strategy").

**Reframe (cryptarithm is still THE lever).** A competitor reportedly reached **~71% on
cryptarithm** (with a neighbor regression) → it IS learnable by this exact model class.
**The math closes for 0.89:** crypt_deduce (8.7% wt) 15%→71% = **+4.8pp**, crypt_guess
(2% wt) → ~+1pp ⇒ **~+5.8pp gross**; need +3pp net ⇒ **can absorb up to ~2.8pp neighbor
regression and still hit 0.89.** Goal = crypt ≈ 71% with regression < ~2.8pp (not zero).

**Deep research (deep-research workflow `wr2l2zh4s`; 23 primary sources; claims verified
3-0 / 2-1; 10 refuted & excluded):**
- **FORMAT is the dominant lever** — search-traces beat optimal/answer traces by **+25pts**
  (Stream-of-Search, COLM'24, 2404.03683). ⇒ **DERIVE > ASSERT** (confirms arm A's failure).
- **GUARDRAIL (3-0, 2406.06467):** a verbose "fully-educated" scratchpad **overfits its
  length / step-count and FAILS OOD**; generalization needs a **step-count-INDEPENDENT
  "inductive scratchpad"** (bounded, repeating state-update). ⇒ our current `derive` trace
  is variable-length → would risk the **same null**; it MUST be recast to a bounded form.
- **LSB-first / reversed-digit arithmetic** is a big, replicated win (mult 88.5% vs 52.8%;
  add ~100% vs ~85%; 2403.05845, 2307.03381) — makes arithmetic a **local** next-token
  function. ⇒ compute operator results least-significant-digit-first in the scratchpad.
- **Single-adapter integration:** data-mixing / replay is the safe default (LoRA-*merging*
  doesn't scale past ~3 skills, 2410.13025); **O-LoRA / LoRI-D** orthogonal subspace is the
  next-best structured option (2310.14152, 2504.07448); **avoid** aggressive DARE/TIES.
- **STaR / expert-iteration** post-SFT adds **+1–5pp**, feasible with generate→filter→SFT
  (no RL loop; 2404.03683, medium confidence).
- **2nd likely cause of arm A's null, independent of format:** **725 unique problems may be
  BELOW the data phase-transition** for a skill this composite.
- **REFUTED (do NOT rely):** trace-pruning / randomized-trace curricula (0-3), aggressive
  DARE 90-99% drop (0-3), CAT > merging (0-3), length-bound-lookup pessimism (0-3).
- **Caveat:** format findings are from small / from-scratch models on simpler tasks —
  magnitudes won't transfer 1:1; the **71% existence proof is the real anchor**.

### REVISED PLAN (replaces §6 Phases 0–4)

- **P1 — Recast the cryptarithm trace [cheap, do first, highest EV].** Add a
  `derive_inductive` style to `crypt_symbolic_trace.py`: a **bounded, step-count-independent
  inductive** format with **LSB-first arithmetic**, borrowing the explicit-decomposition
  shape of the working categories (equation/bit). Fixed 4-section skeleton regardless of
  base/glyph count:
  1. **Detect** — count distinct glyphs → base; scan a result for a sign prefix → mode
     (standard vs LSB-first/little-endian); per operator, name the op from one example.
  2. **One worked example, computed LSB-first** (`6+7→3 carry 1`, next digit…) to teach the
     local arithmetic.
  3. **Map as a fixed-width constraints→assignment table** (one row per glyph) — NOT a
     per-pass narration whose length tracks search depth.
  4. **Apply** to the query LSB-first → tight `\boxed{}`.
  Validate 725/725 round-trip + token budget like before.
  **✅ BUILT + VALIDATED 2026-06-10 (renderer only — train+eval still pending/user-run).**
  Added `derive_inductive` style to `crypt_symbolic_trace.py`. Step 2 is now exactly
  **one fixed-shape row per glyph** in the solver's MRV commit order (row count = glyph
  count ≤10, never search depth → the 2406.06467 OOD guardrail), and — to keep it a real
  *search* not a bare assertion — each forced row **cites the binding equation** that drops
  the glyph's running domain to a singleton (read from the solver's own `intersected_after`,
  so correct by construction). Reversed modes (42%) get **LSB-first result-digit annotations**.
  Validation: **725/725 round-trip, 0 drops**, budget p50=869/**max=1052** (vs old `derive`
  max 1519, vs 7680 cap); 399 `derive_inductive` + 326 `lean` (little-endian / base≠10 keep
  the lean map but still get the LSB note). `assert`/`derive`/`lean` outputs are **byte-identical**
  to before (the `annotate` flag defaults off → clean A/B). Corpus-wired via
  `CRYPT_SYMBOLIC=1 CRYPT_SYMBOLIC_STYLE=derive_inductive`. **Next:** build that corpus arm
  (destructive rmtree+rebuild ~7800 files — user-gated) → train 1 epoch → Kaggle eval at
  temp=0/7680 with cipher + equation canaries.
- **P2 — Scale UNIQUE data [medium, likely needed].** Build a rich generator over the
  solver's op/base/mode distribution → **3–5k fresh, holdout-safe puzzles** → derive traces
  (cross the phase transition; escape the 725-real limit). `BOOST_HARD` upweighting is the
  cheap stopgap, but duplication ≠ new samples.
- **P3 — Regression control [parallel, the crux].** Tune the cryptarithm:neighbor mixing
  ratio with **cipher + equation as canaries**; if data-mixing is insufficient, O-LoRA /
  LoRI-D orthogonal subspace (bigger lift). Budget: ~2.8pp regression is affordable.
- **P4 — Post-SFT STaR / expert-iteration [optional, +1–5pp].** generate → exact-match
  filter → SFT.
- **Secondary (other categories):** bit_manipulation **het** sub-tier (≈50% vs 95% hom) —
  teach an explicit per-position rule table.
- **P5 — Teacher traces (BACKUP / fallback — only if P1–P3 measurably fail to teach the
  procedure).** Two variants, best first:
  1. **Solver-grounded teacher narration:** give a frontier teacher the puzzle **plus the
     solver's CORRECT `(map, ops, mode, base, answer)`** and ask it to write a natural,
     human-style **bounded inductive** deduction. This separates "the teacher can't *solve*
     cryptarithm (~15%)" from "the teacher can *narrate* a known solution fluently" — the bet
     is that natural frontier prose is more learnable than our mechanical template. The solver
     stays the correctness oracle: verify every rendered trace's `\boxed` == gold (drop
     misses) and enforce ≤7680 tokens.
  2. **Pure rejection sampling (weaker):** teacher solves from scratch, exact-match filter vs
     gold, keep only the correct traces. Low yield (~15% teacher pass rate) but correct +
     stylistically diverse; a supplement, not a base.
  Caveat (memory `teacher-distillation-cryptarithm-measured`): *raw* teacher distillation was
  net-negative on the solvable categories and blew the 7680 budget on hard bit — so keep this
  **cryptarithm-only, solver-grounded, and budget-checked**, gated behind the cheaper levers.

**DEAD-ENDS (don't spend on):** assert / answer-only traces, big-endian arithmetic,
trace-pruning curricula, aggressive DARE/TIES merging for a heavy specialist, and *raw*
teacher distillation on solvable categories — **distinct from P5's solver-grounded
cryptarithm narration** (the teacher can't *solve* cryptarithm, so only use it to *narrate*
the solver's known answer, never as the correctness source).

**Immediate next step:** P1 is BUILT. Next = the combined P1+P2+P3 arm (below), then
ONE train+eval. Memory: `cryptarithm-armA-null-and-research-plan`.

---

## P2 / P3 — DETAILED DESIGN (2026-06-10)

Grounded in the actual infra (`corpus.py`, `cryptarithm_deduce_core.py`,
`solver_results.parquet`, the live prompt format). **Decisive fork found:** the *existing*
forward-gen generator (`build_cryptarithm_examples` → `make_trace_problem` →
`cryptarithm_rule`/`cryptarithm_deduce_core`) is the **OLD NARROW MODEL** — only 5 ops
(add/abs_diff/mul/concat/rev_concat), **base 10**, **standard order** (`_OPS`,
`d0*10+d1`). That is exactly the **12% slice** the symbolic-solver breakthrough escaped, so
**it cannot be reused for P2** — it would re-inject the off-distribution data that made
cryptarithm unlearnable. P2 must be a **new rich generator** on the solver's distribution.

### Prompt format the generator must emit (confirmed from live data)
```
In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
<L0L1 OP L3L4> = <RESULT>     ← 4 example lines, glyphs UN-spaced (raw) in the prompt
... (×4)
Now, determine the result for: <Q0Q1 OP Q3Q4>
```
(Glyph space-separation is ONLY in the CoT/reasoning to defeat BPE merge; the prompt itself
is raw glyphs. The renderer + completion assembly already handle this.)

### P2 — Rich symbolic cryptarithm generator [medium effort; the scale lever]
**Objective:** 3k (→5k) FRESH, holdout-safe, **distribution-matched** puzzles → `derive_inductive`
traces, to cross the data phase transition (725 real uniques may be sub-threshold).

**New module `reasoners/crypt_symbolic_gen.py`** (+ `build_crypt_symbolic_examples()` in
corpus.py, env `CRYPT_SYM_N`). Pipeline per puzzle:
1. **Sample a program from the EMPIRICAL marginals** of the 725 solved real puzzles
   (`solver_results.parquet`): base ∈ {10:691, 9:27, 8:6, 7:1}; mode ∈ {standard:423,
   little_endian:302}; per-operator op ∼ the real `solver_ops` frequency over the 47-op
   library; #examples ≈ 4; 1–3 distinct operator glyphs. **Include concat/mixed-concat at
   real frequency** (directly fixes arm A's concat regression — see P3).
2. **Pick a glyph alphabet** from the real printable-punctuation inventory; random bijection
   to digits 0..B−1.
3. **Construct examples + query**: sample operand glyph-pairs + operator glyph; compute each
   RHS by **reusing the renderer's own verified semantics** (`crypt_symbolic_trace._apply_eq`,
   `_encode_mag`, `_op_value`, `_two_digit`) — so generation and rendering share one source of
   truth and can never disagree.
4. **Uniqueness/learnability filter (KEY):** run the **rich solver `AliceEquationSolver`
   WITHOUT the gold hint**; keep the puzzle ONLY if it recovers the planted answer as the
   unique pure-inference solution. This (a) guarantees the trace teaches a genuinely-inferable
   instance, and (b) **bounds the synthetic set to the solver's ~72% learnable region** —
   generating outside it would teach unlearnable global-constraint puzzles.
5. **Holdout/dedup:** prompt-hash dedup vs (real train prompts ∪ val prompts ∪ already-emitted);
   random programs make collision negligible, but gate anyway (keeps val honest).
6. **Render `derive_inductive` + round-trip gate:** `\boxed` must reproduce gold (drop misses),
   ≤7680 tokens. Emit `(pid, prompt, completion, answer)` like `build_cryptarithm_examples`.

**Validation gate (BEFORE any train):** generate 3k and report — yield (attempts/kept &
unique-solvable %), **distribution match vs real** (op/base/mode KL or side-by-side), round-trip
100%, budget p50/p90/max, dedup collisions (=0 expected), glyph un-merge, 10 eyeballed samples.
Reuse the `val/gen_crypt_symbolic.py` harness shape.

**Risks:** (a) unique-solvable **yield** may be low (random rich programs are often
underdetermined) → measure attempts/kept; raise `max_salt`-style resampling. (b) The filter
biases toward EASIER puzzles — acceptable (we want the learnable region), but report the
difficulty/op-count distribution so we know what we're teaching. (c) Distribution drift from
the real test → mitigated by empirical-marginal sampling + the match report.

**✅ BUILT + VALIDATED + COMMITTED 2026-06-10.** `reasoners/crypt_symbolic_gen.py` +
`val/gen_crypt_synth.py`. **Required building the Rust solver accelerator** (`alice_sovler_helper`
via the GNU toolchain — Python search is **5800 ms/solve**, infeasible; Rust is **205 ms p50**,
28×; see memory `rust-solver-accelerator-build`). The learnability filter calls the solver's
normal-level path directly (skips the 38-op deep-search escalation on rejects). **N=300 gate:
67% yield, 300/300 round-trip, 0 dedup collisions vs real, 0 over-budget** (p50=2186/max=4051),
gradeable 263/300 (= real brace cap). **Distribution matches real:** base {10:.95, 9:.04} == real,
mode 61/39 vs 58/42, op-types close (mul slightly over 26 vs 21%, **concat under 2.1 vs 5.8%** —
minor; nudge concat sampling up for P3's concat-protection if wanted). Renders `derive_search` by
default. Speed ~1.1 s/kept incl. a long tail (some 15 s solves) → ~3k ≈ ~1 hr background run.
**Risks (a)/(c) retired** (yield fine, no drift, no synth-only op-types).

### P3 — Regression control [parallel; the crux of a NET gain]
**The tension (made concrete):** the `DUP_TARGETS` cap *is* the mixing ratio. Today
cryptarithm = 627+154 ≈ **10%** of 7830 rows. P1 (replace 725 real, same prompts) is
ratio-neutral; **P2 ADDS uniques**, and to actually USE them the cryptarithm target must RISE
— which inflates cryptarithm's gradient share and risks the arm-A neighbor crash (−2pp). Budget:
**≤2.8pp** neighbor regression is affordable for a +5.8pp gross.

**Lever 1 — mixing ratio (primary).** Set `DUP_TARGETS["cryptarithm_deduce"]` to a MODERATE
value that consumes the synthetic uniques without swamping — start **~1500** (≈18% share; was
627). Because P2 adds *unique* samples (not dups), even a moderate target gives diverse
cryptarithm gradient — strictly better than dup'ing 725. Hold **cipher (1656)** and
**equation (658+126)** at winning-recipe levels as **canaries**.
**Lever 2 — concat-protection (arm A's specific failure).** Arm A regressed concat 100%→40%
because the symbolic override replaced the concat-fallback rows that *were* getting concat
right. Fix at the source: the P2 generator emits **pure-concat + mixed-concat puzzles at their
real frequency**, and the renderer already handles concat (lean verify). Track **concat
sub-accuracy** explicitly in eval.
**Lever 3 — O-LoRA / LoRI-D (fallback only).** If Levers 1–2 can't hold the canaries:
orthogonal-subspace LoRA (penalize overlap between the cryptarithm update subspace and the
frozen neighbor subspace) — a `train_sft.py` change, higher effort, deferred until data-mixing
is proven insufficient (data-mixing is the research-backed default; merging/DARE are DEAD).

**Experimental design — ONE decisive combined arm, then ablate only if needed.** Per the
"don't burn a cycle on format alone" rationale:
- **Arm B1 (primary):** **`derive_search`** on **725 real (P1) + ~2000 synthetic (P2)**,
  `cryptarithm_deduce`≈1500, concat at real freq (P3), cipher+equation held. **1 epoch →
  Kaggle eval temp=0/7680.** The eval **sub-category breakdown** (crypt arithmetic vs concat;
  cipher; equation) is the built-in diagnostic — one run tells us format+data+regression at once.
  (A cheap second arm swaps `derive_search`→`derive_inductive` to A/B long-search vs
  short-bounded if both are affordable; otherwise `derive_search` is the primary bet — see the
  trace-length finding below.)
- **Revert/contain criteria:** cipher or equation drop **>2pp** → dial the cryptarithm target
  down (1500→1000) or invoke Lever 3. Crypt arithmetic still ~0% → format/identifiability
  ceiling (escalate to P4 STaR or P5 teacher narration). Crypt up **and** canaries hold → win;
  push synthetic toward 5k.
- **Ablations (only if B1 is ambiguous):** B1 minus P2 (format-only) and B1 minus P3
  (ratio sensitivity) — to attribute a null. Accepted tradeoff: B1 confounds the three levers
  for the chance of a clean win; the sub-category breakdown de-confounds most outcomes anyway.

### TRACE-LENGTH FINDING + `derive_search` (2026-06-10) — likely the bigger lever

Measured real reasoning-trace token lengths per category against student accuracy:

| Category | p50 tokens | Student acc |
|---|---|---|
| bit_manipulation | 6736 | ~88% |
| equation_numeric_deduce | 5783 | ~87.5% |
| cipher | 2895 | ~100% |
| **cryptarithm_deduce (real)** | **611** | **~8.5%** |
| `derive_inductive` (P1) | 869 | — |
| **`derive_search` (new)** | **2184** | — |

**Every category the student wins on uses LONG traces (2.9k–6.7k tok); cryptarithm — the one
it fails — uses ~600.** The short-trace bias (mine *and* the repo's) was probably backwards:
`equation_numeric` is the **closest analog** (secret-operator induction) and wins with a
**long enumerate-and-test** that shows every WRONG operator candidate before the match. And the
"long cryptarithm traces crash neighbors" fear is **misattributed** — the *short* bit forward-gen
(BIT_N=800) crashed neighbors too (88→22), so the crashes were forward-gen **volume**, not
length (that's P3). I under-weighted Stream-of-Search (+25pt) vs the from-scratch bounded
guardrail; the repo's own winning categories are the stronger prior → **go long**.

**`derive_search` — BUILT + VALIDATED + COMMITTED 2026-06-10.** Mirrors
`reasoning_equation_numeric`: **Step 1** map (bounded `derive_inductive` rows) FIRST, **Step 2**
an enumerate-and-test that tries candidate operations on the now-known digits and shows
wrong/match per candidate (common ops always, rare ops up to the true op), then verify + apply.
Every candidate test reuses the verified `_op_value`/`_encode_mag` semantics (hand-checked in
standard + little_endian). **Universal** — the op-search works in every base/mode, so all 725
render at this tier (no lean fallback, unlike `derive_inductive`'s 326). Validated 725/725
round-trip, 0 drops, **p50=2184 / max=3396 tok** (in the proven range, far under 7680);
`assert/derive/derive_inductive/lean` outputs unchanged. **This is now the primary long arm.**

**Build order:** P2 generator + its validation gate (no train) → fold P3 ratios into the
corpus build → build Arm B1 corpus (destructive rebuild, user-gated) → user trains → eval.

---

## TL;DR

1. **Cryptarithm is the only lever big enough for +0.03.** It is ~9% of train weight but
   **~67% of all errors** (student ~8.5% on cryptarithm_deduce, ~21% on guess; every other
   category is ≥87% or saturated at 100%).
2. **Frontier-teacher distillation is dead as a *general* lever** — measured: on the
   *solvable* categories the rank-32 student already **matches/beats** a frontier teacher's
   pure chain-of-thought (bit 88% vs 55%, equation 87.5% vs 75%), and the hard bit tier is
   **locked by the 7680-token budget** (correct teacher traces run 9k–28k tokens).
3. **Cryptarithm was never solvable by the repo's pipeline.** The repo solver reproduces only
   **12%** of real cryptarithm gold answers; `problems.jsonl` marks **83% `rule_unknown`** —
   the maintainer also gave up. The forward-gen pipeline therefore trained on a **wrong, narrow
   12% slice** of the problem.
4. **BREAKTHROUGH (verified):** `kaggle-nemotron-equation-symbolic` is a symbolic solver that
   **cracks cryptarithm** — **97.2% overall (100% gold-conditioned, 72% pure inference)** — via a
   rich op library, variable number base, reversed-digit modes, and signed results. This *is* the
   reverse-engineered generator we were missing.
5. **The path:** use that solver (gold-conditioned, on train.csv) to generate **correct,
   full-distribution, forced-deduction CoT traces**, render them glyph-separated, blend into the
   corpus at a modest format-isolated ratio, and retrain. Cryptarithm has been stuck at 8.5% only
   because its training data was wrong; with correct learnable traces the realistic ceiling is the
   solver's own deductive procedure.

---

## 1. The error budget (why cryptarithm is the lever)

From the 950-sample local eval (per-category accuracy × train weight):

| Category | ~weight | acc | error mass | remediable |
|---|---|---|---|---|
| **cryptarithm_deduce** | 7.5% | 8.5% | **0.069** | 0.054 (0.015 brace-capped) |
| **cryptarithm_guess** | 1.5% | 21% | 0.012 | 0.009 |
| bit_manipulation | 17.8% | 88% | 0.021 | ~0.004 trunc + ~0.013 het (identifiability-capped) |
| equation_numeric_guess | 0.7% | **0%** | 0.007 | ~0.007 (small) |
| equation_numeric_deduce | 5.1% | 87.5% | 0.006 | modest |
| cipher | 17.1% | 97.5% | 0.004 | small |
| gravity / numeral / unit | ~50% | 100% | 0 | saturated |

**Cryptarithm ≈ 67% of all errors.** Lifting it from ~10% to ~45–50% clears the entire 0.86→0.89
gap by itself. Nothing else has both the mass and the headroom.

Hard caps to respect: **brace bug** — cryptarithm answers containing `}` are ungradeable (the
`\boxed{...}` extractor truncates at the first `}`); measured **19.5%** of deduce / **22.1%** of
guess answers → cryptarithm tops out ~80% gradeable.

## 2. What was measured this session

- **Distillation map (frontier teacher = Claude, pure CoT, real train problems):**
  crypt_deduce 15%, crypt_guess 0%, bit 55% (**only 3/11 correct traces fit 7680 tokens**),
  eq_deduce 75%, eq_guess 20%. Student ≥ teacher on every solvable category → teacher distillation
  would *lower* accuracy there; hard bit is budget-locked. ⇒ distillation is not a general lever.
- **Tokenizer merge (confirmed real):** Nemotron BPE merges glyphs across the positional
  boundaries that define the task — `()+^^` → `['()','+^','^']` (operator fused to a digit).
  Space-separation fixes it (`( ) + ^ ^` = 5 clean tokens). Teacher pre-check on separated glyphs:
  15%→20% — separation helps **parsing**, not the **search**.
- **Reverse-engineering gap:** repo solver coverage **96/781 = 12.3%**; `problems.jsonl` status
  **rule_unknown 687 / hypothesis 71 / rule_found 65**. A 12-op extended search (injective +
  non-injective, signed/mod/floordiv/sum-sq/mod-100, reversed output) fit **0 of 40** sampled
  rule_unknown problems → our 5-op base-10 model is structurally wrong for ~83%.
- **Repo capability facts:** `train_sft.py`'s `ppo/cispo/dro` are **offline reweighted-SFT** on
  static data, **not** RL (no rollout loop). A3B is MoE. 2 epochs / 490 steps / LR 2e-4→0.
  Raising LoRA rank is *not* evidence-backed (the "rank-256" claim was refuted).

## 3. THE BREAKTHROUGH — the symbolic solver (verified)

`kaggle-nemotron-equation-symbolic-main/.../src/solver_eq_symbolic.py` (`AliceEquationSolver`) +
a Rust search helper. Targets the same "Alice's Wonderland transformation rules on equations"
puzzles (its `equation_symbolic` == our cryptarithm).

**Verified accuracy** (its `data/solver_results.parquet`, 823 rows; plus independent
re-application of each reported program — 706/800 reproduce cleanly, the rest are little_endian
rows my quick re-impl mishandled, not solver errors):

| Mode | Accuracy |
|---|---|
| **Gold-conditioned** (uses train gold; for DATA-GEN) | **741/741 = 100%** |
| **Pure inference** (no gold) | **59/82 = 72%** |
| Overall | **800/823 = 97.2%** |

**Why it cracks the 88% our model could not** — the true generator uses:
- **~47 operations:** core (add, sub, rsub, absdiff, neg_absdiff, mul, gcd, lcm, fdiv, rdiv, mod,
  rmod, min, max), **offset ops** (add/mul/absdiff/sub ±1/±2 — e.g. `mul_p1` = a·b+1), scaled/
  polynomial (mul_half/double, sq_diff/sum, mul_plus/minus_a/b, a²+b, a+b²), bitwise (xor/band/bor),
  **signed** (sub_signed, rsub_signed), and concat_fwd/rev.
- **Variable base:** base = number of distinct content symbols (also tries 10). 33 correct solves
  were base 7–9.
- **Three modes:** `standard`, `alice` (reverse operand **and** result digits), `little_endian`.
  **302 of the correct solves were reversed-digit modes** — a huge slice we never modeled.
- **Signed/operator-prefixed results:** if the rhs starts with the operator glyph, the result is
  negative; magnitude is encoded after the prefix.
- Operator = the middle glyph of the 5-char input (ANY glyph, not just `+ - *`).

Concrete examples we had marked unsolvable:
`00457d26`: `*` = **`mul_p1`** (a·b+1), `-` = absdiff, base 10. `00c032a8`: mode **`little_endian`**,
ops `{'!':mul, ']':add, '<':absdiff}`.

**Crucially**, gold-conditioned the solver returns a correct `(mapping, ops, mode, base)` **and** a
structured **forced-deduction trace** (`_derive_order`: per-symbol commit order via MRV with a
max-propagation tie-break, domain narrowing, forced-vs-guess, per-example feasibility) for ~100% of
train problems. That is exactly the **correct, full-distribution, learnable inductive-scratchpad
CoT** the literature says is required and that the old concat-only traces (12% coverage) never gave.

## 4. Why every prior cryptarithm attempt failed (now explained)

- **Real training traces are concat-only** (`reasoners/cryptarithm.py`): they narrate concat
  reasoning but pair it with arithmetic answers for the majority → inconsistent signal; and they
  only ever describe ~12% of the real rule space.
- **The `induct` fix crashed** (0.86→0.83; bit 88.8→22, cipher/equation collapsed) for two
  *separately fixable* reasons: an unlearnable dense `STATE|map:{...}` search-log shape, and a
  too-high mixing ratio (627 replacing real data). It also still used the wrong 5-op/base-10 model.
- **The 5-op base-10 solver** could never express offset ops, reversed-digit modes, variable base,
  or signed results — so `is_uniquely_solvable` filtered forward-gen down to the same 12% slice.

## 5. Research-backed design principles (adversarially verified)

- **Canonical algorithmic / scratchpad traces beat ad-hoc CoT** for systematic tasks, and **SFT
  amplifies the scratchpad benefit** (Nye 2112.00114; Zhou 2211.09066). ⇒ render the solver's
  forced-deduction procedure, not free-form reasoning.
- **Pure-CoT search with backtracking is near-zero** even for frontier models (SearchBench
  2406.12172). ⇒ teach a *deterministic forward replay* of the deduction (the solver gives the
  forcing order), not an open search.
- **STaR / rejection-sampling** is the right data regime when the answer is known; **strip the
  answer-hint** from the training prompt to avoid "Hint Short-cutting" memorization (2203.14465).
- **Zero-pass escape is data-centric curriculum mixing**, not reward shaping (2510.03971); plain
  binary-reward RL is inert at pass@k≈0 (2510.07242). ⇒ if needed, mix *easier* cryptarithm
  variants.
- **Dense algorithmic traces cause negative transfer** to neighbors (2211.09066). ⇒ **isolate the
  format** (flag/delimiter) and keep a **modest mixing ratio**; gate on the neighbor eval.

---

## 6. THE COMPLETE PLAN

### Phase 0 — Trace renderer + feasibility (cheap, ~1–2 days)
**Goal:** turn the solver's `deduction_trace` into training-ready CoT.
1. Write a Python renderer: `AliceEquationSolver(prompt, answer_hint=gold).solve()` →
   `details['deduction_order' / 'deduction_trace']` → natural-language forced-deduction CoT:
   - **Glyph-separated** input ("separate the glyphs: `( ) + ^ ^` → position 0 = `(` …") to defeat
     the BPE merge (Pillar 0).
   - State base + per-operator op identification, then walk the MRV commit order showing each digit
     **forced** by a named example/column, then apply to the query.
   - End with a **tight** `\boxed{gold}` (no spaces — the grader's gold is unspaced; the model can
     still emit the right string token-by-token).
2. **Feasibility gates** (decisive, cheap):
   - **Length:** measure rendered-trace token length (Nemotron tokenizer). Must fit ≤7680 with
     margin; if long-base/many-symbol traces overflow, compress (drop verbose domain dumps, keep
     forced steps).
   - **Coverage:** confirm ~100% of train cryptarithm get a correct rendered trace (solver
     gold-conditioned).
   - **Brace-safety:** flag the 19.5%/22.1% brace-gold problems (unwinnable; keep but don't expect
     credit).

### Phase 1 — Corpus integration + the decisive training A/B
**Goal:** replace the broken concat-only traces with correct ones, retrain, measure.
1. Generate correct forced-deduction traces for all train cryptarithm (deduce + guess), via the
   solver, glyph-separated.
2. In `corpus.py`: replace the 627 deduce + 154 guess concat traces with the new traces (or wire a
   new renderer alongside `_crypt_renderer`). Keep a **modest, format-isolated ratio**; add a flag
   token/delimiter so the dense format doesn't bleed into neighbors.
3. Retrain on Tinker (existing `train_sft.py` recipe). Eval on the **novel-rule holdout**:
   cryptarithm accuracy **and** cipher/equation (hard gate: neighbors must not regress).
4. **Decision:** measure the cryptarithm lift. This is the test of whether a rank-32 LoRA can learn
   the procedure — no longer a reverse-engineering question.

### Phase 2 — Curriculum + iterate (if Phase 1 is partial)
1. If cryptarithm lifts but plateaus, add **easier variants** (fewer symbols → smaller base,
   standard ops only, more worked examples), mixed across difficulties (the zero-pass curriculum
   escape).
2. If neighbors regress, tune the mixing ratio / strengthen format isolation / per-category
   gradient-token reweighting.

### Phase 3 — Polish & combine (the last mile to 0.89)
1. **`equation_numeric_guess` (0%):** the same solver handles `equation_numeric` — apply it to
   generate correct traces for this tiny-but-zero bucket (~+0.005).
2. **bit truncation (2.3% >7680):** compress the longest bit traces to fit (~+0.004).
3. Re-measure full train accuracy; confirm 0.86 → target.

### Phase 4 — RL (optional, only if pass-rate is non-zero post-distillation)
1. Once cryptarithm pass-rate is off zero, GRPO/GSPO with the exact-match reward can sharpen pass@1.
   Requires **building a Tinker rollout loop** (the repo has none). Higher effort, smaller marginal
   gain; do last.

### Expected trajectory
| After | cryptarithm | overall | confidence |
|---|---|---|---|
| Phase 1 (procedure partially learned) | ~25–40% | ~0.875–0.885 | medium |
| Phase 1+2+3 (well learned) | ~45–72% (≤~80% brace cap) | **~0.89–0.92** | medium |
| +Phase 4 | higher | — | lower / high-effort |

To clear **0.89** you need cryptarithm ≈ **45–50%** (plus the small eq_guess/bit fixes). The solver's
own **72% pure-inference** rate is the rough ceiling *if the student learns the procedure*.

## 7. Risks & open questions
- **Learnability (the key risk):** can a rank-32 LoRA learn the multi-step procedure (base
  detection + op identification from ~47 ops + mode detection + MRV mapping deduction) at temp=0
  within 7680 tokens? Necessary precondition (correct learnable traces) is now met; sufficiency is
  the Phase-1 A/B.
- **Trace length** for large-base / many-symbol problems may exceed budget → compression needed.
- **Negative transfer** to cipher/equation (the induct failure mode) → modest ratio + format
  isolation + neighbor eval gate.
- **Brace cap** hard-limits cryptarithm to ~80% gradeable.
- **Generalization to novel test rules:** the deductive procedure is rule-agnostic (the solver
  generalizes across rules), so it should transfer; validate on the novel-rule holdout.

## 8. Key artifacts & tools
- **Solver:** `kaggle-nemotron-equation-symbolic-main/.../src/solver_eq_symbolic.py`
  (`AliceEquationSolver`, `_derive_order` trace), `solve_all_symbolic.py`,
  `data/solver_results.parquet` (823 solved rows). Rust helper optional (Python fallback works).
- **This session's investigation toolkit** (repo root, `_`-prefixed): `_solver_coverage.py`,
  `_failcat.py`, `_fit_extended.py`, `_extract_unknown.py`, `_verify_solver.py`, `_tok_test.py`,
  `_gen_sep_workflow.py`. `val/teacher_probe.py` = frontier-teacher feasibility probe. `runs/teacher/*`.
- **Eval:** novel-rule holdout via `val/kaggle_realholdout_standalone.py` /
  `val/kaggle_newrule_eval_standalone.py`. Grader mirrored in `val/grading.py`.

## 9. References (verified)
- Nye 2021 (Scratchpads, 2112.00114); Zhou 2022 (Algorithmic Prompting, 2211.09066);
  Zelikman 2022 (STaR, 2203.14465); SearchBench 2024 (2406.12172);
  Prakash & Buvanesh 2025 (zero-pass curriculum, 2510.03971); HERO 2025 (2510.07242).
