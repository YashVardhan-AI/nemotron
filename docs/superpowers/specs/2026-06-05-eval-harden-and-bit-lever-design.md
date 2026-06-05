# Eval hardening + bit-complex lever — design (2026-06-05)

## Why

Two facts forced a reset of priorities:

1. **P1 (cryptarithm dense-trace SFT) is dead.** The `induct` style was measured net-negative
   (cryptarithm ~0% on the arithmetic family AND it cratered structurally-similar neighbors:
   cipher 97→52, equation 87→27, overall 0.86→0.83). Root cause = the *globality barrier*: the 30B
   cannot maintain the injective (all-different) digit→symbol map at greedy decode. No dense-trace
   style or scale fixes this. (memory: `induct-sft-negative-result`)
2. **The eval can't be trusted for the next lever.** The synthetic new-rule generators were flagged
   as too easy / regression-insensitive — a known-bad adapter still scored high, masking the cipher
   crash. (memory: `synthetic-eval-too-easy`)

Four research agents audited the code + real data. Findings (each ties to a concrete fix below):

- **Bit generator IS structurally too easy.** It gets the macro family mix right (65% pairwise /
  20% rotation / 15% complex, matching real) but the micro-structure is wrong: every synthetic
  problem applies *one global op + one global operand offset `(a+j,b+j)` across all 8 output bits*.
  Real data is heterogeneous: 42.6% of problems mix ≥2 ops across the 8 bits, 48.2% have per-column
  operand offsets, plus arbitrary routing (`I*`, the single most common op) and arbitrary 3-var
  truth tables. So the eval measures "kept the global-shift heuristic," not "can induce an arbitrary
  per-bit function" — a regressed adapter that retains the heuristic scores high.
- **Cipher generator is NOT too easy.** Byte-faithful clone of real (needs-vocab 61.6% vs 58.9%,
  same 77-word vocab, same example counts). The apparent insensitivity was a *measurement artifact*:
  97.5%→52.5% was *memorized-training* accuracy (n=162), compared against *fresh synthetic new-rule*
  (n=50) — different sets, different skill, underpowered. Fix = harness power, not the generator.
- **Bit-complex-3-input is fundamentally safer than cryptarithm.** It is decomposable-local: each
  output bit is an independent ≤3-input table lookup with NO global coupling. Shared stride means
  induction is over ONE `(a,b,c)+op` triple, not 8 functions. The solver has zero MAJ/CHOICE support
  → this tier (~14% of bit, ~2.5% of overall weight) is entirely untaught. Plausible upside
  +1.5–2.5 overall if it lands without crashing neighbors.
- **Cryptarithm boxed-brace bug = measured 11.9% hard cap.** Answers containing `}` are mis-extracted
  even when correct, because the extractor truncates at the first `}`. The competition grader is fixed
  (we can't change it), so this is a real leaderboard cap on cryptarithm we can only dodge from the
  *generation* side (brace-safe emission / exclude `{}` from generated symbols) — a deferred-cryptarithm
  concern, not a Phase 0 one.

## Scope (this spec = Phase 0 + the plan for Phase 1/2)

### Phase 0 — Make the eval trustworthy (this session, mostly local Python)

**0.1 Bit generator recalibration** (`val/generators/bit_manipulation.py` + the byte-identical inline
copy in `val/kaggle_newrule_eval_standalone.py`). Replace the three monolithic per-word `apply`
builders with a **per-column rule vector**: each of the 8 output bits gets its own op + arbitrarily
chosen operands, drawn so the realized distribution matches measured real micro-structure:
  - Keep example count 7–10 (sample uniformly; currently fixed 8) and the macro family weights
    65/20/15 — both empirically correct.
  - *pairwise-dominant* problems: each output bit independently gets op ∈ {AND,OR,XOR,±NOT} on an
    arbitrary operand pair `(p,q) ∈ {0..7}` (not `(a+j,b+j)`); target ≥40% of pairwise problems use
    ≥2 distinct ops across columns. Mix in const / identity-routing (`I*`) / NOT columns at real rates.
  - *routing* family: mostly constant-k rotation (~90%) but ~10% arbitrary permutation of `0..7`.
  - *complex* family: per affected column pick {MAJ, CHOICE, random 3-var truth table `TT3`} with
    arbitrary operand triples; calibrate ~3% MAJ / ~7% CHOICE problem rates.
  - **Inject genuine ambiguity**: arbitrary operands + low-end example counts so some columns have
    multiple consistent ≤2-input fits in the examples. The answer key stays exact (apply the
    ground-truth per-column rule); only the *examples* are under-determining. This is what forces
    real per-column reasoning and is where a worse adapter drops.

**0.2 Regression-sensitivity acceptance test (local, no GPU).** A pytest that scores two synthetic
predictors against the recalibrated generator: a *perfect oracle* and a *global-shift-heuristic-only*
predictor (induces a single global op+offset from the first column and extrapolates — i.e. exactly the
behavior the OLD generator rewards). Assert the heuristic predictor scores near-perfect on the OLD
generator but **measurably lower** on the NEW one, while the oracle scores ~100% on both. This *proves*
regression-sensitivity without a GPU adapter. (The GPU baseline-vs-induct re-score is the user's
follow-up confirmation on Kaggle.)

**0.3 Harness power** (`val/kaggle_newrule_eval_standalone.py`, `val/kaggle_realholdout_standalone.py`,
`val/report.py`): raise `PER_CATEGORY` for bit/cipher to ≥300 (kills the ±7pt n=50 noise that hid the
cipher regression); add a **fresh-real cipher control** (the real holdout currently excludes cipher by
design) so cipher generalization is measured apples-to-apples; add **stratified metrics** — bit by
rule-family, cipher by needs-vocab — so a regression concentrated in one stratum isn't diluted.

**0.4 Brace-aware diagnostic extractor** (`val/grading.py`): add `extract_final_answer_braceaware`
as a SEPARATE diagnostic (mirroring the `verify`/`verify_strict` split). Do NOT modify the verbatim
`extract_final_answer` mirror — keeping it faithful preserves leaderboard-predictiveness; the
diagnostic lets future cryptarithm evals report the model's true reasoning un-blinded from the grader
cap. Production-side brace-safe emission is deferred with the cryptarithm work.

### Phase 1 — Bit complex 3-input lever (next, the offense move)

Build a short (~150–250 line), locally-verifiable forward-gen trace emitter for MAJ/CHOICE/TT3:
induce `(a,b,c)+op` from an anchor (fully-determined) column → state stride propagation → per-example
verify gate (recompute all 8 bits, discard trace on mismatch) → per-bit apply lines (each a 3-input
lookup checkable from the query alone, zero carried state). Train → A/B on the now-trustworthy eval vs
the 0.86 baseline, **watching cipher/equation neighbors** (the induct failure mode) → ship only if
net-positive with no neighbor regression. Keep traces short to avoid the long-high-loss-gradient
interference that crashed neighbors before.

### Phase 2 — Cryptarithm (deferred, gated)

Run the designed STaR feasibility probe (K=64, temp 0.8, 40 easy-arith + 20 concat controls;
GREEN if ≥~12% arith pass@64, YELLOW if concat-only, RED if nothing). GREEN → small STaR loop;
YELLOW/RED → stay shelved. Only after Phase 0 + Phase 1 land.

## Acceptance criteria (Phase 0)

- Recalibrated bit generator: realized stats (over ≥400 seeds) match real within tolerance —
  ≥40% of pairwise problems mix ≥2 ops, ≥40% have per-column-varying offsets, complex family
  includes TT3, routing includes some arbitrary permutations. Answers still correct by construction.
- `val/kaggle_newrule_eval_standalone.py` inline copy stays byte-identical in logic to the module
  (existing invariant; a test guards it if present, else manual diff).
- Regression-sensitivity test passes: heuristic-only predictor scores lower on NEW than OLD.
- All existing `val/tests` pass (37+); new tests for each change.
- No change to the verbatim `extract_final_answer` / `verify`.

## Risks

- **R-distribution:** over-correcting the bit generator past real difficulty would make the eval
  pessimistic and mis-rank training runs. Mitigate by calibrating realized stats to the measured real
  numbers (the agent's tolerances above), not by maximizing hardness.
- **R-drift:** the inline standalone copy silently diverging from the module. Mitigate: port the same
  logic and keep/extend the byte-identity guard.
- **R-neighbor (Phase 1):** verbose bit traces crashing cipher/equation like induct did. Mitigate:
  short traces + neighbor A/B gate before shipping.
