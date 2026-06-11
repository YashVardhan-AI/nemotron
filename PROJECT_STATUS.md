# Nemotron Reasoning Challenge — Project Status

**Goal:** push the rank-32 LoRA fine-tune of Nemotron-3 Nano 30B-A3B (MoE, ~3B active)
from **0.86 → ≥0.89** exact-match on the hidden reasoning test. Decoded **greedily at
temp=0, max_tokens 7680, single forward pass** (no sampling / self-consistency).

**Pipeline:** procedural datagen (`corpus.py` → `corpus.jsonl`) → SFT LoRA via
`train_sft.py` on the Tinker backend → adapter eval on Kaggle (`val/` standalone cells).
**Shipping baseline:** commit `a52f4a141d`, 7830 rows, 1 epoch / ~245 steps, rank-32,
LR 2e-4 → 0. This recipe is byte-for-byte the prize-winning submission — strong evidence
0.86 is the SFT-family ceiling for this data.

## Where the score lives
9 reasoning categories. Saturated (~100%, no headroom): gravity, numeral,
unit_conversion, cipher. The remaining error budget is concentrated in three induction
families: **cryptarithm**, **heterogeneous bit_manipulation**, **secret-op
equation_numeric**.

## DEAD levers — measured, do not retry
- **Cryptarithm via SFT — DEAD, 4 independent arms.** assert (0.84), induct (0.83),
  derive_search (0.83), honest pure-inference (0.82). Test arithmetic stays **exactly 0%**
  on every op (add/mul/abs_diff/rev_concat); only `concat` (a string op, no arithmetic)
  ever moves (~30%). The honest arm trained on the EXACT program that solves 62% of these
  offline by pure inference and STILL produced 0% — so the bottleneck is NOT data
  honesty / format / length / coverage. A rank-32 LoRA at greedy temp-0 cannot execute the
  latent base-detect + op-induction + MRV + base-N arithmetic search at inference.
  Detail: memory `cryptarithm-via-sft-dead`.
- **Bit forward-gen (BIT_N=800) — DEAD.** Crashed neighbors (bit 88→22, hom 90→9);
  a trace-shape failure, not proof of an info ceiling.
- **Frontier-teacher distillation — DEAD (~+0.006).** The student already matches/beats
  Claude-as-teacher pass@1 on every solvable category (bit 88 vs 55, eq 87.5 vs 75). The
  hard bit tier is budget-locked (teacher traces 9k–28k tok ≫ 7680).
- **2-epoch — FLAT (0.86).** Reverted to 1 epoch.
- **Trace format/length, category coverage, adapter SVD rank — all NULL.**

## OPEN levers — not yet exhausted
- **bit-het lenient-grader lever (~+0.5–1.3pp, risky).** The REAL grader parses binary
  answers as a DECIMAL and accepts within 1% (`math.isclose rel_tol=1e-2`) → only the
  leftmost ~2 of 8 bits must be exact. Het bit is INFO-capped at exact-match (model strict
  3.3% = oracle 6.5%), but under the lenient grader the model (46.7%) sits below the
  determine-then-guess oracle (55%) and MAP oracle (71%) on pairwise/het (5.4% of test) →
  real model-error headroom. Modest, **high neighbor-crash risk**, NOT a standalone path to
  0.89. Detail: memory `bit-het-identifiability`. (The model-free probe + Kaggle
  failure-dump cell were removed in cleanup; recover from git history if resumed.)
- **RLVR (GRPO/RFT, exact-match reward) — the only theory-backed path for the rest
  (~+0.01–0.02).** bit/equation have pass-rate signal; cryptarithm arithmetic does not (so
  even RL is uncertain there). The repo has **no rollout loop** — `train_sft.py`
  ppo/cispo/dro are offline reweighted-SFT, NOT RLVR. Would need a Tinker rollout harness
  that does not exist. Detail: memory `repo-has-no-real-rl-loop`.

## Reusable insight
The **lenient grader rewards high-order bits** — applies to every numeric/binary-answer
category (bit_manipulation, equation_numeric). Cryptarithm is EXEMPT (glyph-string answers
→ exact string match → no partial credit → the honest 0%). Strict-match ceilings
understate the scorable ceiling for bit/equation.

## Kept tooling
- **Rust cryptarithm generator** — `reasoners/crypt_symbolic_gen.py` +
  `val/gen_crypt_synth.py` + `reasoners/crypt_symbolic_trace.py`, backed by the
  Rust-accelerated solver in `kaggle-nemotron-equation-symbolic-main/` + the `dist/` wheel.
  Rich symbolic generator: samples real marginals, variable base, 3 digit-order modes,
  unhinted-solver learnability filter; 28× speedup from the Rust CSP search. Rebuild
  instructions: memory `rust-solver-accelerator-build`. Kept as a reusable asset even though
  the cryptarithm-SFT arm it fed is dead.
- **`val/` validation harness** — per-category / per-difficulty generalization eval through
  the real greedy / `verify()` path (Kaggle standalone cells). Design:
  `docs/superpowers/plans/2026-05-30-val-harness.md`.

## Decode / grading facts
- Live submission: temp=0 greedy, ~7680 max_tokens. Truncation is NOT a lever (all traces fit).
- The `\boxed{}` extractor truncates at the first `}` → answers containing `}` (cryptarithm
  glyphs) are partially ungradeable.

---
_Full evidence trail lives in the project memory (`.claude/projects/.../memory/`). This file
supersedes the per-experiment plan docs (cryptarithm-p1, induction-trace, eval-harden) removed
in the 2026-06-11 cleanup; they remain in git history._
