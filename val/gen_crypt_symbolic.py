"""Driver + validation harness for the symbolic cryptarithm trace renderer.

Reads the verified solver's gold-conditioned programs from
`solver_results.parquet`, renders a CoT for each (Adaptive: full MRV deduction
where sound + affordable, else lean), and validates the result:

  * round-trip: the rendered `\\boxed{}` must reproduce gold (brace-aware), else
    the trace is DROPPED — correctness by construction, never a bad training row.
  * token budget: completion tokenized exactly as corpus.py does
    (`Tokenizer.from_file(tokenizer.json)`, completion = reasoning + closing);
    report distribution + % over the 7680 decode budget.
  * glyph un-merge: confirm space-separated reasoning glyphs tokenize cleanly.
  * yield by mode/base, brace-bug (leaderboard cap) exposure, sample dump.

Run:  uv run python val/gen_crypt_symbolic.py
Outputs: runs/crypt_symbolic/{rendered.jsonl, samples.txt} + a printed readout.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoners import crypt_symbolic_trace as R  # noqa: E402
from val.grading import (  # noqa: E402
    extract_final_answer,
    extract_final_answer_braceaware,
    verify,
)

PARQUET = (
    ROOT
    / "kaggle-nemotron-equation-symbolic-main"
    / "kaggle-nemotron-equation-symbolic-main"
    / "data"
    / "solver_results.parquet"
)
TOKENIZER_PATH = ROOT / "tokenizer.json"
OUT_DIR = ROOT / "runs" / "crypt_symbolic"

BUDGET_HARD = 7680  # eval decode budget; completions must fit under this
BUDGET_FULL = 7000  # informational: headroom before the decode budget

# A/B style: "assert" (MRV forced scratchpad) vs "derive" (genuine propagation) vs
# "derive_inductive" (bounded one-row-per-glyph + LSB-first encoding) vs "lean" (no
# digit-deduction). Base-10 std/alice get the deduction; the rest fall back to lean.
STYLE = sys.argv[1] if len(sys.argv) > 1 else "assert"
assert STYLE in (
    "assert",
    "derive",
    "derive_inductive",
    "derive_search",
    "lean",
), f"bad style {STYLE!r}"

_tok = Tokenizer.from_file(str(TOKENIZER_PATH))


def wrap(reasoning: str, gold: str) -> str:
    """Exactly corpus.py's completion assembly."""
    return f"{reasoning}\n</think>\n\\boxed{{{gold}}}<|im_end|>"


def n_tokens(completion: str) -> int:
    return len(_tok.encode(completion, add_special_tokens=False).ids)


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    k = max(
        0, min(len(sorted_vals) - 1, int(round((p / 100) * (len(sorted_vals) - 1))))
    )
    return sorted_vals[k]


def parse_program(row):
    """Return (mapping, ops, mode, base) or None if this row should be skipped."""
    mode = row["solver_mode"]
    radix = row["solver_radix"]
    if pd.isna(mode) or pd.isna(radix):
        return None  # pure-concat (no arithmetic) — covered by real concat traces
    try:
        mapping = {k: int(v) for k, v in json.loads(row["solver_mapping"]).items()}
        ops = dict(json.loads(row["solver_ops"]))
    except (ValueError, TypeError):
        return None
    return mapping, ops, str(mode), int(radix)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PARQUET)
    gold_pool = df[df["conditioned_on_answer"] & df["solver_correct"]].reset_index(
        drop=True
    )

    rendered = []
    drops = []
    deduct_fallbacks = []
    skipped_concat = 0

    for _, row in gold_pool.iterrows():
        gold = str(row["answer"])
        prog = parse_program(row)
        if prog is None:
            skipped_concat += 1
            continue
        mapping, ops, mode, base = prog
        prompt = row["prompt"]
        rid = row["id"]

        eligible_for_deduction = (
            base == 10 and mode in ("standard", "alice") and STYLE != "lean"
        )
        res = R.render_with_tier(prompt, mapping, ops, mode, base, gold, style=STYLE)
        if res is None:
            drops.append((rid, mode, base, "render_returned_None"))
            continue
        reasoning, tier = res
        # eligible by base/mode but the deduction did not actually render -> _derive
        # failed sanity and we silently used the lean map; track (NOT a drop).
        if eligible_for_deduction and tier == "lean":
            deduct_fallbacks.append(rid)

        completion = wrap(reasoning, gold)
        nt = n_tokens(completion)
        ext_comp = extract_final_answer(completion)  # competition grader (brace bug)
        ext_brace = extract_final_answer_braceaware(completion)  # true emission
        ok_brace = verify(gold, ext_brace)
        gradeable = verify(gold, ext_comp)

        if not ok_brace:
            # renderer emitted the wrong answer string -> drop (must not happen)
            drops.append((rid, mode, base, f"roundtrip_fail ext={ext_brace!r}"))
            continue

        rendered.append(
            {
                "id": rid,
                "mode": mode,
                "base": base,
                "tier": tier,
                "n_tokens": nt,
                "gradeable": bool(gradeable),
                "has_brace": "}" in gold,
                "gold": gold,
                "completion": completion,
            }
        )

    # ---------------- write outputs ----------------
    out_jsonl = OUT_DIR / f"rendered_{STYLE}.jsonl"
    out_samples = OUT_DIR / f"samples_{STYLE}.txt"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rendered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # sample dump: spread across tiers / modes / bases
    seen_keys = set()
    samples = []
    for r in rendered:
        key = (r["tier"], r["mode"], r["base"] == 10)
        if key not in seen_keys:
            seen_keys.add(key)
            samples.append(r)
    # add a couple brace-answer + base!=10 examples explicitly
    for r in rendered:
        if len(samples) >= 10:
            break
        if (r["has_brace"] or r["base"] != 10) and r not in samples:
            samples.append(r)
    with open(out_samples, "w", encoding="utf-8") as f:
        for r in samples[:10]:
            f.write(
                f"===== id={r['id']} tier={r['tier']} mode={r['mode']} "
                f"base={r['base']} tokens={r['n_tokens']} gold={r['gold']!r} "
                f"gradeable={r['gradeable']} =====\n"
            )
            f.write(r["completion"])
            f.write("\n\n")

    # ---------------- glyph un-merge spot check ----------------
    # The cryptarithm task is positional; BPE fusing operator into operand
    # scrambles it. Confirm space-separation un-merges (per cryptarithm-tokenizer-merge).
    merged = "()+^^"
    separated = "( ) + ^ ^"
    merged_toks = _tok.encode(merged, add_special_tokens=False).tokens
    sep_toks = _tok.encode(separated, add_special_tokens=False).tokens

    def _core(t):  # strip byte-level BPE markers (space=Ġ Ġ, newline=Ċ Ċ)
        return t.replace("Ġ", "").replace("Ċ", "")

    sep_clean = [_core(t) for t in sep_toks if _core(t)]
    unmerge_ok = all(len(t) <= 1 for t in sep_clean)

    # ---------------- summary ----------------
    toks = sorted(r["n_tokens"] for r in rendered)
    n = len(rendered)
    over_hard = sum(1 for t in toks if t > BUDGET_HARD)
    over_full = sum(1 for t in toks if t > BUDGET_FULL)
    tier_ct = Counter(r["tier"] for r in rendered)
    gradeable_ct = sum(1 for r in rendered if r["gradeable"])
    brace_ct = sum(1 for r in rendered if r["has_brace"])

    def yield_by(field):
        c = Counter(r[field] for r in rendered)
        return dict(sorted(c.items(), key=lambda kv: str(kv[0])))

    print("=" * 64)
    print(f"SYMBOLIC CRYPTARITHM TRACE — VALIDATION READOUT  [style={STYLE}]")
    print("=" * 64)
    print(f"gold-conditioned+correct pool : {len(gold_pool)}")
    print(f"  skipped (pure-concat nan)    : {skipped_concat}")
    print(f"  attempted (arithmetic/mixed) : {len(gold_pool) - skipped_concat}")
    print(f"  RENDERED + round-trip OK     : {n}")
    print(f"  dropped                      : {len(drops)}")
    print()
    print(f"deduction tier counts          : {dict(tier_ct)}")
    if STYLE != "lean":
        print(
            f"  base-10 std eligible but fell back to lean (_derive fail): "
            f"{len(deduct_fallbacks)}"
        )
    print()
    print("ROUND-TRIP: every rendered trace reproduces gold (brace-aware) by")
    print(
        f"            construction. round-trip failures dropped: "
        f"{sum(1 for d in drops if 'roundtrip' in d[3])}"
    )
    print()
    print(
        f"competition-gradeable          : {gradeable_ct}/{n} "
        f"({100 * gradeable_ct / max(n, 1):.1f}%)"
    )
    print(
        f"  brace-capped (gold has '}}')   : {brace_ct} "
        f"(hard leaderboard cap, still valid training data)"
    )
    print()
    print("TOKEN BUDGET (completion, tokenizer.json):")
    if toks:
        print(
            f"  p50={pct(toks, 50)}  p90={pct(toks, 90)}  p99={pct(toks, 99)}  "
            f"max={toks[-1]}"
        )
    print(f"  > {BUDGET_HARD} (decode budget): {over_hard}  (MUST be ~0)")
    print(f"  > {BUDGET_FULL} (headroom check): {over_full}")
    print()
    print(f"glyph un-merge: '{merged}' -> {merged_toks}")
    print(f"               '{separated}' -> {sep_clean}  clean={unmerge_ok}")
    print()
    print(f"yield by mode : {yield_by('mode')}")
    print(f"yield by base : {yield_by('base')}")
    print()
    if drops:
        print("DROP SAMPLES (first 10):")
        for d in drops[:10]:
            print("  ", d)
    print()
    print(f"outputs -> {out_jsonl} , {out_samples}")


if __name__ == "__main__":
    main()
