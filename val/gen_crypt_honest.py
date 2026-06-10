"""Build HONEST cryptarithm traces: render from the PURE-INFERENCE solve (no gold
hint), keeping ONLY the ~60% of real problems the solver recovers unconditioned.

WHY: `crypt_symbolic_corpus.load_symbolic_reasoning` rendered all 725 traces from
the GOLD-CONDITIONED program (`conditioned_on_answer & solver_correct`) — a
program reverse-fit to the known answer (44% little_endian + exotic ops like
mul_p1/gcd/mod). Those "derivations" don't generalize, which is why all three
trained arms scored ~0% at test even though the task is ~60% inferable (measured:
val/crypt_pi_rerun.py). This emitter instead uses the program the solver finds
WITHOUT the answer, and DROPS the un-inferable ~40% whose only "derivation" is
post-hoc rationalization.

Output: runs/crypt_symbolic/honest_real_<style>.jsonl  ({id, reasoning} per line)
Run: uv run python val/gen_crypt_honest.py [style] [N]
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reasoners import crypt_symbolic_trace as R  # noqa: E402

import solver_eq_symbolic as S  # noqa: E402

PARQUET = (
    ROOT
    / "kaggle-nemotron-equation-symbolic-main"
    / "kaggle-nemotron-equation-symbolic-main"
    / "data"
    / "solver_results.parquet"
)
PROBLEMS = ROOT / "problems.jsonl"
OUT_DIR = ROOT / "runs" / "crypt_symbolic"
_SOLVER_ERRORS = (ValueError, KeyError, IndexError, RecursionError, ZeroDivisionError)
_BOX = re.compile(r"\\boxed\{([^}]*)(?:\}|$)")


def _id2cat() -> dict[str, str]:
    out: dict[str, str] = {}
    with open(PROBLEMS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                out[obj["id"]] = obj.get("category", "?")
    return out


def main() -> None:
    style = sys.argv[1] if len(sys.argv) > 1 else "derive_search"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 659

    df = pd.read_parquet(PARQUET)
    df = df.assign(real_cat=df["id"].map(_id2cat()))
    sub = df[df["real_cat"] == "cryptarithm_deduce"].reset_index(drop=True)
    rows = list(sub.itertuples(index=False))[:n]

    print(
        f"HONEST trace build  style={style}  on {len(rows)} real cryptarithm_deduce\n"
    )
    out: dict[str, str] = {}
    drop_ids: list[str] = []  # un-inferable arithmetic: skip in corpus (no poison)
    solved = rendered = concat_skip = roundtrip_fail = 0
    mode_ct: Counter[str] = Counter()
    char_lens: list[int] = []
    t0 = time.monotonic()

    for i, r in enumerate(rows, 1):
        solver = S.AliceEquationSolver(r.prompt)  # no answer_hint => pure inference
        try:
            ans, det = solver.solve()
        except _SOLVER_ERRORS:
            ans, det = None, None
        gold = str(r.answer).strip()
        if ans is None or str(ans).strip() != gold or not isinstance(det, dict):
            drop_ids.append(r.id)  # pure inference can't recover it honestly -> drop
            continue
        solved += 1
        if det.get("type") != "arithmetic" or det.get("mode") is None:
            concat_skip += 1  # pure-concat: its real reasoning is already correct
            continue
        text = R.render(
            r.prompt,
            det["mapping"],
            det["ops"],
            str(det["mode"]),
            int(det["solver_radix"]),
            gold,
            style=style,
        )
        if text is None or not (m := _BOX.findall(text)) or m[-1].strip() != gold:
            roundtrip_fail += 1
            drop_ids.append(r.id)  # solved but cannot render honestly -> drop too
            continue
        out[r.id] = text
        rendered += 1
        mode_ct[str(det["mode"])] += 1
        char_lens.append(len(text))
        if i % 100 == 0 or i == len(rows):
            el = time.monotonic() - t0
            print(
                f"  [{i:>3}/{len(rows)}] solved {solved} rendered {rendered} "
                f"concat {concat_skip} rtfail {roundtrip_fail}  {el:.0f}s",
                flush=True,
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / f"honest_real_{style}.jsonl"
    with open(cache, "w", encoding="utf-8") as f:
        for pid, text in out.items():
            f.write(json.dumps({"id": pid, "reasoning": text}) + "\n")
    drop_path = OUT_DIR / "crypt_drop_ids.json"
    with open(drop_path, "w", encoding="utf-8") as f:
        json.dump(sorted(drop_ids), f)

    char_lens.sort()
    p50 = char_lens[len(char_lens) // 2] if char_lens else 0
    p90 = char_lens[int(len(char_lens) * 0.9)] if char_lens else 0
    print("\n" + "=" * 60)
    print(f"solved (pure-inf)   : {solved}/{len(rows)} = {solved / len(rows):.1%}")
    print(f"rendered (arith)    : {rendered}   concat-skip {concat_skip}")
    print(f"round-trip failures : {roundtrip_fail}")
    print(f"drop (un-inferable) : {len(drop_ids)}  -> {drop_path.name}")
    print(f"mode mix (rendered) : {dict(mode_ct)}")
    print(f"char len p50/p90    : {p50} / {p90}  (~{p50 // 4}/{p90 // 4} tok est)")
    print(f"wrote {rendered} honest traces -> {cache}")


if __name__ == "__main__":
    main()
