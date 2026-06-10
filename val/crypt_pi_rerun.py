"""Resolve the cryptarithm ceiling: re-run PURE INFERENCE (no gold hint) on the
REAL cryptarithm_deduce problems and count how many the solver recovers.

The parquet only ran 79/659 unconditioned; the other 580 were gold-conditioned,
so the "72% pure-inference" was a conditional accuracy on a sliver. This measures
the ABSOLUTE pure-inference ceiling directly: solved==gold / attempted. If it
lands ~9-15%, cryptarithm_deduce is data-level under-determined (no trace helps,
the 3 failed arms are explained). If ~70%, a correctly-shaped trace is viable.

Run: uv run python val/crypt_pi_rerun.py [N] [wall_budget_s]
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reasoners import crypt_symbolic_trace as R  # noqa: E402,F401  (sets sys.path)

import solver_eq_symbolic as S  # noqa: E402

PARQUET = (
    ROOT
    / "kaggle-nemotron-equation-symbolic-main"
    / "kaggle-nemotron-equation-symbolic-main"
    / "data"
    / "solver_results.parquet"
)
PROBLEMS = ROOT / "problems.jsonl"
_SOLVER_ERRORS = (ValueError, KeyError, IndexError, RecursionError, ZeroDivisionError)


def pure_infer(prompt: str) -> str | None:
    """Best pure-inference answer (no gold hint): concat -> arithmetic -> deep."""
    solver = S.AliceEquationSolver(prompt)  # no answer_hint => pure inference
    ans, _ = solver.solve()
    return ans


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0

    id2cat: dict[str, str] = {}
    with open(PROBLEMS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                id2cat[obj["id"]] = obj.get("category", "?")

    df = pd.read_parquet(PARQUET)
    df = df.assign(real_cat=df["id"].map(id2cat))
    sub = df[df["real_cat"] == "cryptarithm_deduce"].reset_index(drop=True)
    rows = list(sub.itertuples(index=False))
    random.Random(0).shuffle(rows)
    rows = rows[:n]

    print(
        f"PURE-INFERENCE re-run on {len(rows)} real cryptarithm_deduce "
        f"(of {len(sub)}); wall budget {budget:.0f}s\n"
    )
    solved = attempted = errors = 0
    t0 = time.monotonic()
    for i, r in enumerate(rows, 1):
        if time.monotonic() - t0 > budget:
            print(f"\n[wall budget reached after {attempted} attempts]")
            break
        attempted += 1
        try:
            ans = pure_infer(r.prompt)
        except _SOLVER_ERRORS:
            errors += 1
            ans = None
        if ans is not None and str(ans).strip() == str(r.answer).strip():
            solved += 1
        if i % 25 == 0 or i == len(rows):
            el = time.monotonic() - t0
            print(
                f"  [{i:>3}/{len(rows)}] solved {solved}  "
                f"({solved / attempted:.0%})  errs {errors}  {el:.0f}s "
                f"({el / attempted:.2f}s/ea)",
                flush=True,
            )

    print("\n" + "=" * 60)
    print(
        f"ABSOLUTE pure-inference ceiling: {solved}/{attempted} = "
        f"{solved / max(attempted, 1):.1%}"
    )
    print("Compare: simple-solver coverage ~12%, model ~0-10%, parquet 9%.")


if __name__ == "__main__":
    main()
