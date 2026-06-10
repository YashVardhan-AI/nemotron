"""Decisive data-truth probe: resolve the 12%-vs-72% cryptarithm contradiction.

Reads the rich solver's results parquet (NO re-run) and reports, per real
category and SEPARATELY for gold-CONDITIONED vs PURE-INFERENCE solves:
  - coverage (how many real problems the solver even appears for)
  - ABSOLUTE pure-inference solvable rate = correct_pure_inf / all_real (the
    true achievable ceiling, not a conditional accuracy on an easy sliver)
  - base / mode / arith-op distribution among CORRECT solves

The pure-inference distribution is the trustworthy generative signal; the
gold-conditioned one over-uses free DOF (variable base / reversed mode / exotic
ops like mul_p1) and is what all three failed training arms were shaped from.

Run: uv run python val/crypt_truth_probe.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARQUET = (
    ROOT
    / "kaggle-nemotron-equation-symbolic-main"
    / "kaggle-nemotron-equation-symbolic-main"
    / "data"
    / "solver_results.parquet"
)
PROBLEMS = ROOT / "problems.jsonl"

FOCUS = ("cryptarithm_deduce", "equation_numeric_deduce")


def _show(counter: Counter[object]) -> str:
    total = sum(counter.values())
    if total == 0:
        return "(none)"
    return "  ".join(
        f"{k}={v} ({v / total:.0%})"
        for k, v in sorted(counter.items(), key=lambda kv: -kv[1])
    )


def _dist(frame: pd.DataFrame, label: str) -> None:
    base: Counter[object] = Counter()
    mode: Counter[object] = Counter()
    ops: Counter[object] = Counter()
    for _, r in frame.iterrows():
        if pd.notna(r["solver_radix"]):
            base[int(r["solver_radix"])] += 1
        if pd.notna(r["solver_mode"]):
            mode[str(r["solver_mode"])] += 1
        raw = r["solver_ops"]
        if isinstance(raw, str):
            ops.update(json.loads(raw).values())
    print(f"    [{label}]  n={len(frame)}")
    print(f"       base: {_show(base)}")
    print(f"       mode: {_show(mode)}")
    print(f"       ops:  {_show(ops)}")


def main() -> None:
    id2cat: dict[str, str] = {}
    cat_counts: Counter[object] = Counter()
    with open(PROBLEMS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            id2cat[obj["id"]] = obj.get("category", "?")
            cat_counts[obj.get("category", "?")] += 1

    df = pd.read_parquet(PARQUET)
    df = df.assign(real_cat=df["id"].map(id2cat))

    print(f"parquet rows: {len(df)}   distinct ids: {df['id'].nunique()}")
    print("parquet real_cat counts:")
    print(df["real_cat"].value_counts(dropna=False).to_string())
    print()

    for cat in FOCUS:
        n_real = cat_counts[cat]
        sub = df[df["real_cat"] == cat]
        gc = sub[sub["conditioned_on_answer"]]
        pi = sub[~sub["conditioned_on_answer"]]
        pi_correct = int(pi["solver_correct"].sum())
        gc_correct = int(gc["solver_correct"].sum())

        print("=" * 70)
        print(f"{cat}   (real total: {n_real})")
        print(f"  rows in parquet           : {len(sub)}  (ids {sub['id'].nunique()})")
        print(
            f"  gold-conditioned          : {gc_correct}/{len(gc)} correct"
            f"  -> coverage {sub['id'].nunique() / n_real:.1%} of real"
        )
        pi_acc = (pi_correct / len(pi)) if len(pi) else 0.0
        print(
            f"  pure-inference (no gold)  : {pi_correct}/{len(pi)} correct"
            f"  (conditional acc {pi_acc:.0%})"
        )
        print(
            f"  ABSOLUTE pure-inf ceiling : {pi_correct}/{n_real} = "
            f"{pi_correct / n_real:.1%} of ALL real {cat}"
        )
        _dist(gc[gc["solver_correct"]], "gold-cond correct")
        _dist(pi[pi["solver_correct"]], "pure-inf correct")
        print()


if __name__ == "__main__":
    main()
