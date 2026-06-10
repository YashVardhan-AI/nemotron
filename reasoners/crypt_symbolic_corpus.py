"""Build correct cryptarithm reasoning from the verified symbolic solver, keyed by
real train id, for injection into the corpus.

`corpus.py` calls `load_symbolic_reasoning(style)` and, for every real cryptarithm
problem we can solve, substitutes the returned reasoning text for the wrong
concat-fallback `reasoning/<id>.txt` (see memory `crypt-symbolic-renderer-built`).
The reasoning text ends in `So the answer is \\boxed{gold}`, so corpus.py's
existing completion assembly (extract last \\boxed -> gold -> wrap) works unchanged.

`style` is the A/B knob: "assert" (MRV forced/guess scratchpad) vs "derive"
(genuine constraint-propagation deduction) vs "derive_inductive" (bounded
one-row-per-glyph deduction + LSB-first encoding — the step-count-independent
recast) vs "derive_search" (the LONG, equation_numeric-style enumerate-and-test
operator id — ~2.2k tok p50, the high-budget arm) vs "lean" (state map only).
All are validated correct by `val/gen_crypt_symbolic.py` (725/725, 0 round-trip fail).
"""

from __future__ import annotations

import json
from pathlib import Path

from reasoners import crypt_symbolic_trace as R

_PARQUET = (
    Path(__file__).resolve().parent.parent
    / "kaggle-nemotron-equation-symbolic-main"
    / "kaggle-nemotron-equation-symbolic-main"
    / "data"
    / "solver_results.parquet"
)


def load_symbolic_reasoning(style: str = "assert") -> dict[str, str]:
    """Return {train_id: reasoning_text} for every gold-conditioned, solver-correct
    cryptarithm problem the renderer can reproduce (round-trip-checked inside
    `render`). Pure-concat rows with no arithmetic mode are skipped (their real
    concat reasoning is already correct)."""
    import pandas as pd

    df = pd.read_parquet(_PARQUET)
    pool = df[df["conditioned_on_answer"] & df["solver_correct"]]
    out: dict[str, str] = {}
    for _, row in pool.iterrows():
        mode, radix = row["solver_mode"], row["solver_radix"]
        if pd.isna(mode) or pd.isna(radix):
            continue
        try:
            mapping = {k: int(v) for k, v in json.loads(row["solver_mapping"]).items()}
            ops = dict(json.loads(row["solver_ops"]))
        except (ValueError, TypeError):
            continue
        text = R.render(
            row["prompt"],
            mapping,
            ops,
            str(mode),
            int(radix),
            str(row["answer"]),
            style=style,
        )
        if text is not None:
            out[row["id"]] = text
    return out


_HONEST_DIR = Path(__file__).resolve().parent.parent / "runs" / "crypt_symbolic"


def load_honest_symbolic_reasoning(style: str = "derive_search") -> dict[str, str]:
    """Return {train_id: reasoning_text} for HONEST cryptarithm traces — rendered
    from the PURE-INFERENCE solve (no gold hint), keeping only the ~62% of real
    problems the solver recovers unconditioned.

    This is the fix for the gold-conditioned defect: `load_symbolic_reasoning`
    rendered every trace from a program reverse-fit to the known answer (44%
    little_endian + exotic ops), which does not generalize (all three trained
    arms -> ~0% at test). Cache is produced by `val/gen_crypt_honest.py`; returns
    {} if it is missing (so corpus build fails loudly via the empty-dict print)."""
    cache = _HONEST_DIR / f"honest_real_{style}.jsonl"
    out: dict[str, str] = {}
    if not cache.exists():
        return out
    with open(cache, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                out[obj["id"]] = obj["reasoning"]
    return out


def load_crypt_drop_ids() -> set[str]:
    """Ids of un-inferable ARITHMETIC cryptarithm problems to SKIP in the corpus
    (pure inference could not recover them honestly). Their concat-fallback
    reasoning is wrong and the gold-conditioned trace is non-generalizing, so the
    honest arm trains on neither. Produced alongside the honest cache."""
    p = _HONEST_DIR / "crypt_drop_ids.json"
    if not p.exists():
        return set()
    return set(json.loads(p.read_text(encoding="utf-8")))
