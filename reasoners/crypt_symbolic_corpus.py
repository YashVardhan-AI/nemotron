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
recast) vs "lean" (state map only). All are validated correct by
`val/gen_crypt_symbolic.py` (725/725, 0 round-trip fail).
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
