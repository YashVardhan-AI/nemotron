"""Aggregate and render validation results by category and difficulty."""

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from val.scoring import Result


def _stats(results: Sequence[Result]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "accuracy_strict": 0.0}
    return {
        "n": n,
        "accuracy": sum(r.correct for r in results) / n,
        "accuracy_strict": sum(r.correct_strict for r in results) / n,
    }


def aggregate(results: Sequence[Result]) -> dict:
    by_cat: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)

    report: dict[str, dict] = {}
    for category, cat_results in by_cat.items():
        by_diff: dict[int, list[Result]] = defaultdict(list)
        for r in cat_results:
            by_diff[r.difficulty].append(r)
        report[category] = {
            "overall": _stats(cat_results),
            "by_difficulty": {d: _stats(rs) for d, rs in sorted(by_diff.items())},
        }
    return report


def format_table(report: dict) -> str:
    lines = [f"{'Category':<24} {'Diff':>5} {'N':>5} {'Acc%':>8} {'StrictAcc%':>11}"]
    lines.append("-" * 56)
    for category in sorted(report):
        overall = report[category]["overall"]
        lines.append(
            f"{category:<24} {'all':>5} {overall['n']:>5} "
            f"{overall['accuracy'] * 100:>8.1f} {overall['accuracy_strict'] * 100:>11.1f}"
        )
        for diff, stats in report[category]["by_difficulty"].items():
            lines.append(
                f"{'':<24} {diff:>5} {stats['n']:>5} "
                f"{stats['accuracy'] * 100:>8.1f} {stats['accuracy_strict'] * 100:>11.1f}"
            )
    return "\n".join(lines)


def write_json(report: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2))
