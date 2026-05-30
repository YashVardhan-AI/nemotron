"""Load the snapshot-complement real holdout (problems never trained on)."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from reasoners.store_types import Problem

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_INDEX = (
    REPO_ROOT / "training" / "sft" / "04-08-16-14" / "logprobs" / "index.jsonl"
)
PROBLEMS_INDEX = REPO_ROOT / "problems.jsonl"

# Absent from the snapshot via *random downsampling* -> unbiased holdout.
# Other categories are absent via solver failure -> pessimistic floor, not clean.
CLEAN_CATEGORIES = {"numeral", "gravity", "unit_conversion"}

_DUP_SUFFIX = re.compile(r"-(?:p|d)\d+$")


def strip_dup_suffix(problem_id: str) -> str:
    return _DUP_SUFFIX.sub("", problem_id)


def trained_base_ids(index_path: Path = SNAPSHOT_INDEX) -> set[str]:
    ids: set[str] = set()
    with Path(index_path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(strip_dup_suffix(json.loads(line)["problem_id"]))
    return ids


@dataclass
class HoldoutProblem:
    problem: Problem
    clean: bool  # True only for CLEAN_CATEGORIES (unbiased baseline)


def holdout_problems(
    index_path: Path = SNAPSHOT_INDEX,
    problems_index: Path = PROBLEMS_INDEX,
) -> list[HoldoutProblem]:
    trained = trained_base_ids(index_path)
    out: list[HoldoutProblem] = []
    with Path(problems_index).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            pid = entry["id"]
            if pid in trained:
                continue
            problem = Problem.load_from_json(pid)
            out.append(
                HoldoutProblem(
                    problem=problem,
                    clean=problem.category in CLEAN_CATEGORIES,
                )
            )
    return out
