# Rule-Disjoint Validation Harness (P0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the enabling validation harness from the roadmap's P0 gate — a way to measure per-category, per-difficulty generalization accuracy of a trained adapter on rules it has never seen, scored through the *real* greedy/vLLM/`verify()` path.

**Architecture:** A new isolated `val/` package. Pure-Python pieces (grading, scoring loop, reporting, real-holdout loader, holdout registry, synthetic generators) are fully unit-tested locally with an **injectable predictor** so no GPU is needed for tests. A separate `val/run_vllm.py` swaps in the real vLLM+adapter predictor for the GPU/Kaggle scoring run and scores **both** the real holdout and a freshly-generated synthetic holdout. This plan delivers the framework + a real-holdout baseline (clean on numeral/gravity/unit_conversion) + **two** synthetic generators: `cipher` (reference) and `bit_manipulation` (so the first GPU run covers one genuinely *hard* category with a clean baseline, not just saturated ones). The other 7 synthetic generators are an explicitly-scoped follow-on (roadmap lever #2).

**Tech Stack:** Python 3.11, `uv`, pytest, ruff. Reuses `reasoners.store_types.Problem`. Grading/prompt code is copied verbatim from `notebook_tinker.py` and pinned by tests to prevent drift.

---

## Why this is one plan (scope check)

P0 in `C:\Users\porus\.claude\plans\you-are-acting-as-deep-fountain.md` has two halves:
1. **The harness** (this plan): registry, scoring via the real path, reporting by category×difficulty, real-holdout baseline, permanent holdout registry, and **two** generators (`cipher` + `bit_manipulation`) proving the synthetic path end-to-end on one easy and one hard category.
2. **The 7 remaining per-category synthetic generators** — these *are* roadmap lever #2 ("forward-generate diverse rules"), each substantial. They plug into the generator protocol defined here and are a **follow-on plan**. The cryptarithm/`_guess`/equation generators stay here because they need rule-family mining + deducibility control, not just a closed-form rule.

Splitting keeps this plan testable and shippable on its own: after it lands we have clean baselines on numeral/gravity/unit_conversion (real holdout) plus cipher and bit_manipulation (synthetic) — i.e. three saturated sanity categories and one hard category.

## File structure

- `val/__init__.py` — package marker
- `val/conftest.py` — makes repo root importable in tests
- `val/grading.py` — `extract_final_answer`, `verify` (verbatim copies of the real path; pinned)
- `val/scoring.py` — `build_eval_prompt`, `Result`, `score`
- `val/report.py` — `aggregate`, `format_table`, `write_json`
- `val/real_holdout.py` — `trained_base_ids`, `holdout_problems`, `CLEAN_CATEGORIES`
- `val/holdout_registry.py` — `HoldoutRegistry` (reserve/exclude rule signatures, JSON-backed)
- `val/generators/__init__.py` — `GENERATORS` registry, `GeneratorSpec` protocol, auto-import of generator modules
- `val/generators/cipher.py` — `generate` (reference: new substitution cipher), `rule_signature`
- `val/generators/bit_manipulation.py` — `generate`/`build_rule`/`rule_signature` (affine-mod-256, xor, rotate, permutation)
- `val/run_vllm.py` — real vLLM predictor + `build_synthetic_valset` + `main()` (GPU/Kaggle; not unit-tested)
- `val/run_local_demo.py` — end-to-end smoke with a stub predictor (no GPU)
- `val/README.md` — how to run locally and on GPU
- `val/tests/test_*.py` — one test module per unit above

Run tests with explicit paths (overrides the repo's `testpaths=[".claude"]`):
`uv run pytest val/tests -v`

---

## Task 0: Scaffold the `val/` package

**Files:**
- Create: `val/__init__.py`
- Create: `val/conftest.py`
- Create: `val/tests/__init__.py`
- Create: `val/tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

`val/tests/test_smoke.py`:
```python
def test_package_importable():
    import val

    assert val is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest val/tests/test_smoke.py -v`
Expected: FAIL — collection error, no module named `val`.

- [ ] **Step 3: Create the package files**

`val/__init__.py`:
```python
"""Rule-disjoint validation harness (roadmap P0)."""
```

`val/tests/__init__.py`:
```python
```

`val/conftest.py` (lets tests import top-level repo modules like `reasoners`):
```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest val/tests/test_smoke.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add val/__init__.py val/conftest.py val/tests/__init__.py val/tests/test_smoke.py
git commit -m "feat(val): scaffold validation harness package"
```

---

## Task 1: Grading — `extract_final_answer` + `verify` (pinned to the real path)

**Why verbatim copies:** the competition scores through `notebook_tinker.py`'s `extract_final_answer`/`verify`. Importing `notebook_tinker` pulls heavy/Kaggle-only deps, so we copy the two pure functions and pin their behavior with tests. **Gotcha to capture:** `verify` does NOT special-case binary strings — it tries `float()` first, so two *different* binary strings within 1% relative tolerance compare **equal**. That is the real grader's behavior; the harness must reproduce it, and we also expose a strict comparison for diagnostics.

**Files:**
- Create: `val/grading.py`
- Test: `val/tests/test_grading.py`

- [ ] **Step 1: Write the failing test**

`val/tests/test_grading.py`:
```python
from val.grading import extract_final_answer, verify, verify_strict


def test_extract_boxed():
    assert extract_final_answer(r"The answer is \boxed{42}") == "42"


def test_extract_last_nonempty_boxed():
    assert extract_final_answer(r"\boxed{} then \boxed{7}") == "7"


def test_extract_final_answer_phrase():
    assert extract_final_answer("The final answer is: 3.14") == "3.14"


def test_extract_trailing_number():
    assert extract_final_answer("Just a number 100 in text") == "100"


def test_extract_none():
    assert extract_final_answer(None) == "NOT_FOUND"


def test_verify_numeric_tolerance():
    assert verify("24.64", "24.6401") is True


def test_verify_case_insensitive_text():
    assert verify("XLVII", "xlvii") is True


def test_verify_binary_is_float_lenient():
    # Real-grader gotcha: binary strings go through float(), so two DIFFERENT
    # binary strings within 1% compare EQUAL. This pins that behavior.
    assert verify("11011011", "11011010") is True


def test_verify_strict_binary_is_exact():
    assert verify_strict("11011011", "11011010") is False
    assert verify_strict("11011011", "11011011") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest val/tests/test_grading.py -v`
Expected: FAIL — no module named `val.grading`.

- [ ] **Step 3: Write the implementation**

`val/grading.py`:
```python
"""Grading copied verbatim from notebook_tinker.py (the real scoring path).

Pinned by val/tests/test_grading.py. If notebook_tinker.py's extract_final_answer
or verify change, update these and the tests together.
"""

import math
import re


def extract_final_answer(text: str | None) -> str:
    """Extract the final answer; mirrors notebook_tinker.extract_final_answer."""
    if text is None:
        return "NOT_FOUND"

    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        return matches[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def verify(stored_answer: str, predicted: str) -> bool:
    """Mirrors notebook_tinker.verify (the real grader). No binary special-case."""
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()
    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()


def verify_strict(stored_answer: str, predicted: str) -> bool:
    """Diagnostic-only: like reasoners.reasoning.compare_answer — exact on binary.

    Used to surface where the lenient float path inflates the headline number.
    """
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()
    if re.fullmatch(r"[01]+", stored_answer):
        return predicted.lower() == stored_answer.lower()
    try:
        return math.isclose(
            float(stored_answer), float(predicted), rel_tol=1e-2, abs_tol=1e-5
        )
    except Exception:
        return predicted.lower() == stored_answer.lower()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest val/tests/test_grading.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Format, lint, commit**

```bash
uv run --frozen ruff format val/grading.py val/tests/test_grading.py
uv run --frozen ruff check val/grading.py val/tests/test_grading.py --fix
git add val/grading.py val/tests/test_grading.py
git commit -m "feat(val): pinned grading copied from real scoring path"
```

---

## Task 2: Scoring loop with an injectable predictor

**Files:**
- Create: `val/scoring.py`
- Test: `val/tests/test_scoring.py`

`build_eval_prompt` must reproduce `generate_predictions` exactly: it appends the same boxed instruction to `problem.prompt`. Chat-template application lives in the real predictor (Task 7), so the predictor receives `user_content` strings and returns raw model text. `score` runs the predictor, extracts, and verifies — recording both lenient (`verify`) and strict (`verify_strict`) correctness.

- [ ] **Step 1: Write the failing test**

`val/tests/test_scoring.py`:
```python
from reasoners.store_types import Example, Problem
from val.scoring import build_eval_prompt, score

BOXED = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def _problem(pid: str, answer: str) -> Problem:
    return Problem(
        id=pid,
        category="cipher",
        examples=[Example("a", "b")],
        question="q",
        answer=answer,
        prompt=f"PROMPT-{pid}",
    )


def test_build_eval_prompt_appends_boxed_instruction():
    p = _problem("x1", "cat")
    assert build_eval_prompt(p) == "PROMPT-x1" + BOXED


def test_score_runs_predictor_and_verifies():
    problems = [_problem("x1", "cat"), _problem("x2", "10101010")]
    difficulty = {"x1": 1, "x2": 3}

    def predictor(user_contents):
        # Predictor receives the exact eval prompts in order.
        assert user_contents == [build_eval_prompt(p) for p in problems]
        return [r"reasoning... \boxed{cat}", r"\boxed{10101011}"]

    results = score(problems, predictor, difficulty=difficulty)

    assert [r.id for r in results] == ["x1", "x2"]
    assert results[0].correct is True
    assert results[0].difficulty == 1
    # x2: binary off-by-one -> lenient verify True, strict False (the gotcha).
    assert results[1].correct is True
    assert results[1].correct_strict is False
    assert results[1].predicted == "10101011"
    assert results[1].raw_text == r"\boxed{10101011}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest val/tests/test_scoring.py -v`
Expected: FAIL — no module named `val.scoring`.

- [ ] **Step 3: Write the implementation**

`val/scoring.py`:
```python
"""Score Problems through the real eval path with an injectable predictor."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from reasoners.store_types import Problem
from val.grading import extract_final_answer, verify, verify_strict

BOXED_INSTRUCTION = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

# A predictor maps eval-prompt strings -> raw model output strings, in order.
Predictor = Callable[[list[str]], list[str]]


@dataclass
class Result:
    id: str
    category: str
    difficulty: int
    answer: str
    predicted: str
    correct: bool
    correct_strict: bool
    raw_text: str


def build_eval_prompt(problem: Problem) -> str:
    """Reproduce notebook_tinker.generate_predictions' user_content exactly."""
    return problem.prompt + BOXED_INSTRUCTION


def score(
    problems: Sequence[Problem],
    predictor: Predictor,
    difficulty: dict[str, int] | None = None,
) -> list[Result]:
    difficulty = difficulty or {}
    prompts = [build_eval_prompt(p) for p in problems]
    raw_texts = predictor(prompts)
    if len(raw_texts) != len(problems):
        raise ValueError(
            f"predictor returned {len(raw_texts)} outputs for {len(problems)} problems"
        )

    results: list[Result] = []
    for problem, raw_text in zip(problems, raw_texts):
        predicted = extract_final_answer(raw_text)
        results.append(
            Result(
                id=problem.id,
                category=problem.category,
                difficulty=difficulty.get(problem.id, 0),
                answer=problem.answer,
                predicted=predicted,
                correct=verify(problem.answer, predicted),
                correct_strict=verify_strict(problem.answer, predicted),
                raw_text=raw_text,
            )
        )
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest val/tests/test_scoring.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Format, lint, commit**

```bash
uv run --frozen ruff format val/scoring.py val/tests/test_scoring.py
uv run --frozen ruff check val/scoring.py val/tests/test_scoring.py --fix
git add val/scoring.py val/tests/test_scoring.py
git commit -m "feat(val): scoring loop with injectable predictor"
```

---

## Task 3: Reporting by category × difficulty

**Files:**
- Create: `val/report.py`
- Test: `val/tests/test_report.py`

Report per `(category, difficulty)` and per-category rollups: count, lenient accuracy, strict accuracy. Difficulty buckets must be reported separately (roadmap: "report accuracy by difficulty, not one number").

- [ ] **Step 1: Write the failing test**

`val/tests/test_report.py`:
```python
import json

from val.report import aggregate, format_table, write_json
from val.scoring import Result


def _r(category, difficulty, correct, correct_strict):
    return Result(
        id="x",
        category=category,
        difficulty=difficulty,
        answer="a",
        predicted="a",
        correct=correct,
        correct_strict=correct_strict,
        raw_text="",
    )


def test_aggregate_groups_by_category_and_difficulty():
    results = [
        _r("cipher", 1, True, True),
        _r("cipher", 1, False, False),
        _r("cipher", 2, True, False),
    ]
    agg = aggregate(results)

    assert agg["cipher"]["overall"]["n"] == 3
    assert agg["cipher"]["overall"]["accuracy"] == 2 / 3
    assert agg["cipher"]["overall"]["accuracy_strict"] == 1 / 3
    assert agg["cipher"]["by_difficulty"][1]["n"] == 2
    assert agg["cipher"]["by_difficulty"][1]["accuracy"] == 0.5
    assert agg["cipher"]["by_difficulty"][2]["accuracy"] == 1.0


def test_format_table_mentions_category_and_counts():
    table = format_table(aggregate([_r("cipher", 1, True, True)]))
    assert "cipher" in table
    assert "100" in table  # 100.0% accuracy somewhere in the row


def test_write_json_roundtrips(tmp_path):
    agg = aggregate([_r("cipher", 1, True, True)])
    out = tmp_path / "report.json"
    write_json(agg, out)
    assert json.loads(out.read_text())["cipher"]["overall"]["n"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest val/tests/test_report.py -v`
Expected: FAIL — no module named `val.report`.

- [ ] **Step 3: Write the implementation**

`val/report.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest val/tests/test_report.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Format, lint, commit**

```bash
uv run --frozen ruff format val/report.py val/tests/test_report.py
uv run --frozen ruff check val/report.py val/tests/test_report.py --fix
git add val/report.py val/tests/test_report.py
git commit -m "feat(val): category x difficulty reporting"
```

---

## Task 4: Real snapshot-complement holdout loader

**Files:**
- Create: `val/real_holdout.py`
- Test: `val/tests/test_real_holdout.py`

The `04-08-16-14` snapshot index (`training/sft/04-08-16-14/logprobs/index.jsonl`) lists trained `problem_id`s, some with `-p0`/`-dN` duplicate suffixes. Stripping the suffix yields the set of *base* problem ids that were trained. The holdout = `problems.jsonl` ids minus that set. Per the roadmap, this holdout is **unbiased only for `numeral`, `gravity`, `unit_conversion`** (absent due to random downsampling); for other categories it is a pessimistic floor (absent due to solver failure) — so we tag each holdout problem `clean=True/False`.

- [ ] **Step 1: Write the failing test**

`val/tests/test_real_holdout.py`:
```python
from val.real_holdout import CLEAN_CATEGORIES, strip_dup_suffix, trained_base_ids


def test_strip_dup_suffix():
    assert strip_dup_suffix("00066667") == "00066667"
    assert strip_dup_suffix("00066667-p0") == "00066667"
    assert strip_dup_suffix("abc123-d11") == "abc123"


def test_trained_base_ids_reads_index(tmp_path):
    index = tmp_path / "index.jsonl"
    index.write_text(
        '{"problem_id": "aaa", "category": "numeral"}\n'
        '{"problem_id": "aaa-d0", "category": "numeral"}\n'
        '{"problem_id": "bbb-p0", "category": "cipher"}\n'
    )
    assert trained_base_ids(index) == {"aaa", "bbb"}


def test_clean_categories_are_the_downsampled_three():
    assert CLEAN_CATEGORIES == {"numeral", "gravity", "unit_conversion"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest val/tests/test_real_holdout.py -v`
Expected: FAIL — no module named `val.real_holdout`.

- [ ] **Step 3: Write the implementation**

`val/real_holdout.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest val/tests/test_real_holdout.py -v`
Expected: PASS (3 tests). (`holdout_problems` reads real repo files; it is exercised by the smoke run in Task 8, not unit-tested here.)

- [ ] **Step 5: Format, lint, commit**

```bash
uv run --frozen ruff format val/real_holdout.py val/tests/test_real_holdout.py
uv run --frozen ruff check val/real_holdout.py val/tests/test_real_holdout.py --fix
git add val/real_holdout.py val/tests/test_real_holdout.py
git commit -m "feat(val): snapshot-complement real holdout loader"
```

---

## Task 5: Permanent holdout registry (rule reservation)

**Files:**
- Create: `val/holdout_registry.py`
- Test: `val/tests/test_holdout_registry.py`

Synthetic val rules must **never** leak into training generation (roadmap #4). The registry persists reserved rule signatures (per category) to `val/holdout_rules.json` so future data-gen can exclude them.

- [ ] **Step 1: Write the failing test**

`val/tests/test_holdout_registry.py`:
```python
from val.holdout_registry import HoldoutRegistry


def test_reserve_and_query(tmp_path):
    path = tmp_path / "holdout_rules.json"
    reg = HoldoutRegistry(path)

    assert reg.is_reserved("cipher", "SIG1") is False
    assert reg.reserve("cipher", "SIG1") is True  # newly reserved
    assert reg.is_reserved("cipher", "SIG1") is True
    assert reg.reserve("cipher", "SIG1") is False  # already reserved


def test_persists_across_instances(tmp_path):
    path = tmp_path / "holdout_rules.json"
    HoldoutRegistry(path).reserve("bit_manipulation", "SIGX")

    reg2 = HoldoutRegistry(path)
    assert reg2.is_reserved("bit_manipulation", "SIGX") is True
    assert reg2.is_reserved("cipher", "SIGX") is False  # category-scoped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest val/tests/test_holdout_registry.py -v`
Expected: FAIL — no module named `val.holdout_registry`.

- [ ] **Step 3: Write the implementation**

`val/holdout_registry.py`:
```python
"""Persistent registry of rule signatures reserved for validation only."""

import json
from pathlib import Path


class HoldoutRegistry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, set[str]] = {}
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self._data = {cat: set(sigs) for cat, sigs in raw.items()}

    def is_reserved(self, category: str, signature: str) -> bool:
        return signature in self._data.get(category, set())

    def reserve(self, category: str, signature: str) -> bool:
        """Reserve a signature. Returns True if newly reserved, False if already."""
        if self.is_reserved(category, signature):
            return False
        self._data.setdefault(category, set()).add(signature)
        self._flush()
        return True

    def _flush(self) -> None:
        serializable = {cat: sorted(sigs) for cat, sigs in self._data.items()}
        self.path.write_text(json.dumps(serializable, indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest val/tests/test_holdout_registry.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Format, lint, commit**

```bash
uv run --frozen ruff format val/holdout_registry.py val/tests/test_holdout_registry.py
uv run --frozen ruff check val/holdout_registry.py val/tests/test_holdout_registry.py --fix
git add val/holdout_registry.py val/tests/test_holdout_registry.py
git commit -m "feat(val): permanent holdout rule registry"
```

---

## Task 6: Reference synthetic generator — `cipher` (new substitution rule)

**Files:**
- Create: `val/generators/__init__.py`
- Create: `val/generators/cipher.py`
- Test: `val/tests/test_cipher_generator.py`

Defines the `Generator` protocol the follow-on generators will implement, plus a working cipher generator. The cipher prompt template is copied verbatim from a real problem (`problems/00189f6a.jsonl`): a decryption task where examples are `ciphertext -> plaintext` and the query is ciphertext. A *new rule* = a fresh random permutation of `a-z` (the substitution map). `rule_signature` = the 26-char cipher alphabet; reserved in the registry so it never enters training. The generated `Problem.answer` is the plaintext of the query, correct by construction; difficulty = number of example sentences.

- [ ] **Step 1: Write the failing test**

`val/tests/test_cipher_generator.py`:
```python
from val.generators import GENERATORS
from val.generators.cipher import generate, rule_signature


def test_registered():
    assert "cipher" in GENERATORS


def test_deterministic_given_seed():
    a = generate(seed=7, difficulty=3)
    b = generate(seed=7, difficulty=3)
    assert a.prompt == b.prompt
    assert a.answer == b.answer


def test_different_seeds_differ():
    assert generate(seed=1, difficulty=3).prompt != generate(seed=2, difficulty=3).prompt


def test_examples_count_matches_difficulty():
    p = generate(seed=7, difficulty=4)
    assert len(p.examples) == 4
    assert p.category == "cipher"


def test_answer_is_recoverable_from_rule():
    # The query ciphertext decrypts (per the examples' substitution) to the answer.
    p = generate(seed=11, difficulty=5)
    sig = rule_signature(seed=11)  # plaintext-letter -> ciphertext-letter alphabet
    decrypt = {sig[i]: chr(ord("a") + i) for i in range(26)}
    recovered = "".join(decrypt.get(ch, ch) for ch in p.question)
    assert recovered == p.answer


def test_prompt_uses_real_template_markers():
    p = generate(seed=7, difficulty=3)
    assert p.prompt.startswith(
        "In Alice's Wonderland, secret encryption rules are used on text."
    )
    assert "Now, decrypt the following text:" in p.prompt
    assert " -> " in p.prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest val/tests/test_cipher_generator.py -v`
Expected: FAIL — no module named `val.generators`.

- [ ] **Step 3: Write the implementation**

`val/generators/__init__.py`:
```python
"""Per-category synthetic new-rule generators.

A generator module exposes and self-registers:
    generate(seed: int, difficulty: int) -> reasoners.store_types.Problem
    rule_signature(seed: int) -> str   # stable id of the rule, for holdout reservation
"""

from collections.abc import Callable
from dataclasses import dataclass

from reasoners.store_types import Problem


@dataclass
class GeneratorSpec:
    generate: Callable[[int, int], Problem]
    rule_signature: Callable[[int], str]


GENERATORS: dict[str, GeneratorSpec] = {}


def register(
    category: str,
    generate: Callable[[int, int], Problem],
    rule_signature: Callable[[int], str],
) -> None:
    GENERATORS[category] = GeneratorSpec(generate, rule_signature)


# Import generator modules so they self-register on `import val.generators`.
from val.generators import cipher  # noqa: E402,F401
```

`val/generators/cipher.py`:
```python
"""Reference generator: a brand-new substitution-cipher decryption problem."""

import random

from reasoners.store_types import Example, Problem
from val.generators import register

# Themed vocabulary mirroring the real cipher problems' Wonderland distribution.
_WORDS = [
    "queen", "dragon", "castle", "secret", "near", "valley", "discovers",
    "dreams", "inside", "student", "creates", "magical", "door", "golden",
    "follows", "princess", "reads", "mysterious", "cat", "imagines", "book",
    "wizard", "the", "guards", "hidden", "garden", "river", "mountain",
    "whispers", "ancient", "key", "opens", "silver", "gate", "forest",
]

_PROMPT_HEADER = "In Alice's Wonderland, secret encryption rules are used on text."


def _alphabet(seed: int) -> str:
    """Plaintext-letter (a..z) -> ciphertext-letter mapping, as a 26-char string."""
    letters = list("abcdefghijklmnopqrstuvwxyz")
    random.Random(seed).shuffle(letters)
    return "".join(letters)


def rule_signature(seed: int) -> str:
    return _alphabet(seed)


def _encrypt(text: str, alphabet: str) -> str:
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(alphabet[ord(ch) - ord("a")])
        else:
            out.append(ch)
    return "".join(out)


def _sentence(rng: random.Random) -> str:
    n = rng.randint(3, 5)
    return " ".join(rng.choice(_WORDS) for _ in range(n))


def generate(seed: int, difficulty: int) -> Problem:
    rng = random.Random(seed)
    alphabet = _alphabet(seed)

    plain_examples = [_sentence(rng) for _ in range(difficulty)]
    plain_query = _sentence(rng)

    example_lines = []
    examples: list[Example] = []
    for plain in plain_examples:
        cipher = _encrypt(plain, alphabet)
        example_lines.append(f"{cipher} -> {plain}")
        examples.append(Example(input_value=cipher, output_value=plain))

    cipher_query = _encrypt(plain_query, alphabet)
    prompt = (
        f"{_PROMPT_HEADER}\nHere are some examples:\n"
        + "\n".join(example_lines)
        + f"\nNow, decrypt the following text: {cipher_query}"
    )

    return Problem(
        id=f"val-cipher-{seed}",
        category="cipher",
        examples=examples,
        question=cipher_query,
        answer=plain_query,
        prompt=prompt,
    )


register("cipher", generate, rule_signature)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest val/tests/test_cipher_generator.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Format, lint, commit**

```bash
uv run --frozen ruff format val/generators/__init__.py val/generators/cipher.py val/tests/test_cipher_generator.py
uv run --frozen ruff check val/generators/__init__.py val/generators/cipher.py val/tests/test_cipher_generator.py --fix
git add val/generators/__init__.py val/generators/cipher.py val/tests/test_cipher_generator.py
git commit -m "feat(val): cipher reference generator + generator protocol"
```

---

## Task 6b: Second generator — `bit_manipulation` (one hard category, clean baseline)

**Files:**
- Create: `val/generators/bit_manipulation.py`
- Modify: `val/generators/__init__.py` (auto-import the new module)
- Test: `val/tests/test_bit_generator.py`

**Why this one and only this one (vs cryptarithm/equation):** the clean baselines we'd otherwise get (numeral/gravity/unit_conversion real holdout + cipher) are all categories that are already ~100% solved — near-zero headroom. The first GPU run would measure four saturated categories and tell us little about *where the ~0.13 gap is*. bit_manipulation is the one **hard** category whose rules are **trivially forward-generable deterministically** (affine-mod-256, XOR-mask, rotation, bit-permutation), and it couples directly to roadmap lever #1 (extending the bit solver grammar). cryptarithm/`_guess` generators need rule-family mining + deducibility control and stay in the follow-on.

Prompt template is copied verbatim from real problems (`problems/0031df9c.jsonl`): 8-bit binary, header listing bit ops, `input -> output` example lines, `Now, determine the output for: <query>`. The answer is computed by applying the chosen rule, correct by construction. `rule_signature` is a canonical string of the rule (type + params), reserved so it never enters training.

- [ ] **Step 1: Write the failing test**

`val/tests/test_bit_generator.py`:
```python
from val.generators import GENERATORS
from val.generators.bit_manipulation import build_rule, generate, rule_signature


def test_registered():
    assert "bit_manipulation" in GENERATORS


def test_deterministic_given_seed():
    a = generate(seed=3, difficulty=8)
    b = generate(seed=3, difficulty=8)
    assert a.prompt == b.prompt and a.answer == b.answer


def test_different_seeds_differ():
    # Across a small range, at least one pair of rules differs.
    sigs = {rule_signature(s) for s in range(8)}
    assert len(sigs) > 1


def test_examples_count_and_category():
    p = generate(seed=3, difficulty=10)
    assert len(p.examples) == 10
    assert p.category == "bit_manipulation"


def test_inputs_and_outputs_are_8bit_binary():
    p = generate(seed=5, difficulty=8)
    for ex in p.examples:
        assert len(ex.input_value) == 8 and set(ex.input_value) <= {"0", "1"}
        assert len(ex.output_value) == 8 and set(ex.output_value) <= {"0", "1"}
    assert len(p.answer) == 8 and set(p.answer) <= {"0", "1"}


def test_answer_recoverable_from_rule():
    p = generate(seed=5, difficulty=8)
    _sig, apply = build_rule(seed=5)
    expected = format(apply(int(p.question, 2)), "08b")
    assert expected == p.answer


def test_examples_satisfy_the_rule():
    p = generate(seed=9, difficulty=8)
    _sig, apply = build_rule(seed=9)
    for ex in p.examples:
        assert format(apply(int(ex.input_value, 2)), "08b") == ex.output_value


def test_prompt_uses_real_template_markers():
    p = generate(seed=3, difficulty=8)
    assert p.prompt.startswith(
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit"
    )
    assert "Here are some examples of input -> output:" in p.prompt
    assert "Now, determine the output for:" in p.prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest val/tests/test_bit_generator.py -v`
Expected: FAIL — no module named `val.generators.bit_manipulation`.

- [ ] **Step 3: Write the implementation**

`val/generators/bit_manipulation.py`:
```python
"""Second generator: brand-new 8-bit transformation rules.

Rule families (all deterministic, answer correct by construction):
  affine : out = (a*x + b) % 256        (a odd)
  xor    : out = x ^ mask
  rotl   : out = rotate-left(x, k)       (8-bit)
  perm   : out bit j = in bit perm[j]    (MSB-first indexing)
"""

import random
from collections.abc import Callable

from reasoners.store_types import Example, Problem
from val.generators import register

_HEADER = (
    "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit "
    "binary numbers. The transformation involves operations like bit shifts, "
    "rotations, XOR, AND, OR, NOT, and possibly majority or choice functions."
)

_RULE_TYPES = ["affine", "xor", "rotl", "perm"]


def build_rule(seed: int) -> tuple[str, Callable[[int], int]]:
    """Return (canonical_signature, apply) for the rule selected by *seed*."""
    rng = random.Random(seed)
    rule_type = rng.choice(_RULE_TYPES)

    if rule_type == "affine":
        a = rng.choice([3, 5, 7, 9, 11, 13, 15])
        b = rng.randint(1, 255)
        signature = f"affine:a={a},b={b}"

        def apply(x: int) -> int:
            return (a * x + b) & 0xFF

    elif rule_type == "xor":
        mask = rng.randint(1, 255)
        signature = f"xor:{mask}"

        def apply(x: int) -> int:
            return x ^ mask

    elif rule_type == "rotl":
        k = rng.randint(1, 7)
        signature = f"rotl:{k}"

        def apply(x: int) -> int:
            return ((x << k) | (x >> (8 - k))) & 0xFF

    else:  # perm
        perm = list(range(8))
        rng.shuffle(perm)
        signature = "perm:" + ",".join(str(i) for i in perm)

        def apply(x: int) -> int:
            in_bits = [(x >> (7 - i)) & 1 for i in range(8)]
            out_bits = [in_bits[perm[j]] for j in range(8)]
            out = 0
            for j in range(8):
                out |= out_bits[j] << (7 - j)
            return out

    return signature, apply


def rule_signature(seed: int) -> str:
    return build_rule(seed)[0]


def _distinct_inputs(seed: int, count: int) -> list[int]:
    rng = random.Random(seed * 1000 + 1)  # separate stream from the rule
    seen: list[int] = []
    while len(seen) < count:
        v = rng.randint(0, 255)
        if v not in seen:
            seen.append(v)
    return seen


def generate(seed: int, difficulty: int) -> Problem:
    _signature, apply = build_rule(seed)
    values = _distinct_inputs(seed, difficulty + 1)
    example_ints, query_int = values[:difficulty], values[difficulty]

    example_lines: list[str] = []
    examples: list[Example] = []
    for x in example_ints:
        x_bits = format(x, "08b")
        y_bits = format(apply(x), "08b")
        example_lines.append(f"{x_bits} -> {y_bits}")
        examples.append(Example(input_value=x_bits, output_value=y_bits))

    query_bits = format(query_int, "08b")
    answer_bits = format(apply(query_int), "08b")
    prompt = (
        f"{_HEADER}\n\nHere are some examples of input -> output:\n"
        + "\n".join(example_lines)
        + f"\n\nNow, determine the output for: {query_bits}"
    )

    return Problem(
        id=f"val-bit_manipulation-{seed}",
        category="bit_manipulation",
        examples=examples,
        question=query_bits,
        answer=answer_bits,
        prompt=prompt,
    )


register("bit_manipulation", generate, rule_signature)
```

- [ ] **Step 4: Auto-import the new module so it self-registers**

In `val/generators/__init__.py`, change the final import line:
```python
from val.generators import cipher  # noqa: E402,F401
```
to:
```python
from val.generators import bit_manipulation, cipher  # noqa: E402,F401
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest val/tests/test_bit_generator.py val/tests/test_cipher_generator.py -v`
Expected: PASS (both generator suites).

- [ ] **Step 6: Format, lint, commit**

```bash
uv run --frozen ruff format val/generators/bit_manipulation.py val/generators/__init__.py val/tests/test_bit_generator.py
uv run --frozen ruff check val/generators/bit_manipulation.py val/generators/__init__.py val/tests/test_bit_generator.py --fix
git add val/generators/bit_manipulation.py val/generators/__init__.py val/tests/test_bit_generator.py
git commit -m "feat(val): bit_manipulation generator (hard-category baseline)"
```

---

## Task 7: Real vLLM predictor + GPU runner (manual run)

**Files:**
- Create: `val/run_vllm.py`

This is the real predictor; it imports vLLM and loads the 30B base + adapter, so it runs only on a GPU box (Kaggle/Tinker), not in local tests. It reproduces `generate_predictions`' chat-template + sampling settings (greedy: `temperature=0.0`, `top_p=1.0`, `max_lora_rank=32`).

- [ ] **Step 1: Write the runner (no unit test — GPU-only path)**

`val/run_vllm.py`:
```python
"""Real-path predictor for the validation harness (GPU/Kaggle only).

Mirrors notebook_tinker.generate_predictions: chat template with thinking,
greedy decoding, rank-32 LoRA adapter. Not imported by local tests.

Usage (on a GPU box):
    uv run python -m val.run_vllm --model <BASE_MODEL_PATH> --adapter <ADAPTER_DIR>
"""

import argparse
from collections.abc import Callable
from pathlib import Path

from val.generators import GENERATORS
from val.holdout_registry import HoldoutRegistry
from val.real_holdout import holdout_problems
from val.report import aggregate, format_table, write_json
from val.scoring import score


def make_vllm_predictor(
    model_path: str,
    adapter_path: str,
    max_lora_rank: int = 32,
    max_tokens: int = 7680,
    max_model_len: int = 8192,
) -> Callable[[list[str]], list[str]]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="auto",
        max_model_len=max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)

    def predict(user_contents: list[str]) -> list[str]:
        prompts = []
        for content in user_contents:
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            except Exception:
                prompt = content
            prompts.append(prompt)
        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=LoRARequest("adapter", 1, adapter_path),
        )
        return [o.outputs[0].text for o in outputs]

    return predict


def build_synthetic_valset(
    per_category: int, difficulty: int, registry: HoldoutRegistry
):
    """Generate fresh new-rule problems from every registered generator and
    reserve each rule's signature so it can never enter training generation."""
    problems = []
    for category, spec in sorted(GENERATORS.items()):
        for seed in range(per_category):
            problems.append(spec.generate(seed, difficulty))
            registry.reserve(category, spec.rule_signature(seed))
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--per-category", type=int, default=50)
    parser.add_argument("--difficulty", type=int, default=6)
    parser.add_argument("--holdout", default="val/holdout_rules.json")
    parser.add_argument("--out-prefix", default="val_report")
    args = parser.parse_args()

    predictor = make_vllm_predictor(args.model, args.adapter)

    # 1) Real snapshot-complement holdout (clean only for the 3 downsampled cats).
    held = holdout_problems()
    real_problems = [hp.problem for hp in held]
    real_diff = {p.id: len(p.examples) for p in real_problems}
    real_report = aggregate(score(real_problems, predictor, difficulty=real_diff))

    # 2) Synthetic new-rule holdout (the only clean signal on the hard categories).
    registry = HoldoutRegistry(Path(args.holdout))
    synth_problems = build_synthetic_valset(
        args.per_category, args.difficulty, registry
    )
    synth_diff = {p.id: len(p.examples) for p in synth_problems}
    synth_report = aggregate(score(synth_problems, predictor, difficulty=synth_diff))

    print("=== REAL snapshot-complement holdout ===")
    print(format_table(real_report))
    print("\n=== SYNTHETIC new-rule holdout ===")
    print(format_table(synth_report))

    write_json(real_report, f"{args.out_prefix}_real.json")
    write_json(synth_report, f"{args.out_prefix}_synth.json")
    print(f"\nWrote {args.out_prefix}_real.json and {args.out_prefix}_synth.json")
    print("NOTE: REAL holdout is unbiased only for numeral/gravity/unit_conversion;")
    print("other categories there are a pessimistic floor (solver-failure bias).")
    print("SYNTHETIC holdout gives the clean baseline on cipher + bit_manipulation.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it imports without the GPU deps firing**

Run: `uv run python -c "import ast; ast.parse(open('val/run_vllm.py').read()); print('parse-ok')"`
Expected: `parse-ok` (vLLM is imported lazily inside `make_vllm_predictor`, so module import is safe).

- [ ] **Step 3: Format, lint, commit**

```bash
uv run --frozen ruff format val/run_vllm.py
uv run --frozen ruff check val/run_vllm.py --fix
git add val/run_vllm.py
git commit -m "feat(val): real vLLM predictor + GPU holdout runner"
```

---

## Task 8: End-to-end local smoke + README

**Files:**
- Create: `val/run_local_demo.py`
- Create: `val/README.md`
- Test: `val/tests/test_local_demo.py`

A stub predictor proves the full pipeline (generate synthetic cipher problems → reserve rules → score → report) runs without a GPU. The stub "cheats" by decrypting via the known rule, so it should score ~100% — confirming the loop, prompts, extraction, and verifier are wired correctly.

- [ ] **Step 1: Write the failing test**

`val/tests/test_local_demo.py`:
```python
from val.run_local_demo import run_demo


def test_demo_runs_and_scores(tmp_path):
    report = run_demo(
        n=5,
        difficulty=4,
        holdout_path=tmp_path / "holdout_rules.json",
        report_path=tmp_path / "demo_report.json",
    )
    assert report["cipher"]["overall"]["n"] == 5
    # Oracle returns ground-truth answers -> full accuracy.
    assert report["cipher"]["overall"]["accuracy"] == 1.0
    assert (tmp_path / "demo_report.json").exists()
    assert (tmp_path / "holdout_rules.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest val/tests/test_local_demo.py -v`
Expected: FAIL — no module named `val.run_local_demo`.

- [ ] **Step 3: Write the implementation**

`val/run_local_demo.py`:
```python
"""Local, GPU-free smoke test of the full harness with an oracle predictor."""

from pathlib import Path

from val.generators.cipher import generate, rule_signature
from val.holdout_registry import HoldoutRegistry
from val.report import aggregate, format_table, write_json
from val.scoring import BOXED_INSTRUCTION, score


def _oracle_predictor_factory(problems):
    """A predictor that returns each problem's known answer.

    run_demo holds the generated Problems, so the oracle looks the answer up by
    prompt rather than re-deriving the rule. This deterministically exercises the
    full scoring path (prompt routing -> extraction -> verify -> report) without
    depending on alphabet coverage. The real model signal comes from Task 7.
    """
    by_prompt = {p.prompt + BOXED_INSTRUCTION: p for p in problems}

    def predict(user_contents):
        return [
            f"reasoning \\boxed{{{by_prompt[content].answer}}}"
            for content in user_contents
        ]

    return predict


def run_demo(
    n: int = 5,
    difficulty: int = 4,
    holdout_path: Path = Path("val/holdout_rules.json"),
    report_path: Path = Path("val/demo_report.json"),
) -> dict:
    registry = HoldoutRegistry(holdout_path)
    problems = []
    for seed in range(n):
        problems.append(generate(seed=seed, difficulty=difficulty))
        registry.reserve("cipher", rule_signature(seed))

    difficulty_map = {p.id: len(p.examples) for p in problems}
    predictor = _oracle_predictor_factory(problems)
    results = score(problems, predictor, difficulty=difficulty_map)

    report = aggregate(results)
    write_json(report, report_path)
    return report


if __name__ == "__main__":
    report = run_demo()
    print(format_table(report))
```

Note: this is a wiring smoke test only — the oracle returns ground-truth answers so accuracy is deterministically 1.0. The real model signal comes from Task 7.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest val/tests/test_local_demo.py -v`
Expected: PASS.

- [ ] **Step 5: Write the README**

`val/README.md`:
```markdown
# Validation harness (roadmap P0)

Measures per-category, per-difficulty generalization accuracy of a trained
adapter on **rules it never saw**, scored through the real greedy/vLLM/`verify()`
path (`notebook_tinker.generate_predictions` semantics).

## Local (no GPU)
- Unit tests: `uv run pytest val/tests -v`
- Pipeline smoke (oracle predictor): `uv run python -m val.run_local_demo`

## Real scoring (GPU / Kaggle)
`uv run python -m val.run_vllm --model <BASE_MODEL_PATH> --adapter <ADAPTER_DIR>`

Scores **two** holdouts and writes `val_report_real.json` + `val_report_synth.json`:
- **Real** snapshot-complement holdout (`val/real_holdout.py`). Read it as:
  **unbiased baseline for numeral / gravity / unit_conversion only**; other
  categories are a *pessimistic floor* (absent due to solver failure, not random
  downsampling).
- **Synthetic** new-rule holdout from `val/generators` — the clean baseline on
  the categories the real holdout can't cover. Today: `cipher` (easy) and
  `bit_manipulation` (one hard category). Rules are auto-reserved (see below).

## Two accuracies
- `accuracy` — the real grader (`verify`, float-lenient; binary strings within 1%
  count as equal).
- `accuracy_strict` — diagnostic (`verify_strict`, exact on binary). A gap means
  the lenient grader is inflating a category's headline number.

## Holdout registry
`val/holdout_rules.json` reserves synthetic val rule signatures. **Future
training data-gen MUST exclude reserved signatures** so val never leaks.

## Follow-on
Synthetic generators for the other 7 categories (roadmap lever #2) plug into
`val/generators` via `register(category, generate, rule_signature)`. cipher and
bit_manipulation are the references; cryptarithm/`_guess`/equation come next.
```

- [ ] **Step 6: Format, lint, commit**

```bash
uv run --frozen ruff format val/run_local_demo.py val/tests/test_local_demo.py
uv run --frozen ruff check val/run_local_demo.py val/tests/test_local_demo.py --fix
git add val/run_local_demo.py val/tests/test_local_demo.py val/README.md
git commit -m "feat(val): end-to-end local smoke + README"
```

- [ ] **Step 7: Full suite green**

Run: `uv run pytest val/tests -v`
Expected: all tests PASS.

---

## Self-review

- **Spec coverage (P0):** real greedy/vLLM/`verify()` scoring (Tasks 2,7) ✓; per-category by-difficulty reporting (Task 3) ✓; synthetic new-rule generators → unbiased baseline path, one easy + one hard category (Tasks 6, 6b) scored on the real model (Task 7 `build_synthetic_valset`) ✓; real downsample-holdout cross-check tagged clean vs floor (Task 4) ✓; permanent holdout list with auto-reservation (Tasks 5, 7) ✓; memorization-ceiling/A-B runs are downstream uses of this harness, not new code. The other 7 generators are explicitly deferred (follow-on) — noted, not silently dropped.
- **Type consistency:** `Result` fields (`correct`, `correct_strict`, `difficulty`, …) defined in Task 2 are used identically in Tasks 3, 8. The `register(category, generate, rule_signature)` / `GeneratorSpec(generate, rule_signature)` protocol (Task 6) is used identically by cipher (Task 6), bit_manipulation (Task 6b), and consumed via `spec.generate`/`spec.rule_signature` in Task 7. `generate(seed, difficulty)` / `rule_signature(seed)` signatures match across Tasks 6, 6b, 7, 8. `holdout_problems()`/`HoldoutProblem.problem` match across Tasks 4, 7. `BOXED_INSTRUCTION` exported in Task 2, reused in Task 8. `HoldoutRegistry.reserve` (Task 5) reused in Task 7.
- **No placeholders:** every code step is complete and runnable.

## Open items to confirm during/after execution (from the roadmap)

- **Which grader the competition actually uses** — `verify` (this harness's primary) vs a strict metric. We report both; confirm on the competition page.
- **Eval wall-clock / token limit** — bounds CoT length; confirm before scaling self-verifying traces.
- After the first GPU run: record per-category baselines and the lenient-vs-strict gap; that sizes where the ~0.13 points are and is the input to roadmap levers #1/#2/#13.

## Execution handoff

Two execution options — see the writing-plans skill's handoff.
1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks.
2. **Inline Execution** — execute tasks in this session with checkpoints.
