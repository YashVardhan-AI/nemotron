"""Create synthetic training corpus with reasoning from reasoning/*.txt files.

The completion for each entry is:
    (reasoning text)</think>\\boxed{(answer)}<|im_end|>

The opening <think>\\n is already part of the prompt (from the chat template),
so the reasoning text flows directly after it.

Outputs:
- corpus.jsonl          - Index with metadata per entry
- corpus/<problem_id>/synthetic.jsonl  - Segment files with interleaved masked/unmasked

Usage:
    uv run corpus.py
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

TRAIN_CSV = Path(__file__).parent / "train.csv"
AUGMENTATIONS_DIR = Path(__file__).parent / "augmentations"
PROBLEMS_INDEX = Path(__file__).parent / "problems.jsonl"
REASONING_DIR = Path(__file__).parent / "reasoning"
CORPUS_DIR = Path(__file__).parent / "corpus"
CORPUS_INDEX = Path(__file__).parent / "corpus.jsonl"
TOKENIZER_PATH = Path(__file__).parent / "tokenizer.json"

# Must match metric_reference.py / query.py
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

TOKEN_LIMIT = 8192

# --- cryptarithm_deduce forward-gen (Phase 5 of the #1-lever plan) ------------
# Add N distinct VERIFIED cryptarithm_deduce traces (the ~x12 exact-copy dup of
# ~54 real traces is already gone, removed in f42e3fbcf6 -- so this ADDS rows).
# Seeds start at a high offset disjoint from the val generator's small seeds, and
# any val-reserved rule_signature is skipped, so no validation rule can leak into
# training. Phase 6 sweeps CRYPT_N (size-matched ~600 vs scale 3-8k) and
# CRYPT_STYLE (deduce | propagate | mixed). Set CRYPT_N = 0 to disable.
CRYPT_N = 600
CRYPT_STYLE = "deduce"
CRYPT_DIFFICULTY = 4
CRYPT_SEED_OFFSET = 1_000_000  # disjoint from val seeds (0..few-thousand)
HOLDOUT_RULES = Path(__file__).parent / "holdout_rules.json"
# The real cryptarithm_deduce reasoning files (reasoning/*.txt) are concat-only:
# 658/659 default to concatenation and NONE show arithmetic, so for arithmetic
# problems they pair concat reasoning with an arithmetic answer (an inconsistent
# signal -- the root cause of the ~6% baseline). When forward-gen is on, REPLACE
# them with the verified forward-gen traces rather than mixing both.
CRYPT_REPLACE_REAL = True


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def tokenize_prompt(
    prompt_text: str,
    chat_tokenizer,  # transformers AutoTokenizer (untyped: dep imported lazily)
    *,
    suffix: str = PROMPT_SUFFIX,
) -> list[int]:
    """Tokenize a problem prompt using the chat template, matching query.py."""
    messages = [{"role": "user", "content": prompt_text + suffix}]
    return chat_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )


@dataclass
class CorpusEntry:
    problem_id: str
    category: str
    tokens: list[int]
    mask: list[int]
    masked_token_count: int
    unmasked_token_count: int
    answer: str
    included: bool = False

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    def to_index_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "segment": "synthetic.jsonl",
            "category": self.category,
            "masked_token_count": self.masked_token_count,
            "unmasked_token_count": self.unmasked_token_count,
            "token_count": self.token_count,
            "answer": self.answer,
            "included": self.included,
        }


def build_segments(
    tokens: list[int],
    mask: list[int],
) -> list[dict]:
    """Build segment list from tokens and mask."""
    if not tokens:
        return []

    segments: list[dict] = []
    seg_start = 0
    current_type = "unmasked" if mask[0] == 1 else "masked"

    for i in range(1, len(tokens)):
        token_type = "unmasked" if mask[i] == 1 else "masked"
        if token_type != current_type:
            segments.append(
                {
                    "type": current_type,
                    "pos": seg_start,
                    "tokens": tokens[seg_start:i],
                }
            )
            seg_start = i
            current_type = token_type

    segments.append(
        {
            "type": current_type,
            "pos": seg_start,
            "tokens": tokens[seg_start:],
        }
    )

    return segments


def _crypt_renderer(style: str, seed: int):
    from reasoners.cryptarithm_trace import (
        reasoning_cryptarithm_arith,
        reasoning_cryptarithm_propagate,
    )

    if style == "deduce":
        return reasoning_cryptarithm_arith
    if style == "propagate":
        return reasoning_cryptarithm_propagate
    if style == "mixed":
        return (
            reasoning_cryptarithm_arith
            if seed % 2 == 0
            else reasoning_cryptarithm_propagate
        )
    raise ValueError(f"unknown cryptarithm trace style: {style!r}")


def build_cryptarithm_examples(
    n: int,
    style: str,
    holdout: set[str],
    *,
    difficulty: int = CRYPT_DIFFICULTY,
    seed_offset: int = CRYPT_SEED_OFFSET,
) -> list[tuple[str, str, str, str]]:
    """Tokenizer-free generator of N distinct verified cryptarithm_deduce rows.

    Returns (problem_id, prompt_text, completion_text, answer) tuples. Each is a
    UNIQUELY-deducible instance (solver-filtered) with a genuine-deduction CoT.
    Skips any seed whose rule_signature is in *holdout* (no val leakage); seeds
    start at *seed_offset*, disjoint from the val generator's small seeds.
    """
    from reasoners.cryptarithm_trace import make_trace_problem
    from val.generators.cryptarithm import rule_signature

    out: list[tuple[str, str, str, str]] = []
    seed = seed_offset
    guard = seed_offset + 100 * (n + 1)  # generous; ~no skips expected
    while len(out) < n and seed < guard:
        if rule_signature(seed) in holdout:
            seed += 1
            continue
        problem, answer = make_trace_problem(seed, difficulty)
        reasoning = _crypt_renderer(style, seed)(problem, answer)
        if reasoning is None:  # solver disagreed (should not happen); skip
            seed += 1
            continue
        completion = f"{reasoning}\n</think>\n\\boxed{{{answer}}}<|im_end|>"
        out.append((f"cryptarithm-{style}-{seed}", problem.prompt, completion, answer))
        seed += 1
    return out


def build_cryptarithm_rows(
    n: int,
    style: str,
    holdout: set[str],
    tokenizer,
    chat_tokenizer,
    *,
    difficulty: int = CRYPT_DIFFICULTY,
    write_segments_to: Path | None = None,
) -> list[CorpusEntry]:
    """Tokenize build_cryptarithm_examples into CorpusEntry rows (prompt masked,
    completion unmasked), matching the real-problem path. Writes per-entry segment
    files under *write_segments_to* when given."""
    rows: list[CorpusEntry] = []
    for pid, prompt_text, completion_text, answer in build_cryptarithm_examples(
        n, style, holdout, difficulty=difficulty
    ):
        completion_ids = tokenizer.encode(completion_text, add_special_tokens=False).ids
        prompt_ids = tokenize_prompt(prompt_text, chat_tokenizer)
        all_tokens = prompt_ids + completion_ids
        mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
        if len(all_tokens) > TOKEN_LIMIT:
            all_tokens = all_tokens[:TOKEN_LIMIT]
            mask = mask[:TOKEN_LIMIT]
        unmasked = sum(mask)
        entry = CorpusEntry(
            problem_id=pid,
            category="cryptarithm_deduce",
            tokens=all_tokens,
            mask=mask,
            masked_token_count=len(mask) - unmasked,
            unmasked_token_count=unmasked,
            answer=answer,
            included=True,
        )
        if write_segments_to is not None:
            problem_dir = write_segments_to / pid
            problem_dir.mkdir(parents=True, exist_ok=True)
            with open(problem_dir / "synthetic.jsonl", "w") as f:
                for seg in build_segments(all_tokens, mask):
                    json.dump(seg, f)
                    f.write("\n")
        rows.append(entry)
    return rows


def _load_cryptarithm_holdout() -> set[str]:
    if not HOLDOUT_RULES.exists():
        return set()
    return set(json.loads(HOLDOUT_RULES.read_text()).get("cryptarithm_deduce", []))


def main() -> None:
    if not PROBLEMS_INDEX.exists():
        print(f"No {PROBLEMS_INDEX} found. Run problems.py first.")
        return

    # Load tokenizers (heavy deps imported lazily so importing corpus stays light)
    from tokenizers import Tokenizer  # type: ignore[import-untyped]
    from transformers import AutoTokenizer  # type: ignore[import-untyped]

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    chat_tokenizer = AutoTokenizer.from_pretrained(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", trust_remote_code=True
    )

    # Load problem prompts from train.csv
    prompts: dict[str, str] = {}
    answers: dict[str, str] = {}
    with open(TRAIN_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["id"]
            prompts[pid] = row["prompt"]
            answers[pid] = row["answer"]

    # Load problem categories
    problem_cats: dict[str, str] = {}
    for prob_raw in load_jsonl(PROBLEMS_INDEX):
        problem_cats[prob_raw["id"]] = prob_raw["category"]

    # Clean and recreate corpus directory
    if CORPUS_DIR.exists():
        shutil.rmtree(CORPUS_DIR)
    CORPUS_DIR.mkdir(parents=True)

    entries: list[CorpusEntry] = []

    # Iterate over problems that have reasoning files
    problem_ids = sorted(
        pid
        for pid in problem_cats
        if (REASONING_DIR / f"{pid}.txt").exists() and pid in prompts
    )

    for problem_id in problem_ids:
        category = problem_cats[problem_id]
        answer = answers[problem_id]

        # Drop the concat-only real cryptarithm_deduce traces; the forward-gen
        # rows below replace them with verified arithmetic-family reasoning.
        if CRYPT_N > 0 and CRYPT_REPLACE_REAL and category == "cryptarithm_deduce":
            continue

        reasoning_text = (REASONING_DIR / f"{problem_id}.txt").read_text().rstrip("\n")

        # Extract answer from reasoning's \boxed{} so they match
        boxed_match = re.findall(r"\\boxed\{([^}]*)\}", reasoning_text)
        reasoning_answer = boxed_match[-1] if boxed_match else answer
        completion_text = (
            f"{reasoning_text}\n</think>\n\\boxed{{{reasoning_answer}}}<|im_end|>"
        )
        completion_ids = tokenizer.encode(completion_text, add_special_tokens=False).ids

        # Tokenize prompt directly (no raw/ dependency)
        prompt_ids = tokenize_prompt(prompts[problem_id], chat_tokenizer)

        all_tokens = prompt_ids + completion_ids
        mask = [0] * len(prompt_ids) + [1] * len(completion_ids)

        # Truncate to token limit
        if len(all_tokens) > TOKEN_LIMIT:
            all_tokens = all_tokens[:TOKEN_LIMIT]
            mask = mask[:TOKEN_LIMIT]

        unmasked_count = sum(mask)
        masked_count = len(mask) - unmasked_count

        entry = CorpusEntry(
            problem_id=problem_id,
            category=category,
            tokens=all_tokens,
            mask=mask,
            masked_token_count=masked_count,
            unmasked_token_count=unmasked_count,
            answer=answer,
            included=True,
        )

        # Build interleaved segments and write segment file
        segments = build_segments(all_tokens, mask)

        problem_dir = CORPUS_DIR / problem_id
        problem_dir.mkdir(parents=True, exist_ok=True)
        seg_path = problem_dir / "synthetic.jsonl"

        with open(seg_path, "w") as f:
            for seg in segments:
                json.dump(seg, f)
                f.write("\n")

        entries.append(entry)

    # Add forward-generated cryptarithm_deduce traces (Phase 5 of the #1 lever).
    if CRYPT_N > 0:
        crypt_rows = build_cryptarithm_rows(
            CRYPT_N,
            CRYPT_STYLE,
            _load_cryptarithm_holdout(),
            tokenizer,
            chat_tokenizer,
            write_segments_to=CORPUS_DIR,
        )
        entries.extend(crypt_rows)
        print(
            f"Added {len(crypt_rows)} cryptarithm_deduce forward-gen rows "
            f"(style={CRYPT_STYLE})"
        )

    # Process augmentations/*.txt (no reasoning, no \boxed{})
    if AUGMENTATIONS_DIR.exists():
        for aug_path in sorted(AUGMENTATIONS_DIR.glob("*.txt")):
            text = aug_path.read_text()
            # Parse [category], [prompt], and [completion] sections
            category = text.split("[category]\n", 1)[1].split("\n[prompt]\n", 1)[0]
            prompt_text = text.split("[prompt]\n", 1)[1].split("\n[completion]\n", 1)[0]
            completion = text.split("\n[completion]\n", 1)[1].rstrip("\n")

            problem_id = aug_path.stem

            completion_text = f"{completion}\n</think><|im_end|>"
            completion_ids = tokenizer.encode(
                completion_text, add_special_tokens=False
            ).ids

            prompt_ids = tokenize_prompt(prompt_text, chat_tokenizer, suffix="")

            all_tokens = prompt_ids + completion_ids
            mask = [0] * len(prompt_ids) + [1] * len(completion_ids)

            assert len(all_tokens) <= TOKEN_LIMIT, (
                f"augmented entry {problem_id} exceeds token limit: "
                f"{len(all_tokens)} > {TOKEN_LIMIT}"
            )

            unmasked_count = sum(mask)
            masked_count = len(mask) - unmasked_count

            entry = CorpusEntry(
                problem_id=problem_id,
                category=category,
                tokens=all_tokens,
                mask=mask,
                masked_token_count=masked_count,
                unmasked_token_count=unmasked_count,
                answer=completion,
                included=True,
            )

            segments = build_segments(all_tokens, mask)
            problem_dir = CORPUS_DIR / problem_id
            problem_dir.mkdir(parents=True, exist_ok=True)
            with open(problem_dir / "synthetic.jsonl", "w") as sf:
                for seg in segments:
                    json.dump(seg, sf)
                    sf.write("\n")

            entries.append(entry)

    entries.sort(key=lambda e: e.problem_id)

    # Write index JSONL
    with open(CORPUS_INDEX, "w") as f:
        for e in entries:
            json.dump(e.to_index_dict(), f)
            f.write("\n")

    # Stats
    cat_counts: dict[str, int] = {cat: 0 for cat in {e.category for e in entries}}
    cat_tokens: dict[str, int] = {cat: 0 for cat in cat_counts}
    for e in entries:
        cat_counts[e.category] += 1
        cat_tokens[e.category] += e.unmasked_token_count

    total_unmasked = sum(e.unmasked_token_count for e in entries)
    total_masked = sum(e.masked_token_count for e in entries)
    max_tokens = max((e.token_count for e in entries), default=0)

    print(f"Corpus (synthetic): {len(entries)} entries")
    print(f"Unmasked tokens: {total_unmasked:,}")
    print(f"Masked tokens:   {total_masked:,}")
    print(f"Max seq length:  {max_tokens:,}")
    print()
    for cat in sorted(cat_counts):
        print(f"  {cat}: {cat_counts[cat]} runs, {cat_tokens[cat]:,} unmasked tokens")


if __name__ == "__main__":
    main()
