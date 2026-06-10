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
import hashlib
import json
import os
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
# NEGATIVE RESULT (measured): induct cryptarithm crashed neighbors (cipher/equation)
# and overall 0.86 -> 0.83. Disabled (CRYPT_N=0 keeps the real cryptarithm rows).
CRYPT_N = 0
CRYPT_STYLE = "induct"
CRYPT_DIFFICULTY = 4
CRYPT_SEED_OFFSET = 1_000_000  # disjoint from val seeds (0..few-thousand)
HOLDOUT_RULES = Path(__file__).parent / "holdout_rules.json"
# The real cryptarithm_deduce reasoning files (reasoning/*.txt) are concat-only:
# 658/659 default to concatenation and NONE show arithmetic, so for arithmetic
# problems they pair concat reasoning with an arithmetic answer (an inconsistent
# signal -- the root cause of the ~6% baseline). When forward-gen is on, REPLACE
# them with the verified forward-gen traces rather than mixing both.
CRYPT_REPLACE_REAL = True

# --- symbolic-solver cryptarithm traces (CORRECT, full-distribution) -----------
# Replace the wrong concat-fallback real cryptarithm reasoning (reasoning/*.txt)
# with traces rendered from the VERIFIED solver (kaggle-nemotron-equation-symbolic).
# Covers ~725 of the ~781 real cryptarithm problems; each is round-trip-checked to
# reproduce gold (see reasoners/crypt_symbolic_corpus.py + memory
# crypt-symbolic-renderer-built). CRYPT_SYMBOLIC_STYLE is the A/B knob:
#   "assert"           = MRV forced/guess scratchpad ;
#   "derive"           = genuine (variable-length) propagation ;
#   "derive_inductive" = bounded one-row-per-glyph + binding-equation citation +
#                        LSB-first encoding (the step-count-independent recast) ;
#   "lean"             = state the map only. Enable per run, e.g.:
#   CRYPT_SYMBOLIC=1 CRYPT_SYMBOLIC_STYLE=derive_inductive uv run corpus.py
CRYPT_SYMBOLIC = os.environ.get("CRYPT_SYMBOLIC", "0") != "0"
CRYPT_SYMBOLIC_STYLE = os.environ.get("CRYPT_SYMBOLIC_STYLE", "assert")

# --- bit_manipulation forward-gen (the bit lever) ----------------------------
# ADD N verified per-bit-induction traces for the families the corpus does NOT
# teach: complex 3-input (the solver reasoners/bit_manipulation.py cannot express
# MAJ/CHOICE, so no good real traces exist) + heterogeneous per-column pairwise.
# These ADD to the real bit rows (which cover the easy 2-input/rotation families).
# Seeds start high (disjoint from val AND cryptarithm seeds); any val-reserved
# rule_signature is skipped, so no validation rule can leak. Set BIT_N = 0 to off.
# NEGATIVE RESULT (measured): BIT_N=800 forward-gen per-bit-assertion traces
# CRASHED bit (bit_retention 88.8% -> 22.0%) by overwriting the global-deduction
# strategy that solved the hom tier (pairwise/hom 90% -> 9%). het tier did NOT
# improve. The het/complex tiers are information-underdetermined (8 examples admit
# multiple rules); the assertion trace cannot teach an unlearnable deduction and
# erases the working hom strategy. Do NOT re-enable without a fundamentally
# different trace shape. See memory eval-harden-bit-lever-plan.
BIT_N = 0
BIT_DIFFICULTY = 8  # real bit gives 7-10 examples (avg ~8.6)
BIT_SEED_OFFSET = 2_000_000  # disjoint from val (~0..few-k) and cryptarithm (1e6)
# Forward-gen only the WEAK tier; skip pairwise/hom + rot the model already does.
BIT_TARGET_TAGS = ("complex/het", "complex/hom", "pairwise/het")

# Downsample the SATURATED categories (numeral/gravity/unit_conversion generalize
# at ~100% on unseen rules) so training capacity goes to the hard categories
# instead -- the winning recipe's documented rates. Deterministic by problem_id
# hash (stable across runs). Set DOWNSAMPLE_RATES = {} to keep every example.
DOWNSAMPLE_RATES = {"numeral": 0.4, "gravity": 0.6, "unit_conversion": 0.6}

# Per-category target row counts, reproducing the winning submission's exact
# composition (frozen snapshot training/sft/04-08-16-14). The winning recipe was
# NOT "all reasoning traces" -- it DOWNSAMPLED the saturated easy categories and
# DUPLICATED the scarce/hard ones to give them more gradient steps. After the
# unique pool is selected (reasoning files + DOWNSAMPLE_RATES), each category is
# duplicated cyclically (or subsampled) to hit its target. Sum = 7830 rows.
# Set DUP_TARGETS = {} to disable rebalancing (one row per unique trace).
DUP_TARGETS = {
    # +BIT_N so the forward-gen weak-tier traces ADD on top of the real bit rows
    # (the lever) rather than being truncated away by the rebalance cap. With
    # BIT_N>0 the unique pool (real + forward-gen) is always < this target, so the
    # cyclic-duplication branch keeps every unique trace -- no forward-gen dropped.
    "bit_manipulation": 1754 + BIT_N,
    "cipher": 1656,
    "unit_conversion": 1070,
    "gravity": 1055,
    "numeral": 730,
    "equation_numeric_deduce": 658,
    "cryptarithm_deduce": 627,
    "cryptarithm_guess": 154,
    "equation_numeric_guess": 126,
}

# --- increase-hard-categories A/B (BOOST_HARD=1) -----------------------------
# User experiment 2026-06-07: INCREASE the number of samples on the HARD
# categories (bit_manipulation, cryptarithm, equation_numeric) while keeping the
# SATURATED categories (gravity/numeral/unit_conversion/cipher) exactly as the
# winning recipe has them. Same CoT throughout -- this only raises DUP_TARGETS,
# adding more COPIES of the existing real traces; it introduces NO new trace
# shape, so it is SAFE re: the measured neighbor-interference crashes
# ([[induct-sft-negative-result]]). Total rises ~7830 -> ~9960.
#
# HONEST CAVEAT (memory training-recipe-unswept-levers): the hard-category unique
# pools are already fully used (no downsampling on them), so "more samples" =
# more REPETITION of the same problems, not new unique data (no same-CoT
# generator exists for these). At 1 epoch, duplicating a category to ~2x means
# the model sees each of its problems ~2x -- i.e. this is effectively "train the
# hard categories ~2x longer" while the saturated ones stay 1x. Expect a small
# effect at best (bit het + cryptarithm are information/structure capped; the
# extra reps cannot beat those ceilings), but it is a cheap, safe arm and a
# measured null pins the ceiling. Tune the targets below freely.
# Set BOOST_HARD=1 (env) to enable; default OFF preserves the winning recipe.
BOOST_HARD = os.environ.get("BOOST_HARD", "0") != "0"
BOOST_HARD_DUP_TARGETS = {
    # HARD categories -> more copies (same CoT). bit is the token-hog (median
    # 6722 tok) so its bump is the costliest in train time + least likely to pay
    # (het tier is info-capped) -- kept modest; raise/lower as you like.
    "bit_manipulation": 2200 + BIT_N,
    "equation_numeric_deduce": 1300,
    "equation_numeric_guess": 300,
    "cryptarithm_deduce": 1300,
    "cryptarithm_guess": 350,
    # SATURATED categories -> UNCHANGED (identical to the winning DUP_TARGETS).
    "cipher": 1656,
    "unit_conversion": 1070,
    "gravity": 1055,
    "numeral": 730,
}
if BOOST_HARD:
    # DOWNSAMPLE_RATES unchanged: saturated categories stay exactly as shipped.
    DUP_TARGETS = BOOST_HARD_DUP_TARGETS
    print(
        "BOOST_HARD=1: more samples on hard cats (bit/cryptarithm/equation), "
        f"saturated held (targets sum {sum(BOOST_HARD_DUP_TARGETS.values())})"
    )

# Build speed: augmentation categories are NOT reasoning categories, so the
# reasoning-only train filter (train_sft.filter_training_examples) discards them
# anyway -- tokenizing ~8.5k of them just to throw them away ~doubles the build
# time. DEFAULT FALSE: the submission/training path is reasoning-only, so we don't
# build augmented traces at all. Set BUILD_AUGMENTATIONS=1 (env) to build the full
# (non-submission) corpus.
BUILD_AUGMENTATIONS = os.environ.get("BUILD_AUGMENTATIONS", "0") != "0"


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
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


def _keep_by_hash(problem_id: str, rate: float) -> bool:
    """Deterministic keep decision for downsampling (stable across runs)."""
    if rate >= 1.0:
        return True
    h = int(hashlib.md5(problem_id.encode()).hexdigest()[:8], 16)
    return (h % 10_000) / 10_000 < rate


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
    if style == "induct":
        from reasoners.cryptarithm_trace import (
            reasoning_cryptarithm_induct,
        )

        def _induct_or_fallback(problem, answer):
            t = reasoning_cryptarithm_induct(problem, answer)
            return t if t is not None else reasoning_cryptarithm_arith(problem, answer)

        return _induct_or_fallback
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
            with open(problem_dir / "synthetic.jsonl", "w", encoding="utf-8") as f:
                for seg in build_segments(all_tokens, mask):
                    json.dump(seg, f)
                    f.write("\n")
        rows.append(entry)
    return rows


def _load_cryptarithm_holdout() -> set[str]:
    if not HOLDOUT_RULES.exists():
        return set()
    return set(
        json.loads(HOLDOUT_RULES.read_text(encoding="utf-8")).get(
            "cryptarithm_deduce", []
        )
    )


def _load_bit_holdout() -> set[str]:
    if not HOLDOUT_RULES.exists():
        return set()
    return set(
        json.loads(HOLDOUT_RULES.read_text(encoding="utf-8")).get(
            "bit_manipulation", []
        )
    )


def build_bit_examples(
    n: int,
    holdout: set[str],
    *,
    difficulty: int = BIT_DIFFICULTY,
    seed_offset: int = BIT_SEED_OFFSET,
    target_tags: tuple[str, ...] = BIT_TARGET_TAGS,
) -> list[tuple[str, str, str, str]]:
    """Tokenizer-free generator of N distinct verified bit_manipulation rows.

    Returns (problem_id, prompt_text, completion_text, answer) tuples for the WEAK
    families (complex 3-input + heterogeneous pairwise). Each trace is sound by
    construction (the renderer returns None unless its per-bit rules reproduce
    every example and its applied output equals the answer). Skips val-reserved
    signatures; seeds start at *seed_offset*, disjoint from the val generator.
    """
    from reasoners.bit_rule import problem_tag, rule_signature
    from reasoners.bit_trace import make_bit_trace_problem, reasoning_bit_perbit

    targets = set(target_tags)
    out: list[tuple[str, str, str, str]] = []
    seed = seed_offset
    guard = seed_offset + 100 * (n + 1)  # generous; weak-tier is ~half of seeds
    while len(out) < n and seed < guard:
        tag = problem_tag(seed)
        if tag not in targets or rule_signature(seed) in holdout:
            seed += 1
            continue
        problem, answer, cols = make_bit_trace_problem(seed, difficulty)
        reasoning = reasoning_bit_perbit(problem, answer, cols)
        if reasoning is None:  # unsound (should not happen for forward-gen); skip
            seed += 1
            continue
        completion = f"{reasoning}\n</think>\n\\boxed{{{answer}}}<|im_end|>"
        pid = f"bit-{tag.replace('/', '-')}-{seed}"
        out.append((pid, problem.prompt, completion, answer))
        seed += 1
    return out


def build_bit_rows(
    n: int,
    holdout: set[str],
    tokenizer,
    chat_tokenizer,
    *,
    difficulty: int = BIT_DIFFICULTY,
    write_segments_to: Path | None = None,
) -> list[CorpusEntry]:
    """Tokenize build_bit_examples into CorpusEntry rows (prompt masked, completion
    unmasked), matching the real-problem path."""
    rows: list[CorpusEntry] = []
    for pid, prompt_text, completion_text, answer in build_bit_examples(
        n, holdout, difficulty=difficulty
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
            category="bit_manipulation",
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
            with open(problem_dir / "synthetic.jsonl", "w", encoding="utf-8") as f:
                for seg in build_segments(all_tokens, mask):
                    json.dump(seg, f)
                    f.write("\n")
        rows.append(entry)
    return rows


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
    with open(TRAIN_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["id"]
            prompts[pid] = row["prompt"]
            answers[pid] = row["answer"]

    # Load problem categories
    problem_cats: dict[str, str] = {}
    for prob_raw in load_jsonl(PROBLEMS_INDEX):
        problem_cats[prob_raw["id"]] = prob_raw["category"]

    # Correct symbolic-solver cryptarithm reasoning, keyed by real train id.
    symbolic_reasoning: dict[str, str] = {}
    if CRYPT_SYMBOLIC:
        from reasoners.crypt_symbolic_corpus import load_symbolic_reasoning

        symbolic_reasoning = load_symbolic_reasoning(CRYPT_SYMBOLIC_STYLE)
        print(
            f"[corpus] symbolic cryptarithm traces: {len(symbolic_reasoning)} "
            f"(style={CRYPT_SYMBOLIC_STYLE})"
        )

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

        # Downsample saturated categories (deterministic, stable across runs).
        rate = DOWNSAMPLE_RATES.get(category)
        if rate is not None and not _keep_by_hash(problem_id, rate):
            continue

        reasoning_text = (
            (REASONING_DIR / f"{problem_id}.txt")
            .read_text(encoding="utf-8")
            .rstrip("\n")
        )

        # Override wrong concat-fallback cryptarithm reasoning with the verified
        # symbolic-solver trace (correct, glyph-separated). Uncovered cryptarithm
        # ids (~56) keep their real reasoning.
        if (
            symbolic_reasoning
            and category.startswith("cryptarithm")
            and problem_id in symbolic_reasoning
        ):
            reasoning_text = symbolic_reasoning[problem_id]

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

        with open(seg_path, "w", encoding="utf-8") as f:
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

    # Add forward-generated bit_manipulation traces for the weak tier (the lever).
    if BIT_N > 0:
        bit_rows = build_bit_rows(
            BIT_N,
            _load_bit_holdout(),
            tokenizer,
            chat_tokenizer,
            write_segments_to=CORPUS_DIR,
        )
        entries.extend(bit_rows)
        print(
            f"Added {len(bit_rows)} bit_manipulation forward-gen rows "
            f"(tags={'/'.join(BIT_TARGET_TAGS)})"
        )

    # Process augmentations/*.txt (no reasoning, no \boxed{}).
    if not BUILD_AUGMENTATIONS:
        print("Skipping augmentations (BUILD_AUGMENTATIONS=0)")
    if BUILD_AUGMENTATIONS and AUGMENTATIONS_DIR.exists():
        for aug_path in sorted(AUGMENTATIONS_DIR.glob("*.txt")):
            text = aug_path.read_text(encoding="utf-8")
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
            with open(problem_dir / "synthetic.jsonl", "w", encoding="utf-8") as sf:
                for seg in segments:
                    json.dump(seg, sf)
                    sf.write("\n")

            entries.append(entry)

    # Rebalance each category to the winning submission's row count by duplicating
    # (or subsampling) its unique pool. Duplicate entries reuse the same
    # problem_id, so get_segment_path() still resolves to the one segment dir on
    # disk -- N identical index lines become N training examples (no disk bloat).
    # Categories absent from DUP_TARGETS (e.g. augmentations) pass through.
    if DUP_TARGETS:
        by_cat: dict[str, list[CorpusEntry]] = {}
        for e in entries:
            by_cat.setdefault(e.category, []).append(e)
        rebalanced: list[CorpusEntry] = []
        for cat, cat_entries in sorted(by_cat.items()):
            target = DUP_TARGETS.get(cat)
            if target is None:
                rebalanced.extend(cat_entries)
                continue
            pool = sorted(cat_entries, key=lambda e: e.problem_id)
            if len(pool) >= target:
                rebalanced.extend(pool[:target])
            else:
                rebalanced.extend(pool[i % len(pool)] for i in range(target))
            print(f"  rebalance {cat}: {len(pool)} unique -> {target} rows")
        entries = rebalanced

    entries.sort(key=lambda e: e.problem_id)

    # Write index JSONL
    with open(CORPUS_INDEX, "w", encoding="utf-8") as f:
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
