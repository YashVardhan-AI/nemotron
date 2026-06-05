# ruff: noqa: E402
# =============================================================================
# REAL SNAPSHOT-COMPLEMENT HOLDOUT EVAL — SELF-CONTAINED (no `val/` import).
#
# Companion to kaggle_newrule_eval_standalone.py. That one scores SYNTHETIC new
# rules (cipher + bit). This one scores the model on REAL problems that were NOT
# in the training snapshot — covering the other 7 categories from unseen real
# data. It needs only the repo's DATA files (problems.jsonl, problems/<id>.jsonl,
# training/sft/04-08-16-14/logprobs/index.jsonl), which your Kaggle dataset has,
# plus the `llm` the notebook already built. It does NOT import `val/`.
#
# Paste each `# %% ── Cell N ──` block as its own cell, in order, AFTER the
# notebook's "Init vLLM" cell (so `llm` exists) and with the adapter at
# /kaggle/working.
#
# READ THE TABLE CAREFULLY:
#   - CLEAN unbiased baseline: numeral / gravity / unit_conversion (absent from
#     training via random downsampling).
#   - FLOOR only (pessimistic): bit_manipulation / cryptarithm_* / equation_* —
#     absent via *solver failure*, so biased toward the hardest; true accuracy
#     is HIGHER than shown.
#   - cipher is mostly absent from the holdout (it was ~100% solved, not
#     downsampled).
# =============================================================================


# %% ── Cell 1: config ───────────────────────────────────────────────────────
DATA_ROOT = (
    ""  # folder containing problems.jsonl; "" => auto-detect under /kaggle/input
)
SNAPSHOT_INDEX_REL = "training/sft/04-08-16-14/logprobs/index.jsonl"
ADAPTER_PATH = "/kaggle/working"
LIMIT_PER_CATEGORY = (
    40  # cap problems per category (controls runtime); raise for tighter numbers
)
BATCH_SIZE = 64  # = vLLM max_num_seqs; accuracy prints after each batch
OUT_JSON = "/kaggle/working/realholdout_report.json"
# cipher is ~100% solved + not downsampled, so it's nearly ABSENT from the
# snapshot-complement holdout -> there was no apples-to-apples real cipher number,
# which is how the induct run's substitution crash (real cipher 97.5->52.5) went
# unseen. This adds a RETENTION canary: score real cipher problems (regardless of
# train membership) as a separate 'cipher_canary' row. It is NOT a generalization
# number (these were likely trained) -- it's a regression tripwire on the
# substitution circuit. A healthy adapter scores ~97% here; a damaged one drops.
LOAD_CIPHER_CANARY = True
CIPHER_CANARY_N = 200
# Scaled in-distribution bit eval (the bit leaderboard proxy + the instrument for
# measuring a bit-lever gain; the seen-data self-eval is only n~33 for bit).
LOAD_BIT_RETENTION = True
BIT_RETENTION_N = 250


# %% ── Cell 2: grading (verbatim from the real scoring path) ─────────────────
import math
import re


def extract_final_answer(text):
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


def verify(stored_answer, predicted):
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()
    try:
        return math.isclose(
            float(stored_answer), float(predicted), rel_tol=1e-2, abs_tol=1e-5
        )
    except (ValueError, TypeError):
        return predicted.lower() == stored_answer.lower()


def verify_strict(stored_answer, predicted):
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()
    if re.fullmatch(r"[01]+", stored_answer):
        return predicted.lower() == stored_answer.lower()
    try:
        return math.isclose(
            float(stored_answer), float(predicted), rel_tol=1e-2, abs_tol=1e-5
        )
    except (ValueError, TypeError):
        return predicted.lower() == stored_answer.lower()


# %% ── Cell 3: load the real snapshot-complement holdout ─────────────────────
import glob
import json
import os
from dataclasses import dataclass

# Absent via random downsampling -> unbiased. Others absent via solver failure.
CLEAN_CATEGORIES = {"numeral", "gravity", "unit_conversion"}
_DUP_SUFFIX = re.compile(r"-(?:p|d)\d+$")


@dataclass
class RealProblem:
    id: str
    category: str
    prompt: str
    answer: str
    n_examples: int
    clean: bool


def _find_data_root():
    if DATA_ROOT and os.path.exists(os.path.join(DATA_ROOT, "problems.jsonl")):
        return DATA_ROOT
    if os.path.exists("problems.jsonl"):
        return "."
    hits = glob.glob("/kaggle/input/**/problems.jsonl", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    raise SystemExit("Could not find problems.jsonl; set DATA_ROOT in Cell 1.")


def load_real_holdout(limit_per_category):
    root = _find_data_root()
    print(f"[data root] {root}")

    # 1) base ids that were trained (strip -p0 / -dN duplicate suffixes)
    trained = set()
    with open(os.path.join(root, SNAPSHOT_INDEX_REL)) as f:
        for line in f:
            line = line.strip()
            if line:
                trained.add(_DUP_SUFFIX.sub("", json.loads(line)["problem_id"]))

    # 2) holdout ids by category (problems.jsonl minus trained), capped per category
    by_cat = {}
    with open(os.path.join(root, "problems.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e["id"] in trained:
                continue
            by_cat.setdefault(e["category"], []).append(e["id"])

    # 3) read full problem files only for the capped ids
    problems = []
    for cat in sorted(by_cat):
        for pid in sorted(by_cat[cat])[:limit_per_category]:
            d = json.loads(open(os.path.join(root, "problems", f"{pid}.jsonl")).read())
            problems.append(
                RealProblem(
                    id=pid,
                    category=cat,
                    prompt=str(d["prompt"]),
                    answer=str(d["answer"]),
                    n_examples=len(d.get("examples", [])),
                    clean=cat in CLEAN_CATEGORIES,
                )
            )
    return problems


def _load_real_sample(real_category, n, row_category):
    """Score N real problems of *real_category* regardless of train membership.

    This is a RETENTION / in-distribution metric (NOT generalization): the hidden
    test is essentially in-distribution (seen-data accuracy ~= leaderboard), so a
    big real sample is the leaderboard-predictive instrument AND a regression
    tripwire. Tagged *row_category* so it never contaminates the clean/floor
    holdout semantics.
    """
    root = _find_data_root()
    ids = []
    with open(os.path.join(root, "problems.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e["category"] == real_category:
                ids.append(e["id"])
    problems = []
    for pid in sorted(ids)[:n]:
        d = json.loads(open(os.path.join(root, "problems", f"{pid}.jsonl")).read())
        problems.append(
            RealProblem(
                id=pid,
                category=row_category,
                prompt=str(d["prompt"]),
                answer=str(d["answer"]),
                n_examples=len(d.get("examples", [])),
                clean=False,
            )
        )
    return problems


def load_cipher_canary(n):
    """Real cipher retention canary (catches a substitution-circuit regression)."""
    return _load_real_sample("cipher", n, "cipher_canary")


def load_bit_retention(n):
    """Real bit in-distribution eval at scale -- the bit leaderboard proxy.

    The seen-data self-eval is only n~33 for bit, too noisy to see a +1.5% bit
    gain (the lever). N>=250 real bit problems makes a +5% bit change clear noise.
    """
    return _load_real_sample("bit_manipulation", n, "bit_retention")


# %% ── Cell 4: scoring + reporting ───────────────────────────────────────────
from collections import defaultdict

BOXED_INSTRUCTION = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def build_eval_prompt(problem):
    return problem.prompt + BOXED_INSTRUCTION


def _stats(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "accuracy_strict": 0.0}
    return {
        "n": n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "accuracy_strict": sum(r["correct_strict"] for r in rows) / n,
    }


def aggregate(results):
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    report = {}
    for cat, rows in by_cat.items():
        report[cat] = {
            "overall": _stats(rows),
            "clean": bool(rows[0]["clean"]),
        }
    return report


def format_table(report):
    lines = [
        f"{'Category':<24} {'baseline':<6} {'N':>5} {'Acc%':>8} {'StrictAcc%':>11}"
    ]
    lines.append("-" * 60)
    for cat in sorted(report):
        ov = report[cat]["overall"]
        if cat in ("cipher_canary", "bit_retention"):
            tag = "real"  # in-distribution retention (leaderboard proxy), not a floor
        else:
            tag = "clean" if report[cat]["clean"] else "floor"
        lines.append(
            f"{cat:<24} {tag:<6} {ov['n']:>5} "
            f"{ov['accuracy'] * 100:>8.1f} {ov['accuracy_strict'] * 100:>11.1f}"
        )
    return "\n".join(lines)


# %% ── Cell 5: run — reuse the notebook's `llm`, batched, accuracy per batch ──
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

assert "llm" in globals(), "Run AFTER the notebook's 'Init vLLM' cell defines `llm`."

_tokenizer = llm.get_tokenizer()  # noqa: F821
_sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=7680)
_lora = LoRARequest("adapter", 1, ADAPTER_PATH)


def predict(eval_prompts):
    rendered = []
    for ep in eval_prompts:
        try:
            r = _tokenizer.apply_chat_template(
                [{"role": "user", "content": ep}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError):
            r = ep
        rendered.append(r)
    outputs = llm.generate(  # noqa: F821
        rendered, sampling_params=_sampling, lora_request=_lora
    )
    return [o.outputs[0].text for o in outputs]


problems = load_real_holdout(LIMIT_PER_CATEGORY)
if LOAD_CIPHER_CANARY:
    problems += load_cipher_canary(CIPHER_CANARY_N)
if LOAD_BIT_RETENTION:
    problems += load_bit_retention(BIT_RETENTION_N)
print(
    f"holdout problems: {len(problems)} across {len({p.category for p in problems})} categories"
)

results = []
seen = hit = 0
cat_seen = defaultdict(int)
cat_hit = defaultdict(int)
n_total = len(problems)
n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE
for bi in range(0, n_total, BATCH_SIZE):
    batch = problems[bi : bi + BATCH_SIZE]
    raw_texts = predict([build_eval_prompt(p) for p in batch])
    for p, raw in zip(batch, raw_texts):
        pred = extract_final_answer(raw)
        ok = verify(p.answer, pred)
        results.append(
            {
                "id": p.id,
                "category": p.category,
                "clean": p.clean,
                "answer": p.answer,
                "predicted": pred,
                "correct": ok,
                "correct_strict": verify_strict(p.answer, pred),
                "raw": raw,
            }
        )
        seen += 1
        hit += int(ok)
        cat_seen[p.category] += 1
        cat_hit[p.category] += int(ok)
    per_cat = "  ".join(
        f"{c} {cat_hit[c] / cat_seen[c] * 100:.0f}%" for c in sorted(cat_seen)
    )
    print(
        f"[batch {bi // BATCH_SIZE + 1:>2}/{n_batches}] {seen:>4}/{n_total} done  "
        f"running {hit / seen * 100:5.1f}% ({hit}/{seen})  |  {per_cat}",
        flush=True,
    )

report = aggregate(results)
print("\n" + "=" * 64)
print("REAL snapshot-complement holdout (unseen real problems)")
print("clean = unbiased baseline; floor = pessimistic (solver-failure bias)")
print("=" * 64)
print(format_table(report))

with open(OUT_JSON, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nwrote {OUT_JSON}")
