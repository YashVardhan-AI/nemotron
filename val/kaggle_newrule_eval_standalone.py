# ruff: noqa: E402
# =============================================================================
# NEW-RULE GENERALIZATION EVAL — SELF-CONTAINED (no repo imports).
#
# Paste these cells into huikang's adapter-validation-notebook, AFTER the
# "Init vLLM" cell has created `llm` and the adapter is at /kaggle/working.
#
# Each `# %% ── Cell N ──` marker below is one notebook cell. Copy each block
# into its own cell, in order, and run top-to-bottom. Cells share globals, so
# later cells use the functions/types defined in earlier ones.
#
# This measures accuracy on BRAND-NEW rules the model never trained on
# (cipher + bit_manipulation), i.e. the actual test skill. Compare it to the
# notebook's TRAIN-set table to see the memorize-vs-generalize gap.
#
# Self-contained: only stdlib + the `llm` the notebook already built. Nothing
# is imported from the repo, so no dataset attach / sys.path setup is needed.
# =============================================================================


# %% ── Cell 1: config ───────────────────────────────────────────────────────
ADAPTER_PATH = "/kaggle/working"  # dir with adapter_config.json + adapter_model.safetensors
PER_CATEGORY = 50                 # how many distinct new rules per generator
DIFFICULTY = 6                    # in-context examples shown per problem
OUT_JSON = "/kaggle/working/newrule_report.json"


# %% ── Cell 2: grading (verbatim from the real scoring path) ─────────────────
import math
import re


def extract_final_answer(text):
    """Mirror of the competition's extract_final_answer."""
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
    """The real grader: float-lenient (binary strings within 1% count equal)."""
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()
    try:
        return math.isclose(
            float(stored_answer), float(predicted), rel_tol=1e-2, abs_tol=1e-5
        )
    except (ValueError, TypeError):
        return predicted.lower() == stored_answer.lower()


def verify_strict(stored_answer, predicted):
    """Diagnostic: exact on binary strings (exposes where `verify` is lenient)."""
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


# %% ── Cell 3: minimal Problem type + new-rule generators ────────────────────
import random
from dataclasses import dataclass


@dataclass
class Problem:
    id: str
    category: str
    prompt: str
    answer: str
    n_examples: int  # used as the "difficulty" bucket


# --- cipher: a brand-new substitution alphabet per seed ----------------------
_CIPHER_WORDS = [
    "queen", "dragon", "castle", "secret", "near", "valley", "discovers",
    "dreams", "inside", "student", "creates", "magical", "door", "golden",
    "follows", "princess", "reads", "mysterious", "cat", "imagines", "book",
    "wizard", "the", "guards", "hidden", "garden", "river", "mountain",
    "whispers", "ancient", "key", "opens", "silver", "gate", "forest",
]
_CIPHER_HEADER = "In Alice's Wonderland, secret encryption rules are used on text."


def _cipher_alphabet(seed):
    letters = list("abcdefghijklmnopqrstuvwxyz")
    random.Random(seed).shuffle(letters)
    return "".join(letters)


def _encrypt(text, alphabet):
    out = []
    for ch in text:
        out.append(alphabet[ord(ch) - ord("a")] if "a" <= ch <= "z" else ch)
    return "".join(out)


def _cipher_sentence(rng):
    return " ".join(rng.choice(_CIPHER_WORDS) for _ in range(rng.randint(3, 5)))


def generate_cipher(seed, difficulty):
    rng = random.Random(seed)
    alphabet = _cipher_alphabet(seed)
    plain_examples = [_cipher_sentence(rng) for _ in range(difficulty)]
    plain_query = _cipher_sentence(rng)
    lines = [f"{_encrypt(p, alphabet)} -> {p}" for p in plain_examples]
    cipher_query = _encrypt(plain_query, alphabet)
    prompt = (
        f"{_CIPHER_HEADER}\nHere are some examples:\n"
        + "\n".join(lines)
        + f"\nNow, decrypt the following text: {cipher_query}"
    )
    return Problem(
        id=f"val-cipher-{seed}",
        category="cipher",
        prompt=prompt,
        answer=plain_query,
        n_examples=difficulty,
    )


# --- bit_manipulation: new 8-bit rules (affine / xor / rotate / permute) -----
_BIT_HEADER = (
    "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit "
    "binary numbers. The transformation involves operations like bit shifts, "
    "rotations, XOR, AND, OR, NOT, and possibly majority or choice functions."
)
_BIT_RULE_TYPES = ["affine", "xor", "rotl", "perm"]


def build_bit_rule(seed):
    """Return an apply(x:int)->int for the rule selected by *seed*."""
    rng = random.Random(seed)
    rule_type = rng.choice(_BIT_RULE_TYPES)
    if rule_type == "affine":
        a = rng.choice([3, 5, 7, 9, 11, 13, 15])
        b = rng.randint(1, 255)

        def apply(x):
            return (a * x + b) & 0xFF
    elif rule_type == "xor":
        mask = rng.randint(1, 255)

        def apply(x):
            return x ^ mask
    elif rule_type == "rotl":
        k = rng.randint(1, 7)

        def apply(x):
            return ((x << k) | (x >> (8 - k))) & 0xFF
    else:  # perm — output bit j = input bit perm[j] (MSB-first)
        perm = list(range(8))
        rng.shuffle(perm)

        def apply(x):
            in_bits = [(x >> (7 - i)) & 1 for i in range(8)]
            out = 0
            for j in range(8):
                out |= in_bits[perm[j]] << (7 - j)
            return out

    return apply


def _bit_distinct_inputs(seed, count):
    if count > 256:
        raise ValueError(f"count={count} exceeds the 8-bit value pool (256)")
    rng = random.Random(seed * 1000 + 1)
    seen = []
    while len(seen) < count:
        v = rng.randint(0, 255)
        if v not in seen:
            seen.append(v)
    return seen


def generate_bit(seed, difficulty):
    apply = build_bit_rule(seed)
    values = _bit_distinct_inputs(seed, difficulty + 1)
    example_ints, query_int = values[:difficulty], values[difficulty]
    lines = [f"{format(x, '08b')} -> {format(apply(x), '08b')}" for x in example_ints]
    query_bits = format(query_int, "08b")
    prompt = (
        f"{_BIT_HEADER}\n\nHere are some examples of input -> output:\n"
        + "\n".join(lines)
        + f"\n\nNow, determine the output for: {query_bits}"
    )
    return Problem(
        id=f"val-bit_manipulation-{seed}",
        category="bit_manipulation",
        prompt=prompt,
        answer=format(apply(query_int), "08b"),
        n_examples=difficulty,
    )


# %% ── Cell 4: scoring + reporting ───────────────────────────────────────────
from collections import defaultdict

BOXED_INSTRUCTION = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def build_eval_prompt(problem):
    """problem.prompt + the boxed instruction — exactly the real eval prompt."""
    return problem.prompt + BOXED_INSTRUCTION


def score(problems, predictor):
    """predictor: list[eval_prompt] -> list[raw_text]. Returns result dicts."""
    raw_texts = predictor([build_eval_prompt(p) for p in problems])
    results = []
    for p, raw in zip(problems, raw_texts):
        pred = extract_final_answer(raw)
        results.append(
            {
                "id": p.id,
                "category": p.category,
                "difficulty": p.n_examples,
                "answer": p.answer,
                "predicted": pred,
                "correct": verify(p.answer, pred),
                "correct_strict": verify_strict(p.answer, pred),
                "raw": raw,
            }
        )
    return results


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
        by_diff = defaultdict(list)
        for r in rows:
            by_diff[r["difficulty"]].append(r)
        report[cat] = {
            "overall": _stats(rows),
            "by_difficulty": {d: _stats(rs) for d, rs in sorted(by_diff.items())},
        }
    return report


def format_table(report):
    lines = [f"{'Category':<24} {'Diff':>5} {'N':>5} {'Acc%':>8} {'StrictAcc%':>11}"]
    lines.append("-" * 56)
    for cat in sorted(report):
        ov = report[cat]["overall"]
        lines.append(
            f"{cat:<24} {'all':>5} {ov['n']:>5} "
            f"{ov['accuracy'] * 100:>8.1f} {ov['accuracy_strict'] * 100:>11.1f}"
        )
        for diff, st in report[cat]["by_difficulty"].items():
            lines.append(
                f"{'':<24} {diff:>5} {st['n']:>5} "
                f"{st['accuracy'] * 100:>8.1f} {st['accuracy_strict'] * 100:>11.1f}"
            )
    return "\n".join(lines)


# %% ── Cell 5: run — reuse the notebook's `llm`, score new rules, print ───────
import json

from vllm import SamplingParams
from vllm.lora.request import LoRARequest

assert "llm" in globals(), "Run AFTER the notebook's 'Init vLLM' cell defines `llm`."

_tokenizer = llm.get_tokenizer()  # noqa: F821  (llm is a notebook global)
_sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=7680)  # greedy, eval config
_lora = LoRARequest("adapter", 1, ADAPTER_PATH)


def predict(user_contents):
    """chat template (w/ thinking) -> greedy -> our adapter; mirrors the eval."""
    prompts = []
    for content in user_contents:
        try:
            p = _tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError):
            p = content
        prompts.append(p)
    outputs = llm.generate(prompts, sampling_params=_sampling, lora_request=_lora)  # noqa: F821
    return [o.outputs[0].text for o in outputs]


problems = [generate_cipher(s, DIFFICULTY) for s in range(PER_CATEGORY)]
problems += [generate_bit(s, DIFFICULTY) for s in range(PER_CATEGORY)]

results = score(problems, predict)
report = aggregate(results)

print("=" * 64)
print("NEW-RULE generalization (synthetic, rules never trained on)")
print("=" * 64)
print(format_table(report))

with open(OUT_JSON, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nwrote {OUT_JSON}")

# Compare against the notebook's TRAIN table (gravity/numeral/unit_conv ~100%,
# bit ~88%, cryptarithm ~8%). A big drop here = the generalization gap.
