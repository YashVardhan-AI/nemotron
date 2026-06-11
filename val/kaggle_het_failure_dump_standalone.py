# ruff: noqa: E402
# =============================================================================
# HET BIT FAILURE DUMP — SELF-CONTAINED Kaggle paste-in.
#
# Characterizes HOW the model fails heterogeneous bit_manipulation, to decide if
# a bit-het train run is worth it. The model-FREE half (val/bit_identifiability.py)
# already proved: het is INFO-CAPPED at exact-match (oracle ceiling ~6.5%/0% =
# model StrictAcc), but under the REAL lenient grader pairwise/het has ~+8..24pp
# MODEL-error headroom (model under-pins high-order bits). This cell confirms the
# FAILURE MODE on the live adapter:
#   * lenient vs strict accuracy (should reproduce ~46.7 / ~3.3 on pairwise/het)
#   * truncation / wrong-length rate
#   * Hamming distance to gold; HIGH-order (bits 0-1, the ones the lenient grader
#     needs) vs LOW-order error rates
#   * HOMOGENEOUS-COLLAPSE: does the model's answer match the best single global
#     (homogeneous) rule fit to the examples? high => the model wrongly assumes one
#     global op (transfer from the dominant hom training) => bit-het SFT targets it
#
# Paste each `# %% Cell` block into its own cell, AFTER the notebook's "Init vLLM"
# cell defines `llm` and the adapter is at /kaggle/working. Self-contained: stdlib
# + the notebook's `llm` only.
# =============================================================================


# %% Cell 1: config ----------------------------------------------------------
ADAPTER_PATH = "/kaggle/working"
N_HET = 300  # het problems to dump (pairwise/het + complex/het; ~37% of seeds are het)
BIT_DIFFICULTY = 8  # matches the eval (real avg 8.6 examples)
BATCH_SIZE = 64
DUMP_PATH = "/kaggle/working/het_failure_dump.jsonl"  # raw completions + diagnostics


# %% Cell 2: grading (verbatim from the real scoring path) -------------------
import json
import math
import re
from collections import defaultdict


def extract_final_answer(text):
    if text is None:
        return "NOT_FOUND"
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        return non_empty[-1] if non_empty else matches[-1].strip()
    for pat in (
        r"The final answer is:\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ):
        m = re.findall(pat, text, re.IGNORECASE)
        if m:
            return m[-1].strip()
    m = re.findall(r"-?\d+(?:\.\d+)?", text)
    if m:
        return m[-1]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def verify(stored, pred):
    """REAL grader: lenient (binary parsed as decimal, within 1%)."""
    stored, pred = stored.strip(), pred.strip()
    try:
        return math.isclose(float(stored), float(pred), rel_tol=1e-2, abs_tol=1e-5)
    except (ValueError, TypeError):
        return pred.lower() == stored.lower()


def verify_strict(stored, pred):
    stored, pred = stored.strip(), pred.strip()
    if re.fullmatch(r"[01]+", stored):
        return pred.lower() == stored.lower()
    return verify(stored, pred)


# %% Cell 3: het bit generator + diagnostics (inlined, matches val core) ------
import random

_BIT_HEADER = (
    "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit "
    "binary numbers. The transformation involves operations like bit shifts, "
    "rotations, XOR, AND, OR, NOT, and possibly majority or choice functions."
)
_PAIR_OPS = ["AND", "OR", "XOR", "AND-NOT", "OR-NOT", "XOR-NOT"]
_HET_PROB = 0.46


def _bitval(x, p):
    return (x >> (7 - p)) & 1


def _pack(bits):
    out = 0
    for j in range(8):
        out |= bits[j] << (7 - j)
    return out


def _eval_col(col, x):
    kind = col[0]
    if kind == "route":
        _, p, neg = col
        b = _bitval(x, p)
        return 1 - b if neg else b
    if kind == "pair":
        _, op, p, q = col
        u, v = _bitval(x, p), _bitval(x, q)
        if op.endswith("-NOT"):
            v = 1 - v
        base = op.split("-")[0]
        return u & v if base == "AND" else (u | v if base == "OR" else u ^ v)
    if kind == "maj":
        _, p, q, r = col
        return 1 if (_bitval(x, p) + _bitval(x, q) + _bitval(x, r)) >= 2 else 0
    _, p, q, r = col
    return _bitval(x, q) if _bitval(x, p) else _bitval(x, r)


def _distinct_pair(rng):
    p = rng.randint(0, 7)
    q = rng.randint(0, 7)
    while q == p:
        q = rng.randint(0, 7)
    return p, q


def _build_columns(rng, profile, het):
    if profile == "rot":
        glob_neg = rng.random() < 0.3
        if rng.random() < 0.90:
            k = rng.randint(1, 7)
            return [("route", (j + k) % 8, glob_neg) for j in range(8)]
        perm = list(range(8))
        rng.shuffle(perm)
        return [("route", perm[j], glob_neg) for j in range(8)]
    global_op = rng.choice(_PAIR_OPS)
    a = rng.randint(0, 7)
    b = (a + rng.randint(1, 7)) % 8
    c = (a + rng.randint(1, 7)) % 8
    three = (
        set(rng.sample(range(8), rng.randint(1, 3))) if profile == "complex" else set()
    )
    cols = []
    for j in range(8):
        if j in three:
            kind = rng.choice(("maj", "choice"))
            if het:
                p, q, r = rng.sample(range(8), 3)
            else:
                p, q, r = (a + j) % 8, (b + j) % 8, (c + j) % 8
            cols.append((kind, p, q, r))
        else:
            if het:
                op = rng.choice(_PAIR_OPS)
                p, q = _distinct_pair(rng)
            else:
                op, p, q = global_op, (a + j) % 8, (b + j) % 8
            cols.append(("pair", op, p, q))
    return cols


def build_bit_rule(seed):
    rng = random.Random(seed)
    roll = rng.random()
    profile = "pairwise" if roll < 0.65 else ("rot" if roll < 0.85 else "complex")
    het = rng.random() < _HET_PROB if profile != "rot" else False
    cols = _build_columns(rng, profile, het)
    return profile, het, cols


def _distinct_inputs(seed, count):
    rng = random.Random(seed * 1000 + 1)
    seen = []
    while len(seen) < count:
        v = rng.randint(0, 255)
        if v not in seen:
            seen.append(v)
    return seen


def gen_het(seed, difficulty):
    """Return a het problem dict, or None if this seed is not heterogeneous."""
    profile, het, cols = build_bit_rule(seed)
    if not het:
        return None

    def apply(x):
        return _pack([_eval_col(c, x) for c in cols])

    values = _distinct_inputs(seed, difficulty + 1)
    ex, q = values[:difficulty], values[difficulty]
    lines = [f"{format(x, '08b')} -> {format(apply(x), '08b')}" for x in ex]
    prompt = (
        f"{_BIT_HEADER}\n\nHere are some examples of input -> output:\n"
        + "\n".join(lines)
        + f"\n\nNow, determine the output for: {format(q, '08b')}"
    )
    return {
        "seed": seed,
        "tag": f"{profile}/het",
        "examples": ex,
        "query": q,
        "cols": cols,
        "prompt": prompt,
        "answer": format(apply(q), "08b"),
    }


# Homogeneous-collapse probe: the best single GLOBAL rule fit to the examples.
# A het-failing model that assumed homogeneity would land on this answer.
def _best_hom_answer(ex, ex_out, q):
    """Fit the best homogeneous rule (pairwise global op+stride, or rotation) to
    the examples; return its query output (08b) and its example-mismatch count."""
    candidates = []
    for op in _PAIR_OPS:
        for a in range(8):
            for b in range(8):
                if a != b:
                    candidates.append(
                        [("pair", op, (a + j) % 8, (b + j) % 8) for j in range(8)]
                    )
    for k in range(1, 8):
        for neg in (False, True):
            candidates.append([("route", (j + k) % 8, neg) for j in range(8)])
    best, best_miss = None, 10**9
    for cand in candidates:
        miss = 0
        for x, gold in zip(ex, ex_out):
            pred = _pack([_eval_col(c, x) for c in cand])
            miss += bin(pred ^ gold).count("1")
        if miss < best_miss:
            best_miss, best = miss, cand
    q_out = _pack([_eval_col(c, q) for c in best])
    return format(q_out, "08b"), best_miss


# %% Cell 4: run on the live adapter + diagnose ------------------------------
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

assert "llm" in globals(), "Run AFTER the notebook's 'Init vLLM' cell defines `llm`."
_tok = llm.get_tokenizer()  # noqa: F821  (llm is a notebook global)
_sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=7680)
_lora = LoRARequest("adapter", 1, ADAPTER_PATH)
_BOXED = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"


def predict(prompts):
    rendered = []
    for p in prompts:
        try:
            r = _tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError):
            r = p
        rendered.append(r)
    outs = llm.generate(  # noqa: F821  (llm is a notebook global)
        rendered, sampling_params=_sp, lora_request=_lora
    )
    return [o.outputs[0].text for o in outs]


# Collect N_HET het problems.
problems, seed = [], 0
while len(problems) < N_HET:
    pr = gen_het(seed, BIT_DIFFICULTY)
    if pr is not None:
        problems.append(pr)
    seed += 1

rows = []
agg = defaultdict(lambda: defaultdict(float))
for bi in range(0, len(problems), BATCH_SIZE):
    batch = problems[bi : bi + BATCH_SIZE]
    raws = predict([p["prompt"] + _BOXED for p in batch])
    for p, raw in zip(batch, raws):
        pred = extract_final_answer(raw)
        gold = p["answer"]
        ok, ok_s = verify(gold, pred), verify_strict(gold, pred)
        well_formed = bool(re.fullmatch(r"[01]{8}", pred.strip()))
        hi_wrong = lo_wrong = hamming = -1
        hom_match = False
        if well_formed:
            pb = pred.strip()
            hamming = sum(a != b for a, b in zip(pb, gold))
            hi_wrong = int(pb[0] != gold[0]) + int(
                pb[1] != gold[1]
            )  # bits the grader needs
            lo_wrong = sum(pb[j] != gold[j] for j in range(2, 8))
            ex_out = [
                _pack([_eval_col(c, x) for c in p["cols"]]) for x in p["examples"]
            ]
            hom_ans, hom_miss = _best_hom_answer(p["examples"], ex_out, p["query"])
            hom_match = pb == hom_ans
        t = p["tag"]
        agg[t]["n"] += 1
        agg[t]["lenient"] += ok
        agg[t]["strict"] += ok_s
        agg[t]["truncated"] += not well_formed
        agg[t]["hom_collapse"] += hom_match
        if well_formed:
            agg[t]["hamming"] += hamming
            agg[t]["hi_wrong"] += hi_wrong
            agg[t]["wf"] += 1
            if not ok:
                agg[t]["fail_hom_collapse"] += hom_match
                agg[t]["fail_n"] += 1
        rows.append(
            {
                "seed": p["seed"],
                "tag": t,
                "gold": gold,
                "pred": pred,
                "lenient": ok,
                "strict": ok_s,
                "well_formed": well_formed,
                "hamming": hamming,
                "hi_wrong": hi_wrong,
                "lo_wrong": lo_wrong,
                "hom_collapse": hom_match,
                "raw_len": len(raw),
            }
        )
    print(f"  [{min(bi + BATCH_SIZE, len(problems))}/{len(problems)}] done", flush=True)

print("\n" + "=" * 78)
print("HET BIT FAILURE DUMP — model-error characterization")
print("=" * 78)
print(
    f"{'tag':<14} {'N':>4} {'len%':>6} {'str%':>6} {'trunc%':>7} "
    f"{'meanHam':>8} {'hiWrong%':>9} {'homColl%':>9} {'failHom%':>9}"
)
for t in sorted(agg):
    d = agg[t]
    n, wf, fn = d["n"], max(d["wf"], 1), max(d["fail_n"], 1)
    print(
        f"{t:<14} {int(n):>4} {100 * d['lenient'] / n:>6.1f} "
        f"{100 * d['strict'] / n:>6.1f} {100 * d['truncated'] / n:>7.1f} "
        f"{d['hamming'] / wf:>8.2f} {100 * d['hi_wrong'] / (2 * wf):>9.1f} "
        f"{100 * d['hom_collapse'] / n:>9.1f} {100 * d['fail_hom_collapse'] / fn:>9.1f}"
    )
print(
    "\nlen%/str% should reproduce ~46.7/~3.3 (pairwise/het) -> validates the "
    "model-free probe.\nhiWrong% = how often the 2 high-order bits (the ones the "
    "lenient grader needs) are wrong.\nhomColl%/failHom% high => model assumes a "
    "single global rule => bit-het SFT targets exactly this."
)

with open(DUMP_PATH, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"\nwrote {len(rows)} rows -> {DUMP_PATH}")
