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
ADAPTER_PATH = (
    "/kaggle/working"  # dir with adapter_config.json + adapter_model.safetensors
)
PER_CATEGORY = 50  # how many distinct new rules per generator
# Match the MEASURED real example counts per category (problems.jsonl): bit
# problems give 7-10 examples (avg 8.6), cipher gives 3-5 (avg 4.0). Running both
# at a flat 6 under-determined bit (pessimistic) and over-fed cipher.
CIPHER_DIFFICULTY = 4  # real cipher: 3-5 examples
BIT_DIFFICULTY = 8  # real bit: 7-10 examples
CRYPTARITHM_DIFFICULTY = 4  # real cryptarithm_deduce: 3-5 examples (mean ~4.1)
BATCH_SIZE = 64  # problems per batched llm.generate() call; = vLLM max_num_seqs, so
# the scheduler stays full (throughput ~= one big call); accuracy prints per batch
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
    meta: str = ""  # rule-family tag (e.g. "pairwise"/"rot"/"complex"), for breakdowns


# --- cipher: a brand-new substitution alphabet per seed ----------------------
# EXACT 77-word Wonderland vocabulary from reasoners/wonderland.txt (inlined here
# since this cell is self-contained). The real cipher task is ~100% solvable only
# because ~47% of queries need vocab recognition to fill unseen-letter positions;
# any out-of-vocab word makes those queries unsolvable (the earlier 35-word list
# had 5 OOD words -> a misleading ~78%).
_CIPHER_WORDS = [
    "above",
    "alice",
    "ancient",
    "around",
    "beyond",
    "bird",
    "book",
    "bright",
    "castle",
    "cat",
    "cave",
    "chases",
    "clever",
    "colorful",
    "creates",
    "crystal",
    "curious",
    "dark",
    "discovers",
    "door",
    "dragon",
    "draws",
    "dreams",
    "explores",
    "follows",
    "forest",
    "found",
    "garden",
    "golden",
    "hatter",
    "hidden",
    "imagines",
    "in",
    "inside",
    "island",
    "key",
    "king",
    "knight",
    "library",
    "magical",
    "map",
    "message",
    "mirror",
    "mountain",
    "mouse",
    "mysterious",
    "near",
    "ocean",
    "palace",
    "potion",
    "princess",
    "puzzle",
    "queen",
    "rabbit",
    "reads",
    "school",
    "secret",
    "sees",
    "silver",
    "story",
    "strange",
    "student",
    "studies",
    "teacher",
    "the",
    "through",
    "tower",
    "treasure",
    "turtle",
    "under",
    "valley",
    "village",
    "watches",
    "wise",
    "wizard",
    "wonderland",
    "writes",
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
        meta="substitution",
    )


# --- bit_manipulation: rules from the REAL grammar (see analysis) ------------
# Real train distribution (1,602 problems): ~65% pairwise 2-input boolean,
# ~20% routing (rotation, rarely an arbitrary permutation), ~15% complex 3-input
# (majority / choice / arbitrary truth table). affine/xor-mask are ~0% of real.
# CRUCIAL: real rules are HETEROGENEOUS across the 8 output bits (42.6% mix >=2
# ops, 48.2% vary operand offset by column), so each output bit is an INDEPENDENT
# function of arbitrary input bits — NOT one global op at one global offset.
# Logic is kept identical to val/generators/bit_manipulation.py.
_BIT_HEADER = (
    "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit "
    "binary numbers. The transformation involves operations like bit shifts, "
    "rotations, XOR, AND, OR, NOT, and possibly majority or choice functions."
)
_BIT_PAIR_OPS = ["AND", "OR", "XOR", "AND-NOT", "OR-NOT", "XOR-NOT"]


def _bitval(x, p):
    return (x >> (7 - p)) & 1


def _pack(bits):
    out = 0
    for j in range(8):
        out |= bits[j] << (7 - j)
    return out


def _bit_eval_column(col, x):
    kind = col[0]
    if kind == "const":
        return col[1]
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
        if base == "AND":
            return u & v
        if base == "OR":
            return u | v
        return u ^ v
    if kind == "maj":
        _, p, q, r = col
        return 1 if (_bitval(x, p) + _bitval(x, q) + _bitval(x, r)) >= 2 else 0
    if kind == "choice":
        _, p, q, r = col
        return _bitval(x, q) if _bitval(x, p) else _bitval(x, r)
    # tt3: arbitrary 3-var truth table
    _, p, q, r, table = col
    idx = (_bitval(x, p) << 2) | (_bitval(x, q) << 1) | _bitval(x, r)
    return (table >> idx) & 1


def _bit_rand_pair_column(rng):
    op = rng.choice(_BIT_PAIR_OPS)
    p = rng.randint(0, 7)
    q = rng.randint(0, 7)
    while q == p:
        q = rng.randint(0, 7)
    return ("pair", op, p, q)


def _bit_rand_route_column(rng):
    return ("route", rng.randint(0, 7), rng.random() < 0.3)


def _bit_rand_three_column(rng):
    kind = rng.choices(("choice", "maj", "tt3"), weights=(0.45, 0.20, 0.35))[0]
    p, q, r = rng.sample(range(8), 3)
    if kind == "maj":
        return ("maj", p, q, r)
    if kind == "choice":
        return ("choice", p, q, r)
    return ("tt3", p, q, r, rng.randint(1, 254))


def _bit_build_columns(rng, profile):
    cols = []
    if profile == "pairwise":
        for _ in range(8):
            roll = rng.random()
            if roll < 0.12:
                cols.append(_bit_rand_route_column(rng))
            elif roll < 0.20:
                cols.append(("const", rng.randint(0, 1)))
            else:
                cols.append(_bit_rand_pair_column(rng))
    elif profile == "rot":
        glob_neg = rng.random() < 0.3
        if rng.random() < 0.90:
            k = rng.randint(1, 7)
            cols = [("route", (j + k) % 8, glob_neg) for j in range(8)]
        else:
            perm = list(range(8))
            rng.shuffle(perm)
            cols = [("route", perm[j], glob_neg) for j in range(8)]
    else:  # complex
        three = set(rng.sample(range(8), rng.randint(1, 4)))
        for j in range(8):
            if j in three:
                cols.append(_bit_rand_three_column(rng))
            elif rng.random() < 0.15:
                cols.append(_bit_rand_route_column(rng))
            else:
                cols.append(_bit_rand_pair_column(rng))
    return cols


def build_bit_rule(seed):
    """Return (family, apply) for the rule selected by *seed*.

    family is one of 'pairwise' / 'rot' / 'complex'; apply(x:int)->int. Each
    output bit is an INDEPENDENT function of arbitrary input bits.
    """
    rng = random.Random(seed)
    roll = rng.random()
    profile = "pairwise" if roll < 0.65 else ("rot" if roll < 0.85 else "complex")
    cols = _bit_build_columns(rng, profile)

    def apply(x):
        return _pack([_bit_eval_column(c, x) for c in cols])

    return profile, apply


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
    family, apply = build_bit_rule(seed)
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
        id=f"val-bit_manipulation-{family}-{seed}",
        category="bit_manipulation",
        prompt=prompt,
        answer=format(apply(query_int), "08b"),
        n_examples=difficulty,
        meta=family,
    )


# --- cryptarithm_deduce: arithmetic-cipher family (the #1 lever) -------------
# Rule logic mirrors reasoners/cryptarithm_rule.py EXACTLY so local and Kaggle
# numbers agree (same seeds -> same problems). A rule = injective glyph<->digit
# map x per-operator-glyph op (add/abs_diff/mul/concat/rev_concat) x re-encoding.
# 5-char inputs `s0 s1 op s3 s4`; operands are two-digit numbers; concat/rev_concat
# pad to 4 digits, arithmetic ops are natural length. Operators are mostly +,-,*
# with a calibrated long tail (may also encode a digit; the role is positional).
# This category is ~0% solved by the deterministic solver, so it is the clean
# new-rule signal on the #1 lever.
_CRYPT_SYMBOLS = list("!\"#$%&'()/:<>?@[]^`{|}\\")
_CRYPT_BASE_OPERATORS = ("+", "-", "*")
_CRYPT_TAIL_OPERATOR_PROB = 0.15
_CRYPT_OP_NAMES = ("add", "abs_diff", "mul", "concat", "rev_concat")
_CRYPT_OPS = {
    "add": lambda a, b: a + b,
    "abs_diff": lambda a, b: abs(a - b),
    "mul": lambda a, b: a * b,
    "concat": lambda a, b: a * 100 + b,
    "rev_concat": lambda a, b: b * 100 + a,
}
_CRYPT_OP_WEIGHTS = {"add": 3, "abs_diff": 3, "mul": 3, "concat": 2, "rev_concat": 1}
_CRYPT_WRAPPERS = [
    (
        "In Alice's Wonderland, a secret set of transformation rules is applied "
        "to equations. Below are a few examples:",
        "Now, determine the result for: ",
    ),
    (
        "In Wonderland, a hidden set of transformation rules is applied to each "
        "equation. Study these examples:",
        "Using the same rules, determine the result for: ",
    ),
    (
        "A secret set of rules transforms equations in Wonderland. Here are some "
        "examples:",
        "Now find the result for: ",
    ),
]


def _crypt_num_to_digits(n):
    if n == 0:
        return (0,)
    d = []
    while n > 0:
        d.append(n % 10)
        n //= 10
    return tuple(reversed(d))


@dataclass
class _CryptRule:
    digit_to_sym: dict
    sym_to_digit: dict
    op_of: dict

    @property
    def operators(self):
        return tuple(self.op_of)

    def result_digits(self, op, left, right):
        name = self.op_of[op]
        val = _CRYPT_OPS[name](left, right)
        if name in ("concat", "rev_concat"):
            return (val // 1000, (val // 100) % 10, (val // 10) % 10, val % 10)
        return _crypt_num_to_digits(val)

    def encode_output(self, digits):
        return "".join(self.digit_to_sym[d] for d in digits)

    def encode_example(self, op, left, right):
        lnum, rnum = 10 * left[0] + left[1], 10 * right[0] + right[1]
        inp = (
            self.digit_to_sym[left[0]]
            + self.digit_to_sym[left[1]]
            + op
            + self.digit_to_sym[right[0]]
            + self.digit_to_sym[right[1]]
        )
        return inp, self.encode_output(self.result_digits(op, lnum, rnum))


def _crypt_build_rule(seed):
    rng = random.Random(seed)
    glyphs = rng.sample(_CRYPT_SYMBOLS, 10)
    digit_to_sym = {d: glyphs[d] for d in range(10)}
    sym_to_digit = {s: d for d, s in digit_to_sym.items()}
    operators = list(_CRYPT_BASE_OPERATORS)
    if rng.random() < _CRYPT_TAIL_OPERATOR_PROB:
        operators.append(rng.choice(_CRYPT_SYMBOLS))
    names = list(_CRYPT_OP_WEIGHTS)
    weights = [_CRYPT_OP_WEIGHTS[n] for n in names]
    op_of = {op: rng.choices(names, weights=weights)[0] for op in operators}
    return _CryptRule(digit_to_sym, sym_to_digit, op_of)


_CRYPT_PROFILE_OPS = {
    "easy": ("add", "concat", "rev_concat"),
    "hard": ("mul", "abs_diff"),
}
_CRYPT_PROFILE_RANGE = {"easy": (0, 5), "hard": (0, 9)}


def _crypt_sample_problem(seed, difficulty=4, determinacy="well", profile=None, salt=0):
    rng = random.Random(seed * 7919 + 1 + salt * 104729)
    rule = _crypt_build_rule(seed)
    ops = rule.operators
    lo_d, hi_d = _CRYPT_PROFILE_RANGE.get(profile or "", (0, 9))

    def digit():
        return rng.randint(lo_d, hi_d)

    if profile in _CRYPT_PROFILE_OPS:
        pref = [
            op for op in ops if rule.op_of[op] in _CRYPT_PROFILE_OPS[profile]
        ] or list(ops)
    else:
        pref = list(ops)
    q_op = rng.choice(pref)
    q_left = (digit(), digit())
    q_right = (digit(), digit())
    q_input, q_answer = rule.encode_example(q_op, q_left, q_right)

    needed = sorted(
        {q_left[0], q_left[1], q_right[0], q_right[1]}
        | {rule.sym_to_digit[g] for g in q_answer}
    )
    total_slots = difficulty * 4
    if len(needed) > total_slots:
        needed = needed[:total_slots]
    digit_seq = list(needed)
    if determinacy != "near":
        while len(digit_seq) < total_slots:
            digit_seq.append(digit())
    else:
        while len(digit_seq) < total_slots:
            digit_seq.append(rng.choice(needed) if needed else digit())
    rng.shuffle(digit_seq)

    triples = []
    for i in range(difficulty):
        chunk = digit_seq[i * 4 : i * 4 + 4]
        left = (chunk[0], chunk[1])
        right = (chunk[2], chunk[3])
        op = q_op if i < 2 else rng.choice(ops)  # first two demos witness query op
        triples.append((op, left, right))
    rng.shuffle(triples)
    examples = [rule.encode_example(op, lo, ro) for op, lo, ro in triples]
    return rule, examples, q_input, q_answer


def _crypt_render_prompt(examples, q_input, wrapper_index=0):
    header, lead = _CRYPT_WRAPPERS[wrapper_index % len(_CRYPT_WRAPPERS)]
    lines = "\n".join(f"{i} = {o}" for i, o in examples)
    return f"{header}\n{lines}\n{lead}{q_input}"


# --- cryptarithm solver (inlined from reasoners/cryptarithm_deduce_core.py) ---
# Filters generated instances to UNIQUELY-deducible ones, so the eval never
# penalizes the model for under-determined problems no one could solve. Mirrors
# the repo solver EXACTLY so local and Kaggle problem sets agree.
from collections import Counter

_CRYPT_OP_LIST = [
    _CRYPT_OPS["add"],
    _CRYPT_OPS["abs_diff"],
    _CRYPT_OPS["mul"],
    _CRYPT_OPS["concat"],
    _CRYPT_OPS["rev_concat"],
]
_CRYPT_OP_NAME_LIST = list(_CRYPT_OP_NAMES)
_CRYPT_NODE_CAP = 2_000_000


def _crypt_is_concat(ex):
    s0, s1, _op, s3, s4, rsyms = ex
    return rsyms == (s0, s1, s3, s4) or rsyms == (s3, s4, s0, s1)


class _CryptSolver:
    def __init__(self, examples, query, unique=True):
        self.examples = examples
        self.query = query
        self.unique = unique
        self.mapping = {}
        self.used = set()
        self.op_assign = {}
        self.answers = Counter()
        self.answer_info = {}
        self.max_solutions = 200
        self.nodes = 0

    def solve_counted(self):
        self._process(0)
        if self.answers:
            best, _ = self.answers.most_common(1)[0]
            return best, self.answer_info.get(best, ({}, {})), len(self.answers)
        return None, ({}, {}), 0

    def _process(self, idx):
        if len(self.answers) >= self.max_solutions:
            return
        self.nodes += 1
        if self.nodes > _CRYPT_NODE_CAP:
            return
        if idx == len(self.examples):
            self._compute_query()
            return
        s0, s1, op_sym, s3, s4, rsyms = self.examples[idx]
        rlen = len(rsyms)
        feasible_ops = []
        if rlen <= 3:
            feasible_ops.append(0)
        if rlen <= 2:
            feasible_ops.append(1)
        if rlen <= 4:
            feasible_ops.append(2)
        if rlen == 4:
            feasible_ops.extend([3, 4])
        for d0 in self._vals(s0):
            n0 = self._assign(s0, d0)
            if n0 is None:
                continue
            for d1 in self._vals(s1):
                n1 = self._assign(s1, d1)
                if n1 is None:
                    continue
                lv = d0 * 10 + d1
                for d3 in self._vals(s3):
                    n3 = self._assign(s3, d3)
                    if n3 is None:
                        continue
                    for d4 in self._vals(s4):
                        n4 = self._assign(s4, d4)
                        if n4 is None:
                            continue
                        rv = d3 * 10 + d4
                        ops_to_try = (
                            [self.op_assign[op_sym]]
                            if op_sym in self.op_assign
                            else feasible_ops
                        )
                        for op_id in ops_to_try:
                            result_val = _CRYPT_OP_LIST[op_id](lv, rv)
                            if op_id >= 3:
                                if result_val < 0 or result_val >= 10000:
                                    continue
                                rd = (
                                    result_val // 1000,
                                    (result_val // 100) % 10,
                                    (result_val // 10) % 10,
                                    result_val % 10,
                                )
                            else:
                                rd = _crypt_num_to_digits(result_val)
                            if len(rd) != rlen:
                                continue
                            assigns = []
                            ok = True
                            for rs, rdig in zip(rsyms, rd):
                                ns = self._assign(rs, rdig)
                                if ns is None:
                                    ok = False
                                    break
                                assigns.append((rs, ns))
                            if ok:
                                op_new = op_sym not in self.op_assign
                                if op_new:
                                    self.op_assign[op_sym] = op_id
                                self._process(idx + 1)
                                if op_new:
                                    del self.op_assign[op_sym]
                            for rs, ns in reversed(assigns):
                                self._undo(rs, ns)
                            if len(self.answers) >= self.max_solutions:
                                self._undo(s4, n4)
                                self._undo(s3, n3)
                                self._undo(s1, n1)
                                self._undo(s0, n0)
                                return
                        self._undo(s4, n4)
                    self._undo(s3, n3)
                self._undo(s1, n1)
            self._undo(s0, n0)

    def _vals(self, sym):
        if sym in self.mapping:
            return (self.mapping[sym],)
        if self.unique:
            return tuple(d for d in range(10) if d not in self.used)
        return range(10)

    def _assign(self, sym, dig):
        if sym in self.mapping:
            return False if self.mapping[sym] == dig else None
        if self.unique and dig in self.used:
            return None
        self.mapping[sym] = dig
        if self.unique:
            self.used.add(dig)
        return True

    def _undo(self, sym, was_new):
        if was_new is True:
            if self.unique:
                self.used.discard(self.mapping[sym])
            del self.mapping[sym]

    def _compute_query(self):
        qs0, qs1, qop, qs3, qs4 = self.query
        for s in (qs0, qs1, qs3, qs4):
            if s not in self.mapping:
                return
        ql = self.mapping[qs0] * 10 + self.mapping[qs1]
        qr = self.mapping[qs3] * 10 + self.mapping[qs4]
        op_candidates = (
            [self.op_assign[qop]]
            if qop in self.op_assign
            else range(len(_CRYPT_OP_NAME_LIST))
        )
        d2s = {}
        for s, d in self.mapping.items():
            if d not in d2s:
                d2s[d] = s
        for op_id in op_candidates:
            result_val = _CRYPT_OP_LIST[op_id](ql, qr)
            if op_id >= 3:
                if result_val < 0 or result_val >= 10000:
                    continue
                rd = (
                    result_val // 1000,
                    (result_val // 100) % 10,
                    (result_val // 10) % 10,
                    result_val % 10,
                )
            else:
                rd = _crypt_num_to_digits(result_val)
            parts = []
            ok = True
            for d in rd:
                if d not in d2s:
                    ok = False
                    break
                parts.append(d2s[d])
            if not ok:
                continue
            ans = "".join(parts)
            self.answers[ans] += 1
            if ans not in self.answer_info:
                self.answer_info[ans] = (dict(self.mapping), {})


def _crypt_solve_counted(examples_raw, q_input):
    examples = [(i[0], i[1], i[2], i[3], i[4], tuple(o)) for i, o in examples_raw]
    query = (q_input[0], q_input[1], q_input[2], q_input[3], q_input[4])
    concat_ops = {ex[2] for ex in examples if _crypt_is_concat(ex)}
    nonconcat_ops = {ex[2] for ex in examples if not _crypt_is_concat(ex)}
    if query[2] in concat_ops and query[2] not in nonconcat_ops:
        for ex in examples:
            if ex[2] == query[2] and _crypt_is_concat(ex):
                s0, s1, _, s3, s4, rsyms = ex
                if rsyms == (s0, s1, s3, s4):
                    return query[0] + query[1] + query[3] + query[4], 1
                return query[3] + query[4] + query[0] + query[1], 1
        return query[0] + query[1] + query[3] + query[4], 1
    arith = [ex for ex in examples if not _crypt_is_concat(ex)]
    ans, _info, n = _CryptSolver(arith, query, unique=True).solve_counted()
    return ans, n


def _crypt_is_uniquely_solvable(examples_raw, q_input, q_answer):
    ans, n = _crypt_solve_counted(examples_raw, q_input)
    return ans == q_answer and n == 1


def _crypt_sample_solvable(seed, difficulty=4, max_salt=40):
    last = None
    for salt in range(max_salt):
        rule, examples, q_input, q_answer = _crypt_sample_problem(
            seed, difficulty, salt=salt
        )
        last = (rule, examples, q_input, q_answer)
        if _crypt_is_uniquely_solvable(examples, q_input, q_answer):
            return rule, examples, q_input, q_answer
    return last


def generate_cryptarithm(seed, difficulty):
    rule, raw, q_input, q_answer = _crypt_sample_solvable(seed, difficulty)
    prompt = _crypt_render_prompt(raw, q_input, 0)
    return Problem(
        id=f"val-cryptarithm-{seed}",
        category="cryptarithm_deduce",
        prompt=prompt,
        answer=q_answer,
        n_examples=difficulty,
        # query op family (concat/rev_concat vs add/abs_diff/mul): the current
        # adapter only saw concat traces, so this isolates the easy concat slice
        # from the arithmetic headroom the deduction CoT targets.
        meta=rule.op_of[q_input[2]],
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


# %% ── Cell 5: run — reuse the notebook's `llm`, batched, accuracy per batch ──
import json

from vllm import SamplingParams
from vllm.lora.request import LoRARequest

# NOTE: this cell uses defaultdict / verify / generators etc. defined in Cells
# 2-4 — run the cells in order (they share the notebook's globals).

assert "llm" in globals(), "Run AFTER the notebook's 'Init vLLM' cell defines `llm`."

_tokenizer = llm.get_tokenizer()  # noqa: F821  (llm is a notebook global)
_sampling = SamplingParams(
    temperature=0.0, top_p=1.0, max_tokens=7680
)  # greedy, eval config
_lora = LoRARequest("adapter", 1, ADAPTER_PATH)


def predict(eval_prompts):
    """Batched: chat template (w/ thinking) -> greedy -> our adapter. vLLM's
    continuous batching runs the whole list together (fast)."""
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


problems = [generate_cipher(s, CIPHER_DIFFICULTY) for s in range(PER_CATEGORY)]
problems += [generate_bit(s, BIT_DIFFICULTY) for s in range(PER_CATEGORY)]
problems += [
    generate_cryptarithm(s, CRYPTARITHM_DIFFICULTY) for s in range(PER_CATEGORY)
]

# Score in batches so we keep vLLM's throughput but still see accuracy climb.
# Each batch is one batched llm.generate() call; bump BATCH_SIZE for fewer,
# bigger (faster) batches or lower it for more frequent updates.
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
                "difficulty": p.n_examples,
                "meta": p.meta,
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
        f"{c} {cat_hit[c] / cat_seen[c] * 100:.0f}% ({cat_hit[c]}/{cat_seen[c]})"
        for c in sorted(cat_seen)
    )
    print(
        f"[batch {bi // BATCH_SIZE + 1:>2}/{n_batches}] {seen:>4}/{n_total} done  "
        f"running {hit / seen * 100:5.1f}% ({hit}/{seen})  |  {per_cat}",
        flush=True,
    )

report = aggregate(results)
print("\n" + "=" * 64)
print("NEW-RULE generalization (synthetic, rules never trained on)")
print("=" * 64)
print(format_table(report))

# Per-rule-family breakdown for bit_manipulation (pairwise / rot / complex).
# Real data is ~65% pairwise, ~20% rot, ~15% complex; 'complex' (3-input
# majority/choice) is the genuinely hard tail. This isolates where the model
# actually fails vs the in-distribution families it should handle.
fam_seen = defaultdict(int)
fam_hit = defaultdict(int)
fam_hit_strict = defaultdict(int)
for r in results:
    if r["category"] != "bit_manipulation":
        continue
    fam_seen[r["meta"]] += 1
    fam_hit[r["meta"]] += int(r["correct"])
    fam_hit_strict[r["meta"]] += int(r["correct_strict"])
if fam_seen:
    print("\nbit_manipulation by rule family:")
    print(f"  {'family':<10} {'N':>4} {'Acc%':>7} {'StrictAcc%':>11}")
    for fam in sorted(fam_seen):
        n = fam_seen[fam]
        print(
            f"  {fam:<10} {n:>4} {fam_hit[fam] / n * 100:>7.1f} "
            f"{fam_hit_strict[fam] / n * 100:>11.1f}"
        )

# Per-query-op breakdown for cryptarithm_deduce. The current adapter trained on
# concat-only traces, so the baseline score should be concentrated in
# concat/rev_concat; add/abs_diff/mul (the arithmetic family) is the headroom the
# deduction CoT (Phase 4-5) targets. After retraining, watch arith climb from ~0.
cr_seen = defaultdict(int)
cr_hit = defaultdict(int)
cr_hit_strict = defaultdict(int)
for r in results:
    if r["category"] != "cryptarithm_deduce":
        continue
    cr_seen[r["meta"]] += 1
    cr_hit[r["meta"]] += int(r["correct"])
    cr_hit_strict[r["meta"]] += int(r["correct_strict"])
if cr_seen:
    print("\ncryptarithm_deduce by query op:")
    print(f"  {'op':<12} {'N':>4} {'Acc%':>7} {'StrictAcc%':>11}")
    for op in sorted(cr_seen):
        n = cr_seen[op]
        print(
            f"  {op:<12} {n:>4} {cr_hit[op] / n * 100:>7.1f} "
            f"{cr_hit_strict[op] / n * 100:>11.1f}"
        )

with open(OUT_JSON, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nwrote {OUT_JSON}")

# Compare against the notebook's TRAIN table (gravity/numeral/unit_conv ~100%,
# bit ~88%, cryptarithm ~8%). A big drop here = the generalization gap.
