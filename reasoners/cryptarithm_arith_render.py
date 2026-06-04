"""Render an integer op as a reversed-digit, carry-annotated column computation.

Why reversed (least-significant-first) with explicit carries: the literature
(Faith and Fate; "What Algorithms can Transformers Learn?"; Goat; Tokenization
counts) shows transformers do multi-digit arithmetic reliably only when each
output digit is a constant-token function of aligned operand digits + the prior
carry, and when digits are single tokens. So we never emit `36 * 47 = 1692`
atomically; we write the columns.

Pure: no I/O. Returns (value, text) where value ALWAYS equals Python's op (the
caller gates the whole trace on the boxed answer, but this keeps each line sound).
"""

from __future__ import annotations


def _digits_lsb(n: int) -> list[int]:
    """Digits of n, least-significant-first. _digits_lsb(0) == [0]."""
    if n == 0:
        return [0]
    out: list[int] = []
    while n > 0:
        out.append(n % 10)
        n //= 10
    return out


def _spaced_msb(digits_lsb: list[int]) -> str:
    """Space-separated digits, most-significant-first (one token each)."""
    return " ".join(str(d) for d in reversed(digits_lsb)) or "0"


def _render_columns(symbol: str, a: int, b: int, da: list[int], db: list[int]) -> str:
    width = max(len(da), len(db))
    da = da + [0] * (width - len(da))
    db = db + [0] * (width - len(db))
    carry = 0
    res: list[int] = []
    lines = [f"  {a} {symbol} {b}, least-significant digit first:"]
    for i in range(width):
        s = da[i] + db[i] + carry
        d, new_carry = s % 10, s // 10
        lines.append(
            f"    col{i}: {da[i]} {symbol} {db[i]} + carry {carry} = {s} "
            f"-> digit {d}, carry {new_carry}"
        )
        carry = new_carry
        res.append(d)
    if carry:
        res.append(carry)
    lines.append(f"    result digits (reversed): {_spaced_msb(res)} = {a + b}")
    return "\n".join(lines)


def _render_sub(a: int, b: int) -> tuple[int, str]:
    hi, lo = (a, b) if a >= b else (b, a)
    dh, dl = _digits_lsb(hi), _digits_lsb(lo)
    width = max(len(dh), len(dl))
    dh = dh + [0] * (width - len(dh))
    dl = dl + [0] * (width - len(dl))
    borrow = 0
    res: list[int] = []
    lines = [f"  |{a} - {b}| = {hi} - {lo}, least-significant digit first:"]
    for i in range(width):
        x = dh[i] - dl[i] - borrow
        if x < 0:
            x += 10
            new_borrow = 1
        else:
            new_borrow = 0
        lines.append(
            f"    col{i}: {dh[i]} - {dl[i]} - borrow {borrow} = {x} "
            f"-> digit {x}, borrow {new_borrow}"
        )
        borrow = new_borrow
        res.append(x)
    while len(res) > 1 and res[-1] == 0:
        res.pop()
    lines.append(f"    result digits (reversed): {_spaced_msb(res)} = {abs(a - b)}")
    return abs(a - b), "\n".join(lines)


def _render_mul(a: int, b: int) -> tuple[int, str]:
    # Single-digit partial products + shift-add (Goat decomposition).
    db = _digits_lsb(b)
    lines = [
        f"  mul {a} * {b}: sum of single-digit partial products; carry handled in the shift-add:"
    ]
    for i, d in enumerate(db):
        p = a * d * (10**i)
        lines.append(f"    {a} * {d} (digit {i} of {b}) shifted x10^{i} = {p}")
    lines.append(f"    sum of partials = {a * b}")
    return a * b, "\n".join(lines)


def long_arith(op: str, a: int, b: int) -> tuple[int, str]:
    """(value, rendered-columns-text) for op in {add, abs_diff, mul}."""
    if op == "add":
        return a + b, _render_columns("+", a, b, _digits_lsb(a), _digits_lsb(b))
    if op == "abs_diff":
        return _render_sub(a, b)
    if op == "mul":
        return _render_mul(a, b)
    raise ValueError(f"long_arith does not render op {op!r}")
