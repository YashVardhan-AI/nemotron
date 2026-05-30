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
    except (ValueError, TypeError):
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
    except (ValueError, TypeError):
        return predicted.lower() == stored_answer.lower()
