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
