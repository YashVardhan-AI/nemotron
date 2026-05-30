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
