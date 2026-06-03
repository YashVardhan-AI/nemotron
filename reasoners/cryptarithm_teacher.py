"""Teacher-LLM rejection-sampled traces for cryptarithm_deduce (Phase 4.4).

Don't bet on one templated CoT: use a strong external teacher to write natural
induction derivations, then KEEP ONLY traces whose boxed answer matches the
constructed answer (rejection sampling via the real grader). External teacher
LLMs are allowed for data generation; the eval stays greedy.

The teacher call is INJECTED (a callable str -> str), so this module has no API
dependency and is fully testable with a mock; the caller wires up the actual LLM
(OpenAI / Anthropic / local vLLM, temp > 0) at data-gen time. Holdout filtering
is the caller's job (pre-filter seeds whose rule_signature is reserved) -- keeps
this module free of any val import. The grader is lazy-imported (or injected) so
importing this module never pulls in val.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from reasoners.cryptarithm_deduce_core import sample_solvable
from reasoners.cryptarithm_rule import render_prompt

TeacherFn = Callable[[str], str]

# Appended to the problem prompt; asks for induction + a boxed answer. The rule
# is NOT given -- the teacher must induce it (so the kept traces teach induction).
_INSTRUCTION = (
    "\n\nThink step by step: induce the hidden rule from the examples (what each "
    "operator does, and which digit each symbol stands for), verify it against "
    "every example, then apply it to the query. Put your final answer in "
    "\\boxed{}."
)


@dataclass
class TeacherTrace:
    problem_id: str
    prompt: str  # the problem prompt (no instruction) -- what training stores
    completion: str  # the teacher's raw reasoning text
    answer: str  # the constructed (ground-truth) answer
    predicted: str  # answer extracted from the teacher text
    accepted: bool  # predicted matches answer under the real grader


def build_teacher_prompt(problem_prompt: str) -> str:
    """The problem prompt + an induction instruction (no rule, no answer)."""
    return problem_prompt + _INSTRUCTION


def _default_grader() -> tuple[Callable[[str | None], str], Callable[[str, str], bool]]:
    from val.grading import extract_final_answer, verify  # lazy: no val at import

    return extract_final_answer, verify


def rejection_sample(
    seeds: Iterable[int],
    teacher_fn: TeacherFn,
    *,
    difficulty: int = 4,
    extract: Callable[[str | None], str] | None = None,
    verify_fn: Callable[[str, str], bool] | None = None,
) -> list[TeacherTrace]:
    """Generate a teacher trace per seed and gate it by the real grader.

    seeds: rule seeds to use (caller pre-filters holdout-reserved signatures).
    teacher_fn: str (teacher prompt) -> str (raw reasoning ending in \\boxed{}).
    Returns one TeacherTrace per seed; filter on `.accepted` for training rows.
    """
    if extract is None or verify_fn is None:
        extract, verify_fn = _default_grader()

    traces: list[TeacherTrace] = []
    for seed in seeds:
        _rule, raw, q_input, q_answer = sample_solvable(seed, difficulty)
        prompt = render_prompt(raw, q_input, wrapper_index=0)
        text = teacher_fn(build_teacher_prompt(prompt))
        predicted = extract(text)
        traces.append(
            TeacherTrace(
                problem_id=f"teacher-cryptarithm-{seed}",
                prompt=prompt,
                completion=text,
                answer=q_answer,
                predicted=predicted,
                accepted=verify_fn(q_answer, predicted),
            )
        )
    return traces


def accept_rate(traces: list[TeacherTrace]) -> float:
    """Fraction accepted (measure this on a real run -- it sets the economics)."""
    if not traces:
        return 0.0
    return sum(t.accepted for t in traces) / len(traces)
