"""Tests for the teacher rejection gate (Phase 4.4) with a MOCKED teacher.

No live LLM call -- the teacher is a stub returning canned text per seed, so the
test exercises the real grader gate: correct boxed answer kept, wrong dropped,
malformed dropped. (In val/tests/ with the rest of the repo's tests.)
"""

from reasoners.cryptarithm_deduce_core import sample_solvable
from reasoners.cryptarithm_teacher import (
    accept_rate,
    build_teacher_prompt,
    rejection_sample,
)


def test_build_teacher_prompt_appends_instruction_not_answer():
    seeds = [0]
    _rule, raw, q_input, q_answer = sample_solvable(0, 4)
    from reasoners.cryptarithm_rule import render_prompt

    tp = build_teacher_prompt(render_prompt(raw, q_input, 0))
    assert "\\boxed{}" in tp
    assert q_answer not in tp  # the answer is never leaked to the teacher
    assert seeds  # silence unused


def test_gate_keeps_correct_drops_wrong_and_malformed():
    seeds = list(range(6))
    # ground-truth answers per seed, in order
    answers = [sample_solvable(s, 4)[3] for s in seeds]
    # teacher behaviour per seed: 0,3 correct; 1,4 wrong; 2,5 malformed
    plan = {}
    for i, s in enumerate(seeds):
        if i % 3 == 0:
            plan[s] = f"... reasoning ...\nThe answer is \\boxed{{{answers[i]}}}"
        elif i % 3 == 1:
            plan[s] = "... reasoning ...\n\\boxed{ZZZZ_definitely_wrong}"
        else:
            plan[s] = "... reasoning with no boxed answer at all ..."

    # map prompt -> seed so the stub returns the right canned text
    prompt_to_seed = {}
    for s in seeds:
        _r, raw, qi, _a = sample_solvable(s, 4)
        from reasoners.cryptarithm_rule import render_prompt

        prompt_to_seed[build_teacher_prompt(render_prompt(raw, qi, 0))] = s

    def teacher(prompt):
        return plan[prompt_to_seed[prompt]]

    traces = rejection_sample(seeds, teacher)
    by_seed = {int(t.problem_id.rsplit("-", 1)[1]): t for t in traces}
    for i, s in enumerate(seeds):
        if i % 3 == 0:
            assert by_seed[s].accepted, (s, by_seed[s].predicted, by_seed[s].answer)
        else:
            assert not by_seed[s].accepted, s


def test_accept_rate_matches_expected_third():
    seeds = list(range(6))
    answers = [sample_solvable(s, 4)[3] for s in seeds]
    from reasoners.cryptarithm_rule import render_prompt

    prompt_to_ans = {}
    for s, a in zip(seeds, answers):
        _r, raw, qi, _a = sample_solvable(s, 4)
        prompt_to_ans[build_teacher_prompt(render_prompt(raw, qi, 0))] = a

    # accept exactly the first two seeds (return correct), reject the rest
    accept_set = set(list(prompt_to_ans)[:2])

    def teacher(prompt):
        if prompt in accept_set:
            return f"\\boxed{{{prompt_to_ans[prompt]}}}"
        return "\\boxed{nope}"

    traces = rejection_sample(seeds, teacher)
    assert abs(accept_rate(traces) - (2 / 6)) < 1e-9
