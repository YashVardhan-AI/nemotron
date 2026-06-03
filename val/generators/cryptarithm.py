"""Reference generator: a brand-new arithmetic-cipher cryptarithm problem.

Calibrated to the measured real cryptarithm_deduce distribution (see
docs/superpowers/plans/2026-06-02-cryptarithm-p1.md): 5-char inputs
`s0 s1 <op> s3 s4`, operands are two-digit numbers, each operator glyph remaps to
one of five integer ops (deduced per problem), results re-encoded to glyphs. This
category is ~0% solved by the deterministic solver (it only handles
concatenation), so this is the clean new-rule signal on the #1 lever.

Thin wrapper over reasoners.cryptarithm_rule -- the SAME `sample_problem` the
training emitter uses, so the measured number and the trained data cannot drift.
Always renders the exact real wrapper (wrapper_index=0) so the eval reflects the
real prompt format. Generated instances are well-determined by construction.
"""

from reasoners.cryptarithm_deduce_core import sample_solvable
from reasoners.cryptarithm_rule import build_rule, render_prompt
from reasoners.store_types import Example, Problem
from val.generators import register


def rule_signature(seed: int) -> str:
    rule = build_rule(seed)
    smap = ",".join(f"{d}{rule.digit_to_sym[d]}" for d in range(10))
    osem = ",".join(f"{op}:{rule.op_of[op]}" for op in sorted(rule.op_of))
    return f"{smap}|{osem}"


def generate(seed: int, difficulty: int) -> Problem:
    # sample_solvable filters to UNIQUELY-deducible instances (the solver recovers
    # exactly the planted answer), so the eval never penalizes the model for
    # under-determined problems no one could solve.
    _rule, raw, q_input, q_answer = sample_solvable(seed, difficulty)
    examples = [Example(input_value=i, output_value=o) for i, o in raw]
    prompt = render_prompt(raw, q_input, wrapper_index=0)
    return Problem(
        id=f"val-cryptarithm-{seed}",
        category="cryptarithm_deduce",
        examples=examples,
        question=q_input,
        answer=q_answer,
        prompt=prompt,
    )


register("cryptarithm_deduce", generate, rule_signature)
