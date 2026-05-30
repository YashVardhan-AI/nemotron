"""Local, GPU-free smoke test of the full harness with an oracle predictor."""

from pathlib import Path

from val.generators.cipher import generate, rule_signature
from val.holdout_registry import HoldoutRegistry
from val.report import aggregate, format_table, write_json
from val.scoring import BOXED_INSTRUCTION, score


def _oracle_predictor_factory(problems):
    """A predictor that returns each problem's known answer.

    run_demo holds the generated Problems, so the oracle looks the answer up by
    prompt rather than re-deriving the rule. This deterministically exercises the
    full scoring path (prompt routing -> extraction -> verify -> report) without
    depending on alphabet coverage. The real model signal comes from Task 7.
    """
    by_prompt = {p.prompt + BOXED_INSTRUCTION: p for p in problems}

    def predict(user_contents):
        return [
            f"reasoning \\boxed{{{by_prompt[content].answer}}}"
            for content in user_contents
        ]

    return predict


def run_demo(
    n: int = 5,
    difficulty: int = 4,
    holdout_path: Path = Path("val/holdout_rules.json"),
    report_path: Path = Path("val/demo_report.json"),
) -> dict:
    registry = HoldoutRegistry(holdout_path)
    problems = []
    for seed in range(n):
        problems.append(generate(seed=seed, difficulty=difficulty))
        registry.reserve("cipher", rule_signature(seed))

    difficulty_map = {p.id: len(p.examples) for p in problems}
    predictor = _oracle_predictor_factory(problems)
    results = score(problems, predictor, difficulty=difficulty_map)

    report = aggregate(results)
    write_json(report, report_path)
    return report


if __name__ == "__main__":
    report = run_demo()
    print(format_table(report))
