"""Per-category synthetic new-rule generators.

A generator module exposes and self-registers:
    generate(seed: int, difficulty: int) -> reasoners.store_types.Problem
    rule_signature(seed: int) -> str   # stable id of the rule, for holdout reservation
"""

from collections.abc import Callable
from dataclasses import dataclass

from reasoners.store_types import Problem


@dataclass
class GeneratorSpec:
    generate: Callable[[int, int], Problem]
    rule_signature: Callable[[int], str]


GENERATORS: dict[str, GeneratorSpec] = {}


def register(
    category: str,
    generate: Callable[[int, int], Problem],
    rule_signature: Callable[[int], str],
) -> None:
    GENERATORS[category] = GeneratorSpec(generate, rule_signature)


# Import generator modules so they self-register on `import val.generators`.
from val.generators import cipher  # noqa: E402,F401
