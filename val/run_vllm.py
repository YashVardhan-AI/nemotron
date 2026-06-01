"""Real-path predictor for the validation harness (GPU/Kaggle only).

Mirrors notebook_tinker.generate_predictions: chat template with thinking,
greedy decoding, rank-32 LoRA adapter. Not imported by local tests.

Usage (on a GPU box):
    uv run python -m val.run_vllm --model <BASE_MODEL_PATH> --adapter <ADAPTER_DIR>
"""

import argparse
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path

from reasoners.store_types import Problem
from val.generators import GENERATORS
from val.holdout_registry import HoldoutRegistry
from val.real_holdout import holdout_problems
from val.report import aggregate, format_table, write_json
from val.scoring import score


def make_vllm_predictor(
    model_path: str,
    adapter_path: str,
    max_lora_rank: int = 32,
    max_tokens: int = 7680,
    max_model_len: int = 8192,
) -> Callable[[list[str]], list[str]]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="auto",
        max_model_len=max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)

    def predict(user_contents: list[str]) -> list[str]:
        prompts = []
        for content in user_contents:
            # nb: a broad bare-except is blocked by the repo hook; this explicit
            # tuple covers the known apply_chat_template failure modes. A malformed
            # Jinja template (jinja2.TemplateError) would fall through, but the
            # eval models have well-formed templates.
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError):
                prompt = content
            prompts.append(prompt)
        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=LoRARequest("adapter", 1, adapter_path),
        )
        return [o.outputs[0].text for o in outputs]

    return predict


def build_synthetic_valset(
    per_category: int, difficulty: int, registry: HoldoutRegistry
) -> list[Problem]:
    """Generate fresh new-rule problems from every registered generator and
    reserve each rule's signature so it can never enter training generation."""
    problems = []
    for category, spec in sorted(GENERATORS.items()):
        for seed in range(per_category):
            problems.append(spec.generate(seed, difficulty))
            registry.reserve(category, spec.rule_signature(seed))
    return problems


def cap_per_category(problems: list[Problem], limit: int | None) -> list[Problem]:
    """Keep at most *limit* problems, sampled round-robin across categories.

    A flat head-slice could return a single category; round-robin keeps a small
    --limit sample category-balanced. Deterministic (categories sorted, original
    order preserved within each). Returns *problems* unchanged if limit is None
    or already within the cap.
    """
    if limit is None or len(problems) <= limit:
        return problems
    buckets: dict[str, deque[Problem]] = defaultdict(deque)
    for p in problems:
        buckets[p.category].append(p)
    queues = [buckets[c] for c in sorted(buckets)]
    out: list[Problem] = []
    while len(out) < limit and any(queues):
        for q in queues:
            if q:
                out.append(q.popleft())
                if len(out) >= limit:
                    break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--per-category", type=int, default=50)
    parser.add_argument("--difficulty", type=int, default=6)
    parser.add_argument("--holdout", default="val/holdout_rules.json")
    parser.add_argument("--out-prefix", default="val_report")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap problems PER holdout (round-robin across categories) for a "
        "fast first pass; omit to score the full holdouts",
    )
    args = parser.parse_args()

    predictor = make_vllm_predictor(args.model, args.adapter)

    # 1) Real snapshot-complement holdout (clean only for the 3 downsampled cats).
    held = holdout_problems()
    real_problems = cap_per_category([hp.problem for hp in held], args.limit)
    real_diff = {p.id: len(p.examples) for p in real_problems}
    real_report = aggregate(score(real_problems, predictor, difficulty=real_diff))

    # 2) Synthetic new-rule holdout (the only clean signal on the hard categories).
    # Reserve ALL generated rules (before the cap) so the holdout list stays
    # complete even when we only score a sample.
    registry = HoldoutRegistry(Path(args.holdout))
    synth_problems = cap_per_category(
        build_synthetic_valset(args.per_category, args.difficulty, registry),
        args.limit,
    )
    synth_diff = {p.id: len(p.examples) for p in synth_problems}
    synth_report = aggregate(score(synth_problems, predictor, difficulty=synth_diff))

    print("=== REAL snapshot-complement holdout ===")
    print(format_table(real_report))
    print("\n=== SYNTHETIC new-rule holdout ===")
    print(format_table(synth_report))

    write_json(real_report, f"{args.out_prefix}_real.json")
    write_json(synth_report, f"{args.out_prefix}_synth.json")
    print(f"\nWrote {args.out_prefix}_real.json and {args.out_prefix}_synth.json")
    print("NOTE: REAL holdout is unbiased only for numeral/gravity/unit_conversion;")
    print("other categories there are a pessimistic floor (solver-failure bias).")
    print("SYNTHETIC holdout gives the clean baseline on cipher + bit_manipulation.")


if __name__ == "__main__":
    main()
