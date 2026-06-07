"""Cryptarithm STaR feasibility probe (GPU/Kaggle only).

The decisive test before any cryptarithm STaR loop: does the BASELINE adapter
ever produce a CORRECT cryptarithm answer when sampled at temperature>0? STaR
(self-taught reasoner) fine-tunes a model on its OWN verifier-correct samples;
it can only work if the model already succeeds *sometimes*. Dense-trace SFT
failed because it taught a format the model could not execute and crashed the
cipher/equation neighbors (memory: induct-sft-negative-result). STaR sidesteps
both -- the traces are the model's own greedy-length output, gated by the real
grader -- but ONLY if pass@K > 0 here.

What this measures (on uniquely-solvable instances, so identifiability is NOT a
confound -- every instance has exactly one correct answer the demos pin down):
  - arith-family pass@K   (add/abs_diff/mul query op) -- the HEADLINE.
  - concat-family pass@K  (concat/rev_concat) -- a sanity floor; the model
    should ace these trivial structural ops.
  - grader-strict vs brace-aware extraction. The competition grader truncates
    \\boxed{...} at the first `}`, so ~10% of cryptarithm answers (those
    containing `}`) are ungradeable even when reasoned correctly. Grader-strict
    pass@K is the leaderboard-relevant number (an answer STaR can actually win);
    brace-aware pass@K is the model's TRUE reasoning ceiling, un-blinded.
  - accept rate (correct samples / total samples) -- sets the STaR economics
    (how many samples to draw per accepted training trace).

Accepted (prompt, completion, answer) pairs are written to JSONL so a STaR loop
can consume them directly as training rows -- no re-sampling needed.

DECISION RULE (arith family, grader-strict pass@K):
  GREEN  >= 12%  -> run the small STaR loop (harvest correct samples -> SFT).
  YELLOW  1-12%, or strong brace-aware but weak grader-strict -> marginal;
                  reasoning exists but the brace cap eats most of it; reconsider.
  RED    ~0%     -> the model cannot do cryptarithm arith at all; STaR is dead
                  too. Accept cryptarithm is unreachable by SFT-family methods.

Usage (GPU box):
    uv run python -m val.probe_crypt_star --model <BASE> --adapter <ADAPTER_DIR>
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

from reasoners.cryptarithm_deduce_core import sample_solvable
from reasoners.cryptarithm_rule import render_prompt
from val.grading import (
    extract_final_answer,
    extract_final_answer_braceaware,
    verify,
)
from val.scoring import BOXED_INSTRUCTION

ARITH_OPS = {"add", "abs_diff", "mul"}
CONCAT_OPS = {"concat", "rev_concat"}

# Disjoint from val generator seeds (0..few-thousand), cryptarithm forward-gen
# (1e6) and bit forward-gen (2e6). These instances are only SAMPLED, never
# trained on, so holdout leakage is irrelevant -- we are measuring the baseline.
PROBE_SEED_OFFSET = 5_000_000


class ProbeInstance:
    """One uniquely-solvable cryptarithm instance + its op family + grade flag."""

    __slots__ = ("seed", "family", "prompt", "answer", "ungradeable")

    def __init__(self, seed: int, family: str, prompt: str, answer: str):
        self.seed = seed
        self.family = family
        self.prompt = prompt
        self.answer = answer
        # The grader regex `\boxed\{([^}]*)` truncates at the FIRST `}`, so ONLY
        # a `}` in the answer makes it ungradeable (a `{` extracts fine). This is
        # the leaderboard-unwinnable set even with perfect reasoning.
        self.ungradeable = "}" in answer


def build_probe_set(
    n_arith: int, n_concat: int, difficulty: int
) -> list[ProbeInstance]:
    """Collect n_arith arith-query + n_concat concat-query solvable instances.

    Classifies by the QUERY operator's semantics (rule.op_of[q_input[2]]).
    Deterministic: walks seeds from PROBE_SEED_OFFSET upward.
    """
    arith: list[ProbeInstance] = []
    concat: list[ProbeInstance] = []
    seed = PROBE_SEED_OFFSET
    guard = PROBE_SEED_OFFSET + 200 * (n_arith + n_concat + 1)
    while (len(arith) < n_arith or len(concat) < n_concat) and seed < guard:
        rule, examples, q_input, q_answer = sample_solvable(seed, difficulty)
        op = rule.op_of[q_input[2]]
        prompt = render_prompt(examples, q_input, wrapper_index=0)
        inst = ProbeInstance(seed, "", prompt, q_answer)
        if op in ARITH_OPS and len(arith) < n_arith:
            inst.family = "arith"
            arith.append(inst)
        elif op in CONCAT_OPS and len(concat) < n_concat:
            inst.family = "concat"
            concat.append(inst)
        seed += 1
    return arith + concat


def make_vllm_multisampler(
    model_path: str,
    adapter_path: str,
    *,
    n: int,
    temperature: float,
    top_p: float,
    max_lora_rank: int = 32,
    max_tokens: int = 7680,
    max_model_len: int = 8192,
) -> Callable[[list[str]], list[list[str]]]:
    """Like run_vllm.make_vllm_predictor but draws `n` samples per prompt and
    returns ALL of them (list-of-lists, aligned to the input prompts)."""
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
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n
    )

    def sample(user_contents: list[str]) -> list[list[str]]:
        prompts = []
        for content in user_contents:
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
        return [[o.text for o in out.outputs] for out in outputs]

    return sample


def _pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den) if den else 0.0:5.1f}%  ({num}/{den})"


def run_probe(
    sampler: Callable[[list[str]], list[list[str]]],
    *,
    n: int,
    temperature: float,
    top_p: float,
    n_arith: int,
    n_concat: int,
    difficulty: int,
    accepted_out: str,
    report_out: str,
) -> dict:
    """Build the probe set, draw K samples via `sampler`, grade, report.

    `sampler(contents) -> list[list[str]]` draws n samples per prompt and returns
    them aligned to the input. This is decoupled from HOW the samples are produced
    so the same logic runs from the CLI (make_vllm_multisampler builds a fresh vLLM
    engine) OR inside a notebook reusing an already-initialised `llm` (see
    val/kaggle_crypt_star_standalone.py -- the path that dodges the Kaggle Blackwell
    ptxas-permission crash, since the notebook's engine is already patched).
    """
    instances = build_probe_set(n_arith, n_concat, difficulty)
    got_arith = sum(i.family == "arith" for i in instances)
    got_concat = sum(i.family == "concat" for i in instances)
    print(
        f"Probe set: {got_arith} arith + {got_concat} concat instances, "
        f"K={n} @ temp={temperature} top_p={top_p}"
    )
    ung_arith = sum(i.family == "arith" and i.ungradeable for i in instances)
    print(
        f"  (of arith instances, {ung_arith} contain `}}` in the answer -> "
        "grader-ungradeable even if reasoned correctly)"
    )

    contents = [inst.prompt + BOXED_INSTRUCTION for inst in instances]
    samples_per_instance = sampler(contents)

    # Per-family tallies.
    stats = {
        "arith": {"pass_g": 0, "pass_b": 0, "n": 0, "correct_samples": 0, "tot": 0},
        "concat": {"pass_g": 0, "pass_b": 0, "n": 0, "correct_samples": 0, "tot": 0},
    }
    accepted_rows: list[dict] = []

    for inst, samples in zip(instances, samples_per_instance):
        s = stats[inst.family]
        s["n"] += 1
        s["tot"] += len(samples)
        any_g = False
        any_b = False
        for text in samples:
            ok_g = verify(inst.answer, extract_final_answer(text))
            ok_b = verify(inst.answer, extract_final_answer_braceaware(text))
            if ok_g:
                s["correct_samples"] += 1
                any_g = True
                accepted_rows.append(
                    {
                        "problem_id": f"star-cryptarithm-{inst.seed}",
                        "family": inst.family,
                        "prompt": inst.prompt,
                        "completion": text,
                        "answer": inst.answer,
                        "gradeable": not inst.ungradeable,
                    }
                )
            if ok_b:
                any_b = True
        if any_g:
            s["pass_g"] += 1
        if any_b:
            s["pass_b"] += 1

    print("\n=== pass@K (>=1 of K samples correct on an instance) ===")
    for fam in ("arith", "concat"):
        s = stats[fam]
        print(f"[{fam}]  n={s['n']}")
        print(f"  grader-strict pass@{n}: {_pct(s['pass_g'], s['n'])}")
        print(f"  brace-aware   pass@{n}: {_pct(s['pass_b'], s['n'])}")
        print(
            f"  accept rate (correct/total samples): {_pct(s['correct_samples'], s['tot'])}"
        )

    a = stats["arith"]
    arith_pass_g = (100.0 * a["pass_g"] / a["n"]) if a["n"] else 0.0
    arith_pass_b = (100.0 * a["pass_b"] / a["n"]) if a["n"] else 0.0
    if arith_pass_g >= 12.0:
        verdict = "GREEN -- run the STaR loop (>=12% arith grader pass@K)"
    elif arith_pass_g >= 1.0 or arith_pass_b >= 12.0:
        verdict = (
            "YELLOW -- marginal: reasoning present but grader-strict thin "
            "(brace cap or low success); weigh effort vs the brace-limited ceiling"
        )
    else:
        verdict = "RED -- model cannot do cryptarithm arith; STaR is dead too"
    print(f"\nVERDICT: {verdict}")
    print(
        f"  arith grader-strict pass@{n} = {arith_pass_g:.1f}% | "
        f"brace-aware = {arith_pass_b:.1f}%"
    )

    with open(accepted_out, "w", encoding="utf-8") as f:
        for row in accepted_rows:
            json.dump(row, f)
            f.write("\n")
    gradeable = sum(r["gradeable"] for r in accepted_rows)
    print(
        f"\nWrote {len(accepted_rows)} accepted samples "
        f"({gradeable} gradeable) -> {accepted_out}"
    )

    summary = {
        "config": {
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "difficulty": difficulty,
        },
        "arith": stats["arith"],
        "concat": stats["concat"],
        "arith_pass_g_pct": arith_pass_g,
        "arith_pass_b_pct": arith_pass_b,
        "verdict": verdict,
        "accepted_total": len(accepted_rows),
        "accepted_gradeable": gradeable,
    }
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary -> {report_out}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="cryptarithm STaR feasibility probe")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--n", type=int, default=64, help="samples per instance (K)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=7680)
    parser.add_argument("--n-arith", type=int, default=40)
    parser.add_argument("--n-concat", type=int, default=20)
    parser.add_argument("--difficulty", type=int, default=4)
    parser.add_argument(
        "--accepted-out",
        default="crypt_star_accepted.jsonl",
        help="JSONL of verifier-correct (prompt, completion, answer) STaR rows",
    )
    parser.add_argument(
        "--report-out", default="crypt_star_probe.json", help="machine-readable summary"
    )
    args = parser.parse_args()

    sampler = make_vllm_multisampler(
        args.model,
        args.adapter,
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    run_probe(
        sampler,
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        n_arith=args.n_arith,
        n_concat=args.n_concat,
        difficulty=args.difficulty,
        accepted_out=args.accepted_out,
        report_out=args.report_out,
    )


if __name__ == "__main__":
    main()
