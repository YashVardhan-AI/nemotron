# ruff: noqa: E402
# =============================================================================
# CRYPTARITHM STaR FEASIBILITY PROBE — NOTEBOOK COMPANION (reuses the notebook's
# already-initialised `llm`).
#
# WHY THIS EXISTS: `python -m val.probe_crypt_star` spins up its OWN vLLM engine
# in a fresh process, which on Kaggle Blackwell GPUs dies during torch.compile
# with `PermissionError: ... ptxas-blackwell` (the vendored ptxas binary isn't
# executable). The competition's "Load model"/"Init vLLM" cells already work
# around that (copy the metric libs to /tmp, chmod +x ptxas*, set
# TRITON_PTXAS_PATH, sys.path.insert(0,"/tmp")) and build ONE working `llm`.
# This companion reuses THAT `llm` (same as kaggle_realholdout_standalone.py),
# so there is no second engine and no ptxas crash.
#
# Paste each `# %% ── Cell N ──` block as its own cell, in order, AFTER the
# notebook's "Init vLLM" cell (so `llm` exists) and with the adapter you want to
# probe at ADAPTER_PATH. Reads the repo's reasoners/ + val/ (for the probe-set
# generator, grader, and tallying logic), so the repo must be importable.
#
# DECISION RULE (arith family, grader-strict pass@K):
#   GREEN  >= 12%  -> run the STaR loop (harvest correct samples -> SFT).
#   YELLOW  1-12% (or strong brace-aware but weak grader-strict) -> marginal.
#   RED    ~0%     -> cryptarithm is unreachable by SFT-family methods.
# =============================================================================


# %% ── Cell 1: config ───────────────────────────────────────────────────────
ADAPTER_PATH = "/kaggle/working"  # the adapter to probe (LoRA dir)
REPO_ROOT = ""  # "" => auto-detect the repo (folder containing reasoners/)
K = 64  # samples per instance
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_TOKENS = 7680
N_ARITH = 40  # uniquely-solvable arith-query instances (the headline)
N_CONCAT = 20  # concat-query control (trivial structural ops)
DIFFICULTY = 4
ACCEPTED_OUT = "/kaggle/working/crypt_star_accepted.jsonl"
REPORT_OUT = "/kaggle/working/crypt_star_probe.json"


# %% ── Cell 2: bootstrap repo on sys.path + imports ─────────────────────────
import glob
import os
import sys


def _find_repo_root():
    if REPO_ROOT and os.path.exists(os.path.join(REPO_ROOT, "reasoners")):
        return REPO_ROOT
    for cand in (".", "/kaggle/working/repo/nemotron"):
        if os.path.exists(
            os.path.join(cand, "reasoners", "cryptarithm_deduce_core.py")
        ):
            return cand
    hits = glob.glob("/kaggle/**/reasoners/cryptarithm_deduce_core.py", recursive=True)
    if hits:
        return os.path.dirname(os.path.dirname(hits[0]))
    raise SystemExit("Could not find the repo (reasoners/); set REPO_ROOT in Cell 1.")


_repo = _find_repo_root()
if _repo not in sys.path:
    sys.path.insert(0, _repo)  # additive: keeps /tmp (patched vllm) ahead for vllm
print(f"[repo] {_repo}")

from val.probe_crypt_star import run_probe  # probe-set build + grade + report


# %% ── Cell 3: sampler over the notebook's `llm` + run ──────────────────────
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

assert "llm" in globals(), "Run AFTER the notebook's 'Init vLLM' cell defines `llm`."

_tokenizer = llm.get_tokenizer()  # noqa: F821


def make_llm_sampler(n, temperature, top_p, max_tokens):
    """A `sampler(contents) -> list[list[str]]` backed by the existing `llm`.

    Draws n samples per prompt in ONE batched llm.generate call (prefix caching
    shares the prompt across the K samples). Mirrors probe_crypt_star's vLLM
    sampler but reuses the notebook engine instead of building a new one.
    """
    sp = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n
    )
    lora = LoRARequest("adapter", 1, ADAPTER_PATH)

    def sample(contents):
        prompts = []
        for content in contents:
            try:
                rendered = _tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError):
                rendered = content
            prompts.append(rendered)
        outputs = llm.generate(  # noqa: F821
            prompts, sampling_params=sp, lora_request=lora
        )
        return [[o.text for o in out.outputs] for out in outputs]

    return sample


sampler = make_llm_sampler(K, TEMPERATURE, TOP_P, MAX_TOKENS)
summary = run_probe(
    sampler,
    n=K,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    n_arith=N_ARITH,
    n_concat=N_CONCAT,
    difficulty=DIFFICULTY,
    accepted_out=ACCEPTED_OUT,
    report_out=REPORT_OUT,
)

# Echo the machine-readable summary to the console too (the JSON file isn't
# visible in the notebook) -- copy this block back for the GREEN/YELLOW/RED call.
import json

print("\n" + "=" * 60)
print("crypt_star_probe summary (JSON)")
print("=" * 60)
print(json.dumps(summary, indent=2))
