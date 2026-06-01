# =============================================================================
# NEW-RULE GENERALIZATION EVAL  —  drop-in cell for the adapter-validation
# notebook (huikang's adapter-validation-notebook.ipynb).
#
# WHAT IT DOES
#   The validation notebook reports accuracy on the TRAIN set (~88% overall) —
#   that's memorization-inclusive. This cell measures accuracy on BRAND-NEW
#   rules the model has never seen (cipher + bit_manipulation generators), which
#   is the actual test skill. A large gap between this and the train number =
#   the memorize-vs-generalize gap.
#
# HOW TO USE (paste as a new cell)
#   1. Run it AFTER the notebook's "Init vLLM" cell — it reuses the existing
#      `llm` object (same engine config the eval uses; do NOT build a 2nd LLM).
#   2. Make sure the adapter is extracted at ADAPTER_PATH (the notebook puts the
#      converted adapter in /kaggle/working).
#   3. Set REPO_PATH to wherever you attached this repo as a Kaggle dataset
#      (the folder that contains the `val/` and `reasoners/` directories).
#      It will try to auto-detect if you leave the placeholder.
#
# OUTPUT
#   A per-category × difficulty table (lenient `verify` + strict diagnostic),
#   plus an optional snapshot-complement REAL holdout, and a JSON dump.
# =============================================================================

import glob
import json
import os
import sys

# ---------------------------------------------------------------- config -----
REPO_PATH = "/kaggle/input/REPLACE-WITH-YOUR-REPO-DATASET"  # contains val/ + reasoners/
ADAPTER_PATH = (
    "/kaggle/working"  # dir with adapter_config.json + adapter_model.safetensors
)
PER_CATEGORY = 50  # new rules per generator (cipher, bit_manipulation)
DIFFICULTY = 6  # in-context examples per generated problem
RUN_REAL_HOLDOUT = False  # also score snapshot-complement real holdout
REAL_LIMIT = 200  # cap real-holdout problems (round-robin by category)
# -----------------------------------------------------------------------------

# Auto-detect the repo dataset path if the placeholder wasn't edited.
if not os.path.isdir(os.path.join(REPO_PATH, "val")):
    hits = glob.glob("/kaggle/input/**/val/generators/__init__.py", recursive=True)
    if hits:
        REPO_PATH = hits[0].split(os.sep + "val" + os.sep)[0]
        print(f"[auto-detected REPO_PATH] {REPO_PATH}")
    else:
        raise SystemExit(
            "Could not find the repo dataset. Set REPO_PATH to the folder that "
            "contains the `val/` directory."
        )

if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

from vllm import SamplingParams  # noqa: E402  (vLLM already loaded by the notebook)
from vllm.lora.request import LoRARequest  # noqa: E402

from val.holdout_registry import HoldoutRegistry  # noqa: E402
from val.report import aggregate, format_table  # noqa: E402
from val.run_vllm import build_synthetic_valset, cap_per_category  # noqa: E402
from val.scoring import score  # noqa: E402

# Reuse the notebook's already-initialized engine (same config as the eval).
assert "llm" in globals(), (
    "Run this cell AFTER the notebook's 'Init vLLM' cell defines `llm`."
)

_tokenizer = llm.get_tokenizer()  # noqa: F821  (llm is a notebook global)
# Greedy, matching the competition eval (temperature 0, top_p 1, 7680 max tokens).
_sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=7680)
_lora = LoRARequest("adapter", 1, ADAPTER_PATH)


def predict(user_contents: list[str]) -> list[str]:
    """Real-path predictor: chat template w/ thinking -> greedy -> our adapter.

    `score()` hands us each problem's eval prompt (problem.prompt + the boxed
    instruction), exactly as the real generate_predictions does; we only add the
    chat template + generation, mirroring the eval.
    """
    prompts = []
    for content in user_contents:
        try:
            p = _tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError):
            p = content
        prompts.append(p)
    outputs = llm.generate(prompts, sampling_params=_sampling, lora_request=_lora)  # noqa: F821
    return [o.outputs[0].text for o in outputs]


# ---- synthetic NEW-RULE holdout (cipher + bit_manipulation) -----------------
_registry = HoldoutRegistry(os.path.join("/kaggle/working", "holdout_rules.json"))
_synth = build_synthetic_valset(PER_CATEGORY, DIFFICULTY, _registry)
_synth_results = score(
    _synth, predict, difficulty={p.id: len(p.examples) for p in _synth}
)
_synth_report = aggregate(_synth_results)

print("=" * 64)
print("NEW-RULE generalization (synthetic, rules never trained on)")
print("=" * 64)
print(format_table(_synth_report))

with open("/kaggle/working/newrule_report.json", "w") as f:
    json.dump(_synth_report, f, indent=2)
print("\nwrote /kaggle/working/newrule_report.json")

# ---- optional: snapshot-complement REAL holdout -----------------------------
if RUN_REAL_HOLDOUT:
    from val.real_holdout import holdout_problems

    _real = cap_per_category([h.problem for h in holdout_problems()], REAL_LIMIT)
    _real_results = score(
        _real, predict, difficulty={p.id: len(p.examples) for p in _real}
    )
    _real_report = aggregate(_real_results)
    print("\n" + "=" * 64)
    print("REAL snapshot-complement holdout")
    print("(clean baseline ONLY for numeral/gravity/unit_conversion; other")
    print(" categories are a pessimistic floor — solver-failure selection bias)")
    print("=" * 64)
    print(format_table(_real_report))
    with open("/kaggle/working/realholdout_report.json", "w") as f:
        json.dump(_real_report, f, indent=2)
    print("\nwrote /kaggle/working/realholdout_report.json")

# Compare the NEW-RULE numbers above against the notebook's TRAIN-set table:
# train ~100% on gravity/numeral/unit_conversion, ~88% bit, ~8% cryptarithm.
# If new-rule accuracy is much lower, that quantifies the generalization gap.
