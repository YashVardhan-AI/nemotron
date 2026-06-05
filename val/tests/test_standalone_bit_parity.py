"""Guard: the inline bit generator in val/kaggle_newrule_eval_standalone.py must
stay logically identical to val/generators/bit_manipulation.py.

The standalone is a self-contained Kaggle copy (no repo imports), so the two can
drift silently. This execs the standalone's pre-vLLM cells and checks that, for
the same seeds, generate_bit produces the same prompt + answer as the module's
generate. If they diverge, local val numbers won't predict the Kaggle run.
"""

from pathlib import Path

from val.generators.bit_manipulation import generate as module_generate
from val.generators.bit_manipulation import problem_tag
from val.generators.cipher import generate as module_cipher_generate
from val.generators.cipher import query_needs_vocab

_STANDALONE = Path(__file__).resolve().parents[1] / "kaggle_newrule_eval_standalone.py"


def _load_standalone_namespace():
    src = _STANDALONE.read_text(encoding="utf-8")
    # Everything before "Cell 5" is pure stdlib; Cell 5 imports vllm (GPU-only).
    marker = "# %% ── Cell 5"
    prefix = src.split(marker)[0]
    ns: dict = {}
    exec(compile(prefix, str(_STANDALONE), "exec"), ns)  # noqa: S102
    return ns


def test_standalone_bit_matches_module():
    ns = _load_standalone_namespace()
    generate_bit = ns["generate_bit"]
    for seed in range(80):
        a = module_generate(seed, difficulty=8)
        b = generate_bit(seed, 8)
        assert a.prompt == b.prompt, f"prompt mismatch at seed {seed}"
        assert a.answer == b.answer, f"answer mismatch at seed {seed}"
        # family must agree (module id suffix vs standalone meta prefix) ...
        assert a.id.split("-")[2] == b.meta.split("/")[0], f"family mismatch {seed}"
        # ... and the full het/hom stratum tag must match problem_tag.
        assert b.meta == problem_tag(seed), f"stratum mismatch at seed {seed}"


def test_standalone_cipher_matches_module():
    ns = _load_standalone_namespace()
    generate_cipher = ns["generate_cipher"]
    for seed in range(80):
        a = module_cipher_generate(seed, difficulty=4)
        b = generate_cipher(seed, 4)
        assert a.prompt == b.prompt, f"prompt mismatch at seed {seed}"
        assert a.answer == b.answer, f"answer mismatch at seed {seed}"
        # the standalone's needs-vocab stratum tag must match the module helper
        expected = "needs_vocab" if query_needs_vocab(a) else "seen"
        assert b.meta == expected, f"needs_vocab tag mismatch at seed {seed}"
