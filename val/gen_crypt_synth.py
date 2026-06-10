"""P2 validation gate: generate N synthetic cryptarithm puzzles and validate them
BEFORE any training (mirrors val/gen_crypt_symbolic.py for the real pool).

Checks: generation yield (attempts/build_fail/dup/learnable_fail/kept + wall time),
distribution match vs the real marginals (base/mode/op-type), 100% trace round-trip,
token budget, dedup vs real train (must be 0 collisions), glyph un-merge, sample dump.

Run:  uv run python val/gen_crypt_synth.py [N] [style] [seed]
Outputs: runs/crypt_symbolic/{synth_rendered.jsonl, synth_samples.txt} + a readout.
"""

import sys
import time
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from reasoners import crypt_symbolic_gen as G  # noqa: E402
from reasoners import crypt_symbolic_trace as R  # noqa: E402
from val.grading import extract_final_answer, extract_final_answer_braceaware, verify  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
STYLE = sys.argv[2] if len(sys.argv) > 2 else "derive_search"
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 90001
BUDGET_HARD = 7680
OUT_DIR = ROOT / "runs" / "crypt_symbolic"
TOKENIZER_PATH = ROOT / "tokenizer.json"
_tok = Tokenizer.from_file(str(TOKENIZER_PATH))


def n_tokens(completion: str) -> int:
    return len(_tok.encode(completion, add_special_tokens=False).ids)


def pct(vals, p):
    if not vals:
        return 0
    k = max(0, min(len(vals) - 1, int(round((p / 100) * (len(vals) - 1)))))
    return vals[k]


def fracs(counter):
    tot = sum(counter.values()) or 1
    return {k: counter[k] / tot for k in counter}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    marg = G.load_marginals()

    # real train prompts -> dedup seed + collision check
    df = pd.read_parquet(G._PARQUET)
    real_prompts = set(df["prompt"].tolist())

    t0 = time.time()
    puzzles, stats = G.generate_dataset(N, SEED, marg, seen_prompts=real_prompts)
    gen_secs = time.time() - t0

    # render + round-trip + budget; collect synth distribution
    rendered, drops = [], []
    s_base, s_mode, s_op = Counter(), Counter(), Counter()
    collisions = 0
    for i, pz in enumerate(puzzles):
        if pz["prompt"] in real_prompts:
            collisions += 1
        reasoning = R.render(
            pz["prompt"],
            pz["mapping"],
            pz["ops"],
            pz["mode"],
            pz["base"],
            pz["gold"],
            STYLE,
        )
        if reasoning is None:
            drops.append((i, "render_None"))
            continue
        completion = f"{reasoning}\n</think>\n\\boxed{{{pz['gold']}}}<|im_end|>"
        if not verify(pz["gold"], extract_final_answer_braceaware(completion)):
            drops.append((i, "roundtrip_fail"))
            continue
        s_base[pz["base"]] += 1
        s_mode[pz["mode"]] += 1
        for ot in pz["ops"].values():
            s_op[ot] += 1
        rendered.append(
            {
                "id": f"crypt-sym-{STYLE}-{SEED}-{i}",
                "base": pz["base"],
                "mode": pz["mode"],
                "n_tokens": n_tokens(completion),
                "gradeable": bool(verify(pz["gold"], extract_final_answer(completion))),
                "gold": pz["gold"],
                "completion": completion,
            }
        )

    out_jsonl = OUT_DIR / "synth_rendered.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rendered:
            import json

            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "synth_samples.txt", "w", encoding="utf-8") as f:
        for r in rendered[:10]:
            f.write(
                f"===== {r['id']} base={r['base']} mode={r['mode']} "
                f"tokens={r['n_tokens']} gold={r['gold']!r} gradeable={r['gradeable']} =====\n"
            )
            f.write(r["completion"] + "\n\n")

    toks = sorted(r["n_tokens"] for r in rendered)
    n = len(rendered)
    rm, sm = fracs(marg["base"]), fracs(s_base)
    rmode, smode = fracs(marg["mode"]), fracs(s_mode)
    rop, sop = fracs(marg["optypes"]), fracs(s_op)

    print("=" * 64)
    print(f"SYNTHETIC CRYPTARITHM — P2 VALIDATION  [N={N} style={STYLE} seed={SEED}]")
    print("=" * 64)
    print(
        f"generation wall time           : {gen_secs:.1f}s  ({gen_secs / max(n, 1):.2f}s/kept)"
    )
    print(f"  attempts                     : {stats['attempts']}")
    print(
        f"  build_fail / dup / learn_fail: {stats['build_fail']} / {stats['dup']} / {stats['learnable_fail']}"
    )
    print(f"  kept (learnable, deduped)    : {stats['kept']}")
    print(
        f"  yield kept/solver_calls      : "
        f"{stats['kept'] / max(stats['solver_calls'], 1):.1%}"
    )
    print()
    print(f"RENDERED + round-trip OK       : {n} / {len(puzzles)}")
    print(f"  drops                        : {len(drops)}  {drops[:5]}")
    print(f"dedup collisions vs real train : {collisions}  (MUST be 0)")
    print()
    print("DISTRIBUTION MATCH (real -> synth fraction):")
    print(f"  base  real={ {k: round(rm.get(k, 0), 2) for k in sorted(rm)} }")
    print(f"        synth={ {k: round(sm.get(k, 0), 2) for k in sorted(sm)} }")
    print(
        f"  mode  real={ {k: round(rmode[k], 2) for k in rmode} }  synth={ {k: round(smode.get(k, 0), 2) for k in smode} }"
    )
    print("  top ops (real% / synth%):")
    for ot, _ in marg["optypes"].most_common(10):
        print(f"    {ot:14s} {100 * rop.get(ot, 0):5.1f} / {100 * sop.get(ot, 0):5.1f}")
    synth_only = [ot for ot in sop if ot not in rop]
    if synth_only:
        print(f"  !! synth op-types absent from real: {synth_only}")
    print()
    print("TOKEN BUDGET (completion):")
    if toks:
        print(
            f"  p50={pct(toks, 50)}  p90={pct(toks, 90)}  max={toks[-1]}  >7680={sum(1 for t in toks if t > BUDGET_HARD)}"
        )
    print(
        f"  gradeable                    : {sum(1 for r in rendered if r['gradeable'])}/{n}"
    )
    print()
    print(f"outputs -> {out_jsonl}")


if __name__ == "__main__":
    main()
