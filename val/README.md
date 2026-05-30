# Validation harness (roadmap P0)

Measures per-category, per-difficulty generalization accuracy of a trained
adapter on **rules it never saw**, scored through the real greedy/vLLM/`verify()`
path (`notebook_tinker.generate_predictions` semantics).

## Local (no GPU)
- Unit tests: `uv run pytest val/tests -v`
- Pipeline smoke (oracle predictor): `uv run python -m val.run_local_demo`

## Real scoring (GPU / Kaggle)
`uv run python -m val.run_vllm --model <BASE_MODEL_PATH> --adapter <ADAPTER_DIR>`

Scores **two** holdouts and writes `val_report_real.json` + `val_report_synth.json`:
- **Real** snapshot-complement holdout (`val/real_holdout.py`). Read it as:
  **unbiased baseline for numeral / gravity / unit_conversion only**; other
  categories are a *pessimistic floor* (absent due to solver failure, not random
  downsampling).
- **Synthetic** new-rule holdout from `val/generators` — the clean baseline on
  the categories the real holdout can't cover. Today: `cipher` (easy) and
  `bit_manipulation` (one hard category). Rules are auto-reserved (see below).

## Two accuracies
- `accuracy` — the real grader (`verify`, float-lenient; binary strings within 1%
  count as equal).
- `accuracy_strict` — diagnostic (`verify_strict`, exact on binary). A gap means
  the lenient grader is inflating a category's headline number.

## Holdout registry
`val/holdout_rules.json` reserves synthetic val rule signatures. **Future
training data-gen MUST exclude reserved signatures** so val never leaks.

## Follow-on
Synthetic generators for the other 7 categories (roadmap lever #2) plug into
`val/generators` via `register(category, generate, rule_signature)`. cipher and
bit_manipulation are the references; cryptarithm/`_guess`/equation come next.
