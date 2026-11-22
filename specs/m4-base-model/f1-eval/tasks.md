# M4 F1. Eval — Tasks

Scope: `ullm/eval.py` (lm-eval adapter, benchmarks, generation test, `results.md`), base prompt set in `ullm/prompts.py`, `ullm eval` in `ullm/cli.py`. Source: `design.md` (Evaluation, Scope, D14), requirement R2a. Reuses `GPT.load`, `default_device`, `generate`, `encode`/`decode`, `ENDOFTEXT`. Full benchmark runs (ours and GPT-2) happen on the GPU node in M4 F2; here only `--limit` smoke runs. lm-eval's `wikitext` task uses the WikiText-2 test split, which equals the WikiText-103 test split. Assistant prompts are M5.

## 1. lm-eval scores an ullm checkpoint [serial]
- [x] 1.1 Add `lm-eval` and `transformers` to `pyproject.toml`. `uv sync` succeeds.
- [x] 1.2 `LogitsLM(lm_eval.api.model.LM)` in `ullm/eval.py`: `loglikelihood(requests)`: sum of log-probs of the continuation tokens given the context, left-truncated to the context length, and whether greedy decoding produces the continuation. Test: equals a manual log-softmax sum on a tiny model.
- [x] 1.3 `loglikelihood_rolling(requests)`: full-text log-likelihood over context-length windows (`lm_eval.utils.get_rolling_token_windows`). Test: text longer than the context sums every token exactly once.
- [x] 1.4 Batch requests on the device; `generate_until` raises `NotImplementedError`. Test: batched and one-by-one scores match.

## 2. Benchmarks give scores beside GPT-2 small [serial, needs 1]
- [x] 2.1 `run_benchmarks(model, limit) -> Benchmarks` in `ullm/eval.py`: `lm_eval.simple_evaluate` on `hellaswag` and `wikitext`, 0-shot; `model` is a checkpoint dir or `gpt2`; both are scored by the same `LogitsLM` adapter. Returns HellaSwag `acc_norm` and WikiText `word_perplexity`. Test with `simple_evaluate` stubbed (external library): right tasks, shots, model; metrics read.
- [x] 2.2 Write `results/<run>/benchmarks.json`. Test: round-trip.

## 3. Generation test produces the samples to judge [parallel, needs 1.1]
- [x] 3.1 `BASE_PROMPTS` in `ullm/prompts.py`: 10 fixed prompts (facts, explanation, story, list, how-to), frozen from here on.
- [x] 3.2 `generation_test(model, seeds=(0, 1, 2), max_new_tokens=256)` in `ullm/eval.py`: 3 samples per prompt, temperature 0.8, top-k 50, stop on `<|endoftext|>`. Test on a tiny model: 30 samples, same seed gives the same text.

## 4. `ullm eval` writes `results.md` [serial, needs 2, 3]
- [x] 4.1 `eval` subcommand: `--model DIR|gpt2`, `--run`, `--limit`, `--device`, `--baseline` (default `results/gpt2-small/benchmarks.json`). `ullm eval --help` works.
- [x] 4.2 `results/<run>/results.md`: table of HellaSwag acc_norm and WikiText word perplexity, ours beside GPT-2 small with pass/fail (higher acc, lower perplexity), "baseline missing" when no baseline file; then all 30 samples, each under its prompt with a checkbox for the human judgment (grammatical, on topic for 150+ tokens, no loops). Generation test skipped for `gpt2`. Test: tiny checkpoint with `simple_evaluate` stubbed.
- [x] 4.3 Mac smoke run: `ullm eval --model gpt2 --run gpt2-smoke --limit 20` and the same on a random-init ullm checkpoint, on CPU. Scores and `results.md` written. Delete the smoke outputs and any downloaded dataset cache after.

## Dependencies
- 2 needs 1.
- 3 needs 1.1.
- 4 needs 2 and 3.

## Open questions
- OQ-E1. How to score GPT-2 small: through our own `LogitsLM` adapter, the same code as ullm, with GPT-2 loaded by `transformers` (decided). No `accelerate` dependency; both models share the GPT-2 tokenizer.
