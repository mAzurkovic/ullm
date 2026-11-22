# ullm-base-eval

`ullm eval` on [mattiasz/ullm-base](https://huggingface.co/mattiasz/ullm-base) beside GPT-2 small, both scored by the same `LogitsLM` adapter, 0-shot.

**Sample size: the first 20 examples of each benchmark** (`--limit 20`), on the Mac CPU. This is a small sample: one HellaSwag example is 0.05 of the score. The full sets (10,042 HellaSwag examples, 62 WikiText documents) were not run.

| Benchmark | ullm-base (360M) | GPT-2 small (124M) | Result |
|---|---|---|---|
| HellaSwag acc_norm | 0.350 | 0.350 | tie |
| WikiText word perplexity | **35.75** | 40.03 | ullm better |

For reference, GPT-2 small's published full-set scores in lm-eval are about 0.31 (HellaSwag acc_norm) and 37.4 (WikiText word perplexity). The 20-example GPT-2 scores above are close to them.

**Run:**
```sh
uv run ullm eval --model gpt2 --run gpt2-smoke --limit 20 --device cpu
uv run ullm eval --model checkpoints/ullm-base --run ullm-base-smoke --limit 20 --device cpu --max-new-tokens 16 --baseline results/gpt2-smoke/benchmarks.json
```

The generation samples of this run were cut to 16 tokens, so they are not kept here. `results/base-7b/results.md` has longer samples.
