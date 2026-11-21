# ullm — Milestones

Goal: a working base model and assistant model. Each feature: one /to-tasks list, one PR. `Parallel with` = no code dependency.

## M1. Model and CLI
- Delivers: model and CLI running a random-init checkpoint on the Mac.
- Builds: config.py, model.py, tokenizer.py, generate.py, cli.py (`complete`, `chat`).
- Covers: R4.
- Exit criteria:
  - Tests pass: forward shapes, ~350M params, render_chat mask.
  - `ullm complete` and `ullm chat` stream tokens on CPU and MPS.
- Needs: none.
- Features:
  - F1. Model: uv project, config.py, model.py, shape tests. Docs: tasks. Needs: none. Status: done.
  - F2. CLI: tokenizer.py, generate.py, `complete` and `chat`, safetensors save and load. Docs: tasks. Needs: F1. Status: done.

## M2. Data
- Delivers: FineWeb-Edu shards on HF Hub and a loader for them.
- Builds: data.py (prepare_pretrain(), loader), cli.py (`prepare`).
- Covers: R3, R5.
- Exit criteria:
  - Shards on HF Hub, downloadable on another machine.
  - Loader test passes: 1025-token windows, disjoint per-GPU slices.
- Needs: M1.
- Features:
  - F1. Prepare: download, tokenize, write shards, push to HF Hub. Docs: tasks. Needs: M1 F2. Parallel with: F2. Status: done.
  - F2. Loader: memmap shards into batches, tested on a synthetic shard. Docs: tasks. Needs: M1 F1. Parallel with: F1. Status: done.

## M3. Training
- Delivers: pre-train loop that works on CPU and on 1×H100.
- Builds: train.py (pretrain), cli.py (`pretrain`).
- Covers: R1, R6.
- Exit criteria:
  - Tiny CPU overfit test passes.
  - 1×H100 test run within ~$25. Measured tok/s projects the real run within $75.
- Needs: M2.
- Features:
  - F1. Loop: AdamW, schedule, logging, val loss, samples, checkpoints, overfit test. Docs: tasks. Needs: M1 F1, M2 F2. Status: not started.
  - F2. Multi-GPU + test run: DDP, bf16, torch.compile, cost/ETA, 1×H100 test run. Docs: tasks. Needs: F1. Status: not started.

## M4. Base model
- Delivers: trained base model that beats GPT-2 small, weights on HF Hub.
- Builds: eval.py (lm-eval, generation test), prompts.py, cli.py (`eval`).
- Covers: R1, R2a, R5.
- Exit criteria:
  - HellaSwag and WikiText beat GPT-2 small.
  - Generation test passes human judgment.
  - Run cost ≤ $75.
- Needs: M3.
- Features:
  - F1. Eval: lm-eval adapter, GPT-2 baseline, generation test, prompts frozen. Docs: tasks. Needs: M1 F2. Parallel with: M2, M3. Status: not started.
  - F2. Real run: 8×H100 pre-train, eval, publish weights. Docs: tasks. Needs: M3 F2, F1. Status: not started.

## M5. Assistant
- Delivers: assistant model that chats, weights on HF Hub.
- Builds: data.py (prepare_sft()), train.py (SFT mode), cli.py (`sft`), eval.py (assistant test).
- Covers: R2b, R5.
- Exit criteria:
  - Assistant test passes human judgment: answers, stops on `<|end|>`.
  - Weights on HF Hub, README has one command per stage.
- Needs: M4.
- Features:
  - F1. SFT: smol-smoltalk prep with loss mask, masked-loss training, 1×H100 run. Docs: tasks. Needs: M4 F2. Status: not started.
  - F2. Assistant eval + publish: assistant test, publish weights, README. Docs: tasks. Needs: F1, M4 F1. Status: not started.

## Coverage
| Requirement or component | Milestones |
|---|---|
| R1 | M3, M4 |
| R2a | M4 |
| R2b | M5 |
| R3 | M2 |
| R4 | M1 |
| R5 | M2, M4, M5 |
| R6 | M3 |
| config.py, model.py, tokenizer.py, generate.py | M1 |
| data.py | M2, M5 |
| train.py | M3, M5 |
| eval.py, prompts.py | M4, M5 |
| cli.py | M1 to M5 |

## Open Questions
- OQ1. GPU provider. Owner: user. Needed by M3 F2.
- OQ5. HF Hub repo names for weights. Owner: user. Needed by M4 F2. Shards: `mattiasz/ullm-fineweb-edu-shards`.
