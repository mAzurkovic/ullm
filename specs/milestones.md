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
  - F1. Loop: AdamW, schedule, logging, val loss, samples, checkpoints, overfit test. Docs: tasks. Needs: M1 F1, M2 F2. Status: done.
  - F2. Multi-GPU + test run: DDP, bf16, torch.compile, cost/ETA, 1×H100 test run. Docs: tasks. Needs: F1. Status: done.

## M4. Base model
- Delivers: trained base model, weights on HF Hub, `ullm complete` works on it.
- Builds: no new code; uses `ullm pretrain` and the README GPU guide.
- Covers: R1, R5.
- Exit criteria:
  - Real run finishes on 8×H100, cost ≤ $75, wall-clock ≤ 24 h.
  - `ullm complete` gives fluent continuations.
  - Weights on HF Hub.
- Needs: M3.
- Features:
  - F1. Real run: 4×H100 pre-train (no 8× capacity), try `ullm complete`, publish weights. Docs: tasks. Needs: M3 F2. Status: done.

## M5. Assistant
- Delivers: assistant model that chats, weights on HF Hub.
- Builds: data.py (prepare_sft()), train.py (SFT mode), cli.py (`sft`).
- Covers: R2b (chat format), R5.
- Exit criteria:
  - `ullm chat` answers and stops on `<|end|>`.
  - Weights on HF Hub, README has one command per stage.
- Needs: M4.
- Features:
  - F1. SFT: smol-smoltalk prep with loss mask, masked-loss training, 1×H100 run. Docs: tasks. Needs: M4 F1. Status: done.
  - F2. Publish: assistant weights on HF Hub, README. Docs: tasks. Needs: F1. Status: done.

## M6. Evaluation (P1, last)
- Delivers: benchmark scores beside GPT-2 small and judged generation tests for both models.
- Builds: eval.py (lm-eval, generation test, assistant test), prompts.py, cli.py (`eval`).
- Covers: R2a, R2b (judged tests).
- Exit criteria:
  - HellaSwag and WikiText beat GPT-2 small.
  - Generation test and assistant test pass human judgment.
- Needs: M4 (base eval), M5 (assistant eval).
- Features:
  - F1. Base eval: lm-eval adapter, GPT-2 baseline, generation test. Docs: tasks (`specs/m4-base-model/f1-eval/`). Needs: M4 F1. Status: in progress, parked on branch `eval` (2.1 open: how to score GPT-2).
  - F2. Assistant eval: 20 prompts + 3 conversations, results.md. Docs: tasks. Needs: F1, M5 F1. Status: not started.

## Coverage
| Requirement or component | Milestones |
|---|---|
| R1 | M3, M4 |
| R2a | M6 |
| R2b | M5, M6 |
| R3 | M2 |
| R4 | M1 |
| R5 | M2, M4, M5 |
| R6 | M3 |
| config.py, model.py, tokenizer.py, generate.py | M1 |
| data.py | M2, M5 |
| train.py | M3, M5 |
| eval.py, prompts.py | M6 |
| cli.py | M1 to M6 |

## Open Questions
- OQ1. GPU provider. Owner: user. Needed by M3 F2.
- OQ5. HF Hub repo names. Shards: `mattiasz/ullm-fineweb-edu-shards`. Base: `mattiasz/ullm-base`. Assistant: decide at M5.
