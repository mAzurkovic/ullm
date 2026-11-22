# M5 F1. SFT — Tasks

Scope: turn smol-smoltalk into masked training rows, fine-tune `mattiasz/ullm-base` on them for 1 epoch on 1×H100, and chat with the result. Source: `design.md` (Model: chat template and SFT example; Training Recipe SFT column; D9 to D12; Inference), requirements R2b, R5. Publishing the weights is F2.

Data facts (test split, 24,229 conversations): 15% have a `system` message; all others alternate user, assistant. Mean 967 tokens, 40% longer than 1024, 1% with a first exchange longer than 1024. Train split: 460,341 conversations in 4 parquet files (970 MB).

## 1. `ullm prepare-sft` writes masked training rows [serial]
- [x] 1.1 `data.py` `sft_example(messages) -> tuple[ids, mask] | None`: skip a conversation with a `system` message or roles that do not alternate user, assistant; `render_chat`; cut at the last turn end that fits 1025 tokens (1024 inputs + 1 shifted target); `None` if the first exchange does not fit. Parametrized tests: short, truncated, first exchange too long, system message, bad roles.
- [x] 1.2 `data.py` `prepare_sft(out, workers, limit=None)`: `hf_hub_download` the 4 train files and the test file of `HuggingFaceTB/smol-smoltalk`, read with pyarrow, `sft_example` in a `Pool`, pad rows to 1025 with `<|endoftext|>` and mask 0, save `train_ids.npy` (uint16), `train_mask.npy` (uint8), `val_ids.npy`, `val_mask.npy`; delete the parquet files; print kept, skipped, truncated, assistant tokens, and one decoded train row with the mask-1 text marked. Test with a small local parquet and `hf_hub_download` patched (network boundary).
- [x] 1.3 `cli.py` `prepare-sft` command: `--out data/sft`, `--workers`, `--limit`. CLI test.

## 2. `ullm sft` fine-tunes on the masked loss [serial, needs 1]
- [x] 2.1 `config.py` `SFTConfig`: `run`, `base` (checkpoint dir, required), `data=data/sft`, `batch_size=64` conversations, `micro_batch=16`, `epochs=1`, `peak_lr=3e-5`, `warmup_steps=100`, linear decay to 0, AdamW and clip as pre-train, log/val/sample/checkpoint intervals, `val_batches`, `seed`, `device`, `compile`, `hourly_rate`, `budget=25`, `budget_check_step`, `gpu_peak_tflops`. `total_steps` from row count. Tests.
- [x] 2.2 `data.py` `SFTLoader`: memmaps a split's ids and mask, one shuffled epoch (fixed seed), rows split across ranks, yields `(x, y)` with `y = -100` where the mask is 0. Tests: shapes, masked targets, every row once per epoch, no overlap across ranks.
- [x] 2.3 `train.py`: `_loss` uses `ignore_index=-100`; move the shared step, log, val, sample, budget, and save code out of `_pretrain` into one loop that both stages call. All existing pre-train tests pass unchanged.
- [x] 2.4 `train.py` `sft(config)`: `GPT.load(config.base)`, linear schedule, `SFTLoader` for train and val, chat samples from `CHAT_PROMPTS` (new in `prompts.py`) as `render_chat([user]) + <|assistant|>`, stop on `<|end|>`, final weights to `checkpoints/<run>/final`. No `--resume` (about 1 h run: restart from the base).
- [x] 2.5 `cli.py` `sft` command from `SFTConfig`. Tests: tiny model on tiny rows, loss falls; loss does not change when user tokens change; samples file has chat prompts.

## 3. The 1×H100 SFT run gives a chat model [serial, needs 2]
- [x] 3.1 README: "Fine-tune the assistant" steps for the GPU node: `hf download mattiasz/ullm-base`, `ullm prepare-sft`, `ullm sft`, `ullm chat`.
- [ ] 3.2 User rents 1×H100 SXM, runs README setup, then `uv run ullm prepare-sft` (pastes the counts and the decoded row; Claude checks the mask covers only assistant text and its `<|end|>`), then `uv run ullm sft --base checkpoints/ullm-base --run sft --device cuda --compile --hourly-rate <node price>`; pastes first logs. Claude checks tok/s ≈ 150K, ETA ≈ 45 min, step-200 budget check.
- [ ] 3.3 User pastes val loss and samples near the end. Claude checks val loss falls and samples answer and end with `<|end|>`.
- [ ] 3.4 On the node: `uv run ullm chat --model checkpoints/sft/final`, 3 turns. Answers stay on topic, the turn stops, no `<|user|>` text written.
- [ ] 3.5 User copies `results/sft/` and `checkpoints/sft/final/` to the Mac, checks files, destroys the node. Claude writes `results/sft/results.md` and commits `results/sft/`.

## Dependencies
- 2 needs 1.
- 3 needs 2.

## Open questions
- OQ-S1. System messages (15%): skip those conversations (decided). About 390K remain; the format has no system prompt.
- OQ-S2. Validation: the dataset's own test split (24K), not 1% of train (decided).
- OQ-S3. Prep runs on the GPU node, no Hub repo for SFT rows (decided). No separate Mac smoke run: the unit tests train a tiny model, and the node prints a decoded row before training.
