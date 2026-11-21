# M3 F1. Loop — Tasks

Scope: single-device pre-train loop in `ullm/train.py`, `PretrainConfig` in `ullm/config.py`, `ullm pretrain` in `ullm/cli.py`, fixed sample prompts in `ullm/prompts.py`. Source: `design.md` (Training Recipe, Logging, Data Model run results and checkpoint). Reuses `GPT`, `GPT.save`/`GPT.load`, `TokenLoader`, `split_shards`, `generate`, `encode`/`decode`. Out of scope (M3 F2): DDP, bf16, `torch.compile`, MFU, ETA, cost, budget check, shard download.

## 1. `ullm pretrain` reads its config from flags [serial]
- [x] 1.1 `PretrainConfig` dataclass in `ullm/config.py`: run name, data dir, out dirs (`results/`, `checkpoints/`), batch 512 × 1024 tokens, micro-batch, max tokens 7e9, peak LR 3e-4, warmup 700, min LR 10%, weight decay 0.1, betas (0.9, 0.95), grad clip 1.0, log/val/sample/checkpoint intervals (10/250/500/1000), 20 val batches, seed, device. Derived: grad accum steps, total steps. Tests: defaults, accum = batch / micro-batch, invalid micro-batch raises.
- [x] 1.2 `add_config_args(parser, cls)` and `config_from_args(cls, args)` in `ullm/config.py`: one flag per dataclass field, typed from the field. Test: `--peak-lr 1e-3 --n-layers 4` overrides.
- [x] 1.3 `pretrain` subcommand in `ullm/cli.py` with `PretrainConfig` and `ModelConfig` flags. `ullm pretrain --help` lists them.

## 2. Training steps make the loss fall [serial, needs 1]
- [x] 2.1 `lr_at(step, config)` in `ullm/train.py`: linear warmup, then cosine to 10% of peak. Test: values at step 0, end of warmup, middle, last, checked by hand.
- [x] 2.2 `make_optimizer(model, config)`: AdamW, weight decay on 2-D params only, norms without decay, tied weight counted once. Test: param groups.
- [x] 2.3 `train_step(model, optimizer, batches, config)`: grad accumulation over micro-batches, mean cross-entropy loss, grad clip, returns loss and grad norm. Test: accumulating 4 micro-batches gives the same gradients as one 4× batch.
- [x] 2.4 Overfit test in `tests/test_train.py`: tiny model on one repeated batch on CPU reaches loss < 0.1.

## 3. A run logs metrics, val loss, and samples [serial, needs 2]
- [x] 3.1 `SAMPLE_PROMPTS` in `ullm/prompts.py`: 5 fixed prompts.
- [x] 3.2 `pretrain(model_config, config)` loop: every log interval, `step | loss | lr | grad_norm | tok/s | elapsed` to console and `results/<run>/metrics.csv`; writes `results/<run>/config.json` at start.
- [x] 3.3 Every val interval and at the end: mean loss over the first N batches of shard 0, no grad, logged to `metrics.csv`. Test: tiny run writes val rows.
- [x] 3.4 Every sample interval: 5 prompts, 64 tokens, fixed seed, to console and `results/<run>/samples.txt`. Test: tiny run writes 5 samples per interval.

## 4. A run saves checkpoints and resumes [serial, needs 3]
- [x] 4.1 `TokenLoader(..., start_batch=0)` in `ullm/data.py`: starts at that batch without reading the batches before it. Test: equals the batches after skipping by iteration.
- [x] 4.2 Every checkpoint interval: `checkpoints/<run>/step_NNNNNN/` with `model.safetensors`, `config.json`, `optimizer.pt` (optimizer state, step); keep the last 2. Final model in `checkpoints/<run>/final/`. Test: files and pruning.
- [x] 4.3 `--resume`: load the latest checkpoint, continue at its step, loader at the matching batch. Test on CPU: stop at step 4, resume to step 8, loss equals an uninterrupted 8-step run.

## 5. Mac smoke run on real shards [serial, needs 4]
- [x] 5.1 `hf download` shards 0 and 1 to `data/shards`; run `ullm pretrain` with a ~10M model on MPS for a few hundred steps. Loss falls, val rows and samples written, checkpoint saved.
- [x] 5.2 Delete downloaded shards, checkpoints, and HF cache data after the run. Keep `results/<run>/` only if under 10 MB.

## Dependencies
- 2 needs 1.
- 3 needs 2.
- 4 needs 3.
- 5 needs 4.
