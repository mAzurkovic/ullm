# M3 F2. Multi-GPU + test run — Tasks

Scope: `torchrun` multi-process training, bf16 autocast and `torch.compile` on CUDA, MFU / ETA / cost logging, budget check, shard download on the GPU node, `loss.png`, then a 1×H100 test run on Vast.ai, run by the user over SSH. Source: `design.md` (Training Recipe, Logging, D3, D15), requirements R1, R6. Reuses `pretrain()`, `train_step()`, `TokenLoader(rank, world_size)`, `download_shards()`, `PretrainConfig`, `add_config_args`. Multi-process tests use the CPU `gloo` backend and tiny models only.

## 1. `torchrun` splits one run across processes [serial]
- [x] 1.1 `ullm/train.py`: init `torch.distributed` from `RANK` / `WORLD_SIZE` / `LOCAL_RANK` when set (`nccl` on CUDA, `gloo` on CPU), one device per process, destroy at the end. Without them, run as today.
- [x] 1.2 `PretrainConfig.grad_accum_steps(context_length, world_size)`: batch = micro-batch × context × world × accum. Test: 8 GPUs, micro-batch 16 gives accum 4.
- [x] 1.3 Wrap the model in `DistributedDataParallel`; loader gets `rank`, `world_size`; skip gradient sync on all but the last micro-batch (`no_sync`); loss and grad norm averaged across ranks for logging.
- [x] 1.4 Only rank 0 writes logs, val loss, samples, checkpoints, `final/`; resume loads on every rank. Test: `torchrun --nproc_per_node=2` tiny CPU run gives the same final weights as a 1-process run with the same global batch.

## 2. A step is fast on CUDA [parallel, needs 1]
- [x] 2.1 bf16 autocast for forward and loss when the device is CUDA; fp32 weights and optimizer state.
- [x] 2.2 `--compile` flag: `torch.compile` the model before DDP; save and load the uncompiled module so checkpoint keys stay the same. Test on CPU: compiled tiny run saves a checkpoint that `GPT.load` reads.

## 3. Logs show MFU, ETA, and cost; over-budget runs stop [parallel, needs 1]
- [x] 3.1 `PretrainConfig`: `hourly_rate` (USD per hour for the whole node), `budget` 75, `gpu_peak_tflops` 989, `budget_check_step` 200.
- [x] 3.2 Log line and `metrics.csv` add MFU (6 × params × tokens/s ÷ (peak × world)), ETA (remaining tokens ÷ tokens/s of the last log window, so startup and compile time do not count), est. cost (elapsed h × hourly rate). Test: values on a tiny run checked by hand from logged tok/s and elapsed.
- [x] 3.3 At `budget_check_step`: if projected cost (elapsed + ETA) × rate exceeds the budget, save a checkpoint and stop with a message giving the `--max-tokens` that fits. Test: tiny run with a high rate stops at the check step.
- [x] 3.4 `loss.png` in `results/<run>/` at the end: train and val loss against step (`matplotlib`, new dependency: ask first). Test: file written.

## 4. A fresh GPU node runs one command [serial, needs 1, 2, 3]
- [x] 4.1 `--shards-repo` (default empty): when given, rank 0 calls `download_shards()`, which completes a partial set; other ranks wait at a barrier (process-group timeout 30 min). Test: with download mocked, run starts after the fake shards appear.
- [x] 4.2 README "Train on a GPU node": rent (Vast.ai on-demand H100, ~80 GB disk, CUDA 13+), SSH in, install `uv`, clone, `uv sync`, check the GPU with `nvidia-smi`, `torchrun --nproc_per_node=N -m ullm pretrain --compile --hourly-rate ...` inside `tmux`, watch logs, resume after a disconnect, copy `results/<run>/` back with `scp`, destroy the instance. Written so the user can follow it alone.

## 5. 1×H100 test run [serial, needs 4]
- [x] 5.1 User rents a 1×H100 on Vast.ai (on-demand) and runs the README commands over SSH; Claude reads the pasted output. Full 360M model, `--compile`, a few hundred steps with checkpoint and `--resume` once. Within the ~$25 test budget.
- [x] 5.2 Record GPU model (SXM or PCIe), tok/s, MFU, and the projected 8×H100 time and cost for 7B tokens in `results/<run>/`; check it fits R1. Commit `results/<run>/`. Terminate the instance; delete shards and checkpoints.

## Dependencies
- 2 and 3 need 1.
- 4 needs 1, 2, 3.
- 5 needs 4.
