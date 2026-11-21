import contextlib
import csv
import json
import math
import os
import re
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import timedelta
from itertools import islice
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from matplotlib.figure import Figure
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from ullm.config import ModelConfig, PretrainConfig
from ullm.data import TokenLoader, download_shards, split_shards
from ullm.generate import generate
from ullm.model import GPT, default_device
from ullm.prompts import SAMPLE_PROMPTS
from ullm.tokenizer import ENDOFTEXT, decode, encode

Batch = tuple[torch.Tensor, torch.Tensor]

METRICS_HEADER = "step,split,loss,lr,grad_norm,tokens_per_s,mfu,eta_s,cost_usd,elapsed_s\n"
SAMPLE_TOKENS = 64


def lr_at(step: int, config: PretrainConfig) -> float:
    if step < config.warmup_steps:
        return config.peak_lr * (step + 1) / config.warmup_steps
    min_lr = config.peak_lr * config.min_lr_ratio
    progress = (step - config.warmup_steps) / max(1, config.total_steps - config.warmup_steps)
    return min_lr + (config.peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def make_optimizer(model: GPT, config: PretrainConfig) -> torch.optim.AdamW:
    params = list(model.parameters())
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": config.weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.peak_lr, betas=(config.beta1, config.beta2))


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    micro_batches: Sequence[Batch],
    lr: float,
    grad_clip: float,
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), device=next(model.parameters()).device)
    for i, (x, y) in enumerate(micro_batches):
        accumulating = i < len(micro_batches) - 1
        skip_sync = accumulating and isinstance(model, DistributedDataParallel)
        with model.no_sync() if skip_sync else contextlib.nullcontext():
            loss = _loss(model, x, y) / len(micro_batches)
            loss.backward()
        total += loss.detach()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    if dist.is_initialized():
        dist.all_reduce(total)
        total /= dist.get_world_size()
    return total.item(), norm.item()


def pretrain(model_config: ModelConfig, config: PretrainConfig) -> Path:
    rank, world_size, device = _start_processes(config.device)
    try:
        return _pretrain(model_config, config, rank, world_size, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _pretrain(
    model_config: ModelConfig, config: PretrainConfig, rank: int, world_size: int, device: str
) -> Path:
    torch.manual_seed(config.seed)
    main = rank == 0
    accum = config.grad_accum_steps(model_config.context_length, world_size)
    run_dir = config.results_dir / config.run
    checkpoints = config.checkpoints_dir / config.run

    if main and config.shards_repo:
        download_shards(config.shards_repo, config.data)
    _barrier()
    val_path, train_paths = split_shards(config.data)

    if config.resume:
        model, optimizer, start_step = _resume(checkpoints, model_config, config, device)
        if main:
            _truncate_logs(run_dir, start_step)
    elif (run_dir / "metrics.csv").exists() or any(checkpoints.glob("*")):
        raise FileExistsError(
            f"run {config.run!r} already exists; resume it, pick another name, "
            f"or delete {run_dir} and {checkpoints}"
        )
    else:
        model = GPT(model_config).to(device)
        optimizer = make_optimizer(model, config)
        start_step = 0
    _barrier()
    train_model: nn.Module = torch.compile(model) if config.compile else model
    if world_size > 1:
        device_ids = [torch.device(device).index] if device.startswith("cuda") else None
        # The only buffers are fixed RoPE tables, so they never need syncing after start.
        train_model = DistributedDataParallel(
            train_model, device_ids=device_ids, forward_sync_buffers=False
        )

    context = model_config.context_length
    loader = iter(
        TokenLoader(
            train_paths,
            config.micro_batch,
            context,
            rank=rank,
            world_size=world_size,
            device=device,
            start_batch=start_step * accum,
        )
    )
    val_loader = TokenLoader([val_path], config.micro_batch, context, device=device)
    flops_per_token = 6 * sum(p.numel() for p in model.parameters())
    peak_flops = config.gpu_peak_tflops * 1e12 * world_size

    metrics_path = run_dir / "metrics.csv"
    if main:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(
                {"model": asdict(model_config), "train": asdict(config), "world_size": world_size},
                indent=2,
                default=str,
            )
        )
        if not metrics_path.exists():
            metrics_path.write_text(METRICS_HEADER)

    if main and config.hourly_rate == 0:
        print("no --hourly-rate given: cost is not tracked and the budget check is off", flush=True)
    start = time.time()
    tokens_since_log = 0
    train_seconds = 0.0
    steady_rate = 0.0
    with open(metrics_path, "a") if main else contextlib.nullcontext() as metrics:
        for step in range(start_step, config.total_steps):
            lr = lr_at(step, config)
            step_start = time.time()
            batches = [next(loader) for _ in range(accum)]
            loss, grad_norm = train_step(train_model, optimizer, batches, lr, config.grad_clip)
            train_seconds += time.time() - step_start
            done = step + 1
            last = done == config.total_steps
            tokens_since_log += config.batch_tokens
            elapsed = time.time() - start
            cost = elapsed / 3600 * config.hourly_rate

            if main and (done % config.log_every == 0 or last):
                rate = steady_rate = tokens_since_log / train_seconds
                eta = _eta(config, done, steady_rate)
                mfu = flops_per_token * rate / peak_flops
                metrics.write(
                    f"{done},train,{loss:.4f},{lr:.6g},{grad_norm:.4f},{rate:.0f},"
                    f"{mfu:.4g},{eta:.0f},{cost:.4f},{elapsed:.1f}\n"
                )
                metrics.flush()
                print(
                    f"step {done}/{config.total_steps} | loss {loss:.4f} | lr {lr:.2e} | "
                    f"grad_norm {grad_norm:.2f} | tok/s {rate:,.0f} | MFU {mfu:.1%} | "
                    f"elapsed {elapsed:.0f}s | ETA {eta / 3600:.2f}h | cost ${cost:.2f}",
                    flush=True,
                )
                tokens_since_log, train_seconds = 0, 0.0
            if main and (done % config.val_every == 0 or last):
                val = _val_loss(model, val_loader, config.val_batches)
                metrics.write(f"{done},val,{val:.4f},,,,,,,{time.time() - start:.1f}\n")
                metrics.flush()
                print(f"step {done} | val loss {val:.4f}", flush=True)
            if main and (done % config.sample_every == 0 or last):
                _write_samples(model, done, run_dir / "samples.txt", config.seed)
            if main and done % config.checkpoint_every == 0:
                _save_checkpoint(model, optimizer, done, checkpoints)

            if done == config.budget_check_step and config.hourly_rate > 0:
                if not steady_rate:
                    steady_rate = tokens_since_log / train_seconds
                seconds_left = _eta(config, done, steady_rate)
                projected = (elapsed + seconds_left) / 3600 * config.hourly_rate
                if _agree(projected > config.budget, device):
                    if main:
                        if done % config.checkpoint_every:
                            _save_checkpoint(model, optimizer, done, checkpoints)
                        _plot_loss(run_dir)
                        budget_seconds = config.budget * 3600 / config.hourly_rate - elapsed
                        # Leave headroom so a rerun with the hint does not stop again.
                        more_steps = int(0.9 * budget_seconds * steady_rate / config.batch_tokens)
                        if more_steps > 0:
                            advice = (
                                f"rerun with --max-tokens "
                                f"{(done + more_steps) * config.batch_tokens} and a new --run"
                            )
                        else:
                            advice = "the budget is already spent at this hourly rate"
                        print(
                            f"projected cost ${projected:.2f} is over budget "
                            f"${config.budget:.2f}; {advice}",
                            flush=True,
                        )
                    return checkpoints / f"step_{done:06d}"

    final = checkpoints / "final"
    if main:
        model.save(final)
        _plot_loss(run_dir)
    _barrier()
    return final


def _eta(config: PretrainConfig, done: int, tokens_per_second: float) -> float:
    return (config.total_steps - done) * config.batch_tokens / tokens_per_second


def _start_processes(setting: str) -> tuple[int, int, str]:
    device = default_device() if setting == "auto" else setting
    if "RANK" not in os.environ:
        return 0, 1, device
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if device == "cuda":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    # Rank 0 may spend many minutes downloading shards while the others wait.
    backend = "nccl" if device.startswith("cuda") else "gloo"
    dist.init_process_group(backend, timeout=timedelta(minutes=30))
    return rank, world_size, device


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _agree(decision: bool, device: str) -> bool:
    if not dist.is_initialized():
        return decision
    flag = torch.tensor([int(decision)], device=device)
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def _plot_loss(run_dir: Path) -> None:
    with open(run_dir / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    figure = Figure(figsize=(8, 5))
    axes = figure.add_subplot()
    for split in ("train", "val"):
        points = [(int(r["step"]), float(r["loss"])) for r in rows if r["split"] == split]
        if points:
            axes.plot(*zip(*points, strict=True), label=split)
    axes.set(xlabel="step", ylabel="loss", title=run_dir.name)
    axes.legend()
    figure.savefig(run_dir / "loss.png", dpi=120)


def _loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        logits = model(x)
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))


@torch.no_grad()
def _val_loss(model: GPT, loader: TokenLoader, batches: int) -> float:
    losses = [_loss(model, x, y).item() for x, y in islice(iter(loader), batches)]
    return sum(losses) / len(losses)


def _write_samples(model: GPT, step: int, path: Path, seed: int) -> None:
    lines = [f"--- step {step}"]
    for prompt in SAMPLE_PROMPTS:
        generator = torch.Generator().manual_seed(seed)
        ids = generate(
            model,
            encode(prompt),
            max_new_tokens=SAMPLE_TOKENS,
            temperature=0.8,
            top_k=50,
            stop_ids={ENDOFTEXT},
            generator=generator,
        )
        lines.append(f"> {prompt}{decode(list(ids))}")
    text = "\n\n".join(lines) + "\n\n"
    print(text, end="", flush=True)
    with open(path, "a") as f:
        f.write(text)


def _save_checkpoint(model: GPT, optimizer: torch.optim.Optimizer, step: int, root: Path) -> None:
    # Written under a temporary name so a crash mid-save never looks like a checkpoint.
    partial = root / f"tmp_step_{step:06d}"
    shutil.rmtree(partial, ignore_errors=True)
    model.save(partial)
    torch.save({"optimizer": optimizer.state_dict(), "step": step}, partial / "optimizer.pt")
    partial.rename(root / f"step_{step:06d}")
    for old in sorted(root.glob("step_*"))[:-2]:
        shutil.rmtree(old)


def _truncate_logs(run_dir: Path, step: int) -> None:
    metrics = run_dir / "metrics.csv"
    if metrics.exists():
        header, *rows = metrics.read_text().splitlines(keepends=True)
        metrics.write_text(header + "".join(r for r in rows if int(r.split(",")[0]) <= step))
    samples = run_dir / "samples.txt"
    if samples.exists():
        parts = re.split(r"^--- step (\d+)$", samples.read_text(), flags=re.MULTILINE)
        blocks = zip(parts[1::2], parts[2::2], strict=True)
        samples.write_text("".join(f"--- step {n}{body}" for n, body in blocks if int(n) <= step))


def _resume(
    root: Path, model_config: ModelConfig, config: PretrainConfig, device: str
) -> tuple[GPT, torch.optim.AdamW, int]:
    saved = sorted(root.glob("step_*"))
    if not saved:
        raise FileNotFoundError(f"no checkpoint to resume from in {root}")
    model = GPT.load(saved[-1], device)
    if model.config != model_config:
        raise ValueError(f"checkpoint model {model.config} differs from {model_config}")
    optimizer = make_optimizer(model, config)
    state = torch.load(saved[-1] / "optimizer.pt", map_location=device, weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    print(f"resumed from {saved[-1]}", flush=True)
    return model, optimizer, state["step"]
