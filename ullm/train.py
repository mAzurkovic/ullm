import json
import math
import re
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F

from ullm.config import ModelConfig, PretrainConfig
from ullm.data import TokenLoader, split_shards
from ullm.generate import generate
from ullm.model import GPT, default_device
from ullm.prompts import SAMPLE_PROMPTS
from ullm.tokenizer import ENDOFTEXT, decode, encode

Batch = tuple[torch.Tensor, torch.Tensor]

METRICS_HEADER = "step,split,loss,lr,grad_norm,tokens_per_s,elapsed_s\n"
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
    model: GPT,
    optimizer: torch.optim.Optimizer,
    micro_batches: Sequence[Batch],
    lr: float,
    grad_clip: float,
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), device=next(model.parameters()).device)
    for x, y in micro_batches:
        loss = _loss(model, x, y) / len(micro_batches)
        loss.backward()
        total += loss.detach()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    return total.item(), norm.item()


def pretrain(model_config: ModelConfig, config: PretrainConfig) -> Path:
    device = default_device() if config.device == "auto" else config.device
    torch.manual_seed(config.seed)
    accum = config.grad_accum_steps(model_config.context_length)
    val_path, train_paths = split_shards(config.data)
    run_dir = config.results_dir / config.run
    checkpoints = config.checkpoints_dir / config.run

    if config.resume:
        model, optimizer, start_step = _resume(checkpoints, model_config, config, device)
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
    loader = iter(
        TokenLoader(
            train_paths,
            config.micro_batch,
            model_config.context_length,
            device=device,
            start_batch=start_step * accum,
        )
    )
    val_loader = TokenLoader(
        [val_path], config.micro_batch, model_config.context_length, device=device
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"model": asdict(model_config), "train": asdict(config)}, indent=2, default=str)
    )
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        metrics_path.write_text(METRICS_HEADER)

    start = time.time()
    tokens_since_log = 0
    train_seconds = 0.0
    with open(metrics_path, "a") as metrics:
        for step in range(start_step, config.total_steps):
            lr = lr_at(step, config)
            step_start = time.time()
            batches = [next(loader) for _ in range(accum)]
            loss, grad_norm = train_step(model, optimizer, batches, lr, config.grad_clip)
            train_seconds += time.time() - step_start
            done = step + 1
            last = done == config.total_steps
            tokens_since_log += config.batch_tokens

            if done % config.log_every == 0 or last:
                rate = tokens_since_log / train_seconds
                elapsed = time.time() - start
                metrics.write(
                    f"{done},train,{loss:.4f},{lr:.6g},{grad_norm:.4f},{rate:.0f},{elapsed:.1f}\n"
                )
                metrics.flush()
                print(
                    f"step {done}/{config.total_steps} | loss {loss:.4f} | lr {lr:.2e} | "
                    f"grad_norm {grad_norm:.2f} | tok/s {rate:,.0f} | elapsed {elapsed:.0f}s",
                    flush=True,
                )
                tokens_since_log, train_seconds = 0, 0.0
            if done % config.val_every == 0 or last:
                val = _val_loss(model, val_loader, config.val_batches)
                metrics.write(f"{done},val,{val:.4f},,,,{time.time() - start:.1f}\n")
                metrics.flush()
                print(f"step {done} | val loss {val:.4f}", flush=True)
            if done % config.sample_every == 0 or last:
                _write_samples(model, done, run_dir / "samples.txt", config.seed)
            if done % config.checkpoint_every == 0:
                _save_checkpoint(model, optimizer, done, checkpoints)

    final = checkpoints / "final"
    model.save(final)
    return final


def _loss(model: GPT, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
