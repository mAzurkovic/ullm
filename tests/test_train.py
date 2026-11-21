import argparse
import csv
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from ullm.config import ModelConfig, PretrainConfig, add_config_args, config_from_args
from ullm.data import shard_path
from ullm.model import GPT
from ullm.prompts import SAMPLE_PROMPTS
from ullm.train import lr_at, make_optimizer, pretrain, train_step

TINY = ModelConfig(n_layers=2, d_model=16, n_heads=2, mlp_hidden=32, context_length=8)


def test_pretrain_defaults():
    c = PretrainConfig()
    assert c.batch_tokens == 512 * 1024
    assert c.max_tokens == 7_000_000_000
    assert (c.peak_lr, c.warmup_steps, c.min_lr_ratio) == (3e-4, 700, 0.1)
    assert (c.weight_decay, c.beta1, c.beta2, c.grad_clip) == (0.1, 0.9, 0.95, 1.0)
    assert (c.log_every, c.val_every, c.sample_every, c.checkpoint_every) == (10, 250, 500, 1000)
    assert c.val_batches == 20
    assert c.total_steps == 7_000_000_000 // (512 * 1024)


def test_grad_accum_steps_fill_the_batch():
    c = PretrainConfig(micro_batch=16)
    assert c.grad_accum_steps(context_length=1024) == 32
    assert PretrainConfig(micro_batch=64).grad_accum_steps(1024) == 8


@pytest.mark.parametrize("micro_batch", [0, 3, 1024])
def test_rejects_micro_batch_that_does_not_divide_the_batch(micro_batch):
    with pytest.raises(ValueError):
        PretrainConfig(micro_batch=micro_batch).grad_accum_steps(context_length=1024)


def test_flags_override_config_fields():
    parser = argparse.ArgumentParser()
    add_config_args(parser, ModelConfig)
    add_config_args(parser, PretrainConfig)
    args = parser.parse_args(
        ["--n-layers", "4", "--peak-lr", "1e-3", "--max-tokens", "2e6", "--data", "x", "--resume"]
    )
    model, train = config_from_args(ModelConfig, args), config_from_args(PretrainConfig, args)
    assert model == ModelConfig(n_layers=4)
    assert train == PretrainConfig(peak_lr=1e-3, max_tokens=2_000_000, data=Path("x"), resume=True)


def test_integer_flags_reject_fractions():
    parser = argparse.ArgumentParser()
    add_config_args(parser, PretrainConfig)
    with pytest.raises(SystemExit):
        parser.parse_args(["--max-tokens", "2.5"])


def test_lr_warms_up_then_follows_cosine_to_min():
    c = PretrainConfig(
        peak_lr=1.0, warmup_steps=10, min_lr_ratio=0.1, max_tokens=110, batch_tokens=1
    )
    assert c.total_steps == 110
    assert lr_at(0, c) == pytest.approx(0.1)
    assert lr_at(4, c) == pytest.approx(0.5)
    assert lr_at(9, c) == pytest.approx(1.0)
    assert lr_at(10, c) == pytest.approx(1.0)
    assert lr_at(60, c) == pytest.approx(0.55)
    assert lr_at(35, c) == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi / 4)))
    assert lr_at(110, c) == pytest.approx(0.1)


def test_optimizer_decays_matrices_only():
    model = GPT(TINY)
    optimizer = make_optimizer(model, PretrainConfig())
    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() == 2 for p in decay["params"])
    assert all(p.dim() == 1 for p in no_decay["params"])
    n_params = sum(1 for _ in model.parameters())
    assert len(decay["params"]) + len(no_decay["params"]) == n_params
    assert sum(p is model.tok_emb.weight for p in decay["params"]) == 1
    assert optimizer.defaults["betas"] == (0.9, 0.95)


def test_accumulated_micro_batches_match_one_large_batch():
    torch.manual_seed(0)
    x = torch.randint(0, TINY.vocab_size, (8, TINY.context_length))
    y = torch.randint(0, TINY.vocab_size, (8, TINY.context_length))
    results = []
    for micro_batches in ([(x, y)], [(x[i : i + 2], y[i : i + 2]) for i in range(0, 8, 2)]):
        torch.manual_seed(1)
        model = GPT(TINY)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        loss, norm = train_step(model, optimizer, micro_batches, lr=0.5, grad_clip=1e9)
        results.append((loss, norm, [p.detach().clone() for p in model.parameters()]))
    (loss_a, norm_a, params_a), (loss_b, norm_b, params_b) = results
    assert loss_a == pytest.approx(loss_b, rel=1e-5)
    assert norm_a == pytest.approx(norm_b, rel=1e-4)
    for a, b in zip(params_a, params_b, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_train_step_clips_gradients_and_sets_lr():
    torch.manual_seed(0)
    model = GPT(TINY)
    optimizer = make_optimizer(model, PretrainConfig())
    x = torch.randint(0, TINY.vocab_size, (2, TINY.context_length))
    loss, norm = train_step(model, optimizer, [(x, x)], lr=0.01, grad_clip=1e-3)
    assert loss > 0 and norm > 1e-3
    assert all(group["lr"] == 0.01 for group in optimizer.param_groups)
    clipped = torch.nn.utils.get_total_norm([p.grad for p in model.parameters()])
    assert clipped.item() == pytest.approx(1e-3, rel=1e-3)


def test_tiny_model_overfits_one_batch():
    torch.manual_seed(0)
    config = ModelConfig(
        n_layers=2, d_model=64, n_heads=4, mlp_hidden=128, context_length=16, vocab_size=64
    )
    model = GPT(config)
    optimizer = make_optimizer(model, PretrainConfig())
    x = torch.randint(0, 64, (4, 16))
    y = torch.randint(0, 64, (4, 16))
    for _ in range(300):
        loss, _ = train_step(model, optimizer, [(x, y)], lr=3e-3, grad_clip=1.0)
    assert loss < 0.1


@pytest.fixture
def shards(tmp_path) -> Path:
    rng = np.random.default_rng(0)
    directory = tmp_path / "shards"
    directory.mkdir()
    for i in range(3):
        rng.integers(0, 50257, 2000, dtype=np.uint16).tofile(shard_path(directory, i))
    return directory


def tiny_run(tmp_path: Path, shards: Path, **overrides) -> PretrainConfig:
    settings = dict(
        run="tiny",
        data=shards,
        results_dir=tmp_path / "results",
        checkpoints_dir=tmp_path / "checkpoints",
        batch_tokens=32,
        micro_batch=2,
        max_tokens=32 * 8,
        warmup_steps=2,
        log_every=2,
        val_every=4,
        sample_every=4,
        checkpoint_every=2,
        val_batches=3,
        device="cpu",
    )
    return PretrainConfig(**(settings | overrides))


def read_metrics(run_dir: Path) -> list[dict[str, str]]:
    with open(run_dir / "metrics.csv") as f:
        return list(csv.DictReader(f))


def test_run_logs_metrics_val_loss_samples_and_config(tmp_path, shards, capsys):
    config = tiny_run(tmp_path, shards)
    pretrain(TINY, config)
    run_dir = tmp_path / "results" / "tiny"

    rows = read_metrics(run_dir)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    assert [int(r["step"]) for r in train_rows] == [2, 4, 6, 8]
    assert [int(r["step"]) for r in val_rows] == [4, 8]
    assert all(float(r["loss"]) > 0 and float(r["tokens_per_s"]) > 0 for r in train_rows)
    assert float(train_rows[0]["lr"]) == pytest.approx(lr_at(1, config))

    samples = (run_dir / "samples.txt").read_text()
    assert samples.count("--- step 4") == 1 and samples.count("--- step 8") == 1
    assert all(samples.count(prompt) == 2 for prompt in SAMPLE_PROMPTS)

    saved = json.loads((run_dir / "config.json").read_text())
    assert saved["model"]["n_layers"] == 2 and saved["train"]["batch_tokens"] == 32
    assert "step 8/8" in capsys.readouterr().out


def test_samples_use_a_fixed_seed(tmp_path, shards):
    pretrain(TINY, tiny_run(tmp_path / "a", shards))
    pretrain(TINY, tiny_run(tmp_path / "b", shards))
    a = (tmp_path / "a" / "results" / "tiny" / "samples.txt").read_text()
    b = (tmp_path / "b" / "results" / "tiny" / "samples.txt").read_text()
    assert a == b


def test_run_keeps_last_two_checkpoints_and_final(tmp_path, shards):
    pretrain(TINY, tiny_run(tmp_path, shards))
    ckpt = tmp_path / "checkpoints" / "tiny"
    assert sorted(p.name for p in ckpt.iterdir()) == ["final", "step_000006", "step_000008"]
    for name in ["model.safetensors", "config.json", "optimizer.pt"]:
        assert (ckpt / "step_000008" / name).is_file()
    assert torch.load(ckpt / "step_000008" / "optimizer.pt", weights_only=True)["step"] == 8
    assert GPT.load(ckpt / "final").config == TINY


def test_resume_matches_an_uninterrupted_run(tmp_path, shards):
    config = tiny_run(tmp_path, shards, checkpoint_every=4)
    pretrain(TINY, config)
    ckpt = tmp_path / "checkpoints" / "tiny"
    uninterrupted = GPT.load(ckpt / "final")
    shutil.rmtree(ckpt / "final")
    shutil.rmtree(ckpt / "step_000008")

    pretrain(TINY, replace(config, resume=True))

    resumed = GPT.load(ckpt / "final")
    for a, b in zip(uninterrupted.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    steps = [
        int(r["step"]) for r in read_metrics(tmp_path / "results" / "tiny") if r["split"] == "train"
    ]
    assert steps == [2, 4, 6, 8]
    samples = (tmp_path / "results" / "tiny" / "samples.txt").read_text()
    assert samples.count("--- step 4") == 1 and samples.count("--- step 8") == 1


def test_resume_without_checkpoint_raises(tmp_path, shards):
    with pytest.raises(FileNotFoundError):
        pretrain(TINY, tiny_run(tmp_path, shards, resume=True))


def test_resume_with_other_model_config_raises(tmp_path, shards):
    config = tiny_run(tmp_path, shards)
    pretrain(TINY, config)
    with pytest.raises(ValueError):
        pretrain(replace(TINY, n_layers=3), replace(config, resume=True))


def test_refuses_to_start_over_an_existing_run(tmp_path, shards):
    config = tiny_run(tmp_path, shards)
    pretrain(TINY, config)
    with pytest.raises(FileExistsError):
        pretrain(TINY, config)


def test_crash_while_saving_leaves_no_checkpoint(tmp_path, shards, monkeypatch):
    def disk_full(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("ullm.train.torch.save", disk_full)
    with pytest.raises(OSError):
        pretrain(TINY, tiny_run(tmp_path, shards, checkpoint_every=4))
    assert not list((tmp_path / "checkpoints" / "tiny").glob("step_*"))


def test_resume_keeps_samples_that_contain_header_text(tmp_path, shards):
    config = tiny_run(tmp_path, shards, checkpoint_every=4)
    pretrain(TINY, config)
    samples = tmp_path / "results" / "tiny" / "samples.txt"
    samples.write_text(samples.read_text().replace("> ", "> --- step 1: mix the flour. ", 1))
    shutil.rmtree(tmp_path / "checkpoints" / "tiny" / "step_000008")
    pretrain(TINY, replace(config, resume=True))
    assert "--- step 1: mix the flour" in samples.read_text()
