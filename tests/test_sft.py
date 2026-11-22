import argparse
import csv
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
import torch

from ullm.cli import main
from ullm.config import ModelConfig, SFTConfig, add_config_args, config_from_args
from ullm.data import IGNORE_INDEX, SFTLoader, sft_example
from ullm.model import GPT
from ullm.prompts import CHAT_PROMPTS
from ullm.tokenizer import ASSISTANT, END, ENDOFTEXT, USER
from ullm.train import sft_lr_at, train_step


def write_rows(directory: Path, split: str, ids: np.ndarray, mask: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{split}_ids.npy", ids.astype(np.uint16))
    np.save(directory / f"{split}_mask.npy", mask.astype(np.uint8))


def numbered_rows(directory: Path, split: str, rows: int, width: int = 6) -> None:
    ids = np.arange(rows * width).reshape(rows, width)
    mask = np.tile(np.arange(width) % 2, (rows, 1))
    write_rows(directory, split, ids, mask)


def test_sft_defaults():
    c = SFTConfig(base=Path("ckpt"))
    assert (c.batch_size, c.micro_batch, c.epochs) == (64, 16, 1)
    assert (c.peak_lr, c.warmup_steps) == (3e-5, 100)
    assert (c.weight_decay, c.beta1, c.beta2, c.grad_clip) == (0.1, 0.9, 0.95, 1.0)
    assert c.data == Path("data/sft") and c.budget == 25.0
    assert c.total_steps(rows=6400) == 100
    assert c.total_steps(rows=6463) == 100
    assert SFTConfig(base=Path("x"), epochs=2).total_steps(rows=6400) == 200


def test_sft_grad_accum_steps_fill_the_batch():
    c = SFTConfig(base=Path("x"))
    assert c.grad_accum_steps() == 4
    assert c.grad_accum_steps(world_size=4) == 1
    with pytest.raises(ValueError):
        SFTConfig(base=Path("x"), micro_batch=24).grad_accum_steps()
    with pytest.raises(ValueError):
        c.grad_accum_steps(world_size=8)


def test_sft_flags_require_base():
    parser = argparse.ArgumentParser()
    add_config_args(parser, SFTConfig)
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["--base", "ckpt", "--peak-lr", "1e-4", "--compile"])
    assert config_from_args(SFTConfig, args) == SFTConfig(
        base=Path("ckpt"), peak_lr=1e-4, compile=True
    )


def test_loader_yields_inputs_and_masked_next_token_targets(tmp_path):
    ids = np.array([[USER, 7, END, ASSISTANT, 8, 9, END, ENDOFTEXT]] * 2)
    mask = np.array([[0, 0, 0, 0, 1, 1, 1, 0]] * 2)
    write_rows(tmp_path, "train", ids, mask)
    x, y = next(iter(SFTLoader(tmp_path, "train", batch_size=2)))
    assert x.dtype == y.dtype == torch.int64
    assert x.tolist() == [[USER, 7, END, ASSISTANT, 8, 9, END]] * 2
    ignore = IGNORE_INDEX
    assert y.tolist() == [[ignore, ignore, ignore, 8, 9, END, ignore]] * 2


def test_loader_reads_every_row_once_per_epoch_in_shuffled_order(tmp_path):
    numbered_rows(tmp_path, "train", rows=12)
    loader = SFTLoader(tmp_path, "train", batch_size=3, seed=1)
    batches = list(islice(iter(loader), 8))
    first, second = [torch.cat([x for x, _ in batches[i : i + 4]]) for i in (0, 4)]
    for epoch in (first, second):
        assert sorted(row[0] // 6 for row in epoch.tolist()) == list(range(12))
    assert first[:, 0].tolist() != list(range(0, 72, 6))
    assert first.tolist() != second.tolist()


def test_loader_order_repeats_for_the_same_seed(tmp_path):
    numbered_rows(tmp_path, "val", rows=8)
    a = [x.tolist() for x, _ in islice(iter(SFTLoader(tmp_path, "val", 2, seed=3)), 4)]
    b = [x.tolist() for x, _ in islice(iter(SFTLoader(tmp_path, "val", 2, seed=3)), 4)]
    assert a == b


def test_processes_read_disjoint_rows_that_cover_the_epoch(tmp_path):
    numbered_rows(tmp_path, "train", rows=12)
    seen = []
    for rank in range(2):
        loader = SFTLoader(tmp_path, "train", batch_size=2, rank=rank, world_size=2)
        seen.append({row[0] // 6 for x, _ in islice(iter(loader), 3) for row in x.tolist()})
    assert not seen[0] & seen[1]
    assert seen[0] | seen[1] == set(range(12))


@pytest.mark.parametrize(
    "rows, batch_size, world_size, rank",
    [(3, 4, 1, 0), (6, 2, 4, 0), (8, 2, 2, 2)],
)
def test_loader_rejects_bad_sizes(tmp_path, rows, batch_size, world_size, rank):
    numbered_rows(tmp_path, "train", rows=rows)
    with pytest.raises(ValueError):
        SFTLoader(tmp_path, "train", batch_size, rank=rank, world_size=world_size)


def test_loader_rejects_mismatched_ids_and_mask(tmp_path):
    write_rows(tmp_path, "train", np.zeros((4, 6)), np.zeros((4, 5)))
    with pytest.raises(ValueError):
        SFTLoader(tmp_path, "train", batch_size=2)


BASE = ModelConfig(n_layers=2, d_model=32, n_heads=4, mlp_hidden=64, context_length=16)
QA = [("Hi", "Hello!"), ("Say yes.", "Yes."), ("Name a color.", "Blue."), ("Bye", "See you.")]


def qa_rows(directory: Path, split: str, copies: int, width: int = BASE.context_length + 1):
    ids, mask = [], []
    for _ in range(copies):
        for question, answer in QA:
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            example = sft_example(messages, max_tokens=width)
            pad = width - len(example.ids)
            ids.append(example.ids + [ENDOFTEXT] * pad)
            mask.append(example.mask + [0] * pad)
    write_rows(directory, split, np.array(ids), np.array(mask))


@pytest.fixture
def setup(tmp_path) -> tuple[Path, Path]:
    torch.manual_seed(0)
    base = tmp_path / "base"
    GPT(BASE).save(base)
    data = tmp_path / "sft"
    qa_rows(data, "train", copies=8)
    qa_rows(data, "val", copies=1)
    return base, data


def sft_args(tmp_path: Path, base: Path, data: Path, **flags: str) -> list[str]:
    settings = {
        "--base": str(base), "--data": str(data), "--run": "tiny",
        "--results-dir": str(tmp_path / "results"), "--checkpoints-dir": str(tmp_path / "ckpt"),
        "--batch-size": "8", "--micro-batch": "4", "--epochs": "6", "--peak-lr": "3e-3",
        "--warmup-steps": "2", "--log-every": "1", "--val-every": "4", "--sample-every": "12",
        "--checkpoint-every": "1000", "--val-batches": "1", "--device": "cpu",
    } | flags  # fmt: skip
    return ["sft", *[part for flag, value in settings.items() for part in (flag, value)]]


def read_metrics(run_dir: Path) -> list[dict[str, str]]:
    with open(run_dir / "metrics.csv") as f:
        return list(csv.DictReader(f))


def test_sft_lr_warms_up_then_falls_linearly_to_zero():
    c = SFTConfig(base=Path("x"), peak_lr=1.0, warmup_steps=10)
    assert sft_lr_at(0, 110, c) == pytest.approx(0.1)
    assert sft_lr_at(9, 110, c) == pytest.approx(1.0)
    assert sft_lr_at(10, 110, c) == pytest.approx(1.0)
    assert sft_lr_at(60, 110, c) == pytest.approx(0.5)
    assert sft_lr_at(110, 110, c) == 0.0


def test_loss_averages_only_the_assistant_targets(setup):
    _, data = setup
    torch.manual_seed(0)
    model = GPT(BASE)
    x, y = next(iter(SFTLoader(data, "train", batch_size=4)))
    with torch.no_grad():
        logits = model(x)
    learn = y != IGNORE_INDEX
    assert 0 < learn.sum() < learn.numel()
    by_hand = torch.nn.functional.cross_entropy(logits[learn], y[learn])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    loss, _ = train_step(model, optimizer, [(x, y)], lr=0.0, grad_clip=1.0)
    assert loss == pytest.approx(by_hand.item(), rel=1e-5)


def test_sft_command_fine_tunes_and_writes_run_outputs(tmp_path, setup, capsys):
    base, data = setup
    main(sft_args(tmp_path, base, data))

    rows = read_metrics(tmp_path / "results" / "tiny")
    train = [float(r["loss"]) for r in rows if r["split"] == "train"]
    val = [float(r["loss"]) for r in rows if r["split"] == "val"]
    assert len(train) == 6 * 32 // 8
    assert train[-1] < train[0] - 1.0
    assert len(val) == 6 and val[-1] < val[0]

    final = tmp_path / "ckpt" / "tiny" / "final"
    assert GPT.load(final).config == BASE
    samples = (tmp_path / "results" / "tiny" / "samples.txt").read_text()
    for prompt in CHAT_PROMPTS:
        assert f"> <|user|>{prompt}<|end|><|assistant|>" in samples
    out = capsys.readouterr().out
    assert "step 24/24" in out


def test_sft_starts_from_the_base_weights(tmp_path, setup):
    base, data = setup
    main(sft_args(tmp_path, base, data, **{"--epochs": "1", "--peak-lr": "1e-9"}))
    tuned, original = GPT.load(tmp_path / "ckpt" / "tiny" / "final"), GPT.load(base)
    for a, b in zip(tuned.parameters(), original.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)


def test_sft_rejects_rows_that_do_not_match_the_context(tmp_path, setup):
    base, data = setup
    qa_rows(data, "train", copies=8, width=BASE.context_length + 5)
    with pytest.raises(ValueError, match="context"):
        main(sft_args(tmp_path, base, data))


def test_sft_rejects_data_smaller_than_one_batch(tmp_path, setup):
    base, data = setup
    with pytest.raises(ValueError, match="one batch"):
        main(sft_args(tmp_path, base, data, **{"--batch-size": "64", "--epochs": "1"}))


def test_sft_refuses_to_start_over_an_existing_run(tmp_path, setup):
    base, data = setup
    main(sft_args(tmp_path, base, data, **{"--epochs": "1"}))
    with pytest.raises(FileExistsError):
        main(sft_args(tmp_path, base, data, **{"--epochs": "1"}))


def test_sft_stops_at_budget_check_with_advice(tmp_path, setup, capsys):
    base, data = setup
    flags = {"--hourly-rate": "1e9", "--budget-check-step": "2"}
    main(sft_args(tmp_path, base, data, **flags))
    assert not (tmp_path / "ckpt" / "tiny" / "final").exists()
    assert (tmp_path / "ckpt" / "tiny" / "step_000002").is_dir()
    assert "over budget" in capsys.readouterr().out


def test_sft_help_lists_flags(capsys):
    with pytest.raises(SystemExit):
        main(["sft", "--help"])
    out = capsys.readouterr().out
    assert all(flag in out for flag in ["--base", "--batch-size", "--epochs", "--hourly-rate"])
