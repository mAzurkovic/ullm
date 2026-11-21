import io
import subprocess
import sys

import numpy as np
import pytest
import torch

from ullm.cli import chat_loop, main, trim_history
from ullm.config import ModelConfig
from ullm.data import shard_path
from ullm.generate import generate
from ullm.model import GPT
from ullm.tokenizer import ASSISTANT, END, Message, decode, encode, render_chat

CONFIG = ModelConfig(
    n_layers=2, d_model=32, n_heads=4, mlp_hidden=96, context_length=64, vocab_size=50304
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("ckpt")
    GPT(CONFIG).save(path)
    return path


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "complete" in out and "chat" in out


def test_complete_streams_tokens(checkpoint, capsys):
    main(["complete", "Once upon a time", "--model", str(checkpoint), "--max-new-tokens", "5"])
    assert capsys.readouterr().out.strip()


def test_complete_is_reproducible_with_seed(checkpoint, capsys):
    args = ["complete", "Hello", "--model", str(checkpoint), "--max-new-tokens", "8"]
    main([*args, "--seed", "3"])
    first = capsys.readouterr().out
    main([*args, "--seed", "3"])
    assert capsys.readouterr().out == first


def test_complete_rejects_empty_prompt(checkpoint, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["complete", "", "--model", str(checkpoint)])
    assert exit_info.value.code == 2


def test_complete_on_cpu_device(checkpoint, capsys):
    main(["complete", "Hi", "--model", str(checkpoint), "--max-new-tokens", "2", "--device", "cpu"])
    assert capsys.readouterr().out


def test_chat_answers_each_line_until_eof(checkpoint, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("Hi there\n\nHow are you?\n"))
    main(["chat", "--model", str(checkpoint), "--max-new-tokens", "4", "--seed", "0"])
    out = capsys.readouterr().out
    assert out.count("> ") == 4


def test_trim_history_keeps_history_that_fits():
    history = [Message("user", "a"), Message("assistant", "b"), Message("user", "c")]
    assert trim_history(history, context_length=64) == history


def test_trim_history_drops_oldest_full_turns():
    turn = [Message("user", "one two three"), Message("assistant", "four five six")]
    last = Message("user", "last")
    assert [len(encode(m.content)) for m in [*turn, last]] == [3, 3, 1]
    # Each turn renders to 10 tokens, the last message to 3, plus 1 for the reply marker.
    history = turn * 3 + [last]
    assert trim_history(history, context_length=24) == turn * 2 + [last]
    assert trim_history(history, context_length=23) == turn + [last]


def test_trim_history_keeps_latest_message_even_if_too_long():
    history = [Message("user", "a"), Message("assistant", "b"), Message("user", "word " * 100)]
    assert trim_history(history, context_length=24) == history[-1:]


def test_ctrl_c_exits_cleanly(checkpoint, capsys, monkeypatch):
    def interrupt(prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    main(["chat", "--model", str(checkpoint)])
    assert capsys.readouterr().out == "\n"


def test_falls_back_to_cpu_without_gpu(checkpoint, capsys, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    main(["complete", "Hi", "--model", str(checkpoint), "--max-new-tokens", "2"])
    assert capsys.readouterr().out


def test_runs_as_module():
    result = subprocess.run(
        [sys.executable, "-m", "ullm", "--help"], capture_output=True, text=True, check=True
    )
    assert "complete" in result.stdout


@pytest.mark.parametrize(
    "flags",
    [
        ["--temperature", "-1"],
        ["--temperature", "nan"],
        ["--top-k", "-1"],
        ["--max-new-tokens", "0"],
    ],
)
def test_rejects_invalid_sampling_flags(checkpoint, flags):
    with pytest.raises(SystemExit) as exit_info:
        main(["complete", "Hi", "--model", str(checkpoint), *flags])
    assert exit_info.value.code == 2


def test_chat_sends_conversation_history_to_model(checkpoint, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("Hi\nHow are you?\n"))
    main(["chat", "--model", str(checkpoint), "--temperature", "0", "--max-new-tokens", "4"])
    out = capsys.readouterr().out

    model = GPT.load(checkpoint).eval()

    def greedy_reply(history: list[Message]) -> str:
        ids = render_chat(history)[0] + [ASSISTANT]
        settings = dict(max_new_tokens=4, temperature=0.0, top_k=50, stop_ids={END})
        return decode(list(generate(model, ids, **settings)))

    first = greedy_reply([Message("user", "Hi")])
    second = greedy_reply(
        [Message("user", "Hi"), Message("assistant", first), Message("user", "How are you?")]
    )
    assert out == f"> {first}\n> {second}\n> \n"


def test_rejects_missing_checkpoint(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["complete", "Hi", "--model", str(tmp_path / "missing")])
    assert exit_info.value.code == 2
    assert "no checkpoint" in capsys.readouterr().err


def test_chat_loop_sends_history_and_new_message(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("Hi\nHow are you?\n"))
    prompts = []

    def reply(ids: list[int], stop_ids) -> str:
        prompts.append((ids, set(stop_ids)))
        return "ok"

    chat_loop(reply, context_length=1024)

    history = [Message("user", "Hi"), Message("assistant", "ok"), Message("user", "How are you?")]
    assert prompts == [
        (render_chat(history[:1])[0] + [ASSISTANT], {END}),
        (render_chat(history)[0] + [ASSISTANT], {END}),
    ]


def test_module_does_not_run_cli_when_imported_by_worker_processes():
    code = "import runpy; runpy.run_module('ullm', run_name='__mp_main__')"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_pretrain_help_lists_model_and_training_flags(capsys):
    with pytest.raises(SystemExit):
        main(["pretrain", "--help"])
    out = capsys.readouterr().out
    assert all(flag in out for flag in ["--n-layers", "--peak-lr", "--micro-batch", "--resume"])


def test_pretrain_command_trains_from_flags(tmp_path):
    data = tmp_path / "shards"
    data.mkdir()
    for i in range(2):
        np.random.default_rng(i).integers(0, 50257, 600, dtype=np.uint16).tofile(
            shard_path(data, i)
        )
    flags = {
        "--n-layers": "1", "--d-model": "16", "--n-heads": "2", "--mlp-hidden": "32",
        "--context-length": "8", "--run": "cli", "--data": str(data),
        "--results-dir": str(tmp_path / "results"), "--checkpoints-dir": str(tmp_path / "ckpt"),
        "--batch-tokens": "16", "--micro-batch": "2", "--max-tokens": "64",
        "--warmup-steps": "1", "--device": "cpu",
    }  # fmt: skip
    main(["pretrain", *[part for pair in flags.items() for part in pair]])
    assert GPT.load(tmp_path / "ckpt" / "cli" / "final").config.n_layers == 1
    assert (tmp_path / "results" / "cli" / "metrics.csv").is_file()
