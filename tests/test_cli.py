import io
import subprocess
import sys

import pytest
import torch

from ullm.cli import main, trim_history
from ullm.config import ModelConfig
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
    "flags", [["--temperature", "-1"], ["--top-k", "-1"], ["--max-new-tokens", "0"]]
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
