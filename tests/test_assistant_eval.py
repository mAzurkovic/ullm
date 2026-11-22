import pytest
import torch

from ullm.cli import main
from ullm.config import ModelConfig
from ullm.eval import Reply, assistant_test, write_assistant_report
from ullm.generate import generate
from ullm.model import GPT
from ullm.prompts import ASSISTANT_CONVERSATIONS, ASSISTANT_PROMPTS, CHAT_PROMPTS
from ullm.tokenizer import ASSISTANT, END, USER, Message, decode, render_chat

TINY = ModelConfig(n_layers=1, d_model=16, n_heads=2, mlp_hidden=32, context_length=256)
PROMPTS = ["Hi", "Name a color."]
CONVERSATIONS = [["Tell me a fact.", "Why?"], ["Say a word.", "Another.", "One more."]]


@pytest.fixture(scope="module")
def model() -> GPT:
    torch.manual_seed(0)
    return GPT(TINY).eval()


def always_next(token: int) -> GPT:
    """A tiny model that answers every prompt ending in <|assistant|> with `token`, repeated."""
    torch.manual_seed(0)
    gpt = GPT(TINY).eval()
    with torch.no_grad():
        for layer in gpt.layers:
            layer.attn.wo.weight.zero_()
            layer.mlp.w2.weight.zero_()
        gpt.tok_emb.weight[ASSISTANT] = 10.0
        gpt.tok_emb.weight[token] = 20.0
    return gpt


def test_assistant_prompts_are_fixed():
    assert len(ASSISTANT_PROMPTS) == 20
    assert len(set(ASSISTANT_PROMPTS)) == 20
    assert not set(ASSISTANT_PROMPTS) & set(CHAT_PROMPTS)


def test_assistant_conversations_have_two_or_three_turns():
    assert len(ASSISTANT_CONVERSATIONS) == 3
    assert all(2 <= len(turns) <= 3 for turns in ASSISTANT_CONVERSATIONS)


def test_one_reply_per_prompt_and_conversation_turn(model):
    replies = assistant_test(model, PROMPTS, CONVERSATIONS, max_new_tokens=6)
    assert len(replies) == 2 + 5
    assert [r.history[-1].content for r in replies[:2]] == PROMPTS
    assert [r.conversation for r in replies] == [None, None, 0, 0, 1, 1, 1]
    assert replies == assistant_test(model, PROMPTS, CONVERSATIONS, max_new_tokens=6)


def test_conversation_turns_see_earlier_turns_and_replies(model):
    first, second = assistant_test(model, [], CONVERSATIONS[:1], max_new_tokens=6)
    assert first.history == [Message("user", "Tell me a fact.")]
    assert second.history == [
        Message("user", "Tell me a fact."),
        Message("assistant", first.text),
        Message("user", "Why?"),
    ]
    ids = render_chat(second.history)[0] + [ASSISTANT]
    tokens = generate(
        model,
        ids,
        max_new_tokens=6,
        temperature=0.5,
        top_k=50,
        stop_ids={END},
        generator=torch.Generator().manual_seed(0),
    )
    assert second.text == decode(list(tokens))


def test_reply_that_ends_its_turn_is_marked_stopped():
    [reply] = assistant_test(always_next(END), ["Hi"], [], max_new_tokens=8)
    assert reply.stopped and not reply.wrote_user and reply.text == ""


def test_reply_that_writes_a_user_turn_is_flagged():
    [reply] = assistant_test(always_next(USER), ["Hi"], [], max_new_tokens=8)
    assert reply.wrote_user and not reply.stopped
    assert reply.text == "<|user|>" * 8


def test_report_lists_replies_with_checks_and_checkboxes(tmp_path):
    replies = [
        Reply([Message("user", "Hi")], "Hello!", stopped=True, wrote_user=False),
        Reply(
            [Message("user", "Say a word.")],
            "Blue blue blue",
            stopped=False,
            wrote_user=True,
            conversation=0,
        ),
    ]
    path = tmp_path / "chat" / "results.md"
    write_assistant_report(path, "chat", replies)
    report = path.read_text()
    assert report.startswith("# chat\n")
    assert "Stopped on `<|end|>`: 1 of 2. Wrote `<|user|>`: 1 of 2." in report
    assert report.count("- [ ] pass") == 2
    assert "### 1. Hi" in report and "Hello!" in report
    assert "### Conversation 1" in report and "**User:** Say a word." in report
    assert "stopped on `<|end|>`: yes | wrote `<|user|>`: no" in report
    assert "stopped on `<|end|>`: no | wrote `<|user|>`: yes" in report


def test_eval_assistant_writes_only_the_assistant_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    GPT(TINY).save(tmp_path / "ckpt")
    main(
        [
            "eval",
            "--assistant",
            "--model",
            str(tmp_path / "ckpt"),
            "--run",
            "chat",
            "--device",
            "cpu",
            "--max-new-tokens",
            "4",
        ]
    )
    run_dir = tmp_path / "results" / "chat"
    report = (run_dir / "results.md").read_text()
    assert not (run_dir / "benchmarks.json").exists()
    assert "HellaSwag" not in report
    turns = sum(len(t) for t in ASSISTANT_CONVERSATIONS)
    assert report.count("- [ ] pass") == len(ASSISTANT_PROMPTS) + turns
    assert all(prompt in report for prompt in ASSISTANT_PROMPTS)


def test_eval_assistant_rejects_gpt2(capsys):
    with pytest.raises(SystemExit):
        main(["eval", "--assistant", "--model", "gpt2", "--run", "x"])
    assert "--assistant" in capsys.readouterr().err
