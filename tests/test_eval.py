from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from transformers import GPT2Config, GPT2LMHeadModel

from ullm.cli import main
from ullm.config import ModelConfig
from ullm.eval import (
    Benchmarks,
    LogitsLM,
    generation_test,
    load_benchmarks,
    load_gpt2,
    run_benchmarks,
    save_benchmarks,
)
from ullm.model import GPT
from ullm.prompts import BASE_PROMPTS
from ullm.tokenizer import ENDOFTEXT, encode

TINY = ModelConfig(n_layers=1, d_model=16, n_heads=2, mlp_hidden=32, context_length=16)


@pytest.fixture(scope="module")
def model() -> GPT:
    torch.manual_seed(0)
    return GPT(TINY).eval()


def request(kind: str, *args: str) -> Instance:
    return Instance(request_type=kind, doc={}, arguments=args, idx=0)


def manual_logprob(model: GPT, tokens: list[int], n_continuation: int) -> tuple[float, bool]:
    with torch.no_grad():
        logits = model(torch.tensor([tokens[:-1]]))[0]
    logprobs = F.log_softmax(logits.float(), dim=-1)
    targets = torch.tensor(tokens[1:])
    picked = logprobs[torch.arange(len(targets)), targets][-n_continuation:]
    greedy = logprobs.argmax(-1)[-n_continuation:]
    return picked.sum().item(), bool((greedy == targets[-n_continuation:]).all())


def test_loglikelihood_sums_continuation_logprobs(model):
    context, continuation = "The cat sat on the", " mat"
    [(score, is_greedy)] = LogitsLM(model, TINY.context_length).loglikelihood(
        [request("loglikelihood", context, continuation)]
    )
    tokens = encode(context + continuation)
    expected, expected_greedy = manual_logprob(model, tokens, len(tokens) - len(encode(context)))
    assert score == pytest.approx(expected, abs=1e-4)
    assert is_greedy == expected_greedy


def test_loglikelihood_left_truncates_long_context(model):
    context = " ".join(f"word{i}" for i in range(40))
    [(score, _)] = LogitsLM(model, TINY.context_length).loglikelihood(
        [request("loglikelihood", context, " end")]
    )
    tokens = encode(context + " end")
    n_continuation = len(tokens) - len(encode(context))
    expected, _ = manual_logprob(model, tokens[-(TINY.context_length + 1) :], n_continuation)
    assert score == pytest.approx(expected, abs=1e-4)


def test_empty_context_starts_from_endoftext(model):
    [(score, _)] = LogitsLM(model, TINY.context_length).loglikelihood(
        [request("loglikelihood", "", "Hello world")]
    )
    tokens = [ENDOFTEXT, *encode("Hello world")]
    expected, _ = manual_logprob(model, tokens, len(tokens) - 1)
    assert score == pytest.approx(expected, abs=1e-4)


def test_trailing_space_of_context_moves_to_continuation(model):
    lm = LogitsLM(model, TINY.context_length)
    [(a, _)] = lm.loglikelihood([request("loglikelihood", "The cat sat on the ", "mat")])
    [(b, _)] = lm.loglikelihood([request("loglikelihood", "The cat sat on the", " mat")])
    assert a == pytest.approx(b, abs=1e-5)


def test_batched_scores_match_one_by_one(model):
    requests = [
        request("loglikelihood", "One two", " three"),
        request("loglikelihood", "A much longer context with many more words in it", " here"),
        request("loglikelihood", "Short", " and then a longer continuation"),
    ]
    batched = LogitsLM(model, TINY.context_length, batch_size=8).loglikelihood(requests)
    single = LogitsLM(model, TINY.context_length, batch_size=1).loglikelihood(requests)
    for (a, ga), (b, gb) in zip(batched, single, strict=True):
        assert a == pytest.approx(b, abs=1e-4) and ga == gb


def test_rolling_loglikelihood_scores_every_token_once(model):
    text = " ".join(f"token{i}" for i in range(30))
    [score] = LogitsLM(model, TINY.context_length).loglikelihood_rolling(
        [request("loglikelihood_rolling", text)]
    )
    tokens = encode(text)
    padded = [ENDOFTEXT, *tokens]
    n = TINY.context_length
    expected = 0.0
    for start in range(0, len(tokens), n):
        end = min(start + n, len(tokens))
        window = padded[max(0, end - n) : end + 1]
        expected += manual_logprob(model, window, end - start)[0]
    assert len(tokens) > 2 * n
    assert score == pytest.approx(expected, abs=1e-3)


def test_generate_until_is_not_supported(model):
    with pytest.raises(NotImplementedError):
        LogitsLM(model, TINY.context_length).generate_until([request("generate_until", "Hi", {})])


def fake_results(acc_norm: float, perplexity: float) -> dict:
    return {
        "results": {
            "hellaswag": {"acc_norm,none": acc_norm},
            "wikitext": {"word_perplexity,none": perplexity},
        }
    }


@pytest.fixture
def checkpoint(tmp_path) -> Path:
    torch.manual_seed(0)
    path = tmp_path / "ckpt"
    GPT(TINY).save(path)
    return path


def test_benchmarks_on_checkpoint_use_the_ullm_adapter(checkpoint, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "lm_eval.simple_evaluate", lambda **kw: calls.append(kw) or fake_results(0.3, 40.0)
    )
    result = run_benchmarks(str(checkpoint), device="cpu", limit=5)
    assert result == Benchmarks(hellaswag_acc_norm=0.3, wikitext_word_perplexity=40.0)
    [call] = calls
    assert isinstance(call["model"], LogitsLM)
    assert call["tasks"] == ["hellaswag", "wikitext"]
    assert call["num_fewshot"] == 0 and call["limit"] == 5


TINY_GPT2 = GPT2Config(n_layer=1, n_embd=16, n_head=2, n_positions=24)


@pytest.fixture
def tiny_gpt2(monkeypatch) -> GPT2LMHeadModel:
    torch.manual_seed(0)
    net = GPT2LMHeadModel(TINY_GPT2)
    monkeypatch.setattr(
        "transformers.GPT2LMHeadModel.from_pretrained", lambda name: net if name == "gpt2" else None
    )
    return net


def test_benchmarks_on_gpt2_use_the_same_adapter(tiny_gpt2, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "lm_eval.simple_evaluate", lambda **kw: calls.append(kw) or fake_results(0.31, 37.4)
    )
    assert run_benchmarks("gpt2", device="cpu", limit=None).hellaswag_acc_norm == 0.31
    [call] = calls
    assert isinstance(call["model"], LogitsLM)
    assert call["model"].context_length == TINY_GPT2.n_positions
    assert call["tasks"] == ["hellaswag", "wikitext"] and call["num_fewshot"] == 0


def test_gpt2_scores_match_its_own_logits(tiny_gpt2):
    lm = load_gpt2("cpu")
    tokens = encode("The cat sat on the mat")
    [(score, _)] = LogitsLM(lm, TINY_GPT2.n_positions).loglikelihood(
        [request("loglikelihood", "The cat sat", " on the mat")]
    )
    with torch.no_grad():
        logprobs = F.log_softmax(tiny_gpt2(torch.tensor([tokens[:-1]])).logits[0], dim=-1)
    n = len(encode(" on the mat"))
    targets = torch.tensor(tokens[1:])
    expected = logprobs[torch.arange(len(targets)), targets][-n:].sum().item()
    assert score == pytest.approx(expected, rel=1e-5)


def test_benchmarks_round_trip(tmp_path):
    benchmarks = Benchmarks(hellaswag_acc_norm=0.3123, wikitext_word_perplexity=35.5)
    save_benchmarks(benchmarks, tmp_path / "benchmarks.json")
    assert load_benchmarks(tmp_path / "benchmarks.json") == benchmarks


def test_base_prompts_are_fixed():
    assert len(BASE_PROMPTS) == 10
    assert len(set(BASE_PROMPTS)) == 10


def test_generation_test_gives_three_samples_per_prompt(model):
    samples = generation_test(model, max_new_tokens=8)
    assert len(samples) == 30
    assert [s.prompt for s in samples[:3]] == [BASE_PROMPTS[0]] * 3
    assert [s.seed for s in samples[:3]] == [0, 1, 2]
    assert samples == generation_test(model, max_new_tokens=8)


def test_eval_command_writes_results_beside_baseline(tmp_path, checkpoint, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("lm_eval.simple_evaluate", lambda **kw: fake_results(0.35, 30.0))
    save_benchmarks(Benchmarks(0.31, 37.4), tmp_path / "results" / "gpt2-small" / "benchmarks.json")
    main(
        [
            "eval",
            "--model",
            str(checkpoint),
            "--run",
            "base",
            "--limit",
            "5",
            "--device",
            "cpu",
            "--max-new-tokens",
            "8",
        ]
    )

    run_dir = tmp_path / "results" / "base"
    assert load_benchmarks(run_dir / "benchmarks.json") == Benchmarks(0.35, 30.0)
    report = (run_dir / "results.md").read_text()
    assert "| HellaSwag acc_norm (0-shot) | 0.3500 | 0.3100 | pass |" in report
    assert "| WikiText word perplexity | 30.00 | 37.40 | pass |" in report
    assert all(prompt in report for prompt in BASE_PROMPTS)
    assert report.count("- [ ] pass") == 10
    assert report.count("**Seed ") == 30


def test_eval_marks_failures_and_missing_baseline(tmp_path, checkpoint, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("lm_eval.simple_evaluate", lambda **kw: fake_results(0.25, 50.0))
    args = [
        "eval",
        "--model",
        str(checkpoint),
        "--run",
        "base",
        "--device",
        "cpu",
        "--max-new-tokens",
        "4",
    ]
    main(args)
    assert "baseline missing" in (tmp_path / "results" / "base" / "results.md").read_text()

    save_benchmarks(Benchmarks(0.31, 37.4), tmp_path / "results" / "gpt2-small" / "benchmarks.json")
    main(args)
    report = (tmp_path / "results" / "base" / "results.md").read_text()
    assert "| 0.2500 | 0.3100 | fail |" in report and "| 50.00 | 37.40 | fail |" in report


def test_eval_of_gpt2_writes_baseline_without_samples(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("lm_eval.simple_evaluate", lambda **kw: fake_results(0.31, 37.4))
    main(["eval", "--model", "gpt2", "--run", "gpt2-small", "--device", "cpu"])
    run_dir = tmp_path / "results" / "gpt2-small"
    assert load_benchmarks(run_dir / "benchmarks.json") == Benchmarks(0.31, 37.4)
    assert "**Seed " not in (run_dir / "results.md").read_text()


def test_eval_help(capsys):
    with pytest.raises(SystemExit):
        main(["eval", "--help"])
    out = capsys.readouterr().out
    assert all(flag in out for flag in ["--model", "--run", "--limit", "--baseline"])
