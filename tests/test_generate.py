import pytest
import torch

from ullm.config import ModelConfig
from ullm.generate import generate
from ullm.model import GPT
from ullm.tokenizer import N_VOCAB

CONFIG = ModelConfig(
    n_layers=2, d_model=32, n_heads=4, mlp_hidden=96, context_length=16, vocab_size=64
)


@pytest.fixture(scope="module")
def model() -> GPT:
    torch.manual_seed(0)
    return GPT(CONFIG).eval()


def run(model: GPT, ids: list[int], **kwargs) -> list[int]:
    settings = dict(max_new_tokens=10, temperature=1.0, top_k=None, stop_ids=())
    return list(generate(model, ids, **(settings | kwargs)))


def greedy_by_hand(model: GPT, ids: list[int], steps: int) -> list[int]:
    ids = list(ids)
    out = []
    for _ in range(steps):
        with torch.no_grad():
            nxt = model(torch.tensor([ids[-CONFIG.context_length :]]))[0, -1].argmax().item()
        ids.append(nxt)
        out.append(nxt)
    return out


def test_temperature_zero_is_greedy(model):
    assert run(model, [1, 2, 3], temperature=0.0) == greedy_by_hand(model, [1, 2, 3], 10)


def test_top_k_one_equals_greedy(model):
    generator = torch.Generator().manual_seed(1)
    got = run(model, [1, 2, 3], top_k=1, generator=generator)
    assert got == greedy_by_hand(model, [1, 2, 3], 10)


def test_same_seed_gives_same_sample(model):
    a = run(model, [5], generator=torch.Generator().manual_seed(7))
    b = run(model, [5], generator=torch.Generator().manual_seed(7))
    assert a == b


def test_top_k_limits_choices_to_most_likely_tokens(model):
    with torch.no_grad():
        top3 = set(model(torch.tensor([[4]]))[0, -1].topk(3).indices.tolist())
    for seed in range(30):
        g = torch.Generator().manual_seed(seed)
        assert run(model, [4], max_new_tokens=1, top_k=3, generator=g)[0] in top3


def test_top_k_larger_than_vocab_is_allowed(model):
    assert len(run(model, [4], top_k=10_000)) == 10


def test_stops_at_max_new_tokens(model):
    assert len(run(model, [1], max_new_tokens=4)) == 4


def test_stops_on_stop_token_without_yielding_it(model):
    greedy = greedy_by_hand(model, [1, 2, 3], 10)
    stop = greedy[-1]
    got = run(model, [1, 2, 3], temperature=0.0, stop_ids={stop})
    assert got == greedy[: greedy.index(stop)]


def test_prompt_longer_than_context_uses_last_tokens(model):
    long_prompt = list(range(40))
    got = run(model, long_prompt, temperature=0.0, max_new_tokens=3)
    assert got == greedy_by_hand(model, long_prompt[-CONFIG.context_length :], 3)


def test_does_not_modify_prompt(model):
    prompt = [1, 2, 3]
    run(model, prompt)
    assert prompt == [1, 2, 3]


def test_rejects_empty_prompt(model):
    with pytest.raises(ValueError):
        run(model, [])


def test_never_samples_padding_ids():
    torch.manual_seed(0)
    model = GPT(ModelConfig(n_layers=1, d_model=32, n_heads=4, mlp_hidden=96, vocab_size=50304))
    with torch.no_grad():
        logits = model(torch.tensor([[1, 2, 3]]))[0, -1]
        best = int(logits.argmax())
        assert logits[best] > 0
        model.tok_emb.weight[N_VOCAB] = model.tok_emb.weight[best] * 10
        assert int(model(torch.tensor([[1, 2, 3]]))[0, -1].argmax()) == N_VOCAB
    assert run(model, [1, 2, 3], temperature=0.0, max_new_tokens=1) == [best]
    for seed in range(20):
        g = torch.Generator().manual_seed(seed)
        assert all(t < N_VOCAB for t in run(model, [1, 2, 3], generator=g, max_new_tokens=5))
