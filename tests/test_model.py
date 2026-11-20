import pytest
import torch

from ullm.config import ModelConfig
from ullm.model import GPT, RMSNorm, apply_rope, rope_tables

TINY = ModelConfig(
    n_layers=2, d_model=64, n_heads=4, mlp_hidden=176, context_length=32, vocab_size=128
)


def test_default_config_matches_350m_model():
    c = ModelConfig()
    assert (c.n_layers, c.d_model, c.n_heads, c.mlp_hidden) == (24, 1024, 16, 2816)
    assert (c.context_length, c.vocab_size) == (1024, 50304)
    assert c.head_dim == 64


def test_config_rejects_d_model_not_divisible_by_heads():
    with pytest.raises(ValueError):
        ModelConfig(d_model=100, n_heads=16)


def test_config_rejects_odd_head_dim():
    with pytest.raises(ValueError):
        ModelConfig(d_model=48, n_heads=16)


def test_rmsnorm_output_has_unit_rms():
    x = torch.randn(4, 10, 64) * 7 + 3
    y = RMSNorm(64)(x)
    rms = y.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-3, rtol=0)


def test_rope_preserves_shape_and_norm():
    cos, sin = rope_tables(head_dim=16, context_length=8)
    x = torch.randn(2, 3, 8, 16)
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))


def test_rope_score_depends_only_on_relative_position():
    cos, sin = rope_tables(head_dim=16, context_length=16)
    q, k = torch.randn(16), torch.randn(16)

    def score(m: int, n: int) -> torch.Tensor:
        qs = apply_rope(q.expand(16, 16), cos, sin)[m]
        ks = apply_rope(k.expand(16, 16), cos, sin)[n]
        return qs @ ks

    torch.testing.assert_close(score(5, 2), score(12, 9))
    torch.testing.assert_close(score(3, 3), score(0, 0))


def test_logits_shape():
    model = GPT(TINY)
    idx = torch.randint(0, TINY.vocab_size, (3, 17))
    assert model(idx).shape == (3, 17, TINY.vocab_size)


def test_full_config_param_count():
    with torch.device("meta"):
        model = GPT(ModelConfig())
    n = sum(p.numel() for p in model.parameters())
    d, h, v, layers = 1024, 2816, 50304, 24
    expected = v * d + layers * (4 * d * d + 3 * d * h + 2 * d) + d
    assert n == expected
    assert 340e6 <= n <= 370e6


def test_head_is_tied_to_token_embedding():
    model = GPT(TINY)
    assert model.head.weight is model.tok_emb.weight


def test_has_no_biases():
    model = GPT(TINY)
    assert not [name for name, _ in model.named_parameters() if name.endswith("bias")]


def test_attention_is_causal():
    torch.manual_seed(0)
    model = GPT(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (1, 20))
    changed = idx.clone()
    changed[0, 10] = (idx[0, 10] + 1) % TINY.vocab_size
    with torch.no_grad():
        a, b = model(idx), model(changed)
    torch.testing.assert_close(a[:, :10], b[:, :10])
    assert not torch.allclose(a[:, 10:], b[:, 10:])


def test_rejects_sequence_longer_than_context():
    model = GPT(TINY)
    with pytest.raises(ValueError):
        model(torch.zeros(1, TINY.context_length + 1, dtype=torch.long))


def test_residual_projections_use_scaled_init():
    torch.manual_seed(0)
    model = GPT(ModelConfig(n_layers=8, d_model=256, n_heads=4, mlp_hidden=704, vocab_size=1024))
    base = 0.02
    scaled = base / (2 * 8) ** 0.5
    block = model.layers[0]
    assert block.attn.wq.weight.std().item() == pytest.approx(base, rel=0.05)
    assert block.attn.wo.weight.std().item() == pytest.approx(scaled, rel=0.05)
    assert block.mlp.w2.weight.std().item() == pytest.approx(scaled, rel=0.05)
