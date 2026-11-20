import math

import torch
import torch.nn.functional as F
from torch import nn

from ullm.config import ModelConfig

ROPE_BASE = 10_000.0
NORM_EPS = 1e-5
INIT_STD = 0.02


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + NORM_EPS) * self.weight


def rope_tables(head_dim: int, context_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = ROPE_BASE ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    angles = torch.outer(torch.arange(context_length, dtype=torch.float32), freqs)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    t = x.shape[-2]
    cos, sin = cos[:t], sin[:t]
    x1, x2 = x.float().chunk(2, dim=-1)
    rotated = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rotated.type_as(x)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model
        self.n_heads = config.n_heads
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape

        def heads(proj: nn.Linear) -> torch.Tensor:
            return proj(x).view(b, t, self.n_heads, -1).transpose(1, 2)

        q = apply_rope(heads(self.wq), cos, sin)
        k = apply_rope(heads(self.wk), cos, sin)
        y = F.scaled_dot_product_attention(q, k, heads(self.wv), is_causal=True)
        return self.wo(y.transpose(1, 2).reshape(b, t, d))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.mlp_hidden, bias=False)
        self.w3 = nn.Linear(config.d_model, config.mlp_hidden, bias=False)
        self.w2 = nn.Linear(config.mlp_hidden, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        cos, sin = rope_tables(config.head_dim, config.context_length)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        # Residual projections are scaled down so the residual stream variance
        # does not grow with depth.
        residual_std = INIT_STD / math.sqrt(2 * self.config.n_layers)
        for name, p in self.named_parameters():
            if name.endswith(("attn.wo.weight", "mlp.w2.weight")):
                nn.init.normal_(p, std=residual_std)
            elif p.dim() == 2:
                nn.init.normal_(p, std=INIT_STD)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        if t > self.config.context_length:
            raise ValueError(f"sequence length {t} exceeds context {self.config.context_length}")
        x = self.tok_emb(idx)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        return self.head(self.norm(x))
