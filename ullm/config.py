from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    n_layers: int = 24
    d_model: int = 1024
    n_heads: int = 16
    mlp_hidden: int = 2816
    context_length: int = 1024
    vocab_size: int = 50304

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model {self.d_model} not divisible by n_heads {self.n_heads}")
        if self.head_dim % 2:
            raise ValueError(f"head_dim {self.head_dim} must be even for RoPE")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
