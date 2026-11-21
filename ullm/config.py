import argparse
from dataclasses import MISSING, dataclass, fields
from pathlib import Path


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


@dataclass(frozen=True)
class PretrainConfig:
    run: str = "pretrain"
    data: Path = Path("data/shards")
    results_dir: Path = Path("results")
    checkpoints_dir: Path = Path("checkpoints")
    batch_tokens: int = 512 * 1024
    micro_batch: int = 16
    max_tokens: int = 7_000_000_000
    peak_lr: float = 3e-4
    warmup_steps: int = 700
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    log_every: int = 10
    val_every: int = 250
    sample_every: int = 500
    checkpoint_every: int = 1000
    val_batches: int = 20
    seed: int = 0
    device: str = "auto"
    resume: bool = False

    @property
    def total_steps(self) -> int:
        return self.max_tokens // self.batch_tokens

    def grad_accum_steps(self, context_length: int) -> int:
        micro_tokens = self.micro_batch * context_length
        if micro_tokens < 1 or self.batch_tokens % micro_tokens:
            raise ValueError(
                f"batch of {self.batch_tokens} tokens is not a multiple of "
                f"micro-batch {self.micro_batch} x context {context_length}"
            )
        return self.batch_tokens // micro_tokens


def add_config_args(parser: argparse.ArgumentParser, cls: type) -> None:
    for field in fields(cls):
        flag = "--" + field.name.replace("_", "-")
        default = field.default if field.default is not MISSING else None
        if field.type is bool:
            parser.add_argument(flag, action="store_true", default=default)
        elif field.type is int:
            parser.add_argument(flag, type=_integer, default=default)
        else:
            parser.add_argument(flag, type=field.type, default=default)


def config_from_args[T](cls: type[T], args: argparse.Namespace) -> T:
    return cls(**{field.name: getattr(args, field.name) for field in fields(cls)})


def _integer(text: str) -> int:
    value = float(text)
    if not value.is_integer():
        raise argparse.ArgumentTypeError(f"{text} is not a whole number")
    return int(value)
