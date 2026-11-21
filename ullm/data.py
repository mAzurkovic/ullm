from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch

SHARD_DTYPE = np.uint16


def shard_path(directory: Path, index: int) -> Path:
    return directory / f"shard_{index:05d}.bin"


def split_shards(directory: Path) -> tuple[Path, list[Path]]:
    paths = sorted(directory.glob("shard_*.bin"))
    if not paths:
        raise FileNotFoundError(f"no shards in {directory}")
    return paths[0], paths[1:]


class TokenLoader:
    def __init__(
        self,
        paths: Sequence[Path],
        batch_size: int,
        context_length: int,
        rank: int = 0,
        world_size: int = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        if not paths:
            raise ValueError("need at least one shard")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} out of range for world size {world_size}")
        self.batch_size = batch_size
        self.context_length = context_length
        self.rank = rank
        self.step_tokens = batch_size * context_length * world_size
        self.device = torch.device(device)
        self.shards = [np.memmap(path, dtype=SHARD_DTYPE, mode="r") for path in paths]
        for path, shard in zip(paths, self.shards, strict=True):
            if len(shard) < self.step_tokens + 1:
                raise ValueError(
                    f"{path} has {len(shard)} tokens, needs at least {self.step_tokens + 1}"
                )

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        block = self.batch_size * self.context_length
        while True:
            for shard in self.shards:
                for start in range(0, len(shard) - self.step_tokens, self.step_tokens):
                    first = start + self.rank * block
                    tokens = torch.from_numpy(shard[first : first + block + 1].astype(np.int64))
                    tokens = self._to_device(tokens)
                    x = tokens[:-1].view(self.batch_size, self.context_length)
                    y = tokens[1:].view(self.batch_size, self.context_length)
                    yield x, y

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.device.type == "cuda":
            return tensor.pin_memory().to(self.device, non_blocking=True)
        return tensor.to(self.device)
