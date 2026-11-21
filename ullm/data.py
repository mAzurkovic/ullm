import shutil
import time
from collections.abc import Iterable, Iterator, Sequence
from itertools import batched
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from ullm.tokenizer import ENDOFTEXT, encode

SHARD_DTYPE = np.uint16
SHARD_TOKENS = 100_000_000
TOKENIZE_BATCH = 8192
SHARD_PATTERN = "shard_*.bin"
FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_FILES = "sample/10BT/"

DATASET_CARD = """---
license: odc-by
language:
- en
---

# ullm FineWeb-Edu shards

Pre-training data for [ullm](https://github.com/mAzurkovic/ullm): the
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) `sample-10BT`
subset, tokenized with the GPT-2 tokenizer. FineWeb-Edu is derived from Common Crawl and
released under ODC-By 1.0; this derived copy keeps that license and credits its source.

## Layout

- `shard_00000.bin`, `shard_00001.bin`, ...: raw `uint16` token ids, no header,
  {shard_tokens:,} tokens per shard (the last shard may be shorter).
- Documents are in dataset order, each followed by `<|endoftext|>` (id 50256).
- `shard_00000.bin` is the validation split; the rest are training data.
- {num_shards} shards, {total_tokens:,} tokens in total.
"""


def shard_path(directory: Path, index: int) -> Path:
    return directory / f"shard_{index:05d}.bin"


def split_shards(directory: Path) -> tuple[Path, list[Path]]:
    paths = sorted(directory.glob(SHARD_PATTERN))
    if not paths:
        raise FileNotFoundError(f"no shards in {directory}")
    return paths[0], paths[1:]


def write_shards(
    texts: Iterable[str], directory: Path, shard_tokens: int, num_shards: int, workers: int
) -> int:
    if shard_tokens < 1 or num_shards < 1:
        raise ValueError("shard_tokens and num_shards must be at least 1")
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob(SHARD_PATTERN):
        stale.unlink()

    buffer = np.empty(shard_tokens, dtype=SHARD_DTYPE)
    filled = written = 0
    start = time.time()

    def flush(n_tokens: int) -> None:
        nonlocal written
        buffer[:n_tokens].tofile(shard_path(directory, written))
        written += 1
        tokens = (written - 1) * shard_tokens + n_tokens
        print(f"{written} shard(s) | {tokens:,} tokens | {time.time() - start:.0f}s", flush=True)

    with Pool(workers) as pool:
        for batch in batched(texts, TOKENIZE_BATCH):
            for ids in pool.map(_tokenize_document, batch, chunksize=64):
                while len(ids):
                    take = min(len(ids), shard_tokens - filled)
                    buffer[filled : filled + take] = ids[:take]
                    filled += take
                    ids = ids[take:]
                    if filled == shard_tokens:
                        flush(filled)
                        filled = 0
                        if written == num_shards:
                            return written
    if filled:
        flush(filled)
    return written


def _tokenize_document(text: str) -> np.ndarray:
    return np.array([*encode(text), ENDOFTEXT], dtype=SHARD_DTYPE)


def push_shards(directory: Path, repo_id: str) -> None:
    shards = sorted(directory.glob(SHARD_PATTERN))
    card = DATASET_CARD.format(
        shard_tokens=SHARD_TOKENS,
        num_shards=len(shards),
        total_tokens=sum(p.stat().st_size for p in shards) // SHARD_DTYPE().itemsize,
    )
    (directory / "README.md").write_text(card)
    HfApi().upload_folder(
        repo_id=repo_id,
        folder_path=directory,
        repo_type="dataset",
        allow_patterns=[SHARD_PATTERN, "README.md"],
    )


def download_shards(repo_id: str, directory: Path) -> Path:
    return Path(
        snapshot_download(
            repo_id, repo_type="dataset", allow_patterns=SHARD_PATTERN, local_dir=directory
        )
    )


def prepare_pretrain(
    directory: Path,
    num_shards: int,
    workers: int,
    push: str | None,
    shard_tokens: int = SHARD_TOKENS,
) -> None:
    raw = directory / "raw"
    try:
        write_shards(_fineweb_texts(raw), directory, shard_tokens, num_shards, workers)
    finally:
        shutil.rmtree(raw, ignore_errors=True)
    if push:
        push_shards(directory, push)


def _fineweb_texts(raw: Path) -> Iterator[str]:
    files = HfApi().list_repo_files(FINEWEB_REPO, repo_type="dataset")
    for name in sorted(f for f in files if f.startswith(FINEWEB_FILES) and f.endswith(".parquet")):
        path = Path(hf_hub_download(FINEWEB_REPO, name, repo_type="dataset", local_dir=raw))
        for batch in pq.ParquetFile(path).iter_batches(columns=["text"]):
            yield from batch.column("text").to_pylist()
        path.unlink()


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
