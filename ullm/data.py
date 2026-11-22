import shutil
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from functools import partial
from itertools import batched, islice
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from ullm.tokenizer import ENDOFTEXT, Message, decode, encode, render_chat

SHARD_DTYPE = np.uint16
SHARD_TOKENS = 100_000_000
TOKENIZE_BATCH = 8192
SHARD_PATTERN = "shard_*.bin"
FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_FILES = "sample/10BT/"
SFT_REPO = "HuggingFaceTB/smol-smoltalk"
SFT_SPLITS = {"train": "data/train-", "val": "data/test-"}
SFT_ROW_TOKENS = 1024 + 1
IGNORE_INDEX = -100

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


class SFTExample(NamedTuple):
    ids: list[int]
    mask: list[int]
    truncated: bool


def sft_example(
    messages: Sequence[Mapping[str, str]], max_tokens: int = SFT_ROW_TOKENS
) -> SFTExample | None:
    roles = [m["role"] for m in messages]
    if len(roles) < 2 or roles != [("user", "assistant")[i % 2] for i in range(len(roles))]:
        return None
    ids: list[int] = []
    mask: list[int] = []
    kept = 0
    for user, assistant in zip(messages[::2], messages[1::2], strict=False):
        pair_ids, pair_mask = render_chat(
            [Message("user", user["content"]), Message("assistant", assistant["content"])]
        )
        if len(ids) + len(pair_ids) > max_tokens:
            break
        ids += pair_ids
        mask += pair_mask
        kept += 1
    if not kept:
        return None
    return SFTExample(ids, mask, truncated=2 * kept < len(messages))


def prepare_sft(
    directory: Path, workers: int, limit: int | None = None, row_tokens: int = SFT_ROW_TOKENS
) -> None:
    raw = directory / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    files = HfApi().list_repo_files(SFT_REPO, repo_type="dataset")
    try:
        with Pool(workers) as pool:
            for split, prefix in SFT_SPLITS.items():
                names = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
                conversations = islice(_smoltalk_conversations(names, raw), limit)
                examples = pool.imap(
                    partial(sft_example, max_tokens=row_tokens), conversations, chunksize=256
                )
                ids, mask = _write_sft_split(directory, split, examples, row_tokens)
                if split == "train" and len(ids):
                    print(_show_mask(ids[0], mask[0]), flush=True)
    finally:
        shutil.rmtree(raw, ignore_errors=True)


def _smoltalk_conversations(names: Sequence[str], raw: Path) -> Iterator[list[dict[str, str]]]:
    for name in names:
        path = Path(hf_hub_download(SFT_REPO, name, repo_type="dataset", local_dir=raw))
        for batch in pq.ParquetFile(path).iter_batches(columns=["messages"]):
            yield from batch.column("messages").to_pylist()
        path.unlink()


def _write_sft_split(
    directory: Path, split: str, examples: Iterable[SFTExample | None], row_tokens: int
) -> tuple[np.ndarray, np.ndarray]:
    id_rows, mask_rows = [], []
    skipped = truncated = assistant_tokens = 0
    for example in examples:
        if example is None:
            skipped += 1
            continue
        pad = row_tokens - len(example.ids)
        id_rows.append(np.array(example.ids + [ENDOFTEXT] * pad, dtype=SHARD_DTYPE))
        mask_rows.append(np.array(example.mask + [0] * pad, dtype=np.uint8))
        truncated += example.truncated
        assistant_tokens += sum(example.mask)
    ids = np.stack(id_rows) if id_rows else np.empty((0, row_tokens), dtype=SHARD_DTYPE)
    mask = np.stack(mask_rows) if mask_rows else np.empty((0, row_tokens), dtype=np.uint8)
    np.save(directory / f"{split}_ids.npy", ids)
    np.save(directory / f"{split}_mask.npy", mask)
    print(
        f"{split}: kept {len(ids):,} | skipped {skipped:,} | truncated {truncated:,} | "
        f"assistant tokens {assistant_tokens:,}",
        flush=True,
    )
    return ids, mask


def _show_mask(ids: np.ndarray, mask: np.ndarray) -> str:
    """Decode a row, wrapping the tokens the model learns from in [[ ]]."""
    end = len(ids)
    while end and ids[end - 1] == ENDOFTEXT and not mask[end - 1]:
        end -= 1
    pieces, start = [], 0
    for i in range(1, end + 1):
        if i == end or mask[i] != mask[start]:
            text = decode(ids[start:i].tolist())
            pieces.append(f"[[{text}]]" if mask[start] else text)
            start = i
    return "".join(pieces)


class TokenLoader:
    def __init__(
        self,
        paths: Sequence[Path],
        batch_size: int,
        context_length: int,
        rank: int = 0,
        world_size: int = 1,
        device: str | torch.device = "cpu",
        start_batch: int = 0,
    ) -> None:
        if not paths:
            raise ValueError("need at least one shard")
        if start_batch < 0:
            raise ValueError(f"start_batch {start_batch} must not be negative")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} out of range for world size {world_size}")
        self.batch_size = batch_size
        self.context_length = context_length
        self.rank = rank
        self.start_batch = start_batch
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
        counts = [len(range(0, len(s) - self.step_tokens, self.step_tokens)) for s in self.shards]
        skip = self.start_batch % sum(counts)
        while True:
            for shard, count in zip(self.shards, counts, strict=True):
                if skip >= count:
                    skip -= count
                    continue
                begin, skip = skip * self.step_tokens, 0
                for start in range(begin, len(shard) - self.step_tokens, self.step_tokens):
                    first = start + self.rank * block
                    tokens = torch.from_numpy(shard[first : first + block + 1].astype(np.int64))
                    tokens = _to_device(tokens, self.device)
                    x = tokens[:-1].view(self.batch_size, self.context_length)
                    y = tokens[1:].view(self.batch_size, self.context_length)
                    yield x, y


class SFTLoader:
    """Shuffled micro-batches of fine-tuning rows; targets the model must not learn are ignored."""

    def __init__(
        self,
        directory: Path,
        split: str,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} out of range for world size {world_size}")
        self.ids = np.load(directory / f"{split}_ids.npy", mmap_mode="r")
        self.mask = np.load(directory / f"{split}_mask.npy", mmap_mode="r")
        if self.ids.ndim != 2 or self.ids.shape != self.mask.shape:
            raise ValueError(f"{split} ids {self.ids.shape} and mask {self.mask.shape} differ")
        self.rows, self.row_tokens = self.ids.shape
        if self.rows < batch_size * world_size:
            raise ValueError(
                f"{split} has {self.rows} rows, needs at least {batch_size * world_size}"
            )
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)
        self.seed = seed

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        slab = self.batch_size * self.world_size
        rng = np.random.default_rng(self.seed)
        while True:
            order = rng.permutation(self.rows)
            for start in range(0, self.rows - slab + 1, slab):
                first = start + self.rank * self.batch_size
                picked = np.sort(order[first : first + self.batch_size])
                tokens = torch.from_numpy(self.ids[picked].astype(np.int64))
                learn = torch.from_numpy(self.mask[picked, 1:].astype(bool))
                x = tokens[:, :-1].contiguous()
                y = tokens[:, 1:].masked_fill(~learn, IGNORE_INDEX)
                yield _to_device(x, self.device), _to_device(y, self.device)


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)
