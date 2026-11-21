from itertools import islice
from pathlib import Path

import numpy as np
import pytest
import torch

from ullm.data import TokenLoader, shard_path, split_shards


def write_shard(directory: Path, index: int, n_tokens: int, first: int = 0) -> Path:
    path = shard_path(directory, index)
    np.arange(first, first + n_tokens, dtype=np.uint16).tofile(path)
    return path


def test_shard_path_layout(tmp_path):
    assert shard_path(tmp_path, 0) == tmp_path / "shard_00000.bin"
    assert shard_path(tmp_path, 123) == tmp_path / "shard_00123.bin"


def test_split_shards_holds_out_first_shard(tmp_path):
    paths = [write_shard(tmp_path, i, 10) for i in (2, 0, 1)]
    val, train = split_shards(tmp_path)
    assert val == paths[1]
    assert train == [paths[2], paths[0]]


def test_split_shards_without_shards_raises(tmp_path):
    (tmp_path / "notes.txt").write_text("not a shard")
    with pytest.raises(FileNotFoundError):
        split_shards(tmp_path)


def test_batches_are_inputs_and_next_token_targets(tmp_path):
    path = write_shard(tmp_path, 0, 100)
    x, y = next(iter(TokenLoader([path], batch_size=2, context_length=4)))
    assert x.dtype == y.dtype == torch.int64
    assert x.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert y.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert x.device.type == "cpu"


def test_large_token_ids_are_read_unsigned(tmp_path):
    path = write_shard(tmp_path, 0, 20, first=50250)
    x, _ = next(iter(TokenLoader([path], batch_size=1, context_length=4)))
    assert x.tolist() == [[50250, 50251, 50252, 50253]]


def test_ranks_read_disjoint_adjacent_blocks(tmp_path):
    path = write_shard(tmp_path, 0, 1000)
    batch, context, world = 2, 3, 4
    block = batch * context
    for step in range(2):
        starts = []
        for rank in range(world):
            loader = TokenLoader([path], batch, context, rank=rank, world_size=world)
            x, y = next(islice(iter(loader), step, None))
            first = (step * world + rank) * block
            assert x.flatten().tolist() == list(range(first, first + block))
            assert y.flatten().tolist() == list(range(first + 1, first + block + 1))
            starts.append(first)
        assert starts == [step * world * block + r * block for r in range(world)]


def test_moves_to_next_shard_and_skips_tail(tmp_path):
    # 2 ranks x 2 rows x 3 tokens = 12 tokens per step. A 30-token shard fits
    # 2 steps (the second reads up to token 24 inclusive); tokens 25 to 29 are skipped.
    shards = [write_shard(tmp_path, 0, 30), write_shard(tmp_path, 1, 30, first=1000)]
    loader = TokenLoader(shards, batch_size=2, context_length=3, rank=1, world_size=2)
    firsts = [int(x[0, 0]) for x, _ in islice(iter(loader), 4)]
    assert firsts == [6, 18, 1006, 1018]


def test_starts_again_after_last_shard(tmp_path):
    path = write_shard(tmp_path, 0, 25)
    loader = TokenLoader([path], batch_size=2, context_length=3, world_size=2)
    batches = list(islice(iter(loader), 3))
    assert [int(x[0, 0]) for x, _ in batches] == [0, 12, 0]
    assert torch.equal(batches[0][1], batches[2][1])


def test_rejects_shard_smaller_than_one_step(tmp_path):
    good = write_shard(tmp_path, 0, 100)
    small = write_shard(tmp_path, 1, 12)
    with pytest.raises(ValueError, match="shard_00001"):
        TokenLoader([good, small], batch_size=2, context_length=3, world_size=2)


def test_rejects_no_shards():
    with pytest.raises(ValueError):
        TokenLoader([], batch_size=2, context_length=3)


@pytest.mark.parametrize("rank, world_size", [(-1, 2), (2, 2), (0, 0)])
def test_rejects_invalid_rank(tmp_path, rank, world_size):
    path = write_shard(tmp_path, 0, 100)
    with pytest.raises(ValueError):
        TokenLoader([path], batch_size=2, context_length=3, rank=rank, world_size=world_size)
