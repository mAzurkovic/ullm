# M2 F2. Loader — Tasks

Scope: `ullm/data.py` shard layout and pre-train loader. Source: `design.md` (Data Model, Pre-training Pipeline step 5). Tested on synthetic shards. No download or tokenizing (M2 F1), no resume (M3 F1).

Shard layout: `<dir>/shard_00000.bin`, `shard_00001.bin`, ... Raw `uint16`, no header. Shard 0 = validation.

## 1. One shard streams packed input and target batches [serial]
- [x] 1.1 `shard_path(directory, index)` and `split_shards(directory) -> (val, train)` in `ullm/data.py`. Raises `FileNotFoundError` when no shards exist. Tests pass.
- [x] 1.2 `TokenLoader(paths, batch_size, context_length, rank=0, world_size=1, device="cpu")`: memmap each shard, yield `(x, y)` int64 tensors of shape `(batch_size, context_length)`, built from `batch_size * context_length + 1` consecutive tokens. Test: on an `arange` shard, `y == x + 1`.
- [x] 1.3 Move batches to `device`, pinned and non-blocking when it is CUDA. Test on CPU.

## 2. Ranks and shards split the data without overlap [serial, needs 1]
- [x] 2.1 Rank `r` of `world_size` reads block `r` of each step; the position advances by `world_size * batch_size * context_length`. Test: 4 ranks see disjoint, adjacent token ranges.
- [x] 2.2 Move to the next shard when the current one cannot fill a whole step for all ranks; drop its tail. Test: 2 shards, batches never mix shards, tail skipped.
- [x] 2.3 After the last shard, start again at the first. Test: iteration continues past one pass.
- [x] 2.4 Reject a shard too small for one step with `ValueError`. Test passes.

## Dependencies
- 2 needs 1.
