# M2 F1. Prepare — Tasks

Scope: `ullm prepare` on the Mac: download FineWeb-Edu `sample-10BT` one parquet file at a time, tokenize on all cores, write 71 shards (shard 0 = val, 70 × 100M train = 7B tokens), push to `mattiasz/ullm-fineweb-edu-shards`. Plus `download_shards()` for GPU nodes. Source: `design.md` (Data Model, Pre-training Pipeline steps 1 to 4, D2, D8). Reuses `shard_path()`, `SHARD_DTYPE` (`ullm/data.py`), `encode()`, `ENDOFTEXT` (`ullm/tokenizer.py`), `cli.py`. Unit tests use small texts and mock only the HF network calls.

## 1. Texts become shards [serial]
- [x] 1.1 Add `huggingface_hub` and `pyarrow` to `pyproject.toml`. `uv sync` succeeds.
- [x] 1.2 `write_shards(texts, directory, shard_tokens, num_shards, workers) -> int` in `ullm/data.py`: tokenize in a process pool (order kept), append `ENDOFTEXT` after each document, write full `uint16` shards via `shard_path()`. Test: decoded shards equal the texts joined by `<|endoftext|>`.
- [x] 1.3 Stop after `num_shards` full shards; when texts run out first, write the partial last shard. Tests: cap respected, partial shard kept, returns shard count.
- [x] 1.4 Result is the same with 1 and with several workers. Test passes.

## 2. Shards go to and come from the HF Hub [parallel, needs 1.1]
- [x] 2.1 `push_shards(directory, repo_id)`: write `README.md` dataset card (source FineWeb-Edu `sample-10BT`, ODC-By, derived from Common Crawl; layout `shard_NNNNN.bin` raw `uint16`, GPT-2 tokens, shard 0 = val), then `HfApi().upload_folder(repo_type="dataset")`. Test with the upload call mocked: card written, right repo and folder.
- [x] 2.2 `download_shards(repo_id, directory) -> Path`: `snapshot_download(repo_type="dataset", allow_patterns="shard_*.bin")`. Test with the call mocked: right repo, patterns, local dir.

## 3. `ullm prepare` runs end to end [serial, needs 1, 2]
- [x] 3.1 `prepare` subcommand in `ullm/cli.py`: `--out` (default `data/shards`), `--num-shards` (default 71), `--workers` (default all cores), `--push REPO` (optional). Downloads `sample/10BT/*.parquet` in order with `hf_hub_download`, reads `text` with `pyarrow`, deletes each file after use, and stops downloading once the shards are full. `ullm prepare --help` works.
- [x] 3.2 Print progress: shards written, tokens, elapsed. Test: `prepare` with the HF download mocked to small parquet files writes shards the `TokenLoader` reads.

## 4. Real shards on the Hub [serial, needs 3]
- [ ] 4.1 User runs `ullm prepare --push mattiasz/ullm-fineweb-edu-shards` in their own terminal on the Mac. 71 shards uploaded, dataset card visible.
- [ ] 4.2 `download_shards()` into a fresh temp dir; `split_shards()` and `TokenLoader` read it. Delete local raw data and shards after.

## Dependencies
- 2 needs 1.1.
- 3 needs 1 and 2.
- 4 needs 3.
