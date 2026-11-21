from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ullm.cli import main
from ullm.data import (
    TokenLoader,
    download_shards,
    prepare_pretrain,
    push_shards,
    shard_path,
    split_shards,
    write_shards,
)
from ullm.tokenizer import ENDOFTEXT, decode, encode

TEXTS = [f"Document number {i} talks about topic {i % 7}." for i in range(40)]


def all_tokens(directory: Path) -> list[int]:
    paths = sorted(directory.glob("shard_*.bin"))
    return np.concatenate([np.fromfile(p, dtype=np.uint16) for p in paths]).tolist()


def expected_tokens(texts: list[str]) -> list[int]:
    return [t for text in texts for t in [*encode(text), ENDOFTEXT]]


def test_shards_hold_documents_separated_by_endoftext(tmp_path):
    count = write_shards(TEXTS, tmp_path, shard_tokens=50, num_shards=1000, workers=2)
    tokens = all_tokens(tmp_path)
    assert tokens == expected_tokens(TEXTS)
    assert decode(tokens) == "<|endoftext|>".join(TEXTS) + "<|endoftext|>"
    assert count == -(-len(tokens) // 50)


def test_full_shards_have_exact_size_and_partial_last_shard_is_kept(tmp_path):
    total = len(expected_tokens(TEXTS))
    count = write_shards(TEXTS, tmp_path, shard_tokens=64, num_shards=1000, workers=2)
    sizes = [shard_path(tmp_path, i).stat().st_size // 2 for i in range(count)]
    assert sizes[:-1] == [64] * (count - 1)
    assert sizes[-1] == total - 64 * (count - 1)
    assert 0 < sizes[-1] < 64


def test_stops_after_num_shards(tmp_path):
    count = write_shards(TEXTS, tmp_path, shard_tokens=30, num_shards=3, workers=2)
    assert count == 3
    assert sorted(p.name for p in tmp_path.glob("shard_*.bin")) == [
        "shard_00000.bin",
        "shard_00001.bin",
        "shard_00002.bin",
    ]
    assert all_tokens(tmp_path) == expected_tokens(TEXTS)[:90]


def test_same_result_with_one_and_many_workers(tmp_path):
    one, many = tmp_path / "one", tmp_path / "many"
    write_shards(TEXTS, one, shard_tokens=40, num_shards=1000, workers=1)
    write_shards(TEXTS, many, shard_tokens=40, num_shards=1000, workers=4)
    assert all_tokens(one) == all_tokens(many)


def test_replaces_shards_from_an_earlier_run(tmp_path):
    write_shards(TEXTS, tmp_path, shard_tokens=20, num_shards=1000, workers=1)
    count = write_shards(TEXTS[:2], tmp_path, shard_tokens=20, num_shards=1000, workers=1)
    assert len(list(tmp_path.glob("shard_*.bin"))) == count
    assert all_tokens(tmp_path) == expected_tokens(TEXTS[:2])


@pytest.mark.parametrize("shard_tokens, num_shards", [(0, 1), (10, 0)])
def test_rejects_invalid_sizes(tmp_path, shard_tokens, num_shards):
    with pytest.raises(ValueError):
        write_shards(TEXTS, tmp_path, shard_tokens=shard_tokens, num_shards=num_shards, workers=1)


def test_push_writes_dataset_card_and_uploads_folder(tmp_path, monkeypatch):
    write_shards(TEXTS, tmp_path, shard_tokens=100, num_shards=1000, workers=1)
    calls = []
    monkeypatch.setattr("huggingface_hub.HfApi.upload_folder", lambda self, **kw: calls.append(kw))

    push_shards(tmp_path, "someone/shards")

    card = (tmp_path / "README.md").read_text()
    assert "license: odc-by" in card
    assert "HuggingFaceFW/fineweb-edu" in card
    assert "Common Crawl" in card
    assert "shard_00000.bin" in card
    assert calls == [
        {
            "repo_id": "someone/shards",
            "folder_path": tmp_path,
            "repo_type": "dataset",
            "allow_patterns": ["shard_*.bin", "README.md"],
        }
    ]


def test_download_fetches_only_shards(tmp_path, monkeypatch):
    calls = []

    def fake_snapshot_download(repo_id, **kw):
        calls.append((repo_id, kw))
        return str(kw["local_dir"])

    monkeypatch.setattr("ullm.data.snapshot_download", fake_snapshot_download)
    assert download_shards("someone/shards", tmp_path) == tmp_path
    assert calls == [
        (
            "someone/shards",
            {"repo_type": "dataset", "allow_patterns": "shard_*.bin", "local_dir": tmp_path},
        )
    ]


def fake_fineweb(monkeypatch, files: dict[str, list[str]]) -> list[str]:
    downloaded = []

    def list_repo_files(self, repo_id, repo_type):
        assert (repo_id, repo_type) == ("HuggingFaceFW/fineweb-edu", "dataset")
        return ["README.md", "sample/100BT/000_00000.parquet", *files]

    def hf_hub_download(repo_id, filename, repo_type, local_dir):
        downloaded.append(filename)
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {"text": files[filename], "id": [str(i) for i in range(len(files[filename]))]}
        )
        pq.write_table(table, path, row_group_size=7)
        return str(path)

    monkeypatch.setattr("huggingface_hub.HfApi.list_repo_files", list_repo_files)
    monkeypatch.setattr("ullm.data.hf_hub_download", hf_hub_download)
    return downloaded


def test_prepare_command_writes_shards_the_loader_reads(tmp_path, monkeypatch, capsys):
    files = {
        "sample/10BT/001_00000.parquet": TEXTS[20:],
        "sample/10BT/000_00000.parquet": TEXTS[:20],
    }
    downloaded = fake_fineweb(monkeypatch, files)
    out = tmp_path / "shards"
    main(["prepare", "--out", str(out), "--workers", "2"])

    assert downloaded == ["sample/10BT/000_00000.parquet", "sample/10BT/001_00000.parquet"]
    assert all_tokens(out) == expected_tokens(TEXTS)
    assert not (out / "raw").exists()
    val, train = split_shards(out)
    assert train == []
    x, _ = next(iter(TokenLoader([val], batch_size=2, context_length=8)))
    assert x.flatten().tolist() == expected_tokens(TEXTS)[:16]
    assert "1 shard" in capsys.readouterr().out


def test_prepare_stops_downloading_once_shards_are_full(tmp_path, monkeypatch):
    files = {f"sample/10BT/{i:03d}_00000.parquet": TEXTS for i in range(3)}
    downloaded = fake_fineweb(monkeypatch, files)
    monkeypatch.setattr("ullm.data.TOKENIZE_BATCH", 10)
    prepare_pretrain(tmp_path, num_shards=1, workers=1, push=None, shard_tokens=50)
    assert downloaded == ["sample/10BT/000_00000.parquet"]
    assert all_tokens(tmp_path) == expected_tokens(TEXTS)[:50]
    assert not (tmp_path / "raw").exists()


def test_prepare_command_pushes_when_asked(tmp_path, monkeypatch):
    fake_fineweb(monkeypatch, {"sample/10BT/000_00000.parquet": ["hello"]})
    pushed = []
    monkeypatch.setattr("huggingface_hub.HfApi.upload_folder", lambda self, **kw: pushed.append(kw))
    main(["prepare", "--out", str(tmp_path), "--workers", "1", "--push", "someone/shards"])
    assert [call["repo_id"] for call in pushed] == ["someone/shards"]


def test_prepare_help(capsys):
    with pytest.raises(SystemExit):
        main(["prepare", "--help"])
    out = capsys.readouterr().out
    assert all(flag in out for flag in ["--out", "--num-shards", "--workers", "--push"])
