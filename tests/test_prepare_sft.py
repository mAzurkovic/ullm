from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ullm.cli import main
from ullm.data import SFT_ROW_TOKENS, prepare_sft, sft_example
from ullm.tokenizer import ASSISTANT, END, ENDOFTEXT, USER, encode


def chat(*turns: str, system: str | None = None) -> list[dict[str, str]]:
    roles = ["user", "assistant"]
    messages = [{"role": roles[i % 2], "content": text} for i, text in enumerate(turns)]
    return ([{"role": "system", "content": system}] if system else []) + messages


def rendered(*turns: str) -> list[int]:
    ids = []
    for i, text in enumerate(turns):
        ids += [USER if i % 2 == 0 else ASSISTANT, *encode(text), END]
    return ids


def rendered_mask(*turns: str) -> list[int]:
    mask = []
    for i, text in enumerate(turns):
        mask += [0] + [i % 2] * (len(encode(text)) + 1)
    return mask


def test_row_holds_inputs_and_one_shifted_target():
    assert SFT_ROW_TOKENS == 1025


def test_short_conversation_is_kept_whole_with_assistant_mask():
    example = sft_example(chat("Hi there", "Hello!", "Bye", "See you"))
    assert example.ids == rendered("Hi there", "Hello!", "Bye", "See you")
    hello, see_you = len(encode("Hello!")), len(encode("See you"))
    user_1 = [0] * (2 + len(encode("Hi there")))
    user_2 = [0] * (2 + len(encode("Bye")))
    assert example.mask == user_1 + [0] + [1] * (hello + 1) + user_2 + [0] + [1] * (see_you + 1)
    assert not example.truncated


def test_long_conversation_is_cut_after_the_last_exchange_that_fits():
    first, second = ("one two", "three four"), ("five six seven", "eight nine ten")
    head = rendered(*first)
    example = sft_example(chat(*first, *second), max_tokens=len(head) + 3)
    assert example.ids == head
    assert example.truncated


def test_cut_drops_a_final_user_turn_without_reply():
    example = sft_example(chat("Hi", "Hello", "Anyone there?"))
    assert example.ids == rendered("Hi", "Hello")
    assert example.truncated


def test_conversation_fits_exactly():
    turns = ("one two", "three four")
    example = sft_example(chat(*turns), max_tokens=len(rendered(*turns)))
    assert example.ids == rendered(*turns)
    assert not example.truncated


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param(chat("word " * 50, "reply"), id="first exchange too long"),
        pytest.param(chat("Hi", "Hello", system="Be brief."), id="system message"),
        pytest.param(chat("Hi"), id="no reply"),
        pytest.param([], id="empty"),
        pytest.param(
            [{"role": "assistant", "content": "Hi"}, {"role": "user", "content": "Yo"}],
            id="starts with assistant",
        ),
        pytest.param(
            [{"role": "user", "content": "Hi"}, {"role": "user", "content": "Yo"}],
            id="roles do not alternate",
        ),
    ],
)
def test_skips_unusable_conversations(messages):
    assert sft_example(messages, max_tokens=40) is None


CONVERSATIONS = {
    "data/train-00000-of-00002.parquet": [
        chat("What is 2 plus 2?", "It is 4."),
        chat("Hi", "Hello", system="Be brief."),
        chat("Name a color.", "Blue.", "Another?", "Red."),
    ],
    "data/train-00001-of-00002.parquet": [
        chat("Say yes.", "Yes."),
        chat("word " * 60, "too long"),
    ],
    "data/test-00000-of-00001.parquet": [
        chat("Say no.", "No."),
        chat("Count to three.", "One, two, three."),
    ],
}


def fake_smoltalk(monkeypatch, files: dict[str, list[list[dict[str, str]]]]) -> list[str]:
    downloaded = []

    def list_repo_files(self, repo_id, repo_type):
        assert (repo_id, repo_type) == ("HuggingFaceTB/smol-smoltalk", "dataset")
        return [".gitattributes", "README.md", *files]

    def hf_hub_download(repo_id, filename, repo_type, local_dir):
        downloaded.append(filename)
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = files[filename]
        pq.write_table(pa.table({"messages": rows, "source": ["test"] * len(rows)}), path)
        return str(path)

    monkeypatch.setattr("huggingface_hub.HfApi.list_repo_files", list_repo_files)
    monkeypatch.setattr("ullm.data.hf_hub_download", hf_hub_download)
    return downloaded


def load_split(directory: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    return np.load(directory / f"{split}_ids.npy"), np.load(directory / f"{split}_mask.npy")


def unpadded(row: np.ndarray) -> list[int]:
    ids = row.tolist()
    while ids and ids[-1] == ENDOFTEXT:
        ids.pop()
    return ids


def test_prepare_writes_padded_train_and_val_rows(tmp_path, monkeypatch, capsys):
    downloaded = fake_smoltalk(monkeypatch, CONVERSATIONS)
    prepare_sft(tmp_path, workers=2, row_tokens=40)

    assert sorted(downloaded) == sorted(CONVERSATIONS)
    ids, mask = load_split(tmp_path, "train")
    assert ids.dtype == np.uint16 and mask.dtype == np.uint8
    assert ids.shape == mask.shape == (3, 40)
    kept = [
        ("What is 2 plus 2?", "It is 4."),
        ("Name a color.", "Blue.", "Another?", "Red."),
        ("Say yes.", "Yes."),
    ]
    for row_ids, row_mask, turns in zip(ids, mask, kept, strict=True):
        want = rendered(*turns)
        pad = 40 - len(want)
        assert row_ids.tolist() == want + [ENDOFTEXT] * pad
        assert row_mask.tolist() == rendered_mask(*turns) + [0] * pad

    val_ids, val_mask = load_split(tmp_path, "val")
    assert [unpadded(row) for row in val_ids] == [
        rendered("Say no.", "No."),
        rendered("Count to three.", "One, two, three."),
    ]
    assert val_mask.sum() == len(encode("No.")) + 1 + len(encode("One, two, three.")) + 1
    assert not (tmp_path / "raw").exists()

    out = capsys.readouterr().out
    assert "train: kept 3 | skipped 2 | truncated 0" in out
    assert "val: kept 2 | skipped 0 | truncated 0" in out
    assert "[[It is 4.<|end|>]]" in out


def test_prepare_counts_truncated_rows_and_limits_conversations(tmp_path, monkeypatch, capsys):
    files = {
        "data/train-00000-of-00001.parquet": [
            chat("Hi", "Hello", "And then?", "word " * 30),
            chat("Say yes.", "Yes."),
            chat("Say no.", "No."),
        ],
        "data/test-00000-of-00001.parquet": [chat("Hi", "Hello"), chat("Yo", "Hey")],
    }
    fake_smoltalk(monkeypatch, files)
    prepare_sft(tmp_path, workers=1, limit=2, row_tokens=20)

    ids, _ = load_split(tmp_path, "train")
    assert [unpadded(row) for row in ids] == [rendered("Hi", "Hello"), rendered("Say yes.", "Yes.")]
    assert len(load_split(tmp_path, "val")[0]) == 2
    assert "train: kept 2 | skipped 0 | truncated 1" in capsys.readouterr().out


def test_prepare_sft_command(tmp_path, monkeypatch):
    fake_smoltalk(monkeypatch, CONVERSATIONS)
    out = tmp_path / "sft"
    main(["prepare-sft", "--out", str(out), "--workers", "1", "--limit", "1"])
    ids, mask = load_split(out, "train")
    assert ids.shape == mask.shape == (1, SFT_ROW_TOKENS)
    assert ids[0, : len(rendered("What is 2 plus 2?", "It is 4."))].tolist() == rendered(
        "What is 2 plus 2?", "It is 4."
    )


def test_prepare_sft_help(capsys):
    with pytest.raises(SystemExit):
        main(["prepare-sft", "--help"])
    out = capsys.readouterr().out
    assert all(flag in out for flag in ["--out", "--workers", "--limit"])
