import codecs
from dataclasses import dataclass
from functools import cache
from typing import Literal

import tiktoken

ENDOFTEXT = 50256
USER = 50257
ASSISTANT = 50258
END = 50259
N_VOCAB = END + 1

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@cache
def _encoding() -> tiktoken.Encoding:
    gpt2 = tiktoken.get_encoding("gpt2")
    return tiktoken.Encoding(
        name="ullm",
        pat_str=gpt2._pat_str,
        mergeable_ranks=gpt2._mergeable_ranks,
        special_tokens={
            "<|endoftext|>": ENDOFTEXT,
            "<|user|>": USER,
            "<|assistant|>": ASSISTANT,
            "<|end|>": END,
        },
    )


def encode(text: str) -> list[int]:
    return _encoding().encode_ordinary(text)


def decode(ids: list[int]) -> str:
    return _encoding().decode(ids)


def render_chat(messages: list[Message]) -> tuple[list[int], list[int]]:
    ids: list[int] = []
    mask: list[int] = []
    for message in messages:
        content = encode(message.content)
        role = USER if message.role == "user" else ASSISTANT
        learn = int(message.role == "assistant")
        ids += [role, *content, END]
        mask += [0] + [learn] * (len(content) + 1)
    return ids, mask


class StreamDecoder:
    def __init__(self) -> None:
        self._utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, token: int) -> str:
        return self._utf8.decode(_encoding().decode_single_token_bytes(token))

    def flush(self) -> str:
        return self._utf8.decode(b"", final=True)
