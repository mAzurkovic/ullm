import argparse
import os
from collections.abc import Callable, Collection, Sequence
from pathlib import Path

import torch

from ullm.data import prepare_pretrain
from ullm.generate import generate
from ullm.model import GPT
from ullm.tokenizer import ASSISTANT, END, ENDOFTEXT, Message, StreamDecoder, encode, render_chat


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_pretrain(args.out, args.num_shards, args.workers, args.push)
        return
    if args.command == "complete" and not args.prompt:
        parser.error("prompt must not be empty")
    if not args.temperature >= 0 or args.top_k < 0 or args.max_new_tokens < 1:
        parser.error("need --temperature >= 0, --top-k >= 0, --max-new-tokens >= 1")
    if not (args.model / "config.json").is_file():
        parser.error(f"no checkpoint at {args.model}")

    model = GPT.load(args.model, args.device or _default_device()).eval()
    generator = torch.Generator()
    if args.seed is None:
        generator.seed()
    else:
        generator.manual_seed(args.seed)

    def reply(ids: list[int], stop_ids: Collection[int]) -> str:
        return _stream(model, ids, args, stop_ids, generator)

    try:
        if args.command == "complete":
            reply(encode(args.prompt), {ENDOFTEXT})
        else:
            chat_loop(reply, model.config.context_length)
    except KeyboardInterrupt:
        print()


def trim_history(history: list[Message], context_length: int) -> list[Message]:
    while len(history) > 1 and len(render_chat(history)[0]) + 1 > context_length:
        history = history[2:]
    return history


def chat_loop(reply: Callable[[list[int], Collection[int]], str], context_length: int) -> None:
    history: list[Message] = []
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            print()
            return
        if not line:
            continue
        history = trim_history([*history, Message("user", line)], context_length)
        ids = render_chat(history)[0] + [ASSISTANT]
        history.append(Message("assistant", reply(ids, {END})))


def _stream(
    model: GPT,
    ids: list[int],
    args: argparse.Namespace,
    stop_ids: Collection[int],
    generator: torch.Generator,
) -> str:
    decoder = StreamDecoder()
    pieces = []
    tokens = generate(
        model,
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        stop_ids=stop_ids,
        generator=generator,
    )
    for token in tokens:
        pieces.append(decoder.push(token))
        print(pieces[-1], end="", flush=True)
    pieces.append(decoder.flush())
    print(pieces[-1])
    return "".join(pieces)


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parser() -> argparse.ArgumentParser:
    sampling = argparse.ArgumentParser(add_help=False)
    sampling.add_argument("--model", type=Path, required=True, help="checkpoint directory")
    sampling.add_argument("--temperature", type=float, default=0.8)
    sampling.add_argument("--top-k", type=int, default=50)
    sampling.add_argument("--max-new-tokens", type=int, default=256)
    sampling.add_argument("--seed", type=int, default=None)
    sampling.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)

    parser = argparse.ArgumentParser(prog="ullm")
    commands = parser.add_subparsers(dest="command", required=True)
    complete = commands.add_parser("complete", parents=[sampling], help="continue a text prompt")
    complete.add_argument("prompt")
    commands.add_parser("chat", parents=[sampling], help="chat with the assistant model")
    prepare = commands.add_parser("prepare", help="tokenize FineWeb-Edu into pre-train shards")
    prepare.add_argument("--out", type=Path, default=Path("data/shards"))
    prepare.add_argument("--num-shards", type=int, default=71)
    prepare.add_argument("--workers", type=int, default=os.cpu_count())
    prepare.add_argument("--push", metavar="REPO", help="HF Hub dataset repo to upload to")
    return parser
