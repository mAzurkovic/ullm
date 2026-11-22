import argparse
import os
import sys
import time
from collections.abc import Callable, Collection, Sequence
from pathlib import Path

import torch

from ullm.config import (
    ModelConfig,
    PretrainConfig,
    SFTConfig,
    add_config_args,
    config_from_args,
)
from ullm.data import prepare_pretrain, prepare_sft
from ullm.eval import (
    generation_test,
    load_benchmarks,
    run_benchmarks,
    save_benchmarks,
    write_report,
)
from ullm.generate import generate
from ullm.model import GPT, default_device
from ullm.tokenizer import ASSISTANT, END, ENDOFTEXT, Message, StreamDecoder, encode, render_chat
from ullm.train import pretrain, sft


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_pretrain(args.out, args.num_shards, args.workers, args.push)
        return
    if args.command == "prepare-sft":
        prepare_sft(args.out, args.workers, args.limit)
        return
    if args.command == "eval":
        _evaluate(args)
        return
    if args.command == "pretrain":
        pretrain(config_from_args(ModelConfig, args), config_from_args(PretrainConfig, args))
        return
    if args.command == "sft":
        sft(config_from_args(SFTConfig, args))
        return
    if args.command == "complete" and not args.prompt:
        parser.error("prompt must not be empty")
    if not args.temperature >= 0 or args.top_k < 0 or args.max_new_tokens < 1:
        parser.error("need --temperature >= 0, --top-k >= 0, --max-new-tokens >= 1")
    if not (args.model / "config.json").is_file():
        parser.error(f"no checkpoint at {args.model}")

    device = args.device or default_device()
    print(f"Loading {args.model} on {device}...", end=" ", file=sys.stderr, flush=True)
    start = time.perf_counter()
    model = GPT.load(args.model, device).eval()
    print(f"ready ({time.perf_counter() - start:.1f}s)", file=sys.stderr)
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


def _evaluate(args: argparse.Namespace) -> None:
    device = args.device or default_device()
    run_dir = Path("results") / args.run
    benchmarks = run_benchmarks(args.model, device, args.limit, args.batch_size)
    save_benchmarks(benchmarks, run_dir / "benchmarks.json")
    samples = []
    baseline = None
    if args.model != "gpt2":
        model = GPT.load(Path(args.model), device)
        samples = generation_test(model, max_new_tokens=args.max_new_tokens)
        if args.baseline.is_file():
            baseline = load_benchmarks(args.baseline)
    write_report(run_dir / "results.md", args.run, benchmarks, baseline, samples)
    print(f"wrote {run_dir / 'results.md'}")


def _sampling(temperature: float) -> argparse.ArgumentParser:
    sampling = argparse.ArgumentParser(add_help=False)
    sampling.add_argument("--model", type=Path, required=True, help="checkpoint directory")
    sampling.add_argument("--temperature", type=float, default=temperature)
    sampling.add_argument("--top-k", type=int, default=50)
    sampling.add_argument("--max-new-tokens", type=int, default=256)
    sampling.add_argument("--seed", type=int, default=None)
    sampling.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    return sampling


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ullm")
    commands = parser.add_subparsers(dest="command", required=True)
    complete = commands.add_parser(
        "complete", parents=[_sampling(temperature=0.8)], help="continue a text prompt"
    )
    complete.add_argument("prompt")
    commands.add_parser(
        "chat", parents=[_sampling(temperature=0.5)], help="chat with the assistant model"
    )
    prepare = commands.add_parser("prepare", help="tokenize FineWeb-Edu into pre-train shards")
    prepare.add_argument("--out", type=Path, default=Path("data/shards"))
    prepare.add_argument("--num-shards", type=int, default=71)
    prepare.add_argument("--workers", type=int, default=os.cpu_count())
    prepare.add_argument("--push", metavar="REPO", help="HF Hub dataset repo to upload to")
    prepare_sft_parser = commands.add_parser(
        "prepare-sft", help="tokenize smol-smoltalk into masked fine-tuning rows"
    )
    prepare_sft_parser.add_argument("--out", type=Path, default=Path("data/sft"))
    prepare_sft_parser.add_argument("--workers", type=int, default=os.cpu_count())
    prepare_sft_parser.add_argument(
        "--limit", type=int, default=None, help="read at most this many conversations per split"
    )
    evaluate = commands.add_parser("eval", help="benchmarks and generation test")
    evaluate.add_argument("--model", required=True, help="checkpoint directory or gpt2")
    evaluate.add_argument("--run", required=True, help="name of the results/<run> directory")
    evaluate.add_argument("--limit", type=int, default=None, help="examples per benchmark")
    evaluate.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--max-new-tokens", type=int, default=256)
    evaluate.add_argument(
        "--baseline", type=Path, default=Path("results/gpt2-small/benchmarks.json")
    )
    pretrain_parser = commands.add_parser("pretrain", help="pre-train the base model")
    add_config_args(pretrain_parser, ModelConfig)
    add_config_args(pretrain_parser, PretrainConfig)
    sft_parser = commands.add_parser("sft", help="fine-tune a base model into the assistant")
    add_config_args(sft_parser, SFTConfig)
    return parser
