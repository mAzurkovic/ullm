import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import lm_eval
import torch
import torch.nn.functional as F
import transformers
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window

from ullm.generate import generate
from ullm.model import GPT
from ullm.prompts import BASE_PROMPTS
from ullm.tokenizer import ENDOFTEXT, decode, encode

TASKS = ["hellaswag", "wikitext"]


@dataclass(frozen=True)
class Benchmarks:
    hellaswag_acc_norm: float
    wikitext_word_perplexity: float


@dataclass(frozen=True)
class Sample:
    prompt: str
    seed: int
    text: str


class LogitsLM(LM):
    """Scores text with any causal model that maps token ids to next-token logits."""

    def __init__(self, model: torch.nn.Module, context_length: int, batch_size: int = 32) -> None:
        super().__init__()
        self.model = model.eval()
        self.context_length = context_length
        self.batch_size = batch_size

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        return self._score([self._split(*r.args) for r in requests])

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        scores = []
        for r in requests:
            windows = get_rolling_token_windows(
                token_list=encode(r.args[0]),
                prefix_token=ENDOFTEXT,
                max_seq_len=self.context_length,
                context_len=1,
            )
            scores.append(
                sum(score for score, _ in self._score(map(make_disjoint_window, windows)))
            )
        return scores

    def generate_until(self, requests: list[Instance]) -> list[str]:
        raise NotImplementedError("the benchmarks used here only need log-likelihoods")

    @staticmethod
    def _split(context: str, continuation: str) -> tuple[list[int], list[int]]:
        # A trailing space belongs to the next word in GPT-2 tokens.
        spaces = len(context) - len(context.rstrip())
        if spaces:
            context, continuation = context[:-spaces], context[-spaces:] + continuation
        if not context:
            return [ENDOFTEXT], encode(continuation)
        context_ids = encode(context)
        return context_ids, encode(context + continuation)[len(context_ids) :]

    @torch.no_grad()
    def _score(self, pairs) -> list[tuple[float, bool]]:
        device = next(self.model.parameters()).device
        limit = self.context_length + 1
        sequences = [((ctx + cont)[-limit:], len(cont)) for ctx, cont in pairs]
        results = []
        for start in range(0, len(sequences), self.batch_size):
            batch = sequences[start : start + self.batch_size]
            width = max(len(tokens) for tokens, _ in batch) - 1
            inputs = torch.zeros(len(batch), width, dtype=torch.long)
            for row, (tokens, _) in enumerate(batch):
                inputs[row, : len(tokens) - 1] = torch.tensor(tokens[:-1])
            logprobs = F.log_softmax(self.model(inputs.to(device)).float(), dim=-1).cpu()
            for row, (tokens, n_continuation) in enumerate(batch):
                end = len(tokens) - 1
                positions = torch.arange(end - n_continuation, end)
                targets = torch.tensor(tokens[-n_continuation:])
                picked = logprobs[row, positions]
                score = picked[torch.arange(n_continuation), targets].sum().item()
                results.append((score, bool((picked.argmax(-1) == targets).all())))
        return results


class GPT2Logits(torch.nn.Module):
    def __init__(self, net: transformers.GPT2LMHeadModel) -> None:
        super().__init__()
        self.net = net

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.net(ids).logits


def load_gpt2(device: str) -> GPT2Logits:
    """GPT-2 small, called like GPT: token ids in, logits out. It shares our tokenizer."""
    return GPT2Logits(transformers.GPT2LMHeadModel.from_pretrained("gpt2")).to(device)


def run_benchmarks(model: str, device: str, limit: int | None, batch_size: int = 32) -> Benchmarks:
    if model == "gpt2":
        gpt2 = load_gpt2(device)
        lm = LogitsLM(gpt2, gpt2.net.config.n_positions, batch_size)
    else:
        ullm = GPT.load(Path(model), device)
        lm = LogitsLM(ullm, ullm.config.context_length, batch_size)
    output = lm_eval.simple_evaluate(
        model=lm,
        tasks=TASKS,
        num_fewshot=0,
        limit=limit,
        batch_size=batch_size,
        device=device,
    )
    results = output["results"]
    return Benchmarks(
        hellaswag_acc_norm=results["hellaswag"]["acc_norm,none"],
        wikitext_word_perplexity=results["wikitext"]["word_perplexity,none"],
    )


def save_benchmarks(benchmarks: Benchmarks, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(benchmarks), indent=2))


def load_benchmarks(path: Path) -> Benchmarks:
    return Benchmarks(**json.loads(path.read_text()))


def generation_test(
    model: GPT,
    prompts: Sequence[str] = BASE_PROMPTS,
    seeds: Sequence[int] = (0, 1, 2),
    max_new_tokens: int = 256,
) -> list[Sample]:
    samples = []
    for prompt in prompts:
        for seed in seeds:
            ids = generate(
                model,
                encode(prompt),
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_k=50,
                stop_ids={ENDOFTEXT},
                generator=torch.Generator().manual_seed(seed),
            )
            samples.append(Sample(prompt, seed, decode(list(ids))))
    return samples


def write_report(
    path: Path,
    run: str,
    benchmarks: Benchmarks,
    baseline: Benchmarks | None,
    samples: Sequence[Sample],
) -> None:
    lines = [f"# {run}", "", "## Benchmarks", ""]
    if baseline is None:
        lines += [
            "| Benchmark | Score |",
            "|---|---|",
            f"| HellaSwag acc_norm (0-shot) | {benchmarks.hellaswag_acc_norm:.4f} |",
            f"| WikiText word perplexity | {benchmarks.wikitext_word_perplexity:.2f} |",
            "",
            "GPT-2 small baseline missing: run `ullm eval --model gpt2 --run gpt2-small` first.",
        ]
    else:
        acc_pass = benchmarks.hellaswag_acc_norm > baseline.hellaswag_acc_norm
        ppl_pass = benchmarks.wikitext_word_perplexity < baseline.wikitext_word_perplexity
        lines += [
            "| Benchmark | ullm | GPT-2 small | Result |",
            "|---|---|---|---|",
            f"| HellaSwag acc_norm (0-shot) | {benchmarks.hellaswag_acc_norm:.4f} | "
            f"{baseline.hellaswag_acc_norm:.4f} | {'pass' if acc_pass else 'fail'} |",
            f"| WikiText word perplexity | {benchmarks.wikitext_word_perplexity:.2f} | "
            f"{baseline.wikitext_word_perplexity:.2f} | {'pass' if ppl_pass else 'fail'} |",
        ]
    if samples:
        lines += [
            "",
            "## Generation test",
            "",
            "Judge each prompt by its best sample: grammatical throughout, "
            "on topic for at least ~150 tokens, no repetition loops.",
        ]
        prompts = list(dict.fromkeys(s.prompt for s in samples))
        for number, prompt in enumerate(prompts, 1):
            lines += ["", f"### {number}. {prompt}", "", "- [ ] pass"]
            for sample in (s for s in samples if s.prompt == prompt):
                lines += ["", f"**Seed {sample.seed}**", "", "```text", prompt + sample.text, "```"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
