from collections.abc import Collection, Iterator

import torch

from ullm.model import GPT
from ullm.tokenizer import N_VOCAB


def generate(
    model: GPT,
    ids: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    stop_ids: Collection[int],
    generator: torch.Generator | None = None,
) -> Iterator[int]:
    if not ids:
        raise ValueError("prompt must contain at least one token")
    device = next(model.parameters()).device
    context = list(ids)
    for _ in range(max_new_tokens):
        window = torch.tensor([context[-model.config.context_length :]], device=device)
        with torch.inference_mode():
            logits = model(window)[0, -1, :N_VOCAB].float().cpu()
        token = _sample(logits, temperature, top_k, generator)
        if token in stop_ids:
            return
        context.append(token)
        yield token


def _sample(
    logits: torch.Tensor, temperature: float, top_k: int | None, generator: torch.Generator | None
) -> int:
    if temperature == 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k:
        kth = logits.topk(min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
