<p align="center">
  <img src="assets/banner.svg" alt="μllm" width="720">
</p>

**μllm** (micro-LLM, or `ullm`) is a small, modern, fully featured large language model built from scratch. It has a pre-trained **base model** and a fine-tuned **assistant model** for chat.

The goal is to demystify LLMs by building every core part by hand: the transformer, the training loop, and inference. See [intro.md](intro.md) for the motivation.

## At a glance

| | |
|---|---|
| Model | Decoder-only GPT, ~350M params, 24 layers, d_model 1024, 16 heads |
| Block | RoPE, RMSNorm (pre-norm), SwiGLU, no biases, tied embeddings |
| Context | 1024 tokens |
| Tokenizer | GPT-2 (`tiktoken`) + 3 chat tokens: `<\|user\|>`, `<\|assistant\|>`, `<\|end\|>` |
| Pre-training | ~7B FineWeb-Edu tokens on one 8×H100 node, ~1.5 h, ≤ $75 |
| Pre-train data | [mattiasz/ullm-fineweb-edu-shards](https://huggingface.co/datasets/mattiasz/ullm-fineweb-edu-shards): 71 tokenized shards (shard 0 = validation) |
| Fine-tuning | SFT on smol-smoltalk, 1×H100 |
| Target | Beat GPT-2 small (124M) on HellaSwag and WikiText-103 |
| Inference | CLI on CUDA, Apple Silicon (MPS), or CPU |

We write the model, training (pre-train and SFT), data pipeline glue, inference, and CLI. Libraries handle tensors and autograd (PyTorch), tokenization (`tiktoken`), dataset download and upload (`huggingface_hub`, `pyarrow`), and benchmarks (`lm-evaluation-harness`). No pretrained weights are used for our model.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/mAzurkovic/ullm.git
cd ullm
uv sync
uv run pytest
```

## Usage

Planned: one command per stage.

```sh
uv run ullm prepare                                  # tokenize FineWeb-Edu into shards
uv run torchrun --nproc_per_node=8 -m ullm pretrain  # pre-train the base model
uv run ullm sft                                      # fine-tune the assistant
uv run ullm eval                                     # benchmarks and human-judged tests
uv run ullm complete "The meaning of life is"        # base model completion
uv run ullm chat                                     # chat with the assistant
```

## Project layout

```
ullm/
  config.py      # model and training config dataclasses
  model.py       # GPT: RoPE, RMSNorm, SwiGLU, attention, block
  tokenizer.py   # GPT-2 tokenizer + chat tokens, render_chat()
  data.py        # data preparation and loaders
  train.py       # shared loop for pre-training and SFT
  generate.py    # sampling loop
  eval.py        # benchmarks and generation tests
  cli.py         # ullm complete | chat | prepare | pretrain | sft | eval
tests/
assets/          # logo and banner
```

## Docs

- [intro.md](intro.md): background and motivation
- [design.md](specs/design.md): requirements and technical design
- [milestones.md](specs/milestones.md): build plan

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)
- Sebastian Raschka, *Build a Large Language Model (From Scratch)* (2024)
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)
