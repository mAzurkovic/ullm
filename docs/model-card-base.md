---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
datasets:
- HuggingFaceFW/fineweb-edu
- mattiasz/ullm-fineweb-edu-shards
tags:
- gpt
- from-scratch
- base-model
---

# μllm base (ullm-base)

A 360M-parameter decoder-only language model trained from scratch with [ullm](https://github.com/mAzurkovic/ullm), a small, readable GPT implementation built to learn how LLMs work end to end.

This is the **base model**: it continues text. It does not follow instructions or chat. The chat model, [mattiasz/ullm-chat](https://huggingface.co/mattiasz/ullm-chat), is trained from it with supervised fine-tuning.

## Model

| | |
|---|---|
| Parameters | 359,842,816 (input and output embeddings tied) |
| Architecture | decoder-only transformer, 24 layers, d_model 1024, 16 heads (head dim 64) |
| Block | pre-norm RMSNorm, RoPE, SwiGLU MLP (hidden 2816), no biases |
| Context | 1024 tokens |
| Tokenizer | GPT-2 (`tiktoken`) plus `<\|user\|>`, `<\|assistant\|>`, `<\|end\|>` (vocab padded to 50,304) |
| Files | `model.safetensors` (fp32 weights), `config.json` |

## Training

| | |
|---|---|
| Data | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) `sample-10BT`, tokenized: [mattiasz/ullm-fineweb-edu-shards](https://huggingface.co/datasets/mattiasz/ullm-fineweb-edu-shards) |
| Tokens | 7B (about 20 tokens per parameter) |
| Hardware | 4 × NVIDIA H100 SXM |
| Optimizer | AdamW, β = (0.9, 0.95), weight decay 0.1 on matrices |
| Schedule | peak LR 3e-4, 700 warmup steps, cosine decay to 10% |
| Batch | 512 × 1024 tokens per step, 13,351 steps |
| Precision | bf16 autocast, fp32 weights, `torch.compile` |

Training time, cost, and final loss: see `results/base-7b/` in the GitHub repo.

## Use

The weights load with ullm, not with `transformers`.

```sh
git clone https://github.com/mAzurkovic/ullm.git && cd ullm && uv sync
hf download mattiasz/ullm-base --local-dir checkpoints/ullm-base
uv run ullm complete "The capital of France is" --model checkpoints/ullm-base
```

It runs on CUDA, Apple Silicon (MPS), or CPU. For question answering, give it a pattern to continue, for example a few `Q: ... A: ...` pairs.

## Limits

- Base model: it continues text and can drift off topic or repeat itself.
- It states wrong facts with confidence. Do not rely on it for facts.
- English only, educational web text. No safety tuning.

## Credit and license

Training data: FineWeb-Edu by Hugging Face, derived from Common Crawl, released under ODC-By 1.0. These weights are released under Apache-2.0.
