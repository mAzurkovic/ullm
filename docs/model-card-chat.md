---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
base_model: mattiasz/ullm-base
datasets:
- HuggingFaceTB/smol-smoltalk
tags:
- gpt
- from-scratch
- chat
---

# μllm chat (ullm-chat)

A 360M-parameter chat model built from scratch with [ullm](https://github.com/mAzurkovic/ullm), a small, readable GPT implementation built to learn how LLMs work end to end.

It is [mattiasz/ullm-base](https://huggingface.co/mattiasz/ullm-base) after supervised fine-tuning on chat conversations. It answers in turns and stops at the end of its reply.

## Model

| | |
|---|---|
| Parameters | 359,842,816 (input and output embeddings tied) |
| Architecture | decoder-only transformer, 24 layers, d_model 1024, 16 heads (head dim 64) |
| Block | pre-norm RMSNorm, RoPE, SwiGLU MLP (hidden 2816), no biases |
| Context | 1024 tokens |
| Tokenizer | GPT-2 (`tiktoken`) plus `<\|user\|>`, `<\|assistant\|>`, `<\|end\|>` (vocab padded to 50,304) |
| Files | `model.safetensors` (fp32 weights), `config.json` |

## Chat format

```
<|user|>What is the capital of France?<|end|><|assistant|>The capital of France is Paris.<|end|>
```

Each turn ends with `<|end|>`. There is no system prompt. To get a reply, end the prompt with `<|assistant|>` and stop at `<|end|>`.

## Training

| | |
|---|---|
| Start | [mattiasz/ullm-base](https://huggingface.co/mattiasz/ullm-base) (7B FineWeb-Edu tokens) |
| Data | [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk): 384,253 conversations; ones with a system message skipped, long ones cut after the last exchange that fits 1024 tokens |
| Loss | assistant replies and their `<\|end\|>` only |
| Steps | 1 epoch, 6,003 steps of 64 conversations |
| Optimizer | AdamW, β = (0.9, 0.95), weight decay 0.1 on matrices |
| Schedule | peak LR 3e-5, 100 warmup steps, linear decay to 0 |
| Hardware | 1 × NVIDIA H100 NVL, 57 min |
| Precision | bf16 autocast, fp32 weights, `torch.compile` |
| Val loss | 1.248 on the dataset's test split (assistant tokens only) |

Full results: `results/sft/` in the GitHub repo.

## Use

The weights load with ullm, not with `transformers`.

```sh
git clone https://github.com/mAzurkovic/ullm.git && cd ullm && uv sync
uv run hf download mattiasz/ullm-chat --local-dir checkpoints/ullm-chat
uv run ullm chat --model checkpoints/ullm-chat
```

It runs on CUDA, Apple Silicon (MPS), or CPU. `ullm chat` keeps the conversation history and drops the oldest turns past 1024 tokens.

## Limits

- It states wrong facts with confidence. Fine-tuning taught the chat format, not new facts. Do not rely on it for facts.
- Long answers drift off topic and can run to the token limit.
- English only. No safety tuning.

## Credit and license

Pre-training data: FineWeb-Edu by Hugging Face, derived from Common Crawl, released under ODC-By 1.0. Fine-tuning data: smol-smoltalk by Hugging Face, released under Apache-2.0. These weights are released under Apache-2.0.
