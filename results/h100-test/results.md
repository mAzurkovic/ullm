# h100-test

Short throughput and correctness test before the real pre-training run.

## Setup

| | |
|---|---|
| Provider | Vast.ai, on-demand, instance 61794468 |
| GPU | 1 × NVIDIA H100 80GB HBM3 (SXM, 700 W), driver 610.57, CUDA 13.3 |
| Price | $3.386 per hour |
| Model | 360M params (24 layers, d_model 1024), context 1024 |
| Training | `--compile`, bf16 autocast, batch 512 × 1024 tokens, micro-batch 16, 32 accumulation steps |
| Run | 95 steps (50M tokens), stopped after step 40 and resumed from `step_000040` |

## Results

| | |
|---|---|
| Throughput | 156,200 tokens/s (steady, steps 20 to 95) |
| MFU | 34.1% (of 989 TFLOPs bf16) |
| Step time | 3.36 s per 524,288-token step |
| First window | 62,253 tokens/s: includes `torch.compile` (about 50 s) |
| Loss | train 10.05 → 7.36, val 8.43 (step 50) → 7.30 (step 95) |
| Resume | loss continued from 8.72 (step 40) to 8.42 (step 50); no duplicate rows in `metrics.csv` |
| Cost | $0.36 of training time; about $1.40 for the whole instance session |

## Projection: 7B tokens on 8 × H100 SXM

| | Estimate |
|---|---|
| Throughput | 156,200 × 8 × 0.95 (scaling) ≈ 1.19M tokens/s |
| Training time | 7e9 ÷ 1.19M ≈ 5,900 s ≈ 1.64 h |
| With setup, download, compile | ≈ 1.8 h |
| Cost at 8 × $3.386 = $27.09 per hour | ≈ $49 |

Within the $75 and 24 h limits, with about $25 of margin. The step-200 budget check stops the real run if the 8× node is slower or more expensive than this.
