# M4 F1. Real run — Tasks

Scope: the real pre-training run on 8×H100 SXM, a first try of `ullm complete`, and the base weights published to `mattiasz/ullm-base`. The user runs the GPU node over SSH; Claude reads pasted output. Source: `design.md` (Training Recipe, D3, D4, D15), requirements R1, R5; README "Train on a GPU node"; `results/h100-test/results.md` (projection: ≈1.8 h, ≈$49 at $3.39 per GPU-hour). No new training code. Benchmarks are M6.

## 1. Model repo and card are ready [serial]
- [ ] 1.1 User creates the public model repo: `hf repo create mattiasz/ullm-base --repo-type model`. Repo visible on the Hub.
- [x] 1.2 Claude writes `docs/model-card-base.md`: what the model is (360M decoder-only GPT, RoPE, RMSNorm, SwiGLU, context 1024, GPT-2 tokenizer + 3 chat tokens), training data (FineWeb-Edu `sample-10BT`, 7B tokens, ODC-By, credit), training setup and cost, how to run it with `ullm complete`, limits (base model: continues text, wrong facts, English only), license. File committed.

## 2. The 8×H100 node is ready [serial, needs 1]
- [ ] 2.1 User rents an on-demand 8×H100 SXM node (Vast.ai or RunPod), ~80 GB disk, CUDA 13+. Node price per hour noted.
- [ ] 2.2 User runs README steps 1 to 2; pastes `nvidia-smi`. Claude checks: 8 × H100 80GB HBM3, CUDA ≥ 13.0, no other processes.

## 3. The real run trains 7B tokens [serial, needs 2]
- [ ] 3.1 User starts in `tmux`: `uv run torchrun --standalone --nproc_per_node=8 -m ullm pretrain --run base-7b --device cuda --compile --hourly-rate <node price> --shards-repo mattiasz/ullm-fineweb-edu-shards`. Defaults give 13,351 steps of 512K tokens, micro-batch 16, 4 accumulation steps per GPU.
- [ ] 3.2 Claude checks the first log lines (steady tok/s ≈ 1.1 to 1.25M, MFU ≈ 30 to 35%) and the step-200 budget check (no stop). If it stops over budget: rerun with the printed `--max-tokens` and `--run base-7b-b`.
- [ ] 3.3 User watches it to the end (val loss every 250 steps, samples every 500); after a crash, rerun the same command with `--resume`.

## 4. The base model is tried and published [serial, needs 3]
- [ ] 4.1 On the node: `uv run ullm complete` on 3 prompts from `ullm/prompts.py`; paste output. Claude checks the text is fluent English.
- [ ] 4.2 User copies `results/base-7b/` and `checkpoints/base-7b/final/` to the Mac (README step 6) and checks the files, then destroys the node.
- [ ] 4.3 User uploads: copy `docs/model-card-base.md` to `checkpoints/base-7b/final/README.md`, then `hf upload mattiasz/ullm-base checkpoints/base-7b/final .`. Repo shows `model.safetensors`, `config.json`, `README.md`.
- [ ] 4.4 On the Mac: `hf download mattiasz/ullm-base --local-dir checkpoints/ullm-base` and `uv run ullm complete "The capital of France is" --model checkpoints/ullm-base`. Text generated from the published weights.
- [ ] 4.5 Claude writes `results/base-7b/results.md` (GPU, price, time, total cost, final train and val loss, tok/s, MFU, samples) and commits `results/base-7b/`.

## Dependencies
- 2 needs 1.
- 3 needs 2.
- 4 needs 3.

## Open questions
- OQ-L. License: Apache-2.0 (decided). FineWeb-Edu credit stays in the card per ODC-By.
