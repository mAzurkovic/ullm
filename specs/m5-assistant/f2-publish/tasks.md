# M5 F2. Publish — Tasks

Scope: put the SFT weights (`checkpoints/sft/final/`) on the Hub as `mattiasz/ullm-chat` with a model card, link the two model repos, and make the README show one real command per stage. No new code. Source: `design.md` (R5, D9, D10, OQ5), `results/sft/results.md`, M4 F1 tasks 1.2 and 4.3 (how the base model was published). Eval is M6.

## 1. The chat model card is ready [serial]
- [x] 1.1 User creates the public model repo: `hf repo create mattiasz/ullm-chat --repo-type model`. Repo visible on the Hub.
- [x] 1.2 Claude writes `docs/model-card-chat.md`, same layout as `docs/model-card-base.md`. Front matter: `license: apache-2.0`, `base_model: mattiasz/ullm-base`, `datasets: HuggingFaceTB/smol-smoltalk`, `pipeline_tag: text-generation`. Sections: chat format (`<|user|>…<|end|><|assistant|>…<|end|>`, no system prompt), SFT setup (1 epoch, 384,253 conversations, assistant-only loss, LR 3e-5, 1 × H100 NVL, 57 min), use with `ullm chat`, limits (wrong facts, drifts on long answers), credit (smol-smoltalk, Apache-2.0). File committed.
- [x] 1.3 `docs/model-card-base.md`: link "The chat model" to `mattiasz/ullm-chat`. File committed.

## 2. The chat weights are on the Hub [serial, needs 1]
- [x] 2.1 User uploads from the Mac: `cp docs/model-card-chat.md checkpoints/sft/final/README.md`, then `uv run hf upload mattiasz/ullm-chat checkpoints/sft/final .`. Repo shows `model.safetensors`, `config.json`, `README.md`.
- [x] 2.2 User updates the base card: `uv run hf upload mattiasz/ullm-base docs/model-card-base.md README.md`. Base repo page links to the chat repo.
- [x] 2.3 On the Mac: `uv run hf download mattiasz/ullm-chat --local-dir checkpoints/ullm-chat`, then `uv run ullm chat --model checkpoints/ullm-chat`, 1 question; paste. Claude checks the reply ends its turn (published copy works).

## 3. The README has one command per stage [parallel, needs 1]
- [x] 3.1 `README.md` "At a glance": add an "Assistant model" row linking `mattiasz/ullm-chat`; "Fine-tuning" row gets the measured run (smol-smoltalk, 1 × H100, 57 min, $3).
- [x] 3.2 `README.md` "Usage": drop "Planned"; real commands for each stage, with the flags they need (`--model` for `complete` and `chat`, `--base` for `sft`), plus `hf download` of both models; `ullm eval` marked as not built yet.
- [x] 3.3 `README.md`: "Fine-tuning" section after "Pre-training", with `results/sft/loss.png`, final val loss 1.248, link to `results/sft/results.md`.

## Dependencies
- 2 needs 1.
- 3 needs 1 (links point at the repo name). 3 can run beside 2.

## Open questions
- OQ-P1. Repo name: `mattiasz/ullm-chat` (decided).
- OQ-P2. License: Apache-2.0, same as the base weights; smol-smoltalk is Apache-2.0 (decided, credit in the card).
