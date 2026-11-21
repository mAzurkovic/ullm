# M1 F2. CLI — Tasks

Scope: `ullm/tokenizer.py`, `ullm/generate.py`, `ullm/cli.py` (`complete`, `chat`), checkpoint save and load in `ullm/model.py`. Source: `design.md` (Data Model, Inference, Components). Reuses `ModelConfig` and `GPT`. Local checkpoint dirs only. No HF Hub load, no SFT truncation rules.

## 1. Text round-trips through the tokenizer and chat format [serial]
- [x] 1.1 Add `tiktoken` and `safetensors` to `pyproject.toml`. `uv sync` succeeds.
- [x] 1.2 `ullm/tokenizer.py`: GPT-2 `tiktoken` encoding plus `<|user|>` 50257, `<|assistant|>` 50258, `<|end|>` 50259; `encode`, `decode`, token id constants. Round-trip test passes.
- [x] 1.3 `encode` treats special-token text in input as plain text. Test: `"<|end|>"` typed by a user does not encode to 50259.
- [x] 1.4 `Message` (role, content) and `render_chat(messages) -> (ids, mask)`: `<|user|>{text}<|end|><|assistant|>{text}<|end|>...`, mask 1 on assistant tokens and their `<|end|>`. Test on a 2-turn conversation passes.

## 2. A model saves and loads as one portable checkpoint [parallel, needs 1.1]
- [x] 2.1 `GPT.save(dir)`: `model.safetensors` via `safetensors.torch.save_model` (tied weights) and `config.json` from `ModelConfig`. Files written.
- [x] 2.2 `GPT.load(dir, device)`: build from `config.json`, load weights. Test: loaded model gives identical logits to the saved one.

## 3. A prompt generates tokens [parallel, needs 1]
- [x] 3.1 `ullm/generate.py`: `generate(model, ids, max_new_tokens, temperature, top_k, stop_ids, generator)` yields token ids; feeds last `context_length` tokens; temperature, then top-k, then sample; temperature 0 = greedy. Tests: greedy is deterministic, top-k 1 equals greedy.
- [x] 3.2 Stops on any id in `stop_ids` (not yielded) or at `max_new_tokens`. Prompt longer than context still generates. Tests pass.
- [x] 3.3 Streaming text decode that never splits a UTF-8 character across tokens. Test: multi-byte text streamed token by token equals `decode` of all tokens.

## 4. `ullm complete` and `ullm chat` run a checkpoint [serial, needs 2, 3]
- [x] 4.1 `ullm/cli.py` with argparse, `ullm/__main__.py`, `[project.scripts] ullm`. Flags `--model`, `--temperature` 0.8, `--top-k` 50, `--max-new-tokens` 256, `--seed` (random default), `--device` auto (cuda, mps, cpu). `ullm --help` works.
- [x] 4.2 `complete PROMPT`: encode, stream output, stop on `<|endoftext|>`. Test with a tiny random checkpoint in a temp dir.
- [x] 4.3 `chat`: read user lines, render history + `<|assistant|>`, stream reply, stop on `<|end|>`, add reply to history, exit on EOF. Test with stdin fed 2 turns.
- [x] 4.4 Chat history drops oldest full turns when the rendered prompt exceeds context. Test with a tiny context.

## Dependencies
- 2 needs 1.1.
- 3 needs 1.
- 4 needs 2 and 3.
