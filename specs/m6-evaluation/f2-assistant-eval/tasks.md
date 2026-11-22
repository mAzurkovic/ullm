# M6 F2. Assistant eval — Tasks

Scope: `ullm eval --assistant` runs the assistant test on `mattiasz/ullm-chat`: 20 fixed prompts and 3 fixed 2 to 3 turn conversations, written to `results/ullm-chat/results.md` for human judgment. Source: `design.md` (R2b, Evaluation table: "Assistant test: 20 prompts + 3 multi-turn convos", "All prompt sets fixed in `ullm/prompts.py`"). Reuses `generate`, `render_chat`, `Message`, `ASSISTANT`, `END`, `USER`, `GPT.load`, and `ullm eval` from F1. No benchmarks for the chat model; wrong facts are acceptable (R2b).

## 1. The assistant prompt sets are fixed [serial]
- [x] 1.1 `ullm/prompts.py` `ASSISTANT_PROMPTS`: 20 single questions, 7 factual, 7 how-to, 6 creative, none from `CHAT_PROMPTS`. Test: 20, all different, none in `CHAT_PROMPTS`.
- [x] 1.2 `ullm/prompts.py` `ASSISTANT_CONVERSATIONS`: 3 conversations of 2 to 3 user turns each; every later turn depends on the earlier reply ("make it shorter", "why?", "give another one"). Test: 3 conversations, 2 to 3 turns each.

## 2. The assistant test produces replies [serial, needs 1]
- [x] 2.1 `ullm/eval.py` `Reply` dataclass (`history: list[Message]`, `text`, `stopped`, `wrote_user`) and `assistant_test(model, prompts, conversations, seed=0, max_new_tokens=256) -> list[Reply]`: prompt is `render_chat(history) + [ASSISTANT]`, temperature 0.5, top-k 50, stop on `<|end|>`; `stopped` is true when the reply ended on `<|end|>`, not at the token cap; `wrote_user` is true when the reply contains the `<|user|>` token. Test on a tiny model: 20 + 8 replies for 20 prompts and 8 conversation turns, same seed gives the same text.
- [x] 2.2 Conversations go turn by turn: each later turn's history holds the earlier user turns and the model's own earlier replies. Test: the second turn's prompt ids equal `render_chat` of turn 1, reply 1, turn 2, plus `<|assistant|>`.

## 3. `ullm eval --assistant` writes the report [serial, needs 2]
- [x] 3.1 `ullm/eval.py` `write_assistant_report(path, run, replies)`: summary line (replies that stopped on `<|end|>`, replies that wrote `<|user|>`), then each prompt and each conversation turn with its reply, the two automatic checks, and a `- [ ] pass` checkbox (relevant, coherent). Test: report text for 2 fake replies, one stopped and one at the cap.
- [x] 3.2 `ullm/cli.py` `eval --assistant`: skip benchmarks and the generation test, load `--model`, run `assistant_test`, write `results/<run>/results.md`. Tests: tiny checkpoint writes the report with no benchmark table; `--assistant --model gpt2` is an error.

## 4. The chat model is judged [serial, needs 3]
- [x] 4.1 Mac run (MPS, 28 replies of at most 256 tokens): `uv run ullm eval --model checkpoints/ullm-chat --run ullm-chat --assistant`. `results/ullm-chat/results.md` written; Claude reports the automatic checks.
- [ ] 4.2 User ticks the checkboxes in `results/ullm-chat/results.md`. Claude adds a findings section and a README "At a glance" row, and commits `results/ullm-chat/`.

## Dependencies
- 2 needs 1.
- 3 needs 2.
- 4 needs 3.

## Open questions
- OQ-A1. Mode: `ullm eval --assistant` flag, not auto-detection from the checkpoint (both models share one config format, so there is nothing to detect). Recommended; change before /implement-feature if you prefer another way.
- OQ-A2. One reply per prompt with seed 0 and temperature 0.5 (the `ullm chat` default), not best of 3. The design asks best of 3 only for the base generation test.
