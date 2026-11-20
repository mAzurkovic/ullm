# M1 F1. Model — Tasks

Scope: uv project, `ullm/config.py`, `ullm/model.py`, shape tests. Source: `design.md` (Model, Components). No tokenizer, checkpoint I/O, or CLI (M1 F2).

## 1. Project runs tests [serial]
- [ ] 1.1 `pyproject.toml`: Python 3.12, deps `torch`, dev dep `pytest`, package `ullm`. `uv sync` succeeds.
- [ ] 1.2 Commit `uv.lock` and empty `ullm/__init__.py`. `uv run pytest` runs with 0 errors.
- [ ] 1.3 `ullm/config.py`: `ModelConfig` dataclass, defaults from design Model table (24 layers, d_model 1024, 16 heads, MLP hidden 2816, context 1024, vocab 50,304). Importable.

## 2. GPT forward pass [serial, needs 1]
- [ ] 2.1 `RMSNorm` in `ullm/model.py`: learned scale, no bias. Output RMS ≈ 1 on random input.
- [ ] 2.2 RoPE: precomputed cos/sin for context length, applied to q and k. Shapes preserved.
- [ ] 2.3 `Attention`: q, k, v, out projections, no bias, RoPE, causal `scaled_dot_product_attention`.
- [ ] 2.4 `SwiGLU` MLP and `Block` (pre-norm residual: attention, then MLP).
- [ ] 2.5 `GPT`: token embedding, 24 blocks, final RMSNorm, head tied to embedding, GPT-2 init. `forward(idx) -> logits`.
- [ ] 2.6 `tests/test_model.py`: tiny config logits shape `(B, T, vocab)`; full config ~350M params (340M to 370M, tied counted once); causality (changing token t leaves logits before t unchanged). Tests pass on CPU.

## Dependencies
- 2 needs 1.
