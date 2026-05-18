# LLM Pretraining: Linear / Subquadratic Attention Mechanism

## Research Question
Design a linear or otherwise subquadratic sequence-mixing mechanism for GPT-style language model pretraining that remains competitive in language-model quality with standard quadratic softmax attention. The mechanism should scale better than O(n²) in sequence length.

## Background
Standard transformer attention has O(n²) compute and memory in sequence length. A growing body of work proposes subquadratic alternatives that retain transformer-level quality on language modeling:

- **RetNet** — Sun et al., 2023, arXiv:2307.08621, "Retentive Network: A Successor to Transformer for Large Language Models". Retention with parallel / recurrent / chunkwise-recurrent dual forms.
- **GLA** (Gated Linear Attention) — Yang et al., 2023, arXiv:2312.06635, "Gated Linear Attention Transformers with Hardware-Efficient Training". Data-dependent gating + FlashLinearAttention kernels.
- **Mamba** — Gu & Dao, 2023, arXiv:2312.00752, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces". Selective SSM with input-dependent parameters.
- **RWKV-6 (Finch)** — Peng et al., 2024, arXiv:2404.05892, "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence". Multi-headed matrix-valued state, dynamic recurrence.
- **DeltaNet** — Yang et al., 2024, arXiv:2406.06484, "Parallelizing Linear Transformers with the Delta Rule over Sequence Length". Delta-rule update with hardware-efficient parallel training.

## What you can modify
Two editable regions in `nanoGPT/custom_pretrain.py`:

1. **`CausalSelfAttention` class** — the attention computation itself, including:
   - Replacing softmax attention with linear / subquadratic alternatives.
   - Feature maps, gating mechanisms, decay factors.
   - Q/K/V projections and transformations.
   - Internal state management (recurrent state, convolutions, etc.).

2. **`Block` class** — the transformer block structure, including:
   - How attention and MLP sublayers are composed.
   - Normalization placement (pre-norm, post-norm).
   - Residual-connection patterns required to make the mechanism train stably.

### Tooling notes
- The `flash-linear-attention` (FLA) library is pre-installed and provides 27+ optimized linear-attention layers with Triton kernels (`fla.layers.GatedLinearAttention`, `DeltaNet`, `MultiScaleRetention`, `LinearAttention`, `HGRN2`, `Mamba2`, …). You may import from FLA or implement your own mechanism from scratch.
- If your attention does not use learned absolute position embeddings, set `self.use_pos_emb = False` in `__init__`; the model then skips adding `wpe` in the forward pass.
- `torch.compile` is disabled for this task because FLA's Triton kernels are not compatible with it.

## Reference baselines
- `gla` — Gated Linear Attention.
- `retnet` — Retentive Network / MultiScaleRetention.
- `deltanet` — DeltaNet (delta-rule linear attention).

## Fixed Pipeline
- **Model**: GPT-2 Medium (24 layers, 16 heads, d=1024, ~355M params).
- **Dataset**: FineWeb 10B (HuggingFace `HuggingFaceFW/fineweb` `sample-10BT`), GPT-2 tokenizer, ~7.1B training tokens.
- **Training**: 13,535 iterations, micro-batch 32, gradient accumulation 16, 2-GPU DDP.
- Dataset, tokenizer, training schedule, evaluation code, and unrelated objectives are out of scope.

## Evaluation
- **Validation loss** — cross-entropy on FineWeb (lower is better, primary).
- **Perplexity** — WikiText-2, LAMBADA (lower is better).
- **Downstream accuracy** — ARC-Easy, HellaSwag, PIQA, WinoGrande (higher is better).
