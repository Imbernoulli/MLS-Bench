# LLM Pretraining: Normalization & Block Architecture Optimization

## Research Question
Design improved normalization and / or transformer block structure for GPT-style language model pretraining. The change should reduce validation loss compared to the standard `LayerNorm` (with bias) in a Pre-LN block.

## Background
The default architecture uses LayerNorm with affine parameters in a pre-normalization block (`x + Attn(LN(x))`, then `x + MLP(LN(x))`). Common modifications at this layer:

- **RMSNorm** — Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019, arXiv:1910.07467. Drops re-centering: `RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma`. Cheaper and used by LLaMA / PaLM / Gemma / Qwen.
- **Post-LN vs Pre-LN** — Xiong et al., "On Layer Normalization in the Transformer Architecture", ICML 2020, arXiv:2002.04745. Pre-LN gives well-behaved gradients at initialization but typically gives slightly worse final loss than Post-LN when both train successfully.
- **Parallel attention + MLP block** — used by GPT-J (Wang & Komatsuzaki, 2021) and PaLM (Chowdhery et al., 2022, arXiv:2204.02311): compute attention and MLP in parallel from the same normalized input, `x + Attn(LN(x)) + MLP(LN(x))`. Reduces sequential depth and is reportedly ~15% faster at large scale; small quality loss at small scale, no degradation at PaLM-62B scale.

## What you can modify
Two regions in `nanoGPT/custom_pretrain.py`:

1. **`LayerNorm` class** — the normalization implementation (default: LayerNorm with bias).
2. **`Block` class** — how attention and MLP are composed with residual connections.

Specifically, you may modify:
- The normalization rule (e.g., LayerNorm → RMSNorm) and affine parameters.
- Where normalization is applied (Pre-LN, Post-LN, or other placements; e.g., adding QK norms, output norms).
- The residual-connection structure.
- How attention and MLP sublayers are combined (sequential vs. parallel).

Changes must remain local to block structure and must not alter the dataset, tokenizer, training loop, optimizer schedule, or evaluation pipeline.

## Reference baselines
- `rmsnorm` — replace LayerNorm with RMSNorm, otherwise Pre-LN.
- `rmsnorm_post` — RMSNorm in a Post-LN block.
- `rmsnorm_parallel` — RMSNorm with the parallel attention+MLP block (GPT-J / PaLM style).

## Fixed Pipeline
- **Model**: GPT-2 Medium (24 layers, 16 heads, d=1024, ~355M params).
- **Dataset**: FineWeb 10B (HuggingFace `HuggingFaceFW/fineweb` `sample-10BT`), GPT-2 tokenizer, ~7.1B training tokens.
- **Training**: 12,030 iterations, micro-batch 96, gradient accumulation 6, 2-GPU DDP.

## Evaluation
- **Validation loss** — cross-entropy on FineWeb (lower is better, primary).
- **Perplexity** — WikiText-2, LAMBADA (lower is better).
- **Downstream accuracy** — ARC-Easy, HellaSwag, PIQA, WinoGrande (higher is better).
