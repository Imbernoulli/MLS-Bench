# LLM Pretraining: Native Low-Bit Linear (BitLinear)

## Research Question
Design a low-bit linear layer for GPT-2 pretraining that uses native low-precision weights (binary / ternary / few-bit) during both training and inference, instead of standard float weights. The goal is to minimize validation loss and preserve downstream language ability while constraining the effective forward weights to a small discrete set.

## Background
Standard neural networks store and compute with full-precision (FP32 / BF16) weights. Post-training quantization (PTQ) and quantization-aware training (QAT) compress these weights after or during training, but the model fundamentally trains with float weights. Native low-bit training takes a different approach: weights are inherently discrete (e.g., {-1, +1} or {-1, 0, +1}) during every forward pass, while a float latent weight is maintained only for gradient accumulation (with a straight-through estimator).

Reference papers:
- **BitNet** — Wang et al., 2023, arXiv:2310.11453, "BitNet: Scaling 1-bit Transformers for Large Language Models". Introduces `BitLinear` as a drop-in replacement for `nn.Linear`, binarizing weights to {-1, +1} via the sign function with per-tensor scale.
- **BitNet b1.58** — Ma et al., 2024, arXiv:2402.17764, "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits". Ternary weights {-1, 0, +1} via absmean quantization (`scale = mean(|W|)`, weights rounded to the nearest of {-1, 0, +1}). Reported to match full-precision LLaMA-style baselines starting around the 3B scale.

Distinction from neighboring tasks:
- **vs. QAT**: QAT keeps float weights during training and only uses fake quantization; BitLinear's forward weights are always discrete.
- **vs. mixed precision**: Mixed precision changes the float format (FP32 → BF16/FP8) but values remain continuous; BitLinear restricts weights to a small discrete set (1–2 bits typically).

## What you can modify
The BitLinear module in `nanoGPT/custom_pretrain.py`:
- `weight_quant(weight)` — quantizes float latent weights to discrete values; returns `(quantized_weight, scale)`.
- `activation_quant(x)` — optional activation quantization; returns `(quantized_x, scale)`.
- `BitLinear` class — linear layer that uses the above functions.

### Interface contract
- `BitLinear.__init__(self, in_features, out_features, bias=True)` must keep `self.weight` as a `Parameter`.
- `BitLinear.forward(self, x) -> output` where `x` has shape `(..., in_features)` and the output has shape `(..., out_features)`.
- Quantization is applied in every forward pass (no separate train/eval path).
- `weight_quant` should return `(quantized_weight, scale)` such that `quantized_weight * scale` approximates the original weight; same convention for `activation_quant`.
- All linear projections in the model (attention, MLP, lm_head) use `BitLinear`.
- Helper classes (`autograd.Function`s, learned parameters) may be added.
- Must remain compatible with `torch.compile` (no `@torch.compiler.disable`).

## Reference baselines (algorithmic templates)
- `binary_sign` — BitNet sign-based binary weights {-1, +1} with absmean scale.
- `ternary_158bit` — BitNet b1.58 ternary {-1, 0, +1} with absmean scale.
- `int2_uniform` — uniform 2-bit quantization grid.

## Fixed Pipeline
- **Model**: GPT-2 Medium (24 layers, 16 heads, d=1024, ~355M params).
- **Dataset**: FineWeb 10B (HuggingFace `HuggingFaceFW/fineweb` `sample-10BT`), GPT-2 tokenizer, ~7.1B training tokens (Chinchilla-optimal D=20N).
- **Training**: 13,535 iterations, micro-batch 64, gradient accumulation 8, 2-GPU DDP.

## Evaluation
- **Validation loss** — cross-entropy on a held-out FineWeb shard (lower is better, primary).
- **Perplexity** — WikiText-2, LAMBADA (lower is better).
- **Downstream accuracy** — ARC-Easy, HellaSwag, PIQA, WinoGrande (higher is better).
