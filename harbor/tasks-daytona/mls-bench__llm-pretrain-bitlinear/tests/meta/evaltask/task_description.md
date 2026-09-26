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

The default template is a naive binary quantizer: `sign(W)` with a per-tensor absmean scale and no activation quantization.

### Interface contract
- `BitLinear.__init__(self, in_features, out_features, bias=True)` must keep `self.weight` as a `Parameter`.
- `BitLinear.forward(self, x) -> output` where `x` has shape `(..., in_features)` and the output has shape `(..., out_features)`.
- Quantization is applied in every forward pass (no separate train/eval path).
- **Level budget (enforced): at most 5 weight levels.** In every output row of the effective weight a `BitLinear` applies in its forward pass, at most 5 distinct values may occur. This admits binary, ternary and the 5-level grid {-1, -2/3, 0, 2/3, 1} that the `int2_uniform` reference rounds to; a per-tensor or per-output-channel scale is allowed. Fixed code recovers each `BitLinear`'s effective weight by probing it with one-hot inputs under the run's bf16 autocast at initialization, at every evaluation interval and before the final evaluation, in both train and eval mode, and aborts the run if any row exceeds 5 levels; such a run gets no score. Randomly chosen forward calls of every evaluation (validation loss and perplexity) are also checked in place: the real output must match that effective weight applied to the activation the layer feeds its matmul (read back from the same layer on the same input, with the signs of its latent weight randomly flipped) within 2 %, so the forward pass must apply the discrete weight to real inputs as well; a mismatch also aborts the run. Transforms that mix input features (e.g. rotations) and per-group scales make the effective weight continuous, so they are not allowed.
- `weight_quant` should return `(quantized_weight, scale)` such that `quantized_weight * scale` approximates the original weight; same convention for `activation_quant`.
- All linear projections in the model (attention, MLP, lm_head) use `BitLinear`.
- Helper classes (`autograd.Function`s, learned parameters) may be added.
- Must remain compatible with `torch.compile` (no `@torch.compiler.disable`).

## Reference baselines (algorithmic templates)
- `binary_sign` — BitNet sign-based binary weights {-1, +1} with absmean scale.
- `ternary_158bit` — BitNet b1.58 ternary {-1, 0, +1} with absmean scale.
- `int2_uniform` — named for a uniform 2-bit grid, but as implemented it rounds to the 5 levels {-1, -2/3, 0, 2/3, 1} with absmean scale.

## Fixed Pipeline
- **Model**: GPT-2 Medium (24 layers, 16 heads, d=1024, ~355M params).
- **Dataset**: FineWeb 10B (HuggingFace `HuggingFaceFW/fineweb` `sample-10BT`), GPT-2 tokenizer, ~7.1B training tokens (Chinchilla-optimal D=20N).
- **Training**: 13,535 iterations, micro-batch 64, gradient accumulation 8, 2-GPU DDP.

## Evaluation
- **Validation loss** — cross-entropy on a held-out FineWeb shard (lower is better, primary).
- **Perplexity** — WikiText-2, LAMBADA (lower is better).
- **Downstream accuracy** — ARC-Easy, HellaSwag, PIQA, WinoGrande (higher is better).
