# LLM Pretraining: Residual Connection Strategy

## Research Question
Improve the residual-connection strategy of a GPT-style language model. The default uses standard Pre-LN additive residuals (`x = x + sublayer(LN(x))`) in each transformer block. The goal is to redesign how information flows through the residual stream across layers to lower validation loss.

## Background

### Standard residual stream
The default GPT-2 block:
```python
x = x + self.attn(self.ln_1(x))   # attention sublayer
x = x + self.mlp(self.ln_2(x))    # MLP sublayer
```
The residual stream is the model's information highway; its design affects gradient flow, feature reuse, and training stability.

### Research directions
- **Per-layer residual scaling** — learnable scalars that gate the contribution of each sublayer. Examples: ReZero (Bachlechner et al., "ReZero is All You Need: Fast Convergence at Large Depth", 2020, arXiv:2003.04887), SkipInit, and the per-layer scalar gates used in modded-nanogpt.
- **Initial-embedding (x0) blending** — re-blend the token embedding back into the residual at each layer to preserve token identity (used in modded-nanogpt and related speedrun work).
- **Hyper-Connections** — Zhu et al. (ByteDance), "Hyper-Connections", ICLR 2025, arXiv:2409.19606. Maintain `m` parallel residual streams with learned transition matrices, addressing the gradient-vanishing / representation-collapse seesaw of vanilla residuals.
- **Attention-over-layers residuals** — softmax attention over all previous layer outputs to dynamically pick which past representations to combine, a recurring idea in recent open-source LM work.

## What you can modify
In `nanoGPT/custom_pretrain.py`:

- **`Block` class** — per-block residual behavior; how attention and MLP outputs are combined with the residual stream within each block.
- **`GPT.__init__`** — additional parameters for your residual strategy (per-layer scalars, transition matrices, query vectors, etc.).
- **The block loop in `GPT.forward`** — how blocks are called and how their outputs are accumulated (e.g., multi-stream processing, attention over layer outputs).
- **`configure_optimizers`** — assign new parameters to optimizer groups with appropriate LR / weight decay.
- **`CONFIG_OVERRIDES`** dict — adjust LR / weight decay if your design needs it.

### Interface contract
- `CausalSelfAttention`, `MLP`, `LayerNorm`, and `GPTConfig` are fixed.
- `Block.forward` must accept `x` and return a tensor of the same shape.
- `GPT.forward` must accept `(idx, targets=None)` and return `(logits, loss)`.

## Reference baselines
- `vanilla` — standard additive Pre-LN residuals (the default).
- `learned_scaling` — ReZero-style per-layer learnable scalar on each sublayer.
- `prores` — initial-embedding blending into the residual stream.
- `full_attnres` — attention over all previous layer outputs.

## Fixed Pipeline
- **Model**: GPT-2 Medium (24 layers, 16 heads, d=1024, ~355M params).
- **Dataset**: FineWeb 10B (HuggingFace `HuggingFaceFW/fineweb` `sample-10BT`), GPT-2 tokenizer, ~7.1B training tokens.
- **Training**: 13,535 iterations, micro-batch 32, gradient accumulation 16, 2-GPU DDP.

## Evaluation
- **Validation loss** — cross-entropy on FineWeb (lower is better, primary).
- **Perplexity** — WikiText-2, LAMBADA (lower is better).
- **Downstream accuracy** — ARC-Easy, HellaSwag, PIQA, WinoGrande (higher is better).
