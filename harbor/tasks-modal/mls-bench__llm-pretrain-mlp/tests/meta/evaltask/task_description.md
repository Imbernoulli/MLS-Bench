# LLM Pretraining: Feed-Forward Network Optimization

## Research Question
Design an improved feed-forward (MLP) sublayer for GPT-style language model pretraining. The change should reduce validation loss compared to the standard two-layer GELU MLP with 4× expansion, while remaining a modular feed-forward-only intervention.

## Background
The default MLP is `Linear(d, 4d) → GELU → Linear(4d, d)`. Common alternatives at this layer:

- **GLU variants** — Shazeer, "GLU Variants Improve Transformer", 2020, arXiv:2002.05202. Replace the first linear with two linears whose elementwise product (one passed through a nonlinearity) gates the activations:
  - **GeGLU**: `(GELU(W_g x)) ⊙ (W_v x)`.
  - **SwiGLU**: `(Swish(W_g x)) ⊙ (W_v x)`. Adopted by PaLM, LLaMA, DeepSeek, Qwen, etc.
  When using a GLU variant, the hidden dimension is typically reduced (commonly to ⌊8d/3⌋) so total parameter count stays comparable to the GELU baseline.
- **Squared ReLU (Primer-EZ)** — So et al., "Primer: Searching for Efficient Transformers for Language Modeling", NeurIPS 2021, arXiv:2109.08668. Replace GELU with `(ReLU(x))^2`. Reported as one of the two robust changes from the Primer architecture search.

## What you can modify
The `MLP` class in `nanoGPT/custom_pretrain.py`:
- Activation function (default: GELU).
- Network architecture (default: two linear layers with 4× expansion).
- Gating mechanisms (e.g., SwiGLU / GeGLU).
- Hidden dimension sizing (with the parameter-count caveat above).

### Interface contract
- The MLP must accept input of shape `(B, T, n_embd)` and return output of the same shape.
- Do not depend on changes to attention, normalization, the dataset, optimizer schedule, or evaluation scripts.

## Reference baselines
- `swiglu` — SwiGLU with reduced hidden width.
- `geglu` — GeGLU with reduced hidden width.
- `relu_squared` — Primer-EZ squared-ReLU activation.

## Fixed Pipeline
- **Model**: GPT-2 Medium (24 layers, 16 heads, d=1024, ~355M params).
- **Dataset**: FineWeb 10B (HuggingFace `HuggingFaceFW/fineweb` `sample-10BT`), GPT-2 tokenizer, ~7.1B training tokens.
- **Training**: 12,030 iterations, micro-batch 96, gradient accumulation 6, 2-GPU DDP.

## Evaluation
- **Validation loss** — cross-entropy on FineWeb (lower is better, primary).
- **Perplexity** — WikiText-2, LAMBADA (lower is better).
- **Downstream accuracy** — ARC-Easy, HellaSwag, PIQA, WinoGrande (higher is better).
