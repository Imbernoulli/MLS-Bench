# LLM Pretraining: Optimizer & Learning Rate Schedule Optimization

## Research Question
Design an improved optimizer and / or learning-rate schedule for GPT-style language model pretraining. The change should reduce validation loss compared to AdamW + cosine annealing under the same model and data budget.

## Background
The default optimizer is AdamW (fused) with weight decay only on 2D parameters and cosine LR decay with linear warmup. Studied alternatives at this layer:

- **Lion** — Chen et al., "Symbolic Discovery of Optimization Algorithms", NeurIPS 2023, arXiv:2302.06675. Sign-momentum optimizer found via program search; tracks only momentum, applies a uniform-magnitude `sign(...)` update; typically uses LR ≈ 0.1× AdamW LR and stronger weight decay.
- **Muon** — Keller Jordan et al. (2024), "Muon: An optimizer for hidden layers in neural networks" (https://kellerjordan.github.io/posts/muon/). Applies SGD-momentum, then orthogonalizes the resulting matrix update via a 5-step Newton–Schulz iteration; intended for 2D hidden-layer matrices, with AdamW kept for embeddings / `lm_head` / 1D parameters. ~35% training-speed improvement reported on the NanoGPT speedrun versus AdamW.
- **AdamW + Nesterov momentum** — straightforward variant adding Nesterov-style lookahead to Adam's first moment.

## What you can modify
Two regions in `nanoGPT/custom_pretrain.py`:

1. **`configure_optimizers` method** — optimizer creation and parameter grouping.
2. **`get_lr` function** — learning-rate schedule.

You may modify:
- The optimization algorithm (default: AdamW fused).
- Parameter grouping strategy (default: weight decay for 2D params, none for 1D params).
- LR schedule shape (default: cosine with linear warmup).
- Any optimizer hyperparameters (betas, eps, weight decay, etc.).

### Interface contract
- `get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr)` — keep this signature.
- The optimizer returned by `configure_optimizers` must support `.zero_grad()`, `.step()`, and `.param_groups`.
- Architecture, tokenizer, dataset, batch construction, and evaluation are fixed.

## Reference baselines
- `lion` — Lion optimizer with cosine schedule.
- `muon` — Muon for 2D hidden weights + AdamW for the rest.
- `adamw_nesterov` — AdamW with Nesterov momentum.

## Fixed Pipeline
- **Model**: GPT-2 Medium (24 layers, 16 heads, d=1024, ~355M params).
- **Dataset**: FineWeb 10B (HuggingFace `HuggingFaceFW/fineweb` `sample-10BT`), GPT-2 tokenizer, ~7.1B training tokens.
- **Training**: 12,030 iterations, micro-batch 96, gradient accumulation 6, 2-GPU DDP.

## Evaluation
- **Validation loss** — cross-entropy on FineWeb (lower is better, primary).
- **Perplexity** — WikiText-2, LAMBADA (lower is better).
- **Downstream accuracy** — ARC-Easy, HellaSwag, PIQA, WinoGrande (higher is better).
