# LLM Pretraining: Learning Rate Schedule Optimization

## Research Question
Design an improved learning-rate schedule for GPT-style language model pretraining. The change should reduce validation loss compared to standard cosine annealing with linear warmup, under the same model, data, optimizer, and total update budget.

## Background
The default schedule is **linear warmup → cosine decay to a small `min_lr`**. Several alternative shapes have been studied as drop-in replacements at the schedule layer:

- **WSD (Warmup–Stable–Decay)** — Hu et al., "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies", 2024, arXiv:2404.06395. Linear warmup → long stable phase at peak LR → final decay phase (often a fast 1−sqrt or exponential decay). Designed for continuous training and to enable mid-training checkpoint reuse.
- **Trapezoidal** — Hägele et al., "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations", 2024, arXiv:2405.18392. Equivalent in shape to WSD: warmup → constant LR → linear cooldown to zero in a short final phase. Removes the need to commit to a fixed step budget upfront and matches cosine quality.
- **WSD with sqrt decay** — variant where the decay phase follows `1 − sqrt(progress)`.

## What you can modify
The `get_lr` function in `nanoGPT/custom_pretrain.py`:
- Schedule shape (default: cosine decay with linear warmup).
- Warmup strategy and duration.
- Decay behavior (shape, rate, final LR).
- Multi-phase scheduling (e.g., warmup-stable-decay, restarts).

### Interface contract
- Signature must remain `get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr)`.
- The training loop calls this at every iteration to set the LR.
- Total update budget is fixed; do not extend the number of training steps.

## Reference baselines
- `wsd` — warmup-stable-decay with the MiniCPM-style decay tail.
- `trapezoidal` — Hägele-style trapezoidal schedule (linear cooldown).
- `wsd_sqrt` — WSD with `1 − sqrt(progress)` decay.

## Fixed Pipeline
- **Model**: GPT-2 Medium (24 layers, 16 heads, d=1024, ~355M params).
- **Dataset**: FineWeb 10B (HuggingFace `HuggingFaceFW/fineweb` `sample-10BT`), GPT-2 tokenizer, ~7.1B training tokens.
- **Training**: 12,030 iterations, micro-batch 96, gradient accumulation 6, 2-GPU DDP.
- Architecture, dataset, optimizer implementation, batch construction, and evaluation are fixed.

## Evaluation
- **Validation loss** — cross-entropy on FineWeb (lower is better, primary).
- **Perplexity** — WikiText-2, LAMBADA (lower is better).
- **Downstream accuracy** — ARC-Easy, HellaSwag, PIQA, WinoGrande (higher is better).
