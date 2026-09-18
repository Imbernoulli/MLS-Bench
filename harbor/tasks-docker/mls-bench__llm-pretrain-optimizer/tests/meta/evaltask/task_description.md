# LLM Pretraining: Optimizer Design

## Research Question
Design an improved optimizer for GPT-style language model pretraining: the parameter update rule, the state it keeps, and how parameters are grouped and treated. The change should reduce validation loss compared to AdamW under the same model, data, step budget, and learning-rate schedule.

This is an optimizer task, not a schedule task. The learning-rate schedule (linear warmup → cosine decay to `min_lr`) is fixed code that you cannot edit. Every iteration the training loop computes the scheduled `lr` and writes it into each `param_group` of whatever optimizer you return; what you control is how that scheduled step size is turned into a parameter update.

## Background
The default optimizer is AdamW (fused) with weight decay only on 2D parameters, driven by the fixed cosine schedule with linear warmup. Studied alternatives at this layer:

- **Lion** — Chen et al., "Symbolic Discovery of Optimization Algorithms", NeurIPS 2023, arXiv:2302.06675. Sign-momentum optimizer found via program search; tracks only momentum, applies a uniform-magnitude `sign(...)` update; typically uses LR ≈ 0.1× AdamW LR and stronger weight decay.
- **Muon** — Keller Jordan et al. (2024), "Muon: An optimizer for hidden layers in neural networks" (https://kellerjordan.github.io/posts/muon/). Applies SGD-momentum, then orthogonalizes the resulting matrix update via a 5-step Newton–Schulz iteration; intended for 2D hidden-layer matrices, with AdamW kept for embeddings / `lm_head` / 1D parameters. ~35% training-speed improvement reported on the NanoGPT speedrun versus AdamW.
- **AdamW + Nesterov momentum** — straightforward variant adding Nesterov-style lookahead to Adam's first moment.

## What you can modify
Two regions in `nanoGPT/custom_pretrain.py`:

1. **`configure_optimizers` method** — everything about the optimizer: the update rule (define any custom `torch.optim.Optimizer` subclass, or a combination of optimizers, inside this method), parameter grouping (default: weight decay on ≥2D params, none on 1D params), per-group treatment (e.g. one rule for hidden matrices and another for embeddings / `lm_head` / 1D params), and optimizer hyperparameters (betas, eps, momentum, weight decay, …).
2. **`CONFIG_OVERRIDES` dict** — scalar training hyperparameters your optimizer needs set differently from the AdamW defaults. Allowed keys: `learning_rate` (peak LR; setting it also resets `min_lr` to `learning_rate / 10`, so put `min_lr` after it if you need both), `weight_decay`, `warmup_iters`, `min_lr`, `grad_clip` (0 disables clipping). Other keys are ignored.

Not editable: the `get_lr` function. The schedule's *shape* (linear warmup → cosine decay) is fixed; only its scalar knobs (peak LR, warmup length, floor) can move, through `CONFIG_OVERRIDES`. Everything outside the two regions — architecture, tokenizer, dataset, batch construction, training loop, evaluation — is fixed, and no new files may be created.

### Interface contract
- `configure_optimizers(self, weight_decay, learning_rate, betas, device_type)` — keep this signature; it receives the config values after `CONFIG_OVERRIDES` is applied.
- The returned object must support `.zero_grad()`, `.step()`, and `.param_groups`, where each `param_group` is a dict holding a `'params'` list. The training loop sets `param_group['lr'] = lr * param_group.get('lr_scale', 1.0)` every iteration, so a group that needs a different step size than the scheduled one (e.g. Muon's much larger LR) should carry an `'lr_scale'` key rather than ignore `'lr'`.
- Gradient clipping (`clip_grad_norm_` to `grad_clip`) runs in the training loop before `.step()`; you can change its threshold via `CONFIG_OVERRIDES` but not move it into the optimizer.

## Reference baselines
- `lion` — Lion (sign of the interpolated momentum, decoupled weight decay), same grouping, driven at the same scheduled learning rate as AdamW (no `lr_scale`), under the fixed cosine schedule.
- `muon` — Muon (Nesterov momentum + 5-step Newton–Schulz orthogonalization, weight decay 0.1) for 2D hidden weights, AdamW for embeddings / `lm_head` / 1D params; Muon's base LR 0.02 is expressed through `lr_scale`, and the AdamW peak LR is raised to 1e-3 through `CONFIG_OVERRIDES`.
- `adamw_nesterov` — AdamW with Nesterov momentum (PyTorch `NAdam` with decoupled weight decay), same grouping and LR.

## Fixed Pipeline
- **Model**: GPT-2 Medium (24 layers, 16 heads, d=1024, ~355M params).
- **Dataset**: FineWeb 10B (HuggingFace `HuggingFaceFW/fineweb` `sample-10BT`), GPT-2 tokenizer, ~7.1B training tokens.
- **Training**: 12,030 iterations, micro-batch 96, gradient accumulation 6, 2-GPU DDP; linear warmup → cosine decay (fixed `get_lr`).

## Evaluation
- **Validation loss** — cross-entropy on FineWeb (lower is better, primary).
- **Perplexity** — WikiText-2, LAMBADA (lower is better).
- **Downstream accuracy** — ARC-Easy, HellaSwag, PIQA, WinoGrande (higher is better).
