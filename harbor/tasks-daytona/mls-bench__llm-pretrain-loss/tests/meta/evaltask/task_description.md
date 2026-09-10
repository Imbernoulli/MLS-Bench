# LLM Pretraining: Loss Function Optimization

## Research Question
Design an improved loss function for GPT-2 next-token language model pretraining. The change should reduce validation loss and improve downstream language ability under the same architecture, data, and optimization budget, compared to standard cross-entropy.

## Background
The default objective is plain next-token cross-entropy. Several modifications have been studied as drop-in replacements at this layer:

- **Label smoothing** — Szegedy et al., "Rethinking the Inception Architecture for Computer Vision", 2015, arXiv:1512.00567 (Section 7). Replaces hard one-hot targets with `(1-eps) * onehot + eps / V` (default eps ≈ 0.1).
- **Logit z-loss** — auxiliary penalty `lambda * (logsumexp(logits))^2` to keep logit magnitudes small. Originally from the Mesh-TensorFlow softmax z-loss; popularized by ST-MoE / PaLM (Zoph et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models", 2022, arXiv:2202.08906). Typical coefficient `lambda ≈ 1e-4`.
- **Logit soft-capping** — `softcap * tanh(logits / softcap)` applied before the softmax, used in Gemma 2 (Gemma Team, "Gemma 2: Improving Open Language Models at a Practical Size", 2024, arXiv:2408.00118), with attention logits capped at 50.0 and final logits capped at 30.0.

## What you can modify
The `compute_loss` function in `nanoGPT/custom_pretrain.py`:
- Loss formulation (default: standard cross-entropy).
- Logit processing (e.g., soft-capping, temperature scaling).
- Regularization terms (e.g., z-loss, entropy penalties).
- Label-distribution modifications (e.g., label smoothing).

### Interface contract
- Signature must remain `compute_loss(logits, targets)`.
- `logits` shape `(B, T, V)`; `targets` shape `(B, T)`.
- The function is called inside the model's forward pass during training.
- Stable throughout training; do not lower reported loss by distorting probabilities (e.g., via temperature) without improving the actual modeling distribution.

## Reference baselines
- `label_smoothing` — eps=0.1.
- `z_loss` — lambda=1e-4.
- `softcap_ce` — Gemma-2-style final-logit soft-cap at 30.0.

## Fixed Pipeline
- **Model**: GPT-2 Medium (24 layers, 16 heads, d=1024, ~355M params).
- **Dataset**: FineWeb 10B (HuggingFace `HuggingFaceFW/fineweb` `sample-10BT`), GPT-2 tokenizer, ~7.1B training tokens.
- **Training**: 13,535 iterations, micro-batch 64, gradient accumulation 8, 2-GPU DDP.
- Architecture, tokenizer, dataset, training loop, and evaluation pipeline are fixed.

## Evaluation
- **Validation loss** — cross-entropy on FineWeb (lower is better, primary).
- **Perplexity** — WikiText-2, LAMBADA (lower is better).
- **Downstream accuracy** — ARC-Easy, HellaSwag, PIQA, WinoGrande (higher is better).
