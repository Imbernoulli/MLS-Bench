# LLM Pretraining: Attention Mechanism Optimization

## Research Question
Design an improved self-attention mechanism for GPT-style language model pretraining. The change should reduce validation loss / perplexity and transfer to downstream tasks compared to standard causal multi-head softmax attention with learned absolute position embeddings.

## Background
The default attention in this task is the standard scaled-dot-product causal multi-head attention used in GPT-2, with separate learned absolute position embeddings (`wpe`). Many architectural improvements have been proposed at this single layer of the stack:

- **RoPE** — Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021, arXiv:2104.09864. Encodes absolute position via a rotation matrix applied to Q/K, producing relative position dependence in the dot product. Adopted by LLaMA / GPT-NeoX / Qwen / etc.
- **QK-Norm** — Henry et al., "Query-Key Normalization for Transformers", Findings of EMNLP 2020, arXiv:2010.04245. Applies L2 normalization to query and key vectors along the head dimension before the dot product, then scales by a learnable parameter, replacing the `1/sqrt(d_k)` scaling.

These two ideas are also commonly combined ("RoPE + QK-Norm").

## What you can modify
The `CausalSelfAttention` class in `nanoGPT/custom_pretrain.py` (the editable region around the attention block), including:

- Position encoding scheme (the default uses learned absolute position embeddings via `wpe`).
- Query / Key / Value computation and projection.
- Attention score computation, scaling, and masking.
- Attention-related hyperparameters local to this module.

If your attention mechanism implements its own position encoding (replacing the learned `wpe`), set `self.use_pos_emb = False` in `__init__` — the model will then skip adding position embeddings in the forward pass.

## Fixed Pipeline
- **Model**: GPT-2 Medium (24 layers, 16 heads, d=1024, ~355M params).
- **Dataset**: FineWeb 10B (Penedo et al., 2024, arXiv:2406.17557; HuggingFace `HuggingFaceFW/fineweb` `sample-10BT`), GPT-2 tokenizer, ~7.1B training tokens (D = 20 N, Chinchilla-optimal for 355M).
- **Training**: 13,535 iterations, micro-batch 64, gradient accumulation 8, 2-GPU DDP.
- Optimizer, schedule, dataset, tokenizer, training loop, and evaluation scripts are fixed.

## Evaluation
- **Validation loss** — cross-entropy on a held-out FineWeb shard (lower is better).
- **Perplexity** — WikiText-2 and LAMBADA (lower is better).
- **Downstream accuracy** — ARC-Easy, HellaSwag, PIQA, WinoGrande (higher is better) via the LM Evaluation Harness.

A strong solution should reduce validation loss / perplexity and transfer to downstream accuracy without depending on changes outside the attention module.
