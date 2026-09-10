# LLM Pretraining: Embedding Strategy Optimization

## Research Question
Design an improved embedding strategy for GPT-style language model pretraining. The change should reduce validation loss compared to standard learned token + position embeddings with weight tying, while remaining a modular embedding-level intervention.

## Background
The default scheme uses:
- Learned token embedding (`wte`) of shape `(vocab_size, n_embd)`.
- Learned absolute position embedding (`wpe`) of shape `(block_size, n_embd)`.
- **Tied weights** between the input token embedding and the output `lm_head` projection (Press & Wolf, "Using the Output Embedding to Improve Language Models", 2016/2017, arXiv:1608.05859).

Common alternatives studied at this layer:
- Untied input/output embeddings.
- Hash-based / bigram / n-gram embeddings to inject sub-token co-occurrence statistics.
- **Value embeddings** (popularized in the modded-nanogpt speedrun, originally inspired by Zhou et al., 2024): a separate embedding table whose output is added to the *value* projections inside attention layers — typically gated and inserted at a few specific layers.

## What you can modify
The `TokenEmbedding` class in `nanoGPT/custom_pretrain.py`:
- Token embedding representation (default: learned token + position embeddings).
- Weight-tying strategy (default: input embedding shares weights with output `lm_head`).
- Additional embedding sources (e.g., n-gram, hash-based).
- Per-layer value embeddings injected via `get_value_embed(layer_idx)`.

### Interface
Your `TokenEmbedding` class must implement:
- `forward(idx) -> x` — takes token indices `(B, T)`, returns embeddings `(B, T, n_embd)`.
- `get_lm_head_weight() -> Tensor` — returns the weight tensor used for the output projection.
- `get_num_pos_params() -> int` — returns the count of position parameters (excluded from the reported parameter count).
- `get_value_embed(layer_idx) -> Optional[Tensor]` — optional per-layer value-embedding residual `(B, T, n_embd)` or `None`.

## Reference baselines
- `untied` — break weight tying between input embedding and `lm_head`.
- `bigram_hash` — hash-based bigram embeddings additive to the token embedding.
- `value_embed` — value-style per-layer embedding injection.

## Fixed Pipeline
- **Model**: GPT-2 Medium (24 layers, 16 heads, d=1024, ~355M params).
- **Dataset**: FineWeb 10B (HuggingFace `HuggingFaceFW/fineweb` `sample-10BT`), GPT-2 tokenizer, ~7.1B training tokens.
- **Training**: 12,030 iterations, micro-batch 96, gradient accumulation 6, 2-GPU DDP.
- The corpus, tokenizer, training loop, optimizer, and unrelated transformer blocks are fixed.
- The benchmark's parameter accounting excludes `get_num_pos_params()` from the reported count, so simply scaling capacity through positional parameters is not a valid escape.

## Evaluation
- **Validation loss** — cross-entropy on FineWeb (lower is better, primary).
- **Perplexity** — WikiText-2, LAMBADA (lower is better).
- **Downstream accuracy** — ARC-Easy, HellaSwag, PIQA, WinoGrande (higher is better).
