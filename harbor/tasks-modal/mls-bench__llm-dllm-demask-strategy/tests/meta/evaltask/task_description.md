# Masked Diffusion LM: Demasking Strategy

## Research Question
Design a better demasking (decoding) strategy for masked diffusion language models. The strategy must generalize across **different decoding regimes**:

- **Block-based semi-autoregressive decoding** for downstream-task accuracy (LLaDA on MATH/HumanEval, following the KLASS protocol).
- **Fully-parallel decoding** for open-ended text generation (Dream on prefix-conditioned C4 continuation, measured by perplexity / diversity).

## Background
Masked diffusion LMs generate by starting from a fully masked generation region and iteratively unmasking over `steps` denoising iterations. A demasking strategy decides at each step:

1. **Schedule**: how many tokens to unmask.
2. **Position selection**: which masked positions to unmask.
3. **Token assignment**: what token id to place.

Decoding can be **semi-autoregressive** (when `block_length < gen_length`, process one block at a time) or **fully parallel** (`block_length == gen_length`, all positions decoded together).

Reference papers:
- LLaDA (Nie et al., 2025; arXiv:2502.09992) — "Large Language Diffusion Models"; introduces LLaDA-8B-Base / LLaDA-8B-Instruct.
- Dream 7B (Ye, Xie, et al., 2025; arXiv:2508.15487) — "Dream 7B: Diffusion Large Language Models"; supports arbitrary-order generation and tunable quality–speed trade-offs.
- KLASS (Kim et al., NeurIPS 2025 Spotlight; arXiv:2511.05664) — "KLASS: KL-Guided Fast Inference in Masked Diffusion Models"; KL-adaptive stability sampling for unmasking multiple tokens per step.

## Fixed Pipeline
- Pretrained models (LLaDA-8B-Instruct, Dream-v0-Instruct-7B), prompts, evaluation data, and task runners are fixed.
- Block scheduling constraint: `gen_length % block_length == 0`. When equal, decoding is fully parallel.
- Blocks are processed sequentially (no early-decoding into later blocks).
- The same `DemaskDecoder` must work in both semi-autoregressive and fully-parallel regimes.

## What you can modify
The `DemaskDecoder` class in `LLaDA/custom_demask_eval.py`.

### Interface
```python
class DemaskDecoder:
    def __init__(self, mask_id, temperature=0.0,
                 conf_threshold=0.9, kl_threshold=0.01, history_length=2):
        ...

    @torch.no_grad()
    def decode(self, model, input_ids, gen_length, steps, block_length):
        # Returns (x_output [1, prompt_len + gen_length], used_steps)
```

`get_num_transfer_tokens(mask, steps)` is available outside the editable region — it returns the uniform schedule (`mask.sum() // steps` per step). Always return shape `[1, prompt_len + gen_length]`. `used_steps` counts model forward passes (lower = more efficient).

## Reference baseline strategies
- `confidence_greedy` — LLaDA's `low_confidence` remasking: top-k by max prob.
- `topk_margin` — Dream's `topk_margin`: top-k by (top1 prob − top2 prob).
- `klass` — KLASS: KL-adaptive stability + confidence thresholds (KLASS paper, default `kl_threshold=0.01`, `conf_threshold=0.9`, `history_length=2`).

## Evaluation
| Label | Task | Model | gen_len | steps | block_len | Metrics |
|-------|------|-------|---------|-------|-----------|---------|
| `llada-math` | MATH-500 | LLaDA-8B-Instruct | 256 | 256 | 64 | accuracy + avg_steps |
| `llada-humaneval` | HumanEval (164) | LLaDA-8B-Instruct | 256 | 256 | 64 | accuracy + avg_steps |
| `dream-text` | C4 prefix-continuation (256 samples, 32-tok prefix → 224-tok continuation) | Dream-v0-Instruct-7B | 224 | 256 | 224 | gen_ppl + MAUVE + entropy + rep2 + avg_steps |

### Metrics
| Metric | Direction | Where | Description |
|--------|-----------|-------|-------------|
| `accuracy` | ↑ | math/humaneval | exact-match (MATH) or pass@1 (HumanEval) |
| `gen_ppl` | ↓ | text | conditional perplexity via GPT-2-Large |
| `mauve` | ↑ | text | distributional similarity to C4 reference text |
| `entropy` | ↑ | text | bigram entropy (lexical diversity) |
| `rep2` | ↓ | text | repeated bigram ratio |
| `avg_steps` | ↓ | all | actual model forward passes used |

For MATH/HumanEval we use the KLASS protocol's `data/math_test.json`, prompts, and `utils.py` for answer extraction (`extract_math_answer`, `compare_answers`). The text-generation setting follows MDLM/ReMDM-style prefix-conditioned C4 continuation.
