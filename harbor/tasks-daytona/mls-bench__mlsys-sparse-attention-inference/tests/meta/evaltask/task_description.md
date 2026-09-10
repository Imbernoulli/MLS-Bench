# Long-Context Inference-Time Sparse Attention

## Research Question

Design a sparse attention module that drops into a pretrained 1.5B-param
causal LLM at inference time and preserves long-context task quality
under a fixed sparsity budget — no retraining, no fine-tuning, no
architectural surgery beyond replacing the attention forward.

## Background

Inference-time sparse attention has been studied along two axes:

- Static patterns: sliding-window, sink + window (StreamingLLM, Xiao et
  al., ICLR 2024; arXiv:2309.17453), block-sparse with global / window /
  random tokens (BigBird, Zaheer et al., NeurIPS 2020;
  arXiv:2007.14062), and dilated patterns.
- Content-adaptive patterns: Reformer LSH, NSA-style block-level top-K
  (Yuan et al., 2025; arXiv:2502.11089 — block-selection branch), and
  Quest-style query-aware selection.

KV-cache compression methods (H2O, SnapKV, StreamingLLM-as-cache) are
designed for autoregressive decode with a mutable KV cache and are not
evaluated here — this benchmark uses a parallel-forward setup where every
forward processes the full prefix in one shot and the same
`SparseAttention` module replays at every generation step. Methods whose
importance signal shifts with the observation window (H2O, SnapKV) drift
during generation under this setting; the baseline set is therefore
restricted to methods that operate correctly under parallel forward.

NIAH retrieval and LongBench QA evaluations need instruction-following
ability, so this task uses an instruction-tuned backbone rather than a
base model. The agent's `SparseAttention` instance (one per attention
layer) is monkey-patched into `Qwen/Qwen2.5-1.5B-Instruct` (12 query
heads, 2 KV heads — the harness handles GQA replication so this module
sees 12 heads on both Q and K/V).

| Env | Metric | Notes |
|---|---|---|
| `niah_8k` | retrieval accuracy | Synthetic Needle-In-A-Haystack at 8K context |
| `longbench_qasper` | QA F1 | LongBench Qasper single-doc scientific paper QA |
| `longbench_multifieldqa_en` | QA F1 | LongBench MultiFieldQA-EN long-document multi-field QA |

## Task

Edit the `SparseAttention` class in
`sparse-attn-eval/custom_sparse_attn.py`. The rest of the file, plus
`harness.py` and `run_llm.py`, are read-only — they handle model loading,
attention monkey-patching, density tracking, and metric computation.

## Interface

```python
class SparseAttention(nn.Module):
    def __init__(self, head_dim, num_heads, block_size=64, density_budget=0.25): ...
    def forward(self, q, k, v, is_causal=False, scale=None) -> torch.Tensor: ...
```

`q`, `k`, `v` arrive as `(B, H, N, D)` in float16/bfloat16. `is_causal=True`
for the causal LLM. Return the attention output in the same shape and
dtype.

After every forward, set `self.last_density` to the fraction of (q, k)
pairs that received non-zero attention (causal-adjusted: divide by
`N(N+1)/2` when `is_causal=True`). The harness aggregates `last_density`
across all attention layers and aborts the run if the mean exceeds the
density budget (`0.25 + 0.02 slack`) for any non-`dense` baseline.
Missing, NaN, infinite, negative, or `>1` density reports are treated as
harness errors, not as zero density.

## Sparsity Budget

- `density_budget = 0.25`.
- Only the reference `dense` baseline is allowed to exceed it: it reports
  the true `last_density = 1.0`, and the dense run is invoked with
  `ALLOW_DENSE_FLAG=1` (set as a baseline-level env var in `config.json`)
  which forwards `--allow-dense` to `run_llm.py` so
  `harness.enforce_budget(allow_dense=True)` skips the budget check.

## Constraints

- Inference only — do not modify weights, do not add training.
- Single A100 80GB; FP16 only (no FP8).
- No Triton kernels; pure PyTorch ops or
  `torch.nn.attention.flex_attention` if available in this PyTorch
  version.
- Branching on `is_causal` and on `N` is fine. The forward signature
  includes `is_causal` for forward compatibility (currently always True).

## Baselines

1. `dense` — full attention oracle (density 1.0; the only baseline
   allowed to exceed the 0.25 budget).
2. `streaming_llm` — 4 attention sinks + sliding window (Xiao et al.,
   ICLR 2024; arXiv:2309.17453). Canonical static sink+window pattern.
3. `bigbird` — global + window + random block-sparse pattern (Zaheer et
   al., NeurIPS 2020; arXiv:2007.14062). Static, theoretically full-rank.
4. `block_topk` — content-adaptive block-level top-K via mean-pooled-key
   scoring, following the block-selection branch of NSA (Yuan et al.,
   2025; arXiv:2502.11089). Diagonal block always retained; importance
   computed per query block at inference time.

All baselines are paper-faithful and operate correctly in
parallel-forward mode (`use_cache=False`). H2O / SnapKV / Quest are
deliberately excluded: their importance signal is keyed off a mutable KV
cache during decode and does not transfer cleanly to single-shot
prefill+generate without cache plumbing that this harness does not
implement.
