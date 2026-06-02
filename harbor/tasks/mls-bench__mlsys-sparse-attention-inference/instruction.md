# MLS-Bench: mlsys-sparse-attention-inference

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

Long-context understanding tasks need instruction-following ability, so
this task uses an instruction-tuned backbone rather than a base model.
The agent's `SparseAttention` instance (one per attention layer) is
monkey-patched into `Qwen/Qwen2.5-1.5B-Instruct` (12 query heads, 2 KV
heads — the harness handles GQA replication so this module sees 12 heads
on both Q and K/V).

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


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/sparse-attn-eval/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `sparse-attn-eval/custom_sparse_attn.py`
- editable lines **31–103**


Other files you may **read** for context (do not modify):
- `sparse-attn-eval/harness.py`
- `sparse-attn-eval/run_llm.py`


## Readable Context


### `sparse-attn-eval/custom_sparse_attn.py`  [EDITABLE — lines 31–103 only]

```python
     1: """Inference-time sparse attention module (agent-editable).
     2: 
     3: Single class ``SparseAttention`` is monkey-patched into Qwen2.5-1.5B-Instruct
     4: by the harness, which calls ``forward(q, k, v, is_causal=True, scale=...)``
     5: once per attention layer (q/k/v already shaped (B, H, N, D), GQA replicated).
     6: The harness reads ``self.last_density`` after every forward and aborts if
     7: density > 0.25 (small slack), except for the dense reference baseline.
     8: 
     9: PERFORMANCE — read this before redesigning forward(). At N=8192 a naïve
    10: ``einsum + softmax + einsum`` materializes a 3 GB bf16 attention matrix per
    11: layer per head and is *slower than dense*. To make sparse actually faster:
    12: 
    13:   1. Use fused SDPA: ``F.scaled_dot_product_attention(q, k, v, attn_mask=
    14:      bool_mask, ...)`` is 5–10× faster than einsum + manual softmax.
    15:   2. Cache the mask across layers — it depends only on (N, pattern), not
    16:      q/k/v values, so build once per prompt and reuse across all 24 layers.
    17:   3. For low density (≤ 10%), use ``torch.nn.attention.flex_attention`` with
    18:      ``create_block_mask`` — it compiles a true block-sparse kernel that
    19:      skips entire blocks (PyTorch-native, no Triton-by-hand needed).
    20:   4. Stay in bf16 — fused SDPA handles numerics safely; fp32 upcast 2×s memory.
    21: """
    22: 
    23: import math
    24: 
    25: import torch
    26: import torch.nn as nn
    27: import torch.nn.functional as F
    28: 
    29: 
    30: # ═══════════════════════════════════════════════════════════════════════════════
    31: # EDITABLE REGION START — design the inference-time sparse attention here.
    32: # Lines 31-103 are agent-editable. Everything outside this region is FIXED.
    33: # ═══════════════════════════════════════════════════════════════════════════════
    34: 
    35: 
    36: class SparseAttention(nn.Module):
    37:     """Inference-time sparse attention for long-context LLM eval.
    38: 
    39:     Default: sliding window + sink, computed via fused SDPA with a cached
    40:     bool mask (same speed regime as dense at this density). Replace with
    41:     your own design — anything that preserves long-context quality under
    42:     the density budget.
    43:     """
    44: 
    45:     def __init__(self, head_dim, num_heads, block_size=64, density_budget=0.25):
    46:         super().__init__()
    47:         self.head_dim = head_dim
    48:         self.num_heads = num_heads
    49:         self.block_size = block_size
    50:         self.density_budget = density_budget
    51: 
    52:         # Default hyperparameters (the agent should tune / replace these).
    53:         self.window = 1024      # local window radius (tokens per side)
    54:         self.num_sinks = 4      # number of "always attended" sink tokens
    55: 
    56:         # Diagnostic: harness reads this after each forward to validate budget.
    57:         self.last_density = None
    58: 
    59:         # (N, is_causal, device) -> (mask BoolTensor, density float).
    60:         # Reused across the 24 layers for the same prompt — built once per N.
    61:         self._mask_cache: dict = {}
    62: 
    63:     @torch.no_grad()
    64:     def _get_mask(self, N: int, device: torch.device, is_causal: bool):
    65:         key = (N, is_causal, device)
    66:         cached = self._mask_cache.get(key)
    67:         if cached is not None:
    68:             return cached
    69:         idx = torch.arange(N, device=device)
    70:         di = idx[:, None] - idx[None, :]
    71:         mask = (di.abs() <= self.window) | (idx[None, :] < self.num_sinks)
    72:         if is_causal:
    73:             mask = mask & (idx[:, None] >= idx[None, :])
    74:         denom = N * (N + 1) / 2.0 if is_causal else float(N * N)
    75:         density = float(mask.sum().item()) / max(denom, 1.0)
    76:         self._mask_cache[key] = (mask, density)
    77:         return mask, density
    78: 
    79:     def forward(self, q, k, v, is_causal=False, scale=None):
    80:         """Sparse attention forward.
    81: 
    82:         Args:
    83:             q, k, v: (B, H, N, D), float16 / bfloat16.
    84:             is_causal: True for LLM (always True in this task).
    85:             scale: softmax temperature; defaults to 1/sqrt(D).
    86:         Returns:
    87:             out: (B, H, N, D), same dtype as q.
    88:         """
    89:         B, H, N, D = q.shape
    90:         scale = scale if scale is not None else 1.0 / math.sqrt(D)
    91: 
    92:         mask, self.last_density = self._get_mask(N, q.device, is_causal)
    93:         # SDPA accepts a bool attn_mask: True = attend. Stay in bf16/fp16 —
    94:         # fused SDPA handles numerics safely.
    95:         out = F.scaled_dot_product_attention(
    96:             q, k, v, attn_mask=mask.view(1, 1, N, N),
    97:             dropout_p=0.0, is_causal=False, scale=scale,
    98:         )
    99:         return out
   100: 
   101: 
   102: # ═══════════════════════════════════════════════════════════════════════════════
   103: # EDITABLE REGION END
   104: # ═══════════════════════════════════════════════════════════════════════════════
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `dense` baseline — editable region  [READ-ONLY — reference implementation]

In `sparse-attn-eval/custom_sparse_attn.py`:

```python
Lines 31–55:
    28: 
    29: 
    30: # ═══════════════════════════════════════════════════════════════════════════════
    31: 
    32: class SparseAttention(nn.Module):
    33:     """Dense attention oracle. Reports true full-attention density.
    34: 
    35:     The point of this baseline is to give an upper-bound quality reference;
    36:     it is not meant to satisfy a real sparsity constraint.
    37:     """
    38: 
    39:     def __init__(self, head_dim, num_heads, block_size=64, density_budget=0.25):
    40:         super().__init__()
    41:         self.head_dim = head_dim
    42:         self.num_heads = num_heads
    43:         self.density_budget = density_budget
    44:         self.last_density = None
    45: 
    46:     def forward(self, q, k, v, is_causal=False, scale=None):
    47:         B, H, N, D = q.shape
    48:         scale = scale if scale is not None else 1.0 / math.sqrt(D)
    49:         # Use PyTorch's fused SDPA for efficient dense attention.
    50:         out = F.scaled_dot_product_attention(
    51:             q, k, v, attn_mask=None, dropout_p=0.0,
    52:             is_causal=is_causal, scale=scale,
    53:         )
    54:         self.last_density = 1.0
    55:         return out
    56: # ═══════════════════════════════════════════════════════════════════════════════
```

### `streaming_llm` baseline — editable region  [READ-ONLY — reference implementation]

In `sparse-attn-eval/custom_sparse_attn.py`:

```python
Lines 31–84:
    28: 
    29: 
    30: # ═══════════════════════════════════════════════════════════════════════════════
    31: 
    32: class SparseAttention(nn.Module):
    33:     """StreamingLLM-style sink + sliding window attention.
    34: 
    35:     Causal mode (LLM): row-relative last-W window (``i-W+1 <= j <= i``) plus
    36:     the first ``num_sinks`` columns.
    37:     Non-causal mode (ViT/DiT): symmetric window (``|i-j| <= W``) plus the
    38:     first ``num_sinks`` columns.
    39:     """
    40: 
    41:     def __init__(self, head_dim, num_heads, block_size=64, density_budget=0.25):
    42:         super().__init__()
    43:         self.head_dim = head_dim
    44:         self.num_heads = num_heads
    45:         self.block_size = block_size
    46:         self.density_budget = density_budget
    47:         # Paper default: 4 attention sinks. Xiao et al. show 4 is enough to
    48:         # recover almost all the dense-attention quality on streaming inputs.
    49:         self.num_sinks = 4
    50:         self.last_density = None
    51: 
    52:     def _build_mask(self, N, device, is_causal):
    53:         idx = torch.arange(N, device=device)
    54:         if is_causal:
    55:             # Solve density = 2*(num_sinks+W)/(N+1) = budget for W.
    56:             W = max(1, int(round(self.density_budget * (N + 1) / 2.0)) - self.num_sinks)
    57:             di = idx[:, None] - idx[None, :]
    58:             local = (di >= 0) & (di < W)
    59:         else:
    60:             # Solve density = (num_sinks + 2W + 1)/N = budget for W.
    61:             W = max(1, (int(round(self.density_budget * float(N))) - self.num_sinks - 1) // 2)
    62:             di = idx[:, None] - idx[None, :]
    63:             local = di.abs() <= W
    64:         sinks = (idx[None, :] < min(self.num_sinks, N))
    65:         mask = local | sinks
    66:         if is_causal:
    67:             mask = mask & (idx[:, None] >= idx[None, :])
    68:         return mask
    69: 
    70:     def forward(self, q, k, v, is_causal=False, scale=None):
    71:         B, H, N, D = q.shape
    72:         scale = scale if scale is not None else 1.0 / math.sqrt(D)
    73: 
    74:         mask = self._build_mask(N, q.device, is_causal)  # (N, N)
    75:         denom = (N * (N + 1) / 2.0) if is_causal else float(N * N)
    76:         self.last_density = float(mask.sum().item()) / max(denom, 1.0)
    77: 
    78:         # Broadcast (N,N) mask across (B,H).
    79:         attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    80:         attn = attn.masked_fill(~mask, float('-inf'))
    81:         attn = torch.softmax(attn, dim=-1)
    82:         attn = torch.nan_to_num(attn, nan=0.0)
    83:         out = torch.matmul(attn, v.float())
    84:         return out.to(q.dtype)
    85: # ═══════════════════════════════════════════════════════════════════════════════
```

### `bigbird` baseline — editable region  [READ-ONLY — reference implementation]

In `sparse-attn-eval/custom_sparse_attn.py`:

```python
Lines 31–121:
    28: 
    29: 
    30: # ═══════════════════════════════════════════════════════════════════════════════
    31: 
    32: class SparseAttention(nn.Module):
    33:     """BigBird — global + window + random block-sparse pattern."""
    34: 
    35:     BLOCK = 64
    36:     NUM_GLOBAL_BLOCKS = 2  # first 2 blocks (128 tokens) act as global sinks
    37:     NUM_WINDOW_BLOCKS = 3  # band of 3 blocks around the query block
    38: 
    39:     def __init__(self, head_dim, num_heads, block_size=64, density_budget=0.25):
    40:         super().__init__()
    41:         self.head_dim = head_dim
    42:         self.num_heads = num_heads
    43:         self.block_size = block_size
    44:         self.density_budget = density_budget
    45:         self.last_density = None
    46:         # Random-block cache, keyed by (N, device) — same pattern across calls
    47:         # for the same sequence length (deterministic per layer instance).
    48:         self._random_cache = {}
    49: 
    50:     def _build_block_keep(self, N, device, is_causal):
    51:         Bk = self.BLOCK
    52:         if N % Bk != 0:
    53:             # Pad-aware: round up to whole blocks; the (N,N) mask gets clipped.
    54:             n_blocks = (N + Bk - 1) // Bk
    55:         else:
    56:             n_blocks = N // Bk
    57:         g = min(self.NUM_GLOBAL_BLOCKS, n_blocks)
    58:         w = self.NUM_WINDOW_BLOCKS
    59:         # Solve random-blocks count from the budget at the BLOCK level.
    60:         # The actual measured density (after random-block sampling and
    61:         # causal AND) tends to land slightly above the linear estimate, so
    62:         # apply a ~12% conservative margin to stay clear of the +0.02 slack
    63:         # ceiling at every context length we evaluate.
    64:         target = max(1, int(round(self.density_budget * 0.88 * n_blocks)))
    65:         used = g + w
    66:         r = max(0, min(target - used, n_blocks - used))
    67:         # Build (n_blocks, n_blocks) bool keep
    68:         keep = torch.zeros(n_blocks, n_blocks, dtype=torch.bool, device=device)
    69:         # global cols (every query block attends to first g blocks)
    70:         if g > 0:
    71:             keep[:, :g] = True
    72:         # global rows (first g blocks attend to everyone)
    73:         if g > 0:
    74:             keep[:g, :] = True
    75:         # window: |bi - bj| <= w//2
    76:         idx = torch.arange(n_blocks, device=device)
    77:         win = (idx[:, None] - idx[None, :]).abs() <= w // 2
    78:         keep |= win
    79:         # random: per query block, sample r blocks from the non-(global|window) pool
    80:         cache_key = (n_blocks, str(device), g, w, r)
    81:         if cache_key not in self._random_cache:
    82:             gen = torch.Generator(device='cpu')
    83:             gen.manual_seed(((0xBB ^ n_blocks) + int(torch.initial_seed()) - 42) & 0xFFFFFFFF)
    84:             rand_keep = torch.zeros(n_blocks, n_blocks, dtype=torch.bool)
    85:             base = keep.detach().to('cpu')
    86:             for bi in range(n_blocks):
    87:                 avail = (~base[bi]).nonzero(as_tuple=False).flatten()
    88:                 if avail.numel() == 0 or r == 0:
    89:                     continue
    90:                 pick = avail[torch.randperm(avail.numel(), generator=gen)[:r]]
    91:                 rand_keep[bi, pick] = True
    92:             self._random_cache[cache_key] = rand_keep.to(device)
    93:         keep |= self._random_cache[cache_key]
    94:         # Apply causal at block level (a query block i may attend to j<=i)
    95:         if is_causal:
    96:             keep = keep & (idx[:, None] >= idx[None, :])
    97:         return keep, n_blocks
    98: 
    99:     def forward(self, q, k, v, is_causal=False, scale=None):
   100:         B, H, N, D = q.shape
   101:         Bk = self.BLOCK
   102:         scale = scale if scale is not None else 1.0 / math.sqrt(D)
   103: 
   104:         block_keep, n_blocks = self._build_block_keep(N, q.device, is_causal)
   105:         # Expand block_keep -> token-level (N, N) by index gather
   106:         q_tok_blk = (torch.arange(N, device=q.device) // Bk).clamp(max=n_blocks - 1)
   107:         k_tok_blk = q_tok_blk
   108:         token_keep = block_keep[q_tok_blk][:, k_tok_blk]   # (N, N) bool
   109:         if is_causal:
   110:             idx = torch.arange(N, device=q.device)
   111:             token_keep = token_keep & (idx[:, None] >= idx[None, :])
   112: 
   113:         denom = (N * (N + 1) / 2.0) if is_causal else float(N * N)
   114:         self.last_density = float(token_keep.sum().item()) / max(denom, 1.0)
   115: 
   116:         attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
   117:         attn = attn.masked_fill(~token_keep, float('-inf'))
   118:         attn = torch.softmax(attn, dim=-1)
   119:         attn = torch.nan_to_num(attn, nan=0.0)
   120:         out = torch.matmul(attn, v.float())
   121:         return out.to(q.dtype)
   122: # ═══════════════════════════════════════════════════════════════════════════════
```

### `block_topk` baseline — editable region  [READ-ONLY — reference implementation]

In `sparse-attn-eval/custom_sparse_attn.py`:

```python
Lines 31–122:
    28: 
    29: 
    30: # ═══════════════════════════════════════════════════════════════════════════════
    31: 
    32: class SparseAttention(nn.Module):
    33:     """NSA-style content-adaptive block-sparse top-K attention."""
    34: 
    35:     BLOCK = 64
    36: 
    37:     def __init__(self, head_dim, num_heads, block_size=64, density_budget=0.25):
    38:         super().__init__()
    39:         self.head_dim = head_dim
    40:         self.num_heads = num_heads
    41:         self.block_size = block_size
    42:         self.density_budget = density_budget
    43:         self.last_density = None
    44: 
    45:     def forward(self, q, k, v, is_causal=False, scale=None):
    46:         B, H, N, D = q.shape
    47:         Bk = self.BLOCK
    48:         scale = scale if scale is not None else 1.0 / math.sqrt(D)
    49: 
    50:         # Pad N up to a multiple of BLOCK so we can pool cleanly.
    51:         Npad = ((N + Bk - 1) // Bk) * Bk
    52:         if Npad != N:
    53:             pad = Npad - N
    54:             qp = torch.nn.functional.pad(q, (0, 0, 0, pad))
    55:             kp = torch.nn.functional.pad(k, (0, 0, 0, pad))
    56:         else:
    57:             qp, kp = q, k
    58: 
    59:         n_blocks = Npad // Bk
    60: 
    61:         # Mean-pooled q / k per block: (B, H, n_blocks, D)
    62:         q_blocks = qp.view(B, H, n_blocks, Bk, D).mean(dim=3)
    63:         k_blocks = kp.view(B, H, n_blocks, Bk, D).mean(dim=3)
    64: 
    65:         # Block-level scores (B, H, n_blocks, n_blocks)
    66:         scores = torch.einsum('bhmd,bhnd->bhmn',
    67:                               q_blocks.float(), k_blocks.float()) * scale
    68: 
    69:         idx = torch.arange(n_blocks, device=q.device)
    70:         if is_causal:
    71:             causal_blk = idx[:, None] >= idx[None, :]   # (n_b, n_b)
    72:             scores = scores.masked_fill(~causal_blk, float('-inf'))
    73: 
    74:         # Force-include diagonal: zero its score (irrelevant for topk choice)
    75:         diag_mask = torch.zeros(n_blocks, n_blocks, dtype=torch.bool, device=q.device)
    76:         diag_mask.fill_diagonal_(True)
    77:         scores_nodiag = scores.masked_fill(diag_mask, float('-inf'))
    78: 
    79:         # Target top-K per query block under causal AND. With per-row keep K,
    80:         # mean kept block-pairs = K*(2n-K+1)/2, denom = n*(n+1)/2 (causal),
    81:         # so density = K*(2n-K+1)/(n*(n+1)). Solve K(2n+1-K) = budget*n*(n+1)
    82:         # via the quadratic root closer to 0:
    83:         #   K = ((2n+1) - sqrt((2n+1)^2 - 4*budget*n*(n+1))) / 2
    84:         n_b = n_blocks
    85:         if is_causal:
    86:             disc = max(0.0, (2 * n_b + 1) ** 2 - 4 * self.density_budget * n_b * (n_b + 1))
    87:             K_top = max(1, int(((2 * n_b + 1) - math.sqrt(disc)) / 2))
    88:         else:
    89:             K_top = max(1, int(round(self.density_budget * n_b)))
    90:         kk = max(0, min(K_top - 1, n_b - 1))
    91:         if kk > 0:
    92:             topk_idx = scores_nodiag.topk(kk, dim=-1).indices  # (B,H,n_b,kk)
    93:         else:
    94:             topk_idx = torch.empty(B, H, n_blocks, 0, dtype=torch.long, device=q.device)
    95:         diag_idx = idx.view(1, 1, n_blocks, 1).expand(B, H, n_blocks, 1)
    96:         sel = torch.cat([topk_idx, diag_idx], dim=-1)
    97:         block_keep = torch.zeros(B, H, n_blocks, n_blocks,
    98:                                  dtype=torch.bool, device=q.device)
    99:         block_keep.scatter_(-1, sel, True)
   100:         if is_causal:
   101:             block_keep = block_keep & causal_blk
   102: 
   103:         # Expand block_keep -> token-level (B, H, Npad, Npad)
   104:         q_tok_blk = (torch.arange(Npad, device=q.device) // Bk)
   105:         k_tok_blk = q_tok_blk
   106:         # Index per-(b,h,qtok,ktok)
   107:         token_keep = block_keep[:, :, q_tok_blk, :][:, :, :, k_tok_blk]
   108:         token_keep = token_keep[:, :, :N, :N]
   109:         if is_causal:
   110:             tidx = torch.arange(N, device=q.device)
   111:             token_keep = token_keep & (tidx[:, None] >= tidx[None, :])
   112: 
   113:         denom = (N * (N + 1) / 2.0) if is_causal else float(N * N)
   114:         # Take per-(b,h) mean for reporting; harness aggregates further.
   115:         self.last_density = float(token_keep[0, 0].sum().item()) / max(denom, 1.0)
   116: 
   117:         attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
   118:         attn = attn.masked_fill(~token_keep, float('-inf'))
   119:         attn = torch.softmax(attn, dim=-1)
   120:         attn = torch.nan_to_num(attn, nan=0.0)
   121:         out = torch.matmul(attn, v.float())
   122:         return out.to(q.dtype)
   123: # ═══════════════════════════════════════════════════════════════════════════════
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
