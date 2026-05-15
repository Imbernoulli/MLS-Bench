"""Inference-time sparse attention module (agent-editable).

Single class ``SparseAttention`` is monkey-patched into Qwen2.5-1.5B-Instruct
by the harness, which calls ``forward(q, k, v, is_causal=True, scale=...)``
once per attention layer (q/k/v already shaped (B, H, N, D), GQA replicated).
The harness reads ``self.last_density`` after every forward and aborts if
density > 0.25 (small slack), except for the dense reference baseline.

PERFORMANCE — read this before redesigning forward(). At N=8192 a naïve
``einsum + softmax + einsum`` materializes a 3 GB bf16 attention matrix per
layer per head and is *slower than dense*. To make sparse actually faster:

  1. Use fused SDPA: ``F.scaled_dot_product_attention(q, k, v, attn_mask=
     bool_mask, ...)`` is 5–10× faster than einsum + manual softmax.
  2. Cache the mask across layers — it depends only on (N, pattern), not
     q/k/v values, so build once per prompt and reuse across all 24 layers.
  3. For low density (≤ 10%), use ``torch.nn.attention.flex_attention`` with
     ``create_block_mask`` — it compiles a true block-sparse kernel that
     skips entire blocks (PyTorch-native, no Triton-by-hand needed).
  4. Stay in bf16 — fused SDPA handles numerics safely; fp32 upcast 2×s memory.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE REGION START — design the inference-time sparse attention here.
# Lines 31-103 are agent-editable. Everything outside this region is FIXED.
# ═══════════════════════════════════════════════════════════════════════════════


class SparseAttention(nn.Module):
    """Inference-time sparse attention for long-context LLM eval.

    Default: sliding window + sink, computed via fused SDPA with a cached
    bool mask (same speed regime as dense at this density). Replace with
    your own design — anything that preserves quality on NIAH / Qasper /
    MultiFieldQA-EN under the density budget.
    """

    def __init__(self, head_dim, num_heads, block_size=64, density_budget=0.25):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.density_budget = density_budget

        # Default hyperparameters (the agent should tune / replace these).
        self.window = 1024      # local window radius (tokens per side)
        self.num_sinks = 4      # number of "always attended" sink tokens

        # Diagnostic: harness reads this after each forward to validate budget.
        self.last_density = None

        # (N, is_causal, device) -> (mask BoolTensor, density float).
        # Reused across the 24 layers for the same prompt — built once per N.
        self._mask_cache: dict = {}

    @torch.no_grad()
    def _get_mask(self, N: int, device: torch.device, is_causal: bool):
        key = (N, is_causal, device)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        idx = torch.arange(N, device=device)
        di = idx[:, None] - idx[None, :]
        mask = (di.abs() <= self.window) | (idx[None, :] < self.num_sinks)
        if is_causal:
            mask = mask & (idx[:, None] >= idx[None, :])
        denom = N * (N + 1) / 2.0 if is_causal else float(N * N)
        density = float(mask.sum().item()) / max(denom, 1.0)
        self._mask_cache[key] = (mask, density)
        return mask, density

    def forward(self, q, k, v, is_causal=False, scale=None):
        """Sparse attention forward.

        Args:
            q, k, v: (B, H, N, D), float16 / bfloat16.
            is_causal: True for LLM (always True in this task).
            scale: softmax temperature; defaults to 1/sqrt(D).
        Returns:
            out: (B, H, N, D), same dtype as q.
        """
        B, H, N, D = q.shape
        scale = scale if scale is not None else 1.0 / math.sqrt(D)

        mask, self.last_density = self._get_mask(N, q.device, is_causal)
        # SDPA accepts a bool attn_mask: True = attend. Stay in bf16/fp16 —
        # fused SDPA handles numerics safely.
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask.view(1, 1, N, N),
            dropout_p=0.0, is_causal=False, scale=scale,
        )
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE REGION END
# ═══════════════════════════════════════════════════════════════════════════════
