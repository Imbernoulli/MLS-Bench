"""StreamingLLM: attention sinks + sliding window.

For each query at position i, attend to:
  (a) the first ``num_sinks`` tokens — the "attention sinks" identified by
      Xiao et al. as carrying disproportionate softmax mass when a model is
      forced to drop early context, and
  (b) a sliding window of size ``W`` covering the most recent keys — for
      causal LLMs this is ``j in [i-W+1, i]``; for non-causal models we apply
      the symmetric variant ``|i-j| <= W``.

Density bookkeeping (matches harness ``denom = N(N+1)/2`` causal, ``N*N``
non-causal — see harness.enforce_budget).

Causal:
  Per row i (large i): ``num_sinks + W`` keys (no overlap once i >= num_sinks+W).
  Total mask sum ≈ N * (num_sinks + W).
  Density = N*(num_sinks+W) / (N*(N+1)/2) ≈ 2*(num_sinks+W) / (N+1).
  Solve for W: W ≈ density_budget*(N+1)/2 - num_sinks.

Non-causal:
  Per row: ``num_sinks + 2W+1`` keys (no overlap at i >= num_sinks+W).
  Density = (num_sinks + 2W+1) / N.
  Solve for W: W ≈ (density_budget*N - num_sinks - 1) / 2.

The naive sizing ``avg_row = budget*(N+1)/2`` used by the original deleted
streaming_llm baseline was *wrong* — it conflated row-relative window count
with column-relative density and over-shot the budget. The formulas above
are derived directly from mask-sum / denom so the harness budget check passes.

Reference: Xiao et al., "Efficient Streaming Language Models with Attention
Sinks" (ICLR 2024). https://arxiv.org/abs/2309.17453
"""

_FILE = "sparse-attn-eval/custom_sparse_attn.py"

_CONTENT = """\

class SparseAttention(nn.Module):
    \"\"\"StreamingLLM-style sink + sliding window attention.

    Causal mode (LLM): row-relative last-W window (``i-W+1 <= j <= i``) plus
    the first ``num_sinks`` columns.
    Non-causal mode (ViT/DiT): symmetric window (``|i-j| <= W``) plus the
    first ``num_sinks`` columns.
    \"\"\"

    def __init__(self, head_dim, num_heads, block_size=64, density_budget=0.25):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.density_budget = density_budget
        # Paper default: 4 attention sinks. Xiao et al. show 4 is enough to
        # recover almost all the dense-attention quality on streaming inputs.
        self.num_sinks = 4
        self.last_density = None

    def _build_mask(self, N, device, is_causal):
        idx = torch.arange(N, device=device)
        if is_causal:
            # Solve density = 2*(num_sinks+W)/(N+1) = budget for W.
            W = max(1, int(round(self.density_budget * (N + 1) / 2.0)) - self.num_sinks)
            di = idx[:, None] - idx[None, :]
            local = (di >= 0) & (di < W)
        else:
            # Solve density = (num_sinks + 2W + 1)/N = budget for W.
            W = max(1, (int(round(self.density_budget * float(N))) - self.num_sinks - 1) // 2)
            di = idx[:, None] - idx[None, :]
            local = di.abs() <= W
        sinks = (idx[None, :] < min(self.num_sinks, N))
        mask = local | sinks
        if is_causal:
            mask = mask & (idx[:, None] >= idx[None, :])
        return mask

    def forward(self, q, k, v, is_causal=False, scale=None):
        B, H, N, D = q.shape
        scale = scale if scale is not None else 1.0 / math.sqrt(D)

        mask = self._build_mask(N, q.device, is_causal)  # (N, N)
        denom = (N * (N + 1) / 2.0) if is_causal else float(N * N)
        self.last_density = float(mask.sum().item()) / max(denom, 1.0)

        # Broadcast (N,N) mask across (B,H).
        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(attn, v.float())
        return out.to(q.dtype)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 31,
        "end_line": 103,
        "content": _CONTENT,
    },
]
