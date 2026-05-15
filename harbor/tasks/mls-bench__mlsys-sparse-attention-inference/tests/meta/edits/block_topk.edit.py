"""Block-Top-K (NSA-style content-adaptive block-sparse attention).

For each query block i, score all key blocks j <= i (causal) by the
mean-pooled query x mean-pooled key dot product, then keep the top-K
scoring blocks plus the diagonal (current) block. Density at 25% with
BLOCK=64 and N=8K → 128 query blocks; per query block keep
TOP_K = round(0.25 * n_blocks) ≈ 32 blocks (including diagonal).

Unlike H2O / SnapKV, the importance signal here is computed PER QUERY
BLOCK from the queries that are about to consume the attention output —
so it works correctly under parallel-forward eval (no decode-time KV
cache mutation, no drift of the observation window).

Reference: NSA-style mean-pooled block selection. Adapted from the
pretrain-task NSA-block baseline with identical scoring logic, made
parallel-forward-friendly for this inference harness.
"""

_FILE = "sparse-attn-eval/custom_sparse_attn.py"

_CONTENT = """\

class SparseAttention(nn.Module):
    \"\"\"NSA-style content-adaptive block-sparse top-K attention.\"\"\"

    BLOCK = 64

    def __init__(self, head_dim, num_heads, block_size=64, density_budget=0.25):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.density_budget = density_budget
        self.last_density = None

    def forward(self, q, k, v, is_causal=False, scale=None):
        B, H, N, D = q.shape
        Bk = self.BLOCK
        scale = scale if scale is not None else 1.0 / math.sqrt(D)

        # Pad N up to a multiple of BLOCK so we can pool cleanly.
        Npad = ((N + Bk - 1) // Bk) * Bk
        if Npad != N:
            pad = Npad - N
            qp = torch.nn.functional.pad(q, (0, 0, 0, pad))
            kp = torch.nn.functional.pad(k, (0, 0, 0, pad))
        else:
            qp, kp = q, k

        n_blocks = Npad // Bk

        # Mean-pooled q / k per block: (B, H, n_blocks, D)
        q_blocks = qp.view(B, H, n_blocks, Bk, D).mean(dim=3)
        k_blocks = kp.view(B, H, n_blocks, Bk, D).mean(dim=3)

        # Block-level scores (B, H, n_blocks, n_blocks)
        scores = torch.einsum('bhmd,bhnd->bhmn',
                              q_blocks.float(), k_blocks.float()) * scale

        idx = torch.arange(n_blocks, device=q.device)
        if is_causal:
            causal_blk = idx[:, None] >= idx[None, :]   # (n_b, n_b)
            scores = scores.masked_fill(~causal_blk, float('-inf'))

        # Force-include diagonal: zero its score (irrelevant for topk choice)
        diag_mask = torch.zeros(n_blocks, n_blocks, dtype=torch.bool, device=q.device)
        diag_mask.fill_diagonal_(True)
        scores_nodiag = scores.masked_fill(diag_mask, float('-inf'))

        # Target top-K per query block under causal AND. With per-row keep K,
        # mean kept block-pairs = K*(2n-K+1)/2, denom = n*(n+1)/2 (causal),
        # so density = K*(2n-K+1)/(n*(n+1)). Solve K(2n+1-K) = budget*n*(n+1)
        # via the quadratic root closer to 0:
        #   K = ((2n+1) - sqrt((2n+1)^2 - 4*budget*n*(n+1))) / 2
        n_b = n_blocks
        if is_causal:
            disc = max(0.0, (2 * n_b + 1) ** 2 - 4 * self.density_budget * n_b * (n_b + 1))
            K_top = max(1, int(((2 * n_b + 1) - math.sqrt(disc)) / 2))
        else:
            K_top = max(1, int(round(self.density_budget * n_b)))
        kk = max(0, min(K_top - 1, n_b - 1))
        if kk > 0:
            topk_idx = scores_nodiag.topk(kk, dim=-1).indices  # (B,H,n_b,kk)
        else:
            topk_idx = torch.empty(B, H, n_blocks, 0, dtype=torch.long, device=q.device)
        diag_idx = idx.view(1, 1, n_blocks, 1).expand(B, H, n_blocks, 1)
        sel = torch.cat([topk_idx, diag_idx], dim=-1)
        block_keep = torch.zeros(B, H, n_blocks, n_blocks,
                                 dtype=torch.bool, device=q.device)
        block_keep.scatter_(-1, sel, True)
        if is_causal:
            block_keep = block_keep & causal_blk

        # Expand block_keep -> token-level (B, H, Npad, Npad)
        q_tok_blk = (torch.arange(Npad, device=q.device) // Bk)
        k_tok_blk = q_tok_blk
        # Index per-(b,h,qtok,ktok)
        token_keep = block_keep[:, :, q_tok_blk, :][:, :, :, k_tok_blk]
        token_keep = token_keep[:, :, :N, :N]
        if is_causal:
            tidx = torch.arange(N, device=q.device)
            token_keep = token_keep & (tidx[:, None] >= tidx[None, :])

        denom = (N * (N + 1) / 2.0) if is_causal else float(N * N)
        # Take per-(b,h) mean for reporting; harness aggregates further.
        self.last_density = float(token_keep[0, 0].sum().item()) / max(denom, 1.0)

        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        attn = attn.masked_fill(~token_keep, float('-inf'))
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
