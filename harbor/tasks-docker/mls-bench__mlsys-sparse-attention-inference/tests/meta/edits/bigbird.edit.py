"""BigBird: global + window + random block-sparse attention.

Each query attends to:
  (1) GLOBAL tokens — the first ``num_global`` tokens (handle dual roles:
      attended *by* all and *to* all, like CLS/sink).
  (2) WINDOW — block-local window of ``num_window`` blocks around the
      query's block (band-limited attention).
  (3) RANDOM — for each block, ``num_random`` randomly chosen non-window
      non-global blocks (added at module-init, fixed across forwards).

Block size 64 → at N=8K, 128 blocks. With density target 25%:
  global=2 blocks (128 tokens) + window=3 blocks (192 tokens) +
  random=27 blocks ≈ 32 blocks attended per query block ≈ 25% of 128.

In parallel-forward eval the same fixed pattern is used by all queries —
no decode-time importance signal needed, so this method works correctly
without KV cache.

Reference: Zaheer et al., "Big Bird: Transformers for Longer Sequences"
(NeurIPS 2020), https://arxiv.org/abs/2007.14062.
"""

_FILE = "sparse-attn-eval/custom_sparse_attn.py"

_CONTENT = """\

class SparseAttention(nn.Module):
    \"\"\"BigBird — global + window + random block-sparse pattern.\"\"\"

    BLOCK = 64
    NUM_GLOBAL_BLOCKS = 2  # first 2 blocks (128 tokens) act as global sinks
    NUM_WINDOW_BLOCKS = 3  # band of 3 blocks around the query block

    def __init__(self, head_dim, num_heads, block_size=64, density_budget=0.25):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.density_budget = density_budget
        self.last_density = None
        # Random-block cache, keyed by (N, device) — same pattern across calls
        # for the same sequence length (deterministic per layer instance).
        self._random_cache = {}

    def _build_block_keep(self, N, device, is_causal):
        Bk = self.BLOCK
        if N % Bk != 0:
            # Pad-aware: round up to whole blocks; the (N,N) mask gets clipped.
            n_blocks = (N + Bk - 1) // Bk
        else:
            n_blocks = N // Bk
        g = min(self.NUM_GLOBAL_BLOCKS, n_blocks)
        w = self.NUM_WINDOW_BLOCKS
        # Solve random-blocks count from the budget at the BLOCK level.
        # The actual measured density (after random-block sampling and
        # causal AND) tends to land slightly above the linear estimate, so
        # apply a ~12% conservative margin to stay clear of the +0.02 slack
        # ceiling at every context length we evaluate.
        target = max(1, int(round(self.density_budget * 0.88 * n_blocks)))
        used = g + w
        r = max(0, min(target - used, n_blocks - used))
        # Build (n_blocks, n_blocks) bool keep
        keep = torch.zeros(n_blocks, n_blocks, dtype=torch.bool, device=device)
        # global cols (every query block attends to first g blocks)
        if g > 0:
            keep[:, :g] = True
        # global rows (first g blocks attend to everyone)
        if g > 0:
            keep[:g, :] = True
        # window: |bi - bj| <= w//2
        idx = torch.arange(n_blocks, device=device)
        win = (idx[:, None] - idx[None, :]).abs() <= w // 2
        keep |= win
        # random: per query block, sample r blocks from the non-(global|window) pool
        cache_key = (n_blocks, str(device), g, w, r)
        if cache_key not in self._random_cache:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(((0xBB ^ n_blocks) + int(torch.initial_seed()) - 42) & 0xFFFFFFFF)
            rand_keep = torch.zeros(n_blocks, n_blocks, dtype=torch.bool)
            base = keep.detach().to('cpu')
            for bi in range(n_blocks):
                avail = (~base[bi]).nonzero(as_tuple=False).flatten()
                if avail.numel() == 0 or r == 0:
                    continue
                pick = avail[torch.randperm(avail.numel(), generator=gen)[:r]]
                rand_keep[bi, pick] = True
            self._random_cache[cache_key] = rand_keep.to(device)
        keep |= self._random_cache[cache_key]
        # Apply causal at block level (a query block i may attend to j<=i)
        if is_causal:
            keep = keep & (idx[:, None] >= idx[None, :])
        return keep, n_blocks

    def forward(self, q, k, v, is_causal=False, scale=None):
        B, H, N, D = q.shape
        Bk = self.BLOCK
        scale = scale if scale is not None else 1.0 / math.sqrt(D)

        block_keep, n_blocks = self._build_block_keep(N, q.device, is_causal)
        # Expand block_keep -> token-level (N, N) by index gather
        q_tok_blk = (torch.arange(N, device=q.device) // Bk).clamp(max=n_blocks - 1)
        k_tok_blk = q_tok_blk
        token_keep = block_keep[q_tok_blk][:, k_tok_blk]   # (N, N) bool
        if is_causal:
            idx = torch.arange(N, device=q.device)
            token_keep = token_keep & (idx[:, None] >= idx[None, :])

        denom = (N * (N + 1) / 2.0) if is_causal else float(N * N)
        self.last_density = float(token_keep.sum().item()) / max(denom, 1.0)

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
