"""Block Attention Residuals (Block AttnRes) baseline.

Instead of attending over ALL previous sublayer outputs (which requires
storing 2*n_layer source tensors and creates O(n_layer^2) memory), this
partitions 24 layers into blocks of 4. Within each block, standard residual
connections are used. At each block boundary, a learned pseudo-query attends
over the outputs of all preceding blocks (not sublayers), selecting which
block representations to combine for the input to the next block.

This reduces the attention source list from 48 sublayer outputs to ~6 block
outputs, cutting memory by ~8x vs full AttnRes while retaining the key
benefit of dynamic depth-wise aggregation.

Reference: "Attention Residuals" (Kimi Team, arXiv:2603.15031, 2026)
  - Section on "Block AttnRes" for practical scaling.

Changes:
  - Block.__init__/forward: Standard Pre-LN residual (unchanged)
  - GPT.__init__: Add per-block-boundary pseudo-query vectors
  - GPT.forward: Block AttnRes loop with attention at block boundaries
  - GPT.configure_optimizers: Update param groups for pseudo-queries
"""

_FILE = "nanoGPT/custom_pretrain.py"

# ── 1. GPT.configure_optimizers: AttnRes param groups (lines 175-192) ──────
_OPTIMIZER = """\
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Separate AttnRes query params from main model params
        attnres_params = [self.attnres_queries, self.attnres_query_out]
        attnres_ids = {id(p) for p in attnres_params}
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and id(p) not in attnres_ids]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 and id(p) not in attnres_ids]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
            {'params': attnres_params, 'lr': learning_rate * 0.1, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        num_attnres_params = sum(p.numel() for p in attnres_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        print(f"num AttnRes parameter tensors: {len(attnres_params)}, with {num_attnres_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer
"""

# ── 2. GPT.forward: Block AttnRes loop (lines 162-164) ─────────────────
_FORWARD_LOOP = """\
        # ── Block Attention Residuals: standard residual within blocks,
        #    attention aggregation at block boundaries ──
        block_size_layers = self.attnres_block_size
        n_blocks = len(self.transformer.h) // block_size_layers
        block_outputs = [x]  # initial embedding is first source
        for blk_idx in range(n_blocks):
            # At block boundary (except first): attend over previous block outputs
            if blk_idx > 0:
                stacked = torch.stack(block_outputs, dim=0)  # (num_sources, B, T, D)
                keys_normed = F.rms_norm(stacked, (stacked.size(-1),))
                logits = torch.einsum('d, n b t d -> n b t', self.attnres_queries[blk_idx - 1], keys_normed)
                weights = logits.softmax(dim=0)  # (num_sources, B, T)
                x = torch.einsum('n b t, n b t d -> b t d', weights, stacked)
            # Run layers within this block with standard residual connections
            start = blk_idx * block_size_layers
            end = start + block_size_layers
            for layer_idx in range(start, end):
                x = self.transformer.h[layer_idx](x)
            block_outputs.append(x)
        # Final output: attend over all block outputs with dedicated query
        stacked = torch.stack(block_outputs, dim=0)
        keys_normed = F.rms_norm(stacked, (stacked.size(-1),))
        logits = torch.einsum('d, n b t d -> n b t', self.attnres_query_out, keys_normed)
        weights = logits.softmax(dim=0)
        x = torch.einsum('n b t, n b t d -> b t d', weights, stacked)
"""

# ── 3. GPT.__init__: AttnRes parameters (lines 128-130) ───────────────────
_INIT = """\
        # ── Block Attention Residuals: partition layers into blocks ──
        # 24 layers / 4 = 6 blocks; attention at 5 boundaries + 1 output query
        self.attnres_block_size = 4  # layers per block
        n_blocks = config.n_layer // self.attnres_block_size
        # n_blocks-1 boundary queries (first block gets embedding directly)
        self.attnres_queries = nn.Parameter(torch.zeros(n_blocks - 1, config.n_embd))
        self.attnres_query_out = nn.Parameter(torch.zeros(config.n_embd))
"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 175,
        "end_line": 192,
        "content": _OPTIMIZER,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 162,
        "end_line": 164,
        "content": _FORWARD_LOOP,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 128,
        "end_line": 130,
        "content": _INIT,
    },
]
