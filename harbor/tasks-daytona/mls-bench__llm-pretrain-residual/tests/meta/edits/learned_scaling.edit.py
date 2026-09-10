"""Learnable Residual Scaling + x0 Injection baseline.

Inspired by the modded-nanogpt speedrun records (KellerJordan) and Nick
Ryan's 334M validation.  Each block's residual connection is controlled
by two learned scalars per layer:

  x_new = resid_lambda[i] * x + delta + x0_lambda[i] * x0

where delta = block(x) - x is the block's contribution and x0 is the
embedding output (post-dropout, post-position-encoding).

resid_lambdas init to 1.0 (standard residual), x0_lambdas init to 0.0
(no injection at start), so at initialization this reduces to vanilla.

The x0 injection provides a "gradient highway" that directly connects
the embedding layer to every depth, improving gradient flow in deep
Pre-LN transformers.

Changes:
  - Block: Unchanged — vanilla Pre-LN residual.
  - GPT.__init__: Add per-layer resid_lambdas and x0_lambdas parameters.
  - GPT.forward: Apply learned scaling with x0 injection.
  - GPT.configure_optimizers: Route scaling params to no-decay group.
"""

_FILE = "nanoGPT/custom_pretrain.py"

# ── 1. GPT.configure_optimizers: scaling param groups (lines 175-192) ────
_OPTIMIZER = """\
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Route residual scaling params to no-decay group
        scaling_ids = {id(self.resid_lambdas), id(self.x0_lambdas)}
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and id(p) not in scaling_ids]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 and id(p) not in scaling_ids]
        scaling_params = [p for n, p in param_dict.items() if id(p) in scaling_ids]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
            {'params': scaling_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        num_scaling_params = sum(p.numel() for p in scaling_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        print(f"num scaling parameter tensors: {len(scaling_params)}, with {num_scaling_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer
"""

# ── 2. GPT.forward: learned scaling + x0 injection (lines 162-164) ──────
_FORWARD_LOOP = """\
        # ── Learnable residual scaling + x0 injection ──
        # x0 = embedding output; provides gradient highway to every depth.
        x0 = x
        for i, block in enumerate(self.transformer.h):
            block_out = block(x)
            delta = block_out - x
            x = self.resid_lambdas[i] * x + delta + self.x0_lambdas[i] * x0
"""

# ── 3. GPT.__init__: scaling parameters (lines 128-130) ──────────────────
_INIT = """\
        # ── Learnable residual scaling + x0 injection ──
        # resid_lambdas[i]: scales the incoming residual stream (init 1.0 = vanilla)
        # x0_lambdas[i]:    scales the embedding injection (init 0.0 = no injection)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
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
