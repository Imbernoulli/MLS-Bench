"""Muon optimizer baseline.

MomentUm Orthogonalized by Newton-schulz. Uses Nesterov momentum
followed by Newton-Schulz orthogonalization for 2D weight matrices.
Falls back to AdamW for non-2D params (embeddings, biases, norms).
Muon-family optimizers have reported training-efficiency gains on language
modeling workloads; this benchmark uses local validation for ranking.

Reference: Jordan et al., modded-nanogpt records #3-4
Paper: Bernstein & Newhouse, "Old Optimizer, New Norm" (2024)
Hyperparams: Muon-family task defaults use wd=0.1;
  "Muon is Scalable" (arXiv:2502.16982) identifies weight decay as
  crucial for scaling beyond 124M. Gradient clipping kept at default 1.0.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_MUON_OPTIMIZER = """\
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [(n, p) for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [(n, p) for n, p in param_dict.items() if p.dim() < 2]
        num_decay_params = sum(p.numel() for _, p in decay_params)
        num_nodecay_params = sum(p.numel() for _, p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Separate 2D projection weights (for Muon) from rest (for AdamW)
        muon_params = [p for n, p in decay_params
                       if 'wte' not in n and 'wpe' not in n and 'lm_head' not in n]
        adam_decay_params = [p for n, p in decay_params
                            if 'wte' in n or 'wpe' in n or 'lm_head' in n]
        adam_nodecay_params = [p for _, p in nodecay_params]

        class Muon(torch.optim.Optimizer):
            \"\"\"Muon — MomentUm Orthogonalized by Newton-schulz.
            Uses Newton-Schulz iteration to orthogonalize momentum-accumulated
            gradients for 2D weight matrices. Based on modded-nanogpt.
            \"\"\"
            def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, weight_decay=0.0):
                defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
                super().__init__(params, defaults)

            @staticmethod
            def zeroth_power_via_newtonschulz5(G, steps=5):
                \"\"\"Approximate G @ (G^T G)^{-1/2} via 5 Newton-Schulz iterations.\"\"\"
                assert G.ndim == 2
                a, b, c = (3.4445, -4.7750, 2.0315)
                X = G.bfloat16()
                X = X / (X.norm() + 1e-7)
                if G.size(0) > G.size(1):
                    X = X.T
                for _ in range(steps):
                    A = X @ X.T
                    X = a * X + b * (A @ X) + c * (A @ (A @ X))
                if G.size(0) > G.size(1):
                    X = X.T
                return X

            @torch.no_grad()
            def step(self):
                for group in self.param_groups:
                    lr = group['lr']
                    momentum = group['momentum']
                    wd = group.get('weight_decay', 0.0)
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        # Decoupled weight decay (before update)
                        if wd > 0:
                            p.mul_(1 - lr * wd)
                        g = p.grad
                        state = self.state[p]
                        if len(state) == 0:
                            state['momentum_buffer'] = torch.zeros_like(g)
                        buf = state['momentum_buffer']
                        # EMA momentum: buf = (1-beta)*grad + beta*buf
                        buf.lerp_(g, 1.0 - momentum)
                        # Nesterov: update = (1-beta)*grad + beta*buf
                        nesterov_g = g.lerp(buf, momentum)
                        if nesterov_g.dim() == 2:
                            orig_shape = nesterov_g.shape
                            # Split fused QKV (c_attn: 3*n_embd x n_embd) into 3 parts
                            if orig_shape[0] == 3 * orig_shape[1]:
                                parts = nesterov_g.split(orig_shape[1])
                                update = torch.cat([
                                    self.zeroth_power_via_newtonschulz5(part, steps=group['ns_steps'])
                                    for part in parts
                                ])
                                scale = max(1, orig_shape[0] // orig_shape[1]) ** 0.5
                            else:
                                update = self.zeroth_power_via_newtonschulz5(nesterov_g, steps=group['ns_steps'])
                                scale = max(1, orig_shape[0] / orig_shape[1]) ** 0.5
                            p.data.add_(update.to(p.dtype), alpha=-lr * scale)
                        else:
                            # Fallback: plain SGD with momentum for non-2D
                            p.add_(buf, alpha=-lr)

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        # Muon LR is typically ~0.02, much higher than Adam LR (~6e-4).
        # Use lr_scale so the training loop's LR schedule scales Muon proportionally.
        muon_base_lr = 0.02
        muon_lr_scale = muon_base_lr / learning_rate
        # Weight decay 0.1 follows the Muon-family task default.
        # "Muon is Scalable" (arXiv:2502.16982) identifies weight decay as
        # crucial for scaling Muon beyond 124M models.
        muon_opt = Muon([{'params': muon_params, 'lr_scale': muon_lr_scale}],
                        lr=muon_base_lr, momentum=0.95, weight_decay=0.1)
        # AdamW for embeddings/norms uses the base config weight_decay (0.1).
        adam_groups = [
            {'params': adam_decay_params, 'weight_decay': weight_decay},
            {'params': adam_nodecay_params, 'weight_decay': 0.0},
        ]
        adam_opt = torch.optim.AdamW(adam_groups, lr=learning_rate, betas=betas, **extra_args)

        class CombinedOptimizer:
            \"\"\"Combines Muon (for projections) with AdamW (for embeddings/norms).\"\"\"
            def __init__(self, optimizers):
                self.optimizers = optimizers
                self.param_groups = []
                for opt in optimizers:
                    self.param_groups.extend(opt.param_groups)
            def zero_grad(self, set_to_none=True):
                for opt in self.optimizers:
                    opt.zero_grad(set_to_none=set_to_none)
            def step(self):
                for opt in self.optimizers:
                    opt.step()
            def state_dict(self):
                return [opt.state_dict() for opt in self.optimizers]

        print(f"using Muon (lr={muon_base_lr}, scale={muon_lr_scale:.1f}) + AdamW combined optimizer")
        return CombinedOptimizer([muon_opt, adam_opt])
"""

_CONFIG_OVERRIDES = """\
    # CONFIG_OVERRIDES: override training hyperparameters for your method.
    # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
    CONFIG_OVERRIDES = {'learning_rate': 1e-3}
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 245,
        "end_line": 247,
        "content": _CONFIG_OVERRIDES,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 171,
        "end_line": 189,
        "content": _MUON_OPTIMIZER,
    },
]
