"""Lion optimizer baseline (medium).

Sign-based optimizer that uses only the sign of the momentum for updates.
Simpler than Adam (no second moment), often competitive or better.

Reference: Chen et al., "Symbolic Discovery of Optimization Algorithms" (2023)
"""

_FILE = "nanoGPT/custom_pretrain.py"

_LION_OPTIMIZER = """\
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        class Lion(torch.optim.Optimizer):
            \"\"\"Lion optimizer — sign-based updates with EMA momentum.\"\"\"
            def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
                defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
                super().__init__(params, defaults)
            @torch.no_grad()
            def step(self):
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        grad = p.grad
                        state = self.state[p]
                        if len(state) == 0:
                            state['exp_avg'] = torch.zeros_like(p)
                        exp_avg = state['exp_avg']
                        beta1, beta2 = group['betas']
                        # Weight decay first (decoupled, before update)
                        if group['weight_decay'] != 0:
                            p.mul_(1 - group['lr'] * group['weight_decay'])
                        update = exp_avg * beta1 + grad * (1 - beta1)
                        p.add_(torch.sign(update), alpha=-group['lr'])
                        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        optimizer = Lion(optim_groups, lr=learning_rate * 0.3, betas=betas)
        print("using Lion optimizer")
        return optimizer
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 171,
        "end_line": 189,
        "content": _LION_OPTIMIZER,
    },
]
