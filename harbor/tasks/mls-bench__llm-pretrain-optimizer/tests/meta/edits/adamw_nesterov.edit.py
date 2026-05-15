"""AdamW with Nesterov momentum baseline (basic).

Standard AdamW but with Nesterov-style momentum (NAdam).
Simple improvement over vanilla AdamW.

Reference: Dozat, "Incorporating Nesterov Momentum into Adam" (2016)
"""

_FILE = "nanoGPT/custom_pretrain.py"

_NADAM_OPTIMIZER = """\
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        optimizer = torch.optim.NAdam(optim_groups, lr=learning_rate, betas=betas,
                                      decoupled_weight_decay=True)
        print("using NAdam optimizer")
        return optimizer
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 171,
        "end_line": 189,
        "content": _NADAM_OPTIMIZER,
    },
]
