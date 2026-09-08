"""MLP encoder baseline — rigorous codebase edit ops.

Replaces the default linear encoder with the original PEARL 3-layer MLP
encoder (200-200-200 hidden units, ReLU activations). Each transition is
encoded independently; the agent aggregates via product of Gaussians.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "oyster/custom_encoder.py"

# ── 1. Replace encoder class (lines 27-53) ──────────────────────────────

_MLP_ENCODER_CLASS = """\
class CustomContextEncoder(PyTorchModule):
    \"\"\"Original PEARL MLP context encoder (3-layer, 200 units).\"\"\"
    def __init__(self, hidden_sizes, input_size, output_size,
                 init_w=3e-3, hidden_activation=F.relu, **kwargs):
        self.save_init_params(locals())
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation

        in_dim = input_size
        self.fcs = nn.ModuleList()
        for h_dim in hidden_sizes:
            fc = nn.Linear(in_dim, h_dim)
            ptu.fanin_init(fc.weight)
            fc.bias.data.fill_(0.1)
            self.fcs.append(fc)
            in_dim = h_dim
        self.last_fc = nn.Linear(in_dim, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))
        preactivation = self.last_fc(h)
        output = preactivation
        if return_preactivations:
            return output, preactivation
        return output

    def reset(self, num_tasks=1):
        pass

"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 27,
        "end_line": 53,
        "content": _MLP_ENCODER_CLASS,
    },
]
