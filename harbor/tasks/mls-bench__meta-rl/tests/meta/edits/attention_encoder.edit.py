"""Attention encoder baseline — rigorous codebase edit ops.

Replaces the default linear encoder with a self-attention context encoder.
Each transition is first embedded via MLP layers, then self-attention lets
transitions attend to each other. Per-transition outputs are preserved for
product-of-Gaussians aggregation in the agent.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "oyster/custom_encoder.py"

# ── 1. Replace encoder class (lines 27-53) ──────────────────────────────

_ATTENTION_ENCODER_CLASS = """\
class CustomContextEncoder(PyTorchModule):
    \"\"\"Self-attention context encoder for cross-transition reasoning.\"\"\"
    def __init__(self, hidden_sizes, input_size, output_size,
                 init_w=3e-3, hidden_activation=F.relu, **kwargs):
        self.save_init_params(locals())
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.hidden_dim = hidden_sizes[-1]

        # Per-transition MLP embedding
        in_dim = input_size
        self.fcs = nn.ModuleList()
        for h_dim in hidden_sizes:
            fc = nn.Linear(in_dim, h_dim)
            ptu.fanin_init(fc.weight)
            fc.bias.data.fill_(0.1)
            self.fcs.append(fc)
            in_dim = h_dim

        # Self-attention for cross-transition reasoning
        self.attn = nn.MultiheadAttention(
            self.hidden_dim, num_heads=4, batch_first=True,
        )
        self.ln = nn.LayerNorm(self.hidden_dim)

        # Output projection
        self.last_fc = nn.Linear(self.hidden_dim, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        # Handle both 2D (batch, feat) and 3D (task, seq, feat) input
        needs_reshape = (input.dim() == 2)
        if needs_reshape:
            input = input.unsqueeze(0)

        task, seq, feat = input.size()
        h = input.view(task * seq, feat)

        # Per-transition MLP embedding
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))
        h = h.view(task, seq, -1)

        # Self-attention + residual + layer norm
        attn_out, _ = self.attn(h, h, h)
        h = self.ln(h + attn_out)

        # Per-transition output (compatible with product-of-Gaussians)
        preactivation = self.last_fc(h)
        output = preactivation

        if needs_reshape:
            output = output.squeeze(0)
            preactivation = preactivation.squeeze(0)

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
        "content": _ATTENTION_ENCODER_CLASS,
    },
]
