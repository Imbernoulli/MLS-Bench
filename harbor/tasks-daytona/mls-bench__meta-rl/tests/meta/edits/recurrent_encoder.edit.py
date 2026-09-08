"""Recurrent encoder baseline — rigorous codebase edit ops.

Replaces the default linear encoder with oyster's PEARL recurrent encoder
layout: per-transition MLP, one-layer LSTM over ordered trajectory context,
then projection from the last hidden state to Gaussian parameters.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "oyster/custom_encoder.py"

# ── 1. Replace encoder class (lines 27-53) ──────────────────────────────

_RECURRENT_ENCODER_CLASS = """\
def _identity(x):
    return x


class CustomContextEncoder(PyTorchModule):
    \"\"\"PEARL recurrent encoder matching oyster.rlkit.torch.networks.\"\"\"
    IS_RECURRENT = True

    def __init__(self, hidden_sizes, input_size, output_size,
                 init_w=3e-3, hidden_activation=F.relu,
                 output_activation=_identity, hidden_init=ptu.fanin_init,
                 b_init_value=0.1, **kwargs):
        self.save_init_params(locals())
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        in_size = input_size
        self.fcs = []
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__(\"fc{}\".format(i), fc)
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

        self.hidden_dim = self.hidden_sizes[-1]
        self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim,
            num_layers=1, batch_first=True,
        )

    def forward(self, input, return_preactivations=False):
        # Oyster's recurrent path supplies ordered context as (task, seq, feat).
        task, seq, feat = input.size()
        out = input.view(task * seq, feat)

        for fc in self.fcs:
            out = self.hidden_activation(fc(out))
        out = out.view(task, seq, -1)

        # Defensive resize: oyster's evaluate() with dump_eval_paths=False
        # never calls clear_z before infer_posterior, leaving hidden sized for
        # the last training meta_batch. Reset when task dim mismatches.
        if self.hidden.size(1) != task:
            self.reset(task)

        zeros = torch.zeros(self.hidden.size()).to(ptu.device)
        out, (hn, cn) = self.lstm(out, (self.hidden, zeros))
        self.hidden = hn
        out = out[:, -1, :]

        preactivation = self.last_fc(out)
        output = self.output_activation(preactivation)

        if return_preactivations:
            return output, preactivation
        return output

    def reset(self, num_tasks=1):
        self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)

"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 27,
        "end_line": 53,
        "content": _RECURRENT_ENCODER_CLASS,
    },
]
