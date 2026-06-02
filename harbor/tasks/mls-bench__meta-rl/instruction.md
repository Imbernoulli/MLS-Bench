# MLS-Bench: meta-rl

# Meta-RL: Context Encoder for PEARL Task Inference

## Research Question
Design a context encoder for the PEARL meta-reinforcement learning
algorithm that maps transition tuples `(state, action, reward, next_state)`
to latent task representations. The encoder should enable effective task
inference from limited interaction data so that the agent can adapt
quickly to unseen tasks.

## Background
PEARL (Probabilistic Embeddings for Actor-Critic RL), introduced in
Rakelly et al., "Efficient Off-Policy Meta-Reinforcement Learning via
Probabilistic Context Variables" (arXiv:1903.08254, ICML 2019), is a
meta-RL algorithm that learns a probabilistic latent task variable `z`
from context transitions. During meta-testing, the agent collects a few
transitions from a new task, encodes them into a posterior distribution
`q(z|c)`, and conditions an SAC-style policy on samples of `z`.

The context encoder processes individual transition tuples and outputs
Gaussian parameters (mean and log-variance). The PEARLAgent aggregates
per-transition outputs via product of Gaussians to form the task
posterior. The SAC policy/value backbone, replay buffers, task sampling,
and outer training loop are all fixed; only the encoder architecture is
open.

You will modify the `CustomContextEncoder` class and may add custom
imports inside the editable region of `custom_encoder.py`.

## Interface
Your `CustomContextEncoder` must:
- Extend `PyTorchModule` and call `self.save_init_params(locals())` in
  `__init__`.
- Accept `hidden_sizes`, `input_size`, `output_size` as constructor
  arguments.
- Set the `self.output_size` attribute in `__init__`.
- Implement `forward(self, input, return_preactivations=False)` returning
  a tensor of shape `(*, output_size)`.
- Implement `reset(self, num_tasks=1)` to reset any stateful components
  such as recurrent hidden state.

## Reference Architectures
- **MLP encoder** — independent per-transition MLP (the original PEARL
  encoder; Rakelly et al., 2019).
- **Recurrent encoder** — GRU over the context sequence in the spirit of
  VariBAD (Zintgraf et al., "VariBAD: A Very Good Method for Bayes-Adaptive
  Deep RL via Meta-Learning", arXiv:1910.08348, ICLR 2020).
- **Attention encoder** — a small Transformer-style aggregator over the
  context tuples.

## Environments
The encoder is evaluated across MuJoCo and point-robot meta-RL task
families with different reward structures:

1. **Half-Cheetah Velocity** (`cheetah-vel`): 30 train / 10 test tasks,
   target velocities in `[0, 3]` m/s. Obs dim 20, action dim 6. Dense
   reward based on velocity matching. Tests encoding quality on a
   continuous task distribution with high-dimensional observations.

2. **Sparse Point Robot** (`sparse-point-robot`): 40 train / 10 test
   tasks. Goals on a half-circle, sparse reward (+1 within goal radius,
   0 otherwise). Obs dim 2, action dim 2. Tests the encoder's ability to
   extract task information from sparse reward signals.

3. **Point Robot** (`point-robot`): 40 train / 10 test tasks. Goals
   sampled uniformly from `[-1, 1]^2`. Dense reward (negative L2 distance
   to goal). Obs dim 2, action dim 2. A simpler diverse continuous task
   distribution.

## Evaluation
Performance is measured by `meta_test_return` on each environment:
average return on held-out test tasks after meta-training under this
benchmark's fixed budget. Higher is better.

## Note on Training Budget
This task intentionally uses a short fixed meta-training budget (20 outer
iterations) to keep wall time per environment near 1 hour. This is far
shorter than the 500+ iteration budgets used in the PEARL/VariBAD/FOCAL
papers (roughly 1.5e6–2.0e6 environment steps), so absolute returns are
not directly comparable to those papers; only relative ordering across
baselines and agents within this fixed budget is meaningful.

On `sparse-point-robot`, methods that report 0 indicate no goal was
reached within the budget rather than algorithmic failure, since the
environment reward is binary.

The companion [`meta-rl-algorithm`](../meta-rl-algorithm/task_description.md)
task uses the same budget convention.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/oyster/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `oyster/custom_encoder.py`
- editable lines **21–23**
- editable lines **27–53**


Other files you may **read** for context (do not modify):
- `oyster/launch_custom.py`
- `oyster/rlkit/torch/networks.py`
- `oyster/rlkit/torch/sac/agent.py`
- `oyster/rlkit/torch/sac/sac.py`
- `oyster/rlkit/core/rl_algorithm.py`
- `oyster/configs/default.py`


## Readable Context


### `oyster/custom_encoder.py`  [EDITABLE — lines 21–23, lines 27–53 only]

```python
     1: """Custom context encoder for PEARL meta-RL.
     2: 
     3: Encodes transition tuples (s, a, r [, s']) into latent representations
     4: for task inference. The PEARLAgent calls this encoder and aggregates
     5: per-transition outputs via product of Gaussians to form the task posterior.
     6: 
     7: Interface requirements:
     8:   - __init__(hidden_sizes, input_size, output_size, **kwargs)
     9:   - forward(input) -> output of shape (*, output_size)
    10:   - reset(num_tasks) -> None (reset stateful components)
    11:   - Must set self.output_size attribute in __init__
    12:   - Must extend PyTorchModule (call self.save_init_params(locals()) first)
    13: """
    14: 
    15: import torch
    16: import torch.nn as nn
    17: import torch.nn.functional as F
    18: 
    19: from rlkit.torch.core import PyTorchModule
    20: import rlkit.torch.pytorch_util as ptu
    21: # ── Custom imports (editable) ────────────────────────────────────────────
    22: 
    23: 
    24: # ======================================================================
    25: # EDITABLE — Custom context encoder
    26: # ======================================================================
    27: class CustomContextEncoder(PyTorchModule):
    28:     """Context encoder for PEARL meta-RL task inference.
    29: 
    30:     Input:  (*, input_size) transition features (obs, action, reward [, next_obs])
    31:     Output: (*, output_size) Gaussian parameters (mean and log_variance)
    32:     """
    33:     def __init__(self, hidden_sizes, input_size, output_size,
    34:                  init_w=3e-3, hidden_activation=F.relu, **kwargs):
    35:         self.save_init_params(locals())
    36:         super().__init__()
    37:         self.input_size = input_size
    38:         self.output_size = output_size
    39: 
    40:         # Default: single linear layer (baseline placeholder)
    41:         self.fc = nn.Linear(input_size, output_size)
    42:         self.fc.weight.data.uniform_(-init_w, init_w)
    43:         self.fc.bias.data.uniform_(-init_w, init_w)
    44: 
    45:     def forward(self, input, return_preactivations=False):
    46:         output = self.fc(input)
    47:         if return_preactivations:
    48:             return output, output
    49:         return output
    50: 
    51:     def reset(self, num_tasks=1):
    52:         pass
    53: 
```

## Parameter Budget

This task enforces a parameter-count cap. Your edits will be rejected if
the resulting model exceeds **1.05×** the strongest
baseline's parameter count. The check runs automatically inside the eval
scripts — you don't need to invoke it.

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `mlp_encoder` baseline — editable region  [READ-ONLY — reference implementation]

In `oyster/custom_encoder.py`:

```python
Lines 21–23:
    18: 
    19: from rlkit.torch.core import PyTorchModule
    20: import rlkit.torch.pytorch_util as ptu
    21: # ── Custom imports (editable) ────────────────────────────────────────────
    22: 
    23: 
    24: # ======================================================================
    25: # EDITABLE — Custom context encoder
    26: # ======================================================================

Lines 27–61:
    24: # ======================================================================
    25: # EDITABLE — Custom context encoder
    26: # ======================================================================
    27: class CustomContextEncoder(PyTorchModule):
    28:     """Original PEARL MLP context encoder (3-layer, 200 units)."""
    29:     def __init__(self, hidden_sizes, input_size, output_size,
    30:                  init_w=3e-3, hidden_activation=F.relu, **kwargs):
    31:         self.save_init_params(locals())
    32:         super().__init__()
    33:         self.input_size = input_size
    34:         self.output_size = output_size
    35:         self.hidden_activation = hidden_activation
    36: 
    37:         in_dim = input_size
    38:         self.fcs = nn.ModuleList()
    39:         for h_dim in hidden_sizes:
    40:             fc = nn.Linear(in_dim, h_dim)
    41:             ptu.fanin_init(fc.weight)
    42:             fc.bias.data.fill_(0.1)
    43:             self.fcs.append(fc)
    44:             in_dim = h_dim
    45:         self.last_fc = nn.Linear(in_dim, output_size)
    46:         self.last_fc.weight.data.uniform_(-init_w, init_w)
    47:         self.last_fc.bias.data.uniform_(-init_w, init_w)
    48: 
    49:     def forward(self, input, return_preactivations=False):
    50:         h = input
    51:         for fc in self.fcs:
    52:             h = self.hidden_activation(fc(h))
    53:         preactivation = self.last_fc(h)
    54:         output = preactivation
    55:         if return_preactivations:
    56:             return output, preactivation
    57:         return output
    58: 
    59:     def reset(self, num_tasks=1):
    60:         pass
    61: 
```

### `recurrent_encoder` baseline — editable region  [READ-ONLY — reference implementation]

In `oyster/custom_encoder.py`:

```python
Lines 21–23:
    18: 
    19: from rlkit.torch.core import PyTorchModule
    20: import rlkit.torch.pytorch_util as ptu
    21: # ── Custom imports (editable) ────────────────────────────────────────────
    22: 
    23: 
    24: # ======================================================================
    25: # EDITABLE — Custom context encoder
    26: # ======================================================================

Lines 27–97:
    24: # ======================================================================
    25: # EDITABLE — Custom context encoder
    26: # ======================================================================
    27: def _identity(x):
    28:     return x
    29: 
    30: 
    31: class CustomContextEncoder(PyTorchModule):
    32:     """PEARL recurrent encoder matching oyster.rlkit.torch.networks."""
    33:     IS_RECURRENT = True
    34: 
    35:     def __init__(self, hidden_sizes, input_size, output_size,
    36:                  init_w=3e-3, hidden_activation=F.relu,
    37:                  output_activation=_identity, hidden_init=ptu.fanin_init,
    38:                  b_init_value=0.1, **kwargs):
    39:         self.save_init_params(locals())
    40:         super().__init__()
    41:         self.input_size = input_size
    42:         self.output_size = output_size
    43:         self.hidden_sizes = hidden_sizes
    44:         self.hidden_activation = hidden_activation
    45:         self.output_activation = output_activation
    46: 
    47:         in_size = input_size
    48:         self.fcs = []
    49:         for i, next_size in enumerate(hidden_sizes):
    50:             fc = nn.Linear(in_size, next_size)
    51:             in_size = next_size
    52:             hidden_init(fc.weight)
    53:             fc.bias.data.fill_(b_init_value)
    54:             self.__setattr__("fc{}".format(i), fc)
    55:             self.fcs.append(fc)
    56: 
    57:         self.last_fc = nn.Linear(in_size, output_size)
    58:         self.last_fc.weight.data.uniform_(-init_w, init_w)
    59:         self.last_fc.bias.data.uniform_(-init_w, init_w)
    60: 
    61:         self.hidden_dim = self.hidden_sizes[-1]
    62:         self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))
    63:         self.lstm = nn.LSTM(
    64:             self.hidden_dim, self.hidden_dim,
    65:             num_layers=1, batch_first=True,
    66:         )
    67: 
    68:     def forward(self, input, return_preactivations=False):
    69:         # Oyster's recurrent path supplies ordered context as (task, seq, feat).
    70:         task, seq, feat = input.size()
    71:         out = input.view(task * seq, feat)
    72: 
    73:         for fc in self.fcs:
    74:             out = self.hidden_activation(fc(out))
    75:         out = out.view(task, seq, -1)
    76: 
    77:         # Defensive resize: oyster's evaluate() with dump_eval_paths=False
    78:         # never calls clear_z before infer_posterior, leaving hidden sized for
    79:         # the last training meta_batch. Reset when task dim mismatches.
    80:         if self.hidden.size(1) != task:
    81:             self.reset(task)
    82: 
    83:         zeros = torch.zeros(self.hidden.size()).to(ptu.device)
    84:         out, (hn, cn) = self.lstm(out, (self.hidden, zeros))
    85:         self.hidden = hn
    86:         out = out[:, -1, :]
    87: 
    88:         preactivation = self.last_fc(out)
    89:         output = self.output_activation(preactivation)
    90: 
    91:         if return_preactivations:
    92:             return output, preactivation
    93:         return output
    94: 
    95:     def reset(self, num_tasks=1):
    96:         self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)
    97: 
```

### `attention_encoder` baseline — editable region  [READ-ONLY — reference implementation]

In `oyster/custom_encoder.py`:

```python
Lines 21–23:
    18: 
    19: from rlkit.torch.core import PyTorchModule
    20: import rlkit.torch.pytorch_util as ptu
    21: # ── Custom imports (editable) ────────────────────────────────────────────
    22: 
    23: 
    24: # ======================================================================
    25: # EDITABLE — Custom context encoder
    26: # ======================================================================

Lines 27–91:
    24: # ======================================================================
    25: # EDITABLE — Custom context encoder
    26: # ======================================================================
    27: class CustomContextEncoder(PyTorchModule):
    28:     """Self-attention context encoder for cross-transition reasoning."""
    29:     def __init__(self, hidden_sizes, input_size, output_size,
    30:                  init_w=3e-3, hidden_activation=F.relu, **kwargs):
    31:         self.save_init_params(locals())
    32:         super().__init__()
    33:         self.input_size = input_size
    34:         self.output_size = output_size
    35:         self.hidden_activation = hidden_activation
    36:         self.hidden_dim = hidden_sizes[-1]
    37: 
    38:         # Per-transition MLP embedding
    39:         in_dim = input_size
    40:         self.fcs = nn.ModuleList()
    41:         for h_dim in hidden_sizes:
    42:             fc = nn.Linear(in_dim, h_dim)
    43:             ptu.fanin_init(fc.weight)
    44:             fc.bias.data.fill_(0.1)
    45:             self.fcs.append(fc)
    46:             in_dim = h_dim
    47: 
    48:         # Self-attention for cross-transition reasoning
    49:         self.attn = nn.MultiheadAttention(
    50:             self.hidden_dim, num_heads=4, batch_first=True,
    51:         )
    52:         self.ln = nn.LayerNorm(self.hidden_dim)
    53: 
    54:         # Output projection
    55:         self.last_fc = nn.Linear(self.hidden_dim, output_size)
    56:         self.last_fc.weight.data.uniform_(-init_w, init_w)
    57:         self.last_fc.bias.data.uniform_(-init_w, init_w)
    58: 
    59:     def forward(self, input, return_preactivations=False):
    60:         # Handle both 2D (batch, feat) and 3D (task, seq, feat) input
    61:         needs_reshape = (input.dim() == 2)
    62:         if needs_reshape:
    63:             input = input.unsqueeze(0)
    64: 
    65:         task, seq, feat = input.size()
    66:         h = input.view(task * seq, feat)
    67: 
    68:         # Per-transition MLP embedding
    69:         for fc in self.fcs:
    70:             h = self.hidden_activation(fc(h))
    71:         h = h.view(task, seq, -1)
    72: 
    73:         # Self-attention + residual + layer norm
    74:         attn_out, _ = self.attn(h, h, h)
    75:         h = self.ln(h + attn_out)
    76: 
    77:         # Per-transition output (compatible with product-of-Gaussians)
    78:         preactivation = self.last_fc(h)
    79:         output = preactivation
    80: 
    81:         if needs_reshape:
    82:             output = output.squeeze(0)
    83:             preactivation = preactivation.squeeze(0)
    84: 
    85:         if return_preactivations:
    86:             return output, preactivation
    87:         return output
    88: 
    89:     def reset(self, num_tasks=1):
    90:         pass
    91: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
