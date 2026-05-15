"""Custom context encoder for PEARL meta-RL.

Encodes transition tuples (s, a, r [, s']) into latent representations
for task inference. The PEARLAgent calls this encoder and aggregates
per-transition outputs via product of Gaussians to form the task posterior.

Interface requirements:
  - __init__(hidden_sizes, input_size, output_size, **kwargs)
  - forward(input) -> output of shape (*, output_size)
  - reset(num_tasks) -> None (reset stateful components)
  - Must set self.output_size attribute in __init__
  - Must extend PyTorchModule (call self.save_init_params(locals()) first)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlkit.torch.core import PyTorchModule
import rlkit.torch.pytorch_util as ptu
# ── Custom imports (editable) ────────────────────────────────────────────


# ======================================================================
# EDITABLE — Custom context encoder
# ======================================================================
class CustomContextEncoder(PyTorchModule):
    """Context encoder for PEARL meta-RL task inference.

    Input:  (*, input_size) transition features (obs, action, reward [, next_obs])
    Output: (*, output_size) Gaussian parameters (mean and log_variance)
    """
    def __init__(self, hidden_sizes, input_size, output_size,
                 init_w=3e-3, hidden_activation=F.relu, **kwargs):
        self.save_init_params(locals())
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Default: single linear layer (baseline placeholder)
        self.fc = nn.Linear(input_size, output_size)
        self.fc.weight.data.uniform_(-init_w, init_w)
        self.fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        output = self.fc(input)
        if return_preactivations:
            return output, output
        return output

    def reset(self, num_tasks=1):
        pass

