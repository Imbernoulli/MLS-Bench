# MLS-Bench: tdmpc2-simnorm

# Latent Representation Normalization for Model-Based RL

## Objective
Design and implement a custom normalization technique for latent state representations in model-based reinforcement learning. Your code goes in the `CustomSimNorm` class in `custom_simnorm.py`. This normalization is applied as the final activation in both the encoder and dynamics networks of the TD-MPC2 world model.

## Background
**TD-MPC2** (Hansen, Su, Wang, ICLR 2024, arXiv:2310.16828) learns an implicit world model in a latent space and uses it for planning. The latent representation geometry is critical for stable learning. The default approach uses **SimNorm (Simplicial Normalization)**, introduced in the TD-MPC2 paper, which reshapes the latent vector into groups of 8 and applies softmax within each group, constraining each group to lie on a simplex.

Alternative normalization strategies could improve learning stability, representation quality, or computational efficiency:
- **L2 normalization**: projects onto a hypersphere.
- **RMSNorm**: root-mean-square normalization without mean centering.
- **Spectral normalization**: controls the Lipschitz constant.
- **Gumbel-softmax**: adds stochasticity to the simplex projection.
- **Hybrid approaches**: combining different normalization strategies.

## What You Can Modify
The `CustomSimNorm` class in `custom_simnorm.py`:
- `__init__(self, cfg)`: initialize parameters (`cfg.simnorm_dim = 8`)
- `forward(self, x)`: normalize the latent vector (must preserve shape)

## Evaluation
- **Metric**: episode reward (higher is better)
- **Environments**: DMControl walker-walk and cheetah-run
- **Model**: TD-MPC2 with 1M parameters, 200K training steps

## Architecture Context
The normalization is used in:
1. **Encoder** (`layers.py: enc()`): maps observations to latent states.
2. **Dynamics** (`world_model.py: __init__`): predicts next latent state from current state + action.

Both use SimNorm as the final activation in their MLP stacks. The latent dimension is 128 with `simnorm_dim = 8` (16 groups).


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/tdmpc2/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `tdmpc2/tdmpc2/common/custom_simnorm.py`
- editable lines **16–43**


Other files you may **read** for context (do not modify):
- `tdmpc2/tdmpc2/common/layers.py`
- `tdmpc2/tdmpc2/common/world_model.py`


## Readable Context


### `tdmpc2/tdmpc2/common/custom_simnorm.py`  [EDITABLE — lines 16–43 only]

```python
     1: """Custom latent normalization for TD-MPC2 world model.
     2: 
     3: Replace the body of CustomSimNorm with your normalization implementation.
     4: The class is used as the final activation in the encoder and dynamics
     5: networks, constraining the latent representation geometry.
     6: """
     7: 
     8: import torch
     9: import torch.nn as nn
    10: import torch.nn.functional as F
    11: 
    12: 
    13: # =====================================================================
    14: # EDITABLE: Custom latent normalization
    15: # =====================================================================
    16: class CustomSimNorm(nn.Module):
    17:     """Custom normalization for latent state representations in world models.
    18: 
    19:     Interface contract (same as SimNorm):
    20:         __init__(cfg)  -- cfg.simnorm_dim is the group size (default: 8)
    21:         forward(x: Tensor) -> Tensor  (same shape as input)
    22: 
    23:     The input tensor has shape (*batch_dims, latent_dim) where latent_dim
    24:     is divisible by simnorm_dim. Your normalization should constrain the
    25:     geometry of the latent space to improve world model learning.
    26: 
    27:     Evaluated on DMControl walker-walk and cheetah-run tasks.
    28:     """
    29: 
    30:     def __init__(self, cfg):
    31:         super().__init__()
    32:         self.dim = cfg.simnorm_dim
    33: 
    34:     def forward(self, x):
    35:         # Default: SimNorm (simplicial normalization)
    36:         # Reshape into groups of size self.dim and apply softmax
    37:         shp = x.shape
    38:         x = x.view(*shp[:-1], -1, self.dim)
    39:         x = F.softmax(x, dim=-1)
    40:         return x.view(*shp)
    41: 
    42:     def __repr__(self):
    43:         return f"CustomSimNorm(dim={self.dim})"
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `simnorm` baseline — editable region  [READ-ONLY — reference implementation]

In `tdmpc2/tdmpc2/common/custom_simnorm.py`:

```python
Lines 16–30:
    13: # =====================================================================
    14: # EDITABLE: Custom latent normalization
    15: # =====================================================================
    16: class CustomSimNorm(nn.Module):
    17:     """SimNorm baseline -- original simplicial normalization from TD-MPC2."""
    18: 
    19:     def __init__(self, cfg):
    20:         super().__init__()
    21:         self.dim = cfg.simnorm_dim
    22: 
    23:     def forward(self, x):
    24:         shp = x.shape
    25:         x = x.view(*shp[:-1], -1, self.dim)
    26:         x = F.softmax(x, dim=-1)
    27:         return x.view(*shp)
    28: 
    29:     def __repr__(self):
    30:         return f"CustomSimNorm(dim={self.dim}, type=SimNorm)"
```

### `l2norm` baseline — editable region  [READ-ONLY — reference implementation]

In `tdmpc2/tdmpc2/common/custom_simnorm.py`:

```python
Lines 16–32:
    13: # =====================================================================
    14: # EDITABLE: Custom latent normalization
    15: # =====================================================================
    16: class CustomSimNorm(nn.Module):
    17:     """L2 normalization baseline -- projects latent vectors onto a hypersphere."""
    18: 
    19:     def __init__(self, cfg):
    20:         super().__init__()
    21:         self.dim = cfg.simnorm_dim
    22:         self.eps = 1e-8
    23:         # Learnable scale parameter
    24:         self.scale = nn.Parameter(torch.ones(1))
    25: 
    26:     def forward(self, x):
    27:         # L2 normalize across the last dimension and apply learnable scale
    28:         norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=self.eps)
    29:         return self.scale * x / norm
    30: 
    31:     def __repr__(self):
    32:         return f"CustomSimNorm(dim={self.dim}, type=L2Norm)"
```

### `rmsnorm` baseline — editable region  [READ-ONLY — reference implementation]

In `tdmpc2/tdmpc2/common/custom_simnorm.py`:

```python
Lines 16–36:
    13: # =====================================================================
    14: # EDITABLE: Custom latent normalization
    15: # =====================================================================
    16: class CustomSimNorm(nn.Module):
    17:     """Group-wise RMSNorm baseline for latent representations."""
    18: 
    19:     def __init__(self, cfg):
    20:         super().__init__()
    21:         self.dim = cfg.simnorm_dim
    22:         self.eps = 1e-8
    23:         # Learnable gain per group element
    24:         self.weight = nn.Parameter(torch.ones(self.dim))
    25: 
    26:     def forward(self, x):
    27:         shp = x.shape
    28:         # Reshape into groups (same as SimNorm)
    29:         x = x.view(*shp[:-1], -1, self.dim)
    30:         # RMS normalization within each group
    31:         rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
    32:         x = (x / rms) * self.weight
    33:         return x.view(*shp)
    34: 
    35:     def __repr__(self):
    36:         return f"CustomSimNorm(dim={self.dim}, type=RMSNorm)"
```

### `identity` baseline — editable region  [READ-ONLY — reference implementation]

In `tdmpc2/tdmpc2/common/custom_simnorm.py`:

```python
Lines 16–27:
    13: # =====================================================================
    14: # EDITABLE: Custom latent normalization
    15: # =====================================================================
    16: class CustomSimNorm(nn.Module):
    17:     """Identity baseline -- no normalization applied to latent representations."""
    18: 
    19:     def __init__(self, cfg):
    20:         super().__init__()
    21:         self.dim = cfg.simnorm_dim
    22: 
    23:     def forward(self, x):
    24:         return x
    25: 
    26:     def __repr__(self):
    27:         return f"CustomSimNorm(dim={self.dim}, type=Identity)"
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
