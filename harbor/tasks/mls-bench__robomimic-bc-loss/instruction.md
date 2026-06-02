# MLS-Bench: robomimic-bc-loss

# Behavioral Cloning: Loss Function Design for Robot Imitation Learning

## Research Question
Design an improved loss function for GMM-based behavioral cloning (BC) in robot manipulation. The policy outputs a Gaussian Mixture Model (GMM) distribution over actions, and the loss function receives this distribution along with expert demonstration actions. Your goal is to design a loss that maximizes imitation learning quality on robot manipulation tasks.

## Background
The training and evaluation pipeline follows **robomimic** (Mandlekar et al., CoRL 2021, arXiv:2108.03298), the standard imitation-learning study and codebase for robot manipulation from offline human demonstrations. The default GMM-BC objective is the negative log-likelihood (NLL) of the expert action under the predicted mixture:

```
loss = -dist.log_prob(target_actions).mean()
```

NLL is convenient but ignores structure such as which mixture component is responsible for the target action, the shape of action errors (e.g. SE(3) end-effector vs. gripper bit), and the relative weight of low- vs. high-probability components. Alternative losses (e.g. cross-entropy on assignments, robust regression on the mean component, mixture-aware terms) may better exploit demonstration data.

## What You Can Modify
The `CustomBCLoss` class in `custom_bc_loss.py`. This class receives a GMM distribution and target action tensors and must return a scalar loss.

Interface:
- **Input**: `dist` (a `torch.distributions.MixtureSameFamily` GMM distribution with 5 modes) and `target_actions: [B, 7]` — 7-dim robot actions (6D end-effector delta + 1D gripper)
- **Output**: scalar loss tensor
- The default implementation is negative log-likelihood: `-dist.log_prob(target_actions).mean()`

You may add parameters to `__init__`, define helper methods, and use any PyTorch operations. The `dist` object supports `.log_prob()`, `.sample()`, `.component_distribution`, and `.mixture_distribution`.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/robomimic/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `robomimic/custom_bc_loss.py`
- editable lines **20–41**




## Readable Context


### `robomimic/custom_bc_loss.py`  [EDITABLE — lines 20–41 only]

```python
     1: """
     2: Custom BC Loss Function for GMM-based Behavioral Cloning.
     3: 
     4: This module defines the loss function used by BC-GMM training in robomimic.
     5: The loss receives the GMM distribution produced by the policy network and
     6: the expert demonstration actions, and returns a scalar loss.
     7: 
     8: The custom loss is imported and used by the patched BC_GMM._compute_losses
     9: method during training.
    10: """
    11: 
    12: import torch
    13: import torch.nn as nn
    14: import torch.nn.functional as F
    15: import torch.distributions as D
    16: 
    17: 
    18: # ── Custom BC Loss Function ────────────────────────────────────────────────
    19: # EDITABLE REGION START
    20: class CustomBCLoss(nn.Module):
    21:     """Custom loss for GMM-based behavioral cloning.
    22: 
    23:     Called during BC-GMM training. Receives the full GMM distribution and
    24:     expert actions, returns a scalar loss to minimize.
    25: 
    26:     Args:
    27:         dist: MixtureSameFamily GMM distribution (5 modes, 7-dim actions)
    28:             Supports: .log_prob(), .sample(), .component_distribution,
    29:                       .mixture_distribution
    30:         target_actions: [B, 7] expert actions
    31: 
    32:     Returns:
    33:         scalar loss tensor
    34:     """
    35: 
    36:     def __init__(self, action_dim=7):
    37:         super().__init__()
    38:         self.action_dim = action_dim
    39: 
    40:     def forward(self, dist, target_actions):
    41:         return -dist.log_prob(target_actions).mean()
    42: # EDITABLE REGION END
```


## Adapter Warnings

Some reference context could not be rendered completely:

- `default` has no edit_ops entry

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `nll_entropy` baseline — editable region  [READ-ONLY — reference implementation]

In `robomimic/custom_bc_loss.py`:

```python
Lines 20–39:
    17: 
    18: # ── Custom BC Loss Function ────────────────────────────────────────────────
    19: # EDITABLE REGION START
    20: class CustomBCLoss(nn.Module):
    21:     """NLL + entropy bonus for GMM-based behavioral cloning.
    22: 
    23:     Adds an entropy regularization term to the standard NLL loss.
    24:     The entropy bonus encourages broader mixture components,
    25:     preventing premature mode collapse.
    26:     """
    27: 
    28:     def __init__(self, action_dim=7, alpha=0.01):
    29:         super().__init__()
    30:         self.action_dim = action_dim
    31:         self.alpha = alpha
    32: 
    33:     def forward(self, dist, target_actions):
    34:         nll = -dist.log_prob(target_actions).mean()
    35:         # MixtureSameFamily has no closed-form entropy; approximate via sampling
    36:         with torch.no_grad():
    37:             samples = dist.sample()
    38:         entropy = -dist.log_prob(samples).mean()
    39:         return nll - self.alpha * entropy
    40: # EDITABLE REGION END
```

### `weighted_nll` baseline — editable region  [READ-ONLY — reference implementation]

In `robomimic/custom_bc_loss.py`:

```python
Lines 20–39:
    17: 
    18: # ── Custom BC Loss Function ────────────────────────────────────────────────
    19: # EDITABLE REGION START
    20: class CustomBCLoss(nn.Module):
    21:     """Dimension-weighted NLL for GMM-based behavioral cloning.
    22: 
    23:     Weights positional action dimensions more heavily than the gripper
    24:     dimension to prioritize accurate end-effector movement prediction.
    25:     """
    26: 
    27:     def __init__(self, action_dim=7, pos_weight=2.0, grip_weight=1.0):
    28:         super().__init__()
    29:         self.action_dim = action_dim
    30:         weights = torch.ones(action_dim)
    31:         weights[:6] = pos_weight
    32:         weights[6:] = grip_weight
    33:         self.register_buffer('weights', weights)
    34: 
    35:     def forward(self, dist, target_actions):
    36:         # Weight targets for importance scaling
    37:         weighted_targets = target_actions * self.weights.unsqueeze(0)
    38:         nll = -dist.log_prob(weighted_targets)
    39:         return nll.mean()
    40: # EDITABLE REGION END
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
