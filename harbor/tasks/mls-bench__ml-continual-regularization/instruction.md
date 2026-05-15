# MLS-Bench: ml-continual-regularization

# Continual Learning: Regularization Strategy Optimization

## Research Question
Design a regularization strategy that mitigates catastrophic forgetting in continual learning. The contribution is the *importance estimator* (which parameters matter for each past context) and the *penalty form* (how their changes are penalized while training on later contexts), implemented within the fixed training loop.

## Background
A continual learner trains on a sequence of contexts and must retain performance on earlier ones. Regularization-based methods add `reg_strength * R(theta)` to the per-step loss, penalizing changes to parameters important for previous tasks.

Reference baselines:
- **EWC (Elastic Weight Consolidation)** — Kirkpatrick et al., PNAS 2017 ([arXiv:1612.00796](https://arxiv.org/abs/1612.00796)). Importance = diagonal Fisher Information `F` at the post-training parameter `theta*`. Penalty: `0.5 * sum_i F_i * (theta_i - theta_i*)^2`. Fishers are stored separately per past context (memory grows with context count).
- **SI (Synaptic Intelligence)** — Zenke, Poole, Ganguli, ICML 2017 ([arXiv:1703.04200](https://arxiv.org/abs/1703.04200)). Online importance: at each step accumulate `omega_i ≈ sum (-grad_i * delta_theta_i)`, normalized by total drift `(theta_i - theta_i*)^2 + epsilon` after the context. Penalty: `sum_i omega_i * (theta_i - theta_i*)^2`.
- **Online EWC** — Schwarz et al., ICML 2018 ([arXiv:1805.06370](https://arxiv.org/abs/1805.06370)). Replace the per-task list of Fishers with a single running estimate: `F <- gamma * F_old + F_new`, with `gamma < 1` (often 0.9). Constant memory in number of contexts.

## Implementation Contract
Implement two functions in `continual-learning/custom_regularization.py`:

```python
def estimate_importance(model, dataset, prev_params, device) -> dict:
    """
    Called once after training on each context finishes.
    Returns a dict {param_name: importance_tensor} (same shapes as the params).
    May do forward/backward passes over `dataset`.
    """

def compute_regularization_loss(model, importance_dict, prev_params_dict) -> Tensor:
    """
    Called at every training step. Must be efficient.
    Returns a scalar tensor — the regularization penalty added to the task loss.
    """
```

Available hooks on the framework:
- `model.param_list` — list of generators yielding `(name, param)` over regularized parameters.
- `model._custom_W` — dict tracking per-step gradient-weighted parameter changes (accumulated by the training loop). Useful for SI-style importance.
- `model._custom_p_old` — dict of parameter snapshots from the previous training step.
- `model.gamma` — decay factor for Fisher accumulation (framework default 1.0; Online EWC typically ≈ 0.9).
- `model.epsilon` — damping constant (default 0.1, used by SI).

Constraints: only modify the editable region of `custom_regularization.py`; do not create new files.

## Fixed Pipeline & Evaluation
Three benchmarks:

| Benchmark | Scenario | Contexts | Description |
|-----------|----------|----------|-------------|
| **Split-MNIST** | Task-incremental | 5 (2 classes each) | MNIST digits split into 5 binary tasks. |
| **Permuted-MNIST** | Domain-incremental | 10 | Same digit classes; each context applies a different fixed pixel permutation. |
| **Split-CIFAR100** | Task-incremental | 10 (10 classes each) | CIFAR-100 split into 10 ten-way tasks. |

Primary metric: **average accuracy across all contexts after training completes** (higher is better).


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/continual-learning/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `continual-learning/custom_regularization.py`
- editable lines **25–115**
- editable lines **117–119**


Other files you may **read** for context (do not modify):
- `continual-learning/models/cl/continual_learner.py`
- `continual-learning/models/classifier.py`
- `continual-learning/train/train_task_based.py`
- `continual-learning/main.py`
- `continual-learning/eval/evaluate.py`


## Readable Context


### `continual-learning/custom_regularization.py`  [EDITABLE — lines 25–115, lines 117–119 only]

```python
     1: """Custom regularization module for continual learning.
     2: 
     3: This module provides two core functions that control how a continual learning
     4: model prevents catastrophic forgetting via parameter regularization:
     5: 
     6:   1. estimate_importance() — called once after each context finishes training
     7:   2. compute_regularization_loss() — called at every training step
     8: 
     9: The model object may have the following attributes set by the training loop:
    10:   - model._custom_importance: dict mapping param_name -> accumulated importance tensor
    11:   - model._custom_prev_params: dict mapping param_name -> param snapshot tensor
    12:   - model._custom_W: dict for per-step accumulation (available during training)
    13:   - model._custom_p_old: dict for per-step old params (available during training)
    14: 
    15: You may also attach new attributes to the model object as needed.
    16: """
    17: 
    18: import torch
    19: import torch.nn.functional as F
    20: from torch.utils.data import DataLoader
    21: 
    22: 
    23: # ======================================================================
    24: # EDITABLE REGION START — estimate_importance
    25: # ======================================================================
    26: def estimate_importance(model, dataset, prev_params, device):
    27:     """Estimate per-parameter importance after training on a context.
    28: 
    29:     Called once after each context's training completes. The returned
    30:     importance dict is accumulated (summed) across contexts by the
    31:     training loop and stored in model._custom_importance.
    32: 
    33:     Args:
    34:         model: The neural network (nn.Module with a ``param_list`` attribute).
    35:                Use model.param_list to get generators for named_parameters.
    36:         dataset: Training dataset of the just-completed context.
    37:         prev_params: Dict mapping param_name -> param tensor from before
    38:                      training on this context started.
    39:         device: torch.device to use.
    40: 
    41:     Returns:
    42:         importance: Dict mapping param_name -> importance tensor (same shape
    43:                     as the parameter). Higher values mean the parameter is
    44:                     more important for the completed context.
    45:     """
    46:     # Default: diagonal Fisher Information (EWC-style)
    47:     est_fisher = {}
    48:     for gen_params in model.param_list:
    49:         for n, p in gen_params():
    50:             if p.requires_grad:
    51:                 n = n.replace('.', '__')
    52:                 est_fisher[n] = p.detach().clone().zero_()
    53: 
    54:     mode = model.training
    55:     model.eval()
    56: 
    57:     data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    58:     n_samples = min(len(data_loader), 200)
    59: 
    60:     for idx, (x, y) in enumerate(data_loader):
    61:         if idx >= n_samples:
    62:             break
    63:         x = x.to(device)
    64:         output = model(x)
    65:         with torch.no_grad():
    66:             label_weights = F.softmax(output, dim=1)
    67:         for label_index in range(output.shape[1]):
    68:             label = torch.LongTensor([label_index]).to(device)
    69:             negloglikelihood = F.cross_entropy(output, label)
    70:             model.zero_grad()
    71:             negloglikelihood.backward(
    72:                 retain_graph=True if (label_index + 1) < output.shape[1] else False
    73:             )
    74:             for gen_params in model.param_list:
    75:                 for n, p in gen_params():
    76:                     if p.requires_grad:
    77:                         n = n.replace('.', '__')
    78:                         if p.grad is not None:
    79:                             est_fisher[n] += label_weights[0][label_index] * (p.grad.detach() ** 2)
    80: 
    81:     est_fisher = {n: v / max(n_samples, 1) for n, v in est_fisher.items()}
    82: 
    83:     model.train(mode=mode)
    84:     return est_fisher
    85: 
    86: 
    87: def compute_regularization_loss(model, importance_dict, prev_params_dict):
    88:     """Compute regularization loss to prevent catastrophic forgetting.
    89: 
    90:     Called at every training step during forward pass.
    91: 
    92:     Args:
    93:         model: Current model (nn.Module with ``param_list``).
    94:         importance_dict: Dict from estimate_importance, accumulated across
    95:                          contexts (summed). Maps param_name -> importance tensor.
    96:         prev_params_dict: Dict of parameter snapshots taken after the last
    97:                           context finished. Maps param_name -> tensor.
    98: 
    99:     Returns:
   100:         loss: Scalar regularization loss (torch scalar tensor).
   101:     """
   102:     # Default: EWC quadratic penalty
   103:     losses = []
   104:     for gen_params in model.param_list:
   105:         for n, p in gen_params():
   106:             if p.requires_grad:
   107:                 n = n.replace('.', '__')
   108:                 if n in importance_dict and n in prev_params_dict:
   109:                     fisher = importance_dict[n]
   110:                     prev = prev_params_dict[n]
   111:                     losses.append((fisher * (p - prev) ** 2).sum())
   112:     if losses:
   113:         return 0.5 * sum(losses)
   114:     return torch.tensor(0.0, device=next(model.parameters()).device)
   115: 
   116: 
   117: # CONFIG_OVERRIDES: override training hyperparameters for your method.
   118: # Allowed keys: reg_strength_scale (multiplier on the per-benchmark reg_strength).
   119: CONFIG_OVERRIDES = {}
   120: # ======================================================================
   121: # EDITABLE REGION END
   122: # ======================================================================
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **split-mnist** — wall-clock budget `1:00:00`, compute share `1`
- **perm-mnist** — wall-clock budget `1:00:00`, compute share `1`
- **split-cifar100** — wall-clock budget `2:00:00`, compute share `1`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.

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


### `ewc` baseline — editable region  [READ-ONLY — reference implementation]

In `continual-learning/custom_regularization.py`:

```python
Lines 25–80:
    22: 
    23: # ======================================================================
    24: # EDITABLE REGION START — estimate_importance
    25: def estimate_importance(model, dataset, prev_params, device):
    26:     """EWC: Diagonal Fisher Information matrix via squared gradients."""
    27:     est_fisher = {}
    28:     for gen_params in model.param_list:
    29:         for n, p in gen_params():
    30:             if p.requires_grad:
    31:                 n = n.replace('.', '__')
    32:                 est_fisher[n] = p.detach().clone().zero_()
    33: 
    34:     mode = model.training
    35:     model.eval()
    36: 
    37:     data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    38:     n_samples = min(len(data_loader), 200)
    39: 
    40:     for idx, (x, y) in enumerate(data_loader):
    41:         if idx >= n_samples:
    42:             break
    43:         x = x.to(device)
    44:         output = model(x)
    45:         with torch.no_grad():
    46:             label_weights = F.softmax(output, dim=1)
    47:         for label_index in range(output.shape[1]):
    48:             label = torch.LongTensor([label_index]).to(device)
    49:             negloglikelihood = F.cross_entropy(output, label)
    50:             model.zero_grad()
    51:             negloglikelihood.backward(
    52:                 retain_graph=True if (label_index + 1) < output.shape[1] else False
    53:             )
    54:             for gen_params in model.param_list:
    55:                 for n, p in gen_params():
    56:                     if p.requires_grad:
    57:                         n = n.replace('.', '__')
    58:                         if p.grad is not None:
    59:                             est_fisher[n] += label_weights[0][label_index] * (p.grad.detach() ** 2)
    60: 
    61:     est_fisher = {n: v / max(n_samples, 1) for n, v in est_fisher.items()}
    62: 
    63:     model.train(mode=mode)
    64:     return est_fisher
    65: 
    66: 
    67: def compute_regularization_loss(model, importance_dict, prev_params_dict):
    68:     """EWC: 0.5 * sum(fisher * (param - prev_param)^2)."""
    69:     losses = []
    70:     for gen_params in model.param_list:
    71:         for n, p in gen_params():
    72:             if p.requires_grad:
    73:                 n = n.replace('.', '__')
    74:                 if n in importance_dict and n in prev_params_dict:
    75:                     fisher = importance_dict[n]
    76:                     prev = prev_params_dict[n]
    77:                     losses.append((fisher * (p - prev) ** 2).sum())
    78:     if losses:
    79:         return 0.5 * sum(losses)
    80:     return torch.tensor(0.0, device=next(model.parameters()).device)
    81: 
    82: # CONFIG_OVERRIDES: override training hyperparameters for your method.
    83: # Allowed keys: reg_strength_scale (multiplier on the per-benchmark reg_strength).

Lines 82–84:
    79:         return 0.5 * sum(losses)
    80:     return torch.tensor(0.0, device=next(model.parameters()).device)
    81: 
    82: # CONFIG_OVERRIDES: override training hyperparameters for your method.
    83: # Allowed keys: reg_strength_scale (multiplier on the per-benchmark reg_strength).
    84: CONFIG_OVERRIDES = {}
    85: # ======================================================================
    86: # EDITABLE REGION END
    87: # ======================================================================
```

### `si` baseline — editable region  [READ-ONLY — reference implementation]

In `continual-learning/custom_regularization.py`:

```python
Lines 25–66:
    22: 
    23: # ======================================================================
    24: # EDITABLE REGION START — estimate_importance
    25: def estimate_importance(model, dataset, prev_params, device):
    26:     """SI: Compute omega from accumulated path integral W and parameter changes.
    27: 
    28:     omega_k = W_k / (delta_k^2 + epsilon)
    29: 
    30:     where W_k is the accumulated per-step gradient-weighted parameter change
    31:     (tracked in model._custom_W by the training loop) and delta_k is the
    32:     total parameter change over the context.
    33:     """
    34:     epsilon = getattr(model, 'epsilon', 0.1)
    35:     omega = {}
    36: 
    37:     # Get the accumulated W from the per-step tracking
    38:     W = getattr(model, '_custom_W', {})
    39: 
    40:     for gen_params in model.param_list:
    41:         for n, p in gen_params():
    42:             if p.requires_grad:
    43:                 n = n.replace('.', '__')
    44:                 p_current = p.detach().clone()
    45:                 p_prev = prev_params.get(n, p_current)
    46:                 p_change = p_current - p_prev
    47:                 w_val = W.get(n, torch.zeros_like(p_current))
    48:                 omega[n] = w_val / (p_change ** 2 + epsilon)
    49: 
    50:     return omega
    51: 
    52: 
    53: def compute_regularization_loss(model, importance_dict, prev_params_dict):
    54:     """SI: sum(omega * (param - prev_param)^2)."""
    55:     losses = []
    56:     for gen_params in model.param_list:
    57:         for n, p in gen_params():
    58:             if p.requires_grad:
    59:                 n = n.replace('.', '__')
    60:                 if n in importance_dict and n in prev_params_dict:
    61:                     omega = importance_dict[n]
    62:                     prev = prev_params_dict[n]
    63:                     losses.append((omega * (p - prev) ** 2).sum())
    64:     if losses:
    65:         return sum(losses)
    66:     return torch.tensor(0.0, device=next(model.parameters()).device)
    67: 
    68: # CONFIG_OVERRIDES: override training hyperparameters for your method.
    69: # Allowed keys: reg_strength_scale (multiplier on the per-benchmark reg_strength).

Lines 68–70:
    65:         return sum(losses)
    66:     return torch.tensor(0.0, device=next(model.parameters()).device)
    67: 
    68: # CONFIG_OVERRIDES: override training hyperparameters for your method.
    69: # Allowed keys: reg_strength_scale (multiplier on the per-benchmark reg_strength).
    70: CONFIG_OVERRIDES = {}
    71: # ======================================================================
    72: # EDITABLE REGION END
    73: # ======================================================================
```

### `online_ewc` baseline — editable region  [READ-ONLY — reference implementation]

In `continual-learning/custom_regularization.py`:

```python
Lines 25–106:
    22: 
    23: # ======================================================================
    24: # EDITABLE REGION START — estimate_importance
    25: def estimate_importance(model, dataset, prev_params, device):
    26:     """Online EWC: Diagonal Fisher with exponential decay accumulation.
    27: 
    28:     When accumulating across contexts: fisher = gamma * fisher_old + fisher_new.
    29:     Uses gamma=0.9 as the online Fisher decay for this benchmark.
    30:     """
    31:     # Explicitly set gamma on the model to override framework default (1.0).
    32:     # With gamma=1.0, Online EWC reduces to standard EWC.
    33:     model.gamma = 0.9
    34:     gamma = model.gamma
    35:     est_fisher = {}
    36:     for gen_params in model.param_list:
    37:         for n, p in gen_params():
    38:             if p.requires_grad:
    39:                 n = n.replace('.', '__')
    40:                 est_fisher[n] = p.detach().clone().zero_()
    41: 
    42:     mode = model.training
    43:     model.eval()
    44: 
    45:     data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    46:     n_samples = min(len(data_loader), 200)
    47: 
    48:     for idx, (x, y) in enumerate(data_loader):
    49:         if idx >= n_samples:
    50:             break
    51:         x = x.to(device)
    52:         output = model(x)
    53:         with torch.no_grad():
    54:             label_weights = F.softmax(output, dim=1)
    55:         for label_index in range(output.shape[1]):
    56:             label = torch.LongTensor([label_index]).to(device)
    57:             negloglikelihood = F.cross_entropy(output, label)
    58:             model.zero_grad()
    59:             negloglikelihood.backward(
    60:                 retain_graph=True if (label_index + 1) < output.shape[1] else False
    61:             )
    62:             for gen_params in model.param_list:
    63:                 for n, p in gen_params():
    64:                     if p.requires_grad:
    65:                         n = n.replace('.', '__')
    66:                         if p.grad is not None:
    67:                             est_fisher[n] += label_weights[0][label_index] * (p.grad.detach() ** 2)
    68: 
    69:     est_fisher = {n: v / max(n_samples, 1) for n, v in est_fisher.items()}
    70: 
    71:     # Apply decay to existing importance before adding new
    72:     existing = getattr(model, '_custom_importance', {})
    73:     for n in est_fisher:
    74:         if n in existing:
    75:             est_fisher[n] = gamma * existing[n] + est_fisher[n]
    76: 
    77:     # We return the full (decayed + new) Fisher, so the training loop
    78:     # should replace (not add to) _custom_importance. To achieve this
    79:     # with the accumulation logic in mid_edit, we subtract the existing
    80:     # importance so that accumulation yields the correct result.
    81:     result = {}
    82:     for n in est_fisher:
    83:         if n in existing:
    84:             result[n] = est_fisher[n] - existing[n]
    85:         else:
    86:             result[n] = est_fisher[n]
    87: 
    88:     model.train(mode=mode)
    89:     return result
    90: 
    91: 
    92: def compute_regularization_loss(model, importance_dict, prev_params_dict):
    93:     """Online EWC: 0.5 * gamma * sum(fisher * (param - prev_param)^2)."""
    94:     gamma = getattr(model, 'gamma', 0.9)  # Already set to 0.9 in estimate_importance
    95:     losses = []
    96:     for gen_params in model.param_list:
    97:         for n, p in gen_params():
    98:             if p.requires_grad:
    99:                 n = n.replace('.', '__')
   100:                 if n in importance_dict and n in prev_params_dict:
   101:                     fisher = importance_dict[n]
   102:                     prev = prev_params_dict[n]
   103:                     losses.append((fisher * (p - prev) ** 2).sum())
   104:     if losses:
   105:         return 0.5 * gamma * sum(losses)
   106:     return torch.tensor(0.0, device=next(model.parameters()).device)
   107: 
   108: # CONFIG_OVERRIDES: override training hyperparameters for your method.
   109: # Allowed keys: reg_strength_scale (multiplier on the per-benchmark reg_strength).

Lines 108–110:
   105:         return 0.5 * gamma * sum(losses)
   106:     return torch.tensor(0.0, device=next(model.parameters()).device)
   107: 
   108: # CONFIG_OVERRIDES: override training hyperparameters for your method.
   109: # Allowed keys: reg_strength_scale (multiplier on the per-benchmark reg_strength).
   110: CONFIG_OVERRIDES = {}
   111: # ======================================================================
   112: # EDITABLE REGION END
   113: # ======================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
