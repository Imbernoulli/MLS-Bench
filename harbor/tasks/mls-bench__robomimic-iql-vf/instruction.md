# MLS-Bench: robomimic-iql-vf

# Implicit Q-Learning: Value Function Loss Design for Offline Robot Learning

## Research Question
Design an improved value function loss for Implicit Q-Learning (IQL) in offline robot manipulation. IQL avoids querying out-of-distribution actions by learning V(s) via asymmetric regression against Q(s, a) estimates. The loss function determines how V(s) approximates the upper quantile of Q-values, directly affecting policy quality.

## Background
**Implicit Q-Learning** (Kostrikov, Nair, Levine, ICLR 2022, arXiv:2110.06169) avoids the policy-evaluation step over out-of-distribution actions by fitting V(s) to an upper expectile of Q(s, a) drawn from the dataset. The default objective is the **expectile regression** loss:

```
diff = q_target - vf_pred
loss = mean( |quantile - 1{diff < 0}| * diff**2 )
```

When `quantile = 0.9`, overestimation (V > Q) is penalized 9× more than underestimation, so V(s) tracks the upper-tail of Q. The value function feeds the actor via advantage-weighted regression:

```
w(s, a) = exp((Q(s, a) - V(s)) / beta)
```

so V quality directly impacts policy learning. Alternative asymmetric losses (quantile regression, asymmetric Huber, log-cosh variants, etc.) may yield smoother gradients or better extrapolation.

The training pipeline uses **robomimic** (Mandlekar et al., CoRL 2021, arXiv:2108.03298).

## What You Can Modify
The `custom_vf_loss` function in `custom_iql_vf.py`. This function computes the loss for training the value network V(s).

Interface:
- **Input**:
  - `vf_pred: [B, 1]` — predicted state values V(s)
  - `q_target: [B, 1]` — target Q-values Q(s, a) from the critic ensemble (detached)
  - `quantile: float` — asymmetry parameter τ (default 0.9)
- **Output**: scalar loss tensor

You may restructure the function body, add helper computations, and use any PyTorch operations.

## Evaluation
- **Metric**: `success_rate` — rollout success rate on the task (higher is better)
- **Tasks**: Lift, Can, Square (robot manipulation with proficient human demonstrations)
- **Dataset**: ~200 demonstrations with (s, a, r, s', done) transitions
- **Training**: IQL with Q-ensemble (2 critics), GMM actor (5 modes), 2000 epochs × 100 steps
- **Hyperparameters**: discount = 0.99, target_tau = 0.01, adv_beta = 1.0, vf_quantile = 0.9
- **Rollout evaluation**: 50 episodes per task, horizon 400 steps, every 50 epochs


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/robomimic/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `robomimic/custom_iql_vf.py`
- editable lines **21–38**




## Readable Context


### `robomimic/custom_iql_vf.py`  [EDITABLE — lines 21–38 only]

```python
     1: """
     2: Custom Value Function Loss for Implicit Q-Learning (IQL).
     3: 
     4: This module defines the value function loss used by IQL training in
     5: robomimic. The loss receives predicted state values V(s), target
     6: Q-values Q(s,a), and an asymmetry parameter (quantile/tau), and
     7: returns a scalar loss that trains V(s) to approximate a high quantile
     8: of the Q-value distribution.
     9: 
    10: The custom loss is imported and used by the patched IQL._compute_critic_loss
    11: method during training.
    12: """
    13: 
    14: import torch
    15: import torch.nn as nn
    16: import torch.nn.functional as F
    17: 
    18: 
    19: # ── Custom Value Function Loss ─────────────────────────────────────────────
    20: # EDITABLE REGION START
    21: def custom_vf_loss(vf_pred, q_target, quantile=0.9):
    22:     """Custom value function loss for IQL.
    23: 
    24:     Computes an asymmetric regression loss that pushes V(s) toward a
    25:     high quantile of Q(s,a) without explicit maximization over actions.
    26:     The default implementation uses expectile regression (IQL paper).
    27: 
    28:     Args:
    29:         vf_pred: [B, 1] predicted state values V(s)
    30:         q_target: [B, 1] target Q-values Q(s,a) (detached)
    31:         quantile: float in (0, 1), asymmetry parameter (tau)
    32: 
    33:     Returns:
    34:         scalar loss tensor
    35:     """
    36:     diff = vf_pred - q_target
    37:     weight = torch.where(diff > 0, 1.0 - quantile, quantile)
    38:     return (weight * (diff ** 2)).mean()
    39: # EDITABLE REGION END
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


### `quantile_regression` baseline — editable region  [READ-ONLY — reference implementation]

In `robomimic/custom_iql_vf.py`:

```python
Lines 21–29:
    18: 
    19: # ── Custom Value Function Loss ─────────────────────────────────────────────
    20: # EDITABLE REGION START
    21: def custom_vf_loss(vf_pred, q_target, quantile=0.9):
    22:     """Quantile regression loss (asymmetric L1).
    23: 
    24:     Uses L1 loss instead of L2, providing robustness to outlier
    25:     Q-value estimates while still pushing V(s) toward a high quantile.
    26:     """
    27:     diff = vf_pred - q_target
    28:     weight = torch.where(diff > 0, 1.0 - quantile, quantile)
    29:     return (weight * diff.abs()).mean()
    30: # EDITABLE REGION END
```

### `huber_pinball` baseline — editable region  [READ-ONLY — reference implementation]

In `robomimic/custom_iql_vf.py`:

```python
Lines 21–31:
    18: 
    19: # ── Custom Value Function Loss ─────────────────────────────────────────────
    20: # EDITABLE REGION START
    21: def custom_vf_loss(vf_pred, q_target, quantile=0.9):
    22:     """Huber-pinball loss: asymmetric Huber for robust value estimation.
    23: 
    24:     Uses Huber loss (smooth L1) with asymmetric weighting. Quadratic
    25:     for small errors, linear for large errors, with higher weight on
    26:     under-estimation to push V(s) toward high quantiles of Q(s,a).
    27:     """
    28:     diff = vf_pred - q_target
    29:     weight = torch.where(diff > 0, 1.0 - quantile, quantile)
    30:     huber = F.smooth_l1_loss(vf_pred, q_target, reduction='none', beta=1.0)
    31:     return (weight * huber).mean()
    32: # EDITABLE REGION END
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
