# MLS-Bench: security-membership-inference-defense

# Membership Inference Defense via Training Regularization

## Research Question
How can we design a stronger training-time regularizer that reduces membership inference leakage while preserving standard predictive accuracy?

## Background
Membership inference attacks exploit the gap between train and non-train examples, often through confidence or loss statistics. Many defenses regularize predictions to reduce overconfidence and shrink the train-test generalization gap, but stronger privacy often hurts utility. Common training-time regularizers include label smoothing, the confidence penalty (Pereyra et al., ICLR Workshop 2017, arXiv:1701.06548) which penalizes low-entropy outputs, and RelaxLoss (Chen, Yu, Fritz, ICLR 2022, arXiv:2207.05801), which relaxes the per-example training objective once confidence is already high so that training and held-out distributions become harder to distinguish.

## Task
Implement a better privacy-preserving training objective in `bench/membership/custom_membership_defense.py`. The fixed harness will train a model on a 50/50 train/non-train split of the full dataset, then run a confidence-based membership inference attack on train versus held-out examples.

Your method should improve the privacy-utility tradeoff: lower membership-attack AUC while retaining high test accuracy.

## Editable Interface
You must implement:

```python
class MembershipDefense:
    def compute_loss(self, logits, labels, epoch):
        ...
```

- `logits`: model outputs for the current minibatch.
- `labels`: ground-truth labels.
- `epoch`: current training epoch (0-indexed).
- Return value: scalar loss tensor used by the fixed training loop.

The optimizer (SGD + CosineAnnealing), architecture, data pipeline, and attack implementation are fixed.

## Evaluation
Benchmarks:

- `resnet20-cifar10`: ResNet-20 on CIFAR-10.
- `vgg16bn-cifar100`: VGG-16-BN on CIFAR-100.
- `mobilenetv2-fmnist`: MobileNetV2 on FashionMNIST.

Reported metrics:
- `test_acc`
- `mia_auc`
- `privacy_gap`
- `privacy_score`

Primary metric: `privacy_score = test_acc - max(mia_auc - 0.5, 0)` (higher is better).

## Baselines
The baselines below run inside the same harness via edit ops; defaults follow the corresponding papers:

- `erm`: standard cross-entropy training.
- `label_smoothing`: smoothed targets (smoothing factor `0.1`).
- `confidence_penalty`: cross-entropy plus predictive entropy penalty (Pereyra et al., ICLR Workshop 2017, arXiv:1701.06548) with default penalty weight `0.1`.
- `relaxloss`: margin-aware loss relaxation (Chen, Yu, Fritz, ICLR 2022, arXiv:2207.05801). Reference code: https://github.com/DingfanChen/RelaxLoss.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-vision/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `pytorch-vision/custom_membership_defense.py`
- editable: **entire file**


Other files you may **read** for context (do not modify):
- `pytorch-vision/run_membership_defense.py`


## Readable Context


### `pytorch-vision/custom_membership_defense.py`  [EDITABLE — entire file only]

```python
     1: """Editable membership-inference defense for MLS-Bench."""
     2: 
     3: import torch
     4: import torch.nn.functional as F
     5: 
     6: # ============================================================
     7: # EDITABLE
     8: # ============================================================
     9: class MembershipDefense:
    10:     """Training-time regularizer for privacy-utility tradeoffs.
    11: 
    12:     The compute_loss method replaces nn.CrossEntropyLoss() in the
    13:     fixed training loop.  Design a loss that reduces membership
    14:     inference leakage (lower MIA AUC) while preserving test accuracy.
    15: 
    16:     Args:
    17:         logits: raw model outputs, shape (batch_size, num_classes)
    18:         labels: ground-truth class indices, shape (batch_size,)
    19:         epoch:  current training epoch (0-indexed)
    20: 
    21:     Returns:
    22:         scalar loss tensor
    23:     """
    24: 
    25:     def __init__(self):
    26:         pass
    27: 
    28:     def compute_loss(self, logits, labels, epoch):
    29:         return F.cross_entropy(logits, labels)
    30: # ============================================================
    31: # END EDITABLE
    32: # ============================================================
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `erm` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_membership_defense.py`:

```python
     1: """Editable membership-inference defense for MLS-Bench."""
     2: 
     3: import torch
     4: import torch.nn.functional as F
     5: 
     6: # ============================================================
     7: # EDITABLE
     8: # ============================================================
     9: class MembershipDefense:
    10:     """Standard cross-entropy training."""
    11: 
    12:     def __init__(self):
    13:         pass
    14: 
    15:     def compute_loss(self, logits, labels, epoch):
    16:         return F.cross_entropy(logits, labels)
    17: # ============================================================
    18: # END EDITABLE
    19: # ============================================================
```

### `label_smoothing` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_membership_defense.py`:

```python
     1: """Editable membership-inference defense for MLS-Bench."""
     2: 
     3: import torch
     4: import torch.nn.functional as F
     5: 
     6: # ============================================================
     7: # EDITABLE
     8: # ============================================================
     9: class MembershipDefense:
    10:     """Cross-entropy with fixed label smoothing."""
    11: 
    12:     def __init__(self):
    13:         self.label_smoothing = 0.1
    14: 
    15:     def compute_loss(self, logits, labels, epoch):
    16:         return F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
    17: # ============================================================
    18: # END EDITABLE
    19: # ============================================================
```

### `confidence_penalty` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_membership_defense.py`:

```python
     1: """Editable membership-inference defense for MLS-Bench."""
     2: 
     3: import torch
     4: import torch.nn.functional as F
     5: 
     6: # ============================================================
     7: # EDITABLE
     8: # ============================================================
     9: class MembershipDefense:
    10:     """Cross-entropy minus predictive entropy bonus."""
    11: 
    12:     def __init__(self):
    13:         self.entropy_weight = 0.1
    14: 
    15:     def compute_loss(self, logits, labels, epoch):
    16:         ce = F.cross_entropy(logits, labels)
    17:         probs = torch.softmax(logits, dim=1)
    18:         entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()
    19:         return ce - self.entropy_weight * entropy
    20: # ============================================================
    21: # END EDITABLE
    22: # ============================================================
```

### `relaxloss` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_membership_defense.py`:

```python
     1: """Editable membership-inference defense for MLS-Bench."""
     2: 
     3: import torch
     4: import torch.nn.functional as F
     5: 
     6: # ============================================================
     7: # EDITABLE
     8: # ============================================================
     9: class MembershipDefense:
    10:     """RelaxLoss training rule (Chen et al., ICLR 2022).
    11: 
    12:     Two-phase alternation per epoch:
    13:       Even epochs: loss = |mean_CE - alpha|   (drives loss toward target level)
    14:       Odd  epochs: if mean_CE > alpha -> CE descent
    15:                    else -> posterior flattening with sign-flipped CE
    16:     See github.com/DingfanChen/RelaxLoss/blob/main/source/cifar/defense/relaxloss.py.
    17:     """
    18: 
    19:     def __init__(self):
    20:         # alpha is the target loss level; chosen per num_classes at call time.
    21:         # upper=1 matches every released config; no clamp effect in practice
    22:         # but kept for faithfulness to the official code.
    23:         self.upper = 1.0
    24: 
    25:     def compute_loss(self, logits, labels, epoch):
    26:         num_classes = logits.size(1)
    27:         # Released configs use dataset/model-specific alpha values; this task
    28:         # selects alpha by class count for the exposed benchmark cases.
    29:         alpha = 0.5 if num_classes == 100 else 1.0
    30: 
    31:         loss_ce_full = F.cross_entropy(logits, labels, reduction='none')
    32:         loss_ce = loss_ce_full.mean()
    33: 
    34:         if epoch % 2 == 0:
    35:             # Gradient ascent / descent toward target level alpha.
    36:             return (loss_ce - alpha).abs()
    37: 
    38:         # Odd epoch.
    39:         if loss_ce.item() > alpha:
    40:             return loss_ce
    41: 
    42:         # Posterior flattening.
    43:         probs = torch.softmax(logits, dim=1)
    44:         confidence_target = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    45:         confidence_target = torch.clamp(confidence_target, min=0.0, max=self.upper)
    46:         confidence_else = (1.0 - confidence_target) / (num_classes - 1)
    47: 
    48:         onehot = F.one_hot(labels, num_classes=num_classes).float()
    49:         soft_targets = (
    50:             onehot * confidence_target.unsqueeze(1)
    51:             + (1.0 - onehot) * confidence_else.unsqueeze(1)
    52:         )
    53:         # Detach targets so the flattening gradient flows only through logits
    54:         # (matches the official implementation: soft_targets is built from a
    55:         # forward pass and used as a target).
    56:         soft_targets = soft_targets.detach()
    57: 
    58:         log_probs = F.log_softmax(logits, dim=1)
    59:         ce_soft = -(soft_targets * log_probs).sum(dim=1)
    60: 
    61:         pred = logits.argmax(dim=1)
    62:         correct = pred.eq(labels).float()
    63: 
    64:         # For correctly-classified samples: -loss_ce_full (gradient ascent on CE).
    65:         # For incorrectly-classified samples: ce_soft - loss_ce_full
    66:         # (pull toward flattened posterior while still descending on CE).
    67:         loss = (1.0 - correct) * ce_soft - loss_ce_full
    68:         return loss.mean()
    69: # ============================================================
    70: # END EDITABLE
    71: # ============================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
