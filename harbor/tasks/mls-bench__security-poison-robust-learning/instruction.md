# MLS-Bench: security-poison-robust-learning

# Poison-Robust Learning under Label-Flip Poisoning

## Research Question
How can we design a stronger loss function or sample-weighting rule that improves robustness to poisoned training labels without changing the model, optimizer, or data pipeline?

## Background
A fraction of poisoned (label-flipped) training labels can disproportionately distort model decision boundaries. Robust learning methods typically modify the objective to downweight suspicious samples or reduce memorization of corrupted targets. Representative approaches include the bootstrapping target (Reed et al., ICLR Workshop 2015, arXiv:1412.6596), Generalized Cross Entropy (Zhang and Sabuncu, NeurIPS 2018, arXiv:1805.07836), and Symmetric Cross Entropy (Wang et al., ICCV 2019, arXiv:1908.06112), each of which introduces a saturation or interpolation mechanism that limits the gradient impact of confidently wrong labels.

This task uses research-scale models trained on full datasets with standard SGD + CosineAnnealing.

## Task
Implement a better poison-robust objective in `bench/poison/custom_robust_loss.py`. The fixed harness injects random label-flip corruption (`(original + 1) % num_classes`) into the training set, trains with your loss, and evaluates on a clean test set.

Your method should improve clean test accuracy under poisoning while reducing how much the model memorizes poisoned labels. The approach must be modular and transferable across architectures and datasets.

## Editable Interface
You must implement:

```python
class RobustLoss:
    def compute_loss(self, logits, labels, epoch):
        ...
```

- `logits`: current minibatch model outputs.
- `labels`: possibly poisoned labels (label-flip: `(original + 1) % num_classes`).
- `epoch`: current training epoch (0-indexed).
- Return value: scalar loss tensor.

The corruption process, model architectures, optimizer, and training schedule are fixed.

## Baselines
The baselines below run inside the same harness via edit ops; defaults follow the corresponding papers:

- `cross_entropy`: standard ERM on poisoned labels.
- `generalized_ce`: Generalized Cross Entropy (Zhang and Sabuncu, NeurIPS 2018, arXiv:1805.07836) with default `q = 0.7`.
- `symmetric_ce`: Symmetric Cross Entropy (Wang et al., ICCV 2019, arXiv:1908.06112), CE plus reverse-CE; CIFAR-10 defaults `alpha = 0.1`, `beta = 1.0`. Reference code: https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels.
- `bootstrap`: bootstrapping target interpolation with model predictions (Reed et al., ICLR Workshop 2015, arXiv:1412.6596).


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-vision/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `pytorch-vision/bench/poison/custom_robust_loss.py`
- editable: **entire file**


Other files you may **read** for context (do not modify):
- `pytorch-vision/bench/poison/run_poison_robust.py`


## Readable Context


### `pytorch-vision/bench/poison/custom_robust_loss.py`  [EDITABLE — entire file only]

```python
     1: """Editable poison-robust loss for MLS-Bench."""
     2: 
     3: import torch
     4: import torch.nn.functional as F
     5: 
     6: # ============================================================
     7: # EDITABLE
     8: # ============================================================
     9: class RobustLoss:
    10:     """Default cross-entropy objective."""
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

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `cross_entropy` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/bench/poison/custom_robust_loss.py`:

```python
     1: """Editable poison-robust loss for MLS-Bench."""
     2: 
     3: import torch
     4: import torch.nn.functional as F
     5: 
     6: # ============================================================
     7: # EDITABLE
     8: class RobustLoss:
     9:     """Standard cross-entropy on poisoned labels."""
    10: 
    11:     def __init__(self):
    12:         pass
    13: 
    14:     def compute_loss(self, logits, labels, epoch):
    15:         return F.cross_entropy(logits, labels)
    16: # ============================================================
    17: # END EDITABLE
    18: # ============================================================
```

### `generalized_ce` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/bench/poison/custom_robust_loss.py`:

```python
     1: """Editable poison-robust loss for MLS-Bench."""
     2: 
     3: import torch
     4: import torch.nn.functional as F
     5: 
     6: # ============================================================
     7: # EDITABLE
     8: class RobustLoss:
     9:     """Generalized cross-entropy for noisy labels."""
    10: 
    11:     def __init__(self):
    12:         self.q = 0.7
    13: 
    14:     def compute_loss(self, logits, labels, epoch):
    15:         probs = torch.softmax(logits, dim=1)
    16:         p = probs.gather(1, labels[:, None]).clamp_min(1e-8)
    17:         return ((1.0 - p.pow(self.q)) / self.q).mean()
    18: # ============================================================
    19: # END EDITABLE
    20: # ============================================================
```

### `symmetric_ce` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/bench/poison/custom_robust_loss.py`:

```python
     1: """Editable poison-robust loss for MLS-Bench."""
     2: 
     3: import torch
     4: import torch.nn.functional as F
     5: 
     6: # ============================================================
     7: # EDITABLE
     8: class RobustLoss:
     9:     """Cross-entropy plus reverse-CE penalty."""
    10: 
    11:     def __init__(self):
    12:         self.alpha = 1.0
    13:         self.beta = 0.5
    14: 
    15:     def compute_loss(self, logits, labels, epoch):
    16:         ce = F.cross_entropy(logits, labels)
    17:         probs = torch.softmax(logits, dim=1).clamp_min(1e-8)
    18:         one_hot = F.one_hot(labels, num_classes=logits.shape[1]).float().clamp_min(1e-4)
    19:         rce = -(probs * torch.log(one_hot)).sum(dim=1).mean()
    20:         return self.alpha * ce + self.beta * rce
    21: # ============================================================
    22: # END EDITABLE
    23: # ============================================================
```

### `bootstrap` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/bench/poison/custom_robust_loss.py`:

```python
     1: """Editable poison-robust loss for MLS-Bench."""
     2: 
     3: import torch
     4: import torch.nn.functional as F
     5: 
     6: # ============================================================
     7: # EDITABLE
     8: class RobustLoss:
     9:     """Interpolate labels with model predictions."""
    10: 
    11:     def __init__(self):
    12:         self.beta = 0.8
    13: 
    14:     def compute_loss(self, logits, labels, epoch):
    15:         hard = F.one_hot(labels, num_classes=logits.shape[1]).float()
    16:         soft = torch.softmax(logits.detach(), dim=1)
    17:         target = self.beta * hard + (1.0 - self.beta) * soft
    18:         log_probs = F.log_softmax(logits, dim=1)
    19:         return -(target * log_probs).sum(dim=1).mean()
    20: # ============================================================
    21: # END EDITABLE
    22: # ============================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
