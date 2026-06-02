# MLS-Bench: security-machine-unlearning

# Machine Unlearning via Targeted Update Rules

## Research Question
How can we design a stronger unlearning update rule that removes information about a forget set while retaining as much utility as possible on the retained data?

## Background
Machine unlearning methods approximate the effect of retraining without the deleted data. The central tradeoff is clear: aggressive forgetting reduces utility, while conservative updates leave measurable traces of the forgotten examples. Approximate-unlearning approaches range from continued retain-only finetuning, to gradient ascent on the forget loss (NegGrad / "Eternal Sunshine of the Spotless Net": Golatkar, Achille, Soatto, CVPR 2020, arXiv:1911.04933), to incompetent-teacher distillation (Bad-T: Chundawat et al., AAAI 2023, arXiv:2205.08096), and selective student-teacher scrubbing (SCRUB: Kurmanji et al., NeurIPS 2023, arXiv:2302.09880).

The harness pretrains a standard vision model (ResNet-20, VGG-16-BN, or MobileNetV2) on the full training set for 80 epochs using SGD with cosine annealing. After pretraining, a single class is designated as the forget set. The unlearning method then runs for 20 epochs, receiving both retain-set and forget-set minibatches each step, with an Adam optimizer (`lr = 0.001`).

## Task
Implement a better unlearning rule in `bench/unlearning/custom_unlearning.py`. The fixed harness trains an initial model, defines a forget split, and then applies your update rule for a fixed number of unlearning steps using retain and forget minibatches.

Your method should lower forget-set memorization while preserving retained-task accuracy.

## Editable Interface
You must implement:

```python
class UnlearningMethod:
    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        ...
```

- `retain_batch`: `(images, labels)` tuple from retained data (already on device).
- `forget_batch`: `(images, labels)` tuple from the forget set (already on device).
- `optimizer`: fixed Adam optimizer instance (`lr = 0.001`).
- Return value: dict with at least `loss`.

The architecture, initial training, forget split, and evaluation probes are fixed.

## Evaluation
Benchmarks:

- `resnet20-cifar10-class0`: ResNet-20 on CIFAR-10, forgetting class 0.
- `vgg16bn-cifar100-class0`: VGG-16-BN on CIFAR-100, forgetting class 0.
- `mobilenetv2-fmnist-class0`: MobileNetV2 on FashionMNIST, forgetting class 0.

Reported metrics:
- `retain_acc`: accuracy on non-forget test data.
- `forget_acc`: accuracy on forget-class test data (lower is better).
- `forget_mia_auc`: membership inference attack AUC on forget set (lower is better).
- `unlearn_score`: `(retain_acc + (1 - forget_acc) + (1 - forget_mia_auc)) / 3`.

Primary metric: `unlearn_score` (higher is better).

## Baselines
The baselines below run inside the same harness via edit ops; defaults follow the corresponding papers:

- `retain_finetune`: continue training only on retained data with the supplied Adam optimizer.
- `negative_gradient`: NegGrad-style ascent on forget loss combined with descent on retain loss (Golatkar et al., CVPR 2020, arXiv:1911.04933).
- `bad_teacher`: incompetent-teacher distillation forgetting (Chundawat et al., AAAI 2023, arXiv:2205.08096). Reference code: https://github.com/vikram2000b/bad-teaching-unlearning.
- `scrub`: SCRUB selective student-teacher scrubbing (Kurmanji et al., NeurIPS 2023, arXiv:2302.09880).


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-vision/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `pytorch-vision/bench/unlearning/custom_unlearning.py`
- editable: **entire file**


Other files you may **read** for context (do not modify):
- `pytorch-vision/bench/unlearning/run_unlearning.py`


## Readable Context


### `pytorch-vision/bench/unlearning/custom_unlearning.py`  [EDITABLE — entire file only]

```python
     1: """Editable unlearning method for MLS-Bench."""
     2: 
     3: import torch
     4: import torch.nn.functional as F
     5: 
     6: # ============================================================
     7: # EDITABLE
     8: # ============================================================
     9: class UnlearningMethod:
    10:     """Default retain-only finetuning update."""
    11: 
    12:     def __init__(self):
    13:         self.forget_weight = 0.0
    14: 
    15:     def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
    16:         retain_x, retain_y = retain_batch
    17:         logits = model(retain_x)
    18:         loss = F.cross_entropy(logits, retain_y)
    19:         optimizer.zero_grad()
    20:         loss.backward()
    21:         optimizer.step()
    22:         return {"loss": loss.item()}
    23: # ============================================================
    24: # END EDITABLE
    25: # ============================================================
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `retain_finetune` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/bench/unlearning/custom_unlearning.py`:

```python
     1: """Editable unlearning method for MLS-Bench."""
     2: 
     3: import torch
     4: import torch.nn.functional as F
     5: 
     6: # ============================================================
     7: # EDITABLE
     8: class UnlearningMethod:
     9:     """Continue training on retained data only."""
    10: 
    11:     def __init__(self):
    12:         pass
    13: 
    14:     def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
    15:         retain_x, retain_y = retain_batch
    16:         logits = model(retain_x)
    17:         loss = F.cross_entropy(logits, retain_y)
    18:         optimizer.zero_grad()
    19:         loss.backward()
    20:         optimizer.step()
    21:         return {"loss": loss.item()}
    22: # ============================================================
    23: # END EDITABLE
    24: # ============================================================
```

### `negative_gradient` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/bench/unlearning/custom_unlearning.py`:

```python
     1: """Editable unlearning method for MLS-Bench."""
     2: 
     3: import torch
     4: import torch.nn.functional as F
     5: 
     6: # ============================================================
     7: # EDITABLE
     8: class UnlearningMethod:
     9:     """Descend retain loss while ascending forget loss."""
    10: 
    11:     def __init__(self):
    12:         self.forget_weight = 0.5
    13: 
    14:     def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
    15:         retain_x, retain_y = retain_batch
    16:         forget_x, forget_y = forget_batch
    17:         retain_loss = F.cross_entropy(model(retain_x), retain_y)
    18:         forget_loss = F.cross_entropy(model(forget_x), forget_y)
    19:         loss = retain_loss - self.forget_weight * forget_loss
    20:         optimizer.zero_grad()
    21:         loss.backward()
    22:         optimizer.step()
    23:         return {"loss": loss.item(), "retain_loss": retain_loss.item(), "forget_loss": forget_loss.item()}
    24: # ============================================================
    25: # END EDITABLE
    26: # ============================================================
```

### `bad_teacher` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/bench/unlearning/custom_unlearning.py`:

```python
     1: """Editable unlearning method for MLS-Bench."""
     2: 
     3: import torch
     4: import torch.nn.functional as F
     5: 
     6: # ============================================================
     7: # EDITABLE
     8: import copy
     9: import torch.nn as nn
    10: 
    11: class UnlearningMethod:
    12:     """Bad Teacher: dual-teacher KD with competent + incompetent teachers.
    13: 
    14:     Paper: https://arxiv.org/abs/2205.08096
    15:     Reference code: https://github.com/vikram2000b/bad-teaching-unlearning
    16:     """
    17: 
    18:     def __init__(self):
    19:         self.KL_temperature = 1.0
    20:         self.competent = None       # = frozen original model
    21:         self.incompetent = None     # = randomly re-initialised same-arch model
    22: 
    23:     def _freeze(self, m):
    24:         for p in m.parameters():
    25:             p.requires_grad_(False)
    26:         m.eval()
    27: 
    28:     def _random_reinit(self, m):
    29:         # Kaiming init identical to initialize_weights() in run_unlearning.py.
    30:         for mod in m.modules():
    31:             if isinstance(mod, nn.Conv2d):
    32:                 nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
    33:             elif isinstance(mod, nn.BatchNorm2d):
    34:                 nn.init.constant_(mod.weight, 1)
    35:                 nn.init.constant_(mod.bias, 0)
    36:             elif isinstance(mod, nn.Linear):
    37:                 nn.init.kaiming_normal_(mod.weight, mode='fan_in', nonlinearity='relu')
    38:                 if mod.bias is not None:
    39:                     nn.init.constant_(mod.bias, 0)
    40: 
    41:     def _capture_teachers(self, model):
    42:         self.competent = copy.deepcopy(model)
    43:         self._freeze(self.competent)
    44: 
    45:         self.incompetent = copy.deepcopy(model)
    46:         self._random_reinit(self.incompetent)
    47:         self._freeze(self.incompetent)
    48: 
    49:     def _unlearner_loss(self, student_logits, full_teacher_logits,
    50:                         unlearn_teacher_logits, is_forget):
    51:         # Ref: UnlearnerLoss in vikram2000b/bad-teaching-unlearning.
    52:         T = self.KL_temperature
    53:         f_t = F.softmax(full_teacher_logits / T, dim=1)
    54:         u_t = F.softmax(unlearn_teacher_logits / T, dim=1)
    55:         lbl = is_forget.view(-1, 1).float()
    56:         target = lbl * u_t + (1.0 - lbl) * f_t
    57:         log_s = F.log_softmax(student_logits / T, dim=1)
    58:         return F.kl_div(log_s, target, reduction='batchmean')
    59: 
    60:     def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
    61:         if self.competent is None:
    62:             self._capture_teachers(model)
    63: 
    64:         retain_x, _ = retain_batch
    65:         forget_x, _ = forget_batch
    66: 
    67:         # Balanced mini-batch: concatenate retain + forget samples.
    68:         x = torch.cat([retain_x, forget_x], dim=0)
    69:         is_forget = torch.cat([
    70:             torch.zeros(retain_x.size(0), device=retain_x.device),
    71:             torch.ones(forget_x.size(0), device=forget_x.device),
    72:         ], dim=0)
    73: 
    74:         student_logits = model(x)
    75:         with torch.no_grad():
    76:             full_t = self.competent(x)
    77:             unl_t = self.incompetent(x)
    78: 
    79:         loss = self._unlearner_loss(student_logits, full_t, unl_t, is_forget)
    80: 
    81:         optimizer.zero_grad()
    82:         loss.backward()
    83:         optimizer.step()
    84: 
    85:         return {"loss": float(loss.item())}
    86: # ============================================================
    87: # END EDITABLE
    88: # ============================================================
```

### `scrub` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/bench/unlearning/custom_unlearning.py`:

```python
     1: """Editable unlearning method for MLS-Bench."""
     2: 
     3: import torch
     4: import torch.nn.functional as F
     5: 
     6: # ============================================================
     7: # EDITABLE
     8: import copy
     9: 
    10: class UnlearningMethod:
    11:     """SCRUB: min-max KL distillation vs a frozen original model.
    12: 
    13:     Paper: https://arxiv.org/abs/2302.09880
    14:     Reference code: https://github.com/meghdadk/SCRUB
    15:     """
    16: 
    17:     def __init__(self):
    18:         # Defaults from the authors' VGG notebook.
    19:         self.msteps = 2        # number of max-step epochs (rewind)
    20:         self.kd_T = 4.0        # KD temperature
    21:         self.alpha = 0.01      # weight on KL(student || teacher) in min step
    22:         self.gamma = 0.99      # weight on CE(student, y) in min step
    23:         self.teacher = None    # lazily captured on first step
    24: 
    25:     def _kd_kl(self, student_logits, teacher_logits):
    26:         # KL(student || teacher) with temperature, as in Hinton KD.
    27:         T = self.kd_T
    28:         p_s = F.log_softmax(student_logits / T, dim=1)
    29:         p_t = F.softmax(teacher_logits / T, dim=1)
    30:         return F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)
    31: 
    32:     def _capture_teacher(self, model):
    33:         self.teacher = copy.deepcopy(model)
    34:         for p in self.teacher.parameters():
    35:             p.requires_grad_(False)
    36:         self.teacher.eval()
    37: 
    38:     def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
    39:         if self.teacher is None:
    40:             self._capture_teacher(model)
    41: 
    42:         retain_x, retain_y = retain_batch
    43:         forget_x, _ = forget_batch
    44: 
    45:         # ---- Max step on forget set (only during the first msteps epochs) ----
    46:         forget_kl_val = 0.0
    47:         if epoch < self.msteps:
    48:             optimizer.zero_grad()
    49:             s_forget = model(forget_x)
    50:             with torch.no_grad():
    51:                 t_forget = self.teacher(forget_x)
    52:             forget_kl = self._kd_kl(s_forget, t_forget)
    53:             (-forget_kl).backward()
    54:             optimizer.step()
    55:             forget_kl_val = forget_kl.item()
    56: 
    57:         # ---- Min step on retain set (every epoch) ----
    58:         optimizer.zero_grad()
    59:         s_retain = model(retain_x)
    60:         with torch.no_grad():
    61:             t_retain = self.teacher(retain_x)
    62:         retain_ce = F.cross_entropy(s_retain, retain_y)
    63:         retain_kl = self._kd_kl(s_retain, t_retain)
    64:         loss = self.gamma * retain_ce + self.alpha * retain_kl
    65:         loss.backward()
    66:         optimizer.step()
    67: 
    68:         return {
    69:             "loss": float(loss.item()),
    70:             "retain_ce": float(retain_ce.item()),
    71:             "retain_kl": float(retain_kl.item()),
    72:             "forget_kl": float(forget_kl_val),
    73:         }
    74: # ============================================================
    75: # END EDITABLE
    76: # ============================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
