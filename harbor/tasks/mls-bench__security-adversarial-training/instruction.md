# MLS-Bench: security-adversarial-training

# Adversarial Training for Model Robustness

## Research Question
How can we design better adversarial training methods to enhance model robustness against `L_inf` adversarial attacks?

## Background
Adversarial training is the most effective approach for improving neural-network robustness against adversarial examples. The standard method (Madry et al., 2018, arXiv:1706.06083) trains on PGD-generated adversarial examples using cross-entropy loss but suffers from a tradeoff between clean accuracy and robust accuracy. Advanced methods such as TRADES (Zhang et al., ICML 2019, arXiv:1901.08573) and MART (Wang et al., ICLR 2020, OpenReview rklOg6EFwS) address this through different loss formulations that decouple the robustness objective from clean classification. AWP (Wu, Xia, Wang, NeurIPS 2020, arXiv:2004.05884) further regularizes the flatness of the weight loss landscape and combines naturally with TRADES-style training.

## Task
Implement a novel adversarial training method in `bench/custom_adv_train.py` by modifying the `AdversarialTrainer` class. Your method should improve robust accuracy against white-box `L_inf` attacks while maintaining reasonable clean accuracy.

## Editable Interface
You must implement the `AdversarialTrainer` class with two methods:

```python
class AdversarialTrainer:
    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        ...

    def train_step(self, images, labels, optimizer) -> dict:
        ...
```

`__init__`:
- `model`: the neural network to train (`nn.Module`).
- `eps`: `L_inf` perturbation budget (`0.3` for MNIST, `8/255` for CIFAR).
- `alpha`: step size for the inner PGD attack.
- `attack_steps`: number of PGD steps for adversarial example generation.
- `num_classes`: number of output classes (10 or 100).

`train_step`:
- `images`: clean images, shape `(N, C, H, W)`, values in `[0, 1]`.
- `labels`: ground-truth labels, shape `(N,)`.
- `optimizer`: SGD optimizer (`lr`, `momentum`, `weight_decay` already configured).
- Returns: dict with at least a `'loss'` key (float).

The training loop, learning-rate schedule (cosine annealing), model architecture, and data loading are handled externally. You only control the adversarial training procedure within each step.

## Evaluation
After training, models are evaluated on:
- **Clean accuracy**: accuracy on unperturbed test images.
- **Robust accuracy (FGSM)**: accuracy under one-step FGSM attack.
- **Robust accuracy (PGD-50)**: accuracy under a 50-step PGD attack — primary metric.

Scenarios (model + dataset):
- SmallCNN on MNIST (`eps = 0.3`)
- PreActResNet-18 on CIFAR-10 (`eps = 8/255`)
- VGG-11-BN on CIFAR-10 (`eps = 8/255`)
- PreActResNet-18 on CIFAR-100 (`eps = 8/255`)

Higher robust accuracy under PGD-50 across all scenarios is better.

## Baselines
The baselines below run inside the same harness via edit ops; defaults follow the corresponding papers:

- `pgdat`: PGD adversarial training (Madry et al., 2018, arXiv:1706.06083). Standard PGD inner attack, cross-entropy loss on adversarial examples.
- `trades`: TRADES (Zhang et al., ICML 2019, arXiv:1901.08573). Cross-entropy on clean inputs plus a KL-divergence robustness regularizer with default `beta = 6.0` from the paper. Reference code: https://github.com/yaodongyu/TRADES.
- `mart`: MART (Wang et al., ICLR 2020). Misclassification-aware regularization that focuses on hard examples; default loss weight `lambda = 5.0` from the paper. Reference code: https://github.com/YisenWang/MART.
- `awp`: AWP combined with TRADES (Wu, Xia, Wang, NeurIPS 2020, arXiv:2004.05884). Adversarial weight perturbation on top of TRADES with default weight-perturbation magnitude `gamma = 5e-3` from the paper. Reference code: https://github.com/csdongxian/AWP.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/torchattacks/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `torchattacks/bench/custom_adv_train.py`
- editable lines **10–54**


Other files you may **read** for context (do not modify):
- `torchattacks/bench/run_adv_train.py`
- `torchattacks/bench/models.py`


## Readable Context


### `torchattacks/bench/custom_adv_train.py`  [EDITABLE — lines 10–54 only]

```python
     1: """Custom adversarial training method for MLS-Bench."""
     2: 
     3: import torch
     4: import torch.nn as nn
     5: import torch.nn.functional as F
     6: 
     7: # ═══════════════════════════════════════════════════════════════════
     8: # EDITABLE — implement AdversarialTrainer below
     9: # ═══════════════════════════════════════════════════════════════════
    10: class AdversarialTrainer:
    11:     """
    12:     Adversarial training method.
    13: 
    14:     The agent should modify this class to implement a better adversarial
    15:     training procedure that improves model robustness against L_inf attacks.
    16: 
    17:     Args:
    18:         model (nn.Module): The model to train.
    19:         eps (float): L_inf perturbation budget.
    20:         alpha (float): Step size for adversarial perturbation generation.
    21:         attack_steps (int): Number of PGD steps for adversarial example generation.
    22:         num_classes (int): Number of output classes.
    23:     """
    24: 
    25:     def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
    26:         self.model = model
    27:         self.eps = eps
    28:         self.alpha = alpha
    29:         self.attack_steps = attack_steps
    30:         self.num_classes = num_classes
    31: 
    32:     def train_step(self, images, labels, optimizer):
    33:         """
    34:         Perform one adversarial training step.
    35: 
    36:         Args:
    37:             images: Clean images, shape (N, C, H, W), values in [0, 1].
    38:             labels: Ground truth labels, shape (N,).
    39:             optimizer: Model optimizer (already configured).
    40: 
    41:         Returns:
    42:             dict: Must contain 'loss' key (float).
    43:         """
    44:         # Default: standard (non-adversarial) training
    45:         self.model.train()
    46:         outputs = self.model(images)
    47:         loss = F.cross_entropy(outputs, labels)
    48: 
    49:         optimizer.zero_grad()
    50:         loss.backward()
    51:         optimizer.step()
    52: 
    53:         return {'loss': loss.item()}
    54: 
    55: # ═══════════════════════════════════════════════════════════════════
    56: # END EDITABLE
    57: # ═══════════════════════════════════════════════════════════════════
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `pgdat` baseline — editable region  [READ-ONLY — reference implementation]

In `torchattacks/bench/custom_adv_train.py`:

```python
Lines 10–46:
     7: # ═══════════════════════════════════════════════════════════════════
     8: # EDITABLE — implement AdversarialTrainer below
     9: # ═══════════════════════════════════════════════════════════════════
    10: class AdversarialTrainer:
    11:     """PGD Adversarial Training (Madry et al., 2018)."""
    12: 
    13:     def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
    14:         self.model = model
    15:         self.eps = eps
    16:         self.alpha = alpha
    17:         self.attack_steps = attack_steps
    18:         self.num_classes = num_classes
    19: 
    20:     def train_step(self, images, labels, optimizer):
    21:         # Generate adversarial examples using PGD
    22:         self.model.eval()
    23:         adv_images = images.clone().detach()
    24:         adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
    25:         adv_images = torch.clamp(adv_images, 0.0, 1.0)
    26: 
    27:         for _ in range(self.attack_steps):
    28:             adv_images.requires_grad_(True)
    29:             outputs = self.model(adv_images)
    30:             loss = F.cross_entropy(outputs, labels)
    31:             grad = torch.autograd.grad(loss, adv_images)[0]
    32:             adv_images = adv_images.detach() + self.alpha * grad.sign()
    33:             delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
    34:             adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()
    35: 
    36:         # Train on adversarial examples
    37:         self.model.train()
    38:         outputs = self.model(adv_images)
    39:         loss = F.cross_entropy(outputs, labels)
    40: 
    41:         optimizer.zero_grad()
    42:         loss.backward()
    43:         optimizer.step()
    44: 
    45:         return {'loss': loss.item()}
    46: 
    47: # ═══════════════════════════════════════════════════════════════════
    48: # END EDITABLE
    49: # ═══════════════════════════════════════════════════════════════════
```

### `trades` baseline — editable region  [READ-ONLY — reference implementation]

In `torchattacks/bench/custom_adv_train.py`:

```python
Lines 10–66:
     7: # ═══════════════════════════════════════════════════════════════════
     8: # EDITABLE — implement AdversarialTrainer below
     9: # ═══════════════════════════════════════════════════════════════════
    10: class AdversarialTrainer:
    11:     """TRADES (Zhang et al., 2019)."""
    12: 
    13:     def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
    14:         self.model = model
    15:         self.eps = eps
    16:         self.alpha = alpha
    17:         self.attack_steps = attack_steps
    18:         self.num_classes = num_classes
    19:         self.beta = 6.0  # TRADES regularization weight
    20: 
    21:     def train_step(self, images, labels, optimizer):
    22:         self.model.train()
    23: 
    24:         # Clean forward pass
    25:         logits_clean = self.model(images)
    26:         loss_clean = F.cross_entropy(logits_clean, labels)
    27: 
    28:         # Generate adversarial examples by maximizing KL divergence
    29:         self.model.eval()
    30:         adv_images = images.clone().detach()
    31:         adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
    32:         adv_images = torch.clamp(adv_images, 0.0, 1.0)
    33: 
    34:         for _ in range(self.attack_steps):
    35:             adv_images.requires_grad_(True)
    36:             logits_adv = self.model(adv_images)
    37:             loss_kl = F.kl_div(
    38:                 F.log_softmax(logits_adv, dim=1),
    39:                 F.softmax(logits_clean.detach(), dim=1),
    40:                 reduction='batchmean',
    41:             )
    42:             grad = torch.autograd.grad(loss_kl, adv_images)[0]
    43:             adv_images = adv_images.detach() + self.alpha * grad.sign()
    44:             delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
    45:             adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()
    46: 
    47:         # TRADES loss: clean CE + beta * KL(clean || adv)
    48:         self.model.train()
    49:         logits_adv = self.model(adv_images)
    50:         loss_kl = F.kl_div(
    51:             F.log_softmax(logits_adv, dim=1),
    52:             F.softmax(logits_clean.detach(), dim=1),
    53:             reduction='batchmean',
    54:         )
    55:         loss = loss_clean + self.beta * loss_kl
    56: 
    57:         optimizer.zero_grad()
    58:         loss.backward()
    59:         optimizer.step()
    60: 
    61:         return {
    62:             'loss': loss.item(),
    63:             'loss_clean': loss_clean.item(),
    64:             'loss_kl': loss_kl.item(),
    65:         }
    66: 
    67: # ═══════════════════════════════════════════════════════════════════
    68: # END EDITABLE
    69: # ═══════════════════════════════════════════════════════════════════
```

### `mart` baseline — editable region  [READ-ONLY — reference implementation]

In `torchattacks/bench/custom_adv_train.py`:

```python
Lines 10–71:
     7: # ═══════════════════════════════════════════════════════════════════
     8: # EDITABLE — implement AdversarialTrainer below
     9: # ═══════════════════════════════════════════════════════════════════
    10: class AdversarialTrainer:
    11:     """MART (Wang et al., 2020)."""
    12: 
    13:     def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
    14:         self.model = model
    15:         self.eps = eps
    16:         self.alpha = alpha
    17:         self.attack_steps = attack_steps
    18:         self.num_classes = num_classes
    19:         self.beta = 6.0  # MART regularization weight
    20: 
    21:     def train_step(self, images, labels, optimizer):
    22:         # Generate adversarial examples using PGD (maximize CE loss).
    23:         # Follows official https://github.com/YisenWang/MART/blob/master/mart.py
    24:         self.model.eval()
    25:         adv_images = images.detach() + 0.001 * torch.randn_like(images)
    26:         adv_images = torch.clamp(adv_images, 0.0, 1.0)
    27: 
    28:         for _ in range(self.attack_steps):
    29:             adv_images.requires_grad_(True)
    30:             outputs = self.model(adv_images)
    31:             loss = F.cross_entropy(outputs, labels)
    32:             grad = torch.autograd.grad(loss, adv_images)[0]
    33:             adv_images = adv_images.detach() + self.alpha * grad.sign()
    34:             delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
    35:             adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()
    36: 
    37:         # MART loss (exactly as in official mart.py)
    38:         self.model.train()
    39:         optimizer.zero_grad()
    40: 
    41:         logits_clean = self.model(images).detach()  # detach for stable KL target + weighting (matches official mart.py)
    42:         logits_adv = self.model(adv_images)
    43:         adv_probs = F.softmax(logits_adv, dim=1)
    44: 
    45:         # Boosted CE: standard CE + penalize runner-up class
    46:         tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    47:         new_y = torch.where(
    48:             tmp1[:, -1] == labels, tmp1[:, -2], tmp1[:, -1],
    49:         )
    50:         loss_adv = F.cross_entropy(logits_adv, labels) + F.nll_loss(
    51:             torch.log(1.0001 - adv_probs + 1e-12), new_y,
    52:         )
    53: 
    54:         # Misclassification-aware KL regularization
    55:         nat_probs = F.softmax(logits_clean, dim=1)
    56:         true_probs = nat_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    57:         kl_per_sample = F.kl_div(
    58:             torch.log(adv_probs + 1e-12), nat_probs, reduction='none',
    59:         ).sum(dim=1)
    60:         batch_size = images.size(0)
    61:         loss_robust = (1.0 / batch_size) * torch.sum(
    62:             kl_per_sample * (1.0000001 - true_probs)
    63:         )
    64: 
    65:         loss = loss_adv + self.beta * loss_robust
    66: 
    67:         loss.backward()
    68:         optimizer.step()
    69: 
    70:         return {'loss': loss.item()}
    71: 
    72: # ═══════════════════════════════════════════════════════════════════
    73: # END EDITABLE
    74: # ═══════════════════════════════════════════════════════════════════
```

### `awp` baseline — editable region  [READ-ONLY — reference implementation]

In `torchattacks/bench/custom_adv_train.py`:

```python
Lines 10–114:
     7: # ═══════════════════════════════════════════════════════════════════
     8: # EDITABLE — implement AdversarialTrainer below
     9: # ═══════════════════════════════════════════════════════════════════
    10: class AdversarialTrainer:
    11:     """AWP + TRADES (Wu et al., 2020 + Zhang et al., 2019).
    12: 
    13:     Follows official trades_AWP/utils_awp.py:TradesAWP and
    14:     trades_AWP/train_trades_cifar.py (csdongxian/AWP).
    15:     """
    16: 
    17:     def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
    18:         import copy
    19:         from collections import OrderedDict
    20:         self.model = model
    21:         self.eps = eps
    22:         self.alpha = alpha
    23:         self.attack_steps = attack_steps
    24:         self.num_classes = num_classes
    25:         self.beta = 6.0        # TRADES regularization weight
    26:         self.gamma = 0.005     # AWP perturbation magnitude (paper default)
    27:         self._EPS_AWP = 1e-20
    28:         # Proxy model + optimizer (lr matches main model's lr=0.1 as in paper).
    29:         # Using a proxy ensures BN running stats of the main model are untouched.
    30:         self.proxy = copy.deepcopy(model)
    31:         # Official uses proxy_optim lr == main lr (0.1). Magnitude is controlled
    32:         # by gamma + weight-norm normalization in _diff_in_weights, not proxy lr.
    33:         self.proxy_optim = torch.optim.SGD(self.proxy.parameters(), lr=0.1)
    34: 
    35:     def _diff_in_weights(self):
    36:         """Return OrderedDict {name: (old.norm()/diff.norm())*diff} for multi-dim weight params."""
    37:         from collections import OrderedDict
    38:         diff = OrderedDict()
    39:         model_sd = self.model.state_dict()
    40:         proxy_sd = self.proxy.state_dict()
    41:         for (old_k, old_w), (new_k, new_w) in zip(model_sd.items(), proxy_sd.items()):
    42:             if old_w.dim() <= 1:
    43:                 continue
    44:             if 'weight' in old_k:
    45:                 diff_w = new_w - old_w
    46:                 diff[old_k] = old_w.norm() / (diff_w.norm() + self._EPS_AWP) * diff_w
    47:         return diff
    48: 
    49:     def _add_into_weights(self, diff, coeff):
    50:         with torch.no_grad():
    51:             names = diff.keys()
    52:             for name, param in self.model.named_parameters():
    53:                 if name in names:
    54:                     param.add_(coeff * diff[name])
    55: 
    56:     def _calc_awp(self, adv_images, clean_images, labels):
    57:         """Optimize proxy to INCREASE TRADES loss (=> gradient ASCENT via negated loss)."""
    58:         self.proxy.load_state_dict(self.model.state_dict())
    59:         self.proxy.train()
    60:         loss_natural = F.cross_entropy(self.proxy(clean_images), labels)
    61:         loss_robust = F.kl_div(
    62:             F.log_softmax(self.proxy(adv_images), dim=1),
    63:             F.softmax(self.proxy(clean_images), dim=1),
    64:             reduction='batchmean',
    65:         )
    66:         loss = -1.0 * (loss_natural + self.beta * loss_robust)
    67:         self.proxy_optim.zero_grad()
    68:         loss.backward()
    69:         self.proxy_optim.step()
    70:         return self._diff_in_weights()
    71: 
    72:     def train_step(self, images, labels, optimizer):
    73:         # Step 1: generate adversarial examples (TRADES-style, maximize KL)
    74:         self.model.eval()
    75:         adv_images = images.detach() + 0.001 * torch.randn_like(images)
    76:         adv_images = torch.clamp(adv_images, 0.0, 1.0)
    77:         for _ in range(self.attack_steps):
    78:             adv_images.requires_grad_(True)
    79:             loss_kl = F.kl_div(
    80:                 F.log_softmax(self.model(adv_images), dim=1),
    81:                 F.softmax(self.model(images), dim=1),
    82:                 reduction='sum',
    83:             )
    84:             grad = torch.autograd.grad(loss_kl, adv_images)[0]
    85:             adv_images = adv_images.detach() + self.alpha * grad.sign().detach()
    86:             delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
    87:             adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()
    88: 
    89:         self.model.train()
    90: 
    91:         # Step 2: AWP — compute weight perturbation via proxy, then apply to model
    92:         diff = self._calc_awp(adv_images, images, labels)
    93:         self._add_into_weights(diff, coeff=1.0 * self.gamma)
    94: 
    95:         # Step 3: TRADES loss under perturbed weights
    96:         optimizer.zero_grad()
    97:         logits_adv = self.model(adv_images)
    98:         loss_robust = F.kl_div(
    99:             F.log_softmax(logits_adv, dim=1),
   100:             F.softmax(self.model(images), dim=1),
   101:             reduction='batchmean',
   102:         )
   103:         logits_clean = self.model(images)
   104:         loss_clean = F.cross_entropy(logits_clean, labels)
   105:         loss = loss_clean + self.beta * loss_robust
   106: 
   107:         # Step 4: backward, step optimizer, then RESTORE original weights
   108:         optimizer.zero_grad()
   109:         loss.backward()
   110:         optimizer.step()
   111:         self._add_into_weights(diff, coeff=-1.0 * self.gamma)
   112: 
   113:         return {'loss': loss.item()}
   114: 
   115: # ═══════════════════════════════════════════════════════════════════
   116: # END EDITABLE
   117: # ═══════════════════════════════════════════════════════════════════
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
